#!/usr/bin/env python3
"""VLM orientation eval with per-object bbox crops, sent serially (one-by-one).

For each anchor object we crop the image to its detected bbox (from
union_detections.jsonl) with padding and send only that crop to the VLM. If the
object was not detected, we fall back to the full image. Requests are issued
strictly one at a time to avoid overloading the shared VLM service.
"""
from __future__ import annotations
import argparse, base64, io, json, re, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from tools.run_geometric_eval import load_results_map, normalize_prompt_kind, read_jsonl, write_jsonl

PROMPT_KINDS = ("FoR", "Cam")
ORIENTATION_SYSTEM = """You are a strict visual verifier for object orientation.

You will receive one image and one yes/no question about the facing direction of
one visible anchor object.

Judge only the object's actual visible orientation using geometry in the image.
Ignore style, realism, background, labels, and any written text.

Output exactly one word:
YES
or
NO

Do not output explanations."""

DIRECTION_QUESTION = {
    "facing_image_left": "Is the {label} facing image-left?",
    "facing_image_right": "Is the {label} facing image-right?",
    "facing_image_top": "Is the {label} facing image-top?",
    "facing_image_bottom": "Is the {label} facing image-bottom?",
    "facing_viewer": "Is the {label} facing the viewer?",
    "facing_away_from_viewer": "Is the {label} facing away from the viewer?",
}


def resolve_model(row: dict[str, Any]) -> str:
    model = str(row.get("model") or row.get("eval_model") or row.get("gen_model") or "")
    return model or "unknown"


def chat(url: str, model: str, messages: list[dict[str, Any]], api_key: str | None = None, max_tokens: int = 512, retries: int = 3) -> str:
    import requests

    headers = {}
    if api_key:
        headers["Authorization"] = api_key
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                headers=headers,
                timeout=180,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(str(last_error) if last_error else "chat request failed")


def parse_yes_no(text: str) -> tuple[int | None, str | None]:
    tail = (text or "").strip()[-300:]
    yes = bool(re.search(r"\bYES\b", tail, re.IGNORECASE))
    no = bool(re.search(r"\bNO\b", tail, re.IGNORECASE))
    if yes and not no:
        return 1, "YES"
    if no and not yes:
        return 0, "NO"
    if yes and no:
        if re.search(r"YES\s*$", tail, re.IGNORECASE):
            return 1, "YES"
        if re.search(r"NO\s*$", tail, re.IGNORECASE):
            return 0, "NO"
    return None, None


def object_orientation_question(label: str, direction: str) -> str:
    if direction not in DIRECTION_QUESTION:
        raise ValueError(f"unsupported orientation direction: {direction}")
    return DIRECTION_QUESTION[direction].format(label=label)


def anchors_for_sample(sample: dict[str, Any]) -> list[dict[str, Any]]:
    layout = sample.get("layout_result") or sample.get("layout") or {}
    anchors = []
    for obj in layout.get("objects") or []:
        if obj.get("is_anchor") and obj.get("direction"):
            anchors.append({
                "object_id": obj.get("id"),
                "label": obj.get("label"),
                "direction": obj.get("direction"),
            })
    return anchors


def merge_rows(geometric_row: dict[str, Any] | None, orientation_row: dict[str, Any] | None) -> dict[str, Any]:
    if geometric_row is None and orientation_row is None:
        return {}

    base = dict(geometric_row or orientation_row or {})
    geo_score = geometric_row.get("score_num") if geometric_row else None
    geo_label = geometric_row.get("label") if geometric_row else None
    geo_status = geometric_row.get("eval_status") if geometric_row else None
    orientation_score = orientation_row.get("orientation_score_num") if orientation_row else None
    orientation_label = orientation_row.get("orientation_label") if orientation_row else None

    final_ok = (geo_score == 1) and (orientation_score == 1)
    base.update({
        "geom_score_num": geo_score,
        "geom_label": geo_label,
        "geom_eval_status": geo_status,
        "geom_eval_mode": geometric_row.get("eval_mode") if geometric_row else None,
        "geom_eval_model": geometric_row.get("eval_model") if geometric_row else None,
        "geom_spatial_checklist": geometric_row.get("spatial_checklist") if geometric_row else None,
        "geom_verdict": geometric_row.get("verdict") if geometric_row else None,
        "geom_missing_detection_labels": geometric_row.get("missing_detection_labels") if geometric_row else None,
        "geom_missing_depth_relations": geometric_row.get("missing_depth_relations") if geometric_row else None,
        "orientation_eval_status": orientation_row.get("orientation_eval_status") if orientation_row else None,
        "orientation_eval_mode": orientation_row.get("orientation_eval_mode") if orientation_row else None,
        "orientation_eval_model": orientation_row.get("orientation_eval_model") if orientation_row else None,
        "orientation_score_num": orientation_score,
        "orientation_label": orientation_label,
        "orientation_verdict": orientation_row.get("orientation_verdict") if orientation_row else None,
        "orientation_questions": orientation_row.get("orientation_questions") if orientation_row else None,
        "orientation_judge_failed_default_no": orientation_row.get("orientation_judge_failed_default_no") if orientation_row else None,
        "eval_mode": "owl_sam_da3_geometric_plus_vlm_orientation",
        "eval_model": "OWL-ViT google/owlvit-base-patch32 + SAM3.1 fallback + Depth Anything 3 + VLM orientation",
        "eval_status": "ok",
        "score_num": 1 if final_ok else 0,
        "score_den": 1,
        "label": "pass" if final_ok else "fail",
        "auto_label": "correct" if final_ok else "wrong",
        "verdict": "combined geometric + orientation pass" if final_ok else "combined geometric + orientation fail",
    })
    return base


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    geo_total = sum(1 for row in rows if row.get("geom_score_num") is not None)
    geo_pass = sum(1 for row in rows if row.get("geom_score_num") == 1)
    ori_total = sum(1 for row in rows if row.get("orientation_score_num") is not None)
    ori_pass = sum(1 for row in rows if row.get("orientation_score_num") == 1)
    final_pass = sum(1 for row in rows if row.get("score_num") == 1)
    return {
        "rows": len(rows),
        "geometry_scored": geo_total,
        "geometry_pass": geo_pass,
        "orientation_scored": ori_total,
        "orientation_pass": ori_pass,
        "final_pass": final_pass,
        "final_fail": len(rows) - final_pass,
    }


def encode_pil(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def crop_with_padding(img: Image.Image, box: list[float], pad_frac: float = 0.18) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    nx1 = max(0, int(round(x1 - px)))
    ny1 = max(0, int(round(y1 - py)))
    nx2 = min(w, int(round(x2 + px)))
    ny2 = min(h, int(round(y2 + py)))
    if nx2 <= nx1 or ny2 <= ny1:
        return img
    return img.crop((nx1, ny1, nx2, ny2))


def orientation_for_row_crop(row, selected_boxes, vllm_url, model, api_key, votes, request_delay):
    image_path = Path(str(row.get("image_path") or ""))
    anchors = row.get("anchors") or []
    base = {
        "id": row.get("id"), "sample_index": row.get("sample_index"), "prompt_kind": row.get("prompt_kind"),
        "image_path": str(image_path), "orientation_eval_status": "ok",
        "orientation_eval_mode": "vlm_orientation_anchor_crop", "orientation_eval_model": model,
        "orientation_question_count": len(anchors), "orientation_questions": [],
        "orientation_score_num": None, "orientation_score_den": 1, "orientation_label": None,
        "orientation_verdict": None, "orientation_judge_failed_default_no": False,
    }
    if not image_path.exists():
        return {**base, "orientation_eval_status": "error_image", "orientation_verdict": "missing image", "orientation_label": "fail", "orientation_score_num": 0}
    if not anchors:
        return {**base, "orientation_eval_status": "skip_no_anchor", "orientation_verdict": "no anchor orientation found", "orientation_label": "missing", "orientation_score_num": None}
    try:
        full = Image.open(image_path).convert("RGB")
    except Exception as exc:
        return {**base, "orientation_eval_status": "error_image", "orientation_error": str(exc), "orientation_verdict": "image open failed", "orientation_label": "fail", "orientation_score_num": 0}

    question_rows, question_scores = [], []
    judge_failed_default_no = False
    for anchor in anchors:
        label = str(anchor.get("label") or "anchor")
        direction = str(anchor.get("direction") or "")
        if direction not in DIRECTION_QUESTION:
            question_rows.append({"object_id": anchor.get("object_id"), "label": label, "direction": direction, "question": None, "verdicts": [], "votes": [], "score_num": None, "error": f"unsupported direction: {direction}"})
            continue
        det = (selected_boxes or {}).get(label)
        if det and det.get("box"):
            crop_source = "bbox_crop"
            image_input = encode_pil(crop_with_padding(full, det["box"]))
        else:
            crop_source = "full_image_fallback"
            image_input = encode_pil(full)
        question = object_orientation_question(label, direction)
        verdicts, scores = [], []
        for _ in range(votes):
            try:
                if request_delay > 0:
                    time.sleep(request_delay)
                content = chat(vllm_url, model, [
                    {"role": "system", "content": ORIENTATION_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_input}},
                        {"type": "text", "text": question},
                    ]},
                ], api_key=api_key, max_tokens=256)
                verdicts.append(content)
                score, _ = parse_yes_no(content)
                if score is None:
                    judge_failed_default_no = True
                    scores.append(0)
                    verdicts[-1] = f"JUDGE_ERROR_DEFAULT_NO: unparsable verdict: {content}"
                else:
                    scores.append(score)
            except Exception as exc:
                judge_failed_default_no = True
                verdicts.append(f"JUDGE_ERROR_DEFAULT_NO: {exc}")
                scores.append(0)
        yes_count = sum(scores)
        score_num = 1 if yes_count > len(scores) / 2 else 0
        question_rows.append({"object_id": anchor.get("object_id"), "label": label, "direction": direction, "question": question, "image_source": crop_source, "verdicts": verdicts, "votes": scores, "score_num": score_num, "score_den": len(scores) or 1, "score_frac": f"({yes_count}/{len(scores)})"})
        question_scores.append(score_num)

    orientation_pass = bool(question_scores) and all(s == 1 for s in question_scores)
    status = "error_orientation" if any(q.get("score_num") is None for q in question_rows) else "ok"
    return {**base, "orientation_questions": question_rows, "orientation_judge_failed_default_no": judge_failed_default_no,
            "orientation_eval_status": status, "orientation_score_num": 1 if orientation_pass else 0, "orientation_score_den": 1,
            "orientation_label": "pass" if orientation_pass else "fail",
            "orientation_verdict": "anchor orientations pass" if orientation_pass else "anchor orientation fail"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--geometric-results", type=Path, required=True)
    p.add_argument("--union-detections", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--vllm-url", required=True)
    p.add_argument("--model", default="qwen3-vl-235b-a22b-instruct")
    p.add_argument("--api-key")
    p.add_argument("--votes", type=int, default=1)
    p.add_argument("--request-delay", type=float, default=0.4)
    p.add_argument("--prompt-kinds", default="FoR,Cam")
    args = p.parse_args()
    prompt_kinds = tuple(normalize_prompt_kind(item) for item in args.prompt_kinds.split(",") if item.strip())
    invalid = set(prompt_kinds) - set(PROMPT_KINDS)
    if invalid:
        raise ValueError(f"invalid prompt kinds: {sorted(invalid)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    samples = read_jsonl(args.sample)
    results_map = load_results_map(args.results)
    geometric_rows = read_jsonl(args.geometric_results)
    geometric_map = {(str(r.get("id")), str(r.get("prompt_kind"))): r for r in geometric_rows}
    union_map = {(str(r.get("id")), str(r.get("prompt_kind"))): (r.get("selected") or {}) for r in read_jsonl(args.union_detections)}

    orientation_rows = []
    for sample in samples:
        sid = str(sample.get("id"))
        for kind in prompt_kinds:
            result = results_map.get((sid, kind), {})
            image_path = str(result.get("image_path") or "")
            orientation_rows.append({
                "id": sid, "sample_index": sample.get("sample_index"), "prompt_kind": kind,
                "image_path": image_path, "status": str(result.get("status") or ("ok" if image_path else "missing")),
                "model": resolve_model(result), "anchors": anchors_for_sample(sample),
            })

    outputs = []
    total = len(orientation_rows)
    for i, row in enumerate(orientation_rows, 1):
        boxes = union_map.get((str(row["id"]), str(row["prompt_kind"])), {})
        outputs.append(orientation_for_row_crop(row, boxes, args.vllm_url, args.model, args.api_key, args.votes, args.request_delay))
        if i % 10 == 0 or i == total:
            print(f"[orient-crop] {i}/{total}", flush=True)

    outputs.sort(key=lambda r: (str(r.get("id")), str(r.get("prompt_kind"))))
    orientation_path = args.out_dir / "orientation_eval_results.jsonl"
    write_jsonl(orientation_path, outputs)

    orientation_map = {(str(r.get("id")), str(r.get("prompt_kind"))): r for r in outputs}
    combined = []
    for sample in samples:
        sid = str(sample.get("id"))
        for kind in prompt_kinds:
            m = merge_rows(geometric_map.get((sid, kind)), orientation_map.get((sid, kind)))
            if m:
                combined.append(m)
    combined_path = args.out_dir / "combined_eval_results.jsonl"
    write_jsonl(combined_path, combined)

    report = build_report(combined)
    fallback_used = sum(1 for r in outputs for q in (r.get("orientation_questions") or []) if q.get("image_source") == "full_image_fallback")
    crop_used = sum(1 for r in outputs for q in (r.get("orientation_questions") or []) if q.get("image_source") == "bbox_crop")
    report.update({"samples": len(samples), "orientation_rows": len(outputs),
                   "orientation_bbox_crop_questions": crop_used, "orientation_fullimage_fallback_questions": fallback_used,
                   "combined_results": str(combined_path), "orientation_results": str(orientation_path),
                   "geometric_results": str(args.geometric_results)})
    (args.out_dir / "orientation_eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
