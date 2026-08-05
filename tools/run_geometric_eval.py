#!/usr/bin/env python3
"""Run OWL-ViT + SAM3.1 + Depth Anything geometric eval for FoR-T2I images."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROMPT_KINDS = ("FoR", "Cam")
DEPTH_TAGS = {"foreground", "background"}
POSITION_TAGS = {"image_left", "image_right", "image_top", "image_bottom", "foreground", "background"}

try:
    from tools.for_location_mapping import MAPPING as FOR_TO_IMAGE_MAPPING
except ImportError:  # pragma: no cover - supports running this file directly.
    from for_location_mapping import MAPPING as FOR_TO_IMAGE_MAPPING


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_prompt_kind(prompt_kind: str) -> str:
    return str(prompt_kind).strip()


def load_results_map(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result_map: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("id", ""))
        prompt_kind = normalize_prompt_kind(str(row.get("prompt_kind", "")))
        if sample_id and prompt_kind in PROMPT_KINDS:
            result_map[(sample_id, prompt_kind)] = row
    return result_map


def dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def objects_by_id(sample: dict[str, Any]) -> dict[str, dict[str, Any]]:
    sig = sample.get("layout_signature") or sample.get("layout_result") or sample.get("layout") or {}
    return {str(obj.get("id")): obj for obj in sig.get("objects", []) if obj.get("id")}


def expected_tags_from_relation(rel: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for tag in rel.get("image_level_primary") or []:
        tag = str(tag)
        if tag in POSITION_TAGS:
            tags.append(tag)
    tag = str(rel.get("expected_image_position") or "")
    if tag in POSITION_TAGS and tag not in tags:
        tags.append(tag)
    return tags


def manifest_rows(samples_path: Path, results_path: Path, prompt_kinds: tuple[str, ...] = PROMPT_KINDS) -> list[dict[str, Any]]:
    samples = read_jsonl(samples_path)
    results = load_results_map(results_path)
    rows: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample["id"])
        obj_map = objects_by_id(sample)
        labels = dedupe([str(obj.get("label", "")) for obj in obj_map.values()])
        relations: list[dict[str, Any]] = []
        sig = sample.get("layout_signature") or {}
        for rel in sig.get("relations", []):
            anchor_id = str(rel.get("anchor", ""))
            target_id = str(rel.get("target", ""))
            anchor = obj_map.get(anchor_id, {})
            target = obj_map.get(target_id, {})
            expected = expected_tags_from_relation(rel)
            if anchor.get("label") and target.get("label") and expected:
                relations.append({
                    "anchor_id": anchor_id,
                    "target_id": target_id,
                    "anchor": str(anchor.get("label")),
                    "target": str(target.get("label")),
                    "expected": expected,
                    "expected_image_position": expected[0] if len(expected) == 1 else expected,
                    "relation_frame": "anchor_frame",
                    "expected_anchor_relation": rel.get("anchor_primary_relation"),
                    "anchor_direction": anchor.get("direction"),
                    "relation_source": "layout_signature.relations",
                })
        construction = sample.get("construction") or {}
        image_level_target_id = str(construction.get("image_level_target_id") or "")
        image_level_position = str(construction.get("image_level_position") or "")
        if image_level_target_id and image_level_position in POSITION_TAGS:
            anchor = obj_map.get("anchor", {})
            target = obj_map.get(image_level_target_id, {})
            if anchor.get("label") and target.get("label"):
                relation = {
                    "anchor_id": "anchor",
                    "target_id": image_level_target_id,
                    "anchor": str(anchor.get("label")),
                    "target": str(target.get("label")),
                    "expected": [image_level_position],
                    "expected_image_position": image_level_position,
                    "relation_frame": "image_frame",
                    "expected_anchor_relation": None,
                    "anchor_direction": anchor.get("direction"),
                    "relation_source": "construction.image_level_position",
                }
                if relation not in relations:
                    relations.append(relation)
        for kind in prompt_kinds:
            result = results.get((sample_id, kind), {})
            image_path = str(result.get("image_path") or "")
            rows.append({
                "id": sample_id,
                "sample_index": sample.get("sample_index"),
                "prompt_kind": kind,
                "image_path": image_path,
                "image_exists": bool(image_path and Path(image_path).exists()),
                "labels": labels,
                "relations": relations,
                "level": sample.get("level"),
                "camera_view": sample.get("camera_view"),
                "layout_family": sample.get("layout_family"),
            })
    return rows


def build_manifest(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_kinds = tuple(normalize_prompt_kind(item) for item in args.prompt_kinds.split(",") if item.strip())
    invalid = set(prompt_kinds) - set(PROMPT_KINDS)
    if invalid:
        raise ValueError(f"invalid prompt kinds: {sorted(invalid)}")
    rows = manifest_rows(args.sample, args.results, prompt_kinds)
    write_jsonl(args.out_dir / "manifest.jsonl", rows)
    report = {
        "images": len(rows),
        "existing_images": sum(1 for row in rows if row.get("image_exists")),
        "samples": len({row["id"] for row in rows}),
        "levels": dict(Counter(row.get("level") for row in rows)),
        "camera_views": dict(Counter(row.get("camera_view") for row in rows)),
        "layout_families": dict(Counter(row.get("layout_family") for row in rows)),
        "relation_counts": dict(Counter(len(row.get("relations") or []) for row in rows)),
    }
    (args.out_dir / "manifest_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def run_owl(args: argparse.Namespace) -> None:
    from PIL import Image
    import torch
    from transformers import OwlViTForObjectDetection, OwlViTProcessor

    rows = read_jsonl(args.manifest)
    processor = OwlViTProcessor.from_pretrained(args.owl_model)
    model = OwlViTForObjectDetection.from_pretrained(args.owl_model).to(args.device).eval()
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        labels = row.get("labels") or []
        top1 = {label: None for label in labels}
        status = "ok"
        error = None
        image_path = Path(str(row.get("image_path") or ""))
        start = time.time()
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.inference_mode():
                inputs = processor(text=[labels], images=image, return_tensors="pt")
                inputs = {key: value.to(args.device) for key, value in inputs.items()}
                outputs = model(**inputs)
                target_sizes = torch.tensor([[image.height, image.width]], device=args.device)
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=args.owl_threshold,
                    target_sizes=target_sizes,
                    text_labels=[labels],
                )[0]
            scores = results["scores"].detach().cpu().tolist()
            boxes = results["boxes"].detach().cpu().tolist()
            text_labels = results.get("text_labels") or []
            for score, label, box in zip(scores, text_labels, boxes):
                if label not in top1:
                    continue
                item = {"label": label, "score": float(score), "box": [float(v) for v in box]}
                current = top1[label]
                if current is None or item["score"] > float(current.get("score", 0.0)):
                    top1[label] = item
            size = list(image.size)
        except Exception as exc:
            status = "error"
            error = repr(exc)
            size = None
        out.append({
            "id": row.get("id"),
            "sample_index": row.get("sample_index"),
            "prompt_kind": row.get("prompt_kind"),
            "image_path": str(image_path),
            "image_size": size,
            "labels": labels,
            "top1": top1,
            "status": status,
            "error": error,
            "elapsed_sec": time.time() - start,
        })
        if index % args.log_every == 0:
            print(f"[owl] {index}/{len(rows)}", flush=True)
    write_jsonl(args.out_dir / "owlvit_detections_top1.jsonl", out)


def to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_python(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(item) for item in value]
    return value


def flatten_numbers(value: Any) -> list[float]:
    value = to_python(value)
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    out: list[float] = []
    if isinstance(value, list):
        for item in value:
            out.extend(flatten_numbers(item))
    return out


def normalize_xywh_box(box: list[float], width: int, height: int) -> list[float]:
    x, y, w, h = [float(v) for v in box[:4]]
    if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
        x *= width
        w *= width
        y *= height
        h *= height
    return [x, y, x + w, y + h]


def parse_sam_top1(outputs: dict[str, Any], width: int, height: int, label: str) -> dict[str, Any] | None:
    data = to_python(outputs)
    boxes = data.get("out_boxes_xywh") or data.get("out_boxes") or []
    if boxes and isinstance(boxes[0], (int, float)):
        boxes = [boxes]
    probs = flatten_numbers(data.get("out_probs") or data.get("scores"))
    if not boxes:
        return None
    usable = min(len(boxes), len(probs)) if probs else len(boxes)
    if usable < 1:
        return None
    best_i = max(range(usable), key=lambda i: float(probs[i])) if probs else 0
    score = float(probs[best_i]) if probs else 1.0
    box = normalize_xywh_box([float(v) for v in boxes[best_i][:4]], width, height)
    return {"label": label, "score": score, "box": box}


def run_sam(args: argparse.Namespace) -> None:
    from PIL import Image
    from sam3 import build_sam3_predictor

    manifest = read_jsonl(args.manifest)
    owl_map = {(row["id"], row["prompt_kind"]): row for row in read_jsonl(args.owl_detections)}
    predictor = build_sam3_predictor(
        version="sam3.1",
        checkpoint_path=str(args.sam_checkpoint),
        compile=False,
        warm_up=False,
        async_loading_frames=False,
        max_num_objects=args.sam_max_objects,
        use_fa3=False,
    )
    out: list[dict[str, Any]] = []
    temp_root = Path(tempfile.mkdtemp(prefix="sfb_sam31_"))
    try:
        for index, row in enumerate(manifest, 1):
            labels = row.get("labels") or []
            owl = owl_map.get((row["id"], row["prompt_kind"]), {})
            owl_top1 = owl.get("top1") or {}
            missing = [label for label in labels if not owl_top1.get(label)]
            top1 = {label: None for label in labels}
            status = "ok"
            error = None
            image_path = Path(str(row.get("image_path") or ""))
            start = time.time()
            try:
                with Image.open(image_path) as image:
                    rgb = image.convert("RGB")
                    width, height = rgb.size
                    frame_dir = temp_root / f"{index:06d}"
                    frame_dir.mkdir(parents=True, exist_ok=True)
                    rgb.save(frame_dir / "000.jpg")
                if missing:
                    resp = predictor.handle_request({"type": "start_session", "resource_path": str(frame_dir)})
                    session_id = resp["session_id"]
                    try:
                        for label in missing:
                            prompt_resp = predictor.handle_request({
                                "type": "add_prompt",
                                "session_id": session_id,
                                "frame_index": 0,
                                "text": label,
                            })
                            outputs = prompt_resp.get("outputs") or prompt_resp
                            top1[label] = parse_sam_top1(outputs, width, height, label)
                    finally:
                        predictor.handle_request({"type": "close_session", "session_id": session_id})
            except Exception as exc:
                status = "error"
                error = repr(exc)
            out.append({
                "id": row.get("id"),
                "sample_index": row.get("sample_index"),
                "prompt_kind": row.get("prompt_kind"),
                "image_path": str(image_path),
                "labels": labels,
                "sam_requested_labels": missing,
                "top1": top1,
                "status": status,
                "error": error,
                "elapsed_sec": time.time() - start,
            })
            if index % args.log_every == 0:
                print(f"[sam] {index}/{len(manifest)}", flush=True)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    write_jsonl(args.out_dir / "sam31_detections.jsonl", out)


def pick_detection(label: str, owl_top1: dict[str, Any], sam_top1: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    if owl_top1.get(label):
        return "owl", owl_top1[label]
    if sam_top1.get(label):
        return "sam", sam_top1[label]
    return None, None


def union_rows(manifest_path: Path, owl_path: Path, sam_path: Path) -> list[dict[str, Any]]:
    manifest = read_jsonl(manifest_path)
    owl_map = {(row["id"], row["prompt_kind"]): row for row in read_jsonl(owl_path)}
    sam_map = {(row["id"], row["prompt_kind"]): row for row in read_jsonl(sam_path)}
    rows: list[dict[str, Any]] = []
    for row in manifest:
        key = (row["id"], row["prompt_kind"])
        owl = owl_map.get(key, {})
        sam = sam_map.get(key, {})
        selected: dict[str, Any] = {}
        sources: dict[str, Any] = {}
        scores: dict[str, Any] = {}
        missing: list[str] = []
        for label in row.get("labels") or []:
            source, det = pick_detection(label, owl.get("top1") or {}, sam.get("top1") or {})
            selected[label] = det
            sources[label] = source
            scores[label] = det.get("score") if det else None
            if det is None:
                missing.append(label)
        rows.append({**row, "selected": selected, "sources": sources, "scores": scores, "missing_detection_labels": missing, "all_detected": not missing})
    return rows


def run_union(args: argparse.Namespace) -> None:
    rows = union_rows(args.manifest, args.owl_detections, args.sam_detections)
    write_jsonl(args.out_dir / "union_detections.jsonl", rows)


def image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def run_depth(args: argparse.Namespace) -> None:
    import requests

    rows = read_jsonl(args.union_detections)
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        selected = row.get("selected") or {}
        detections = []
        for label, det in selected.items():
            if det:
                detections.append({"label": label, "score": float(det.get("score", 1.0)), "box": det.get("box")})
        status = "ok"
        error = None
        object_depths: dict[str, Any] = {}
        response_meta: dict[str, Any] = {}
        start = time.time()
        try:
            if detections:
                resp = requests.post(
                    args.depth_api_url.rstrip("/") + "/objects/depth",
                    json={
                        "image_id": f"{row['id']}::{row['prompt_kind']}",
                        "image_base64": image_base64(Path(str(row.get("image_path")))),
                        "queries": [],
                        "detections": detections,
                        "include_depth_map": False,
                        "depth_format": "npy_base64",
                        "process_res": args.process_res,
                    },
                    timeout=args.depth_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                response_meta = {
                    "is_metric": data.get("is_metric"),
                    "scale_factor": data.get("scale_factor"),
                    "original_width": data.get("original_width"),
                    "original_height": data.get("original_height"),
                    "processing_time": data.get("processing_time"),
                }
                for obj in data.get("objects") or []:
                    label = obj.get("query")
                    dets = obj.get("detections") or []
                    if label and dets:
                        object_depths[label] = dets[0]
        except Exception as exc:
            status = "error"
            error = repr(exc)
        out.append({
            "id": row.get("id"),
            "sample_index": row.get("sample_index"),
            "prompt_kind": row.get("prompt_kind"),
            "image_path": row.get("image_path"),
            "status": status,
            "error": error,
            "object_depths": object_depths,
            "elapsed_sec": time.time() - start,
            **response_meta,
        })
        if index % args.log_every == 0:
            print(f"[depth] {index}/{len(rows)}", flush=True)
    write_jsonl(args.out_dir / "da3_depth_for_union_detections.jsonl", out)


def center(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def infer_image_relation(anchor_det: dict[str, Any], target_det: dict[str, Any], anchor_depth: dict[str, Any] | None, target_depth: dict[str, Any] | None, expected: list[str]) -> tuple[list[str], bool]:
    ax, ay = center(anchor_det["box"])
    tx, ty = center(target_det["box"])
    predicted = ["image_left" if tx < ax else "image_right", "image_top" if ty < ay else "image_bottom"]
    needs_depth = any(tag in DEPTH_TAGS for tag in expected)
    if needs_depth:
        ad = None if not anchor_depth else anchor_depth.get("depth_median")
        td = None if not target_depth else target_depth.get("depth_median")
        if ad is None or td is None:
            return predicted, True
        predicted.append("foreground" if float(td) < float(ad) else "background")
    return predicted, False


def image_tags_to_anchor_relations(camera_view: str | None, anchor_direction: str | None, image_tags: list[str]) -> list[str]:
    if not camera_view or not anchor_direction:
        return []
    mapping = FOR_TO_IMAGE_MAPPING.get(str(camera_view), {}).get(str(anchor_direction), {})
    out: list[str] = []
    for anchor_relation, image_position in mapping.items():
        if image_position in image_tags and image_position != "Unknown" and anchor_relation not in out:
            out.append(anchor_relation)
    return out


def run_combine(args: argparse.Namespace) -> None:
    union = read_jsonl(args.union_detections)
    depth_map = {(row["id"], row["prompt_kind"]): row for row in read_jsonl(args.depth_results)}
    image_rows: list[dict[str, Any]] = []
    for row in union:
        depth = depth_map.get((row["id"], row["prompt_kind"]), {})
        object_depths = depth.get("object_depths") or {}
        selected = row.get("selected") or {}
        sources = row.get("sources") or {}
        relation_rows: list[dict[str, Any]] = []
        missing_depth_relations = 0
        for rel in row.get("relations") or []:
            anchor = rel["anchor"]
            target = rel["target"]
            anchor_det = selected.get(anchor)
            target_det = selected.get(target)
            expected = rel.get("expected") or []
            relation_frame = rel.get("relation_frame") or ("anchor_frame" if rel.get("expected_anchor_relation") else "image_frame")
            expected_anchor_relation = rel.get("expected_anchor_relation")
            predicted: list[str] = []
            predicted_anchor_relations: list[str] = []
            ok = False
            relation_missing_depth = False
            if anchor_det and target_det:
                predicted, relation_missing_depth = infer_image_relation(anchor_det, target_det, object_depths.get(anchor), object_depths.get(target), expected)
                predicted_anchor_relations = image_tags_to_anchor_relations(row.get("camera_view"), rel.get("anchor_direction"), predicted)
                if relation_frame == "anchor_frame" and expected_anchor_relation:
                    ok = (not relation_missing_depth) and expected_anchor_relation in predicted_anchor_relations
                else:
                    ok = (not relation_missing_depth) and all(tag in predicted for tag in expected)
            if relation_missing_depth:
                missing_depth_relations += 1
            relation_rows.append({
                **rel,
                "relation_frame": relation_frame,
                "predicted": predicted,
                "predicted_image_positions": predicted,
                "predicted_anchor_relations": predicted_anchor_relations,
                "ok": ok,
                "missing_depth": relation_missing_depth,
                "anchor_source": sources.get(anchor),
                "target_source": sources.get(target),
                "anchor_depth_median": (object_depths.get(anchor) or {}).get("depth_median"),
                "target_depth_median": (object_depths.get(target) or {}).get("depth_median"),
            })
        auto_label = "correct" if row.get("all_detected") and relation_rows and all(rel.get("ok") for rel in relation_rows) else "wrong"
        score_num = 1 if auto_label == "correct" else 0
        checklist = []
        if row.get("missing_detection_labels"):
            checklist.append("missing detections: " + ", ".join(row["missing_detection_labels"]))
        for rel in relation_rows:
            if rel.get("relation_frame") == "anchor_frame":
                checklist.append(
                    f"{rel['target']} vs {rel['anchor']}: frame=anchor_frame "
                    f"expected_anchor={rel.get('expected_anchor_relation')} anchor_direction={rel.get('anchor_direction')} "
                    f"expected_image={rel.get('expected')} predicted_anchor={rel.get('predicted_anchor_relations')} "
                    f"predicted_image={rel.get('predicted_image_positions')} ok={rel['ok']}"
                )
            else:
                checklist.append(f"{rel['target']} vs {rel['anchor']}: frame=image_frame expected={rel['expected']} predicted={rel['predicted']} ok={rel['ok']}")
        image_rows.append({
            "id": row.get("id"),
            "sample_index": row.get("sample_index"),
            "prompt_kind": row.get("prompt_kind"),
            "eval_status": "ok",
            "eval_mode": "owl_sam_da3_geometric",
            "eval_model": "OWL-ViT google/owlvit-base-patch32 + SAM3.1 fallback + Depth Anything 3",
            "auto_label": auto_label,
            "label": "pass" if score_num == 1 else "fail",
            "score_num": score_num,
            "spatial_checklist": "\n".join(checklist),
            "verdict": "geometric layout pass" if score_num == 1 else "geometric layout fail",
            "level": row.get("level"),
            "camera_view": row.get("camera_view"),
            "layout_family": row.get("layout_family"),
            "needs_depth": any(any(tag in DEPTH_TAGS for tag in rel.get("expected", [])) for rel in row.get("relations") or []),
            "labels": row.get("labels"),
            "sources": sources,
            "scores": row.get("scores"),
            "missing_detection_labels": row.get("missing_detection_labels"),
            "missing_depth_relations": missing_depth_relations,
            "all_detected": row.get("all_detected"),
            "relations": relation_rows,
        })
    write_jsonl(args.out_dir / "geometric_eval_results.jsonl", image_rows)

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in image_rows:
        by_prompt[row["id"]].append(row)
    prompt_complete = [rows for rows in by_prompt.values() if len(rows) == 2 and all(r.get("all_detected") for r in rows)]
    report = {
        "images_total": len(image_rows),
        "images_all_detected": sum(1 for row in image_rows if row.get("all_detected")),
        "samples_total": len(by_prompt),
        "prompt_pairs_all_detected": len(prompt_complete),
        "image_pass": sum(1 for row in image_rows if row.get("score_num") == 1),
        "image_fail": sum(1 for row in image_rows if row.get("score_num") == 0),
        "by_level": dict(Counter(row.get("level") for row in image_rows)),
        "by_camera_view": dict(Counter(row.get("camera_view") for row in image_rows)),
        "by_layout_family": dict(Counter(row.get("layout_family") for row in image_rows)),
        "source_usage": dict(Counter(source for row in image_rows for source in (row.get("sources") or {}).values() if source)),
        "images_with_sam_fallback": sum(1 for row in image_rows if "sam" in set((row.get("sources") or {}).values())),
    }
    (args.out_dir / "geometric_eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = [
        "# FoR-T2I Geometric Eval",
        "",
        "## Method",
        "- OWL-ViT primary detection: `google/owlvit-base-patch32`.",
        "- SAM3.1 fallback only for labels missed by OWL-ViT.",
        "- Depth Anything 3 samples depth on selected union boxes.",
        "- Image-frame positions use bbox center ordering; foreground/background uses smaller DA3 depth median.",
        "- FoR relations are evaluated in anchor frame by converting predicted image positions through the intended anchor direction.",
        "- Anchor orientation correctness itself is judged by the separate VLM orientation eval.",
        "",
        "## Results",
        f"- Image-level all detected: `{report['images_all_detected']}/{report['images_total']}`.",
        f"- Prompt-pair all detected: `{report['prompt_pairs_all_detected']}/{report['samples_total']}`.",
        f"- Image-level pass: `{report['image_pass']}/{report['images_total']}`.",
        f"- Images using SAM fallback: `{report['images_with_sam_fallback']}`.",
        f"- Source usage: `{json.dumps(report['source_usage'], ensure_ascii=False)}`.",
        "",
        "## Artifacts",
        f"- `{args.out_dir / 'geometric_eval_results.jsonl'}`",
        f"- `{args.out_dir / 'geometric_eval_report.json'}`",
        f"- `{args.out_dir / 'owlvit_detections_top1.jsonl'}`",
        f"- `{args.out_dir / 'sam31_detections.jsonl'}`",
        f"- `{args.out_dir / 'da3_depth_for_union_detections.jsonl'}`",
    ]
    (args.out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def run_all(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    manifest = args.out_dir / "manifest.jsonl"
    owl = args.out_dir / "owlvit_detections_top1.jsonl"
    sam = args.out_dir / "sam31_detections.jsonl"
    union = args.out_dir / "union_detections.jsonl"
    depth = args.out_dir / "da3_depth_for_union_detections.jsonl"

    def call(cmd: list[str], env: dict[str, str] | None = None) -> None:
        print("[cmd] " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)

    call([sys.executable, str(script), "build-manifest", "--sample", str(args.sample), "--results", str(args.results), "--out-dir", str(args.out_dir)])
    da3_env = os.environ.copy()
    da3_env["PYTHONPATH"] = str(args.da3_root / "src") + os.pathsep + da3_env.get("PYTHONPATH", "")
    call([str(args.da3_python), str(script), "run-owl", "--manifest", str(manifest), "--out-dir", str(args.out_dir), "--da3-root", str(args.da3_root), "--device", args.device, "--owl-model", args.owl_model], env=da3_env)
    sam_env = os.environ.copy()
    sam_env["PYTHONPATH"] = str(args.sam_code) + os.pathsep + sam_env.get("PYTHONPATH", "")
    sam_env["HF_HOME"] = str(args.sam_home)
    call([str(args.sam_python), str(script), "run-sam", "--manifest", str(manifest), "--owl-detections", str(owl), "--out-dir", str(args.out_dir), "--sam-checkpoint", str(args.sam_checkpoint)], env=sam_env)
    call([sys.executable, str(script), "run-union", "--manifest", str(manifest), "--owl-detections", str(owl), "--sam-detections", str(sam), "--out-dir", str(args.out_dir)])
    call([sys.executable, str(script), "run-depth", "--union-detections", str(union), "--out-dir", str(args.out_dir), "--depth-api-url", args.depth_api_url])
    call([sys.executable, str(script), "combine", "--union-detections", str(union), "--depth-results", str(depth), "--out-dir", str(args.out_dir)])


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-manifest")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--prompt-kinds", default="FoR,Cam")
    p.set_defaults(func=build_manifest)

    p = sub.add_parser("run-owl")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--owl-model", default="google/owlvit-base-patch32")
    p.add_argument("--owl-threshold", type=float, default=0.0)
    p.add_argument("--owl-max-detections", type=int, default=50)
    p.add_argument("--log-every", type=int, default=10)
    p.set_defaults(func=run_owl)

    p = sub.add_parser("run-sam")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--owl-detections", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sam-checkpoint", type=Path, required=True)
    p.add_argument("--sam-max-objects", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10)
    p.set_defaults(func=run_sam)

    p = sub.add_parser("run-union")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--owl-detections", type=Path, required=True)
    p.add_argument("--sam-detections", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=run_union)

    p = sub.add_parser("run-depth")
    p.add_argument("--union-detections", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--depth-api-url", required=True)
    p.add_argument("--depth-timeout", type=float, default=180.0)
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--log-every", type=int, default=10)
    p.set_defaults(func=run_depth)

    p = sub.add_parser("combine")
    p.add_argument("--union-detections", type=Path, required=True)
    p.add_argument("--depth-results", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=run_combine)

    p = sub.add_parser("all")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--da3-root", type=Path, required=True)
    p.add_argument("--da3-python", type=Path, required=True)
    p.add_argument("--sam-python", type=Path, required=True)
    p.add_argument("--sam-code", type=Path, required=True)
    p.add_argument("--sam-home", type=Path, required=True)
    p.add_argument("--sam-checkpoint", type=Path, required=True)
    p.add_argument("--depth-api-url", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--owl-model", default="google/owlvit-base-patch32")
    p.set_defaults(func=run_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
