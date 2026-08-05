#!/usr/bin/env python3
"""Collect OWL-ViT and SAM3.1 top-1 detections via the remote service ports.

Produces owlvit_detections_top1.jsonl and sam31_detections.jsonl in the exact
format that run_geometric_eval.py's run-union step consumes, so the rest
of the pipeline (union -> depth -> combine) can be reused unchanged.
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def owl_top1_for_row(row: dict[str, Any], owl_url: str, timeout: float, box_threshold: float, max_detections: int) -> dict[str, Any]:
    labels = row.get("labels") or []
    top1: dict[str, Any] = {label: None for label in labels}
    status, error, size = "ok", None, None
    start = time.time()
    try:
        resp = requests.post(
            owl_url.rstrip("/") + "/objects/depth",
            json={
                "image_id": f"{row['id']}::{row['prompt_kind']}",
                "image_base64": image_b64(Path(str(row.get("image_path")))),
                "queries": labels,
                "process_res": 504,
                "box_threshold": box_threshold,
                "max_detections": max_detections,
                "include_depth_map": False,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        size = [data.get("original_width"), data.get("original_height")]
        for obj in data.get("objects") or []:
            label = obj.get("query")
            if label not in top1:
                continue
            dets = obj.get("detections") or []
            if not dets:
                continue
            best = max(dets, key=lambda d: float(d.get("score", 0.0))) if any("score" in d for d in dets) else dets[0]
            box = best.get("box")
            if box:
                top1[label] = {"label": label, "score": float(best.get("score", 1.0)), "box": [float(v) for v in box]}
    except Exception as exc:  # noqa: BLE001
        status, error = "error", repr(exc)
    return {
        "id": row.get("id"),
        "sample_index": row.get("sample_index"),
        "prompt_kind": row.get("prompt_kind"),
        "image_path": str(row.get("image_path")),
        "image_size": size,
        "labels": labels,
        "top1": top1,
        "status": status,
        "error": error,
        "elapsed_sec": time.time() - start,
    }


def sam_top1_for_row(row: dict[str, Any], owl_row: dict[str, Any], sam_url: str, timeout: float) -> dict[str, Any]:
    labels = row.get("labels") or []
    owl_top1 = owl_row.get("top1") or {}
    missing = [label for label in labels if not owl_top1.get(label)]
    top1: dict[str, Any] = {label: None for label in labels}
    status, error = "ok", None
    start = time.time()
    size = owl_row.get("image_size") or [None, None]
    width = size[0] if size and size[0] else None
    height = size[1] if size and len(size) > 1 and size[1] else None
    try:
        for label in missing:
            resp = requests.post(
                sam_url.rstrip("/") + "/segment/base64",
                json={
                    "image_id": f"{row['id']}::{row['prompt_kind']}",
                    "image_base64": image_b64(Path(str(row.get("image_path")))),
                    "text": label,
                    "include_mask_rle": False,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            objs = data.get("objects") or []
            if not objs:
                continue
            # infer pixel size from mask_shape [h, w] if service size missing
            ms = objs[0].get("mask_shape")
            w = width or (ms[1] if ms and len(ms) > 1 else None)
            h = height or (ms[0] if ms else None)
            best = max(objs, key=lambda o: float(o.get("score", 0.0)))
            box_xywh = best.get("box_xywh")
            if not box_xywh or not w or not h:
                continue
            x, y, bw, bh = [float(v) for v in box_xywh[:4]]
            if max(abs(x), abs(y), abs(bw), abs(bh)) <= 1.5:
                x, bw = x * w, bw * w
                y, bh = y * h, bh * h
            top1[label] = {"label": label, "score": float(best.get("score", 1.0)), "box": [x, y, x + bw, y + bh]}
    except Exception as exc:  # noqa: BLE001
        status, error = "error", repr(exc)
    return {
        "id": row.get("id"),
        "sample_index": row.get("sample_index"),
        "prompt_kind": row.get("prompt_kind"),
        "image_path": str(row.get("image_path")),
        "labels": labels,
        "top1": top1,
        "status": status,
        "error": error,
        "elapsed_sec": time.time() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--owl-url", required=True)
    parser.add_argument("--sam-url", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--box-threshold", type=float, default=0.0)
    parser.add_argument("--max-detections", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    owl_out: list[dict[str, Any]] = [None] * len(rows)  # type: ignore
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(owl_top1_for_row, row, args.owl_url, args.timeout, args.box_threshold, args.max_detections): i for i, row in enumerate(rows)}
        done = 0
        for fut in as_completed(futures):
            owl_out[futures[fut]] = fut.result()
            done += 1
            if done % args.log_every == 0:
                print(f"[owl-port] {done}/{len(rows)}", flush=True)
    write_jsonl(args.out_dir / "owlvit_detections_top1.jsonl", owl_out)
    owl_errors = sum(1 for r in owl_out if r.get("status") != "ok")
    print(f"[owl-port] done errors={owl_errors}", flush=True)

    owl_map = {(r["id"], r["prompt_kind"]): r for r in owl_out}
    sam_out: list[dict[str, Any]] = [None] * len(rows)  # type: ignore
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(sam_top1_for_row, row, owl_map[(row["id"], row["prompt_kind"])], args.sam_url, args.timeout): i for i, row in enumerate(rows)}
        done = 0
        for fut in as_completed(futures):
            sam_out[futures[fut]] = fut.result()
            done += 1
            if done % args.log_every == 0:
                print(f"[sam-port] {done}/{len(rows)}", flush=True)
    write_jsonl(args.out_dir / "sam31_detections.jsonl", sam_out)
    sam_errors = sum(1 for r in sam_out if r.get("status") != "ok")
    print(f"[sam-port] done errors={sam_errors}", flush=True)


if __name__ == "__main__":
    main()
