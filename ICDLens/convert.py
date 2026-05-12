#!/usr/bin/env python3
# Convert results to format readable by ICDLens.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    # Try regular JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping invalid JSON on line {line_no}: {e}")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_content(entry: Dict[str, Any], content_mode: str) -> str:
    diagnosis = entry.get("diagnosis")
    reasoning = entry.get("reasoning")
    raw_output = entry.get("raw_output")

    diagnosis = "" if diagnosis is None else str(diagnosis).strip()
    reasoning = "" if reasoning is None else str(reasoning).strip()
    raw_output = "" if raw_output is None else str(raw_output).strip()

    if content_mode == "reasoning_only":
        return reasoning
    if content_mode == "diagnosis_only":
        return diagnosis
    if content_mode == "raw_output":
        return raw_output

    # default: diagnosis + reasoning
    parts = []
    if diagnosis:
        parts.append(f"Diagnosis: {diagnosis}")
    if reasoning:
        parts.append(f"Reasoning: {reasoning}")
    return "\n".join(parts).strip()


def convert_entry(
    entry: Dict[str, Any],
    author_field: str,
    content_mode: str,
    image_index: int,
) -> Dict[str, Any]:
    encounter_id = entry.get("encounter_id")
    model_id = entry.get(author_field) or entry.get("model_id") or "unknown_model"
    image_ids = entry.get("image_ids") or []

    response = {
        "author_id": model_id,
        "image_index": image_index,
        "content_en": build_content(entry, content_mode),
        "diagnosis": entry.get("diagnosis"),
        "confidence": entry.get("confidence"),
    }

    return {
        "encounter_id": encounter_id,
        "author_id": model_id,
        "image_ids": image_ids,
        "responses": [response],
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Convert flat prediction JSON/JSONL into nested responses JSONL."
    )
    ap.add_argument("--input", required=True, type=Path, help="Input JSON or JSONL file")
    ap.add_argument("--output", required=True, type=Path, help="Output JSONL file")
    ap.add_argument(
        "--author-field",
        default="model_id",
        help="Field to use as author_id (default: model_id)",
    )
    ap.add_argument(
        "--content-mode",
        choices=["diagnosis_reasoning", "reasoning_only", "diagnosis_only", "raw_output"],
        default="diagnosis_reasoning",
        help="How to populate responses[].content_en",
    )
    ap.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Value to assign to responses[].image_index (default: 0)",
    )
    args = ap.parse_args()

    rows = read_json_or_jsonl(args.input)
    converted = [
        convert_entry(
            row,
            author_field=args.author_field,
            content_mode=args.content_mode,
            image_index=args.image_index,
        )
        for row in rows
    ]

    write_jsonl(args.output, converted)
    print(f"[DONE] Wrote {len(converted)} rows to {args.output}")


if __name__ == "__main__":
    main()


