import argparse
import json, re
import os
import sys
from typing import Iterable, Dict, Any, List


def _basename(p: str) -> str:
    if not isinstance(p, str):
        return p
    return os.path.basename(p)


def _collect_image_basenames(record: Dict[str, Any]) -> List[str]:
    seen = set()
    out: List[str] = []

    def add(path: str):
        if not path:
            return
        fn = _basename(path)
        if fn not in seen:
            seen.add(fn)
            out.append(fn)

    for p in record.get("image_ids", []) or []:
        add(p)
    for pi in record.get("per_image", []) or []:
        add(pi.get("image_id"))

    return out


def convert_record(record: Dict[str, Any]) -> Dict[str, Any]:
    encounter_id = record.get("encounter_id")
    model_id = record.get("model_id") or record.get("author_id") or "unknown-model"

    image_ids = _collect_image_basenames(record)

    responses: List[Dict[str, Any]] = []

    per_image = record.get("per_image") or []
    if isinstance(per_image, list):
        per_image = sorted(
            per_image,
            key=lambda x: (x.get("image_index") is None, x.get("image_index", 0)),
        )

    if per_image:
        for pi in per_image:
            content = (
                pi.get("reasoning")
                or pi.get("diagnosis")
                or record.get("reasoning")
                or ""
            )
            diagnosis = pi.get("diagnosis", record.get("diagnosis"))
            confidence = (
                pi["confidence"]
                if "confidence" in pi
                else record.get("confidence")
            )

            responses.append(
                {
                    "author_id": model_id,
                    "image_index": pi.get("image_index"),
                    "content_en": content,
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                }
            )
    else:
        # Fallback to a single response from the top-level record
        if any(record.get(k) is None for k in ("diagnosis", "confidence", "reasoning")) and record.get("raw_output"):
            m = re.search(r"```json\s*(\{.*?\})\s*```", record["raw_output"], flags=re.S | re.I)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                    for k in ("diagnosis", "confidence", "reasoning"):
                        if record.get(k) is None and k in parsed:
                            record[k] = parsed[k]
                except Exception:
                    pass

        responses.append(
            {
                "author_id": model_id,
                "content_en": record.get("reasoning") or record.get("diagnosis") or "",
                "diagnosis": record.get("diagnosis"),
                "confidence": record.get("confidence"),
            }
        )

    return {
        "encounter_id": encounter_id,
        "author_id": model_id,  # top-level author set to model id/name
        "image_ids": image_ids,
        "responses": responses,
    }


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Iterate JSON records from JSONL. Skips blank lines. Logs malformed lines to stderr."""
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                print(
                    f"[warn] Skipping line {ln}: JSON decode error: {e}",
                    file=sys.stderr,
                )
                continue
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item
                    else:
                        print(
                            f"[warn] Line {ln}: list element is not an object, skipped.",
                            file=sys.stderr,
                        )
            else:
                print(
                    f"[warn] Line {ln}: top-level value is not an object, skipped.",
                    file=sys.stderr,
                )


def _iter_records_flex(path: str) -> Iterable[Dict[str, Any]]:
    """
    Try JSONL first; if the file appears to be a single JSON object/array,
    parse whole-file JSON as a fallback.
    """
    # First pass: try JSONL (works even if it's one JSON line)
    any_yielded = False
    for rec in _iter_jsonl(path):
        any_yielded = True
        yield rec
    if any_yielded:
        return

    # Fallback: try whole-file JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise SystemExit(f"[error] Could not parse file as JSON/JSONL: {e}")

    if isinstance(data, dict):
        yield data
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
            else:
                print("[warn] Array element is not an object, skipped.", file=sys.stderr)
    else:
        raise SystemExit("[error] Top-level JSON must be an object or an array.")


def convert_file(input_path: str, output_path: str = None) -> str:
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}.converted.jsonl"

    written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for rec in _iter_records_flex(input_path):
            converted = convert_record(rec)
            out_f.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    print(f"[info] Wrote {written} record(s) to {output_path}", file=sys.stderr)
    return output_path


def main():
    ap = argparse.ArgumentParser(
        description="Convert derm-agent JSON/JSONL to English-only JSONL with responses."
    )
    ap.add_argument("--input", help="Path to input .json or .jsonl file")
    ap.add_argument(
        "-o",
        "--output",
        help='Path to output .jsonl (default: "<input>.converted.jsonl")',
    )
    args = ap.parse_args()
    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()

