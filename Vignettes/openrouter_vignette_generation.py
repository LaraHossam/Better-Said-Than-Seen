#!/usr/bin/env python3
"""
OpenRouter vignette generator (non-agentic) — strict schema via prompt, no validation.

Stage 1: (images + query) -> OpenRouter VLM/LLM -> vignette JSON (strict template enforced by prompt)
- No Pydantic, no validation
- Extract first JSON object from model output and write to JSONL
- Preserve ALL original metadata

Deps:
  pip install requests pillow

"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from PIL import Image, ImageOps


# =========================
# Prompt: stricter + more comprehensive (but concise)
# =========================

VIGNETTE_SYSTEM = """You are a board-certified dermatologist writing a structured dermatology vignette for benchmarking.

INPUTS
- A user query (free text), plus one or more clinical photographs.

NON-NEGOTIABLE OUTPUT RULES
- Output MUST be valid JSON: exactly ONE JSON object.
- No markdown, no code fences, no commentary, no extra keys.
- Use EXACT key names and nesting from the template below.
- Do NOT provide diagnosis, differential diagnosis, or management/treatment.

EVIDENCE RULES
- Extract history ONLY from the user query. Do NOT guess age/sex/race/PMH if not stated.
- race_or_ethnicity_reported: ONLY if explicitly stated by the user; otherwise null.
- From photos: describe what is VISIBLE. If uncertain, say so in uncertainty_notes.

WHAT "COMPREHENSIVE BUT TO THE POINT" MEANS
- Populate every field you can.
- For image_findings fields, use dense dermatology language in short phrases:
  morphology + primary lesion(s) + count (approx) + size (approx) + shape + border + color + scale/crust/erosion
  + configuration (annular/linear/grouped/reticular/targetoid/etc) + distribution and laterality + site(s).
- If multiple sites are visible, describe each site distinctly within image_findings.summary.
- Mention background skin (e.g., xerosis, post-inflammatory hyperpigmentation), if visible.
- Note nails/hair/mucosa only if visible or specifically absent/unclear.
- Add 5–10 missing_high_yield_questions (high yield, short, prioritized).

JSON TEMPLATE (copy exactly; fill values only; unknown -> null or []):
{
  "encounter_id": "<string>",
  "age": null,
  "sex_or_gender": null,
  "race_or_ethnicity_reported": null,
  "skin_tone_visual_estimate": null,
  "fitzpatrick_type_visual_estimate": null,

  "chief_concern": null,
  "hpi": null,

  "body_sites_reported": [],
  "symptoms_reported": [],
  "duration_or_timeline": null,
  "course_pattern": null,

  "prior_treatments_reported": [],
  "exposures_triggers": [],
  "relevant_history": [],
  "medications_reported": [],
  "allergies_reported": [],

  "image_findings": {
    "summary": null,
    "primary_lesion": null,
    "morphology": null,
    "distribution": null,
    "color": null,
    "surface_secondary_changes": null,
    "mucosa_nails_hair": null,
    "photo_quality_issues": []
  },

  "missing_high_yield_questions": [],
  "uncertainty_notes": null
}
"""


# =========================
# OpenRouter client
# =========================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

def _or_headers() -> Dict[str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.environ.get("OPENROUTER_REFERER", "").strip()
    title = os.environ.get("OPENROUTER_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers

def openrouter_chat(payload: Dict[str, Any], timeout_s: int = 180) -> Dict[str, Any]:
    r = requests.post(OPENROUTER_URL, headers=_or_headers(), json=payload, timeout=timeout_s)
    print(f"OpenRouter response status: {r.status_code}")
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text}")
    return r.json()


# =========================
# JSONL + input helpers
# =========================

def load_existing_encounter_ids(output_jsonl: Path) -> set:
    """
    Return a set of encounter_id values already present in output_jsonl.
    Robust to blank lines / malformed JSON lines.
    """
    ids = set()
    if not output_jsonl.exists():
        return ids

    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            eid = obj.get("encounter_id")
            if eid is not None and str(eid).strip():
                ids.add(str(eid).strip())
    return ids


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def build_query_text(rec: Dict[str, Any]) -> str:
    if isinstance(rec.get("query"), str) and rec["query"].strip():
        return rec["query"].strip()
    title = (rec.get("query_title_en") or rec.get("title") or "").strip()
    content = (rec.get("query_content_en") or rec.get("content") or "").strip()
    combined = (title + "\n\n" + content).strip()
    return combined if combined else "Not provided"

def extract_image_ids(rec: Dict[str, Any]) -> List[str]:
    for key in ("image_ids", "images", "image_paths", "image_files"):
        val = rec.get(key)
        if isinstance(val, list) and val:
            return [str(x) for x in val]
    for key in ("image", "image_path", "image_file"):
        val = rec.get(key)
        if isinstance(val, str) and val.strip():
            return [val.strip()]
    return []

def encode_image_as_data_url(image_path: str, max_side: int = 1024) -> str:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(p) as img:
        # Fix common camera orientation issues
        img = ImageOps.exif_transpose(img)
        img = img.convert("RGB")

        w, h = img.size
        scale = max(w, h) / float(max_side) if max(w, h) > max_side else 1.0
        if scale > 1.0:
            img = img.resize((int(round(w / scale)), int(round(h / scale))), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


# =========================
# Robust JSON extraction (no validation)
# =========================

def extract_first_json_object(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Empty model output.")

    s = text.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)

    # Extract first {...} by brace matching
    if "{" in s and "}" in s:
        start = s.find("{")
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]

    # If it looks like JSON kv-pairs without braces, wrap it
    if s.startswith('"') and ":" in s:
        return "{" + s + "}"

    raise ValueError("Could not extract JSON object from model output.")


# =========================
# Vignette generation (JSON-only)
# =========================

def build_vignette_messages(encounter_id: str, query: str, image_paths: List[str], max_image_side: int):
    n_imgs = len(image_paths)

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"encounter_id: {encounter_id}\n"
                f"images_count: {n_imgs}\n\n"
                f"USER QUERY (verbatim):\n{query}\n\n"
                "STRICT INSTRUCTIONS:\n"
                "- Return ONLY the JSON object exactly matching the JSON TEMPLATE in the system message.\n"
                "- Use key 'encounter_id' (not 'Encounter ID'). Do not output keys like 'Clinical Details'.\n"
                "- If age/sex/race/PMH/meds/allergies are not stated, use null/[] and add follow-up questions.\n"
                "- From photos: write a dense, dermatology-style description (count/size/shape/border/configuration/"
                "color/scale/crust/erosion; distribution; laterality; sites).\n"
                "- If you did not receive any images, include 'No images received by model' in "
                "image_findings.photo_quality_issues and explain in uncertainty_notes.\n"
            ),
        }
    ]

    # Interleave labels + images to make multi-image conditioning more reliable
    for i, p in enumerate(image_paths, start=1):
        data_url = encode_image_as_data_url(p, max_side=max_image_side)
        content.append({"type": "text", "text": f"Image {i} of {n_imgs}:"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    return [
        {"role": "system", "content": VIGNETTE_SYSTEM},
        {"role": "user", "content": content},
    ]

def generate_vignette_openrouter(
    model: str,
    encounter_id: str,
    query: str,
    image_paths: List[str],
    max_image_side: int = 1024,
    temperature: float = 0.0,
    max_tokens: int = 900,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": build_vignette_messages(encounter_id, query, image_paths, max_image_side),
    }

    data = openrouter_chat(payload)
    raw = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

    json_str = extract_first_json_object(raw)
    obj = json.loads(json_str)

    # Force encounter_id consistency
    obj["encounter_id"] = str(encounter_id)

    return {"raw_text": raw, "json": obj}


# =========================
# Runner
# =========================

def run(
    input_jsonl: Path,
    output_jsonl: Path,
    image_base_dir: Optional[Path],
    vignette_model: str,
    max_images: int,
    max_image_side: int,
    vignette_temperature: float,
    vignette_max_tokens: int,
) -> None:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Resume behavior: skip encounters already written
    existing_ids = load_existing_encounter_ids(output_jsonl)
    print(f"[resume] Found {len(existing_ids)} encounter_id(s) already in {output_jsonl}")

    # Append new lines (do not overwrite existing output)
    with output_jsonl.open("a", encoding="utf-8") as out_f:
        for idx, rec in enumerate(iter_jsonl(input_jsonl)):
            encounter_id = rec.get("encounter_id") or f"enc_{idx:06d}"
            encounter_id = str(encounter_id).strip()

            if encounter_id in existing_ids:
                # Already processed → skip
                continue

            query = build_query_text(rec)

            image_ids = extract_image_ids(rec)[:max_images]
            image_paths: List[str] = []
            for img in image_ids:
                p = Path(str(img))
                if image_base_dir is not None and not p.is_absolute():
                    p = image_base_dir / p
                image_paths.append(str(p))

            print(f"[{encounter_id}] query_len={len(query)} images={len(image_paths)}")

            out_rec = dict(rec)  # preserve ALL metadata
            out_rec["encounter_id"] = encounter_id
            out_rec["image_ids"] = image_ids

            try:
                vignette_out = generate_vignette_openrouter(
                    model=vignette_model,
                    encounter_id=encounter_id,
                    query=query,
                    image_paths=image_paths,
                    max_image_side=max_image_side,
                    temperature=vignette_temperature,
                    max_tokens=vignette_max_tokens,
                )
                out_rec.update(
                    {
                        "vignette_model": vignette_model,
                        "vignette_raw_text": vignette_out["raw_text"],
                        "vignette": vignette_out["json"],
                    }
                )
            except Exception as e:
                out_rec.update(
                    {
                        "vignette_model": vignette_model,
                        "vignette_raw_text": None,
                        "vignette": None,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )

            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()

            # Important: mark as done so reruns in the same session don’t duplicate
            existing_ids.add(encounter_id)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--image-base-dir", type=Path, default=None)
    ap.add_argument("--vignette-model", type=str, required=True)

    ap.add_argument("--max-images", type=int, default=5)
    ap.add_argument("--max-image-side", type=int, default=1024)
    ap.add_argument("--vignette-temperature", type=float, default=0.0)
    ap.add_argument("--vignette-max-tokens", type=int, default=900)
    args = ap.parse_args()

    run(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        image_base_dir=args.image_base_dir,
        vignette_model=args.vignette_model,
        max_images=args.max_images,
        max_image_side=args.max_image_side,
        vignette_temperature=args.vignette_temperature,
        vignette_max_tokens=args.vignette_max_tokens,
    )


if __name__ == "__main__":
    main()
''' 
python /Users/USER/Documents/Dermatology-Evaluation/Vignettes/openrouter_vignette_generation.py \
     --input-jsonl /Users/USER/Documents/Dermatology-Evaluation/Data/test.jsonl   \
         --output-jsonl /Users/USER/Documents/Dermatology-Evaluation/Vignettes/test_vignettes_qwen-Qwen2.5-VL-32B.jsonl  \
            --image-base-dir /Users/USER/Documents/Dermatology-Evaluation/Data/images_test \
                --vignette-model qwen/qwen2.5-vl-72b-instruct
                
'''