#!/usr/bin/env python3
"""
OpenRouter / DeepInfra multimodal diagnosis inference over vignette JSONL
(resume-safe, append-only) with optional images.

Input JSONL: one encounter per line. Each line should contain:
- encounter_id (string)
- vignette (dict) OR vignette_raw_text (string) OR vignette_text (string/dict)
- OPTIONAL: image_ids (list of strings)  e.g. ["IMG_ENC00001_00001.jpg", ...]

Output JSONL (append-only):
<model>_<provider>_inference_vignettes_plus_images.jsonl
- Preserves all original fields
- Adds/overwrites "responses" with ONE model response:
  [{"author_id": <provider:model>, "content_en": <diagnosis>}]

Deps:
  pip install requests tqdm pillow

Example:
python openrouter_vignette_inference_mm.py \
  --input-jsonl /path/to/test_vignettes.jsonl \
  --provider openrouter \
  --model qwen/qwen2.5-vl-72b-instruct \
  --images-dir /path/to/images \
  --max-images 6
"""

from __future__ import annotations

from tqdm.auto import tqdm
import argparse
import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests

try:
    from PIL import Image
except ImportError:
    Image = None  # optional


# =========================
# Prompt
# =========================
SYSTEM_PROMPT = """You are a board-certified dermatologist.

You will be given:
- one structured dermatology case vignette (history + reported symptoms), and
- one or more clinical photographs.

TASK
Respond as you would to a patient asking a medical question on a public forum:

- Clearly state the most likely diagnosis.
- Briefly explain the clinical reasoning in 2–3 concise sentences, using ONLY evidence from:
  (a) the vignette, and (b) what is visibly present in the image(s).
- If relevant, briefly acknowledge alternative diagnoses or uncertainty within the same explanation.

IMPORTANT RULES
- Do NOT introduce information not present in the vignette or images.
- Do NOT assume tests/biopsy/treatment response.
- Do NOT provide treatment, management advice, or next steps.
- Do NOT include disclaimers or “see a doctor” language.
- Do NOT mention image quality unless it directly limits diagnostic confidence.

STYLE AND SCOPE
- Direct, neutral, non-chatty.
- One short paragraph total (2–3 sentences).

OUTPUT FORMAT (STRICT)
Return plain text only:

<Diagnosis stated clearly. One paragraph containing the diagnosis followed by 2–3 sentences of reasoning.>
"""


# =========================
# Providers: headers + chat
# =========================
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _openrouter_headers() -> Dict[str, str]:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    referer = os.environ.get("OPENROUTER_REFERER", "").strip()
    title = os.environ.get("OPENROUTER_TITLE", "").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _deepinfra_headers() -> Dict[str, str]:
    api_key = os.environ.get("DEEPINFRA_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPINFRA_API_KEY not set.")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def provider_chat(
    provider: str,
    model: str,
    messages: list,
    temperature: float = 0.0,
    max_tokens: int = 700,
    timeout_s: int = 180,
) -> Dict[str, Any]:
    """
    Unified chat call for supported providers (OpenAI-compatible).
    - provider: 'openrouter' or 'deepinfra'
    - model: provider model id
    - messages: OpenAI messages list
    """
    if provider == "openrouter":
        url = OPENROUTER_URL
        headers = _openrouter_headers()
        payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
    elif provider == "deepinfra":
        base = os.environ.get("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
        url = f"{base}/chat/completions"
        headers = _deepinfra_headers()
        payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
    else:
        raise ValueError("Unsupported provider: choose 'openrouter' or 'deepinfra'")

    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if r.status_code >= 400:
        body = r.text
        try:
            j = r.json()
            body = json.dumps(j, ensure_ascii=False)
        except Exception:
            pass
        raise RuntimeError(f"{provider} HTTP {r.status_code}: {body}")

    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"{provider} returned error payload: {data.get('error')}")
    return data


# =========================
# JSONL utilities
# =========================
def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_model_filename(model: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")
    return s or "model"


def load_seen_encounter_ids(output_jsonl: Path) -> Set[str]:
    seen: Set[str] = set()
    if not output_jsonl.exists():
        return seen
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                eid = rec.get("encounter_id")
                if isinstance(eid, str) and eid.strip():
                    seen.add(eid.strip())
            except Exception:
                continue
    return seen


def vignette_to_text(rec: Dict[str, Any]) -> str:
    v = rec.get("vignette")
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False, indent=2)

    vt = rec.get("vignette_text")
    if isinstance(vt, dict):
        return json.dumps(vt, ensure_ascii=False, indent=2)
    if isinstance(vt, str) and vt.strip():
        return vt.strip()

    raw = rec.get("vignette_raw_text")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    raise ValueError("No vignette found in record (expected vignette / vignette_text / vignette_raw_text).")


# =========================
# Image helpers (base64 data URL)
# =========================
_MIME_BY_EXT = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _file_to_data_url(path: Path) -> str:
    ext = path.suffix.lower()
    mime = _MIME_BY_EXT.get(ext)
    if not mime:
        raise ValueError(f"Unsupported image extension: {ext} ({path.name})")

    b = path.read_bytes()
    b64 = base64.b64encode(b).decode("ascii")
    return f"data:{mime};base64,{b64}"


def resolve_image_paths(
    rec: Dict[str, Any],
    images_dir: Optional[Path],
    max_images: int,
) -> Tuple[List[Path], List[str]]:
    """
    Returns (paths_found, errors)
    """
    errs: List[str] = []
    image_ids = rec.get("image_ids")

    if not image_ids:
        return [], errs
    if not isinstance(image_ids, list):
        return [], [f"image_ids not a list (type={type(image_ids).__name__})"]

    if images_dir is None:
        return [], ["image_ids present but --images-dir not provided"]

    paths: List[Path] = []
    for img_id in image_ids[: max_images if max_images > 0 else len(image_ids)]:
        if not isinstance(img_id, str) or not img_id.strip():
            errs.append(f"bad image_id: {img_id!r}")
            continue
        p = images_dir / img_id
        if not p.exists():
            errs.append(f"missing image file: {p}")
            continue
        paths.append(p)

    return paths, errs


# =========================
# Multimodal message builder
# =========================
def build_mm_messages(vignette_text: str, image_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    OpenAI-style multimodal message:
    messages = [
      {"role":"system","content":SYSTEM_PROMPT},
      {"role":"user","content":[
         {"type":"text","text": "..."},
         {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}},
         ...
      ]}
    ]
    """
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": vignette_text}]

    for p in image_paths:
        url = _file_to_data_url(p)
        user_content.append({"type": "image_url", "image_url": {"url": url}})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def diagnose_provider_mm(
    provider: str,
    model: str,
    vignette_text: str,
    image_paths: List[Path],
    temperature: float = 0.0,
    max_tokens: int = 700,
    retries: int = 1,
) -> Dict[str, Any]:
    messages = build_mm_messages(vignette_text=vignette_text, image_paths=image_paths)

    last_exc = None
    for attempt in range(retries + 1):
        try:
            data = provider_chat(
                provider=provider,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not isinstance(data, dict):
                return {"content": None, "raw": data, "error": "non-dict response"}

            choices = data.get("choices") or []
            if not choices:
                return {"content": None, "raw": data, "error": "no choices in response"}

            message = choices[0].get("message") or {}
            content = (message.get("content") or "").strip()
            if not content:
                return {"content": None, "raw": data, "error": "empty message content"}

            return {"content": content, "raw": data, "error": None}

        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(1 + 2 * attempt)
                continue
            return {"content": None, "raw": None, "error": f"exception: {type(e).__name__}: {e}"}


# =========================
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--provider", type=str, choices=("openrouter", "deepinfra"), default="openrouter")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=700)

    ap.add_argument("--images-dir", type=Path, default=None,
                    help="Directory containing the image files referenced by image_ids.")
    ap.add_argument("--max-images", type=int, default=8,
                    help="Max number of images per case to send (useful for token/cost control).")
    ap.add_argument("--vignette-model", type=str, default="gpt-4.1",)
    args = ap.parse_args()

    input_path = args.input_jsonl
    model = args.model.strip()
    provider = args.provider

    out_name = f"{safe_model_filename(model)}_{provider}_inference_vignettes_plus_images.jsonl"
    # output_path = input_path.parent / out_name
    output_path = f"/Users/USER/Documents/Dermatology-Evaluation/Vignettes/Logs/{args.vignette_model}" / out_name

    seen = load_seen_encounter_ids(output_path)
    print(f"Resuming inference. Already seen {len(seen)} encounter_id(s).")

    total = sum(1 for _ in iter_jsonl(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as out_f, tqdm(total=total, desc=f"{provider}:{model}", unit="rec") as pbar:
        for i, rec in enumerate(iter_jsonl(input_path)):
            encounter_id = str(rec.get("encounter_id") or f"enc_{i:06d}").strip()
            if encounter_id in seen:
                pbar.update(1)
                continue

            out_rec = dict(rec)
            out_rec["encounter_id"] = encounter_id
            out_rec["inference_provider"] = provider
            out_rec["inference_model"] = model
            out_rec["inference_mode"] = "vignette+images"

            # Vignette
            try:
                vignette_text = vignette_to_text(rec)
            except Exception as e:
                out_rec["responses"] = []
                out_rec["error"] = f"No vignette: {type(e).__name__}: {e}"
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                out_f.flush()
                seen.add(encounter_id)
                pbar.update(1)
                continue

            # Images
            img_paths, img_errs = resolve_image_paths(
                rec=rec,
                images_dir=args.images_dir,
                max_images=max(0, args.max_images),
            )
            if img_errs:
                out_rec["image_loading_warnings"] = img_errs
            out_rec["num_images_sent"] = len(img_paths)

            # Inference
            result = diagnose_provider_mm(
                provider=provider,
                model=model,
                vignette_text=vignette_text,
                image_paths=img_paths,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                retries=1,
            )

            out_rec["raw_provider_response"] = result.get("raw")
            if result.get("error") is not None:
                out_rec["responses"] = []
                out_rec["error"] = result["error"]
            else:
                out_rec["responses"] = [{"author_id": f"{provider}:{model}", "content_en": result["content"]}]

            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()
            seen.add(encounter_id)
            pbar.update(1)

    print(f"Done. Wrote/updated: {output_path}")


if __name__ == "__main__":
    main()


# python /Users/USER/Documents/Dermatology-Evaluation/Vignettes/openrouter_vignette_inference_mm.py \
#   --input-jsonl /Users/USER/Documents/Dermatology-Evaluation/Vignettes/test_vignettes_google_gemini-2.5-flash-lite.jsonl \
#   --provider openrouter \
#   --model openai/gpt-4.1 \
#   --images-dir /Users/USER/Documents/Dermatology-Evaluation/Data/images_test \
#   --max-images 6