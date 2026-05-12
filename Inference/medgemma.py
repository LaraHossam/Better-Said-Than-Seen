#!/usr/bin/env python3
"""
local_medgemma_inference.py

Run google/medgemma-4b-it locally (Transformers) on Derma-style JSONL datasets.

Modes:
  --mode multi         : all images at once + (query_title_en/query_content_en)
  --mode text          : text-only using (query_title_en/query_content_en)
  --mode vignette_full : text-only using structured `vignette` from vignette JSONL (keeps image_findings)
  --mode vignette_text : text-only using structured `vignette` but removes vignette["image_findings"]

Default vignette JSONL path:
  /home/USER/Documents/Dermatology-Evaluation-Framework/Thesis_Final/vignettes/test_vignettes.jsonl

Output JSONL: one row per encounter (written sequentially).
"""

from __future__ import annotations
import random
import argparse
import copy
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_MODEL_ID = "google/medgemma-4b-it"
DEFAULT_VIGNETTE_JSONL = (
    "/home/USER/Documents/Dermatology-Evaluation-Framework/Thesis_Final/vignettes/test_vignettes.jsonl"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a board-certified dermatologist. "
    "Given the patient's text history (and images if provided), respond ONLY in valid JSON with keys: "
    '{"diagnosis": string, "confidence": number (0-1), "reasoning": string}. '
    "Do not output any extra text outside the JSON."
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def resolve_image_path(images_root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(images_root, p)


def load_images(images_root: str, image_paths: List[str], max_images: int) -> Tuple[List[Image.Image], List[str]]:
    selected = (image_paths or [])[:max_images]
    pil_images: List[Image.Image] = []
    image_ids: List[str] = []
    for p in selected:
        ap = resolve_image_path(images_root, p)
        img = Image.open(ap).convert("RGB")
        pil_images.append(img)
        image_ids.append(p)
    return pil_images, image_ids


def build_user_text_from_query(query_title_en: str, query_content_en: str) -> str:
    title = (query_title_en or "").strip()
    body = (query_content_en or "").strip()
    if title and body:
        return f"Title: {title}\n\nUser: {body}"
    if title:
        return f"Title: {title}"
    return body


def _json_dumps_compact(obj: Any) -> str:
    # Compact but stable; easier for models than pretty-print with lots of whitespace
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_user_text_from_vignette(vignette_obj: Dict[str, Any], drop_image_findings: bool) -> str:
    v = copy.deepcopy(vignette_obj) if vignette_obj else {}
    if drop_image_findings and isinstance(v, dict):
        v.pop("image_findings", None)

    # Send ONLY the structured vignette, as requested.
    # Wrap with a short header so the model knows what it is.
    return "Structured vignette (JSON):\n" + _json_dumps_compact(v)


_JSON_BLOCK_RE = re.compile(
    r"(?s)(?:```|~~~)\s*(?:json\b)?\s*\n\s*(\{.*?\})\s*\n(?:```|~~~)"
)
def extract_json_from_text(s: str) -> Dict[str, Any]:
    print(s)
    s = (s or "").strip()

    def _try_parse(x: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # direct parse
    direct = _try_parse(s)
    if direct:
        return direct

    # fenced block parse (capture only inner JSON)
    m = _JSON_BLOCK_RE.search(s)
    if m:
        parsed = _try_parse(m.group(1))
        if parsed:
            return parsed

    return {"diagnosis": None, "confidence": None, "reasoning": None}

def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None
    
def build_derangement(indices: List[int], rng: random.Random) -> List[int]:
    """
    Return a permutation p of indices such that p[i] != indices[i] for all i
    (a derangement), when possible.
    """
    n = len(indices)
    if n < 2:
        raise ValueError("Need at least 2 samples to build a shuffled-image derangement.")

    p = indices[:]
    for _ in range(200):
        rng.shuffle(p)
        if all(p[i] != indices[i] for i in range(n)):
            return p

    # fallback fix
    p = indices[:]
    rng.shuffle(p)
    for i in range(n):
        if p[i] == indices[i]:
            j = (i + 1) % n
            p[i], p[j] = p[j], p[i]

    if not all(p[i] != indices[i] for i in range(n)):
        raise ValueError("Failed to construct derangement; try a different seed.")
    return p

def build_output_row(
    encounter_id: str,
    author_id: str,
    image_ids: List[str],
    parsed: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the exact output schema requested by the user.
    """
    return {
        "encounter_id": encounter_id,
        "author_id": author_id,
        "image_ids": image_ids,
        "responses": [
            {
                "author_id": author_id,
                "image_index": 0,
                "content_en": parsed.get("reasoning"),
                "diagnosis": parsed.get("diagnosis"),
                "confidence": safe_float(parsed.get("confidence")),
            }
        ],
    }

@torch.inference_mode()
def medgemma_generate(
    model,
    processor,
    system_prompt: str,
    user_text: str,
    images: Optional[List[Image.Image]],
    temperature: float,
    max_new_tokens: int,
    device: str,
) -> Tuple[str, int, int]:
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    if images:
        for img in images:
            user_content.append({"type": "image", "image": img})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    prompt_text = None
    if hasattr(processor, "apply_chat_template"):
        prompt_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        proc_inputs = processor(
            text=prompt_text,
            images=images if images else None,
            return_tensors="pt",
        )
    else:
        proc_inputs = processor(
            text=messages,
            images=images if images else None,
            return_tensors="pt",
        )

    proc_inputs = {k: v.to(device) for k, v in proc_inputs.items() if torch.is_tensor(v)}

    prompt_tokens_est = int(proc_inputs["input_ids"].shape[-1]) if "input_ids" in proc_inputs else 0
    do_sample = temperature is not None and temperature > 0.0

    gen_ids = model.generate(
        **proc_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
    )

    decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    if prompt_text and prompt_text in decoded:
        generated = decoded.split(prompt_text, 1)[-1].strip()
        completion_tokens_est = max(0, int(gen_ids.shape[-1] - prompt_tokens_est))
        return generated, prompt_tokens_est, completion_tokens_est

    if "input_ids" in proc_inputs:
        prompt_len = proc_inputs["input_ids"].shape[-1]
        completion_tokens_est = int(gen_ids.shape[-1] - prompt_len)
    else:
        completion_tokens_est = 0

    return decoded.strip(), prompt_tokens_est, completion_tokens_est


def _infer_dataset_path(args) -> str:
    # If user explicitly passes --jsonl, always respect it.
    # Otherwise, vignette modes default to the vignette JSONL path.
    if args.jsonl is not None:
        return args.jsonl
    if args.mode in ("vignette_full", "vignette_text"):
        return args.vignette_jsonl
    raise ValueError("No --jsonl provided and mode is not a vignette mode.")


def main():
    ap = argparse.ArgumentParser()

    # For backward compatibility: allow either --jsonl OR (for vignette modes) default vignette path.
    ap.add_argument("--jsonl", default=None, help="Path to dataset.jsonl (for multi/text) OR override for vignette modes")
    ap.add_argument(
        "--vignette-jsonl",
        default=DEFAULT_VIGNETTE_JSONL,
        help="Path to vignette JSONL (used when --mode vignette_full|vignette_text and --jsonl not provided)",
    )

    ap.add_argument("--images-root", default="", help="Root folder for image files (required for multi mode)")
    ap.add_argument("--output", required=True, help="Output JSONL path")

    ap.add_argument(
    "--mode",
    choices=["multi", "multi_shuffled", "text", "vignette_full", "vignette_text", "image", "image_shuffled"],
    default="multi",
    help=(
        "multi=all images at once; "
        "multi_shuffled=text from encounter i + images from different encounter; "
        "text=text-only; "
        "vignette_full/vignette_text use structured vignettes; "
        "image=image-only; "
        "image_shuffled=image-only but images from different encounter"
    ),
    )
    ap.add_argument("--shuffle-seed", type=int, default=1337)

    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max-images", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--start-id", type=int, default=None, help="Start encounter numeric id (expects ENCxxxxx)")
    ap.add_argument("--end-id", type=int, default=None, help="End encounter numeric id (expects ENCxxxxx)")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    dataset_path = _infer_dataset_path(args)
    data = read_jsonl(dataset_path)

    if args.start_id is not None and args.end_id is not None:
        filtered = []
        for ex in data:
            enc = ex.get("encounter_id", "")
            try:
                n = int(str(enc).replace("ENC", ""))
            except Exception:
                continue
            if args.start_id <= n <= args.end_id:
                filtered.append(ex)
        data = filtered
        print(f"📦 Filtered to {len(data)} samples (ENC{args.start_id:05d}–ENC{args.end_id:05d})")
        
    donor_perm: Optional[List[int]] = None
    if args.mode in ("multi_shuffled", "image_shuffled"):
        if len(data) < 2:
            raise ValueError(f"{args.mode} requires at least 2 samples.")
        rng = random.Random(args.shuffle_seed)
        idxs = list(range(len(data)))
        donor_perm = build_derangement(idxs, rng)
        print(f"🔀 Shuffled image mode enabled ({args.mode}, seed={args.shuffle_seed}).")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for i, ex in enumerate(tqdm(data, desc=f"Inferring ({args.mode})")):            
            encounter_id = ex.get("encounter_id")
            shuffled_images = False
            shuffled_from_encounter_id: Optional[str] = None
            t0 = time.time()
            author_id = args.model_id  # or set to a shorter name if you want
            row: Dict[str, Any] = {
                "encounter_id": encounter_id,
                "author_id": author_id,
                "image_ids": [],
                "responses": [],
            }

            try:
                images: Optional[List[Image.Image]] = None
                image_ids: List[str] = []

                # ---- Build input depending on mode ----
                if args.mode in ("text", "multi", "multi_shuffled"):
                    query_title_en = ex.get("query_title_en", "")
                    query_content_en = ex.get("query_content_en", "")
                    user_text = build_user_text_from_query(query_title_en, query_content_en)

                    if args.mode in ("multi", "multi_shuffled"):
                        if not args.images_root:
                            raise ValueError("--images-root is required for --mode multi/multi_shuffled")

                        donor_ex = ex
                        if args.mode == "multi_shuffled":
                            assert donor_perm is not None
                            donor_ex = data[donor_perm[i]]
                            shuffled_images = True
                            shuffled_from_encounter_id = donor_ex.get("encounter_id")

                        image_paths = donor_ex.get("image_ids", []) or []
                        images, image_ids = load_images(args.images_root, image_paths, args.max_images)
                    else:
                        images = None
                        image_ids = []

                elif args.mode in ("image", "image_shuffled"):
                    user_text = ""  # image-only (you already did this)

                    if not args.images_root:
                        raise ValueError("--images-root is required for --mode image/image_shuffled")

                    donor_ex = ex
                    if args.mode == "image_shuffled":
                        assert donor_perm is not None
                        donor_ex = data[donor_perm[i]]
                        shuffled_images = True
                        shuffled_from_encounter_id = donor_ex.get("encounter_id")

                    image_paths = donor_ex.get("image_ids", []) or []
                    images, image_ids = load_images(args.images_root, image_paths, args.max_images)

                elif args.mode in ("vignette_full", "vignette_text"):
                    vignette_obj = ex.get("vignette")
                    if vignette_obj is None:
                        raw = ex.get("vignette_raw_text")
                        if isinstance(raw, str):
                            vignette_obj = json.loads(raw)
                        else:
                            vignette_obj = None

                    if not isinstance(vignette_obj, dict):
                        raise ValueError("Missing or invalid `vignette` object in vignette JSONL line.")

                    drop_img_findings = (args.mode == "vignette_text")
                    user_text = build_user_text_from_vignette(vignette_obj, drop_image_findings=drop_img_findings)

                    # IMPORTANT: no images in vignette modes (as requested)
                    images = None
                    image_ids = ex.get("image_ids", []) or []

                else:
                    raise ValueError(f"Unknown mode: {args.mode}")

                gen_text, prompt_tok, completion_tok = medgemma_generate(
                    model=model,
                    processor=processor,
                    system_prompt=args.system_prompt,
                    user_text=user_text,
                    images=images,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )

                parsed = extract_json_from_text(gen_text)

                row = build_output_row(
                    encounter_id=encounter_id,
                    author_id=author_id,
                    image_ids=image_ids,
                    parsed=parsed,
                )

                row["shuffled_images"] = shuffled_images
                row["shuffled_from_encounter_id"] = shuffled_from_encounter_id

            except Exception as e:
                row = {
                    "encounter_id": encounter_id,
                    "author_id": args.model_id,
                    "image_ids": image_ids if "image_ids" in locals() else [],
                    "responses": [
                        {
                            "author_id": args.model_id,
                            "image_index": 0,
                            "content_en": None,
                            "diagnosis": None,
                            "confidence": None,
                            "error": str(e),
                        }
                    ],
                }

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"✅ Done. Wrote: {args.output}")


if __name__ == "__main__":
    main()

"""
python /home/USER/Documents/Dermatology-Evaluation-Framework/medgemma.py \
  --jsonl /home/USER/Documents/DermAgent-Chat/dataset/test/test.jsonl \
  --images-root /home/USER/Documents/DermAgent-Chat/dataset/test/images_test \
  --output local_medgemma_4b_image_shuffled.jsonl \
  --mode multi_shuffled \
--shuffle-seed 1337 \
  --max-images 5 \
  --temperature 0.0 \
  --max-new-tokens 256

"""