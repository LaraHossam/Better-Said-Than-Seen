#!/usr/bin/env python3
"""
local_medmo_inference.py

Run MBZUAI/MedMO-4B (Qwen3-VL architecture) locally (Transformers) on Derma-style JSONL datasets.

Modes:
  --mode multi         : images + (query_title_en/query_content_en)
  --mode text          : text-only using (query_title_en/query_content_en)
  --mode image         : image-only
  --mode vignette_full : text-only using structured `vignette` from vignette JSONL (keeps image_findings)
  --mode vignette_text : text-only using structured `vignette` but removes vignette["image_findings"]

Default vignette JSONL path:
  /home/USER/Documents/Dermatology-Evaluation-Framework/Thesis_Final/vignettes/test_vignettes.jsonl

Output JSONL: one row per encounter (written sequentially).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    # pip install qwen-vl-utils
    from qwen_vl_utils import process_vision_info
except Exception as e:
    raise ImportError(
        "Missing dependency `qwen-vl-utils`. Install with: pip install qwen-vl-utils\n"
        f"Original import error: {e}"
    )

DEFAULT_MODEL_ID = "MBZUAI/MedMO-4B"

# DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_VIGNETTE_JSONL = (
    "/home/USER/Documents/Dermatology-Evaluation-Framework/Thesis_Final/vignettes/test_vignettes.jsonl"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a board-certified dermatologist. "
    "Given the patient's text history (and images if provided), respond ONLY in valid JSON with keys: "
    '{"diagnosis": string, "confidence": number (0-1), "reasoning": string}. '
    "Do not output any extra text outside the JSON."
)

_JSON_BLOCK_RE = re.compile(
    r"(?s)(?:```|~~~)\s*(?:json\b)?\s*\n\s*(\{.*?\})\s*\n(?:```|~~~)"
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


def load_image_paths(images_root: str, image_paths: List[str], max_images: int) -> Tuple[List[str], List[str]]:
    """
    For Qwen3-VL utils, we pass file paths (strings) into the message content.
    Returns:
      - abs_paths: absolute paths to feed the model
      - image_ids: the original (possibly relative) ids for traceability/output
    """
    selected = (image_paths or [])[:max_images]
    abs_paths: List[str] = []
    image_ids: List[str] = []
    for p in selected:
        ap = resolve_image_path(images_root, p)
        abs_paths.append(ap)
        image_ids.append(p)
    return abs_paths, image_ids


def build_user_text_from_query(query_title_en: str, query_content_en: str) -> str:
    title = (query_title_en or "").strip()
    body = (query_content_en or "").strip()
    if title and body:
        return f"Title: {title}\n\nUser: {body}"
    if title:
        return f"Title: {title}"
    return body


def _json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_user_text_from_vignette(vignette_obj: Dict[str, Any], drop_image_findings: bool) -> str:
    v = copy.deepcopy(vignette_obj) if vignette_obj else {}
    if drop_image_findings and isinstance(v, dict):
        v.pop("image_findings", None)
    return "Structured vignette (JSON):\n" + _json_dumps_compact(v)


def extract_json_from_text(s: str) -> Dict[str, Any]:
    s = (s or "").strip()

    def _try_parse(x: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    direct = _try_parse(s)
    if direct:
        return direct

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


def build_output_row(
    encounter_id: str,
    author_id: str,
    image_ids: List[str],
    parsed: Dict[str, Any],
) -> Dict[str, Any]:
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
def medmo_generate(
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    system_prompt: str,
    user_text: str,
    image_paths_abs: Optional[List[str]],
    temperature: float,
    max_new_tokens: int,
) -> Tuple[str, int, int]:
    """
    Uses Qwen3-VL chat template + qwen_vl_utils.process_vision_info.
    We pass image file paths in the message content; process_vision_info turns them into tensors.
    """
    content: List[Dict[str, Any]] = []
    if image_paths_abs:
        for ap in image_paths_abs:
            content.append({"type": "image", "image": ap})
    if user_text:
        content.append({"type": "text", "text": user_text})

    # Put instructions in system role (matching your prior script behavior)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    prompt_tokens_est = int(inputs.input_ids.shape[-1]) if hasattr(inputs, "input_ids") else 0
    do_sample = temperature is not None and temperature > 0.0

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
    )

    # Trim prompt tokens (recommended in Qwen3-VL examples)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    completion_tokens_est = int(generated_ids.shape[-1] - inputs.input_ids.shape[-1])
    return output_text, prompt_tokens_est, completion_tokens_est


def _infer_dataset_path(args) -> str:
    if args.mode in ("vignette_full", "vignette_text"):
        return "/home/USER/Documents/Dermatology-Evaluation-Framework/Thesis_Final/vignettes/test_vignettes.jsonl"
   
    if args.jsonl is not None:
        return args.jsonl
    


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--jsonl", default=None, help="Path to dataset.jsonl (for multi/text/image) OR override for vignette modes")
    ap.add_argument(
        "--vignette-jsonl",
        default=DEFAULT_VIGNETTE_JSONL,
        help="Path to vignette JSONL (used when --mode vignette_full|vignette_text and --jsonl not provided)",
    )

    ap.add_argument("--images-root", default="", help="Root folder for image files (required for multi/image mode)")
    ap.add_argument("--output", required=True, help="Output JSONL path")

    ap.add_argument(
        "--mode",
        choices=["multi", "text", "vignette_full", "vignette_text", "image"],
        default="multi",
        help="multi=images+text; text=text-only; image=image-only; vignette_* use structured vignettes",
    )

    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max-images", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--start-id", type=int, default=None, help="Start encounter numeric id (expects ENCxxxxx)")
    ap.add_argument("--end-id", type=int, default=None, help="End encounter numeric id (expects ENCxxxxx)")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--attn-impl", default="flash_attention_2", help="Attention impl hint (e.g., flash_attention_2).")
    args = ap.parse_args()

    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # Load model
    model_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    # attn_implementation is optional; keep it best-effort
    if args.attn_impl:
        model_kwargs["attn_implementation"] = args.attn_impl

    def _load_medmo(model_id: str, torch_dtype, attn_impl: str):
        # Try requested attention impl first, then fall back safely.
        tried = []
        for impl in [attn_impl, "sdpa", "eager", None]:
            if impl in tried:
                continue
            tried.append(impl)
            kwargs = dict(torch_dtype=torch_dtype, device_map="auto")
            if impl is not None:
                kwargs["attn_implementation"] = impl
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
                return model, impl
            except Exception as e:
                msg = str(e).lower()
                # Only swallow errors that look like FlashAttention2 missing / unsupported
                if impl == attn_impl and ("flashattention2" in msg or "flash_attn" in msg):
                    print(f"⚠️ {attn_impl} unavailable ({e}). Falling back...")
                    continue
                # If sdpa fails due to hardware/torch, try eager; otherwise re-raise
                if impl in ("sdpa",) and ("sdpa" in msg or "scaled_dot_product_attention" in msg):
                    print(f"⚠️ SDPA unavailable ({e}). Falling back to eager...")
                    continue
                if impl is None:
                    raise
                # For other errors, don't hide them
                raise

    # ---- Load model ----
    model, used_impl = _load_medmo(args.model_id, torch_dtype=torch_dtype, attn_impl=args.attn_impl)
    print(f"✅ Loaded {args.model_id} with attn_implementation={used_impl}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    dataset_path = _infer_dataset_path(args)
    data = read_jsonl(dataset_path)
    print(data[0] if data else "No data found in JSONL.")

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

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for ex in tqdm(data, desc=f"Inferring ({args.mode})"):
            encounter_id = ex.get("encounter_id")
            author_id = args.model_id

            image_ids: List[str] = []
            image_paths_abs: Optional[List[str]] = None

            try:
                # ---- Build input depending on mode ----
                if args.mode in ("text", "multi"):
                    query_title_en = ex.get("query_title_en", "")
                    query_content_en = ex.get("query_content_en", "")
                    user_text = build_user_text_from_query(query_title_en, query_content_en)

                    if args.mode == "multi":
                        if not args.images_root:
                            raise ValueError("--images-root is required for --mode multi")
                        rels = ex.get("image_ids", []) or []
                        image_paths_abs, image_ids = load_image_paths(args.images_root, rels, args.max_images)
                    else:
                        image_paths_abs = None
                        image_ids = []

                elif args.mode == "image":
                    user_text = ""
                    if not args.images_root:
                        raise ValueError("--images-root is required for --mode image")
                    rels = ex.get("image_ids", []) or []
                    image_paths_abs, image_ids = load_image_paths(args.images_root, rels, args.max_images)

                elif args.mode in ("vignette_full", "vignette_text"):
                    vignette_obj = ex.get("vignette")
                    print(f"Debug: encounter_id={encounter_id}, vignette_obj type={type(vignette_obj)}, keys={list(vignette_obj.keys()) if isinstance(vignette_obj, dict) else 'N/A'}")
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

                    # IMPORTANT: no images in vignette modes
                    image_paths_abs = None
                    image_ids = ex.get("image_ids", []) or []

                else:
                    raise ValueError(f"Unknown mode: {args.mode}")

                gen_text, prompt_tok, completion_tok = medmo_generate(
                    model=model,
                    processor=processor,
                    system_prompt=args.system_prompt,
                    user_text=user_text,
                    image_paths_abs=image_paths_abs,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )

                parsed = extract_json_from_text(gen_text)

                row = build_output_row(
                    encounter_id=encounter_id,
                    author_id=author_id,
                    image_ids=image_ids,
                    parsed=parsed,
                )

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
python /home/USER/Documents/Dermatology-Evaluation-Framework/medmo.py \
  --jsonl /home/USER/Documents/DermAgent-Chat/dataset/test/test.jsonl \
  --images-root /home/USER/Documents/DermAgent-Chat/dataset/test/images_test \
  --output local_qwen3_vl_4b_instruct_vignette_text.jsonl \
  --mode vignette_text \
  --max-images 5 \
  --temperature 0.0 \
  --max-new-tokens 256
"""