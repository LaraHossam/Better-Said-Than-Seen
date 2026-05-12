#!/usr/bin/env python3


from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
import csv
from pathlib import Path


DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_VIGNETTE_JSONL = (
    "/home/lara.hassan/Documents/Dermatology-Evaluation-Framework/"
    "Thesis_Final/vignettes/test_vignettes.jsonl"
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a board-certified dermatologist. "
    "Given the patient's text history (and images if provided), respond ONLY in valid JSON with keys: "
    '{"diagnosis": string, "confidence": number (0-1), "reasoning": string}. '
    "Do not output any extra text outside the JSON."
)
DEFAULT_SYSTEM_PROMPT_IMAGES_ONLY = (
    "You are a board-certified dermatologist. "
    "Given the patient's provided images, respond ONLY in valid JSON with keys: "
    '{"diagnosis": string, "confidence": number (0-1), "reasoning": string}. '
    "Do not output any extra text outside the JSON."
)

NUM_LAYERS = 36
NUM_HEADS = 16 
GLOBAL_LAYER_INDICES = [2, 5, 11, 17, 23, 27, 35]

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



def load_images(
    images_root: str, image_paths: List[str], max_images: int,
    max_size: int = 512,  
) -> Tuple[List[Image.Image], List[str]]:
    selected = (image_paths or [])[:max_images]
    pil_images: List[Image.Image] = []
    image_ids: List[str] = []
    for p in selected:
        ap = resolve_image_path(images_root, p)
        try:
            img = Image.open(ap).convert("RGB")
            # Resize so longest side <= max_size
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            pil_images.append(img)
            image_ids.append(p)
        except Exception as e:
            print(f"    WARNING: Skipping image {ap} — {e}")
    return pil_images, image_ids

def build_user_text_from_query(
    query_title_en: str, query_content_en: str
) -> str:
    title = (query_title_en or "").strip()
    body = (query_content_en or "").strip()
    if title and body:
        return f"Title: {title}\n\nUser: {body}"
    if title:
        return f"Title: {title}"
    return body


def _json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_user_text_from_vignette(
    vignette_obj: Dict[str, Any], drop_image_findings: bool
) -> str:
    v = copy.deepcopy(vignette_obj) if vignette_obj else {}
    if drop_image_findings and isinstance(v, dict):
        v.pop("image_findings", None)
    return "Structured vignette (JSON):\n" + _json_dumps_compact(v)


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


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

def find_token_segments(
    input_ids: torch.Tensor, processor, model
) -> Dict[str, Any]:
    ids = input_ids[0].tolist()
    tokens = [processor.tokenizer.decode([t]) for t in ids]

    # Try standard attribute first
    image_token_id = getattr(processor, 'image_token_id', None)
    
    # Qwen3-VL specific fallback
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_token_id == processor.tokenizer.unk_token_id:
            image_token_id = None
    
    # MedGemma fallback
    if image_token_id is None:
        image_token_id = getattr(model.config, 'image_token_index', None)
    
    # Last resort: frequency heuristic (your existing code)
    if image_token_id is None:
        mid_region = ids[10: len(ids) - 20]
        if mid_region:
            counts = Counter(mid_region)
            candidate_id, count = counts.most_common(1)[0]
            if count > 50:
                image_token_id = candidate_id

    image_positions = []
    text_positions = []
    for i, tid in enumerate(ids):
        if image_token_id is not None and tid == image_token_id:
            image_positions.append(i)
        else:
            text_positions.append(i)

    return {
        'image': image_positions,
        'text': text_positions,
        'tokens': tokens,
        'token_ids': ids,
    }


def identify_sink_tokens(
    prefill_attentions: List[torch.Tensor],
    segments: Dict[str, Any],
    threshold_multiplier: float = 3.0,
) -> List[int]:
    """
    Identify sink tokens from prefill attention matrices.
    A sink receives > threshold_multiplier × median attention at ALL global layers.
    """
    text_set = set(segments['text'])
    seq_len = prefill_attentions[0].shape[-1]

    per_layer_received = []
    for layer_idx in GLOBAL_LAYER_INDICES:
        if layer_idx >= len(prefill_attentions):
            continue
        attn = prefill_attentions[layer_idx][0].float()
        received = attn.mean(dim=0).sum(dim=0).cpu().numpy()
        per_layer_received.append(received)

    if not per_layer_received:
        return []

    per_layer_received = np.array(per_layer_received)

    sink_candidates = None
    for gi in range(len(per_layer_received)):
        layer_scores = per_layer_received[gi]
        text_scores = [layer_scores[p] for p in text_set if p < seq_len]
        if not text_scores:
            return []
        median_score = np.median(text_scores)
        threshold = median_score * threshold_multiplier

        above = {p for p in text_set if p < seq_len and layer_scores[p] > threshold}
        if sink_candidates is None:
            sink_candidates = above
        else:
            sink_candidates = sink_candidates & above

    return sorted(sink_candidates) if sink_candidates else []


# ─── Generation with Attention Extraction ───────────────────────────────────


# @torch.inference_mode()
# def generate_with_attention(
#     model,
#     processor,
#     system_prompt: str,
#     user_text: str,
#     images: Optional[List[Image.Image]],
#     max_new_tokens: int,
#     device: str,
# ) -> Dict[str, Any]:
#     """
#     Generate token-by-token withouttt KV cache, extracting attention at each step.

#     At each step, the full sequence is reprocessed from scratch.
#     All sequence-length tensors (input_ids, attention_mask, token_type_ids,
#     cache_position) are grown together to stay synchronized.
#     """

#     user_content: List[Dict[str, Any]] = []
#     if images:
#         for img in images:
#             user_content.append({"type": "image", "image": img})
#     user_content.append({"type": "text", "text": user_text})

#     messages = [
#         {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
#         {"role": "user", "content": user_content},
#     ]

#     inputs = processor.apply_chat_template(
#         messages,
#         add_generation_prompt=True,
#         tokenize=True,
#         return_dict=True,
#         return_tensors="pt",
#     ).to(device)

#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]
#     prompt_length = input_ids.shape[1]

#     # Identify image vs text positions in the prompt
#     segments = find_token_segments(input_ids, processor, model)

#     # ── Separate inputs into: sequence-length-dependent vs static ──
#     token_type_ids = inputs.get("token_type_ids", None)

#     # These tensors are static (pixel_values = image data, doesn't change):
#     static_kwargs = {}
#     for k, v in inputs.items():
#         if k not in ("input_ids", "attention_mask", "token_type_ids"):
#             static_kwargs[k] = v

#     print(f"    Prompt: {prompt_length} tokens | "
#           f"Image tokens: {len(segments['image'])} | "
#           f"Text tokens: {len(segments['text'])}")
#     extra_keys = [k for k in inputs.keys() if k not in ("input_ids", "attention_mask")]
#     print(f"    Extra input keys: {extra_keys}")
#     if token_type_ids is not None:
#         print(f"    token_type_ids shape: {token_type_ids.shape}, "
#               f"unique values: {token_type_ids.unique().tolist()}")

#     eos_id = processor.tokenizer.eos_token_id
#     stop_ids = {eos_id}
#     # Try to find end_of_turn token ID
#     end_of_turn_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
#     if isinstance(end_of_turn_id, int) and end_of_turn_id != processor.tokenizer.unk_token_id:
#         stop_ids.add(end_of_turn_id)


#     print(f"    Stop token IDs: {stop_ids}")
#     decode_attentions = []
#     generated_ids = []
#     token_type_ids = None  
#     # ── 1) PREFILL ──
#     outputs = model(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         cache_position=torch.arange(prompt_length, device=device),
#         use_cache=True,          # ← enable
#         output_attentions=True,
#         **static_kwargs,
#     )

#     prefill_attentions = [a.cpu().float() for a in outputs.attentions]
#     past_key_values = outputs.past_key_values   # ← save cache
#     next_id = outputs.logits[:, -1, :].argmax(dim=-1)
    
#     VISUAL_KEYS_TO_DROP = {
#         "pixel_values",
#         "image_grid_thw", 
#         "pixel_values_videos",
#         "video_grid_thw",
#         "second_per_grid_ts"
#     }

#     decode_static_kwargs = {
#         k: v for k, v in static_kwargs.items() 
#         if k not in VISUAL_KEYS_TO_DROP
#     }
#     print("decode_static_kwargs keys:", list(decode_static_kwargs.keys()))
#     print("rope_deltas" in decode_static_kwargs)

#     first_step_attn = [a[:, :, -1, :].cpu().float() for a in prefill_attentions]
#     decode_attentions.append(first_step_attn)
#     generated_ids.append(next_id.item())


#     # ── step 2: DECODE (full reprocessing, no cache) ──
#     cur_attention_mask = attention_mask  # will grow each step

#     for step in range(1, max_new_tokens):
#         if next_id.item() in stop_ids:
#             break

#         cur_attention_mask = torch.cat([
#             cur_attention_mask,
#             torch.ones((1, 1), device=device, dtype=cur_attention_mask.dtype)
#         ], dim=1)

#         cur_seq_len = prompt_length + step  # position of the new token

#         outputs = model(
#             input_ids=next_id.view(1, 1),          # ← only the new token
#             attention_mask=cur_attention_mask,      # ← full grown mask
#             past_key_values=past_key_values,        # ← pass cache in
#             cache_position=torch.tensor([cur_seq_len - 1], device=device),
#             use_cache=True,
#             output_attentions=True,
#             **decode_static_kwargs,
#         )

#         past_key_values = outputs.past_key_values   # ← update cache
#         next_id = outputs.logits[:, -1, :].argmax(dim=-1)

#         step_attn = [a[:, :, -1, :].cpu().float() for a in outputs.attentions]
#         decode_attentions.append(step_attn)
#         generated_ids.append(next_id.item())

#         del outputs
#         torch.cuda.empty_cache()

#         if step % 10 == 0:
#             partial = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
#             print(f"      Step {step}/{max_new_tokens}: ...{partial[-60:]}")

#     generated_text = processor.tokenizer.decode( [t for t in generated_ids if t not in stop_ids], skip_special_tokens=True)
#     generated_tokens = [processor.tokenizer.decode([t]) for t in generated_ids]

#     print(f"    Generated {len(generated_ids)} tokens")

#     return {
#         'generated_text': generated_text,
#         'generated_ids': generated_ids,
#         'generated_tokens': generated_tokens,
#         'prefill_attentions': prefill_attentions,
#         'decode_attentions': decode_attentions,
#         'segments': segments,
#         'prompt_length': prompt_length,
#     }


@torch.inference_mode()
def generate_with_attention(
    model,
    processor,
    system_prompt: str,
    user_text: str,
    images: Optional[List[Image.Image]],
    max_new_tokens: int,
    device: str,
) -> Dict[str, Any]:
    """
    Generate token-by-token withouttt KV cache, extracting attention at each step.

    At each step, the full sequence is reprocessed from scratch.
    All sequence-length tensors (input_ids, attention_mask, token_type_ids,
    cache_position) are grown together to stay synchronized.
    """

    user_content: List[Dict[str, Any]] = []
    if images:
        for img in images:
            user_content.append({"type": "image", "image": img})
    user_content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_length = input_ids.shape[1]

    # Identify image vs text positions in the prompt
    segments = find_token_segments(input_ids, processor, model)

    # ── Separate inputs into: sequence-length-dependent vs static ──
    token_type_ids = inputs.get("token_type_ids", None)

    # These tensors are static (pixel_values = image data, doesn't change):
    static_kwargs = {}
    for k, v in inputs.items():
        if k not in ("input_ids", "attention_mask", "token_type_ids"):
            static_kwargs[k] = v

    print(f"    Prompt: {prompt_length} tokens | "
          f"Image tokens: {len(segments['image'])} | "
          f"Text tokens: {len(segments['text'])}")
    extra_keys = [k for k in inputs.keys() if k not in ("input_ids", "attention_mask")]
    print(f"    Extra input keys: {extra_keys}")
    if token_type_ids is not None:
        print(f"    token_type_ids shape: {token_type_ids.shape}, "
              f"unique values: {token_type_ids.unique().tolist()}")

    eos_id = processor.tokenizer.eos_token_id
    stop_ids = {eos_id}
    # Try to find end_of_turn token ID
    end_of_turn_id = processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn_id, int) and end_of_turn_id != processor.tokenizer.unk_token_id:
        stop_ids.add(end_of_turn_id)
    print(f"    Stop token IDs: {stop_ids}")

    def forward_pass(cur_ids, cur_mask, cur_ttids):
        """Run a full forward pass with all sequence-length tensors consistent."""
        seq_len = cur_ids.shape[1]
        kwargs = dict(
            input_ids=cur_ids,
            attention_mask=cur_mask,
            cache_position=torch.arange(seq_len, device=device),
            use_cache=False,
            output_attentions=True,
            **static_kwargs,
        )
        if cur_ttids is not None:
            kwargs["token_type_ids"] = cur_ttids
        return model(**kwargs)

    # ── 1) PREFILL ──
    outputs = forward_pass(input_ids, attention_mask, token_type_ids)

    prefill_attentions = [a.cpu().float() for a in outputs.attentions]

    # First generated token's attention = last row of prefill matrices
    first_step_attn = [a[:, :, -1, :].cpu().float() for a in outputs.attentions]

    # First predicted token
    next_id = outputs.logits[:, -1, :].argmax(dim=-1)  # shape (1,)

    decode_attentions = [first_step_attn]
    generated_ids = [next_id.item()]

    del outputs
    torch.cuda.empty_cache()

    # ── step 2: DECODE (full reprocessing, no cache) ──
    cur_input_ids = input_ids                
    cur_attention_mask = attention_mask     
    cur_token_type_ids = token_type_ids    

    for step in range(1, max_new_tokens):
        if next_id.item() in stop_ids:
            break

        # ── Grow ALL sequence-length tensors by 1 ──
        cur_input_ids = torch.cat(
            [cur_input_ids, next_id.view(1, 1)], dim=1
        )
        cur_attention_mask = torch.cat(
            [cur_attention_mask,
             torch.ones((1, 1), device=device, dtype=cur_attention_mask.dtype)],
            dim=1,
        )
        if cur_token_type_ids is not None:
            # Generated tokens are TEXT → token_type = 0
            cur_token_type_ids = torch.cat(
                [cur_token_type_ids,
                 torch.zeros((1, 1), device=device, dtype=cur_token_type_ids.dtype)],
                dim=1,
            )

        #  Forward pass 
        outputs = forward_pass(cur_input_ids, cur_attention_mask, cur_token_type_ids)

        next_id = outputs.logits[:, -1, :].argmax(dim=-1)

        # Last row = new token's attention to all previous positions
        step_attn = [a[:, :, -1, :].cpu().float() for a in outputs.attentions]
        decode_attentions.append(step_attn)
        generated_ids.append(next_id.item())

        del outputs
        torch.cuda.empty_cache()

        if step % 10 == 0:
            partial = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(f"      Step {step}/{max_new_tokens}: ...{partial[-60:]}")

    generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = [processor.tokenizer.decode([t]) for t in generated_ids]
    print(f"    Generated {len(generated_ids)} tokens")

    return {
        'generated_text': generated_text,
        'generated_ids': generated_ids,
        'generated_tokens': generated_tokens,
        'prefill_attentions': prefill_attentions,
        'decode_attentions': decode_attentions,
        'segments': segments,
        'prompt_length': prompt_length,
    }
# ─── Attention Ratio Computation ────────────────────────────────────────────


def compute_three_way_ratios(
    decode_attentions: List[List[torch.Tensor]],
    segments: Dict[str, Any],
    sink_positions: List[int],
    prompt_length: int,
) -> np.ndarray:
    """
    Compute image/text/sink attention ratios for every generated token
    at every global layer. Only counts attention to PROMPT positions.
    """
    image_set = set(segments['image'])
    sink_set = set(sink_positions)
    text_set = set(segments['text']) - sink_set

    num_steps = len(decode_attentions)
    num_global = len(GLOBAL_LAYER_INDICES)
    ratios = np.zeros((num_steps, num_global, 3))

    for step in range(num_steps):
        step_layers = decode_attentions[step]

        # Find seq_len from the first non-None layer
        seq_len = None
        for sl in step_layers:
            if sl is not None:
                seq_len = sl.shape[-1]
                break
        if seq_len is None:
            continue

        for gi, layer_idx in enumerate(GLOBAL_LAYER_INDICES):
            if layer_idx >= len(step_layers):
                continue
            if step_layers[layer_idx] is None:
                continue

            attn = step_layers[layer_idx][0].mean(dim=0).numpy()  # (seq_len,)

            img_attn = sum(attn[p] for p in image_set if p < seq_len)
            txt_attn = sum(attn[p] for p in text_set if p < seq_len)
            snk_attn = sum(attn[p] for p in sink_set if p < seq_len)

            total = img_attn + txt_attn + snk_attn
            if total > 1e-9:
                ratios[step, gi, 0] = img_attn / total
                ratios[step, gi, 1] = txt_attn / total
                ratios[step, gi, 2] = snk_attn / total

    return ratios


# ─── Visualization ──────────────────────────────────────────────────────────


def plot_full_generation_heatmap(
    ratios: np.ndarray,
    generated_tokens: List[str],
    encounter_id: str,
    mode: str,
    sink_token_strs: List[str],
    output_path: str,
):
    """
    Three side-by-side heatmaps: Image / Text / Sink attention.
    Rows = generated tokens, Columns = 4 global layers.
    """
    n_tokens = ratios.shape[0]
    n_layers = ratios.shape[1]

    labels = []
    for t in generated_tokens:
        t_clean = t.strip()
        if not t_clean:
            t_clean = "·"
        if len(t_clean) > 12:
            t_clean = t_clean[:12] + "…"
        labels.append(t_clean)

    fig_height = max(5, n_tokens * 0.28)
    fig, axes = plt.subplots(1, 3, figsize=(16, fig_height))

    titles = ['Image Attention', 'Text Attention', 'Sink Attention']
    cmaps = ['Blues', 'Greens', 'Oranges']

    for i, (title, cmap) in enumerate(zip(titles, cmaps)):
        ax = axes[i]
        data = ratios[:, :, i]

        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                        interpolation='nearest')
        ax.set_xticks(range(n_layers))
        ax.set_xticklabels([f'L{l}' for l in GLOBAL_LAYER_INDICES], fontsize=9)
        ax.set_xlabel('Decoder Layer', fontsize=10)

        if i == 0:
            ax.set_yticks(range(n_tokens))
            ax.set_yticklabels(labels, fontsize=7, fontfamily='monospace')
            ax.set_ylabel('Generated Token (in order)', fontsize=10)
        else:
            ax.set_yticks([])

        ax.set_title(title, fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.6, label='Ratio')

        for yi in range(n_tokens):
            for xi in range(n_layers):
                val = data[yi, xi]
                if val > 0.03:
                    color = 'white' if val > 0.5 else 'black'
                    ax.text(xi, yi, f'{val:.2f}', ha='center', va='center',
                            fontsize=5.5, color=color)

    sink_display = ", ".join(repr(s) for s in sink_token_strs[:5])
    fig.suptitle(
        f'Attention Flow — {encounter_id} — Mode: {mode}\n'
        f'Sink tokens: [{sink_display}]',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Heatmap saved: {output_path}")


# ─── Main Pipeline ──────────────────────────────────────────────────────────


def _infer_dataset_path(args) -> str:
    if args.jsonl is not None:
        return args.jsonl
    if args.mode in ("vignette_full", "vignette_text"):
        return args.vignette_jsonl
    raise ValueError("No --jsonl provided and mode is not a vignette mode.")


def main():
    ap = argparse.ArgumentParser(
        description="MedGemma inference with attention heatmap visualization"
    )
    ap.add_argument("--jsonl", default=None)
    ap.add_argument("--vignette-jsonl", default=DEFAULT_VIGNETTE_JSONL)
    ap.add_argument("--images-root", default="")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--output_attn", required=True, help="Output JSONL attn path")
    ap.add_argument("--version", required=True, help="version")

    # ap.add_argument("--heatmap-dir", required=True, help="Directory to save attention heatmap PNGs")
    ap.add_argument(
        "--mode",
        choices=["multi", "text", "vignette_full", "vignette_text", "image-only"],
        default="multi",
    )
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max-images", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--start-id", type=int, default=None)
    ap.add_argument("--end-id", type=int, default=None)
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = ap.parse_args()

    # os.makedirs(args.heatmap_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]

    print(f"Loading {args.model_id} with attn_implementation='eager'...")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    print("Model loaded.")

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
        print(f"Filtered to {len(data)} samples")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)



    layer_csv_dir = Path(args.output_attn)
    layer_csv_dir.mkdir(parents=True, exist_ok=True)

    layer_csv_paths = {l: layer_csv_dir / f"layer_{l}.csv" for l in GLOBAL_LAYER_INDICES}

    # Create each CSV only if it doesn't already exist
    for l, p in layer_csv_paths.items():
        if not p.exists():
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["encounter_id", "version", "image_avg", "text_avg", "sink_avg"])

    with open(args.output, "w", encoding="utf-8") as out_f:
        for ex in tqdm(data, desc=f"Inferring ({args.mode}) + Attention"):
            encounter_id = ex.get("encounter_id", "unknown")
            t0 = time.time()

            row: Dict[str, Any] = {
                "encounter_id": encounter_id,
                "model_id": args.model_id,
                "mode": args.mode,
            }

            try:
                images: Optional[List[Image.Image]] = None
                image_ids: List[str] = []
                system_prompt = args.system_prompt

                if args.mode in ("text", "multi"):
                    user_text = build_user_text_from_query(
                        ex.get("query_title_en", ""),
                        ex.get("query_content_en", ""),
                    )
                    if args.mode == "multi":
                        image_paths = ex.get("image_ids", []) or []
                        images, image_ids = load_images(
                            args.images_root, image_paths, args.max_images
                        )

                elif args.mode == "image-only":
                    system_prompt = DEFAULT_SYSTEM_PROMPT_IMAGES_ONLY
                    user_text = ""
                    image_paths = ex.get("image_ids", []) or []
                    images, image_ids = load_images(
                        args.images_root, image_paths, args.max_images
                    )

                elif args.mode in ("vignette_full", "vignette_text"):
                    vignette_obj = ex.get("vignette")
                    if vignette_obj is None:
                        raw = ex.get("vignette_raw_text")
                        if isinstance(raw, str):
                            vignette_obj = json.loads(raw)
                    if not isinstance(vignette_obj, dict):
                        raise ValueError("Missing or invalid vignette object.")
                    drop_img = (args.mode == "vignette_text")
                    user_text = build_user_text_from_vignette(vignette_obj, drop_img)
                    images = None
                else:
                    raise ValueError(f"Unknown mode: {args.mode}")

                print(f"\n  Processing {encounter_id}...")
                gen_result = generate_with_attention(
                    model=model,
                    processor=processor,
                    system_prompt=system_prompt,
                    user_text=user_text,
                    images=images,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )

                sink_positions = identify_sink_tokens(
                    gen_result['prefill_attentions'],
                    gen_result['segments'],
                )
                sink_strs = [
                    gen_result['segments']['tokens'][p]
                    for p in sink_positions
                    if p < len(gen_result['segments']['tokens'])
                ]
                print(f"    Sink tokens ({len(sink_positions)}): "
                      f"{[repr(s) for s in sink_strs[:5]]}")

                ratios = compute_three_way_ratios(
                    gen_result['decode_attentions'],
                    gen_result['segments'],
                    sink_positions,
                    gen_result['prompt_length'],
                )

                # heatmap_path = os.path.join(
                #     args.heatmap_dir, f"{encounter_id}_{args.mode}_heatmap.png"
                # )

                print(f"\n    Attention Ratios (averaged over {ratios.shape[0]} generated tokens):")
                for gi, l in enumerate(GLOBAL_LAYER_INDICES):
                    img_avg = float(ratios[:, gi, 0].mean())
                    txt_avg = float(ratios[:, gi, 1].mean())
                    snk_avg = float(ratios[:, gi, 2].mean())

                    csv_path = layer_csv_paths[l]
                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow([encounter_id, args.version, img_avg, txt_avg, snk_avg])

                # plot_full_generation_heatmap(
                #     ratios=ratios,
                #     generated_tokens=gen_result['generated_tokens'],
                #     encounter_id=encounter_id,
                #     mode=args.mode,
                #     sink_token_strs=sink_strs,
                #     output_path=heatmap_path,
                # )

                # With this — save full per-token attention data
                attn_csv_path = layer_csv_dir / "per_token_attention.csv"
                if not attn_csv_path.exists():
                    with open(attn_csv_path, "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["encounter_id", "version", "token_idx", "token_str", 
                                    "layer", "image_ratio", "text_ratio", "sink_ratio"])

                with open(attn_csv_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for step in range(ratios.shape[0]):
                        tok_str = gen_result['generated_tokens'][step] if step < len(gen_result['generated_tokens']) else ""
                        for gi, l in enumerate(GLOBAL_LAYER_INDICES):
                            w.writerow([
                                encounter_id,
                                args.version,
                                step,
                                tok_str.strip(),
                                l,
                                float(ratios[step, gi, 0]),
                                float(ratios[step, gi, 1]),
                                float(ratios[step, gi, 2]),
                            ])
                            

                parsed = extract_json_from_text(gen_result['generated_text'])

                avg_ratios = {
                    f"layer_{l}": {
                        "image": float(ratios[:, gi, 0].mean()),
                        "text": float(ratios[:, gi, 1].mean()),
                        "sink": float(ratios[:, gi, 2].mean()),
                    }
                    for gi, l in enumerate(GLOBAL_LAYER_INDICES)
                }

                row.update({
                    "image_ids": image_ids,
                    "diagnosis": parsed.get("diagnosis"),
                    "confidence": safe_float(parsed.get("confidence")),
                    "reasoning": parsed.get("reasoning"),
                    "raw_output": gen_result['generated_text'],
                    "prompt_tokens": gen_result['prompt_length'],
                    "completion_tokens": len(gen_result['generated_ids']),
                    "num_image_tokens": len(gen_result['segments']['image']),
                    "sink_tokens": sink_strs,
                    "sink_positions": sink_positions,
                    "attention_ratios_avg": avg_ratios,
                    # "heatmap_path": heatmap_path,
                    "time_seconds": round(time.time() - t0, 2),
                })

                del gen_result['prefill_attentions']
                del gen_result['decode_attentions']
                torch.cuda.empty_cache()

            except Exception as e:
                import traceback
                traceback.print_exc()
                row.update({
                    "error": str(e),
                    "time_seconds": round(time.time() - t0, 2),
                })

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\nDone. Results: {args.output}")
    print(f"   Heatmaps: {args.heatmap_dir}/")


if __name__ == "__main__":
    main()


"""
python medmo_pertoken_attn.py \
  --jsonl train_2.jsonl.v1.jsonl \
  --images-root images_train \
  --output qwen_train_results_v1.jsonl \
  --output_attn qwen_train_results_attn.jsonl \
  --version v1 \
  --heatmap-dir ./heatmaps_multi \
  --mode multi
"""