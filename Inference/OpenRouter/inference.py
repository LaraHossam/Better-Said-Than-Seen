#!/usr/bin/env python3
import argparse
import json
import os
import time
import random
from typing import Dict, Any, List, Optional

from tqdm import tqdm
from postprocess_and_predict import run_postprocess_and_predict
from dotenv import load_dotenv

from data_loader import DermJSONLDataset
from prompts import DERM_SYSTEM_PROMPT, DERM_SYSTEM_PROMPT_IMAGE_ONLY, build_user_text
from image_utils import build_image_parts
from deepinfra_client import DeepInfraClient


OPENROUTER_MODELS = {
    "openai/gpt-4.1",
    "openai/gpt-4o",
    "qwen/qwen3-vl-8b-instruct",
    "qwen/qwen2.5-vl-32b-instruct",
    "google/gemini-3-pro-preview",
    "anthropic/claude-haiku-4.5",
    "google/gemma-3-4b-it",
    "google/gemini-2.5-flash-lite-preview-09-2025",
    "x-ai/grok-2-vision-1212",
    "black-forest-labs/flux.2-klein-4b",
    "google/gemma-3-4b-it"
}
DEEPINFRA_MODELS = {
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
}

DEFAULT_BASE = {
    "openrouter": "https://openrouter.ai/api/v1",
    "deepinfra": "https://api.deepinfra.com/v1/openai",
}


def resolve_endpoint_and_key(model_id: str):
    if model_id in OPENROUTER_MODELS:
        provider = "openrouter"
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")  # ← fallback
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_TOKEN")   # ← both names

    elif model_id in DEEPINFRA_MODELS:
        provider = "deepinfra"
        base_url = os.getenv("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai")  # ← fallback
        api_key = os.getenv("DEEPINFRA_TOKEN") or os.getenv("DEEPINFRA_API_KEY")           # ← fixed

    else:
        raise RuntimeError(
            f"Unknown model_id provider for '{model_id}'. "
            "Add it to OPENROUTER_MODELS or DEEPINFRA_MODELS."
        )

    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider '{provider}'. "
            "Set OPENROUTER_API_KEY (OpenRouter) or DEEPINFRA_TOKEN (DeepInfra)."
        )

    return base_url, api_key

def build_derangement(indices: List[int], rng: random.Random) -> List[int]:
    """
    Return a permutation p of indices such that p[i] != indices[i] for all i
    (a derangement), when possible.

    Since user says all encounters have images, we can derange encounters directly.
    """
    n = len(indices)
    if n < 2:
        raise ValueError("Need at least 2 samples to build a shuffled-image derangement.")

    p = indices[:]
    for _ in range(200):  # small bounded retry
        rng.shuffle(p)
        if all(p[i] != indices[i] for i in range(n)):
            return p

    # deterministic fix: swap adjacent until no fixed points (works for n>=2)
    p = indices[:]
    rng.shuffle(p)
    for i in range(n):
        if p[i] == indices[i]:
            j = (i + 1) % n
            p[i], p[j] = p[j]
    # verify
    if not all(p[i] != indices[i] for i in range(n)):
        raise ValueError("Failed to construct derangement; try a different seed.")
    return p


def main():
    parser = argparse.ArgumentParser(description="DermAgent DeepInfra/OpenRouter Inference")
    parser.add_argument("--jsonl", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--images-root", required=True, help="Root folder for image files")
    parser.add_argument("--model", required=True, help="Model id")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--output", required=True, help="Output directory (folder)")
    parser.add_argument("--aliases-path",  type=str)
    parser.add_argument("--start-id", type=int, default=None, help="Start range for encounter IDs (e.g., 100)")
    parser.add_argument("--end-id", type=int, default=None, help="End range for encounter IDs (e.g., 200)")

    parser.add_argument(
        "--image-mode",
        choices=[
            "per-image",
            "single-call",
            "text-only",
            "single-call-shuffled",
            "images-only",          
        ],
        default="per-image",
        help=(
            "How to send images: "
            "'per-image' (one call per image, with text), "
            "'single-call' (all images + text in one prompt), "
            "'text-only' (no images; text only), "
            "'single-call-shuffled' (all images, but from a DIFFERENT encounter, with text), "
            "'images-only' (all images; NO clinical text – image modality only)."
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=1337)
    parser.add_argument("--derm-detail-version", type=str, default="v1", help="Version string to append to output filename for DermDetail testing (e.g., 'v1', 'v2', etc.)")
    args = parser.parse_args()
    load_dotenv()

    model_id = args.model

    os.makedirs(args.output, exist_ok=True)
    base_url, api_key = resolve_endpoint_and_key(model_id)
    client = DeepInfraClient(api_key=api_key, base_url=base_url)
    dataset_obj = DermJSONLDataset(args.jsonl, args.images_root, max_images=5)

    # Materialize dataset into a list so we can derange it deterministically.
    dataset: List[Dict[str, Any]] = list(dataset_obj)

    safe_model_name = model_id.replace("/", "_")
    total_cost = 0.0
    total_time = 0.0
    processed = 0

    range_suffix = ""
    if args.start_id is not None and args.end_id is not None:
        filtered = []
        for ex in dataset:
            try:
                # enc_num = int(ex["encounter_id"].replace("ENC", ""))
                # if args.start_id + 907 <= enc_num <= args.end_id + 907:
                filtered.append(ex)
            except (KeyError, ValueError):
                continue
        dataset = filtered
        print(
            f"📦 Filtered dataset to {len(dataset)} samples "
            f"(ENC{args.start_id:05d}–ENC{args.end_id:05d})"
        )
        range_suffix = f"_ENC{args.start_id:05d}-ENC{args.end_id:05d}"

    if args.image_mode == "single-call-shuffled":
        if len(dataset) < 2:
            raise ValueError("single-call-shuffled requires at least 2 samples.")
        rng = random.Random(args.shuffle_seed)
        idxs = list(range(len(dataset)))
        donor_perm = build_derangement(idxs, rng)  # donor_perm[i] is donor index for receiver i
        print(f"🔀 Shuffled image mode enabled (seed={args.shuffle_seed}).")

    if args.derm_detail_version:
        output_file = os.path.join(args.output, f"{safe_model_name}_DermDetail_{args.derm_detail_version}{range_suffix}.jsonl")
    else:
        output_file = os.path.join(args.output, f"{safe_model_name}_{range_suffix}_{args.image_mode}.jsonl")

    print(f"📁 Saving results to: {output_file}")

    CHUNK_SIZE = 10
    buffer = []

    with open(output_file, "w", encoding="utf-8") as out_f:
        for i, ex in enumerate(tqdm(dataset, desc="Inferring")):
            encounter_id = ex["encounter_id"]

            # ---------- Build user text depending on mode ----------
            if args.image_mode == "images-only":
                # IMAGE-ONLY: No clinical notes at all, to isolate vision signal
                user_text = "Task: Diagnose the condition using ONLY the images. Return ONLY the JSON object."
            else:
                user_text = build_user_text(ex["query_title_en"], ex["query_content_en"])

            augmented_text = user_text

            # Choose which images to use based on mode
            shuffled_from_encounter_id: Optional[str] = None
            if args.image_mode == "single-call-shuffled":
                donor_ex = dataset[donor_perm[i]]
                shuffled_from_encounter_id = donor_ex["encounter_id"]
                imgs = donor_ex.get("image_paths", []) or []
            else:
                imgs = ex.get("image_paths", []) or []

            start_time = time.time()

            # Build image parts only when using images
            image_parts = None
            if args.image_mode in ("per-image", "single-call", "single-call-shuffled", "images-only"):
                image_parts = build_image_parts(imgs)
                if not image_parts:
                    raise ValueError(
                        f"No images available for encounter ({encounter_id}) under image_mode={args.image_mode}"
                    )

            if args.image_mode == "per-image":
                # ---------- PER-IMAGE MODE ----------
                per_image_results = []
                sample_prompt_tokens = 0
                sample_completion_tokens = 0
                sample_cost = 0.0

                for j, img_part in enumerate(image_parts):
                    try:
                        resp = client.chat_multimodal(
                            model=model_id,
                            system_prompt=DERM_SYSTEM_PROMPT,
                            image_parts=[img_part],
                            user_text=augmented_text,
                            temperature=args.temperature,
                            max_output_tokens=args.max_output_tokens,
                            mock=False,
                            print_bool=False,
                        )

                        data_j = DeepInfraClient.extract_json(resp.choices[0].message.content)
                        usage_j = getattr(resp, "usage", None)

                        est_cost_j = 0.0
                        if usage_j and hasattr(usage_j, "estimated_cost"):
                            est_cost_j = usage_j.estimated_cost or 0.0
                            sample_cost += est_cost_j
                            total_cost += est_cost_j

                        if usage_j and hasattr(usage_j, "prompt_tokens"):
                            sample_prompt_tokens += usage_j.prompt_tokens or 0
                        if usage_j and hasattr(usage_j, "completion_tokens"):
                            sample_completion_tokens += usage_j.completion_tokens or 0

                        per_image_results.append({
                            "image_index": j,
                            "image_id": imgs[j] if j < len(imgs) else None,
                            "diagnosis": data_j.get("diagnosis"),
                            "confidence": data_j.get("confidence"),
                            "reasoning": data_j.get("reasoning"),
                            "estimated_cost": est_cost_j,
                            "prompt_tokens": getattr(usage_j, "prompt_tokens", None),
                            "completion_tokens": getattr(usage_j, "completion_tokens", None),
                        })

                    except Exception as e_img:
                        per_image_results.append({
                            "image_index": j,
                            "image_id": imgs[j] if j < len(imgs) else None,
                            "error": str(e_img),
                        })

                def _to_float(x):
                    try:
                        return float(x)
                    except (TypeError, ValueError):
                        return None

                best = None
                for r in per_image_results:
                    c = _to_float(r.get("confidence"))
                    if c is not None and (best is None or c > _to_float(best.get("confidence"))):
                        best = r
                if best is None and per_image_results:
                    best = per_image_results[0]

                elapsed = time.time() - start_time
                total_time += elapsed

                row = {
                    "encounter_id": encounter_id,
                    "model_id": model_id,

                    "image_ids": imgs,
                    "shuffled_images": False,
                    "shuffled_from_encounter_id": None,

                    "diagnosis": best.get("diagnosis") if best else None,
                    "confidence": best.get("confidence") if best else None,
                    "reasoning": best.get("reasoning") if best else None,

                    "per_image": per_image_results,

                    "prompt_tokens": sample_prompt_tokens or None,
                    "completion_tokens": sample_completion_tokens or None,
                    "estimated_cost": sample_cost,
                    "time_seconds": round(elapsed, 2),
                }

            elif args.image_mode in ("single-call", "single-call-shuffled", "images-only"):
                # ---------- SINGLE-CALL FAMILY (NORMAL, SHUFFLED, or IMAGES-ONLY) ----------
                try:
                    resp = client.chat_multimodal(
                        model=model_id,
                        system_prompt = (
                            DERM_SYSTEM_PROMPT_IMAGE_ONLY
                            if args.image_mode == "images-only"
                            else DERM_SYSTEM_PROMPT
                        ),
                        image_parts=image_parts,
                        user_text=augmented_text,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                        mock=False,
                        print_bool=False,
                    )

                    data = DeepInfraClient.extract_json(resp.choices[0].message.content)
                    usage = getattr(resp, "usage", None)

                    sample_prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                    sample_completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

                    sample_cost = 0.0
                    if usage and hasattr(usage, "estimated_cost") and usage.estimated_cost:
                        sample_cost = usage.estimated_cost
                        total_cost += sample_cost

                    elapsed = time.time() - start_time
                    total_time += elapsed

                    per_image_results = data.get("per_image")

                    row = {
                        "encounter_id": encounter_id,
                        "model_id": model_id,

                        "image_ids": imgs,
                        "shuffled_images": (args.image_mode == "single-call-shuffled"),
                        "shuffled_from_encounter_id": shuffled_from_encounter_id,

                        "diagnosis": data.get("diagnosis"),
                        "confidence": data.get("confidence"),
                        "reasoning": data.get("reasoning"),

                        "per_image": per_image_results,

                        "prompt_tokens": sample_prompt_tokens,
                        "completion_tokens": sample_completion_tokens,
                        "estimated_cost": sample_cost,
                        "time_seconds": round(elapsed, 2),
                    }

                except Exception as e:
                    elapsed = time.time() - start_time
                    total_time += elapsed
                    row = {
                        "encounter_id": encounter_id,
                        "model_id": model_id,
                        "image_ids": imgs,
                        "shuffled_images": (args.image_mode == "single-call-shuffled"),
                        "shuffled_from_encounter_id": shuffled_from_encounter_id,
                        "error": str(e),
                        "time_seconds": round(elapsed, 2),
                    }

            elif args.image_mode == "text-only":
                # ---------- TEXT-ONLY MODE ----------
                try:
                    resp = client.chat_multimodal(
                        model=model_id,
                        system_prompt=DERM_SYSTEM_PROMPT,
                        image_parts=[],
                        user_text=augmented_text,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                        mock=False,
                        print_bool=False,
                    )

                    data = DeepInfraClient.extract_json(resp.choices[0].message.content)
                    usage = getattr(resp, "usage", None)

                    sample_prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
                    sample_completion_tokens = getattr(usage, "completion_tokens", None) if usage else None

                    sample_cost = 0.0
                    if usage and hasattr(usage, "estimated_cost") and usage.estimated_cost:
                        sample_cost = usage.estimated_cost
                        total_cost += sample_cost

                    elapsed = time.time() - start_time
                    total_time += elapsed

                    row = {
                        "encounter_id": encounter_id,
                        "model_id": model_id,

                        "image_ids": [],
                        "shuffled_images": False,
                        "shuffled_from_encounter_id": None,

                        "diagnosis": data.get("diagnosis"),
                        "confidence": data.get("confidence"),
                        "reasoning": data.get("reasoning"),

                        "per_image": None,

                        "prompt_tokens": sample_prompt_tokens,
                        "completion_tokens": sample_completion_tokens,
                        "estimated_cost": sample_cost,
                        "time_seconds": round(elapsed, 2),
                    }

                except Exception as e:
                    elapsed = time.time() - start_time
                    total_time += elapsed
                    row = {
                        "encounter_id": encounter_id,
                        "model_id": model_id,
                        "image_ids": [],
                        "shuffled_images": False,
                        "shuffled_from_encounter_id": None,
                        "error": str(e),
                        "time_seconds": round(elapsed, 2),
                    }

            buffer.append(row)
            processed += 1

            if len(buffer) >= CHUNK_SIZE:
                out_f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in buffer) + "\n")
                out_f.flush()
                try:
                    os.fsync(out_f.fileno())
                except OSError:
                    pass
                buffer.clear()

        if buffer:
            out_f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in buffer) + "\n")
            out_f.flush()
            try:
                os.fsync(out_f.fileno())
            except OSError:
                pass

    print(f"✅ Inference complete. Processed {processed} samples.")
    run_postprocess_and_predict(output_file)
if __name__ == "__main__":
    main()
