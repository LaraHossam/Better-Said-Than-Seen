#!/usr/bin/env python3
"""
vLLM inference over vignette JSONL (resume-safe, append-only) using:
  google/medgemma-27b-it (Hugging Face)

Input JSONL: one encounter per line. Each line should contain:
- encounter_id (string)
- vignette (dict) OR vignette_raw_text (string) OR vignette_text (string/dict)

Output JSONL (append-only):
<model_name>_vllm_inference_vignettes.jsonl
- Preserves all original fields
- Adds/overwrites "responses" with ONE model response:
  [{"author_id": <model>, "content_en": <diagnosis>}]

Install:
  pip install "vllm>=0.6.0" transformers tqdm

Example:
  python /home/lara.hassan/Documents/Dermatology-Evaluation-Framework/Thesis_Final/hf_vignette_inference.py \
    --input-jsonl /home/lara.hassan/Documents/Dermatology-Evaluation-Framework/Thesis_Final/vignettes/test_vignettes.jsonl \
    --model microsoft/MediPhi-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --temperature 0.0 \
    --max-tokens 220
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from tqdm.auto import tqdm

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# =========================
# YOU provide this prompt
# =========================
SYSTEM_PROMPT = """You are a board-certified dermatologist.

You will be given a structured dermatology case vignette derived from:
- the patient’s reported history, and
- visible findings from one or more clinical photographs.

TASK
Based ONLY on the information in the vignette, respond as you would to a patient asking a medical question on a public forum:

- Clearly state the most likely diagnosis.
- Briefly explain the clinical reasoning in 2–3 concise sentences, using only details from the vignette
  (history, symptoms, lesion morphology, color, distribution, and duration).
- If relevant, briefly acknowledge alternative diagnoses or diagnostic uncertainty within the same explanation.

IMPORTANT RULES
- Do NOT introduce any information not present in the vignette.
- Do NOT assume test results, biopsy findings, or treatment response.
- Do NOT provide treatment recommendations, management advice, or next steps.
- Do NOT include disclaimers, warnings, or “see a doctor” language.
- Do NOT mention image quality unless it directly limits diagnostic confidence.

STYLE AND SCOPE
- Be direct, neutral, and non-chatty.
- Use patient-friendly medical language; brief dermatologic terms in parentheses are acceptable.
- Limit the entire response to one short paragraph (2–3 sentences total).
- If uncertainty exists, state it briefly and clearly.

OUTPUT FORMAT (STRICT)
Return plain text only, exactly in the following format:

<Diagnosis stated clearly. One paragraph containing the diagnosis followed by 2–3 sentences of clinical reasoning.>
"""


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
    # google/medgemma-27b-it -> google_medgemma-27b-it
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_")
    return s or "model"


def load_seen_encounter_ids(output_jsonl: Path) -> Set[str]:
    """
    Resume logic:
    - If output file exists, parse it and collect encounter_id values.
    - Anything already present is skipped on the next run.
    """
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
    """
    Convert whatever vignette representation you have into a string for the LLM.
    Priority:
      1) vignette (dict) -> pretty JSON
      2) vignette_text (dict/str)
      3) vignette_raw_text (str)
    """
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

    raise ValueError("No vignette found (expected vignette / vignette_text / vignette_raw_text).")


# =========================
# vLLM inference helpers
# =========================
def build_prompt(tokenizer: AutoTokenizer, vignette_text: str) -> str:
    """
    Build a chat-formatted prompt using the tokenizer's chat template when available.
    Falls back to a simple concatenation if no chat template exists.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": vignette_text},
    ]

    # Many instruction-tuned models expose a chat template.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback prompt (still works, just less "native" formatting)
        return f"{SYSTEM_PROMPT}\n\nVIGNETTE:\n{vignette_text}\n\nANSWER:\n"


def generate_one(
    llm: LLM,
    tokenizer: AutoTokenizer,
    vignette_text: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    seed: Optional[int],
) -> Dict[str, Any]:
    prompt = build_prompt(tokenizer, vignette_text)

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )

    # vLLM returns a list aligned with prompts; each item has .outputs (n-best).
    out = llm.generate([prompt], sampling_params=sampling, use_tqdm=False)[0]
    text = ""
    if out.outputs and out.outputs[0].text:
        text = out.outputs[0].text.strip()

    return {
        "content": text if text else None,
        "raw": {
            "prompt": prompt,
            # keep small, useful debug bits; avoid dumping huge internals
            "finish_reason": getattr(out.outputs[0], "finish_reason", None) if out.outputs else None,
        },
        "error": None if text else "empty_generation",
    }


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--model", type=str, default="microsoft/MediPhi-Instruct")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)

    # vLLM engine knobs (common ones you’ll likely want)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--dtype", type=str, default="auto", choices=("auto", "half", "bfloat16", "float16", "float32"))

    # Optional: for gated models, HF token via env var
    # export HF_TOKEN=...
    args = ap.parse_args()

    input_path = args.input_jsonl
    model = args.model.strip()

    out_name = f"{safe_model_filename(model)}_vllm_inference_vignettes.jsonl"
    output_path = input_path.parent / out_name

    seen = load_seen_encounter_ids(output_path)
    print(f"Resuming inference. Already seen {len(seen)} encounter_id(s).")
    total = sum(1 for _ in iter_jsonl(input_path))

    # Hugging Face auth (if needed)
    # vLLM/transformers will pick these up automatically in most setups.
    if os.environ.get("HUGGING_FACE_HUB_TOKEN") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

    # Tokenizer (for chat template) + vLLM engine
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(
        model=model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as out_f, tqdm(
        total=total, desc=f"vllm:{model}", unit="rec"
    ) as pbar:
        for i, rec in enumerate(iter_jsonl(input_path)):
            encounter_id = str(rec.get("encounter_id") or f"enc_{i:06d}").strip()
            if encounter_id in seen:
                pbar.update(1)
                continue

            out_rec = dict(rec)
            out_rec["encounter_id"] = encounter_id
            out_rec["inference_provider"] = "vllm"
            out_rec["inference_model"] = model

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

            # Robustness: if something transient happens, retry once
            last_err: Optional[str] = None
            result: Dict[str, Any] = {"content": None, "raw": None, "error": "unknown"}
            for attempt in range(2):
                try:
                    result = generate_one(
                        llm=llm,
                        tokenizer=tokenizer,
                        vignette_text=vignette_text,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                        seed=args.seed,
                    )
                    if result.get("error") is None:
                        break
                    last_err = str(result.get("error"))
                except Exception as e:
                    last_err = f"exception: {type(e).__name__}: {e}"
                    time.sleep(1.0 + attempt)

            out_rec["raw_provider_response"] = result.get("raw")
            if result.get("error") is not None:
                out_rec["responses"] = []
                out_rec["error"] = last_err or str(result.get("error"))
            else:
                out_rec["responses"] = [{"author_id": model, "content_en": result["content"]}]

            out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out_f.flush()
            seen.add(encounter_id)
            pbar.update(1)

    print(f"Done. Wrote/updated: {output_path}")


if __name__ == "__main__":
    main()
