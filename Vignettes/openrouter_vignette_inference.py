#!/usr/bin/env python3
"""
OpenRouter diagnosis inference over vignette JSONL (resume-safe, append-only).

Input JSONL: one encounter per line. Each line should contain:
- encounter_id (string)
- vignette (dict) OR vignette_raw_text (string) OR vignette_text (string/dict)

Output JSONL (append-only):
<model_name>_inference_vignettes.jsonl
- Preserves all original fields
- Adds/overwrites "responses" with ONE model response:
  [{"author_id": <model>, "content_en": <diagnosis>}]


Deps:
  pip install requests
  
python /home/USER/Documents/Dermatology-Evaluation-Framework/Thesis_Final/openrouter_vignette_inference.py \
  --input-jsonl /home/USER/Documents/Dermatology-Evaluation-Framework/Thesis_Final/vignettes/test_vignettes.jsonl \
  --model anthropic/claude-haiku-4.5 \
--vignette-mode no-image-findings \
--print-prompt 

python /Users/USER/Documents/Dermatology-Evaluation/Vignettes/openrouter_vignette_inference.py \
  --input-jsonl /Users/USER/Documents/Dermatology-Evaluation/Vignettes/test_vignettes_google_gemini-2.5-flash-lite.jsonl \
  --provider openrouter \
  --model openai/gpt-4.1 \
--vignette-mode with-image-findings \
--print-prompt \
--vignette-model google_gemini-2.5-flash-lite


"""

from __future__ import annotations
from tqdm.auto import tqdm
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set
import time

import requests

def drop_image_findings(obj: Any) -> Any:
    """
    Recursively remove any key named 'image_findings' from dicts.
    Works for nested structures.
    """
    if isinstance(obj, dict):
        return {k: drop_image_findings(v) for k, v in obj.items() if k != "image_findings"}
    if isinstance(obj, list):
        return [drop_image_findings(x) for x in obj]
    return obj

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
- Limit the entire response to **one short paragraph (2–3 sentences total)**.  
- If uncertainty exists, state it briefly and clearly.

OUTPUT FORMAT (STRICT)  
Return plain text only, exactly in the following format:

<Diagnosis stated clearly. One paragraph containing the diagnosis followed by 2–3 sentences of clinical reasoning.>

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
    # HTTP error (4xx/5xx) -> raise with body
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {r.text}")

    data = r.json()

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"OpenRouter returned error payload: {data.get('error')}")

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
    # openai/gpt-4o -> openai_gpt-4o
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
                # Ignore malformed lines; keep going.
                continue
    return seen


def vignette_to_text(rec: Dict[str, Any], vignette_mode: str = "with-image-findings") -> str:
    """
    Convert whatever vignette representation you have into a string for the LLM.
    If vignette_mode == 'no-image-findings', remove 'image_findings' from dict vignettes.
    """
    v = rec.get("vignette")
    if isinstance(v, dict):
        v2 = drop_image_findings(v) if vignette_mode == "no-image-findings" else v
        return json.dumps(v2, ensure_ascii=False, indent=2)

    vt = rec.get("vignette_text")
    if isinstance(vt, dict):
        vt2 = drop_image_findings(vt) if vignette_mode == "no-image-findings" else vt
        return json.dumps(vt2, ensure_ascii=False, indent=2)
    if isinstance(vt, str) and vt.strip():
        # Leave raw text untouched (safe default)
        return vt.strip()

    raw = rec.get("vignette_raw_text")
    if isinstance(raw, str) and raw.strip():
        # Leave raw text untouched (safe default)
        return raw.strip()

    raise ValueError("No vignette found in record (expected vignette / vignette_text / vignette_raw_text).")

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
    Unified chat call for supported providers.
    - provider: 'openrouter' or 'deepinfra' (OpenAI-compatible endpoint)
    - model: provider model id
    - messages: list of {"role":..,"content":..}
    """
    if provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = _openrouter_headers()
        payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
    elif provider == "deepinfra":
        base = os.environ.get("DEEPINFRA_BASE_URL", "https://api.deepinfra.com/v1/openai").rstrip("/")
        # DeepInfra openai-compatible chat endpoint:
        url = f"{base}/chat/completions"
        headers = _deepinfra_headers()
        # payload follows OpenAI chat completions shape
        payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}
    else:
        raise ValueError("Unsupported provider: choose 'openrouter' or 'deepinfra'")

    # POST
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    # HTTP-level failure -> raise to be caught by caller
    if r.status_code >= 400:
        # attempt to include json error if present
        body = r.text
        try:
            j = r.json()
            body = json.dumps(j, ensure_ascii=False)
        except Exception:
            pass
        raise RuntimeError(f"{provider} HTTP {r.status_code}: {body}")

    data = r.json()
    # Some providers return an "error" object inside 200 -> treat as error
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"{provider} returned error payload: {data.get('error')}")
    return data

def diagnose_provider(
    provider: str,
    model: str,
    vignette_text: str,
    temperature: float = 0.0,
    max_tokens: int = 700,
    retries: int = 1,
):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": vignette_text}]
    last_exc = None
    for attempt in range(retries + 1):
        try:
            data = provider_chat(provider=provider, model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            # Defensive checks
            if not isinstance(data, dict):
                return {"content": None, "raw": data, "error": "non-dict response"}
            if data.get("error"):
                return {"content": None, "raw": data, "error": f"provider_error: {data.get('error')}"}
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
    ap.add_argument("--vignette-model", type=str, default="gpt-4.1",)

    ap.add_argument(
    "--vignette-mode",
    choices=("with-image-findings", "no-image-findings"),
    default="with-image-findings",
    help="Whether to include vignette.image_findings when sending to the model.",
    )
    ap.add_argument(
    "--print-prompt",
    action="store_true",
    help="Print the full system+user prompt (messages) before calling the provider.",
    )


    args = ap.parse_args()

    input_path = args.input_jsonl
    model = args.model.strip()
    provider = args.provider
    if args.vignette_mode == "no-image-findings":
        out_name = f"{safe_model_filename(model)}_{provider}_inference_no-image-findings_vignette.jsonl"
    else:
        # out_name = f"{safe_model_filename(model)}_{provider}_inference_vignettes.jsonl"
        out_name = f"{safe_model_filename(model)}_{provider}_inference_vignettes_plus_images_by_{args.vignette_model}.jsonl"
    
    output_path = input_path.parent / "Logs" / args.vignette_model / out_name

    seen = load_seen_encounter_ids(output_path)
    print(f"Resuming inference. Already seen {len(seen)} encounter_id(s).")

    # Count total lines in input (to seed tqdm). iter_jsonl opens the file each call so it's safe.
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

            try:
                vignette_text = vignette_to_text(rec, vignette_mode=args.vignette_mode)
                if args.print_prompt:
                    messages_dbg = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": vignette_text},
                    ]
                    print("\n" + "=" * 90)
                    print(f"[DEBUG] encounter_id={encounter_id}  provider={provider}  model={model}  mode={args.vignette_mode}")
                    print(json.dumps(messages_dbg, ensure_ascii=False, indent=2))
                    print("=" * 90 + "\n")

            except Exception as e:
                out_rec["responses"] = []
                out_rec["error"] = f"No vignette: {type(e).__name__}: {e}"
                out_f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                out_f.flush()
                seen.add(encounter_id)
                pbar.update(1)
                continue

            result = diagnose_provider(
                provider=provider,
                model=model,
                vignette_text=vignette_text,
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
