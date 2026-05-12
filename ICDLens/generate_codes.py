from __future__ import annotations
import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from difflib import SequenceMatcher
import requests
from transformers import pipeline
from tqdm.auto import tqdm  


HF_LABEL_TO_STATUS = {"PRESENT": "present", "ABSENT": "negated", "POSSIBLE": "possible"}
ALLOWED_ASSERTIONS = {"present", "negated", "possible", "not_known"}

EM_TAG_RE = re.compile(r"</?em[^>]*>")

DEFAULT_OPENROUTER_SYSTEM_PROMPT = (
    "You are a biomedical NER + assertion tagger.\n"
    "Return ONLY valid JSON. No prose.\n"
    "Extract medical concepts (disease, disoarder, differential diagnosis).\n"
    "For EACH extracted term, also assign an assertion from this CLOSED SET:\n"
    "  present | negated | possible | not_known\n"
    "CRITICAL RULES:\n"
    "- 'term' MUST be copied EXACTLY as it appears in the input (same casing, spacing, punctuation).\n"
    "- Do NOT normalize, correct spelling, or paraphrase.\n"
    "- Do NOT return character offsets.\n"
    "Output schema exactly:\n"
    '{ "entities": [ {"term": str, "llm_assertion": str} ] }\n'
    "If unsure about a term or its assertion, omit the term or use not_known.\n"
)

DEFAULT_OPENROUTER_USER_PROMPT_TEMPLATE = (
    "Text:\n"
    "{text}\n\n"
    "Guidance for assertion:\n"
    "- negated if clearly denied/absent (e.g., 'no', 'denies', 'exclude')\n"
    "- possible if hedged/uncertain (e.g., 'would not exclude', '?', 'possible', 'might')\n"
    "- present if affirmed\n"
    "- not_known otherwise\n"
)

DEFAULT_OPENROUTER_ICD11_SELECT_SYSTEM_PROMPT = (
    "You are an expert ICD-11 clinical coder.\n"
    "You will be given:\n"
    "  (1) Clinical text context\n"
    "  (2) An extracted term\n"
    "  (3) A short list of ICD-11 candidate codes with titles\n"
    "Task: choose the ONE best candidate that matches the term in context.\n"
    "If NONE of the candidates fit (or the term is not actually a diagnosis), choose -1.\n\n"
    "Return ONLY valid JSON with this schema:\n"
    "{\"choice_index\": int}\n\n"
    "Rules:\n"
    "- choice_index must be one of the provided indices or -1.\n"
    "- Do NOT invent codes/titles.\n"
    "- Prefer the most specific clinically correct option.\n"
)

DEFAULT_OPENROUTER_ICD11_SELECT_USER_PROMPT_TEMPLATE = (
    "Clinical text (snippet):\n"
    "{text}\n\n"
    "Extracted term:\n"
    "{term}\n\n"
    "Candidates:\n"
    "{candidates}\n"
)

DEFAULT_PROP_WEIGHTS = {
    "Title": 1.20,
    "Synonym": 1.00,
    "Definition": 0.90,
    "Inclusion": 0.95,
}

DEFAULT_RERANK_WEIGHTS = {"api": 0.65, "pv": 0.30}

@dataclass
class Config:
    # I/O
    input_path: str = ""
    output_path: str = ""
    max_lines: Optional[int] = None  
    
    output_dir: str = str(Path(__file__).parent / "Logs")
    output_candidates_path: str = ""  
    output_final_path: str = ""      


    # Caching
    cache_dir: str = ".cache/icd11_framework"
    cache_enabled: bool = True

    # OpenRouter
    openrouter_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY"))
    openrouter_base: str = "https://openrouter.ai/api/v1"
    # openrouter_model: str = "mistralai/mistral-7b-instruct" # Model we use for Term extraction + Assertion v1
    openrouter_model: str = "openai/gpt-4o-mini" 
    openrouter_timeout_s: int = 60
    openrouter_temperature: float = 0.0
    openrouter_system_prompt: str = DEFAULT_OPENROUTER_SYSTEM_PROMPT
    openrouter_user_prompt_template: str = DEFAULT_OPENROUTER_USER_PROMPT_TEMPLATE
    openrouter_max_retries: int = 2
    openrouter_retry_backoff_s: float = 2.0
    
    # OpenRouter ICD-11 candidate chooser (post-retrieval)
    openrouter_select_enabled: bool = True
    openrouter_select_model: Optional[str] = "openai/gpt-4o-mini"  
    openrouter_select_top_k: int = 4
    openrouter_select_window_chars: int = 240  # take +/- window around first occurrence
    openrouter_select_system_prompt: str = DEFAULT_OPENROUTER_ICD11_SELECT_SYSTEM_PROMPT
    openrouter_select_user_prompt_template: str = DEFAULT_OPENROUTER_ICD11_SELECT_USER_PROMPT_TEMPLATE


    # HF assertion model
    hf_assert_model: str = "bvanaken/clinical-assertion-negation-bert"
    hf_min_conf: float = 0.60
    hf_device: int = -1 
    hf_truncation: bool = True

    # WHO ICD-11 API
    who_client_id: Optional[str] = field(default_factory=lambda: os.environ.get("ICD11_CLIENT_ID"))
    who_client_secret: Optional[str] = field(default_factory=lambda: os.environ.get("ICD11_CLIENT_SECRET"))
    who_token_url: str = "https://icdaccessmanagement.who.int/connect/token"
    who_api_base: str = "https://id.who.int"
    who_release_id: str = "2025-01"
    who_language: str = "en"
    who_timeout_s: int = 30
    who_use_flexisearch: bool = True
    who_top_n_per_term: Optional[int] = 10

    # Mention handling
    all_occurrences: bool = True
    code_negated: bool = True

    # Final response-level codes
    top_k_codes_per_response: Optional[int] = None  # how many codes to include per response

    # Reranking
    prop_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PROP_WEIGHTS))
    default_prop_weight: float = 0.85
    fuzzy_similarity_cap: float = 0.75
    rerank_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RERANK_WEIGHTS))

    # Extra ranking tunables (promotions / penalties)
    exact_title_bonus: float = 0.90   # large promotion when title == query (normalized)
    injected_bonus: float = 0.10      # small boost given to injected parents
    residual_penalty: float = 0.25    # penalty for "residual"/unspecified titles

    # Runtime behaviour
    sleep_s: float = 0.0
    fail_fast: bool = False
    verbose: bool = True

def load_config_file(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _json_value(value_str: str) -> Any:
    """Best-effort parse: JSON if possible, else string."""
    try:
        return json.loads(value_str)
    except Exception:
        return value_str

def derive_output_paths_from_input(cfg: Config) -> Tuple[Path, Path]:
    """
    If explicit output paths are provided, use them.
    Otherwise write to cfg.output_dir with names:
      {input_stem}_full_coded.jsonl
      {input_stem}_best_coded.jsonl
    """
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cfg.output_candidates_path:
        cand_path = Path(cfg.output_candidates_path)
    else:
        stem = Path(cfg.input_path).stem
        cand_path = outdir / f"{stem}_full_coded.jsonl"

    if cfg.output_final_path:
        final_path = Path(cfg.output_final_path)
    else:
        stem = Path(cfg.input_path).stem
        final_path = outdir / f"{stem}_best_coded.jsonl"

    cand_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    return cand_path, final_path

def apply_overrides(cfg: Config, overrides: Dict[str, Any]) -> None:
    """
    Update dataclass fields from a dict.
    Supports dict fields too.
    Unknown keys are ignored (but logged in verbose mode).
    """
    for k, v in (overrides or {}).items():
        if not hasattr(cfg, k):
            if cfg.verbose:
                print(f"[WARN] Unknown config key ignored: {k}", file=sys.stderr)
            continue
        setattr(cfg, k, v)

def apply_set_kv(cfg: Config, kv_list: List[str]) -> None:
    """
    Apply repeatable --set key=value.
    Supports dot paths for dict fields: rerank_weights.api=0.6
    """
    for item in kv_list or []:
        if "=" not in item:
            if cfg.verbose:
                print(f"[WARN] --set ignored (missing '='): {item}", file=sys.stderr)
            continue
        key, value_str = item.split("=", 1)
        value = _json_value(value_str)

        if "." in key:
            root, subkey = key.split(".", 1)
            if not hasattr(cfg, root):
                if cfg.verbose:
                    print(f"[WARN] --set unknown key ignored: {key}", file=sys.stderr)
                continue
            root_val = getattr(cfg, root)
            if not isinstance(root_val, dict):
                if cfg.verbose:
                    print(f"[WARN] --set expects dict field for '{root}', got {type(root_val)}", file=sys.stderr)
                continue
            root_val[subkey] = value
        else:
            if not hasattr(cfg, key):
                if cfg.verbose:
                    print(f"[WARN] --set unknown key ignored: {key}", file=sys.stderr)
                continue
            setattr(cfg, key, value)

class DiskCache:
    """
    Simple persistent JSON cache on disk.

    - Namespaces separate different request types.
    - Keys are arbitrary strings (we pass SHA-256 hashes).
    """

    def __init__(self, root_dir: str, enabled: bool = True):
        self.enabled = bool(enabled)
        self.root = Path(root_dir)
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _ns_dir(self, namespace: str) -> Path:
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get(self, namespace: str, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        path = self._ns_dir(namespace) / f"{key}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        path = self._ns_dir(namespace) / f"{key}.json"
        tmp = path.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            tmp.replace(path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

def stable_hash(obj: Any) -> str:
    """Stable SHA-256 of JSON-serializable content."""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def strip_em(html: str) -> str:
    """Remove <em> tags from WHO labels."""
    if not isinstance(html, str):
        return html
    return EM_TAG_RE.sub("", html)

def normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

def match_strength(query: str, text: str, fuzzy_cap: float = 0.75) -> float:
    """
    0..1 based on how well `text` matches `query`.
    Kept intentionally simple: fuzzy similarity only, capped.
    """
    q = normalize_text(query)
    t = normalize_text(text)
    if not q or not t:
        return 0.0
    return similarity(q, t) * float(fuzzy_cap)

def find_exact_spans(text: str, term: str) -> List[Tuple[int, int]]:
    if not term:
        return []
    return [(m.start(), m.end()) for m in re.finditer(re.escape(term), text)]

def mark_entity(text: str, start: int, end: int) -> str:
    return text[:start] + "[entity] " + text[start:end] + " [entity]" + text[end:]

def normalize_llm_assertion(a: Any) -> str:
    if not isinstance(a, str):
        return "not_known"
    a = a.strip().lower()
    if a in ("affirmed", "positive", "present"):
        return "present"
    if a in ("absent", "negative", "negated", "no"):
        return "negated"
    if a in ("possible", "suspected", "probable", "uncertain"):
        return "possible"
    if a in ("unknown", "not_known", "not known", "unk"):
        return "not_known"
    return a if a in ALLOWED_ASSERTIONS else "not_known"

def combine_assertions(llm_a: str, hf_a: str) -> str:
    """
    Your requested rule:
    - If HF and LLM both agree, return that assertion
    - If HF and LLM both have a known assertion and they disagree => "possible"
    - Otherwise => "not_known"
    """
    llm_a = normalize_llm_assertion(llm_a)
    hf_a = normalize_llm_assertion(hf_a)

    if llm_a == hf_a:
        return llm_a

    if llm_a != hf_a:
        if llm_a != "not_known" and hf_a != "not_known":
            return "possible"
        return "not_known"

    return "not_known"

def aggregate_final_assertions(final_assertions: List[str]) -> str:
    # TODO: aggregation
    """
    Reduce mention-level final assertions to one term-level assertion.
    Priority: present > possible > negated > not_known
    """
    s = set(a for a in (final_assertions or []) if a)
    if "present" in s:
        return "present"
    if "possible" in s:
        return "possible"
    if "negated" in s:
        return "negated"
    return "not_known"


def _normalize_list_scores(candidates: List[Dict[str, Any]], score_key: str) -> None:
    """
    Mutates candidates in-place so that candidates[*][score_key] are in 0..1 range.
    If max is 0, leaves values as-is (coerced to float).
    """
    vals = []
    for c in candidates:
        v = c.get(score_key)
        try:
            vals.append(float(v) if v is not None else 0.0)
        except Exception:
            vals.append(0.0)
    max_v = max(vals) if vals else 0.0
    if max_v <= 0.0:
        # just coerce to float 0..1
        for i, c in enumerate(candidates):
            c[score_key] = float(vals[i])
    else:
        for i, c in enumerate(candidates):
            c[score_key] = float(vals[i]) / float(max_v)

def rerank_icd11_candidates(term: str, candidates: List[Dict[str, Any]], cfg: Config) -> List[Dict[str, Any]]:
    """
    TODO: revisit again
    Normalizes api/title/pv scores and computes a rerank_score.
    Expects candidates to have:
      - icd11_code
      - title
      - score (WHO API score) 
      - matching_property_values: list of dicts with property_id, matched_text, pv_score, ...
      - entity_id
    Returns candidates sorted best-first (mutates candidate dicts with added fields).
    """
    # Defensive copy of list to avoid surprising external mutation
    cand = list(candidates)

    # Normalize server scores (api 'score' field) to 0..1
    _normalize_list_scores(cand, "score")

    # Compute best pv-based rank per candidate (0..1) using your property weights + match_strength
    for c in cand:
        term_text = term or ""
        best_pv_rank_score = 0.0
        best_pv_property_id = None
        best_pv_match_text = None

        for pv in (c.get("matching_property_values") or []):
            prop_id = pv.get("property_id")
            matched_text = pv.get("matched_text") or ""
            pv_score = pv.get("pv_score")

            w = float(cfg.prop_weights.get(str(prop_id), cfg.default_prop_weight))
            m = match_strength(term_text, matched_text, fuzzy_cap=cfg.fuzzy_similarity_cap)
            
            # even if it is in a very important field, the text similarity will count more
            pv_rank_score = w * m

            # modest influence from provided pv_score if numeric (normalize is not strictly necessary)
            if pv_score is not None:
                try:
                    pv_rank_score *= (0.85 + 0.15 * float(pv_score))
                except Exception:
                    pass

            if pv_rank_score > best_pv_rank_score:
                best_pv_rank_score = pv_rank_score
                best_pv_property_id = prop_id
                best_pv_match_text = matched_text

        c["best_pv_rank_score"] = best_pv_rank_score
        c["best_pv_property_id"] = best_pv_property_id
        c["best_pv_match_text"] = best_pv_match_text

    # Normalize pv rank scores across candidates (so pv influence is comparable)
    _normalize_list_scores(cand, "best_pv_rank_score")


    # Combine signals into final rerank_score
    api_w = float(cfg.rerank_weights.get("api", 0.65))
    pv_w = float(cfg.rerank_weights.get("pv", 0.30))


    for c in cand:
        api_score = float(c.get("score") or 0.0)
        best_pv = float(c.get("best_pv_rank_score") or 0.0)
        # Base weighted sum (server/api remains dominant when available)
        base_score = api_w * api_score + pv_w * best_pv 

        # Add promotions / penalties
        rerank_score = base_score

        # Attach diagnostic fields
        c["rerank_score"] = float(rerank_score)
        c["api_score_norm"] = api_score
        c["pv_score_norm"] = best_pv

    return sorted(cand, key=lambda x: float(x.get("rerank_score") or 0.0), reverse=True)

class OpenRouterClient:
    def __init__(self, cfg: Config, cache: DiskCache):
        self.cfg = cfg
        self.cache = cache

    def ner_with_assertion(self, text: str) -> List[Dict[str, Any]]:
        if not self.cfg.openrouter_api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY (env) or openrouter_api_key (config).")

        system = self.cfg.openrouter_system_prompt
        user = self.cfg.openrouter_user_prompt_template.format(text=text)

        payload = {
            "model": self.cfg.openrouter_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.openrouter_temperature,
            "response_format": {"type": "json_object"},
        }

        cache_key = stable_hash(payload)
        cached = self.cache.get("openrouter", cache_key)
        if cached is not None:
            return cached.get("entities", []) if isinstance(cached, dict) else []

        headers = {
            "Authorization": f"Bearer {self.cfg.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        last_err = None
        for attempt in range(self.cfg.openrouter_max_retries + 1):
            try:
                r = requests.post(
                    f"{self.cfg.openrouter_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.openrouter_timeout_s,
                )
                # Retry for rate limit / transient
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} {r.text}", response=r)

                r.raise_for_status()
                data = r.json()
                content = data["choices"][0]["message"]["content"]
                obj = json.loads(content) if isinstance(content, str) else {}
                entities = obj.get("entities", [])
                out = {"entities": entities if isinstance(entities, list) else []}
                self.cache.set("openrouter", cache_key, out)
                return out["entities"]
            except Exception as ex:
                last_err = ex
                if attempt < self.cfg.openrouter_max_retries:
                    time.sleep(self.cfg.openrouter_retry_backoff_s * (attempt + 1))
                else:
                    break

        # No cache write on failure
        if self.cfg.fail_fast:
            raise last_err
        if self.cfg.verbose:
            print(f"[WARN] OpenRouter failed, returning empty entities: {last_err}", file=sys.stderr)
        return []
    
    def pick_best_icd11_candidate(
        self,
        full_text: str,
        term: str,
        candidates_topk: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Returns a dict like {"choice_index": int} on success, or None on failure.
        choice_index == -1 means "none fit".
        """
        if not self.cfg.openrouter_select_enabled:
            return None
        if not candidates_topk:
            return None
        if not self.cfg.openrouter_api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY (env) or openrouter_api_key (config).")

        # Build a focused context snippet around the first occurrence of the term (best signal, fewer tokens)
        snippet = full_text or ""
        spans = find_exact_spans(full_text or "", term or "")
        if spans:
            start, end = spans[0]
            w = int(self.cfg.openrouter_select_window_chars)
            s0 = max(0, start - w)
            s1 = min(len(full_text), end + w)
            snippet = full_text[s0:s1]
            # mark entity in the snippet (indices relative to snippet)
            snippet = mark_entity(snippet, start - s0, end - s0)

        # Compose candidate list text (top-k only)
        lines = []
        for i, c in enumerate(candidates_topk):
            code = c.get("icd11_code") or ""
            title = strip_em(c.get("title") or "")
            lines.append(f"{i}) {code} — {title}")
        candidates_block = "\n".join(lines)

        system = self.cfg.openrouter_select_system_prompt
        user = self.cfg.openrouter_select_user_prompt_template.format(
            text=snippet,
            term=term,
            candidates=candidates_block,
        )
        # print(user)

        model = self.cfg.openrouter_select_model or self.cfg.openrouter_model
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        cache_key = stable_hash(payload)
        cached = self.cache.get("openrouter_icd11_select", cache_key)
        if cached is not None and isinstance(cached, dict):
            return cached

        headers = {
            "Authorization": f"Bearer {self.cfg.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        last_err = None
        for attempt in range(self.cfg.openrouter_max_retries + 1):
            try:
                r = requests.post(
                    f"{self.cfg.openrouter_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.cfg.openrouter_timeout_s,
                )
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} {r.text}", response=r)

                r.raise_for_status()
                data = r.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON (with a small fallback)
                try:
                    obj = json.loads(content) if isinstance(content, str) else (content or {})
                except Exception:
                    m = re.search(r"\{.*\}", str(content), flags=re.S)
                    obj = json.loads(m.group(0)) if m else {}

                if not isinstance(obj, dict):
                    obj = {}

                # Validate choice_index
                ci = obj.get("choice_index", None)
                try:
                    ci = int(ci)
                except Exception:
                    ci = -999

                if ci != -1 and not (0 <= ci < len(candidates_topk)):
                    # Invalid -> treat as failure (fallback logic will handle)
                    raise ValueError(f"Invalid choice_index={ci}")

                out = {"choice_index": ci}
                self.cache.set("openrouter_icd11_select", cache_key, out)
                return out

            except Exception as ex:
                last_err = ex
                if attempt < self.cfg.openrouter_max_retries:
                    time.sleep(self.cfg.openrouter_retry_backoff_s * (attempt + 1))
                else:
                    break

        if self.cfg.verbose:
            print(f"[WARN] ICD-11 candidate picker failed for term '{term}': {last_err}", file=sys.stderr)
        return None



class WHOICD11Client:
    def __init__(self, cfg: Config, cache: DiskCache):
        self.cfg = cfg
        self.cache = cache
        self._token: Optional[str] = None
        self._token_exp: int = 0
        self._token_cache_path = Path(cfg.cache_dir) / "who_token.json"

    def _load_token_from_disk(self) -> None:
        if not self.cfg.cache_enabled:
            return
        p = self._token_cache_path
        if not p.exists():
            return
        try:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            tok = obj.get("access_token")
            exp = int(obj.get("expires_at") or 0)
            if tok and exp:
                self._token = tok
                self._token_exp = exp
        except Exception:
            pass

    def _save_token_to_disk(self, token: str, expires_at: int) -> None:
        if not self.cfg.cache_enabled:
            return
        p = self._token_cache_path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump({"access_token": token, "expires_at": expires_at}, f)
            tmp.replace(p)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except Exception:
                    pass

    def get_token(self, force_refresh: bool = False) -> str:
        if not self.cfg.who_client_id or not self.cfg.who_client_secret:
            raise RuntimeError("Set ICD11_CLIENT_ID and ICD11_CLIENT_SECRET (env) or in config.")

        now = int(time.time())

        if self._token is None and self.cfg.cache_enabled:
            self._load_token_from_disk()

        if (not force_refresh) and self._token and now < self._token_exp - 30:
            return self._token

        data = {
            "grant_type": "client_credentials",
            "client_id": self.cfg.who_client_id,
            "client_secret": self.cfg.who_client_secret,
            "scope": "icdapi_access",
        }
        r = requests.post(self.cfg.who_token_url, data=data, timeout=self.cfg.who_timeout_s)
        r.raise_for_status()
        tok = r.json()

        self._token = tok["access_token"]
        self._token_exp = now + int(tok.get("expires_in", 3600))
        self._save_token_to_disk(self._token, self._token_exp)
        return self._token

    def mms_search_raw(self, term: str) -> Dict[str, Any]:
        """
        Cached WHO MMS search response (raw JSON).
        Cache key excludes bearer token so reranking changes won't require re-fetch.
        """
        params = {"q": term, "flatResults": "true"}
        if self.cfg.who_use_flexisearch:
            params["useFlexisearch"] = "true"

        cache_payload = {
            "release_id": self.cfg.who_release_id,
            "language": self.cfg.who_language,
            "params": params,
        }
        cache_key = stable_hash(cache_payload)
        cached = self.cache.get("who_search_raw", cache_key)
        if cached is not None:
            return cached if isinstance(cached, dict) else {}

        token = self.get_token()
        url = f"{self.cfg.who_api_base}/icd/release/11/{self.cfg.who_release_id}/mms/search"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Accept-Language": self.cfg.who_language,
            "API-Version": "v2",
        }

        r = requests.get(url, headers=headers, params=params, timeout=self.cfg.who_timeout_s)
        if r.status_code == 401:
            # token expired/invalid; refresh once
            token = self.get_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            r = requests.get(url, headers=headers, params=params, timeout=self.cfg.who_timeout_s)

        r.raise_for_status()
        data = r.json()
        self.cache.set("who_search_raw", cache_key, data)
        return data
 
    def get_parents(self, entity_id: str) -> List[Dict[str, Any]]:
        """
        Query WHO API for parents (ancestors) of an entity.
        Returns a list of dicts with keys such as 'uri'/'code'/'title'.
        This implementation uses a plausible WHO endpoint; adjust path if WHO API differs.
        """
        if not entity_id:
            return []

        try:
            token = self.get_token()
            url = entity_id
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "Accept-Language": self.cfg.who_language,
                "API-Version": "v2",
            }
            r = requests.get(url, headers=headers, timeout=self.cfg.who_timeout_s)
            if r.status_code == 401:
                token = self.get_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                r = requests.get(url, headers=headers, timeout=self.cfg.who_timeout_s)
            r.raise_for_status()
            data = r.json()
            parents = []
            if 'parent' in data:
                for parent_uri in data['parent']:
                    # Get parent details
                    parent_response = requests.get(parent_uri, headers=headers)
                    if parent_response.status_code == 200:
                        parent_data = parent_response.json()
                        parents.append({
                            'code': parent_data.get('code', 'N/A'),
                            'title': parent_data.get('title', {}).get('@value', 'N/A'),
                            'uri': parent_uri
                        })
    
            return parents
            
        except Exception:
            # swallow and return empty if failure (consistent with other client methods)
            return []

    def search_candidates(self, term: str, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        try:
            # 1) get raw & parse
            raw = self.mms_search_raw(term)
            candidates = self.parse_candidates(raw)

            # If no candidates, return early
            if not candidates:
                return []

            # Index by entity_id (or fallback to title+code) so we can merge/dedupe
            by_uri: Dict[str, Dict[str, Any]] = {}
            for c in candidates:
                key = c.get("entity_id") or f"{c.get('icd11_code')}_{strip_em(c.get('title') or '')}"
                by_uri[key] = dict(c)

            # rerank (use our rerank function which normalizes scores internally)
            merged = list(by_uri.values())
            ranked = rerank_icd11_candidates(term, merged, self.cfg)

            # apply top_n behavior (if None -> use cfg.who_top_n_per_term)
            n = top_n if top_n is not None else self.cfg.who_top_n_per_term
            if n is None:
                return ranked
            return ranked[: int(n)]
        except Exception as ex:
            if self.cfg.fail_fast:
                raise
            if self.cfg.verbose:
                print(f"[WARN] WHO search failed for term '{term}': {ex}", file=sys.stderr)
            return []


    @staticmethod
    def parse_candidates(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract a list of candidate dicts from the raw WHO search response.
        """
        dest = raw.get("destinationEntities", []) or []
        out: List[Dict[str, Any]] = []
        for d in dest:
            code = d.get("theCode")
            title = d.get("title")
            if not code or not title:
                continue

            api_score = d.get("score")
            try:
                api_score_f = float(api_score) if api_score is not None else 0.0
            except Exception:
                api_score_f = 0.0

            matching_pvs = d.get("matchingPVs", []) or []
            matching: List[Dict[str, Any]] = []
            for pv in matching_pvs:
                prop_id = pv.get("propertyId")
                label_html = pv.get("label", "") or ""
                label_text = strip_em(label_html)
                matching.append(
                    {
                        "property_id": prop_id,
                        "matched_text": label_text,
                        "matched_html": label_html,
                        "pv_score": pv.get("score"),
                        "important": pv.get("important"),
                        "foundation_uri": pv.get("foundationUri"),
                        "property_value_type": pv.get("propertyValueType"),
                    }
                )

            out.append(
                {
                    "icd11_code": code,
                    "title": title,
                    "score": api_score_f,
                    "entity_id": d.get("id"),
                    "matching_property_values": matching,
                }
            )
        return out

class ICD11CodingFramework:
    """
    Reusable framework object. Import and use:
        fw = ICD11CodingFramework(Config(...))
        terms = fw.extract_terms_with_candidates("Consider pompholyx...")
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache = DiskCache(cfg.cache_dir, enabled=cfg.cache_enabled)
        self.openrouter = OpenRouterClient(cfg, self.cache)
        self.who = WHOICD11Client(cfg, self.cache)
        self._hf_pipe = None
        self._hf_cache: Dict[str, Tuple[str, float]] = {}  # in-memory; key = stable_hash(marked_text)

    def _get_hf_pipe(self):
        if self._hf_pipe is None:
            self._hf_pipe = pipeline(
                "text-classification",
                model=self.cfg.hf_assert_model,
                device=self.cfg.hf_device,
            )
        return self._hf_pipe

    def hf_assertion_for_span(self, text: str, start: int, end: int) -> Tuple[str, float]:
        """
        Returns (status, score). status in {present, negated, possible, not_known}.
        """
        marked = mark_entity(text, start, end)
        key = stable_hash({"marked": marked, "model": self.cfg.hf_assert_model, "min_conf": self.cfg.hf_min_conf})
        if key in self._hf_cache:
            return self._hf_cache[key]

        pipe = self._get_hf_pipe()
        pred = pipe(marked, truncation=self.cfg.hf_truncation)[0]
        label = str(pred.get("label", "")).upper()
        score = float(pred.get("score", 0.0))
        status = HF_LABEL_TO_STATUS.get(label, "not_known")
        if score < float(self.cfg.hf_min_conf):
            status = "not_known"

        self._hf_cache[key] = (status, score)
        return status, score

    def extract_terms_with_candidates(self, text: str) -> List[Dict[str, Any]]:
        """
        Term-level extraction:
          - term
          - llm assertion
          - mention-level HF + final assertion
          - term-level final assertion (aggregated across mentions)
          - WHO candidates (full metadata + rerank fields)
          - best candidate (top of ranked list)
        """
        entities = self.openrouter.ner_with_assertion(text)
        if not entities:
            return []

        # De-duplicate by term (keep first assertion seen)
        term_to_llm_assert: Dict[str, str] = {}
        for e in entities:
            if not isinstance(e, dict):
                continue
            term = e.get("term")
            if not isinstance(term, str) or not term:
                continue
            if term not in term_to_llm_assert:
                term_to_llm_assert[term] = normalize_llm_assertion(e.get("llm_assertion"))

        out_terms: List[Dict[str, Any]] = []

        for term, llm_a in term_to_llm_assert.items():
            spans = find_exact_spans(text, term)
            if not spans:
                continue

            span_list = spans if self.cfg.all_occurrences else spans[:1]

            mentions: List[Dict[str, Any]] = []
            mention_final_assertions: List[str] = []

            for start, end in span_list:
                hf_a, hf_score = self.hf_assertion_for_span(text, start, end)
                final_a = combine_assertions(llm_a, hf_a)

                mentions.append(
                    {
                        "start": start,
                        "end": end,
                        "llm_assertion": llm_a,
                        "hf_assertion": hf_a,
                        "hf_score": float(hf_score),
                    }
                )
                mention_final_assertions.append(final_a)

            term_final_assertion = aggregate_final_assertions(mention_final_assertions)

            # Determine whether to query WHO / code the term
            do_code = (term_final_assertion in ("present", "possible")) or (
                self.cfg.code_negated and term_final_assertion == "negated"
            )

            candidates: List[Dict[str, Any]] = []
            best_candidate: Optional[Dict[str, Any]] = None

            if do_code:
                candidates = self.who.search_candidates(term, top_n=self.cfg.who_top_n_per_term)

                if candidates:
                    # Only pass top-4 to the LLM chooser
                    k = max(1, int(self.cfg.openrouter_select_top_k))
                    topk = candidates[:k]

                    pick = self.openrouter.pick_best_icd11_candidate(text, term, topk)
                    # print(pick)
                    # If LLM explicitly says "none fit" => remove the term
                    if pick is not None and int(pick.get("choice_index", -999)) == -1:
                        continue  # <-- term removed from output

                    # If picker failed (None) or gave valid index => choose
                    chosen = topk[0]  # fallback
                    if pick is not None:
                        idx = int(pick.get("choice_index", 0))
                        if 0 <= idx < len(topk):
                            chosen = topk[idx]

                    best_candidate = {
                        "icd11_code": chosen.get("icd11_code"),
                        "title": strip_em(chosen.get("title")),
                        "entity_id": chosen.get("entity_id"),
                        "rerank_score": float(chosen.get("rerank_score") or 0.0),
                    }


            out_terms.append(
                {
                    "term": term,
                    "mentions": mentions,
                    "final_assertion": term_final_assertion,
                    "candidates": candidates,
                    "best_candidate": best_candidate,
                }
            )

        return out_terms


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        # 1) Try normal JSON first
        try:
            data = json.load(f)

            if isinstance(data, dict):
                yield data
                return

            if isinstance(data, list):
                for i, item in enumerate(data, 1):
                    if isinstance(item, dict):
                        yield item
                    else:
                        print(
                            f"[WARN] Skipping non-object item #{i} in JSON array: {type(item).__name__}",
                            file=sys.stderr,
                        )
                return

            print(
                f"[WARN] JSON root is {type(data).__name__}, expected dict or list; nothing yielded.",
                file=sys.stderr,
            )
            return

        except json.JSONDecodeError:
            # Not valid single-JSON (often JSONL); fall back to JSONL
            f.seek(0)

        # 2) JSONL fallback
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as ex:
                print(f"[WARN] Skipping invalid JSON at line {line_no}: {ex}", file=sys.stderr)
                continue

            if isinstance(obj, dict):
                yield obj
            else:
                print(
                    f"[WARN] Skipping non-object JSON at line {line_no}: {type(obj).__name__}",
                    file=sys.stderr,
                )
    
def write_jsonl(path: str, items: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")



def process_dataset(cfg: Config) -> None:
    fw = ICD11CodingFramework(cfg)

    cand_path, final_path = derive_output_paths_from_input(cfg)

    processed = 0
    written = 0

    # Count total lines for tqdm (fast scan)
    total_lines = None
    try:
        total_lines = sum(1 for _ in Path(cfg.input_path).open("r", encoding="utf-8") if _.strip())
    except Exception:
        total_lines = None

    with cand_path.open("w", encoding="utf-8") as f_cand, final_path.open("w", encoding="utf-8") as f_final:
        pbar = tqdm(total=total_lines, desc="Processing encounters", unit="enc")
        try:
            for enc in read_jsonl(cfg.input_path):
                processed += 1
                if cfg.max_lines is not None and processed > int(cfg.max_lines):
                    break

                encounter_id = enc.get("encounter_id")
                author_id = enc.get("author_id")
                image_ids = enc.get("image_ids", [])

                responses_in = enc.get("responses", []) or []
                responses_out_candidates: List[Dict[str, Any]] = []
                responses_out_final: List[Dict[str, Any]] = []

                for r in responses_in:
                    if not isinstance(r, dict):
                        continue

                    r_author = r.get("author_id")
                    content_en = r.get("content_en")
                    completeness = r.get("completeness", "")
                    contains_freq_ans = r.get("contains_freq_ans", "")

                    if not isinstance(content_en, str):
                        content_en = "" if content_en is None else str(content_en)

                    try:
                        terms_full = fw.extract_terms_with_candidates(content_en) if content_en.strip() else []

                        # Build final-only terms (no candidates list)
                        terms_final: List[Dict[str, Any]] = []
                        seen = set()
                        candidate_codes: List[str] = []

                        for t in terms_full:
                            bc = t.get("best_candidate")
                            if bc:
                                terms_final.append(
                                    {
                                        "term": t.get("term"),
                                        "final_assertion": t.get("final_assertion"),
                                        "best_candidate": bc,
                                    }
                                )
                                code = (bc or {}).get("icd11_code")
                                if code and code not in seen:
                                    seen.add(code)
                                    candidate_codes.append(code)

                    except Exception as ex:
                        if cfg.fail_fast:
                            raise
                        if cfg.verbose:
                            print(
                                f"[WARN] Failed coding response (enc={encounter_id}, resp_author={r_author}): {ex}",
                                file=sys.stderr,
                            )
                        terms_full = []
                        terms_final = []
                        candidate_codes = []

                    responses_out_candidates.append(
                        {
                            "author_id": r_author,
                            "content_en": content_en,
                            "terms": terms_full,  # includes candidates
                            "completeness": completeness,
                            "contains_freq_ans": contains_freq_ans,
                        }
                    )

                    responses_out_final.append(
                        {
                            "author_id": r_author,
                            "content_en": content_en,
                            "terms": terms_final,  # best only
                            "candidate_codes": candidate_codes,
                            "completeness": completeness,
                            "contains_freq_ans": contains_freq_ans,
                        }
                    )

                    if cfg.sleep_s and cfg.sleep_s > 0:
                        time.sleep(float(cfg.sleep_s))

                out_obj_candidates = {
                    "encounter_id": encounter_id,
                    "author_id": author_id,
                    "image_ids": image_ids,
                    "responses": responses_out_candidates,
                }
                out_obj_final = {
                    "encounter_id": encounter_id,
                    "author_id": author_id,
                    "image_ids": image_ids,
                    "responses": responses_out_final,
                }

                f_cand.write(json.dumps(out_obj_candidates, ensure_ascii=False) + "\n")
                f_final.write(json.dumps(out_obj_final, ensure_ascii=False) + "\n")

                written += 1
                if cfg.verbose and (written % 10 == 0):
                    print(f"[INFO] Wrote {written} encounter lines...", file=sys.stderr)

                # update progress bar with useful postfix
                pbar.set_postfix({"written": written, "enc": encounter_id})
                pbar.update(1)

        finally:
            pbar.close()

    if cfg.verbose:
        print(f"[DONE] Wrote candidates to {cand_path}", file=sys.stderr)
        print(f"[DONE] Wrote final to {final_path}", file=sys.stderr)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ICD-11 candidate code extraction for JSONL dataset.")
    p.add_argument("--input", dest="input_path", required=True, help="Path to input JSONL dataset")

    # output is OPTIONAL now
    p.add_argument("--outdir", dest="output_dir", default=str(Path(__file__).parent / "Logs"), help="Output directory (default: ICDLens/Logs)")
    p.add_argument("--candidates-out", dest="output_candidates_path", default=None, help="Optional explicit candidates output path")
    p.add_argument("--final-out", dest="output_final_path", default=None, help="Optional explicit final output path")

    # keep the rest as-is...
    p.add_argument("--config", dest="config_path", default=None, help="Optional JSON config file")
    p.add_argument("--cache-dir", dest="cache_dir", default=None, help="Cache directory (overrides config)")
    p.add_argument("--max-lines", dest="max_lines", type=int, default=None, help="Process only first N encounters")
    p.add_argument("--sleep", dest="sleep_s", type=float, default=None, help="Sleep seconds between responses")
    p.add_argument("--fail-fast", dest="fail_fast", action="store_true", help="Crash on first error")
    p.add_argument("--quiet", dest="quiet", action="store_true", help="Less logging")
    p.add_argument("--set", dest="set_kv", action="append", default=[],
                   help="Override any config key, e.g. --set who_release_id=\"2025-01\"")
    return p



def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    cfg = Config()
    cfg.input_path = args.input_path
    cfg.output_dir = args.output_dir

    if args.output_candidates_path is not None:
        cfg.output_candidates_path = args.output_candidates_path
    if args.output_final_path is not None:
        cfg.output_final_path = args.output_final_path


    if args.quiet:
        cfg.verbose = False

    if args.config_path:
        overrides = load_config_file(args.config_path)
        apply_overrides(cfg, overrides)

    # CLI overrides
    if args.cache_dir is not None:
        cfg.cache_dir = args.cache_dir
    if args.max_lines is not None:
        cfg.max_lines = args.max_lines
    if args.sleep_s is not None:
        cfg.sleep_s = args.sleep_s
    if args.fail_fast:
        cfg.fail_fast = True

    # Apply any --set overrides last
    apply_set_kv(cfg, args.set_kv)

    # Ensure required paths
    if not cfg.input_path:
        raise SystemExit("--input is required")


    process_dataset(cfg)


if __name__ == "__main__":
    main()
