#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote, urlparse

import requests

Item = Tuple[str, str, str]  # (code, uri, title)
_URI_RE = re.compile(r"(https?://[^\s,;&]+|id\.who\.int/[^\s,;&]+)", re.IGNORECASE)


# =========================
# WHO ICD-11 minimal client
# =========================

@dataclass
class WHOConfig:
    client_id: str
    client_secret: str
    language: str = "en"
    timeout_s: int = 30
    token_url: str = "https://icdaccessmanagement.who.int/connect/token"


class WHOICD11Client:
    def __init__(self, cfg: WHOConfig):
        self.cfg = cfg
        self._token: Optional[str] = None
        self._token_exp: int = 0  # unix seconds

    def get_token(self, *, force_refresh: bool = False) -> str:
        if not self.cfg.client_id or not self.cfg.client_secret:
            raise RuntimeError("Missing WHO ICD-11 credentials (client_id / client_secret).")

        now = int(time.time())
        if not force_refresh and self._token and now < self._token_exp - 30:
            return self._token

        r = requests.post(
            self.cfg.token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": self.cfg.client_id,
                "client_secret": self.cfg.client_secret,
                "scope": "icdapi_access",
            },
            timeout=self.cfg.timeout_s,
        )
        r.raise_for_status()
        tok = r.json()
        self._token = tok["access_token"]
        self._token_exp = now + int(tok.get("expires_in", 3600))
        return self._token

    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.get_token()}",
            "Accept": "application/json",
            "Accept-Language": self.cfg.language,
            "API-Version": "v2",
        }

    def get_json(self, url: str) -> Dict[str, Any]:
        h = self.headers()
        r = requests.get(url, headers=h, timeout=self.cfg.timeout_s)
        if r.status_code == 401:  # refresh once
            h["Authorization"] = f"Bearer {self.get_token(force_refresh=True)}"
            r = requests.get(url, headers=h, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        return r.json()


# ======================
# Persistent disk cache
# ======================

class DiskParentsCache:
    """JSON cache on disk: uri -> [codes along ancestry, self..chapter]."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: Dict[str, List[str]] = {}
        self._dirty = False
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def get(self, key: str) -> Optional[List[str]]:
        return self._data.get(key)

    def put(self, key: str, value: List[str]) -> None:
        self._data[key] = value
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        self._dirty = False

    @property
    def size(self) -> int:
        return len(self._data)


# ==================
# Parsing utilities
# ==================

def flat_prf_if_any(gt_codes: List[str], pr_codes: List[str]) -> Dict[str, float]:
    G = set(gt_codes)
    P = set(pr_codes)
    inter = G & P

    # If no overlap at all, force everything to 0 (your rule)
    if not inter:
        return {"FlatP": 0.0, "FlatR": 0.0, "FlatF1_if_any": 0.0, "FlatInter": 0.0}

    flat_p = len(inter) / len(P) if P else 0.0
    flat_r = len(inter) / len(G) if G else 0.0
    flat_f1 = (2 * flat_p * flat_r / (flat_p + flat_r)) if (flat_p + flat_r) else 0.0

    return {
        "FlatP": flat_p,
        "FlatR": flat_r,
        "FlatF1_if_any": flat_f1,
        "FlatInter": float(len(inter)),
    }

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _clean_uri(raw: str) -> Optional[str]:
    m = _URI_RE.search(raw.strip())
    if not m:
        return None
    s = m.group(0)
    if s.startswith("id."):
        s = "https://" + s
    p = urlparse(s)
    if not (p.scheme and p.netloc):
        return None
    path = quote(p.path, safe="/:@&+$,;=")
    return f"{p.scheme}://{p.netloc}{path}{'?' + p.query if p.query else ''}{'#' + p.fragment if p.fragment else ''}"


def normalize_entity_uri(field: Any) -> Optional[str]:
    """Return the first plausible ICD-11 entity URI found in `field`."""
    if field is None:
        return None
    if isinstance(field, list):
        for el in field:
            u = normalize_entity_uri(el)
            if u:
                return u
        return None
    if isinstance(field, dict):
        for k in ("@id", "id", "uri", "entity_id"):
            u = normalize_entity_uri(field.get(k))
            if u:
                return u
        return _clean_uri(json.dumps(field))
    if isinstance(field, str):
        return _clean_uri(field)
    return None


def extract_codes(enc: Dict[str, Any]) -> List[str]:
    codes: Set[str] = set()
    for resp in enc.get("responses") or []:
        for term in resp.get("terms") or []:
            if term.get("final_assertion") in ("present", "possible"):
                bc = term.get("best_candidate") or {}
                code = bc.get("icd11_code") or bc.get("code")
                if code:
                    codes.add(code)
            else:
                print(f"Skipping code extraction for term with assertion {term.get('final_assertion')}: {term}")
    return sorted(codes)


def extract_items(enc: Dict[str, Any]) -> List[Item]:
    out: List[Item] = []
    for resp in enc.get("responses") or []:
        for term in resp.get("terms") or []:
            bc = term.get("best_candidate") or {}
            code = bc.get("icd11_code") or bc.get("code")
            uri = normalize_entity_uri(bc.get("entity_id") or bc.get("uri") or bc.get("id"))
            if code and uri:
                out.append((code, uri, bc.get("title") or ""))
    # dedupe by (code, uri)
    seen: Set[Tuple[str, str]] = set()
    dedup: List[Item] = []
    for code, uri, title in out:
        key = (code, uri)
        if key not in seen:
            seen.add(key)
            dedup.append((code, uri, title))
    return dedup


# =====================
# Hierarchy augmentation
# =====================

def _first_parent_uri(parent: Any) -> Optional[str]:
    if isinstance(parent, list) and parent:
        parent = parent[0]
    if isinstance(parent, dict):
        return parent.get("@id") or parent.get("id")
    return parent if isinstance(parent, str) else None


def ancestry_codes_to_chapter(
    client: WHOICD11Client,
    start_uri: str,
    *,
    max_hops: int,
    cache: DiskParentsCache,
) -> List[str]:
    hit = cache.get(start_uri)
    if hit is not None:
        return hit

    codes: List[str] = []
    cur = start_uri
    visited: Set[str] = set()

    for _ in range(max_hops):
        if not cur or cur in visited:
            break
        visited.add(cur)

        data = client.get_json(cur)
        if data.get("code"):
            codes.append(data["code"])

        if data.get("classKind") == "chapter":
            break

        cur = _first_parent_uri(data.get("parent"))
        if cur and cur.startswith("id."):
            cur = "https://" + cur  # rare but safe

    cache.put(start_uri, codes)
    return codes


def augment_set(
    items: List[Item],
    client: WHOICD11Client,
    *,
    include_self: bool,
    max_hops: int,
    cache: DiskParentsCache,
) -> Set[str]:
    out: Set[str] = set()
    for _code, uri, _title in items:
        path = ancestry_codes_to_chapter(client, uri, max_hops=max_hops, cache=cache)
        out.update(path if include_self else path[1:])
    return out


def hierarchical_prf_case(
    gt_items: List[Item],
    pred_items: List[Item],
    client: WHOICD11Client,
    *,
    include_self: bool,
    max_hops: int,
    cache: DiskParentsCache,
) -> Dict[str, Any]:
    C = augment_set(gt_items, client, include_self=include_self, max_hops=max_hops, cache=cache)
    Chat = augment_set(pred_items, client, include_self=include_self, max_hops=max_hops, cache=cache)
    inter = C & Chat

    HP = len(inter) / len(Chat) if Chat else 0.0
    HR = len(inter) / len(C) if C else 0.0
    HF1 = (2 * HP * HR / (HP + HR)) if (HP + HR) else 0.0

    return {
        "HP": HP,
        "HR": HR,
        "HF1": HF1,
        "C_size": float(len(C)),
        "Chat_size": float(len(Chat)),
        "Intersection_size": float(len(inter)),
        "C_codes_aug": sorted(C),
        "Chat_codes_aug": sorted(Chat),
        "C_intersection_aug": sorted(inter),
    }


# ==============
# File evaluation
# ==============

def evaluate(
    gt_path: Path,
    pred_path: Path,
    client: WHOICD11Client,
    *,
    out_dir: Path,
    include_self: bool = True,
    max_hops: int = 10,
    top_k: int = 20,
    skip_missing_preds: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    gt = read_jsonl(gt_path)
    pred = read_jsonl(pred_path)

    gt_by_id = {x["encounter_id"]: x for x in gt if "encounter_id" in x}
    pr_by_id = {x["encounter_id"]: x for x in pred if "encounter_id" in x}

    cache = DiskParentsCache("/Users/USER/Documents/Dermatology-Evaluation/ICDLens/icd11_ancestry_cache.json")

    per: List[Dict[str, Any]] = []
    missing_ids: List[str] = []

    for eid in sorted(gt_by_id):
        gt_obj = gt_by_id[eid]
        pr_obj = pr_by_id.get(eid)

        if pr_obj is None:
            missing_ids.append(eid)
            if skip_missing_preds:
                continue
            pr_obj = {}

        gt_codes = extract_codes(gt_obj)
        pr_codes = extract_codes(pr_obj)
        # TODO: make a new metric where you introduce weights based on the authors level of professionalism
        flat = flat_prf_if_any(gt_codes, pr_codes)
        overlap_any = 1.0 if flat["FlatInter"] > 0 else 0.0
        # overlap_any = 1.0 if (set(gt_codes) & set(pr_codes)) else 0.0

        gt_items = extract_items(gt_obj)
        pr_items = extract_items(pr_obj)

        if verbose:
            print(f"Evaluating {eid} (gt_codes={len(gt_codes)}, pred_codes={len(pr_codes)})")

        scores = hierarchical_prf_case(
            gt_items,
            pr_items,
            client,
            include_self=include_self,
            max_hops=max_hops,
            cache=cache,
        )

        per.append(
            {
                "encounter_id": eid,
                "HP": scores["HP"],
                "HR": scores["HR"],
                "HF1": scores["HF1"],
                "OverlapAny": overlap_any,
                "gt_codes_n": len(gt_codes),
                "pred_codes_n": len(pr_codes),
                "gt_items_n": len(gt_items),
                "pred_items_n": len(pr_items),
                "C_size": int(scores["C_size"]),
                "Chat_size": int(scores["Chat_size"]),
                "Intersection_size": int(scores["Intersection_size"]),
                "gt_codes": gt_codes,
                "pred_codes": pr_codes,
                
                "FlatP": flat["FlatP"],
                "FlatR": flat["FlatR"],
                "FlatF1_if_any": flat["FlatF1_if_any"],
                "FlatInter": int(flat["FlatInter"]),

                # NEW: write C and Chat into the output files
                "C_codes_aug": scores["C_codes_aug"],
                "Chat_codes_aug": scores["Chat_codes_aug"],
                "C_intersection_aug": scores["C_intersection_aug"],
            }
        )


    cache.flush()

    def mean(key: str) -> float:
        return sum(x[key] for x in per) / len(per) if per else 0.0

    best = sorted(per, key=lambda x: x["HF1"], reverse=True)[:top_k]
    worst = sorted(per, key=lambda x: x["HF1"])[:top_k]

    summary = {
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "ground_truth_path": str(gt_path),
            "predictions_path": str(pred_path),
            "ground_truth_sha256": sha256_file(gt_path),
            "predictions_sha256": sha256_file(pred_path),
        },
        "params": {
            "include_self": include_self,
            "max_hops": max_hops,
            "top_k": top_k,
            "skip_missing_preds": skip_missing_preds,
            "who_language": client.cfg.language,
        },
        "counts": {
            "n_gt": len(gt_by_id),
            "n_pred": len(pr_by_id),
            "n_evaluated": len(per),
            "missing_predictions": len(missing_ids),
            "cache_entries": cache.size,
        },
        "macro": {
        "HP": mean("HP"),
        "HR": mean("HR"),
        "HF1": mean("HF1"),
        "OverlapAny": mean("OverlapAny"),

        # NEW flat exact-match
        "FlatP": mean("FlatP"),
        "FlatR": mean("FlatR"),
        "FlatF1_if_any": mean("FlatF1_if_any"),
    },
        "best_cases": best,
        "worst_cases": worst,
        "missing_prediction_ids": missing_ids,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_jsonl(out_dir / "per_encounter.jsonl", per)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="ICD-11 hierarchical evaluation (HP/HR/HF1) + OverlapAny")
    ap.add_argument("--gt", required=True, type=Path, help="Ground-truth JSONL")
    ap.add_argument("--pred", required=True, type=Path, help="Predictions JSONL")
    ap.add_argument("--out_root", type=Path, default=None, help="Output root folder (default: predictions parent)")
    ap.add_argument("--cache", type=Path, default=None, help="Ancestry cache path (default: <out_dir>/icd11_ancestry_cache.json)")
    ap.add_argument("--max_hops", type=int, default=10, help="Max parent hops to chapter")
    ap.add_argument("--no_self", action="store_true", help="Exclude self code from augmented sets")
    ap.add_argument("--top_k", type=int, default=20, help="Top/bottom K cases to include in summary")
    ap.add_argument("--skip_missing_preds", action="store_true", help="Skip encounters not present in predictions file")
    ap.add_argument("--verbose", action="store_true", help="Print progress")
    ap.add_argument("--who_client_id", default=os.getenv("ICD11_CLIENT_ID", ""), help="WHO ICD-11 client id")
    ap.add_argument("--who_client_secret", default=os.getenv("ICD11_CLIENT_SECRET", ""), help="WHO ICD-11 client secret")
    ap.add_argument("--who_language", default=os.getenv("ICD11_LANG", "en"), help="WHO language header (default: en)")
    args = ap.parse_args()

    pred_path = args.pred.resolve()
    out_root = (args.out_root or pred_path.parent).resolve()
    out_dir = out_root / pred_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_path = args.cache or ("/Users/USER/Documents/Dermatology-Evaluation/ICDLens/icd11_ancestry_cache.json")

    cfg = WHOConfig(
        client_id=args.who_client_id,
        client_secret=args.who_client_secret,
        language=args.who_language,
    )
    client = WHOICD11Client(cfg)

    # wire the cache path into out_dir (so you can override with --cache)
    if cache_path != out_dir / "icd11_ancestry_cache.json":
        # keep a pointer for reproducibility
        (out_dir / "cache_path.txt").write_text(str(cache_path), encoding="utf-8")

    summary = evaluate(
        args.gt.resolve(),
        pred_path,
        client,
        out_dir=out_dir,
        include_self=not args.no_self,
        max_hops=args.max_hops,
        top_k=args.top_k,
        skip_missing_preds=args.skip_missing_preds,
        verbose=args.verbose,
    )

    print("Wrote:")
    print(" -", out_dir / "summary.json")
    print(" -", out_dir / "per_encounter.jsonl")
    print("Macro HF1/HP/HR/OverlapAny/FlatF1_if_any:",
      f"{summary['macro']['HF1']:.4f}",
      f"{summary['macro']['HP']:.4f}",
      f"{summary['macro']['HR']:.4f}",
      f"{summary['macro']['OverlapAny']:.4f}",
      f"{summary['macro']['FlatF1_if_any']:.4f}")

if __name__ == "__main__":
    main()
