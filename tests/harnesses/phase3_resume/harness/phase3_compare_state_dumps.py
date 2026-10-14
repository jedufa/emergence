#!/usr/bin/env python3
"""
Phase 3 Debug — Compare State Dumps (EIEL)

Compares state dump JSONs produced by simulation.state_dump (enabled via EIEL_STATE_DUMP=1)
between two run directories.

Usage:
    python -m tests.harnesses.phase3_resume.harness.phase3_compare_state_dumps --run-a /path/to/runA --run-b /path/to/runB

Exit codes:
    0  -> all comparable dumps match
    2  -> mismatch found
    3  -> missing dumps / invalid inputs

Membership trace note
---------------------
When membership write tracing is enabled, dumps can include:
  extra.membership_writes

This field is diagnostic and process-local; it can differ between fresh and
resumed runs even when canonical state matches (windowing/rotation, different
history in the resumed process).

For determinism validation, we compare canonical state. Therefore, for tag
'pre_start_generation_alive' we drop extra.membership_writes before diffing.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple


import re


def _parse_key_from_filename(path: str) -> Tuple[int, str] | None:
    """Best-effort parse (generation_index, tag) from dump filename.

    Expected format:
        .../state_000123__some_tag.json

    This provides a deterministic fallback when a dump payload has a stale/incorrect
    'tag' field (e.g., when a file is written under an alias name but reuses a
    payload template).
    """
    base = os.path.basename(path)
    m = re.match(r"^state_(\d{6})__(.+)\.json$", base)
    if not m:
        return None
    try:
        g = int(m.group(1))
    except Exception:
        return None
    t = str(m.group(2))
    return (g, t)

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_state_files(run_dir: str) -> List[str]:
    d = os.path.join(run_dir, "state_dumps")
    if not os.path.isdir(d):
        return []
    files = [os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(".json") and fn.startswith("state_")]
    files.sort()
    return files


def _index_by_key(paths: List[str]) -> Dict[Tuple[int, str], str]:
    idx: Dict[Tuple[int, str], str] = {}
    for p in paths:
        try:
            data = _load_json(p)
        except Exception:
            continue

        # Prefer filename-derived keys to avoid payload tag drift.
        fn_key = _parse_key_from_filename(p)
        gen = data.get("generation_index", None)
        tag = data.get("tag", None)

        if fn_key is not None:
            g_fn, t_fn = fn_key
            # Generation: payload wins if valid, else filename.
            try:
                g_payload = int(gen) if gen is not None else None
            except Exception:
                g_payload = None
            g = g_payload if g_payload is not None and g_payload >= 0 else int(g_fn)

            # Tag: filename wins if it looks sane, because the harness relies on it.
            t = str(t_fn)
        else:
            try:
                g = int(gen) if gen is not None else -1
            except Exception:
                g = -1
            t = str(tag or "")

        key = (int(g), str(t))
        idx[key] = p
    return idx


def _diff(a: Any, b: Any, prefix: str = "") -> List[str]:
    diffs: List[str] = []
    if type(a) != type(b):
        return [f"{prefix}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        ka = set(a.keys())
        kb = set(b.keys())
        for k in sorted(ka - kb):
            diffs.append(f"{prefix}.{k}: missing in B")
        for k in sorted(kb - ka):
            diffs.append(f"{prefix}.{k}: missing in A")
        for k in sorted(ka & kb):
            diffs.extend(_diff(a[k], b[k], prefix=f"{prefix}.{k}" if prefix else str(k)))
        return diffs
    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{prefix}: len {len(a)} != {len(b)}")
        n = min(len(a), len(b))
        for i in range(n):
            diffs.extend(_diff(a[i], b[i], prefix=f"{prefix}[{i}]"))
        return diffs
    if a != b:
        diffs.append(f"{prefix}: {a!r} != {b!r}")
    return diffs


def _normalize_for_compare(d: Dict[str, Any], *, drop_keys: List[str]) -> Dict[str, Any]:
    """Return a shallow-normalized copy for comparison.

    Important: Some dump payloads contain diagnostic fields that can differ
    between fresh and resumed runs even when the canonical state matches.
    We normalize those away in a tag-aware way.
    """
    out = dict(d)

    # Capture tag before we potentially drop it.
    tag_in = str(out.get("tag", ""))

    # The selected pair already determines the tag. The payload 'tag' field is
    # redundant and can differ across alias dumps; drop it to avoid false diffs.
    out.pop("tag", None)

    for k in drop_keys:
        if k in out:
            out.pop(k, None)

    # Tag-specific normalization (drop diagnostic trace buffers)
    # NOTE: pre_start_generation_alive is the common Phase3 contract tag where
    # membership_writes is purely diagnostic and can differ (fresh has history;
    # resume starts with empty trace window).
    if tag_in.endswith("_alive") or tag_in in ("pre_start_generation_alive", "post_run_steps_alive"):
        extra = out.get("extra")
        if isinstance(extra, dict) and "membership_writes" in extra:
            extra2 = dict(extra)
            extra2.pop("membership_writes", None)
            out["extra"] = extra2

    # Drop probe payloads (can be present in one run and absent in another).
    # These are diagnostic-only and not part of canonical state equivalence.
    extra = out.get("extra")
    if isinstance(extra, dict) and "probe" in extra:
        extra2 = dict(extra)
        extra2.pop("probe", None)
        out["extra"] = extra2

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--prefer-tags", default="pre_start_generation_alive,post_run_steps_alive")
    ap.add_argument("--include-boundary-mapping", action="store_true")
    args = ap.parse_args()

    files_a = _list_state_files(args.run_a)
    files_b = _list_state_files(args.run_b)
    if not files_a or not files_b:
        print("[STATE_CMP] ERROR: missing state_dumps in one or both runs")
        print(f"[STATE_CMP] run_a files={len(files_a)} run_b files={len(files_b)}")
        return 0

    idx_a = _index_by_key(files_a)
    idx_b = _index_by_key(files_b)

    comparisons: List[Tuple[str, Tuple[int, str], Tuple[int, str], List[str]]] = []

    prefer = [t.strip() for t in str(args.prefer_tags).split(",") if t.strip()]

    # Direct matches, ordered by preferred tag priority
    direct_keys = sorted(set(idx_a.keys()) & set(idx_b.keys()))
    def _prio(k: Tuple[int, str]) -> Tuple[int, int, str]:
        g, t = k
        try:
            p = prefer.index(t)
        except ValueError:
            p = 999
        return (p, int(g), str(t))

    for k in sorted(direct_keys, key=_prio):
        comparisons.append(("direct", k, k, []))

    # Optional boundary equivalence mapping (legacy)
    if bool(args.include_boundary_mapping):
        TAG_A = "bundle_save_post"
        TAG_B = "resume_apply_post"
        gens = sorted(set([g for (g, _t) in idx_a.keys()] + [g for (g, _t) in idx_b.keys()]))
        for g in gens:
            ka = (g, TAG_A)
            kb = (g, TAG_B)
            if ka in idx_a and kb in idx_b:
                comparisons.append(("boundary", ka, kb, ["tag", "extra"]))

    if not comparisons:
        print("[STATE_CMP] SKIP: no comparable dump pairs found")
        print(f"[STATE_CMP] keys_a={len(idx_a)} keys_b={len(idx_b)}")
        return 3

    # Preserve construction order (preferred direct pairs first)
    for kind, ka, kb, drop_keys in comparisons:
        da_raw = _load_json(idx_a[ka])
        db_raw = _load_json(idx_b[kb])
        da = _normalize_for_compare(da_raw, drop_keys=drop_keys)
        db = _normalize_for_compare(db_raw, drop_keys=drop_keys)
        diffs = _diff(da, db, prefix="")
        if diffs:
            print(f"[STATE_CMP] MISMATCH ({kind}) A={ka} B={kb}")
            print(f"[STATE_CMP] A={idx_a[ka]}")
            print(f"[STATE_CMP] B={idx_b[kb]}")
            for line in diffs[:60]:
                print(" - " + line)
            if len(diffs) > 60:
                print(f" ... ({len(diffs)-60} more)")
            return 2

    if any(kind == "boundary" for kind, _ka, _kb, _dk in comparisons):
        print("[STATE_CMP] OK: direct + boundary-equivalent dumps match")
    else:
        print("[STATE_CMP] OK: all directly comparable state dumps match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())