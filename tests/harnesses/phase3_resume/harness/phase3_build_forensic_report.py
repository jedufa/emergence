#!/usr/bin/env python3
"""scripts/phase3_build_forensic_report.py

Phase 3 helper: build a *zip-friendly* forensic report from two run folders.

Last modified: 2026-02-22 17:56 Europe/Vienna

Why this exists
---------------
Humans keep tossing gigantic run zips around and then acting surprised when
someone has to do 20+ manual parsing steps.

This script turns the most common Phase 3 "what diverged, where, and how" checks
into two small artifacts that can be embedded in the run zip:

- <out_dir>/_FORENSIC_REPORT.json
- <out_dir>/_FORENSIC_REPORT.md
- <out_dir>/_FORENSIC_ARTIFACTS/ (small pre-sliced context files)

It is intentionally dependency-free (stdlib only).

Inputs
------
--run-a PATH   Run directory for the "fresh" run
--run-b PATH   Run directory for the "resume" run

Optional
--------
--out-dir PATH     Where to write the report files (default: parent of run-a)
--label-a STR      Display label for run A (default: "fresh")
--label-b STR      Display label for run B (default: "resume")
--start-gen INT    Only compare probe/CSV overlap for generations >= start-gen.
                   If omitted, we auto-pick the first overlapping generation.

Report contents (best-effort)
-----------------------------
- Determinism fingerprint equality (byte + canonical hash)
- First divergence in metrics.csv and behaviors.csv (overlap-only)
- First mismatch in debug/determinism_probe.jsonl (tag/occurrence + which RNG hash key)
- Probe mismatch context window (surrounding tags + resync hints)
- Per-hash-key probe mismatch summary (first mismatch + mismatch count + resync point)
- EVOPY trace mismatch summary (if console logs include [EVOPY_CALL] lines)
- policy_slots.log first differing line (quick smell test)
- run_config.csv diff summary (when present)
- checkpoint_bundles_v2 inventory (when present)

Exit codes
----------
0  Report written (even if partial)
2  Could not write report (missing inputs or file IO error)
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sys as _sys

# -----------------------------------------------------------------------------
# Import-path stabilization
# -----------------------------------------------------------------------------
# This file is frequently executed as a script:
#   python phase3_checkpoint_resume_determinism/harness/phase3_build_forensic_report.py
# In that mode, sys.path[0] becomes "<repo>/scripts", which breaks `import scripts.*`.
# Force-add the repo root so sibling modules remain importable.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


# -----------------------------
# Small utilities
# -----------------------------

def _sha256_bytes(data: bytes) -> str:
    """Return hex SHA256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def _read_bytes(path: Path) -> Optional[bytes]:
    """Read bytes from path; return None on failure."""
    try:
        return path.read_bytes()
    except Exception:
        return None


def _read_text(path: Path, *, max_bytes: int = 5_000_000) -> Optional[str]:
    """Read text from path with a hard cap; return None on failure."""
    try:
        b = path.read_bytes()
        if len(b) > max_bytes:
            b = b[:max_bytes]
        return b.decode("utf-8", errors="replace")
    except Exception:
        return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file; return None on failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _canonical_json_bytes(obj: Any) -> bytes:
    """Serialize object into canonical JSON bytes (sorted keys, minified)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _utc_iso_now() -> str:
    """Return an ISO timestamp in UTC."""
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _short(v: Any, *, max_len: int = 80) -> Any:
    """Shorten long string-ish values for report friendliness."""
    if isinstance(v, str) and len(v) > max_len:
        return v[: max_len - 3] + "..."
    return v


# -----------------------------
# CSV divergence (overlap-only)
# -----------------------------


@dataclass(frozen=True)
class _CsvDivergence:
    """Represents first divergence in a CSV artifact."""

    artifact: str
    generation: Optional[int]
    reason: str


def _read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read CSV and return (fieldnames, rows) preserving raw string values."""
    with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [row for row in reader]
        return list(reader.fieldnames), rows


def _extract_generation_column(fieldnames: List[str]) -> str:
    """Return the generation column name if present."""
    for cand in ("generation", "gen"):
        if cand in fieldnames:
            return cand
    raise ValueError(f"No generation column found. Expected one of: generation, gen. Got: {fieldnames}")


def _group_by_gen(gen_col: str, rows: List[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    """Group rows by integer generation."""
    out: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        g = int(r[gen_col])
        out.setdefault(g, []).append(r)
    return out


def _first_metrics_divergence(path_a: Path, path_b: Path, *, start_gen: Optional[int]) -> _CsvDivergence:
    """Find first divergence in metrics.csv comparing only overlapping gens."""
    fna, rows_a = _read_csv_rows(path_a)
    fnb, rows_b = _read_csv_rows(path_b)

    ga = _extract_generation_column(fna)
    gb = _extract_generation_column(fnb)

    by_gen_a = {int(r[ga]): r for r in rows_a}
    by_gen_b = {int(r[gb]): r for r in rows_b}

    common_gens = sorted(set(by_gen_a.keys()) & set(by_gen_b.keys()))
    if start_gen is not None:
        common_gens = [g for g in common_gens if g >= start_gen]

    if not common_gens:
        return _CsvDivergence("metrics.csv", None, "no overlapping generations")

    cols_a = [c for c in fna if c != ga]
    cols_b = [c for c in fnb if c != gb]
    common_cols = [c for c in cols_a if c in set(cols_b)]

    for g in common_gens:
        ra = by_gen_a[g]
        rb = by_gen_b[g]
        diffs = [c for c in common_cols if ra.get(c, "") != rb.get(c, "")]
        if diffs:
            shown = diffs[:12]
            more = len(diffs) - len(shown)
            suffix = f" (+{more} more)" if more > 0 else ""
            return _CsvDivergence("metrics.csv", g, f"differing columns: {', '.join(shown)}{suffix}")

    return _CsvDivergence("metrics.csv", None, "no divergence in overlapping generations")


def _sort_behavior_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return a stable ordering for behaviors rows."""
    if not rows:
        return rows
    if "agent_id" in rows[0]:
        return sorted(rows, key=lambda r: int(r.get("agent_id", "-1")))
    if "slot" in rows[0]:
        return sorted(rows, key=lambda r: int(r.get("slot", "-1")))
    return sorted(rows, key=lambda r: tuple(sorted(r.items())))


def _first_behaviors_divergence(path_a: Path, path_b: Path, *, start_gen: Optional[int]) -> _CsvDivergence:
    """Find first divergence in behaviors.csv comparing only overlapping gens."""
    fna, rows_a = _read_csv_rows(path_a)
    fnb, rows_b = _read_csv_rows(path_b)

    ga = _extract_generation_column(fna)
    gb = _extract_generation_column(fnb)

    by_gen_a = _group_by_gen(ga, rows_a)
    by_gen_b = _group_by_gen(gb, rows_b)

    common_gens = sorted(set(by_gen_a.keys()) & set(by_gen_b.keys()))
    if start_gen is not None:
        common_gens = [g for g in common_gens if g >= start_gen]

    if not common_gens:
        return _CsvDivergence("behaviors.csv", None, "no overlapping generations")

    cols_a = [c for c in fna if c != ga]
    cols_b = [c for c in fnb if c != gb]
    common_cols = [c for c in cols_a if c in set(cols_b)]

    for g in common_gens:
        ga_rows = _sort_behavior_rows(by_gen_a[g])
        gb_rows = _sort_behavior_rows(by_gen_b[g])

        if len(ga_rows) != len(gb_rows):
            return _CsvDivergence(
                "behaviors.csv",
                g,
                f"row-shape differs: A={len(ga_rows)} rows, B={len(gb_rows)} rows",
            )

        for i, (ra, rb) in enumerate(zip(ga_rows, gb_rows)):
            diffs = [c for c in common_cols if ra.get(c, "") != rb.get(c, "")]
            if diffs:
                shown = diffs[:10]
                more = len(diffs) - len(shown)
                suffix = f" (+{more} more)" if more > 0 else ""
                anchor = f" (agent_id={ra.get('agent_id')})" if "agent_id" in ra else ""
                return _CsvDivergence(
                    "behaviors.csv",
                    g,
                    f"row {i}{anchor}: differing columns: {', '.join(shown)}{suffix}",
                )

    return _CsvDivergence("behaviors.csv", None, "no divergence in overlapping generations")


# -----------------------------
# determinism_probe.jsonl mismatch finder
# -----------------------------


def _read_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL records, ignoring parse errors.

    NOTE
    ----
    Do **not** use _read_text() here: determinism_probe/policy_exec_probe can exceed
    the global 5MB cap, and late mismatches must still be detectable.
    """
    out: List[Dict[str, Any]] = []

    # Preferred: stream line-by-line (bounded) to avoid truncation.
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 500_000:  # hard safety cap
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        out.append(rec)
                except Exception:
                    continue
        return out
    except Exception:
        pass

    # Fallback: best-effort bounded read.
    txt = _read_text(path, max_bytes=50_000_000)
    if txt is None:
        return out
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            if isinstance(rec, dict):
                out.append(rec)
        except Exception:
            continue
    return out



def _probe_min_gen(recs: List[Dict[str, Any]]) -> Optional[int]:
    """Return minimum generation_index observed in probe records."""
    gens: List[int] = []
    for r in recs:
        g = r.get("generation_index")
        if isinstance(g, int):
            gens.append(g)
        elif isinstance(g, str) and g.isdigit():
            gens.append(int(g))
    return min(gens) if gens else None


def _make_occurrence_seq(recs: List[Dict[str, Any]], *, gen_min: int) -> List[Tuple[int, str, int, Dict[str, Any]]]:
    """Create ordered (gen, tag, occurrence_index, record) sequence."""
    occ: Dict[Tuple[int, str], int] = {}
    seq: List[Tuple[int, str, int, Dict[str, Any]]] = []
    for r in recs:
        g = r.get("generation_index")
        if g is None:
            continue
        try:
            gi = int(g)
        except Exception:
            continue
        if gi < gen_min:
            continue
        tag = str(r.get("tag", ""))
        k = (gi, tag)
        occ[k] = occ.get(k, 0) + 1
        seq.append((gi, tag, occ[k], r))
    return seq



# Probe diff ignore rules (Phase 3)
# We intentionally ignore "*_full_sha256" fields when determining the first actionable mismatch.
# Those fields are debug-only superset hashes that can include irrelevant cache rows (e.g., MLP rows for GRU slots),
# and therefore can differ without affecting behavior or the determinism_fingerprint / CSV outputs.
_IGNORE_PROBE_HASH_KEYS_SUFFIXES = ("_full_sha256",)

def _probe_hash_key_is_ignored(k: str) -> bool:
    """Return True if probe hash key should be ignored for mismatch purposes."""
    try:
        return any(k.endswith(suf) for suf in _IGNORE_PROBE_HASH_KEYS_SUFFIXES)
    except Exception:
        return False


def _diff_hashes(ha: Dict[str, Any], hb: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-key diffs between two hashes dicts."""
    diff_keys = sorted(set(ha.keys()) | set(hb.keys()))
    diffs: Dict[str, Any] = {}
    for k in diff_keys:
        if _probe_hash_key_is_ignored(str(k)):
            continue
        if ha.get(k) != hb.get(k):
            diffs[k] = {"a": _short(ha.get(k)), "b": _short(hb.get(k))}
    return diffs


def _first_probe_hash_mismatch(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    *,
    start_gen: int,
) -> Optional[Dict[str, Any]]:
    """Return a dict describing the first probe mismatch, or None."""
    seq_a = _make_occurrence_seq(recs_a, gen_min=start_gen)
    seq_b = _make_occurrence_seq(recs_b, gen_min=start_gen)

    b_map: Dict[Tuple[int, str, int], Dict[str, Any]] = {(g, t, n): r for (g, t, n, r) in seq_b}

    for idx, (g, t, n, ra) in enumerate(seq_a):
        key = (g, t, n)
        rb = b_map.get(key)
        if rb is None:
            continue
        ha = ra.get("hashes")
        hb = rb.get("hashes")
        if not isinstance(ha, dict) or not isinstance(hb, dict):
            continue
        if ha != hb:
            diffs = _diff_hashes(ha, hb)
            # If the only differences are in ignored keys (e.g. *_full_sha256),
            # treat this as "no actionable mismatch" and keep scanning.
            if not diffs:
                continue
            return {
                "generation_index": g,
                "tag": t,
                "occurrence": n,
                "index_in_a": idx,
                "diffs": diffs,
            }

    return None



def _first_probe_hash_mismatch_fallback(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    *,
    start_gen: int,
) -> Optional[Dict[str, Any]]:
    """Fallback mismatch finder that aligns by (gen, tag) only.

    Why: if any producer accidentally drops/duplicates a record for the same (gen,tag),
    the occurrence-based alignment can skip real mismatches. This fallback is slower
    but more robust and is only used when the primary method reports no mismatch.
    """
    # Build maps keyed by (gen, tag) -> record. If duplicates exist, keep the first
    # to preserve "earliest" semantics.
    amap: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for r in recs_a:
        g = r.get("generation_index")
        try:
            gi = int(g)
        except Exception:
            continue
        if gi < start_gen:
            continue
        tag = str(r.get("tag", ""))
        k = (gi, tag)
        if k not in amap:
            amap[k] = r

    bmap: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for r in recs_b:
        g = r.get("generation_index")
        try:
            gi = int(g)
        except Exception:
            continue
        if gi < start_gen:
            continue
        tag = str(r.get("tag", ""))
        k = (gi, tag)
        if k not in bmap:
            bmap[k] = r

    # Walk in generation order.
    for (gi, tag) in sorted(set(amap.keys()) & set(bmap.keys())):
        ra = amap[(gi, tag)]
        rb = bmap[(gi, tag)]
        ha = ra.get("hashes")
        hb = rb.get("hashes")
        if not isinstance(ha, dict) or not isinstance(hb, dict):
            continue
        if ha != hb:
            diffs = _diff_hashes(ha, hb)
            # If the only differences are in ignored keys, keep scanning.
            if not diffs:
                continue
            return {
                "generation_index": gi,
                "tag": tag,
                "occurrence": 1,
                "index_in_a": None,
                "diffs": diffs,
                "note": "fallback_alignment",
            }
    return None


def _probe_key_summary(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    *,
    start_gen: int,
    max_keys: int = 50,
) -> Dict[str, Any]:
    """Summarize per-hash-key probe mismatches (first mismatch + count + resync)."""
    seq_a = _make_occurrence_seq(recs_a, gen_min=start_gen)
    seq_b = _make_occurrence_seq(recs_b, gen_min=start_gen)
    b_map: Dict[Tuple[int, str, int], Dict[str, Any]] = {(g, t, n): r for (g, t, n, r) in seq_b}

    first: Dict[str, Dict[str, Any]] = {}
    counts: Dict[str, int] = {}
    resync: Dict[str, Dict[str, Any]] = {}
    mismatched_since: Dict[str, int] = {}

    # Ignored-key tracking (e.g. *_full_sha256). These are recorded for diagnostics,
    # but they do NOT contribute to the "actionable mismatch" decision.
    first_ignored: Dict[str, Dict[str, Any]] = {}
    counts_ignored: Dict[str, int] = {}
    resync_ignored: Dict[str, Dict[str, Any]] = {}
    mismatched_since_ignored: Dict[str, int] = {}

    missing_in_b = 0

    for idx, (g, t, n, ra) in enumerate(seq_a):
        rb = b_map.get((g, t, n))
        if rb is None:
            missing_in_b += 1
            continue

        ha = ra.get("hashes")
        hb = rb.get("hashes")
        if not isinstance(ha, dict) or not isinstance(hb, dict):
            continue

        all_keys = set(ha.keys()) | set(hb.keys())

        for k in all_keys:
            k_str = str(k)
            av = ha.get(k)
            bv = hb.get(k)
            is_ignored = _probe_hash_key_is_ignored(k_str)

            if av != bv:
                if is_ignored:
                    counts_ignored[k_str] = counts_ignored.get(k_str, 0) + 1
                    if k_str not in first_ignored:
                        first_ignored[k_str] = {
                            "generation_index": g,
                            "tag": t,
                            "occurrence": n,
                            "index_in_a": idx,
                            "a": _short(av),
                            "b": _short(bv),
                        }
                        mismatched_since_ignored[k_str] = idx
                else:
                    counts[k_str] = counts.get(k_str, 0) + 1
                    if k_str not in first:
                        first[k_str] = {
                            "generation_index": g,
                            "tag": t,
                            "occurrence": n,
                            "index_in_a": idx,
                            "a": _short(av),
                            "b": _short(bv),
                        }
                        mismatched_since[k_str] = idx
            else:
                if is_ignored:
                    if k_str in mismatched_since_ignored and k_str not in resync_ignored:
                        resync_ignored[k_str] = {
                            "generation_index": g,
                            "tag": t,
                            "occurrence": n,
                            "index_in_a": idx,
                        }
                else:
                    if k_str in mismatched_since and k_str not in resync:
                        resync[k_str] = {
                            "generation_index": g,
                            "tag": t,
                            "occurrence": n,
                            "index_in_a": idx,
                        }

    # Sort by earliest mismatch index, then key name.
    items = sorted(first.items(), key=lambda kv: (kv[1].get("index_in_a", 10**9), kv[0]))
    ignored_items = sorted(
        first_ignored.items(), key=lambda kv: (kv[1].get("index_in_a", 10**9), kv[0])
    )

    keys_out: List[Dict[str, Any]] = []
    for k, info in items[:max_keys]:
        keys_out.append(
            {
                "key": k,
                "mismatch_count": counts.get(k, 0),
                "first_mismatch": info,
                "resync_at": resync.get(k),
            }
        )

    ignored_keys_out: List[Dict[str, Any]] = []
    for k, info in ignored_items[:max_keys]:
        ignored_keys_out.append(
            {
                "key": k,
                "mismatch_count": counts_ignored.get(k, 0),
                "first_mismatch": info,
                "resync_at": resync_ignored.get(k),
            }
        )

    return {
        "missing_records_in_b": missing_in_b,
        "total_keys_with_mismatch": len(first),
        "keys": keys_out,
        "ignored_total_keys_with_mismatch": len(first_ignored),
        "ignored_keys": ignored_keys_out,
    }


def _probe_context_window(
    recs_a: List[Dict[str, Any]],
    recs_b: List[Dict[str, Any]],
    *,
    start_gen: int,
    mismatch: Optional[Dict[str, Any]],
    window: int = 6,
) -> Optional[Dict[str, Any]]:
    """Return a small window of probe records around the first mismatch index."""
    if not mismatch:
        return None
    idx0 = mismatch.get("index_in_a")
    if not isinstance(idx0, int):
        return None

    seq_a = _make_occurrence_seq(recs_a, gen_min=start_gen)
    seq_b = _make_occurrence_seq(recs_b, gen_min=start_gen)
    b_map: Dict[Tuple[int, str, int], Dict[str, Any]] = {(g, t, n): r for (g, t, n, r) in seq_b}

    lo = max(0, idx0 - window)
    hi = min(len(seq_a), idx0 + window + 1)

    rows: List[Dict[str, Any]] = []
    for idx in range(lo, hi):
        g, t, n, ra = seq_a[idx]
        rb = b_map.get((g, t, n))
        ha = ra.get("hashes") if isinstance(ra.get("hashes"), dict) else {}
        hb = rb.get("hashes") if (rb and isinstance(rb.get("hashes"), dict)) else {}
        diffs = _diff_hashes(ha, hb) if (isinstance(ha, dict) and isinstance(hb, dict)) else {}
        rows.append(
            {
                "index_in_a": idx,
                "generation_index": g,
                "tag": t,
                "occurrence": n,
                "diff_keys": list(diffs.keys())[:12],
                "rng_manager_snapshot_a": _short(ha.get("rng_manager_snapshot")),
                "rng_manager_snapshot_b": _short(hb.get("rng_manager_snapshot")),
            }
        )

    # Resync hint for rng_manager_snapshot specifically.
    resync_at = None
    for idx in range(idx0 + 1, len(seq_a)):
        g, t, n, ra = seq_a[idx]
        rb = b_map.get((g, t, n))
        if rb is None:
            continue
        ha = ra.get("hashes")
        hb = rb.get("hashes")
        if not isinstance(ha, dict) or not isinstance(hb, dict):
            continue
        if ha.get("rng_manager_snapshot") == hb.get("rng_manager_snapshot"):
            resync_at = {"index_in_a": idx, "generation_index": g, "tag": t, "occurrence": n}
            break

    return {
        "mismatch_index_in_a": idx0,
        "range": {"lo": lo, "hi": hi},
        "rows": rows,
        "rng_manager_snapshot_resync_at": resync_at,
    }


# -----------------------------
# policy_slots.log quick diff
# -----------------------------


def _first_line_diff(path_a: Path, path_b: Path, *, max_lines: int = 200_000) -> Optional[Dict[str, Any]]:
    """Return first differing line between two text files (best-effort)."""
    try:
        # Fast-path: tolerate common "fresh has extra prefix lines" cases by checking for an
        # exact *byte suffix* match first. This happens for debug logs that only diverge in
        # their early, pre-resume prefix.
        try:
            sa = int(path_a.stat().st_size)
            sb = int(path_b.stat().st_size)

            def _is_b_suffix_of_a(pa: Path, pb: Path) -> bool:
                sa0 = int(pa.stat().st_size)
                sb0 = int(pb.stat().st_size)
                if sb0 > sa0:
                    return False
                off = sa0 - sb0
                # Stream-compare to avoid slurping large logs into RAM.
                with pa.open("rb") as fa, pb.open("rb") as fb:
                    fa.seek(off)
                    while True:
                        ba = fa.read(1024 * 1024)
                        bb = fb.read(1024 * 1024)
                        if not ba and not bb:
                            return True
                        if ba != bb:
                            return False

            if sb > 0 and sa >= sb and _is_b_suffix_of_a(path_a, path_b):
                return {
                    "equal_after_alignment": True,
                    "alignment": {
                        "kind": "b_is_suffix_of_a",
                        "a_prefix_bytes": int(sa - sb),
                        "a_total_bytes": int(sa),
                        "b_total_bytes": int(sb),
                    },
                }
            if sa > 0 and sb >= sa and _is_b_suffix_of_a(path_b, path_a):
                return {
                    "equal_after_alignment": True,
                    "alignment": {
                        "kind": "a_is_suffix_of_b",
                        "b_prefix_bytes": int(sb - sa),
                        "a_total_bytes": int(sa),
                        "b_total_bytes": int(sb),
                    },
                }
        except Exception:
            # If suffix matching fails for any reason, fall back to line-by-line diff.
            pass

        with path_a.open("r", encoding="utf-8", errors="replace") as fa, path_b.open(
            "r", encoding="utf-8", errors="replace"
        ) as fb:
            i = 0
            while i < max_lines:
                la = fa.readline()
                lb = fb.readline()
                if not la and not lb:
                    return None
                if la != lb:
                    return {
                        "line_number": i,
                        "a": la.rstrip("\n"),
                        "b": lb.rstrip("\n"),
                    }
                i += 1
    except Exception:
        return None
    return None


# -----------------------------
# run_config.csv diff summary (optional)
# -----------------------------


def _read_run_config(path: Path) -> Dict[str, str]:
    """Read run_config.csv into a dict(key->value) best-effort."""
    out: Dict[str, str] = {}
    try:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return out
            # Expected columns: key,value (but we tolerate other column names)
            key_col = "key" if "key" in reader.fieldnames else (reader.fieldnames[0] if reader.fieldnames else "key")
            val_col = "value" if "value" in reader.fieldnames else (
                reader.fieldnames[1] if reader.fieldnames and len(reader.fieldnames) > 1 else "value"
            )
            for row in reader:
                k = str(row.get(key_col, "")).strip()
                if not k:
                    continue
                if k not in out:
                    out[k] = str(row.get(val_col, "")).strip()
    except Exception:
        return out
    return out


def _diff_run_config(cfg_a: Dict[str, str], cfg_b: Dict[str, str], *, max_items: int = 40) -> Dict[str, Any]:
    """Compute key/value diffs between two run_config dicts."""
    keys = sorted(set(cfg_a.keys()) | set(cfg_b.keys()))
    diffs: List[Dict[str, Any]] = []
    for k in keys:
        av = cfg_a.get(k)
        bv = cfg_b.get(k)
        if av != bv:
            diffs.append({"key": k, "a": _short(av), "b": _short(bv)})
            if len(diffs) >= max_items:
                break
    return {"diff_count": len([k for k in keys if cfg_a.get(k) != cfg_b.get(k)]), "sample": diffs}


# -----------------------------
# checkpoint bundle inventory (optional)
# -----------------------------


def _bundle_inventory(run_dir: Path) -> Dict[str, Any]:
    """Inventory checkpoint_bundles_v2 bundles present."""
    root = run_dir / "checkpoint_bundles_v2"
    if not root.exists() or not root.is_dir():
        return {"present": False, "bundles": []}
    bundles = sorted([p.name for p in root.glob("bundle_gen*") if p.is_dir()])
    gens: List[int] = []
    for b in bundles:
        m = b.replace("bundle_gen", "")
        if m.isdigit():
            gens.append(int(m))
    inv: Dict[str, Any] = {"present": True, "bundles": bundles}
    if gens:
        inv.update({"min_gen": min(gens), "max_gen": max(gens), "count": len(gens)})
    else:
        inv.update({"min_gen": None, "max_gen": None, "count": 0})
    return inv


# -----------------------------
# EVOPY compare (optional)
# -----------------------------


def _try_evopy_compare(console_a: Optional[Path], console_b: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Compare EVOPY traces inside console logs if possible."""
    if console_a is None or console_b is None:
        return None
    try:
        # Local import to keep this script usable even if scripts module changes.
        from tests.harnesses.phase3_resume.harness.phase3_compare_evopy_calls import compare_logs  # type: ignore

        code, msg = compare_logs(str(console_a), str(console_b), epoch="boundary")
        out: Dict[str, Any] = {"exit_code": int(code), "message": str(msg)}
        return out
    except Exception as e:
        return {"exit_code": 99, "message": f"EVOPY compare error: {type(e).__name__}: {e}"}


def _find_console_log(run_dir: Path, prefer_contains: str) -> Optional[Path]:
    """Pick a console_*.log file from run_dir, optionally preferring a substring."""
    cands = sorted(run_dir.glob("console_*.log"))
    if not cands:
        return None
    for p in cands:
        if prefer_contains in p.name:
            return p
    return cands[0]


# -----------------------------
# Main report builder
# -----------------------------


def _read_text_context(
    path: Path,
    center_line: int,
    radius: int = 6,
) -> Optional[Dict[str, Any]]:
    """Read a small context window from a text file.

    Parameters:
    - path: input file path
    - center_line: 0-based line number that should be centered in the window
    - radius: number of lines to include before/after the center

    Returns:
    - dict with start_line, end_line, lines (each {line_number, text})
      or None if the file does not exist.
    """
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    if center_line < 0:
        center_line = 0
    start = max(0, center_line - int(radius))
    end = min(len(lines) - 1, center_line + int(radius))
    out_lines: List[Dict[str, Any]] = []
    for ln in range(start, end + 1):
        out_lines.append({"line_number": int(ln), "text": lines[ln]})
    return {"start_line": int(start), "end_line": int(end), "lines": out_lines}


def _policy_exec_seq(
    records: List[Dict[str, Any]],
    start_gen: int,
) -> List[Dict[str, Any]]:
    """Normalize policy_exec_probe records into a stable, comparable sequence.

    Each output row contains:
    - index_a: position in the filtered sequence
    - key: (gen, tick, stage, k, occurrence)
    - gen/tick/stage/k/occurrence (redundant but readable)
    - digests/samples/extra passthrough

    We include an occurrence counter because stages can repeat within a tick.
    """
    out: List[Dict[str, Any]] = []
    occ: Dict[Tuple[int, int, str, int], int] = {}
    for rec in records:
        try:
            gen = int(rec.get("gen", -1))
            if gen < start_gen:
                continue
            tick = int(rec.get("tick", -1))
            stage = str(rec.get("stage", ""))
            k = int(rec.get("k", 0))
            base = (gen, tick, stage, k)
            occ[base] = occ.get(base, 0) + 1
            key = (gen, tick, stage, k, occ[base])
            out.append(
                {
                    "index": int(len(out)),
                    "key": key,
                    "gen": gen,
                    "tick": tick,
                    "stage": stage,
                    "k": k,
                    "occurrence": int(occ[base]),
                    "digests": rec.get("digests", {}) or {},
                    "samples": rec.get("samples", {}) or {},
                    "extra": rec.get("extra", {}) or {},
                }
            )
        except Exception:
            continue
    return out


def _policy_exec_first_mismatch(
    seq_a: List[Dict[str, Any]],
    seq_b: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the first policy_exec_probe mismatch between A and B.

    Comparison is keyed by (gen, tick, stage, k, occurrence). For each aligned record:
    - if missing in B => mismatch
    - else compare digests dict, key-wise

    Returns a dict describing the first mismatch, or None if no mismatch is found.
    """
    b_map: Dict[Tuple[int, int, str, int, int], Dict[str, Any]] = {row["key"]: row for row in seq_b}

    for row_a in seq_a:
        key = row_a["key"]
        row_b = b_map.get(key)
        if row_b is None:
            return {
                "index_a": row_a["index"],
                "index_b": None,
                "gen": row_a["gen"],
                "tick": row_a["tick"],
                "stage": row_a["stage"],
                "k": row_a["k"],
                "occurrence": row_a["occurrence"],
                "missing_in_b": True,
                "diff_digest_keys": ["<missing_record_in_b>"],
                "digests_a": row_a.get("digests", {}) or {},
                "digests_b": {},
            }

        dig_a = row_a.get("digests", {}) or {}
        dig_b = row_b.get("digests", {}) or {}
        keys = sorted(set(dig_a.keys()) | set(dig_b.keys()))
        diff_keys: List[str] = []
        for k2 in keys:
            if dig_a.get(k2) != dig_b.get(k2):
                diff_keys.append(k2)

        if diff_keys:
            return {
                "index_a": row_a["index"],
                "index_b": row_b["index"],
                "gen": row_a["gen"],
                "tick": row_a["tick"],
                "stage": row_a["stage"],
                "k": row_a["k"],
                "occurrence": row_a["occurrence"],
                "missing_in_b": False,
                "diff_digest_keys": diff_keys,
                "digests_a": {k2: dig_a.get(k2) for k2 in diff_keys},
                "digests_b": {k2: dig_b.get(k2) for k2 in diff_keys},
                "samples_a": row_a.get("samples", {}) or {},
                "samples_b": row_b.get("samples", {}) or {},
            }

    return None


def _policy_exec_context_window(
    seq_a: List[Dict[str, Any]],
    seq_b: List[Dict[str, Any]],
    mismatch: Dict[str, Any],
    radius: int = 6,
) -> Optional[List[Dict[str, Any]]]:
    """Build a context window around the first policy_exec_probe mismatch."""
    idx0 = mismatch.get("index_a")
    if idx0 is None:
        return None
    try:
        idx0 = int(idx0)
    except Exception:
        return None

    b_map: Dict[Tuple[int, int, str, int, int], Dict[str, Any]] = {row["key"]: row for row in seq_b}
    lo = max(0, idx0 - int(radius))
    hi = min(len(seq_a) - 1, idx0 + int(radius))

    preferred = [
        "actions_out",
        "logits",
        "torch_cpu_rng",
        "torch_cuda_rng",
        "alive_ids",
        "mlp_ids",
        "gru_ids",
        "mlp_slots",
        "gru_slots",
        "mlp_tokens",
        "gru_tokens",
    ]

    out: List[Dict[str, Any]] = []
    for i in range(lo, hi + 1):
        row_a = seq_a[i]
        row_b = b_map.get(row_a["key"])
        dig_a = row_a.get("digests", {}) or {}
        dig_b = (row_b.get("digests", {}) or {}) if row_b else {}

        diff_keys = sorted({k for k in set(dig_a.keys()) | set(dig_b.keys()) if dig_a.get(k) != dig_b.get(k)})
        present = [k for k in preferred if (k in dig_a) or (k in dig_b)]

        out.append(
            {
                "index_a": int(row_a["index"]),
                "index_b": int(row_b["index"]) if row_b else None,
                "gen": int(row_a["gen"]),
                "tick": int(row_a["tick"]),
                "stage": str(row_a["stage"]),
                "k": int(row_a["k"]),
                "occurrence": int(row_a["occurrence"]),
                "diff_digest_keys": diff_keys,
                "digests_a": {k: dig_a.get(k) for k in present},
                "digests_b": {k: dig_b.get(k) for k in present},
                "samples_a": row_a.get("samples", {}) or {},
                "samples_b": (row_b.get("samples", {}) or {}) if row_b else {},
            }
        )
    return out


def _policy_exec_digest_summary(
    seq_a: List[Dict[str, Any]],
    seq_b: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize mismatches across policy_exec_probe digest keys."""
    b_map: Dict[Tuple[int, int, str, int, int], Dict[str, Any]] = {row["key"]: row for row in seq_b}

    union_keys: set = set()
    mismatch_counts: Dict[str, int] = {}
    missing_records_in_b = 0

    for row_a in seq_a:
        row_b = b_map.get(row_a["key"])
        if row_b is None:
            missing_records_in_b += 1
            continue
        dig_a = row_a.get("digests", {}) or {}
        dig_b = row_b.get("digests", {}) or {}
        union = set(dig_a.keys()) | set(dig_b.keys())
        union_keys |= union
        for k in union:
            if dig_a.get(k) != dig_b.get(k):
                mismatch_counts[k] = mismatch_counts.get(k, 0) + 1

    top = sorted(mismatch_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "missing_records_in_b": int(missing_records_in_b),
        "digests_present": sorted(union_keys),
        "total_digest_keys_with_mismatch": int(len(mismatch_counts)),
        "top_mismatching_digests": [{"key": k, "mismatch_count": int(c)} for k, c in top[:20]],
        "has_actions_out_digest": bool("actions_out" in union_keys),
        "has_logits_digest": bool("logits" in union_keys),
        "has_slots_digests": bool(("mlp_slots" in union_keys) or ("gru_slots" in union_keys)),
    }


def _compare_policy_exec_probe(
    run_a_dir: Path,
    run_b_dir: Path,
    start_gen: int,
) -> Dict[str, Any]:
    """Compare policy_exec_probe.jsonl between two runs.

    The intent is not to replicate determinism_probe. It's to answer:
    - Do policy outputs diverge (actions/logits)?
    - If so, *where* (stage, tick) does that happen?

    Returns a dict with:
    - status: ok/missing
    - first_mismatch: detailed info
    - context_window: records around mismatch (digest + small samples)
    - digest_summary: aggregate mismatches + presence of key digest types
    """
    pe_a = run_a_dir / "debug" / "policy_exec_probe.jsonl"
    pe_b = run_b_dir / "debug" / "policy_exec_probe.jsonl"

    if not pe_a.exists() or not pe_b.exists():
        return {
            "status": "missing",
            "path_a": str(pe_a),
            "path_b": str(pe_b),
            "records_a": int(0 if not pe_a.exists() else sum(1 for _ in pe_a.open("r", encoding="utf-8", errors="replace"))),
            "records_b": int(0 if not pe_b.exists() else sum(1 for _ in pe_b.open("r", encoding="utf-8", errors="replace"))),
            "note": "One or both policy_exec_probe.jsonl files are missing.",
        }

    recs_a = _read_jsonl_records(pe_a)
    recs_b = _read_jsonl_records(pe_b)

    # policy_exec_probe uses 'gen' not 'generation_index'
    gen_min_a = min((int(r.get("gen", 10**9)) for r in recs_a if isinstance(r, dict) and "gen" in r), default=10**9)
    gen_min_b = min((int(r.get("gen", 10**9)) for r in recs_b if isinstance(r, dict) and "gen" in r), default=10**9)
    eff_start_gen = max(int(start_gen), int(min(gen_min_a, gen_min_b)) if min(gen_min_a, gen_min_b) != 10**9 else int(start_gen))

    seq_a = _policy_exec_seq(recs_a, eff_start_gen)
    seq_b = _policy_exec_seq(recs_b, eff_start_gen)

    mismatch = _policy_exec_first_mismatch(seq_a, seq_b)
    ctx = _policy_exec_context_window(seq_a, seq_b, mismatch) if mismatch else None
    summary = _policy_exec_digest_summary(seq_a, seq_b)

    return {
        "status": "ok",
        "schema_note": "policy_exec_probe_v1/v2 tolerated; comparison uses digest keys present.",
        "path_a": str(pe_a),
        "path_b": str(pe_b),
        "records_a": int(len(seq_a)),
        "records_b": int(len(seq_b)),
        "effective_start_gen": int(eff_start_gen),
        "first_mismatch": mismatch,
        "context_window": ctx,
        "digest_summary": summary,
    }



def build_report(
    run_a: Path,
    run_b: Path,
    *,
    label_a: str,
    label_b: str,
    start_gen: Optional[int],
) -> Dict[str, Any]:
    """Build the forensic report payload (dict)."""

    # Fingerprints
    fp_a_path = run_a / "determinism_fingerprint.json"
    fp_b_path = run_b / "determinism_fingerprint.json"
    fp_a_obj = _read_json(fp_a_path) if fp_a_path.exists() else None
    fp_b_obj = _read_json(fp_b_path) if fp_b_path.exists() else None

    fp_a_file_bytes = _read_bytes(fp_a_path) if fp_a_path.exists() else None
    fp_b_file_bytes = _read_bytes(fp_b_path) if fp_b_path.exists() else None

    fp_a_file_sha = _sha256_bytes(fp_a_file_bytes) if fp_a_file_bytes is not None else None
    fp_b_file_sha = _sha256_bytes(fp_b_file_bytes) if fp_b_file_bytes is not None else None

    fp_a_canon_sha = _sha256_bytes(_canonical_json_bytes(fp_a_obj)) if fp_a_obj is not None else None
    fp_b_canon_sha = _sha256_bytes(_canonical_json_bytes(fp_b_obj)) if fp_b_obj is not None else None

    fingerprints = {
        "a": {
            "path": str(fp_a_path) if fp_a_path.exists() else None,
            "file_sha256": fp_a_file_sha,
            "canonical_sha256": fp_a_canon_sha,
            "status": (fp_a_obj or {}).get("status") if isinstance(fp_a_obj, dict) else None,
        },
        "b": {
            "path": str(fp_b_path) if fp_b_path.exists() else None,
            "file_sha256": fp_b_file_sha,
            "canonical_sha256": fp_b_canon_sha,
            "status": (fp_b_obj or {}).get("status") if isinstance(fp_b_obj, dict) else None,
        },
        "byte_equal": (fp_a_file_bytes == fp_b_file_bytes) if (fp_a_file_bytes and fp_b_file_bytes) else None,
        "canonical_equal": (fp_a_canon_sha == fp_b_canon_sha) if (fp_a_canon_sha and fp_b_canon_sha) else None,
    }

    # CSV divergence
    metrics_a = run_a / "metrics.csv"
    metrics_b = run_b / "metrics.csv"
    beh_a = run_a / "behaviors.csv"
    beh_b = run_b / "behaviors.csv"

    csv_div: Dict[str, Any] = {}
    if metrics_a.exists() and metrics_b.exists():
        try:
            res = _first_metrics_divergence(metrics_a, metrics_b, start_gen=start_gen)
            csv_div["metrics.csv"] = {"generation": res.generation, "reason": res.reason}
        except Exception as e:
            csv_div["metrics.csv"] = {"generation": None, "reason": f"error: {type(e).__name__}: {e}"}
    else:
        csv_div["metrics.csv"] = {"generation": None, "reason": "missing"}

    if beh_a.exists() and beh_b.exists():
        try:
            res = _first_behaviors_divergence(beh_a, beh_b, start_gen=start_gen)
            csv_div["behaviors.csv"] = {"generation": res.generation, "reason": res.reason}
        except Exception as e:
            csv_div["behaviors.csv"] = {"generation": None, "reason": f"error: {type(e).__name__}: {e}"}
    else:
        csv_div["behaviors.csv"] = {"generation": None, "reason": "missing"}

    # determinism_probe mismatch
    probe_a_path = run_a / "debug" / "determinism_probe.jsonl"
    probe_b_path = run_b / "debug" / "determinism_probe.jsonl"

    recs_a = _read_jsonl_records(probe_a_path) if probe_a_path.exists() else []
    recs_b = _read_jsonl_records(probe_b_path) if probe_b_path.exists() else []

    auto_start_gen = None
    if start_gen is None:
        min_a = _probe_min_gen(recs_a)
        min_b = _probe_min_gen(recs_b)
        if min_a is not None and min_b is not None:
            auto_start_gen = max(min_a, min_b)

    eff_start_gen = start_gen if start_gen is not None else (auto_start_gen if auto_start_gen is not None else 0)

    probe_mismatch = None
    probe_key_summary = None
    probe_context = None
    if recs_a and recs_b:
        probe_mismatch = _first_probe_hash_mismatch(recs_a, recs_b, start_gen=eff_start_gen)
        # Defensive fallback: if the primary occurrence-aligned method claims "no mismatch",
        # run a (gen,tag)-aligned pass to avoid false negatives.
        if probe_mismatch is None:
            probe_mismatch = _first_probe_hash_mismatch_fallback(recs_a, recs_b, start_gen=eff_start_gen)
        probe_key_summary = _probe_key_summary(recs_a, recs_b, start_gen=eff_start_gen)
        probe_context = _probe_context_window(recs_a, recs_b, start_gen=eff_start_gen, mismatch=probe_mismatch)

    probe_status = "ok" if (probe_a_path.exists() and probe_b_path.exists()) else "missing"

    probe_block = {
        "path_a": str(probe_a_path) if probe_a_path.exists() else None,
        "path_b": str(probe_b_path) if probe_b_path.exists() else None,
        "status": str(probe_status),
        "records_a": len(recs_a),
        "records_b": len(recs_b),
        "effective_start_gen": int(eff_start_gen),
        "first_mismatch": probe_mismatch,
        "key_summary": probe_key_summary,
        "context_window": probe_context,
    }

    # policy_exec_probe compare (policy-side digests: slots/tokens/logits/actions)
    policy_exec_probe = _compare_policy_exec_probe(run_a, run_b, start_gen=eff_start_gen)


    # EVOPY compare
    console_a = _find_console_log(run_a, prefer_contains=label_a)
    console_b = _find_console_log(run_b, prefer_contains=label_b)
    evopy = _try_evopy_compare(console_a, console_b)

    # policy_slots diff
    ps_a = run_a / "debug" / "policy_slots.log"
    ps_b = run_b / "debug" / "policy_slots.log"
    ps_diff = None
    if ps_a.exists() and ps_b.exists():
        ps_diff = _first_line_diff(ps_a, ps_b)

    policy_slots = {
        "path_a": str(ps_a) if ps_a.exists() else None,
        "path_b": str(ps_b) if ps_b.exists() else None,
        "first_diff": ps_diff,
    }

    # Add a small line-context window around the first diff to avoid manual log scrolling.
    ps_ln = None
    if isinstance(ps_diff, dict):
        ps_ln = ps_diff.get("line_number")
    if isinstance(ps_ln, int) and ps_ln >= 0:
        policy_slots["context_radius"] = 6
        policy_slots["context_a"] = _read_text_context(ps_a, ps_ln, radius=policy_slots["context_radius"])
        policy_slots["context_b"] = _read_text_context(ps_b, ps_ln, radius=policy_slots["context_radius"])

    # run_config diff
    rca = run_a / "run_config.csv"
    rcb = run_b / "run_config.csv"
    run_config = {"path_a": str(rca) if rca.exists() else None, "path_b": str(rcb) if rcb.exists() else None}
    if rca.exists() and rcb.exists():
        cfg_a = _read_run_config(rca)
        cfg_b = _read_run_config(rcb)
        run_config["diff"] = _diff_run_config(cfg_a, cfg_b)
        # Pull a small set of resume-related keys (most common human question).
        resume_keys = sorted([k for k in set(cfg_a.keys()) | set(cfg_b.keys()) if "resume" in k.lower()])
        run_config["resume_keys"] = {
            "a": {k: _short(cfg_a.get(k)) for k in resume_keys[:30]},
            "b": {k: _short(cfg_b.get(k)) for k in resume_keys[:30]},
        }
    else:
        run_config["diff"] = {"diff_count": None, "sample": []}

    # checkpoint bundle inventories
    bundles = {"a": _bundle_inventory(run_a), "b": _bundle_inventory(run_b)}


    # ---- schema compatibility / stable keys ----
    # Older notebooks expected flat keys like `first_determinism_probe_mismatch`.
    # Keep aliases so investigators don't waste time on KeyError churn.
    if isinstance(policy_exec_probe, dict):
        policy_exec_probe.setdefault("first_mismatch", None)

    compat_first_det = None
    try:
        if isinstance(probe_block, dict):
            compat_first_det = probe_block.get("first_mismatch")
    except Exception:
        compat_first_det = None

    compat_first_policy = None
    try:
        if isinstance(policy_exec_probe, dict):
            compat_first_policy = policy_exec_probe.get("first_mismatch")
    except Exception:
        compat_first_policy = None
    return {
        "schema_version": "phase3_forensic_report_v3",
        "generated_at": _utc_iso_now(),
        "runs": {
            "a": {"label": label_a, "dir": str(run_a)},
            "b": {"label": label_b, "dir": str(run_b)},
        },
        "fingerprints": fingerprints,
        "csv_divergence": csv_div,
        "determinism_probe": probe_block,
        "first_determinism_probe_mismatch": compat_first_det,
        "first_policy_exec_probe_mismatch": compat_first_policy,
        "policy_exec_probe": policy_exec_probe,
        "evopy_compare": evopy,
        "policy_slots": policy_slots,
        "run_config": run_config,
        "checkpoint_bundles_v2": bundles,
    }


def _write_report_files(out_dir: Path, payload: Dict[str, Any]) -> Tuple[Path, Path]:
    """Write JSON and Markdown report files."""
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "_FORENSIC_REPORT.json"
    md_path = out_dir / "_FORENSIC_REPORT.md"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")

    # Markdown (human snack-sized)
    a = payload.get("runs", {}).get("a", {})
    b = payload.get("runs", {}).get("b", {})

    fp = payload.get("fingerprints", {})
    div = payload.get("csv_divergence", {})
    probe = payload.get("determinism_probe", {})
    policy_exec = payload.get("policy_exec_probe", {})
    aux = payload.get("aux_artifacts", {})
    evopy = payload.get("evopy_compare", {})
    policy_slots = payload.get("policy_slots", {})
    run_cfg = payload.get("run_config", {})
    bundles = payload.get("checkpoint_bundles_v2", {})

    lines: List[str] = []
    lines.append("# Phase 3 Forensic Report")
    lines.append("")
    lines.append(f"Generated: {payload.get('generated_at')}")
    lines.append("")
    lines.append("## Runs")
    lines.append(f"- A ({a.get('label')}): {a.get('dir')}")
    lines.append(f"- B ({b.get('label')}): {b.get('dir')}")
    if isinstance(aux, dict) and aux.get("dir"):
        lines.append(f"- Aux artifacts: {aux.get('dir')}")
    lines.append("")
    lines.append("## Determinism fingerprint")
    lines.append(f"- byte_equal: {fp.get('byte_equal')}")
    lines.append(f"- canonical_equal: {fp.get('canonical_equal')}")
    lines.append(f"- A canonical_sha256: {fp.get('a', {}).get('canonical_sha256')}")
    lines.append(f"- B canonical_sha256: {fp.get('b', {}).get('canonical_sha256')}")
    lines.append("")
    lines.append("## First divergence (CSV overlap)")
    lines.append(f"- metrics.csv: {div.get('metrics.csv', {})}")
    lines.append(f"- behaviors.csv: {div.get('behaviors.csv', {})}")
    lines.append("")
    lines.append("## First determinism_probe mismatch")
    lines.append(f"- ignored_probe_hash_key_suffixes: {_IGNORE_PROBE_HASH_KEYS_SUFFIXES}")
    lines.append(f"- effective_start_gen: {probe.get('effective_start_gen')}")
    ks0 = probe.get("key_summary", {})
    if isinstance(ks0, dict) and ks0.get("ignored_total_keys_with_mismatch"):
        lines.append(
            f"- ignored_key_mismatches: {ks0.get('ignored_total_keys_with_mismatch')} (non-actionable; suffixes={_IGNORE_PROBE_HASH_KEYS_SUFFIXES})"
        )
    fm = probe.get("first_mismatch")
    if fm:
        lines.append(f"- mismatch: gen={fm.get('generation_index')} tag={fm.get('tag')} occurrence={fm.get('occurrence')} index_in_a={fm.get('index_in_a')}")
        diffs = fm.get("diffs", {}) if isinstance(fm.get("diffs"), dict) else {}
        keys = list(diffs.keys())[:10]
        lines.append(f"- diff_keys(first10): {keys}")
        # show first few diffs with values
        for k in keys[:5]:
            d = diffs.get(k, {})
            lines.append(f"  - {k}: A={d.get('a')} B={d.get('b')}")
    else:
        lines.append("- mismatch: None (after ignoring non-actionable keys, or probes missing)")
    lines.append("")
    lines.append("## Probe context window (around first mismatch)")
    cw = probe.get("context_window")
    if isinstance(cw, dict) and isinstance(cw.get("rows"), list):
        lines.append(f"- mismatch_index_in_a: {cw.get('mismatch_index_in_a')}")
        ra = cw.get("range", {})
        lines.append(f"- window_range: {ra}")
        resync = cw.get("rng_manager_snapshot_resync_at")
        if resync:
            lines.append(f"- rng_manager_snapshot_resync_at: {resync}")
        # show a compact per-row view
        for row in cw.get("rows", [])[:20]:
            if not isinstance(row, dict):
                continue
            tag = row.get("tag")
            diffk = row.get("diff_keys")
            idx = row.get("index_in_a")
            lines.append(f"  - idx={idx} {tag}: diff_keys={diffk}")
    else:
        lines.append("- (not available)")
    lines.append("")
    lines.append("## Probe per-key mismatch summary (top)")
    ks = probe.get("key_summary", {})
    if isinstance(ks, dict):
        lines.append(f"- total_keys_with_mismatch: {ks.get('total_keys_with_mismatch')}")
        lines.append(f"- ignored_keys_with_mismatch: {ks.get('ignored_total_keys_with_mismatch')}")
        lines.append(f"- missing_records_in_b: {ks.get('missing_records_in_b')}")
        for row in (ks.get("keys") or [])[:12]:
            if not isinstance(row, dict):
                continue
            k = row.get("key")
            cnt = row.get("mismatch_count")
            fm2 = row.get("first_mismatch")
            rs = row.get("resync_at")
            lines.append(f"  - {k}: mismatches={cnt} first={fm2} resync={rs}")
        ignored_rows = (ks.get("ignored_keys") or [])[:12]
        if ignored_rows:
            lines.append("  - ignored (non-actionable):")
            for row in ignored_rows:
                if not isinstance(row, dict):
                    continue
                k = row.get("key")
                cnt = row.get("mismatch_count")
                fm2 = row.get("first_mismatch")
                rs = row.get("resync_at")
                lines.append(f"    - {k}: mismatches={cnt} first={fm2} resync={rs}")
    else:
        lines.append("- (not available)")
    lines.append("")
    lines.append("## policy_exec_probe first mismatch")
    if isinstance(policy_exec, dict):
        lines.append(f"- status: {policy_exec.get('status')}")
        lines.append(f"- effective_start_gen: {policy_exec.get('effective_start_gen')}")
        ds = policy_exec.get("digest_summary", {}) if isinstance(policy_exec.get("digest_summary"), dict) else {}
        if ds:
            lines.append(f"- digests_present: {len(ds.get('digests_present') or [])}")
            lines.append(f"- has_actions_out_digest: {ds.get('has_actions_out_digest')}")
            lines.append(f"- has_logits_digest: {ds.get('has_logits_digest')}")
            lines.append(f"- has_slots_digests: {ds.get('has_slots_digests')}")
            topd = ds.get("top_mismatching_digests") or []
            if isinstance(topd, list) and topd:
                lines.append(f"- top_mismatching_digests(first8): {topd[:8]}")
        fm_pe = policy_exec.get("first_mismatch")
        if isinstance(fm_pe, dict):
            lines.append(
                f"- mismatch: gen={fm_pe.get('gen')} tick={fm_pe.get('tick')} stage={fm_pe.get('stage')} occ={fm_pe.get('occurrence')} index_in_a={fm_pe.get('index_a')}"
            )
            dk = fm_pe.get("diff_digest_keys") or []
            if isinstance(dk, list):
                lines.append(f"- diff_digests: {dk[:12]}")
            # Highlight action/logit mismatches if present
            for k in ["actions_out", "logits", "torch_cpu_rng", "torch_cuda_rng"]:
                if isinstance(fm_pe.get("digests_a"), dict) and k in fm_pe["digests_a"]:
                    lines.append(f"  - {k}: A={fm_pe['digests_a'].get(k)} B={fm_pe.get('digests_b', {}).get(k)}")
        else:
            lines.append("- mismatch: None (or probe files missing)")
    else:
        lines.append("- (not available)")
    lines.append("")
    lines.append("## policy_slots.log first diff")
    fd = policy_slots.get("first_diff") if isinstance(policy_slots, dict) else None
    if isinstance(fd, dict):
        if fd.get("equal_after_alignment"):
            lines.append("- (no diff found after suffix alignment)")
            if isinstance(fd.get("alignment"), dict):
                lines.append(f"- alignment: {fd.get('alignment')}")
        else:
            lines.append(f"- line_number: {fd.get('line_number')}")
            lines.append(f"- A: {fd.get('a')}")
            lines.append(f"- B: {fd.get('b')}")
            # Show a tiny excerpt if we captured context.
            ca = policy_slots.get("context_a") if isinstance(policy_slots, dict) else None
            cb = policy_slots.get("context_b") if isinstance(policy_slots, dict) else None
            if isinstance(ca, dict) and isinstance(cb, dict):
                lines.append("- policy_slots context (excerpt):")
                ln0 = fd.get("line_number")
                try:
                    ln0 = int(ln0) if ln0 is not None else None
                except Exception:
                    ln0 = None
                for row in (ca.get("lines") or []):
                    if not isinstance(row, dict):
                        continue
                    ln = row.get("line_number")
                    if ln0 is None or (isinstance(ln, int) and abs(int(ln) - ln0) <= 1):
                        lines.append(f"  - A[{row.get('line_number')}]: {row.get('text')}")
                for row in (cb.get("lines") or []):
                    if not isinstance(row, dict):
                        continue
                    ln = row.get("line_number")
                    if ln0 is None or (isinstance(ln, int) and abs(int(ln) - ln0) <= 1):
                        lines.append(f"  - B[{row.get('line_number')}]: {row.get('text')}")
    else:
        lines.append("- (no diff found or files missing)")
    lines.append("")
    lines.append("## run_config.csv diff (sample)")
    if isinstance(run_cfg, dict) and isinstance(run_cfg.get("diff"), dict):
        lines.append(f"- diff_count: {run_cfg['diff'].get('diff_count')}")
        for d in (run_cfg["diff"].get("sample") or [])[:12]:
            if isinstance(d, dict):
                lines.append(f"  - {d.get('key')}: A={d.get('a')} B={d.get('b')}")
    else:
        lines.append("- (not available)")
    lines.append("")
    lines.append("## checkpoint_bundles_v2 inventory")
    if isinstance(bundles, dict):
        lines.append(f"- A: {bundles.get('a')}")
        lines.append(f"- B: {bundles.get('b')}")
    else:
        lines.append("- (not available)")
    lines.append("")
    lines.append("## EVOPY trace compare")
    if isinstance(evopy, dict):
        lines.append(f"- exit_code: {evopy.get('exit_code')}")
        lines.append(f"- message: {evopy.get('message')}")
    else:
        lines.append("- (not available)")
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json_path, md_path


def _write_jsonl_slice_by_filtered_index(
    src_path: Path,
    dst_path: Path,
    gen_key: str,
    start_gen: int,
    lo: int,
    hi: int,
) -> bool:
    """Write a JSONL slice selected by *filtered* index.

    determinism_probe.jsonl and policy_exec_probe.jsonl are often large enough that investigators
    do not want to scroll them manually. This helper writes a tiny, self-contained slice that
    corresponds to the context window shown in the forensic report.

    Indexing semantics:
    - We stream the source JSONL.
    - We only count records whose generation index (gen_key) is >= start_gen.
    - 'lo'/'hi' are inclusive indices in that filtered stream.

    Returns True if a slice was written, False otherwise.
    """
    if not src_path.exists():
        return False
    if lo < 0 or hi < 0 or hi < lo:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_any = False
    idx = -1
    try:
        with src_path.open("r", encoding="utf-8", errors="replace") as f_in, dst_path.open(
            "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                rec = json.loads(line)
                gen = int(rec.get(gen_key, -1))
                if gen < int(start_gen):
                    continue
                idx += 1
                if idx < lo:
                    continue
                if idx > hi:
                    break
                f_out.write(json.dumps(rec, sort_keys=True) + "\n")
                wrote_any = True
    except Exception:
        return False
    return wrote_any


def _write_aux_artifacts(out_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Write small, investigator-friendly slices and context files.

    Outputs under:
      <out_dir>/_FORENSIC_ARTIFACTS/

    This is deliberately redundant with the JSON report. The difference is ergonomics:
    these files can be opened directly (even inside the run ZIP) without needing to load
    the full raw probes.

    Returns a small index describing what was written.
    """
    aux_dir = out_dir / "_FORENSIC_ARTIFACTS"
    aux_dir.mkdir(parents=True, exist_ok=True)

    index: Dict[str, Any] = {"dir": str(aux_dir), "written": []}

    # README
    readme = [
        "# _FORENSIC_ARTIFACTS",
        "",
        "This directory contains **small, pre-sliced** artifacts derived from the full run logs.",
        "They exist to reduce repetitive manual work when triaging determinism/resume issues.",
        "",
        "Files you may care about:",
        "- determinism_probe__A__context.jsonl / determinism_probe__B__context.jsonl",
        "  - the determinism_probe context window shown in the forensic report, as raw JSONL records.",
        "- policy_exec_probe__A__context.jsonl / policy_exec_probe__B__context.jsonl",
        "  - policy-side digest records around the first policy_exec mismatch (slots/tokens/logits/actions).",
        "- policy_slots__first_diff_context.txt",
        "  - a small excerpt around the first differing policy_slots.log line.",
        "",
        "If a file is missing, it usually means the underlying probe/log file was not present in the run.",
        "",
    ]
    (aux_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    index["written"].append("README.md")

    # determinism_probe slices
    det = payload.get("determinism_probe", {}) or {}
    if det.get("status") == "ok":
        ctx = det.get("context_window") or {}
        rng = (ctx.get("range") or {}) if isinstance(ctx, dict) else {}
        lo_a = rng.get("lo")
        hi_a = rng.get("hi")
        eff_gen = int(det.get("effective_start_gen", payload.get("start_gen", 0) or 0))
        if isinstance(lo_a, int) and isinstance(hi_a, int):
            pa = Path(det.get("path_a", ""))
            pb = Path(det.get("path_b", ""))
            wrote_a = _write_jsonl_slice_by_filtered_index(
                pa,
                aux_dir / "determinism_probe__A__context.jsonl",
                gen_key="generation_index",
                start_gen=eff_gen,
                lo=int(lo_a),
                hi=int(hi_a),
            )
            if wrote_a:
                index["written"].append("determinism_probe__A__context.jsonl")

            # For B, prefer the min/max index_b visible in the context window rows, if available.
            rows = ctx.get("rows") if isinstance(ctx, dict) else None
            idx_bs: List[int] = []
            if isinstance(rows, list):
                for r in rows:
                    ib = r.get("index_b") if isinstance(r, dict) else None
                    if isinstance(ib, int):
                        idx_bs.append(int(ib))
            lo_b, hi_b = (min(idx_bs), max(idx_bs)) if idx_bs else (lo_a, hi_a)

            wrote_b = _write_jsonl_slice_by_filtered_index(
                pb,
                aux_dir / "determinism_probe__B__context.jsonl",
                gen_key="generation_index",
                start_gen=eff_gen,
                lo=int(lo_b),
                hi=int(hi_b),
            )
            if wrote_b:
                index["written"].append("determinism_probe__B__context.jsonl")

    # policy_exec_probe slices
    pep = payload.get("policy_exec_probe", {}) or {}
    if pep.get("status") == "ok":
        ctx = pep.get("context_window") or []
        eff_gen = int(pep.get("effective_start_gen", payload.get("start_gen", 0) or 0))
        idx_as: List[int] = []
        idx_bs: List[int] = []
        if isinstance(ctx, list):
            for r in ctx:
                if not isinstance(r, dict):
                    continue
                ia = r.get("index_a")
                ib = r.get("index_b")
                if isinstance(ia, int):
                    idx_as.append(int(ia))
                if isinstance(ib, int):
                    idx_bs.append(int(ib))
        if idx_as:
            lo_a, hi_a = min(idx_as), max(idx_as)
            lo_b, hi_b = (min(idx_bs), max(idx_bs)) if idx_bs else (lo_a, hi_a)
            pa = Path(pep.get("path_a", ""))
            pb = Path(pep.get("path_b", ""))

            wrote_a = _write_jsonl_slice_by_filtered_index(
                pa,
                aux_dir / "policy_exec_probe__A__context.jsonl",
                gen_key="gen",
                start_gen=eff_gen,
                lo=int(lo_a),
                hi=int(hi_a),
            )
            if wrote_a:
                index["written"].append("policy_exec_probe__A__context.jsonl")

            wrote_b = _write_jsonl_slice_by_filtered_index(
                pb,
                aux_dir / "policy_exec_probe__B__context.jsonl",
                gen_key="gen",
                start_gen=eff_gen,
                lo=int(lo_b),
                hi=int(hi_b),
            )
            if wrote_b:
                index["written"].append("policy_exec_probe__B__context.jsonl")

    # policy_slots excerpt
    ps = payload.get("policy_slots", {}) or {}
    ps_diff = ps.get("first_diff")
    if isinstance(ps_diff, dict) and isinstance(ps_diff.get("line_number"), int):
        ln = int(ps_diff["line_number"])
        ctx_a = ps.get("context_a")
        ctx_b = ps.get("context_b")
        parts: List[str] = []
        parts.append("# policy_slots.log first-diff context")
        parts.append("")
        parts.append(f"First diff at line_number={ln} (0-based).")
        parts.append("")
        parts.append("## A")
        if isinstance(ctx_a, dict) and isinstance(ctx_a.get("lines"), list):
            for row in ctx_a["lines"]:
                parts.append(f"{row.get('line_number')}: {row.get('text')}")
        else:
            parts.append("(no context)")
        parts.append("")
        parts.append("## B")
        if isinstance(ctx_b, dict) and isinstance(ctx_b.get("lines"), list):
            for row in ctx_b["lines"]:
                parts.append(f"{row.get('line_number')}: {row.get('text')}")
        else:
            parts.append("(no context)")
        (aux_dir / "policy_slots__first_diff_context.txt").write_text("\n".join(parts) + "\n", encoding="utf-8")
        index["written"].append("policy_slots__first_diff_context.txt")

    # index
    (aux_dir / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    index["written"].append("index.json")
    return index



def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--label-a", default="fresh")
    ap.add_argument("--label-b", default="resume")
    ap.add_argument("--start-gen", type=int, default=None)
    args = ap.parse_args(argv)

    run_a = Path(args.run_a)
    run_b = Path(args.run_b)

    if not run_a.exists() or not run_b.exists():
        print("[FORENSIC_REPORT] Missing run dir(s).")
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else run_a.parent

    payload = build_report(
        run_a,
        run_b,
        label_a=str(args.label_a),
        label_b=str(args.label_b),
        start_gen=args.start_gen,
    )

    # Write auxiliary, investigator-friendly slices (best effort).
    try:
        payload["aux_artifacts"] = _write_aux_artifacts(out_dir, payload)
    except Exception as e:
        print(f"[FORENSIC_REPORT] Aux artifacts failed: {type(e).__name__}: {e}")

    try:
        _write_report_files(out_dir, payload)
    except Exception as e:
        print(f"[FORENSIC_REPORT] Failed to write report: {type(e).__name__}: {e}")
        return 2

    print(f"[FORENSIC_REPORT] Wrote: {out_dir / '_FORENSIC_REPORT.md'}")
    print(f"[FORENSIC_REPORT] Wrote: {out_dir / '_FORENSIC_REPORT.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
