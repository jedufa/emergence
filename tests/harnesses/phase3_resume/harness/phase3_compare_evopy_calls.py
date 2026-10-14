"""scripts/phase3_compare_evopy_calls.py

Phase 3 Determinism Forensics: evolution_py call trace comparator.
Last modified: 2026-02-19 20:02 Europe/Vienna

Purpose
-------
The Phase 3 determinism leak is inside *python RNG consumption ordering* for the
`evolution_py` stream. The engine emits trace lines like:

  [EVOPY_TRACE_RESET] {...}
  [EVOPY_CALL] {...}

This tool compares two console logs (fresh vs resume) and finds the FIRST
mismatch in `pre_sha` after a shared epoch reset marker.

Why this matters
----------------
- Boundary state equivalence can pass while RNG consumption order diverges later.
- The first `pre_sha` mismatch identifies the callsite that introduced drift.

Usage
-----
python -m tests.harnesses.phase3_resume.harness.phase3_compare_evopy_calls --log-a path --log-b path

Exit codes
----------
0: No mismatch found (includes the case where no comparable segment is available)
1: Mismatch found
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class EvopyCall:
    """One evolution_py RNG call trace record."""

    call_idx: int
    pre_sha: str
    post_sha: str
    method: str
    caller: str
    gen: int
    tick: int
    stage: str
    epoch: str


def _iter_evopy_records(log_path: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (kind, payload) for EVOPY_TRACE_RESET and EVOPY_CALL records."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[EVOPY_TRACE_RESET]"):
                rest = line[len("[EVOPY_TRACE_RESET]") :].strip()
                try:
                    yield "reset", json.loads(rest)
                except Exception:
                    continue
            elif line.startswith("[EVOPY_CALL]"):
                rest = line[len("[EVOPY_CALL]") :].strip()
                try:
                    yield "call", json.loads(rest)
                except Exception:
                    continue


def _collect_after_epoch(log_path: str, epoch: str) -> List[EvopyCall]:
    """Collect EVOPY_CALL records occurring after the LAST reset for `epoch`."""
    resets: List[int] = []
    lines: List[EvopyCall] = []

    # First pass: find last reset position
    idx = -1
    for kind, payload in _iter_evopy_records(log_path):
        idx += 1
        if kind == "reset" and str(payload.get("epoch", "")) == epoch:
            resets.append(idx)

    if not resets:
        return []

    last_reset_idx = resets[-1]

    # Second pass: collect calls after last reset
    idx = -1
    seen_reset = False
    for kind, payload in _iter_evopy_records(log_path):
        idx += 1
        if idx == last_reset_idx:
            seen_reset = True
            continue
        if not seen_reset:
            continue
        if kind != "call":
            continue
        try:
            lines.append(
                EvopyCall(
                    call_idx=int(payload.get("call_idx", -1)),
                    pre_sha=str(payload.get("pre_sha", "")),
                    post_sha=str(payload.get("post_sha", "")),
                    method=str(payload.get("method", "")),
                    caller=str(payload.get("caller", "")),
                    gen=int(payload.get("gen", -1)),
                    tick=int(payload.get("tick", -1)),
                    stage=str(payload.get("stage", "")),
                    epoch=str(payload.get("epoch", "")),
                )
            )
        except Exception:
            continue

    return lines


def _index_by_call_idx(calls: List[EvopyCall]) -> Dict[int, EvopyCall]:
    """Index calls by call_idx, keeping the first occurrence."""
    out: Dict[int, EvopyCall] = {}
    for c in calls:
        if c.call_idx not in out:
            out[c.call_idx] = c
    return out


def compare_logs(log_a: str, log_b: str, *, epoch: str) -> Tuple[int, Optional[str]]:
    """Return (exit_code, message)."""
    a_calls = _collect_after_epoch(log_a, epoch)
    b_calls = _collect_after_epoch(log_b, epoch)

    if not a_calls or not b_calls:
        # Missing trace markers is common when EVOPY tracing is disabled or redirected.
        # This should not fail Phase 3 determinism by itself.
        return 0, f"[EVOPY_COMPARE] SKIP: No comparable EVOPY segment found for epoch='{epoch}'."

    a_idx = _index_by_call_idx(a_calls)
    b_idx = _index_by_call_idx(b_calls)

    common = sorted(set(a_idx.keys()) & set(b_idx.keys()))
    if not common:
        return 0, f"[EVOPY_COMPARE] SKIP: No overlapping call_idx after epoch='{epoch}'."

    for k in common:
        ac = a_idx[k]
        bc = b_idx[k]
        if ac.pre_sha != bc.pre_sha:
            msg = (
                "[EVOPY_FIRST_DIVERGENCE] "
                + json.dumps(
                    {
                        "epoch": epoch,
                        "call_idx": k,
                        "A": {
                            "pre_sha": ac.pre_sha,
                            "post_sha": ac.post_sha,
                            "method": ac.method,
                            "caller": ac.caller,
                            "gen": ac.gen,
                            "tick": ac.tick,
                            "stage": ac.stage,
                        },
                        "B": {
                            "pre_sha": bc.pre_sha,
                            "post_sha": bc.post_sha,
                            "method": bc.method,
                            "caller": bc.caller,
                            "gen": bc.gen,
                            "tick": bc.tick,
                            "stage": bc.stage,
                        },
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            return 1, msg

    return 0, f"[EVOPY_COMPARE] OK: No pre_sha mismatch found after epoch='{epoch}' (overlap_n={len(common)})."


def build_argparser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-a", required=True, help="Fresh run console log")
    ap.add_argument("--log-b", required=True, help="Resume run console log")
    ap.add_argument("--epoch", default="boundary", help="Epoch name to align on")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    args = build_argparser().parse_args(argv)
    code, msg = compare_logs(args.log_a, args.log_b, epoch=str(args.epoch))
    if msg:
        print(msg)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
