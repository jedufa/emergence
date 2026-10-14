#!/usr/bin/env python3
"""
scripts/phase3_find_first_divergence.py

Phase 3 helper: find the *first* generation where two runs diverge, while
comparing only the **overlapping generation range**.

Optional:
- --start-gen: only compare generations >= start-gen (after overlap filtering)

Strictness:
- Values are compared as strings exactly as written in the CSV.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass(frozen=True)
class DivergenceResult:
    """Represents the first divergence found in a CSV artifact."""
    artifact: str
    generation: Optional[int]
    reason: str


def _read_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read a CSV file and return (fieldnames, rows). Values remain raw strings."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = [row for row in reader]
        return list(reader.fieldnames), rows


def _extract_generation_column(fieldnames: List[str]) -> str:
    """Return the column name used for generation."""
    for cand in ("generation", "gen"):
        if cand in fieldnames:
            return cand
    raise ValueError(f"No generation column found. Expected one of: generation, gen. Got: {fieldnames}")


def _group_by_gen(gen_col: str, rows: List[Dict[str, str]]) -> Dict[int, List[Dict[str, str]]]:
    """Group rows by integer generation."""
    out: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        if gen_col not in r:
            raise ValueError(f"Row missing generation column '{gen_col}'")
        g = int(r[gen_col])
        out.setdefault(g, []).append(r)
    return out


def _first_metrics_divergence(path_a: str, path_b: str, *, start_gen: Optional[int] = None) -> DivergenceResult:
    """Find first divergence in metrics.csv comparing only overlapping gens."""
    fna, rows_a = _read_csv_rows(path_a)
    fnb, rows_b = _read_csv_rows(path_b)

    gen_col_a = _extract_generation_column(fna)
    gen_col_b = _extract_generation_column(fnb)

    by_gen_a = {int(r[gen_col_a]): r for r in rows_a}
    by_gen_b = {int(r[gen_col_b]): r for r in rows_b}

    common_gens = sorted(set(by_gen_a.keys()) & set(by_gen_b.keys()))
    if start_gen is not None:
        common_gens = [g for g in common_gens if g >= start_gen]

    if not common_gens:
        return DivergenceResult("metrics.csv", None, "no overlapping generations")

    cols_a = [c for c in fna if c != gen_col_a]
    cols_b = [c for c in fnb if c != gen_col_b]
    common_cols = [c for c in cols_a if c in set(cols_b)]

    for g in common_gens:
        ra = by_gen_a[g]
        rb = by_gen_b[g]
        diffs = [c for c in common_cols if ra.get(c, "") != rb.get(c, "")]
        if diffs:
            shown = diffs[:12]
            more = len(diffs) - len(shown)
            suffix = f" (+{more} more)" if more > 0 else ""
            return DivergenceResult("metrics.csv", g, f"differing columns: {', '.join(shown)}{suffix}")

    return DivergenceResult("metrics.csv", None, "no divergence in overlapping generations")


def _sort_behavior_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return a stable ordering for behavior rows."""
    if not rows:
        return rows
    if "agent_id" in rows[0]:
        return sorted(rows, key=lambda r: int(r.get("agent_id", "-1")))
    if "slot" in rows[0]:
        return sorted(rows, key=lambda r: int(r.get("slot", "-1")))
    return sorted(rows, key=lambda r: tuple(sorted(r.items())))


def _first_behaviors_divergence(path_a: str, path_b: str, *, start_gen: Optional[int] = None) -> DivergenceResult:
    """Find first divergence in behaviors.csv comparing only overlapping gens."""
    fna, rows_a = _read_csv_rows(path_a)
    fnb, rows_b = _read_csv_rows(path_b)

    gen_col_a = _extract_generation_column(fna)
    gen_col_b = _extract_generation_column(fnb)

    by_gen_a = _group_by_gen(gen_col_a, rows_a)
    by_gen_b = _group_by_gen(gen_col_b, rows_b)

    common_gens = sorted(set(by_gen_a.keys()) & set(by_gen_b.keys()))
    if start_gen is not None:
        common_gens = [g for g in common_gens if g >= start_gen]

    if not common_gens:
        return DivergenceResult("behaviors.csv", None, "no overlapping generations")

    cols_a = [c for c in fna if c != gen_col_a]
    cols_b = [c for c in fnb if c != gen_col_b]
    common_cols = [c for c in cols_a if c in set(cols_b)]

    for g in common_gens:
        ga = _sort_behavior_rows(by_gen_a[g])
        gb = _sort_behavior_rows(by_gen_b[g])

        if len(ga) != len(gb):
            return DivergenceResult(
                "behaviors.csv",
                g,
                f"row-shape differs: A=({len(ga)}, {len(common_cols)+1}) B=({len(gb)}, {len(common_cols)+1})",
            )

        for i, (ra, rb) in enumerate(zip(ga, gb)):
            diffs = [c for c in common_cols if ra.get(c, "") != rb.get(c, "")]
            if diffs:
                shown = diffs[:10]
                more = len(diffs) - len(shown)
                suffix = f" (+{more} more)" if more > 0 else ""
                anchor = f" (agent_id={ra.get('agent_id')})" if "agent_id" in ra else ""
                return DivergenceResult("behaviors.csv", g, f"row {i}{anchor}: differing columns: {', '.join(shown)}{suffix}")

    return DivergenceResult("behaviors.csv", None, "no divergence in overlapping generations")


def _resolve_run_file(run_dir: str, filename: str) -> str:
    """Return absolute path to a file inside a run directory."""
    p = os.path.join(run_dir, filename)
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True)
    ap.add_argument("--run-b", required=True)
    ap.add_argument("--start-gen", type=int, default=None,
                    help="Only compare generations >= this value (after overlap filtering)")
    args = ap.parse_args(argv)

    run_a = args.run_a
    run_b = args.run_b

    print("=== Phase 3 divergence scan (overlap-only, optional start-gen) ===")
    print(f"Run A: {run_a}")
    print(f"Run B: {run_b}")

    try:
        metrics_a = _resolve_run_file(run_a, "metrics.csv")
        metrics_b = _resolve_run_file(run_b, "metrics.csv")
        beh_a = _resolve_run_file(run_a, "behaviors.csv")
        beh_b = _resolve_run_file(run_b, "behaviors.csv")

        mres = _first_metrics_divergence(metrics_a, metrics_b, start_gen=args.start_gen)
        bres = _first_behaviors_divergence(beh_a, beh_b, start_gen=args.start_gen)

        if mres.generation is None:
            print(f"metrics.csv: {mres.reason}")
        else:
            print(f"metrics.csv: first divergence at gen={mres.generation}: {mres.reason}")

        if bres.generation is None:
            print(f"behaviors.csv: {bres.reason}")
        else:
            print(f"behaviors.csv: first divergence at gen={bres.generation}: {bres.reason}")

        diverged = (mres.generation is not None) or (bres.generation is not None)
        return 2 if diverged else 0

    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
