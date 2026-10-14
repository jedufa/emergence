#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


DEFAULT_FIELDS = [
    "generation",
    "alive",
    "deaths",
    "successful_births",
    "failed_births",
    "birth_fail_no_space",
    "birth_fail_no_mate",
    "birth_fail_injury_gate",
    "lineage_active_count",
    "lineage_top1_alive_share",
    "lineage_effective_alive_lineages",
    "final_effective_alive_lineages",
    "final_top1_alive_share",
    "wdl_delayed_hazard_agent_cue_exposure_count",
    "wdl_delayed_hazard_agent_active_exposure_count",
    "wdl_delayed_hazard_damage_total",
    "wdl_cue_seen_agents",
    "wdl_hazard_enter_count",
    "wdl_hazard_avoidance_proxy",
]


def _metrics_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "metrics.csv"
    if candidate.is_file():
        return candidate
    matches = sorted(path.rglob("metrics.csv"))
    if not matches:
        raise FileNotFoundError(f"metrics.csv not found under {path}")
    if len(matches) > 1:
        raise ValueError(f"multiple metrics.csv files under {path}; pass the run directory or file directly")
    return matches[0]


def _read_final(path: Path) -> tuple[int, dict[str, str]]:
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return len(rows), rows[-1]


def _checkpoint_presence(path: Path) -> dict[str, Any]:
    root = path if path.is_dir() else path.parent
    bundle_dir = root / "checkpoint_bundles_v2"
    bundle_files = sorted(bundle_dir.rglob("*")) if bundle_dir.exists() else []
    bundle_files = [p for p in bundle_files if p.is_file()]
    index_path = root / "viz" / "checkpoints_index.json"
    return {
        "checkpoint_bundles_v2_exists": bundle_dir.exists(),
        "checkpoint_bundle_file_count": len(bundle_files),
        "checkpoints_index_exists": index_path.exists(),
    }


def _coerce(value: str) -> float | str:
    try:
        if value is None:
            return ""
        text = str(value)
        if text == "":
            return ""
        return float(text)
    except Exception:
        return str(value)


def _same(a: float | str, b: float | str, *, abs_tol: float, rel_tol: float) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return a == b


def compare(before: Path, after: Path, *, fields: list[str], abs_tol: float, rel_tol: float) -> dict[str, Any]:
    before_metrics = _metrics_path(before)
    after_metrics = _metrics_path(after)
    before_rows, before_final = _read_final(before_metrics)
    after_rows, after_final = _read_final(after_metrics)
    rows: list[dict[str, Any]] = []
    mismatches = 0
    for field in fields:
        b_raw = before_final.get(field, "")
        a_raw = after_final.get(field, "")
        b_val = _coerce(b_raw)
        a_val = _coerce(a_raw)
        ok = _same(b_val, a_val, abs_tol=abs_tol, rel_tol=rel_tol)
        if not ok:
            mismatches += 1
        rows.append(
            {
                "field": field,
                "before": b_raw,
                "after": a_raw,
                "match": ok,
            }
        )
    row_count_match = before_rows == after_rows
    if not row_count_match:
        mismatches += 1
    before_checkpoint = _checkpoint_presence(before_metrics)
    after_checkpoint = _checkpoint_presence(after_metrics)
    checkpoint_presence_match = (
        before_checkpoint["checkpoint_bundles_v2_exists"] == after_checkpoint["checkpoint_bundles_v2_exists"]
        and before_checkpoint["checkpoints_index_exists"] == after_checkpoint["checkpoints_index_exists"]
    )
    if not checkpoint_presence_match:
        mismatches += 1
    return {
        "schema_version": "eiel_perf_parity_compare_v1",
        "before_metrics": str(before_metrics),
        "after_metrics": str(after_metrics),
        "before_row_count": before_rows,
        "after_row_count": after_rows,
        "row_count_match": row_count_match,
        "before_checkpoint": before_checkpoint,
        "after_checkpoint": after_checkpoint,
        "checkpoint_presence_match": checkpoint_presence_match,
        "field_results": rows,
        "mismatches": mismatches,
        "passed": mismatches == 0,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# EIEL Perf Parity Compare",
        "",
        f"- before: `{payload['before_metrics']}`",
        f"- after: `{payload['after_metrics']}`",
        f"- row counts: {payload['before_row_count']} vs {payload['after_row_count']} ({'match' if payload['row_count_match'] else 'mismatch'})",
        f"- checkpoint presence: {'match' if payload['checkpoint_presence_match'] else 'mismatch'}",
        f"- result: {'PASS' if payload['passed'] else 'FAIL'}",
        "",
        "| field | before | after | match |",
        "|---|---:|---:|---|",
    ]
    for row in payload["field_results"]:
        lines.append(f"| {row['field']} | {row['before']} | {row['after']} | {row['match']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare final metrics parity for two same-seed EIEL runs")
    parser.add_argument("--before", required=True, help="Before run dir or metrics.csv")
    parser.add_argument("--after", required=True, help="After run dir or metrics.csv")
    parser.add_argument("--field", action="append", default=[], help="Metric field to compare; defaults cover acceptance fields")
    parser.add_argument("--abs-tol", type=float, default=1e-9)
    parser.add_argument("--rel-tol", type=float, default=1e-9)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown", default="")
    args = parser.parse_args()
    payload = compare(
        Path(args.before),
        Path(args.after),
        fields=[str(x) for x in args.field] or DEFAULT_FIELDS,
        abs_tol=float(args.abs_tol),
        rel_tol=float(args.rel_tol),
    )
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "mismatches": payload["mismatches"]}, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
