#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "predrisk_fixed1_validation_v1"
csv.field_size_limit(sys.maxsize)
WORLDS = (
    "true_flat",
    "resource_only",
    "predator_only",
    "cover_only",
    "label_only_sham",
    "uncorrelated_risk_resource",
    "budget_matched_uniform",
    "correlated_tradeoff",
)
ARMS = ("risk_exploit", "balanced_generalist", "safe_cover")
ZONES = ("exposed_rich", "safe_covered")
PRIMARY_WORLD = "correlated_tradeoff"
CONTROL_WORLDS = ("true_flat", "label_only_sham", "budget_matched_uniform")
PREDRISK_ENABLED_WORLDS = tuple(world for world in WORLDS if world != "true_flat")
REQUIRED_TELEMETRY_FIELDS = (
    "exposed_rich_available_resource_agent_ticks",
    "exposed_rich_food_gain",
    "exposed_rich_movement_cost_before_move_actions",
    "exposed_rich_effective_predator_damage_cost",
    "safe_covered_eligible_parent_events_after_zone_use",
    "safe_covered_selected_parent_events_after_zone_use",
)
RUN_ID_PREFIXES = {
    "predrisk_fixed1_behavioral_probe_validation",
    "predrisk_fixed1_behavioral_probe_confirmation",
}
SEED_DIRECTION_RULES = {
    96: {"min_adjacent_pass": 3, "max_adjacent_reverse": 1},
    192: {"min_adjacent_pass": 6, "max_adjacent_reverse": 2},
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(statistics.fmean(rows)) if rows else 0.0


def _rate_per_1000(num: float, den: float) -> float | None:
    return 1000.0 * float(num) / float(den) if float(den) > 0.0 else None


def parse_run_id(run_id: str) -> dict[str, Any]:
    parts = str(run_id).split("__")
    if len(parts) != 4 or parts[0] not in RUN_ID_PREFIXES:
        return {"world_family": "", "fixed_probe": "", "seed": -1}
    return {"world_family": parts[1], "fixed_probe": parts[2], "seed": int(parts[3].replace("seed", ""))}


def _campaign_results(root: Path) -> list[dict[str, Any]]:
    summary = _read_json(root / "parallel_campaign_summary.json")
    return list(summary.get("results") or [])


def _sum_rows(rows: list[dict[str, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        for key, value in row.items():
            if key in {"arm", "generation"}:
                continue
            out[key] = out.get(key, 0.0) + _float(value)
    return out


def _net_return(sums: Mapping[str, float], zone: str) -> float:
    return (
        float(sums.get(f"{zone}_food_gain", 0.0))
        - float(sums.get(f"{zone}_movement_cost_before_move_actions", 0.0))
        - float(sums.get(f"{zone}_effective_predator_damage_cost", 0.0))
    )


def _cell_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    run_id = str(result.get("run_id", ""))
    parsed = parse_run_id(run_id)
    run_dir = Path(str(result.get("run_dir", "")))
    telemetry_path = run_dir / "metrics" / "predrisk_fixed1_generation_summary.csv"
    contract_path = run_dir / "metrics" / "predrisk_fixed1_probe_contract.json"
    metrics_path = run_dir / "metrics.csv"
    missing = [str(path) for path in (telemetry_path, contract_path, metrics_path) if not path.exists()]
    if missing:
        return {**parsed, "run_id": run_id, "run_dir": str(run_dir), "status": "MISSING", "missing": ";".join(missing)}
    telemetry = _read_csv(telemetry_path)
    contract = _read_json(contract_path)
    final_metrics = _read_csv(metrics_path)[-1]
    sums = _sum_rows(telemetry)
    missing_fields = [
        field
        for field in REQUIRED_TELEMETRY_FIELDS
        if parsed["world_family"] in PREDRISK_ENABLED_WORLDS and field not in sums
    ]
    out: dict[str, Any] = {
        **parsed,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": str(result.get("status", "")),
        "telemetry_schema_version": contract.get("telemetry_schema_version", ""),
        "alive_final": _float(final_metrics.get("alive")),
        "successful_births_final": _float(final_metrics.get("successful_births")),
        "failed_births_final": _float(final_metrics.get("failed_births")),
        "missing_required_telemetry_fields": ";".join(missing_fields),
    }
    for zone in ZONES:
        available = float(sums.get(f"{zone}_available_resource_agent_ticks", 0.0))
        eligible = float(sums.get(f"{zone}_eligible_parent_events_after_zone_use", 0.0))
        selected = float(sums.get(f"{zone}_selected_parent_events_after_zone_use", 0.0))
        net = _net_return(sums, zone)
        out[f"{zone}_available_resource_agent_ticks"] = available
        out[f"{zone}_food_gain"] = float(sums.get(f"{zone}_food_gain", 0.0))
        out[f"{zone}_movement_cost_before_move_actions"] = float(
            sums.get(f"{zone}_movement_cost_before_move_actions", 0.0)
        )
        out[f"{zone}_effective_predator_damage_cost"] = float(sums.get(f"{zone}_effective_predator_damage_cost", 0.0))
        out[f"{zone}_net_return"] = net
        out[f"{zone}_net_return_per_1000_available_resource_agent_ticks"] = _rate_per_1000(net, available)
        out[f"{zone}_eligible_parent_events_after_zone_use"] = eligible
        out[f"{zone}_selected_parent_events_after_zone_use"] = selected
        out[f"{zone}_selected_parent_events_per_1000_eligible_parent_events_after_zone_use"] = _rate_per_1000(
            selected,
            eligible,
        )
        out[f"{zone}_successful_births_after_zone_use"] = float(
            sums.get(f"{zone}_successful_births_after_zone_use", 0.0)
        )
        out[f"{zone}_predator_events"] = float(sums.get(f"{zone}_predator_events", 0.0))
        out[f"{zone}_predator_damage_prevented"] = float(sums.get(f"{zone}_predator_damage_prevented", 0.0))
    return out


def collect_cells(root: Path) -> list[dict[str, Any]]:
    cells = [_cell_from_result(result) for result in _campaign_results(root)]
    return sorted(cells, key=lambda r: (WORLDS.index(r["world_family"]), ARMS.index(r["fixed_probe"]), int(r["seed"])))


def aggregate_cells(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    numeric_keys = [
        key
        for key, value in cells[0].items()
        if isinstance(value, (int, float)) and key not in {"seed"}
    ] if cells else []
    for world in WORLDS:
        for arm in ARMS:
            subset = [cell for cell in cells if cell["world_family"] == world and cell["fixed_probe"] == arm]
            row: dict[str, Any] = {"world_family": world, "fixed_probe": arm, "n": len(subset)}
            for key in numeric_keys:
                row[key] = _mean(float(cell[key]) for cell in subset if cell.get(key) is not None)
            rows.append(row)
    return rows


def _mean_metric(cells: list[dict[str, Any]], world: str, arm: str, metric: str) -> float | None:
    vals = [cell.get(metric) for cell in cells if cell["world_family"] == world and cell["fixed_probe"] == arm]
    nums = [float(v) for v in vals if v is not None]
    return _mean(nums) if nums else None


def _direction(cells: list[dict[str, Any]], world: str, metric: str, order: tuple[str, str, str]) -> dict[str, Any]:
    vals = {arm: _mean_metric(cells, world, arm, metric) for arm in order}
    valid = all(vals[arm] is not None for arm in order)
    deltas = {
        f"{order[0]}_minus_{order[1]}": (float(vals[order[0]]) - float(vals[order[1]])) if valid else None,
        f"{order[1]}_minus_{order[2]}": (float(vals[order[1]]) - float(vals[order[2]])) if valid else None,
    }
    passed = bool(valid and deltas[f"{order[0]}_minus_{order[1]}"] > 0.0 and deltas[f"{order[1]}_minus_{order[2]}"] > 0.0)
    return {"world": world, "metric": metric, "expected_order": list(order), "means": vals, "deltas": deltas, "pass": passed}


def _seed_metric(cells: list[dict[str, Any]], world: str, seed: int, arm: str, metric: str) -> float | None:
    vals = [
        cell.get(metric)
        for cell in cells
        if cell["world_family"] == world and int(cell["seed"]) == int(seed) and cell["fixed_probe"] == arm
    ]
    nums = [float(v) for v in vals if v is not None]
    return nums[0] if nums else None


def _direction_by_seed(
    cells: list[dict[str, Any]],
    world: str,
    metric: str,
    order: tuple[str, str, str],
    *,
    expected_cells: int | None,
) -> dict[str, Any]:
    seeds = sorted({int(cell["seed"]) for cell in cells if cell["world_family"] == world})
    comparisons = (f"{order[0]}_minus_{order[1]}", f"{order[1]}_minus_{order[2]}")
    seed_rows: list[dict[str, Any]] = []
    counts = {
        name: {"pass": 0, "reverse": 0, "null": 0}
        for name in comparisons
    }
    for seed in seeds:
        vals = {arm: _seed_metric(cells, world, seed, arm, metric) for arm in order}
        row: dict[str, Any] = {"seed": seed, "values": vals}
        for left, right in ((order[0], order[1]), (order[1], order[2])):
            name = f"{left}_minus_{right}"
            if vals[left] is None or vals[right] is None:
                delta = None
                passed = False
                reverse = False
                counts[name]["null"] += 1
            else:
                delta = float(vals[left]) - float(vals[right])
                passed = delta > 0.0
                reverse = delta < 0.0
                counts[name]["pass"] += int(passed)
                counts[name]["reverse"] += int(reverse)
            row[f"{name}_delta"] = delta
            row[f"{name}_pass"] = passed
            row[f"{name}_reverse"] = reverse
        seed_rows.append(row)
    rule = SEED_DIRECTION_RULES.get(int(expected_cells)) if expected_cells is not None else None
    if rule:
        passed = all(
            counts[name]["pass"] >= int(rule["min_adjacent_pass"])
            and counts[name]["reverse"] <= int(rule["max_adjacent_reverse"])
            for name in comparisons
        )
    else:
        passed = None
    return {
        "world": world,
        "metric": metric,
        "expected_order": list(order),
        "rule": rule,
        "seed_count": len(seeds),
        "comparison_counts": counts,
        "seed_results": seed_rows,
        "pass": passed,
    }


def decision_summary(cells: list[dict[str, Any]], *, expected_cells: int | None = None) -> dict[str, Any]:
    complete = expected_cells is None or len(cells) == int(expected_cells)
    missing = [cell for cell in cells if cell.get("missing_required_telemetry_fields")]
    statuses_ok = all(str(cell.get("status")) in {"completed", "ok", "success"} for cell in cells)
    primary = [
        _direction(
            cells,
            PRIMARY_WORLD,
            "exposed_rich_net_return_per_1000_available_resource_agent_ticks",
            ("risk_exploit", "balanced_generalist", "safe_cover"),
        ),
        _direction(
            cells,
            PRIMARY_WORLD,
            "safe_covered_selected_parent_events_per_1000_eligible_parent_events_after_zone_use",
            ("safe_cover", "balanced_generalist", "risk_exploit"),
        ),
    ]
    seed_primary = [
        _direction_by_seed(
            cells,
            PRIMARY_WORLD,
            "exposed_rich_net_return_per_1000_available_resource_agent_ticks",
            ("risk_exploit", "balanced_generalist", "safe_cover"),
            expected_cells=expected_cells,
        ),
        _direction_by_seed(
            cells,
            PRIMARY_WORLD,
            "safe_covered_selected_parent_events_per_1000_eligible_parent_events_after_zone_use",
            ("safe_cover", "balanced_generalist", "risk_exploit"),
            expected_cells=expected_cells,
        ),
    ]
    controls = [
        _direction(
            cells,
            world,
            "exposed_rich_net_return_per_1000_available_resource_agent_ticks",
            ("risk_exploit", "balanced_generalist", "safe_cover"),
        )
        for world in CONTROL_WORLDS
    ]
    seed_controls = [
        _direction_by_seed(
            cells,
            world,
            "exposed_rich_net_return_per_1000_available_resource_agent_ticks",
            ("risk_exploit", "balanced_generalist", "safe_cover"),
            expected_cells=expected_cells,
        )
        for world in CONTROL_WORLDS
    ]
    seed_rule_applicable = expected_cells in SEED_DIRECTION_RULES
    aggregate_primary_pass = all(row["pass"] for row in primary)
    seed_primary_pass = all(row["pass"] is True for row in seed_primary) if seed_rule_applicable else None
    primary_pass = bool(seed_primary_pass) if seed_rule_applicable else aggregate_primary_pass
    control_leak = any(row["pass"] for row in controls)
    seed_control_leak = (
        any(row["pass"] is True for row in seed_controls)
        if seed_rule_applicable
        else None
    )
    if not complete:
        decision = "PREDRISK-FIXED-1V-D"
        reason = "incomplete campaign"
    elif not statuses_ok:
        decision = "PREDRISK-FIXED-1V-F"
        reason = "execution status integrity failure"
    elif missing:
        decision = "PREDRISK-FIXED-1V-E"
        reason = "decision-critical telemetry missing"
    elif primary_pass and not control_leak and seed_control_leak is not True:
        decision = "PREDRISK-FIXED-1V-A"
        reason = "primary reciprocal demand directions pass without disabled/control leakage"
    else:
        decision = "PREDRISK-FIXED-1V-B"
        reason = "reciprocal predator-resource fixed-probe demand not supported"
    return {
        "schema_version": SCHEMA_VERSION,
        "expected_cells": expected_cells,
        "observed_cells": len(cells),
        "complete": complete,
        "statuses_ok": statuses_ok,
        "missing_required_telemetry_cell_count": len(missing),
        "seed_direction_rule_applicable": seed_rule_applicable,
        "primary_directions": primary,
        "seed_primary_directions": seed_primary,
        "control_directions": controls,
        "seed_control_directions": seed_controls,
        "control_leak": control_leak,
        "seed_control_leak": seed_control_leak,
        "decision": decision,
        "decision_reason": reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--expected-cells", type=int, default=None)
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    cells = collect_cells(runs_root)
    aggregate = aggregate_cells(cells)
    decision = decision_summary(cells, expected_cells=args.expected_cells)
    out_dir.mkdir(parents=True, exist_ok=True)
    cell_path = out_dir / f"predrisk_fixed1v_{args.label}_cell_metrics.csv"
    agg_path = out_dir / f"predrisk_fixed1v_{args.label}_aggregate_metrics.csv"
    decision_path = out_dir / f"predrisk_fixed1v_{args.label}_decision_summary.json"
    manifest_path = out_dir / f"predrisk_fixed1v_{args.label}_analysis_manifest.json"
    _write_csv(cell_path, cells)
    _write_csv(agg_path, aggregate)
    _write_json(decision_path, decision)
    _write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "label": args.label,
            "runs_root": str(runs_root),
            "cell_metrics": str(cell_path),
            "aggregate_metrics": str(agg_path),
            "decision_summary": str(decision_path),
            "cell_metrics_sha256": _sha256(cell_path),
            "aggregate_metrics_sha256": _sha256(agg_path),
            "decision_summary_sha256": _sha256(decision_path),
        },
    )
    print(json.dumps({"decision": decision["decision"], "observed_cells": len(cells), "manifest": str(manifest_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
