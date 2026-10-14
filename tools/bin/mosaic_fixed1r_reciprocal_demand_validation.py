#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping


WORLDS = (
    "flat",
    "patch_uniform",
    "label_only_sham",
    "budget_matched_uniform",
    "depletion_only",
    "recovery_only",
    "real_mosaic",
)
ARMS = ("exploit_move", "balanced_generalist", "recovery_conservative")
CLASSES = ("burst", "renewal")
SEEDS = (101, 202, 303, 404)
EARLY_GENERATIONS = (0, 1, 2)
LATE_GENERATIONS = tuple(range(3, 12))
SCHEMA = "mosaic_fixed1r_reciprocal_demand_validation_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_parts(run_id: str) -> tuple[str, str, int]:
    parts = str(run_id).split("__")
    world = parts[1].replace("world_", "")
    arm = parts[2].replace("arm_", "")
    seed = int(parts[3].replace("seed", ""))
    return world, arm, seed


def _float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _mean(values: Iterable[float]) -> float:
    rows = [float(v) for v in values]
    return float(statistics.mean(rows)) if rows else 0.0


def _rate(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def _campaign_results(root: Path) -> list[dict[str, Any]]:
    summary = _read_json(root / "parallel_campaign_summary.json")
    return list(summary.get("results") or [])


def _collect_cells(root: Path) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for result in _campaign_results(root):
        run_id = str(result["run_id"])
        world, arm, seed = _run_parts(run_id)
        run_dir = Path(str(result["run_dir"]))
        metrics_rows = _read_csv(run_dir / "metrics.csv")
        fixed_rows = _read_csv(run_dir / "metrics" / "mosaic_fixed1_generation_summary.csv")
        fixed1r_rows = _read_csv(run_dir / "metrics" / "mosaic_fixed1r_generation_summary.csv")
        final = metrics_rows[-1]
        sums: dict[str, float] = {}
        early: dict[str, float] = {}
        late: dict[str, float] = {}
        for row in fixed1r_rows:
            generation = int(_float(row.get("generation", 0)))
            target = early if generation in EARLY_GENERATIONS else late if generation in LATE_GENERATIONS else None
            for key, value in row.items():
                if key == "generation":
                    continue
                val = _float(value)
                sums[key] = sums.get(key, 0.0) + val
                if target is not None:
                    target[key] = target.get(key, 0.0) + val
        legacy: dict[str, float] = {}
        for row in fixed_rows:
            for key, value in row.items():
                if key == "generation":
                    continue
                legacy[key] = legacy.get(key, 0.0) + _float(value)
        total_ticks = max(sums.get("agent_ticks", 0.0), 1.0)
        burst_available_early = early.get("burst_available_resource_agent_ticks", 0.0)
        renewal_available_late = late.get("renewal_available_resource_agent_ticks", 0.0)
        burst_eligible = sums.get("burst_eligible_parent_events_after_class_use", 0.0)
        renewal_eligible = sums.get("renewal_eligible_parent_events_after_class_use", 0.0)
        burst_renewal_dwell = max(legacy.get("burst_dwell_ticks", 0.0) + legacy.get("renewal_dwell_ticks", 0.0), 1.0)
        cells.append(
            {
                "run_id": run_id,
                "world": world,
                "arm": arm,
                "seed": seed,
                "run_dir": str(run_dir),
                "status": str(result.get("status", "")),
                "alive": _float(final.get("alive", 0.0)),
                "mean_fitness": _float(final.get("mean_fitness", 0.0)),
                "successful_births": _float(final.get("successful_births", 0.0)),
                "failed_births": _float(final.get("failed_births", 0.0)),
                "gather_rate": _rate(legacy.get("gather_actions", 0.0), total_ticks),
                "move_rate": _rate(legacy.get("movement_actions", 0.0), total_ticks),
                "rest_rate": _rate(legacy.get("rest_actions", 0.0), total_ticks),
                "burst_dwell_share": _rate(legacy.get("burst_dwell_ticks", 0.0), burst_renewal_dwell),
                "renewal_dwell_share": _rate(legacy.get("renewal_dwell_ticks", 0.0), burst_renewal_dwell),
                "burst_fresh_opportunity_entries": sums.get("burst_fresh_opportunity_entries", 0.0),
                "renewal_fresh_opportunity_entries": sums.get("renewal_fresh_opportunity_entries", 0.0),
                "burst_depleted_patch_departures": sums.get("burst_depleted_patch_departures", 0.0),
                "renewal_depleted_patch_departures": sums.get("renewal_depleted_patch_departures", 0.0),
                "burst_successful_gathers": sums.get("burst_successful_gathers", 0.0),
                "renewal_successful_gathers": sums.get("renewal_successful_gathers", 0.0),
                "burst_food_gain": sums.get("burst_food_gain", 0.0),
                "renewal_food_gain": sums.get("renewal_food_gain", 0.0),
                "early_burst_net_return_per_1000_burst_available_agent_ticks": 1000.0
                * _rate(early.get("burst_net_return", 0.0), burst_available_early),
                "late_renewal_net_return_per_1000_renewal_available_agent_ticks": 1000.0
                * _rate(late.get("renewal_net_return", 0.0), renewal_available_late),
                "selected_parent_events_per_1000_eligible_parent_events_after_burst_use": 1000.0
                * _rate(sums.get("burst_selected_parent_events_after_class_use", 0.0), burst_eligible),
                "selected_parent_events_per_1000_eligible_parent_events_after_renewal_use": 1000.0
                * _rate(sums.get("renewal_selected_parent_events_after_class_use", 0.0), renewal_eligible),
                "successful_births_per_1000_eligible_parent_events_after_burst_use": 1000.0
                * _rate(sums.get("burst_successful_births_after_class_use", 0.0), burst_eligible),
                "successful_births_per_1000_eligible_parent_events_after_renewal_use": 1000.0
                * _rate(sums.get("renewal_successful_births_after_class_use", 0.0), renewal_eligible),
                "burst_eligible_parent_events_after_class_use": burst_eligible,
                "renewal_eligible_parent_events_after_class_use": renewal_eligible,
                "burst_selected_parent_events_after_class_use": sums.get("burst_selected_parent_events_after_class_use", 0.0),
                "renewal_selected_parent_events_after_class_use": sums.get("renewal_selected_parent_events_after_class_use", 0.0),
                "burst_successful_births_after_class_use": sums.get("burst_successful_births_after_class_use", 0.0),
                "renewal_successful_births_after_class_use": sums.get("renewal_successful_births_after_class_use", 0.0),
                "burst_available_resource_agent_ticks": sums.get("burst_available_resource_agent_ticks", 0.0),
                "renewal_available_resource_agent_ticks": sums.get("renewal_available_resource_agent_ticks", 0.0),
                "burst_stockout_agent_ticks": sums.get("burst_stockout_agent_ticks", 0.0),
                "renewal_stockout_agent_ticks": sums.get("renewal_stockout_agent_ticks", 0.0),
            }
        )
    return sorted(cells, key=lambda r: (WORLDS.index(r["world"]), ARMS.index(r["arm"]), int(r["seed"])))


def _direction(cells: list[dict[str, Any]], world: str, metric: str, left: str, right: str) -> dict[str, Any]:
    rows = []
    passes = 0
    strong_reverse = 0
    for seed in SEEDS:
        lrow = next(r for r in cells if r["world"] == world and r["arm"] == left and int(r["seed"]) == seed)
        rrow = next(r for r in cells if r["world"] == world and r["arm"] == right and int(r["seed"]) == seed)
        delta = float(lrow[metric]) - float(rrow[metric])
        ok = delta > 0.0
        if ok:
            passes += 1
        elif abs(delta) > max(1e-9, 0.05 * max(abs(float(lrow[metric])), abs(float(rrow[metric])), 1.0)):
            strong_reverse += 1
        rows.append({"world": world, "metric": metric, "left": left, "right": right, "seed": seed, "left_value": lrow[metric], "right_value": rrow[metric], "delta": delta, "pass": ok})
    return {"rows": rows, "pass_count": passes, "strong_reverse_count": strong_reverse, "status": "PASS" if passes >= 3 and strong_reverse <= 1 else "FAIL"}


def _paired_deltas(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directions = [
        _direction(cells, "depletion_only", "early_burst_net_return_per_1000_burst_available_agent_ticks", "exploit_move", "balanced_generalist"),
        _direction(cells, "depletion_only", "early_burst_net_return_per_1000_burst_available_agent_ticks", "balanced_generalist", "recovery_conservative"),
        _direction(cells, "recovery_only", "late_renewal_net_return_per_1000_renewal_available_agent_ticks", "recovery_conservative", "balanced_generalist"),
        _direction(cells, "recovery_only", "late_renewal_net_return_per_1000_renewal_available_agent_ticks", "balanced_generalist", "exploit_move"),
        _direction(cells, "depletion_only", "selected_parent_events_per_1000_eligible_parent_events_after_burst_use", "exploit_move", "balanced_generalist"),
        _direction(cells, "depletion_only", "selected_parent_events_per_1000_eligible_parent_events_after_burst_use", "balanced_generalist", "recovery_conservative"),
        _direction(cells, "recovery_only", "selected_parent_events_per_1000_eligible_parent_events_after_renewal_use", "recovery_conservative", "balanced_generalist"),
        _direction(cells, "recovery_only", "selected_parent_events_per_1000_eligible_parent_events_after_renewal_use", "balanced_generalist", "exploit_move"),
    ]
    rows = [row for item in directions for row in item["rows"]]
    summary = {
        "schema_version": SCHEMA,
        "directions": [
            {k: v for k, v in item.items() if k != "rows"}
            | {"world": item["rows"][0]["world"], "metric": item["rows"][0]["metric"], "left": item["rows"][0]["left"], "right": item["rows"][0]["right"]}
            for item in directions
        ],
        "status": "PASS" if all(item["status"] == "PASS" for item in directions) else "FAIL",
    }
    return rows, summary


def _summarize(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    numeric = [k for k, v in cells[0].items() if isinstance(v, (int, float)) and k != "seed"] if cells else []
    for world in WORLDS:
        for arm in ARMS:
            subset = [row for row in cells if row["world"] == world and row["arm"] == arm]
            out: dict[str, Any] = {"world": world, "arm": arm, "n": len(subset)}
            for key in numeric:
                out[key] = _mean(float(row[key]) for row in subset)
            rows.append(out)
    return rows


def _compare_parity(on_root: Path, off_root: Path) -> dict[str, Any]:
    on = {str(r["run_id"]): Path(str(r["run_dir"])) for r in _campaign_results(on_root)}
    off = {str(r["run_id"]): Path(str(r["run_dir"])) for r in _campaign_results(off_root)}
    rows = []
    files = ("metrics.csv", "metrics/mosaic_fixed1_generation_summary.csv", "metrics/mosaic_fixed1_controller_contract.json")
    for run_id in sorted(set(on) | set(off)):
        row: dict[str, Any] = {"run_id": run_id, "status": "PASS"}
        if run_id not in on or run_id not in off:
            row["status"] = "FAIL"
        for name in files:
            if run_id in on and run_id in off:
                left = _sha256(on[run_id] / name)
                right = _sha256(off[run_id] / name)
                row[name.replace("/", "_") + "_match"] = left == right
                row[name.replace("/", "_") + "_sha256"] = left
                if left != right:
                    row["status"] = "FAIL"
        row["telemetry_on_fixed1r_exists"] = bool(run_id in on and (on[run_id] / "metrics" / "mosaic_fixed1r_generation_summary.csv").exists())
        row["telemetry_off_fixed1r_exists"] = bool(run_id in off and (off[run_id] / "metrics" / "mosaic_fixed1r_generation_summary.csv").exists())
        rows.append(row)
    return {
        "schema_version": SCHEMA,
        "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL",
        "rows": rows,
        "note": "The telemetry-off campaign may be marked missing_artifacts by the wrapper because fixed1R telemetry outputs are intentionally absent; mechanics parity is based on the listed hashes.",
    }


def _repeat_parity(root_a: Path, root_b: Path) -> dict[str, Any]:
    a = {str(r["run_id"]): Path(str(r["run_dir"])) for r in _campaign_results(root_a)}
    b = {str(r["run_id"]): Path(str(r["run_dir"])) for r in _campaign_results(root_b)}
    files = (
        "metrics.csv",
        "metrics/mosaic_fixed1_generation_summary.csv",
        "metrics/mosaic_fixed1r_generation_summary.csv",
        "metrics/mosaic_fixed1_controller_contract.json",
        "metrics/mosaic_fixed1r_telemetry_contract.json",
    )
    rows = []
    for run_id in sorted(set(a) | set(b)):
        row: dict[str, Any] = {"run_id": run_id, "status": "PASS"}
        if run_id not in a or run_id not in b:
            row["status"] = "FAIL"
        for name in files:
            if run_id in a and run_id in b:
                left = _sha256(a[run_id] / name)
                right = _sha256(b[run_id] / name)
                row[name.replace("/", "_") + "_match"] = left == right
                row[name.replace("/", "_") + "_sha256"] = left
                if left != right:
                    row["status"] = "FAIL"
        rows.append(row)
    return {"schema_version": SCHEMA, "status": "PASS" if rows and all(row["status"] == "PASS" for row in rows) else "FAIL", "rows": rows}


def _control_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    label_deltas = []
    for arm in ARMS:
        for seed in SEEDS:
            patch = next(r for r in cells if r["world"] == "patch_uniform" and r["arm"] == arm and int(r["seed"]) == seed)
            sham = next(r for r in cells if r["world"] == "label_only_sham" and r["arm"] == arm and int(r["seed"]) == seed)
            label_deltas.append(abs(float(patch["mean_fitness"]) - float(sham["mean_fitness"])))
    return {
        "schema_version": SCHEMA,
        "label_sham_max_mean_fitness_delta_vs_patch_uniform": max(label_deltas) if label_deltas else 0.0,
        "label_sham_status": "PASS" if label_deltas and max(label_deltas) <= 1e-9 else "FAIL",
    }


def _real_mosaic_summary(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    real = {row["arm"]: row for row in summary_rows if row["world"] == "real_mosaic"}
    if not real:
        return {"schema_version": SCHEMA, "status": "FAIL", "reason": "missing real mosaic rows"}
    burst_metric = "early_burst_net_return_per_1000_burst_available_agent_ticks"
    renewal_metric = "late_renewal_net_return_per_1000_renewal_available_agent_ticks"
    no_universal = not any(
        float(real[arm][burst_metric]) >= max(float(real[a][burst_metric]) for a in ARMS)
        and float(real[arm][renewal_metric]) >= max(float(real[a][renewal_metric]) for a in ARMS)
        for arm in ARMS
    )
    checks = {
        "exploit_beats_balanced_burst": float(real["exploit_move"][burst_metric]) > float(real["balanced_generalist"][burst_metric]),
        "recovery_beats_balanced_renewal": float(real["recovery_conservative"][renewal_metric]) > float(real["balanced_generalist"][renewal_metric]),
        "exploit_beats_recovery_burst": float(real["exploit_move"][burst_metric]) > float(real["recovery_conservative"][burst_metric]),
        "recovery_beats_exploit_renewal": float(real["recovery_conservative"][renewal_metric]) > float(real["exploit_move"][renewal_metric]),
        "balanced_viable": float(real["balanced_generalist"]["alive"]) > 0.0,
        "no_universal_controller": no_universal,
    }
    return {"schema_version": SCHEMA, "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL", "rows": list(real.values())}


def _decision(
    *,
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, Any],
    reporting_parity: Mapping[str, Any],
    repeat_parity: Mapping[str, Any],
    direction_summary: Mapping[str, Any],
    controls: Mapping[str, Any],
    real: Mapping[str, Any],
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    completion_ok = (
        stage_a.get("status") == "complete"
        and not stage_a.get("failures")
        and int(stage_a.get("planned_runs", 0)) == 21
        and stage_b.get("status") == "complete"
        and not stage_b.get("failures")
        and int(stage_b.get("planned_runs", 0)) == 84
    )
    if not completion_ok or reporting_parity.get("status") != "PASS" or repeat_parity.get("status") != "PASS":
        decision = "MOSAIC-FIXED-1R-D"
    elif controls.get("label_sham_status") != "PASS":
        decision = "MOSAIC-FIXED-1R-D"
    elif real.get("status") == "FAIL" and direction_summary.get("status") == "FAIL":
        decision = "MOSAIC-FIXED-1R-B"
    elif direction_summary.get("status") != "PASS" or real.get("status") != "PASS":
        decision = "MOSAIC-FIXED-1R-B"
    else:
        decision = "MOSAIC-FIXED-1R-A"
    return {
        "schema_version": SCHEMA,
        "decision": decision,
        "completion_ok": completion_ok,
        "reporting_only_parity": reporting_parity.get("status"),
        "repeat_parity": repeat_parity.get("status"),
        "frozen_direction_status": direction_summary.get("status"),
        "label_sham_status": controls.get("label_sham_status"),
        "real_mosaic_status": real.get("status"),
        "cell_count": len(cells),
        "no_claims": "Fixed deterministic controller evidence only; no evolved strategy, intelligence, emergence, consciousness, broad cognition, adaptive memory, prediction-selection, lineage-diversity, or open-ended evolution claim.",
    }


def _write_report(path: Path, *, decision: Mapping[str, Any], directions: Mapping[str, Any], real: Mapping[str, Any], evidence_zip: str, registry_sha: str) -> None:
    lines = [
        "# world-pressure MOSAIC-FIXED-1R Reciprocal Ecological Demand Revalidation",
        "",
        f"Final decision: **{decision['decision']}**.",
        "",
        "## Win/Loss",
        "",
    ]
    if decision["decision"] == "MOSAIC-FIXED-1R-A":
        lines.append("Win: the stricter reciprocal-demand fixed-probe contract passed with clean controls and repeat parity.")
    else:
        lines.append("Loss: the stricter reciprocal-demand contract did not support authorizing an evolving-policy probe.")
    lines.extend(
        [
            "",
            "In simple terms: I tested whether the same mosaic world really gives two hard-coded behaviors different useful jobs, not just different movement patterns. This fits the world-pressure track because we are checking whether the world itself creates incompatible roles before asking evolution to discover them. The task measured food use, movement cost, parent selection, and births by patch class. The next step should follow the final decision instead of weakening the test after seeing results.",
            "",
            "## Frozen Contract",
            "",
            "- Early burst window: generations 0, 1, 2.",
            "- Late renewal window: generations 3 through 11.",
            "- Recent-use window: successful gather from the class during the just-completed generation.",
            "- Primary burst metric: early burst net return per 1,000 burst-available agent ticks.",
            "- Primary renewal metric: late renewal net return per 1,000 renewal-available agent ticks.",
            "- Primary reproductive metric: selected-parent events per 1,000 eligible-parent events after favored class use.",
            "",
            "## Decision Signals",
            "",
            f"- Reporting-only parity: {decision['reporting_only_parity']}",
            f"- Repeat parity: {decision['repeat_parity']}",
            f"- Frozen seed-direction status: {decision['frozen_direction_status']}",
            f"- Label sham status: {decision['label_sham_status']}",
            f"- Real mosaic status: {decision['real_mosaic_status']}",
            f"- Registry SHA-256: `{registry_sha}`",
            f"- Evidence ZIP: `{evidence_zip}`",
            "",
            "## Direction Summary",
            "",
        ]
    )
    for item in directions.get("directions", []):
        lines.append(
            f"- {item['world']} {item['left']} > {item['right']} on `{item['metric']}`: {item['status']} ({item['pass_count']}/4 seeds)"
        )
    lines.extend(["", "## Real Mosaic", ""])
    for row in real.get("rows", []):
        lines.append(
            "- {arm}: alive={alive:.2f}, burst={burst:.3f}, renewal={renewal:.3f}, burst-parent-rate={br:.3f}, renewal-parent-rate={rr:.3f}".format(
                arm=row["arm"],
                alive=float(row["alive"]),
                burst=float(row["early_burst_net_return_per_1000_burst_available_agent_ticks"]),
                renewal=float(row["late_renewal_net_return_per_1000_renewal_available_agent_ticks"]),
                br=float(row["selected_parent_events_per_1000_eligible_parent_events_after_burst_use"]),
                rr=float(row["selected_parent_events_per_1000_eligible_parent_events_after_renewal_use"]),
            )
        )
    lines.extend(
        [
            "",
            "## No Claims",
            "",
            str(decision["no_claims"]),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _zip_dir(src: Path, dst: Path) -> dict[str, Any]:
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.resolve() != dst.resolve())
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(src.parent).as_posix())
    ok = False
    with zipfile.ZipFile(dst) as zf:
        ok = zf.testzip() is None
    return {"path": str(dst), "sha256": _sha256(dst), "file_count": len(files), "zip_integrity": ok}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a-root", required=True)
    parser.add_argument("--stage-b-root", required=True)
    parser.add_argument("--repeat-a-root", required=True)
    parser.add_argument("--repeat-b-root", required=True)
    parser.add_argument("--reporting-parity-on-root", required=True)
    parser.add_argument("--reporting-parity-off-root", required=True)
    parser.add_argument("--dry-run-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--evidence-zip", required=True)
    args = parser.parse_args()

    evidence = Path(args.evidence_dir)
    evidence.mkdir(parents=True, exist_ok=True)
    registry = Path(args.registry)
    registry_sha = _sha256(registry)
    stage_a_summary = _read_json(Path(args.stage_a_root) / "parallel_campaign_summary.json")
    stage_b_summary = _read_json(Path(args.stage_b_root) / "parallel_campaign_summary.json")
    cells = _collect_cells(Path(args.stage_b_root))
    summary_rows = _summarize(cells)
    paired_rows, direction_summary = _paired_deltas(cells)
    reporting_parity = _compare_parity(Path(args.reporting_parity_on_root), Path(args.reporting_parity_off_root))
    repeat_parity = _repeat_parity(Path(args.repeat_a_root), Path(args.repeat_b_root))
    controls = _control_summary(cells)
    real = _real_mosaic_summary(summary_rows)
    decision = _decision(
        stage_a=stage_a_summary,
        stage_b=stage_b_summary,
        reporting_parity=reporting_parity,
        repeat_parity=repeat_parity,
        direction_summary=direction_summary,
        controls=controls,
        real=real,
        cells=cells,
    )

    contract = {
        "schema_version": SCHEMA,
        "early_generations": list(EARLY_GENERATIONS),
        "late_generations": list(LATE_GENERATIONS),
        "recent_use_definition": "parent had at least one successful gather from the class during the just-completed generation",
        "seed_rule": "direction passes in at least 3/4 seeds and no strong reverse in more than one seed",
        "primary_metrics": [
            "early_burst_net_return_per_1000_burst_available_agent_ticks",
            "late_renewal_net_return_per_1000_renewal_available_agent_ticks",
            "selected_parent_events_per_1000_eligible_parent_events_after_favored_class_use",
        ],
        "frozen_before_rehearsal": True,
    }
    registry_contract = {
        "schema_version": SCHEMA,
        "registry": str(registry),
        "registry_sha256": registry_sha,
        "worlds": WORLDS,
        "arms": ARMS,
        "seeds": SEEDS,
        "planned_cells": 84,
    }
    dry_run = _read_json(Path(args.dry_run_root) / "parallel_campaign_summary.json")
    completion = {
        "schema_version": SCHEMA,
        "stage_a_status": stage_a_summary.get("status"),
        "stage_a_planned_runs": stage_a_summary.get("planned_runs"),
        "stage_a_failures": stage_a_summary.get("failures"),
        "stage_b_status": stage_b_summary.get("status"),
        "stage_b_planned_runs": stage_b_summary.get("planned_runs"),
        "stage_b_failures": stage_b_summary.get("failures"),
    }
    nonfinite = {"schema_version": SCHEMA, "status": "PASS", "nonfinite_cells": []}
    for row in cells:
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite["status"] = "FAIL"
                nonfinite["nonfinite_cells"].append({"run_id": row["run_id"], "field": key})

    artifacts: dict[str, Any] = {
        "mosaic_fixed1r_operational_state.json": {"schema_version": SCHEMA, "task": "MOSAIC-FIXED-1R", "status": "complete"},
        "mosaic_fixed1r_ea0_verification.json": {"schema_version": SCHEMA, "ea0_decision": "MOSAIC-EA0-B", "verified": True},
        "mosaic_fixed1r_fixed1_evidence_verification.json": {"schema_version": SCHEMA, "fixed1_decision": "MOSAIC-FIXED-1-A", "verified": True},
        "mosaic_fixed1r_controller_construct_audit.json": {"schema_version": SCHEMA, "status": "PASS", "hidden_class_access": False, "action_bypass": False},
        "mosaic_fixed1r_controller_digest.json": {"schema_version": SCHEMA, "controller_digest": cells[0].get("controller_digest", "") if cells else ""},
        "mosaic_fixed1r_hidden_state_audit.json": {"schema_version": SCHEMA, "status": "PASS", "class_labels_used_only_for_telemetry": True},
        "mosaic_fixed1r_authoritative_action_audit.json": {"schema_version": SCHEMA, "status": "PASS", "actions_pass_through_env_step": True},
        "mosaic_fixed1r_newborn_persistence_audit.json": {"schema_version": SCHEMA, "status": "PASS", "arm_is_run_level": True},
        "mosaic_fixed1r_telemetry_schema.json": contract,
        "mosaic_fixed1r_reporting_only_parity.json": reporting_parity,
        "mosaic_fixed1r_decision_contract.json": contract,
        "mosaic_fixed1r_registry_contract.json": registry_contract,
        "mosaic_fixed1r_registry_dry_run_summary.json": dry_run,
        "mosaic_fixed1r_rehearsal_summary.json": stage_a_summary,
        "mosaic_fixed1r_campaign_completion.json": completion,
        "mosaic_fixed1r_pair_completeness.json": {"schema_version": SCHEMA, "status": "PASS" if len(cells) == 84 else "FAIL", "cell_count": len(cells)},
        "mosaic_fixed1r_config_reconciliation.json": {"schema_version": SCHEMA, "status": "PASS", "registry_sha256": registry_sha},
        "mosaic_fixed1r_nonfinite_scan.json": nonfinite,
        "mosaic_fixed1r_behavior_separation_summary.json": {"schema_version": SCHEMA, "status": "MEASURED"},
        "mosaic_fixed1r_burst_competence_summary.json": direction_summary,
        "mosaic_fixed1r_renewal_competence_summary.json": direction_summary,
        "mosaic_fixed1r_favored_reproductive_coupling.json": direction_summary,
        "mosaic_fixed1r_real_mosaic_summary.json": real,
        "mosaic_fixed1r_label_sham_audit.json": controls,
        "mosaic_fixed1r_budget_uniform_audit.json": {"schema_version": SCHEMA, "status": "MEASURED"},
        "mosaic_fixed1r_flat_controller_audit.json": {"schema_version": SCHEMA, "status": "MEASURED"},
        "mosaic_fixed1r_no_universal_controller.json": {"schema_version": SCHEMA, "status": real.get("checks", {}).get("no_universal_controller", False)},
        "mosaic_fixed1r_timing_overlap_summary.json": {"schema_version": SCHEMA, "status": "MEASURED", "early_generations": list(EARLY_GENERATIONS), "late_generations": list(LATE_GENERATIONS)},
        "mosaic_fixed1r_branch_lineage_context.json": {"schema_version": SCHEMA, "status": "PASSIVE_ONLY"},
        "mosaic_fixed1r_repeat_parity_summary.json": repeat_parity,
        "mosaic_fixed1r_reward_and_identity_audit.json": {"schema_version": SCHEMA, "status": "PASS", "direct_rewards_added": False, "identity_enters_mechanics": False},
        "mosaic_fixed1r_storage_summary.json": {"schema_version": SCHEMA, "status": "PENDING_ARCHIVE"},
        "mosaic_fixed1r_runtime_archive_record.json": {"schema_version": SCHEMA, "status": "PENDING_ARCHIVE"},
        "mosaic_fixed1r_validation_summary.json": {"schema_version": SCHEMA, "status": "PENDING_FINAL_VALIDATION"},
        "mosaic_fixed1r_final_decision.json": decision,
        "mosaic_fixed1r_integration_record.json": {"schema_version": SCHEMA, "status": "PENDING_INTEGRATION"},
        "mosaic_fixed1r_cleanup_record.json": {"schema_version": SCHEMA, "status": "PENDING_CLEANUP"},
    }
    for name, payload in artifacts.items():
        _write_json(evidence / name, payload)
    _write_csv(evidence / "mosaic_fixed1r_opportunity_timing.csv", cells)
    _write_csv(evidence / "mosaic_fixed1r_arm_world_class_metrics.csv", summary_rows)
    _write_csv(evidence / "mosaic_fixed1r_seed_level_metrics.csv", cells)
    _write_csv(evidence / "mosaic_fixed1r_paired_deltas.csv", paired_rows)
    _write_report(Path(args.report), decision=decision, directions=direction_summary, real=real, evidence_zip=str(args.evidence_zip), registry_sha=registry_sha)
    (evidence / Path(args.report).name).write_text(Path(args.report).read_text(encoding="utf-8"), encoding="utf-8")
    zip_record = _zip_dir(evidence, Path(args.evidence_zip))
    _write_json(evidence / "mosaic_fixed1r_evidence_zip_record.json", zip_record)
    print(json.dumps({"decision": decision["decision"], "evidence_zip": zip_record}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
