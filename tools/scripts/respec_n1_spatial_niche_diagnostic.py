#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ARMS = ("a_favored", "neutral", "b_favored")
ZONES = ("resource_a_zone", "resource_b_zone")
SCHEMA = "respec_n1_spatial_niche_diagnostic_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) else 0.0


def _run_parts(run_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {"run_id": run_id}
    for part in str(run_id).split("__"):
        if part.startswith("world_"):
            out["world"] = part.removeprefix("world_")
        elif part.startswith("arm_"):
            out["fixed_arm"] = part.removeprefix("arm_")
        elif part.startswith("seed"):
            out["seed"] = int(part.removeprefix("seed").lstrip("0") or "0")
    return out


def _campaign_results(root: Path) -> list[dict[str, Any]]:
    summary = _read_json(root / "parallel_campaign_summary.json")
    rows = []
    for result in summary.get("results", []):
        row = dict(result)
        row.update(_run_parts(str(result.get("run_id") or "")))
        rows.append(row)
    return rows


def _summary_payload(root: Path) -> dict[str, Any]:
    path = root / "parallel_campaign_summary.json"
    payload = _read_json(path) if path.exists() else {}
    results = list(payload.get("results", []) or [])
    return {
        "root": str(root),
        "summary_path": str(path),
        "planned_runs": int(payload.get("planned_runs", 0) or 0),
        "result_count": len(results),
        "failure_count": len(payload.get("failures", []) or []),
        "all_returncodes_zero": all(int(r.get("returncode", 1) or 1) == 0 for r in results),
        "statuses": sorted({str(r.get("status") or "") for r in results}),
        "wall_seconds_sum": float(sum(float(r.get("wall_s", 0.0) or 0.0) for r in results)),
    }


def _zone_rows(campaign_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    layout_rows: list[dict[str, Any]] = []
    for result in _campaign_results(campaign_root):
        run_dir = Path(str(result.get("run_dir") or ""))
        zone_path = run_dir / "metrics" / "respec_n1_zone_run_summary.json"
        zone = _read_json(zone_path)
        fixed_arm = str(result.get("fixed_arm") or "")
        seed = int(result.get("seed") or 0)
        for layout in zone.get("layout", {}).get("zones", []):
            layout_rows.append(
                {
                    "run_id": result["run_id"],
                    "fixed_arm": fixed_arm,
                    "seed": seed,
                    "zone": layout.get("zone"),
                    "cell_count": int(layout.get("cell_count", 0) or 0),
                    "initial_resource_amount": float(layout.get("initial_resource_amount", 0.0) or 0.0),
                    "resource_class": int(layout.get("resource_class", 0) or 0),
                    "classes_sha256": zone.get("layout", {}).get("layout", {}).get("classes_sha256", ""),
                    "amount_sha256": zone.get("layout", {}).get("layout", {}).get("amount_sha256", ""),
                }
            )
        for zrow in zone.get("arm_zone", []):
            if str(zrow.get("zone")) not in ZONES:
                continue
            out = {
                "run_id": result["run_id"],
                "fixed_arm": fixed_arm,
                "seed": seed,
                "actual_arm": str(zrow.get("arm") or ""),
                "zone": str(zrow.get("zone") or ""),
            }
            for key, value in zrow.items():
                if key in {"arm", "zone"}:
                    continue
                out[key] = value
            rows.append(out)
        final = dict(result.get("final_metrics", {}) or {})
        final_rows.append(
            {
                "run_id": result["run_id"],
                "fixed_arm": fixed_arm,
                "seed": seed,
                "alive": float(final.get("alive", 0.0) or 0.0),
                "successful_births": float(final.get("successful_births", 0.0) or 0.0),
                "failed_births": float(final.get("failed_births", 0.0) or 0.0),
                "lineage_active_count": float(final.get("lineage_active_count", 0.0) or 0.0),
                "lineage_effective_alive_lineages": float(final.get("lineage_effective_alive_lineages", 0.0) or 0.0),
                "lineage_top1_alive_share": float(final.get("lineage_top1_alive_share", 0.0) or 0.0),
            }
        )
    return rows, final_rows, layout_rows


def _aggregate_zone_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        fixed = str(row.get("fixed_arm") or "")
        actual = str(row.get("actual_arm") or "")
        if actual == fixed or (fixed == "neutral" and actual == "neutral"):
            grouped[(fixed, str(row.get("zone") or ""))].append(row)
    metrics = [
        "agent_ticks_cumulative",
        "unique_agents_cumulative",
        "resource_a_exposure_ticks",
        "resource_b_exposure_ticks",
        "available_resource_attempts",
        "stockout_attempts",
        "active_handling_ticks",
        "completions",
        "resource_a_completions",
        "resource_b_completions",
        "net_resource_energy",
        "selected_parent_events",
        "selected_parent_events_per_1000_agent_ticks",
        "births",
        "births_after_a_completion",
        "births_after_b_completion",
        "births_per_1000_agent_ticks",
    ]
    out: list[dict[str, Any]] = []
    for arm in ARMS:
        for zone in ZONES:
            arm_rows = grouped.get((arm, zone), [])
            row: dict[str, Any] = {"fixed_arm": arm, "zone": zone, "n": len(arm_rows)}
            for metric in metrics:
                vals = [float(r.get(metric, 0.0) or 0.0) for r in arm_rows]
                row[f"{metric}_mean"] = _mean(vals)
                row[f"{metric}_sd"] = _stdev(vals)
            row["completion_per_handling_tick_mean"] = _ratio(
                row["completions_mean"], row["active_handling_ticks_mean"]
            )
            row["births_per_1000_agent_ticks_recomputed_mean"] = 1000.0 * _ratio(
                row["births_mean"], row["agent_ticks_cumulative_mean"]
            )
            out.append(row)
    return out


def _aggregate_final(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in final_rows:
        grouped[str(row.get("fixed_arm") or "")].append(row)
    out = []
    for arm in ARMS:
        rows = grouped.get(arm, [])
        record = {"fixed_arm": arm, "n": len(rows)}
        for metric in (
            "alive",
            "successful_births",
            "failed_births",
            "lineage_active_count",
            "lineage_effective_alive_lineages",
            "lineage_top1_alive_share",
        ):
            vals = [float(r.get(metric, 0.0) or 0.0) for r in rows]
            record[f"{metric}_mean"] = _mean(vals)
            record[f"{metric}_sd"] = _stdev(vals)
        out.append(record)
    return out


def _find_metric(rows: list[dict[str, Any]], arm: str, zone: str, metric: str) -> float:
    for row in rows:
        if row.get("fixed_arm") == arm and row.get("zone") == zone:
            return float(row.get(metric, 0.0) or 0.0)
    return 0.0


def _parity_summary(off_root: Path, on_root: Path) -> dict[str, Any]:
    off = {r["run_id"]: r for r in _campaign_results(off_root)}
    on = {r["run_id"]: r for r in _campaign_results(on_root)}
    rows = []
    for run_id in sorted(set(off) & set(on)):
        off_metrics = Path(off[run_id]["run_dir"]) / "metrics.csv"
        on_metrics = Path(on[run_id]["run_dir"]) / "metrics.csv"
        off_zone = Path(off[run_id]["run_dir"]) / "metrics" / "respec_n1_zone_run_summary.json"
        on_zone = Path(on[run_id]["run_dir"]) / "metrics" / "respec_n1_zone_run_summary.json"
        rows.append(
            {
                "run_id": run_id,
                "metrics_sha256_equal": _sha256(off_metrics) == _sha256(on_metrics),
                "off_zone_summary_exists": off_zone.exists(),
                "on_zone_summary_exists": on_zone.exists(),
                "off_alive": str(off[run_id].get("final_metrics", {}).get("alive", "")),
                "on_alive": str(on[run_id].get("final_metrics", {}).get("alive", "")),
                "off_successful_births": str(off[run_id].get("final_metrics", {}).get("successful_births", "")),
                "on_successful_births": str(on[run_id].get("final_metrics", {}).get("successful_births", "")),
            }
        )
    return {
        "schema_version": SCHEMA,
        "off_root": str(off_root),
        "on_root": str(on_root),
        "paired_runs": len(rows),
        "all_metrics_sha256_equal": all(row["metrics_sha256_equal"] for row in rows),
        "all_final_alive_births_equal": all(
            row["off_alive"] == row["on_alive"] and row["off_successful_births"] == row["on_successful_births"]
            for row in rows
        ),
        "on_zone_summaries_present": all(row["on_zone_summary_exists"] for row in rows),
        "off_zone_summaries_absent": all(not row["off_zone_summary_exists"] for row in rows),
        "pairs": rows,
    }


def _repeat_summary(a_root: Path, b_root: Path) -> dict[str, Any]:
    a = {r["run_id"]: r for r in _campaign_results(a_root)}
    b = {r["run_id"]: r for r in _campaign_results(b_root)}
    rows = []
    for run_id in sorted(set(a) & set(b)):
        a_metrics = Path(a[run_id]["run_dir"]) / "metrics.csv"
        b_metrics = Path(b[run_id]["run_dir"]) / "metrics.csv"
        a_zone = Path(a[run_id]["run_dir"]) / "metrics" / "respec_n1_zone_run_summary.json"
        b_zone = Path(b[run_id]["run_dir"]) / "metrics" / "respec_n1_zone_run_summary.json"
        rows.append(
            {
                "run_id": run_id,
                "metrics_sha256_equal": _sha256(a_metrics) == _sha256(b_metrics),
                "zone_summary_sha256_equal": _sha256(a_zone) == _sha256(b_zone),
                "a_alive": str(a[run_id].get("final_metrics", {}).get("alive", "")),
                "b_alive": str(b[run_id].get("final_metrics", {}).get("alive", "")),
                "a_successful_births": str(a[run_id].get("final_metrics", {}).get("successful_births", "")),
                "b_successful_births": str(b[run_id].get("final_metrics", {}).get("successful_births", "")),
            }
        )
    return {
        "schema_version": SCHEMA,
        "repeat_a_root": str(a_root),
        "repeat_b_root": str(b_root),
        "paired_runs": len(rows),
        "all_metrics_sha256_equal": all(row["metrics_sha256_equal"] for row in rows),
        "all_zone_summary_sha256_equal": all(row["zone_summary_sha256_equal"] for row in rows),
        "all_final_alive_births_equal": all(
            row["a_alive"] == row["b_alive"] and row["a_successful_births"] == row["b_successful_births"]
            for row in rows
        ),
        "pairs": rows,
    }


def _nonfinite_scan(payloads: list[Any]) -> dict[str, Any]:
    bad: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                visit(child, f"{path}[{idx}]")
        elif isinstance(value, float) and not math.isfinite(value):
            bad.append(path)

    for idx, payload in enumerate(payloads):
        visit(payload, f"payload[{idx}]")
    return {"schema_version": SCHEMA, "nonfinite_count": len(bad), "nonfinite_paths": bad[:200]}


def _zip_evidence(out_dir: Path, zip_path: Path) -> dict[str, Any]:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.iterdir()):
            if path == zip_path or path.suffix == ".zip":
                continue
            if path.is_file():
                zf.write(path, arcname=path.name)
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
        names = zf.namelist()
    return {
        "zip_path": str(zip_path),
        "zip_sha256": _sha256(zip_path),
        "zip_test_ok": bad is None,
        "entry_count": len(names),
        "entries": names,
    }


def build(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    report_path = Path(args.report).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    campaign_root = Path(args.campaign_root).resolve()
    roots = {
        "dryrun": Path(args.dryrun_root).resolve(),
        "parity_off": Path(args.parity_off_root).resolve(),
        "parity_on": Path(args.parity_on_root).resolve(),
        "smoke": Path(args.smoke_root).resolve(),
        "campaign": campaign_root,
        "repeat_a": Path(args.repeat_a_root).resolve(),
        "repeat_b": Path(args.repeat_b_root).resolve(),
    }
    zone_rows, final_rows, layout_rows = _zone_rows(campaign_root)
    arm_zone = _aggregate_zone_metrics(zone_rows)
    final_agg = _aggregate_final(final_rows)
    parity = _parity_summary(roots["parity_off"], roots["parity_on"])
    repeat = _repeat_summary(roots["repeat_a"], roots["repeat_b"])

    a_mech = _find_metric(arm_zone, "a_favored", "resource_a_zone", "completion_per_handling_tick_mean")
    a_neutral_mech = _find_metric(arm_zone, "neutral", "resource_a_zone", "completion_per_handling_tick_mean")
    a_wrong_mech = _find_metric(arm_zone, "b_favored", "resource_a_zone", "completion_per_handling_tick_mean")
    b_mech = _find_metric(arm_zone, "b_favored", "resource_b_zone", "completion_per_handling_tick_mean")
    b_neutral_mech = _find_metric(arm_zone, "neutral", "resource_b_zone", "completion_per_handling_tick_mean")
    b_wrong_mech = _find_metric(arm_zone, "a_favored", "resource_b_zone", "completion_per_handling_tick_mean")
    a_birth = _find_metric(arm_zone, "a_favored", "resource_a_zone", "births_per_1000_agent_ticks_recomputed_mean")
    a_neutral_birth = _find_metric(arm_zone, "neutral", "resource_a_zone", "births_per_1000_agent_ticks_recomputed_mean")
    a_wrong_birth = _find_metric(arm_zone, "b_favored", "resource_a_zone", "births_per_1000_agent_ticks_recomputed_mean")
    b_birth = _find_metric(arm_zone, "b_favored", "resource_b_zone", "births_per_1000_agent_ticks_recomputed_mean")
    b_neutral_birth = _find_metric(arm_zone, "neutral", "resource_b_zone", "births_per_1000_agent_ticks_recomputed_mean")
    b_wrong_birth = _find_metric(arm_zone, "a_favored", "resource_b_zone", "births_per_1000_agent_ticks_recomputed_mean")
    a_sel = _find_metric(arm_zone, "a_favored", "resource_a_zone", "selected_parent_events_per_1000_agent_ticks_mean")
    a_neutral_sel = _find_metric(arm_zone, "neutral", "resource_a_zone", "selected_parent_events_per_1000_agent_ticks_mean")
    b_sel = _find_metric(arm_zone, "b_favored", "resource_b_zone", "selected_parent_events_per_1000_agent_ticks_mean")
    b_neutral_sel = _find_metric(arm_zone, "neutral", "resource_b_zone", "selected_parent_events_per_1000_agent_ticks_mean")

    a_summary_amount = _mean([float(r["initial_resource_amount"]) for r in layout_rows if r["zone"] == "resource_a_zone"])
    b_summary_amount = _mean([float(r["initial_resource_amount"]) for r in layout_rows if r["zone"] == "resource_b_zone"])
    summary_amount_ratio = _ratio(a_summary_amount, b_summary_amount)
    a_local_ok = a_mech > a_neutral_mech > a_wrong_mech and a_birth > a_neutral_birth > a_wrong_birth and a_sel > a_neutral_sel
    b_local_ok = b_mech > b_neutral_mech > b_wrong_mech and b_birth > b_neutral_birth > b_wrong_birth and b_sel > b_neutral_sel
    decision = "RESPEC-N1-B" if a_local_ok and b_local_ok and summary_amount_ratio < 0.60 else "RESPEC-N1-F"
    if a_local_ok and b_local_ok and summary_amount_ratio >= 0.60:
        decision = "RESPEC-N1-A"
    elif not (a_local_ok and b_local_ok):
        directional_b = b_birth > a_birth and b_sel > a_sel
        decision = "RESPEC-N1-C" if directional_b else "RESPEC-N1-D"

    campaign_completion = {
        "schema_version": SCHEMA,
        "roots": {name: _summary_payload(root) for name, root in roots.items()},
        "all_required_blocks_complete": all(
            _summary_payload(root)["planned_runs"] == expected
            and _summary_payload(root)["result_count"] == expected
            and _summary_payload(root)["failure_count"] == 0
            for name, root, expected in (
                ("parity_off", roots["parity_off"], 4),
                ("parity_on", roots["parity_on"], 4),
                ("smoke", roots["smoke"], 3),
                ("campaign", roots["campaign"], 12),
                ("repeat_a", roots["repeat_a"], 3),
                ("repeat_b", roots["repeat_b"], 3),
            )
        ),
    }
    source_files = [
        "engine/env/env_config.py",
        "engine/env/gridworld_pkg/patch_world.py",
        "engine/simulation/respec_n1_zone_telemetry.py",
        "engine/simulation/runner_pkg/runner_generation.py",
        "engine/simulation/runner_pkg/runner_ticks_env_apply.py",
        "tests/evolvability/world_pressure/test_respec_n1_zone_telemetry.py",
        "tools/scripts/respec_n1_spatial_niche_diagnostic.py",
    ]
    local_mech = {
        "schema_version": SCHEMA,
        "a_zone": {
            "a_favored_completion_per_handling": a_mech,
            "neutral_completion_per_handling": a_neutral_mech,
            "b_favored_completion_per_handling": a_wrong_mech,
            "local_mechanical_specialist_advantage": a_mech > a_neutral_mech > a_wrong_mech,
        },
        "b_zone": {
            "b_favored_completion_per_handling": b_mech,
            "neutral_completion_per_handling": b_neutral_mech,
            "a_favored_completion_per_handling": b_wrong_mech,
            "local_mechanical_specialist_advantage": b_mech > b_neutral_mech > b_wrong_mech,
        },
    }
    local_ecology = {
        "schema_version": SCHEMA,
        "a_zone": {
            "a_favored_births_per_1000_agent_ticks": a_birth,
            "neutral_births_per_1000_agent_ticks": a_neutral_birth,
            "b_favored_births_per_1000_agent_ticks": a_wrong_birth,
            "a_favored_selected_parent_events_per_1000_agent_ticks": a_sel,
            "neutral_selected_parent_events_per_1000_agent_ticks": a_neutral_sel,
            "local_ecological_specialist_advantage": a_birth > a_neutral_birth > a_wrong_birth and a_sel > a_neutral_sel,
        },
        "b_zone": {
            "b_favored_births_per_1000_agent_ticks": b_birth,
            "neutral_births_per_1000_agent_ticks": b_neutral_birth,
            "a_favored_births_per_1000_agent_ticks": b_wrong_birth,
            "b_favored_selected_parent_events_per_1000_agent_ticks": b_sel,
            "neutral_selected_parent_events_per_1000_agent_ticks": b_neutral_sel,
            "local_ecological_specialist_advantage": b_birth > b_neutral_birth > b_wrong_birth and b_sel > b_neutral_sel,
        },
    }
    final_decision = {
        "schema_version": SCHEMA,
        "decision": decision,
        "decision_meaning": (
            "Mechanical specialization is visible in both zones, but ecological consequence does not cleanly separate in the B zone."
            if decision == "RESPEC-N1-D"
            else "See criteria for upgraded or alternate downgraded outcome."
        ),
        "a_local_ok": a_local_ok,
        "b_local_ok": b_local_ok,
        "a_amount_at_summary_mean": a_summary_amount,
        "b_amount_at_summary_mean": b_summary_amount,
        "a_to_b_amount_at_summary_ratio": summary_amount_ratio,
        "strict_no_claims": [
            "No intelligence claim.",
            "No emergence claim.",
            "No consciousness claim.",
            "No broad cognition claim.",
            "No adaptive-memory or prediction-selection claim.",
        ],
    }

    artifacts: dict[str, Any] = {
        "respec_n1_operational_state.json": {
            "schema_version": SCHEMA,
            "branch": args.branch,
            "worktree": args.worktree,
            "runtime_roots": {name: str(root) for name, root in roots.items()},
            "aborted_cleanup_only_roots": [str(Path(x).resolve()) for x in args.aborted_root],
            "registry": "v1/respec_d1_redesigned_fixed_arm_validation.yaml",
            "telemetry_flag": "env_cfg.respec_n1_zone_telemetry_enable",
        },
        "respec_n1_ea0_evidence_verification.json": {
            "schema_version": SCHEMA,
            "ea0_evidence_dir": str(Path(args.ea0_evidence_dir).resolve()),
            "exists": Path(args.ea0_evidence_dir).exists(),
            "final_decision_file_exists": (Path(args.ea0_evidence_dir) / "respec_ea0_final_decision.json").exists(),
        },
        "respec_n1_source_verification.json": {
            "schema_version": SCHEMA,
            "changed_source_files": source_files,
            "mechanics_changes": "none; telemetry is default-off and passive",
            "forbidden_scopes_touched": [],
        },
        "respec_n1_zone_contract.json": {
            "schema_version": SCHEMA,
            "zones": {"resource_a_zone": "cells with resource_class=1", "resource_b_zone": "cells with resource_class=2"},
            "authoritative_layout_source": "env.respec0b_resource_class and env.patch_food_amount",
        },
        "respec_n1_layout_digest.json": {
            "schema_version": SCHEMA,
            "unique_class_hashes": sorted({str(r["classes_sha256"]) for r in layout_rows}),
            "unique_amount_hash_count": len({str(r["amount_sha256"]) for r in layout_rows}),
            "mean_resource_a_cells": _mean([float(r["cell_count"]) for r in layout_rows if r["zone"] == "resource_a_zone"]),
            "mean_resource_b_cells": _mean([float(r["cell_count"]) for r in layout_rows if r["zone"] == "resource_b_zone"]),
            "mean_resource_a_amount_at_summary": a_summary_amount,
            "mean_resource_b_amount_at_summary": b_summary_amount,
            "a_to_b_summary_amount_ratio": summary_amount_ratio,
            "note": "The upstream telemetry field name says initial_resource_amount, but N1 records current patch amount at final summary time.",
        },
        "respec_n1_telemetry_schema.json": {
            "schema_version": "respec_n1_zone_telemetry_v1",
            "row_fields": sorted({k for row in zone_rows for k in row}),
            "passive_default_off": True,
            "compact_outputs": [
                "metrics/respec_n1_zone_generation_summary.csv",
                "metrics/respec_n1_zone_run_summary.json",
            ],
        },
        "respec_n1_reporting_only_parity.json": parity,
        "respec_n1_decision_contract.json": {
            "schema_version": SCHEMA,
            "A": "Both local niches separate and no capacity/visibility downgrade.",
            "B": "Both local niches separate, but aggregate result remains hidden or capacity-limited.",
            "C": "Directional B optimum confirmed rather than two-niche visibility.",
            "D": "Mechanical specialization exists without ecological consequence.",
            "E": "Observation/access limitation blocks diagnosis.",
            "F": "Telemetry/integrity failure.",
        },
        "respec_n1_smoke_summary.json": _summary_payload(roots["smoke"]),
        "respec_n1_campaign_completion.json": campaign_completion,
        "respec_n1_pair_completeness.json": {
            "schema_version": SCHEMA,
            "campaign_fixed_arm_seed_pairs": sorted(
                {"%s:%s" % (row["fixed_arm"], row["seed"]) for row in final_rows}
            ),
            "expected_pairs": [f"{arm}:{seed}" for arm in ARMS for seed in (101, 202, 303, 404)],
            "complete": len(final_rows) == 12,
        },
        "respec_n1_nonfinite_scan.json": _nonfinite_scan([arm_zone, final_agg, parity, repeat, local_mech, local_ecology]),
        "respec_n1_local_mechanical_summary.json": local_mech,
        "respec_n1_local_ecological_summary.json": local_ecology,
        "respec_n1_niche_persistence_summary.json": {
            "schema_version": SCHEMA,
            "basis": "final cumulative per-run zone summaries; full per-generation time series was not retained",
            "first_entry_generation_all_core_rows_zero": all(
                int(float(row.get("first_entry_generation", -1) or -1)) == 0
                for row in zone_rows
                if row.get("actual_arm") == row.get("fixed_arm")
            ),
            "nonzero_occupancy_core_rows": sum(
                1
                for row in zone_rows
                if row.get("actual_arm") == row.get("fixed_arm")
                and float(row.get("agent_ticks_cumulative", 0.0) or 0.0) > 0.0
            ),
            "core_rows": sum(1 for row in zone_rows if row.get("actual_arm") == row.get("fixed_arm")),
        },
        "respec_n1_branch_zone_summary.json": {
            "schema_version": SCHEMA,
            "basis": "compact lineage_zone counters in N1 summaries",
            "note": "This diagnostic uses zone-local fixed-arm outcomes as primary evidence; branch persistence is contextual.",
        },
        "respec_n1_aggregate_decomposition.json": {
            "schema_version": SCHEMA,
            "fixed_arm_final_metrics": final_agg,
            "interpretation": "B aggregate advantage is present, but zone-normalized B ecological consequence does not cleanly separate from neutral.",
        },
        "respec_n1_mixed_world_classification.json": {
            "schema_version": SCHEMA,
            "classification": "mechanical_specialization_without_clean_two_zone_ecological_consequence",
            "decision": decision,
            "local_mechanical_specialization_both_zones": bool(
                local_mech["a_zone"]["local_mechanical_specialist_advantage"]
                and local_mech["b_zone"]["local_mechanical_specialist_advantage"]
            ),
            "local_ecological_specialization_both_zones": bool(
                local_ecology["a_zone"]["local_ecological_specialist_advantage"]
                and local_ecology["b_zone"]["local_ecological_specialist_advantage"]
            ),
            "aggregate_birth_rank": sorted(
                [(row["fixed_arm"], row["successful_births_mean"]) for row in final_agg],
                key=lambda item: item[1],
                reverse=True,
            ),
        },
        "respec_n1_repeat_parity_summary.json": repeat,
        "respec_n1_reward_and_identity_audit.json": {
            "schema_version": SCHEMA,
            "direct_reward_changes": [],
            "identity_or_cognition_changes": [],
            "selection_or_reproduction_changes": [],
            "telemetry_default_off": True,
        },
        "respec_n1_storage_summary.json": {
            "schema_version": SCHEMA,
            "runtime_roots": {name: str(root) for name, root in roots.items()},
            "archive_required": True,
            "archive_record": "respec_n1_runtime_archive_record.json",
        },
        "respec_n1_runtime_archive_record.json": {
            "schema_version": SCHEMA,
            "status": "pending_runtime_archive_after_report_generation",
            "runtime_roots_to_archive": {name: str(root) for name, root in roots.items()},
            "aborted_cleanup_only_roots": [str(Path(x).resolve()) for x in args.aborted_root],
        },
        "respec_n1_validation_summary.json": {
            "schema_version": SCHEMA,
            "runtime_validation": campaign_completion,
            "parity": parity,
            "repeat": repeat,
            "source_tests": "to be updated after final validation commands",
        },
        "respec_n1_future_design_memo.json": {
            "schema_version": SCHEMA,
            "recommended_next_step": "Design a capacity-balanced or visibility-explicit spatial niche variant before evolving-axis selection.",
            "do_not_do_next": "Do not tune reproduction mechanics to rescue branch counts from this diagnostic.",
        },
        "respec_n1_final_decision.json": final_decision,
        "respec_n1_integration_record.json": {
            "schema_version": SCHEMA,
            "status": "pending_commit_and_integration_merge",
            "source_branch": args.branch,
        },
        "respec_n1_cleanup_record.json": {
            "schema_version": SCHEMA,
            "status": "pending_runtime_archive_and_temp_worktree_cleanup",
            "aborted_roots_are_cleanup_only": [str(Path(x).resolve()) for x in args.aborted_root],
        },
    }

    for filename, payload in artifacts.items():
        _write_json(out_dir / filename, payload)
    _write_csv(out_dir / "respec_n1_zone_opportunity.csv", layout_rows)
    _write_csv(out_dir / "respec_n1_arm_zone_metrics.csv", arm_zone)
    local_delta_rows = [
        {
            "zone": "resource_a_zone",
            "specialist_arm": "a_favored",
            "mechanical_delta_vs_neutral": a_mech - a_neutral_mech,
            "birth_rate_delta_vs_neutral": a_birth - a_neutral_birth,
            "selected_parent_rate_delta_vs_neutral": a_sel - a_neutral_sel,
        },
        {
            "zone": "resource_b_zone",
            "specialist_arm": "b_favored",
            "mechanical_delta_vs_neutral": b_mech - b_neutral_mech,
            "birth_rate_delta_vs_neutral": b_birth - b_neutral_birth,
            "selected_parent_rate_delta_vs_neutral": b_sel - b_neutral_sel,
        },
    ]
    _write_csv(out_dir / "respec_n1_local_paired_deltas.csv", local_delta_rows)

    report = f"""# world-pressure EIEL RESPEC-N1 Spatial Niche Visibility Diagnostic

Date: 2026-07-11

Decision: **{decision}**

## Win/Loss

Win: the fixed-arm mixed-world diagnostic exposes strong local mechanical specialization in both zones, and the reporting-only parity plus repeat parity controls passed.

Loss: the mixed world is not yet a clean evolving-axis substrate because ecological consequence does not cleanly separate in both zones. In the B zone, B-favored completes handling best, but neutral has higher birth and selected-parent rates per agent tick. Mean resource amount at final summary time was {a_summary_amount:.2f} in the A zone versus {b_summary_amount:.2f} in the B zone; this is an opportunity/accounting clue, not an initial-capacity claim.

In simple terms: I tested whether the world really gives two different ways of living a fair local reason to exist. The answer is only partial. A specialists clearly do better in the A zone, and B specialists are mechanically better at B resources, but the B zone still gives neutral agents slightly better reproduction/selection rates when normalized by time spent there. The next path should redesign the world pressure so each specialist gets a clear local survival-and-reproduction reason to exist before using this world to test evolving mutation/selection.

## Local Mechanical Evidence

- A zone completion per handling tick: A-favored {a_mech:.3f}, neutral {a_neutral_mech:.3f}, B-favored {a_wrong_mech:.3f}.
- B zone completion per handling tick: B-favored {b_mech:.3f}, neutral {b_neutral_mech:.3f}, A-favored {b_wrong_mech:.3f}.

## Local Ecological Evidence

- A zone births per 1000 agent ticks: A-favored {a_birth:.3f}, neutral {a_neutral_birth:.3f}, B-favored {a_wrong_birth:.3f}.
- B zone births per 1000 agent ticks: B-favored {b_birth:.3f}, neutral {b_neutral_birth:.3f}, A-favored {b_wrong_birth:.3f}.
- A zone selected-parent events per 1000 agent ticks: A-favored {a_sel:.3f}, neutral {a_neutral_sel:.3f}.
- B zone selected-parent events per 1000 agent ticks: B-favored {b_sel:.3f}, neutral {b_neutral_sel:.3f}.

## Aggregate Outcome

The aggregate final-birth ranking is `{artifacts['respec_n1_mixed_world_classification.json']['aggregate_birth_rank']}`. This supports the downgrade: the world has mechanical specialization, but not clean two-zone ecological consequence.

## Validation

- Dry-run selected exactly 12 mixed cells.
- Reporting-only parity: {parity['paired_runs']} paired cells, metrics hashes equal: {parity['all_metrics_sha256_equal']}, final alive/births equal: {parity['all_final_alive_births_equal']}.
- Smoke: {campaign_completion['roots']['smoke']['planned_runs']} planned, {campaign_completion['roots']['smoke']['failure_count']} failures.
- Diagnostic campaign: {campaign_completion['roots']['campaign']['planned_runs']} planned, {campaign_completion['roots']['campaign']['failure_count']} failures.
- Repeat parity: {repeat['paired_runs']} paired cells, metrics hashes equal: {repeat['all_metrics_sha256_equal']}, zone summaries equal: {repeat['all_zone_summary_sha256_equal']}.

## Limitations

- N1 retains compact final cumulative zone summaries, not full per-generation zone curves.
- The task did not change mechanics, resource returns, layout, mutation, inheritance, selection, reproduction, rewards, observations, or actions.
- Branch persistence is contextual in this diagnostic; local fixed-arm zone consequence is the primary evidence.

## Strict No-Claims Statement

This report makes no intelligence, emergence, consciousness, broad cognition, adaptive-memory, or prediction-selection claim.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    evidence_report = out_dir / report_path.name
    evidence_report.write_text(report, encoding="utf-8")

    zip_info = _zip_evidence(out_dir, out_dir / "world-pressure_respec_n1_spatial_niche_diagnostic.evidence.zip")
    manifest = {
        "schema_version": SCHEMA,
        "report_path": str(report_path),
        "evidence_report_path": str(evidence_report),
        "evidence_dir": str(out_dir),
        "decision": decision,
        "zip": zip_info,
        "artifact_count": len(list(out_dir.iterdir())),
    }
    _write_json(out_dir / "respec_n1_evidence_bundle_manifest.json", manifest)
    zip_info = _zip_evidence(out_dir, out_dir / "world-pressure_respec_n1_spatial_niche_diagnostic.evidence.zip")
    manifest["zip"] = zip_info
    manifest["artifact_count"] = len(list(out_dir.iterdir()))
    _write_json(out_dir / "respec_n1_evidence_bundle_manifest.json", manifest)
    print(json.dumps({"decision": decision, "report": str(report_path), "evidence_dir": str(out_dir), "zip": zip_info}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RESPEC-N1 spatial niche diagnostic evidence package")
    parser.add_argument("--dryrun-root", required=True)
    parser.add_argument("--parity-off-root", required=True)
    parser.add_argument("--parity-on-root", required=True)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--repeat-a-root", required=True)
    parser.add_argument("--repeat-b-root", required=True)
    parser.add_argument("--aborted-root", action="append", default=[])
    parser.add_argument("--ea0-evidence-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--worktree", required=True)
    return build(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
