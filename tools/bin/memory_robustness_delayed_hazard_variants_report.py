#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping


PLANNED_RUNS = 336
SEEDS = (101, 202, 303, 404, 505, 606)
TRAIT_ARMS = ("memory_normal", "memory_ablated")
SUBSTRATES = ("tuned_jungle", "clean_ecology_helper")
HAZARD_CONTEXTS = (
    "matched_no_delayed_hazard_baseline",
    "reference_delayed_hazard",
    "lower_damage_delayed_hazard",
    "higher_damage_delayed_hazard",
    "shorter_delay_delayed_hazard",
    "longer_delay_delayed_hazard",
    "noisy_cue_delayed_hazard",
)
NON_REFERENCE_HAZARDS = (
    "lower_damage_delayed_hazard",
    "higher_damage_delayed_hazard",
    "shorter_delay_delayed_hazard",
    "longer_delay_delayed_hazard",
    "noisy_cue_delayed_hazard",
)
INITS = ("init_legacy", "init_founder_balanced")

DIRECT_REWARD_FIELDS = (
    "direct_cognition_reward_present_flag",
    "direct_internal_selection_pressure_present_flag",
    "direct_prediction_error_selection_pressure_present_flag",
    "direct_development_reward_present_flag",
    "direct_introspection_reward_present_flag",
    "direct_internal_state_reward_present_flag",
    "direct_self_regulation_reward_present_flag",
    "direct_self_modification_reward_present_flag",
    "direct_directed_inheritance_reward_present_flag",
    "direct_parental_investment_reward_present_flag",
    "direct_offspring_quality_reward_present_flag",
)
SUMMARY_METRICS = (
    "alive",
    "mean_offspring_total",
    "memory_cost_charged_total",
    "memory_cost_hypothetical_total",
    "cog_cost_total",
    "wdl_delayed_hazard_enable",
    "wdl_delayed_hazard_agent_cue_exposure_count",
    "wdl_delayed_hazard_agent_active_exposure_count",
    "wdl_delayed_hazard_damage_total",
    "wdl_delayed_hazard_active_without_prior_cue_count",
    "wdl_hazard_avoidance_proxy",
    "wdl_pre_hazard_exit_count",
    "founder_balance_enable",
    "founder_balance_violation_count",
    "lineage_forensics_enabled",
    "lineage_effective_alive_lineages",
    "lineage_alive_share_entropy",
    "final_effective_alive_lineages",
    "final_alive_lineage_entropy",
    "final_top1_alive_share",
    "first_effective_lineage_collapse_generation",
    "branch_candidates_after_collapse",
)
DELTA_METRICS = (
    "alive",
    "mean_offspring_total",
    "wdl_delayed_hazard_agent_cue_exposure_count",
    "wdl_delayed_hazard_agent_active_exposure_count",
    "wdl_delayed_hazard_damage_total",
    "wdl_hazard_avoidance_proxy",
    "wdl_pre_hazard_exit_count",
    "memory_cost_charged_total",
    "final_effective_alive_lineages",
    "final_alive_lineage_entropy",
    "final_top1_alive_share",
    "branch_candidates_after_collapse",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _mean(values: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(clean) / float(len(clean)) if clean else None


def _sum(values: Iterable[float | None]) -> float:
    return float(sum(float(v) for v in values if v is not None and math.isfinite(float(v))))


def _hash_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_paths(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    count = 0
    for path in sorted((p for p in paths if p.is_file()), key=lambda p: str(p)):
        count += 1
        h.update(str(path.name).encode("utf-8"))
        h.update(_hash_file(path).encode("ascii"))
    return h.hexdigest() if count else ""


def _stable_manifest_hash(path: Path) -> str:
    data = _read_json(path)
    if not data:
        return _hash_file(path)
    data.pop("started_at_utc", None)
    data.pop("finished_at_utc", None)
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def _matrix_rows(output_root: Path, exclude_dirs: Iterable[str] = ()) -> list[dict[str, str]]:
    excluded = set(exclude_dirs)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    matrices = sorted(output_root.rglob("matrix.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for matrix_path in matrices:
        if excluded and any(part in excluded for part in matrix_path.parts):
            continue
        for row in _read_csv(matrix_path):
            run_dir = row.get("run_dir") or ""
            if not run_dir or run_dir in seen:
                continue
            if not (Path(run_dir) / "metrics.csv").exists():
                continue
            row["_matrix_path"] = str(matrix_path)
            rows.append(row)
            seen.add(run_dir)
    return rows


def _condition(trait_arm: str) -> str:
    if trait_arm == "memory_normal":
        return "normal"
    if trait_arm == "memory_ablated":
        return "ablated"
    return ""


def _run_record(row: Mapping[str, str]) -> dict[str, Any]:
    run_dir = Path(row["run_dir"])
    metrics_rows = _read_csv(run_dir / "metrics.csv")
    final = metrics_rows[-1] if metrics_rows else {}
    config = _read_json(run_dir / "config.json")
    env_cfg = config.get("env_cfg") or {}
    sim_cfg = config.get("sim_cfg") or {}
    trait_arm = str(row.get("trait_arm") or "")
    seed: int | None = None
    try:
        seed = int(row.get("seed") or "")
    except Exception:
        seed = None
    return {
        "run_id": row.get("run_id") or run_dir.name,
        "runtime_run_id": row.get("runtime_run_id") or run_dir.name,
        "run_dir": str(run_dir),
        "trait_arm": trait_arm,
        "condition": _condition(trait_arm),
        "substrate_arm": str(row.get("substrate_arm") or ""),
        "hazard_context": str(row.get("hazard_context") or ""),
        "init_arm": str(row.get("init_arm") or ""),
        "seed": seed,
        "metrics_rows": len(metrics_rows),
        "metrics_hash": _hash_file(run_dir / "metrics.csv"),
        "config_hash": _hash_file(run_dir / "config.json"),
        "manifest_hash": _stable_manifest_hash(run_dir / "manifest.json"),
        "checkpoint_hash": _hash_paths((run_dir / "checkpoint_bundles_v2").rglob("*")),
        "checkpoint_bundle_count": len(list((run_dir / "checkpoint_bundles_v2").glob("bundle_gen*"))),
        "final_metrics": {key: _num(value) for key, value in final.items()},
        "config_flags": {
            "wdl_delayed_hazard_enable": bool(env_cfg.get("wdl_delayed_hazard_enable", False)),
            "wdl_delayed_hazard_observation_enable": bool(env_cfg.get("wdl_delayed_hazard_observation_enable", False)),
            "wdl_delayed_hazard_observation_dropout_rate": _num(env_cfg.get("wdl_delayed_hazard_observation_dropout_rate")) or 0.0,
            "patch_world_enable": bool(env_cfg.get("patch_world_enable", False)),
            "lineage_diversity_reporting_enable": bool(env_cfg.get("lineage_diversity_reporting_enable", True)),
            "lineage_founder_balance_enable": bool(sim_cfg.get("lineage_founder_balance_enable", False)),
            "force_memory_disabled": bool(sim_cfg.get("force_memory_disabled", False)),
            "reset_memory_each_tick": bool(sim_cfg.get("reset_memory_each_tick", False)),
            "disable_prediction_error_selection_pressure": bool(sim_cfg.get("disable_prediction_error_selection_pressure", False)),
            "lineage_qd_archive_enable": bool(sim_cfg.get("lineage_qd_archive_enable", True)),
            "lineage_collapse_forensics_enable": bool(sim_cfg.get("lineage_collapse_forensics_enable", False)),
            "pressure_autopilot_enable": bool(sim_cfg.get("pressure_autopilot_enable", False)),
        },
    }


def _runs(output_root: Path, exclude_dirs: Iterable[str] = ()) -> list[dict[str, Any]]:
    return sorted(
        (_run_record(row) for row in _matrix_rows(output_root, exclude_dirs=exclude_dirs)),
        key=lambda r: (
            str(r["trait_arm"]),
            str(r["substrate_arm"]),
            str(r["hazard_context"]),
            str(r["init_arm"]),
            int(r["seed"] or -1),
        ),
    )


def _reward_audit(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    active: list[dict[str, Any]] = []
    missing: set[str] = set()
    for run in runs:
        final = run.get("final_metrics") or {}
        for field in DIRECT_REWARD_FIELDS:
            val = final.get(field)
            if val is None:
                missing.add(field)
            elif float(val) != 0.0:
                active.append({"run_id": run.get("run_id"), "field": field, "value": val})
        disabled = final.get("prediction_direct_selection_pressure_disabled_flag")
        if disabled is not None and float(disabled) != 1.0:
            active.append({"run_id": run.get("run_id"), "field": "prediction_direct_selection_pressure_disabled_flag", "value": disabled})
    return {
        "schema_version": "memory_robustness_reward_audit_v1",
        "direct_reward_active": bool(active),
        "active_fields": active,
        "missing_fields": sorted(missing),
    }


def _scenario_summary(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r["trait_arm"], r["substrate_arm"], r["hazard_context"], r["init_arm"]) for r in runs})
    for trait_arm, substrate, hazard, init_arm in keys:
        subset = [
            r for r in runs
            if (r["trait_arm"], r["substrate_arm"], r["hazard_context"], r["init_arm"]) == (trait_arm, substrate, hazard, init_arm)
        ]
        row: dict[str, Any] = {
            "trait_arm": trait_arm,
            "substrate_arm": substrate,
            "hazard_context": hazard,
            "init_arm": init_arm,
            "run_count": len(subset),
        }
        for metric in SUMMARY_METRICS:
            row[f"{metric}_mean"] = _mean((r.get("final_metrics") or {}).get(metric) for r in subset)
        out.append(row)
    return out


def _paired_deltas(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["condition"], r["substrate_arm"], r["hazard_context"], r["init_arm"], r["seed"]): r for r in runs}
    rows: list[dict[str, Any]] = []
    for substrate in SUBSTRATES:
        for hazard in HAZARD_CONTEXTS:
            for init_arm in INITS:
                row: dict[str, Any] = {
                    "substrate_arm": substrate,
                    "hazard_context": hazard,
                    "init_arm": init_arm,
                    "paired_seed_count": 0,
                }
                deltas: dict[str, list[float]] = {m: [] for m in DELTA_METRICS}
                for seed in SEEDS:
                    normal = by_key.get(("normal", substrate, hazard, init_arm, seed))
                    ablated = by_key.get(("ablated", substrate, hazard, init_arm, seed))
                    if not normal or not ablated:
                        continue
                    row["paired_seed_count"] += 1
                    for metric in DELTA_METRICS:
                        nv = (normal.get("final_metrics") or {}).get(metric)
                        av = (ablated.get("final_metrics") or {}).get(metric)
                        if nv is not None and av is not None:
                            deltas[metric].append(float(nv) - float(av))
                for metric, values in deltas.items():
                    row[f"delta_{metric}_mean"] = _mean(values)
                row["positive_memory_support"] = _positive_memory_support(row)
                rows.append(row)
    return rows


def _interaction_summary(paired: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["substrate_arm"], r["hazard_context"], r["init_arm"]): r for r in paired}
    out: list[dict[str, Any]] = []
    for substrate in SUBSTRATES:
        for hazard in HAZARD_CONTEXTS:
            if hazard == "matched_no_delayed_hazard_baseline":
                continue
            for init_arm in INITS:
                baseline = by_key.get((substrate, "matched_no_delayed_hazard_baseline", init_arm), {})
                variant = by_key.get((substrate, hazard, init_arm), {})
                row: dict[str, Any] = {
                    "substrate_arm": substrate,
                    "hazard_context": hazard,
                    "init_arm": init_arm,
                    "paired_seed_count": min(int(baseline.get("paired_seed_count") or 0), int(variant.get("paired_seed_count") or 0)),
                }
                for metric in DELTA_METRICS:
                    bv = baseline.get(f"delta_{metric}_mean")
                    vv = variant.get(f"delta_{metric}_mean")
                    row[f"baseline_delta_{metric}_mean"] = bv
                    row[f"variant_delta_{metric}_mean"] = vv
                    row[f"interaction_{metric}_mean"] = (float(vv) - float(bv)) if bv is not None and vv is not None else None
                row["positive_memory_support"] = _positive_memory_support(variant)
                row["positive_interaction"] = _positive_interaction(row)
                row["supportive"] = bool(row["positive_memory_support"] and row["positive_interaction"])
                out.append(row)
    return out


def _positive_memory_support(row: Mapping[str, Any]) -> bool:
    return float(row.get("delta_alive_mean") or 0.0) > 0.0 or float(row.get("delta_mean_offspring_total_mean") or 0.0) > 0.0


def _positive_interaction(row: Mapping[str, Any]) -> bool:
    return float(row.get("interaction_alive_mean") or 0.0) > 0.0 or float(row.get("interaction_mean_offspring_total_mean") or 0.0) > 0.0


def _variant_robustness_map(paired: list[Mapping[str, Any]], interactions: list[Mapping[str, Any]]) -> dict[str, Any]:
    variant_rows: list[dict[str, Any]] = []
    for row in interactions:
        variant_rows.append({
            "hazard_context": row.get("hazard_context"),
            "substrate_arm": row.get("substrate_arm"),
            "init_arm": row.get("init_arm"),
            "paired_seed_count": row.get("paired_seed_count"),
            "positive_memory_support": bool(row.get("positive_memory_support")),
            "positive_interaction": bool(row.get("positive_interaction")),
            "supportive": bool(row.get("supportive")),
            "variant_delta_alive_mean": row.get("variant_delta_alive_mean"),
            "variant_delta_mean_offspring_total_mean": row.get("variant_delta_mean_offspring_total_mean"),
            "interaction_alive_mean": row.get("interaction_alive_mean"),
            "interaction_mean_offspring_total_mean": row.get("interaction_mean_offspring_total_mean"),
        })
    reference_supportive = any(r["hazard_context"] == "reference_delayed_hazard" and r["supportive"] for r in variant_rows)
    non_reference_positive = sorted({
        str(r["hazard_context"]) for r in variant_rows
        if r["hazard_context"] in NON_REFERENCE_HAZARDS and r["positive_memory_support"]
    })
    init_support = sorted({str(r["init_arm"]) for r in variant_rows if r["supportive"]})
    substrate_support = sorted({str(r["substrate_arm"]) for r in variant_rows if r["supportive"]})
    return {
        "schema_version": "memory_robustness_variant_map_v1",
        "rows": variant_rows,
        "reference_supportive": reference_supportive,
        "non_reference_positive_memory_support_variants": non_reference_positive,
        "non_reference_positive_memory_support_variant_count": len(non_reference_positive),
        "supportive_init_arms": init_support,
        "supportive_substrates": substrate_support,
        "support_depends_entirely_on_founder_balanced": bool(init_support == ["init_founder_balanced"]),
    }


def _demand_validation(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for substrate in SUBSTRATES:
        for hazard in HAZARD_CONTEXTS:
            subset = [r for r in runs if r["substrate_arm"] == substrate and r["hazard_context"] == hazard]
            row = {
                "substrate_arm": substrate,
                "hazard_context": hazard,
                "run_count": len(subset),
                "wdl_enabled_expected": hazard != "matched_no_delayed_hazard_baseline",
                "damage_total": _sum((r.get("final_metrics") or {}).get("wdl_delayed_hazard_damage_total") for r in subset),
                "cue_exposure_total": _sum((r.get("final_metrics") or {}).get("wdl_delayed_hazard_agent_cue_exposure_count") for r in subset),
                "active_exposure_total": _sum((r.get("final_metrics") or {}).get("wdl_delayed_hazard_agent_active_exposure_count") for r in subset),
                "active_without_prior_cue_total": _sum((r.get("final_metrics") or {}).get("wdl_delayed_hazard_active_without_prior_cue_count") for r in subset),
            }
            rows.append(row)
    failures = [
        f"{r['substrate_arm']} {r['hazard_context']} demand inactive"
        for r in rows
        if r["wdl_enabled_expected"] and (float(r["damage_total"]) <= 0.0 or float(r["cue_exposure_total"]) <= 0.0)
    ]
    failures.extend(
        f"{r['substrate_arm']} {r['hazard_context']} baseline has hazard activity"
        for r in rows
        if not r["wdl_enabled_expected"] and (float(r["damage_total"]) != 0.0 or float(r["cue_exposure_total"]) != 0.0)
    )
    failures.extend(
        f"{r['substrate_arm']} {r['hazard_context']} active without prior cue"
        for r in rows
        if float(r["active_without_prior_cue_total"]) > 0.0
    )
    return {"schema_version": "memory_robustness_demand_validation_v1", "failures": failures, "rows": rows}


def _cost_sanity(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    missing: set[str] = set()
    negative: list[str] = []
    for r in runs:
        final = r.get("final_metrics") or {}
        for field in ("memory_cost_charged_total", "memory_cost_hypothetical_total", "cog_cost_total"):
            if final.get(field) is None:
                missing.add(field)
            elif float(final[field]) < 0.0:
                negative.append(f"{r.get('run_id')} negative {field}")
    by_key = {(r["condition"], r["substrate_arm"], r["hazard_context"], r["init_arm"], r["seed"]): r for r in runs}
    comparisons: list[dict[str, Any]] = []
    normal_gt = 0
    for substrate in SUBSTRATES:
        for hazard in HAZARD_CONTEXTS:
            for init_arm in INITS:
                for seed in SEEDS:
                    n = by_key.get(("normal", substrate, hazard, init_arm, seed))
                    a = by_key.get(("ablated", substrate, hazard, init_arm, seed))
                    if not n or not a:
                        continue
                    nv = (n.get("final_metrics") or {}).get("memory_cost_charged_total")
                    av = (a.get("final_metrics") or {}).get("memory_cost_charged_total")
                    delta = (float(nv) - float(av)) if nv is not None and av is not None else None
                    if delta is not None and delta > 0.0:
                        normal_gt += 1
                    comparisons.append({
                        "substrate_arm": substrate,
                        "hazard_context": hazard,
                        "init_arm": init_arm,
                        "seed": seed,
                        "normal_minus_ablated_memory_cost": delta,
                    })
    return {
        "schema_version": "memory_robustness_cost_sanity_v1",
        "status": "PASS" if not missing and not negative else "CHECK",
        "missing_cost_fields": sorted(missing),
        "negative_cost_failures": negative,
        "paired_cost_comparison_count": len(comparisons),
        "normal_memory_cost_gt_ablated_pairs": normal_gt,
        "memory_cost_charged_total": _sum((r.get("final_metrics") or {}).get("memory_cost_charged_total") for r in runs),
        "paired_cost_rows": comparisons,
    }


def _lineage_sanity(runs: list[Mapping[str, Any]], interactions: list[Mapping[str, Any]]) -> dict[str, Any]:
    supportive_keys = {
        (r.get("substrate_arm"), r.get("hazard_context"), r.get("init_arm"))
        for r in interactions
        if r.get("supportive")
    }
    supportive_normal = [
        r for r in runs
        if r.get("condition") == "normal"
        and (r.get("substrate_arm"), r.get("hazard_context"), r.get("init_arm")) in supportive_keys
    ]
    dominant = [
        r for r in supportive_normal
        if float((r.get("final_metrics") or {}).get("final_top1_alive_share") or 0.0) >= 0.95
        or float((r.get("final_metrics") or {}).get("final_effective_alive_lineages") or 99.0) <= 1.5
    ]
    dominant_fraction = len(dominant) / float(len(supportive_normal)) if supportive_normal else 0.0
    return {
        "schema_version": "memory_robustness_lineage_sanity_v1",
        "founder_balance_legacy_runs": sum(1 for r in runs if r.get("init_arm") == "init_legacy"),
        "founder_balance_enabled_runs": sum(1 for r in runs if r.get("init_arm") == "init_founder_balanced"),
        "founder_balance_violation_total": _sum((r.get("final_metrics") or {}).get("founder_balance_violation_count") for r in runs),
        "forensics_enabled_runs": sum(1 for r in runs if int((r.get("final_metrics") or {}).get("lineage_forensics_enabled") or 0) == 1),
        "mean_final_top1_alive_share": _mean((r.get("final_metrics") or {}).get("final_top1_alive_share") for r in runs),
        "mean_final_effective_alive_lineages": _mean((r.get("final_metrics") or {}).get("final_effective_alive_lineages") for r in runs),
        "supportive_normal_runs": len(supportive_normal),
        "supportive_normal_single_or_top1_dominant_runs": len(dominant),
        "supportive_normal_single_or_top1_dominant_fraction": dominant_fraction,
        "lineage_confounded": bool(supportive_normal and dominant_fraction >= 0.75),
    }


def _integrity(runs: list[Mapping[str, Any]], reward: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    completed = [r for r in runs if int(r.get("metrics_rows") or 0) > 0]
    if len(completed) != PLANNED_RUNS:
        failures.append(f"completed_runs={len(completed)} planned_runs={PLANNED_RUNS}")
    for r in completed:
        run_id = str(r.get("run_id"))
        hazard = str(r.get("hazard_context") or "")
        init_arm = str(r.get("init_arm") or "")
        condition = str(r.get("condition") or "")
        flags = r.get("config_flags") or {}
        final = r.get("final_metrics") or {}
        wdl_expected = hazard != "matched_no_delayed_hazard_baseline"
        if bool(flags.get("wdl_delayed_hazard_enable")) != wdl_expected:
            failures.append(f"{run_id} delayed-hazard config mismatch")
        if bool(final.get("wdl_delayed_hazard_enable") or 0.0) != wdl_expected:
            failures.append(f"{run_id} delayed-hazard metric mismatch")
        if wdl_expected:
            if (final.get("wdl_delayed_hazard_agent_cue_exposure_count") or 0.0) <= 0.0:
                failures.append(f"{run_id} missing cue exposure")
            if (final.get("wdl_delayed_hazard_damage_total") or 0.0) <= 0.0:
                failures.append(f"{run_id} missing hazard damage")
        if (final.get("wdl_delayed_hazard_active_without_prior_cue_count") or 0.0) > 0.0:
            failures.append(f"{run_id} active hazard without prior cue")
        if bool(flags.get("lineage_founder_balance_enable")) != (init_arm == "init_founder_balanced"):
            failures.append(f"{run_id} founder-balance config mismatch")
        if int(final.get("founder_balance_enable") or 0) != int(init_arm == "init_founder_balanced"):
            failures.append(f"{run_id} founder-balance metric mismatch")
        ablated = condition == "ablated"
        if bool(flags.get("force_memory_disabled")) != ablated or bool(flags.get("reset_memory_each_tick")) != ablated:
            failures.append(f"{run_id} memory ablation config mismatch")
        if ablated and (final.get("memory_force_disabled_active_flag") or 0.0) <= 0.0:
            failures.append(f"{run_id} memory ablation inactive")
        if not bool(flags.get("disable_prediction_error_selection_pressure")):
            failures.append(f"{run_id} prediction direct pressure not disabled")
        if bool(flags.get("lineage_qd_archive_enable")):
            failures.append(f"{run_id} lineage qd archive unexpectedly enabled")
        if not bool(flags.get("lineage_collapse_forensics_enable")) or int(final.get("lineage_forensics_enabled") or 0) != 1:
            failures.append(f"{run_id} lineage forensics disabled")
        if bool(flags.get("pressure_autopilot_enable")):
            failures.append(f"{run_id} pressure autopilot unexpectedly enabled")
        if int(r.get("checkpoint_bundle_count") or 0) <= 0:
            failures.append(f"{run_id} missing checkpoint bundles")
        for metric, value in final.items():
            if value is not None and not math.isfinite(float(value)):
                failures.append(f"{run_id} non-finite metric {metric}")
                break
    if bool(reward.get("direct_reward_active")):
        failures.append("direct reward audit active")
    return {
        "schema_version": "memory_robustness_integrity_v1",
        "planned_runs": PLANNED_RUNS,
        "discovered_runs": len(runs),
        "completed_runs": len(completed),
        "failed_or_missing_runs": max(0, PLANNED_RUNS - len(completed)),
        "failures": failures,
    }


def _decision(
    integrity: Mapping[str, Any],
    reward: Mapping[str, Any],
    demand: Mapping[str, Any],
    cost: Mapping[str, Any],
    lineage: Mapping[str, Any],
    variant_map: Mapping[str, Any],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    support = {
        "reference_supportive": bool(variant_map.get("reference_supportive")),
        "non_reference_positive_memory_support_variant_count": int(variant_map.get("non_reference_positive_memory_support_variant_count") or 0),
        "supportive_init_arms": list(variant_map.get("supportive_init_arms") or []),
        "supportive_substrates": list(variant_map.get("supportive_substrates") or []),
        "support_depends_entirely_on_founder_balanced": bool(variant_map.get("support_depends_entirely_on_founder_balanced")),
        "lineage_confounded": bool(lineage.get("lineage_confounded")),
    }
    if integrity.get("failures") or reward.get("direct_reward_active") or demand.get("failures"):
        return {
            "schema_version": "memory_robustness_decision_v1",
            "decision": "MEMORY-ROBUSTNESS-F",
            "decision_wording": "invalidated; metric integrity, demand validation, config flags, checkpoint bundles, or direct reward audit failed",
            "recommended_next_step": "Infrastructure repair before interpreting memory robustness",
            "support": support,
        }
    if parity and not bool(parity.get("all_required_ok")):
        return {
            "schema_version": "memory_robustness_decision_v1",
            "decision": "MEMORY-ROBUSTNESS-F",
            "decision_wording": "invalidated; repeat parity failed",
            "recommended_next_step": "Repair determinism or artifact hashing before interpreting the battery",
            "support": support,
        }
    if str(cost.get("status")) != "PASS" or int(lineage.get("forensics_enabled_runs") or 0) != PLANNED_RUNS:
        return {
            "schema_version": "memory_robustness_decision_v1",
            "decision": "MEMORY-ROBUSTNESS-E",
            "decision_wording": "reporting/control insufficient; cost sanity or lineage-forensics coverage is incomplete",
            "recommended_next_step": "Repair reporting/control summaries using existing raw data if valid",
            "support": support,
        }
    if float(lineage.get("founder_balance_violation_total") or 0.0) > 0.0:
        return {
            "schema_version": "memory_robustness_decision_v1",
            "decision": "MEMORY-ROBUSTNESS-E",
            "decision_wording": "reporting/control insufficient; founder-balanced lineage violations were detected",
            "recommended_next_step": "Repair founder-balanced initialization controls before interpreting selection",
            "support": support,
        }
    if not support["reference_supportive"] or support["non_reference_positive_memory_support_variant_count"] <= 1:
        return {
            "schema_version": "memory_robustness_decision_v1",
            "decision": "MEMORY-ROBUSTNESS-B",
            "decision_wording": "memory robustness not supported; reference support or non-reference delayed-hazard variant support is insufficient",
            "recommended_next_step": "PAI-1 policy/action interface review or one bounded delayed-hazard demand redesign diagnostic",
            "support": support,
        }
    if support["support_depends_entirely_on_founder_balanced"] or len(support["supportive_init_arms"]) < 2:
        return {
            "schema_version": "memory_robustness_decision_v1",
            "decision": "MEMORY-ROBUSTNESS-D",
            "decision_wording": "mixed or initialization-sensitive memory signal; support depends materially on initialization",
            "recommended_next_step": "One bounded initialization diagnostic, not broader memory claims",
            "support": support,
        }
    if support["lineage_confounded"]:
        return {
            "schema_version": "memory_robustness_decision_v1",
            "decision": "MEMORY-ROBUSTNESS-C",
            "decision_wording": "memory signal persists but lineage forensics indicate a conservative one-lineage or early-lineage confound",
            "recommended_next_step": "Lineage-aware diagnostic or branch replay before broader robustness claims",
            "support": support,
        }
    if support["non_reference_positive_memory_support_variant_count"] >= 3:
        return {
            "schema_version": "memory_robustness_decision_v1",
            "decision": "MEMORY-ROBUSTNESS-A",
            "decision_wording": "robust memory support across delayed-hazard world/demand loop variants under the bounded decision rule",
            "recommended_next_step": "Proceed to the next primary track only as a bounded memory robustness follow-up; keep prediction out of scope",
            "support": support,
        }
    return {
        "schema_version": "memory_robustness_decision_v1",
        "decision": "MEMORY-ROBUSTNESS-D",
        "decision_wording": "mixed delayed-hazard variant support; signal is not broad enough for the robust A decision",
        "recommended_next_step": "One bounded diagnostic focused on the failed variant family",
        "support": support,
    }


def _write_parity(root_a: Path, root_b: Path, out_path: Path) -> dict[str, Any]:
    a = {str(r.get("run_id")): r for r in _runs(root_a)}
    b = {str(r.get("run_id")): r for r in _runs(root_b)}
    items: list[dict[str, Any]] = []
    failures: list[str] = []
    for run_id in sorted(set(a) | set(b)):
        ar = a.get(run_id)
        br = b.get(run_id)
        if not ar or not br:
            failures.append(f"{run_id} missing in one parity root")
            continue
        checks = {
            "metrics_hash": ar["metrics_hash"] == br["metrics_hash"],
            "config_hash": ar["config_hash"] == br["config_hash"],
            "manifest_hash": ar["manifest_hash"] == br["manifest_hash"],
            "checkpoint_hash": ar["checkpoint_hash"] == br["checkpoint_hash"],
            "final_metrics": ar["final_metrics"] == br["final_metrics"],
        }
        required_checks = {
            key: checks[key]
            for key in ("metrics_hash", "config_hash", "checkpoint_hash", "final_metrics")
        }
        if not all(required_checks.values()):
            failures.append(f"{run_id} parity mismatch {required_checks}")
        items.append({"run_id": run_id, "checks": checks, "a": ar, "b": br})
    summary = {
        "schema_version": "memory_robustness_repeat_parity_v1",
        "root_a": str(root_a),
        "root_b": str(root_b),
        "run_count": len(items),
        "all_required_ok": not failures,
        "failures": failures,
        "items": items,
    }
    _write_json(out_path, summary)
    return summary


def _git_status_short(workdir: str) -> str:
    try:
        status = subprocess.check_output(["git", "status", "--short"], cwd=workdir, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"
    return status.replace("\n", "; ") if status else "clean"


def _report(
    *,
    summary: Mapping[str, Any],
    integrity: Mapping[str, Any],
    reward: Mapping[str, Any],
    demand: Mapping[str, Any],
    cost: Mapping[str, Any],
    lineage: Mapping[str, Any],
    variant_map: Mapping[str, Any],
    parity: Mapping[str, Any],
    decision: Mapping[str, Any],
    tests_summary: Mapping[str, Any],
    base_commit: str,
    workdir: str,
    device: str,
) -> str:
    support = decision.get("support") or {}
    lines = [
        "# EIEL Phase IV Memory Robustness Delayed-Hazard Variants Report",
        "",
        "This report evaluates a bounded memory robustness battery across delayed-hazard world/demand loop variants. It does not claim intelligence, emergence, consciousness, adaptive memory, prediction selection, or general cognition.",
        "",
        "## Execution",
        f"- Workdir: `{workdir}`",
        f"- Base commit: `{base_commit}`",
        f"- Device: `{device}`",
        f"- Git status at report time: `{_git_status_short(workdir)}`",
        f"- Planned runs: `{summary.get('planned_runs')}`",
        f"- Completed runs: `{summary.get('completed_runs')}`",
        f"- Evidence ZIP: `{tests_summary.get('evidence_zip', '')}`",
        "",
        "## Battery Design",
        "- Memory arms: `memory_normal`, `memory_ablated`.",
        "- Substrate arms: `tuned_jungle`, `clean_ecology_helper`.",
        "- Delayed-hazard contexts: matched no-hazard baseline, reference, lower damage, higher damage, shorter delay, longer delay, noisy cue.",
        "- Initialization arms: `init_legacy`, `init_founder_balanced`.",
        "- Seeds: `101, 202, 303, 404, 505, 606`.",
        "",
        "## Audit Summary",
        f"- Integrity failures: `{len(integrity.get('failures') or [])}`.",
        f"- Direct reward active: `{bool(reward.get('direct_reward_active'))}`.",
        f"- Delayed-hazard demand validation failures: `{len(demand.get('failures') or [])}`.",
        f"- Cost sanity: `{cost.get('status')}`.",
        f"- Lineage forensics enabled runs: `{lineage.get('forensics_enabled_runs')}`.",
        f"- Repeat parity all required ok: `{parity.get('all_required_ok', 'not_run')}`.",
        "",
        "## Robustness Map",
        f"- Reference supportive: `{support.get('reference_supportive')}`.",
        f"- Non-reference variants with positive memory support: `{support.get('non_reference_positive_memory_support_variant_count')}`.",
        f"- Supportive init arms: `{support.get('supportive_init_arms')}`.",
        f"- Supportive substrates: `{support.get('supportive_substrates')}`.",
        f"- Founder-only support: `{support.get('support_depends_entirely_on_founder_balanced')}`.",
        f"- Lineage-confounded: `{support.get('lineage_confounded')}`.",
        "",
        "## Decision",
        f"{decision.get('decision')}: {decision.get('decision_wording')}",
        "",
        f"Recommended next step: {decision.get('recommended_next_step')}",
        "",
        "## Artifact Notes",
        "- `memory_robustness_summary.json` contains discovered run records and hashes.",
        "- `memory_robustness_variant_map.json` contains the per-variant support rows.",
        "- `memory_robustness_repeat_parity_summary.json` contains repeated-run parity details when parity was run.",
    ]
    if integrity.get("failures"):
        lines.extend(["", "## Integrity Failures"])
        lines.extend(f"- {item}" for item in list(integrity.get("failures") or [])[:80])
    if demand.get("failures"):
        lines.extend(["", "## Demand Failures"])
        lines.extend(f"- {item}" for item in list(demand.get("failures") or [])[:80])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--evidence-zip", required=True)
    parser.add_argument("--base-commit", required=True)
    parser.add_argument("--workdir", default=os.getcwd())
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--parity-a", default="")
    parser.add_argument("--parity-b", default="")
    parser.add_argument("--tests-summary", default="")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.parity_a and args.parity_b:
        _write_parity(Path(args.parity_a), Path(args.parity_b), output_root / "memory_robustness_repeat_parity_summary.json")

    runs = _runs(output_root, exclude_dirs=("repeat_parity_a", "repeat_parity_b"))
    completed = [r for r in runs if int(r.get("metrics_rows") or 0) > 0]
    reward = _reward_audit(completed)
    scenario = _scenario_summary(completed)
    paired = _paired_deltas(completed)
    interactions = _interaction_summary(paired)
    variant_map = _variant_robustness_map(paired, interactions)
    demand = _demand_validation(completed)
    cost = _cost_sanity(completed)
    lineage = _lineage_sanity(completed, interactions)
    integrity = _integrity(completed, reward)
    parity = _read_json(output_root / "memory_robustness_repeat_parity_summary.json")
    decision = _decision(integrity, reward, demand, cost, lineage, variant_map, parity)
    tests_summary = _read_json(Path(args.tests_summary)) if args.tests_summary else {}
    tests_summary["evidence_zip"] = str(Path(args.evidence_zip))
    summary = {
        "schema_version": "memory_robustness_summary_v1",
        "planned_runs": PLANNED_RUNS,
        "discovered_runs": len(runs),
        "completed_runs": len(completed),
        "runs": completed,
    }

    paths = {
        "summary": output_root / "memory_robustness_summary.json",
        "integrity": output_root / "memory_robustness_integrity_summary.json",
        "reward": output_root / "memory_robustness_reward_audit_summary.json",
        "scenario": output_root / "memory_robustness_scenario_summary.csv",
        "paired": output_root / "memory_robustness_paired_deltas.csv",
        "interaction": output_root / "memory_robustness_delayed_hazard_interaction_summary.json",
        "variant_map": output_root / "memory_robustness_variant_map.json",
        "demand": output_root / "memory_robustness_demand_validation_summary.json",
        "cost": output_root / "memory_robustness_cost_sanity_summary.json",
        "lineage": output_root / "memory_robustness_lineage_sanity_summary.json",
        "decision": output_root / "memory_robustness_decision.json",
        "tests": output_root / "memory_robustness_test_command_summary.json",
    }
    _write_json(paths["summary"], summary)
    _write_json(paths["integrity"], integrity)
    _write_json(paths["reward"], reward)
    _write_csv(paths["scenario"], scenario)
    _write_csv(paths["paired"], paired)
    _write_json(paths["interaction"], {"schema_version": "memory_robustness_interaction_v1", "rows": interactions})
    _write_json(paths["variant_map"], variant_map)
    _write_json(paths["demand"], demand)
    _write_json(paths["cost"], cost)
    _write_json(paths["lineage"], lineage)
    _write_json(paths["decision"], decision)
    _write_json(paths["tests"], tests_summary)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _report(
            summary=summary,
            integrity=integrity,
            reward=reward,
            demand=demand,
            cost=cost,
            lineage=lineage,
            variant_map=variant_map,
            parity=parity,
            decision=decision,
            tests_summary=tests_summary,
            base_commit=args.base_commit,
            workdir=args.workdir,
            device=args.device,
        ),
        encoding="utf-8",
    )

    evidence_zip = Path(args.evidence_zip)
    evidence_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(evidence_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in [report_path, *paths.values(), output_root / "memory_robustness_repeat_parity_summary.json"]:
            if path.exists():
                zf.write(path, arcname=path.name)
        for pattern in (
            str(output_root / "e" / "*" / "*" / "matrix.csv"),
            str(output_root / "e" / "*" / "*" / "summary.json"),
            str(output_root / "e" / "*" / "*" / "r" / "*" / "manifest.json"),
            str(output_root / "e" / "*" / "*" / "r" / "*" / "console.txt"),
            str(Path(args.workdir) / "_spec" / "registry" / "v1" / "memory_robustness_delayed_hazard_variants.yaml"),
            str(Path(args.tests_summary)),
        ):
            matches = sorted(glob.glob(pattern))
            if pattern.endswith("manifest.json") or pattern.endswith("console.txt"):
                matches = matches[:16]
            for raw in matches:
                path = Path(raw)
                if not path.exists():
                    continue
                try:
                    arcname = str(path.relative_to(output_root))
                except ValueError:
                    try:
                        arcname = str(path.relative_to(Path(args.workdir)))
                    except ValueError:
                        arcname = path.name
                zf.write(path, arcname=arcname)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
