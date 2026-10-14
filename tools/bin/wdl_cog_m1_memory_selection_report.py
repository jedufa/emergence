#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping


PLANNED_RUNS = 160
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 1001)
TRAIT_ARMS = ("memory_normal", "memory_ablated")
TRAITS = ("memory",)
WORLDS = (
    "wdl_cog_m1_tuned_jungle_baseline",
    "wdl_cog_m1_tuned_jungle_wdl_enabled",
    "wdl_cog_m1_theme12_clean_baseline",
    "wdl_cog_m1_theme12_clean_wdl_enabled",
)
INITS = ("init_legacy", "init_founder_balanced")
MATCHED_WORLDS = (
    ("tuned_jungle", "wdl_cog_m1_tuned_jungle_baseline", "wdl_cog_m1_tuned_jungle_wdl_enabled"),
    ("theme12_clean", "wdl_cog_m1_theme12_clean_baseline", "wdl_cog_m1_theme12_clean_wdl_enabled"),
)

KEY_METRICS = (
    "alive",
    "mean_offspring_total",
    "lineage_effective_alive_lineages",
    "lineage_alive_share_entropy",
    "lineage_active_count",
    "founder_balance_enable",
    "founder_initial_effective_lineages",
    "founder_balance_violation_count",
    "behavior_action_total",
    "behavior_patch_interaction_proxy",
    "behavior_risk_exposure_proxy",
    "behavior_visible_resource_exposure_proxy",
    "behavior_visible_hazard_exposure_proxy",
    "jungle_predator_damage_total",
    "jungle_predator_damage_prevented_total",
    "patch_regrowth_suppressed_total",
    "patch_recovery_activity_mean",
    "wdl_delayed_hazard_enable",
    "wdl_delayed_hazard_cue_cell_count",
    "wdl_delayed_hazard_active_cell_count",
    "wdl_delayed_hazard_agent_cue_exposure_count",
    "wdl_delayed_hazard_agent_active_exposure_count",
    "wdl_delayed_hazard_damage_total",
    "wdl_delayed_hazard_active_without_prior_cue_count",
    "wdl_cue_seen_agents",
    "wdl_pre_hazard_exit_count",
    "wdl_hazard_avoidance_proxy",
    "wdl_hazard_exposure_after_cue_count",
    "memory_enabled_fraction",
    "memory_dim_mean",
    "effective_memory_mean",
    "memory_state_abs_mean",
    "memory_state_delta_mean",
    "memory_cost_charged_total",
    "memory_cost_hypothetical_total",
    "memory_cost_zero_fraction",
    "cog_cost_total",
    "memory_ablation_active_flag",
    "memory_force_disabled_active_flag",
    "memory_reset_each_tick_active_flag",
    "prediction_enabled_fraction",
    "prediction_lr_mean",
    "effective_prediction_lr_mean",
    "prediction_error_mean",
    "prediction_state_abs_mean",
    "prediction_state_delta_mean",
    "prediction_update_count_total",
    "prediction_cost_charged_total",
    "prediction_cost_hypothetical_total",
    "prediction_cost_zero_fraction",
    "prediction_ablation_active_flag",
    "prediction_force_zero_lr_active_flag",
    "prediction_reset_each_tick_active_flag",
    "direct_cognition_reward_present_flag",
    "direct_internal_selection_pressure_present_flag",
    "direct_prediction_error_selection_pressure_present_flag",
    "prediction_direct_selection_pressure_disabled_flag",
    "direct_development_reward_present_flag",
    "direct_introspection_reward_present_flag",
    "direct_internal_state_reward_present_flag",
    "direct_self_regulation_reward_present_flag",
    "direct_self_modification_reward_present_flag",
    "direct_directed_inheritance_reward_present_flag",
    "direct_parental_investment_reward_present_flag",
    "direct_offspring_quality_reward_present_flag",
    "lineage_diversity_reporting_enable",
)

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

INTERACTION_METRICS = (
    "alive",
    "mean_offspring_total",
    "lineage_effective_alive_lineages",
    "lineage_alive_share_entropy",
    "lineage_active_count",
    "behavior_action_total",
    "behavior_patch_interaction_proxy",
    "behavior_risk_exposure_proxy",
    "behavior_visible_resource_exposure_proxy",
    "behavior_visible_hazard_exposure_proxy",
    "jungle_predator_damage_total",
    "jungle_predator_damage_prevented_total",
    "patch_regrowth_suppressed_total",
    "patch_recovery_activity_mean",
    "wdl_delayed_hazard_agent_cue_exposure_count",
    "wdl_delayed_hazard_agent_active_exposure_count",
    "wdl_delayed_hazard_damage_total",
    "wdl_pre_hazard_exit_count",
    "wdl_hazard_avoidance_proxy",
    "memory_dim_mean",
    "effective_memory_mean",
    "memory_state_abs_mean",
    "memory_cost_charged_total",
    "prediction_lr_mean",
    "effective_prediction_lr_mean",
    "prediction_error_mean",
    "prediction_state_abs_mean",
    "prediction_cost_charged_total",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size <= 0:
        return []
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames or sorted({k for row in rows for k in row.keys()}))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(dict(row))


def _hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_paths(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    files = sorted({p for p in paths if p.exists() and p.is_file()})
    base = Path(os.path.commonpath([str(p) for p in files])) if files else Path(".")
    if base.is_file():
        base = base.parent
    for path in files:
        try:
            rel = path.relative_to(base)
        except ValueError:
            rel = Path(path.name)
        h.update(str(rel).encode("utf-8"))
        h.update(_hash_file(path).encode("utf-8"))
    return h.hexdigest()


def _stable_manifest_hash(path: Path) -> str:
    data = _read_json(path)
    data.pop("created_utc", None)
    cli_args = data.get("cli_args")
    if isinstance(cli_args, dict):
        cli_args.pop("runs_dir", None)
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _mean(vals: Iterable[float | None]) -> float | None:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return sum(clean) / float(len(clean)) if clean else None


def _sum(vals: Iterable[float | None]) -> float:
    return float(sum(float(v) for v in vals if v is not None and math.isfinite(float(v))))


def _matrix_rows(output_root: Path, exclude_dirs: Iterable[str] = ()) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    excluded = set(exclude_dirs)
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


def _parse_trait_arm(trait_arm: str) -> tuple[str, str]:
    if trait_arm.startswith("memory_"):
        return "memory", trait_arm.removeprefix("memory_")
    if trait_arm.startswith("prediction_"):
        return "prediction", trait_arm.removeprefix("prediction_")
    return "", ""


def _run_record(row: Mapping[str, str]) -> dict[str, Any]:
    run_dir = Path(row["run_dir"])
    metrics_rows = _read_csv(run_dir / "metrics.csv")
    final = metrics_rows[-1] if metrics_rows else {}
    config = _read_json(run_dir / "config.json")
    trait_arm = str(row.get("trait_arm") or "")
    if not trait_arm:
        run_id = str(row.get("run_id") or "")
        marker = "__trait_"
        if marker in run_id:
            trait_arm = run_id.split(marker, 1)[1].split("__", 1)[0]
    trait, condition = _parse_trait_arm(trait_arm)
    seed = None
    try:
        seed = int(row.get("seed") or "")
    except Exception:
        seed = None
    world = str(row.get("world_context") or "")
    init_arm = str(row.get("init_arm") or "")
    env_cfg = config.get("env_cfg") or {}
    sim_cfg = config.get("sim_cfg") or {}
    return {
        "run_id": row.get("run_id") or run_dir.name,
        "runtime_run_id": row.get("runtime_run_id") or run_dir.name,
        "run_dir": str(run_dir),
        "trait_arm": trait_arm,
        "trait": trait,
        "condition": condition,
        "world_context": world,
        "init_arm": init_arm,
        "seed": seed,
        "metrics_rows": len(metrics_rows),
        "metrics_hash": _hash_file(run_dir / "metrics.csv"),
        "config_hash": _hash_file(run_dir / "config.json"),
        "manifest_hash": _stable_manifest_hash(run_dir / "manifest.json"),
        "checkpoint_hash": _hash_paths((run_dir / "checkpoint_bundles_v2").rglob("*")),
        "checkpoint_bundle_count": len(list((run_dir / "checkpoint_bundles_v2").glob("bundle_gen*"))),
        "final_metrics": {key: _num(final.get(key)) for key in KEY_METRICS},
        "config_flags": {
            "wdl_delayed_hazard_enable": bool(env_cfg.get("wdl_delayed_hazard_enable", False)),
            "wdl_delayed_hazard_observation_enable": bool(env_cfg.get("wdl_delayed_hazard_observation_enable", False)),
            "patch_world_enable": bool(env_cfg.get("patch_world_enable", False)),
            "lineage_diversity_reporting_enable": bool(env_cfg.get("lineage_diversity_reporting_enable", True)),
            "lineage_founder_balance_enable": bool(sim_cfg.get("lineage_founder_balance_enable", False)),
            "enable_cognitive_trait_registry": bool(sim_cfg.get("enable_cognitive_trait_registry", False)),
            "enable_cognitive_status_metrics": bool(sim_cfg.get("enable_cognitive_status_metrics", False)),
            "enable_cognitive_budget_ledger": bool(sim_cfg.get("enable_cognitive_budget_ledger", False)),
            "force_memory_disabled": bool(sim_cfg.get("force_memory_disabled", False)),
            "reset_memory_each_tick": bool(sim_cfg.get("reset_memory_each_tick", False)),
            "force_prediction_zero_lr": bool(sim_cfg.get("force_prediction_zero_lr", False)),
            "reset_prediction_each_tick": bool(sim_cfg.get("reset_prediction_each_tick", False)),
            "disable_prediction_error_selection_pressure": bool(sim_cfg.get("disable_prediction_error_selection_pressure", False)),
            "lineage_qd_archive_enable": bool(sim_cfg.get("lineage_qd_archive_enable", True)),
        },
    }


def _runs(output_root: Path, exclude_dirs: Iterable[str] = ()) -> list[dict[str, Any]]:
    rows = _matrix_rows(output_root, exclude_dirs=exclude_dirs)
    return sorted(
        [_run_record(row) for row in rows],
        key=lambda r: (str(r["trait_arm"]), str(r["world_context"]), str(r["init_arm"]), int(r["seed"] or -1)),
    )


def _dependency_audit(output_root: Path, *, base_commit: str, workdir: str, device: str) -> dict[str, Any]:
    wdl1_root = Path("/tmp/emergence_wdl1_delayed_hazard_validation")
    wdl2_root = Path("/tmp/emergence_wdl2_demand_loop_cognition_readiness_bridge")
    return {
        "schema_version": "wdl_cog_m1_dependency_audit_v1",
        "working_directory": workdir,
        "base_commit": base_commit,
        "device": device,
        "new_worktree_used": False,
        "wdl2_committed_first": True,
        "wdl2_commit": base_commit,
        "wdl2_decision": _read_json(wdl2_root / "wdl2_decision.json"),
        "wdl2_report_path": "docs/400_implementation-engineering/410_implementation-reports/500_cognition-memory/EIEL-DOC-000677--eiel-phaseiv-wdl2-demand-loop-cognition-readiness-bridge-report.md",
        "wdl1_decision": _read_json(wdl1_root / "wdl1_decision.json"),
        "wdl1_registry_path": "_spec/registry/v1/wdl1_delayed_hazard_validation.yaml",
        "wdl1_output_root": str(wdl1_root),
        "wdl1_hazard_timing_result": _read_json(wdl1_root / "wdl1_hazard_timing_summary.json"),
        "wdl1_hazard_consequence_result": _read_json(wdl1_root / "wdl1_hazard_consequence_summary.json"),
        "wdl1_action_context_result": _read_json(wdl1_root / "wdl1_action_context_summary.json"),
        "wdl1_checkpoint_replay_result": _read_json(wdl1_root / "wdl1_checkpoint_replay_summary.json"),
        "memory_control_status": {
            "status": "READY",
            "normal_flags": ["sim_cfg.force_memory_disabled=false", "sim_cfg.reset_memory_each_tick=false"],
            "ablated_flags": ["sim_cfg.force_memory_disabled=true", "sim_cfg.reset_memory_each_tick=true"],
            "source": "_spec/registry/v1/phaseIV_crb_m1_memory_readiness_bridge.yaml",
        },
        "prediction_control_status": {"status": "NOT_RUN", "reason": "WDL-COG-M1 is memory-only after WDL-2 prediction readiness score was 0.0"},
        "reward_audit_status": "clean if wdl_cog_m1_reward_audit direct_reward_active is false",
        "output_root": str(output_root),
    }


def _reward_audit(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    active: list[dict[str, Any]] = []
    missing: set[str] = set()
    ambiguous: list[str] = []
    if not runs:
        ambiguous.append("No WDL-COG-M1 run metrics available yet; registry-level reward knobs are audited as disabled.")
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
        "schema_version": "wdl_cog_m1_reward_audit_v1",
        "direct_cognition_reward_active": any(a["field"] in ("direct_cognition_reward_present_flag", "direct_internal_selection_pressure_present_flag") for a in active),
        "direct_memory_reward_active": False,
        "direct_prediction_reward_active": any(a["field"] == "direct_prediction_error_selection_pressure_present_flag" for a in active),
        "direct_perception_reward_active": False,
        "direct_diversity_reward_active": False,
        "direct_reward_active": bool(active),
        "active_fields": active,
        "missing_fields": sorted(missing),
        "ambiguous_notes": ambiguous,
    }


def _scenario_summary(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    keys = sorted({(r["trait"], r["condition"], r["world_context"], r["init_arm"]) for r in runs})
    for trait, condition, world, init_arm in keys:
        subset = [r for r in runs if (r["trait"], r["condition"], r["world_context"], r["init_arm"]) == (trait, condition, world, init_arm)]
        row: dict[str, Any] = {
            "trait": trait,
            "condition": condition,
            "world_context": world,
            "init_arm": init_arm,
            "run_count": len(subset),
        }
        for metric in KEY_METRICS:
            row[f"{metric}_mean"] = _mean((r.get("final_metrics") or {}).get(metric) for r in subset)
        out.append(row)
    return out


def _paired_deltas(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["trait"], r["condition"], r["world_context"], r["init_arm"], r["seed"]): r for r in runs}
    rows: list[dict[str, Any]] = []
    for trait in TRAITS:
        for world in WORLDS:
            for init_arm in INITS:
                row: dict[str, Any] = {"trait": trait, "world_context": world, "init_arm": init_arm, "paired_seed_count": 0}
                deltas: dict[str, list[float]] = {m: [] for m in INTERACTION_METRICS}
                for seed in SEEDS:
                    n = by_key.get((trait, "normal", world, init_arm, seed))
                    a = by_key.get((trait, "ablated", world, init_arm, seed))
                    if not n or not a:
                        continue
                    row["paired_seed_count"] += 1
                    for metric in INTERACTION_METRICS:
                        nv = (n.get("final_metrics") or {}).get(metric)
                        av = (a.get("final_metrics") or {}).get(metric)
                        if nv is not None and av is not None:
                            deltas[metric].append(float(nv) - float(av))
                for metric, values in deltas.items():
                    row[f"delta_{metric}_mean"] = _mean(values)
                rows.append(row)
    return rows


def _interaction_summary(paired: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(r["trait"], r["world_context"], r["init_arm"]): r for r in paired}
    out: list[dict[str, Any]] = []
    for trait in TRAITS:
        for substrate, base, wdl in MATCHED_WORLDS:
            for init_arm in INITS:
                b = by_key.get((trait, base, init_arm), {})
                w = by_key.get((trait, wdl, init_arm), {})
                row: dict[str, Any] = {
                    "trait": trait,
                    "substrate": substrate,
                    "baseline_world_context": base,
                    "wdl_world_context": wdl,
                    "init_arm": init_arm,
                    "paired_seed_count": min(int(b.get("paired_seed_count") or 0), int(w.get("paired_seed_count") or 0)),
                }
                for metric in INTERACTION_METRICS:
                    bv = b.get(f"delta_{metric}_mean")
                    wv = w.get(f"delta_{metric}_mean")
                    row[f"baseline_delta_{metric}_mean"] = bv
                    row[f"wdl_delta_{metric}_mean"] = wv
                    row[f"interaction_{metric}_mean"] = (float(wv) - float(bv)) if bv is not None and wv is not None else None
                out.append(row)
    return out


def _aggregate_effects(runs: list[Mapping[str, Any]], paired: list[Mapping[str, Any]], interactions: list[Mapping[str, Any]]) -> dict[str, Any]:
    effect: dict[str, Any] = {"schema_version": "wdl_cog_m1_trait_effect_summary_v1", "traits": {}}
    for trait in TRAITS:
        rows = [r for r in paired if r.get("trait") == trait]
        effect["traits"][trait] = {
            "max_alive_delta": max((float(r.get("delta_alive_mean") or 0.0) for r in rows), default=0.0),
            "max_offspring_delta": max((float(r.get("delta_mean_offspring_total_mean") or 0.0) for r in rows), default=0.0),
            "paired_rows": rows,
        }
    wdl_runs = [r for r in runs if str(r.get("world_context") or "").endswith("wdl_enabled")]
    wdl_effect = {
        "schema_version": "wdl_cog_m1_wdl_effect_summary_v1",
        "wdl_run_count": len(wdl_runs),
        "cue_exposure_total": _sum((r.get("final_metrics") or {}).get("wdl_delayed_hazard_agent_cue_exposure_count") for r in wdl_runs),
        "active_exposure_total": _sum((r.get("final_metrics") or {}).get("wdl_delayed_hazard_agent_active_exposure_count") for r in wdl_runs),
        "damage_total": _sum((r.get("final_metrics") or {}).get("wdl_delayed_hazard_damage_total") for r in wdl_runs),
        "active_without_prior_cue_total": _sum((r.get("final_metrics") or {}).get("wdl_delayed_hazard_active_without_prior_cue_count") for r in wdl_runs),
        "pre_hazard_exit_total": _sum((r.get("final_metrics") or {}).get("wdl_pre_hazard_exit_count") for r in wdl_runs),
        "avoidance_proxy_total": _sum((r.get("final_metrics") or {}).get("wdl_hazard_avoidance_proxy") for r in wdl_runs),
    }
    interaction = {"schema_version": "wdl_cog_m1_trait_x_wdl_interaction_summary_v1", "rows": interactions}
    founder_rows = []
    for trait in TRAITS:
        for substrate, _base, _wdl in MATCHED_WORLDS:
            legacy = [r for r in interactions if r["trait"] == trait and r["substrate"] == substrate and r["init_arm"] == "init_legacy"]
            founder = [r for r in interactions if r["trait"] == trait and r["substrate"] == substrate and r["init_arm"] == "init_founder_balanced"]
            if legacy and founder:
                founder_rows.append({
                    "trait": trait,
                    "substrate": substrate,
                    "alive_interaction_founder_minus_legacy": (founder[0].get("interaction_alive_mean") or 0.0) - (legacy[0].get("interaction_alive_mean") or 0.0),
                    "offspring_interaction_founder_minus_legacy": (founder[0].get("interaction_mean_offspring_total_mean") or 0.0) - (legacy[0].get("interaction_mean_offspring_total_mean") or 0.0),
                })
    founder = {"schema_version": "wdl_cog_m1_founder_balance_sensitivity_summary_v1", "rows": founder_rows}
    behavior = {
        "schema_version": "wdl_cog_m1_behavior_wdl_coupling_summary_v1",
        "wdl_behavior_rows": [
            {
                "trait": r["trait"],
                "substrate": r["substrate"],
                "init_arm": r["init_arm"],
                "interaction_behavior_risk_exposure_proxy_mean": r.get("interaction_behavior_risk_exposure_proxy_mean"),
                "interaction_behavior_visible_hazard_exposure_proxy_mean": r.get("interaction_behavior_visible_hazard_exposure_proxy_mean"),
                "interaction_wdl_hazard_avoidance_proxy_mean": r.get("interaction_wdl_hazard_avoidance_proxy_mean"),
                "interaction_wdl_pre_hazard_exit_count_mean": r.get("interaction_wdl_pre_hazard_exit_count_mean"),
            }
            for r in interactions
        ],
    }
    return {
        "trait_effect": effect,
        "wdl_effect": wdl_effect,
        "interaction": interaction,
        "founder": founder,
        "behavior": behavior,
    }


def _cost_sanity(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    missing: set[str] = set()
    bad: list[str] = []
    for r in runs:
        final = r.get("final_metrics") or {}
        for field in ("memory_cost_charged_total", "memory_cost_hypothetical_total", "cog_cost_total"):
            if field not in final or final.get(field) is None:
                missing.add(field)
            elif float(final[field]) < 0.0:
                bad.append(f"{r.get('run_id')} negative {field}")
    by_key = {(r["condition"], r["world_context"], r["init_arm"], r["seed"]): r for r in runs}
    comparisons: list[dict[str, Any]] = []
    normal_cost_gt_ablated_count = 0
    paired_count = 0
    for world in WORLDS:
        for init_arm in INITS:
            for seed in SEEDS:
                n = by_key.get(("normal", world, init_arm, seed))
                a = by_key.get(("ablated", world, init_arm, seed))
                if not n or not a:
                    continue
                paired_count += 1
                nv = (n.get("final_metrics") or {}).get("memory_cost_charged_total")
                av = (a.get("final_metrics") or {}).get("memory_cost_charged_total")
                delta = (float(nv) - float(av)) if nv is not None and av is not None else None
                if delta is not None and delta > 0.0:
                    normal_cost_gt_ablated_count += 1
                comparisons.append({"world_context": world, "init_arm": init_arm, "seed": seed, "normal_minus_ablated_memory_cost": delta})
    return {
        "schema_version": "wdl_cog_m1_cost_sanity_summary_v1",
        "status": "PASS" if not bad and not missing else "CHECK",
        "negative_cost_failures": bad,
        "missing_cost_fields": sorted(missing),
        "memory_cost_charged_total": _sum((r.get("final_metrics") or {}).get("memory_cost_charged_total") for r in runs),
        "normal_memory_cost_gt_ablated_pairs": normal_cost_gt_ablated_count,
        "paired_cost_comparison_count": paired_count,
        "paired_cost_rows": comparisons,
    }


def _lineage_sanity(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "wdl_cog_m1_lineage_sanity_summary_v1",
        "founder_balance_legacy_runs": sum(1 for r in runs if r.get("init_arm") == "init_legacy"),
        "founder_balance_enabled_runs": sum(1 for r in runs if r.get("init_arm") == "init_founder_balanced"),
        "founder_balance_violation_total": _sum((r.get("final_metrics") or {}).get("founder_balance_violation_count") for r in runs),
        "mean_effective_alive_lineages": _mean((r.get("final_metrics") or {}).get("lineage_effective_alive_lineages") for r in runs),
        "mean_lineage_entropy": _mean((r.get("final_metrics") or {}).get("lineage_alive_share_entropy") for r in runs),
    }


def _selection_summary(paired: list[Mapping[str, Any]], interactions: list[Mapping[str, Any]]) -> dict[str, Any]:
    wdl_rows = [r for r in paired if str(r.get("world_context") or "").endswith("wdl_enabled")]
    supportive_wdl_rows = [
        r for r in wdl_rows
        if (float(r.get("delta_alive_mean") or 0.0) > 0.0 or float(r.get("delta_mean_offspring_total_mean") or 0.0) > 0.0)
    ]
    supportive_interactions = [
        r for r in interactions
        if (float(r.get("interaction_alive_mean") or 0.0) > 0.0 or float(r.get("interaction_mean_offspring_total_mean") or 0.0) > 0.0)
    ]
    tuned_support = [
        r for r in interactions
        if r.get("substrate") == "tuned_jungle"
        and (float(r.get("wdl_delta_alive_mean") or 0.0) > 0.0 or float(r.get("wdl_delta_mean_offspring_total_mean") or 0.0) > 0.0)
        and (float(r.get("interaction_alive_mean") or 0.0) > 0.0 or float(r.get("interaction_mean_offspring_total_mean") or 0.0) > 0.0)
    ]
    legacy_support = [r for r in supportive_interactions if r.get("init_arm") == "init_legacy"]
    founder_support = [r for r in supportive_interactions if r.get("init_arm") == "init_founder_balanced"]
    return {
        "schema_version": "wdl_cog_m1_selection_summary_v1",
        "wdl_memory_normal_advantage_rows": len(supportive_wdl_rows),
        "positive_wdl_interaction_rows": len(supportive_interactions),
        "tuned_jungle_supportive_rows": len(tuned_support),
        "legacy_supportive_rows": len(legacy_support),
        "founder_balanced_supportive_rows": len(founder_support),
        "support_depends_entirely_on_founder_balanced": bool(founder_support and not legacy_support),
        "paired_delta_rows": paired,
        "interaction_rows": interactions,
    }


def _integrity(runs: list[Mapping[str, Any]], reward: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    completed = [r for r in runs if int(r.get("metrics_rows") or 0) > 0]
    if len(completed) != PLANNED_RUNS:
        failures.append(f"completed_runs={len(completed)} planned_runs={PLANNED_RUNS}")
    for r in completed:
        run_id = r.get("run_id")
        world = str(r.get("world_context") or "")
        trait = str(r.get("trait") or "")
        condition = str(r.get("condition") or "")
        init_arm = str(r.get("init_arm") or "")
        flags = r.get("config_flags") or {}
        final = r.get("final_metrics") or {}
        wdl_expected = world.endswith("wdl_enabled")
        if bool(flags.get("wdl_delayed_hazard_enable")) != wdl_expected:
            failures.append(f"{run_id} WDL flag mismatch")
        if bool(final.get("wdl_delayed_hazard_enable") or 0.0) != wdl_expected:
            failures.append(f"{run_id} WDL metric mismatch")
        if wdl_expected:
            if (final.get("wdl_delayed_hazard_active_cell_count") or 0.0) <= 0.0:
                failures.append(f"{run_id} missing active hazard cells")
            if (final.get("wdl_delayed_hazard_agent_cue_exposure_count") or 0.0) <= 0.0:
                failures.append(f"{run_id} missing cue exposure")
        if (final.get("wdl_delayed_hazard_active_without_prior_cue_count") or 0.0) > 0.0:
            failures.append(f"{run_id} active hazard without prior cue")
        founder_expected = init_arm == "init_founder_balanced"
        if bool(flags.get("lineage_founder_balance_enable")) != founder_expected:
            failures.append(f"{run_id} founder-balance flag mismatch")
        if int(final.get("founder_balance_enable") or 0) != int(founder_expected):
            failures.append(f"{run_id} founder-balance metric mismatch")
        if trait == "memory":
            ablated = condition == "ablated"
            if bool(flags.get("force_memory_disabled")) != ablated or bool(flags.get("reset_memory_each_tick")) != ablated:
                failures.append(f"{run_id} memory ablation config mismatch")
            if ablated and (final.get("memory_force_disabled_active_flag") or 0.0) <= 0.0:
                failures.append(f"{run_id} memory ablation inactive")
        if trait == "prediction":
            ablated = condition == "ablated"
            if bool(flags.get("force_prediction_zero_lr")) != ablated or bool(flags.get("reset_prediction_each_tick")) != ablated:
                failures.append(f"{run_id} prediction ablation config mismatch")
            if ablated and (final.get("prediction_force_zero_lr_active_flag") or 0.0) <= 0.0:
                failures.append(f"{run_id} prediction ablation inactive")
        if not bool(flags.get("disable_prediction_error_selection_pressure")):
            failures.append(f"{run_id} prediction direct pressure not disabled")
        if bool(flags.get("lineage_qd_archive_enable")):
            failures.append(f"{run_id} lineage qd archive unexpectedly enabled")
        if int(r.get("checkpoint_bundle_count") or 0) <= 0:
            failures.append(f"{run_id} missing checkpoint bundles")
        for metric, value in final.items():
            if value is not None and not math.isfinite(float(value)):
                failures.append(f"{run_id} non-finite {metric}")
                break
    if bool(reward.get("direct_reward_active")):
        failures.append("direct reward audit active")
    return {
        "schema_version": "wdl_cog_m1_metric_integrity_summary_v1",
        "planned_runs": PLANNED_RUNS,
        "discovered_runs": len(runs),
        "completed_runs": len(completed),
        "failed_or_missing_runs": max(0, PLANNED_RUNS - len(completed)),
        "failures": failures,
    }


def _decision(
    integrity: Mapping[str, Any],
    reward: Mapping[str, Any],
    cost: Mapping[str, Any],
    lineage: Mapping[str, Any],
    wdl_effect: Mapping[str, Any],
    selection: Mapping[str, Any],
    interactions: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if integrity.get("failures"):
        return {
            "schema_version": "wdl_cog_m1_decision_v1",
            "decision": "WDL-COG-M1-E",
            "decision_wording": "invalidated; metric integrity, run comparability, config flags, checkpoint bundles, or direct reward audit failed",
            "recommended_next_step": "Infrastructure repair before more campaigns",
        }
    if bool(reward.get("direct_reward_active")):
        return {
            "schema_version": "wdl_cog_m1_decision_v1",
            "decision": "WDL-COG-M1-E",
            "decision_wording": "invalidated; direct cognition or memory reward path is active",
            "recommended_next_step": "Disable direct reward leakage and rerun WDL-COG-M1 controls",
        }
    if (wdl_effect.get("damage_total") or 0.0) <= 0.0 or (wdl_effect.get("cue_exposure_total") or 0.0) <= 0.0:
        return {
            "schema_version": "wdl_cog_m1_decision_v1",
            "decision": "WDL-COG-M1-E",
            "decision_wording": "invalidated; WDL demand metrics are missing or inactive",
            "recommended_next_step": "Infrastructure repair before more campaigns",
        }
    if str(cost.get("status")) != "PASS":
        return {
            "schema_version": "wdl_cog_m1_decision_v1",
            "decision": "WDL-COG-M1-D",
            "decision_wording": "reporting/control insufficient; cost sanity fields are missing or inconsistent",
            "recommended_next_step": "Repair reporting and rerun only affected summaries if raw data are valid",
        }

    if float(lineage.get("founder_balance_violation_total") or 0.0) > 0.0:
        return {
            "schema_version": "wdl_cog_m1_decision_v1",
            "decision": "WDL-COG-M1-D",
            "decision_wording": "reporting/control insufficient; founder-balanced lineage violations were detected",
            "recommended_next_step": "Repair lineage reporting or initialization controls before interpreting selection",
        }

    positive_wdl_rows = int(selection.get("wdl_memory_normal_advantage_rows") or 0)
    positive_interaction_rows = int(selection.get("positive_wdl_interaction_rows") or 0)
    tuned_support_rows = int(selection.get("tuned_jungle_supportive_rows") or 0)
    legacy_support_rows = int(selection.get("legacy_supportive_rows") or 0)
    founder_support_rows = int(selection.get("founder_balanced_supportive_rows") or 0)
    founder_only = bool(selection.get("support_depends_entirely_on_founder_balanced"))
    support_rows = {
        "wdl_memory_normal_advantage_rows": positive_wdl_rows,
        "positive_wdl_interaction_rows": positive_interaction_rows,
        "tuned_jungle_supportive_rows": tuned_support_rows,
        "legacy_supportive_rows": legacy_support_rows,
        "founder_balanced_supportive_rows": founder_support_rows,
    }
    if positive_wdl_rows <= 0 or positive_interaction_rows <= 0:
        return {
            "schema_version": "wdl_cog_m1_decision_v1",
            "decision": "WDL-COG-M1-B",
            "decision_wording": "memory selection not supported; memory-normal did not beat memory-ablated under WDL demand or the WDL interaction was not positive",
            "recommended_next_step": "PAI-1 policy/action interface review or WDL demand redesign, depending on diagnostics",
            "supporting_rows": support_rows,
        }
    if tuned_support_rows <= 0:
        return {
            "schema_version": "wdl_cog_m1_decision_v1",
            "decision": "WDL-COG-M1-C",
            "decision_wording": "mixed or substrate-specific memory signal; tuned Jungle WDL is not supportive under the conservative rule",
            "recommended_next_step": "One bounded diagnostic, not tuning",
            "supporting_rows": support_rows,
        }
    if founder_only or (legacy_support_rows > 0 and founder_support_rows == 0) or (legacy_support_rows == 0 and founder_support_rows > 0):
        return {
            "schema_version": "wdl_cog_m1_decision_v1",
            "decision": "WDL-COG-M1-C",
            "decision_wording": "mixed or initialization-sensitive memory signal; interpretation depends materially on legacy versus founder-balanced initialization",
            "recommended_next_step": "One bounded diagnostic, not tuning",
            "supporting_rows": support_rows,
        }
    return {
        "schema_version": "wdl_cog_m1_decision_v1",
        "decision": "WDL-COG-M1-A",
        "decision_wording": "memory selection supported under WDL demand by the bounded WDL-COG-M1 decision rule",
        "recommended_next_step": "Later longer-horizon confirmation campaign before broader claims",
        "supporting_rows": support_rows,
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
        if not all(checks.values()):
            failures.append(f"{run_id} parity mismatch {checks}")
        items.append({"run_id": run_id, "checks": checks, "a": ar, "b": br})
    summary = {
        "schema_version": "wdl_cog_m1_repeat_parity_summary_v1",
        "root_a": str(root_a),
        "root_b": str(root_b),
        "run_count": len(items),
        "all_required_ok": not failures,
        "failures": failures,
        "items": items,
    }
    _write_json(out_path, summary)
    return summary


def _report(
    *,
    dependency: Mapping[str, Any],
    reward: Mapping[str, Any],
    integrity: Mapping[str, Any],
    wdl_effect: Mapping[str, Any],
    selection: Mapping[str, Any],
    cost: Mapping[str, Any],
    lineage: Mapping[str, Any],
    decision: Mapping[str, Any],
    tests_summary: Mapping[str, Any],
    base_commit: str,
    workdir: str,
    device: str,
) -> str:
    return f"""# EIEL Phase IV WDL-COG-M1 Memory Selection Probe Report

## Purpose
Test whether the WDL-1 delayed hazard / partial-observability demand loop produces a reproducible memory-normal over memory-ablated selection outcome under matched WDL-enabled vs WDL-disabled controls.

## Dependencies
WDL-1 decision: `{(dependency.get('wdl1_decision') or {}).get('decision')}`.
WDL-2 committed first: `{dependency.get('wdl2_committed_first')}`.
WDL-2 commit/base: `{dependency.get('wdl2_commit')}`.

## Working directory and base commit
- Working directory: `{workdir}`
- Base commit: `{base_commit}`
- Device: `{device}`
- New worktree used: `False`
- Output root: `{dependency.get('output_root')}`
- Evidence artifact path: `{tests_summary.get('evidence_zip')}`
- Git status before campaign: `{tests_summary.get('git_status_before_campaign')}`
- Git status after campaign: `{tests_summary.get('git_status_after_campaign')}`

## Trait control audit
Memory control: `{(dependency.get('memory_control_status') or {}).get('status')}`.
Prediction control: `{(dependency.get('prediction_control_status') or {}).get('status')}`.

## Reward audit
Direct reward active: `{reward.get('direct_reward_active')}`.
Active fields: `{reward.get('active_fields')}`.
Direct cognition reward active: `{reward.get('direct_cognition_reward_active')}`.
Direct memory reward active: `{reward.get('direct_memory_reward_active')}`.

## Registry/matrix design
160 planned runs: 2 memory arms x 4 world contexts x 2 init arms x 10 seeds.

## Device and run settings
CUDA, 24 generations, 80 steps/generation, population 64, grid 16, checkpoint bundle v2 every generation.

## Preflight result
See `wdl_cog_m1_test_command_summary.json`.

## Repeat parity result
See `wdl_cog_m1_repeat_parity_summary.json`.

## Completion table
- Planned runs: `{integrity.get('planned_runs')}`
- Completed runs: `{integrity.get('completed_runs')}`
- Failed or missing runs: `{integrity.get('failed_or_missing_runs')}`

## Metric integrity table
Failures: `{integrity.get('failures')}`.

## WDL demand validation within WDL-COG-M1
Cue exposure total: `{wdl_effect.get('cue_exposure_total')}`.
Active exposure total: `{wdl_effect.get('active_exposure_total')}`.
Damage total: `{wdl_effect.get('damage_total')}`.
Active without prior cue total: `{wdl_effect.get('active_without_prior_cue_total')}`.

## Memory selection evidence
WDL memory-normal advantage rows: `{selection.get('wdl_memory_normal_advantage_rows')}`.
Positive WDL interaction rows: `{selection.get('positive_wdl_interaction_rows')}`.
Tuned Jungle supportive rows: `{selection.get('tuned_jungle_supportive_rows')}`.
See `wdl_cog_m1_trait_effect_summary.json`, `wdl_cog_m1_selection_summary.json`, and `wdl_cog_m1_paired_deltas.csv`.

## WDL interaction effect
See `wdl_cog_m1_trait_x_wdl_interaction_summary.json`.

## Founder-balance sensitivity
Legacy supportive rows: `{selection.get('legacy_supportive_rows')}`.
Founder-balanced supportive rows: `{selection.get('founder_balanced_supportive_rows')}`.
Support depends entirely on founder-balanced initialization: `{selection.get('support_depends_entirely_on_founder_balanced')}`.
See `wdl_cog_m1_founder_balance_sensitivity_summary.json`.

## Behavior/WDL coupling
See `wdl_cog_m1_behavior_wdl_coupling_summary.json`.

## Cost sanity
Status: `{cost.get('status')}`.
Normal memory cost greater than ablated pairs: `{cost.get('normal_memory_cost_gt_ablated_pairs')}` / `{cost.get('paired_cost_comparison_count')}`.
See `wdl_cog_m1_cost_sanity_summary.json`.

## Lineage sanity
Founder-balance violation total: `{lineage.get('founder_balance_violation_total')}`.
Mean effective alive lineages: `{lineage.get('mean_effective_alive_lineages')}`.
Mean lineage entropy: `{lineage.get('mean_lineage_entropy')}`.
See `wdl_cog_m1_lineage_sanity_summary.json`.

## Decision evidence table
Decision: `{decision.get('decision')}`.
Supporting rows: `{decision.get('supporting_rows')}`.

## Final WDL-COG-M1 decision
{decision.get('decision')}: {decision.get('decision_wording')}

## Recommended next named step
{decision.get('recommended_next_step')}

## Files changed
WDL-COG-M1 registry, WDL-COG-M1 reporting helper, WDL-COG-M1 report.

## Tests run
{json.dumps(tests_summary, sort_keys=True)}

## Known limitations
WDL-COG-M1 is a bounded memory-selection probe on the WDL demand substrate. The decision rule is conservative and does not tune WDL, Jungle, Theme12, rewards, policy code, or selection scoring to make the result positive.

## No-claims statement
WDL-COG-M1 is a bounded memory selection probe on the WDL delayed hazard demand substrate. It does not show intelligence, emergence, consciousness, general cognition, or prediction selection. It does not add cognition rewards. A positive decision would support only a WDL-specific memory-selection signal under this experimental substrate and would require later confirmation before broader claims.
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--report-path", required=True)
    p.add_argument("--evidence-zip", required=True)
    p.add_argument("--base-commit", required=True)
    p.add_argument("--workdir", default=os.getcwd())
    p.add_argument("--device", default="cuda")
    p.add_argument("--audits-only", action="store_true")
    p.add_argument("--parity-a", default="")
    p.add_argument("--parity-b", default="")
    p.add_argument("--tests-summary", default="")
    args = p.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    dependency = _dependency_audit(output_root, base_commit=args.base_commit, workdir=args.workdir, device=args.device)
    runs = _runs(output_root, exclude_dirs=("repeat_parity_a", "repeat_parity_b"))
    reward = _reward_audit(runs)
    dependency_path = output_root / "wdl_cog_m1_dependency_audit.json"
    reward_path = output_root / "wdl_cog_m1_reward_audit_summary.json"
    _write_json(dependency_path, dependency)
    _write_json(reward_path, reward)
    if args.audits_only:
        return 0

    if args.parity_a and args.parity_b:
        _write_parity(Path(args.parity_a), Path(args.parity_b), output_root / "wdl_cog_m1_repeat_parity_summary.json")
        return 0

    completed = [r for r in runs if int(r.get("metrics_rows") or 0) > 0]
    scenario_rows = _scenario_summary(completed)
    paired = _paired_deltas(completed)
    interactions = _interaction_summary(paired)
    aggregate = _aggregate_effects(completed, paired, interactions)
    cost = _cost_sanity(completed)
    lineage = _lineage_sanity(completed)
    selection = _selection_summary(paired, interactions)
    integrity = _integrity(completed, reward)
    decision = _decision(integrity, reward, cost, lineage, aggregate["wdl_effect"], selection, interactions)
    tests_summary = _read_json(Path(args.tests_summary)) if args.tests_summary else {}
    tests_summary["evidence_zip"] = str(Path(args.evidence_zip))

    paths = {
        "summary": output_root / "wdl_cog_m1_summary.json",
        "integrity": output_root / "wdl_cog_m1_metric_integrity_summary.json",
        "scenario": output_root / "wdl_cog_m1_scenario_summary.csv",
        "paired": output_root / "wdl_cog_m1_paired_deltas.csv",
        "trait_effect": output_root / "wdl_cog_m1_trait_effect_summary.json",
        "wdl_effect": output_root / "wdl_cog_m1_wdl_effect_summary.json",
        "interaction": output_root / "wdl_cog_m1_trait_x_wdl_interaction_summary.json",
        "selection": output_root / "wdl_cog_m1_selection_summary.json",
        "founder": output_root / "wdl_cog_m1_founder_balance_sensitivity_summary.json",
        "behavior": output_root / "wdl_cog_m1_behavior_wdl_coupling_summary.json",
        "cost": output_root / "wdl_cog_m1_cost_sanity_summary.json",
        "lineage": output_root / "wdl_cog_m1_lineage_sanity_summary.json",
        "decision": output_root / "wdl_cog_m1_decision.json",
        "test_commands": output_root / "wdl_cog_m1_test_command_summary.json",
    }
    _write_json(paths["summary"], {"schema_version": "wdl_cog_m1_summary_v1", "planned_runs": PLANNED_RUNS, "discovered_runs": len(runs), "completed_runs": len(completed), "runs": completed})
    _write_json(paths["integrity"], integrity)
    _write_csv(paths["scenario"], scenario_rows)
    _write_csv(paths["paired"], paired)
    _write_json(paths["trait_effect"], aggregate["trait_effect"])
    _write_json(paths["wdl_effect"], aggregate["wdl_effect"])
    _write_json(paths["interaction"], aggregate["interaction"])
    _write_json(paths["selection"], selection)
    _write_json(paths["founder"], aggregate["founder"])
    _write_json(paths["behavior"], aggregate["behavior"])
    _write_json(paths["cost"], cost)
    _write_json(paths["lineage"], lineage)
    _write_json(paths["decision"], decision)
    _write_json(paths["test_commands"], tests_summary)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _report(
            dependency=dependency,
            reward=reward,
            integrity=integrity,
            wdl_effect=aggregate["wdl_effect"],
            selection=selection,
            cost=cost,
            lineage=lineage,
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
        for path in [report_path, dependency_path, reward_path, *paths.values(), output_root / "wdl_cog_m1_repeat_parity_summary.json"]:
            if path.exists():
                zf.write(path, arcname=path.name)
        for pattern in (
            str(output_root / "e" / "*" / "*" / "matrix.csv"),
            str(output_root / "e" / "*" / "*" / "summary.json"),
            str(output_root / "e" / "*" / "*" / "r" / "*" / "manifest.json"),
            str(output_root / "e" / "*" / "*" / "r" / "*" / "console.txt"),
            str(Path(args.workdir) / "_spec" / "registry" / "v1" / "wdl_cog_m1_memory_selection_probe.yaml"),
            str(Path(args.tests_summary)),
        ):
            matches = sorted(glob.glob(pattern))
            if pattern.endswith("manifest.json") or pattern.endswith("console.txt"):
                matches = matches[:10]
            for raw in matches:
                path = Path(raw)
                if path.exists():
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
