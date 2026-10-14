#!/usr/bin/env python3
"""Phase IV CRB-M1 memory readiness bridge reporting helper.

This helper is reporting-only. It validates the registry-run matrix, audits
direct reward/selection confounds, computes paired normal-minus-ablated deltas,
and writes CRB-M1 decision artifacts. It does not alter simulation behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


csv.field_size_limit(sys.maxsize)

CAMPAIGN_NAME = "phaseIV_crb_m1_memory_readiness_bridge"
REGISTRY_PATH = Path("_spec/registry/v1/phaseIV_crb_m1_memory_readiness_bridge.yaml")
REPORT_PATH = Path("docs/400_implementation-engineering/410_implementation-reports/500_cognition-memory/EIEL-DOC-000662--eiel-phaseiv-crb-m1-memory-readiness-bridge-report.md")
EVIDENCE_PATH = Path("evidences/phaseIV_crb_m1_memory_readiness_bridge_evidence.zip")
COGP1_EVIDENCE_PATH = Path("evidences/phaseIV_cog_p1_perception_selection_probe_evidence.zip")
COGP1_REPORT_PATH = Path("docs/400_implementation-engineering/410_implementation-reports/000_programme-cross-cutting/EIEL-DOC-000661--eiel-phaseiv-cog-p1-perception-selection-probe-report.md")

ECOLOGIES = (
    "crbm1_flat_control_clean",
    "crbm1_tuned_full_jungle_recovery",
    "crbm1_theme12_clean_pressure",
)
PRESSURE_ECOLOGIES = ("crbm1_tuned_full_jungle_recovery", "crbm1_theme12_clean_pressure")
INIT_ARMS = ("init_legacy", "init_founder_balanced")
MEMORY_CONDITIONS = ("memory_normal", "memory_ablated")
SEEDS = (101, 202, 303, 404, 505)

SUMMARY_METRICS = (
    "alive",
    "mean_offspring_total",
    "lineage_effective_alive_lineages",
    "lineage_alive_share_entropy",
    "lineage_active_count",
    "founder_initial_effective_lineages",
    "founder_initial_top1_share",
    "early_lineage_effective_gen_1",
    "early_lineage_effective_gen_2",
    "early_lineage_top1_share_gen_1",
    "early_lineage_top1_share_gen_2",
    "cog_capacity_memory_component_mean",
    "memory_dim_mean",
    "effective_memory_mean",
    "memory_capacity_fraction_mean",
    "memory_cost_charged_mean",
    "memory_cost_hypothetical_mean",
    "memory_state_abs_mean",
    "memory_state_delta_mean",
    "memory_reset_count_total",
    "memory_use_measurement_available_flag",
    "memory_force_disabled_active_flag",
    "memory_reset_each_tick_active_flag",
    "policy_memory_active_fraction",
    "policy_memory_cost_mean",
    "gru_enabled_fraction",
    "behavior_action_total",
    "behavior_patch_interaction_proxy",
    "behavior_risk_exposure_proxy",
    "behavior_visible_resource_exposure_proxy",
    "behavior_visible_hazard_exposure_proxy",
    "jungle_predator_events",
    "jungle_predator_damage_total",
    "jungle_predator_damage_prevented_total",
    "patch_regrowth_suppressed_total",
    "patch_recovery_activity_mean",
    "patch_recovery_added_regrowth_total",
    "direct_cognition_reward_present_flag",
    "direct_internal_selection_pressure_present_flag",
    "direct_prediction_error_selection_pressure_present_flag",
    "direct_development_reward_present_flag",
    "direct_introspection_reward_present_flag",
    "direct_self_regulation_reward_present_flag",
    "direct_self_modification_reward_present_flag",
    "direct_directed_inheritance_reward_present_flag",
)

CRITICAL_METRICS = (
    "alive",
    "mean_offspring_total",
    "memory_cost_charged_mean",
    "memory_force_disabled_active_flag",
    "memory_reset_each_tick_active_flag",
    "effective_memory_mean",
    "direct_cognition_reward_present_flag",
)

THEME4_DISABLED_FIELDS = (
    "enable_introspection_channels",
    "enable_introspection_policy_input",
    "enable_self_regulation",
    "enable_self_regulation_surface",
    "enable_self_regulation_policy_output",
    "enable_self_modification",
    "enable_self_modification_policy_output",
    "enable_directed_inheritance",
    "enable_directed_inheritance_surface",
    "enable_directed_inheritance_policy_output",
)

FORBIDDEN_INTERVENTION_FIELDS = (
    "lineage_protection_enable",
    "lineage_rescue_enable",
    "lineage_diversity_reward_enable",
    "diversity_reward_enable",
    "archive_reseeding_enable",
    "lineage_population_quota_enable",
)

DIRECT_PRESSURE_GUARD_FIELDS = (
    "disable_prediction_error_selection_pressure",
    "episode0_fitness_weight",
    "episode0_pred_err_fitness_weight",
    "episode0_thrashing_fitness_weight",
    "episode0_drive_jitter_fitness_weight",
    "episode0_selfharm_fitness_weight",
    "late_pred_cost_enable",
    "late_coherence_eff_enable",
    "adult_move_penalty_enable",
    "identity_drift_enable",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the CRB-M1 registry")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _knobs_from_flags(flags: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for idx, item in enumerate(flags):
        if item == "--knob" and idx + 1 < len(flags):
            key, _, value = str(flags[idx + 1]).partition("=")
            out[key] = value
    return out


def registry_knobs(registry: Mapping[str, Any]) -> dict[str, str]:
    knobs: dict[str, str] = {}
    for pair in registry.get("cli_defaults") or []:
        if len(pair) == 2 and pair[0] == "--knob":
            key, _, value = str(pair[1]).partition("=")
            knobs[key] = value
    return knobs


def expected_run_count(registry: Mapping[str, Any]) -> int:
    axes = registry.get("axes") or {}
    total = len(registry.get("seeds") or [])
    for axis_name in ("ecology_context", "init_arm", "memory_condition"):
        total *= len((axes.get(axis_name) or {}).get("values") or [])
    return int(total)


def collect_runs(runs_root: Path) -> list[dict[str, Any]]:
    by_run_id: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(runs_root.rglob("summary.json")):
        try:
            summary = _read_json(summary_path)
        except Exception:
            continue
        for row in summary.get("runs") or []:
            run_id = str(row.get("run_id") or "")
            if not run_id.startswith(f"{CAMPAIGN_NAME}__"):
                continue
            run_dir = Path(str(row.get("run_dir") or ""))
            metrics_path = run_dir / "metrics.csv"
            item = dict(row)
            item["summary_path"] = str(summary_path)
            item["run_dir"] = str(run_dir)
            item["_mtime"] = metrics_path.stat().st_mtime if metrics_path.exists() else summary_path.stat().st_mtime
            prev = by_run_id.get(run_id)
            if prev is None or float(item["_mtime"]) >= float(prev.get("_mtime", 0.0)):
                by_run_id[run_id] = item
    runs = list(by_run_id.values())
    runs.sort(
        key=lambda r: (
            ECOLOGIES.index(str((r.get("axes") or {}).get("ecology_context")))
            if str((r.get("axes") or {}).get("ecology_context")) in ECOLOGIES
            else 99,
            INIT_ARMS.index(str((r.get("axes") or {}).get("init_arm")))
            if str((r.get("axes") or {}).get("init_arm")) in INIT_ARMS
            else 99,
            MEMORY_CONDITIONS.index(str((r.get("axes") or {}).get("memory_condition")))
            if str((r.get("axes") or {}).get("memory_condition")) in MEMORY_CONDITIONS
            else 99,
            int(r.get("seed", 0) or 0),
        )
    )
    return runs


def _final_and_first(metrics_path: Path) -> tuple[dict[str, str], dict[str, str], list[dict[str, str]]]:
    rows = _read_csv(metrics_path)
    if not rows:
        return {}, {}, []
    return rows[0], rows[-1], rows


def _cfg_bool(cfg: Mapping[str, Any], field: str) -> bool:
    return bool(cfg.get(field, False))


def _cfg_float(cfg: Mapping[str, Any], field: str, default: float = 0.0) -> float:
    try:
        return float(cfg.get(field, default))
    except Exception:
        return default


def _validate_flags(axes: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    env_cfg = config.get("env_cfg") or {}
    sim_cfg = config.get("sim_cfg") or {}
    ecology = str(axes.get("ecology_context") or "")
    init_arm = str(axes.get("init_arm") or "")
    memory_condition = str(axes.get("memory_condition") or "")

    if ecology == "crbm1_flat_control_clean":
        for field in (
            "patch_world_enable",
            "patch_world_observation_enable",
            "patch_world_waste_consequence_enable",
            "jungle_predator_pressure_enable",
            "jungle_reproduction_emissions_enable",
            "jungle_terrain_protection_enable",
            "jungle_ecological_recovery_enable",
        ):
            if _cfg_bool(env_cfg, field):
                failures.append(f"flat control has {field}=true")
    elif ecology in PRESSURE_ECOLOGIES:
        for field in (
            "patch_world_enable",
            "patch_world_observation_enable",
            "patch_world_waste_consequence_enable",
            "jungle_predator_pressure_enable",
            "jungle_reproduction_emissions_enable",
            "jungle_terrain_protection_enable",
            "jungle_ecological_recovery_enable",
        ):
            if not _cfg_bool(env_cfg, field):
                failures.append(f"pressure context has {field}=false")
    else:
        failures.append(f"unknown ecology_context={ecology}")

    founder_enabled = _cfg_bool(sim_cfg, "lineage_founder_balance_enable")
    if init_arm == "init_legacy":
        if founder_enabled:
            failures.append("legacy init has lineage_founder_balance_enable=true")
    elif init_arm == "init_founder_balanced":
        if not founder_enabled:
            failures.append("founder-balanced init has lineage_founder_balance_enable=false")
        if int(sim_cfg.get("lineage_founder_group_count", 0) or 0) != 8:
            failures.append("founder-balanced init has wrong founder group count")
        if int(sim_cfg.get("lineage_founder_min_agents_per_group", 0) or 0) != 8:
            failures.append("founder-balanced init has wrong founder min agents")
    else:
        failures.append(f"unknown init_arm={init_arm}")

    memory_disabled = _cfg_bool(sim_cfg, "force_memory_disabled")
    memory_reset = _cfg_bool(sim_cfg, "reset_memory_each_tick")
    if memory_condition == "memory_normal":
        if memory_disabled:
            failures.append("memory_normal has force_memory_disabled=true")
        if memory_reset:
            failures.append("memory_normal has reset_memory_each_tick=true")
    elif memory_condition == "memory_ablated":
        if not memory_disabled:
            failures.append("memory_ablated has force_memory_disabled=false")
        if not memory_reset:
            failures.append("memory_ablated has reset_memory_each_tick=false")
    else:
        failures.append(f"unknown memory_condition={memory_condition}")
    if _cfg_bool(sim_cfg, "force_minimal_sensor_channels_enabled"):
        failures.append("perception ablation flag force_minimal_sensor_channels_enabled=true")

    for field in (
        "enable_cognitive_trait_registry",
        "enable_cognitive_status_metrics",
        "enable_cognitive_budget_ledger",
        "enable_evolvable_sensor_channel_mask",
        "enable_experimental_memory_cost_charging",
        "enable_experimental_sensor_radius_cost_charging",
        "force_default_sensor_radius",
    ):
        if not _cfg_bool(sim_cfg, field):
            failures.append(f"required fixed cognition/profile flag {field}=false")
    if _cfg_bool(sim_cfg, "enable_evolvable_sensor_radius"):
        failures.append("enable_evolvable_sensor_radius=true")
    for field in THEME4_DISABLED_FIELDS:
        if _cfg_bool(sim_cfg, field):
            failures.append(f"unrelated Theme 4 flag {field}=true")
    for field in FORBIDDEN_INTERVENTION_FIELDS:
        if _cfg_bool(sim_cfg, field):
            failures.append(f"forbidden intervention flag {field}=true")
    if not _cfg_bool(sim_cfg, "disable_prediction_error_selection_pressure"):
        failures.append("disable_prediction_error_selection_pressure=false")
    for field in (
        "episode0_fitness_weight",
        "episode0_pred_err_fitness_weight",
        "episode0_thrashing_fitness_weight",
        "episode0_drive_jitter_fitness_weight",
        "episode0_selfharm_fitness_weight",
    ):
        if _cfg_float(sim_cfg, field) != 0.0:
            failures.append(f"{field} is nonzero")
    for field in ("late_pred_cost_enable", "late_coherence_eff_enable", "adult_move_penalty_enable", "identity_drift_enable"):
        if _cfg_bool(sim_cfg, field):
            failures.append(f"{field}=true")
    return failures


def _zip_json(zip_path: Path, suffix: str) -> Any | None:
    if not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(suffix):
                with zf.open(name) as f:
                    return json.loads(f.read().decode("utf-8"))
    return None


def build_cogp1_closure_audit(output_root: Path) -> dict[str, Any]:
    decision = _zip_json(COGP1_EVIDENCE_PATH, "cogp1_decision.json")
    reward = _zip_json(COGP1_EVIDENCE_PATH, "cogp1_reward_audit.json")
    summary = _zip_json(COGP1_EVIDENCE_PATH, "cogp1_summary.json")
    pressure = _zip_json(COGP1_EVIDENCE_PATH, "cogp1_pressure_specificity_summary.json")
    behavior = _zip_json(COGP1_EVIDENCE_PATH, "cogp1_behavior_coupling_summary.json")
    effect = _zip_json(COGP1_EVIDENCE_PATH, "cogp1_perception_effect_summary.json")
    integrity = _zip_json(COGP1_EVIDENCE_PATH, "cogp1_integrity_summary.json")
    audit = {
        "schema_version": "crbm1_cogp1_closure_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cogp1_report_path": str(COGP1_REPORT_PATH),
        "cogp1_evidence_path": str(COGP1_EVIDENCE_PATH),
        "cogp1_decision": (decision or {}).get("decision"),
        "cogp1_decision_wording": (decision or {}).get("decision_wording"),
        "cogp1_run_count": (summary or {}).get("integrity", {}).get("completed_runs")
        or (integrity or {}).get("completed_runs"),
        "cogp1_reward_audit_result": {
            "valid": (reward or {}).get("valid"),
            "direct_cognition_reward_active": (reward or {}).get("direct_cognition_reward_active"),
            "direct_perception_reward_active": (reward or {}).get("direct_perception_reward_active"),
            "direct_diversity_reward_active": (reward or {}).get("direct_diversity_reward_active"),
        },
        "cogp1_perception_effect_summary": effect,
        "cogp1_pressure_specificity_result": pressure,
        "cogp1_behavior_coupling_result": behavior,
        "cogp1_conclusion": "perception path stopped/deferred",
        "valid": (
            (decision or {}).get("decision") == "COG-P1-B"
            and bool((reward or {}).get("valid"))
            and not bool((reward or {}).get("direct_cognition_reward_active"))
            and not bool((reward or {}).get("direct_perception_reward_active"))
            and not bool((reward or {}).get("direct_diversity_reward_active"))
        ),
    }
    _write_json(output_root / "crbm1_cogp1_closure_audit.json", audit)
    return audit


def build_reward_audit(output_root: Path) -> dict[str, Any]:
    registry = load_registry()
    knobs = registry_knobs(registry)
    guard_status = {
        "disable_prediction_error_selection_pressure": knobs.get("sim_cfg.disable_prediction_error_selection_pressure") == "true",
        "episode0_fitness_weight_zero": knobs.get("sim_cfg.episode0_fitness_weight") == "0.0",
        "episode0_pred_err_fitness_weight_zero": knobs.get("sim_cfg.episode0_pred_err_fitness_weight") == "0.0",
        "episode0_thrashing_fitness_weight_zero": knobs.get("sim_cfg.episode0_thrashing_fitness_weight") == "0.0",
        "episode0_drive_jitter_fitness_weight_zero": knobs.get("sim_cfg.episode0_drive_jitter_fitness_weight") == "0.0",
        "episode0_selfharm_fitness_weight_zero": knobs.get("sim_cfg.episode0_selfharm_fitness_weight") == "0.0",
        "late_pred_cost_enable_false": knobs.get("sim_cfg.late_pred_cost_enable") == "false",
        "late_coherence_eff_enable_false": knobs.get("sim_cfg.late_coherence_eff_enable") == "false",
        "adult_move_penalty_enable_false": knobs.get("sim_cfg.adult_move_penalty_enable") == "false",
        "identity_drift_enable_false": knobs.get("sim_cfg.identity_drift_enable") == "false",
    }
    selection_sources = [
        Path("engine/simulation/runner_pkg/runner_selection.py"),
        Path("engine/evolution/population_pkg/core.py"),
        Path("engine/evolution/selection.py"),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in selection_sources if path.exists())
    direct_memory_reward_tokens_present = any(
        token in text for token in ("memory_reward", "direct_memory_reward", "memory_selection_reward")
    )
    direct_diversity_reward_tokens_present = any(
        token in text for token in ("diversity_reward", "lineage_diversity_reward")
    )
    direct_cognition_reward_tokens_present = any(
        token in text for token in ("cognition_reward", "cognitive_capacity_reward", "direct_cognition_reward")
    )
    direct_cognition_reward_active = direct_cognition_reward_tokens_present or not all(guard_status.values())
    audit = {
        "schema_version": "crbm1_reward_audit_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry": str(REGISTRY_PATH),
        "direct_cognition_reward_active": bool(direct_cognition_reward_active),
        "direct_memory_reward_active": bool(direct_memory_reward_tokens_present),
        "direct_diversity_reward_active": bool(direct_diversity_reward_tokens_present),
        "direct_cognition_reward_present_flag_interpretation": (
            "`direct_cognition_reward_present_flag` aliases internal prediction/coherence selection pressure telemetry; "
            "CRB-M1 disables that pressure through explicit registry guards."
        ),
        "registry_guard_status": guard_status,
        "selection_sources_audited": [str(path) for path in selection_sources],
        "valid": (
            not direct_cognition_reward_active
            and not direct_memory_reward_tokens_present
            and not direct_diversity_reward_tokens_present
            and all(guard_status.values())
        ),
    }
    _write_json(output_root / "crbm1_reward_audit.json", audit)
    return audit


def build_summary(
    runs_root: Path,
    reward_audit: Mapping[str, Any] | None = None,
    cogp1_closure_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    registry = load_registry()
    expected = expected_run_count(registry)
    rows: list[dict[str, Any]] = []
    nan_inf: list[dict[str, Any]] = []
    flag_failures: list[dict[str, Any]] = []
    missing_metrics: list[str] = []
    missing_configs: list[str] = []

    for run in collect_runs(runs_root):
        axes = run.get("axes") or {}
        run_dir = Path(str(run.get("run_dir") or ""))
        metrics_path = run_dir / "metrics.csv"
        config_path = run_dir / "config.json"
        first, final, metrics_rows = _final_and_first(metrics_path)
        config = _read_json(config_path) if config_path.exists() else {}
        if not metrics_rows:
            missing_metrics.append(str(run.get("run_id")))
        if not config:
            missing_configs.append(str(run.get("run_id")))

        failures = _validate_flags(axes, config) if config else ["config missing"]
        if failures:
            flag_failures.append({"run_id": run.get("run_id"), "failures": failures})

        metric_values: dict[str, float | None] = {}
        for metric in SUMMARY_METRICS:
            value = _num(final.get(metric))
            metric_values[metric] = value
            if final.get(metric) not in (None, "") and value is None:
                nan_inf.append({"run_id": run.get("run_id"), "metric": metric, "value": final.get(metric)})

        first_values = {
            f"first_{metric}": _num(first.get(metric))
            for metric in (
                "founder_initial_effective_lineages",
                "founder_initial_top1_share",
                "founder_initial_lineage_count",
            )
        }
        row = {
            "run_id": run.get("run_id"),
            "run_dir": str(run_dir),
            "seed": int(run.get("seed", 0) or 0),
            "ecology_context": str(axes.get("ecology_context") or ""),
            "init_arm": str(axes.get("init_arm") or ""),
            "memory_condition": str(axes.get("memory_condition") or ""),
            "status": run.get("status"),
            "metrics_exists": metrics_path.exists(),
            "metrics_non_empty": metrics_path.exists() and metrics_path.stat().st_size > 0 and bool(metrics_rows),
            "config_exists": config_path.exists(),
            "flag_failures": failures,
            **first_values,
            **metric_values,
        }
        rows.append(row)

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["ecology_context"]), str(row["init_arm"]), str(row["memory_condition"]))].append(row)

    scenario_summary: list[dict[str, Any]] = []
    for ecology in ECOLOGIES:
        for init in INIT_ARMS:
            for perception in MEMORY_CONDITIONS:
                group = grouped.get((ecology, init, perception), [])
                item: dict[str, Any] = {
                    "ecology_context": ecology,
                    "init_arm": init,
                    "memory_condition": perception,
                    "run_count": len(group),
                    "completed": sum(1 for r in group if r.get("status") == "ok"),
                }
                for metric in SUMMARY_METRICS:
                    item[f"mean_{metric}"] = _mean([_num(r.get(metric)) for r in group])
                scenario_summary.append(item)

    index = {
        (str(row["ecology_context"]), str(row["init_arm"]), str(row["memory_condition"]), int(row["seed"])): row
        for row in rows
    }
    paired_deltas: list[dict[str, Any]] = []
    for ecology in ECOLOGIES:
        for init in INIT_ARMS:
            for seed in SEEDS:
                normal = index.get((ecology, init, "memory_normal", seed))
                ablated = index.get((ecology, init, "memory_ablated", seed))
                item: dict[str, Any] = {
                    "ecology_context": ecology,
                    "init_arm": init,
                    "seed": seed,
                    "normal_run_id": normal.get("run_id") if normal else "",
                    "ablated_run_id": ablated.get("run_id") if ablated else "",
                    "status": "success" if normal and ablated else "missing_pair",
                }
                for metric in SUMMARY_METRICS:
                    n = _num(normal.get(metric)) if normal else None
                    a = _num(ablated.get(metric)) if ablated else None
                    item[f"normal_{metric}"] = n
                    item[f"ablated_{metric}"] = a
                    item[f"delta_{metric}"] = (n - a) if n is not None and a is not None else None
                paired_deltas.append(item)

    effect_summary: list[dict[str, Any]] = []
    by_effect_key: dict[tuple[str, str], dict[str, Any]] = {}
    for ecology in ECOLOGIES:
        for init in INIT_ARMS:
            group = [r for r in paired_deltas if r["ecology_context"] == ecology and r["init_arm"] == init and r["status"] == "success"]
            item: dict[str, Any] = {
                "ecology_context": ecology,
                "init_arm": init,
                "pair_count": len(group),
                "delta_alive_positive_fraction": _mean([1.0 if (_num(r.get("delta_alive")) or 0.0) > 0.0 else 0.0 for r in group]),
                "delta_births_positive_fraction": _mean([1.0 if (_num(r.get("delta_mean_offspring_total")) or 0.0) > 0.0 else 0.0 for r in group]),
            }
            for metric in SUMMARY_METRICS:
                item[f"mean_delta_{metric}"] = _mean([_num(r.get(f"delta_{metric}")) for r in group])
                item[f"mean_normal_{metric}"] = _mean([_num(r.get(f"normal_{metric}")) for r in group])
                item[f"mean_ablated_{metric}"] = _mean([_num(r.get(f"ablated_{metric}")) for r in group])
            effect_summary.append(item)
            by_effect_key[(ecology, init)] = item

    pressure_specificity: list[dict[str, Any]] = []
    for pressure in PRESSURE_ECOLOGIES:
        for init in INIT_ARMS:
            flat = by_effect_key.get(("crbm1_flat_control_clean", init), {})
            press = by_effect_key.get((pressure, init), {})
            pair_rows = [
                r for r in paired_deltas
                if r["ecology_context"] in (pressure, "crbm1_flat_control_clean")
                and r["init_arm"] == init
                and r["status"] == "success"
            ]
            by_seed = defaultdict(dict)
            for row in pair_rows:
                by_seed[int(row["seed"])][str(row["ecology_context"])] = row
            seed_alive_wins = []
            seed_birth_wins = []
            for seed, seed_rows in by_seed.items():
                p = seed_rows.get(pressure)
                f = seed_rows.get("crbm1_flat_control_clean")
                if p and f:
                    seed_alive_wins.append(1.0 if (_num(p.get("delta_alive")) or 0.0) > (_num(f.get("delta_alive")) or 0.0) else 0.0)
                    seed_birth_wins.append(
                        1.0
                        if (_num(p.get("delta_mean_offspring_total")) or 0.0)
                        > (_num(f.get("delta_mean_offspring_total")) or 0.0)
                        else 0.0
                    )
            item: dict[str, Any] = {
                "pressure_context": pressure,
                "init_arm": init,
                "seed_alive_pressure_exceeds_flat_fraction": _mean(seed_alive_wins),
                "seed_births_pressure_exceeds_flat_fraction": _mean(seed_birth_wins),
            }
            for metric in (
                "alive",
                "mean_offspring_total",
                "behavior_patch_interaction_proxy",
                "behavior_risk_exposure_proxy",
                "behavior_visible_resource_exposure_proxy",
                "behavior_visible_hazard_exposure_proxy",
            ):
                pval = _num(press.get(f"mean_delta_{metric}")) or 0.0
                fval = _num(flat.get(f"mean_delta_{metric}")) or 0.0
                item[f"pressure_minus_flat_delta_{metric}"] = pval - fval
                item[f"pressure_mean_delta_{metric}"] = pval
                item[f"flat_mean_delta_{metric}"] = fval
            pressure_specificity.append(item)

    founder_summary: list[dict[str, Any]] = []
    for ecology in ECOLOGIES:
        legacy = by_effect_key.get((ecology, "init_legacy"), {})
        balanced = by_effect_key.get((ecology, "init_founder_balanced"), {})
        item: dict[str, Any] = {"ecology_context": ecology}
        for metric in (
            "alive",
            "mean_offspring_total",
            "lineage_effective_alive_lineages",
            "founder_initial_effective_lineages",
            "behavior_patch_interaction_proxy",
        ):
            lval = _num(legacy.get(f"mean_delta_{metric}")) or 0.0
            bval = _num(balanced.get(f"mean_delta_{metric}")) or 0.0
            item[f"legacy_mean_delta_{metric}"] = lval
            item[f"balanced_mean_delta_{metric}"] = bval
            item[f"balanced_minus_legacy_delta_{metric}"] = bval - lval
        founder_summary.append(item)

    behavior_summary: list[dict[str, Any]] = []
    for row in pressure_specificity:
        behavior_summary.append(
            {
                "pressure_context": row["pressure_context"],
                "init_arm": row["init_arm"],
                "patch_interaction_pressure_minus_flat": row["pressure_minus_flat_delta_behavior_patch_interaction_proxy"],
                "risk_exposure_pressure_minus_flat": row["pressure_minus_flat_delta_behavior_risk_exposure_proxy"],
                "resource_exposure_pressure_minus_flat": row["pressure_minus_flat_delta_behavior_visible_resource_exposure_proxy"],
                "hazard_exposure_pressure_minus_flat": row["pressure_minus_flat_delta_behavior_visible_hazard_exposure_proxy"],
                "any_behavior_coupling": any(
                    abs(float(row.get(f"pressure_minus_flat_delta_{metric}") or 0.0)) > 0.0
                    for metric in (
                        "behavior_patch_interaction_proxy",
                        "behavior_risk_exposure_proxy",
                        "behavior_visible_resource_exposure_proxy",
                        "behavior_visible_hazard_exposure_proxy",
                    )
                ),
            }
        )

    cost_sanity = {
        "memory_cost_available_runs": sum(1 for r in rows if _num(r.get("memory_cost_charged_mean")) is not None),
        "memory_cost_positive_runs": sum(1 for r in rows if (_num(r.get("memory_cost_charged_mean")) or 0.0) > 0.0),
        "ablated_force_memory_disabled_runs": sum(
            1 for r in rows
            if r.get("memory_condition") == "memory_ablated"
            and (_num(r.get("memory_force_disabled_active_flag")) or 0.0) >= 1.0
        ),
        "ablated_reset_each_tick_runs": sum(
            1 for r in rows
            if r.get("memory_condition") == "memory_ablated"
            and (_num(r.get("memory_reset_each_tick_active_flag")) or 0.0) >= 1.0
        ),
        "normal_force_memory_disabled_runs": sum(
            1 for r in rows
            if r.get("memory_condition") == "memory_normal"
            and (_num(r.get("memory_force_disabled_active_flag")) or 0.0) >= 1.0
        ),
        "normal_reset_each_tick_runs": sum(
            1 for r in rows
            if r.get("memory_condition") == "memory_normal"
            and (_num(r.get("memory_reset_each_tick_active_flag")) or 0.0) >= 1.0
        ),
        "ablated_effective_memory_zero_runs": sum(
            1 for r in rows
            if r.get("memory_condition") == "memory_ablated"
            and (_num(r.get("effective_memory_mean")) or 0.0) == 0.0
        ),
    }
    lineage_sanity = {
        "legacy_founder_balance_enabled_runs": sum(
            1 for r in rows if r.get("init_arm") == "init_legacy" and (_num(r.get("first_founder_initial_effective_lineages")) or 0.0) > 0.0
        ),
        "founder_balanced_initial_effective_mean": _mean(
            [_num(r.get("first_founder_initial_effective_lineages")) for r in rows if r.get("init_arm") == "init_founder_balanced"]
        ),
        "final_effective_alive_lineages_mean": _mean([_num(r.get("lineage_effective_alive_lineages")) for r in rows]),
        "lineage_collapse_limitation_documented": True,
    }
    reward_rows = {
        "direct_cognition_reward_present_final_max": max([_num(r.get("direct_cognition_reward_present_flag")) or 0.0 for r in rows], default=0.0),
        "direct_internal_selection_pressure_final_max": max([_num(r.get("direct_internal_selection_pressure_present_flag")) or 0.0 for r in rows], default=0.0),
        "direct_prediction_error_selection_pressure_final_max": max([_num(r.get("direct_prediction_error_selection_pressure_present_flag")) or 0.0 for r in rows], default=0.0),
    }

    integrity = {
        "schema_version": "crbm1_integrity_v1",
        "expected_runs": expected,
        "observed_runs": len(rows),
        "completed_runs": sum(1 for r in rows if r.get("status") == "ok"),
        "failed_runs": sum(1 for r in rows if r.get("status") != "ok"),
        "missing_metrics": missing_metrics,
        "missing_configs": missing_configs,
        "flag_failures": flag_failures,
        "nan_inf_issues": nan_inf,
        "critical_metric_availability": {
            metric: sum(1 for r in rows if _num(r.get(metric)) is not None)
            for metric in CRITICAL_METRICS
        },
        "reward_metric_integrity": reward_rows,
        "valid": (
            len(rows) == expected
            and all(r.get("status") == "ok" for r in rows)
            and not missing_metrics
            and not missing_configs
            and not flag_failures
            and not nan_inf
            and (reward_audit or {}).get("valid", True)
            and reward_rows["direct_cognition_reward_present_final_max"] == 0.0
            and reward_rows["direct_internal_selection_pressure_final_max"] == 0.0
            and reward_rows["direct_prediction_error_selection_pressure_final_max"] == 0.0
        ),
    }

    return {
        "schema_version": "crbm1_summary_v1",
        "campaign": CAMPAIGN_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "cogp1_closure_audit": dict(cogp1_closure_audit or {}),
        "reward_audit": dict(reward_audit or {}),
        "integrity": integrity,
        "runs": rows,
        "scenario_summary": scenario_summary,
        "paired_deltas": paired_deltas,
        "memory_effect_summary": effect_summary,
        "pressure_specificity": pressure_specificity,
        "founder_balance_sensitivity": founder_summary,
        "behavior_coupling": behavior_summary,
        "cost_sanity": cost_sanity,
        "lineage_sanity": lineage_sanity,
    }


def decide(summary: Mapping[str, Any]) -> dict[str, Any]:
    integrity = summary.get("integrity") or {}
    reward = summary.get("reward_audit") or {}
    pressure = summary.get("pressure_specificity") or []
    rows = summary.get("runs") or []
    if not reward.get("valid", False):
        return {
            "decision": "CRB-M1-D",
            "decision_wording": "Invalidated; direct reward or internal selection pressure audit did not pass.",
            "supporting_evidence": ["reward audit invalid"],
            "contradicting_evidence": [],
            "recommended_next_step": "Repair direct reward/selection-pressure configuration before rerunning CRB-M1.",
        }
    if not integrity.get("valid"):
        return {
            "decision": "CRB-M1-D",
            "decision_wording": "Invalidated; infrastructure, flags, metrics, reward metrics, or run integrity failed.",
            "supporting_evidence": ["integrity validation failed"],
            "contradicting_evidence": [],
            "recommended_next_step": "Repair the failing preflight, parity, run, flag, reward, or metric integrity issue.",
        }

    cost = summary.get("cost_sanity") or {}
    expected_ablated = len(ECOLOGIES) * len(INIT_ARMS) * len(SEEDS)
    cost_available = int(cost.get("memory_cost_available_runs", 0)) == len(rows)
    cost_positive = int(cost.get("memory_cost_positive_runs", 0)) > 0
    ablation_active = (
        int(cost.get("ablated_force_memory_disabled_runs", 0)) == expected_ablated
        and int(cost.get("ablated_reset_each_tick_runs", 0)) == expected_ablated
        and int(cost.get("ablated_effective_memory_zero_runs", 0)) == expected_ablated
    )
    normal_not_ablated = (
        int(cost.get("normal_force_memory_disabled_runs", 0)) == 0
        and int(cost.get("normal_reset_each_tick_runs", 0)) == 0
    )
    if not (cost_available and ablation_active and normal_not_ablated):
        return {
            "decision": "CRB-M1-C",
            "decision_wording": "Controls/reporting insufficient; repair exactly one cost or ablation observability issue.",
            "supporting_evidence": [
                f"cost_available={cost_available}",
                f"cost_positive={cost_positive}",
                f"ablation_active={ablation_active}",
                f"normal_not_ablated={normal_not_ablated}",
            ],
            "contradicting_evidence": [],
            "recommended_next_step": "Repair memory cost/ablation reporting before making a selection-probe decision.",
        }

    stronger_pressure_cells = 0
    positive_pressure_cells = 0
    consistent_pressure_cells = 0
    behavior_coupled_cells = 0
    evidence_lines: list[str] = []
    for item in pressure:
        alive_specific = _num(item.get("pressure_minus_flat_delta_alive")) or 0.0
        births_specific = _num(item.get("pressure_minus_flat_delta_mean_offspring_total")) or 0.0
        pressure_alive = _num(item.get("pressure_mean_delta_alive")) or 0.0
        pressure_births = _num(item.get("pressure_mean_delta_mean_offspring_total")) or 0.0
        alive_seed_rate = _num(item.get("seed_alive_pressure_exceeds_flat_fraction")) or 0.0
        birth_seed_rate = _num(item.get("seed_births_pressure_exceeds_flat_fraction")) or 0.0
        behavior_values = [
            abs(_num(item.get(f"pressure_minus_flat_delta_{metric}")) or 0.0)
            for metric in (
                "behavior_patch_interaction_proxy",
                "behavior_risk_exposure_proxy",
                "behavior_visible_resource_exposure_proxy",
                "behavior_visible_hazard_exposure_proxy",
            )
        ]
        if alive_specific > 0.0 or births_specific > 0.0:
            stronger_pressure_cells += 1
        if pressure_alive > 0.0 or pressure_births > 0.0:
            positive_pressure_cells += 1
        if alive_seed_rate >= 0.55 or birth_seed_rate >= 0.55:
            consistent_pressure_cells += 1
        if any(value > 0.0 for value in behavior_values):
            behavior_coupled_cells += 1
        evidence_lines.append(
            f"{item['pressure_context']}/{item['init_arm']}: pressure-minus-flat alive={alive_specific:.3f}, "
            f"births={births_specific:.3f}, seed_alive_win_rate={alive_seed_rate:.2f}, "
            f"pressure normal-minus-ablated alive={pressure_alive:.3f}, births={pressure_births:.3f}"
        )

    contradicted_by_balanced = False
    by_key = {(r["pressure_context"], r["init_arm"]): r for r in pressure}
    for pressure_context in PRESSURE_ECOLOGIES:
        legacy = by_key.get((pressure_context, "init_legacy"), {})
        balanced = by_key.get((pressure_context, "init_founder_balanced"), {})
        legacy_score = (_num(legacy.get("pressure_mean_delta_alive")) or 0.0) + (
            _num(legacy.get("pressure_mean_delta_mean_offspring_total")) or 0.0
        )
        balanced_score = (_num(balanced.get("pressure_mean_delta_alive")) or 0.0) + (
            _num(balanced.get("pressure_mean_delta_mean_offspring_total")) or 0.0
        )
        if legacy_score > 0.0 and balanced_score < 0.0:
            contradicted_by_balanced = True

    if (
        stronger_pressure_cells >= 3
        and positive_pressure_cells >= 2
        and consistent_pressure_cells >= 2
        and behavior_coupled_cells >= 2
        and not contradicted_by_balanced
    ):
        decision = "CRB-M1-A"
        wording = (
            "CRB-M1 supports a pressure-specific memory readiness signal. "
            "Proceed to COG-M1: Memory Selection Probe."
        )
        next_step = "Proceed to COG-M1: Memory Selection Probe."
    else:
        decision = "CRB-M1-B"
        wording = "Memory signal not supported strongly enough at CRB-M1 scale."
        next_step = "Do not run COG-M1; proceed to CRB-X1 Prediction Readiness Bridge or stop cognition trait testing pending theory review."

    return {
        "decision": decision,
        "decision_wording": wording,
        "supporting_evidence": evidence_lines
        + [
            f"stronger_pressure_cells={stronger_pressure_cells}",
            f"positive_pressure_cells={positive_pressure_cells}",
            f"consistent_pressure_cells={consistent_pressure_cells}",
            f"behavior_coupled_cells={behavior_coupled_cells}",
            f"contradicted_by_balanced={contradicted_by_balanced}",
            "reward audit confirms no direct memory/cognition/diversity reward active",
        ],
        "contradicting_evidence": [],
        "recommended_next_step": next_step,
    }


def write_outputs(summary: Mapping[str, Any], output_root: Path) -> dict[str, Path]:
    paths = {
        "cogp1_closure_audit": output_root / "crbm1_cogp1_closure_audit.json",
        "reward_audit": output_root / "crbm1_reward_audit.json",
        "integrity": output_root / "crbm1_integrity_summary.json",
        "summary": output_root / "crbm1_summary.json",
        "scenario_summary": output_root / "crbm1_scenario_summary.csv",
        "paired_deltas": output_root / "crbm1_paired_deltas.csv",
        "memory_effect": output_root / "crbm1_memory_effect_summary.json",
        "pressure_specificity": output_root / "crbm1_pressure_specificity_summary.json",
        "founder_balance": output_root / "crbm1_founder_balance_sensitivity_summary.json",
        "behavior_coupling": output_root / "crbm1_behavior_coupling_summary.json",
        "cost_sanity": output_root / "crbm1_cost_sanity_summary.json",
        "lineage_sanity": output_root / "crbm1_lineage_sanity_summary.json",
        "decision": output_root / "crbm1_decision.json",
    }
    decision = decide(summary)
    _write_json(paths["cogp1_closure_audit"], summary["cogp1_closure_audit"])
    _write_json(paths["reward_audit"], summary["reward_audit"])
    _write_json(paths["integrity"], summary["integrity"])
    _write_json(paths["summary"], summary)
    _write_json(paths["memory_effect"], {"memory_effect_summary": summary["memory_effect_summary"]})
    _write_json(paths["pressure_specificity"], {"pressure_specificity": summary["pressure_specificity"]})
    _write_json(paths["founder_balance"], {"founder_balance_sensitivity": summary["founder_balance_sensitivity"]})
    _write_json(paths["behavior_coupling"], {"behavior_coupling": summary["behavior_coupling"]})
    _write_json(paths["cost_sanity"], summary["cost_sanity"])
    _write_json(paths["lineage_sanity"], summary["lineage_sanity"])
    _write_json(paths["decision"], decision)
    scenario_rows = summary["scenario_summary"]
    _write_csv(paths["scenario_summary"], scenario_rows, sorted({key for row in scenario_rows for key in row}))
    delta_rows = summary["paired_deltas"]
    _write_csv(paths["paired_deltas"], delta_rows, sorted({key for row in delta_rows for key in row}))
    return paths


def _fmt(value: Any, digits: int = 3) -> str:
    num = _num(value)
    return f"{num:.{digits}f}" if num is not None else "NA"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def render_report(
    summary: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    working_directory: str,
    base_commit: str,
    git_status_before: str,
    git_status_after: str,
    device: str,
    preflight_summary: str,
    dry_run_summary: str,
    repeat_parity_summary: Mapping[str, Any] | None,
    tests_run: Sequence[str],
    files_changed: Sequence[str],
    output_root: Path,
    evidence_path: Path,
) -> str:
    scenario_rows = [
        [
            r["ecology_context"],
            r["init_arm"],
            r["memory_condition"],
            r["run_count"],
            _fmt(r.get("mean_alive")),
            _fmt(r.get("mean_mean_offspring_total")),
            _fmt(r.get("mean_memory_cost_charged_mean"), 5),
            _fmt(r.get("mean_direct_cognition_reward_present_flag")),
        ]
        for r in summary["scenario_summary"]
    ]
    effect_rows = [
        [
            r["ecology_context"],
            r["init_arm"],
            r["pair_count"],
            _fmt(r.get("mean_delta_alive")),
            _fmt(r.get("mean_delta_mean_offspring_total")),
            _fmt(r.get("delta_alive_positive_fraction")),
            _fmt(r.get("mean_delta_behavior_patch_interaction_proxy")),
            _fmt(r.get("mean_delta_behavior_risk_exposure_proxy")),
        ]
        for r in summary["memory_effect_summary"]
    ]
    specificity_rows = [
        [
            r["pressure_context"],
            r["init_arm"],
            _fmt(r.get("pressure_minus_flat_delta_alive")),
            _fmt(r.get("pressure_minus_flat_delta_mean_offspring_total")),
            _fmt(r.get("seed_alive_pressure_exceeds_flat_fraction")),
            _fmt(r.get("pressure_minus_flat_delta_behavior_patch_interaction_proxy")),
            _fmt(r.get("pressure_minus_flat_delta_behavior_visible_resource_exposure_proxy")),
        ]
        for r in summary["pressure_specificity"]
    ]
    founder_rows = [
        [
            r["ecology_context"],
            _fmt(r.get("legacy_mean_delta_alive")),
            _fmt(r.get("balanced_mean_delta_alive")),
            _fmt(r.get("balanced_minus_legacy_delta_alive")),
            _fmt(r.get("balanced_minus_legacy_delta_founder_initial_effective_lineages")),
        ]
        for r in summary["founder_balance_sensitivity"]
    ]
    repeat_text = json.dumps(repeat_parity_summary or {}, indent=2, sort_keys=True)
    cogp1_text = json.dumps(summary.get("cogp1_closure_audit") or {}, indent=2, sort_keys=True)
    reward_text = json.dumps(summary.get("reward_audit") or {}, indent=2, sort_keys=True)
    no_claims = (
        "CRB-M1 is a memory-readiness bridge, not a memory-selection proof.\n"
        "It does not show intelligence, emergence, memory adaptiveness, perception adaptiveness, prediction selection, or general cognition.\n"
        "It does not add cognition rewards.\n"
        "It does not change Patch/Jungle mechanics.\n"
        "It does not make B1 founder balance the default.\n"
        "It only decides whether memory is ready for a trait-specific COG-M1 campaign."
    )
    decision_rows = []
    for label in ("CRB-M1-A", "CRB-M1-B", "CRB-M1-C", "CRB-M1-D"):
        selected = decision["decision"] == label
        decision_rows.append(
            [
                label,
                "; ".join(decision.get("supporting_evidence") or []) if selected else "Not selected.",
                "; ".join(decision.get("contradicting_evidence") or []) if selected else "None recorded.",
                "medium" if selected else "low",
            ]
        )
    return f"""# EIEL Phase IV CRB-M1 Memory Readiness Bridge Report

## 1. Purpose

CRB-M1 asks whether memory-related machinery shows stronger, cleaner, and
more interpretable usefulness under calibrated ecological pressure than under
flat/control conditions. It is not an intelligence, emergence, memory
selection, or prediction-selection proof.

## 2. Perception Track Closure Context

COG-P1 ended with COG-P1-B, so Perception Track V0 is stopped/deferred.
CRB-M1 is the bounded alternate trait bridge for memory and does not rerun or
retune perception.

## 3. COG-P1 Dependency Audit

```json
{cogp1_text}
```

## 4. Working Directory and Base Commit

- Working directory: `{working_directory}`
- Base commit: `{base_commit}`
- Git status before work:

```text
{git_status_before or "(clean)"}
```

- Git status after work:

```text
{git_status_after or "(clean)"}
```

## 5. Reward Audit

```json
{reward_text}
```

The `direct_cognition_reward_present_flag` name is ambiguous. In this codebase
it aliases internal prediction/coherence selection-pressure telemetry. CRB-M1
explicitly disables that pressure and requires final reward/internal pressure
metrics to remain zero.

## 6. Memory Ablation/Control Availability

CRB-M1 uses existing Theme 4 memory controls:
`force_memory_disabled=true` and `reset_memory_each_tick=true` in the
`memory_ablated` arm. The `memory_normal` arm explicitly keeps both controls
false. No memory architecture, observation channel, or reward path is added.

## 7. Registry and Matrix Design

- Registry: `{REGISTRY_PATH}`
- Output root: `{output_root}`
- Matrix: 3 ecology contexts x 2 initialization arms x 2 memory conditions x 5 paired seeds = 60 runs.
- Ecology contexts: `crbm1_flat_control_clean`, `crbm1_tuned_full_jungle_recovery`, `crbm1_theme12_clean_pressure`.
- Initialization arms: `init_legacy`, `init_founder_balanced`.
- Memory conditions: `memory_normal`, `memory_ablated`.

## 8. Device and Run Settings

- Device: `{device}`
- Generations: 8
- Steps per generation: 80
- Population: 64
- Grid size: 16
- Fixed perception/sensor baseline is inherited from the current calibrated cognition profile and is not varied.
- Memory cost charging: enabled for observability.
- Ablation: `force_memory_disabled=true` and `reset_memory_each_tick=true` in ablated cells only.
- Direct/internal prediction selection pressure: disabled.

## 9. Preflight Results

```text
{preflight_summary}
```

Dry-run:

```text
{dry_run_summary}
```

## 10. Repeat Parity Result

```json
{repeat_text}
```

## 11. Completion Table

- Expected runs: `{summary['integrity']['expected_runs']}`
- Observed runs: `{summary['integrity']['observed_runs']}`
- Completed runs: `{summary['integrity']['completed_runs']}`
- Failed runs: `{summary['integrity']['failed_runs']}`

## 12. Metric Integrity Table

- Integrity valid: `{summary['integrity']['valid']}`
- Missing metrics: `{len(summary['integrity']['missing_metrics'])}`
- Missing configs: `{len(summary['integrity']['missing_configs'])}`
- Flag failures: `{len(summary['integrity']['flag_failures'])}`
- NaN/inf issues: `{len(summary['integrity']['nan_inf_issues'])}`

{_table(['ecology', 'init', 'memory', 'runs', 'alive', 'births', 'memory_cost', 'direct_reward_flag'], scenario_rows)}

## 13. Memory Effect by Ecology/Init Arm

Normal-minus-ablated paired deltas:

{_table(['ecology', 'init', 'pairs', 'delta_alive', 'delta_births', 'alive_positive_fraction', 'delta_patch_proxy', 'delta_risk_proxy'], effect_rows)}

## 14. Pressure Specificity

Pressure-context memory deltas minus flat-control memory deltas:

{_table(['pressure_context', 'init', 'alive', 'births', 'seed_alive_win_rate', 'patch_proxy', 'resource_proxy'], specificity_rows)}

## 15. Founder-Balance Sensitivity

{_table(['ecology', 'legacy_delta_alive', 'balanced_delta_alive', 'balanced_minus_legacy_alive', 'balanced_minus_legacy_initial_effective'], founder_rows)}

## 16. Behavior Coupling

Behavior coupling summary:

```json
{json.dumps(summary.get('behavior_coupling') or [], indent=2, sort_keys=True)}
```

## 17. Cost Sanity

```json
{json.dumps(summary.get('cost_sanity') or {}, indent=2, sort_keys=True)}
```

## 18. Lineage Sanity

```json
{json.dumps(summary.get('lineage_sanity') or {}, indent=2, sort_keys=True)}
```

## 19. Decision Evidence Table

{_table(['Decision', 'Supporting Evidence', 'Contradicting Evidence', 'Confidence'], decision_rows)}

## 20. Final CRB-M1 Decision

Final decision: `{decision['decision']}`

Decision wording: {decision['decision_wording']}

## 21. Consequence of Decision

{decision['recommended_next_step']}

## 22. Files Changed

{chr(10).join(f'- `{path}`' for path in files_changed) or '- None recorded.'}

## 23. Tests Run

{chr(10).join(f'- `{cmd}`' for cmd in tests_run) or '- None recorded.'}

## 24. Known Limitations

- CRB-M1 is still a controlled probe, not proof of adaptiveness.
- B1 founder balance remains disabled by default and is only an explicit arm.
- Behavior proxies are diagnostic couplings, not direct evidence of intelligence.
- The campaign does not test memory selection or prediction selection.

## 25. No-Claims Statement

{no_claims}

## Evidence Artifact

`{evidence_path}`
"""


def write_evidence_zip(evidence_path: Path, files: Sequence[Path]) -> None:
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(evidence_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            if path.exists():
                zf.write(path, path.name)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase IV CRB-M1 memory readiness reports")
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--working_directory", default=str(Path.cwd()))
    parser.add_argument("--base_commit", default="")
    parser.add_argument("--git_status_before", default="")
    parser.add_argument("--git_status_after", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight_summary", default="")
    parser.add_argument("--dry_run_summary", default="")
    parser.add_argument("--repeat_parity_summary", default="")
    parser.add_argument("--tests_run", action="append", default=[])
    parser.add_argument("--files_changed", action="append", default=[])
    parser.add_argument("--write_report", action="store_true")
    parser.add_argument("--write_evidence", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    runs_root = Path(args.runs_root)
    output_root = Path(args.output_root)
    cogp1_closure_audit = build_cogp1_closure_audit(output_root)
    if not cogp1_closure_audit.get("valid", False):
        summary = {
            "schema_version": "crbm1_summary_v1",
            "campaign": CAMPAIGN_NAME,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "runs_root": str(runs_root),
            "cogp1_closure_audit": cogp1_closure_audit,
            "reward_audit": {},
            "integrity": {
                "schema_version": "crbm1_integrity_v1",
                "expected_runs": expected_run_count(load_registry()),
                "observed_runs": 0,
                "completed_runs": 0,
                "failed_runs": 0,
                "missing_metrics": [],
                "missing_configs": [],
                "flag_failures": [{"run_id": "cogp1_dependency", "failures": ["COG-P1 closure audit invalid"]}],
                "nan_inf_issues": [],
                "critical_metric_availability": {},
                "reward_metric_integrity": {},
                "valid": False,
            },
            "runs": [],
            "scenario_summary": [],
            "paired_deltas": [],
            "memory_effect_summary": [],
            "pressure_specificity": [],
            "founder_balance_sensitivity": [],
            "behavior_coupling": [],
            "cost_sanity": {},
            "lineage_sanity": {},
        }
        _write_json(output_root / "crbm1_summary.json", summary)
        _write_json(output_root / "crbm1_decision.json", {
            "decision": "CRB-M1-D",
            "decision_wording": "Invalidated; COG-P1 closure audit did not pass.",
            "supporting_evidence": ["COG-P1 dependency audit invalid"],
            "contradicting_evidence": [],
            "recommended_next_step": "Repair or provide valid COG-P1 closure evidence before CRB-M1.",
        })
        print(json.dumps({"decision": _read_json(output_root / "crbm1_decision.json"), "integrity": summary["integrity"]}, indent=2, sort_keys=True))
        return 1
    reward_audit = build_reward_audit(output_root)
    summary = build_summary(runs_root, reward_audit=reward_audit, cogp1_closure_audit=cogp1_closure_audit)
    paths = write_outputs(summary, output_root)
    decision = _read_json(paths["decision"])
    repeat_summary = {}
    if args.repeat_parity_summary:
        repeat_path = Path(args.repeat_parity_summary)
        if repeat_path.exists():
            repeat_summary = _read_json(repeat_path)
    if args.write_report:
        report = render_report(
            summary,
            decision,
            working_directory=args.working_directory,
            base_commit=args.base_commit,
            git_status_before=args.git_status_before,
            git_status_after=args.git_status_after,
            device=args.device,
            preflight_summary=args.preflight_summary,
            dry_run_summary=args.dry_run_summary,
            repeat_parity_summary=repeat_summary,
            tests_run=args.tests_run,
            files_changed=args.files_changed,
            output_root=output_root,
            evidence_path=EVIDENCE_PATH,
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
    if args.write_evidence:
        evidence_files = [
            REPORT_PATH,
            REGISTRY_PATH,
            output_root / "crbm1_dry_run_result.txt",
            output_root / "crbm1_cogp1_closure_audit.json",
            output_root / "crbm1_reward_audit.json",
            output_root / "crbm1_repeat_parity_summary.json",
            output_root / "crbm1_integrity_summary.json",
            output_root / "crbm1_summary.json",
            output_root / "crbm1_scenario_summary.csv",
            output_root / "crbm1_paired_deltas.csv",
            output_root / "crbm1_memory_effect_summary.json",
            output_root / "crbm1_pressure_specificity_summary.json",
            output_root / "crbm1_founder_balance_sensitivity_summary.json",
            output_root / "crbm1_behavior_coupling_summary.json",
            output_root / "crbm1_cost_sanity_summary.json",
            output_root / "crbm1_lineage_sanity_summary.json",
            output_root / "crbm1_decision.json",
            output_root / "crbm1_test_command_summary.txt",
        ]
        write_evidence_zip(EVIDENCE_PATH, evidence_files)
    print(json.dumps({"decision": decision, "integrity": summary["integrity"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
