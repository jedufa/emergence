#!/usr/bin/env python3
"""Phase III CRB-1 perception-readiness bridge reporting helper.

This helper is reporting-only. It reads registry-run outputs, validates the
factor matrix, computes paired normal-minus-ablated deltas, and writes the
CRB-1 decision artifacts and report. It does not alter simulation behavior.
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

CAMPAIGN_NAME = "phaseIII_crb1_perception_readiness_bridge"
REGISTRY_PATH = Path("_spec/registry/v1/phaseIII_crb1_perception_readiness_bridge.yaml")
REPORT_PATH = Path("docs/400_implementation-engineering/410_implementation-reports/000_programme-cross-cutting/EIEL-DOC-000656--eiel-phaseiii-crb1-perception-readiness-bridge-report.md")
EVIDENCE_PATH = Path("evidences/phaseIII_crb1_perception_readiness_bridge_evidence.zip")

ECOLOGIES = (
    "crb1_flat_control_clean",
    "crb1_tuned_full_jungle_recovery",
    "crb1_theme12_clean_pressure",
)
INIT_ARMS = ("init_legacy", "init_founder_balanced")
PERCEPTION_CONDITIONS = ("perception_normal", "perception_ablated")
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
    "cog_capacity_perception_component_mean",
    "perception_capacity_component_mean",
    "perception_cost_charged_mean",
    "sensor_radius_cost_charged_mean",
    "sensor_radius_cost_hypothetical_mean",
    "sensor_channel_active_count_mean",
    "sensor_channel_force_minimal_active_flag",
    "sensor_radius_force_default_active_flag",
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
)

CRITICAL_METRICS = (
    "alive",
    "mean_offspring_total",
    "perception_cost_charged_mean",
    "sensor_channel_force_minimal_active_flag",
    "sensor_radius_force_default_active_flag",
)

THEME4_DISABLED_FIELDS = (
    "ablate_memory",
    "force_memory_disabled",
    "reset_memory_each_tick",
    "enable_experimental_memory_cost_charging",
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
    "archive_reseeding_enable",
    "lineage_population_quota_enable",
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
        raise RuntimeError("PyYAML is required to read the CRB-1 registry")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def expected_run_count(registry: Mapping[str, Any]) -> int:
    axes = registry.get("axes") or {}
    total = len(registry.get("seeds") or [])
    for axis_name in ("ecology_context", "init_arm", "perception_condition"):
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
            PERCEPTION_CONDITIONS.index(str((r.get("axes") or {}).get("perception_condition")))
            if str((r.get("axes") or {}).get("perception_condition")) in PERCEPTION_CONDITIONS
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


def _validate_flags(axes: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    env_cfg = config.get("env_cfg") or {}
    sim_cfg = config.get("sim_cfg") or {}
    ecology = str(axes.get("ecology_context") or "")
    init_arm = str(axes.get("init_arm") or "")
    perception = str(axes.get("perception_condition") or "")

    if ecology == "crb1_flat_control_clean":
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
    elif ecology in ("crb1_tuned_full_jungle_recovery", "crb1_theme12_clean_pressure"):
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

    minimal = _cfg_bool(sim_cfg, "force_minimal_sensor_channels_enabled")
    if perception == "perception_normal":
        if minimal:
            failures.append("perception_normal has force_minimal_sensor_channels_enabled=true")
    elif perception == "perception_ablated":
        if not minimal:
            failures.append("perception_ablated has force_minimal_sensor_channels_enabled=false")
    else:
        failures.append(f"unknown perception_condition={perception}")

    for field in (
        "enable_cognitive_trait_registry",
        "enable_cognitive_status_metrics",
        "enable_cognitive_budget_ledger",
        "enable_evolvable_sensor_channel_mask",
        "enable_experimental_perception_cost_charging",
        "enable_experimental_sensor_radius_cost_charging",
        "force_default_sensor_radius",
    ):
        if not _cfg_bool(sim_cfg, field):
            failures.append(f"required perception/profile flag {field}=false")
    if _cfg_bool(sim_cfg, "enable_evolvable_sensor_radius"):
        failures.append("enable_evolvable_sensor_radius=true")
    for field in THEME4_DISABLED_FIELDS:
        if _cfg_bool(sim_cfg, field):
            failures.append(f"unrelated Theme 4 flag {field}=true")
    for field in FORBIDDEN_INTERVENTION_FIELDS:
        if _cfg_bool(sim_cfg, field):
            failures.append(f"forbidden intervention flag {field}=true")
    return failures


def build_summary(runs_root: Path) -> dict[str, Any]:
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
            "perception_condition": str(axes.get("perception_condition") or ""),
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
        grouped[
            (
                str(row["ecology_context"]),
                str(row["init_arm"]),
                str(row["perception_condition"]),
            )
        ].append(row)

    scenario_summary: list[dict[str, Any]] = []
    for ecology in ECOLOGIES:
        for init in INIT_ARMS:
            for perception in PERCEPTION_CONDITIONS:
                group = grouped.get((ecology, init, perception), [])
                item: dict[str, Any] = {
                    "ecology_context": ecology,
                    "init_arm": init,
                    "perception_condition": perception,
                    "run_count": len(group),
                    "completed": sum(1 for r in group if r.get("status") == "ok"),
                }
                for metric in SUMMARY_METRICS:
                    item[f"mean_{metric}"] = _mean([_num(r.get(metric)) for r in group])
                for metric in ("first_founder_initial_effective_lineages", "first_founder_initial_top1_share"):
                    item[f"mean_{metric}"] = _mean([_num(r.get(metric)) for r in group])
                scenario_summary.append(item)

    paired_deltas: list[dict[str, Any]] = []
    index = {
        (
            str(row["ecology_context"]),
            str(row["init_arm"]),
            str(row["perception_condition"]),
            int(row["seed"]),
        ): row
        for row in rows
    }
    for ecology in ECOLOGIES:
        for init in INIT_ARMS:
            for seed in SEEDS:
                normal = index.get((ecology, init, "perception_normal", seed))
                ablated = index.get((ecology, init, "perception_ablated", seed))
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
            item = {
                "ecology_context": ecology,
                "init_arm": init,
                "pair_count": len(group),
            }
            for metric in SUMMARY_METRICS:
                item[f"mean_delta_{metric}"] = _mean([_num(r.get(f"delta_{metric}")) for r in group])
                item[f"mean_normal_{metric}"] = _mean([_num(r.get(f"normal_{metric}")) for r in group])
                item[f"mean_ablated_{metric}"] = _mean([_num(r.get(f"ablated_{metric}")) for r in group])
            effect_summary.append(item)
            by_effect_key[(ecology, init)] = item

    pressure_specificity: list[dict[str, Any]] = []
    for pressure in ("crb1_tuned_full_jungle_recovery", "crb1_theme12_clean_pressure"):
        for init in INIT_ARMS:
            flat = by_effect_key.get(("crb1_flat_control_clean", init), {})
            press = by_effect_key.get((pressure, init), {})
            item: dict[str, Any] = {"pressure_context": pressure, "init_arm": init}
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

    init_summary: list[dict[str, Any]] = []
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
        init_summary.append(item)

    integrity = {
        "schema_version": "phaseIII_crb1_integrity_v1",
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
        "valid": (
            len(rows) == expected
            and all(r.get("status") == "ok" for r in rows)
            and not missing_metrics
            and not missing_configs
            and not flag_failures
            and not nan_inf
        ),
    }

    return {
        "schema_version": "phaseIII_crb1_summary_v1",
        "campaign": CAMPAIGN_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_root": str(runs_root),
        "integrity": integrity,
        "runs": rows,
        "scenario_summary": scenario_summary,
        "paired_deltas": paired_deltas,
        "perception_effect_summary": effect_summary,
        "pressure_specificity": pressure_specificity,
        "init_arm_summary": init_summary,
    }


def decide(summary: Mapping[str, Any]) -> dict[str, Any]:
    integrity = summary.get("integrity") or {}
    effects = summary.get("perception_effect_summary") or []
    pressure = summary.get("pressure_specificity") or []
    rows = summary.get("runs") or []
    if not integrity.get("valid"):
        return {
            "decision": "CRB-D",
            "decision_wording": "Invalidated; infrastructure repair required before perception-readiness interpretation.",
            "supporting_evidence": ["integrity validation failed"],
            "contradicting_evidence": [],
            "recommended_next_step": "Repair the failing preflight, run, flag, or metric integrity issue before rerunning CRB-1.",
        }

    cost_available = all(_num(row.get("perception_cost_charged_mean")) is not None for row in rows)
    cost_positive = any((_num(row.get("perception_cost_charged_mean")) or 0.0) > 0.0 for row in rows)
    force_default_ok = all((_num(row.get("sensor_radius_force_default_active_flag")) or 0.0) >= 1.0 for row in rows)
    ablation_rows = [r for r in rows if r.get("perception_condition") == "perception_ablated"]
    normal_rows = [r for r in rows if r.get("perception_condition") == "perception_normal"]
    ablation_active = all((_num(r.get("sensor_channel_force_minimal_active_flag")) or 0.0) >= 1.0 for r in ablation_rows)
    normal_not_ablated = all((_num(r.get("sensor_channel_force_minimal_active_flag")) or 0.0) <= 0.0 for r in normal_rows)
    if not (cost_available and cost_positive and force_default_ok and ablation_active and normal_not_ablated):
        return {
            "decision": "CRB-C",
            "decision_wording": "Controls/reporting insufficient; repair perception cost or ablation observability before COG-P1.",
            "supporting_evidence": [
                f"cost_available={cost_available}",
                f"cost_positive={cost_positive}",
                f"force_default_ok={force_default_ok}",
                f"ablation_active={ablation_active}",
                f"normal_not_ablated={normal_not_ablated}",
            ],
            "contradicting_evidence": [],
            "recommended_next_step": "Repair exactly one perception-cost or ablation-reporting issue, then rerun CRB-1.",
        }

    pressure_contexts = {
        (row["pressure_context"], row["init_arm"]): row
        for row in pressure
    }
    stronger_pressure_cells = 0
    positive_pressure_cells = 0
    behavior_coupled_cells = 0
    contradicted_by_balanced = False
    evidence_lines: list[str] = []
    for pressure_context in ("crb1_tuned_full_jungle_recovery", "crb1_theme12_clean_pressure"):
        for init in INIT_ARMS:
            item = pressure_contexts.get((pressure_context, init), {})
            alive = _num(item.get("pressure_minus_flat_delta_alive")) or 0.0
            births = _num(item.get("pressure_minus_flat_delta_mean_offspring_total")) or 0.0
            p_alive = _num(item.get("pressure_mean_delta_alive")) or 0.0
            p_births = _num(item.get("pressure_mean_delta_mean_offspring_total")) or 0.0
            behavior_values = [
                abs(_num(item.get(f"pressure_minus_flat_delta_{metric}")) or 0.0)
                for metric in (
                    "behavior_patch_interaction_proxy",
                    "behavior_risk_exposure_proxy",
                    "behavior_visible_resource_exposure_proxy",
                    "behavior_visible_hazard_exposure_proxy",
                )
            ]
            if alive > 0.0 or births > 0.0:
                stronger_pressure_cells += 1
            if p_alive > 0.0 or p_births > 0.0:
                positive_pressure_cells += 1
            if any(value > 0.0 for value in behavior_values):
                behavior_coupled_cells += 1
            evidence_lines.append(
                f"{pressure_context}/{init}: pressure-minus-flat alive={alive:.3f}, births={births:.3f}; "
                f"pressure normal-minus-ablated alive={p_alive:.3f}, births={p_births:.3f}"
            )

    for pressure_context in ("crb1_tuned_full_jungle_recovery", "crb1_theme12_clean_pressure"):
        legacy = pressure_contexts.get((pressure_context, "init_legacy"), {})
        balanced = pressure_contexts.get((pressure_context, "init_founder_balanced"), {})
        legacy_score = (_num(legacy.get("pressure_mean_delta_alive")) or 0.0) + (
            _num(legacy.get("pressure_mean_delta_mean_offspring_total")) or 0.0
        )
        balanced_score = (_num(balanced.get("pressure_mean_delta_alive")) or 0.0) + (
            _num(balanced.get("pressure_mean_delta_mean_offspring_total")) or 0.0
        )
        if legacy_score > 0.0 and balanced_score < 0.0:
            contradicted_by_balanced = True

    if (
        stronger_pressure_cells >= 2
        and positive_pressure_cells >= 2
        and behavior_coupled_cells >= 2
        and not contradicted_by_balanced
    ):
        decision = "CRB-A"
        wording = "Proceed to COG-P1: Perception Selection Probe."
        next_step = "Run COG-P1 as a trait-specific perception campaign with the same no-claims discipline."
    else:
        decision = "CRB-B"
        wording = "NO-GO; perception signal is not pressure-specific enough for COG-P1."
        next_step = "Do not run COG-P1 now; return to one bounded cost/ecology/control review or choose another trait bridge later."

    return {
        "decision": decision,
        "decision_wording": wording,
        "supporting_evidence": evidence_lines
        + [
            f"stronger_pressure_cells={stronger_pressure_cells}",
            f"positive_pressure_cells={positive_pressure_cells}",
            f"behavior_coupled_cells={behavior_coupled_cells}",
            f"contradicted_by_balanced={contradicted_by_balanced}",
        ],
        "contradicting_evidence": [],
        "recommended_next_step": next_step,
    }


def write_outputs(summary: Mapping[str, Any], output_root: Path) -> dict[str, Path]:
    paths = {
        "integrity": output_root / "crb1_integrity_summary.json",
        "summary": output_root / "crb1_summary.json",
        "scenario_summary": output_root / "crb1_scenario_summary.csv",
        "paired_deltas": output_root / "crb1_paired_deltas.csv",
        "perception_effect": output_root / "crb1_perception_effect_summary.json",
        "init_arm": output_root / "crb1_init_arm_summary.json",
        "decision": output_root / "crb1_decision.json",
    }
    decision = decide(summary)
    _write_json(paths["integrity"], summary["integrity"])
    _write_json(paths["summary"], summary)
    _write_json(paths["perception_effect"], {
        "perception_effect_summary": summary["perception_effect_summary"],
        "pressure_specificity": summary["pressure_specificity"],
    })
    _write_json(paths["init_arm"], {"init_arm_summary": summary["init_arm_summary"]})
    _write_json(paths["decision"], decision)

    scenario_rows = summary["scenario_summary"]
    scenario_fields = sorted({key for row in scenario_rows for key in row})
    _write_csv(paths["scenario_summary"], scenario_rows, scenario_fields)
    delta_rows = summary["paired_deltas"]
    delta_fields = sorted({key for row in delta_rows for key in row})
    _write_csv(paths["paired_deltas"], delta_rows, delta_fields)
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
    repeat_parity_summary: Mapping[str, Any] | None,
    tests_run: Sequence[str],
    files_changed: Sequence[str],
    output_root: Path,
    evidence_path: Path,
) -> str:
    scenario_rows = []
    for row in summary["scenario_summary"]:
        scenario_rows.append(
            [
                row["ecology_context"],
                row["init_arm"],
                row["perception_condition"],
                row["run_count"],
                _fmt(row.get("mean_alive")),
                _fmt(row.get("mean_mean_offspring_total")),
                _fmt(row.get("mean_perception_cost_charged_mean"), 5),
                _fmt(row.get("mean_sensor_channel_active_count_mean")),
            ]
        )
    effect_rows = []
    for row in summary["perception_effect_summary"]:
        effect_rows.append(
            [
                row["ecology_context"],
                row["init_arm"],
                row["pair_count"],
                _fmt(row.get("mean_delta_alive")),
                _fmt(row.get("mean_delta_mean_offspring_total")),
                _fmt(row.get("mean_delta_behavior_patch_interaction_proxy")),
                _fmt(row.get("mean_delta_behavior_risk_exposure_proxy")),
                _fmt(row.get("mean_delta_perception_cost_charged_mean"), 5),
            ]
        )
    specificity_rows = []
    for row in summary["pressure_specificity"]:
        specificity_rows.append(
            [
                row["pressure_context"],
                row["init_arm"],
                _fmt(row.get("pressure_minus_flat_delta_alive")),
                _fmt(row.get("pressure_minus_flat_delta_mean_offspring_total")),
                _fmt(row.get("pressure_minus_flat_delta_behavior_patch_interaction_proxy")),
                _fmt(row.get("pressure_minus_flat_delta_behavior_visible_resource_exposure_proxy")),
            ]
        )
    init_rows = []
    for row in summary["init_arm_summary"]:
        init_rows.append(
            [
                row["ecology_context"],
                _fmt(row.get("legacy_mean_delta_alive")),
                _fmt(row.get("balanced_mean_delta_alive")),
                _fmt(row.get("balanced_minus_legacy_delta_alive")),
                _fmt(row.get("balanced_minus_legacy_delta_founder_initial_effective_lineages")),
            ]
        )

    repeat_text = json.dumps(repeat_parity_summary or {}, indent=2, sort_keys=True)
    no_claims = (
        "CRB-1 is a cognition-readiness bridge, not a cognition-selection proof.\n"
        "It does not show intelligence, emergence, perception adaptiveness, memory selection, or prediction selection.\n"
        "It does not add cognition rewards.\n"
        "It does not change Patch/Jungle mechanics.\n"
        "It does not make B1 founder balance the default.\n"
        "It only decides whether perception is ready for a trait-specific COG-P1 campaign."
    )
    return f"""# EIEL Phase III CRB-1 Perception Readiness Bridge Report

## 1. Purpose

CRB-1 asks whether existing perception-related cognition machinery shows
stronger and cleaner usefulness under calibrated ecological pressure than under
flat/control conditions. This is a bridge decision for a possible later COG-P1
perception campaign, not a cognition-selection campaign.

## 2. Phase I/II Closure Context

Phase I accepted Patch/Jungle V0 as an ecological pressure substrate with a
known early lineage-collapse limitation. Phase II closed the lineage bottleneck
decision with one targeted B1 founder/lineage balance fix. B1 remains disabled
by default and is used here only as an explicit initialization arm.

## 3. B1-LIMITED Handling

CRB-1 includes `init_legacy` and `init_founder_balanced` arms. The
founder-balanced arm uses the B1 validation settings: eight founder groups,
eight initial agents per group, and deterministic round-robin assignment. No
ongoing lineage intervention, lineage protection, diversity reward, or archive
reseeding is enabled.

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

## 5. Registry and Matrix Design

- Registry: `{REGISTRY_PATH}`
- Output root: `{output_root}`
- Matrix: 3 ecology contexts x 2 initialization arms x 2 perception conditions x 5 paired seeds = 60 runs.
- Ecology contexts: `crb1_flat_control_clean`, `crb1_tuned_full_jungle_recovery`, `crb1_theme12_clean_pressure`.
- Initialization arms: `init_legacy`, `init_founder_balanced`.
- Perception conditions: `perception_normal`, `perception_ablated`.

## 6. Device and Run Settings

- Device: `{device}`
- Generations: 8
- Steps per generation: 60
- Population: 64
- Grid size: 16
- Perception cost profile: `medium_from_stage10c`
- Sensor radius handling: `force_default_sensor_radius`
- Ablation: `force_minimal_sensor_channels_enabled` in ablated cells only.

## 7. Preflight Results

```text
{preflight_summary}
```

## 8. Repeat Parity Result

```json
{repeat_text}
```

## 9. Completion Table

- Expected runs: `{summary['integrity']['expected_runs']}`
- Observed runs: `{summary['integrity']['observed_runs']}`
- Completed runs: `{summary['integrity']['completed_runs']}`
- Failed runs: `{summary['integrity']['failed_runs']}`

## 10. Metric Integrity Table

- Integrity valid: `{summary['integrity']['valid']}`
- Missing metrics: `{len(summary['integrity']['missing_metrics'])}`
- Missing configs: `{len(summary['integrity']['missing_configs'])}`
- Flag failures: `{len(summary['integrity']['flag_failures'])}`
- NaN/inf issues: `{len(summary['integrity']['nan_inf_issues'])}`

## 11. Perception Effect by Ecology

Normal-minus-ablated paired deltas:

{_table(['ecology', 'init', 'pairs', 'delta_alive', 'delta_births', 'delta_patch_proxy', 'delta_risk_proxy', 'delta_cost'], effect_rows)}

## 12. Pressure Specificity

Pressure-context normal-minus-ablated deltas minus flat-control deltas:

{_table(['pressure_context', 'init', 'alive', 'births', 'patch_proxy', 'resource_proxy'], specificity_rows)}

## 13. Founder-Balance Sensitivity

{_table(['ecology', 'legacy_delta_alive', 'balanced_delta_alive', 'balanced_minus_legacy_alive', 'balanced_minus_legacy_initial_effective'], init_rows)}

## 14. Behavior Coupling

Behavior proxies were summarized through action, patch-interaction,
risk-exposure, visible-resource, and visible-hazard proxy deltas. These are
diagnostic couplings only. They are not evidence that perception was selected
or adaptive.

## 15. Cost Sanity

Cost charging was required to be observable through
`perception_cost_charged_mean`, `sensor_radius_cost_charged_mean`, and ablation
flags. The decision logic marks CRB-C if cost or ablation observability is not
auditable.

## 16. Lineage Sanity

Lineage and founder metrics are retained in the summary. The founder-balanced
arm is an explicit experimental arm only; B1 remains disabled by default.
Lineage collapse, if present, is documented as a substrate limitation.

## 17. Decision Evidence Table

| Decision | Supporting Evidence | Contradicting Evidence | Confidence |
| --- | --- | --- | --- |
| CRB-A | {', '.join(decision['supporting_evidence']) if decision['decision'] == 'CRB-A' else 'Not selected.'} | {', '.join(decision.get('contradicting_evidence') or []) or 'None recorded.'} | {'medium' if decision['decision'] == 'CRB-A' else 'low'} |
| CRB-B | {', '.join(decision['supporting_evidence']) if decision['decision'] == 'CRB-B' else 'Not selected.'} | {', '.join(decision.get('contradicting_evidence') or []) or 'None recorded.'} | {'medium' if decision['decision'] == 'CRB-B' else 'low'} |
| CRB-C | {', '.join(decision['supporting_evidence']) if decision['decision'] == 'CRB-C' else 'Not selected.'} | {', '.join(decision.get('contradicting_evidence') or []) or 'None recorded.'} | {'medium' if decision['decision'] == 'CRB-C' else 'low'} |
| CRB-D | {', '.join(decision['supporting_evidence']) if decision['decision'] == 'CRB-D' else 'Not selected.'} | {', '.join(decision.get('contradicting_evidence') or []) or 'None recorded.'} | {'high' if decision['decision'] == 'CRB-D' else 'low'} |

## 18. Final CRB-1 Decision

Final decision: `{decision['decision']}`

Decision wording: {decision['decision_wording']}

## 19. Consequence of Decision

{decision['recommended_next_step']}

## 20. Files Changed

{chr(10).join(f'- `{path}`' for path in files_changed)}

## 21. Tests Run

{chr(10).join(f'- `{cmd}`' for cmd in tests_run)}

## 22. Known Limitations

- CRB-1 is short-horizon and bridge-scale.
- B1 founder balance is an explicit arm and remains disabled by default.
- Behavior proxies are diagnostic, not proof of adaptive perception.
- The campaign does not test memory or prediction selection.

## 23. No-Claims Statement

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
    parser = argparse.ArgumentParser(description="Generate Phase III CRB-1 perception-readiness reports")
    parser.add_argument("--runs_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--working_directory", default=str(Path.cwd()))
    parser.add_argument("--base_commit", default="")
    parser.add_argument("--git_status_before", default="")
    parser.add_argument("--git_status_after", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight_summary", default="")
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
    summary = build_summary(runs_root)
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
            output_root / "crb1_dry_run_result.txt",
            output_root / "crb1_repeat_parity_summary.json",
            paths["integrity"],
            paths["summary"],
            paths["scenario_summary"],
            paths["paired_deltas"],
            paths["perception_effect"],
            paths["init_arm"],
            paths["decision"],
            output_root / "crb1_test_command_summary.txt",
        ]
        write_evidence_zip(EVIDENCE_PATH, evidence_files)
    print(json.dumps({"decision": decision, "integrity": summary["integrity"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
