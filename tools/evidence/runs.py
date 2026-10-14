"""Run completeness and anomaly classification."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any, Mapping

from tools.evidence.anomalies import summarize_anomalies
from tools.evidence.manifest import scenario_dimension_values
from tools.evidence.profiles import ProfileContract, expected_dimension_values
from tools.evidence.schema import (
    json_field,
    missing_columns,
    non_finite_values,
    parse_number,
    read_csv_rows,
    read_json,
    truthy_number,
    values_match,
)


RUN_STATUS_FIELDS: tuple[str, ...] = (
    "campaign_id",
    "profile",
    "run_id",
    "scenario_id",
    "ecology_context",
    "init_arm",
    "perception_condition",
    "seed",
    "status",
    "metrics_path",
    "config_path",
    "errors",
    "warnings",
)


def scenario_id(axes: Mapping[str, Any]) -> str:
    parts = []
    for key in sorted(k for k in axes if k != "seed"):
        parts.append(f"{key}={axes[key]}")
    return "|".join(parts)


def _run_key(axes: Mapping[str, Any], seed: Any) -> tuple[Any, ...]:
    return (
        axes.get("ecology_context", ""),
        axes.get("init_arm", ""),
        axes.get("perception_condition", axes.get("cognition_condition", "")),
        seed,
    )


def expected_run_keys(manifest: Mapping[str, Any], profile: ProfileContract) -> set[tuple[Any, ...]]:
    matrix_values = scenario_dimension_values(manifest)
    dim_values = expected_dimension_values(profile, matrix_values)
    seeds = list(manifest.get("seed_list") or profile.expected_values.get("seed") or ())
    scenario_dims = [dim for dim in profile.required_dimensions if dim != "seed"]
    value_lists = [list(dim_values.get(dim) or matrix_values.get(dim) or ()) for dim in scenario_dims]
    keys: set[tuple[Any, ...]] = set()
    for values in itertools.product(*value_lists):
        axes = dict(zip(scenario_dims, values))
        for seed in seeds:
            keys.add(_run_key(axes, seed))
    return keys


def collect_harness_runs(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    summary_paths = [Path(path) for path in manifest.get("harness_summary_paths") or []]
    if not summary_paths:
        run_root = Path(str(manifest.get("run_root") or ""))
        summary_paths = sorted(run_root.rglob("summary.json")) if run_root.exists() else []

    records: list[dict[str, Any]] = []
    for summary_path in sorted(summary_paths):
        payload, error = read_json(summary_path)
        if error or not isinstance(payload, Mapping):
            records.append(
                {
                    "run_id": f"invalid-summary:{summary_path}",
                    "status": "failed",
                    "axes": {},
                    "seed": "",
                    "summary_path": str(summary_path),
                    "_summary_error": error or "summary JSON is not an object",
                }
            )
            continue
        for item in payload.get("runs") or []:
            if not isinstance(item, Mapping):
                continue
            record = dict(item)
            run_dir = Path(str(record.get("run_dir") or ""))
            record["summary_path"] = str(summary_path)
            record["metrics_path"] = str(run_dir / "metrics.csv") if str(run_dir) else ""
            record["config_path"] = str(run_dir / "config.json") if str(run_dir) else ""
            records.append(record)
    records.sort(key=lambda row: (str(row.get("run_id") or ""), str(row.get("summary_path") or "")))
    return records


def _metric_checks(
    metrics_path: Path,
    profile: ProfileContract,
    errors: list[str],
    warnings: list[str],
    anomalies: list[str],
) -> dict[str, Any]:
    rows, columns, csv_error = read_csv_rows(metrics_path)
    if csv_error:
        errors.append(f"metrics CSV unreadable: {csv_error}")
        anomalies.append("schema_failure")
        return {}
    if not rows:
        errors.append("metrics CSV is empty")
        anomalies.append("empty_metrics")
        return {}
    final = rows[-1]
    if not final:
        errors.append("metrics CSV has no final row")
        anomalies.append("missing_final_metrics_row")
        return {}

    missing = missing_columns(columns, profile.required_metric_columns)
    if missing:
        errors.append(f"missing required metrics columns: {', '.join(missing)}")
        anomalies.append("schema_failure")

    bad = non_finite_values(final, profile.finite_metric_columns)
    if bad:
        errors.append(f"non-finite metric values in final row: {', '.join(bad)}")
        anomalies.append("nan_inf_failure")

    for column in profile.reward_leakage_columns:
        if column in final and truthy_number(final.get(column)):
            errors.append(f"reward leakage flag active: {column}")
            anomalies.append("reward_leakage_suspicion")

    alive = parse_number(final.get("alive"))
    if alive is not None and alive <= 0:
        warnings.append("final alive count is zero")
        anomalies.append("extinction_collapse_anomaly")

    return final


def _config_checks(
    config_path: Path,
    profile: ProfileContract,
    axes: Mapping[str, Any],
    errors: list[str],
    anomalies: list[str],
) -> None:
    config, error = read_json(config_path)
    if error or not isinstance(config, Mapping):
        errors.append(f"config JSON unreadable: {error or 'config root is not an object'}")
        anomalies.append("schema_failure")
        return

    for expectation in profile.config_expectations:
        actual = json_field(config, expectation.section, expectation.field)
        if not values_match(actual, expectation.expected):
            errors.append(f"{expectation.section}.{expectation.field}: {expectation.message}")
            anomalies.append("schema_failure")

    sim_cfg = config.get("sim_cfg") if isinstance(config.get("sim_cfg"), Mapping) else {}
    env_cfg = config.get("env_cfg") if isinstance(config.get("env_cfg"), Mapping) else {}
    init_arm = str(axes.get("init_arm") or "")
    perception = str(axes.get("perception_condition") or "")
    ecology = str(axes.get("ecology_context") or "")

    founder_enabled = bool(sim_cfg.get("lineage_founder_balance_enable", False))
    if init_arm == "init_legacy" and founder_enabled:
        errors.append("init_legacy has lineage_founder_balance_enable=true")
        anomalies.append("schema_failure")
    if init_arm == "init_founder_balanced" and not founder_enabled:
        errors.append("init_founder_balanced has lineage_founder_balance_enable=false")
        anomalies.append("schema_failure")

    minimal_channels = bool(sim_cfg.get("force_minimal_sensor_channels_enabled", False))
    if perception == "perception_normal" and minimal_channels:
        errors.append("perception_normal has force_minimal_sensor_channels_enabled=true")
        anomalies.append("schema_failure")
    if perception == "perception_ablated" and not minimal_channels:
        errors.append("perception_ablated has force_minimal_sensor_channels_enabled=false")
        anomalies.append("schema_failure")

    pressure_fields = (
        "patch_world_enable",
        "patch_world_observation_enable",
        "patch_world_waste_consequence_enable",
        "jungle_predator_pressure_enable",
        "jungle_reproduction_emissions_enable",
        "jungle_terrain_protection_enable",
        "jungle_ecological_recovery_enable",
    )
    if ecology.endswith("flat_control_clean"):
        bad = [field for field in pressure_fields if bool(env_cfg.get(field, False))]
        if bad:
            errors.append(f"flat control has pressure flags active: {', '.join(bad)}")
            anomalies.append("schema_failure")
    elif ecology:
        bad = [field for field in pressure_fields if not bool(env_cfg.get(field, False))]
        if bad and ("jungle" in ecology or "pressure" in ecology or "recovery" in ecology):
            errors.append(f"pressure context has inactive flags: {', '.join(bad)}")
            anomalies.append("schema_failure")


def classify_runs(manifest: Mapping[str, Any], profile: ProfileContract) -> dict[str, Any]:
    expected = expected_run_keys(manifest, profile)
    records = collect_harness_runs(manifest)
    observed: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []

    for record in records:
        axes = record.get("axes") if isinstance(record.get("axes"), Mapping) else {}
        seed = record.get("seed")
        key = _run_key(axes, seed)
        observed.setdefault(key, []).append(record)

    seen_run_ids: dict[str, int] = {}
    for record in records:
        run_id = str(record.get("run_id") or "")
        seen_run_ids[run_id] = seen_run_ids.get(run_id, 0) + 1

    for record in records:
        axes = record.get("axes") if isinstance(record.get("axes"), Mapping) else {}
        seed = record.get("seed")
        key = _run_key(axes, seed)
        run_id = str(record.get("run_id") or "")
        errors: list[str] = []
        warnings: list[str] = []
        anomalies: list[str] = []

        if key not in expected:
            anomalies.append("unknown_scenario" if axes else "scenario_mismatch")
            errors.append("run scenario/seed is not declared by manifest/profile")
        if seed not in (manifest.get("seed_list") or ()):
            anomalies.append("seed_mismatch")
            errors.append(f"unexpected seed: {seed}")
        if len(observed.get(key, [])) > 1:
            anomalies.append("duplicate_scenario_seed")
            errors.append("duplicate scenario/seed run")
        if run_id and seen_run_ids.get(run_id, 0) > 1:
            anomalies.append("duplicate_run")
            errors.append("duplicate run_id")
        if key not in expected:
            anomalies.append("extra_run")
        status = str(record.get("status") or "")
        if status != "ok" or int(record.get("returncode", 0) or 0) != 0:
            anomalies.append("failed_run")
            errors.append(f"run failed or incomplete: status={status!r} returncode={record.get('returncode')!r}")
        if record.get("_summary_error"):
            anomalies.append("schema_failure")
            errors.append(str(record["_summary_error"]))

        metrics_path = Path(str(record.get("metrics_path") or ""))
        config_path = Path(str(record.get("config_path") or ""))
        final_metrics: dict[str, Any] = {}
        if not str(metrics_path) or not metrics_path.exists():
            errors.append("metrics.csv missing")
            anomalies.append("missing_run" if status == "ok" else "failed_run")
            anomalies.append("empty_metrics")
        else:
            final_metrics = _metric_checks(metrics_path, profile, errors, warnings, anomalies)
        if not str(config_path) or not config_path.exists():
            errors.append("config.json missing")
            anomalies.append("missing_config")
        else:
            _config_checks(config_path, profile, axes, errors, anomalies)

        if final_metrics:
            no_effect_columns = [
                "behavior_patch_interaction_proxy",
                "behavior_risk_exposure_proxy",
                "behavior_visible_resource_exposure_proxy",
            ]
            if any(column in final_metrics for column in no_effect_columns) and all(
                (parse_number(final_metrics.get(column)) or 0.0) == 0.0
                for column in no_effect_columns
                if column in final_metrics
            ):
                warnings.append("all tracked behavior effect proxies are zero")
                anomalies.append("suspicious_no_effect_scenario")

        if not anomalies:
            anomalies.append("complete")
        row = {
            "campaign_id": manifest.get("campaign_id"),
            "profile": profile.name,
            "run_id": run_id,
            "scenario_id": scenario_id(axes),
            "ecology_context": axes.get("ecology_context", ""),
            "init_arm": axes.get("init_arm", ""),
            "perception_condition": axes.get("perception_condition", axes.get("cognition_condition", "")),
            "seed": seed,
            "status": "complete" if anomalies == ["complete"] else "anomalous",
            "metrics_path": str(metrics_path) if str(metrics_path) else "",
            "config_path": str(config_path) if str(config_path) else "",
            "errors": "; ".join(dict.fromkeys(errors)),
            "warnings": "; ".join(dict.fromkeys(warnings)),
            "anomalies": list(dict.fromkeys(anomalies)),
            "final_metrics": final_metrics,
        }
        rows.append(row)

    for key in sorted(expected - set(observed), key=lambda item: tuple(str(part) for part in item)):
        ecology, init_arm, perception_condition, seed = key
        axes = {
            "ecology_context": ecology,
            "init_arm": init_arm,
            "perception_condition": perception_condition,
        }
        rows.append(
            {
                "campaign_id": manifest.get("campaign_id"),
                "profile": profile.name,
                "run_id": "",
                "scenario_id": scenario_id(axes),
                "ecology_context": ecology,
                "init_arm": init_arm,
                "perception_condition": perception_condition,
                "seed": seed,
                "status": "anomalous",
                "metrics_path": "",
                "config_path": "",
                "errors": "expected scenario/seed run is missing",
                "warnings": "",
                "anomalies": ["missing_run"],
                "final_metrics": {},
            }
        )

    rows.sort(
        key=lambda row: (
            str(row.get("ecology_context") or ""),
            str(row.get("init_arm") or ""),
            str(row.get("perception_condition") or ""),
            int(row.get("seed") or 0),
            str(row.get("run_id") or ""),
        )
    )
    return {
        "rows": rows,
        "anomaly_summary": summarize_anomalies(rows),
        "observed_keys": set(observed),
        "expected_keys": expected,
    }


def run_status_csv_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in RUN_STATUS_FIELDS} for row in rows]


def write_run_status_csv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RUN_STATUS_FIELDS))
        writer.writeheader()
        writer.writerows(run_status_csv_rows(rows))
