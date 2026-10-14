"""Anomaly names and summary helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


ANOMALY_TYPES: tuple[str, ...] = (
    "complete",
    "missing_run",
    "failed_run",
    "empty_metrics",
    "schema_failure",
    "nan_inf_failure",
    "scenario_mismatch",
    "seed_mismatch",
    "suspicious_no_effect_scenario",
    "extinction_collapse_anomaly",
    "reward_leakage_suspicion",
    "duplicate_run",
    "duplicate_scenario_seed",
    "extra_run",
    "unknown_scenario",
    "missing_config",
    "missing_final_metrics_row",
)

FAILURE_ANOMALIES = tuple(name for name in ANOMALY_TYPES if name != "complete")


def summarize_anomalies(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, set[str]]] = {
        name: {"run_ids": set(), "scenario_ids": set()} for name in ANOMALY_TYPES
    }
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        anomalies = row.get("anomalies") or []
        if isinstance(anomalies, str):
            anomalies = [item for item in anomalies.split(";") if item]
        for anomaly in anomalies:
            if anomaly not in grouped:
                grouped[anomaly] = {"run_ids": set(), "scenario_ids": set()}
            counts[anomaly] += 1
            run_id = str(row.get("run_id") or "")
            scenario_id = str(row.get("scenario_id") or "")
            if run_id:
                grouped[anomaly]["run_ids"].add(run_id)
            if scenario_id:
                grouped[anomaly]["scenario_ids"].add(scenario_id)
    return {
        "counts": {name: counts.get(name, 0) for name in sorted(grouped)},
        "affected": {
            name: {
                "run_ids": sorted(values["run_ids"]),
                "scenario_ids": sorted(values["scenario_ids"]),
            }
            for name, values in sorted(grouped.items())
            if counts.get(name, 0)
        },
    }
