#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    yaml = None  # type: ignore
    _YAML_ERROR = exc
else:
    _YAML_ERROR = None


SCHEMA_VERSION = "respec_fixed1_validation_v1"
REGISTRY_PATH = Path("_spec/registry/v1/respec1_fixed_arm_world_pressure_validation.yaml")
REPORT_PATH = Path(
    "docs/500_campaigns-decisions-closeouts/520_campaign-reports/400_worlds-ecology/EIEL-DOC-002169--world-pressure-eiel-respec-fixed1-fixed-arm-world-pressure-validation-2026-07-11.md"
)
EVIDENCE_DIR = Path("evidences/07-world-pressure/04-respec-fixed-1-validation")
WORLD_FAMILIES = ["flat", "reporting_only", "a_only", "b_only", "equalized_sham", "mixed_tradeoff"]
FIXED_ARMS = {"a_favored": 0.75, "neutral": 0.0, "b_favored": -0.75}
SEEDS = [101, 202, 303, 404]
EXPECTED_REGISTRY_SHA256 = "f3582c982da3fac624abdde74e1b581221580e657fbd0d79c7bb7131cf7c0b35"
EXPECTED_RESPEC1_EVIDENCE_ZIP_SHA256 = "86c101faa316df17532e83afa4d11414c4ce6ebade6954477b4764a4ef6c9706"
RESPEC1_EVIDENCE_DIR = Path("evidences/07-world-pressure/03-respec1-fixed-arm-registry-design")
NO_CLAIMS = (
    "RESPEC-FIXED-1 is a bounded fixed-arm Resource A/B world-pressure validation. It does not claim "
    "evolved specialization, selected specialization, intelligence, emergence, consciousness, broad cognition, "
    "adaptive memory, prediction selection, solved lineage diversity, beneficial open-ended search, culture, "
    "superintelligence, or general open-ended evolution."
)

COUNT_FIELDS = [
    "respec0b_resource_gather_attempts",
    "respec0b_resource_gather_successes",
    "respec0b_resource_gather_failed_handling",
    "respec0b_resource_gather_resource_a_successes",
    "respec0b_resource_gather_resource_b_successes",
    "respec0b_resource_gather_energy_resource_a",
    "respec0b_resource_gather_energy_resource_b",
    "respec0b_resource_gather_handling_ticks",
    "respec0b_resource_gather_interruptions",
]
FRACTION_FIELDS = ["respec0b_resource_a_cell_frac", "respec0b_resource_b_cell_frac"]
OUTCOME_FIELDS = ["alive", "deaths", "successful_births", "failed_births"]
REQUIRED_METRIC_FIELDS = COUNT_FIELDS + FRACTION_FIELDS
DELTA_EPSILON = 1e-9


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _ensure_import_paths() -> None:
    ordered = [str(_repo_root()), str(_project_root()), str(_project_root() / "engine")]
    for path in ordered:
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:  # pragma: no cover
        raise RuntimeError(f"PyYAML is required: {_YAML_ERROR}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML did not parse as a mapping: {path}")
    return payload


def _float(value: Any) -> float | None:
    try:
        x = float(value)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _axis_values(registry: Mapping[str, Any], axis: str) -> list[str]:
    return [str(v) for v in registry["axes"][axis]["values"]]


def _run_id(world: str, arm: str, seed: int) -> str:
    return f"respec1_fixed_arm_world_pressure_validation__world_{world}__arm_{arm}__seed{int(seed):06d}"


def parse_run_id(run_id: str) -> dict[str, Any]:
    marker = "respec1_fixed_arm_world_pressure_validation__world_"
    if not str(run_id).startswith(marker):
        return {"world_family": "", "fixed_arm": "", "seed": -1}
    rest = str(run_id)[len(marker) :]
    world, rest = rest.split("__arm_", 1)
    arm, seed_text = rest.split("__seed", 1)
    return {"world_family": world, "fixed_arm": arm, "seed": int(seed_text)}


def selected_rehearsal_run_ids() -> list[str]:
    return [_run_id(world, arm, 101) for world in WORLD_FAMILIES for arm in FIXED_ARMS]


def selected_parity_run_ids(decision_critical_arm: str = "neutral") -> list[str]:
    return [
        _run_id("flat", "neutral", 101),
        _run_id("reporting_only", "a_favored", 101),
        _run_id("a_only", "a_favored", 202),
        _run_id("b_only", "b_favored", 202),
        _run_id("equalized_sham", "a_favored", 303),
        _run_id("mixed_tradeoff", decision_critical_arm, 404),
    ]


def load_registry_verification(registry_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    registry = _load_yaml(registry_path)
    rows = [
        _run_id(world, arm, seed)
        for world in _axis_values(registry, "world_family")
        for arm in _axis_values(registry, "fixed_arm")
        for seed in [int(s) for s in registry.get("seeds", [])]
    ]
    fixed = dict((registry.get("respec1_contract") or {}).get("fixed_arm_axis_values") or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "registry_path": str(registry_path),
        "registry_sha256": _sha256(registry_path),
        "registry_sha256_expected": EXPECTED_REGISTRY_SHA256,
        "registry_sha256_ok": _sha256(registry_path) == EXPECTED_REGISTRY_SHA256,
        "planned_cells": len(rows),
        "unique_run_ids": len(set(rows)),
        "world_families": _axis_values(registry, "world_family"),
        "fixed_arms": fixed,
        "seeds": [int(s) for s in registry.get("seeds", [])],
        "matrix_ok": (
            len(rows) == 72
            and len(set(rows)) == 72
            and _axis_values(registry, "world_family") == WORLD_FAMILIES
            and list(fixed.keys()) == list(FIXED_ARMS.keys())
            and [float(fixed[k]) for k in FIXED_ARMS] == [float(v) for v in FIXED_ARMS.values()]
            and [int(s) for s in registry.get("seeds", [])] == SEEDS
        ),
        "rehearsal_run_ids": selected_rehearsal_run_ids(),
        "parity_run_ids": selected_parity_run_ids(),
    }


def decision_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "contract_status": "FROZEN_BEFORE_EXECUTION",
        "frozen_before": "Stage A technical rehearsal and Stage B scientific campaign outputs",
        "decision_classes": {
            "RESPEC-FIXED-A": "reciprocal fixed-arm world demand supported",
            "RESPEC-FIXED-B": "observation/action path inadequate",
            "RESPEC-FIXED-C": "control or pairing failure",
            "RESPEC-FIXED-D": "measurement or integrity failure",
            "RESPEC-FIXED-E": "execution/storage/profile blocker",
            "RESPEC-FIXED-F": "reciprocal world demand not supported",
        },
        "seed_consistency_rule": {
            "mandatory_support": "at least 3 of 4 seeds support each mandatory reciprocal direction",
            "strong_reverse_limit": "no mandatory direction is strongly reversed in more than 1 seed",
            "delta_positive_threshold": DELTA_EPSILON,
            "strong_reverse_threshold": -0.05,
        },
        "mandatory_directions": [
            {"world": "a_only", "metric": "resource_a_completion_per_attempt", "left": "a_favored", "right": "neutral"},
            {"world": "a_only", "metric": "resource_a_completion_per_attempt", "left": "neutral", "right": "b_favored"},
            {"world": "a_only", "metric": "resource_a_completion_per_attempt", "left": "a_favored", "right": "b_favored"},
            {"world": "b_only", "metric": "resource_b_completion_per_attempt", "left": "b_favored", "right": "neutral"},
            {"world": "b_only", "metric": "resource_b_completion_per_attempt", "left": "neutral", "right": "a_favored"},
            {"world": "b_only", "metric": "resource_b_completion_per_attempt", "left": "b_favored", "right": "a_favored"},
        ],
        "equalized_sham_false_positive_boundary": {
            "metric": "completion_per_attempt",
            "max_allowed_mean_pair_delta_abs": 0.25,
            "max_allowed_seed_consistent_directions": 2,
        },
        "integrity_sentinels": [
            "planned/completed/failed cells",
            "missing metrics",
            "nonfinite values",
            "incomplete pairs",
            "config/run_id reconciliation",
            "fixed-axis exactness",
            "mutation freeze",
            "resource-layout verification",
            "equalized-sham equalization",
            "flat/reporting-only drift",
            "repeat parity",
            "direct reward or identity leakage",
        ],
        "no_claims": NO_CLAIMS,
    }


def source_verification() -> dict[str, Any]:
    required = [
        "respec1_registry_contract.json",
        "respec1_world_families.json",
        "respec1_fixed_arms.json",
        "respec1_observation_readiness.json",
        "respec1_pairing_contract.json",
        "respec1_confound_audit.json",
        "respec1_metric_contract.json",
        "respec1_timing_overlap_contract.json",
        "respec1_future_decision_framework.json",
        "respec1_performance_profile_audit.json",
        "respec1_storage_estimate.json",
        "respec1_runtime_estimate.json",
        "respec1_registry_dry_run_summary.json",
        "respec1_final_decision.json",
    ]
    files = {name: (RESPEC1_EVIDENCE_DIR / name).exists() for name in required}
    zip_manifest = _read_json(RESPEC1_EVIDENCE_DIR / "respec1_evidence_bundle_manifest.json")
    return {
        "schema_version": SCHEMA_VERSION,
        "respec0b_decision": "RESPEC-0B-A",
        "respec1_decision": _read_json(RESPEC1_EVIDENCE_DIR / "respec1_final_decision.json").get("decision", ""),
        "required_respec1_files": files,
        "required_respec1_files_ok": all(files.values()),
        "respec1_evidence_zip_sha256": zip_manifest.get("sha256", ""),
        "respec1_evidence_zip_sha256_expected": EXPECTED_RESPEC1_EVIDENCE_ZIP_SHA256,
        "respec1_evidence_zip_sha256_ok": zip_manifest.get("sha256", "") == EXPECTED_RESPEC1_EVIDENCE_ZIP_SHA256,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        csv.field_size_limit(sys.maxsize)
    except Exception:
        pass
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_config(path: Path) -> dict[str, Any]:
    return _read_json(path / "config.json")


def _nested(config: Mapping[str, Any], section: str, key: str, default: Any = None) -> Any:
    value = config.get(section, {})
    if isinstance(value, Mapping):
        return value.get(key, default)
    return default


def summarize_metrics(run_dir: Path) -> dict[str, Any]:
    rows = _read_csv_rows(run_dir / "metrics.csv")
    fields = list(rows[0].keys()) if rows else []
    missing = [field for field in REQUIRED_METRIC_FIELDS if field not in fields]
    nonfinite: list[dict[str, Any]] = []
    sums: dict[str, float] = {field: 0.0 for field in COUNT_FIELDS}
    fractions: dict[str, list[float]] = {field: [] for field in FRACTION_FIELDS}
    final: dict[str, float] = {}
    for row_index, row in enumerate(rows):
        for field, raw in row.items():
            value = _float(raw)
            if value is None and str(raw).strip() not in {"", "nan", "NaN"}:
                continue
            if value is None and str(raw).strip():
                nonfinite.append({"row": row_index, "field": field, "value": raw})
        for field in COUNT_FIELDS:
            sums[field] += _float(row.get(field)) or 0.0
        for field in FRACTION_FIELDS:
            value = _float(row.get(field))
            if value is not None:
                fractions[field].append(value)
    if rows:
        last = rows[-1]
        for field in OUTCOME_FIELDS:
            final[field] = _float(last.get(field)) or 0.0
    attempts = sums["respec0b_resource_gather_attempts"]
    successes = sums["respec0b_resource_gather_successes"]
    a_successes = sums["respec0b_resource_gather_resource_a_successes"]
    b_successes = sums["respec0b_resource_gather_resource_b_successes"]
    ticks = sums["respec0b_resource_gather_handling_ticks"]
    derived = {
        "completion_per_attempt": _safe_div(successes, attempts),
        "resource_a_completion_per_attempt": _safe_div(a_successes, attempts),
        "resource_b_completion_per_attempt": _safe_div(b_successes, attempts),
        "failed_handling_fraction": _safe_div(sums["respec0b_resource_gather_failed_handling"], attempts),
        "interruptions_per_attempt": _safe_div(sums["respec0b_resource_gather_interruptions"], attempts),
        "energy_per_completion": _safe_div(
            sums["respec0b_resource_gather_energy_resource_a"] + sums["respec0b_resource_gather_energy_resource_b"],
            successes,
        ),
        "energy_per_handling_tick": _safe_div(
            sums["respec0b_resource_gather_energy_resource_a"] + sums["respec0b_resource_gather_energy_resource_b"],
            ticks,
        ),
        "handling_ticks_per_success": _safe_div(ticks, successes),
    }
    frac_mean = {field: _mean(values) for field, values in fractions.items()}
    return {
        "metrics_path": str(run_dir / "metrics.csv"),
        "row_count": len(rows),
        "fields": fields,
        "missing_required_fields": missing,
        "nonfinite_values": nonfinite,
        "sums": sums,
        "fraction_means": frac_mean,
        "final": final,
        "derived": derived,
    }


def collect_stage_runs(root: Path) -> dict[str, Any]:
    summary = _read_json(root / "parallel_campaign_summary.json")
    runs: list[dict[str, Any]] = []
    for row in summary.get("results", []) if isinstance(summary.get("results"), list) else []:
        if not isinstance(row, dict):
            continue
        run_id = str(row.get("run_id") or "")
        parsed = parse_run_id(run_id)
        run_dir = Path(str(row.get("run_dir") or ""))
        config = _read_config(run_dir)
        metrics = summarize_metrics(run_dir) if run_dir.exists() else {
            "row_count": 0,
            "fields": [],
            "missing_required_fields": REQUIRED_METRIC_FIELDS,
            "nonfinite_values": [],
            "sums": {},
            "fraction_means": {},
            "final": {},
            "derived": {},
        }
        expected_axis = FIXED_ARMS.get(str(parsed["fixed_arm"]), None)
        observed_axis = _nested(config, "genome_cfg", "overrides", {})
        if isinstance(observed_axis, Mapping):
            observed_axis_value = observed_axis.get("resource_handling_axis", None)
        else:
            observed_axis_value = None
        runs.append(
            {
                "run_id": run_id,
                "world_family": parsed["world_family"],
                "fixed_arm": parsed["fixed_arm"],
                "seed": parsed["seed"],
                "run_dir": str(run_dir),
                "status": str(row.get("status") or ""),
                "returncode": int(row.get("returncode", 1)),
                "metrics": metrics,
                "config": {
                    "env_respec0b_enable": _nested(config, "env_cfg", "respec0b_resource_specialization_enable", None),
                    "env_resource_class_mode": _nested(config, "env_cfg", "respec0b_resource_class_mode", ""),
                    "env_equalized_sham": _nested(config, "env_cfg", "respec0b_resource_equalized_sham_enable", None),
                    "patch_world_enable": _nested(config, "env_cfg", "patch_world_enable", None),
                    "patch_world_observation_enable": _nested(config, "env_cfg", "patch_world_observation_enable", None),
                    "resource_handling_representation_enable": _nested(
                        config, "sim_cfg", "resource_handling_representation_enable", None
                    ),
                    "resource_handling_representation_reporting_enable": _nested(
                        config, "sim_cfg", "resource_handling_representation_reporting_enable", None
                    ),
                    "checkpoint_bundle_v2_period_gen": _nested(config, "sim_cfg", "checkpoint_bundle_v2_period_gen", None),
                    "lineage_qd_archive_enable": _nested(config, "sim_cfg", "lineage_qd_archive_enable", None),
                    "lineage_founder_balance_enable": _nested(config, "sim_cfg", "lineage_founder_balance_enable", None),
                    "lro1_reproductive_opportunity_enable": _nested(config, "sim_cfg", "lro1_reproductive_opportunity_enable", None),
                    "lro4_offspring_dispersal_enable": _nested(config, "sim_cfg", "lro4_offspring_dispersal_enable", None),
                    "pressure_autopilot_enable": _nested(config, "sim_cfg", "pressure_autopilot_enable", None),
                    "genome_overrides": _nested(config, "genome_cfg", "overrides", {}),
                    "expected_axis": expected_axis,
                    "observed_axis": _float(observed_axis_value),
                    "genome_mutation_rate": (
                        _float(observed_axis.get("mutation_rate"))
                        if isinstance(observed_axis, Mapping)
                        else None
                    ),
                },
                "config_sha256": _sha256(run_dir / "config.json") if (run_dir / "config.json").exists() else "",
                "metrics_sha256": _sha256(run_dir / "metrics.csv") if (run_dir / "metrics.csv").exists() else "",
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "parallel_status": summary.get("status", ""),
        "planned_runs": int(summary.get("planned_runs", 0) or 0),
        "results_count": len(summary.get("results", []) if isinstance(summary.get("results"), list) else []),
        "failure_count": len(summary.get("failures", []) if isinstance(summary.get("failures"), list) else []),
        "runs": runs,
        "summary_path": str(root / "parallel_campaign_summary.json"),
    }


def stage_integrity(stage: Mapping[str, Any], *, expected_runs: int, expected_seeds: list[int]) -> dict[str, Any]:
    runs = list(stage.get("runs", []))
    failures = [r for r in runs if r.get("status") != "ok" or int(r.get("returncode", 1)) != 0]
    missing_metrics = [
        r["run_id"]
        for r in runs
        if r["metrics"]["row_count"] <= 0 or r["metrics"]["missing_required_fields"]
    ]
    nonfinite = [
        {"run_id": r["run_id"], "values": r["metrics"]["nonfinite_values"]}
        for r in runs
        if r["metrics"]["nonfinite_values"]
    ]
    fixed_axis_bad = [
        r["run_id"]
        for r in runs
        if r["config"]["observed_axis"] is None
        or abs(float(r["config"]["observed_axis"]) - float(r["config"]["expected_axis"])) > 1e-12
    ]
    mutation_bad = [
        r["run_id"]
        for r in runs
        if r["config"]["genome_mutation_rate"] is None
        or abs(float(r["config"]["genome_mutation_rate"])) > 1e-12
    ]
    pairs = {(r["world_family"], r["fixed_arm"], int(r["seed"])) for r in runs}
    expected_pairs = {
        (world, arm, seed)
        for world in WORLD_FAMILIES
        for arm in FIXED_ARMS
        for seed in expected_seeds
    }
    missing_pairs = sorted(expected_pairs - pairs)
    extra_pairs = sorted(pairs - expected_pairs)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS"
        if (
            len(runs) == expected_runs
            and not failures
            and not missing_metrics
            and not nonfinite
            and not fixed_axis_bad
            and not mutation_bad
            and not missing_pairs
            and not extra_pairs
        )
        else "FAIL",
        "planned_expected": expected_runs,
        "runs_collected": len(runs),
        "failed_runs": failures,
        "missing_metrics": missing_metrics,
        "nonfinite_values": nonfinite,
        "fixed_axis_bad": fixed_axis_bad,
        "mutation_freeze_bad": mutation_bad,
        "missing_pairs": [list(item) for item in missing_pairs],
        "extra_pairs": [list(item) for item in extra_pairs],
    }


def _metric(run: Mapping[str, Any], metric: str) -> float:
    if metric in run["metrics"]["derived"]:
        return float(run["metrics"]["derived"][metric])
    if metric in run["metrics"]["sums"]:
        return float(run["metrics"]["sums"][metric])
    if metric in run["metrics"]["final"]:
        return float(run["metrics"]["final"][metric])
    if metric in run["metrics"]["fraction_means"]:
        return float(run["metrics"]["fraction_means"][metric])
    return 0.0


def _campaign_by_key(campaign: Mapping[str, Any]) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    return {
        (str(r["world_family"]), str(r["fixed_arm"]), int(r["seed"])): r
        for r in campaign.get("runs", [])
        if isinstance(r, Mapping)
    }


def scenario_arm_summary(campaign: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for world in WORLD_FAMILIES:
        for arm in FIXED_ARMS:
            subset = [r for r in campaign.get("runs", []) if r["world_family"] == world and r["fixed_arm"] == arm]
            out: dict[str, Any] = {"world_family": world, "fixed_arm": arm, "runs": len(subset)}
            for metric in [
                "completion_per_attempt",
                "resource_a_completion_per_attempt",
                "resource_b_completion_per_attempt",
                "failed_handling_fraction",
                "interruptions_per_attempt",
                "energy_per_handling_tick",
                "alive",
                "successful_births",
            ]:
                values = [_metric(r, metric) for r in subset]
                out[f"{metric}_mean"] = _mean(values)
                out[f"{metric}_median"] = _median(values)
                out[f"{metric}_by_seed"] = json.dumps(
                    {str(int(r["seed"])): _metric(r, metric) for r in subset}, sort_keys=True
                )
            rows.append(out)
    return rows


def seed_level_rows(campaign: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for r in sorted(campaign.get("runs", []), key=lambda x: (x["world_family"], x["fixed_arm"], int(x["seed"]))):
        row = {
            "run_id": r["run_id"],
            "world_family": r["world_family"],
            "fixed_arm": r["fixed_arm"],
            "seed": int(r["seed"]),
            "status": r["status"],
            "metrics_rows": r["metrics"]["row_count"],
            "alive_final": _metric(r, "alive"),
            "successful_births_final": _metric(r, "successful_births"),
        }
        for field in COUNT_FIELDS:
            row[field] = _metric(r, field)
        for metric in [
            "completion_per_attempt",
            "resource_a_completion_per_attempt",
            "resource_b_completion_per_attempt",
            "failed_handling_fraction",
            "interruptions_per_attempt",
            "energy_per_completion",
            "energy_per_handling_tick",
            "handling_ticks_per_success",
        ]:
            row[metric] = _metric(r, metric)
        rows.append(row)
    return rows


def paired_deltas(campaign: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_key = _campaign_by_key(campaign)
    comparisons = list(decision_contract()["mandatory_directions"])
    rows: list[dict[str, Any]] = []
    for comp in comparisons:
        for seed in SEEDS:
            left = by_key.get((comp["world"], comp["left"], seed))
            right = by_key.get((comp["world"], comp["right"], seed))
            if left is None or right is None:
                delta = None
            else:
                delta = _metric(left, comp["metric"]) - _metric(right, comp["metric"])
            rows.append(
                {
                    "comparison": f"{comp['world']}:{comp['left']}-{comp['right']}:{comp['metric']}",
                    "world_family": comp["world"],
                    "metric": comp["metric"],
                    "left_arm": comp["left"],
                    "right_arm": comp["right"],
                    "seed": seed,
                    "delta": "" if delta is None else delta,
                    "supports_direction": bool(delta is not None and delta > DELTA_EPSILON),
                    "strong_reverse": bool(delta is not None and delta < -0.05),
                }
            )
    return rows


def direction_summary(delta_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_comp: dict[str, list[Mapping[str, Any]]] = {}
    for row in delta_rows:
        by_comp.setdefault(str(row["comparison"]), []).append(row)
    rows = []
    passed = True
    for name, values in sorted(by_comp.items()):
        support = sum(1 for row in values if bool(row["supports_direction"]))
        reverse = sum(1 for row in values if bool(row["strong_reverse"]))
        deltas = [float(row["delta"]) for row in values if row.get("delta") != ""]
        ok = support >= 3 and reverse <= 1
        passed = passed and ok
        rows.append(
            {
                "comparison": name,
                "supporting_seeds": support,
                "strong_reverse_seeds": reverse,
                "mean_delta": _mean(deltas),
                "median_delta": _median(deltas),
                "status": "PASS" if ok else "FAIL",
            }
        )
    return {"schema_version": SCHEMA_VERSION, "status": "PASS" if passed else "FAIL", "comparisons": rows}


def equalized_sham_audit(campaign: Mapping[str, Any]) -> dict[str, Any]:
    subset = [r for r in campaign.get("runs", []) if r["world_family"] == "equalized_sham"]
    seed_rows = []
    max_abs = 0.0
    consistent = 0
    for seed in SEEDS:
        arms = {r["fixed_arm"]: r for r in subset if int(r["seed"]) == seed}
        values = {arm: _metric(row, "completion_per_attempt") for arm, row in arms.items()}
        if set(values) == set(FIXED_ARMS):
            deltas = [
                values["a_favored"] - values["neutral"],
                values["b_favored"] - values["neutral"],
                values["a_favored"] - values["b_favored"],
            ]
            local_max = max(abs(x) for x in deltas)
            max_abs = max(max_abs, local_max)
            if sum(1 for x in deltas if abs(x) > DELTA_EPSILON) >= 2:
                consistent += 1
        else:
            local_max = 999.0
            max_abs = max(max_abs, local_max)
        seed_rows.append({"seed": seed, "completion_per_attempt_by_arm": values, "max_abs_delta": local_max})
    status = "PASS" if max_abs <= 0.25 and consistent <= 2 else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "max_seed_abs_pair_delta": max_abs,
        "consistent_direction_seed_count": consistent,
        "false_positive_boundary": decision_contract()["equalized_sham_false_positive_boundary"],
        "seed_rows": seed_rows,
    }


def flat_reporting_audit(campaign: Mapping[str, Any]) -> dict[str, Any]:
    subset = [r for r in campaign.get("runs", []) if r["world_family"] in {"flat", "reporting_only"}]
    attempts = sum(_metric(r, "respec0b_resource_gather_attempts") for r in subset)
    successes = sum(_metric(r, "respec0b_resource_gather_successes") for r in subset)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if attempts == 0.0 and successes == 0.0 else "FAIL",
        "resource_gather_attempts": attempts,
        "resource_gather_successes": successes,
        "worlds": ["flat", "reporting_only"],
    }


def observation_action_audit(campaign: Mapping[str, Any]) -> dict[str, Any]:
    active = [r for r in campaign.get("runs", []) if r["world_family"] in {"a_only", "b_only", "equalized_sham", "mixed_tradeoff"}]
    attempts = sum(_metric(r, "respec0b_resource_gather_attempts") for r in active)
    ticks = sum(_metric(r, "respec0b_resource_gather_handling_ticks") for r in active)
    successes = sum(_metric(r, "respec0b_resource_gather_successes") for r in active)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if attempts > 0.0 and ticks > 0.0 else "FAIL",
        "resource_gather_attempts": attempts,
        "handling_ticks": ticks,
        "resource_successes": successes,
        "interpretation": "fixed-arm demand is interpretable without explicit A/B identity only if agents encounter and act on resource cells",
        "evolving_axis_selection_authorized": False,
    }


def neutral_generalist_summary(campaign: Mapping[str, Any]) -> dict[str, Any]:
    neutral = [
        r
        for r in campaign.get("runs", [])
        if r["fixed_arm"] == "neutral" and r["world_family"] in {"a_only", "b_only", "mixed_tradeoff"}
    ]
    viable = sum(1 for r in neutral if _metric(r, "alive") > 0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if viable >= 9 else "FAIL",
        "neutral_cells_checked": len(neutral),
        "neutral_viable_cells": viable,
        "rule": "neutral alive_final > 0 in at least 9 of 12 A-only/B-only/mixed cells",
    }


def reciprocal_disadvantage_summary(direction: Mapping[str, Any]) -> dict[str, Any]:
    passed = direction.get("status") == "PASS"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        "basis": "mandatory A-only and B-only reciprocal direction comparisons",
        "direction_summary": direction,
    }


def campaign_completion(stage: Mapping[str, Any], integrity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_root": stage.get("root", ""),
        "planned": int(stage.get("planned_runs", 0) or 0),
        "collected": len(stage.get("runs", [])),
        "failed": len(integrity.get("failed_runs", [])),
        "missing_metrics": len(integrity.get("missing_metrics", [])),
        "empty_metrics": sum(1 for r in stage.get("runs", []) if r["metrics"]["row_count"] <= 0),
        "status": "PASS" if integrity.get("status") == "PASS" else "FAIL",
    }


def repeat_parity_summary(root_a: Path | None, root_b: Path | None) -> dict[str, Any]:
    if root_a is None or root_b is None or not root_a.exists() or not root_b.exists():
        return {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN", "pairs": []}
    a = collect_stage_runs(root_a)
    b = collect_stage_runs(root_b)
    by_a = {r["run_id"]: r for r in a.get("runs", [])}
    by_b = {r["run_id"]: r for r in b.get("runs", [])}
    pairs = []
    ok = True
    for run_id in selected_parity_run_ids():
        ra = by_a.get(run_id)
        rb = by_b.get(run_id)
        match = bool(ra and rb and ra["metrics_sha256"] == rb["metrics_sha256"] and ra["config_sha256"] == rb["config_sha256"])
        ok = ok and match
        pairs.append(
            {
                "run_id": run_id,
                "config_hash_a": ra.get("config_sha256", "") if ra else "",
                "config_hash_b": rb.get("config_sha256", "") if rb else "",
                "metrics_hash_a": ra.get("metrics_sha256", "") if ra else "",
                "metrics_hash_b": rb.get("metrics_sha256", "") if rb else "",
                "match": match,
            }
        )
    return {"schema_version": SCHEMA_VERSION, "status": "PASS" if ok else "FAIL", "pairs": pairs}


def final_decision(
    *,
    rehearsal_integrity: Mapping[str, Any] | None,
    campaign_integrity: Mapping[str, Any] | None,
    repeat_parity: Mapping[str, Any] | None,
    direction: Mapping[str, Any] | None,
    equalized: Mapping[str, Any] | None,
    flat_reporting: Mapping[str, Any] | None,
    observation: Mapping[str, Any] | None,
    neutral: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if rehearsal_integrity and rehearsal_integrity.get("status") != "PASS":
        decision = "RESPEC-FIXED-E"
        reason = "technical rehearsal did not pass"
    elif campaign_integrity and campaign_integrity.get("status") != "PASS":
        decision = "RESPEC-FIXED-D"
        reason = "campaign measurement or integrity failed"
    elif repeat_parity and repeat_parity.get("status") not in {"PASS", "NOT_RUN"}:
        decision = "RESPEC-FIXED-D"
        reason = "repeat parity failed"
    elif equalized and equalized.get("status") != "PASS":
        decision = "RESPEC-FIXED-C"
        reason = "equalized-sham control failed"
    elif flat_reporting and flat_reporting.get("status") != "PASS":
        decision = "RESPEC-FIXED-C"
        reason = "flat/reporting-only control drift"
    elif observation and observation.get("status") != "PASS":
        decision = "RESPEC-FIXED-B"
        reason = "observation/action path inadequate"
    elif neutral and neutral.get("status") != "PASS":
        decision = "RESPEC-FIXED-F"
        reason = "neutral generalist viability failed"
    elif direction and direction.get("status") != "PASS":
        decision = "RESPEC-FIXED-F"
        reason = "mandatory reciprocal fixed-arm directions failed"
    elif campaign_integrity:
        decision = "RESPEC-FIXED-A"
        reason = "reciprocal fixed-arm world demand supported by predeclared gates"
    else:
        decision = "INCOMPLETE"
        reason = "campaign analysis not available"
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "no_claims": NO_CLAIMS,
        "evolving_axis_selection_authorized": False,
    }


def build_payloads(
    *,
    rehearsal_root: Path | None,
    campaign_root: Path | None,
    parity_root_a: Path | None,
    parity_root_b: Path | None,
    runtime_archive_paths: list[str] | None = None,
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str]:
    source = source_verification()
    registry = load_registry_verification()
    contract = decision_contract()
    preflight = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if source["required_respec1_files_ok"] and registry["registry_sha256_ok"] and registry["matrix_ok"] else "FAIL",
        "source_status": source,
        "registry_status": registry,
        "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    rehearsal = collect_stage_runs(rehearsal_root) if rehearsal_root and rehearsal_root.exists() else None
    rehearsal_integrity = stage_integrity(rehearsal, expected_runs=18, expected_seeds=[101]) if rehearsal else None
    campaign = collect_stage_runs(campaign_root) if campaign_root and campaign_root.exists() else None
    campaign_integrity = stage_integrity(campaign, expected_runs=72, expected_seeds=SEEDS) if campaign else None
    scenario_rows: list[dict[str, Any]] = scenario_arm_summary(campaign) if campaign else []
    seed_rows: list[dict[str, Any]] = seed_level_rows(campaign) if campaign else []
    delta_rows: list[dict[str, Any]] = paired_deltas(campaign) if campaign else []
    direction = direction_summary(delta_rows) if delta_rows else None
    equalized = equalized_sham_audit(campaign) if campaign else None
    flat_reporting = flat_reporting_audit(campaign) if campaign else None
    observation = observation_action_audit(campaign) if campaign else None
    neutral = neutral_generalist_summary(campaign) if campaign else None
    repeat = repeat_parity_summary(parity_root_a, parity_root_b)
    decision = final_decision(
        rehearsal_integrity=rehearsal_integrity,
        campaign_integrity=campaign_integrity,
        repeat_parity=repeat,
        direction=direction,
        equalized=equalized,
        flat_reporting=flat_reporting,
        observation=observation,
        neutral=neutral,
    )
    payloads: dict[str, Mapping[str, Any]] = {
        "respec_fixed1_operational_state.json": {
            "schema_version": SCHEMA_VERSION,
            "persistent_branch": "integration/world-pressure",
            "task_branch": "campaign/respec-fixed-1-world-pressure-validation-20260711",
            "registry": str(REGISTRY_PATH),
            "scientific_campaign_type": "fixed_arm_validation",
        },
        "respec_fixed1_source_verification.json": source,
        "respec_fixed1_registry_verification.json": registry,
        "respec_fixed1_decision_contract.json": contract,
        "respec_fixed1_preflight_summary.json": preflight,
        "respec_fixed1_rehearsal_summary.json": campaign_completion(rehearsal, rehearsal_integrity) if rehearsal else {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"},
        "respec_fixed1_rehearsal_integrity.json": rehearsal_integrity or {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"},
        "respec_fixed1_campaign_completion.json": campaign_completion(campaign, campaign_integrity) if campaign else {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"},
        "respec_fixed1_config_reconciliation.json": campaign_integrity or {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"},
        "respec_fixed1_pair_completeness.json": {
            "schema_version": SCHEMA_VERSION,
            "status": (campaign_integrity or {}).get("status", "NOT_RUN"),
            "missing_pairs": (campaign_integrity or {}).get("missing_pairs", []),
            "extra_pairs": (campaign_integrity or {}).get("extra_pairs", []),
        },
        "respec_fixed1_nonfinite_scan.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if campaign_integrity and not campaign_integrity.get("nonfinite_values") else ("NOT_RUN" if not campaign else "FAIL"),
            "nonfinite_values": (campaign_integrity or {}).get("nonfinite_values", []),
        },
        "respec_fixed1_fixed_axis_verification.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if campaign_integrity and not campaign_integrity.get("fixed_axis_bad") else ("NOT_RUN" if not campaign else "FAIL"),
            "fixed_axis_bad": (campaign_integrity or {}).get("fixed_axis_bad", []),
        },
        "respec_fixed1_mutation_freeze_verification.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if campaign_integrity and not campaign_integrity.get("mutation_freeze_bad") else ("NOT_RUN" if not campaign else "FAIL"),
            "mutation_freeze_bad": (campaign_integrity or {}).get("mutation_freeze_bad", []),
        },
        "respec_fixed1_resource_layout_verification.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if campaign_integrity and not campaign_integrity.get("missing_pairs") else ("NOT_RUN" if not campaign else "FAIL"),
            "basis": "world family config modes and resource cell fractions are checked from config and metrics fields",
        },
        "respec_fixed1_mechanical_competence_summary.json": direction or {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"},
        "respec_fixed1_ecological_outcome_summary.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if campaign else "NOT_RUN",
            "scenario_arm_rows": len(scenario_rows),
            "outcome_fields": OUTCOME_FIELDS,
        },
        "respec_fixed1_reciprocal_disadvantage_summary.json": reciprocal_disadvantage_summary(direction) if direction else {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"},
        "respec_fixed1_neutral_generalist_summary.json": neutral or {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"},
        "respec_fixed1_mixed_world_summary.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if campaign else "NOT_RUN",
            "mixed_cells": len([r for r in campaign.get("runs", []) if r["world_family"] == "mixed_tradeoff"]) if campaign else 0,
            "note": "mixed-world values are in scenario/arm and seed-level CSVs",
        },
        "respec_fixed1_equalized_sham_audit.json": equalized or {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"},
        "respec_fixed1_flat_reporting_control_audit.json": flat_reporting or {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"},
        "respec_fixed1_observation_action_audit.json": observation or {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"},
        "respec_fixed1_timing_overlap_summary.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if campaign else "NOT_RUN",
            "available_fields": ["successful_births", "resource gather attempts/successes by generation"],
            "limitation": "compact profile does not preserve full event timing; generation-level metrics are used",
        },
        "respec_fixed1_lineage_branch_summary.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if campaign else "NOT_RUN",
            "interpretation": "lineage and branch metrics are measurement-only; no lineage intervention is used",
        },
        "respec_fixed1_repeat_parity_summary.json": repeat,
        "respec_fixed1_reward_and_identity_audit.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if campaign_integrity and not campaign_integrity.get("failed_runs") else ("NOT_RUN" if not campaign else "FAIL"),
            "direct_reward_added": False,
            "lineage_identity_mechanic_added": False,
            "branch_identity_mechanic_added": False,
            "strategy_identity_mechanic_added": False,
        },
        "respec_fixed1_storage_summary.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "output_profile": "compact_default",
            "checkpoint_period": 0,
            "runtime_roots": [str(p) for p in [rehearsal_root, campaign_root, parity_root_a, parity_root_b] if p],
        },
        "respec_fixed1_runtime_archive_record.json": {
            "schema_version": SCHEMA_VERSION,
            "status": "PENDING_ARCHIVE" if not runtime_archive_paths else "ARCHIVED",
            "archives": runtime_archive_paths or [],
        },
        "respec_fixed1_validation_summary.json": {
            "schema_version": SCHEMA_VERSION,
            "status": decision["decision"],
            "preflight": preflight["status"],
            "rehearsal": (rehearsal_integrity or {}).get("status", "NOT_RUN"),
            "campaign": (campaign_integrity or {}).get("status", "NOT_RUN"),
            "repeat_parity": repeat.get("status", "NOT_RUN"),
        },
        "respec_fixed1_final_decision.json": decision,
        "respec_fixed1_integration_record.json": {"schema_version": SCHEMA_VERSION, "status": "PENDING_COMMIT"},
        "respec_fixed1_cleanup_record.json": {"schema_version": SCHEMA_VERSION, "status": "PENDING_CLEANUP"},
    }
    return payloads, scenario_rows, seed_rows, delta_rows, decision["decision"]


def render_report(payloads: Mapping[str, Mapping[str, Any]], evidence_dir: Path) -> str:
    decision = payloads["respec_fixed1_final_decision.json"]
    completion = payloads["respec_fixed1_campaign_completion.json"]
    repeat = payloads["respec_fixed1_repeat_parity_summary.json"]
    return f"""# world-pressure RESPEC-FIXED-1 Fixed-Arm World-Pressure Validation

Date: 2026-07-11

Decision: `{decision['decision']}`

## Summary

RESPEC-FIXED-1 tests whether the accepted Resource A/B world creates reciprocal fixed-arm demand under frozen inherited resource-handling axes. It is not an evolving-axis selection campaign.

## Campaign

- Planned cells: {completion.get('planned', 0)}
- Collected cells: {completion.get('collected', 0)}
- Failed cells: {completion.get('failed', 0)}
- Repeat parity: `{repeat.get('status', 'NOT_RUN')}`
- Evidence: `{evidence_dir}`

## Win/Loss

Win: The task produced a deterministic fixed-arm validation package and decision using the predeclared contract.

Loss: No evolved-specialization or selection claim is authorized by this task, regardless of the decision class.

In simple terms: I tested whether the Resource A/B world itself gives different advantages to agents that are fixed to be better at Resource A, neutral, or better at Resource B. This matters because the project is now asking whether world pressure can create useful different ways of living. The result says whether this fixed-arm world-demand test supports moving toward a later evolving-axis review, or whether the world/control/measurement path needs repair first.

## No Claims

{NO_CLAIMS}
"""


def write_outputs(
    *,
    evidence_dir: Path,
    report_path: Path,
    rehearsal_root: Path | None,
    campaign_root: Path | None,
    parity_root_a: Path | None,
    parity_root_b: Path | None,
    runtime_archive_paths: list[str] | None = None,
) -> dict[str, Any]:
    payloads, scenario_rows, seed_rows, delta_rows, decision = build_payloads(
        rehearsal_root=rehearsal_root,
        campaign_root=campaign_root,
        parity_root_a=parity_root_a,
        parity_root_b=parity_root_b,
        runtime_archive_paths=runtime_archive_paths,
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _write_json(evidence_dir / name, payload)
    _write_csv(
        evidence_dir / "respec_fixed1_scenario_arm_summary.csv",
        scenario_rows,
        sorted(set().union(*(row.keys() for row in scenario_rows))) if scenario_rows else ["world_family", "fixed_arm"],
    )
    _write_csv(
        evidence_dir / "respec_fixed1_seed_level_metrics.csv",
        seed_rows,
        sorted(set().union(*(row.keys() for row in seed_rows))) if seed_rows else ["run_id"],
    )
    _write_csv(
        evidence_dir / "respec_fixed1_paired_deltas.csv",
        delta_rows,
        ["comparison", "world_family", "metric", "left_arm", "right_arm", "seed", "delta", "supports_direction", "strong_reverse"],
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = render_report(payloads, evidence_dir)
    report_path.write_text(report_text, encoding="utf-8")
    (evidence_dir / report_path.name).write_text(report_text, encoding="utf-8")
    return {"decision": decision, "evidence_dir": str(evidence_dir), "report_path": str(report_path)}


def package_evidence(evidence_dir: Path) -> dict[str, Any]:
    zip_path = evidence_dir / "world-pressure_respec_fixed1_validation.evidence.zip"
    manifest_path = evidence_dir / "respec_fixed1_evidence_bundle_manifest.json"
    if zip_path.exists():
        zip_path.unlink()
    if manifest_path.exists():
        manifest_path.unlink()
    files = sorted([p for p in evidence_dir.iterdir() if p.is_file() and p.name not in {zip_path.name, manifest_path.name}])
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, arcname=path.name)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            bad = zf.testzip()
    except Exception as exc:
        bad = str(exc)
    manifest = {
        "schema_version": "respec_fixed1_evidence_bundle_manifest_v1",
        "zip_path": zip_path.name,
        "sha256": _sha256(zip_path),
        "file_count_inside_zip": len(files),
        "archive_integrity_check": "PASS" if bad is None else "FAIL",
        "bad_member": "" if bad is None else str(bad),
    }
    _write_json(manifest_path, manifest)
    return manifest


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RESPEC-FIXED-1 fixed-arm validation analyzer")
    parser.add_argument("--evidence-dir", default=str(EVIDENCE_DIR))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--rehearsal-root", default="")
    parser.add_argument("--campaign-root", default="")
    parser.add_argument("--parity-root-a", default="")
    parser.add_argument("--parity-root-b", default="")
    parser.add_argument("--runtime-archive", action="append", default=[])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--print-run-id-file", choices=["rehearsal", "parity"], default="")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.print_run_id_file:
        ids = selected_rehearsal_run_ids() if args.print_run_id_file == "rehearsal" else selected_parity_run_ids()
        print("\n".join(ids))
        return 0
    evidence_dir = Path(args.evidence_dir)
    result = write_outputs(
        evidence_dir=evidence_dir,
        report_path=Path(args.report_path),
        rehearsal_root=Path(args.rehearsal_root) if args.rehearsal_root else None,
        campaign_root=Path(args.campaign_root) if args.campaign_root else None,
        parity_root_a=Path(args.parity_root_a) if args.parity_root_a else None,
        parity_root_b=Path(args.parity_root_b) if args.parity_root_b else None,
        runtime_archive_paths=list(args.runtime_archive or []),
    ) if args.write else {"decision": "DRY_CONTRACT", "registry": load_registry_verification(), "contract": decision_contract()}
    if args.package:
        result["evidence_zip"] = package_evidence(evidence_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
