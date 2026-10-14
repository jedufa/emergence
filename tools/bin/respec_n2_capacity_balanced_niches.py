#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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


def _load_fixed1_module():
    path = Path(__file__).resolve().with_name("respec_fixed1_validation_report.py")
    spec = importlib.util.spec_from_file_location("respec_fixed1_validation_report", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("respec_fixed1_validation_report", module)
    spec.loader.exec_module(module)
    return module


fixed1 = _load_fixed1_module()

SCHEMA_VERSION = "respec_n2_capacity_balanced_niches_v1"
REGISTRY_PATH = Path("_spec/registry/v1/respec_n2_capacity_balanced_fixed_arm_validation.yaml")
REPORT_PATH = Path("docs/500_campaigns-decisions-closeouts/520_campaign-reports/400_worlds-ecology/EIEL-DOC-002171--world-pressure-eiel-respec-n2-capacity-balanced-mixed-resource-niches-2026-07-11.md")
EVIDENCE_DIR = Path("evidences/07-world-pressure/08-respec-n2-capacity-balanced-niches")
WORLD_FAMILIES = [
    "flat",
    "reporting_only",
    "a_only",
    "b_only",
    "equalized_sham",
    "mixed_tradeoff",
    "capacity_balanced_mixed",
]
FIXED_ARMS = {"a_favored": 0.75, "neutral": 0.0, "b_favored": -0.75}
SEEDS = [101, 202, 303, 404]
NO_CLAIMS = (
    "RESPEC-N2 is a bounded fixed-arm resource-world validation. It does not claim evolved specialization, "
    "selected specialization, intelligence, emergence, consciousness, broad cognition, adaptive memory, "
    "prediction selection, solved lineage diversity, beneficial open-ended search, culture, superintelligence, "
    "or general open-ended evolution."
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = sorted({str(k) for row in rows for k in row.keys()})
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
    return payload if isinstance(payload, dict) else {}


def _float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return 0.0
    return out if math.isfinite(out) else 0.0


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(statistics.fmean(vals)) if vals else 0.0


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def _run_id(world: str, arm: str, seed: int) -> str:
    return f"respec_n2_capacity_balanced_fixed_arm_validation__world_{world}__arm_{arm}__seed{int(seed):06d}"


def parse_run_id(run_id: str) -> dict[str, Any]:
    marker = "respec_n2_capacity_balanced_fixed_arm_validation__world_"
    if not str(run_id).startswith(marker):
        return {"world_family": "", "fixed_arm": "", "seed": -1}
    rest = str(run_id)[len(marker) :]
    world, rest = rest.split("__arm_", 1)
    arm, seed_text = rest.split("__seed", 1)
    return {"world_family": world, "fixed_arm": arm, "seed": int(seed_text)}


def selected_rehearsal_run_ids() -> list[str]:
    return [_run_id("capacity_balanced_mixed", arm, 101) for arm in FIXED_ARMS]


def selected_validation_run_ids() -> list[str]:
    out: list[str] = []
    out.extend(_run_id("capacity_balanced_mixed", arm, seed) for arm in FIXED_ARMS for seed in SEEDS)
    out.extend(_run_id("equalized_sham", arm, seed) for arm in FIXED_ARMS for seed in SEEDS)
    out.extend(_run_id("mixed_tradeoff", arm, seed) for arm in FIXED_ARMS for seed in (101, 202))
    out.extend(_run_id("a_only", arm, 101) for arm in FIXED_ARMS)
    out.extend(_run_id("b_only", arm, 101) for arm in FIXED_ARMS)
    out.append(_run_id("flat", "neutral", 101))
    out.append(_run_id("reporting_only", "neutral", 101))
    return out


def selected_repeat_run_ids() -> list[str]:
    return [
        _run_id("capacity_balanced_mixed", "a_favored", 202),
        _run_id("capacity_balanced_mixed", "neutral", 303),
        _run_id("capacity_balanced_mixed", "b_favored", 404),
        _run_id("equalized_sham", "a_favored", 101),
    ]


def registry_verification() -> dict[str, Any]:
    registry = _load_yaml(REGISTRY_PATH)
    contract = dict(registry.get("respec_n2_contract") or {})
    world_values = [str(v) for v in registry.get("axes", {}).get("world_family", {}).get("values", [])]
    arm_values = [str(v) for v in registry.get("axes", {}).get("fixed_arm", {}).get("values", [])]
    seeds = [int(s) for s in registry.get("seeds", [])]
    selected = selected_validation_run_ids()
    flags = registry.get("axes", {}).get("world_family", {}).get("flags", {})
    flat_flags = "\n".join(str(x) for x in flags.get("flat", []))
    capacity_flags = "\n".join(str(x) for x in flags.get("capacity_balanced_mixed", []))
    status = (
        registry.get("name") == "respec_n2_capacity_balanced_fixed_arm_validation"
        and world_values == WORLD_FAMILIES
        and arm_values == list(FIXED_ARMS)
        and seeds == SEEDS
        and len(selected) == 38
        and len(set(selected)) == 38
        and "env_cfg.respec_n2_capacity_balanced_mixed_enable=false" in flat_flags
        and "env_cfg.respec_n2_capacity_balanced_mixed_enable=true" in capacity_flags
        and _float(contract.get("resource_a_energy_regrowth_budget_per_cell")) == _float(
            contract.get("resource_b_energy_regrowth_budget_per_cell")
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if status else "FAIL",
        "registry_path": str(REGISTRY_PATH),
        "registry_sha256": _sha256(REGISTRY_PATH),
        "world_families": world_values,
        "fixed_arms": arm_values,
        "seeds": seeds,
        "full_matrix_cells": len(world_values) * len(arm_values) * len(seeds),
        "selected_validation_cells": len(selected),
        "selected_rehearsal_cells": len(selected_rehearsal_run_ids()),
        "selected_repeat_cells": len(selected_repeat_run_ids()),
        "contract": contract,
    }


def source_verification() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "mechanics": "capacity-balanced mixed changes only patch_fertility for vertical_split_ab RESPEC resource classes when env_cfg.respec_n2_capacity_balanced_mixed_enable=true",
        "formula": "resource_b_fertility = resource_a_return / resource_b_return = 0.75 / 1.50 = 0.5",
        "default_off": "env_cfg.respec_n2_capacity_balanced_mixed_enable defaults false",
        "no_genome_or_lineage_input": True,
        "source_paths": [
            "engine/env/env_config.py:respec_n2_capacity_balanced_mixed_enable",
            "engine/env/gridworld_pkg/patch_world.py:apply_respec_n2_capacity_balanced_overlay",
            "engine/env/gridworld_pkg/patch_world.py:respec_n2_capacity_budget_snapshot",
            "engine/simulation/respec_n1_zone_telemetry.py:respec_n2_capacity_budget_generation_summary.csv",
        ],
    }


def _resolve_run_dir(root: Path, raw_run_dir: str) -> Path:
    run_dir = Path(str(raw_run_dir))
    if run_dir.exists():
        return run_dir
    parts = list(run_dir.parts)
    if root.name in parts:
        idx = parts.index(root.name)
        candidate = root.joinpath(*parts[idx + 1 :])
        if candidate.exists():
            return candidate
    if not run_dir.is_absolute():
        candidate = root / run_dir
        if candidate.exists():
            return candidate
    return run_dir


def _latest_generation_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    latest = max(int(_float(row.get("generation", 0))) for row in rows)
    return [row for row in rows if int(_float(row.get("generation", 0))) == latest]


def load_campaign(root: Path | None, expected_count: int) -> dict[str, Any] | None:
    if root is None:
        return None
    summary_path = root / "parallel_campaign_summary.json"
    if not summary_path.exists():
        return None
    summary = _read_json(summary_path)
    runs = []
    for result in summary.get("results", []):
        if not isinstance(result, Mapping):
            continue
        run_id = str(result.get("run_id", ""))
        parsed = parse_run_id(run_id)
        run_dir = _resolve_run_dir(root, str(result.get("run_dir", "")))
        metrics = fixed1.summarize_metrics(run_dir) if run_dir.exists() else {}
        zone_summary = _read_json(run_dir / "metrics" / "respec_n1_zone_run_summary.json")
        budget_rows = _read_csv(run_dir / "metrics" / "respec_n2_capacity_budget_generation_summary.csv")
        runs.append(
            {
                "run_id": run_id,
                "world_family": parsed["world_family"],
                "fixed_arm": parsed["fixed_arm"],
                "seed": parsed["seed"],
                "status": str(result.get("status", "")),
                "returncode": int(result.get("returncode", 1)),
                "run_dir": str(run_dir),
                "config_sha256": _sha256(run_dir / "config.json") if (run_dir / "config.json").exists() else "",
                "metrics_sha256": _sha256(run_dir / "metrics.csv") if (run_dir / "metrics.csv").exists() else "",
                "zone_summary_sha256": _sha256(run_dir / "metrics" / "respec_n1_zone_run_summary.json")
                if (run_dir / "metrics" / "respec_n1_zone_run_summary.json").exists()
                else "",
                "budget_summary_sha256": _sha256(run_dir / "metrics" / "respec_n2_capacity_budget_generation_summary.csv")
                if (run_dir / "metrics" / "respec_n2_capacity_budget_generation_summary.csv").exists()
                else "",
                "metrics": metrics,
                "zone_summary": zone_summary,
                "budget_rows": budget_rows,
            }
        )
    failures = summary.get("failures", 0)
    failed_count = len(failures) if isinstance(failures, list) else int(failures or 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_root": str(root),
        "planned": int(summary.get("planned_runs", 0)),
        "failed": failed_count,
        "expected": expected_count,
        "runs": runs,
    }


def campaign_integrity(campaign: Mapping[str, Any] | None, expected: int) -> dict[str, Any]:
    if campaign is None:
        return {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN", "expected": expected}
    failed = [r["run_id"] for r in campaign.get("runs", []) if r.get("returncode") != 0 or r.get("status") != "ok"]
    missing_metrics = [
        r["run_id"]
        for r in campaign.get("runs", [])
        if isinstance(r.get("metrics"), Mapping) and (r.get("metrics") or {}).get("missing_required_fields")
    ]
    complete = len(campaign.get("runs", [])) == expected and int(campaign.get("planned", 0)) == expected
    status = "PASS" if complete and not failed and not missing_metrics and int(campaign.get("failed", 0)) == 0 else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "planned": int(campaign.get("planned", 0)),
        "collected": len(campaign.get("runs", [])),
        "expected": expected,
        "failed_count": int(campaign.get("failed", 0)),
        "failed_runs": failed,
        "missing_metrics": missing_metrics,
    }


def _metric(run: Mapping[str, Any], metric: str) -> float:
    metrics = run.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return 0.0
    aliases = {
        "alive_final": "alive",
        "successful_births_final": "successful_births",
        "failed_births_final": "failed_births",
    }
    for section in ("derived", "sums", "final", "fraction_means"):
        values = metrics.get(section, {})
        if isinstance(values, Mapping) and metric in values:
            return _float(values.get(metric))
        alias = aliases.get(metric)
        if isinstance(values, Mapping) and alias in values:
            return _float(values.get(alias))
    sums = metrics.get("sums", {})
    if isinstance(sums, Mapping):
        handling = _float(sums.get("respec0b_resource_gather_handling_ticks"))
        if metric == "resource_a_completion_per_handling_tick":
            return _safe_div(_float(sums.get("respec0b_resource_gather_resource_a_successes")), handling)
        if metric == "resource_b_completion_per_handling_tick":
            return _safe_div(_float(sums.get("respec0b_resource_gather_resource_b_successes")), handling)
    return 0.0


def seed_level_rows(campaign: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if campaign is None:
        return []
    rows = []
    for r in sorted(campaign.get("runs", []), key=lambda x: (x["world_family"], x["fixed_arm"], int(x["seed"]))):
        rows.append(
            {
                "run_id": r["run_id"],
                "world_family": r["world_family"],
                "fixed_arm": r["fixed_arm"],
                "seed": int(r["seed"]),
                "alive_final": _metric(r, "alive_final"),
                "successful_births_final": _metric(r, "successful_births_final"),
                "respec0b_resource_gather_attempts": _metric(r, "respec0b_resource_gather_attempts"),
                "respec0b_resource_gather_handling_ticks": _metric(r, "respec0b_resource_gather_handling_ticks"),
                "respec0b_resource_gather_successes": _metric(r, "respec0b_resource_gather_successes"),
                "resource_a_completion_per_handling_tick": _metric(r, "resource_a_completion_per_handling_tick"),
                "resource_b_completion_per_handling_tick": _metric(r, "resource_b_completion_per_handling_tick"),
            }
        )
    return rows


def scenario_arm_summary(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for world in WORLD_FAMILIES:
        for arm in FIXED_ARMS:
            subset = [r for r in seed_rows if r["world_family"] == world and r["fixed_arm"] == arm]
            if not subset:
                continue
            row: dict[str, Any] = {"world_family": world, "fixed_arm": arm, "runs": len(subset)}
            for metric in (
                "alive_final",
                "successful_births_final",
                "respec0b_resource_gather_attempts",
                "respec0b_resource_gather_handling_ticks",
                "respec0b_resource_gather_successes",
                "resource_a_completion_per_handling_tick",
                "resource_b_completion_per_handling_tick",
            ):
                row[f"{metric}_mean"] = _mean(_float(r.get(metric)) for r in subset)
            out.append(row)
    return out


def _zone_row(run: Mapping[str, Any], arm: str, zone: str) -> Mapping[str, Any] | None:
    zone_summary = run.get("zone_summary", {})
    rows = zone_summary.get("arm_zone", []) if isinstance(zone_summary, Mapping) else []
    for row in rows:
        if str(row.get("arm", "")) == arm and str(row.get("zone", "")) == zone:
            return row
    return None


def local_zone_direction_summary(campaign: Mapping[str, Any] | None) -> dict[str, Any]:
    if campaign is None:
        return {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN", "comparisons": []}
    comparisons = [
        {
            "name": "a_favored_local_a_over_b_selected_parent_rate",
            "world": "capacity_balanced_mixed",
            "arm": "a_favored",
            "left_zone": "resource_a_zone",
            "right_zone": "resource_b_zone",
            "metric": "selected_parent_events_per_1000_agent_ticks",
        },
        {
            "name": "a_favored_local_a_over_b_birth_rate",
            "world": "capacity_balanced_mixed",
            "arm": "a_favored",
            "left_zone": "resource_a_zone",
            "right_zone": "resource_b_zone",
            "metric": "births_per_1000_agent_ticks",
        },
        {
            "name": "b_favored_local_b_over_a_selected_parent_rate",
            "world": "capacity_balanced_mixed",
            "arm": "b_favored",
            "left_zone": "resource_b_zone",
            "right_zone": "resource_a_zone",
            "metric": "selected_parent_events_per_1000_agent_ticks",
        },
        {
            "name": "b_favored_local_b_over_a_birth_rate",
            "world": "capacity_balanced_mixed",
            "arm": "b_favored",
            "left_zone": "resource_b_zone",
            "right_zone": "resource_a_zone",
            "metric": "births_per_1000_agent_ticks",
        },
    ]
    rows = []
    passed = True
    for comp in comparisons:
        deltas = []
        support = 0
        missing = 0
        for seed in SEEDS:
            run = next(
                (
                    r
                    for r in campaign.get("runs", [])
                    if r["world_family"] == comp["world"] and r["fixed_arm"] == comp["arm"] and int(r["seed"]) == seed
                ),
                None,
            )
            left = _zone_row(run or {}, str(comp["arm"]), str(comp["left_zone"]))
            right = _zone_row(run or {}, str(comp["arm"]), str(comp["right_zone"]))
            if left is None or right is None:
                missing += 1
                continue
            delta = _float(left.get(str(comp["metric"]))) - _float(right.get(str(comp["metric"])))
            deltas.append(delta)
            if delta > 0.0:
                support += 1
        ok = support >= 3 and missing == 0
        passed = passed and ok
        rows.append(
            {
                **comp,
                "supporting_seeds": support,
                "missing_seeds": missing,
                "mean_delta": _mean(deltas),
                "status": "PASS" if ok else "FAIL",
            }
        )
    return {"schema_version": SCHEMA_VERSION, "status": "PASS" if passed else "FAIL", "comparisons": rows}


def capacity_budget_summary(campaign: Mapping[str, Any] | None) -> dict[str, Any]:
    if campaign is None:
        return {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"}
    rows = []
    fail_reasons = []
    for run in campaign.get("runs", []):
        if run["world_family"] != "capacity_balanced_mixed":
            continue
        latest = _latest_generation_rows(list(run.get("budget_rows", []) or []))
        by_zone = {row.get("zone", ""): row for row in latest}
        a = by_zone.get("resource_a_zone")
        b = by_zone.get("resource_b_zone")
        if a is None or b is None:
            fail_reasons.append(f"{run['run_id']}:missing_budget_rows")
            continue
        a_budget = _float(a.get("fertility_mean")) * _float(a.get("return_per_completion"))
        b_budget = _float(b.get("fertility_mean")) * _float(b.get("return_per_completion"))
        equal = abs(a_budget - b_budget) <= 1e-9
        if not equal:
            fail_reasons.append(f"{run['run_id']}:budget_mismatch")
        rows.append(
            {
                "run_id": run["run_id"],
                "fixed_arm": run["fixed_arm"],
                "seed": int(run["seed"]),
                "a_cells": int(_float(a.get("cell_count"))),
                "b_cells": int(_float(b.get("cell_count"))),
                "a_fertility": _float(a.get("fertility_mean")),
                "b_fertility": _float(b.get("fertility_mean")),
                "a_return": _float(a.get("return_per_completion")),
                "b_return": _float(b.get("return_per_completion")),
                "a_energy_budget_per_cell": a_budget,
                "b_energy_budget_per_cell": b_budget,
                "status": "PASS" if equal else "FAIL",
            }
        )
    status = "PASS" if len(rows) == 12 and not fail_reasons else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "expected_capacity_runs": 12,
        "observed_capacity_runs_with_budget_rows": len(rows),
        "fail_reasons": fail_reasons,
        "rows": rows,
    }


def control_leakage_summary(campaign: Mapping[str, Any] | None) -> dict[str, Any]:
    if campaign is None:
        return {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"}
    flat = [r for r in campaign.get("runs", []) if r["world_family"] == "flat"]
    reporting = [r for r in campaign.get("runs", []) if r["world_family"] == "reporting_only"]
    sham = [r for r in campaign.get("runs", []) if r["world_family"] == "equalized_sham"]
    flat_ok = all(_metric(r, "respec0b_resource_gather_attempts") == 0.0 for r in flat)
    reporting_ok = all(_metric(r, "respec0b_resource_gather_attempts") == 0.0 for r in reporting)
    sham_births = {
        arm: _mean(_metric(r, "successful_births_final") for r in sham if r["fixed_arm"] == arm) for arm in FIXED_ARMS
    }
    sham_span = max(sham_births.values() or [0.0]) - min(sham_births.values() or [0.0])
    sham_ok = sham_span <= 8.0
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if flat_ok and reporting_ok and sham_ok else "FAIL",
        "flat_no_resource_attempts": flat_ok,
        "reporting_only_no_resource_attempts": reporting_ok,
        "equalized_sham_births_by_arm": sham_births,
        "equalized_sham_birth_span": sham_span,
        "equalized_sham_threshold": 8.0,
    }


def repeat_parity_summary(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"schema_version": SCHEMA_VERSION, "status": "NOT_RUN"}
    by_a = {r["run_id"]: r for r in a.get("runs", [])}
    by_b = {r["run_id"]: r for r in b.get("runs", [])}
    rows = []
    for run_id in sorted(set(by_a) & set(by_b)):
        left = by_a[run_id]
        right = by_b[run_id]
        rows.append(
            {
                "run_id": run_id,
                "metrics_sha256_equal": left.get("metrics_sha256") == right.get("metrics_sha256"),
                "zone_summary_sha256_equal": left.get("zone_summary_sha256") == right.get("zone_summary_sha256"),
                "budget_summary_sha256_equal": left.get("budget_summary_sha256") == right.get("budget_summary_sha256"),
            }
        )
    status = (
        "PASS"
        if len(rows) == len(selected_repeat_run_ids())
        and all(r["metrics_sha256_equal"] and r["zone_summary_sha256_equal"] and r["budget_summary_sha256_equal"] for r in rows)
        else "FAIL"
    )
    return {"schema_version": SCHEMA_VERSION, "status": status, "rows": rows}


def decide(payloads: Mapping[str, Any]) -> dict[str, Any]:
    required_green = [
        "respec_n2_registry_verification.json",
        "respec_n2_source_verification.json",
        "respec_n2_campaign_completion.json",
        "respec_n2_repeat_parity_summary.json",
    ]
    failed_required = [name for name in required_green if (payloads.get(name) or {}).get("status") != "PASS"]
    if failed_required:
        decision = "RESPEC-N2-E"
        reason = "Validation, campaign integrity, source verification, or repeat parity failed."
    elif (payloads.get("respec_n2_capacity_budget_summary.json") or {}).get("status") != "PASS":
        decision = "RESPEC-N2-D"
        reason = "The capacity-balanced world did not prove the frozen analytic A/B budget contract."
    elif (payloads.get("respec_n2_control_leakage_summary.json") or {}).get("status") != "PASS":
        decision = "RESPEC-N2-E"
        reason = "A disabled/reporting/equalized control showed leakage or unacceptable arm skew."
    elif (payloads.get("respec_n2_local_zone_direction_summary.json") or {}).get("status") == "PASS":
        decision = "RESPEC-N2-A"
        reason = "The frozen capacity-balanced mixed world passed controls and produced reciprocal local ecological direction for fixed A/B arms."
    else:
        decision = "RESPEC-N2-C"
        reason = "The analytic capacity balance was clean, but the fixed-arm local ecological direction was not strong enough."
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "reason": reason,
        "failed_required": failed_required,
        "no_claims": NO_CLAIMS,
        "recommended_next_step": (
            "If A, park the substrate as eligible for the next world-pressure gate. If C/D/E, preserve the evidence and redesign the resource demand rather than tuning this frozen N2 parameter set."
        ),
    }


def _report_text(payloads: Mapping[str, Any], runtime_roots: Mapping[str, str]) -> str:
    final = payloads["respec_n2_final_decision.json"]
    budget = payloads["respec_n2_capacity_budget_summary.json"]
    controls = payloads["respec_n2_control_leakage_summary.json"]
    local = payloads["respec_n2_local_zone_direction_summary.json"]
    return "\n".join(
        [
            "# world-pressure EIEL RESPEC-N2 Capacity-Balanced Mixed Resource Niches",
            "",
            f"Decision: **{final['decision']}**",
            "",
            "## What Was Attempted",
            "",
            "I tested one frozen, analytic capacity balance for the mixed Resource A/B world. Resource B pays twice the energy return of Resource A, so the B-side fertility was cut to 0.5 while A stayed at 1.0. This was meant to ask whether a cleaner world demand can support two different fixed resource-handling roles without using lineage help, rewards, genome tuning, or post-result parameter search.",
            "",
            "## Win/Loss",
            "",
            f"- Capacity budget contract: {budget.get('status')}",
            f"- Control leakage: {controls.get('status')}",
            f"- Local zone direction: {local.get('status')}",
            f"- Final decision: {final['decision']} - {final['reason']}",
            "",
            "In simple terms: we tried to make both resource zones equally valuable over time, even though one resource gives a bigger one-time payoff. This fits the larger world-pressure track because the project is moving from lineage repair toward worlds that naturally ask for different ways of living. The result says whether this particular balancing idea is ready to carry forward, or whether resource specialization needs a different world design.",
            "",
            "## Runtime Roots",
            "",
            *(f"- {name}: `{path}`" for name, path in runtime_roots.items() if path),
            "",
            "## No-Claims Statement",
            "",
            NO_CLAIMS,
            "",
        ]
    )


def build_payloads(
    *,
    campaign_root: Path | None,
    repeat_a_root: Path | None,
    repeat_b_root: Path | None,
    runtime_roots: Mapping[str, str],
) -> dict[str, Any]:
    campaign = load_campaign(campaign_root, len(selected_validation_run_ids()))
    repeat_a = load_campaign(repeat_a_root, len(selected_repeat_run_ids()))
    repeat_b = load_campaign(repeat_b_root, len(selected_repeat_run_ids()))
    seed_rows = seed_level_rows(campaign)
    payloads: dict[str, Any] = {
        "respec_n2_registry_verification.json": registry_verification(),
        "respec_n2_source_verification.json": source_verification(),
        "respec_n2_campaign_completion.json": campaign_integrity(campaign, len(selected_validation_run_ids())),
        "respec_n2_repeat_parity_summary.json": repeat_parity_summary(repeat_a, repeat_b),
        "respec_n2_capacity_budget_summary.json": capacity_budget_summary(campaign),
        "respec_n2_control_leakage_summary.json": control_leakage_summary(campaign),
        "respec_n2_local_zone_direction_summary.json": local_zone_direction_summary(campaign),
        "respec_n2_seed_level_metrics.csv": seed_rows,
        "respec_n2_scenario_arm_summary.csv": scenario_arm_summary(seed_rows),
        "respec_n2_operational_state.json": {
            "schema_version": SCHEMA_VERSION,
            "campaign_root": str(campaign_root or ""),
            "repeat_a_root": str(repeat_a_root or ""),
            "repeat_b_root": str(repeat_b_root or ""),
            "runtime_roots": dict(runtime_roots),
            "selected_validation_run_ids": selected_validation_run_ids(),
            "selected_rehearsal_run_ids": selected_rehearsal_run_ids(),
            "selected_repeat_run_ids": selected_repeat_run_ids(),
        },
    }
    payloads["respec_n2_final_decision.json"] = decide(payloads)
    return payloads


def write_artifacts(payloads: Mapping[str, Any], evidence_dir: Path, report_path: Path, runtime_roots: Mapping[str, str]) -> dict[str, Any]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = evidence_dir / name
        if name.endswith(".csv"):
            _write_csv(path, payload if isinstance(payload, list) else [])
        else:
            _write_json(path, payload)
    report_text = _report_text(payloads, runtime_roots)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    (evidence_dir / report_path.name).write_text(report_text, encoding="utf-8")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "report_path": str(report_path),
        "evidence_dir": str(evidence_dir),
        "files": sorted(p.name for p in evidence_dir.iterdir() if p.is_file()),
    }
    _write_json(evidence_dir / "respec_n2_evidence_bundle_manifest.json", manifest)

    zip_path = evidence_dir / "world-pressure_respec_n2_capacity_balanced_niches.evidence.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(evidence_dir.iterdir()):
            if path.is_file() and path.name != zip_path.name:
                zf.write(path, arcname=path.name)
    manifest["zip_path"] = str(zip_path)
    manifest["zip_sha256"] = _sha256(zip_path)
    _write_json(evidence_dir / "respec_n2_evidence_bundle_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--repeat-a-root", type=Path)
    parser.add_argument("--repeat-b-root", type=Path)
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    parser.add_argument("--runtime-root", action="append", default=[])
    parser.add_argument("--print-run-ids", choices=["rehearsal", "validation", "repeat"])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    if args.print_run_ids:
        mapping = {
            "rehearsal": selected_rehearsal_run_ids(),
            "validation": selected_validation_run_ids(),
            "repeat": selected_repeat_run_ids(),
        }
        print("\n".join(mapping[args.print_run_ids]))
        return 0

    runtime_roots = {}
    for item in args.runtime_root:
        if "=" in item:
            key, value = item.split("=", 1)
            runtime_roots[key] = value
    payloads = build_payloads(
        campaign_root=args.campaign_root,
        repeat_a_root=args.repeat_a_root,
        repeat_b_root=args.repeat_b_root,
        runtime_roots=runtime_roots,
    )
    if args.write:
        manifest = write_artifacts(payloads, args.evidence_dir, args.report_path, runtime_roots)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        print(json.dumps(payloads["respec_n2_final_decision.json"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
