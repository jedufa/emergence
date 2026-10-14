"""Campaign schema contract checker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.evidence.manifest import ManifestError, load_manifest, scenario_dimension_values
from tools.evidence.profiles import get_profile
from tools.evidence.runs import classify_runs, collect_harness_runs


def check_campaign_schema(manifest_path: str | Path) -> dict[str, Any]:
    checked_files: list[str] = []
    missing_files: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []

    try:
        evidence_manifest = load_manifest(manifest_path)
    except ManifestError as exc:
        return {
            "ok": False,
            "campaign_id": "",
            "profile": "",
            "errors": exc.errors,
            "warnings": [],
            "checked_files": [str(exc.path)],
            "missing_files": [],
            "run_count_observed": 0,
            "run_count_expected": 0,
            "scenario_count_observed": 0,
            "seed_count_observed": 0,
            "run_status": [],
            "anomaly_summary": {"counts": {}, "affected": {}},
        }

    manifest = evidence_manifest.data
    checked_files.append(str(evidence_manifest.path))
    for path_field in ("registry_path",):
        path = Path(str(manifest.get(path_field) or ""))
        if path.exists():
            checked_files.append(str(path))
        else:
            missing_files.append(str(path))
            warnings.append(f"{path_field} does not exist: {path}")
    for field in ("harness_manifest_paths", "harness_summary_paths", "config_files", "output_paths"):
        for value in manifest.get(field) or []:
            path = Path(str(value))
            if path.exists():
                checked_files.append(str(path))
            else:
                missing_files.append(str(path))

    try:
        profile = get_profile(str(manifest.get("profile")))
    except KeyError as exc:
        errors.append(str(exc))
        profile = None

    run_result = {"rows": [], "anomaly_summary": {"counts": {}, "affected": {}}, "observed_keys": set(), "expected_keys": set()}
    if profile is not None:
        matrix_values = scenario_dimension_values(manifest)
        for dim in profile.required_dimensions:
            if dim == "seed":
                continue
            if dim not in matrix_values:
                errors.append(f"required scenario dimension missing: {dim}")
        for dim, expected_values in profile.expected_values.items():
            if dim == "seed":
                actual = tuple(manifest.get("seed_list") or ())
            else:
                actual = tuple(matrix_values.get(dim) or ())
            if expected_values and actual and actual != tuple(expected_values):
                errors.append(f"{dim} values do not match {profile.name} contract: {actual!r}")
        run_result = classify_runs(manifest, profile)
        for row in run_result["rows"]:
            if row.get("metrics_path"):
                checked_files.append(str(row["metrics_path"]))
            if row.get("config_path"):
                checked_files.append(str(row["config_path"]))
            if row.get("errors"):
                errors.append(f"{row.get('run_id') or row.get('scenario_id')}: {row['errors']}")
            if row.get("warnings"):
                warnings.append(f"{row.get('run_id') or row.get('scenario_id')}: {row['warnings']}")

    observed_records = collect_harness_runs(manifest)
    observed_scenarios = {
        tuple(sorted((record.get("axes") or {}).items()))
        for record in observed_records
        if isinstance(record.get("axes"), dict)
    }
    observed_seeds = {record.get("seed") for record in observed_records if record.get("seed") is not None}
    expected = int(manifest.get("planned_run_count") or len(run_result["expected_keys"]))
    observed = len(observed_records)

    return {
        "ok": not errors,
        "campaign_id": manifest.get("campaign_id"),
        "profile": manifest.get("profile"),
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "checked_files": sorted(set(checked_files)),
        "missing_files": sorted(set(missing_files)),
        "run_count_observed": observed,
        "run_count_expected": expected,
        "scenario_count_observed": len(observed_scenarios),
        "seed_count_observed": len(observed_seeds),
        "run_status": run_result["rows"],
        "anomaly_summary": run_result["anomaly_summary"],
    }


def dumps_result(result: dict[str, Any]) -> str:
    def default(value: Any) -> Any:
        if isinstance(value, set):
            return sorted(value)
        raise TypeError(f"not JSON serializable: {type(value).__name__}")

    return json.dumps(result, indent=2, sort_keys=True, default=default) + "\n"
