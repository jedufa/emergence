from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.evidence.manifest import ManifestError, dumps_manifest, load_manifest


def _manifest(tmp_path: Path) -> dict:
    return {
        "campaign_id": "example",
        "profile": "COG-P2",
        "git_commit": "abc",
        "branch": "example",
        "worktree": ".",
        "run_root": "runs",
        "registry_path": "registry.yaml",
        "scenario_matrix": {"dimensions": [{"name": "ecology_context", "values": ["flat"]}]},
        "seed_list": [0],
        "generations": 1,
        "steps_per_generation": 2,
        "population_size": 3,
        "grid_size": 4,
        "device": "cpu",
        "planned_run_count": 1,
        "completed_run_count": 0,
        "failed_run_count": 0,
        "config_files": ["config.json"],
        "output_paths": ["out.json"],
        "metrics_glob": "**/metrics.csv",
        "config_glob": "**/config.json",
        "harness_manifest_paths": [],
        "harness_summary_paths": ["summary.json"],
        "timestamp_utc": "2026-07-02T00:00:00Z",
    }


def test_manifest_loader_normalizes_paths_and_serializes_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(tmp_path)), encoding="utf-8")
    manifest = load_manifest(path)

    assert manifest.data["run_root"] == str((tmp_path / "runs").resolve())
    assert manifest.data["registry_path"] == str((tmp_path / "registry.yaml").resolve())
    assert manifest.data["harness_summary_paths"] == [str((tmp_path / "summary.json").resolve())]
    assert dumps_manifest({"b": 1, "a": 2}).splitlines()[1].strip() == '"a": 2,'


def test_manifest_loader_reports_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    data = _manifest(tmp_path)
    del data["campaign_id"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ManifestError) as exc:
        load_manifest(path)
    assert "missing required field: campaign_id" in exc.value.errors


def test_manifest_loader_rejects_bad_scenario_matrix(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    data = _manifest(tmp_path)
    data["scenario_matrix"] = {"dimensions": [{"name": "ecology_context", "values": []}]}
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ManifestError) as exc:
        load_manifest(path)
    assert any("non-empty values" in error for error in exc.value.errors)
