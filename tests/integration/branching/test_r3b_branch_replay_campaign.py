from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "bin" / "integrated_r3b_branch_replay_report.py"
SPEC = importlib.util.spec_from_file_location("integrated_r3b_branch_replay_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
r3b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r3b)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _valid_entry(tmp_path: Path) -> dict[str, Any]:
    run = tmp_path / "run"
    checkpoint = run / "checkpoint_bundles_v2" / "bundle_gen000000"
    checkpoint.mkdir(parents=True)
    _write_json(checkpoint / "metadata.json", {"schema_version": "checkpoint_bundle_v2", "generation_index": 0})
    config = run / "config.json"
    metrics = run / "metrics.csv"
    _write_json(
        config,
        {
            "env_cfg": {"patch_world_enable": True},
            "pop_cfg": {"population_size": 64},
            "sim_cfg": {"device": "cuda", "bank_device": "cuda", "enable_developmental_cognition": False},
        },
    )
    metrics.write_text("generation,alive_count\n0,64\n", encoding="utf-8")
    return {
        "manifest_version": "r3b_branch_replay_campaign_manifest_v1",
        "manifest_id": "r3b",
        "entry_id": "entry",
        "source_campaign_id": "integrated_r3a_checkpoint_enabled_branch_source",
        "source_output_root": str(tmp_path),
        "source_run_id": "run",
        "source_scenario_id": "r3a_00_tuned_full_jungle_recovery",
        "source_seed": 101,
        "source_device": "cuda",
        "source_generation": 0,
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_metadata_path": str(checkpoint / "metadata.json"),
        "source_config_path": str(config),
        "source_metrics_path": str(metrics),
        "candidate_lineage_id": "L00000001",
        "candidate_biological_lineage_id": "L00000001",
        "candidate_category": "dominant_lineage",
        "candidate_dominance_class": "dominant",
        "candidate_generation_emitted": 0,
        "candidate_alive_share_at_emission": 0.1,
        "candidate_birth_share_at_emission": 0.2,
        "candidate_descriptor_snapshot": {"lineage_id": "L00000001"},
        "candidate_archive_entry_id": "archive",
        "checkpoint_reference_generation": 0,
        "replay_mode": "checkpoint_continuation",
        "replay_start_generation": 0,
        "replay_generations": 8,
        "replay_steps_per_generation": 60,
        "population_handling_mode": "restored_from_checkpoint",
        "allowed_mutation_policy": "unchanged",
        "allowed_selection_policy": "unchanged",
        "allowed_reproduction_policy": "unchanged",
        "allowed_ecology_policy": "unchanged",
        "expected_flags": {"patch_world_enable": True},
        "expected_tuned_overrides": {},
        "expected_theme4_disabled_status": {"enable_developmental_cognition": True},
        "expected_theme5_reporting_status": {"lineage_qd_archive_enable": True},
        "source_run_settings": {"population_size": 64, "grid_width": 16, "grid_height": 16},
        "output_root": str(tmp_path / "out"),
        "output_dir": str(tmp_path / "out" / "replay"),
    }


def test_r3a_dependency_validation_requires_ready_artifacts(tmp_path: Path) -> None:
    _write_json(tmp_path / "r3a_branch_replay_manifest_validation.json", {"valid": True})
    _write_json(tmp_path / "r3a_smoke_summary.json", {"executed": True, "failed": 0})
    _write_json(tmp_path / "r3a_repeat_parity_summary.json", {"executed": True, "passed": True})
    _write_json(
        tmp_path / "r3a_source_completion.json",
        {"runs": [{"has_checkpoint_generations_0_to_6": True}]},
    )
    dependency = r3b.verify_r3a_dependency(tmp_path)
    assert dependency["valid"] is True


def test_r3b_manifest_validation_passes(tmp_path: Path) -> None:
    manifest = {"entries": [_valid_entry(tmp_path)]}
    validation = r3b.validate_manifest(manifest)
    assert validation["valid"] is True


def test_r3b_manifest_validation_fails_missing_checkpoint(tmp_path: Path) -> None:
    entry = _valid_entry(tmp_path)
    entry["source_checkpoint_path"] = str(tmp_path / "missing")
    validation = r3b.validate_manifest({"entries": [entry]})
    assert validation["valid"] is False
    assert any(err["field"] == "source_checkpoint_path" for err in validation["errors"])


def test_classify_fate_extinct_and_recovered() -> None:
    extinct = [
        {"candidate_alive_count": 1, "candidate_alive_share": 0.01},
        {"candidate_alive_count": 0, "candidate_alive_share": 0.0},
    ]
    recovered = [
        {"candidate_alive_count": 1, "candidate_alive_share": 0.01},
        {"candidate_alive_count": 3, "candidate_alive_share": 0.03},
    ]
    assert r3b.classify_fate(extinct) == "extinct"
    assert r3b.classify_fate(recovered) == "recovered"


def test_grouped_comparisons_reports_dominant_and_non_dominant() -> None:
    fates = [
        {
            "candidate_dominance_class": "dominant",
            "source_scenario_id": "s",
            "alive_share_start": 0.1,
            "alive_share_end": 0.2,
            "births_during_replay": 2,
            "fate": "recovered",
        },
        {
            "candidate_dominance_class": "non_dominant",
            "source_scenario_id": "s",
            "alive_share_start": 0.01,
            "alive_share_end": 0.0,
            "births_during_replay": 0,
            "fate": "extinct",
        },
    ]
    rows, scenario = r3b.grouped_comparisons(fates)
    groups = {row["group"] for row in rows}
    assert groups == {"dominant", "non_dominant"}
    assert scenario["scenarios"]["s"]["fates"]["extinct"] == 1
