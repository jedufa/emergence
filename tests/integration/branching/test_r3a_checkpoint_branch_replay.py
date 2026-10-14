from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "bin" / "integrated_r3a_branch_replay.py"
SPEC = importlib.util.spec_from_file_location("integrated_r3a_branch_replay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
r3a = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r3a)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _valid_entry(tmp_path: Path) -> dict[str, Any]:
    run = tmp_path / "run"
    checkpoint = run / "checkpoint_bundles_v2" / "bundle_gen000003"
    checkpoint.mkdir(parents=True)
    _write_json(checkpoint / "metadata.json", {"schema_version": "checkpoint_bundle_v2", "generation_index": 3})
    config = run / "config.json"
    metrics = run / "metrics.csv"
    _write_json(
        config,
        {
            "env_cfg": {"patch_world_enable": True, "jungle_predator_pressure_enable": True},
            "pop_cfg": {"population_size": 64},
            "sim_cfg": {
                "device": "cuda",
                "bank_device": "cuda",
                "enable_developmental_cognition": False,
                "phase_e_e1_enable": False,
                "lineage_qd_archive_enable": True,
            },
        },
    )
    metrics.write_text("generation,alive_count\n3,64\n", encoding="utf-8")
    return {
        "manifest_version": "r3a_branch_replay_manifest_v1",
        "manifest_id": "r3a_test",
        "entry_id": "entry",
        "source_campaign_id": "integrated_r3a_checkpoint_enabled_branch_source",
        "source_output_root": str(tmp_path),
        "source_run_id": "run",
        "source_scenario_id": "r3a_00_tuned_full_jungle_recovery",
        "source_seed": 101,
        "source_device": "cuda",
        "source_generation": 3,
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_metadata_path": str(checkpoint / "metadata.json"),
        "source_config_path": str(config),
        "source_metrics_path": str(metrics),
        "candidate_lineage_id": "L00000001",
        "candidate_biological_lineage_id": "L00000001",
        "candidate_category": "survival_specialist",
        "candidate_generation_emitted": 3,
        "candidate_alive_share_at_emission": 0.1,
        "candidate_birth_share_at_emission": 0.2,
        "candidate_descriptor_snapshot": {"lineage_id": "L00000001"},
        "candidate_archive_entry_id": "archive",
        "checkpoint_reference_generation": 3,
        "replay_mode": "checkpoint_continuation",
        "replay_start_generation": 3,
        "replay_generations": 4,
        "replay_steps_per_generation": 60,
        "population_handling_mode": "restored_from_checkpoint",
        "allowed_mutation_policy": "unchanged",
        "allowed_selection_policy": "unchanged",
        "allowed_reproduction_policy": "unchanged",
        "allowed_ecology_policy": "unchanged",
        "expected_flags": {"patch_world_enable": True, "jungle_predator_pressure_enable": True},
        "expected_tuned_overrides": {},
        "expected_theme4_disabled_status": {"enable_developmental_cognition": True, "phase_e_e1_enable": True},
        "expected_theme5_reporting_status": {"lineage_qd_archive_enable": True},
        "source_run_settings": {
            "generations": 8,
            "steps_per_generation": 60,
            "population_size": 64,
            "grid_width": 16,
            "grid_height": 16,
            "policy_tick_stride": 1,
            "bank_device": "cuda",
        },
        "output_root": str(tmp_path / "replay"),
    }


def test_checkpoint_backed_manifest_validation_passes(tmp_path: Path) -> None:
    validation = r3a.validate_manifest({"entries": [_valid_entry(tmp_path)]})
    assert validation["valid"] is True


def test_missing_source_checkpoint_path_fails(tmp_path: Path) -> None:
    entry = _valid_entry(tmp_path)
    entry["source_checkpoint_path"] = str(tmp_path / "missing")
    validation = r3a.validate_manifest({"entries": [entry]})
    assert validation["valid"] is False
    assert any(err["field"] == "source_checkpoint_path" for err in validation["errors"])


def test_missing_source_config_path_fails(tmp_path: Path) -> None:
    entry = _valid_entry(tmp_path)
    entry["source_config_path"] = str(tmp_path / "missing_config.json")
    validation = r3a.validate_manifest({"entries": [entry]})
    assert validation["valid"] is False
    assert any(err["field"] == "source_config_path" for err in validation["errors"])


def test_missing_candidate_lineage_id_fails(tmp_path: Path) -> None:
    entry = _valid_entry(tmp_path)
    entry["candidate_lineage_id"] = ""
    validation = r3a.validate_manifest({"entries": [entry]})
    assert validation["valid"] is False
    assert any(err["field"] == "candidate_lineage_id" for err in validation["errors"])


def test_invalid_replay_mode_fails(tmp_path: Path) -> None:
    entry = _valid_entry(tmp_path)
    entry["replay_mode"] = "archive_reseed"
    validation = r3a.validate_manifest({"entries": [entry]})
    assert validation["valid"] is False
    assert any(err["field"] == "replay_mode" for err in validation["errors"])


def test_lineage_protection_and_diversity_reward_flags_fail(tmp_path: Path) -> None:
    entry = _valid_entry(tmp_path)
    config = Path(entry["source_config_path"])
    data = json.loads(config.read_text(encoding="utf-8"))
    data["sim_cfg"]["lineage_protection_enable"] = True
    data["sim_cfg"]["diversity_reward_enable"] = True
    _write_json(config, data)
    validation = r3a.validate_manifest({"entries": [entry]})
    assert validation["valid"] is False
    fields = {err["field"] for err in validation["errors"]}
    assert "sim_cfg.lineage_protection_enable" in fields
    assert "sim_cfg.diversity_reward_enable" in fields


def test_candidate_links_to_exact_checkpoint_generation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for gen in (0, 3):
        bundle = run / "checkpoint_bundles_v2" / f"bundle_gen{gen:06d}"
        bundle.mkdir(parents=True)
        _write_json(bundle / "metadata.json", {"generation_index": gen})
    linked = r3a.link_checkpoint_for_generation(run, 3)
    assert linked["valid"] is True
    assert linked["checkpoint_generation"] == 3


def test_candidate_links_to_nearest_prior_checkpoint_when_allowed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    for gen in (0, 2):
        bundle = run / "checkpoint_bundles_v2" / f"bundle_gen{gen:06d}"
        bundle.mkdir(parents=True)
        _write_json(bundle / "metadata.json", {"generation_index": gen})
    linked = r3a.link_checkpoint_for_generation(run, 3)
    assert linked["valid"] is True
    assert linked["checkpoint_generation"] == 2


def test_candidate_without_prior_checkpoint_is_invalid(tmp_path: Path) -> None:
    run = tmp_path / "run"
    bundle = run / "checkpoint_bundles_v2" / "bundle_gen000004"
    bundle.mkdir(parents=True)
    _write_json(bundle / "metadata.json", {"generation_index": 4})
    linked = r3a.link_checkpoint_for_generation(run, 3)
    assert linked["valid"] is False


def test_replay_cli_preserves_no_intervention_policy(tmp_path: Path) -> None:
    entry = _valid_entry(tmp_path)
    cli = r3a._replay_cli(entry, runs_dir=tmp_path / "runs", run_id="replay")
    joined = " ".join(cli)
    assert "--resume_from" in cli
    assert "lineage_protection" not in joined
    assert "diversity_reward" not in joined
    assert "archive_reseed" not in joined
    assert "sim_cfg.enable_developmental_cognition=false" in cli
