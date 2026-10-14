from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "bin" / "integrated_r3_branch_replay_readiness.py"
SPEC = importlib.util.spec_from_file_location("integrated_r3_branch_replay_readiness", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
r3 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r3)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _valid_candidate(tmp_path: Path) -> dict[str, Any]:
    checkpoint = tmp_path / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint.mkdir(parents=True)
    config = tmp_path / "config.json"
    metrics = tmp_path / "metrics.csv"
    _write_json(config, {"env_cfg": {}, "sim_cfg": {}})
    metrics.write_text("generation,alive\n1,1\n", encoding="utf-8")
    return {
        "manifest_version": "r3_branch_replay_manifest_v1",
        "manifest_id": "test",
        "source_campaign_id": "r2",
        "source_output_root": str(tmp_path),
        "source_run_id": "run",
        "source_scenario_id": "scenario",
        "source_seed": 101,
        "source_device": "cuda",
        "source_generation": 1,
        "source_step_or_tick": r3._marker("missing_optional", "source_step_or_tick", "not emitted"),
        "source_checkpoint_path": str(checkpoint),
        "source_config_path": str(config),
        "source_metrics_path": str(metrics),
        "candidate_lineage_id": "L00000001",
        "candidate_biological_lineage_id": "L00000001",
        "candidate_category": "rare_persistent",
        "candidate_generation_emitted": 1,
        "candidate_alive_share_at_emission": 0.1,
        "candidate_birth_share_at_emission": 0.2,
        "candidate_descriptor_snapshot": {"lineage_id": "L00000001"},
        "candidate_archive_entry_id": r3._marker("missing_optional", "candidate_archive_entry_id", "not archived"),
        "replay_mode": "checkpoint_continuation",
        "replay_start_generation": 1,
        "replay_generations": 4,
        "replay_steps_per_generation": 60,
        "population_handling_mode": "restore_from_checkpoint_without_population_modification",
        "allowed_mutation_policy": "source_config_only_no_override",
        "allowed_selection_policy": "source_config_only_no_override",
        "expected_flags": {"patch_world_enable": True},
        "expected_tuned_overrides": {},
        "expected_theme4_disabled_status": {"enable_developmental_cognition": True},
        "expected_theme5_reporting_status": {"lineage_qd_archive_enable": True},
        "output_root": str(tmp_path / "out"),
    }


def test_valid_r3_manifest_passes_with_existing_required_paths(tmp_path: Path) -> None:
    manifest = {"candidates": [_valid_candidate(tmp_path)]}
    validation = r3.validate_manifest(manifest)
    assert validation["valid"] is True
    assert validation["warnings"]


def test_missing_required_source_checkpoint_fails(tmp_path: Path) -> None:
    candidate = _valid_candidate(tmp_path)
    candidate["source_checkpoint_path"] = r3._marker(
        "missing_required", "source_checkpoint_path", "source did not save checkpoints"
    )
    validation = r3.validate_manifest({"candidates": [candidate]})
    assert validation["valid"] is False
    assert any(err["field"] == "source_checkpoint_path" for err in validation["errors"])


def test_missing_lineage_id_fails(tmp_path: Path) -> None:
    candidate = _valid_candidate(tmp_path)
    candidate["candidate_lineage_id"] = ""
    validation = r3.validate_manifest({"candidates": [candidate]})
    assert validation["valid"] is False
    assert any(err["field"] == "candidate_lineage_id" for err in validation["errors"])


def test_candidate_extraction_distinguishes_dominant_and_non_dominant(tmp_path: Path) -> None:
    r2_root = tmp_path / "r2"
    run = r2_root / "runs" / "r0001"
    run.mkdir(parents=True)
    _write_json(
        r2_root / "r2_summary.json",
        {
            "lineage_fate_summary": {
                "runs": [
                    {"scenario": "r2_03_tuned_full_jungle_recovery", "seed": 101, "run_dir": str(run)},
                    {"scenario": "r2_01_default_full_jungle", "seed": 101, "run_dir": str(run)},
                    {"scenario": "r2_04_theme12_clean_pressure_reference", "seed": 101, "run_dir": str(run)},
                ]
            }
        },
    )
    _write_json(run / "config.json", {"env_cfg": {"patch_world_enable": True}, "sim_cfg": {"device": "cuda"}})
    (run / "metrics.csv").write_text("generation,alive\n0,1\n", encoding="utf-8")
    _write_json(run / "manifest.json", {"runtime_run_id": "r0001"})
    _write_jsonl(
        run / "metrics" / "lineage_metrics.jsonl",
        [
            {"generation": 0, "lineage_id": "L00000001", "population_share": 0.8, "reproduction_share": 0.8},
            {"generation": 0, "lineage_id": "L00000002", "population_share": 0.2, "reproduction_share": 0.2},
        ],
    )
    _write_jsonl(
        run / "metrics" / "lineage_descriptors.jsonl",
        [{"generation": 0, "lineages": [{"lineage_id": "L00000001"}, {"lineage_id": "L00000002"}]}],
    )
    _write_jsonl(
        run / "metrics" / "branch_candidates.jsonl",
        [
            {
                "candidates": [
                    {
                        "generation": 0,
                        "lineage_id": "L00000001",
                        "candidate_type": "survival_specialist",
                        "candidate_score": 0.9,
                        "population_share": 0.8,
                        "reproduction_share": 0.8,
                    },
                    {
                        "generation": 0,
                        "lineage_id": "L00000002",
                        "candidate_type": "rare_persistent",
                        "candidate_score": 0.8,
                        "population_share": 0.2,
                        "reproduction_share": 0.2,
                    },
                ]
            }
        ],
    )
    _write_json(
        run / "archives" / "lineage_qd" / "entries" / "a.json",
        {"lineage_id": "L00000002", "generation": 0, "archive_entry_id": "a", "candidate_score": 0.8},
    )
    selection = r3.select_candidates_from_r2(r2_root, tmp_path / "out")
    roles = {item["candidate_category"] for item in selection["selected_candidates"]}
    assert "dominant_lineage" in roles
    assert "rare_persistent" in roles


def test_manifest_policies_do_not_modify_mechanics(tmp_path: Path) -> None:
    candidate = _valid_candidate(tmp_path)
    assert candidate["population_handling_mode"] == "restore_from_checkpoint_without_population_modification"
    assert candidate["allowed_mutation_policy"] == "source_config_only_no_override"
    assert candidate["allowed_selection_policy"] == "source_config_only_no_override"
    assert all(candidate["expected_theme4_disabled_status"].values())
    validation = r3.validate_manifest({"candidates": [candidate]})
    assert validation["valid"] is True
