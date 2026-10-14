from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


MODULE_PATH = Path(__file__).resolve().parents[3] / "tools" / "bin" / "lbc1_lineage_bottleneck_closure_report.py"
SPEC = importlib.util.spec_from_file_location("lbc1_lineage_bottleneck_closure_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
lbc1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lbc1)


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
        "manifest_version": "lbc1_branch_replay_manifest_v1",
        "manifest_id": "lbc1",
        "entry_id": "entry",
        "source_campaign_id": "lbc1_lineage_bottleneck_closure_source",
        "source_output_root": str(tmp_path),
        "source_run_id": "run",
        "source_scenario_id": "lbc1_00_tuned_full_jungle_recovery",
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
        "candidate_class": "dominant",
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
        "output_root": str(tmp_path / "out"),
        "output_dir": str(tmp_path / "out" / "replay"),
    }


def test_lbc1_manifest_validation_passes(tmp_path: Path) -> None:
    validation = lbc1.validate_manifest({"entries": [_valid_entry(tmp_path)]})
    assert validation["valid"] is True


def test_lbc1_manifest_validation_rejects_unknown_candidate_class(tmp_path: Path) -> None:
    entry = _valid_entry(tmp_path)
    entry["candidate_class"] = "lineage_rescue"
    validation = lbc1.validate_manifest({"entries": [entry]})
    assert validation["valid"] is False
    assert any(err["field"] == "candidate_class" for err in validation["errors"])


def test_exposure_field_availability_reports_available_fields() -> None:
    run = {
        "tracking": {
            "trajectory": [
                {field: 1.0 for field in lbc1.EXPOSURE_FIELDS},
                {field: 2.0 for field in lbc1.EXPOSURE_FIELDS},
            ]
        }
    }
    availability = lbc1.exposure_field_availability([run])
    assert availability["sufficient_for_proxy"] is True
    assert all(data["status"] == "available" for data in availability["fields"].values())


def test_fate_classification_thresholds() -> None:
    assert lbc1.classify_fate(
        [
            {"candidate_present": True, "candidate_alive_count": 1, "candidate_alive_share": 0.01, "candidate_birth_count": 0, "candidate_birth_share": 0.0, "dominant_lineage_alive_share": 0.2},
            {"candidate_present": False, "candidate_alive_count": 0, "candidate_alive_share": 0.0, "candidate_birth_count": 0, "candidate_birth_share": 0.0, "dominant_lineage_alive_share": 0.8},
        ]
    ) == "dominant_sweep"
    assert lbc1.classify_fate(
        [
            {"candidate_present": True, "candidate_alive_count": 1, "candidate_alive_share": 0.01, "candidate_birth_count": 0, "candidate_birth_share": 0.0, "dominant_lineage_alive_share": 0.2},
            {"candidate_present": True, "candidate_alive_count": 3, "candidate_alive_share": 0.03, "candidate_birth_count": 1, "candidate_birth_share": 0.1, "dominant_lineage_alive_share": 0.3},
        ]
    ) == "recovered"


def test_cause_proxy_detects_low_share_and_low_reproduction() -> None:
    entry = {"candidate_alive_share_at_emission": 0.01}
    trajectory = [
        {
            "candidate_present": True,
            "candidate_alive_share": 0.01,
            "candidate_birth_count": 0,
            "candidate_alive_count": 1,
            "dominant_lineage_alive_share": 0.2,
            "food_exposure": 0.5,
            "waste_exposure": 0.2,
        },
        {
            "candidate_present": False,
            "candidate_alive_share": 0.0,
            "candidate_birth_count": 0,
            "candidate_alive_count": 0,
            "dominant_lineage_alive_share": 0.8,
            "food_exposure": 0.5,
            "waste_exposure": 0.2,
        },
    ]
    labels = set(lbc1.cause_proxy(entry, trajectory).split(";"))
    assert {"already_low_share", "low_reproduction", "global_dominant_sweep"} <= labels


def test_decision_json_schema_for_acceptance() -> None:
    cohort = {"selected_count": 8}
    validation = {"valid": True}
    parity = {"passed": True}
    availability = {"sufficient_for_proxy": True}
    replay_summary = {
        "candidate_fates": [
            {"candidate_dominance_class": "dominant", "fate": "recovered", "births_during_replay": 3},
            {"candidate_dominance_class": "dominant", "fate": "persisted", "births_during_replay": 1},
            {"candidate_dominance_class": "dominant", "fate": "recovered", "births_during_replay": 2},
            {"candidate_dominance_class": "non_dominant", "fate": "dominant_sweep", "births_during_replay": 0},
            {"candidate_dominance_class": "non_dominant", "fate": "extinct_after_start", "births_during_replay": 0},
            {"candidate_dominance_class": "non_dominant", "fate": "extinct_after_start", "births_during_replay": 0},
            {"candidate_dominance_class": "non_dominant", "fate": "dominant_sweep", "births_during_replay": 0},
            {"candidate_dominance_class": "non_dominant", "fate": "extinct_after_start", "births_during_replay": 0},
        ]
    }
    decision = lbc1.decide(
        validation=validation,
        parity=parity,
        cohort=cohort,
        availability=availability,
        replay_summary=replay_summary,
        comparison=[],
    )
    assert decision["phase_ii_decision"] == "A"
    assert decision["decision_wording"].startswith("Accept collapse")


def test_decision_does_not_force_c_when_only_exposure_is_missing() -> None:
    cohort = {"selected_count": 8}
    replay_summary = {
        "candidate_fates": [
            {"candidate_dominance_class": "dominant", "fate": "recovered", "births_during_replay": 3},
            {"candidate_dominance_class": "dominant", "fate": "persisted", "births_during_replay": 1},
            {"candidate_dominance_class": "dominant", "fate": "recovered", "births_during_replay": 2},
            {"candidate_dominance_class": "non_dominant", "fate": "dominant_sweep", "births_during_replay": 0},
            {"candidate_dominance_class": "non_dominant", "fate": "extinct_after_start", "births_during_replay": 0},
            {"candidate_dominance_class": "non_dominant", "fate": "extinct_after_start", "births_during_replay": 0},
            {"candidate_dominance_class": "non_dominant", "fate": "dominant_sweep", "births_during_replay": 0},
            {"candidate_dominance_class": "non_dominant", "fate": "extinct_after_start", "births_during_replay": 0},
        ]
    }
    decision = lbc1.decide(
        validation={"valid": True},
        parity={"passed": True},
        cohort=cohort,
        availability={"sufficient_for_proxy": False},
        replay_summary=replay_summary,
        comparison=[],
    )
    assert decision["phase_ii_decision"] == "A"


def test_no_forbidden_intervention_flags(tmp_path: Path) -> None:
    entry = _valid_entry(tmp_path)
    config = Path(entry["source_config_path"])
    data = json.loads(config.read_text(encoding="utf-8"))
    data["sim_cfg"]["archive_reseed_enable"] = True
    _write_json(config, data)
    validation = lbc1.validate_manifest({"entries": [entry]})
    assert validation["valid"] is False
    assert any("archive_reseed" in err["field"] for err in validation["errors"])
