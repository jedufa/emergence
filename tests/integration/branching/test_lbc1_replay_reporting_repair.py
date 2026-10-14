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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _entry(tmp_path: Path, *, lineage: str = "L00000002") -> dict[str, Any]:
    run = tmp_path / "source"
    checkpoint = run / "checkpoint_bundles_v2" / "bundle_gen000000"
    checkpoint.mkdir(parents=True)
    _write_json(checkpoint / "metadata.json", {"schema_version": "checkpoint_bundle_v2", "generation_index": 0})
    config = run / "config.json"
    metrics = run / "metrics.csv"
    _write_json(
        config,
        {
            "env_cfg": {"patch_world_enable": True},
            "sim_cfg": {
                "enable_developmental_cognition": False,
                "lineage_protection": False,
                "archive_reseed_enable": False,
            },
        },
    )
    metrics.write_text("generation,alive_count\n0,64\n", encoding="utf-8")
    return {
        "manifest_version": "lbc1_branch_replay_manifest_v1",
        "manifest_id": "lbc1r_fixture",
        "entry_id": "entry",
        "source_campaign_id": "lbc1_fixture",
        "source_output_root": str(tmp_path),
        "source_run_id": "source",
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_metadata_path": str(checkpoint / "metadata.json"),
        "source_config_path": str(config),
        "source_metrics_path": str(metrics),
        "source_scenario_id": "lbc1_00_tuned_full_jungle_recovery",
        "source_seed": 101,
        "source_device": "cuda",
        "source_generation": 0,
        "candidate_lineage_id": lineage,
        "candidate_biological_lineage_id": lineage,
        "candidate_category": "branch_candidate",
        "candidate_class": "early_non_dominant",
        "candidate_dominance_class": "non_dominant",
        "candidate_generation_emitted": 0,
        "candidate_alive_share_at_emission": 0.1,
        "candidate_birth_share_at_emission": 0.0,
        "candidate_descriptor_snapshot": {"lineage_id": lineage},
        "candidate_archive_entry_id": "missing_optional",
        "replay_mode": "checkpoint_continuation",
        "replay_start_generation": 0,
        "replay_generations": 8,
        "replay_steps_per_generation": 60,
        "population_handling_mode": "restored_from_checkpoint",
        "allowed_mutation_policy": "unchanged",
        "allowed_selection_policy": "unchanged",
        "allowed_reproduction_policy": "unchanged",
        "allowed_ecology_policy": "unchanged",
        "expected_theme4_disabled_status": {"enable_developmental_cognition": True},
        "expected_theme5_reporting_status": {"lineage_qd_archive_enable": True},
        "expected_flags": {"patch_world_enable": True},
        "expected_tuned_overrides": {},
        "output_root": str(tmp_path / "out"),
        "output_dir": str(tmp_path / "out"),
    }


def _replay_dir(tmp_path: Path, *, include_candidate: bool = True) -> Path:
    replay = tmp_path / "replay"
    rows = [
        {"generation": 0, "lineage_id": "L00000001", "population_count": 10, "population_share": 0.5, "offspring_count": 2, "reproduction_share": 0.5},
        {"generation": 1, "lineage_id": "L00000001", "population_count": 12, "population_share": 0.6, "offspring_count": 3, "reproduction_share": 0.75},
    ]
    if include_candidate:
        rows.extend(
            [
                {"generation": 0, "lineage_id": "L00000002", "population_count": 2, "population_share": 0.1, "offspring_count": 0, "reproduction_share": 0.0},
                {"generation": 1, "lineage_id": "L00000002", "population_count": 3, "population_share": 0.15, "offspring_count": 1, "reproduction_share": 0.25},
            ]
        )
    _write_jsonl(replay / "metrics" / "lineage_metrics.jsonl", rows)
    descriptors = [
        {
            "generation": 0,
            "lineages": [
                {"lineage_id": "L00000001", "population_count": 10},
                {"lineage_id": "L00000002", "population_count": 2, "lineage_patch_waste_mean": 0.2},
            ],
        },
        {
            "generation": 1,
            "lineages": [
                {"lineage_id": "L00000001", "population_count": 12},
                {"lineage_id": "L00000002", "population_count": 3, "lineage_patch_waste_mean": 0.25},
            ],
        },
    ]
    if not include_candidate:
        for row in descriptors:
            row["lineages"] = [item for item in row["lineages"] if item["lineage_id"] != "L00000002"]
    _write_jsonl(replay / "metrics" / "lineage_descriptors.jsonl", descriptors)
    return replay


def test_candidate_identity_mapping_present_and_dominant_from_replay_state(tmp_path: Path) -> None:
    tracking = lbc1._build_replay_lineage_tracking(_replay_dir(tmp_path), _entry(tmp_path))["candidate"]
    assert tracking["identity_mapping"]["candidate_resolved_in_replay_start"] is True
    assert tracking["dominant_lineage_id"] == "L00000001"
    assert tracking["trajectory"][0]["candidate_present"] is True
    assert tracking["trajectory"][0]["candidate_biological_lineage_id"] == "L00000002"


def test_candidate_identity_mapping_absent_at_replay_start(tmp_path: Path) -> None:
    tracking = lbc1._build_replay_lineage_tracking(_replay_dir(tmp_path, include_candidate=False), _entry(tmp_path))["candidate"]
    assert tracking["identity_mapping"]["candidate_absent_at_replay_start"] is True
    assert tracking["trajectory"][0]["candidate_present"] is False
    assert lbc1.classify_fate(tracking["trajectory"]) == "absent_at_replay_start"


def test_trajectory_and_availability_schema(tmp_path: Path) -> None:
    bundle = lbc1._build_replay_lineage_tracking(_replay_dir(tmp_path), _entry(tmp_path))
    candidate_row = bundle["candidate"]["trajectory"][0]
    dominant_row = bundle["dominant"]["trajectory"][0]
    assert {"candidate_alive_count", "candidate_alive_share", "candidate_birth_count", "candidate_birth_share"} <= set(candidate_row)
    assert {"dominant_alive_count", "dominant_alive_share", "dominant_birth_count", "dominant_birth_share"} <= set(dominant_row)
    assert bundle["candidate"]["field_availability"]["waste_exposure"]["available"] is True
    assert bundle["candidate"]["field_availability"]["predator_damage"]["available"] is False
    assert bundle["candidate"]["field_availability"]["predator_damage"]["reason_if_unavailable"]


def test_fate_classification_repaired_labels() -> None:
    assert lbc1.classify_fate(
        [
            {"candidate_present": True, "candidate_alive_count": 2, "candidate_alive_share": 0.1, "candidate_birth_count": 0, "candidate_birth_share": 0.0, "dominant_lineage_alive_share": 0.3},
            {"candidate_present": True, "candidate_alive_count": 2, "candidate_alive_share": 0.1, "candidate_birth_count": 0, "candidate_birth_share": 0.0, "dominant_lineage_alive_share": 0.3},
        ]
    ) == "persisted"
    assert lbc1.classify_fate(
        [
            {"candidate_present": True, "candidate_alive_count": 2, "candidate_alive_share": 0.1, "candidate_birth_count": 0, "candidate_birth_share": 0.0, "dominant_lineage_alive_share": 0.3},
            {"candidate_present": False, "candidate_alive_count": 0, "candidate_alive_share": 0.0, "candidate_birth_count": 0, "candidate_birth_share": 0.0, "dominant_lineage_alive_share": 0.5},
        ]
    ) == "extinct_after_start"
    assert lbc1.classify_fate([{"candidate_present": True}]) == "unclear"


def test_proxy_label_discipline_requires_available_exposure_fields() -> None:
    labels = set(
        lbc1.cause_proxy(
            {},
            [
                {"candidate_present": True, "candidate_alive_share": 0.01, "candidate_birth_count": 0, "candidate_alive_count": 1, "dominant_lineage_alive_share": 0.2},
                {"candidate_present": True, "candidate_alive_share": 0.01, "candidate_birth_count": 0, "candidate_alive_count": 1, "dominant_lineage_alive_share": 0.2},
            ],
        ).split(";")
    )
    assert {"already_low_share", "low_reproduction"} <= labels
    assert "high_predator_damage" not in labels
    assert "low_food_exposure" not in labels


def test_no_intervention_manifest_validation(tmp_path: Path) -> None:
    entry = _entry(tmp_path)
    validation = lbc1.validate_manifest({"entries": [entry]})
    assert validation["valid"] is True
    config = Path(entry["source_config_path"])
    data = json.loads(config.read_text(encoding="utf-8"))
    data["sim_cfg"]["lineage_protection"] = True
    _write_json(config, data)
    validation = lbc1.validate_manifest({"entries": [entry]})
    assert validation["valid"] is False
    assert any("lineage_protection" in err["field"] for err in validation["errors"])
