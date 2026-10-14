from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simulation.chronicle.compare import compare_chronicles
from simulation.chronicle.config import ChronicleConfig, storage_preflight
from simulation.chronicle.recorder import ScientificChronicleRecorder
from simulation.chronicle.retrospective import regenerate_chronicle
from simulation.chronicle.storage import ARTIFACT_NAMES, read_json, read_jsonl


@dataclass
class _Stats:
    lineage_id: str
    founder_id: int


@dataclass
class _Agent:
    agent_id: int
    stats: _Stats


def _agent(agent_id: int, lineage_id: str) -> _Agent:
    return _Agent(agent_id=agent_id, stats=_Stats(lineage_id=lineage_id, founder_id=agent_id))


def _recorder(
    root: Path,
    *,
    run_id: str = "run-a",
    profile: str = "notable",
    maximum: int = 32,
    persistence: int = 1,
) -> ScientificChronicleRecorder:
    cfg = SimpleNamespace(
        chronicle_enable=True,
        chronicle_profile=profile,
        chronicle_run_id=run_id,
        chronicle_max_stored_events_per_generation=maximum,
        chronicle_minimum_persistence_generations=persistence,
        campaign_id="campaign",
        branch_id=run_id,
        generations=4,
        allow_large_output_profile=profile in {"scientific_full", "debug_trace"},
    )
    return ScientificChronicleRecorder(SimpleNamespace(sim_cfg=cfg, run_dir=str(root)))


def _emit_two_generations(recorder: ScientificChronicleRecorder, *, divergent: bool = False) -> None:
    recorder.observe_generation(
        generation=0,
        step=9,
        alive_agents=[_agent(1, "L1"), _agent(2, "L2")],
        deaths=0,
        births=0,
        failed_births=0,
        cohort_size=2,
        patch_metrics={"patch_food_depleted_frac": 0.05, "patch_density_hotspot_frac": 0.0},
    )
    recorder.observe_generation(
        generation=1,
        step=9,
        alive_agents=[_agent(1, "L1")],
        deaths=1,
        births=0,
        failed_births=1,
        cohort_size=2,
        patch_metrics={
            "patch_food_depleted_frac": 0.40 if divergent else 0.30,
            "patch_density_hotspot_frac": 0.30,
            "occupied_patch_waste_mean": 0.40,
            "jungle_predator_events": 2,
            "jungle_predator_damage_total": 0.5,
            "jungle_predator_deaths": 1,
            "wdl_delayed_hazard_event_count": 1,
            "wdl_delayed_hazard_damage_total": 0.2,
        },
        death_contexts=[{"agent_id": 2, "biological_lineage_id": "L2", "y": 2, "x": 3}],
        predator_contexts=[{"agent_id": 2, "y": 2, "x": 3}],
        wdl_contexts=[{"phase": "active"}],
    )
    recorder.finalize()


def test_recorder_emits_traceable_events_episodes_and_context(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path / "run")
    _emit_two_generations(recorder)
    events = read_jsonl(recorder.root / ARTIFACT_NAMES["events"])
    entries = read_jsonl(recorder.root / ARTIFACT_NAMES["entries"])
    episodes = read_json(recorder.root / ARTIFACT_NAMES["episodes"])
    contexts = read_jsonl(recorder.root / ARTIFACT_NAMES["contexts"])
    event_ids = {event["event_id"] for event in events}

    assert sum(event["event_type"] == "generation_summary" for event in events) == 2
    assert {"notable_death", "lineage_extinction", "regional_food_depletion", "predator_pressure_spike", "delayed_hazard_cycle"}.issubset(
        {event["event_type"] for event in events}
    )
    assert all(set(entry["source_event_ids"]).issubset(event_ids) for entry in entries)
    assert all(set(episode["source_event_ids"]).issubset(event_ids) for episode in episodes)
    assert contexts and all(set(window["source_event_ids"]).issubset(event_ids) for window in contexts)
    assert all("thought" not in entry["scientific_summary"].lower() for entry in entries)


def test_bottleneck_uses_surviving_start_cohort_not_population_with_births(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path / "cohort-survival")
    recorder.observe_generation(
        generation=0,
        step=9,
        alive_agents=[_agent(1, "L1"), _agent(2, "L2")],
        deaths=1,
        births=1,
        failed_births=0,
        cohort_size=2,
        cohort_survival_fraction=0.5,
        patch_metrics={"patch_food_depleted_frac": 0.0},
    )
    recorder.finalize()

    events = read_jsonl(recorder.root / ARTIFACT_NAMES["events"])
    summary = next(event for event in events if event["event_type"] == "generation_summary")
    bottleneck = next(event for event in events if event["event_type"] == "population_bottleneck")

    assert summary["measurements"]["cohort_survival_fraction"] == 0.5
    assert bottleneck["measurements"]["survival_fraction"] == 0.5


def test_notable_birth_dominance_and_recovery_families(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path / "families")
    recorder.observe_generation(
        generation=0,
        step=9,
        alive_agents=[_agent(1, "L1"), _agent(2, "L2"), _agent(3, "L2")],
        deaths=0,
        births=0,
        failed_births=0,
        cohort_size=3,
        patch_metrics={"patch_food_depleted_frac": 0.30},
    )
    recorder.observe_generation(
        generation=1,
        step=9,
        alive_agents=[_agent(1, "L1"), _agent(4, "L1"), _agent(5, "L1"), _agent(6, "L3")],
        deaths=2,
        births=3,
        failed_births=0,
        cohort_size=3,
        patch_metrics={"patch_food_depleted_frac": 0.05},
        birth_contexts=[
            {"agent_id": 6, "biological_lineage_id": "L3", "y": 1, "x": 2},
        ],
    )
    recorder.finalize()

    event_types = {event["event_type"] for event in recorder.events}
    assert {"notable_birth", "lineage_dominance_crossing", "regional_food_recovery"}.issubset(event_types)


def test_summary_profile_and_budget_preserve_one_summary_per_generation(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path / "summary", profile="summary", maximum=1)
    _emit_two_generations(recorder)

    events = read_jsonl(recorder.root / ARTIFACT_NAMES["events"])
    assert [(event["generation"], event["event_type"]) for event in events] == [
        (0, "generation_summary"),
        (1, "generation_summary"),
    ]
    assert recorder.suppressed_counts["generation_budget"] > 0


def test_configured_persistence_delays_threshold_event_deterministically(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path / "persistence", persistence=2)
    for generation in (0, 1):
        recorder.observe_generation(
            generation=generation,
            step=9,
            alive_agents=[_agent(1, "L1"), _agent(2, "L2")],
            deaths=0,
            births=0,
            failed_births=0,
            cohort_size=2,
            patch_metrics={"patch_food_depleted_frac": 0.40},
        )

    depletion = [event for event in recorder.events if event["event_type"] == "regional_food_depletion"]
    assert len(depletion) == 1
    assert depletion[0]["generation"] == 1
    assert depletion[0]["measurements"]["persistence_generations"] == 2


def test_checkpoint_state_roundtrip_has_no_duplicate_or_gap(tmp_path: Path) -> None:
    uninterrupted = _recorder(tmp_path / "uninterrupted", run_id="same-run")
    _emit_two_generations(uninterrupted)

    first_leg = _recorder(tmp_path / "first-leg", run_id="same-run")
    first_leg.observe_generation(
        generation=0,
        step=9,
        alive_agents=[_agent(1, "L1"), _agent(2, "L2")],
        deaths=0,
        births=0,
        failed_births=0,
        cohort_size=2,
        patch_metrics={"patch_food_depleted_frac": 0.05, "patch_density_hotspot_frac": 0.0},
    )
    state = json.loads(json.dumps(first_leg.export_state()))
    resumed = _recorder(tmp_path / "resumed", run_id="same-run")
    resumed.import_state(state)
    resumed.observe_generation(
        generation=1,
        step=9,
        alive_agents=[_agent(1, "L1")],
        deaths=1,
        births=0,
        failed_births=1,
        cohort_size=2,
        patch_metrics={
            "patch_food_depleted_frac": 0.30,
            "patch_density_hotspot_frac": 0.30,
            "occupied_patch_waste_mean": 0.40,
            "jungle_predator_events": 2,
            "jungle_predator_damage_total": 0.5,
            "jungle_predator_deaths": 1,
            "wdl_delayed_hazard_event_count": 1,
            "wdl_delayed_hazard_damage_total": 0.2,
        },
        death_contexts=[{"agent_id": 2, "biological_lineage_id": "L2", "y": 2, "x": 3}],
        predator_contexts=[{"agent_id": 2, "y": 2, "x": 3}],
        wdl_contexts=[{"phase": "active"}],
    )
    resumed.finalize()

    assert uninterrupted.events == resumed.events
    assert len({event["event_id"] for event in resumed.events}) == len(resumed.events)
    assert {event["generation"] for event in resumed.events} == {0, 1}


def test_storage_preflight_rejects_unacknowledged_heavy_profile() -> None:
    config = ChronicleConfig(enabled=True, profile="scientific_full", allow_large_output=False)
    with pytest.raises(ValueError, match="requires allow_large_output"):
        config.validate()

    accepted = ChronicleConfig(enabled=True, profile="scientific_full", allow_large_output=True)
    result = storage_preflight(accepted, generations=10, planned_cells=2)
    assert result["accepted"] is True
    assert result["risk_classification"] == "large"


def test_native_typed_lifecycle_rows_remain_traceable(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path / "native", profile="scientific_full")
    recorder.observe_generation(
        generation=0,
        step=4,
        alive_agents=[_agent(1, "L00000007"), _agent(2, "L00000007"), _agent(3, "L00000007")],
        deaths=0,
        births=1,
        failed_births=1,
        cohort_size=3,
        patch_metrics={},
        native_lifecycle_rows=[
            {
                "event_type": 1,
                "generation": 0,
                "tick": 2,
                "sequence": 11,
                "parent_agent_id": 1,
                "child_agent_id": 3,
                "lineage_founder_id": 7,
                "reason_code": -1,
                "value": 0.25,
            },
            {
                "event_type": 3,
                "generation": 0,
                "tick": 3,
                "sequence": 12,
                "actor_agent_id": 2,
                "lineage_founder_id": 7,
                "reason_code": 1,
                "value": 1.0,
            },
        ],
    )
    recorder.finalize()

    events = read_jsonl(recorder.root / ARTIFACT_NAMES["events"])
    native_events = [
        event for event in events if event["source_component"] == "native_runtime.typed_lifecycle_events"
    ]
    assert [event["event_type"] for event in native_events] == [
        "birth_observation",
        "birth_failure_observation",
    ]
    assert native_events[0]["agent_ids"] == [3]
    assert native_events[0]["lineage_ids"] == ["L00000007"]
    assert native_events[0]["source_record_refs"] == [
        "metrics/native_typed_lifecycle_events.jsonl:g0:tick2:sequence11"
    ]


def test_native_aggregate_counts_override_stale_python_population_summary(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path / "native-aggregate")
    recorder.observe_generation(
        generation=0,
        step=7,
        alive_agents=[_agent(1, "L1")],
        deaths=0,
        births=6,
        failed_births=7,
        cohort_size=16,
        patch_metrics={},
        native_public_output_counts={
            "alive": 46,
            "birth_attempts": 6,
            "capacity": 96,
            "free_slots": 50,
        },
    )
    recorder.finalize()

    summary_event = next(event for event in recorder.events if event["event_type"] == "generation_summary")
    run_summary = read_json(recorder.root / ARTIFACT_NAMES["summary"])
    assert summary_event["measurements"]["alive"] == 46
    assert summary_event["measurements"]["active_lineages"] is None
    assert summary_event["measurements"]["native_lifecycle_authority"] is True
    assert run_summary["source_completeness"] == "live_generation_boundary_native_aggregate_lifecycle"
    assert "no typed lifecycle rows" in run_summary["limitations"][-1]


def test_regeneration_is_byte_idempotent_and_isolated(tmp_path: Path) -> None:
    source = _recorder(tmp_path / "source")
    _emit_two_generations(source)
    output_a = tmp_path / "regenerated-a"
    output_b = tmp_path / "regenerated-b"

    regenerate_chronicle(source.root.parent, output_a)
    regenerate_chronicle(source.root.parent, output_b)

    files_a = {path.name: path.read_bytes() for path in output_a.iterdir() if path.is_file()}
    files_b = {path.name: path.read_bytes() for path in output_b.iterdir() if path.is_file()}
    assert files_a == files_b
    assert (source.root / ARTIFACT_NAMES["events"]).exists()
    assert set(files_a) == set(ARTIFACT_NAMES.values())


def test_metrics_only_regeneration_records_incomplete_source_event(tmp_path: Path) -> None:
    source = tmp_path / "metrics-only"
    source.mkdir()
    (source / "metrics.csv").write_text(
        "generation,alive,patch_food_depleted_frac,jungle_predator_events\n0,3,0.4,1\n",
        encoding="utf-8",
    )

    result = regenerate_chronicle(source, tmp_path / "regenerated")
    events = read_jsonl(tmp_path / "regenerated" / ARTIFACT_NAMES["events"])

    assert result["source_completeness"] == "metrics_only_partial"
    assert any(event["event_type"] == "retrospective_source_incomplete" for event in events)
    assert set(path.name for path in (tmp_path / "regenerated").iterdir()) == set(ARTIFACT_NAMES.values())


def test_structured_comparison_reports_matches_and_divergence(tmp_path: Path) -> None:
    run_a = _recorder(tmp_path / "run-a", run_id="run-a")
    run_b = _recorder(tmp_path / "run-b", run_id="run-b")
    _emit_two_generations(run_a, divergent=False)
    _emit_two_generations(run_b, divergent=True)

    summary = compare_chronicles(run_a.root.parent, run_b.root.parent, tmp_path / "comparison")

    assert summary["matched_episode_count"] > 0
    assert summary["first_meaningful_divergence"]["status"] == "diverged"
    for name in (
        "comparison_summary.json",
        "comparison_report.md",
        "matched_episodes.json",
        "unmatched_episodes.json",
        "first_meaningful_divergence.json",
        "key_metric_context.json",
    ):
        assert (tmp_path / "comparison" / name).exists()


def test_comparison_recognizes_common_parent_checkpoint(tmp_path: Path) -> None:
    run_a = _recorder(tmp_path / "branch-a", run_id="branch-a")
    run_b = _recorder(tmp_path / "branch-b", run_id="branch-b")
    _emit_two_generations(run_a)
    _emit_two_generations(run_b, divergent=True)
    run_a.set_parent_checkpoint("checkpoint-shared")
    run_b.set_parent_checkpoint("checkpoint-shared")

    summary = compare_chronicles(run_a.root.parent, run_b.root.parent, tmp_path / "branch-comparison")

    assert summary["common_parent_checkpoint"] is True
    assert summary["parent_checkpoint_id"] == "checkpoint-shared"


def test_manifest_and_website_index_are_complete_and_stable(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path / "website")
    _emit_two_generations(recorder)

    manifest = read_json(recorder.root / ARTIFACT_NAMES["manifest"])
    index = read_json(recorder.root / ARTIFACT_NAMES["index"])
    storage = read_json(recorder.root / ARTIFACT_NAMES["storage"])

    assert manifest["source_commit"] != "unknown"
    assert manifest["config_hash"].startswith("sha256:")
    assert manifest["thresholds_frozen"] is True
    assert set(manifest["artifact_names"]) == set(ARTIFACT_NAMES)
    assert index["feeds"]["formal_events"] == ARTIFACT_NAMES["events"]
    assert {"event_type", "generation", "lineage_ids", "region_id"}.issubset(index["filter_fields"])
    assert {row["artifact"] for row in storage["artifacts"]} == {
        "events", "entries", "episodes", "contexts", "summary", "index"
    }


def test_checkpoint_metadata_mismatch_is_recorded_before_rejection(tmp_path: Path) -> None:
    recorder = _recorder(tmp_path / "mismatch")

    with pytest.raises(ValueError, match="config hash mismatch"):
        recorder.import_state(
            {
                "schema_version": "eiel_scientific_chronicle_state_v1",
                "config_hash": "sha256:not-the-run-config",
            }
        )

    events = read_jsonl(recorder.root / ARTIFACT_NAMES["events"])
    assert events[-1]["event_type"] == "checkpoint_chronicle_metadata_mismatch"
    assert events[-1]["measurements"]["reason"] == "config_hash_mismatch"
