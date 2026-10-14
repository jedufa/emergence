from __future__ import annotations

from native_runtime.flat_population_reporting import native_flat_population_reporting_records
from native_runtime.public_metrics_overlay import (
    NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS,
    public_metrics_overlay,
)
from simulation.metrics_pkg.cognitive_status import (
    compute_cognitive_status_metrics_from_records,
)
from simulation.output_cost_timing import (
    begin_flat_output_cost_timing,
    finalize_flat_output_cost_timing,
)
from tests.perf.test_eiel_native_flat_reporting_projection import (
    _behavior_rows,
    _configs,
    _owner_projection,
)


def test_native_overlay_publishes_complete_shared_reducer_surface() -> None:
    projection = _owner_projection([30, 10, 20])
    projection["tensors"]["offspring_count"][0] = 2
    behavior_rows = _behavior_rows([30, 10, 20], selected_ids={20, 30})
    env_cfg, sim_cfg = _configs()
    sim_cfg.lineage_observability_enable = False

    records = native_flat_population_reporting_records(
        tensors=projection["tensors"],
        behavior_rows=behavior_rows,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metadata=projection["metadata"],
    )
    reduced = compute_cognitive_status_metrics_from_records(
        records,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        generation=4,
    )

    native_counters = {
        "birth_attempt_rows": 13,
        "committed_birth_rows": 8,
        "public_failed_birth_rows": 5,
        "injury_blocked_births": 3,
    }
    overlay = public_metrics_overlay(
        {"behavior_rows": behavior_rows, **native_counters},
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )

    assert len(reduced) > len(NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS) == 66
    assert "juvenile_reproduction_attempt_rate" not in NATIVE_FLAT_RESIDENT_REPORTING_COLUMNS
    assert reduced["juvenile_reproduction_attempt_rate"] == 1.0
    assert {key: overlay[key] for key in reduced} == dict(reduced)
    assert {
        "birth_attempts": overlay["birth_attempts"],
        "successful_births": overlay["successful_births"],
        "failed_births": overlay["failed_births"],
        "injury_blocked_births": overlay["injury_blocked_births"],
    } == {
        "birth_attempts": 13,
        "successful_births": 8,
        "failed_births": 5,
        "injury_blocked_births": 3,
    }


def test_native_reporting_entry_point_records_projection_and_reducer_timing(tmp_path) -> None:
    projection = _owner_projection([30, 10, 20])
    behavior_rows = _behavior_rows([30, 10, 20], selected_ids={20, 30})
    env_cfg, sim_cfg = _configs()
    sim_cfg.lineage_observability_enable = False
    timing = begin_flat_output_cost_timing(tmp_path / "native-output-cost.json")
    assert timing is not None

    public_metrics_overlay(
        {"behavior_rows": behavior_rows},
        checkpoint_projection=projection,
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        metrics_generation=4,
    )
    for context in (
        timing.simulation_phase,
        timing.serialization_phase,
        timing.write_phase,
        timing.flush_phase,
    ):
        with context():
            pass
    payload = finalize_flat_output_cost_timing(timing)

    assert payload is not None
    assert payload["complete"] is True
    assert payload["call_counts"]["reporting_projection"] == 1
    assert payload["call_counts"]["reporting_reducer"] == 1
