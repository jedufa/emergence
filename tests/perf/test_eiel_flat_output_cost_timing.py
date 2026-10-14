from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.metrics_pkg.behavior import BehaviorLogger
from simulation.output_cost_timing import (
    CALL_COUNT_KEYS,
    PHASES,
    SCHEMA_VERSION,
    begin_flat_output_cost_timing,
    current_flat_output_cost_timing,
    finalize_flat_output_cost_timing,
    serialization_phase,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now

    def advance(self, nanoseconds: int) -> None:
        self.now += int(nanoseconds)


def _touch_required_phases(timing, clock: _Clock) -> None:
    for context in (
        timing.simulation_phase,
        timing.reporting_projection_phase,
        timing.reporting_reducer_phase,
        timing.serialization_phase,
        timing.write_phase,
        timing.flush_phase,
    ):
        with context():
            clock.advance(1)


def test_unset_sidecar_is_a_true_noop_without_reading_the_clock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("EIEL_FLAT_OUTPUT_COST_SIDECAR", raising=False)

    def forbidden_clock() -> int:
        raise AssertionError("disabled output-cost timing read perf_counter_ns")

    assert begin_flat_output_cost_timing(clock=forbidden_clock) is None
    assert current_flat_output_cost_timing() is None
    with serialization_phase():
        pass
    assert list(tmp_path.iterdir()) == []


def test_nested_phases_emit_exclusive_strict_sample(tmp_path: Path) -> None:
    clock = _Clock()
    run_dir = tmp_path / "run"
    sidecar = tmp_path / "sidecars" / "sample.json"
    timing = begin_flat_output_cost_timing(sidecar, clock=clock)
    assert timing is not None

    clock.advance(10)
    with timing.simulation_phase():
        clock.advance(10)
        with timing.reporting_projection_phase():
            clock.advance(30)
        clock.advance(20)
    with timing.reporting_reducer_phase():
        clock.advance(10)
    with timing.serialization_phase():
        clock.advance(10)
    with timing.write_phase():
        clock.advance(5)
        with timing.flush_phase():
            clock.advance(7)
        clock.advance(3)
    with timing.checkpoint_write_phase():
        clock.advance(4)
    clock.advance(15)

    payload = finalize_flat_output_cost_timing(timing, run_dir=run_dir)

    assert payload is not None
    assert set(payload) == {
        "schema_version",
        "exclusive_ns",
        "call_counts",
        "run_total_ns",
        "complete",
        "errors",
    }
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["exclusive_ns"] == {
        "simulation": 30,
        "reporting_projection": 30,
        "reporting_reducer": 10,
        "serialization": 10,
        "write": 12,
        "flush": 7,
    }
    assert payload["call_counts"] == {
        "simulation": 1,
        "reporting_projection": 1,
        "reporting_reducer": 1,
        "serialization": 1,
        "write": 2,
        "flush": 1,
    }
    assert payload["run_total_ns"] == 124
    assert payload["complete"] is True
    assert payload["errors"] == []
    assert json.loads(sidecar.read_text(encoding="utf-8")) == payload
    assert set(payload["exclusive_ns"]) == set(PHASES)
    assert set(payload["call_counts"]) == set(CALL_COUNT_KEYS)


def test_nested_same_phase_is_not_double_counted(tmp_path: Path) -> None:
    clock = _Clock()
    timing = begin_flat_output_cost_timing(tmp_path / "sample.json", clock=clock)
    assert timing is not None

    with timing.serialization_phase():
        clock.advance(2)
        with timing.serialization_phase():
            clock.advance(3)
        clock.advance(7)
    clock.advance(8)
    payload = finalize_flat_output_cost_timing(timing)

    assert payload is not None
    assert payload["exclusive_ns"]["serialization"] == 12
    assert payload["call_counts"]["serialization"] == 2
    assert payload["complete"] is False
    assert "missing_phase_timing:simulation,reporting_projection,reporting_reducer,write,flush" in payload["errors"]


def test_missing_explicit_simulation_stays_zero_and_incomplete(tmp_path: Path) -> None:
    clock = _Clock()
    timing = begin_flat_output_cost_timing(tmp_path / "sample.json", clock=clock)
    assert timing is not None

    for context in (
        timing.reporting_projection_phase,
        timing.reporting_reducer_phase,
        timing.serialization_phase,
        timing.write_phase,
        timing.flush_phase,
    ):
        with context():
            clock.advance(2)
    payload = finalize_flat_output_cost_timing(timing)

    assert payload is not None
    assert payload["exclusive_ns"]["simulation"] == 0
    assert payload["complete"] is False
    assert payload["errors"] == ["missing_phase_timing:simulation"]


def test_sidecar_inside_run_dir_is_rejected(tmp_path: Path) -> None:
    clock = _Clock()
    run_dir = tmp_path / "run"
    timing = begin_flat_output_cost_timing(run_dir / "sample.json", clock=clock)
    assert timing is not None
    _touch_required_phases(timing, clock)

    with pytest.raises(ValueError, match="must point outside run_dir"):
        finalize_flat_output_cost_timing(timing, run_dir=run_dir)
    assert not (run_dir / "sample.json").exists()
    assert current_flat_output_cost_timing() is None


def test_behavior_csv_bytes_match_when_timing_is_enabled(tmp_path: Path) -> None:
    action_ids = {
        "MOVE_N": 0,
        "MOVE_S": 1,
        "MOVE_E": 2,
        "MOVE_W": 3,
        "GATHER": 4,
        "REST": 5,
        "ATTACK": 6,
    }
    baseline = BehaviorLogger(tmp_path / "baseline", action_ids)
    baseline.start_generation(0, [7], injuries={7: 0.25})
    baseline.set_outcomes(7, injury_end=0.75, alive_end=1)
    baseline.end_generation()
    baseline._fh.close()

    clock = _Clock()
    sidecar = tmp_path / "sidecar.json"
    timing = begin_flat_output_cost_timing(sidecar, clock=clock)
    assert timing is not None
    instrumented = BehaviorLogger(tmp_path / "instrumented", action_ids)
    instrumented.start_generation(0, [7], injuries={7: 0.25})
    instrumented.set_outcomes(7, injury_end=0.75, alive_end=1)
    instrumented.end_generation()
    instrumented._fh.close()
    with timing.simulation_phase():
        clock.advance(1)
    with timing.reporting_projection_phase():
        clock.advance(1)
    with timing.reporting_reducer_phase():
        clock.advance(1)
    payload = finalize_flat_output_cost_timing(
        timing,
        run_dir=tmp_path / "instrumented",
    )

    assert (tmp_path / "baseline" / "behaviors.csv").read_bytes() == (
        tmp_path / "instrumented" / "behaviors.csv"
    ).read_bytes()
    assert payload is not None
    assert payload["complete"] is True
    assert payload["call_counts"]["serialization"] == 2
    assert payload["call_counts"]["write"] == 2
    assert payload["call_counts"]["flush"] == 2
