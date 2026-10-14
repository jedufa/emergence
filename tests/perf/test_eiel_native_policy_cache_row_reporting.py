from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from native_runtime.engine import _compact_raw_result
from native_runtime.public_metrics_overlay import native_policy_cache_metrics, public_metrics_overlay
from simulation.metrics_pkg.metrics_logger import MetricsLogger, _native_policy_cache_metrics_from_public_payloads
from simulation.metrics_pkg.rows import MetricsRow
from simulation.runner_pkg.runner_ticks_init import TickInitMixin
from simulation.runner_pkg.runner_ticks_loop import TickLoopMixin
from simulation.runnercore.tick_loop import _native_policy_cache_public_counts


REQUIRED_COUNTERS = (
    "policy_cache_rows_requested",
    "policy_cache_rows_hit",
    "policy_cache_rows_missed",
    "policy_cache_rows_refilled",
    "policy_cache_full_refill_ticks",
    "policy_cache_partial_refill_ticks",
)

EXPECTED_NATIVE_PARENT_GENOME_METRICS = {
    "gen_hidden_size_mean": 64.5,
    "gen_hidden_size_std": 0.5,
    "gen_exploration_bias_mean": 0.0,
    "gen_exploration_bias_std": 1.0,
    "gen_aggression_bias_mean": 0.0,
    "gen_aggression_bias_std": 2.0,
    "gen_reproduction_threshold_mean": 46.0,
    "gen_reproduction_threshold_std": 44.0,
    "gen_mutation_rate_mean": 0.1005,
    "gen_mutation_rate_std": 0.0995,
    "genome_diversity_norm_std_mean": 0.5,
}


def _native_public_reporting_fixture() -> tuple[dict[str, object], dict[str, object]]:
    def behavior_row(
        agent_id: int,
        cohort: str,
        *,
        alive_end: int,
        eligible_for_selection: int,
        kept_next: int,
    ) -> list[object]:
        return [
            7,
            agent_id,
            cohort,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            12,
            3.5,
            0,
            0,
            1.0,
            alive_end,
            eligible_for_selection,
            kept_next,
            0.0,
            0.0,
            0.0,
        ]

    raw: dict[str, object] = {
        "behavior_rows": [
            behavior_row(100, "start", alive_end=1, eligible_for_selection=1, kept_next=1),
            behavior_row(200, "born", alive_end=1, eligible_for_selection=0, kept_next=0),
            behavior_row(101, "start", alive_end=0, eligible_for_selection=0, kept_next=0),
            behavior_row(201, "born", alive_end=1, eligible_for_selection=0, kept_next=0),
            behavior_row(102, "start", alive_end=1, eligible_for_selection=1, kept_next=1),
        ],
    }
    projection: dict[str, object] = {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": {
            "world": "flat",
            "generation": 7,
            "capacity": 5,
            "alive_count": 4,
        },
        "tensors": {
            # Deliberately order slots differently from behavior rows. Only IDs
            # 100 and 102 are selected parents; born/live rows are decoys.
            "alive_mask": [True, True, True, True, False],
            "slot_to_agent_id": [201, 200, 100, 102, -1],
            "genome_hidden_size": [64, 65, 64, 65, 64],
            "genome_exploration_bias": [-0.25, 0.75, -1.0, 1.0, 0.0],
            "genome_aggression_bias": [-1.0, 1.0, -2.0, 2.0, 0.0],
            "genome_reproduction_threshold": [20.0, 70.0, 2.0, 90.0, 46.0],
            "genome_mutation_rate": [0.02, 0.15, 0.001, 0.2, 0.1],
            "pos_y": [0, 1, 0, 1, 0],
            "pos_x": [1, 2, 0, 0, 2],
            "survival_steps": [12, 12, 12, 12, 0],
            "biome_grid": [
                [0, 1, 0],
                [1, 0, 1],
            ],
        },
    }
    return raw, projection


def _microcycle_result() -> dict[str, object]:
    return {
        "schema_version": "cxxbackbone26a_flat_generation_lifecycle_microcycle_v1",
        "native_lifecycle_microcycles": True,
        "native_tick_loop_count": 3,
        "policy_loop_tick_calls": 3,
        "policy_refill_calls": 2,
        "policy_cache_hit_calls": 1,
        "microcycles": [
            {
                "policy_loop_tick_calls": 1,
                "policy_cache_rows_requested": 4,
                "policy_cache_rows_hit": 0,
                "policy_cache_rows_missed": 4,
                "policy_cache_rows_refilled": 4,
                "policy_cache_full_refill_ticks": 1,
                "policy_cache_partial_refill_ticks": 0,
            },
            {
                "policy_loop_tick_calls": 1,
                "policy_cache_rows_requested": 4,
                "policy_cache_rows_hit": 4,
                "policy_cache_rows_missed": 0,
                "policy_cache_rows_refilled": 0,
                "policy_cache_full_refill_ticks": 0,
                "policy_cache_partial_refill_ticks": 0,
            },
            {
                "policy_loop_tick_calls": 1,
                "policy_cache_rows_requested": 4,
                "policy_cache_rows_hit": 2,
                "policy_cache_rows_missed": 2,
                "policy_cache_rows_refilled": 2,
                "policy_cache_full_refill_ticks": 0,
                "policy_cache_partial_refill_ticks": 1,
            },
        ],
    }


def test_low_level_cache_service_aliases_become_canonical_row_metrics() -> None:
    raw = {
        "schema_version": "cxxbackbone26b_flat_action_cache_service_v1",
        "cache_hit_rows": 3,
        "policy_refill_rows": 2,
        "policy_refill_calls": 1,
    }

    compact = _compact_raw_result(raw)

    assert {name: compact[name] for name in REQUIRED_COUNTERS} == {
        "policy_cache_rows_requested": 5,
        "policy_cache_rows_hit": 3,
        "policy_cache_rows_missed": 2,
        "policy_cache_rows_refilled": 2,
        "policy_cache_full_refill_ticks": 0,
        "policy_cache_partial_refill_ticks": 1,
    }
    assert compact["policy_refill_rows"] == compact["policy_cache_rows_refilled"]
    assert compact["cache_hit_rows"] == compact["policy_cache_rows_hit"]
    assert compact["policy_refill_calls"] == 1


def test_one_tick_wrapper_propagates_nested_flat_generation_metrics() -> None:
    raw = {
        "schema_version": "cxxbackbone16_flat_generation_population_birth_commit_v1",
        "steps_per_generation": 1,
        "native_tick_loop_count": 1,
        "policy_refill_calls": 1,
        "flat_generation_core": {
            "schema_version": "cxxbackbone10_flat_generation_v1",
            "policy_loop_tick_calls": 1,
            "policy_cache_rows_requested": 4,
            "policy_cache_rows_hit": 0,
            "policy_cache_rows_missed": 4,
            "policy_cache_rows_refilled": 4,
            "policy_cache_full_refill_ticks": 1,
            "policy_cache_partial_refill_ticks": 0,
        },
    }

    compact = _compact_raw_result(raw)

    assert compact["policy_cache_rows_requested"] == 4
    assert compact["policy_cache_rows_hit"] == 0
    assert compact["policy_cache_rows_missed"] == compact["policy_cache_rows_refilled"] == 4
    assert compact["policy_refill_rows"] == 4
    assert compact["cache_hit_rows"] == 0
    assert compact["policy_refill_calls"] == 1
    assert compact["policy_cache_hit_calls"] == 0


def test_multi_tick_microcycle_metrics_sum_and_preserve_legacy_call_semantics() -> None:
    compact = _compact_raw_result(_microcycle_result())

    assert {name: compact[name] for name in REQUIRED_COUNTERS} == {
        "policy_cache_rows_requested": 12,
        "policy_cache_rows_hit": 6,
        "policy_cache_rows_missed": 6,
        "policy_cache_rows_refilled": 6,
        "policy_cache_full_refill_ticks": 1,
        "policy_cache_partial_refill_ticks": 1,
    }
    assert compact["policy_refill_rows"] == 6
    assert compact["cache_hit_rows"] == 6
    assert compact["policy_refill_calls"] == 2
    assert compact["policy_cache_hit_calls"] == 1
    assert compact["policy_loop_tick_calls"] == 3


def test_policy_cache_metrics_reach_public_overlay_tick_payload_and_logger_handoff() -> None:
    raw = _microcycle_result()
    compact = _compact_raw_result(raw)
    expected = native_policy_cache_metrics(raw, compact)

    overlay_values = public_metrics_overlay(raw, compact)
    tick_counts = _native_policy_cache_public_counts(raw, compact)
    logger_counts = _native_policy_cache_metrics_from_public_payloads(
        {"source": "native_compact_result", **tick_counts},
        {
            "source": "native_owner_checkpoint_projection",
            "values": overlay_values,
        },
    )

    assert tick_counts == expected
    assert logger_counts == expected
    assert {name: overlay_values[name] for name in REQUIRED_COUNTERS} == {
        name: expected[name] for name in REQUIRED_COUNTERS
    }


def test_metrics_logger_exposes_native_policy_cache_metrics_in_last_row(tmp_path: Path) -> None:
    raw = _microcycle_result()
    compact = _compact_raw_result(raw)
    overlay_values = public_metrics_overlay(raw, compact)
    tick_counts = _native_policy_cache_public_counts(raw, compact)
    logger = MetricsLogger(str(tmp_path))

    logger.log_generation(
        generation=0,
        agents=[],
        deaths_this_gen=0,
        native_public_output_counts={"source": "native_compact_result", **tick_counts},
        native_public_metrics_overlay={
            "source": "native_owner_checkpoint_projection",
            "values": overlay_values,
        },
    )

    assert logger.last_native_policy_cache_metrics == tick_counts
    assert logger.last_row is not None
    assert {name: logger.last_row[name] for name in REQUIRED_COUNTERS} == {
        name: tick_counts[name] for name in REQUIRED_COUNTERS
    }
    assert logger.last_policy_cache_metric_source == "native_compact_result"
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        emitted = list(csv.DictReader(stream))
    assert len(emitted) == 1
    assert {name: int(emitted[0][name]) for name in REQUIRED_COUNTERS} == {
        name: tick_counts[name] for name in REQUIRED_COUNTERS
    }


def test_policy_cache_columns_are_append_only_canonical_metrics_fields() -> None:
    fields = tuple(MetricsRow.__annotations__)

    assert fields[-len(REQUIRED_COUNTERS) :] == REQUIRED_COUNTERS


def test_reference_policy_cache_metrics_are_public_without_native_mislabel(tmp_path: Path) -> None:
    counts = {
        "source": "reference_policy_cache_service",
        "policy_cache_rows_requested": 12,
        "policy_cache_rows_hit": 6,
        "policy_cache_rows_missed": 6,
        "policy_cache_rows_refilled": 6,
        "policy_cache_full_refill_ticks": 1,
        "policy_cache_partial_refill_ticks": 1,
    }
    logger = MetricsLogger(str(tmp_path))

    logger.log_generation(
        generation=0,
        agents=[],
        deaths_this_gen=0,
        policy_cache_public_counts=counts,
    )

    assert logger.last_policy_cache_metric_source == "reference_policy_cache_service"
    assert {name: logger.last_policy_cache_metrics[name] for name in REQUIRED_COUNTERS} == {
        name: counts[name] for name in REQUIRED_COUNTERS
    }
    assert logger.last_native_policy_cache_metrics == {}
    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        emitted = list(csv.DictReader(stream))
    assert {name: int(emitted[0][name]) for name in REQUIRED_COUNTERS} == {
        name: counts[name] for name in REQUIRED_COUNTERS
    }


def test_reference_policy_cache_service_accumulates_and_resets_per_generation() -> None:
    class Counter(TickInitMixin, TickLoopMixin):
        pass

    counter = Counter()
    counter._init_generation_tick_state([])
    counter._record_reference_policy_cache_rows(requested=4, hit=0, missed=4)
    counter._record_reference_policy_cache_rows(requested=4, hit=4, missed=0)
    counter._record_reference_policy_cache_rows(requested=4, hit=2, missed=2)

    assert counter._reference_policy_cache_public_counts == {
        "source": "reference_policy_cache_service",
        "policy_cache_rows_requested": 12,
        "policy_cache_rows_hit": 6,
        "policy_cache_rows_missed": 6,
        "policy_cache_rows_refilled": 6,
        "policy_cache_full_refill_ticks": 1,
        "policy_cache_partial_refill_ticks": 1,
    }

    counter._init_generation_tick_state([])
    assert {name: counter._reference_policy_cache_public_counts[name] for name in REQUIRED_COUNTERS} == {
        name: 0 for name in REQUIRED_COUNTERS
    }


def test_reference_policy_cache_service_rejects_inconsistent_rows() -> None:
    class Counter(TickLoopMixin):
        pass

    with pytest.raises(ValueError, match=r"requested=hit\+missed"):
        Counter()._record_reference_policy_cache_rows(requested=4, hit=3, missed=0)


def test_policy_cache_public_source_fails_closed(tmp_path: Path) -> None:
    logger = MetricsLogger(str(tmp_path))

    with pytest.raises(ValueError, match="unsupported policy-cache metric source"):
        logger.log_generation(
            generation=0,
            agents=[],
            deaths_this_gen=0,
            policy_cache_public_counts={"source": "guessed_cache_owner"},
        )


def test_native_overlay_derives_start_cohort_survival_from_behavior_rows() -> None:
    raw, _projection = _native_public_reporting_fixture()

    overlay = public_metrics_overlay(raw)

    assert overlay["cohort_size"] == 3
    assert overlay["cohort_survivors"] == 2
    assert overlay["cohort_survival_rate"] == pytest.approx(2.0 / 3.0, rel=0.0, abs=1.0e-15)


def test_native_overlay_derives_genome_metrics_from_kept_parent_projection_rows() -> None:
    raw, projection = _native_public_reporting_fixture()

    overlay = public_metrics_overlay(raw, checkpoint_projection=projection)

    assert {name: overlay[name] for name in EXPECTED_NATIVE_PARENT_GENOME_METRICS} == pytest.approx(
        EXPECTED_NATIVE_PARENT_GENOME_METRICS,
        rel=0.0,
        abs=1.0e-12,
    )


def test_metrics_logger_uses_native_overlay_for_genome_values_and_collapse_state(tmp_path: Path) -> None:
    raw, projection = _native_public_reporting_fixture()
    overlay = public_metrics_overlay(raw, checkpoint_projection=projection)
    logger = MetricsLogger(str(tmp_path))
    logger.restore_state({"schema_version": "metrics_state_v1", "collapse_streak": 2})

    logger.log_generation(
        generation=0,
        agents=[],
        selected_parent_agents=[],
        deaths_this_gen=1,
        sim_cfg=SimpleNamespace(
            diversity_collapse_threshold=0.25,
            diversity_collapse_streak_required=2,
        ),
        native_public_metrics_overlay={
            "source": "native_owner_checkpoint_projection",
            "values": overlay,
        },
    )

    assert logger.last_row is not None
    assert logger.last_row["alive"] == 4
    assert logger.last_row["cohort_size"] == 3
    assert logger.last_row["cohort_survivors"] == 2
    assert logger.last_row["cohort_survival_rate"] == pytest.approx(2.0 / 3.0, rel=0.0, abs=1.0e-15)
    assert logger.last_row["biome_safe_frac_alive"] == 0.25
    assert logger.last_row["biome_wild_frac_alive"] == 0.75
    assert {
        name: logger.last_row[name] for name in EXPECTED_NATIVE_PARENT_GENOME_METRICS
    } == pytest.approx(EXPECTED_NATIVE_PARENT_GENOME_METRICS, rel=0.0, abs=1.0e-12)

    # A two-generation empty Python parent mirror would cross the configured
    # collapse threshold here. The diverse authoritative native pool resets it.
    assert logger.last_row["diversity_collapse"] == 0
    assert logger.last_row["diversity_collapse_streak"] == 0
    assert logger.export_state()["collapse_streak"] == 0

    with (tmp_path / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        emitted_rows = list(csv.DictReader(stream))
    assert len(emitted_rows) == 1
    emitted = emitted_rows[0]
    assert int(emitted["cohort_size"]) == 3
    assert int(emitted["cohort_survivors"]) == 2
    assert float(emitted["cohort_survival_rate"]) == pytest.approx(2.0 / 3.0, rel=0.0, abs=1.0e-15)
    assert float(emitted["biome_safe_frac_alive"]) == 0.25
    assert float(emitted["biome_wild_frac_alive"]) == 0.75
    assert int(emitted["diversity_collapse"]) == 0
    assert int(emitted["diversity_collapse_streak"]) == 0
    for name, expected in EXPECTED_NATIVE_PARENT_GENOME_METRICS.items():
        assert float(emitted[name]) == pytest.approx(expected, rel=0.0, abs=1.0e-12)


def test_native_overlay_derives_flat_biome_fractions_from_live_projection_positions() -> None:
    raw, projection = _native_public_reporting_fixture()

    overlay = public_metrics_overlay(raw, checkpoint_projection=projection)

    assert overlay["biome_safe_frac_alive"] == 0.25
    assert overlay["biome_wild_frac_alive"] == 0.75


@pytest.mark.parametrize(
    "raw, message",
    [
        (
            {
                "policy_cache_rows_requested": 7,
                "policy_cache_rows_hit": 3,
                "policy_cache_rows_missed": 3,
                "policy_cache_rows_refilled": 3,
            },
            "requested=hit\\+miss",
        ),
        (
            {
                "policy_cache_rows_requested": 6,
                "policy_cache_rows_hit": 3,
                "policy_cache_rows_missed": 3,
                "policy_cache_rows_refilled": 2,
            },
            "missed=refilled",
        ),
        (
            {
                "policy_loop_tick_calls": 3,
                "policy_refill_calls": 1,
                "policy_cache_full_refill_ticks": 1,
                "policy_cache_partial_refill_ticks": 1,
            },
            "policy_refill_calls=full\\+partial",
        ),
    ],
)
def test_inconsistent_policy_cache_metrics_fail_closed(raw: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _compact_raw_result(raw)
