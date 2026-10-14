from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import numpy as np
import pytest

from engine.native_runtime import compact_summary_cpu as cpu
from engine.native_runtime import compact_summary_digests as digests
from engine.native_runtime import compact_summary_planes as planes
from engine.native_runtime import compact_summary_projection as projection


SHA = "a" * 64


def _behavior_row(agent_id: int, *, cohort: str, alive_end: int) -> dict[str, Any]:
    return {
        "generation": 3,
        "agent_id": agent_id,
        "cohort": cohort,
        "move_n": agent_id,
        "move_s": 1,
        "move_e": 0,
        "move_w": 1,
        "gather": 2,
        "rest": 1,
        "attack": 0,
        "lifespan_steps": 80,
        "energy_collected": 3.5,
        "offspring": 0,
        "attacks_made": 0,
        "fitness": 8.25,
        "alive_end": alive_end,
        "eligible_for_selection": 1,
        "kept_next": alive_end,
        "injury_end": 0.5,
        "injury_delta": -0.25,
        "aggression_bias": 0.125,
    }


def _lineage_record(agent_id: int) -> dict[str, Any]:
    return {
        "agent_id": agent_id,
        "energy": 14.5,
        "health": 0.75,
        "injury": 0.25,
        "drives": [0.1, 0.2, 0.3, 0.4],
        "y": 2,
        "x": 3,
        "stats": {
            "lineage_id": f"L{agent_id:08d}",
            "founder_id": agent_id,
            "parent_ids": [],
            "parent_lineage_ids": [],
            "birth_generation": 0,
            "birth_tick": 0,
            "lineage_depth": 0,
            "offspring_count": 1,
            "survival_steps": 80.0,
            "energy_collected": 5.0,
            "first_repro_step": 24,
            "pred_err_count": 3,
            "pred_err_sum": 0.75,
            "episode0_pred_err_mean": 0.25,
            "identity_drift_ema": 0.0,
            "drive_jitter": 0.1,
            "ecological_role": "general_agent",
            "role_source": "default",
            "role_confidence": None,
            "patch_habitat_lineage_id": agent_id,
        },
        "genome": {
            "hidden_size": 64,
            "memory_dim": 16,
            "use_gru": False,
            "aggression_bias": 0.125,
            "exploration_bias": -0.25,
        },
    }


def _policy_tape() -> dict[str, Any]:
    return {
        "policy_birth_crossover_mask_tape": np.array(
            [[False, True], [True, False]], dtype=np.bool_
        ),
        "policy_birth_delta_tape": np.array(
            [[0.1, -0.2], [9.0, 10.0]], dtype=np.float32
        ),
        "policy_birth_mutation_mask_tape": np.array(
            [[True, True], [False, False]], dtype=np.bool_
        ),
        "policy_birth_post_rng_state_tape": np.array(
            [[1, 2, 3], [90, 91, 92]], dtype=np.uint8
        ),
        "policy_birth_semantic_draws_tape": np.array([7, 99], dtype=np.int64),
        "policy_birth_tape_cursor": np.array([1], dtype=np.int64),
    }


def _patch_metrics() -> dict[str, Any]:
    return {
        "patch_movement_energy_event_count": 4,
        "patch_spawn_sync_rows": 2,
        "typed_death_event_rows": 1,
        "env_rng_padding_draws": 0,
        "patch_food_harvest_total": 5.0,
        "patch_scent_emit_total": 2.0,
        "patch_waste_emit_total": 1.0,
        "patch_movement_energy_total": 0.5,
        "energy_collected_delta": 5.0,
        "patch_scent_move_queued_total": 1.0,
        "patch_waste_move_queued_total": 0.5,
        "patch_scent_gather_queued_total": 1.0,
        "patch_waste_gather_queued_total": 0.5,
        "patch_movement_energy_extra_total": 0.5,
        "patch_scent_emitted_total": 2.0,
        "patch_waste_emitted_total": 1.0,
        "patch_food_depleted_frac": 0.25,
        "patch_food_full_frac": 0.5,
        "patch_density_hotspot_frac": 0.125,
    }


def _source(*, world: str = "flat") -> dict[str, Any]:
    behavior_rows = [
        _behavior_row(1, cohort="start", alive_end=1),
        _behavior_row(2, cohort="born", alive_end=0),
    ]
    return {
        "schema_version": cpu.SOURCE_SCHEMA_VERSION,
        "pre_turnover": {
            "world": world,
            "generation_start": {"generation": 3, "tick": 0},
            "generation_end": {"generation": 3, "tick_exclusive": 80},
            "alive": 6,
            "births": 2,
            "failed_births": 1,
            "deaths": 1,
            "deaths_by_class": {
                "energy_depleted": 1,
                "health_depleted": 0,
            },
            "event_cursor_start": 10,
            "event_cursor_pre_turnover": 14,
            "generation_event_count": 4,
            "generation_counts_by_type": {
                "birth_committed": 2,
                "mutation": 0,
                "birth_failed": 1,
                "death_committed": 1,
            },
            "generation_metrics": {
                "capacity": 10,
                "free_slots": 4,
                "birth_attempts": 3,
                "native_failed_birth_event_rows": 1,
                "injury_blocked_births": 0,
                "moved": 4,
                "gathered": 2,
                "movement_attempts": 8,
                "movement_blocked_occupied": 3,
                "movement_blocked_bounds": 1,
                "peak_injury": 0.5,
                "policy_loop_tick_calls": 80,
                "policy_refill_calls": 2,
                "policy_refill_rows": 5,
                "policy_cache_hit_calls": 78,
                "policy_cache_rows_requested": 20,
                "policy_cache_rows_hit": 15,
                "policy_cache_rows_missed": 5,
                "policy_cache_rows_refilled": 5,
                "policy_cache_full_refill_ticks": 1,
                "policy_cache_partial_refill_ticks": 1,
            },
        },
        "post_turnover": {
            "next_generation_start": {"generation": 4, "tick": 0},
            "turnover": {"count": 1, "status": "committed"},
            "event_cursor_end": 16,
            "authoritative_event_cursor": 16,
            "event_row_count": 6,
            "turnover_event_count": 2,
            "turnover_counts_by_type": {
                "birth_committed": 0,
                "mutation": 2,
                "birth_failed": 0,
                "death_committed": 0,
            },
            "identity": {
                "population_next_id": 10,
                "active_agent_count": 8,
                "active_agent_id_min": 1,
                "active_agent_id_max": 9,
                "free_slot_count": 2,
            },
            "active_agent_ids": np.array(
                [1, 2, 3, 4, 5, 6, 8, 9], dtype=np.int64
            ),
        },
        "settled": {
            "summary_valid": True,
            "summary_status": "complete",
            "terminal": {
                "status": "not_terminal",
                "recoverable_error_status": "none",
                "recoverable_error_code": None,
                "recoverable_error_details": None,
                "sticky_failure": False,
                "failure_generation": None,
                "failure_tick": None,
                "failure_stage": None,
            },
        },
        "schedule": {
            "schedule_origin_generation": 0,
            "explicit_next_generations": [],
            "cancellation_due": False,
        },
        "patch": None if world == "flat" else _patch_metrics(),
        "behavior": {
            "rows": behavior_rows,
            "row_count": 2,
            "cohort_size": 1,
            "cohort_survivors": 1,
            "action_total": 13,
            "move_total": 7,
            "move_n": 3,
            "move_s": 2,
            "move_e": 0,
            "move_w": 2,
            "gather": 4,
            "rest": 2,
            "attack": 0,
        },
        "lineage": {
            "generation": 3,
            "records": [_lineage_record(1), _lineage_record(2)],
            "record_count": 2,
            "active_lineages": 2,
            "dominant_lineage_id": "L00000001",
            "dominant_lineage_population_share": 0.5,
            "top3_lineage_population_share": 1.0,
            "lineage_population_entropy": 0.9,
            "lineage_reproduction_entropy": 0.8,
            "dominant_lineage_reproduction_share": 0.5,
            "rare_persistent_lineages": 1,
            "lineage_collapse_warning": False,
            "lineage_collapse_streak": 0,
        },
        "rng": {
            "authority_digest_sha256": SHA,
            "policy_mutation_tape": _policy_tape(),
            "authority_complete": True,
            "export_non_consuming": True,
        },
        "digests": {
            "events_active_prefix_digest_sha256": SHA,
            "authoritative_state_digest_sha256": SHA,
            "canonical_checkpoint_storage_digest_sha256": SHA,
            "next_generation_start_state_digest_sha256": SHA,
        },
    }


def _assert_source_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        lhs = left[key]
        rhs = right[key]
        if isinstance(lhs, np.ndarray):
            assert isinstance(rhs, np.ndarray)
            assert np.array_equal(lhs, rhs, equal_nan=True)
        elif isinstance(lhs, dict):
            assert isinstance(rhs, dict)
            _assert_source_equal(lhs, rhs)
        elif isinstance(lhs, list):
            assert lhs == rhs
        else:
            assert lhs == rhs


@pytest.mark.parametrize("world", ["flat", "patch"])
def test_flat_and_patch_fixtures_build_exact_valid_rows(world: str) -> None:
    source = _source(world=world)
    row = cpu.build_compact_summary_row(source)

    receipt = cpu.validate_compact_summary_row(row)

    assert receipt["ok"] is True
    assert row["world"] == world
    assert (row["patch_metrics"] is None) == (world == "flat")
    assert row["behavior_metrics"]["rows_digest_sha256"] == (
        digests.behavior_rows_digest_sha256(source["behavior"]["rows"])
    )
    assert row["lineage_metrics"]["records_digest_sha256"] == (
        digests.lineage_records_digest_sha256(
            3, source["lineage"]["records"]
        )
    )
    assert row["identity"]["active_agent_ids_digest_sha256"] == (
        digests.active_agent_ids_digest_sha256(
            source["post_turnover"]["active_agent_ids"]
        )
    )
    assert row["rng"]["policy_mutation_tape_digest_sha256"] == (
        digests.policy_mutation_tape_digest_sha256(
            source["rng"]["policy_mutation_tape"]
        )
    )


def test_source_is_read_only_and_same_input_is_deterministic() -> None:
    source = _source(world="patch")
    before = copy.deepcopy(source)

    row_one = cpu.build_compact_summary_row(source)
    row_before_encoding = copy.deepcopy(row_one)
    bytes_one = cpu.compact_summary_canonical_json_bytes(row_one)
    row_two = cpu.build_compact_summary_cpu(source)
    bytes_two = cpu.build_compact_summary_json_bytes(source)

    _assert_source_equal(source, before)
    assert row_one == row_before_encoding
    assert row_one == row_two
    assert bytes_one == bytes_two
    assert bytes_one == (
        json.dumps(
            row_one,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert bytes_one.endswith(b"\n")
    assert not bytes_one.endswith(b"\n\n")


def test_typed_envelope_and_mapping_have_identical_output() -> None:
    source = _source()
    typed = cpu.CompactSummarySourceEnvelope.from_mapping(source)

    assert cpu.build_canonical_row(typed) == cpu.build_canonical_row(source)


def test_every_and_only_null_path_is_sorted() -> None:
    source = _source()
    source["lineage"].update(
        active_lineages=None,
        dominant_lineage_id=None,
        dominant_lineage_population_share=None,
        top3_lineage_population_share=None,
        lineage_population_entropy=None,
        lineage_reproduction_entropy=None,
        dominant_lineage_reproduction_share=None,
        rare_persistent_lineages=None,
        lineage_collapse_warning=None,
        lineage_collapse_streak=None,
    )

    row = cpu.build_compact_summary_row(source)

    expected = [
        "lineage_metrics.active_lineages",
        "lineage_metrics.dominant_lineage_id",
        "lineage_metrics.dominant_lineage_population_share",
        "lineage_metrics.dominant_lineage_reproduction_share",
        "lineage_metrics.lineage_collapse_streak",
        "lineage_metrics.lineage_collapse_warning",
        "lineage_metrics.lineage_population_entropy",
        "lineage_metrics.lineage_reproduction_entropy",
        "lineage_metrics.rare_persistent_lineages",
        "lineage_metrics.top3_lineage_population_share",
        "patch_metrics",
        "terminal.failure_generation",
        "terminal.failure_stage",
        "terminal.failure_tick",
        "terminal.recoverable_error_code",
        "terminal.recoverable_error_digest_sha256",
    ]
    assert row["undefined_fields"] == sorted(expected)


def test_signed_zero_is_preserved_in_row_and_canonical_bytes() -> None:
    source = _source()
    source["pre_turnover"]["generation_metrics"]["peak_injury"] = -0.0

    row = cpu.build_compact_summary_row(source)
    encoded = cpu.canonical_json_bytes(row)

    assert math.copysign(1.0, row["generation_metrics"]["peak_injury"]) == -1.0
    assert b'"peak_injury":-0.0' in encoded


def test_generation_cell_matrix_projection_is_exact_and_rectangular() -> None:
    flat = cpu.build_compact_summary_row(_source(world="flat"))
    patch = cpu.build_compact_summary_row(_source(world="patch"))
    rows = [[flat, patch], [copy.deepcopy(flat), copy.deepcopy(patch)]]

    encoded = projection.encode_compact_summary_matrix(rows)

    assert planes.validate_compact_summary_planes(encoded) == (2, 2)
    assert projection.project_compact_summary_matrix(encoded) == rows
    assert encoded.validity.tolist() == [[1, 1], [1, 1]]


def test_matrix_encoding_rejects_ragged_or_invalid_rows_without_target() -> None:
    flat = cpu.build_compact_summary_row(_source(world="flat"))
    invalid = copy.deepcopy(flat)
    invalid["summary_valid"] = False

    with pytest.raises(ValueError, match="rectangular"):
        projection.encode_compact_summary_matrix([[flat], [flat, flat]])
    with pytest.raises(ValueError, match="only complete valid rows"):
        projection.encode_compact_summary_matrix([[flat], [invalid]])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda source: source.pop("behavior"), "missing=.*behavior"),
        (
            lambda source: source["pre_turnover"].update(cuda_summary=[1.0]),
            "extra=.*cuda_summary",
        ),
        (
            lambda source: source["behavior"].pop("rows"),
            "missing=.*rows",
        ),
        (
            lambda source: source["digests"].update(
                authoritative_state_digest_sha256="A" * 64
            ),
            "lowercase SHA-256",
        ),
        (
            lambda source: source["post_turnover"]["identity"].update(
                active_agent_count=7
            ),
            "contradicts active_agent_ids",
        ),
        (
            lambda source: source["behavior"].update(action_total=12),
            "contradicts behavior rows",
        ),
    ],
)
def test_missing_malformed_and_cuda_derived_sources_fail_closed(
    mutate: Any, message: str
) -> None:
    source = _source()
    mutate(source)

    with pytest.raises(cpu.CompactSummarySourceError, match=message):
        cpu.build_compact_summary_row(source)


def test_malformed_rng_tape_and_nonfinite_values_fail_closed() -> None:
    source = _source()
    source["rng"]["policy_mutation_tape"]["policy_birth_tape_cursor"][0] = 3
    with pytest.raises(cpu.CompactSummarySourceError, match="exceeds tape capacity"):
        cpu.build_compact_summary_row(source)

    source = _source()
    source["pre_turnover"]["generation_metrics"]["peak_injury"] = float("nan")
    with pytest.raises(ValueError, match="JSON compliant|finite"):
        cpu.build_compact_summary_row(source)


def test_recoverable_error_digest_uses_strict_canonical_preimage() -> None:
    source = _source()
    source["settled"]["terminal"].update(
        recoverable_error_status="present",
        recoverable_error_code="capacity_warning",
        recoverable_error_details={"remaining": 2, "ratio": -0.0},
    )

    row = cpu.build_compact_summary_row(source)
    expected_preimage = {
        "schema_version": cpu.RECOVERABLE_ERROR_SCHEMA_VERSION,
        "code": "capacity_warning",
        "details": {"remaining": 2, "ratio": -0.0},
    }
    expected = hashlib.sha256(
        json.dumps(
            expected_preimage,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert row["terminal"]["recoverable_error_digest_sha256"] == expected
    assert "terminal.recoverable_error_code" not in row["undefined_fields"]
    assert (
        "terminal.recoverable_error_digest_sha256"
        not in row["undefined_fields"]
    )


def test_schedule_uses_original_origin_explicit_and_cancellation_authority() -> None:
    source = _source()
    source["schedule"].update(
        schedule_origin_generation=5,
        explicit_next_generations=[13],
    )
    source["pre_turnover"]["generation_start"]["generation"] = 12
    source["pre_turnover"]["generation_end"]["generation"] = 12
    source["post_turnover"]["next_generation_start"]["generation"] = 13
    source["lineage"]["generation"] = 12
    for row in source["behavior"]["rows"]:
        row["generation"] = 12
    assert cpu.build_compact_summary_row(source)["checkpoint_due"] is True

    cancelled = _source()
    cancelled["settled"]["terminal"]["status"] = "cancelled"
    cancelled["schedule"]["cancellation_due"] = True
    assert cpu.build_compact_summary_row(cancelled)["checkpoint_due"] is True


def test_patch_authority_is_required_only_for_patch_world() -> None:
    patch = _source(world="patch")
    patch["patch"] = None
    with pytest.raises(cpu.CompactSummarySourceError, match="source.patch"):
        cpu.build_compact_summary_row(patch)

    flat = _source()
    flat["patch"] = _patch_metrics()
    with pytest.raises(cpu.CompactSummarySourceError, match="must be null"):
        cpu.build_compact_summary_row(flat)


@pytest.mark.parametrize("world", ["flat", "patch"])
def test_cpu_row_round_trips_through_exact_abi_v2_planes(world: str) -> None:
    row = cpu.build_compact_summary_row(_source(world=world))
    storage = planes.allocate_compact_summary_planes(2, 1)

    projection.encode_compact_summary_row(
        row, storage, generation=1, cell=0
    )
    projected = projection.project_compact_summary_row(
        storage, generation=1, cell=0
    )

    assert projected == row
    assert projection.canonical_compact_summary_bytes(projected) == (
        cpu.canonical_json_bytes(row)
    )
    with pytest.raises(ValueError, match="invalid/no-op"):
        projection.project_compact_summary_row(storage, generation=0, cell=0)


def test_plane_encoding_failure_is_atomic_and_keeps_invalid_noop_row() -> None:
    source = _source()
    source["settled"]["terminal"].update(
        recoverable_error_status="present",
        recoverable_error_code="schema_valid_but_not_native_codebook",
        recoverable_error_details={"bounded": True},
    )
    row = cpu.build_compact_summary_row(source)
    storage = planes.allocate_compact_summary_planes(1, 1)

    with pytest.raises(ValueError, match="unsupported"):
        projection.encode_compact_summary_row(
            row, storage, generation=0, cell=0
        )

    assert storage.validity[0, 0] == 0
    assert all(
        not np.any(array)
        for array in storage.as_mapping().values()
    )


def test_plane_projection_preserves_negative_zero_bits() -> None:
    source = _source()
    source["pre_turnover"]["generation_metrics"]["peak_injury"] = -0.0
    row = cpu.build_compact_summary_row(source)

    projected = projection.round_trip_compact_summary_row(row)

    value = projected["generation_metrics"]["peak_injury"]
    assert math.copysign(1.0, value) == -1.0
    assert b'"peak_injury":-0.0' in projection.canonical_compact_summary_bytes(
        projected
    )
