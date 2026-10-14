from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from engine.native_runtime.checkpoint_digests import (
    DigestLayer,
    checkpoint_digest_manifest,
    checkpoint_source_projection_digest_manifest,
)
from engine.native_runtime.compact_summary_authority_adapter import (
    CompactSummaryAuthorityAdapterError,
    CompactSummaryAuthorityInputs,
    assemble_compact_summary_source,
)
from engine.native_runtime.compact_summary_cpu import build_compact_summary_row
from engine.native_runtime.compact_summary_digests import (
    checkpoint_rng_digest_from_tensor_manifest,
    checkpoint_source_projection_digest_from_tensor_manifest,
)
from engine.native_runtime.lifecycle_events import FIELD_DTYPES
from engine.native_runtime.types import (
    NativeEngineCounters,
    NativeGenerationResult,
    NativeRuntimeCapability,
)
from simulation.metrics_pkg.lineage import LineageMetricsState


def _behavior_row(*, generation: int = 0) -> dict[str, Any]:
    return {
        "generation": generation,
        "agent_id": 1,
        "cohort": "start",
        "move_n": 0,
        "move_s": 0,
        "move_e": 0,
        "move_w": 0,
        "gather": 0,
        "rest": 0,
        "attack": 0,
        "lifespan_steps": 1,
        "energy_collected": 0.0,
        "offspring": 0,
        "attacks_made": 0,
        "fitness": 1.0,
        "alive_end": 1,
        "eligible_for_selection": 1,
        "kept_next": 1,
        "injury_end": 0.0,
        "injury_delta": 0.0,
        "aggression_bias": 0.0,
    }


def _lineage_record() -> dict[str, Any]:
    return {
        "agent_id": 1,
        "energy": 10.0,
        "health": 1.0,
        "injury": 0.0,
        "drives": [0.1, 0.2, 0.3, 0.4],
        "y": 0,
        "x": 0,
        "stats": {
            "lineage_id": "L00000001",
            "founder_id": 1,
            "parent_ids": [],
            "parent_lineage_ids": [],
            "birth_generation": 0,
            "birth_tick": 0,
            "lineage_depth": 0,
            "offspring_count": 0,
            "survival_steps": 1.0,
            "energy_collected": 0.0,
            "first_repro_step": None,
            "pred_err_count": 0,
            "pred_err_sum": 0.0,
            "episode0_pred_err_mean": 0.0,
            "identity_drift_ema": 0.0,
            "drive_jitter": 0.0,
            "ecological_role": "general_agent",
            "role_source": "default",
            "role_confidence": None,
            "patch_habitat_lineage_id": 1,
        },
        "genome": {
            "hidden_size": 64,
            "memory_dim": 16,
            "use_gru": False,
            "aggression_bias": 0.0,
            "exploration_bias": 0.0,
        },
    }


def _policy_tape() -> dict[str, np.ndarray]:
    return {
        "policy_birth_crossover_mask_tape": np.zeros((1, 2), dtype=np.bool_),
        "policy_birth_delta_tape": np.zeros((1, 2), dtype=np.float32),
        "policy_birth_mutation_mask_tape": np.zeros((1, 2), dtype=np.bool_),
        "policy_birth_post_rng_state_tape": np.zeros((1, 4), dtype=np.uint8),
        "policy_birth_semantic_draws_tape": np.zeros((1,), dtype=np.int64),
        "policy_birth_tape_cursor": np.zeros((1,), dtype=np.int64),
    }


def _rng_state(seed: int) -> dict[str, Any]:
    return {
        "bit_generator": "PCG64",
        "state": {"state": seed, "inc": seed * 2 + 1},
        "has_uint32": 0,
        "uinteger": 0,
    }


def _projection(*, world: str, generation: int = 1) -> dict[str, Any]:
    tensors: dict[str, torch.Tensor] = {
        "alive_mask": torch.tensor([True, False], dtype=torch.bool),
        "slot_tokens": torch.tensor([1, 0], dtype=torch.int64),
        "slot_to_agent_id": torch.tensor([1, -1], dtype=torch.int64),
        "agent_ids_by_alive_order": torch.tensor([1], dtype=torch.int64),
        "action_cache_actions": torch.zeros((2, 1), dtype=torch.int32),
        "action_cache_token": torch.tensor([-1, -1], dtype=torch.int64),
        "action_cache_cursor": torch.tensor([-1, -1], dtype=torch.int32),
        "action_cache_next": torch.tensor([0, 0], dtype=torch.int32),
        "rng_manager_mutation_torch_cpu_state": torch.tensor([1, 2], dtype=torch.uint8),
    }
    for name, dtype in FIELD_DTYPES.items():
        tensors[f"typed_lifecycle_event_{name}"] = torch.empty((0,), dtype=dtype)
    metadata = {
        "world": world,
        "generation": generation,
        "tick": 0,
        "configured_steps_per_generation": 1,
        "policy_tick_stride": 1,
        "agent_grid_runner_phase": "generation_boundary",
        "agent_grid_authority_epoch": f"generation:{generation}:tick:0",
        "alive_count": 1,
        "population_next_id": 2,
        "typed_lifecycle_event_active_count": 0,
        "native_runtime_numpy_env_state": _rng_state(1),
        "native_runtime_numpy_runner_state": _rng_state(2),
        "native_runtime_numpy_policy_state": _rng_state(1),
        "native_runtime_env_policy_numpy_alias": True,
        "native_runtime_numpy_population_state": _rng_state(3),
        "native_runtime_python_random_state": [3, [1, 2, 3], None],
        "native_runtime_rng_manager_stream_indices": {
            "policy": {"py": 0, "np": 0, "torch_cpu": 0, "torch_cuda": {}},
            "mutation": {"py": 0, "np": 0, "torch_cpu": 0, "torch_cuda": {}},
            "metrics": {"py": 0, "np": 0, "torch_cpu": 0, "torch_cuda": {}},
            "stress": {"py": 0, "np": 0, "torch_cpu": 0, "torch_cuda": {}},
        },
    }
    return {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": metadata,
        "tensors": tensors,
        "tensor_manifest": {
            name: {
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "bytes": int(tensor.numel()) * int(tensor.element_size()),
            }
            for name, tensor in tensors.items()
        },
        "restore_count": 0,
        "generation_calls": generation,
        "event_cursor": 0,
        "runtime_event_cursor": 0,
        "last_applied_actions": 0,
        "last_moved": 0,
        "last_gathered": 0,
        "last_policy_refill_calls": 1,
        "last_policy_refill_rows": 1,
        "pending_generation_free_agent_ids": [],
        "pending_generation_free_slots": [],
        "last_world": world,
    }


def _raw(*, world: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "schema_version": "eiel_native_generation_fixture_v1",
        "executed": True,
        "hidden_fallback": False,
        "native_owner": True,
        "single_native_generation_call": True,
        "generation_index": 0,
        "steps_per_generation": 1,
        "native_tick_loop_count": 1,
        "committed_birth_rows": 0,
        "public_failed_birth_rows": 0,
        "failed_birth_rows": 0,
        "birth_attempt_rows": 0,
        "injury_blocked_births": 0,
        "death_rows": 0,
        "typed_event_rows": 0,
        "alive_count": 1,
        "capacity": 2,
        "free_slots": 1,
        "applied_actions": 0,
        "action_move_total": 0,
        "action_move_n": 0,
        "action_move_s": 0,
        "action_move_e": 0,
        "action_move_w": 0,
        "action_gather": 0,
        "action_rest": 0,
        "action_attack": 0,
        "moved": 0,
        "gathered": 0,
        "movement_attempts": 0,
        "movement_blocked_occupied": 0,
        "movement_blocked_bounds": 0,
        "peak_injury": 0.0,
        "policy_loop_tick_calls": 1,
        "policy_refill_calls": 1,
        "policy_refill_rows": 1,
        "policy_cache_hit_calls": 0,
        "policy_cache_rows_requested": 1,
        "policy_cache_rows_hit": 0,
        "policy_cache_rows_missed": 1,
        "policy_cache_rows_refilled": 1,
        "policy_cache_full_refill_ticks": 1,
        "policy_cache_partial_refill_ticks": 0,
        "behavior_rows": [_behavior_row()],
    }
    if world == "patch":
        for field in (
            "patch_movement_energy_event_count",
            "patch_spawn_sync_rows",
            "typed_death_event_rows",
            "env_rng_padding_draws",
        ):
            values[field] = 0
        for field in (
            "patch_food_harvest_total",
            "patch_scent_emit_total",
            "patch_waste_emit_total",
            "patch_movement_energy_total",
            "energy_collected_delta",
            "patch_scent_move_queued_total",
            "patch_waste_move_queued_total",
            "patch_scent_gather_queued_total",
            "patch_waste_gather_queued_total",
            "patch_movement_energy_extra_total",
            "patch_scent_emitted_total",
            "patch_waste_emitted_total",
            "patch_food_depleted_frac",
            "patch_food_full_frac",
            "patch_density_hotspot_frac",
        ):
            values[field] = 0.0
    return values


def _overlay(*, world: str) -> dict[str, Any]:
    lineage_input: dict[str, Any] = {
        "schema_version": "eiel_native_lineage_metrics_input_v1",
        "source": "native_public_outputs_and_single_checkpoint_projection",
        "world": world,
        "generation": 0,
        "projection_generation": 1,
        "record_count": 1,
        "agent_ids": [1],
        "records": [_lineage_record()],
    }
    if world == "patch":
        lineage_input.update(
            {
                "patch_profile": "P2",
                "patch_profile_scope": "full_profile",
                "patch_contract_schema_version": "eiel_patch_native_import_contract_v1",
                "promotion_accepted": False,
            }
        )
    return {
        "schema_version": "eiel_native_public_metrics_overlay_v1",
        "source": "native_owner_checkpoint_projection",
        "values": {
            "behavior_action_total": 0,
            "behavior_move_total": 0,
            "behavior_move_n_total": 0,
            "behavior_move_s_total": 0,
            "behavior_move_e_total": 0,
            "behavior_move_w_total": 0,
            "behavior_gather_total": 0,
            "behavior_rest_total": 0,
            "behavior_attack_total": 0,
            "native_lineage_metrics_input": lineage_input,
        },
    }


def _inputs(
    *,
    world: str = "flat",
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> CompactSummaryAuthorityInputs:
    raw = _raw(world=world)
    reporting = _projection(world=world)
    post = copy.deepcopy(reporting)
    if world == "flat":
        source = checkpoint_source_projection_digest_manifest(post)
        trajectory = checkpoint_digest_manifest(post, layer=DigestLayer.TRAJECTORY_STATE)
        storage = checkpoint_digest_manifest(post, layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE)
    else:
        assert monkeypatch is not None
        source = {
            "digest": "1" * 64,
            "rng": {
                "digest": "2" * 64,
                "complete": True,
                "missing_required_metadata_fields": [],
            },
        }
        trajectory = {"digest": "3" * 64}
        storage = {"digest": "4" * 64}

        def source_manifest(projection: Any) -> dict[str, Any]:
            assert projection is post
            return source

        def layer_manifest(projection: Any, *, layer: DigestLayer) -> dict[str, Any]:
            assert projection is post
            return trajectory if layer is DigestLayer.TRAJECTORY_STATE else storage

        monkeypatch.setattr(
            "engine.native_runtime.compact_summary_authority_adapter.checkpoint_source_projection_digest_manifest",
            source_manifest,
        )
        monkeypatch.setattr(
            "engine.native_runtime.compact_summary_authority_adapter.checkpoint_digest_manifest",
            layer_manifest,
        )
    boundary = {
        "schema_version": "eiel_native_architecture_b_boundary_v1",
        "completed_generation": 0,
        "next_generation": 1,
        "tick": 0,
        "phase": "generation_complete",
        "generation_result": copy.deepcopy(raw),
        "generation_end_reporting_frame": reporting,
        "post_turnover_checkpoint_projection": post,
        "turnover_result": {
            "executed": True,
            "hidden_fallback": False,
            "native_owner": True,
            "next_generation": 1,
            "next_tick": 0,
            "alive_before": 1,
            "alive_after": 1,
            "population_size": 1,
            "typed_event_rows": 0,
        },
        "checkpoint_projection_digest_manifests": {
            "source_projection": source,
            "trajectory_state": trajectory,
            "canonical_checkpoint_storage": storage,
        },
        "schedule_state": {
            "schema_version": "eiel_native_snapshot_schedule_v1",
            "schedule_origin_generation": 0,
            "explicit_next_generations": [],
        },
    }
    result = NativeGenerationResult(
        schema_version="eiel_native_generation_result_v1",
        decision="test",
        executed=True,
        generation_index=0,
        steps_per_generation=1,
        world=world,
        native_tick_loop_count=1,
        committed_birth_rows=0,
        counters=NativeEngineCounters(),
        capability=NativeRuntimeCapability(active_world=world),
        compact_result={},
        raw_result=raw,
    )
    return CompactSummaryAuthorityInputs(
        generation_result=result,
        run_result={
            "schema_version": "eiel_native_architecture_b_run_result_v1",
            "terminal_status": "run_complete",
            "boundaries": [boundary],
        },
        boundary_index=0,
        public_metrics_overlay=_overlay(world=world),
        lineage_state_before=LineageMetricsState(),
        sim_cfg=SimpleNamespace(lineage_observability_enable=True),
        policy_mutation_tape=_policy_tape(),
    )


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    elif isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for lhs, rhs in zip(left, right, strict=True):
            _assert_nested_equal(lhs, rhs)
    else:
        assert left == right


def test_flat_adapter_uses_one_projection_for_all_state_and_rng_digests() -> None:
    inputs = _inputs()
    post = inputs.run_result["boundaries"][0]["post_turnover_checkpoint_projection"]

    source = assemble_compact_summary_source(inputs)
    row = build_compact_summary_row(source)
    source_manifest = checkpoint_source_projection_digest_manifest(post)

    assert row["summary_status"] == "complete"
    assert row["terminal"]["status"] == "run_complete"
    assert row["state_digests"]["next_generation_start_state_digest_sha256"] == source_manifest["digest"]
    assert row["rng"]["authority_digest_sha256"] == source_manifest["rng"]["digest"]
    assert row["lineage_metrics"]["active_lineages"] == 1


def test_run_many_boundary_does_not_require_a_redundant_native_wrapper() -> None:
    inputs = _inputs()
    boundary_only = CompactSummaryAuthorityInputs(
        generation_result=None,
        run_result=inputs.run_result,
        boundary_index=inputs.boundary_index,
        public_metrics_overlay=inputs.public_metrics_overlay,
        lineage_state_before=inputs.lineage_state_before,
        sim_cfg=inputs.sim_cfg,
        policy_mutation_tape=inputs.policy_mutation_tape,
    )

    row = build_compact_summary_row(assemble_compact_summary_source(boundary_only))

    assert row["world"] == "flat"
    assert row["generation_end"] == {"generation": 0, "tick_exclusive": 1}


def test_patch_adapter_preserves_patch_optionality_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = assemble_compact_summary_source(_inputs(world="patch", monkeypatch=monkeypatch))
    row = build_compact_summary_row(source)

    assert row["world"] == "patch"
    assert row["patch_metrics"] is not None
    assert row["patch_metrics"]["patch_food_harvest_total"] == 0.0


def test_patch_adapter_derives_movement_energy_public_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(world="patch", monkeypatch=monkeypatch)
    raw = inputs.run_result["boundaries"][0]["generation_result"]
    raw["patch_movement_energy_total"] = -0.0
    del raw["patch_movement_energy_extra_total"]

    row = build_compact_summary_row(assemble_compact_summary_source(inputs))

    patch = row["patch_metrics"]
    assert patch is not None
    assert patch["patch_movement_energy_extra_total"].hex() == (-0.0).hex()


def test_patch_adapter_rejects_contradictory_movement_energy_public_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(world="patch", monkeypatch=monkeypatch)
    raw = inputs.run_result["boundaries"][0]["generation_result"]
    raw["patch_movement_energy_total"] = -0.0
    raw["patch_movement_energy_extra_total"] = 0.0

    with pytest.raises(
        CompactSummaryAuthorityAdapterError,
        match="must exactly alias patch_movement_energy_total",
    ):
        assemble_compact_summary_source(inputs)


def test_adapter_fails_closed_when_policy_tape_is_not_exported() -> None:
    inputs = _inputs()
    missing = CompactSummaryAuthorityInputs(
        generation_result=inputs.generation_result,
        run_result=inputs.run_result,
        boundary_index=inputs.boundary_index,
        public_metrics_overlay=inputs.public_metrics_overlay,
        lineage_state_before=inputs.lineage_state_before,
        sim_cfg=inputs.sim_cfg,
        policy_mutation_tape=None,
    )

    with pytest.raises(
        CompactSummaryAuthorityAdapterError,
        match="does not export the resident consumed-prefix tape",
    ):
        assemble_compact_summary_source(missing)


def test_adapter_fails_closed_when_rng_authority_is_incomplete() -> None:
    inputs = _inputs()
    boundary = inputs.run_result["boundaries"][0]
    del boundary["post_turnover_checkpoint_projection"]["metadata"][
        "native_runtime_numpy_runner_state"
    ]

    with pytest.raises(
        CompactSummaryAuthorityAdapterError,
        match="authority is incomplete",
    ):
        assemble_compact_summary_source(inputs)


def test_event_row_count_excludes_prior_active_history() -> None:
    inputs = _inputs()
    boundary = inputs.run_result["boundaries"][0]
    reporting = boundary["generation_end_reporting_frame"]
    post = boundary["post_turnover_checkpoint_projection"]
    values = {
        "event_type": 1,
        "generation": 0,
        "tick": 0,
        "sequence": 0,
        "actor_agent_id": 1,
        "target_agent_id": 2,
        "actor_slot": 0,
        "target_slot": 1,
        "parent_agent_id": 1,
        "child_agent_id": 2,
        "lineage_founder_id": 1,
        "lineage_depth": 0,
        "subtype_code": 1,
        "reason_code": -1,
        "success": 1,
        "value": 0.0,
    }
    for projection in (reporting, post):
        tensors = projection["tensors"]
        for name, dtype in FIELD_DTYPES.items():
            tensors[f"typed_lifecycle_event_{name}"] = torch.tensor(
                [values[name]], dtype=dtype
            )
        projection["metadata"]["typed_lifecycle_event_active_count"] = 1
        projection["event_cursor"] = 1
    boundary["checkpoint_projection_digest_manifests"] = {
        "source_projection": checkpoint_source_projection_digest_manifest(post),
        "trajectory_state": checkpoint_digest_manifest(
            post, layer=DigestLayer.TRAJECTORY_STATE
        ),
        "canonical_checkpoint_storage": checkpoint_digest_manifest(
            post, layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE
        ),
    }

    row = build_compact_summary_row(assemble_compact_summary_source(inputs))

    assert row["events"]["cursor_start"] == 1
    assert row["events"]["cursor_end"] == 1
    assert row["events"]["event_row_count"] == 0


def test_flat_source_projection_digest_rebuilds_from_device_tensor_hashes() -> None:
    projection = _projection(world="flat")
    expected = checkpoint_source_projection_digest_manifest(projection)["digest"]
    descriptors = {}
    for name, tensor in projection["tensors"].items():
        array = tensor.detach().to("cpu").contiguous().numpy()
        descriptors[name] = {
            "device": str(tensor.device),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }

    actual = checkpoint_source_projection_digest_from_tensor_manifest(
        projection_fields={
            name: value for name, value in projection.items() if name != "tensors"
        },
        tensor_manifest=descriptors,
    )

    assert actual == expected


def test_rng_digest_rebuilds_from_metadata_and_device_tensor_hashes() -> None:
    projection = _projection(world="flat")
    manifest = checkpoint_source_projection_digest_manifest(projection)
    descriptors = {}
    for name, tensor in projection["tensors"].items():
        array = tensor.detach().to("cpu").contiguous().numpy()
        descriptors[name] = {
            "device": str(tensor.device),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }

    actual = checkpoint_rng_digest_from_tensor_manifest(
        metadata=projection["metadata"],
        tensor_manifest=descriptors,
    )

    assert actual == manifest["rng"]["digest"]


def test_adapter_does_not_mutate_runtime_authorities_or_lineage_state() -> None:
    inputs = _inputs()
    before_result = copy.deepcopy(inputs.generation_result.raw_result)
    before_run = copy.deepcopy(inputs.run_result)
    before_overlay = copy.deepcopy(inputs.public_metrics_overlay)
    before_tape = copy.deepcopy(inputs.policy_mutation_tape)
    before_lineage = inputs.lineage_state_before.to_dict()

    assemble_compact_summary_source(inputs)

    _assert_nested_equal(inputs.generation_result.raw_result, before_result)
    _assert_nested_equal(inputs.run_result, before_run)
    _assert_nested_equal(inputs.public_metrics_overlay, before_overlay)
    _assert_nested_equal(inputs.policy_mutation_tape, before_tape)
    assert inputs.lineage_state_before.to_dict() == before_lineage


def test_adapter_accepts_exact_integral_public_behavior_count_floats() -> None:
    inputs = _inputs()
    values = inputs.public_metrics_overlay["values"]
    for field in (
        "behavior_action_total",
        "behavior_move_total",
        "behavior_move_n_total",
        "behavior_move_s_total",
        "behavior_move_e_total",
        "behavior_move_w_total",
        "behavior_gather_total",
        "behavior_rest_total",
        "behavior_attack_total",
    ):
        values[field] = float(values[field])

    source = assemble_compact_summary_source(inputs)

    assert source.behavior["action_total"] == 0
    assert type(source.behavior["action_total"]) is int


def test_adapter_rejects_fractional_public_behavior_count() -> None:
    inputs = _inputs()
    inputs.public_metrics_overlay["values"]["behavior_action_total"] = 1.5

    with pytest.raises(
        CompactSummaryAuthorityAdapterError,
        match="must be an int or an exactly integral finite float",
    ):
        assemble_compact_summary_source(inputs)


def test_failed_birth_summary_uses_typed_event_counter_not_public_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    raw = inputs.generation_result.raw_result
    raw["failed_birth_rows"] = 2
    raw["public_failed_birth_rows"] = 0
    boundary_raw = inputs.run_result["boundaries"][0]["generation_result"]
    boundary_raw["failed_birth_rows"] = 2
    boundary_raw["public_failed_birth_rows"] = 0
    zero_counts = {
        "birth_committed": 0,
        "mutation": 0,
        "birth_failed": 0,
        "death_committed": 0,
    }
    generation_counts = {**zero_counts, "birth_failed": 2}
    event_authority = {
        "event_cursor_start": 0,
        "event_cursor_pre_turnover": 2,
        "event_cursor_end": 2,
        "authoritative_event_cursor": 2,
        "event_row_count": 2,
        "generation_event_count": 2,
        "turnover_event_count": 0,
    }
    monkeypatch.setattr(
        "engine.native_runtime.compact_summary_authority_adapter._lifecycle_authority",
        lambda *_args: (
            event_authority,
            generation_counts,
            zero_counts,
            {"energy_depleted": 0, "health_depleted": 0},
        ),
    )

    row = build_compact_summary_row(assemble_compact_summary_source(inputs))

    assert row["failed_births"] == 2


def test_failed_terminal_requires_missing_compact_failure_stage_authority() -> None:
    inputs = _inputs()
    inputs.run_result["terminal_status"] = "failed"
    inputs.run_result["root_failure"] = {
        "error_code": "native_generation_transaction_failed",
        "active_generation": 1,
    }

    with pytest.raises(
        CompactSummaryAuthorityAdapterError,
        match="has no compact failure-stage authority",
    ):
        assemble_compact_summary_source(inputs)
