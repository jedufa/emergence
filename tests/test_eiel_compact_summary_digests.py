from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from engine.native_runtime import compact_summary_digests as digests
from engine.perf.cuda_architecture_b_torch_rng_state_pack_v1 import (
    pack_torch_cpu_rng_state_v1,
)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_entry(field: str, array: np.ndarray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "field": field,
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _policy_tape() -> dict[str, object]:
    return {
        "policy_birth_crossover_mask_tape": np.array(
            [[False, True], [True, False], [False, False]], dtype=np.bool_
        ),
        "policy_birth_delta_tape": np.array(
            [[0.1, -0.2], [0.3, 0.4], [9.0, 10.0]], dtype=np.float32
        ),
        "policy_birth_mutation_mask_tape": np.array(
            [[True, True], [False, True], [False, False]], dtype=np.bool_
        ),
        "policy_birth_post_rng_state_tape": np.array(
            [[1, 2, 3], [4, 5, 6], [90, 91, 92]], dtype=np.uint8
        ),
        "policy_birth_semantic_draws_tape": np.array([7, 11, 99], dtype=np.int64),
        "policy_birth_tape_cursor": np.array([2], dtype=np.int64),
    }


def _behavior_row(*, generation: int = 0, agent_id: int = 1) -> dict[str, object]:
    return {
        "generation": generation,
        "agent_id": agent_id,
        "cohort": "start",
        "move_n": 1,
        "move_s": 2,
        "move_e": 3,
        "move_w": 4,
        "gather": 5,
        "rest": 6,
        "attack": 7,
        "lifespan_steps": 28,
        "energy_collected": 3.5,
        "offspring": 2,
        "attacks_made": 1,
        "fitness": 8.25,
        "alive_end": 1,
        "eligible_for_selection": 1,
        "kept_next": 0,
        "injury_end": 0.5,
        "injury_delta": -0.25,
        "aggression_bias": 0.125,
    }


def _lineage_record(agent_id: int = 1) -> dict[str, object]:
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


def test_active_agent_ids_uses_exact_tensor_manifest_preimage() -> None:
    ids = np.array([2, 7, 11], dtype=np.int64)
    expected = {
        "schema_version": "eiel_active_agent_ids_digest_v1",
        "fields": [_tensor_entry("agent_ids_by_alive_order", ids)],
    }

    assert digests.active_agent_ids_digest_sha256(ids) == _canonical_digest(expected)


def test_active_agent_ids_digest_rebuilds_from_device_tensor_sha() -> None:
    ids = np.array([2, 7, 11], dtype=np.int64)

    rebuilt = digests.active_agent_ids_digest_from_tensor_sha256(
        int(ids.size),
        hashlib.sha256(ids.tobytes(order="C")).hexdigest(),
    )

    assert rebuilt == digests.active_agent_ids_digest_sha256(ids)


@pytest.mark.parametrize(
    "ids,match",
    [
        (np.array([1, 1], dtype=np.int64), "strictly ascending"),
        (np.array([2, 1], dtype=np.int64), "strictly ascending"),
        (np.array([-1, 2], dtype=np.int64), "nonnegative"),
        (np.array([1, 2], dtype=np.int32), "dtype"),
        (np.array([[1, 2]], dtype=np.int64), "rank"),
    ],
)
def test_active_agent_ids_rejects_invalid_authority(
    ids: np.ndarray, match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        digests.active_agent_ids_digest_sha256(ids)


def test_policy_tape_matches_exact_consumed_prefix_manifest() -> None:
    tape = _policy_tape()
    fields = []
    for field in sorted(name for name in tape if name != "policy_birth_tape_cursor"):
        fields.append(_tensor_entry(field, tape[field][:2]))  # type: ignore[index]
    expected = {
        "schema_version": "eiel_policy_mutation_tape_consumed_prefix_v1",
        "cursor": 2,
        "fields": fields,
    }

    assert digests.policy_mutation_tape_digest_sha256(tape) == _canonical_digest(expected)


def test_policy_tape_digest_rebuilds_from_consumed_device_hashes() -> None:
    tape = _policy_tape()
    cursor = int(tape["policy_birth_tape_cursor"][0])  # type: ignore[index]
    arrays = {
        field: np.ascontiguousarray(value[:cursor])  # type: ignore[index]
        for field, value in tape.items()
        if field != "policy_birth_tape_cursor"
    }

    rebuilt = digests.policy_mutation_tape_digest_from_tensor_sha256(
        cursor,
        {field: array.shape for field, array in arrays.items()},
        {
            field: hashlib.sha256(array.tobytes(order="C")).hexdigest()
            for field, array in arrays.items()
        },
    )

    assert rebuilt == digests.policy_mutation_tape_digest_sha256(tape)


def test_policy_tape_accepts_packed_post_rng_state_with_canonical_digest() -> None:
    import torch

    dense = _policy_tape()
    generator = torch.Generator(device="cpu").manual_seed(20260731)
    rows = []
    for _ in range(3):
        torch.rand((17,), generator=generator, dtype=torch.float64)
        rows.append(generator.get_state().clone())
    dense["policy_birth_post_rng_state_tape"] = torch.stack(rows).numpy()
    packed = copy.deepcopy(dense)
    packed["policy_birth_post_rng_state_packed_tape"] = (
        pack_torch_cpu_rng_state_v1(torch.stack(rows)).numpy()
    )
    packed.pop("policy_birth_post_rng_state_tape")

    assert digests.policy_mutation_tape_digest_sha256(packed) == (
        digests.policy_mutation_tape_digest_sha256(dense)
    )


@pytest.mark.parametrize("layer", ["trajectory_state", "canonical_checkpoint_storage"])
def test_checkpoint_layer_digest_rebuilds_from_device_field_hashes(
    layer: str,
) -> None:
    import torch

    from engine.native_runtime.checkpoint_digests import checkpoint_digest_manifest

    projection = {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": {"world": "flat", "generation": 4, "tick": 0},
        "tensors": {
            "alive_mask": torch.tensor([True, False], dtype=torch.bool),
            "energy": torch.tensor([2.5, 9.0], dtype=torch.float32),
        },
    }
    manifest = checkpoint_digest_manifest(projection, layer=layer)

    rebuilt = digests.checkpoint_layer_digest_from_tensor_manifest(
        layer=layer,
        projection_schema_version=projection["schema_version"],
        world="flat",
        generation=4,
        tick=0,
        fields=manifest["fields"],
    )

    assert rebuilt == manifest["digest"]


def _patch_p2_source_projection() -> dict[str, object]:
    import torch

    agent_grid = torch.tensor([[0, -1], [-1, -1]], dtype=torch.int32)
    patch_food = torch.tensor([[0.9, 0.1], [0.8, 0.2]], dtype=torch.float32)
    patch_cover = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    tensors = {
        "alive_mask": torch.tensor([True, False], dtype=torch.bool),
        "slot_tokens": torch.tensor([11, 22], dtype=torch.int64),
        "action_cache_actions": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        "action_cache_token": torch.tensor([11, -1], dtype=torch.int64),
        "action_cache_cursor": torch.tensor([0, -1], dtype=torch.int32),
        "action_cache_next": torch.tensor([1, -1], dtype=torch.int32),
        "agent_grid": agent_grid,
        "food_grid": patch_food.gt(np.float32(0.5)).to(dtype=torch.int32),
        "patch_habitat_lineage_id": torch.tensor([7, 0], dtype=torch.int32),
        "patch_food_amount": patch_food,
        "patch_fertility": torch.full((2, 2), 0.7, dtype=torch.float32),
        "patch_cover": patch_cover,
        "patch_shelter": torch.full((2, 2), 0.6, dtype=torch.float32),
        "patch_movement_cost": torch.full((2, 2), 1.2, dtype=torch.float32),
        "patch_scent": torch.full((2, 2), 0.2, dtype=torch.float32),
        "patch_scent_emit_pending": torch.full(
            (2, 2), 0.03, dtype=torch.float32
        ),
        "patch_waste": torch.full((2, 2), 0.1, dtype=torch.float32),
        "patch_waste_emit_pending": torch.full(
            (2, 2), 0.02, dtype=torch.float32
        ),
        "patch_recovery_activity": torch.full(
            (2, 2), 0.25, dtype=torch.float32
        ),
        # The source digest replaces these stored views with canonical rebuilds.
        "patch_visibility": torch.full((2, 2), -17.0, dtype=torch.float32),
        "patch_density": torch.full((2, 2), 23.0, dtype=torch.float32),
    }
    metadata = {
        "world": "patch",
        "generation": 2,
        "tick": 1,
        "configured_steps_per_generation": 4,
        "agent_grid_runner_phase": "partial_generation",
        "agent_grid_authority_epoch": "generation:2:tick:1",
        "native_runtime_patch_profile": "P2",
        "native_runtime_patch_profile_scope": "full_profile",
        "native_runtime_patch_contract_schema_version": (
            "eiel_patch_native_import_contract_v1"
        ),
        "patch_world_enable": True,
        "patch_world_schema_version": 1,
        "patch_world_observation_enable": True,
        "patch_world_observation_schema_version": 1,
        "patch_world_diagnostics_enable": True,
        "patch_world_lineage_descriptors_enable": True,
        "patch_world_descriptor_schema_version": 1,
        "patch_movement_cost_apply_enable": True,
        "patch_world_waste_consequence_enable": True,
        "patch_array_import_phase": "post_reference_setup_once",
        "patch_visibility_authority": "derived_from_patch_cover",
        "patch_visibility_from_cover_enable": True,
        "patch_visibility_default": 1.0,
        "patch_visibility_min": 0.0,
        "patch_visibility_max": 1.0,
        "patch_density_authority": "derived_pre_action_from_agent_grid",
        "patch_density_radius": 1,
        "patch_density_include_self": True,
        "patch_density_normalize": True,
        "patch_food_presence_threshold": 0.5,
        "patch_food_binary_projection": (
            "patch_food_amount_strictly_greater_than_threshold"
        ),
        "patch_pending_emission_phase": "queued_until_next_tick",
    }
    return {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": metadata,
        "tensors": tensors,
        "tensor_manifest": {
            name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            for name, tensor in tensors.items()
        },
        "restore_count": 0,
        "generation_calls": 2,
        "event_cursor": 17,
        "runtime_event_cursor": 21,
        "last_applied_actions": 3,
        "last_moved": 2,
        "last_gathered": 1,
        "last_policy_refill_calls": 1,
        "last_policy_refill_rows": 1,
        "pending_generation_free_agent_ids": [71],
        "pending_generation_free_slots": [1],
        "last_world": "patch",
    }


def _patch_p2_prehashed_manifest(
    projection: dict[str, object],
    source_manifest: dict[str, object],
) -> dict[str, dict[str, object]]:
    tensors = projection["tensors"]
    assert isinstance(tensors, dict)
    descriptors = {}
    for name, tensor in tensors.items():
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        descriptors[name] = {
            "device": str(tensor.device),
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    patch_authority = source_manifest["patch_authority"]
    assert isinstance(patch_authority, dict)
    derived_fields = patch_authority["derived_tensor_fields"]
    assert isinstance(derived_fields, list)
    for row in derived_fields:
        name = row["field"]
        descriptors[name] = {
            "device": "cpu",
            "dtype": row["dtype"],
            "shape": row["shape"],
            "sha256": row["sha256"],
        }
    return descriptors


def test_patch_p2_source_digest_rebuilds_exact_canonical_manifest() -> None:
    from engine.native_runtime.checkpoint_digests import (
        checkpoint_source_projection_digest_manifest,
    )

    projection = _patch_p2_source_projection()
    source_manifest = checkpoint_source_projection_digest_manifest(projection)
    tensor_manifest = _patch_p2_prehashed_manifest(projection, source_manifest)
    projection_fields = {
        name: value for name, value in projection.items() if name != "tensors"
    }

    actual = digests.checkpoint_source_projection_digest_from_tensor_manifest(
        projection_fields=projection_fields,
        tensor_manifest=tensor_manifest,
    )

    assert actual == source_manifest["digest"]
    assert "patch_visibility" in projection_fields["tensor_manifest"]
    assert "patch_density" in projection_fields["tensor_manifest"]


def test_patch_p2_source_digest_fails_closed_on_incomplete_authority() -> None:
    from engine.native_runtime.checkpoint_digests import (
        checkpoint_source_projection_digest_manifest,
    )

    projection = _patch_p2_source_projection()
    source_manifest = checkpoint_source_projection_digest_manifest(projection)
    tensor_manifest = _patch_p2_prehashed_manifest(projection, source_manifest)
    projection_fields = {
        name: value for name, value in projection.items() if name != "tensors"
    }

    unknown_world = copy.deepcopy(projection_fields)
    unknown_world["metadata"]["world"] = "unknown"
    with pytest.raises(ValueError, match="must be 'flat' or 'patch'"):
        digests.checkpoint_source_projection_digest_from_tensor_manifest(
            projection_fields=unknown_world,
            tensor_manifest=tensor_manifest,
        )

    missing_metadata = copy.deepcopy(projection_fields)
    del missing_metadata["metadata"]["patch_pending_emission_phase"]
    with pytest.raises(ValueError, match="missing required fields"):
        digests.checkpoint_source_projection_digest_from_tensor_manifest(
            projection_fields=missing_metadata,
            tensor_manifest=tensor_manifest,
        )

    missing_tensor = copy.deepcopy(tensor_manifest)
    del missing_tensor["patch_recovery_activity"]
    with pytest.raises(ValueError, match="missing required Patch P2 fields"):
        digests.checkpoint_source_projection_digest_from_tensor_manifest(
            projection_fields=projection_fields,
            tensor_manifest=missing_tensor,
        )

    malformed_tensor = copy.deepcopy(tensor_manifest)
    malformed_tensor["patch_cover"]["dtype"] = "float64"
    with pytest.raises(ValueError, match="patch_cover.dtype must be 'float32'"):
        digests.checkpoint_source_projection_digest_from_tensor_manifest(
            projection_fields=projection_fields,
            tensor_manifest=malformed_tensor,
        )


def test_policy_tape_ignores_unused_capacity_tail() -> None:
    first = _policy_tape()
    second = copy.deepcopy(first)
    for field in (
        "policy_birth_crossover_mask_tape",
        "policy_birth_delta_tape",
        "policy_birth_mutation_mask_tape",
        "policy_birth_post_rng_state_tape",
        "policy_birth_semantic_draws_tape",
    ):
        array = second[field]
        second[field] = np.concatenate((array, array[-1:]), axis=0)  # type: ignore[arg-type,index]
    second["policy_birth_crossover_mask_tape"][2:] = True  # type: ignore[index]
    second["policy_birth_delta_tape"][2:] = [-100.0, 100.0]  # type: ignore[index]
    second["policy_birth_mutation_mask_tape"][2:] = True  # type: ignore[index]
    second["policy_birth_post_rng_state_tape"][2:] = [8, 8, 8]  # type: ignore[index]
    second["policy_birth_semantic_draws_tape"][2:] = 1000  # type: ignore[index]

    assert digests.policy_mutation_tape_digest_sha256(first) == (
        digests.policy_mutation_tape_digest_sha256(second)
    )


def test_policy_tape_consumed_value_changes_digest() -> None:
    first = _policy_tape()
    second = copy.deepcopy(first)
    second["policy_birth_delta_tape"][1, 0] += np.float32(0.5)  # type: ignore[index]

    assert digests.policy_mutation_tape_digest_sha256(first) != (
        digests.policy_mutation_tape_digest_sha256(second)
    )


@pytest.mark.parametrize("cursor", [-1, 4])
def test_policy_tape_rejects_out_of_range_cursor(cursor: int) -> None:
    tape = _policy_tape()
    tape["policy_birth_tape_cursor"] = cursor
    with pytest.raises((TypeError, ValueError), match="cursor|capacity|must be"):
        digests.policy_mutation_tape_digest_sha256(tape)


def test_policy_tape_rejects_wrong_fields_dtype_shape_and_nonfinite() -> None:
    missing = _policy_tape()
    missing.pop("policy_birth_crossover_mask_tape")
    with pytest.raises(ValueError, match="missing"):
        digests.policy_mutation_tape_digest_sha256(missing)

    extra = _policy_tape()
    extra["unused"] = np.array([], dtype=np.uint8)
    with pytest.raises(ValueError, match="extra"):
        digests.policy_mutation_tape_digest_sha256(extra)

    wrong_dtype = _policy_tape()
    wrong_dtype["policy_birth_delta_tape"] = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="dtype"):
        digests.policy_mutation_tape_digest_sha256(wrong_dtype)

    wrong_shape = _policy_tape()
    wrong_shape["policy_birth_mutation_mask_tape"] = np.zeros((3, 3), dtype=np.bool_)
    with pytest.raises(ValueError, match="shape"):
        digests.policy_mutation_tape_digest_sha256(wrong_shape)

    nonfinite = _policy_tape()
    nonfinite["policy_birth_delta_tape"][0, 0] = np.nan  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        digests.policy_mutation_tape_digest_sha256(nonfinite)


def test_recoverable_error_digest_matches_exact_settled_envelope() -> None:
    details = {
        "available": 7,
        "required": 8,
        "context": {"generation": 2, "retryable": True},
    }
    expected = {
        "schema_version": "eiel_recoverable_error_digest_v1",
        "code": "event_capacity_exceeded",
        "details": details,
    }
    before = copy.deepcopy(details)

    actual = digests.recoverable_error_digest_sha256(
        "event_capacity_exceeded", details
    )

    assert actual == _canonical_digest(expected)
    assert details == before
    assert actual == digests.recoverable_error_digest_sha256(
        "event_capacity_exceeded", details
    )


def test_recoverable_error_digest_rejects_invalid_code_details_and_numbers() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        digests.recoverable_error_digest_sha256("", {})
    with pytest.raises(TypeError, match="mapping"):
        digests.recoverable_error_digest_sha256("code", [])
    with pytest.raises(TypeError, match="keys must be strings"):
        digests.recoverable_error_digest_sha256("code", {1: "bad"})
    with pytest.raises(ValueError, match="non-finite"):
        digests.recoverable_error_digest_sha256(
            "code", {"nested": [0.0, float("nan")]}
        )
    with pytest.raises(TypeError, match="non-JSON"):
        digests.recoverable_error_digest_sha256("code", {"bad": object()})


def test_behavior_rows_use_exact_field_order_and_change_on_one_value() -> None:
    rows = [_behavior_row(agent_id=1), _behavior_row(agent_id=2)]
    arrays = [[row[field] for field in digests.BEHAVIOR_ROW_FIELDS] for row in rows]
    expected = {
        "schema_version": "eiel_behavior_rows_digest_v1",
        "fields": list(digests.BEHAVIOR_ROW_FIELDS),
        "rows": arrays,
    }
    changed = copy.deepcopy(rows)
    changed[1]["attack"] = 8

    actual = digests.behavior_rows_digest_sha256(rows)
    assert actual == _canonical_digest(expected)
    assert actual != digests.behavior_rows_digest_sha256(changed)


def test_behavior_rows_reject_order_duplicates_fields_types_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="sorted"):
        digests.behavior_rows_digest_sha256(
            [_behavior_row(agent_id=2), _behavior_row(agent_id=1)]
        )
    with pytest.raises(ValueError, match="unique"):
        digests.behavior_rows_digest_sha256(
            [_behavior_row(agent_id=1), _behavior_row(agent_id=1)]
        )

    missing = _behavior_row()
    missing.pop("cohort")
    with pytest.raises(ValueError, match="missing"):
        digests.behavior_rows_digest_sha256([missing])

    extra = _behavior_row()
    extra["other"] = 1
    with pytest.raises(ValueError, match="extra"):
        digests.behavior_rows_digest_sha256([extra])

    wrong_type = _behavior_row()
    wrong_type["generation"] = 0.0
    with pytest.raises(TypeError, match="int"):
        digests.behavior_rows_digest_sha256([wrong_type])

    nonfinite = _behavior_row()
    nonfinite["fitness"] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        digests.behavior_rows_digest_sha256([nonfinite])


def test_lineage_records_match_exact_envelope_and_change_on_one_value() -> None:
    records = [_lineage_record(1), _lineage_record(2)]
    expected = {
        "schema_version": "eiel_lineage_records_digest_v1",
        "generation": 4,
        "records": records,
    }
    changed = copy.deepcopy(records)
    changed[1]["genome"]["exploration_bias"] = -0.5  # type: ignore[index]

    actual = digests.lineage_records_digest_sha256(4, records)
    assert actual == _canonical_digest(expected)
    assert actual != digests.lineage_records_digest_sha256(4, changed)


def test_lineage_records_reject_order_fields_types_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="ascending"):
        digests.lineage_records_digest_sha256(
            0, [_lineage_record(2), _lineage_record(1)]
        )
    with pytest.raises(ValueError, match="unique"):
        digests.lineage_records_digest_sha256(
            0, [_lineage_record(1), _lineage_record(1)]
        )

    missing = _lineage_record()
    missing["stats"].pop("role_source")  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="missing"):
        digests.lineage_records_digest_sha256(0, [missing])

    extra = _lineage_record()
    extra["genome"]["other"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="extra"):
        digests.lineage_records_digest_sha256(0, [extra])

    wrong_type = _lineage_record()
    wrong_type["genome"]["use_gru"] = 0  # type: ignore[index]
    with pytest.raises(TypeError, match="bool"):
        digests.lineage_records_digest_sha256(0, [wrong_type])

    nonfinite = _lineage_record()
    nonfinite["drives"][2] = float("nan")  # type: ignore[index]
    with pytest.raises(ValueError, match="finite"):
        digests.lineage_records_digest_sha256(0, [nonfinite])


def test_digest_producers_do_not_mutate_inputs() -> None:
    ids = np.array([1, 2], dtype=np.int64)
    tape = _policy_tape()
    behavior = [_behavior_row()]
    lineage = [_lineage_record()]
    ids_before = ids.copy()
    tape_before = {name: value.copy() for name, value in tape.items()}
    behavior_before = copy.deepcopy(behavior)
    lineage_before = copy.deepcopy(lineage)

    digests.active_agent_ids_digest_sha256(ids)
    digests.policy_mutation_tape_digest_sha256(tape)
    digests.behavior_rows_digest_sha256(behavior)
    digests.lineage_records_digest_sha256(0, lineage)

    assert np.array_equal(ids, ids_before)
    assert all(np.array_equal(tape[name], tape_before[name]) for name in tape)
    assert behavior == behavior_before
    assert lineage == lineage_before


def test_all_producers_fail_closed_on_non_little_endian_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(digests.sys, "byteorder", "big")

    producers = (
        lambda: digests.active_agent_ids_digest_sha256(
            np.array([1], dtype=np.int64)
        ),
        lambda: digests.policy_mutation_tape_digest_sha256(_policy_tape()),
        lambda: digests.behavior_rows_digest_sha256([_behavior_row()]),
        lambda: digests.lineage_records_digest_sha256(0, [_lineage_record()]),
    )
    for producer in producers:
        with pytest.raises(RuntimeError, match="little-endian"):
            producer()
