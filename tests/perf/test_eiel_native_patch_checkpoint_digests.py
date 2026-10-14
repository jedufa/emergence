from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch

from native_runtime.checkpoint_digests import (
    DEFAULT_FIELD_POLICIES,
    SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
    SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
    SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS,
    SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
    checkpoint_source_projection_authority_manifest,
    checkpoint_source_projection_digest_manifest,
    validate_checkpoint_source_projection,
)


_PROFILE_FEATURES = {
    "P0": (False, False, False),
    "P1": (True, False, False),
    "P2": (True, True, True),
}


def test_patch_derived_views_have_explicit_reconstructible_policies() -> None:
    for field in SOURCE_PATCH_DERIVED_TENSOR_FIELDS:
        policy = DEFAULT_FIELD_POLICIES[field]
        assert policy.reconstructible is True
        assert policy.trajectory_digest is False
        assert policy.canonical_storage_digest is False
        assert policy.diagnostic_projection_digest is False


def _clone_projection(projection: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(projection)
    cloned["tensors"] = {
        name: tensor.detach().clone()
        for name, tensor in projection["tensors"].items()
    }
    return cloned


def _patch_source_projection(profile: str) -> dict[str, Any]:
    observation, movement_cost, waste_consequence = _PROFILE_FEATURES[profile]
    env_rng_state = {"bit_generator": "PCG64", "state": {"state": 11, "inc": 3}}
    agent_grid = torch.tensor(
        [[0, -1, -1], [-1, -1, -1], [-1, -1, -1]],
        dtype=torch.int32,
    )
    patch_food = torch.tensor(
        [[0.9, 0.1, 0.8], [0.0, 0.7, 0.2], [0.6, 0.4, 0.0]],
        dtype=torch.float32,
    )
    patch_cover = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        dtype=torch.float32,
    )
    tensors: dict[str, torch.Tensor] = {
        "alive_mask": torch.tensor([True, False]),
        "slot_tokens": torch.tensor([11, 22], dtype=torch.int64),
        "action_cache_actions": torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        "action_cache_token": torch.tensor([11, -1], dtype=torch.int64),
        "action_cache_cursor": torch.tensor([0, -1], dtype=torch.int32),
        "action_cache_next": torch.tensor([1, -1], dtype=torch.int32),
        "agent_grid": agent_grid,
        "food_grid": patch_food.gt(np.float32(0.5)).to(dtype=torch.int32),
        "patch_habitat_lineage_id": torch.tensor([7, 0], dtype=torch.int32),
        "patch_food_amount": patch_food,
        "patch_fertility": torch.tensor(
            [[0.9, 0.8, 0.7], [0.6, 0.5, 0.4], [0.3, 0.2, 0.1]],
            dtype=torch.float32,
        ),
        "patch_cover": patch_cover,
        "patch_shelter": torch.flip(patch_cover, dims=(0,)).contiguous(),
        "patch_movement_cost": torch.tensor(
            [[1.0, 1.1, 1.2], [1.3, 1.4, 1.5], [1.6, 1.7, 1.8]],
            dtype=torch.float32,
        ),
        "patch_scent": torch.full((3, 3), 0.2, dtype=torch.float32),
        "patch_scent_emit_pending": torch.full((3, 3), 0.03, dtype=torch.float32),
        "patch_waste": torch.full((3, 3), 0.1, dtype=torch.float32),
        "patch_waste_emit_pending": torch.full((3, 3), 0.02, dtype=torch.float32),
        # Deliberately stale values prove that digest authority rebuilds these views.
        "patch_visibility": torch.full((3, 3), -17.0, dtype=torch.float32),
        "patch_density": torch.full((3, 3), 23.0, dtype=torch.float32),
    }
    if profile == "P2":
        tensors["patch_recovery_activity"] = torch.full(
            (3, 3),
            0.25,
            dtype=torch.float32,
        )

    metadata = {
        "world": "patch",
        "generation": 2,
        "tick": 1,
        "configured_steps_per_generation": 4,
        "agent_grid_runner_phase": "partial_generation",
        "agent_grid_authority_epoch": "generation:2:tick:1",
        "alive_count": 1,
        "native_runtime_patch_profile": profile,
        "native_runtime_patch_profile_scope": "full_profile",
        "native_runtime_patch_contract_schema_version": (
            "eiel_patch_native_import_contract_v1"
        ),
        "patch_world_enable": True,
        "patch_world_schema_version": 1,
        "patch_world_observation_enable": observation,
        "patch_world_observation_schema_version": 1,
        "patch_world_diagnostics_enable": True,
        "patch_world_lineage_descriptors_enable": True,
        "patch_world_descriptor_schema_version": 1,
        "patch_movement_cost_apply_enable": movement_cost,
        "patch_world_waste_consequence_enable": waste_consequence,
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
        "native_runtime_numpy_env_state": copy.deepcopy(env_rng_state),
        "native_runtime_numpy_runner_state": {"state": 19},
        "native_runtime_numpy_policy_state": copy.deepcopy(env_rng_state),
        "native_runtime_env_policy_numpy_alias": True,
        "native_runtime_numpy_population_state": {"state": 29},
        "native_runtime_python_random_state": [3, [1, 2, 3], None],
        "native_runtime_rng_manager_stream_indices": {"policy": {"np": 0}},
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
        "runtime_event_cursor": 17,
        "last_applied_actions": 3,
        "last_moved": 2,
        "last_gathered": 1,
        "last_policy_refill_calls": 1,
        "last_policy_refill_rows": 1,
        "pending_generation_free_agent_ids": [71],
        "pending_generation_free_slots": [1],
        "last_world": "patch",
    }


@pytest.mark.parametrize("profile", ["P0", "P1", "P2"])
def test_patch_source_authority_digest_rebuilds_derived_views(
    profile: str,
) -> None:
    baseline = _patch_source_projection(profile)
    changed_derived = _clone_projection(baseline)
    changed_derived["tensors"]["patch_visibility"].fill_(91.0)
    changed_derived["tensors"]["patch_density"].fill_(-43.0)

    manifest = checkpoint_source_projection_digest_manifest(baseline)
    changed_manifest = checkpoint_source_projection_digest_manifest(changed_derived)
    authority = checkpoint_source_projection_authority_manifest(baseline)

    assert manifest["digest"] == changed_manifest["digest"]
    assert manifest["top_level_field_digests"] == changed_manifest[
        "top_level_field_digests"
    ]
    assert authority["source_projection_sha256"] == manifest["digest"]
    assert manifest["patch_authority"]["profile"] == profile
    derived = {
        row["field"]: row
        for row in manifest["patch_authority"]["derived_tensor_fields"]
    }
    assert set(derived) == set(SOURCE_PATCH_DERIVED_TENSOR_FIELDS)
    assert all(
        row["classification"] == "reconstructible_derived"
        for row in derived.values()
    )
    assert derived["patch_visibility"]["rebuild_source_fields"] == ["patch_cover"]
    assert derived["patch_density"]["rebuild_source_fields"] == ["agent_grid"]


@pytest.mark.parametrize("profile", ["P0", "P1", "P2"])
def test_patch_source_authority_digest_covers_every_profile_tensor(
    profile: str,
) -> None:
    baseline = _patch_source_projection(profile)
    manifest = checkpoint_source_projection_digest_manifest(baseline)
    expected_fields = [
        "agent_grid",
        "food_grid",
        *SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS,
        *SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
        *(
            SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS
            if profile == "P2"
            else ()
        ),
    ]
    assert manifest["patch_authority"]["authoritative_tensor_fields"] == expected_fields

    mutation_fields = [
        "agent_grid",
        *SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS,
        *SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
        *(
            SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS
            if profile == "P2"
            else ()
        ),
    ]
    for field in mutation_fields:
        changed = _clone_projection(baseline)
        tensor = changed["tensors"][field]
        tensor.reshape(-1)[0] += 0.05 if tensor.is_floating_point() else 1
        assert checkpoint_source_projection_digest_manifest(changed)["digest"] != (
            manifest["digest"]
        ), field


@pytest.mark.parametrize(
    "field",
    [
        *SOURCE_PATCH_REQUIRED_ROW_TENSOR_FIELDS,
        *SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
        *SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
        *SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
    ],
)
def test_patch_source_authority_rejects_missing_profile_tensor(field: str) -> None:
    projection = _patch_source_projection("P2")
    del projection["tensors"][field]

    with pytest.raises(ValueError, match="missing required fields"):
        validate_checkpoint_source_projection(projection)


def _drop_profile(projection: dict[str, Any]) -> None:
    del projection["metadata"]["native_runtime_patch_profile"]


def _malform_profile(projection: dict[str, Any]) -> None:
    projection["metadata"]["native_runtime_patch_profile"] = ["P0"]


def _unknown_profile(projection: dict[str, Any]) -> None:
    projection["metadata"]["native_runtime_patch_profile"] = "P9"


def _diagnostic_scope(projection: dict[str, Any]) -> None:
    projection["metadata"]["native_runtime_patch_profile_scope"] = (
        "focused_p2_subcell"
    )


def _contradict_profile_flags(projection: dict[str, Any]) -> None:
    projection["metadata"]["patch_world_observation_enable"] = False


def _malform_profile_flag(projection: dict[str, Any]) -> None:
    projection["metadata"]["patch_world_waste_consequence_enable"] = 1


def _contradict_derived_marker(projection: dict[str, Any]) -> None:
    projection["metadata"]["patch_density_authority"] = "stored_authoritative"


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (_drop_profile, "missing required fields"),
        (_malform_profile, "exact P0/P1/P2 string"),
        (_unknown_profile, "profile is unknown"),
        (_diagnostic_scope, "scope must be 'full_profile'"),
        (_contradict_profile_flags, "contradicts feature flags"),
        (_malform_profile_flag, "must be a bool"),
        (_contradict_derived_marker, "contract marker mismatch"),
    ],
)
def test_patch_source_authority_rejects_malformed_or_unknown_profile_state(
    mutate: Callable[[dict[str, Any]], None],
    error: str,
) -> None:
    projection = _patch_source_projection("P2")
    mutate(projection)

    with pytest.raises(ValueError, match=error):
        validate_checkpoint_source_projection(projection)


def test_patch_source_authority_rejects_profile_inactive_recovery_state() -> None:
    projection = _patch_source_projection("P0")
    projection["tensors"]["patch_recovery_activity"] = torch.zeros(
        (3, 3),
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="inactive patch_recovery_activity"):
        validate_checkpoint_source_projection(projection)


def test_patch_source_authority_rejects_unknown_patch_tensor_state() -> None:
    projection = _patch_source_projection("P2")
    projection["tensors"]["patch_delayed_hazard_active"] = torch.zeros(
        (3, 3),
        dtype=torch.bool,
    )

    with pytest.raises(ValueError, match="unsupported state"):
        validate_checkpoint_source_projection(projection)


def test_patch_source_authority_rejects_malformed_authoritative_tensor() -> None:
    projection = _patch_source_projection("P2")
    projection["tensors"]["patch_cover"] = projection["tensors"][
        "patch_cover"
    ].to(dtype=torch.float64)

    with pytest.raises(ValueError, match="patch_cover.*torch.float32 rank 2"):
        validate_checkpoint_source_projection(projection)


def test_patch_source_authority_rejects_binary_food_contradiction() -> None:
    projection = _patch_source_projection("P2")
    projection["tensors"]["food_grid"][0, 0] = 0

    with pytest.raises(ValueError, match="binary food projection is inconsistent"):
        validate_checkpoint_source_projection(projection)


def test_source_authority_still_rejects_unknown_world() -> None:
    projection = _patch_source_projection("P2")
    projection["metadata"]["world"] = "jungle"

    with pytest.raises(ValueError, match="world in"):
        validate_checkpoint_source_projection(projection)
