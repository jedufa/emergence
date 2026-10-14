from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from perf.cuda_architecture_b_modular_state_abi import (
    MODULAR_STORAGE_METADATA_KEY,
    MODULAR_STORAGE_PROJECTION_PREFIX,
    build_modular_storage_layout,
)
from native_runtime.checkpoint_digests import (
    DEFAULT_FIELD_POLICIES,
    EED_CHECKPOINT_TENSOR_NAMES,
    GENOME_V2_CHECKPOINT_TENSOR_NAMES,
    DigestLayer,
    canonicalize_checkpoint_storage_field,
    checkpoint_digest_manifest,
    checkpoint_source_projection_digest_manifest,
    field_policy_manifest,
    register_field_digest_policy,
    validate_checkpoint_source_projection,
)
from perf.cuda_diploid_state_bank_v1 import (
    CHECKPOINT_TENSOR_NAMES as DIPLOID_CHECKPOINT_TENSOR_NAMES,
)


def _flat_source_projection(
    *,
    generation: int = 2,
    tick: int = 1,
    configured_steps: int = 4,
) -> dict[str, object]:
    tensors = {
        "alive_mask": torch.tensor([True, True, False]),
        "slot_tokens": torch.tensor([11, 22, 33], dtype=torch.int64),
        "action_cache_actions": torch.tensor(
            [[1, 2, 3], [4, 5, 6], [0, 1, 2]],
            dtype=torch.int32,
        ),
        "action_cache_token": torch.tensor([11, -1, 33], dtype=torch.int64),
        "action_cache_cursor": torch.tensor([1, -1, 0], dtype=torch.int32),
        "action_cache_next": torch.tensor([2, -1, 0], dtype=torch.int32),
    }
    return {
        "schema_version": "cxxbackbone10_checkpoint_projection_v1",
        "metadata": {
            "world": "flat",
            "generation": generation,
            "tick": tick,
            "configured_steps_per_generation": configured_steps,
            "policy_tick_stride": 3,
            "agent_grid_phase_contract_schema_version": "eiel_agent_grid_phase_contract_v1",
            "agent_grid_runner_phase": "generation_boundary" if tick == 0 else "partial_generation",
            "agent_grid_authority_epoch": f"generation:{generation}:tick:{tick}",
            "agent_grid_preserve_exact_stored": True,
        },
        "tensors": tensors,
        "tensor_manifest": {
            name: {"dtype": str(tensor.dtype), "shape": list(tensor.shape)}
            for name, tensor in tensors.items()
        },
        "restore_count": 0,
        "generation_calls": 2,
        "event_cursor": 17,
        "runtime_event_cursor": 23,
        "last_applied_actions": 3,
        "last_moved": 2,
        "last_gathered": 1,
        "last_policy_refill_calls": 1,
        "last_policy_refill_rows": 2,
        "pending_generation_free_agent_ids": [71],
        "pending_generation_free_slots": [2],
        "last_world": "flat",
    }


def test_default_field_policy_manifest_classifies_required_residual_fields() -> None:
    fields = {
        "agent_grid",
        "drive_inertia",
        "drive_jitter",
        "drive_noise",
        "drive_noise_samples_by_tick_slot",
        "drive_volatility",
        "drives",
        "early_life_weight",
        "episode0_drive_jitter_sum",
        "event_tape_actor_id",
        "event_tape_target_id",
        "event_tape_type",
        "event_tape_value",
        "genome_action_temperature",
        "genome_aggression_bias",
        "genome_drive_birth_noise",
        "genome_drive_inertia",
        "genome_drive_logit_gain",
        "genome_drive_noise",
        "genome_drive_sp_fatigue",
        "genome_drive_sp_fear",
        "genome_drive_sp_hunger",
        "genome_drive_sp_pain",
        "genome_exp_prior_noise",
        "genome_exp_prior_strength",
        "genome_exploration_bias",
        "genome_mutation_rate",
        "genome_pred_lr",
        "genome_reproduction_threshold",
        "last_energy_delta",
        "mlp_pack",
        "policy_action_temperature",
        "policy_attack_bias",
        "policy_drive_logit_gain",
        "prev_drive_delta",
        "repro_threshold",
        "survival_steps",
    }

    manifest = field_policy_manifest(fields)

    assert {row["field"] for row in manifest["fields"]} == fields
    assert all(row["primary_layer"] for row in manifest["fields"])


def test_complete_field_policy_manifest_is_internally_coherent() -> None:
    manifest = field_policy_manifest()

    assert {row["field"] for row in manifest["fields"]} == set(DEFAULT_FIELD_POLICIES)
    for row in manifest["fields"]:
        assert row["primary_source_layer"] == row["primary_layer"]
        assert row["canonicalizer"] == row["canonicalization"]
        assert row["authority_classification"] in {
            "authoritative",
            "reconstructible_derived",
            "diagnostic_only",
        }
        if row["diagnostic_only"]:
            assert row["authority_classification"] == "diagnostic_only"
            assert row["included_digest_layers"] == []
        elif row["included_digest_layers"]:
            assert row["primary_source_layer"] in row["included_digest_layers"]


def test_all_phase_e_modular_planes_are_exact_authoritative_digest_fields() -> None:
    names = {
        f"{MODULAR_STORAGE_PROJECTION_PREFIX}{field.name}"
        for field in build_modular_storage_layout(agent_capacity=1).fields
    }

    assert len(names) == 57
    assert names <= set(DEFAULT_FIELD_POLICIES)
    for name in names:
        policy = DEFAULT_FIELD_POLICIES[name]
        assert policy.reconstructible is False
        assert policy.diagnostic_only is False
        assert all(policy.includes(layer) for layer in DigestLayer)
        assert policy.canonicalization == "preserve_exact_stored"


def test_diploid_and_genome_v2_checkpoint_fields_are_exact_authority() -> None:
    names = {
        "diploid_allele_storage",
        *DIPLOID_CHECKPOINT_TENSOR_NAMES,
        *GENOME_V2_CHECKPOINT_TENSOR_NAMES,
    }

    assert names <= set(DEFAULT_FIELD_POLICIES)
    for name in names:
        policy = DEFAULT_FIELD_POLICIES[name]
        assert policy.primary_layer is DigestLayer.TRAJECTORY_STATE
        assert policy.reconstructible is False
        assert policy.diagnostic_only is False
        assert all(policy.includes(layer) for layer in DigestLayer)
        assert policy.canonicalization == "preserve_exact_stored"


def test_eed_checkpoint_fields_are_exact_authority() -> None:
    names = set(EED_CHECKPOINT_TENSOR_NAMES)

    assert len(names) == 15
    assert names <= set(DEFAULT_FIELD_POLICIES)
    for name in names:
        policy = DEFAULT_FIELD_POLICIES[name]
        assert policy.primary_layer is DigestLayer.TRAJECTORY_STATE
        assert policy.reconstructible is False
        assert policy.diagnostic_only is False
        assert all(policy.includes(layer) for layer in DigestLayer)
        assert policy.canonicalization == "preserve_exact_stored"


def test_duplicate_field_policy_registration_fails_closed() -> None:
    registry = {}
    policy = DEFAULT_FIELD_POLICIES["agent_grid"]
    register_field_digest_policy(registry, policy)

    with pytest.raises(KeyError, match="duplicate checkpoint digest policy registration"):
        register_field_digest_policy(registry, policy)


def test_contradictory_primary_layer_exclusion_fails_closed() -> None:
    policy = replace(
        DEFAULT_FIELD_POLICIES["agent_grid"],
        trajectory_digest=False,
    )

    with pytest.raises(ValueError, match="excludes its primary digest layer"):
        register_field_digest_policy({}, policy)


def test_field_policy_manifest_rejects_registry_key_field_mismatch() -> None:
    policy = DEFAULT_FIELD_POLICIES["agent_grid"]

    with pytest.raises(ValueError, match="does not match field"):
        field_policy_manifest(policies={"wrong_key": policy})


def test_agent_grid_is_authoritative_and_not_reconstructible() -> None:
    policy = DEFAULT_FIELD_POLICIES["agent_grid"]

    assert policy.primary_layer is DigestLayer.TRAJECTORY_STATE
    assert policy.reconstructible is False
    assert policy.canonicalization == "preserve_exact_stored_grid"
    assert policy.trajectory_digest is True
    assert policy.canonical_storage_digest is True


def test_canonical_storage_digest_normalizes_projection_wrapper_schema() -> None:
    reference = _flat_source_projection()
    native = _flat_source_projection()
    reference["schema_version"] = "cxxbackbone10_resident_state_owner_v1"
    native["schema_version"] = "cxxbackbone10_checkpoint_projection_v1"

    reference_manifest = checkpoint_digest_manifest(
        reference,
        layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
    )
    native_manifest = checkpoint_digest_manifest(
        native,
        layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
    )

    assert reference_manifest["digest"] == native_manifest["digest"]
    assert reference_manifest["metadata"] == native_manifest["metadata"]


@pytest.mark.parametrize(
    ("layer", "expected_digest"),
    [
        (
            DigestLayer.TRAJECTORY_STATE,
            "5f6d086d07974be734a3e5ba3933813de892e0e2ce949eb76dc8ba2a1227ecb6",
        ),
        (
            DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
            "1da8290ce1cb9bbd23cf8b5dcf1a117219387c4b0102d399e44de46a87558c28",
        ),
        (
            DigestLayer.DIAGNOSTIC_PROJECTION,
            "e8f707dc900094c34fa42b5a58d1506a99356bdf53198653fc9424b3a5efea50",
        ),
    ],
)
def test_legacy_layer_digest_is_frozen_when_modular_envelope_is_absent(
    layer: DigestLayer,
    expected_digest: str,
) -> None:
    manifest = checkpoint_digest_manifest(_flat_source_projection(), layer=layer)

    assert manifest["digest"] == expected_digest
    assert MODULAR_STORAGE_METADATA_KEY not in manifest["metadata"]


def test_event_tape_digest_uses_active_prefix_only_for_trajectory_layer() -> None:
    projection = {
        "schema_version": "native_checkpoint_projection",
        "metadata": {"world": "flat", "generation": 2, "tick": 3, "active_event_count": 2},
        "tensors": {
            "event_tape_type": torch.tensor([1, 3, 99, 99], dtype=torch.int32),
            "event_tape_actor_id": torch.tensor([10, 11, -1, -1], dtype=torch.int64),
            "event_tape_target_id": torch.tensor([20, -1, -1, -1], dtype=torch.int64),
            "event_tape_value": torch.tensor([1.0, 4.0, 0.0, 0.0], dtype=torch.float32),
        },
    }

    digest = checkpoint_digest_manifest(projection, layer=DigestLayer.TRAJECTORY_STATE)

    rows = {row["field"]: row for row in digest["fields"]}
    assert rows["event_tape_type"]["shape"] == [2]
    assert rows["event_tape_actor_id"]["shape"] == [2]
    assert rows["event_tape_value"]["shape"] == [2]


def test_event_tape_digest_rejects_count_cursor_contradiction_before_hashing() -> None:
    projection = {
        "metadata": {
            "active_event_count": 2,
            "event_tape_active_count": 2,
            "event_tape_cursor": 1,
        },
        "tensors": {
            "event_tape_type": torch.tensor([1, 3], dtype=torch.int32),
            "event_tape_actor_id": torch.tensor([10, 11], dtype=torch.int64),
            "event_tape_target_id": torch.tensor([20, -1], dtype=torch.int64),
            "event_tape_value": torch.tensor([1.0, 4.0], dtype=torch.float32),
        },
    }

    with pytest.raises(RuntimeError, match="typed lifecycle projection cursor contradiction"):
        checkpoint_digest_manifest(projection, layer=DigestLayer.TRAJECTORY_STATE)


def test_event_tape_digest_rejects_column_length_mismatch_before_hashing() -> None:
    projection = {
        "metadata": {"active_event_count": 1},
        "tensors": {
            "event_tape_type": torch.tensor([1, 3], dtype=torch.int32),
            "event_tape_actor_id": torch.tensor([10], dtype=torch.int64),
            "event_tape_target_id": torch.tensor([20, -1], dtype=torch.int64),
            "event_tape_value": torch.tensor([1.0, 4.0], dtype=torch.float32),
        },
    }

    with pytest.raises(RuntimeError, match="columns have inconsistent lengths"):
        checkpoint_digest_manifest(projection, layer=DigestLayer.TRAJECTORY_STATE)


def test_unclassified_field_fails_closed() -> None:
    projection = {
        "schema_version": "native_checkpoint_projection",
        "metadata": {"world": "flat"},
        "tensors": {"new_authoritative_field": torch.tensor([1], dtype=torch.int64)},
    }

    with pytest.raises(KeyError, match="unclassified checkpoint digest field"):
        checkpoint_digest_manifest(projection, layer=DigestLayer.TRAJECTORY_STATE)


def test_dead_free_row_normalization_is_field_policy_explicit() -> None:
    values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    alive = torch.tensor([True, False, True])

    unchanged = canonicalize_checkpoint_storage_field(
        "drive_noise",
        values,
        alive_mask=alive,
        allow_dead_free_row_normalization=False,
    )
    normalized = canonicalize_checkpoint_storage_field(
        "drive_noise",
        values,
        alive_mask=alive,
        allow_dead_free_row_normalization=True,
    )

    assert unchanged.tolist() == [1.0, 2.0, 3.0]
    assert normalized.tolist() == [1.0, 0.0, 3.0]


def test_live_row_manifest_digest_ignores_dead_free_row_payload() -> None:
    base_projection = {
        "schema_version": "native_checkpoint_projection",
        "metadata": {"world": "flat"},
        "tensors": {
            "alive_mask": torch.tensor([True, False, True]),
            "drives": torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [9.0, 8.0, 7.0, 6.0],
                    [0.5, 0.6, 0.7, 0.8],
                ],
                dtype=torch.float32,
            ),
        },
    }
    dead_row_changed = {
        **base_projection,
        "tensors": {
            **base_projection["tensors"],
            "drives": torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [1.0, 2.0, 3.0, 4.0],
                    [0.5, 0.6, 0.7, 0.8],
                ],
                dtype=torch.float32,
            ),
        },
    }
    live_row_changed = {
        **base_projection,
        "tensors": {
            **base_projection["tensors"],
            "drives": torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [9.0, 8.0, 7.0, 6.0],
                    [0.5, 0.6, 0.7, 0.9],
                ],
                dtype=torch.float32,
            ),
        },
    }

    base_digest = checkpoint_digest_manifest(
        base_projection,
        fields={"alive_mask", "drives"},
        layer=DigestLayer.TRAJECTORY_STATE,
    )["digest"]

    assert (
        checkpoint_digest_manifest(
            dead_row_changed,
            fields={"alive_mask", "drives"},
            layer=DigestLayer.TRAJECTORY_STATE,
        )["digest"]
        == base_digest
    )
    assert (
        checkpoint_digest_manifest(
            live_row_changed,
            fields={"alive_mask", "drives"},
            layer=DigestLayer.TRAJECTORY_STATE,
        )["digest"]
        != base_digest
    )


def test_agent_grid_is_never_dead_free_row_normalized() -> None:
    grid = torch.tensor([[7, -1], [-1, 9]], dtype=torch.int32)
    alive = torch.tensor([True, False])

    normalized = canonicalize_checkpoint_storage_field(
        "agent_grid",
        grid,
        alive_mask=alive,
        allow_dead_free_row_normalization=True,
    )

    assert torch.equal(normalized, grid)


def test_source_projection_digest_covers_cache_cursor_and_all_top_level_fields() -> None:
    projection = _flat_source_projection()

    manifest = checkpoint_source_projection_digest_manifest(projection)

    assert manifest["authority_cursor"] == {
        "generation": 2,
        "tick": 1,
        "configured_steps_per_generation": 4,
        "agent_grid_runner_phase": "partial_generation",
        "agent_grid_authority_epoch": "generation:2:tick:1",
    }
    assert [row["field"] for row in manifest["action_cache"]["fields"]] == [
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
    ]
    assert manifest["action_cache"]["valid_slots"] == [0]
    assert manifest["action_cache"]["boundary_status"] == "captured"
    assert manifest["top_level_fields"] == sorted(projection)
    assert set(manifest["top_level_field_digests"]) == set(projection)


@pytest.mark.parametrize(
    ("field", "index", "replacement"),
    [
        ("action_cache_actions", (0, 2), 6),
        ("action_cache_token", (1,), -2),
        ("action_cache_cursor", (1,), -2),
        ("action_cache_next", (1,), -2),
    ],
)
def test_source_projection_digest_changes_for_each_action_cache_field(
    field: str,
    index: tuple[int, ...],
    replacement: int,
) -> None:
    baseline = _flat_source_projection()
    changed = _flat_source_projection()
    changed["tensors"][field][index] = replacement

    baseline_digest = checkpoint_source_projection_digest_manifest(baseline)["digest"]
    changed_digest = checkpoint_source_projection_digest_manifest(changed)["digest"]

    assert changed_digest != baseline_digest


def test_source_projection_digest_records_action_cache_validity() -> None:
    projection = _flat_source_projection()
    projection["tensors"]["action_cache_token"][1] = 22
    projection["tensors"]["action_cache_cursor"][1] = 0
    projection["tensors"]["action_cache_next"][1] = 4

    cache = checkpoint_source_projection_digest_manifest(projection)["action_cache"]

    assert cache["valid_slots"] == [0, 1]
    assert cache["valid_slot_count"] == 2


def test_source_projection_digest_requires_and_hashes_runtime_event_cursor() -> None:
    baseline = _flat_source_projection()
    changed = _flat_source_projection()
    changed["runtime_event_cursor"] += 1

    baseline_manifest = checkpoint_source_projection_digest_manifest(baseline)
    changed_manifest = checkpoint_source_projection_digest_manifest(changed)

    assert "runtime_event_cursor" in baseline_manifest["top_level_fields"]
    assert (
        baseline_manifest["top_level_field_digests"]["runtime_event_cursor"]
        != changed_manifest["top_level_field_digests"]["runtime_event_cursor"]
    )
    assert baseline_manifest["digest"] != changed_manifest["digest"]

    del changed["runtime_event_cursor"]
    with pytest.raises(ValueError, match="runtime_event_cursor"):
        validate_checkpoint_source_projection(changed)


@pytest.mark.parametrize(
    "field",
    [
        "restore_count",
        "generation_calls",
        "event_cursor",
        "runtime_event_cursor",
        "last_applied_actions",
        "last_moved",
        "last_gathered",
        "last_policy_refill_calls",
        "last_policy_refill_rows",
        "pending_generation_free_agent_ids",
        "pending_generation_free_slots",
        "last_world",
    ],
)
def test_source_projection_digest_changes_for_native_top_level_continuation_field(
    field: str,
) -> None:
    baseline = _flat_source_projection()
    changed = _flat_source_projection()
    if isinstance(changed[field], list):
        changed[field][0] += 1
    elif isinstance(changed[field], str):
        changed[field] += "_changed"
    else:
        changed[field] += 1

    assert checkpoint_source_projection_digest_manifest(changed)["digest"] != (
        checkpoint_source_projection_digest_manifest(baseline)["digest"]
    )


def test_source_projection_digest_includes_future_top_level_field() -> None:
    baseline = _flat_source_projection()
    changed = _flat_source_projection()
    changed["future_native_cursor"] = {"epoch": 9}

    manifest = checkpoint_source_projection_digest_manifest(changed)

    assert "future_native_cursor" in manifest["top_level_fields"]
    assert manifest["digest"] != checkpoint_source_projection_digest_manifest(baseline)["digest"]


@pytest.mark.parametrize(
    "field",
    [
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
    ],
)
def test_source_projection_validation_requires_complete_action_cache(field: str) -> None:
    projection = _flat_source_projection()
    del projection["tensors"][field]

    with pytest.raises(ValueError, match="complete action-cache tensor fields"):
        validate_checkpoint_source_projection(projection)


@pytest.mark.parametrize(
    ("metadata_updates", "error"),
    [
        ({"agent_grid_runner_phase": "generation_boundary"}, "phase contradiction"),
        ({"agent_grid_authority_epoch": "generation:2:tick:9"}, "epoch contradiction"),
        ({"tick": 5, "agent_grid_authority_epoch": "generation:2:tick:5"}, "exceeds"),
    ],
)
def test_source_projection_validation_rejects_authority_cursor_contradictions(
    metadata_updates: dict[str, object],
    error: str,
) -> None:
    projection = _flat_source_projection()
    projection["metadata"].update(metadata_updates)

    with pytest.raises(ValueError, match=error):
        validate_checkpoint_source_projection(projection)


def test_source_projection_validation_requires_native_top_level_authority() -> None:
    projection = _flat_source_projection()
    del projection["event_cursor"]

    with pytest.raises(ValueError, match="top-level authority fields"):
        validate_checkpoint_source_projection(projection)


def test_layered_digest_semantics_ignore_native_top_level_continuation_fields() -> None:
    baseline = _flat_source_projection()
    changed = _flat_source_projection()
    changed["event_cursor"] += 1
    changed["pending_generation_free_agent_ids"][0] += 1

    fields = {
        "alive_mask",
        "slot_tokens",
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
    }
    baseline_digest = checkpoint_digest_manifest(
        baseline,
        fields=fields,
        layer=DigestLayer.TRAJECTORY_STATE,
    )["digest"]
    changed_digest = checkpoint_digest_manifest(
        changed,
        fields=fields,
        layer=DigestLayer.TRAJECTORY_STATE,
    )["digest"]

    assert changed_digest == baseline_digest
