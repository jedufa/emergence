from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from native_runtime.checkpoint_digests import (
    DEFAULT_FIELD_POLICIES,
    SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
    SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
    SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
    DigestLayer,
    checkpoint_digest_manifest,
)
from native_runtime.compact_summary_checkpoint_descriptor_plan import (
    CUDA_HASH_BATCH_CAPACITY,
    FLAT_PROJECTION_FIXTURE,
    FLAT_PROJECTION_FIXTURE_SHA256,
    MODE_ACTIVE_PREFIX,
    MODE_DEAD_ROW_ZERO,
    MODE_EXACT,
    MODE_TORCH_RNG_PACKED_PREFIX,
    WORLD_FLAT,
    WORLD_PATCH,
    CompactSummaryCheckpointDescriptorPlanError,
    descriptor_plan_json_bytes,
    generate_compact_summary_checkpoint_descriptor_plan,
    validate_compact_summary_checkpoint_descriptor_plan,
)
from native_runtime.compact_summary_digests import (
    active_agent_ids_digest_from_tensor_sha256,
    active_agent_ids_digest_sha256,
    checkpoint_layer_digest_from_tensor_manifest,
    checkpoint_rng_digest_from_tensor_manifest,
    checkpoint_source_projection_digest_from_tensor_manifest,
)
from native_runtime.engine import decode_bundled_native_checkpoint_projection
from perf.cuda_architecture_b_state_bank import load_accepted_state_bank_manifest


ROOT = Path(__file__).resolve().parents[1]
PLAN_JSON = (
    ROOT
    / "docs/300_architecture-specifications/370_specifications/EIEL-DOC-000840--compact-summary-checkpoint-descriptor-plan-v1.json"
)
PATCH_ONLY_FIELDS = {
    *SOURCE_PATCH_REQUIRED_WORLD_TENSOR_FIELDS,
    *SOURCE_PATCH_P2_REQUIRED_WORLD_TENSOR_FIELDS,
    *SOURCE_PATCH_DERIVED_TENSOR_FIELDS,
}
FIXTURE_LAYER_DIGESTS = {
    DigestLayer.TRAJECTORY_STATE: (
        "2c0b9e5e154982630b74dbe5c948b25565002502816f92c62ca2bfcd5390f012"
    ),
    DigestLayer.CANONICAL_CHECKPOINT_STORAGE: (
        "d6ed440e73f8a6cd0019bbb0bd3b85deeccbe329a0d9bd1e22e7390d1903b5cc"
    ),
    DigestLayer.DIAGNOSTIC_PROJECTION: (
        "0865d3094145445bf6d3c36e27c0d105c3e619f942b3a155f727011c25a854e4"
    ),
}


@pytest.fixture(scope="module")
def plan() -> dict:
    return generate_compact_summary_checkpoint_descriptor_plan()


@pytest.fixture(scope="module")
def fixture_sidecar() -> dict:
    if not FLAT_PROJECTION_FIXTURE.is_file():
        pytest.skip(
            "the immutable external Flat checkpoint projection fixture is unavailable"
        )
    raw = FLAT_PROJECTION_FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FLAT_PROJECTION_FIXTURE_SHA256
    return json.loads(raw)


def test_all_flat_authorities_are_covered_once_and_tapes_are_explicit(
    plan: dict,
) -> None:
    manifest = load_accepted_state_bank_manifest()
    expected_flat = {field.name for field in manifest.fields} - PATCH_ONLY_FIELDS
    checkpoint_rows = [
        row
        for row in plan["descriptors"]
        if row["source_scope"] == "checkpoint_projection"
    ]
    tape_rows = [
        row
        for row in plan["descriptors"]
        if row["source_scope"] == "policy_mutation_tape"
    ]

    assert len(manifest.fields) == 139
    assert len(expected_flat) == 127
    assert len(checkpoint_rows) == len({row["tensor_name"] for row in checkpoint_rows})
    assert {row["tensor_name"] for row in checkpoint_rows} == expected_flat
    assert {row["tensor_name"] for row in tape_rows} == {
        "policy_birth_crossover_mask_tape",
        "policy_birth_delta_tape",
        "policy_birth_mutation_mask_tape",
        "policy_birth_post_rng_state_tape",
        "policy_birth_semantic_draws_tape",
    }
    active_ids = next(
        row for row in checkpoint_rows if row["tensor_name"] == "agent_ids_by_alive_order"
    )
    assert active_ids["descriptor_id"] == "checkpoint:agent_ids_by_alive_order"
    assert active_ids["physical_source_name"] == "active_agent_ids"
    assert active_ids["hash_mode"] == "active_prefix"
    assert active_ids["active_length_source"] == "active_count_state"
    assert active_ids["raw_canonical_mode"] == MODE_ACTIVE_PREFIX
    assert active_ids["length_authority"] == "active_count_state"
    assert active_ids["raw_shape_symbols"] == ["capacity"]
    assert active_ids["canonical_shape_symbols"] == ["alive_count"]
    assert active_ids["digest_inclusion"]["active_agent_ids"] is True
    assert active_ids["digest_inclusion"]["active_agent_ids_output"] == "canonical"

    physical_overrides = {
        row["tensor_name"]: row["physical_source_name"]
        for row in plan["descriptors"]
        if row["physical_source_name"] != row["tensor_name"]
    }
    assert physical_overrides == {
        "agent_id_to_slot_agent_ids": "active_agent_ids",
        "agent_id_to_slot_slots": "active_agent_slots",
        "agent_ids_by_alive_order": "active_agent_ids",
        "policy_birth_post_rng_state_tape": "policy_birth_post_rng_state_packed_tape",
    }

    post_state = next(
        row for row in tape_rows if row["tensor_name"] == "policy_birth_post_rng_state_tape"
    )
    assert post_state["descriptor_id"] == (
        "policy_mutation_tape:policy_birth_post_rng_state_tape"
    )
    assert post_state["physical_source_name"] == "policy_birth_post_rng_state_packed_tape"
    assert post_state["raw_shape_symbols"] == [
        "policy_birth_tape_capacity",
        "packed_torch_rng_state_bytes",
    ]
    assert post_state["canonical_shape_symbols"] == [
        "policy_birth_tape_cursor",
        "mutation_rng_state_bytes",
    ]
    assert post_state["raw_canonical_mode"] == MODE_TORCH_RNG_PACKED_PREFIX
    representation = post_state["physical_representation"]
    assert representation == {
        "field_name": "policy_birth_post_rng_state_packed_tape",
        "encoding": "torch_cpu_rng_state_5056_u32_middle_v1",
        "abi_schema_version": "eiel_torch_cpu_mt19937_state_2560_abi_v1",
        "abi_fingerprint_sha256": (
            "dd3966ffb664fc6bb7b6c99a42436a8945e8b0fc4a0adf15238de16d07ddf5b4"
        ),
        "row_bytes": 2560,
        "canonical_semantic_field": "policy_birth_post_rng_state_tape",
        "canonical_row_bytes": 5056,
        "canonical_hash_bytes": "exact reconstructed 5056-byte Torch CPU RNG state rows",
        "full_dense_tape_materialization_allowed": False,
        "canonical_materialization_rule": (
            "bounded row decode only; complete dense post-state tape materialization "
            "is forbidden"
        ),
        "codec_route_binding": "execution_route_sha256",
    }


def test_patch_p2_plan_uses_exact_accepted_state_bank_registry(plan: dict) -> None:
    patch_plan = generate_compact_summary_checkpoint_descriptor_plan(
        world=WORLD_PATCH
    )
    manifest = load_accepted_state_bank_manifest()
    accepted = {field.name for field in manifest.fields}
    flat_fields = {
        row["tensor_name"]
        for row in plan["descriptors"]
        if row["source_scope"] == "checkpoint_projection"
    }
    patch_rows = [
        row
        for row in patch_plan["descriptors"]
        if row["source_scope"] == "checkpoint_projection"
    ]
    patch_fields = {row["tensor_name"] for row in patch_rows}

    assert patch_plan["status"] == "flat_patch_p2_ready"
    assert patch_plan["source_contract"]["descriptor_world"] == WORLD_PATCH
    assert patch_plan["source_contract"]["patch_profile"] == "P2"
    assert patch_fields == accepted
    assert patch_fields - flat_fields == PATCH_ONLY_FIELDS
    assert len(patch_rows) == len(patch_fields) == 139
    assert patch_plan["counts"] == {
        "accepted_state_bank_authorities": 139,
        "flat_excluded_patch_authorities": 12,
        "patch_p2_checkpoint_tensor_authorities": 139,
        "policy_mutation_tape_fields": 5,
        "descriptors": 144,
        "batches": 9,
        "mode_counts": {
            MODE_EXACT: 23,
            MODE_DEAD_ROW_ZERO: 90,
            MODE_ACTIVE_PREFIX: 30,
            MODE_TORCH_RNG_PACKED_PREFIX: 1,
        },
        "digest_inclusion_counts": {
            "source_projection": 139,
            "checkpoint_rng": 1,
            "active_agent_ids": 1,
            "policy_mutation_tape": 5,
            "checkpoint_layer_each": 134,
        },
        "unresolved_metadata": 0,
    }
    derived = {
        row["tensor_name"]: row
        for row in patch_rows
        if row["tensor_name"] in SOURCE_PATCH_DERIVED_TENSOR_FIELDS
    }
    assert set(derived) == set(SOURCE_PATCH_DERIVED_TENSOR_FIELDS)
    assert all(row["raw_canonical_mode"] == MODE_EXACT for row in derived.values())
    assert all(
        row["digest_inclusion"]["source_projection"] is True
        and row["digest_inclusion"]["checkpoint_layers"] == []
        for row in derived.values()
    )
    validate_compact_summary_checkpoint_descriptor_plan(patch_plan)
    validate_compact_summary_checkpoint_descriptor_plan(
        patch_plan,
        world=WORLD_PATCH,
    )


def test_active_prefix_digest_matches_trimmed_canonical_projection() -> None:
    active_agent_ids = np.array([2, 7, 11, -1, -1], dtype=np.int64)
    active_count_state = np.array([3], dtype=np.int64)
    active_count = int(active_count_state[0])
    canonical_projection = np.ascontiguousarray(active_agent_ids[:active_count])

    physical_prefix_bytes = active_agent_ids.view(np.uint8)[
        : active_count * active_agent_ids.dtype.itemsize
    ]
    physical_prefix_sha256 = hashlib.sha256(physical_prefix_bytes).hexdigest()
    canonical_projection_sha256 = hashlib.sha256(
        canonical_projection.tobytes(order="C")
    ).hexdigest()

    assert physical_prefix_sha256 == canonical_projection_sha256
    assert active_agent_ids_digest_from_tensor_sha256(
        active_count,
        physical_prefix_sha256,
    ) == active_agent_ids_digest_sha256(canonical_projection)


def test_indices_modes_digest_layers_and_batches_are_frozen(plan: dict) -> None:
    descriptors = plan["descriptors"]
    assert plan["counts"] == {
        "accepted_state_bank_authorities": 139,
        "flat_excluded_patch_authorities": 12,
        "flat_checkpoint_tensor_authorities": 127,
        "policy_mutation_tape_fields": 5,
        "descriptors": 132,
        "batches": 9,
        "mode_counts": {
            MODE_EXACT: 11,
            MODE_DEAD_ROW_ZERO: 90,
            MODE_ACTIVE_PREFIX: 30,
            MODE_TORCH_RNG_PACKED_PREFIX: 1,
        },
        "digest_inclusion_counts": {
            "source_projection": 127,
            "checkpoint_rng": 1,
            "active_agent_ids": 1,
            "policy_mutation_tape": 5,
            "checkpoint_layer_each": 124,
        },
        "unresolved_metadata": 1,
    }
    assert [row["tensor_name"] for row in descriptors] == sorted(
        row["tensor_name"] for row in descriptors
    )
    for index, row in enumerate(descriptors):
        assert row["descriptor_index"] == index
        assert row["raw_output_index"] == index
        assert row["canonical_output_index"] == index
        assert (row["batch_index"], row["batch_slot"]) == divmod(
            index, CUDA_HASH_BATCH_CAPACITY
        )
    assert [batch["count"] for batch in plan["batches"]] == [16] * 8 + [4]
    assert all(batch["count"] <= CUDA_HASH_BATCH_CAPACITY for batch in plan["batches"])
    assert plan["unresolved_metadata"] == [
        {
            "scope": "patch_source_projection",
            "status": "explicit_projection_pending",
            "reason": (
                "checkpoint_source_projection_digest_from_tensor_manifest currently "
                "accepts Flat only; no Patch source projection is guessed"
            ),
        }
    ]


def test_outer_manifest_device_and_dtype_semantics_are_explicit(plan: dict) -> None:
    semantics = plan["outer_manifest_semantics"]
    source = semantics["source_projection_prehashed_tensor_manifest"]
    stored = semantics["stored_projection_tensor_manifest"]
    layers = semantics["checkpoint_layer_tensor_manifest"]
    assert source["device"] == "cpu"
    assert source["dtype"] == "NumPy spelling from descriptor.dtype"
    assert stored["device"] == "cpu"
    assert stored["is_cuda"] is False
    assert stored["dtype"].startswith("C++ Torch spelling")
    assert layers["device_field"] == "omitted"
    assert layers["shape_and_sha256"] == "canonical output"
    assert semantics["host_finalization"]["status"] == (
        "required_and_supported_for_flat"
    )


def test_generated_json_exactly_matches_code() -> None:
    assert PLAN_JSON.read_bytes() == descriptor_plan_json_bytes()
    assert descriptor_plan_json_bytes() == descriptor_plan_json_bytes(world=WORLD_FLAT)


def test_source_registries_and_generated_plan_fail_closed(plan: dict) -> None:
    manifest = load_accepted_state_bank_manifest()
    duplicate_manifest = replace(manifest, fields=(*manifest.fields, manifest.fields[0]))
    with pytest.raises(CompactSummaryCheckpointDescriptorPlanError, match="duplicate"):
        generate_compact_summary_checkpoint_descriptor_plan(manifest=duplicate_manifest)

    missing_manifest = replace(manifest, fields=manifest.fields[1:])
    with pytest.raises(CompactSummaryCheckpointDescriptorPlanError, match="missing"):
        generate_compact_summary_checkpoint_descriptor_plan(manifest=missing_manifest)

    missing_policies = dict(DEFAULT_FIELD_POLICIES)
    missing_policies.pop("energy")
    with pytest.raises(CompactSummaryCheckpointDescriptorPlanError, match="unclassified"):
        generate_compact_summary_checkpoint_descriptor_plan(policies=missing_policies)

    contradictory_policies = dict(DEFAULT_FIELD_POLICIES)
    contradictory_policies["event_tape_actor_id"] = replace(
        contradictory_policies["event_tape_actor_id"],
        canonicalization="preserve_exact_stored",
    )
    with pytest.raises(CompactSummaryCheckpointDescriptorPlanError, match="prefix"):
        generate_compact_summary_checkpoint_descriptor_plan(
            policies=contradictory_policies
        )

    missing_tape = {
        "policy_birth_crossover_mask_tape": (np.dtype(np.bool_), 2),
    }
    with pytest.raises(CompactSummaryCheckpointDescriptorPlanError, match="field mismatch"):
        generate_compact_summary_checkpoint_descriptor_plan(
            policy_tape_registry=missing_tape
        )

    malformed = copy.deepcopy(plan)
    malformed["descriptors"][0]["canonical_output_index"] = 1
    with pytest.raises(CompactSummaryCheckpointDescriptorPlanError, match="exactly match"):
        validate_compact_summary_checkpoint_descriptor_plan(malformed)

    with pytest.raises(CompactSummaryCheckpointDescriptorPlanError, match="world"):
        generate_compact_summary_checkpoint_descriptor_plan(world="jungle")

    patch_plan = generate_compact_summary_checkpoint_descriptor_plan(
        world=WORLD_PATCH
    )
    malformed_patch = copy.deepcopy(patch_plan)
    malformed_patch["source_contract"]["descriptor_world"] = "jungle"
    with pytest.raises(CompactSummaryCheckpointDescriptorPlanError, match="world"):
        validate_compact_summary_checkpoint_descriptor_plan(malformed_patch)

    with pytest.raises(CompactSummaryCheckpointDescriptorPlanError, match="exactly match"):
        validate_compact_summary_checkpoint_descriptor_plan(
            plan,
            world=WORLD_PATCH,
        )

    contradictory_patch_policies = dict(DEFAULT_FIELD_POLICIES)
    contradictory_patch_policies["patch_visibility"] = replace(
        contradictory_patch_policies["patch_visibility"],
        canonicalization="preserve_exact_stored",
    )
    with pytest.raises(
        CompactSummaryCheckpointDescriptorPlanError,
        match="authoritative registry",
    ):
        generate_compact_summary_checkpoint_descriptor_plan(
            policies=contradictory_patch_policies,
            world=WORLD_PATCH,
        )


def _prehashed_source_tensor_manifest(projection: dict) -> dict[str, dict]:
    return {
        name: {
            "device": row["device"],
            "dtype": row["dtype"],
            "shape": row["shape"],
            "sha256": row["sha256"],
        }
        for name, row in projection["tensors"].items()
    }


def test_host_outer_digest_helpers_rebuild_exact_flat_fixture(
    plan: dict,
    fixture_sidecar: dict,
) -> None:
    packed_projection = fixture_sidecar["projection"]
    tensor_manifest = _prehashed_source_tensor_manifest(packed_projection)
    projection_fields = {
        name: value
        for name, value in packed_projection.items()
        if name != "tensors"
    }
    checkpoint_rows = {
        row["tensor_name"]: row
        for row in plan["descriptors"]
        if row["source_scope"] == "checkpoint_projection"
    }

    assert set(tensor_manifest) == set(checkpoint_rows) - {
        "early_energy_collected_exact",
        "energy_collected_exact",
    }
    for name, packed in tensor_manifest.items():
        descriptor = checkpoint_rows[name]
        assert packed["device"] == "cpu"
        assert packed["dtype"] == descriptor["dtype"]
        assert len(packed["shape"]) == descriptor["raw_rank"]

    assert checkpoint_source_projection_digest_from_tensor_manifest(
        projection_fields=projection_fields,
        tensor_manifest=tensor_manifest,
    ) == fixture_sidecar["source_projection_sha256"]
    assert fixture_sidecar["source_projection_sha256"] == (
        "fa3eea55932cec463d13b2e032e978c19d8f04a8dea1f143370736c8f7dc8301"
    )
    assert checkpoint_rng_digest_from_tensor_manifest(
        metadata=packed_projection["metadata"],
        tensor_manifest=tensor_manifest,
    ) == fixture_sidecar["source_rng_sha256"]
    assert fixture_sidecar["source_rng_sha256"] == (
        "e771372041de29a073005bb2d6a1d69ca91b9b225cc5cbe22b6cdca99c1e836a"
    )

    decoded = decode_bundled_native_checkpoint_projection(fixture_sidecar)
    metadata = decoded["metadata"]
    for layer, expected_digest in FIXTURE_LAYER_DIGESTS.items():
        direct = checkpoint_digest_manifest(decoded, layer=layer)
        assert direct["field_count"] == 122
        assert direct["digest"] == expected_digest
        rebuilt = checkpoint_layer_digest_from_tensor_manifest(
            layer=layer,
            projection_schema_version=decoded["schema_version"],
            world=metadata["world"],
            generation=metadata["generation"],
            tick=metadata["tick"],
            fields=direct["fields"],
        )
        assert rebuilt == expected_digest
