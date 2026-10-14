from __future__ import annotations

import json
from copy import deepcopy
from math import prod

import pytest

from engine.perf.cuda_architecture_b_checkpoint_clone_v1 import (
    ACCEPTED_STATE_BANK_MANIFEST_SHA256_V1,
    CHECKPOINT_CLONE_PLAN_SCHEMA_VERSION_V1,
    CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_REQUIRED_FIELDS_V1,
    CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_SCHEMA_VERSION_V1,
    LINEAGE_EXACT_COPY_CLASSIFICATION_V1,
    LINEAGE_SEMANTIC_CLASSIFICATION_V1,
    RNG_EXACT_COPY_CLASSIFICATION_V1,
    RNG_SEMANTIC_CLASSIFICATION_V1,
    CheckpointCloneContractError,
    CheckpointClonePlanV1,
    CheckpointCloneZeroD2HReceiptV1,
    build_accepted_checkpoint_clone_state_manifest_v1,
    build_checkpoint_clone_zero_d2h_receipt_v1,
    plan_checkpoint_clone_v1,
)


SHAPE_BINDINGS = {
    "capacity": 8,
    "cell": 48,
    "chemistry_channels": 2,
    "drive_channels": 4,
    "event_capacity": 32,
    "gru_hidden": 4,
    "gru_pack_width": 16,
    "height": 4,
    "mlp_pack_width": 16,
    "mutation_rng_state_bytes": 32,
    "policy_tick_stride": 1,
    "width": 4,
}


def _manifest() -> dict[str, object]:
    return build_accepted_checkpoint_clone_state_manifest_v1(SHAPE_BINDINGS)


def _descriptors() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    pointer = 0x100000
    for field in _manifest()["fields"]:
        shape = tuple(field["shape"])
        element_size = int(field["element_size_bytes"])
        bytes_per_cell = prod(shape[1:]) * element_size
        result.append(
            {
                "field_name": field["field_name"],
                "pointer_identity": pointer,
                "shape": shape,
                "cell_stride": bytes_per_cell,
                "bytes_per_cell": bytes_per_cell,
                "element_size_bytes": element_size,
            }
        )
        pointer += bytes_per_cell * shape[0] + 0x1000
    return result


def _replace(
    descriptors: list[dict[str, object]],
    replacement: dict[str, object],
) -> list[dict[str, object]]:
    name = replacement["field_name"]
    return [replacement if row["field_name"] == name else row for row in descriptors]


def _by_name(descriptors: list[dict[str, object]], name: str) -> dict[str, object]:
    return next(row for row in descriptors if row["field_name"] == name)


def _plan(
    descriptors: list[dict[str, object]] | None = None,
    *,
    state_manifest: dict[str, object] | None = None,
    generation_boundary_complete: bool = True,
    source_cell_index: int = 0,
    target_cell_indices: object = (1,),
):
    return plan_checkpoint_clone_v1(
        _descriptors() if descriptors is None else descriptors,
        state_manifest=_manifest() if state_manifest is None else state_manifest,
        generation_boundary_complete=generation_boundary_complete,
        source_cell_index=source_cell_index,
        target_cell_indices=target_cell_indices,
    )


def test_one_source_to_47_targets_is_complete_and_deterministic() -> None:
    first = _plan(target_cell_indices=reversed(range(1, 48)))
    second = _plan(
        list(reversed(_descriptors())),
        target_cell_indices=range(1, 48),
    )

    assert first.target_cell_indices == tuple(range(1, 48))
    assert first.logical_field_count == 139
    assert first.physical_field_count == 139
    assert first.clone_entry_count == 139 * 47
    assert first.to_dict() == second.to_dict()
    assert first.plan_sha256 == second.plan_sha256
    payload = first.to_dict()
    assert payload["schema_version"] == CHECKPOINT_CLONE_PLAN_SCHEMA_VERSION_V1
    assert payload["manifest_completeness_contract"] == {
        "authority_manifest_sha256": ACCEPTED_STATE_BANK_MANIFEST_SHA256_V1,
        "required_logical_field_count": 139,
        "provided_logical_field_count": 139,
        "accepted_tensor_state_complete": True,
        "accepted_non_tensor_state_surface_count": 21,
        "non_tensor_state_device_clone_qualified": False,
        "full_semantic_checkpoint_clone_qualified": False,
        "field_sets_exact": True,
        "shape_byte_widths_exact": True,
    }
    json.dumps(payload, sort_keys=True)


def test_aliases_are_deduplicated_only_with_identical_layout() -> None:
    descriptors = _descriptors()
    energy = deepcopy(_by_name(descriptors, "energy"))
    health = deepcopy(_by_name(descriptors, "health"))
    health["pointer_identity"] = energy["pointer_identity"]
    health["cell_stride"] = energy["cell_stride"]
    health["bytes_per_cell"] = energy["bytes_per_cell"]
    descriptors = _replace(descriptors, health)

    plan = _plan(descriptors)
    aliases = next(
        field.field_aliases
        for field in plan.physical_fields
        if "energy" in field.field_aliases
    )
    assert aliases == ("energy", "health")
    assert plan.physical_field_count == 138


def test_cyclic_duplicate_and_out_of_bounds_targets_are_rejected() -> None:
    with pytest.raises(CheckpointCloneContractError, match="cyclic/self target"):
        _plan(source_cell_index=1, target_cell_indices=(2, 1))
    with pytest.raises(CheckpointCloneContractError, match="duplicates"):
        _plan(target_cell_indices=(1, 1))
    with pytest.raises(CheckpointCloneContractError, match="source_cell_index"):
        _plan(source_cell_index=48)
    with pytest.raises(CheckpointCloneContractError, match="outside descriptor"):
        _plan(target_cell_indices=(48,))


def test_source_preservation_is_structural_and_not_an_execution_claim() -> None:
    plan = _plan(target_cell_indices=range(1, 48))
    preservation = plan.to_dict()["source_preservation_contract"]

    assert preservation["source_cell_is_read_only"] is True
    assert preservation["source_write_entry_count"] == 0
    assert preservation["source_preserved_by_contract"] is True
    assert all(entry.target_cell_index != 0 for entry in plan.entries)
    assert plan.cuda_execution_performed is False
    assert plan.simulation_parity_claimed is False


def test_lineage_and_rng_fields_are_mandatory_exact_copy_classes() -> None:
    plan = _plan()
    by_name = {field.field_name: field for field in plan.physical_fields}

    lineage = by_name["lineage_founder_id"]
    assert lineage.semantic_classification == LINEAGE_SEMANTIC_CLASSIFICATION_V1
    assert lineage.copy_classification == LINEAGE_EXACT_COPY_CLASSIFICATION_V1
    rng = by_name["rng_manager_mutation_torch_cpu_state"]
    assert rng.semantic_classification == RNG_SEMANTIC_CLASSIFICATION_V1
    assert rng.copy_classification == RNG_EXACT_COPY_CLASSIFICATION_V1


def test_generation_boundary_and_manifest_authority_are_required() -> None:
    with pytest.raises(CheckpointCloneContractError, match="must be exactly true"):
        _plan(generation_boundary_complete=False)

    manifest = _manifest()
    manifest["accepted_state_bank_manifest_sha256"] = "00" * 32
    with pytest.raises(CheckpointCloneContractError, match="immutable accepted"):
        _plan(state_manifest=manifest)


def test_manifest_and_descriptor_omissions_fail_closed() -> None:
    manifest = _manifest()
    manifest["fields"] = manifest["fields"][:-1]
    with pytest.raises(CheckpointCloneContractError, match="immutable accepted"):
        _plan(state_manifest=manifest)

    descriptors = [
        row
        for row in _descriptors()
        if row["field_name"] != "rng_manager_mutation_torch_cpu_state"
    ]
    with pytest.raises(CheckpointCloneContractError, match="missing=.*rng_manager"):
        _plan(descriptors)


def test_shape_binding_and_descriptor_width_mismatch_fail_closed() -> None:
    manifest = _manifest()
    manifest["shape_bindings"]["capacity"] = 4
    with pytest.raises(CheckpointCloneContractError, match="immutable accepted"):
        _plan(state_manifest=manifest)

    descriptors = _descriptors()
    energy = deepcopy(_by_name(descriptors, "energy"))
    energy["shape"] = (48, 1)
    energy["bytes_per_cell"] = 4
    energy["cell_stride"] = 4
    descriptors = _replace(descriptors, energy)
    with pytest.raises(CheckpointCloneContractError, match="frozen manifest"):
        _plan(descriptors)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("overlap", "physical allocation overlap"),
        ("alias", "alias ambiguity"),
        ("duplicate", "one logical field resolves to multiple pointers"),
        ("overflow", "overflows uint64"),
        ("stride", "overlaps adjacent cells"),
    ],
)
def test_overlap_overflow_and_alias_ambiguity_fail_closed(
    mutation: str,
    message: str,
) -> None:
    descriptors = _descriptors()
    energy = deepcopy(_by_name(descriptors, "energy"))
    health = deepcopy(_by_name(descriptors, "health"))
    if mutation == "overlap":
        health["pointer_identity"] = int(energy["pointer_identity"]) + 4
        descriptors = _replace(descriptors, health)
    elif mutation == "alias":
        health["pointer_identity"] = energy["pointer_identity"]
        health["cell_stride"] = int(energy["cell_stride"]) * 2
        descriptors = _replace(descriptors, health)
    elif mutation == "duplicate":
        duplicate = deepcopy(energy)
        duplicate["pointer_identity"] = int(energy["pointer_identity"]) + 0x800
        descriptors.append(duplicate)
    elif mutation == "overflow":
        energy["pointer_identity"] = (1 << 64) - 16
        descriptors = _replace(descriptors, energy)
    else:
        energy["cell_stride"] = max(1, int(energy["bytes_per_cell"]) - 1)
        descriptors = _replace(descriptors, energy)

    with pytest.raises(CheckpointCloneContractError, match=message):
        _plan(descriptors)


def test_lineage_names_cannot_be_downgraded_to_generic_state() -> None:
    descriptors = _descriptors()
    lineage = deepcopy(_by_name(descriptors, "lineage_founder_id"))
    lineage["semantic_classification"] = "state"
    lineage["copy_classification"] = "state_exact_copy"

    with pytest.raises(CheckpointCloneContractError, match="contradicts inferred"):
        _plan(_replace(descriptors, lineage))


def test_host_receipt_builder_cannot_mint_cuda_execution_success() -> None:
    plan = _plan()
    with pytest.raises(CheckpointCloneContractError, match="cannot authenticate"):
        build_checkpoint_clone_zero_d2h_receipt_v1(
            plan,
            executed_copy_operations=plan.clone_entry_count,
            executed_copy_bytes=plan.total_copy_bytes,
            device_to_host_operations=0,
            device_to_host_bytes=0,
            source_preserved=True,
            hidden_host_materialization=False,
            execution_performed=True,
        )

    receipt = build_checkpoint_clone_zero_d2h_receipt_v1(
        plan,
        executed_copy_operations=0,
        executed_copy_bytes=0,
        device_to_host_operations=0,
        device_to_host_bytes=0,
        source_preserved=False,
        hidden_host_materialization=False,
        execution_performed=False,
    )
    payload = receipt.to_dict()
    assert tuple(payload) == CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_REQUIRED_FIELDS_V1
    assert payload["schema_version"] == CHECKPOINT_CLONE_ZERO_D2H_RECEIPT_SCHEMA_VERSION_V1
    assert receipt.zero_d2h_qualified is False
    assert payload["simulation_parity_claimed"] is False


def test_public_dataclass_constructors_cannot_bypass_validated_factories() -> None:
    with pytest.raises(TypeError):
        CheckpointClonePlanV1(state_manifest_sha256="00" * 32)
    with pytest.raises(TypeError):
        CheckpointCloneZeroD2HReceiptV1(execution_performed=True)
