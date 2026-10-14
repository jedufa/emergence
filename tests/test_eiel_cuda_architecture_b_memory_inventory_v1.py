from __future__ import annotations

import json
from typing import Any

import pytest

from engine.perf.cuda_architecture_b_memory_inventory_v1 import (
    AUTHORITY_BINDING_FIELDS_V1,
    MAX_ACCOUNTED_BYTES_V1,
    MEMORY_INVENTORY_SCHEMA_VERSION_V1,
    MemoryInventoryContractError,
    TensorStorageMetadataV1,
    batch_sizing_bytes_from_memory_inventory_v1,
    build_cuda_architecture_b_memory_inventory_v1,
)


_EXTENSION_SHA256 = "a" * 64
_STATE_LAYOUT_SHA256 = "b" * 64
_DEFAULT_BINDING = object()


def _authority_binding(**overrides: Any) -> dict[str, Any]:
    binding = {
        "world": "patch",
        "profile": "patch_p2",
        "build_identity": "architecture_b_v8",
        "extension_sha256": _EXTENSION_SHA256,
        "state_layout_sha256": _STATE_LAYOUT_SHA256,
        "active_policy_family": "fixed_mlp",
        "generation_count": 64,
        "ticks_per_generation": 80,
        "summary_mode": "compact",
        "tick_output_storage": "rolling_single_row",
    }
    binding.update(overrides)
    return binding


def _tensor(
    name: str,
    storage_id: str,
    storage_nbytes: int,
    *,
    allocation_scope: str = "per_cell",
    memory_tier: str = "device",
    policy_family: str = "non_policy",
    role: str = "resident_active",
    canonical_checkpoint: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "storage_id": storage_id,
        "storage_nbytes": storage_nbytes,
        "allocation_scope": allocation_scope,
        "memory_tier": memory_tier,
        "policy_family": policy_family,
        "role": role,
        "canonical_checkpoint": canonical_checkpoint,
    }


def _fixed_mlp_inventory() -> list[dict[str, Any]]:
    return [
        _tensor(
            "mlp_pack",
            "device:mlp-pack",
            400,
            policy_family="fixed_mlp",
        ),
        _tensor(
            "mlp_pack_turnover_source_alias",
            "device:mlp-pack",
            400,
            policy_family="fixed_mlp",
        ),
        _tensor(
            "gru_pack",
            "host:gru-pack",
            160,
            memory_tier="host",
            policy_family="gru",
            role="host_checkpoint_only",
        ),
        _tensor(
            "tick_scratch",
            "device:tick-scratch",
            80,
            role="rolling_scratch",
            canonical_checkpoint=False,
        ),
        _tensor(
            "event_history",
            "device:event-history",
            120,
            role="retained_history",
        ),
        _tensor(
            "turnover_shadow",
            "device:turnover-shadow",
            40,
            role="turnover_shadow",
            canonical_checkpoint=False,
        ),
        _tensor(
            "policy_episode_tape",
            "device:policy-tape",
            64,
            allocation_scope="shared",
            policy_family="fixed_mlp",
            role="immutable_tape",
            canonical_checkpoint=False,
        ),
        _tensor(
            "fertility",
            "device:fertility",
            16,
            allocation_scope="shared",
        ),
    ]


def _build(
    metadata: list[dict[str, Any]] | tuple[TensorStorageMetadataV1, ...],
    *,
    active_policy_family: str = "fixed_mlp",
    cell_count: int = 4,
    inventory_complete: bool = True,
    authority_binding: object = _DEFAULT_BINDING,
) -> dict[str, Any]:
    binding = (
        _authority_binding(active_policy_family=active_policy_family)
        if authority_binding is _DEFAULT_BINDING
        else authority_binding
    )
    return build_cuda_architecture_b_memory_inventory_v1(
        metadata,
        cell_count=cell_count,
        active_policy_family=active_policy_family,
        inventory_complete=inventory_complete,
        authority_binding=binding,  # type: ignore[arg-type]
    )


def test_deduplicates_aliases_and_emits_batch_sizing_truth() -> None:
    receipt = _build(_fixed_mlp_inventory())

    assert receipt["schema_version"] == MEMORY_INVENTORY_SCHEMA_VERSION_V1
    assert receipt["status"] == "accepted"
    assert tuple(receipt["authority_binding"]) == AUTHORITY_BINDING_FIELDS_V1
    assert len(receipt["authority_binding_sha256"]) == 64
    assert len(receipt["receipt_sha256"]) == 64
    assert receipt["logical_tensor_count"] == 8
    assert receipt["physical_storage_count"] == 7
    assert receipt["deduplicated_alias_count"] == 1
    assert receipt["totals"]["observed_bytes"] == 880
    assert receipt["totals"]["per_cell_bytes"] == 200
    assert receipt["totals"]["shared_bytes"] == 80
    assert receipt["totals"]["by_memory_tier"]["device"] == {
        "observed_bytes": 720,
        "per_cell_bytes": 160,
        "shared_bytes": 80,
    }
    assert receipt["totals"]["by_memory_tier"]["host"] == {
        "observed_bytes": 160,
        "per_cell_bytes": 40,
        "shared_bytes": 0,
    }
    assert receipt["automatic_batch_sizing"] == {
        "planning_usable": True,
        "production_usable": False,
        "per_cell_bytes": 160,
        "shared_bytes": 80,
        "basis": "deduplicated_device_physical_storage",
        "host_bytes_excluded": 160,
    }
    with pytest.raises(MemoryInventoryContractError, match="opaque native CUDA"):
        batch_sizing_bytes_from_memory_inventory_v1(
            receipt,
            expected_authority_binding=_authority_binding(),
        )


def test_classifies_every_role_and_keeps_inactive_gru_canonical_on_host() -> None:
    receipt = _build(_fixed_mlp_inventory())
    gru = receipt["by_policy_family"]["gru"]

    assert gru["totals"]["observed_bytes"] == 160
    assert gru["totals"]["by_memory_tier"]["host"]["per_cell_bytes"] == 40
    assert gru["by_role"]["host_checkpoint_only"]["observed_bytes"] == 160
    assert receipt["canonical_checkpoint_totals"]["observed_bytes"] == 696
    assert receipt["contract"]["inactive_gru_is_assumed_discardable"] is False
    assert receipt["contract"]["inactive_policy_storage_rule"] == (
        "canonical_host_checkpoint_only"
    )
    assert {
        storage["role"] for storage in receipt["storages"]
    } == {
        "resident_active",
        "host_checkpoint_only",
        "rolling_scratch",
        "retained_history",
        "turnover_shadow",
        "immutable_tape",
    }


def test_future_gru_and_inactive_fixed_mlp_are_accounted_separately() -> None:
    metadata = (
        TensorStorageMetadataV1(
            name="gru_pack",
            storage_id=0x1000,
            storage_nbytes=256,
            allocation_scope="per_cell",
            memory_tier="device",
            policy_family="gru",
            role="resident_active",
            canonical_checkpoint=True,
        ),
        TensorStorageMetadataV1(
            name="mlp_pack",
            storage_id=0x2000,
            storage_nbytes=128,
            allocation_scope="per_cell",
            memory_tier="host",
            policy_family="fixed_mlp",
            role="host_checkpoint_only",
            canonical_checkpoint=True,
        ),
    )

    receipt = _build(metadata, active_policy_family="gru")

    assert receipt["ok"] is True
    assert receipt["by_policy_family"]["gru"]["totals"]["observed_bytes"] == 256
    assert (
        receipt["by_policy_family"]["fixed_mlp"]["totals"]["observed_bytes"]
        == 128
    )
    assert receipt["automatic_batch_sizing"]["per_cell_bytes"] == 64
    assert receipt["automatic_batch_sizing"]["host_bytes_excluded"] == 128


def test_receipt_is_independent_of_logical_metadata_order() -> None:
    metadata = _fixed_mlp_inventory()

    forward = _build(metadata)
    reverse = _build(list(reversed(metadata)))

    assert forward == reverse
    assert forward["storages"][0]["storage_id"] == "device:event-history"
    assert next(
        storage
        for storage in forward["storages"]
        if storage["storage_id"] == "device:mlp-pack"
    )["logical_names"] == ["mlp_pack", "mlp_pack_turnover_source_alias"]
    json.dumps(forward, sort_keys=True, allow_nan=False)


def test_inactive_gru_cannot_be_silently_resident_or_noncanonical() -> None:
    resident_gru = _tensor(
        "gru_pack",
        "device:gru-pack",
        160,
        policy_family="gru",
    )

    receipt = _build([resident_gru])

    assert receipt["status"] == "rejected"
    assert receipt["automatic_batch_sizing"]["planning_usable"] is False
    assert receipt["automatic_batch_sizing"]["production_usable"] is False
    assert receipt["automatic_batch_sizing"]["per_cell_bytes"] is None
    assert receipt["totals"]["observed_bytes"] == 0
    assert any("inactive gru storage" in error for error in receipt["errors"])


def test_alias_classification_ambiguity_fails_closed_without_partial_totals() -> None:
    metadata = [
        _tensor("energy", "device:state", 64),
        _tensor(
            "energy_turnover_alias",
            "device:state",
            64,
            role="turnover_shadow",
            canonical_checkpoint=False,
        ),
    ]

    receipt = _build(metadata)

    assert receipt["ok"] is False
    assert receipt["storages"] == []
    assert receipt["totals"]["observed_bytes"] == 0
    assert receipt["automatic_batch_sizing"]["shared_bytes"] is None
    assert any("aliases disagree" in error for error in receipt["errors"])


@pytest.mark.parametrize(
    ("metadata", "overrides", "message"),
    [
        ([_tensor("state", "s", 10)], {}, "not divisible"),
        ([_tensor("state", "s", True)], {}, "storage_nbytes"),
        (
            [_tensor("state", "s", 40, role="unknown")],
            {},
            "role",
        ),
        (
            [
                _tensor(
                    "gru_pack",
                    "host:gru",
                    40,
                    memory_tier="host",
                    policy_family="gru",
                    role="host_checkpoint_only",
                    canonical_checkpoint=False,
                )
            ],
            {},
            "must remain canonical",
        ),
        ([_tensor("state", "s", 40)], {"inventory_complete": False}, "exactly true"),
        ([_tensor("state", "s", 40)], {"cell_count": True}, "cell_count"),
        (
            [_tensor("state", "s", 40)],
            {"active_policy_family": "auto"},
            "active_policy_family",
        ),
    ],
)
def test_malformed_or_incomplete_inputs_return_rejected_receipts(
    metadata: list[dict[str, Any]],
    overrides: dict[str, Any],
    message: str,
) -> None:
    arguments = {
        "active_policy_family": "fixed_mlp",
        "cell_count": 4,
        "inventory_complete": True,
        **overrides,
    }

    receipt = _build(metadata, **arguments)

    assert receipt["ok"] is False
    assert receipt["automatic_batch_sizing"]["planning_usable"] is False
    assert receipt["automatic_batch_sizing"]["production_usable"] is False
    assert any(message in error for error in receipt["errors"])


def test_accounting_overflow_fails_closed() -> None:
    receipt = _build(
        [
            _tensor(
                "shared_a",
                "a",
                MAX_ACCOUNTED_BYTES_V1,
                allocation_scope="shared",
            ),
            _tensor("shared_b", "b", 1, allocation_scope="shared"),
        ]
    )

    assert receipt["ok"] is False
    assert receipt["automatic_batch_sizing"]["per_cell_bytes"] is None
    assert any("accounted byte sum exceeds" in error for error in receipt["errors"])


def test_batch_sizing_extractor_refuses_rejected_or_forged_receipts() -> None:
    rejected = _build([], inventory_complete=True)
    with pytest.raises(MemoryInventoryContractError, match="opaque native CUDA"):
        batch_sizing_bytes_from_memory_inventory_v1(
            rejected,
            expected_authority_binding=_authority_binding(),
        )

    accepted = _build([_tensor("state", "s", 40)])
    plausible_forgery = json.loads(json.dumps(accepted))
    plausible_forgery["automatic_batch_sizing"]["per_cell_bytes"] = 39
    plausible_forgery["receipt_sha256"] = "00" * 32
    with pytest.raises(MemoryInventoryContractError, match="opaque native CUDA"):
        batch_sizing_bytes_from_memory_inventory_v1(
            plausible_forgery,
            expected_authority_binding=_authority_binding(),
        )

    huge_forgery = {"automatic_batch_sizing": {"per_cell_bytes": 1 << 10000}}
    with pytest.raises(MemoryInventoryContractError, match="opaque native CUDA"):
        batch_sizing_bytes_from_memory_inventory_v1(
            huge_forgery,
            expected_authority_binding=_authority_binding(),
        )


def test_shared_only_inventory_cannot_masquerade_as_batch_sizing_input() -> None:
    receipt = _build(
        [_tensor("shared", "shared", 40, allocation_scope="shared")]
    )

    assert receipt["ok"] is True
    assert receipt["automatic_batch_sizing"] == {
        "planning_usable": False,
        "production_usable": False,
        "per_cell_bytes": None,
        "shared_bytes": None,
        "basis": "no_device_per_cell_storage",
        "host_bytes_excluded": 0,
    }
    with pytest.raises(MemoryInventoryContractError, match="opaque native CUDA"):
        batch_sizing_bytes_from_memory_inventory_v1(
            receipt,
            expected_authority_binding=_authority_binding(),
        )


@pytest.mark.parametrize(
    "authority_binding",
    [
        None,
        {},
        _authority_binding(extension_sha256="A" * 64),
        _authority_binding(state_layout_sha256="short"),
        {**_authority_binding(), "unexpected": "field"},
        _authority_binding(active_policy_family="gru"),
        _authority_binding(generation_count=1 << 10000),
    ],
)
def test_unbound_or_malformed_authority_identity_fails_closed(
    authority_binding: object,
) -> None:
    receipt = _build(
        [_tensor("state", "s", 40)],
        authority_binding=authority_binding,
    )

    assert receipt["status"] == "rejected"
    assert receipt["authority_binding"] is None
    assert receipt["automatic_batch_sizing"]["planning_usable"] is False
    assert receipt["automatic_batch_sizing"]["production_usable"] is False
    assert receipt["errors"]


def test_extractor_rejects_mismatched_authority_without_host_trust() -> None:
    mutated = _build([_tensor("state", "s", 40)])
    mutated["authority_binding"]["profile"] = "patch_p3"
    with pytest.raises(MemoryInventoryContractError, match="opaque native CUDA"):
        batch_sizing_bytes_from_memory_inventory_v1(
            mutated,
            expected_authority_binding=_authority_binding(),
        )

    accepted = _build([_tensor("state", "s", 40)])
    with pytest.raises(MemoryInventoryContractError, match="opaque native CUDA"):
        batch_sizing_bytes_from_memory_inventory_v1(
            accepted,
            expected_authority_binding=_authority_binding(
                state_layout_sha256="c" * 64
            ),
        )
