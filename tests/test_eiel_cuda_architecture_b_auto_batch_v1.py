from __future__ import annotations

import json
from typing import Any

import pytest

from engine.perf.cuda_architecture_b_auto_batch_v1 import (
    RECEIPT_SCHEMA_VERSION,
    select_auto_batch_from_memory_inventory_v1,
    select_auto_batch_v1,
)
from engine.perf.cuda_architecture_b_memory_inventory_v1 import (
    build_cuda_architecture_b_memory_inventory_v1,
)


_EXTENSION_SHA256 = "a" * 64
_STATE_LAYOUT_SHA256 = "b" * 64


def _request(**overrides: Any) -> dict[str, Any]:
    request = {
        "world": "patch",
        "profile": "patch_p2",
        "build_identity": "architecture_b_v8",
        "extension_sha256": _EXTENSION_SHA256,
        "state_layout_sha256": _STATE_LAYOUT_SHA256,
        "active_policy_family": "fixed_mlp",
        "generations": 64,
        "ticks": 80,
        "summary_mode": "compact",
        "tick_output_storage": "rolling_single_row",
    }
    request.update(overrides)
    return request


def _resources(**overrides: Any) -> dict[str, Any]:
    resources = {
        "total_memory_bytes": 10_000,
        "free_memory_bytes": 10_000,
        "reserve_floor_bytes": 500,
        "reserve_fraction": 0.05,
        "shared_bytes": 100,
        "per_cell_bytes": 50,
        "sm_count": 24,
        "mandatory_kernel_blocks_per_sm": {
            "controller": 4,
            "summary": 8,
        },
    }
    resources.update(overrides)
    return resources


def _envelope(
    envelope_id: str,
    batch_size: int,
    *,
    qualified: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    envelope = {
        "envelope_id": envelope_id,
        "qualified": qualified,
        **_request(),
        "batch_size": batch_size,
    }
    envelope.update(overrides)
    return envelope


def _candidate(receipt: dict[str, Any], envelope_id: str) -> dict[str, Any]:
    return next(
        row for row in receipt["candidates"] if row["envelope_id"] == envelope_id
    )


def _split_request(**overrides: Any) -> dict[str, Any]:
    request = {
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
    request.update(overrides)
    return request


def _split_envelope(
    batch_cells: int,
    *,
    qualified: bool = True,
    **overrides: Any,
) -> dict[str, Any]:
    envelope = {
        **_split_request(),
        "batch_cells": batch_cells,
        "qualified": qualified,
    }
    envelope.update(overrides)
    return envelope


def test_selects_only_an_explicit_nonmonotonic_qualified_shape() -> None:
    receipt = select_auto_batch_v1(
        request=_request(),
        resources=_resources(),
        qualified_envelopes=[
            _envelope("g128-qualified", 128),
            _envelope("g32-not-qualified", 32, qualified=False),
            _envelope("g16-qualified", 16),
            _envelope("other-world-g96", 96, world="flat"),
            _envelope("g64-qualified", 64),
        ],
    )

    assert receipt["schema_version"] == RECEIPT_SCHEMA_VERSION
    assert receipt["decision"] == "selected"
    assert receipt["computations"] == {
        "reserve_fraction_bytes": 500,
        "reserve_bytes": 500,
        "available_batch_bytes": 9_400,
        "memory_max": 188,
        "minimum_mandatory_blocks_per_sm": 4,
        "occupancy_wave": 96,
        "mechanical_max": 96,
    }
    assert receipt["selected_batch_size"] == 64
    assert receipt["selected_envelope_id"] == "g64-qualified"
    assert _candidate(receipt, "g32-not-qualified")["rejection_reasons"] == [
        "not_explicitly_qualified"
    ]
    assert _candidate(receipt, "other-world-g96")["rejection_reasons"] == [
        "world_mismatch"
    ]
    assert _candidate(receipt, "g128-qualified")["rejection_reasons"] == [
        "above_occupancy_wave"
    ]
    assert receipt["selection_policy"]["mechanical_max_implies_qualification"] is False


@pytest.mark.parametrize(
    ("free_memory_bytes", "expected_memory_max", "expected_batch_size"),
    [(700, 4, 4), (699, 3, 2)],
)
def test_memory_thresholds_never_round_up_to_an_unqualified_shape(
    free_memory_bytes: int,
    expected_memory_max: int,
    expected_batch_size: int,
) -> None:
    resources = _resources(
        total_memory_bytes=2_000,
        free_memory_bytes=free_memory_bytes,
        reserve_floor_bytes=100,
        reserve_fraction=0.10,
        shared_bytes=100,
        per_cell_bytes=100,
        sm_count=8,
        mandatory_kernel_blocks_per_sm={"controller": 4, "summary": 5},
    )

    receipt = select_auto_batch_v1(
        request=_request(),
        resources=resources,
        qualified_envelopes=[
            _envelope("b1", 1),
            _envelope("b2", 2),
            _envelope("b4", 4),
        ],
    )

    assert receipt["computations"]["reserve_bytes"] == 200
    assert receipt["computations"]["memory_max"] == expected_memory_max
    assert receipt["selected_batch_size"] == expected_batch_size


def test_all_workload_dimensions_must_match_exactly() -> None:
    receipt = select_auto_batch_v1(
        request=_request(),
        resources=_resources(),
        qualified_envelopes=[
            _envelope("exact", 4),
            _envelope("wrong-world", 80, world="flat"),
            _envelope("wrong-profile", 80, profile="patch_p3"),
            _envelope("wrong-build", 80, build_identity="architecture_b_v9"),
            _envelope("wrong-extension", 80, extension_sha256="c" * 64),
            _envelope("wrong-layout", 80, state_layout_sha256="d" * 64),
            _envelope("wrong-policy", 80, active_policy_family="gru"),
            _envelope("wrong-generations", 80, generations=128),
            _envelope("wrong-ticks", 80, ticks=160),
            _envelope("wrong-summary", 80, summary_mode="full"),
            _envelope(
                "wrong-tick-storage",
                80,
                tick_output_storage="full_tick_history",
            ),
        ],
    )

    assert receipt["selected_envelope_id"] == "exact"
    expected_reasons = {
        "wrong-world": "world_mismatch",
        "wrong-profile": "profile_mismatch",
        "wrong-build": "build_identity_mismatch",
        "wrong-extension": "extension_sha256_mismatch",
        "wrong-layout": "state_layout_sha256_mismatch",
        "wrong-policy": "active_policy_family_mismatch",
        "wrong-generations": "generations_mismatch",
        "wrong-ticks": "ticks_mismatch",
        "wrong-summary": "summary_mismatch",
        "wrong-tick-storage": "tick_output_storage_mismatch",
    }
    for envelope_id, reason in expected_reasons.items():
        assert _candidate(receipt, envelope_id)["rejection_reasons"] == [reason]


def test_rolling_b48_cannot_satisfy_full_tick_history_request() -> None:
    receipt = select_auto_batch_v1(
        request=_request(tick_output_storage="full_tick_history"),
        resources=_resources(),
        qualified_envelopes=[
            _envelope("b8-full", 8, tick_output_storage="full_tick_history"),
            _envelope("b48-rolling", 48),
        ],
    )

    assert receipt["decision"] == "selected"
    assert receipt["selected_batch_size"] == 8
    assert _candidate(receipt, "b48-rolling")["rejection_reasons"] == [
        "tick_output_storage_mismatch"
    ]


def _memory_inventory(
    *,
    complete: bool = True,
    **binding_overrides: Any,
) -> dict[str, Any]:
    authority_binding = _split_request(generation_count=2)
    authority_binding.update(binding_overrides)
    return build_cuda_architecture_b_memory_inventory_v1(
        [
            {
                "name": "resident_state",
                "storage_id": "device:resident-state",
                "storage_nbytes": 4_800,
                "allocation_scope": "per_cell",
                "memory_tier": "device",
                "policy_family": "non_policy",
                "role": "resident_active",
                "canonical_checkpoint": True,
            },
            {
                "name": "shared_tape",
                "storage_id": "device:shared-tape",
                "storage_nbytes": 100,
                "allocation_scope": "shared",
                "memory_tier": "device",
                "policy_family": "fixed_mlp",
                "role": "immutable_tape",
                "canonical_checkpoint": False,
            },
        ],
        cell_count=48,
        active_policy_family="fixed_mlp",
        inventory_complete=complete,
        authority_binding=authority_binding,
    )


def _inventory_resources(**overrides: Any) -> dict[str, Any]:
    resources = _resources()
    del resources["per_cell_bytes"]
    del resources["shared_bytes"]
    resources.update(overrides)
    return resources


def test_host_inventory_is_planning_only_and_cannot_authorize_b48() -> None:
    receipt = select_auto_batch_from_memory_inventory_v1(
        request=_request(generations=2),
        resources=_inventory_resources(
            mandatory_kernel_blocks_per_sm={"phase_1": 2}
        ),
        memory_inventory=_memory_inventory(),
        qualified_envelopes=[
            _envelope("b2", 2, generations=2),
            _envelope("b8", 8, generations=2),
            _envelope("b48", 48, generations=2),
        ],
    )

    assert receipt["decision"] == "rejected"
    assert receipt["selected_batch_size"] is None
    assert receipt["rejection_reasons"][0]["code"] == (
        "memory_inventory_not_usable"
    )
    assert "opaque native CUDA runtime authority" in (
        receipt["rejection_reasons"][0]["details"][0]
    )


def test_rejected_or_conflicting_memory_inventory_fails_closed() -> None:
    rejected = select_auto_batch_from_memory_inventory_v1(
        request=_request(generations=2),
        resources=_inventory_resources(),
        memory_inventory=_memory_inventory(complete=False),
        qualified_envelopes=[_envelope("b48", 48, generations=2)],
    )
    conflicting = select_auto_batch_from_memory_inventory_v1(
        request=_request(generations=2),
        resources=_inventory_resources(per_cell_bytes=99),
        memory_inventory=_memory_inventory(),
        qualified_envelopes=[_envelope("b48", 48, generations=2)],
    )

    assert rejected["decision"] == "rejected"
    assert rejected["rejection_reasons"][0]["code"] == (
        "memory_inventory_not_usable"
    )
    assert conflicting["decision"] == "rejected"
    assert conflicting["rejection_reasons"][0]["code"] == (
        "memory_inventory_not_usable"
    )


@pytest.mark.parametrize(
    ("request_overrides", "inventory_overrides", "mismatch_field"),
    [
        ({"world": "flat"}, {}, "world"),
        ({"profile": "patch_p3"}, {}, "profile"),
        ({"build_identity": "architecture_b_v9"}, {}, "build_identity"),
        ({"extension_sha256": "c" * 64}, {}, "extension_sha256"),
        ({"state_layout_sha256": "d" * 64}, {}, "state_layout_sha256"),
        ({"active_policy_family": "gru"}, {}, "active_policy_family"),
        ({"generations": 4}, {}, "generation_count"),
        ({"ticks": 160}, {}, "ticks_per_generation"),
        ({"summary_mode": "full"}, {}, "summary_mode"),
        (
            {"tick_output_storage": "full_tick_history"},
            {},
            "tick_output_storage",
        ),
    ],
)
def test_inventory_authority_must_match_the_exact_auto_batch_request(
    request_overrides: dict[str, Any],
    inventory_overrides: dict[str, Any],
    mismatch_field: str,
) -> None:
    request = _request(generations=2)
    request.update(request_overrides)
    envelope = _envelope("b48", 48, generations=2)
    envelope.update(request_overrides)
    receipt = select_auto_batch_from_memory_inventory_v1(
        request=request,
        resources=_inventory_resources(),
        memory_inventory=_memory_inventory(**inventory_overrides),
        qualified_envelopes=[envelope],
    )

    assert receipt["decision"] == "rejected"
    assert receipt["selected_batch_size"] is None
    assert receipt["rejection_reasons"][0]["code"] == (
        "memory_inventory_not_usable"
    )
    assert mismatch_field
    assert "opaque native CUDA runtime authority" in (
        receipt["rejection_reasons"][0]["details"][0]
    )


def test_non_mapping_inventory_returns_a_structured_rejection() -> None:
    receipt = select_auto_batch_from_memory_inventory_v1(
        request=_request(generations=2),
        resources=_inventory_resources(),
        memory_inventory=object(),
        qualified_envelopes=[_envelope("b48", 48, generations=2)],
    )

    assert receipt["decision"] == "rejected"
    assert receipt["selected_batch_size"] is None
    assert receipt["rejection_reasons"][0]["code"] == (
        "memory_inventory_not_usable"
    )
    assert receipt["memory_inventory"]["ok"] is False


@pytest.mark.parametrize(
    "resources",
    [
        {key: value for key, value in _resources().items() if key != "sm_count"},
        _resources(free_memory_bytes=10_001),
        _resources(reserve_fraction=float("nan")),
        _resources(per_cell_bytes=0),
        _resources(sm_count=True),
        _resources(mandatory_kernel_blocks_per_sm={}),
        _resources(mandatory_kernel_blocks_per_sm={"controller": 0}),
        _resources(total_memory_bytes=1 << 10000),
        _resources(free_memory_bytes=1 << 10000),
        _resources(reserve_floor_bytes=1 << 10000),
        _resources(reserve_fraction=1 << 10000),
        _resources(per_cell_bytes=1 << 10000),
        _resources(sm_count=1 << 10000),
        _resources(
            mandatory_kernel_blocks_per_sm={"controller": 1 << 10000}
        ),
    ],
)
def test_malformed_resources_fail_closed_with_a_rejection_receipt(
    resources: dict[str, Any],
) -> None:
    receipt = select_auto_batch_v1(
        request=_request(),
        resources=resources,
        qualified_envelopes=[_envelope("b4", 4)],
    )

    assert receipt["decision"] == "rejected"
    assert receipt["selected_batch_size"] is None
    assert receipt["selected_envelope_id"] is None
    assert receipt["candidates"] == []
    assert receipt["rejection_reasons"][0]["code"] == "malformed_resources"
    assert receipt["rejection_reasons"][0]["details"]


@pytest.mark.parametrize(
    ("request_input", "resources", "expected_code"),
    [
        (object(), _inventory_resources(), "malformed_request"),
        (
            _request(generations=2),
            object(),
            "memory_inventory_not_usable",
        ),
    ],
)
def test_inventory_selector_outer_malformed_inputs_never_escape(
    request_input: object,
    resources: object,
    expected_code: str,
) -> None:
    receipt = select_auto_batch_from_memory_inventory_v1(
        request=request_input,
        resources=resources,
        memory_inventory=_memory_inventory(),
        qualified_envelopes=[_envelope("b48", 48, generations=2)],
    )

    assert receipt["decision"] == "rejected"
    assert receipt["selected_batch_size"] is None
    assert receipt["rejection_reasons"][0]["code"] == expected_code


def test_one_malformed_envelope_rejects_the_entire_decision() -> None:
    malformed = _envelope("malformed", 8)
    del malformed["summary_mode"]

    receipt = select_auto_batch_v1(
        request=_request(),
        resources=_resources(),
        qualified_envelopes=[_envelope("otherwise-valid", 4), malformed],
    )

    assert receipt["decision"] == "rejected"
    assert receipt["selected_batch_size"] is None
    assert receipt["rejection_reasons"][0]["code"] == (
        "malformed_qualified_envelopes"
    )


def test_empty_and_none_envelopes_fail_closed() -> None:
    empty_receipt = select_auto_batch_v1(
        request=_request(), resources=_resources(), qualified_envelopes=[]
    )
    none_receipt = select_auto_batch_v1(
        request=_request(),
        resources=_resources(),
        qualified_envelopes=None,  # type: ignore[arg-type]
    )

    assert empty_receipt["decision"] == "rejected"
    assert empty_receipt["rejection_reasons"][0]["code"] == (
        "no_qualified_matching_envelope"
    )
    assert none_receipt["decision"] == "rejected"
    assert none_receipt["rejection_reasons"][0]["code"] == (
        "malformed_qualified_envelopes"
    )


def test_selection_and_receipt_are_stable_across_input_order() -> None:
    envelopes = [
        _envelope("z-envelope", 64),
        _envelope("small", 16),
        _envelope("a-envelope", 64),
    ]

    forward = select_auto_batch_v1(
        request=_request(),
        resources=_resources(),
        qualified_envelopes=envelopes,
    )
    reverse = select_auto_batch_v1(
        request=_request(),
        resources=_resources(),
        qualified_envelopes=list(reversed(envelopes)),
    )

    assert forward == reverse
    assert forward["selected_envelope_id"] == "a-envelope"
    assert _candidate(forward, "z-envelope")["rejection_reasons"] == [
        "deterministic_envelope_id_tiebreak"
    ]
    json.dumps(forward, sort_keys=True, allow_nan=False)


def test_matching_shapes_above_limits_return_complete_rejections() -> None:
    receipt = select_auto_batch_v1(
        request=_request(),
        resources=_resources(
            total_memory_bytes=1_000,
            free_memory_bytes=300,
            reserve_floor_bytes=100,
            reserve_fraction=0.10,
            shared_bytes=100,
            per_cell_bytes=100,
            sm_count=1,
            mandatory_kernel_blocks_per_sm={"controller": 1},
        ),
        qualified_envelopes=[_envelope("b2", 2), _envelope("b4", 4)],
    )

    assert receipt["decision"] == "rejected"
    assert receipt["computations"]["memory_max"] == 1
    assert receipt["computations"]["occupancy_wave"] == 1
    assert receipt["rejection_reasons"][0]["code"] == (
        "no_qualified_shape_within_limits"
    )
    assert _candidate(receipt, "b2")["rejection_reasons"] == [
        "above_memory_max",
        "above_occupancy_wave",
    ]


def test_preferred_split_resource_contract_is_stable_and_explicit() -> None:
    envelopes = [
        _split_envelope(128),
        _split_envelope(32, qualified=False),
        _split_envelope(16),
        _split_envelope(96, world="flat"),
        _split_envelope(64),
    ]
    arguments = {
        "request": _split_request(),
        "device_resources": {
            "total_bytes": 10_000,
            "free_bytes": 10_000,
            "sm_count": 24,
        },
        "memory_policy": {
            "reserve_floor_bytes": 500,
            "reserve_fraction": 0.05,
            "shared_bytes": 100,
            "per_cell_bytes": 50,
        },
        "kernel_resources": {
            "controller": {"max_blocks_per_multiprocessor": 4},
            "summary": {"max_blocks_per_multiprocessor": 8},
        },
    }

    forward = select_auto_batch_v1(
        **arguments,
        qualified_envelopes=envelopes,
        mandatory_kernels=["controller", "summary"],
    )
    reverse = select_auto_batch_v1(
        **arguments,
        qualified_envelopes=list(reversed(envelopes)),
        mandatory_kernels=["summary", "controller"],
    )

    assert forward == reverse
    assert forward["status"] == "selected"
    assert forward["selected_batch_cells"] == 64
    assert forward["qualified_matching_shapes"] == [16, 64, 128]
    assert forward["feasible_qualified_shapes"] == [16, 64]
    assert forward["calculations"]["memory_max_cells"] == 188
    assert forward["calculations"]["occupancy_wave_cells"] == 96
    assert forward["calculations"]["mechanical_max_cells"] == 96


def test_preferred_split_resource_contract_rejects_missing_kernel_receipt() -> None:
    receipt = select_auto_batch_v1(
        request=_split_request(),
        qualified_envelopes=[_split_envelope(4)],
        device_resources={
            "total_bytes": 10_000,
            "free_bytes": 10_000,
            "sm_count": 24,
        },
        memory_policy={
            "reserve_floor_bytes": 500,
            "reserve_fraction": 0.05,
            "shared_bytes": 100,
            "per_cell_bytes": 50,
        },
        kernel_resources={
            "controller": {"max_blocks_per_multiprocessor": 4}
        },
        mandatory_kernels=["controller", "summary"],
    )

    assert receipt["status"] == "rejected"
    assert receipt["decision"] == "rejected_fail_closed_malformed_input"
    assert receipt["selected_batch_cells"] is None
    assert receipt["rejection_reasons"] == ["malformed_or_ambiguous_input"]
    assert any("kernel_resources.summary" in error for error in receipt["input_errors"])
