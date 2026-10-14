"""Strict, representation-opaque Tensor World C6 contract tests."""

from __future__ import annotations

import pytest

from engine.tensor_world.c6_contract import (
    TensorWorldC6CapabilityStatusV1,
    TensorWorldC6FailureCodeV1,
    TensorWorldC6RouteRequestV1,
    TensorWorldC6ResourceEstimateV1,
    admit_tensor_world_resources_v1,
    default_tensor_world_c6_provider_manifest_v1,
    resolve_tensor_world_c6_route_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def _request(*, strict: bool) -> TensorWorldC6RouteRequestV1:
    return TensorWorldC6RouteRequestV1(
        batch=48,
        grid_size=32,
        initial_population=256,
        slot_capacity=512,
        policy_family="fixed_mlp",
        mechanics=("patch", "jungle", "wdl", "eed", "observation", "fixed_mlp"),
        strict=strict,
    )


def test_c6_manifest_has_explicit_status_for_every_route_surface() -> None:
    manifest = default_tensor_world_c6_provider_manifest_v1()
    statuses = {capability.name: capability.status for capability in manifest.capabilities}
    assert statuses["patch"] is TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED
    assert statuses["population"] is TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED
    assert statuses["visualizer"] is TensorWorldC6CapabilityStatusV1.UNSUPPORTED
    assert len(manifest.fingerprint()) == 64


def test_c6_strict_route_rejects_not_yet_qualified_without_fallback() -> None:
    with pytest.raises(TensorWorldContractError, match="NOT_YET_QUALIFIED"):
        resolve_tensor_world_c6_route_v1(_request(strict=True))


def test_c6_non_strict_resolution_retains_failure_identity() -> None:
    resolution = resolve_tensor_world_c6_route_v1(_request(strict=False))
    assert resolution.accepted is False
    assert resolution.failure_code is TensorWorldC6FailureCodeV1.NOT_YET_QUALIFIED
    assert resolution.capability_statuses["patch"] is TensorWorldC6CapabilityStatusV1.NOT_YET_QUALIFIED
    assert set(resolution.capability_statuses) == {
        "patch", "jungle", "wdl", "eed", "observation", "fixed_mlp"
    }
    assert len(resolution.route_fingerprint) == 64


def test_c6_resource_admission_requires_frozen_headroom_and_device() -> None:
    estimate = TensorWorldC6ResourceEstimateV1(
        resident_vram_bytes=100,
        temporary_vram_bytes=20,
        headroom_bytes=10,
        device_capability="sm_75",
    )
    accepted = admit_tensor_world_resources_v1(
        estimate, available_vram_bytes=130, device_capability="sm_75"
    )
    assert accepted.accepted is True
    rejected = admit_tensor_world_resources_v1(
        estimate, available_vram_bytes=129, device_capability="sm_75"
    )
    assert rejected.accepted is False
    assert rejected.failure_code is TensorWorldC6FailureCodeV1.RESOURCE_ADMISSION
    mismatch = admit_tensor_world_resources_v1(
        estimate, available_vram_bytes=1000, device_capability="sm_86"
    )
    assert mismatch.accepted is False


def test_c6_request_rejects_duplicate_domain_mechanics() -> None:
    with pytest.raises(TensorWorldContractError, match="duplicates"):
        TensorWorldC6RouteRequestV1(
            batch=1,
            grid_size=32,
            initial_population=256,
            slot_capacity=512,
            policy_family="fixed_mlp",
            mechanics=("patch", "patch"),
        )
