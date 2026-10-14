"""Closed shared-development resource policy for CUDA Full Evolution V2 W32.

The policy changes only physical packing and GPU admission.  It never replaces
the canonical transition/profile identity and never grants science,
performance, production, QualPrep, or Qualification authority.
"""

from __future__ import annotations

from typing import Any, Final, Mapping, Sequence


SHARED_W32_GPU_POLICY_SCHEMA_V1: Final[str] = (
    "eiel_cuda_full_evolution_v2_shared_gpu_policy_v1"
)
SHARED_W32_GPU_POLICY_ID_V1: Final[str] = "w32x32p256c512_gpuv2_shared"
SHARED_W32_CANONICAL_PROFILE_ID_V1: Final[str] = (
    "eiel_full_evolution_cuda_native_v2"
)
SHARED_W32_CANONICAL_ROUTE_ID_V1: Final[str] = (
    "tensor_world_cuda_full_evolution_v2"
)
SHARED_W32_PEAK_VRAM_GIB_V1: Final[float] = 1.25
SHARED_W32_MINIMUM_PHYSICAL_B_V1: Final[int] = 32
SHARED_W32_MAXIMUM_PHYSICAL_B_V1: Final[int] = 136
SHARED_W32_ALLOWED_TEMPORAL_K_V1: Final[tuple[int, ...]] = (1, 2, 3, 4)
SHARED_W32_MAX_ACS_SLOTS_V1: Final[int] = 1
SHARED_W32_LANE_COUNT_V1: Final[int] = 3
SHARED_W32_DEFAULT_LOGICAL_WORLDS_V1: Final[int] = 1024
SHARED_W32_LANE_TOPOLOGY_V1: Final[str] = (
    "cuda-full-evolution-v2-w32-shared-v1"
)


def shared_w32_gpu_policy_v1() -> dict[str, Any]:
    """Return the Campaign-owned shared-development policy."""

    return {
        "schema_version": SHARED_W32_GPU_POLICY_SCHEMA_V1,
        "policy_id": SHARED_W32_GPU_POLICY_ID_V1,
        "canonical_profile_id": SHARED_W32_CANONICAL_PROFILE_ID_V1,
        "canonical_route_id": SHARED_W32_CANONICAL_ROUTE_ID_V1,
        "purpose": "cooperative_functional_development",
        "gpu_admission_mode": "shared_functional",
        "exclusive_gpu": False,
        "peak_vram_gib": SHARED_W32_PEAK_VRAM_GIB_V1,
        "shared_lane_count": SHARED_W32_LANE_COUNT_V1,
        "physical_batch_cells": {
            "minimum": SHARED_W32_MINIMUM_PHYSICAL_B_V1,
            "maximum": SHARED_W32_MAXIMUM_PHYSICAL_B_V1,
            "uniform_across_campaign": True,
            "padding_allowed": False,
        },
        "temporal_k": {
            "allowed": list(SHARED_W32_ALLOWED_TEMPORAL_K_V1),
            "uniform_across_campaign": True,
        },
        "acs_checkpoint_slots": {
            "default": 0,
            "maximum": SHARED_W32_MAX_ACS_SLOTS_V1,
        },
        "logical_campaign": {
            "default_world_count": SHARED_W32_DEFAULT_LOGICAL_WORLDS_V1,
            "world_identity_independent_of_physical_partition": True,
        },
        "evidence": {
            "classification": "TECHNICAL_ONLY",
            "throughput_claims_allowed": False,
            "science_claims_allowed": False,
            "qualification_claims_allowed": False,
        },
    }


class CudaFullEvolutionV2SharedRouteError(ValueError):
    """The closed shared W32 resource policy is absent or inconsistent."""


def resolve_shared_w32_gpu_policy_v1(value: object) -> dict[str, Any]:
    """Resolve the one accepted campaign selector into its hashed plan policy."""

    if value != SHARED_W32_GPU_POLICY_ID_V1:
        raise CudaFullEvolutionV2SharedRouteError(
            "execution.gpu_resource_policy must select the exact shared W32 policy"
        )
    return shared_w32_gpu_policy_v1()


def shared_w32_gpu_policy_from_plan_v1(
    plan: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a validated policy, or ``None`` for the unchanged exclusive route."""

    execution = plan.get("execution")
    if not isinstance(execution, Mapping):
        raise CudaFullEvolutionV2SharedRouteError("campaign plan execution is absent")
    raw = execution.get("gpu_resource_policy")
    if raw is None:
        return None
    expected = shared_w32_gpu_policy_v1()
    if not isinstance(raw, Mapping) or dict(raw) != expected:
        raise CudaFullEvolutionV2SharedRouteError(
            "campaign plan shared W32 policy differs from the closed contract"
        )
    return expected


def validate_shared_w32_plan_v1(
    plan: Mapping[str, Any],
    *,
    temporal_k: int | None = None,
    vram_bytes_upper_bound: int | None = None,
) -> dict[str, Any] | None:
    """Cross-check physical packing, ACS, K, and peak VRAM against the policy."""

    policy = shared_w32_gpu_policy_from_plan_v1(plan)
    if policy is None:
        return None
    if plan.get("resolved_profile") != SHARED_W32_CANONICAL_PROFILE_ID_V1:
        raise CudaFullEvolutionV2SharedRouteError(
            "shared W32 policy is valid only for the canonical W32 profile"
        )
    batches = plan.get("physical_batches")
    if not isinstance(batches, Sequence) or isinstance(batches, (str, bytes)):
        raise CudaFullEvolutionV2SharedRouteError(
            "shared W32 plan lacks physical batches"
        )
    batch_sizes: list[int] = []
    for row in batches:
        if not isinstance(row, Mapping):
            raise CudaFullEvolutionV2SharedRouteError(
                "shared W32 physical batch is malformed"
            )
        cell_ids = row.get("cell_ids")
        if not isinstance(cell_ids, Sequence) or isinstance(cell_ids, (str, bytes)):
            raise CudaFullEvolutionV2SharedRouteError(
                "shared W32 physical batch lacks cell identities"
            )
        batch_sizes.append(len(cell_ids))
    if (
        not batch_sizes
        or len(set(batch_sizes)) != 1
        or any(
            size < SHARED_W32_MINIMUM_PHYSICAL_B_V1
            or size > SHARED_W32_MAXIMUM_PHYSICAL_B_V1
            for size in batch_sizes
        )
    ):
        raise CudaFullEvolutionV2SharedRouteError(
            "shared W32 batches must be uniform and each remain within B32-B136"
        )
    execution = plan["execution"]
    acs_slots = execution.get("acs_checkpoint_slots", 0)
    if type(acs_slots) is not int or not 0 <= acs_slots <= (
        SHARED_W32_MAX_ACS_SLOTS_V1
    ):
        raise CudaFullEvolutionV2SharedRouteError(
            "shared W32 permits ACS0 by default and at most ACS1"
        )
    if temporal_k is not None and temporal_k not in SHARED_W32_ALLOWED_TEMPORAL_K_V1:
        raise CudaFullEvolutionV2SharedRouteError(
            "shared W32 temporal K must be one of K1, K2, K3, or K4"
        )
    if vram_bytes_upper_bound is not None:
        if type(vram_bytes_upper_bound) is not int or vram_bytes_upper_bound < 0:
            raise CudaFullEvolutionV2SharedRouteError(
                "shared W32 provider VRAM bound is invalid"
            )
        declared_bytes = int(SHARED_W32_PEAK_VRAM_GIB_V1 * 1024**3)
        if vram_bytes_upper_bound > declared_bytes:
            raise CudaFullEvolutionV2SharedRouteError(
                "shared W32 provider VRAM bound exceeds its declared 1.25 GiB peak"
            )
    if "TECHNICAL_ONLY" not in str(plan.get("no_claims") or "").upper():
        raise CudaFullEvolutionV2SharedRouteError(
            "shared W32 campaign must explicitly declare TECHNICAL_ONLY"
        )
    return policy


__all__ = [
    "CudaFullEvolutionV2SharedRouteError",
    "SHARED_W32_ALLOWED_TEMPORAL_K_V1",
    "SHARED_W32_DEFAULT_LOGICAL_WORLDS_V1",
    "SHARED_W32_GPU_POLICY_ID_V1",
    "SHARED_W32_GPU_POLICY_SCHEMA_V1",
    "SHARED_W32_LANE_COUNT_V1",
    "SHARED_W32_LANE_TOPOLOGY_V1",
    "SHARED_W32_MAXIMUM_PHYSICAL_B_V1",
    "SHARED_W32_MAX_ACS_SLOTS_V1",
    "SHARED_W32_MINIMUM_PHYSICAL_B_V1",
    "SHARED_W32_PEAK_VRAM_GIB_V1",
    "resolve_shared_w32_gpu_policy_v1",
    "shared_w32_gpu_policy_from_plan_v1",
    "shared_w32_gpu_policy_v1",
    "validate_shared_w32_plan_v1",
]
