"""Presentation-only names for the established CUDA Full Evolution V2 routes.

These aliases are deliberately outside every execution, build, semantic,
evidence, and checkpoint contract.  They help humans distinguish the existing
W32 micro route from the W64 and W128 workhorse routes, but are never accepted
as contract input or emitted in an authoritative identity payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Mapping

from .contract import CUDA_NATIVE_PROFILE_ID_V2, CUDA_NATIVE_ROUTE_V2
from .workhorse.contract import SUPPORTED_WORKHORSE_SHAPES_BY_ID_V2


@dataclass(frozen=True, slots=True)
class CudaFullEvolutionV2VariantPresentationV1:
    """A non-authoritative route/capability description for human-facing use."""

    public_variant_name: str
    canonical_route_id: str
    canonical_profile_id: str
    canonical_shape_id: str | None
    route_description: str
    capability_description: str
    intended_usages: tuple[str, ...]
    gpu_resource_policy_id: str | None = None
    default_logical_campaign_world_count: int = 1024
    physical_batch_policy: str = "device_calibrated_B_and_temporal_k"
    logical_campaign_packing_policy: str = (
        "freeze_logical_world_identities_not_physical_cuda_batch"
    )
    light_mode_policy: str = "resource_manager_budgeted_subbatching"
    qualification_status: str = "NOT_QUALIFIED"
    science_authority_granted: bool = False
    authority: str = "presentation_only_not_an_execution_identity"


def _workhorse_presentation(
    public_variant_name: str,
    shape_id: str,
    route_description: str,
    intended_usages: tuple[str, ...],
) -> CudaFullEvolutionV2VariantPresentationV1:
    shape = SUPPORTED_WORKHORSE_SHAPES_BY_ID_V2[shape_id]
    return CudaFullEvolutionV2VariantPresentationV1(
        public_variant_name=public_variant_name,
        canonical_route_id=shape.route_id,
        canonical_profile_id=shape.profile_id,
        canonical_shape_id=shape.shape_id,
        route_description=route_description,
        capability_description=(
            "Full Evolution V2 capability names remain those declared by the "
            "canonical workhorse contract."
        ),
        intended_usages=intended_usages,
    )


CUDA_FULL_EVOLUTION_V2_VARIANT_PRESENTATIONS_V1: Final[
    Mapping[str, CudaFullEvolutionV2VariantPresentationV1]
] = MappingProxyType(
    {
        "w32x32p256c512_gpuv2": CudaFullEvolutionV2VariantPresentationV1(
            public_variant_name="w32x32p256c512_gpuv2",
            canonical_route_id=CUDA_NATIVE_ROUTE_V2,
            canonical_profile_id=CUDA_NATIVE_PROFILE_ID_V2,
            canonical_shape_id=None,
            route_description="Existing W32 micro route (32x32, P256, C512).",
            capability_description=(
                "Full Evolution V2 capability names remain those declared by the "
                "canonical CUDA Native V2 contract."
            ),
            intended_usages=(
                "quick_engine_tests",
                "default_1024_world_campaign",
                "shared_gpu_light_development",
            ),
        ),
        "w32x32p256c512_gpuv2_shared": CudaFullEvolutionV2VariantPresentationV1(
            public_variant_name="w32x32p256c512_gpuv2_shared",
            canonical_route_id=CUDA_NATIVE_ROUTE_V2,
            canonical_profile_id=CUDA_NATIVE_PROFILE_ID_V2,
            canonical_shape_id=None,
            route_description=(
                "Shared-development W32 physical packing over the unchanged "
                "canonical CUDA Native V2 transition."
            ),
            capability_description=(
                "Full Evolution V2 capabilities and logical world identities are "
                "unchanged; only plan-bound GPU admission and physical packing differ."
            ),
            intended_usages=(
                "cooperative_functional_development",
                "tiny_cuda_tests",
                "multiple_codex_gpu_sharing",
            ),
            gpu_resource_policy_id="w32x32p256c512_gpuv2_shared",
            physical_batch_policy="closed_uniform_B32_B136_and_K1_K4",
            light_mode_policy=(
                "resource_manager_shared_functional_1_25_gib_three_lanes"
            ),
        ),
        "w64x64p1024c2048_gpuv2": _workhorse_presentation(
            "w64x64p1024c2048_gpuv2",
            "w64h64p1024c2048",
            "Existing W64 workhorse route (64x64, P1024, C2048).",
            (
                "four_x_finite_size_experiment",
                "science_workhorse",
                "bounded_large_world_development",
            ),
        ),
        "w128x80p2560c5120_gpuv2": _workhorse_presentation(
            "w128x80p2560c5120_gpuv2",
            "w128h80p2560c5120",
            "Existing W128 workhorse route (128x80, P2560, C5120).",
            (
                "ten_x_finite_size_experiment",
                "science_workhorse",
                "scale_demonstration",
            ),
        ),
    }
)


def cuda_full_evolution_v2_variant_presentation_v1(
    public_variant_name: str,
) -> CudaFullEvolutionV2VariantPresentationV1 | None:
    """Look up an exact human-facing alias without resolving contract input."""
    return CUDA_FULL_EVOLUTION_V2_VARIANT_PRESENTATIONS_V1.get(public_variant_name)
