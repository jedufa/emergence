from __future__ import annotations

import pytest

from engine.perf.cuda_full_evolution_v2 import (
    CUDA_FULL_EVOLUTION_V2_VARIANT_PRESENTATIONS_V1,
    CUDA_NATIVE_PROFILE_ID_V2,
    CUDA_NATIVE_ROUTE_V2,
    CudaFullEvolutionContractError,
    CudaFullEvolutionNativeContractV2,
    cuda_full_evolution_v2_variant_presentation_v1,
)
from engine.perf.cuda_full_evolution_v2.workhorse import (
    SUPPORTED_WORKHORSE_SHAPES_V2,
)
from engine.campaigns.cuda_full_evolution_v2_shared_route import (
    SHARED_W32_GPU_POLICY_ID_V1,
)


def test_public_variant_presentations_include_three_science_routes_and_shared_w32() -> None:
    assert tuple(CUDA_FULL_EVOLUTION_V2_VARIANT_PRESENTATIONS_V1) == (
        "w32x32p256c512_gpuv2",
        "w32x32p256c512_gpuv2_shared",
        "w64x64p1024c2048_gpuv2",
        "w128x80p2560c5120_gpuv2",
    )
    assert all(
        value.authority == "presentation_only_not_an_execution_identity"
        for value in CUDA_FULL_EVOLUTION_V2_VARIANT_PRESENTATIONS_V1.values()
    )
    assert all(
        value.default_logical_campaign_world_count == 1024
        and value.logical_campaign_packing_policy
        == "freeze_logical_world_identities_not_physical_cuda_batch"
        and value.qualification_status == "NOT_QUALIFIED"
        and value.science_authority_granted is False
        for value in CUDA_FULL_EVOLUTION_V2_VARIANT_PRESENTATIONS_V1.values()
    )
    assert all(
        value.physical_batch_policy == "device_calibrated_B_and_temporal_k"
        and value.light_mode_policy == "resource_manager_budgeted_subbatching"
        for name, value in CUDA_FULL_EVOLUTION_V2_VARIANT_PRESENTATIONS_V1.items()
        if name != SHARED_W32_GPU_POLICY_ID_V1
    )


def test_public_variant_presentations_describe_existing_routes_without_replacing_ids() -> None:
    micro = cuda_full_evolution_v2_variant_presentation_v1(
        "w32x32p256c512_gpuv2"
    )
    assert micro is not None
    assert micro.canonical_route_id == CUDA_NATIVE_ROUTE_V2
    assert micro.canonical_profile_id == CUDA_NATIVE_PROFILE_ID_V2
    assert micro.canonical_shape_id is None
    assert micro.intended_usages == (
        "quick_engine_tests",
        "default_1024_world_campaign",
        "shared_gpu_light_development",
    )
    shared = cuda_full_evolution_v2_variant_presentation_v1(
        "w32x32p256c512_gpuv2_shared"
    )
    assert shared is not None
    assert shared.canonical_route_id == CUDA_NATIVE_ROUTE_V2
    assert shared.canonical_profile_id == CUDA_NATIVE_PROFILE_ID_V2
    assert shared.gpu_resource_policy_id == SHARED_W32_GPU_POLICY_ID_V1
    assert shared.physical_batch_policy == "closed_uniform_B32_B136_and_K1_K4"
    assert shared.qualification_status == "NOT_QUALIFIED"
    assert shared.science_authority_granted is False

    for alias, shape in zip(
        ("w64x64p1024c2048_gpuv2", "w128x80p2560c5120_gpuv2"),
        SUPPORTED_WORKHORSE_SHAPES_V2[:2],
        strict=True,
    ):
        presentation = cuda_full_evolution_v2_variant_presentation_v1(alias)
        assert presentation is not None
        assert presentation.canonical_shape_id == shape.shape_id
        assert presentation.canonical_route_id == shape.route_id
        assert presentation.canonical_profile_id == shape.profile_id
        assert alias not in (shape.shape_id, shape.route_id, shape.profile_id)
    assert (
        "four_x_finite_size_experiment"
        in CUDA_FULL_EVOLUTION_V2_VARIANT_PRESENTATIONS_V1[
            "w64x64p1024c2048_gpuv2"
        ].intended_usages
    )
    assert (
        "ten_x_finite_size_experiment"
        in CUDA_FULL_EVOLUTION_V2_VARIANT_PRESENTATIONS_V1[
            "w128x80p2560c5120_gpuv2"
        ].intended_usages
    )


def test_public_variant_aliases_are_not_contract_input() -> None:
    assert cuda_full_evolution_v2_variant_presentation_v1("unknown_gpuv2") is None
    with pytest.raises(CudaFullEvolutionContractError, match="route_id"):
        CudaFullEvolutionNativeContractV2(
            B=1,
            total_ticks=1,
            route_id="w32x32p256c512_gpuv2",
        )
    with pytest.raises(TypeError):
        CUDA_FULL_EVOLUTION_V2_VARIANT_PRESENTATIONS_V1["other_gpuv2"] = None
