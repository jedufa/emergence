from __future__ import annotations

from pathlib import Path
import json

import pytest

from engine.campaigns.eiel_full_evolution_v2_packed_resources_v1 import (
    ESTIMATOR_ID,
    FE_V2_PACKED_ALLOCATOR_JITTER_PAD_BYTES,
    FE_V2_PACKED_B48_PREPARE_RESERVED_BYTES,
    FE_V2_PACKED_B144_TICK_USED_BYTES,
    FE_V2_PACKED_B144_NVIDIA_SMI_USED_BYTES,
    FE_V2_PACKED_PHYSICAL_VRAM_BYTES,
    eiel_full_evolution_v2_packed_reserved_bytes_v1,
    estimate_eiel_full_evolution_v2_packed_resident_resources_v1,
    is_eiel_full_evolution_v2_packed_identity_v1,
)
from engine.authoritative_state.runtime_abi import _small_int_vector
from engine.campaigns.production_engine_native_bindings import (
    eed_fixed_mlp_native_binding_v1,
    jungle_fixed_mlp_native_binding_v1,
    wdl_jungle_eed_fixed_mlp_native_binding_v1,
    wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1,
    wdl_jungle_fixed_mlp_native_binding_v1,
)
from engine.campaigns.resources import ResourceEstimateError
from engine.tensor_world.c5_rngv2_route import V1_ROUTE_ID, resolve_rng_route


REPO = Path(__file__).resolve().parents[1]
MEASURED_B48_RESERVED = 1_788_870_656
USABLE_BUDGET_BYTES = 5_637_144_576
STALE_B48_PROJECTION_BYTES = 8_618_080_896


def _v2_estimate(batch_cells: int):
    return estimate_eiel_full_evolution_v2_packed_resident_resources_v1(
        batch_cells=batch_cells,
        logical_generations=3,
        logical_trajectories=batch_cells + 1,
        checkpoint_boundaries=2,
        resident_segment_generations=3,
        physical_window_generations=1,
        ticks_per_generation=160,
        resident_capacity=512,
        cpython_gauss_segment_capacity=0,
        policy_birth_segment_refill_enable=False,
    )


def test_fe_binding_successor_max_is_256() -> None:
    binding = wdl_jungle_eed_fixed_mlp_native_binding_v1()
    assert binding.batch_max == 256


def test_non_fe_bindings_keep_prior_batch_max() -> None:
    assert wdl_jungle_fixed_mlp_native_binding_v1().batch_max == 48
    assert eed_fixed_mlp_native_binding_v1().batch_max == 48
    assert jungle_fixed_mlp_native_binding_v1().batch_max == 1
    assert wdl_jungle_eed_hapb_fixed_mlp_native_binding_v1().batch_max == 256


@pytest.mark.parametrize("batch_cells", (48, 49, 64, 72, 96, 128, 132, 136, 140, 144, 145, 160, 256))
def test_native_and_estimator_admit_supported_b(batch_cells: int) -> None:
    binding = wdl_jungle_eed_fixed_mlp_native_binding_v1()
    assert 1 <= batch_cells <= binding.batch_max
    estimate = _v2_estimate(batch_cells)
    assert estimate["resource_estimator_id"] == ESTIMATOR_ID
    assert type(estimate["vram_bytes_upper_bound"]) is int
    assert estimate["vram_bytes_upper_bound"] >= estimate["vram_bytes_projection"]


def test_max_plus_one_fails_closed() -> None:
    binding = wdl_jungle_eed_fixed_mlp_native_binding_v1()
    with pytest.raises(ResourceEstimateError, match=r"\[1, 256\]"):
        eiel_full_evolution_v2_packed_reserved_bytes_v1(binding.batch_max + 1)


def test_shared_owner_source_admits_b256_not_b48() -> None:
    jungle = (
        REPO / "engine/perf/cuda_architecture_b_direct_resident_wdl_jungle_v1.py"
    ).read_text(encoding="utf-8")
    wdl = (
        REPO / "engine/perf/cuda_architecture_b_direct_resident_wdl_v2.py"
    ).read_text(encoding="utf-8")
    assert "requires B1..B256 requests" in jungle
    assert "requires B1..B48 requests" not in jungle
    assert "admitted_max = 256" in wdl
    assert "admitted_max = 256 if hapb_host_policy_enable else 72" not in wdl


def test_b48_estimate_is_calibrated_not_stale_8gib() -> None:
    estimate = _v2_estimate(48)
    assert estimate["vram_bytes_projection"] == MEASURED_B48_RESERVED
    assert estimate["vram_bytes_projection"] == FE_V2_PACKED_B48_PREPARE_RESERVED_BYTES
    assert estimate["vram_bytes_upper_bound"] == (
        MEASURED_B48_RESERVED + FE_V2_PACKED_ALLOCATOR_JITTER_PAD_BYTES
    )
    assert estimate["vram_bytes_upper_bound"] < 3 * 1024 * 1024 * 1024
    assert estimate["vram_bytes_upper_bound"] < STALE_B48_PROJECTION_BYTES // 2


def test_host_projection_is_not_counted_as_vram() -> None:
    estimate = _v2_estimate(48)
    assumptions = estimate["resource_estimator_assumptions"]
    assert assumptions["memory_domain"] == "gpu_reserved_cuda_only"
    assert assumptions["host_projection_counted_as_vram"] is False
    assert assumptions["gru_controller_allowance_bytes"] == 0
    assert assumptions["per_cell_allocator_margin_bytes"] == 0
    assert assumptions["tick_transient_extrapolated"] is False


def test_estimator_is_monotonic_in_b() -> None:
    previous = 0
    for batch_cells in range(1, 257):
        reserved = eiel_full_evolution_v2_packed_reserved_bytes_v1(batch_cells)
        assert reserved > previous
        previous = reserved


def test_packed_b144_projection_is_not_under_measured_occupancy() -> None:
    fixture_path = REPO / "tests/fixtures/tw_rngv2_fe_b144_q4_acs_gpu_receipts.json"
    fixture = json.loads(fixture_path.read_text())
    gpu = fixture["q4b_capture_barrier_gpu_before"]
    nvidia_smi_used_bytes = int(fixture["nvidia_smi_tick_used_bytes"])
    q4b_max_allocated = int(gpu["max_memory_allocated_bytes"])
    q4b_acs_memory_reserved = int(gpu["memory_reserved_bytes"])
    assert "max_memory_reserved" not in gpu
    estimate = _v2_estimate(144)
    assert estimate["vram_bytes_projection"] >= nvidia_smi_used_bytes
    assert estimate["vram_bytes_projection"] >= q4b_max_allocated
    assert estimate["vram_bytes_projection"] >= q4b_acs_memory_reserved
    assert estimate["vram_bytes_upper_bound"] >= estimate["vram_bytes_projection"]
    assumptions = estimate["resource_estimator_assumptions"]
    assert assumptions["host_projection_counted_as_vram"] is False
    assert assumptions["memory_domain"] == "gpu_reserved_cuda_only"
    assert assumptions.get("cuda_headroom_fraction_as_production_reserve") is not True


def test_packed_b140_projection_is_between_b136_and_b144() -> None:
    b136 = _v2_estimate(136)
    b140 = _v2_estimate(140)
    b144 = _v2_estimate(144)
    assert b136["vram_bytes_projection"] < b140["vram_bytes_projection"] < b144["vram_bytes_projection"]
    assert b140["physical_batch_cells"] == 140


def test_measured_b144_peak_plus_jitter_fits_torch_declared_budget() -> None:
    import math

    torch_visible_total = 6_027_608_064
    headroom = math.ceil(torch_visible_total * 0.04)
    declared_budget = torch_visible_total - headroom
    estimate = _v2_estimate(144)
    assert estimate["vram_bytes_upper_bound"] <= declared_budget


def test_b145_is_not_truncated_and_is_outside_this_qualification_envelope() -> None:
    binding = wdl_jungle_eed_fixed_mlp_native_binding_v1()
    assert 145 <= binding.batch_max
    b144 = _v2_estimate(144)
    b145 = _v2_estimate(145)
    assert b145["vram_bytes_projection"] > b144["vram_bytes_projection"]
    assert b145["physical_batch_cells"] == 145


def test_b160_exceeds_torch_visible_card_after_b144_calibration() -> None:
    torch_visible_total = 6_027_608_064
    b160 = _v2_estimate(160)
    assert b160["vram_bytes_upper_bound"] > torch_visible_total


def test_abi_small_vector_admits_b144_and_fails_closed_past_256() -> None:
    import torch
    from engine.authoritative_state.errors import AuthoritativeStateContractError

    assert _small_int_vector(
        torch.arange(144, dtype=torch.int64),
        path="prepared.source.active_count_state",
    ) == list(range(144))
    with pytest.raises(AuthoritativeStateContractError, match="at most 256 values"):
        _small_int_vector(
            torch.arange(257, dtype=torch.int64),
            path="prepared.source.active_count_state",
        )


def test_v1_identity_is_rejected_by_v2_estimator() -> None:
    assert is_eiel_full_evolution_v2_packed_identity_v1(
        {
            "cpython_gauss_segment_capacity": 65_536,
            "policy_birth_segment_refill_enable": True,
            "resident_capacity": 512,
            "ticks_per_generation": 160,
        }
    ) is False
    with pytest.raises(ResourceEstimateError, match="tape-free FE identity"):
        estimate_eiel_full_evolution_v2_packed_resident_resources_v1(
            batch_cells=48,
            logical_generations=3,
            logical_trajectories=49,
            checkpoint_boundaries=2,
            resident_segment_generations=3,
            ticks_per_generation=160,
            resident_capacity=512,
            cpython_gauss_segment_capacity=65_536,
            policy_birth_segment_refill_enable=True,
        )


def test_v1_default_route_unchanged() -> None:
    assert resolve_rng_route()["route_id"] == V1_ROUTE_ID
