"""CPU/static checks for the C4 R11 exact MT Gaussian continuation boundary."""

from __future__ import annotations

import random
import struct
from pathlib import Path

import pytest

from engine.perf.cuda_architecture_b_lifecycle import (
    pack_cpython_random_state_cuda,
)
from engine.tensor_world.c4_resident_mt_continuation_r11 import (
    C4_RESIDENT_MT_CONTINUATION_R11_DECISION_V1,
    C4_RESIDENT_MT_CONTINUATION_R11_SCHEMA_VERSION_V1,
    C4ResidentMtGaussianRingWindowR11V1,
    c4_resident_mt_continuation_audit_r11_v1,
    pack_authoritative_c4_gaussian_ring_r11_v1,
    read_authoritative_c4_gaussian_ring_r11_v1,
)
from engine.tensor_world.c4_resident_rng_admission_r11 import _mt_window_bounds
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]


def _bits(value: float) -> bytes:
    return struct.pack("=d", value)


def test_current_source_receipt_separates_exact_ring_from_missing_producer() -> None:
    receipt = c4_resident_mt_continuation_audit_r11_v1()

    assert receipt["schema_version"] == C4_RESIDENT_MT_CONTINUATION_R11_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENT_MT_CONTINUATION_R11_DECISION_V1
    assert receipt["gate_claim"] is False
    assert receipt["frozen_semantics"]["pair_transform_symbols_present"] is True
    assert receipt["current_source"] == {
        "r11_device_consumes_precomputed_tape": True,
        "r11_device_gaussian_producer": False,
        "r11_state_has_ring_base_and_valid_end": False,
        "existing_refill_executes_on_host": True,
        "existing_refill_materializes_device_state_on_host": True,
        "existing_refill_copies_new_tape_to_device": True,
    }
    assert receipt["bounded_ring"]["exact_layout_oracle_available"] is True
    assert receipt["bounded_ring"]["new_value_generation"] is False
    assert receipt["blocker"]["kind"] == "MISSING_BIT_EXACT_DEVICE_GAUSSIAN_PRODUCER"
    assert receipt["blocker"]["mathematically_impossible"] is False
    assert receipt["blocker"]["source_contract_portable_across_libm_implementations"] is False
    assert all(len(row["sha256"]) == 64 for row in receipt["sources"].values())


def test_five_tick_window_is_bounded_but_not_reusable_without_new_values() -> None:
    receipt = c4_resident_mt_continuation_audit_r11_v1()
    window = receipt["five_tick_window"]

    assert window == {
        "max_births": 320,
        "retry_cap": 64,
        "required_tape_headroom": 33_211,
        "resident_tape_capacity": 65_536,
        "admitted_from_zero": True,
    }
    alternatives = {row["name"]: row for row in receipt["smallest_admissible_alternatives"]}
    assert alternatives["finite_exact_checkpoint_window"]["long_run_no_host_refill"] is False
    assert alternatives["device_software_libm_clone"]["long_run_no_host_refill"] is True
    assert alternatives["specified_integer_normal_sampler"]["preserves_frozen_oracle"] is False

    false_cache = _mt_window_bounds(births=320, initial_has_gauss=False)
    true_cache = _mt_window_bounds(births=320, initial_has_gauss=True)
    assert max(false_cache[1], true_cache[1]) == window["required_tape_headroom"]


def test_receipt_claims_are_bound_to_current_host_refill_and_device_consumers() -> None:
    r4 = (
        ROOT / "engine/tensor_world/native/tensor_world_c4_resident_mutation_r4.cu"
    ).read_text(encoding="utf-8")
    scalar = (
        ROOT / "engine/perf/cuda_architecture_b_persistent_birth_scalar_genome.cuh"
    ).read_text(encoding="utf-8")
    r11 = (
        ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
    ).read_text(encoding="utf-8")
    host_native = (
        ROOT / "engine/perf/cuda_architecture_b_resident_window_refill_v1.hpp"
    ).read_text(encoding="utf-8")
    host_python = (
        ROOT / "engine/perf/cuda_architecture_b_resident_window_runtime_v1.py"
    ).read_text(encoding="utf-8")
    state = (ROOT / "engine/tensor_world/c4_state.py").read_text(encoding="utf-8")

    assert "rng.gauss_primary_tape[tape_index]" in r4
    assert "gauss_primary_tape[tape_index]" in scalar
    assert all(
        symbol not in r4 + scalar + r11
        for symbol in ("std::log(", "std::sqrt(", "std::cos(", "std::sin(")
    )
    for expression in (
        "std::sqrt(-2.0 * std::log(1.0 - second))",
        "std::cos(x2pi) * radius",
        "std::sin(x2pi) * radius",
        "words.detach().cpu().contiguous()",
        "primary.copy_(staged_primary)",
    ):
        assert expression in host_native
    assert 'device="cpu"' in host_python
    assert "copy_(primary_host)" in host_python
    assert "python_random_window_base" not in state


def test_ring_layout_wraps_storage_without_changing_authoritative_float_bits() -> None:
    source = random.Random(0xC411)
    packed = pack_cpython_random_state_cuda(
        source.getstate(), device="cpu", gauss_tape_raw_capacity=8
    )
    primary = tuple(float(value) for value in packed["python_random_gauss_primary_tape"])
    secondary = tuple(float(value) for value in packed["python_random_gauss_secondary_tape"])

    window, primary_ring, secondary_ring = pack_authoritative_c4_gaussian_ring_r11_v1(
        primary,
        secondary,
        base_raw_draw=37,
        capacity=8,
    )

    assert [window.physical_index(37 + offset) for offset in range(8)] == [
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        4,
    ]
    for offset, expected in enumerate(zip(primary, secondary, strict=True)):
        actual = read_authoritative_c4_gaussian_ring_r11_v1(
            window,
            primary_ring,
            secondary_ring,
            absolute_raw_draw=37 + offset,
        )
        assert tuple(_bits(value) for value in actual) == tuple(
            _bits(value) for value in expected
        )


@pytest.mark.parametrize("absolute", (36, 45))
def test_ring_read_fails_closed_instead_of_repeating_stale_values(absolute: int) -> None:
    window, primary, secondary = pack_authoritative_c4_gaussian_ring_r11_v1(
        [1.0, 2.0], [3.0, 4.0], base_raw_draw=37, capacity=2
    )

    with pytest.raises(TensorWorldContractError, match="authoritative interval"):
        read_authoritative_c4_gaussian_ring_r11_v1(
            window,
            primary,
            secondary,
            absolute_raw_draw=absolute,
        )


@pytest.mark.parametrize(
    "factory",
    (
        lambda: C4ResidentMtGaussianRingWindowR11V1(-1, 0, 2),
        lambda: C4ResidentMtGaussianRingWindowR11V1(2, 1, 2),
        lambda: C4ResidentMtGaussianRingWindowR11V1(0, 3, 2),
        lambda: pack_authoritative_c4_gaussian_ring_r11_v1(
            [1.0], [], base_raw_draw=0, capacity=2
        ),
        lambda: pack_authoritative_c4_gaussian_ring_r11_v1(
            [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], base_raw_draw=0, capacity=2
        ),
        lambda: pack_authoritative_c4_gaussian_ring_r11_v1(
            [float("nan")], [0.0], base_raw_draw=0, capacity=2
        ),
    ),
)
def test_malformed_ring_contract_fails_before_any_authoritative_mutation(factory) -> None:
    with pytest.raises(TensorWorldContractError):
        factory()
