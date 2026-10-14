"""Exact audit and bounded-ring oracle for C4 R11 CPython Gaussian state.

The frozen scalar mutation oracle advances CPython MT19937 on device but reads
the Box-Muller results from a host-generated float64 tape.  A circular layout
can reuse the physical storage without changing any value bits.  It cannot,
by itself, create the next source-oracle values.  This module makes that seam
explicit and fails closed at the end of the values known to be valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import inspect
import math
from pathlib import Path
import platform
import random
from typing import Final, Sequence

from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
CPU_GENOME = ROOT / "engine/tensor_world/c4_genome.py"
C4_STATE = ROOT / "engine/tensor_world/c4_state.py"
R4_NATIVE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_mutation_r4.cu"
R11_NATIVE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
SCALAR_HEADER = ROOT / "engine/perf/cuda_architecture_b_persistent_birth_scalar_genome.cuh"
HOST_REFILL_PY = ROOT / "engine/perf/cuda_architecture_b_resident_window_runtime_v1.py"
HOST_REFILL_HPP = ROOT / "engine/perf/cuda_architecture_b_resident_window_refill_v1.hpp"
PYTHON_RANDOM = Path(inspect.getsourcefile(random.Random.gauss) or random.__file__).resolve()

C4_RESIDENT_MT_CONTINUATION_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_mt_continuation_r11_v1"
)
C4_RESIDENT_MT_CONTINUATION_R11_DECISION_V1: Final[str] = (
    "EXACT_DEVICE_REFILL_NOT_IMPLEMENTED"
)
R11_GAUSSIAN_TAPE_CAPACITY_V1: Final[int] = 65_536
R11_FIVE_TICK_MAX_BIRTHS_V1: Final[int] = 320
R11_FIVE_TICK_MT_TAPE_HEADROOM_V1: Final[int] = 33_211


def _exact_nonnegative_int(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise TensorWorldContractError(f"C4 R11 {name} must be a nonnegative int")
    return value


@dataclass(frozen=True, slots=True)
class C4ResidentMtGaussianRingWindowR11V1:
    """One world's valid absolute interval in circular physical storage."""

    base_raw_draw: int
    valid_raw_draw_end: int
    capacity: int
    schema_version: str = C4_RESIDENT_MT_CONTINUATION_R11_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        base = _exact_nonnegative_int(self.base_raw_draw, name="ring base")
        end = _exact_nonnegative_int(self.valid_raw_draw_end, name="ring valid end")
        capacity = _exact_nonnegative_int(self.capacity, name="ring capacity")
        if capacity < 1:
            raise TensorWorldContractError("C4 R11 ring capacity must be positive")
        if end < base:
            raise TensorWorldContractError("C4 R11 ring valid end precedes its base")
        if end - base > capacity:
            raise TensorWorldContractError(
                "C4 R11 ring valid interval exceeds physical capacity"
            )

    def physical_index(self, absolute_raw_draw: int) -> int:
        """Resolve one valid absolute tape index without silently repeating."""

        absolute = _exact_nonnegative_int(
            absolute_raw_draw, name="absolute raw draw"
        )
        if absolute < self.base_raw_draw or absolute >= self.valid_raw_draw_end:
            raise TensorWorldContractError(
                "C4 R11 Gaussian ring index is outside the authoritative interval"
            )
        return absolute % self.capacity


def pack_authoritative_c4_gaussian_ring_r11_v1(
    primary: Sequence[float],
    secondary: Sequence[float],
    *,
    base_raw_draw: int,
    capacity: int,
) -> tuple[
    C4ResidentMtGaussianRingWindowR11V1,
    tuple[float, ...],
    tuple[float, ...],
]:
    """Place already-authoritative Gaussian pairs in circular CPU storage.

    This is a reference layout transform, not a Gaussian producer or resident
    refill.  The source sequences must already contain the exact CPython oracle
    values for consecutive absolute raw-draw indices starting at ``base``.
    """

    base = _exact_nonnegative_int(base_raw_draw, name="ring base")
    physical_capacity = _exact_nonnegative_int(capacity, name="ring capacity")
    if physical_capacity < 1:
        raise TensorWorldContractError("C4 R11 ring capacity must be positive")
    if len(primary) != len(secondary):
        raise TensorWorldContractError("C4 R11 Gaussian ring pair lengths differ")
    if len(primary) > physical_capacity:
        raise TensorWorldContractError(
            "C4 R11 Gaussian ring values exceed physical capacity"
        )
    primary_values = tuple(float(value) for value in primary)
    secondary_values = tuple(float(value) for value in secondary)
    if not all(math.isfinite(value) for value in primary_values + secondary_values):
        raise TensorWorldContractError("C4 R11 Gaussian ring values must be finite")

    window = C4ResidentMtGaussianRingWindowR11V1(
        base_raw_draw=base,
        valid_raw_draw_end=base + len(primary_values),
        capacity=physical_capacity,
    )
    primary_ring = [0.0] * physical_capacity
    secondary_ring = [0.0] * physical_capacity
    for offset, (left, right) in enumerate(
        zip(primary_values, secondary_values, strict=True)
    ):
        physical = (base + offset) % physical_capacity
        primary_ring[physical] = left
        secondary_ring[physical] = right
    return window, tuple(primary_ring), tuple(secondary_ring)


def read_authoritative_c4_gaussian_ring_r11_v1(
    window: C4ResidentMtGaussianRingWindowR11V1,
    primary_ring: Sequence[float],
    secondary_ring: Sequence[float],
    *,
    absolute_raw_draw: int,
) -> tuple[float, float]:
    """Read a pair only when the absolute index remains authoritative."""

    if len(primary_ring) != window.capacity or len(secondary_ring) != window.capacity:
        raise TensorWorldContractError("C4 R11 Gaussian ring storage shape mismatch")
    physical = window.physical_index(absolute_raw_draw)
    return float(primary_ring[physical]), float(secondary_ring[physical])


def c4_resident_mt_continuation_audit_r11_v1() -> dict[str, object]:
    """Return the current-source decision for exact no-host MT continuation."""

    sources = {
        "python_random_oracle": PYTHON_RANDOM,
        "cpu_genome_cursor": CPU_GENOME,
        "c4_rng_state": C4_STATE,
        "r4_native_consumer": R4_NATIVE,
        "r11_native_owner": R11_NATIVE,
        "scalar_native_consumer": SCALAR_HEADER,
        "existing_host_refill_python": HOST_REFILL_PY,
        "existing_host_refill_native": HOST_REFILL_HPP,
    }
    gauss_source = inspect.getsource(random.Random.gauss)
    return {
        "schema_version": C4_RESIDENT_MT_CONTINUATION_R11_SCHEMA_VERSION_V1,
        "decision": C4_RESIDENT_MT_CONTINUATION_R11_DECISION_V1,
        "gate_claim": False,
        "frozen_semantics": {
            "uniform_bits": "two_MT_raw32_outputs_per_random_double",
            "fresh_gaussian_raw32_outputs": 4,
            "pair_transform": "CPython_random.py_Box_Muller_using_host_math",
            "pair_transform_symbols_present": all(
                symbol in gauss_source
                for symbol in ("_sqrt", "_log", "_cos", "_sin")
            ),
            "cache": "fresh_call_returns_primary_and_caches_secondary",
            "tape_key": "absolute_python_random_raw_draws_before_fresh_pair",
            "oracle_runtime": {
                "python": platform.python_version(),
                "libc": list(platform.libc_ver()),
            },
        },
        "five_tick_window": {
            "max_births": R11_FIVE_TICK_MAX_BIRTHS_V1,
            "retry_cap": 64,
            "required_tape_headroom": R11_FIVE_TICK_MT_TAPE_HEADROOM_V1,
            "resident_tape_capacity": R11_GAUSSIAN_TAPE_CAPACITY_V1,
            "admitted_from_zero": (
                R11_GAUSSIAN_TAPE_CAPACITY_V1
                >= R11_FIVE_TICK_MT_TAPE_HEADROOM_V1
            ),
        },
        "bounded_ring": {
            "exact_layout_oracle_available": True,
            "physical_index": "absolute_raw_draw_mod_capacity",
            "validity_guard": "base_raw_draw<=absolute_raw_draw<valid_raw_draw_end",
            "storage_reusable_after_new_values_exist": True,
            "new_value_generation": False,
            "silent_wrap_forbidden": True,
        },
        "current_source": {
            "r11_device_consumes_precomputed_tape": True,
            "r11_device_gaussian_producer": False,
            "r11_state_has_ring_base_and_valid_end": False,
            "existing_refill_executes_on_host": True,
            "existing_refill_materializes_device_state_on_host": True,
            "existing_refill_copies_new_tape_to_device": True,
        },
        "blocker": {
            "kind": "MISSING_BIT_EXACT_DEVICE_GAUSSIAN_PRODUCER",
            "mathematically_impossible": False,
            "source_contract_portable_across_libm_implementations": False,
            "reason": (
                "the resident MT words determine the uniform inputs, but the frozen "
                "oracle also requires the exact host math.log/sqrt/cos/sin float64 "
                "results; current CUDA sources only read those results from tape"
            ),
            "why_ring_alone_is_insufficient": (
                "bounded circular storage eventually overwrites every authoritative "
                "pair and cannot derive the next pair without the missing producer"
            ),
        },
        "smallest_admissible_alternatives": (
            {
                "name": "finite_exact_checkpoint_window",
                "preserves_frozen_oracle": True,
                "no_host_refill_during_window": True,
                "long_run_no_host_refill": False,
                "action": (
                    "retain five-tick admission and fail closed before the valid "
                    "authoritative interval is exhausted"
                ),
            },
            {
                "name": "device_software_libm_clone",
                "preserves_frozen_oracle": "only_after_exhaustive_bit_parity",
                "no_host_refill_during_window": True,
                "long_run_no_host_refill": True,
                "action": (
                    "freeze the host libm ABI and implement its correctly rounded "
                    "log/sqrt/sin/cos behavior on device"
                ),
            },
            {
                "name": "specified_integer_normal_sampler",
                "preserves_frozen_oracle": False,
                "no_host_refill_during_window": True,
                "long_run_no_host_refill": True,
                "action": "admit as a semantic change and requalify mutation parity",
            },
        ),
        "sources": {
            name: {
                "path": (
                    str(path.relative_to(ROOT))
                    if path.is_relative_to(ROOT)
                    else str(path)
                ),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C4_RESIDENT_MT_CONTINUATION_R11_DECISION_V1",
    "C4_RESIDENT_MT_CONTINUATION_R11_SCHEMA_VERSION_V1",
    "C4ResidentMtGaussianRingWindowR11V1",
    "c4_resident_mt_continuation_audit_r11_v1",
    "pack_authoritative_c4_gaussian_ring_r11_v1",
    "read_authoritative_c4_gaussian_ring_r11_v1",
]
