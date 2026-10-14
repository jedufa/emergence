"""R11-only RNG capacity admission and capped-draw reference semantics.

This module is the CPU/static oracle for the resident R11 transaction.  It
does not mutate the replay owner during admission and deliberately does not
change the frozen R2-R9 adapters.  The native R11 route mirrors these bounds
before its single guarded commit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch

from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import TensorWorldC5ReplayContextV1
from .contracts import TensorWorldContractError


C4_RESIDENT_RNG_ADMISSION_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_rng_admission_r11_v1"
)
PCG_MAX_RAW32_PER_BOUNDED_DRAW_R11_V1: Final[int] = 64
MT_MAX_RANDBELOW_ATTEMPTS_R11_V1: Final[int] = 64
R11_GRAPH_WINDOW_TICKS_V1: Final[int] = 5
R11_MAX_BIRTHS_PER_TICK_V1: Final[int] = 64
INT64_MAX_V1: Final[int] = (1 << 63) - 1

C4_R11_RNG_STATUS_OK_V1: Final[int] = 0
C4_R11_RNG_STATUS_INVALID_ABI_V1: Final[int] = 68
C4_R11_RNG_STATUS_PCG_CAPACITY_V1: Final[int] = 69
C4_R11_RNG_STATUS_MT_CAPACITY_V1: Final[int] = 70
C4_R11_RNG_STATUS_COUNTER_OVERFLOW_V1: Final[int] = 71
C4_R11_RNG_STATUS_PEER_FAILED_V1: Final[int] = 72


@dataclass(frozen=True, slots=True)
class C4ResidentRngAdmissionR11V1:
    """Whole-batch, no-mutation admission result."""

    local_status: torch.Tensor
    batch_status: torch.Tensor
    pcg_environment_raw32: int
    pcg_runner_raw32: int
    mt_raw32: int
    mt_semantic: int
    mt_gaussian_tape_headroom: int
    window_ticks: int
    max_total_births: int
    pcg_retry_exhaustion_union_bound: float
    mt_retry_exhaustion_union_bound: float
    schema_version: str = C4_RESIDENT_RNG_ADMISSION_R11_SCHEMA_VERSION_V1

    @property
    def admitted(self) -> bool:
        return bool((self.batch_status == C4_R11_RNG_STATUS_OK_V1).all().item())


@dataclass(slots=True)
class C4ResidentRngNativeStagingR11V1:
    """Caller-owned post-counter shadow for one batch-atomic native call."""

    post_raw64_transition_counts: torch.Tensor
    post_output32_return_counts: torch.Tensor

    def validate(self, *, batch: int, streams: int, device: torch.device) -> None:
        for name, tensor in (
            ("post_raw64_transition_counts", self.post_raw64_transition_counts),
            ("post_output32_return_counts", self.post_output32_return_counts),
        ):
            if (
                tuple(tensor.shape) != (batch, streams)
                or tensor.dtype != torch.int64
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise TensorWorldContractError(f"C4 R11 RNG staging {name} ABI mismatch")


def make_c4_resident_rng_native_staging_r11_v1(
    *, batch: int, streams: int, device: torch.device | str
) -> C4ResidentRngNativeStagingR11V1:
    if type(batch) is not int or batch <= 0 or type(streams) is not int or streams <= 0:
        raise TensorWorldContractError("C4 R11 RNG staging dimensions must be positive")
    target = torch.device(device)
    shape = (batch, streams)
    return C4ResidentRngNativeStagingR11V1(
        post_raw64_transition_counts=torch.zeros(shape, dtype=torch.int64, device=target),
        post_output32_return_counts=torch.zeros(shape, dtype=torch.int64, device=target),
    )


def _would_overflow(value: int, addition: int) -> bool:
    return value < 0 or addition < 0 or value > INT64_MAX_V1 - addition


def _raw64_transitions(raw32_outputs: int, *, cached32: bool) -> int:
    cached = 1 if cached32 else 0
    return max(0, (raw32_outputs - cached + 1) // 2)


def _mt_window_bounds(*, births: int, initial_has_gauss: bool) -> tuple[int, int]:
    """Return continuous-window raw32 count and last tape-index headroom."""

    raw32 = 0
    last_tape_index = -1
    has_gauss = initial_has_gauss

    def random_double() -> None:
        nonlocal raw32
        raw32 += 2

    def gauss() -> None:
        nonlocal raw32, last_tape_index, has_gauss
        if has_gauss:
            has_gauss = False
            return
        last_tape_index = raw32
        raw32 += 4
        has_gauss = True

    for _birth in range(births):
        random_double()
        for _gene in range(8):
            random_double()
            gauss()
        random_double()
        gauss()
        random_double()
        raw32 += MT_MAX_RANDBELOW_ATTEMPTS_R11_V1
    return raw32, last_tape_index + 1


def preflight_c4_resident_rng_admission_r11_v1(
    bundle: TensorWorldStateBundleV4,
    replay: TensorWorldC5ReplayContextV1,
    *,
    window_ticks: int = R11_GRAPH_WINDOW_TICKS_V1,
    max_births: int = R11_MAX_BIRTHS_PER_TICK_V1,
) -> C4ResidentRngAdmissionR11V1:
    """Preflight the exact B48/C512/M64 RNG boundary without mutation.

    Every counter is checked against its maximum possible increment under the
    fixed retry caps.  PCG raw64 headroom is cache-dependent; MT Gaussian tape
    headroom is the exact maximum fresh-tape index span for either initial
    Gaussian-cache state.
    """

    if type(window_ticks) is not int or not 1 <= window_ticks <= R11_GRAPH_WINDOW_TICKS_V1:
        raise TensorWorldContractError("C4 R11 RNG window_ticks must be in [1,5]")
    if type(max_births) is not int or not 0 <= max_births <= R11_MAX_BIRTHS_PER_TICK_V1:
        raise TensorWorldContractError("C4 R11 RNG max_births must be in [0,64]")
    bundle.validate()
    replay.validate_typed_continuation_for_bundle(bundle)
    if bundle.device.type != "cpu" or replay.device.type != "cpu":
        raise TensorWorldContractError("C4 R11 RNG admission oracle requires CPU state")
    if bundle.batch != 48 or bundle.capacity not in (32, 512) or max_births > bundle.capacity:
        raise TensorWorldContractError("C4 R11 RNG admission requires B48/C32-or-C512")
    if bundle.scalar_rng is None:
        raise TensorWorldContractError("C4 R11 RNG admission requires CPython RNG state")
    indices = replay.typed_role_indices
    raw64 = replay.raw64_transition_counts
    output32 = replay.output32_return_counts
    semantic = replay.semantic_draw_counters
    buffer = replay.rng_buffer_words
    assert indices is not None and raw64 is not None and output32 is not None
    assert semantic is not None and buffer is not None

    active_counts = [int(value) for value in bundle.population.active_count.tolist()]
    if any(value < 0 or value > bundle.capacity for value in active_counts):
        raise TensorWorldContractError("C4 R11 RNG active counts are invalid")
    max_decisions = 0 if max_births == 0 else max(active_counts, default=0)
    env_raw32 = window_ticks * max_decisions * 259
    runner_raw32 = window_ticks * max_decisions * 66
    total_births = window_ticks * max_births
    mt_false = _mt_window_bounds(births=total_births, initial_has_gauss=False)
    mt_true = _mt_window_bounds(births=total_births, initial_has_gauss=True)
    mt_raw32 = max(mt_false[0], mt_true[0])
    mt_semantic = total_births * 21
    mt_tape = max(mt_false[1], mt_true[1])
    env_stream, runner_stream = indices[:2]
    local = torch.zeros(bundle.batch, dtype=torch.int32)
    scalar = bundle.scalar_rng

    for world in range(bundle.batch):
        decisions = 0 if max_births == 0 else active_counts[world]
        world_env_raw32 = window_ticks * decisions * 259
        world_runner_raw32 = window_ticks * decisions * 66
        if scalar.tape_capacity - int(scalar.python_random_raw_draws[world]) < mt_tape:
            local[world] = C4_R11_RNG_STATUS_MT_CAPACITY_V1
            continue
        if _would_overflow(int(scalar.python_random_raw_draws[world]), mt_raw32) or (
            _would_overflow(int(scalar.mutation_py_semantic_draws[world]), mt_semantic)
        ):
            local[world] = C4_R11_RNG_STATUS_COUNTER_OVERFLOW_V1
            continue
        for stream, required_raw32, direct_raw64, bounded, semantic_added in (
            (
                env_stream,
                world_env_raw32,
                0,
                window_ticks * decisions * 7,
                window_ticks * decisions * 8,
            ),
            (
                runner_stream,
                world_runner_raw32,
                window_ticks * decisions * 4,
                window_ticks * decisions * 3,
                window_ticks * decisions * 4,
            ),
        ):
            cached32 = int(buffer[world, stream, 0]) == 1
            required_raw64 = direct_raw64 + _raw64_transitions(
                required_raw32, cached32=cached32
            )
            counters = (
                (int(raw64[world, stream]), required_raw64),
                (int(output32[world, stream]), required_raw32),
                (int(replay.rng_cursors[world, stream]), bounded),
                (int(semantic[world, stream]), semantic_added),
            )
            if any(_would_overflow(value, addition) for value, addition in counters):
                local[world] = C4_R11_RNG_STATUS_COUNTER_OVERFLOW_V1
                break

    failed = bool((local != C4_R11_RNG_STATUS_OK_V1).any().item())
    batch_status = local.clone()
    if failed:
        batch_status[batch_status == C4_R11_RNG_STATUS_OK_V1] = (
            C4_R11_RNG_STATUS_PEER_FAILED_V1
        )
    return C4ResidentRngAdmissionR11V1(
        local_status=local,
        batch_status=batch_status,
        pcg_environment_raw32=env_raw32,
        pcg_runner_raw32=runner_raw32,
        mt_raw32=mt_raw32,
        mt_semantic=mt_semantic,
        mt_gaussian_tape_headroom=mt_tape,
        window_ticks=window_ticks,
        max_total_births=total_births,
        pcg_retry_exhaustion_union_bound=(
            window_ticks * sum(active_counts) * 5 * (3.0 / 8.0) ** 64
            if max_births > 0 else 0.0
        ),
        mt_retry_exhaustion_union_bound=(
            bundle.batch * total_births * (0.5 ** 64)
        ),
    )


def capped_rejection_draw_r11_v1(
    raw_values: tuple[int, ...], *, maximum: int, attempts: int
) -> tuple[int, int]:
    """Reference the R11 inclusive bounded-draw retry contract.

    Successful draws return exactly the first source-current accepted value.
    Exhaustion raises before any caller-owned RNG state is committed; the
    native implementation stages its RNG words/counters for the same reason.
    """

    if type(maximum) is not int or not 0 <= maximum <= 0xFFFFFFFF:
        raise TensorWorldContractError("C4 R11 bounded maximum is outside uint32")
    if type(attempts) is not int or attempts < 1:
        raise TensorWorldContractError("C4 R11 retry cap must be a positive int")
    mask = maximum
    mask |= mask >> 1
    mask |= mask >> 2
    mask |= mask >> 4
    mask |= mask >> 8
    mask |= mask >> 16
    for index, raw in enumerate(raw_values[:attempts], start=1):
        value = int(raw) & mask
        if value <= maximum:
            return value, index
    raise TensorWorldContractError("C4 R11 bounded RNG retry cap exhausted")


__all__ = [
    "C4_RESIDENT_RNG_ADMISSION_R11_SCHEMA_VERSION_V1",
    "C4_R11_RNG_STATUS_COUNTER_OVERFLOW_V1",
    "C4_R11_RNG_STATUS_MT_CAPACITY_V1",
    "C4_R11_RNG_STATUS_PEER_FAILED_V1",
    "MT_MAX_RANDBELOW_ATTEMPTS_R11_V1",
    "PCG_MAX_RAW32_PER_BOUNDED_DRAW_R11_V1",
    "R11_GRAPH_WINDOW_TICKS_V1",
    "R11_MAX_BIRTHS_PER_TICK_V1",
    "C4ResidentRngAdmissionR11V1",
    "C4ResidentRngNativeStagingR11V1",
    "capped_rejection_draw_r11_v1",
    "make_c4_resident_rng_native_staging_r11_v1",
    "preflight_c4_resident_rng_admission_r11_v1",
]
