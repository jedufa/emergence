"""Exact caller-owned Gaussian-window continuation for Tensor World C5 R11.

Host generation is permitted only at an explicit control/checkpoint boundary.
Normal R11 graph ticks consume the published device tensors directly and fail
closed if the absolute draw cursor is outside the authoritative interval.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final, Literal

import torch

from engine.perf.cuda_architecture_b_resident_window_runtime_v1 import (
    ResidentWindowContractError,
    refill_cpython_gaussian_window_,
)

from .c4_state import C4PythonRandomTensorStateV1, TensorWorldStateBundleV4
from .c5_determinism import (
    C5_GAUSSIAN_WINDOW_AUTHORITATIVE_R11_V1,
    C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1,
    C5_GAUSSIAN_WINDOW_REFILL_IN_PROGRESS_R11_V1,
    C5_GAUSSIAN_WINDOW_SCHEMA_VERSION_R11_V1,
    TensorWorldC5ReplayContextV1,
)
from .contracts import TensorWorldContractError


C5_GAUSSIAN_REFILL_SCHEMA_VERSION_R11_V1: Final[str] = (
    "tensor_world_c5_gaussian_refill_r11_v1"
)
C5_GAUSSIAN_REFILL_DECISION_R11_V1: Final[str] = "NO_GATE_DECISION"
C5_GAUSSIAN_MT_MAX_RANDBELOW_ATTEMPTS_R11_V1: Final[int] = 64
C5_GAUSSIAN_NATIVE_WINDOW_TENSOR_COUNT_R11_V1: Final[int] = 6

BoundaryKind = Literal["initialization", "checkpoint", "control"]


@dataclass(frozen=True, slots=True)
class C5GaussianWindowAdmissionR11V1:
    boundary_kind: BoundaryKind
    source_commit: str
    boundary_label: str
    requested_ticks: int
    max_births_per_tick: int
    required_tape_headroom_by_world: tuple[int, ...]
    refill_performed: bool
    refill_generation_by_world: tuple[int, ...]
    base_raw_draws_by_world: tuple[int, ...]
    valid_raw_draw_ends_by_world: tuple[int, ...]
    absolute_raw_draws_by_world: tuple[int, ...]
    schema_version: str = C5_GAUSSIAN_REFILL_SCHEMA_VERSION_R11_V1

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "decision": C5_GAUSSIAN_REFILL_DECISION_R11_V1,
            "boundary_kind": self.boundary_kind,
            "source_commit": self.source_commit,
            "boundary_label": self.boundary_label,
            "requested_ticks": self.requested_ticks,
            "max_births_per_tick": self.max_births_per_tick,
            "required_tape_headroom_by_world": list(
                self.required_tape_headroom_by_world
            ),
            "refill_performed": self.refill_performed,
            "refill_generation_by_world": list(self.refill_generation_by_world),
            "base_raw_draws_by_world": list(self.base_raw_draws_by_world),
            "valid_raw_draw_ends_by_world": list(
                self.valid_raw_draw_ends_by_world
            ),
            "absolute_raw_draws_by_world": list(
                self.absolute_raw_draws_by_world
            ),
            "gate_claim": False,
        }


def gaussian_window_tensors_r11_v1(
    replay: TensorWorldC5ReplayContextV1,
) -> list[torch.Tensor]:
    window = replay.gaussian_window
    if window is None:
        raise TensorWorldContractError(
            "C5 R11 Gaussian-window native ABI requires typed authority"
        )
    return [
        window.base_raw_draws,
        window.valid_raw_draw_ends,
        window.refill_generations,
        window.provenance_sha256,
        window.status,
        window.exhaustion_raw_draws,
    ]


def _scalar_rng_mapping(
    state: C4PythonRandomTensorStateV1,
) -> dict[str, torch.Tensor]:
    return {
        "python_random_words": state.python_random_words,
        "python_random_index": state.python_random_index,
        "python_random_has_gauss": state.python_random_has_gauss,
        "python_random_gauss_next": state.python_random_gauss_next,
        "python_random_gauss_primary_tape": (
            state.python_random_gauss_primary_tape
        ),
        "python_random_gauss_secondary_tape": (
            state.python_random_gauss_secondary_tape
        ),
        "python_random_raw_draws": state.python_random_raw_draws,
        "mutation_py_semantic_draws": state.mutation_py_semantic_draws,
    }


def _mt_tape_headroom(*, births: int, initial_has_gauss: bool) -> int:
    if type(births) is not int or births < 0:
        raise TensorWorldContractError("C5 R11 birth bound must be nonnegative")
    raw = 0
    last_tape = -1
    has_gauss = bool(initial_has_gauss)
    for _ in range(births):
        raw += 2
        for _ in range(8):
            raw += 2
            if has_gauss:
                has_gauss = False
            else:
                last_tape = raw
                raw += 4
                has_gauss = True
        raw += 2
        if has_gauss:
            has_gauss = False
        else:
            last_tape = raw
            raw += 4
            has_gauss = True
        raw += 2 + C5_GAUSSIAN_MT_MAX_RANDBELOW_ATTEMPTS_R11_V1
    return last_tape + 1


def _provenance_rows(
    state: C4PythonRandomTensorStateV1,
    replay: TensorWorldC5ReplayContextV1,
    *,
    source_commit: str,
    boundary_label: str,
) -> torch.Tensor:
    window = replay.gaussian_window
    assert window is not None
    base = window.base_raw_draws.detach().cpu()
    end = window.valid_raw_draw_ends.detach().cpu()
    generations = window.refill_generations.detach().cpu()
    rows: list[list[int]] = []
    for world in range(state.batch):
        hasher = hashlib.sha256()
        for value in (
            C5_GAUSSIAN_WINDOW_SCHEMA_VERSION_R11_V1,
            source_commit,
            boundary_label,
        ):
            hasher.update(value.encode("utf-8"))
            hasher.update(b"\0")
        for value in (world, int(base[world]), int(end[world]), int(generations[world])):
            hasher.update(int(value).to_bytes(8, "little", signed=True))
        for tensor in (
            state.python_random_words[world],
            state.python_random_index[world : world + 1],
            state.python_random_has_gauss[world : world + 1],
            state.python_random_gauss_next[world : world + 1],
            state.python_random_gauss_primary_tape[world],
            state.python_random_gauss_secondary_tape[world],
        ):
            host = tensor.detach().to(device="cpu", copy=True).contiguous()
            hasher.update(host.view(torch.uint8).numpy().tobytes())
        rows.append(list(hasher.digest()))
    return torch.tensor(rows, dtype=torch.uint8, device=state.device)


def refill_c5_gaussian_window_r11_v1(
    bundle: TensorWorldStateBundleV4,
    replay: TensorWorldC5ReplayContextV1,
    *,
    boundary_kind: BoundaryKind,
    source_commit: str,
    boundary_label: str,
) -> dict[str, object]:
    """Generate and publish one exact window at a declared safe boundary."""

    if boundary_kind not in {"initialization", "checkpoint", "control"}:
        raise TensorWorldContractError("C5 Gaussian refill boundary is unsupported")
    if not source_commit or not boundary_label:
        raise TensorWorldContractError("C5 Gaussian refill identity is incomplete")
    replay.validate_typed_continuation_for_bundle(bundle)
    state = bundle.scalar_rng
    window = replay.gaussian_window
    assert state is not None and window is not None
    absolute_before = state.python_random_raw_draws.detach().clone()
    semantic_before = state.mutation_py_semantic_draws.detach().clone()
    generation_before = window.refill_generations.detach().clone()
    window.status.fill_(C5_GAUSSIAN_WINDOW_REFILL_IN_PROGRESS_R11_V1)
    window.exhaustion_raw_draws.copy_(state.python_random_raw_draws)
    try:
        receipt = refill_cpython_gaussian_window_(
            _scalar_rng_mapping(state),
            python_random_window_base=window.base_raw_draws,
        )
        if not torch.equal(state.python_random_raw_draws, absolute_before):
            raise TensorWorldContractError(
                "C5 Gaussian refill changed the absolute raw-draw counter"
            )
        if not torch.equal(state.mutation_py_semantic_draws, semantic_before):
            raise TensorWorldContractError(
                "C5 Gaussian refill changed the semantic draw counter"
            )
        window.valid_raw_draw_ends.copy_(
            window.base_raw_draws + int(state.tape_capacity)
        )
        window.refill_generations.copy_(generation_before + 1)
        window.provenance_sha256.copy_(
            _provenance_rows(
                state,
                replay,
                source_commit=source_commit,
                boundary_label=boundary_label,
            )
        )
        window.exhaustion_raw_draws.fill_(-1)
        # Publication is deliberately last.  Any exception above leaves the
        # context in REFILL_IN_PROGRESS and native replay must reject it.
        window.status.fill_(C5_GAUSSIAN_WINDOW_AUTHORITATIVE_R11_V1)
    except (ResidentWindowContractError, RuntimeError, TensorWorldContractError):
        raise
    replay.validate_typed_continuation_for_bundle(bundle)
    return {
        **receipt,
        "schema_version": C5_GAUSSIAN_REFILL_SCHEMA_VERSION_R11_V1,
        "boundary_kind": boundary_kind,
        "source_commit": source_commit,
        "boundary_label": boundary_label,
        "refill_generation_before": generation_before.detach().cpu().tolist(),
        "refill_generation_after": (
            window.refill_generations.detach().cpu().tolist()
        ),
        "publish_last": True,
        "gate_claim": False,
    }


def admit_c5_gaussian_window_r11_v1(
    bundle: TensorWorldStateBundleV4,
    replay: TensorWorldC5ReplayContextV1,
    *,
    requested_ticks: int,
    max_births_per_tick: int,
    boundary_kind: BoundaryKind,
    source_commit: str,
    boundary_label: str,
) -> C5GaussianWindowAdmissionR11V1:
    """Refill if needed, then prove the whole graph window is admissible."""

    if type(requested_ticks) is not int or requested_ticks < 1:
        raise TensorWorldContractError("C5 requested tick count must be positive")
    if type(max_births_per_tick) is not int or max_births_per_tick < 0:
        raise TensorWorldContractError("C5 max births per tick must be nonnegative")
    replay.validate_typed_continuation_for_bundle(bundle)
    state = bundle.scalar_rng
    window = replay.gaussian_window
    assert state is not None and window is not None
    raw = state.python_random_raw_draws.detach().cpu()
    base = window.base_raw_draws.detach().cpu()
    end = window.valid_raw_draw_ends.detach().cpu()
    status = window.status.detach().cpu()
    has_gauss = state.python_random_has_gauss.detach().cpu()
    headroom = tuple(
        _mt_tape_headroom(
            births=requested_ticks * max_births_per_tick,
            initial_has_gauss=bool(has_gauss[world]),
        )
        for world in range(bundle.batch)
    )
    usable = all(
        int(status[world]) == C5_GAUSSIAN_WINDOW_AUTHORITATIVE_R11_V1
        and int(base[world]) <= int(raw[world]) <= int(end[world])
        and headroom[world] <= int(end[world]) - int(raw[world])
        for world in range(bundle.batch)
    )
    refilled = False
    if not usable:
        if state.tape_capacity == 0:
            window.status.fill_(C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1)
            window.exhaustion_raw_draws.copy_(state.python_random_raw_draws)
            raise TensorWorldContractError(
                "C5 Gaussian physical capacity cannot admit the requested graph window"
            )
        refill_c5_gaussian_window_r11_v1(
            bundle,
            replay,
            boundary_kind=boundary_kind,
            source_commit=source_commit,
            boundary_label=boundary_label,
        )
        refilled = True
        raw = state.python_random_raw_draws.detach().cpu()
        base = window.base_raw_draws.detach().cpu()
        end = window.valid_raw_draw_ends.detach().cpu()
    insufficient = [
        world
        for world in range(bundle.batch)
        if headroom[world] > int(end[world]) - int(raw[world])
    ]
    if insufficient:
        window.status.fill_(C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1)
        window.exhaustion_raw_draws.copy_(state.python_random_raw_draws)
        raise TensorWorldContractError(
            "C5 Gaussian physical capacity cannot admit the requested graph window"
        )
    return C5GaussianWindowAdmissionR11V1(
        boundary_kind=boundary_kind,
        source_commit=source_commit,
        boundary_label=boundary_label,
        requested_ticks=requested_ticks,
        max_births_per_tick=max_births_per_tick,
        required_tape_headroom_by_world=headroom,
        refill_performed=refilled,
        refill_generation_by_world=tuple(
            int(value) for value in window.refill_generations.detach().cpu().tolist()
        ),
        base_raw_draws_by_world=tuple(int(value) for value in base.tolist()),
        valid_raw_draw_ends_by_world=tuple(int(value) for value in end.tolist()),
        absolute_raw_draws_by_world=tuple(int(value) for value in raw.tolist()),
    )


__all__ = [
    "C5_GAUSSIAN_NATIVE_WINDOW_TENSOR_COUNT_R11_V1",
    "C5_GAUSSIAN_REFILL_DECISION_R11_V1",
    "C5_GAUSSIAN_REFILL_SCHEMA_VERSION_R11_V1",
    "C5GaussianWindowAdmissionR11V1",
    "admit_c5_gaussian_window_r11_v1",
    "gaussian_window_tensors_r11_v1",
    "refill_c5_gaussian_window_r11_v1",
]
