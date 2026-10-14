"""Exact boundary helpers for reusable Architecture B resident windows."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, MutableSequence, Sequence
from typing import Any, Final, TypeAlias

import torch

from .cuda_architecture_b_boundary import EVENT_COLUMN_DTYPES, EVENT_COLUMN_NAMES
from .cuda_architecture_b_child_policy import (
    CROSSOVER_MASK_ENCODING_V1,
    packed_crossover_row_bytes_v1,
    prepare_fixed_mlp_turnover_child_policy_tape,
)
from .cuda_architecture_b_lifecycle import pack_cpython_random_state_cuda
from .cuda_architecture_b_torch_rng_state_pack_v1 import (
    TORCH_CPU_RNG_STATE_ENCODING_V1,
    TORCH_CPU_RNG_STATE_FULL_BYTES_V1,
    TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
    torch_cpu_rng_state_abi_receipt_v1,
)


RESIDENT_WINDOW_RUNTIME_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_resident_window_runtime_v1"
)

EventChunk: TypeAlias = dict[str, torch.Tensor]
EventArchive: TypeAlias = MutableSequence[EventChunk]

_CPYTHON_STATE_FIELDS: Final[tuple[tuple[str, torch.dtype], ...]] = (
    ("python_random_words", torch.uint32),
    ("python_random_index", torch.int64),
    ("python_random_has_gauss", torch.bool),
    ("python_random_gauss_next", torch.float64),
)
_CPYTHON_WINDOW_FIELDS: Final[tuple[tuple[str, torch.dtype], ...]] = (
    ("python_random_gauss_primary_tape", torch.float64),
    ("python_random_gauss_secondary_tape", torch.float64),
)
_CPYTHON_ABSOLUTE_COUNTER_FIELDS: Final[tuple[str, ...]] = (
    "python_random_raw_draws",
    "mutation_py_semantic_draws",
)

_POLICY_TAPE_FIELDS: Final[tuple[tuple[str, torch.dtype], ...]] = (
    ("policy_birth_delta_tape", torch.float32),
    ("policy_birth_mutation_mask_tape", torch.bool),
    ("policy_birth_crossover_mask_packed_tape", torch.uint8),
    ("policy_birth_post_rng_state_packed_tape", torch.uint8),
    ("policy_birth_semantic_draws_tape", torch.int64),
)
_POLICY_ABSOLUTE_COUNTER_FIELDS: Final[tuple[str, ...]] = (
    "policy_birth_tape_cursor",
    "mutation_torch_cpu_semantic_draws",
)


class ResidentWindowContractError(ValueError):
    """Raised before a malformed resident-window boundary can mutate state."""


def _exact_int(value: object, *, path: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ResidentWindowContractError(
            f"{path} must be an exact integer >= {minimum}"
        )
    return value


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _require_tensor(
    value: object,
    *,
    path: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ResidentWindowContractError(f"{path} must be a Torch tensor")
    tensor = value
    if tensor.dtype != dtype:
        raise ResidentWindowContractError(f"{path} must have dtype {dtype}")
    if tuple(tensor.shape) != shape:
        raise ResidentWindowContractError(
            f"{path} must have shape {list(shape)}"
        )
    if device is not None and tensor.device != device:
        raise ResidentWindowContractError(
            f"{path} must share resident device {device}"
        )
    if not tensor.is_contiguous():
        raise ResidentWindowContractError(f"{path} must be contiguous")
    if tensor.requires_grad:
        raise ResidentWindowContractError(f"{path} must not require gradients")
    return tensor


def _mapping_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    device: torch.device | None = None,
) -> torch.Tensor:
    return _require_tensor(
        tensors.get(name),
        path=name,
        dtype=dtype,
        shape=shape,
        device=device,
    )


def _host_copy(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").contiguous()


def _host_ints(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in _host_copy(tensor).tolist()]


def _pointer_snapshot(tensors: Mapping[str, torch.Tensor]) -> dict[str, int]:
    return {name: int(tensors[name].data_ptr()) for name in sorted(tensors)}


def _pointer_receipt(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, Any]:
    names = sorted(set(before) | set(after))
    rows = [
        {
            "surface": name,
            "before": before.get(name),
            "after": after.get(name),
            "exact": before.get(name) == after.get(name),
        }
        for name in names
    ]
    return {
        "all_exact": all(row["exact"] for row in rows),
        "pointer_count": len(rows),
        "rows": rows,
    }


def _counter_values(
    tensor: torch.Tensor, *, path: str, minimum: int = 0
) -> list[int]:
    values = _host_ints(tensor)
    if any(value < minimum for value in values):
        raise ResidentWindowContractError(
            f"{path} values must be >= {minimum}"
        )
    return values


def _transfer_bytes(tensors: Sequence[torch.Tensor]) -> int:
    return sum(_tensor_nbytes(tensor) for tensor in tensors)


def validate_resident_window_plan(
    logical_generations: int,
    window_generations: int,
    chunk_generations: Sequence[int],
    *,
    batch_cells: int,
    ticks_per_generation: int,
) -> dict[str, Any]:
    """Validate an exact logical run decomposed into bounded resident windows."""

    logical = _exact_int(
        logical_generations, path="logical_generations", minimum=1
    )
    window = _exact_int(
        window_generations, path="window_generations", minimum=1
    )
    cells = _exact_int(batch_cells, path="batch_cells", minimum=1)
    ticks = _exact_int(
        ticks_per_generation, path="ticks_per_generation", minimum=1
    )
    if not isinstance(chunk_generations, Sequence) or isinstance(
        chunk_generations, (str, bytes, bytearray)
    ):
        raise ResidentWindowContractError(
            "chunk_generations must be a nonempty sequence"
        )
    if not chunk_generations:
        raise ResidentWindowContractError(
            "chunk_generations must be a nonempty sequence"
        )
    chunks = tuple(
        _exact_int(value, path=f"chunk_generations[{index}]", minimum=1)
        for index, value in enumerate(chunk_generations)
    )
    if any(value > window for value in chunks):
        raise ResidentWindowContractError(
            "every chunk generation count must fit in window_generations"
        )
    if sum(chunks) != logical:
        raise ResidentWindowContractError(
            "chunk_generations must sum exactly to logical_generations"
        )
    offsets: list[dict[str, int]] = []
    cursor = 0
    for index, chunk in enumerate(chunks):
        offsets.append(
            {
                "chunk_index": index,
                "generation_start": cursor,
                "generation_end": cursor + chunk,
                "tick_start": cursor * ticks,
                "tick_end": (cursor + chunk) * ticks,
            }
        )
        cursor += chunk
    return {
        "schema_version": RESIDENT_WINDOW_RUNTIME_SCHEMA_VERSION,
        "operation": "validate_resident_window_plan",
        "logical_generations": logical,
        "window_generations": window,
        "chunk_generations": list(chunks),
        "chunk_count": len(chunks),
        "batch_cells": cells,
        "ticks_per_generation": ticks,
        "logical_ticks_per_cell": logical * ticks,
        "window_tick_capacity_per_cell": window * ticks,
        "logical_cell_generations": logical * cells,
        "chunk_intervals": offsets,
        "exact_coverage": True,
    }


def _validate_cpython_state_tensors(
    scalar_rng_state: Mapping[str, torch.Tensor],
) -> tuple[int, torch.device, dict[str, torch.Tensor]]:
    if not isinstance(scalar_rng_state, Mapping):
        raise ResidentWindowContractError("scalar_rng_state must be a mapping")
    words = scalar_rng_state.get("python_random_words")
    if not isinstance(words, torch.Tensor) or words.ndim != 2:
        raise ResidentWindowContractError(
            "python_random_words must be batched rank-2 uint32"
        )
    cells = int(words.shape[0])
    if cells < 1:
        raise ResidentWindowContractError(
            "python_random_words must contain at least one cell"
        )
    device = words.device
    validated: dict[str, torch.Tensor] = {
        "python_random_words": _mapping_tensor(
            scalar_rng_state,
            "python_random_words",
            dtype=torch.uint32,
            shape=(cells, 624),
            device=device,
        )
    }
    for name, dtype in _CPYTHON_STATE_FIELDS[1:]:
        validated[name] = _mapping_tensor(
            scalar_rng_state,
            name,
            dtype=dtype,
            shape=(cells,),
            device=device,
        )
    indices = _counter_values(
        validated["python_random_index"],
        path="python_random_index",
    )
    if any(index > 624 for index in indices):
        raise ResidentWindowContractError(
            "python_random_index values must be <= 624"
        )
    gauss_next = _host_copy(validated["python_random_gauss_next"])
    if not bool(torch.isfinite(gauss_next).all().item()):
        raise ResidentWindowContractError(
            "python_random_gauss_next values must be finite"
        )
    return cells, device, validated


def reconstruct_cpython_random_states(
    scalar_rng_state: Mapping[str, torch.Tensor],
) -> tuple[tuple[int, tuple[int, ...], float | None], ...]:
    """Reconstruct exact ``random.Random`` version-3 states for every cell."""

    cells, _, tensors = _validate_cpython_state_tensors(scalar_rng_state)
    words = _host_copy(tensors["python_random_words"])
    indices = _host_copy(tensors["python_random_index"])
    has_gauss = _host_copy(tensors["python_random_has_gauss"])
    gauss_next = _host_copy(tensors["python_random_gauss_next"])
    states: list[tuple[int, tuple[int, ...], float | None]] = []
    for cell in range(cells):
        internal = tuple(int(value) for value in words[cell].tolist()) + (
            int(indices[cell].item()),
        )
        cached = (
            float(gauss_next[cell].item())
            if bool(has_gauss[cell].item())
            else None
        )
        state = (3, internal, cached)
        try:
            validator = random.Random()
            validator.setstate(state)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ResidentWindowContractError(
                f"cell {cell} CPython random state is invalid"
            ) from exc
        states.append(state)
    return tuple(states)


def refill_cpython_gaussian_window_(
    scalar_rng_state: Mapping[str, torch.Tensor],
    *,
    python_random_window_base: torch.Tensor,
) -> dict[str, Any]:
    """Refill Gaussian tapes in place without changing absolute RNG counters."""

    cells, device, state_tensors = _validate_cpython_state_tensors(
        scalar_rng_state
    )
    primary = scalar_rng_state.get("python_random_gauss_primary_tape")
    if not isinstance(primary, torch.Tensor) or primary.ndim != 2:
        raise ResidentWindowContractError(
            "python_random_gauss_primary_tape must be batched rank-2 float64"
        )
    capacity = int(primary.shape[1])
    if capacity < 1:
        raise ResidentWindowContractError(
            "CPython Gaussian physical capacity must be positive"
        )
    window_tensors = {
        name: _mapping_tensor(
            scalar_rng_state,
            name,
            dtype=dtype,
            shape=(cells, capacity),
            device=device,
        )
        for name, dtype in _CPYTHON_WINDOW_FIELDS
    }
    absolute_raw = _mapping_tensor(
        scalar_rng_state,
        "python_random_raw_draws",
        dtype=torch.int64,
        shape=(cells,),
        device=device,
    )
    semantic = _mapping_tensor(
        scalar_rng_state,
        "mutation_py_semantic_draws",
        dtype=torch.int64,
        shape=(cells,),
        device=device,
    )
    base = _require_tensor(
        python_random_window_base,
        path="python_random_window_base",
        dtype=torch.int64,
        shape=(cells,),
        device=device,
    )
    absolute_before = _counter_values(
        absolute_raw, path="python_random_raw_draws"
    )
    semantic_before = _counter_values(
        semantic, path="mutation_py_semantic_draws"
    )
    base_before = _counter_values(base, path="python_random_window_base")
    if any(old > absolute for old, absolute in zip(base_before, absolute_before)):
        raise ResidentWindowContractError(
            "python_random_window_base must not exceed the absolute raw-draw counter"
        )

    states = reconstruct_cpython_random_states(scalar_rng_state)
    staged_primary: list[torch.Tensor] = []
    staged_secondary: list[torch.Tensor] = []
    for cell, state in enumerate(states):
        try:
            prepared = pack_cpython_random_state_cuda(
                state,
                device="cpu",
                semantic_draws=semantic_before[cell],
                gauss_tape_raw_capacity=capacity,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ResidentWindowContractError(
                f"cell {cell} CPython Gaussian window preparation failed"
            ) from exc
        staged_primary.append(
            prepared["python_random_gauss_primary_tape"].contiguous()
        )
        staged_secondary.append(
            prepared["python_random_gauss_secondary_tape"].contiguous()
        )
    primary_host = torch.stack(staged_primary)
    secondary_host = torch.stack(staged_secondary)

    mutable = {**window_tensors, "python_random_window_base": base}
    before_pointers = _pointer_snapshot(mutable)
    absolute_snapshot = absolute_raw.clone()
    semantic_snapshot = semantic.clone()
    with torch.no_grad():
        window_tensors["python_random_gauss_primary_tape"].copy_(primary_host)
        window_tensors["python_random_gauss_secondary_tape"].copy_(secondary_host)
        base.copy_(absolute_raw)
    if not torch.equal(absolute_raw, absolute_snapshot):
        raise RuntimeError("absolute CPython raw-draw counter changed during refill")
    if not torch.equal(semantic, semantic_snapshot):
        raise RuntimeError("CPython semantic counter changed during refill")
    after_pointers = _pointer_snapshot(mutable)

    absolute_after = _counter_values(
        absolute_raw, path="python_random_raw_draws"
    )
    semantic_after = _counter_values(
        semantic, path="mutation_py_semantic_draws"
    )
    base_after = _counter_values(base, path="python_random_window_base")
    state_bytes = _transfer_bytes(tuple(state_tensors.values()))
    counter_bytes = (
        _tensor_nbytes(absolute_raw)
        + _tensor_nbytes(semantic)
        + _tensor_nbytes(base)
    )
    # Validation reads index/gauss once; reconstruction reads the full state;
    # before/after receipts each read all three absolute/base counters.
    device_read_bytes = (
        state_bytes
        + _tensor_nbytes(state_tensors["python_random_index"])
        + _tensor_nbytes(state_tensors["python_random_gauss_next"])
        + 2 * counter_bytes
    )
    host_prepared_bytes = _tensor_nbytes(primary_host) + _tensor_nbytes(
        secondary_host
    )
    resident_written_bytes = host_prepared_bytes + _tensor_nbytes(base)
    return {
        "schema_version": RESIDENT_WINDOW_RUNTIME_SCHEMA_VERSION,
        "operation": "refill_cpython_gaussian_window_",
        "batch_cells": cells,
        "physical_capacity": capacity,
        "absolute_raw_draws_before": absolute_before,
        "absolute_raw_draws_after": absolute_after,
        "window_base_before": base_before,
        "window_base_after": base_after,
        "window_base_delta": [
            after - before
            for before, after in zip(base_before, base_after, strict=True)
        ],
        "semantic_draws_before": semantic_before,
        "semantic_draws_after": semantic_after,
        "absolute_raw_draws_preserved": absolute_after == absolute_before,
        "semantic_draws_preserved": semantic_after == semantic_before,
        "host_state_bytes": state_bytes + counter_bytes,
        "host_prepared_bytes": host_prepared_bytes,
        "resident_window_bytes": _transfer_bytes(tuple(mutable.values())),
        "resident_bytes_written": resident_written_bytes,
        "device_to_host_bytes": (
            device_read_bytes if device.type != "cpu" else 0
        ),
        "host_to_device_bytes": (
            resident_written_bytes if device.type != "cpu" else 0
        ),
        "pointer_identity": _pointer_receipt(
            before_pointers, after_pointers
        ),
    }


def _validate_policy_parameters(
    *,
    policy_weight_mutation_enable: bool,
    policy_weight_mutation_mode: str,
    policy_weight_mutation_rate: float,
    policy_weight_mutation_std: float,
) -> tuple[bool, str, float, float]:
    if type(policy_weight_mutation_enable) is not bool:
        raise ResidentWindowContractError(
            "policy_weight_mutation_enable must be an exact bool"
        )
    if type(policy_weight_mutation_mode) is not str:
        raise ResidentWindowContractError(
            "policy_weight_mutation_mode must be a string"
        )
    mode = policy_weight_mutation_mode.lower()
    if mode in {"", "gaussian"}:
        mode = "dense"
    if mode not in {"dense", "sparse"}:
        raise ResidentWindowContractError(
            "policy_weight_mutation_mode must be dense or sparse"
        )
    for path, value in (
        ("policy_weight_mutation_rate", policy_weight_mutation_rate),
        ("policy_weight_mutation_std", policy_weight_mutation_std),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ResidentWindowContractError(f"{path} must be numeric")
        if not math.isfinite(float(value)):
            raise ResidentWindowContractError(f"{path} must be finite")
    rate = float(policy_weight_mutation_rate)
    std = float(policy_weight_mutation_std)
    if not 0.0 <= rate <= 1.0:
        raise ResidentWindowContractError(
            "policy_weight_mutation_rate must be in [0, 1]"
        )
    if std < 0.0:
        raise ResidentWindowContractError(
            "policy_weight_mutation_std must be nonnegative"
        )
    return policy_weight_mutation_enable, mode, rate, std


def refill_fixed_mlp_policy_window_(
    policy_state: Mapping[str, torch.Tensor],
    *,
    policy_birth_window_base: torch.Tensor,
    policy_weight_mutation_enable: bool,
    policy_row_width: int | None = None,
    policy_weight_mutation_mode: str = "dense",
    policy_weight_mutation_rate: float = 1.0,
    policy_weight_mutation_std: float = 0.0,
) -> dict[str, Any]:
    """Refill fixed-MLP tapes in place while preserving absolute counters."""

    if not isinstance(policy_state, Mapping):
        raise ResidentWindowContractError("policy_state must be a mapping")
    enabled, mode, rate, std = _validate_policy_parameters(
        policy_weight_mutation_enable=policy_weight_mutation_enable,
        policy_weight_mutation_mode=policy_weight_mutation_mode,
        policy_weight_mutation_rate=policy_weight_mutation_rate,
        policy_weight_mutation_std=policy_weight_mutation_std,
    )
    source = policy_state.get("rng_manager_mutation_torch_cpu_state")
    delta = policy_state.get("policy_birth_delta_tape")
    if "policy_birth_post_rng_state_tape" in policy_state:
        raise ResidentWindowContractError(
            "legacy dense policy_birth_post_rng_state_tape cannot coexist "
            "with packed Torch RNG post-state schema"
        )
    rng_state_encoding = policy_state.get("policy_birth_post_rng_state_encoding")
    if isinstance(rng_state_encoding, torch.Tensor):
        raise ResidentWindowContractError(
            "policy_birth_post_rng_state_encoding must be route metadata, not a tensor"
        )
    if (
        rng_state_encoding is not None
        and rng_state_encoding != TORCH_CPU_RNG_STATE_ENCODING_V1
    ):
        raise ResidentWindowContractError(
            "unsupported policy_birth_post_rng_state_encoding"
        )
    if "policy_birth_crossover_mask_tape" in policy_state:
        raise ResidentWindowContractError(
            "legacy dense policy_birth_crossover_mask_tape cannot coexist "
            "with packed policy crossover schema"
        )
    encoding = policy_state.get("policy_birth_crossover_mask_encoding")
    if isinstance(encoding, torch.Tensor):
        raise ResidentWindowContractError(
            "policy_birth_crossover_mask_encoding must be route metadata, not a tensor"
        )
    if encoding is not None and encoding != CROSSOVER_MASK_ENCODING_V1:
        raise ResidentWindowContractError(
            "unsupported policy_birth_crossover_mask_encoding"
        )
    if not isinstance(source, torch.Tensor) or source.ndim != 2:
        raise ResidentWindowContractError(
            "rng_manager_mutation_torch_cpu_state must be batched rank-2 uint8"
        )
    if not isinstance(delta, torch.Tensor) or delta.ndim != 3:
        raise ResidentWindowContractError(
            "policy_birth_delta_tape must be batched rank-3 float32"
        )
    cells, state_bytes = (int(value) for value in source.shape)
    tape_cells, episodes, mutation_width = (int(value) for value in delta.shape)
    crossover = policy_state.get("policy_birth_crossover_mask_packed_tape")
    if not isinstance(crossover, torch.Tensor) or crossover.ndim != 3:
        raise ResidentWindowContractError(
            "policy_birth_crossover_mask_packed_tape must be batched rank-3 uint8"
        )
    crossover_cells, crossover_episodes, packed_row_bytes = (
        int(value) for value in crossover.shape
    )
    if policy_row_width is None:
        metadata_width = policy_state.get("policy_birth_crossover_mask_logical_width")
        if isinstance(metadata_width, torch.Tensor):
            if metadata_width.numel() != 1:
                raise ResidentWindowContractError(
                    "policy_birth_crossover_mask_logical_width must be scalar"
                )
            policy_row_width = int(_host_copy(metadata_width.reshape(1))[0].item())
        elif metadata_width is not None:
            policy_row_width = int(metadata_width)
    row_width = _exact_int(
        policy_row_width,
        path="policy_row_width",
        minimum=1,
    )
    if (
        cells < 1
        or episodes < 1
        or mutation_width < 1
        or packed_row_bytes < 1
    ):
        raise ResidentWindowContractError(
            "policy resident dimensions must all be positive"
        )
    if state_bytes != TORCH_CPU_RNG_STATE_FULL_BYTES_V1:
        raise ResidentWindowContractError(
            "rng_manager_mutation_torch_cpu_state has an unsupported Torch CPU RNG layout"
        )
    if tape_cells != cells or crossover_cells != cells:
        raise ResidentWindowContractError(
            "policy source-state and tape cell counts disagree"
        )
    if crossover_episodes != episodes:
        raise ResidentWindowContractError(
            "policy crossover and mutation tape capacities disagree"
        )
    if packed_row_bytes != packed_crossover_row_bytes_v1(row_width):
        raise ResidentWindowContractError(
            "packed policy crossover row bytes do not match policy_row_width"
        )
    if mutation_width not in (1, row_width):
        raise ResidentWindowContractError(
            "policy mutation tape width must be compact width 1 or full policy width"
        )
    if mutation_width == 1 and enabled:
        raise ResidentWindowContractError(
            "compact policy mutation tape requires disabled policy mutation"
        )
    device = source.device
    source = _mapping_tensor(
        policy_state,
        "rng_manager_mutation_torch_cpu_state",
        dtype=torch.uint8,
        shape=(cells, state_bytes),
        device=device,
    )
    replay_post_state_rows = mutation_width == 1
    expected_shapes = {
        "policy_birth_delta_tape": (cells, episodes, mutation_width),
        "policy_birth_mutation_mask_tape": (cells, episodes, mutation_width),
        "policy_birth_crossover_mask_packed_tape": (
            cells,
            episodes,
            packed_row_bytes,
        ),
        "policy_birth_post_rng_state_packed_tape": (
            (cells, episodes, 0)
            if replay_post_state_rows
            else (cells, episodes, TORCH_CPU_RNG_STATE_PACKED_BYTES_V1)
        ),
        "policy_birth_semantic_draws_tape": (cells, episodes),
    }
    tape_tensors = {
        name: _mapping_tensor(
            policy_state,
            name,
            dtype=dtype,
            shape=expected_shapes[name],
            device=device,
        )
        for name, dtype in _POLICY_TAPE_FIELDS
    }
    absolute_cursor = _mapping_tensor(
        policy_state,
        "policy_birth_tape_cursor",
        dtype=torch.int64,
        shape=(cells,),
        device=device,
    )
    semantic = _mapping_tensor(
        policy_state,
        "mutation_torch_cpu_semantic_draws",
        dtype=torch.int64,
        shape=(cells,),
        device=device,
    )
    base = _require_tensor(
        policy_birth_window_base,
        path="policy_birth_window_base",
        dtype=torch.int64,
        shape=(cells,),
        device=device,
    )
    cursor_before = _counter_values(
        absolute_cursor, path="policy_birth_tape_cursor"
    )
    semantic_before = _counter_values(
        semantic, path="mutation_torch_cpu_semantic_draws"
    )
    base_before = _counter_values(base, path="policy_birth_window_base")
    if any(old > absolute for old, absolute in zip(base_before, cursor_before)):
        raise ResidentWindowContractError(
            "policy_birth_window_base must not exceed the absolute tape cursor"
        )

    source_host = _host_copy(source)
    staged: dict[str, list[torch.Tensor]] = {
        name: [] for name, _ in _POLICY_TAPE_FIELDS
    }
    for cell in range(cells):
        # CPUGeneratorImpl requires a zero-offset state buffer. A row view can
        # report contiguous while retaining a nonzero storage offset.
        cell_source = source_host[cell].clone().contiguous()
        try:
            prepared = prepare_fixed_mlp_turnover_child_policy_tape(
                cell_source,
                episode_capacity=episodes,
                row_width=row_width,
                device="cpu",
                policy_weight_mutation_enable=enabled,
                policy_weight_mutation_mode=mode,
                policy_weight_mutation_rate=rate,
                policy_weight_mutation_std=std,
                semantic_draws=semantic_before[cell],
                compact_no_mutation_tape=mutation_width == 1,
                include_post_state_tape=not replay_post_state_rows,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ResidentWindowContractError(
                f"cell {cell} fixed-MLP policy window preparation failed"
            ) from exc
        if not torch.equal(
            prepared["rng_manager_mutation_torch_cpu_state"], cell_source
        ):
            raise RuntimeError(
                "policy preparation changed the authoritative source RNG state"
            )
        for name, dtype in _POLICY_TAPE_FIELDS:
            tensor = prepared.get(name)
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"policy preparation omitted {name}")
            expected_shape = expected_shapes[name][1:]
            if tensor.dtype != dtype or tuple(tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"policy preparation returned a contradictory {name}"
                )
            staged[name].append(tensor.contiguous())
    staged_tensors = {
        name: torch.stack(rows) for name, rows in staged.items()
    }

    mutable = {**tape_tensors, "policy_birth_window_base": base}
    before_pointers = _pointer_snapshot(mutable)
    source_snapshot = source.clone()
    cursor_snapshot = absolute_cursor.clone()
    semantic_snapshot = semantic.clone()
    with torch.no_grad():
        for name, target in tape_tensors.items():
            target.copy_(staged_tensors[name])
        base.copy_(absolute_cursor)
    if not torch.equal(source, source_snapshot):
        raise RuntimeError("authoritative Torch CPU RNG state changed during refill")
    if not torch.equal(absolute_cursor, cursor_snapshot):
        raise RuntimeError("absolute policy tape cursor changed during refill")
    if not torch.equal(semantic, semantic_snapshot):
        raise RuntimeError("absolute policy semantic counter changed during refill")
    after_pointers = _pointer_snapshot(mutable)

    cursor_after = _counter_values(
        absolute_cursor, path="policy_birth_tape_cursor"
    )
    semantic_after = _counter_values(
        semantic, path="mutation_torch_cpu_semantic_draws"
    )
    base_after = _counter_values(base, path="policy_birth_window_base")
    source_bytes = _tensor_nbytes(source)
    counter_read_bytes = (
        _tensor_nbytes(absolute_cursor)
        + _tensor_nbytes(semantic)
        + _tensor_nbytes(base)
    )
    host_prepared_bytes = _transfer_bytes(tuple(staged_tensors.values()))
    resident_written_bytes = host_prepared_bytes + _tensor_nbytes(base)
    return {
        "schema_version": RESIDENT_WINDOW_RUNTIME_SCHEMA_VERSION,
        "operation": "refill_fixed_mlp_policy_window_",
        "batch_cells": cells,
        "physical_capacity": episodes,
        "policy_row_width": row_width,
        "policy_crossover_mask_encoding": CROSSOVER_MASK_ENCODING_V1,
        "policy_crossover_packed_row_bytes": packed_row_bytes,
        "policy_crossover_logical_shape": [cells, episodes, row_width],
        "policy_crossover_physical_shape": [cells, episodes, packed_row_bytes],
        "policy_crossover_logical_dense_bytes": cells * episodes * row_width,
        "policy_crossover_physical_bytes": cells * episodes * packed_row_bytes,
        "policy_crossover_compression_ratio": (
            (cells * episodes * packed_row_bytes) / (cells * episodes * row_width)
        ),
        "policy_crossover_dense_allocation_materialized": False,
        "policy_crossover_max_logical_temp_bytes": row_width,
        "torch_rng_state_bytes_per_cell": state_bytes,
        "torch_rng_post_state_encoding": TORCH_CPU_RNG_STATE_ENCODING_V1,
        "torch_rng_post_state_logical_row_bytes": state_bytes,
        "torch_rng_post_state_packed_row_bytes": (
            TORCH_CPU_RNG_STATE_PACKED_BYTES_V1
        ),
        "torch_rng_post_state_dense_allocation_materialized": False,
        "torch_rng_post_state_resident_rows_materialized": (
            not replay_post_state_rows
        ),
        "torch_rng_post_state_replay_schema_version": (
            "eiel_torch_cpu_mt19937_replay_v1"
            if replay_post_state_rows
            else None
        ),
        "torch_rng_post_state_abi": torch_cpu_rng_state_abi_receipt_v1(),
        "absolute_cursor_before": cursor_before,
        "absolute_cursor_after": cursor_after,
        "window_base_before": base_before,
        "window_base_after": base_after,
        "window_base_delta": [
            after - before
            for before, after in zip(base_before, base_after, strict=True)
        ],
        "semantic_draws_before": semantic_before,
        "semantic_draws_after": semantic_after,
        "absolute_cursor_preserved": cursor_after == cursor_before,
        "semantic_draws_preserved": semantic_after == semantic_before,
        "source_rng_state_preserved": True,
        "host_state_bytes": source_bytes + counter_read_bytes,
        "host_prepared_bytes": host_prepared_bytes,
        "resident_window_bytes": _transfer_bytes(tuple(mutable.values())),
        "resident_bytes_written": resident_written_bytes,
        "device_to_host_bytes": (
            source_bytes + 2 * counter_read_bytes
            if device.type != "cpu"
            else 0
        ),
        "host_to_device_bytes": (
            resident_written_bytes if device.type != "cpu" else 0
        ),
        "pointer_identity": _pointer_receipt(
            before_pointers, after_pointers
        ),
    }


def _validate_event_bank(
    event_columns: Mapping[str, torch.Tensor],
    *,
    active_count: torch.Tensor,
    event_sequence_base: torch.Tensor,
    event_status: torch.Tensor,
) -> tuple[
    int,
    int,
    torch.device,
    dict[str, torch.Tensor],
    list[int],
    list[int],
]:
    if not isinstance(event_columns, Mapping):
        raise ResidentWindowContractError("event_columns must be a mapping")
    first = event_columns.get(EVENT_COLUMN_NAMES[0])
    if not isinstance(first, torch.Tensor) or first.ndim != 2:
        raise ResidentWindowContractError(
            f"{EVENT_COLUMN_NAMES[0]} must be batched rank 2"
        )
    cells, capacity = (int(value) for value in first.shape)
    if cells < 1 or capacity < 1:
        raise ResidentWindowContractError(
            "event window cell count and physical capacity must be positive"
        )
    device = first.device
    columns = {
        name: _mapping_tensor(
            event_columns,
            name,
            dtype=EVENT_COLUMN_DTYPES[name],
            shape=(cells, capacity),
            device=device,
        )
        for name in EVENT_COLUMN_NAMES
    }
    active = _require_tensor(
        active_count,
        path="active_count",
        dtype=torch.int64,
        shape=(cells,),
        device=device,
    )
    base = _require_tensor(
        event_sequence_base,
        path="event_sequence_base",
        dtype=torch.int64,
        shape=(cells,),
        device=device,
    )
    status = _require_tensor(
        event_status,
        path="event_status",
        dtype=torch.int32,
        shape=(cells,),
        device=device,
    )
    counts = _counter_values(active, path="active_count")
    bases = _counter_values(base, path="event_sequence_base")
    if any(count > capacity for count in counts):
        raise ResidentWindowContractError(
            "active_count must not exceed event physical capacity"
        )
    statuses = _host_ints(status)
    if any(value != 0 for value in statuses):
        raise ResidentWindowContractError(
            "event_status must be zero before draining or materializing"
        )
    sequence = _host_copy(columns["typed_lifecycle_event_sequence"])
    for cell, (count, base_value) in enumerate(zip(counts, bases, strict=True)):
        expected = torch.arange(
            base_value,
            base_value + count,
            dtype=torch.int64,
        )
        if not torch.equal(sequence[cell, :count], expected):
            raise ResidentWindowContractError(
                f"cell {cell} active event sequence must equal "
                "event_sequence_base + physical row index"
            )
    return cells, capacity, device, columns, counts, bases


def _validate_event_archives(
    archives: Sequence[EventArchive],
    *,
    cells: int,
    expected_bases: Sequence[int],
) -> None:
    if not isinstance(archives, Sequence) or isinstance(
        archives, (str, bytes, bytearray)
    ):
        raise ResidentWindowContractError("archives must be a per-cell sequence")
    if len(archives) != cells:
        raise ResidentWindowContractError(
            "archives length must equal the resident cell count"
        )
    for cell, (archive, expected_base) in enumerate(
        zip(archives, expected_bases, strict=True)
    ):
        if not isinstance(archive, MutableSequence):
            raise ResidentWindowContractError(
                f"archives[{cell}] must be an append-only mutable sequence"
            )
        cursor = 0
        for chunk_index, chunk in enumerate(archive):
            if not isinstance(chunk, Mapping) or set(chunk) != set(
                EVENT_COLUMN_NAMES
            ):
                raise ResidentWindowContractError(
                    f"archives[{cell}][{chunk_index}] has an invalid event schema"
                )
            sequence = chunk.get("typed_lifecycle_event_sequence")
            if not isinstance(sequence, torch.Tensor) or sequence.ndim != 1:
                raise ResidentWindowContractError(
                    f"archives[{cell}][{chunk_index}] sequence must be rank 1"
                )
            rows = int(sequence.numel())
            for name in EVENT_COLUMN_NAMES:
                _require_tensor(
                    chunk.get(name),
                    path=f"archives[{cell}][{chunk_index}].{name}",
                    dtype=EVENT_COLUMN_DTYPES[name],
                    shape=(rows,),
                    device=torch.device("cpu"),
                )
            expected = torch.arange(cursor, cursor + rows, dtype=torch.int64)
            if not torch.equal(sequence, expected):
                raise ResidentWindowContractError(
                    f"archives[{cell}][{chunk_index}] sequence is not contiguous"
                )
            cursor += rows
        if cursor != expected_base:
            raise ResidentWindowContractError(
                f"archives[{cell}] row count disagrees with event_sequence_base"
            )


def _stage_active_events(
    columns: Mapping[str, torch.Tensor], counts: Sequence[int]
) -> tuple[list[EventChunk], int]:
    staged: list[EventChunk] = []
    total_bytes = 0
    for cell, count in enumerate(counts):
        chunk = {
            name: _host_copy(tensor[cell, :count]).clone()
            for name, tensor in columns.items()
        }
        total_bytes += _transfer_bytes(tuple(chunk.values()))
        staged.append(chunk)
    return staged, total_bytes


def drain_event_window(
    event_columns: Mapping[str, torch.Tensor],
    *,
    active_count: torch.Tensor,
    event_sequence_base: torch.Tensor,
    event_status: torch.Tensor,
    archives: Sequence[EventArchive],
) -> dict[str, Any]:
    """Drain active event rows to CPU archives and reset only physical counts."""

    cells, capacity, device, columns, counts, bases = _validate_event_bank(
        event_columns,
        active_count=active_count,
        event_sequence_base=event_sequence_base,
        event_status=event_status,
    )
    _validate_event_archives(
        archives, cells=cells, expected_bases=bases
    )
    staged, copied_bytes = _stage_active_events(columns, counts)
    mutable = {
        **columns,
        "active_count": active_count,
        "event_sequence_base": event_sequence_base,
        "event_status": event_status,
    }
    before_pointers = _pointer_snapshot(mutable)
    for cell, chunk in enumerate(staged):
        if counts[cell] > 0:
            archives[cell].append(chunk)
    base_after = [base + count for base, count in zip(bases, counts, strict=True)]
    with torch.no_grad():
        event_sequence_base.copy_(
            torch.tensor(base_after, dtype=torch.int64, device=device)
        )
        active_count.zero_()
    after_pointers = _pointer_snapshot(mutable)
    return {
        "schema_version": RESIDENT_WINDOW_RUNTIME_SCHEMA_VERSION,
        "operation": "drain_event_window",
        "batch_cells": cells,
        "physical_capacity": capacity,
        "active_count_before": list(counts),
        "active_count_after": [0] * cells,
        "event_sequence_base_before": list(bases),
        "event_sequence_base_after": base_after,
        "drained_rows_by_cell": list(counts),
        "drained_rows_total": sum(counts),
        "host_archive_bytes_appended": copied_bytes,
        "device_to_host_bytes": (
            copied_bytes
            + cells * (8 + 8 + 4)
            + sum(counts) * 8
            if device.type != "cpu"
            else 0
        ),
        "host_to_device_bytes": (
            _tensor_nbytes(event_sequence_base) if device.type != "cpu" else 0
        ),
        "resident_bytes_mutated": (
            _tensor_nbytes(event_sequence_base) + _tensor_nbytes(active_count)
        ),
        "resident_event_window_bytes": _transfer_bytes(tuple(mutable.values())),
        "pointer_identity": _pointer_receipt(
            before_pointers, after_pointers
        ),
    }


def materialize_event_history(
    archives: Sequence[EventArchive],
    event_columns: Mapping[str, torch.Tensor],
    *,
    active_count: torch.Tensor,
    event_sequence_base: torch.Tensor,
    event_status: torch.Tensor,
) -> tuple[tuple[dict[str, torch.Tensor], ...], dict[str, Any]]:
    """Materialize exact archived plus active rows for a final checkpoint."""

    cells, capacity, device, columns, counts, bases = _validate_event_bank(
        event_columns,
        active_count=active_count,
        event_sequence_base=event_sequence_base,
        event_status=event_status,
    )
    _validate_event_archives(
        archives, cells=cells, expected_bases=bases
    )
    active_chunks, copied_bytes = _stage_active_events(columns, counts)
    pointers = _pointer_snapshot(
        {
            **columns,
            "active_count": active_count,
            "event_sequence_base": event_sequence_base,
            "event_status": event_status,
        }
    )
    histories: list[dict[str, torch.Tensor]] = []
    rows_by_cell: list[int] = []
    host_output_bytes = 0
    for cell in range(cells):
        chunks = [*archives[cell]]
        if counts[cell] > 0:
            chunks.append(active_chunks[cell])
        history: dict[str, torch.Tensor] = {}
        total_rows = bases[cell] + counts[cell]
        for name in EVENT_COLUMN_NAMES:
            if chunks:
                value = torch.cat([chunk[name] for chunk in chunks]).contiguous()
            else:
                value = torch.empty(
                    (0,), dtype=EVENT_COLUMN_DTYPES[name], device="cpu"
                )
            if value.dtype != EVENT_COLUMN_DTYPES[name] or tuple(value.shape) != (
                total_rows,
            ):
                raise RuntimeError(
                    f"materialized cell {cell} event column {name} is contradictory"
                )
            history[name] = value
            host_output_bytes += _tensor_nbytes(value)
        expected_sequence = torch.arange(total_rows, dtype=torch.int64)
        if not torch.equal(
            history["typed_lifecycle_event_sequence"], expected_sequence
        ):
            raise RuntimeError(
                f"materialized cell {cell} event history is not sequence exact"
            )
        histories.append(history)
        rows_by_cell.append(total_rows)
    return tuple(histories), {
        "schema_version": RESIDENT_WINDOW_RUNTIME_SCHEMA_VERSION,
        "operation": "materialize_event_history",
        "batch_cells": cells,
        "physical_capacity": capacity,
        "archived_rows_by_cell": list(bases),
        "active_rows_by_cell": list(counts),
        "materialized_rows_by_cell": rows_by_cell,
        "host_active_bytes_read": copied_bytes,
        "host_output_bytes": host_output_bytes,
        "device_to_host_bytes": (
            copied_bytes
            + cells * (8 + 8 + 4)
            + sum(counts) * 8
            if device.type != "cpu"
            else 0
        ),
        "host_to_device_bytes": 0,
        "sequence_exact": True,
        "source_pointer_identity": _pointer_receipt(pointers, pointers),
    }


__all__ = [
    "EventArchive",
    "EventChunk",
    "RESIDENT_WINDOW_RUNTIME_SCHEMA_VERSION",
    "ResidentWindowContractError",
    "drain_event_window",
    "materialize_event_history",
    "reconstruct_cpython_random_states",
    "refill_cpython_gaussian_window_",
    "refill_fixed_mlp_policy_window_",
    "validate_resident_window_plan",
]
