"""Torch storage helpers for the frozen compact generation-summary ABI v2.

This module owns allocation and bounded storage validation only. It does not
populate summaries, run inside the generation loop, or qualify a CUDA runtime
capability. NumPy conversion is an explicit final-boundary operation.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import torch

from engine.native_runtime import compact_summary_planes as host_planes


PLANE_NAMES: Final[tuple[str, ...]] = (
    "validity",
    "defined_mask",
    "status_codes",
    "integer_values",
    "float_values",
    "digest_values",
    "text_values",
)
PLANE_DTYPES: Final[Mapping[str, torch.dtype]] = MappingProxyType(
    {
        "validity": torch.uint8,
        "defined_mask": torch.uint64,
        "status_codes": torch.int32,
        "integer_values": torch.int64,
        "float_values": torch.float64,
        "digest_values": torch.uint8,
        "text_values": torch.uint8,
    }
)
PLANE_TRAILING_SHAPES: Final[Mapping[str, tuple[int, ...]]] = MappingProxyType(
    {
        "validity": (),
        "defined_mask": (host_planes.DEFINED_MASK_WORD_COUNT,),
        "status_codes": (len(host_planes.STATUS_FIELDS),),
        "integer_values": (len(host_planes.INTEGER_FIELDS),),
        "float_values": (len(host_planes.FLOAT_FIELDS),),
        "digest_values": (len(host_planes.DIGEST_FIELDS), 32),
        "text_values": (len(host_planes.TEXT_FIELDS), host_planes.TEXT_RECORD_BYTES),
    }
)
LOGICAL_BYTES_PER_GENERATION_CELL: Final[int] = (
    host_planes.LOGICAL_BYTES_PER_GENERATION_CELL
)


@dataclass(frozen=True)
class CompactSummaryTensors:
    """Exactly seven primitive Torch planes for compact ABI v2."""

    validity: torch.Tensor
    defined_mask: torch.Tensor
    status_codes: torch.Tensor
    integer_values: torch.Tensor
    float_values: torch.Tensor
    digest_values: torch.Tensor
    text_values: torch.Tensor

    def as_mapping(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(
            {
                "validity": self.validity,
                "defined_mask": self.defined_mask,
                "status_codes": self.status_codes,
                "integer_values": self.integer_values,
                "float_values": self.float_values,
                "digest_values": self.digest_values,
                "text_values": self.text_values,
            }
        )

    @property
    def nbytes(self) -> int:
        return compact_summary_tensor_nbytes(self)


def _positive_dimension(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    if value > sys.maxsize:
        raise OverflowError(f"{name} exceeds platform index range")
    return value


def _allocation_device(device: str | torch.device) -> torch.device:
    requested = _supported_device(device)
    if requested.type == "cpu":
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA compact summary tensors require available CUDA")
    index = requested.index
    if index is None:
        index = torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is unavailable: {index}")
    return torch.device("cuda", index)


def _supported_device(device: str | torch.device) -> torch.device:
    try:
        requested = torch.device(device)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"invalid Torch device: {device!r}") from exc
    if requested.type not in {"cpu", "cuda"}:
        raise ValueError("compact summary tensors require a CPU or CUDA device")
    return requested


def allocate_compact_summary_tensors(
    generation_count: int,
    cell_count: int,
    *,
    device: str | torch.device,
) -> CompactSummaryTensors:
    """Allocate exactly seven zeroed, contiguous ABI-v2 tensor planes."""

    generations = _positive_dimension(generation_count, name="generation_count")
    cells = _positive_dimension(cell_count, name="cell_count")
    target = _allocation_device(device)
    prefix = (generations, cells)
    tensors = {
        name: torch.zeros(
            prefix + PLANE_TRAILING_SHAPES[name],
            dtype=PLANE_DTYPES[name],
            device=target,
        )
        for name in PLANE_NAMES
    }
    return CompactSummaryTensors(**tensors)


def _plane_mapping(
    tensors: CompactSummaryTensors | Mapping[str, torch.Tensor],
) -> Mapping[str, torch.Tensor]:
    if isinstance(tensors, CompactSummaryTensors):
        return tensors.as_mapping()
    if not isinstance(tensors, Mapping):
        raise TypeError("tensors must be CompactSummaryTensors or a mapping")
    names = set(tensors)
    expected = set(PLANE_NAMES)
    if names != expected:
        missing = sorted(expected - names)
        extra = sorted(names - expected, key=repr)
        raise ValueError(f"plane names mismatch: missing={missing} extra={extra}")
    return tensors


def _matches_requested_device(actual: torch.device, requested: torch.device) -> bool:
    if actual.type != requested.type:
        return False
    return requested.index is None or actual.index == requested.index


def _device_any(value: torch.Tensor) -> bool:
    return bool(value.any().item())


def validate_compact_summary_tensors(
    tensors: CompactSummaryTensors | Mapping[str, torch.Tensor],
    *,
    generation_count: int | None = None,
    cell_count: int | None = None,
    device: str | torch.device | None = None,
) -> tuple[int, int]:
    """Validate the bounded Torch storage contract and return its row shape.

    Content checks execute device reductions and may synchronize in order to
    raise deterministic Python errors. This helper is not a generation-loop
    validator and makes no synchronization-free execution claim.
    """

    values = _plane_mapping(tensors)
    for name in PLANE_NAMES:
        if not isinstance(values[name], torch.Tensor):
            raise TypeError(f"{name} must be a Torch tensor")

    validity = values["validity"]
    if validity.ndim != 2:
        raise ValueError("validity must be a rank-2 Torch tensor")
    generations, cells = (int(size) for size in validity.shape)
    _positive_dimension(generations, name="generation_count")
    _positive_dimension(cells, name="cell_count")
    if generation_count is not None and generations != _positive_dimension(
        generation_count, name="generation_count"
    ):
        raise ValueError("generation_count does not match tensor shape")
    if cell_count is not None and cells != _positive_dimension(
        cell_count, name="cell_count"
    ):
        raise ValueError("cell_count does not match tensor shape")

    devices = {values[name].device for name in PLANE_NAMES}
    if len(devices) != 1:
        raise ValueError("all compact summary tensors must use the same device")
    actual_device = next(iter(devices))
    if actual_device.type not in {"cpu", "cuda"}:
        raise ValueError("compact summary tensors require a CPU or CUDA device")
    if device is not None:
        requested_device = _supported_device(device)
        if requested_device.type == "cpu":
            requested_device = torch.device("cpu")
        if not _matches_requested_device(actual_device, requested_device):
            raise ValueError(
                f"tensor device must be {requested_device}, got {actual_device}"
            )

    prefix = (generations, cells)
    for name in PLANE_NAMES:
        tensor = values[name]
        expected_shape = prefix + PLANE_TRAILING_SHAPES[name]
        if tensor.dtype != PLANE_DTYPES[name]:
            raise ValueError(
                f"{name} dtype must be {PLANE_DTYPES[name]}, got {tensor.dtype}"
            )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} shape must be {expected_shape}, got {tuple(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if _device_any((validity != 0) & (validity != 1)):
        raise ValueError("validity entries must be 0 or 1")
    if _device_any(~torch.isfinite(values["float_values"])):
        raise ValueError("float_values must be finite")

    invalid_rows = validity == 0
    if _device_any(invalid_rows):
        for name in PLANE_NAMES[1:]:
            raw_rows = values[name].view(torch.uint8).reshape(generations, cells, -1)
            dirty_rows = torch.any(raw_rows != 0, dim=-1)
            if _device_any(dirty_rows & invalid_rows):
                raise ValueError(
                    f"invalid rows must retain bit-zero no-op storage in {name}"
                )
    return generations, cells


def compact_summary_tensor_nbytes(
    tensors: CompactSummaryTensors | Mapping[str, torch.Tensor],
) -> int:
    """Return physical bytes occupied by the seven tensor planes."""

    values = _plane_mapping(tensors)
    total = 0
    for name in PLANE_NAMES:
        tensor = values[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a Torch tensor")
        total += tensor.numel() * tensor.element_size()
    return total


def to_numpy_compact_summary_planes(
    tensors: CompactSummaryTensors | Mapping[str, torch.Tensor],
) -> host_planes.CompactSummaryPlanes:
    """Explicitly copy Torch planes to NumPy at the final host boundary."""

    validate_compact_summary_tensors(tensors)
    values = _plane_mapping(tensors)
    arrays = {
        name: values[name]
        .detach()
        .to(device="cpu", copy=True)
        .contiguous()
        .numpy()
        for name in PLANE_NAMES
    }
    result = host_planes.CompactSummaryPlanes(**arrays)
    host_planes.validate_compact_summary_plane_layout(result)
    return result


__all__ = [
    "CompactSummaryTensors",
    "LOGICAL_BYTES_PER_GENERATION_CELL",
    "PLANE_DTYPES",
    "PLANE_NAMES",
    "PLANE_TRAILING_SHAPES",
    "allocate_compact_summary_tensors",
    "compact_summary_tensor_nbytes",
    "to_numpy_compact_summary_planes",
    "validate_compact_summary_tensors",
]
