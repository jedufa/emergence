from __future__ import annotations

"""Deterministic represented-payload accounting for Architecture B crossings."""

import json
import math
import operator
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA_VERSION = "eiel_architecture_b_crossing_payload_v1"


def _exact_integer(value: Any, *, label: str, path: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid {label} at {path}: expected an exact integer")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(
            f"invalid {label} at {path}: expected an exact integer"
        ) from exc


def _validate_view_byte_bounds(
    *,
    path: str,
    label: str,
    byte_offset: int,
    shape: Sequence[int],
    byte_strides: Sequence[int],
    itemsize: int,
    storage_nbytes: int,
) -> None:
    if itemsize <= 0 or storage_nbytes < 0 or byte_offset < 0:
        raise ValueError(f"invalid {label} at {path}: unrepresentable byte bounds")
    if len(shape) != len(byte_strides) or any(dimension < 0 for dimension in shape):
        raise ValueError(f"invalid {label} at {path}: malformed shape or strides")

    if any(dimension == 0 for dimension in shape):
        if byte_offset > storage_nbytes:
            raise ValueError(f"invalid {label} at {path}: offset exceeds storage")
        return

    lowest_byte = byte_offset
    highest_byte_exclusive = byte_offset + itemsize
    for dimension, stride in zip(shape, byte_strides, strict=True):
        extent = (dimension - 1) * stride
        lowest_byte += min(0, extent)
        highest_byte_exclusive += max(0, extent)
    if lowest_byte < 0 or highest_byte_exclusive > storage_nbytes:
        raise ValueError(f"invalid {label} at {path}: view exceeds backing storage")


def _numpy_backing_array(value: np.ndarray, *, path: str) -> np.ndarray:
    current: Any = value
    root = value
    seen: set[int] = set()
    while True:
        object_id = id(current)
        if object_id in seen:
            raise ValueError(f"invalid NumPy byte_offset at {path}: cyclic base chain")
        seen.add(object_id)
        base = getattr(current, "base", None)
        if base is None:
            break
        current = base
        if isinstance(current, np.ndarray):
            root = current

    if not (root.flags.c_contiguous or root.flags.f_contiguous):
        raise ValueError(
            f"invalid NumPy byte_offset at {path}: backing array is noncontiguous"
        )
    return root


def _numpy_data_pointer(value: np.ndarray, *, path: str) -> int:
    try:
        pointer = value.__array_interface__["data"][0]
    except (AttributeError, KeyError, TypeError, IndexError) as exc:
        raise ValueError(
            f"invalid NumPy byte_offset at {path}: data pointer is unavailable"
        ) from exc
    return _exact_integer(pointer, label="NumPy byte_offset", path=path)


def architecture_b_crossing_payload_metrics(
    run_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Account for canonical header bytes plus unique tensor backing storage.

    This is a stable representation metric for the in-process pybind crossing.
    It intentionally excludes CPython container/object heap overhead because
    that overhead has no portable serialized byte representation.
    """

    if not isinstance(run_result, Mapping):
        raise TypeError("run_result must be a mapping")

    storage_ids: dict[tuple[Any, ...], str] = {}
    storage_bytes: dict[str, int] = {}
    tensor_references = 0

    def storage_id(key: tuple[Any, ...], byte_count: int) -> str:
        if key not in storage_ids:
            identifier = f"storage-{len(storage_ids):06d}"
            storage_ids[key] = identifier
            storage_bytes[identifier] = int(byte_count)
        return storage_ids[key]

    def normalize(value: Any, path: str) -> Any:
        nonlocal tensor_references
        if isinstance(value, torch.Tensor):
            tensor_references += 1
            tensor = value.detach()
            storage = tensor.untyped_storage()
            byte_count = int(storage.nbytes())
            element_size = int(tensor.element_size())
            storage_offset = _exact_integer(
                tensor.storage_offset(),
                label="Torch storage_offset",
                path=path,
            )
            shape = [int(item) for item in tensor.shape]
            stride = [int(item) for item in tensor.stride()]
            _validate_view_byte_bounds(
                path=path,
                label="Torch storage_offset",
                byte_offset=storage_offset * element_size,
                shape=shape,
                byte_strides=[item * element_size for item in stride],
                itemsize=element_size,
                storage_nbytes=byte_count,
            )
            key = (
                "torch",
                str(tensor.device),
                int(storage.data_ptr()),
                byte_count,
            )
            identifier = storage_id(key, byte_count)
            return {
                "type": "torch.Tensor",
                "dtype": str(tensor.dtype),
                "shape": shape,
                "stride": stride,
                "storage_offset": storage_offset,
                "storage_id": identifier,
                "storage_nbytes": byte_count,
                "view_nbytes": int(tensor.numel() * element_size),
            }
        if isinstance(value, np.ndarray):
            tensor_references += 1
            root = _numpy_backing_array(value, path=path)
            byte_count = int(root.nbytes)
            byte_offset = _numpy_data_pointer(value, path=path) - _numpy_data_pointer(
                root,
                path=path,
            )
            shape = [int(item) for item in value.shape]
            strides = [int(item) for item in value.strides]
            _validate_view_byte_bounds(
                path=path,
                label="NumPy byte_offset",
                byte_offset=byte_offset,
                shape=shape,
                byte_strides=strides,
                itemsize=int(value.dtype.itemsize),
                storage_nbytes=byte_count,
            )
            key = ("numpy", id(root), byte_count)
            identifier = storage_id(key, byte_count)
            return {
                "type": "numpy.ndarray",
                "dtype": str(value.dtype),
                "shape": shape,
                "strides": strides,
                "byte_offset": byte_offset,
                "storage_id": identifier,
                "storage_nbytes": byte_count,
                "view_nbytes": int(value.nbytes),
            }
        if isinstance(value, np.generic):
            return normalize(value.item(), path)
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item, f"{path}.{key}")
                for key, item in sorted(value.items(), key=lambda row: str(row[0]))
            }
        if isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            return [normalize(item, f"{path}[{index}]") for index, item in enumerate(value)]
        if isinstance(value, (set, frozenset)):
            return [
                normalize(item, f"{path}[set]")
                for item in sorted(value, key=lambda item: repr(item))
            ]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, bytes):
            return {"type": "bytes", "hex": value.hex()}
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(f"non-finite float at {path}")
            return value
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        if any(hasattr(value, name) for name in ("shape", "dtype", "nbytes")):
            raise TypeError(
                f"unsupported tensor-like value at {path}: {type(value).__name__}"
            )
        raise TypeError(f"unsupported payload value at {path}: {type(value).__name__}")

    header = normalize(run_result, "$<run_result>")
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    unique_tensor_storage_bytes = sum(storage_bytes.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "representation": "canonical_json_header_plus_unique_tensor_storage",
        "canonical_non_tensor_header_bytes": len(header_bytes),
        "unique_tensor_storage_bytes": unique_tensor_storage_bytes,
        "represented_payload_bytes": len(header_bytes) + unique_tensor_storage_bytes,
        "tensor_reference_count": tensor_references,
        "unique_tensor_storage_count": len(storage_bytes),
        "unique_tensor_storages": [
            {"storage_id": identifier, "bytes": storage_bytes[identifier]}
            for identifier in sorted(storage_bytes)
        ],
        "python_heap_and_container_overhead_included": False,
        "caveat": (
            "In-process represented payload only; excludes CPython and pybind "
            "object/container heap overhead and allocator bookkeeping."
        ),
    }
