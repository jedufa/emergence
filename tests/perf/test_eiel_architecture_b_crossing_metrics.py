from __future__ import annotations

import numpy as np
import pytest
import torch

from native_runtime.architecture_b_crossing_metrics import (
    architecture_b_crossing_payload_metrics,
)


def test_accounts_unique_torch_storage_once_for_aliases() -> None:
    tensor = torch.arange(12, dtype=torch.float64)
    result = architecture_b_crossing_payload_metrics(
        {"first": tensor, "alias": tensor, "view": tensor.reshape(3, 4)}
    )

    assert result["tensor_reference_count"] == 3
    assert result["unique_tensor_storage_count"] == 1
    assert result["unique_tensor_storage_bytes"] == tensor.untyped_storage().nbytes()
    assert result["represented_payload_bytes"] == (
        result["canonical_non_tensor_header_bytes"]
        + result["unique_tensor_storage_bytes"]
    )


def test_accounts_numpy_base_storage_once() -> None:
    array = np.arange(20, dtype=np.int32)
    result = architecture_b_crossing_payload_metrics(
        {"array": array, "view": array[2:10]}
    )

    assert result["tensor_reference_count"] == 2
    assert result["unique_tensor_storage_count"] == 1
    assert result["unique_tensor_storage_bytes"] == array.nbytes


def test_torch_view_offsets_change_header_metric_without_duplicate_storage() -> None:
    tensor = torch.arange(16, dtype=torch.float64)
    first = tensor[:3]
    shifted = tensor[10:13]

    same_offset = architecture_b_crossing_payload_metrics(
        {"first": first, "second": first}
    )
    different_offsets = architecture_b_crossing_payload_metrics(
        {"first": first, "second": shifted}
    )

    assert different_offsets["canonical_non_tensor_header_bytes"] != same_offset[
        "canonical_non_tensor_header_bytes"
    ]
    assert different_offsets["unique_tensor_storage_count"] == 1
    assert different_offsets["unique_tensor_storage_bytes"] == (
        tensor.untyped_storage().nbytes()
    )


def test_numpy_view_offsets_change_header_metric_without_duplicate_storage() -> None:
    array = np.arange(16, dtype=np.int32)
    first = array[:3]
    shifted = array[10:13]

    same_offset = architecture_b_crossing_payload_metrics(
        {"first": first, "second": first}
    )
    different_offsets = architecture_b_crossing_payload_metrics(
        {"first": first, "second": shifted}
    )

    assert different_offsets["canonical_non_tensor_header_bytes"] != same_offset[
        "canonical_non_tensor_header_bytes"
    ]
    assert different_offsets["unique_tensor_storage_count"] == 1
    assert different_offsets["unique_tensor_storage_bytes"] == array.nbytes


def test_header_size_is_deterministic_across_mapping_order() -> None:
    tensor = torch.tensor([1, 2, 3], dtype=torch.int64)
    left = architecture_b_crossing_payload_metrics({"z": 4, "a": tensor})
    right = architecture_b_crossing_payload_metrics({"a": tensor, "z": 4})

    assert left == right


def test_rejects_invalid_root_and_nonfinite_or_unknown_values() -> None:
    with pytest.raises(TypeError, match="mapping"):
        architecture_b_crossing_payload_metrics([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-finite"):
        architecture_b_crossing_payload_metrics({"value": float("nan")})
    with pytest.raises(TypeError, match="unsupported payload"):
        architecture_b_crossing_payload_metrics({"value": object()})


def test_rejects_torch_storage_offset_outside_backing_storage() -> None:
    class InvalidStorageOffsetTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, tensor: torch.Tensor) -> InvalidStorageOffsetTensor:
            return torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)

        def storage_offset(self) -> int:
            return 10_000

    tensor = InvalidStorageOffsetTensor(torch.arange(4, dtype=torch.int64))

    with pytest.raises(ValueError, match="Torch storage_offset"):
        architecture_b_crossing_payload_metrics({"tensor": tensor})


def test_rejects_numpy_view_outside_backing_storage() -> None:
    array = np.arange(4, dtype=np.int64)
    invalid = np.lib.stride_tricks.as_strided(
        array[3:],
        shape=(2,),
        strides=(array.itemsize,),
    )

    with pytest.raises(ValueError, match="NumPy byte_offset"):
        architecture_b_crossing_payload_metrics({"array": invalid})


def test_names_exclusion_of_python_heap_overhead() -> None:
    result = architecture_b_crossing_payload_metrics({"ok": True})

    assert result["python_heap_and_container_overhead_included"] is False
    assert "excludes CPython" in result["caveat"]
