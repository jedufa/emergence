from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from engine.native_runtime import compact_summary_planes as host_planes
from engine.perf import cuda_architecture_b_compact_summary as summary_tensors


def test_cpu_allocation_has_frozen_shapes_dtypes_zeroes_and_bytes() -> None:
    value = summary_tensors.allocate_compact_summary_tensors(2, 3, device="cpu")
    expected = {
        "validity": ((2, 3), torch.uint8),
        "defined_mask": ((2, 3, 1), torch.uint64),
        "status_codes": ((2, 3, 12), torch.int32),
        "integer_values": ((2, 3, 76), torch.int64),
        "float_values": ((2, 3, 21), torch.float64),
        "digest_values": ((2, 3, 10, 32), torch.uint8),
        "text_values": ((2, 3, 1, 65), torch.uint8),
    }

    assert tuple(value.as_mapping()) == tuple(expected)
    for name, tensor in value.as_mapping().items():
        shape, dtype = expected[name]
        assert tuple(tensor.shape) == shape
        assert tensor.dtype == dtype
        assert tensor.device == torch.device("cpu")
        assert tensor.is_contiguous()
        assert not bool(tensor.view(torch.uint8).any().item())

    assert summary_tensors.LOGICAL_BYTES_PER_GENERATION_CELL == 1_218
    assert summary_tensors.LOGICAL_BYTES_PER_GENERATION_CELL == (
        host_planes.LOGICAL_BYTES_PER_GENERATION_CELL
    )
    assert value.nbytes == 2 * 3 * 1_218
    assert summary_tensors.compact_summary_tensor_nbytes(value) == value.nbytes
    assert summary_tensors.validate_compact_summary_tensors(
        value, generation_count=2, cell_count=3, device="cpu"
    ) == (2, 3)
    assert summary_tensors.validate_compact_summary_tensors(
        value, device="cpu:0"
    ) == (2, 3)


@pytest.mark.parametrize("generations,cells", [(0, 1), (1, 0), (-1, 1)])
def test_cpu_allocation_rejects_nonpositive_dimensions(
    generations: int, cells: int
) -> None:
    with pytest.raises(ValueError, match="positive"):
        summary_tensors.allocate_compact_summary_tensors(
            generations, cells, device="cpu"
        )


@pytest.mark.parametrize("generations,cells", [(True, 1), (1, False), (1.0, 1)])
def test_cpu_allocation_rejects_noninteger_dimensions(
    generations: object, cells: object
) -> None:
    with pytest.raises(TypeError, match="must be an int"):
        summary_tensors.allocate_compact_summary_tensors(  # type: ignore[arg-type]
            generations, cells, device="cpu"
        )


def test_allocation_rejects_non_cpu_cuda_device() -> None:
    with pytest.raises(ValueError, match="CPU or CUDA"):
        summary_tensors.allocate_compact_summary_tensors(1, 1, device="meta")


def test_cpu_validation_rejects_names_type_shape_dtype_and_contiguity() -> None:
    value = summary_tensors.allocate_compact_summary_tensors(2, 3, device="cpu")
    incomplete = dict(value.as_mapping())
    incomplete.pop("text_values")
    with pytest.raises(ValueError, match="plane names mismatch"):
        summary_tensors.validate_compact_summary_tensors(incomplete)

    with pytest.raises(TypeError, match="status_codes must be a Torch tensor"):
        summary_tensors.validate_compact_summary_tensors(
            replace(value, status_codes=np.zeros((2, 3, 12), dtype=np.int32))
        )
    with pytest.raises(ValueError, match="defined_mask dtype"):
        summary_tensors.validate_compact_summary_tensors(
            replace(value, defined_mask=value.defined_mask.to(torch.int64))
        )
    with pytest.raises(ValueError, match="status_codes shape"):
        summary_tensors.validate_compact_summary_tensors(
            replace(value, status_codes=value.status_codes[:, :, :-1])
        )

    noncontiguous = torch.zeros((2, 3, 42), dtype=torch.float64)[:, :, ::2]
    assert tuple(noncontiguous.shape) == (2, 3, 21)
    assert not noncontiguous.is_contiguous()
    with pytest.raises(ValueError, match="float_values must be contiguous"):
        summary_tensors.validate_compact_summary_tensors(
            replace(value, float_values=noncontiguous)
        )


def test_cpu_validation_rejects_mixed_device_and_dimension_mismatches() -> None:
    value = summary_tensors.allocate_compact_summary_tensors(2, 3, device="cpu")
    meta_validity = torch.zeros((2, 3), dtype=torch.uint8, device="meta")
    with pytest.raises(ValueError, match="same device"):
        summary_tensors.validate_compact_summary_tensors(
            replace(value, validity=meta_validity)
        )
    with pytest.raises(ValueError, match="generation_count"):
        summary_tensors.validate_compact_summary_tensors(value, generation_count=1)
    with pytest.raises(ValueError, match="cell_count"):
        summary_tensors.validate_compact_summary_tensors(value, cell_count=2)
    with pytest.raises(ValueError, match="tensor device must be cuda"):
        summary_tensors.validate_compact_summary_tensors(value, device="cuda")
    with pytest.raises(ValueError, match="CPU or CUDA"):
        summary_tensors.validate_compact_summary_tensors(value, device="meta")


def test_cpu_validation_enforces_validity_no_nan_and_invalid_bit_zero_noop() -> None:
    invalid_validity = summary_tensors.allocate_compact_summary_tensors(
        1, 1, device="cpu"
    )
    invalid_validity.validity[0, 0] = 2
    with pytest.raises(ValueError, match="0 or 1"):
        summary_tensors.validate_compact_summary_tensors(invalid_validity)

    nan_value = summary_tensors.allocate_compact_summary_tensors(1, 1, device="cpu")
    nan_value.validity[0, 0] = 1
    nan_value.float_values[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="must be finite"):
        summary_tensors.validate_compact_summary_tensors(nan_value)

    infinite_value = summary_tensors.allocate_compact_summary_tensors(
        1, 1, device="cpu"
    )
    infinite_value.validity[0, 0] = 1
    infinite_value.float_values[0, 0, 0] = torch.inf
    with pytest.raises(ValueError, match="must be finite"):
        summary_tensors.validate_compact_summary_tensors(infinite_value)

    dirty_value = summary_tensors.allocate_compact_summary_tensors(1, 1, device="cpu")
    dirty_value.integer_values[0, 0, 4] = 1
    with pytest.raises(ValueError, match="bit-zero no-op storage in integer_values"):
        summary_tensors.validate_compact_summary_tensors(dirty_value)

    negative_zero = summary_tensors.allocate_compact_summary_tensors(
        1, 1, device="cpu"
    )
    negative_zero.float_values[0, 0, 0] = -0.0
    with pytest.raises(ValueError, match="bit-zero no-op storage in float_values"):
        summary_tensors.validate_compact_summary_tensors(negative_zero)


def test_explicit_cpu_final_boundary_returns_independent_numpy_planes() -> None:
    value = summary_tensors.allocate_compact_summary_tensors(1, 2, device="cpu")
    value.validity.fill_(1)
    value.integer_values[0, 1, 7] = 42
    value.float_values[0, 0, 3] = 0.25
    value.digest_values[0, 1, 2, 4] = 0xA5

    converted = summary_tensors.to_numpy_compact_summary_planes(value)

    assert isinstance(converted, host_planes.CompactSummaryPlanes)
    assert host_planes.validate_compact_summary_plane_layout(converted) == (1, 2)
    assert converted.integer_values[0, 1, 7] == 42
    assert converted.float_values[0, 0, 3] == 0.25
    assert converted.digest_values[0, 1, 2, 4] == 0xA5
    assert all(array.flags.c_contiguous for array in converted.as_mapping().values())
    assert sum(array.nbytes for array in converted.as_mapping().values()) == 2 * 1_218

    converted.integer_values[0, 1, 7] = 99
    assert value.integer_values[0, 1, 7].item() == 42


@pytest.mark.skipif(not torch.cuda.is_available(), reason="safe CUDA is unavailable")
def test_tiny_cuda_allocation_validation_and_explicit_boundary_smoke() -> None:
    try:
        value = summary_tensors.allocate_compact_summary_tensors(
            1, 1, device="cuda"
        )
    except RuntimeError as exc:
        pytest.skip(f"safe CUDA allocation is unavailable: {exc}")

    assert all(tensor.device.type == "cuda" for tensor in value.as_mapping().values())
    assert summary_tensors.validate_compact_summary_tensors(value, device="cuda") == (
        1,
        1,
    )
    converted = summary_tensors.to_numpy_compact_summary_planes(value)
    assert isinstance(converted, host_planes.CompactSummaryPlanes)
    assert sum(array.nbytes for array in converted.as_mapping().values()) == 1_218
