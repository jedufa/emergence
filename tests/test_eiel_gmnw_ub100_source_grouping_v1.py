from __future__ import annotations

import inspect

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_source_allocator_v1 as allocator


def test_grouped_construction_alias_is_copied_into_private_batch_rows() -> None:
    first = torch.tensor([1, 2, 3], dtype=torch.int64)
    second = torch.tensor([4, 5, 6], dtype=torch.int64)

    stacked = allocator._stack_per_cell_values_v1(
        [first, first, second, second],
        name="fixture",
        device=torch.device("cpu"),
        source_groups=(0, 0, 1, 1),
    )

    assert stacked.tolist() == [[1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6]]
    stacked[0, 0] = 99
    assert stacked[1, 0].item() == 1


def test_grouped_construction_rejects_cross_source_storage_alias() -> None:
    value = torch.tensor([1], dtype=torch.int64)

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="storage alias",
    ):
        allocator._stack_per_cell_values_v1(
            [value, value],
            name="fixture",
            device=torch.device("cpu"),
            source_groups=(0, 1),
        )


def test_grouped_source_api_is_explicit_and_disabled_by_default() -> None:
    parameter = inspect.signature(
        allocator.allocate_direct_resident_source_v1
    ).parameters["per_cell_source_groups"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_per_cell_gaussian_tapes_admit_ragged_lengths_with_one_width() -> None:
    sources = [
        {
            "python_random_gauss_primary_tape": torch.tensor(
                [1.0, 2.0], dtype=torch.float64
            ),
            "python_random_gauss_secondary_tape": torch.tensor(
                [3.0, 4.0], dtype=torch.float64
            ),
        },
        {
            "python_random_gauss_primary_tape": torch.tensor(
                [5.0], dtype=torch.float64
            ),
            "python_random_gauss_secondary_tape": torch.tensor(
                [6.0], dtype=torch.float64
            ),
        },
    ]

    width = allocator._validated_python_gauss_tape_width_v1(sources)
    stacked = allocator._stack_ragged_vectors_v1(
        [source["python_random_gauss_primary_tape"] for source in sources],
        name="python_random_gauss_primary_tape",
        dtype=torch.float64,
        device=torch.device("cpu"),
        width=width,
    )

    assert width == 2
    assert stacked.tolist() == [[1.0, 2.0], [5.0, -1.0]]


def test_per_cell_gaussian_tape_pair_lengths_must_match() -> None:
    sources = [
        {
            "python_random_gauss_primary_tape": torch.zeros(
                2, dtype=torch.float64
            ),
            "python_random_gauss_secondary_tape": torch.zeros(
                1, dtype=torch.float64
            ),
        }
    ]

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="tape pair length mismatch",
    ):
        allocator._validated_python_gauss_tape_width_v1(sources)
