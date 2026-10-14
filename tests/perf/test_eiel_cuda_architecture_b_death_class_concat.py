from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from perf.cuda_architecture_b_death_class_concat import (
    DEATH_CLASS_CONCAT_SOURCE,
    DEATH_CLASS_CONCAT_STATUS_CAPACITY_EXCEEDED,
    DEATH_CLASS_CONCAT_STATUS_INVALID_COUNT,
    DEATH_CLASS_CONCAT_STATUS_OK,
    concat_death_classes_cuda,
    load_cuda_death_class_concat_extension,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-death-class-concat-build")


@pytest.fixture(scope="module")
def cuda_module(cuda_build_directory: Path) -> Any:
    return load_cuda_death_class_concat_extension(
        build_directory=cuda_build_directory
    )


def _fixed_vector(
    values: list[int],
    *,
    capacity: int,
    dtype: torch.dtype,
    poison: int,
) -> torch.Tensor:
    tensor = torch.full((capacity,), poison, device="cuda", dtype=dtype)
    if values:
        tensor[: len(values)] = torch.tensor(values, device="cuda", dtype=dtype)
    return tensor


def _inputs(
    *,
    action_agent_ids: list[int],
    action_slots: list[int],
    action_reasons: list[int],
    residual_agent_ids: list[int],
    residual_slots: list[int],
    residual_reasons: list[int],
    capacity: int = 6,
) -> dict[str, torch.Tensor]:
    assert len(action_agent_ids) == len(action_slots) == len(action_reasons)
    assert len(residual_agent_ids) == len(residual_slots) == len(residual_reasons)
    return {
        "action_agent_ids": _fixed_vector(
            action_agent_ids,
            capacity=capacity,
            dtype=torch.int64,
            poison=-901,
        ),
        "action_slots": _fixed_vector(
            action_slots,
            capacity=capacity,
            dtype=torch.int64,
            poison=991,
        ),
        "action_reason_codes": _fixed_vector(
            action_reasons,
            capacity=capacity,
            dtype=torch.int32,
            poison=81,
        ),
        "action_count": torch.tensor(
            [len(action_agent_ids)], device="cuda", dtype=torch.int64
        ),
        "residual_agent_ids": _fixed_vector(
            residual_agent_ids,
            capacity=capacity,
            dtype=torch.int64,
            poison=-902,
        ),
        "residual_slots": _fixed_vector(
            residual_slots,
            capacity=capacity,
            dtype=torch.int64,
            poison=992,
        ),
        "residual_reason_codes": _fixed_vector(
            residual_reasons,
            capacity=capacity,
            dtype=torch.int32,
            poison=82,
        ),
        "residual_count": torch.tensor(
            [len(residual_agent_ids)], device="cuda", dtype=torch.int64
        ),
    }


def _run(
    inputs: dict[str, torch.Tensor],
    *,
    build_directory: Path,
):
    result = concat_death_classes_cuda(
        **inputs,
        build_directory=build_directory,
    )
    torch.cuda.synchronize()
    return result


def _assert_default_output(result: Any, *, capacity: int) -> None:
    assert result.combined_agent_ids.cpu().tolist() == [-1] * capacity
    assert result.combined_slots.cpu().tolist() == [-1] * capacity
    assert result.combined_reason_codes.cpu().tolist() == [0] * capacity
    assert result.combined_count.cpu().tolist() == [0]


def test_normal_concat_preserves_stable_action_then_residual_order(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    inputs = _inputs(
        action_agent_ids=[20, 90],
        action_slots=[3, 1],
        action_reasons=[2, 1],
        residual_agent_ids=[30, 50],
        residual_slots=[5, 2],
        residual_reasons=[2, 1],
    )

    result = _run(inputs, build_directory=cuda_build_directory)

    assert result.status.cpu().tolist() == [DEATH_CLASS_CONCAT_STATUS_OK]
    assert result.combined_count.cpu().tolist() == [4]
    assert result.combined_agent_ids.cpu().tolist() == [20, 90, 30, 50, -1, -1]
    assert result.combined_slots.cpu().tolist() == [3, 1, 5, 2, -1, -1]
    assert result.combined_reason_codes.cpu().tolist() == [2, 1, 2, 1, 0, 0]


@pytest.mark.parametrize(
    ("action_values", "residual_values", "expected"),
    [
        (([], [], []), ([30, 50], [5, 2], [2, 1]), [30, 50]),
        (([20, 90], [3, 1], [2, 1]), ([], [], []), [20, 90]),
        (([], [], []), ([], [], []), []),
    ],
    ids=["zero_action", "zero_residual", "both_zero"],
)
def test_zero_class_is_an_exact_fixed_capacity_concat(
    action_values: tuple[list[int], list[int], list[int]],
    residual_values: tuple[list[int], list[int], list[int]],
    expected: list[int],
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    inputs = _inputs(
        action_agent_ids=action_values[0],
        action_slots=action_values[1],
        action_reasons=action_values[2],
        residual_agent_ids=residual_values[0],
        residual_slots=residual_values[1],
        residual_reasons=residual_values[2],
        capacity=4,
    )

    result = _run(inputs, build_directory=cuda_build_directory)

    assert result.status.cpu().tolist() == [DEATH_CLASS_CONCAT_STATUS_OK]
    assert result.combined_count.cpu().tolist() == [len(expected)]
    assert result.combined_agent_ids.cpu().tolist() == expected + [-1] * (
        4 - len(expected)
    )


def test_poisoned_inactive_tails_are_ignored(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    inputs = _inputs(
        action_agent_ids=[12],
        action_slots=[2],
        action_reasons=[1],
        residual_agent_ids=[44],
        residual_slots=[7],
        residual_reasons=[2],
        capacity=5,
    )

    result = _run(inputs, build_directory=cuda_build_directory)

    assert result.status.cpu().tolist() == [DEATH_CLASS_CONCAT_STATUS_OK]
    assert result.combined_agent_ids.cpu().tolist() == [12, 44, -1, -1, -1]
    assert result.combined_slots.cpu().tolist() == [2, 7, -1, -1, -1]
    assert result.combined_reason_codes.cpu().tolist() == [1, 2, 0, 0, 0]


def test_sum_above_capacity_fails_atomically(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    inputs = _inputs(
        action_agent_ids=[10, 20, 30],
        action_slots=[1, 2, 3],
        action_reasons=[1, 2, 1],
        residual_agent_ids=[40, 50],
        residual_slots=[4, 5],
        residual_reasons=[2, 1],
        capacity=4,
    )

    result = _run(inputs, build_directory=cuda_build_directory)

    assert result.status.cpu().tolist() == [
        DEATH_CLASS_CONCAT_STATUS_CAPACITY_EXCEEDED
    ]
    _assert_default_output(result, capacity=4)


@pytest.mark.parametrize(
    ("count_name", "count_value"),
    [
        ("action_count", -1),
        ("action_count", 5),
        ("residual_count", -1),
        ("residual_count", 5),
    ],
)
def test_invalid_count_fails_atomically(
    count_name: str,
    count_value: int,
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    inputs = _inputs(
        action_agent_ids=[10],
        action_slots=[1],
        action_reasons=[1],
        residual_agent_ids=[20],
        residual_slots=[2],
        residual_reasons=[2],
        capacity=4,
    )
    inputs[count_name].fill_(count_value)

    result = _run(inputs, build_directory=cuda_build_directory)

    assert result.status.cpu().tolist() == [DEATH_CLASS_CONCAT_STATUS_INVALID_COUNT]
    _assert_default_output(result, capacity=4)


def test_sources_forbid_sync_and_dynamic_shape_primitives() -> None:
    native_source = DEATH_CLASS_CONCAT_SOURCE.read_text(encoding="utf-8")
    wrapper_source = Path(concat_death_classes_cuda.__code__.co_filename).read_text(
        encoding="utf-8"
    )
    for forbidden in (
        ".item",
        "masked_select",
        "nonzero",
        "cudaDeviceSynchronize",
        "cudaStreamSynchronize",
        "cudaMemcpy",
    ):
        assert forbidden not in native_source
    for forbidden in (
        ".item(",
        ".cpu(",
        "masked_select",
        "nonzero",
        "torch.cuda.synchronize",
    ):
        assert forbidden not in wrapper_source
    assert "death_class_concat_kernel<<<1, 1" in native_source
    assert "#ifndef EIEL_CUDA_ARCHITECTURE_B_NO_PYBIND" in native_source
    assert "PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)" in native_source
    assert "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND" in wrapper_source
