from __future__ import annotations

import os
import shutil
from pathlib import Path
from types import ModuleType

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
HEADER = (
    ROOT
    / "engine/perf/native/cpu_compact_summary_torch_adapter_v1.hpp"
)
SOURCE = HEADER.with_suffix(".cpp")
BINDING = (
    ROOT
    / "tests/perf/test_eiel_cpu_compact_summary_torch_adapter_v1_binding.cpp"
)
MODULE_NAME = "eiel_cpu_compact_summary_torch_adapter_v1_test"

SUPPORTED_DTYPES = (
    (torch.bool, "bool", 0),
    (torch.int8, "int8", 1),
    (torch.uint8, "uint8", 2),
    (torch.int32, "int32", 3),
    (torch.float32, "float32", 4),
    (torch.int64, "int64", 5),
)


@pytest.fixture(scope="module")
def native_adapter(tmp_path_factory: pytest.TempPathFactory) -> ModuleType:
    if shutil.which("c++") is None:
        pytest.skip("a C++ compiler is unavailable")

    from torch.utils.cpp_extension import load

    build = tmp_path_factory.mktemp("cpu_compact_summary_torch_adapter_v1")
    old_max_jobs = os.environ.get("MAX_JOBS")
    os.environ["MAX_JOBS"] = "1"
    try:
        return load(
            name=MODULE_NAME,
            sources=[str(BINDING), str(SOURCE)],
            extra_include_paths=[str(ROOT / "engine/perf/native")],
            extra_cflags=[
                "-O1",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
            ],
            build_directory=str(build),
            with_cuda=False,
            verbose=False,
        )
    finally:
        if old_max_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = old_max_jobs


def test_production_adapter_has_no_copy_or_python_projection() -> None:
    production = HEADER.read_text(encoding="ascii") + SOURCE.read_text(
        encoding="ascii"
    )
    for forbidden in (
        "Python.h",
        "pybind11",
        "cuda_runtime",
        ".cpu(",
        ".to(",
        ".contiguous(",
        ".clone(",
        ".copy_(",
    ):
        assert forbidden not in production


@pytest.mark.parametrize(("dtype", "dtype_name", "dtype_code"), SUPPORTED_DTYPES)
@pytest.mark.parametrize("shape", [(5,), (2, 3)])
def test_supported_dtypes_preserve_exact_rank_shape_pointer_and_bytes(
    native_adapter: ModuleType,
    dtype: torch.dtype,
    dtype_name: str,
    dtype_code: int,
    shape: tuple[int, ...],
) -> None:
    tensor = torch.zeros(shape, dtype=dtype)
    output = native_adapter.inspect(tensor)

    assert output == {
        "data_ptr": tensor.data_ptr(),
        "dtype": dtype_name,
        "dtype_code": dtype_code,
        "rank": len(shape),
        "extent0": shape[0],
        "extent1": shape[1] if len(shape) == 2 else 0,
        "storage_bytes": tensor.numel() * tensor.element_size(),
        "c_contiguous": True,
    }


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0)])
def test_empty_tensor_preserves_visible_pointer_and_zero_logical_bytes(
    native_adapter: ModuleType, shape: tuple[int, ...]
) -> None:
    tensor = torch.empty(shape, dtype=torch.float32)
    output = native_adapter.inspect(tensor)

    assert output["data_ptr"] == tensor.data_ptr()
    assert output["rank"] == len(shape)
    assert output["extent0"] == shape[0]
    assert output["extent1"] == (shape[1] if len(shape) == 2 else 0)
    assert output["storage_bytes"] == 0


def test_contiguous_storage_offset_view_uses_visible_pointer_and_bytes(
    native_adapter: ModuleType,
) -> None:
    storage = torch.arange(12, dtype=torch.int64)
    tensor = storage[3:9]
    assert tensor.is_contiguous()
    assert tensor.storage_offset() == 3

    output = native_adapter.inspect(tensor)

    assert output["data_ptr"] == tensor.data_ptr()
    assert output["data_ptr"] == storage.data_ptr() + 3 * storage.element_size()
    assert output["storage_bytes"] == 6 * tensor.element_size()


@pytest.mark.parametrize(
    "tensor",
    [
        torch.arange(12, dtype=torch.int64).reshape(3, 4).t(),
        torch.arange(12, dtype=torch.int64)[::2],
    ],
)
def test_noncontiguous_tensor_is_rejected(
    native_adapter: ModuleType, tensor: torch.Tensor
) -> None:
    assert not tensor.is_contiguous()
    with pytest.raises(RuntimeError, match="must be contiguous"):
        native_adapter.inspect(tensor)


@pytest.mark.parametrize("dtype", [torch.int16, torch.float64])
def test_unsupported_dtype_is_rejected(
    native_adapter: ModuleType, dtype: torch.dtype
) -> None:
    with pytest.raises(RuntimeError, match="unsupported dtype"):
        native_adapter.inspect(torch.empty((4,), dtype=dtype))


@pytest.mark.parametrize(
    "tensor",
    [torch.tensor(1, dtype=torch.int64), torch.empty((1, 2, 3), dtype=torch.int64)],
)
def test_unsupported_rank_is_rejected(
    native_adapter: ModuleType, tensor: torch.Tensor
) -> None:
    with pytest.raises(RuntimeError, match="rank must be 1 or 2"):
        native_adapter.inspect(tensor)


def test_non_cpu_device_is_rejected_without_requiring_cuda(
    native_adapter: ModuleType,
) -> None:
    tensor = torch.empty((4,), dtype=torch.int64, device="meta")
    with pytest.raises(RuntimeError, match="must be on CPU"):
        native_adapter.inspect(tensor)


def test_logical_byte_count_overflow_is_rejected(
    native_adapter: ModuleType,
) -> None:
    tensor = torch.empty((1,), dtype=torch.int64).as_strided((2**61,), (0,))
    with pytest.raises(RuntimeError, match="logical byte count overflow"):
        native_adapter.inspect(tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_tensor_is_rejected_when_available(
    native_adapter: ModuleType,
) -> None:
    tensor = torch.empty((4,), dtype=torch.int64, device="cuda")
    with pytest.raises(RuntimeError, match="must be on CPU"):
        native_adapter.inspect(tensor)
