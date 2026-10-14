from __future__ import annotations

"""Fixed-capacity CUDA concatenation for action and residual death rows."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "engine" / "perf" / "native"
DEATH_CLASS_CONCAT_SOURCE = NATIVE / "cuda_architecture_b_death_class_concat.cu"
COMPILE_SOURCES: Final[tuple[Path, ...]] = (DEATH_CLASS_CONCAT_SOURCE,)

DEATH_CLASS_CONCAT_STATUS_OK = 0
DEATH_CLASS_CONCAT_STATUS_CAPACITY_EXCEEDED = 1
DEATH_CLASS_CONCAT_STATUS_INVALID_COUNT = 2

_MODULES: dict[str, Any] = {}


@dataclass(frozen=True)
class CudaDeathClassConcatResult:
    combined_agent_ids: torch.Tensor
    combined_slots: torch.Tensor
    combined_reason_codes: torch.Tensor
    combined_count: torch.Tensor
    status: torch.Tensor


def _source_digest() -> str:
    digest = sha256()
    for source in COMPILE_SOURCES:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_cuda_death_class_concat_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Architecture B death-class concatenation requires an available CUDA device"
        )
    source_digest = _source_digest()
    module = _MODULES.get(source_digest)
    if module is not None:
        return module
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor>
    eiel_cuda_architecture_b_death_class_concat_v1(
        torch::Tensor action_agent_ids,
        torch::Tensor action_slots,
        torch::Tensor action_reason_codes,
        torch::Tensor action_count,
        torch::Tensor residual_agent_ids,
        torch::Tensor residual_slots,
        torch::Tensor residual_reason_codes,
        torch::Tensor residual_count
    );
    """
    module = load_inline(
        name=f"eiel_cuda_architecture_b_death_class_concat_{source_digest[:16]}",
        cpp_sources=[cpp_source],
        cuda_sources=[DEATH_CLASS_CONCAT_SOURCE.read_text(encoding="utf-8")],
        functions=["eiel_cuda_architecture_b_death_class_concat_v1"],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=[
            "-O2",
            "--fmad=false",
            "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND",
        ],
        with_cuda=True,
        verbose=bool(verbose),
    )
    _MODULES[source_digest] = module
    return module


def _validate_vector(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    capacity: int,
) -> None:
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must share the death-class CUDA device")
    if not tensor.is_contiguous() or tensor.ndim != 1:
        raise ValueError(f"{name} must be a contiguous rank-1 tensor")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}")
    if tensor.numel() != capacity:
        raise ValueError("death class tensors must share one fixed capacity")


def concat_death_classes_cuda(
    *,
    action_agent_ids: torch.Tensor,
    action_slots: torch.Tensor,
    action_reason_codes: torch.Tensor,
    action_count: torch.Tensor,
    residual_agent_ids: torch.Tensor,
    residual_slots: torch.Tensor,
    residual_reason_codes: torch.Tensor,
    residual_count: torch.Tensor,
    build_directory: str | Path | None = None,
) -> CudaDeathClassConcatResult:
    """Concatenate active action rows before active residual rows on CUDA."""
    if not action_agent_ids.is_cuda:
        raise ValueError("death class inputs must be CUDA resident")
    device = action_agent_ids.device
    capacity = action_agent_ids.numel()
    for name, tensor, dtype in (
        ("action_agent_ids", action_agent_ids, torch.int64),
        ("action_slots", action_slots, torch.int64),
        ("action_reason_codes", action_reason_codes, torch.int32),
        ("residual_agent_ids", residual_agent_ids, torch.int64),
        ("residual_slots", residual_slots, torch.int64),
        ("residual_reason_codes", residual_reason_codes, torch.int32),
    ):
        _validate_vector(
            name,
            tensor,
            dtype=dtype,
            device=device,
            capacity=capacity,
        )
    for name, tensor in (
        ("action_count", action_count),
        ("residual_count", residual_count),
    ):
        _validate_vector(
            name,
            tensor,
            dtype=torch.int64,
            device=device,
            capacity=1,
        )

    module = load_cuda_death_class_concat_extension(
        build_directory=build_directory
    )
    outputs = module.eiel_cuda_architecture_b_death_class_concat_v1(
        action_agent_ids,
        action_slots,
        action_reason_codes,
        action_count,
        residual_agent_ids,
        residual_slots,
        residual_reason_codes,
        residual_count,
    )
    return CudaDeathClassConcatResult(*outputs)


__all__ = [
    "CudaDeathClassConcatResult",
    "DEATH_CLASS_CONCAT_SOURCE",
    "DEATH_CLASS_CONCAT_STATUS_CAPACITY_EXCEEDED",
    "DEATH_CLASS_CONCAT_STATUS_INVALID_COUNT",
    "DEATH_CLASS_CONCAT_STATUS_OK",
    "concat_death_classes_cuda",
    "load_cuda_death_class_concat_extension",
]
