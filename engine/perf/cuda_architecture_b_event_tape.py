from __future__ import annotations

"""CUDA-resident atomic append storage for Architecture B lifecycle events."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = ROOT / "engine" / "perf" / "native" / "cuda_architecture_b_event_tape.cu"
_MODULE: Any | None = None

EVENT_TAPE_STATUS_OK = 0
EVENT_TAPE_STATUS_CAPACITY_EXCEEDED = 1

CUDA_EVENT_TAPE_CAPABILITIES = {
    "schema_version": "eiel_cuda_architecture_b_event_tape_capabilities_v1",
    "resident_preallocated_columns": True,
    "atomic_capacity_failure": True,
    "trajectory_host_transfer": False,
    "legacy_and_typed_schema_storage": True,
    "semantic_event_construction": False,
    "runtime_jit": True,
    "promotion_qualified": False,
}


@dataclass(frozen=True)
class CudaEventTapeAppendResult:
    active_count: torch.Tensor
    status: torch.Tensor


def load_cuda_event_tape_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B event tape requires an available CUDA device")
    cuda_source = CUDA_SOURCE.read_text(encoding="utf-8")
    suffix = sha256(cuda_source.encode("utf-8")).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>
    std::vector<torch::Tensor> eiel_cuda_architecture_b_event_tape_append_v1(
        const std::vector<torch::Tensor>& tape_columns,
        const std::vector<torch::Tensor>& event_rows,
        torch::Tensor active_count,
        torch::Tensor status
    );
    """
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load_inline(
        name=f"eiel_cuda_architecture_b_event_tape_{suffix}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["eiel_cuda_architecture_b_event_tape_append_v1"],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def append_event_rows_cuda(
    *,
    tape_columns: Sequence[torch.Tensor],
    event_rows: Sequence[torch.Tensor],
    active_count: torch.Tensor,
    status: torch.Tensor,
    build_directory: str | Path | None = None,
) -> CudaEventTapeAppendResult:
    module = load_cuda_event_tape_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_event_tape_append_v1(
        list(tape_columns),
        list(event_rows),
        active_count,
        status,
    )
    return CudaEventTapeAppendResult(*outputs)


__all__ = [
    "CUDA_EVENT_TAPE_CAPABILITIES",
    "CudaEventTapeAppendResult",
    "EVENT_TAPE_STATUS_CAPACITY_EXCEEDED",
    "EVENT_TAPE_STATUS_OK",
    "append_event_rows_cuda",
    "load_cuda_event_tape_extension",
]
