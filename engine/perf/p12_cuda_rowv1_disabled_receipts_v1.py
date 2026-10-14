from __future__ import annotations

"""CUDA RowV1 disabled-event receipt writer for P12 Gate 2."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "engine" / "perf" / "native"
SOURCE = NATIVE / "p12_cuda_rowv1_disabled_receipts_v1.cu"

ROWV1_ABI_VERSION: Final[int] = 3
ROWV1_SOURCE_LAYOUT_VERSION: Final[int] = 2
ROWV1_WIDTH: Final[int] = 64
DISABLED_REASON_CODES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
CUDA_DISABLED_RECEIPT_CAPABILITIES: Final[dict[str, object]] = {
    "schema_version": "p12_cuda_rowv1_disabled_receipts_capabilities_v1",
    "decision": "P12-CUDA-ROWV1-DISABLED-EVENT-WRITER-A",
    "rowv1_abi_version": ROWV1_ABI_VERSION,
    "rowv1_width": ROWV1_WIDTH,
    "disabled_reason_codes": DISABLED_REASON_CODES,
    "device_written_raw_disabled_reason_code": True,
    "device_written_transaction_sequence": True,
    "device_written_counts": True,
    "device_written_overflow": True,
    "host_projection_required": True,
    "successor_campaign_authorized": False,
}

_MODULES: dict[str, Any] = {}


@dataclass(frozen=True)
class P12CudaRowV1DisabledReceiptResult:
    rows: torch.Tensor
    counts: torch.Tensor
    overflow: torch.Tensor


def _source_digest() -> str:
    digest = sha256()
    digest.update(SOURCE.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(SOURCE.read_bytes())
    digest.update(b"\0")
    return digest.hexdigest()


def load_p12_cuda_rowv1_disabled_receipts_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("P12 CUDA RowV1 disabled receipts require CUDA")
    source_digest = _source_digest()
    module = _MODULES.get(source_digest)
    if module is not None:
        return module
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_p12_cuda_rowv1_disabled_receipts_write_v1(
        torch::Tensor rows,
        torch::Tensor counts,
        torch::Tensor overflow,
        torch::Tensor disabled_reason_codes,
        torch::Tensor parent_agent_ids,
        torch::Tensor parent_slots,
        torch::Tensor parent_slot_tokens,
        torch::Tensor child_agent_ids,
        torch::Tensor child_slots,
        torch::Tensor child_slot_tokens,
        torch::Tensor generation,
        torch::Tensor tick
    );
    """
    module = load_inline(
        name=f"eiel_p12_cuda_rowv1_disabled_receipts_{source_digest[:16]}",
        cpp_sources=[cpp_source],
        cuda_sources=[SOURCE.read_text(encoding="utf-8")],
        functions=["eiel_p12_cuda_rowv1_disabled_receipts_write_v1"],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=[
            "-O2",
            "--fmad=false",
            "-DEIEL_CUDA_ARCHITECTURE_B_NO_PYBIND",
        ],
        extra_include_paths=[str(ROOT / "engine" / "perf")],
        with_cuda=True,
        verbose=bool(verbose),
    )
    _MODULES[source_digest] = module
    return module


def _require_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device | None = None,
    ndim: int | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be CUDA resident")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must share one CUDA device")
    if not tensor.is_contiguous() or tensor.requires_grad:
        raise ValueError(f"{name} must be contiguous and gradient-free")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} has an invalid dtype")
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")


def write_p12_cuda_rowv1_disabled_receipts_v1(
    *,
    rows: torch.Tensor,
    counts: torch.Tensor,
    overflow: torch.Tensor,
    disabled_reason_codes: torch.Tensor,
    parent_agent_ids: torch.Tensor,
    parent_slots: torch.Tensor,
    parent_slot_tokens: torch.Tensor,
    child_agent_ids: torch.Tensor,
    child_slots: torch.Tensor,
    child_slot_tokens: torch.Tensor,
    generation: torch.Tensor,
    tick: torch.Tensor,
    build_directory: str | Path | None = None,
) -> P12CudaRowV1DisabledReceiptResult:
    _require_cuda_tensor("rows", rows, dtype=torch.int64, ndim=3)
    device = rows.device
    _require_cuda_tensor("counts", counts, dtype=torch.int64, device=device, ndim=1)
    _require_cuda_tensor("overflow", overflow, dtype=torch.int32, device=device, ndim=1)
    _require_cuda_tensor(
        "disabled_reason_codes",
        disabled_reason_codes,
        dtype=torch.int32,
        device=device,
        ndim=2,
    )
    for name, tensor in (
        ("parent_agent_ids", parent_agent_ids),
        ("parent_slots", parent_slots),
        ("parent_slot_tokens", parent_slot_tokens),
        ("child_agent_ids", child_agent_ids),
        ("child_slots", child_slots),
        ("child_slot_tokens", child_slot_tokens),
    ):
        _require_cuda_tensor(name, tensor, dtype=torch.int64, device=device, ndim=2)
    _require_cuda_tensor("generation", generation, dtype=torch.int64, device=device, ndim=1)
    _require_cuda_tensor("tick", tick, dtype=torch.int64, device=device, ndim=1)
    if rows.shape[2] != ROWV1_WIDTH:
        raise ValueError("rows must have RowV1 width 64")
    expected = tuple(disabled_reason_codes.shape)
    for name, tensor in (
        ("parent_agent_ids", parent_agent_ids),
        ("parent_slots", parent_slots),
        ("parent_slot_tokens", parent_slot_tokens),
        ("child_agent_ids", child_agent_ids),
        ("child_slots", child_slots),
        ("child_slot_tokens", child_slot_tokens),
    ):
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} shape must match disabled_reason_codes")
    cells = expected[0]
    if tuple(rows.shape[:2]) != (cells, rows.shape[1]):
        raise ValueError("rows cell dimension must match disabled_reason_codes")
    if tuple(counts.shape) != (cells,) or tuple(overflow.shape) != (cells,):
        raise ValueError("counts and overflow must be one value per cell")
    if tuple(generation.shape) != (cells,) or tuple(tick.shape) != (cells,):
        raise ValueError("generation and tick must be one value per cell")

    module = load_p12_cuda_rowv1_disabled_receipts_extension(
        build_directory=build_directory
    )
    result = module.eiel_p12_cuda_rowv1_disabled_receipts_write_v1(
        rows,
        counts,
        overflow,
        disabled_reason_codes,
        parent_agent_ids,
        parent_slots,
        parent_slot_tokens,
        child_agent_ids,
        child_slots,
        child_slot_tokens,
        generation,
        tick,
    )
    return P12CudaRowV1DisabledReceiptResult(*result)
