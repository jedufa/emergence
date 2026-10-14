from __future__ import annotations

"""Exact standalone CUDA mutation/birth event publication for Architecture B."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Sequence

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "engine" / "perf" / "native"
BIRTH_EVENTS_SOURCE = NATIVE / "cuda_architecture_b_birth_events.cpp"
EVENT_TAPE_SOURCE = NATIVE / "cuda_architecture_b_event_tape.cu"
COMPILE_SOURCES: Final[tuple[Path, ...]] = (
    BIRTH_EVENTS_SOURCE,
    EVENT_TAPE_SOURCE,
)

EVENT_TAPE_STATUS_OK = 0
EVENT_TAPE_STATUS_CAPACITY_EXCEEDED = 1
EVENT_TAPE_COLUMN_NAMES: Final[tuple[str, ...]] = (
    "event_tape_type",
    "event_tape_generation",
    "event_tape_tick",
    "event_tape_actor_id",
    "event_tape_target_id",
    "event_tape_value",
    "typed_lifecycle_event_event_type",
    "typed_lifecycle_event_generation",
    "typed_lifecycle_event_tick",
    "typed_lifecycle_event_sequence",
    "typed_lifecycle_event_actor_agent_id",
    "typed_lifecycle_event_target_agent_id",
    "typed_lifecycle_event_actor_slot",
    "typed_lifecycle_event_target_slot",
    "typed_lifecycle_event_parent_agent_id",
    "typed_lifecycle_event_child_agent_id",
    "typed_lifecycle_event_lineage_founder_id",
    "typed_lifecycle_event_lineage_depth",
    "typed_lifecycle_event_subtype_code",
    "typed_lifecycle_event_reason_code",
    "typed_lifecycle_event_success",
    "typed_lifecycle_event_value",
)
EVENT_TAPE_DTYPES: Final[tuple[torch.dtype, ...]] = (
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.float32,
    torch.int32,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int64,
    torch.int32,
    torch.int32,
    torch.int32,
    torch.int8,
    torch.float32,
)

CUDA_BIRTH_EVENT_CAPABILITIES: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_birth_event_capabilities_v1",
    "decision": "ARCHB-CUDA/EXACT-STANDALONE-BIRTH-EVENTS",
    "resident_semantic_inputs": True,
    "cpu_owner_mutation_then_birth_order": True,
    "mutation_event_omitted_when_absent": True,
    "legacy_and_16_typed_columns": True,
    "resident_cursor_derived_sequence": True,
    "single_atomic_event_tape_append": True,
    "capacity_failure": "atomic_fail_closed",
    "generation_integration": False,
    "generation_turnover": False,
    "gru": False,
    "jungle": False,
    "wdl": False,
    "promotion_qualified": False,
}

_MODULES: dict[str, Any] = {}


@dataclass(frozen=True)
class CudaBirthEventAppendResult:
    """Asynchronous resident cursor/status returned by the generic append."""

    active_count: torch.Tensor
    status: torch.Tensor

    def status_code(self) -> int:
        return int(self.status.item())

    def raise_for_status(self) -> None:
        code = self.status_code()
        if code == EVENT_TAPE_STATUS_OK:
            return
        if code == EVENT_TAPE_STATUS_CAPACITY_EXCEEDED:
            raise RuntimeError("Architecture B birth-event tape capacity exceeded")
        raise RuntimeError(f"Architecture B birth-event append failed with status {code}")


def _source_digest() -> str:
    digest = sha256()
    for source in COMPILE_SOURCES:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_cuda_birth_events_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B birth events require an available CUDA device")
    source_digest = _source_digest()
    module = _MODULES.get(source_digest)
    if module is not None:
        return module
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    module = load(
        name=f"eiel_cuda_architecture_b_birth_events_{source_digest[:16]}",
        sources=[str(source) for source in COMPILE_SOURCES],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
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
    rows: int,
) -> None:
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must share the birth-event CUDA device")
    if not tensor.is_contiguous() or tensor.ndim != 1:
        raise ValueError(f"{name} must be a contiguous rank-1 tensor")
    if tensor.dtype != dtype or tensor.numel() != rows:
        raise ValueError(f"{name} must be {dtype} with {rows} rows")


def append_birth_events_cuda(
    *,
    tape_columns: Sequence[torch.Tensor],
    active_count: torch.Tensor,
    status: torch.Tensor,
    parent_agent_ids: torch.Tensor,
    child_agent_ids: torch.Tensor,
    parent_slots: torch.Tensor,
    child_slots: torch.Tensor,
    lineage_founder_ids: torch.Tensor,
    lineage_depths: torch.Tensor,
    generation: torch.Tensor,
    tick: torch.Tensor,
    mutation_occurred: torch.Tensor,
    mutation_delta_total: torch.Tensor,
    child_energy: torch.Tensor,
    build_directory: str | Path | None = None,
) -> CudaBirthEventAppendResult:
    """Append owner-equivalent mutation/birth rows as one fail-closed transaction."""
    if not parent_agent_ids.is_cuda:
        raise ValueError("birth-event semantic inputs must be CUDA resident")
    device = parent_agent_ids.device
    birth_count = parent_agent_ids.numel()
    for name, tensor, dtype in (
        ("parent_agent_ids", parent_agent_ids, torch.int64),
        ("child_agent_ids", child_agent_ids, torch.int64),
        ("parent_slots", parent_slots, torch.int64),
        ("child_slots", child_slots, torch.int64),
        ("lineage_founder_ids", lineage_founder_ids, torch.int64),
        ("lineage_depths", lineage_depths, torch.int32),
        ("mutation_occurred", mutation_occurred, torch.bool),
        ("mutation_delta_total", mutation_delta_total, torch.float32),
        ("child_energy", child_energy, torch.float32),
    ):
        _validate_vector(
            name,
            tensor,
            dtype=dtype,
            device=device,
            rows=birth_count,
        )
    for name, tensor in (("generation", generation), ("tick", tick)):
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"{name} must share the birth-event CUDA device")
        if not tensor.is_contiguous() or tensor.ndim != 1 or tensor.dtype != torch.int64:
            raise ValueError(f"{name} must be a contiguous rank-1 int64 tensor")
        if tensor.numel() not in {1, birth_count}:
            raise ValueError(f"{name} must have one value or one value per birth")

    columns = list(tape_columns)
    if len(columns) != len(EVENT_TAPE_COLUMN_NAMES):
        raise ValueError("birth-event tape requires 6 legacy and 16 typed columns")
    capacity = columns[0].numel()
    for name, tensor, dtype in zip(
        EVENT_TAPE_COLUMN_NAMES,
        columns,
        EVENT_TAPE_DTYPES,
        strict=True,
    ):
        _validate_vector(name, tensor, dtype=dtype, device=device, rows=capacity)
    _validate_vector(
        "active_count",
        active_count,
        dtype=torch.int64,
        device=device,
        rows=1,
    )
    _validate_vector(
        "status",
        status,
        dtype=torch.int32,
        device=device,
        rows=1,
    )

    module = load_cuda_birth_events_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_birth_events_append_v1(
        columns,
        active_count,
        status,
        parent_agent_ids,
        child_agent_ids,
        parent_slots,
        child_slots,
        lineage_founder_ids,
        lineage_depths,
        generation,
        tick,
        mutation_occurred,
        mutation_delta_total,
        child_energy,
    )
    return CudaBirthEventAppendResult(*outputs)


__all__ = [
    "CUDA_BIRTH_EVENT_CAPABILITIES",
    "CudaBirthEventAppendResult",
    "EVENT_TAPE_COLUMN_NAMES",
    "EVENT_TAPE_DTYPES",
    "EVENT_TAPE_STATUS_CAPACITY_EXCEEDED",
    "EVENT_TAPE_STATUS_OK",
    "append_birth_events_cuda",
    "load_cuda_birth_events_extension",
]
