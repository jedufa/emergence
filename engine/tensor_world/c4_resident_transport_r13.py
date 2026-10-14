"""Manifest-driven raw-byte transport prototype for the R11 resident owner."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Final

import torch
from torch.utils.cpp_extension import load

from .c4_resident_composed_r11 import _R11_GUARDED_SHARED_TENSOR_PATHS_V1
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
NATIVE_SOURCE = (
    ROOT / "engine/tensor_world/native/tensor_world_c4_resident_transport_r13.cu"
)
R11_OWNER_SOURCE = ROOT / "engine/tensor_world/c4_resident_composed_r11.py"

C4_RESIDENT_TRANSPORT_R13_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_transport_r13_v1"
)
C4_RESIDENT_TRANSPORT_R13_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_TRANSPORT_R13_CHUNK_BYTES_V1: Final[int] = 256 * 1024
C4_RESIDENT_TRANSPORT_R13_THREADS_V1: Final[int] = 256

_MODULE: ModuleType | None = None


@dataclass(frozen=True, slots=True)
class C4ResidentTransportFieldR13V1:
    paths: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    byte_count: int
    source_address: int
    destination_address: int


@dataclass(frozen=True, slots=True)
class C4ResidentTransportInventoryR13V1:
    fields: tuple[C4ResidentTransportFieldR13V1, ...]
    shared_alias_paths: tuple[str, ...]
    total_bytes: int
    chunk_count: int
    device: torch.device
    schema_version: str = C4_RESIDENT_TRANSPORT_R13_SCHEMA_VERSION_V1

    @property
    def unique_field_count(self) -> int:
        return len(self.fields)

    @property
    def covered_paths(self) -> tuple[str, ...]:
        return tuple(path for field in self.fields for path in field.paths)


@dataclass(frozen=True, slots=True)
class C4ResidentTransportPlanR13V1:
    inventory: C4ResidentTransportInventoryR13V1
    copy_source_addresses: torch.Tensor
    copy_destination_addresses: torch.Tensor
    copy_byte_counts: torch.Tensor
    publish_source_addresses: torch.Tensor
    publish_destination_addresses: torch.Tensor
    publish_byte_counts: torch.Tensor
    schema_version: str = C4_RESIDENT_TRANSPORT_R13_SCHEMA_VERSION_V1

    def validate(self) -> None:
        if self.schema_version != C4_RESIDENT_TRANSPORT_R13_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("R13 transport plan schema is unsupported")
        expected = self.inventory.chunk_count
        device = self.inventory.device
        for name, tensor in (
            ("copy_source_addresses", self.copy_source_addresses),
            ("copy_destination_addresses", self.copy_destination_addresses),
            ("copy_byte_counts", self.copy_byte_counts),
            ("publish_source_addresses", self.publish_source_addresses),
            ("publish_destination_addresses", self.publish_destination_addresses),
            ("publish_byte_counts", self.publish_byte_counts),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != (expected,)
                or tensor.dtype != torch.int64
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise TensorWorldContractError(
                    f"R13 transport plan {name} ABI mismatch"
                )


def _tensor_path_pairs(
    source: object,
    destination: object,
    *,
    path: str,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []

    def visit(src: object, dst: object, current: str) -> None:
        if isinstance(src, torch.Tensor):
            if not isinstance(dst, torch.Tensor):
                raise TensorWorldContractError("R13 destination tensor is missing")
            pairs.append((current, src, dst))
            return
        if isinstance(src, tuple):
            if not isinstance(dst, tuple) or len(src) != len(dst):
                raise TensorWorldContractError("R13 tuple structure drifted")
            for index, (src_item, dst_item) in enumerate(
                zip(src, dst, strict=True)
            ):
                visit(src_item, dst_item, f"{current}[{index}]")
            return
        if isinstance(src, list):
            if not isinstance(dst, list) or len(src) != len(dst):
                raise TensorWorldContractError("R13 list structure drifted")
            for index, (src_item, dst_item) in enumerate(
                zip(src, dst, strict=True)
            ):
                visit(src_item, dst_item, f"{current}[{index}]")
            return
        if is_dataclass(src) and not isinstance(src, type):
            if not is_dataclass(dst) or type(src) is not type(dst):
                raise TensorWorldContractError("R13 dataclass structure drifted")
            for field in fields(src):
                child = f"{current}.{field.name}" if current else field.name
                visit(getattr(src, field.name), getattr(dst, field.name), child)

    visit(source, destination, path)
    return pairs


def _spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def inventory_c4_resident_transport_r13_v1(
    source_bundle: object,
    destination_bundle: object,
    source_replay: object,
    destination_replay: object,
) -> C4ResidentTransportInventoryR13V1:
    """Build a complete, alias-deduplicated field inventory at bind time."""

    pairs = _tensor_path_pairs(
        source_bundle, destination_bundle, path="bundle"
    ) + _tensor_path_pairs(source_replay, destination_replay, path="replay")
    allowed_shared = {
        f"bundle.{path}" for path in _R11_GUARDED_SHARED_TENSOR_PATHS_V1
    }
    shared: list[str] = []
    unique: dict[tuple[int, int, int], dict[str, object]] = {}
    source_owners: dict[tuple[int, int], tuple[int, int, int]] = {}
    destination_owners: dict[tuple[int, int], tuple[int, int, int]] = {}
    device: torch.device | None = None
    for path, source, destination in pairs:
        if (
            source.shape != destination.shape
            or source.dtype != destination.dtype
            or source.device != destination.device
            or not source.is_contiguous()
            or not destination.is_contiguous()
        ):
            raise TensorWorldContractError(
                f"R13 transport tensor ABI differs at {path}"
            )
        if source.numel() <= 0:
            # V2 mutation/birth fusion uses tape_capacity=0.  Empty Gaussian
            # tapes have no bytes to copy and must not recreate a host refill.
            if path.endswith(
                (
                    "python_random_gauss_primary_tape",
                    "python_random_gauss_secondary_tape",
                )
            ):
                continue
            raise TensorWorldContractError(
                f"R13 transport does not admit empty tensor {path}"
            )
        if device is None:
            device = source.device
        elif source.device != device:
            raise TensorWorldContractError(
                "R13 transport tensors must share one device"
            )
        byte_count = source.numel() * source.element_size()
        source_address = source.data_ptr()
        destination_address = destination.data_ptr()
        if source_address == destination_address:
            if path not in allowed_shared:
                raise TensorWorldContractError(
                    f"R13 unapproved canonical/shadow alias at {path}"
                )
            shared.append(path)
            continue
        if path in allowed_shared:
            raise TensorWorldContractError(
                f"R13 guarded shared policy tensor is not aliased at {path}"
            )
        if source_address % 16 != 0 or destination_address % 16 != 0:
            raise TensorWorldContractError(
                f"R13 transport tensor alignment is unsupported at {path}"
            )
        key = (source_address, destination_address, byte_count)
        source_span = (source_address, source_address + byte_count)
        destination_span = (
            destination_address,
            destination_address + byte_count,
        )
        if key in unique:
            unique[key]["paths"].append(path)  # type: ignore[union-attr]
            continue
        for span, owner in source_owners.items():
            if _spans_overlap(source_span, span):
                raise TensorWorldContractError(
                    f"R13 source alias mismatch at {path}; owner={owner}"
                )
        for span, owner in destination_owners.items():
            if _spans_overlap(destination_span, span):
                raise TensorWorldContractError(
                    f"R13 destination alias mismatch at {path}; owner={owner}"
                )
        for span in destination_owners:
            if _spans_overlap(source_span, span):
                raise TensorWorldContractError(
                    f"R13 source/destination ownership overlaps at {path}"
                )
        for span in source_owners:
            if _spans_overlap(destination_span, span):
                raise TensorWorldContractError(
                    f"R13 destination/source ownership overlaps at {path}"
                )
        unique[key] = {
            "paths": [path],
            "shape": tuple(source.shape),
            "dtype": str(source.dtype),
        }
        source_owners[source_span] = key
        destination_owners[destination_span] = key
    if device is None or not unique:
        raise TensorWorldContractError("R13 transport inventory is empty")
    if set(shared) != allowed_shared:
        raise TensorWorldContractError(
            "R13 guarded shared policy tensor coverage is incomplete"
        )
    manifest_fields = tuple(
        C4ResidentTransportFieldR13V1(
            paths=tuple(row["paths"]),  # type: ignore[arg-type]
            shape=row["shape"],  # type: ignore[arg-type]
            dtype=row["dtype"],  # type: ignore[arg-type]
            byte_count=key[2],
            source_address=key[0],
            destination_address=key[1],
        )
        for key, row in unique.items()
    )
    total_bytes = sum(field.byte_count for field in manifest_fields)
    chunk_count = sum(
        (field.byte_count + C4_RESIDENT_TRANSPORT_R13_CHUNK_BYTES_V1 - 1)
        // C4_RESIDENT_TRANSPORT_R13_CHUNK_BYTES_V1
        for field in manifest_fields
    )
    return C4ResidentTransportInventoryR13V1(
        fields=manifest_fields,
        shared_alias_paths=tuple(shared),
        total_bytes=total_bytes,
        chunk_count=chunk_count,
        device=device,
    )


def _chunk_columns(
    inventory: C4ResidentTransportInventoryR13V1,
    *,
    reverse: bool,
) -> tuple[list[int], list[int], list[int]]:
    sources: list[int] = []
    destinations: list[int] = []
    sizes: list[int] = []
    for field in inventory.fields:
        source = (
            field.destination_address if reverse else field.source_address
        )
        destination = (
            field.source_address if reverse else field.destination_address
        )
        for offset in range(
            0, field.byte_count, C4_RESIDENT_TRANSPORT_R13_CHUNK_BYTES_V1
        ):
            sources.append(source + offset)
            destinations.append(destination + offset)
            sizes.append(
                min(
                    C4_RESIDENT_TRANSPORT_R13_CHUNK_BYTES_V1,
                    field.byte_count - offset,
                )
            )
    return sources, destinations, sizes


def materialize_c4_resident_transport_plan_r13_v1(
    inventory: C4ResidentTransportInventoryR13V1,
) -> C4ResidentTransportPlanR13V1:
    """Materialize fixed device POD descriptors at the control boundary."""

    if inventory.device.type != "cuda":
        raise TensorWorldContractError("R13 native transport plan requires CUDA")
    copy_source, copy_destination, copy_sizes = _chunk_columns(
        inventory, reverse=False
    )
    publish_source, publish_destination, publish_sizes = _chunk_columns(
        inventory, reverse=True
    )
    make = lambda values: torch.tensor(  # noqa: E731
        values, dtype=torch.int64, device=inventory.device
    ).contiguous()
    plan = C4ResidentTransportPlanR13V1(
        inventory=inventory,
        copy_source_addresses=make(copy_source),
        copy_destination_addresses=make(copy_destination),
        copy_byte_counts=make(copy_sizes),
        publish_source_addresses=make(publish_source),
        publish_destination_addresses=make(publish_destination),
        publish_byte_counts=make(publish_sizes),
    )
    plan.validate()
    return plan


def load_c4_resident_transport_r13_extension(
    *, build_directory: str | Path | None = None,
) -> ModuleType:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError("R13 native transport requires CUDA")
    digest = sha256(NATIVE_SOURCE.read_bytes()).hexdigest()[:16]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c4_resident_transport_r13_{digest}",
        sources=[str(NATIVE_SOURCE)],
        build_directory=(
            str(Path(build_directory).resolve())
            if build_directory is not None
            else None
        ),
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=False,
    )
    return _MODULE


def c4_resident_transport_r13_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__),
        "native": NATIVE_SOURCE,
        "current_r11_owner": R11_OWNER_SOURCE,
    }
    return {
        "schema_version": C4_RESIDENT_TRANSPORT_R13_SCHEMA_VERSION_V1,
        "decision": C4_RESIDENT_TRANSPORT_R13_DECISION_V1,
        "copy_launches_per_tick": 1,
        "guarded_publish_launches_per_tick": 1,
        "chunk_bytes": C4_RESIDENT_TRANSPORT_R13_CHUNK_BYTES_V1,
        "threads": C4_RESIDENT_TRANSPORT_R13_THREADS_V1,
        "shared_policy_paths": sorted(_R11_GUARDED_SHARED_TENSOR_PATHS_V1),
        "alias_policy": "exact_duplicate_deduplicate_other_overlap_fail_closed",
        "rollback": "publish_kernel_writes_no_bytes_if_any_world_status_nonzero",
        "hardware_fault_atomicity_claim": False,
        "production_bound": True,
        "integration_surface": "TensorWorldC4ResidentComposedR11OwnerV1",
        "gate_claim": False,
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C4_RESIDENT_TRANSPORT_R13_DECISION_V1",
    "C4_RESIDENT_TRANSPORT_R13_SCHEMA_VERSION_V1",
    "C4ResidentTransportFieldR13V1",
    "C4ResidentTransportInventoryR13V1",
    "C4ResidentTransportPlanR13V1",
    "c4_resident_transport_r13_source_receipt_v1",
    "inventory_c4_resident_transport_r13_v1",
    "load_c4_resident_transport_r13_extension",
    "materialize_c4_resident_transport_plan_r13_v1",
]
