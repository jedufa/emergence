"""CUDA-resident mutation-off linked-inheritance identity sidecar.

The production scalar-genome kernels own the hereditary values, while this
sidecar owns the compact identity needed to order parents canonically and to
checkpoint exact linked-block ancestry.  It is deliberately independent of a
campaign, seed set, generation count, matrix shape, and checkpoint schedule.

No execution-device fallback is supported.  Source allocation and fresh-owner
restore must both name a CUDA device explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final, Mapping, Sequence

import torch

from engine.evolution.inheritance_linkage_digest_table_v1 import (
    GENOTYPE_CODE_VERSION,
    LinkageDigestTableV1,
    SCHEMA_VERSION as DIGEST_TABLE_SCHEMA_VERSION,
)


SCHEMA_VERSION: Final[str] = "eiel_cuda_architecture_b_linkage_resident_sidecar_v1"
CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_linkage_resident_sidecar_checkpoint_v1"
)
MUTATION_POLICY_VERSION: Final[str] = "mutation_off_v1"
NO_GENOTYPE_CODE: Final[int] = -1
NO_SLOT: Final[int] = -1
NO_DONOR_ORDINAL: Final[int] = 255
DONOR_STATUS_OK: Final[int] = 0

_TENSOR_NAMES: Final[tuple[str, ...]] = (
    "genotype_codes",
    "digest_rank_by_code",
    "canonical_parent_0_slots",
    "canonical_parent_1_slots",
    "child_slots",
    "donor_ordinals",
    "donor_statuses",
    "staged_counts",
    "committed_counts",
    "donor_draws_added",
    "transaction_status",
)


class CudaLinkageResidentSidecarError(ValueError):
    """Raised when resident linked-inheritance identity is not exact."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_cuda_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise CudaLinkageResidentSidecarError(
            "linked-inheritance resident owner requires an explicit CUDA device"
        )
    if not torch.cuda.is_available():
        raise CudaLinkageResidentSidecarError(
            "CUDA is unavailable; CPU fallback is prohibited"
        )
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _tensor_manifest(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.numpy().tobytes(order="C")
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _projection_digest(
    *,
    identity: Mapping[str, Any],
    shape: Mapping[str, Any],
    tensors: Mapping[str, torch.Tensor],
) -> tuple[str, dict[str, dict[str, Any]]]:
    manifests = {name: _tensor_manifest(tensors[name]) for name in _TENSOR_NAMES}
    return (
        _canonical_sha256(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity": dict(identity),
                "shape": dict(shape),
                "tensor_manifests": manifests,
            }
        ),
        manifests,
    )


@dataclass(frozen=True)
class LinkageResidentShapeV1:
    batch_cells: int
    slot_capacity: int
    birth_capacity: int
    block_count: int
    genotype_count: int

    def __post_init__(self) -> None:
        if (
            type(self.batch_cells) is not int
            or self.batch_cells < 1
            or type(self.slot_capacity) is not int
            or self.slot_capacity < 2
            or type(self.birth_capacity) is not int
            or not 1 <= self.birth_capacity <= self.slot_capacity
            or type(self.block_count) is not int
            or self.block_count < 1
            or type(self.genotype_count) is not int
            or self.genotype_count != 1 << self.block_count
        ):
            raise CudaLinkageResidentSidecarError(
                "linked-inheritance resident shape is invalid"
            )

    def payload(self) -> dict[str, int]:
        return {
            "batch_cells": self.batch_cells,
            "slot_capacity": self.slot_capacity,
            "birth_capacity": self.birth_capacity,
            "block_count": self.block_count,
            "genotype_count": self.genotype_count,
        }


def _identity(tables: Sequence[LinkageDigestTableV1]) -> dict[str, Any]:
    if not tables:
        raise CudaLinkageResidentSidecarError(
            "at least one per-cell linkage digest table is required"
        )
    first = tables[0]
    if any(
        table.registry_version != first.registry_version
        or table.parent_order_version != first.parent_order_version
        or table.genotype_code_version != first.genotype_code_version
        or table.block_ids != first.block_ids
        or table.genotype_count != first.genotype_count
        for table in tables[1:]
    ):
        raise CudaLinkageResidentSidecarError(
            "per-cell linkage tables do not share one registry ABI"
        )
    registry_binding = {
        "registry_version": first.registry_version,
        "parent_order_version": first.parent_order_version,
        "genotype_code_version": GENOTYPE_CODE_VERSION,
        "block_ids": list(first.block_ids),
    }
    return {
        "sidecar_schema_version": SCHEMA_VERSION,
        "digest_table_schema_version": DIGEST_TABLE_SCHEMA_VERSION,
        **registry_binding,
        "registry_binding_sha256": _canonical_sha256(registry_binding),
        "mutation_policy_version": MUTATION_POLICY_VERSION,
        "table_sha256_by_cell": [table.table_sha256 for table in tables],
    }


def _normalize_tables(
    tables: LinkageDigestTableV1 | Sequence[LinkageDigestTableV1],
    *,
    batch_cells: int,
) -> tuple[LinkageDigestTableV1, ...]:
    if isinstance(tables, LinkageDigestTableV1):
        resolved = (tables,) * batch_cells
    else:
        resolved = tuple(tables)
    if len(resolved) != batch_cells or any(
        not isinstance(table, LinkageDigestTableV1) for table in resolved
    ):
        raise CudaLinkageResidentSidecarError(
            "one hash-bound linkage digest table is required per cell"
        )
    _identity(resolved)
    return resolved


@dataclass(slots=True)
class CudaLinkageResidentSidecarV1:
    """Typed CUDA owner for compact linkage identity and birth-row telemetry."""

    tables: tuple[LinkageDigestTableV1, ...]
    shape: LinkageResidentShapeV1
    genotype_codes: torch.Tensor
    digest_rank_by_code: torch.Tensor
    canonical_parent_0_slots: torch.Tensor
    canonical_parent_1_slots: torch.Tensor
    child_slots: torch.Tensor
    donor_ordinals: torch.Tensor
    donor_statuses: torch.Tensor
    staged_counts: torch.Tensor
    committed_counts: torch.Tensor
    donor_draws_added: torch.Tensor
    transaction_status: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.genotype_codes.device

    @property
    def identity(self) -> dict[str, Any]:
        return _identity(self.tables)

    def _tensor_map(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in _TENSOR_NAMES}

    def validate_exact(
        self,
        *,
        allowed_failure_cell_indices: tuple[int, ...] = (),
    ) -> None:
        """Validate shape, device, table binding, and resident value domains."""

        if (
            not isinstance(allowed_failure_cell_indices, tuple)
            or any(
                type(cell) is not int
                or cell < 0
                or cell >= self.shape.batch_cells
                for cell in allowed_failure_cell_indices
            )
            or len(set(allowed_failure_cell_indices))
            != len(allowed_failure_cell_indices)
        ):
            raise CudaLinkageResidentSidecarError(
                "allowed failure cells must be unique in-range indices"
            )

        expected = {
            "genotype_codes": (
                torch.int64,
                (self.shape.batch_cells, self.shape.slot_capacity),
            ),
            "digest_rank_by_code": (
                torch.int64,
                (self.shape.batch_cells, self.shape.genotype_count),
            ),
            "canonical_parent_0_slots": (
                torch.int64,
                (self.shape.batch_cells, self.shape.birth_capacity),
            ),
            "canonical_parent_1_slots": (
                torch.int64,
                (self.shape.batch_cells, self.shape.birth_capacity),
            ),
            "child_slots": (
                torch.int64,
                (self.shape.batch_cells, self.shape.birth_capacity),
            ),
            "donor_ordinals": (
                torch.uint8,
                (
                    self.shape.batch_cells,
                    self.shape.birth_capacity,
                    self.shape.block_count,
                ),
            ),
            "donor_statuses": (
                torch.uint8,
                (
                    self.shape.batch_cells,
                    self.shape.birth_capacity,
                    self.shape.block_count,
                ),
            ),
            "staged_counts": (torch.int64, (self.shape.batch_cells,)),
            "committed_counts": (torch.int64, (self.shape.batch_cells,)),
            "donor_draws_added": (torch.int64, (self.shape.batch_cells,)),
            "transaction_status": (torch.int32, (self.shape.batch_cells,)),
        }
        tensors = self._tensor_map()
        devices = {value.device for value in tensors.values()}
        if (
            len(devices) != 1
            or next(iter(devices)).type != "cuda"
            or any(
                not isinstance(tensor, torch.Tensor)
                or tensor.dtype != expected[name][0]
                or tuple(tensor.shape) != expected[name][1]
                or tensor.requires_grad
                for name, tensor in tensors.items()
            )
        ):
            raise CudaLinkageResidentSidecarError(
                "resident tensor shape, dtype, or CUDA ownership is invalid"
            )
        if len(self.tables) != self.shape.batch_cells or any(
            self.shape.block_count != table.block_count
            or self.shape.genotype_count != table.genotype_count
            for table in self.tables
        ):
            raise CudaLinkageResidentSidecarError(
                "resident shape contradicts the linkage digest table"
            )
        expected_ranks = torch.tensor(
            [table.rank_by_code for table in self.tables],
            dtype=torch.int64,
            device=self.device,
        )
        if not torch.equal(self.digest_rank_by_code, expected_ranks):
            raise CudaLinkageResidentSidecarError(
                "resident digest ranks contradict the hash-bound table"
            )
        if bool(
            (
                (self.genotype_codes < NO_GENOTYPE_CODE)
                | (self.genotype_codes >= self.shape.genotype_count)
            )
            .any()
            .item()
        ):
            raise CudaLinkageResidentSidecarError(
                "resident genotype code is outside the digest table"
            )
        for name in ("staged_counts", "committed_counts"):
            counts = tensors[name]
            if bool(((counts < 0) | (counts > self.shape.birth_capacity)).any().item()):
                raise CudaLinkageResidentSidecarError(
                    f"{name} is outside the birth-row capacity"
                )
        if bool((self.committed_counts > self.staged_counts).any().item()):
            raise CudaLinkageResidentSidecarError(
                "committed birth count exceeds staged count"
            )
        active = (
            torch.arange(self.shape.birth_capacity, device=self.device)[None, :]
            < self.staged_counts[:, None]
        )
        for slots in (
            self.canonical_parent_0_slots,
            self.canonical_parent_1_slots,
        ):
            if bool(
                (active & ((slots < 0) | (slots >= self.shape.slot_capacity)))
                .any()
                .item()
            ) or bool(((~active) & (slots != NO_SLOT)).any().item()):
                raise CudaLinkageResidentSidecarError(
                    "canonical parent birth rows are invalid"
                )
        safe_parent_0 = self.canonical_parent_0_slots.clamp(min=0)
        safe_parent_1 = self.canonical_parent_1_slots.clamp(min=0)
        parent_0_codes = self.genotype_codes.gather(1, safe_parent_0)
        parent_1_codes = self.genotype_codes.gather(1, safe_parent_1)
        parent_0_ranks = torch.gather(
            self.digest_rank_by_code, 1, parent_0_codes.clamp(min=0)
        )
        parent_1_ranks = torch.gather(
            self.digest_rank_by_code, 1, parent_1_codes.clamp(min=0)
        )
        if bool(
            (
                active
                & (
                    (parent_0_codes < 0)
                    | (parent_1_codes < 0)
                    | (parent_0_ranks > parent_1_ranks)
                )
            )
            .any()
            .item()
        ):
            raise CudaLinkageResidentSidecarError(
                "canonical parent rows contradict resident digest order"
            )
        committed = (
            torch.arange(self.shape.birth_capacity, device=self.device)[None, :]
            < self.committed_counts[:, None]
        )
        committed_groups = committed[:, :, None]
        if bool(
            (
                committed
                & (
                    (self.child_slots < 0)
                    | (self.child_slots >= self.shape.slot_capacity)
                )
            )
            .any()
            .item()
        ) or bool(((~committed) & (self.child_slots != NO_SLOT)).any().item()):
            raise CudaLinkageResidentSidecarError(
                "committed child birth rows are invalid"
            )
        if bool(
            (
                committed_groups
                & ((self.donor_ordinals != 0) & (self.donor_ordinals != 1))
            )
            .any()
            .item()
        ) or bool(
            ((~committed_groups) & (self.donor_ordinals != NO_DONOR_ORDINAL))
            .any()
            .item()
        ):
            raise CudaLinkageResidentSidecarError("donor ordinal rows are invalid")
        if bool((self.donor_statuses != DONOR_STATUS_OK).any().item()):
            raise CudaLinkageResidentSidecarError(
                "linked donor status is not exact-success"
            )
        if not torch.equal(
            self.donor_draws_added,
            self.committed_counts * self.shape.block_count,
        ):
            raise CudaLinkageResidentSidecarError(
                "linked donor draw accounting is invalid"
            )
        failed_transactions = self.transaction_status != 0
        if allowed_failure_cell_indices:
            allowed = torch.zeros(
                self.shape.batch_cells,
                dtype=torch.bool,
                device=self.device,
            )
            allowed[list(allowed_failure_cell_indices)] = True
            failed_transactions = failed_transactions & ~allowed
        if bool(failed_transactions.any().item()):
            raise CudaLinkageResidentSidecarError(
                "resident linkage transaction status is not successful"
            )
        bit_positions = torch.arange(
            self.shape.block_count, dtype=torch.int64, device=self.device
        )
        expected_codes = torch.sum(
            torch.where(
                self.donor_ordinals.to(torch.bool),
                (parent_1_codes[:, :, None] >> bit_positions) & 1,
                (parent_0_codes[:, :, None] >> bit_positions) & 1,
            )
            << bit_positions,
            dim=2,
        )
        observed_codes = self.genotype_codes.gather(1, self.child_slots.clamp(min=0))
        if bool((committed & (observed_codes != expected_codes)).any().item()):
            raise CudaLinkageResidentSidecarError(
                "committed child code contradicts its linked donor rows"
            )

    def project_digest_ranks(self) -> torch.Tensor:
        """Return a CUDA rank plane aligned with resident agent slots."""

        safe = self.genotype_codes.clamp(min=0)
        projected = torch.gather(self.digest_rank_by_code, 1, safe)
        return torch.where(
            self.genotype_codes >= 0,
            projected,
            torch.full_like(projected, -1),
        )

    def stage_canonical_parents(
        self,
        *,
        parent_slots: torch.Tensor,
        mate_slots: torch.Tensor,
        birth_counts: torch.Tensor,
    ) -> None:
        """Stage digest-ordered parent slots for birth-ordered native rows."""

        expected_rows = (
            self.shape.batch_cells,
            self.shape.birth_capacity,
        )
        for name, value in (
            ("parent_slots", parent_slots),
            ("mate_slots", mate_slots),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.device != self.device
                or value.dtype != torch.int64
                or tuple(value.shape) != expected_rows
            ):
                raise CudaLinkageResidentSidecarError(
                    f"{name} must be an exact resident CUDA int64 birth plane"
                )
        if (
            not isinstance(birth_counts, torch.Tensor)
            or birth_counts.device != self.device
            or birth_counts.dtype != torch.int64
            or tuple(birth_counts.shape) != (self.shape.batch_cells,)
            or bool(
                ((birth_counts < 0) | (birth_counts > self.shape.birth_capacity))
                .any()
                .item()
            )
        ):
            raise CudaLinkageResidentSidecarError(
                "birth counts are not an exact resident CUDA count plane"
            )
        rows = torch.arange(self.shape.birth_capacity, device=self.device)[None, :]
        active = rows < birth_counts[:, None]
        invalid_slots = (
            (parent_slots < 0)
            | (parent_slots >= self.shape.slot_capacity)
            | (mate_slots < 0)
            | (mate_slots >= self.shape.slot_capacity)
            | (parent_slots == mate_slots)
        )
        if bool((active & invalid_slots).any().item()):
            raise CudaLinkageResidentSidecarError("active parent pair is invalid")
        safe_parent = torch.where(active, parent_slots, torch.zeros_like(parent_slots))
        safe_mate = torch.where(active, mate_slots, torch.zeros_like(mate_slots))
        parent_codes = self.genotype_codes.gather(1, safe_parent)
        mate_codes = self.genotype_codes.gather(1, safe_mate)
        if bool(
            (
                active
                & (
                    (parent_codes < 0)
                    | (mate_codes < 0)
                    | (parent_codes >= self.shape.genotype_count)
                    | (mate_codes >= self.shape.genotype_count)
                )
            )
            .any()
            .item()
        ):
            raise CudaLinkageResidentSidecarError(
                "active parent has no valid resident genotype code"
            )
        parent_ranks = torch.gather(
            self.digest_rank_by_code, 1, parent_codes.clamp(min=0)
        )
        mate_ranks = torch.gather(self.digest_rank_by_code, 1, mate_codes.clamp(min=0))
        parent_first = parent_ranks <= mate_ranks
        canonical_0 = torch.where(parent_first, parent_slots, mate_slots)
        canonical_1 = torch.where(parent_first, mate_slots, parent_slots)
        inactive = torch.full_like(canonical_0, NO_SLOT)

        self.canonical_parent_0_slots.copy_(torch.where(active, canonical_0, inactive))
        self.canonical_parent_1_slots.copy_(torch.where(active, canonical_1, inactive))
        self.child_slots.fill_(NO_SLOT)
        self.staged_counts.copy_(birth_counts)
        self.committed_counts.zero_()
        self.donor_draws_added.zero_()
        self.transaction_status.zero_()
        self.donor_ordinals.fill_(NO_DONOR_ORDINAL)
        self.donor_statuses.fill_(DONOR_STATUS_OK)

    def retire_slots(
        self,
        *,
        slots: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        """Retire turnover slots before reusing them for child commits."""

        expected = (self.shape.batch_cells, self.shape.birth_capacity)
        if (
            not isinstance(slots, torch.Tensor)
            or slots.device != self.device
            or slots.dtype != torch.int64
            or tuple(slots.shape) != expected
            or not isinstance(counts, torch.Tensor)
            or counts.device != self.device
            or counts.dtype != torch.int64
            or tuple(counts.shape) != (self.shape.batch_cells,)
            or bool(((counts < 0) | (counts > self.shape.birth_capacity)).any().item())
        ):
            raise CudaLinkageResidentSidecarError(
                "retirement rows are not exact resident CUDA tensors"
            )
        active = (
            torch.arange(self.shape.birth_capacity, device=self.device)[None, :]
            < counts[:, None]
        )
        if bool(
            (active & ((slots < 0) | (slots >= self.shape.slot_capacity))).any().item()
        ):
            raise CudaLinkageResidentSidecarError(
                "retirement slot is outside resident capacity"
            )
        for cell in range(self.shape.batch_cells):
            count = int(counts[cell].item())
            if count:
                selected = slots[cell, :count]
                if int(torch.unique(selected).numel()) != count:
                    raise CudaLinkageResidentSidecarError(
                        "retirement slots must be unique within each cell"
                    )
        for cell in range(self.shape.batch_cells):
            count = int(counts[cell].item())
            if count:
                self.genotype_codes[cell, slots[cell, :count]] = NO_GENOTYPE_CODE

    def commit_linked_births(
        self,
        *,
        child_slots: torch.Tensor,
        donor_ordinals: torch.Tensor,
        donor_statuses: torch.Tensor,
        committed_counts: torch.Tensor,
        donor_draws_added: torch.Tensor,
        transaction_status: torch.Tensor,
    ) -> None:
        """Atomically commit exact mutation-off linked child genotype codes."""

        row_shape = (self.shape.batch_cells, self.shape.birth_capacity)
        group_shape = (*row_shape, self.shape.block_count)
        exact = (
            isinstance(child_slots, torch.Tensor)
            and child_slots.device == self.device
            and child_slots.dtype == torch.int64
            and tuple(child_slots.shape) == row_shape
            and isinstance(donor_ordinals, torch.Tensor)
            and donor_ordinals.device == self.device
            and donor_ordinals.dtype == torch.uint8
            and tuple(donor_ordinals.shape) == group_shape
            and isinstance(donor_statuses, torch.Tensor)
            and donor_statuses.device == self.device
            and donor_statuses.dtype == torch.uint8
            and tuple(donor_statuses.shape) == group_shape
            and isinstance(committed_counts, torch.Tensor)
            and committed_counts.device == self.device
            and committed_counts.dtype == torch.int64
            and tuple(committed_counts.shape) == (self.shape.batch_cells,)
            and isinstance(donor_draws_added, torch.Tensor)
            and donor_draws_added.device == self.device
            and donor_draws_added.dtype == torch.int64
            and tuple(donor_draws_added.shape) == (self.shape.batch_cells,)
            and isinstance(transaction_status, torch.Tensor)
            and transaction_status.device == self.device
            and transaction_status.dtype == torch.int32
            and tuple(transaction_status.shape) == (self.shape.batch_cells,)
        )
        if not exact:
            raise CudaLinkageResidentSidecarError(
                "linked birth commit tensors do not match the resident ABI"
            )
        if bool(
            ((committed_counts < 0) | (committed_counts > self.staged_counts))
            .any()
            .item()
        ):
            raise CudaLinkageResidentSidecarError(
                "linked committed count contradicts staged rows"
            )
        if bool((transaction_status != 0).any().item()):
            raise CudaLinkageResidentSidecarError(
                "native linked inheritance transaction did not succeed"
            )
        expected_draws = committed_counts * self.shape.block_count
        if not torch.equal(donor_draws_added, expected_draws):
            raise CudaLinkageResidentSidecarError(
                "native linked donor draw accounting is invalid"
            )
        active = (
            torch.arange(self.shape.birth_capacity, device=self.device)[None, :]
            < committed_counts[:, None]
        )
        active_groups = active[:, :, None]
        if bool(
            (active & ((child_slots < 0) | (child_slots >= self.shape.slot_capacity)))
            .any()
            .item()
        ):
            raise CudaLinkageResidentSidecarError(
                "active child slot is outside resident capacity"
            )
        if bool(
            (active_groups & ((donor_ordinals != 0) & (donor_ordinals != 1)))
            .any()
            .item()
        ):
            raise CudaLinkageResidentSidecarError("active donor ordinal is not binary")
        if bool((active_groups & (donor_statuses != DONOR_STATUS_OK)).any().item()):
            raise CudaLinkageResidentSidecarError(
                "native linked donor selection did not succeed"
            )
        safe_children = torch.where(active, child_slots, torch.zeros_like(child_slots))
        if bool(
            (
                active
                & (self.genotype_codes.gather(1, safe_children) != NO_GENOTYPE_CODE)
            )
            .any()
            .item()
        ):
            raise CudaLinkageResidentSidecarError(
                "child slot must be retired before linked commit"
            )
        for cell in range(self.shape.batch_cells):
            count = int(committed_counts[cell].item())
            if count:
                selected = child_slots[cell, :count]
                if int(torch.unique(selected).numel()) != count:
                    raise CudaLinkageResidentSidecarError(
                        "child slots must be unique within each cell"
                    )

        safe_parent_0 = self.canonical_parent_0_slots.clamp(min=0)
        safe_parent_1 = self.canonical_parent_1_slots.clamp(min=0)
        code_0 = self.genotype_codes.gather(1, safe_parent_0)
        code_1 = self.genotype_codes.gather(1, safe_parent_1)
        bit_positions = torch.arange(
            self.shape.block_count, dtype=torch.int64, device=self.device
        )
        alleles_0 = (code_0[:, :, None] >> bit_positions) & 1
        alleles_1 = (code_1[:, :, None] >> bit_positions) & 1
        selected = torch.where(donor_ordinals.to(torch.bool), alleles_1, alleles_0)
        child_codes = torch.sum(selected << bit_positions, dim=2)
        if bool(
            (active & ((child_codes < 0) | (child_codes >= self.shape.genotype_count)))
            .any()
            .item()
        ):
            raise CudaLinkageResidentSidecarError(
                "computed child code is outside the digest table"
            )

        # All validation precedes the first authoritative mutation.
        for cell in range(self.shape.batch_cells):
            count = int(committed_counts[cell].item())
            if count:
                self.genotype_codes[cell, child_slots[cell, :count]] = child_codes[
                    cell, :count
                ]
        self.donor_ordinals.copy_(
            torch.where(
                active_groups,
                donor_ordinals,
                torch.full_like(donor_ordinals, NO_DONOR_ORDINAL),
            )
        )
        self.donor_statuses.fill_(DONOR_STATUS_OK)
        self.child_slots.copy_(
            torch.where(
                active,
                child_slots,
                torch.full_like(child_slots, NO_SLOT),
            )
        )
        self.committed_counts.copy_(committed_counts)
        self.donor_draws_added.copy_(donor_draws_added)
        self.transaction_status.copy_(transaction_status)
        self.validate_exact()

    def export_checkpoint_projection(
        self,
        *,
        allowed_failure_cell_indices: tuple[int, ...] = (),
    ) -> dict[str, Any]:
        """Export a hash-bound CPU checkpoint projection for a fresh owner."""

        self.validate_exact(
            allowed_failure_cell_indices=allowed_failure_cell_indices
        )
        tensors = {
            name: value.detach().to(device="cpu").clone()
            for name, value in self._tensor_map().items()
        }
        digest, manifests = _projection_digest(
            identity=self.identity,
            shape=self.shape.payload(),
            tensors=tensors,
        )
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "identity": self.identity,
            "shape": self.shape.payload(),
            "tensors": tensors,
            "tensor_manifests": manifests,
            "projection_sha256": digest,
        }

    @classmethod
    def from_checkpoint_projection(
        cls,
        tables: LinkageDigestTableV1 | Sequence[LinkageDigestTableV1],
        projection: Mapping[str, Any],
        *,
        device: str | torch.device,
        allowed_failure_cell_indices: tuple[int, ...] = (),
    ) -> "CudaLinkageResidentSidecarV1":
        """Restore onto a new CUDA owner, rejecting stale identity or bytes."""

        resolved = _require_cuda_device(device)
        raw_shape = projection.get("shape") if isinstance(projection, Mapping) else None
        try:
            batch_cells = int(raw_shape["batch_cells"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CudaLinkageResidentSidecarError(
                "checkpoint projection has no valid cell count"
            ) from exc
        resolved_tables = _normalize_tables(tables, batch_cells=batch_cells)
        if (
            not isinstance(projection, Mapping)
            or projection.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or projection.get("identity") != _identity(resolved_tables)
            or not isinstance(projection.get("shape"), Mapping)
            or not isinstance(projection.get("tensors"), Mapping)
        ):
            raise CudaLinkageResidentSidecarError(
                "checkpoint identity does not match the hash-bound table"
            )
        try:
            shape = LinkageResidentShapeV1(
                **{
                    name: projection["shape"][name]
                    for name in (
                        "batch_cells",
                        "slot_capacity",
                        "birth_capacity",
                        "block_count",
                        "genotype_count",
                    )
                }
            )
            source = {name: projection["tensors"][name] for name in _TENSOR_NAMES}
        except (KeyError, TypeError) as exc:
            raise CudaLinkageResidentSidecarError(
                "checkpoint projection is incomplete"
            ) from exc
        if any(
            not isinstance(value, torch.Tensor) or value.device.type != "cpu"
            for value in source.values()
        ):
            raise CudaLinkageResidentSidecarError(
                "checkpoint tensors must be explicit CPU transport values"
            )
        digest, manifests = _projection_digest(
            identity=_identity(resolved_tables),
            shape=shape.payload(),
            tensors=source,
        )
        if (
            projection.get("tensor_manifests") != manifests
            or projection.get("projection_sha256") != digest
        ):
            raise CudaLinkageResidentSidecarError("checkpoint projection hash mismatch")
        owner = cls(
            tables=resolved_tables,
            shape=shape,
            **{
                name: value.to(device=resolved).clone()
                for name, value in source.items()
            },
        )
        owner.validate_exact(
            allowed_failure_cell_indices=allowed_failure_cell_indices
        )
        return owner


def allocate_cuda_linkage_resident_sidecar_v1(
    tables: LinkageDigestTableV1 | Sequence[LinkageDigestTableV1],
    *,
    source_genotype_codes: torch.Tensor,
    birth_capacity: int,
    device: str | torch.device,
) -> CudaLinkageResidentSidecarV1:
    """Allocate a source-bound CUDA sidecar without an execution fallback."""

    resolved = _require_cuda_device(device)
    if (
        not isinstance(source_genotype_codes, torch.Tensor)
        or source_genotype_codes.device != resolved
        or source_genotype_codes.dtype != torch.int64
        or source_genotype_codes.ndim != 2
        or source_genotype_codes.requires_grad
    ):
        raise CudaLinkageResidentSidecarError(
            "source genotype codes must already be a CUDA int64 [cell, slot] tensor"
        )
    batch_cells, slot_capacity = source_genotype_codes.shape
    resolved_tables = _normalize_tables(tables, batch_cells=int(batch_cells))
    first_table = resolved_tables[0]
    shape = LinkageResidentShapeV1(
        batch_cells=int(batch_cells),
        slot_capacity=int(slot_capacity),
        birth_capacity=birth_capacity,
        block_count=first_table.block_count,
        genotype_count=first_table.genotype_count,
    )
    row_shape = (shape.batch_cells, shape.birth_capacity)
    owner = CudaLinkageResidentSidecarV1(
        tables=resolved_tables,
        shape=shape,
        genotype_codes=source_genotype_codes.clone(),
        digest_rank_by_code=torch.tensor(
            [table.rank_by_code for table in resolved_tables],
            dtype=torch.int64,
            device=resolved,
        ),
        canonical_parent_0_slots=torch.full(
            row_shape, NO_SLOT, dtype=torch.int64, device=resolved
        ),
        canonical_parent_1_slots=torch.full(
            row_shape, NO_SLOT, dtype=torch.int64, device=resolved
        ),
        child_slots=torch.full(row_shape, NO_SLOT, dtype=torch.int64, device=resolved),
        donor_ordinals=torch.full(
            (*row_shape, shape.block_count),
            NO_DONOR_ORDINAL,
            dtype=torch.uint8,
            device=resolved,
        ),
        donor_statuses=torch.full(
            (*row_shape, shape.block_count),
            DONOR_STATUS_OK,
            dtype=torch.uint8,
            device=resolved,
        ),
        staged_counts=torch.zeros(
            shape.batch_cells, dtype=torch.int64, device=resolved
        ),
        committed_counts=torch.zeros(
            shape.batch_cells, dtype=torch.int64, device=resolved
        ),
        donor_draws_added=torch.zeros(
            shape.batch_cells, dtype=torch.int64, device=resolved
        ),
        transaction_status=torch.zeros(
            shape.batch_cells, dtype=torch.int32, device=resolved
        ),
    )
    owner.validate_exact()
    return owner


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CudaLinkageResidentSidecarError",
    "CudaLinkageResidentSidecarV1",
    "DONOR_STATUS_OK",
    "LinkageResidentShapeV1",
    "MUTATION_POLICY_VERSION",
    "NO_DONOR_ORDINAL",
    "NO_GENOTYPE_CODE",
    "NO_SLOT",
    "SCHEMA_VERSION",
    "allocate_cuda_linkage_resident_sidecar_v1",
]
