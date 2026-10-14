"""Resident CUDA SoA and checkpoint contract for diploid scalar inheritance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final, Mapping

import torch

from evolution.diploid_inheritance_v1 import (
    DIPLOID_EXPRESSION_PROFILE,
    DIPLOID_GENOME_SCHEMA_VERSION,
    DIPLOID_INHERITANCE_PROFILE,
    DIPLOID_LINKAGE_BLOCKS,
    DIPLOID_MUTATION_PROFILE,
    DIPLOID_RECOMBINATION_PROFILE,
    DIPLOID_SCALAR_LOCI,
    LINKAGE_MAP_SHA256,
    LOCUS_REGISTRY_SHA256,
)


STATE_BANK_SCHEMA_VERSION: Final[str] = "cuda_diploid_state_bank_v2"
CHECKPOINT_SCHEMA_VERSION: Final[str] = "cuda_diploid_checkpoint_projection_v2"
ALLELE_PLANE_COUNT: Final[int] = 2 * len(DIPLOID_SCALAR_LOCI)
CANONICAL_HEREDITARY_DIGEST_BYTES: Final[int] = 32
CANONICAL_DIGEST_TEMPLATE_PACK_BYTES: Final[int] = 8448
INCREMENTAL_ARRAY_COUNT: Final[int] = ALLELE_PLANE_COUNT + 5

PHENOTYPE_SOURCE_FIELDS: Final[dict[str, str]] = {
    "exploration_bias": "genome_exploration_bias",
    "aggression_bias": "genome_aggression_bias",
    "action_temperature": "genome_action_temperature",
    "drive_inertia": "genome_drive_inertia",
    "drive_noise": "genome_drive_noise",
    "drive_logit_gain": "genome_drive_logit_gain",
    "pred_lr": "genome_pred_lr",
    "drive_sp_hunger": "genome_drive_sp_hunger",
    "drive_sp_pain": "genome_drive_sp_pain",
    "drive_sp_fatigue": "genome_drive_sp_fatigue",
    "drive_sp_fear": "genome_drive_sp_fear",
    "exp_prior_strength": "genome_exp_prior_strength",
    "exp_prior_noise": "genome_exp_prior_noise",
    "drive_birth_noise": "genome_drive_birth_noise",
    "reproduction_threshold": "genome_reproduction_threshold",
    "mutation_rate": "genome_mutation_rate",
    "resource_handling_axis": "genome_resource_handling_axis",
}

ALLELE_FIELD_NAMES: Final[tuple[str, ...]] = tuple(
    f"diploid_allele_{homolog}_{locus}"
    for homolog in range(2)
    for locus in DIPLOID_SCALAR_LOCI
)
CHECKPOINT_TENSOR_NAMES: Final[tuple[str, ...]] = (
    *ALLELE_FIELD_NAMES,
    "genome_resource_handling_axis",
    "canonical_hereditary_digest",
    "canonical_hereditary_template",
    "diploid_payload_template",
    "diploid_genotype_template",
)


class CudaDiploidStateBankError(ValueError):
    """Raised when the resident diploid bank is incomplete or incompatible."""


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


def checkpoint_metadata_v1(*, cells: int, capacity: int) -> dict[str, Any]:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state_bank_schema_version": STATE_BANK_SCHEMA_VERSION,
        "inheritance_profile": DIPLOID_INHERITANCE_PROFILE,
        "genome_schema_version": DIPLOID_GENOME_SCHEMA_VERSION,
        "expression_profile": DIPLOID_EXPRESSION_PROFILE,
        "recombination_profile": DIPLOID_RECOMBINATION_PROFILE,
        "mutation_profile": DIPLOID_MUTATION_PROFILE,
        "linkage_map_sha256": LINKAGE_MAP_SHA256,
        "admitted_locus_sha256": LOCUS_REGISTRY_SHA256,
        "loci": list(DIPLOID_SCALAR_LOCI),
        "linkage_blocks": [
            [block_id, list(loci)]
            for block_id, loci in DIPLOID_LINKAGE_BLOCKS
        ],
        "allele_array_count": ALLELE_PLANE_COUNT,
        "incremental_array_count": INCREMENTAL_ARRAY_COUNT,
        "canonical_hereditary_digest_bytes": CANONICAL_HEREDITARY_DIGEST_BYTES,
        "canonical_hereditary_digest_dtype": "torch.uint8",
        "canonical_parent_order_version": (
            "canonical_hereditary_payload_sha256_v1"
        ),
        "dtype": "torch.float32",
        "cells": int(cells),
        "capacity": int(capacity),
        "layout": "homolog_locus_cell_slot_soa_v1",
        "migration": "x_to_[x,x]_zero_draws",
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_metadata(
    value: Any, *, cells: int | None = None, capacity: int | None = None
) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise CudaDiploidStateBankError("diploid checkpoint metadata is missing")
    metadata = dict(value)
    observed_cells = metadata.get("cells")
    observed_capacity = metadata.get("capacity")
    if (
        type(observed_cells) is not int
        or observed_cells < 1
        or type(observed_capacity) is not int
        or observed_capacity < 1
    ):
        raise CudaDiploidStateBankError(
            "diploid checkpoint dimensions are invalid"
        )
    expected = checkpoint_metadata_v1(
        cells=observed_cells, capacity=observed_capacity
    )
    if metadata != expected:
        raise CudaDiploidStateBankError(
            "diploid checkpoint contract or hash mismatch"
        )
    if cells is not None and observed_cells != cells:
        raise CudaDiploidStateBankError("diploid checkpoint cell count mismatch")
    if capacity is not None and observed_capacity != capacity:
        raise CudaDiploidStateBankError("diploid checkpoint capacity mismatch")
    return observed_cells, observed_capacity


def _require_phenotype_tensor(
    state: Mapping[str, Any],
    *,
    name: str,
    cells: int,
    capacity: int,
    device: torch.device,
) -> torch.Tensor:
    value = state.get(name)
    if not isinstance(value, torch.Tensor):
        raise CudaDiploidStateBankError(
            f"resident phenotype tensor is missing: {name}"
        )
    if (
        value.dtype != torch.float32
        or tuple(value.shape) != (cells, capacity)
        or value.device.type != device.type
        or (
            device.index is not None
            and value.device.index != device.index
        )
        or not value.is_contiguous()
        or not bool(torch.isfinite(value).all().item())
    ):
        raise CudaDiploidStateBankError(
            f"resident phenotype tensor is incompatible: {name}"
        )
    return value


@dataclass(frozen=True, slots=True)
class CudaDiploidStateBankV1:
    """Resident diploid alleles, phenotype, and canonical hereditary identity."""

    storage: torch.Tensor
    genome_resource_handling_axis: torch.Tensor
    canonical_hereditary_digest: torch.Tensor
    canonical_hereditary_template: torch.Tensor
    diploid_payload_template: torch.Tensor
    diploid_genotype_template: torch.Tensor
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.storage, torch.Tensor):
            raise CudaDiploidStateBankError("diploid storage must be a tensor")
        if (
            self.storage.dtype != torch.float32
            or self.storage.ndim != 3
            or self.storage.size(1) != ALLELE_PLANE_COUNT
            or not self.storage.is_contiguous()
        ):
            raise CudaDiploidStateBankError(
                "diploid storage must be contiguous [cells,34,capacity] float32"
            )
        cells, _planes, capacity = self.storage.shape
        if (
            not isinstance(self.genome_resource_handling_axis, torch.Tensor)
            or self.genome_resource_handling_axis.dtype != torch.float32
            or tuple(self.genome_resource_handling_axis.shape)
            != (cells, capacity)
            or self.genome_resource_handling_axis.device != self.storage.device
            or not self.genome_resource_handling_axis.is_contiguous()
        ):
            raise CudaDiploidStateBankError(
                "resource-handling phenotype array is incompatible"
            )
        if (
            not isinstance(self.canonical_hereditary_digest, torch.Tensor)
            or self.canonical_hereditary_digest.dtype != torch.uint8
            or tuple(self.canonical_hereditary_digest.shape)
            != (cells, capacity, CANONICAL_HEREDITARY_DIGEST_BYTES)
            or self.canonical_hereditary_digest.device != self.storage.device
            or not self.canonical_hereditary_digest.is_contiguous()
        ):
            raise CudaDiploidStateBankError(
                "canonical hereditary digest must be contiguous "
                "uint8[cells,capacity,32]"
            )
        for name, template in (
            ("canonical hereditary", self.canonical_hereditary_template),
            ("diploid payload", self.diploid_payload_template),
            ("diploid genotype", self.diploid_genotype_template),
        ):
            if (
                not isinstance(template, torch.Tensor)
                or template.dtype != torch.uint8
                or template.ndim != 2
                or template.size(0) != cells
                or template.size(1) < 1
                or template.device != self.storage.device
                or not template.is_contiguous()
            ):
                raise CudaDiploidStateBankError(
                    f"{name} template must be contiguous uint8[cells,bytes]"
                )
        _validate_metadata(
            self.metadata, cells=int(cells), capacity=int(capacity)
        )

    @property
    def cells(self) -> int:
        return int(self.storage.size(0))

    @property
    def capacity(self) -> int:
        return int(self.storage.size(2))

    @property
    def device(self) -> torch.device:
        return self.storage.device

    @property
    def incremental_nbytes(self) -> int:
        return int(
            self.storage.numel() * self.storage.element_size()
            + self.genome_resource_handling_axis.numel()
            * self.genome_resource_handling_axis.element_size()
            + self.canonical_hereditary_digest.numel()
            * self.canonical_hereditary_digest.element_size()
            + self.canonical_hereditary_template.numel()
            * self.canonical_hereditary_template.element_size()
            + self.diploid_payload_template.numel()
            * self.diploid_payload_template.element_size()
            + self.diploid_genotype_template.numel()
            * self.diploid_genotype_template.element_size()
        )

    def allele_plane(self, homolog: int, locus: str) -> torch.Tensor:
        if type(homolog) is not int or homolog not in {0, 1}:
            raise CudaDiploidStateBankError("homolog must be 0 or 1")
        try:
            locus_index = DIPLOID_SCALAR_LOCI.index(str(locus))
        except ValueError as exc:
            raise CudaDiploidStateBankError(
                f"unknown diploid locus: {locus!r}"
            ) from exc
        return self.storage[:, homolog * len(DIPLOID_SCALAR_LOCI) + locus_index, :]

    def named_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            f"diploid_allele_{homolog}_{locus}": self.allele_plane(
                homolog, locus
            )
            for homolog in range(2)
            for locus in DIPLOID_SCALAR_LOCI
        }
        tensors["genome_resource_handling_axis"] = (
            self.genome_resource_handling_axis
        )
        tensors["canonical_hereditary_digest"] = (
            self.canonical_hereditary_digest
        )
        tensors["canonical_hereditary_template"] = (
            self.canonical_hereditary_template
        )
        tensors["diploid_payload_template"] = self.diploid_payload_template
        tensors["diploid_genotype_template"] = (
            self.diploid_genotype_template
        )
        return tensors

    def checkpoint_projection(self) -> dict[str, Any]:
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "metadata": dict(self.metadata),
            "tensors": {
                name: tensor.detach().cpu().contiguous().clone()
                for name, tensor in self.named_tensors().items()
            },
        }


def pack_canonical_digest_templates_v1(
    bank: CudaDiploidStateBankV1,
) -> torch.Tensor:
    """Pack immutable digest templates behind the resident CUDA ABI.

    The first eight bytes are retained as a reserved prefix for the fixed
    scratch layout.  Canonical digest state is passed through the explicit
    resident birth-state pointer; no raw device pointer is serialized into
    scratch metadata.
    """
    templates = (
        bank.canonical_hereditary_template,
        bank.diploid_payload_template,
        bank.diploid_genotype_template,
    )
    cells = bank.cells
    lengths = tuple(int(template.size(1)) for template in templates)
    required = 20 + sum(lengths)
    if required > CANONICAL_DIGEST_TEMPLATE_PACK_BYTES:
        raise CudaDiploidStateBankError(
            "canonical digest templates exceed fixed CUDA ABI pack: "
            f"{required} > {CANONICAL_DIGEST_TEMPLATE_PACK_BYTES}"
        )
    packed = torch.zeros(
        (cells, CANONICAL_DIGEST_TEMPLATE_PACK_BYTES),
        dtype=torch.uint8,
        device=bank.device,
    )
    for index, length in enumerate(lengths):
        for byte_index in range(4):
            packed[:, 8 + index * 4 + byte_index] = (
                length >> (8 * byte_index)
            ) & 0xFF
    offset = 20
    for template, length in zip(templates, lengths, strict=True):
        packed[:, offset : offset + length].copy_(template)
        offset += length
    return packed.contiguous()


def migrate_homozygous_from_phenotype_v1(
    phenotype_state: Mapping[str, Any],
    *,
    cells: int,
    capacity: int,
    device: torch.device | str,
) -> CudaDiploidStateBankV1:
    """Materialize x -> [x,x] without drawing or changing the phenotype."""

    if type(cells) is not int or cells < 1:
        raise CudaDiploidStateBankError("cells must be a positive exact integer")
    if type(capacity) is not int or capacity < 1:
        raise CudaDiploidStateBankError(
            "capacity must be a positive exact integer"
        )
    target = torch.device(device)
    resource_name = PHENOTYPE_SOURCE_FIELDS["resource_handling_axis"]
    if resource_name in phenotype_state:
        resource_axis = _require_phenotype_tensor(
            phenotype_state,
            name=resource_name,
            cells=cells,
            capacity=capacity,
            device=target,
        ).clone()
    else:
        resource_axis = torch.zeros(
            (cells, capacity), dtype=torch.float32, device=target
        )
    sources = []
    for locus in DIPLOID_SCALAR_LOCI:
        name = PHENOTYPE_SOURCE_FIELDS[locus]
        source = (
            resource_axis
            if locus == "resource_handling_axis"
            else _require_phenotype_tensor(
                phenotype_state,
                name=name,
                cells=cells,
                capacity=capacity,
                device=target,
            )
        )
        sources.append(source)
    phenotype_planes = torch.stack(sources, dim=1)
    storage = torch.cat((phenotype_planes, phenotype_planes), dim=1).contiguous()
    digest = phenotype_state.get("canonical_hereditary_digest")
    if (
        not isinstance(digest, torch.Tensor)
        or digest.dtype != torch.uint8
        or tuple(digest.shape)
        != (cells, capacity, CANONICAL_HEREDITARY_DIGEST_BYTES)
        or digest.device.type != target.type
        or (
            target.index is not None
            and digest.device.index != target.index
        )
        or not digest.is_contiguous()
    ):
        raise CudaDiploidStateBankError(
            "zero-draw diploid migration requires contiguous resident "
            "canonical_hereditary_digest uint8[cells,capacity,32]"
        )
    templates: dict[str, torch.Tensor] = {}
    for name in (
        "canonical_hereditary_template",
        "diploid_payload_template",
        "diploid_genotype_template",
    ):
        value = phenotype_state.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 2
            or value.size(0) != cells
            or value.size(1) < 1
            or value.device.type != target.type
            or (
                target.index is not None
                and value.device.index != target.index
            )
            or not value.is_contiguous()
        ):
            raise CudaDiploidStateBankError(
                f"zero-draw diploid migration requires resident {name}"
            )
        templates[name] = value
    return CudaDiploidStateBankV1(
        storage=storage,
        genome_resource_handling_axis=resource_axis.contiguous(),
        canonical_hereditary_digest=digest.clone(),
        canonical_hereditary_template=templates[
            "canonical_hereditary_template"
        ].clone(),
        diploid_payload_template=templates[
            "diploid_payload_template"
        ].clone(),
        diploid_genotype_template=templates[
            "diploid_genotype_template"
        ].clone(),
        metadata=checkpoint_metadata_v1(cells=cells, capacity=capacity),
    )


def restore_cuda_diploid_state_bank_v1(
    projection: Any,
    *,
    device: torch.device | str,
) -> CudaDiploidStateBankV1:
    if (
        not isinstance(projection, Mapping)
        or projection.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise CudaDiploidStateBankError(
            "diploid checkpoint projection schema mismatch"
        )
    cells, capacity = _validate_metadata(projection.get("metadata"))
    tensors = projection.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != set(
        CHECKPOINT_TENSOR_NAMES
    ):
        raise CudaDiploidStateBankError(
            "diploid checkpoint tensor set mismatch"
        )
    target = torch.device(device)
    planes: list[torch.Tensor] = []
    for name in ALLELE_FIELD_NAMES:
        value = tensors[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.float32
            or tuple(value.shape) != (cells, capacity)
            or not value.is_contiguous()
            or not bool(torch.isfinite(value).all().item())
        ):
            raise CudaDiploidStateBankError(
                f"diploid checkpoint tensor is incompatible: {name}"
            )
        planes.append(value)
    resource = tensors["genome_resource_handling_axis"]
    if (
        not isinstance(resource, torch.Tensor)
        or resource.device.type != "cpu"
        or resource.dtype != torch.float32
        or tuple(resource.shape) != (cells, capacity)
        or not resource.is_contiguous()
        or not bool(torch.isfinite(resource).all().item())
    ):
        raise CudaDiploidStateBankError(
            "resource-handling checkpoint tensor is incompatible"
        )
    digest = tensors["canonical_hereditary_digest"]
    if (
        not isinstance(digest, torch.Tensor)
        or digest.device.type != "cpu"
        or digest.dtype != torch.uint8
        or tuple(digest.shape)
        != (cells, capacity, CANONICAL_HEREDITARY_DIGEST_BYTES)
        or not digest.is_contiguous()
    ):
        raise CudaDiploidStateBankError(
            "canonical hereditary digest checkpoint tensor is incompatible"
        )
    templates: dict[str, torch.Tensor] = {}
    for name in (
        "canonical_hereditary_template",
        "diploid_payload_template",
        "diploid_genotype_template",
    ):
        value = tensors[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.dtype != torch.uint8
            or value.ndim != 2
            or value.size(0) != cells
            or value.size(1) < 1
            or not value.is_contiguous()
        ):
            raise CudaDiploidStateBankError(
                f"diploid checkpoint template is incompatible: {name}"
            )
        templates[name] = value
    storage = torch.stack(planes, dim=1).to(device=target).contiguous()
    return CudaDiploidStateBankV1(
        storage=storage,
        genome_resource_handling_axis=resource.to(device=target).contiguous(),
        canonical_hereditary_digest=digest.to(device=target).contiguous(),
        canonical_hereditary_template=templates[
            "canonical_hereditary_template"
        ].to(device=target).contiguous(),
        diploid_payload_template=templates[
            "diploid_payload_template"
        ].to(device=target).contiguous(),
        diploid_genotype_template=templates[
            "diploid_genotype_template"
        ].to(device=target).contiguous(),
        metadata=dict(projection["metadata"]),
    )


__all__ = [
    "ALLELE_FIELD_NAMES",
    "ALLELE_PLANE_COUNT",
    "CANONICAL_HEREDITARY_DIGEST_BYTES",
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_TENSOR_NAMES",
    "CudaDiploidStateBankError",
    "CudaDiploidStateBankV1",
    "INCREMENTAL_ARRAY_COUNT",
    "PHENOTYPE_SOURCE_FIELDS",
    "STATE_BANK_SCHEMA_VERSION",
    "checkpoint_metadata_v1",
    "migrate_homozygous_from_phenotype_v1",
    "restore_cuda_diploid_state_bank_v1",
]
