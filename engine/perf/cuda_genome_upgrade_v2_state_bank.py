"""Resident CPU/CUDA arrays for the fixed GENOME-UPGRADE-V2 envelope."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import torch

from evolution.genome_upgrade_v2 import (
    CONTEXT_COUNT,
    CONTRACT_SHA256,
    PARAMETER_COUNT,
    PHENOTYPE_COUNT,
    PRODUCT_COUNT,
    Q10_SCALE,
    GenomeUpgradeV2Error,
    GenomeUpgradeV2V1,
    express_tensor_v1,
)
from evolution.genome_upgrade_v2_capability_contract_v1 import (
    GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256,
)
from evolution.genome_upgrade_v2_recomb72_founders_v1 import (
    FOUNDER_PROFILE_IDS as RECOMB72_GV2_FOUNDER_PROFILE_IDS,
)


CUDA_GENOME_UPGRADE_V2_BANK_SCHEMA: str = "cuda_genome_upgrade_v2_state_bank_v1"
GENOME_V2_FOUNDER_PROFILE_SCHEMA: str = "genome_upgrade_v2_founder_profile_v1"
GENOME_V2_FOUNDER_PROFILE_IDS: tuple[str, ...] = (
    "neutral",
    "pleiotropic_tradeoffs",
    "regulatory_activation_inhibition",
    *RECOMB72_GV2_FOUNDER_PROFILE_IDS,
)


def resolve_genome_v2_founder_profile_v1(
    profile_id: str,
) -> dict[str, Any]:
    """Resolve one admitted founder payload from frozen Genome V2 authority."""

    if profile_id not in GENOME_V2_FOUNDER_PROFILE_IDS:
        raise GenomeUpgradeV2Error(
            f"unsupported Genome V2 founder profile: {profile_id!r}"
        )
    founder_population = None
    if profile_id == "neutral":
        genome = GenomeUpgradeV2V1.neutral()
        genotype = tuple(
            value
            for homolog in genome.alleles
            for product in homolog
            for value in product
        ) + tuple(value for homolog in genome.modifiers for value in homolog)
        source = {"kind": "canonical_neutral", "arm": "neutral"}
    elif profile_id in RECOMB72_GV2_FOUNDER_PROFILE_IDS:
        from evolution.genome_upgrade_v2_recomb72_founders_v1 import (
            resolve_recomb72_gv2_founder_profile_v1,
        )

        founder_population = resolve_recomb72_gv2_founder_profile_v1(profile_id)
        genotype = tuple(founder_population["genotype_q10"]["ab"])
        source = {
            "kind": "genome_upgrade_v2_recomb72_founder_population_v1",
            "pattern_id": founder_population["pattern_id"],
            "state_id": founder_population["state_id"],
            "population_payload_sha256": founder_population["payload_sha256"],
        }
    else:
        from evolution.genome_upgrade_v2_causal_gate_v1 import (
            fixture_input_dimensions_v1,
        )

        genotype, _context = fixture_input_dimensions_v1(profile_id, "treatment")
        source = {
            "kind": "genome_upgrade_v2_c1_fixture_v2",
            "family_id": profile_id,
            "arm": "treatment",
        }
    payload = {
        "schema_version": GENOME_V2_FOUNDER_PROFILE_SCHEMA,
        "profile_id": profile_id,
        "source": source,
        "genotype_q10": list(genotype),
    }
    if founder_population is not None:
        payload["founder_population"] = founder_population
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "payload_sha256": digest}


def _require_dimensions(cells: int, capacity: int) -> tuple[int, int]:
    if isinstance(cells, bool) or not isinstance(cells, int) or cells <= 0:
        raise GenomeUpgradeV2Error("cells must be a positive integer")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise GenomeUpgradeV2Error("capacity must be a positive integer")
    return cells, capacity


@dataclass
class CudaGenomeUpgradeV2StateBank:
    """Fixed-shape SoA storage; slots are the final contiguous dimension."""

    alleles_q10: torch.Tensor
    modifiers_q10: torch.Tensor
    products_q10: torch.Tensor
    phenotypes_q10: torch.Tensor
    expression_cost_q10: torch.Tensor
    active_product_count: torch.Tensor
    saturated_product_count: torch.Tensor

    def __post_init__(self) -> None:
        if self.alleles_q10.ndim != 5:
            raise GenomeUpgradeV2Error("alleles bank must have rank 5")
        cells, homologs, products, parameters, capacity = self.alleles_q10.shape
        if (homologs, products, parameters) != (2, PRODUCT_COUNT, PARAMETER_COUNT):
            raise GenomeUpgradeV2Error("alleles bank has the wrong fixed shape")
        expected = {
            "modifiers_q10": (cells, 2, PRODUCT_COUNT, capacity),
            "products_q10": (cells, PRODUCT_COUNT, capacity),
            "phenotypes_q10": (cells, PHENOTYPE_COUNT, capacity),
            "expression_cost_q10": (cells, capacity),
            "active_product_count": (cells, capacity),
            "saturated_product_count": (cells, capacity),
        }
        for name, shape in expected.items():
            tensor = getattr(self, name)
            if tuple(tensor.shape) != shape:
                raise GenomeUpgradeV2Error(f"{name} has the wrong fixed shape")
            if tensor.device != self.alleles_q10.device:
                raise GenomeUpgradeV2Error(f"{name} is on a different device")
        for name, tensor in self.named_tensors().items():
            if tensor.dtype != torch.int64:
                raise GenomeUpgradeV2Error(f"{name} must use exact int64 storage")
            if not tensor.is_contiguous():
                raise GenomeUpgradeV2Error(f"{name} must be contiguous")

    @property
    def cells(self) -> int:
        return int(self.alleles_q10.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.alleles_q10.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.alleles_q10.device

    @classmethod
    def allocate(
        cls,
        *,
        cells: int,
        capacity: int,
        device: torch.device | str,
    ) -> "CudaGenomeUpgradeV2StateBank":
        cells, capacity = _require_dimensions(cells, capacity)
        resolved = torch.device(device)
        alleles = torch.zeros(
            (cells, 2, PRODUCT_COUNT, PARAMETER_COUNT, capacity),
            dtype=torch.int64,
            device=resolved,
        )
        alleles[:, :, :, 1, :].fill_(Q10_SCALE)
        alleles[:, :, :, 2, :].fill_(Q10_SCALE)
        return cls(
            alleles_q10=alleles,
            modifiers_q10=torch.zeros(
                (cells, 2, PRODUCT_COUNT, capacity),
                dtype=torch.int64,
                device=resolved,
            ),
            products_q10=torch.zeros(
                (cells, PRODUCT_COUNT, capacity),
                dtype=torch.int64,
                device=resolved,
            ),
            phenotypes_q10=torch.zeros(
                (cells, PHENOTYPE_COUNT, capacity),
                dtype=torch.int64,
                device=resolved,
            ),
            expression_cost_q10=torch.zeros(
                (cells, capacity), dtype=torch.int64, device=resolved
            ),
            active_product_count=torch.zeros(
                (cells, capacity), dtype=torch.int64, device=resolved
            ),
            saturated_product_count=torch.zeros(
                (cells, capacity), dtype=torch.int64, device=resolved
            ),
        )

    def named_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "genome_v2_alleles_q10": self.alleles_q10,
            "genome_v2_modifiers_q10": self.modifiers_q10,
            "genome_v2_products_q10": self.products_q10,
            "genome_v2_phenotypes_q10": self.phenotypes_q10,
            "genome_v2_expression_cost_q10": self.expression_cost_q10,
            "genome_v2_active_product_count": self.active_product_count,
            "genome_v2_saturated_product_count": self.saturated_product_count,
        }

    def storage_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in self.named_tensors().values()
        )

    def _require_slot(self, cell: int, slot: int) -> tuple[int, int]:
        if not 0 <= int(cell) < self.cells:
            raise GenomeUpgradeV2Error("cell is outside the state bank")
        if not 0 <= int(slot) < self.capacity:
            raise GenomeUpgradeV2Error("slot is outside the state bank")
        return int(cell), int(slot)

    def set_genotype(
        self,
        cell: int,
        slot: int,
        value: GenomeUpgradeV2V1,
    ) -> None:
        cell, slot = self._require_slot(cell, slot)
        alleles = torch.tensor(value.alleles, dtype=torch.int64, device=self.device)
        modifiers = torch.tensor(value.modifiers, dtype=torch.int64, device=self.device)
        self.alleles_q10[cell, :, :, :, slot].copy_(alleles)
        self.modifiers_q10[cell, :, :, slot].copy_(modifiers)

    def initialize_founders_v1(
        self,
        genotype_q10: tuple[int, ...],
        *,
        alive_mask: torch.Tensor,
    ) -> None:
        """Install one frozen genotype into generation-zero live slots."""

        expected = 2 * PRODUCT_COUNT * PARAMETER_COUNT + 2 * PRODUCT_COUNT
        if (
            not isinstance(genotype_q10, tuple)
            or len(genotype_q10) != expected
            or any(type(value) is not int for value in genotype_q10)
        ):
            raise GenomeUpgradeV2Error(
                "founder genotype must be the exact frozen int tuple"
            )
        if (
            not isinstance(alive_mask, torch.Tensor)
            or alive_mask.dtype != torch.bool
            or alive_mask.device != self.device
            or tuple(alive_mask.shape) != (self.cells, self.capacity)
        ):
            raise GenomeUpgradeV2Error(
                "founder alive mask must match the resident bank"
            )
        allele_width = 2 * PRODUCT_COUNT * PARAMETER_COUNT
        alleles = torch.tensor(
            genotype_q10[:allele_width],
            dtype=torch.int64,
            device=self.device,
        ).reshape(2, PRODUCT_COUNT, PARAMETER_COUNT)
        modifiers = torch.tensor(
            genotype_q10[allele_width:],
            dtype=torch.int64,
            device=self.device,
        ).reshape(2, PRODUCT_COUNT)
        self.alleles_q10.permute(0, 4, 1, 2, 3)[alive_mask] = alleles
        self.modifiers_q10.permute(0, 3, 1, 2)[alive_mask] = modifiers

    def initialize_founders_by_cell_v1(
        self,
        genotypes_q10: tuple[tuple[int, ...], ...],
        *,
        alive_mask: torch.Tensor,
    ) -> None:
        """Install one frozen founder genotype per physical cell."""

        if not isinstance(genotypes_q10, tuple) or len(genotypes_q10) != self.cells:
            raise GenomeUpgradeV2Error(
                "founder genotype count must equal resident cells"
            )
        if (
            not isinstance(alive_mask, torch.Tensor)
            or alive_mask.dtype != torch.bool
            or alive_mask.device != self.device
            or tuple(alive_mask.shape) != (self.cells, self.capacity)
        ):
            raise GenomeUpgradeV2Error(
                "founder alive mask must match the resident bank"
            )
        expected = 2 * PRODUCT_COUNT * PARAMETER_COUNT + 2 * PRODUCT_COUNT
        allele_width = 2 * PRODUCT_COUNT * PARAMETER_COUNT
        for cell, genotype in enumerate(genotypes_q10):
            if (
                not isinstance(genotype, tuple)
                or len(genotype) != expected
                or any(type(value) is not int for value in genotype)
            ):
                raise GenomeUpgradeV2Error(
                    f"founder genotype for cell {cell} is invalid"
                )
            alleles = torch.tensor(
                genotype[:allele_width],
                dtype=torch.int64,
                device=self.device,
            ).reshape(2, PRODUCT_COUNT, PARAMETER_COUNT)
            modifiers = torch.tensor(
                genotype[allele_width:],
                dtype=torch.int64,
                device=self.device,
            ).reshape(2, PRODUCT_COUNT)
            self.alleles_q10[cell].permute(3, 0, 1, 2)[alive_mask[cell]] = alleles
            self.modifiers_q10[cell].permute(2, 0, 1)[alive_mask[cell]] = modifiers

    def initialize_founder_populations_by_cell_v1(
        self,
        profile_ids: tuple[str, ...],
        *,
        alive_mask: torch.Tensor,
    ) -> tuple[dict[str, Any], ...]:
        """Install one deterministic heterogeneous founder population per cell."""

        if (
            not isinstance(profile_ids, tuple)
            or len(profile_ids) != self.cells
            or any(type(profile_id) is not str for profile_id in profile_ids)
        ):
            raise GenomeUpgradeV2Error(
                "founder population profile count must equal resident cells"
            )
        if (
            not isinstance(alive_mask, torch.Tensor)
            or alive_mask.dtype != torch.bool
            or alive_mask.device != self.device
            or tuple(alive_mask.shape) != (self.cells, self.capacity)
        ):
            raise GenomeUpgradeV2Error(
                "founder alive mask must match the resident bank"
            )
        return tuple(
            self.initialize_founder_population_cell_v1(
                cell,
                profile_id,
                alive_mask=alive_mask,
            )
            for cell, profile_id in enumerate(profile_ids)
        )

    def initialize_founder_population_cell_v1(
        self,
        cell: int,
        profile_id: str,
        *,
        alive_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Install one deterministic heterogeneous founder population in one cell."""

        from evolution.genome_upgrade_v2_recomb72_founders_v1 import (
            flatten_genome_q10_v1,
            founder_population_genotypes_v1,
            founder_population_geometry_v1,
        )

        if type(profile_id) is not str:
            raise GenomeUpgradeV2Error("founder population profile id must be a string")
        if type(cell) is not int or not 0 <= cell < self.cells:
            raise GenomeUpgradeV2Error("founder population cell is outside the bank")
        if (
            not isinstance(alive_mask, torch.Tensor)
            or alive_mask.dtype != torch.bool
            or alive_mask.device != self.device
            or tuple(alive_mask.shape) != (self.cells, self.capacity)
        ):
            raise GenomeUpgradeV2Error(
                "founder alive mask must match the resident bank"
            )
        active_count = int(alive_mask[cell].sum().item())
        genomes = founder_population_genotypes_v1(
            profile_id,
            active_count,
        )
        flattened = tuple(flatten_genome_q10_v1(value) for value in genomes)
        if flattened:
            allele_width = 2 * PRODUCT_COUNT * PARAMETER_COUNT
            exact = torch.tensor(
                flattened,
                dtype=torch.int64,
                device=self.device,
            )
            self.alleles_q10[cell].permute(3, 0, 1, 2)[alive_mask[cell]] = exact[
                :, :allele_width
            ].reshape(
                active_count,
                2,
                PRODUCT_COUNT,
                PARAMETER_COUNT,
            )
            self.modifiers_q10[cell].permute(2, 0, 1)[alive_mask[cell]] = exact[
                :, allele_width:
            ].reshape(
                active_count,
                2,
                PRODUCT_COUNT,
            )
        return founder_population_geometry_v1(profile_id, active_count)

    def genotype(self, cell: int, slot: int) -> GenomeUpgradeV2V1:
        cell, slot = self._require_slot(cell, slot)
        return GenomeUpgradeV2V1(
            alleles=tuple(
                tuple(
                    tuple(int(value) for value in parameters) for parameters in homolog
                )
                for homolog in self.alleles_q10[cell, :, :, :, slot]
                .detach()
                .cpu()
                .tolist()
            ),
            modifiers=tuple(
                tuple(int(value) for value in homolog)
                for homolog in self.modifiers_q10[cell, :, :, slot]
                .detach()
                .cpu()
                .tolist()
            ),
        )

    def express(self, context_q10: torch.Tensor) -> None:
        if context_q10.dtype != torch.int64:
            raise GenomeUpgradeV2Error("context bank must use int64")
        if context_q10.device != self.device:
            raise GenomeUpgradeV2Error("context bank is on a different device")
        if tuple(context_q10.shape) != (
            self.cells,
            self.capacity,
            CONTEXT_COUNT,
        ):
            raise GenomeUpgradeV2Error("context bank has the wrong fixed shape")
        # Compute layout is [cell, slot, homolog, product, parameter].
        alleles = self.alleles_q10.permute(0, 4, 1, 2, 3)
        modifiers = self.modifiers_q10.permute(0, 3, 1, 2)
        result = express_tensor_v1(alleles, modifiers, context_q10)
        self.products_q10.copy_(result["products_q10"].permute(0, 2, 1))
        self.phenotypes_q10.copy_(result["phenotypes_q10"].permute(0, 2, 1))
        self.expression_cost_q10.copy_(result["expression_cost_q10"])
        self.active_product_count.copy_(result["active_product_count"])
        self.saturated_product_count.copy_(result["saturated_product_count"])

    def checkpoint_cpu_tensors_v1(self) -> dict[str, torch.Tensor]:
        """Freeze one host snapshot for checkpoint serialization and reduction."""

        return {
            name: tensor.detach().to(device="cpu", dtype=torch.int64, copy=True)
            for name, tensor in self.named_tensors().items()
        }

    def checkpoint_projection(
        self,
        *,
        cpu_tensors: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        tensors = (
            self.checkpoint_cpu_tensors_v1()
            if cpu_tensors is None
            else dict(cpu_tensors)
        )
        expected = self.named_tensors()
        if set(tensors) != set(expected):
            raise GenomeUpgradeV2Error(
                "CUDA genome-upgrade checkpoint tensor set differs"
            )
        for name, tensor in tensors.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.dtype != torch.int64
                or tuple(tensor.shape) != tuple(expected[name].shape)
            ):
                raise GenomeUpgradeV2Error(
                    f"CUDA genome-upgrade checkpoint tensor {name} differs"
                )
        return {
            "schema_version": CUDA_GENOME_UPGRADE_V2_BANK_SCHEMA,
            "contract_sha256": CONTRACT_SHA256,
            "capability_contract_sha256": (GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256),
            "cells": self.cells,
            "capacity": self.capacity,
            "dtype": "int64",
            "tensors": {name: tensor.tolist() for name, tensor in tensors.items()},
        }


def restore_cuda_genome_upgrade_v2_state_bank(
    payload: Mapping[str, Any],
    *,
    device: torch.device | str,
    expected_founder_profile: Mapping[str, Any] | None = None,
) -> CudaGenomeUpgradeV2StateBank:
    if payload.get("schema_version") != CUDA_GENOME_UPGRADE_V2_BANK_SCHEMA:
        raise GenomeUpgradeV2Error("unsupported CUDA genome-upgrade checkpoint")
    if payload.get("contract_sha256") != CONTRACT_SHA256:
        raise GenomeUpgradeV2Error("CUDA genome-upgrade contract hash mismatch")
    if (
        payload.get("capability_contract_sha256")
        != GENOME_V2_P10_CAPABILITY_CONTRACT_SHA256
    ):
        raise GenomeUpgradeV2Error(
            "CUDA genome-upgrade capability contract hash mismatch"
        )
    if payload.get("dtype") != "int64":
        raise GenomeUpgradeV2Error("CUDA genome-upgrade checkpoint dtype mismatch")
    if expected_founder_profile is not None:
        expected_identity = {
            "schema_version": GENOME_V2_FOUNDER_PROFILE_SCHEMA,
            "profile_id": expected_founder_profile.get("profile_id"),
            "payload_sha256": expected_founder_profile.get("payload_sha256"),
        }
        if payload.get("founder_profile") != expected_identity:
            raise GenomeUpgradeV2Error(
                "CUDA genome-upgrade founder profile authority mismatch"
            )
    cells, capacity = _require_dimensions(payload.get("cells"), payload.get("capacity"))
    raw_tensors = payload.get("tensors")
    if not isinstance(raw_tensors, Mapping):
        raise GenomeUpgradeV2Error("CUDA genome-upgrade tensors are absent")
    bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=cells, capacity=capacity, device=device
    )
    expected = bank.named_tensors()
    if set(raw_tensors) != set(expected):
        raise GenomeUpgradeV2Error("CUDA genome-upgrade tensor set differs")
    for name, destination in expected.items():
        source = torch.tensor(
            raw_tensors[name], dtype=torch.int64, device=destination.device
        )
        if source.shape != destination.shape:
            raise GenomeUpgradeV2Error(
                f"CUDA genome-upgrade tensor {name} shape differs"
            )
        destination.copy_(source)
    return bank


__all__ = [
    "CUDA_GENOME_UPGRADE_V2_BANK_SCHEMA",
    "CudaGenomeUpgradeV2StateBank",
    "GENOME_V2_FOUNDER_PROFILE_IDS",
    "GENOME_V2_FOUNDER_PROFILE_SCHEMA",
    "resolve_genome_v2_founder_profile_v1",
    "restore_cuda_genome_upgrade_v2_state_bank",
]
