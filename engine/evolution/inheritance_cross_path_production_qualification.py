from __future__ import annotations

"""Typed observations of the real production inheritance boundary.

This module deliberately does not implement inheritance.  It invokes
``crossover_genome`` with the authoritative ``RNGManager`` mutation stream and
returns a small immutable observation suitable for mechanic qualification.
"""

from dataclasses import dataclass
from typing import Any, Iterable

from agents.genome import Genome
from evolution.inheritance_registry_v1 import active_inheritance_profile
from evolution.reproduction import crossover_genome
from simulation.rng_manager import RNGManager


class ProductionInheritanceQualificationError(RuntimeError):
    """Raised when production crossover does not expose its frozen contract."""


@dataclass(frozen=True)
class ProductionInheritanceProfileContract:
    """Declarative observation contract for one production inheritance profile."""

    profile_id: str
    provenance_attribute: str | None = None
    donor_ordinals_key: str | None = None
    donor_digests_key: str | None = None
    linkage_block_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise ValueError("production profile id must be non-empty")
        provenance_fields = (
            self.provenance_attribute,
            self.donor_ordinals_key,
            self.donor_digests_key,
        )
        if self.linkage_block_ids:
            if any(not value for value in provenance_fields):
                raise ValueError(
                    "linked production profile requires provenance field names"
                )
            if (
                any(not block_id for block_id in self.linkage_block_ids)
                or len(set(self.linkage_block_ids)) != len(self.linkage_block_ids)
            ):
                raise ValueError("linkage block ids must be unique and non-empty")
        elif any(value is not None for value in provenance_fields):
            raise ValueError(
                "non-linked production profile cannot declare donor provenance"
            )


def current_production_profile_contracts(
) -> tuple[ProductionInheritanceProfileContract, ...]:
    """Bind the observer only to profiles admitted for new production runs."""

    from evolution.inheritance_registry_v1 import (
        LEGACY_INHERITANCE_PROFILE,
    )

    return (
        ProductionInheritanceProfileContract(
            profile_id=LEGACY_INHERITANCE_PROFILE,
        ),
    )


@dataclass(frozen=True)
class ProductionInheritanceObservation:
    profile_id: str
    global_seed: int
    mutation_py_index_before: int
    mutation_py_index_after: int
    donor_ordinals: tuple[int, ...]
    donor_digests: tuple[tuple[str, str], ...]
    child: Genome

    @property
    def total_mutation_py_draws(self) -> int:
        return self.mutation_py_index_after - self.mutation_py_index_before

    @property
    def linkage_draw_count(self) -> int:
        return len(self.donor_ordinals)


def observe_production_crossover(
    parent_a: Genome,
    parent_b: Genome,
    *,
    global_seed: int,
    sim_cfg: Any,
    profile_contracts: Iterable[ProductionInheritanceProfileContract] | None = None,
) -> ProductionInheritanceObservation:
    """Run one real production crossover and capture its typed RNG/provenance row."""

    if not isinstance(parent_a, Genome) or not isinstance(parent_b, Genome):
        raise ProductionInheritanceQualificationError(
            "production qualification requires two canonical Genome parents"
        )
    contracts = tuple(
        current_production_profile_contracts()
        if profile_contracts is None
        else profile_contracts
    )
    contract_by_profile = {row.profile_id: row for row in contracts}
    if not contracts or len(contract_by_profile) != len(contracts):
        raise ProductionInheritanceQualificationError(
            "production profile contracts must be non-empty and unique"
        )
    profile_id = active_inheritance_profile(sim_cfg)
    profile_contract = contract_by_profile.get(profile_id)
    if profile_contract is None:
        raise ProductionInheritanceQualificationError(
            f"production inheritance profile is not declared: {profile_id!r}"
        )
    manager = RNGManager(global_seed=int(global_seed))
    mutation_rng = manager.py_rng("mutation")
    before = int(manager.indices_snapshot().get("mutation.py", 0))
    child = crossover_genome(
        parent_a,
        parent_b,
        mutation_rng,
        rng_manager=manager,
        sim_cfg=sim_cfg,
    )
    after = int(manager.indices_snapshot().get("mutation.py", 0))
    if not isinstance(child, Genome):
        raise ProductionInheritanceQualificationError(
            "production qualification requires a materialized Genome child"
        )

    donor_ordinals: tuple[int, ...] = ()
    donor_digests: tuple[tuple[str, str], ...] = ()
    if profile_contract.linkage_block_ids:
        context = getattr(child, str(profile_contract.provenance_attribute), None)
        if not isinstance(context, dict):
            raise ProductionInheritanceQualificationError(
                "production child lacks declared donor provenance"
            )
        raw_ordinals = context.get(str(profile_contract.donor_ordinals_key))
        raw_digests = context.get(str(profile_contract.donor_digests_key))
        if (
            not isinstance(raw_ordinals, tuple)
            or len(raw_ordinals) != len(profile_contract.linkage_block_ids)
            or any(value not in (0, 1) for value in raw_ordinals)
            or not isinstance(raw_digests, dict)
            or tuple(raw_digests) != profile_contract.linkage_block_ids
        ):
            raise ProductionInheritanceQualificationError(
                "production donor provenance violates its declared contract"
            )
        donor_ordinals = tuple(int(value) for value in raw_ordinals)
        donor_digests = tuple(
            (block_id, str(raw_digests[block_id]))
            for block_id in profile_contract.linkage_block_ids
        )

    return ProductionInheritanceObservation(
        profile_id=profile_id,
        global_seed=int(global_seed),
        mutation_py_index_before=before,
        mutation_py_index_after=after,
        donor_ordinals=donor_ordinals,
        donor_digests=donor_digests,
        child=child,
    )


__all__ = [
    "ProductionInheritanceObservation",
    "ProductionInheritanceProfileContract",
    "ProductionInheritanceQualificationError",
    "current_production_profile_contracts",
    "observe_production_crossover",
]
