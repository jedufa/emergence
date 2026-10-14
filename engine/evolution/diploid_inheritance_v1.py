from __future__ import annotations

"""Deterministic diploid inheritance for the stable scalar genome surface."""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Final, Mapping

from agents.modular_genome_v1 import canon_float


DIPLOID_INHERITANCE_PROFILE: Final[str] = "diploid_linkage_v1"
DIPLOID_GENOME_SCHEMA_VERSION: Final[str] = "diploid_genome_bundle_v1"
DIPLOID_EXPRESSION_PROFILE: Final[str] = "diploid_additive_expression_v1"
DIPLOID_RECOMBINATION_PROFILE: Final[str] = (
    "fixed_linkage_homolog_donation_v1"
)
DIPLOID_MUTATION_PROFILE: Final[str] = (
    "diploid_allele_mutation_after_assembly_v1"
)
DIPLOID_LINEAGE_SCHEMA_VERSION: Final[str] = "diploid_inheritance_receipt_v1"
DIPLOID_MIGRATION_SCHEMA_VERSION: Final[str] = (
    "diploid_homozygous_legacy_migration_v1"
)
DIPLOID_RNG_BUDGET_BOUNDARY: Final[str] = (
    "mutation.reproduction.diploid_inheritance"
)

DIPLOID_SCALAR_LOCI: Final[tuple[str, ...]] = (
    "exploration_bias",
    "aggression_bias",
    "action_temperature",
    "drive_inertia",
    "drive_noise",
    "drive_logit_gain",
    "pred_lr",
    "drive_sp_hunger",
    "drive_sp_pain",
    "drive_sp_fatigue",
    "drive_sp_fear",
    "exp_prior_strength",
    "exp_prior_noise",
    "drive_birth_noise",
    "reproduction_threshold",
    "mutation_rate",
    "resource_handling_axis",
)

DIPLOID_LINKAGE_BLOCKS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "behavioral_strategy_v1",
        ("exploration_bias", "aggression_bias", "action_temperature"),
    ),
    (
        "regulation_prediction_v1",
        (
            "drive_inertia",
            "drive_noise",
            "drive_logit_gain",
            "pred_lr",
            "drive_sp_hunger",
            "drive_sp_pain",
            "drive_sp_fatigue",
            "drive_sp_fear",
            "exp_prior_strength",
            "exp_prior_noise",
            "drive_birth_noise",
        ),
    ),
    ("life_history_v1", ("reproduction_threshold",)),
    ("mutation_rate_v1", ("mutation_rate",)),
    ("resource_handling_v1", ("resource_handling_axis",)),
)

LOCUS_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "exploration_bias": (-1.0, 1.0),
    "aggression_bias": (-2.0, 2.0),
    "action_temperature": (0.25, 2.5),
    "drive_inertia": (0.0, 0.999),
    "drive_noise": (0.0, 0.1),
    "drive_logit_gain": (0.0, 3.0),
    "pred_lr": (0.0, 1.0),
    "drive_sp_hunger": (0.0, 1.0),
    "drive_sp_pain": (0.0, 1.0),
    "drive_sp_fatigue": (0.0, 1.0),
    "drive_sp_fear": (0.0, 1.0),
    "exp_prior_strength": (0.0, 1.0),
    "exp_prior_noise": (0.0, 0.5),
    "drive_birth_noise": (0.0, 0.2),
    "reproduction_threshold": (2.0, 90.0),
    "mutation_rate": (0.001, 0.2),
    "resource_handling_axis": (-1.0, 1.0),
}

MUTATION_SIGMA: Final[dict[str, float]] = {
    "exploration_bias": 0.10,
    "aggression_bias": 0.15,
    "action_temperature": 0.10,
    "drive_inertia": 0.05,
    "drive_noise": 0.01,
    "drive_logit_gain": 0.20,
    "pred_lr": 0.03,
    "reproduction_threshold": 2.0,
    "mutation_rate": 0.005,
    "resource_handling_axis": 0.10,
}

_LOCUS_INDEX: Final[dict[str, int]] = {
    locus: index for index, locus in enumerate(DIPLOID_SCALAR_LOCI)
}
_LOCUS_BLOCK: Final[dict[str, str]] = {
    locus: block_id
    for block_id, loci in DIPLOID_LINKAGE_BLOCKS
    for locus in loci
}


class DiploidInheritanceError(ValueError):
    """Raised when a diploid genotype or deterministic receipt is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DiploidInheritanceError("diploid payload is not canonical JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def locus_registry_payload() -> dict[str, Any]:
    return {
        "schema_version": "diploid_allin_admitted_locus_registry_v1",
        "loci": [
            {
                "locus_id": locus,
                "ordinal": index,
                "bounds": list(LOCUS_BOUNDS[locus]),
                "mutation_operator": (
                    "bounded_gaussian_existing_semantics_v1"
                    if locus in MUTATION_SIGMA
                    else "explicit_noop_v1"
                ),
            }
            for index, locus in enumerate(DIPLOID_SCALAR_LOCI)
        ],
    }


def linkage_map_payload() -> dict[str, Any]:
    return {
        "schema_version": "diploid_linkage_map_v1",
        "blocks": [
            {
                "block_id": block_id,
                "ordinal": index,
                "member_loci": list(loci),
                "donor_draw_index_by_parent": {
                    "parent_a": index,
                    "parent_b": len(DIPLOID_LINKAGE_BLOCKS) + index,
                },
            }
            for index, (block_id, loci) in enumerate(DIPLOID_LINKAGE_BLOCKS)
        ],
        "internal_crossover": False,
    }


LOCUS_REGISTRY_SHA256: Final[str] = _sha256(locus_registry_payload())
LINKAGE_MAP_SHA256: Final[str] = _sha256(linkage_map_payload())
MEIOSIS_DRAW_COUNT: Final[int] = 2 * len(DIPLOID_LINKAGE_BLOCKS)
MUTATION_SEMANTIC_DRAW_COUNT: Final[int] = 3 * len(DIPLOID_SCALAR_LOCI)


def _finite_allele(value: Any, *, locus: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DiploidInheritanceError(f"{locus} allele must be numeric") from exc
    if not math.isfinite(result):
        raise DiploidInheritanceError(f"{locus} allele must be finite")
    lo, hi = LOCUS_BOUNDS[locus]
    if result < lo or result > hi:
        raise DiploidInheritanceError(
            f"{locus} allele {result!r} outside [{lo}, {hi}]"
        )
    return canon_float(result, field=f"diploid.{locus}")


@dataclass(frozen=True)
class DiploidScalarGenomeV1:
    allele_0: tuple[float, ...]
    allele_1: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.allele_0) != len(DIPLOID_SCALAR_LOCI):
            raise DiploidInheritanceError("allele_0 has the wrong locus count")
        if len(self.allele_1) != len(DIPLOID_SCALAR_LOCI):
            raise DiploidInheritanceError("allele_1 has the wrong locus count")
        object.__setattr__(
            self,
            "allele_0",
            tuple(
                _finite_allele(value, locus=locus)
                for locus, value in zip(DIPLOID_SCALAR_LOCI, self.allele_0)
            ),
        )
        object.__setattr__(
            self,
            "allele_1",
            tuple(
                _finite_allele(value, locus=locus)
                for locus, value in zip(DIPLOID_SCALAR_LOCI, self.allele_1)
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": DIPLOID_GENOME_SCHEMA_VERSION,
            "locus_registry_sha256": LOCUS_REGISTRY_SHA256,
            "linkage_map_sha256": LINKAGE_MAP_SHA256,
            "expression_profile": DIPLOID_EXPRESSION_PROFILE,
            "recombination_profile": DIPLOID_RECOMBINATION_PROFILE,
            "mutation_profile": DIPLOID_MUTATION_PROFILE,
            "allele_0": list(self.allele_0),
            "allele_1": list(self.allele_1),
        }
        payload["genotype_sha256"] = _sha256(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DiploidScalarGenomeV1":
        value = dict(payload)
        if value.get("schema_version") != DIPLOID_GENOME_SCHEMA_VERSION:
            raise DiploidInheritanceError("unsupported diploid genome schema")
        if value.get("locus_registry_sha256") != LOCUS_REGISTRY_SHA256:
            raise DiploidInheritanceError("diploid locus registry hash mismatch")
        if value.get("linkage_map_sha256") != LINKAGE_MAP_SHA256:
            raise DiploidInheritanceError("diploid linkage-map hash mismatch")
        expected_hash = value.pop("genotype_sha256", None)
        if expected_hash != _sha256(value):
            raise DiploidInheritanceError("diploid genotype hash mismatch")
        return cls(
            allele_0=tuple(value.get("allele_0", ())),
            allele_1=tuple(value.get("allele_1", ())),
        )

    def allele(self, homolog: int, locus: str) -> float:
        values = self.allele_0 if int(homolog) == 0 else self.allele_1
        return values[_LOCUS_INDEX[locus]]

    def phenotype(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for index, locus in enumerate(DIPLOID_SCALAR_LOCI):
            lo, hi = LOCUS_BOUNDS[locus]
            value = 0.5 * self.allele_0[index] + 0.5 * self.allele_1[index]
            result[locus] = canon_float(
                min(hi, max(lo, value)),
                field=f"diploid.phenotype.{locus}",
            )
        return result


def homozygous_genome_from_phenotype(genome: Any) -> DiploidScalarGenomeV1:
    values = tuple(
        _finite_allele(getattr(genome, locus), locus=locus)
        for locus in DIPLOID_SCALAR_LOCI
    )
    return DiploidScalarGenomeV1(allele_0=values, allele_1=values)


def diploid_genome_from_genome(genome: Any) -> DiploidScalarGenomeV1:
    payload = getattr(genome, "_diploid_scalar_genome_v1", None)
    if isinstance(payload, DiploidScalarGenomeV1):
        return payload
    if isinstance(payload, Mapping):
        return DiploidScalarGenomeV1.from_payload(payload)
    return homozygous_genome_from_phenotype(genome)


def attach_diploid_genome(genome: Any, value: DiploidScalarGenomeV1) -> Any:
    setattr(genome, "_diploid_scalar_genome_v1", value)
    refresh = getattr(genome, "refresh_canonical_payload_cache", None)
    if callable(refresh):
        refresh()
    return genome


def migrate_legacy_genome(genome: Any) -> tuple[DiploidScalarGenomeV1, dict[str, Any]]:
    diploid = homozygous_genome_from_phenotype(genome)
    phenotype = diploid.phenotype()
    for locus in DIPLOID_SCALAR_LOCI:
        observed = canon_float(getattr(genome, locus), field=locus)
        if phenotype[locus] != observed:
            raise DiploidInheritanceError(
                f"legacy migration changed phenotype at {locus}"
            )
    receipt = {
        "schema_version": DIPLOID_MIGRATION_SCHEMA_VERSION,
        "source": "legacy_haploid_scalar_genome",
        "destination": DIPLOID_GENOME_SCHEMA_VERSION,
        "migration_rng_draws": 0,
        "migration_kind": "homozygous_legacy_migration",
        "genotype_sha256": diploid.to_payload()["genotype_sha256"],
        "phenotype_sha256": _sha256(phenotype),
    }
    return diploid, receipt


def migrate_legacy_population(
    genomes: Any,
) -> tuple[list[DiploidScalarGenomeV1], dict[str, Any]]:
    """Transactionally migrate legacy scalar genomes without consuming RNG."""

    genome_rows = list(genomes)
    prepared = [migrate_legacy_genome(genome) for genome in genome_rows]
    for genome, (diploid, _receipt) in zip(genome_rows, prepared):
        attach_diploid_genome(genome, diploid)
    receipts = [receipt for _diploid, receipt in prepared]
    aggregate = {
        "schema_version": "diploid_population_migration_receipt_v1",
        "migration_kind": "homozygous_legacy_migration",
        "migration_rng_draws": 0,
        "genome_count": len(prepared),
        "genotype_hashes": [
            receipt["genotype_sha256"] for receipt in receipts
        ],
        "receipt_sha256": _sha256(receipts),
    }
    return [diploid for diploid, _receipt in prepared], aggregate


def assemble_diploid_child(
    parent_a: Any,
    parent_b: Any,
    rng: Any,
    *,
    mutation_enabled: bool,
    resource_handling_enabled: bool,
    mutation_sigma_multiplier: float = 1.0,
) -> tuple[DiploidScalarGenomeV1, dict[str, float], dict[str, Any]]:
    parent_genomes = (
        diploid_genome_from_genome(parent_a),
        diploid_genome_from_genome(parent_b),
    )
    gametes: list[list[float]] = [[], []]
    donor_rows: list[dict[str, Any]] = []
    draw_index = 0
    selected_by_parent: list[dict[str, int]] = [{}, {}]
    for parent_index, parent_genome in enumerate(parent_genomes):
        for block_id, loci in DIPLOID_LINKAGE_BLOCKS:
            homolog = 0 if float(rng.random()) < 0.5 else 1
            selected_by_parent[parent_index][block_id] = homolog
            donor_rows.append(
                {
                    "draw_index": draw_index,
                    "parent": "A" if parent_index == 0 else "B",
                    "block_id": block_id,
                    "homolog": homolog,
                }
            )
            draw_index += 1
        gametes[parent_index] = [
            parent_genome.allele(
                selected_by_parent[parent_index][_LOCUS_BLOCK[locus]],
                locus,
            )
            for locus in DIPLOID_SCALAR_LOCI
        ]

    assembled = DiploidScalarGenomeV1(
        allele_0=tuple(gametes[0]),
        allele_1=tuple(gametes[1]),
    )
    allele_rows = [list(assembled.allele_0), list(assembled.allele_1)]
    pre_mutation_rate = assembled.phenotype()["mutation_rate"]
    mutation_rows: list[dict[str, Any]] = []
    mutation_draw_index = 0
    if mutation_enabled:
        for locus_index, locus in enumerate(DIPLOID_SCALAR_LOCI):
            gate_draw = float(rng.random())
            homolog_draw = float(rng.random())
            delta_draw = float(rng.gauss(0.0, 1.0))
            selected_homolog = 0 if homolog_draw < 0.5 else 1
            gate_rate = 0.2 if locus == "mutation_rate" else pre_mutation_rate
            operator_enabled = locus in MUTATION_SIGMA and (
                locus != "resource_handling_axis" or resource_handling_enabled
            )
            pre_value = float(allele_rows[selected_homolog][locus_index])
            semantic_change = bool(operator_enabled and gate_draw < gate_rate)
            delta = (
                delta_draw
                * MUTATION_SIGMA.get(locus, 0.0)
                * float(mutation_sigma_multiplier)
                if semantic_change
                else 0.0
            )
            lo, hi = LOCUS_BOUNDS[locus]
            post_value = canon_float(
                min(hi, max(lo, pre_value + delta)),
                field=f"diploid.mutation.{locus}",
            )
            allele_rows[selected_homolog][locus_index] = post_value
            mutation_rows.append(
                {
                    "locus_id": locus,
                    "linkage_block": _LOCUS_BLOCK[locus],
                    "selected_homolog": selected_homolog,
                    "pre_value": pre_value,
                    "delta": canon_float(delta, field=f"diploid.delta.{locus}"),
                    "post_value": post_value,
                    "semantic_change": bool(post_value != pre_value),
                    "operator_enabled": operator_enabled,
                    "draw_indexes": {
                        "gate": mutation_draw_index,
                        "homolog": mutation_draw_index + 1,
                        "delta": mutation_draw_index + 2,
                    },
                }
            )
            mutation_draw_index += 3

    child = DiploidScalarGenomeV1(
        allele_0=tuple(allele_rows[0]),
        allele_1=tuple(allele_rows[1]),
    )
    phenotype = child.phenotype()
    receipt = {
        "schema_version": DIPLOID_LINEAGE_SCHEMA_VERSION,
        "profile": DIPLOID_INHERITANCE_PROFILE,
        "parent_order": "canonical_hereditary_payload_sha256_v1",
        "meiosis": {
            "draw_count": MEIOSIS_DRAW_COUNT,
            "donors": donor_rows,
        },
        "mutation": {
            "enabled": bool(mutation_enabled),
            "semantic_draw_count": (
                MUTATION_SEMANTIC_DRAW_COUNT if mutation_enabled else 0
            ),
            "events": mutation_rows,
        },
        "genotype_sha256": child.to_payload()["genotype_sha256"],
        "phenotype_sha256": _sha256(phenotype),
    }
    return child, phenotype, receipt


__all__ = [
    "DIPLOID_EXPRESSION_PROFILE",
    "DIPLOID_GENOME_SCHEMA_VERSION",
    "DIPLOID_INHERITANCE_PROFILE",
    "DIPLOID_LINKAGE_BLOCKS",
    "DIPLOID_MUTATION_PROFILE",
    "DIPLOID_RECOMBINATION_PROFILE",
    "DIPLOID_RNG_BUDGET_BOUNDARY",
    "DIPLOID_SCALAR_LOCI",
    "DiploidInheritanceError",
    "DiploidScalarGenomeV1",
    "LINKAGE_MAP_SHA256",
    "LOCUS_REGISTRY_SHA256",
    "MEIOSIS_DRAW_COUNT",
    "MUTATION_SEMANTIC_DRAW_COUNT",
    "assemble_diploid_child",
    "attach_diploid_genome",
    "diploid_genome_from_genome",
    "homozygous_genome_from_phenotype",
    "linkage_map_payload",
    "locus_registry_payload",
    "migrate_legacy_genome",
    "migrate_legacy_population",
]
