from __future__ import annotations

"""Fixed-envelope GENOME-UPGRADE-V2 genotype-to-phenotype mechanics."""

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Final, Mapping, Sequence


GENOME_UPGRADE_V2_PROFILE: Final[str] = "genome_upgrade_v2_fixed_q10_v1"
GENOME_UPGRADE_V2_SCHEMA_VERSION: Final[str] = "genome_upgrade_v2_bundle_v1"
GENOME_UPGRADE_V2_EXPRESSION_VERSION: Final[str] = (
    "genome_upgrade_v2_polygenic_regulatory_expression_v2"
)
GENOME_UPGRADE_V2_INHERITANCE_VERSION: Final[str] = (
    "genome_upgrade_v2_linked_diploid_tape_v1"
)
GENOME_UPGRADE_V2_RNG_BUDGET_BOUNDARY: Final[str] = (
    "mutation.reproduction.genome_upgrade_v2_inheritance"
)
Q10_SCALE: Final[int] = 1024
RUNTIME_FACTOR_DELTA_DIVISOR: Final[int] = 4
RUNTIME_FACTOR_MIN_Q10: Final[int] = 3 * Q10_SCALE // 4
RUNTIME_FACTOR_MAX_Q10: Final[int] = 5 * Q10_SCALE // 4
RUNTIME_MEMORY_LADDER: Final[tuple[int, ...]] = (0, 8, 16, 32, 64)
CONTEXT_SENESCENT_MATURITY_MULTIPLIER: Final[int] = 4

PRODUCT_CHANNELS: Final[tuple[str, ...]] = (
    "energy_flux",
    "structural_support",
    "contractile_output",
    "repair_activity",
    "barrier_activity",
    "chemosensory_gain",
    "hazard_transduction",
    "reproductive_signal",
    "offspring_provisioning",
    "stress_mediator",
    "plasticity_factor",
    "storage_transport",
)
PHENOTYPE_CHANNELS: Final[tuple[str, ...]] = (
    "body_size",
    "movement_capacity",
    "damage_resistance",
    "metabolic_efficiency",
    "recovery_capacity",
    "sensory_fidelity",
    "reproductive_investment",
    "offspring_starting_energy",
    "threat_response",
    "memory_allocation",
)
CONTEXT_CHANNELS: Final[tuple[str, ...]] = (
    "energy_deficit",
    "injury",
    "crowding",
    "resource_a",
    "resource_b",
    "hazard",
    "juvenile",
    "senescent",
)
ALLELE_PARAMETERS: Final[tuple[str, ...]] = (
    "activity",
    "quantity",
    "stability",
    "sensitivity",
)

PRODUCT_COUNT: Final[int] = len(PRODUCT_CHANNELS)
PHENOTYPE_COUNT: Final[int] = len(PHENOTYPE_CHANNELS)
CONTEXT_COUNT: Final[int] = len(CONTEXT_CHANNELS)
PARAMETER_COUNT: Final[int] = len(ALLELE_PARAMETERS)
LINKAGE_BLOCK_COUNT: Final[int] = PRODUCT_COUNT // 2
HEREDITARY_LOCUS_COUNT: Final[int] = PRODUCT_COUNT * (PARAMETER_COUNT + 1)
MEIOSIS_DRAW_COUNT: Final[int] = 2 * LINKAGE_BLOCK_COUNT
MUTATION_SEMANTIC_DRAW_COUNT: Final[int] = 3 * HEREDITARY_LOCUS_COUNT

# Every phenotype has at least three contributing products and every product
# contributes to multiple phenotypes. Signed entries create explicit trade-offs.
POLYGENIC_WEIGHTS_Q10: Final[tuple[tuple[int, ...], ...]] = (
    (640, 512, 0, 0, 192, 0, 0, 0, 0, 0, 0, 448),
    (-256, 0, 768, 0, -192, 0, 0, 0, 0, 128, 0, 192),
    (-128, 512, 0, 0, 704, 0, 0, 0, 0, -128, 0, 128),
    (704, -128, 0, 0, 0, 0, 0, 0, 0, -128, 0, 448),
    (-192, 0, 0, 768, 256, 0, 0, 0, 0, -128, 128, 0),
    (-128, 0, 0, 0, 0, 640, 512, 0, 0, 0, 128, 0),
    (-256, 0, 0, -128, 0, 0, 0, 704, 256, 0, 0, 192),
    (-192, 128, 0, 0, 0, 0, 0, 256, 768, 0, 0, 0),
    (-128, 0, 256, 0, 0, 0, 576, 0, 0, 640, -128, 0),
    (-192, 0, 0, 0, 0, 128, 0, 0, 0, -128, 768, 256),
)
PRODUCT_CONTEXT_INDEX: Final[tuple[int, ...]] = (
    0, 2, 0, 1, 1, 3, 5, 0, 6, 5, 6, 4
)
PRODUCT_COST_Q10: Final[tuple[int, ...]] = (
    96, 128, 144, 128, 144, 96, 96, 160, 160, 112, 128, 112
)

_PARAMETER_BOUNDS: Final[tuple[tuple[int, int], ...]] = (
    (-Q10_SCALE, Q10_SCALE),
    (0, Q10_SCALE),
    (0, Q10_SCALE),
    (-Q10_SCALE, Q10_SCALE),
)
_MODIFIER_BOUNDS: Final[tuple[int, int]] = (-Q10_SCALE, Q10_SCALE)


class GenomeUpgradeV2Error(ValueError):
    """Stable validation failure for the P1.0 fixed-envelope mechanics."""


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
        raise GenomeUpgradeV2Error("payload is not canonical JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _trunc_div(value: int, divisor: int) -> int:
    if divisor <= 0:
        raise GenomeUpgradeV2Error("divisor must be positive")
    magnitude = abs(int(value)) // divisor
    return -magnitude if value < 0 else magnitude


def _clamp(value: int, lower: int, upper: int) -> int:
    return min(upper, max(lower, int(value)))


def _validated_vector(
    values: Sequence[Any],
    *,
    count: int,
    lower: int,
    upper: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or len(values) != count:
        raise GenomeUpgradeV2Error(f"{name} has the wrong length")
    result: list[int] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, int):
            raise GenomeUpgradeV2Error(f"{name}[{index}] must be an integer")
        if value < lower or value > upper:
            raise GenomeUpgradeV2Error(
                f"{name}[{index}] is outside [{lower}, {upper}]"
            )
        result.append(int(value))
    return tuple(result)


def _validated_alleles(values: Sequence[Any]) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if isinstance(values, (str, bytes)) or len(values) != 2:
        raise GenomeUpgradeV2Error("alleles must contain two homologs")
    homologs: list[tuple[tuple[int, ...], ...]] = []
    for homolog_index, homolog in enumerate(values):
        if isinstance(homolog, (str, bytes)) or len(homolog) != PRODUCT_COUNT:
            raise GenomeUpgradeV2Error(
                f"alleles[{homolog_index}] has the wrong product count"
            )
        products: list[tuple[int, ...]] = []
        for product_index, parameters in enumerate(homolog):
            if isinstance(parameters, (str, bytes)) or len(parameters) != PARAMETER_COUNT:
                raise GenomeUpgradeV2Error(
                    f"alleles[{homolog_index}][{product_index}] "
                    "has the wrong parameter count"
                )
            row: list[int] = []
            for parameter_index, (value, bounds) in enumerate(
                zip(parameters, _PARAMETER_BOUNDS)
            ):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise GenomeUpgradeV2Error(
                        f"alleles[{homolog_index}][{product_index}]"
                        f"[{parameter_index}] must be an integer"
                    )
                lower, upper = bounds
                if value < lower or value > upper:
                    raise GenomeUpgradeV2Error(
                        f"alleles[{homolog_index}][{product_index}]"
                        f"[{parameter_index}] is outside [{lower}, {upper}]"
                    )
                row.append(int(value))
            products.append(tuple(row))
        homologs.append(tuple(products))
    return tuple(homologs)


def contract_payload() -> dict[str, Any]:
    from .genome_upgrade_v2_bd_capabilities_v1 import (
        CAPABILITY_CONTRACT_SHA256,
        capability_contract_payload_v1,
    )
    from .genome_upgrade_v2_regulatory_network_v1 import (
        REGULATORY_CONTRACT_SHA256,
        regulatory_contract_payload_v1,
    )

    return {
        "profile": GENOME_UPGRADE_V2_PROFILE,
        "schema_version": GENOME_UPGRADE_V2_SCHEMA_VERSION,
        "expression_version": GENOME_UPGRADE_V2_EXPRESSION_VERSION,
        "inheritance_version": GENOME_UPGRADE_V2_INHERITANCE_VERSION,
        "numeric_contract": "signed_q10_int32_truncation_toward_zero_v1",
        "product_channels": list(PRODUCT_CHANNELS),
        "phenotype_channels": list(PHENOTYPE_CHANNELS),
        "context_channels": list(CONTEXT_CHANNELS),
        "allele_parameters": list(ALLELE_PARAMETERS),
        "polygenic_weights_q10": [list(row) for row in POLYGENIC_WEIGHTS_Q10],
        "product_context_index": list(PRODUCT_CONTEXT_INDEX),
        "product_cost_q10": list(PRODUCT_COST_Q10),
        "runtime_projection": {
            "factor_delta_divisor": RUNTIME_FACTOR_DELTA_DIVISOR,
            "factor_min_q10": RUNTIME_FACTOR_MIN_Q10,
            "factor_max_q10": RUNTIME_FACTOR_MAX_Q10,
            "damage_resistance_inverse": True,
            "reproductive_investment_inverse": True,
            "memory_dimension_ladder": list(RUNTIME_MEMORY_LADDER),
            "expression_energy_debit": "expression_cost_q10",
        },
        "runtime_context": {
            "normalization": "clamped_truncation_toward_zero_q10_v1",
            "energy_deficit": "(max_energy-energy)/max_energy",
            "injury": "injury/start_health",
            "crowding": "normalized_local_occupancy",
            "resource_a": "normalized_local_primary_resource",
            "resource_b": "normalized_local_secondary_resource",
            "hazard": "normalized_local_hazard",
            "juvenile": "age<maturity_age",
            "senescent": (
                "age>=maturity_age*"
                f"{CONTEXT_SENESCENT_MATURITY_MULTIPLIER}"
            ),
        },
        "runtime_consequences": {
            "snapshot_order": (
                "context_then_expression_then_projection_then_consequence_v1"
            ),
            "movement_cost": (
                "successful_move_cost*body_size_factor/"
                "metabolic_efficiency_factor/"
                "movement_capacity_factor_once"
            ),
            "expression_cost": (
                "expression_energy_debit*body_size_factor/"
                "metabolic_efficiency_factor_once_per_live_tick"
            ),
            "reproduction_threshold": (
                "hereditary_baseline*reproduction_threshold_factor;"
                "eligibility_only"
            ),
            "offspring_energy": (
                "parental_transfer*offspring_energy_factor;"
                "parent_debit_equals_child_credit"
            ),
            "sensory_fidelity": (
                "bounded_environment_observation*"
                "sensory_fidelity_factor"
            ),
            "damage": "same_tick_health_loss*damage_received_factor",
            "recovery": "same_tick_health_gain*recovery_factor",
            "threat_response": (
                "hereditary_aggression_bias+threat_response_q10"
            ),
            "memory": (
                "min(hereditary_memory_dimension,"
                "projected_memory_dimension,admitted_hmem_width)"
            ),
            "default_off": "bit_identical_no_reads_no_writes",
        },
        "linkage_block_by_product": [
            product_index // 2 for product_index in range(PRODUCT_COUNT)
        ],
        "meiosis_draw_count": MEIOSIS_DRAW_COUNT,
        "mutation_semantic_draw_count": MUTATION_SEMANTIC_DRAW_COUNT,
        "hidden_rng_draws": 0,
        "bd_capability_contract_sha256": CAPABILITY_CONTRACT_SHA256,
        "bd_capability_contract": capability_contract_payload_v1(),
        "regulatory_contract_sha256": REGULATORY_CONTRACT_SHA256,
        "regulatory_contract": regulatory_contract_payload_v1(),
    }


CONTRACT_SHA256: Final[str] = _sha256(contract_payload())


@dataclass(frozen=True)
class GenomeUpgradeV2V1:
    """Diploid product and modifier alleles in the frozen Q10 envelope."""

    alleles: tuple[tuple[tuple[int, ...], ...], ...]
    modifiers: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "alleles", _validated_alleles(self.alleles))
        if len(self.modifiers) != 2:
            raise GenomeUpgradeV2Error("modifiers must contain two homologs")
        object.__setattr__(
            self,
            "modifiers",
            tuple(
                _validated_vector(
                    homolog,
                    count=PRODUCT_COUNT,
                    lower=_MODIFIER_BOUNDS[0],
                    upper=_MODIFIER_BOUNDS[1],
                    name=f"modifiers[{homolog_index}]",
                )
                for homolog_index, homolog in enumerate(self.modifiers)
            ),
        )

    @classmethod
    def neutral(cls) -> "GenomeUpgradeV2V1":
        allele = tuple(
            (0, Q10_SCALE, Q10_SCALE, 0) for _ in range(PRODUCT_COUNT)
        )
        modifiers = tuple(0 for _ in range(PRODUCT_COUNT))
        return cls(
            alleles=(allele, allele),
            modifiers=(modifiers, modifiers),
        )

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": GENOME_UPGRADE_V2_SCHEMA_VERSION,
            "contract_sha256": CONTRACT_SHA256,
            "alleles": [
                [list(parameters) for parameters in homolog]
                for homolog in self.alleles
            ],
            "modifiers": [list(homolog) for homolog in self.modifiers],
        }
        payload["genotype_sha256"] = _sha256(payload)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GenomeUpgradeV2V1":
        value = dict(payload)
        if value.get("schema_version") != GENOME_UPGRADE_V2_SCHEMA_VERSION:
            raise GenomeUpgradeV2Error("unsupported genome-upgrade schema")
        if value.get("contract_sha256") != CONTRACT_SHA256:
            raise GenomeUpgradeV2Error("genome-upgrade contract hash mismatch")
        expected = value.pop("genotype_sha256", None)
        if expected != _sha256(value):
            raise GenomeUpgradeV2Error("genome-upgrade genotype hash mismatch")
        return cls(
            alleles=value.get("alleles", ()),
            modifiers=value.get("modifiers", ()),
        )


@dataclass(frozen=True)
class GenomeUpgradeV2ExpressionV1:
    products_q10: tuple[int, ...]
    phenotypes_q10: tuple[int, ...]
    allele_products_q10: tuple[tuple[int, ...], tuple[int, ...]]
    expression_cost_q10: int
    active_product_count: int
    saturated_product_count: int
    regulator_nodes_q10: tuple[int, ...]
    epistatic_delta_q10: tuple[int, ...]
    regulatory_cost_q10: int
    active_regulator_count: int
    active_regulatory_edge_count: int
    developmental_stage: str

    def product_dict(self) -> dict[str, int]:
        return dict(zip(PRODUCT_CHANNELS, self.products_q10))

    def phenotype_dict(self) -> dict[str, int]:
        return dict(zip(PHENOTYPE_CHANNELS, self.phenotypes_q10))


@dataclass(frozen=True)
class GenomeUpgradeV2RuntimeProjectionV1:
    body_size_factor_q10: int
    movement_capacity_factor_q10: int
    damage_received_factor_q10: int
    metabolic_efficiency_factor_q10: int
    recovery_factor_q10: int
    sensory_fidelity_factor_q10: int
    reproduction_threshold_factor_q10: int
    offspring_energy_factor_q10: int
    threat_response_q10: int
    memory_allocation_q10: int
    memory_dimension: int
    expression_energy_debit_q10: int


def _allele_product(parameters: Sequence[int]) -> int:
    activity, quantity, stability, _sensitivity = parameters
    quantity_output = _trunc_div(activity * quantity, Q10_SCALE)
    return _trunc_div(quantity_output * stability, Q10_SCALE)


def express_reference_v1(
    genome: GenomeUpgradeV2V1,
    context_q10: Sequence[int],
) -> GenomeUpgradeV2ExpressionV1:
    context = _validated_vector(
        context_q10,
        count=CONTEXT_COUNT,
        lower=-Q10_SCALE,
        upper=Q10_SCALE,
        name="context_q10",
    )
    allele_products = [
        [_allele_product(genome.alleles[h][p]) for p in range(PRODUCT_COUNT)]
        for h in range(2)
    ]
    products: list[int] = []
    saturated = 0
    for product_index in range(PRODUCT_COUNT):
        dosage = allele_products[0][product_index] + allele_products[1][product_index]
        sensitivity = _trunc_div(
            genome.alleles[0][product_index][3]
            + genome.alleles[1][product_index][3],
            2,
        )
        modifier = _trunc_div(
            genome.modifiers[0][product_index]
            + genome.modifiers[1][product_index],
            2,
        )
        context_effect = _trunc_div(
            sensitivity * context[PRODUCT_CONTEXT_INDEX[product_index]],
            Q10_SCALE,
        )
        unsaturated = dosage + context_effect + _trunc_div(modifier, 4)
        product = _clamp(unsaturated, -Q10_SCALE, Q10_SCALE)
        saturated += int(product != unsaturated)
        products.append(product)

    from .genome_upgrade_v2_regulatory_network_v1 import (
        express_regulatory_reference_v1,
    )

    expressed_modifiers = tuple(
        _trunc_div(
            genome.modifiers[0][product_index]
            + genome.modifiers[1][product_index],
            2,
        )
        for product_index in range(PRODUCT_COUNT)
    )
    regulatory = express_regulatory_reference_v1(
        products, context, expressed_modifiers
    )
    products = list(regulatory.regulated_products_q10)

    phenotypes: list[int] = []
    for weights in POLYGENIC_WEIGHTS_Q10:
        accumulator = 0
        for product, weight in zip(products, weights):
            accumulator += _trunc_div(product * weight, Q10_SCALE)
        phenotypes.append(_clamp(accumulator, -Q10_SCALE, Q10_SCALE))

    expression_cost = regulatory.regulatory_cost_q10 + sum(
        _trunc_div(abs(product) * cost, Q10_SCALE)
        for product, cost in zip(products, PRODUCT_COST_Q10)
    )
    return GenomeUpgradeV2ExpressionV1(
        products_q10=tuple(products),
        phenotypes_q10=tuple(phenotypes),
        allele_products_q10=(
            tuple(allele_products[0]),
            tuple(allele_products[1]),
        ),
        expression_cost_q10=int(expression_cost),
        active_product_count=sum(product != 0 for product in products),
        saturated_product_count=saturated,
        regulator_nodes_q10=regulatory.regulator_nodes_q10,
        epistatic_delta_q10=regulatory.epistatic_delta_q10,
        regulatory_cost_q10=regulatory.regulatory_cost_q10,
        active_regulator_count=regulatory.active_regulator_count,
        active_regulatory_edge_count=regulatory.active_edge_count,
        developmental_stage=regulatory.developmental_stage,
    )


def project_runtime_traits_v1(
    expression: GenomeUpgradeV2ExpressionV1,
) -> GenomeUpgradeV2RuntimeProjectionV1:
    phenotype = expression.phenotype_dict()

    def factor(name: str, *, inverse: bool = False) -> int:
        delta = _trunc_div(
            phenotype[name], RUNTIME_FACTOR_DELTA_DIVISOR
        )
        return _clamp(
            Q10_SCALE - delta if inverse else Q10_SCALE + delta,
            RUNTIME_FACTOR_MIN_Q10,
            RUNTIME_FACTOR_MAX_Q10,
        )

    memory_q10 = phenotype["memory_allocation"]
    ladder_index = _clamp(
        _trunc_div(
            (memory_q10 + Q10_SCALE)
            * (len(RUNTIME_MEMORY_LADDER) - 1),
            2 * Q10_SCALE,
        ),
        0,
        len(RUNTIME_MEMORY_LADDER) - 1,
    )
    return GenomeUpgradeV2RuntimeProjectionV1(
        body_size_factor_q10=factor("body_size"),
        movement_capacity_factor_q10=factor("movement_capacity"),
        damage_received_factor_q10=factor(
            "damage_resistance", inverse=True
        ),
        metabolic_efficiency_factor_q10=factor("metabolic_efficiency"),
        recovery_factor_q10=factor("recovery_capacity"),
        sensory_fidelity_factor_q10=factor("sensory_fidelity"),
        reproduction_threshold_factor_q10=factor(
            "reproductive_investment", inverse=True
        ),
        offspring_energy_factor_q10=factor("offspring_starting_energy"),
        threat_response_q10=phenotype["threat_response"],
        memory_allocation_q10=memory_q10,
        memory_dimension=RUNTIME_MEMORY_LADDER[ladder_index],
        expression_energy_debit_q10=expression.expression_cost_q10,
    )


def _normalized_context_q10(value: float, scale: float) -> int:
    try:
        valid = math.isfinite(value) and math.isfinite(scale) and scale > 0.0
    except TypeError as exc:
        raise GenomeUpgradeV2Error(
            "runtime context normalization is invalid"
        ) from exc
    if not valid:
        raise GenomeUpgradeV2Error("runtime context normalization is invalid")
    return _clamp(int(value * Q10_SCALE / scale), 0, Q10_SCALE)


def build_runtime_context_q10_v1(
    *,
    energy: float,
    max_energy: float,
    injury: float,
    start_health: float,
    crowding: float,
    resource_a: float,
    resource_b: float,
    hazard: float,
    age: int,
    maturity_age: int,
) -> tuple[int, ...]:
    if (
        isinstance(age, bool)
        or not isinstance(age, int)
        or age < 0
        or isinstance(maturity_age, bool)
        or not isinstance(maturity_age, int)
        or maturity_age < 0
    ):
        raise GenomeUpgradeV2Error("runtime developmental context is invalid")
    try:
        energy_deficit = max_energy - energy
    except TypeError as exc:
        raise GenomeUpgradeV2Error(
            "runtime context normalization is invalid"
        ) from exc
    return (
        _normalized_context_q10(energy_deficit, max_energy),
        _normalized_context_q10(injury, start_health),
        _normalized_context_q10(crowding, 1.0),
        _normalized_context_q10(resource_a, 1.0),
        _normalized_context_q10(resource_b, 1.0),
        _normalized_context_q10(hazard, 1.0),
        Q10_SCALE if maturity_age > 0 and age < maturity_age else 0,
        (
            Q10_SCALE
            if (
                maturity_age > 0
                and age
                >= maturity_age * CONTEXT_SENESCENT_MATURITY_MULTIPLIER
            )
            else 0
        ),
    )


def dominance_fixture_v1(
    genome: GenomeUpgradeV2V1,
    context_q10: Sequence[int],
    *,
    product_index: int,
    threshold_q10: int,
) -> dict[str, Any]:
    if product_index < 0 or product_index >= PRODUCT_COUNT:
        raise GenomeUpgradeV2Error("product_index is outside the fixed envelope")
    if threshold_q10 < 0 or threshold_q10 > Q10_SCALE:
        raise GenomeUpgradeV2Error("threshold_q10 is outside [0, 1024]")

    rows: list[int] = []
    for enabled in ((True, False), (False, True), (True, True)):
        alleles = [
            [list(parameters) for parameters in homolog]
            for homolog in genome.alleles
        ]
        for homolog_index, keep in enumerate(enabled):
            if not keep:
                alleles[homolog_index][product_index][0] = 0
        probe = GenomeUpgradeV2V1(
            alleles=tuple(
                tuple(tuple(parameters) for parameters in homolog)
                for homolog in alleles
            ),
            modifiers=genome.modifiers,
        )
        rows.append(
            abs(express_reference_v1(probe, context_q10).products_q10[product_index])
        )
    hom0, hom1, heterozygote = rows
    return {
        "schema_version": "genome_upgrade_v2_dominance_fixture_v1",
        "product_channel": PRODUCT_CHANNELS[product_index],
        "threshold_q10": int(threshold_q10),
        "homolog_0_output_q10": hom0,
        "homolog_1_output_q10": hom1,
        "heterozygote_output_q10": heterozygote,
        "homolog_0_expressed": hom0 >= threshold_q10,
        "homolog_1_expressed": hom1 >= threshold_q10,
        "heterozygote_expressed": heterozygote >= threshold_q10,
        "dominance_is_derived": True,
    }


def diploid_dominance_series_v1(
    genome: GenomeUpgradeV2V1,
    context_q10: Sequence[int],
    *,
    product_index: int,
) -> dict[str, Any]:
    """Compare AA, AB, and BB outputs without a dominance parameter."""

    if product_index < 0 or product_index >= PRODUCT_COUNT:
        raise GenomeUpgradeV2Error(
            "product_index is outside the fixed envelope"
        )

    def homozygote(source_homolog: int) -> GenomeUpgradeV2V1:
        alleles = [
            [list(parameters) for parameters in homolog]
            for homolog in genome.alleles
        ]
        modifiers = [list(homolog) for homolog in genome.modifiers]
        allele = list(genome.alleles[source_homolog][product_index])
        modifier = genome.modifiers[source_homolog][product_index]
        for homolog in range(2):
            alleles[homolog][product_index] = list(allele)
            modifiers[homolog][product_index] = modifier
        return GenomeUpgradeV2V1(
            alleles=tuple(
                tuple(tuple(parameters) for parameters in homolog)
                for homolog in alleles
            ),
            modifiers=tuple(tuple(homolog) for homolog in modifiers),
        )

    outputs = tuple(
        express_reference_v1(candidate, context_q10).products_q10[
            product_index
        ]
        for candidate in (homozygote(0), genome, homozygote(1))
    )
    allele_a, heterozygote, allele_b = outputs
    lower = min(allele_a, allele_b)
    upper = max(allele_a, allele_b)
    expression = express_reference_v1(genome, context_q10)
    return {
        "schema_version": "genome_upgrade_v2_diploid_dominance_series_v1",
        "contract_sha256": CONTRACT_SHA256,
        "product_channel": PRODUCT_CHANNELS[product_index],
        "allele_a_homozygote_q10": allele_a,
        "heterozygote_q10": heterozygote,
        "allele_b_homozygote_q10": allele_b,
        "incomplete_dominance": lower < heterozygote < upper,
        "overdominance": heterozygote > upper,
        "underdominance": heterozygote < lower,
        "codominant_contribution": all(
            expression.allele_products_q10[homolog][product_index] != 0
            for homolog in range(2)
        ),
        "dominance_is_derived": True,
        "free_dominance_scalar": False,
    }


def assemble_child_from_tape_v1(
    parent_a: GenomeUpgradeV2V1,
    parent_b: GenomeUpgradeV2V1,
    *,
    donor_homologs: Sequence[int],
    mutation_homologs: Sequence[int],
    mutation_enabled: Sequence[int],
    mutation_delta_q10: Sequence[int],
) -> tuple[GenomeUpgradeV2V1, dict[str, Any]]:
    donors = _validated_vector(
        donor_homologs,
        count=MEIOSIS_DRAW_COUNT,
        lower=0,
        upper=1,
        name="donor_homologs",
    )
    mutation_h = _validated_vector(
        mutation_homologs,
        count=HEREDITARY_LOCUS_COUNT,
        lower=0,
        upper=1,
        name="mutation_homologs",
    )
    mutation_on = _validated_vector(
        mutation_enabled,
        count=HEREDITARY_LOCUS_COUNT,
        lower=0,
        upper=1,
        name="mutation_enabled",
    )
    deltas = _validated_vector(
        mutation_delta_q10,
        count=HEREDITARY_LOCUS_COUNT,
        lower=-Q10_SCALE,
        upper=Q10_SCALE,
        name="mutation_delta_q10",
    )

    child_alleles: list[list[list[int]]] = [[], []]
    child_modifiers: list[list[int]] = [[], []]
    for child_homolog, parent in enumerate((parent_a, parent_b)):
        donor_offset = child_homolog * LINKAGE_BLOCK_COUNT
        for product_index in range(PRODUCT_COUNT):
            donor = donors[donor_offset + product_index // 2]
            child_alleles[child_homolog].append(
                list(parent.alleles[donor][product_index])
            )
            child_modifiers[child_homolog].append(
                parent.modifiers[donor][product_index]
            )

    changed = 0
    locus_index = 0
    for product_index in range(PRODUCT_COUNT):
        for parameter_index, (lower, upper) in enumerate(_PARAMETER_BOUNDS):
            homolog = mutation_h[locus_index]
            if mutation_on[locus_index]:
                before = child_alleles[homolog][product_index][parameter_index]
                after = _clamp(before + deltas[locus_index], lower, upper)
                child_alleles[homolog][product_index][parameter_index] = after
                changed += int(after != before)
            locus_index += 1
        homolog = mutation_h[locus_index]
        if mutation_on[locus_index]:
            before = child_modifiers[homolog][product_index]
            after = _clamp(
                before + deltas[locus_index],
                _MODIFIER_BOUNDS[0],
                _MODIFIER_BOUNDS[1],
            )
            child_modifiers[homolog][product_index] = after
            changed += int(after != before)
        locus_index += 1

    child = GenomeUpgradeV2V1(
        alleles=tuple(
            tuple(tuple(parameters) for parameters in homolog)
            for homolog in child_alleles
        ),
        modifiers=tuple(tuple(homolog) for homolog in child_modifiers),
    )
    receipt = {
        "schema_version": "genome_upgrade_v2_inheritance_receipt_v1",
        "contract_sha256": CONTRACT_SHA256,
        "inheritance_version": GENOME_UPGRADE_V2_INHERITANCE_VERSION,
        "meiosis_semantic_draws": MEIOSIS_DRAW_COUNT,
        "mutation_semantic_draws": MUTATION_SEMANTIC_DRAW_COUNT,
        "hidden_rng_draws": 0,
        "changed_loci": changed,
        "child_genotype_sha256": child.to_payload()["genotype_sha256"],
    }
    return child, receipt


def assemble_child_from_rng_v1(
    parent_a: GenomeUpgradeV2V1,
    parent_b: GenomeUpgradeV2V1,
    rng: Any,
    *,
    mutation_rate_q10: int,
    mutation_step_q10: int,
) -> tuple[GenomeUpgradeV2V1, dict[str, Any]]:
    if (
        isinstance(mutation_rate_q10, bool)
        or not isinstance(mutation_rate_q10, int)
        or not 0 <= mutation_rate_q10 <= Q10_SCALE
    ):
        raise GenomeUpgradeV2Error("mutation_rate_q10 is outside [0, 1024]")
    if (
        isinstance(mutation_step_q10, bool)
        or not isinstance(mutation_step_q10, int)
        or not 0 <= mutation_step_q10 <= Q10_SCALE
    ):
        raise GenomeUpgradeV2Error("mutation_step_q10 is outside [0, 1024]")
    required = ("randrange", "random", "randint")
    if any(not callable(getattr(rng, name, None)) for name in required):
        raise GenomeUpgradeV2Error("RNG does not provide the fixed tape API")
    donors = [int(rng.randrange(2)) for _ in range(MEIOSIS_DRAW_COUNT)]
    mutation_homologs: list[int] = []
    mutation_enabled: list[int] = []
    mutation_deltas: list[int] = []
    threshold = mutation_rate_q10 / Q10_SCALE
    for _ in range(HEREDITARY_LOCUS_COUNT):
        mutation_homologs.append(int(rng.randrange(2)))
        mutation_enabled.append(int(float(rng.random()) < threshold))
        mutation_deltas.append(
            int(rng.randint(-mutation_step_q10, mutation_step_q10))
        )
    child, receipt = assemble_child_from_tape_v1(
        parent_a,
        parent_b,
        donor_homologs=donors,
        mutation_homologs=mutation_homologs,
        mutation_enabled=mutation_enabled,
        mutation_delta_q10=mutation_deltas,
    )
    receipt.update(
        {
            "rng_budget_boundary": GENOME_UPGRADE_V2_RNG_BUDGET_BOUNDARY,
            "total_semantic_draws": (
                MEIOSIS_DRAW_COUNT + MUTATION_SEMANTIC_DRAW_COUNT
            ),
            "mutation_rate_q10": int(mutation_rate_q10),
            "mutation_step_q10": int(mutation_step_q10),
        }
    )
    return child, receipt


def express_tensor_v1(
    alleles: Any,
    modifiers: Any,
    context_q10: Any,
) -> dict[str, Any]:
    """Run the exact Q10 contract on torch CPU or CUDA tensors."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - project runtime includes torch
        raise GenomeUpgradeV2Error("torch is required for tensor expression") from exc
    if (
        not isinstance(alleles, torch.Tensor)
        or not isinstance(modifiers, torch.Tensor)
        or not isinstance(context_q10, torch.Tensor)
    ):
        raise GenomeUpgradeV2Error("tensor expression requires torch tensors")
    if tuple(alleles.shape[-3:]) != (2, PRODUCT_COUNT, PARAMETER_COUNT):
        raise GenomeUpgradeV2Error("alleles tensor has the wrong trailing shape")
    if tuple(modifiers.shape[-2:]) != (2, PRODUCT_COUNT):
        raise GenomeUpgradeV2Error("modifiers tensor has the wrong trailing shape")
    if context_q10.shape[-1] != CONTEXT_COUNT:
        raise GenomeUpgradeV2Error("context tensor has the wrong trailing shape")
    if alleles.device != modifiers.device or alleles.device != context_q10.device:
        raise GenomeUpgradeV2Error("tensor expression devices differ")
    if (
        alleles.dtype != torch.int64
        or modifiers.dtype != torch.int64
        or context_q10.dtype != torch.int64
    ):
        raise GenomeUpgradeV2Error("tensor expression requires int64 tensors")

    activity = alleles[..., :, :, 0]
    quantity = alleles[..., :, :, 1]
    stability = alleles[..., :, :, 2]
    sensitivity = torch.div(
        alleles[..., 0, :, 3] + alleles[..., 1, :, 3],
        2,
        rounding_mode="trunc",
    )
    allele_products = torch.div(
        activity * quantity, Q10_SCALE, rounding_mode="trunc"
    )
    allele_products = torch.div(
        allele_products * stability, Q10_SCALE, rounding_mode="trunc"
    )
    dosage = allele_products[..., 0, :] + allele_products[..., 1, :]
    modifier = torch.div(
        modifiers[..., 0, :] + modifiers[..., 1, :],
        2,
        rounding_mode="trunc",
    )
    context_indices = torch.tensor(
        PRODUCT_CONTEXT_INDEX,
        device=context_q10.device,
        dtype=torch.int64,
    )
    selected_context = context_q10.index_select(-1, context_indices)
    context_effect = torch.div(
        sensitivity * selected_context, Q10_SCALE, rounding_mode="trunc"
    )
    unsaturated = (
        dosage
        + context_effect
        + torch.div(modifier, 4, rounding_mode="trunc")
    )
    products = torch.clamp(unsaturated, -Q10_SCALE, Q10_SCALE)
    saturated_product_count = (products != unsaturated).sum(dim=-1)
    from .genome_upgrade_v2_regulatory_network_v1 import (
        express_regulatory_tensor_v1,
    )

    regulatory = express_regulatory_tensor_v1(
        products, context_q10, modifier
    )
    products = regulatory["regulated_products_q10"]

    weight_rows = torch.tensor(
        POLYGENIC_WEIGHTS_Q10,
        device=products.device,
        dtype=torch.int64,
    )
    phenotype_rows = []
    for weights in weight_rows:
        terms = torch.div(
            products * weights, Q10_SCALE, rounding_mode="trunc"
        )
        phenotype_rows.append(
            torch.clamp(terms.sum(dim=-1), -Q10_SCALE, Q10_SCALE)
        )
    phenotypes = torch.stack(phenotype_rows, dim=-1)
    costs = torch.tensor(
        PRODUCT_COST_Q10,
        device=products.device,
        dtype=torch.int64,
    )
    expression_cost = regulatory["regulatory_cost_q10"] + torch.div(
        products.abs() * costs,
        Q10_SCALE,
        rounding_mode="trunc",
    ).sum(dim=-1)
    return {
        "products_q10": products,
        "phenotypes_q10": phenotypes,
        "allele_products_q10": allele_products,
        "expression_cost_q10": expression_cost,
        "active_product_count": (products != 0).sum(dim=-1),
        "saturated_product_count": saturated_product_count,
        "regulator_nodes_q10": regulatory["regulator_nodes_q10"],
        "epistatic_delta_q10": regulatory["epistatic_delta_q10"],
        "regulatory_cost_q10": regulatory["regulatory_cost_q10"],
        "active_regulator_count": regulatory["active_regulator_count"],
        "active_regulatory_edge_count": regulatory[
            "active_regulatory_edge_count"
        ],
    }


def attach_genome_upgrade_v2(genome: Any, value: GenomeUpgradeV2V1) -> Any:
    setattr(genome, "_genome_upgrade_v2_v1", value)
    refresh = getattr(genome, "refresh_canonical_payload_cache", None)
    if callable(refresh):
        refresh()
    return genome


def genome_upgrade_v2_from_genome(genome: Any) -> GenomeUpgradeV2V1 | None:
    value = getattr(genome, "_genome_upgrade_v2_v1", None)
    if value is None:
        return None
    if isinstance(value, GenomeUpgradeV2V1):
        return value
    if isinstance(value, Mapping):
        return GenomeUpgradeV2V1.from_payload(value)
    raise GenomeUpgradeV2Error("genome-upgrade attachment is invalid")


__all__ = [
    "ALLELE_PARAMETERS",
    "CONTEXT_CHANNELS",
    "CONTEXT_SENESCENT_MATURITY_MULTIPLIER",
    "CONTRACT_SHA256",
    "GENOME_UPGRADE_V2_EXPRESSION_VERSION",
    "GENOME_UPGRADE_V2_INHERITANCE_VERSION",
    "GENOME_UPGRADE_V2_PROFILE",
    "GENOME_UPGRADE_V2_RNG_BUDGET_BOUNDARY",
    "GENOME_UPGRADE_V2_SCHEMA_VERSION",
    "GenomeUpgradeV2Error",
    "GenomeUpgradeV2ExpressionV1",
    "GenomeUpgradeV2RuntimeProjectionV1",
    "GenomeUpgradeV2V1",
    "HEREDITARY_LOCUS_COUNT",
    "LINKAGE_BLOCK_COUNT",
    "MEIOSIS_DRAW_COUNT",
    "MUTATION_SEMANTIC_DRAW_COUNT",
    "PHENOTYPE_CHANNELS",
    "POLYGENIC_WEIGHTS_Q10",
    "PRODUCT_CHANNELS",
    "Q10_SCALE",
    "RUNTIME_FACTOR_DELTA_DIVISOR",
    "RUNTIME_FACTOR_MAX_Q10",
    "RUNTIME_FACTOR_MIN_Q10",
    "RUNTIME_MEMORY_LADDER",
    "assemble_child_from_tape_v1",
    "assemble_child_from_rng_v1",
    "attach_genome_upgrade_v2",
    "build_runtime_context_q10_v1",
    "contract_payload",
    "dominance_fixture_v1",
    "diploid_dominance_series_v1",
    "express_reference_v1",
    "express_tensor_v1",
    "genome_upgrade_v2_from_genome",
    "project_runtime_traits_v1",
]
