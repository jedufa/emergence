"""Bounded GV2-B semantic records and GV2-D decomposition telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Sequence

from engine.authoritative_state.schema import content_sha256
from .genome_upgrade_v2 import (
    PHENOTYPE_CHANNELS,
    POLYGENIC_WEIGHTS_Q10,
    PRODUCT_CHANNELS,
    Q10_SCALE,
    RUNTIME_MEMORY_LADDER,
    GenomeUpgradeV2Error,
)


CAPABILITY_SCHEMA_VERSION: Final[str] = (
    "genome_upgrade_v2_bd_capabilities_v1"
)
POLICY_CHANNELS: Final[tuple[str, ...]] = (
    "movement",
    "gather",
    "rest",
    "attack",
)
PERCEPTION_CHANNELS: Final[tuple[str, ...]] = (
    "self_state",
    "hazard",
    "resource_a",
    "resource_b",
)
POLICY_RESIDUAL_CAP_Q10: Final[int] = Q10_SCALE // 4
SENSOR_RADIUS_BOUNDS: Final[tuple[int, int]] = (1, 8)
TOP_CONTRIBUTOR_COUNT: Final[int] = 3
ADMITTED_RUNTIME_SURFACES: Final[tuple[dict[str, Any], ...]] = (
    {
        "surface": "body_size",
        "group": "physical",
        "unit": "signed_q10_factor_delta",
        "runtime_consequence": "maintenance_and_movement_cost_scaling",
    },
    {
        "surface": "movement_capacity",
        "group": "physical",
        "unit": "signed_q10_factor_delta",
        "runtime_consequence": "successful_move_energy_cost_scaling",
    },
    {
        "surface": "damage_resistance",
        "group": "physical",
        "unit": "signed_q10_factor_delta",
        "runtime_consequence": "same_tick_health_loss_scaling",
    },
    {
        "surface": "recovery_capacity",
        "group": "physical",
        "unit": "signed_q10_factor_delta",
        "runtime_consequence": "same_tick_health_gain_scaling",
    },
    {
        "surface": "metabolic_efficiency",
        "group": "metabolic",
        "unit": "signed_q10_factor_delta",
        "runtime_consequence": "movement_and_expression_energy_cost_scaling",
    },
    {
        "surface": "reproductive_investment",
        "group": "life_history",
        "unit": "signed_q10_factor_delta",
        "runtime_consequence": "reproduction_eligibility_threshold_scaling",
    },
    {
        "surface": "offspring_starting_energy",
        "group": "life_history",
        "unit": "signed_q10_factor_delta",
        "runtime_consequence": "conserved_parent_to_child_energy_transfer_scaling",
    },
    {
        "surface": "sensory_fidelity",
        "group": "sensory",
        "unit": "signed_q10_factor_delta",
        "runtime_consequence": "bounded_environment_observation_scaling",
    },
    {
        "surface": "memory_allocation",
        "group": "memory",
        "unit": "signed_q10_ladder_projection",
        "runtime_consequence": "maximum_admitted_recurrent_memory_dimension",
    },
)

_LOCUS_GROUPS: Final[tuple[dict[str, Any], ...]] = (
    {
        "group": "physical",
        "unit": "signed_q10_factor_delta",
        "bounds": (-Q10_SCALE, Q10_SCALE),
        "mutation": "bounded_additive_q10_step_v1",
        "cost": "benefit_scaled_maintenance_or_action_debit_v1",
        "loci": (
            "body_size",
            "movement_speed",
            "carrying_capacity",
            "attack_force",
            "damage_resistance",
            "healing_rate",
            "movement_cost",
            "acceleration_cost",
            "crowding_sensitivity",
        ),
    },
    {
        "group": "metabolic",
        "unit": "signed_q10_factor_delta",
        "bounds": (-Q10_SCALE, Q10_SCALE),
        "mutation": "bounded_additive_q10_step_v1",
        "cost": "efficiency_scaled_basal_or_processing_debit_v1",
        "loci": (
            "basal_metabolic_rate",
            "digestion_efficiency",
            "food_conversion_efficiency",
            "starvation_tolerance",
            "recovery_allocation",
            "repair_efficiency",
            "toxin_tolerance",
            "resource_a_processing_efficiency",
            "resource_b_processing_efficiency",
        ),
    },
    {
        "group": "life_history",
        "unit": "signed_q10_factor_delta",
        "bounds": (-Q10_SCALE, Q10_SCALE),
        "mutation": "bounded_additive_q10_step_v1",
        "cost": "allocation_conserving_energy_or_time_tradeoff_v1",
        "loci": (
            "maturity_age",
            "reproductive_investment",
            "offspring_starting_energy",
            "reproduction_recovery_delay",
            "fertility_decline",
            "senescence_rate",
            "maximum_lifespan",
            "juvenile_growth_allocation",
            "maintenance_reproduction_allocation",
        ),
    },
    {
        "group": "sensory",
        "unit": "signed_q10_factor_delta",
        "bounds": (-Q10_SCALE, Q10_SCALE),
        "mutation": "bounded_additive_q10_step_v1",
        "cost": "acuity_range_or_refresh_scaled_energy_debit_v1",
        "loci": (
            "food_sensing_accuracy",
            "hazard_sensing_accuracy",
            "sensory_range",
            "sensory_noise_resistance",
            "channel_specific_acuity",
            "sensing_energy_cost",
            "refresh_interval",
            "false_positive_tolerance",
        ),
    },
)


def capability_contract_payload_v1() -> dict[str, Any]:
    """Return the canonical, strategy-free B-D capability contract."""

    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "numeric_contract": "signed_q10_integer_truncation_toward_zero_v1",
        "compact_records": {
            "runtime_admission": (
                "candidate_fixtures_only_except_existing_memory_projection"
            ),
            "family_head_policy": {
                "channels": list(POLICY_CHANNELS),
                "homolog_count": 2,
                "bounds_q10": [
                    -POLICY_RESIDUAL_CAP_Q10,
                    POLICY_RESIDUAL_CAP_Q10,
                ],
                "expression": "mean_then_center_and_project_v1",
                "invariants": ["sum_zero", "cap_preserved"],
                "strategy_encoding": "forbidden",
            },
            "perception": {
                "channels": list(PERCEPTION_CHANNELS),
                "sensor_radius_bounds_steps": list(SENSOR_RADIUS_BOUNDS),
                "mask_expression": "either_allele_union_v1",
                "radius_expression": "maximum_expressed_allele_v1",
                "numeric_categorical_averaging": "forbidden",
            },
            "memory": {
                "dimension_ladder": list(RUNTIME_MEMORY_LADDER),
                "dimension_expression": "maximum_admitted_allele_v1",
                "module_expression": "either_allele_union_v1",
                "illegal_intermediate_dimensions": "forbidden",
            },
        },
        "admitted_runtime_surfaces": [
            {
                **row,
                "bounds_q10": [-Q10_SCALE, Q10_SCALE],
                "diploid_storage": (
                    "two_homolog_product_parameter_and_modifier_loci"
                ),
                "expression": "fixed_sparse_product_to_phenotype_projection",
                "mutation": "bounded_parameter_and_modifier_mutation_v1",
                "recombination": (
                    "genome_v2_biparental_gamete_v1_six_fixed_blocks"
                ),
                "neutral_migration": "homozygous_neutral_product_bundle",
                "checkpoint": "cuda_genome_upgrade_v2_state_bank_v1",
                "route_support": "reference_native_cuda_exact",
                "observability": (
                    "compact_product_phenotype_contribution_attribution"
                ),
            }
            for row in ADMITTED_RUNTIME_SURFACES
        ],
        "candidate_locus_groups": [
            {
                **{
                    key: list(value) if key == "bounds" else value
                    for key, value in row.items()
                    if key != "loci"
                },
                "loci": list(row["loci"]),
            }
            for row in _LOCUS_GROUPS
        ],
        "candidate_admission": {
            "checkpoint_schema": "required_before_runtime_admission",
            "route_support": "required_before_runtime_admission",
            "lineage_telemetry": "required_before_runtime_admission",
            "direct_strategy_encoding": "forbidden",
        },
        "polygenic_decomposition": {
            "product_channels": list(PRODUCT_CHANNELS),
            "phenotype_channels": list(PHENOTYPE_CHANNELS),
            "weights_q10": [list(row) for row in POLYGENIC_WEIGHTS_Q10],
            "top_contributor_count": TOP_CONTRIBUTOR_COUNT,
            "redundancy": "active_contributors_beyond_one_v1",
            "compensation": "opposing_signed_contribution_pairs_v1",
        },
    }


CAPABILITY_CONTRACT_SHA256: Final[str] = content_sha256(
    capability_contract_payload_v1()
)


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GenomeUpgradeV2Error(f"{name} must be an integer")
    return int(value)


def _bounded_vector(
    values: Sequence[Any],
    *,
    count: int,
    lower: int,
    upper: int,
    name: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or len(values) != count:
        raise GenomeUpgradeV2Error(f"{name} has the wrong length")
    result = tuple(_integer(value, name=f"{name}[{index}]") for index, value in enumerate(values))
    if any(value < lower or value > upper for value in result):
        raise GenomeUpgradeV2Error(f"{name} is outside [{lower}, {upper}]")
    return result


def _trunc_div(value: int, divisor: int) -> int:
    magnitude = abs(value) // divisor
    return -magnitude if value < 0 else magnitude


def _center_and_project(values: Sequence[int]) -> tuple[int, ...]:
    mean = _trunc_div(sum(values), len(values))
    projected = [
        min(
            POLICY_RESIDUAL_CAP_Q10,
            max(-POLICY_RESIDUAL_CAP_Q10, value - mean),
        )
        for value in values
    ]
    imbalance = sum(projected)
    while imbalance:
        direction = -1 if imbalance > 0 else 1
        for index, value in enumerate(projected):
            candidate = value + direction
            if -POLICY_RESIDUAL_CAP_Q10 <= candidate <= POLICY_RESIDUAL_CAP_Q10:
                projected[index] = candidate
                imbalance += direction
                break
        else:  # pragma: no cover - impossible for a non-empty bounded vector
            raise GenomeUpgradeV2Error("policy projection cannot be balanced")
    return tuple(projected)


@dataclass(frozen=True, slots=True)
class FamilyHeadPolicyRecordV1:
    """Two homologs of bounded action-family residuals."""

    homolog_0_q10: tuple[int, ...]
    homolog_1_q10: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("homolog_0_q10", "homolog_1_q10"):
            object.__setattr__(
                self,
                name,
                _bounded_vector(
                    getattr(self, name),
                    count=len(POLICY_CHANNELS),
                    lower=-POLICY_RESIDUAL_CAP_Q10,
                    upper=POLICY_RESIDUAL_CAP_Q10,
                    name=name,
                ),
            )

    def express_q10(self) -> tuple[int, ...]:
        averaged = tuple(
            _trunc_div(left + right, 2)
            for left, right in zip(self.homolog_0_q10, self.homolog_1_q10)
        )
        return _center_and_project(averaged)

    def payload(self) -> dict[str, Any]:
        value = {
            "schema_version": "genome_v2_family_head_policy_record_v1",
            "contract_sha256": CAPABILITY_CONTRACT_SHA256,
            "homologs_q10": [
                list(self.homolog_0_q10),
                list(self.homolog_1_q10),
            ],
            "expressed_q10": list(self.express_q10()),
        }
        return {**value, "record_sha256": content_sha256(value)}


@dataclass(frozen=True, slots=True)
class PerceptionRecordV1:
    """Discrete perception alleles with explicit non-averaging expression."""

    sensor_radius_steps: tuple[int, int]
    channel_masks: tuple[int, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sensor_radius_steps",
            _bounded_vector(
                self.sensor_radius_steps,
                count=2,
                lower=SENSOR_RADIUS_BOUNDS[0],
                upper=SENSOR_RADIUS_BOUNDS[1],
                name="sensor_radius_steps",
            ),
        )
        mask_limit = (1 << len(PERCEPTION_CHANNELS)) - 1
        object.__setattr__(
            self,
            "channel_masks",
            _bounded_vector(
                self.channel_masks,
                count=2,
                lower=0,
                upper=mask_limit,
                name="channel_masks",
            ),
        )

    def express(self) -> tuple[int, int]:
        return max(self.sensor_radius_steps), self.channel_masks[0] | self.channel_masks[1]


@dataclass(frozen=True, slots=True)
class MemoryRecordV1:
    """Discrete memory alleles constrained to the established ladder."""

    dimensions: tuple[int, int]
    module_masks: tuple[int, int]

    def __post_init__(self) -> None:
        dimensions = _bounded_vector(
            self.dimensions,
            count=2,
            lower=RUNTIME_MEMORY_LADDER[0],
            upper=RUNTIME_MEMORY_LADDER[-1],
            name="dimensions",
        )
        if any(value not in RUNTIME_MEMORY_LADDER for value in dimensions):
            raise GenomeUpgradeV2Error("dimensions contains an illegal ladder value")
        object.__setattr__(self, "dimensions", dimensions)
        object.__setattr__(
            self,
            "module_masks",
            _bounded_vector(
                self.module_masks,
                count=2,
                lower=0,
                upper=0xFF,
                name="module_masks",
            ),
        )

    def express(self) -> tuple[int, int]:
        return max(self.dimensions), self.module_masks[0] | self.module_masks[1]


def polygenic_decomposition_v1(products_q10: Sequence[Any]) -> dict[str, Any]:
    """Decompose the frozen mapping without reading RNG or mutable state."""

    products = _bounded_vector(
        products_q10,
        count=len(PRODUCT_CHANNELS),
        lower=-Q10_SCALE,
        upper=Q10_SCALE,
        name="products_q10",
    )
    phenotypes: list[dict[str, Any]] = []
    for phenotype_index, (phenotype, weights) in enumerate(
        zip(PHENOTYPE_CHANNELS, POLYGENIC_WEIGHTS_Q10)
    ):
        contributions = tuple(
            _trunc_div(product * weight, Q10_SCALE)
            for product, weight in zip(products, weights)
        )
        raw_sum = sum(contributions)
        output = min(Q10_SCALE, max(-Q10_SCALE, raw_sum))
        ranked = sorted(
            range(len(PRODUCT_CHANNELS)),
            key=lambda index: (-abs(contributions[index]), index),
        )[:TOP_CONTRIBUTOR_COUNT]
        positive = [value for value in contributions if value > 0]
        negative = [value for value in contributions if value < 0]
        opposing_pair_count = len(positive) * len(negative)
        compensation_q10 = (
            _trunc_div(
                min(sum(positive), abs(sum(negative))) * Q10_SCALE,
                max(sum(positive), abs(sum(negative))),
            )
            if positive and negative
            else 0
        )
        active_count = sum(value != 0 for value in contributions)
        phenotypes.append(
            {
                "phenotype_index": phenotype_index,
                "phenotype": phenotype,
                "contributions_q10": list(contributions),
                "raw_sum_q10": raw_sum,
                "output_q10": output,
                "top_product_indices": ranked,
                "top_contributions_q10": [
                    contributions[index] for index in ranked
                ],
                "active_contributor_count": active_count,
                "redundant_contributor_count": max(0, active_count - 1),
                "opposing_pair_count": opposing_pair_count,
                "compensation_q10": compensation_q10,
                "saturated": output != raw_sum,
            }
        )
    value = {
        "schema_version": "genome_v2_polygenic_decomposition_v1",
        "contract_sha256": CAPABILITY_CONTRACT_SHA256,
        "products_q10": list(products),
        "phenotypes": phenotypes,
    }
    return {**value, "decomposition_sha256": content_sha256(value)}


__all__ = [
    "ADMITTED_RUNTIME_SURFACES",
    "CAPABILITY_CONTRACT_SHA256",
    "CAPABILITY_SCHEMA_VERSION",
    "FamilyHeadPolicyRecordV1",
    "MemoryRecordV1",
    "PERCEPTION_CHANNELS",
    "POLICY_CHANNELS",
    "PerceptionRecordV1",
    "capability_contract_payload_v1",
    "polygenic_decomposition_v1",
]
