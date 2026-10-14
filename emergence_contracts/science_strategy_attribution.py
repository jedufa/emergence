"""Reusable strategy-attribution contract for science observations.

The native carrier is intentionally unaware of study names or strategy
semantics.  A world may opt into this contract by declaring a strategy
attribution binding in its reducer configuration.  The binding makes the
identity, parent/offspring joins, exact event denominators, and lineage
ownership requirements explicit at admission time; reducers may then report
strategy-specific results without adding a campaign route.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final


SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1: Final[str] = (
    "eiel_science_strategy_attribution_v1"
)

# Keep the tokens stable and route-neutral.  They describe observations, not
# any particular test, founder panel, or scientific conclusion.
SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1: Final[tuple[str, ...]] = (
    "strategy.birth_attribution",
    "strategy.death_attribution",
    "strategy.exact_denominators",
    "strategy.identity",
    "strategy.lineage_attribution",
    "strategy.offspring_attribution",
    "strategy.parent_attribution",
    "strategy.pressure_exposure",
    "strategy.reproduction_funnel",
    "strategy.snapshot_metrics",
)

# These capabilities are a stronger, opt-in extension of the generic strategy
# contract.  The identity is the exact diploid genotype at the explicitly
# declared tracked policy/trait loci.  Keeping the projection explicit avoids
# mistaking policy/ancestry labels for genotype evidence while keeping direct
# attribution bounded when recombination creates many untracked genome states.
SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1: Final[tuple[str, ...]] = (
    "genotype.identity",
    "genotype.reproductive_attribution",
    "genotype.reproduction_funnel",
)
SCIENCE_GENOTYPE_IDENTITY_MODE_V1: Final[str] = (
    "exact_tracked_genotype_state_id_v1"
)
SCIENCE_GENOTYPE_ATTRIBUTION_SCHEMA_V1: Final[str] = (
    "eiel_science_genotype_reproductive_attribution_v1"
)

# Full hereditary attribution is a separate opt-in contract.  It keeps the
# older tracked-locus strategy contract compatible while making a decision
# about population composition explicitly depend on full genome identity,
# parent/child joins, and absolute-tick snapshots.
SCIENCE_HEREDITARY_ATTRIBUTION_SCHEMA_V1: Final[str] = (
    "eiel_science_hereditary_attribution_v1"
)
SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1: Final[tuple[str, ...]] = (
    "hereditary.full_identity",
    "hereditary.parent_child_attribution",
    "hereditary.lineage_attribution",
    "hereditary.reproduction_funnel",
    "hereditary.frequency_trajectory",
)
SCIENCE_HEREDITARY_ATTRIBUTION_IDENTITY_MODE_V1: Final[str] = (
    "full_genome_state_id_v1"
)
SCIENCE_HEREDITARY_ATTRIBUTION_TRACKED_IDENTITY_MODE_V1: Final[str] = (
    SCIENCE_GENOTYPE_IDENTITY_MODE_V1
)
SCIENCE_HEREDITARY_ATTRIBUTION_REQUIRED_FIELDS_V1: Final[tuple[str, ...]] = (
    "event_kinds",
    "identity_mode",
    "tracked_identity_mode",
    "lineage_mode",
    "focal_tracked_genotype_state_ids",
    "denominator_fields",
)

SCIENCE_STRATEGY_ATTRIBUTION_REQUIRED_FIELDS_V1: Final[tuple[str, ...]] = (
    "event_kinds",
    "identity_mode",
    "lineage_mode",
    "strategy_ids",
)

SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1: Final[tuple[str, ...]] = (
    "birth",
    "death",
    "lineage",
    "offspring",
    "parent",
)

SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1: Final[tuple[str, ...]] = (
    "reproduction_eligible_count",
    "mate_candidate_count",
    "birth_attempt_count",
    "successful_parent_count",
    "births_as_parent_count",
)


def strategy_attribution_configuration_v1(
    configuration: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return an opted-in binding, or ``None`` for ordinary reducers.

    This helper deliberately does not infer strategy attribution from a test
    name or a founder panel.  Only an explicit declarative binding activates
    the stronger admission contract.
    """

    candidate = configuration.get("strategy_attribution")
    if candidate is None:
        return None
    if not isinstance(candidate, Mapping):
        raise ValueError("strategy_attribution must be an object")
    return candidate


def validate_strategy_attribution_configuration_v1(
    configuration: Mapping[str, Any],
    *,
    required_observations: list[str] | tuple[str, ...] | set[str],
) -> tuple[str, ...]:
    """Return deterministic admission errors for one explicit binding.

    The function is intentionally pure and independent of the CUDA runtime,
    allowing contract tests and future instruments to reuse it directly.
    """

    binding = strategy_attribution_configuration_v1(configuration)
    if binding is None:
        return ()
    errors: list[str] = []
    if binding.get("schema_version") != SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1:
        errors.append("strategy_attribution.schema_version")
    for field in SCIENCE_STRATEGY_ATTRIBUTION_REQUIRED_FIELDS_V1:
        if field not in binding:
            errors.append(f"strategy_attribution.{field}")
    strategy_ids = binding.get("strategy_ids")
    if (
        not isinstance(strategy_ids, list)
        or not strategy_ids
        or any(not isinstance(value, str) or not value for value in strategy_ids)
        or strategy_ids != sorted(set(strategy_ids))
    ):
        errors.append("strategy_attribution.strategy_ids")
    if binding.get("event_kinds") != list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1):
        errors.append("strategy_attribution.event_kinds")
    identity_mode = binding.get("identity_mode")
    if identity_mode not in {"policy_id_v1", "declared_policy_mapping_v1"}:
        errors.append("strategy_attribution.identity_mode")
    if binding.get("lineage_mode") != "parent_ancestry_v1":
        errors.append("strategy_attribution.lineage_mode")
    mapping = binding.get("policy_id_to_strategy_id", {})
    if not isinstance(mapping, Mapping) or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not value
        or value not in set(strategy_ids or [])
        for key, value in mapping.items()
    ):
        errors.append("strategy_attribution.policy_id_to_strategy_id")
    if identity_mode == "declared_policy_mapping_v1" and not mapping:
        errors.append("strategy_attribution.policy_id_to_strategy_id.empty")
    denominators = binding.get("denominator_fields")
    if denominators != list(SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1):
        errors.append("strategy_attribution.denominator_fields")
    if (
        not isinstance(required_observations, (list, tuple, set, frozenset))
        or any(not isinstance(value, str) or not value for value in required_observations)
    ):
        errors.append("required_observations.type")
        required = set()
    else:
        required = set(required_observations)
    for capability in SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1:
        if capability not in required:
            errors.append(f"required_observations:{capability}")
    direct_genotype = binding.get("direct_genotype_attribution", False)
    if type(direct_genotype) is not bool:
        errors.append("strategy_attribution.direct_genotype_attribution")
    elif direct_genotype:
        if binding.get("genotype_identity_mode") != SCIENCE_GENOTYPE_IDENTITY_MODE_V1:
            errors.append("strategy_attribution.genotype_identity_mode")
        if binding.get("genotype_denominator_fields") != list(
            SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
        ):
            errors.append("strategy_attribution.genotype_denominator_fields")
        for capability in SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1:
            if capability not in required:
                errors.append(f"required_observations:{capability}")
    return tuple(sorted(set(errors)))


def hereditary_attribution_configuration_v1(
    configuration: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    candidate = configuration.get("hereditary_attribution")
    if candidate is None:
        return None
    if not isinstance(candidate, Mapping):
        raise ValueError("hereditary_attribution must be an object")
    return candidate


def validate_hereditary_attribution_configuration_v1(
    configuration: Mapping[str, Any],
    *,
    required_observations: list[str] | tuple[str, ...] | set[str],
) -> tuple[str, ...]:
    """Return deterministic admission errors for full-identity attribution."""

    binding = hereditary_attribution_configuration_v1(configuration)
    if binding is None:
        return ()
    errors: list[str] = []
    if binding.get("schema_version") != SCIENCE_HEREDITARY_ATTRIBUTION_SCHEMA_V1:
        errors.append("hereditary_attribution.schema_version")
    for field in SCIENCE_HEREDITARY_ATTRIBUTION_REQUIRED_FIELDS_V1:
        if field not in binding:
            errors.append(f"hereditary_attribution.{field}")
    if binding.get("event_kinds") != list(SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1):
        errors.append("hereditary_attribution.event_kinds")
    if binding.get("identity_mode") != SCIENCE_HEREDITARY_ATTRIBUTION_IDENTITY_MODE_V1:
        errors.append("hereditary_attribution.identity_mode")
    if binding.get("tracked_identity_mode") != SCIENCE_HEREDITARY_ATTRIBUTION_TRACKED_IDENTITY_MODE_V1:
        errors.append("hereditary_attribution.tracked_identity_mode")
    if binding.get("lineage_mode") != "parent_ancestry_v1":
        errors.append("hereditary_attribution.lineage_mode")
    focal_ids = binding.get("focal_tracked_genotype_state_ids")
    if (
        not isinstance(focal_ids, list)
        or not focal_ids
        or any(
            not isinstance(value, str)
            or len(value) != 71
            or not value.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in value[7:])
            for value in focal_ids
        )
        or focal_ids != sorted(set(focal_ids))
    ):
        errors.append("hereditary_attribution.focal_tracked_genotype_state_ids")
    if binding.get("denominator_fields") != list(
        SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1
    ):
        errors.append("hereditary_attribution.denominator_fields")
    required = set(required_observations) if isinstance(
        required_observations, (list, tuple, set, frozenset)
    ) else set()
    for capability in SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1:
        if capability not in required:
            errors.append(f"required_observations:{capability}")
    return tuple(sorted(set(errors)))


__all__ = (
    "SCIENCE_GENOTYPE_ATTRIBUTION_CAPABILITIES_V1",
    "SCIENCE_GENOTYPE_ATTRIBUTION_SCHEMA_V1",
    "SCIENCE_GENOTYPE_IDENTITY_MODE_V1",
    "SCIENCE_HEREDITARY_ATTRIBUTION_CAPABILITIES_V1",
    "SCIENCE_HEREDITARY_ATTRIBUTION_IDENTITY_MODE_V1",
    "SCIENCE_HEREDITARY_ATTRIBUTION_REQUIRED_FIELDS_V1",
    "SCIENCE_HEREDITARY_ATTRIBUTION_SCHEMA_V1",
    "SCIENCE_HEREDITARY_ATTRIBUTION_TRACKED_IDENTITY_MODE_V1",
    "SCIENCE_STRATEGY_ATTRIBUTION_CAPABILITIES_V1",
    "SCIENCE_STRATEGY_ATTRIBUTION_DENOMINATOR_FIELDS_V1",
    "SCIENCE_STRATEGY_ATTRIBUTION_REQUIRED_FIELDS_V1",
    "SCIENCE_STRATEGY_ATTRIBUTION_EVENT_KINDS_V1",
    "SCIENCE_STRATEGY_ATTRIBUTION_SCHEMA_V1",
    "strategy_attribution_configuration_v1",
    "hereditary_attribution_configuration_v1",
    "validate_hereditary_attribution_configuration_v1",
    "validate_strategy_attribution_configuration_v1",
)
