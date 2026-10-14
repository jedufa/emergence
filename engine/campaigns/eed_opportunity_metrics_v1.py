"""Observational EED reproductive-opportunity metrics from lifecycle authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import torch

from engine.native_runtime.lifecycle_events import (
    LifecycleEventType,
    typed_lifecycle_event_digest,
    typed_lifecycle_events_from_projection,
    validated_typed_lifecycle_event_cursor,
)


SCHEMA_VERSION: Final[str] = "eiel_eed_opportunity_measurement_v1"
HEREDITARY_CHANGE_OUTCOME_DEFINITION: Final[str] = (
    "typed_lifecycle_committed_mutation_event_count_v1"
)
PRE_BIRTH_OPPORTUNITY_FAILURE_REASON_CODES: Final[frozenset[int]] = frozenset(
    {2, 3, 5, 6}
)
BIRTH_ATTEMPT_FAILURE_REASON_CODES: Final[frozenset[int]] = frozenset({3, 5, 6})
KNOWN_BIRTH_FAILURE_REASON_CODES: Final[frozenset[int]] = frozenset(range(1, 12))


class EEDOpportunityMeasurementError(ValueError):
    """Raised when lifecycle authority cannot support the frozen EED measure."""


@dataclass(frozen=True, slots=True)
class EEDOpportunityMeasurementV1:
    terminal_metrics: Mapping[str, int | float | None]
    generation_metrics: tuple[dict[str, Any], ...]
    audit: Mapping[str, Any]


def _exact_nonnegative_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise EEDOpportunityMeasurementError(
            f"{name} must be an exact nonnegative integer"
        )
    return value


def _mask_for_codes(values: torch.Tensor, codes: frozenset[int]) -> torch.Tensor:
    mask = torch.zeros_like(values, dtype=torch.bool)
    for code in sorted(codes):
        mask |= values == code
    return mask


def _count(mask: torch.Tensor) -> int:
    return int(mask.sum().item())


def _unique_nonnegative_count(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    name: str,
) -> int:
    selected = values[mask]
    if selected.numel() and bool((selected < 0).any().item()):
        raise EEDOpportunityMeasurementError(
            f"{name} contains an unavailable actor identity"
        )
    return int(torch.unique(selected).numel())


def _per_denominator(numerator: int, denominator: int) -> float | None:
    return float(numerator) / denominator if denominator else None


def _validate_event_semantics(events: Mapping[str, torch.Tensor]) -> None:
    event_type = events["event_type"]
    generation = events["generation"]
    tick = events["tick"]
    sequence = events["sequence"]
    reason = events["reason_code"]
    success = events["success"]
    count = int(event_type.shape[0])

    expected_sequence = torch.arange(count, dtype=torch.int64)
    if not torch.equal(sequence, expected_sequence):
        raise EEDOpportunityMeasurementError(
            "typed lifecycle sequence is not a contiguous zero-based cursor"
        )
    if bool((generation < 0).any().item()) or bool((tick < 0).any().item()):
        raise EEDOpportunityMeasurementError(
            "typed lifecycle generation and tick must be nonnegative"
        )
    if count > 1:
        generation_decreases = generation[1:] < generation[:-1]
        tick_decreases = (generation[1:] == generation[:-1]) & (
            tick[1:] < tick[:-1]
        )
        if bool((generation_decreases | tick_decreases).any().item()):
            raise EEDOpportunityMeasurementError(
                "typed lifecycle rows are not ordered by generation and tick"
            )

    supported_types = {
        int(LifecycleEventType.BIRTH_COMMITTED),
        int(LifecycleEventType.MUTATION),
        int(LifecycleEventType.BIRTH_FAILED),
        int(LifecycleEventType.DEATH_COMMITTED),
    }
    observed_types = {int(value) for value in torch.unique(event_type).tolist()}
    if not observed_types.issubset(supported_types):
        raise EEDOpportunityMeasurementError(
            "typed lifecycle contains an unsupported event type"
        )

    failed = event_type == int(LifecycleEventType.BIRTH_FAILED)
    committed = ~failed
    if bool((success[failed] != 0).any().item()) or bool(
        (success[committed] != 1).any().item()
    ):
        raise EEDOpportunityMeasurementError(
            "typed lifecycle success flags contradict committed/failed event types"
        )
    observed_failure_reasons = {
        int(value) for value in torch.unique(reason[failed]).tolist()
    }
    if not observed_failure_reasons.issubset(KNOWN_BIRTH_FAILURE_REASON_CODES):
        raise EEDOpportunityMeasurementError(
            "typed lifecycle contains an unknown failed-birth reason code"
        )


def project_eed_opportunity_metrics_v1(
    projection: Mapping[str, Any],
    generation_metrics: Sequence[Mapping[str, Any]],
) -> EEDOpportunityMeasurementV1:
    """Project the frozen EED opportunity measure without affecting execution.

    The checkpoint event tape is an accumulated prefix. Only events whose
    absolute generation belongs to ``generation_metrics`` contribute to the
    returned additive segment metrics. The full prefix remains authoritative
    for cumulative unique-parent reporting and resume-continuity auditing.
    """

    rows = [dict(row) for row in generation_metrics]
    if not rows:
        raise EEDOpportunityMeasurementError(
            "EED opportunity measurement requires generation rows"
        )
    generations = [
        _exact_nonnegative_int(row.get("generation"), name="generation")
        for row in rows
    ]
    if any(
        right != left + 1 for left, right in zip(generations, generations[1:])
    ):
        raise EEDOpportunityMeasurementError(
            "EED opportunity generation rows are not contiguous"
        )
    for row in rows:
        _exact_nonnegative_int(row.get("alive"), name="alive")
        _exact_nonnegative_int(
            row.get("successful_births"),
            name="successful_births",
        )
        if "eed" in row:
            raise EEDOpportunityMeasurementError(
                "EED opportunity generation row is already projected"
            )

    events = typed_lifecycle_events_from_projection(projection)
    validated_typed_lifecycle_event_cursor(projection, events)
    _validate_event_semantics(events)

    event_type = events["event_type"]
    event_generation = events["generation"]
    reason = events["reason_code"]
    actor = events["actor_agent_id"]
    segment_start = generations[0]
    segment_end = generations[-1]
    if bool((event_generation > segment_end).any().item()):
        raise EEDOpportunityMeasurementError(
            "typed lifecycle prefix contains events beyond the projected segment"
        )

    birth = event_type == int(LifecycleEventType.BIRTH_COMMITTED)
    failed = event_type == int(LifecycleEventType.BIRTH_FAILED)
    mutation = event_type == int(LifecycleEventType.MUTATION)
    pre_birth_opportunity = birth | (
        failed
        & _mask_for_codes(reason, PRE_BIRTH_OPPORTUNITY_FAILURE_REASON_CODES)
    )
    birth_attempt = birth | (
        failed & _mask_for_codes(reason, BIRTH_ATTEMPT_FAILURE_REASON_CODES)
    )
    full_unique_parents = _unique_nonnegative_count(
        actor,
        pre_birth_opportunity,
        name="pre-birth opportunity rows",
    )

    projected_rows: list[dict[str, Any]] = []
    for generation, row in zip(generations, rows, strict=True):
        in_generation = event_generation == generation
        opportunity_count = _count(pre_birth_opportunity & in_generation)
        attempt_count = _count(birth_attempt & in_generation)
        successful_births = _count(birth & in_generation)
        committed_mutations = _count(mutation & in_generation)
        expected_births = _exact_nonnegative_int(
            row["successful_births"],
            name=f"generation {generation} successful_births",
        )
        if successful_births != expected_births:
            raise EEDOpportunityMeasurementError(
                "typed lifecycle committed births contradict generation metrics: "
                f"generation={generation} events={successful_births} "
                f"metrics={expected_births}"
            )
        generation_unique_parents = _unique_nonnegative_count(
            actor,
            pre_birth_opportunity & in_generation,
            name=f"generation {generation} opportunity rows",
        )
        alive = _exact_nonnegative_int(
            row["alive"],
            name=f"generation {generation} alive",
        )
        row["eed"] = {
            "pre_birth_opportunity_count": opportunity_count,
            "pre_birth_opportunity_mass": float(opportunity_count),
            "birth_attempts": attempt_count,
            "successful_births": successful_births,
            "unique_opportunity_bearing_parents": generation_unique_parents,
            "committed_mutation_events": committed_mutations,
            "hereditary_change_outcomes": committed_mutations,
            "population_abundance": alive,
            "opportunity_mass_per_living_agent": _per_denominator(
                opportunity_count,
                alive,
            ),
            "opportunity_mass_per_opportunity_parent": _per_denominator(
                opportunity_count,
                generation_unique_parents,
            ),
        }
        projected_rows.append(row)

    segment_mask = (event_generation >= segment_start) & (
        event_generation <= segment_end
    )
    segment_opportunities = _count(pre_birth_opportunity & segment_mask)
    segment_attempts = _count(birth_attempt & segment_mask)
    segment_births = _count(birth & segment_mask)
    segment_mutations = _count(mutation & segment_mask)
    terminal_alive = _exact_nonnegative_int(
        rows[-1]["alive"],
        name="terminal alive",
    )

    prior_prefix_mask = event_generation < segment_start
    prior_prefix_count = _count(prior_prefix_mask)
    if prior_prefix_count:
        prefix_events = {
            name: tensor[:prior_prefix_count].contiguous()
            for name, tensor in events.items()
        }
        if not bool(prior_prefix_mask[:prior_prefix_count].all().item()) or bool(
            prior_prefix_mask[prior_prefix_count:].any().item()
        ):
            raise EEDOpportunityMeasurementError(
                "typed lifecycle prior generations are not an exact event prefix"
            )
    else:
        prefix_events = {
            name: tensor[:0].contiguous() for name, tensor in events.items()
        }

    terminal_metrics: dict[str, int | float | None] = {
        "eed.pre_birth_opportunity_count": segment_opportunities,
        "eed.pre_birth_opportunity_mass": float(segment_opportunities),
        "eed.birth_attempts": segment_attempts,
        "eed.successful_births": segment_births,
        "eed.unique_opportunity_bearing_parents": full_unique_parents,
        "eed.committed_mutation_events": segment_mutations,
        "eed.hereditary_change_outcomes": segment_mutations,
        "eed.population_abundance": terminal_alive,
        "eed.opportunity_mass_per_living_agent": _per_denominator(
            segment_opportunities,
            terminal_alive,
        ),
        "eed.opportunity_mass_per_opportunity_parent": _per_denominator(
            segment_opportunities,
            full_unique_parents,
        ),
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "primary_metric": "pre_birth_opportunity_mass",
        "hereditary_change_outcome_definition": (
            HEREDITARY_CHANGE_OUTCOME_DEFINITION
        ),
        "event_prefix_active_count": int(event_type.shape[0]),
        "event_prefix_digest_sha256": typed_lifecycle_event_digest(events),
        "prior_generation_event_prefix_count": prior_prefix_count,
        "prior_generation_event_prefix_digest_sha256": (
            typed_lifecycle_event_digest(prefix_events)
        ),
        "segment_generation_start": segment_start,
        "segment_generation_end": segment_end,
        "segment_event_count": _count(segment_mask),
        "prefix_unique_opportunity_bearing_parents": full_unique_parents,
        "reporting_influence": "forbidden",
    }
    return EEDOpportunityMeasurementV1(
        terminal_metrics=terminal_metrics,
        generation_metrics=tuple(projected_rows),
        audit=audit,
    )


def merge_eed_opportunity_metrics_v1(
    prior_row: Mapping[str, Any],
    projected_row: Mapping[str, Any],
    generation_metrics: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int | float | None], dict[str, Any]]:
    """Merge additive segments after proving exact event-prefix continuity."""

    prior_audit = prior_row.get("eed_opportunity_measurement")
    projected_audit = projected_row.get("eed_opportunity_measurement")
    if not isinstance(prior_audit, Mapping) or not isinstance(
        projected_audit, Mapping
    ):
        raise EEDOpportunityMeasurementError(
            "EED opportunity continuation lacks measurement audit authority"
        )
    if (
        prior_audit.get("schema_version") != SCHEMA_VERSION
        or projected_audit.get("schema_version") != SCHEMA_VERSION
        or projected_audit.get("hereditary_change_outcome_definition")
        != HEREDITARY_CHANGE_OUTCOME_DEFINITION
    ):
        raise EEDOpportunityMeasurementError(
            "EED opportunity continuation changed measurement identity"
        )
    prior_event_count = _exact_nonnegative_int(
        prior_audit.get("event_prefix_active_count"),
        name="prior event prefix active count",
    )
    projected_prior_count = _exact_nonnegative_int(
        projected_audit.get("prior_generation_event_prefix_count"),
        name="projected prior-generation event prefix count",
    )
    if (
        projected_prior_count != prior_event_count
        or projected_audit.get("prior_generation_event_prefix_digest_sha256")
        != prior_audit.get("event_prefix_digest_sha256")
    ):
        raise EEDOpportunityMeasurementError(
            "EED opportunity continuation does not preserve the exact event prefix"
        )

    rows = [dict(row) for row in generation_metrics]
    if not rows:
        raise EEDOpportunityMeasurementError(
            "EED opportunity continuation lacks generation rows"
        )
    generations = [
        _exact_nonnegative_int(row.get("generation"), name="generation")
        for row in rows
    ]
    if any(
        right != left + 1 for left, right in zip(generations, generations[1:])
    ):
        raise EEDOpportunityMeasurementError(
            "EED opportunity continuation generations are not contiguous"
        )

    integer_fields = (
        "pre_birth_opportunity_count",
        "birth_attempts",
        "successful_births",
        "committed_mutation_events",
        "hereditary_change_outcomes",
    )
    totals = {name: 0 for name in integer_fields}
    for row in rows:
        eed = row.get("eed")
        if not isinstance(eed, Mapping):
            raise EEDOpportunityMeasurementError(
                "EED opportunity continuation generation row lacks EED metrics"
            )
        for name in integer_fields:
            totals[name] += _exact_nonnegative_int(
                eed.get(name),
                name=f"generation EED {name}",
            )
        if (
            totals["successful_births"] < 0
            or eed.get("successful_births") != row.get("successful_births")
            or eed.get("committed_mutation_events")
            != eed.get("hereditary_change_outcomes")
        ):
            raise EEDOpportunityMeasurementError(
                "EED opportunity continuation generation metrics contradict "
                "birth or mutation authority"
            )
        abundance = _exact_nonnegative_int(
            eed.get("population_abundance"),
            name="generation EED population abundance",
        )
        if abundance != row.get("alive"):
            raise EEDOpportunityMeasurementError(
                "EED opportunity continuation abundance contradicts population"
            )

    projected_metrics = projected_row.get("metrics")
    if not isinstance(projected_metrics, Mapping):
        raise EEDOpportunityMeasurementError(
            "EED opportunity continuation lacks projected terminal metrics"
        )
    unique_parents = _exact_nonnegative_int(
        projected_metrics.get("eed.unique_opportunity_bearing_parents"),
        name="cumulative unique opportunity-bearing parents",
    )
    if (
        projected_audit.get("prefix_unique_opportunity_bearing_parents")
        != unique_parents
    ):
        raise EEDOpportunityMeasurementError(
            "EED opportunity continuation unique-parent audit contradicts metrics"
        )
    terminal_alive = _exact_nonnegative_int(
        rows[-1].get("alive"),
        name="terminal alive",
    )
    opportunity_count = totals["pre_birth_opportunity_count"]
    updates: dict[str, int | float | None] = {
        "eed.pre_birth_opportunity_count": opportunity_count,
        "eed.pre_birth_opportunity_mass": float(opportunity_count),
        "eed.birth_attempts": totals["birth_attempts"],
        "eed.successful_births": totals["successful_births"],
        "eed.unique_opportunity_bearing_parents": unique_parents,
        "eed.committed_mutation_events": totals["committed_mutation_events"],
        "eed.hereditary_change_outcomes": totals[
            "hereditary_change_outcomes"
        ],
        "eed.population_abundance": terminal_alive,
        "eed.opportunity_mass_per_living_agent": _per_denominator(
            opportunity_count,
            terminal_alive,
        ),
        "eed.opportunity_mass_per_opportunity_parent": _per_denominator(
            opportunity_count,
            unique_parents,
        ),
    }
    return updates, dict(projected_audit)
