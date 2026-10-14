from __future__ import annotations

from typing import Any

import pytest
import torch

from engine.campaigns.eed_opportunity_metrics_v1 import (
    EEDOpportunityMeasurementError,
    merge_eed_opportunity_metrics_v1,
    project_eed_opportunity_metrics_v1,
)
from engine.native_runtime.lifecycle_events import FIELD_DTYPES, LifecycleEventType


def _event(
    event_type: LifecycleEventType,
    *,
    generation: int,
    actor: int,
    reason: int = -1,
) -> dict[str, int | float]:
    committed = event_type != LifecycleEventType.BIRTH_FAILED
    target = actor + 100 if event_type in {
        LifecycleEventType.BIRTH_COMMITTED,
        LifecycleEventType.MUTATION,
    } else -1
    return {
        "event_type": int(event_type),
        "generation": generation,
        "tick": generation + 1,
        "actor_agent_id": actor,
        "target_agent_id": target,
        "actor_slot": actor,
        "target_slot": target,
        "parent_agent_id": actor
        if event_type == LifecycleEventType.BIRTH_COMMITTED
        else -1,
        "child_agent_id": target,
        "lineage_founder_id": 1,
        "lineage_depth": 1,
        "subtype_code": int(event_type),
        "reason_code": reason,
        "success": int(committed),
        "value": float(reason if reason >= 0 else 0),
    }


def _projection(rows: list[dict[str, int | float]]) -> dict[str, Any]:
    columns: dict[str, torch.Tensor] = {}
    for name, dtype in FIELD_DTYPES.items():
        if name == "sequence":
            columns[name] = torch.arange(len(rows), dtype=dtype)
        else:
            columns[name] = torch.tensor(
                [row[name] for row in rows],
                dtype=dtype,
            )
    return {
        "metadata": {
            "typed_lifecycle_event_active_count": len(rows),
            "event_tape_active_count": len(rows),
        },
        "tensors": {
            f"typed_lifecycle_event_{name}": tensor
            for name, tensor in columns.items()
        },
    }


def _events() -> list[dict[str, int | float]]:
    return [
        _event(LifecycleEventType.BIRTH_COMMITTED, generation=0, actor=10),
        _event(
            LifecycleEventType.BIRTH_FAILED,
            generation=0,
            actor=11,
            reason=2,
        ),
        _event(
            LifecycleEventType.BIRTH_FAILED,
            generation=0,
            actor=12,
            reason=3,
        ),
        _event(LifecycleEventType.MUTATION, generation=0, actor=110),
        _event(
            LifecycleEventType.DEATH_COMMITTED,
            generation=0,
            actor=20,
            reason=1,
        ),
        _event(LifecycleEventType.BIRTH_COMMITTED, generation=1, actor=10),
        _event(
            LifecycleEventType.BIRTH_FAILED,
            generation=1,
            actor=13,
            reason=1,
        ),
        _event(
            LifecycleEventType.BIRTH_FAILED,
            generation=1,
            actor=14,
            reason=5,
        ),
    ]


def test_projects_frozen_opportunity_contract_and_separate_mutation_count() -> None:
    measurement = project_eed_opportunity_metrics_v1(
        _projection(_events()),
        (
            {"generation": 0, "alive": 5, "successful_births": 1},
            {"generation": 1, "alive": 4, "successful_births": 1},
        ),
    )

    assert measurement.terminal_metrics == {
        "eed.pre_birth_opportunity_count": 5,
        "eed.pre_birth_opportunity_mass": 5.0,
        "eed.birth_attempts": 4,
        "eed.successful_births": 2,
        "eed.unique_opportunity_bearing_parents": 4,
        "eed.committed_mutation_events": 1,
        "eed.hereditary_change_outcomes": 1,
        "eed.population_abundance": 4,
        "eed.opportunity_mass_per_living_agent": 1.25,
        "eed.opportunity_mass_per_opportunity_parent": 1.25,
    }
    assert measurement.generation_metrics[0]["eed"] == {
        "pre_birth_opportunity_count": 3,
        "pre_birth_opportunity_mass": 3.0,
        "birth_attempts": 2,
        "successful_births": 1,
        "unique_opportunity_bearing_parents": 3,
        "committed_mutation_events": 1,
        "hereditary_change_outcomes": 1,
        "population_abundance": 5,
        "opportunity_mass_per_living_agent": 0.6,
        "opportunity_mass_per_opportunity_parent": 1.0,
    }
    assert measurement.audit["event_prefix_active_count"] == 8
    assert (
        measurement.audit["hereditary_change_outcome_definition"]
        == "typed_lifecycle_committed_mutation_event_count_v1"
    )


def test_resume_segment_filters_accumulated_prefix_and_merges_once() -> None:
    events = _events()
    prior = project_eed_opportunity_metrics_v1(
        _projection(events[:5]),
        ({"generation": 0, "alive": 5, "successful_births": 1},),
    )
    resumed = project_eed_opportunity_metrics_v1(
        _projection(events),
        ({"generation": 1, "alive": 4, "successful_births": 1},),
    )
    assert resumed.terminal_metrics["eed.pre_birth_opportunity_count"] == 2
    assert resumed.terminal_metrics["eed.unique_opportunity_bearing_parents"] == 4

    prior_row = {
        "metrics": dict(prior.terminal_metrics),
        "eed_opportunity_measurement": dict(prior.audit),
    }
    projected_row = {
        "metrics": dict(resumed.terminal_metrics),
        "eed_opportunity_measurement": dict(resumed.audit),
    }
    updates, audit = merge_eed_opportunity_metrics_v1(
        prior_row,
        projected_row,
        (*prior.generation_metrics, *resumed.generation_metrics),
    )

    assert updates["eed.pre_birth_opportunity_count"] == 5
    assert updates["eed.birth_attempts"] == 4
    assert updates["eed.successful_births"] == 2
    assert updates["eed.unique_opportunity_bearing_parents"] == 4
    assert updates["eed.committed_mutation_events"] == 1
    assert audit["event_prefix_active_count"] == 8


def test_resume_merge_rejects_event_prefix_substitution() -> None:
    events = _events()
    prior = project_eed_opportunity_metrics_v1(
        _projection(events[:5]),
        ({"generation": 0, "alive": 5, "successful_births": 1},),
    )
    resumed = project_eed_opportunity_metrics_v1(
        _projection(events),
        ({"generation": 1, "alive": 4, "successful_births": 1},),
    )
    projected_audit = dict(resumed.audit)
    projected_audit["prior_generation_event_prefix_digest_sha256"] = "0" * 64

    with pytest.raises(
        EEDOpportunityMeasurementError,
        match="exact event prefix",
    ):
        merge_eed_opportunity_metrics_v1(
            {
                "metrics": dict(prior.terminal_metrics),
                "eed_opportunity_measurement": dict(prior.audit),
            },
            {
                "metrics": dict(resumed.terminal_metrics),
                "eed_opportunity_measurement": projected_audit,
            },
            (*prior.generation_metrics, *resumed.generation_metrics),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("birth_count", "committed births contradict"),
        ("reason", "unknown failed-birth reason"),
        ("sequence", "contiguous zero-based cursor"),
        ("future", "beyond the projected segment"),
    ],
)
def test_measurement_fails_closed_on_contradictory_authority(
    mutation: str,
    message: str,
) -> None:
    events = _events()
    projection = _projection(events)
    generation_rows = [
        {"generation": 0, "alive": 5, "successful_births": 1},
        {"generation": 1, "alive": 4, "successful_births": 1},
    ]
    tensors = projection["tensors"]
    if mutation == "birth_count":
        generation_rows[1]["successful_births"] = 2
    elif mutation == "reason":
        tensors["typed_lifecycle_event_reason_code"][7] = 99
    elif mutation == "sequence":
        tensors["typed_lifecycle_event_sequence"][7] = 12
    elif mutation == "future":
        tensors["typed_lifecycle_event_generation"][7] = 2

    with pytest.raises(EEDOpportunityMeasurementError, match=message):
        project_eed_opportunity_metrics_v1(projection, generation_rows)
