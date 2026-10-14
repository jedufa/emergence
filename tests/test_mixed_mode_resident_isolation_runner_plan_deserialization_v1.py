from __future__ import annotations

from copy import deepcopy

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.mixed_mode_resident_isolation_gate import (
    MixedModeIsolationPlanV1,
)
from engine.campaigns.mixed_mode_resident_isolation_plan_composer_v1 import (
    CONFIG_SCHEMA_VERSION,
    MixedModeIsolationComposerConfigV1,
    compose_mixed_mode_resident_isolation_plan_v1,
)
from engine.campaigns.mixed_mode_resident_isolation_runner import (
    MixedModeIsolationRunnerError,
    MixedModeResidentIsolationRunnerPlanV1,
    RegisteredIsolationRunV1,
    RegisteredResidentCellV1,
)


REFILL = "generation_boundary_refill"
LIVE = "ordinary_live_birth"


def _cell(logical_id: str, path_id: str, mode_id: str, seed: int) -> dict:
    return {
        "config": {
            "logical_cell_id": logical_id,
            "equivalence_key": logical_id,
            "reproduction_path": path_id,
            "selection": {
                "selection_mode": "uniform_eligible_parent_v1",
                "eligible_parent_source": "authoritative_eligible_parent_set_v1",
                "draw_distribution": "uniform_v1",
                "deterministic": True,
                "blinded_dimensions": [
                    "genotype",
                    "lineage",
                    "policy",
                    "strategy",
                ],
            },
            "parent_arity": 2,
            "donor_draws_per_successful_birth": 2,
            "authorized_rng_counter_ids": ["selection_transaction"],
        },
        "mode_id": mode_id,
        "seed": seed,
    }


def _runner_payload(tmp_path) -> dict:
    config = MixedModeIsolationComposerConfigV1.from_payload(
        {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "execution_callback_id": "mixed-mode-isolation-callback-v1",
            "execution_authority_sha256": "a" * 64,
            "candidate_artifact_bindings": {
                "exact_full": {
                    "candidate_authority_sha256": "b" * 64,
                    "artifact_sha256": "c" * 64,
                },
                "compact_complete": {
                    "candidate_authority_sha256": "d" * 64,
                    "artifact_sha256": "e" * 64,
                },
            },
            "preregistered_at_utc": "2026-07-24T09:01:00+00:00",
            "runtime_roots": {
                "runs": str(tmp_path / "runs"),
                "owners": str(tmp_path / "owners"),
                "checkpoints": str(tmp_path / "checkpoints"),
            },
            "execution": {
                "generations": 2,
                "ticks_per_generation": 3,
                "checkpoint_split_generation": 1,
            },
            "exact_full_capacities": {
                "A": 24,
                "S": 48,
                "C": 4,
                "K": 300,
                "max_resident_population": 4,
                "current_agent_id_capacity": 4,
                "population_next_id": 4,
                "max_births_per_tick": 1,
                "retained_agent_id_capacity": 4,
            },
            "minimum_live_birth_exposure": {
                "threshold_id": "frozen-live-birth-minimum-v1",
                "metric": "successful_births",
                "minimum": 2,
                "frozen_at_utc": "2026-07-24T09:00:00+00:00",
            },
            "overflow": {
                "target_path_id": LIVE,
                "error_code": "attempt_capacity_exceeded",
                "capacity_kind": "attempt",
                "effective_bound": 0,
                "expected_status_code": 2,
            },
            "live_birth_path_ids": [LIVE],
            "cells": [
                _cell("refill-cell", REFILL, "refill-mode", 1101),
                _cell("live-cell", LIVE, "live-mode", 1102),
            ],
        }
    )
    artifact = compose_mixed_mode_resident_isolation_plan_v1(config)
    return artifact["runner_plan"]


def test_composed_runner_plan_strictly_deserializes_and_round_trips(
    tmp_path,
) -> None:
    payload = _runner_payload(tmp_path)

    restored = MixedModeResidentIsolationRunnerPlanV1.from_payload(payload)

    assert type(restored.gate_plan) is MixedModeIsolationPlanV1
    assert all(type(run) is RegisteredIsolationRunV1 for run in restored.runs)
    assert all(
        type(cell) is RegisteredResidentCellV1
        for run in restored.runs
        for cell in run.cells
    )
    assert restored.payload() == payload
    assert restored.digest_sha256 == canonical_sha256(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update({"unexpected": True}),
            "runner plan field set is invalid",
        ),
        (
            lambda payload: payload["runs"][0].update(
                {"schema_version": "wrong"}
            ),
            "registered run schema_version is invalid",
        ),
        (
            lambda payload: payload["runs"][0]["cells"][0]["config"].update(
                {"parent_arity": 0}
            ),
            "registered cell violates its nested contract",
        ),
    ],
)
def test_deserializer_rejects_unknown_or_invalid_nested_payloads(
    tmp_path,
    mutate,
    message: str,
) -> None:
    payload = deepcopy(_runner_payload(tmp_path))
    mutate(payload)

    with pytest.raises(MixedModeIsolationRunnerError, match=message):
        MixedModeResidentIsolationRunnerPlanV1.from_payload(payload)


def test_deserializer_rejects_stale_gate_digest_before_reconstruction(
    tmp_path,
) -> None:
    payload = deepcopy(_runner_payload(tmp_path))
    payload["gate_plan"]["live_birth_path_ids"] = [REFILL]

    with pytest.raises(
        MixedModeIsolationRunnerError,
        match="gate_plan_sha256 does not bind gate_plan",
    ):
        MixedModeResidentIsolationRunnerPlanV1.from_payload(payload)


def test_deserializer_reapplies_cross_run_registration_invariants(
    tmp_path,
) -> None:
    payload = deepcopy(_runner_payload(tmp_path))
    payload["gate_plan"]["mixed_reference_run_id"] = "missing-reference"
    payload["gate_plan_sha256"] = canonical_sha256(payload["gate_plan"])

    with pytest.raises(
        MixedModeIsolationRunnerError,
        match="registered run 'missing-reference' must exist",
    ):
        MixedModeResidentIsolationRunnerPlanV1.from_payload(payload)
