"""Contract tests for the isolation runner.

The deterministic callback below is a fake used only to test orchestration,
hash binding, fail-closed parsing, and artifact durability.  Its green result
is not scientific evidence and does not qualify any execution route.
"""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from engine.campaigns.mixed_mode_resident_isolation_gate import (
    RUN_KIND_ALONE,
    RUN_KIND_CHECKPOINT_BASELINE,
    RUN_KIND_CHECKPOINT_FRESH_OWNER,
    RUN_KIND_DUPLICATE,
    RUN_KIND_MIXED,
    RUN_KIND_OVERFLOW,
    RUN_KIND_PERMUTED,
    CheckpointCaseV1,
    FrozenMinimumInheritanceExposureV1,
    InheritanceAttemptV1,
    MixedModeIsolationPlanV1,
    OverflowCaseV1,
    ParentSelectionContractV1,
    ReproductionPathReceiptV1,
    ResidentCellCaptureV1,
    ResidentReproductionCellConfigV1,
)
from engine.campaigns.mixed_mode_resident_isolation_runner import (
    EXECUTION_OBSERVATION_SCHEMA_VERSION,
    MixedModeIsolationRunnerError,
    MixedModeResidentIsolationRunnerPlanV1,
    RegisteredIsolationRunV1,
    RegisteredResidentCellV1,
    bind_resident_isolation_observation_v1,
    execute_mixed_mode_resident_isolation_gate_v1,
)


REFILL = "generation_boundary_refill_v1"
LIVE = "ordinary_live_birth_v1"
AUTHORITY = "a" * 64
CHECKPOINT = "b" * 64
LIFECYCLE = "c" * 64
OVERFLOW_CODE = "event_capacity_overflow_v1"


def _selection() -> ParentSelectionContractV1:
    return ParentSelectionContractV1(
        selection_mode="uniform_eligible_parent_v1",
        eligible_parent_source="authoritative_eligible_parent_set_v1",
        draw_distribution="uniform_v1",
        deterministic=True,
        blinded_dimensions=("genotype", "lineage", "policy", "strategy"),
    )


def _config(
    logical_id: str,
    path_id: str,
    *,
    equivalence_key: str | None = None,
) -> ResidentReproductionCellConfigV1:
    return ResidentReproductionCellConfigV1(
        logical_cell_id=logical_id,
        equivalence_key=equivalence_key or logical_id,
        reproduction_path=path_id,
        selection=_selection(),
        parent_arity=2,
        donor_draws_per_successful_birth=7,
        authorized_rng_counter_ids=("parent_selection",),
    )


def _registered_cell(
    logical_id: str,
    path_id: str,
    position: int,
    neighbors: tuple[str, ...],
    *,
    occurrence_id: str | None = None,
    equivalence_key: str | None = None,
) -> RegisteredResidentCellV1:
    return RegisteredResidentCellV1(
        occurrence_id=occurrence_id or logical_id,
        config=_config(
            logical_id,
            path_id,
            equivalence_key=equivalence_key,
        ),
        batch_position=position,
        expected_neighbor_path_ids=neighbors,
    )


def _registered_run(
    run_id: str,
    kind: str,
    cells: tuple[RegisteredResidentCellV1, ...],
    *,
    fresh_owner: bool = False,
    **execution_payload,
) -> RegisteredIsolationRunV1:
    return RegisteredIsolationRunV1(
        run_id=run_id,
        run_kind=kind,
        fresh_owner=fresh_owner,
        cells=cells,
        execution_payload={
            "provider_request_schema": "deterministic_fake_contract_test_v1",
            **execution_payload,
        },
    )


def _plan() -> MixedModeResidentIsolationRunnerPlanV1:
    gate = MixedModeIsolationPlanV1(
        path_ids=(REFILL, LIVE),
        mixed_reference_run_id="mixed",
        permutation_run_ids=("permuted",),
        duplicate_run_ids=("duplicates",),
        duplicate_required_path_ids=(REFILL, LIVE),
        overflow_cases=(
            OverflowCaseV1(
                reference_run_id="overflow-reference",
                fault_run_id="overflow",
                target_occurrence_id="live",
                overflow_error_code=OVERFLOW_CODE,
            ),
        ),
        checkpoint_cases=(
            CheckpointCaseV1(
                uninterrupted_run_id="checkpoint-baseline",
                fresh_owner_run_id="checkpoint-resumed",
            ),
        ),
        minimum_live_birth_exposure=FrozenMinimumInheritanceExposureV1(
            threshold_id="test-only-frozen-threshold-v1",
            metric="successful_births",
            minimum=2,
            frozen_at_utc="2026-07-23T20:00:00+00:00",
        ),
        live_birth_path_ids=(LIVE,),
    )
    refill = _registered_cell("refill", REFILL, 0, (LIVE,))
    live = _registered_cell("live", LIVE, 1, (REFILL,))
    return MixedModeResidentIsolationRunnerPlanV1(
        gate_plan=gate,
        execution_callback_id="deterministic_fake_contract_test_callback_v1",
        execution_authority_sha256=AUTHORITY,
        preregistered_at_utc="2026-07-23T21:00:00+00:00",
        runs=(
            _registered_run("mixed", RUN_KIND_MIXED, (refill, live)),
            _registered_run(
                "alone-refill",
                RUN_KIND_ALONE,
                (_registered_cell("refill", REFILL, 0, ()),),
            ),
            _registered_run(
                "alone-live",
                RUN_KIND_ALONE,
                (_registered_cell("live", LIVE, 0, ()),),
            ),
            _registered_run(
                "permuted",
                RUN_KIND_PERMUTED,
                (
                    _registered_cell("live", LIVE, 0, (REFILL,)),
                    _registered_cell("refill", REFILL, 1, (LIVE,)),
                ),
            ),
            _registered_run(
                "duplicates",
                RUN_KIND_DUPLICATE,
                (
                    _registered_cell(
                        "dup-r-a",
                        REFILL,
                        0,
                        (LIVE,),
                        equivalence_key="same-r",
                    ),
                    _registered_cell(
                        "dup-l-a",
                        LIVE,
                        1,
                        (REFILL, REFILL),
                        equivalence_key="same-l",
                    ),
                    _registered_cell(
                        "dup-r-b",
                        REFILL,
                        2,
                        (LIVE, LIVE),
                        equivalence_key="same-r",
                    ),
                    _registered_cell(
                        "dup-l-b",
                        LIVE,
                        3,
                        (REFILL,),
                        equivalence_key="same-l",
                    ),
                ),
            ),
            _registered_run(
                "overflow-reference",
                RUN_KIND_MIXED,
                (refill, live),
            ),
            _registered_run(
                "overflow",
                RUN_KIND_OVERFLOW,
                (refill, live),
                overflow_target="live",
            ),
            _registered_run(
                "checkpoint-baseline",
                RUN_KIND_CHECKPOINT_BASELINE,
                (refill, live),
            ),
            _registered_run(
                "checkpoint-resumed",
                RUN_KIND_CHECKPOINT_FRESH_OWNER,
                (refill, live),
                fresh_owner=True,
                restore_owner="new_process",
            ),
        ),
    )


def _attempt() -> InheritanceAttemptV1:
    return InheritanceAttemptV1(
        primary_eligible_parent_ids=("p0", "p1", "p2"),
        mate_eligible_parent_ids=("p1", "p2"),
        selected_parent_ids=("p0", "p1"),
        success=True,
        contributing_parent_ids=("p0", "p1"),
    )


def _path_receipts(path_id: str) -> tuple[ReproductionPathReceiptV1, ...]:
    receipts = []
    for candidate in (REFILL, LIVE):
        enabled = candidate == path_id
        attempts = (_attempt(), _attempt()) if enabled else ()
        receipts.append(
            ReproductionPathReceiptV1(
                path_id=candidate,
                enabled=enabled,
                rng_counters=(("parent_selection", 4),) if enabled else (),
                donor_draw_ordinals=tuple(range(14)) if enabled else (),
                counter_writes=2 if enabled else 0,
                sidecar_writes=2 if enabled else 0,
                event_row_writes=2 if enabled else 0,
                rng_state_writes=4 if enabled else 0,
                attempts=attempts,
            )
        )
    return tuple(receipts)


def _state_digest(cell: RegisteredResidentCellV1) -> str:
    key = cell.config.equivalence_key
    digit = {
        "refill": "1",
        "live": "2",
        "same-r": "3",
        "same-l": "4",
    }[key]
    return digit * 64


def _capture_cell(
    cell: RegisteredResidentCellV1,
    *,
    overflow: bool,
) -> ResidentCellCaptureV1:
    state = _state_digest(cell)
    is_target = overflow and cell.occurrence_id == "live"
    return ResidentCellCaptureV1(
        occurrence_id=cell.occurrence_id,
        config=cell.config,
        batch_position=cell.batch_position,
        neighbor_path_ids=cell.expected_neighbor_path_ids,
        final_state_sha256=state,
        pre_fault_state_sha256=state,
        checkpoint_sidecar_sha256=CHECKPOINT,
        lifecycle_state_sha256=LIFECYCLE,
        final_authority_sha256=state,
        pre_fault_authority_sha256=state if is_target else None,
        ok=not is_target,
        fail_closed=is_target,
        error_code=OVERFLOW_CODE if is_target else None,
        path_receipts=_path_receipts(cell.config.reproduction_path),
    )


class _DeterministicFakeContractCallback:
    """Fake contract source; never a scientific or route-qualification owner."""

    def __init__(self, plan_path):
        self.plan_path = plan_path
        self.calls: list[str] = []

    def __call__(self, request):
        # The harness must have durably preregistered the plan before execution.
        persisted = json.loads(self.plan_path.read_text(encoding="utf-8"))
        assert persisted["runner_plan_sha256"] == request.runner_plan_sha256
        self.calls.append(request.run.run_id)
        cells = [
            _capture_cell(
                cell,
                overflow=request.run.run_kind == RUN_KIND_OVERFLOW,
            ).payload()
            for cell in request.run.cells
        ]
        return bind_resident_isolation_observation_v1(
            {
                "schema_version": EXECUTION_OBSERVATION_SCHEMA_VERSION,
                "runner_plan_sha256": request.runner_plan_sha256,
                "gate_plan_sha256": request.gate_plan_sha256,
                "execution_callback_id": request.execution_callback_id,
                "execution_authority_sha256": request.execution_authority_sha256,
                "registered_run_sha256": request.run.digest_sha256,
                "run_id": request.run.run_id,
                "run_kind": request.run.run_kind,
                "captured_at_utc": "2026-07-23T22:00:00+00:00",
                "fresh_owner": request.run.fresh_owner,
                "cells": cells,
                "execution_evidence": {
                    "schema_version": "deterministic_fake_contract_evidence_v1",
                    "scientific_evidence": False,
                    "route_qualification": False,
                },
            }
        )


def test_runner_preregisters_executes_all_cases_and_persists_green_result(
    tmp_path,
) -> None:
    plan = _plan()
    root = tmp_path / "gate"
    callback = _DeterministicFakeContractCallback(root / "plan.json")

    receipt = execute_mixed_mode_resident_isolation_gate_v1(
        plan, callback, root
    )

    assert receipt.gate_result.ok is True
    assert len(receipt.gate_result.checks) == 9
    assert callback.calls == [run.run_id for run in plan.runs]
    assert len(receipt.capture_artifact_sha256s) == len(plan.runs)
    persisted_plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    persisted_result = json.loads(
        (root / "result.json").read_text(encoding="utf-8")
    )
    assert persisted_plan["gate_plan"]["minimum_live_birth_exposure"] == {
        "schema_version": "eiel_frozen_minimum_inheritance_exposure_v1",
        "threshold_id": "test-only-frozen-threshold-v1",
        "metric": "successful_births",
        "minimum": 2,
        "frozen_at_utc": "2026-07-23T20:00:00+00:00",
    }
    assert persisted_result["runner_plan_sha256"] == plan.digest_sha256
    assert persisted_result["gate_result"]["ok"] is True
    assert len(list((root / "captures").glob("*.json"))) == len(plan.runs)


@pytest.mark.parametrize("field", ["cells", "execution_authority_sha256"])
def test_callback_missing_required_field_fails_closed_and_leaves_preregistration(
    tmp_path,
    field,
) -> None:
    plan = _plan()
    root = tmp_path / f"missing-{field}"
    good = _DeterministicFakeContractCallback(root / "plan.json")

    def incomplete(request):
        observation = good(request)
        observation.pop(field)
        return observation

    with pytest.raises(MixedModeIsolationRunnerError, match="field set is invalid"):
        execute_mixed_mode_resident_isolation_gate_v1(plan, incomplete, root)

    assert (root / "plan.json").is_file()
    assert not (root / "result.json").exists()
    assert not list((root / "captures").glob("*.json"))


def test_callback_digest_or_registered_topology_mismatch_fails_closed(
    tmp_path,
) -> None:
    plan = _plan()
    root = tmp_path / "tampered"
    good = _DeterministicFakeContractCallback(root / "plan.json")

    def tampered(request):
        observation = good(request)
        observation["run_id"] = "different"
        return observation

    with pytest.raises(MixedModeIsolationRunnerError, match="digest does not match"):
        execute_mixed_mode_resident_isolation_gate_v1(plan, tampered, root)

    topology_root = tmp_path / "topology"
    topology_good = _DeterministicFakeContractCallback(topology_root / "plan.json")

    def wrong_topology(request):
        observation = topology_good(request)
        unbound = dict(observation)
        unbound.pop("observation_sha256")
        unbound["cells"][0]["batch_position"] = 99
        return bind_resident_isolation_observation_v1(unbound)

    with pytest.raises(
        MixedModeIsolationRunnerError, match="does not match preregistered topology"
    ):
        execute_mixed_mode_resident_isolation_gate_v1(
            plan, wrong_topology, topology_root
        )


def test_plan_rejects_missing_fresh_owner_case_and_artifacts_are_nonreplaceable(
    tmp_path,
) -> None:
    plan = _plan()
    resumed_index = next(
        index
        for index, run in enumerate(plan.runs)
        if run.run_id == "checkpoint-resumed"
    )
    runs = list(plan.runs)
    runs[resumed_index] = replace(runs[resumed_index], fresh_owner=False)
    with pytest.raises(
        MixedModeIsolationRunnerError,
        match="checkpoint continuation must be marked fresh_owner",
    ):
        replace(plan, runs=tuple(runs))

    root = tmp_path / "immutable"
    callback = _DeterministicFakeContractCallback(root / "plan.json")
    execute_mixed_mode_resident_isolation_gate_v1(plan, callback, root)
    with pytest.raises(
        MixedModeIsolationRunnerError, match="artifact root already exists"
    ):
        execute_mixed_mode_resident_isolation_gate_v1(plan, callback, root)
