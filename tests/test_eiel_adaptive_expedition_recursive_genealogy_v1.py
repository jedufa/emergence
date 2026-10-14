from __future__ import annotations

from copy import deepcopy

import pytest

from eiel_composition.adaptive_expedition.genealogy import (
    BranchabilityClassV1,
    GenealogyIntegrityError,
    StateRefAssociationV1,
    associate_observation_state_ref_v1,
    build_experiment_dag_v1,
    semantic_state_identity_sha256_v1,
    validate_experiment_dag_v1,
)
from eiel_composition.adaptive_expedition.general_closed_loop import (
    run_general_closed_loop_v1,
)
from eiel_composition.adaptive_expedition.ledger import ACTION_SETTLED
from emergence_contracts import canonical_sha256
from emergence_contracts.adaptive_science_expedition_v1 import StateRefV1
from tests.test_eiel_adaptive_expedition_general_closed_loop_v1 import (
    _Runner,
    _ScriptedPolicy,
    _founder_basis_request,
    _frame,
    _frame_for_state,
    _general_manifest,
    _loop_inputs,
    _synthetic_state,
)


def test_state_ref_association_round_trips_and_ignores_locator_identity() -> None:
    manifest = _general_manifest(
        _ScriptedPolicy("branch_only"), _founder_basis_request()
    )
    state = _synthetic_state(manifest, 7)
    frame = _frame_for_state(manifest, state)
    association = associate_observation_state_ref_v1(
        frame=frame,
        source_result_sha256=frame.observation_frame_sha256(),
        source_frontier_ordinal=0,
    )

    assert association.branchability_class is BranchabilityClassV1.EXACT_BRANCHABLE
    assert StateRefAssociationV1.from_dict(association.to_dict()) == association
    relocated_raw = state.to_dict()
    relocated_raw["source_artifact_ref"]["locator"] = "opaque://relocated/state"
    relocated = StateRefV1.from_dict(relocated_raw)
    assert relocated.state_ref_sha256() != state.state_ref_sha256()
    assert semantic_state_identity_sha256_v1(relocated) == (
        semantic_state_identity_sha256_v1(state)
    )


def test_association_rejects_exact_evidence_on_observation_only_state() -> None:
    frame = _frame()
    raw = frame.source_state_ref.to_dict()
    raw["state_class"] = "OBSERVATION_ONLY"
    raw["exact_state_sha256"] = canonical_sha256({"tamper": True}).to_dict()
    with pytest.raises(ValueError):
        StateRefV1.from_dict(raw)


def test_real_loop_publishes_parent_action_child_dag_and_retention_bindings(
    tmp_path,
) -> None:
    result = run_general_closed_loop_v1(**_loop_inputs(tmp_path, runner=_Runner()))
    dag = validate_experiment_dag_v1(build_experiment_dag_v1(result.ledger))

    assert dag["integrity"] == {
        "duplicate_historical_actions": 0,
        "duplicate_child_roots": 0,
        "orphaned_descendant_state_refs": 0,
        "missing_state_ref_associations": 0,
        "source": "append_only_expedition_ledger_only",
    }
    edges = dag["edges"]
    assert any(edge["edge_kind"] == "PARENT_TO_ACTION" for edge in edges)
    assert any(edge["edge_kind"] == "ACTION_TO_CHILD" for edge in edges)
    decision_payload = next(
        event["payload"]
        for event in result.ledger["events"]
        if event["event_type"] == "DECISION_PREPARED"
    )
    assert decision_payload["retained_parent_bindings"]
    assert all(
        row["deletion_authority"] is False
        for row in decision_payload["retained_parent_bindings"]
    )


def test_cpu_fixture_propagates_child_state_to_grandchild_frontier(tmp_path) -> None:
    runner = _Runner()
    inputs = _loop_inputs(tmp_path, scenario="branch_only", runner=runner)
    inputs["maximum_frontiers"] = 3

    result = run_general_closed_loop_v1(**inputs)
    dispatches = [
        event
        for event in result.ledger["events"]
        if event["event_type"] == "ACTION_DISPATCH_BOUND"
    ]
    settlements = [
        event
        for event in result.ledger["events"]
        if event["event_type"] == ACTION_SETTLED
    ]

    assert result.terminal is True
    assert len(dispatches) == len(settlements) == 3
    for previous, following in zip(settlements, dispatches[1:]):
        settled_state_id = previous["payload"]["state_ref_associations"][0][
            "state_ref_sha256"
        ]["hex"]
        parent_state = StateRefV1.from_dict(
            following["payload"]["continuation"]["parent_state_ref"]
        )
        assert parent_state.state_ref_sha256().hex == settled_state_id

    dag = validate_experiment_dag_v1(build_experiment_dag_v1(result.ledger))
    assert dag["integrity"]["orphaned_descendant_state_refs"] == 0
    assert dag["integrity"]["duplicate_historical_actions"] == 0


def test_dag_rejects_child_request_reroll_and_duplicate_state_producer(tmp_path) -> None:
    result = run_general_closed_loop_v1(**_loop_inputs(tmp_path, runner=_Runner()))
    tampered = deepcopy(result.ledger)
    settled = [
        event
        for event in tampered["events"]
        if event["event_type"] == ACTION_SETTLED
    ]
    assert settled
    settled[0]["payload"]["state_ref_associations"][0]["child_request_sha256"] = {
        "algorithm": "sha256",
        "hex": "0" * 64,
    }
    with pytest.raises(GenealogyIntegrityError, match="rerolled"):
        build_experiment_dag_v1(tampered)


def test_dag_digest_and_ordering_are_authenticated(tmp_path) -> None:
    result = run_general_closed_loop_v1(
        **_loop_inputs(tmp_path, runner=_Runner())
    )
    dag = build_experiment_dag_v1(result.ledger)
    reordered = deepcopy(dag)
    reordered["nodes"] = list(reversed(reordered["nodes"]))
    with pytest.raises(GenealogyIntegrityError, match="deterministically ordered"):
        validate_experiment_dag_v1(reordered)
