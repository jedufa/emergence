"""CRCA-G8 adversarial tests for the non-authoritative impact shadow."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from eiel_composition.qualification_impact_golden import (
    GOLDEN_ACCEPTED_OUTCOMES_DIGEST,
    GOLDEN_CASES_DIGEST,
    GOLDEN_CORPUS_ID,
    GOLDEN_OUTCOME_SOURCE,
    GOLDEN_OUTCOME_SOURCE_DIGEST,
    qualification_impact_golden_cases_v1,
)
from eiel_composition.qualification_impact_shadow import (
    evaluate_qualification_impact_shadow_v1,
    run_qualification_impact_golden_corpus_v1,
    unsafe_inheritance_recommendation_v1,
)
from emergence_contracts.canonical import canonical_sha256
from emergence_contracts import ContractError
from emergence_contracts import ContentDigest
from emergence_contracts.qualification_impact import (
    ImpactAssuranceEffectV1,
    ImpactDependencyKindV1,
    ImpactShadowStatusV1,
    QualificationImpactClassificationV1,
    QualificationImpactDependencyEdgeV1,
    QualificationImpactRecommendationV1,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _case(case_id_prefix: str):
    return next(
        case
        for case in qualification_impact_golden_cases_v1()
        if case.case_id.startswith(case_id_prefix)
    )


def _request_with_edges(case, edges):
    graph = replace(
        case.request.graph,
        edges=tuple(sorted(edges, key=lambda item: item.edge_id)),
    )
    return replace(case.request, graph=graph)


def test_source_bound_golden_corpus_has_zero_unsafe_inheritance() -> None:
    source = REPO_ROOT / GOLDEN_OUTCOME_SOURCE
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        GOLDEN_OUTCOME_SOURCE_DIGEST.hex
    )

    cases = qualification_impact_golden_cases_v1()
    run_a = run_qualification_impact_golden_corpus_v1(
        corpus_id=GOLDEN_CORPUS_ID,
        cases=cases,
    )
    run_b = run_qualification_impact_golden_corpus_v1(
        corpus_id=GOLDEN_CORPUS_ID,
        cases=tuple(reversed(cases)),
    )

    assert len(cases) == 13
    assert canonical_sha256([case.to_dict() for case in cases]) == GOLDEN_CASES_DIGEST
    assert canonical_sha256(
        [
            {"case_id": case.case_id, **outcome.to_dict()}
            for case in cases
            for outcome in case.reviewed_outcomes
        ]
    ) == GOLDEN_ACCEPTED_OUTCOMES_DIGEST
    assert run_a.to_dict() == run_b.to_dict()
    assert run_a.status == "PASS"
    assert run_a.exact_match_count == 13
    assert run_a.unsafe_inheritance_count == 0
    assert run_a.uncertainty_expected_count == 2
    assert run_a.uncertainty_preserved_count == 2
    assert run_a.binding_issue_codes == ()
    assert run_a.authority_effect is False
    assert run_a.qualification_authority is False


def test_corpus_exercises_every_required_typed_dependency_kind() -> None:
    observed = {
        edge.kind
        for case in qualification_impact_golden_cases_v1()
        for edge in case.request.graph.edges
    }
    assert observed == set(ImpactDependencyKindV1)
    for case in qualification_impact_golden_cases_v1():
        for edge in case.request.graph.edges:
            assert edge.unknown_handling.value == "IMPACT_UNCERTAIN"
            assert edge.provenance_uri
            assert edge.provenance_digest == GOLDEN_OUTCOME_SOURCE_DIGEST
            assert edge.creation_evidence_ref
            assert edge.freshness.rule_id


def test_incomplete_stale_and_unowned_truth_remains_uncertain() -> None:
    crash = _case("G8-08")
    crash_run = evaluate_qualification_impact_shadow_v1(crash.request)
    assert crash_run.status is ImpactShadowStatusV1.IMPACT_UNCERTAIN
    assert crash_run.recommendations[0].classification is (
        QualificationImpactClassificationV1.IMPACT_UNCERTAIN
    )
    assert set(crash_run.issue_codes) == {
        "EDGE_OWNER_MISSING",
        "GRAPH_COVERAGE_GAP",
        "GRAPH_DECLARED_INCOMPLETE",
    }

    holds = _case("G8-10")
    holds_run = evaluate_qualification_impact_shadow_v1(holds.request)
    assert holds_run.recommendations[0].classification is (
        QualificationImpactClassificationV1.IMPACT_UNCERTAIN
    )
    assert "EDGE_STALE" in holds_run.issue_codes


@pytest.mark.parametrize(
    ("mutation", "issue"),
    [
        ("missing_provenance", "EDGE_PROVENANCE_MISSING"),
        ("missing_creation_evidence", "EDGE_CREATION_EVIDENCE_MISSING"),
        ("unknown_freshness", "EDGE_FRESHNESS_UNKNOWN"),
        ("ambiguous", "EDGE_AMBIGUOUS"),
        ("unknown_effect", "EDGE_EFFECT_UNKNOWN"),
    ],
)
def test_edge_truth_gaps_fail_closed(mutation: str, issue: str) -> None:
    case = _case("G8-04")
    first = case.request.graph.edges[0]
    if mutation == "missing_provenance":
        changed = replace(first, provenance_uri=None, provenance_digest=None)
    elif mutation == "missing_creation_evidence":
        changed = replace(first, creation_evidence_ref=None)
    elif mutation == "unknown_freshness":
        changed = replace(
            first,
            freshness=replace(
                first.freshness,
                rule_id=None,
                checked_at_utc=None,
                valid_until_utc=None,
            ),
        )
    elif mutation == "ambiguous":
        changed = replace(first, ambiguous=True)
    else:
        changed = replace(first, assurance_effect=ImpactAssuranceEffectV1.UNKNOWN)
    edges = (changed, *case.request.graph.edges[1:])
    run = evaluate_qualification_impact_shadow_v1(
        _request_with_edges(case, edges)
    )
    assert issue in run.issue_codes
    assert all(
        recommendation.classification
        is QualificationImpactClassificationV1.IMPACT_UNCERTAIN
        for recommendation in run.recommendations
    )


def test_declared_contradiction_and_dependency_cycle_fail_closed() -> None:
    case = _case("G8-04")
    contradictory = replace(
        case.request,
        graph=replace(
            case.request.graph,
            contradictions=("two owners disagree on route identity",),
        ),
    )
    contradiction_run = evaluate_qualification_impact_shadow_v1(contradictory)
    assert "GRAPH_DECLARED_CONTRADICTION" in contradiction_run.issue_codes

    first = case.request.graph.edges[0]
    back_edge = QualificationImpactDependencyEdgeV1(
        edge_id=f"{case.case_id}:edge:cycle",
        source_node_id=case.request.qualification_group_node_ids[0],
        target_node_id=case.request.changes[0].node_id,
        kind=ImpactDependencyKindV1.SUPERSEDES,
        assurance_effect=ImpactAssuranceEffectV1.UNCHANGED,
        owner_id="impact-graph-owner",
        provenance_uri=first.provenance_uri,
        provenance_digest=first.provenance_digest,
        creation_evidence_ref=first.creation_evidence_ref,
        freshness=first.freshness,
    )
    cycle_run = evaluate_qualification_impact_shadow_v1(
        _request_with_edges(case, (*case.request.graph.edges, back_edge))
    )
    assert "DEPENDENCY_CYCLE" in cycle_run.issue_codes
    assert cycle_run.status is ImpactShadowStatusV1.IMPACT_UNCERTAIN


def test_parallel_dependency_bindings_fail_closed() -> None:
    case = _case("G8-04")
    first = case.request.graph.edges[0]
    contradictory = replace(
        first,
        edge_id=f"{first.edge_id}:contradictory",
        assurance_effect=ImpactAssuranceEffectV1.BROAD_REQUIRED,
    )
    contradictory_run = evaluate_qualification_impact_shadow_v1(
        _request_with_edges(case, (*case.request.graph.edges, contradictory))
    )
    assert "EDGE_PARALLEL_BINDING_CONTRADICTION" in contradictory_run.issue_codes
    assert contradictory_run.status is ImpactShadowStatusV1.IMPACT_UNCERTAIN

    duplicate = replace(first, edge_id=f"{first.edge_id}:duplicate")
    duplicate_run = evaluate_qualification_impact_shadow_v1(
        _request_with_edges(case, (*case.request.graph.edges, duplicate))
    )
    assert "EDGE_PARALLEL_BINDING_DUPLICATE" in duplicate_run.issue_codes
    assert duplicate_run.status is ImpactShadowStatusV1.IMPACT_UNCERTAIN


def test_golden_comparator_detects_optimistic_inheritance() -> None:
    assert unsafe_inheritance_recommendation_v1(
        QualificationImpactClassificationV1.PRESERVED_UNCHANGED,
        QualificationImpactClassificationV1.DELTA_QUALIFICATION_REQUIRED,
    )


def test_fabricated_golden_corpus_binding_fails_closed() -> None:
    reporting = _case("G8-03")
    reviewed = replace(
        reporting.reviewed_outcomes[0],
        classification=(
            QualificationImpactClassificationV1.DELTA_QUALIFICATION_REQUIRED
        ),
    )
    tampered = replace(reporting, reviewed_outcomes=(reviewed,))
    run = run_qualification_impact_golden_corpus_v1(
        corpus_id="unsafe-inheritance-sentinel",
        cases=(tampered,),
    )
    assert run.status == "FAIL"
    assert run.exact_match_count == 0
    assert run.unsafe_inheritance_count == 0
    assert set(run.binding_issue_codes) == {
        "GOLDEN_ACCEPTED_OUTCOMES_DIGEST_MISMATCH",
        "GOLDEN_CASES_DIGEST_MISMATCH",
        "GOLDEN_CORPUS_ID_MISMATCH",
    }
    assert all(
        recommendation.classification
        is QualificationImpactClassificationV1.IMPACT_UNCERTAIN
        for result in run.case_results
        for recommendation in result.shadow_run.recommendations
    )


def test_fabricated_golden_source_binding_fails_closed() -> None:
    cases = qualification_impact_golden_cases_v1()
    reporting = _case("G8-03")
    fabricated = replace(
        reporting.reviewed_outcomes[0],
        reviewed_outcome_ref="fabricated://outcome",
        reviewed_outcome_digest=ContentDigest(
            algorithm="sha256",
            hex="0" * 64,
        ),
        reviewer_id="fabricated-reviewer",
    )
    tampered = replace(reporting, reviewed_outcomes=(fabricated,))
    tampered_cases = tuple(
        tampered if case.case_id == reporting.case_id else case for case in cases
    )
    run = run_qualification_impact_golden_corpus_v1(
        corpus_id=GOLDEN_CORPUS_ID,
        cases=tampered_cases,
    )
    assert run.status == "FAIL"
    assert "GOLDEN_SOURCE_BINDING_MISMATCH" in run.binding_issue_codes
    assert "GOLDEN_CASES_DIGEST_MISMATCH" in run.binding_issue_codes
    assert "GOLDEN_ACCEPTED_OUTCOMES_DIGEST_MISMATCH" in run.binding_issue_codes
    assert all(
        recommendation.classification
        is QualificationImpactClassificationV1.IMPACT_UNCERTAIN
        for result in run.case_results
        for recommendation in result.shadow_run.recommendations
    )


def test_authority_flags_cannot_be_enabled() -> None:
    case = _case("G8-04")
    run = evaluate_qualification_impact_shadow_v1(case.request)
    recommendation = run.recommendations[0]
    with pytest.raises(ContractError, match="shadow-only"):
        QualificationImpactRecommendationV1(
            qualification_group_node_id=recommendation.qualification_group_node_id,
            classification=recommendation.classification,
            reason_codes=recommendation.reason_codes,
            affected_node_ids=recommendation.affected_node_ids,
            graph_digest=recommendation.graph_digest,
            qualification_authority=True,
        )
    assert recommendation.authority_effect is False
    assert recommendation.automatic_invalidation is False
    assert recommendation.requires_semantic_review is True


def test_shadow_is_not_integrated_with_qualification_manager_or_lifecycle() -> None:
    manager = REPO_ROOT / "tools/actors/qualification_manager.py"
    tree = ast.parse(manager.read_text(encoding="utf-8"), filename=str(manager))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("qualification_impact_shadow" in module for module in imported)
    assert not any("qualification_impact_golden" in module for module in imported)

    run = evaluate_qualification_impact_shadow_v1(_case("G8-04").request)
    payload = run.to_dict()
    assert payload["authority_effect"] is False
    assert payload["qualification_authority"] is False
    assert "release" not in repr(payload).lower()
