"""Mechanical, non-authoritative qualification-impact shadow evaluation."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime
from typing import Iterable

from emergence_contracts.canonical import canonical_sha256
from emergence_contracts.qualification_impact import (
    ImpactAssuranceEffectV1,
    ImpactNodeKindV1,
    ImpactShadowStatusV1,
    QualificationImpactClassificationV1,
    QualificationImpactGoldenCaseResultV1,
    QualificationImpactGoldenCaseV1,
    QualificationImpactGoldenRunV1,
    QualificationImpactRecommendationV1,
    QualificationImpactShadowRequestV1,
    QualificationImpactShadowRunV1,
)


class QualificationImpactShadowError(ValueError):
    """The caller supplied a contradictory corpus invocation."""


_EFFECT_RANK = {
    ImpactAssuranceEffectV1.UNCHANGED: 0,
    ImpactAssuranceEffectV1.COMPATIBLE_INTERFACE: 1,
    ImpactAssuranceEffectV1.DELTA_REQUIRED: 2,
    ImpactAssuranceEffectV1.BROAD_REQUIRED: 3,
    ImpactAssuranceEffectV1.UNKNOWN: 4,
}
_CLASSIFICATION_BY_RANK = {
    0: QualificationImpactClassificationV1.PRESERVED_UNCHANGED,
    1: QualificationImpactClassificationV1.PRESERVED_COMPATIBLE_INTERFACE,
    2: QualificationImpactClassificationV1.DELTA_QUALIFICATION_REQUIRED,
    3: QualificationImpactClassificationV1.BROAD_QUALIFICATION_REQUIRED,
    4: QualificationImpactClassificationV1.IMPACT_UNCERTAIN,
}


def _parse_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")


def _cycle_present(adjacency: dict[str, tuple[str, ...]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> bool:
        if node_id in visiting:
            return True
        if node_id in visited:
            return False
        visiting.add(node_id)
        if any(visit(target) for target in adjacency.get(node_id, ())):
            return True
        visiting.remove(node_id)
        visited.add(node_id)
        return False

    return any(visit(node_id) for node_id in sorted(adjacency))


def _graph_issue_codes(request: QualificationImpactShadowRequestV1) -> tuple[str, ...]:
    graph = request.graph
    issues: set[str] = set()
    nodes = {node.node_id: node for node in graph.nodes}
    evaluated_at = _parse_utc(request.evaluated_at_utc)

    if not graph.declared_complete:
        issues.add("GRAPH_DECLARED_INCOMPLETE")
    if graph.coverage_gaps:
        issues.add("GRAPH_COVERAGE_GAP")
    if graph.contradictions:
        issues.add("GRAPH_DECLARED_CONTRADICTION")

    for node in graph.nodes:
        if node.owner_id is None:
            issues.add("NODE_OWNER_MISSING")
        if node.identity_digest is None:
            issues.add("NODE_IDENTITY_DIGEST_MISSING")

    adjacency_lists: dict[str, list[str]] = defaultdict(list)
    parallel_edges: dict[tuple[str, str, object], list[object]] = defaultdict(list)
    for edge in graph.edges:
        parallel_edges[(edge.source_node_id, edge.target_node_id, edge.kind)].append(edge)
        if edge.source_node_id not in nodes or edge.target_node_id not in nodes:
            issues.add("EDGE_ENDPOINT_MISSING")
        else:
            adjacency_lists[edge.source_node_id].append(edge.target_node_id)
        if edge.owner_id is None:
            issues.add("EDGE_OWNER_MISSING")
        if edge.provenance_uri is None or edge.provenance_digest is None:
            issues.add("EDGE_PROVENANCE_MISSING")
        if edge.creation_evidence_ref is None:
            issues.add("EDGE_CREATION_EVIDENCE_MISSING")
        freshness = edge.freshness
        if (
            freshness.rule_id is None
            or freshness.checked_at_utc is None
            or freshness.valid_until_utc is None
        ):
            issues.add("EDGE_FRESHNESS_UNKNOWN")
        else:
            checked_at = _parse_utc(freshness.checked_at_utc)
            valid_until = _parse_utc(freshness.valid_until_utc)
            if checked_at > evaluated_at:
                issues.add("EDGE_FRESHNESS_CHECK_IN_FUTURE")
            if valid_until < checked_at:
                issues.add("EDGE_FRESHNESS_CONTRADICTORY")
            if evaluated_at > valid_until:
                issues.add("EDGE_STALE")
        if edge.ambiguous:
            issues.add("EDGE_AMBIGUOUS")
        if edge.assurance_effect is ImpactAssuranceEffectV1.UNKNOWN:
            issues.add("EDGE_EFFECT_UNKNOWN")

    for edges in parallel_edges.values():
        if len(edges) < 2:
            continue
        bindings = {
            (
                edge.assurance_effect,
                edge.owner_id,
                edge.provenance_uri,
                str(edge.provenance_digest),
                edge.creation_evidence_ref,
                tuple(sorted(edge.freshness.to_dict().items())),
                edge.ambiguous,
                edge.unknown_handling,
            )
            for edge in edges
        }
        issues.add(
            "EDGE_PARALLEL_BINDING_CONTRADICTION"
            if len(bindings) > 1
            else "EDGE_PARALLEL_BINDING_DUPLICATE"
        )

    adjacency = {
        source: tuple(sorted(set(targets)))
        for source, targets in adjacency_lists.items()
    }
    if _cycle_present(adjacency):
        issues.add("DEPENDENCY_CYCLE")

    for change in request.changes:
        if change.node_id not in nodes:
            issues.add("CHANGE_NODE_MISSING")
        if change.owner_id is None:
            issues.add("CHANGE_OWNER_MISSING")
        if change.provenance_ref is None or change.provenance_digest is None:
            issues.add("CHANGE_PROVENANCE_MISSING")
        if change.assurance_effect is ImpactAssuranceEffectV1.UNKNOWN:
            issues.add("CHANGE_EFFECT_UNKNOWN")

    for group_id in request.qualification_group_node_ids:
        node = nodes.get(group_id)
        if node is None:
            issues.add("QUALIFICATION_GROUP_NODE_MISSING")
        elif node.kind is not ImpactNodeKindV1.QUALIFICATION_GROUP:
            issues.add("QUALIFICATION_GROUP_KIND_INVALID")
    return tuple(sorted(issues))


def _recommendation_effects(
    request: QualificationImpactShadowRequestV1,
) -> tuple[dict[str, int], dict[str, tuple[str, ...]]]:
    adjacency: dict[str, list[tuple[str, int]]] = defaultdict(list)
    reverse_adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in request.graph.edges:
        adjacency[edge.source_node_id].append(
            (edge.target_node_id, _EFFECT_RANK[edge.assurance_effect])
        )
        reverse_adjacency[edge.target_node_id].append(edge.source_node_id)
    for values in adjacency.values():
        values.sort()
    for values in reverse_adjacency.values():
        values.sort()

    ranks: dict[str, int] = {}
    reached: set[str] = set()
    queue: deque[tuple[str, int]] = deque()
    for change in request.changes:
        rank = _EFFECT_RANK[change.assurance_effect]
        if rank > ranks.get(change.node_id, -1):
            ranks[change.node_id] = rank
            queue.append((change.node_id, rank))
        reached.add(change.node_id)

    while queue:
        node_id, path_rank = queue.popleft()
        for target_id, edge_rank in adjacency.get(node_id, []):
            target_rank = max(path_rank, edge_rank)
            reached.add(target_id)
            if target_rank > ranks.get(target_id, -1):
                ranks[target_id] = target_rank
                queue.append((target_id, target_rank))

    affected_by_group: dict[str, tuple[str, ...]] = {}
    for group_id in request.qualification_group_node_ids:
        if group_id not in reached:
            affected_by_group[group_id] = ()
            continue
        ancestors = {group_id}
        ancestor_queue = deque((group_id,))
        while ancestor_queue:
            node_id = ancestor_queue.popleft()
            for source_id in reverse_adjacency.get(node_id, []):
                if source_id not in ancestors:
                    ancestors.add(source_id)
                    ancestor_queue.append(source_id)
        affected_by_group[group_id] = tuple(sorted(reached & ancestors))
    return (
        {
            group_id: ranks.get(group_id, 0)
            for group_id in request.qualification_group_node_ids
        },
        affected_by_group,
    )


def evaluate_qualification_impact_shadow_v1(
    request: QualificationImpactShadowRequestV1,
) -> QualificationImpactShadowRunV1:
    """Return recommendations only; never apply, publish, or inherit them."""

    if not isinstance(request, QualificationImpactShadowRequestV1):
        raise QualificationImpactShadowError(
            "request must be QualificationImpactShadowRequestV1"
        )
    issues = _graph_issue_codes(request)
    graph_digest = request.graph.graph_digest()
    recommendations: list[QualificationImpactRecommendationV1] = []
    if issues:
        for group_id in request.qualification_group_node_ids:
            recommendations.append(
                QualificationImpactRecommendationV1(
                    qualification_group_node_id=group_id,
                    classification=(
                        QualificationImpactClassificationV1.IMPACT_UNCERTAIN
                    ),
                    reason_codes=issues,
                    affected_node_ids=(),
                    graph_digest=graph_digest,
                )
            )
        status = ImpactShadowStatusV1.IMPACT_UNCERTAIN
    else:
        ranks, affected = _recommendation_effects(request)
        for group_id in request.qualification_group_node_ids:
            reached = bool(affected[group_id])
            reason = (
                f"MECHANICAL_CLOSURE_{_CLASSIFICATION_BY_RANK[ranks[group_id]].value}"
                if reached
                else "NO_DEPENDENCY_PATH_IN_DECLARED_COMPLETE_SCOPE"
            )
            recommendations.append(
                QualificationImpactRecommendationV1(
                    qualification_group_node_id=group_id,
                    classification=_CLASSIFICATION_BY_RANK[ranks[group_id]],
                    reason_codes=(reason, "SHADOW_ONLY_SEMANTIC_REVIEW_REQUIRED"),
                    affected_node_ids=affected[group_id],
                    graph_digest=graph_digest,
                )
            )
        status = ImpactShadowStatusV1.RECOMMENDATIONS_READY
    return QualificationImpactShadowRunV1(
        request_id=request.request_id,
        status=status,
        issue_codes=issues,
        recommendations=tuple(recommendations),
        graph_digest=graph_digest,
    )


def unsafe_inheritance_recommendation_v1(
    recommended: QualificationImpactClassificationV1,
    reviewed: QualificationImpactClassificationV1,
) -> bool:
    if recommended is QualificationImpactClassificationV1.PRESERVED_UNCHANGED:
        return reviewed is not QualificationImpactClassificationV1.PRESERVED_UNCHANGED
    if recommended is QualificationImpactClassificationV1.PRESERVED_COMPATIBLE_INTERFACE:
        return reviewed not in {
            QualificationImpactClassificationV1.PRESERVED_UNCHANGED,
            QualificationImpactClassificationV1.PRESERVED_COMPATIBLE_INTERFACE,
        }
    return False


def _golden_binding_issue_codes(
    *,
    corpus_id: str,
    cases: tuple[QualificationImpactGoldenCaseV1, ...],
) -> tuple[str, ...]:
    from .qualification_impact_golden import (
        GOLDEN_ACCEPTED_OUTCOMES_DIGEST,
        GOLDEN_CASES_DIGEST,
        GOLDEN_CORPUS_ID,
        GOLDEN_OUTCOME_SOURCE,
        GOLDEN_OUTCOME_SOURCE_DIGEST,
        GOLDEN_REVIEWER_ID,
    )

    issues: set[str] = set()
    if corpus_id != GOLDEN_CORPUS_ID:
        issues.add("GOLDEN_CORPUS_ID_MISMATCH")
    if canonical_sha256([case.to_dict() for case in cases]) != GOLDEN_CASES_DIGEST:
        issues.add("GOLDEN_CASES_DIGEST_MISMATCH")
    accepted_outcomes = [
        {"case_id": case.case_id, **outcome.to_dict()}
        for case in cases
        for outcome in case.reviewed_outcomes
    ]
    if canonical_sha256(accepted_outcomes) != GOLDEN_ACCEPTED_OUTCOMES_DIGEST:
        issues.add("GOLDEN_ACCEPTED_OUTCOMES_DIGEST_MISMATCH")
    source_prefix = f"{GOLDEN_OUTCOME_SOURCE}#Historical adversarial corpus:"
    if any(
        not outcome.reviewed_outcome_ref.startswith(source_prefix)
        or outcome.reviewed_outcome_digest != GOLDEN_OUTCOME_SOURCE_DIGEST
        or outcome.reviewer_id != GOLDEN_REVIEWER_ID
        for case in cases
        for outcome in case.reviewed_outcomes
    ):
        issues.add("GOLDEN_SOURCE_BINDING_MISMATCH")
    return tuple(sorted(issues))


def _forced_uncertain_shadow_run(
    request: QualificationImpactShadowRequestV1,
    binding_issues: tuple[str, ...],
) -> QualificationImpactShadowRunV1:
    graph_digest = request.graph.graph_digest()
    issues = tuple(sorted(set(_graph_issue_codes(request)) | set(binding_issues)))
    return QualificationImpactShadowRunV1(
        request_id=request.request_id,
        status=ImpactShadowStatusV1.IMPACT_UNCERTAIN,
        issue_codes=issues,
        recommendations=tuple(
            QualificationImpactRecommendationV1(
                qualification_group_node_id=group_id,
                classification=QualificationImpactClassificationV1.IMPACT_UNCERTAIN,
                reason_codes=issues,
                affected_node_ids=(),
                graph_digest=graph_digest,
            )
            for group_id in request.qualification_group_node_ids
        ),
        graph_digest=graph_digest,
    )


def run_qualification_impact_golden_corpus_v1(
    *,
    corpus_id: str,
    cases: Iterable[QualificationImpactGoldenCaseV1],
) -> QualificationImpactGoldenRunV1:
    case_tuple = tuple(sorted(cases, key=lambda item: item.case_id))
    if not corpus_id or not case_tuple:
        raise QualificationImpactShadowError("corpus_id and cases are required")
    if len({case.case_id for case in case_tuple}) != len(case_tuple):
        raise QualificationImpactShadowError("golden case ids must be unique")
    binding_issues = _golden_binding_issue_codes(
        corpus_id=corpus_id,
        cases=case_tuple,
    )

    results: list[QualificationImpactGoldenCaseResultV1] = []
    for case in case_tuple:
        run = (
            _forced_uncertain_shadow_run(case.request, binding_issues)
            if binding_issues
            else evaluate_qualification_impact_shadow_v1(case.request)
        )
        recommended = {
            item.qualification_group_node_id: item.classification
            for item in run.recommendations
        }
        reviewed = {
            item.qualification_group_node_id: item.classification
            for item in case.reviewed_outcomes
        }
        exact = recommended == reviewed
        unsafe = sum(
            unsafe_inheritance_recommendation_v1(
                recommended[group_id], classification
            )
            for group_id, classification in reviewed.items()
        )
        uncertainty_expected = any(
            classification is QualificationImpactClassificationV1.IMPACT_UNCERTAIN
            for classification in reviewed.values()
        )
        uncertainty_preserved = all(
            recommended[group_id]
            is QualificationImpactClassificationV1.IMPACT_UNCERTAIN
            for group_id, classification in reviewed.items()
            if classification is QualificationImpactClassificationV1.IMPACT_UNCERTAIN
        )
        results.append(
            QualificationImpactGoldenCaseResultV1(
                case_id=case.case_id,
                shadow_run=run,
                exact_match=exact,
                unsafe_inheritance_count=unsafe,
                uncertainty_expected=uncertainty_expected,
                uncertainty_preserved=uncertainty_preserved,
            )
        )

    unsafe_count = sum(item.unsafe_inheritance_count for item in results)
    exact_count = sum(item.exact_match for item in results)
    uncertainty_expected_count = sum(item.uncertainty_expected for item in results)
    uncertainty_preserved_count = sum(
        item.uncertainty_expected and item.uncertainty_preserved for item in results
    )
    status = (
        "PASS"
        if unsafe_count == 0
        and exact_count == len(results)
        and uncertainty_preserved_count == uncertainty_expected_count
        and not binding_issues
        else "FAIL"
    )
    return QualificationImpactGoldenRunV1(
        corpus_id=corpus_id,
        case_results=tuple(results),
        unsafe_inheritance_count=unsafe_count,
        exact_match_count=exact_count,
        uncertainty_expected_count=uncertainty_expected_count,
        uncertainty_preserved_count=uncertainty_preserved_count,
        status=status,
        binding_issue_codes=binding_issues,
    )
