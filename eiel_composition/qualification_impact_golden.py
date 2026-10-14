"""Source-bound CRCA-G8 adversarial qualification-impact golden corpus."""

from __future__ import annotations

import hashlib

from emergence_contracts import ContentDigest
from emergence_contracts.qualification_impact import (
    ImpactAssuranceEffectV1,
    ImpactChangeKindV1,
    ImpactDependencyKindV1,
    ImpactNodeKindV1,
    QualificationImpactChangeV1,
    QualificationImpactClassificationV1,
    QualificationImpactDependencyEdgeV1,
    QualificationImpactFreshnessV1,
    QualificationImpactGoldenCaseV1,
    QualificationImpactGoldenOutcomeV1,
    QualificationImpactGraphV1,
    QualificationImpactNodeV1,
    QualificationImpactShadowRequestV1,
)


GOLDEN_CORPUS_ID = "crca_g8_qualification_impact_golden_v1"
GOLDEN_OUTCOME_SOURCE = (
    "_artifacts/eiel-platform-2-1-adversarial-audit-20260822/"
    "09_qualification_impact_risk.md"
)
GOLDEN_OUTCOME_SOURCE_DIGEST = ContentDigest(
    algorithm="sha256",
    hex="5ac270c17c58022aaedbfa0e416b69f299383890f6af2c31fe850310be6a464f",
)
GOLDEN_REVIEWER_ID = "operator-accepted-eiel-platform-2.1-design-package"
GOLDEN_CASES_DIGEST = ContentDigest(
    algorithm="sha256",
    hex="98ac918b4c2aa06ea1bb6e62942593f86570259b7c6667416bca049741c2c015",
)
GOLDEN_ACCEPTED_OUTCOMES_DIGEST = ContentDigest(
    algorithm="sha256",
    hex="f2774311bb1a80751341824778afcd1a447eae508978dfe15d2ed2d4b2e6563d",
)
GENERATED_AT = "2026-08-22T00:00:00Z"
CHECKED_AT = "2026-08-22T00:00:00Z"
VALID_UNTIL = "2026-08-24T00:00:00Z"
EVALUATED_AT = "2026-08-23T00:00:00Z"


def _digest(label: str) -> ContentDigest:
    return ContentDigest(
        algorithm="sha256",
        hex=hashlib.sha256(label.encode("utf-8")).hexdigest(),
    )


def _node(node_id: str, kind: ImpactNodeKindV1) -> QualificationImpactNodeV1:
    return QualificationImpactNodeV1(
        node_id=node_id,
        kind=kind,
        owner_id=f"owner:{node_id}",
        identity_digest=_digest(f"node:{node_id}"),
    )


_TARGET_KIND_BY_EDGE = {
    ImpactDependencyKindV1.IMPLEMENTS: ImpactNodeKindV1.CONTRACT,
    ImpactDependencyKindV1.IMPORTS: ImpactNodeKindV1.CONTRACT,
    ImpactDependencyKindV1.BUILDS: ImpactNodeKindV1.BUILD,
    ImpactDependencyKindV1.MATERIALIZES: ImpactNodeKindV1.ARTIFACT,
    ImpactDependencyKindV1.EXECUTES_VIA: ImpactNodeKindV1.ROUTE,
    ImpactDependencyKindV1.PRODUCES: ImpactNodeKindV1.EVIDENCE,
    ImpactDependencyKindV1.EVALUATED_BY: ImpactNodeKindV1.EVIDENCE,
    ImpactDependencyKindV1.SUPPORTS_CLAIM: ImpactNodeKindV1.CLAIM,
    ImpactDependencyKindV1.REVIEWED_BY: ImpactNodeKindV1.REVIEW,
    ImpactDependencyKindV1.QUALIFIES: ImpactNodeKindV1.QUALIFICATION_GROUP,
    ImpactDependencyKindV1.HOLDS: ImpactNodeKindV1.LIFECYCLE_OCCURRENCE,
    ImpactDependencyKindV1.SUPERSEDES: ImpactNodeKindV1.SOURCE,
}


def _case(
    *,
    ordinal: int,
    label: str,
    change_kind: ImpactChangeKindV1,
    change_effect: ImpactAssuranceEffectV1,
    expected: QualificationImpactClassificationV1,
    edge_kinds: tuple[ImpactDependencyKindV1, ...],
    declared_complete: bool = True,
    coverage_gaps: tuple[str, ...] = (),
    disconnected_group: bool = False,
    first_edge_owner_missing: bool = False,
    stale_first_edge: bool = False,
) -> QualificationImpactGoldenCaseV1:
    case_id = f"G8-{ordinal:02d}-{label}"
    change_node_id = f"{case_id}:changed"
    group_id = f"{case_id}:qualification"
    nodes = [_node(change_node_id, ImpactNodeKindV1.SOURCE)]
    edges: list[QualificationImpactDependencyEdgeV1] = []
    source_id = change_node_id
    for index, edge_kind in enumerate(edge_kinds):
        is_last = index == len(edge_kinds) - 1
        target_id = (
            group_id
            if is_last
            and edge_kind is ImpactDependencyKindV1.QUALIFIES
            and not disconnected_group
            else f"{case_id}:node:{index:02d}"
        )
        target_kind = (
            ImpactNodeKindV1.QUALIFICATION_GROUP
            if target_id == group_id
            else _TARGET_KIND_BY_EDGE[edge_kind]
        )
        nodes.append(_node(target_id, target_kind))
        edges.append(
            QualificationImpactDependencyEdgeV1(
                edge_id=f"{case_id}:edge:{index:02d}",
                source_node_id=source_id,
                target_node_id=target_id,
                kind=edge_kind,
                assurance_effect=ImpactAssuranceEffectV1.UNCHANGED,
                owner_id=(
                    None if index == 0 and first_edge_owner_missing else "impact-graph-owner"
                ),
                provenance_uri=GOLDEN_OUTCOME_SOURCE,
                provenance_digest=GOLDEN_OUTCOME_SOURCE_DIGEST,
                creation_evidence_ref=(
                    f"{GOLDEN_OUTCOME_SOURCE}#Historical adversarial corpus:{label}"
                ),
                freshness=QualificationImpactFreshnessV1(
                    rule_id="source_digest_and_two_day_review_window_v1",
                    checked_at_utc=CHECKED_AT,
                    valid_until_utc=(
                        "2026-08-22T12:00:00Z"
                        if index == 0 and stale_first_edge
                        else VALID_UNTIL
                    ),
                ),
            )
        )
        source_id = target_id
    if all(node.node_id != group_id for node in nodes):
        nodes.append(_node(group_id, ImpactNodeKindV1.QUALIFICATION_GROUP))

    graph = QualificationImpactGraphV1(
        graph_id=f"{case_id}:graph",
        nodes=tuple(sorted(nodes, key=lambda item: item.node_id)),
        edges=tuple(sorted(edges, key=lambda item: item.edge_id)),
        declared_complete=declared_complete,
        completeness_scope_ref=(
            f"{GOLDEN_OUTCOME_SOURCE}#Historical adversarial corpus:{label}"
        ),
        completeness_scope_digest=GOLDEN_OUTCOME_SOURCE_DIGEST,
        coverage_gaps=tuple(sorted(coverage_gaps)),
        contradictions=(),
        generated_at_utc=GENERATED_AT,
    )
    change = QualificationImpactChangeV1(
        change_id=f"{case_id}:change",
        node_id=change_node_id,
        kind=change_kind,
        assurance_effect=change_effect,
        owner_id="impact-change-owner",
        provenance_ref=GOLDEN_OUTCOME_SOURCE,
        provenance_digest=GOLDEN_OUTCOME_SOURCE_DIGEST,
    )
    request = QualificationImpactShadowRequestV1(
        request_id=f"{case_id}:request",
        graph=graph,
        changes=(change,),
        qualification_group_node_ids=(group_id,),
        evaluated_at_utc=EVALUATED_AT,
    )
    outcome = QualificationImpactGoldenOutcomeV1(
        qualification_group_node_id=group_id,
        classification=expected,
        reviewed_outcome_ref=(
            f"{GOLDEN_OUTCOME_SOURCE}#Historical adversarial corpus:{label}"
        ),
        reviewed_outcome_digest=GOLDEN_OUTCOME_SOURCE_DIGEST,
        reviewer_id=GOLDEN_REVIEWER_ID,
    )
    return QualificationImpactGoldenCaseV1(
        case_id=case_id,
        request=request,
        reviewed_outcomes=(outcome,),
    )


def qualification_impact_golden_cases_v1() -> tuple[QualificationImpactGoldenCaseV1, ...]:
    """Return the 13 source-bound cases required by the supplied risk audit."""

    cases = (
        _case(
            ordinal=1,
            label="dormant-adapter-unchanged-default-route",
            change_kind=ImpactChangeKindV1.DORMANT_ADAPTER,
            change_effect=ImpactAssuranceEffectV1.COMPATIBLE_INTERFACE,
            expected=QualificationImpactClassificationV1.PRESERVED_COMPATIBLE_INTERFACE,
            edge_kinds=(
                ImpactDependencyKindV1.IMPORTS,
                ImpactDependencyKindV1.IMPLEMENTS,
                ImpactDependencyKindV1.QUALIFIES,
            ),
        ),
        _case(
            ordinal=2,
            label="opt-in-facade-beside-unchanged-direct-route",
            change_kind=ImpactChangeKindV1.OPT_IN_ROUTE,
            change_effect=ImpactAssuranceEffectV1.DELTA_REQUIRED,
            expected=QualificationImpactClassificationV1.PRESERVED_UNCHANGED,
            edge_kinds=(ImpactDependencyKindV1.IMPLEMENTS,),
            disconnected_group=True,
        ),
        _case(
            ordinal=3,
            label="reporting-only-change",
            change_kind=ImpactChangeKindV1.REPORTING_ONLY,
            change_effect=ImpactAssuranceEffectV1.UNCHANGED,
            expected=QualificationImpactClassificationV1.PRESERVED_UNCHANGED,
            edge_kinds=(
                ImpactDependencyKindV1.PRODUCES,
                ImpactDependencyKindV1.QUALIFIES,
            ),
        ),
        _case(
            ordinal=4,
            label="native-abi-or-rebuilt-artifact-change",
            change_kind=ImpactChangeKindV1.NATIVE_ABI,
            change_effect=ImpactAssuranceEffectV1.DELTA_REQUIRED,
            expected=QualificationImpactClassificationV1.DELTA_QUALIFICATION_REQUIRED,
            edge_kinds=(
                ImpactDependencyKindV1.BUILDS,
                ImpactDependencyKindV1.MATERIALIZES,
                ImpactDependencyKindV1.EXECUTES_VIA,
                ImpactDependencyKindV1.QUALIFIES,
            ),
        ),
        _case(
            ordinal=5,
            label="byte-different-rebuild-equivalence-claim",
            change_kind=ImpactChangeKindV1.ARTIFACT_BYTES,
            change_effect=ImpactAssuranceEffectV1.DELTA_REQUIRED,
            expected=QualificationImpactClassificationV1.DELTA_QUALIFICATION_REQUIRED,
            edge_kinds=(
                ImpactDependencyKindV1.MATERIALIZES,
                ImpactDependencyKindV1.SUPPORTS_CLAIM,
                ImpactDependencyKindV1.QUALIFIES,
            ),
        ),
        _case(
            ordinal=6,
            label="legacy-generation-versus-absolute-tick-route",
            change_kind=ImpactChangeKindV1.LEGACY_TO_TICK_NATIVE,
            change_effect=ImpactAssuranceEffectV1.BROAD_REQUIRED,
            expected=QualificationImpactClassificationV1.BROAD_QUALIFICATION_REQUIRED,
            edge_kinds=(
                ImpactDependencyKindV1.SUPERSEDES,
                ImpactDependencyKindV1.EXECUTES_VIA,
                ImpactDependencyKindV1.QUALIFIES,
            ),
        ),
        _case(
            ordinal=7,
            label="per-cell-parameter-crosses-qualified-envelope",
            change_kind=ImpactChangeKindV1.PARAMETER_TRANSPORT,
            change_effect=ImpactAssuranceEffectV1.DELTA_REQUIRED,
            expected=QualificationImpactClassificationV1.DELTA_QUALIFICATION_REQUIRED,
            edge_kinds=(
                ImpactDependencyKindV1.IMPLEMENTS,
                ImpactDependencyKindV1.EXECUTES_VIA,
                ImpactDependencyKindV1.QUALIFIES,
            ),
        ),
        _case(
            ordinal=8,
            label="crash-before-producer-inventory",
            change_kind=ImpactChangeKindV1.PRODUCER_CRASH,
            change_effect=ImpactAssuranceEffectV1.DELTA_REQUIRED,
            expected=QualificationImpactClassificationV1.IMPACT_UNCERTAIN,
            edge_kinds=(
                ImpactDependencyKindV1.PRODUCES,
                ImpactDependencyKindV1.QUALIFIES,
            ),
            declared_complete=False,
            coverage_gaps=("producer inventory absent",),
            first_edge_owner_missing=True,
        ),
        _case(
            ordinal=9,
            label="crash-after-sealed-checkpoint",
            change_kind=ImpactChangeKindV1.SEALED_CHECKPOINT,
            change_effect=ImpactAssuranceEffectV1.COMPATIBLE_INTERFACE,
            expected=QualificationImpactClassificationV1.PRESERVED_COMPATIBLE_INTERFACE,
            edge_kinds=(
                ImpactDependencyKindV1.MATERIALIZES,
                ImpactDependencyKindV1.PRODUCES,
                ImpactDependencyKindV1.QUALIFIES,
            ),
        ),
        _case(
            ordinal=10,
            label="identical-blob-two-runs-different-holds",
            change_kind=ImpactChangeKindV1.SHARED_BLOB_HOLDS,
            change_effect=ImpactAssuranceEffectV1.DELTA_REQUIRED,
            expected=QualificationImpactClassificationV1.IMPACT_UNCERTAIN,
            edge_kinds=(
                ImpactDependencyKindV1.MATERIALIZES,
                ImpactDependencyKindV1.HOLDS,
                ImpactDependencyKindV1.QUALIFIES,
            ),
            declared_complete=False,
            coverage_gaps=("second occurrence hold disposition unknown",),
            stale_first_edge=True,
        ),
        _case(
            ordinal=11,
            label="path-rename-stable-contract-id",
            change_kind=ImpactChangeKindV1.PATH_RENAME_STABLE_CONTRACT,
            change_effect=ImpactAssuranceEffectV1.COMPATIBLE_INTERFACE,
            expected=QualificationImpactClassificationV1.PRESERVED_COMPATIBLE_INTERFACE,
            edge_kinds=(
                ImpactDependencyKindV1.IMPORTS,
                ImpactDependencyKindV1.IMPLEMENTS,
                ImpactDependencyKindV1.QUALIFIES,
            ),
        ),
        _case(
            ordinal=12,
            label="evidence-rule-or-reviewer-policy-change",
            change_kind=ImpactChangeKindV1.EVIDENCE_OR_REVIEW_POLICY,
            change_effect=ImpactAssuranceEffectV1.DELTA_REQUIRED,
            expected=QualificationImpactClassificationV1.DELTA_QUALIFICATION_REQUIRED,
            edge_kinds=(
                ImpactDependencyKindV1.EVALUATED_BY,
                ImpactDependencyKindV1.SUPPORTS_CLAIM,
                ImpactDependencyKindV1.REVIEWED_BY,
                ImpactDependencyKindV1.QUALIFIES,
            ),
        ),
        _case(
            ordinal=13,
            label="q6-integration-escape-repaired-successor",
            change_kind=ImpactChangeKindV1.QUALIFICATION_INTEGRATION_ESCAPE,
            change_effect=ImpactAssuranceEffectV1.DELTA_REQUIRED,
            expected=QualificationImpactClassificationV1.DELTA_QUALIFICATION_REQUIRED,
            edge_kinds=(
                ImpactDependencyKindV1.SUPERSEDES,
                ImpactDependencyKindV1.BUILDS,
                ImpactDependencyKindV1.MATERIALIZES,
                ImpactDependencyKindV1.EXECUTES_VIA,
                ImpactDependencyKindV1.EVALUATED_BY,
                ImpactDependencyKindV1.SUPPORTS_CLAIM,
                ImpactDependencyKindV1.REVIEWED_BY,
                ImpactDependencyKindV1.QUALIFIES,
            ),
        ),
    )
    return tuple(sorted(cases, key=lambda item: item.case_id))
