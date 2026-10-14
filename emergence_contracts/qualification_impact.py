"""Typed, shadow-only qualification-impact graph contracts.

These DTOs can describe and compare an impact recommendation.  They cannot
invalidate evidence, inherit qualification, publish authority, release an
artifact, or replace substantive semantic review.  Missing dependency truth
is represented explicitly so the composition layer can return
``IMPACT_UNCERTAIN`` without guessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ._validation import require_instance, require_sorted_unique_text, require_text
from .canonical import canonical_sha256
from .errors import contract_invalid
from .ids import ContentDigest, SchemaVersion


QUALIFICATION_IMPACT_NODE_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_node_v1"
)
QUALIFICATION_IMPACT_FRESHNESS_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_freshness_v1"
)
QUALIFICATION_IMPACT_EDGE_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_edge_v1"
)
QUALIFICATION_IMPACT_GRAPH_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_graph_v1"
)
QUALIFICATION_IMPACT_CHANGE_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_change_v1"
)
QUALIFICATION_IMPACT_SHADOW_REQUEST_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_shadow_request_v1"
)
QUALIFICATION_IMPACT_RECOMMENDATION_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_recommendation_v1"
)
QUALIFICATION_IMPACT_SHADOW_RUN_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_shadow_run_v1"
)
QUALIFICATION_IMPACT_GOLDEN_OUTCOME_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_golden_outcome_v1"
)
QUALIFICATION_IMPACT_GOLDEN_CASE_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_golden_case_v1"
)
QUALIFICATION_IMPACT_GOLDEN_CASE_RESULT_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_golden_case_result_v1"
)
QUALIFICATION_IMPACT_GOLDEN_RUN_SCHEMA_VERSION = SchemaVersion(
    "crca_qualification_impact_golden_run_v1"
)


def _utc(value: object, *, field: str) -> str:
    require_text(value, field=field)
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z"):
        raise contract_invalid(
            "timestamp must use canonical second-resolution UTC",
            field=field,
        )
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise contract_invalid("timestamp is invalid", field=field) from exc
    return value


def _optional_text(value: str | None, *, field: str) -> None:
    if value is not None:
        require_text(value, field=field)


def _optional_digest(value: ContentDigest | None, *, field: str) -> None:
    if value is not None:
        require_instance(value, ContentDigest, field=field)


def _schema(actual: SchemaVersion, expected: SchemaVersion) -> None:
    require_instance(actual, SchemaVersion, field="schema_version")
    if actual != expected:
        raise contract_invalid("unsupported qualification-impact schema", field="schema_version")


def _digest(value: ContentDigest | None) -> dict[str, str] | None:
    return None if value is None else value.to_dict()


class ImpactNodeKindV1(str, Enum):
    SOURCE = "SOURCE"
    CONTRACT = "CONTRACT"
    BUILD = "BUILD"
    ARTIFACT = "ARTIFACT"
    ROUTE = "ROUTE"
    RUN_OCCURRENCE = "RUN_OCCURRENCE"
    EVIDENCE = "EVIDENCE"
    CLAIM = "CLAIM"
    REVIEW = "REVIEW"
    QUALIFICATION_GROUP = "QUALIFICATION_GROUP"
    LIFECYCLE_OCCURRENCE = "LIFECYCLE_OCCURRENCE"


class ImpactDependencyKindV1(str, Enum):
    IMPLEMENTS = "IMPLEMENTS"
    IMPORTS = "IMPORTS"
    BUILDS = "BUILDS"
    MATERIALIZES = "MATERIALIZES"
    EXECUTES_VIA = "EXECUTES_VIA"
    PRODUCES = "PRODUCES"
    EVALUATED_BY = "EVALUATED_BY"
    SUPPORTS_CLAIM = "SUPPORTS_CLAIM"
    REVIEWED_BY = "REVIEWED_BY"
    QUALIFIES = "QUALIFIES"
    HOLDS = "HOLDS"
    SUPERSEDES = "SUPERSEDES"


class ImpactAssuranceEffectV1(str, Enum):
    UNCHANGED = "UNCHANGED"
    COMPATIBLE_INTERFACE = "COMPATIBLE_INTERFACE"
    DELTA_REQUIRED = "DELTA_REQUIRED"
    BROAD_REQUIRED = "BROAD_REQUIRED"
    UNKNOWN = "UNKNOWN"


class QualificationImpactClassificationV1(str, Enum):
    PRESERVED_UNCHANGED = "PRESERVED_UNCHANGED"
    PRESERVED_COMPATIBLE_INTERFACE = "PRESERVED_COMPATIBLE_INTERFACE"
    DELTA_QUALIFICATION_REQUIRED = "DELTA_QUALIFICATION_REQUIRED"
    BROAD_QUALIFICATION_REQUIRED = "BROAD_QUALIFICATION_REQUIRED"
    IMPACT_UNCERTAIN = "IMPACT_UNCERTAIN"


class ImpactChangeKindV1(str, Enum):
    DORMANT_ADAPTER = "DORMANT_ADAPTER"
    OPT_IN_ROUTE = "OPT_IN_ROUTE"
    REPORTING_ONLY = "REPORTING_ONLY"
    NATIVE_ABI = "NATIVE_ABI"
    ARTIFACT_BYTES = "ARTIFACT_BYTES"
    LEGACY_TO_TICK_NATIVE = "LEGACY_TO_TICK_NATIVE"
    PARAMETER_TRANSPORT = "PARAMETER_TRANSPORT"
    PRODUCER_CRASH = "PRODUCER_CRASH"
    SEALED_CHECKPOINT = "SEALED_CHECKPOINT"
    SHARED_BLOB_HOLDS = "SHARED_BLOB_HOLDS"
    PATH_RENAME_STABLE_CONTRACT = "PATH_RENAME_STABLE_CONTRACT"
    EVIDENCE_OR_REVIEW_POLICY = "EVIDENCE_OR_REVIEW_POLICY"
    QUALIFICATION_INTEGRATION_ESCAPE = "QUALIFICATION_INTEGRATION_ESCAPE"
    UNKNOWN = "UNKNOWN"


class ImpactShadowStatusV1(str, Enum):
    RECOMMENDATIONS_READY = "RECOMMENDATIONS_READY"
    IMPACT_UNCERTAIN = "IMPACT_UNCERTAIN"


class ImpactUnknownHandlingV1(str, Enum):
    IMPACT_UNCERTAIN = "IMPACT_UNCERTAIN"


@dataclass(frozen=True, slots=True)
class QualificationImpactNodeV1:
    node_id: str
    kind: ImpactNodeKindV1
    owner_id: str | None
    identity_digest: ContentDigest | None
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_NODE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.node_id, field="node_id")
        require_instance(self.kind, ImpactNodeKindV1, field="kind")
        _optional_text(self.owner_id, field="owner_id")
        _optional_digest(self.identity_digest, field="identity_digest")
        _schema(self.schema_version, QUALIFICATION_IMPACT_NODE_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "node_id": self.node_id,
            "kind": self.kind.value,
            "owner_id": self.owner_id,
            "identity_digest": _digest(self.identity_digest),
        }


@dataclass(frozen=True, slots=True)
class QualificationImpactFreshnessV1:
    rule_id: str | None
    checked_at_utc: str | None
    valid_until_utc: str | None
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_FRESHNESS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _optional_text(self.rule_id, field="rule_id")
        if self.checked_at_utc is not None:
            _utc(self.checked_at_utc, field="checked_at_utc")
        if self.valid_until_utc is not None:
            _utc(self.valid_until_utc, field="valid_until_utc")
        _schema(self.schema_version, QUALIFICATION_IMPACT_FRESHNESS_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "rule_id": self.rule_id,
            "checked_at_utc": self.checked_at_utc,
            "valid_until_utc": self.valid_until_utc,
        }


@dataclass(frozen=True, slots=True)
class QualificationImpactDependencyEdgeV1:
    edge_id: str
    source_node_id: str
    target_node_id: str
    kind: ImpactDependencyKindV1
    assurance_effect: ImpactAssuranceEffectV1
    owner_id: str | None
    provenance_uri: str | None
    provenance_digest: ContentDigest | None
    creation_evidence_ref: str | None
    freshness: QualificationImpactFreshnessV1
    ambiguous: bool = False
    unknown_handling: ImpactUnknownHandlingV1 = (
        ImpactUnknownHandlingV1.IMPACT_UNCERTAIN
    )
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_EDGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("edge_id", "source_node_id", "target_node_id"):
            require_text(getattr(self, field_name), field=field_name)
        require_instance(self.kind, ImpactDependencyKindV1, field="kind")
        require_instance(
            self.assurance_effect,
            ImpactAssuranceEffectV1,
            field="assurance_effect",
        )
        _optional_text(self.owner_id, field="owner_id")
        _optional_text(self.provenance_uri, field="provenance_uri")
        _optional_digest(self.provenance_digest, field="provenance_digest")
        _optional_text(self.creation_evidence_ref, field="creation_evidence_ref")
        require_instance(
            self.freshness,
            QualificationImpactFreshnessV1,
            field="freshness",
        )
        if type(self.ambiguous) is not bool:
            raise contract_invalid("ambiguous must be an exact boolean", field="ambiguous")
        if self.unknown_handling is not ImpactUnknownHandlingV1.IMPACT_UNCERTAIN:
            raise contract_invalid(
                "unknown dependency truth must fail closed",
                field="unknown_handling",
            )
        _schema(self.schema_version, QUALIFICATION_IMPACT_EDGE_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "edge_id": self.edge_id,
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "kind": self.kind.value,
            "assurance_effect": self.assurance_effect.value,
            "owner_id": self.owner_id,
            "provenance_uri": self.provenance_uri,
            "provenance_digest": _digest(self.provenance_digest),
            "creation_evidence_ref": self.creation_evidence_ref,
            "freshness": self.freshness.to_dict(),
            "ambiguous": self.ambiguous,
            "unknown_handling": self.unknown_handling.value,
        }


@dataclass(frozen=True, slots=True)
class QualificationImpactGraphV1:
    graph_id: str
    nodes: tuple[QualificationImpactNodeV1, ...]
    edges: tuple[QualificationImpactDependencyEdgeV1, ...]
    declared_complete: bool
    completeness_scope_ref: str
    completeness_scope_digest: ContentDigest
    coverage_gaps: tuple[str, ...]
    contradictions: tuple[str, ...]
    generated_at_utc: str
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_GRAPH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.graph_id, field="graph_id")
        if not isinstance(self.nodes, tuple) or any(
            not isinstance(node, QualificationImpactNodeV1) for node in self.nodes
        ):
            raise contract_invalid("nodes must be a canonical node tuple", field="nodes")
        if not isinstance(self.edges, tuple) or any(
            not isinstance(edge, QualificationImpactDependencyEdgeV1)
            for edge in self.edges
        ):
            raise contract_invalid("edges must be a canonical edge tuple", field="edges")
        node_ids = [node.node_id for node in self.nodes]
        edge_ids = [edge.edge_id for edge in self.edges]
        if node_ids != sorted(node_ids) or len(node_ids) != len(set(node_ids)):
            raise contract_invalid("nodes must be sorted and unique", field="nodes")
        if edge_ids != sorted(edge_ids) or len(edge_ids) != len(set(edge_ids)):
            raise contract_invalid("edges must be sorted and unique", field="edges")
        if type(self.declared_complete) is not bool:
            raise contract_invalid(
                "declared_complete must be an exact boolean",
                field="declared_complete",
            )
        require_text(self.completeness_scope_ref, field="completeness_scope_ref")
        require_instance(
            self.completeness_scope_digest,
            ContentDigest,
            field="completeness_scope_digest",
        )
        require_sorted_unique_text(self.coverage_gaps, field="coverage_gaps")
        require_sorted_unique_text(self.contradictions, field="contradictions")
        _utc(self.generated_at_utc, field="generated_at_utc")
        _schema(self.schema_version, QUALIFICATION_IMPACT_GRAPH_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "graph_id": self.graph_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "declared_complete": self.declared_complete,
            "completeness_scope_ref": self.completeness_scope_ref,
            "completeness_scope_digest": self.completeness_scope_digest.to_dict(),
            "coverage_gaps": list(self.coverage_gaps),
            "contradictions": list(self.contradictions),
            "generated_at_utc": self.generated_at_utc,
        }

    def graph_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class QualificationImpactChangeV1:
    change_id: str
    node_id: str
    kind: ImpactChangeKindV1
    assurance_effect: ImpactAssuranceEffectV1
    owner_id: str | None
    provenance_ref: str | None
    provenance_digest: ContentDigest | None
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_CHANGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.change_id, field="change_id")
        require_text(self.node_id, field="node_id")
        require_instance(self.kind, ImpactChangeKindV1, field="kind")
        require_instance(
            self.assurance_effect,
            ImpactAssuranceEffectV1,
            field="assurance_effect",
        )
        _optional_text(self.owner_id, field="owner_id")
        _optional_text(self.provenance_ref, field="provenance_ref")
        _optional_digest(self.provenance_digest, field="provenance_digest")
        _schema(self.schema_version, QUALIFICATION_IMPACT_CHANGE_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "change_id": self.change_id,
            "node_id": self.node_id,
            "kind": self.kind.value,
            "assurance_effect": self.assurance_effect.value,
            "owner_id": self.owner_id,
            "provenance_ref": self.provenance_ref,
            "provenance_digest": _digest(self.provenance_digest),
        }


@dataclass(frozen=True, slots=True)
class QualificationImpactShadowRequestV1:
    request_id: str
    graph: QualificationImpactGraphV1
    changes: tuple[QualificationImpactChangeV1, ...]
    qualification_group_node_ids: tuple[str, ...]
    evaluated_at_utc: str
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_SHADOW_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.request_id, field="request_id")
        require_instance(self.graph, QualificationImpactGraphV1, field="graph")
        if not isinstance(self.changes, tuple) or not self.changes or any(
            not isinstance(change, QualificationImpactChangeV1)
            for change in self.changes
        ):
            raise contract_invalid("changes must be a nonempty canonical tuple", field="changes")
        change_ids = [change.change_id for change in self.changes]
        if change_ids != sorted(change_ids) or len(change_ids) != len(set(change_ids)):
            raise contract_invalid("changes must be sorted and unique", field="changes")
        require_sorted_unique_text(
            self.qualification_group_node_ids,
            field="qualification_group_node_ids",
        )
        if not self.qualification_group_node_ids:
            raise contract_invalid(
                "at least one qualification group is required",
                field="qualification_group_node_ids",
            )
        _utc(self.evaluated_at_utc, field="evaluated_at_utc")
        _schema(
            self.schema_version,
            QUALIFICATION_IMPACT_SHADOW_REQUEST_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "request_id": self.request_id,
            "graph": self.graph.to_dict(),
            "changes": [change.to_dict() for change in self.changes],
            "qualification_group_node_ids": list(self.qualification_group_node_ids),
            "evaluated_at_utc": self.evaluated_at_utc,
        }


@dataclass(frozen=True, slots=True)
class QualificationImpactRecommendationV1:
    qualification_group_node_id: str
    classification: QualificationImpactClassificationV1
    reason_codes: tuple[str, ...]
    affected_node_ids: tuple[str, ...]
    graph_digest: ContentDigest
    authority_effect: bool = False
    qualification_authority: bool = False
    automatic_invalidation: bool = False
    requires_semantic_review: bool = True
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_RECOMMENDATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(
            self.qualification_group_node_id,
            field="qualification_group_node_id",
        )
        require_instance(
            self.classification,
            QualificationImpactClassificationV1,
            field="classification",
        )
        require_sorted_unique_text(self.reason_codes, field="reason_codes")
        require_sorted_unique_text(self.affected_node_ids, field="affected_node_ids")
        require_instance(self.graph_digest, ContentDigest, field="graph_digest")
        if any(
            type(value) is not bool
            for value in (
                self.authority_effect,
                self.qualification_authority,
                self.automatic_invalidation,
                self.requires_semantic_review,
            )
        ):
            raise contract_invalid("authority flags must be exact booleans", field="authority_flags")
        if (
            self.authority_effect
            or self.qualification_authority
            or self.automatic_invalidation
            or not self.requires_semantic_review
        ):
            raise contract_invalid(
                "impact recommendations are permanently shadow-only",
                field="authority_flags",
            )
        _schema(
            self.schema_version,
            QUALIFICATION_IMPACT_RECOMMENDATION_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "qualification_group_node_id": self.qualification_group_node_id,
            "classification": self.classification.value,
            "reason_codes": list(self.reason_codes),
            "affected_node_ids": list(self.affected_node_ids),
            "graph_digest": self.graph_digest.to_dict(),
            "authority_effect": self.authority_effect,
            "qualification_authority": self.qualification_authority,
            "automatic_invalidation": self.automatic_invalidation,
            "requires_semantic_review": self.requires_semantic_review,
        }


@dataclass(frozen=True, slots=True)
class QualificationImpactShadowRunV1:
    request_id: str
    status: ImpactShadowStatusV1
    issue_codes: tuple[str, ...]
    recommendations: tuple[QualificationImpactRecommendationV1, ...]
    graph_digest: ContentDigest
    authority_effect: bool = False
    qualification_authority: bool = False
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_SHADOW_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.request_id, field="request_id")
        require_instance(self.status, ImpactShadowStatusV1, field="status")
        require_sorted_unique_text(self.issue_codes, field="issue_codes")
        if not isinstance(self.recommendations, tuple) or any(
            not isinstance(item, QualificationImpactRecommendationV1)
            for item in self.recommendations
        ):
            raise contract_invalid(
                "recommendations must be a canonical tuple",
                field="recommendations",
            )
        groups = [item.qualification_group_node_id for item in self.recommendations]
        if groups != sorted(groups) or len(groups) != len(set(groups)):
            raise contract_invalid(
                "recommendations must be sorted and unique by group",
                field="recommendations",
            )
        require_instance(self.graph_digest, ContentDigest, field="graph_digest")
        if self.authority_effect or self.qualification_authority:
            raise contract_invalid("shadow run cannot grant authority", field="authority_flags")
        has_uncertain = any(
            item.classification
            is QualificationImpactClassificationV1.IMPACT_UNCERTAIN
            for item in self.recommendations
        )
        if self.issue_codes:
            if self.status is not ImpactShadowStatusV1.IMPACT_UNCERTAIN or not all(
                item.classification
                is QualificationImpactClassificationV1.IMPACT_UNCERTAIN
                for item in self.recommendations
            ):
                raise contract_invalid(
                    "graph issues must make every recommendation uncertain",
                    field="status",
                )
        elif self.status is not ImpactShadowStatusV1.RECOMMENDATIONS_READY or has_uncertain:
            raise contract_invalid(
                "issue-free shadow runs must contain ready recommendations",
                field="status",
            )
        _schema(self.schema_version, QUALIFICATION_IMPACT_SHADOW_RUN_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "request_id": self.request_id,
            "status": self.status.value,
            "issue_codes": list(self.issue_codes),
            "recommendations": [item.to_dict() for item in self.recommendations],
            "graph_digest": self.graph_digest.to_dict(),
            "authority_effect": self.authority_effect,
            "qualification_authority": self.qualification_authority,
        }

    def run_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class QualificationImpactGoldenOutcomeV1:
    qualification_group_node_id: str
    classification: QualificationImpactClassificationV1
    reviewed_outcome_ref: str
    reviewed_outcome_digest: ContentDigest
    reviewer_id: str
    authority_effect: bool = False
    qualification_authority: bool = False
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_GOLDEN_OUTCOME_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.qualification_group_node_id, field="qualification_group_node_id")
        require_instance(
            self.classification,
            QualificationImpactClassificationV1,
            field="classification",
        )
        require_text(self.reviewed_outcome_ref, field="reviewed_outcome_ref")
        require_instance(
            self.reviewed_outcome_digest,
            ContentDigest,
            field="reviewed_outcome_digest",
        )
        require_text(self.reviewer_id, field="reviewer_id")
        if self.authority_effect or self.qualification_authority:
            raise contract_invalid(
                "golden comparison outcomes do not publish qualification authority",
                field="authority_flags",
            )
        _schema(
            self.schema_version,
            QUALIFICATION_IMPACT_GOLDEN_OUTCOME_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "qualification_group_node_id": self.qualification_group_node_id,
            "classification": self.classification.value,
            "reviewed_outcome_ref": self.reviewed_outcome_ref,
            "reviewed_outcome_digest": self.reviewed_outcome_digest.to_dict(),
            "reviewer_id": self.reviewer_id,
            "authority_effect": self.authority_effect,
            "qualification_authority": self.qualification_authority,
        }


@dataclass(frozen=True, slots=True)
class QualificationImpactGoldenCaseV1:
    case_id: str
    request: QualificationImpactShadowRequestV1
    reviewed_outcomes: tuple[QualificationImpactGoldenOutcomeV1, ...]
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_GOLDEN_CASE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.case_id, field="case_id")
        require_instance(
            self.request,
            QualificationImpactShadowRequestV1,
            field="request",
        )
        if (
            not isinstance(self.reviewed_outcomes, tuple)
            or not self.reviewed_outcomes
            or any(
                not isinstance(item, QualificationImpactGoldenOutcomeV1)
                for item in self.reviewed_outcomes
            )
        ):
            raise contract_invalid(
                "reviewed_outcomes must be a nonempty tuple",
                field="reviewed_outcomes",
            )
        groups = [item.qualification_group_node_id for item in self.reviewed_outcomes]
        if groups != sorted(groups) or len(groups) != len(set(groups)):
            raise contract_invalid(
                "reviewed outcomes must be sorted and unique",
                field="reviewed_outcomes",
            )
        if tuple(groups) != self.request.qualification_group_node_ids:
            raise contract_invalid(
                "reviewed outcomes must cover every requested group",
                field="reviewed_outcomes",
            )
        _schema(self.schema_version, QUALIFICATION_IMPACT_GOLDEN_CASE_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "case_id": self.case_id,
            "request": self.request.to_dict(),
            "reviewed_outcomes": [item.to_dict() for item in self.reviewed_outcomes],
        }


@dataclass(frozen=True, slots=True)
class QualificationImpactGoldenCaseResultV1:
    case_id: str
    shadow_run: QualificationImpactShadowRunV1
    exact_match: bool
    unsafe_inheritance_count: int
    uncertainty_expected: bool
    uncertainty_preserved: bool
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_GOLDEN_CASE_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.case_id, field="case_id")
        require_instance(self.shadow_run, QualificationImpactShadowRunV1, field="shadow_run")
        for field_name in ("exact_match", "uncertainty_expected", "uncertainty_preserved"):
            if type(getattr(self, field_name)) is not bool:
                raise contract_invalid("golden result flags must be exact booleans", field=field_name)
        if (
            not isinstance(self.unsafe_inheritance_count, int)
            or isinstance(self.unsafe_inheritance_count, bool)
            or self.unsafe_inheritance_count < 0
        ):
            raise contract_invalid(
                "unsafe_inheritance_count must be non-negative",
                field="unsafe_inheritance_count",
            )
        _schema(
            self.schema_version,
            QUALIFICATION_IMPACT_GOLDEN_CASE_RESULT_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "case_id": self.case_id,
            "shadow_run": self.shadow_run.to_dict(),
            "exact_match": self.exact_match,
            "unsafe_inheritance_count": self.unsafe_inheritance_count,
            "uncertainty_expected": self.uncertainty_expected,
            "uncertainty_preserved": self.uncertainty_preserved,
        }


@dataclass(frozen=True, slots=True)
class QualificationImpactGoldenRunV1:
    corpus_id: str
    case_results: tuple[QualificationImpactGoldenCaseResultV1, ...]
    unsafe_inheritance_count: int
    exact_match_count: int
    uncertainty_expected_count: int
    uncertainty_preserved_count: int
    status: str
    binding_issue_codes: tuple[str, ...] = ()
    authority_effect: bool = False
    qualification_authority: bool = False
    schema_version: SchemaVersion = QUALIFICATION_IMPACT_GOLDEN_RUN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        require_text(self.corpus_id, field="corpus_id")
        if not isinstance(self.case_results, tuple) or not self.case_results:
            raise contract_invalid("case_results must be nonempty", field="case_results")
        case_ids = [item.case_id for item in self.case_results]
        if case_ids != sorted(case_ids) or len(case_ids) != len(set(case_ids)):
            raise contract_invalid("case_results must be sorted and unique", field="case_results")
        for field_name in (
            "unsafe_inheritance_count",
            "exact_match_count",
            "uncertainty_expected_count",
            "uncertainty_preserved_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise contract_invalid("golden counters must be non-negative", field=field_name)
        if self.status not in {"PASS", "FAIL"}:
            raise contract_invalid("golden status must be PASS or FAIL", field="status")
        require_sorted_unique_text(
            self.binding_issue_codes,
            field="binding_issue_codes",
        )
        if self.authority_effect or self.qualification_authority:
            raise contract_invalid("golden run cannot grant authority", field="authority_flags")
        expected_counters = {
            "unsafe_inheritance_count": sum(
                item.unsafe_inheritance_count for item in self.case_results
            ),
            "exact_match_count": sum(item.exact_match for item in self.case_results),
            "uncertainty_expected_count": sum(
                item.uncertainty_expected for item in self.case_results
            ),
            "uncertainty_preserved_count": sum(
                item.uncertainty_expected and item.uncertainty_preserved
                for item in self.case_results
            ),
        }
        if any(
            getattr(self, field_name) != expected
            for field_name, expected in expected_counters.items()
        ):
            raise contract_invalid(
                "golden counters do not match case results",
                field="golden_counters",
            )
        expected_status = (
            "PASS"
            if self.unsafe_inheritance_count == 0
            and self.exact_match_count == len(self.case_results)
            and self.uncertainty_expected_count == self.uncertainty_preserved_count
            and not self.binding_issue_codes
            else "FAIL"
        )
        if self.status != expected_status:
            raise contract_invalid(
                "golden status does not match case results",
                field="status",
            )
        _schema(self.schema_version, QUALIFICATION_IMPACT_GOLDEN_RUN_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "corpus_id": self.corpus_id,
            "case_count": len(self.case_results),
            "case_results": [item.to_dict() for item in self.case_results],
            "unsafe_inheritance_count": self.unsafe_inheritance_count,
            "exact_match_count": self.exact_match_count,
            "uncertainty_expected_count": self.uncertainty_expected_count,
            "uncertainty_preserved_count": self.uncertainty_preserved_count,
            "status": self.status,
            "binding_issue_codes": list(self.binding_issue_codes),
            "authority_effect": self.authority_effect,
            "qualification_authority": self.qualification_authority,
        }

    def run_digest(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())
