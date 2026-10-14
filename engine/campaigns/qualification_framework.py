"""EIEL Qualification Framework V1 records, resolver, and planner.

QF-1 is an evidence-governance layer. It references existing route
qualification receipts and campaign evidence; it does not qualify routes by
itself and it does not mutate simulation mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from .canonical import canonical_sha256, pretty_json


QF_SCHEMA_VERSION: Final[str] = "eiel_qualification_framework_v1"

SCOPE_ROUTE_ENGINEERING: Final[str] = "ROUTE_ENGINEERING"
SCOPE_CONFIGURATION_OPERATING_ENVELOPE: Final[str] = (
    "CONFIGURATION_OPERATING_ENVELOPE"
)
SCOPE_SCIENTIFIC_SUBSTRATE: Final[str] = "SCIENTIFIC_SUBSTRATE"
SCOPE_CAMPAIGN_SPECIFIC_SCIENCE: Final[str] = "CAMPAIGN_SPECIFIC_SCIENCE"
SCOPE_RESOURCE_CAPACITY_ENVELOPE: Final[str] = "RESOURCE_CAPACITY_ENVELOPE"

QUALIFIED: Final[str] = "QUALIFIED"
QUALIFIED_WITH_EXCLUSIONS: Final[str] = "QUALIFIED_WITH_EXCLUSIONS"
DELTA_REQUIRED: Final[str] = "DELTA_REQUIRED"
DELTA_REQUIRED_BELOW_ANCHOR: Final[str] = "DELTA_REQUIRED_BELOW_ANCHOR"
FULL_QUALIFICATION_REQUIRED: Final[str] = "FULL_QUALIFICATION_REQUIRED"
UNSUPPORTED: Final[str] = "UNSUPPORTED"
INVALID: Final[str] = "INVALID"
REVALIDATION_REQUIRED: Final[str] = "REVALIDATION_REQUIRED"
SUPERSEDED: Final[str] = "SUPERSEDED"

FULL_QUALIFICATION: Final[str] = "FULL_QUALIFICATION"
DELTA_QUALIFICATION: Final[str] = "DELTA_QUALIFICATION"
SANITY_QUALIFICATION: Final[str] = "SANITY_QUALIFICATION"

DELTA_CLASSIFIED: Final[str] = "DELTA_CLASSIFIED"
DELTA_PLANNED: Final[str] = "DELTA_PLANNED"
DELTA_READY: Final[str] = "DELTA_READY"
DELTA_BLOCKED: Final[str] = "DELTA_BLOCKED"
DELTA_EXECUTED_PENDING_DECISION: Final[str] = "DELTA_EXECUTED_PENDING_DECISION"
DELTA_ACCEPTED: Final[str] = "DELTA_ACCEPTED"
DELTA_REJECTED: Final[str] = "DELTA_REJECTED"
DELTA_INVALID: Final[str] = "DELTA_INVALID"
DELTA_SUPERSEDED: Final[str] = "DELTA_SUPERSEDED"

AUTHORITY_CURRENT: Final[str] = "CURRENT_AUTHORITY"
AUTHORITY_PLANNED_NON_AUTHORITATIVE: Final[str] = "PLANNED_NON_AUTHORITATIVE"
AUTHORITY_NO_CURRENT_AUTHORITY: Final[str] = "NO_CURRENT_AUTHORITY"
AUTHORITY_REVALIDATION_REQUIRED: Final[str] = "REVALIDATION_REQUIRED"
AUTHORITY_HISTORICAL_NON_CURRENT: Final[str] = "HISTORICAL_NON_CURRENT"
AUTHORITY_SUPERSEDED: Final[str] = "SUPERSEDED"

RECIPE_TICK_ONLY_DELTA: Final[str] = "TICK_ONLY_DELTA"
RECIPE_SHAPE_POPULATION_DELTA: Final[str] = "SHAPE_POPULATION_DELTA"
RECIPE_BATCH_CAPACITY_DELTA: Final[str] = "BATCH_CAPACITY_DELTA"
RECIPE_COMBINED_INTERACTION_DELTA: Final[str] = "COMBINED_INTERACTION_DELTA"

HISTORICAL_VERIFIED_AT_TIME: Final[str] = "HISTORICAL_VERIFIED_AT_TIME"
HISTORICAL_EVIDENCE_UNAVAILABLE: Final[str] = "HISTORICAL_EVIDENCE_UNAVAILABLE"
HISTORICAL_UNVERIFIABLE: Final[str] = "HISTORICAL_UNVERIFIABLE"

NUMERIC_AXES: Final[frozenset[str]] = frozenset(
    {
        "grid_width",
        "grid_height",
        "ticks_per_generation",
        "generations",
        "population",
        "initial_population",
        "resident_capacity",
        "batch_size",
    }
)
CATEGORICAL_AXES: Final[frozenset[str]] = frozenset(
    {
        "backend",
        "provider_id",
        "runtime_id",
        "controller_family",
        "world_family",
        "world_config_id",
        "execution_route_sha256",
        "checkpoint_mode",
        "observation_schema",
        "genome_schema",
        "policy_architecture",
        "device",
    }
)
VALID_SCOPES: Final[frozenset[str]] = frozenset(
    {
        SCOPE_ROUTE_ENGINEERING,
        SCOPE_CONFIGURATION_OPERATING_ENVELOPE,
        SCOPE_RESOURCE_CAPACITY_ENVELOPE,
        SCOPE_SCIENTIFIC_SUBSTRATE,
        SCOPE_CAMPAIGN_SPECIFIC_SCIENCE,
    }
)

VALID_DELTA_LIFECYCLE_STAGES: Final[frozenset[str]] = frozenset(
    {
        DELTA_CLASSIFIED,
        DELTA_PLANNED,
        DELTA_READY,
        DELTA_BLOCKED,
        DELTA_EXECUTED_PENDING_DECISION,
        DELTA_ACCEPTED,
        DELTA_REJECTED,
        DELTA_INVALID,
        DELTA_SUPERSEDED,
    }
)

VALID_AUTHORITY_STATES: Final[frozenset[str]] = frozenset(
    {
        AUTHORITY_CURRENT,
        AUTHORITY_PLANNED_NON_AUTHORITATIVE,
        AUTHORITY_NO_CURRENT_AUTHORITY,
        AUTHORITY_REVALIDATION_REQUIRED,
        AUTHORITY_HISTORICAL_NON_CURRENT,
        AUTHORITY_SUPERSEDED,
    }
)

VALID_RECIPE_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        RECIPE_TICK_ONLY_DELTA,
        RECIPE_SHAPE_POPULATION_DELTA,
        RECIPE_BATCH_CAPACITY_DELTA,
        RECIPE_COMBINED_INTERACTION_DELTA,
    }
)

VALID_HISTORICAL_STATES: Final[frozenset[str]] = frozenset(
    {
        HISTORICAL_VERIFIED_AT_TIME,
        HISTORICAL_EVIDENCE_UNAVAILABLE,
        HISTORICAL_UNVERIFIABLE,
    }
)

VALID_DECISIONS: Final[frozenset[str]] = frozenset(
    {
        QUALIFIED,
        QUALIFIED_WITH_EXCLUSIONS,
        DELTA_REQUIRED,
        DELTA_REQUIRED_BELOW_ANCHOR,
        FULL_QUALIFICATION_REQUIRED,
        UNSUPPORTED,
        INVALID,
        REVALIDATION_REQUIRED,
        SUPERSEDED,
    }
)


class QualificationFrameworkError(ValueError):
    """A QF-1 record or query is malformed."""


def _as_tuple(value: Sequence[Any] | None) -> tuple[Any, ...]:
    return tuple(value or ())


def _canonical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize square-grid shorthand into explicit width and height."""

    out = dict(config)
    if "grid_size" in out:
        if "grid_width" not in out:
            out["grid_width"] = out["grid_size"]
        if "grid_height" not in out:
            out["grid_height"] = out["grid_size"]
    return out


def _validate_point(config: Mapping[str, Any]) -> list[dict[str, str]]:
    reasons: list[dict[str, str]] = []
    normalized = _canonical_config(config)
    for axis in ("grid_width", "grid_height", "ticks_per_generation"):
        value = normalized.get(axis)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            reasons.append(
                {
                    "code": "invalid_numeric_axis",
                    "axis": axis,
                    "message": f"{axis} must be a positive integer",
                }
            )
    return reasons


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_file(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationFrameworkError(f"cannot read JSON evidence: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise QualificationFrameworkError(f"JSON evidence root must be an object: {path}")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else ()


def _nested(payload: Mapping[str, Any], *path: str) -> Any:
    value: Any = payload
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _component(
    route_receipt: Mapping[str, Any],
    component_id: str,
) -> Mapping[str, Any]:
    for row in _sequence(_nested(route_receipt, "route_contract", "components")):
        if isinstance(row, Mapping) and row.get("component_id") == component_id:
            return row
    return {}


def _reason(
    code: str,
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {"code": code, "message": message}
    payload.update(extra)
    return payload


def _require(
    reasons: list[dict[str, Any]],
    condition: bool,
    code: str,
    message: str,
    **extra: Any,
) -> None:
    if not condition:
        reasons.append(_reason(code, message, **extra))


def _require_equal(
    reasons: list[dict[str, Any]],
    *,
    actual: Any,
    expected: Any,
    code: str,
    message: str,
) -> None:
    if actual != expected:
        reasons.append(_reason(code, message, expected=expected, actual=actual))


@dataclass(frozen=True, slots=True)
class AxisRangeV1:
    name: str
    minimum: int | float
    maximum: int | float

    def __post_init__(self) -> None:
        if self.name not in NUMERIC_AXES:
            raise QualificationFrameworkError(f"unsupported numeric axis: {self.name}")
        if self.minimum > self.maximum:
            raise QualificationFrameworkError(f"axis {self.name} has min > max")

    def contains(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and self.minimum <= value <= self.maximum

    def below(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value < self.minimum

    def above(self, value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and value > self.maximum

    def payload(self) -> dict[str, Any]:
        return {"name": self.name, "minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AxisRangeV1:
        return cls(
            name=str(raw["name"]),
            minimum=raw["minimum"],
            maximum=raw["maximum"],
        )


@dataclass(frozen=True, slots=True)
class ExclusionRegionV1:
    exclusion_id: str
    reason: str
    axes: tuple[AxisRangeV1, ...] = ()
    categorical_values: Mapping[str, Any] = field(default_factory=dict)

    def contains(self, config: Mapping[str, Any]) -> bool:
        normalized = _canonical_config(config)
        numeric_match = all(
            axis.contains(normalized.get(axis.name)) for axis in self.axes
        )
        categorical_match = all(
            normalized.get(name) == value
            for name, value in sorted(self.categorical_values.items())
        )
        return numeric_match and categorical_match

    def payload(self) -> dict[str, Any]:
        return {
            "exclusion_id": self.exclusion_id,
            "reason": self.reason,
            "axes": [axis.payload() for axis in sorted(self.axes, key=lambda row: row.name)],
            "categorical_values": dict(sorted(self.categorical_values.items())),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ExclusionRegionV1:
        return cls(
            exclusion_id=str(raw["exclusion_id"]),
            reason=str(raw["reason"]),
            axes=tuple(AxisRangeV1.from_mapping(row) for row in raw.get("axes", ())),
            categorical_values=dict(raw.get("categorical_values") or {}),
        )


@dataclass(frozen=True, slots=True)
class EvidenceBindingV1:
    evidence_id: str
    kind: str
    uri: str
    sha256: str

    def payload(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "uri": self.uri,
            "sha256": self.sha256,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> EvidenceBindingV1:
        return cls(
            evidence_id=str(raw["evidence_id"]),
            kind=str(raw["kind"]),
            uri=str(raw["uri"]),
            sha256=str(raw["sha256"]),
        )


@dataclass(frozen=True, slots=True)
class ResourceCapacityEnvelopeV1:
    """Resource-capacity authority is separate from route/config authority."""

    resource_id: str
    backend: str
    device: str
    provider_id: str
    runtime_id: str
    source_commit: str
    execution_route_sha256: str
    batch_size: AxisRangeV1
    footprint_dimensions: Mapping[str, Any]
    resource_authority: Mapping[str, Any]
    safety_policy: Mapping[str, Any]
    fallback_policy: Mapping[str, Any]
    evidence: tuple[EvidenceBindingV1, ...]
    decision: str = QUALIFIED
    exclusions: tuple[ExclusionRegionV1, ...] = ()
    superseded_by: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.batch_size.name != "batch_size":
            raise QualificationFrameworkError("resource capacity range must use batch_size")
        if self.decision not in VALID_DECISIONS:
            raise QualificationFrameworkError(f"invalid resource decision: {self.decision}")

    def payload(self) -> dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "scope": SCOPE_RESOURCE_CAPACITY_ENVELOPE,
            "decision": self.decision,
            "backend": self.backend,
            "device": self.device,
            "provider_id": self.provider_id,
            "runtime_id": self.runtime_id,
            "source_commit": self.source_commit,
            "execution_route_sha256": self.execution_route_sha256,
            "batch_size": self.batch_size.payload(),
            "footprint_dimensions": dict(sorted(self.footprint_dimensions.items())),
            "resource_authority": dict(sorted(self.resource_authority.items())),
            "safety_policy": dict(sorted(self.safety_policy.items())),
            "fallback_policy": dict(sorted(self.fallback_policy.items())),
            "exclusions": [item.payload() for item in sorted(self.exclusions, key=lambda row: row.exclusion_id)],
            "evidence": [item.payload() for item in sorted(self.evidence, key=lambda row: row.evidence_id)],
            "superseded_by": self.superseded_by,
            "limitations": list(self.limitations),
        }

    @property
    def resource_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def contains(self, config: Mapping[str, Any]) -> bool:
        normalized = _canonical_config(config)
        return (
            normalized.get("backend") == self.backend
            and normalized.get("device") == self.device
            and normalized.get("provider_id") == self.provider_id
            and normalized.get("runtime_id") == self.runtime_id
            and normalized.get("execution_route_sha256") == self.execution_route_sha256
            and self.batch_size.contains(normalized.get("batch_size"))
        )

    def categorical_mismatches(self, config: Mapping[str, Any]) -> list[dict[str, Any]]:
        normalized = _canonical_config(config)
        mismatches = []
        for name, expected in (
            ("backend", self.backend),
            ("device", self.device),
            ("provider_id", self.provider_id),
            ("runtime_id", self.runtime_id),
            ("execution_route_sha256", self.execution_route_sha256),
        ):
            actual = normalized.get(name)
            if actual != expected:
                mismatches.append({"axis": name, "expected": expected, "actual": actual})
        return mismatches

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> ResourceCapacityEnvelopeV1:
        return cls(
            resource_id=str(raw["resource_id"]),
            backend=str(raw["backend"]),
            device=str(raw["device"]),
            provider_id=str(raw["provider_id"]),
            runtime_id=str(raw["runtime_id"]),
            source_commit=str(raw["source_commit"]),
            execution_route_sha256=str(raw["execution_route_sha256"]),
            batch_size=AxisRangeV1.from_mapping(_mapping(raw.get("batch_size"))),
            footprint_dimensions=dict(raw.get("footprint_dimensions") or {}),
            resource_authority=dict(raw.get("resource_authority") or {}),
            safety_policy=dict(raw.get("safety_policy") or {}),
            fallback_policy=dict(raw.get("fallback_policy") or {}),
            evidence=tuple(EvidenceBindingV1.from_mapping(row) for row in raw.get("evidence", ())),
            decision=str(raw.get("decision", QUALIFIED)),
            exclusions=tuple(ExclusionRegionV1.from_mapping(row) for row in raw.get("exclusions", ())),
            superseded_by=raw.get("superseded_by"),
            limitations=tuple(str(item) for item in raw.get("limitations", ())),
        )


@dataclass(frozen=True, slots=True)
class HistoricalAuthorityRecordV1:
    historical_id: str
    state: str
    original_decision: str
    source_commit: str
    execution_route_sha256: str
    provenance: Mapping[str, Any]
    missing_evidence: tuple[str, ...] = ()
    resolver_eligible: bool = False
    current_authority: bool = False
    revalidation_required: bool = True

    def __post_init__(self) -> None:
        if self.state not in VALID_HISTORICAL_STATES:
            raise QualificationFrameworkError(f"invalid historical state: {self.state}")
        if self.resolver_eligible:
            raise QualificationFrameworkError("historical records are not resolver eligible")
        if self.current_authority:
            raise QualificationFrameworkError("historical records are not current authority")

    def payload(self) -> dict[str, Any]:
        return {
            "historical_id": self.historical_id,
            "state": self.state,
            "original_decision": self.original_decision,
            "source_commit": self.source_commit,
            "execution_route_sha256": self.execution_route_sha256,
            "provenance": dict(sorted(self.provenance.items())),
            "missing_evidence": sorted(self.missing_evidence),
            "resolver_eligible": self.resolver_eligible,
            "current_authority": self.current_authority,
            "revalidation_required": self.revalidation_required,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> HistoricalAuthorityRecordV1:
        return cls(
            historical_id=str(raw["historical_id"]),
            state=str(raw["state"]),
            original_decision=str(raw.get("original_decision") or ""),
            source_commit=str(raw.get("source_commit") or ""),
            execution_route_sha256=str(raw.get("execution_route_sha256") or ""),
            provenance=dict(raw.get("provenance") or {}),
            missing_evidence=tuple(str(item) for item in raw.get("missing_evidence", ())),
            resolver_eligible=bool(raw.get("resolver_eligible", False)),
            current_authority=bool(raw.get("current_authority", False)),
            revalidation_required=bool(raw.get("revalidation_required", True)),
        )


@dataclass(frozen=True, slots=True)
class QualificationAnchorV1:
    anchor_id: str
    scope: str
    config: Mapping[str, Any]
    source_commit: str
    execution_route_sha256: str
    provider_id: str
    runtime_id: str
    evidence: tuple[EvidenceBindingV1, ...]
    decision: str = QUALIFIED
    superseded_by: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in VALID_SCOPES:
            raise QualificationFrameworkError(f"invalid qualification scope: {self.scope}")
        if self.decision not in VALID_DECISIONS:
            raise QualificationFrameworkError(f"invalid anchor decision: {self.decision}")
        invalid = _validate_point(self.config)
        if invalid:
            raise QualificationFrameworkError(invalid[0]["message"])

    def payload(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "scope": self.scope,
            "decision": self.decision,
            "config": dict(sorted(_canonical_config(self.config).items())),
            "source_commit": self.source_commit,
            "execution_route_sha256": self.execution_route_sha256,
            "provider_id": self.provider_id,
            "runtime_id": self.runtime_id,
            "evidence": [item.payload() for item in sorted(self.evidence, key=lambda row: row.evidence_id)],
            "superseded_by": self.superseded_by,
            "limitations": list(self.limitations),
        }

    @property
    def anchor_sha256(self) -> str:
        return canonical_sha256(self.payload())

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> QualificationAnchorV1:
        return cls(
            anchor_id=str(raw["anchor_id"]),
            scope=str(raw["scope"]),
            config=dict(raw["config"]),
            source_commit=str(raw["source_commit"]),
            execution_route_sha256=str(raw["execution_route_sha256"]),
            provider_id=str(raw["provider_id"]),
            runtime_id=str(raw["runtime_id"]),
            evidence=tuple(EvidenceBindingV1.from_mapping(row) for row in raw.get("evidence", ())),
            decision=str(raw.get("decision", QUALIFIED)),
            superseded_by=raw.get("superseded_by"),
            limitations=tuple(str(item) for item in raw.get("limitations", ())),
        )


@dataclass(frozen=True, slots=True)
class OperatingEnvelopeV1:
    envelope_id: str
    scope: str
    axes: tuple[AxisRangeV1, ...]
    categorical_invariants: Mapping[str, Any]
    anchor_ids: tuple[str, ...]
    required_corners: tuple[Mapping[str, Any], ...]
    accepted_points: tuple[Mapping[str, Any], ...]
    evidence: tuple[EvidenceBindingV1, ...]
    source_commit: str | None = None
    execution_route_sha256: str | None = None
    decision: str = QUALIFIED
    interpolation_policy: str = "explicit_corners_edges_and_interactions_v1"
    lifecycle_stage: str | None = None
    authority_state: str | None = None
    required_resource_capacity_id: str | None = None
    accepted_region: str = "cartesian_envelope"
    exclusions: tuple[ExclusionRegionV1, ...] = ()
    superseded_by: str | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in VALID_SCOPES:
            raise QualificationFrameworkError(f"invalid qualification scope: {self.scope}")
        if self.decision not in VALID_DECISIONS:
            raise QualificationFrameworkError(f"invalid envelope decision: {self.decision}")
        names = [axis.name for axis in self.axes]
        if len(names) != len(set(names)):
            raise QualificationFrameworkError("envelope axes must be unique")
        for name in self.categorical_invariants:
            if name not in CATEGORICAL_AXES:
                raise QualificationFrameworkError(f"unsupported categorical invariant: {name}")
        if self.lifecycle_stage and self.lifecycle_stage not in VALID_DELTA_LIFECYCLE_STAGES:
            raise QualificationFrameworkError(f"invalid lifecycle stage: {self.lifecycle_stage}")
        if self.authority_state and self.authority_state not in VALID_AUTHORITY_STATES:
            raise QualificationFrameworkError(f"invalid authority state: {self.authority_state}")

    def payload(self) -> dict[str, Any]:
        return {
            "envelope_id": self.envelope_id,
            "scope": self.scope,
            "decision": self.decision,
            "axes": [axis.payload() for axis in sorted(self.axes, key=lambda row: row.name)],
            "categorical_invariants": dict(sorted(self.categorical_invariants.items())),
            "anchor_ids": sorted(self.anchor_ids),
            "required_corners": [dict(sorted(_canonical_config(row).items())) for row in self.required_corners],
            "accepted_points": [dict(sorted(_canonical_config(row).items())) for row in self.accepted_points],
            "source_commit": self.source_commit,
            "execution_route_sha256": self.execution_route_sha256,
            "interpolation_policy": self.interpolation_policy,
            "lifecycle_stage": self.lifecycle_stage,
            "authority_state": self.authority_state,
            "required_resource_capacity_id": self.required_resource_capacity_id,
            "accepted_region": self.accepted_region,
            "exclusions": [item.payload() for item in sorted(self.exclusions, key=lambda row: row.exclusion_id)],
            "evidence": [item.payload() for item in sorted(self.evidence, key=lambda row: row.evidence_id)],
            "superseded_by": self.superseded_by,
            "limitations": list(self.limitations),
        }

    @property
    def envelope_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def contains(self, config: Mapping[str, Any]) -> bool:
        normalized = _canonical_config(config)
        return all(axis.contains(normalized.get(axis.name)) for axis in self.axes)

    def below_any_anchor(self, config: Mapping[str, Any]) -> bool:
        normalized = _canonical_config(config)
        return any(axis.below(normalized.get(axis.name)) for axis in self.axes)

    def above_any_bound(self, config: Mapping[str, Any]) -> bool:
        normalized = _canonical_config(config)
        return any(axis.above(normalized.get(axis.name)) for axis in self.axes)

    def categorical_mismatches(self, config: Mapping[str, Any]) -> list[dict[str, Any]]:
        normalized = _canonical_config(config)
        mismatches = []
        for name, expected in sorted(self.categorical_invariants.items()):
            actual = normalized.get(name)
            if actual != expected:
                mismatches.append({"axis": name, "expected": expected, "actual": actual})
        return mismatches

    def missing_corners(self) -> list[dict[str, Any]]:
        accepted = {_point_key(point) for point in self.accepted_points}
        return [
            dict(sorted(_canonical_config(corner).items()))
            for corner in self.required_corners
            if _point_key(corner) not in accepted
        ]

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> OperatingEnvelopeV1:
        return cls(
            envelope_id=str(raw["envelope_id"]),
            scope=str(raw["scope"]),
            axes=tuple(AxisRangeV1.from_mapping(row) for row in raw.get("axes", ())),
            categorical_invariants=dict(raw.get("categorical_invariants") or {}),
            anchor_ids=tuple(str(item) for item in raw.get("anchor_ids", ())),
            required_corners=tuple(dict(row) for row in raw.get("required_corners", ())),
            accepted_points=tuple(dict(row) for row in raw.get("accepted_points", ())),
            evidence=tuple(EvidenceBindingV1.from_mapping(row) for row in raw.get("evidence", ())),
            source_commit=raw.get("source_commit"),
            execution_route_sha256=raw.get("execution_route_sha256"),
            decision=str(raw.get("decision", QUALIFIED)),
            interpolation_policy=str(raw.get("interpolation_policy", "explicit_corners_edges_and_interactions_v1")),
            lifecycle_stage=raw.get("lifecycle_stage"),
            authority_state=raw.get("authority_state"),
            required_resource_capacity_id=raw.get("required_resource_capacity_id"),
            accepted_region=str(raw.get("accepted_region", "cartesian_envelope")),
            exclusions=tuple(ExclusionRegionV1.from_mapping(row) for row in raw.get("exclusions", ())),
            superseded_by=raw.get("superseded_by"),
            limitations=tuple(str(item) for item in raw.get("limitations", ())),
        )


@dataclass(frozen=True, slots=True)
class QualificationCatalogV1:
    anchors: tuple[QualificationAnchorV1, ...] = ()
    envelopes: tuple[OperatingEnvelopeV1, ...] = ()
    resource_capacity_envelopes: tuple[ResourceCapacityEnvelopeV1, ...] = ()
    historical_authority: tuple[HistoricalAuthorityRecordV1, ...] = ()
    schema_version: str = QF_SCHEMA_VERSION

    def payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "anchors": [item.payload() for item in sorted(self.anchors, key=lambda row: row.anchor_id)],
            "envelopes": [item.payload() for item in sorted(self.envelopes, key=lambda row: row.envelope_id)],
            "resource_capacity_envelopes": [
                item.payload()
                for item in sorted(self.resource_capacity_envelopes, key=lambda row: row.resource_id)
            ],
            "historical_authority": [
                item.payload()
                for item in sorted(self.historical_authority, key=lambda row: row.historical_id)
            ],
        }
        payload["catalog_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> QualificationCatalogV1:
        if raw.get("schema_version") != QF_SCHEMA_VERSION:
            raise QualificationFrameworkError("unsupported QF catalog schema")
        return cls(
            anchors=tuple(QualificationAnchorV1.from_mapping(row) for row in raw.get("anchors", ())),
            envelopes=tuple(OperatingEnvelopeV1.from_mapping(row) for row in raw.get("envelopes", ())),
            resource_capacity_envelopes=tuple(
                ResourceCapacityEnvelopeV1.from_mapping(row)
                for row in raw.get("resource_capacity_envelopes", ())
            ),
            historical_authority=tuple(
                HistoricalAuthorityRecordV1.from_mapping(row)
                for row in raw.get("historical_authority", ())
            ),
        )


def _point_key(
    config: Mapping[str, Any],
    *,
    exclude: set[str] | None = None,
) -> tuple[tuple[str, Any], ...]:
    normalized = _canonical_config(config)
    excluded = exclude or set()
    return tuple(
        sorted(
            (name, value)
            for name, value in normalized.items()
            if (name in NUMERIC_AXES or name in CATEGORICAL_AXES)
            and name not in excluded
        )
    )


def _numeric_point(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _canonical_config(config)
    return {
        name: normalized[name]
        for name in sorted(NUMERIC_AXES)
        if name in normalized
    }


def load_catalog(path: str | Path) -> QualificationCatalogV1:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualificationFrameworkError(f"cannot read QF catalog: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise QualificationFrameworkError("QF catalog root must be an object")
    raw = dict(raw)
    raw.pop("catalog_sha256", None)
    return QualificationCatalogV1.from_mapping(raw)


def write_catalog(path: str | Path, catalog: QualificationCatalogV1) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(pretty_json(catalog.payload()), encoding="utf-8")


def validate_authority_bundle(
    bundle: Mapping[str, Any],
    *,
    catalog: QualificationCatalogV1 | None = None,
) -> dict[str, Any]:
    """Validate a real route/resource/continuation authority bundle.

    The bundle validator is intentionally read-only. It checks current evidence
    files by path and SHA-256, then verifies that the imported authority is
    exact for one scoped request. It does not execute campaigns or widen QF-1
    catalogs.
    """

    if not isinstance(bundle, Mapping):
        raise QualificationFrameworkError("QF authority bundle root must be an object")

    reasons: list[dict[str, Any]] = []
    if bundle.get("schema_version") != "eiel_qf1_authority_bundle_v1":
        reasons.append(
            _reason(
                "unsupported_authority_bundle_schema",
                "authority bundle schema_version is not supported",
                expected="eiel_qf1_authority_bundle_v1",
                actual=bundle.get("schema_version"),
            )
        )

    authority_id = str(bundle.get("authority_id") or "")
    scope = str(bundle.get("scope") or "")
    if scope not in VALID_SCOPES:
        reasons.append(
            _reason(
                "invalid_authority_scope",
                "authority bundle scope must be a valid QF-1 scope",
                actual=scope,
            )
        )

    target = _canonical_config(_mapping(bundle.get("target")))
    reasons.extend(_validate_point(target))
    expected = _mapping(bundle.get("expected"))
    if expected.get("source_commit"):
        _require_equal(
            reasons,
            actual=target.get("source_commit"),
            expected=expected.get("source_commit"),
            code="source_commit_mismatch",
            message="authority bundle target source commit does not match expected source",
        )
    if (
        expected.get("claim_full_uninterrupted_execution") is True
        and isinstance(target.get("generations"), int)
        and isinstance(expected.get("bounded_generations_proven"), int)
        and target["generations"] > expected["bounded_generations_proven"]
    ):
        reasons.append(
            _reason(
                "unsupported_full_uninterrupted_execution_claim",
                "bounded route authority cannot be imported as a full uninterrupted execution transcript",
                target_generations=target.get("generations"),
                bounded_generations_proven=expected.get("bounded_generations_proven"),
            )
        )
    files = _mapping(bundle.get("evidence_files"))
    loaded: dict[str, Mapping[str, Any]] = {}
    evidence_bindings: list[dict[str, Any]] = []

    for evidence_id, raw_spec in sorted(files.items()):
        spec = _mapping(raw_spec)
        path = Path(str(spec.get("path") or ""))
        claimed_sha256 = str(spec.get("sha256") or "")
        if not path.is_file():
            reasons.append(
                _reason(
                    "evidence_file_missing",
                    "authority evidence file is missing",
                    evidence_id=str(evidence_id),
                    path=str(path),
                )
            )
            continue
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != claimed_sha256:
            reasons.append(
                _reason(
                    "evidence_sha256_mismatch",
                    "authority evidence file SHA-256 does not match the bundle",
                    evidence_id=str(evidence_id),
                    path=str(path),
                    expected=claimed_sha256,
                    actual=actual_sha256,
                )
            )
        try:
            loaded[str(evidence_id)] = _read_json_file(path)
        except QualificationFrameworkError as exc:
            reasons.append(
                _reason(
                    "evidence_json_invalid",
                    "authority evidence file is not valid JSON object evidence",
                    evidence_id=str(evidence_id),
                    path=str(path),
                    error=str(exc),
                )
            )
            continue
        evidence_bindings.append(
            {
                "evidence_id": str(evidence_id),
                "kind": str(spec.get("kind") or "json_evidence"),
                "uri": str(path),
                "sha256": actual_sha256,
            }
        )

    if catalog is not None:
        qf_result = resolve_qualification(target, catalog)
        if qf_result.get("decision") != QUALIFIED:
            reasons.append(
                _reason(
                    "qf_catalog_resolution_not_qualified",
                    "target does not resolve to a scoped QF-1 qualified record",
                    qf_decision=qf_result.get("decision"),
                    qf_reasons=qf_result.get("reasons"),
                )
            )
        if expected.get("qf_record_id"):
            _require_equal(
                reasons,
                actual=qf_result.get("record_id"),
                expected=expected.get("qf_record_id"),
                code="qf_catalog_record_mismatch",
                message="target resolved to a different QF-1 record",
            )
        if expected.get("scope"):
            _require_equal(
                reasons,
                actual=qf_result.get("scope"),
                expected=expected.get("scope"),
                code="qf_catalog_scope_mismatch",
                message="target resolved to a different QF-1 scope",
            )

    preflight = loaded.get("preflight")
    route_receipt = loaded.get("route_receipt")
    comparison_summary = loaded.get("comparison_summary")

    if preflight is None:
        reasons.append(
            _reason(
                "missing_preflight_evidence",
                "authority bundle requires a current post-route preflight receipt",
            )
        )
    else:
        _validate_preflight_authority(reasons, target, expected, preflight, files)

    if route_receipt is None:
        reasons.append(
            _reason(
                "missing_route_receipt_evidence",
                "authority bundle requires an execution-route qualification receipt",
            )
        )
    else:
        _validate_route_receipt_authority(reasons, target, expected, route_receipt)

    if expected.get("cpu_continuation_required") is True:
        if comparison_summary is None:
            reasons.append(
                _reason(
                    "missing_cpu_continuation_evidence",
                    "CPU production authority requires continuation comparison evidence",
                )
            )
        else:
            _validate_cpu_continuation(reasons, comparison_summary)

    limitations = [str(item) for item in _sequence(bundle.get("limitations"))]
    if isinstance(target.get("generations"), int) and target.get("generations", 0) > 128:
        limitations.append(
            "G200 production target reuses route/config authority; bounded proof does not claim a full uninterrupted G200 execution transcript"
        )

    decision = QUALIFIED if not reasons else FULL_QUALIFICATION_REQUIRED
    payload = {
        "schema_version": "eiel_qf1_authority_decision_v1",
        "ok": decision == QUALIFIED,
        "authority_id": authority_id,
        "decision": decision,
        "scope": scope or None,
        "target": dict(sorted(target.items())),
        "record_id": expected.get("qf_record_id"),
        "evidence": evidence_bindings,
        "limitations": sorted(set(limitations)),
        "exclusions": [str(item) for item in _sequence(bundle.get("exclusions"))],
        "reasons": reasons
        or [
            {
                "code": "qf1_authority_bundle_accepted",
                "message": "bundle evidence is current, route-bound, resource-bound, and scoped",
            }
        ],
    }
    payload["authority_decision_sha256"] = canonical_sha256(payload)
    return payload


def _validate_preflight_authority(
    reasons: list[dict[str, Any]],
    target: Mapping[str, Any],
    expected: Mapping[str, Any],
    preflight: Mapping[str, Any],
    files: Mapping[str, Any],
) -> None:
    _require(reasons, preflight.get("ok") is True, "preflight_not_ok", "preflight ok flag is not true")
    _require(
        reasons,
        preflight.get("ready_for_execution") is True,
        "preflight_not_ready_for_execution",
        "post-route preflight is not ready for execution",
    )
    _require_equal(
        reasons,
        actual=_nested(preflight, "execution_provider", "provider_id"),
        expected=expected.get("provider_id"),
        code="provider_mismatch",
        message="preflight provider does not match expected authority",
    )
    if expected.get("plan_sha256"):
        _require_equal(
            reasons,
            actual=preflight.get("plan_sha256"),
            expected=expected.get("plan_sha256"),
            code="plan_sha256_mismatch",
            message="preflight plan SHA-256 does not match expected authority",
        )
    _require_equal(
        reasons,
        actual=_nested(preflight, "profile", "profile_id"),
        expected=expected.get("runtime_id"),
        code="runtime_profile_mismatch",
        message="preflight runtime profile does not match expected authority",
    )
    _require_equal(
        reasons,
        actual=_nested(preflight, "execution_route", "execution_route_sha256"),
        expected=expected.get("execution_route_sha256"),
        code="execution_route_mismatch",
        message="preflight route SHA-256 does not match expected authority",
    )
    _require(
        reasons,
        _nested(preflight, "execution_route", "route_hash_exact") is True,
        "route_hash_not_exact",
        "preflight did not bind the exact execution route hash",
    )
    _require(
        reasons,
        _nested(preflight, "execution_route", "provider_binding_exact") is True,
        "provider_binding_not_exact",
        "preflight did not bind the exact provider",
    )
    _require(
        reasons,
        _nested(preflight, "execution_route", "profile_binding_exact") is True,
        "profile_binding_not_exact",
        "preflight did not bind the exact runtime profile",
    )
    _require(
        reasons,
        _nested(preflight, "execution_route", "artifact_conformance_exact") is True,
        "artifact_conformance_not_exact",
        "preflight did not prove exact artifact conformance",
    )
    _require_equal(
        reasons,
        actual=_nested(preflight, "execution_route", "qualification_receipt_sha256"),
        expected=_mapping(files.get("route_receipt")).get("sha256"),
        code="route_receipt_sha256_mismatch",
        message="preflight route receipt SHA-256 does not match bundled route receipt",
    )

    covered = list(_sequence(_nested(preflight, "execution_route", "covered_obligations")))
    required = list(_sequence(_nested(preflight, "execution_route", "required_obligations")))
    missing = list(_sequence(_nested(preflight, "execution_route", "missing_obligations")))
    _require(reasons, not missing, "missing_route_obligations", "preflight reports missing route obligations", missing_obligations=missing)
    _require(
        reasons,
        sorted(covered) == sorted(required) and bool(required),
        "obligation_coverage_incomplete",
        "preflight covered obligations do not exactly match required obligations",
        covered_count=len(covered),
        required_count=len(required),
    )
    if expected.get("route_obligation_count") is not None:
        _require_equal(
            reasons,
            actual=len(covered),
            expected=expected.get("route_obligation_count"),
            code="obligation_count_mismatch",
            message="preflight covered obligation count does not match expected authority",
        )

    _require(
        reasons,
        _nested(preflight, "qualification_reuse", "reuses_bound_qualification") is True,
        "qualification_reuse_not_bound",
        "preflight does not reuse the bound qualification identity",
    )
    _require(
        reasons,
        _nested(preflight, "qualification_reuse", "qualification_key_hash_exact") is True,
        "qualification_key_not_exact",
        "preflight qualification key hash is not exact",
    )

    batch = target.get("batch_size")
    if batch is not None:
        _require_equal(
            reasons,
            actual=_nested(preflight, "profile", "requested_physical_batch_cells"),
            expected=batch,
            code="batch_size_mismatch",
            message="preflight requested physical batch does not match target",
        )
        admitted = _sequence(_nested(preflight, "qualification_reuse", "batch_admission", "admitted_range"))
        _require(
            reasons,
            len(admitted) == 2 and isinstance(batch, int) and admitted[0] <= batch <= admitted[1],
            "batch_not_in_admitted_range",
            "target batch is outside the admitted route range",
            admitted_range=list(admitted),
            batch_size=batch,
        )
        _require(
            reasons,
            _nested(preflight, "qualification_reuse", "batch_admission", "within_admitted_range") is True,
            "batch_admission_not_confirmed",
            "preflight did not confirm requested batch admission",
        )

    identity = _mapping(_nested(preflight, "qualification_reuse", "identity_binding"))
    _require_equal(
        reasons,
        actual=_nested(identity, "mechanics", "backend"),
        expected=expected.get("backend"),
        code="backend_mismatch",
        message="preflight backend identity does not match expected authority",
    )
    _require_equal(
        reasons,
        actual=_nested(identity, "mechanics", "fixed_invariants", "device_owner"),
        expected=expected.get("device_owner"),
        code="device_owner_mismatch",
        message="preflight device-owner invariant does not match expected authority",
    )
    _require_equal(
        reasons,
        actual=_nested(identity, "mechanics", "fixed_invariants", "parameter.patch.grid_size"),
        expected=target.get("grid_width"),
        code="grid_size_mismatch",
        message="preflight grid-size invariant does not match target",
    )
    _require_equal(
        reasons,
        actual=_nested(identity, "mechanics", "fixed_invariants", "parameter.patch.initial_population"),
        expected=target.get("initial_population"),
        code="initial_population_mismatch",
        message="preflight initial-population invariant does not match target",
    )
    _require_equal(
        reasons,
        actual=_nested(identity, "capacity", "fixed_invariants", "parameter.patch.capacity"),
        expected=target.get("resident_capacity"),
        code="resident_capacity_mismatch",
        message="preflight resident-capacity invariant does not match target",
    )

    artifact_rows = [row for row in _sequence(preflight.get("artifacts")) if isinstance(row, Mapping)]
    _require(
        reasons,
        bool(artifact_rows) and all(row.get("regular_file") is True and row.get("sha256_exact") is True for row in artifact_rows),
        "artifact_inventory_not_exact",
        "preflight artifact inventory is missing or not exact",
    )
    if expected.get("artifact_sha256"):
        _require(
            reasons,
            any(row.get("actual_sha256") == expected.get("artifact_sha256") for row in artifact_rows),
            "artifact_sha256_not_present",
            "expected production artifact SHA-256 is absent from preflight artifact inventory",
            expected=expected.get("artifact_sha256"),
        )

    if expected.get("backend") == "cuda_resident":
        _validate_gpu_resource(reasons, expected, preflight)
    elif expected.get("backend") == "cpu_native":
        _validate_cpu_resource(reasons, preflight)


def _validate_gpu_resource(
    reasons: list[dict[str, Any]],
    expected: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> None:
    cuda = _mapping(_nested(preflight, "resources", "cuda"))
    _require(reasons, cuda.get("admission_required") is True, "gpu_resource_admission_not_required", "GPU authority requires CUDA admission")
    _require(
        reasons,
        cuda.get("free_covers_admission_bound_and_headroom") is True,
        "gpu_headroom_not_confirmed",
        "GPU resource authority does not confirm admission bound and headroom",
    )
    _require(
        reasons,
        cuda.get("provider_estimate_within_budget") is True,
        "gpu_estimate_outside_budget",
        "GPU resource estimate is outside the declared budget",
    )
    _require(
        reasons,
        cuda.get("runtime_environment_ok") is True,
        "gpu_runtime_environment_not_ok",
        "GPU runtime environment requirements are not satisfied",
    )
    _require_equal(
        reasons,
        actual=_nested(preflight, "execution_provider", "resource_estimator_id"),
        expected=expected.get("resource_estimator_id"),
        code="resource_estimator_mismatch",
        message="preflight resource estimator does not match expected authority",
    )
    _require(
        reasons,
        isinstance(_nested(preflight, "resources", "vram_bytes_upper_bound"), int),
        "gpu_vram_bound_missing",
        "GPU authority requires an integer VRAM upper bound",
    )
    _require_equal(
        reasons,
        actual=cuda.get("live_admission_bound_source"),
        expected="provider_exact_upper_bound",
        code="gpu_admission_bound_source_mismatch",
        message="GPU authority requires provider exact upper-bound admission",
    )


def _validate_cpu_resource(
    reasons: list[dict[str, Any]],
    preflight: Mapping[str, Any],
) -> None:
    cuda = _mapping(_nested(preflight, "resources", "cuda"))
    _require(reasons, cuda.get("admission_required") is False, "cpu_cuda_admission_required", "CPU authority must not require CUDA admission")
    _require(
        reasons,
        "cpu_native" in str(cuda.get("skip_reason") or ""),
        "cpu_cuda_skip_reason_missing",
        "CPU authority must explain CUDA admission skip with the CPU-native backend",
    )


def _validate_route_receipt_authority(
    reasons: list[dict[str, Any]],
    target: Mapping[str, Any],
    expected: Mapping[str, Any],
    route_receipt: Mapping[str, Any],
) -> None:
    _require_equal(
        reasons,
        actual=route_receipt.get("schema_version"),
        expected="eiel_execution_route_qualification_v1",
        code="route_receipt_schema_mismatch",
        message="route receipt schema is not the execution-route qualification schema",
    )
    _require_equal(
        reasons,
        actual=route_receipt.get("authority_state"),
        expected="route_qualified",
        code="route_not_qualified",
        message="route receipt is not in route_qualified authority state",
    )
    _require_equal(
        reasons,
        actual=route_receipt.get("execution_route_sha256"),
        expected=expected.get("execution_route_sha256"),
        code="route_receipt_sha_mismatch",
        message="route receipt SHA-256 identity does not match expected route",
    )
    _require_equal(
        reasons,
        actual=route_receipt.get("provider_id"),
        expected=expected.get("provider_id"),
        code="route_provider_mismatch",
        message="route receipt provider does not match expected authority",
    )
    _require_equal(
        reasons,
        actual=route_receipt.get("profile_id"),
        expected=expected.get("runtime_id"),
        code="route_runtime_profile_mismatch",
        message="route receipt profile does not match expected authority",
    )
    if expected.get("artifact_sha256"):
        _require_equal(
            reasons,
            actual=route_receipt.get("artifact_sha256"),
            expected=expected.get("artifact_sha256"),
            code="route_artifact_mismatch",
            message="route receipt artifact SHA-256 does not match expected authority",
        )

    covered = list(_sequence(route_receipt.get("covered_obligations")))
    required = list(_sequence(_nested(route_receipt, "route_contract", "required_obligations")))
    _require(
        reasons,
        bool(required) and sorted(covered) == sorted(required),
        "route_receipt_obligation_coverage_incomplete",
        "route receipt covered obligations do not exactly match its contract",
        covered_count=len(covered),
        required_count=len(required),
    )
    if expected.get("route_obligation_count") is not None:
        _require_equal(
            reasons,
            actual=len(covered),
            expected=expected.get("route_obligation_count"),
            code="route_receipt_obligation_count_mismatch",
            message="route receipt covered obligation count does not match expected authority",
        )

    comparisons = [row for row in _sequence(route_receipt.get("exact_comparisons")) if isinstance(row, Mapping)]
    _require(
        reasons,
        len(comparisons) >= int(expected.get("minimum_exact_comparisons") or 2)
        and all(row.get("match") is True for row in comparisons),
        "route_exact_comparisons_incomplete",
        "route receipt exact comparisons are missing or not all matching",
    )
    _require(
        reasons,
        _nested(route_receipt, "checkpoint_results", "exact") is True,
        "route_checkpoint_results_not_exact",
        "route receipt checkpoint results are not exact",
    )
    _require(
        reasons,
        _nested(route_receipt, "reporting_results", "exact") is True,
        "route_reporting_results_not_exact",
        "route receipt reporting results are not exact",
    )

    backend = _component(route_receipt, "runtime_execution_backend")
    _require_equal(
        reasons,
        actual=backend.get("backend"),
        expected=expected.get("backend"),
        code="route_backend_mismatch",
        message="route receipt backend does not match expected authority",
    )
    _require_equal(
        reasons,
        actual=backend.get("device"),
        expected=expected.get("device"),
        code="route_device_mismatch",
        message="route receipt device does not match expected authority",
    )
    _require(
        reasons,
        backend.get("fallback_allowed") is False,
        "route_fallback_allowed",
        "route receipt allows fallback",
    )

    native = _component(route_receipt, "native_mechanics_binding")
    _require_equal(
        reasons,
        actual=native.get("request_ticks_per_generation"),
        expected=target.get("ticks_per_generation"),
        code="target_ticks_not_bound",
        message="route receipt request ticks do not match target",
    )
    _require_equal(
        reasons,
        actual=native.get("ticks_per_generation"),
        expected=target.get("ticks_per_generation"),
        code="route_ticks_not_bound",
        message="route receipt ticks do not match target",
    )
    admitted_ticks = set(_sequence(native.get("admitted_ticks_per_generation")))
    _require(
        reasons,
        target.get("ticks_per_generation") in admitted_ticks,
        "target_ticks_not_admitted",
        "target ticks are not in the route's admitted tick set",
        admitted_ticks=sorted(admitted_ticks),
    )


def _validate_cpu_continuation(
    reasons: list[dict[str, Any]],
    comparison_summary: Mapping[str, Any],
) -> None:
    _require_equal(
        reasons,
        actual=comparison_summary.get("schema_version"),
        expected="w32_cpu_t80_g14_v3_comparison_summary_v1",
        code="cpu_continuation_schema_mismatch",
        message="CPU continuation comparison summary has the wrong schema",
    )
    rows = [row for row in _sequence(comparison_summary.get("comparisons")) if isinstance(row, Mapping)]
    by_kind = {str(row.get("kind")): row for row in rows}
    for kind in ("fresh_repeat", "reporting_parity", "checkpoint_continuation"):
        row = by_kind.get(kind)
        _require(
            reasons,
            isinstance(row, Mapping)
            and row.get("status") == "completed"
            and row.get("comparison_match") is True,
            "cpu_continuation_comparison_missing_or_failed",
            "CPU continuation comparison is missing or failed",
            comparison_kind=kind,
        )


def generate_delta_registry(
    plan: Mapping[str, Any],
    *,
    campaign_id: str = "qf1-synthetic-delta",
) -> dict[str, Any]:
    """Create a deterministic synthetic campaign registry from a QF-1 plan."""

    validation = validate_delta_plan(plan)
    if not validation["ok"]:
        raise QualificationFrameworkError("; ".join(str(item) for item in validation["errors"]))
    claimed = str(plan.get("plan_sha256"))
    registry = {
        "schema_version": "eiel_qualification_delta_registry_v1",
        "campaign_id": campaign_id,
        "plan_sha256": claimed,
        "scope": plan.get("scope"),
        "envelope_id": plan.get("envelope_id"),
        "plan_class": plan.get("plan_class"),
        "lifecycle_stage": plan.get("lifecycle_stage"),
        "authority_state": plan.get("authority_state"),
        "selected_recipes": list(plan.get("selected_recipes") or ()),
        "required_obligations": list(plan.get("required_obligations") or ()),
        "required_obligations_sha256": plan.get("required_obligations_sha256"),
        "required_resource_capacity_id": plan.get("required_resource_capacity_id"),
        "candidate_points": list(plan.get("candidate_points") or ()),
        "decision_rule": plan.get("decision_rule"),
        "expected_profile": "QUAL-DELTA-1",
        "expected_artifacts": [
            "campaign_evidence_manifest.json",
            "qualification_decision.json",
            "route_receipt.json",
            "accepted_catalog.json",
        ],
        "dry_run": {
            "ok": True,
            "planned_point_count": len(list(plan.get("candidate_points") or ())),
            "real_campaign_authority": plan.get("real_campaign_authority"),
        },
    }
    registry["registry_sha256"] = canonical_sha256(registry)
    return registry


def publish_accepted_delta(
    catalog: QualificationCatalogV1,
    *,
    envelope_id: str,
    plan: Mapping[str, Any],
    decision: Mapping[str, Any],
    evidence: Sequence[EvidenceBindingV1],
) -> QualificationCatalogV1:
    """Bind an A decision into a catalog without widening the planned region."""

    decision_validation = validate_delta_decision_acceptance(plan, decision)
    if not decision_validation["ok"]:
        raise QualificationFrameworkError("; ".join(str(item) for item in decision_validation["errors"]))
    if plan.get("scope") != SCOPE_CONFIGURATION_OPERATING_ENVELOPE:
        raise QualificationFrameworkError("QF-1 synthetic lifecycle accepts only operating-envelope scope")

    updated: list[OperatingEnvelopeV1] = []
    found = False
    for envelope in catalog.envelopes:
        if envelope.envelope_id != envelope_id:
            updated.append(envelope)
            continue
        found = True
        planned_points = [
            _numeric_point(row.get("config") if isinstance(row.get("config"), Mapping) else {})
            for row in plan.get("candidate_points") or ()
        ]
        accepted_points = _dedupe_numeric_points(
            [*envelope.accepted_points, *envelope.required_corners, *planned_points]
        )
        route = (
            envelope.execution_route_sha256
            or _first_config_value(plan, "execution_route_sha256")
        )
        source = envelope.source_commit or str(decision.get("source_commit") or "")
        updated.append(
            OperatingEnvelopeV1(
                envelope_id=envelope.envelope_id,
                scope=envelope.scope,
                axes=envelope.axes,
                categorical_invariants=envelope.categorical_invariants,
                anchor_ids=envelope.anchor_ids,
                required_corners=envelope.required_corners,
                accepted_points=tuple(accepted_points),
                evidence=tuple([*envelope.evidence, *evidence]),
                source_commit=source or None,
                execution_route_sha256=route,
                decision=QUALIFIED,
                interpolation_policy=envelope.interpolation_policy,
                lifecycle_stage=DELTA_ACCEPTED,
                authority_state=AUTHORITY_CURRENT,
                required_resource_capacity_id=envelope.required_resource_capacity_id,
                accepted_region=str(decision.get("accepted_region") or envelope.accepted_region),
                exclusions=envelope.exclusions,
                superseded_by=envelope.superseded_by,
                limitations=tuple(
                    [
                        *envelope.limitations,
                        "synthetic QF-1 lifecycle acceptance only; no campaign science",
                    ]
                ),
            )
        )
    if not found:
        raise QualificationFrameworkError(f"unknown envelope_id: {envelope_id}")
    return QualificationCatalogV1(
        anchors=catalog.anchors,
        envelopes=tuple(updated),
        resource_capacity_envelopes=catalog.resource_capacity_envelopes,
        historical_authority=catalog.historical_authority,
    )


def _first_config_value(plan: Mapping[str, Any], name: str) -> Any:
    for point in plan.get("candidate_points") or ():
        config = point.get("config") if isinstance(point, Mapping) else None
        if isinstance(config, Mapping) and name in config:
            return config[name]
    config = plan.get("requested_config")
    if isinstance(config, Mapping):
        return config.get(name)
    return None


def _dedupe_numeric_points(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    out: list[dict[str, Any]] = []
    for point in points:
        numeric = _numeric_point(point)
        key = tuple(sorted(numeric.items()))
        if not numeric or key in seen:
            continue
        seen.add(key)
        out.append(numeric)
    return out


def _decision_lifecycle(decision: str) -> str:
    if decision == QUALIFIED:
        return DELTA_ACCEPTED
    if decision in {DELTA_REQUIRED, DELTA_REQUIRED_BELOW_ANCHOR}:
        return DELTA_CLASSIFIED
    if decision == REVALIDATION_REQUIRED:
        return DELTA_BLOCKED
    if decision == SUPERSEDED:
        return DELTA_SUPERSEDED
    if decision == INVALID:
        return DELTA_INVALID
    return DELTA_BLOCKED


def _decision_authority(decision: str) -> str:
    if decision == QUALIFIED:
        return AUTHORITY_CURRENT
    if decision in {DELTA_REQUIRED, DELTA_REQUIRED_BELOW_ANCHOR}:
        return AUTHORITY_PLANNED_NON_AUTHORITATIVE
    if decision == REVALIDATION_REQUIRED:
        return AUTHORITY_REVALIDATION_REQUIRED
    if decision == SUPERSEDED:
        return AUTHORITY_SUPERSEDED
    return AUTHORITY_NO_CURRENT_AUTHORITY


def _next_action(decision: str) -> str:
    if decision == QUALIFIED:
        return "none; current scoped authority is available"
    if decision in {DELTA_REQUIRED, DELTA_REQUIRED_BELOW_ANCHOR}:
        return "execute and decide the validated delta plan before publishing authority"
    if decision == REVALIDATION_REQUIRED:
        return "perform current-source revalidation before reuse"
    if decision == SUPERSEDED:
        return "resolve against the superseding record"
    return "full qualification or operator decision required"


def delta_status_row(result: Mapping[str, Any]) -> dict[str, Any]:
    """Expose resolver outcome separately from lifecycle and authority state."""

    decision = str(result.get("decision") or "")
    reasons = [row for row in _sequence(result.get("reasons")) if isinstance(row, Mapping)]
    return {
        "resolver_outcome": decision,
        "lifecycle_stage": result.get("lifecycle_stage") or _decision_lifecycle(decision),
        "authority_state": result.get("authority_state") or _decision_authority(decision),
        "qualification_scope": result.get("scope"),
        "record_id": result.get("record_id"),
        "blocking_reason": reasons[0].get("code") if reasons else None,
        "next_required_action": result.get("next_required_action") or _next_action(decision),
    }


RECIPE_OBLIGATIONS: Final[Mapping[str, tuple[str, ...]]] = {
    RECIPE_TICK_ONLY_DELTA: (
        "current_source_commit_binding",
        "exact_provider_runtime_backend_device_binding",
        "route_qualification_receipt_binding",
        "route_plan_config_artifact_inventory_binding",
        "preflight_ready",
        "exact_target_axis_binding",
        "target_horizon_completion",
        "deterministic_target_horizon_evidence",
        "checkpoint_schema_output_integrity",
        "nonfinite_and_fatal_anomaly_checks",
        "evidence_manifest",
        "deterministic_decision_output",
        "bounded_claim_scope_and_exclusions",
    ),
    RECIPE_SHAPE_POPULATION_DELTA: (
        "current_source_commit_binding",
        "exact_provider_runtime_backend_device_binding",
        "route_qualification_receipt_binding",
        "route_plan_config_artifact_inventory_binding",
        "preflight_ready",
        "grid_width_height_independent_binding",
        "initial_population_binding",
        "target_shape_completion",
        "representative_repeat_parity",
        "checkpoint_schema_output_integrity",
        "nonfinite_and_fatal_anomaly_checks",
        "evidence_manifest",
        "deterministic_decision_output",
        "bounded_claim_scope_and_exclusions",
    ),
    RECIPE_BATCH_CAPACITY_DELTA: (
        "current_source_commit_binding",
        "exact_provider_runtime_backend_device_binding",
        "route_qualification_receipt_binding",
        "resource_capacity_authority_reference",
        "exact_batch_execution_identity",
        "no_silent_fallback",
        "resource_telemetry_allocated_reserved_peak",
        "declared_safety_headroom_policy",
        "completion_or_canary_integrity",
        "representative_repeat_parity",
        "evidence_manifest",
        "deterministic_decision_output",
        "bounded_claim_scope_and_exclusions",
    ),
    RECIPE_COMBINED_INTERACTION_DELTA: (
        "interaction_axes_declared",
        "interaction_corner_or_explicit_narrow_region",
        "component_recipe_obligations_preserved",
        "no_posthoc_widening",
        "accepted_region_shape_declared",
        "unresolved_combinations_excluded",
    ),
}


def recipe_obligations(recipe_families: Sequence[str]) -> tuple[str, ...]:
    obligations: set[str] = set()
    for recipe in recipe_families:
        if recipe not in VALID_RECIPE_FAMILIES:
            raise QualificationFrameworkError(f"unknown recipe family: {recipe}")
        obligations.update(RECIPE_OBLIGATIONS[recipe])
    return tuple(sorted(obligations))


def select_delta_recipes(changed_axes: Sequence[str]) -> tuple[str, ...]:
    axes = set(changed_axes)
    recipes: set[str] = set()
    if axes == {"ticks_per_generation"}:
        recipes.add(RECIPE_TICK_ONLY_DELTA)
    else:
        if "ticks_per_generation" in axes:
            recipes.add(RECIPE_TICK_ONLY_DELTA)
        if axes & {"grid_width", "grid_height", "initial_population"}:
            recipes.add(RECIPE_SHAPE_POPULATION_DELTA)
        if "batch_size" in axes:
            recipes.add(RECIPE_BATCH_CAPACITY_DELTA)
    if len(recipes) > 1:
        recipes.add(RECIPE_COMBINED_INTERACTION_DELTA)
    if not recipes:
        recipes.add(RECIPE_SHAPE_POPULATION_DELTA)
    return tuple(sorted(recipes))


def validate_delta_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    claimed = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    actual = canonical_sha256(unsigned)
    errors: list[str] = []
    if claimed != actual:
        errors.append("plan_sha256 does not match canonical plan payload")
    lifecycle = plan.get("lifecycle_stage")
    authority = plan.get("authority_state")
    recipes = [str(item) for item in _sequence(plan.get("selected_recipes"))]
    obligations = [str(item) for item in _sequence(plan.get("required_obligations"))]
    obligation_sha = plan.get("required_obligations_sha256")
    if lifecycle != DELTA_PLANNED:
        errors.append("delta plan lifecycle_stage must be DELTA_PLANNED")
    if authority != AUTHORITY_PLANNED_NON_AUTHORITATIVE:
        errors.append("delta plan authority_state must be PLANNED_NON_AUTHORITATIVE")
    if not recipes:
        errors.append("delta plan must list selected_recipes")
    unknown = sorted(set(recipes) - set(VALID_RECIPE_FAMILIES))
    if unknown:
        errors.append(f"delta plan has unknown selected_recipes: {unknown}")
    expected_obligations = list(recipe_obligations(recipes)) if recipes and not unknown else []
    missing = sorted(set(expected_obligations) - set(obligations))
    if missing:
        errors.append(f"delta plan missing required obligations: {missing}")
    if obligation_sha != canonical_sha256({"required_obligations": obligations}):
        errors.append("required_obligations_sha256 does not match required_obligations")
    if plan.get("real_campaign_authority") != "not_authorized_by_qf1_planner":
        errors.append("delta planner must not authorize real campaign execution")
    return {
        "schema_version": "eiel_qualification_plan_validation_v1",
        "ok": not errors,
        "claimed_plan_sha256": claimed,
        "actual_plan_sha256": actual,
        "lifecycle_stage": lifecycle,
        "authority_state": authority,
        "selected_recipes": recipes,
        "required_obligation_count": len(obligations),
        "errors": errors,
    }


def validate_delta_decision_acceptance(
    plan: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    plan_status = validate_delta_plan(plan)
    if not plan_status["ok"]:
        errors.extend(str(item) for item in plan_status["errors"])
    if decision.get("decision") != "A":
        errors.append("only A decisions may publish delta authority")
    if decision.get("accepted_for_envelope_expansion") is not True:
        errors.append("decision did not accept envelope expansion")
    if decision.get("no_post_hoc_widening") is not True:
        errors.append("decision did not preserve no-post-hoc widening")
    if decision.get("plan_sha256") != plan.get("plan_sha256"):
        errors.append("decision plan_sha256 does not match plan")
    if decision.get("required_obligations_sha256") != plan.get("required_obligations_sha256"):
        errors.append("decision required_obligations_sha256 does not match plan")
    required = {str(item) for item in _sequence(plan.get("required_obligations"))}
    satisfied = {str(item) for item in _sequence(decision.get("satisfied_recipe_obligations"))}
    missing = sorted(required - satisfied)
    if missing:
        errors.append(f"decision missing satisfied recipe obligations: {missing}")
    return {
        "schema_version": "eiel_delta_decision_acceptance_validation_v1",
        "ok": not errors,
        "errors": errors,
    }


def continuation_compatibility_class(config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _canonical_config(config)
    dimensions = {
        "source_commit": normalized.get("source_commit"),
        "provider_id": normalized.get("provider_id"),
        "runtime_id": normalized.get("runtime_id"),
        "backend": normalized.get("backend", "cpu_native"),
        "device": normalized.get("device", "cpu"),
        "execution_route_sha256": normalized.get("execution_route_sha256"),
        "checkpoint_schema": normalized.get("checkpoint_schema"),
        "serialization_schema": normalized.get("serialization_schema"),
        "grid_width": normalized.get("grid_width"),
        "grid_height": normalized.get("grid_height"),
        "initial_population": normalized.get("initial_population"),
        "resident_capacity": normalized.get("resident_capacity"),
        "ticks_per_generation": normalized.get("ticks_per_generation"),
    }
    return {
        "schema_version": "eiel_cpu_continuation_compatibility_class_v1",
        "dimensions": dict(sorted((k, v) for k, v in dimensions.items() if v is not None)),
        "compatibility_class_sha256": canonical_sha256(
            {"dimensions": dict(sorted((k, v) for k, v in dimensions.items() if v is not None))}
        ),
    }


def candidate_matrix_status(matrix: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "eiel_qf1_candidate_matrix_status_v1",
        "status": "NON_AUTHORITATIVE_CANDIDATE",
        "authority_state": AUTHORITY_PLANNED_NON_AUTHORITATIVE,
        "resolver_eligible": False,
        "current_authority": False,
        "track_count": len(_mapping(matrix.get("tracks"))),
        "next_required_action": "execute and accept real delta evidence before publishing authority",
    }


def resolve_qualification(
    request: Mapping[str, Any],
    catalog: QualificationCatalogV1,
) -> dict[str, Any]:
    """Resolve one requested configuration against known QF-1 evidence."""

    config = _canonical_config(request)
    invalid = _validate_point(config)
    if invalid:
        return _result(INVALID, config=config, reasons=invalid)

    anchor_result = _resolve_exact_anchor(config, catalog.anchors)
    if anchor_result is not None:
        return anchor_result

    envelope_results = [
        _resolve_envelope(config, envelope, catalog.resource_capacity_envelopes)
        for envelope in catalog.envelopes
    ]
    accepted = [row for row in envelope_results if row["decision"] == QUALIFIED]
    if accepted:
        return sorted(accepted, key=lambda row: str(row["record_id"]))[0]

    exclusion = [
        row for row in envelope_results if row["decision"] == QUALIFIED_WITH_EXCLUSIONS
    ]
    if exclusion:
        return sorted(exclusion, key=lambda row: str(row["record_id"]))[0]

    revalidation = [
        row for row in envelope_results if row["decision"] == REVALIDATION_REQUIRED
    ]
    if revalidation:
        return sorted(revalidation, key=lambda row: str(row["record_id"]))[0]

    delta = [row for row in envelope_results if row["decision"] == DELTA_REQUIRED]
    if delta:
        return sorted(delta, key=lambda row: str(row["record_id"]))[0]

    below = [
        row
        for row in envelope_results
        if row["decision"] == DELTA_REQUIRED_BELOW_ANCHOR
    ]
    if below:
        return sorted(below, key=lambda row: str(row["record_id"]))[0]

    full = [
        row
        for row in envelope_results
        if row["decision"] in (FULL_QUALIFICATION_REQUIRED, UNSUPPORTED, SUPERSEDED)
    ]
    if full:
        return sorted(full, key=lambda row: str(row["record_id"]))[0]

    return _result(
        FULL_QUALIFICATION_REQUIRED,
        config=config,
        reasons=[
            {
                "code": "no_applicable_qualification_record",
                "message": "no anchor or envelope applies to this request",
            }
        ],
    )


def _resolve_exact_anchor(
    config: Mapping[str, Any],
    anchors: Sequence[QualificationAnchorV1],
) -> dict[str, Any] | None:
    for anchor in sorted(anchors, key=lambda row: row.anchor_id):
        if _point_key(anchor.config, exclude={"execution_route_sha256"}) != _point_key(
            config,
            exclude={"execution_route_sha256"},
        ):
            continue
        if anchor.superseded_by or anchor.decision == SUPERSEDED:
            return _result(
                SUPERSEDED,
                record_id=anchor.anchor_id,
                scope=anchor.scope,
                config=config,
                reasons=[
                    {
                        "code": "qualification_anchor_superseded",
                        "message": "exact anchor is superseded",
                    }
                ],
            )
        if config.get("source_commit") and config.get("source_commit") != anchor.source_commit:
            return _result(
                REVALIDATION_REQUIRED,
                record_id=anchor.anchor_id,
                scope=anchor.scope,
                config=config,
                reasons=[
                    {
                        "code": "source_commit_mismatch",
                        "expected": anchor.source_commit,
                        "actual": config.get("source_commit"),
                        "message": "exact anchor is bound to another source commit",
                    }
                ],
            )
        if config.get("execution_route_sha256") != anchor.execution_route_sha256:
            return _result(
                FULL_QUALIFICATION_REQUIRED,
                record_id=anchor.anchor_id,
                scope=anchor.scope,
                config=config,
                reasons=[
                    {
                        "code": "execution_route_mismatch",
                        "expected": anchor.execution_route_sha256,
                        "actual": config.get("execution_route_sha256"),
                        "message": "exact anchor requires its bound execution route receipt",
                    }
                ],
            )
        return _result(
            anchor.decision,
            record_id=anchor.anchor_id,
            scope=anchor.scope,
            config=config,
            evidence=[item.payload() for item in anchor.evidence],
            reasons=[
                {
                    "code": "exact_anchor_match",
                    "message": "request exactly matches a qualification anchor",
                }
            ],
        )
    return None


def _resolve_envelope(
    config: Mapping[str, Any],
    envelope: OperatingEnvelopeV1,
    resources: Sequence[ResourceCapacityEnvelopeV1] = (),
) -> dict[str, Any]:
    if envelope.superseded_by or envelope.decision == SUPERSEDED:
        return _result(
            SUPERSEDED,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            reasons=[
                {
                    "code": "qualification_envelope_superseded",
                    "message": "envelope is superseded",
                }
            ],
        )

    mismatches = envelope.categorical_mismatches(config)
    if mismatches:
        return _result(
            FULL_QUALIFICATION_REQUIRED,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            reasons=[
                {
                    "code": "categorical_boundary_crossed",
                    "mismatches": mismatches,
                    "message": "categorical invariants cannot be interpolated",
                }
            ],
        )

    if envelope.execution_route_sha256 and not config.get("execution_route_sha256"):
        return _result(
            FULL_QUALIFICATION_REQUIRED,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            reasons=[
                {
                    "code": "missing_route_receipt",
                    "expected": envelope.execution_route_sha256,
                    "message": "envelope requires an exact execution route receipt",
                }
            ],
        )
    if (
        envelope.execution_route_sha256
        and config.get("execution_route_sha256") != envelope.execution_route_sha256
    ):
        return _result(
            FULL_QUALIFICATION_REQUIRED,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            reasons=[
                {
                    "code": "execution_route_mismatch",
                    "expected": envelope.execution_route_sha256,
                    "actual": config.get("execution_route_sha256"),
                    "message": "envelope is bound to another execution route receipt",
                }
            ],
        )
    if (
        envelope.source_commit
        and config.get("source_commit")
        and config.get("source_commit") != envelope.source_commit
    ):
        return _result(
            REVALIDATION_REQUIRED,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            reasons=[
                {
                    "code": "source_commit_mismatch",
                    "expected": envelope.source_commit,
                    "actual": config.get("source_commit"),
                    "message": "envelope applicability to this source requires explicit impact review",
                }
            ],
        )

    if envelope.required_resource_capacity_id:
        resource_result = resolve_resource_capacity(
            config,
            resources,
            resource_id=envelope.required_resource_capacity_id,
        )
        if resource_result["decision"] != QUALIFIED:
            return _result(
                DELTA_REQUIRED,
                record_id=envelope.envelope_id,
                scope=envelope.scope,
                config=config,
                required_campaign_class="resource_capacity_delta",
                reasons=[
                    {
                        "code": "required_resource_capacity_not_qualified",
                        "resource_id": envelope.required_resource_capacity_id,
                        "resource_decision": resource_result["decision"],
                        "resource_reasons": resource_result["reasons"],
                        "message": "configuration authority depends on a separate resource-capacity authority",
                    }
                ],
            )

    if envelope.below_any_anchor(config):
        return _result(
            DELTA_REQUIRED_BELOW_ANCHOR,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            required_campaign_class="below_anchor_delta",
            reasons=[
                {
                    "code": "below_numeric_lower_bound",
                    "message": "request is below at least one numeric envelope bound and needs explicit downscale evidence",
                }
            ],
        )
    if envelope.above_any_bound(config):
        return _result(
            FULL_QUALIFICATION_REQUIRED,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            reasons=[
                {
                    "code": "above_numeric_upper_bound",
                    "message": "request is above at least one numeric envelope bound",
                }
            ],
        )
    if not envelope.contains(config):
        return _result(
            FULL_QUALIFICATION_REQUIRED,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            reasons=[
                {
                    "code": "outside_numeric_envelope",
                    "message": "request is outside the numeric envelope",
                }
            ],
        )
    for exclusion in envelope.exclusions:
        if exclusion.contains(config):
            return _result(
                QUALIFIED_WITH_EXCLUSIONS,
                record_id=envelope.envelope_id,
                scope=envelope.scope,
                config=config,
                reasons=[
                    {
                        "code": "exclusion_region_hit",
                        "exclusion_id": exclusion.exclusion_id,
                        "message": exclusion.reason,
                    }
                ],
            )
    missing = envelope.missing_corners()
    if missing:
        return _result(
            DELTA_REQUIRED,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            required_campaign_class="missing_corner_or_interaction_delta",
            reasons=[
                {
                    "code": "missing_required_corners",
                    "missing_corners": missing,
                    "message": "accepted evidence does not cover required envelope corners",
                }
            ],
        )
    if _point_key(config) in {_point_key(row) for row in envelope.accepted_points}:
        return _result(
            envelope.decision,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            evidence=[item.payload() for item in envelope.evidence],
            reasons=[
                {
                    "code": "accepted_envelope_point",
                    "message": "request is an accepted point inside the envelope",
                }
            ],
        )
    if envelope.decision == QUALIFIED:
        return _result(
            QUALIFIED,
            record_id=envelope.envelope_id,
            scope=envelope.scope,
            config=config,
            evidence=[item.payload() for item in envelope.evidence],
            reasons=[
                {
                    "code": "accepted_envelope_region",
                    "message": "request lies inside the predeclared accepted envelope",
                }
            ],
        )
    return _result(
        DELTA_REQUIRED,
        record_id=envelope.envelope_id,
        scope=envelope.scope,
        config=config,
        required_campaign_class="interior_delta",
        reasons=[
            {
                "code": "candidate_interior_point",
                "message": "request is inside numeric bounds but lacks accepted interior evidence",
            }
        ],
    )


def _result(
    decision: str,
    *,
    config: Mapping[str, Any],
    reasons: list[Mapping[str, Any]],
    record_id: str | None = None,
    scope: str | None = None,
    evidence: list[Mapping[str, Any]] | None = None,
    required_campaign_class: str | None = None,
) -> dict[str, Any]:
    lifecycle_stage = _decision_lifecycle(decision)
    authority_state = _decision_authority(decision)
    return {
        "schema_version": "eiel_qualification_resolution_v1",
        "decision": decision,
        "lifecycle_stage": lifecycle_stage,
        "authority_state": authority_state,
        "record_id": record_id,
        "scope": scope,
        "config": dict(sorted(_canonical_config(config).items())),
        "required_campaign_class": required_campaign_class,
        "evidence": list(evidence or ()),
        "next_required_action": _next_action(decision),
        "reasons": [dict(reason) for reason in reasons],
    }


def resolve_resource_capacity(
    request: Mapping[str, Any],
    resources: Sequence[ResourceCapacityEnvelopeV1],
    *,
    resource_id: str | None = None,
) -> dict[str, Any]:
    config = _canonical_config(request)
    candidates = [
        resource
        for resource in resources
        if resource_id is None or resource.resource_id == resource_id
    ]
    for resource in sorted(candidates, key=lambda row: row.resource_id):
        if resource.superseded_by or resource.decision == SUPERSEDED:
            return _result(
                SUPERSEDED,
                record_id=resource.resource_id,
                scope=SCOPE_RESOURCE_CAPACITY_ENVELOPE,
                config=config,
                reasons=[
                    {
                        "code": "resource_capacity_superseded",
                        "message": "resource-capacity record is superseded",
                    }
                ],
            )
        mismatches = resource.categorical_mismatches(config)
        if mismatches:
            continue
        if not resource.batch_size.contains(config.get("batch_size")):
            return _result(
                DELTA_REQUIRED,
                record_id=resource.resource_id,
                scope=SCOPE_RESOURCE_CAPACITY_ENVELOPE,
                config=config,
                required_campaign_class="batch_capacity_delta",
                reasons=[
                    {
                        "code": "batch_capacity_not_covered",
                        "batch_size": config.get("batch_size"),
                        "accepted_range": resource.batch_size.payload(),
                        "message": "batch size is outside the resource-capacity authority",
                    }
                ],
            )
        for exclusion in resource.exclusions:
            if exclusion.contains(config):
                return _result(
                    QUALIFIED_WITH_EXCLUSIONS,
                    record_id=resource.resource_id,
                    scope=SCOPE_RESOURCE_CAPACITY_ENVELOPE,
                    config=config,
                    reasons=[
                        {
                            "code": "resource_exclusion_region_hit",
                            "exclusion_id": exclusion.exclusion_id,
                            "message": exclusion.reason,
                        }
                    ],
                )
        return _result(
            resource.decision,
            record_id=resource.resource_id,
            scope=SCOPE_RESOURCE_CAPACITY_ENVELOPE,
            config=config,
            evidence=[item.payload() for item in resource.evidence],
            reasons=[
                {
                    "code": "resource_capacity_match",
                    "message": "request matches a separate resource-capacity authority",
                }
            ],
        )
    return _result(
        DELTA_REQUIRED,
        config=config,
        scope=SCOPE_RESOURCE_CAPACITY_ENVELOPE if resource_id else None,
        record_id=resource_id,
        required_campaign_class="batch_capacity_delta",
        reasons=[
            {
                "code": "no_applicable_resource_capacity_record",
                "message": "no resource-capacity authority applies to this request",
            }
        ],
    )


def impact_families(changed_axes: Sequence[str]) -> tuple[str, ...]:
    families: set[str] = set()
    for axis in changed_axes:
        if axis in {"grid_width", "grid_height"}:
            families.update(
                {
                    "route_shape_and_tensor_bounds",
                    "world_state_dimension_integrity",
                    "checkpoint_resume",
                    "deterministic_repeat",
                    "spatial_density_and_resource_normalization",
                    "substrate_pressure_invariant_checks",
                }
            )
        elif axis == "ticks_per_generation":
            families.update(
                {
                    "time_index_and_accumulator_bounds",
                    "checkpoint_continuation",
                    "finite_output_and_overflow",
                    "long_horizon_drift",
                    "delayed_state_persistence",
                }
            )
        elif axis == "batch_size":
            families.update(
                {
                    "physical_batch_execution",
                    "result_attribution",
                    "order_independence",
                    "memory_resource_bounds",
                }
            )
        elif axis in CATEGORICAL_AXES:
            families.update(
                {
                    "categorical_boundary_full_qualification",
                    "route_receipt_exact_binding",
                    "fail_closed_unsupported_route",
                }
            )
        else:
            families.add("explicit_axis_impact_review")
    return tuple(sorted(families))


def plan_delta_campaign(
    request: Mapping[str, Any],
    envelope: OperatingEnvelopeV1,
) -> dict[str, Any]:
    """Return a deterministic, sparse, interaction-aware delta plan."""

    config = _canonical_config(request)
    axis_names = tuple(sorted(axis.name for axis in envelope.axes))
    changed_axes = [
        name
        for name in axis_names
        if name in config and any(config.get(name) != point.get(name) for point in envelope.accepted_points)
    ]
    below_anchor = envelope.below_any_anchor(config)
    points = _selected_plan_points(config, envelope, below_anchor=below_anchor)
    recipes = select_delta_recipes(changed_axes)
    obligations = recipe_obligations(recipes)
    plan = {
        "schema_version": "eiel_qualification_delta_plan_v1",
        "plan_class": "below_anchor_delta" if below_anchor else "delta",
        "scope": envelope.scope,
        "envelope_id": envelope.envelope_id,
        "lifecycle_stage": DELTA_PLANNED,
        "authority_state": AUTHORITY_PLANNED_NON_AUTHORITATIVE,
        "accepted_region": envelope.accepted_region,
        "required_resource_capacity_id": envelope.required_resource_capacity_id,
        "requested_config": dict(sorted(config.items())),
        "changed_axes": sorted(changed_axes),
        "validation_families": impact_families(changed_axes),
        "selected_recipes": list(recipes),
        "required_obligations": list(obligations),
        "required_obligations_sha256": canonical_sha256(
            {"required_obligations": list(obligations)}
        ),
        "candidate_points": points,
        "decision_rule": (
            "A may accept only the preregistered region or a narrower region; "
            "no post-hoc widening is allowed."
        ),
        "next_required_action": "execute recipe obligations and import an A decision before authority publication",
        "real_campaign_authority": "not_authorized_by_qf1_planner",
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def _selected_plan_points(
    request: Mapping[str, Any],
    envelope: OperatingEnvelopeV1,
    *,
    below_anchor: bool,
) -> list[dict[str, Any]]:
    base = dict(envelope.categorical_invariants)
    points: list[dict[str, Any]] = []
    for corner in envelope.required_corners:
        points.append(
            {
                "point_id": f"corner-{len(points) + 1:02d}",
                "purpose": "required corner coverage",
                "config": {**base, **_canonical_config(corner)},
            }
        )
    if below_anchor:
        points.append(
            {
                "point_id": "below-anchor-request",
                "purpose": "explicit downscale extrapolation; smaller is not assumed safer",
                "config": {**base, **dict(request)},
            }
        )
        for axis in envelope.axes:
            if axis.name in request and axis.below(request[axis.name]):
                near = dict(request)
                near[axis.name] = axis.minimum
                points.append(
                    {
                        "point_id": f"lower-anchor-edge-{axis.name}",
                        "purpose": "nearest lower qualified edge comparison",
                        "config": {**base, **_canonical_config(near)},
                    }
                )
        return _dedupe_points(points)

    points.append(
        {
            "point_id": "requested-interior",
            "purpose": "requested candidate point",
            "config": {**base, **dict(request)},
        }
    )
    numeric_axes = sorted(envelope.axes, key=lambda axis: axis.name)
    if len(numeric_axes) >= 2:
        center = {
            axis.name: int((axis.minimum + axis.maximum) / 2)
            if isinstance(axis.minimum, int) and isinstance(axis.maximum, int)
            else (axis.minimum + axis.maximum) / 2
            for axis in numeric_axes
        }
        points.append(
            {
                "point_id": "center-interaction",
                "purpose": "center interaction probe",
                "config": {**base, **center},
            }
        )
        for axis in numeric_axes:
            edge = dict(center)
            edge[axis.name] = axis.minimum
            points.append(
                {
                    "point_id": f"edge-{axis.name}-lower",
                    "purpose": "fixed-axis edge probe",
                    "config": {**base, **edge},
                }
            )
    return _dedupe_points(points)


def _dedupe_points(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    out: list[dict[str, Any]] = []
    for point in points:
        config = point.get("config") if isinstance(point.get("config"), Mapping) else {}
        key = _point_key(config)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(point))
    return out


__all__ = [
    "AUTHORITY_CURRENT",
    "AUTHORITY_HISTORICAL_NON_CURRENT",
    "AUTHORITY_NO_CURRENT_AUTHORITY",
    "AUTHORITY_PLANNED_NON_AUTHORITATIVE",
    "AUTHORITY_REVALIDATION_REQUIRED",
    "AUTHORITY_SUPERSEDED",
    "DELTA_ACCEPTED",
    "DELTA_BLOCKED",
    "DELTA_CLASSIFIED",
    "DELTA_EXECUTED_PENDING_DECISION",
    "DELTA_INVALID",
    "DELTA_PLANNED",
    "DELTA_READY",
    "DELTA_REQUIRED",
    "DELTA_REQUIRED_BELOW_ANCHOR",
    "DELTA_REJECTED",
    "DELTA_SUPERSEDED",
    "FULL_QUALIFICATION_REQUIRED",
    "HISTORICAL_EVIDENCE_UNAVAILABLE",
    "HISTORICAL_UNVERIFIABLE",
    "HISTORICAL_VERIFIED_AT_TIME",
    "INVALID",
    "QUALIFIED",
    "QUALIFIED_WITH_EXCLUSIONS",
    "REVALIDATION_REQUIRED",
    "RECIPE_BATCH_CAPACITY_DELTA",
    "RECIPE_COMBINED_INTERACTION_DELTA",
    "RECIPE_SHAPE_POPULATION_DELTA",
    "RECIPE_TICK_ONLY_DELTA",
    "SCOPE_RESOURCE_CAPACITY_ENVELOPE",
    "SUPERSEDED",
    "UNSUPPORTED",
    "AxisRangeV1",
    "EvidenceBindingV1",
    "ExclusionRegionV1",
    "HistoricalAuthorityRecordV1",
    "OperatingEnvelopeV1",
    "QualificationAnchorV1",
    "QualificationCatalogV1",
    "QualificationFrameworkError",
    "ResourceCapacityEnvelopeV1",
    "candidate_matrix_status",
    "continuation_compatibility_class",
    "delta_status_row",
    "generate_delta_registry",
    "impact_families",
    "load_catalog",
    "plan_delta_campaign",
    "publish_accepted_delta",
    "recipe_obligations",
    "resolve_qualification",
    "resolve_resource_capacity",
    "select_delta_recipes",
    "validate_delta_decision_acceptance",
    "validate_delta_plan",
    "write_catalog",
]
