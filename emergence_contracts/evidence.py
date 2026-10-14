"""C-140 deterministic read-only Evidence evaluator contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ._validation import require_instance, require_text
from .artifacts import RunArtifactRefV1
from .ids import SchemaVersion
from .payloads import CanonicalPayloadV1
from .provider import ValidationResultV1


CONTRACT_ID = "C-140"
CONTRACT_OWNER = "Evidence"
API_VERSION = SchemaVersion("c140_v1")


@dataclass(frozen=True, slots=True)
class EvidenceProfileV1:
    profile_id: str
    profile_version: SchemaVersion
    rules: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_text(self.profile_id, field="profile_id")
        require_instance(self.profile_version, SchemaVersion, field="profile_version")
        require_instance(self.rules, CanonicalPayloadV1, field="rules")


@dataclass(frozen=True, slots=True)
class EvidenceEvaluationV1:
    profile: EvidenceProfileV1
    validation: ValidationResultV1
    comparison: CanonicalPayloadV1
    decision: CanonicalPayloadV1

    def __post_init__(self) -> None:
        require_instance(self.profile, EvidenceProfileV1, field="profile")
        require_instance(self.validation, ValidationResultV1, field="validation")
        require_instance(self.comparison, CanonicalPayloadV1, field="comparison")
        require_instance(self.decision, CanonicalPayloadV1, field="decision")


@runtime_checkable
class EvidenceEvaluatorV1(Protocol):
    def ingest(self, manifest_or_result_set_ref: RunArtifactRefV1) -> None: ...

    def validate(self) -> ValidationResultV1: ...

    def compare(self, profile: EvidenceProfileV1) -> CanonicalPayloadV1: ...

    def decide(self, profile: EvidenceProfileV1) -> EvidenceEvaluationV1: ...

    def materialize_report(self, output_ref: RunArtifactRefV1) -> RunArtifactRefV1: ...
