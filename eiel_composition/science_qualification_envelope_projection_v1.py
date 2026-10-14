"""Route-neutral exact-domain projection for science-instrument admission.

The qualification register and QRR remain authoritative.  This module merely
binds accepted exact values into a deterministic projection that can be checked
without effects.  It never infers ranges or grants qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from emergence_contracts import (
    CanonicalPayloadV1,
    ContentDigest,
    SchemaVersion,
    ScienceFindingV1,
    canonical_json_bytes,
    canonical_sha256,
)

SCIENCE_QUALIFIED_DOMAIN_ROW_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_qualified_domain_row_v1"
)
SCIENCE_QUALIFIED_ENVELOPE_PROJECTION_SCHEMA_VERSION = SchemaVersion(
    "eiel_science_qualified_envelope_projection_v1"
)


class ScienceEnvelopeProjectionError(ValueError):
    """An exact qualification projection is malformed or contradictory."""


@dataclass(frozen=True, slots=True)
class ScienceQualifiedDomainRowV1:
    subject_path: str
    accepted_values: CanonicalPayloadV1
    envelope_component_id: str
    evidence_refs: tuple[str, ...]
    schema_version: SchemaVersion = SCIENCE_QUALIFIED_DOMAIN_ROW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.subject_path, str)
            or not self.subject_path
            or self.subject_path.strip() != self.subject_path
        ):
            raise ScienceEnvelopeProjectionError("subject_path is invalid")
        if (
            not isinstance(self.envelope_component_id, str)
            or not self.envelope_component_id
            or self.envelope_component_id.strip() != self.envelope_component_id
        ):
            raise ScienceEnvelopeProjectionError("envelope_component_id is invalid")
        if tuple(sorted(set(self.evidence_refs))) != self.evidence_refs:
            raise ScienceEnvelopeProjectionError(
                "evidence_refs must be sorted and unique"
            )
        values = self.accepted_values.to_value()
        if not isinstance(values, list) or not values:
            raise ScienceEnvelopeProjectionError(
                "accepted_values must be a non-empty array"
            )
        keys = [canonical_json_bytes(item) for item in values]
        if keys != sorted(set(keys)):
            raise ScienceEnvelopeProjectionError(
                "accepted_values must be canonical, sorted, and unique"
            )
        if self.schema_version != SCIENCE_QUALIFIED_DOMAIN_ROW_SCHEMA_VERSION:
            raise ScienceEnvelopeProjectionError("unsupported domain-row schema")

    @classmethod
    def exact(
        cls,
        *,
        subject_path: str,
        values: tuple[object, ...],
        envelope_component_id: str,
        evidence_refs: tuple[str, ...],
    ) -> "ScienceQualifiedDomainRowV1":
        ordered = [
            value
            for _, value in sorted(
                {canonical_json_bytes(value): value for value in values}.items()
            )
        ]
        return cls(
            subject_path=subject_path,
            accepted_values=CanonicalPayloadV1.from_value(
                SchemaVersion("eiel_science_exact_qualified_values_v1"), ordered
            ),
            envelope_component_id=envelope_component_id,
            evidence_refs=evidence_refs,
        )

    def accepts(self, value: object) -> bool:
        requested = canonical_json_bytes(value)
        return requested in {
            canonical_json_bytes(item) for item in self.accepted_values.to_value()
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "subject_path": self.subject_path,
            "accepted_values": self.accepted_values.to_dict(),
            "envelope_component_id": self.envelope_component_id,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class ScienceQualifiedEnvelopeProjectionV1:
    instrument_manifest_sha256: ContentDigest
    route_id: str
    machine_profile_id: str
    qualification_authority_sha256: ContentDigest
    rows: tuple[ScienceQualifiedDomainRowV1, ...]
    schema_version: SchemaVersion = SCIENCE_QUALIFIED_ENVELOPE_PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        row_paths = [row.subject_path for row in self.rows]
        if row_paths != sorted(set(row_paths)):
            raise ScienceEnvelopeProjectionError(
                "qualified domain rows must be sorted and unique by subject_path"
            )
        if self.schema_version != SCIENCE_QUALIFIED_ENVELOPE_PROJECTION_SCHEMA_VERSION:
            raise ScienceEnvelopeProjectionError(
                "unsupported qualified-envelope projection schema"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": str(self.schema_version),
            "instrument_manifest_sha256": self.instrument_manifest_sha256.to_dict(),
            "route_id": self.route_id,
            "machine_profile_id": self.machine_profile_id,
            "qualification_authority_sha256": self.qualification_authority_sha256.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
        }

    def projection_sha256(self) -> ContentDigest:
        return canonical_sha256(self.to_dict())

    def findings_for(
        self, requested_subjects: dict[str, Any]
    ) -> tuple[ScienceFindingV1, ...]:
        by_path = {row.subject_path: row for row in self.rows}
        findings: list[ScienceFindingV1] = []
        for path, value in sorted(requested_subjects.items()):
            row = by_path.get(path)
            if row is None:
                findings.append(
                    ScienceFindingV1(
                        code="QUALIFIED_DOMAIN_ABSENT",
                        subject_path=path,
                        detail="No accepted exact qualification row exists.",
                    )
                )
            elif not row.accepts(value):
                findings.append(
                    ScienceFindingV1(
                        code="OUTSIDE_EXACT_QUALIFIED_DOMAIN",
                        subject_path=path,
                        detail=(
                            "Requested canonical value is absent from the accepted "
                            f"exact domain {row.envelope_component_id}."
                        ),
                        evidence_refs=row.evidence_refs,
                    )
                )
        return tuple(findings)


__all__ = (
    "SCIENCE_QUALIFIED_DOMAIN_ROW_SCHEMA_VERSION",
    "SCIENCE_QUALIFIED_ENVELOPE_PROJECTION_SCHEMA_VERSION",
    "ScienceEnvelopeProjectionError",
    "ScienceQualifiedDomainRowV1",
    "ScienceQualifiedEnvelopeProjectionV1",
)
