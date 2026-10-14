"""Typed composers for cross-path telemetry and qualification authority.

These functions only compose canonical payloads from already verified
evidence.  They do not run a gate, bless an unsuccessful result, load CUDA, or
publish files.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from .canonical import canonical_sha256
from .production_engine_provider import (
    CHECKPOINT_SCHEMA_VERSION,
    PROFILE_ID,
    PROVIDER_ID,
)
from .production_engine_identity import PRODUCTION_LIFECYCLE_OWNER_ID
from .mixed_mode_resident_isolation_gate import (
    GATE_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
)


REPORTING_QUALIFICATION_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_reporting_qualification_v1"
)
TELEMETRY_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_telemetry_contract_v1"
)
MECHANIC_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_mechanic_evidence_binding_v1"
)
CROSS_STORAGE_PARITY_EVIDENCE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_cross_storage_parity_evidence_v1"
)
CROSS_STORAGE_PARITY_CAPTURE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_cross_storage_resident_capture_v1"
)
CROSS_STORAGE_PARITY_PLAN_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_cross_storage_parity_plan_v1"
)
CROSS_STORAGE_PARITY_RESULT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_cross_storage_parity_result_v1"
)
MECHANIC_QUALIFICATION_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_mechanic_qualification_v2"
)
SAME_STORAGE_QUALIFICATION_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_same_storage_qualification_v1"
)

UPSTREAM_ARTIFACT_IDS: Final[tuple[str, ...]] = (
    "cuda_extension",
    "extension_manifest",
    "inheritance_registry",
    "telemetry_contract",
)
PARITY_DIMENSION_IDS: Final[tuple[str, ...]] = (
    "cells",
    "cell_order",
    "modes",
    "seeds",
    "generations",
    "checkpoints",
)
PARITY_EXACTNESS_IDS: Final[tuple[str, ...]] = (
    "final_state",
    "rng",
    "checkpoint",
    "attempt_content",
    "outcome_content",
    "selected_parent_content",
    "child_content",
    "commit_content",
    "donor_content",
    "primary_eligibility_aggregate",
    "mate_eligibility_aggregate",
    "reporting_independence",
)
STORAGE_MODES: Final[frozenset[str]] = frozenset(
    {"exact_full", "compact_complete"}
)


class InheritanceCrossPathQualificationReceiptError(ValueError):
    """Qualification evidence is incomplete, inconsistent, or unsuccessful."""


def _sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise InheritanceCrossPathQualificationReceiptError(
            f"{field} must be a lowercase SHA-256 digest"
        )
    return value


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InheritanceCrossPathQualificationReceiptError(
            f"{field} must be non-empty text"
        )
    return value


def _positive(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise InheritanceCrossPathQualificationReceiptError(
            f"{field} must be a positive integer"
        )
    return value


def _artifact_hashes(
    value: Mapping[str, Any],
    *,
    required: Sequence[str],
) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(required):
        raise InheritanceCrossPathQualificationReceiptError(
            "artifact hash set is incomplete"
        )
    return {
        artifact_id: _sha256(
            value[artifact_id], field=f"artifact_hashes.{artifact_id}"
        )
        for artifact_id in required
    }


@dataclass(frozen=True, slots=True)
class MechanicExecutionEnvelopeV1:
    """Execution extent established by the lower-level mechanic gate."""

    minimum_physical_batch: int
    maximum_physical_batch: int
    minimum_segment_generations: int
    maximum_segment_generations: int
    ticks_per_generation: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _positive(getattr(self, field), field=field)
        if self.minimum_physical_batch > self.maximum_physical_batch:
            raise InheritanceCrossPathQualificationReceiptError(
                "physical batch envelope is reversed"
            )
        if self.minimum_segment_generations > self.maximum_segment_generations:
            raise InheritanceCrossPathQualificationReceiptError(
                "logical generation envelope is reversed"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "lifecycle_owner": PRODUCTION_LIFECYCLE_OWNER_ID,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "physical_batch_range": [
                self.minimum_physical_batch,
                self.maximum_physical_batch,
            ],
            "logical_generation_range": [
                self.minimum_segment_generations,
                self.maximum_segment_generations,
            ],
            "ticks_per_generation": self.ticks_per_generation,
        }


@dataclass(frozen=True, slots=True)
class MixedIsolationEvidenceBindingV1:
    """Immutable file identities from the successful nine-check gate."""

    plan_artifact_sha256: str
    capture_manifest_sha256: str
    result_artifact_sha256: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            _sha256(getattr(self, field), field=field)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": MECHANIC_EVIDENCE_SCHEMA_VERSION,
            "plan_artifact_sha256": self.plan_artifact_sha256,
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "result_artifact_sha256": self.result_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class CrossStorageResidentCaptureV1:
    """One successful preregistered real-resident storage-mode capture."""

    storage_mode: str
    capture_sha256: str
    extension_sha256: str
    extension_manifest_sha256: str
    native_abi: str
    registry_sha256: str
    telemetry_contract_sha256: str
    dimension_sha256s: Mapping[str, str]
    ok: bool = True
    preregistered: bool = True
    real_resident_execution: bool = True

    def __post_init__(self) -> None:
        if self.storage_mode not in {"exact_full", "compact_complete"}:
            raise InheritanceCrossPathQualificationReceiptError(
                "capture storage_mode must be exact_full or compact_complete"
            )
        for field in (
            "capture_sha256",
            "extension_sha256",
            "extension_manifest_sha256",
            "registry_sha256",
            "telemetry_contract_sha256",
        ):
            _sha256(getattr(self, field), field=field)
        _text(self.native_abi, field="native_abi")
        object.__setattr__(
            self,
            "dimension_sha256s",
            _artifact_hashes(
                self.dimension_sha256s,
                required=PARITY_DIMENSION_IDS,
            ),
        )
        if (
            self.ok is not True
            or self.preregistered is not True
            or self.real_resident_execution is not True
        ):
            raise InheritanceCrossPathQualificationReceiptError(
                "cross-storage capture must be successful, preregistered, "
                "and a real resident execution"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CROSS_STORAGE_PARITY_CAPTURE_SCHEMA_VERSION,
            "storage_mode": self.storage_mode,
            "capture_sha256": self.capture_sha256,
            "ok": self.ok,
            "preregistered": self.preregistered,
            "real_resident_execution": self.real_resident_execution,
            "extension_sha256": self.extension_sha256,
            "extension_manifest_sha256": self.extension_manifest_sha256,
            "native_abi": self.native_abi,
            "registry_sha256": self.registry_sha256,
            "telemetry_contract_sha256": self.telemetry_contract_sha256,
            "dimension_sha256s": dict(self.dimension_sha256s),
        }


@dataclass(frozen=True, slots=True)
class CrossStorageParityEvidenceBindingV1:
    """Immutable exact-full to compact-complete production promotion proof."""

    parity_plan_sha256: str
    capture_manifest_sha256: str
    parity_result_sha256: str
    exact_full: CrossStorageResidentCaptureV1
    compact_complete: CrossStorageResidentCaptureV1
    exactness: Mapping[str, bool]

    def __post_init__(self) -> None:
        for field in (
            "parity_plan_sha256",
            "capture_manifest_sha256",
            "parity_result_sha256",
        ):
            _sha256(getattr(self, field), field=field)
        if (
            self.exact_full.storage_mode != "exact_full"
            or self.compact_complete.storage_mode != "compact_complete"
        ):
            raise InheritanceCrossPathQualificationReceiptError(
                "cross-storage evidence requires exact_full and compact_complete"
            )
        if (
            self.exact_full.extension_sha256
            != self.compact_complete.extension_sha256
            or self.exact_full.extension_manifest_sha256
            != self.compact_complete.extension_manifest_sha256
            or self.exact_full.native_abi != self.compact_complete.native_abi
            or self.exact_full.dimension_sha256s
            != self.compact_complete.dimension_sha256s
        ):
            raise InheritanceCrossPathQualificationReceiptError(
                "cross-storage captures differ outside storage registry/telemetry"
            )
        if (
            self.exact_full.capture_sha256
            == self.compact_complete.capture_sha256
            or
            self.exact_full.registry_sha256
            == self.compact_complete.registry_sha256
            or self.exact_full.telemetry_contract_sha256
            == self.compact_complete.telemetry_contract_sha256
        ):
            raise InheritanceCrossPathQualificationReceiptError(
                "cross-storage registry and telemetry hashes must be distinct"
            )
        if (
            not isinstance(self.exactness, Mapping)
            or set(self.exactness) != set(PARITY_EXACTNESS_IDS)
            or any(self.exactness[name] is not True for name in PARITY_EXACTNESS_IDS)
        ):
            raise InheritanceCrossPathQualificationReceiptError(
                "cross-storage parity obligations are incomplete or not exact"
            )
        object.__setattr__(
            self,
            "exactness",
            {name: True for name in PARITY_EXACTNESS_IDS},
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CROSS_STORAGE_PARITY_EVIDENCE_SCHEMA_VERSION,
            "ok": True,
            "parity_plan_sha256": self.parity_plan_sha256,
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "parity_result_sha256": self.parity_result_sha256,
            "captures": {
                "exact_full": self.exact_full.payload(),
                "compact_complete": self.compact_complete.payload(),
            },
            "exactness": dict(self.exactness),
        }


@dataclass(frozen=True, slots=True)
class CrossStorageParityPlanV1:
    """Preregistered artifact and dimension identity for two resident captures."""

    native_abi: str
    exact_full_artifacts: Mapping[str, str]
    compact_complete_artifacts: Mapping[str, str]
    dimension_sha256s: Mapping[str, str]

    def __post_init__(self) -> None:
        _text(self.native_abi, field="native_abi")
        exact = _artifact_hashes(
            self.exact_full_artifacts,
            required=UPSTREAM_ARTIFACT_IDS,
        )
        compact = _artifact_hashes(
            self.compact_complete_artifacts,
            required=UPSTREAM_ARTIFACT_IDS,
        )
        dimensions = _artifact_hashes(
            self.dimension_sha256s,
            required=PARITY_DIMENSION_IDS,
        )
        if (
            exact["cuda_extension"] != compact["cuda_extension"]
            or exact["extension_manifest"] != compact["extension_manifest"]
            or exact["inheritance_registry"] == compact["inheritance_registry"]
            or exact["telemetry_contract"] == compact["telemetry_contract"]
        ):
            raise InheritanceCrossPathQualificationReceiptError(
                "parity plan must vary only distinct registry and telemetry bindings"
            )
        object.__setattr__(self, "exact_full_artifacts", exact)
        object.__setattr__(self, "compact_complete_artifacts", compact)
        object.__setattr__(self, "dimension_sha256s", dimensions)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CROSS_STORAGE_PARITY_PLAN_SCHEMA_VERSION,
            "provider_id": PROVIDER_ID,
            "profile_id": PROFILE_ID,
            "execution_route": "real_resident",
            "preregistered_capture_count": 2,
            "native_abi": self.native_abi,
            "captures": {
                "exact_full": {
                    "storage_mode": "exact_full",
                    "artifact_hashes": dict(self.exact_full_artifacts),
                },
                "compact_complete": {
                    "storage_mode": "compact_complete",
                    "artifact_hashes": dict(self.compact_complete_artifacts),
                },
            },
            "dimension_sha256s": dict(self.dimension_sha256s),
            "required_exactness": list(PARITY_EXACTNESS_IDS),
        }

    @property
    def plan_sha256(self) -> str:
        return canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class CrossStorageParityResultV1:
    """Typed successful output from the preregistered parity verifier."""

    plan_sha256: str
    capture_manifest_sha256: str
    exact_full: CrossStorageResidentCaptureV1
    compact_complete: CrossStorageResidentCaptureV1
    exactness: Mapping[str, bool]
    ok: bool = True

    def __post_init__(self) -> None:
        _sha256(self.plan_sha256, field="plan_sha256")
        _sha256(self.capture_manifest_sha256, field="capture_manifest_sha256")
        if self.ok is not True:
            raise InheritanceCrossPathQualificationReceiptError(
                "cross-storage parity result must be successful"
            )
        # Reuse the complete capture and obligation checks rather than allowing
        # a second, weaker result shape.
        CrossStorageParityEvidenceBindingV1(
            parity_plan_sha256=self.plan_sha256,
            capture_manifest_sha256=self.capture_manifest_sha256,
            parity_result_sha256="0" * 64,
            exact_full=self.exact_full,
            compact_complete=self.compact_complete,
            exactness=self.exactness,
        )
        object.__setattr__(
            self,
            "exactness",
            {name: True for name in PARITY_EXACTNESS_IDS},
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CROSS_STORAGE_PARITY_RESULT_SCHEMA_VERSION,
            "verifier_id": (
                "inheritance_cross_path_cross_storage_parity_verifier_v1"
            ),
            "ok": self.ok,
            "plan_sha256": self.plan_sha256,
            "capture_manifest_sha256": self.capture_manifest_sha256,
            "captures": {
                "exact_full": self.exact_full.payload(),
                "compact_complete": self.compact_complete.payload(),
            },
            "exactness": dict(self.exactness),
        }

    @property
    def result_sha256(self) -> str:
        return canonical_sha256(self.payload())


def _validate_cross_storage_payloads(
    *,
    plan: CrossStorageParityPlanV1,
    result: CrossStorageParityResultV1,
    evidence: CrossStorageParityEvidenceBindingV1,
) -> None:
    if (
        result.plan_sha256 != plan.plan_sha256
        or evidence.parity_plan_sha256 != plan.plan_sha256
        or evidence.capture_manifest_sha256 != result.capture_manifest_sha256
        or evidence.parity_result_sha256 != result.result_sha256
        or evidence.exact_full != result.exact_full
        or evidence.compact_complete != result.compact_complete
        or dict(evidence.exactness) != dict(result.exactness)
    ):
        raise InheritanceCrossPathQualificationReceiptError(
            "parity evidence does not bind the typed plan and verifier result"
        )
    exact = result.exact_full
    compact = result.compact_complete
    if (
        dict(plan.exact_full_artifacts)
        != {
            "cuda_extension": exact.extension_sha256,
            "extension_manifest": exact.extension_manifest_sha256,
            "inheritance_registry": exact.registry_sha256,
            "telemetry_contract": exact.telemetry_contract_sha256,
        }
        or dict(plan.compact_complete_artifacts)
        != {
            "cuda_extension": compact.extension_sha256,
            "extension_manifest": compact.extension_manifest_sha256,
            "inheritance_registry": compact.registry_sha256,
            "telemetry_contract": compact.telemetry_contract_sha256,
        }
        or dict(plan.dimension_sha256s) != dict(exact.dimension_sha256s)
        or dict(plan.dimension_sha256s) != dict(compact.dimension_sha256s)
        or plan.native_abi != exact.native_abi
        or plan.native_abi != compact.native_abi
    ):
        raise InheritanceCrossPathQualificationReceiptError(
            "typed parity result differs from its preregistered plan"
        )


@dataclass(frozen=True, slots=True)
class ReportingModeEvidenceV1:
    evidence_sha256: str
    authoritative_state_sha256: str
    reporting_rows_present: bool

    def __post_init__(self) -> None:
        _sha256(self.evidence_sha256, field="evidence_sha256")
        _sha256(
            self.authoritative_state_sha256,
            field="authoritative_state_sha256",
        )
        if type(self.reporting_rows_present) is not bool:
            raise InheritanceCrossPathQualificationReceiptError(
                "reporting_rows_present must be boolean"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "evidence_sha256": self.evidence_sha256,
            "authoritative_state_sha256": self.authoritative_state_sha256,
            "reporting_rows_present": self.reporting_rows_present,
        }


def compose_cross_path_telemetry_contract_v1(
    *,
    native_abi: str,
    registry_sha256: str,
    projection_contract: Mapping[str, Any],
) -> dict[str, Any]:
    raise InheritanceCrossPathQualificationReceiptError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path telemetry contract composition is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if not isinstance(projection_contract, Mapping) or not projection_contract:
        raise InheritanceCrossPathQualificationReceiptError(
            "projection contract must be a non-empty object"
        )
    return {
        "schema_version": TELEMETRY_CONTRACT_SCHEMA_VERSION,
        "provider_id": PROVIDER_ID,
        "profile_id": PROFILE_ID,
        "native_abi": _text(native_abi, field="native_abi"),
        "registry_sha256": _sha256(
            registry_sha256, field="registry_sha256"
        ),
        "projection_contract": dict(projection_contract),
    }


def compose_cross_path_mechanic_receipt_v1(
    *,
    native_abi: str,
    upstream_artifact_hashes: Mapping[str, Any],
    gate_plan: Mapping[str, Any],
    gate_result: Mapping[str, Any],
    evidence: MixedIsolationEvidenceBindingV1,
    envelope: MechanicExecutionEnvelopeV1,
    production_storage_mode: str,
    production_artifact_hashes: Mapping[str, Any] | None = None,
    cross_storage_parity: CrossStorageParityEvidenceBindingV1 | None = None,
    cross_storage_plan: CrossStorageParityPlanV1 | None = None,
    cross_storage_result: CrossStorageParityResultV1 | None = None,
) -> dict[str, Any]:
    raise InheritanceCrossPathQualificationReceiptError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path mechanic receipt composition is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if production_storage_mode not in STORAGE_MODES:
        raise InheritanceCrossPathQualificationReceiptError(
            "production_storage_mode is invalid"
        )
    qualified_hashes = _artifact_hashes(
        upstream_artifact_hashes, required=UPSTREAM_ARTIFACT_IDS
    )
    production_hashes = _artifact_hashes(
        (
            qualified_hashes
            if production_artifact_hashes is None
            else production_artifact_hashes
        ),
        required=UPSTREAM_ARTIFACT_IDS,
    )
    if (
        not isinstance(gate_plan, Mapping)
        or gate_plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or not isinstance(gate_result, Mapping)
        or gate_result.get("schema_version") != GATE_SCHEMA_VERSION
        or gate_result.get("ok") is not True
        or gate_result.get("plan_sha256") is None
    ):
        raise InheritanceCrossPathQualificationReceiptError(
            "mixed-isolation evidence is not a successful v1 gate"
        )
    plan_digest = gate_result["plan_sha256"]
    _sha256(plan_digest, field="gate_result.plan_sha256")
    if canonical_sha256(gate_plan) != plan_digest:
        raise InheritanceCrossPathQualificationReceiptError(
            "mixed-isolation result binds another plan"
        )
    if cross_storage_parity is None:
        if cross_storage_plan is not None or cross_storage_result is not None:
            raise InheritanceCrossPathQualificationReceiptError(
                "typed parity plan/result require cross-storage evidence"
            )
        if qualified_hashes != production_hashes:
            raise InheritanceCrossPathQualificationReceiptError(
                "different production hashes require cross-storage parity evidence"
            )
        storage_qualification = {
            "schema_version": SAME_STORAGE_QUALIFICATION_SCHEMA_VERSION,
            "mode": "same_storage",
            "storage_mode": production_storage_mode,
            "hashes_identical": True,
            "qualified_gate_artifacts_sha256": canonical_sha256(qualified_hashes),
            "production_artifacts_sha256": canonical_sha256(production_hashes),
        }
    else:
        if (
            production_storage_mode != "compact_complete"
            or cross_storage_plan is None
            or cross_storage_result is None
        ):
            raise InheritanceCrossPathQualificationReceiptError(
                "compact promotion requires typed parity plan and result payloads"
            )
        _validate_cross_storage_payloads(
            plan=cross_storage_plan,
            result=cross_storage_result,
            evidence=cross_storage_parity,
        )
        exact_capture = cross_storage_parity.exact_full
        compact_capture = cross_storage_parity.compact_complete
        expected_qualified = {
            "cuda_extension": exact_capture.extension_sha256,
            "extension_manifest": exact_capture.extension_manifest_sha256,
            "inheritance_registry": exact_capture.registry_sha256,
            "telemetry_contract": exact_capture.telemetry_contract_sha256,
        }
        expected_production = {
            "cuda_extension": compact_capture.extension_sha256,
            "extension_manifest": compact_capture.extension_manifest_sha256,
            "inheritance_registry": compact_capture.registry_sha256,
            "telemetry_contract": compact_capture.telemetry_contract_sha256,
        }
        if (
            qualified_hashes != expected_qualified
            or production_hashes != expected_production
            or exact_capture.native_abi != native_abi
        ):
            raise InheritanceCrossPathQualificationReceiptError(
                "cross-storage parity evidence does not bind the supplied artifact chains"
            )
        storage_qualification = {
            "mode": "cross_storage_promotion",
            "from_storage_mode": "exact_full",
            "to_storage_mode": "compact_complete",
            "parity_evidence": cross_storage_parity.payload(),
            "parity_plan": cross_storage_plan.payload(),
            "parity_result": cross_storage_result.payload(),
        }
    return {
        "schema_version": MECHANIC_QUALIFICATION_SCHEMA_VERSION,
        "provider_id": PROVIDER_ID,
        "profile_id": PROFILE_ID,
        "ok": True,
        "registry_sha256": production_hashes["inheritance_registry"],
        "artifact": {
            "sha256": production_hashes["cuda_extension"],
            "abi": _text(native_abi, field="native_abi"),
        },
        "artifact_bindings": production_hashes,
        "qualified_gate_artifacts": qualified_hashes,
        "production_artifacts": production_hashes,
        "storage_qualification": storage_qualification,
        "execution_contract": envelope.payload(),
        "mixed_mode_resident_isolation": evidence.payload(),
        "gate_plan_sha256": plan_digest,
    }


def compose_cross_path_reporting_receipt_v1(
    *,
    upstream_artifact_hashes: Mapping[str, Any],
    mechanic_receipt_sha256: str,
    enabled: ReportingModeEvidenceV1,
    disabled: ReportingModeEvidenceV1,
    native_reproduction_exposure_exact: bool,
) -> dict[str, Any]:
    raise InheritanceCrossPathQualificationReceiptError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path reporting receipt composition is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    hashes = _artifact_hashes(
        upstream_artifact_hashes, required=UPSTREAM_ARTIFACT_IDS
    )
    mechanic_hash = _sha256(
        mechanic_receipt_sha256, field="mechanic_receipt_sha256"
    )
    if (
        enabled.reporting_rows_present is not True
        or disabled.reporting_rows_present is not False
        or enabled.authoritative_state_sha256
        != disabled.authoritative_state_sha256
        or native_reproduction_exposure_exact is not True
    ):
        raise InheritanceCrossPathQualificationReceiptError(
            "enabled/disabled reporting evidence is not exact"
        )
    return {
        "schema_version": REPORTING_QUALIFICATION_SCHEMA_VERSION,
        "provider_id": PROVIDER_ID,
        "profile_id": PROFILE_ID,
        "ok": True,
        "artifact_bindings": {
            **hashes,
            "qualification_receipt": mechanic_hash,
        },
        "reporting_parity": {
            "authoritative_state_exact": True,
            "native_reproduction_exposure_exact": True,
            "enabled": enabled.payload(),
            "disabled": disabled.payload(),
        },
    }


__all__ = [
    "CROSS_STORAGE_PARITY_CAPTURE_SCHEMA_VERSION",
    "CROSS_STORAGE_PARITY_EVIDENCE_SCHEMA_VERSION",
    "CROSS_STORAGE_PARITY_PLAN_SCHEMA_VERSION",
    "CROSS_STORAGE_PARITY_RESULT_SCHEMA_VERSION",
    "CrossStorageParityEvidenceBindingV1",
    "CrossStorageParityPlanV1",
    "CrossStorageParityResultV1",
    "CrossStorageResidentCaptureV1",
    "InheritanceCrossPathQualificationReceiptError",
    "MECHANIC_EVIDENCE_SCHEMA_VERSION",
    "MECHANIC_QUALIFICATION_SCHEMA_VERSION",
    "MechanicExecutionEnvelopeV1",
    "MixedIsolationEvidenceBindingV1",
    "REPORTING_QUALIFICATION_SCHEMA_VERSION",
    "ReportingModeEvidenceV1",
    "TELEMETRY_CONTRACT_SCHEMA_VERSION",
    "PARITY_DIMENSION_IDS",
    "PARITY_EXACTNESS_IDS",
    "SAME_STORAGE_QUALIFICATION_SCHEMA_VERSION",
    "UPSTREAM_ARTIFACT_IDS",
    "compose_cross_path_mechanic_receipt_v1",
    "compose_cross_path_reporting_receipt_v1",
    "compose_cross_path_telemetry_contract_v1",
]
