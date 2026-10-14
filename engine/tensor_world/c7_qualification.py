"""Frozen-input and evidence-integrity contracts for the C7 qualification seam.

This module does not run qualification or confer admission.  It prevents a
future C7 run from silently mixing source commits, route fingerprints, device
conditions, seeds, cadence, or evidence roots.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

from .contracts import TensorWorldContractError


C7_CONTRACT_SCHEMA_VERSION_V1: Final[str] = "tensor_world_c7_qualification_contract_v1"

# These are the identity/configuration fields which must be copied into every
# C7 evidence receipt.  A freeze fingerprint alone is insufficient: it proves
# that a receipt names a freeze, but does not prove that the individual
# evidence inputs were produced under that freeze.  Keep this list explicit so
# adding a freeze field cannot silently leave it unbound in a future decision.
C7_EVIDENCE_BINDING_KEYS_V1: Final[tuple[str, ...]] = (
    "source_commit",
    "oracle_commit",
    "route_fingerprint",
    "backend_id",
    "provider_contract_id",
    "batch",
    "grid_size",
    "initial_population",
    "slot_capacity",
    "policy_family",
    "mechanics",
    "checkpoint_cadence",
    "seeds",
    "case_ids",
    "evidence_root",
    "working_tree_clean",
)

# Keep the qualification ladder and decision surface in source as well as in
# the human-facing contract artifact.  A future runner can use this sequence
# to reject an incomplete receipt instead of inferring coverage from a subset
# of booleans.  These are preparation metadata only; they do not execute a
# qualification step or authorize a gate.
C7_QUALIFICATION_LADDER_V1: Final[tuple[str, ...]] = (
    "Q0_source_contract_integrity",
    "Q1_focused_suites",
    "Q2_one_tick_oracle_parity",
    "Q3_short_multitick_parity",
    "Q4_population_lineage_parity",
    "Q5_WDL_EED_temporal_parity",
    "Q6_deterministic_repeat",
    "Q7_checkpoint_resume",
    "Q8_provider_B48",
    "Q9_performance",
    "Q10_resource_VRAM",
    "Q11_external_metrics_evidence",
    "Q12_failure_injection",
)

C7_DECISION_SURFACES_V1: Final[tuple[str, ...]] = (
    "semantic_parity",
    "determinism_parity",
    "checkpoint_replay_parity",
    "external_contract",
    "residency",
    "resource_safe",
    "material_performance",
)


class TensorWorldC7DecisionV1(StrEnum):
    QUALIFIED_FOR_ADMISSION_REVIEW = "TW-C7-QUALIFIED-FOR-ADMISSION-REVIEW"
    SEMANTIC_FAIL = "TW-C7-SEMANTIC-FAIL"
    DETERMINISM_FAIL = "TW-C7-DETERMINISM-FAIL"
    CHECKPOINT_REPLAY_FAIL = "TW-C7-CHECKPOINT-REPLAY-FAIL"
    EXTERNAL_CONTRACT_FAIL = "TW-C7-EXTERNAL-CONTRACT-FAIL"
    RESIDENCY_FAIL = "TW-C7-RESIDENCY-FAIL"
    RESOURCE_FAIL = "TW-C7-RESOURCE-FAIL"
    PERFORMANCE_NOT_MATERIAL = "TW-C7-PERFORMANCE-NOT-MATERIAL"
    QUALIFICATION_INVALID = "TW-C7-QUALIFICATION-INVALID"


@dataclass(frozen=True, slots=True)
class TensorWorldC7QualificationFreezeV1:
    source_commit: str
    oracle_commit: str
    route_fingerprint: str
    backend_id: str
    provider_contract_id: str
    batch: int
    grid_size: int
    initial_population: int
    slot_capacity: int
    policy_family: str
    mechanics: tuple[str, ...]
    checkpoint_cadence: int
    seeds: tuple[int, ...]
    case_ids: tuple[str, ...]
    evidence_root: str
    working_tree_clean: bool
    schema_version: str = C7_CONTRACT_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C7_CONTRACT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C7 qualification schema version is unsupported")
        for name in (
            "source_commit",
            "oracle_commit",
            "route_fingerprint",
            "backend_id",
            "provider_contract_id",
            "policy_family",
            "evidence_root",
        ):
            if not getattr(self, name):
                raise TensorWorldContractError(f"C7 {name} is required")
        if len(self.route_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.route_fingerprint
        ):
            raise TensorWorldContractError("C7 route fingerprint is not SHA256")
        for name in (
            "batch",
            "grid_size",
            "initial_population",
            "slot_capacity",
            "checkpoint_cadence",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise TensorWorldContractError(f"C7 {name} must be a positive int")
        if not self.mechanics or len(set(self.mechanics)) != len(self.mechanics):
            raise TensorWorldContractError("C7 mechanics must be nonempty and unique")
        if not self.seeds or any(type(seed) is not int for seed in self.seeds):
            raise TensorWorldContractError("C7 seeds must contain at least one int")
        if not self.case_ids or any(not case_id for case_id in self.case_ids):
            raise TensorWorldContractError("C7 case_ids must contain at least one identity")
        if len(self.seeds) != len(self.case_ids):
            raise TensorWorldContractError("C7 seed and case counts must match")
        if type(self.working_tree_clean) is not bool:
            raise TensorWorldContractError("C7 working_tree_clean must be bool")
        if not self.working_tree_clean:
            raise TensorWorldContractError("C7 qualification requires a clean working tree")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "oracle_commit": self.oracle_commit,
            "route_fingerprint": self.route_fingerprint,
            "backend_id": self.backend_id,
            "provider_contract_id": self.provider_contract_id,
            "batch": self.batch,
            "grid_size": self.grid_size,
            "initial_population": self.initial_population,
            "slot_capacity": self.slot_capacity,
            "policy_family": self.policy_family,
            "mechanics": list(self.mechanics),
            "checkpoint_cadence": self.checkpoint_cadence,
            "seeds": list(self.seeds),
            "case_ids": list(self.case_ids),
            "evidence_root": self.evidence_root,
            "working_tree_clean": self.working_tree_clean,
        }

    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def c7_freeze_evidence_bindings_v1(
    freeze: TensorWorldC7QualificationFreezeV1,
) -> dict[str, str]:
    """Return the canonical identity bindings required by a C7 receipt.

    Values are strings because evidence manifests commonly cross a JSON or
    process boundary.  Structured freeze values use canonical compact JSON so
    tuple/list representation and whitespace cannot create ambiguous bindings.
    """

    return {
        "source_commit": freeze.source_commit,
        "oracle_commit": freeze.oracle_commit,
        "route_fingerprint": freeze.route_fingerprint,
        "backend_id": freeze.backend_id,
        "provider_contract_id": freeze.provider_contract_id,
        "batch": str(freeze.batch),
        "grid_size": str(freeze.grid_size),
        "initial_population": str(freeze.initial_population),
        "slot_capacity": str(freeze.slot_capacity),
        "policy_family": freeze.policy_family,
        "mechanics": json.dumps(list(freeze.mechanics), separators=(",", ":")),
        "checkpoint_cadence": str(freeze.checkpoint_cadence),
        "seeds": json.dumps(list(freeze.seeds), separators=(",", ":")),
        "case_ids": json.dumps(list(freeze.case_ids), separators=(",", ":")),
        "evidence_root": freeze.evidence_root,
        "working_tree_clean": str(freeze.working_tree_clean).lower(),
    }


def c7_evidence_bindings_match_freeze_v1(
    freeze: TensorWorldC7QualificationFreezeV1,
    evidence_bindings: Mapping[str, str],
) -> bool:
    """Return whether a receipt binds exactly to one frozen input packet.

    Requiring an exact key set is intentional.  Missing bindings are unsafe,
    but silently accepting extra unbound identity fields is also unsafe: a
    producer could vary a configuration field without the C7 decision
    surface noticing.  The helper remains a predicate so callers can report
    a normal ``QUALIFICATION_INVALID`` decision rather than turning a receipt
    validation failure into an exception.
    """

    expected = c7_freeze_evidence_bindings_v1(freeze)
    return set(evidence_bindings) == set(expected) and all(
        evidence_bindings.get(name) == value
        for name, value in expected.items()
    )


def c7_verify_artifact_bindings_v1(
    repository_root: str | Path,
    artifact_bindings: Mapping[str, str],
) -> dict[str, object]:
    """Verify relative repository artifact hashes without widening the root.

    C7 packets carry source/artifact identity, but a packet consumer still
    needs one explicit, fail-closed check that the named files are present and
    unchanged.  Paths are repository-relative by contract; absolute paths and
    traversal are reported as invalid instead of being resolved outside the
    supplied root.  The result is a diagnostic mapping so a qualification
    runner can turn a mismatch into ``QUALIFICATION_INVALID`` without treating
    an operator typo as an implementation exception.
    """

    root = Path(repository_root).resolve()
    if not isinstance(artifact_bindings, Mapping):
        raise TensorWorldContractError("C7 artifact bindings must be a mapping")
    missing: list[str] = []
    invalid: list[str] = []
    mismatches: list[dict[str, str]] = []
    checked: list[str] = []
    for relative_name, expected in sorted(artifact_bindings.items()):
        if (
            type(relative_name) is not str
            or not relative_name
            or type(expected) is not str
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            invalid.append(str(relative_name))
            continue
        candidate = Path(relative_name)
        if candidate.is_absolute() or ".." in candidate.parts:
            invalid.append(relative_name)
            continue
        path = (root / candidate).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            invalid.append(relative_name)
            continue
        if not path.is_file():
            missing.append(relative_name)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        checked.append(relative_name)
        if actual != expected:
            mismatches.append(
                {"path": relative_name, "expected": expected, "actual": actual}
            )
    return {
        "pass": not missing and not invalid and not mismatches and bool(checked),
        "checked": checked,
        "missing": missing,
        "invalid": invalid,
        "mismatches": mismatches,
    }


@dataclass(frozen=True, slots=True)
class TensorWorldC7EvidenceReceiptV1:
    freeze_fingerprint: str
    semantic_parity: bool
    determinism_parity: bool
    checkpoint_replay_parity: bool
    external_contract: bool
    residency: bool
    resource_safe: bool
    material_performance: bool
    evidence_bindings: Mapping[str, str]
    schema_version: str = C7_CONTRACT_SCHEMA_VERSION_V1

    def __post_init__(self) -> None:
        if self.schema_version != C7_CONTRACT_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C7 evidence schema version is unsupported")
        if len(self.freeze_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in self.freeze_fingerprint
        ):
            raise TensorWorldContractError("C7 freeze fingerprint is not SHA256")
        for name in (
            "semantic_parity",
            "determinism_parity",
            "checkpoint_replay_parity",
            "external_contract",
            "residency",
            "resource_safe",
            "material_performance",
        ):
            if type(getattr(self, name)) is not bool:
                raise TensorWorldContractError(f"C7 {name} must be bool")
        if not self.evidence_bindings or any(
            type(key) is not str
            or not key
            or type(value) is not str
            or not value
            for key, value in self.evidence_bindings.items()
        ):
            raise TensorWorldContractError(
                "C7 evidence bindings must be a nonempty string mapping"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the receipt without changing its evidence meaning."""

        return {
            "schema_version": self.schema_version,
            "freeze_fingerprint": self.freeze_fingerprint,
            "semantic_parity": self.semantic_parity,
            "determinism_parity": self.determinism_parity,
            "checkpoint_replay_parity": self.checkpoint_replay_parity,
            "external_contract": self.external_contract,
            "residency": self.residency,
            "resource_safe": self.resource_safe,
            "material_performance": self.material_performance,
            "evidence_bindings": dict(
                sorted(self.evidence_bindings.items(), key=lambda item: item[0])
            ),
        }


def decide_tensor_world_c7_v1(
    freeze: TensorWorldC7QualificationFreezeV1,
    evidence: TensorWorldC7EvidenceReceiptV1,
) -> TensorWorldC7DecisionV1:
    """Return a decision only from complete, freeze-bound evidence."""

    if evidence.freeze_fingerprint != freeze.fingerprint():
        return TensorWorldC7DecisionV1.QUALIFICATION_INVALID
    if not c7_evidence_bindings_match_freeze_v1(
        freeze, evidence.evidence_bindings
    ):
        return TensorWorldC7DecisionV1.QUALIFICATION_INVALID
    required = (
        (evidence.semantic_parity, TensorWorldC7DecisionV1.SEMANTIC_FAIL),
        (evidence.determinism_parity, TensorWorldC7DecisionV1.DETERMINISM_FAIL),
        (evidence.checkpoint_replay_parity, TensorWorldC7DecisionV1.CHECKPOINT_REPLAY_FAIL),
        (evidence.external_contract, TensorWorldC7DecisionV1.EXTERNAL_CONTRACT_FAIL),
        (evidence.residency, TensorWorldC7DecisionV1.RESIDENCY_FAIL),
        (evidence.resource_safe, TensorWorldC7DecisionV1.RESOURCE_FAIL),
        (evidence.material_performance, TensorWorldC7DecisionV1.PERFORMANCE_NOT_MATERIAL),
    )
    for passed, decision in required:
        if not passed:
            return decision
    return TensorWorldC7DecisionV1.QUALIFIED_FOR_ADMISSION_REVIEW


__all__ = [
    "C7_CONTRACT_SCHEMA_VERSION_V1",
    "C7_DECISION_SURFACES_V1",
    "C7_EVIDENCE_BINDING_KEYS_V1",
    "C7_QUALIFICATION_LADDER_V1",
    "TensorWorldC7DecisionV1",
    "TensorWorldC7EvidenceReceiptV1",
    "TensorWorldC7QualificationFreezeV1",
    "c7_freeze_evidence_bindings_v1",
    "c7_evidence_bindings_match_freeze_v1",
    "c7_verify_artifact_bindings_v1",
    "decide_tensor_world_c7_v1",
]
