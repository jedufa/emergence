"""C7 freeze/evidence integrity tests; no qualification is run here."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.tensor_world.c7_qualification import (
    C7_DECISION_SURFACES_V1,
    TensorWorldC7DecisionV1,
    TensorWorldC7EvidenceReceiptV1,
    TensorWorldC7QualificationFreezeV1,
    C7_EVIDENCE_BINDING_KEYS_V1,
    C7_QUALIFICATION_LADDER_V1,
    c7_freeze_evidence_bindings_v1,
    c7_evidence_bindings_match_freeze_v1,
    c7_verify_artifact_bindings_v1,
    decide_tensor_world_c7_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


def _freeze() -> TensorWorldC7QualificationFreezeV1:
    return TensorWorldC7QualificationFreezeV1(
        source_commit="source-commit",
        oracle_commit="oracle-commit",
        route_fingerprint="a" * 64,
        backend_id="tensor_world_c3_v1",
        provider_contract_id="tensor_world_provider_contract_v1",
        batch=48,
        grid_size=32,
        initial_population=256,
        slot_capacity=512,
        policy_family="fixed_mlp",
        mechanics=("patch", "jungle", "wdl", "eed"),
        checkpoint_cadence=8,
        seeds=(41, 42),
        case_ids=("b48_case_41", "b48_case_42"),
        evidence_root="/tmp/emergence-test/runtime/evidence_raw/tensor_world/c7",
        working_tree_clean=True,
    )


def _evidence(freeze: TensorWorldC7QualificationFreezeV1, **overrides: bool):
    values = {
        "semantic_parity": True,
        "determinism_parity": True,
        "checkpoint_replay_parity": True,
        "external_contract": True,
        "residency": True,
        "resource_safe": True,
        "material_performance": True,
    }
    values.update(overrides)
    return TensorWorldC7EvidenceReceiptV1(
        freeze_fingerprint=freeze.fingerprint(),
        evidence_bindings=c7_freeze_evidence_bindings_v1(freeze),
        **values,
    )


def test_c7_freeze_requires_clean_tree_and_matching_case_seed_sets() -> None:
    with pytest.raises(TensorWorldContractError, match="clean working tree"):
        values = _freeze().to_dict()
        values["working_tree_clean"] = False
        TensorWorldC7QualificationFreezeV1(**values)


def test_c7_complete_evidence_decides_only_admission_review() -> None:
    freeze = _freeze()
    assert decide_tensor_world_c7_v1(freeze, _evidence(freeze)) is (
        TensorWorldC7DecisionV1.QUALIFIED_FOR_ADMISSION_REVIEW
    )


def test_c7_failure_precedence_is_explicit() -> None:
    freeze = _freeze()
    assert decide_tensor_world_c7_v1(
        freeze, _evidence(freeze, semantic_parity=False, residency=False)
    ) is TensorWorldC7DecisionV1.SEMANTIC_FAIL
    assert decide_tensor_world_c7_v1(
        freeze, _evidence(freeze, material_performance=False)
    ) is TensorWorldC7DecisionV1.PERFORMANCE_NOT_MATERIAL


def test_c7_mixed_freeze_evidence_is_invalid() -> None:
    freeze = _freeze()
    evidence = _evidence(freeze)
    other = _freeze().to_dict()
    other["source_commit"] = "different-source"
    changed = TensorWorldC7QualificationFreezeV1(**other)
    assert decide_tensor_world_c7_v1(changed, evidence) is (
        TensorWorldC7DecisionV1.QUALIFICATION_INVALID
    )


def test_c7_evidence_requires_every_frozen_identity_binding() -> None:
    freeze = _freeze()
    bindings = c7_freeze_evidence_bindings_v1(freeze)
    del bindings["route_fingerprint"]
    evidence = TensorWorldC7EvidenceReceiptV1(
        freeze_fingerprint=freeze.fingerprint(),
        evidence_bindings=bindings,
        semantic_parity=True,
        determinism_parity=True,
        checkpoint_replay_parity=True,
        external_contract=True,
        residency=True,
        resource_safe=True,
        material_performance=True,
    )
    assert decide_tensor_world_c7_v1(freeze, evidence) is (
        TensorWorldC7DecisionV1.QUALIFICATION_INVALID
    )


def test_c7_evidence_rejects_binding_value_drift() -> None:
    freeze = _freeze()
    bindings = c7_freeze_evidence_bindings_v1(freeze)
    bindings["checkpoint_cadence"] = "16"
    evidence = TensorWorldC7EvidenceReceiptV1(
        freeze_fingerprint=freeze.fingerprint(),
        evidence_bindings=bindings,
        semantic_parity=True,
        determinism_parity=True,
        checkpoint_replay_parity=True,
        external_contract=True,
        residency=True,
        resource_safe=True,
        material_performance=True,
    )
    assert decide_tensor_world_c7_v1(freeze, evidence) is (
        TensorWorldC7DecisionV1.QUALIFICATION_INVALID
    )


def test_c7_evidence_rejects_unbound_extra_identity_fields() -> None:
    freeze = _freeze()
    bindings = c7_freeze_evidence_bindings_v1(freeze)
    bindings["device_uuid"] = "GPU-should-not-be-implicit"
    evidence = _evidence(freeze)
    evidence = TensorWorldC7EvidenceReceiptV1(
        **{
            **evidence.to_dict(),
            "evidence_bindings": bindings,
        }
    )
    assert not c7_evidence_bindings_match_freeze_v1(freeze, bindings)
    assert decide_tensor_world_c7_v1(freeze, evidence) is (
        TensorWorldC7DecisionV1.QUALIFICATION_INVALID
    )


def test_c7_artifact_binding_verifier_is_fail_closed_and_repository_relative(
    tmp_path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

    verified = c7_verify_artifact_bindings_v1(
        tmp_path, {"artifact.json": digest}
    )
    assert verified == {
        "pass": True,
        "checked": ["artifact.json"],
        "missing": [],
        "invalid": [],
        "mismatches": [],
    }

    artifact.write_text("changed\n", encoding="utf-8")
    changed = c7_verify_artifact_bindings_v1(
        tmp_path, {"artifact.json": digest, "missing.json": "a" * 64}
    )
    assert changed["pass"] is False
    assert changed["missing"] == ["missing.json"]
    assert changed["mismatches"][0]["path"] == "artifact.json"

    invalid = c7_verify_artifact_bindings_v1(
        tmp_path, {"../outside.json": digest, "/tmp/outside.json": digest}
    )
    assert invalid["pass"] is False
    assert invalid["invalid"] == ["../outside.json", "/tmp/outside.json"]


def test_c7_current_preparation_packet_artifacts_are_hash_current() -> None:
    root = Path(__file__).resolve().parents[1]
    packet = json.loads(
        (
            root
            / "_artifacts/tensor_world_c0_c7_v3/c7/"
            "tensor_world_C7_qualification_freeze_contract.json"
        ).read_text(encoding="utf-8")
    )
    binding = packet["current_preparation_binding"]
    result = c7_verify_artifact_bindings_v1(
        root,
        {
            **binding["artifact_bindings"],
            **binding["implementation_bindings"],
        },
    )
    assert result["pass"] is True
    assert result["missing"] == []
    assert result["invalid"] == []
    assert result["mismatches"] == []


def test_c7_receipt_serialization_is_canonical_and_complete() -> None:
    freeze = _freeze()
    receipt = _evidence(freeze)
    payload = receipt.to_dict()
    assert tuple(payload) == (
        "schema_version",
        "freeze_fingerprint",
        "semantic_parity",
        "determinism_parity",
        "checkpoint_replay_parity",
        "external_contract",
        "residency",
        "resource_safe",
        "material_performance",
        "evidence_bindings",
    )
    assert tuple(payload["evidence_bindings"]) == tuple(sorted(C7_EVIDENCE_BINDING_KEYS_V1))
    assert payload["freeze_fingerprint"] == freeze.fingerprint()


def test_c7_ladder_and_decision_surface_are_unique_and_ordered() -> None:
    assert len(C7_QUALIFICATION_LADDER_V1) == 13
    assert len(set(C7_QUALIFICATION_LADDER_V1)) == len(C7_QUALIFICATION_LADDER_V1)
    assert C7_QUALIFICATION_LADDER_V1[0].startswith("Q0_")
    assert C7_QUALIFICATION_LADDER_V1[-1].startswith("Q12_")
    assert len(C7_DECISION_SURFACES_V1) == 7
    assert len(set(C7_DECISION_SURFACES_V1)) == len(C7_DECISION_SURFACES_V1)
