from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.p12_final_package_candidate_binding_v1 import (
    CONTRACT_ID,
    DECISION,
    P12FinalPackageCandidateBindingError,
    build_p12_final_package_candidate_binding_v1,
    canonical_p12_final_package_candidate_binding_bytes_v1,
    p12_final_package_candidate_identity_v1,
    p12_package_candidate_invalidation_decision_v1,
)


ROOT = Path(__file__).resolve().parents[1]
PERSISTED_RECEIPT_PATH = (
    ROOT
    / "evidences/P12_SUBSTRATE_GATES_20260727/"
    "p12_final_package_candidate_binding_v1.json"
)


def test_final_package_candidate_binds_current_p12_mechanic_receipts() -> None:
    receipt = build_p12_final_package_candidate_binding_v1(source_root=ROOT)
    bindings = receipt["mechanic_bindings"]

    assert receipt["contract_id"] == CONTRACT_ID
    assert receipt["decision"] == DECISION
    assert receipt["provider_id"] == "eiel-production-provider-v1"
    assert receipt["runtime_profile_id"] == "eiel-production-runtime-v1"
    assert receipt["genome_v2_sole_production_reproduction_authority"] is True
    assert receipt["linked_parent_copy_v1_retired"] is True
    assert receipt["native_cpu_low_parallelism_route_preserved"] is True
    assert receipt["cuda_batched_route_preserved"] is True
    assert receipt["final_package_candidate_complete"] is True
    assert receipt["final_p12_package_qualified"] is True
    assert receipt["p12_a_qualified"] is True
    assert receipt["route_package_qualification_authorized"] is True
    assert receipt["successor_campaign_authorized"] is False
    assert receipt["cuda_immutable_artifact_binding_status"] == (
        "bound_guarded_final_artifact"
    )
    assert receipt["final_route_package_qualification_status"] == "route_qualified"

    assert bindings["native_cpu_gate1"]["qualified"] is True
    assert bindings["native_cpu_gate1"]["route"] == "native_cpu_low_parallelism"
    assert bindings["dense_id_route"]["qualified"] is True
    assert bindings["cuda_receipt_surface_admission"]["admitted"] is True
    assert bindings["cuda_rowv1_disabled_event_writer"]["sanitizer_passed"] is True
    assert bindings["cuda_rowv1_compact_cpu_cuda_parity"]["complete"] is True
    assert bindings["final_cuda_artifact_route"]["ready_for_execution"] is True
    assert receipt["native_cpu_artifact_sha256"]
    assert receipt["native_cpu_build_manifest_sha256"]
    assert receipt["cuda_artifact_sha256"] == (
        "2120169ff863c9317f305bfc1010ca51ac3fa8b91a35e1f5f4742e9f6de91e35"
    )
    assert receipt["genome_v2_package_id_sha256"] == (
        "03b5c84f8ad25fe19d9a118f220417937f51b7b136472b9a792e73304fde16cf"
    )
    assert receipt["cuda_execution_route_sha256"] == (
        "08892b2eec4d45e83d747552042fcb0f494f19362ec0d54154ecda52a3f7f27c"
    )
    assert len(receipt["mechanic_binding_set_sha256"]) == 64
    assert len(receipt["package_candidate_identity_sha256"]) == 64


def test_final_package_candidate_identity_invalidates_on_bound_hash_change() -> None:
    baseline = build_p12_final_package_candidate_binding_v1(source_root=ROOT)
    candidate = {
        **baseline,
        "cuda_rowv1_compact_cpu_cuda_parity_sha256": "f" * 64,
    }

    decision = p12_package_candidate_invalidation_decision_v1(
        baseline,
        candidate,
    )

    assert decision["reusable"] is False
    assert decision["action"] == "requalify"
    assert decision["changed_fields"] == [
        "cuda_rowv1_compact_cpu_cuda_parity_sha256"
    ]
    assert (
        p12_package_candidate_invalidation_decision_v1(baseline, {**baseline})[
            "reusable"
        ]
        is True
    )


def test_final_package_candidate_identity_rejects_missing_bound_fields() -> None:
    receipt = build_p12_final_package_candidate_binding_v1(source_root=ROOT)
    receipt.pop("rowv1_schema_sha256")

    with pytest.raises(
        P12FinalPackageCandidateBindingError,
        match="missing fields: rowv1_schema_sha256",
    ):
        p12_final_package_candidate_identity_v1(receipt)


def test_final_package_candidate_serialization_is_canonical() -> None:
    receipt = build_p12_final_package_candidate_binding_v1(source_root=ROOT)
    supplied = receipt.pop("contract_sha256")

    assert content_sha256(receipt) == supplied
    receipt["contract_sha256"] = supplied
    wire = canonical_p12_final_package_candidate_binding_bytes_v1(receipt)
    assert wire.endswith(b"\n")
    assert json.loads(wire.decode("ascii")) == receipt


def test_persisted_final_package_candidate_reconciles_to_live_source() -> None:
    persisted = json.loads(PERSISTED_RECEIPT_PATH.read_text(encoding="utf-8"))
    regenerated = build_p12_final_package_candidate_binding_v1(
        source_root=ROOT,
        source_commit=persisted["source_commit"],
    )

    assert persisted == regenerated
    assert persisted["final_package_candidate_complete"] is True
    assert persisted["final_p12_package_qualified"] is True


def test_final_package_candidate_fails_closed_when_source_fragment_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "engine/evolution").mkdir(parents=True)
    (source / "engine/evolution/genome_upgrade_v2_package_v1.py").write_text(
        "fragment absent\n",
        encoding="utf-8",
    )

    with pytest.raises(
        P12FinalPackageCandidateBindingError,
        match=(
            "evidence fragment is absent|missing P12 package-candidate evidence|"
            "dense-ID qualification evidence fragment is absent"
        ),
    ):
        build_p12_final_package_candidate_binding_v1(source_root=source)
