from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.p12_integrated_final_closure_decision_v1 import (
    DECISION,
    P12IntegratedFinalClosureDecisionError,
    build_p12_integrated_final_closure_decision_v1,
    canonical_p12_integrated_final_closure_decision_bytes_v1,
)


ROOT = Path(__file__).resolve().parents[1]
PERSISTED_DECISION_PATH = (
    ROOT
    / "evidences/P12_SUBSTRATE_GATES_20260727/"
    "p12_integrated_final_closure_decision_v1.json"
)


def _closed_gates(decision: dict) -> dict[str, dict]:
    return {row["gate"]: row for row in decision["closed_gates"]}


def test_integrated_final_closure_decision_is_p12_a() -> None:
    decision = build_p12_integrated_final_closure_decision_v1(source_root=ROOT)
    closed_gates = _closed_gates(decision)

    assert decision["decision"] == DECISION
    assert decision["decision"] == "P12-A"
    assert decision["architecture_sound"] is True
    assert decision["p12_a_qualified"] is True
    assert decision["linked_parent_copy_v1_retired"] is True
    assert decision["genome_v2_sole_production_reproduction_authority"] is True
    assert decision["successor_campaign_started"] is False
    assert decision["blockers"] == []
    assert (
        decision["structural_payload_sha_progress_contract_id"]
        == "p12-native-core-ordinary-birth-structural-payload-sha-v1"
    )
    assert decision["structural_payload_sha_progress_sha256"]
    assert (
        decision["route_bound_birth_receipt_contract_id"]
        == "p12-native-cpu-route-bound-birth-lifecycle-receipt-v1"
    )
    assert decision["route_bound_birth_receipt_sha256"]
    assert (
        decision["native_gate1_contract_id"]
        == "p12-native-cpu-gate1-turnover-route-package-qualification-v1"
    )
    assert decision["native_gate1_sha256"]
    assert decision["native_gate1_qualified"] is True
    assert (
        decision["final_package_candidate_contract_id"]
        == "p12-final-package-candidate-binding-v1"
    )
    assert decision["final_package_candidate_sha256"]
    assert decision["final_package_candidate_identity_sha256"]
    assert decision["final_package_candidate_complete"] is True
    assert decision["final_p12_package_qualified"] is True
    assert decision["cuda_artifact_sha256"] == (
        "2120169ff863c9317f305bfc1010ca51ac3fa8b91a35e1f5f4742e9f6de91e35"
    )
    assert decision["genome_v2_package_id_sha256"] == (
        "03b5c84f8ad25fe19d9a118f220417937f51b7b136472b9a792e73304fde16cf"
    )
    assert decision["cuda_execution_route_sha256"] == (
        "08892b2eec4d45e83d747552042fcb0f494f19362ec0d54154ecda52a3f7f27c"
    )
    assert decision["cuda_execution_route_receipt_sha256"] == (
        "06b4a76fa74a09069fed4e69170c60ad00f72c55a594bed15ea4b81076770dd8"
    )
    assert closed_gates["ordinary_birth_structural_mutation"]["status"] == (
        "closed_native_gate1_route_package_qualified"
    )
    structural_conclusion = closed_gates["ordinary_birth_structural_mutation"][
        "conclusion"
    ]
    assert "route-package qualified" in structural_conclusion
    assert "native-generated RowV1 structural payload SHA identity" in (
        structural_conclusion
    )
    assert "generation-turnover checkpoint/resume repeatability" in (
        structural_conclusion
    )
    assert "build-manifest identity" in (
        structural_conclusion
    )
    assert closed_gates["ordinary_birth_recurrent_mutation"]["status"] == (
        "closed_native_gate1_route_package_qualified"
    )
    assert (
        "native-built canonical recurrent receipt digest construction"
        in closed_gates["ordinary_birth_recurrent_mutation"]["conclusion"]
    )
    assert "artifact-bound multicell checkpoint restore" in closed_gates[
        "ordinary_birth_recurrent_mutation"
    ]["conclusion"]
    assert closed_gates["cuda_receipt_authority"]["status"] == (
        "closed_admitted_rowv1_abi_and_compact_parity"
    )
    cuda_conclusion = closed_gates["cuda_receipt_authority"]["conclusion"]
    assert "bounded CUDA RowV1 device-writer ABI is admitted" in cuda_conclusion
    assert "compact CPU/CUDA RowV1 ABI parity is complete" in cuda_conclusion
    assert closed_gates["dense_historical_biological_id_scaling"]["status"] == (
        "closed_route_qualified"
    )
    assert "Dense historical-ID scaling is route-qualified" in closed_gates[
        "dense_historical_biological_id_scaling"
    ]["conclusion"]
    assert closed_gates["final_package_route_qualification"]["status"] == (
        "closed_final_package_and_route_qualified"
    )
    package_conclusion = closed_gates["final_package_route_qualification"][
        "conclusion"
    ]
    assert "final P12 package now binds" in package_conclusion
    assert "final guarded WDL CUDA artifact" in package_conclusion


def test_integrated_final_closure_decision_binds_source_fragments() -> None:
    decision = build_p12_integrated_final_closure_decision_v1(source_root=ROOT)

    for gate in decision["closed_gates"]:
        assert gate["evidence"]
        for evidence in gate["evidence"]:
            assert evidence["sha256"]
            assert evidence["fragment_sha256"]
            assert evidence["line"] > 0


def test_integrated_final_closure_decision_serialization_is_canonical() -> None:
    decision = build_p12_integrated_final_closure_decision_v1(source_root=ROOT)
    supplied = decision.pop("decision_sha256")

    assert content_sha256(decision) == supplied
    decision["decision_sha256"] = supplied
    wire = canonical_p12_integrated_final_closure_decision_bytes_v1(decision)
    assert wire.endswith(b"\n")
    assert json.loads(wire.decode("ascii")) == decision


def test_integrated_final_closure_decision_fails_closed_when_fragment_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "engine/evolution").mkdir(parents=True)
    (source / "engine/evolution/p12_stage2_mutation_oracles_v1.py").write_text(
        "fragment absent\n",
        encoding="utf-8",
    )

    with pytest.raises(
        P12IntegratedFinalClosureDecisionError,
        match=(
            "evidence fragment is absent|missing P12 final-closure evidence|"
            "missing P1.2 source evidence"
        ),
    ):
        build_p12_integrated_final_closure_decision_v1(source_root=source)
