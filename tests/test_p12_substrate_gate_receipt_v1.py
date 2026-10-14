from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.p12_substrate_gate_receipt_v1 import (
    DECISION,
    GATE_CLOSED,
    GATE_OPEN,
    P12SubstrateGateReceiptError,
    build_p12_substrate_gate_receipt_v1,
    canonical_p12_substrate_gate_receipt_bytes_v1,
)


ROOT = Path(__file__).resolve().parents[1]
PERSISTED_RECEIPT_PATH = (
    ROOT
    / "evidences/P12_SUBSTRATE_GATES_20260727/p12_substrate_gate_receipt_v1.json"
)
PERSISTED_REPORT_PATH = (
    ROOT / "docs/500_campaigns-decisions-closeouts/520_campaign-reports/000_programme-cross-cutting/EIEL-DOC-000597--eiel-p12-substrate-gate-receipt-2026-07-27.md"
)


def _gates(receipt: dict) -> dict[str, dict]:
    return {gate["gate_id"]: gate for gate in receipt["gates"]}


def test_receipt_binds_current_open_p12_gates() -> None:
    receipt = build_p12_substrate_gate_receipt_v1(
        source_root=ROOT,
        source_commit="2ee4450446363d5c7465893645f392e1ee4d1b6a",
    )
    gates = _gates(receipt)

    assert receipt["decision"] == DECISION
    assert receipt["all_gates_closed"] is False
    assert receipt["provider_id"] == "eiel-production-provider-v1"
    assert receipt["runtime_profile_id"] == "eiel-production-runtime-v1"
    assert gates["ordinary_birth_structural_recurrent_mutation_parity"][
        "status"
    ] == GATE_OPEN
    assert gates["dense_historical_biological_id_scaling"]["status"] == GATE_CLOSED
    assert receipt["dense_id_scaling_final_qualified"] is True
    assert gates["current_python_native_cuda_route_qualification"][
        "status"
    ] == GATE_OPEN
    assert "ordinary births disabled" in gates[
        "current_python_native_cuda_route_qualification"
    ]["blocker_summary"]
    assert all(
        evidence["sha256"] and evidence["fragment_sha256"] and evidence["line"] > 0
        for gate in receipt["gates"]
        for evidence in gate["current_evidence"]
    )


def test_receipt_hash_and_wire_are_canonical() -> None:
    receipt = build_p12_substrate_gate_receipt_v1(source_root=ROOT)
    supplied = receipt.pop("receipt_sha256")

    assert content_sha256(receipt) == supplied
    receipt["receipt_sha256"] = supplied
    wire = canonical_p12_substrate_gate_receipt_bytes_v1(receipt)
    assert wire.endswith(b"\n")
    assert json.loads(wire.decode("ascii")) == receipt
    assert canonical_p12_substrate_gate_receipt_bytes_v1(
        json.loads(wire.decode("ascii"))
    ) == wire


def test_persisted_receipt_reconciles_to_live_source_fragments() -> None:
    persisted = json.loads(PERSISTED_RECEIPT_PATH.read_text(encoding="utf-8"))
    regenerated = build_p12_substrate_gate_receipt_v1(
        source_root=ROOT,
        source_commit=persisted["source_commit"],
    )

    assert persisted == regenerated
    assert persisted["all_gates_closed"] is False
    assert {gate["status"] for gate in persisted["gates"]} == {
        GATE_CLOSED,
        GATE_OPEN,
    }


def test_persisted_receipt_report_matches_json_identity() -> None:
    receipt = json.loads(PERSISTED_RECEIPT_PATH.read_text(encoding="utf-8"))
    report = PERSISTED_REPORT_PATH.read_text(encoding="utf-8")

    assert f"Source commit: `{receipt['source_commit']}`" in report
    assert f"Decision: `{receipt['decision']}`" in report
    assert f"Receipt SHA-256: `{receipt['receipt_sha256']}`" in report


def test_receipt_fails_closed_when_source_sentinel_is_missing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for evidence in build_p12_substrate_gate_receipt_v1(source_root=ROOT)["gates"]:
        for row in evidence["current_evidence"]:
            path = source / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("sentinel absent\n", encoding="utf-8")

    with pytest.raises(
        P12SubstrateGateReceiptError,
        match="source evidence fragment is absent",
    ):
        build_p12_substrate_gate_receipt_v1(source_root=source)
