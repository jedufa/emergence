from __future__ import annotations

from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.genome_upgrade_v2_p10_completion_gate_v1 import (
    DECISION_A,
    DECISION_B,
    DECISION_C,
    DEFERRED_SCOPE,
    EVIDENCE_SCHEMA_VERSION,
    GATE_IDS,
    OUTPUT_NAMES,
    PACKAGE_CAPABILITIES,
    PREREGISTRATION_SHA256,
    WORK_PACKAGES,
    reduce_p10_completion_v1,
)
from tools.scripts.genome_upgrade_v2_p10_completion_gate_v1 import materialize


def _authority() -> dict[str, object]:
    return {
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "source_commit": "1" * 40,
        "package_id_sha256": "2" * 64,
        "package_receipt_sha256": "3" * 64,
        "artifact_sha256": "4" * 64,
        "execution_route_sha256s": {
            "patch": "5" * 64,
            "wdl": "6" * 64,
        },
        "c1_decision_sha256": "7" * 64,
        "e1_decision_sha256": "8" * 64,
    }


def _gate(gate_id: str, *, decision: str) -> dict[str, object]:
    authority = _authority()
    decision_sha256 = {
        "c1": authority["c1_decision_sha256"],
        "e1": authority["e1_decision_sha256"],
    }.get(gate_id, content_sha256({"gate_id": gate_id, "decision": decision}))
    value = {
        "schema_version": "genome_v2_p10_completion_gate_receipt_v1",
        "gate_id": gate_id,
        "decision": decision,
        "decision_sha256": decision_sha256,
        "source_commit": authority["source_commit"],
        "package_id_sha256": authority["package_id_sha256"],
        "package_receipt_sha256": authority["package_receipt_sha256"],
        "artifact_sha256": authority["artifact_sha256"],
        "execution_route_sha256s": list(
            authority["execution_route_sha256s"].values()
        ),
    }
    return {**value, "receipt_sha256": content_sha256(value)}


def _evidence(*, e1: str = "E1-A") -> dict[str, object]:
    decisions = {
        "technical": "TECHNICAL-A",
        "c1": "C1-A",
        "e1": e1,
        "hereditary_capacity": "HEREDITARY-CAPACITY-A",
        "environment_switch": "ENVIRONMENT-SWITCH-A",
    }
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "authority": _authority(),
        "gate_receipts": {
            gate: _gate(gate, decision=decisions[gate])
            for gate in GATE_IDS
        },
        "work_packages": [
            {
                "work_package": package,
                "mandatory_capabilities": list(PACKAGE_CAPABILITIES[package]),
                "status": "implemented_and_qualified",
                "evidence_sha256": content_sha256(
                    {"work_package": package, "index": index}
                ),
                "disposition": "accepted exact evidence",
            }
            for index, package in enumerate(WORK_PACKAGES)
        ],
    }


def test_completion_emits_exact_artifacts_and_terminal_decisions() -> None:
    positive = reduce_p10_completion_v1(_evidence())
    assert set(positive) == set(OUTPUT_NAMES.values())
    assert positive[OUTPUT_NAMES["audit"]]["decision"] == DECISION_A
    assert {
        row["id"] for row in positive[OUTPUT_NAMES["deferred"]]["items"]
    } == {row["id"] for row in DEFERRED_SCOPE}
    assert all(
        row["status"] == "deferred_by_plan"
        for row in positive[OUTPUT_NAMES["deferred"]]["items"]
    )

    negative = reduce_p10_completion_v1(_evidence(e1="E1-B"))
    assert negative[OUTPUT_NAMES["audit"]]["decision"] == DECISION_B


def test_completion_rejects_mixed_identity_and_requires_pleiotropy() -> None:
    mixed = _evidence()
    mixed["gate_receipts"]["c1"]["source_commit"] = "f" * 40
    unsigned = dict(mixed["gate_receipts"]["c1"])
    unsigned.pop("receipt_sha256")
    mixed["gate_receipts"]["c1"]["receipt_sha256"] = content_sha256(unsigned)
    with pytest.raises(ValueError, match="mixed"):
        reduce_p10_completion_v1(mixed)

    mixed_route = _evidence()
    mixed_route["gate_receipts"]["e1"]["execution_route_sha256s"] = [
        "f" * 64
    ]
    unsigned = dict(mixed_route["gate_receipts"]["e1"])
    unsigned.pop("receipt_sha256")
    mixed_route["gate_receipts"]["e1"]["receipt_sha256"] = content_sha256(
        unsigned
    )
    with pytest.raises(ValueError, match="mixed"):
        reduce_p10_completion_v1(mixed_route)

    incomplete = _evidence()
    incomplete["work_packages"][-1]["status"] = "deferred_by_plan"
    result = reduce_p10_completion_v1(incomplete)
    assert result[OUTPUT_NAMES["audit"]]["decision"] == DECISION_C
    assert "bounded_pleiotropy_is_mandatory" in result[OUTPUT_NAMES["audit"]]["reasons"]


def test_completion_cli_writes_only_five_named_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "completion"
    result = materialize(evidence=_evidence(), output_directory=output)
    assert result["decision"] == DECISION_A
    assert {path.name for path in output.iterdir()} == set(OUTPUT_NAMES.values())
    with pytest.raises(ValueError, match="already exists"):
        materialize(evidence=_evidence(), output_directory=output)
