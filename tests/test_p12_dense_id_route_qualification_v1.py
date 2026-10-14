from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.authoritative_state.schema import content_sha256
from engine.evolution.p12_dense_id_route_qualification_v1 import (
    DECISION,
    STATUS,
    P12DenseIDRouteQualificationError,
    build_p12_dense_id_route_qualification_v1,
    canonical_p12_dense_id_route_qualification_bytes_v1,
)


ROOT = Path(__file__).resolve().parents[1]


def _groups(receipt: dict) -> dict[str, dict]:
    return {group["group_id"]: group for group in receipt["evidence_groups"]}


def test_dense_id_route_qualification_receipt_binds_qualified_dense_route() -> None:
    receipt = build_p12_dense_id_route_qualification_v1(
        source_root=ROOT,
        source_commit="live-test",
    )
    groups = _groups(receipt)

    assert receipt["decision"] == DECISION
    assert receipt["status"] == STATUS
    assert receipt["dense_id_scaling_final_qualified"] is True
    assert receipt["dense_id_route_package_qualified"] is True
    assert receipt["route_envelope_contract_id"] == "p12-dense-id-route-envelope-v1"
    assert receipt["route_envelope"]["ragged_multicell"] is True
    assert receipt["route_envelope"]["does_not_scale_with"] == (
        "max(stable_biological_id)+1"
    )
    assert receipt["native_high_id_checkpoint_evidence_present"] is True
    assert receipt["cuda_current_route_dense_lookup_reads"] == []
    assert receipt["cuda_shadow_retired"] is False
    assert receipt["cuda_shadow_authoritative"] is False
    assert receipt["cuda_shadow_non_authority_admitted"] is True
    assert receipt["final_package_route_qualified"] is False
    assert receipt["successor_campaign_authorized"] is False
    assert groups["native_high_id_checkpoint_resume"]["status"] == "route_qualified"
    assert groups["wdl_capacity_contract"]["status"] == "route_qualified"
    assert groups["eed_respec_slot_token_cursors"]["status"] == (
        "route_qualified_reference_boundary"
    )
    assert groups["cuda_shadow_disposition"]["status"] == (
        "quarantined_non_authoritative_admitted"
    )
    assert groups["ragged_multicell_and_sentinel_bounds"]["status"] == (
        "route_qualified"
    )
    assert receipt["wdl_capacity_contract_sha256"]
    assert receipt["eed_cursor_contract_sha256"]
    assert receipt["respec_cursor_contract_sha256"]
    assert receipt["cuda_shadow_audit_sha256"]
    assert receipt["route_envelope_contract_sha256"]
    assert receipt["cuda_legacy_dense_lookup_debt_files"]


def test_dense_id_route_qualification_receipt_binds_source_fragments() -> None:
    receipt = build_p12_dense_id_route_qualification_v1(source_root=ROOT)

    for group in receipt["evidence_groups"]:
        assert group["evidence"]
        for evidence in group["evidence"]:
            assert evidence["sha256"]
            assert evidence["fragment_sha256"]
            assert evidence["line"] > 0


def test_dense_id_route_qualification_serialization_is_canonical() -> None:
    receipt = build_p12_dense_id_route_qualification_v1(source_root=ROOT)
    supplied = receipt.pop("receipt_sha256")

    assert content_sha256(receipt) == supplied
    receipt["receipt_sha256"] = supplied
    wire = canonical_p12_dense_id_route_qualification_bytes_v1(receipt)

    assert wire.endswith(b"\n")
    assert json.loads(wire.decode("ascii")) == receipt
    assert canonical_p12_dense_id_route_qualification_bytes_v1(
        json.loads(wire.decode("ascii"))
    ) == wire


def test_dense_id_route_qualification_fails_closed_when_fragment_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for group in build_p12_dense_id_route_qualification_v1(source_root=ROOT)[
        "evidence_groups"
    ]:
        for row in group["evidence"]:
            path = source / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("sentinel absent\n", encoding="utf-8")

    with pytest.raises(
        P12DenseIDRouteQualificationError,
        match="qualification evidence fragment is absent",
    ):
        build_p12_dense_id_route_qualification_v1(source_root=source)
