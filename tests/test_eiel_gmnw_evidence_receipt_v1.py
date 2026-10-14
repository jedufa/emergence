from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.perf.gmnw_evidence_receipt_v1 import (
    GmnwEvidenceReceiptError,
    consistency_receipt,
    load_contract,
    replay_boundary_receipt,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/300_architecture-specifications/370_specifications/EIEL-DOC-000900--gmnw-authoritative-surface-contract.json"


def _source(*, trajectory: bool = True, canonical: bool = True) -> dict[str, object]:
    return {
        "ok": False,
        "comparison": {
            "all_authoritative_exact": False,
            "exact": {
                "trajectory": trajectory,
                "canonical_storage": canonical,
                "source_projection": False,
            },
        },
        "gmnw_sidecar_exact": True,
        "runtime_jit": False,
        "hidden_fallback": False,
    }


def _replay(source: dict[str, object]) -> dict[str, object]:
    contract = load_contract(CONTRACT)
    return replay_boundary_receipt(
        source,
        source_path=Path("source.json"),
        source_sha256="a" * 64,
        contract=contract,
        contract_path=CONTRACT,
        contract_sha256=hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
    )


def test_frozen_contract_loads_exact_surface_classification() -> None:
    contract = load_contract(CONTRACT)

    assert contract["surfaces"]["trajectory"]["authoritative"] is True
    assert contract["surfaces"]["canonical_checkpoint"]["authoritative"] is True
    assert contract["surfaces"]["gmnw_sidecar"]["authoritative"] is True
    assert contract["surfaces"]["source_projection"]["authoritative"] is False
    assert contract["surfaces"]["source_projection"]["diagnostic"] is True


def test_diagnostic_provenance_mismatch_does_not_fail_exact_boundary() -> None:
    receipt = _replay(_source())

    assert receipt["authoritative_exact"] is True
    assert receipt["diagnostic_exact"] is False
    assert receipt["first_authoritative_difference"] is None
    assert receipt["first_diagnostic_difference"] == "digest:source_projection"
    assert receipt["ok"] is True
    assert receipt["simulation_rerun"] is False


@pytest.mark.parametrize(
    ("trajectory", "canonical", "difference"),
    (
        (False, True, "digest:trajectory"),
        (True, False, "digest:canonical_checkpoint"),
    ),
)
def test_authoritative_mismatch_fails_closed(
    trajectory: bool, canonical: bool, difference: str
) -> None:
    receipt = _replay(_source(trajectory=trajectory, canonical=canonical))

    assert receipt["authoritative_exact"] is False
    assert receipt["first_authoritative_difference"] == difference
    assert receipt["ok"] is False


def test_sidecar_mismatch_fails_closed() -> None:
    source = _source()
    source["gmnw_sidecar_exact"] = False

    receipt = _replay(source)

    assert receipt["first_authoritative_difference"] == "digest:gmnw_sidecar"
    assert receipt["ok"] is False


def test_runtime_jit_and_hidden_fallback_fail_exact_state() -> None:
    for field in ("runtime_jit", "hidden_fallback"):
        source = _source()
        source[field] = True
        assert _replay(source)["ok"] is False


def test_contract_rejects_reclassified_authoritative_surface(tmp_path: Path) -> None:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    value["surfaces"]["trajectory"]["authoritative"] = False
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(GmnwEvidenceReceiptError, match="classification mismatch"):
        load_contract(path)


def test_consistency_receipt_requires_both_boundaries_exact() -> None:
    exact = _replay(_source())
    failed = _replay(_source(canonical=False))

    accepted = consistency_receipt(
        [exact, exact], contract_path=CONTRACT, contract_sha256="b" * 64
    )
    rejected = consistency_receipt(
        [exact, failed], contract_path=CONTRACT, contract_sha256="b" * 64
    )

    assert accepted["ok"] is True
    assert accepted["decision"].startswith("GMNW-EVIDENCE-A/")
    assert accepted["authoritative_exact"] is True
    assert accepted["diagnostic_exact"] is False
    assert accepted["source_projection_authoritative"] is False
    assert accepted["source_provenance_diagnostic"] is True
    assert rejected["ok"] is False
    assert rejected["first_authoritative_difference"] == (
        "digest:canonical_checkpoint"
    )
