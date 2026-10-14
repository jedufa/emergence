"""Strict replay analyzer for GMNW fresh-owner boundary receipts."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any, Final


CONTRACT_SCHEMA_VERSION: Final[str] = (
    "eiel_gmnw_authoritative_surface_contract_v1"
)
RECEIPT_SCHEMA_VERSION: Final[str] = "eiel_gmnw_boundary_evidence_receipt_v2"
DECISION_SCHEMA_VERSION: Final[str] = "eiel_gmnw_evidence_consistency_receipt_v1"
CONTRACT_ID: Final[str] = "gmnw_fresh_owner_boundary_v1"

_EXPECTED_SURFACES: Final[dict[str, dict[str, Any]]] = {
    "trajectory": {
        "authoritative": True,
        "diagnostic": False,
        "comparison_key": "trajectory",
    },
    "canonical_checkpoint": {
        "authoritative": True,
        "diagnostic": False,
        "comparison_key": "canonical_storage",
    },
    "gmnw_sidecar": {
        "authoritative": True,
        "diagnostic": False,
        "comparison_key": "gmnw_sidecar_exact",
    },
    "source_projection": {
        "authoritative": False,
        "diagnostic": True,
        "comparison_key": "source_projection",
    },
}


class GmnwEvidenceReceiptError(ValueError):
    """A GMNW authority contract or source receipt is malformed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GmnwEvidenceReceiptError(f"cannot load GMNW contract: {path}") from exc
    if not isinstance(value, dict):
        raise GmnwEvidenceReceiptError("GMNW contract must be an object")
    expected_keys = {"schema_version", "contract_id", "decision_rule", "surfaces"}
    if set(value) != expected_keys:
        raise GmnwEvidenceReceiptError("GMNW contract field set mismatch")
    if value["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise GmnwEvidenceReceiptError("GMNW contract schema mismatch")
    if value["contract_id"] != CONTRACT_ID:
        raise GmnwEvidenceReceiptError("GMNW contract identity mismatch")
    if value["surfaces"] != _EXPECTED_SURFACES:
        raise GmnwEvidenceReceiptError("GMNW authority surface classification mismatch")
    if not isinstance(value["decision_rule"], str) or not value["decision_rule"]:
        raise GmnwEvidenceReceiptError("GMNW decision rule must be non-empty")
    return value


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise GmnwEvidenceReceiptError(f"{name} must be a string-keyed mapping")
    return dict(value)


def _exact_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise GmnwEvidenceReceiptError(f"{name} must be an exact boolean")
    return value


def _first_authoritative_difference(
    *, trajectory_exact: bool, canonical_exact: bool, sidecar_exact: bool
) -> str | None:
    for name, exact in (
        ("trajectory", trajectory_exact),
        ("canonical_checkpoint", canonical_exact),
        ("gmnw_sidecar", sidecar_exact),
    ):
        if not exact:
            return f"digest:{name}"
    return None


def replay_boundary_receipt(
    source: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    contract: Mapping[str, Any],
    contract_path: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    """Re-derive a boundary verdict from frozen authority classifications."""

    checked_contract = dict(contract)
    if checked_contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise GmnwEvidenceReceiptError("unvalidated GMNW contract")
    payload = _mapping(source, name="source receipt")
    comparison = _mapping(payload.get("comparison"), name="comparison")
    exact = _mapping(comparison.get("exact"), name="comparison.exact")
    trajectory_exact = _exact_bool(
        exact.get("trajectory"), name="comparison.exact.trajectory"
    )
    canonical_exact = _exact_bool(
        exact.get("canonical_storage"),
        name="comparison.exact.canonical_storage",
    )
    source_projection_exact = _exact_bool(
        exact.get("source_projection"),
        name="comparison.exact.source_projection",
    )
    sidecar_exact = _exact_bool(
        payload.get("gmnw_sidecar_exact"), name="gmnw_sidecar_exact"
    )
    runtime_jit = _exact_bool(payload.get("runtime_jit"), name="runtime_jit")
    hidden_fallback = _exact_bool(
        payload.get("hidden_fallback"), name="hidden_fallback"
    )
    first_authoritative_difference = _first_authoritative_difference(
        trajectory_exact=trajectory_exact,
        canonical_exact=canonical_exact,
        sidecar_exact=sidecar_exact,
    )
    authoritative_exact = bool(
        trajectory_exact and canonical_exact and sidecar_exact
    )
    diagnostic_exact = source_projection_exact
    ok = bool(
        authoritative_exact
        and first_authoritative_difference is None
        and not runtime_jit
        and not hidden_fallback
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "source_receipt": str(source_path),
        "source_receipt_sha256": source_sha256,
        "source_original_ok": payload.get("ok"),
        "source_original_all_authoritative_exact": comparison.get(
            "all_authoritative_exact"
        ),
        "trajectory_authoritative": True,
        "canonical_checkpoint_authoritative": True,
        "sidecar_authoritative": True,
        "source_projection_authoritative": False,
        "source_provenance_diagnostic": True,
        "trajectory_exact": trajectory_exact,
        "canonical_checkpoint_exact": canonical_exact,
        "sidecar_exact": sidecar_exact,
        "source_projection_exact": source_projection_exact,
        "authoritative_exact": authoritative_exact,
        "diagnostic_exact": diagnostic_exact,
        "first_authoritative_difference": first_authoritative_difference,
        "first_diagnostic_difference": (
            None if diagnostic_exact else "digest:source_projection"
        ),
        "runtime_jit": runtime_jit,
        "hidden_fallback": hidden_fallback,
        "simulation_rerun": False,
        "ok": ok,
        "decision": (
            "GMNW-BOUNDARY-A/AUTHORITATIVE-EXACT"
            if ok
            else "GMNW-BOUNDARY-C/FIRST-AUTHORITATIVE-DIFFERENCE"
        ),
    }
    receipt["receipt_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    return receipt


def consistency_receipt(
    receipts: list[Mapping[str, Any]],
    *,
    contract_path: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    if len(receipts) != 2:
        raise GmnwEvidenceReceiptError("consistency replay requires exactly two boundaries")
    rows = [dict(receipt) for receipt in receipts]
    for row in rows:
        if row.get("schema_version") != RECEIPT_SCHEMA_VERSION:
            raise GmnwEvidenceReceiptError("boundary replay schema mismatch")
    ok = all(row.get("ok") is True for row in rows)
    authoritative_exact = all(
        row.get("authoritative_exact") is True for row in rows
    )
    diagnostic_exact = all(row.get("diagnostic_exact") is True for row in rows)
    first_difference = next(
        (
            row.get("first_authoritative_difference")
            for row in rows
            if row.get("first_authoritative_difference") is not None
        ),
        None,
    )
    receipt = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "trajectory_authoritative": True,
        "canonical_checkpoint_authoritative": True,
        "sidecar_authoritative": True,
        "source_projection_authoritative": False,
        "source_provenance_diagnostic": True,
        "authoritative_exact": authoritative_exact,
        "diagnostic_exact": diagnostic_exact,
        "boundary_receipts": [
            {
                "source_receipt": row["source_receipt"],
                "receipt_sha256": row["receipt_sha256"],
                "authoritative_exact": row["authoritative_exact"],
                "diagnostic_exact": row["diagnostic_exact"],
                "ok": row["ok"],
            }
            for row in rows
        ],
        "first_authoritative_difference": first_difference,
        "first_diagnostic_difference": next(
            (
                row.get("first_diagnostic_difference")
                for row in rows
                if row.get("first_diagnostic_difference") is not None
            ),
            None,
        ),
        "simulation_rerun": False,
        "ok": ok,
        "decision": (
            "GMNW-EVIDENCE-A/AUTHORITATIVE-AND-DIAGNOSTIC-SURFACES-CONSISTENT"
            if ok
            else "GMNW-EVIDENCE-C/FIRST-AUTHORITATIVE-DIFFERENCE"
        ),
    }
    receipt["receipt_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    return receipt


__all__ = [
    "CONTRACT_ID",
    "CONTRACT_SCHEMA_VERSION",
    "DECISION_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "GmnwEvidenceReceiptError",
    "consistency_receipt",
    "file_sha256",
    "load_contract",
    "replay_boundary_receipt",
]
