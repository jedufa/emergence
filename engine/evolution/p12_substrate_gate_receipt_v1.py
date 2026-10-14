"""Deterministic P1.2 substrate gate receipt.

This module records the current engineering gate state before any fresh
Genome V2 RECOMB-72 successor can be honestly preregistered as qualified.
It is deliberately conservative: static source evidence may prove a gate is
still open, but it cannot promote a gate to closed without route-bound parity
evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

from engine.authoritative_state.schema import content_sha256
from engine.evolution.p12_dense_id_route_qualification_v1 import (
    P12DenseIDRouteQualificationError,
    build_p12_dense_id_route_qualification_v1,
)


SCHEMA_VERSION: Final[str] = "p12_substrate_gate_receipt_v1"
DECISION: Final[str] = "P12-SUBSTRATE-GATES-OPEN"
GATE_OPEN: Final[str] = "open"
GATE_PARTIAL: Final[str] = "partial_not_qualified"
GATE_CLOSED: Final[str] = "closed"


class P12SubstrateGateReceiptError(RuntimeError):
    """The receipt could not bind the expected current source evidence."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_evidence(source_root: Path, relative_path: str, fragment: str) -> dict[str, Any]:
    path = source_root / relative_path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise P12SubstrateGateReceiptError(
            f"missing P1.2 source evidence: {relative_path}"
        ) from exc
    if fragment not in text:
        raise P12SubstrateGateReceiptError(
            f"P1.2 source evidence fragment is absent: {relative_path}"
        )
    line = text[: text.index(fragment)].count("\n") + 1
    return {
        "path": relative_path,
        "sha256": _file_sha256(path),
        "line": line,
        "fragment_sha256": content_sha256(
            {"path": relative_path, "fragment": fragment}
        ),
    }


def _gate_receipt(
    *,
    gate_id: str,
    status: str,
    required_evidence: tuple[str, ...],
    current_evidence: tuple[dict[str, Any], ...],
    blocker_summary: str,
) -> dict[str, Any]:
    return {
        "gate_id": gate_id,
        "status": status,
        "required_evidence": list(required_evidence),
        "current_evidence": list(current_evidence),
        "blocker_summary": blocker_summary,
    }


def build_p12_substrate_gate_receipt_v1(
    *,
    source_root: Path | str = ".",
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Return a hash-bound receipt for the remaining P1.2 substrate gates."""

    root = Path(source_root).resolve()
    try:
        dense_id_progress = build_p12_dense_id_route_qualification_v1(
            source_root=root,
            source_commit=source_commit,
        )
    except P12DenseIDRouteQualificationError as exc:
        raise P12SubstrateGateReceiptError(
            f"P1.2 source evidence fragment is absent: {exc}"
        ) from exc
    gates = [
        _gate_receipt(
            gate_id="ordinary_birth_structural_recurrent_mutation_parity",
            status=GATE_OPEN,
            required_evidence=(
                "Python/reference ordinary-birth structural and recurrent mutation oracle",
                "native CPU ordinary-birth structural and recurrent mutation parity",
                "CUDA ordinary-birth structural and recurrent mutation parity",
                "RNG draw-budget, rollback, checkpoint, Chronicle, and reporting parity",
                "route-bound package obligations admitting the mechanic",
            ),
            current_evidence=(
                _source_evidence(
                    root,
                    "engine/native_runtime/safe_default.py",
                    '"c1_structural_mutation"',
                ),
                _source_evidence(
                    root,
                    "engine/native_runtime/safe_default.py",
                    '"c1_structural_mutation_not_supported"',
                ),
                _source_evidence(
                    root,
                    "engine/perf/native/cxxbackbone10_resident_state_owner.cpp",
                    "generation turnover refill does not support C1/C2 structural genetics",
                ),
                _source_evidence(
                    root,
                    "engine/perf/native/cxxbackbone10_resident_state_owner.cpp",
                    "native_runtime_birth_mutation_sigma",
                ),
                _source_evidence(
                    root,
                    "engine/perf/native/cxxbackbone10_resident_state_owner.cpp",
                    "diploid native birth currently requires the qualified fixed-MLP envelope",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_cuda_architecture_b_birth_microcycle_cpu_parity.py",
                    "patch-p2-policy-mutation",
                ),
            ),
            blocker_summary=(
                "Patch P2 ordinary-birth fixed-MLP policy-weight mutation now "
                "has exact CPU/CUDA microcycle parity evidence, but current "
                "production safety surfaces still reject or narrow the C1/C2 "
                "structural, recurrent-controller, diploid, and route-bound "
                "ordinary-birth mutation envelopes."
            ),
        ),
        _gate_receipt(
            gate_id="dense_historical_biological_id_scaling",
            status=GATE_CLOSED,
            required_evidence=(
                "bounded sparse historical-ID representation across live birth, death, checkpoint, and resume",
                "proof that active arrays scale with resident capacity rather than max historical agent ID",
                "route/package evidence covering the admitted Genome V2 campaign profile",
            ),
            current_evidence=(
                _source_evidence(
                    root,
                    "engine/perf/native/cxxbackbone10_resident_state_owner.cpp",
                    "compact identity v2 rejects dense agent_id_to_slot",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_cxxbackbone10_resident_state_owner.py",
                    "test_cxxbackbone10r_high_id_birth_and_death_keep_compact_storage_bounded",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_cxxbackbone10_resident_state_owner.py",
                    "test_high_id_checkpoint_materializes_slot_world_authority_without_dense_growth",
                ),
                _source_evidence(
                    root,
                    "engine/native_runtime/patch_public_reporting.py",
                    "agent_positions_by_id",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_native_patch_public_reporting.py",
                    "test_patch_public_reporting_is_slot_bounded_for_large_historical_ids",
                ),
                _source_evidence(
                    root,
                    "engine/evolution/p12_dense_id_route_qualification_v1.py",
                    "p12-dense-id-route-qualification-v2",
                ),
            ),
            blocker_summary=(
                "High-ID compact identity, native checkpoint, WDL capacity, "
                "EED/RESPEC cursor ownership, ragged multicell sizing, INT64_MAX "
                "sentinel failure, and CUDA compatibility-shadow non-authority "
                "are now bound by a deterministic dense-ID route qualification "
                "receipt. The remaining package/route gate belongs to the "
                "unresolved structural, recurrent, and CUDA receipt mechanics."
            ),
        ),
        _gate_receipt(
            gate_id="current_python_native_cuda_route_qualification",
            status=GATE_OPEN,
            required_evidence=(
                "current Python/reference, native CPU, and CUDA agreement for live births and generation refills",
                "execution-route qualification receipt for the exact Genome V2 successor profile",
                "fresh RECOMB-72 successor preregistration using Genome V2 as sole reproduction authority",
                "single-batch b72 run evidence with G0/G5/G10/G20/G30 rows",
            ),
            current_evidence=(
                _source_evidence(
                    root,
                    "engine/evolution/genome_upgrade_v2_recomb72_gate_v1.py",
                    '"ordinary_births_enabled": False',
                ),
                _source_evidence(
                    root,
                    "tools/scripts/genome_upgrade_v2_recomb72_materialize_spec_v1.py",
                    '"production_lifecycle.ordinary_births_enabled"',
                ),
                _source_evidence(
                    root,
                    "tests/test_genome_upgrade_v2_recomb72_gate_v1.py",
                    'assert payload["campaign_design"]["ordinary_births_enabled"] is False',
                ),
                _source_evidence(
                    root,
                    "engine/evolution/p12_rowv1_cuda_entry_gate_v1.py",
                    "bounded_device_writer_abi_and_compact_parity_complete_but_route_not_package_qualified",
                ),
                _source_evidence(
                    root,
                    "engine/evolution/p12_rowv1_cuda_entry_gate_v1.py",
                    "P12-CUDA-ROWV1-COMPACT-CPU-CUDA-PARITY-A",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_p12_cuda_modular_rowv1_device_writer_v1.cu",
                    "rowv1_device_writer_probe",
                ),
            ),
            blocker_summary=(
                "The current RECOMB-72 Genome V2 preregistration and materializer "
                "are intentionally frozen with ordinary births disabled; they are "
                "not the fresh sole-authority successor requested by P1.2. The "
                "bounded CUDA RowV1 modular device-writer ABI is now admitted, "
                "and compact CPU/CUDA RowV1 ABI parity is complete for accepted "
                "and disabled fixture fields. Immutable artifact binding and "
                "final route/package qualification remain open."
            ),
        ),
    ]
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "source_commit": source_commit,
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "recomb_successor_boundary": (
            "Do not launch the fresh Genome V2 RECOMB-72 successor until all "
            "P1.2 substrate gates have route-bound evidence."
        ),
        "dense_id_route_contract_id": dense_id_progress["contract_id"],
        "dense_id_route_status": dense_id_progress["status"],
        "dense_id_route_receipt_sha256": dense_id_progress["receipt_sha256"],
        "dense_id_scaling_final_qualified": dense_id_progress[
            "dense_id_scaling_final_qualified"
        ],
        "gates": gates,
        "all_gates_closed": False,
    }
    payload["receipt_sha256"] = content_sha256(payload)
    return payload


def canonical_p12_substrate_gate_receipt_bytes_v1(
    receipt: Mapping[str, Any],
) -> bytes:
    """Serialize a receipt with stable JSON ordering and no non-ASCII output."""

    return (
        json.dumps(
            dict(receipt),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
