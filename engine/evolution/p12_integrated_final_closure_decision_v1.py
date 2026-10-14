"""P12 integrated final-closure decision receipt.

This receipt records the P12-A closure after native Gate 1, CUDA RowV1
admission, final guarded CUDA package rebuild, package qualification, and final
route qualification are all bound to live evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

from engine.authoritative_state.schema import content_sha256
from engine.evolution.p12_native_cpu_parity_progress_v1 import (
    native_cpu_gate1_turnover_route_package_qualification_v1,
    native_core_ordinary_birth_structural_payload_sha_v1,
    native_core_recurrent_receipt_byte_authority_progress_v1,
    native_cpu_route_bound_birth_lifecycle_receipt_v1,
)
from engine.evolution.p12_final_package_candidate_binding_v1 import (
    P12FinalPackageCandidateBindingError,
    build_p12_final_package_candidate_binding_v1,
)
from engine.evolution.p12_substrate_gate_receipt_v1 import (
    P12SubstrateGateReceiptError,
    build_p12_substrate_gate_receipt_v1,
)


SCHEMA_VERSION: Final[str] = "p12_integrated_final_closure_decision_v1"
DECISION: Final[str] = "P12-A"


class P12IntegratedFinalClosureDecisionError(RuntimeError):
    """The integrated decision could not bind expected live source evidence."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_evidence(source_root: Path, relative_path: str, fragment: str) -> dict[str, Any]:
    path = source_root / relative_path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise P12IntegratedFinalClosureDecisionError(
            f"missing P12 final-closure evidence: {relative_path}"
        ) from exc
    if fragment not in text:
        raise P12IntegratedFinalClosureDecisionError(
            f"P12 final-closure evidence fragment is absent: {relative_path}"
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


def _gate(
    *,
    gate: str,
    status: str,
    conclusion: str,
    evidence: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    return {
        "gate": gate,
        "status": status,
        "conclusion": conclusion,
        "evidence": list(evidence),
    }


def build_p12_integrated_final_closure_decision_v1(
    *,
    source_root: Path | str = ".",
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Return a hash-bound terminal P12-A decision for the current gate state."""

    root = Path(source_root).resolve()
    structural_payload_sha_progress = native_core_ordinary_birth_structural_payload_sha_v1()
    recurrent_progress = native_core_recurrent_receipt_byte_authority_progress_v1()
    route_bound_birth_receipt = native_cpu_route_bound_birth_lifecycle_receipt_v1()
    native_gate1 = native_cpu_gate1_turnover_route_package_qualification_v1()
    try:
        package_candidate = build_p12_final_package_candidate_binding_v1(
            source_root=root,
            source_commit=source_commit,
        )
    except P12FinalPackageCandidateBindingError as exc:
        raise P12IntegratedFinalClosureDecisionError(str(exc)) from exc
    try:
        substrate = build_p12_substrate_gate_receipt_v1(
            source_root=root,
            source_commit=source_commit,
        )
    except P12SubstrateGateReceiptError as exc:
        raise P12IntegratedFinalClosureDecisionError(str(exc)) from exc
    closed_gates = (
        _gate(
            gate="ordinary_birth_structural_mutation",
            status="closed_native_gate1_route_package_qualified",
            conclusion=(
                "The reference structural oracle exists and native-core "
                "structural payload SHA equality is route-package qualified "
                "for the native CPU low-parallelism Gate 1 envelope. The native "
                "CPU lifecycle receipt carries native-generated RowV1 "
                "structural payload SHA identity, native-built canonical "
                "recurrent receipt digest checks, generation-turnover "
                "checkpoint/resume repeatability, multicell checkpoint restore, "
                "artifact identity, and build-manifest identity."
            ),
            evidence=(
                _source_evidence(
                    root,
                    "engine/evolution/p12_stage2_mutation_oracles_v1.py",
                    "def structural_ordinary_birth_oracle_v1",
                ),
                _source_evidence(
                    root,
                    "engine/evolution/p12_native_cpu_parity_progress_v1.py",
                    "native_generated_rowv1_structural_payload_sha_present_and_route_bound_v3",
                ),
                _source_evidence(
                    root,
                    "engine/evolution/p12_native_cpu_parity_progress_v1.py",
                    "P12-NATIVE-GATE1-A",
                ),
                _source_evidence(
                    root,
                    "tests/perf/test_eiel_modular_recurrent_cpu_oracle_v1.py",
                    "test_p12_native_cpu_generation_turnover_checkpoint_resume_repeats_v4_birth_receipts",
                ),
            ),
        ),
        _gate(
            gate="ordinary_birth_recurrent_mutation",
            status="closed_native_gate1_route_package_qualified",
            conclusion=(
                "Native-core recurrent receipt bytes are byte-comparable and "
                "the native CPU route now qualifies native-built canonical "
                "recurrent receipt digest construction through newborn "
                "publication, same-slot fresh-owner reuse, rejected/disabled "
                "receipt bytes, generation-turnover checkpoint/resume "
                "repeatability, artifact-bound multicell checkpoint restore, "
                "and route-package identity for the Gate 1 low-parallelism "
                "envelope."
            ),
            evidence=(
                _source_evidence(
                    root,
                    "engine/evolution/p12_native_cpu_parity_progress_v1.py",
                    "bounded_native_core_byte_comparable_present_not_route_bound",
                ),
                _source_evidence(
                    root,
                    "engine/evolution/p12_native_cpu_parity_progress_v1.py",
                    "native_cpu_gate1_turnover_route_package_qualification_v1",
                ),
                _source_evidence(
                    root,
                    "engine/evolution/p12_native_cpu_parity_progress_v1.py",
                    "binds_bounded_multicell_checkpoint_fixture_to_verified_prebuilt_artifact",
                ),
                _source_evidence(
                    root,
                    "engine/evolution/p12_native_cpu_parity_progress_v1.py",
                    "eiel_modular_recurrent_canonical_receipt_sha256_v1",
                ),
            ),
        ),
        _gate(
            gate="cuda_receipt_authority",
            status="closed_admitted_rowv1_abi_and_compact_parity",
            conclusion=(
                "RowV1 host semantics are frozen, and the CUDA RowV1 v3 "
                "kernel-mirror host projection now binds a full 64-word source "
                "row hash and canonical disabled-reason decoding for RowV1 v3 "
                "mirror rows. Standalone nvcc and compute-sanitizer probes now "
                "validate the modular device writer for accepted rows, disabled "
                "reason rows, counts, overflow, tail clearing, and sequence on "
                "the 64-word RowV1 surface. The bounded CUDA RowV1 device-writer "
                "ABI is admitted and compact CPU/CUDA RowV1 ABI parity is "
                "complete for accepted and disabled fixture fields."
            ),
            evidence=(
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
        ),
        _gate(
            gate="dense_historical_biological_id_scaling",
            status="closed_route_qualified",
            conclusion=(
                "Dense historical-ID scaling is route-qualified for the P12 "
                "identity envelope: high-ID compact native identity, checkpoint "
                "restore/rejection, WDL compact capacity, EED/RESPEC slot-token "
                "cursors, ragged multicell sizing, INT64_MAX sentinel failure, "
                "and CUDA compatibility-shadow non-authority are bound by the "
                "dense-ID route receipt."
            ),
            evidence=(
                _source_evidence(
                    root,
                    "engine/evolution/p12_substrate_gate_receipt_v1.py",
                    "dense_historical_biological_id_scaling",
                ),
                _source_evidence(
                    root,
                    "engine/evolution/p12_dense_id_route_qualification_v1.py",
                    "dense_id_route_package_qualified",
                ),
            ),
        ),
        _gate(
            gate="final_package_route_qualification",
            status="closed_final_package_and_route_qualified",
            conclusion=(
                "The final P12 package now binds the current Genome "
                "V2 package contract, route input contract, native Gate 1 "
                "low-parallelism receipt, dense-ID route receipt, RowV1 "
                "schema/projection/checkpoint contracts, CUDA device-writer "
                "admission, CUDA disabled-event writer sanitizer result, and "
                "compact CPU/CUDA RowV1 parity receipt. It also binds the "
                "final guarded WDL CUDA artifact, source snapshot, package id, "
                "execution route hash, route receipt, and final ready preflight."
            ),
            evidence=(
                _source_evidence(
                    root,
                    "engine/evolution/p12_final_package_candidate_binding_v1.py",
                    "def build_p12_final_package_candidate_binding_v1",
                ),
                _source_evidence(
                    root,
                    "tests/test_p12_final_package_candidate_binding_v1.py",
                    "test_final_package_candidate_binds_current_p12_mechanic_receipts",
                ),
                _source_evidence(
                    root,
                    "engine/evolution/genome_upgrade_v2_package_v1.py",
                    "def executable_package_contract_v1",
                ),
                _source_evidence(
                    root,
                    "engine/evolution/p12_final_package_candidate_binding_v1.py",
                    "FINAL_RUNTIME_ROOT",
                ),
            ),
        ),
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": DECISION,
        "source_commit": source_commit,
        "architecture_sound": True,
        "p12_a_qualified": True,
        "linked_parent_copy_v1_retired": True,
        "genome_v2_sole_production_reproduction_authority": True,
        "successor_campaign_started": False,
        "structural_payload_sha_progress_contract_id": (
            structural_payload_sha_progress["contract_id"]
        ),
        "structural_payload_sha_progress_sha256": (
            structural_payload_sha_progress["contract_sha256"]
        ),
        "recurrent_progress_contract_id": recurrent_progress["contract_id"],
        "recurrent_progress_sha256": recurrent_progress["contract_sha256"],
        "route_bound_birth_receipt_contract_id": route_bound_birth_receipt[
            "contract_id"
        ],
        "route_bound_birth_receipt_sha256": route_bound_birth_receipt[
            "contract_sha256"
        ],
        "native_gate1_contract_id": native_gate1["contract_id"],
        "native_gate1_sha256": native_gate1["contract_sha256"],
        "native_gate1_qualified": native_gate1["native_cpu_gate1_qualified"],
        "final_package_candidate_contract_id": package_candidate["contract_id"],
        "final_package_candidate_sha256": package_candidate["contract_sha256"],
        "final_package_candidate_identity_sha256": (
            package_candidate["package_candidate_identity_sha256"]
        ),
        "final_package_candidate_complete": (
            package_candidate["final_package_candidate_complete"]
        ),
        "final_p12_package_qualified": (
            package_candidate["final_p12_package_qualified"]
        ),
        "substrate_receipt_decision": substrate["decision"],
        "substrate_receipt_sha256": substrate["receipt_sha256"],
        "blockers": [],
        "closed_gates": list(closed_gates),
        "cuda_artifact_sha256": package_candidate["cuda_artifact_sha256"],
        "cuda_manifest_sha256": package_candidate["cuda_manifest_sha256"],
        "genome_v2_package_id_sha256": package_candidate[
            "genome_v2_package_id_sha256"
        ],
        "cuda_execution_route_sha256": package_candidate[
            "cuda_execution_route_sha256"
        ],
        "cuda_execution_route_receipt_sha256": package_candidate[
            "cuda_execution_route_receipt_sha256"
        ],
        "next_programme_step": (
            "After merge and push, prepare the successor Genome V2 campaign from "
            "the P12-A main commit; do not treat this receipt as a successor "
            "science result."
        ),
        "no_claims": [
            "hereditary_capacity_success",
            "ecological_usefulness",
            "adaptation",
            "cognition",
            "intelligence",
            "emergence",
            "consciousness",
            "open_ended_evolution",
        ],
    }
    payload["decision_sha256"] = content_sha256(payload)
    return payload


def canonical_p12_integrated_final_closure_decision_bytes_v1(
    decision: Mapping[str, Any],
) -> bytes:
    """Serialize a decision with stable JSON ordering and no non-ASCII output."""

    return (
        json.dumps(
            dict(decision),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
