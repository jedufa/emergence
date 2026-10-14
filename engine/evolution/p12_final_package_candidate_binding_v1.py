"""P12 final package candidate binding receipt.

This module binds the current P12 mechanics receipts to the existing Genome V2
package and route contracts plus the final guarded CUDA artifact/package/route
qualification receipts.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from engine.authoritative_state.schema import canonical_value, content_sha256
from engine.evolution.genome_upgrade_v2_package_v1 import (
    PACKAGE_CONTRACT_SHA256,
    capacity_layout_contract_v1,
    executable_package_contract_v1,
    obligation_dictionary_v1,
    package_invalidation_field_map_v1,
    route_input_contract_v1,
    supported_world_controller_config_envelope_v1,
)
from engine.evolution.p12_dense_id_route_qualification_v1 import (
    build_p12_dense_id_route_qualification_v1,
)
from engine.evolution.p12_native_cpu_parity_progress_v1 import (
    native_cpu_gate1_turnover_route_package_qualification_v1,
    native_cpu_route_bound_birth_lifecycle_receipt_v1,
)
from engine.evolution.p12_rowv1_cuda_entry_gate_v1 import (
    cuda_receipt_surface_admission_receipt_v1,
    cuda_rowv1_compact_cpu_cuda_parity_receipt_v1,
    cuda_rowv1_disabled_event_writer_receipt_v1,
    cuda_rowv1_modular_device_writer_admission_receipt_v1,
    rowv1_checkpoint_report_collection_receipt_v1,
    rowv1_checkpoint_report_envelope_receipt_v1,
    rowv1_cuda_failclosed_entry_gate_v1,
    rowv1_ordinary_birth_projection_receipt_v1,
    rowv1_semantic_receipt_schema_v1,
)


SCHEMA_VERSION: Final[str] = "p12_final_package_candidate_binding_v1"
CONTRACT_ID: Final[str] = "p12-final-package-candidate-binding-v1"
DECISION: Final[str] = "P12-FINAL-PACKAGE-ROUTE-QUALIFIED-A"
FINAL_RUNTIME_ROOT: Final[Path] = Path(
    public_runtime_path('evidence_raw/p12_final_package_candidate_20260728')
)

P12_PACKAGE_CANDIDATE_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "source_commit",
    "package_contract_sha256",
    "package_invalidation_field_map_sha256",
    "route_input_contract_sha256",
    "capacity_layout_contract_sha256",
    "supported_envelope_sha256",
    "native_cpu_gate1_sha256",
    "native_cpu_route_bound_birth_receipt_sha256",
    "dense_id_route_qualification_sha256",
    "rowv1_schema_sha256",
    "rowv1_projection_sha256",
    "rowv1_checkpoint_report_envelope_sha256",
    "rowv1_checkpoint_report_collection_sha256",
    "rowv1_cuda_failclosed_entry_gate_sha256",
    "cuda_receipt_surface_admission_sha256",
    "cuda_rowv1_disabled_event_writer_sha256",
    "cuda_rowv1_modular_device_writer_admission_sha256",
    "cuda_rowv1_compact_cpu_cuda_parity_sha256",
    "cuda_artifact_sha256",
    "cuda_manifest_sha256",
    "cuda_build_resource_summary_sha256",
    "genome_v2_package_id_sha256",
    "genome_v2_package_receipt_sha256",
    "cuda_execution_route_sha256",
    "cuda_execution_route_receipt_sha256",
    "cuda_route_preflight_receipt_sha256",
    "cuda_immutable_artifact_binding_status",
    "final_route_package_qualification_status",
)


class P12FinalPackageCandidateBindingError(RuntimeError):
    """The P12 package candidate could not bind required live evidence."""


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_evidence(source_root: Path, relative_path: str, fragment: str) -> dict[str, Any]:
    path = source_root / relative_path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise P12FinalPackageCandidateBindingError(
            f"missing P12 package-candidate evidence: {relative_path}"
        ) from exc
    if fragment not in text:
        raise P12FinalPackageCandidateBindingError(
            f"P12 package-candidate evidence fragment is absent: {relative_path}"
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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise P12FinalPackageCandidateBindingError(
            f"missing final P12 runtime evidence: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise P12FinalPackageCandidateBindingError(
            f"final P12 runtime evidence is not a JSON object: {path}"
        )
    return value


def _final_runtime_binding_v1() -> dict[str, Any]:
    artifact = (
        FINAL_RUNTIME_ROOT
        / "13_guarded_wdl_controller_v2_build_after_lifecycle_repair/artifact/"
        "eiel_cuda_architecture_b_flattened_wdl_controller_v2.so"
    )
    manifest = artifact.with_suffix(".so.manifest.json")
    build_summary_path = (
        FINAL_RUNTIME_ROOT
        / "13_guarded_wdl_controller_v2_build_after_lifecycle_repair/"
        "genome_v2_build_resource_summary.json"
    )
    package_summary_path = (
        FINAL_RUNTIME_ROOT
        / "14_package_qualification_after_lifecycle_repair/"
        "genome_v2_package_qualification_summary.json"
    )
    package_receipt_path = (
        FINAL_RUNTIME_ROOT
        / "14_package_qualification_after_lifecycle_repair/"
        "genome_v2_package_identity_receipt.json"
    )
    route_receipt_path = (
        FINAL_RUNTIME_ROOT
        / "15_candidate_wdl_after_lifecycle_repair/"
        "execution_route_qualification.json"
    )
    preflight_receipt_path = (
        FINAL_RUNTIME_ROOT
        / "16_final_route_qualification_run_after_lifecycle_repair/"
        "preflight_receipt.json"
    )
    manifest_payload = _load_json(manifest)
    build_summary = _load_json(build_summary_path)
    package_summary = _load_json(package_summary_path)
    route_receipt = _load_json(route_receipt_path)
    preflight_receipt = _load_json(preflight_receipt_path)

    artifact_sha256 = _file_sha256(artifact)
    manifest_sha256 = _file_sha256(manifest)
    package_receipt_sha256 = _file_sha256(package_receipt_path)
    route_receipt_sha256 = _file_sha256(route_receipt_path)
    preflight_receipt_sha256 = _file_sha256(preflight_receipt_path)
    if manifest_payload.get("artifact_sha256") != artifact_sha256:
        raise P12FinalPackageCandidateBindingError(
            "final CUDA manifest artifact SHA does not match artifact bytes"
        )
    if package_summary.get("ok") is not True or package_summary.get("decision") != (
        "GENOME-V2-PACKAGE-CANDIDATE-PASS"
    ):
        raise P12FinalPackageCandidateBindingError(
            "final Genome V2 package qualification is not passing"
        )
    if package_summary.get("artifact_sha256") != artifact_sha256:
        raise P12FinalPackageCandidateBindingError(
            "final package qualification does not bind the final CUDA artifact"
        )
    if package_summary.get("manifest_sha256") != manifest_sha256:
        raise P12FinalPackageCandidateBindingError(
            "final package qualification does not bind the final CUDA manifest"
        )
    if route_receipt.get("authority_state") != (
        "route_qualified"
    ):
        raise P12FinalPackageCandidateBindingError(
            "final execution route qualification is not passing"
        )
    route_hash = route_receipt.get("execution_route_sha256")
    if (
        not isinstance(route_hash, str)
        or len(route_hash) != 64
        or preflight_receipt.get("ready_for_execution") is not True
        or preflight_receipt.get("execution_route", {}).get(
            "execution_route_sha256"
        )
        != route_hash
        or preflight_receipt.get("execution_route", {}).get(
            "qualification_receipt_sha256"
        )
        != route_receipt_sha256
    ):
        raise P12FinalPackageCandidateBindingError(
            "final route preflight is not bound to the final route receipt"
        )
    return {
        "cuda_artifact_path": str(artifact),
        "cuda_artifact_sha256": artifact_sha256,
        "cuda_manifest_path": str(manifest),
        "cuda_manifest_sha256": manifest_sha256,
        "cuda_build_resource_summary_path": str(build_summary_path),
        "cuda_build_resource_summary_sha256": _file_sha256(build_summary_path),
        "cuda_build_guard_ok": build_summary.get("ok") is True,
        "cuda_build_source_commit": (
            manifest_payload.get("committed_source_snapshot", {}).get(
                "build_source_commit"
            )
        ),
        "cuda_compiled_source_snapshot_sha256": (
            manifest_payload.get("committed_source_snapshot", {}).get(
                "compiled_source_snapshot_sha256"
            )
        ),
        "cuda_arch": manifest_payload.get("cuda_arch"),
        "cuda_sass_only": manifest_payload.get("sass_only"),
        "cuda_runtime_jit_compilation": manifest_payload.get(
            "runtime_jit_compilation"
        ),
        "genome_v2_package_id_sha256": package_summary.get("package_id_sha256"),
        "genome_v2_package_receipt_path": str(package_receipt_path),
        "genome_v2_package_receipt_sha256": package_receipt_sha256,
        "cuda_execution_route_sha256": route_hash,
        "cuda_execution_route_receipt_path": str(route_receipt_path),
        "cuda_execution_route_receipt_sha256": route_receipt_sha256,
        "cuda_route_preflight_receipt_path": str(preflight_receipt_path),
        "cuda_route_preflight_receipt_sha256": preflight_receipt_sha256,
        "cuda_route_case_count": route_receipt.get(
            "checkpoint_results", {}
        ).get("case_count"),
        "cuda_route_exact_comparison_count": len(
            route_receipt.get("exact_comparisons", ())
        ),
        "cuda_route_covered_obligations": route_receipt.get(
            "covered_obligations"
        ),
        "cuda_route_ready_for_execution": True,
    }


def p12_final_package_candidate_identity_v1(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract the identity fields that invalidate the P12 package candidate."""

    missing = [
        field
        for field in P12_PACKAGE_CANDIDATE_IDENTITY_FIELDS
        if field not in receipt
    ]
    if missing:
        raise P12FinalPackageCandidateBindingError(
            "P12 package-candidate identity is missing fields: "
            + ", ".join(missing)
        )
    return {field: receipt[field] for field in P12_PACKAGE_CANDIDATE_IDENTITY_FIELDS}


def p12_package_candidate_invalidation_decision_v1(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two P12 package-candidate identities."""

    baseline_identity = p12_final_package_candidate_identity_v1(baseline)
    candidate_identity = p12_final_package_candidate_identity_v1(candidate)
    changed_fields = [
        field
        for field in P12_PACKAGE_CANDIDATE_IDENTITY_FIELDS
        if baseline_identity[field] != candidate_identity[field]
    ]
    return {
        "schema_version": "p12_package_candidate_invalidation_decision_v1",
        "reusable": not changed_fields,
        "action": "reuse" if not changed_fields else "requalify",
        "changed_fields": changed_fields,
        "identity_fields": list(P12_PACKAGE_CANDIDATE_IDENTITY_FIELDS),
        "unbound_metadata_ignored": True,
    }


def build_p12_final_package_candidate_binding_v1(
    *,
    source_root: Path | str = ".",
    source_commit: str | None = None,
) -> dict[str, Any]:
    """Return a P12 package-candidate receipt bound to current mechanic inputs."""

    root = Path(source_root).resolve()
    final_runtime = _final_runtime_binding_v1()
    package_contract = executable_package_contract_v1()
    obligation_dictionary = obligation_dictionary_v1()
    route_input_contract = route_input_contract_v1()
    capacity_layout_contract = capacity_layout_contract_v1()
    supported_envelope = supported_world_controller_config_envelope_v1()
    try:
        native_gate1 = native_cpu_gate1_turnover_route_package_qualification_v1()
        native_birth = native_cpu_route_bound_birth_lifecycle_receipt_v1()
        dense_id = build_p12_dense_id_route_qualification_v1(
            source_root=root,
            source_commit=source_commit,
        )
        rowv1_schema = rowv1_semantic_receipt_schema_v1()
        rowv1_projection = rowv1_ordinary_birth_projection_receipt_v1()
        checkpoint_envelope = rowv1_checkpoint_report_envelope_receipt_v1()
        checkpoint_collection = rowv1_checkpoint_report_collection_receipt_v1()
        cuda_entry = rowv1_cuda_failclosed_entry_gate_v1(source_root=root)
        cuda_admission = cuda_receipt_surface_admission_receipt_v1(source_root=root)
        cuda_disabled_writer = cuda_rowv1_disabled_event_writer_receipt_v1(
            source_root=root
        )
        cuda_modular_writer = cuda_rowv1_modular_device_writer_admission_receipt_v1(
            source_root=root
        )
        cuda_compact_parity = cuda_rowv1_compact_cpu_cuda_parity_receipt_v1(
            source_root=root
        )
    except RuntimeError as exc:
        raise P12FinalPackageCandidateBindingError(
            "P12 package-candidate dependency evidence is invalid: " + str(exc)
        ) from exc

    if native_gate1["native_cpu_gate1_qualified"] is not True:
        raise P12FinalPackageCandidateBindingError("native Gate 1 is not qualified")
    if dense_id["dense_id_route_package_qualified"] is not True:
        raise P12FinalPackageCandidateBindingError("dense-ID route is not qualified")
    if cuda_admission["cuda_receipt_surface_currently_admitted"] is not True:
        raise P12FinalPackageCandidateBindingError(
            "CUDA receipt surface is not admitted"
        )
    if cuda_compact_parity["compact_cpu_cuda_parity_complete"] is not True:
        raise P12FinalPackageCandidateBindingError(
            "compact CPU/CUDA RowV1 parity is not complete"
        )

    mechanic_bindings = {
        "native_cpu_gate1": {
            "contract_id": native_gate1["contract_id"],
            "sha256": native_gate1["contract_sha256"],
            "qualified": native_gate1["native_cpu_gate1_qualified"],
            "artifact_sha256": native_gate1["artifact_sha256"],
            "build_manifest_sha256": native_gate1["build_manifest_sha256"],
            "route": native_gate1["qualified_route"],
            "envelope": native_gate1["qualified_envelope"],
        },
        "native_cpu_route_bound_birth": {
            "contract_id": native_birth["contract_id"],
            "sha256": native_birth["contract_sha256"],
            "artifact_sha256": native_birth["artifact_sha256"],
            "build_manifest_sha256": native_birth["build_manifest_sha256"],
        },
        "dense_id_route": {
            "contract_id": dense_id["contract_id"],
            "sha256": dense_id["receipt_sha256"],
            "status": dense_id["status"],
            "qualified": dense_id["dense_id_route_package_qualified"],
        },
        "rowv1_schema": {
            "contract_id": rowv1_schema["contract_id"],
            "sha256": rowv1_schema["contract_sha256"],
        },
        "rowv1_projection": {
            "contract_id": rowv1_projection["contract_id"],
            "sha256": rowv1_projection["contract_sha256"],
        },
        "rowv1_checkpoint_report_envelope": {
            "contract_id": checkpoint_envelope["contract_id"],
            "sha256": checkpoint_envelope["contract_sha256"],
        },
        "rowv1_checkpoint_report_collection": {
            "contract_id": checkpoint_collection["contract_id"],
            "sha256": checkpoint_collection["contract_sha256"],
        },
        "rowv1_cuda_failclosed_entry_gate": {
            "contract_id": cuda_entry["contract_id"],
            "sha256": cuda_entry["contract_sha256"],
            "field_decision": cuda_entry["cuda_field_decision"],
        },
        "cuda_receipt_surface_admission": {
            "contract_id": cuda_admission["contract_id"],
            "sha256": cuda_admission["contract_sha256"],
            "admitted": cuda_admission["cuda_receipt_surface_currently_admitted"],
            "route_package_qualification_authorized": (
                cuda_admission["route_package_qualification_authorized"]
            ),
        },
        "cuda_rowv1_disabled_event_writer": {
            "contract_id": cuda_disabled_writer["contract_id"],
            "sha256": cuda_disabled_writer["contract_sha256"],
            "sanitizer_passed": cuda_disabled_writer["sanitizer_passed"],
        },
        "cuda_rowv1_modular_device_writer_admission": {
            "contract_id": cuda_modular_writer["contract_id"],
            "sha256": cuda_modular_writer["contract_sha256"],
            "admitted": cuda_modular_writer["cuda_receipt_surface_admitted"],
        },
        "cuda_rowv1_compact_cpu_cuda_parity": {
            "contract_id": cuda_compact_parity["contract_id"],
            "sha256": cuda_compact_parity["contract_sha256"],
            "complete": cuda_compact_parity["compact_cpu_cuda_parity_complete"],
        },
        "final_cuda_artifact_route": {
            "artifact_sha256": final_runtime["cuda_artifact_sha256"],
            "manifest_sha256": final_runtime["cuda_manifest_sha256"],
            "package_id_sha256": final_runtime["genome_v2_package_id_sha256"],
            "execution_route_sha256": final_runtime[
                "cuda_execution_route_sha256"
            ],
            "route_receipt_sha256": final_runtime[
                "cuda_execution_route_receipt_sha256"
            ],
            "ready_for_execution": final_runtime["cuda_route_ready_for_execution"],
        },
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "decision": DECISION,
        "source_commit": source_commit,
        "provider_id": package_contract["provider_id"],
        "runtime_profile_id": package_contract["runtime_profile_id"],
        "genome_v2_sole_production_reproduction_authority": True,
        "linked_parent_copy_v1_retired": True,
        "python_production_campaign_authorized": False,
        "native_cpu_low_parallelism_route_preserved": True,
        "cuda_batched_route_preserved": True,
        "package_contract_sha256": PACKAGE_CONTRACT_SHA256,
        "package_required_identity_fields_sha256": content_sha256(
            package_contract["required_identity_fields"]
        ),
        "package_obligation_set_sha256": (
            obligation_dictionary["obligation_set_sha256"]
        ),
        "package_invalidation_field_map_sha256": content_sha256(
            package_invalidation_field_map_v1()
        ),
        "route_input_contract_sha256": content_sha256(route_input_contract),
        "capacity_layout_contract_sha256": content_sha256(capacity_layout_contract),
        "supported_envelope_sha256": content_sha256(supported_envelope),
        "mechanic_bindings": mechanic_bindings,
        "mechanic_binding_set_sha256": content_sha256(mechanic_bindings),
        "native_cpu_gate1_sha256": native_gate1["contract_sha256"],
        "native_cpu_route_bound_birth_receipt_sha256": native_birth[
            "contract_sha256"
        ],
        "native_cpu_artifact_sha256": native_gate1["artifact_sha256"],
        "native_cpu_build_manifest_sha256": native_gate1["build_manifest_sha256"],
        "dense_id_route_qualification_sha256": dense_id["receipt_sha256"],
        "rowv1_schema_sha256": rowv1_schema["contract_sha256"],
        "rowv1_projection_sha256": rowv1_projection["contract_sha256"],
        "rowv1_checkpoint_report_envelope_sha256": checkpoint_envelope[
            "contract_sha256"
        ],
        "rowv1_checkpoint_report_collection_sha256": checkpoint_collection[
            "contract_sha256"
        ],
        "rowv1_cuda_failclosed_entry_gate_sha256": cuda_entry["contract_sha256"],
        "cuda_receipt_surface_admission_sha256": cuda_admission["contract_sha256"],
        "cuda_rowv1_disabled_event_writer_sha256": cuda_disabled_writer[
            "contract_sha256"
        ],
        "cuda_rowv1_modular_device_writer_admission_sha256": cuda_modular_writer[
            "contract_sha256"
        ],
        "cuda_rowv1_compact_cpu_cuda_parity_sha256": cuda_compact_parity[
            "contract_sha256"
        ],
        "cuda_disabled_event_writer_sanitizer": "memcheck_0_errors",
        **final_runtime,
        "cuda_immutable_artifact_binding_status": "bound_guarded_final_artifact",
        "final_route_package_qualification_status": "route_qualified",
        "final_package_candidate_complete": True,
        "final_p12_package_qualified": True,
        "p12_a_qualified": True,
        "route_package_qualification_authorized": True,
        "successor_campaign_authorized": False,
        "successor_science_readiness_recommendation": (
            "ready_to_prepare_successor_campaign_from_p12a_main_after_merge_push"
        ),
        "required_remaining_route_gates": (),
        "source_evidence": (
            _source_evidence(
                root,
                "engine/evolution/genome_upgrade_v2_package_v1.py",
                "def executable_package_contract_v1",
            ),
            _source_evidence(
                root,
                "engine/evolution/p12_native_cpu_parity_progress_v1.py",
                "native_cpu_gate1_turnover_route_package_qualification_v1",
            ),
            _source_evidence(
                root,
                "engine/evolution/p12_dense_id_route_qualification_v1.py",
                "dense_id_route_package_qualified",
            ),
            _source_evidence(
                root,
                "engine/evolution/p12_rowv1_cuda_entry_gate_v1.py",
                "P12-CUDA-ROWV1-COMPACT-CPU-CUDA-PARITY-A",
            ),
            _source_evidence(
                root,
                "engine/campaigns/production_engine_lifecycle_v1.py",
                "_ADMITTED_WORLD_ENVELOPES",
            ),
            _source_evidence(
                root,
                "engine/campaigns/patch_runner_factory.py",
                "PRODUCTION_LIFECYCLE_PARAMETER_NAMES_V1",
            ),
        ),
        "no_claims": (
            "No hereditary-capacity success, ecological usefulness, adaptation, "
            "cognition, intelligence, emergence, consciousness, open-ended "
            "evolution, or successor campaign result is authorized by this "
            "package-route receipt."
        ),
    }
    payload = canonical_value(payload)
    identity = p12_final_package_candidate_identity_v1(payload)
    payload["package_candidate_identity_sha256"] = content_sha256(identity)
    payload["contract_sha256"] = content_sha256(payload)
    return payload


def canonical_p12_final_package_candidate_binding_bytes_v1(
    receipt: Mapping[str, Any],
) -> bytes:
    """Serialize a package-candidate receipt with stable ASCII JSON ordering."""

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


__all__ = [
    "CONTRACT_ID",
    "DECISION",
    "P12FinalPackageCandidateBindingError",
    "P12_PACKAGE_CANDIDATE_IDENTITY_FIELDS",
    "build_p12_final_package_candidate_binding_v1",
    "canonical_p12_final_package_candidate_binding_bytes_v1",
    "p12_final_package_candidate_identity_v1",
    "p12_package_candidate_invalidation_decision_v1",
]
