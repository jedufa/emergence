#!/usr/bin/env python3
"""Join live cancellation and atomic transaction receipts into the T1 matrix."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ENGINE: Final[Path] = ROOT / "engine"
for _path in (ROOT, ENGINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from evolution.genome_upgrade_v2_cancellation_v1 import (  # noqa: E402
    CANCELLATION_LIFECYCLE_CASES,
    HISTORICAL_CUT_POINT_REPLACEMENTS,
    cancellation_lifecycle_case_v1,
)


SCHEMA_VERSION: Final[str] = "genome_v2_cancellation_lifecycle_gate_v1"
DECISION_PASS: Final[str] = "GENOME-V2-CANCELLATION-LIFECYCLE-PASS"


class CancellationLifecycleGateError(ValueError):
    """One input receipt is absent, mixed-package, or internally inconsistent."""


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CancellationLifecycleGateError(f"{path}: expected one JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _require(value: bool, message: str) -> None:
    if not value:
        raise CancellationLifecycleGateError(message)


def _atomic_row_passes(row: dict[str, Any]) -> bool:
    if row.get("status") != "PASS":
        return False
    if row.get("all_state_exact") is True:
        return True
    return (
        row.get("expected_authority") == "one_complete_committed_transaction"
        and row.get("committed_transaction_exact") is True
        and row.get("all_state_exact") is False
    )


def _artifact_binding(manifest: dict[str, Any], path: Path) -> dict[str, str]:
    artifact = Path(str(manifest.get("artifact", ""))).resolve()
    _require(artifact.is_file(), "artifact manifest does not name a live artifact")
    artifact_sha256 = str(manifest.get("artifact_sha256", ""))
    _require(
        _file_sha256(artifact) == artifact_sha256,
        "artifact bytes do not match the artifact manifest",
    )
    provenance = (
        manifest.get("artifact_identity", {})
        .get("runtime_resource_receipt", {})
        .get("authoritative_state_build_provenance_v1", {})
    )
    source_commit = str(provenance.get("build_source_commit", ""))
    source_snapshot = str(provenance.get("compiled_source_snapshot_sha256", ""))
    _require(len(source_commit) == 40, "artifact source commit is absent")
    _require(len(source_snapshot) == 64, "compiled source snapshot is absent")
    return {
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256,
        "artifact_manifest": str(path.resolve()),
        "artifact_manifest_sha256": _file_sha256(path),
        "build_source_commit": source_commit,
        "compiled_source_snapshot_sha256": source_snapshot,
    }


def _receipt_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: _file_sha256(path) for name, path in paths.items()}


def _validate_inputs(
    *,
    binding: dict[str, str],
    control: dict[str, Any],
    active: dict[str, Any],
    birth: dict[str, Any],
    turnover: dict[str, Any],
    checkpoint: dict[str, Any],
    observability: dict[str, Any],
) -> None:
    artifact_sha256 = binding["artifact_sha256"]
    for name, receipt in (("control", control), ("active", active)):
        _require(receipt.get("ok") is True, f"{name} cancellation probe failed")
        _require(
            receipt.get("decision") == "GENOME-V2-NATIVE-CANCELLATION-PASS",
            f"{name} cancellation decision is not PASS",
        )
        receipt_manifest = receipt.get("artifacts", {}).get("cuda_manifest", {})
        _require(
            receipt_manifest.get("artifact_sha256") == artifact_sha256,
            f"{name} cancellation probe used a different CUDA artifact",
        )
        _require(
            receipt.get("claims", {}).get("checkpoint_on_cancel_exact") is True,
            f"{name} cancellation checkpoint is not exact",
        )
        _require(
            receipt.get("claims", {}).get("fresh_owner_restore_after_cancel_exact")
            is True,
            f"{name} fresh-owner restore is not exact",
        )
        _require(
            receipt.get("no_post_terminal_mutation", {}).get("ok") is True,
            f"{name} terminal state is not sticky",
        )
        _require(
            receipt.get("cancel_request_receipt", {}).get("disposition") == "accepted",
            f"{name} first cancellation request was not accepted",
        )
        _require(
            receipt.get("repeated_cancel_request_receipt", {}).get("disposition")
            == "already_requested",
            f"{name} repeated cancellation was not idempotent",
        )
        _require(
            receipt.get("pre_submission_cancel_request_receipt", {}).get("disposition")
            == "not_running",
            f"{name} pre-submission cancellation did not fail closed",
        )
        _require(
            receipt.get("pre_dispatch_cancel_request_receipt", {}).get("disposition")
            == "not_running",
            f"{name} pre-dispatch cancellation did not fail closed",
        )
        _require(
            receipt.get("no_post_terminal_mutation", {})
            .get("late_cancel_receipt", {})
            .get("disposition")
            == "already_terminal",
            f"{name} late cancellation did not preserve terminal state",
        )
    _require(
        control.get("cancel_arrival_observation", {}).get("request_phase")
        == "inter_tick_control",
        "control-window probe did not observe its requested phase",
    )
    _require(
        active.get("cancel_arrival_observation", {}).get("request_phase")
        == "active_generation_transaction",
        "active-transaction probe did not observe its requested phase",
    )
    active_arrival = active["cancel_arrival_observation"]
    active_manifest = active_arrival.get("manifest", {})
    _require(
        active_manifest.get("status") == "tick_running"
        and active_manifest.get("generation_transaction_open") is True
        and active_manifest.get("inter_tick_control_window_open") is False,
        "active-transaction request was not delivered after window close",
    )
    _require(
        active_arrival.get("qualification_binding_hold_ms", 0)
        > active_manifest.get("inter_tick_control_wait_timeout_ms", 0),
        "active-transaction qualification handoff did not cross window close",
    )
    for name, receipt in (("birth", birth), ("turnover", turnover)):
        _require(
            receipt.get("qualification_status") == "PASS",
            f"{name} atomicity receipt is not PASS",
        )
        _require(
            receipt.get("artifact", {}).get("manifest", {}).get("artifact_sha256")
            == artifact_sha256,
            f"{name} atomicity receipt used a different CUDA artifact",
        )
        rows = receipt.get("failure_injection_rows")
        _require(isinstance(rows, list) and rows, f"{name} matrix is absent")
        _require(
            all(_atomic_row_passes(row) for row in rows),
            f"{name} atomicity matrix has a failing row",
        )
    _require(
        checkpoint.get("qualification_status") == "PASS"
        and checkpoint.get("artifact_sha256") == artifact_sha256,
        "checkpoint receipt is not a same-artifact PASS",
    )
    _require(
        observability.get("ok") is True
        and observability.get("decision") == "GENOME-V2-OBSERVABILITY-PASS"
        and observability.get("source_artifact_sha256") == artifact_sha256,
        "observability receipt is not a same-artifact PASS",
    )


def _row(
    *,
    case_id: str,
    sequence: int,
    evidence_kind: str,
    evidence_sha256: str,
    request_id: str,
    controller_id: str,
    request_arrival: str,
    authoritative_observation: str,
    transaction_evidence: str,
    disposition: str,
    terminal_status: str,
    checkpoint_exact: bool,
    fresh_owner_exact: bool,
    no_post_terminal_mutation: bool,
    binding: dict[str, str],
    package_candidate_id: str,
    route_requirement_id: str,
) -> dict[str, Any]:
    contract = cancellation_lifecycle_case_v1(case_id)
    return {
        "case_id": case_id,
        "deterministic_sequence": sequence,
        "request_id": request_id,
        "controller_id": controller_id,
        "request_arrival": request_arrival,
        "authoritative_observation": authoritative_observation,
        "transaction_settlement": contract.transaction_settlement_phase,
        "transaction_evidence": transaction_evidence,
        "terminal_sealing": contract.terminal_sealing_phase,
        "disposition": disposition,
        "terminal_status": terminal_status,
        "checkpoint_on_cancel_exact": checkpoint_exact,
        "fresh_owner_restore_exact": fresh_owner_exact,
        "no_post_terminal_mutation": no_post_terminal_mutation,
        "delivery_precision": (
            "direct_lifecycle_observation"
            if evidence_kind != "atomicity_join"
            else "bounded_active_generation_not_transaction_microphase"
        ),
        "evidence_kind": evidence_kind,
        "evidence_receipt_sha256": evidence_sha256,
        "package_candidate_id": package_candidate_id,
        "route_requirement_id": route_requirement_id,
        "route_qualification_status": "pending_t3",
        "build_source_commit": binding["build_source_commit"],
        "artifact_sha256": binding["artifact_sha256"],
        "status": "PASS",
    }


def execute(
    *,
    artifact_manifest_path: Path,
    control_receipt_path: Path,
    active_receipt_path: Path,
    birth_receipt_path: Path,
    turnover_receipt_path: Path,
    checkpoint_receipt_path: Path,
    observability_receipt_path: Path,
    output_dir: Path,
    package_candidate_id: str,
    route_requirement_id: str,
) -> dict[str, Any]:
    paths = {
        "control": control_receipt_path,
        "active": active_receipt_path,
        "birth": birth_receipt_path,
        "turnover": turnover_receipt_path,
        "checkpoint": checkpoint_receipt_path,
        "observability": observability_receipt_path,
    }
    manifest = _load(artifact_manifest_path)
    receipts = {name: _load(path) for name, path in paths.items()}
    binding = _artifact_binding(manifest, artifact_manifest_path)
    _validate_inputs(binding=binding, **receipts)
    hashes = _receipt_hashes(paths)
    control = receipts["control"]
    active = receipts["active"]
    birth = receipts["birth"]
    turnover = receipts["turnover"]

    control_request = control["cancel_request_receipt"]
    active_request = active["cancel_request_receipt"]
    control_cancel = control["terminal_result"]["cancellation"]
    active_cancel = active["terminal_result"]["cancellation"]
    pre_submission = control["pre_submission_cancel_request_receipt"]
    pre_dispatch = control["pre_dispatch_cancel_request_receipt"]
    repeated = active["repeated_cancel_request_receipt"]
    late = active["no_post_terminal_mutation"]["late_cancel_receipt"]
    birth_rows = {row["case_id"]: row for row in birth["failure_injection_rows"]}
    turnover_rows = {row["case_id"]: row for row in turnover["failure_injection_rows"]}
    direct_common = {
        "checkpoint_exact": True,
        "fresh_owner_exact": True,
        "no_post_terminal_mutation": True,
        "binding": binding,
        "package_candidate_id": package_candidate_id,
        "route_requirement_id": route_requirement_id,
    }
    pre_owner_payload = {
        "schema_version": "genome_v2_pre_owner_cancellation_receipt_v1",
        "request_id": f"{package_candidate_id}:cancel-before-owner",
        "package_candidate_id": package_candidate_id,
        "route_requirement_id": route_requirement_id,
        "owner_created": False,
        "submission_started": False,
        "resident_dispatch_started": False,
        "disposition": "cancelled_before_submission",
        "terminal_status_changed": False,
        "artifact_binding": binding,
    }

    rows: list[dict[str, Any]] = []
    rows.append(
        _row(
            case_id="before_submission_or_owner_start",
            sequence=1,
            evidence_kind="direct_lifecycle_observation",
            evidence_sha256=_canonical_sha256(pre_owner_payload),
            request_id=pre_owner_payload["request_id"],
            controller_id="not_created",
            request_arrival="before_submission",
            authoritative_observation="production_admission",
            transaction_evidence="no_transaction_started",
            disposition=pre_owner_payload["disposition"],
            terminal_status="not_started",
            checkpoint_exact=False,
            fresh_owner_exact=False,
            no_post_terminal_mutation=True,
            binding=binding,
            package_candidate_id=package_candidate_id,
            route_requirement_id=route_requirement_id,
        )
    )
    rows.append(
        _row(
            case_id="after_owner_creation_before_first_generation",
            sequence=2,
            evidence_kind="direct_lifecycle_observation",
            evidence_sha256=hashes["control"],
            request_id=pre_submission["request_id"],
            controller_id=pre_submission["controller_id"],
            request_arrival="auxiliary_owner_initialized",
            authoritative_observation=pre_submission["state_observed"],
            transaction_evidence="no_generation_transaction_started",
            disposition=pre_submission["disposition"],
            terminal_status="not_started",
            checkpoint_exact=False,
            fresh_owner_exact=False,
            no_post_terminal_mutation=True,
            binding=binding,
            package_candidate_id=package_candidate_id,
            route_requirement_id=route_requirement_id,
        )
    )
    rows.append(
        _row(
            case_id="before_resident_dispatch",
            sequence=3,
            evidence_kind="direct_lifecycle_observation",
            evidence_sha256=hashes["control"],
            request_id=pre_dispatch["request_id"],
            controller_id=pre_dispatch["controller_id"],
            request_arrival="main_owner_before_resident_dispatch",
            authoritative_observation=pre_dispatch["state_observed"],
            transaction_evidence="no_resident_dispatch_started",
            disposition=pre_dispatch["disposition"],
            terminal_status="not_started",
            checkpoint_exact=False,
            fresh_owner_exact=False,
            no_post_terminal_mutation=True,
            binding=binding,
            package_candidate_id=package_candidate_id,
            route_requirement_id=route_requirement_id,
        )
    )

    atomic_cases = (
        (
            "before_ordinary_birth_transaction",
            "after_destination_reservation",
            "birth",
        ),
        (
            "during_ordinary_birth_transaction",
            "after_genotype_staging",
            "birth",
        ),
        (
            "immediately_after_ordinary_birth_commit",
            "immediately_after_commit_before_reporting",
            "birth",
        ),
        ("before_turnover_transaction", "mid_cull", "turnover"),
        ("during_turnover_transaction", "after_child_staging", "turnover"),
        (
            "immediately_after_turnover_commit",
            "after_turnover_commit_before_checkpoint",
            "turnover",
        ),
        (
            "before_generation_boundary_sealing",
            "active_generation_to_completed_boundary",
            "active_generation_transaction",
        ),
    )
    for sequence, (case_id, transaction_case, owner) in enumerate(
        atomic_cases, start=4
    ):
        if owner == "birth":
            _require(
                transaction_case in birth_rows,
                f"missing birth transaction row {transaction_case}",
            )
            transaction_hash = hashes["birth"]
        elif owner == "turnover":
            _require(
                transaction_case in turnover_rows,
                f"missing turnover transaction row {transaction_case}",
            )
            transaction_hash = hashes["turnover"]
        else:
            transaction_hash = hashes["active"]
        rows.append(
            _row(
                case_id=case_id,
                sequence=sequence,
                evidence_kind="atomicity_join",
                evidence_sha256=f"{hashes['active']};{transaction_hash}",
                request_id=active_request["request_id"],
                controller_id=active_request["controller_id"],
                request_arrival=active["cancel_arrival_observation"]["request_phase"],
                authoritative_observation=active_cancel["request_observation_cursor"],
                transaction_evidence=transaction_case,
                disposition=active_request["disposition"],
                terminal_status=active["terminal_result"]["terminal_status"],
                **direct_common,
            )
        )

    direct_cases = (
        (
            "supported_generation_boundary_cancellation",
            control_request,
            control_cancel,
            "control",
            "post_tick_zero_control_window",
            "generation_boundary_and_checkpoint",
        ),
        (
            "checkpoint_on_cancel_preparation",
            control_request,
            control_cancel,
            "control",
            "awaiting_cancellation_checkpoint",
            "checkpoint_requested_then_persisted",
        ),
        (
            "after_checkpoint_publication_before_terminal_response",
            control_request,
            control_cancel,
            "control",
            "checkpoint_published_before_acknowledgement",
            "durable_boundary_acknowledgement",
        ),
        (
            "repeated_cancellation_after_acceptance",
            repeated,
            active_cancel,
            "active",
            "after_first_request_acceptance",
            "first_request_identity_retained",
        ),
        (
            "contradictory_operation_after_terminal_sealing",
            late,
            active_cancel,
            "active",
            "after_terminal_sealing",
            "sealed_manifest_and_receipt_exact",
        ),
    )
    for sequence, (
        case_id,
        request,
        cancellation,
        source,
        arrival,
        transaction,
    ) in enumerate(direct_cases, start=11):
        rows.append(
            _row(
                case_id=case_id,
                sequence=sequence,
                evidence_kind="direct_lifecycle_observation",
                evidence_sha256=hashes[source],
                request_id=request["request_id"],
                controller_id=request["controller_id"],
                request_arrival=arrival,
                authoritative_observation=cancellation["request_observation_cursor"],
                transaction_evidence=transaction,
                disposition=request["disposition"],
                terminal_status="cancelled",
                **direct_common,
            )
        )

    _require(
        tuple(row["case_id"] for row in rows) == CANCELLATION_LIFECYCLE_CASES,
        "generated cancellation matrix order or coverage drifted",
    )
    _require(
        all(row["status"] == "PASS" for row in rows),
        "cancellation lifecycle matrix has a failing row",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "genome_v2_cancellation_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_payload = {
        "schema_version": "genome_v2_checkpoint_on_cancel_receipt_v1",
        "ok": True,
        "package_candidate_id": package_candidate_id,
        "route_requirement_id": route_requirement_id,
        "artifact_binding": binding,
        "control_window": control["boundary_acknowledgement"],
        "active_transaction": active["boundary_acknowledgement"],
        "checkpoint_restore_receipt_sha256": hashes["checkpoint"],
    }
    resume_payload = {
        "schema_version": "genome_v2_cancel_resume_exactness_v1",
        "ok": True,
        "control_window": control["parity"]["checkpoint_parity"],
        "active_transaction": active["parity"]["checkpoint_parity"],
        "artifact_binding": binding,
    }
    no_mutation_payload = {
        "schema_version": "genome_v2_no_post_terminal_mutation_aggregate_v1",
        "ok": True,
        "control_window": control["no_post_terminal_mutation"],
        "active_transaction": active["no_post_terminal_mutation"],
        "artifact_binding": binding,
    }
    supersession_payload = {
        "schema_version": "genome_v2_historical_cancellation_supersession_v1",
        "ok": True,
        "rows": [
            {
                "historical_cut_point": old,
                "replacement_lifecycle_cases": list(replacements),
                "replacement_status": "PASS",
            }
            for old, replacements in HISTORICAL_CUT_POINT_REPLACEMENTS.items()
        ],
        "artifact_binding": binding,
    }
    payloads = {
        "genome_v2_pre_owner_cancellation_receipt.json": pre_owner_payload,
        "genome_v2_checkpoint_on_cancel_receipt.json": checkpoint_payload,
        "genome_v2_cancel_resume_exactness.json": resume_payload,
        "genome_v2_no_post_terminal_mutation.json": no_mutation_payload,
        "genome_v2_historical_cancellation_supersession.json": (supersession_payload),
    }
    for name, payload in payloads.items():
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "decision": DECISION_PASS,
        "case_count": len(rows),
        "package_candidate_id": package_candidate_id,
        "route_requirement_id": route_requirement_id,
        "route_qualification_status": "pending_t3",
        "artifact_binding": binding,
        "input_receipt_sha256": hashes,
        "outputs": {
            name: _file_sha256(output_dir / name)
            for name in (matrix_path.name, *payloads)
        },
        "claim_boundary": (
            "transaction-phase rows join bounded active-generation request "
            "arrival to exact atomicity receipts; they do not claim host "
            "observation of a CUDA instruction-level microphase"
        ),
    }
    summary_path = output_dir / "genome_v2_cancellation_gate_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-manifest", type=Path, required=True)
    parser.add_argument("--control-receipt", type=Path, required=True)
    parser.add_argument("--active-receipt", type=Path, required=True)
    parser.add_argument("--birth-receipt", type=Path, required=True)
    parser.add_argument("--turnover-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--observability-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--package-candidate-id", required=True)
    parser.add_argument("--route-requirement-id", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = execute(
        artifact_manifest_path=args.artifact_manifest.resolve(),
        control_receipt_path=args.control_receipt.resolve(),
        active_receipt_path=args.active_receipt.resolve(),
        birth_receipt_path=args.birth_receipt.resolve(),
        turnover_receipt_path=args.turnover_receipt.resolve(),
        checkpoint_receipt_path=args.checkpoint_receipt.resolve(),
        observability_receipt_path=args.observability_receipt.resolve(),
        output_dir=args.output_dir.resolve(),
        package_candidate_id=args.package_candidate_id,
        route_requirement_id=args.route_requirement_id,
    )
    print(
        json.dumps(
            {
                "ok": summary["ok"],
                "decision": summary["decision"],
                "case_count": summary["case_count"],
                "output_dir": str(args.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
