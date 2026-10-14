#!/usr/bin/env python3
"""Composition CLI for the bounded CRCA PID-less GPU lease migration.

This tool is deliberately separate from Launchpad admission and automatic
reconciliation.  It can consume only the digest-pinned manifest accepted by
``SchedulerLaunchpadAttemptIssuerV1``.  ``render-canonical-manifest`` is a
read-only provenance aid for the original seven host records; it grants no
migration authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.actors.common import canonical_host_actor_state_root
from tools.scheduler.contracts import ActorError, read_json
from tools.scheduler.launchpad_attempts import (
    LEGACY_GPU_MIGRATION_MANIFEST_SCHEMA_VERSION,
    SchedulerLaunchpadAttemptIssuerV1,
    _continuation_event_digest,
    _continuation_lease_digest,
    _continuation_ledger_state,
    _safe_key,
)
from tools.scheduler.programme_contracts import (
    ContentDigest,
    LaunchpadAttemptClaimV1,
    LaunchpadResourceStartLeaseV1,
    canonical_sha256,
)


CANONICAL_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools/scheduler/migrations/crca_pidless_gpu_started_20260827_v1.json"
)
CANONICAL_ROWS = {
    "LPRSTART-0821AC1B00E620DC015602D2": "SEALED_PROCESS_CONTEXT",
    "LPRSTART-ADC91A35045464D12164591B": "TRUE_PIDLESS_DEVICE_IDLE",
    "LPRSTART-6D9EDFA9DD8AF11F7C5FAEAC": "SEALED_PROCESS_CONTEXT",
    "LPRSTART-698230340528484EC507F181": "SEALED_PROCESS_CONTEXT",
    "LPRSTART-2EB05148123296D1190A16FB": "SEALED_PROCESS_CONTEXT",
    "LPRSTART-6D52F3BABC922E17D8236F3A": "TRUE_PIDLESS_DEVICE_IDLE",
    "LPRSTART-5BB1DB292DA2C0641A8DC6F1": "SEALED_PROCESS_CONTEXT",
}
CANONICAL_DEVICE_UUID = "7f52eb4f-61e3-8c2b-d463-eadb4254b011"
MIGRATION_ID = "MIG-CRCA-PIDLESS-GPU-STARTED-20260827-V1"
CUTOVER_SOURCE_COMMIT = "9a778ade1a576d552077673277d467c0ba6d3f70"
OPERATOR_EXCLUSIVE_HOST_ATTESTATION = {
    "schema_version": "eiel_operator_exclusive_host_attestation_v1",
    "attestation_id": "OPERATOR-EXCLUSIVE-CPU-GPU-20260827",
    "operator_statement": (
        "i have stopped all other branches, the CPU/gpu are yours alone."
    ),
    "scope": "one-time canonical legacy GPU lease cutover",
}


def _manifest(path: Path) -> dict[str, Any]:
    return read_json(path.expanduser().resolve(), name="LegacyGpuMigrationManifest")


def _digest_dict(value: Mapping[str, Any]) -> ContentDigest:
    return ContentDigest(
        algorithm=str(value.get("algorithm") or ""),
        hex=str(value.get("hex") or ""),
    )


def render_canonical_manifest() -> dict[str, Any]:
    """Reconstruct the exact seven-row source manifest without writing state."""

    state_root = canonical_host_actor_state_root().resolve()
    issuer = SchedulerLaunchpadAttemptIssuerV1(state_root)
    rows: list[dict[str, Any]] = []
    observed_devices: set[str] = set()
    for lease_id, classification in CANONICAL_ROWS.items():
        lease_path = issuer.scheduler / "resource-start-leases" / f"{lease_id}.json"
        lease_raw = read_json(lease_path, name="LegacyResourceStartLease")
        resource = LaunchpadResourceStartLeaseV1.from_dict(lease_raw)
        attempt_id = resource.attempt_id
        bundle = issuer._authoritative_bundle(attempt_id)
        claim = LaunchpadAttemptClaimV1.from_dict(dict(bundle.get("claim") or {}))
        effect = dict(bundle.get("start_effect") or {})
        resource_events = issuer.resource_ledger.events(lease_id)
        if (
            effect.get("schema_version") != "crca_launchpad_attempt_start_effect_v1"
            or len(resource_events) != 1
            or resource_events[0].get("code") != "STARTED"
            or dict(bundle.get("resource_start_lease") or {}) != resource.to_dict()
            or resource.resource_ids != (f"gpu:{claim.device_id}",)
        ):
            raise ActorError(
                f"canonical migration source row changed: {lease_id}",
                code="launchpad_legacy_gpu_migration_denied",
            )
        observed_devices.add(claim.device_id)
        leases, events, _head = issuer._continuation_state(
            claim.lineage_scope_digest
        )
        attempt_leases = [
            row for row in leases if row.get("attempt_id") == attempt_id
        ]
        if len(attempt_leases) != 1:
            raise ActorError(
                f"canonical migration continuation changed: {lease_id}",
                code="launchpad_legacy_gpu_migration_denied",
            )
        continuation = attempt_leases[0]
        continuation_digest = _continuation_lease_digest(continuation)
        expired_event = next(
            (
                event
                for event in events
                if event.get("code") == "EXPIRED"
                and _digest_dict(event["lease_digest"]) == continuation_digest
            ),
            None,
        )
        successor_event = next(
            (
                event
                for event in events
                if event.get("code") == "RECLAIMED"
                and _digest_dict(event["lease_digest"]) == continuation_digest
            ),
            None,
        )
        state = _continuation_ledger_state(leases, events).value
        fence_code = "ACTIVE" if expired_event is None else "EXPIRED"
        if (
            (fence_code == "ACTIVE" and state != "ACTIVE")
            or (classification == "TRUE_PIDLESS_DEVICE_IDLE" and fence_code != "EXPIRED")
        ):
            raise ActorError(
                f"canonical migration fence changed: {lease_id}",
                code="launchpad_legacy_gpu_migration_denied",
            )

        context: dict[str, Any] | None = None
        run_root = Path(claim.run_root).expanduser().resolve()
        context_paths = (
            sorted(run_root.rglob("workhorse-parent-context.json"))
            if run_root.is_dir()
            else []
        )
        if classification == "SEALED_PROCESS_CONTEXT":
            if len(context_paths) != 1:
                raise ActorError(
                    f"canonical sealed context changed: {lease_id}",
                    code="launchpad_legacy_gpu_migration_denied",
                )
            context_path = context_paths[0]
            context_value = read_json(
                context_path, name="LegacyWorkhorseParentContext"
            )
            context = {
                "relative_path": str(context_path.relative_to(run_root)),
                "file_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
                "schema_version": context_value.get("schema_version"),
                "binding_sha256": context_value.get("binding_sha256"),
                "scheduler_bundle_digest": context_value.get(
                    "scheduler_bundle_digest"
                ),
                "owner_pid": context_value.get("owner_pid"),
                "owner_process_start_ticks": context_value.get(
                    "owner_process_start_ticks"
                ),
            }
            if (
                context_value.get("attempt_id") != attempt_id
                or context_value.get("run_id") != claim.run_id
                or context["scheduler_bundle_digest"]
                != canonical_sha256(bundle).hex
            ):
                raise ActorError(
                    f"canonical sealed context binding changed: {lease_id}",
                    code="launchpad_legacy_gpu_migration_denied",
                )
        elif context_paths:
            raise ActorError(
                f"canonical true PID-less row unexpectedly has a context: {lease_id}",
                code="launchpad_legacy_gpu_migration_denied",
            )

        row_core = {
            "classification": classification,
            "attempt_id": attempt_id,
            "run_id": claim.run_id,
            "claim_digest": claim.claim_digest().to_dict(),
            "start_effect_digest": canonical_sha256(effect).to_dict(),
            "authoritative_bundle_digest": canonical_sha256(bundle).to_dict(),
            "resource_lease_id": resource.lease_id,
            "resource_lease_digest": resource.lease_digest().to_dict(),
            "resource_ids": list(resource.resource_ids),
            "resource_started_event_digest": canonical_sha256(
                resource_events[0]
            ).to_dict(),
            "resource_expires_at_utc": resource.expires_at_utc,
            "lineage_scope_digest": claim.lineage_scope_digest.to_dict(),
            "attempt_continuation_lease_digest": continuation_digest.to_dict(),
            "continuation_fence_code": fence_code,
            "continuation_terminal_event_digest": (
                None
                if expired_event is None
                else _continuation_event_digest(expired_event).to_dict()
            ),
            "continuation_successor_event_digest": (
                None
                if successor_event is None
                else _continuation_event_digest(successor_event).to_dict()
            ),
            "device_uuid": claim.device_id,
            "sealed_process_context": context,
        }
        rows.append({**row_core, "row_sha256": canonical_sha256(row_core).hex})
    if observed_devices != {CANONICAL_DEVICE_UUID}:
        raise ActorError(
            "canonical migration spans an unexpected device",
            code="launchpad_legacy_gpu_migration_denied",
        )
    core = {
        "schema_version": LEGACY_GPU_MIGRATION_MANIFEST_SCHEMA_VERSION,
        "migration_id": MIGRATION_ID,
        "canonical_state_root": str(state_root),
        "device_uuid": CANONICAL_DEVICE_UUID,
        "cutover_commit": CUTOVER_SOURCE_COMMIT,
        "eligible_row_count": len(rows),
        "operator_exclusive_host_attestation": OPERATOR_EXCLUSIVE_HOST_ATTESTATION,
        "rows": rows,
    }
    return {**core, "manifest_sha256": canonical_sha256(core).hex}


def _proof_for_row(
    *,
    issuer: SchedulerLaunchpadAttemptIssuerV1,
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    bundle = issuer._authoritative_bundle(str(row["attempt_id"]))
    resource = LaunchpadResourceStartLeaseV1.from_dict(
        dict(bundle.get("resource_start_lease") or {})
    )
    manifest_digest = ContentDigest(
        algorithm="sha256", hex=str(manifest["manifest_sha256"])
    )
    proof = issuer._observe_legacy_gpu_migration_row_v1(
        manifest_digest=manifest_digest,
        row=row,
        resource=resource,
        repo_root=repo_root,
    )
    issuer._validate_legacy_gpu_migration_proof_v1(
        manifest_digest=manifest_digest,
        row=row,
        attempt_id=str(row["attempt_id"]),
        proof=proof,
    )
    return proof


def inspect_manifest(
    issuer: SchedulerLaunchpadAttemptIssuerV1, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    digest, rows = issuer._validate_legacy_gpu_migration_manifest_v1(manifest)
    statuses: list[dict[str, Any]] = []
    for row in rows:
        events = issuer.resource_ledger.events(str(row["resource_lease_id"]))
        proof_dir = (
            issuer.legacy_gpu_migration_proofs_dir
            / _safe_key(str(row["attempt_id"]))
        )
        proof_count = (
            len(list(proof_dir.glob("*.json"))) if proof_dir.is_dir() else 0
        )
        receipt_dir = (
            issuer.legacy_gpu_migration_receipts_dir
            / _safe_key(str(row["attempt_id"]))
        )
        statuses.append(
            {
                "attempt_id": row["attempt_id"],
                "resource_lease_id": row["resource_lease_id"],
                "classification": row["classification"],
                "resource_event_codes": [event.get("code") for event in events],
                "proof_stored": proof_count > 0,
                "proof_count": proof_count,
                "receipt_count": len(list(receipt_dir.glob("*.json")))
                if receipt_dir.is_dir()
                else 0,
            }
        )
    return {
        "schema_version": "crca_legacy_gpu_migration_inspection_v1",
        "manifest_digest": digest.to_dict(),
        "row_count": len(rows),
        "rows": statuses,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("render-canonical-manifest")
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST_PATH)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument(
        "--manifest", type=Path, default=CANONICAL_MANIFEST_PATH
    )
    preflight_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    apply_parser = subparsers.add_parser("apply")
    apply_parser.add_argument("--manifest", type=Path, default=CANONICAL_MANIFEST_PATH)
    selection = apply_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--attempt-id")
    selection.add_argument("--all", action="store_true")
    apply_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "render-canonical-manifest":
        print(json.dumps(render_canonical_manifest(), indent=2, sort_keys=True))
        return 0
    state_root = canonical_host_actor_state_root().resolve()
    issuer = SchedulerLaunchpadAttemptIssuerV1(state_root)
    manifest = _manifest(args.manifest)
    if args.command == "inspect":
        print(json.dumps(inspect_manifest(issuer, manifest), indent=2, sort_keys=True))
        return 0
    _digest, rows = issuer._validate_legacy_gpu_migration_manifest_v1(manifest)
    if args.command == "preflight":
        outcomes = []
        for row in rows:
            proof = _proof_for_row(
                issuer=issuer,
                manifest=manifest,
                row=row,
                repo_root=args.repo_root.expanduser().resolve(),
            )
            outcomes.append(
                {
                    "attempt_id": row["attempt_id"],
                    "resource_lease_id": row["resource_lease_id"],
                    "classification": row["classification"],
                    "decision": proof.get("decision"),
                    "reason_codes": proof.get("reason_codes"),
                    "proof_sha256": proof.get("proof_sha256"),
                }
            )
        print(
            json.dumps(
                {
                    "schema_version": "crca_legacy_gpu_migration_preflight_v1",
                    "outcomes": outcomes,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if all(row["decision"] == "RECLAIM" for row in outcomes) else 2
    selected = (
        rows
        if args.all
        else [row for row in rows if row.get("attempt_id") == args.attempt_id]
    )
    if len(selected) != (len(rows) if args.all else 1):
        raise ActorError(
            "requested attempt is absent from the pinned migration manifest",
            code="launchpad_legacy_gpu_migration_denied",
        )
    outcomes: list[dict[str, Any]] = []
    for row in selected:
        proof = _proof_for_row(
            issuer=issuer,
            manifest=manifest,
            row=row,
            repo_root=args.repo_root.expanduser().resolve(),
        )
        if proof.get("decision") != "RECLAIM":
            outcomes.append(
                {
                    "attempt_id": row["attempt_id"],
                    "status": "WAIT",
                    "reason_codes": proof.get("reason_codes"),
                }
            )
            continue
        outcomes.append(
            issuer.apply_legacy_gpu_migration_row_v1(
                manifest=manifest,
                attempt_id=str(row["attempt_id"]),
                repo_root=args.repo_root.expanduser().resolve(),
            )
        )
    print(json.dumps({"outcomes": outcomes}, indent=2, sort_keys=True))
    return 0 if all(row.get("status") != "WAIT" for row in outcomes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
