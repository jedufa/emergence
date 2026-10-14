"""Compact evidence certificates and explicit gate-boundary reclamation.

The certificate is a deterministic reduction and release precondition.  It is
not an execution, review, qualification, or destructive-cleanup authority.
Exact release rows are compiled into the existing artifact disposition
manifest and consumed by the existing prune actuator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Final

from .artifact_lifecycle import (
    ArtifactLifecycleError,
    DEBUG_FORENSIC,
    RunStorageGuard,
    SCIENCE_COMPACT,
    STORAGE_RECEIPT_SCHEMA_VERSION,
    build_artifact_disposition_manifest_v1,
    build_sealed_artifact_inventory_row_v1,
    validate_artifact_disposition_manifest_v1,
    validate_run_artifact_lifecycle_contract_v1,
)
from .canopy_supervision_v1 import (
    CANOPY_SUPERVISION_SCHEMA_VERSION,
    canopy_relay_process_matches_v1,
)
from .canonical import canonical_sha256, pretty_json
from .storage_availability import DATA2_FILESYSTEM_ROOT

REQUEST_SCHEMA_VERSION: Final[str] = "eiel_evidence_certificate_request_v1"
CERTIFICATE_SCHEMA_VERSION: Final[str] = "eiel_evidence_certificate_v1"
INTENT_SCHEMA_VERSION: Final[str] = "eiel_evidence_reclamation_intent_v1"
HISTORICAL_AUDIT_SCHEMA_VERSION: Final[str] = (
    "eiel_evidence_certificate_historical_audit_v1"
)

EVIDENCE_CLASSES: Final[tuple[str, ...]] = (
    "OPERATIONAL_STATE",
    "RAW_TRANSIENT_EVIDENCE",
    "DURABLE_EVIDENCE_CERTIFICATE",
    "SCIENTIFIC_ENGINEERING_ASSET",
    "FORENSIC_HOLD",
)
DISPOSITIONS: Final[tuple[str, ...]] = (
    "RELEASE_NOW",
    "HOLD_UNTIL_LAST_USE",
    "FORENSIC_HOLD",
    "RETAIN_ASSET",
    "RETAIN_DURABLE",
    "UNKNOWN_POLICY",
)
RETENTION_AUTHORITIES: Final[tuple[str, ...]] = (
    "audit_exact_bytes",
    "durable_provenance",
    "qualification_exact_bytes",
    "replay_exact_bytes",
)
DECISIONS: Final[tuple[str, ...]] = (
    "ACCEPTED",
    "REJECTED",
    "VALID_NEGATIVE",
    "NO_SUPPORTED_RESULT",
    "RAW_DIAGNOSTIC_ABANDONED",
)
REPRODUCIBILITY_GUARANTEES: Final[tuple[str, ...]] = (
    "BIT_IDENTITY",
    "SEMANTIC_EQUIVALENCE",
    "REVALIDATION_ONLY",
)
DIAGNOSTIC_CLOSEOUT_DISPOSITIONS: Final[tuple[str, ...]] = (
    "RETAIN",
    "SUPERSEDED_BY_CERTIFICATE",
)
RETENTION_CLASSES: Final[tuple[str, ...]] = (
    "R0",
    "R1",
    "R2",
    "R3",
    "R4",
    "R5",
    "RX",
)
MAX_SOURCE_BINDING_MANIFESTS: Final[int] = 64
MAX_SOURCE_BINDING_ROWS: Final[int] = 16_384
DIAGNOSTIC_UPGRADE_SPEC_SCHEMA_VERSION: Final[str] = (
    "eiel_evidence_certificate_diagnostic_upgrade_spec_v1"
)
DIAGNOSTIC_SOURCE_LIVE_MARKER_CLOSEOUT_SCHEMA_VERSION: Final[str] = (
    "eiel_debug_source_live_marker_closeout_v1"
)
MAX_DIAGNOSTIC_LIVE_AUDIT_AGE_SECONDS: Final[int] = 60 * 60
MAX_DIAGNOSTIC_LIVE_AUDIT_FUTURE_SKEW_SECONDS: Final[int] = 5 * 60
DIAGNOSTIC_LOCAL_LIVE_STATE_AUDIT_SCHEMA_VERSION: Final[str] = (
    "eiel_diagnostic_local_live_state_audit_v1"
)
MAX_DIAGNOSTIC_PROCESS_ROWS: Final[int] = 131_072
MAX_DIAGNOSTIC_PROCESS_FDS: Final[int] = 16_384
MAX_DIAGNOSTIC_RELAY_ROWS: Final[int] = 4_096
MAX_DIAGNOSTIC_RELAY_RECEIPT_BYTES: Final[int] = 1024 * 1024
DIAGNOSTIC_SANDBOXED_NON_RUNTIME_PROCESS_NAMES: Final[frozenset[str]] = frozenset(
    {
        "systemd",
        "(sd-pam)",
        "ssh-agent",
        "sd_espeak-ng-mb",
        "chrome-sandbox",
        "Proton Mail Bet",
        "Proton Pass",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*\Z")


class EvidenceCertificateError(ArtifactLifecycleError):
    """An evidence certificate request, certificate, or intent is invalid."""


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceCertificateError(f"{path} must be an object")
    return dict(value)


def _text(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceCertificateError(f"{path} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, *, path: str) -> str:
    result = _text(value, path=path)
    if _ID_RE.fullmatch(result) is None:
        raise EvidenceCertificateError(f"{path} must be a stable identifier")
    return result


def validate_diagnostic_source_live_marker_closeout_v1(
    value: Any,
    *,
    selected_artifact_ids: Sequence[str],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate an exact selected-set/current-source stale-marker closure."""

    raw = _mapping(value, path="diagnostic source live-marker closeout")
    expected = {
        "schema_version",
        "status",
        "audited_at_utc",
        "source_manifest_sha256",
        "source_campaign_id",
        "source_run_id",
        "source_runtime_root",
        "selected_artifact_count",
        "selected_artifact_ids_sha256",
        "checks",
        "scope",
        "audit_method",
    }
    if set(raw) != expected:
        raise EvidenceCertificateError(
            "diagnostic source live-marker closeout fields differ"
        )
    selected = sorted(selected_artifact_ids)
    if (
        raw.get("schema_version")
        != DIAGNOSTIC_SOURCE_LIVE_MARKER_CLOSEOUT_SCHEMA_VERSION
        or raw.get("status") != "CLOSED_BY_CURRENT_STATE_AUDIT"
        or raw.get("scope") != "SOURCE_MANIFEST_LIVE_REFERENCE_MARKER_ONLY"
        or raw.get("source_manifest_sha256") != source_manifest.get("manifest_sha256")
        or raw.get("source_campaign_id") != source_manifest.get("campaign_id")
        or raw.get("source_run_id") != source_manifest.get("run_id")
        or raw.get("source_runtime_root") != source_manifest.get("runtime_root")
        or raw.get("selected_artifact_count") != len(selected)
        or raw.get("selected_artifact_ids_sha256") != canonical_sha256(selected)
    ):
        raise EvidenceCertificateError(
            "diagnostic source live-marker closeout authority binding differs"
        )
    audited_at = _text(
        raw.get("audited_at_utc"),
        path="diagnostic source live-marker closeout.audited_at_utc",
    )
    try:
        parsed = datetime.fromisoformat(audited_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceCertificateError(
            "diagnostic source live-marker audit timestamp is invalid"
        ) from exc
    if not audited_at.endswith("Z") or parsed.utcoffset() != timezone.utc.utcoffset(
        parsed
    ):
        raise EvidenceCertificateError(
            "diagnostic source live-marker audit timestamp is not normalized UTC"
        )
    checks = _mapping(
        raw.get("checks"), path="diagnostic source live-marker closeout.checks"
    )
    boolean_checks = {
        "writer",
        "reference",
        "resume",
        "reducer",
        "consumer",
        "canonical_reservation_overlap",
    }
    if set(checks) != {*boolean_checks, "relay_process_references"}:
        raise EvidenceCertificateError(
            "diagnostic source live-marker current-state checks differ"
        )
    if any(checks.get(name) is not False for name in boolean_checks):
        raise EvidenceCertificateError(
            "diagnostic source live-marker current-state audit remains active"
        )
    relay_references = _string_list(
        checks.get("relay_process_references"),
        path=("diagnostic source live-marker closeout.checks.relay_process_references"),
    )
    if relay_references:
        raise EvidenceCertificateError(
            "diagnostic source live-marker audit retains relay/process references"
        )
    return {
        "schema_version": DIAGNOSTIC_SOURCE_LIVE_MARKER_CLOSEOUT_SCHEMA_VERSION,
        "status": "CLOSED_BY_CURRENT_STATE_AUDIT",
        "audited_at_utc": audited_at,
        "source_manifest_sha256": str(source_manifest["manifest_sha256"]),
        "source_campaign_id": str(source_manifest["campaign_id"]),
        "source_run_id": str(source_manifest["run_id"]),
        "source_runtime_root": str(source_manifest["runtime_root"]),
        "selected_artifact_count": len(selected),
        "selected_artifact_ids_sha256": canonical_sha256(selected),
        "checks": {
            **{name: False for name in sorted(boolean_checks)},
            "relay_process_references": relay_references,
        },
        "scope": "SOURCE_MANIFEST_LIVE_REFERENCE_MARKER_ONLY",
        "audit_method": _text(
            raw.get("audit_method"),
            path="diagnostic source live-marker closeout.audit_method",
        ),
    }


def validate_diagnostic_source_live_marker_freshness_v1(
    value: Mapping[str, Any], *, now: datetime | None = None
) -> None:
    """Require a current audit at certificate build and destructive recheck."""

    audited_at = str(value.get("audited_at_utc") or "")
    try:
        observed = datetime.fromisoformat(audited_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceCertificateError(
            "diagnostic source live-marker audit timestamp is invalid"
        ) from exc
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise EvidenceCertificateError(
            "diagnostic freshness clock must be timezone-aware"
        )
    delta_seconds = (current.astimezone(timezone.utc) - observed).total_seconds()
    if delta_seconds < -MAX_DIAGNOSTIC_LIVE_AUDIT_FUTURE_SKEW_SECONDS:
        raise EvidenceCertificateError(
            "diagnostic source live-marker audit timestamp is in the future"
        )
    if delta_seconds > MAX_DIAGNOSTIC_LIVE_AUDIT_AGE_SECONDS:
        raise EvidenceCertificateError(
            "diagnostic source live-marker current-state audit is stale"
        )


def _path_references_root(value: str, *, root_text: str) -> bool:
    return value == root_text or value.startswith(root_text + os.sep)


def diagnostic_local_live_state_audit_v1(evidence_root: Path) -> dict[str, Any]:
    """Observe exact-root reservations, same-UID processes, and recorded relays.

    This is a bounded local observation, not a filesystem discovery pass.  It
    opens only ``/proc`` and the exact run-local Canopy supervision pathname.
    """

    root = evidence_root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise EvidenceCertificateError(
            "diagnostic local live-state audit requires a real evidence root"
        )
    root_text = str(root)
    process_references: list[dict[str, Any]] = []
    inspection_errors: list[dict[str, Any]] = []
    proc_root = Path("/proc")
    try:
        pid_paths = sorted(
            (path for path in proc_root.iterdir() if path.name.isdigit()),
            key=lambda path: int(path.name),
        )
    except OSError as exc:
        raise EvidenceCertificateError(
            "diagnostic local process reference inspection is unavailable"
        ) from exc
    if len(pid_paths) > MAX_DIAGNOSTIC_PROCESS_ROWS:
        raise EvidenceCertificateError(
            "diagnostic local process reference inspection exceeds its PID bound"
        )
    for pid_path in pid_paths:
        pid = int(pid_path.name)
        if pid == os.getpid():
            continue
        try:
            if pid_path.stat().st_uid != os.geteuid():
                continue
        except FileNotFoundError:
            continue
        except OSError:
            inspection_errors.append({"pid": pid, "probe": "owner"})
            continue
        reasons: set[str] = set()
        pid_errors: list[dict[str, Any]] = []
        try:
            process_name = (pid_path / "comm").read_text(encoding="utf-8").strip()
        except OSError:
            process_name = ""
        try:
            cwd = os.readlink(pid_path / "cwd")
            if _path_references_root(cwd, root_text=root_text):
                reasons.add("cwd")
        except FileNotFoundError:
            continue
        except OSError:
            pid_errors.append({"pid": pid, "probe": "cwd"})
        try:
            arguments = (pid_path / "cmdline").read_bytes().split(b"\0")
            for argument in arguments:
                try:
                    text = argument.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                candidates = [text]
                if "=" in text:
                    candidates.append(text.split("=", 1)[1])
                if any(
                    _path_references_root(candidate, root_text=root_text)
                    for candidate in candidates
                ):
                    reasons.add("cmdline")
                    break
        except FileNotFoundError:
            continue
        except OSError:
            pid_errors.append({"pid": pid, "probe": "cmdline"})
        try:
            fd_paths = list((pid_path / "fd").iterdir())
        except FileNotFoundError:
            fd_paths = []
        except OSError:
            pid_errors.append({"pid": pid, "probe": "fd_directory"})
            fd_paths = []
        if len(fd_paths) > MAX_DIAGNOSTIC_PROCESS_FDS:
            pid_errors.append({"pid": pid, "probe": "fd_bound"})
            fd_paths = []
        for fd_path in fd_paths:
            try:
                target = os.readlink(fd_path)
            except FileNotFoundError:
                continue
            except OSError:
                pid_errors.append({"pid": pid, "probe": "fd", "fd": fd_path.name})
                continue
            if _path_references_root(target, root_text=root_text):
                reasons.add("fd")
                break
        if reasons:
            process_references.append({"pid": pid, "reasons": sorted(reasons)})
        if (
            pid_errors
            and process_name not in DIAGNOSTIC_SANDBOXED_NON_RUNTIME_PROCESS_NAMES
        ):
            inspection_errors.extend(pid_errors)

    recorded_relay_references: list[dict[str, Any]] = []
    relay_path = root / "canopy_relay_supervision.json"
    if relay_path.exists() or relay_path.is_symlink():
        if relay_path.is_symlink() or not relay_path.is_file():
            inspection_errors.append({"probe": "canopy_supervision_type"})
        else:
            try:
                if relay_path.stat().st_size > MAX_DIAGNOSTIC_RELAY_RECEIPT_BYTES:
                    raise EvidenceCertificateError(
                        "diagnostic Canopy supervision receipt exceeds its byte bound"
                    )
                relay_receipt = load_json_object(relay_path)
                relay_core = dict(relay_receipt)
                relay_sha256 = relay_core.pop("receipt_sha256", None)
                relays = relay_receipt.get("relays")
                if (
                    relay_receipt.get("schema_version")
                    != CANOPY_SUPERVISION_SCHEMA_VERSION
                    or relay_sha256 != canonical_sha256(relay_core)
                    or Path(str(relay_receipt.get("run_root")))
                    .expanduser()
                    .resolve(strict=False)
                    != root
                    or not isinstance(relays, list)
                    or len(relays) > MAX_DIAGNOSTIC_RELAY_ROWS
                    or any(not isinstance(row, Mapping) for row in relays)
                ):
                    raise EvidenceCertificateError(
                        "diagnostic Canopy supervision receipt is invalid or foreign"
                    )
                for row in relays:
                    relay = dict(row)
                    pid = relay.get("pid")
                    if pid is None:
                        continue
                    output_root = relay.get("output_root")
                    if (
                        type(pid) is not int
                        or pid <= 0
                        or not isinstance(output_root, str)
                    ):
                        raise EvidenceCertificateError(
                            "diagnostic Canopy relay row is malformed"
                        )
                    if canopy_relay_process_matches_v1(
                        pid, output_root=Path(output_root)
                    ):
                        recorded_relay_references.append(
                            {
                                "pid": pid,
                                "cell_id": str(relay.get("cell_id") or ""),
                                "output_root": output_root,
                            }
                        )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise EvidenceCertificateError(
                    "diagnostic Canopy supervision receipt cannot be inspected"
                ) from exc

    return {
        "schema_version": DIAGNOSTIC_LOCAL_LIVE_STATE_AUDIT_SCHEMA_VERSION,
        "runtime_root": root_text,
        "canonical_reservation_overlap": RunStorageGuard.run_root_is_reserved(root),
        "process_references": process_references,
        "recorded_relay_references": recorded_relay_references,
        "inspection_errors": inspection_errors,
    }


def _require_diagnostic_local_live_state_clear_v1(evidence_root: Path) -> None:
    audit = diagnostic_local_live_state_audit_v1(evidence_root)
    if audit["canonical_reservation_overlap"]:
        raise EvidenceCertificateError(
            "diagnostic source root retains a canonical storage reservation"
        )
    if audit["inspection_errors"]:
        raise EvidenceCertificateError(
            "diagnostic local process/relay inspection was incomplete: "
            + repr(audit["inspection_errors"][:8])
        )
    references = [
        *audit["process_references"],
        *audit["recorded_relay_references"],
    ]
    if references:
        raise EvidenceCertificateError(
            "diagnostic source root retains local process/relay references: "
            + repr(references[:8])
        )


def require_diagnostic_local_live_state_clear_v1(evidence_root: Path) -> None:
    """Fail closed unless one exact diagnostic root is locally quiescent."""

    _require_diagnostic_local_live_state_clear_v1(evidence_root)


def _sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise EvidenceCertificateError(f"{path} must be a lowercase SHA-256")
    return value


def _bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        raise EvidenceCertificateError(f"{path} must be boolean")
    return value


def _string_list(value: Any, *, path: str, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise EvidenceCertificateError(f"{path} must be an array of strings")
    result = [item.strip() for item in value]
    if len(result) != len(set(result)):
        raise EvidenceCertificateError(f"{path} must not contain duplicates")
    if nonempty and not result:
        raise EvidenceCertificateError(f"{path} must not be empty")
    return result


def _object_list(
    value: Any, *, path: str, nonempty: bool = False
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise EvidenceCertificateError(f"{path} must be an array of objects")
    result = [dict(item) for item in value]
    if nonempty and not result:
        raise EvidenceCertificateError(f"{path} must not be empty")
    return result


def _relative_path(value: Any, *, path: str) -> str:
    raw = _text(value, path=path).replace("\\", "/").strip("/")
    candidate = Path(raw)
    if candidate.is_absolute() or raw in {"", "."} or ".." in candidate.parts:
        raise EvidenceCertificateError(f"{path} must be a safe relative path")
    return raw


def _validate_identity(value: Any) -> dict[str, Any]:
    identity = _mapping(value, path="request.identity")
    source = _mapping(identity.get("source"), path="request.identity.source")
    source = {
        "repository": _text(
            source.get("repository"), path="request.identity.source.repository"
        ),
        "commit_sha": _text(
            source.get("commit_sha"), path="request.identity.source.commit_sha"
        ),
    }
    if not re.fullmatch(r"[0-9a-f]{40,64}", source["commit_sha"]):
        raise EvidenceCertificateError(
            "request.identity.source.commit_sha must be a lowercase Git/content identity"
        )
    artifacts = _mapping(
        identity.get("artifact_hashes", {}), path="request.identity.artifact_hashes"
    )
    artifact_hashes = {
        _identifier(name, path="request.identity.artifact_hashes key"): _sha256(
            digest, path=f"request.identity.artifact_hashes.{name}"
        )
        for name, digest in sorted(artifacts.items())
    }
    environment = _mapping(
        identity.get("environment"), path="request.identity.environment"
    )
    device = _mapping(identity.get("device"), path="request.identity.device")
    if not environment or not device:
        raise EvidenceCertificateError(
            "request identity requires non-empty environment and device descriptions"
        )
    rng_bindings = _object_list(
        identity.get("rng_bindings", []), path="request.identity.rng_bindings"
    )
    rng_not_applicable_reason = identity.get("rng_not_applicable_reason")
    if not rng_bindings and not (
        isinstance(rng_not_applicable_reason, str) and rng_not_applicable_reason.strip()
    ):
        raise EvidenceCertificateError(
            "request identity requires RNG bindings or rng_not_applicable_reason"
        )
    return {
        "source": source,
        "artifact_hashes": artifact_hashes,
        "runtime": _mapping(identity.get("runtime"), path="request.identity.runtime"),
        "config_sha256": _sha256(
            identity.get("config_sha256"), path="request.identity.config_sha256"
        ),
        "request_sha256": _sha256(
            identity.get("request_sha256"), path="request.identity.request_sha256"
        ),
        "plan_sha256": _sha256(
            identity.get("plan_sha256"), path="request.identity.plan_sha256"
        ),
        "environment": environment,
        "device": device,
        "rng_bindings": rng_bindings,
        "rng_not_applicable_reason": (
            str(rng_not_applicable_reason).strip()
            if isinstance(rng_not_applicable_reason, str)
            else None
        ),
    }


def _validate_programme_plan(
    value: Any, *, programme_id: str, gate_id: str
) -> dict[str, Any]:
    plan = _mapping(value, path="request.programme_plan")
    ordered = _string_list(
        plan.get("ordered_gates"),
        path="request.programme_plan.ordered_gates",
        nonempty=True,
    )
    closed = _string_list(
        plan.get("closed_gates"), path="request.programme_plan.closed_gates"
    )
    if gate_id not in ordered or gate_id not in closed:
        raise EvidenceCertificateError(
            "certificate gate must exist in ordered_gates and be closed at this boundary"
        )
    if any(gate not in ordered for gate in closed):
        raise EvidenceCertificateError(
            "closed_gates contains an unknown programme gate"
        )
    gate_position = ordered.index(gate_id)
    expected_closed = ordered[: gate_position + 1]
    if closed != expected_closed:
        raise EvidenceCertificateError(
            "closed_gates must be the ordered closure prefix through the current gate"
        )
    plan_id = _identifier(plan.get("plan_id"), path="request.programme_plan.plan_id")
    plan_sha256 = _sha256(
        plan.get("plan_sha256"), path="request.programme_plan.plan_sha256"
    )
    plan_identity = {
        "programme_id": programme_id,
        "plan_id": plan_id,
        "ordered_gates": ordered,
    }
    if plan_sha256 != canonical_sha256(plan_identity):
        raise EvidenceCertificateError(
            "programme_plan.plan_sha256 does not bind the declared ordered gate graph"
        )
    closure_receipt = _mapping(
        plan.get("closure_receipt"),
        path="request.programme_plan.closure_receipt",
    )
    supplied_receipt_sha256 = closure_receipt.pop("receipt_sha256", None)
    expected_receipt = {
        "schema_version": "eiel_programme_gate_closure_receipt_v1",
        "programme_id": programme_id,
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "gate_id": gate_id,
        "closed_gates": closed,
        "status": "CLOSED",
        "decision_receipt_sha256": _sha256(
            closure_receipt.get("decision_receipt_sha256"),
            path=("request.programme_plan.closure_receipt.decision_receipt_sha256"),
        ),
    }
    if closure_receipt != expected_receipt or (
        supplied_receipt_sha256 != canonical_sha256(expected_receipt)
    ):
        raise EvidenceCertificateError(
            "programme gate closure receipt is stale or semantically invalid"
        )
    declared_receipt_sha256 = _sha256(
        plan.get("closure_receipt_sha256"),
        path="request.programme_plan.closure_receipt_sha256",
    )
    if declared_receipt_sha256 != supplied_receipt_sha256:
        raise EvidenceCertificateError(
            "programme plan closure receipt hash does not match its authority"
        )
    return {
        "plan_id": plan_id,
        "plan_sha256": plan_sha256,
        "closure_receipt_sha256": declared_receipt_sha256,
        "closure_receipt": {
            **expected_receipt,
            "receipt_sha256": supplied_receipt_sha256,
        },
        "ordered_gates": ordered,
        "closed_gates": closed,
        "remaining_gates": [gate for gate in ordered if gate not in closed],
    }


def _validate_live_state(value: Any, *, index: int) -> dict[str, bool]:
    raw = _mapping(value, path=f"request.artifacts[{index}].live_state")
    names = ("writer", "reference", "resume", "reducer", "consumer")
    return {
        name: _bool(raw.get(name), path=f"request.artifacts[{index}].live_state.{name}")
        for name in names
    }


def _validate_forensic_hold(value: Any, *, index: int) -> dict[str, Any]:
    raw = _mapping(value, path=f"request.artifacts[{index}].forensic_hold")
    status = _text(
        raw.get("status"), path=f"request.artifacts[{index}].forensic_hold.status"
    )
    if status not in {"active", "closed", "not_applicable"}:
        raise EvidenceCertificateError("forensic hold status is unsupported")
    receipt = raw.get("closure_receipt_sha256")
    if status == "closed":
        receipt = _sha256(
            receipt,
            path=f"request.artifacts[{index}].forensic_hold.closure_receipt_sha256",
        )
    elif receipt is not None:
        receipt = _sha256(
            receipt,
            path=f"request.artifacts[{index}].forensic_hold.closure_receipt_sha256",
        )
    return {"status": status, "closure_receipt_sha256": receipt}


def evidence_root_requires_permanent_capsule_v1(evidence_root: Path) -> bool:
    root = evidence_root.expanduser().resolve(strict=False)
    return root == DATA2_FILESYSTEM_ROOT or DATA2_FILESYSTEM_ROOT in root.parents


def _validate_source_manifest_binding(value: Any, *, index: int) -> dict[str, Any]:
    raw = _mapping(value, path=f"request.artifacts[{index}].source_manifest_binding")
    base_fields = {
        "manifest_path",
        "manifest_sha256",
        "artifact_id",
        "bytes",
        "retention_class",
    }
    digest_fields = {name for name in ("sha256", "tree_sha256") if name in raw}
    if set(raw) != base_fields | digest_fields or len(digest_fields) != 1:
        raise EvidenceCertificateError(
            f"request.artifacts[{index}].source_manifest_binding must contain exact manifest/artifact/bytes/class identity and one digest"
        )
    size = raw.get("bytes")
    if type(size) is not int or size < 0:
        raise EvidenceCertificateError(
            f"request.artifacts[{index}].source_manifest_binding.bytes must be non-negative"
        )
    retention_class = _text(
        raw.get("retention_class"),
        path=f"request.artifacts[{index}].source_manifest_binding.retention_class",
    )
    if retention_class not in RETENTION_CLASSES:
        raise EvidenceCertificateError(
            f"request.artifacts[{index}].source_manifest_binding retention class is unsupported"
        )
    digest_name = next(iter(digest_fields))
    return {
        "manifest_path": _relative_path(
            raw.get("manifest_path"),
            path=f"request.artifacts[{index}].source_manifest_binding.manifest_path",
        ),
        "manifest_sha256": _sha256(
            raw.get("manifest_sha256"),
            path=f"request.artifacts[{index}].source_manifest_binding.manifest_sha256",
        ),
        "artifact_id": _identifier(
            raw.get("artifact_id"),
            path=f"request.artifacts[{index}].source_manifest_binding.artifact_id",
        ),
        "bytes": size,
        "retention_class": retention_class,
        digest_name: _sha256(
            raw.get(digest_name),
            path=f"request.artifacts[{index}].source_manifest_binding.{digest_name}",
        ),
    }


def _validate_legacy_partial_source_manifest_binding(
    value: Any, *, index: int
) -> dict[str, Any]:
    """Accept only the exact pre-manifest-binding row used by older requests."""

    raw = _mapping(
        value,
        path=f"request.artifacts[{index}].source_manifest_binding",
    )
    base_fields = {"artifact_id", "bytes", "retention_class"}
    digest_fields = {name for name in ("sha256", "tree_sha256") if name in raw}
    if set(raw) != base_fields | digest_fields or len(digest_fields) != 1:
        raise EvidenceCertificateError(
            f"request.artifacts[{index}].source_manifest_binding is neither a complete binding nor the exact legacy partial shape"
        )
    size = raw.get("bytes")
    if type(size) is not int or size < 0:
        raise EvidenceCertificateError(
            f"request.artifacts[{index}].source_manifest_binding.bytes must be non-negative"
        )
    retention_class = _text(
        raw.get("retention_class"),
        path=f"request.artifacts[{index}].source_manifest_binding.retention_class",
    )
    if retention_class not in RETENTION_CLASSES:
        raise EvidenceCertificateError(
            f"request.artifacts[{index}].source_manifest_binding retention class is unsupported"
        )
    digest_name = next(iter(digest_fields))
    return {
        "artifact_id": _identifier(
            raw.get("artifact_id"),
            path=f"request.artifacts[{index}].source_manifest_binding.artifact_id",
        ),
        "bytes": size,
        "retention_class": retention_class,
        digest_name: _sha256(
            raw.get(digest_name),
            path=f"request.artifacts[{index}].source_manifest_binding.{digest_name}",
        ),
    }


def _validate_diagnostic_closeout(value: Any) -> dict[str, Any]:
    """Validate the exact, fail-closed authority needed to close DEBUG_FORENSIC."""

    raw = _mapping(value, path="request.diagnostic_closeout")
    expected = {
        "source_manifest",
        "substantive_review",
        "closure_receipt_sha256",
        "conclusion",
        "limitations",
        "unresolved_anomalies",
        "live_state",
        "active_exact_byte_holds",
        "superseded_artifact_ids",
    }
    allowed_fields = (expected, expected | {"source_live_marker_closeout"})
    if set(raw) not in allowed_fields:
        raise EvidenceCertificateError(
            "request.diagnostic_closeout must contain exactly: "
            + ",".join(sorted(expected))
            + " and optional source_live_marker_closeout"
        )
    source = _mapping(
        raw.get("source_manifest"),
        path="request.diagnostic_closeout.source_manifest",
    )
    source_fields = {
        "path",
        "manifest_sha256",
        "campaign_id",
        "run_id",
        "plan_sha256",
        "lifecycle_contract_sha256",
    }
    source_live_marker_supplied = "source_live_marker_closeout" in raw
    if set(source) != (
        source_fields | {"runtime_root"}
        if source_live_marker_supplied
        else source_fields
    ):
        raise EvidenceCertificateError(
            "diagnostic source_manifest must contain exact path, manifest, campaign, run, plan, and lifecycle identities"
        )
    runtime_root: str | None = None
    if source_live_marker_supplied:
        runtime_root = _text(
            source.get("runtime_root"),
            path="request.diagnostic_closeout.source_manifest.runtime_root",
        )
        runtime_path = Path(runtime_root).expanduser()
        if (
            not runtime_path.is_absolute()
            or str(runtime_path.resolve(strict=False)) != runtime_root
        ):
            raise EvidenceCertificateError(
                "diagnostic source_manifest runtime_root must be an exact normalized absolute path"
            )
    normalized_source = {
        "path": _relative_path(
            source.get("path"),
            path="request.diagnostic_closeout.source_manifest.path",
        ),
        "manifest_sha256": _sha256(
            source.get("manifest_sha256"),
            path="request.diagnostic_closeout.source_manifest.manifest_sha256",
        ),
        "campaign_id": _text(
            source.get("campaign_id"),
            path="request.diagnostic_closeout.source_manifest.campaign_id",
        ),
        "run_id": _text(
            source.get("run_id"),
            path="request.diagnostic_closeout.source_manifest.run_id",
        ),
        "plan_sha256": _sha256(
            source.get("plan_sha256"),
            path="request.diagnostic_closeout.source_manifest.plan_sha256",
        ),
        "lifecycle_contract_sha256": _sha256(
            source.get("lifecycle_contract_sha256"),
            path=(
                "request.diagnostic_closeout.source_manifest.lifecycle_contract_sha256"
            ),
        ),
        **({"runtime_root": runtime_root} if runtime_root is not None else {}),
    }
    review = _mapping(
        raw.get("substantive_review"),
        path="request.diagnostic_closeout.substantive_review",
    )
    if set(review) != {"status", "reviewer", "receipt_sha256"}:
        raise EvidenceCertificateError(
            "diagnostic substantive_review must contain exactly status, reviewer, and receipt_sha256"
        )
    status = _text(
        review.get("status"),
        path="request.diagnostic_closeout.substantive_review.status",
    )
    if status not in {"accepted", "accepted_with_limitations"}:
        raise EvidenceCertificateError("diagnostic substantive review must be accepted")
    anomalies = _string_list(
        raw.get("unresolved_anomalies"),
        path="request.diagnostic_closeout.unresolved_anomalies",
    )
    if anomalies:
        raise EvidenceCertificateError(
            "DEBUG_FORENSIC closeout cannot retain unresolved anomalies"
        )
    live = _mapping(
        raw.get("live_state"), path="request.diagnostic_closeout.live_state"
    )
    live_names = {"writer", "reference", "resume", "reducer", "consumer"}
    if set(live) != live_names:
        raise EvidenceCertificateError(
            "diagnostic closeout live_state must account for every live role"
        )
    normalized_live = {
        name: _bool(
            live.get(name), path=f"request.diagnostic_closeout.live_state.{name}"
        )
        for name in sorted(live_names)
    }
    active_live = sorted(name for name, active in normalized_live.items() if active)
    if active_live:
        raise EvidenceCertificateError(
            "DEBUG_FORENSIC closeout has live state: " + ",".join(active_live)
        )
    holds = _mapping(
        raw.get("active_exact_byte_holds"),
        path="request.diagnostic_closeout.active_exact_byte_holds",
    )
    hold_names = {"forensic", "audit", "qualification", "replay"}
    if set(holds) != hold_names:
        raise EvidenceCertificateError(
            "diagnostic closeout must account for forensic, audit, qualification, and replay exact-byte holds"
        )
    normalized_holds = {
        name: _string_list(
            holds.get(name),
            path=f"request.diagnostic_closeout.active_exact_byte_holds.{name}",
        )
        for name in sorted(hold_names)
    }
    active_holds = sorted(
        f"{name}:{item}" for name, values in normalized_holds.items() for item in values
    )
    if active_holds:
        raise EvidenceCertificateError(
            "DEBUG_FORENSIC closeout has active exact-byte holds: "
            + ",".join(active_holds)
        )
    superseded_ids = _string_list(
        raw.get("superseded_artifact_ids"),
        path="request.diagnostic_closeout.superseded_artifact_ids",
        nonempty=True,
    )
    source_live_marker_closeout = (
        validate_diagnostic_source_live_marker_closeout_v1(
            raw.get("source_live_marker_closeout"),
            selected_artifact_ids=superseded_ids,
            source_manifest=normalized_source,
        )
        if "source_live_marker_closeout" in raw
        else None
    )
    return {
        "source_manifest": normalized_source,
        "substantive_review": {
            "status": status,
            "reviewer": _text(
                review.get("reviewer"),
                path="request.diagnostic_closeout.substantive_review.reviewer",
            ),
            "receipt_sha256": _sha256(
                review.get("receipt_sha256"),
                path=("request.diagnostic_closeout.substantive_review.receipt_sha256"),
            ),
        },
        "closure_receipt_sha256": _sha256(
            raw.get("closure_receipt_sha256"),
            path="request.diagnostic_closeout.closure_receipt_sha256",
        ),
        "conclusion": _text(
            raw.get("conclusion"), path="request.diagnostic_closeout.conclusion"
        ),
        "limitations": _string_list(
            raw.get("limitations"),
            path="request.diagnostic_closeout.limitations",
            nonempty=True,
        ),
        "unresolved_anomalies": anomalies,
        "live_state": normalized_live,
        "active_exact_byte_holds": normalized_holds,
        "superseded_artifact_ids": superseded_ids,
        **(
            {"source_live_marker_closeout": source_live_marker_closeout}
            if source_live_marker_closeout is not None
            else {}
        ),
    }


def validate_evidence_certificate_request_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one explicit gate-closure request."""

    request = _mapping(value, path="request")
    if request.get("schema_version") != REQUEST_SCHEMA_VERSION:
        raise EvidenceCertificateError("evidence certificate request schema mismatch")
    programme_id = _identifier(request.get("programme_id"), path="request.programme_id")
    gate_id = _identifier(request.get("gate_id"), path="request.gate_id")
    plan = _validate_programme_plan(
        request.get("programme_plan"), programme_id=programme_id, gate_id=gate_id
    )
    lifecycle = validate_run_artifact_lifecycle_contract_v1(
        _mapping(
            request.get("artifact_lifecycle_contract"),
            path="request.artifact_lifecycle_contract",
        )
    )
    lifecycle_profile = lifecycle.get("lifecycle_profile")
    diagnostic_closeout = (
        _validate_diagnostic_closeout(request.get("diagnostic_closeout"))
        if lifecycle_profile == DEBUG_FORENSIC
        else None
    )
    if lifecycle_profile not in {
        SCIENCE_COMPACT,
        "CHECKPOINT_TRANSIT",
        "QUALIFICATION",
        DEBUG_FORENSIC,
    }:
        raise EvidenceCertificateError(
            "certificate reclamation requires an explicit lifecycle profile"
        )
    automatic_prune_allowed = lifecycle.get("prune", {}).get("automatic_prune_allowed")
    if lifecycle_profile == DEBUG_FORENSIC and automatic_prune_allowed is not False:
        raise EvidenceCertificateError(
            "DEBUG_FORENSIC source lifecycle contract must remain non-prunable"
        )
    if lifecycle_profile != DEBUG_FORENSIC and automatic_prune_allowed is not True:
        raise EvidenceCertificateError(
            "source lifecycle contract does not allow manifest-authorized release"
        )
    qualification = _mapping(
        request.get("qualification_context"), path="request.qualification_context"
    )
    qualification_status = _text(
        qualification.get("status"), path="request.qualification_context.status"
    )
    if qualification_status not in {"not_applicable", "active", "released"}:
        raise EvidenceCertificateError("qualification_context.status is unsupported")
    exact_ids = _string_list(
        qualification.get("exact_required_artifact_ids", []),
        path="request.qualification_context.exact_required_artifact_ids",
    )
    if qualification_status == "active" and not isinstance(
        qualification.get("authority_reference"), str
    ):
        raise EvidenceCertificateError(
            "active qualification context requires authority_reference"
        )
    audit = _mapping(request.get("audit_context"), path="request.audit_context")
    audit_status = _text(audit.get("status"), path="request.audit_context.status")
    if audit_status not in {"not_applicable", "active", "released"}:
        raise EvidenceCertificateError("audit_context.status is unsupported")
    audit_exact_ids = _string_list(
        audit.get("exact_required_artifact_ids", []),
        path="request.audit_context.exact_required_artifact_ids",
    )
    if audit_status in {"active", "released"} and not isinstance(
        audit.get("authority_reference"), str
    ):
        raise EvidenceCertificateError(
            "active or released audit context requires authority_reference"
        )
    if audit_status == "active" and not audit_exact_ids:
        raise EvidenceCertificateError(
            "active audit context must name its exact protected artifact IDs"
        )
    if lifecycle_profile == DEBUG_FORENSIC and (
        qualification_status == "active"
        or audit_status == "active"
        or exact_ids
        or audit_exact_ids
    ):
        raise EvidenceCertificateError(
            "DEBUG_FORENSIC closeout cannot retain active audit or qualification exact-byte holds"
        )
    raw_artifacts = _object_list(
        request.get("artifacts"), path="request.artifacts", nonempty=True
    )
    normalized_artifacts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    gate_positions = {gate: index for index, gate in enumerate(plan["ordered_gates"])}
    for index, raw in enumerate(raw_artifacts):
        artifact_id = _identifier(
            raw.get("artifact_id"), path=f"request.artifacts[{index}].artifact_id"
        )
        relative = _relative_path(
            raw.get("path"), path=f"request.artifacts[{index}].path"
        )
        if artifact_id in seen_ids or relative in seen_paths:
            raise EvidenceCertificateError("artifact IDs and paths must be unique")
        seen_ids.add(artifact_id)
        seen_paths.add(relative)
        evidence_class = _text(
            raw.get("evidence_class"),
            path=f"request.artifacts[{index}].evidence_class",
        )
        if evidence_class not in EVIDENCE_CLASSES:
            raise EvidenceCertificateError("artifact evidence_class is unsupported")
        produced_by = _identifier(
            raw.get("produced_by"),
            path=f"request.artifacts[{index}].produced_by",
        )
        if produced_by not in gate_positions:
            raise EvidenceCertificateError(
                f"artifact {artifact_id} producer is outside the bound programme plan"
            )
        if gate_positions[produced_by] > gate_positions[gate_id]:
            raise EvidenceCertificateError(
                f"artifact {artifact_id} producer is later than the closing gate"
            )
        required_dependency_fields = {"consumers", "last_use", "dependency_state"}
        omitted_dependency_fields = sorted(required_dependency_fields - set(raw))
        if omitted_dependency_fields:
            raise EvidenceCertificateError(
                f"artifact {artifact_id} omits explicit dependency metadata: "
                + ",".join(omitted_dependency_fields)
            )
        consumers = _string_list(
            raw.get("consumers"), path=f"request.artifacts[{index}].consumers"
        )
        if any(consumer not in gate_positions for consumer in consumers):
            raise EvidenceCertificateError(
                f"artifact {artifact_id} has a consumer outside the bound programme plan"
            )
        if any(
            gate_positions[consumer] < gate_positions[produced_by]
            for consumer in consumers
        ):
            raise EvidenceCertificateError(
                f"artifact {artifact_id} consumer precedes its producer"
            )
        last_use = raw.get("last_use")
        if consumers:
            expected_last = max(consumers, key=gate_positions.__getitem__)
            if last_use != expected_last:
                raise EvidenceCertificateError(
                    f"artifact {artifact_id} last_use must equal its final named consumer"
                )
        elif last_use is not None:
            raise EvidenceCertificateError(
                f"artifact {artifact_id} cannot declare last_use without consumers"
            )
        authorities = _string_list(
            raw.get("retention_authorities", []),
            path=f"request.artifacts[{index}].retention_authorities",
        )
        if any(authority not in RETENTION_AUTHORITIES for authority in authorities):
            raise EvidenceCertificateError(
                f"artifact {artifact_id} has an unsupported retention authority"
            )
        policy = _text(
            raw.get("policy", "KNOWN"), path=f"request.artifacts[{index}].policy"
        )
        if policy not in {"KNOWN", "UNKNOWN_POLICY"}:
            raise EvidenceCertificateError("artifact policy is unsupported")
        dependency_state = _text(
            raw.get("dependency_state"),
            path=f"request.artifacts[{index}].dependency_state",
        )
        if dependency_state not in {"known", "uncertain_bounded"}:
            raise EvidenceCertificateError("artifact dependency_state is unsupported")
        grace_until_gate = raw.get("grace_until_gate")
        if dependency_state == "uncertain_bounded":
            grace_until_gate = _identifier(
                grace_until_gate,
                path=f"request.artifacts[{index}].grace_until_gate",
            )
            if grace_until_gate not in gate_positions:
                raise EvidenceCertificateError(
                    f"artifact {artifact_id} grace gate is outside the bound programme plan"
                )
        elif grace_until_gate is not None:
            raise EvidenceCertificateError(
                f"artifact {artifact_id} cannot declare grace_until_gate for a known dependency"
            )
        normalized_artifacts.append(
            {
                "artifact_id": artifact_id,
                "path": relative,
                "artifact_family": _identifier(
                    raw.get("artifact_family"),
                    path=f"request.artifacts[{index}].artifact_family",
                ),
                "owner_writer": _identifier(
                    raw.get("owner_writer"),
                    path=f"request.artifacts[{index}].owner_writer",
                ),
                "evidence_class": evidence_class,
                "reproduction_role": _text(
                    raw.get("reproduction_role"),
                    path=f"request.artifacts[{index}].reproduction_role",
                ),
                "produced_by": produced_by,
                "consumers": consumers,
                "last_use": last_use,
                "live_state": _validate_live_state(raw.get("live_state"), index=index),
                "forensic_hold": _validate_forensic_hold(
                    raw.get("forensic_hold"), index=index
                ),
                "retention_authorities": authorities,
                "policy": policy,
                "dependency_state": dependency_state,
                "grace_until_gate": grace_until_gate,
                "materialization_group": (
                    _identifier(
                        raw.get("materialization_group"),
                        path=f"request.artifacts[{index}].materialization_group",
                    )
                    if raw.get("materialization_group") is not None
                    else None
                ),
                **(
                    {
                        "source_manifest_binding": _validate_source_manifest_binding(
                            raw.get("source_manifest_binding"), index=index
                        )
                    }
                    if raw.get("source_manifest_binding") is not None
                    else {}
                ),
                **(
                    {
                        "source_artifact_id": _identifier(
                            raw.get("source_artifact_id"),
                            path=(f"request.artifacts[{index}].source_artifact_id"),
                        ),
                        "source_retention_class": _text(
                            raw.get("source_retention_class"),
                            path=(f"request.artifacts[{index}].source_retention_class"),
                        ),
                        "diagnostic_closeout_disposition": _text(
                            raw.get("diagnostic_closeout_disposition"),
                            path=(
                                "request.artifacts["
                                f"{index}].diagnostic_closeout_disposition"
                            ),
                        ),
                    }
                    if lifecycle_profile == DEBUG_FORENSIC
                    else {}
                ),
            }
        )
        if lifecycle_profile == DEBUG_FORENSIC:
            normalized_row = normalized_artifacts[-1]
            source_class = normalized_row["source_retention_class"]
            closeout_disposition = normalized_row["diagnostic_closeout_disposition"]
            if source_class not in RETENTION_CLASSES:
                raise EvidenceCertificateError(
                    f"artifact {artifact_id} source retention class is unsupported"
                )
            if closeout_disposition not in DIAGNOSTIC_CLOSEOUT_DISPOSITIONS:
                raise EvidenceCertificateError(
                    f"artifact {artifact_id} diagnostic closeout disposition is unsupported"
                )
            if any(normalized_row["live_state"].values()):
                raise EvidenceCertificateError(
                    f"artifact {artifact_id} retains a live writer, reference, resume, reducer, or consumer"
                )
            if normalized_row["forensic_hold"]["status"] == "active":
                raise EvidenceCertificateError(
                    f"artifact {artifact_id} retains an active forensic hold"
                )
            exact_byte_authorities = {
                "audit_exact_bytes",
                "qualification_exact_bytes",
                "replay_exact_bytes",
            }
            active_exact = sorted(
                set(normalized_row["retention_authorities"]) & exact_byte_authorities
            )
            if active_exact:
                raise EvidenceCertificateError(
                    f"artifact {artifact_id} retains exact-byte authority: "
                    + ",".join(active_exact)
                )
            if closeout_disposition == "SUPERSEDED_BY_CERTIFICATE":
                if evidence_class not in {
                    "OPERATIONAL_STATE",
                    "RAW_TRANSIENT_EVIDENCE",
                }:
                    raise EvidenceCertificateError(
                        f"artifact {artifact_id} is not raw/transient evidence and cannot be superseded"
                    )
                if source_class not in {"R0", "R1", "R2", "R3"}:
                    raise EvidenceCertificateError(
                        f"artifact {artifact_id} source {source_class} must be retained"
                    )
    missing_qualification = sorted(set(exact_ids) - seen_ids)
    if missing_qualification:
        raise EvidenceCertificateError(
            "qualification exact-required artifact IDs are absent: "
            + ", ".join(missing_qualification)
        )
    missing_audit = sorted(set(audit_exact_ids) - seen_ids)
    if missing_audit:
        raise EvidenceCertificateError(
            "audit exact-required artifact IDs are absent: " + ", ".join(missing_audit)
        )
    if diagnostic_closeout is not None:
        superseded = sorted(
            row["artifact_id"]
            for row in normalized_artifacts
            if row["diagnostic_closeout_disposition"] == "SUPERSEDED_BY_CERTIFICATE"
        )
        if sorted(diagnostic_closeout["superseded_artifact_ids"]) != superseded:
            raise EvidenceCertificateError(
                "diagnostic closeout superseded_artifact_ids must exactly account for superseded rows"
            )
    obligations = _object_list(
        request.get("evidence_obligations"),
        path="request.evidence_obligations",
        nonempty=True,
    )
    for index, obligation in enumerate(obligations):
        if obligation.get("status") not in {"satisfied", "valid_negative"}:
            raise EvidenceCertificateError(
                f"request.evidence_obligations[{index}] is not decision-complete"
            )
        _identifier(
            obligation.get("obligation_id"),
            path=f"request.evidence_obligations[{index}].obligation_id",
        )
        _text(
            obligation.get("description"),
            path=f"request.evidence_obligations[{index}].description",
        )
    validation = _mapping(request.get("validation"), path="request.validation")
    receipts = _object_list(
        validation.get("mechanical_receipts"),
        path="request.validation.mechanical_receipts",
        nonempty=True,
    )
    if any(receipt.get("status") != "passed" for receipt in receipts):
        raise EvidenceCertificateError("all mechanical validation receipts must pass")
    review = _mapping(
        validation.get("semantic_review"), path="request.validation.semantic_review"
    )
    if review.get("status") not in {"accepted", "accepted_with_limitations"}:
        raise EvidenceCertificateError("semantic review is not complete")
    result = _mapping(request.get("result"), path="request.result")
    decision = _text(result.get("decision"), path="request.result.decision")
    if decision not in DECISIONS:
        raise EvidenceCertificateError("semantic decision is unsupported or incomplete")
    if decision in {"NO_SUPPORTED_RESULT", "RAW_DIAGNOSTIC_ABANDONED"} and (
        lifecycle_profile != DEBUG_FORENSIC
    ):
        raise EvidenceCertificateError(
            "no-result and abandoned decisions are limited to DEBUG_FORENSIC closeout"
        )
    criteria = _object_list(
        result.get("criteria"), path="request.result.criteria", nonempty=True
    )
    if any(
        criterion.get("status") not in {"passed", "failed", "valid_negative"}
        for criterion in criteria
    ):
        raise EvidenceCertificateError("result criteria contain an incomplete status")
    reproduction = _mapping(request.get("reproduction"), path="request.reproduction")
    reproduction = {
        "prerequisites": _string_list(
            reproduction.get("prerequisites"),
            path="request.reproduction.prerequisites",
            nonempty=True,
        ),
        "commands": _string_list(
            reproduction.get("commands"),
            path="request.reproduction.commands",
            nonempty=True,
        ),
        "verification_rules": _string_list(
            reproduction.get("verification_rules"),
            path="request.reproduction.verification_rules",
            nonempty=True,
        ),
        "expected_results": _object_list(
            reproduction.get("expected_results"),
            path="request.reproduction.expected_results",
            nonempty=True,
        ),
        "guarantee": _text(
            reproduction.get("guarantee"), path="request.reproduction.guarantee"
        ),
    }
    if reproduction["guarantee"] not in REPRODUCIBILITY_GUARANTEES:
        raise EvidenceCertificateError("reproduction guarantee is unsupported")
    if request.get("inventory_complete") is not True:
        raise EvidenceCertificateError(
            "release certification requires an explicitly complete producer inventory"
        )
    normalized_identity = _validate_identity(request.get("identity"))
    if normalized_identity["plan_sha256"] != plan["plan_sha256"]:
        raise EvidenceCertificateError(
            "request identity plan hash differs from the authoritative programme plan"
        )
    normalized_request = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "programme_id": programme_id,
        "work_id": _identifier(request.get("work_id"), path="request.work_id"),
        "gate_id": gate_id,
        "obligation_id": _identifier(
            request.get("obligation_id"), path="request.obligation_id"
        ),
        "bounded_claim": _text(
            request.get("bounded_claim"), path="request.bounded_claim"
        ),
        "sealed_at_utc": _text(
            request.get("sealed_at_utc"), path="request.sealed_at_utc"
        ),
        "identity": normalized_identity,
        "programme_plan": plan,
        "artifact_lifecycle_contract": lifecycle,
        "governing_procedures": _string_list(
            request.get("governing_procedures"),
            path="request.governing_procedures",
            nonempty=True,
        ),
        "evidence_obligations": obligations,
        "validation": {"mechanical_receipts": receipts, "semantic_review": review},
        "result": {
            "decision": decision,
            "criteria": criteria,
            "measurements": _mapping(
                result.get("measurements"), path="request.result.measurements"
            ),
            "digests": _mapping(result.get("digests"), path="request.result.digests"),
            "limitations": _string_list(
                result.get("limitations", []), path="request.result.limitations"
            ),
            "no_claims": _string_list(
                result.get("no_claims"), path="request.result.no_claims", nonempty=True
            ),
        },
        "reproduction": reproduction,
        "qualification_context": {
            "status": qualification_status,
            "authority_reference": qualification.get("authority_reference"),
            "exact_required_artifact_ids": exact_ids,
        },
        "audit_context": {
            "status": audit_status,
            "authority_reference": audit.get("authority_reference"),
            "exact_required_artifact_ids": audit_exact_ids,
        },
        "inventory_complete": True,
        "artifacts": normalized_artifacts,
    }
    if diagnostic_closeout is not None:
        normalized_request["diagnostic_closeout"] = diagnostic_closeout
    return normalized_request


def _artifact_outcome(
    row: Mapping[str, Any],
    *,
    closed_gates: set[str],
    qualification_required: set[str],
    audit_required: set[str],
    materialization_group_live: bool,
    diagnostic_closeout: Mapping[str, Any] | None = None,
) -> tuple[str, str, str, bool]:
    artifact_id = str(row["artifact_id"])
    evidence_class = str(row["evidence_class"])
    authorities = set(row["retention_authorities"])
    if artifact_id in qualification_required:
        authorities.add("qualification_exact_bytes")
    if artifact_id in audit_required:
        authorities.add("audit_exact_bytes")
    if row["policy"] == "UNKNOWN_POLICY":
        return "UNKNOWN_POLICY", "RX", "retention policy is unresolved", False
    if row["dependency_state"] == "uncertain_bounded":
        grace_gate = str(row["grace_until_gate"])
        if grace_gate not in closed_gates:
            return (
                "HOLD_UNTIL_LAST_USE",
                "R1",
                f"future dependency uncertain through bounded grace gate {grace_gate}",
                False,
            )
        return (
            "UNKNOWN_POLICY",
            "RX",
            f"bounded dependency grace expired at {grace_gate}; explicit policy review required",
            False,
        )
    if diagnostic_closeout is not None:
        source_class = str(row["source_retention_class"])
        closeout_disposition = str(row["diagnostic_closeout_disposition"])
        if authorities:
            return (
                "RETAIN_DURABLE",
                "R4",
                "exact durable authority remains active: "
                + ",".join(sorted(authorities)),
                False,
            )
        if closeout_disposition == "RETAIN":
            if source_class == "RX":
                return (
                    "UNKNOWN_POLICY",
                    "RX",
                    "diagnostic closeout explicitly preserves unresolved source row",
                    False,
                )
            if source_class in {"R3", "R4", "R5"}:
                return (
                    "RETAIN_DURABLE",
                    source_class,
                    "diagnostic closeout explicitly preserves source retention authority",
                    False,
                )
            return (
                "HOLD_UNTIL_LAST_USE",
                source_class,
                "diagnostic closeout did not supersede this exact source row",
                False,
            )
        effective_class = "R2" if source_class == "R3" else source_class
        reason = (
            "accepted diagnostic certificate supersedes exact decision evidence"
            if source_class == "R3"
            else "accepted diagnostic certificate supersedes exact raw/transient evidence"
        )
        return "RELEASE_NOW", effective_class, reason, True
    source_binding = row.get("source_manifest_binding")
    bound_source_class = (
        str(source_binding.get("retention_class"))
        if isinstance(source_binding, Mapping)
        else None
    )
    if bound_source_class == "RX":
        return (
            "UNKNOWN_POLICY",
            "RX",
            "bound producer retention class remains unresolved",
            False,
        )
    if bound_source_class in {"R3", "R4", "R5"}:
        return (
            "RETAIN_DURABLE",
            bound_source_class,
            "ordinary certificate cannot supersede bound durable producer authority",
            False,
        )
    if evidence_class == "DURABLE_EVIDENCE_CERTIFICATE":
        return "RETAIN_DURABLE", "R5", "compact certificate is durable authority", False
    if evidence_class == "SCIENTIFIC_ENGINEERING_ASSET":
        return "RETAIN_ASSET", "R4", "explicit scientific or engineering asset", False
    if authorities:
        return (
            "RETAIN_DURABLE",
            "R4",
            "exact durable authority remains active: " + ",".join(sorted(authorities)),
            False,
        )
    forensic = row["forensic_hold"]
    if forensic["status"] == "active":
        return "FORENSIC_HOLD", "R3", "unresolved forensic hold is active", False
    if evidence_class == "FORENSIC_HOLD" and forensic["status"] != "closed":
        return "FORENSIC_HOLD", "R3", "forensic evidence lacks closure", False
    live_reasons = sorted(name for name, active in row["live_state"].items() if active)
    if materialization_group_live and not live_reasons:
        live_reasons = ["duplicate_materialization_group"]
    remaining_consumers = [
        gate for gate in row["consumers"] if gate not in closed_gates
    ]
    if live_reasons or remaining_consumers:
        reasons = []
        if remaining_consumers:
            reasons.append("future consumers=" + ",".join(remaining_consumers))
        if live_reasons:
            reasons.append("live=" + ",".join(live_reasons))
        return "HOLD_UNTIL_LAST_USE", "R1", "; ".join(reasons), False
    if bound_source_class in {"R0", "R1", "R2"}:
        retention_class = bound_source_class
    elif evidence_class == "OPERATIONAL_STATE":
        retention_class = "R1"
    elif row.get("reproduction_role") == "reproducible_derived":
        retention_class = "R2"
    else:
        retention_class = "R0"
    return (
        "RELEASE_NOW",
        retention_class,
        "certificate complete and final justified consumer/hold has closed",
        True,
    )


def _validate_sealed_source_manifest_bindings_v1(
    *,
    evidence_root: Path,
    sealed_rows: Sequence[Mapping[str, Any]],
    require_all: bool,
    authenticated_absent_paths: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    manifests: dict[str, dict[str, Any]] = {}
    source_runtime_roots: set[str] = set()
    bound_rows: list[dict[str, Any]] = []
    for sealed in sealed_rows:
        binding = sealed.get("source_manifest_binding")
        if binding is None:
            if require_all:
                raise EvidenceCertificateError(
                    f"artifact {sealed['artifact_id']} requires source_manifest_binding for configured runtime root release"
                )
            continue
        binding = _mapping(
            binding,
            path=f"artifact {sealed['artifact_id']} source_manifest_binding",
        )
        digest_name = "sha256" if "sha256" in binding else "tree_sha256"
        if (
            sealed.get("bytes") != binding.get("bytes")
            or sealed.get(digest_name) != binding.get(digest_name)
            or ("sha256" in sealed) != (digest_name == "sha256")
        ):
            raise EvidenceCertificateError(
                f"artifact {sealed['artifact_id']} actual sealed bytes differ from source_manifest_binding"
            )
        manifest_relative = str(binding["manifest_path"])
        manifest_path = evidence_root / manifest_relative
        if manifest_path.is_symlink():
            raise EvidenceCertificateError(
                "source disposition manifest cannot be a symlink"
            )
        resolved_manifest = manifest_path.resolve(strict=True)
        try:
            resolved_manifest.relative_to(evidence_root)
        except ValueError as exc:
            raise EvidenceCertificateError(
                "source disposition manifest escapes evidence_root"
            ) from exc
        cache_key = str(resolved_manifest)
        source_manifest = manifests.get(cache_key)
        if source_manifest is None:
            source_manifest = validate_artifact_disposition_manifest_v1(
                load_json_object(resolved_manifest)
            )
            if source_manifest["manifest_sha256"] != binding["manifest_sha256"]:
                raise EvidenceCertificateError(
                    "source disposition manifest SHA-256 differs from artifact binding"
                )
            if source_manifest.get("inventory") != {
                "mode": "producer_declared",
                "complete": True,
            }:
                raise EvidenceCertificateError(
                    "source disposition binding requires complete producer-declared inventory"
                )
            manifests[cache_key] = source_manifest
        elif source_manifest["manifest_sha256"] != binding["manifest_sha256"]:
            raise EvidenceCertificateError(
                "one source manifest path is bound to conflicting SHA-256 identities"
            )
        source_matches = [
            dict(row)
            for row in source_manifest.get("artifacts", [])
            if isinstance(row, Mapping)
            and row.get("artifact_id") == binding["artifact_id"]
        ]
        if len(source_matches) != 1:
            raise EvidenceCertificateError(
                f"source manifest artifact identity is missing or ambiguous: {binding['artifact_id']}"
            )
        source_row = source_matches[0]
        expected = {
            "artifact_id": binding["artifact_id"],
            "bytes": binding["bytes"],
            "retention_class": binding["retention_class"],
            digest_name: binding[digest_name],
            "exact_object_identity": True,
        }
        mismatched = sorted(
            name for name, value in expected.items() if source_row.get(name) != value
        )
        source_runtime_root = Path(
            _text(
                source_manifest.get("runtime_root"),
                path="source disposition manifest runtime_root",
            )
        ).expanduser()
        if source_runtime_root.is_symlink():
            raise EvidenceCertificateError(
                "source manifest runtime_root cannot be a symlink"
            )
        source_runtime_root = source_runtime_root.resolve(strict=True)
        try:
            source_runtime_root.relative_to(evidence_root)
        except ValueError as exc:
            raise EvidenceCertificateError(
                "source manifest runtime_root escapes evidence_root"
            ) from exc
        source_runtime_roots.add(str(source_runtime_root))
        source_relative = Path(
            _relative_path(
                source_row.get("path"), path="source disposition artifact path"
            )
        )
        source_target = source_runtime_root / source_relative
        if source_target.is_symlink():
            raise EvidenceCertificateError(
                "source disposition artifact cannot be a symlink"
            )
        sealed_relative = Path(
            _relative_path(sealed.get("path"), path="sealed artifact path")
        )
        actual_target = evidence_root / sealed_relative
        if str(sealed_relative) in authenticated_absent_paths:
            for base, relative in (
                (source_runtime_root, source_relative),
                (evidence_root, sealed_relative),
            ):
                candidate = base
                for part in relative.parts:
                    candidate /= part
                    if candidate.is_symlink():
                        raise EvidenceCertificateError(
                            "authenticated absent artifact path cannot contain a symlink"
                        )
            source_identity = source_target.resolve(strict=False)
            actual_identity = actual_target.resolve(strict=False)
        else:
            source_identity = source_target.resolve(strict=True)
            actual_identity = actual_target.resolve(strict=True)
        if source_identity != actual_identity:
            mismatched.append("path")
        if mismatched:
            raise EvidenceCertificateError(
                f"source manifest binding for {sealed['artifact_id']} differs at: "
                + ",".join(sorted(set(mismatched)))
            )
        bound_rows.append(
            {
                "request_artifact_id": sealed["artifact_id"],
                "source_manifest_binding": binding,
            }
        )
    if require_all and len(bound_rows) != len(sealed_rows):
        raise EvidenceCertificateError(
            "every configured runtime root certificate artifact requires a source manifest binding"
        )
    return {
        "binding_count": len(bound_rows),
        "manifest_count": len(manifests),
        "bindings_sha256": canonical_sha256(bound_rows),
        "bindings": bound_rows,
        "source_runtime_roots": sorted(source_runtime_roots),
    }


def _load_diagnostic_source_manifest_v1(
    normalized: Mapping[str, Any],
    *,
    evidence_root: Path,
    sealed_rows: Sequence[Mapping[str, Any]],
    require_fresh_review_audit: bool = True,
    recheck_local_live_state: bool = True,
) -> dict[str, Any]:
    closeout = _mapping(
        normalized.get("diagnostic_closeout"), path="request.diagnostic_closeout"
    )
    source_identity = _mapping(
        closeout.get("source_manifest"),
        path="request.diagnostic_closeout.source_manifest",
    )
    if (
        source_identity.get("runtime_root") is not None
        and Path(str(source_identity["runtime_root"])).expanduser().resolve(strict=True)
        != evidence_root
    ):
        raise EvidenceCertificateError(
            "diagnostic source manifest runtime root differs from evidence_root"
        )
    unresolved = evidence_root / str(source_identity["path"])
    if unresolved.is_symlink():
        raise EvidenceCertificateError("diagnostic source manifest cannot be a symlink")
    source_path = unresolved.resolve(strict=True)
    try:
        source_path.relative_to(evidence_root)
    except ValueError as exc:
        raise EvidenceCertificateError(
            "diagnostic source manifest escapes evidence_root"
        ) from exc
    if not source_path.is_file():
        raise EvidenceCertificateError(
            "diagnostic source manifest must be a regular file"
        )
    source = validate_artifact_disposition_manifest_v1(load_json_object(source_path))
    expected_identity = {
        "manifest_sha256": source_identity["manifest_sha256"],
        "runtime_root": str(evidence_root),
        "campaign_id": source_identity["campaign_id"],
        "run_id": source_identity["run_id"],
        "plan_sha256": source_identity["plan_sha256"],
        "lifecycle_profile": DEBUG_FORENSIC,
        "lifecycle_contract_sha256": source_identity["lifecycle_contract_sha256"],
        "terminal_status": "completed",
    }
    mismatched = sorted(
        name
        for name, expected in expected_identity.items()
        if source.get(name) != expected
    )
    if mismatched:
        raise EvidenceCertificateError(
            "diagnostic source manifest identity differs at: " + ",".join(mismatched)
        )
    if (
        source_identity["lifecycle_contract_sha256"]
        != normalized["artifact_lifecycle_contract"]["contract_sha256"]
    ):
        raise EvidenceCertificateError(
            "diagnostic request lifecycle contract is not the exact plan-bound source contract"
        )
    if source.get("inventory") != {
        "mode": "producer_declared",
        "complete": True,
    }:
        raise EvidenceCertificateError(
            "diagnostic source manifest requires a complete producer-declared inventory"
        )
    source_rows = {
        str(row.get("path")): dict(row)
        for row in source.get("artifacts", [])
        if isinstance(row, Mapping)
    }
    sealed_by_path = {str(row["path"]): dict(row) for row in sealed_rows}
    if len(source_rows) != len(source.get("artifacts", [])) or set(source_rows) != set(
        sealed_by_path
    ):
        raise EvidenceCertificateError(
            "diagnostic source manifest does not exactly account for request artifacts"
        )
    for path, sealed in sealed_by_path.items():
        source_row = source_rows[path]
        expected_fields = {
            "artifact_id": sealed["source_artifact_id"],
            "path": sealed["path"],
            "artifact_family": sealed["artifact_family"],
            "artifact_type": sealed["artifact_type"],
            "owner_writer": sealed["owner_writer"],
            "bytes": sealed["bytes"],
            "retention_class": sealed["source_retention_class"],
            "exact_object_identity": True,
        }
        digest_name = "sha256" if "sha256" in sealed else "tree_sha256"
        expected_fields[digest_name] = sealed[digest_name]
        mismatched_fields = sorted(
            name
            for name, expected in expected_fields.items()
            if source_row.get(name) != expected
        )
        if mismatched_fields:
            raise EvidenceCertificateError(
                f"diagnostic source row {path} differs at: "
                + ",".join(mismatched_fields)
            )
        if sealed.get("diagnostic_closeout_disposition") == (
            "SUPERSEDED_BY_CERTIFICATE"
        ):
            if source_row.get("audit_hold_status") not in {
                "released",
                "not_required",
            } or source_row.get("qualification_hold_status") not in {
                "released",
                "not_required",
            }:
                raise EvidenceCertificateError(
                    f"diagnostic source row {path} retains or lacks closed audit/qualification hold authority"
                )
            if source_row.get("live_reference_status") not in {"none", "active"}:
                raise EvidenceCertificateError(
                    f"diagnostic source row {path} has unknown live-reference authority"
                )
    superseded_ids = sorted(
        str(row["artifact_id"])
        for row in sealed_rows
        if row.get("diagnostic_closeout_disposition") == "SUPERSEDED_BY_CERTIFICATE"
    )
    active_superseded = [
        str(row["artifact_id"])
        for path, row in sealed_by_path.items()
        if row.get("diagnostic_closeout_disposition") == "SUPERSEDED_BY_CERTIFICATE"
        and source_rows[path].get("live_reference_status") == "active"
    ]
    marker = closeout.get("source_live_marker_closeout")
    if active_superseded and marker is None:
        raise EvidenceCertificateError(
            "active diagnostic source live markers require an exact current-state closeout"
        )
    if marker is not None:
        validated_marker = validate_diagnostic_source_live_marker_closeout_v1(
            marker,
            selected_artifact_ids=superseded_ids,
            source_manifest={
                "manifest_sha256": source["manifest_sha256"],
                "campaign_id": source["campaign_id"],
                "run_id": source["run_id"],
                "runtime_root": source["runtime_root"],
            },
        )
        if validated_marker != marker:
            raise EvidenceCertificateError(
                "diagnostic source live-marker closeout is not canonical"
            )
        if require_fresh_review_audit:
            validate_diagnostic_source_live_marker_freshness_v1(validated_marker)
    if recheck_local_live_state:
        require_diagnostic_local_live_state_clear_v1(evidence_root)
    return source


def build_evidence_certificate_v1(
    request: Mapping[str, Any], *, evidence_root: Path
) -> dict[str, Any]:
    """Seal exact inventory and build a deterministic pre-release certificate."""

    normalized = validate_evidence_certificate_request_v1(request)
    root = evidence_root.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise EvidenceCertificateError("evidence_root must be a real directory")
    sealed_rows: list[dict[str, Any]] = []
    for row in normalized["artifacts"]:
        sealed = build_sealed_artifact_inventory_row_v1(
            run_root=root,
            path=root / row["path"],
            artifact_family=row["artifact_family"],
            retention_class="RX",
            owner_writer=row["owner_writer"],
        )
        sealed_rows.append({**row, **sealed})
    source_binding_validation = _validate_sealed_source_manifest_bindings_v1(
        evidence_root=root,
        sealed_rows=sealed_rows,
        require_all=evidence_root_requires_permanent_capsule_v1(root),
    )
    diagnostic_closeout = normalized.get("diagnostic_closeout")
    diagnostic_source_manifest = None
    if diagnostic_closeout is not None:
        diagnostic_source_manifest = _load_diagnostic_source_manifest_v1(
            normalized, evidence_root=root, sealed_rows=sealed_rows
        )
    live_groups = {
        str(row["materialization_group"])
        for row in sealed_rows
        if row.get("materialization_group") is not None
        and any(row["live_state"].values())
    }
    closed = set(normalized["programme_plan"]["closed_gates"])
    qualification_required = set(
        normalized["qualification_context"]["exact_required_artifact_ids"]
    )
    audit_required = set(normalized["audit_context"]["exact_required_artifact_ids"])
    inventory: list[dict[str, Any]] = []
    for row in sealed_rows:
        disposition, retention_class, reason, release_eligible = _artifact_outcome(
            row,
            closed_gates=closed,
            qualification_required=qualification_required,
            audit_required=audit_required,
            materialization_group_live=(
                row.get("materialization_group") in live_groups
                if row.get("materialization_group") is not None
                else False
            ),
            diagnostic_closeout=(
                diagnostic_closeout
                if isinstance(diagnostic_closeout, Mapping)
                else None
            ),
        )
        outcome_row = {
            **row,
            "disposition": disposition,
            "retention_class": retention_class,
            "release_eligible": release_eligible,
            "disposition_reason": reason,
            "remaining_consumers": [
                gate for gate in row["consumers"] if gate not in closed
            ],
        }
        if (
            diagnostic_closeout is None
            and row.get("source_manifest_binding") is not None
        ):
            outcome_row["source_retention_class"] = row["source_manifest_binding"][
                "retention_class"
            ]
        if diagnostic_source_manifest is not None:
            superseded = (
                row["diagnostic_closeout_disposition"] == "SUPERSEDED_BY_CERTIFICATE"
            )
            outcome_row["retention_transition"] = {
                "kind": (
                    "DECISION_EVIDENCE_SUPERSEDED_BY_CERTIFICATE"
                    if superseded and row["source_retention_class"] == "R3"
                    else "RAW_TRANSIENT_SUPERSEDED_BY_CERTIFICATE"
                    if superseded
                    else "SOURCE_RETENTION_PRESERVED"
                ),
                "from_retention_class": row["source_retention_class"],
                "to_retention_class": retention_class,
                "source_manifest_sha256": diagnostic_source_manifest["manifest_sha256"],
                "release_eligible": release_eligible,
            }
        inventory.append(outcome_row)
    request_sha256 = canonical_sha256(normalized)
    immutable = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "phase": "SEALED_PRE_RELEASE",
        "programme_id": normalized["programme_id"],
        "work_id": normalized["work_id"],
        "gate_id": normalized["gate_id"],
        "obligation_id": normalized["obligation_id"],
        "bounded_claim": normalized["bounded_claim"],
        "sealed_at_utc": normalized["sealed_at_utc"],
        "request_sha256": request_sha256,
        "identity": normalized["identity"],
        "programme_plan": normalized["programme_plan"],
        "artifact_lifecycle_contract_sha256": normalized["artifact_lifecycle_contract"][
            "contract_sha256"
        ],
        "governing_procedures": normalized["governing_procedures"],
        "evidence_obligations": normalized["evidence_obligations"],
        "validation": normalized["validation"],
        "result": normalized["result"],
        "reproduction": normalized["reproduction"],
        "qualification_context": normalized["qualification_context"],
        "audit_context": normalized["audit_context"],
        "inventory_complete": True,
        "artifacts": inventory,
        "disposition_summary": {
            "produced_bytes": sum(int(row["bytes"]) for row in inventory),
            "certified_releasable_bytes": sum(
                int(row["bytes"]) for row in inventory if row["release_eligible"]
            ),
            "released_bytes": 0,
            "retained_bytes": sum(
                int(row["bytes"]) for row in inventory if not row["release_eligible"]
            ),
            "by_disposition": {
                disposition: {
                    "objects": sum(
                        1 for row in inventory if row["disposition"] == disposition
                    ),
                    "bytes": sum(
                        int(row["bytes"])
                        for row in inventory
                        if row["disposition"] == disposition
                    ),
                }
                for disposition in DISPOSITIONS
            },
        },
    }
    if diagnostic_closeout is not None:
        immutable["diagnostic_closeout"] = diagnostic_closeout
    if source_binding_validation["binding_count"]:
        immutable["source_manifest_binding_summary"] = {
            name: source_binding_validation[name]
            for name in ("binding_count", "manifest_count", "bindings_sha256")
        }
    certificate_id = "eiel-cert-" + canonical_sha256(immutable)[:24]
    core = {**immutable, "certificate_id": certificate_id}
    return {**core, "certificate_sha256": canonical_sha256(core)}


def validate_evidence_certificate_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    certificate = _mapping(value, path="certificate")
    supplied = certificate.pop("certificate_sha256", None)
    if certificate.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
        raise EvidenceCertificateError("evidence certificate schema mismatch")
    if certificate.get("phase") not in {"SEALED_PRE_RELEASE", "FINAL"}:
        raise EvidenceCertificateError("evidence certificate phase is unsupported")
    if certificate.get("inventory_complete") is not True:
        raise EvidenceCertificateError("evidence certificate inventory is incomplete")
    if supplied != canonical_sha256(certificate):
        raise EvidenceCertificateError("evidence certificate SHA-256 mismatch")
    rows = certificate.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise EvidenceCertificateError("evidence certificate artifacts are missing")
    diagnostic = certificate.get("diagnostic_closeout")
    normalized_diagnostic = None
    if diagnostic is not None:
        normalized_diagnostic = _validate_diagnostic_closeout(diagnostic)
        if normalized_diagnostic != diagnostic:
            raise EvidenceCertificateError(
                "evidence certificate diagnostic closeout is not canonical"
            )
    superseded_ids: list[str] = []
    certificate_source_bindings: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise EvidenceCertificateError(f"certificate artifact {index} is malformed")
        if row.get("evidence_class") not in EVIDENCE_CLASSES:
            raise EvidenceCertificateError(
                f"certificate artifact {index} class is unsupported"
            )
        if row.get("disposition") not in DISPOSITIONS:
            raise EvidenceCertificateError(
                f"certificate artifact {index} disposition is unsupported"
            )
        if (
            row.get("release_eligible") is True
            and row.get("disposition") != "RELEASE_NOW"
        ):
            raise EvidenceCertificateError(
                "only RELEASE_NOW certificate rows can be release eligible"
            )
        digest = row.get("sha256") or row.get("tree_sha256")
        _sha256(digest, path=f"certificate.artifacts[{index}].identity")
        if row.get("source_manifest_binding") is not None:
            binding = _validate_source_manifest_binding(
                row.get("source_manifest_binding"), index=index
            )
            digest_name = "sha256" if "sha256" in binding else "tree_sha256"
            if (
                binding != row.get("source_manifest_binding")
                or binding["bytes"] != row.get("bytes")
                or binding[digest_name] != row.get(digest_name)
            ):
                raise EvidenceCertificateError(
                    f"certificate artifact {index} source manifest binding differs from sealed identity"
                )
            if row.get("source_retention_class") != binding["retention_class"]:
                raise EvidenceCertificateError(
                    f"certificate artifact {index} omits its bound source retention class"
                )
            if (
                normalized_diagnostic is None
                and binding["retention_class"] not in {"R0", "R1", "R2"}
                and (
                    row.get("release_eligible") is True
                    or row.get("retention_class") != binding["retention_class"]
                    or (
                        binding["retention_class"] == "RX"
                        and row.get("disposition") != "UNKNOWN_POLICY"
                    )
                    or (
                        binding["retention_class"] in {"R3", "R4", "R5"}
                        and row.get("disposition") != "RETAIN_DURABLE"
                    )
                )
            ):
                raise EvidenceCertificateError(
                    "ordinary certificate cannot relabel or release a bound R3-R5/RX source row"
                )
            certificate_source_bindings.append(
                {
                    "request_artifact_id": row.get("artifact_id"),
                    "source_manifest_binding": binding,
                }
            )
        if normalized_diagnostic is not None:
            source_class = row.get("source_retention_class")
            source_artifact_id = row.get("source_artifact_id")
            closeout_disposition = row.get("diagnostic_closeout_disposition")
            if (
                source_class not in RETENTION_CLASSES
                or not isinstance(source_artifact_id, str)
                or closeout_disposition not in DIAGNOSTIC_CLOSEOUT_DISPOSITIONS
                or not isinstance(row.get("retention_transition"), Mapping)
            ):
                raise EvidenceCertificateError(
                    f"certificate diagnostic artifact {index} accounting is malformed"
                )
            if closeout_disposition == "SUPERSEDED_BY_CERTIFICATE":
                superseded_ids.append(str(row.get("artifact_id")))
                if row.get("retention_class") not in {"R0", "R1", "R2"} or (
                    row.get("release_eligible") is not True
                ):
                    raise EvidenceCertificateError(
                        "superseded diagnostic rows must be release-eligible R0-R2"
                    )
            elif row.get("release_eligible") is True:
                raise EvidenceCertificateError(
                    "diagnostic rows not superseded by the certificate cannot be released"
                )
    if normalized_diagnostic is not None and sorted(superseded_ids) != sorted(
        normalized_diagnostic["superseded_artifact_ids"]
    ):
        raise EvidenceCertificateError(
            "certificate diagnostic artifact accounting is incomplete"
        )
    source_summary = certificate.get("source_manifest_binding_summary")
    if certificate_source_bindings:
        manifest_count = len(
            {
                (
                    row["source_manifest_binding"]["manifest_path"],
                    row["source_manifest_binding"]["manifest_sha256"],
                )
                for row in certificate_source_bindings
            }
        )
        expected_summary = {
            "binding_count": len(certificate_source_bindings),
            "manifest_count": manifest_count,
            "bindings_sha256": canonical_sha256(certificate_source_bindings),
        }
        if source_summary != expected_summary:
            raise EvidenceCertificateError(
                "certificate source manifest binding summary is incomplete"
            )
    elif source_summary is not None:
        raise EvidenceCertificateError(
            "certificate source manifest binding summary has no bound rows"
        )
    summary = certificate.get("disposition_summary")
    if not isinstance(summary, Mapping):
        raise EvidenceCertificateError(
            "evidence certificate disposition summary is missing"
        )
    produced_bytes = sum(int(row.get("bytes", 0)) for row in rows)
    releasable_bytes = sum(
        int(row.get("bytes", 0)) for row in rows if row.get("release_eligible") is True
    )
    if (
        summary.get("produced_bytes") != produced_bytes
        or summary.get("certified_releasable_bytes") != releasable_bytes
    ):
        raise EvidenceCertificateError(
            "evidence certificate disposition byte summary is inconsistent"
        )
    final_fields = {
        "prior_certificate_sha256",
        "finalized_at_utc",
        "storage_disposition_receipt_sha256",
    }
    if certificate.get("phase") == "SEALED_PRE_RELEASE":
        if final_fields & set(certificate) or any(
            any(
                name in row
                for name in (
                    "released",
                    "released_at_utc",
                    "release_receipt_sha256",
                )
            )
            for row in rows
        ):
            raise EvidenceCertificateError(
                "pre-release certificate contains FINAL-only release fields"
            )
        if (
            summary.get("released_bytes") != 0
            or summary.get("retained_bytes") != produced_bytes - releasable_bytes
        ):
            raise EvidenceCertificateError(
                "pre-release certificate disposition summary is inconsistent"
            )
        identity_input = dict(certificate)
        certificate_id = identity_input.pop("certificate_id", None)
        expected_id = "eiel-cert-" + canonical_sha256(identity_input)[:24]
        if certificate_id != expected_id:
            raise EvidenceCertificateError("evidence certificate ID binding mismatch")
    else:
        if not final_fields.issubset(certificate):
            raise EvidenceCertificateError("FINAL certificate fields are incomplete")
        prior_sha256 = _sha256(
            certificate.get("prior_certificate_sha256"),
            path="certificate.prior_certificate_sha256",
        )
        receipt_sha256 = _sha256(
            certificate.get("storage_disposition_receipt_sha256"),
            path="certificate.storage_disposition_receipt_sha256",
        )
        finalized_at = _text(
            certificate.get("finalized_at_utc"),
            path="certificate.finalized_at_utc",
        )
        released_bytes = 0
        reconstructed_rows: list[dict[str, Any]] = []
        for index, raw_row in enumerate(rows):
            row = dict(raw_row)
            released = row.get("released")
            if type(released) is not bool:
                raise EvidenceCertificateError(
                    f"FINAL certificate artifact {index} released must be boolean"
                )
            if released:
                if row.get("release_eligible") is not True:
                    raise EvidenceCertificateError(
                        "FINAL certificate cannot release an ineligible artifact"
                    )
                if (
                    row.get("released_at_utc") != finalized_at
                    or row.get("release_receipt_sha256") != receipt_sha256
                ):
                    raise EvidenceCertificateError(
                        "FINAL certificate released row lacks exact time/receipt binding"
                    )
                released_bytes += int(row.get("bytes", 0))
            elif (
                row.get("released_at_utc") is not None
                or row.get("release_receipt_sha256") is not None
            ):
                raise EvidenceCertificateError(
                    "FINAL certificate retained row carries release metadata"
                )
            if row.get("release_eligible") is True and not released:
                raise EvidenceCertificateError(
                    "successful FINAL certificate omits an eligible release row"
                )
            for name in (
                "released",
                "released_at_utc",
                "release_receipt_sha256",
            ):
                row.pop(name, None)
            reconstructed_rows.append(row)
        if (
            summary.get("released_bytes") != released_bytes
            or summary.get("retained_bytes") != produced_bytes - released_bytes
        ):
            raise EvidenceCertificateError(
                "FINAL certificate disposition summary is inconsistent"
            )
        reconstructed = dict(certificate)
        for name in final_fields:
            reconstructed.pop(name, None)
        reconstructed["phase"] = "SEALED_PRE_RELEASE"
        reconstructed["artifacts"] = reconstructed_rows
        reconstructed["disposition_summary"] = {
            **dict(summary),
            "released_bytes": 0,
            "retained_bytes": produced_bytes - releasable_bytes,
        }
        validate_evidence_certificate_v1(
            {**reconstructed, "certificate_sha256": prior_sha256}
        )
    return {**certificate, "certificate_sha256": supplied}


def validate_certificate_request_binding_v1(
    request: Mapping[str, Any], certificate: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove that certificate semantics still derive from the bound request."""

    normalized = validate_evidence_certificate_request_v1(request)
    cert = validate_evidence_certificate_v1(certificate)
    if cert["request_sha256"] != canonical_sha256(normalized):
        raise EvidenceCertificateError(
            "certificate is stale for the gate-closure request"
        )
    top_level_bindings = {
        "programme_id": normalized["programme_id"],
        "work_id": normalized["work_id"],
        "gate_id": normalized["gate_id"],
        "obligation_id": normalized["obligation_id"],
        "bounded_claim": normalized["bounded_claim"],
        "sealed_at_utc": normalized["sealed_at_utc"],
        "identity": normalized["identity"],
        "programme_plan": normalized["programme_plan"],
        "governing_procedures": normalized["governing_procedures"],
        "evidence_obligations": normalized["evidence_obligations"],
        "validation": normalized["validation"],
        "result": normalized["result"],
        "reproduction": normalized["reproduction"],
        "qualification_context": normalized["qualification_context"],
        "audit_context": normalized["audit_context"],
        "artifact_lifecycle_contract_sha256": normalized["artifact_lifecycle_contract"][
            "contract_sha256"
        ],
        "inventory_complete": True,
    }
    if "diagnostic_closeout" in normalized:
        top_level_bindings["diagnostic_closeout"] = normalized["diagnostic_closeout"]
    for name, expected in top_level_bindings.items():
        if cert.get(name) != expected:
            raise EvidenceCertificateError(
                f"certificate {name} does not match its gate-closure request"
            )
    certificate_rows = {
        str(row.get("artifact_id")): row
        for row in cert["artifacts"]
        if isinstance(row, Mapping)
    }
    if len(certificate_rows) != len(normalized["artifacts"]):
        raise EvidenceCertificateError("certificate artifact set differs from request")
    live_groups = {
        str(row["materialization_group"])
        for row in normalized["artifacts"]
        if row.get("materialization_group") is not None
        and any(row["live_state"].values())
    }
    closed = set(normalized["programme_plan"]["closed_gates"])
    qualification_required = set(
        normalized["qualification_context"]["exact_required_artifact_ids"]
    )
    audit_required = set(normalized["audit_context"]["exact_required_artifact_ids"])
    for row in normalized["artifacts"]:
        observed = certificate_rows.get(row["artifact_id"])
        if observed is None:
            raise EvidenceCertificateError(
                f"certificate omits artifact {row['artifact_id']}"
            )
        for name in (
            "path",
            "artifact_family",
            "owner_writer",
            "evidence_class",
            "reproduction_role",
            "produced_by",
            "consumers",
            "last_use",
            "live_state",
            "forensic_hold",
            "retention_authorities",
            "policy",
            "dependency_state",
            "grace_until_gate",
            "materialization_group",
            "source_manifest_binding",
            *(
                (
                    "source_artifact_id",
                    "source_retention_class",
                    "diagnostic_closeout_disposition",
                )
                if "diagnostic_closeout" in normalized
                else ()
            ),
        ):
            if observed.get(name) != row.get(name):
                raise EvidenceCertificateError(
                    f"certificate artifact {row['artifact_id']} {name} differs from request"
                )
        if (
            row.get("source_manifest_binding") is not None
            and observed.get("source_retention_class")
            != row["source_manifest_binding"]["retention_class"]
        ):
            raise EvidenceCertificateError(
                f"certificate artifact {row['artifact_id']} source retention class is not request-derived"
            )
        disposition, retention_class, reason, release_eligible = _artifact_outcome(
            row,
            closed_gates=closed,
            qualification_required=qualification_required,
            audit_required=audit_required,
            materialization_group_live=(
                row.get("materialization_group") in live_groups
                if row.get("materialization_group") is not None
                else False
            ),
            diagnostic_closeout=normalized.get("diagnostic_closeout"),
        )
        expected_outcome = {
            "disposition": disposition,
            "retention_class": retention_class,
            "disposition_reason": reason,
            "release_eligible": release_eligible,
        }
        if "diagnostic_closeout" in normalized:
            superseded = (
                row["diagnostic_closeout_disposition"] == "SUPERSEDED_BY_CERTIFICATE"
            )
            expected_outcome["retention_transition"] = {
                "kind": (
                    "DECISION_EVIDENCE_SUPERSEDED_BY_CERTIFICATE"
                    if superseded and row["source_retention_class"] == "R3"
                    else "RAW_TRANSIENT_SUPERSEDED_BY_CERTIFICATE"
                    if superseded
                    else "SOURCE_RETENTION_PRESERVED"
                ),
                "from_retention_class": row["source_retention_class"],
                "to_retention_class": retention_class,
                "source_manifest_sha256": normalized["diagnostic_closeout"][
                    "source_manifest"
                ]["manifest_sha256"],
                "release_eligible": release_eligible,
            }
        if any(
            observed.get(name) != expected
            for name, expected in expected_outcome.items()
        ):
            raise EvidenceCertificateError(
                f"certificate artifact {row['artifact_id']} disposition is not request-derived"
            )
    return cert


def validate_diagnostic_source_manifest_binding_v1(
    request: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    evidence_root: Path,
) -> dict[str, Any] | None:
    """Recheck the cheap source-manifest identity immediately before release."""

    normalized = validate_evidence_certificate_request_v1(request)
    cert = validate_certificate_request_binding_v1(normalized, certificate)
    if cert["phase"] != "SEALED_PRE_RELEASE":
        raise EvidenceCertificateError(
            "diagnostic source recheck requires a pre-release certificate"
        )
    if "diagnostic_closeout" not in normalized:
        return None
    root = evidence_root.expanduser().resolve(strict=True)
    return _load_diagnostic_source_manifest_v1(
        normalized,
        evidence_root=root,
        sealed_rows=cert["artifacts"],
    )


def validate_certificate_source_manifest_bindings_v1(
    request: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    evidence_root: Path,
    authenticated_absent_paths: set[str] | frozenset[str] = frozenset(),
    require_fresh_diagnostic_review_audit: bool = True,
    recheck_diagnostic_local_live_state: bool = True,
) -> dict[str, Any]:
    """Re-seal source bytes, allowing only actuator-authenticated absences."""

    if (
        type(require_fresh_diagnostic_review_audit) is not bool
        or type(recheck_diagnostic_local_live_state) is not bool
    ):
        raise EvidenceCertificateError(
            "diagnostic source recheck controls must be boolean"
        )
    normalized = validate_evidence_certificate_request_v1(request)
    cert = validate_certificate_request_binding_v1(normalized, certificate)
    root = evidence_root.expanduser().resolve(strict=True)
    if not isinstance(authenticated_absent_paths, (set, frozenset)) or any(
        not isinstance(path, str) for path in authenticated_absent_paths
    ):
        raise EvidenceCertificateError(
            "authenticated_absent_paths must be a set of relative paths"
        )
    authenticated_absent = frozenset(
        str(_relative_path(path, path="authenticated absent artifact path"))
        for path in authenticated_absent_paths
    )
    request_paths = {str(row["path"]) for row in normalized["artifacts"]}
    if not authenticated_absent.issubset(request_paths):
        raise EvidenceCertificateError(
            "authenticated absent paths exceed the certificate request"
        )
    cert_by_id = {str(row["artifact_id"]): dict(row) for row in cert["artifacts"]}
    sealed_rows: list[dict[str, Any]] = []
    observed_absent: set[str] = set()
    for row in normalized["artifacts"]:
        observed = cert_by_id.get(row["artifact_id"])
        try:
            sealed = build_sealed_artifact_inventory_row_v1(
                run_root=root,
                path=root / row["path"],
                artifact_family=row["artifact_family"],
                retention_class="RX",
                owner_writer=row["owner_writer"],
            )
        except FileNotFoundError:
            relative = str(row["path"])
            if relative not in authenticated_absent or observed is None:
                raise
            observed_absent.add(relative)
            digest_name = (
                "sha256" if observed.get("artifact_type") == "file" else "tree_sha256"
            )
            sealed = {
                "path": relative,
                "artifact_family": row["artifact_family"],
                "retention_class": "RX",
                "owner_writer": row["owner_writer"],
                "artifact_type": observed.get("artifact_type"),
                "bytes": observed.get("bytes"),
                digest_name: observed.get(digest_name),
            }
        digest_name = "sha256" if "sha256" in sealed else "tree_sha256"
        if observed is None or any(
            observed.get(name) != sealed.get(name)
            for name in ("path", "artifact_type", "bytes", digest_name)
        ):
            raise EvidenceCertificateError(
                f"certificate artifact {row['artifact_id']} differs from immediately sealed actual bytes"
            )
        sealed_rows.append({**row, **sealed})
    if observed_absent != set(authenticated_absent):
        raise EvidenceCertificateError(
            "authenticated absent paths differ from currently absent certificate rows"
        )
    validation = _validate_sealed_source_manifest_bindings_v1(
        evidence_root=root,
        sealed_rows=sealed_rows,
        require_all=evidence_root_requires_permanent_capsule_v1(root),
        authenticated_absent_paths=authenticated_absent,
    )
    expected_summary = (
        {
            name: validation[name]
            for name in ("binding_count", "manifest_count", "bindings_sha256")
        }
        if validation["binding_count"]
        else None
    )
    if cert.get("source_manifest_binding_summary") != expected_summary:
        raise EvidenceCertificateError(
            "certificate source manifest bindings differ from immediate recheck"
        )
    if "diagnostic_closeout" in normalized:
        _load_diagnostic_source_manifest_v1(
            normalized,
            evidence_root=root,
            sealed_rows=sealed_rows,
            require_fresh_review_audit=require_fresh_diagnostic_review_audit,
            recheck_local_live_state=recheck_diagnostic_local_live_state,
        )
    return validation


def upgrade_evidence_certificate_request_source_bindings_v1(
    request: Mapping[str, Any],
    *,
    evidence_root: Path,
    source_manifest_paths: Sequence[Path],
    diagnostic_upgrade_spec: Mapping[str, Any] | None = None,
    max_manifests: int = MAX_SOURCE_BINDING_MANIFESTS,
    max_rows: int = MAX_SOURCE_BINDING_ROWS,
) -> dict[str, Any]:
    """Return a validated copy bound only to explicitly named source manifests.

    The transformer deliberately tolerates one historical input defect: the
    exact four-field partial ``source_manifest_binding`` shape.  It validates
    that partial identity against the explicitly named manifest row and then
    replaces it.  Normal request validation remains strict.
    """

    if (
        type(max_manifests) is not int
        or not 0 < max_manifests <= MAX_SOURCE_BINDING_MANIFESTS
    ):
        raise EvidenceCertificateError("source binding max_manifests is out of bounds")
    if type(max_rows) is not int or not 0 < max_rows <= MAX_SOURCE_BINDING_ROWS:
        raise EvidenceCertificateError("source binding max_rows is out of bounds")
    if not source_manifest_paths or len(source_manifest_paths) > max_manifests:
        raise EvidenceCertificateError(
            "source binding upgrade requires a bounded non-empty explicit manifest list"
        )
    original = _mapping(request, path="request")
    upgraded = json.loads(json.dumps(original))
    raw_rows = upgraded.get("artifacts")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise EvidenceCertificateError("request artifacts are malformed")

    # Remove only the exact historical partial shape before strict validation.
    # Complete bindings remain in place and are validated normally.
    legacy_partials: dict[str, dict[str, Any]] = {}
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, dict):
            raise EvidenceCertificateError(f"request artifact {index} is malformed")
        existing = raw_row.get("source_manifest_binding")
        if existing is None:
            continue
        if isinstance(existing, Mapping) and "manifest_path" in existing:
            _validate_source_manifest_binding(existing, index=index)
            continue
        partial = _validate_legacy_partial_source_manifest_binding(
            existing, index=index
        )
        artifact_id = _identifier(
            raw_row.get("artifact_id"),
            path=f"request.artifacts[{index}].artifact_id",
        )
        legacy_partials[artifact_id] = partial
        raw_row.pop("source_manifest_binding", None)

    lifecycle_raw = _mapping(
        upgraded.get("artifact_lifecycle_contract"),
        path="request.artifact_lifecycle_contract",
    )
    lifecycle = validate_run_artifact_lifecycle_contract_v1(lifecycle_raw)
    is_diagnostic = lifecycle.get("lifecycle_profile") == DEBUG_FORENSIC
    if diagnostic_upgrade_spec is not None:
        if not is_diagnostic:
            raise EvidenceCertificateError(
                "diagnostic upgrade spec is valid only for DEBUG_FORENSIC"
            )
        spec = _mapping(
            diagnostic_upgrade_spec,
            path="diagnostic_upgrade_spec",
        )
        if (
            set(spec)
            != {
                "schema_version",
                "diagnostic_closeout",
                "artifact_mappings",
            }
            or spec.get("schema_version") != DIAGNOSTIC_UPGRADE_SPEC_SCHEMA_VERSION
        ):
            raise EvidenceCertificateError(
                "diagnostic upgrade spec must contain exact schema_version, diagnostic_closeout, and artifact_mappings"
            )
        supplied_closeout = _validate_diagnostic_closeout(
            spec.get("diagnostic_closeout")
        )
        existing_closeout = upgraded.get("diagnostic_closeout")
        if existing_closeout is not None and (
            _validate_diagnostic_closeout(existing_closeout) != supplied_closeout
        ):
            raise EvidenceCertificateError(
                "diagnostic upgrade spec conflicts with the request closeout"
            )
        upgraded["diagnostic_closeout"] = supplied_closeout
        mappings = _object_list(
            spec.get("artifact_mappings"),
            path="diagnostic_upgrade_spec.artifact_mappings",
            nonempty=True,
        )
        mapping_by_id: dict[str, dict[str, Any]] = {}
        for index, mapping in enumerate(mappings):
            if set(mapping) != {
                "artifact_id",
                "source_artifact_id",
                "source_retention_class",
                "diagnostic_closeout_disposition",
            }:
                raise EvidenceCertificateError(
                    f"diagnostic upgrade artifact mapping {index} fields differ"
                )
            artifact_id = _identifier(
                mapping.get("artifact_id"),
                path=f"diagnostic_upgrade_spec.artifact_mappings[{index}].artifact_id",
            )
            if artifact_id in mapping_by_id:
                raise EvidenceCertificateError(
                    "diagnostic upgrade artifact mappings contain duplicate IDs"
                )
            source_class = _text(
                mapping.get("source_retention_class"),
                path=(
                    "diagnostic_upgrade_spec.artifact_mappings["
                    f"{index}].source_retention_class"
                ),
            )
            disposition = _text(
                mapping.get("diagnostic_closeout_disposition"),
                path=(
                    "diagnostic_upgrade_spec.artifact_mappings["
                    f"{index}].diagnostic_closeout_disposition"
                ),
            )
            if source_class not in RETENTION_CLASSES:
                raise EvidenceCertificateError(
                    "diagnostic upgrade source retention class is unsupported"
                )
            if disposition not in DIAGNOSTIC_CLOSEOUT_DISPOSITIONS:
                raise EvidenceCertificateError(
                    "diagnostic upgrade closeout disposition is unsupported"
                )
            mapping_by_id[artifact_id] = {
                "source_artifact_id": _identifier(
                    mapping.get("source_artifact_id"),
                    path=(
                        "diagnostic_upgrade_spec.artifact_mappings["
                        f"{index}].source_artifact_id"
                    ),
                ),
                "source_retention_class": source_class,
                "diagnostic_closeout_disposition": disposition,
            }
        request_ids = {
            _identifier(
                row.get("artifact_id"),
                path=f"request.artifacts[{index}].artifact_id",
            )
            for index, row in enumerate(raw_rows)
        }
        if set(mapping_by_id) != request_ids:
            raise EvidenceCertificateError(
                "diagnostic upgrade artifact mappings must exactly cover request artifact IDs"
            )
        for row in raw_rows:
            artifact_id = str(row["artifact_id"])
            expected = mapping_by_id[artifact_id]
            for name, value in expected.items():
                if name in row and row[name] != value:
                    raise EvidenceCertificateError(
                        f"diagnostic upgrade mapping conflicts for artifact {artifact_id}"
                    )
                row[name] = value
    elif is_diagnostic and upgraded.get("diagnostic_closeout") is None:
        raise EvidenceCertificateError(
            "older DEBUG_FORENSIC requests require an explicit diagnostic upgrade spec"
        )

    # This validates every non-binding request law, including the complete
    # DEBUG closeout and exact artifact mapping, before reading manifests.
    normalized = validate_evidence_certificate_request_v1(upgraded)
    root = evidence_root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise EvidenceCertificateError("evidence_root must be a directory")
    indexed: dict[str, list[tuple[dict[str, Any], dict[str, Any], str]]] = {}
    seen_manifest_paths: set[str] = set()
    manifests_by_relative: dict[str, dict[str, Any]] = {}
    total_rows = 0
    for supplied_path in source_manifest_paths:
        unresolved = supplied_path.expanduser()
        if not unresolved.is_absolute():
            unresolved = root / unresolved
        if unresolved.is_symlink():
            raise EvidenceCertificateError(
                "source binding manifest cannot be a symlink"
            )
        manifest_path = unresolved.resolve(strict=True)
        try:
            relative_manifest = str(manifest_path.relative_to(root)).replace("\\", "/")
        except ValueError as exc:
            raise EvidenceCertificateError(
                "source binding manifest escapes evidence_root"
            ) from exc
        if str(manifest_path) in seen_manifest_paths:
            raise EvidenceCertificateError(
                "source binding manifest list has duplicates"
            )
        seen_manifest_paths.add(str(manifest_path))
        manifest = validate_artifact_disposition_manifest_v1(
            load_json_object(manifest_path)
        )
        if manifest.get("inventory") != {
            "mode": "producer_declared",
            "complete": True,
        }:
            raise EvidenceCertificateError(
                "source binding upgrade requires complete producer-declared manifests"
            )
        manifests_by_relative[relative_manifest] = manifest
        runtime_root = Path(str(manifest.get("runtime_root"))).expanduser()
        if runtime_root.is_symlink():
            raise EvidenceCertificateError(
                "source manifest runtime_root cannot be a symlink"
            )
        runtime_root = runtime_root.resolve(strict=True)
        try:
            runtime_root.relative_to(root)
        except ValueError as exc:
            raise EvidenceCertificateError(
                "source manifest runtime_root escapes evidence_root"
            ) from exc
        rows = manifest.get("artifacts", [])
        total_rows += len(rows)
        if total_rows > max_rows:
            raise EvidenceCertificateError(
                "source binding manifest row budget exceeded"
            )
        for source_row in rows:
            if not isinstance(source_row, Mapping):
                raise EvidenceCertificateError(
                    "source binding manifest row is malformed"
                )
            row = dict(source_row)
            if row.get("exact_object_identity") is not True:
                continue
            source_target = runtime_root / _relative_path(
                row.get("path"), path="source binding manifest artifact path"
            )
            absolute_key = str(source_target.resolve(strict=False))
            indexed.setdefault(absolute_key, []).append(
                (manifest, row, relative_manifest)
            )

    if is_diagnostic:
        diagnostic_path = normalized["diagnostic_closeout"]["source_manifest"]["path"]
        diagnostic_manifest = manifests_by_relative.get(diagnostic_path)
        if diagnostic_manifest is None:
            raise EvidenceCertificateError(
                "diagnostic closeout source manifest must be explicitly named for upgrade"
            )
        expected_source_identity = normalized["diagnostic_closeout"]["source_manifest"]
        observed_source_identity = {
            "path": diagnostic_path,
            "manifest_sha256": diagnostic_manifest["manifest_sha256"],
            "campaign_id": diagnostic_manifest["campaign_id"],
            "run_id": diagnostic_manifest["run_id"],
            "plan_sha256": diagnostic_manifest["plan_sha256"],
            "lifecycle_contract_sha256": diagnostic_manifest[
                "lifecycle_contract_sha256"
            ],
            **(
                {"runtime_root": diagnostic_manifest["runtime_root"]}
                if "runtime_root" in expected_source_identity
                else {}
            ),
        }
        if observed_source_identity != expected_source_identity:
            raise EvidenceCertificateError(
                "diagnostic upgrade source manifest identity differs from diagnostic_closeout"
            )

    for index, raw_row in enumerate(raw_rows):
        request_path = root / normalized["artifacts"][index]["path"]
        matches = indexed.get(str(request_path.resolve(strict=False)), [])
        if len(matches) != 1:
            raise EvidenceCertificateError(
                f"artifact {normalized['artifacts'][index]['artifact_id']} has no unique explicit source manifest row"
            )
        manifest, source_row, relative_manifest = matches[0]
        digest_fields = [
            name
            for name in ("sha256", "tree_sha256")
            if isinstance(source_row.get(name), str)
        ]
        if len(digest_fields) != 1:
            raise EvidenceCertificateError(
                "source binding row must contain exactly one sealed digest"
            )
        binding = {
            "manifest_path": relative_manifest,
            "manifest_sha256": manifest["manifest_sha256"],
            "artifact_id": source_row["artifact_id"],
            "bytes": source_row["bytes"],
            "retention_class": source_row["retention_class"],
            digest_fields[0]: source_row[digest_fields[0]],
        }
        artifact_id = normalized["artifacts"][index]["artifact_id"]
        partial = legacy_partials.get(artifact_id)
        if partial is not None:
            expected_partial = {
                name: binding[name]
                for name in (
                    "artifact_id",
                    "bytes",
                    "retention_class",
                    digest_fields[0],
                )
            }
            if partial != expected_partial:
                raise EvidenceCertificateError(
                    f"artifact {artifact_id} legacy partial source binding differs from the named manifest row"
                )
        if is_diagnostic:
            expected_diagnostic = {
                "source_artifact_id": binding["artifact_id"],
                "source_retention_class": binding["retention_class"],
            }
            if any(
                raw_row.get(name) != value
                for name, value in expected_diagnostic.items()
            ):
                raise EvidenceCertificateError(
                    f"diagnostic upgrade mapping differs from the named source row for {artifact_id}"
                )
        existing = raw_row.get("source_manifest_binding")
        if existing is not None and existing != binding:
            raise EvidenceCertificateError(
                f"artifact {artifact_id} already has a conflicting source binding"
            )
        raw_row["source_manifest_binding"] = binding
    validated_upgrade = validate_evidence_certificate_request_v1(upgraded)
    sealed_rows: list[dict[str, Any]] = []
    for row in validated_upgrade["artifacts"]:
        sealed = build_sealed_artifact_inventory_row_v1(
            run_root=root,
            path=root / row["path"],
            artifact_family=row["artifact_family"],
            retention_class="RX",
            owner_writer=row["owner_writer"],
        )
        sealed_rows.append({**row, **sealed})
    _validate_sealed_source_manifest_bindings_v1(
        evidence_root=root,
        sealed_rows=sealed_rows,
        require_all=True,
    )
    if is_diagnostic:
        _load_diagnostic_source_manifest_v1(
            validated_upgrade,
            evidence_root=root,
            sealed_rows=sealed_rows,
        )
    return validated_upgrade


def certificate_disposition_manifest_v1(
    request: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    evidence_root: Path,
    certificate_path: Path | None = None,
    permanent_publication: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile certificate decisions into the existing manifest authority."""

    normalized = validate_evidence_certificate_request_v1(request)
    cert = validate_certificate_request_binding_v1(normalized, certificate)
    if cert["phase"] != "SEALED_PRE_RELEASE":
        raise EvidenceCertificateError(
            "release disposition manifest requires a pre-release certificate"
        )
    root = evidence_root.expanduser().resolve(strict=True)
    diagnostic = normalized.get("diagnostic_closeout")
    diagnostic_source_manifest: dict[str, Any] | None = None
    requires_permanent = evidence_root_requires_permanent_capsule_v1(root)
    if requires_permanent or cert.get("source_manifest_binding_summary") is not None:
        validate_certificate_source_manifest_bindings_v1(
            normalized, cert, evidence_root=root
        )
    if diagnostic is not None:
        diagnostic_source_manifest = _load_diagnostic_source_manifest_v1(
            normalized,
            evidence_root=root,
            sealed_rows=cert["artifacts"],
            require_fresh_review_audit=False,
            recheck_local_live_state=False,
        )
    publication = None
    if (
        diagnostic is not None
        or requires_permanent
        or permanent_publication is not None
    ):
        if permanent_publication is None:
            raise EvidenceCertificateError(
                "certificate release manifest requires a verified permanent pre-release capsule"
            )
        from .permanent_evidence_capsule import (
            validate_permanent_evidence_publication_v1,
        )

        publication = validate_permanent_evidence_publication_v1(
            permanent_publication,
            certificate=cert,
            source_root=root,
            require_outstanding_final_capacity=True,
        )
        if publication.get("phase") != "SEALED_PRE_RELEASE":
            raise EvidenceCertificateError(
                "certificate release requires a pre-release permanent capsule"
            )
    inventory: list[dict[str, Any]] = []
    for row in cert["artifacts"]:
        inventory.append(
            {
                "artifact_id": row["artifact_id"],
                "path": row["path"],
                "artifact_family": row["artifact_family"],
                "artifact_type": row["artifact_type"],
                "owner_writer": row["owner_writer"],
                "bytes": row["bytes"],
                **({"sha256": row["sha256"]} if "sha256" in row else {}),
                **({"tree_sha256": row["tree_sha256"]} if "tree_sha256" in row else {}),
                "retention_class": row["retention_class"],
                "live_reference_status": (
                    "active"
                    if (
                        any(row["live_state"].values())
                        or row["disposition"] == "HOLD_UNTIL_LAST_USE"
                    )
                    else "none"
                ),
                "audit_hold_status": (
                    "active"
                    if (
                        "audit_exact_bytes" in row["retention_authorities"]
                        or row["artifact_id"]
                        in normalized["audit_context"]["exact_required_artifact_ids"]
                    )
                    else "not_required"
                ),
                "qualification_hold_status": (
                    "active"
                    if (
                        "qualification_exact_bytes" in row["retention_authorities"]
                        or row["artifact_id"]
                        in normalized["qualification_context"][
                            "exact_required_artifact_ids"
                        ]
                    )
                    else "not_required"
                ),
                "replacement_verified": row["retention_class"] in {"R0", "R1", "R2"},
                "release_predicate": row["disposition_reason"],
                "reason": row["disposition_reason"],
                "evidence_certificate_id": cert["certificate_id"],
                "evidence_certificate_sha256": cert["certificate_sha256"],
                "certificate_disposition": row["disposition"],
                "downstream_consumers": row["consumers"],
                "last_use": row["last_use"],
                "evidence_class": row["evidence_class"],
                **(
                    {"source_retention_class": row["source_retention_class"]}
                    if "source_retention_class" in row
                    else {}
                ),
            }
        )
    if certificate_path is not None:
        path = certificate_path.expanduser().resolve(strict=True)
        inventory.append(
            build_sealed_artifact_inventory_row_v1(
                run_root=root,
                path=path,
                artifact_family="evidence_certificate",
                retention_class="R5",
                owner_writer="evidence-certificate-reclaim",
                evidence_certificate_id=cert["certificate_id"],
            )
        )
    plan = {
        "campaign_id": normalized["programme_id"],
        "plan_sha256": normalized["identity"]["plan_sha256"],
        "artifact_lifecycle": normalized["artifact_lifecycle_contract"],
    }
    manifest = build_artifact_disposition_manifest_v1(
        plan,
        run_root=root,
        terminal_status="completed",
        live_reference_status="none",
        audit_hold_status=(
            "active"
            if normalized["audit_context"]["status"] == "active"
            else (
                "released"
                if normalized["audit_context"]["status"] == "released"
                else "not_required"
            )
        ),
        qualification_hold_status=(
            "active"
            if normalized["qualification_context"]["status"] == "active"
            else "not_required"
        ),
        replacement_verified=True,
        artifact_inventory=inventory,
        inventory_complete=True,
    )
    core = dict(manifest)
    core.pop("manifest_sha256", None)
    cert_rows_by_id = {str(row["artifact_id"]): dict(row) for row in cert["artifacts"]}
    if cert.get("source_manifest_binding_summary") is not None:
        updated_rows: list[dict[str, Any]] = []
        for raw_row in core["artifacts"]:
            manifest_row = dict(raw_row)
            cert_row = cert_rows_by_id.get(str(manifest_row.get("artifact_id")))
            if (
                cert_row is not None
                and cert_row.get("source_manifest_binding") is not None
            ):
                manifest_row["source_manifest_binding"] = cert_row[
                    "source_manifest_binding"
                ]
                manifest_row["source_retention_class"] = cert_row[
                    "source_retention_class"
                ]
            updated_rows.append(manifest_row)
        core["artifacts"] = updated_rows
    source_authority = (
        {
            "source_manifest_binding_count": cert["source_manifest_binding_summary"][
                "binding_count"
            ],
            "source_manifest_bindings_sha256": cert["source_manifest_binding_summary"][
                "bindings_sha256"
            ],
        }
        if cert.get("source_manifest_binding_summary") is not None
        else {}
    )
    publication_authority = (
        {
            "permanent_capsule_publication_sha256": publication["publication_sha256"],
            "permanent_capsule_sha256": publication["capsule_sha256"],
        }
        if publication is not None
        else {}
    )
    if diagnostic is None:
        core["release_authority"] = {
            "kind": "EVIDENCE_CERTIFICATE_GATE_CLOSURE",
            "certificate_id": cert["certificate_id"],
            "certificate_sha256": cert["certificate_sha256"],
            "request_sha256": cert["request_sha256"],
            "programme_plan_sha256": cert["programme_plan"]["plan_sha256"],
            "gate_id": cert["gate_id"],
            **source_authority,
            **publication_authority,
            "ordinary_releasable_archive_policy": "DIRECT_RELEASE_NO_ARCHIVE",
        }
    else:
        assert diagnostic_source_manifest is not None
        cert_rows = cert_rows_by_id
        source_rows_by_path = {
            str(row["path"]): dict(row)
            for row in diagnostic_source_manifest["artifacts"]
            if isinstance(row, Mapping)
        }
        manifested_rows: list[dict[str, Any]] = []
        accounted_ids: set[str] = set()
        for raw_row in core["artifacts"]:
            manifest_row = dict(raw_row)
            cert_row = cert_rows.get(str(manifest_row.get("artifact_id")))
            if cert_row is not None:
                accounted_ids.add(str(cert_row["artifact_id"]))
                source_row = source_rows_by_path.get(str(cert_row["path"]))
                if source_row is None:
                    raise EvidenceCertificateError(
                        "diagnostic derived manifest lacks its exact source row"
                    )
                superseded = (
                    cert_row["diagnostic_closeout_disposition"]
                    == "SUPERSEDED_BY_CERTIFICATE"
                )
                manifest_row.update(
                    {
                        "retention_class": cert_row["retention_class"],
                        "release_predicate": cert_row["disposition_reason"],
                        "release_eligible": cert_row["release_eligible"],
                        "disposition": (
                            "RELEASE"
                            if cert_row["release_eligible"]
                            else "RETAIN_DURABLE"
                            if cert_row["retention_class"] in {"R3", "R4", "R5"}
                            else "RX"
                            if cert_row["retention_class"] == "RX"
                            else "KEEP_ACTIVE"
                        ),
                        "reason": cert_row["disposition_reason"],
                        "live_reference_status": (
                            "none"
                            if superseded
                            else str(
                                source_row.get("live_reference_status") or "unknown"
                            )
                        ),
                        "audit_hold_status": str(
                            source_row.get("audit_hold_status") or "unknown"
                        ),
                        "qualification_hold_status": str(
                            source_row.get("qualification_hold_status") or "unknown"
                        ),
                        "source_artifact_id": cert_row["source_artifact_id"],
                        "source_retention_class": cert_row["source_retention_class"],
                        "diagnostic_closeout_disposition": cert_row[
                            "diagnostic_closeout_disposition"
                        ],
                        "retention_transition": cert_row["retention_transition"],
                        "replacement_artifact_identity": (
                            publication["capsule_sha256"]
                            if cert_row["release_eligible"]
                            else manifest_row.get("replacement_artifact_identity")
                        ),
                    }
                )
            manifested_rows.append(manifest_row)
        if accounted_ids != set(cert_rows):
            raise EvidenceCertificateError(
                "diagnostic derived manifest does not exactly account for certificate rows"
            )
        core["artifacts"] = manifested_rows
        source_rows = [
            row for row in manifested_rows if row.get("artifact_id") in cert_rows
        ]
        superseded_ids = list(diagnostic["superseded_artifact_ids"])
        preserved_ids = sorted(set(cert_rows) - set(superseded_ids))
        core["summary"] = {
            **core["summary"],
            "produced_bytes": sum(int(row["bytes"]) for row in manifested_rows),
            "release_eligible_bytes": sum(
                int(row["bytes"])
                for row in manifested_rows
                if row.get("release_eligible") is True
            ),
            "unresolved_rx_bytes": sum(
                int(row["bytes"])
                for row in manifested_rows
                if row.get("retention_class") == "RX"
            ),
            "by_retention_class": {
                retention_class: {
                    "objects": sum(
                        1
                        for row in manifested_rows
                        if row.get("retention_class") == retention_class
                    ),
                    "bytes": sum(
                        int(row["bytes"])
                        for row in manifested_rows
                        if row.get("retention_class") == retention_class
                    ),
                }
                for retention_class in RETENTION_CLASSES
            },
        }

        def aggregate_hold_status(name: str) -> str:
            statuses = {str(row.get(name) or "unknown") for row in source_rows}
            if "active" in statuses:
                return "active"
            unknown = statuses - {"released", "not_required"}
            if unknown:
                return "unknown"
            return "released" if "released" in statuses else "not_required"

        core["retention"] = {
            **core["retention"],
            "audit_hold_status": aggregate_hold_status("audit_hold_status"),
            "qualification_hold_status": aggregate_hold_status(
                "qualification_hold_status"
            ),
            "unknown_artifact_scope": "exact_object",
        }
        core["prune"] = {
            **core["prune"],
            "automatic_prune_allowed": True,
            "automatic_prune_eligible": bool(superseded_ids),
        }
        core["release_authority"] = {
            "kind": "EVIDENCE_CERTIFICATE_DIAGNOSTIC_CLOSEOUT",
            "certificate_id": cert["certificate_id"],
            "certificate_sha256": cert["certificate_sha256"],
            "request_sha256": cert["request_sha256"],
            "programme_plan_sha256": cert["programme_plan"]["plan_sha256"],
            "gate_id": cert["gate_id"],
            "diagnostic_closeout_sha256": canonical_sha256(diagnostic),
            "source_manifest": diagnostic["source_manifest"],
            "substantive_review": diagnostic["substantive_review"],
            "closure_receipt_sha256": diagnostic["closure_receipt_sha256"],
            "closed_live_state": diagnostic["live_state"],
            "closed_exact_byte_holds": diagnostic["active_exact_byte_holds"],
            **(
                {
                    "source_live_marker_closeout": diagnostic[
                        "source_live_marker_closeout"
                    ]
                }
                if "source_live_marker_closeout" in diagnostic
                else {}
            ),
            "source_row_count": len(source_rows),
            "accounted_source_row_count": len(accounted_ids),
            "superseded_row_count": len(superseded_ids),
            "preserved_row_count": len(preserved_ids),
            "superseded_artifact_ids": superseded_ids,
            "preserved_artifact_ids": preserved_ids,
            "permanent_capsule_publication_sha256": publication["publication_sha256"],
            "permanent_capsule_sha256": publication["capsule_sha256"],
            **source_authority,
            "ordinary_releasable_archive_policy": "DIRECT_RELEASE_NO_ARCHIVE",
        }
    return validate_artifact_disposition_manifest_v1(
        {**core, "manifest_sha256": canonical_sha256(core)}
    )


def build_reclamation_intent_v1(
    *,
    request: Mapping[str, Any],
    certificate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    status: str = "PREPARED",
    created_at_utc: str,
    previous_intent_sha256: str | None = None,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized = validate_evidence_certificate_request_v1(request)
    cert = validate_evidence_certificate_v1(certificate)
    disposition = validate_artifact_disposition_manifest_v1(manifest)
    if status not in {"PREPARED", "APPLYING", "APPLIED", "PARTIAL_FAILURE"}:
        raise EvidenceCertificateError("reclamation intent status is unsupported")
    rows = [
        {
            "path": _relative_path(row.get("path"), path="intent.selected.path"),
            "bytes": int(row.get("bytes", 0)),
            "sha256": row.get("sha256"),
            "tree_sha256": row.get("tree_sha256"),
        }
        for row in selected
    ]
    core = {
        "schema_version": INTENT_SCHEMA_VERSION,
        "status": status,
        "programme_id": normalized["programme_id"],
        "gate_id": normalized["gate_id"],
        "request_sha256": canonical_sha256(normalized),
        "certificate_id": cert["certificate_id"],
        "certificate_sha256": cert["certificate_sha256"],
        "manifest_sha256": disposition["manifest_sha256"],
        "programme_plan_sha256": normalized["programme_plan"]["plan_sha256"],
        "selected": rows,
        "selected_bytes": sum(int(row["bytes"]) for row in rows),
        "created_at_utc": created_at_utc,
        "previous_intent_sha256": previous_intent_sha256,
        "result": dict(result or {}),
    }
    return {**core, "intent_sha256": canonical_sha256(core)}


def validate_reclamation_intent_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    intent = _mapping(value, path="intent")
    supplied = intent.pop("intent_sha256", None)
    if intent.get("schema_version") != INTENT_SCHEMA_VERSION:
        raise EvidenceCertificateError("reclamation intent schema mismatch")
    if intent.get("status") not in {
        "PREPARED",
        "APPLYING",
        "APPLIED",
        "PARTIAL_FAILURE",
    }:
        raise EvidenceCertificateError("reclamation intent status is unsupported")
    if supplied != canonical_sha256(intent):
        raise EvidenceCertificateError("reclamation intent SHA-256 mismatch")
    return {**intent, "intent_sha256": supplied}


def validate_storage_receipt_certificate_binding_v1(
    storage_receipt: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    pre_certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the complete successful-release tuple used by FINAL certificates."""

    disposition = validate_artifact_disposition_manifest_v1(manifest)
    certificate = validate_evidence_certificate_v1(pre_certificate)
    if certificate["phase"] != "SEALED_PRE_RELEASE":
        raise EvidenceCertificateError(
            "storage receipt must bind a pre-release certificate"
        )
    receipt = _mapping(storage_receipt, path="storage_receipt")
    supplied = receipt.pop("receipt_sha256", None)
    if supplied != canonical_sha256(receipt):
        raise EvidenceCertificateError("storage disposition receipt SHA-256 mismatch")
    required = {
        "schema_version",
        "campaign_id",
        "run_id",
        "lifecycle_contract_sha256",
        "manifest_sha256",
        "plan_sha256",
        "bytes_before",
        "bytes_retained",
        "bytes_released",
        "bytes_unresolved_RX",
        "removed",
        "authorization_reason",
        "replacement_references",
        "retention_references",
        "timestamp_utc",
        "result",
    }
    if set(receipt) != required:
        raise EvidenceCertificateError(
            "successful storage disposition receipt fields differ from the exact schema"
        )
    if receipt.get("schema_version") != STORAGE_RECEIPT_SCHEMA_VERSION:
        raise EvidenceCertificateError("storage disposition receipt schema mismatch")
    if receipt.get("result") != "applied":
        raise EvidenceCertificateError(
            "FINAL certificate requires an applied storage receipt"
        )
    expected_manifest_fields = {
        "campaign_id": disposition.get("campaign_id"),
        "run_id": disposition.get("run_id"),
        "lifecycle_contract_sha256": disposition.get("lifecycle_contract_sha256"),
        "manifest_sha256": disposition["manifest_sha256"],
        "plan_sha256": disposition.get("plan_sha256"),
    }
    mismatched_manifest = sorted(
        name
        for name, expected in expected_manifest_fields.items()
        if receipt.get(name) != expected
    )
    if mismatched_manifest:
        raise EvidenceCertificateError(
            "storage receipt differs from manifest at: " + ",".join(mismatched_manifest)
        )
    for name in (
        "bytes_before",
        "bytes_retained",
        "bytes_released",
        "bytes_unresolved_RX",
    ):
        if type(receipt.get(name)) is not int or int(receipt[name]) < 0:
            raise EvidenceCertificateError(
                f"storage disposition receipt {name} must be non-negative"
            )
    removed = receipt.get("removed")
    if not isinstance(removed, list) or not all(
        isinstance(row, Mapping) for row in removed
    ):
        raise EvidenceCertificateError(
            "storage receipt removed must be an array of rows"
        )
    removed_by_path = {str(row.get("path")): dict(row) for row in removed}
    if len(removed_by_path) != len(removed):
        raise EvidenceCertificateError("storage receipt removed paths must be unique")
    certificate_release = {
        str(row["path"]): dict(row)
        for row in certificate["artifacts"]
        if row.get("release_eligible") is True
    }
    manifest_release = {
        str(row.get("path")): dict(row)
        for row in disposition.get("artifacts", [])
        if isinstance(row, Mapping) and row.get("release_eligible") is True
    }
    if set(removed_by_path) != set(certificate_release) or set(manifest_release) != set(
        certificate_release
    ):
        raise EvidenceCertificateError(
            "storage receipt removed rows differ from selected pre-certificate release rows"
        )
    for path, cert_row in certificate_release.items():
        receipt_row = removed_by_path[path]
        manifest_row = manifest_release[path]
        digest_name = "sha256" if "sha256" in cert_row else "tree_sha256"
        expected_identity = {
            "path": path,
            "bytes": cert_row["bytes"],
            "retention_class": cert_row["retention_class"],
            digest_name: cert_row[digest_name],
        }
        if any(
            manifest_row.get(name) != expected
            for name, expected in expected_identity.items()
        ):
            raise EvidenceCertificateError(
                f"storage receipt removed identity differs for {path}"
            )
        checkpoint_role = manifest_row.get("checkpoint_role")
        family = str(manifest_row.get("artifact_family") or "").lower()
        retention_class = str(manifest_row["retention_class"])
        if checkpoint_role == "intermediate":
            category = "intermediate_checkpoints"
        elif "developmental" in family and "trace" in family:
            category = "developmental_traces"
        elif "event" in family:
            category = "event_jsonl"
        elif "trace" in family or "debug" in family:
            category = "debug_traces"
        elif retention_class == "R2":
            category = "reproducible_derived"
        else:
            category = "scratch"
        expected_removed_row = {
            "path": path,
            "bytes": cert_row["bytes"],
            "category": category,
            "retention_class": retention_class,
            "checkpoint_role": checkpoint_role,
            "release_predicate": manifest_row.get("release_predicate"),
            "artifact_type": manifest_row.get("artifact_type", "file"),
            digest_name: cert_row[digest_name],
        }
        if receipt_row != expected_removed_row:
            raise EvidenceCertificateError(
                f"storage receipt removed row is not the exact selected row for {path}"
            )
    expected_released = sum(int(row["bytes"]) for row in certificate_release.values())
    expected_before = int(disposition["summary"]["produced_bytes"])
    if (
        receipt["bytes_released"] != expected_released
        or receipt["bytes_before"] != expected_before
        or receipt["bytes_retained"] != expected_before - expected_released
        or receipt["bytes_unresolved_RX"]
        != int(disposition["summary"]["unresolved_rx_bytes"])
    ):
        raise EvidenceCertificateError(
            "storage receipt byte totals differ from manifest/certificate accounting"
        )
    expected_replacement = [
        {
            "certificate_id": certificate["certificate_id"],
            "certificate_sha256": certificate["certificate_sha256"],
        }
    ]
    if receipt.get("replacement_references") != expected_replacement:
        raise EvidenceCertificateError(
            "storage receipt replacement reference differs from pre-certificate"
        )
    retention_references = receipt.get("retention_references")
    if (
        not isinstance(retention_references, list)
        or not retention_references
        or not all(
            isinstance(value, str) and value.strip() for value in retention_references
        )
        or len(retention_references) != len(set(retention_references))
    ):
        raise EvidenceCertificateError("storage receipt lacks its retention reference")
    certificate_rows = [
        dict(row)
        for row in disposition.get("artifacts", [])
        if isinstance(row, Mapping)
        and row.get("evidence_certificate_id") == certificate["certificate_id"]
        and row.get("retention_class") == "R5"
        and row.get("artifact_family") == "evidence_certificate"
        and row.get("owner_writer") == "evidence-certificate-reclaim"
    ]
    if len(certificate_rows) != 1:
        raise EvidenceCertificateError(
            "FINAL storage receipt requires one exact retained pre-certificate row"
        )
    runtime_root = (
        Path(str(disposition["runtime_root"])).expanduser().resolve(strict=True)
    )
    certificate_path = (
        runtime_root
        / _relative_path(
            certificate_rows[0].get("path"),
            path="manifest retained pre-certificate path",
        )
    ).resolve(strict=True)
    if retention_references != [str(certificate_path)]:
        raise EvidenceCertificateError(
            "storage receipt retention reference differs from the retained pre-certificate"
        )
    observed_certificate_row = build_sealed_artifact_inventory_row_v1(
        run_root=runtime_root,
        path=certificate_path,
        artifact_family=str(certificate_rows[0]["artifact_family"]),
        retention_class="R5",
        owner_writer=str(certificate_rows[0]["owner_writer"]),
        evidence_certificate_id=certificate["certificate_id"],
    )
    if observed_certificate_row != {
        name: certificate_rows[0][name] for name in observed_certificate_row
    }:
        raise EvidenceCertificateError(
            "retained pre-certificate bytes differ from the disposition manifest"
        )
    staging_root = certificate_path.parent / ".release-staging"
    for path in certificate_release:
        source = runtime_root / path
        staged = staging_root / (
            canonical_sha256({"path": path})[:24] + "-" + Path(path).name
        )
        if (
            source.exists()
            or source.is_symlink()
            or staged.exists()
            or staged.is_symlink()
        ):
            raise EvidenceCertificateError(
                f"storage receipt claims release while exact bytes still exist: {path}"
            )
    _text(
        receipt.get("authorization_reason"), path="storage_receipt.authorization_reason"
    )
    _text(receipt.get("timestamp_utc"), path="storage_receipt.timestamp_utc")
    return {**receipt, "receipt_sha256": supplied}


def finalize_evidence_certificate_v1(
    certificate: Mapping[str, Any],
    *,
    storage_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    finalized_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a post-action certificate revision without losing pre-release identity."""

    previous = validate_evidence_certificate_v1(certificate)
    if previous["phase"] != "SEALED_PRE_RELEASE":
        raise EvidenceCertificateError(
            "only a pre-release certificate can be finalized"
        )
    receipt = validate_storage_receipt_certificate_binding_v1(
        storage_receipt,
        manifest=manifest,
        pre_certificate=previous,
    )
    receipt_sha256 = receipt["receipt_sha256"]
    receipt_timestamp = str(receipt["timestamp_utc"])
    if finalized_at_utc is not None and finalized_at_utc != receipt_timestamp:
        raise EvidenceCertificateError(
            "FINAL timestamp must be the deterministic storage receipt timestamp"
        )
    finalized_at_utc = receipt_timestamp
    released_paths = {
        str(row.get("path"))
        for row in receipt.get("removed", [])
        if isinstance(row, Mapping)
    }
    rows = []
    for row in previous["artifacts"]:
        rows.append(
            {
                **row,
                "released": row["path"] in released_paths,
                "released_at_utc": (
                    finalized_at_utc if row["path"] in released_paths else None
                ),
                "release_receipt_sha256": (
                    receipt_sha256 if row["path"] in released_paths else None
                ),
            }
        )
    core = dict(previous)
    core.pop("certificate_sha256", None)
    core.update(
        {
            "phase": "FINAL",
            "prior_certificate_sha256": previous["certificate_sha256"],
            "finalized_at_utc": finalized_at_utc,
            "storage_disposition_receipt_sha256": receipt_sha256,
            "artifacts": rows,
            "disposition_summary": {
                **previous["disposition_summary"],
                "released_bytes": int(receipt.get("bytes_released", 0)),
                "retained_bytes": max(
                    0,
                    int(previous["disposition_summary"]["produced_bytes"])
                    - int(receipt.get("bytes_released", 0)),
                ),
            },
        }
    )
    final = {**core, "certificate_sha256": canonical_sha256(core)}
    return validate_evidence_certificate_v1(final)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(pretty_json(dict(value)), encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvidenceCertificateError(f"expected JSON object: {path}")
    return value


__all__ = [
    "CERTIFICATE_SCHEMA_VERSION",
    "DIAGNOSTIC_LOCAL_LIVE_STATE_AUDIT_SCHEMA_VERSION",
    "DIAGNOSTIC_SOURCE_LIVE_MARKER_CLOSEOUT_SCHEMA_VERSION",
    "DIAGNOSTIC_UPGRADE_SPEC_SCHEMA_VERSION",
    "DISPOSITIONS",
    "EVIDENCE_CLASSES",
    "EvidenceCertificateError",
    "HISTORICAL_AUDIT_SCHEMA_VERSION",
    "INTENT_SCHEMA_VERSION",
    "MAX_SOURCE_BINDING_MANIFESTS",
    "MAX_SOURCE_BINDING_ROWS",
    "REQUEST_SCHEMA_VERSION",
    "build_evidence_certificate_v1",
    "build_reclamation_intent_v1",
    "certificate_disposition_manifest_v1",
    "diagnostic_local_live_state_audit_v1",
    "evidence_root_requires_permanent_capsule_v1",
    "finalize_evidence_certificate_v1",
    "load_json_object",
    "require_diagnostic_local_live_state_clear_v1",
    "utc_now",
    "validate_evidence_certificate_request_v1",
    "validate_evidence_certificate_v1",
    "validate_certificate_request_binding_v1",
    "validate_certificate_source_manifest_bindings_v1",
    "validate_diagnostic_source_manifest_binding_v1",
    "validate_diagnostic_source_live_marker_closeout_v1",
    "validate_diagnostic_source_live_marker_freshness_v1",
    "validate_reclamation_intent_v1",
    "validate_storage_receipt_certificate_binding_v1",
    "upgrade_evidence_certificate_request_source_bindings_v1",
    "write_json_atomic",
]
