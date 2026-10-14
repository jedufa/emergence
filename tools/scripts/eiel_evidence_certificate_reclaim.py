#!/usr/bin/env python3
"""Dry-run-first gate-boundary evidence certification and reclamation.

Ordinary certificate-backed transient evidence is released in place through
the existing exact manifest/prune authority.  This command never invokes the
archive worker.  Historical mode is bounded and read-only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from engine.campaigns.evidence_certificate_api import (
    ArtifactLifecycleError,
    DATA2_FILESYSTEM_ROOT,
    DEBUG_CLOSEOUT_PREPARATION_PLAN_SCHEMA_VERSION,
    DEBUG_CLOSEOUT_REVIEW_INPUT_SCHEMA_VERSION,
    DEFAULT_MAX_CAPSULE_BYTES,
    DEFAULT_MAX_PERMANENT_ROOT_BYTES,
    DIAGNOSTIC_UPGRADE_SPEC_SCHEMA_VERSION,
    EvidenceCertificateError,
    MAX_SOURCE_BINDING_MANIFESTS,
    MAX_SOURCE_BINDING_ROWS,
    RootRetirementError,
    apply_root_retirement_v1,
    build_permanent_evidence_capsule_v1,
    build_evidence_certificate_v1,
    build_debug_closeout_bundle_v1,
    build_reclamation_intent_v1,
    build_root_retirement_intent_v1,
    build_root_retirement_inventory_v1,
    build_storage_disposition_receipt_v1,
    canonical_sha256,
    certificate_disposition_manifest_v1,
    default_permanent_evidence_root_v1,
    draft_debug_closeout_review_spec_v1,
    evidence_root_requires_permanent_capsule_v1,
    finalize_evidence_certificate_v1,
    HISTORICAL_AUDIT_SCHEMA_VERSION,
    load_json_object,
    publish_permanent_evidence_capsule_v1,
    recover_existing_permanent_evidence_publication_v1,
    pretty_json,
    run_artifact_transaction_fence_v1,
    utc_now,
    validate_artifact_disposition_manifest_v1,
    validate_certificate_request_binding_v1,
    validate_certificate_source_manifest_bindings_v1,
    validate_evidence_certificate_request_v1,
    validate_evidence_certificate_v1,
    validate_reclamation_intent_v1,
    validate_permanent_evidence_publication_v1,
    validate_root_retirement_intent_v1,
    validate_root_retirement_inventory_v1,
    validate_root_retirement_review_v1,
    validate_root_retirement_request_v1,
    validate_storage_receipt_certificate_binding_v1,
    upgrade_evidence_certificate_request_source_bindings_v1,
    write_json_atomic,
    write_storage_disposition_receipt_v1,
)
from tools.scripts.eiel_legacy_evidence_reconstruct import reconstruct_root
from tools.scripts.eiel_runtime_storage_prune import (
    CATEGORIES,
    apply_prune_plan,
    build_prune_plan,
)


_LEGACY_V4D_APPLIED_REQUEST_SHA256 = (
    "21b805e90068f81be95d1fa344f167dc36b43f567db5da99ad25d7491f08e538"
)
_LEGACY_V4D_APPLIED_CERTIFICATE_SHA256 = (
    "a95ebdd70faf2525286cbd79d9d4a5ebb3273b16ba653a84dbe2469df4a319d5"
)
_LEGACY_V4D_APPLIED_MANIFEST_SHA256 = (
    "ba8ee0766a8d250f55f46c1ab30adbe98d2c31a8c4930e4b305deaa911ee9360"
)
_LEGACY_V4D_APPLIED_PROGRAMME_ID = (
    "tensorworld-barrier-breaking-microprototype-20260824"
)
_LEGACY_V4D_APPLIED_GATE_ID = "V4D-VALID-NEGATIVE-CLOSEOUT"
_CLOSEOUT_TRANSACTION_LOCK_NAME = ".eiel-evidence-closeout.lock"
_MAX_CLOSEOUT_TRANSACTION_LOCK_BYTES = 4096
_MAX_CLOSEOUT_DURABLE_FILE_BYTES = 128 * 1024 * 1024
_MAX_DEBUG_REVIEW_INPUT_BYTES = 256 * 1024


@contextmanager
def _closeout_transaction_lock(output_root: Path):
    """Serialize all durable state transitions for one closeout output root."""

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory_fd = os.open(output_root, directory_flags)
    lock_fd: int | None = None
    try:
        lock_flags = os.O_RDWR | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            lock_flags |= os.O_NOFOLLOW
        created = False
        try:
            lock_fd = os.open(
                _CLOSEOUT_TRANSACTION_LOCK_NAME,
                lock_flags | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            created = True
        except FileExistsError:
            lock_fd = os.open(
                _CLOSEOUT_TRANSACTION_LOCK_NAME,
                lock_flags,
                dir_fd=directory_fd,
            )
        opened = os.fstat(lock_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_size > _MAX_CLOSEOUT_TRANSACTION_LOCK_BYTES
        ):
            raise EvidenceCertificateError(
                "closeout transaction lock must be a bounded single-link regular file"
            )
        if created:
            os.fsync(lock_fd)
            os.fsync(directory_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        locked = os.fstat(lock_fd)
        named = os.stat(
            _CLOSEOUT_TRANSACTION_LOCK_NAME,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(locked.st_mode)
            or locked.st_nlink != 1
            or locked.st_size > _MAX_CLOSEOUT_TRANSACTION_LOCK_BYTES
            or (locked.st_dev, locked.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise EvidenceCertificateError(
                "closeout transaction lock identity changed while acquiring it"
            )
        yield
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(directory_fd)


@contextmanager
def _run_artifact_transaction_fences_v1(roots: list[Path]):
    """Hold canonical per-run fences in stable order for one closeout."""

    ordered = sorted(
        {root.expanduser().resolve(strict=True) for root in roots}, key=str
    )
    with ExitStack() as stack:
        for root in ordered:
            stack.enter_context(run_artifact_transaction_fence_v1(root))
        yield


def _bound_source_transaction_roots_v1(
    certificate: Mapping[str, Any], *, evidence_root: Path
) -> list[Path]:
    """Derive only exact bound producer roots when released bytes are absent."""

    manifests: dict[Path, str] = {}
    for row in certificate.get("artifacts", []):
        if not isinstance(row, Mapping):
            continue
        binding = row.get("source_manifest_binding")
        if not isinstance(binding, Mapping):
            continue
        relative = Path(str(binding.get("manifest_path") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise EvidenceCertificateError(
                "source manifest binding path is not a safe relative path"
            )
        unresolved = evidence_root / relative
        if unresolved.is_symlink():
            raise EvidenceCertificateError(
                "source disposition manifest cannot be a symlink"
            )
        resolved = unresolved.resolve(strict=True)
        try:
            resolved.relative_to(evidence_root)
        except ValueError as exc:
            raise EvidenceCertificateError(
                "source disposition manifest escapes evidence_root"
            ) from exc
        expected_sha256 = str(binding.get("manifest_sha256") or "")
        previous = manifests.setdefault(resolved, expected_sha256)
        if previous != expected_sha256:
            raise EvidenceCertificateError(
                "one source manifest path has conflicting certificate identities"
            )
    roots: set[Path] = set()
    for path, expected_sha256 in sorted(
        manifests.items(), key=lambda item: str(item[0])
    ):
        if not path.is_file():
            raise EvidenceCertificateError(
                "source disposition manifest must be a regular file"
            )
        manifest = validate_artifact_disposition_manifest_v1(load_json_object(path))
        if manifest.get("manifest_sha256") != expected_sha256:
            raise EvidenceCertificateError(
                "source disposition manifest SHA-256 differs from artifact binding"
            )
        unresolved_root = Path(str(manifest.get("runtime_root") or "")).expanduser()
        if unresolved_root.is_symlink():
            raise EvidenceCertificateError(
                "source manifest runtime_root cannot be a symlink"
            )
        source_root = unresolved_root.resolve(strict=True)
        try:
            source_root.relative_to(evidence_root)
        except ValueError as exc:
            raise EvidenceCertificateError(
                "source manifest runtime_root escapes evidence_root"
            ) from exc
        roots.add(source_root)
    return sorted(roots, key=str)


def _relative_to_root(path: Path, root: Path, *, label: str) -> Path:
    unresolved = path.expanduser()
    if unresolved.is_symlink():
        raise EvidenceCertificateError(f"{label} cannot be a symlink")
    resolved = unresolved.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvidenceCertificateError(f"{label} must be inside evidence_root") from exc
    return resolved


def _paths(args: argparse.Namespace, request: Mapping[str, Any]) -> dict[str, Path]:
    unresolved_root = args.evidence_root.expanduser()
    if unresolved_root.is_symlink():
        raise EvidenceCertificateError("evidence_root cannot be a symlink")
    root = unresolved_root.resolve(strict=True)
    output_root = _relative_to_root(args.output_root, root, label="output_root")
    output_root.mkdir(parents=True, exist_ok=True)
    gate_id = str(request["gate_id"])
    return {
        "root": root,
        "output_root": output_root,
        "transaction_lock": output_root / _CLOSEOUT_TRANSACTION_LOCK_NAME,
        "pre_certificate": output_root / "evidence_certificate.pre_release.json",
        "final_certificate": output_root / "evidence_certificate.json",
        "manifest": output_root / "artifact_disposition_manifest.json",
        "dry_receipt": output_root / "storage_disposition_receipt.dry_run.json",
        "receipt": output_root / "storage_disposition_receipt.json",
        "intent": output_root / "reclamation_intent.json",
        "pre_publication": output_root / "permanent_capsule.pre_release.json",
        "final_publication": output_root / "permanent_capsule.final.json",
        "result": output_root / f"gate_{gate_id}_reclamation_result.json",
    }


def _permanent_root_for_closeout(
    args: argparse.Namespace,
    request: Mapping[str, Any],
    *,
    evidence_root: Path,
) -> Path | None:
    configured = getattr(args, "permanent_root", None)
    if configured is not None:
        configured_path = Path(configured)
        try:
            canonical_root = default_permanent_evidence_root_v1(evidence_root)
        except EvidenceCertificateError:
            canonical_root = None
        if canonical_root is not None and configured_path.expanduser().resolve(
            strict=False
        ) != canonical_root.resolve(strict=False):
            raise EvidenceCertificateError(
                "configured runtime root runtime evidence cannot override the canonical permanent evidence root"
            )
        return configured_path
    try:
        return default_permanent_evidence_root_v1(evidence_root)
    except EvidenceCertificateError:
        if request["artifact_lifecycle_contract"]["lifecycle_profile"] == (
            "DEBUG_FORENSIC"
        ):
            raise
        return None


def _publish_permanent(
    args: argparse.Namespace,
    request: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    evidence_root: Path,
    allow_unreserved_final: bool = False,
) -> dict[str, Any] | None:
    permanent_root = _permanent_root_for_closeout(
        args, request, evidence_root=evidence_root
    )
    if permanent_root is None:
        return None
    return publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=permanent_root,
        source_root=evidence_root,
        max_capsule_bytes=int(
            getattr(args, "max_permanent_capsule_bytes", DEFAULT_MAX_CAPSULE_BYTES)
        ),
        max_root_bytes=int(
            getattr(
                args,
                "max_permanent_root_bytes",
                DEFAULT_MAX_PERMANENT_ROOT_BYTES,
            )
        ),
        allow_unreserved_final=allow_unreserved_final,
    )


def _assert_output_disjoint(
    request: Mapping[str, Any], *, root: Path, output_root: Path
) -> None:
    for row in request["artifacts"]:
        unresolved = root / str(row["path"])
        if unresolved.is_symlink():
            raise EvidenceCertificateError(
                f"declared evidence artifact became a symlink: {row['path']}"
            )
        artifact = unresolved.resolve(strict=False)
        if (
            artifact == output_root
            or artifact in output_root.parents
            or output_root in artifact.parents
        ):
            raise EvidenceCertificateError(
                f"output_root overlaps declared evidence artifact: {row['path']}"
            )


def _precheck_existing_pre_publication_v1(
    args: argparse.Namespace,
    request: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    evidence_root: Path,
    receipt_path: Path,
) -> None:
    """Prove an existing PRE publication is exact before validation can publish."""

    publication = load_json_object(receipt_path)
    publication_core = dict(publication)
    publication_sha256 = publication_core.pop("publication_sha256", None)
    capsule = build_permanent_evidence_capsule_v1(certificate)
    if (
        publication_sha256 != canonical_sha256(publication_core)
        or publication.get("phase") != "SEALED_PRE_RELEASE"
        or publication.get("certificate_id") != certificate.get("certificate_id")
        or publication.get("certificate_sha256")
        != certificate.get("certificate_sha256")
        or publication.get("capsule_id") != capsule.get("capsule_id")
        or publication.get("capsule_sha256") != capsule.get("capsule_sha256")
        or publication.get("final_capacity_reservation_status") != "RESERVED"
    ):
        raise EvidenceCertificateError(
            "dry-run PRE publication differs from the rebuilt certificate"
        )
    permanent_root = _permanent_root_for_closeout(
        args, request, evidence_root=evidence_root
    )
    if permanent_root is None:
        raise EvidenceCertificateError(
            "dry-run PRE publication lacks its configured permanent root"
        )
    root = permanent_root.expanduser().resolve(strict=False)
    if publication.get("permanent_root") != str(root):
        raise EvidenceCertificateError(
            "dry-run PRE publication root differs from its configuration"
        )
    total_bytes = publication.get("total_bytes")
    if (
        type(total_bytes) is not int
        or total_bytes <= 0
        or total_bytes > DEFAULT_MAX_CAPSULE_BYTES
    ):
        raise EvidenceCertificateError(
            "dry-run PRE publication byte accounting is invalid"
        )
    observed_total = 0
    contents: dict[str, bytes] = {}
    for path_field, sha_field, bytes_field in (
        ("json_path", "json_sha256", "json_bytes"),
        ("markdown_path", "markdown_sha256", "markdown_bytes"),
        ("complete_path", "complete_sha256", "complete_bytes"),
    ):
        raw_path = publication.get(path_field)
        expected_bytes = publication.get(bytes_field)
        expected_sha256 = publication.get(sha_field)
        if (
            not isinstance(raw_path, str)
            or type(expected_bytes) is not int
            or expected_bytes <= 0
            or expected_bytes > total_bytes
            or not isinstance(expected_sha256, str)
        ):
            raise EvidenceCertificateError(
                "dry-run PRE publication file accounting is invalid"
            )
        path = Path(raw_path)
        if (
            not path.is_absolute()
            or path.parent != root
            or path.name != Path(raw_path).name
            or path.is_symlink()
            or not path.is_file()
        ):
            raise EvidenceCertificateError(
                "dry-run PRE publication file identity is invalid"
            )
        metadata = path.stat()
        if metadata.st_nlink != 1 or metadata.st_size != expected_bytes:
            raise EvidenceCertificateError(
                "dry-run PRE publication file metadata differs"
            )
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            raise EvidenceCertificateError(
                "dry-run PRE publication file digest differs"
            )
        contents[path_field] = content
        observed_total += expected_bytes
    if observed_total != total_bytes:
        raise EvidenceCertificateError(
            "dry-run PRE publication total byte accounting differs"
        )
    try:
        observed_capsule = json.loads(contents["json_path"])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceCertificateError(
            "dry-run PRE capsule JSON cannot be decoded"
        ) from exc
    if observed_capsule != capsule:
        raise EvidenceCertificateError(
            "dry-run PRE capsule differs from the rebuilt certificate"
        )
    reservation_raw = publication.get("final_capacity_reservation_path")
    reservation_sha256 = publication.get("final_capacity_reservation_file_sha256")
    if not isinstance(reservation_raw, str) or not isinstance(reservation_sha256, str):
        raise EvidenceCertificateError(
            "dry-run PRE publication reservation identity is invalid"
        )
    reservation_path = Path(reservation_raw)
    if (
        not reservation_path.is_absolute()
        or reservation_path.parent != root
        or reservation_path.is_symlink()
        or not reservation_path.is_file()
        or reservation_path.stat().st_nlink != 1
        or hashlib.sha256(reservation_path.read_bytes()).hexdigest()
        != reservation_sha256
    ):
        raise EvidenceCertificateError(
            "dry-run PRE publication reservation file differs"
        )


def _dry_run(args: argparse.Namespace) -> dict[str, Any]:
    raw_request = load_json_object(args.request)
    request = validate_evidence_certificate_request_v1(raw_request)
    paths = _paths(args, request)
    with _closeout_transaction_lock(paths["output_root"]):
        return _dry_run_under_lock(
            args,
            request=request,
            paths=paths,
        )


def _dry_run_under_lock(
    args: argparse.Namespace,
    *,
    request: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, Any]:
    _assert_output_disjoint(
        request, root=paths["root"], output_root=paths["output_root"]
    )
    known_names = {
        path.name for key, path in paths.items() if key not in {"root", "output_root"}
    }
    unexpected = sorted(
        entry.name
        for entry in paths["output_root"].iterdir()
        if entry.name not in known_names
    )
    if unexpected:
        raise EvidenceCertificateError(
            "dry-run output root contains entries outside its exact durable prefix: "
            + ", ".join(unexpected[:8])
        )
    existing = {
        key: path
        for key, path in paths.items()
        if key not in {"root", "output_root", "staging", "transaction_lock"}
        and (path.exists() or path.is_symlink())
    }
    if any(path.is_symlink() for path in existing.values()):
        raise EvidenceCertificateError("dry-run durable prefix contains a symlink")
    if any(
        not path.is_file()
        or path.stat().st_nlink != 1
        or path.stat().st_size > _MAX_CLOSEOUT_DURABLE_FILE_BYTES
        for path in existing.values()
    ):
        raise EvidenceCertificateError(
            "dry-run durable prefix must contain bounded single-link regular files"
        )
    permanent_expected = (
        _permanent_root_for_closeout(
            args,
            request,
            evidence_root=paths["root"],
        )
        is not None
    )
    durable_order = [
        *(["pre_publication"] if permanent_expected else []),
        "pre_certificate",
        "manifest",
        "dry_receipt",
        "intent",
        "result",
    ]
    existing_names = set(existing)
    prefix_length = next(
        (
            length
            for length in range(len(durable_order) + 1)
            if existing_names == set(durable_order[:length])
        ),
        None,
    )
    if prefix_length is None:
        raise EvidenceCertificateError(
            "dry-run output root is not an exact durable prefix: "
            + ", ".join(str(path) for path in existing.values())
        )
    certificate = build_evidence_certificate_v1(request, evidence_root=paths["root"])
    if "pre_publication" in existing_names:
        _precheck_existing_pre_publication_v1(
            args,
            request,
            certificate,
            evidence_root=paths["root"],
            receipt_path=paths["pre_publication"],
        )
        permanent_publication = _validated_or_published_capsule(
            args,
            request,
            certificate,
            evidence_root=paths["root"],
            receipt_path=paths["pre_publication"],
            allow_publish=False,
            required=True,
            require_outstanding_final_capacity=True,
        )
    elif permanent_expected:
        permanent_publication = _publish_permanent(
            args,
            request,
            certificate,
            evidence_root=paths["root"],
        )
        if permanent_publication is not None:
            write_json_atomic(paths["pre_publication"], permanent_publication)
    else:
        permanent_publication = None
    if "pre_certificate" in existing_names:
        observed_certificate = validate_evidence_certificate_v1(
            load_json_object(paths["pre_certificate"])
        )
        if observed_certificate != certificate:
            raise EvidenceCertificateError(
                "dry-run pre-release certificate differs from deterministic recovery"
            )
    else:
        write_json_atomic(paths["pre_certificate"], certificate)
    manifest = certificate_disposition_manifest_v1(
        request,
        certificate,
        evidence_root=paths["root"],
        certificate_path=paths["pre_certificate"],
        permanent_publication=permanent_publication,
    )
    if "manifest" in existing_names:
        observed_manifest = validate_artifact_disposition_manifest_v1(
            load_json_object(paths["manifest"])
        )
        if observed_manifest != manifest:
            raise EvidenceCertificateError(
                "dry-run disposition manifest differs from deterministic recovery"
            )
    else:
        write_json_atomic(paths["manifest"], manifest)
    plan = build_prune_plan(
        paths["root"], set(CATEGORIES), disposition_manifest=manifest
    )
    observed_dry_receipt = (
        load_json_object(paths["dry_receipt"])
        if "dry_receipt" in existing_names
        else None
    )
    dry_receipt = build_storage_disposition_receipt_v1(
        manifest=manifest,
        bytes_before=int(manifest["summary"]["produced_bytes"]),
        bytes_retained=int(manifest["summary"]["produced_bytes"]),
        bytes_released=0,
        bytes_unresolved_rx=int(manifest["summary"]["unresolved_rx_bytes"]),
        authorization_reason="evidence certificate gate-closure dry run",
        replacement_references=[
            {
                "certificate_id": certificate["certificate_id"],
                "certificate_sha256": certificate["certificate_sha256"],
            }
        ],
        retention_references=[str(paths["pre_certificate"])],
        plan_sha256=manifest.get("plan_sha256"),
        timestamp_utc=(
            str(observed_dry_receipt.get("timestamp_utc"))
            if observed_dry_receipt is not None
            else utc_now()
        ),
        result="dry_run",
    )
    if observed_dry_receipt is not None:
        if observed_dry_receipt != dry_receipt:
            raise EvidenceCertificateError(
                "dry-run storage receipt differs from deterministic recovery"
            )
    else:
        write_storage_disposition_receipt_v1(paths["dry_receipt"], dry_receipt)
    observed_intent = (
        validate_reclamation_intent_v1(load_json_object(paths["intent"]))
        if "intent" in existing_names
        else None
    )
    intent = build_reclamation_intent_v1(
        request=request,
        certificate=certificate,
        manifest=manifest,
        selected=plan["selected"],
        status="PREPARED",
        created_at_utc=(
            str(observed_intent.get("created_at_utc"))
            if observed_intent is not None
            else utc_now()
        ),
    )
    if observed_intent is not None:
        if observed_intent != intent:
            raise EvidenceCertificateError(
                "dry-run reclamation intent differs from deterministic recovery"
            )
    else:
        write_json_atomic(paths["intent"], intent)
    result = {
        "ok": True,
        "mode": "dry_run",
        "gate_id": request["gate_id"],
        "programme_complete": len(request["programme_plan"]["remaining_gates"]) == 0,
        "certificate_path": str(paths["pre_certificate"]),
        "certificate_id": certificate["certificate_id"],
        "certificate_sha256": certificate["certificate_sha256"],
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "intent_path": str(paths["intent"]),
        "intent_sha256": intent["intent_sha256"],
        "selected_count": int(plan["selected_count"]),
        "selected_bytes": int(plan["selected_bytes"]),
        "permanent_capsule_publication_path": (
            str(paths["pre_publication"]) if permanent_publication is not None else None
        ),
        "permanent_capsule_path": (
            permanent_publication["json_path"]
            if permanent_publication is not None
            else None
        ),
        "archive_invoked": False,
        "apply_required": True,
    }
    if "result" in existing_names:
        if load_json_object(paths["result"]) != result:
            raise EvidenceCertificateError(
                "dry-run result differs from deterministic recovery"
            )
    else:
        write_json_atomic(paths["result"], result)
    return result


def _selected_signature(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(row.get("path")),
            "bytes": int(row.get("bytes", 0)),
            "sha256": row.get("sha256"),
            "tree_sha256": row.get("tree_sha256"),
        }
        for row in rows
    ]


def _exact_legacy_v4d_applied_replay_v1(
    *,
    request: Mapping[str, Any],
    pre_certificate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> bool:
    """Recognize the sole pre-capsule `configured runtime root` closeout already applied."""

    return (
        intent.get("status") == "APPLIED"
        and request.get("programme_id") == _LEGACY_V4D_APPLIED_PROGRAMME_ID
        and request.get("gate_id") == _LEGACY_V4D_APPLIED_GATE_ID
        and canonical_sha256(request) == _LEGACY_V4D_APPLIED_REQUEST_SHA256
        and pre_certificate.get("certificate_sha256")
        == _LEGACY_V4D_APPLIED_CERTIFICATE_SHA256
        and manifest.get("manifest_sha256") == _LEGACY_V4D_APPLIED_MANIFEST_SHA256
        and intent.get("request_sha256") == _LEGACY_V4D_APPLIED_REQUEST_SHA256
        and intent.get("certificate_sha256") == _LEGACY_V4D_APPLIED_CERTIFICATE_SHA256
        and intent.get("manifest_sha256") == _LEGACY_V4D_APPLIED_MANIFEST_SHA256
    )


def _release_requires_permanent_capsule_v1(
    request: Mapping[str, Any],
    *,
    evidence_root: Path,
    manifest: Mapping[str, Any],
) -> bool:
    authority = manifest.get("release_authority", {})
    return (
        evidence_root_requires_permanent_capsule_v1(evidence_root)
        or request["artifact_lifecycle_contract"]["lifecycle_profile"]
        == "DEBUG_FORENSIC"
        or (
            isinstance(authority, Mapping)
            and authority.get("permanent_capsule_publication_sha256") is not None
        )
    )


def _validate_manifest_capsule_binding_v1(
    manifest: Mapping[str, Any],
    publication: Mapping[str, Any] | None,
    *,
    required: bool,
) -> None:
    authority = manifest.get("release_authority", {})
    if not required and publication is None:
        return
    if publication is None:
        raise EvidenceCertificateError(
            "certificate release lacks its required permanent capsule publication"
        )
    if (
        publication.get("phase") != "SEALED_PRE_RELEASE"
        or not isinstance(authority, Mapping)
        or authority.get("permanent_capsule_publication_sha256")
        != publication.get("publication_sha256")
        or authority.get("permanent_capsule_sha256")
        != publication.get("capsule_sha256")
    ):
        raise EvidenceCertificateError(
            "certificate manifest is not bound to the verified pre-release permanent capsule"
        )


def _validate_data2_source_authority_v1(
    certificate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    evidence_root: Path,
) -> None:
    if not evidence_root_requires_permanent_capsule_v1(evidence_root):
        return
    summary = certificate.get("source_manifest_binding_summary")
    authority = manifest.get("release_authority", {})
    if (
        not isinstance(summary, Mapping)
        or summary.get("binding_count") != len(certificate.get("artifacts", []))
        or not isinstance(authority, Mapping)
        or authority.get("source_manifest_binding_count")
        != summary.get("binding_count")
        or authority.get("source_manifest_bindings_sha256")
        != summary.get("bindings_sha256")
    ):
        raise EvidenceCertificateError(
            "configured runtime root certificate release lacks complete source-manifest authority"
        )


def _validated_or_published_capsule(
    args: argparse.Namespace,
    request: Mapping[str, Any],
    certificate: Mapping[str, Any],
    *,
    evidence_root: Path,
    receipt_path: Path,
    allow_publish: bool,
    required: bool,
    require_outstanding_final_capacity: bool = False,
    allow_unreserved_final: bool = False,
) -> dict[str, Any] | None:
    configured_root = _permanent_root_for_closeout(
        args, request, evidence_root=evidence_root
    )
    if receipt_path.is_file() and not receipt_path.is_symlink():
        publication = validate_permanent_evidence_publication_v1(
            load_json_object(receipt_path),
            certificate=certificate,
            source_root=evidence_root,
            require_outstanding_final_capacity=require_outstanding_final_capacity,
            allow_unreserved_final=allow_unreserved_final,
        )
        if configured_root is None or Path(publication["permanent_root"]) != (
            configured_root.expanduser().resolve(strict=False)
        ):
            raise EvidenceCertificateError(
                "permanent capsule root differs from the close-gate configuration"
            )
        return publication
    if receipt_path.exists() or receipt_path.is_symlink():
        raise EvidenceCertificateError(
            "permanent capsule publication receipt is not a regular file"
        )
    if allow_publish and configured_root is not None:
        publication = _publish_permanent(
            args,
            request,
            certificate,
            evidence_root=evidence_root,
            allow_unreserved_final=allow_unreserved_final,
        )
        if publication is None:
            raise EvidenceCertificateError("permanent capsule publication was required")
        write_json_atomic(receipt_path, publication)
        return publication
    if required:
        raise EvidenceCertificateError(
            "close-gate release requires its verified permanent capsule publication"
        )
    return None


def _recover_existing_global_final_publication_v1(
    args: argparse.Namespace,
    request: Mapping[str, Any],
    pre_certificate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    intent: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
    required: bool,
) -> bool:
    """Recover a missing local receipt for an exact completed global FINAL."""

    if intent.get("status") != "APPLYING":
        return False
    receipt_path = paths["receipt"]
    final_certificate_path = paths["final_certificate"]
    if not all(
        path.is_file() and not path.is_symlink()
        for path in (receipt_path, final_certificate_path)
    ):
        return False
    receipt = validate_storage_receipt_certificate_binding_v1(
        load_json_object(receipt_path),
        manifest=manifest,
        pre_certificate=pre_certificate,
    )
    if receipt.get("result") != "applied":
        return False
    if sorted(
        _selected_signature(list(receipt["removed"])), key=lambda row: row["path"]
    ) != sorted(
        [dict(row) for row in intent.get("selected", [])],
        key=lambda row: str(row.get("path")),
    ):
        raise EvidenceCertificateError(
            "completed storage receipt differs from the durable APPLYING selection"
        )
    expected_final = finalize_evidence_certificate_v1(
        pre_certificate,
        storage_receipt=receipt,
        manifest=manifest,
    )
    observed_final = validate_certificate_request_binding_v1(
        request,
        load_json_object(final_certificate_path),
    )
    if observed_final != expected_final:
        raise EvidenceCertificateError(
            "durable FINAL certificate differs from deterministic recovery"
        )
    final_publication_path = paths["final_publication"]
    if final_publication_path.exists() or final_publication_path.is_symlink():
        publication = _validated_or_published_capsule(
            args,
            request,
            expected_final,
            evidence_root=paths["root"],
            receipt_path=final_publication_path,
            allow_publish=False,
            required=required,
        )
        return publication is not None
    if not required:
        return False
    permanent_root = _permanent_root_for_closeout(
        args, request, evidence_root=paths["root"]
    )
    if permanent_root is None:
        return False
    publication = recover_existing_permanent_evidence_publication_v1(
        expected_final,
        permanent_root=permanent_root,
        source_root=paths["root"],
        max_capsule_bytes=int(
            getattr(args, "max_permanent_capsule_bytes", DEFAULT_MAX_CAPSULE_BYTES)
        ),
        max_root_bytes=int(
            getattr(
                args,
                "max_permanent_root_bytes",
                DEFAULT_MAX_PERMANENT_ROOT_BYTES,
            )
        ),
    )
    if publication is None:
        return False
    write_json_atomic(final_publication_path, publication)
    return True


def _applied_replay_result(
    *,
    request: Mapping[str, Any],
    pre_certificate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    intent: Mapping[str, Any],
    paths: Mapping[str, Path],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Reconstruct an idempotent result only from hash-checked durable records."""

    for name in ("receipt", "final_certificate"):
        if not paths[name].is_file() or paths[name].is_symlink():
            raise EvidenceCertificateError(
                "APPLIED intent lacks its durable receipt/final certificate"
            )
    legacy_v4d = _exact_legacy_v4d_applied_replay_v1(
        request=request,
        pre_certificate=pre_certificate,
        manifest=manifest,
        intent=intent,
    )
    receipt = validate_storage_receipt_certificate_binding_v1(
        load_json_object(paths["receipt"]),
        manifest=manifest,
        pre_certificate=pre_certificate,
    )
    receipt_sha256 = receipt["receipt_sha256"]
    final_certificate = validate_certificate_request_binding_v1(
        request, load_json_object(paths["final_certificate"])
    )
    expected_final = finalize_evidence_certificate_v1(
        pre_certificate,
        storage_receipt=receipt,
        manifest=manifest,
    )
    if not legacy_v4d and final_certificate != expected_final:
        raise EvidenceCertificateError(
            "durable FINAL certificate is not the deterministic receipt-bound revision"
        )
    released_paths = sorted(
        str(row.get("path"))
        for row in receipt.get("removed", [])
        if isinstance(row, Mapping)
    )
    if (
        final_certificate.get("phase") != "FINAL"
        or final_certificate.get("prior_certificate_sha256")
        != pre_certificate.get("certificate_sha256")
        or final_certificate.get("storage_disposition_receipt_sha256") != receipt_sha256
        or intent.get("result", {}).get("storage_disposition_receipt_sha256")
        != receipt_sha256
        or sorted(intent.get("result", {}).get("released_paths", [])) != released_paths
        or intent.get("manifest_sha256") != manifest.get("manifest_sha256")
    ):
        raise EvidenceCertificateError(
            "APPLIED intent/certificate/receipt binding mismatch"
        )
    requires_permanent = _release_requires_permanent_capsule_v1(
        request,
        evidence_root=paths["root"],
        manifest=manifest,
    )
    if not legacy_v4d:
        _validate_data2_source_authority_v1(
            pre_certificate,
            manifest,
            evidence_root=paths["root"],
        )
    pre_publication = _validated_or_published_capsule(
        args,
        request,
        pre_certificate,
        evidence_root=paths["root"],
        receipt_path=paths["pre_publication"],
        allow_publish=False,
        required=requires_permanent and not legacy_v4d,
    )
    if not legacy_v4d:
        _validate_manifest_capsule_binding_v1(
            manifest,
            pre_publication,
            required=requires_permanent,
        )
    legacy_unreserved_final = legacy_v4d or (
        not evidence_root_requires_permanent_capsule_v1(paths["root"])
        and pre_publication is None
        and intent.get("status") == "APPLIED"
    )
    final_publication = _validated_or_published_capsule(
        args,
        request,
        final_certificate,
        evidence_root=paths["root"],
        receipt_path=paths["final_publication"],
        allow_publish=True,
        required=requires_permanent,
        allow_unreserved_final=legacy_unreserved_final,
    )
    return {
        "ok": True,
        "mode": "apply",
        "gate_id": request["gate_id"],
        "programme_complete": len(request["programme_plan"]["remaining_gates"]) == 0,
        "certificate_path": str(paths["final_certificate"]),
        "pre_release_certificate_path": str(paths["pre_certificate"]),
        "manifest_path": str(paths["manifest"]),
        "intent_path": str(paths["intent"]),
        "storage_disposition_receipt_path": str(paths["receipt"]),
        "permanent_pre_release_publication_path": (
            str(paths["pre_publication"]) if pre_publication is not None else None
        ),
        "permanent_final_publication_path": (
            str(paths["final_publication"]) if final_publication is not None else None
        ),
        "permanent_final_capsule_path": (
            final_publication["json_path"] if final_publication is not None else None
        ),
        "released_count": len(released_paths),
        "released_bytes": int(receipt.get("bytes_released", 0)),
        "failures": [],
        "archive_invoked": False,
        "idempotent_replay": True,
    }


def _recover_completed_apply_v1(
    *,
    request: Mapping[str, Any],
    pre_certificate: Mapping[str, Any],
    manifest: Mapping[str, Any],
    intent: Mapping[str, Any],
    paths: Mapping[str, Path],
    args: argparse.Namespace,
    authenticated_released_paths: set[str],
    release_journal: str | None,
    release_journal_status: str | None,
) -> dict[str, Any] | None:
    """Finish the deterministic durable tuple after a post-release crash."""

    receipt_path = paths["receipt"]
    if not receipt_path.exists() and not receipt_path.is_symlink():
        return None
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise EvidenceCertificateError(
            "storage disposition receipt is not a regular file"
        )
    raw_receipt = load_json_object(receipt_path)
    if raw_receipt.get("result") != "applied":
        return None
    if intent.get("status") != "APPLYING":
        raise EvidenceCertificateError(
            "completed storage receipt requires its durable APPLYING intent"
        )
    receipt = validate_storage_receipt_certificate_binding_v1(
        raw_receipt,
        manifest=manifest,
        pre_certificate=pre_certificate,
    )
    released_paths = sorted(str(row["path"]) for row in receipt["removed"])
    receipt_selection = sorted(
        _selected_signature(list(receipt["removed"])), key=lambda row: row["path"]
    )
    intent_selection = sorted(
        [dict(row) for row in intent.get("selected", [])],
        key=lambda row: str(row.get("path")),
    )
    if receipt_selection != intent_selection:
        raise EvidenceCertificateError(
            "completed storage receipt differs from the durable APPLYING selection"
        )
    if set(released_paths) != authenticated_released_paths:
        raise EvidenceCertificateError(
            "completed storage receipt differs from the actuator-owned release journal"
        )
    requires_permanent = _release_requires_permanent_capsule_v1(
        request,
        evidence_root=paths["root"],
        manifest=manifest,
    )
    _validate_data2_source_authority_v1(
        pre_certificate,
        manifest,
        evidence_root=paths["root"],
    )
    pre_publication = _validated_or_published_capsule(
        args,
        request,
        pre_certificate,
        evidence_root=paths["root"],
        receipt_path=paths["pre_publication"],
        allow_publish=False,
        required=requires_permanent,
    )
    _validate_manifest_capsule_binding_v1(
        manifest,
        pre_publication,
        required=requires_permanent,
    )
    expected_final = finalize_evidence_certificate_v1(
        pre_certificate,
        storage_receipt=receipt,
        manifest=manifest,
    )
    final_path = paths["final_certificate"]
    if final_path.exists() or final_path.is_symlink():
        if final_path.is_symlink() or not final_path.is_file():
            raise EvidenceCertificateError(
                "durable FINAL certificate is not a regular file"
            )
        observed_final = validate_certificate_request_binding_v1(
            request,
            load_json_object(final_path),
        )
        if observed_final != expected_final:
            raise EvidenceCertificateError(
                "durable FINAL certificate differs from deterministic recovery"
            )
    else:
        write_json_atomic(final_path, expected_final)
    _validated_or_published_capsule(
        args,
        request,
        expected_final,
        evidence_root=paths["root"],
        receipt_path=paths["final_publication"],
        allow_publish=True,
        required=requires_permanent,
    )
    final_intent = build_reclamation_intent_v1(
        request=request,
        certificate=pre_certificate,
        manifest=manifest,
        selected=list(receipt["removed"]),
        status="APPLIED",
        created_at_utc=str(intent["created_at_utc"]),
        previous_intent_sha256=str(intent["intent_sha256"]),
        result={
            "storage_disposition_receipt_sha256": receipt["receipt_sha256"],
            "released_paths": released_paths,
            "failures": [],
            "release_journal": release_journal,
            "release_journal_status": release_journal_status,
        },
    )
    write_json_atomic(paths["intent"], final_intent)
    return _applied_replay_result(
        request=request,
        pre_certificate=pre_certificate,
        manifest=manifest,
        intent=final_intent,
        paths=paths,
        args=args,
    )


def _apply(args: argparse.Namespace) -> dict[str, Any]:
    raw_request = load_json_object(args.request)
    request = validate_evidence_certificate_request_v1(raw_request)
    paths = _paths(args, request)
    with _closeout_transaction_lock(paths["output_root"]):
        return _apply_under_lock(
            args,
            request=request,
            paths=paths,
        )


def _apply_under_lock(
    args: argparse.Namespace,
    *,
    request: Mapping[str, Any],
    paths: Mapping[str, Path],
    _transaction_fences_held: bool = False,
) -> dict[str, Any]:
    for required in ("pre_certificate", "manifest", "intent"):
        if not paths[required].is_file() or paths[required].is_symlink():
            raise EvidenceCertificateError(
                "--apply requires the exact prior dry-run artifacts"
            )
    certificate = validate_evidence_certificate_v1(
        load_json_object(paths["pre_certificate"])
    )
    manifest = validate_artifact_disposition_manifest_v1(
        load_json_object(paths["manifest"])
    )
    intent = validate_reclamation_intent_v1(load_json_object(paths["intent"]))
    current_request_sha256 = canonical_sha256(request)
    if (
        certificate["request_sha256"] != current_request_sha256
        or intent["request_sha256"] != current_request_sha256
        or intent["certificate_sha256"] != certificate["certificate_sha256"]
        or intent["manifest_sha256"] != manifest["manifest_sha256"]
        or manifest.get("release_authority", {}).get("programme_plan_sha256")
        != request["programme_plan"]["plan_sha256"]
    ):
        raise EvidenceCertificateError(
            "request/certificate/plan/manifest changed after dry-run; regenerate at the gate boundary"
        )
    certificate = validate_certificate_request_binding_v1(request, certificate)
    if intent["status"] == "APPLIED":
        return _applied_replay_result(
            request=request,
            pre_certificate=certificate,
            manifest=manifest,
            intent=intent,
            paths=paths,
            args=args,
        )
    requires_permanent = _release_requires_permanent_capsule_v1(
        request,
        evidence_root=paths["root"],
        manifest=manifest,
    )
    _validate_data2_source_authority_v1(
        certificate,
        manifest,
        evidence_root=paths["root"],
    )
    source_binding_required = (
        evidence_root_requires_permanent_capsule_v1(paths["root"])
        or certificate.get("source_manifest_binding_summary") is not None
    )
    source_runtime_roots: list[Path] = []
    if source_binding_required:
        try:
            source_binding_validation = (
                validate_certificate_source_manifest_bindings_v1(
                    request,
                    certificate,
                    evidence_root=paths["root"],
                    require_fresh_diagnostic_review_audit=False,
                    recheck_diagnostic_local_live_state=False,
                )
            )
        except FileNotFoundError:
            if intent.get("status") not in {"APPLYING", "PARTIAL_FAILURE"}:
                raise
            source_runtime_roots = _bound_source_transaction_roots_v1(
                certificate,
                evidence_root=paths["root"],
            )
        else:
            source_runtime_roots = [
                Path(value)
                for value in source_binding_validation["source_runtime_roots"]
            ]
    if not _transaction_fences_held:
        transaction_roots = [paths["root"]]
        transaction_roots.extend(source_runtime_roots)
        with _run_artifact_transaction_fences_v1(transaction_roots):
            return _apply_under_lock(
                args,
                request=request,
                paths=paths,
                _transaction_fences_held=True,
            )
    finalization_recovery_candidate = _recover_existing_global_final_publication_v1(
        args,
        request,
        certificate,
        manifest,
        intent,
        paths=paths,
        required=requires_permanent,
    )
    permanent_publication = _validated_or_published_capsule(
        args,
        request,
        certificate,
        evidence_root=paths["root"],
        receipt_path=paths["pre_publication"],
        allow_publish=True,
        required=requires_permanent,
        require_outstanding_final_capacity=(
            requires_permanent and not finalization_recovery_candidate
        ),
    )
    _validate_manifest_capsule_binding_v1(
        manifest,
        permanent_publication,
        required=requires_permanent,
    )
    _assert_output_disjoint(
        request, root=paths["root"], output_root=paths["output_root"]
    )
    plan = build_prune_plan(
        paths["root"], set(CATEGORIES), disposition_manifest=manifest
    )
    if _selected_signature(plan["selected"]) != intent["selected"]:
        raise EvidenceCertificateError("prune selection changed after dry-run")
    current_intent = build_reclamation_intent_v1(
        request=request,
        certificate=certificate,
        manifest=manifest,
        selected=plan["selected"],
        status="APPLYING",
        created_at_utc=str(intent["created_at_utc"]),
        previous_intent_sha256=intent["intent_sha256"],
        result={},
    )
    write_json_atomic(paths["intent"], current_intent)

    delete_result = apply_prune_plan(
        paths["root"],
        plan,
        disposition_manifest=manifest,
        certificate_release_context={
            "request": request,
            "certificate": certificate,
            "intent": current_intent,
            "permanent_publication": permanent_publication,
        },
        _transaction_fence_held=True,
    )
    authenticated_released = set(delete_result.get("deleted", [])) | set(
        delete_result.get("recovered_released", [])
    )
    unauthenticated_absent = sorted(set(delete_result.get("already_absent", [])))
    if authenticated_released & set(unauthenticated_absent):
        raise EvidenceCertificateError(
            "actuator release journal returned contradictory artifact states"
        )
    rows_by_path = {str(row["path"]): dict(row) for row in plan["selected"]}
    if not authenticated_released.issubset(rows_by_path):
        raise EvidenceCertificateError(
            "actuator release journal exceeds the exact prune selection"
        )
    release_failures = list(delete_result.get("failures", []))
    release_failures.extend(
        {
            "path": path,
            "type": "UnauthenticatedAbsence",
            "message": "artifact absent without actuator-owned journal authority",
        }
        for path in unauthenticated_absent
    )
    complete_release = not release_failures and authenticated_released == set(
        rows_by_path
    )
    if source_binding_required and complete_release:
        validate_certificate_source_manifest_bindings_v1(
            request,
            certificate,
            evidence_root=paths["root"],
            authenticated_absent_paths=frozenset(authenticated_released),
            require_fresh_diagnostic_review_audit=False,
            recheck_diagnostic_local_live_state=False,
        )
    recovered = _recover_completed_apply_v1(
        request=request,
        pre_certificate=certificate,
        manifest=manifest,
        intent=current_intent,
        paths=paths,
        args=args,
        authenticated_released_paths=authenticated_released,
        release_journal=delete_result.get("release_journal"),
        release_journal_status=delete_result.get("release_journal_status"),
    )
    if recovered is not None:
        return recovered
    effective_paths = authenticated_released
    effective_removed = [rows_by_path[path] for path in sorted(effective_paths)]
    released_bytes = sum(int(row.get("bytes", 0)) for row in effective_removed)
    failures = release_failures
    receipt = build_storage_disposition_receipt_v1(
        manifest=manifest,
        bytes_before=int(manifest["summary"]["produced_bytes"]),
        bytes_retained=int(manifest["summary"]["produced_bytes"]) - released_bytes,
        bytes_released=released_bytes,
        bytes_unresolved_rx=int(manifest["summary"]["unresolved_rx_bytes"]),
        removed=effective_removed,
        action_failures=failures,
        skipped_paths=unauthenticated_absent,
        authorization_reason="complete evidence certificate and closed last-use predicates",
        replacement_references=[
            {
                "certificate_id": certificate["certificate_id"],
                "certificate_sha256": certificate["certificate_sha256"],
            }
        ],
        retention_references=[str(paths["pre_certificate"])],
        plan_sha256=manifest.get("plan_sha256"),
        timestamp_utc=utc_now(),
        result="partial_failure" if failures else "applied",
    )
    write_storage_disposition_receipt_v1(paths["receipt"], receipt)
    status = "PARTIAL_FAILURE" if failures else "APPLIED"
    final_certificate_path: str | None = None
    final_publication: dict[str, Any] | None = None
    if not failures:
        final_certificate = finalize_evidence_certificate_v1(
            certificate,
            storage_receipt=receipt,
            manifest=manifest,
        )
        write_json_atomic(paths["final_certificate"], final_certificate)
        final_certificate_path = str(paths["final_certificate"])
        final_publication = _validated_or_published_capsule(
            args,
            request,
            final_certificate,
            evidence_root=paths["root"],
            receipt_path=paths["final_publication"],
            allow_publish=True,
            required=requires_permanent,
        )
    final_intent = build_reclamation_intent_v1(
        request=request,
        certificate=certificate,
        manifest=manifest,
        selected=plan["selected"],
        status=status,
        created_at_utc=str(intent["created_at_utc"]),
        previous_intent_sha256=current_intent["intent_sha256"],
        result={
            "storage_disposition_receipt_sha256": receipt["receipt_sha256"],
            "released_paths": sorted(effective_paths),
            "failures": failures,
            "release_journal": delete_result.get("release_journal"),
            "release_journal_status": delete_result.get("release_journal_status"),
        },
    )
    write_json_atomic(paths["intent"], final_intent)
    result = {
        "ok": not failures,
        "mode": "apply" if not failures else "partial_failure",
        "gate_id": request["gate_id"],
        "programme_complete": len(request["programme_plan"]["remaining_gates"]) == 0,
        "certificate_path": final_certificate_path,
        "pre_release_certificate_path": str(paths["pre_certificate"]),
        "manifest_path": str(paths["manifest"]),
        "intent_path": str(paths["intent"]),
        "storage_disposition_receipt_path": str(paths["receipt"]),
        "permanent_pre_release_publication_path": (
            str(paths["pre_publication"]) if permanent_publication is not None else None
        ),
        "permanent_final_publication_path": (
            str(paths["final_publication"]) if final_publication is not None else None
        ),
        "permanent_final_capsule_path": (
            final_publication["json_path"] if final_publication is not None else None
        ),
        "released_count": len(effective_removed),
        "released_bytes": released_bytes,
        "release_journal": delete_result.get("release_journal"),
        "release_journal_status": delete_result.get("release_journal_status"),
        "failures": failures,
        "archive_invoked": False,
    }
    write_json_atomic(paths["result"], result)
    return result


def historical_audit(root: Path, *, max_objects: int, max_bytes: int) -> dict[str, Any]:
    """Classify one exact historical root without writing, deleting, or archiving."""

    runtime_root = root.expanduser().resolve(strict=True)
    manifest_path = runtime_root / "artifact_disposition_manifest.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        manifest = validate_artifact_disposition_manifest_v1(
            load_json_object(manifest_path)
        )
        source = "existing_manifest"
    else:
        manifest = reconstruct_root(
            runtime_root, max_objects=max_objects, max_bytes=max_bytes
        )
        source = "bounded_legacy_reconstruction_preview"
    certificate_backed = (
        manifest.get("release_authority", {}).get("kind")
        == "EVIDENCE_CERTIFICATE_GATE_CLOSURE"
    )
    classes = {
        "certificate_sufficient_candidate_releasable": 0,
        "needs_reduction_certificate": 0,
        "dependency_held": 0,
        "forensic_unresolved": 0,
        "explicit_asset": 0,
        "durable_authority": 0,
        "unknown_policy": 0,
    }
    rows: list[dict[str, Any]] = []
    for row in manifest.get("artifacts", []):
        if not isinstance(row, Mapping):
            continue
        evidence_class = row.get("evidence_class")
        retention_class = row.get("retention_class")
        if retention_class == "RX":
            classification = "unknown_policy"
        elif evidence_class == "SCIENTIFIC_ENGINEERING_ASSET":
            classification = "explicit_asset"
        elif (
            evidence_class == "FORENSIC_HOLD"
            or row.get("certificate_disposition") == "FORENSIC_HOLD"
            or row.get("disposition") == "HOLD_DIAGNOSTIC"
        ):
            classification = "forensic_unresolved"
        elif (
            row.get("certificate_disposition") == "HOLD_UNTIL_LAST_USE"
            or row.get("disposition") == "KEEP_ACTIVE"
            or row.get("live_reference_status") == "active"
        ):
            classification = "dependency_held"
        elif retention_class in {"R4", "R5"}:
            classification = "durable_authority"
        elif certificate_backed and row.get("release_eligible") is True:
            classification = "certificate_sufficient_candidate_releasable"
        else:
            classification = "needs_reduction_certificate"
        size = int(row.get("bytes", 0))
        classes[classification] += size
        rows.append(
            {
                "path": row.get("path"),
                "bytes": size,
                "classification": classification,
                "release_authorized": False,
            }
        )
    core = {
        "schema_version": HISTORICAL_AUDIT_SCHEMA_VERSION,
        "root": str(runtime_root),
        "mode": "DRY_RUN_READ_ONLY",
        "source": source,
        "manifest_sha256": manifest.get("manifest_sha256"),
        "max_objects": int(max_objects),
        "max_bytes": int(max_bytes),
        "classes_bytes": classes,
        "rows": rows,
        "historical_deletion_authorized": False,
        "archive_invoked": False,
        "limitation": (
            "candidate classifications are not release authority; missing original manifests remain REQUIRES_REVIEW"
        ),
    }
    return {**core, "audit_sha256": canonical_sha256(core)}


def _write_exact_json_v1(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or load_json_object(path) != dict(value)
        ):
            raise EvidenceCertificateError(
                f"DEBUG closeout output differs from existing file: {path.name}"
            )
        return
    write_json_atomic(path, dict(value))
    if load_json_object(path) != dict(value):
        raise EvidenceCertificateError(
            f"DEBUG closeout output differs after publication: {path.name}"
        )


def _write_exact_text_v1(path: Path, value: str) -> None:
    payload = value.encode("utf-8")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise EvidenceCertificateError(
                f"DEBUG closeout output differs from existing file: {path.name}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise EvidenceCertificateError(
                f"DEBUG closeout output appeared during publication: {path.name}"
            )
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _bounded_json_input_v1(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise EvidenceCertificateError(f"{label} cannot be a symlink")
    resolved = supplied.resolve(strict=True)
    observed = os.lstat(resolved)
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_size > _MAX_DEBUG_REVIEW_INPUT_BYTES
    ):
        raise EvidenceCertificateError(
            f"{label} must be a bounded single-link regular file"
        )
    payload = resolved.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise EvidenceCertificateError(f"{label} must contain one JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _debug_output_root_v1(*, evidence_root: Path, supplied: Path) -> Path:
    unresolved = supplied.expanduser()
    if unresolved.is_symlink():
        raise EvidenceCertificateError("output_root cannot be a symlink")
    output_root = unresolved.resolve(strict=False)
    canonical_data2 = DATA2_FILESYSTEM_ROOT.expanduser().resolve(strict=False)
    if output_root == canonical_data2 or canonical_data2 in output_root.parents:
        raise EvidenceCertificateError("DEBUG output_root must remain outside configured runtime root")
    if (
        output_root == evidence_root
        or output_root in evidence_root.parents
        or evidence_root in output_root.parents
    ):
        raise EvidenceCertificateError(
            "DEBUG output_root must be disjoint from evidence_root"
        )
    return output_root


def _draft_debug_closeout_review(args: argparse.Namespace) -> dict[str, Any]:
    """Dry-run or publish one fully hash-bound review spec; never release bytes."""

    unresolved_evidence = args.evidence_root.expanduser()
    if unresolved_evidence.is_symlink():
        raise EvidenceCertificateError("evidence_root cannot be a symlink")
    evidence_root = unresolved_evidence.resolve(strict=True)
    if not evidence_root.is_dir():
        raise EvidenceCertificateError("evidence_root must be a directory")
    manifest_path = args.source_manifest.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = evidence_root / manifest_path
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise EvidenceCertificateError("source manifest must be a regular file")
    manifest_path = manifest_path.resolve(strict=True)
    review_input, review_input_file_sha256 = _bounded_json_input_v1(
        args.review_input, label="review input"
    )
    if review_input.get("schema_version") != DEBUG_CLOSEOUT_REVIEW_INPUT_SCHEMA_VERSION:
        raise EvidenceCertificateError("DEBUG closeout review input schema mismatch")
    lifecycle_contract, lifecycle_file_sha256 = _bounded_json_input_v1(
        args.lifecycle_contract, label="lifecycle contract"
    )
    supporting: list[dict[str, str]] = []
    for supplied in args.supporting_artifact:
        if "=" not in supplied:
            raise EvidenceCertificateError(
                "--supporting-artifact must use ROLE=MANIFEST_RELATIVE_PATH"
            )
        role, relative = supplied.split("=", 1)
        if not role.strip() or not relative.strip():
            raise EvidenceCertificateError(
                "--supporting-artifact must use ROLE=MANIFEST_RELATIVE_PATH"
            )
        supporting.append({"role": role.strip(), "path": relative.strip()})
    spec = draft_debug_closeout_review_spec_v1(
        evidence_root=evidence_root,
        source_manifest_path=manifest_path,
        artifact_lifecycle_contract=lifecycle_contract,
        review_input=review_input,
        supporting_artifacts=supporting,
    )
    output_root = _debug_output_root_v1(
        evidence_root=evidence_root, supplied=args.output_root
    )
    paths = {
        "spec": output_root / "diagnostic_closeout_review_spec.json",
        "plan": output_root / "diagnostic_closeout_review_draft_plan.json",
        "result": output_root / "diagnostic_closeout_review_draft_result.json",
    }
    selected = [
        {
            "artifact_id": row["artifact_id"],
            "evidence_class": row["evidence_class"],
        }
        for row in spec["superseded_artifacts"]
    ]
    manifest = validate_artifact_disposition_manifest_v1(
        load_json_object(manifest_path)
    )
    spec_bytes = pretty_json(spec).encode("utf-8")
    plan_core = {
        "schema_version": "eiel_debug_forensic_closeout_review_draft_plan_v1",
        "mode": "DRY_RUN_REVIEW_DRAFT",
        "evidence_root": str(evidence_root),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "review_input_path": str(args.review_input.expanduser().resolve(strict=True)),
        "review_input_file_sha256": review_input_file_sha256,
        "lifecycle_contract_path": str(
            args.lifecycle_contract.expanduser().resolve(strict=True)
        ),
        "lifecycle_contract_file_sha256": lifecycle_file_sha256,
        "output_root": str(output_root),
        "output_paths": {name: str(path) for name, path in paths.items()},
        "review_spec_sha256": spec["spec_sha256"],
        "review_spec_bytes": len(spec_bytes),
        "supporting_artifact_count": len(spec["control_receipts"]),
        "supporting_artifacts_sha256": canonical_sha256(spec["control_receipts"]),
        "superseded_artifact_count": len(selected),
        "superseded_artifacts_sha256": canonical_sha256(selected),
        "raw_payload_embedded": False,
        "release_authorized": False,
        "filesystem_scan_performed": False,
        "apply_required": True,
    }
    plan = {**plan_core, "plan_sha256": canonical_sha256(plan_core)}
    if not args.apply:
        return {
            "ok": True,
            "mode": "dry_run_review_draft",
            "plan": plan,
            "apply_performed": False,
            "raw_release_performed": False,
        }
    if args.expected_plan_sha256 != plan["plan_sha256"]:
        raise EvidenceCertificateError(
            "--apply requires the exact plan SHA-256 from a prior draft dry-run"
        )
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _closeout_transaction_lock(output_root):
        _write_exact_json_v1(paths["spec"], spec)
        _write_exact_json_v1(paths["plan"], plan)
        result = {
            "ok": True,
            "mode": "review_spec_materialized",
            "plan_sha256": plan["plan_sha256"],
            "review_spec_path": str(paths["spec"]),
            "review_spec_sha256": spec["spec_sha256"],
            "supporting_artifact_count": len(spec["control_receipts"]),
            "superseded_artifact_count": len(selected),
            "raw_payload_embedded": False,
            "release_authorized": False,
            "raw_release_performed": False,
        }
        _write_exact_json_v1(paths["result"], result)
    return result


def _prepare_debug_closeout(args: argparse.Namespace) -> dict[str, Any]:
    """Dry-run or materialize one explicit, non-destructive DEBUG review bundle."""

    unresolved_evidence = args.evidence_root.expanduser()
    if unresolved_evidence.is_symlink():
        raise EvidenceCertificateError("evidence_root cannot be a symlink")
    evidence_root = unresolved_evidence.resolve(strict=True)
    if not evidence_root.is_dir():
        raise EvidenceCertificateError("evidence_root must be a directory")
    spec_path = args.review_spec.expanduser()
    if spec_path.is_symlink() or not spec_path.is_file():
        raise EvidenceCertificateError("review spec must be a regular file")
    spec_path = spec_path.resolve(strict=True)
    manifest_path = args.source_manifest.expanduser()
    if not manifest_path.is_absolute():
        manifest_path = evidence_root / manifest_path
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise EvidenceCertificateError("source manifest must be a regular file")
    manifest_path = manifest_path.resolve(strict=True)
    output_root = _debug_output_root_v1(
        evidence_root=evidence_root, supplied=args.output_root
    )
    bundle = build_debug_closeout_bundle_v1(
        evidence_root=evidence_root,
        source_manifest_path=manifest_path,
        review_spec=load_json_object(spec_path),
    )
    paths = {
        "request": output_root / "diagnostic_closeout_request.json",
        "review_receipt": output_root / "diagnostic_closeout_review_receipt.json",
        "report": output_root / "diagnostic_closeout_report.md",
        "plan": output_root / "diagnostic_closeout_preparation_plan.json",
        "result": output_root / "diagnostic_closeout_preparation_result.json",
    }
    request_bytes = pretty_json(bundle["request"]).encode("utf-8")
    receipt_bytes = pretty_json(bundle["review_receipt"]).encode("utf-8")
    report_bytes = bundle["report_markdown"].encode("utf-8")
    plan_core = {
        "schema_version": DEBUG_CLOSEOUT_PREPARATION_PLAN_SCHEMA_VERSION,
        "mode": "DRY_RUN_REVIEW_PREPARATION",
        "evidence_root": str(evidence_root),
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": bundle["source_manifest"]["manifest_sha256"],
        "review_spec_path": str(spec_path),
        "review_spec_sha256": bundle["review_spec_sha256"],
        "output_root": str(output_root),
        "output_paths": {name: str(path) for name, path in paths.items()},
        "request_sha256": bundle["request_sha256"],
        "review_receipt_sha256": bundle["review_receipt_sha256"],
        "report_sha256": bundle["report_sha256"],
        "request_bytes": len(request_bytes),
        "review_receipt_bytes": len(receipt_bytes),
        "report_bytes": len(report_bytes),
        "row_accounting": bundle["row_accounting"],
        "raw_payload_embedded": False,
        "release_authorized": False,
        "filesystem_scan_performed": False,
        "apply_required": True,
    }
    plan = {**plan_core, "plan_sha256": canonical_sha256(plan_core)}
    if not args.apply:
        return {
            "ok": True,
            "mode": "dry_run_review_preparation",
            "plan": plan,
            "apply_performed": False,
            "raw_release_performed": False,
        }
    if args.expected_plan_sha256 != plan["plan_sha256"]:
        raise EvidenceCertificateError(
            "--apply requires the exact plan SHA-256 from a prior dry-run"
        )
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with _closeout_transaction_lock(output_root):
        _write_exact_json_v1(paths["request"], bundle["request"])
        _write_exact_json_v1(paths["review_receipt"], bundle["review_receipt"])
        _write_exact_text_v1(paths["report"], bundle["report_markdown"])
        _write_exact_json_v1(paths["plan"], plan)
        result = {
            "ok": True,
            "mode": "review_bundle_materialized",
            "plan_sha256": plan["plan_sha256"],
            "request_path": str(paths["request"]),
            "request_sha256": bundle["request_sha256"],
            "review_receipt_path": str(paths["review_receipt"]),
            "review_receipt_sha256": bundle["review_receipt_sha256"],
            "report_path": str(paths["report"]),
            "report_sha256": bundle["report_sha256"],
            "row_accounting": bundle["row_accounting"],
            "raw_payload_embedded": False,
            "release_authorized": False,
            "raw_release_performed": False,
        }
        _write_exact_json_v1(paths["result"], result)
    return result


def _upgrade_request_bindings(args: argparse.Namespace) -> dict[str, Any]:
    """Write one bounded upgraded copy; never mutate or apply the source request."""

    request_path = args.request.expanduser()
    if request_path.is_symlink() or not request_path.is_file():
        raise EvidenceCertificateError("upgrade request must be a regular file")
    request_path = request_path.resolve(strict=True)
    evidence_root = args.evidence_root.expanduser()
    if evidence_root.is_symlink():
        raise EvidenceCertificateError("upgrade evidence_root cannot be a symlink")
    evidence_root = evidence_root.resolve(strict=True)
    if not evidence_root.is_dir():
        raise EvidenceCertificateError("upgrade evidence_root must be a directory")
    source_manifest_paths = [Path(path) for path in args.source_manifest]
    protected_inputs = {request_path}
    for supplied in source_manifest_paths:
        unresolved = supplied.expanduser()
        if not unresolved.is_absolute():
            unresolved = evidence_root / unresolved
        protected_inputs.add(unresolved.resolve(strict=True))
    diagnostic_spec = None
    if args.diagnostic_upgrade_spec is not None:
        spec_path = args.diagnostic_upgrade_spec.expanduser()
        if spec_path.is_symlink() or not spec_path.is_file():
            raise EvidenceCertificateError(
                "diagnostic upgrade spec must be a regular file"
            )
        spec_path = spec_path.resolve(strict=True)
        protected_inputs.add(spec_path)
        diagnostic_spec = load_json_object(spec_path)
        if diagnostic_spec.get("schema_version") != (
            DIAGNOSTIC_UPGRADE_SPEC_SCHEMA_VERSION
        ):
            raise EvidenceCertificateError("diagnostic upgrade spec schema mismatch")
    output_path = args.out.expanduser()
    if output_path.exists() or output_path.is_symlink():
        raise EvidenceCertificateError("upgrade output must be a fresh path")
    output_path = output_path.resolve(strict=False)
    if output_path in protected_inputs:
        raise EvidenceCertificateError(
            "upgrade output must not overwrite any explicit input"
        )
    upgraded = upgrade_evidence_certificate_request_source_bindings_v1(
        load_json_object(request_path),
        evidence_root=evidence_root,
        source_manifest_paths=source_manifest_paths,
        diagnostic_upgrade_spec=diagnostic_spec,
        max_manifests=args.max_manifests,
        max_rows=args.max_rows,
    )
    write_json_atomic(output_path, upgraded)
    binding_count = sum(
        1 for row in upgraded["artifacts"] if "source_manifest_binding" in row
    )
    return {
        "ok": True,
        "mode": "upgrade_request_bindings",
        "source_request_path": str(request_path),
        "source_request_sha256": canonical_sha256(load_json_object(request_path)),
        "upgraded_request_path": str(output_path),
        "upgraded_request_sha256": canonical_sha256(upgraded),
        "source_manifest_count": len(source_manifest_paths),
        "source_binding_count": binding_count,
        "diagnostic_upgrade_applied": diagnostic_spec is not None,
        "apply_performed": False,
        "archive_invoked": False,
    }


def _root_retirement_paths(
    args: argparse.Namespace, request: Mapping[str, Any]
) -> dict[str, Path]:
    unresolved = args.output_root.expanduser()
    if unresolved.is_symlink():
        raise RootRetirementError("retirement output_root cannot be a symlink")
    output_root = unresolved.resolve(strict=False)
    for row in request["roots"]:
        root = Path(str(row["path"])).resolve(strict=False)
        if (
            output_root == root
            or output_root in root.parents
            or root in output_root.parents
        ):
            raise RootRetirementError(
                "retirement output_root must be disjoint from every target root"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "output_root": output_root,
        "inventory": output_root / "root_retirement_inventory.json",
        "intent": output_root / "root_retirement_intent.json",
        "review": output_root / "root_retirement_review.json",
        "dry_receipt": output_root / "root_retirement_receipt.dry_run.json",
        "receipt": output_root / "root_retirement_receipt.json",
        "result": output_root / "root_retirement_result.json",
    }


def _root_retirement_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    request = validate_root_retirement_request_v1(load_json_object(args.request))
    paths = _root_retirement_paths(args, request)
    occupied = [
        path
        for name, path in paths.items()
        if name != "output_root" and (path.exists() or path.is_symlink())
    ]
    if occupied:
        raise RootRetirementError(
            "root-retirement dry-run output is not fresh: "
            + ", ".join(str(path) for path in occupied)
        )
    inventory = build_root_retirement_inventory_v1(request)
    intent = build_root_retirement_intent_v1(
        request=request, inventory=inventory, status="PREPARED"
    )
    dry_core = {
        "schema_version": "eiel_runtime_root_retirement_dry_run_receipt_v1",
        "retirement_id": request["retirement_id"],
        "request_sha256": canonical_sha256(request),
        "inventory_sha256": inventory["inventory_sha256"],
        "intent_sha256": intent["intent_sha256"],
        "root_count": inventory["root_count"],
        "member_count": inventory["member_count"],
        "bytes_selected": inventory["bytes"],
        "result": "dry_run",
        "archive_invoked": False,
        "slow_storage_waited": False,
    }
    dry_receipt = {**dry_core, "receipt_sha256": canonical_sha256(dry_core)}
    write_json_atomic(paths["inventory"], inventory)
    write_json_atomic(paths["intent"], intent)
    write_json_atomic(paths["dry_receipt"], dry_receipt)
    result = {
        "ok": True,
        "mode": "dry_run",
        "retirement_id": request["retirement_id"],
        "inventory_path": str(paths["inventory"]),
        "inventory_sha256": inventory["inventory_sha256"],
        "intent_path": str(paths["intent"]),
        "intent_sha256": intent["intent_sha256"],
        "review_path": str(paths["review"]),
        "dry_run_receipt_path": str(paths["dry_receipt"]),
        "root_count": inventory["root_count"],
        "member_count": inventory["member_count"],
        "bytes_selected": inventory["bytes"],
        "apply_required": True,
        "review_required": True,
        "archive_invoked": False,
    }
    write_json_atomic(paths["result"], result)
    return result


def _validated_root_retirement_dry_receipt(path: Path) -> dict[str, Any]:
    value = load_json_object(path)
    supplied = value.pop("receipt_sha256", None)
    required = {
        "schema_version",
        "retirement_id",
        "request_sha256",
        "inventory_sha256",
        "intent_sha256",
        "root_count",
        "member_count",
        "bytes_selected",
        "result",
        "archive_invoked",
        "slow_storage_waited",
    }
    if set(value) != required or supplied != canonical_sha256(value):
        raise RootRetirementError("root-retirement dry-run receipt integrity mismatch")
    if (
        value.get("schema_version") != "eiel_runtime_root_retirement_dry_run_receipt_v1"
        or value.get("result") != "dry_run"
        or value.get("archive_invoked") is not False
        or value.get("slow_storage_waited") is not False
    ):
        raise RootRetirementError("root-retirement dry-run receipt semantics mismatch")
    return {**value, "receipt_sha256": supplied}


def _validated_applied_receipt(
    path: Path,
    *,
    request: Mapping[str, Any],
    inventory: Mapping[str, Any],
    review: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    value = load_json_object(path)
    supplied = value.pop("receipt_sha256", None)
    required = {
        "schema_version",
        "retirement_id",
        "programme_id",
        "work_id",
        "request_sha256",
        "inventory_sha256",
        "final_intent_sha256",
        "result",
        "retired_roots",
        "root_count",
        "bytes_released",
        "compact_authority",
        "review_sha256",
        "archive_invoked",
        "slow_storage_waited",
        "completed_at_utc",
    }
    if set(value) != required or supplied != canonical_sha256(value):
        raise RootRetirementError("durable root-retirement receipt integrity mismatch")
    inventory_by_id = {
        str(row["root_id"]): dict(row) for row in inventory.get("roots", [])
    }
    request_ids = {str(row["root_id"]) for row in request["roots"]}
    if set(inventory_by_id) != request_ids or any(
        inventory_by_id[str(row["root_id"])]["path"] != row["path"]
        for row in request["roots"]
    ):
        raise RootRetirementError("durable receipt request/inventory root sets differ")
    expected_roots = [
        {
            "root_id": row["root_id"],
            "path": row["path"],
            "inventory_sha256": inventory_by_id[str(row["root_id"])][
                "inventory_sha256"
            ],
            "bytes_released": inventory_by_id[str(row["root_id"])]["bytes"],
        }
        for row in request["roots"]
    ]
    retired_events = [
        event
        for event in intent.get("events", [])
        if isinstance(event, Mapping) and event.get("event") == "RETIRED"
    ]
    expected_completed_at = retired_events[-1].get("at_utc") if retired_events else None
    if (
        value.get("schema_version") != "eiel_runtime_root_retirement_receipt_v1"
        or value.get("result") != "applied"
        or value.get("retirement_id") != request["retirement_id"]
        or value.get("programme_id") != request["programme_id"]
        or value.get("work_id") != request["work_id"]
        or value.get("request_sha256") != canonical_sha256(request)
        or value.get("inventory_sha256") != inventory["inventory_sha256"]
        or value.get("review_sha256") != review["review_sha256"]
        or value.get("retired_roots") != expected_roots
        or value.get("root_count") != len(expected_roots)
        or value.get("bytes_released") != inventory["bytes"]
        or value.get("compact_authority") != request["compact_authority"]
        or value.get("archive_invoked") is not False
        or value.get("slow_storage_waited") is not False
        or intent.get("status") != "APPLIED"
        or value.get("final_intent_sha256") != intent.get("intent_sha256")
        or len(retired_events) != len(expected_roots)
        or intent.get("events", [])[-1].get("event") != "RETIRED"
        or value.get("completed_at_utc") != expected_completed_at
    ):
        raise RootRetirementError("durable root-retirement receipt semantics mismatch")
    return {**value, "receipt_sha256": supplied}


def _assert_root_retirement_targets_absent_v1(
    request: Mapping[str, Any],
) -> None:
    """Prove the exact retired roots and their stages remain absent."""

    for row in request["roots"]:
        root = Path(str(row["path"]))
        stage = root.parent / (
            ".eiel-retirement-"
            + canonical_sha256(
                {
                    "retirement_id": request["retirement_id"],
                    "root_id": row["root_id"],
                }
            )[:24]
        )
        if root.exists() or root.is_symlink() or stage.exists() or stage.is_symlink():
            raise RootRetirementError(
                "durable retirement receipt conflicts with a surviving source or stage: "
                f"{root}"
            )


def _root_retirement_apply(
    args: argparse.Namespace, *, _transaction_fences_held: bool = False
) -> dict[str, Any]:
    request = validate_root_retirement_request_v1(load_json_object(args.request))
    paths = _root_retirement_paths(args, request)
    if not _transaction_fences_held:
        roots = sorted({Path(row["path"]) for row in request["roots"]}, key=str)
        with ExitStack() as stack:
            for root in roots:
                stack.enter_context(run_artifact_transaction_fence_v1(root))
            return _root_retirement_apply(args, _transaction_fences_held=True)
    for name in ("inventory", "intent", "dry_receipt", "review"):
        if not paths[name].is_file() or paths[name].is_symlink():
            raise RootRetirementError(
                "--apply requires the exact prior dry-run artifacts"
            )
    inventory = validate_root_retirement_inventory_v1(
        load_json_object(paths["inventory"])
    )
    intent = validate_root_retirement_intent_v1(load_json_object(paths["intent"]))
    review = validate_root_retirement_review_v1(
        load_json_object(paths["review"]),
        request_sha256=canonical_sha256(request),
        inventory_sha256=inventory["inventory_sha256"],
    )
    dry_receipt = _validated_root_retirement_dry_receipt(paths["dry_receipt"])
    if (
        dry_receipt["request_sha256"] != canonical_sha256(request)
        or dry_receipt["inventory_sha256"] != inventory["inventory_sha256"]
        or dry_receipt["root_count"] != inventory["root_count"]
        or dry_receipt["member_count"] != inventory["member_count"]
        or dry_receipt["bytes_selected"] != inventory["bytes"]
    ):
        raise RootRetirementError("dry-run request/inventory summary binding mismatch")
    if paths["receipt"].is_file() and not paths["receipt"].is_symlink():
        receipt = _validated_applied_receipt(
            paths["receipt"],
            request=request,
            inventory=inventory,
            review=review,
            intent=intent,
        )
        if intent.get("status") != "APPLIED" or intent.get(
            "intent_sha256"
        ) != receipt.get("final_intent_sha256"):
            raise RootRetirementError("applied receipt/final intent binding mismatch")
        if intent.get("prepared_intent_sha256") != dry_receipt["intent_sha256"]:
            raise RootRetirementError("applied intent/dry-run origin binding mismatch")
        _assert_root_retirement_targets_absent_v1(request)
        return {
            "ok": True,
            "mode": "apply",
            "retirement_id": request["retirement_id"],
            "receipt_path": str(paths["receipt"]),
            "receipt_sha256": receipt["receipt_sha256"],
            "root_count": receipt["root_count"],
            "bytes_released": receipt["bytes_released"],
            "idempotent_replay": True,
            "archive_invoked": False,
        }
    if intent.get("status") == "PREPARED":
        intent_origin_matches = dry_receipt["intent_sha256"] == intent["intent_sha256"]
    elif intent.get("status") in {"APPLYING", "APPLIED"}:
        intent_origin_matches = (
            intent.get("prepared_intent_sha256") == dry_receipt["intent_sha256"]
        )
    else:
        intent_origin_matches = False
    if not intent_origin_matches:
        raise RootRetirementError("dry-run request/inventory/intent binding mismatch")
    receipt = apply_root_retirement_v1(
        request=request,
        inventory=inventory,
        intent=intent,
        review=review,
        progress_path=paths["intent"],
        _transaction_fences_held=True,
    )
    write_json_atomic(paths["receipt"], receipt)
    final_intent = validate_root_retirement_intent_v1(load_json_object(paths["intent"]))
    durable_receipt = _validated_applied_receipt(
        paths["receipt"],
        request=request,
        inventory=inventory,
        review=review,
        intent=final_intent,
    )
    if durable_receipt != receipt:
        raise RootRetirementError(
            "durable root-retirement receipt differs after publication"
        )
    _assert_root_retirement_targets_absent_v1(request)
    result = {
        "ok": True,
        "mode": "apply",
        "retirement_id": request["retirement_id"],
        "receipt_path": str(paths["receipt"]),
        "receipt_sha256": receipt["receipt_sha256"],
        "root_count": receipt["root_count"],
        "bytes_released": receipt["bytes_released"],
        "idempotent_replay": False,
        "archive_invoked": False,
    }
    write_json_atomic(paths["result"], result)
    if load_json_object(paths["result"]) != result:
        raise RootRetirementError(
            "durable root-retirement result differs after publication"
        )
    _assert_root_retirement_targets_absent_v1(request)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evidence certificate and continuous gate-boundary reclamation"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    close = sub.add_parser("close-gate")
    close.add_argument("--request", required=True, type=Path)
    close.add_argument("--evidence-root", required=True, type=Path)
    close.add_argument("--output-root", required=True, type=Path)
    close.add_argument("--permanent-root", type=Path)
    close.add_argument(
        "--max-permanent-capsule-bytes",
        type=int,
        default=DEFAULT_MAX_CAPSULE_BYTES,
    )
    close.add_argument(
        "--max-permanent-root-bytes",
        type=int,
        default=DEFAULT_MAX_PERMANENT_ROOT_BYTES,
    )
    close.add_argument("--apply", action="store_true")
    audit = sub.add_parser("audit-legacy")
    audit.add_argument("root", type=Path)
    audit.add_argument("--max-objects", type=int, default=4096)
    audit.add_argument("--max-bytes", type=int, default=512 * 1024 * 1024 * 1024)
    audit.add_argument("--out", type=Path)
    retire = sub.add_parser("retire-roots")
    retire.add_argument("--request", required=True, type=Path)
    retire.add_argument("--output-root", required=True, type=Path)
    retire.add_argument("--apply", action="store_true")
    draft_debug = sub.add_parser(
        "draft-debug-review",
        help="draft one exact manifest-bound DEBUG closeout review spec",
    )
    draft_debug.add_argument("--evidence-root", required=True, type=Path)
    draft_debug.add_argument("--source-manifest", required=True, type=Path)
    draft_debug.add_argument("--lifecycle-contract", required=True, type=Path)
    draft_debug.add_argument("--review-input", required=True, type=Path)
    draft_debug.add_argument(
        "--supporting-artifact",
        required=True,
        action="append",
        help="exact ROLE=MANIFEST_RELATIVE_PATH; repeat for each small support file",
    )
    draft_debug.add_argument("--output-root", required=True, type=Path)
    draft_debug.add_argument("--expected-plan-sha256")
    draft_debug.add_argument("--apply", action="store_true")
    prepare_debug = sub.add_parser(
        "prepare-debug-closeout",
        help="prepare a complete request/report from one reviewed DEBUG spec",
    )
    prepare_debug.add_argument("--evidence-root", required=True, type=Path)
    prepare_debug.add_argument("--source-manifest", required=True, type=Path)
    prepare_debug.add_argument("--review-spec", required=True, type=Path)
    prepare_debug.add_argument("--output-root", required=True, type=Path)
    prepare_debug.add_argument("--expected-plan-sha256")
    prepare_debug.add_argument("--apply", action="store_true")
    upgrade = sub.add_parser(
        "upgrade-request-bindings",
        help="write a validated request copy from explicitly named source manifests",
    )
    upgrade.add_argument("--request", required=True, type=Path)
    upgrade.add_argument("--evidence-root", required=True, type=Path)
    upgrade.add_argument(
        "--source-manifest",
        required=True,
        action="append",
        type=Path,
        help="exact source disposition manifest; repeat for multiple manifests",
    )
    upgrade.add_argument("--diagnostic-upgrade-spec", type=Path)
    upgrade.add_argument("--out", required=True, type=Path)
    upgrade.add_argument(
        "--max-manifests",
        type=int,
        default=MAX_SOURCE_BINDING_MANIFESTS,
    )
    upgrade.add_argument(
        "--max-rows",
        type=int,
        default=MAX_SOURCE_BINDING_ROWS,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "close-gate":
            result = _apply(args) if args.apply else _dry_run(args)
        elif args.command == "draft-debug-review":
            result = _draft_debug_closeout_review(args)
        elif args.command == "prepare-debug-closeout":
            result = _prepare_debug_closeout(args)
        elif args.command == "upgrade-request-bindings":
            result = _upgrade_request_bindings(args)
        elif args.command == "retire-roots":
            result = (
                _root_retirement_apply(args)
                if args.apply
                else _root_retirement_dry_run(args)
            )
        else:
            result = historical_audit(
                args.root, max_objects=args.max_objects, max_bytes=args.max_bytes
            )
            if args.out is not None:
                write_json_atomic(args.out, result)
            result = {"ok": True, "audit": result}
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ArtifactLifecycleError,
        EvidenceCertificateError,
        RootRetirementError,
        RuntimeError,
    ) as exc:
        result = {
            "ok": False,
            "mode": "blocked",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "archive_invoked": False,
        }
    print(json.dumps(result, indent=2, sort_keys=True) + "\n", end="")
    return 0 if result.get("ok") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
