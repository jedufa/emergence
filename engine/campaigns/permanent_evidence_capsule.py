"""Bounded, immutable publication of compact Evidence Certificate capsules.

The permanent capsule is a metadata-only projection of an Evidence Certificate.
It deliberately contains identities, conclusions, limitations, reproduction
instructions, and exact inventory digests, but never copies the certified raw
objects.  Publication is content-addressed and must complete before reclamation.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Mapping, Sequence
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Final

from .canonical import canonical_bytes, canonical_sha256
from .evidence_certificate import (
    EvidenceCertificateError,
    validate_evidence_certificate_v1,
)
from .storage_availability import (
    DATA2_PERMANENT_EVIDENCE_MAX_CAPSULE_BYTES,
    DATA2_PERMANENT_EVIDENCE_MAX_TOTAL_BYTES,
    DATA2_PERMANENT_EVIDENCE_ROOT,
    load_data2_storage_availability_policy_v1,
)


CAPSULE_SCHEMA_VERSION: Final[str] = "eiel_permanent_evidence_capsule_v1"
CAPSULE_PUBLICATION_SCHEMA_VERSION: Final[str] = (
    "eiel_permanent_evidence_capsule_publication_v1"
)
CAPSULE_COMPLETE_SCHEMA_VERSION: Final[str] = (
    "eiel_permanent_evidence_capsule_complete_v1"
)
CAPSULE_FAMILY_RESERVATION_SCHEMA_VERSION: Final[str] = (
    "eiel_permanent_evidence_capsule_family_reservation_v1"
)
DEFAULT_PERMANENT_EVIDENCE_ROOT: Final[Path] = DATA2_PERMANENT_EVIDENCE_ROOT
DEFAULT_MAX_CAPSULE_BYTES: Final[int] = DATA2_PERMANENT_EVIDENCE_MAX_CAPSULE_BYTES
DEFAULT_MAX_PERMANENT_ROOT_BYTES: Final[int] = DATA2_PERMANENT_EVIDENCE_MAX_TOTAL_BYTES

_ARTIFACT_FIELDS: Final[tuple[str, ...]] = (
    "artifact_id",
    "path",
    "artifact_family",
    "artifact_type",
    "owner_writer",
    "bytes",
    "sha256",
    "tree_sha256",
    "evidence_class",
    "source_retention_class",
    "retention_class",
    "diagnostic_closeout_disposition",
    "retention_transition",
    "disposition",
    "disposition_reason",
    "release_eligible",
    "produced_by",
    "consumers",
    "last_use",
    "remaining_consumers",
    "retention_authorities",
    "forensic_hold",
    "source_manifest_binding",
)


def _mapping(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceCertificateError(f"{path} must be an object")
    return dict(value)


def _positive_limit(value: int, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise EvidenceCertificateError(f"{name} must be a positive integer")
    return value


def default_permanent_evidence_root_v1(evidence_root: Path) -> Path:
    """Return the canonical permanent root for evidence anywhere under `configured runtime root`."""

    root = evidence_root.expanduser().resolve(strict=False)
    data2_root = Path(public_runtime_path())
    if root != data2_root and data2_root not in root.parents:
        raise EvidenceCertificateError(
            "a permanent evidence root is required outside configured runtime root"
        )
    if root == DEFAULT_PERMANENT_EVIDENCE_ROOT.resolve(strict=False):
        raise EvidenceCertificateError(
            "the permanent evidence root cannot be its own evidence source"
        )
    return DEFAULT_PERMANENT_EVIDENCE_ROOT


def _artifact_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {name: row[name] for name in _ARTIFACT_FIELDS if name in row}


def _count_bytes_by(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[str, dict[str, int]]:
    names = sorted({str(row.get(key)) for row in rows})
    return {
        name: {
            "objects": sum(1 for row in rows if str(row.get(key)) == name),
            "bytes": sum(
                int(row.get("bytes", 0)) for row in rows if str(row.get(key)) == name
            ),
        }
        for name in names
    }


def _artifact_accounting(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    projections = [_artifact_projection(row) for row in rows]
    releasable = [row for row in rows if row.get("release_eligible") is True]
    superseded_ids = sorted(
        str(row.get("artifact_id"))
        for row in rows
        if row.get("diagnostic_closeout_disposition") == "SUPERSEDED_BY_CERTIFICATE"
    )
    artifact_ids = sorted(str(row.get("artifact_id")) for row in rows)
    return {
        "object_count": len(rows),
        "produced_bytes": sum(int(row.get("bytes", 0)) for row in rows),
        "release_eligible_objects": len(releasable),
        "release_eligible_bytes": sum(int(row.get("bytes", 0)) for row in releasable),
        "retained_objects": len(rows) - len(releasable),
        "retained_bytes": sum(
            int(row.get("bytes", 0))
            for row in rows
            if row.get("release_eligible") is not True
        ),
        "inventory_sha256": canonical_sha256(projections),
        "artifact_ids_sha256": canonical_sha256(artifact_ids),
        "superseded_objects": len(superseded_ids),
        "superseded_artifact_ids_sha256": canonical_sha256(superseded_ids),
        "by_source_retention_class": _count_bytes_by(rows, "source_retention_class"),
        "by_effective_retention_class": _count_bytes_by(rows, "retention_class"),
        "by_disposition": _count_bytes_by(rows, "disposition"),
    }


def _identity_projection(identity: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(identity)
    artifact_hashes = _mapping(
        raw.get("artifact_hashes", {}), path="certificate.identity.artifact_hashes"
    )
    rng_bindings = list(raw.get("rng_bindings", []))
    return {
        "source": raw["source"],
        "runtime": raw["runtime"],
        "config_sha256": raw["config_sha256"],
        "request_sha256": raw["request_sha256"],
        "plan_sha256": raw["plan_sha256"],
        "environment": raw["environment"],
        "device": raw["device"],
        "artifact_hash_count": len(artifact_hashes),
        "artifact_hashes_sha256": canonical_sha256(artifact_hashes),
        "rng_binding_count": len(rng_bindings),
        "rng_bindings_sha256": canonical_sha256(rng_bindings),
        "rng_not_applicable_reason": raw.get("rng_not_applicable_reason"),
    }


def _context_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = dict(value)
    exact_ids = list(raw.get("exact_required_artifact_ids", []))
    return {
        "status": raw.get("status"),
        "authority_reference": raw.get("authority_reference"),
        "exact_required_artifact_count": len(exact_ids),
        "exact_required_artifact_ids_sha256": canonical_sha256(exact_ids),
    }


def build_permanent_evidence_capsule_v1(
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the deterministic metadata-only durable projection."""

    cert = validate_evidence_certificate_v1(certificate)
    validation = _mapping(cert["validation"], path="certificate.validation")
    mechanical_receipts = list(validation.get("mechanical_receipts", []))
    programme_plan = _mapping(cert["programme_plan"], path="certificate.programme_plan")
    core: dict[str, Any] = {
        "schema_version": CAPSULE_SCHEMA_VERSION,
        "phase": cert["phase"],
        "certificate_id": cert["certificate_id"],
        "certificate_sha256": cert["certificate_sha256"],
        "request_sha256": cert["request_sha256"],
        "programme_id": cert["programme_id"],
        "work_id": cert["work_id"],
        "gate_id": cert["gate_id"],
        "obligation_id": cert["obligation_id"],
        "bounded_claim": cert["bounded_claim"],
        "identity": _identity_projection(cert["identity"]),
        "programme_plan": {
            "plan_id": programme_plan["plan_id"],
            "plan_sha256": programme_plan["plan_sha256"],
            "closure_receipt_sha256": programme_plan["closure_receipt_sha256"],
            "ordered_gates_sha256": canonical_sha256(programme_plan["ordered_gates"]),
            "closed_gates_sha256": canonical_sha256(programme_plan["closed_gates"]),
            "remaining_gate_count": len(programme_plan["remaining_gates"]),
        },
        "evidence_obligations": cert["evidence_obligations"],
        "validation_summary": {
            "mechanical_receipt_count": len(mechanical_receipts),
            "mechanical_receipts_sha256": canonical_sha256(mechanical_receipts),
            "semantic_review": validation["semantic_review"],
        },
        "result": cert["result"],
        "reproduction": cert["reproduction"],
        "qualification_context": _context_projection(cert["qualification_context"]),
        "audit_context": _context_projection(cert["audit_context"]),
        "inventory_complete": cert["inventory_complete"],
        "artifact_accounting": _artifact_accounting(cert["artifacts"]),
        "raw_payload_policy": "IDENTITIES_AND_DIGESTS_ONLY_NO_RAW_PAYLOAD",
    }
    if "diagnostic_closeout" in cert:
        core["diagnostic_closeout"] = cert["diagnostic_closeout"]
    for name in (
        "prior_certificate_sha256",
        "finalized_at_utc",
        "storage_disposition_receipt_sha256",
    ):
        if name in cert:
            core[name] = cert[name]
    capsule_id = "eiel-capsule-" + canonical_sha256(core)[:24]
    body = {**core, "capsule_id": capsule_id}
    return {**body, "capsule_sha256": canonical_sha256(body)}


def validate_permanent_evidence_capsule_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    capsule = _mapping(value, path="capsule")
    supplied = capsule.pop("capsule_sha256", None)
    if capsule.get("schema_version") != CAPSULE_SCHEMA_VERSION:
        raise EvidenceCertificateError("permanent evidence capsule schema mismatch")
    if capsule.get("phase") not in {"SEALED_PRE_RELEASE", "FINAL"}:
        raise EvidenceCertificateError(
            "permanent evidence capsule phase is unsupported"
        )
    if capsule.get("raw_payload_policy") != (
        "IDENTITIES_AND_DIGESTS_ONLY_NO_RAW_PAYLOAD"
    ):
        raise EvidenceCertificateError("permanent capsule raw-payload policy mismatch")
    accounting = capsule.get("artifact_accounting")
    if not isinstance(accounting, Mapping):
        raise EvidenceCertificateError(
            "permanent capsule artifact accounting is missing"
        )
    if any(
        type(accounting.get(name)) is not int or int(accounting[name]) < 0
        for name in (
            "object_count",
            "produced_bytes",
            "release_eligible_objects",
            "release_eligible_bytes",
            "retained_objects",
            "retained_bytes",
            "superseded_objects",
        )
    ) or any(
        not isinstance(accounting.get(name), str) or len(str(accounting[name])) != 64
        for name in (
            "inventory_sha256",
            "artifact_ids_sha256",
            "superseded_artifact_ids_sha256",
        )
    ):
        raise EvidenceCertificateError(
            "permanent capsule artifact accounting is malformed"
        )
    capsule_id = capsule.get("capsule_id")
    identity_input = dict(capsule)
    identity_input.pop("capsule_id", None)
    expected_id = "eiel-capsule-" + canonical_sha256(identity_input)[:24]
    if capsule_id != expected_id:
        raise EvidenceCertificateError("permanent capsule ID binding mismatch")
    if supplied != canonical_sha256(capsule):
        raise EvidenceCertificateError("permanent capsule SHA-256 mismatch")
    return {**capsule, "capsule_sha256": supplied}


def _markdown_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_list(lines: list[str], values: Sequence[Any]) -> None:
    if values:
        lines.extend(f"- {_markdown_text(value)}" for value in values)
    else:
        lines.append("- None")


def render_permanent_evidence_report_v1(
    capsule: Mapping[str, Any],
) -> str:
    """Render a deterministic, detailed human-readable reading copy."""

    value = validate_permanent_evidence_capsule_v1(capsule)
    result = _mapping(value["result"], path="capsule.result")
    reproduction = _mapping(value["reproduction"], path="capsule.reproduction")
    identity = _mapping(value["identity"], path="capsule.identity")
    source = _mapping(identity["source"], path="capsule.identity.source")
    lines = [
        f"# Evidence capsule {value['capsule_id']}",
        "",
        "## Conclusion",
        "",
        f"- Decision: `{_markdown_text(result.get('decision'))}`",
        f"- Bounded claim: {_markdown_text(value['bounded_claim'])}",
        f"- Certificate phase: `{value['phase']}`",
        f"- Certificate SHA-256: `{value['certificate_sha256']}`",
        "",
        "## Exact identity",
        "",
        f"- Repository: `{_markdown_text(source.get('repository'))}`",
        f"- Source commit: `{_markdown_text(source.get('commit_sha'))}`",
        f"- Request SHA-256: `{value['request_sha256']}`",
        f"- Plan SHA-256: `{_markdown_text(identity.get('plan_sha256'))}`",
        f"- Configuration SHA-256: `{_markdown_text(identity.get('config_sha256'))}`",
        f"- Runtime: `{_markdown_text(json.dumps(identity.get('runtime'), sort_keys=True, separators=(',', ':')))}`",
        f"- Environment: `{_markdown_text(json.dumps(identity.get('environment'), sort_keys=True, separators=(',', ':')))}`",
        f"- Device: `{_markdown_text(json.dumps(identity.get('device'), sort_keys=True, separators=(',', ':')))}`",
        "",
        "## Evidence obligations and criteria",
        "",
    ]
    for row in value["evidence_obligations"]:
        lines.append(
            f"- `{_markdown_text(row.get('obligation_id'))}`: "
            f"{_markdown_text(row.get('status'))} — "
            f"{_markdown_text(row.get('description'))}"
        )
    lines.extend(["", "Criteria:", ""])
    for row in result.get("criteria", []):
        lines.append(
            "- "
            + _markdown_text(json.dumps(row, sort_keys=True, separators=(",", ":")))
        )
    lines.extend(
        [
            "",
            "Measurements:",
            "",
            "```json",
            json.dumps(result.get("measurements", {}), indent=2, sort_keys=True),
            "```",
            "",
            "Digests:",
            "",
            "```json",
            json.dumps(result.get("digests", {}), indent=2, sort_keys=True),
            "```",
            "",
            "## Limitations and no-claims",
            "",
            "Limitations:",
            "",
        ]
    )
    _markdown_list(lines, list(result.get("limitations", [])))
    lines.extend(["", "No-claims:", ""])
    _markdown_list(lines, list(result.get("no_claims", [])))
    diagnostic = value.get("diagnostic_closeout")
    if isinstance(diagnostic, Mapping):
        lines.extend(
            [
                "",
                "## Diagnostic closeout",
                "",
                f"- Conclusion: {_markdown_text(diagnostic.get('conclusion'))}",
                f"- Closure receipt SHA-256: `{_markdown_text(diagnostic.get('closure_receipt_sha256'))}`",
                f"- Source manifest SHA-256: `{_markdown_text(_mapping(diagnostic.get('source_manifest'), path='capsule.diagnostic_closeout.source_manifest').get('manifest_sha256'))}`",
                f"- Substantive review: `{_markdown_text(_mapping(diagnostic.get('substantive_review'), path='capsule.diagnostic_closeout.substantive_review').get('status'))}`",
                "",
                "Closeout limitations:",
                "",
            ]
        )
        _markdown_list(lines, list(diagnostic.get("limitations", [])))
    lines.extend(["", "## Reproduction", "", "Prerequisites:", ""])
    _markdown_list(lines, list(reproduction.get("prerequisites", [])))
    lines.extend(["", "Commands:", ""])
    for command in reproduction.get("commands", []):
        lines.extend(["```bash", str(command), "```", ""])
    lines.extend(["Verification rules:", ""])
    _markdown_list(lines, list(reproduction.get("verification_rules", [])))
    lines.extend(["", "Expected results:", ""])
    for row in reproduction.get("expected_results", []):
        lines.append(
            "- "
            + _markdown_text(json.dumps(row, sort_keys=True, separators=(",", ":")))
        )
    lines.extend(
        [
            "",
            f"Guarantee: `{_markdown_text(reproduction.get('guarantee'))}`",
            "",
            "## Exact sealed inventory accounting",
            "",
        ]
    )
    accounting = _mapping(
        value["artifact_accounting"], path="capsule.artifact_accounting"
    )
    lines.extend(
        [
            f"- Objects: `{accounting['object_count']}`",
            f"- Produced bytes: `{accounting['produced_bytes']}`",
            f"- Release eligible: `{accounting['release_eligible_objects']}` objects / `{accounting['release_eligible_bytes']}` bytes",
            f"- Retained: `{accounting['retained_objects']}` objects / `{accounting['retained_bytes']}` bytes",
            f"- Superseded objects: `{accounting['superseded_objects']}`",
            f"- Exact inventory SHA-256: `{accounting['inventory_sha256']}`",
            f"- Artifact-ID set SHA-256: `{accounting['artifact_ids_sha256']}`",
            f"- Superseded-ID set SHA-256: `{accounting['superseded_artifact_ids_sha256']}`",
            "",
            "Class and disposition counts:",
            "",
            "```json",
            json.dumps(
                {
                    "by_source_retention_class": accounting[
                        "by_source_retention_class"
                    ],
                    "by_effective_retention_class": accounting[
                        "by_effective_retention_class"
                    ],
                    "by_disposition": accounting["by_disposition"],
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "",
            "This capsule contains identities, conclusions, and reproduction metadata only. "
            "Its exact inventory is represented by digest and accounting summaries; it embeds "
            "no checkpoint, tensor, trace, log, archive, per-object inventory, or other raw payload.",
            "",
        ]
    )
    return "\n".join(lines)


def _family_reservation(
    certificate: Mapping[str, Any],
    *,
    capsule_limit: int,
    root_limit: int,
) -> tuple[dict[str, Any], bytes, str]:
    cert = validate_evidence_certificate_v1(certificate)
    pre_certificate_sha256 = (
        cert["certificate_sha256"]
        if cert["phase"] == "SEALED_PRE_RELEASE"
        else cert.get("prior_certificate_sha256")
    )
    if not isinstance(pre_certificate_sha256, str) or len(pre_certificate_sha256) != 64:
        raise EvidenceCertificateError(
            "FINAL capsule lacks its pre-release certificate family identity"
        )
    family_id = f"eiel-capsule-family-{pre_certificate_sha256[:24]}"
    core = {
        "schema_version": CAPSULE_FAMILY_RESERVATION_SCHEMA_VERSION,
        "family_id": family_id,
        "certificate_id": cert["certificate_id"],
        "pre_certificate_sha256": pre_certificate_sha256,
        "max_final_publication_bytes": capsule_limit,
        "max_root_bytes": root_limit,
    }
    value = {**core, "reservation_sha256": canonical_sha256(core)}
    content = canonical_bytes(value) + b"\n"
    name = f"{family_id}-{hashlib.sha256(content).hexdigest()[:16]}.reservation.json"
    return value, content, name


def _validate_family_reservation_bytes(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceCertificateError(
            "permanent capsule family reservation is not valid JSON"
        ) from exc
    reservation = _mapping(value, path="capsule family reservation")
    supplied = reservation.pop("reservation_sha256", None)
    if set(reservation) != {
        "schema_version",
        "family_id",
        "certificate_id",
        "pre_certificate_sha256",
        "max_final_publication_bytes",
        "max_root_bytes",
    }:
        raise EvidenceCertificateError(
            "permanent capsule family reservation schema is invalid"
        )
    pre_sha256 = reservation.get("pre_certificate_sha256")
    capsule_limit = reservation.get("max_final_publication_bytes")
    root_limit = reservation.get("max_root_bytes")
    if (
        reservation.get("schema_version") != CAPSULE_FAMILY_RESERVATION_SCHEMA_VERSION
        or not isinstance(pre_sha256, str)
        or len(pre_sha256) != 64
        or reservation.get("family_id") != f"eiel-capsule-family-{pre_sha256[:24]}"
        or type(capsule_limit) is not int
        or not 0 < capsule_limit <= DEFAULT_MAX_CAPSULE_BYTES
        or type(root_limit) is not int
        or not 0 < root_limit <= DEFAULT_MAX_PERMANENT_ROOT_BYTES
        or supplied != canonical_sha256(reservation)
    ):
        raise EvidenceCertificateError(
            "permanent capsule family reservation binding is invalid"
        )
    normalized = {**reservation, "reservation_sha256": supplied}
    if content != canonical_bytes(normalized) + b"\n":
        raise EvidenceCertificateError(
            "permanent capsule family reservation is not canonical"
        )
    return normalized


def _validate_complete_marker_bytes(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceCertificateError(
            "permanent capsule complete marker is not valid JSON"
        ) from exc
    marker = _mapping(value, path="capsule complete marker")
    supplied = marker.pop("complete_sha256", None)
    if marker.get(
        "schema_version"
    ) != CAPSULE_COMPLETE_SCHEMA_VERSION or supplied != canonical_sha256(marker):
        raise EvidenceCertificateError(
            "permanent capsule complete marker binding is invalid"
        )
    normalized = {**marker, "complete_sha256": supplied}
    if content != canonical_bytes(normalized) + b"\n":
        raise EvidenceCertificateError(
            "permanent capsule complete marker is not canonical"
        )
    return normalized


def _named_root_file(root: Path, name: Any, *, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise EvidenceCertificateError(f"{label} has an invalid file name")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise EvidenceCertificateError(f"{label} is not a regular file")
    return path


def _root_usage(root: Path) -> tuple[int, int, set[str]]:
    total = 0
    reservations: dict[str, tuple[dict[str, Any], Path, str]] = {}
    complete_markers: list[dict[str, Any]] = []
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise EvidenceCertificateError(
                f"permanent evidence root contains an unsupported entry: {path.name}"
            )
        total += path.stat().st_size
        if path.name.endswith(".reservation.json"):
            content = path.read_bytes()
            reservation = _validate_family_reservation_bytes(content)
            file_sha256 = hashlib.sha256(content).hexdigest()
            expected_name = (
                f"{reservation['family_id']}-{file_sha256[:16]}.reservation.json"
            )
            if path.name != expected_name or path.stat().st_nlink != 1:
                raise EvidenceCertificateError(
                    "permanent capsule reservation file identity is invalid"
                )
            semantic_sha256 = str(reservation["reservation_sha256"])
            if semantic_sha256 in reservations:
                raise EvidenceCertificateError(
                    "permanent evidence root contains a duplicate family reservation"
                )
            reservations[semantic_sha256] = (reservation, path, file_sha256)
        elif path.name.endswith(".complete.json"):
            complete_markers.append(_validate_complete_marker_bytes(path.read_bytes()))

    consumed: set[str] = set()
    for marker in complete_markers:
        reservation_sha256 = marker.get("family_reservation_sha256")
        if reservation_sha256 is None:
            continue
        if (
            not isinstance(reservation_sha256, str)
            or reservation_sha256 not in reservations
        ):
            raise EvidenceCertificateError(
                "FINAL capsule references an unknown family reservation"
            )
        reservation, reservation_path, reservation_file_sha256 = reservations[
            reservation_sha256
        ]
        if (
            marker.get("family_reservation_name") != reservation_path.name
            or marker.get("family_reservation_file_sha256") != reservation_file_sha256
            or marker.get("prior_certificate_sha256")
            != reservation["pre_certificate_sha256"]
        ):
            raise EvidenceCertificateError(
                "FINAL capsule family reservation binding is invalid"
            )
        json_path = _named_root_file(
            root, marker.get("json_name"), label="FINAL capsule JSON"
        )
        markdown_path = _named_root_file(
            root, marker.get("markdown_name"), label="FINAL capsule Markdown"
        )
        json_content = json_path.read_bytes()
        markdown_content = markdown_path.read_bytes()
        if hashlib.sha256(json_content).hexdigest() != marker.get(
            "json_sha256"
        ) or hashlib.sha256(markdown_content).hexdigest() != marker.get(
            "markdown_sha256"
        ):
            raise EvidenceCertificateError(
                "FINAL capsule family files differ from their complete marker"
            )
        try:
            capsule_value = json.loads(json_content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvidenceCertificateError("FINAL capsule JSON is invalid") from exc
        capsule = validate_permanent_evidence_capsule_v1(capsule_value)
        if (
            capsule.get("phase") != "FINAL"
            or capsule.get("prior_certificate_sha256")
            != reservation["pre_certificate_sha256"]
        ):
            raise EvidenceCertificateError(
                "FINAL capsule does not consume its exact family reservation"
            )
        consumed.add(reservation_sha256)
    reserved = sum(
        int(reservation["max_final_publication_bytes"])
        for semantic_sha256, (reservation, _, _) in reservations.items()
        if semantic_sha256 not in consumed
    )
    return total, reserved, consumed


def _write_immutable(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise EvidenceCertificateError(
                f"immutable permanent capsule path conflicts with existing content: {path}"
            )
        return
    temporary = path.parent / (
        f".{path.name}.{os.getpid()}-{canonical_sha256({'path': str(path), 'bytes': len(content)})[:12]}.tmp"
    )
    fd: int | None = None
    try:
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o640,
        )
        with os.fdopen(fd, "wb", closefd=True) as handle:
            fd = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise EvidenceCertificateError(
                    f"immutable permanent capsule path raced with different content: {path}"
                )
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def publish_permanent_evidence_capsule_v1(
    certificate: Mapping[str, Any],
    *,
    permanent_root: Path,
    source_root: Path,
    max_capsule_bytes: int = DEFAULT_MAX_CAPSULE_BYTES,
    max_root_bytes: int = DEFAULT_MAX_PERMANENT_ROOT_BYTES,
    allow_unreserved_final: bool = False,
) -> dict[str, Any]:
    """Publish and verify one immutable JSON/Markdown capsule pair."""

    capsule_limit = _positive_limit(max_capsule_bytes, name="max_capsule_bytes")
    root_limit = _positive_limit(max_root_bytes, name="max_root_bytes")
    if capsule_limit > DEFAULT_MAX_CAPSULE_BYTES:
        raise EvidenceCertificateError(
            "max_capsule_bytes cannot exceed the canonical storage policy"
        )
    if root_limit > DEFAULT_MAX_PERMANENT_ROOT_BYTES:
        raise EvidenceCertificateError(
            "max_root_bytes cannot exceed the canonical storage policy"
        )
    source = source_root.expanduser().resolve(strict=True)
    unresolved_root = permanent_root.expanduser()
    if unresolved_root.is_symlink():
        raise EvidenceCertificateError("permanent evidence root cannot be a symlink")
    root = unresolved_root.resolve(strict=False)
    try:
        required_data2_root = default_permanent_evidence_root_v1(source)
    except EvidenceCertificateError:
        required_data2_root = None
    if required_data2_root is not None and root != required_data2_root.resolve(
        strict=False
    ):
        raise EvidenceCertificateError(
            "configured runtime root runtime evidence must publish to the canonical permanent evidence root"
        )
    if root == DEFAULT_PERMANENT_EVIDENCE_ROOT.resolve(strict=False):
        policy = load_data2_storage_availability_policy_v1()
        permanent_policy = _mapping(
            policy.get("permanent_evidence_capsules"),
            path="storage policy permanent_evidence_capsules",
        )
        if (
            Path(str(permanent_policy.get("root"))).resolve(strict=False) != root
            or int(permanent_policy.get("max_capsule_bytes", 0))
            != DEFAULT_MAX_CAPSULE_BYTES
            or int(permanent_policy.get("max_total_bytes", 0))
            != DEFAULT_MAX_PERMANENT_ROOT_BYTES
        ):
            raise EvidenceCertificateError(
                "canonical permanent evidence policy binding is inconsistent"
            )
    if root == source or root in source.parents or source in root.parents:
        raise EvidenceCertificateError(
            "permanent evidence root must be disjoint from the evidence root"
        )
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise EvidenceCertificateError(
            "permanent evidence root is not a real directory"
        )

    capsule = build_permanent_evidence_capsule_v1(certificate)
    reservation, reservation_bytes, reservation_name = _family_reservation(
        certificate,
        capsule_limit=capsule_limit,
        root_limit=root_limit,
    )
    reservation_path = root / reservation_name
    reservation_file_sha256 = hashlib.sha256(reservation_bytes).hexdigest()
    json_bytes = canonical_bytes(capsule) + b"\n"
    markdown_bytes = render_permanent_evidence_report_v1(capsule).encode("utf-8")
    stem = f"{capsule['capsule_id']}-{capsule['capsule_sha256'][:16]}"
    json_path = root / f"{stem}.json"
    markdown_path = root / f"{stem}.md"
    complete_path = root / f"{stem}.complete.json"
    complete_core = {
        "schema_version": CAPSULE_COMPLETE_SCHEMA_VERSION,
        "capsule_id": capsule["capsule_id"],
        "capsule_sha256": capsule["capsule_sha256"],
        "json_name": json_path.name,
        "json_sha256": hashlib.sha256(json_bytes).hexdigest(),
        "markdown_name": markdown_path.name,
        "markdown_sha256": hashlib.sha256(markdown_bytes).hexdigest(),
    }
    if capsule["phase"] == "FINAL" and not allow_unreserved_final:
        complete_core.update(
            {
                "prior_certificate_sha256": capsule["prior_certificate_sha256"],
                "family_reservation_name": reservation_name,
                "family_reservation_sha256": reservation["reservation_sha256"],
                "family_reservation_file_sha256": reservation_file_sha256,
            }
        )
    complete_value = {
        **complete_core,
        "complete_sha256": canonical_sha256(complete_core),
    }
    complete_bytes = canonical_bytes(complete_value) + b"\n"
    total_bytes = len(json_bytes) + len(markdown_bytes) + len(complete_bytes)
    if total_bytes > capsule_limit:
        raise EvidenceCertificateError(
            f"permanent capsule exceeds max_capsule_bytes ({total_bytes} > {capsule_limit})"
        )
    pair = (
        (json_path, json_bytes),
        (markdown_path, markdown_bytes),
        (complete_path, complete_bytes),
    )
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX)
        existing_bytes, reserved_bytes, consumed_reservations = _root_usage(root)
        added_bytes = sum(len(content) for path, content in pair if not path.exists())
        reservation_exists = reservation_path.exists() or reservation_path.is_symlink()
        if reservation_exists:
            if (
                reservation_path.is_symlink()
                or not reservation_path.is_file()
                or reservation_path.read_bytes() != reservation_bytes
                or reservation_path.stat().st_nlink != 1
            ):
                raise EvidenceCertificateError(
                    "permanent capsule family reservation conflicts with existing content"
                )
        reservation_outstanding = (
            reservation_exists
            and reservation["reservation_sha256"] not in consumed_reservations
        )
        if capsule["phase"] == "SEALED_PRE_RELEASE":
            added_reservation_bytes = (
                0 if reservation_exists else len(reservation_bytes)
            )
            added_reserved_bytes = 0 if reservation_exists else capsule_limit
            projected_bytes = (
                existing_bytes
                + added_bytes
                + added_reservation_bytes
                + reserved_bytes
                + added_reserved_bytes
            )
        elif allow_unreserved_final:
            projected_bytes = existing_bytes + added_bytes + reserved_bytes
        else:
            complete_already_exists = all(
                path.is_file()
                and not path.is_symlink()
                and path.read_bytes() == content
                for path, content in pair
            )
            if not reservation_outstanding and not complete_already_exists:
                raise EvidenceCertificateError(
                    "FINAL permanent capsule lacks its outstanding pre-release capacity reservation"
                )
            if total_bytes > int(reservation["max_final_publication_bytes"]):
                raise EvidenceCertificateError(
                    "FINAL permanent capsule exceeds its pre-release capacity reservation"
                )
            projected_bytes = (
                existing_bytes
                + added_bytes
                + reserved_bytes
                - (capsule_limit if reservation_outstanding else 0)
            )
        if projected_bytes > root_limit:
            raise EvidenceCertificateError(
                "permanent evidence root byte budget would be exceeded"
            )
        if capsule["phase"] == "SEALED_PRE_RELEASE" and not reservation_exists:
            _write_immutable(reservation_path, reservation_bytes)
        # The marker is deliberately last.  A crash before it leaves only a
        # recoverable partial pair; idempotent publication completes and marks it.
        for path, content in pair:
            _write_immutable(path, content)
        if any(path.read_bytes() != content for path, content in pair):
            raise EvidenceCertificateError(
                "permanent capsule verification failed after publish"
            )
        verified_bytes, verified_reserved_bytes, verified_consumed = _root_usage(root)
        if (
            verified_bytes + verified_reserved_bytes > root_limit
            or (
                capsule["phase"] == "SEALED_PRE_RELEASE"
                and (not reservation_path.is_file() or reservation_path.is_symlink())
            )
            or (
                capsule["phase"] == "FINAL"
                and not allow_unreserved_final
                and reservation["reservation_sha256"] not in verified_consumed
            )
        ):
            raise EvidenceCertificateError(
                "permanent capsule family capacity verification failed after publish"
            )
    finally:
        fcntl.flock(directory_fd, fcntl.LOCK_UN)
        os.close(directory_fd)
    core = {
        "schema_version": CAPSULE_PUBLICATION_SCHEMA_VERSION,
        "capsule_id": capsule["capsule_id"],
        "capsule_sha256": capsule["capsule_sha256"],
        "certificate_id": capsule["certificate_id"],
        "certificate_sha256": capsule["certificate_sha256"],
        "phase": capsule["phase"],
        "permanent_root": str(root),
        "json_path": str(json_path),
        "json_sha256": hashlib.sha256(json_bytes).hexdigest(),
        "markdown_path": str(markdown_path),
        "markdown_sha256": hashlib.sha256(markdown_bytes).hexdigest(),
        "json_bytes": len(json_bytes),
        "markdown_bytes": len(markdown_bytes),
        "complete_path": str(complete_path),
        "complete_sha256": hashlib.sha256(complete_bytes).hexdigest(),
        "complete_bytes": len(complete_bytes),
        "total_bytes": total_bytes,
        "family_id": reservation["family_id"],
        "final_capacity_reservation_path": str(reservation_path),
        "final_capacity_reservation_sha256": reservation["reservation_sha256"],
        "final_capacity_reservation_file_sha256": reservation_file_sha256,
        "final_capacity_reserved_bytes": reservation["max_final_publication_bytes"],
        "final_capacity_reservation_status": (
            "UNRESERVED_LEGACY_MIGRATION"
            if capsule["phase"] == "FINAL" and allow_unreserved_final
            else "CONSUMED"
            if capsule["phase"] == "FINAL"
            else "RESERVED"
        ),
        "max_capsule_bytes": capsule_limit,
        "max_root_bytes": root_limit,
        "raw_payload_embedded": False,
        "verified": True,
    }
    return {**core, "publication_sha256": canonical_sha256(core)}


def recover_existing_permanent_evidence_publication_v1(
    certificate: Mapping[str, Any],
    *,
    permanent_root: Path,
    source_root: Path,
    max_capsule_bytes: int = DEFAULT_MAX_CAPSULE_BYTES,
    max_root_bytes: int = DEFAULT_MAX_PERMANENT_ROOT_BYTES,
) -> dict[str, Any] | None:
    """Recover only an already-complete content-addressed FINAL publication."""

    cert = validate_evidence_certificate_v1(certificate)
    if cert.get("phase") != "FINAL":
        raise EvidenceCertificateError(
            "permanent publication recovery requires a FINAL certificate"
        )
    source = source_root.expanduser().resolve(strict=True)
    unresolved_root = permanent_root.expanduser()
    if unresolved_root.is_symlink():
        raise EvidenceCertificateError("permanent evidence root cannot be a symlink")
    root = unresolved_root.resolve(strict=False)
    try:
        required_data2_root = default_permanent_evidence_root_v1(source)
    except EvidenceCertificateError:
        required_data2_root = None
    if required_data2_root is not None and root != required_data2_root.resolve(
        strict=False
    ):
        raise EvidenceCertificateError(
            "configured runtime root runtime evidence must recover from the canonical permanent evidence root"
        )
    if root == source or root in source.parents or source in root.parents:
        raise EvidenceCertificateError(
            "permanent evidence root must be disjoint from the evidence root"
        )
    if not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise EvidenceCertificateError(
            "permanent evidence root is not a real directory"
        )
    capsule = build_permanent_evidence_capsule_v1(cert)
    stem = f"{capsule['capsule_id']}-{capsule['capsule_sha256'][:16]}"
    complete_path = root / f"{stem}.complete.json"
    if complete_path.is_symlink():
        raise EvidenceCertificateError(
            "content-addressed FINAL capsule complete marker cannot be a symlink"
        )
    if not complete_path.exists():
        return None
    if not complete_path.is_file():
        raise EvidenceCertificateError(
            "content-addressed FINAL capsule complete marker is not a regular file"
        )
    return publish_permanent_evidence_capsule_v1(
        cert,
        permanent_root=root,
        source_root=source,
        max_capsule_bytes=max_capsule_bytes,
        max_root_bytes=max_root_bytes,
    )


def validate_permanent_evidence_publication_v1(
    value: Mapping[str, Any],
    *,
    certificate: Mapping[str, Any],
    source_root: Path,
    require_outstanding_final_capacity: bool = False,
    allow_unreserved_final: bool = False,
) -> dict[str, Any]:
    receipt = _mapping(value, path="publication")
    supplied = receipt.pop("publication_sha256", None)
    if (
        receipt.get("schema_version") != CAPSULE_PUBLICATION_SCHEMA_VERSION
        or receipt.get("verified") is not True
        or receipt.get("raw_payload_embedded") is not False
        or supplied != canonical_sha256(receipt)
    ):
        raise EvidenceCertificateError(
            "permanent capsule publication receipt is invalid"
        )
    expected = publish_permanent_evidence_capsule_v1(
        certificate,
        permanent_root=Path(str(receipt.get("permanent_root"))),
        source_root=source_root,
        max_capsule_bytes=int(receipt.get("max_capsule_bytes", 0)),
        max_root_bytes=int(receipt.get("max_root_bytes", 0)),
        allow_unreserved_final=allow_unreserved_final,
    )
    if expected != {**receipt, "publication_sha256": supplied}:
        raise EvidenceCertificateError(
            "permanent capsule publication differs from the bound certificate"
        )
    if require_outstanding_final_capacity:
        if expected.get("phase") != "SEALED_PRE_RELEASE":
            raise EvidenceCertificateError(
                "only a pre-release capsule can authorize future FINAL capacity"
            )
        root = Path(str(expected["permanent_root"]))
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(directory_fd, fcntl.LOCK_EX)
            actual_bytes, reserved_bytes, consumed = _root_usage(root)
            reservation_path = Path(str(expected["final_capacity_reservation_path"]))
            if (
                reservation_path.parent != root
                or reservation_path.is_symlink()
                or not reservation_path.is_file()
                or hashlib.sha256(reservation_path.read_bytes()).hexdigest()
                != expected["final_capacity_reservation_file_sha256"]
                or expected["final_capacity_reservation_sha256"] in consumed
                or actual_bytes + reserved_bytes > int(expected["max_root_bytes"])
            ):
                raise EvidenceCertificateError(
                    "pre-release capsule FINAL capacity reservation is not outstanding"
                )
        finally:
            fcntl.flock(directory_fd, fcntl.LOCK_UN)
            os.close(directory_fd)
    return expected


__all__ = [
    "CAPSULE_COMPLETE_SCHEMA_VERSION",
    "CAPSULE_FAMILY_RESERVATION_SCHEMA_VERSION",
    "CAPSULE_PUBLICATION_SCHEMA_VERSION",
    "CAPSULE_SCHEMA_VERSION",
    "DEFAULT_MAX_CAPSULE_BYTES",
    "DEFAULT_MAX_PERMANENT_ROOT_BYTES",
    "DEFAULT_PERMANENT_EVIDENCE_ROOT",
    "build_permanent_evidence_capsule_v1",
    "default_permanent_evidence_root_v1",
    "publish_permanent_evidence_capsule_v1",
    "recover_existing_permanent_evidence_publication_v1",
    "render_permanent_evidence_report_v1",
    "validate_permanent_evidence_capsule_v1",
    "validate_permanent_evidence_publication_v1",
]
