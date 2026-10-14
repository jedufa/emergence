#!/usr/bin/env python3
"""Validate and create bounded EIEL bridge records without transport side effects."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
import zipfile

from jsonschema import Draft202012Validator, FormatChecker


AUDIT_SCHEMA_VERSION = "eiel_bridge_record_audit_v1"
STATUS_SOURCE_SCHEMA_VERSION = "eiel_bridge_status_source_v1"
LATEST_STATUS_VERSION = "eiel_bridge_latest_status_v1"
ACTIVATION_RECEIPT_VERSION = "eiel_prompt_activation_receipt_v1"
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORBIDDEN_SECRET_KEYS = {
    "secret",
    "secrets",
    "token",
    "tokens",
    "password",
    "credential",
    "credentials",
    "private_key",
    "cookie",
    "cookies",
}


class BridgeRecordError(RuntimeError):
    """A bridge record cannot be accepted without weakening its contract."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BridgeRecordError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BridgeRecordError(f"JSON document must be an object: {path}")
    return value


def _git(path: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BridgeRecordError(f"git {' '.join(args)} failed: {detail}")
    return completed


def _project_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not (root / "bridge" / "templates").is_dir():
        raise BridgeRecordError(f"not an Emergence project root: {root}")
    return root


def _relative_file(project_root: Path, value: str, *, field: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or not value:
        raise BridgeRecordError(f"{field} must be a non-escaping project-relative path")
    unresolved = project_root / relative
    if unresolved.is_symlink():
        raise BridgeRecordError(f"{field} must not be a symbolic link: {value}")
    path = unresolved.resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise BridgeRecordError(f"{field} escapes the project root") from exc
    if not path.is_file():
        raise BridgeRecordError(f"{field} is not a safe regular file: {value}")
    return path


def _iter_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _iter_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_keys(child)


def _assert_no_secret_keys(value: Mapping[str, Any]) -> None:
    forbidden = sorted({key.lower() for key in _iter_keys(value)} & _FORBIDDEN_SECRET_KEYS)
    if forbidden:
        raise BridgeRecordError(f"bridge record contains forbidden secret keys: {forbidden}")


def _schema_registry(project_root: Path) -> dict[str, Path]:
    return {
        LATEST_STATUS_VERSION: project_root / "bridge/templates/latest_status_v1.schema.json",
        ACTIVATION_RECEIPT_VERSION: project_root / "bridge/templates/prompt_receipt_v1.schema.json",
    }


def _validator(project_root: Path, schema_version: str) -> Draft202012Validator:
    schema_path = _schema_registry(project_root).get(schema_version)
    if schema_path is None:
        raise BridgeRecordError(f"unsupported bridge schema version: {schema_version or '<missing>'}")
    schema = _read_object(schema_path)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _validation_errors(
    project_root: Path, schema_version: str, value: Mapping[str, Any]
) -> list[dict[str, Any]]:
    validator = _validator(project_root, schema_version)
    rows = []
    for error in sorted(validator.iter_errors(value), key=lambda item: (list(item.path), item.message)):
        rows.append(
            {
                "record_path": [str(part) for part in error.absolute_path],
                "schema_path": [str(part) for part in error.absolute_schema_path],
                "message": error.message,
            }
        )
    return rows


def audit_records(project_root: Path) -> dict[str, Any]:
    project_root = _project_root(project_root)
    targets = [(project_root / "bridge/latest_status.json", "mutable_latest_status")]
    targets.extend(
        (path, "append_only_prompt_receipt")
        for path in sorted((project_root / "bridge/prompt_receipts").glob("*.json"))
    )
    records: list[dict[str, Any]] = []
    blocking_count = 0
    finding_count = 0
    registry = _schema_registry(project_root)
    for path, role in targets:
        relative = str(path.relative_to(project_root))
        try:
            value = _read_object(path)
            _assert_no_secret_keys(value)
            declared = str(value.get("schema_version") or "")
            if declared not in registry:
                errors = [
                    {
                        "record_path": ["schema_version"],
                        "schema_path": [],
                        "message": f"unsupported bridge schema version: {declared or '<missing>'}",
                    }
                ]
                schema_status = "unsupported"
            else:
                errors = _validation_errors(project_root, declared, value)
                schema_status = "known"
            valid = not errors
            row = {
                "path": relative,
                "role": role,
                "declared_schema_version": declared or None,
                "schema_status": schema_status,
                "record_sha256": _sha256_file(path),
                "valid": valid,
                "errors": errors,
            }
        except BridgeRecordError as exc:
            row = {
                "path": relative,
                "role": role,
                "declared_schema_version": None,
                "schema_status": "unreadable",
                "record_sha256": _sha256_file(path) if path.is_file() else None,
                "valid": False,
                "errors": [{"record_path": [], "schema_path": [], "message": str(exc)}],
            }
        if not row["valid"]:
            if role == "mutable_latest_status":
                row["severity"] = "blocking"
                blocking_count += 1
            else:
                row["severity"] = "historical_finding"
                finding_count += 1
        else:
            row["severity"] = "none"
        records.append(row)
    core = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "project_root": str(project_root),
        "blocking_count": blocking_count,
        "historical_finding_count": finding_count,
        "ok": blocking_count == 0,
        "records": records,
    }
    return {**core, "audit_sha256": _sha256_bytes(_canonical_bytes(core))}


def _zip_prompt(source_zip: Path) -> tuple[str, bytes]:
    source_zip = source_zip.expanduser().resolve()
    if source_zip.is_symlink() or not source_zip.is_file():
        raise BridgeRecordError(f"source ZIP is not a safe regular file: {source_zip}")
    try:
        with zipfile.ZipFile(source_zip) as archive:
            infos = archive.infolist()
            if len(infos) != 1:
                raise BridgeRecordError("activation ZIP must contain exactly one member")
            info = infos[0]
            mode = (info.external_attr >> 16) & 0o170000
            if info.is_dir() or mode == stat.S_IFLNK:
                raise BridgeRecordError("activation ZIP member must be a regular Markdown file")
            if (
                Path(info.filename).name != info.filename
                or "\\" in info.filename
                or not info.filename.lower().endswith(".md")
            ):
                raise BridgeRecordError("activation ZIP member must be one root-level Markdown file")
            payload = archive.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, BridgeRecordError):
            raise
        raise BridgeRecordError(f"cannot inspect activation ZIP: {exc}") from exc
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BridgeRecordError("activation prompt must be UTF-8 Markdown") from exc
    if b"\x00" in payload:
        raise BridgeRecordError("activation prompt contains a NUL byte")
    return info.filename, payload


def _utc_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BridgeRecordError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise BridgeRecordError(f"{field} must be UTC")
    return parsed.astimezone(timezone.utc)


def _git_common_dir(path: Path) -> Path:
    top = Path(_git(path, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    raw = Path(_git(path, "rev-parse", "--git-common-dir").stdout.strip())
    return raw.resolve() if raw.is_absolute() else (top / raw).resolve()


def _validate_target_worktree(project_root: Path, target: Path, branch: str) -> str:
    target = target.expanduser().resolve()
    if not target.is_dir():
        raise BridgeRecordError(f"target worktree does not exist: {target}")
    project_git = _git_common_dir(project_root)
    target_git = _git_common_dir(target)
    if target_git != project_git:
        raise BridgeRecordError("target worktree belongs to a different Git repository")
    observed_branch = _git(target, "branch", "--show-current").stdout.strip()
    if observed_branch != branch:
        raise BridgeRecordError(
            f"target branch mismatch: expected {branch!r}, observed {observed_branch!r}"
        )
    return _git(target, "rev-parse", "HEAD").stdout.strip()


def build_activation_receipt(
    project_root: Path,
    *,
    source_zip: Path,
    extracted_prompt: Path,
    work_item: str,
    work_package_path: str,
    target_branch: str,
    target_worktree: Path,
    created_at_utc: str,
    operator_instruction_text: str,
    operator_instruction_timestamp: str,
    source_thread_url: str | None,
    active_goal_before: str | None,
    execution_policy: str,
    bridge_policy: str,
    mail_policy: str,
    trash_after_start: bool,
    activation_result: str,
    limitations: Sequence[str],
    supersedes_receipt: str | None = None,
) -> dict[str, Any]:
    project_root = _project_root(project_root)
    if not _SAFE_ID.fullmatch(work_item):
        raise BridgeRecordError("work_item must be one safe identifier")
    created = _utc_timestamp(created_at_utc, field="created_at_utc")
    _utc_timestamp(operator_instruction_timestamp, field="operator_instruction_timestamp")
    receipt_id = f"{created.strftime('%Y%m%dT%H%M%SZ')}_{work_item}"
    if supersedes_receipt == receipt_id:
        raise BridgeRecordError("activation receipt cannot supersede itself")
    if supersedes_receipt is not None:
        raise BridgeRecordError(
            "active receipt supersession requires a separately versioned transition helper"
        )
    member, prompt_payload = _zip_prompt(source_zip)
    extracted_prompt = extracted_prompt.expanduser().resolve()
    if extracted_prompt.is_symlink() or not extracted_prompt.is_file():
        raise BridgeRecordError("extracted prompt is not a safe regular file")
    extracted_payload = extracted_prompt.read_bytes()
    if extracted_payload != prompt_payload:
        raise BridgeRecordError("extracted prompt bytes do not match the ZIP member")
    work_package = _relative_file(
        project_root, work_package_path, field="work_package_path"
    )
    base_commit = _validate_target_worktree(project_root, target_worktree, target_branch)
    value: dict[str, Any] = {
        "schema_version": ACTIVATION_RECEIPT_VERSION,
        "receipt_id": receipt_id,
        "created_at_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "active",
        "source_path": str(source_zip.expanduser().resolve()),
        "source_sha256": _sha256_file(source_zip.expanduser().resolve()),
        "archive_type": "zip",
        "archive_listing": [member],
        "archive_member_count": 1,
        "extracted_prompt_path": str(extracted_prompt),
        "prompt_sha256": _sha256_bytes(prompt_payload),
        "prompt_media_type": "text/markdown",
        "source_thread_url": source_thread_url,
        "operator_instruction_text": operator_instruction_text,
        "operator_instruction_timestamp": operator_instruction_timestamp,
        "work_item": work_item,
        "work_package_path": work_package_path,
        "work_package_sha256": _sha256_file(work_package),
        "target_branch": target_branch,
        "target_worktree": str(target_worktree.expanduser().resolve()),
        "base_commit": base_commit,
        "active_goal_before": active_goal_before,
        "supersedes_receipt": None,
        "execution_policy": execution_policy,
        "bridge_policy": bridge_policy,
        "mail_policy": mail_policy,
        "trash_after_start": bool(trash_after_start),
        "activation_result": activation_result,
        "limitations": [str(item) for item in limitations],
    }
    _assert_no_secret_keys(value)
    errors = _validation_errors(project_root, ACTIVATION_RECEIPT_VERSION, value)
    if errors:
        raise BridgeRecordError(f"generated activation receipt is invalid: {errors}")
    return value


def write_activation_receipt(project_root: Path, output: Path, value: Mapping[str, Any]) -> None:
    project_root = _project_root(project_root)
    receipt_id = str(value.get("receipt_id") or "")
    if not _SAFE_ID.fullmatch(receipt_id):
        raise BridgeRecordError("receipt_id is not a safe filename component")
    expected_parent = (project_root / "bridge/prompt_receipts").resolve()
    output = output.expanduser().resolve()
    if output.parent != expected_parent or output.name != f"{receipt_id}.json":
        raise BridgeRecordError(
            f"activation output must be {expected_parent / (receipt_id + '.json')}"
        )
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise BridgeRecordError(f"refusing to overwrite activation receipt: {output}") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def _validate_hash_bound_path(
    project_root: Path,
    value: Mapping[str, Any],
    *,
    path_field: str,
    sha_field: str,
) -> None:
    path_value = value.get(path_field)
    sha_value = value.get(sha_field)
    if path_value is None and sha_value is None:
        return
    if not isinstance(path_value, str) or not isinstance(sha_value, str):
        raise BridgeRecordError(f"{path_field} and {sha_field} must be supplied together")
    path = Path(path_value).expanduser()
    if path_field in {"work_package_path", "activation_receipt_path"}:
        path = _relative_file(project_root, path_value, field=path_field)
    else:
        if not path.is_absolute() and ".." in path.parts:
            raise BridgeRecordError(f"{path_field} must not escape the project root")
        path = path.resolve() if path.is_absolute() else (project_root / path).resolve()
        if not Path(path_value).is_absolute():
            try:
                path.relative_to(project_root)
            except ValueError as exc:
                raise BridgeRecordError(f"{path_field} escapes the project root") from exc
        if path.is_symlink() or not path.is_file():
            raise BridgeRecordError(f"{path_field} is not a safe regular file")
    if not _SHA256.fullmatch(sha_value) or _sha256_file(path) != sha_value:
        raise BridgeRecordError(f"{sha_field} does not match {path_field}")


def render_status(project_root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    project_root = _project_root(project_root)
    if source.get("schema_version") != STATUS_SOURCE_SCHEMA_VERSION:
        raise BridgeRecordError("unsupported status source schema_version")
    status = source.get("status")
    if not isinstance(status, Mapping):
        raise BridgeRecordError("status source must contain one status object")
    value = {
        "schema_version": LATEST_STATUS_VERSION,
        "record_role": "non_authoritative_projection",
        **dict(status),
    }
    _assert_no_secret_keys(value)
    errors = _validation_errors(project_root, LATEST_STATUS_VERSION, value)
    if errors:
        raise BridgeRecordError(f"rendered latest status is invalid: {errors}")
    _validate_hash_bound_path(
        project_root,
        value,
        path_field="work_package_path",
        sha_field="work_package_sha256",
    )
    _validate_hash_bound_path(
        project_root,
        value,
        path_field="activation_receipt_path",
        sha_field="activation_receipt_sha256",
    )
    _validate_hash_bound_path(
        project_root,
        value,
        path_field="handoff_bundle_path",
        sha_field="handoff_bundle_sha256",
    )
    for field in ("report_path", "evidence_index_path"):
        path_value = value.get(field)
        if path_value is not None:
            _relative_file(project_root, str(path_value), field=field)
    for field in ("base_commit", "current_commit"):
        commit = value.get(field)
        if commit is not None:
            result = _git(project_root, "cat-file", "-e", f"{commit}^{{commit}}", check=False)
            if result.returncode != 0:
                raise BridgeRecordError(f"{field} is not a commit in the repository")
    return value


def write_status(output: Path, value: Mapping[str, Any]) -> None:
    output = output.expanduser().resolve()
    if not output.parent.is_dir() or output.is_symlink() or output.is_dir():
        raise BridgeRecordError(f"unsafe status output path: {output}")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _load_limitations(values: Sequence[str]) -> list[str]:
    return [value for value in values if value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="read and validate current bridge records")
    audit_parser.add_argument("--output", type=Path)

    create = subparsers.add_parser("create-activation", help="create one append-only activation receipt")
    create.add_argument("--source-zip", type=Path, required=True)
    create.add_argument("--extracted-prompt", type=Path, required=True)
    create.add_argument("--work-item", required=True)
    create.add_argument("--work-package-path", required=True)
    create.add_argument("--target-branch", required=True)
    create.add_argument("--target-worktree", type=Path, required=True)
    create.add_argument("--created-at-utc", required=True)
    create.add_argument("--operator-instruction-text", required=True)
    create.add_argument("--operator-instruction-timestamp", required=True)
    create.add_argument("--source-thread-url")
    create.add_argument("--active-goal-before")
    create.add_argument("--execution-policy", default="manual_phase2")
    create.add_argument("--bridge-policy", default="explicit_handoff_only")
    create.add_argument("--mail-policy", default="no_email")
    create.add_argument("--trash-after-start", action="store_true")
    create.add_argument("--activation-result", default="activated_by_eiel_bridge_records_v1")
    create.add_argument("--limitation", action="append", default=[])
    create.add_argument("--supersedes-receipt")
    create.add_argument("--output", type=Path, required=True)

    render = subparsers.add_parser("render-status", help="validate and render a mutable status projection")
    render.add_argument("--source", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project_root = _project_root(args.project_root)
        if args.command == "audit":
            result = audit_records(project_root)
            text = json.dumps(result, indent=2, sort_keys=True) + "\n"
            if args.output is None:
                sys.stdout.write(text)
            else:
                output = args.output.expanduser().resolve()
                if not output.parent.is_dir() or output.is_symlink():
                    raise BridgeRecordError(f"unsafe audit output path: {output}")
                output.write_text(text, encoding="utf-8")
            return 0 if result["ok"] else 1
        if args.command == "create-activation":
            value = build_activation_receipt(
                project_root,
                source_zip=args.source_zip,
                extracted_prompt=args.extracted_prompt,
                work_item=args.work_item,
                work_package_path=args.work_package_path,
                target_branch=args.target_branch,
                target_worktree=args.target_worktree,
                created_at_utc=args.created_at_utc,
                operator_instruction_text=args.operator_instruction_text,
                operator_instruction_timestamp=args.operator_instruction_timestamp,
                source_thread_url=args.source_thread_url,
                active_goal_before=args.active_goal_before,
                execution_policy=args.execution_policy,
                bridge_policy=args.bridge_policy,
                mail_policy=args.mail_policy,
                trash_after_start=args.trash_after_start,
                activation_result=args.activation_result,
                limitations=_load_limitations(args.limitation),
                supersedes_receipt=args.supersedes_receipt,
            )
            write_activation_receipt(project_root, args.output, value)
            sys.stdout.write(
                json.dumps(
                    {
                        "ok": True,
                        "path": str(args.output.expanduser().resolve()),
                        "receipt_id": value["receipt_id"],
                        "receipt_sha256": _sha256_file(args.output.expanduser().resolve()),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            return 0
        source = _read_object(args.source)
        value = render_status(project_root, source)
        write_status(args.output, value)
        sys.stdout.write(
            json.dumps(
                {
                    "ok": True,
                    "path": str(args.output.expanduser().resolve()),
                    "status_sha256": _sha256_file(args.output.expanduser().resolve()),
                },
                sort_keys=True,
            )
            + "\n"
        )
        return 0
    except (BridgeRecordError, OSError, ValueError) as exc:
        sys.stderr.write(f"eiel_bridge_records: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
