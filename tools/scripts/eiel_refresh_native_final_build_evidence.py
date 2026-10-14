#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence


FINAL_EVIDENCE_SCHEMA = "eiel_native_runtime_final_build_evidence_v1"
BUILD_MANIFEST_SCHEMA = "eiel_native_runtime_build_manifest_v1"
PYTHON_DIGEST_ALGORITHM = "sha256 of sorted NUL-delimited path and file-sha256 pairs"
DIRTY_DIGEST_ALGORITHM = (
    "sha256 of canonical component lines for porcelain-v1-z, git diff --binary HEAD, "
    "and sorted NUL-delimited untracked path and file-sha256 pairs"
)


class EvidenceError(RuntimeError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_content_sha256(path: Path) -> str:
    if path.is_symlink():
        return _sha256_bytes(os.fsencode(os.readlink(path)))
    if not path.is_file():
        raise EvidenceError(f"digest input is not a file: {path}")
    return _sha256_file(path)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must be a JSON object")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise EvidenceError(f"{name} must be a JSON array")
    return value


def _string(mapping: Mapping[str, Any], key: str, name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise EvidenceError(f"{name}.{key} must be a non-empty string")
    return value


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceError(f"{name} does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{name} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"{name} must contain a JSON object: {path}")
    return value


def _git(repo: Path, *args: str) -> bytes:
    env = os.environ.copy()
    env.update({"GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"})
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise EvidenceError(f"git {' '.join(args)} failed in {repo}: {detail}") from exc
    return completed.stdout


def _adjacent_manifest(extension: Path) -> Path:
    return extension.with_suffix(extension.suffix + ".manifest.json")


def _adjacent_evidence(extension: Path) -> Path:
    return extension.with_suffix(extension.suffix + ".build-evidence.json")


def _same_path(recorded: str, actual: Path) -> bool:
    return Path(recorded).expanduser().resolve() == actual.resolve()


def _verify_false(value: Any, name: str) -> None:
    if value is not False:
        raise EvidenceError(f"{name} must be false")


def _verify_true(value: Any, name: str) -> None:
    if value is not True:
        raise EvidenceError(f"{name} must be true")


def _verify_source_record(
    record: Mapping[str, Any],
    name: str,
    *,
    manifest_path: str | None = None,
    manifest_sha256: str | None = None,
) -> tuple[Path, str]:
    source_path = Path(_string(record, "path", name)).expanduser().resolve()
    recorded_sha256 = _string(record, "sha256", name)
    if manifest_path is not None and not _same_path(manifest_path, source_path):
        raise EvidenceError(f"{name} path disagrees with the build manifest")
    if manifest_sha256 is not None and manifest_sha256 != recorded_sha256:
        raise EvidenceError(f"{name} SHA-256 disagrees with the build manifest")
    if not source_path.is_file():
        raise EvidenceError(f"{name} source does not exist: {source_path}")
    actual_sha256 = _sha256_file(source_path)
    if actual_sha256 != recorded_sha256:
        raise EvidenceError(
            f"{name} source SHA-256 mismatch: expected {recorded_sha256}, got {actual_sha256}"
        )
    return source_path, actual_sha256


def verify_immutable_inputs(extension: Path, existing: Mapping[str, Any]) -> dict[str, Any]:
    extension = extension.expanduser().resolve()
    if not extension.is_file():
        raise EvidenceError(f"extension does not exist: {extension}")
    if existing.get("schema_version") != FINAL_EVIDENCE_SCHEMA:
        raise EvidenceError(f"existing evidence schema must be {FINAL_EVIDENCE_SCHEMA}")

    artifact = _mapping(existing.get("artifact"), "artifact")
    manifest_path = _adjacent_manifest(extension).resolve()
    recorded_manifest_path = _string(artifact, "tool_manifest_path", "artifact")
    if not _same_path(recorded_manifest_path, manifest_path):
        raise EvidenceError("artifact.tool_manifest_path is not adjacent to the requested extension")
    manifest = _load_json(manifest_path, "build manifest")
    if manifest.get("schema_version") != BUILD_MANIFEST_SCHEMA:
        raise EvidenceError(f"build manifest schema must be {BUILD_MANIFEST_SCHEMA}")

    extension_sha256 = _sha256_file(extension)
    recorded_extension_sha256 = _string(artifact, "sha256", "artifact")
    manifest_extension_sha256 = _string(manifest, "output_sha256", "build manifest")
    if len({extension_sha256, recorded_extension_sha256, manifest_extension_sha256}) != 1:
        raise EvidenceError(
            "extension SHA-256 mismatch between the file, build manifest, and existing evidence"
        )
    if not _same_path(_string(manifest, "output", "build manifest"), extension):
        raise EvidenceError("build manifest output path does not identify the requested extension")
    if not _same_path(_string(artifact, "path", "artifact"), extension):
        raise EvidenceError("existing evidence artifact path does not identify the requested extension")

    manifest_sha256 = _sha256_file(manifest_path)
    if manifest_sha256 != _string(artifact, "tool_manifest_sha256", "artifact"):
        raise EvidenceError("build manifest SHA-256 does not match existing evidence")

    abi_version = _string(manifest, "extension_abi_version", "build manifest")
    extension_version = _string(manifest, "extension_version", "build manifest")
    if abi_version != _string(artifact, "extension_abi_version", "artifact"):
        raise EvidenceError("extension ABI version disagrees with existing evidence")
    if extension_version != _string(artifact, "extension_version", "artifact"):
        raise EvidenceError("extension version disagrees with existing evidence")

    _verify_false(manifest.get("runtime_jit_in_measured_execution"), "build manifest runtime JIT")
    _verify_false(artifact.get("runtime_jit_in_measured_execution"), "artifact runtime JIT")
    load_validation = _mapping(existing.get("load_validation"), "load_validation")
    _verify_false(load_validation.get("runtime_jit"), "load_validation.runtime_jit")
    _verify_false(
        load_validation.get("runtime_jit_in_measured_execution"),
        "load_validation.runtime_jit_in_measured_execution",
    )
    _verify_true(load_validation.get("manifest_sha256_verified"), "load_validation.manifest_sha256_verified")
    _verify_true(load_validation.get("abi_compatible"), "load_validation.abi_compatible")
    _verify_true(
        load_validation.get("module_abi_version_verified"),
        "load_validation.module_abi_version_verified",
    )
    _verify_true(
        load_validation.get("module_extension_version_verified"),
        "load_validation.module_extension_version_verified",
    )

    source_hashes = _mapping(existing.get("source_hashes"), "source_hashes")
    resident_record = _mapping(source_hashes.get("resident_owner_cpp"), "source_hashes.resident_owner_cpp")
    resident_path, resident_sha256 = _verify_source_record(
        resident_record,
        "source_hashes.resident_owner_cpp",
        manifest_path=_string(manifest, "resident_owner_source", "build manifest"),
        manifest_sha256=_string(manifest, "resident_owner_source_sha256", "build manifest"),
    )

    binding_records = _sequence(source_hashes.get("binding_sources"), "source_hashes.binding_sources")
    if not binding_records:
        raise EvidenceError("source_hashes.binding_sources must not be empty")
    manifest_binding_path = _string(manifest, "source", "build manifest")
    manifest_binding_sha256 = _string(manifest, "source_sha256", "build manifest")
    verified_bindings: list[dict[str, str]] = []
    primary_binding_found = False
    for index, value in enumerate(binding_records):
        name = f"source_hashes.binding_sources[{index}]"
        record = _mapping(value, name)
        source_path = _string(record, "path", name)
        is_primary = _same_path(source_path, Path(manifest_binding_path))
        path, sha256 = _verify_source_record(
            record,
            name,
            manifest_path=manifest_binding_path if is_primary else None,
            manifest_sha256=manifest_binding_sha256 if is_primary else None,
        )
        primary_binding_found = primary_binding_found or is_primary
        verified_bindings.append({"path": str(path), "sha256": sha256})
    if not primary_binding_found:
        raise EvidenceError("build manifest binding source is absent from source_hashes.binding_sources")

    build = _mapping(existing.get("build"), "build")
    _string(build, "command", "build")
    _sequence(build.get("argv"), "build.argv")

    return {
        "extension": extension,
        "extension_sha256": extension_sha256,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha256,
        "abi_version": abi_version,
        "extension_version": extension_version,
        "resident_path": resident_path,
        "resident_sha256": resident_sha256,
        "primary_binding_path": Path(manifest_binding_path).expanduser().resolve(),
        "binding_sources": verified_bindings,
    }


def _git_scope(existing: Mapping[str, Any], source_path: Path) -> tuple[Path, Path, str]:
    git_root = Path(_git(source_path.parent, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    python_digest = _mapping(existing.get("python_source_digest"), "python_source_digest")
    scope_value = _string(python_digest, "scope", "python_source_digest")
    scope_path = Path(scope_value)
    project_root = (scope_path if scope_path.is_absolute() else git_root / scope_path).resolve()
    if not project_root.is_dir():
        raise EvidenceError(f"Python digest scope does not exist: {project_root}")
    try:
        source_path.relative_to(project_root)
    except ValueError as exc:
        raise EvidenceError("binding source is outside the recorded Python digest scope") from exc
    return git_root, project_root, scope_value


def _nul_records(payload: bytes) -> list[bytes]:
    return [record for record in payload.split(b"\0") if record]


def _porcelain_entry_count(payload: bytes) -> int:
    records = _nul_records(payload)
    count = 0
    index = 0
    while index < len(records):
        record = records[index]
        if len(record) < 3 or record[2:3] != b" ":
            raise EvidenceError(f"unsupported git porcelain-v1-z record: {record[:80]!r}")
        status = record[:2]
        count += 1
        index += 1
        if b"R" in status or b"C" in status:
            if index >= len(records):
                raise EvidenceError("truncated rename/copy record in git porcelain-v1-z output")
            index += 1
    return count


def _path_hash_manifest(scope: Path, raw_paths: Sequence[bytes]) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    for raw_path in sorted(set(raw_paths)):
        path = scope / os.fsdecode(raw_path)
        if not path.is_file() and not path.is_symlink():
            continue
        digest.update(raw_path)
        digest.update(b"\0")
        digest.update(_path_content_sha256(path).encode("ascii"))
        digest.update(b"\0")
        count += 1
    return count, digest.hexdigest()


def _capture_repository_state(project_root: Path, scope_value: str) -> dict[str, Any]:
    branch = _git(project_root, "branch", "--show-current").decode().strip()
    head = _git(project_root, "rev-parse", "HEAD").decode().strip()

    python_paths = _nul_records(
        _git(project_root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py")
    )
    python_file_count, python_sha256 = _path_hash_manifest(project_root, python_paths)

    status_payload = _git(
        project_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--",
        ".",
    )
    tracked_diff = _git(project_root, "diff", "--binary", "HEAD", "--", ".")
    untracked_paths = _nul_records(
        _git(project_root, "ls-files", "-z", "--others", "--exclude-standard", "--", ".")
    )
    untracked_file_count, untracked_sha256 = _path_hash_manifest(project_root, untracked_paths)
    status_sha256 = _sha256_bytes(status_payload)
    tracked_diff_sha256 = _sha256_bytes(tracked_diff)
    canonical_components = (
        f"status_porcelain_v1_z_sha256={status_sha256}\n"
        f"tracked_binary_diff_sha256={tracked_diff_sha256}\n"
        f"untracked_path_content_manifest_sha256={untracked_sha256}\n"
    ).encode("ascii")

    return {
        "python_source_digest": {
            "scope": scope_value,
            "selection": "git tracked and untracked non-ignored .py files",
            "file_count": python_file_count,
            "algorithm": PYTHON_DIGEST_ALGORITHM,
            "sha256": python_sha256,
            "stable_across_two_consecutive_passes": True,
        },
        "dirty_tree_digest": {
            "scope": scope_value,
            "branch": branch,
            "head": head,
            "porcelain_entry_count": _porcelain_entry_count(status_payload),
            "untracked_file_count": untracked_file_count,
            "algorithm": DIRTY_DIGEST_ALGORITHM,
            "status_porcelain_v1_z_sha256": status_sha256,
            "tracked_binary_diff_sha256": tracked_diff_sha256,
            "untracked_path_content_manifest_sha256": untracked_sha256,
            "sha256": _sha256_bytes(canonical_components),
            "stable_across_two_consecutive_passes": True,
        },
    }


def build_refreshed_evidence(
    extension: Path,
    existing: Mapping[str, Any],
    *,
    captured_at: str,
) -> dict[str, Any]:
    verified_before = verify_immutable_inputs(extension, existing)
    _, project_root, scope_value = _git_scope(existing, verified_before["primary_binding_path"])
    first_state = _capture_repository_state(project_root, scope_value)
    second_state = _capture_repository_state(project_root, scope_value)
    if first_state != second_state:
        raise EvidenceError("repository state changed while computing stable source and dirty-tree digests")
    verified_after = verify_immutable_inputs(extension, existing)
    if verified_before != verified_after:
        raise EvidenceError("immutable extension inputs changed while evidence was being refreshed")

    refreshed = copy.deepcopy(dict(existing))
    refreshed["captured_at"] = captured_at
    artifact = refreshed["artifact"]
    artifact.update(
        {
            "path": str(verified_before["extension"]),
            "sha256": verified_before["extension_sha256"],
            "tool_manifest_path": str(verified_before["manifest_path"]),
            "tool_manifest_sha256": verified_before["manifest_sha256"],
            "extension_abi_version": verified_before["abi_version"],
            "extension_version": verified_before["extension_version"],
            "runtime_jit_in_measured_execution": False,
        }
    )
    refreshed["python_source_digest"] = first_state["python_source_digest"]
    refreshed["dirty_tree_digest"] = first_state["dirty_tree_digest"]
    return refreshed


def _comparison_payload(evidence: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(evidence))
    payload.pop("captured_at", None)
    return payload


def _changed_top_level_fields(current: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    keys = (set(current) | set(expected)) - {"captured_at"}
    return sorted(key for key in keys if current.get(key) != expected.get(key))


def _atomic_write_json(path: Path, evidence: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    output_mode = path.stat().st_mode & 0o777 if path.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            os.fchmod(handle.fileno(), output_mode)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _captured_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify or refresh final evidence for an already-built immutable EIEL native extension."
    )
    parser.add_argument("--extension", required=True, type=Path)
    parser.add_argument("--output-evidence", type=Path)
    parser.add_argument("--check", action="store_true", help="verify freshness without writing")
    args = parser.parse_args(argv)

    extension = args.extension.expanduser().resolve()
    adjacent_evidence = _adjacent_evidence(extension).resolve()
    output_evidence = (
        args.output_evidence.expanduser().resolve() if args.output_evidence else adjacent_evidence
    )
    if args.check and not output_evidence.is_file():
        print(
            json.dumps(
                {"ok": False, "reason": "output_evidence_missing", "path": str(output_evidence)},
                sort_keys=True,
            )
        )
        return 1
    source_evidence = output_evidence if output_evidence.is_file() else adjacent_evidence

    try:
        existing = _load_json(source_evidence, "existing build evidence")
        captured_at = str(existing.get("captured_at", "")) if args.check else _captured_at()
        refreshed = build_refreshed_evidence(extension, existing, captured_at=captured_at)
        if args.check:
            if _comparison_payload(existing) != _comparison_payload(refreshed):
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "reason": "build_evidence_stale",
                            "changed_fields": _changed_top_level_fields(existing, refreshed),
                            "path": str(output_evidence),
                        },
                        sort_keys=True,
                    )
                )
                return 1
            print(json.dumps({"ok": True, "mode": "check", "path": str(output_evidence)}, sort_keys=True))
            return 0

        _atomic_write_json(output_evidence, refreshed)
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": "refresh",
                    "path": str(output_evidence),
                    "sha256": _sha256_file(output_evidence),
                },
                sort_keys=True,
            )
        )
        return 0
    except (EvidenceError, OSError) as exc:
        print(json.dumps({"ok": False, "reason": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
