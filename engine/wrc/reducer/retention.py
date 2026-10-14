"""Fail-closed, hash-verified retention transactions for reducer artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_sha256, pretty_json


RETENTION_SCHEMA_VERSION = "eiel_wrc_retention_manifest_v1"
RETAIN_ALL = "retain_all"
RETAIN_COMPACT = "retain_compact"
RETAIN_SELECTED = "retain_selected"
RETENTION_MODES = (RETAIN_ALL, RETAIN_COMPACT, RETAIN_SELECTED)
_WORLD_COUNT_FIELDS = (
    "complete_world_count",
    "world_count",
    "worlds_complete",
    "worlds_expected",
    "expected_world_count",
)
_UNSAFE_COUNT_FIELDS = (
    "incomplete_world_count",
    "failed_world_count",
    "anomalous_world_count",
    "integrity_failure_count",
    "unverified_world_count",
    "contract_required_world_count",
)
_UNSAFE_STATUS_WORDS = {
    "anomalous",
    "contract_required",
    "failed",
    "incomplete",
    "integrity_failure",
    "unverified",
}


class RetentionError(ValueError):
    """Retention could not be completed safely; raw input remains untouched."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise RetentionError(f"retention source is not a directory: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file())


def _inventory(root: Path, files: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(files):
        try:
            relative = path.relative_to(root).as_posix()
            rows.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        except (OSError, ValueError) as exc:
            raise RetentionError(f"cannot inventory retention artifact {path}") from exc
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(pretty_json(value), encoding="utf-8")
    temporary.replace(path)


def _selected_source_files(
    source_root: Path,
    *,
    mode: str,
    selected_paths: Iterable[str] = (),
) -> list[Path]:
    all_files = _files(source_root)
    if mode == RETAIN_ALL:
        return all_files
    requested = {str(value).strip().lstrip("/") for value in selected_paths if str(value).strip()}
    if mode == RETAIN_SELECTED and not requested:
        raise RetentionError("retain_selected requires selected paths")
    if mode == RETAIN_COMPACT:
        # Compact output is an explicit artifact boundary.  Keep manifests,
        # summaries, indexes, and provenance; raw trajectory files are copied
        # only when explicitly selected.
        compact_suffixes = {".json", ".jsonl", ".csv", ".md"}
        requested.update(
            path.relative_to(source_root).as_posix()
            for path in all_files
            if path.suffix.lower() in compact_suffixes
            and "raw" not in path.name.lower()
            and "metrics" not in path.name.lower()
        )
    selected = []
    for path in all_files:
        relative = path.relative_to(source_root).as_posix()
        if relative in requested:
            selected.append(path)
    missing = sorted(requested - {path.relative_to(source_root).as_posix() for path in selected})
    if missing:
        raise RetentionError(f"selected retention artifacts are missing: {missing}")
    return selected


def _manifest_documents(source_root: Path, files: Iterable[Path]) -> list[Mapping[str, Any]]:
    documents: list[Mapping[str, Any]] = []
    for path in files:
        if path.suffix.lower() != ".json" or "manifest" not in path.name.lower():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RetentionError(f"cannot verify retention manifest source {path}") from exc
        if not isinstance(value, Mapping):
            raise RetentionError(f"retention manifest source is not an object: {path}")
        documents.append(value)
    return documents


def _walk_manifest(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).strip().lower()
            yield name, child
            yield from _walk_manifest(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_manifest(child)


def _verified_world_count(
    source_root: Path,
    files: Iterable[Path],
    *,
    expected_world_count: int | None,
) -> None:
    documents = _manifest_documents(source_root, files)
    observed: list[int] = []
    for document in documents:
        for name, value in _walk_manifest(document):
            if name in _WORLD_COUNT_FIELDS and isinstance(value, int) and not isinstance(value, bool):
                observed.append(int(value))
    if expected_world_count is None:
        return
    if not observed:
        raise RetentionError("expected world count requires a verifiable reduction manifest")
    if any(value != int(expected_world_count) for value in observed):
        raise RetentionError("retention world count differs from expected count")


def _verify_prune_safety(source_root: Path, files: Iterable[Path]) -> None:
    documents = _manifest_documents(source_root, files)
    if not documents:
        raise RetentionError("raw pruning requires a verifiable reduction manifest")
    for document in documents:
        for name, value in _walk_manifest(document):
            if name in _UNSAFE_COUNT_FIELDS:
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                    raise RetentionError("raw pruning is unsafe for incomplete or unverified worlds")
                if isinstance(value, (list, tuple, set)) and value:
                    raise RetentionError("raw pruning is unsafe for incomplete or unverified worlds")
            if name in {"status", "world_status", "completion_status"}:
                if str(value).strip().lower() in _UNSAFE_STATUS_WORDS:
                    raise RetentionError("raw pruning is unsafe for incomplete or unverified worlds")


@dataclass(frozen=True, slots=True)
class RetentionManifest:
    payload_value: Mapping[str, Any]

    @property
    def payload(self) -> dict[str, Any]:
        return dict(self.payload_value)

    @property
    def manifest_sha256(self) -> str:
        return str(self.payload_value["manifest_sha256"])


def retain(
    source_root: str | Path,
    retained_root: str | Path,
    *,
    mode: str = RETAIN_ALL,
    selected_paths: Iterable[str] = (),
    expected_count: int | None = None,
    expected_world_count: int | None = None,
    prune_raw: bool = False,
    raw_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Materialize a retention transaction and optionally prune verified raw files.

    The source is never modified until all destination hashes, counts, and the
    complete manifest have been written and re-read successfully.  Any error
    before that point raises ``RetentionError`` and leaves the source intact.
    """

    if mode not in RETENTION_MODES:
        raise RetentionError(f"unsupported retention mode: {mode}")
    source = Path(source_root).resolve()
    destination = Path(retained_root).resolve()
    if source == destination:
        raise RetentionError("retention source and destination must differ")
    nested = False
    try:
        destination.relative_to(source)
        nested = True
    except ValueError:
        pass
    try:
        source.relative_to(destination)
        nested = True
    except ValueError:
        pass
    if nested:
        raise RetentionError("retention source and destination must not contain one another")
    selected = _selected_source_files(source, mode=mode, selected_paths=selected_paths)
    source_files = _files(source)
    if expected_count is not None and len(source_files) != int(expected_count):
        raise RetentionError("retention source count differs from expected count")
    _verified_world_count(
        source,
        source_files,
        expected_world_count=expected_world_count,
    )
    if prune_raw:
        _verify_prune_safety(source, source_files)
    destination.mkdir(parents=True, exist_ok=True)
    existing_destination = [path for path in destination.iterdir()]
    if existing_destination:
        raise RetentionError("retention destination must be empty")

    staged = destination / ".retention-staging"
    if staged.exists():
        raise RetentionError("incomplete prior retention transaction is present")
    staged.mkdir(parents=True)
    try:
        for source_file in selected:
            relative = source_file.relative_to(source)
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
        retained_files = _files(staged)
        source_inventory = _inventory(source, source_files)
        retained_inventory = _inventory(staged, retained_files)
        if len(retained_inventory) != len(selected):
            raise RetentionError("retained artifact count verification failed")
        by_source = {row["path"]: row for row in source_inventory}
        for row in retained_inventory:
            original = by_source.get(row["path"])
            if original is None or original["sha256"] != row["sha256"] or original["bytes"] != row["bytes"]:
                raise RetentionError("retained artifact hash verification failed")
        manifest: dict[str, Any] = {
            "schema_version": RETENTION_SCHEMA_VERSION,
            "mode": mode,
            "source_artifacts": source_inventory,
            "retained_artifacts": retained_inventory,
            "source_count": len(source_inventory),
            "retained_count": len(retained_inventory),
            "source_bytes": sum(int(row["bytes"]) for row in source_inventory),
            "retained_bytes": sum(int(row["bytes"]) for row in retained_inventory),
            "expected_world_count": expected_world_count,
            "raw_prune_requested": bool(prune_raw),
            "raw_prune_complete": False,
            "complete": True,
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _write_json(staged / "retention_manifest.json", manifest)
        verified = json.loads((staged / "retention_manifest.json").read_text(encoding="utf-8"))
        if not isinstance(verified, Mapping) or verified.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise RetentionError("retention manifest verification failed")
        final_manifest = destination / "retention_manifest.json"
        for path in sorted(staged.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(staged)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            path.replace(target)
        shutil.rmtree(staged)

        if prune_raw:
            requested_raw = {str(value).lstrip("/") for value in raw_paths if str(value)}
            if not requested_raw:
                raise RetentionError("raw pruning requires explicit raw_paths")
            selected_names = {row["path"] for row in source_inventory}
            if not requested_raw.issubset(selected_names):
                raise RetentionError("raw pruning includes an unverified artifact")
            for relative in sorted(requested_raw):
                path = source / relative
                if not path.is_file() or sha256_file(path) != next(row["sha256"] for row in source_inventory if row["path"] == relative):
                    raise RetentionError("raw artifact changed before pruning")
            backup = destination / ".raw-prune-backup"
            try:
                for relative in sorted(requested_raw):
                    backup_target = backup / relative
                    backup_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source / relative, backup_target)
                for relative in sorted(requested_raw):
                    (source / relative).unlink()
            except Exception as exc:
                for backup_file in sorted(backup.rglob("*")) if backup.exists() else ():
                    if backup_file.is_file():
                        relative = backup_file.relative_to(backup)
                        restore = source / relative
                        restore.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(backup_file, restore)
                if backup.exists():
                    shutil.rmtree(backup)
                raise RetentionError("raw pruning failed; raw artifacts were restored") from exc
            shutil.rmtree(backup)
            manifest["raw_prune_complete"] = True
            manifest["manifest_sha256"] = canonical_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
            _write_json(final_manifest, manifest)
        return manifest
    except Exception:
        if staged.exists():
            shutil.rmtree(staged)
        raise


def retain_all(source_root: str | Path, retained_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    return retain(source_root, retained_root, mode=RETAIN_ALL, **kwargs)


def retain_compact(source_root: str | Path, retained_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    return retain(source_root, retained_root, mode=RETAIN_COMPACT, **kwargs)


def retain_selected(source_root: str | Path, retained_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    return retain(source_root, retained_root, mode=RETAIN_SELECTED, **kwargs)


__all__ = [
    "RETAIN_ALL",
    "RETAIN_COMPACT",
    "RETAIN_SELECTED",
    "RETENTION_MODES",
    "RETENTION_SCHEMA_VERSION",
    "RetentionError",
    "RetentionManifest",
    "retain",
    "retain_all",
    "retain_compact",
    "retain_selected",
    "sha256_file",
]
