from __future__ import annotations

"""Verified, deterministic, ZIP-backed history for completed Chronicles.

The ZIP is the authority.  The SQLite catalogue is deliberately rebuildable
from archive manifests and may be deleted without losing Chronicle data.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping
import zipfile
import zlib

from simulation.chronicle.schema import canonical_json, canonical_payload


ARCHIVE_SCHEMA_VERSION = "eiel_chronicle_archive_manifest_v1"
ARCHIVE_NAME = "chronicles.zip"
CHECKSUM_NAME = "chronicles.zip.sha256"
CATALOG_NAME = "catalog.sqlite3"
MANIFEST_NAME = "chronicle_archive_manifest.json"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
DEFAULT_MAX_MEMBER_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
DEFAULT_READ_LIMIT_BYTES = 16 * 1024 * 1024


class ChronicleArchiveError(RuntimeError):
    """Base class for Chronicle history failures."""


class ArchiveValidationError(ChronicleArchiveError):
    """An archive, source tree, identifier, or checksum is invalid."""


class ArchiveCollisionError(ChronicleArchiveError):
    """A different archive already occupies a campaign/run identity."""


@dataclass(frozen=True)
class ArchiveRecord:
    campaign_id: str
    run_id: str
    archive_path: Path
    sha256: str
    size_bytes: int
    world_ids: tuple[str, ...]
    member_count: int
    uncompressed_bytes: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ReconcileResult:
    indexed: int
    removed: int
    invalid: tuple[tuple[str, str], ...]


def _validate_identifier(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or value in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
    ):
        raise ArchiveValidationError(f"{label} must be a non-empty path segment")
    return value


def _validate_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ArchiveValidationError("archive member name is invalid")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveValidationError(f"unsafe archive member path: {name!r}")
    return path.as_posix()


def _sha256_stream(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        size += len(block)
        digest.update(block)
    return digest.hexdigest(), size


def _sha256_file(path: Path) -> tuple[str, int]:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _published_archive_matches(path: Path, expected_sha256: str) -> None:
    current_sha256, _ = _sha256_file(path)
    if current_sha256 != expected_sha256:
        raise ArchiveCollisionError(
            f"different Chronicle archive already exists: {path}"
        )


def _source_files(root: Path, archive_prefix: str) -> Iterator[tuple[str, Path]]:
    if root.is_symlink() or not root.is_dir():
        raise ArchiveValidationError(f"Chronicle source is not a real directory: {root}")
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        relative = path.relative_to(root)
        member_name = _validate_member_name(
            f"{archive_prefix}/{PurePosixPath(*relative.parts).as_posix()}"
        )
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ArchiveValidationError(f"symlinks are not allowed in Chronicle sources: {path}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ArchiveValidationError(f"non-regular Chronicle source is not allowed: {path}")
        yield member_name, path


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_validate_member_name(name), date_time=ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _write_file(
    archive: zipfile.ZipFile,
    *,
    member_name: str,
    source: Path,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as input_handle, archive.open(
        _zip_info(member_name), "w", force_zip64=True
    ) as output_handle:
        for block in iter(lambda: input_handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
            output_handle.write(block)
    return {
        "path": member_name,
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def _write_bytes(archive: zipfile.ZipFile, member_name: str, payload: bytes) -> None:
    with archive.open(_zip_info(member_name), "w", force_zip64=True) as handle:
        handle.write(payload)


def archive_path(history_root: Path, campaign_id: str, run_id: str) -> Path:
    """Return the canonical archive path after validating both identifiers."""

    return (
        Path(history_root)
        / _validate_identifier(campaign_id, "campaign_id")
        / _validate_identifier(run_id, "run_id")
        / ARCHIVE_NAME
    )


def create_completed_run_archive(
    history_root: Path,
    *,
    campaign_id: str,
    run_id: str,
    worlds: Mapping[str, Path],
    metadata: Mapping[str, Any] | None = None,
    objects_root: Path | None = None,
    update_catalog: bool = True,
) -> ArchiveRecord:
    """Create and publish one verified archive for a successfully completed run.

    ``worlds`` maps stable world IDs to finalized Chronicle directories.
    ``objects_root``, when present, is stored once at ``objects/genomes``.
    Existing byte-identical output is idempotent; a different archive raises
    :class:`ArchiveCollisionError` and is never overwritten.
    """

    campaign_id = _validate_identifier(campaign_id, "campaign_id")
    run_id = _validate_identifier(run_id, "run_id")
    if not worlds:
        raise ArchiveValidationError("at least one completed world is required")
    normalized_worlds: list[tuple[str, Path]] = []
    for world_id, root in sorted(worlds.items()):
        normalized_worlds.append(
            (_validate_identifier(world_id, "world_id"), Path(root))
        )
    if len({world_id for world_id, _ in normalized_worlds}) != len(normalized_worlds):
        raise ArchiveValidationError("world IDs must be unique")

    destination = archive_path(history_root, campaign_id, run_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    campaign_directory = Path(history_root) / campaign_id
    if campaign_directory.is_symlink() or destination.parent.is_symlink():
        raise ArchiveValidationError(
            "campaign and run archive directories must not be symlinks"
        )
    metadata_payload = canonical_payload(dict(metadata or {}))

    file_sources: list[tuple[str, Path]] = []
    for world_id, root in normalized_worlds:
        file_sources.extend(_source_files(root, f"worlds/{world_id}/chronicle"))
    if objects_root is not None:
        file_sources.extend(_source_files(Path(objects_root), "objects/genomes"))
    file_sources.sort(key=lambda row: row[0])
    member_names = [name for name, _ in file_sources]
    if len(member_names) != len(set(member_names)):
        raise ArchiveValidationError("duplicate archive member path")

    fd, temporary_name = tempfile.mkstemp(
        prefix=".chronicles.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary: Path | None = Path(temporary_name)
    try:
        inventory: list[dict[str, Any]] = []
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as archive:
            for member_name, source in file_sources:
                inventory.append(
                    _write_file(archive, member_name=member_name, source=source)
                )
            manifest = canonical_payload(
                {
                    "schema_version": ARCHIVE_SCHEMA_VERSION,
                    "campaign_id": campaign_id,
                    "run_id": run_id,
                    "worlds": [
                        {"world_id": world_id} for world_id, _ in normalized_worlds
                    ],
                    "world_ids": [world_id for world_id, _ in normalized_worlds],
                    "metadata": metadata_payload,
                    "files": inventory,
                }
            )
            _write_bytes(
                archive,
                MANIFEST_NAME,
                (canonical_json(manifest) + "\n").encode("utf-8"),
            )

        # Validate the complete temporary ZIP before it becomes authoritative.
        candidate = verify_archive(temporary)
        candidate_sha256 = candidate.sha256
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if destination.exists():
            _published_archive_matches(destination, candidate_sha256)
        else:
            try:
                # Hard-link publication is atomic and fails rather than
                # overwriting if another closeout wins the same identity.
                os.link(temporary, destination)
            except FileExistsError:
                _published_archive_matches(destination, candidate_sha256)
            else:
                temporary.unlink()
                temporary = None

        checksum_path = destination.with_name(CHECKSUM_NAME)
        checksum_text = f"{candidate_sha256}  {ARCHIVE_NAME}\n"
        checksum_fd, checksum_temporary_name = tempfile.mkstemp(
            prefix=f".{CHECKSUM_NAME}.", suffix=".tmp", dir=destination.parent
        )
        checksum_temporary = Path(checksum_temporary_name)
        try:
            with os.fdopen(checksum_fd, "w", encoding="ascii", newline="\n") as handle:
                handle.write(checksum_text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(checksum_temporary, checksum_path)
        finally:
            if checksum_temporary.exists():
                checksum_temporary.unlink()
        record = verify_archive(destination, require_sidecar=True)
        if update_catalog:
            catalog = ChronicleHistoryCatalog(Path(history_root))
            catalog.upsert(record)
        return record
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_manifest(archive: zipfile.ZipFile) -> dict[str, Any]:
    try:
        info = archive.getinfo(MANIFEST_NAME)
    except KeyError as exc:
        raise ArchiveValidationError("archive manifest is missing") from exc
    if info.file_size > DEFAULT_READ_LIMIT_BYTES:
        raise ArchiveValidationError("archive manifest is too large")
    try:
        payload = json.loads(archive.read(info))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveValidationError("archive manifest is invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != ARCHIVE_SCHEMA_VERSION:
        raise ArchiveValidationError("archive manifest schema is unsupported")
    return payload


def verify_archive(
    path: Path,
    *,
    require_sidecar: bool = False,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
) -> ArchiveRecord:
    """Validate paths, member hashes, manifest, ZIP CRCs, and optional sidecar."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ArchiveValidationError(f"Chronicle archive is not a regular file: {path}")
    try:
        archive_sha256, archive_size = _sha256_file(path)
    except OSError as exc:
        raise ArchiveValidationError(f"cannot read Chronicle archive: {path}") from exc

    if require_sidecar:
        sidecar = path.with_name(CHECKSUM_NAME)
        if sidecar.is_symlink():
            raise ArchiveValidationError("archive checksum sidecar must not be a symlink")
        try:
            fields = sidecar.read_text(encoding="ascii").strip().split()
        except OSError as exc:
            raise ArchiveValidationError("archive checksum sidecar is missing") from exc
        if len(fields) != 2 or fields[0] != archive_sha256 or fields[1] != ARCHIVE_NAME:
            raise ArchiveValidationError("archive checksum sidecar does not match")

    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names: set[str] = set()
            total_bytes = 0
            for info in infos:
                name = _validate_member_name(info.filename)
                if name in names:
                    raise ArchiveValidationError(f"duplicate archive member: {name}")
                names.add(name)
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ArchiveValidationError(f"archive symlink is not allowed: {name}")
                if info.flag_bits & 0x1:
                    raise ArchiveValidationError(f"encrypted archive member is not allowed: {name}")
                if info.file_size > max_member_bytes:
                    raise ArchiveValidationError(f"archive member exceeds size limit: {name}")
                total_bytes += info.file_size
                if total_bytes > max_total_bytes:
                    raise ArchiveValidationError("archive exceeds total uncompressed size limit")
            manifest = _load_manifest(archive)
            campaign_id = _validate_identifier(manifest.get("campaign_id"), "campaign_id")
            run_id = _validate_identifier(manifest.get("run_id"), "run_id")
            raw_world_ids = manifest.get("world_ids")
            if not isinstance(raw_world_ids, list) or not raw_world_ids:
                raise ArchiveValidationError("archive manifest world_ids is invalid")
            world_ids = tuple(
                _validate_identifier(world_id, "world_id") for world_id in raw_world_ids
            )
            if len(world_ids) != len(set(world_ids)):
                raise ArchiveValidationError("archive manifest has duplicate world IDs")
            files = manifest.get("files")
            if not isinstance(files, list):
                raise ArchiveValidationError("archive manifest files is invalid")
            expected_names = {MANIFEST_NAME}
            for row in files:
                if not isinstance(row, dict):
                    raise ArchiveValidationError("archive manifest file row is invalid")
                name = _validate_member_name(row.get("path"))
                if name in expected_names:
                    raise ArchiveValidationError(f"duplicate manifest file row: {name}")
                expected_names.add(name)
                try:
                    expected_size = int(row["bytes"])
                    expected_sha256 = str(row["sha256"])
                    info = archive.getinfo(name)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ArchiveValidationError(f"invalid manifest row: {name}") from exc
                with archive.open(info, "r") as handle:
                    actual_sha256, actual_size = _sha256_stream(handle)
                if actual_size != expected_size or actual_sha256 != expected_sha256:
                    raise ArchiveValidationError(f"archive member hash mismatch: {name}")
            if names != expected_names:
                extras = sorted(names - expected_names)
                missing = sorted(expected_names - names)
                raise ArchiveValidationError(
                    f"archive inventory mismatch; extras={extras}, missing={missing}"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ArchiveValidationError(f"archive CRC failure: {bad_member}")
            metadata = manifest.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ArchiveValidationError("archive manifest metadata is invalid")
    except ArchiveValidationError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, zlib.error) as exc:
        raise ArchiveValidationError(f"invalid Chronicle archive: {path}") from exc

    return ArchiveRecord(
        campaign_id=campaign_id,
        run_id=run_id,
        archive_path=path,
        sha256=archive_sha256,
        size_bytes=archive_size,
        world_ids=world_ids,
        member_count=len(names),
        uncompressed_bytes=total_bytes,
        metadata=canonical_payload(metadata),
    )


def read_archive_member(
    path: Path,
    member_name: str,
    *,
    max_bytes: int = DEFAULT_READ_LIMIT_BYTES,
    verify: bool = True,
) -> bytes:
    """Read one bounded member without extracting the archive."""

    member_name = _validate_member_name(member_name)
    if verify:
        verify_archive(path)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            info = archive.getinfo(member_name)
            if info.file_size > max_bytes:
                raise ArchiveValidationError(
                    f"archive member exceeds read limit ({max_bytes} bytes): {member_name}"
                )
            with archive.open(info, "r") as handle:
                payload = handle.read(max_bytes + 1)
            if len(payload) > max_bytes:
                raise ArchiveValidationError(
                    f"archive member exceeds read limit ({max_bytes} bytes): {member_name}"
                )
            return payload
    except ArchiveValidationError:
        raise
    except (KeyError, OSError, zipfile.BadZipFile, RuntimeError, zlib.error) as exc:
        raise ArchiveValidationError(
            f"cannot read archive member: {member_name}"
        ) from exc


class ChronicleHistoryCatalog:
    """Rebuildable SQLite index for Chronicle history archives."""

    def __init__(self, history_root: Path):
        self.history_root = Path(history_root)
        self.path = self.history_root / CATALOG_NAME

    def _connect(self) -> sqlite3.Connection:
        self.history_root.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            raise ArchiveValidationError("Chronicle catalogue must not be a symlink")
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS archives (
                campaign_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                world_ids_json TEXT NOT NULL,
                member_count INTEGER NOT NULL,
                uncompressed_bytes INTEGER NOT NULL,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (campaign_id, run_id)
            )
            """
        )
        return connection

    def upsert(self, record: ArchiveRecord) -> None:
        try:
            relative_path = record.archive_path.resolve().relative_to(
                self.history_root.resolve()
            )
        except ValueError as exc:
            raise ArchiveValidationError("archive is outside the history root") from exc
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO archives (
                    campaign_id, run_id, relative_path, sha256, size_bytes,
                    world_ids_json, member_count, uncompressed_bytes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(campaign_id, run_id) DO UPDATE SET
                    relative_path=excluded.relative_path,
                    sha256=excluded.sha256,
                    size_bytes=excluded.size_bytes,
                    world_ids_json=excluded.world_ids_json,
                    member_count=excluded.member_count,
                    uncompressed_bytes=excluded.uncompressed_bytes,
                    metadata_json=excluded.metadata_json
                """,
                (
                    record.campaign_id,
                    record.run_id,
                    relative_path.as_posix(),
                    record.sha256,
                    record.size_bytes,
                    canonical_json(list(record.world_ids)),
                    record.member_count,
                    record.uncompressed_bytes,
                    canonical_json(record.metadata),
                ),
            )

    def list_records(self) -> list[ArchiveRecord]:
        if not self.path.exists():
            return []
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM archives ORDER BY campaign_id, run_id"
            ).fetchall()
        return [
            ArchiveRecord(
                campaign_id=row["campaign_id"],
                run_id=row["run_id"],
                archive_path=self.history_root / row["relative_path"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
                world_ids=tuple(json.loads(row["world_ids_json"])),
                member_count=row["member_count"],
                uncompressed_bytes=row["uncompressed_bytes"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def reconcile(self) -> ReconcileResult:
        """Replace catalogue rows with all currently valid filesystem archives."""

        records: list[ArchiveRecord] = []
        invalid: list[tuple[str, str]] = []
        for path in sorted(self.history_root.glob(f"*/*/{ARCHIVE_NAME}")):
            try:
                relative = path.relative_to(self.history_root)
                campaign_directory = self.history_root / relative.parts[0]
                run_directory = campaign_directory / relative.parts[1]
                if (
                    campaign_directory.is_symlink()
                    or run_directory.is_symlink()
                    or path.is_symlink()
                ):
                    raise ArchiveValidationError(
                        "archive path contains a symlink below the history root"
                    )
                record = verify_archive(path, require_sidecar=True)
                expected = archive_path(
                    self.history_root, record.campaign_id, record.run_id
                )
                if path.resolve() != expected.resolve():
                    raise ArchiveValidationError(
                        "archive path does not match manifest campaign/run identity"
                    )
                records.append(record)
            except ChronicleArchiveError as exc:
                invalid.append((str(path), str(exc)))

        with self._connect() as connection:
            previous_keys = {
                (row[0], row[1])
                for row in connection.execute(
                    "SELECT campaign_id, run_id FROM archives"
                ).fetchall()
            }
            connection.execute("DELETE FROM archives")
            for record in records:
                relative_path = record.archive_path.relative_to(self.history_root)
                connection.execute(
                    """
                    INSERT INTO archives VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.campaign_id,
                        record.run_id,
                        relative_path.as_posix(),
                        record.sha256,
                        record.size_bytes,
                        canonical_json(list(record.world_ids)),
                        record.member_count,
                        record.uncompressed_bytes,
                        canonical_json(record.metadata),
                    ),
                )
        current_keys = {(record.campaign_id, record.run_id) for record in records}
        removed = len(previous_keys - current_keys)
        return ReconcileResult(
            indexed=len(records), removed=removed, invalid=tuple(invalid)
        )
