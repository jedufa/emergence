from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[5]
ENGINE_ROOT = REPO_ROOT / "experiments" / "projects" / "emergence" / "engine"
for required in (str(REPO_ROOT), str(ENGINE_ROOT)):
    while required in sys.path:
        sys.path.remove(required)
sys.path.insert(0, str(ENGINE_ROOT))
sys.path.insert(1, str(REPO_ROOT))

from simulation.chronicle.archive_history import (
    ARCHIVE_NAME,
    CHECKSUM_NAME,
    MANIFEST_NAME,
    ArchiveCollisionError,
    ArchiveValidationError,
    ChronicleHistoryCatalog,
    archive_path,
    create_completed_run_archive,
    read_archive_member,
    verify_archive,
)


def _world(root: Path, world_id: str, event: str) -> Path:
    chronicle = root / world_id
    chronicle.mkdir(parents=True)
    (chronicle / "formal_events.jsonl").write_text(event + "\n", encoding="utf-8")
    (chronicle / "chronicle_manifest.json").write_text(
        json.dumps({"world_id": world_id}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return chronicle


def _archive(tmp_path: Path):
    worlds_root = tmp_path / "input"
    worlds = {
        "world-b": _world(worlds_root, "world-b", '{"event":2}'),
        "world-a": _world(worlds_root, "world-a", '{"event":1}'),
    }
    return create_completed_run_archive(
        tmp_path / "history",
        campaign_id="campaign-a",
        run_id="run-1",
        worlds=worlds,
        metadata={"profile": "notable", "source_commit": "abc123"},
    )


def test_deterministic_multi_world_archive_checksum_and_direct_read(tmp_path: Path) -> None:
    first = _archive(tmp_path / "first")
    second = _archive(tmp_path / "second")
    assert first.sha256 == second.sha256
    assert first.archive_path.read_bytes() == second.archive_path.read_bytes()
    assert first.world_ids == ("world-a", "world-b")
    assert first.member_count == 5

    sidecar = first.archive_path.with_name(CHECKSUM_NAME)
    assert sidecar.read_text(encoding="ascii") == (
        f"{first.sha256}  {ARCHIVE_NAME}\n"
    )
    verified = verify_archive(first.archive_path, require_sidecar=True)
    assert verified == first
    assert read_archive_member(
        first.archive_path, "worlds/world-a/chronicle/formal_events.jsonl"
    ) == b'{"event":1}\n'
    assert not any(path.is_dir() for path in first.archive_path.parent.glob("*extract*"))

    manifest = json.loads(read_archive_member(first.archive_path, MANIFEST_NAME))
    assert manifest["campaign_id"] == "campaign-a"
    assert manifest["run_id"] == "run-1"
    assert manifest["world_ids"] == ["world-a", "world-b"]
    assert manifest["worlds"] == [
        {"world_id": "world-a"},
        {"world_id": "world-b"},
    ]
    assert [row["path"] for row in manifest["files"]] == sorted(
        row["path"] for row in manifest["files"]
    )


def test_shared_genome_objects_are_stored_once_at_run_level(tmp_path: Path) -> None:
    world = _world(tmp_path / "input", "world-a", "{}")
    objects = tmp_path / "objects"
    objects.mkdir()
    (objects / "abc123.json").write_text('{"genome":"abc123"}\n', encoding="utf-8")
    record = create_completed_run_archive(
        tmp_path / "history",
        campaign_id="campaign-a",
        run_id="run-1",
        worlds={"world-a": world},
        objects_root=objects,
    )
    assert read_archive_member(
        record.archive_path, "objects/genomes/abc123.json"
    ) == b'{"genome":"abc123"}\n'
    manifest = json.loads(read_archive_member(record.archive_path, MANIFEST_NAME))
    assert [row["path"] for row in manifest["files"]] == sorted(
        row["path"] for row in manifest["files"]
    )


def test_identical_archive_is_idempotent_and_collision_never_overwrites(tmp_path: Path) -> None:
    history = tmp_path / "history"
    world = _world(tmp_path / "input", "world-a", '{"event":1}')
    first = create_completed_run_archive(
        history,
        campaign_id="campaign-a",
        run_id="run-1",
        worlds={"world-a": world},
    )
    second = create_completed_run_archive(
        history,
        campaign_id="campaign-a",
        run_id="run-1",
        worlds={"world-a": world},
    )
    assert first.sha256 == second.sha256

    before = first.archive_path.read_bytes()
    (world / "formal_events.jsonl").write_text('{"event":2}\n', encoding="utf-8")
    with pytest.raises(ArchiveCollisionError):
        create_completed_run_archive(
            history,
            campaign_id="campaign-a",
            run_id="run-1",
            worlds={"world-a": world},
        )
    assert first.archive_path.read_bytes() == before


@pytest.mark.parametrize(
    "identifier", ["../escape", "/absolute", r"bad\\name", "", "has space", "a:b"]
)
def test_rejects_unsafe_identifiers(tmp_path: Path, identifier: str) -> None:
    world = _world(tmp_path / "input", "world-a", "{}")
    with pytest.raises(ArchiveValidationError):
        create_completed_run_archive(
            tmp_path / "history",
            campaign_id=identifier,
            run_id="run-1",
            worlds={"world-a": world},
        )


def test_rejects_source_symlinks_and_unsafe_zip_members(tmp_path: Path) -> None:
    world = _world(tmp_path / "input", "world-a", "{}")
    (world / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(ArchiveValidationError, match="symlink"):
        create_completed_run_archive(
            tmp_path / "history",
            campaign_id="campaign-a",
            run_id="run-1",
            worlds={"world-a": world},
        )

    unsafe = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../outside", b"bad")
    with pytest.raises(ArchiveValidationError, match="unsafe"):
        verify_archive(unsafe)
    with pytest.raises(ArchiveValidationError):
        read_archive_member(unsafe, "../outside", verify=False)


def test_rejects_archive_and_catalog_symlinks(tmp_path: Path) -> None:
    record = _archive(tmp_path / "valid")
    linked_archive = tmp_path / "linked.zip"
    linked_archive.symlink_to(record.archive_path)
    with pytest.raises(ArchiveValidationError, match="regular file"):
        verify_archive(linked_archive)

    catalog = ChronicleHistoryCatalog(tmp_path / "catalog-history")
    catalog.history_root.mkdir()
    catalog.path.symlink_to(tmp_path / "outside.sqlite3")
    with pytest.raises(ArchiveValidationError, match="symlink"):
        catalog.reconcile()


def test_rejects_zip_symlink_and_corrupt_or_uninventoried_members(tmp_path: Path) -> None:
    symlink_archive = tmp_path / "symlink.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(symlink_archive, "w") as archive:
        archive.writestr(link, "target")
    with pytest.raises(ArchiveValidationError, match="symlink"):
        verify_archive(symlink_archive)

    record = _archive(tmp_path / "valid")
    corrupt = tmp_path / "corrupt.zip"
    shutil.copyfile(record.archive_path, corrupt)
    payload = bytearray(corrupt.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    corrupt.write_bytes(payload)
    with pytest.raises(ArchiveValidationError):
        verify_archive(corrupt)

    extra = tmp_path / "extra.zip"
    shutil.copyfile(record.archive_path, extra)
    with zipfile.ZipFile(extra, "a") as archive:
        archive.writestr("untracked.txt", "not in manifest")
    with pytest.raises(ArchiveValidationError, match="inventory mismatch"):
        verify_archive(extra)


def test_member_reads_are_bounded_and_missing_member_is_clean_error(tmp_path: Path) -> None:
    record = _archive(tmp_path)
    member = "worlds/world-a/chronicle/formal_events.jsonl"
    with pytest.raises(ArchiveValidationError, match="read limit"):
        read_archive_member(record.archive_path, member, max_bytes=2)
    with pytest.raises(ArchiveValidationError, match="cannot read"):
        read_archive_member(record.archive_path, "worlds/world-a/missing.json")


def test_catalog_reconcile_rebuilds_and_removes_manually_deleted_archives(
    tmp_path: Path,
) -> None:
    first = _archive(tmp_path / "one")
    history = first.archive_path.parents[2]
    catalog = ChronicleHistoryCatalog(history)
    assert [(row.campaign_id, row.run_id) for row in catalog.list_records()] == [
        ("campaign-a", "run-1")
    ]

    second_world = _world(tmp_path / "second-input", "world-c", "{}")
    second = create_completed_run_archive(
        history,
        campaign_id="campaign-b",
        run_id="run-2",
        worlds={"world-c": second_world},
        update_catalog=False,
    )
    result = catalog.reconcile()
    assert result.indexed == 2
    assert result.invalid == ()

    first.archive_path.unlink()
    first.archive_path.with_name(CHECKSUM_NAME).unlink()
    result = catalog.reconcile()
    assert result.indexed == 1
    assert result.removed == 1
    assert [row.archive_path for row in catalog.list_records()] == [second.archive_path]


def test_catalog_reconcile_isolates_corrupt_archives(tmp_path: Path) -> None:
    record = _archive(tmp_path)
    history = record.archive_path.parents[2]
    corrupt_path = archive_path(history, "campaign-b", "run-b")
    corrupt_path.parent.mkdir(parents=True)
    corrupt_path.write_bytes(b"not a zip")
    corrupt_path.with_name(CHECKSUM_NAME).write_text(
        f"{hashlib.sha256(b'not a zip').hexdigest()}  {ARCHIVE_NAME}\n",
        encoding="ascii",
    )

    result = ChronicleHistoryCatalog(history).reconcile()
    assert result.indexed == 1
    assert len(result.invalid) == 1
    assert result.invalid[0][0] == str(corrupt_path)
