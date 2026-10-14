from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path

import pytest

from engine.campaigns import canopy_supervision_v1 as canopy
from engine.campaigns import artifact_lifecycle
from engine.campaigns import report_only_closeout as module
from engine.campaigns import root_retirement as retirement


REPORT = "# Run\n\n## Inputs\nSeed 42; fresh world.\n## Settings\n100 absolute ticks.\n## Results\nNo persistence.\n## Limitations\nOne seed only.\n## Reproduction\nRecreate the fresh world with seed 42 and run 100 ticks.\n"
REAL_PROCESS_REFERENCES = retirement._process_references


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    runtime = tmp_path / "causyl-runtime"
    root = runtime / "runs" / "one"
    root.mkdir(parents=True)
    (root / "checkpoint").write_bytes(b"temporary checkpoint")
    sibling = root.parent / "unrelated"
    sibling.mkdir()
    (sibling / "keep").write_text("unrelated")
    monkeypatch.setattr(retirement, "CANONICAL_RUNTIME_ROOTS", frozenset({runtime}))
    monkeypatch.setattr(retirement, "CANONICAL_MANAGEMENT_ROOTS", frozenset({runtime / "lifecycle"}))
    monkeypatch.setattr(retirement, "_process_references", lambda root: ([], []))
    return dict(report_path=tmp_path / "reports" / "run.md", report_markdown=REPORT,
                roots=[root], source_commit="a" * 40, run_id="test-run")


def test_only_report_remains_and_replay_is_idempotent(fixture):
    result = module.closeout_report_only_run_v1(**fixture, apply=True)
    assert result["status"] == "COMPLETED"
    assert not fixture["roots"][0].exists()
    assert list(fixture["report_path"].parent.iterdir()) == [fixture["report_path"]]
    assert (fixture["roots"][0].parent / "unrelated" / "keep").read_text() == "unrelated"
    assert module.closeout_report_only_run_v1(**fixture, apply=True) == result
    assert module.read_report_markdown_v1(fixture["report_path"]) == REPORT


def test_report_reader_rejects_modified_completed_report(fixture):
    module.closeout_report_only_run_v1(**fixture, apply=True)
    path = fixture["report_path"]
    path.write_text(path.read_text().replace("Seed 42", "Seed 99"))
    with pytest.raises(retirement.RootRetirementError, match="checksum"):
        module.read_report_markdown_v1(path)


def test_dry_run_never_reseals_changed_root(fixture):
    result = module.closeout_report_only_run_v1(**fixture)
    assert result["status"] == "CLEANUP_PENDING"
    (fixture["roots"][0] / "new").write_text("unexpected")
    with pytest.raises(retirement.RootRetirementError, match="changed"):
        module.closeout_report_only_run_v1(**fixture, apply=True)
    assert fixture["roots"][0].exists()


def test_report_failure_precedes_any_deletion(fixture, monkeypatch):
    def fail(*args):
        raise OSError("publication failed")
    monkeypatch.setattr(module, "_write", fail)
    with pytest.raises(OSError, match="publication failed"):
        module.closeout_report_only_run_v1(**fixture, apply=True)
    assert (fixture["roots"][0] / "checkpoint").exists()


def test_active_reference_protects_root(fixture, monkeypatch):
    monkeypatch.setattr(retirement, "_process_references", lambda root: ([{"pid": 123}], []))
    with pytest.raises(retirement.RootRetirementError, match="live process"):
        module.closeout_report_only_run_v1(**fixture, apply=True)
    assert fixture["roots"][0].exists()


def test_interruption_after_apply_resumes_from_exact_journal(fixture, monkeypatch):
    original = module._write
    def fail_footer(path, text):
        if module._MARKER in text:
            raise OSError("interrupted before report receipt")
        return original(path, text)
    monkeypatch.setattr(module, "_write", fail_footer)
    with pytest.raises(OSError, match="interrupted"):
        module.closeout_report_only_run_v1(**fixture, apply=True)
    assert not fixture["roots"][0].exists()
    monkeypatch.setattr(module, "_write", original)
    assert module.closeout_report_only_run_v1(**fixture, apply=True)["status"] == "COMPLETED"
    assert list(fixture["report_path"].parent.iterdir()) == [fixture["report_path"]]


def test_invalid_report_and_symlink_fail_closed(fixture):
    with pytest.raises(retirement.RootRetirementError, match="sections"):
        module.closeout_report_only_run_v1(**{**fixture, "report_markdown": "# Empty"}, apply=True)
    root = fixture["roots"][0]
    (root / "escape").symlink_to(root.parent / "unrelated")
    with pytest.raises(retirement.RootRetirementError):
        module.closeout_report_only_run_v1(**fixture, apply=True)
    assert root.exists()


def test_reappeared_root_not_deleted_by_completed_replay(fixture):
    module.closeout_report_only_run_v1(**fixture, apply=True)
    fixture["roots"][0].mkdir()
    with pytest.raises(retirement.RootRetirementError, match="reappeared"):
        module.closeout_report_only_run_v1(**fixture, apply=True)
    assert fixture["roots"][0].exists()


def test_interruption_after_staging_resumes_without_resealing(fixture, monkeypatch):
    original = retirement._delete_inventory_subset_fd
    def fail(*args, **kwargs):
        raise OSError("interrupted staged deletion")
    monkeypatch.setattr(retirement, "_delete_inventory_subset_fd", fail)
    with pytest.raises(OSError, match="interrupted staged"):
        module.closeout_report_only_run_v1(**fixture, apply=True)
    assert not fixture["roots"][0].exists()
    assert list(fixture["roots"][0].parent.glob(".eiel-retirement-*"))
    monkeypatch.setattr(retirement, "_delete_inventory_subset_fd", original)
    assert module.closeout_report_only_run_v1(**fixture, apply=True)["status"] == "COMPLETED"
    assert not list(fixture["roots"][0].parent.glob(".eiel-retirement-*"))


def test_concurrent_closeouts_are_serialized(fixture):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(module.closeout_report_only_run_v1, **fixture, apply=True) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    assert results[0] == results[1]
    assert list(fixture["report_path"].parent.iterdir()) == [fixture["report_path"]]


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _terminal_relay(fixture, monkeypatch, *, second_source=False, retarget_second=False):
    source_root = fixture["roots"][0].resolve()
    runtime = source_root.parents[1]
    relay_base = runtime / "live" / "canopy-relays"
    group_id, cell_id = "one-terminal-group", "cell-a"
    relay_root = relay_base / group_id / cell_id
    view = relay_root / "_work" / "source_view"
    source_files = [source_root / "checkpoint"]
    if second_source:
        second = source_root / "events" / "segment.json"
        second.parent.mkdir()
        second.write_bytes(b"source event segment")
        source_files.append(second)
    rows = [{"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in source_files]
    source_view_paths = [view / path.relative_to(source_root) for path in source_files]
    for index, (view_path, source_path) in enumerate(zip(source_view_paths, source_files)):
        view_path.parent.mkdir(parents=True, exist_ok=True)
        target = source_path.resolve()
        if retarget_second and index == 1:
            outside = fixture["report_path"].parent / "outside-source.txt"
            outside.parent.mkdir(parents=True, exist_ok=True)
            outside.write_text("must remain unchanged")
            target = outside
        view_path.symlink_to(target)

    playback = {"schema_version": "llc_lineage_playback_v1", "metadata": {"source_cell_id": cell_id}, "agents": []}
    provenance = {
        "schema_version": "eiel_lineage_playback_export_provenance_v1",
        "source_root": str(source_root),
        "source_root_resolved": str(source_root),
        "source_cell_id": cell_id,
        "source_files": rows,
    }
    cross_check = {"ok": True}
    payloads = {
        "playback_snapshot.json": playback,
        "provenance.json": provenance,
        "cross_check.json": cross_check,
    }
    hashes = {}
    for name, value in payloads.items():
        payload = _json_bytes(value)
        (relay_root / name).write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    relay_id = f"manual-{cell_id}"
    manifest = {
        "schema_version": "llc_near_live_playback_relay_v0_manifest",
        "relay_id": relay_id,
        "source_run_id": source_root.name,
        "source_root": str(source_root),
        "source_state": "source_complete",
        "sequence_number": 1,
        "playback_file_name": "playback_snapshot.json",
        "playback_sha256": hashes["playback_snapshot.json"],
        "provenance_file_name": "provenance.json",
        "provenance_sha256": hashes["provenance.json"],
        "cross_check_file_name": "cross_check.json",
        "cross_check_sha256": hashes["cross_check.json"],
        "status_file_name": "relay_status.json",
    }
    (relay_root / "relay_manifest.json").write_bytes(_json_bytes(manifest))
    (relay_root / "relay_status.json").write_bytes(_json_bytes({
        "schema_version": "llc_near_live_playback_relay_v0_status",
        "relay_id": relay_id,
        "state": "source_complete",
        "relay_sequence": 1,
    }))
    monkeypatch.setattr(canopy, "DEFAULT_CANOPY_RELAY_ROOT", relay_base)
    fixture["roots"].append(relay_root)
    return relay_root, source_files, source_view_paths


def test_report_closeout_removes_only_terminal_relay_source_view_links(fixture, monkeypatch):
    relay_root, source_files, source_view_paths = _terminal_relay(fixture, monkeypatch)
    originals = [path.read_bytes() for path in source_files]

    result = module.closeout_report_only_run_v1(**fixture)

    assert result["status"] == "CLEANUP_PENDING"
    assert result["phase"] == "CANOPY_SCRATCH_PENDING"
    assert all(path.is_symlink() for path in source_view_paths)
    assert [path.read_bytes() for path in source_files] == originals
    bundle_path = fixture["report_path"].with_name("." + fixture["report_path"].name + ".closeout") / "transaction.json"
    bundle = json.loads(bundle_path.read_text())
    assert bundle["phase"] == "CANOPY_SCRATCH_PENDING"
    assert bundle["canopy_scratch_plans"][0]["relay_root"] == str(relay_root)
    assert len(bundle["canopy_scratch_plans"][0]["links"]) == 1

    completed = module.closeout_report_only_run_v1(**fixture, apply=True)
    assert completed["status"] == "COMPLETED"
    assert not relay_root.exists()
    assert not any(path.is_symlink() for path in source_view_paths)
    assert [path.exists() for path in source_files] == [False]


def test_terminal_relay_retargeted_link_blocks_all_unlink(fixture, monkeypatch):
    relay_root, source_files, source_view_paths = _terminal_relay(
        fixture, monkeypatch, second_source=True, retarget_second=True
    )
    originals = [path.read_bytes() for path in source_files]
    outside = fixture["report_path"].parent / "outside-source.txt"
    with pytest.raises(retirement.RootRetirementError, match="target differs"):
        module.closeout_report_only_run_v1(**fixture)
    assert all(path.is_symlink() for path in source_view_paths)
    assert [path.read_bytes() for path in source_files] == originals
    assert outside.read_text() == "must remain unchanged"
    assert relay_root.exists()


def test_terminal_relay_scratch_cleanup_resumes_after_partial_unlink(fixture, monkeypatch):
    relay_root, source_files, source_view_paths = _terminal_relay(
        fixture, monkeypatch, second_source=True
    )
    original_unlink = os.unlink
    calls = 0

    def fail_second_link(path, *args, **kwargs):
        nonlocal calls
        if kwargs.get("dir_fd") is not None:
            calls += 1
            if calls == 2:
                raise OSError("injected scratch unlink interruption")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "unlink", fail_second_link)
    plan = module._terminal_canopy_scratch_receipt(relay_root, fixture["roots"], apply=False)
    assert len(plan["links"]) == 2
    with pytest.raises(OSError, match="injected scratch unlink"):
        module._terminal_canopy_scratch_receipt(
            relay_root, fixture["roots"], apply=True, prior_plan=plan
        )
    assert sum(path.is_symlink() for path in source_view_paths) == 1
    monkeypatch.setattr(module.os, "unlink", original_unlink)
    resumed = module._terminal_canopy_scratch_receipt(
        relay_root, fixture["roots"], apply=True, prior_plan=plan
    )
    assert len(resumed["removed_links"]) == 1
    assert not any(path.is_symlink() for path in source_view_paths)
    assert [path.read_bytes() for path in source_files] == [b"temporary checkpoint", b"source event segment"]


def test_closeout_persists_plan_and_resumes_after_partial_relay_unlink(fixture, monkeypatch):
    relay_root, source_files, source_view_paths = _terminal_relay(
        fixture, monkeypatch, second_source=True
    )
    originals = [path.read_bytes() for path in source_files]
    original_unlink = os.unlink
    removed = 0

    def fail_second_view_link(path, *args, **kwargs):
        nonlocal removed
        if kwargs.get("dir_fd") is not None and path in {"checkpoint", "segment.json"}:
            removed += 1
            if removed == 2:
                raise OSError("injected closeout scratch interruption")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(module.os, "unlink", fail_second_view_link)
    with pytest.raises(OSError, match="injected closeout scratch interruption"):
        module.closeout_report_only_run_v1(**fixture, apply=True)

    transaction = fixture["report_path"].with_name("." + fixture["report_path"].name + ".closeout")
    bundle_path = transaction / "transaction.json"
    bundle = json.loads(bundle_path.read_text())
    assert bundle["phase"] == "CANOPY_SCRATCH_PENDING"
    assert len(bundle["canopy_scratch_plans"][0]["links"]) == 2
    assert sum(path.is_symlink() for path in source_view_paths) == 1
    assert [path.read_bytes() for path in source_files] == originals
    assert fixture["roots"][0].exists()

    monkeypatch.setattr(module.os, "unlink", original_unlink)
    completed = module.closeout_report_only_run_v1(**fixture, apply=True)
    assert completed["status"] == "COMPLETED"
    assert not relay_root.exists()
    assert not any(path.is_symlink() for path in source_view_paths)
    assert [path.exists() for path in source_files] == [False, False]


def test_closeout_rejects_malformed_or_changed_preparation_plan(fixture, monkeypatch):
    relay_root, source_files, source_view_paths = _terminal_relay(fixture, monkeypatch)
    result = module.closeout_report_only_run_v1(**fixture)
    assert result["phase"] == "CANOPY_SCRATCH_PENDING"
    transaction = fixture["report_path"].with_name("." + fixture["report_path"].name + ".closeout")
    bundle_path = transaction / "transaction.json"
    bundle = json.loads(bundle_path.read_text())
    original = json.loads(json.dumps(bundle))

    bundle["canopy_scratch_plans"] = ["malformed"]
    bundle["preparation_sha256"] = retirement.canonical_sha256(
        {key: value for key, value in bundle.items() if key != "preparation_sha256"}
    )
    bundle_path.write_text(json.dumps(bundle, sort_keys=True))
    with pytest.raises(retirement.RootRetirementError, match="plan is malformed"):
        module.closeout_report_only_run_v1(**fixture, apply=True)
    assert all(path.is_symlink() for path in source_view_paths)

    changed = original
    changed["canopy_scratch_plans"][0]["links"][0]["target"] = str(fixture["report_path"])
    changed["preparation_sha256"] = retirement.canonical_sha256(
        {key: value for key, value in changed.items() if key != "preparation_sha256"}
    )
    bundle_path.write_text(json.dumps(changed, sort_keys=True))
    with pytest.raises(retirement.RootRetirementError, match="link identity changed"):
        module.closeout_report_only_run_v1(**fixture, apply=True)
    assert all(path.is_symlink() for path in source_view_paths)
    assert [path.read_bytes() for path in source_files] == [b"temporary checkpoint"]
    assert relay_root.exists()


def test_non_relay_root_skips_canopy_transaction_fence(fixture, monkeypatch):
    monkeypatch.setattr(
        artifact_lifecycle,
        "run_artifact_transaction_fence_v1",
        lambda *_args, **_kwargs: pytest.fail("non-relay root acquired a Canopy fence"),
    )
    assert module._terminal_canopy_scratch_receipt(fixture["roots"][0], fixture["roots"]) is None


def test_terminal_relay_uses_live_proc_probe_without_self_fd_false_positive(fixture, monkeypatch):
    relay_root, source_files, source_view_paths = _terminal_relay(fixture, monkeypatch)
    monkeypatch.setattr(retirement, "_process_references", REAL_PROCESS_REFERENCES)

    held_fd = os.open(relay_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        references, errors = REAL_PROCESS_REFERENCES(relay_root)
        assert errors == []
        assert {"pid": os.getpid(), "reasons": ["fd"]} in references
        with pytest.raises(retirement.RootRetirementError, match="not quiescent"):
            module._terminal_canopy_scratch_receipt(
                relay_root, fixture["roots"], apply=True
            )
    finally:
        os.close(held_fd)

    original_bytes = [path.read_bytes() for path in source_files]
    result = module._terminal_canopy_scratch_receipt(
        relay_root, fixture["roots"], apply=True
    )
    assert len(result["removed_links"]) == 1
    assert not any(path.is_symlink() for path in source_view_paths)
    assert [path.read_bytes() for path in source_files] == original_bytes
