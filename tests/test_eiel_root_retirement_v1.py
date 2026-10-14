from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import threading

import pytest

from engine.campaigns import root_retirement as retirement_module
from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.root_retirement import (
    RootRetirementError,
    apply_root_retirement_v1,
    build_root_retirement_intent_v1,
    build_root_retirement_inventory_v1,
    validate_root_retirement_request_v1,
)
from tools.scripts import eiel_evidence_certificate_reclaim as reclaim_module
from tools.scripts.eiel_evidence_certificate_reclaim import (
    _root_retirement_apply,
    _root_retirement_dry_run,
)

_PROCESS_REFERENCES = retirement_module._process_references


@pytest.fixture(autouse=True)
def _allow_fixture_runtime_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = (tmp_path / "causyl-runtime").resolve()
    monkeypatch.setattr(
        retirement_module,
        "CANONICAL_RUNTIME_ROOTS",
        frozenset(
            {
                runtime,
                (tmp_path / "second" / "causyl-runtime").resolve(),
            }
        ),
    )
    monkeypatch.setattr(
        retirement_module,
        "CANONICAL_MANAGEMENT_ROOTS",
        frozenset(
            {
                runtime / "lifecycle",
                (tmp_path / "_permanent_evidences").resolve(),
            }
        ),
    )
    monkeypatch.setattr(
        retirement_module, "_process_references", lambda _root: ([], [])
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[dict, Path, Path]:
    runtime = tmp_path / "causyl-runtime"
    root = runtime / "programme" / "run-a"
    root.mkdir(parents=True)
    (root / "nested").mkdir()
    (root / "payload.bin").write_bytes(b"payload")
    (root / "nested" / "receipt.json").write_text("{}\n", encoding="utf-8")
    authority_root = tmp_path / "authority"
    authority_root.mkdir()
    compact = authority_root / "compact.json"
    terminal = authority_root / "terminal.json"
    lifecycle = authority_root / "lifecycle.json"
    for path, content in (
        (compact, "compact\n"),
        (terminal, "terminal\n"),
        (lifecycle, "lifecycle\n"),
    ):
        path.write_text(content, encoding="utf-8")
    request = {
        "schema_version": "eiel_runtime_root_retirement_request_v1",
        "retirement_id": "retirement-fixture-v1",
        "programme_id": "programme-fixture",
        "work_id": "work-fixture",
        "sealed_at_utc": "2026-08-25T00:00:00Z",
        "source": {"repository": "/repo/emergence", "commit_sha": "1" * 40},
        "allowed_runtime_roots": [str(runtime)],
        "compact_authority": {"path": str(compact), "sha256": _digest(compact)},
        "terminal_authorities": [{"path": str(terminal), "sha256": _digest(terminal)}],
        "validation": {
            "predicates": {
                "terminal_cleanup_authorized": True,
                "no_future_consumers": True,
                "audit_hold_released_or_not_required": True,
                "qualification_hold_released_or_not_required": True,
                "unknown_artifacts_resolved": True,
            },
            "semantic_review": {"status": "accepted", "reviewer": "fixture"},
        },
        "roots": [
            {
                "root_id": "run-a",
                "path": str(root),
                "kind": "run_artifact",
                "terminal_status": "completed",
                "unresolved_rx_bytes": 0,
                "live_reference_count": 0,
                "authority_receipts": [
                    {"path": str(lifecycle), "sha256": _digest(lifecycle)}
                ],
            }
        ],
        "claim_limits": ["storage retirement only; no qualification claim"],
    }
    return request, root, runtime


def _args(tmp_path: Path, request: dict) -> argparse.Namespace:
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    return argparse.Namespace(
        request=request_path,
        output_root=tmp_path / "control" / "retirement",
        apply=False,
    )


def _accept_inventory(args: argparse.Namespace, request: dict) -> dict:
    inventory_path = args.output_root / "root_retirement_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    core = {
        "schema_version": "eiel_runtime_root_retirement_review_v1",
        "retirement_id": request["retirement_id"],
        "request_sha256": canonical_sha256(
            validate_root_retirement_request_v1(request)
        ),
        "inventory_sha256": inventory["inventory_sha256"],
        "status": "accepted",
        "reviewer": "fixture-reviewer",
        "reviewed_at_utc": "2026-08-25T00:01:00Z",
        "findings": [],
        "limitations": ["fixture only"],
    }
    review = {**core, "review_sha256": canonical_sha256(core)}
    (args.output_root / "root_retirement_review.json").write_text(
        json.dumps(review, indent=2) + "\n", encoding="utf-8"
    )
    return review


def _direct_inputs(request: dict) -> tuple[dict, dict, dict, dict]:
    normalized = validate_root_retirement_request_v1(request)
    inventory = build_root_retirement_inventory_v1(normalized)
    intent = build_root_retirement_intent_v1(
        request=normalized,
        inventory=inventory,
        status="PREPARED",
    )
    review_core = {
        "schema_version": "eiel_runtime_root_retirement_review_v1",
        "retirement_id": request["retirement_id"],
        "request_sha256": canonical_sha256(normalized),
        "inventory_sha256": inventory["inventory_sha256"],
        "status": "accepted",
        "reviewer": "fixture-reviewer",
        "reviewed_at_utc": "2026-08-25T00:01:00Z",
        "findings": [],
        "limitations": ["fixture only"],
    }
    review = {**review_core, "review_sha256": canonical_sha256(review_core)}
    return normalized, inventory, intent, review


def test_request_is_confined_to_exact_causyl_runtime_parent(tmp_path: Path) -> None:
    request, _, _ = _fixture(tmp_path)
    request["allowed_runtime_roots"] = [str(tmp_path)]
    with pytest.raises(RootRetirementError, match="canonical causyl-runtime"):
        validate_root_retirement_request_v1(request)


def test_symlinked_runtime_ancestor_cannot_escape_physical_confinement(
    tmp_path: Path,
) -> None:
    request, _, runtime = _fixture(tmp_path)
    outside_parent = tmp_path / "outside"
    outside_parent.mkdir()
    outside_runtime = outside_parent / "causyl-runtime"
    runtime.rename(outside_runtime)
    link = tmp_path / "runtime-link"
    link.symlink_to(outside_parent, target_is_directory=True)
    request["allowed_runtime_roots"] = [str(link / "causyl-runtime")]
    request["roots"][0]["path"] = str(link / "causyl-runtime" / "programme" / "run-a")
    with pytest.raises(RootRetirementError, match="canonical causyl-runtime"):
        validate_root_retirement_request_v1(request)


def test_management_roots_and_aliases_cannot_be_retirement_targets(
    tmp_path: Path,
) -> None:
    request, _, runtime = _fixture(tmp_path)
    lifecycle = runtime / "lifecycle"
    pending = lifecycle / "pending"
    pending.mkdir(parents=True)
    permanent = tmp_path / "_permanent_evidences"
    permanent.mkdir()
    targets = [
        runtime,
        lifecycle,
        pending,
        lifecycle / "budgets",
        lifecycle / "status",
        runtime / "programme" / ".eiel-storage-reservations",
        permanent,
    ]
    for target in targets:
        candidate = copy.deepcopy(request)
        candidate["roots"][0]["path"] = str(target)
        with pytest.raises(RootRetirementError, match="storage management root"):
            validate_root_retirement_request_v1(candidate)

    alias = runtime / "programme" / "management-alias"
    alias.symlink_to(lifecycle, target_is_directory=True)
    aliased = copy.deepcopy(request)
    aliased["roots"][0]["path"] = str(alias / "pending")
    with pytest.raises(RootRetirementError, match="storage management root"):
        validate_root_retirement_request_v1(aliased)


def test_dry_run_seals_exact_inventory_without_mutation(tmp_path: Path) -> None:
    request, root, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    result = _root_retirement_dry_run(args)
    inventory = json.loads(Path(result["inventory_path"]).read_text())
    assert result["mode"] == "dry_run"
    assert result["member_count"] == 3
    assert inventory["roots"][0]["bytes"] == len(b"payload") + len(b"{}\n")
    assert root.is_dir()


def test_apply_removes_only_sealed_root_and_replays_idempotently(
    tmp_path: Path,
) -> None:
    request, root, runtime = _fixture(tmp_path)
    sibling = runtime / "unrelated"
    sibling.mkdir()
    (sibling / "keep").write_text("keep", encoding="utf-8")
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    args.apply = True
    first = _root_retirement_apply(args)
    second = _root_retirement_apply(args)
    assert first["idempotent_replay"] is False
    assert second["idempotent_replay"] is True
    assert not root.exists()
    assert (sibling / "keep").read_text(encoding="utf-8") == "keep"


def test_multi_root_apply_uses_one_fence_per_runtime_namespace(tmp_path: Path) -> None:
    request, root, runtime = _fixture(tmp_path)
    second = runtime / "programme" / "run-b"
    second.mkdir()
    (second / "payload.bin").write_bytes(b"second")
    second_row = copy.deepcopy(request["roots"][0])
    second_row["root_id"] = "run-b"
    second_row["path"] = str(second)
    request["roots"].append(second_row)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    args.apply = True
    result = _root_retirement_apply(args)
    assert result["root_count"] == 2
    assert not root.exists()
    assert not second.exists()


def test_apply_rejects_late_or_changed_members(tmp_path: Path) -> None:
    request, root, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    (root / "late.bin").write_bytes(b"late")
    args.apply = True
    with pytest.raises(RootRetirementError, match="changed after dry-run"):
        _root_retirement_apply(args)
    assert root.is_dir()


def test_apply_rejects_missing_original_member_before_staging(tmp_path: Path) -> None:
    request, root, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    (root / "payload.bin").unlink()
    args.apply = True
    with pytest.raises(RootRetirementError, match="changed after dry-run"):
        _root_retirement_apply(args)
    assert root.is_dir()
    assert not list(root.parent.glob(".eiel-retirement-*"))


def test_canonical_reseal_waits_from_final_inventory_check_through_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, root, _ = _fixture(tmp_path)
    normalized, inventory, intent, review = _direct_inputs(request)
    progress = tmp_path / "control" / "progress.json"
    original_rename = retirement_module._rename_noreplace
    attempted = threading.Event()
    acquired = threading.Event()
    reseal_thread: threading.Thread | None = None

    def rename_after_reseal_attempt(src: str, dst: str, **kwargs: int) -> None:
        nonlocal reseal_thread
        if src == root.name and reseal_thread is None:

            def attempt_canonical_reseal() -> None:
                attempted.set()
                with retirement_module.run_artifact_transaction_fence_v1(root):
                    acquired.set()

            reseal_thread = threading.Thread(target=attempt_canonical_reseal)
            reseal_thread.start()
            assert attempted.wait(timeout=2)
            assert acquired.wait(timeout=0.1) is False
        original_rename(src, dst, **kwargs)

    monkeypatch.setattr(
        retirement_module, "_rename_noreplace", rename_after_reseal_attempt
    )
    receipt = apply_root_retirement_v1(
        request=normalized,
        inventory=inventory,
        intent=intent,
        review=review,
        progress_path=progress,
    )
    assert receipt["result"] == "applied"
    assert reseal_thread is not None
    reseal_thread.join(timeout=2)
    assert acquired.is_set()
    assert not root.exists()


def test_atomic_member_stage_never_deletes_a_racing_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, root, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    original_rename = retirement_module._rename_noreplace
    raced = False

    def racing_rename(src: str, dst: str, **kwargs: int) -> None:
        nonlocal raced
        if src == "payload.bin" and not raced:
            raced = True
            directory_fd = int(kwargs["src_dir_fd"])
            os.rename(
                "payload.bin",
                "attacker-old.bin",
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            replacement_fd = os.open(
                "payload.bin",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            try:
                os.write(replacement_fd, b"payload")
            finally:
                os.close(replacement_fd)
        original_rename(src, dst, **kwargs)

    monkeypatch.setattr(retirement_module, "_rename_noreplace", racing_rename)
    args.apply = True
    with pytest.raises(
        RootRetirementError, match="identity changed during atomic staging"
    ):
        _root_retirement_apply(args)
    staged_roots = list(root.parent.glob(".eiel-retirement-*"))
    assert len(staged_roots) == 1
    surviving_bytes = [
        path.read_bytes() for path in staged_roots[0].rglob("*") if path.is_file()
    ]
    assert surviving_bytes.count(b"payload") == 2


def test_apply_requires_review_of_the_exact_sealed_inventory(tmp_path: Path) -> None:
    request, root, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    args.apply = True
    with pytest.raises(RootRetirementError, match="exact prior dry-run artifacts"):
        _root_retirement_apply(args)
    review = _accept_inventory(args, request)
    review["inventory_sha256"] = "f" * 64
    review.pop("review_sha256")
    review["review_sha256"] = canonical_sha256(review)
    (args.output_root / "root_retirement_review.json").write_text(
        json.dumps(review, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(RootRetirementError, match="inventory binding mismatch"):
        _root_retirement_apply(args)
    assert root.is_dir()


def test_idempotent_replay_rejects_semantically_rewritten_receipt(
    tmp_path: Path,
) -> None:
    request, _, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    args.apply = True
    _root_retirement_apply(args)
    receipt_path = args.output_root / "root_retirement_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt["programme_id"] = "rewritten-programme"
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RootRetirementError, match="receipt semantics mismatch"):
        _root_retirement_apply(args)


def test_idempotent_replay_rejects_receipt_completion_outside_final_event(
    tmp_path: Path,
) -> None:
    request, _, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    args.apply = True
    _root_retirement_apply(args)
    receipt_path = args.output_root / "root_retirement_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("receipt_sha256")
    receipt["completed_at_utc"] = "2026-08-25T23:59:59Z"
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(RootRetirementError, match="receipt semantics mismatch"):
        _root_retirement_apply(args)


def test_inventory_rejects_symlinks_and_hardlinks(tmp_path: Path) -> None:
    request, root, _ = _fixture(tmp_path)
    os.symlink(root / "payload.bin", root / "link.bin")
    with pytest.raises(RootRetirementError, match="symlink"):
        build_root_retirement_inventory_v1(request)
    (root / "link.bin").unlink()
    os.link(root / "payload.bin", root / "hard.bin")
    with pytest.raises(RootRetirementError, match="hard-linked"):
        build_root_retirement_inventory_v1(request)


def test_authority_hash_tamper_blocks_apply(tmp_path: Path) -> None:
    request, root, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    Path(request["compact_authority"]["path"]).write_text(
        "tampered\n", encoding="utf-8"
    )
    args.apply = True
    with pytest.raises(RootRetirementError, match="SHA-256 mismatch"):
        _root_retirement_apply(args)
    assert root.is_dir()


def test_authority_must_survive_outside_every_target(tmp_path: Path) -> None:
    request, root, _ = _fixture(tmp_path)
    internal = root / "internal-authority.json"
    internal.write_text("authority\n", encoding="utf-8")
    request["roots"][0]["authority_receipts"] = [
        {"path": str(internal), "sha256": _digest(internal)}
    ]
    with pytest.raises(RootRetirementError, match="authority must be external"):
        validate_root_retirement_request_v1(request)


def test_direct_apply_rejects_progress_overlap_and_symlink_alias_before_stage(
    tmp_path: Path,
) -> None:
    request, root, _ = _fixture(tmp_path)
    normalized, inventory, intent, review = _direct_inputs(request)
    stage = root.parent / (
        ".eiel-retirement-"
        + canonical_sha256(
            {"retirement_id": request["retirement_id"], "root_id": "run-a"}
        )[:24]
    )
    alias = tmp_path / "progress-alias"
    alias.symlink_to(root, target_is_directory=True)
    for progress in (root / "progress.json", stage / "progress.json", alias / "p.json"):
        with pytest.raises(RootRetirementError, match="disjoint"):
            apply_root_retirement_v1(
                request=normalized,
                inventory=inventory,
                intent=intent,
                review=review,
                progress_path=progress,
            )
        assert root.is_dir()
        assert not stage.exists()


def test_pre_staged_event_recovery_rejects_an_altered_stage(tmp_path: Path) -> None:
    request, root, _ = _fixture(tmp_path)
    normalized, inventory, intent, review = _direct_inputs(request)
    stage = root.parent / (
        ".eiel-retirement-"
        + canonical_sha256(
            {"retirement_id": request["retirement_id"], "root_id": "run-a"}
        )[:24]
    )
    root.rename(stage)
    (stage / "payload.bin").unlink()
    progress = tmp_path / "control" / "progress.json"
    with pytest.raises(RootRetirementError, match="differs from inventory"):
        apply_root_retirement_v1(
            request=normalized,
            inventory=inventory,
            intent=intent,
            review=review,
            progress_path=progress,
        )
    assert stage.is_dir()
    assert not progress.exists()


def test_crash_after_root_rename_authenticates_stage_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, root, _ = _fixture(tmp_path)
    normalized, inventory, intent, review = _direct_inputs(request)
    progress = tmp_path / "control" / "progress.json"
    stage = root.parent / (
        ".eiel-retirement-"
        + canonical_sha256(
            {"retirement_id": request["retirement_id"], "root_id": "run-a"}
        )[:24]
    )
    real_atomic = retirement_module._atomic_json
    failed = False

    def crash_before_staged_event(path: Path, value: dict) -> Path:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("simulated crash before durable STAGED")
        return real_atomic(path, value)

    monkeypatch.setattr(
        retirement_module,
        "_atomic_json",
        crash_before_staged_event,
    )
    with pytest.raises(RuntimeError, match="before durable STAGED"):
        apply_root_retirement_v1(
            request=normalized,
            inventory=inventory,
            intent=intent,
            review=review,
            progress_path=progress,
        )
    assert failed is True
    assert not root.exists()
    assert stage.is_dir()
    assert (stage / "payload.bin").read_bytes() == b"payload"
    assert not progress.exists()

    monkeypatch.setattr(retirement_module, "_atomic_json", real_atomic)
    receipt = apply_root_retirement_v1(
        request=normalized,
        inventory=inventory,
        intent=intent,
        review=review,
        progress_path=progress,
    )
    assert receipt["result"] == "applied"
    assert not stage.exists()
    persisted = json.loads(progress.read_text(encoding="utf-8"))
    assert [event["event"] for event in persisted["events"]] == [
        "STAGED",
        "RETIRED",
    ]


def test_fsync_failure_after_stage_rmdir_cannot_publish_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, root, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    args.apply = True
    real_rmdir = retirement_module.os.rmdir
    real_fsync = retirement_module.os.fsync
    fail_next_fsync = False
    failed = False

    def observe_rmdir(path: object, *call_args: object, **call_kwargs: object) -> None:
        nonlocal fail_next_fsync
        real_rmdir(path, *call_args, **call_kwargs)
        if str(path).startswith(".eiel-retirement-"):
            fail_next_fsync = True

    def fail_after_stage_rmdir(fd: int) -> None:
        nonlocal fail_next_fsync, failed
        if fail_next_fsync and not failed:
            fail_next_fsync = False
            failed = True
            raise OSError("simulated parent fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(retirement_module.os, "rmdir", observe_rmdir)
    monkeypatch.setattr(retirement_module.os, "fsync", fail_after_stage_rmdir)
    with pytest.raises(OSError, match="parent fsync failure"):
        _root_retirement_apply(args)
    assert failed is True
    assert not root.exists()
    assert not (args.output_root / "root_retirement_receipt.json").exists()
    progress = json.loads(
        (args.output_root / "root_retirement_intent.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "APPLYING"
    assert [event["event"] for event in progress["events"]] == ["STAGED"]

    monkeypatch.setattr(retirement_module.os, "rmdir", real_rmdir)
    monkeypatch.setattr(retirement_module.os, "fsync", real_fsync)
    recovered = _root_retirement_apply(args)
    assert recovered["idempotent_replay"] is False
    assert (args.output_root / "root_retirement_receipt.json").is_file()


def test_cli_replays_an_authenticated_applying_intent_after_stage_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, root, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    original_delete = retirement_module._delete_inventory_subset_fd
    failed_once = False

    def fail_once(*call_args: object, **call_kwargs: object) -> None:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            raise RuntimeError("simulated crash after root staging")
        original_delete(*call_args, **call_kwargs)

    monkeypatch.setattr(retirement_module, "_delete_inventory_subset_fd", fail_once)
    args.apply = True
    with pytest.raises(RuntimeError, match="simulated crash"):
        _root_retirement_apply(args)
    progress = json.loads(
        (args.output_root / "root_retirement_intent.json").read_text(encoding="utf-8")
    )
    assert progress["status"] == "APPLYING"
    result = _root_retirement_apply(args)
    assert result["idempotent_replay"] is False
    assert not root.exists()


@pytest.mark.parametrize(
    ("tamper", "error_match"),
    [
        (None, None),
        ("changed", "differs from sealed inventory"),
        ("extra", "unsealed members"),
    ],
)
def test_readonly_nested_payload_is_unlocked_only_in_stage_and_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str | None,
    error_match: str | None,
) -> None:
    request, root, _ = _fixture(tmp_path)
    (root / "payload.bin").unlink()
    (root / "nested" / "receipt.json").unlink()
    relative = "nested/workhorse-semantic-event-sink/tick-000/" + "a" * 64
    payload_dir = root / relative
    payload_dir.mkdir(parents=True)
    payload = payload_dir / "packed_events.bin"
    payload.write_bytes(b"sealed payload")
    payload.chmod(0o444)
    (payload_dir / "a_first.bin").write_bytes(b"first sealed member")
    (payload_dir / "z_survivor.bin").write_bytes(b"remaining sealed member")
    readonly_dirs = [
        root,
        root / "nested",
        root / "nested" / "workhorse-semantic-event-sink",
        root / "nested" / "workhorse-semantic-event-sink" / "tick-000",
        payload_dir,
    ]
    for directory in readonly_dirs:
        directory.chmod(0o555)

    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    args.apply = True
    stage = root.parent / retirement_module._stage_name(
        request["retirement_id"], request["roots"][0]["root_id"]
    )
    staged_member = retirement_module._member_stage_name(
        f"{relative}/packed_events.bin"
    )
    real_unlink = os.unlink
    failed_once = False

    def fail_one_staged_unlink(path: str, *args: object, **kwargs: object) -> None:
        nonlocal failed_once
        if path == staged_member and not failed_once:
            failed_once = True
            raise PermissionError(
                errno.EACCES, "simulated transient unlink denial", "packed_events.bin"
            )
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(retirement_module.os, "unlink", fail_one_staged_unlink)
    with pytest.raises(PermissionError, match="simulated transient unlink denial"):
        _root_retirement_apply(args)
    assert failed_once
    assert not root.exists()
    assert stage.is_dir()
    staged_payload_dir = stage / relative
    assert (staged_payload_dir / staged_member).read_bytes() == b"sealed payload"
    assert not (staged_payload_dir / "a_first.bin").exists()
    assert (staged_payload_dir / "z_survivor.bin").read_bytes() == (
        b"remaining sealed member"
    )
    staged_readonly_dirs = [
        stage,
        stage / "nested",
        stage / "nested" / "workhorse-semantic-event-sink",
        stage / "nested" / "workhorse-semantic-event-sink" / "tick-000",
        staged_payload_dir,
    ]
    assert all(
        directory.stat().st_mode & stat.S_IWUSR
        for directory in staged_readonly_dirs
    )
    progress_path = args.output_root / "root_retirement_intent.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "APPLYING"
    assert [event["event"] for event in progress["events"]] == ["STAGED"]

    monkeypatch.setattr(retirement_module.os, "unlink", real_unlink)
    if tamper == "changed":
        (staged_payload_dir / "z_survivor.bin").write_bytes(b"changed member")
    elif tamper == "extra":
        (staged_payload_dir / "extra.bin").write_bytes(b"unsealed member")
    if error_match is not None:
        with pytest.raises(RootRetirementError, match=error_match):
            _root_retirement_apply(args)
        assert stage.is_dir()
        return

    result = _root_retirement_apply(args)
    assert result["idempotent_replay"] is False
    assert not root.exists()
    assert not stage.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert [event["event"] for event in progress["events"]] == ["STAGED", "RETIRED"]


def test_cli_rematerializes_receipt_after_final_intent_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, root, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    original_write = reclaim_module.write_json_atomic
    failed_once = False
    first_receipt: dict | None = None

    def fail_receipt_once(path: Path, value: dict) -> Path:
        nonlocal failed_once, first_receipt
        if path.name == "root_retirement_receipt.json" and not failed_once:
            failed_once = True
            first_receipt = copy.deepcopy(value)
            raise RuntimeError("simulated crash before receipt publication")
        return original_write(path, value)

    monkeypatch.setattr(reclaim_module, "write_json_atomic", fail_receipt_once)
    args.apply = True
    with pytest.raises(RuntimeError, match="before receipt publication"):
        _root_retirement_apply(args)
    assert not root.exists()
    assert not (args.output_root / "root_retirement_receipt.json").exists()
    final_intent = json.loads(
        (args.output_root / "root_retirement_intent.json").read_text(encoding="utf-8")
    )
    assert final_intent["status"] == "APPLIED"
    result = _root_retirement_apply(args)
    assert result["idempotent_replay"] is False
    receipt = json.loads(
        (args.output_root / "root_retirement_receipt.json").read_text(encoding="utf-8")
    )
    assert first_receipt is not None
    assert receipt == first_receipt
    assert result["receipt_sha256"] == first_receipt["receipt_sha256"]


def test_cli_holds_producer_fence_through_receipt_and_result_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, root, _ = _fixture(tmp_path)
    args = _args(tmp_path, request)
    _root_retirement_dry_run(args)
    _accept_inventory(args, request)
    args.apply = True
    original_write = reclaim_module.write_json_atomic
    producer_attempted = threading.Event()
    producer_acquired = threading.Event()
    producer: threading.Thread | None = None

    def recreate_under_canonical_fence() -> None:
        producer_attempted.set()
        with retirement_module.run_artifact_transaction_fence_v1(root):
            producer_acquired.set()
            root.mkdir(parents=True)
            (root / "new-producer-output.bin").write_bytes(b"new")

    def write_while_producer_waits(path: Path, value: dict) -> Path:
        nonlocal producer
        if path.name == "root_retirement_receipt.json" and producer is None:
            producer = threading.Thread(target=recreate_under_canonical_fence)
            producer.start()
            assert producer_attempted.wait(timeout=2)
            assert producer_acquired.wait(timeout=0.1) is False
        if path.name == "root_retirement_result.json":
            assert producer is not None
            assert producer_acquired.is_set() is False
        return original_write(path, value)

    monkeypatch.setattr(reclaim_module, "write_json_atomic", write_while_producer_waits)
    result = _root_retirement_apply(args)
    assert producer is not None
    producer.join(timeout=2)
    assert producer_acquired.is_set()
    assert (
        result["receipt_sha256"]
        == json.loads(
            (args.output_root / "root_retirement_receipt.json").read_text(
                encoding="utf-8"
            )
        )["receipt_sha256"]
    )
    assert (root / "new-producer-output.bin").read_bytes() == b"new"


def test_forged_retired_event_cannot_authorize_a_missing_root(tmp_path: Path) -> None:
    request, _, _ = _fixture(tmp_path)
    normalized = validate_root_retirement_request_v1(request)
    inventory = build_root_retirement_inventory_v1(normalized)
    prepared = build_root_retirement_intent_v1(
        request=normalized, inventory=inventory, status="PREPARED"
    )
    forged = build_root_retirement_intent_v1(
        request=normalized,
        inventory=inventory,
        status="APPLYING",
        previous_intent_sha256=prepared["intent_sha256"],
        prepared_intent_sha256=prepared["intent_sha256"],
        events=[
            {
                "event": "STAGED",
                "root_id": "run-a",
                "inventory_sha256": inventory["roots"][0]["inventory_sha256"],
                "at_utc": "2026-08-25T00:02:00Z",
            },
            {
                "event": "RETIRED",
                "root_id": "run-a",
                "inventory_sha256": "f" * 64,
                "at_utc": "2026-08-25T00:03:00Z",
            },
        ],
    )
    review_core = {
        "schema_version": "eiel_runtime_root_retirement_review_v1",
        "retirement_id": request["retirement_id"],
        "request_sha256": canonical_sha256(normalized),
        "inventory_sha256": inventory["inventory_sha256"],
        "status": "accepted",
        "reviewer": "fixture-reviewer",
        "reviewed_at_utc": "2026-08-25T00:04:00Z",
        "findings": [],
        "limitations": ["fixture only"],
    }
    review = {**review_core, "review_sha256": canonical_sha256(review_core)}
    with pytest.raises(RootRetirementError, match="event root inventory binding"):
        apply_root_retirement_v1(
            request=normalized,
            inventory=inventory,
            intent=forged,
            review=review,
            progress_path=tmp_path / "forged-progress.json",
        )


def test_overlapping_targets_and_output_root_are_rejected(tmp_path: Path) -> None:
    request, root, _ = _fixture(tmp_path)
    duplicate = copy.deepcopy(request["roots"][0])
    duplicate["root_id"] = "nested"
    duplicate["path"] = str(root / "nested")
    request["roots"].append(duplicate)
    with pytest.raises(RootRetirementError, match="overlapping"):
        validate_root_retirement_request_v1(request)

    request.pop("roots")
    clean_request, clean_root, _ = _fixture(tmp_path / "second")
    args = _args(tmp_path / "second", clean_request)
    args.output_root = clean_root / "control"
    with pytest.raises(RootRetirementError, match="disjoint"):
        _root_retirement_dry_run(args)


def _fake_process(
    proc_root: Path,
    pid: int,
    *,
    cwd: Path,
    cmdline: bytes,
    fd_targets: tuple[Path, ...] = (),
) -> Path:
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / "comm").write_text("python\n", encoding="utf-8")
    (process / "cmdline").write_bytes(cmdline)
    (process / "cwd").symlink_to(cwd, target_is_directory=True)
    fd_root = process / "fd"
    fd_root.mkdir()
    for index, target in enumerate(fd_targets, start=3):
        (fd_root / str(index)).symlink_to(target)
    return process


def test_own_canonical_finish_run_cmdline_does_not_block_retirement_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "causyl-runtime" / "runs" / "finished"
    root.mkdir(parents=True)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    own_pid = 41001
    cmdline = (
        b"./tools/bin/eiel\0storage\0finish-run\0report\0--run-root\0"
        + os.fsencode(root)
    )
    _fake_process(
        proc_root,
        own_pid,
        cwd=tmp_path,
        cmdline=cmdline,
    )
    monkeypatch.setattr(retirement_module, "PROC_ROOT", proc_root)
    monkeypatch.setattr(retirement_module.os, "getpid", lambda: own_pid)
    monkeypatch.setattr(retirement_module, "_process_references", _PROCESS_REFERENCES)

    references, inspection_errors = retirement_module._process_references(root)

    assert references == []
    assert inspection_errors == []


def test_concurrent_process_cmdline_still_protects_exact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "causyl-runtime" / "runs" / "finished"
    root.mkdir(parents=True)
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    own_pid, active_pid = 41001, 41002
    _fake_process(
        proc_root,
        own_pid,
        cwd=tmp_path,
        cmdline=b"eiel\0storage\0finish-run\0report\0--run-root\0" + os.fsencode(root),
    )
    _fake_process(
        proc_root,
        active_pid,
        cwd=tmp_path,
        cmdline=b"worker\0--run-root\0" + os.fsencode(root),
    )
    monkeypatch.setattr(retirement_module, "PROC_ROOT", proc_root)
    monkeypatch.setattr(retirement_module.os, "getpid", lambda: own_pid)
    monkeypatch.setattr(retirement_module, "_process_references", _PROCESS_REFERENCES)

    references, inspection_errors = retirement_module._process_references(root)

    assert references == [{"pid": active_pid, "reasons": ["cmdline"]}]
    assert inspection_errors == []


def test_own_cwd_and_open_fd_still_protect_exact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "causyl-runtime" / "runs" / "finished"
    root.mkdir(parents=True)
    member = root / "held-open.dat"
    member.write_bytes(b"live")
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    own_pid = 41001
    process = _fake_process(
        proc_root,
        own_pid,
        cwd=root,
        cmdline=b"eiel\0storage\0finish-run\0--run-root\0" + os.fsencode(root),
    )
    monkeypatch.setattr(retirement_module, "PROC_ROOT", proc_root)
    monkeypatch.setattr(retirement_module.os, "getpid", lambda: own_pid)
    monkeypatch.setattr(retirement_module, "_process_references", _PROCESS_REFERENCES)

    references, inspection_errors = retirement_module._process_references(root)
    assert references == [{"pid": own_pid, "reasons": ["cwd"]}]
    assert inspection_errors == []

    (process / "cwd").unlink()
    (process / "cwd").symlink_to(tmp_path, target_is_directory=True)
    (process / "fd" / "3").symlink_to(member)
    references, inspection_errors = retirement_module._process_references(root)
    assert references == [{"pid": own_pid, "reasons": ["fd"]}]
    assert inspection_errors == []
