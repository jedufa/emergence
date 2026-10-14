from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.campaigns.artifact_lifecycle import (
    CHECKPOINT_TRANSIT,
    RunStorageGuard,
    build_artifact_disposition_manifest_v1,
    build_sealed_artifact_inventory_row_v1,
    lifecycle_profile_contract_v1,
)
from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.storage_availability import (
    DATA2_FILESYSTEM_RESERVE_BYTES,
    evidence_backed_execution_reserve_v1,
    load_data2_storage_availability_policy_v1,
)
from engine.campaigns.storage_reclamation import reclaim_before_storage_wait_v1
from tools.actors.common import ActorError, atomic_write_json, actor_state_root, file_sha256
from tests.eiel_actor_v14_helpers import make_bootstrap_repo
from tools.actors.repository_truth import (
    ARTIFACT_REFERENCE_SCHEMA_VERSION,
    INTEGRATION_RECEIPT_SCHEMA_VERSION,
)
from tools.actors.storage_lifecycle import closeout_merged_work_storage_v1


def _plan(contract: dict) -> dict:
    return {
        "campaign_id": "storage-policy-reform-test",
        "plan_sha256": "a" * 64,
        "artifact_lifecycle": contract,
    }


def _terminal_root(
    base: Path,
    *,
    name: str = "completed-run",
    retained: bool = False,
    held: bool = False,
) -> tuple[Path, dict, dict]:
    root = base / name
    root.mkdir(parents=True)
    scratch = root / "scratch" / "transient.bin"
    scratch.parent.mkdir()
    scratch.write_bytes(b"x" * 4096)
    if retained:
        asset = root / "assets" / "selected-profile.bin"
        asset.parent.mkdir()
        asset.write_bytes(b"asset" * 100)
    else:
        asset = None
    contract = lifecycle_profile_contract_v1(
        CHECKPOINT_TRANSIT,
        total_max_bytes=1024,
        checkpoint_schedule=[1],
    )
    inventory = [
        build_sealed_artifact_inventory_row_v1(
            run_root=root,
            path=scratch,
            artifact_family="scratch",
            retention_class="R1" if not held else "R2",
            owner_writer="storage-policy-test",
            live_reference_status="active" if held else "none",
        )
    ]
    if asset is not None:
        inventory.append(
            build_sealed_artifact_inventory_row_v1(
                run_root=root,
                path=asset,
                artifact_family="selected_asset",
                retention_class="R3",
                owner_writer="storage-policy-test",
            )
        )
    manifest = build_artifact_disposition_manifest_v1(
        _plan(contract),
        run_root=root,
        terminal_status="completed",
        live_reference_status="active" if held else "none",
        artifact_inventory=inventory,
        inventory_complete=True,
    )
    manifest_path = root / "artifact_disposition_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return root, manifest, {"scratch": scratch, "asset": asset, "manifest_path": manifest_path}


def _pointer(queue: Path, root: Path, manifest: dict) -> Path:
    queue.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "artifact_disposition_manifest.json"
    core = {
        "schema_version": "eiel_durable_archive_pointer_v1",
        "run_root": str(root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "archive_root": "/unused/cold-storage",
        "registered_at_utc": "2026-08-31T00:00:00Z",
    }
    path = queue / f"{root.name}.json"
    atomic_write_json(path, {**core, "pointer_sha256": canonical_sha256(core)})
    return path


def _integration_reference(state_root: Path, repo_root: Path) -> dict:
    integration_sha = __import__("subprocess").check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip().lower()
    body = {
        "schema_version": INTEGRATION_RECEIPT_SCHEMA_VERSION,
        "artifact_id": "INTEGRATION-STORAGE-TEST",
        "work_id": "WORK-STORAGE-TEST",
        "review_session_id": "REVIEW-STORAGE-TEST",
        "reviewed_delivery_proof": {},
        "reviewed_commit": integration_sha,
        "integration_commit": integration_sha,
        "relation": "EXACT_COMMIT",
        "verified_paths": [],
        "note": "synthetic storage-policy reform test",
        "recorded_at_utc": "2026-08-31T00:00:00Z",
    }
    body["artifact_sha256"] = canonical_sha256(body)
    path = actor_state_root(state_root) / "repository-truth" / "integrations" / "integration-test.json"
    atomic_write_json(path, body)
    return {
        "schema_version": ARTIFACT_REFERENCE_SCHEMA_VERSION,
        "artifact_id": body["artifact_id"],
        "artifact_schema_version": body["schema_version"],
        "path": str(path),
        "state_root_locator": path.relative_to(actor_state_root(state_root)).as_posix(),
        "sha256": file_sha256(path),
        "canonical_sha256": canonical_sha256(body),
    }


def test_data2_reserve_is_measured_and_not_a_global_percentage() -> None:
    policy = load_data2_storage_availability_policy_v1()
    reserve = evidence_backed_execution_reserve_v1(
        policy=policy,
        remaining_run_bytes=20 * 1024**3,
        reservation_window_bytes=512 * 1024**2,
        lifecycle_reserve_bytes=256 * 1024**2,
        active_checkpoint_bytes=768 * 1024**2,
        acs_staging_bytes=128 * 1024**2,
        reducer_staging_bytes=128 * 1024**2,
        quarantine_allowance_bytes=64 * 1024**2,
    )
    assert policy["protected_free_floor"] == {
        "minimum_bytes": 0,
        "minimum_fraction_numerator": 0,
        "minimum_fraction_denominator": 1,
    }
    assert reserve["execution_reserve_bytes"] >= DATA2_FILESYSTEM_RESERVE_BYTES
    assert reserve["execution_reserve_bytes"] < 20 * 1024**3
    assert reserve["uncertainty_margin_bytes"] > 0


def test_pre_admission_reclaims_completed_evidence_then_admits(tmp_path: Path) -> None:
    current = tmp_path / "causyl-runtime" / "runs" / "current"
    current.mkdir(parents=True)
    old_root, manifest, paths = _terminal_root(
        tmp_path / "causyl-runtime" / "runs",
        name="completed-run",
    )
    queue = tmp_path / "causyl-runtime" / "lifecycle" / "pending"
    _pointer(queue, old_root, manifest)
    def usage(_path: str):
        return type(
            "Usage",
            (),
            {"free": 10 if paths["scratch"].exists() else 100, "total": 1000},
        )()

    guard = RunStorageGuard(
        lifecycle_profile_contract_v1(
            CHECKPOINT_TRANSIT,
            total_max_bytes=1024,
            checkpoint_schedule=[1],
        ),
        run_root=current,
        planned_bytes=1,
        disk_usage=usage,
    )
    try:
        # The simulated filesystem observes the exact deletion between the
        # first failed admission and the retry.
        original = paths["scratch"]
        result = guard.enforce(
            stage="test-pre-admission",
            pre_admission_reclaimer=lambda g, failed: reclaim_before_storage_wait_v1(
                g, failed, receipt_root=tmp_path / "receipts"
            ),
        )
        if original.exists():
            raise AssertionError("pre-admission callback did not run")
        assert result["ok"] is True
        reclaim = result["pre_admission_reclamation"]
        assert reclaim["status"] == "RECLAIMED"
        assert reclaim["bytes_reclaimed"] == 4096
        receipt = json.loads(Path(reclaim["receipt_path"]).read_text())
        assert receipt["receipt_sha256"] == reclaim["receipt_sha256"]
        repeated = reclaim_before_storage_wait_v1(
            guard,
            result,
            pending_root=queue,
            receipt_root=tmp_path / "receipts",
        )
        assert repeated["bytes_reclaimed"] == 0
    finally:
        guard.release()


def test_pre_admission_fair_order_reaches_releasable_pointer(tmp_path: Path) -> None:
    current = tmp_path / "causyl-runtime" / "runs" / "current"
    current.mkdir(parents=True)
    valid_root, valid_manifest, valid_paths = _terminal_root(
        tmp_path / "causyl-runtime" / "runs",
        name="z-valid-run",
    )
    queue = tmp_path / "causyl-runtime" / "lifecycle" / "pending"
    queue.mkdir(parents=True)
    _pointer(queue, valid_root, valid_manifest)
    invalid = queue / "a-invalid-pointer.json"
    atomic_write_json(invalid, {"invalid": True})

    # The invalid pointer sorts first by name but the valid pointer is older,
    # which is the order the canonical worker uses after rotating failures.
    import os

    os.utime(queue / "z-valid-run.json", (100, 100), follow_symlinks=False)
    os.utime(invalid, (200, 200), follow_symlinks=False)

    contract = lifecycle_profile_contract_v1(
        CHECKPOINT_TRANSIT,
        total_max_bytes=1024,
        checkpoint_schedule=[1],
    )
    guard = RunStorageGuard(
        contract,
        run_root=current,
        planned_bytes=1,
        disk_usage=lambda _: type("U", (), {"free": 1, "total": 1000})(),
    )
    try:
        result = reclaim_before_storage_wait_v1(
            guard,
            guard.reserve(),
            pending_root=queue,
            receipt_root=tmp_path / "receipts",
            max_candidates=1,
        )
    finally:
        guard.release()

    assert result["bytes_reclaimed"] == 4096
    assert result["attempts"][0]["status"] == "RELEASED"
    assert not valid_paths["scratch"].exists()
    assert invalid.exists()


def test_pre_admission_preserves_live_dependency_and_retained_asset(tmp_path: Path) -> None:
    current = tmp_path / "causyl-runtime" / "runs" / "current"
    current.mkdir(parents=True)
    held_root, held_manifest, held_paths = _terminal_root(
        tmp_path / "causyl-runtime" / "runs",
        name="held-run",
        retained=True,
        held=True,
    )
    queue = tmp_path / "causyl-runtime" / "lifecycle" / "pending"
    _pointer(queue, held_root, held_manifest)
    contract = lifecycle_profile_contract_v1(
        CHECKPOINT_TRANSIT,
        total_max_bytes=1024,
        checkpoint_schedule=[1],
    )
    guard = RunStorageGuard(contract, run_root=current, planned_bytes=1, disk_usage=lambda _: type("U", (), {"free": 1, "total": 1000})())
    try:
        result = reclaim_before_storage_wait_v1(guard, guard.reserve(), receipt_root=tmp_path / "receipts")
    finally:
        guard.release()
    assert result["bytes_reclaimed"] == 0
    assert held_paths["scratch"].exists()
    assert held_paths["asset"].exists()
    assert result["attempts"][0]["status"] == "HELD"


def test_pre_admission_retries_dependency_root_after_last_consumer_closes(tmp_path: Path) -> None:
    current = tmp_path / "causyl-runtime" / "runs" / "current"
    current.mkdir(parents=True)
    held_root, initial_manifest, held_paths = _terminal_root(
        tmp_path / "causyl-runtime" / "runs",
        name="dependency-run",
    )
    contract = lifecycle_profile_contract_v1(
        CHECKPOINT_TRANSIT,
        total_max_bytes=1024,
        checkpoint_schedule=[1],
    )
    held_inventory = []
    for row in initial_manifest["artifacts"]:
        held = dict(row)
        held["live_reference_status"] = "active"
        held_inventory.append(held)
    held_manifest = build_artifact_disposition_manifest_v1(
        _plan(contract),
        run_root=held_root,
        terminal_status="completed",
        live_reference_status="active",
        artifact_inventory=held_inventory,
        inventory_complete=True,
    )
    atomic_write_json(held_paths["manifest_path"], held_manifest)
    queue = tmp_path / "causyl-runtime" / "lifecycle" / "pending"
    _pointer(queue, held_root, held_manifest)
    usage = lambda _path: type(
        "Usage", (), {"free": 1 if held_paths["scratch"].exists() else 100, "total": 1000}
    )()
    guard = RunStorageGuard(contract, run_root=current, planned_bytes=1, disk_usage=usage)
    try:
        first = reclaim_before_storage_wait_v1(
            guard,
            guard.reserve(),
            pending_root=queue,
            receipt_root=tmp_path / "receipts",
        )
        assert first["bytes_reclaimed"] == 0
        assert held_paths["scratch"].exists()

        # The producer's exact terminal inventory is updated when the named
        # consumer closes.  Rebuild the same manifest identity from the
        # producer rows; no filename or age heuristic is involved.
        released_inventory = []
        for row in held_manifest["artifacts"]:
            released = dict(row)
            released["live_reference_status"] = "none"
            released_inventory.append(released)
        released_manifest = build_artifact_disposition_manifest_v1(
            _plan(contract),
            run_root=held_root,
            terminal_status="completed",
            live_reference_status="none",
            artifact_inventory=released_inventory,
            inventory_complete=True,
        )
        atomic_write_json(held_paths["manifest_path"], released_manifest)
        _pointer(queue, held_root, released_manifest)

        second = guard.enforce(
            stage="dependency-closed",
            pre_admission_reclaimer=lambda admitted, failed: reclaim_before_storage_wait_v1(
                admitted,
                failed,
                pending_root=queue,
                receipt_root=tmp_path / "receipts",
            ),
        )
        assert second["ok"] is True
        assert second["pre_admission_reclamation"]["bytes_reclaimed"] == 4096
        assert not held_paths["scratch"].exists()
    finally:
        guard.release()


def test_managed_merge_closeout_releases_transient_and_keeps_asset(tmp_path: Path) -> None:
    repo_root = make_bootstrap_repo(tmp_path / "repo-fixture")
    state_root = tmp_path / "actor-state"
    root, manifest, paths = _terminal_root(
        tmp_path / "runs",
        name="merged-run",
        retained=True,
    )
    (root / "evidence_certificate.json").write_text("certificate\n")
    (root / "closeout_report.md").write_text("report\n")
    (root / "reproduction_recipe.json").write_text("{}\n")
    integration = _integration_reference(state_root, repo_root)
    payload = {
        "accepted": True,
        "acceptance_scope": "UNCONDITIONAL",
        "completion_basis": "ACCEPTED_REPOSITORY_DELIVERY",
        "accepted_review_session_id": "REVIEW-STORAGE-TEST",
        "storage_evidence_roots": [str(root)],
        "references": {"report_paths": [], "artifact_paths": []},
    }
    result = closeout_merged_work_storage_v1(
        work_id="WORK-STORAGE-TEST",
        payload=payload,
        integration_reference=integration,
        repo_root=repo_root,
        state_root=state_root,
    )
    assert result["status"] == "RELEASED"
    assert result["bytes_reclaimed"] == 4096
    assert not paths["scratch"].exists()
    assert paths["asset"].exists()
    assert Path(result["receipt"]["path"]).is_file()


def test_merge_identity_failure_does_not_delete_unique_work(tmp_path: Path) -> None:
    root, _manifest, paths = _terminal_root(tmp_path / "runs", name="unmerged-run")
    payload = {
        "accepted": True,
        "storage_evidence_roots": [str(root)],
        "references": {"report_paths": [], "artifact_paths": []},
    }
    repo_root = make_bootstrap_repo(tmp_path / "repo-fixture")
    integration = _integration_reference(tmp_path / "state", repo_root)
    integration_path = Path(integration["path"])
    body = json.loads(integration_path.read_text())
    body["integration_commit"] = "0" * 40
    body["artifact_sha256"] = canonical_sha256(body)
    atomic_write_json(integration_path, body)
    integration["sha256"] = file_sha256(integration_path)
    try:
        closeout_merged_work_storage_v1(
            work_id="WORK-STORAGE-TEST",
            payload=payload,
            integration_reference=integration,
            repo_root=repo_root,
            state_root=tmp_path / "state",
        )
    except Exception:
        pass
    assert paths["scratch"].exists()


def test_dirty_target_checkout_does_not_trigger_merge_cleanup(tmp_path: Path) -> None:
    repo_root = make_bootstrap_repo(tmp_path / "repo-fixture")
    state_root = tmp_path / "actor-state"
    root, _manifest, paths = _terminal_root(tmp_path / "runs", name="dirty-target")
    (root / "evidence_certificate.json").write_text("certificate\n")
    (root / "closeout_report.md").write_text("report\n")
    (root / "reproduction_recipe.json").write_text("{}\n")
    integration = _integration_reference(state_root, repo_root)
    dirty_path = repo_root / ".storage-policy-reform-test-dirty"
    dirty_path.write_text("must remain dirty\n", encoding="utf-8")
    try:
        with pytest.raises(ActorError, match="clean integrated target"):
            closeout_merged_work_storage_v1(
                work_id="WORK-STORAGE-TEST",
                payload={
                    "accepted": True,
                    "storage_evidence_roots": [str(root)],
                    "references": {"report_paths": [], "artifact_paths": []},
                },
                integration_reference=integration,
                repo_root=repo_root,
                state_root=state_root,
            )
    finally:
        dirty_path.unlink()
    assert paths["scratch"].exists()
