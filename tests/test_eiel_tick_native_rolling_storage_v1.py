from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.campaigns.artifact_lifecycle import (
    ArtifactLifecycleError,
    SCIENCE_COMPACT,
    build_artifact_disposition_manifest_v1,
    build_sealed_artifact_inventory_row_v1,
    lifecycle_profile_contract_v1,
    validate_run_artifact_lifecycle_contract_v1,
)
from engine.campaigns.canonical import canonical_sha256
import engine.campaigns.orchestration as orchestration
from engine.campaigns.orchestration import (
    BatchExecutionContextV1,
    CampaignExecutionError,
    _collect_lifecycle_inventory_v1,
    _lifecycle_checkpoint_catalog,
)
from engine.campaigns.providers import (
    CampaignProviderError,
    _release_tick_native_transient_storage_v1,
    _tick_native_retained_prior_checkpoint_catalog_v1,
)


def _lifecycle_plan(tmp_path: Path) -> tuple[dict, dict]:
    contract = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=100_000_000,
        checkpoint_schedule=[2500, 5000],
    )
    contract.pop("contract_sha256", None)
    contract = validate_run_artifact_lifecycle_contract_v1(contract)
    plan = {
        "campaign_id": "rolling-storage-fixture",
        "plan_sha256": "a" * 64,
        "resolved_profile": "eiel-production-runtime-v1",
        "artifact_lifecycle": contract,
        "artifacts": {},
        "cells": [{"cell_id": "cell-0", "parameters": {}, "seeds": {}}],
        "execution": {"total_ticks": 5000, "checkpoint_ticks": [2500, 5000]},
    }
    return plan, contract


def _checkpoint(path: Path) -> dict:
    return {"cell_bundles": [{"path": str(path)}]}


def _native_coordinate(tick: int) -> dict[str, object]:
    return {
        "time_authority": "absolute_tick_v1",
        "absolute_tick": tick,
        "next_absolute_tick": tick,
        "checkpoint_at_tick": tick,
    }


def _sealed_inventory_row(
    *, run_root: Path, path: Path, checkpoint_identity_sha256: str
) -> dict:
    row = build_sealed_artifact_inventory_row_v1(
        run_root=run_root,
        path=path,
        artifact_family="checkpoint.bundle_v2",
        retention_class="R1",
        owner_writer="fixture-provider",
        producer_checkpoint_identity_sha256=checkpoint_identity_sha256,
    )
    observed = path.stat()
    row["producer_sealed_stat"] = {
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "bytes": int(observed.st_size),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }
    return row


def test_tick_native_prior_checkpoint_catalog_is_flat_and_requested_only() -> None:
    catalog = {
        "1120": {
            "absolute_tick": 1120,
            "prior_checkpoint_catalog": {"0": {"legacy": True}},
        },
        "2500": {
            "absolute_tick": 2500,
            "prior_checkpoint_catalog": {"1120": {"recursive": True}},
        },
        "5000": {
            "absolute_tick": 5000,
            "prior_checkpoint_catalog": {"2500": {"recursive": True}},
        },
    }

    retained = _tick_native_retained_prior_checkpoint_catalog_v1(
        catalog,
        requested_checkpoint_ticks=[2500, 5000],
    )

    assert set(retained) == {"2500", "5000"}
    assert all("prior_checkpoint_catalog" not in value for value in retained.values())
    assert "prior_checkpoint_catalog" in catalog["2500"]


def test_tick_native_transient_release_is_exact_and_updates_inventory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    transient_root = batch_root / "checkpoint_native_aggregate_v1"
    requested_root = batch_root / "checkpoint_bundles_v2" / "tick-2500.bundle"
    replacement_root = transient_root / "tick-5000.acsaggregate"
    old_owner_root = transient_root / "tick-1120.acsaggregate"
    transient_root.mkdir(parents=True)
    requested_root.parent.mkdir(parents=True)
    old_owner_root.write_bytes(b"old-owner")
    requested_root.write_bytes(b"requested")
    replacement_root.write_bytes(b"replacement")
    source_root = batch_root / "source_import_v1"
    source_root.mkdir(parents=True)
    (source_root / "projection.bin").write_bytes(b"source")

    replacement_identity = "f" * 64
    manifest_path = batch_root / "batch_manifest.json"
    inventory_paths = (
        old_owner_root,
        requested_root,
        replacement_root,
        source_root,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "cell_ids": ["cell-0"],
                "artifact_inventory": [
                    _sealed_inventory_row(
                        run_root=run_root,
                        path=path,
                        checkpoint_identity_sha256=replacement_identity,
                    )
                    for path in inventory_paths
                ],
            }
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
    )
    released = _release_tick_native_transient_storage_v1(
        context=context,
        prior_checkpoint_catalog={
            "1120": _checkpoint(old_owner_root),
            "2500": _checkpoint(requested_root),
        },
        retained_checkpoint_catalog={"2500": _checkpoint(requested_root)},
        replacement_checkpoint={
            **_native_coordinate(5000),
            "checkpoint_identity_sha256": replacement_identity,
            "next_absolute_tick": 5000,
            "cell_bundles": [
                {"cell_id": "cell-0", "path": str(replacement_root)}
            ],
            "source_import_projection_digest_sha256s": {"cell-0": "1" * 64},
            "terminal_cell_digests": {"cell-0": "2" * 64},
        },
        requested_checkpoint_ticks=[2500, 5000],
        release_source_import=True,
    )

    assert not old_owner_root.exists()
    assert requested_root.exists()
    assert replacement_root.exists()
    assert not source_root.exists()
    assert {row["artifact_kind"] for row in released} == {
        "rolling_owner_checkpoint",
        "initial_source_import_store",
    }
    for receipt in released:
        core = {
            key: value for key, value in receipt.items() if key != "receipt_sha256"
        }
        assert receipt["receipt_sha256"] == canonical_sha256(core)
        assert receipt["replacement_checkpoint_identity_sha256"] == (
            replacement_identity
        )

    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    remaining_paths = {row["path"] for row in updated_manifest["artifact_inventory"]}
    assert str(old_owner_root.relative_to(run_root)) not in remaining_paths
    assert str(source_root.relative_to(run_root)) not in remaining_paths
    assert str(requested_root.relative_to(run_root)) in remaining_paths
    assert str(replacement_root.relative_to(run_root)) in remaining_paths


def test_tick_native_transient_release_accepts_sealed_per_cell_native_checkpoint(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    checkpoint_root = batch_root / "checkpoint_native_v1" / "cell-0"
    checkpoint_root.mkdir(parents=True)
    old_checkpoint = checkpoint_root / "checkpoint_tick000000003359.acsnative"
    replacement = checkpoint_root / "checkpoint_tick000000003360.acsnative"
    old_checkpoint.write_bytes(b"old-checkpoint")
    replacement.write_bytes(b"replacement")
    source_root = batch_root / "source_import_v1"
    source_root.mkdir(parents=True)
    (source_root / "projection.bin").write_bytes(b"source")
    replacement_identity = "f" * 64
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cell_ids": ["cell-0"],
                "artifact_inventory_complete": True,
                "artifact_inventory": [
                    _sealed_inventory_row(
                        run_root=run_root,
                        path=path,
                        checkpoint_identity_sha256=replacement_identity,
                    )
                    for path in (old_checkpoint, replacement, source_root)
                ],
            }
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
    )

    released = _release_tick_native_transient_storage_v1(
        context=context,
        prior_checkpoint_catalog={"3359": _checkpoint(old_checkpoint)},
        retained_checkpoint_catalog={},
        replacement_checkpoint={
            **_native_coordinate(3360),
            "checkpoint_identity_sha256": replacement_identity,
            "cell_bundles": [
                {"cell_id": "cell-0", "path": str(replacement)}
            ],
            "source_import_projection_digest_sha256s": {"cell-0": "1" * 64},
            "terminal_cell_digests": {"cell-0": "2" * 64},
        },
        requested_checkpoint_ticks=[3360],
        release_source_import=True,
    )

    assert old_checkpoint.exists() is False
    assert replacement.is_file()
    assert source_root.exists() is False
    assert {row["artifact_kind"] for row in released} == {
        "rolling_owner_checkpoint",
        "initial_source_import_store",
    }


def test_tick_native_transient_release_accepts_sealed_world_sharded_checkpoint(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    old_root = batch_root / "checkpoint_native_aggregate_v1"
    world_root = (
        batch_root
        / "checkpoint_world_sharded_v1"
        / "checkpoint_tick000000003360"
        / ("a" * 64)
        / "objects"
    )
    old_root.mkdir(parents=True)
    world_root.mkdir(parents=True)
    old_checkpoint = old_root / "checkpoint_tick000000003359.acsaggregate"
    replacement = world_root / f"world-000-{'b' * 64}.acsworld"
    old_checkpoint.write_bytes(b"old-checkpoint")
    replacement.write_bytes(b"replacement")
    replacement_identity = "f" * 64
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cell_ids": ["cell-0"],
                "artifact_inventory_complete": True,
                "artifact_inventory": [
                    _sealed_inventory_row(
                        run_root=run_root,
                        path=replacement,
                        checkpoint_identity_sha256=replacement_identity,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
    )

    released = _release_tick_native_transient_storage_v1(
        context=context,
        prior_checkpoint_catalog={"3359": _checkpoint(old_checkpoint)},
        retained_checkpoint_catalog={},
        replacement_checkpoint={
            **_native_coordinate(3360),
            "checkpoint_identity_sha256": replacement_identity,
            "cell_bundles": [
                {"cell_id": "cell-0", "path": str(replacement)}
            ],
        },
        requested_checkpoint_ticks=[3360],
        release_source_import=False,
    )

    assert old_checkpoint.exists() is False
    assert replacement.is_file()
    assert [row["artifact_kind"] for row in released] == [
        "rolling_owner_checkpoint"
    ]


def test_tick_native_transient_release_rejects_sealed_unapproved_checkpoint_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    approved_root = batch_root / "checkpoint_native_v1" / "cell-0"
    unapproved_root = batch_root / "unapproved_checkpoints" / "cell-0"
    approved_root.mkdir(parents=True)
    unapproved_root.mkdir(parents=True)
    old_checkpoint = approved_root / "checkpoint_tick000000003359.acsnative"
    replacement = unapproved_root / "checkpoint_tick000000003360.acsnative"
    old_checkpoint.write_bytes(b"old-checkpoint")
    replacement.write_bytes(b"replacement")
    replacement_identity = "f" * 64
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cell_ids": ["cell-0"],
                "artifact_inventory_complete": True,
                "artifact_inventory": [
                    _sealed_inventory_row(
                        run_root=run_root,
                        path=replacement,
                        checkpoint_identity_sha256=replacement_identity,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
    )

    with pytest.raises(
        CampaignProviderError,
        match="replacement checkpoint escapes its physical batch roots",
    ):
        _release_tick_native_transient_storage_v1(
            context=context,
            prior_checkpoint_catalog={"3359": _checkpoint(old_checkpoint)},
            retained_checkpoint_catalog={},
            replacement_checkpoint={
                **_native_coordinate(3360),
                "checkpoint_identity_sha256": replacement_identity,
                "cell_bundles": [
                    {"cell_id": "cell-0", "path": str(replacement)}
                ],
            },
            requested_checkpoint_ticks=[3360],
            release_source_import=False,
        )

    assert old_checkpoint.is_file()
    assert replacement.is_file()


def test_tick_native_transient_release_rejects_symlinked_per_cell_native_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    target_root = run_root / "symlink-target" / "cell-0"
    target_root.mkdir(parents=True)
    native_root = batch_root / "checkpoint_native_v1"
    native_root.parent.mkdir(parents=True)
    native_root.symlink_to(target_root.parent, target_is_directory=True)
    replacement = native_root / "cell-0" / "checkpoint_tick000000003360.acsnative"
    replacement.write_bytes(b"replacement")
    replacement_identity = "f" * 64
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cell_ids": ["cell-0"],
                "artifact_inventory_complete": True,
                "artifact_inventory": [
                    _sealed_inventory_row(
                        run_root=run_root,
                        path=replacement,
                        checkpoint_identity_sha256=replacement_identity,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
    )

    with pytest.raises(
        CampaignProviderError,
        match="transient storage root cannot be a symlink",
    ):
        _release_tick_native_transient_storage_v1(
            context=context,
            prior_checkpoint_catalog={},
            retained_checkpoint_catalog={},
            replacement_checkpoint={
                **_native_coordinate(3360),
                "checkpoint_identity_sha256": replacement_identity,
                "cell_bundles": [
                    {"cell_id": "cell-0", "path": str(replacement)}
                ],
            },
            requested_checkpoint_ticks=[3360],
            release_source_import=True,
        )

    assert replacement.is_file()


def test_transient_release_rejects_path_only_replacement_inventory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    checkpoint_root = batch_root / "checkpoint_native_aggregate_v1"
    checkpoint_root.mkdir(parents=True)
    old_checkpoint = checkpoint_root / "tick-1120.acsaggregate"
    replacement = checkpoint_root / "tick-2240.acsaggregate"
    old_checkpoint.write_bytes(b"old")
    replacement.write_bytes(b"replacement")
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cell_ids": ["cell-0"],
                "artifact_inventory": [
                    {"path": str(replacement.relative_to(run_root))}
                ]
            }
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
    )

    with pytest.raises(
        CampaignProviderError,
        match="replacement seal is incomplete or stale",
    ):
        _release_tick_native_transient_storage_v1(
            context=context,
            prior_checkpoint_catalog={"1120": _checkpoint(old_checkpoint)},
            retained_checkpoint_catalog={},
            replacement_checkpoint={
                **_native_coordinate(2240),
                "checkpoint_identity_sha256": "f" * 64,
                "next_absolute_tick": 2240,
                "cell_bundles": [
                    {"cell_id": "cell-0", "path": str(replacement)}
                ],
            },
            requested_checkpoint_ticks=[2500, 5000],
            release_source_import=False,
        )

    assert old_checkpoint.exists()
    assert replacement.exists()


def test_source_import_release_requires_exact_digest_coverage(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    checkpoint_root = batch_root / "checkpoint_native_aggregate_v1"
    checkpoint_root.mkdir(parents=True)
    replacement = checkpoint_root / "tick-1120.acsaggregate"
    replacement.write_bytes(b"replacement")
    source_root = batch_root / "source_import_v1"
    source_root.mkdir(parents=True)
    (source_root / "projection.bin").write_bytes(b"source")
    replacement_identity = "f" * 64
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cell_ids": ["cell-0", "cell-1"],
                "artifact_inventory": [
                    _sealed_inventory_row(
                        run_root=run_root,
                        path=replacement,
                        checkpoint_identity_sha256=replacement_identity,
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    context = SimpleNamespace(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
    )

    with pytest.raises(
        CampaignProviderError,
        match="exact source digest coverage",
    ):
        _release_tick_native_transient_storage_v1(
            context=context,
            prior_checkpoint_catalog={},
            retained_checkpoint_catalog={},
            replacement_checkpoint={
                **_native_coordinate(1120),
                "checkpoint_identity_sha256": replacement_identity,
                "next_absolute_tick": 1120,
                "cell_bundles": [
                    {"cell_id": "cell-0", "path": str(replacement)},
                    {"cell_id": "cell-1", "path": str(replacement)},
                ],
                "source_import_projection_digest_sha256s": {
                    "cell-0": "1" * 64
                },
                "terminal_cell_digests": {
                    "cell-0": "2" * 64,
                    "cell-1": "3" * 64,
                },
            },
            requested_checkpoint_ticks=[2500, 5000],
            release_source_import=True,
        )

    assert source_root.exists()
    assert replacement.exists()


def test_lifecycle_checkpoint_catalog_accepts_tick_native_absolute_coordinates(
    tmp_path: Path,
) -> None:
    lifecycle = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=100_000_000,
        checkpoint_schedule=[5000],
        checkpoint_authority="absolute_tick",
    )
    role = lifecycle["checkpoints"]["roles"][0]

    catalog = _lifecycle_checkpoint_catalog(
        tmp_path,
        [
            {
                "checkpoint": {
                    **_native_coordinate(5000),
                    "checkpoint_file_sha256": "a" * 64,
                }
            }
        ],
        lifecycle,
    )

    assert catalog == [
        {
            "identity": role["identity"],
            "checkpoint_tick": 5000,
            "role": role["role"],
            "retention_class": role["retention_class"],
            "sha256": "a" * 64,
        }
    ]

    with pytest.raises(CampaignExecutionError, match="coordinates conflict"):
        _lifecycle_checkpoint_catalog(
            tmp_path,
            [
                {
                    "checkpoint": {
                        **_native_coordinate(5000),
                        "absolute_tick": 4999,
                    }
                }
            ],
            lifecycle,
        )


def test_lifecycle_checkpoint_catalog_expands_native_aggregate_schedule(
    tmp_path: Path,
) -> None:
    lifecycle = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=100_000_000,
        checkpoint_schedule=[16, 32, 40],
        checkpoint_authority="absolute_tick",
    )
    lifecycle["checkpoints"]["roles"] = [
        {
            "identity": f"checkpoint-tick-{tick:012d}",
            "checkpoint_tick": tick,
            "role": "terminal" if tick == 40 else "transit_durable",
            "retention_class": "R3",
        }
        for tick in (16, 32, 40)
    ]
    lifecycle.pop("contract_sha256", None)
    lifecycle = validate_run_artifact_lifecycle_contract_v1(lifecycle)

    catalog = _lifecycle_checkpoint_catalog(
        tmp_path,
        [
            {
                "checkpoint": {
                    "time_authority": "absolute_tick_v1",
                    "next_absolute_tick": 40,
                    "checkpoint_ticks": [16, 32, 40],
                    "checkpoint_file_sha256": "a" * 64,
                    "terminal_digest": "b" * 64,
                }
            }
        ],
        lifecycle,
    )

    assert catalog == [
        {
            "identity": "checkpoint-tick-000000000016",
            "checkpoint_tick": 16,
            "role": "transit_durable",
            "retention_class": "R3",
        },
        {
            "identity": "checkpoint-tick-000000000032",
            "checkpoint_tick": 32,
            "role": "transit_durable",
            "retention_class": "R3",
        },
        {
            "identity": "checkpoint-tick-000000000040",
            "checkpoint_tick": 40,
            "role": "terminal",
            "retention_class": "R3",
            "sha256": "a" * 64,
            "state_sha256": "b" * 64,
        },
    ]


@pytest.mark.parametrize(
    "aggregate_ticks, message",
    (
        ([32, 16, 40], "schedule is invalid"),
        ([16, 40], "schedule conflicts with lifecycle"),
        ([16, 32], "schedule conflicts with lifecycle"),
    ),
)
def test_lifecycle_checkpoint_catalog_rejects_invalid_native_aggregate_schedule(
    tmp_path: Path,
    aggregate_ticks: list[int],
    message: str,
) -> None:
    lifecycle = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=100_000_000,
        checkpoint_schedule=[16, 32, 40],
        checkpoint_authority="absolute_tick",
    )

    with pytest.raises(CampaignExecutionError, match=message):
        _lifecycle_checkpoint_catalog(
            tmp_path,
            [
                {
                    "checkpoint": {
                        "time_authority": "absolute_tick_v1",
                        "next_absolute_tick": 40,
                        "checkpoint_ticks": aggregate_ticks,
                    }
                }
            ],
            lifecycle,
        )


def test_batch_context_rejects_next_tick_only_before_checkpoint_persistence(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    batch_root.mkdir(parents=True)
    manifest_path = batch_root / "batch_manifest.json"
    original = {
        "status": "running",
        "campaign_id": "rolling-storage-fixture",
        "batch_id": "batch-0",
        "artifact_inventory": [],
    }
    manifest_path.write_text(json.dumps(original), encoding="utf-8")
    lifecycle = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=100_000_000,
        checkpoint_schedule=[5000],
        checkpoint_authority="absolute_tick",
    )
    context = BatchExecutionContextV1(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
        lifecycle_contract=lifecycle,
        provider_id="fixture-provider",
    )

    with pytest.raises(CampaignExecutionError, match="next_absolute_tick"):
        context.record_checkpoint(
            {
                "time_authority": "absolute_tick_v1",
                "next_tick": 5000,
                "cell_bundles": [{"cell_id": "cell-0", "path": "unused"}],
            }
        )

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == original


def test_lifecycle_collection_seals_large_batch_trees_as_bounded_inventory(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    trees = tuple(
        batch_root / name
        for name in ("cells", "event_history_v1", "source_import_v1")
    )
    for tree in trees:
        tree.mkdir(parents=True)
        for index in range(4097):
            (tree / f"child-{index:04d}.bin").write_bytes(b"x")

    plan, contract = _lifecycle_plan(run_root)
    inventory, complete = _collect_lifecycle_inventory_v1(
        run_root=run_root,
        batch_manifests=[
            {
                "batch_id": "batch-0",
                "provider_id": "fixture-provider",
                "status": "completed",
                "checkpoint": {
                    "checkpoint_identity_sha256": "e" * 64,
                },
                "artifact_inventory": [],
            }
        ],
        lifecycle=contract,
        terminal_status="completed",
    )

    assert complete is True
    assert len(inventory) == 3
    assert {row["artifact_type"] for row in inventory} == {"sealed_tree"}
    assert {row["path"] for row in inventory} == {
        "batches/batch-0/cells",
        "batches/batch-0/event_history_v1",
        "batches/batch-0/source_import_v1",
    }
    assert all(row["bytes"] == 4097 for row in inventory)

    disposition = build_artifact_disposition_manifest_v1(
        plan,
        run_root=run_root,
        terminal_status="completed",
        live_reference_status="none",
        audit_hold_status="not_required",
        qualification_hold_status="not_required",
        artifact_inventory=inventory,
        inventory_complete=True,
        max_objects=4096,
    )
    assert disposition["inventory"]["mode"] == "producer_declared"
    assert len(disposition["artifacts"]) == 3
    assert disposition["summary"]["produced_bytes"] == 3 * 4097


def test_batch_context_seals_shared_b136_aggregate_once_and_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    aggregate_root = (
        batch_root / "checkpoint_native_aggregate_v1" / "tick-2500.acsaggregate"
    )
    aggregate_root.parent.mkdir(parents=True)
    aggregate_root.write_bytes(b"aggregate")
    failed_aggregate = (
        batch_root / "checkpoint_native_aggregate_v1" / "tick-5000.acsaggregate"
    )
    failed_aggregate.write_bytes(b"failed-aggregate")
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "running",
                "campaign_id": "rolling-storage-fixture",
                "batch_id": "batch-0",
                "artifact_inventory": [],
            }
        ),
        encoding="utf-8",
    )
    lifecycle = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=100_000_000,
        checkpoint_schedule=[2500, 5000],
        checkpoint_authority="absolute_tick",
    )
    context = BatchExecutionContextV1(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
        lifecycle_contract=lifecycle,
        provider_id="fixture-provider",
    )
    real_builder = orchestration.build_sealed_artifact_inventory_row_v1
    builder_paths: list[Path] = []

    def counting_builder(**kwargs):
        builder_paths.append(Path(kwargs["path"]).resolve())
        return real_builder(**kwargs)

    monkeypatch.setattr(
        orchestration,
        "build_sealed_artifact_inventory_row_v1",
        counting_builder,
    )
    checkpoint_identity = "b" * 64
    bundles = [
        {"cell_id": f"cell-{index:03d}", "path": str(aggregate_root)}
        for index in range(136)
    ]
    context.record_checkpoint(
        {
            **_native_coordinate(2500),
            "checkpoint_identity_sha256": checkpoint_identity,
            "cell_bundles": bundles,
        }
    )

    assert builder_paths == [aggregate_root.resolve()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["artifact_inventory"]) == 1
    assert manifest["artifact_inventory"][0]["path"] == (
        "batches/batch-0/checkpoint_native_aggregate_v1/tick-2500.acsaggregate"
    )
    assert manifest["artifact_inventory"][0]["artifact_type"] == "file"

    context.record_checkpoint(
        {
            **_native_coordinate(2500),
            "checkpoint_identity_sha256": checkpoint_identity,
            "cell_bundles": bundles,
        }
    )
    assert builder_paths == [aggregate_root.resolve()]

    def fail_closed_builder(**kwargs):
        builder_paths.append(Path(kwargs["path"]).resolve())
        raise ArtifactLifecycleError("fixture aggregate changed while hashing")

    monkeypatch.setattr(
        orchestration,
        "build_sealed_artifact_inventory_row_v1",
        fail_closed_builder,
    )
    failed_bundles = [
        {"cell_id": f"cell-{index:03d}", "path": str(failed_aggregate)}
        for index in range(136)
    ]
    context.record_checkpoint(
        {
            **_native_coordinate(5000),
            "checkpoint_identity_sha256": "c" * 64,
            "cell_bundles": failed_bundles,
        }
    )

    assert builder_paths == [aggregate_root.resolve(), failed_aggregate.resolve()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["checkpoint"]["checkpoint_at_tick"] == 5000
    assert len(manifest["artifact_inventory"]) == 1
    assert manifest["artifact_inventory_complete"] is False
    assert manifest["artifact_inventory_failures"] == [
        {
            "path": str(failed_aggregate.resolve()),
            "error_type": "ArtifactLifecycleError",
        }
    ]
    _inventory, inventory_complete = _collect_lifecycle_inventory_v1(
        run_root=run_root,
        batch_manifests=[{**manifest, "status": "completed"}],
        lifecycle=lifecycle,
        terminal_status="completed",
    )
    assert inventory_complete is False


def test_batch_context_marks_empty_bundle_inventory_incomplete(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    batch_root.mkdir(parents=True)
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "running",
                "campaign_id": "rolling-storage-fixture",
                "batch_id": "batch-0",
                "artifact_inventory": [],
            }
        ),
        encoding="utf-8",
    )
    lifecycle = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=100_000_000,
        checkpoint_schedule=[5000],
        checkpoint_authority="absolute_tick",
    )
    context = BatchExecutionContextV1(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
        lifecycle_contract=lifecycle,
        provider_id="fixture-provider",
    )

    context.record_checkpoint(
        {
            **_native_coordinate(5000),
            "checkpoint_identity_sha256": "d" * 64,
            "cell_bundles": [],
        }
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact_inventory_complete"] is False
    assert manifest["artifact_inventory_failures"] == [
        {
            "path": "<checkpoint.cell_bundles>",
            "error_type": "MalformedCheckpointBundleInventory",
        }
    ]


def test_batch_context_inventories_world_objects_and_atomic_set_manifest(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    batch_root = run_root / "batches" / "batch-0"
    set_root = batch_root / "checkpoint_world_sharded_v1" / "tick-5000" / "set"
    objects_root = set_root / "objects"
    objects_root.mkdir(parents=True)
    object_paths = []
    for index in range(3):
        path = objects_root / f"world-{index:03d}.acsworld"
        path.write_bytes(f"world-{index}".encode("ascii"))
        object_paths.append(path)
    set_manifest = set_root / "checkpoint_set_manifest.json"
    set_manifest.write_text('{"status":"complete"}', encoding="ascii")
    manifest_path = batch_root / "batch_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "running",
                "campaign_id": "world-sharded-lifecycle-fixture",
                "batch_id": "batch-0",
                "artifact_inventory": [],
            }
        ),
        encoding="utf-8",
    )
    lifecycle = lifecycle_profile_contract_v1(
        SCIENCE_COMPACT,
        total_max_bytes=100_000_000,
        checkpoint_schedule=[5000],
        checkpoint_authority="absolute_tick",
    )
    context = BatchExecutionContextV1(
        run_root=run_root,
        batch_root=batch_root,
        manifest_path=manifest_path,
        lifecycle_contract=lifecycle,
        provider_id="fixture-provider",
    )

    context.record_checkpoint(
        {
            **_native_coordinate(5000),
            "checkpoint_identity_sha256": "e" * 64,
            "checkpoint_set_manifest_path": str(set_manifest),
            "cell_bundles": [
                {"cell_id": f"cell-{index:03d}", "path": str(path)}
                for index, path in enumerate(object_paths)
            ],
        }
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = manifest["artifact_inventory"]
    assert manifest.get("artifact_inventory_complete", True) is True
    assert len(inventory) == 4
    assert {row["artifact_family"] for row in inventory} == {
        "checkpoint.bundle_v2",
        "checkpoint.world_set_manifest_v1",
    }
    assert any(row["path"].endswith("checkpoint_set_manifest.json") for row in inventory)
