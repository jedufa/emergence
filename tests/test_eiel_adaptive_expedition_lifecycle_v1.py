from __future__ import annotations

from pathlib import Path
import json

import pytest

from eiel_composition.adaptive_expedition import lifecycle as lifecycle_module
from eiel_composition.adaptive_expedition.lifecycle import (
    ExpeditionLifecycleError,
    close_expedition_lifecycle_v1,
    open_expedition_lifecycle_v1,
    release_expedition_storage_v1,
)
from tests.test_eiel_adaptive_expedition_coordinator_v1 import _manifest
from tests.test_eiel_adaptive_expedition_founder_loop_v1 import _founder_policy


def test_closeout_recovery_registers_verified_receipts_under_the_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(_founder_policy())
    run_root = tmp_path / "outer"
    context = open_expedition_lifecycle_v1(manifest=manifest, run_root=run_root)
    original = lifecycle_module.register_pending_durable_archive_v1

    def crash_after_receipts(_manifest_path, **_options):
        raise RuntimeError("crash before pending archive registration")

    monkeypatch.setattr(
        lifecycle_module,
        "register_pending_durable_archive_v1",
        crash_after_receipts,
    )
    with pytest.raises(RuntimeError, match="crash before pending"):
        close_expedition_lifecycle_v1(context)
    assert (run_root / "artifact_disposition_manifest.json").is_file()
    assert (run_root / "storage_disposition_receipt.json").is_file()

    calls: list[bool] = []

    def record(manifest_path, **options):
        calls.append(options.get("_transaction_fence_held") is True)
        return original(manifest_path, **options)

    monkeypatch.setattr(
        lifecycle_module,
        "register_pending_durable_archive_v1",
        record,
    )
    recovered = open_expedition_lifecycle_v1(
        manifest=manifest,
        run_root=run_root,
    )
    closeout = close_expedition_lifecycle_v1(recovered)
    assert closeout["terminal_status"] == "completed"
    assert calls == [True]

    # Registration and verified closeout are idempotent on another recovery.
    replay = close_expedition_lifecycle_v1(recovered)
    assert replay == closeout
    assert calls == [True, True]


def test_outer_inventory_excess_fails_closed_at_a_fixed_object_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(_founder_policy())
    run_root = tmp_path / "outer-cap"
    context = open_expedition_lifecycle_v1(manifest=manifest, run_root=run_root)
    monkeypatch.setattr(
        lifecycle_module,
        "MAX_OUTER_LIFECYCLE_OBJECTS_V1",
        3,
    )
    extra = run_root / "control" / "extra.json"
    extra.parent.mkdir()
    extra.write_text("{}", encoding="ascii")
    try:
        with pytest.raises(ExpeditionLifecycleError, match="fixed object cap"):
            close_expedition_lifecycle_v1(context)
    finally:
        release_expedition_storage_v1(context)
    assert not (run_root / "artifact_disposition_manifest.json").exists()
    assert not (run_root / "storage_disposition_receipt.json").exists()


def test_append_only_control_members_are_one_faithful_sealed_tree_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(_founder_policy())
    run_root = tmp_path / "outer-sealed-control"
    context = open_expedition_lifecycle_v1(manifest=manifest, run_root=run_root)
    control = run_root / "control"
    control.mkdir()
    for ordinal in range(32):
        (control / f"{ordinal:08d}.json").write_text("{}", encoding="ascii")
    monkeypatch.setattr(lifecycle_module, "MAX_OUTER_LIFECYCLE_OBJECTS_V1", 4)

    closeout = close_expedition_lifecycle_v1(context)
    disposition = json.loads(
        Path(closeout["artifact_disposition_manifest_path"]).read_bytes()
    )
    control_rows = [row for row in disposition["artifacts"] if row["path"] == "control"]
    assert len(control_rows) == 1
    assert control_rows[0]["artifact_type"] == "sealed_tree"
    assert not any(
        row["path"].startswith("control/") for row in disposition["artifacts"]
    )
