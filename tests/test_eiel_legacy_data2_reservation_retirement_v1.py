from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from engine.campaigns.canonical import canonical_sha256, pretty_json
from engine.campaigns import legacy_data2_reservation_retirement as retirement


BOOT_ID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture(autouse=True)
def _stable_observers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retirement, "_read_boot_id", lambda: BOOT_ID)
    monkeypatch.setattr(retirement, "_probe_pid", lambda _pid: "absent")
    monkeypatch.setattr(retirement, "_now", lambda: "2026-08-28T01:02:03.000000Z")
    monkeypatch.setattr(retirement, "_fault_point", lambda _name: None)


def _legacy_reservation(
    *,
    run_root: str,
    contract_sha256: str,
    pid: int,
    planned_bytes: int | None = 4096,
    remaining_bytes: int = 1024,
) -> dict[str, object]:
    reservation_id = retirement.legacy_data2_reservation_id_v1(
        run_root=run_root,
        lifecycle_contract_sha256=contract_sha256,
    )
    core: dict[str, object] = {
        "schema_version": retirement.LEGACY_STORAGE_GUARD_SCHEMA_VERSION,
        "reservation_id": reservation_id,
        "run_root": run_root,
        "lifecycle_contract_sha256": contract_sha256,
        "planned_bytes": planned_bytes,
        "remaining_bytes": remaining_bytes,
        "pid": pid,
        "status": "active",
        "updated_at_utc": "2026-08-20T12:34:56.000000Z",
    }
    return {**core, "reservation_sha256": canonical_sha256(core)}


def _write_reservation(
    root: Path,
    *,
    suffix: str,
    pid: int,
    contract_digit: str,
) -> tuple[str, dict[str, object]]:
    row = _legacy_reservation(
        run_root=f"/tmp/emergence-test/runtime/tests/{suffix}",
        contract_sha256=contract_digit * 64,
        pid=pid,
    )
    filename = f"{row['reservation_id']}.json"
    (root / filename).write_text(pretty_json(row), encoding="utf-8")
    return filename, row


def _control_paths(root: Path, filenames: list[str]) -> dict[str, Path]:
    transaction_id = retirement._transaction_id(sorted(filenames))
    return retirement._control_paths(root, transaction_id)


def _preview_plan(root: Path, filenames: list[str]) -> str:
    report = retirement.retire_legacy_data2_reservations_v1(
        filenames,
        unsafe_test_only_reservation_root=root,
    )
    assert report["status"] == "DRY_RUN_READY"
    return str(report["plan_sha256"])


def test_dry_run_is_default_and_does_not_create_retirement_control(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    first, _ = _write_reservation(
        root, suffix="run-a", pid=7_000_001, contract_digit="a"
    )
    second, _ = _write_reservation(
        root, suffix="run-b", pid=7_000_002, contract_digit="b"
    )

    report = retirement.retire_legacy_data2_reservations_v1(
        [second, first],
        unsafe_test_only_reservation_root=root,
    )

    assert report["status"] == "DRY_RUN_READY"
    assert report["reservation_filenames"] == sorted([first, second])
    assert report["all_or_nothing_validated_before_first_rename"] is True
    assert report["plan_sha256"] == retirement._retirement_plan_sha256(
        root=root,
        transaction_id=report["transaction_id"],
        filenames=report["reservation_filenames"],
        targets=report["targets"],
    )
    assert report["relay_absence_used_as_authority"] is False
    assert report["independent_root_protection_not_evaluated"] is True
    assert report["report_sha256"] == canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    assert (root / first).is_file()
    assert (root / second).is_file()
    assert (root / ".lock").is_file()
    assert not (root / retirement.RETIREMENT_CONTROL_DIRECTORY).exists()
    authorization = report["authorization"]
    assert authorization["boot_id"] == BOOT_ID
    assert len(authorization["pre_lock_liveness"]) == 2
    assert len(authorization["under_lock_liveness"]) == 2
    assert {
        row["state"]
        for row in (
            authorization["pre_lock_liveness"] + authorization["under_lock_liveness"]
        )
    } == {"absent"}


def test_exact_legacy_null_plan_is_accepted_only_with_zero_remaining(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    row = _legacy_reservation(
        run_root="/tmp/emergence-test/runtime/tests/legacy-unbounded",
        contract_sha256="d" * 64,
        pid=7_000_000,
        planned_bytes=None,
        remaining_bytes=0,
    )
    filename = f"{row['reservation_id']}.json"
    (root / filename).write_text(pretty_json(row), encoding="utf-8")

    report = retirement.retire_legacy_data2_reservations_v1(
        [filename], unsafe_test_only_reservation_root=root
    )
    assert report["status"] == "DRY_RUN_READY"
    assert report["targets"][0]["planned_bytes"] is None


def test_apply_writes_hash_bound_internal_receipt_and_replays_idempotently(
    tmp_path: Path,
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-c", pid=7_000_003, contract_digit="c"
    )
    plan_sha256 = _preview_plan(root, [filename])

    receipt = retirement.retire_legacy_data2_reservations_v1(
        [filename],
        apply=True,
        expected_plan_sha256=plan_sha256,
        unsafe_test_only_reservation_root=root,
    )

    assert receipt["status"] == "APPLIED"
    assert receipt["retirement_scope"] == "legacy_reservation_metadata_files_only"
    assert receipt["relay_absence_used_as_authority"] is False
    assert receipt["independent_root_protection_not_evaluated"] is True
    assert receipt["plan_sha256"] == plan_sha256
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )
    assert not (root / filename).exists()
    paths = _control_paths(root, [filename])
    assert paths["receipt"].is_file()
    assert not paths["transaction"].exists()

    replay = retirement.retire_legacy_data2_reservations_v1(
        [filename],
        apply=True,
        expected_plan_sha256=plan_sha256,
        unsafe_test_only_reservation_root=root,
    )
    assert replay == receipt


def test_apply_rejects_missing_or_wrong_prior_plan(tmp_path: Path) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-plan-required", pid=7_000_070, contract_digit="d"
    )

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="expected-plan-sha256.*prior dry run",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            unsafe_test_only_reservation_root=root,
        )

    plan_sha256 = _preview_plan(root, [filename])
    wrong_plan_sha256 = "f" * 64
    assert wrong_plan_sha256 != plan_sha256
    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="exact plan SHA-256.*prior dry run",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            expected_plan_sha256=wrong_plan_sha256,
            unsafe_test_only_reservation_root=root,
        )

    assert (root / filename).is_file()
    assert not (root / retirement.RETIREMENT_CONTROL_DIRECTORY).exists()


def test_apply_rejects_valid_target_changed_after_preview(tmp_path: Path) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, original = _write_reservation(
        root, suffix="run-plan-changed", pid=7_000_071, contract_digit="e"
    )
    plan_sha256 = _preview_plan(root, [filename])

    changed = dict(original)
    changed["remaining_bytes"] = 512
    changed["updated_at_utc"] = "2026-08-28T01:02:04.000000Z"
    changed["reservation_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "reservation_sha256"}
    )
    (root / filename).write_text(pretty_json(changed), encoding="utf-8")

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="exact plan SHA-256.*prior dry run",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )

    assert (root / filename).is_file()
    assert not (root / retirement.RETIREMENT_CONTROL_DIRECTORY).exists()


def test_plan_hash_excludes_fresh_liveness_observation_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-plan-stable", pid=7_000_072, contract_digit="f"
    )
    first = retirement.retire_legacy_data2_reservations_v1(
        [filename], unsafe_test_only_reservation_root=root
    )
    monkeypatch.setattr(retirement, "_now", lambda: "2026-08-28T01:02:05.000000Z")
    second = retirement.retire_legacy_data2_reservations_v1(
        [filename], unsafe_test_only_reservation_root=root
    )

    assert second["plan_sha256"] == first["plan_sha256"]
    assert second["report_sha256"] != first["report_sha256"]


def test_all_targets_validate_before_first_rename(tmp_path: Path) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    first, _ = _write_reservation(
        root, suffix="run-d", pid=7_000_004, contract_digit="d"
    )
    second, second_row = _write_reservation(
        root, suffix="run-e", pid=7_000_005, contract_digit="e"
    )
    tampered = dict(second_row)
    tampered["unexpected"] = True
    (root / second).write_text(pretty_json(tampered), encoding="utf-8")

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="fields differ",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [first, second],
            apply=True,
            expected_plan_sha256="0" * 64,
            unsafe_test_only_reservation_root=root,
        )

    assert (root / first).is_file()
    assert (root / second).is_file()
    assert not (root / retirement.RETIREMENT_CONTROL_DIRECTORY).exists()


@pytest.mark.parametrize("bad_name", ["../" + "a" * 24 + ".json", "a.json"])
def test_explicit_filename_must_be_exact_basename(
    tmp_path: Path, bad_name: str
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="exact 24-hex",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [bad_name], unsafe_test_only_reservation_root=root
        )


@pytest.mark.parametrize("alias_kind", ["symlink", "hardlink"])
def test_reservation_aliases_are_rejected(tmp_path: Path, alias_kind: str) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    row = _legacy_reservation(
        run_root="/tmp/emergence-test/runtime/tests/run-f",
        contract_sha256="f" * 64,
        pid=7_000_006,
    )
    filename = f"{row['reservation_id']}.json"
    backing = tmp_path / "backing.json"
    backing.write_text(pretty_json(row), encoding="utf-8")
    if alias_kind == "symlink":
        (root / filename).symlink_to(backing)
    else:
        os.link(backing, root / filename)

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="regular, non-symlink, and single-link",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename], unsafe_test_only_reservation_root=root
        )


def test_reservation_root_alias_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    filename, _ = _write_reservation(
        root, suffix="run-10", pid=7_000_010, contract_digit="1"
    )

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="real directory|contains an alias",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename], unsafe_test_only_reservation_root=alias
        )


def test_canonical_lock_alias_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-11", pid=7_000_011, contract_digit="2"
    )
    backing = tmp_path / "lock"
    backing.touch()
    (root / ".lock").symlink_to(backing)

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="canonical reservation lock",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename], unsafe_test_only_reservation_root=root
        )


@pytest.mark.parametrize("tamper", ["sha256", "identity", "filename", "extra"])
def test_exact_legacy_binding_rejects_tampering(tmp_path: Path, tamper: str) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, original = _write_reservation(
        root, suffix=f"tamper-{tamper}", pid=7_000_020, contract_digit="3"
    )
    row = dict(original)
    selected_filename = filename
    if tamper == "sha256":
        row["reservation_sha256"] = "0" * 64
    elif tamper == "identity":
        row["reservation_id"] = "b" * 24
        core = {key: value for key, value in row.items() if key != "reservation_sha256"}
        row["reservation_sha256"] = canonical_sha256(core)
        selected_filename = "b" * 24 + ".json"
        (root / filename).unlink()
    elif tamper == "filename":
        selected_filename = "c" * 24 + ".json"
        (root / filename).unlink()
    else:
        row["unexpected"] = True
    (root / selected_filename).write_text(pretty_json(row), encoding="utf-8")

    with pytest.raises(retirement.LegacyData2ReservationRetirementError):
        retirement.retire_legacy_data2_reservations_v1(
            [selected_filename], unsafe_test_only_reservation_root=root
        )


def test_live_pid_blocks_before_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-live-a", pid=7_000_030, contract_digit="4"
    )
    plan_sha256 = _preview_plan(root, [filename])
    monkeypatch.setattr(retirement, "_probe_pid", lambda _pid: "live")

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="not absent",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )
    assert (root / filename).is_file()
    assert not (root / retirement.RETIREMENT_CONTROL_DIRECTORY).exists()


def test_pid_becoming_live_under_lock_blocks_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-live-b", pid=7_000_031, contract_digit="5"
    )
    plan_sha256 = _preview_plan(root, [filename])
    states = iter(["absent", "live"])
    monkeypatch.setattr(retirement, "_probe_pid", lambda _pid: next(states))

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="not absent",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )
    assert (root / filename).is_file()
    assert not (root / retirement.RETIREMENT_CONTROL_DIRECTORY).exists()


def test_second_live_pid_under_lock_blocks_every_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    first, _ = _write_reservation(
        root, suffix="run-live-c", pid=7_000_033, contract_digit="6"
    )
    second, _ = _write_reservation(
        root, suffix="run-live-d", pid=7_000_034, contract_digit="7"
    )
    plan_sha256 = _preview_plan(root, [first, second])
    states = iter(["absent", "absent", "absent", "live"])
    monkeypatch.setattr(retirement, "_probe_pid", lambda _pid: next(states))

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="not absent",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [first, second],
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )
    assert (root / first).is_file()
    assert (root / second).is_file()
    assert not (root / retirement.RETIREMENT_CONTROL_DIRECTORY).exists()


def test_boot_change_between_absence_checks_blocks_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-boot", pid=7_000_032, contract_digit="6"
    )
    plan_sha256 = _preview_plan(root, [filename])
    other_boot = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    boots = iter([BOOT_ID, BOOT_ID, other_boot, other_boot])
    monkeypatch.setattr(retirement, "_read_boot_id", lambda: next(boots))

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="between PID absence checks",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )
    assert (root / filename).is_file()
    assert not (root / retirement.RETIREMENT_CONTROL_DIRECTORY).exists()


def test_crash_after_partial_staging_recovers_same_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    first, _ = _write_reservation(
        root, suffix="run-stage-a", pid=7_000_040, contract_digit="7"
    )
    second, _ = _write_reservation(
        root, suffix="run-stage-b", pid=7_000_041, contract_digit="8"
    )
    filenames = sorted([first, second])
    plan_sha256 = _preview_plan(root, filenames)

    def crash(point: str) -> None:
        if point == f"after_stage:{filenames[0]}":
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(retirement, "_fault_point", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        retirement.retire_legacy_data2_reservations_v1(
            filenames,
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )

    paths = _control_paths(root, filenames)
    assert paths["journal"].is_file()
    assert (paths["stage"] / filenames[0]).is_file()
    assert (root / filenames[1]).is_file()

    monkeypatch.setattr(retirement, "_fault_point", lambda _point: None)
    receipt = retirement.retire_legacy_data2_reservations_v1(
        filenames,
        apply=True,
        expected_plan_sha256=plan_sha256,
        unsafe_test_only_reservation_root=root,
    )
    assert receipt["status"] == "APPLIED"
    assert all(not (root / filename).exists() for filename in filenames)
    assert not paths["transaction"].exists()


def test_crash_after_deletion_recovers_receipt_without_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-delete", pid=7_000_050, contract_digit="9"
    )
    plan_sha256 = _preview_plan(root, [filename])

    def crash(point: str) -> None:
        if point == "after_delete_all_before_receipt":
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(retirement, "_fault_point", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )

    paths = _control_paths(root, [filename])
    assert not (root / filename).exists()
    assert paths["journal"].is_file()
    assert not paths["receipt"].exists()

    monkeypatch.setattr(retirement, "_fault_point", lambda _point: None)
    receipt = retirement.retire_legacy_data2_reservations_v1(
        [filename],
        apply=True,
        expected_plan_sha256=plan_sha256,
        unsafe_test_only_reservation_root=root,
    )
    assert receipt["status"] == "APPLIED"
    assert paths["receipt"].is_file()
    assert not paths["transaction"].exists()


def test_post_deletion_recovery_ignores_reused_recorded_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-delete-pid-reuse", pid=7_000_051, contract_digit="a"
    )
    plan_sha256 = _preview_plan(root, [filename])

    def crash(point: str) -> None:
        if point == "after_delete_all_before_receipt":
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(retirement, "_fault_point", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )

    paths = _control_paths(root, [filename])
    journal = json.loads(paths["journal"].read_text())
    assert journal["state"] == "STAGED"
    assert not (root / filename).exists()
    assert not (paths["stage"] / filename).exists()

    monkeypatch.setattr(retirement, "_fault_point", lambda _point: None)
    monkeypatch.setattr(
        retirement,
        "_probe_pid",
        lambda _pid: (_ for _ in ()).throw(AssertionError("PID was re-probed")),
    )
    receipt = retirement.retire_legacy_data2_reservations_v1(
        [filename],
        apply=True,
        expected_plan_sha256=plan_sha256,
        unsafe_test_only_reservation_root=root,
    )

    assert receipt["status"] == "APPLIED"
    assert receipt["authorization"] == journal["authorization"]
    assert paths["receipt"].is_file()
    assert not paths["transaction"].exists()


def test_recomputed_but_semantically_tampered_journal_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-journal", pid=7_000_055, contract_digit="b"
    )
    plan_sha256 = _preview_plan(root, [filename])

    def crash(point: str) -> None:
        if point == "after_prepared_journal":
            raise RuntimeError("simulated crash")

    monkeypatch.setattr(retirement, "_fault_point", crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )
    paths = _control_paths(root, [filename])
    value = json.loads(paths["journal"].read_text(encoding="utf-8"))
    value["state"] = "UNSAFE"
    value["journal_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "journal_sha256"}
    )
    paths["journal"].write_text(pretty_json(value), encoding="utf-8")
    monkeypatch.setattr(retirement, "_fault_point", lambda _point: None)

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="state is unsupported",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )
    assert (root / filename).is_file()


def test_tampered_applied_receipt_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "reservations"
    root.mkdir()
    filename, _ = _write_reservation(
        root, suffix="run-receipt", pid=7_000_060, contract_digit="a"
    )
    plan_sha256 = _preview_plan(root, [filename])
    retirement.retire_legacy_data2_reservations_v1(
        [filename],
        apply=True,
        expected_plan_sha256=plan_sha256,
        unsafe_test_only_reservation_root=root,
    )
    paths = _control_paths(root, [filename])
    value = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    value["independent_root_protection_not_evaluated"] = False
    value["receipt_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "receipt_sha256"}
    )
    paths["receipt"].write_text(pretty_json(value), encoding="utf-8")

    with pytest.raises(
        retirement.LegacyData2ReservationRetirementError,
        match="independent root protection|SHA-256 mismatch",
    ):
        retirement.retire_legacy_data2_reservations_v1(
            [filename],
            apply=True,
            expected_plan_sha256=plan_sha256,
            unsafe_test_only_reservation_root=root,
        )
