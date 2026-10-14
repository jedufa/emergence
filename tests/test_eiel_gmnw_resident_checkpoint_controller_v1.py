from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from tools.scripts import eiel_gmnw_resident_checkpoint_controller_v1 as subject


CELL_IDS = ("cell-a", "cell-b")
PROVENANCE = {
    "schema_version": subject.PROVENANCE_SCHEMA_VERSION,
    "extension_sha256": "a" * 64,
    "configuration_sha256": "b" * 64,
    "checkpoint_contract_sha256": "c" * 64,
}


@dataclass
class _FakeOwner:
    generation: int
    values: list[int]
    continuation: bool = False
    released: bool = False


class _FakeRuntime:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[int, int, int | None, str]] = []
        self.prepared: list[_FakeOwner] = []
        self.released: list[_FakeOwner] = []
        self.acknowledgements: list[tuple[Path, str]] = []
        self.fresh_prepare_count = 0
        self.continuation_generations: list[int] = []
        self.fail_validation_at: int | None = None
        self.restore_generation_offset = 0
        self.restore_digest_offset = 0
        self.restore_digest_offset_at: int | None = None
        self.fail_restore_at: int | None = None
        self.fail_continuation_digest_at: int | None = None
        self.fail_acknowledgement_at: int | None = None

    def prepare_fresh(
        self, cell_ids: Sequence[str], *, run_root: Path, generations: int
    ) -> _FakeOwner:
        assert tuple(cell_ids) == CELL_IDS
        assert run_root.name == "segment-g000000"
        assert generations > 0
        self.fresh_prepare_count += 1
        return self._record(_FakeOwner(generation=0, values=[10, 20]))

    def execute_segment(
        self,
        prepared: _FakeOwner,
        *,
        start_generation: int,
        completed_generations: int,
        terminal_generation: int,
        run_id: str,
    ) -> Mapping[str, Any]:
        assert not prepared.released
        assert prepared.generation == start_generation
        prepared.values = [value + completed_generations for value in prepared.values]
        prepared.generation += completed_generations
        self.execute_calls.append(
            (start_generation, completed_generations, terminal_generation, run_id)
        )
        return {
            "completed_generations": completed_generations,
            "absolute_generation": prepared.generation,
        }

    def project_owner(
        self,
        prepared: _FakeOwner,
        result: Mapping[str, Any],
        *,
        completed_generations: int,
        cell_ids: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        assert result["completed_generations"] == completed_generations
        assert tuple(cell_ids) == CELL_IDS
        return [
            {
                "metadata": {"generation": prepared.generation, "tick": 0},
                "tensors": {"state": torch.tensor([value], dtype=torch.int64)},
            }
            for value in prepared.values
        ]

    def export_checkpoint(
        self,
        prepared: _FakeOwner,
        *,
        absolute_generation: int,
        cell_ids: Sequence[str],
    ) -> Mapping[str, Any]:
        assert absolute_generation == prepared.generation
        assert tuple(cell_ids) == CELL_IDS
        return {
            "schema_version": "fake_runtime_checkpoint_v1",
            "absolute_generation": absolute_generation,
            "values": list(prepared.values),
        }

    def validate_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
        *,
        absolute_generation: int,
        cell_ids: Sequence[str],
    ) -> Mapping[str, Any]:
        if absolute_generation == self.fail_validation_at:
            raise subject.ResidentCheckpointError("injected validation failure")
        if (
            checkpoint.get("schema_version") != "fake_runtime_checkpoint_v1"
            or checkpoint.get("absolute_generation") != absolute_generation
            or len(checkpoint.get("values", [])) != len(cell_ids)
        ):
            raise subject.ResidentCheckpointError("fake checkpoint mismatch")
        return checkpoint

    def acknowledge_completed_checkpoint(
        self,
        prepared: _FakeOwner,
        result: Mapping[str, Any],
        *,
        checkpoint_path: Path,
        checkpoint_sha256: str,
    ) -> Mapping[str, Any]:
        assert prepared.generation == result["absolute_generation"]
        if prepared.generation == self.fail_acknowledgement_at:
            raise subject.ResidentCheckpointError("injected acknowledgement failure")
        self.acknowledgements.append((checkpoint_path, checkpoint_sha256))
        return {"acknowledged": True}

    def prepare_continuation(
        self,
        cell_ids: Sequence[str],
        *,
        run_root: Path,
        owner_projections: Sequence[Mapping[str, Any]],
        initial_generation: int,
        generations: int,
    ) -> _FakeOwner:
        assert tuple(cell_ids) == CELL_IDS
        assert run_root.name.startswith(
            (
                f"segment-g{initial_generation:06d}",
                f"resume-g{initial_generation:06d}",
            )
        )
        assert generations >= initial_generation
        self.continuation_generations.append(initial_generation)
        values = [int(row["tensors"]["state"].item()) for row in owner_projections]
        return self._record(_FakeOwner(initial_generation, values, continuation=True))

    def restore_checkpoint(
        self,
        prepared: _FakeOwner,
        checkpoint: Mapping[str, Any],
        *,
        absolute_generation: int,
        cell_ids: Sequence[str],
    ) -> int:
        if absolute_generation == self.fail_restore_at:
            raise subject.ResidentCheckpointError("injected restore failure")
        self.validate_checkpoint(
            checkpoint,
            absolute_generation=absolute_generation,
            cell_ids=cell_ids,
        )
        prepared.generation = absolute_generation
        offset = (
            self.restore_digest_offset
            if self.restore_digest_offset_at in (None, absolute_generation)
            else 0
        )
        prepared.values = [int(value) + offset for value in checkpoint["values"]]
        return absolute_generation + self.restore_generation_offset

    def boundary_state_digests(
        self, prepared: _FakeOwner, *, cell_ids: Sequence[str]
    ) -> Sequence[str]:
        assert tuple(cell_ids) == CELL_IDS
        if (
            prepared.continuation
            and prepared.generation == self.fail_continuation_digest_at
        ):
            raise subject.ResidentCheckpointError("injected digest failure")
        return [
            hashlib.sha256(f"{prepared.generation}:{value}".encode("ascii")).hexdigest()
            for value in prepared.values
        ]

    def release(self, *owners: _FakeOwner) -> None:
        for owner in owners:
            owner.released = True
            self.released.append(owner)

    def _record(self, owner: _FakeOwner) -> _FakeOwner:
        self.prepared.append(owner)
        return owner


def _persist(
    tmp_path: Path,
    *,
    runtime: _FakeRuntime | None = None,
    generation: int = 2,
    schedule: tuple[int, ...] = (2, 4),
) -> tuple[_FakeRuntime, _FakeOwner, subject.DurableCheckpointReceiptV1]:
    runtime = runtime or _FakeRuntime()
    owner = runtime.prepare_fresh(
        CELL_IDS,
        run_root=tmp_path / "segment-g000000",
        generations=schedule[-1],
    )
    result = subject.execute_native_segment_v1(
        runtime,
        owner,
        start_generation=0,
        target_generation=generation,
        terminal_generation=schedule[-1],
        run_id="run-segment",
    )
    receipt = subject.persist_owner_checkpoint_v1(
        runtime,
        owner,
        result,
        checkpoint_root=tmp_path / "checkpoints",
        run_id="run-1",
        cell_ids=CELL_IDS,
        completed_segment_generations=generation,
        absolute_generation=generation,
        schedule=schedule,
        terminal_generation=schedule[-1],
        provenance=PROVENANCE,
    )
    return runtime, owner, receipt


def _rewrite_payload(
    receipt: subject.DurableCheckpointReceiptV1,
    mutate: Any,
) -> subject.DurableCheckpointReceiptV1:
    payload = torch.load(receipt.path, map_location="cpu", weights_only=False)
    mutate(payload)
    torch.save(payload, receipt.path)
    return replace(
        receipt,
        sha256=subject._file_sha256(receipt.path),
        bytes=receipt.path.stat().st_size,
    )


@pytest.mark.parametrize(
    ("values", "terminal", "message"),
    [
        ((), 4, "must not be empty"),
        ((2, 4), 0, "positive exact integer"),
        ((2, 4), True, "positive exact integer"),
        ("2,4", 4, "must be a sequence"),
        ((2, True, 4), 4, "exact integers"),
        ((0, 4), 4, "outside the terminal horizon"),
        ((2, 5), 4, "outside the terminal horizon"),
        ((2, 2, 4), 4, "strictly increasing"),
        ((3, 2, 4), 4, "strictly increasing"),
        ((1, 3), 4, "must end at terminal_generation"),
    ],
)
def test_schedule_validation_fails_closed(
    values: Any, terminal: int, message: str
) -> None:
    with pytest.raises(subject.ResidentCheckpointError, match=message):
        subject.validate_checkpoint_schedule_v1(values, terminal_generation=terminal)


def test_schedule_validation_and_hash_are_exact_and_stable() -> None:
    schedule = subject.validate_checkpoint_schedule_v1([1, 3, 4], terminal_generation=4)

    assert schedule == (1, 3, 4)
    assert subject.checkpoint_schedule_sha256_v1(
        schedule, terminal_generation=4
    ) == subject.checkpoint_schedule_sha256_v1(schedule, terminal_generation=4)


def test_native_segment_receives_terminal_horizon_and_rejects_overshoot(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()
    owner = runtime.prepare_fresh(
        CELL_IDS, run_root=tmp_path / "segment-g000000", generations=6
    )

    subject.execute_native_segment_v1(
        runtime,
        owner,
        start_generation=0,
        target_generation=2,
        terminal_generation=6,
        run_id="segment",
    )
    assert runtime.execute_calls == [(0, 2, 6, "segment")]

    with pytest.raises(subject.ResidentCheckpointError, match="boundaries are invalid"):
        subject.execute_native_segment_v1(
            runtime,
            owner,
            start_generation=2,
            target_generation=7,
            terminal_generation=6,
            run_id="overshoot",
        )
    assert len(runtime.execute_calls) == 1


def test_native_segment_passes_optional_run_control_without_changing_defaults(
    tmp_path: Path,
) -> None:
    class _InjectingRuntime(_FakeRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.run_controls: list[Any | None] = []

        def execute_segment(
            self,
            prepared: _FakeOwner,
            *,
            start_generation: int,
            completed_generations: int,
            terminal_generation: int,
            run_id: str,
            run_control: Any | None = None,
        ) -> Mapping[str, Any]:
            self.run_controls.append(run_control)
            return super().execute_segment(
                prepared,
                start_generation=start_generation,
                completed_generations=completed_generations,
                terminal_generation=terminal_generation,
                run_id=run_id,
            )

    runtime = _InjectingRuntime()
    owner = runtime.prepare_fresh(
        CELL_IDS, run_root=tmp_path / "segment-g000000", generations=4
    )
    control = object()

    subject.execute_native_segment_v1(
        runtime,
        owner,
        start_generation=0,
        target_generation=2,
        terminal_generation=4,
        run_id="segment",
        run_control=control,
    )

    assert runtime.run_controls == [control]


def test_checkpoint_publication_is_immutable_and_cleans_temporary_files(
    tmp_path: Path,
) -> None:
    runtime, owner, receipt = _persist(tmp_path)
    original = receipt.path.read_bytes()
    result = {"completed_generations": 2, "absolute_generation": 2}

    reused = subject.persist_owner_checkpoint_v1(
        runtime,
        owner,
        result,
        checkpoint_root=tmp_path / "checkpoints",
        run_id="run-1",
        cell_ids=CELL_IDS,
        completed_segment_generations=2,
        absolute_generation=2,
        schedule=(2, 4),
        terminal_generation=4,
        provenance=PROVENANCE,
    )

    assert reused == receipt
    assert receipt.path.read_bytes() == original
    assert not list(receipt.path.parent.glob(".*.tmp"))

    owner.values[0] += 1
    with pytest.raises(
        subject.ResidentCheckpointError, match="digest receipt mismatch"
    ):
        subject.persist_owner_checkpoint_v1(
            runtime,
            owner,
            result,
            checkpoint_root=tmp_path / "checkpoints",
            run_id="run-1",
            cell_ids=CELL_IDS,
            completed_segment_generations=2,
            absolute_generation=2,
            schedule=(2, 4),
            terminal_generation=4,
            provenance=PROVENANCE,
        )


def test_first_checkpoint_orphan_is_adopted_by_a_fresh_controller(
    tmp_path: Path,
) -> None:
    _runtime, _owner, orphan = _persist(tmp_path)
    original_bytes = orphan.path.read_bytes()
    assert not (orphan.path.parent / subject.LATEST_POINTER_NAME).exists()

    runtime = _FakeRuntime()
    owner = runtime.prepare_fresh(
        CELL_IDS, run_root=tmp_path / "segment-g000000", generations=4
    )
    result = subject.execute_native_segment_v1(
        runtime,
        owner,
        start_generation=0,
        target_generation=2,
        terminal_generation=4,
        run_id="replacement-segment",
    )
    adopted = subject.persist_owner_checkpoint_v1(
        runtime,
        owner,
        result,
        checkpoint_root=tmp_path / "checkpoints",
        run_id="run-1",
        cell_ids=CELL_IDS,
        completed_segment_generations=2,
        absolute_generation=2,
        schedule=(2, 4),
        terminal_generation=4,
        provenance=PROVENANCE,
    )

    assert adopted == orphan
    assert adopted.path.read_bytes() == original_bytes
    assert runtime.acknowledgements == []


def test_corrupted_first_checkpoint_orphan_is_rejected(tmp_path: Path) -> None:
    runtime, owner, orphan = _persist(tmp_path)
    corrupt_bytes = b"not-a-checkpoint"
    orphan.path.write_bytes(corrupt_bytes)
    result = {"completed_generations": 2, "absolute_generation": 2}

    with pytest.raises(subject.ResidentCheckpointError, match="cannot be reloaded"):
        subject.persist_owner_checkpoint_v1(
            runtime,
            owner,
            result,
            checkpoint_root=tmp_path / "checkpoints",
            run_id="run-1",
            cell_ids=CELL_IDS,
            completed_segment_generations=2,
            absolute_generation=2,
            schedule=(2, 4),
            terminal_generation=4,
            provenance=PROVENANCE,
        )

    assert orphan.path.read_bytes() == corrupt_bytes
    assert runtime.acknowledgements == []


def test_nested_checkpoint_directories_and_publications_are_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fsynced_directories: list[Path] = []
    monkeypatch.setattr(
        subject,
        "_fsync_directory",
        lambda path: fsynced_directories.append(Path(path)),
    )
    run_root = tmp_path / "nested" / "run"

    _runtime, _owner, receipt = _persist(run_root)
    checkpoint_root = run_root / "checkpoints"

    assert checkpoint_root.is_dir()
    assert fsynced_directories == [
        tmp_path,
        tmp_path / "nested",
        run_root,
        checkpoint_root,
    ]

    fsynced_directories.clear()
    subject.commit_latest_checkpoint_v1(
        checkpoint_root,
        receipt,
        run_id="run-1",
        cell_ids=CELL_IDS,
        schedule=(2, 4),
        terminal_generation=4,
        expected_previous_generation=0,
    )
    assert fsynced_directories == [checkpoint_root, checkpoint_root]


def test_checkpoint_publication_failure_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _FakeRuntime()
    owner = runtime.prepare_fresh(
        CELL_IDS, run_root=tmp_path / "segment-g000000", generations=4
    )
    result = subject.execute_native_segment_v1(
        runtime,
        owner,
        start_generation=0,
        target_generation=2,
        terminal_generation=4,
        run_id="segment",
    )

    def fail_link(_source: Path, _target: Path) -> None:
        raise OSError("injected link failure")

    monkeypatch.setattr(subject.os, "link", fail_link)
    with pytest.raises(OSError, match="injected link failure"):
        subject.persist_owner_checkpoint_v1(
            runtime,
            owner,
            result,
            checkpoint_root=tmp_path / "checkpoints",
            run_id="run-1",
            cell_ids=CELL_IDS,
            completed_segment_generations=2,
            absolute_generation=2,
            schedule=(2, 4),
            terminal_generation=4,
            provenance=PROVENANCE,
        )

    checkpoint_root = tmp_path / "checkpoints"
    assert not (checkpoint_root / "g000002-owner.pt").exists()
    assert not list(checkpoint_root.glob(".*.tmp"))


def test_reload_accepts_exact_checkpoint_and_invokes_runtime_validation(
    tmp_path: Path,
) -> None:
    runtime, _owner, receipt = _persist(tmp_path)

    payload = subject.reload_validate_checkpoint_v1(
        runtime,
        receipt,
        cell_ids=CELL_IDS,
        schedule=(2, 4),
        terminal_generation=4,
        provenance=PROVENANCE,
    )

    assert payload["absolute_generation"] == 2
    assert payload["cell_ids"] == list(CELL_IDS)


def test_reload_rejects_byte_hash_mismatch(tmp_path: Path) -> None:
    runtime, _owner, receipt = _persist(tmp_path)
    with receipt.path.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(subject.ResidentCheckpointError, match="bytes do not match"):
        subject.reload_validate_checkpoint_v1(
            runtime,
            receipt,
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            provenance=PROVENANCE,
        )


@pytest.mark.parametrize("corrupt_bytes", [b"", b"not-a-torch-checkpoint"])
def test_reload_wraps_all_deserialization_failures(
    tmp_path: Path, corrupt_bytes: bytes
) -> None:
    path = tmp_path / "corrupt.pt"
    path.write_bytes(corrupt_bytes)
    receipt = subject.DurableCheckpointReceiptV1(
        path=path,
        sha256=subject._file_sha256(path),
        bytes=len(corrupt_bytes),
        run_id="run-1",
        absolute_generation=2,
        schedule_sha256=subject.checkpoint_schedule_sha256_v1(
            (2, 4), terminal_generation=4
        ),
        boundary_state_digests=("1" * 64, "2" * 64),
    )

    with pytest.raises(subject.ResidentCheckpointError, match="cannot be reloaded"):
        subject.reload_validate_checkpoint_v1(
            _FakeRuntime(),
            receipt,
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            provenance=PROVENANCE,
        )


def test_reload_rejects_provenance_mismatch(tmp_path: Path) -> None:
    runtime, _owner, receipt = _persist(tmp_path)

    with pytest.raises(subject.ResidentCheckpointError, match="provenance mismatch"):
        subject.reload_validate_checkpoint_v1(
            runtime,
            receipt,
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            provenance={**PROVENANCE, "configuration_sha256": "c" * 64},
        )


@pytest.mark.parametrize(
    "provenance",
    [
        {key: value for key, value in PROVENANCE.items() if key != "schema_version"},
        {**PROVENANCE, "schema_version": "future"},
        {
            key: value
            for key, value in PROVENANCE.items()
            if key != "configuration_sha256"
        },
        {**PROVENANCE, "extension_sha256": "A" * 64},
    ],
)
def test_persist_requires_complete_canonical_provenance_schema(
    tmp_path: Path, provenance: Mapping[str, Any]
) -> None:
    runtime = _FakeRuntime()
    owner = runtime.prepare_fresh(
        CELL_IDS, run_root=tmp_path / "segment-g000000", generations=4
    )
    result = subject.execute_native_segment_v1(
        runtime,
        owner,
        start_generation=0,
        target_generation=2,
        terminal_generation=4,
        run_id="segment",
    )

    with pytest.raises(subject.ResidentCheckpointError, match="provenance"):
        subject.persist_owner_checkpoint_v1(
            runtime,
            owner,
            result,
            checkpoint_root=tmp_path / "checkpoints",
            run_id="run-1",
            cell_ids=CELL_IDS,
            completed_segment_generations=2,
            absolute_generation=2,
            schedule=(2, 4),
            terminal_generation=4,
            provenance=provenance,
        )


def test_reload_requires_projection_cell_binding(tmp_path: Path) -> None:
    runtime, _owner, receipt = _persist(tmp_path)
    changed = _rewrite_payload(
        receipt,
        lambda payload: payload.__setitem__(
            "owner_projection_cell_ids", list(reversed(CELL_IDS))
        ),
    )

    with pytest.raises(
        subject.ResidentCheckpointError, match="identity or provenance mismatch"
    ):
        subject.reload_validate_checkpoint_v1(
            runtime,
            changed,
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            provenance=PROVENANCE,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.__setitem__("schema_version", "future"),
        lambda payload: payload.__setitem__("unexpected", True),
        lambda payload: payload.pop("gmnw_multicell_checkpoint"),
    ],
)
def test_reload_rejects_schema_changes(tmp_path: Path, mutate: Any) -> None:
    runtime, _owner, receipt = _persist(tmp_path)
    changed = _rewrite_payload(receipt, mutate)

    with pytest.raises(subject.ResidentCheckpointError, match="schema mismatch"):
        subject.reload_validate_checkpoint_v1(
            runtime,
            changed,
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            provenance=PROVENANCE,
        )


def test_reload_rejects_boundary_digest_receipt_mismatch(tmp_path: Path) -> None:
    runtime, _owner, receipt = _persist(tmp_path)
    changed = _rewrite_payload(
        receipt,
        lambda payload: payload["boundary_state_digests"].__setitem__(0, "f" * 64),
    )

    with pytest.raises(
        subject.ResidentCheckpointError, match="digest receipt mismatch"
    ):
        subject.reload_validate_checkpoint_v1(
            runtime,
            changed,
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            provenance=PROVENANCE,
        )


def test_pointer_roundtrip_is_atomic_and_bound_to_checkpoint(tmp_path: Path) -> None:
    _runtime, _owner, receipt = _persist(tmp_path)
    checkpoint_root = tmp_path / "checkpoints"

    pointer = subject.commit_latest_checkpoint_v1(
        checkpoint_root,
        receipt,
        run_id="run-1",
        cell_ids=CELL_IDS,
        schedule=(2, 4),
        terminal_generation=4,
        expected_previous_generation=0,
    )
    loaded_pointer, loaded_receipt = subject.load_latest_checkpoint_v1(
        checkpoint_root,
        run_id="run-1",
        cell_ids=CELL_IDS,
        schedule=(2, 4),
        terminal_generation=4,
    )

    assert loaded_pointer == pointer
    assert loaded_receipt == receipt
    assert pointer["checkpoint_path"] == "g000002-owner.pt"
    assert not list(checkpoint_root.glob(".*.tmp"))


@pytest.mark.parametrize("receipt_field", ["run_id", "sha256", "bytes"])
def test_pointer_publication_revalidates_receipt_identity_and_bytes(
    tmp_path: Path, receipt_field: str
) -> None:
    _runtime, _owner, receipt = _persist(tmp_path)
    replacement: Any = {
        "run_id": "other-run",
        "sha256": "0" * 64,
        "bytes": receipt.bytes + 1,
    }[receipt_field]
    changed = replace(receipt, **{receipt_field: replacement})
    checkpoint_root = tmp_path / "checkpoints"

    with pytest.raises(
        subject.ResidentCheckpointError,
        match="not publishable at the requested frontier",
    ):
        subject.commit_latest_checkpoint_v1(
            checkpoint_root,
            changed,
            run_id="run-1",
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            expected_previous_generation=0,
        )

    assert not (checkpoint_root / subject.LATEST_POINTER_NAME).exists()


def test_latest_pointer_falls_back_to_previous_after_corruption(tmp_path: Path) -> None:
    execution = subject.execute_checkpointed_ub100_v1(
        _FakeRuntime(),
        run_root=tmp_path,
        run_id="run-1",
        cell_ids=CELL_IDS,
        provenance=PROVENANCE,
        checkpoint_generations=(2, 4, 6),
        terminal_generation=6,
    )
    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / subject.LATEST_POINTER_NAME).write_bytes(b"{corrupt")

    pointer, receipt = subject.load_latest_checkpoint_v1(
        checkpoint_root,
        run_id="run-1",
        cell_ids=CELL_IDS,
        schedule=(2, 4, 6),
        terminal_generation=6,
    )

    assert execution.latest_pointer["absolute_generation"] == 6
    assert pointer["absolute_generation"] == 4
    assert receipt.absolute_generation == 4
    assert receipt.path.name == "g000004-owner.pt"


def test_checkpoint_retention_keeps_only_latest_and_previous_payloads(
    tmp_path: Path,
) -> None:
    subject.execute_checkpointed_ub100_v1(
        _FakeRuntime(),
        run_root=tmp_path,
        run_id="run-1",
        cell_ids=CELL_IDS,
        provenance=PROVENANCE,
        checkpoint_generations=(2, 4, 6),
        terminal_generation=6,
    )
    checkpoint_root = tmp_path / "checkpoints"

    assert sorted(path.name for path in checkpoint_root.glob("g*-owner.pt")) == [
        "g000004-owner.pt",
        "g000006-owner.pt",
    ]
    previous = json.loads(
        (checkpoint_root / subject.PREVIOUS_POINTER_NAME).read_text(encoding="ascii")
    )
    latest = json.loads(
        (checkpoint_root / subject.LATEST_POINTER_NAME).read_text(encoding="ascii")
    )
    assert (previous["absolute_generation"], latest["absolute_generation"]) == (4, 6)


def test_checkpoint_retention_can_keep_every_scheduled_payload(tmp_path: Path) -> None:
    subject.execute_checkpointed_ub100_v1(
        _FakeRuntime(),
        run_root=tmp_path,
        run_id="retain-all",
        cell_ids=CELL_IDS,
        provenance=PROVENANCE,
        checkpoint_generations=(2, 4, 6),
        terminal_generation=6,
        retain_all_checkpoints=True,
    )
    root = tmp_path / "checkpoints"

    assert sorted(path.name for path in root.glob("g*-owner.pt")) == [
        "g000002-owner.pt",
        "g000004-owner.pt",
        "g000006-owner.pt",
    ]


def test_pointer_publication_rejects_cas_and_monotonicity_violations(
    tmp_path: Path,
) -> None:
    runtime, owner, receipt_2 = _persist(tmp_path, schedule=(2, 4, 6))
    checkpoint_root = tmp_path / "checkpoints"
    subject.commit_latest_checkpoint_v1(
        checkpoint_root,
        receipt_2,
        run_id="run-1",
        cell_ids=CELL_IDS,
        schedule=(2, 4, 6),
        terminal_generation=6,
        expected_previous_generation=0,
    )
    result_4 = subject.execute_native_segment_v1(
        runtime,
        owner,
        start_generation=2,
        target_generation=4,
        terminal_generation=6,
        run_id="segment-4",
    )
    receipt_4 = subject.persist_owner_checkpoint_v1(
        runtime,
        owner,
        result_4,
        checkpoint_root=checkpoint_root,
        run_id="run-1",
        cell_ids=CELL_IDS,
        completed_segment_generations=2,
        absolute_generation=4,
        schedule=(2, 4, 6),
        terminal_generation=6,
        provenance=PROVENANCE,
    )
    latest_path = checkpoint_root / subject.LATEST_POINTER_NAME
    generation_2_pointer = latest_path.read_bytes()

    with pytest.raises(
        subject.ResidentCheckpointError, match="predecessor changed before publication"
    ):
        subject.commit_latest_checkpoint_v1(
            checkpoint_root,
            receipt_4,
            run_id="run-1",
            cell_ids=CELL_IDS,
            schedule=(2, 4, 6),
            terminal_generation=6,
            expected_previous_generation=0,
        )
    assert latest_path.read_bytes() == generation_2_pointer

    subject.commit_latest_checkpoint_v1(
        checkpoint_root,
        receipt_4,
        run_id="run-1",
        cell_ids=CELL_IDS,
        schedule=(2, 4, 6),
        terminal_generation=6,
        expected_previous_generation=2,
    )
    generation_4_pointer = latest_path.read_bytes()
    with pytest.raises(
        subject.ResidentCheckpointError,
        match="not publishable at the requested frontier",
    ):
        subject.commit_latest_checkpoint_v1(
            checkpoint_root,
            receipt_2,
            run_id="run-1",
            cell_ids=CELL_IDS,
            schedule=(2, 4, 6),
            terminal_generation=6,
            expected_previous_generation=4,
        )
    assert latest_path.read_bytes() == generation_4_pointer


def test_latest_pointer_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    _runtime, _owner, receipt = _persist(tmp_path)
    checkpoint_root = tmp_path / "checkpoints"
    pointer = dict(
        subject.commit_latest_checkpoint_v1(
            checkpoint_root,
            receipt,
            run_id="run-1",
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            expected_previous_generation=0,
        )
    )
    pointer["checkpoint_sha256"] = "0" * 64
    (checkpoint_root / "latest_checkpoint.json").write_text(
        json.dumps(pointer), encoding="ascii"
    )

    with pytest.raises(subject.ResidentCheckpointError, match="size or path mismatch"):
        subject.load_latest_checkpoint_v1(
            checkpoint_root,
            run_id="run-1",
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
        )


def test_pointer_publication_rejects_checkpoint_outside_root(tmp_path: Path) -> None:
    _runtime, _owner, receipt = _persist(tmp_path / "source")

    with pytest.raises(subject.ResidentCheckpointError, match="outside"):
        subject.commit_latest_checkpoint_v1(
            tmp_path / "different-root",
            receipt,
            run_id="run-1",
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            expected_previous_generation=0,
        )


@pytest.mark.parametrize("unsafe_path", ["../outside.pt", "/tmp/outside.pt"])
def test_latest_pointer_rejects_lexical_path_traversal(
    tmp_path: Path, unsafe_path: str
) -> None:
    _runtime, _owner, receipt = _persist(tmp_path)
    checkpoint_root = tmp_path / "checkpoints"
    pointer = dict(
        subject.commit_latest_checkpoint_v1(
            checkpoint_root,
            receipt,
            run_id="run-1",
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            expected_previous_generation=0,
        )
    )
    pointer["checkpoint_path"] = unsafe_path
    (checkpoint_root / "latest_checkpoint.json").write_text(
        json.dumps(pointer), encoding="ascii"
    )

    with pytest.raises(subject.ResidentCheckpointError, match="path is unsafe"):
        subject.load_latest_checkpoint_v1(
            checkpoint_root,
            run_id="run-1",
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
        )


def test_latest_pointer_rejects_symlink_escape(tmp_path: Path) -> None:
    _runtime, _owner, receipt = _persist(tmp_path)
    checkpoint_root = tmp_path / "checkpoints"
    outside = tmp_path / "outside.pt"
    outside.write_bytes(receipt.path.read_bytes())
    (checkpoint_root / "escape.pt").symlink_to(outside)
    pointer = dict(
        subject.commit_latest_checkpoint_v1(
            checkpoint_root,
            receipt,
            run_id="run-1",
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
            expected_previous_generation=0,
        )
    )
    pointer["checkpoint_path"] = "escape.pt"
    (checkpoint_root / "latest_checkpoint.json").write_text(
        json.dumps(pointer), encoding="ascii"
    )

    with pytest.raises(subject.ResidentCheckpointError, match="escapes its root"):
        subject.load_latest_checkpoint_v1(
            checkpoint_root,
            run_id="run-1",
            cell_ids=CELL_IDS,
            schedule=(2, 4),
            terminal_generation=4,
        )


def test_fresh_owner_restore_requires_exact_boundary_digests(tmp_path: Path) -> None:
    runtime, _owner, receipt = _persist(tmp_path)
    payload = subject.reload_validate_checkpoint_v1(
        runtime,
        receipt,
        cell_ids=CELL_IDS,
        schedule=(2, 4),
        terminal_generation=4,
        provenance=PROVENANCE,
    )

    restored = subject.restore_fresh_owner_v1(
        runtime,
        payload,
        run_root=tmp_path / "segment-g000002",
        cell_ids=CELL_IDS,
        terminal_generation=4,
    )
    assert restored.generation == 2
    assert restored.values == [12, 22]

    runtime.restore_digest_offset = 1
    with pytest.raises(subject.ResidentCheckpointError, match="boundary state differs"):
        subject.restore_fresh_owner_v1(
            runtime,
            payload,
            run_root=tmp_path / "segment-g000002",
            cell_ids=CELL_IDS,
            terminal_generation=4,
        )
    assert runtime.released[-1].released


def test_fresh_owner_restore_uses_new_directory_after_interruption(
    tmp_path: Path,
) -> None:
    runtime, _owner, receipt = _persist(tmp_path)
    payload = subject.reload_validate_checkpoint_v1(
        runtime,
        receipt,
        cell_ids=CELL_IDS,
        schedule=(2, 4),
        terminal_generation=4,
        provenance=PROVENANCE,
    )
    occupied = tmp_path / "segment-g000002"
    occupied.mkdir()
    assert subject._available_restore_run_root_v1(occupied) == (
        tmp_path / "segment-g000002-attempt-0002"
    )

    restored = subject.restore_fresh_owner_v1(
        runtime,
        payload,
        run_root=occupied,
        cell_ids=CELL_IDS,
        terminal_generation=4,
    )

    assert restored.generation == 2


def test_fresh_owner_restore_releases_wrong_generation(tmp_path: Path) -> None:
    runtime, _owner, receipt = _persist(tmp_path)
    payload = subject.reload_validate_checkpoint_v1(
        runtime,
        receipt,
        cell_ids=CELL_IDS,
        schedule=(2, 4),
        terminal_generation=4,
        provenance=PROVENANCE,
    )
    runtime.restore_generation_offset = 1

    with pytest.raises(
        subject.ResidentCheckpointError, match="wrong absolute generation"
    ):
        subject.restore_fresh_owner_v1(
            runtime,
            payload,
            run_root=tmp_path / "segment-g000002",
            cell_ids=CELL_IDS,
            terminal_generation=4,
        )
    assert runtime.released[-1].released


def test_interruption_preserves_previous_latest_pointer(tmp_path: Path) -> None:
    runtime = _FakeRuntime()
    runtime.restore_digest_offset = 1
    runtime.restore_digest_offset_at = 4

    with pytest.raises(subject.ResidentCheckpointError, match="boundary state differs"):
        subject.execute_checkpointed_ub100_v1(
            runtime,
            run_root=tmp_path,
            run_id="run-1",
            cell_ids=CELL_IDS,
            provenance=PROVENANCE,
            checkpoint_generations=(2, 4, 6),
            terminal_generation=6,
        )

    pointer, receipt = subject.load_latest_checkpoint_v1(
        tmp_path / "checkpoints",
        run_id="run-1",
        cell_ids=CELL_IDS,
        schedule=(2, 4, 6),
        terminal_generation=6,
    )
    assert pointer["absolute_generation"] == 2
    assert receipt.path.name == "g000002-owner.pt"
    assert (tmp_path / "checkpoints/g000004-owner.pt").is_file()
    assert runtime.released[-1].released


def test_pointer_failure_prevents_acknowledgement_and_releases_each_owner_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _FakeRuntime()

    def fail_pointer_publication(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        raise subject.ResidentCheckpointError("injected pointer failure")

    monkeypatch.setattr(
        subject, "commit_latest_checkpoint_v1", fail_pointer_publication
    )
    with pytest.raises(subject.ResidentCheckpointError, match="pointer failure"):
        subject.execute_checkpointed_ub100_v1(
            runtime,
            run_root=tmp_path,
            run_id="run-1",
            cell_ids=CELL_IDS,
            provenance=PROVENANCE,
            checkpoint_generations=(2,),
            terminal_generation=2,
        )

    assert runtime.acknowledgements == []
    assert not (tmp_path / "checkpoints" / subject.LATEST_POINTER_NAME).exists()
    assert len(runtime.prepared) == 2
    assert [
        sum(released is owner for released in runtime.released)
        for owner in runtime.prepared
    ] == [1, 1]


@pytest.mark.parametrize(
    ("failure_attribute", "message"),
    [
        ("fail_restore_at", "restore failure"),
        ("fail_continuation_digest_at", "digest failure"),
        ("fail_acknowledgement_at", "acknowledgement failure"),
    ],
)
def test_restore_digest_and_ack_failures_release_each_owner_exactly_once(
    tmp_path: Path, failure_attribute: str, message: str
) -> None:
    runtime = _FakeRuntime()
    setattr(runtime, failure_attribute, 2)

    with pytest.raises(subject.ResidentCheckpointError, match=message):
        subject.execute_checkpointed_ub100_v1(
            runtime,
            run_root=tmp_path,
            run_id="run-1",
            cell_ids=CELL_IDS,
            provenance=PROVENANCE,
            checkpoint_generations=(2,),
            terminal_generation=2,
        )

    assert len(runtime.prepared) == 2
    assert [
        sum(released is owner for released in runtime.released)
        for owner in runtime.prepared
    ] == [1, 1]


def test_exact_small_segmented_run_restores_fresh_owner_at_every_boundary(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime()

    execution = subject.execute_checkpointed_ub100_v1(
        runtime,
        run_root=tmp_path,
        run_id="run-1",
        cell_ids=CELL_IDS,
        provenance=PROVENANCE,
        checkpoint_generations=(2, 4, 6),
        terminal_generation=6,
    )

    assert [
        (start, count) for start, count, _terminal, _run_id in runtime.execute_calls
    ] == [
        (0, 2),
        (2, 2),
        (4, 2),
    ]
    assert execution.prepared.generation == 6
    assert execution.prepared.values == [16, 26]
    assert not execution.prepared.released
    assert len(execution.segment_results) == 3
    assert [
        receipt.absolute_generation for receipt in execution.checkpoint_receipts
    ] == [
        2,
        4,
        6,
    ]
    assert execution.latest_pointer["absolute_generation"] == 6
    assert len(runtime.acknowledgements) == 3
    assert len(runtime.prepared) == 4
    assert all(owner.released for owner in runtime.prepared[:-1])


def test_resume_from_latest_does_not_rerun_completed_segments(tmp_path: Path) -> None:
    initial_runtime, initial_owner, receipt = _persist(
        tmp_path, generation=2, schedule=(2, 4, 6)
    )
    payload = subject.reload_validate_checkpoint_v1(
        initial_runtime,
        receipt,
        cell_ids=CELL_IDS,
        schedule=(2, 4, 6),
        terminal_generation=6,
        provenance=PROVENANCE,
    )
    verified = subject.restore_fresh_owner_v1(
        initial_runtime,
        payload,
        run_root=tmp_path / "segment-g000002",
        cell_ids=CELL_IDS,
        terminal_generation=6,
    )
    subject.commit_latest_checkpoint_v1(
        tmp_path / "checkpoints",
        receipt,
        run_id="run-1",
        cell_ids=CELL_IDS,
        schedule=(2, 4, 6),
        terminal_generation=6,
        expected_previous_generation=0,
    )
    initial_runtime.release(initial_owner, verified)

    resumed_runtime = _FakeRuntime()
    execution = subject.resume_checkpointed_ub100_v1(
        resumed_runtime,
        run_root=tmp_path,
        run_id="run-1",
        cell_ids=CELL_IDS,
        provenance=PROVENANCE,
        checkpoint_generations=(2, 4, 6),
        terminal_generation=6,
    )

    assert resumed_runtime.fresh_prepare_count == 0
    assert resumed_runtime.continuation_generations == [2, 4, 6]
    assert [
        (start, count, terminal)
        for start, count, terminal, _run_id in resumed_runtime.execute_calls
    ] == [(2, 2, 6), (4, 2, 6)]
    assert execution.prepared.generation == 6
    assert execution.prepared.values == [16, 26]
    assert [
        receipt.absolute_generation for receipt in execution.checkpoint_receipts
    ] == [
        2,
        4,
        6,
    ]


def test_resume_recovers_after_unpublished_next_checkpoint(tmp_path: Path) -> None:
    interrupted_runtime = _FakeRuntime()
    interrupted_runtime.restore_digest_offset = 1
    interrupted_runtime.restore_digest_offset_at = 4
    with pytest.raises(subject.ResidentCheckpointError, match="boundary state differs"):
        subject.execute_checkpointed_ub100_v1(
            interrupted_runtime,
            run_root=tmp_path,
            run_id="run-1",
            cell_ids=CELL_IDS,
            provenance=PROVENANCE,
            checkpoint_generations=(2, 4, 6),
            terminal_generation=6,
        )

    resumed = subject.resume_checkpointed_ub100_v1(
        _FakeRuntime(),
        run_root=tmp_path,
        run_id="run-1",
        cell_ids=CELL_IDS,
        provenance=PROVENANCE,
        checkpoint_generations=(2, 4, 6),
        terminal_generation=6,
    )
    assert resumed.prepared.generation == 6
