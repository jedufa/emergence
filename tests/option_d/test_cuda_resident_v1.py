from __future__ import annotations

from queue import Queue
import threading

import pytest
import torch

from engine.option_d import (
    CudaResidentEngine,
    DrawOperation,
    DrawPredicate,
    DrawSchedule,
    OptionDContractError,
    OptionDExecutionError,
    RNGStateBundle,
    execute_cpu,
    restore_checkpoint,
    state_from_seed,
)
from engine.option_d.cuda import _schedule_tensors


SOURCE = "test-current-source-cuda-resident"
ARTIFACT = "test-option-d-resident-artifact-v1"
CUDA_AVAILABLE = torch.cuda.is_available()


def _bundle(worlds: int) -> RNGStateBundle:
    return RNGStateBundle(
        tuple(state_from_seed(0xD4000000 + world) for world in range(worlds)),
        source_binding=SOURCE,
    )


def _raw_schedule(worlds: int, window: int) -> DrawSchedule:
    scheduled: list[tuple[DrawOperation, ...]] = []
    for world in range(worlds):
        operations: list[DrawOperation] = []
        for step in range(1 + (world + window) % 7):
            source = len(operations)
            operations.extend(
                (
                    DrawOperation.random(),
                    DrawOperation.getrandbits((world * 17 + step) % 65),
                    DrawOperation.randbelow(1 + (world + 1) * (step + 1)),
                    DrawOperation.randbelow(
                        65537,
                        predicate=DrawPredicate.random_lt(source, 0.35),
                    ),
                )
            )
        scheduled.append(tuple(operations))
    return DrawSchedule(tuple(scheduled))


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
@pytest.mark.parametrize("worlds", [1, 8, 56])
def test_resident_repeated_windows_match_cpu_and_repeat(worlds: int) -> None:
    initial = _bundle(worlds)
    first = CudaResidentEngine(initial)
    second = CudaResidentEngine(initial)
    expected = initial

    for window in range(3):
        schedule = _raw_schedule(worlds, window)
        cpu = execute_cpu(expected, schedule)
        first_result = first.execute(schedule)
        second_result = second.execute(schedule)

        assert first_result.material == cpu.material
        assert first_result.successor == cpu.successor
        assert second_result.material == first_result.material
        assert second_result.successor == first_result.successor
        expected = cpu.successor

    assert first.bundle == expected
    assert first.window_count == 3
    assert first.resident_device_bytes > 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_resident_contract_failure_does_not_publish_successor() -> None:
    engine = CudaResidentEngine(_bundle(2))
    before = engine.bundle
    schedule = DrawSchedule.same_for_all((DrawOperation.gauss(),), world_count=2)

    with pytest.raises(OptionDContractError, match="explicit Gaussian strategy"):
        engine.execute(schedule)

    assert engine.bundle == before
    assert engine.window_count == 0


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_native_device_failure_mutates_only_unpublished_successor() -> None:
    engine = CudaResidentEngine(_bundle(2))
    before_bundle = engine.bundle
    before_tensors = tuple(
        tensor.clone() for tensor in engine._state_tensors  # noqa: SLF001
    )
    schedule = DrawSchedule.same_for_all(
        (DrawOperation.raw32(), DrawOperation.raw32()),
        world_count=2,
    )
    schedule_tensors = list(_schedule_tensors(schedule, device=engine.device))
    schedule_tensors[0][:, 1] = 999

    outputs = engine._module.execute(  # noqa: SLF001
        *engine._state_tensors,  # noqa: SLF001
        *schedule_tensors,
        0,
    )

    assert outputs[7].cpu().tolist() == [1, 1]
    assert outputs[8].cpu().tolist() == [1, 1]
    assert engine.bundle == before_bundle
    assert engine.window_count == 0
    for original, current in zip(before_tensors, engine._state_tensors, strict=True):
        assert torch.equal(original, current)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_resident_state_rejects_non_owner_thread() -> None:
    engine = CudaResidentEngine(_bundle(1))
    failures: Queue[BaseException] = Queue()

    def inspect_from_foreign_thread() -> None:
        try:
            _ = engine.bundle
        except BaseException as error:  # pragma: no branch - expected boundary
            failures.put(error)

    thread = threading.Thread(target=inspect_from_foreign_thread)
    thread.start()
    thread.join(timeout=5)

    assert not thread.is_alive()
    error = failures.get_nowait()
    assert isinstance(error, OptionDExecutionError)
    assert "owner thread" in str(error)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_checkpoint_resume_in_fresh_resident_owner_matches_uninterrupted() -> None:
    initial = _bundle(8)
    prefix_schedule = _raw_schedule(8, 1)
    suffix_schedule = _raw_schedule(8, 2)
    uninterrupted_owner = CudaResidentEngine(initial)
    uninterrupted_prefix = uninterrupted_owner.execute(prefix_schedule)
    uninterrupted_suffix = uninterrupted_owner.execute(suffix_schedule)

    owner = CudaResidentEngine(initial)
    prefix = owner.execute(prefix_schedule)
    snapshot = owner.capture_checkpoint(artifact_binding=ARTIFACT)
    restored = restore_checkpoint(
        snapshot.data,
        expected_source_binding=SOURCE,
        expected_artifact_binding=ARTIFACT,
    )
    resumed_results: Queue[object] = Queue()

    def resume_in_fresh_owner() -> None:
        try:
            fresh_owner = CudaResidentEngine(restored)
            resumed_results.put(fresh_owner.execute(suffix_schedule))
        except BaseException as error:  # pragma: no cover - surfaced below
            resumed_results.put(error)

    thread = threading.Thread(target=resume_in_fresh_owner)
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive()
    resumed = resumed_results.get_nowait()
    if isinstance(resumed, BaseException):
        raise resumed

    assert prefix.material == uninterrupted_prefix.material
    assert resumed.material == uninterrupted_suffix.material
    assert resumed.successor == uninterrupted_suffix.successor
