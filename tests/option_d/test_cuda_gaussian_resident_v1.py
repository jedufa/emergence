from __future__ import annotations

import pytest
import torch

from engine.option_d.benchmark import (
    CURRENT_C5_TAPE_CAPACITY,
    _allocate_current_helper_tape,
    _candidate_sample,
    _current_c5_admission,
    _current_schedule,
    _refill_current_helper_tape,
)
from engine.option_d import (
    CudaResidentEngine,
    DrawOperation,
    DrawSchedule,
    RNGStateBundle,
    execute_cpu,
    restore_checkpoint,
    state_from_seed,
)
from engine.option_d.workloads import c4_scalar_schedule


SOURCE = "test-current-source-cuda-gaussian-resident"
ARTIFACT = "test-option-d-gaussian-resident-artifact-v1"
CUDA_AVAILABLE = torch.cuda.is_available()


def _bundle(worlds: int) -> RNGStateBundle:
    return RNGStateBundle(
        tuple(state_from_seed(0xD4500000 + world) for world in range(worlds)),
        source_binding=SOURCE,
    )


def _c4_schedule(worlds: int, births: int, window: int = 0) -> DrawSchedule:
    rates = tuple(
        tuple(0.01 + 0.19 * ((world + birth + window) % 11) / 10.0 for birth in range(births))
        for world in range(worlds)
    )
    memory = tuple(
        tuple(0.002 + 0.048 * ((world * 3 + birth + window) % 7) / 6.0 for birth in range(births))
        for world in range(worlds)
    )
    return c4_scalar_schedule(rates, memory)


def _c4_variable_schedule(births_by_world: tuple[int, ...]) -> DrawSchedule:
    rates = tuple(
        tuple(0.01 + 0.19 * ((world + birth) % 11) / 10.0 for birth in range(births))
        for world, births in enumerate(births_by_world)
    )
    memory = tuple(
        tuple(0.002 + 0.048 * ((world * 3 + birth) % 7) / 6.0 for birth in range(births))
        for world, births in enumerate(births_by_world)
    )
    return c4_scalar_schedule(rates, memory)


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
@pytest.mark.parametrize(("worlds", "births"), [(1, 1), (8, 7), (48, 64)])
def test_software_gaussian_current_c4_schedule_exact(worlds: int, births: int) -> None:
    initial = _bundle(worlds)
    schedule = _c4_schedule(worlds, births)
    expected = execute_cpu(initial, schedule)
    first = CudaResidentEngine(initial).execute(
        schedule,
        gaussian_strategy="software",
    )
    second = CudaResidentEngine(initial).execute(
        schedule,
        gaussian_strategy="software",
    )

    assert first.material == expected.material
    assert first.successor == expected.successor
    assert second.material == first.material
    assert second.successor == first.successor


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_software_gaussian_repeated_resident_windows_exact() -> None:
    initial = _bundle(8)
    owner = CudaResidentEngine(initial)
    expected = initial

    for window in range(3):
        schedule = _c4_schedule(8, 9, window)
        cpu = execute_cpu(expected, schedule)
        actual = owner.execute(schedule, gaussian_strategy="software")
        assert actual.material == cpu.material
        assert actual.successor == cpu.successor
        expected = cpu.successor

    assert owner.bundle == expected
    assert owner.window_count == 3


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_software_gaussian_variable_worlds_are_independent() -> None:
    initial = _bundle(48)
    schedule = _c4_variable_schedule(tuple(world % 8 for world in range(48)))
    combined = CudaResidentEngine(initial).execute(
        schedule,
        gaussian_strategy="software",
    )
    expected = execute_cpu(initial, schedule)

    assert combined.material == expected.material
    assert combined.successor == expected.successor
    for world in (0, 1, 17, 47):
        isolated = CudaResidentEngine(
            RNGStateBundle((initial.worlds[world],), source_binding=SOURCE)
        ).execute(
            DrawSchedule((schedule.worlds[world],)),
            gaussian_strategy="software",
        )
        assert combined.material.worlds[world] == isolated.material.worlds[0]
        assert combined.successor.worlds[world] == isolated.successor.worlds[0]


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_current_host_tape_baseline_reuses_suffix_exactly() -> None:
    initial = _bundle(8)
    owner = CudaResidentEngine(initial)
    tape_tensors = _allocate_current_helper_tape(
        owner,
        capacity=2048,
    )
    pointers = tuple(tensor.data_ptr() for tensor in tape_tensors)
    _profile, helper_receipt = _refill_current_helper_tape(owner, tape_tensors)
    expected = initial

    assert tuple(tensor.data_ptr() for tensor in tape_tensors) == pointers
    assert helper_receipt["pointer_identity"]["all_exact"] is True
    assert helper_receipt["absolute_raw_draws_preserved"] is True
    assert helper_receipt["semantic_draws_preserved"] is True
    for window in range(3):
        schedule = _c4_schedule(8, 9, window)
        cpu = execute_cpu(expected, schedule)
        actual, _ = _candidate_sample(
            owner,
            schedule,
            gaussian_strategy="host_tape_baseline",
            tape_tensors=tape_tensors,
        )
        assert actual.material == cpu.material
        assert actual.successor == cpu.successor
        expected = cpu.successor


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_current_c5_helper_frozen_trajectory_refills_at_tick_25() -> None:
    initial = _bundle(48)
    owner = CudaResidentEngine(initial)
    schedule = _current_schedule(48, 64)
    tape_tensors = _allocate_current_helper_tape(
        owner,
        capacity=CURRENT_C5_TAPE_CAPACITY,
    )
    pointers = tuple(tensor.data_ptr() for tensor in tape_tensors)
    _record, receipt = _refill_current_helper_tape(owner, tape_tensors)
    expected = initial
    first_refill_tick: int | None = None

    assert receipt["pointer_identity"]["all_exact"] is True
    for tick in range(1, 26):
        cpu = execute_cpu(expected, schedule)
        actual, _ = _candidate_sample(
            owner,
            schedule,
            gaussian_strategy="host_tape_baseline",
            tape_tensors=tape_tensors,
        )
        assert actual.material == cpu.material
        assert actual.successor == cpu.successor
        expected = cpu.successor
        if tick % 5 == 0:
            admission = _current_c5_admission(
                owner.bundle,
                base_raw_words=tuple(
                    int(value) for value in tape_tensors[2].cpu().tolist()
                ),
                capacity=CURRENT_C5_TAPE_CAPACITY,
            )
            if admission["refill_required"]:
                first_refill_tick = tick
                break

    assert first_refill_tick == 25
    counters_before = tuple(world.counters for world in owner.bundle.worlds)
    _record, receipt = _refill_current_helper_tape(owner, tape_tensors)
    assert tuple(tensor.data_ptr() for tensor in tape_tensors) == pointers
    assert tuple(world.counters for world in owner.bundle.worlds) == counters_before
    assert receipt["pointer_identity"]["all_exact"] is True


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_software_gaussian_cache_checkpoint_fresh_owner_exact() -> None:
    initial = _bundle(4)
    prefix_schedule = DrawSchedule.same_for_all(
        (DrawOperation.gauss(), DrawOperation.random()),
        world_count=4,
    )
    suffix_schedule = DrawSchedule.same_for_all(
        (
            DrawOperation.gauss(),
            DrawOperation.gauss(mu=-2.5, sigma=0.125),
            DrawOperation.getrandbits(64),
        ),
        world_count=4,
    )
    cpu_prefix = execute_cpu(initial, prefix_schedule)
    cpu_suffix = execute_cpu(cpu_prefix.successor, suffix_schedule)

    owner = CudaResidentEngine(initial)
    prefix = owner.execute(prefix_schedule, gaussian_strategy="software")
    snapshot = owner.capture_checkpoint(artifact_binding=ARTIFACT)
    restored = restore_checkpoint(
        snapshot.data,
        expected_source_binding=SOURCE,
        expected_artifact_binding=ARTIFACT,
    )
    fresh_owner = CudaResidentEngine(restored)
    suffix = fresh_owner.execute(suffix_schedule, gaussian_strategy="software")

    assert prefix.material == cpu_prefix.material
    assert suffix.material == cpu_suffix.material
    assert suffix.successor == cpu_suffix.successor
