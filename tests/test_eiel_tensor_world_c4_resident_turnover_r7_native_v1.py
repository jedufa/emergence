"""Focused five-tick CUDA parity and rollback for the R7 R5 harness."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_resident_birth_r2 import (
    _half_tensors,
    _inherit_tensors,
    _minus_one_tensors,
    _one_tensors,
    _primary_tensors,
    _zero_tensors,
)
from engine.tensor_world.c4_resident_mutation_r4 import (
    _replay_tensors,
    _scalar_rng_tensors,
)
from engine.tensor_world.c4_resident_population_r5 import (
    execute_c4_resident_population_r5_cuda_v1,
    make_c4_resident_population_r5_staging_v1,
)
from engine.tensor_world.c4_resident_turnover_r3 import _death_authoritative
from engine.tensor_world.c4_resident_turnover_r7_native import (
    C4_RESIDENT_TURNOVER_R7_NATIVE_DECISION_V1,
    C4ResidentTurnoverR7NativeFixtureV1,
    c4_resident_turnover_r7_native_source_receipt_v1,
    execute_c4_resident_turnover_r7_cuda_tick_v1,
    execute_c4_resident_turnover_r7_oracle_tick_v1,
    make_c4_resident_turnover_r7_native_fixture_v1,
    prepare_c4_resident_turnover_r7_tick_v1,
)


def _move_dataclass_tensors(value: object, device: torch.device) -> object:
    clone = copy.deepcopy(value)

    def move(owner: object) -> None:
        if not is_dataclass(owner):
            return
        for field in fields(owner):
            item = getattr(owner, field.name)
            if isinstance(item, torch.Tensor):
                setattr(owner, field.name, item.to(device=device))
            elif is_dataclass(item):
                move(item)

    move(clone)
    return clone


def _authoritative(fixture) -> list[torch.Tensor]:
    bundle = fixture.bundle
    return [
        *_primary_tensors(bundle), *_death_authoritative(bundle),
        *_inherit_tensors(bundle), *_zero_tensors(bundle),
        *_minus_one_tensors(bundle), *_half_tensors(bundle), *_one_tensors(bundle),
        *_scalar_rng_tensors(bundle), *_replay_tensors(fixture.replay),
    ]


def test_r7_native_receipt_is_capacity_safe_mutating_harness_only() -> None:
    fixture = make_c4_resident_turnover_r7_native_fixture_v1()
    receipt = c4_resident_turnover_r7_native_source_receipt_v1()

    assert fixture.bundle.batch == 4
    assert fixture.bundle.capacity == 32
    assert fixture.reused_slot == 1
    assert fixture.config.allow_mutation is True
    assert receipt["decision"] == C4_RESIDENT_TURNOVER_R7_NATIVE_DECISION_V1
    assert receipt["ticks"] == 5
    assert receipt["mutation"] is True
    assert receipt["execution_classification"] == (
        "host_launched_component_test_not_production_residency"
    )
    assert receipt["gate_claim"] is False
    assert all(len(source["sha256"]) == 64 for source in receipt["sources"].values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r7_native_five_tick_exact_parity_and_event_rollback(tmp_path: Path) -> None:
    build_dir = Path(os.environ.get("EIEL_C4_R7_BUILD_DIR", tmp_path / "build"))
    cpu_fixture = make_c4_resident_turnover_r7_native_fixture_v1(seed=53)
    cuda_fixture = _move_dataclass_tensors(cpu_fixture, torch.device("cuda"))
    assert isinstance(cuda_fixture, C4ResidentTurnoverR7NativeFixtureV1)
    staging = make_c4_resident_population_r5_staging_v1(
        batch=4, capacity=32, grid_size=8, streams=2, device="cuda"
    )

    for tick_index in range(5):
        execute_c4_resident_turnover_r7_oracle_tick_v1(
            cpu_fixture, tick_index=tick_index
        )
        execute_c4_resident_turnover_r7_cuda_tick_v1(
            cuda_fixture, staging, tick_index=tick_index,
            build_directory=build_dir,
        )
        torch.cuda.synchronize()
        assert staging.turnover.death_status.cpu().tolist() == [0, 0, 0, 0]
        assert staging.mutation.birth.status.cpu().tolist() == [0, 0, 0, 0]
        assert staging.mutation.birth.child_slots[:, 0].cpu().tolist() == [1] * 4
        assert staging.mutation.mutation_occurred[:, 0].cpu().tolist() == [True] * 4
        for index, (expected, actual) in enumerate(zip(
            _authoritative(cpu_fixture), _authoritative(cuda_fixture), strict=True
        )):
            torch.testing.assert_close(
                actual.cpu(), expected, equal_nan=True,
                msg=f"tick {tick_index} authoritative tensor {index}",
            )

    population = cuda_fixture.bundle.population
    events = cuda_fixture.bundle.events
    assert events is not None
    assert population.next_agent_id.cpu().tolist() == [10] * 4
    assert population.slot_tokens[:, 1].cpu().tolist() == [6] * 4
    assert events.active_count.cpu().tolist() == [15] * 4
    for world in range(4):
        assert events.typed_event_type[world, :15].cpu().tolist() == [4, 2, 1] * 5
        assert events.typed_sequence[world, :15].cpu().tolist() == list(range(15))

    failed = make_c4_resident_turnover_r7_native_fixture_v1(seed=59)
    failed_cuda = _move_dataclass_tensors(failed, torch.device("cuda"))
    failed_staging = make_c4_resident_population_r5_staging_v1(
        batch=4, capacity=32, grid_size=8, streams=2, device="cuda"
    )
    for tick_index in range(3):
        execute_c4_resident_turnover_r7_cuda_tick_v1(
            failed_cuda, failed_staging, tick_index=tick_index,
            build_directory=build_dir,
        )
    prepare_c4_resident_turnover_r7_tick_v1(failed_cuda.bundle)
    failed_events = failed_cuda.bundle.events
    assert failed_events is not None
    failed_events.active_count[2] = failed_events.capacity - 1
    failed_cuda.replay.event_sequence.copy_(failed_events.active_count)
    before = [tensor.clone() for tensor in _authoritative(failed_cuda)]
    execute_c4_resident_population_r5_cuda_v1(
        failed_cuda.bundle,
        failed_staging,
        copy.copy(failed_cuda.config),
        rng_context=failed_cuda.replay,
        build_directory=build_dir,
    )
    torch.cuda.synchronize()
    assert all(
        status != 0 for status in failed_staging.mutation.birth.status.cpu().tolist()
    )
    for expected, actual in zip(before, _authoritative(failed_cuda), strict=True):
        torch.testing.assert_close(actual, expected, equal_nan=True)
