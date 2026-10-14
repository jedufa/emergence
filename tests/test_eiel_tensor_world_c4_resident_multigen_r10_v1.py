"""CPU/static and deferred CUDA tests for the B48 multi-generation R10 harness."""

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
from engine.tensor_world.c4_resident_multigen_r10 import (
    C4_RESIDENT_MULTIGEN_R10_DECISION_V1,
    C4_RESIDENT_MULTIGEN_R10_SCHEMA_VERSION_V1,
    C4ResidentMultigenR10FixtureV1,
    c4_resident_multigen_r10_source_receipt_v1,
    execute_c4_resident_multigen_r10_cuda_tick_v1,
    execute_c4_resident_multigen_r10_oracle_tick_v1,
    make_c4_resident_multigen_r10_fixture_v1,
    prepare_c4_resident_multigen_r10_tick_v1,
)
from engine.tensor_world.c4_resident_mutation_r4 import (
    _replay_tensors,
    _scalar_rng_tensors,
)
from engine.tensor_world.c4_resident_population_r9 import (
    execute_c4_resident_population_r9_cuda_v1,
    make_c4_resident_population_r9_staging_v1,
)
from engine.tensor_world.c4_resident_turnover_r3 import _death_authoritative
from engine.tensor_world.c4_transaction import (
    C4PopulationTransactionError,
    execute_c4_population_transaction_oracle_v1,
)


ROOT = Path(__file__).resolve().parents[1]


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


def _authoritative(fixture: C4ResidentMultigenR10FixtureV1) -> list[torch.Tensor]:
    bundle = fixture.bundle
    return [
        *_primary_tensors(bundle),
        *_death_authoritative(bundle),
        *_inherit_tensors(bundle),
        *_zero_tensors(bundle),
        *_minus_one_tensors(bundle),
        *_half_tensors(bundle),
        *_one_tensors(bundle),
        *_scalar_rng_tensors(bundle),
        *_replay_tensors(fixture.replay),
    ]


def _assert_authoritative_equal(
    expected: C4ResidentMultigenR10FixtureV1,
    actual: C4ResidentMultigenR10FixtureV1,
    *,
    label: str,
) -> None:
    for index, (left, right) in enumerate(
        zip(_authoritative(expected), _authoritative(actual), strict=True)
    ):
        torch.testing.assert_close(
            right.cpu(),
            left.cpu(),
            equal_nan=True,
            msg=f"{label} authoritative tensor {index}",
        )


def test_c4_r10_receipt_freezes_host_loop_scope_and_r9_limitations() -> None:
    receipt = c4_resident_multigen_r10_source_receipt_v1()

    assert receipt["schema_version"] == C4_RESIDENT_MULTIGEN_R10_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENT_MULTIGEN_R10_DECISION_V1
    assert receipt["route"] == "R10_B48_C32_FIVE_TICK_FIVE_GENERATION_R9_HARNESS"
    assert receipt["batch"] == 48
    assert receipt["capacity"] == 32
    assert receipt["ticks"] == 5
    assert receipt["generations"] == [10, 11, 12, 13, 14]
    assert receipt["rollback_boundary"] == "one_R9_tick_batch_global_semantic_failure"
    assert receipt["runtime_fault_atomicity"] == (
        "inherits_R9_not_claimed_for_multi_kernel_launch_faults"
    )
    assert receipt["execution_classification"] == (
        "host_loop_harness_not_single_resident_multi_tick_kernel"
    )
    assert receipt["gate_claim"] is False
    for source in receipt["sources"].values():
        assert len(source["sha256"]) == 64
        assert (ROOT / source["path"]).is_file()


def test_c4_r10_cpu_five_generations_and_injected_failure_rollback() -> None:
    fixture = make_c4_resident_multigen_r10_fixture_v1(seed=83)
    dead_ids: list[int] = []
    child_ids: list[int] = []
    semantic_draws: list[int] = []

    for tick_index in range(fixture.ticks):
        dead_ids.append(
            int(
                fixture.bundle.base.base.core.agents.agent_id[
                    0, fixture.reused_slot
                ]
            )
        )
        execute_c4_resident_multigen_r10_oracle_tick_v1(
            fixture, tick_index=tick_index
        )
        child_ids.append(
            int(
                fixture.bundle.base.base.core.agents.agent_id[
                    0, fixture.reused_slot
                ]
            )
        )
        semantic_draws.append(
            int(fixture.replay.semantic_draw_counters[0].sum())
        )

    population = fixture.bundle.population
    events = fixture.bundle.events
    assert events is not None
    assert dead_ids == [2, 5, 6, 7, 8]
    assert child_ids == [5, 6, 7, 8, 9]
    assert population.next_agent_id.tolist() == [10] * 48
    assert population.slot_tokens[:, 1].tolist() == [6] * 48
    assert semantic_draws == sorted(semantic_draws)
    assert len(set(semantic_draws)) == 5
    assert events.active_count.tolist() == [15] * 48
    assert events.typed_event_type[0, :15].tolist() == [4, 2, 1] * 5
    assert events.typed_generation[0, :15].tolist() == [
        0, 10, 10, 0, 11, 11, 0, 12, 12, 0, 13, 13, 0, 14, 14
    ]
    assert events.typed_tick[0, :15].tolist() == [
        1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5
    ]
    assert events.typed_sequence[0, :15].tolist() == list(range(15))

    failed_config = prepare_c4_resident_multigen_r10_tick_v1(
        fixture, tick_index=4
    )
    events.active_count[17] = events.capacity - 1
    fixture.replay.event_sequence.copy_(events.active_count)
    before = [tensor.clone() for tensor in _authoritative(fixture)]
    with pytest.raises(C4PopulationTransactionError) as raised:
        execute_c4_population_transaction_oracle_v1(
            fixture.bundle,
            mate_config=failed_config,
            rng_context=fixture.replay,
        )
    assert raised.value.receipt.rollback_applied is True
    for prior, after in zip(before, _authoritative(fixture), strict=True):
        torch.testing.assert_close(after, prior, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r10_cuda_five_tick_exact_parity_and_event_rollback(tmp_path: Path) -> None:
    build_dir = Path(os.environ.get("EIEL_C4_R10_BUILD_DIR", tmp_path / "build"))
    expected = make_c4_resident_multigen_r10_fixture_v1(seed=89)
    actual = _move_dataclass_tensors(expected, torch.device("cuda"))
    assert isinstance(actual, C4ResidentMultigenR10FixtureV1)
    staging = make_c4_resident_population_r9_staging_v1(
        batch=48, capacity=32, grid_size=8, streams=2, device="cuda"
    )

    for tick_index in range(expected.ticks):
        execute_c4_resident_multigen_r10_oracle_tick_v1(
            expected, tick_index=tick_index
        )
        execute_c4_resident_multigen_r10_cuda_tick_v1(
            actual,
            staging,
            tick_index=tick_index,
            build_directory=build_dir,
        )
        torch.cuda.synchronize()
        assert staging.preflight_status.cpu().tolist() == [0] * 48
        assert staging.turnover.death_count.cpu().tolist() == [1] * 48
        assert staging.mutation.birth.planned_count.cpu().tolist() == [1] * 48
        _assert_authoritative_equal(
            expected, actual, label=f"R10 tick {tick_index}"
        )

    expected_config = prepare_c4_resident_multigen_r10_tick_v1(
        expected, tick_index=4
    )
    actual_config = prepare_c4_resident_multigen_r10_tick_v1(actual, tick_index=4)
    expected.bundle.events.active_count[17] = expected.bundle.events.capacity - 1
    actual.bundle.events.active_count[17] = actual.bundle.events.capacity - 1
    expected.replay.event_sequence.copy_(expected.bundle.events.active_count)
    actual.replay.event_sequence.copy_(actual.bundle.events.active_count)
    cpu_before = [tensor.clone() for tensor in _authoritative(expected)]
    cuda_before = [tensor.clone() for tensor in _authoritative(actual)]

    with pytest.raises(C4PopulationTransactionError):
        execute_c4_population_transaction_oracle_v1(
            expected.bundle,
            mate_config=expected_config,
            rng_context=expected.replay,
        )
    execute_c4_resident_population_r9_cuda_v1(
        actual.bundle,
        staging,
        actual_config,
        rng_context=actual.replay,
        build_directory=build_dir,
    )
    torch.cuda.synchronize()
    statuses = (
        staging.preflight_status.cpu().tolist()
        + staging.turnover.death_status.cpu().tolist()
        + staging.mutation.birth.status.cpu().tolist()
    )
    assert any(status != 0 for status in statuses)
    for before, after in zip(cpu_before, _authoritative(expected), strict=True):
        torch.testing.assert_close(after, before, equal_nan=True)
    for before, after in zip(cuda_before, _authoritative(actual), strict=True):
        torch.testing.assert_close(after, before, equal_nan=True)
    _assert_authoritative_equal(expected, actual, label="R10 rollback")
