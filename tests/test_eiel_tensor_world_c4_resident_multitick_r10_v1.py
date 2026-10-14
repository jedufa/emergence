"""CPU/static and deferred CUDA tests for the B48 multi-tick R10 driver."""

from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import os
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_resident_mixed_r5 import (
    capture_c4_resident_mixed_r5_authoritative_v1,
    compare_c4_resident_mixed_r5_authoritative_v1,
)
from engine.tensor_world.c4_resident_multitick_r10 import (
    C4_RESIDENT_MULTITICK_R10_BATCH_V1,
    C4_RESIDENT_MULTITICK_R10_CAPACITY_V1,
    C4_RESIDENT_MULTITICK_R10_DECISION_V1,
    C4_RESIDENT_MULTITICK_R10_TICKS_V1,
    C4ResidentMultiTickR10FixtureV1,
    c4_resident_multitick_r10_source_receipt_v1,
    make_c4_resident_multitick_r10_fixture_v1,
    run_c4_resident_multitick_r10_cuda_v1,
    run_c4_resident_multitick_r10_oracle_v1,
)
from engine.tensor_world.c4_resident_population_r9 import (
    make_c4_resident_population_r9_staging_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


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


def _tensor_leaves(value: object, *, path: str, output: dict[str, torch.Tensor]) -> None:
    if isinstance(value, torch.Tensor):
        output[path] = value
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _tensor_leaves(
                getattr(value, field.name), path=f"{path}.{field.name}", output=output
            )


def _assert_full_boundary_equal(expected, actual) -> None:
    expected_leaves: dict[str, torch.Tensor] = {}
    actual_leaves: dict[str, torch.Tensor] = {}
    _tensor_leaves(expected.bundle, path="bundle", output=expected_leaves)
    _tensor_leaves(expected.replay, path="replay", output=expected_leaves)
    _tensor_leaves(actual.bundle, path="bundle", output=actual_leaves)
    _tensor_leaves(actual.replay, path="replay", output=actual_leaves)
    assert expected_leaves.keys() == actual_leaves.keys()
    for path in expected_leaves:
        torch.testing.assert_close(
            actual_leaves[path].detach().cpu(),
            expected_leaves[path].detach().cpu(),
            atol=0.0,
            rtol=0.0,
            equal_nan=True,
            msg=path,
        )
    assert actual.bundle.base.base.core.tick == expected.bundle.base.base.core.tick


def test_c4_r10_receipt_admits_only_b48_c32_host_orchestration() -> None:
    fixture = make_c4_resident_multitick_r10_fixture_v1()
    receipt = c4_resident_multitick_r10_source_receipt_v1()

    assert fixture.bundle.batch == C4_RESIDENT_MULTITICK_R10_BATCH_V1 == 48
    assert fixture.bundle.capacity == C4_RESIDENT_MULTITICK_R10_CAPACITY_V1 == 32
    assert receipt["decision"] == C4_RESIDENT_MULTITICK_R10_DECISION_V1
    assert receipt["ticks"] == C4_RESIDENT_MULTITICK_R10_TICKS_V1 == 5
    assert receipt["execution_classification"] == (
        "host_python_tick_loop_with_per_tick_state_status_d2h_and_synchronization"
    )
    assert receipt["staging_reused"] is True
    assert receipt["rollback_boundary"] == (
        "all_bundle_and_replay_tensor_leaves_plus_scalar_tick"
    )
    assert receipt["rollback_scope"] == (
        "semantic_preflight_failure; runtime_fault_atomicity_not_claimed"
    )
    assert receipt["persistent_residency_claim"] is False
    assert receipt["cuda_graph_claim"] is False
    assert receipt["gate_claim"] is False
    assert all(len(row["sha256"]) == 64 for row in receipt["sources"].values())
    with pytest.raises(TensorWorldContractError, match="capacity=32"):
        make_c4_resident_multitick_r10_fixture_v1(capacity=31)


def test_c4_r10_cpu_five_ticks_seal_identity_events_generations_and_rng() -> None:
    fixture = make_c4_resident_multitick_r10_fixture_v1(seed=67)
    receipt = run_c4_resident_multitick_r10_oracle_v1(fixture)

    assert receipt.status == "MULTITICK_PASS"
    assert receipt.completed_ticks == receipt.requested_ticks == 5
    assert receipt.gate_claim is False
    for world in range(48):
        assert [row.dead_agent_ids[world] for row in receipt.ticks] == [2, 5, 6, 7, 8]
        assert [row.child_agent_ids[world] for row in receipt.ticks] == [5, 6, 7, 8, 9]
        assert [row.slot_tokens[world] for row in receipt.ticks] == [2, 3, 4, 5, 6]
        assert [row.next_agent_ids[world] for row in receipt.ticks] == [6, 7, 8, 9, 10]
        assert [row.event_active_counts[world] for row in receipt.ticks] == [3, 6, 9, 12, 15]
        assert [row.generation for row in receipt.ticks] == [11, 12, 13, 14, 15]
        assert [row.tick for row in receipt.ticks] == [100, 101, 102, 103, 104]
        assert all(row.preflight_status[world] == 0 for row in receipt.ticks)
        assert all(
            left < right
            for left, right in zip(
                [row.pcg_cursors[world] for row in receipt.ticks],
                [row.pcg_cursors[world] for row in receipt.ticks][1:],
            )
        )
        assert all(
            left < right
            for left, right in zip(
                [row.python_semantic_draws[world] for row in receipt.ticks],
                [row.python_semantic_draws[world] for row in receipt.ticks][1:],
            )
        )

    population = fixture.bundle.population
    events = fixture.bundle.events
    assert events is not None
    assert population.next_agent_id.tolist() == [10] * 48
    assert population.slot_tokens[:, 1].tolist() == [6] * 48
    assert events.active_count.tolist() == [15] * 48
    for world in range(48):
        assert events.typed_event_type[world, :15].tolist() == [4, 2, 1] * 5
        assert events.typed_generation[world, :15].tolist() == [
            0, 11, 11, 0, 12, 12, 0, 13, 13, 0, 14, 14, 0, 15, 15
        ]
        assert events.typed_tick[world, :15].tolist() == [
            100, 100, 100, 101, 101, 101, 102, 102, 102,
            103, 103, 103, 104, 104, 104,
        ]
        assert events.typed_sequence[world, :15].tolist() == list(range(15))


def test_c4_r10_cpu_late_failure_restores_full_batch_boundary() -> None:
    control = make_c4_resident_multitick_r10_fixture_v1(seed=71)
    failed = make_c4_resident_multitick_r10_fixture_v1(seed=71)
    assert run_c4_resident_multitick_r10_oracle_v1(control, ticks=3).status == (
        "MULTITICK_PASS"
    )
    receipt = run_c4_resident_multitick_r10_oracle_v1(
        failed, ticks=5, inject_failure_tick=3, inject_failure_world=47
    )

    assert receipt.status == "INJECTED_FAILURE_ROLLED_BACK"
    assert receipt.completed_ticks == 3
    assert receipt.failure_tick == 3
    assert receipt.rollback_applied is True
    comparison = compare_c4_resident_mixed_r5_authoritative_v1(
        capture_c4_resident_mixed_r5_authoritative_v1(control.bundle, control.replay),
        capture_c4_resident_mixed_r5_authoritative_v1(failed.bundle, failed.replay),
    )
    assert comparison["pass"] is True
    assert comparison["mismatches"] == []
    assert failed.bundle.base.base.core.tick == control.bundle.base.base.core.tick == 102


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_c4_r10_cuda_five_tick_exact_parity_and_late_rollback(tmp_path: Path) -> None:
    build_dir = Path(os.environ.get("EIEL_C4_R10_BUILD_DIR", tmp_path / "build"))
    cpu = make_c4_resident_multitick_r10_fixture_v1(seed=73)
    cuda = _move_dataclass_tensors(cpu, torch.device("cuda"))
    assert isinstance(cuda, C4ResidentMultiTickR10FixtureV1)
    staging = make_c4_resident_population_r9_staging_v1(
        batch=48, capacity=32, grid_size=8, streams=2, device="cuda"
    )

    cpu_receipt = run_c4_resident_multitick_r10_oracle_v1(cpu)
    cuda_receipt = run_c4_resident_multitick_r10_cuda_v1(
        cuda, staging, build_directory=build_dir
    )
    assert cuda_receipt == cpu_receipt
    _assert_full_boundary_equal(cpu, cuda)

    control = _move_dataclass_tensors(
        make_c4_resident_multitick_r10_fixture_v1(seed=79), torch.device("cuda")
    )
    failed = _move_dataclass_tensors(
        make_c4_resident_multitick_r10_fixture_v1(seed=79), torch.device("cuda")
    )
    assert isinstance(control, C4ResidentMultiTickR10FixtureV1)
    assert isinstance(failed, C4ResidentMultiTickR10FixtureV1)
    control_staging = make_c4_resident_population_r9_staging_v1(
        batch=48, capacity=32, grid_size=8, streams=2, device="cuda"
    )
    failed_staging = make_c4_resident_population_r9_staging_v1(
        batch=48, capacity=32, grid_size=8, streams=2, device="cuda"
    )
    assert run_c4_resident_multitick_r10_cuda_v1(
        control, control_staging, ticks=3, build_directory=build_dir
    ).status == "MULTITICK_PASS"
    failed_receipt = run_c4_resident_multitick_r10_cuda_v1(
        failed,
        failed_staging,
        ticks=5,
        inject_failure_tick=3,
        inject_failure_world=47,
        build_directory=build_dir,
    )
    assert failed_receipt.status == "INJECTED_FAILURE_ROLLED_BACK"
    assert failed_receipt.completed_ticks == 3
    assert failed_receipt.failure_tick == 3
    assert failed_receipt.rollback_applied is True
    assert failed_receipt.failure_status is not None
    assert all(value != 0 for value in failed_receipt.failure_status)
    _assert_full_boundary_equal(control, failed)
