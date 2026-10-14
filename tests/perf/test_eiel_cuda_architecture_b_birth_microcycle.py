from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b_birth_lifecycle_events import EVENT_TAPE_DTYPES
from perf.cuda_architecture_b_birth_microcycle import (
    BIRTH_MICROCYCLE_CAPABILITIES,
    execute_birth_microcycle_cuda,
)
from perf.cuda_architecture_b_pcg64 import pcg64_state_tensors
from tests.perf.test_eiel_cuda_architecture_b_birth_transaction import _make_bundle


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-birth-microcycle-build")


def _case(*, event_capacity: int = 32) -> dict[str, Any]:
    bundle = _make_bundle(policy_episode_capacity=2)
    planner = np.random.PCG64(202)
    planner_words, planner_buffer = pcg64_state_tensors(
        copy.deepcopy(planner.state),
        device="cuda",
    )
    runner = np.random.PCG64(303)
    runner_words, runner_buffer = pcg64_state_tensors(
        copy.deepcopy(runner.state),
        device="cuda",
    )
    return {
        **bundle,
        "planner_pcg64_state_words": planner_words,
        "planner_pcg64_buffer_words": planner_buffer,
        "policy_np_semantic_draws": torch.tensor(
            [41], dtype=torch.int64, device="cuda"
        ),
        "runner_pcg64_state_words": runner_words,
        "runner_pcg64_buffer_words": runner_buffer,
        "metrics_np_semantic_draws": torch.tensor(
            [17], dtype=torch.int64, device="cuda"
        ),
        "event_tape_columns": [
            torch.full((event_capacity,), -77, dtype=dtype, device="cuda")
            for dtype in EVENT_TAPE_DTYPES
        ],
        "event_tape_active_count": torch.zeros(
            (1,), dtype=torch.int64, device="cuda"
        ),
        "event_tape_status": torch.zeros((1,), dtype=torch.int32, device="cuda"),
    }


def _mutable(case: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for group in ("identity", "scalar_rng", "genome", "drive", "policy"):
        for name, tensor in case[group].items():
            tensors[f"{group}.{name}"] = tensor
    for name in (
        "policy_mem_dim",
        "policy_use_gru",
        "planner_pcg64_state_words",
        "planner_pcg64_buffer_words",
        "policy_np_semantic_draws",
        "runner_pcg64_state_words",
        "runner_pcg64_buffer_words",
        "metrics_np_semantic_draws",
        "event_tape_active_count",
        "event_tape_status",
    ):
        tensors[name] = case[name]
    for index, tensor in enumerate(case["event_tape_columns"]):
        tensors[f"event_tape_columns.{index}"] = tensor
    return tensors


def _execute(case: dict[str, Any], build_directory: Path) -> dict[str, torch.Tensor]:
    return execute_birth_microcycle_cuda(
        identity_state=case["identity"],
        scalar_rng_state=case["scalar_rng"],
        genome_state=case["genome"],
        policy_mem_dim=case["policy_mem_dim"],
        policy_use_gru=case["policy_use_gru"],
        drive_state=case["drive"],
        policy_state=case["policy"],
        planner_pcg64_state_words=case["planner_pcg64_state_words"],
        planner_pcg64_buffer_words=case["planner_pcg64_buffer_words"],
        policy_np_semantic_draws=case["policy_np_semantic_draws"],
        runner_pcg64_state_words=case["runner_pcg64_state_words"],
        runner_pcg64_buffer_words=case["runner_pcg64_buffer_words"],
        metrics_np_semantic_draws=case["metrics_np_semantic_draws"],
        event_tape_columns=case["event_tape_columns"],
        event_tape_active_count=case["event_tape_active_count"],
        event_tape_status=case["event_tape_status"],
        parent_energy_share=0.25,
        minimum_energy_buffer=2.0,
        start_health=1.5,
        cooldown_steps=7,
        generation=4,
        tick=12,
        sensor_radius_max=3,
        max_energy=100.0,
        max_births=2,
        external_gate_status=case.get("external_gate_status"),
        build_directory=build_directory,
    )


def test_birth_microcycle_commits_state_rng_and_decision_ordered_events(
    cuda_build_directory: Path,
) -> None:
    case = _case()
    pointers = {name: tensor.data_ptr() for name, tensor in _mutable(case).items()}

    result = _execute(case, cuda_build_directory)
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 0
    assert int(result["planner_status"].cpu()[0]) == 0
    assert int(result["transaction_status"].cpu()[0]) == 0
    assert int(result["event_tape_status"].cpu()[0]) == 0
    planned = int(result["planned_count"].cpu()[0])
    assert planned == int(result["committed_count"].cpu()[0]) == 2
    mutation_events = int(result["policy_mutation_events"].cpu()[0])
    event_count = int(result["event_tape_active_count"].cpu()[0])
    assert event_count == planned + mutation_events
    event_types = case["event_tape_columns"][0][:event_count].cpu().tolist()
    expected_types: list[int] = []
    for mutated in result["policy_mutation_occurred_by_birth"][:planned].cpu():
        if bool(mutated):
            expected_types.append(2)
        expected_types.append(1)
    assert event_types == expected_types
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in _mutable(case).items()
    }


def test_event_capacity_failure_rolls_back_entire_birth_microcycle(
    cuda_build_directory: Path,
) -> None:
    case = _case(event_capacity=1)
    before = {name: tensor.clone() for name, tensor in _mutable(case).items()}
    pointers = {name: tensor.data_ptr() for name, tensor in _mutable(case).items()}

    result = _execute(case, cuda_build_directory)
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 3
    assert int(result["event_tape_status"].cpu()[0]) != 0
    for name, tensor in _mutable(case).items():
        assert tensor.data_ptr() == pointers[name], name
        torch.testing.assert_close(
            tensor.cpu(),
            before[name].cpu(),
            rtol=0,
            atol=0,
            equal_nan=True,
        )


def test_external_generation_gate_rolls_back_entire_birth_microcycle(
    cuda_build_directory: Path,
) -> None:
    case = _case()
    case["external_gate_status"] = torch.tensor(
        [1], dtype=torch.int32, device="cuda"
    )
    before = {name: tensor.clone() for name, tensor in _mutable(case).items()}
    pointers = {name: tensor.data_ptr() for name, tensor in _mutable(case).items()}

    result = _execute(case, cuda_build_directory)
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 4
    assert int(result["committed_count"].cpu()[0]) > 0
    for name, tensor in _mutable(case).items():
        assert tensor.data_ptr() == pointers[name], name
        torch.testing.assert_close(
            tensor.cpu(),
            before[name].cpu(),
            rtol=0,
            atol=0,
            equal_nan=True,
        )


def test_birth_microcycle_capability_receipt_is_bounded() -> None:
    assert BIRTH_MICROCYCLE_CAPABILITIES["device_resident_plan_count"] is True
    assert BIRTH_MICROCYCLE_CAPABILITIES["per_tick_host_sync"] is False
    assert BIRTH_MICROCYCLE_CAPABILITIES["generation_turnover"] is False
    assert BIRTH_MICROCYCLE_CAPABILITIES["promotion_qualified"] is False
