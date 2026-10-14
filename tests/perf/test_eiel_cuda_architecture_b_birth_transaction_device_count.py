from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from perf.cuda_architecture_b_birth_transaction import (
    execute_fixed_mlp_birth_transaction_cuda,
)
from tests.perf.test_eiel_cuda_architecture_b_birth_transaction import (
    _assert_bit_exact,
    _clone_bundle,
    _make_bundle,
    _mutable_tensors,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-birth-transaction-device-count-build")


def _execute(
    bundle: dict[str, Any],
    planned_count: torch.Tensor,
    build_directory: Path,
) -> dict[str, torch.Tensor]:
    return execute_fixed_mlp_birth_transaction_cuda(
        identity_state=bundle["identity"],
        parent_slots=bundle["parent_slots"],
        spawn_positions=bundle["spawn_positions"],
        planned_count=planned_count,
        scalar_rng_state=bundle["scalar_rng"],
        genome_state=bundle["genome"],
        policy_mem_dim=bundle["policy_mem_dim"],
        policy_use_gru=bundle["policy_use_gru"],
        drive_state=bundle["drive"],
        policy_state=bundle["policy"],
        parent_energy_share=0.25,
        minimum_energy_buffer=2.0,
        start_health=1.5,
        cooldown_steps=7,
        generation=4,
        tick=12,
        sensor_radius_max=3,
        max_energy=100.0,
        build_directory=build_directory,
    )


def _all_input_tensors(
    bundle: dict[str, Any],
    planned_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        **_mutable_tensors(bundle),
        "parent_slots": bundle["parent_slots"],
        "spawn_positions": bundle["spawn_positions"],
        "planned_count": planned_count,
    }


def _assert_zero_transaction_counters(result: dict[str, torch.Tensor]) -> None:
    for name in (
        "committed_count",
        "scalar_semantic_draws_added",
        "population_birth_noise_rows_consumed",
        "population_exp_prior_noise_rows_consumed",
        "native_population_standard_normal_rows_generated",
        "drive_semantic_draws_added",
        "drive_raw64_draws_added",
        "policy_semantic_draws_added",
        "policy_mutation_events",
        "policy_mutated_elements",
        "policy_mutation_delta_total",
    ):
        assert float(result[name].cpu()[0]) == 0.0, name
    for name in (
        "policy_mutation_occurred_by_birth",
        "policy_mutation_delta_by_birth",
    ):
        assert int(torch.count_nonzero(result[name]).cpu()) == 0, name


def test_fixed_capacity_plan_ignores_invalid_inactive_tail(
    cuda_build_directory: Path,
) -> None:
    fixed = _make_bundle()
    compact = _clone_bundle(fixed)
    fixed["parent_slots"][1] = 999
    fixed["spawn_positions"][1] = torch.tensor(
        [-999, 999], dtype=torch.int32, device="cuda"
    )
    compact["parent_slots"] = compact["parent_slots"][:1].contiguous()
    compact["spawn_positions"] = compact["spawn_positions"][:1].contiguous()
    fixed_count = torch.tensor([1], dtype=torch.int64, device="cuda")
    compact_count = torch.tensor([1], dtype=torch.int64, device="cuda")
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in _all_input_tensors(fixed, fixed_count).items()
    }

    fixed_result = _execute(fixed, fixed_count, cuda_build_directory)
    compact_result = _execute(compact, compact_count, cuda_build_directory)
    torch.cuda.synchronize()

    assert int(fixed_result["status"].cpu()[0]) == 0
    assert int(fixed_result["committed_count"].cpu()[0]) == 1
    assert int(compact_result["status"].cpu()[0]) == 0
    assert pointers == {
        name: tensor.data_ptr()
        for name, tensor in _all_input_tensors(fixed, fixed_count).items()
    }
    for name, actual in _mutable_tensors(fixed).items():
        _assert_bit_exact(actual, _mutable_tensors(compact)[name], name)
    for name in (
        "committed_count",
        "alive_agent_ids_by_order",
        "alive_count",
        "scalar_semantic_draws_added",
        "population_birth_noise_rows_consumed",
        "population_exp_prior_noise_rows_consumed",
        "native_population_standard_normal_rows_generated",
        "drive_semantic_draws_added",
        "drive_raw64_draws_added",
        "policy_semantic_draws_added",
        "policy_mutation_events",
        "policy_mutated_elements",
        "policy_mutation_delta_total",
        "identity_status",
        "scalar_status",
        "drive_status",
        "policy_status",
        "status",
    ):
        _assert_bit_exact(fixed_result[name], compact_result[name], name)
    for name in ("parent_agent_ids", "child_agent_ids", "child_slots"):
        _assert_bit_exact(fixed_result[name][:1], compact_result[name], name)
        assert int(fixed_result[name][1].cpu()) == -1, name
    for name in (
        "policy_mutation_occurred_by_birth",
        "policy_mutation_delta_by_birth",
    ):
        _assert_bit_exact(fixed_result[name][:1], compact_result[name], name)
    assert bool(fixed_result["policy_mutation_occurred_by_birth"][1].cpu()) is False
    assert float(fixed_result["policy_mutation_delta_by_birth"][1].cpu()) == 0.0
    assert fixed["parent_slots"].cpu().tolist() == [1, 999]
    assert fixed["spawn_positions"].cpu().tolist() == [[1, 2], [-999, 999]]


def test_zero_planned_count_is_exact_noop_success(
    cuda_build_directory: Path,
) -> None:
    bundle = _make_bundle()
    bundle["parent_slots"].fill_(999)
    bundle["spawn_positions"].fill_(-999)
    planned_count = torch.tensor([0], dtype=torch.int64, device="cuda")
    tensors = _all_input_tensors(bundle, planned_count)
    before = {name: tensor.clone() for name, tensor in tensors.items()}
    pointers = {name: tensor.data_ptr() for name, tensor in tensors.items()}

    result = _execute(bundle, planned_count, cuda_build_directory)
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 0
    for name in ("identity_status", "scalar_status", "drive_status", "policy_status"):
        assert int(result[name].cpu()[0]) == 0, name
    _assert_zero_transaction_counters(result)
    for name, tensor in _all_input_tensors(bundle, planned_count).items():
        assert tensor.data_ptr() == pointers[name], name
        _assert_bit_exact(tensor, before[name], name)


@pytest.mark.parametrize("count", [-1, 3])
def test_out_of_range_planned_count_fails_atomically(
    count: int,
    cuda_build_directory: Path,
) -> None:
    bundle = _make_bundle()
    planned_count = torch.tensor([count], dtype=torch.int64, device="cuda")
    tensors = _all_input_tensors(bundle, planned_count)
    before = {name: tensor.clone() for name, tensor in tensors.items()}
    pointers = {name: tensor.data_ptr() for name, tensor in tensors.items()}

    result = _execute(bundle, planned_count, cuda_build_directory)
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 1
    assert int(result["identity_status"].cpu()[0]) != 0
    _assert_zero_transaction_counters(result)
    for name, tensor in _all_input_tensors(bundle, planned_count).items():
        assert tensor.data_ptr() == pointers[name], name
        _assert_bit_exact(tensor, before[name], name)
