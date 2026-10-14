from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from perf.cuda_architecture_b_child_policy import unpack_crossover_mask_lsb0_v1
from perf.cuda_architecture_b_birth_transaction import (
    execute_fixed_mlp_turnover_refill_transaction_cuda,
    load_cuda_birth_transaction_extension,
)
from tests.perf.test_eiel_cuda_architecture_b_birth_transaction import (
    _assert_bit_exact,
    _make_bundle,
    _mutable_tensors,
)
from tests.perf.test_eiel_cuda_architecture_b_lifecycle import (
    _canon_genome_double,
    _canon_genome_float,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-turnover-transaction-build")


@pytest.fixture(scope="module")
def cuda_module(cuda_build_directory: Path) -> Any:
    return load_cuda_birth_transaction_extension(build_directory=cuda_build_directory)


def _make_turnover_bundle(*, policy_episode_capacity: int = 2) -> dict[str, Any]:
    bundle = _make_bundle(policy_episode_capacity=policy_episode_capacity)
    capacity = int(bundle["identity"]["slot_to_agent_id"].numel())
    bundle["parent_pool_slots"] = torch.full(
        (capacity,), -1, dtype=torch.int64, device="cuda"
    )
    bundle["parent_pool_slots"][:2] = torch.tensor(
        [3, 1], dtype=torch.int64, device="cuda"
    )
    bundle["parent_pool_count"] = torch.tensor(
        [2], dtype=torch.int64, device="cuda"
    )
    bundle["spawn_positions"] = torch.full(
        (capacity, 2), -1, dtype=torch.int32, device="cuda"
    )
    bundle["spawn_positions"][:2] = torch.tensor(
        [[2, 3], [4, 4]], dtype=torch.int32, device="cuda"
    )
    bundle["planned_count"] = torch.tensor([2], dtype=torch.int64, device="cuda")
    return bundle


def _execute_turnover_transaction(
    bundle: dict[str, Any],
    build_directory: Path,
) -> dict[str, torch.Tensor]:
    return execute_fixed_mlp_turnover_refill_transaction_cuda(
        identity_state=bundle["identity"],
        parent_pool_slots=bundle["parent_pool_slots"],
        parent_pool_count=bundle["parent_pool_count"],
        spawn_positions=bundle["spawn_positions"],
        planned_count=bundle["planned_count"],
        scalar_rng_state=bundle["scalar_rng"],
        genome_state=bundle["genome"],
        policy_mem_dim=bundle["policy_mem_dim"],
        policy_use_gru=bundle["policy_use_gru"],
        drive_state=bundle["drive"],
        policy_state=bundle["policy"],
        start_energy=60.0,
        start_health=1.0,
        generation=4,
        sensor_radius_max=3,
        max_energy=100.0,
        build_directory=build_directory,
    )


def test_turnover_transaction_commits_all_stages_atomically(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    bundle = _make_turnover_bundle()
    mutable = _mutable_tensors(bundle)
    pointers = {name: tensor.data_ptr() for name, tensor in mutable.items()}
    parent_energy_before = bundle["identity"]["energy"][[1, 3]].clone()
    parent_policy_energy_before = bundle["identity"]["policy_energy_observation"][
        [1, 3]
    ].clone()
    offspring_before = bundle["identity"]["offspring_count"][[1, 3]].clone()
    original_agent_ids = bundle["identity"]["slot_to_agent_id"].clone()
    policy_rows_before = bundle["policy"]["mlp_pack"].clone()
    result = _execute_turnover_transaction(bundle, cuda_build_directory)
    torch.cuda.synchronize()

    component_statuses = {
        name: result[name].cpu().tolist()
        for name in (
            "identity_status",
            "scalar_status",
            "drive_status",
            "policy_status",
        )
    }
    assert result["status"].cpu().tolist() == [0], component_statuses
    for component_status in (
        "identity_status",
        "scalar_status",
        "drive_status",
        "policy_status",
    ):
        assert result[component_status].cpu().tolist() == [0]
    assert result["committed_count"].cpu().tolist() == [2]
    parent_ids = result["parent_agent_ids"][:2].cpu().tolist()
    mate_ids = result["mate_agent_ids"][:2].cpu().tolist()
    assert all(parent_id in {3, 17} for parent_id in parent_ids)
    assert all(mate_id in {3, 17} for mate_id in mate_ids)
    assert all(parent != mate for parent, mate in zip(parent_ids, mate_ids, strict=True))
    assert result["child_agent_ids"][:2].cpu().tolist() == [41, 42]
    assert result["child_slots"][:2].cpu().tolist() == [6, 2]
    for name in (
        "parent_agent_ids",
        "mate_agent_ids",
        "child_agent_ids",
        "child_slots",
    ):
        assert bool(result[name][2:].eq(-1).all())
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in _mutable_tensors(bundle).items()
    }

    torch.testing.assert_close(
        bundle["identity"]["energy"][[1, 3]], parent_energy_before, rtol=0, atol=0
    )
    torch.testing.assert_close(
        bundle["identity"]["policy_energy_observation"][[1, 3]],
        parent_policy_energy_before,
        rtol=0,
        atol=0,
    )
    expected_offspring = offspring_before.clone()
    for reproductive_id in (*parent_ids, *mate_ids):
        reproductive_slot = int(
            (original_agent_ids == reproductive_id).nonzero()[0].item()
        )
        expected_offspring[0 if reproductive_slot == 1 else 1] += 1
    torch.testing.assert_close(
        bundle["identity"]["offspring_count"][[1, 3]],
        expected_offspring,
        rtol=0,
        atol=0,
    )

    child_slots = result["child_slots"][:2].cpu().tolist()
    assert bundle["identity"]["energy"][child_slots].cpu().tolist() == [60.0, 60.0]
    expected_policy_rows: list[torch.Tensor] = []
    for birth, (parent_id, mate_id) in enumerate(
        zip(parent_ids, mate_ids, strict=True)
    ):
        parent_slot = int((original_agent_ids == parent_id).nonzero()[0].item())
        mate_slot = int((original_agent_ids == mate_id).nonzero()[0].item())
        crossover_mask = unpack_crossover_mask_lsb0_v1(
            bundle["policy"]["policy_birth_crossover_mask_packed_tape"][birth],
            logical_width=int(policy_rows_before.shape[1]),
        )
        expected_policy_rows.append(
            torch.where(
                crossover_mask,
                policy_rows_before[parent_slot],
                policy_rows_before[mate_slot],
            )
            + bundle["policy"]["policy_birth_delta_tape"][birth]
        )
        expected_exploration = _canon_genome_float(
            0.5
            * (
                _canon_genome_double(
                    float(bundle["genome"]["genome_exploration_bias"][parent_slot])
                )
                + _canon_genome_double(
                    float(bundle["genome"]["genome_exploration_bias"][mate_slot])
                )
            )
        )
        assert float(
            bundle["genome"]["genome_exploration_bias"][child_slots[birth]]
        ) == expected_exploration
    for field in ("last_action", "last_last_action"):
        assert bundle["identity"][field][child_slots].cpu().tolist() == [-1, -1]
    for field in (
        "action_switches",
        "action_oscillations",
        "episode0_switches",
        "episode0_oscillations",
    ):
        assert bundle["identity"][field][child_slots].cpu().tolist() == [0, 0]

    expected_chemistry = torch.tensor(
        [0.64, 0.55, 0.0, 0.0, 0.50, 0.25],
        dtype=torch.float32,
        device="cuda",
    ).expand(2, -1)
    torch.testing.assert_close(
        bundle["drive"]["chemistry"][child_slots],
        expected_chemistry,
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(bundle["drive"]["drives"][child_slots]).all()
    assert result["drive_semantic_draws_added"].cpu().tolist()[0] > 0
    assert result["drive_raw64_draws_added"].cpu().tolist()[0] >= result[
        "drive_semantic_draws_added"
    ].cpu().tolist()[0]

    for child_slot, expected_row in zip(
        child_slots, expected_policy_rows, strict=True
    ):
        _assert_bit_exact(
            bundle["policy"]["mlp_pack"][child_slot],
            expected_row,
            f"policy row {child_slot}",
        )
    assert int(torch.count_nonzero(bundle["policy"]["hmem"][child_slots]).cpu()) == 0
    assert bundle["policy_use_gru"][child_slots].cpu().tolist() == [False, False]
    assert bundle["policy"]["policy_birth_tape_cursor"].cpu().tolist() == [2]
    assert result["policy_semantic_draws_added"].cpu().tolist() == [
        int(bundle["policy"]["policy_birth_semantic_draws_tape"].sum().cpu())
    ]


def test_turnover_policy_failure_rolls_back_every_staged_state(
    cuda_module: Any,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    bundle = _make_turnover_bundle(policy_episode_capacity=1)
    before = {
        name: tensor.clone() for name, tensor in _mutable_tensors(bundle).items()
    }
    pointers = {
        name: tensor.data_ptr() for name, tensor in _mutable_tensors(bundle).items()
    }

    result = _execute_turnover_transaction(bundle, cuda_build_directory)
    torch.cuda.synchronize()

    assert result["status"].cpu().tolist() == [4]
    assert result["identity_status"].cpu().tolist() == [0]
    assert result["scalar_status"].cpu().tolist() == [0]
    assert result["drive_status"].cpu().tolist() == [0]
    assert result["policy_status"].cpu().tolist() == [2]
    assert result["committed_count"].cpu().tolist() == [0]
    for counter in (
        "scalar_semantic_draws_added",
        "drive_semantic_draws_added",
        "drive_raw64_draws_added",
        "policy_semantic_draws_added",
        "policy_mutation_events",
        "policy_mutated_elements",
    ):
        assert result[counter].cpu().tolist() == [0]
    for name, tensor in _mutable_tensors(bundle).items():
        assert tensor.data_ptr() == pointers[name]
        _assert_bit_exact(tensor, before[name], name)
