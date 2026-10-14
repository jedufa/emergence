from __future__ import annotations

from pathlib import Path

import pytest
import torch

from perf.cuda_architecture_b_child_policy import (
    CHILD_POLICY_CAPABILITIES,
    execute_fixed_mlp_turnover_child_policy_cuda,
    load_cuda_child_policy_extension,
    packed_crossover_row_bytes_v1,
    prepare_fixed_mlp_turnover_child_policy_tape,
    unpack_crossover_mask_lsb0_v1,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-turnover-child-policy-build")


@pytest.fixture(scope="module")
def cuda_module(cuda_build_directory: Path):
    return load_cuda_child_policy_extension(build_directory=cuda_build_directory)


def _torch_state(seed: int = 808) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator.get_state()


def _resident_state(width: int) -> dict[str, torch.Tensor]:
    rows = torch.arange(6 * width, dtype=torch.float32).reshape(6, width) / 37.0
    return {
        "slot_to_agent_id": torch.tensor(
            [10, 20, 30, 40, 50, 60], dtype=torch.int64, device="cuda"
        ),
        "alive_mask": torch.ones(6, dtype=torch.bool, device="cuda"),
        "parent_slots": torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
        "mate_slots": torch.tensor([2, 3], dtype=torch.int64, device="cuda"),
        "child_slots": torch.tensor([4, 5], dtype=torch.int64, device="cuda"),
        "mlp_pack": rows.cuda().contiguous(),
        "hmem": torch.full((6, 5), 7.0, dtype=torch.float32, device="cuda"),
        "policy_use_gru": torch.tensor(
            [False, False, False, False, True, True],
            dtype=torch.bool,
            device="cuda",
        ),
    }


def test_turnover_policy_matches_exact_cpu_crossover_and_mutation_replay(
    cuda_module,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    width = 17
    std = 0.075
    semantic_start = 23
    initial_state = _torch_state()
    resident = _resident_state(width)
    initial_rows = resident["mlp_pack"].cpu().clone()

    generator = torch.Generator(device="cpu")
    generator.set_state(initial_state)
    expected_rows = initial_rows.clone()
    expected_crossover_masks: list[torch.Tensor] = []
    expected_delta_by_birth: list[float] = []
    expected_delta_total = 0.0
    for parent_slot, mate_slot, child_slot in ((0, 2, 4), (1, 3, 5)):
        crossover_mask = torch.rand(
            (width,), generator=generator, dtype=torch.float32
        ).lt(0.5)
        mutation_delta = torch.randn(
            (width,), generator=generator, dtype=torch.float32
        ) * std
        expected_rows[child_slot] = torch.where(
            crossover_mask,
            expected_rows[parent_slot],
            expected_rows[mate_slot],
        ) + mutation_delta
        row_delta = sum(float(value) for value in mutation_delta)
        expected_crossover_masks.append(crossover_mask)
        expected_delta_by_birth.append(row_delta)
        expected_delta_total += row_delta

    assert any(bool(mask.any()) and bool((~mask).any()) for mask in expected_crossover_masks)
    tape = prepare_fixed_mlp_turnover_child_policy_tape(
        initial_state,
        episode_capacity=2,
        row_width=width,
        device="cuda",
        policy_weight_mutation_enable=True,
        policy_weight_mutation_mode="dense",
        policy_weight_mutation_std=std,
        semantic_draws=semantic_start,
    )
    torch.testing.assert_close(
        unpack_crossover_mask_lsb0_v1(
            tape["policy_birth_crossover_mask_packed_tape"].cpu(),
            logical_width=width,
        ),
        torch.stack(expected_crossover_masks),
        rtol=0,
        atol=0,
    )
    assert tuple(tape["policy_birth_crossover_mask_packed_tape"].shape) == (
        2,
        packed_crossover_row_bytes_v1(width),
    )

    result = execute_fixed_mlp_turnover_child_policy_cuda(
        **resident,
        **tape,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    expected_draws = 2 * width * 2
    assert result["status"].cpu().tolist() == [0]
    assert result["committed_count"].cpu().tolist() == [2]
    assert result["semantic_draws_added"].cpu().tolist() == [expected_draws]
    assert result["mutation_events"].cpu().tolist() == [2]
    assert result["mutated_elements"].cpu().tolist() == [2 * width]
    assert float(result["mutation_delta_total"].cpu()[0]) == expected_delta_total
    assert result["mutation_occurred_by_birth"].cpu().tolist() == [True, True]
    torch.testing.assert_close(
        result["mutation_delta_by_birth"].cpu(),
        torch.tensor(expected_delta_by_birth, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(resident["mlp_pack"].cpu(), expected_rows, rtol=0, atol=0)
    assert int(torch.count_nonzero(resident["hmem"][[4, 5]]).cpu()) == 0
    assert resident["policy_use_gru"][[4, 5]].cpu().tolist() == [False, False]
    assert tape["policy_birth_tape_cursor"].cpu().tolist() == [2]
    assert tape["mutation_torch_cpu_semantic_draws"].cpu().tolist() == [
        semantic_start + expected_draws
    ]
    torch.testing.assert_close(
        tape["rng_manager_mutation_torch_cpu_state"].cpu(),
        generator.get_state(),
        rtol=0,
        atol=0,
    )
    assert CHILD_POLICY_CAPABILITIES["dedicated_generation_turnover_crossover"] is True
    assert CHILD_POLICY_CAPABILITIES["generation_turnover_crossover"] is False


def test_turnover_policy_replays_terminal_torch_state_without_post_state_tape(
    cuda_module,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    width = 17
    semantic_start = 23
    initial_state = _torch_state(818)
    resident = _resident_state(width)
    initial_rows = resident["mlp_pack"].cpu().clone()

    generator = torch.Generator(device="cpu")
    generator.set_state(initial_state)
    expected_rows = initial_rows.clone()
    expected_masks: list[torch.Tensor] = []
    for parent_slot, mate_slot, child_slot in ((0, 2, 4), (1, 3, 5)):
        crossover_mask = torch.rand(
            (width,), generator=generator, dtype=torch.float32
        ).lt(0.5)
        expected_masks.append(crossover_mask)
        expected_rows[child_slot] = torch.where(
            crossover_mask,
            expected_rows[parent_slot],
            expected_rows[mate_slot],
        )

    tape = prepare_fixed_mlp_turnover_child_policy_tape(
        initial_state,
        episode_capacity=2,
        row_width=width,
        device="cuda",
        policy_weight_mutation_enable=False,
        semantic_draws=semantic_start,
        compact_no_mutation_tape=True,
        include_post_state_tape=False,
    )
    assert tuple(tape["policy_birth_post_rng_state_packed_tape"].shape) == (2, 0)
    torch.testing.assert_close(
        unpack_crossover_mask_lsb0_v1(
            tape["policy_birth_crossover_mask_packed_tape"].cpu(),
            logical_width=width,
        ),
        torch.stack(expected_masks),
        rtol=0,
        atol=0,
    )

    result = execute_fixed_mlp_turnover_child_policy_cuda(
        **resident,
        **tape,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert result["status"].cpu().tolist() == [0]
    assert result["semantic_draws_added"].cpu().tolist() == [2 * width]
    assert result["mutation_events"].cpu().tolist() == [0]
    assert result["mutated_elements"].cpu().tolist() == [0]
    torch.testing.assert_close(resident["mlp_pack"].cpu(), expected_rows, rtol=0, atol=0)
    torch.testing.assert_close(
        tape["rng_manager_mutation_torch_cpu_state"].cpu(),
        generator.get_state(),
        rtol=0,
        atol=0,
    )
    assert tape["mutation_torch_cpu_semantic_draws"].cpu().tolist() == [
        semantic_start + 2 * width
    ]


@pytest.mark.parametrize(
    ("invalid_case", "expected_status"),
    [
        ("parent_row", 1),
        ("duplicate_ids", 1),
        ("gru_mate", 1),
        ("tape_bounds", 2),
    ],
)
def test_turnover_policy_rejects_invalid_sources_atomically(
    cuda_module,
    cuda_build_directory: Path,
    invalid_case: str,
    expected_status: int,
) -> None:
    del cuda_module
    width = 11
    resident = _resident_state(width)
    resident["parent_slots"] = resident["parent_slots"][:1]
    resident["mate_slots"] = resident["mate_slots"][:1]
    resident["child_slots"] = resident["child_slots"][:1]
    tape = prepare_fixed_mlp_turnover_child_policy_tape(
        _torch_state(909),
        episode_capacity=1,
        row_width=width,
        device="cuda",
        policy_weight_mutation_enable=True,
        policy_weight_mutation_mode="dense",
        policy_weight_mutation_std=0.05,
        semantic_draws=7,
    )
    if invalid_case == "parent_row":
        resident["parent_slots"] = torch.tensor([9], dtype=torch.int64, device="cuda")
    elif invalid_case == "duplicate_ids":
        resident["slot_to_agent_id"][2] = resident["slot_to_agent_id"][0]
    elif invalid_case == "gru_mate":
        resident["policy_use_gru"][2] = True
    elif invalid_case == "tape_bounds":
        tape["policy_birth_tape_cursor"].fill_(1)
    else:
        raise AssertionError(f"unhandled invalid case: {invalid_case}")
    before = {
        name: tensor.clone()
        for name, tensor in {
            "mlp_pack": resident["mlp_pack"],
            "hmem": resident["hmem"],
            "policy_use_gru": resident["policy_use_gru"],
            **tape,
        }.items()
    }

    result = execute_fixed_mlp_turnover_child_policy_cuda(
        **resident,
        **tape,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert result["status"].cpu().tolist() == [expected_status]
    assert result["committed_count"].cpu().tolist() == [0]
    for name, expected in before.items():
        actual = tape[name] if name in tape else resident[name]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
