from __future__ import annotations

from pathlib import Path

import pytest
import torch

from perf.cuda_architecture_b_child_policy import (
    CHILD_POLICY_CAPABILITIES,
    execute_fixed_mlp_child_policy_cuda,
    load_cuda_child_policy_extension,
    prepare_fixed_mlp_child_policy_tape,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture(scope="module")
def cuda_build_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("archb-child-policy-build")


@pytest.fixture(scope="module")
def cuda_module(cuda_build_directory: Path):
    return load_cuda_child_policy_extension(build_directory=cuda_build_directory)


def _identity_state() -> dict[str, torch.Tensor]:
    return {
        "slot_to_agent_id": torch.tensor(
            [-1, 10, 20, -1, 30, 31], dtype=torch.int64, device="cuda"
        ),
        "alive_mask": torch.tensor(
            [False, True, True, False, True, True], dtype=torch.bool, device="cuda"
        ),
        "parent_slots": torch.tensor([1, 2], dtype=torch.int64, device="cuda"),
        "child_slots": torch.tensor([4, 5], dtype=torch.int64, device="cuda"),
    }


def _policy_state(width: int = 17) -> dict[str, torch.Tensor]:
    rows = torch.arange(6 * width, dtype=torch.float32).reshape(6, width) / 100.0
    return {
        "mlp_pack": rows.cuda().contiguous(),
        "hmem": torch.full((6, 4), 9.0, dtype=torch.float32, device="cuda"),
        "policy_use_gru": torch.tensor(
            [False, False, False, False, True, True],
            dtype=torch.bool,
            device="cuda",
        ),
    }


def _torch_state(seed: int = 303) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator.get_state()


@pytest.mark.parametrize("mode", ["dense", "sparse"])
def test_child_policy_row_and_torch_state_match_cpu_draw_order_exactly(
    cuda_module,
    cuda_build_directory: Path,
    mode: str,
) -> None:
    del cuda_module
    identity = _identity_state()
    policy = _policy_state()
    width = policy["mlp_pack"].size(1)
    initial_mlp = policy["mlp_pack"].cpu().clone()
    initial_state = _torch_state()
    std = 0.075
    rate = 0.35

    expected_generator = torch.Generator(device="cpu")
    expected_generator.set_state(initial_state)
    expected_rows = initial_mlp.clone()
    expected_draws = 0
    expected_mutated = 0
    expected_events = 0
    expected_delta_total = 0.0
    expected_mutation_occurred: list[bool] = []
    expected_delta_by_birth: list[float] = []
    for parent_slot, child_slot in ((1, 4), (2, 5)):
        torch.rand((width,), generator=expected_generator, dtype=torch.float32)
        expected_draws += width
        if mode == "sparse":
            mask = torch.rand(
                (width,), generator=expected_generator, dtype=torch.float32
            ).lt(rate)
            expected_draws += width
        else:
            mask = torch.ones((width,), dtype=torch.bool)
        noise = torch.randn(
            (width,), generator=expected_generator, dtype=torch.float32
        ) * std
        expected_draws += width
        delta = torch.where(mask, noise, torch.zeros_like(noise))
        expected_rows[child_slot] = expected_rows[parent_slot] + delta
        count = int(mask.sum())
        expected_mutated += count
        expected_events += int(count > 0)
        row_delta = sum(float(value) for value in delta[mask])
        expected_mutation_occurred.append(count > 0)
        expected_delta_by_birth.append(row_delta)
        expected_delta_total += row_delta

    tape = prepare_fixed_mlp_child_policy_tape(
        initial_state,
        episode_capacity=2,
        row_width=width,
        device="cuda",
        policy_weight_mutation_enable=True,
        policy_weight_mutation_mode=mode,
        policy_weight_mutation_rate=rate,
        policy_weight_mutation_std=std,
        semantic_draws=11,
    )
    pointers = {
        name: tensor.data_ptr() for name, tensor in {**policy, **tape}.items()
    }
    result = execute_fixed_mlp_child_policy_cuda(
        **identity,
        **policy,
        **tape,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 0
    assert int(result["committed_count"].cpu()[0]) == 2
    assert int(result["semantic_draws_added"].cpu()[0]) == expected_draws
    assert int(result["mutation_events"].cpu()[0]) == expected_events
    assert int(result["mutated_elements"].cpu()[0]) == expected_mutated
    assert float(result["mutation_delta_total"].cpu()[0]) == expected_delta_total
    assert result["mutation_occurred_by_birth"].cpu().tolist() == expected_mutation_occurred
    torch.testing.assert_close(
        result["mutation_delta_by_birth"].cpu(),
        torch.tensor(expected_delta_by_birth, dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(policy["mlp_pack"].cpu(), expected_rows, rtol=0, atol=0)
    assert int(torch.count_nonzero(policy["hmem"][[4, 5]]).cpu()) == 0
    assert policy["policy_use_gru"][[4, 5]].cpu().tolist() == [False, False]
    assert int(tape["policy_birth_tape_cursor"].cpu()[0]) == 2
    assert int(tape["mutation_torch_cpu_semantic_draws"].cpu()[0]) == 11 + expected_draws
    torch.testing.assert_close(
        tape["rng_manager_mutation_torch_cpu_state"].cpu(),
        expected_generator.get_state(),
        rtol=0,
        atol=0,
    )
    assert pointers == {
        name: tensor.data_ptr() for name, tensor in {**policy, **tape}.items()
    }


def test_split_birth_consumption_matches_one_call(
    cuda_module,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    initial_state = _torch_state(101)

    def run(split: bool) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        identity = _identity_state()
        policy = _policy_state(width=9)
        tape = prepare_fixed_mlp_child_policy_tape(
            initial_state,
            episode_capacity=2,
            row_width=9,
            device="cuda",
            policy_weight_mutation_enable=True,
            policy_weight_mutation_mode="dense",
            policy_weight_mutation_std=0.05,
        )
        if split:
            for index in range(2):
                execute_fixed_mlp_child_policy_cuda(
                    slot_to_agent_id=identity["slot_to_agent_id"],
                    alive_mask=identity["alive_mask"],
                    parent_slots=identity["parent_slots"][index : index + 1],
                    child_slots=identity["child_slots"][index : index + 1],
                    **policy,
                    **tape,
                    build_directory=cuda_build_directory,
                )
        else:
            execute_fixed_mlp_child_policy_cuda(
                **identity,
                **policy,
                **tape,
                build_directory=cuda_build_directory,
            )
        torch.cuda.synchronize()
        return policy, tape

    full_policy, full_tape = run(False)
    split_policy, split_tape = run(True)
    for name in full_policy:
        torch.testing.assert_close(full_policy[name], split_policy[name], rtol=0, atol=0)
    for name in (
        "rng_manager_mutation_torch_cpu_state",
        "policy_birth_tape_cursor",
        "mutation_torch_cpu_semantic_draws",
    ):
        torch.testing.assert_close(full_tape[name], split_tape[name], rtol=0, atol=0)


def test_child_policy_replays_terminal_torch_state_without_post_state_tape(
    cuda_module,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    identity = _identity_state()
    policy = _policy_state(width=17)
    initial_state = _torch_state(404)
    initial_rows = policy["mlp_pack"].cpu().clone()
    tape = prepare_fixed_mlp_child_policy_tape(
        initial_state,
        episode_capacity=2,
        row_width=17,
        device="cuda",
        policy_weight_mutation_enable=False,
        compact_no_mutation_tape=True,
        include_post_state_tape=False,
        semantic_draws=31,
    )
    assert tuple(tape["policy_birth_post_rng_state_packed_tape"].shape) == (0, 0)

    result = execute_fixed_mlp_child_policy_cuda(
        **identity,
        **policy,
        **tape,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    expected_generator = torch.Generator(device="cpu")
    expected_generator.set_state(initial_state)
    torch.rand((17,), generator=expected_generator, dtype=torch.float32)
    torch.rand((17,), generator=expected_generator, dtype=torch.float32)
    expected_rows = initial_rows.clone()
    expected_rows[4] = expected_rows[1]
    expected_rows[5] = expected_rows[2]

    assert int(result["status"].cpu()[0]) == 0
    assert int(result["semantic_draws_added"].cpu()[0]) == 34
    torch.testing.assert_close(policy["mlp_pack"].cpu(), expected_rows, rtol=0, atol=0)
    torch.testing.assert_close(
        tape["rng_manager_mutation_torch_cpu_state"].cpu(),
        expected_generator.get_state(),
        rtol=0,
        atol=0,
    )
    assert int(tape["mutation_torch_cpu_semantic_draws"].cpu()[0]) == 65


def test_tape_exhaustion_is_atomic(
    cuda_module,
    cuda_build_directory: Path,
) -> None:
    del cuda_module
    identity = _identity_state()
    policy = _policy_state(width=7)
    tape = prepare_fixed_mlp_child_policy_tape(
        _torch_state(202),
        episode_capacity=1,
        row_width=7,
        device="cuda",
        policy_weight_mutation_enable=True,
        policy_weight_mutation_std=0.1,
    )
    before = {
        name: tensor.clone() for name, tensor in {**policy, **tape}.items()
    }

    result = execute_fixed_mlp_child_policy_cuda(
        **identity,
        **policy,
        **tape,
        build_directory=cuda_build_directory,
    )
    torch.cuda.synchronize()

    assert int(result["status"].cpu()[0]) == 2
    assert int(result["committed_count"].cpu()[0]) == 0
    for name, tensor in {**policy, **tape}.items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


def test_capability_receipt_is_bounded() -> None:
    for supported in (
        "fixed_mlp_parent_row_copy",
        "hmem_zero_initialization",
        "dense_policy_weight_mutation",
        "sparse_policy_weight_mutation",
        "exact_torch_cpu_mutation_state",
        "generation_boundary_policy_tape",
        "resident_variable_birth_consumption",
    ):
        assert CHILD_POLICY_CAPABILITIES[supported] is True
    for unsupported in (
        "generation_turnover_crossover",
        "global_constructor_rng_state",
        "child_drives_or_pcg64",
        "birth_events",
        "gru",
        "promotion_qualified",
    ):
        assert CHILD_POLICY_CAPABILITIES[unsupported] is False
