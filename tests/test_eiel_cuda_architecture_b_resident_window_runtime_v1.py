from __future__ import annotations

import json
import random
from collections.abc import Mapping

import pytest
import torch

from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_DTYPES,
    EVENT_COLUMN_NAMES,
)
from engine.perf.cuda_architecture_b_child_policy import (
    packed_crossover_row_bytes_v1,
    prepare_fixed_mlp_turnover_child_policy_tape,
)
from engine.perf.cuda_architecture_b_lifecycle import (
    pack_cpython_random_state_cuda,
)
from engine.perf.cuda_architecture_b_resident_window_runtime_v1 import (
    ResidentWindowContractError,
    drain_event_window,
    materialize_event_history,
    reconstruct_cpython_random_states,
    refill_cpython_gaussian_window_,
    refill_fixed_mlp_policy_window_,
    validate_resident_window_plan,
)
from engine.perf.cuda_architecture_b_torch_rng_state_pack_v1 import (
    TORCH_CPU_RNG_STATE_ENCODING_V1,
    TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
    unpack_torch_cpu_rng_state_v1,
)


def _assert_json_safe(receipt: Mapping[str, object]) -> None:
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt


def _data_ptrs(tensors: Mapping[str, torch.Tensor]) -> dict[str, int]:
    return {name: tensor.data_ptr() for name, tensor in tensors.items()}


def _stack_cpython_states(
    states: list[tuple[object, ...]], *, capacity: int
) -> dict[str, torch.Tensor]:
    packed = [
        pack_cpython_random_state_cuda(
            state, device="cpu", gauss_tape_raw_capacity=capacity
        )
        for state in states
    ]
    scalar_fields = {
        "python_random_index",
        "python_random_has_gauss",
        "python_random_gauss_next",
        "python_random_raw_draws",
        "mutation_py_semantic_draws",
    }
    return {
        name: (
            torch.cat([cell[name] for cell in packed])
            if name in scalar_fields
            else torch.stack([cell[name] for cell in packed])
        )
        for name in packed[0]
    }


def _replace_cpython_authority(
    resident: dict[str, torch.Tensor],
    states: list[tuple[object, ...]],
) -> None:
    packed = [pack_cpython_random_state_cuda(state, device="cpu") for state in states]
    for name in (
        "python_random_words",
        "python_random_index",
        "python_random_has_gauss",
        "python_random_gauss_next",
    ):
        rows = [cell[name] for cell in packed]
        resident[name].copy_(
            torch.stack(rows)
            if name == "python_random_words"
            else torch.cat(rows)
        )


def _expected_gaussian_tapes(
    states: tuple[tuple[int, tuple[int, ...], float | None], ...],
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = [
        pack_cpython_random_state_cuda(
            state, device="cpu", gauss_tape_raw_capacity=capacity
        )
        for state in states
    ]
    return (
        torch.stack(
            [cell["python_random_gauss_primary_tape"] for cell in prepared]
        ),
        torch.stack(
            [cell["python_random_gauss_secondary_tape"] for cell in prepared]
        ),
    )


def _policy_state(
    *, cells: int = 2, episodes: int = 4, width: int = 3
) -> dict[str, torch.Tensor]:
    sources = []
    prepared = []
    for seed in range(cells):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(700 + seed)
        source = generator.get_state().clone()
        sources.append(source)
        prepared.append(
            prepare_fixed_mlp_turnover_child_policy_tape(
                source,
                episode_capacity=episodes,
                row_width=width,
                device="cpu",
                policy_weight_mutation_enable=True,
                policy_weight_mutation_mode="sparse",
                policy_weight_mutation_rate=0.5,
                policy_weight_mutation_std=0.125,
                semantic_draws=0,
            )
        )
    names = (
        "policy_birth_delta_tape",
        "policy_birth_mutation_mask_tape",
        "policy_birth_crossover_mask_packed_tape",
        "policy_birth_post_rng_state_packed_tape",
        "policy_birth_semantic_draws_tape",
    )
    return {
        "rng_manager_mutation_torch_cpu_state": torch.stack(sources),
        **{
            name: torch.stack([cell[name] for cell in prepared])
            for name in names
        },
        "policy_birth_tape_cursor": torch.tensor([3, 5], dtype=torch.int64)[
            :cells
        ],
        "mutation_torch_cpu_semantic_draws": torch.tensor(
            [101, 211], dtype=torch.int64
        )[:cells],
    }


def _expected_policy_tapes(
    policy: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    rows = []
    for cell in range(int(policy["policy_birth_delta_tape"].shape[0])):
        rows.append(
            prepare_fixed_mlp_turnover_child_policy_tape(
                policy["rng_manager_mutation_torch_cpu_state"][cell]
                .clone()
                .contiguous(),
                episode_capacity=int(policy["policy_birth_delta_tape"].shape[1]),
                row_width=int(policy["policy_birth_delta_tape"].shape[2]),
                device="cpu",
                policy_weight_mutation_enable=True,
                policy_weight_mutation_mode="sparse",
                policy_weight_mutation_rate=0.5,
                policy_weight_mutation_std=0.125,
                semantic_draws=int(
                    policy["mutation_torch_cpu_semantic_draws"][cell]
                ),
            )
        )
    return {
        name: torch.stack([row[name] for row in rows])
        for name in (
            "policy_birth_delta_tape",
            "policy_birth_mutation_mask_tape",
            "policy_birth_crossover_mask_packed_tape",
            "policy_birth_post_rng_state_packed_tape",
            "policy_birth_semantic_draws_tape",
        )
    }


def _event_bank(cells: int = 2, capacity: int = 5) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros((cells, capacity), dtype=dtype)
        for name, dtype in EVENT_COLUMN_DTYPES.items()
    }


def _fill_event_rows(
    bank: Mapping[str, torch.Tensor],
    *,
    cell: int,
    base: int,
    count: int,
    value_offset: int,
) -> None:
    for row in range(count):
        absolute = base + row
        for field_index, name in enumerate(EVENT_COLUMN_NAMES):
            tensor = bank[name]
            if name == "typed_lifecycle_event_sequence":
                value: int | float = absolute
            elif tensor.dtype == torch.bool:
                value = (absolute + field_index) % 2 == 0
            elif tensor.dtype.is_floating_point:
                value = float(value_offset + absolute) + field_index / 100.0
            elif tensor.dtype == torch.int8:
                value = (absolute + field_index) % 2
            else:
                value = value_offset + absolute + field_index
            tensor[cell, row] = value


def test_validate_resident_window_plan_is_exact_and_json_safe() -> None:
    receipt = validate_resident_window_plan(
        128,
        32,
        (32, 32, 32, 32),
        batch_cells=48,
        ticks_per_generation=150,
    )
    assert receipt["logical_ticks_per_cell"] == 19_200
    assert receipt["window_tick_capacity_per_cell"] == 4_800
    assert receipt["logical_cell_generations"] == 6_144
    assert receipt["chunk_intervals"][-1] == {
        "chunk_index": 3,
        "generation_start": 96,
        "generation_end": 128,
        "tick_start": 14_400,
        "tick_end": 19_200,
    }
    _assert_json_safe(receipt)


@pytest.mark.parametrize(
    ("logical", "window", "chunks", "cells", "ticks"),
    [
        (0, 2, (2,), 1, 80),
        (4, True, (2, 2), 1, 80),
        (4, 2, (), 1, 80),
        (4, 2, (3, 1), 1, 80),
        (4, 2, (2, True), 1, 80),
        (4, 2, (2, 1), 1, 80),
        (4, 2, (2, 2), 0, 80),
        (4, 2, (2, 2), 1, 80.0),
    ],
)
def test_validate_resident_window_plan_fails_closed(
    logical: object,
    window: object,
    chunks: object,
    cells: object,
    ticks: object,
) -> None:
    with pytest.raises(ResidentWindowContractError):
        validate_resident_window_plan(  # type: ignore[arg-type]
            logical,
            window,
            chunks,
            batch_cells=cells,
            ticks_per_generation=ticks,
        )


def test_cpython_gaussian_window_two_refills_preserve_absolute_authority() -> None:
    rngs = [random.Random(101), random.Random(202)]
    rngs[0].gauss(0.0, 1.0)
    states = [rng.getstate() for rng in rngs]
    resident = _stack_cpython_states(states, capacity=6)
    resident["python_random_raw_draws"].copy_(
        torch.tensor([7, 11], dtype=torch.int64)
    )
    resident["mutation_py_semantic_draws"].copy_(
        torch.tensor([31, 47], dtype=torch.int64)
    )
    base = torch.tensor([2, 4], dtype=torch.int64)
    mutable = {
        "primary": resident["python_random_gauss_primary_tape"],
        "secondary": resident["python_random_gauss_secondary_tape"],
        "base": base,
    }
    pointers = _data_ptrs(mutable)
    reconstructed = reconstruct_cpython_random_states(resident)
    expected_primary, expected_secondary = _expected_gaussian_tapes(
        reconstructed, 6
    )

    first = refill_cpython_gaussian_window_(
        resident, python_random_window_base=base
    )

    assert torch.equal(
        resident["python_random_gauss_primary_tape"], expected_primary
    )
    assert torch.equal(
        resident["python_random_gauss_secondary_tape"], expected_secondary
    )
    assert resident["python_random_raw_draws"].tolist() == [7, 11]
    assert resident["mutation_py_semantic_draws"].tolist() == [31, 47]
    assert base.tolist() == [7, 11]
    assert _data_ptrs(mutable) == pointers
    assert first["pointer_identity"]["all_exact"] is True
    _assert_json_safe(first)

    for index, rng in enumerate(rngs):
        rng.setstate(reconstructed[index])
        rng.gauss(0.0, 1.0)
        rng.getrandbits(32)
    next_states = [rng.getstate() for rng in rngs]
    _replace_cpython_authority(resident, next_states)
    resident["python_random_raw_draws"].copy_(
        torch.tensor([13, 19], dtype=torch.int64)
    )
    resident["mutation_py_semantic_draws"].copy_(
        torch.tensor([55, 71], dtype=torch.int64)
    )
    next_reconstructed = reconstruct_cpython_random_states(resident)
    next_primary, next_secondary = _expected_gaussian_tapes(
        next_reconstructed, 6
    )

    second = refill_cpython_gaussian_window_(
        resident, python_random_window_base=base
    )

    assert torch.equal(
        resident["python_random_gauss_primary_tape"], next_primary
    )
    assert torch.equal(
        resident["python_random_gauss_secondary_tape"], next_secondary
    )
    assert resident["python_random_raw_draws"].tolist() == [13, 19]
    assert resident["mutation_py_semantic_draws"].tolist() == [55, 71]
    assert base.tolist() == [13, 19]
    assert second["window_base_before"] == [7, 11]
    assert second["window_base_after"] == [13, 19]
    assert _data_ptrs(mutable) == pointers
    _assert_json_safe(second)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_index",
        "bad_words_dtype",
        "bad_tape_shape",
        "bad_counter_dtype",
        "base_ahead",
    ],
)
def test_cpython_gaussian_window_fails_closed(mutation: str) -> None:
    resident = _stack_cpython_states([random.Random(4).getstate()], capacity=3)
    resident["python_random_raw_draws"].fill_(2)
    base = torch.zeros((1,), dtype=torch.int64)
    if mutation == "missing_index":
        del resident["python_random_index"]
    elif mutation == "bad_words_dtype":
        resident["python_random_words"] = resident["python_random_words"].to(
            torch.int64
        )
    elif mutation == "bad_tape_shape":
        resident["python_random_gauss_secondary_tape"] = torch.zeros(
            (1, 4), dtype=torch.float64
        )
    elif mutation == "bad_counter_dtype":
        resident["python_random_raw_draws"] = torch.tensor([2], dtype=torch.int32)
    else:
        base.fill_(3)
    primary_before = resident["python_random_gauss_primary_tape"].clone()
    secondary_before = resident["python_random_gauss_secondary_tape"].clone()
    base_before = base.clone()
    with pytest.raises(ResidentWindowContractError):
        refill_cpython_gaussian_window_(
            resident, python_random_window_base=base
        )
    assert torch.equal(resident["python_random_gauss_primary_tape"], primary_before)
    assert torch.equal(
        resident["python_random_gauss_secondary_tape"], secondary_before
    )
    assert torch.equal(base, base_before)


def test_fixed_mlp_policy_window_two_refills_preserve_absolute_authority() -> None:
    policy = _policy_state()
    base = torch.tensor([1, 2], dtype=torch.int64)
    mutable = {
        name: policy[name]
        for name in (
            "policy_birth_delta_tape",
            "policy_birth_mutation_mask_tape",
            "policy_birth_crossover_mask_packed_tape",
            "policy_birth_post_rng_state_packed_tape",
            "policy_birth_semantic_draws_tape",
        )
    }
    mutable["base"] = base
    pointers = _data_ptrs(mutable)
    source_before = policy["rng_manager_mutation_torch_cpu_state"].clone()
    expected = _expected_policy_tapes(policy)

    first = refill_fixed_mlp_policy_window_(
        policy,
        policy_birth_window_base=base,
        policy_row_width=3,
        policy_weight_mutation_enable=True,
        policy_weight_mutation_mode="sparse",
        policy_weight_mutation_rate=0.5,
        policy_weight_mutation_std=0.125,
    )

    for name, value in expected.items():
        assert torch.equal(policy[name], value), name
    assert torch.equal(
        policy["rng_manager_mutation_torch_cpu_state"], source_before
    )
    assert policy["policy_birth_tape_cursor"].tolist() == [3, 5]
    assert policy["mutation_torch_cpu_semantic_draws"].tolist() == [101, 211]
    assert base.tolist() == [3, 5]
    assert _data_ptrs(mutable) == pointers
    assert first["absolute_cursor_preserved"] is True
    assert first["torch_rng_post_state_encoding"] == TORCH_CPU_RNG_STATE_ENCODING_V1
    assert first["torch_rng_post_state_packed_row_bytes"] == (
        TORCH_CPU_RNG_STATE_PACKED_BYTES_V1
    )
    assert first["torch_rng_post_state_dense_allocation_materialized"] is False
    assert first["torch_rng_post_state_abi"]["torch_version"] == "2.10.0+cu128"
    assert first["torch_rng_post_state_abi"]["byteorder"] == "little"
    assert first["torch_rng_post_state_abi"]["portable_checkpoint_format"] is False
    _assert_json_safe(first)

    consumed = 2
    next_sources = unpack_torch_cpu_rng_state_v1(
        policy["policy_birth_post_rng_state_packed_tape"][:, consumed - 1]
    )
    added = policy["policy_birth_semantic_draws_tape"][:, :consumed].sum(dim=1)
    policy["rng_manager_mutation_torch_cpu_state"].copy_(next_sources)
    policy["policy_birth_tape_cursor"].add_(consumed)
    policy["mutation_torch_cpu_semantic_draws"].add_(added)
    expected_next = _expected_policy_tapes(policy)

    second = refill_fixed_mlp_policy_window_(
        policy,
        policy_birth_window_base=base,
        policy_row_width=3,
        policy_weight_mutation_enable=True,
        policy_weight_mutation_mode="sparse",
        policy_weight_mutation_rate=0.5,
        policy_weight_mutation_std=0.125,
    )

    for name, value in expected_next.items():
        assert torch.equal(policy[name], value), name
    assert policy["policy_birth_tape_cursor"].tolist() == [5, 7]
    assert base.tolist() == [5, 7]
    assert second["window_base_before"] == [3, 5]
    assert second["window_base_after"] == [5, 7]
    assert second["semantic_draws_before"] == second["semantic_draws_after"]
    assert _data_ptrs(mutable) == pointers
    _assert_json_safe(second)


def test_fixed_mlp_policy_window_accepts_compact_disabled_mutation_tape() -> None:
    cells = 2
    episodes = 4
    row_width = 3
    sources = []
    prepared = []
    for seed in range(cells):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1700 + seed)
        source = generator.get_state().clone()
        sources.append(source)
        prepared.append(
            prepare_fixed_mlp_turnover_child_policy_tape(
                source,
                episode_capacity=episodes,
                row_width=row_width,
                device="cpu",
                policy_weight_mutation_enable=False,
                policy_weight_mutation_std=0.0,
                compact_no_mutation_tape=True,
                include_post_state_tape=False,
            )
        )
    policy = {
        "rng_manager_mutation_torch_cpu_state": torch.stack(sources),
        **{
            name: torch.stack([cell[name] for cell in prepared])
            for name in (
                "policy_birth_delta_tape",
                "policy_birth_mutation_mask_tape",
                "policy_birth_crossover_mask_packed_tape",
                "policy_birth_post_rng_state_packed_tape",
                "policy_birth_semantic_draws_tape",
            )
        },
        "policy_birth_tape_cursor": torch.tensor([3, 5], dtype=torch.int64),
        "mutation_torch_cpu_semantic_draws": torch.tensor(
            [101, 211], dtype=torch.int64
        ),
    }
    base = torch.tensor([1, 2], dtype=torch.int64)
    pointers = _data_ptrs(
        {
            "delta": policy["policy_birth_delta_tape"],
            "mask": policy["policy_birth_mutation_mask_tape"],
            "crossover": policy["policy_birth_crossover_mask_packed_tape"],
            "post": policy["policy_birth_post_rng_state_packed_tape"],
            "base": base,
        }
    )

    receipt = refill_fixed_mlp_policy_window_(
        policy,
        policy_birth_window_base=base,
        policy_row_width=row_width,
        policy_weight_mutation_enable=False,
        policy_weight_mutation_std=0.0,
    )

    assert tuple(policy["policy_birth_delta_tape"].shape) == (cells, episodes, 1)
    assert tuple(policy["policy_birth_mutation_mask_tape"].shape) == (
        cells,
        episodes,
        1,
    )
    assert tuple(policy["policy_birth_crossover_mask_packed_tape"].shape) == (
        cells,
        episodes,
        packed_crossover_row_bytes_v1(row_width),
    )
    assert tuple(policy["policy_birth_post_rng_state_packed_tape"].shape) == (
        cells,
        episodes,
        0,
    )
    assert torch.equal(policy["policy_birth_delta_tape"], torch.zeros_like(policy["policy_birth_delta_tape"]))
    assert torch.equal(policy["policy_birth_mutation_mask_tape"], torch.zeros_like(policy["policy_birth_mutation_mask_tape"]))
    assert base.tolist() == [3, 5]
    assert _data_ptrs(
        {
            "delta": policy["policy_birth_delta_tape"],
            "mask": policy["policy_birth_mutation_mask_tape"],
            "crossover": policy["policy_birth_crossover_mask_packed_tape"],
            "post": policy["policy_birth_post_rng_state_packed_tape"],
            "base": base,
        }
    ) == pointers
    assert receipt["policy_row_width"] == row_width
    assert receipt["absolute_cursor_preserved"] is True
    assert receipt["torch_rng_post_state_resident_rows_materialized"] is False
    assert (
        receipt["torch_rng_post_state_replay_schema_version"]
        == "eiel_torch_cpu_mt19937_replay_v1"
    )
    _assert_json_safe(receipt)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_crossover",
        "legacy_dense_crossover",
        "bad_crossover_dtype",
        "bad_crossover_row_width",
        "unsupported_encoding",
        "bad_mask_dtype",
        "bad_post_shape",
        "base_ahead",
    ],
)
def test_fixed_mlp_policy_window_fails_closed(mutation: str) -> None:
    policy = _policy_state(cells=1)
    base = torch.zeros((1,), dtype=torch.int64)
    if mutation == "missing_crossover":
        del policy["policy_birth_crossover_mask_packed_tape"]
    elif mutation == "legacy_dense_crossover":
        policy["policy_birth_crossover_mask_tape"] = torch.zeros(
            (1, 4, 3), dtype=torch.bool
        )
    elif mutation == "bad_crossover_dtype":
        policy["policy_birth_crossover_mask_packed_tape"] = policy[
            "policy_birth_crossover_mask_packed_tape"
        ].to(torch.bool)
    elif mutation == "bad_crossover_row_width":
        policy["policy_birth_crossover_mask_packed_tape"] = torch.zeros(
            (1, 4, 2), dtype=torch.uint8
        )
    elif mutation == "unsupported_encoding":
        policy["policy_birth_crossover_mask_encoding"] = "bitpack_msb0_v1"
    elif mutation == "bad_mask_dtype":
        policy["policy_birth_mutation_mask_tape"] = policy[
            "policy_birth_mutation_mask_tape"
        ].to(torch.uint8)
    elif mutation == "bad_post_shape":
        policy["policy_birth_post_rng_state_packed_tape"] = policy[
            "policy_birth_post_rng_state_packed_tape"
        ][:, :-1]
    else:
        base.fill_(4)
    tapes_before = {
        name: tensor.clone()
        for name, tensor in policy.items()
        if name.startswith("policy_birth_") and isinstance(tensor, torch.Tensor)
    }
    source_before = policy["rng_manager_mutation_torch_cpu_state"].clone()
    semantic_before = policy["mutation_torch_cpu_semantic_draws"].clone()
    base_before = base.clone()
    with pytest.raises(ResidentWindowContractError):
        refill_fixed_mlp_policy_window_(
            policy,
            policy_birth_window_base=base,
            policy_row_width=3,
            policy_weight_mutation_enable=True,
            policy_weight_mutation_mode="sparse",
            policy_weight_mutation_rate=0.5,
            policy_weight_mutation_std=0.125,
        )
    for name, before in tapes_before.items():
        assert torch.equal(policy[name], before), name
    assert torch.equal(policy["rng_manager_mutation_torch_cpu_state"], source_before)
    assert torch.equal(policy["mutation_torch_cpu_semantic_draws"], semantic_before)
    assert torch.equal(base, base_before)


def test_event_windows_two_drains_multicell_and_final_materialization() -> None:
    bank = _event_bank()
    active = torch.tensor([2, 1], dtype=torch.int64)
    base = torch.zeros((2,), dtype=torch.int64)
    status = torch.zeros((2,), dtype=torch.int32)
    archives: list[list[dict[str, torch.Tensor]]] = [[], []]
    _fill_event_rows(bank, cell=0, base=0, count=2, value_offset=100)
    _fill_event_rows(bank, cell=1, base=0, count=1, value_offset=200)
    mutable = {
        **bank,
        "active": active,
        "base": base,
        "status": status,
    }
    pointers = _data_ptrs(mutable)

    first = drain_event_window(
        bank,
        active_count=active,
        event_sequence_base=base,
        event_status=status,
        archives=archives,
    )

    assert active.tolist() == [0, 0]
    assert base.tolist() == [2, 1]
    assert [len(value) for value in archives] == [1, 1]
    assert first["drained_rows_by_cell"] == [2, 1]
    assert _data_ptrs(mutable) == pointers
    _assert_json_safe(first)

    _fill_event_rows(bank, cell=0, base=2, count=1, value_offset=300)
    _fill_event_rows(bank, cell=1, base=1, count=2, value_offset=400)
    active.copy_(torch.tensor([1, 2], dtype=torch.int64))
    second = drain_event_window(
        bank,
        active_count=active,
        event_sequence_base=base,
        event_status=status,
        archives=archives,
    )
    assert active.tolist() == [0, 0]
    assert base.tolist() == [3, 3]
    assert [len(value) for value in archives] == [2, 2]
    assert second["event_sequence_base_before"] == [2, 1]
    assert second["event_sequence_base_after"] == [3, 3]
    assert _data_ptrs(mutable) == pointers
    _assert_json_safe(second)

    _fill_event_rows(bank, cell=0, base=3, count=1, value_offset=500)
    active.copy_(torch.tensor([1, 0], dtype=torch.int64))
    histories, receipt = materialize_event_history(
        archives,
        bank,
        active_count=active,
        event_sequence_base=base,
        event_status=status,
    )
    assert [history["typed_lifecycle_event_sequence"].tolist() for history in histories] == [
        [0, 1, 2, 3],
        [0, 1, 2],
    ]
    assert histories[0]["event_tape_value"].tolist() == pytest.approx(
        [100.05, 101.05, 302.05, 503.05]
    )
    assert histories[1]["event_tape_value"].tolist() == pytest.approx(
        [200.05, 401.05, 402.05]
    )
    for history in histories:
        assert tuple(history) == EVENT_COLUMN_NAMES
        row_count = int(history["typed_lifecycle_event_sequence"].numel())
        for name in EVENT_COLUMN_NAMES:
            assert history[name].device.type == "cpu"
            assert history[name].dtype == EVENT_COLUMN_DTYPES[name]
            assert tuple(history[name].shape) == (row_count,)
    assert receipt["materialized_rows_by_cell"] == [4, 3]
    assert receipt["sequence_exact"] is True
    assert active.tolist() == [1, 0]
    assert base.tolist() == [3, 3]
    assert _data_ptrs(mutable) == pointers
    _assert_json_safe(receipt)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_column",
        "bad_column_dtype",
        "unequal_capacity",
        "count_overflow",
        "bad_status",
        "bad_sequence",
    ],
)
def test_event_window_fails_closed_without_archive_mutation(mutation: str) -> None:
    bank = _event_bank(cells=1, capacity=3)
    active = torch.tensor([1], dtype=torch.int64)
    base = torch.zeros((1,), dtype=torch.int64)
    status = torch.zeros((1,), dtype=torch.int32)
    archives: list[list[dict[str, torch.Tensor]]] = [[]]
    _fill_event_rows(bank, cell=0, base=0, count=1, value_offset=10)
    if mutation == "missing_column":
        del bank["event_tape_target_id"]
    elif mutation == "bad_column_dtype":
        bank["event_tape_type"] = bank["event_tape_type"].to(torch.int64)
    elif mutation == "unequal_capacity":
        bank["event_tape_tick"] = torch.zeros((1, 4), dtype=torch.int64)
    elif mutation == "count_overflow":
        active.fill_(4)
    elif mutation == "bad_status":
        status.fill_(1)
    else:
        bank["typed_lifecycle_event_sequence"][0, 0] = 9
    bank_before = {name: tensor.clone() for name, tensor in bank.items()}
    active_before = active.clone()
    base_before = base.clone()
    status_before = status.clone()
    with pytest.raises(ResidentWindowContractError):
        drain_event_window(
            bank,
            active_count=active,
            event_sequence_base=base,
            event_status=status,
            archives=archives,
        )
    assert archives == [[]]
    for name, before in bank_before.items():
        assert torch.equal(bank[name], before), name
    assert torch.equal(active, active_before)
    assert torch.equal(base, base_before)
    assert torch.equal(status, status_before)


def test_event_archive_base_mismatch_fails_closed() -> None:
    bank = _event_bank(cells=1, capacity=2)
    active = torch.zeros((1,), dtype=torch.int64)
    base = torch.ones((1,), dtype=torch.int64)
    status = torch.zeros((1,), dtype=torch.int32)
    with pytest.raises(ResidentWindowContractError, match="row count"):
        materialize_event_history(
            [[]],
            bank,
            active_count=active,
            event_sequence_base=base,
            event_status=status,
        )
