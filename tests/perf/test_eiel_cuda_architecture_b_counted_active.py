from __future__ import annotations

import copy
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b import (
    ResidentCudaOwnerActionState,
    load_cuda_architecture_b_extension,
)
from perf.cuda_architecture_b_fixed_mlp import (
    fixed_mlp_refill_counted_cuda,
    fixed_mlp_refill_cuda,
)
from perf.cuda_architecture_b_pcg64 import pcg64_state_tensors


ACTIVE = 3
CAPACITY = 6
ACTION_COUNT = 7
CACHE_WIDTH = 2


@pytest.fixture(scope="module")
def owner_module(tmp_path_factory: pytest.TempPathFactory):
    return load_cuda_architecture_b_extension(
        build_directory=tmp_path_factory.mktemp("counted-owner-build")
    )


def _owner_fixture(*, tail: tuple[int, int, int] = (-17, 99_999, -1)) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    ids = torch.tensor((2, 0, 1, *tail), device=device, dtype=torch.int64)
    agent_id_to_slot = torch.tensor((4, 1, 3), device=device, dtype=torch.int32)
    pos_y = torch.zeros(CAPACITY, device=device, dtype=torch.int32)
    pos_x = torch.zeros(CAPACITY, device=device, dtype=torch.int32)
    pos_y[[4, 1, 3]] = torch.tensor((1, 1, 2), device=device, dtype=torch.int32)
    pos_x[[4, 1, 3]] = torch.tensor((1, 2, 1), device=device, dtype=torch.int32)
    agent_grid = torch.full((4, 4), -1, device=device, dtype=torch.int32)
    agent_grid[1, 1] = 0
    agent_grid[1, 2] = 1
    agent_grid[2, 1] = 2
    food_grid = torch.zeros_like(agent_grid)
    food_grid[2, 2] = 1
    patch_food_amount = torch.linspace(0.0, 1.0, 16, device=device).reshape(4, 4)
    return {
        "agent_ids": ids,
        "active_count": torch.tensor([ACTIVE], device=device, dtype=torch.int64),
        "agent_id_to_slot": agent_id_to_slot,
        "pos_y": pos_y,
        "pos_x": pos_x,
        "agent_grid": agent_grid,
        "food_grid": food_grid,
        "energy": torch.tensor((70, 60, 50, 80, 90, 40), device=device, dtype=torch.float32),
        "health": torch.ones(CAPACITY, device=device, dtype=torch.float32),
        "patch_food_amount": patch_food_amount.contiguous(),
        "patch_cover": torch.linspace(0.1, 0.8, 16, device=device).reshape(4, 4).contiguous(),
        "patch_shelter": torch.linspace(0.2, 0.9, 16, device=device).reshape(4, 4).contiguous(),
        "patch_scent": torch.linspace(0.0, 0.5, 16, device=device).reshape(4, 4).contiguous(),
        "patch_waste": torch.linspace(0.5, 0.0, 16, device=device).reshape(4, 4).contiguous(),
        "patch_scent_emit_pending": torch.zeros((4, 4), device=device),
        "patch_waste_emit_pending": torch.zeros((4, 4), device=device),
        "patch_movement_cost": torch.ones((4, 4), device=device),
    }


def _clone_tensors(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in source.items()}


def _observation_call(
    module,
    tensors: dict[str, torch.Tensor],
    *,
    counted: bool,
    patch_enabled: bool = True,
) -> torch.Tensor:
    function = (
        module.eiel_cuda_owner_observation_frame_counted_v1
        if counted
        else module.eiel_cuda_owner_observation_frame_v1
    )
    prefix = (tensors["agent_ids"], tensors["active_count"])
    if not counted:
        prefix = (tensors["agent_ids"][:ACTIVE],)
    return function(
        *prefix,
        tensors["agent_id_to_slot"],
        tensors["pos_y"],
        tensors["pos_x"],
        tensors["agent_grid"],
        tensors["food_grid"],
        tensors["energy"],
        tensors["health"],
        tensors["patch_food_amount"],
        tensors["patch_cover"],
        tensors["patch_shelter"],
        tensors["patch_scent"],
        tensors["patch_waste"],
        tensors["patch_movement_cost"],
        patch_enabled,
        1,
        100.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1,
        False,
        False,
        0.0,
        2.0,
    )


@pytest.mark.parametrize("patch_enabled", [False, True])
def test_batched_observations_match_isolated_counted_cells_exactly(
    owner_module,
    patch_enabled: bool,
) -> None:
    cells = [_owner_fixture(), _owner_fixture()]
    cells[1]["energy"].add_(3.0)
    cells[1]["patch_scent"].mul_(0.5)
    expected = [
        _observation_call(
            owner_module,
            tensors,
            counted=True,
            patch_enabled=patch_enabled,
        )
        for tensors in cells
    ]
    batched = owner_module.eiel_cuda_owner_observation_frame_batched_counted_v1(
        torch.stack([cell["agent_ids"] for cell in cells]).contiguous(),
        torch.stack([cell["active_count"] for cell in cells]).contiguous(),
        torch.stack([cell["agent_id_to_slot"] for cell in cells]).contiguous(),
        torch.stack([cell["pos_y"] for cell in cells]).contiguous(),
        torch.stack([cell["pos_x"] for cell in cells]).contiguous(),
        torch.stack([cell["agent_grid"] for cell in cells]).contiguous(),
        torch.stack([cell["food_grid"] for cell in cells]).contiguous(),
        torch.stack([cell["energy"] for cell in cells]).contiguous(),
        torch.stack([cell["health"] for cell in cells]).contiguous(),
        torch.stack([cell["patch_food_amount"] for cell in cells]).contiguous(),
        torch.stack([cell["patch_cover"] for cell in cells]).contiguous(),
        torch.stack([cell["patch_shelter"] for cell in cells]).contiguous(),
        torch.stack([cell["patch_scent"] for cell in cells]).contiguous(),
        torch.stack([cell["patch_waste"] for cell in cells]).contiguous(),
        torch.stack([cell["patch_movement_cost"] for cell in cells]).contiguous(),
        patch_enabled,
        1,
        100.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1,
        False,
        False,
        0.0,
        2.0,
    )
    assert batched.shape[0] == len(cells)
    for index, isolated in enumerate(expected):
        assert torch.equal(batched[index], isolated)


def _action_call(module, tensors: dict[str, torch.Tensor], *, counted: bool):
    actions = torch.full((2, CAPACITY), 5, device="cuda", dtype=torch.int32)
    function = (
        module.eiel_cuda_scripted_owner_action_generation_counted_v1
        if counted
        else module.eiel_cuda_scripted_owner_action_generation_v1
    )
    prefix = (tensors["agent_ids"], tensors["active_count"])
    if not counted:
        prefix = (tensors["agent_ids"][:ACTIVE],)
    return function(
        *prefix,
        tensors["agent_id_to_slot"],
        actions,
        tensors["pos_y"],
        tensors["pos_x"],
        tensors["agent_grid"],
        tensors["food_grid"],
        tensors["patch_food_amount"],
        tensors["patch_scent_emit_pending"],
        tensors["patch_waste_emit_pending"],
        tensors["patch_movement_cost"],
        tensors["energy"],
        tensors["health"],
        False,
        1.0,
        1.0,
        0.5,
        1.0,
        1.0,
        1.0,
        100.0,
        20.0,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        False,
        0.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_counted_observation_matches_compact_prefix_and_ignores_tail(owner_module) -> None:
    baseline = _owner_fixture()
    compact = _observation_call(owner_module, baseline, counted=False)
    counted = _observation_call(owner_module, baseline, counted=True)
    alternate_tail = _observation_call(
        owner_module,
        _owner_fixture(tail=(2**40, -2**40, 777)),
        counted=True,
    )

    assert tuple(counted.shape) == (CAPACITY, compact.size(1))
    torch.testing.assert_close(counted[:ACTIVE], compact, rtol=0, atol=0)
    assert torch.equal(counted[ACTIVE:], torch.zeros_like(counted[ACTIVE:]))
    assert torch.equal(counted, alternate_tail)

    zero = _owner_fixture(tail=(2**40, -2**40, 777))
    zero["agent_ids"].fill_(-999)
    zero["active_count"].zero_()
    zero_output = _observation_call(owner_module, zero, counted=True)
    assert torch.equal(zero_output, torch.zeros_like(zero_output))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_counted_action_matches_compact_prefix_isolates_tail_and_keeps_pointers(owner_module) -> None:
    compact_tensors = _owner_fixture()
    counted_tensors = _clone_tensors(compact_tensors)
    alternate_tensors = _owner_fixture(tail=(2**40, -2**40, 777))
    energy_pointer = counted_tensors["energy"].data_ptr()

    compact = _action_call(owner_module, compact_tensors, counted=False)
    counted = _action_call(owner_module, counted_tensors, counted=True)
    alternate = _action_call(owner_module, alternate_tensors, counted=True)

    assert counted[0].data_ptr() == energy_pointer
    for index in range(1, 4):
        torch.testing.assert_close(counted[index][:ACTIVE], compact[index], rtol=0, atol=0)
        assert torch.equal(counted[index][ACTIVE:], torch.zeros_like(counted[index][ACTIVE:]))
    for index in (0, 4, 5):
        torch.testing.assert_close(counted[index], compact[index], rtol=0, atol=0)
    assert torch.equal(counted[6][:ACTIVE], compact[6])
    assert torch.equal(counted[6][ACTIVE:], torch.full_like(counted[6][ACTIVE:], -1))
    for name in (
        "pos_y",
        "pos_x",
        "agent_grid",
        "food_grid",
        "patch_food_amount",
        "patch_scent_emit_pending",
        "patch_waste_emit_pending",
        "energy",
    ):
        assert torch.equal(counted_tensors[name], compact_tensors[name]), name
    for left, right in zip(counted, alternate, strict=True):
        assert torch.equal(left, right)

    zero = _owner_fixture(tail=(2**40, -2**40, 777))
    zero["agent_ids"].fill_(-999)
    zero["active_count"].zero_()
    before = _clone_tensors(zero)
    zero_outputs = _action_call(owner_module, zero, counted=True)
    for name in ("pos_y", "pos_x", "agent_grid", "food_grid", "energy"):
        assert torch.equal(zero[name], before[name]), name
    assert torch.equal(zero_outputs[1], torch.zeros_like(zero_outputs[1]))
    assert torch.equal(zero_outputs[2], torch.zeros_like(zero_outputs[2]))
    assert torch.equal(zero_outputs[3], torch.zeros_like(zero_outputs[3]))
    assert torch.equal(zero_outputs[6], torch.full_like(zero_outputs[6], -1))


def _mlp_fixture(*, tail_slots: tuple[int, int, int] = (-10, 1000, -1)) -> dict[str, torch.Tensor]:
    device = torch.device("cuda")
    observation_size = 4
    hidden_size = 3
    packed_width = (
        hidden_size * observation_size
        + hidden_size
        + hidden_size * hidden_size
        + hidden_size
        + ACTION_COUNT * hidden_size
        + ACTION_COUNT
    )
    generator = torch.Generator(device=device).manual_seed(710_202_607)
    observations = torch.randn(
        (CAPACITY, observation_size), generator=generator, device=device
    )
    observations[ACTIVE:] = torch.nan
    return {
        "observations": observations.contiguous(),
        "slots": torch.tensor((4, 1, 5, *tail_slots), device=device, dtype=torch.int32),
        "active_count": torch.tensor([ACTIVE], device=device, dtype=torch.int64),
        "mlp_pack": torch.randn(
            (CAPACITY, packed_width), generator=generator, device=device
        ).contiguous(),
        "drives": torch.rand((CAPACITY, 4), generator=generator, device=device).contiguous(),
        "attack_bias": torch.linspace(-0.2, 0.3, CAPACITY, device=device),
        "action_temperature": torch.linspace(0.2, 1.2, CAPACITY, device=device),
        "drive_logit_gain": torch.linspace(0.1, 0.8, CAPACITY, device=device),
        "slot_tokens": torch.arange(50, 50 + CAPACITY, device=device, dtype=torch.int64),
        "action_cache_actions": torch.full((CAPACITY, 3), -1, device=device, dtype=torch.int32),
        "action_cache_token": torch.full((CAPACITY,), -1, device=device, dtype=torch.int64),
        "action_cache_cursor": torch.full((CAPACITY,), -1, device=device, dtype=torch.int32),
        "action_cache_next": torch.full((CAPACITY,), -1, device=device, dtype=torch.int32),
    }


def _mlp_call(
    tensors: dict[str, torch.Tensor],
    build: Path,
    *,
    counted: bool,
    state_words: torch.Tensor,
    buffer_words: torch.Tensor,
):
    common = {
        name: tensors[name]
        for name in (
            "mlp_pack",
            "drives",
            "attack_bias",
            "action_temperature",
            "drive_logit_gain",
            "slot_tokens",
            "action_cache_actions",
            "action_cache_token",
            "action_cache_cursor",
            "action_cache_next",
        )
    }
    common.update(
        hidden_size=3,
        action_count=ACTION_COUNT,
        cache_width=CACHE_WIDTH,
        deterministic=False,
        rng_counter=70,
        pcg64_state_words=state_words,
        pcg64_buffer_words=buffer_words,
        build_directory=build,
    )
    if counted:
        return fixed_mlp_refill_counted_cuda(
            observations=tensors["observations"],
            slots=tensors["slots"],
            active_count=tensors["active_count"],
            **common,
        )
    return fixed_mlp_refill_cuda(
        observations=tensors["observations"][:ACTIVE],
        slots=tensors["slots"][:ACTIVE],
        **common,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_counted_fixed_mlp_matches_compact_exact_pcg_and_zero_count(tmp_path: Path) -> None:
    initial_state = copy.deepcopy(np.random.PCG64(303).state)
    compact_tensors = _mlp_fixture()
    counted_tensors = _clone_tensors(compact_tensors)
    compact_state, compact_buffer = pcg64_state_tensors(initial_state, device="cuda")
    counted_state, counted_buffer = pcg64_state_tensors(initial_state, device="cuda")
    state_pointer = counted_state.data_ptr()
    buffer_pointer = counted_buffer.data_ptr()
    cache_pointer = counted_tensors["action_cache_actions"].data_ptr()

    compact = _mlp_call(
        compact_tensors,
        tmp_path / "fixed-mlp-build",
        counted=False,
        state_words=compact_state,
        buffer_words=compact_buffer,
    )
    counted = _mlp_call(
        counted_tensors,
        tmp_path / "fixed-mlp-build",
        counted=True,
        state_words=counted_state,
        buffer_words=counted_buffer,
    )
    alternate_tensors = _mlp_fixture(tail_slots=(2**20, -2**20, 999))
    alternate_state, alternate_buffer = pcg64_state_tensors(initial_state, device="cuda")
    alternate = _mlp_call(
        alternate_tensors,
        tmp_path / "fixed-mlp-build",
        counted=True,
        state_words=alternate_state,
        buffer_words=alternate_buffer,
    )

    torch.testing.assert_close(counted.logits[:ACTIVE], compact.logits, rtol=0, atol=0)
    assert torch.equal(counted.actions[:ACTIVE], compact.actions)
    assert torch.equal(counted.logits[ACTIVE:], torch.zeros_like(counted.logits[ACTIVE:]))
    assert torch.equal(counted.actions[ACTIVE:], torch.full_like(counted.actions[ACTIVE:], -1))
    for name in (
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
        "next_rng_counter",
        "pcg64_state_words",
        "pcg64_buffer_words",
    ):
        assert torch.equal(getattr(counted, name), getattr(compact, name)), name
        assert torch.equal(getattr(counted, name), getattr(alternate, name)), name
    assert torch.equal(counted.logits, alternate.logits)
    assert torch.equal(counted.actions, alternate.actions)
    assert counted.action_cache_actions.data_ptr() == cache_pointer
    assert counted.pcg64_state_words.data_ptr() == state_pointer
    assert counted.pcg64_buffer_words.data_ptr() == buffer_pointer

    zero_tensors = _mlp_fixture(tail_slots=(2**20, -2**20, 999))
    zero_tensors["slots"].fill_(-999)
    zero_tensors["active_count"].zero_()
    zero_state, zero_buffer = pcg64_state_tensors(initial_state, device="cuda")
    initial_words = zero_state.clone()
    initial_buffer = zero_buffer.clone()
    initial_cache = zero_tensors["action_cache_actions"].clone()
    zero = _mlp_call(
        zero_tensors,
        tmp_path / "fixed-mlp-build",
        counted=True,
        state_words=zero_state,
        buffer_words=zero_buffer,
    )
    assert torch.equal(zero.logits, torch.zeros_like(zero.logits))
    assert torch.equal(zero.actions, torch.full_like(zero.actions, -1))
    assert torch.equal(zero.action_cache_actions, initial_cache)
    assert torch.equal(zero.pcg64_state_words, initial_words)
    assert torch.equal(zero.pcg64_buffer_words, initial_buffer)


def test_counted_entrypoints_have_static_no_sync_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    native_sources = (
        root / "engine/perf/native/cxxbackbone8r_generation_island_cuda.cu",
        root / "engine/perf/native/cuda_architecture_b_fixed_mlp.cu",
    )
    for source_path in native_sources:
        source = source_path.read_text(encoding="utf-8")
        for forbidden in (".item", ".cpu(", "torch::equal", "masked_select", "nonzero"):
            assert forbidden not in source, (source_path.name, forbidden)
        assert "active_count.data_ptr<int64_t>()" in source

    counted_python = "\n".join(
        (
            inspect.getsource(fixed_mlp_refill_counted_cuda),
            inspect.getsource(ResidentCudaOwnerActionState.execute_scripted_generation_counted),
            inspect.getsource(ResidentCudaOwnerActionState.build_observation_frame_counted),
        )
    )
    for forbidden in (".item(", ".cpu(", "torch.equal", "masked_select", "nonzero"):
        assert forbidden not in counted_python
    assert "eiel_cuda_fixed_mlp_refill_counted_v1" in counted_python
    assert "eiel_cuda_scripted_owner_action_generation_counted_v1" in counted_python
    assert "eiel_cuda_owner_observation_frame_counted_v1" in counted_python
