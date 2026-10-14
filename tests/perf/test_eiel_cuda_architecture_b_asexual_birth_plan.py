from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b_asexual_birth_plan import (
    ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT,
    ASEXUAL_BIRTH_PLAN_STATUS,
    ASEXUAL_BIRTH_REASON,
    execute_asexual_birth_plan_cuda,
)
from perf.cuda_architecture_b_pcg64 import pcg64_state_tensors


REASON_ACCEPTED = 0
REASON_NO_SPACE = 1
REASON_BUFFER = 2
REASON_NO_FREE_SLOT = 3
REASON_CONCEPTION = 5
REASON_GESTATION = 6
REASON_HEALTH = 7
REASON_COOLDOWN = 8
REASON_NO_ENERGY = 9

DEFAULT_GATE_COEFFICIENTS = {
    "injury_repro_k_injury": 0.35,
    "injury_repro_k_scar": 0.9,
    "injury_birth_fail_scar": 0.12,
    "injury_birth_fail_injury": 0.03,
}


def _pcg64_state_from_tensors(
    state_words: torch.Tensor,
    buffer_words: torch.Tensor,
) -> dict[str, object]:
    words = [int(value) for value in state_words.detach().cpu().tolist()]
    buffer = [int(value) for value in buffer_words.detach().cpu().tolist()]
    return {
        "bit_generator": "PCG64",
        "state": {
            "state": words[0] | (words[1] << 64),
            "inc": words[2] | (words[3] << 64),
        },
        "has_uint32": buffer[0],
        "uinteger": buffer[1],
    }


def _pcg64_continuation(state: dict[str, object]) -> np.ndarray:
    bit_generator = np.random.PCG64()
    bit_generator.state = copy.deepcopy(state)
    return np.random.Generator(bit_generator).integers(
        0,
        1 << 32,
        size=9,
        dtype=np.uint32,
    )


def _make_state(
    *,
    seed: int,
    parents: list[dict[str, Any]],
    free_queue: list[int],
    capacity: int = 12,
    shape: tuple[int, int] = (5, 5),
    buffered_uint32: bool = False,
    runner_seed: int = 909,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    bit_generator = np.random.PCG64(seed)
    if buffered_uint32:
        np.random.Generator(bit_generator).integers(0, 7, dtype=np.int32)
        assert int(bit_generator.state["has_uint32"]) == 1
    initial_rng_state = copy.deepcopy(bit_generator.state)
    pcg64_state_words, pcg64_buffer_words = pcg64_state_tensors(
        initial_rng_state,
        device="cuda",
    )
    runner_bit_generator = np.random.PCG64(runner_seed)
    np.random.Generator(runner_bit_generator).integers(0, 7, dtype=np.int32)
    assert int(runner_bit_generator.state["has_uint32"]) == 1
    runner_pcg64_state_words, runner_pcg64_buffer_words = pcg64_state_tensors(
        copy.deepcopy(runner_bit_generator.state),
        device="cuda",
    )

    slot_to_agent_id = torch.full(
        (capacity,), -1, dtype=torch.int64, device="cuda"
    )
    alive_mask = torch.zeros((capacity,), dtype=torch.bool, device="cuda")
    policy_use_gru = torch.zeros((capacity,), dtype=torch.bool, device="cuda")
    pos_y = torch.full((capacity,), -1, dtype=torch.int32, device="cuda")
    pos_x = torch.full((capacity,), -1, dtype=torch.int32, device="cuda")
    energy = torch.zeros((capacity,), dtype=torch.float32, device="cuda")
    policy_energy = torch.full(
        (capacity,), float("nan"), dtype=torch.float64, device="cuda"
    )
    health = torch.zeros((capacity,), dtype=torch.float32, device="cuda")
    injury = torch.zeros((capacity,), dtype=torch.float32, device="cuda")
    scar = torch.zeros((capacity,), dtype=torch.float32, device="cuda")
    cooldown = torch.zeros((capacity,), dtype=torch.int32, device="cuda")
    threshold = torch.zeros((capacity,), dtype=torch.float32, device="cuda")
    grid = torch.full(shape, -1, dtype=torch.int32, device="cuda")

    for row in parents:
        slot = int(row["slot"])
        agent_id = int(row["agent_id"])
        y, x = (int(value) for value in row["position"])
        slot_to_agent_id[slot] = agent_id
        alive_mask[slot] = True
        pos_y[slot] = y
        pos_x[slot] = x
        energy[slot] = float(row.get("energy", 60.0))
        policy_energy[slot] = float(row.get("policy_energy", row.get("energy", 60.0)))
        health[slot] = float(row.get("health", 1.0))
        injury[slot] = float(row.get("injury", 0.0))
        scar[slot] = float(row.get("scar", 0.0))
        cooldown[slot] = int(row.get("cooldown", 0))
        threshold[slot] = float(row.get("threshold", 20.0))
        policy_use_gru[slot] = bool(row.get("use_gru", False))
        grid[y, x] = agent_id

    return (
        {
            "slot_to_agent_id": slot_to_agent_id,
            "alive_mask": alive_mask,
            "policy_use_gru": policy_use_gru,
            "pos_y": pos_y,
            "pos_x": pos_x,
            "agent_grid": grid,
            "energy": energy,
            "policy_energy_observation": policy_energy,
            "health": health,
            "injury": injury,
            "scar": scar,
            "repro_cooldown": cooldown,
            "repro_threshold": threshold,
            "slot_free_queue": torch.tensor(
                free_queue, dtype=torch.int64, device="cuda"
            ),
            "slot_free_queue_head": torch.zeros(
                (1,), dtype=torch.int64, device="cuda"
            ),
            "pcg64_state_words": pcg64_state_words,
            "pcg64_buffer_words": pcg64_buffer_words,
            "policy_np_semantic_draws": torch.tensor(
                [31], dtype=torch.int64, device="cuda"
            ),
            "runner_pcg64_state_words": runner_pcg64_state_words,
            "runner_pcg64_buffer_words": runner_pcg64_buffer_words,
            "metrics_np_semantic_draws": torch.tensor(
                [17], dtype=torch.int64, device="cuda"
            ),
        },
        initial_rng_state,
    )


def _cpu_owner_oracle(
    state: dict[str, torch.Tensor],
    initial_rng_state: dict[str, object],
    *,
    parent_energy_share: float,
    min_energy_buffer: float,
    max_births: int,
    env_policy_alias: bool,
) -> dict[str, Any]:
    runner_rng_state = _pcg64_state_from_tensors(
        state["runner_pcg64_state_words"],
        state["runner_pcg64_buffer_words"],
    )
    arrays = {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in state.items()
        if name
        not in {
            "pcg64_state_words",
            "pcg64_buffer_words",
            "runner_pcg64_state_words",
            "runner_pcg64_buffer_words",
        }
    }
    bit_generator = np.random.PCG64()
    bit_generator.state = copy.deepcopy(initial_rng_state)
    rng = np.random.Generator(bit_generator)
    parent_slots: list[int] = []
    child_slots: list[int] = []
    spawn_positions: list[tuple[int, int]] = []
    child_energy: list[np.float32] = []
    decisions: list[tuple[int, int, int, int, int, int, int, bool]] = []
    reason_counts = np.zeros((10,), dtype=np.int64)
    draw_count = 0
    semantic_draws = 0
    free_head = int(arrays["slot_free_queue_head"][0])
    free_queue = arrays["slot_free_queue"].tolist()
    height, width = arrays["agent_grid"].shape
    alive_slots = [
        slot
        for slot in range(len(arrays["alive_mask"]))
        if bool(arrays["alive_mask"][slot])
        and int(arrays["slot_to_agent_id"][slot]) >= 0
    ]
    alive_slots.sort(key=lambda slot: int(arrays["slot_to_agent_id"][slot]))

    for parent_slot in alive_slots:
        if len(parent_slots) >= max_births:
            break
        draw_start = draw_count
        reason = REASON_ACCEPTED
        if np.float32(arrays["health"][parent_slot]) <= np.float32(0.0):
            reason = REASON_HEALTH
        else:
            if int(arrays["repro_cooldown"][parent_slot]) > 0:
                arrays["repro_cooldown"][parent_slot] -= np.int32(1)
            if int(arrays["repro_cooldown"][parent_slot]) != 0:
                reason = REASON_COOLDOWN
            elif np.float32(arrays["energy"][parent_slot]) < np.float32(
                arrays["repro_threshold"][parent_slot]
            ):
                reason = REASON_NO_ENERGY

        candidates: list[tuple[int, int]] = []
        selected = (-1, -1)
        selected_reserved = False
        if reason == REASON_ACCEPTED:
            y = int(arrays["pos_y"][parent_slot])
            x = int(arrays["pos_x"][parent_slot])
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and int(arrays["agent_grid"][ny, nx]) == -1
                ):
                    candidates.append((ny, nx))
            semantic_draws += len(candidates)
            indices = np.arange(len(candidates), dtype=np.int32)
            rng.shuffle(indices)
            draw_count += max(0, len(candidates) - 1)
            if candidates:
                selected = candidates[int(indices[0])]
                selected_reserved = selected in spawn_positions
            if not candidates or selected_reserved:
                reason = REASON_NO_SPACE

        if reason == REASON_ACCEPTED:
            observed = float(arrays["policy_energy_observation"][parent_slot])
            parent_energy = (
                observed
                if np.isfinite(observed)
                else float(np.float32(arrays["energy"][parent_slot]))
            )
            share = parent_energy * float(parent_energy_share)
            if (parent_energy - share) < float(np.float32(min_energy_buffer)):
                reason = REASON_BUFFER
            else:
                child_slot = -1
                while free_head < len(free_queue):
                    candidate_slot = int(free_queue[free_head])
                    free_head += 1
                    if (
                        0 <= candidate_slot < len(arrays["alive_mask"])
                        and not bool(arrays["alive_mask"][candidate_slot])
                        and int(arrays["slot_to_agent_id"][candidate_slot]) < 0
                        and candidate_slot not in child_slots
                    ):
                        child_slot = candidate_slot
                        break
                if child_slot < 0:
                    reason = REASON_NO_FREE_SLOT
                else:
                    parent_slots.append(parent_slot)
                    child_slots.append(child_slot)
                    spawn_positions.append(selected)
                    child_energy.append(np.float32(share))

        reason_counts[reason] += 1
        decisions.append(
            (
                parent_slot,
                reason,
                len(candidates),
                draw_start,
                draw_count,
                selected[0],
                selected[1],
                selected_reserved,
            )
        )
        if reason == REASON_NO_FREE_SLOT:
            break

    semantic_total = int(arrays["policy_np_semantic_draws"][0])
    if env_policy_alias:
        semantic_total += semantic_draws
    return {
        "repro_cooldown": arrays["repro_cooldown"],
        "free_head": free_head,
        "rng_state": copy.deepcopy(bit_generator.state),
        "runner_rng_state": runner_rng_state,
        "policy_semantic": semantic_total,
        "metrics_semantic": int(arrays["metrics_np_semantic_draws"][0]),
        "parent_slots": parent_slots,
        "child_slots": child_slots,
        "spawn_positions": spawn_positions,
        "child_energy": child_energy,
        "decisions": decisions,
        "reason_counts": reason_counts,
        "draw_count": draw_count,
        "semantic_draws": semantic_draws,
    }


def _execute_and_compare(
    state: dict[str, torch.Tensor],
    initial_rng_state: dict[str, object],
    *,
    parent_energy_share: float = 0.5,
    min_energy_buffer: float = 1.0,
    max_births: int = 8,
    env_policy_alias: bool = True,
    build_directory: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    expected = _cpu_owner_oracle(
        state,
        initial_rng_state,
        parent_energy_share=parent_energy_share,
        min_energy_buffer=min_energy_buffer,
        max_births=max_births,
        env_policy_alias=env_policy_alias,
    )
    mutable = (
        "repro_cooldown",
        "slot_free_queue_head",
        "pcg64_state_words",
        "pcg64_buffer_words",
        "policy_np_semantic_draws",
        "runner_pcg64_state_words",
        "runner_pcg64_buffer_words",
        "metrics_np_semantic_draws",
    )
    pointers = {name: state[name].data_ptr() for name in mutable}
    result = execute_asexual_birth_plan_cuda(
        **state,
        parent_energy_share=parent_energy_share,
        min_energy_buffer=min_energy_buffer,
        max_births=max_births,
        env_policy_alias=env_policy_alias,
        build_directory=build_directory,
    )
    assert int(result["status"].item()) == 0
    assert all(state[name].data_ptr() == pointers[name] for name in mutable)
    planned = int(result["planned_count"].item())
    decisions = int(result["decision_count"].item())
    assert planned == len(expected["parent_slots"])
    assert decisions == len(expected["decisions"])
    assert result["parent_slots"][:planned].cpu().tolist() == expected["parent_slots"]
    assert result["child_slots"][:planned].cpu().tolist() == expected["child_slots"]
    assert [
        tuple(int(value) for value in row)
        for row in result["spawn_positions"][:planned].cpu().tolist()
    ] == expected["spawn_positions"]
    np.testing.assert_array_equal(
        result["child_energy"][:planned].cpu().numpy(),
        np.asarray(expected["child_energy"], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        state["repro_cooldown"].cpu().numpy(), expected["repro_cooldown"]
    )
    assert int(state["slot_free_queue_head"].item()) == expected["free_head"]
    assert _pcg64_state_from_tensors(
        state["pcg64_state_words"], state["pcg64_buffer_words"]
    ) == expected["rng_state"]
    assert int(state["policy_np_semantic_draws"].item()) == expected["policy_semantic"]
    assert _pcg64_state_from_tensors(
        state["runner_pcg64_state_words"],
        state["runner_pcg64_buffer_words"],
    ) == expected["runner_rng_state"]
    assert (
        int(state["metrics_np_semantic_draws"].item())
        == expected["metrics_semantic"]
    )
    for name in (
        "runner_pcg64_state_words",
        "runner_pcg64_buffer_words",
        "metrics_np_semantic_draws",
    ):
        assert result[name].data_ptr() == state[name].data_ptr()
    assert int(result["spawn_rng_draws_added"].item()) == expected["draw_count"]
    assert (
        int(result["spawn_rng_semantic_draws_added"].item())
        == expected["semantic_draws"]
    )
    assert int(result["runner_rng_draws_added"].item()) == 0
    assert int(result["runner_rng_semantic_draws_added"].item()) == 0
    assert result["reason_counts"].cpu().tolist() == expected["reason_counts"].tolist()

    actual_decisions = list(
        zip(
            result["decision_parent_slots"][:decisions].cpu().tolist(),
            result["decision_reason_codes"][:decisions].cpu().tolist(),
            result["decision_candidate_counts"][:decisions].cpu().tolist(),
            result["decision_draw_starts"][:decisions].cpu().tolist(),
            result["decision_draw_ends"][:decisions].cpu().tolist(),
            result["decision_selected_positions"][:decisions, 0].cpu().tolist(),
            result["decision_selected_positions"][:decisions, 1].cpu().tolist(),
            result["decision_selected_reserved"][:decisions].cpu().tolist(),
            strict=True,
        )
    )
    assert actual_decisions == expected["decisions"]
    return result, expected


def _conflict_seed(parents: list[dict[str, Any]]) -> int:
    for seed in range(512):
        state, initial = _make_state(
            seed=seed,
            parents=parents,
            free_queue=[6, 7, 8],
            buffered_uint32=True,
        )
        expected = _cpu_owner_oracle(
            state,
            initial,
            parent_energy_share=0.5,
            min_energy_buffer=1.0,
            max_births=8,
            env_policy_alias=True,
        )
        if any(bool(row[7]) for row in expected["decisions"]):
            return seed
    raise AssertionError("bounded seed search did not expose a reserved-tile conflict")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_asexual_plan_matches_selected_first_reserved_failure_and_filters(
    tmp_path: Path,
) -> None:
    parents = [
        {"slot": 4, "agent_id": 40, "position": (0, 0), "cooldown": 2},
        {"slot": 1, "agent_id": 10, "position": (2, 1)},
        {"slot": 5, "agent_id": 50, "position": (4, 0), "energy": 10.0},
        {"slot": 3, "agent_id": 30, "position": (4, 4), "health": 0.0},
        {"slot": 2, "agent_id": 20, "position": (2, 3)},
    ]
    seed = _conflict_seed(parents)
    state, initial = _make_state(
        seed=seed,
        parents=parents,
        free_queue=[6, 7, 8],
        buffered_uint32=True,
    )
    result, _ = _execute_and_compare(
        state,
        initial,
        build_directory=tmp_path / "build",
    )
    decisions = int(result["decision_count"].item())
    reasons = result["decision_reason_codes"][:decisions].cpu().tolist()
    assert REASON_NO_SPACE in reasons
    assert REASON_HEALTH in reasons
    assert REASON_COOLDOWN in reasons
    assert REASON_NO_ENERGY in reasons
    conflict = reasons.index(REASON_NO_SPACE)
    assert bool(result["decision_selected_reserved"][conflict].item()) is True
    assert tuple(result["decision_selected_positions"][conflict].cpu().tolist()) == tuple(
        result["spawn_positions"][0].cpu().tolist()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_asexual_plan_matches_buffer_then_success_then_free_slot_exhaustion(
    tmp_path: Path,
) -> None:
    parents = [
        {"slot": 0, "agent_id": 10, "position": (0, 0), "energy": 40.0},
        {"slot": 1, "agent_id": 20, "position": (2, 2), "energy": 60.0},
        {"slot": 2, "agent_id": 30, "position": (4, 4), "energy": 60.0},
        {"slot": 3, "agent_id": 40, "position": (0, 4), "energy": 60.0},
    ]
    state, initial = _make_state(
        seed=303,
        parents=parents,
        free_queue=[-5, 1, 7, 7, 99],
        buffered_uint32=True,
    )
    result, _ = _execute_and_compare(
        state,
        initial,
        min_energy_buffer=25.0,
        build_directory=tmp_path / "build",
    )
    decisions = int(result["decision_count"].item())
    assert result["decision_reason_codes"][:decisions].cpu().tolist() == [
        REASON_BUFFER,
        REASON_ACCEPTED,
        REASON_NO_FREE_SLOT,
    ]
    assert int(result["planned_count"].item()) == 1
    assert int(state["slot_free_queue_head"].item()) == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_asexual_plan_max_births_stops_before_later_cooldown_decrement(
    tmp_path: Path,
) -> None:
    parents = [
        {"slot": 0, "agent_id": 10, "position": (0, 0)},
        {"slot": 1, "agent_id": 20, "position": (4, 4), "cooldown": 2},
    ]
    state, initial = _make_state(seed=101, parents=parents, free_queue=[5, 6])
    result, _ = _execute_and_compare(
        state,
        initial,
        max_births=1,
        build_directory=tmp_path / "build",
    )
    assert int(result["decision_count"].item()) == 1
    assert int(state["repro_cooldown"][1].item()) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_asexual_plan_ignores_valid_slots_beyond_resident_queue_count(
    tmp_path: Path,
) -> None:
    parents = [
        {"slot": 0, "agent_id": 10, "position": (0, 0)},
        {"slot": 1, "agent_id": 20, "position": (4, 4)},
    ]
    state, _ = _make_state(seed=101, parents=parents, free_queue=[5, 6])
    result = execute_asexual_birth_plan_cuda(
        **state,
        slot_free_queue_count=torch.tensor([1], dtype=torch.int64, device="cuda"),
        parent_energy_share=0.5,
        min_energy_buffer=1.0,
        max_births=2,
        build_directory=tmp_path / "build",
    )

    assert int(result["status"].cpu()[0]) == 0
    assert int(result["planned_count"].cpu()[0]) == 1
    assert int(result["decision_count"].cpu()[0]) == 2
    assert result["decision_reason_codes"][:2].cpu().tolist() == [
        REASON_ACCEPTED,
        REASON_NO_FREE_SLOT,
    ]
    assert int(result["child_slots"][0].cpu()) == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_asexual_plan_nonaliased_stream_preserves_counter_but_advances_rng(
    tmp_path: Path,
) -> None:
    state, initial = _make_state(
        seed=202,
        parents=[{"slot": 0, "agent_id": 10, "position": (2, 2)}],
        free_queue=[5],
    )
    result, _ = _execute_and_compare(
        state,
        initial,
        env_policy_alias=False,
        build_directory=tmp_path / "build",
    )
    assert int(state["policy_np_semantic_draws"].item()) == 31
    assert int(result["spawn_rng_semantic_draws_added"].item()) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("runner_seed", "injury", "scar", "expected_reason", "expected_draws"),
    (
        pytest.param(909, 0.0, 0.0, REASON_ACCEPTED, 2, id="accepted"),
        pytest.param(909, 1.0, 1.0, REASON_CONCEPTION, 1, id="conception"),
        pytest.param(0, 0.0, 1.0, REASON_GESTATION, 2, id="gestation"),
    ),
)
def test_asexual_plan_runner_gates_match_numpy_pcg64_exactly(
    tmp_path: Path,
    runner_seed: int,
    injury: float,
    scar: float,
    expected_reason: int,
    expected_draws: int,
) -> None:
    state, initial_spawn_state = _make_state(
        seed=202,
        runner_seed=runner_seed,
        parents=[
            {
                "slot": 0,
                "agent_id": 10,
                "position": (2, 2),
                "injury": injury,
                "scar": scar,
            }
        ],
        free_queue=[5],
    )
    expected_spawn = _cpu_owner_oracle(
        state,
        initial_spawn_state,
        parent_energy_share=0.5,
        min_energy_buffer=1.0,
        max_births=1,
        env_policy_alias=True,
    )
    initial_runner_state = _pcg64_state_from_tensors(
        state["runner_pcg64_state_words"],
        state["runner_pcg64_buffer_words"],
    )
    initial_runner_buffer = state["runner_pcg64_buffer_words"].clone()
    initial_metrics_draws = int(state["metrics_np_semantic_draws"].item())

    expected_runner_bit_generator = np.random.PCG64()
    expected_runner_bit_generator.state = copy.deepcopy(initial_runner_state)
    expected_runner_rng = np.random.Generator(expected_runner_bit_generator)
    expected_runner_rng.random(expected_draws)
    expected_runner_state = copy.deepcopy(expected_runner_bit_generator.state)

    result = execute_asexual_birth_plan_cuda(
        **state,
        parent_energy_share=0.5,
        min_energy_buffer=1.0,
        max_births=1,
        conception_gate_active=True,
        gestation_gate_active=True,
        build_directory=tmp_path / "build",
        **DEFAULT_GATE_COEFFICIENTS,
    )

    assert int(result["status"].item()) == 0
    assert int(result["decision_count"].item()) == 1
    assert int(result["decision_reason_codes"][0].item()) == expected_reason
    assert int(result["reason_counts"].sum().item()) == 1
    assert int(result["reason_counts"][expected_reason].item()) == 1
    assert int(result["planned_count"].item()) == int(
        expected_reason == REASON_ACCEPTED
    )
    assert int(state["slot_free_queue_head"].item()) == int(
        expected_reason == REASON_ACCEPTED
    )

    assert _pcg64_state_from_tensors(
        state["pcg64_state_words"], state["pcg64_buffer_words"]
    ) == expected_spawn["rng_state"]
    assert int(result["spawn_rng_draws_added"].item()) == expected_spawn["draw_count"]
    assert (
        int(result["spawn_rng_semantic_draws_added"].item())
        == expected_spawn["semantic_draws"]
    )
    assert int(state["policy_np_semantic_draws"].item()) == expected_spawn[
        "policy_semantic"
    ]

    actual_runner_state = _pcg64_state_from_tensors(
        result["runner_pcg64_state_words"],
        result["runner_pcg64_buffer_words"],
    )
    assert actual_runner_state == expected_runner_state
    np.testing.assert_array_equal(
        _pcg64_continuation(actual_runner_state),
        _pcg64_continuation(expected_runner_state),
    )
    assert torch.equal(result["runner_pcg64_buffer_words"], initial_runner_buffer)
    assert int(result["runner_rng_draws_added"].item()) == expected_draws
    assert int(result["runner_rng_semantic_draws_added"].item()) == expected_draws
    assert int(result["metrics_np_semantic_draws"].item()) == (
        initial_metrics_draws + expected_draws
    )
    for name in (
        "runner_pcg64_state_words",
        "runner_pcg64_buffer_words",
        "metrics_np_semantic_draws",
    ):
        assert result[name].data_ptr() == state[name].data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("failure", "expected_status"),
    (("duplicate_agent_id", 2), ("gru", 3), ("pcg64", 4)),
)
def test_asexual_plan_invalid_device_state_is_atomic(
    tmp_path: Path,
    failure: str,
    expected_status: int,
) -> None:
    state, _ = _make_state(
        seed=15,
        parents=[
            {"slot": 0, "agent_id": 10, "position": (1, 1), "cooldown": 2},
            {"slot": 1, "agent_id": 20, "position": (3, 3)},
        ],
        free_queue=[5, 6],
        buffered_uint32=True,
    )
    if failure == "duplicate_agent_id":
        state["slot_to_agent_id"][1] = 10
    elif failure == "gru":
        state["policy_use_gru"][0] = True
    else:
        state["pcg64_buffer_words"][0] = 2

    mutable = (
        "repro_cooldown",
        "slot_free_queue_head",
        "pcg64_state_words",
        "pcg64_buffer_words",
        "policy_np_semantic_draws",
        "runner_pcg64_state_words",
        "runner_pcg64_buffer_words",
        "metrics_np_semantic_draws",
    )
    before = {name: state[name].clone() for name in mutable}
    result = execute_asexual_birth_plan_cuda(
        **state,
        parent_energy_share=0.5,
        min_energy_buffer=1.0,
        max_births=4,
        build_directory=tmp_path / "build",
    )
    assert int(result["status"].item()) == expected_status
    assert int(result["planned_count"].item()) == 0
    assert int(result["decision_count"].item()) == 0
    for name in mutable:
        assert torch.equal(state[name], before[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "unsupported_mode",
    (
        "requires_mate",
        "injury_gate_active",
    ),
)
def test_asexual_plan_rejects_hard_unsupported_gates_atomically(
    tmp_path: Path,
    unsupported_mode: str,
) -> None:
    state, _ = _make_state(
        seed=15,
        parents=[{"slot": 0, "agent_id": 10, "position": (2, 2), "cooldown": 2}],
        free_queue=[5],
    )
    before = {
        name: state[name].clone()
        for name in (
            "repro_cooldown",
            "slot_free_queue_head",
            "pcg64_state_words",
            "pcg64_buffer_words",
            "policy_np_semantic_draws",
            "runner_pcg64_state_words",
            "runner_pcg64_buffer_words",
            "metrics_np_semantic_draws",
        )
    }
    result = execute_asexual_birth_plan_cuda(
        **state,
        parent_energy_share=0.5,
        min_energy_buffer=1.0,
        max_births=1,
        build_directory=tmp_path / "build",
        **{unsupported_mode: True},
    )
    assert int(result["status"].item()) == 5
    for name, expected in before.items():
        assert torch.equal(state[name], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "gate_kwargs",
    (
        pytest.param(
            {"injury_repro_k_injury": 0.35},
            id="conception-coefficient-without-gate",
        ),
        pytest.param(
            {"conception_gate_active": True},
            id="conception-gate-without-coefficient",
        ),
        pytest.param(
            {"injury_birth_fail_scar": 0.12},
            id="gestation-coefficient-without-gate",
        ),
        pytest.param(
            {"gestation_gate_active": True},
            id="gestation-gate-without-coefficient",
        ),
    ),
)
def test_asexual_plan_rejects_runner_gate_coefficient_mismatch_atomically(
    tmp_path: Path,
    gate_kwargs: dict[str, float | bool],
) -> None:
    state, _ = _make_state(
        seed=15,
        parents=[{"slot": 0, "agent_id": 10, "position": (2, 2)}],
        free_queue=[5],
    )
    mutable = (
        "repro_cooldown",
        "slot_free_queue_head",
        "pcg64_state_words",
        "pcg64_buffer_words",
        "policy_np_semantic_draws",
        "runner_pcg64_state_words",
        "runner_pcg64_buffer_words",
        "metrics_np_semantic_draws",
    )
    before = {name: state[name].clone() for name in mutable}

    result = execute_asexual_birth_plan_cuda(
        **state,
        parent_energy_share=0.5,
        min_energy_buffer=1.0,
        max_births=1,
        build_directory=tmp_path / "build",
        **gate_kwargs,
    )

    assert int(result["status"].item()) == 5
    assert int(result["planned_count"].item()) == 0
    assert int(result["decision_count"].item()) == 0
    for name, expected in before.items():
        assert torch.equal(state[name], expected)


def test_asexual_plan_capability_receipt_is_bounded() -> None:
    assert ASEXUAL_BIRTH_PLAN_STATUS[0] == "ok"
    assert ASEXUAL_BIRTH_REASON[1] == "birth_fail_no_space"
    assert ASEXUAL_BIRTH_REASON[REASON_CONCEPTION] == "birth_fail_conception"
    assert ASEXUAL_BIRTH_REASON[REASON_GESTATION] == "birth_fail_gestation"
    assert ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT[
        "selected_first_reserved_tile_failure"
    ] is True
    assert ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT["mate_gate"] == "explicitly_rejected"
    assert ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT["injury_gate"] == "explicitly_rejected"
    assert ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT["conception_gate"] == "exact_runner_pcg64"
    assert ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT["gestation_gate"] == "exact_runner_pcg64"
    assert ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT["identity_commit"] is False
    assert ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT["generation_turnover"] is False
    assert ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT["gru"] is False
    assert ASEXUAL_BIRTH_PLAN_CAPABILITY_RECEIPT["promotion_qualified"] is False
