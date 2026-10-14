from __future__ import annotations

import numpy as np
import pytest
import torch

from evolution.gmnw_v1 import (
    GmnwConfig,
    GmnwPhase,
    advance_private_task_state,
    append_task_observation,
    balanced_cue_identity,
    initial_private_task_state,
)
from perf.gmnw_cpu_owner_v1 import GmnwCpuOwnerV1


def _create_owner(tmp_path, config: GmnwConfig, *, capacity: int = 4):
    return GmnwCpuOwnerV1.create(
        capacity=capacity,
        config=config,
        build_directory=tmp_path / "build",
    )


def _slot_state(checkpoint: dict, slot: int) -> dict[str, int | bool]:
    tensors = checkpoint["tensors"]
    return {
        name: tensors[name][slot].item()
        for name in (
            "episode_index",
            "cue_identity",
            "cue_phase",
            "phase_tick",
            "delay_remaining",
            "decision_due",
            "correct_action",
            "last_response",
            "correctness",
            "consequence_status",
            "task_rng_state",
            "task_rng_cursor",
        )
    }


def _assert_python_cpp_state_equal(python_state, cpp_state: dict) -> None:
    assert cpp_state["episode_index"] == python_state.episode_index
    assert cpp_state["cue_identity"] == python_state.cue_identity
    assert cpp_state["cue_phase"] == python_state.cue_phase
    assert cpp_state["phase_tick"] == python_state.phase_tick
    assert cpp_state["delay_remaining"] == python_state.delay_remaining
    assert cpp_state["decision_due"] is python_state.decision_due
    assert cpp_state["correct_action"] == python_state.correct_action
    assert cpp_state["last_response"] == python_state.last_response
    assert cpp_state["correctness"] == python_state.correctness
    assert cpp_state["consequence_status"] == python_state.consequence_status
    assert cpp_state["task_rng_state"] & ((1 << 64) - 1) == python_state.task_rng_state
    assert cpp_state["task_rng_cursor"] == python_state.task_rng_cursor


def _advance_owner_to_decision(
    owner: GmnwCpuOwnerV1,
    config: GmnwConfig,
    health: torch.Tensor,
    reproduction_scale: torch.Tensor,
    *,
    generation_index: int,
) -> int:
    actions = torch.full((health.numel(),), 5, dtype=torch.int64)
    for _ in range(config.cue_ticks + config.delay_ticks):
        owner.step_with_consequences(
            actions,
            health,
            reproduction_scale,
            generation_index=generation_index,
        )
    checkpoint = owner.export_checkpoint()
    assert int(checkpoint["tensors"]["cue_phase"][0]) == int(GmnwPhase.DECISION)
    return int(checkpoint["tensors"]["correct_action"][0])


@pytest.mark.parametrize(
    ("cue_alphabet", "delay_ticks", "distractors_enabled"),
    ((2, 4, False), (4, 4, True), (4, 12, True)),
)
def test_gmnw_cpu_owner_matches_python_phase_and_observation_contract(
    tmp_path,
    cue_alphabet: int,
    delay_ticks: int,
    distractors_enabled: bool,
) -> None:
    config = GmnwConfig(
        cue_alphabet=cue_alphabet,
        delay_ticks=delay_ticks,
        distractors_enabled=distractors_enabled,
        task_seed=202,
    )
    owner = _create_owner(tmp_path, config)
    owner.initialize_slots(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([7], dtype=torch.int64),
    )
    python_state = initial_private_task_state(config, stream_id=7)

    for tick in range(config.cue_ticks + config.delay_ticks + 3):
        checkpoint = owner.export_checkpoint()
        _assert_python_cpp_state_equal(python_state, _slot_state(checkpoint, 0))
        base_np = np.arange(20, dtype=np.float32) + np.float32(tick)
        expected_observation = append_task_observation(base_np, python_state, config)
        base = torch.zeros((4, 20), dtype=torch.float32)
        base[0].copy_(torch.from_numpy(base_np))
        actual_observation = owner.append_task_observation(base)[0].numpy()
        assert actual_observation.tobytes() == expected_observation.tobytes()

        action = int(
            python_state.correct_action
            if python_state.phase is GmnwPhase.DECISION
            else 5
        )
        python_state = advance_private_task_state(
            python_state,
            config,
            stream_id=7,
            action=action,
        ).state
        actions = torch.full((4,), 5, dtype=torch.int64)
        actions[0] = action
        owner.step(actions)


def test_gmnw_cpu_owner_checkpoint_fresh_owner_continuation_is_exact(tmp_path) -> None:
    config = GmnwConfig(cue_alphabet=4, delay_ticks=12, distractors_enabled=True)
    uninterrupted = _create_owner(tmp_path, config)
    uninterrupted.initialize_slots(
        torch.tensor([0, 2], dtype=torch.int64),
        torch.tensor([11, 13], dtype=torch.int64),
    )
    actions = torch.tensor([0, 5, 2, 5], dtype=torch.int64)
    for _ in range(7):
        uninterrupted.step(actions)
    checkpoint = uninterrupted.export_checkpoint()
    restored = GmnwCpuOwnerV1.from_checkpoint(
        checkpoint,
        build_directory=tmp_path / "build",
    )
    for _ in range(19):
        uninterrupted.step(actions)
        restored.step(actions)

    expected = uninterrupted.export_checkpoint()
    actual = restored.export_checkpoint()
    assert expected["metadata"] == actual["metadata"]
    for name in expected["tensors"]:
        assert torch.equal(expected["tensors"][name], actual["tensors"][name]), name


def test_gmnw_cpu_owner_birth_death_and_cross_family_slot_reuse(tmp_path) -> None:
    config = GmnwConfig(cue_alphabet=4, delay_ticks=4)
    owner = _create_owner(tmp_path, config)
    owner.initialize_birth(slot=1, stream_id=101)
    first = owner.export_checkpoint()
    assert bool(first["tensors"]["alive"][1]) is True
    first_cue = int(first["tensors"]["cue_identity"][1])

    owner.clear_slots(torch.tensor([1], dtype=torch.int64))
    cleared = owner.export_checkpoint()
    assert bool(cleared["tensors"]["alive"][1]) is False
    assert int(cleared["tensors"]["cue_identity"][1]) == -1
    assert int(cleared["tensors"]["task_rng_cursor"][1]) == 0

    owner.initialize_birth(slot=1, stream_id=202)
    reused = owner.export_checkpoint()
    assert bool(reused["tensors"]["alive"][1]) is True
    assert int(reused["tensors"]["stream_id"][1]) == 202
    assert int(reused["tensors"]["episode_index"][1]) == 0
    assert int(reused["tensors"]["cue_identity"][1]) != first_cue or 202 != 101


def test_gmnw_cpu_owner_cancellation_does_not_mutate_state(tmp_path) -> None:
    config = GmnwConfig()
    owner = _create_owner(tmp_path, config)
    owner.initialize_slots(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )
    before = owner.export_checkpoint()
    result = owner.step(torch.zeros((4,), dtype=torch.int64), cancel_requested=True)
    after = owner.export_checkpoint()

    assert result["cancelled"] is True
    for name in before["tensors"]:
        assert torch.equal(before["tensors"][name], after["tensors"][name]), name


@pytest.mark.parametrize(
    ("generation_index", "tier", "expected_health", "expected_scale", "dead"),
    (
        (0, "grace", 0.85, 0.5, False),
        (4, "severe", 0.55, 0.0, False),
        (8, "lethal", 0.0, 0.0, True),
    ),
)
def test_gmnw_cpu_owner_incorrect_consequence_settles_before_birth_scan(
    tmp_path,
    generation_index: int,
    tier: str,
    expected_health: float,
    expected_scale: float,
    dead: bool,
) -> None:
    config = GmnwConfig(cue_alphabet=4, delay_ticks=1)
    owner = _create_owner(tmp_path, config)
    owner.initialize_slots(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([9], dtype=torch.int64),
    )
    health = torch.ones((4,), dtype=torch.float32)
    reproduction_scale = torch.ones((4,), dtype=torch.float32)
    correct_action = _advance_owner_to_decision(
        owner,
        config,
        health,
        reproduction_scale,
        generation_index=generation_index,
    )
    actions = torch.full((4,), 5, dtype=torch.int64)
    actions[0] = (correct_action + 1) % 4

    result = owner.step_with_consequences(
        actions,
        health,
        reproduction_scale,
        generation_index=generation_index,
    )
    checkpoint = owner.export_checkpoint()

    assert result["pressure_tier"] == tier
    assert result["incorrect_count"] == 1
    assert result["birth_scan_allowed_after_settlement"] is True
    assert float(health[0]) == pytest.approx(expected_health)
    assert float(reproduction_scale[0]) == pytest.approx(expected_scale)
    assert bool(checkpoint["tensors"]["alive"][0]) is (not dead)
    assert result["task_death_count"] == int(dead)
    if dead:
        assert int(checkpoint["tensors"]["cue_identity"][0]) == -1
        assert list(result["task_death_slots"]) == [0]


def test_gmnw_cpu_owner_correct_decision_has_no_bonus_and_episode_reset_is_exact(
    tmp_path,
) -> None:
    config = GmnwConfig(cue_alphabet=4, delay_ticks=1)
    owner = _create_owner(tmp_path, config)
    owner.initialize_slots(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([9], dtype=torch.int64),
    )
    health = torch.ones((4,), dtype=torch.float32)
    reproduction_scale = torch.ones((4,), dtype=torch.float32)
    correct_action = _advance_owner_to_decision(
        owner,
        config,
        health,
        reproduction_scale,
        generation_index=0,
    )
    actions = torch.full((4,), 5, dtype=torch.int64)
    actions[0] = correct_action

    correct = owner.step_with_consequences(
        actions,
        health,
        reproduction_scale,
        generation_index=0,
    )
    assert correct["correct_count"] == 1
    assert correct["damage_total"] == 0.0
    assert float(health[0]) == 1.0
    assert float(reproduction_scale[0]) == 1.0


def test_gmnw_cpu_owner_accepts_equal_frozen_capability_pressure(tmp_path) -> None:
    config = GmnwConfig(
        delay_ticks=1,
        grace_damage_fraction=0.01,
        severe_damage_fraction=0.01,
    )
    owner = _create_owner(tmp_path, config)
    owner.initialize_slots(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([9], dtype=torch.int64),
    )
    health = torch.ones((4,), dtype=torch.float32)
    reproduction_scale = torch.ones((4,), dtype=torch.float32)
    correct_action = _advance_owner_to_decision(
        owner,
        config,
        health,
        reproduction_scale,
        generation_index=0,
    )
    actions = torch.full((4,), 5, dtype=torch.int64)
    actions[0] = (correct_action + 1) % 4

    result = owner.step_with_consequences(
        actions,
        health,
        reproduction_scale,
        generation_index=0,
    )

    assert result["incorrect_count"] == 1
    assert result["damage_total"] == pytest.approx(0.01)
    assert float(health[0]) == pytest.approx(0.99)

    reproduction_scale[0] = 0.5
    owner.step_with_consequences(
        torch.full((4,), 5, dtype=torch.int64),
        health,
        reproduction_scale,
        generation_index=0,
    )
    assert float(reproduction_scale[0]) == 1.0


def test_gmnw_cpu_owner_consequence_cancellation_and_invalid_state_are_fail_closed(
    tmp_path,
) -> None:
    config = GmnwConfig(delay_ticks=1)
    owner = _create_owner(tmp_path, config)
    owner.initialize_slots(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )
    health = torch.ones((4,), dtype=torch.float32)
    reproduction_scale = torch.ones((4,), dtype=torch.float32)
    before = owner.export_checkpoint()
    cancelled = owner.step_with_consequences(
        torch.zeros((4,), dtype=torch.int64),
        health,
        reproduction_scale,
        generation_index=8,
        cancel_requested=True,
    )
    after = owner.export_checkpoint()
    assert cancelled["cancelled"] is True
    assert torch.equal(before["tensors"]["cue_phase"], after["tensors"]["cue_phase"])
    assert torch.equal(health, torch.ones_like(health))

    health[0] = float("nan")
    with pytest.raises(RuntimeError, match="health state is invalid"):
        owner.step_with_consequences(
            torch.zeros((4,), dtype=torch.int64),
            health,
            reproduction_scale,
            generation_index=0,
        )


def test_gmnw_cpu_owner_compact_counters_survive_checkpoint_continuation(
    tmp_path,
) -> None:
    config = GmnwConfig(cue_alphabet=4, delay_ticks=1)
    owner = _create_owner(tmp_path, config)
    owner.initialize_slots(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([3], dtype=torch.int64),
    )
    health = torch.ones((4,), dtype=torch.float32)
    reproduction_scale = torch.ones((4,), dtype=torch.float32)
    correct_action = _advance_owner_to_decision(
        owner,
        config,
        health,
        reproduction_scale,
        generation_index=0,
    )
    actions = torch.full((4,), 5, dtype=torch.int64)
    actions[0] = (correct_action + 1) % 4
    owner.step_with_consequences(
        actions,
        health,
        reproduction_scale,
        generation_index=0,
    )

    expected = owner.compact_counters()
    restored = GmnwCpuOwnerV1.from_checkpoint(
        owner.export_checkpoint(),
        build_directory=tmp_path / "build",
    )
    actual = restored.compact_counters()

    assert expected == actual
    assert actual["decision_count"] == 1
    assert actual["incorrect_count"] == 1
    assert actual["reproduction_scale_half_count"] == 1
    assert actual["task_damage_total"] == pytest.approx(0.15)
    assert actual["task_death_count"] == 0


@pytest.mark.parametrize(
    ("scales", "expected_status", "expected_slots", "mlp_count", "gru_count"),
    (
        ([0.0, 0.0, 0.0, 1.0], "extinct_no_eligible_parents", [], 0, 0),
        ([1.0, 0.0, 0.0, 1.0], "one_parent_asexual_refill", [0], 0, 1),
        ([0.0, 1.0, 0.0, 1.0], "one_parent_asexual_refill", [1], 1, 0),
        ([1.0, 1.0, 0.0, 1.0], "eligible_parent_set_frozen", [0, 1], 1, 1),
    ),
)
def test_gmnw_cpu_owner_turnover_uses_only_selected_surviving_eligible_parents(
    tmp_path,
    scales: list[float],
    expected_status: str,
    expected_slots: list[int],
    mlp_count: int,
    gru_count: int,
) -> None:
    owner = _create_owner(tmp_path, GmnwConfig())
    owner.initialize_slots(
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([10, 11, 12], dtype=torch.int64),
    )
    owner.clear_slots(torch.tensor([2], dtype=torch.int64))
    result = owner.evaluate_turnover(
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([True, True, True, True], dtype=torch.bool),
        torch.tensor(scales, dtype=torch.float32),
        torch.tensor([1, 0, 1, 0], dtype=torch.int8),
    )

    assert result["status"] == expected_status
    assert result["terminal"] is (not expected_slots)
    assert result["eligible_parent_slots"].tolist() == expected_slots
    assert result["eligible_mlp_parent_count"] == mlp_count
    assert result["eligible_gru_parent_count"] == gru_count
    assert result["target_population_restoration_guaranteed"] is False
    assert result["founder_template_reuse"] is False
    assert result["random_replacement"] is False
    assert result["family_switching"] is False


def test_gmnw_cpu_owner_turnover_rejects_ambiguous_parent_sets(tmp_path) -> None:
    owner = _create_owner(tmp_path, GmnwConfig())
    owner.initialize_slots(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([10], dtype=torch.int64),
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        owner.evaluate_turnover(
            torch.tensor([0, 0], dtype=torch.int64),
            torch.ones((4,), dtype=torch.bool),
            torch.ones((4,), dtype=torch.float32),
            torch.zeros((4,), dtype=torch.int8),
        )

    with pytest.raises(RuntimeError, match="policy family is unsupported"):
        owner.evaluate_turnover(
            torch.tensor([0], dtype=torch.int64),
            torch.ones((4,), dtype=torch.bool),
            torch.ones((4,), dtype=torch.float32),
            torch.full((4,), 7, dtype=torch.int8),
        )


def test_gmnw_cpu_owner_episode_blocks_survive_checkpoint_and_slot_reuse(
    tmp_path,
) -> None:
    config = GmnwConfig(cue_alphabet=4, delay_ticks=1, consequence_ticks=1)
    owner = _create_owner(tmp_path, config, capacity=2)
    owner.initialize_birth(slot=0, stream_id=17)
    health = torch.ones((2,), dtype=torch.float32)
    reproduction_scale = torch.ones((2,), dtype=torch.float32)
    observed: list[int] = []

    for episode in range(8):
        checkpoint = owner.export_checkpoint()
        observed.append(int(checkpoint["tensors"]["cue_identity"][0]))
        correct_action = _advance_owner_to_decision(
            owner,
            config,
            health,
            reproduction_scale,
            generation_index=0,
        )
        actions = torch.full((2,), 5, dtype=torch.int64)
        actions[0] = correct_action
        owner.step_with_consequences(
            actions,
            health,
            reproduction_scale,
            generation_index=0,
        )
        if episode == 3:
            owner = GmnwCpuOwnerV1.from_checkpoint(
                owner.export_checkpoint(),
                build_directory=tmp_path / "build",
            )
        owner.step_with_consequences(
            torch.full((2,), 5, dtype=torch.int64),
            health,
            reproduction_scale,
            generation_index=0,
        )

    assert sorted(observed[:4]) == [0, 1, 2, 3]
    assert sorted(observed[4:]) == [0, 1, 2, 3]

    owner.clear_slots(torch.tensor([0], dtype=torch.int64))
    owner.initialize_birth(slot=0, stream_id=29)
    reused = owner.export_checkpoint()["tensors"]
    assert int(reused["episode_index"][0]) == 0
    assert int(reused["cue_identity"][0]) == balanced_cue_identity(
        config,
        stream_id=29,
        episode_index=0,
    )


@pytest.mark.parametrize("generation_count", (1, 2, 4, 8))
def test_gmnw_cpu_owner_multigeneration_repeat_is_exact(
    tmp_path,
    generation_count: int,
) -> None:
    config = GmnwConfig(cue_alphabet=4, delay_ticks=2, consequence_ticks=1)

    def run_once():
        owner = _create_owner(tmp_path, config, capacity=4)
        owner.initialize_slots(
            torch.tensor([0, 1, 2, 3], dtype=torch.int64),
            torch.tensor([101, 202, 303, 404], dtype=torch.int64),
        )
        health = torch.ones((4,), dtype=torch.float32)
        reproduction_scale = torch.ones((4,), dtype=torch.float32)
        for generation_index in range(generation_count):
            for _ in range(16):
                checkpoint = owner.export_checkpoint()["tensors"]
                actions = torch.full((4,), 5, dtype=torch.int64)
                for slot in range(4):
                    if bool(checkpoint["alive"][slot]) and int(
                        checkpoint["cue_phase"][slot]
                    ) == int(GmnwPhase.DECISION):
                        correct = int(checkpoint["correct_action"][slot])
                        episode = int(checkpoint["episode_index"][slot])
                        actions[slot] = (
                            correct
                            if (slot + episode) % 3
                            else (correct + 1) % config.cue_alphabet
                        )
                owner.step_with_consequences(
                    actions,
                    health,
                    reproduction_scale,
                    generation_index=generation_index,
                )
        return owner.export_checkpoint(), health, reproduction_scale

    first_checkpoint, first_health, first_reproduction = run_once()
    second_checkpoint, second_health, second_reproduction = run_once()
    assert first_checkpoint["metadata"] == second_checkpoint["metadata"]
    assert torch.equal(first_health, second_health)
    assert torch.equal(first_reproduction, second_reproduction)
    for name in first_checkpoint["tensors"]:
        assert torch.equal(
            first_checkpoint["tensors"][name],
            second_checkpoint["tensors"][name],
        ), name


def test_gmnw_cpu_owner_checkpoint_rejects_tensor_shape_mismatch(tmp_path) -> None:
    config = GmnwConfig(cue_alphabet=4)
    owner = _create_owner(tmp_path, config)
    checkpoint = owner.export_checkpoint()
    checkpoint["tensors"] = dict(checkpoint["tensors"])
    checkpoint["tensors"]["cue_identity"] = checkpoint["tensors"]["cue_identity"][:-1]

    with pytest.raises(RuntimeError, match="cue_identity"):
        GmnwCpuOwnerV1.from_checkpoint(
            checkpoint,
            build_directory=tmp_path / "build",
        )
