from __future__ import annotations

from pathlib import Path

from simulation.mem_max_pre_cue_checkpoint import (
    deterministic_focal_policy_sample,
    execute_exact_mid_generation_pair,
    execute_mid_generation_branch_continuation,
)
from tests.evolvability.wdl.test_mem_max_23_pre_cue_checkpoint import _make_sim


def _snapshot(tmp_path: Path):
    sim = _make_sim(tmp_path)
    sim.env.patch_world_step(0)
    return sim, sim.mem_max_pre_cue_checkpoint_recorder.snapshots[0]


def test_mem_max_24_selects_memory_capable_gru_focal(tmp_path: Path) -> None:
    sim, _snap = _snapshot(tmp_path)
    sample = deterministic_focal_policy_sample(sim)

    assert sample["gru_memory_positive_agent_ids"] == [1, 2]
    assert sample["mlp_or_memory_zero_agent_ids"] == [3, 4]
    assert sample["agents"][0]["policy_type"] == "GRUPolicy"
    assert sample["agents"][0]["effective_memory_dim"] == 16


def test_mem_max_24_mid_generation_continuation_executes_env_and_consequence(tmp_path: Path) -> None:
    _sim, snap = _snapshot(tmp_path)
    result = execute_mid_generation_branch_continuation(
        snap["snapshot_path"],
        forced_correct_target=1,
        focal_agent_id=2,
        label="test_gru_target_a",
    )

    assert result["focal_policy_type"] == "GRUPolicy"
    assert result["focal_memory_dim"] == 16
    assert result["focal_use_gru"] is True
    assert result["decision_abs_step"] == 3
    assert result["event_end_abs_step"] == 5
    assert len(result["ticks"]) == 6
    assert result["decision"]["phase"] == "decision_active"
    assert result["decision"]["executed_action"] == result["decision"]["chosen_action"]
    assert "new_target_entry" in result["decision"]
    assert result["counter_delta"]["_mem_max_delayed_choice_consequence_applied_count_gen"] >= 1.0


def test_mem_max_24_exact_gru_pair_and_controls(tmp_path: Path) -> None:
    _sim, snap = _snapshot(tmp_path)
    normal = execute_exact_mid_generation_pair(snap["snapshot_path"], focal_agent_id=2, label="normal_gru")
    ablated = execute_exact_mid_generation_pair(
        snap["snapshot_path"],
        focal_agent_id=2,
        memory_ablated=True,
        label="ablated_gru",
    )
    cue_off = execute_exact_mid_generation_pair(
        snap["snapshot_path"],
        focal_agent_id=2,
        cue_enabled=False,
        label="cue_off_gru",
    )
    mlp = execute_exact_mid_generation_pair(snap["snapshot_path"], focal_agent_id=3, label="mlp_control")

    assert normal["hidden_state_separation"] == 1.0
    assert normal["logit_divergence"] == 1.0
    assert ablated["hidden_state_separation"] == 0.0
    assert cue_off["logit_divergence"] == 0.0
    assert mlp["hidden_state_separation"] == 0.0
    assert normal["chosen_action_differs"] is False
