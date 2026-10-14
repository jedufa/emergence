from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from perf.cxxwave1g_full_live_backbone import (
    RESIDENT_STATE_SCHEMA_VERSION,
    capability_payload,
    capture_resident_runtime_state_v3,
    cpp_backbone_capability_set_v1,
    execute_generation,
    first_divergence_proof,
    resident_runtime_state_payload,
)
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runner_pkg.runner_ticks_loop import TickLoopMixin


def test_cxxwave1g_flags_default_off() -> None:
    cfg = SimConfig()

    assert cfg.cxxwave1g_full_cpp_backbone_enable is False
    assert cfg.cxxwave1g_full_cpp_backbone_fail_closed is True
    assert cfg.cxxwave1g_full_cpp_backbone_write_summary is True


def test_cpp_backbone_capability_set_fails_closed_until_all_required_groups_native() -> None:
    capability = cpp_backbone_capability_set_v1(policy_family="MLPPolicy+GRUPolicy", patch_features=("patch_world",))
    payload = capability_payload(capability)

    assert payload["schema_version"] == "cpp_backbone_capability_set_v1"
    assert payload["supported"] is False
    assert "missing_native_authority:policy_row_mappings" in payload["blockers"]
    assert "full_resident_checkpoint_not_native" in payload["blockers"]
    assert "lifecycle_reproduction_mutation_python_owned" in payload["blockers"]
    assert payload["major_block_decisions"]["policy_hmem"]["decision"] == "CXXWAVE1G-POLICY-C/PARITY-BLOCKED"
    assert payload["major_block_decisions"]["patch_wdl_environment"]["decision"] == "CXXWAVE1G-ENV-D/PARITY-BLOCKED"


def test_resident_runtime_state_v3_capture_and_execute_generation_fail_closed() -> None:
    runner = _fake_runner()
    state = capture_resident_runtime_state_v3(runner, generation=3, tick=0, cell_id="patch_seed101")
    payload = resident_runtime_state_payload(state)
    result = execute_generation(state, {"envelope": "patch_seed101"}, generation_index=3, steps_per_generation=80)
    proof = first_divergence_proof(result)

    assert payload["schema_version"] == RESIDENT_STATE_SCHEMA_VERSION
    assert payload["cell_id"] == "patch_seed101"
    assert len(payload["field_authority"]) >= 19
    assert payload["capability"]["supported"] is False
    assert result["executed"] is False
    assert result["hidden_fallback"] is False
    assert result["state_digest_before"] == state.state_digest
    assert result["state_digest_after"] == state.state_digest
    assert result["major_block_decisions"]["resident_checkpoint"]["decision"] == "CXXWAVE1G-CKPT-D/PARITY-BLOCKED"
    assert proof["kind"] == "pre_execution_support_blocker"
    assert proof["science_state_mutated"] is False


def test_execute_generation_with_all_capabilities_reaches_backend_missing_without_fallback() -> None:
    native_authority = {
        group: True
        for group in (
            "execution_cell_metadata",
            "population_slots_stable_ids",
            "mlp_gru_policy_banks",
            "policy_row_mappings",
            "hmem",
            "action_cache",
            "observations_staging",
            "drives",
            "chemistry_maintenance",
            "patch_world_state",
            "jungle_state",
            "wdl_state",
            "lifecycle_reproduction_mutation",
            "lineage_founder",
            "typed_event_tape",
            "checkpoint_cursors",
            "rng_state_indices",
            "capability_support_mask",
            "error_fallback_status",
        )
    }
    capability = cpp_backbone_capability_set_v1(
        policy_family="MLPPolicy",
        lifecycle_churn="native_authoritative",
        mutation_type="native_authoritative",
        checkpoint_schema="full_resident_checkpoint_native_v1",
        output_mode="native_public_compatible",
        multi_cell_compatibility="single_cell_only",
        native_authority=native_authority,
    )
    state = capture_resident_runtime_state_v3(_fake_runner(), generation=0, capability=capability)
    result = execute_generation(state, {}, generation_index=0, steps_per_generation=4)

    assert capability_payload(capability)["supported"] is True
    assert result["executed"] is False
    assert result["hidden_fallback"] is False
    assert result["blockers"] == ["cpp_backbone_execute_generation_backend_missing"]
    assert result["major_block_decisions"]["policy_hmem"]["supported"] is True


def test_live_tick_loop_gate_fails_closed_before_python_loop(tmp_path: Path) -> None:
    runner = _GateRunner()
    fake = _fake_runner()
    runner.pop = fake.pop
    runner.env = fake.env
    runner.env_cfg = fake.env_cfg
    runner.np_rng = fake.np_rng
    runner.run_dir = str(tmp_path)
    runner.sim_cfg = SimConfig(generations=1, steps_per_generation=4)
    runner.sim_cfg.cxxwave1g_full_cpp_backbone_enable = True
    runner.sim_cfg.cxxwave1g_full_cpp_backbone_fail_closed = True

    with pytest.raises(RuntimeError, match="CXXWAVE1G full C\\+\\+ backbone unsupported"):
        runner._run_ticks(
            gen=0,
            buckets={},
            injury_recover_mult=1.0,
            deaths_this_gen=0,
            successful_births=0,
            failed_births=0,
            gen_peak_injury=0.0,
            rest_streaks={},
        )

    summary_path = tmp_path / "metrics" / "cxxwave1g_full_live_cpp_backbone_summary.json"
    assert summary_path.exists()
    assert "missing_native_authority:policy_row_mappings" in summary_path.read_text(encoding="utf-8")


class _GateRunner(TickLoopMixin):
    pass


def _fake_runner() -> SimpleNamespace:
    agents = [
        SimpleNamespace(agent_id=2, energy=7.5, health=9.0, injury=0.1, age=4, policy=SimpleNamespace()),
        SimpleNamespace(agent_id=5, energy=6.0, health=8.0, injury=0.0, age=2, policy=SimpleNamespace()),
    ]
    pop = SimpleNamespace(alive_agents=lambda: list(agents))
    env = SimpleNamespace(
        agent_alive=np.asarray([0, 0, 1, 0, 0, 1], dtype=np.uint8),
        agent_pos_y=np.asarray([0, 0, 1, 0, 0, 2], dtype=np.int32),
        agent_pos_x=np.asarray([0, 0, 3, 0, 0, 4], dtype=np.int32),
        food=np.zeros((6, 6), dtype=np.int8),
        patch_food_amount=np.ones((6, 6), dtype=np.float32),
        patch_cover=np.zeros((6, 6), dtype=np.float32),
        patch_shelter=np.zeros((6, 6), dtype=np.float32),
    )
    cfg = SimpleNamespace(
        steps_per_generation=80,
        policy_deterministic_actions=True,
        policy_tick_stride=2,
        cxxwave1f_scalar_backbone_enable=True,
    )
    env_cfg = SimpleNamespace(
        patch_world_enable=True,
        patch_world_observation_enable=True,
        wdl_delayed_hazard_enable=False,
    )
    return SimpleNamespace(pop=pop, env=env, sim_cfg=cfg, env_cfg=env_cfg, np_rng=np.random.default_rng(123))
