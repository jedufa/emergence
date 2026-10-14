from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from agents.cognition import (
    MEMORY_LADDER,
    compute_cognitive_budget_ledger,
    effective_memory_dim_for_genome,
    express_cognitive_phenotype,
    get_cognitive_trait,
)
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from env.env_config import EnvConfig
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runner_pkg.runner_policy import PolicyMixin


def _genome(*, memory_dim: int, use_gru: bool | None = None) -> Genome:
    md = int(memory_dim)
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=bool(md > 0) if use_gru is None else bool(use_gru),
        memory_dim=md,
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )


def _agent(agent_id: int, genome: Genome) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=int(agent_id),
        genome=genome,
        energy=100.0,
        health=10.0,
        alive=True,
        drives=np.zeros((4,), dtype=np.float32),
        stats=SimpleNamespace(
            survival_steps=10,
            pred_err_count=0,
            pred_err_sum=0.0,
            offspring_count=0,
            patch_habitat_lineage_id=int(agent_id),
        ),
    )


class _TelemetryProbe(PolicyMixin):
    pass


@dataclass(frozen=True)
class _FoodRegrowthReturnFixture:
    width: int = 5
    height: int = 5
    regrowth_tick: int = 3
    food_pos: tuple[int, int] = (3, 2)

    def metadata(self) -> dict[str, object]:
        return {
            "memory_pressure_scenario": True,
            "memory_pressure_type": "food_regrowth_return",
            "direct_memory_reward": False,
        }

    def food_at(self, tick: int) -> tuple[int, int] | None:
        return self.food_pos if int(tick) >= int(self.regrowth_tick) else None

    def observations(self, *, ablated: bool) -> list[np.ndarray]:
        shape = (1, 27)
        rows = []
        for tick in range(5):
            obs = np.zeros(shape, dtype=np.float32)
            pos = self.food_at(tick)
            if pos is not None:
                obs[0, 12] = 1.0
            if not ablated and tick > 0:
                obs[0, 26] = 1.0
            rows.append(obs)
        return rows


def test_stage2_memory_traits_are_registered_honestly() -> None:
    for trait_id in (
        "memory_enabled",
        "memory_dim",
        "effective_memory_dim",
        "memory_capacity",
        "memory_cost",
        "policy_memory_enabled",
        "recurrent_state_enabled",
        "memory_ladder_index",
        "memory_ablation_state",
        "memory_use_signal",
        "memory_pressure_test_ready",
    ):
        assert get_cognitive_trait(trait_id).category == "memory" or trait_id == "memory_cost"

    assert get_cognitive_trait("memory_dim").directly_mutable is True
    assert get_cognitive_trait("effective_memory_dim").directly_mutable is False
    assert get_cognitive_trait("recurrent_state_enabled").directly_mutable is False
    assert get_cognitive_trait("memory_use_signal").status == "partially_active"
    assert get_cognitive_trait("policy_hidden_size").status == "fixed"


def test_stage2_memory_phenotype_ablation_overrides_without_mutating_genome() -> None:
    env_cfg = EnvConfig(width=8, height=8)
    sim_cfg = SimConfig(seed=1)
    g = _genome(memory_dim=64)

    normal = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert normal.memory_enabled is True
    assert normal.effective_memory_dim == 64
    assert normal.memory_capacity_fraction == 1.0

    sim_cfg.force_memory_disabled = True
    disabled = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert disabled.memory_enabled is False
    assert disabled.policy_memory_enabled is False
    assert disabled.recurrent_state_enabled is True
    assert disabled.effective_memory_dim == 0
    assert disabled.force_memory_disabled_active is True
    assert g.memory_dim == 64
    assert effective_memory_dim_for_genome(g, sim_cfg=sim_cfg) == 0

    sim_cfg.force_memory_disabled = False
    sim_cfg.force_min_memory_dim = True
    assert express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg).effective_memory_dim == 0
    sim_cfg.force_min_memory_dim = False
    sim_cfg.force_max_memory_dim = True
    assert express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg).effective_memory_dim == max(MEMORY_LADDER)
    sim_cfg.force_max_memory_dim = False
    sim_cfg.force_default_memory_dim = True
    assert express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg).effective_memory_dim == 0


def test_stage2_memory_budget_reports_existing_cost_without_double_charge() -> None:
    env_cfg = EnvConfig(width=8, height=8)
    sim_cfg = SimConfig(seed=2)
    low = express_cognitive_phenotype(_genome(memory_dim=0, use_gru=False), env_cfg=env_cfg, sim_cfg=sim_cfg)
    high = express_cognitive_phenotype(_genome(memory_dim=64), env_cfg=env_cfg, sim_cfg=sim_cfg)
    low_ledger = compute_cognitive_budget_ledger(low, env_cfg=env_cfg, sim_cfg=sim_cfg)
    high_ledger = compute_cognitive_budget_ledger(high, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert high_ledger.memory_upkeep_cost_hypothetical > low_ledger.memory_upkeep_cost_hypothetical
    assert high_ledger.memory_upkeep_cost_charged == high_ledger.memory_upkeep_cost_hypothetical
    assert high_ledger.charged_energy_delta == 0.0

    sim_cfg.enable_experimental_memory_cost_charging = True
    charged = compute_cognitive_budget_ledger(high, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert charged.memory_cost_charging_enabled is True
    assert charged.memory_upkeep_cost_charged == high_ledger.memory_upkeep_cost_charged

    sim_cfg.force_memory_disabled = True
    disabled = express_cognitive_phenotype(_genome(memory_dim=64), env_cfg=env_cfg, sim_cfg=sim_cfg)
    disabled_ledger = compute_cognitive_budget_ledger(disabled, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert disabled_ledger.memory_upkeep_cost_charged == 0.0


def test_stage2_memory_runtime_telemetry_is_deterministic_and_shape_preserving() -> None:
    probe_a = _TelemetryProbe()
    probe_b = _TelemetryProbe()
    probe_a._reset_memory_runtime_telemetry()
    probe_b._reset_memory_runtime_telemetry()
    h_prev = torch.zeros((2, 64), dtype=torch.float32)
    h_next = torch.zeros((2, 64), dtype=torch.float32)
    h_next[0, :8] = 0.5
    h_next[1, :16] = -0.25
    idx = torch.arange(64, dtype=torch.long).unsqueeze(0)
    dims = torch.tensor([8, 16], dtype=torch.long)
    mask = (idx < dims.unsqueeze(1)).to(torch.float32)

    probe_a._record_memory_runtime_telemetry(h_prev=h_prev, h_next=h_next * mask, mask=mask, reset_count=2)
    probe_b._record_memory_runtime_telemetry(h_prev=h_prev, h_next=h_next * mask, mask=mask, reset_count=2)

    assert tuple(h_prev.shape) == (2, 64)
    assert probe_a._memory_runtime_telemetry_summary() == probe_b._memory_runtime_telemetry_summary()
    summary = probe_a._memory_runtime_telemetry_summary()
    assert summary["memory_use_measurement_available_flag"] == 1.0
    assert summary["memory_mask_leakage_mean"] == 0.0
    assert summary["memory_reset_count_total"] == 2.0


def test_stage2_memory_metrics_columns_diversity_and_pressure_flags() -> None:
    env_cfg = EnvConfig(width=8, height=8)
    sim_cfg = SimConfig(seed=3)
    sim_cfg.memory_pressure_scenario = True
    sim_cfg.memory_pressure_type = "food_regrowth_return"
    sim_cfg.reset_memory_each_tick = True
    agents = [_agent(1, _genome(memory_dim=0, use_gru=False)), _agent(2, _genome(memory_dim=64))]
    telemetry = {
        "memory_state_abs_mean": 0.2,
        "memory_state_abs_p90": 0.5,
        "memory_state_delta_mean": 0.1,
        "memory_state_delta_p90": 0.3,
        "memory_active_slice_abs_mean": 0.4,
        "memory_inactive_slice_abs_mean": 0.0,
        "memory_mask_leakage_mean": 0.0,
        "memory_nonzero_fraction_mean": 0.25,
        "memory_effective_utilization_mean": 0.5,
        "memory_saturation_proxy_mean": 0.0,
        "memory_reset_count_total": 2.0,
        "memory_use_measurement_available_flag": 1.0,
    }
    a = compute_cognitive_status_metrics(
        agents,
        selected_parent_agents=agents[:1],
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        memory_runtime_telemetry=telemetry,
    )
    b = compute_cognitive_status_metrics(
        agents,
        selected_parent_agents=agents[:1],
        env_cfg=env_cfg,
        sim_cfg=sim_cfg,
        memory_runtime_telemetry=telemetry,
    )

    assert a == b
    assert a["memory_dim_min"] == 0.0
    assert a["memory_dim_max"] == 64.0
    assert a["memory_dim_diversity"] == 2.0
    assert a["memory_strategy_count_approx"] >= 2.0
    assert a["memory_reset_each_tick_active_flag"] == 1.0
    assert a["memory_state_abs_mean"] == 0.2
    assert a["memory_pressure_scenario_flag"] == 1.0
    assert a["memory_pressure_food_regrowth_return_flag"] == 1.0
    assert "memory_perception_strategy_entropy" in a
    assert "memory_enabled_without_cost_fraction" in a


def test_stage2_food_regrowth_memory_pressure_fixture_is_deterministic_and_reward_free() -> None:
    fixture = _FoodRegrowthReturnFixture()
    meta = fixture.metadata()
    assert meta["memory_pressure_scenario"] is True
    assert meta["memory_pressure_type"] == "food_regrowth_return"
    assert meta["direct_memory_reward"] is False

    normal_a = fixture.observations(ablated=False)
    normal_b = fixture.observations(ablated=False)
    ablated = fixture.observations(ablated=True)

    assert [x.tolist() for x in normal_a] == [x.tolist() for x in normal_b]
    assert [tuple(x.shape) for x in normal_a] == [tuple(x.shape) for x in ablated]
    assert fixture.food_at(2) is None
    assert fixture.food_at(3) == fixture.food_pos

    root = Path(__file__).resolve().parents[3]
    selection_text = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (
            root / "engine" / "simulation" / "runner_pkg" / "runner_selection.py",
            root / "engine" / "evolution" / "selection.py",
        )
    )
    assert "memory_reward" not in selection_text
    assert "direct_memory_reward" not in selection_text
