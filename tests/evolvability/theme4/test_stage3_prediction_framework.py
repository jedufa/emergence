from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents.cognition import (
    compute_cognitive_budget_ledger,
    direct_prediction_selection_pressure_present,
    effective_prediction_learning_rate_for_genome,
    express_cognitive_phenotype,
    get_cognitive_trait,
)
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from env.env_config import EnvConfig
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runner_pkg.runner_selection import SelectionMixin
from simulation.runner_pkg.runner_ticks_drives import TickDrivesMixin


def _genome(*, pred_lr: float, memory_dim: int = 0) -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=30.0,
        mutation_rate=0.03,
        action_temperature=1.0,
        use_gru=bool(memory_dim > 0),
        memory_dim=int(memory_dim),
        drive_inertia=0.85,
        drive_noise=0.0,
        drive_logit_gain=1.1,
        pred_lr=float(pred_lr),
        developmental_program=canonicalize_developmental_program(None),
    )


def _stats(**overrides: object) -> SimpleNamespace:
    base = dict(
        survival_steps=10,
        energy_collected=0.0,
        offspring_count=0,
        attacks=0,
        pred_err_count=2,
        pred_err_sum=4.0,
        episode0_pred_err_count=2,
        episode0_pred_err_sum=4.0,
        late_pred_err_count=1,
        late_pred_err_sum=2.0,
        offspring=0,
        patch_habitat_lineage_id=0,
        exp_dE=np.asarray([0.0, 0.2, -0.1, 0.0], dtype=np.float32),
        exp_dInj=np.asarray([0.0, 0.0, 0.3, 0.0], dtype=np.float32),
        pred_queue=[],
        prediction_update_count=0,
        prediction_state_delta_abs_sum=0.0,
        prediction_state_delta_abs_max=0.0,
        prediction_reset_count=0,
        last_energy_delta=0.0,
        last_health_delta=0.0,
        last_injury_delta=0.0,
        first_repro_step=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _agent(agent_id: int, genome: Genome, *, stats: SimpleNamespace | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=int(agent_id),
        genome=genome,
        energy=100.0,
        health=10.0,
        injury=0.0,
        alive=True,
        slot=0,
        slot_token=1,
        drives=np.zeros((4,), dtype=np.float32),
        stats=stats if stats is not None else _stats(patch_habitat_lineage_id=int(agent_id)),
    )


class _SelectionProbe(SelectionMixin):
    def __init__(self, sim_cfg: SimConfig) -> None:
        self.sim_cfg = sim_cfg


class _DriveProbe(TickDrivesMixin):
    def __init__(self, sim_cfg: SimConfig, env_cfg: EnvConfig) -> None:
        self.sim_cfg = sim_cfg
        self.env_cfg = env_cfg

    def _ensure_drive_arrays(self, required_slots: int = 1) -> None:
        n = max(1, int(required_slots))
        self._drive_hunger = np.zeros((n,), dtype=np.float32)
        self._drive_pain = np.zeros((n,), dtype=np.float32)
        self._drive_fatigue = np.zeros((n,), dtype=np.float32)
        self._drive_fear = np.zeros((n,), dtype=np.float32)


@dataclass(frozen=True)
class _DelayedConsequenceFixture:
    delay: int = 2
    injury_tick: int = 3

    def metadata(self) -> dict[str, object]:
        return {
            "prediction_pressure_scenario": True,
            "prediction_pressure_type": "delayed_consequence",
            "direct_prediction_reward": False,
        }

    def observations(self, *, ablated: bool) -> list[np.ndarray]:
        rows: list[np.ndarray] = []
        for tick in range(5):
            obs = np.zeros((1, 27), dtype=np.float32)
            obs[0, 3] = 1.0 if tick >= self.injury_tick else 0.0
            if not ablated and tick >= self.delay:
                obs[0, 26] = float(tick - self.delay + 1)
            rows.append(obs)
        return rows


def test_stage3_prediction_traits_registered_honestly() -> None:
    expected = (
        "prediction_enabled",
        "prediction_learning_rate",
        "prediction_update_enabled",
        "prediction_error_tracking",
        "prediction_error_mean",
        "prediction_capacity",
        "prediction_cost",
        "prediction_ablation_state",
        "prediction_state_available",
        "prediction_pressure_test_ready",
    )
    for trait_id in expected:
        assert get_cognitive_trait(trait_id).category in {"prediction", "cognition_budget"}

    assert get_cognitive_trait("prediction_learning_rate").directly_mutable is True
    assert get_cognitive_trait("prediction_enabled").directly_mutable is False
    assert get_cognitive_trait("prediction_capacity").status == "partially_active"
    assert get_cognitive_trait("prediction_pressure_test_ready").status == "partially_active"
    assert get_cognitive_trait("prediction_cost").costed is True


def test_stage3_prediction_phenotype_ablation_overrides_without_mutating_genome() -> None:
    env_cfg = EnvConfig(width=8, height=8)
    sim_cfg = SimConfig(seed=1)
    g = _genome(pred_lr=0.25)

    normal = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert normal.prediction_enabled is True
    assert normal.effective_prediction_learning_rate == 0.25
    assert normal.prediction_state_available is True

    zero = express_cognitive_phenotype(_genome(pred_lr=0.0), env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert zero.prediction_enabled is False
    assert zero.effective_prediction_learning_rate == 0.0

    sim_cfg.force_prediction_disabled = True
    disabled = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert disabled.prediction_enabled is False
    assert disabled.prediction_state_available is False
    assert disabled.effective_prediction_learning_rate == 0.0
    assert g.pred_lr == 0.25

    sim_cfg.force_prediction_disabled = False
    sim_cfg.force_prediction_frozen = True
    frozen = express_cognitive_phenotype(g, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert frozen.prediction_state_available is True
    assert frozen.prediction_update_enabled is False
    assert frozen.effective_prediction_learning_rate == 0.0

    sim_cfg.force_prediction_frozen = False
    sim_cfg.force_prediction_default = True
    assert effective_prediction_learning_rate_for_genome(g, sim_cfg=sim_cfg) == 0.08


def test_stage3_prediction_budget_is_telemetry_only_by_default_and_charged_when_enabled() -> None:
    env_cfg = EnvConfig(width=8, height=8)
    sim_cfg = SimConfig(seed=2)
    low = express_cognitive_phenotype(_genome(pred_lr=0.0), env_cfg=env_cfg, sim_cfg=sim_cfg)
    high = express_cognitive_phenotype(_genome(pred_lr=0.5), env_cfg=env_cfg, sim_cfg=sim_cfg)
    low_ledger = compute_cognitive_budget_ledger(low, env_cfg=env_cfg, sim_cfg=sim_cfg)
    high_ledger = compute_cognitive_budget_ledger(high, env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert high_ledger.prediction_update_cost_hypothetical > low_ledger.prediction_update_cost_hypothetical
    assert high_ledger.prediction_update_cost_charged == 0.0
    assert high_ledger.charged_energy_delta == 0.0
    assert high_ledger.prediction_cost_telemetry_only is True

    sim_cfg.enable_experimental_prediction_cost_charging = True
    high_charged = express_cognitive_phenotype(_genome(pred_lr=0.5), env_cfg=env_cfg, sim_cfg=sim_cfg)
    charged = compute_cognitive_budget_ledger(high_charged, env_cfg=env_cfg, sim_cfg=sim_cfg)
    assert charged.prediction_cost_charging_enabled is True
    assert charged.prediction_update_cost_charged == high_ledger.prediction_update_cost_hypothetical
    assert charged.charged_energy_delta >= charged.prediction_update_cost_charged


def test_stage3_prediction_metrics_are_deterministic_and_include_ablation_and_guardrails() -> None:
    env_cfg = EnvConfig(width=8, height=8)
    sim_cfg = SimConfig(seed=3)
    sim_cfg.force_prediction_zero_lr = True
    sim_cfg.prediction_pressure_scenario = True
    sim_cfg.prediction_pressure_type = "delayed_consequence"
    agents = [
        _agent(1, _genome(pred_lr=0.0), stats=_stats(pred_err_sum=1.0, pred_err_count=1, patch_habitat_lineage_id=1)),
        _agent(2, _genome(pred_lr=0.5), stats=_stats(pred_err_sum=3.0, pred_err_count=1, patch_habitat_lineage_id=2)),
    ]

    a = compute_cognitive_status_metrics(agents, selected_parent_agents=agents[:1], env_cfg=env_cfg, sim_cfg=sim_cfg)
    b = compute_cognitive_status_metrics(agents, selected_parent_agents=agents[:1], env_cfg=env_cfg, sim_cfg=sim_cfg)

    assert a == b
    assert a["prediction_lr_min"] == 0.0
    assert a["prediction_lr_max"] == 0.5
    assert a["effective_prediction_lr_mean"] == 0.0
    assert a["prediction_force_zero_lr_active_flag"] == 1.0
    assert a["prediction_pressure_scenario_flag"] == 1.0
    assert a["prediction_pressure_delayed_consequence_flag"] == 1.0
    assert "memory_perception_prediction_strategy_count_approx" in a
    assert "prediction_enabled_without_cost_fraction" in a
    assert a["direct_prediction_error_selection_pressure_present_flag"] == 1.0


def test_stage3_direct_prediction_selection_pressure_can_be_audited_and_disabled() -> None:
    sim_cfg = SimConfig(seed=4)
    agent = _agent(1, _genome(pred_lr=0.5), stats=_stats(survival_steps=30, pred_err_sum=10.0, pred_err_count=2))
    probe = _SelectionProbe(sim_cfg)
    pressured = probe._effective_fitness(agent)
    assert direct_prediction_selection_pressure_present(sim_cfg) is True

    sim_cfg.disable_prediction_error_selection_pressure = True
    probe_disabled = _SelectionProbe(sim_cfg)
    unpressured = probe_disabled._effective_fitness(agent)
    assert direct_prediction_selection_pressure_present(sim_cfg) is False
    assert unpressured > pressured

    metrics = compute_cognitive_status_metrics([agent], env_cfg=EnvConfig(width=8, height=8), sim_cfg=sim_cfg)
    assert metrics["prediction_direct_selection_pressure_disabled_flag"] == 1.0
    assert metrics["direct_prediction_error_selection_pressure_present_flag"] == 0.0


def test_stage3_reset_each_tick_clears_prediction_state_without_shape_change() -> None:
    env_cfg = EnvConfig(width=8, height=8)
    sim_cfg = SimConfig(seed=5)
    sim_cfg.pred_horizon = 0
    sim_cfg.reset_prediction_each_tick = True
    agent = _agent(1, _genome(pred_lr=0.5))
    agent.stats.exp_dE[:] = [1.0, 2.0, 3.0, 4.0]
    agent.stats.exp_dInj[:] = [0.4, 0.3, 0.2, 0.1]
    shape_before = tuple(agent.stats.exp_dE.shape), tuple(agent.stats.exp_dInj.shape)

    probe = _DriveProbe(sim_cfg, env_cfg)
    probe._update_internal_drives([agent], np.asarray([1], dtype=np.int32), tick_index=0)

    assert (tuple(agent.stats.exp_dE.shape), tuple(agent.stats.exp_dInj.shape)) == shape_before
    assert agent.stats.prediction_reset_count == 1
    assert np.allclose(agent.stats.exp_dE, np.zeros_like(agent.stats.exp_dE))
    assert np.allclose(agent.stats.exp_dInj, np.zeros_like(agent.stats.exp_dInj))


def test_stage3_prediction_pressure_fixture_is_deterministic_shape_preserving_and_reward_free() -> None:
    fixture = _DelayedConsequenceFixture()
    meta = fixture.metadata()
    assert meta["prediction_pressure_scenario"] is True
    assert meta["prediction_pressure_type"] == "delayed_consequence"
    assert meta["direct_prediction_reward"] is False

    normal_a = fixture.observations(ablated=False)
    normal_b = fixture.observations(ablated=False)
    ablated = fixture.observations(ablated=True)
    assert [x.tolist() for x in normal_a] == [x.tolist() for x in normal_b]
    assert [tuple(x.shape) for x in normal_a] == [tuple(x.shape) for x in ablated]

    root = Path(__file__).resolve().parents[3]
    selection_text = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (
            root / "engine" / "simulation" / "runner_pkg" / "runner_selection.py",
            root / "engine" / "evolution" / "selection.py",
        )
    )
    assert "prediction_reward" not in selection_text
    assert "direct_prediction_reward" not in selection_text
