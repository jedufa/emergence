from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from agents.agent import Agent
from agents.genome import Genome
from agents.structural_identity_v1 import new_native_identity
from env.env_config import EnvConfig
from evolution.population import PopConfig
from evolution.recombination_c2 import recombine_lineage_payloads
from evolution.structural_development_runtime_v1 import (
    resolve_agent_development_runtime,
)
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    DEVELOPMENTAL_FEATURE_PROFILE,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
)
from evolution.structural_live_modules_v1 import (
    LIVE_PROFILE,
    reconcile_runtime_state_for_genome,
)
from simulation.checkpoint_bundle_v2 import (
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _identity(index: int) -> dict:
    return new_native_identity(
        context={
            "campaign_id": "escap6-boundary-test",
            "branch_id": "development-lifecycle",
            "generation": 1,
            "tick": 0,
            "child_agent_id": index,
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap6-boundary-lineage",
        },
        operator_id="fixture",
        attempt_index=index,
        operator_role=f"module-{index}",
    )


def _development(
    gains: tuple[float, float, float],
    *,
    prerequisites: tuple[str, ...] = (),
    latch: bool = False,
) -> dict:
    return {
        "schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
        "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
        "stage_gains": list(gains),
        "prerequisite_homology_ids": list(prerequisites),
        "latch_dependency_readiness": latch,
    }


def _module(
    index: int,
    *,
    gains: tuple[float, float, float] = (0.25, 0.5, 0.75),
    prerequisites: tuple[str, ...] = (),
    latch: bool = False,
) -> dict:
    identity = _identity(index)
    return {
        "module_id": f"m{index:03d}",
        "module_type": "generic",
        "params": {
            "live_operator": "bounded_scalar_integrator_v1",
            "expression_scale": 1.0,
            "live_retention": 0.5,
            "live_input_gain": 1.0,
            "live_output_gain": 0.5,
            DEVELOPMENTAL_EXPRESSION_BLOCK_KEY: _development(
                gains,
                prerequisites=prerequisites,
                latch=latch,
            ),
        },
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _cfg() -> SimConfig:
    cfg = SimConfig()
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.policy_tick_stride = 1
    cfg.structural_live_modules_enable = True
    cfg.structural_live_modules_profile = LIVE_PROFILE
    cfg.structural_development_enable = True
    cfg.structural_development_profile = DEVELOPMENTAL_FEATURE_PROFILE
    return cfg


def _real_runner(root) -> SimulationRunner:
    cfg = _cfg()
    # This frozen seed gives the four-agent fixture both policy families.
    cfg.seed = 18105
    cfg.generations = 1
    cfg.steps_per_generation = 1
    cfg.behavior_log_enable = False
    cfg.policy_deterministic_actions = True
    cfg.mem_max_live_level4_instrumentation_enable = True
    sim = SimulationRunner(
        EnvConfig(width=8, height=8),
        PopConfig(population_size=4),
        cfg,
        NoOpMetricsLogger(),
        str(root),
        perf=False,
    )
    sim.run(start_gen=0, skip_init=False)
    return sim


def _install_module(sim: SimulationRunner, module: dict, *, age: int) -> list[Agent]:
    alive = [agent for agent in sim.pop.agents.values() if agent.alive]
    for agent in alive:
        raw = agent.genome.to_dict()
        raw["modules"] = [deepcopy(module)]
        raw["connections"] = []
        raw["b2_invalid_policy"] = "reject"
        agent.genome = Genome.from_dict(raw)
        agent.stats.survival_steps = age
        agent.structural_live_module_state = {}
        agent.structural_live_module_outputs = {}
        agent.structural_live_module_telemetry = {}
        agent.structural_development_latch_by_instance_id = {}
        agent.structural_development_stage_id = ""
        agent.structural_development_effective_gain_by_instance_id = {}
        agent.drives[2] = np.float32(0.4)
        sim._drive_fatigue[int(agent.slot)] = np.float32(0.4)
    return alive


def _set_age_and_fatigue(sim: SimulationRunner, age: int) -> list[Agent]:
    alive = [agent for agent in sim.pop.agents.values() if agent.alive]
    for agent in alive:
        agent.stats.survival_steps = age
        agent.drives[2] = np.float32(0.4)
        sim._drive_fatigue[int(agent.slot)] = np.float32(0.4)
    return alive


def _f32(value: float) -> float:
    return float(np.float32(value))


def _assert_runtime_row(
    agent: Agent,
    *,
    instance_id: str,
    stage_id: str,
    gain: float,
    state_before: float,
    state_after: float,
    output: float,
) -> None:
    assert agent.structural_development_stage_id == stage_id
    assert agent.structural_development_effective_gain_by_instance_id == {
        instance_id: gain
    }
    assert agent.structural_live_module_state == {instance_id: state_after}
    assert agent.structural_live_module_outputs == {instance_id: output}
    telemetry = agent.structural_live_module_telemetry
    assert telemetry["random_draws"] == 0
    assert telemetry["structural_development"]["random_draws"] == 0
    assert telemetry["structural_development"]["stage_id"] == stage_id
    assert telemetry["structural_development"]["age_steps"] == agent.stats.survival_steps
    row = telemetry["state_rows"][0]
    assert row["state_before"] == state_before
    assert row["state_after"] == state_after
    assert row["developmental_expression_gain"] == gain
    assert row["base_live_output"] == _f32(0.5 * state_after)
    assert row["typed_output"] == output


@pytest.mark.parametrize(
    ("start_age", "end_age", "start_stage", "end_stage", "start_gain", "end_gain"),
    (
        (19, 20, "early", "mature", 0.25, 0.5),
        (39, 40, "mature", "late", 0.5, 0.75),
    ),
)
def test_real_runner_boundaries_checkpoint_resume_exact_for_mlp_and_gru(
    tmp_path,
    start_age: int,
    end_age: int,
    start_stage: str,
    end_stage: str,
    start_gain: float,
    end_gain: float,
) -> None:
    module = _module(1)
    instance_id = module["module_instance_id"]
    source = _real_runner(tmp_path / f"source-{start_age}")
    source_alive = _install_module(source, module, age=start_age)
    assert {"gru" if agent.genome.use_gru else "mlp" for agent in source_alive} == {
        "mlp",
        "gru",
    }

    policy_rng_before = source.rng_manager.capture_snapshot(
        generation_index=source.gen
    )["streams"]["policy"]["indices"]
    source._batched_actions(source_alive)
    policy_rng_after = source.rng_manager.capture_snapshot(
        generation_index=source.gen
    )["streams"]["policy"]["indices"]
    assert policy_rng_after == policy_rng_before

    fatigue = float(np.float32(0.4))
    first_state = _f32(fatigue)
    first_output = _f32(start_gain * _f32(0.5 * first_state))
    for agent in source_alive:
        _assert_runtime_row(
            agent,
            instance_id=instance_id,
            stage_id=start_stage,
            gain=start_gain,
            state_before=0.0,
            state_after=first_state,
            output=first_output,
        )

    bundle = checkpoint_bundle_v2_save(
        source,
        str(tmp_path / f"bundle-{start_age}"),
        generation_index=1,
        strict_replay_mode=True,
    )
    loaded = checkpoint_bundle_v2_load(bundle, strict=True)
    restored = _real_runner(tmp_path / f"restored-{start_age}")
    checkpoint_bundle_v2_apply(restored, loaded, restore_rng=True)

    source_alive = _set_age_and_fatigue(source, end_age)
    restored_alive = _set_age_and_fatigue(restored, end_age)
    source_actions = source._batched_actions(source_alive)
    restored_actions = restored._batched_actions(restored_alive)
    assert source_actions.tobytes() == restored_actions.tobytes()

    second_state = _f32(0.5 * first_state + fatigue)
    second_output = _f32(end_gain * _f32(0.5 * second_state))
    for source_agent, restored_agent in zip(
        source_alive, restored_alive, strict=True
    ):
        assert source_agent.agent_id == restored_agent.agent_id
        _assert_runtime_row(
            source_agent,
            instance_id=instance_id,
            stage_id=end_stage,
            gain=end_gain,
            state_before=first_state,
            state_after=second_state,
            output=second_output,
        )
        assert (
            restored_agent.structural_live_module_state
            == source_agent.structural_live_module_state
        )
        assert (
            restored_agent.structural_live_module_outputs
            == source_agent.structural_live_module_outputs
        )
        assert (
            restored_agent.structural_live_module_telemetry
            == source_agent.structural_live_module_telemetry
        )
        assert (
            restored_agent.structural_development_stage_id
            == source_agent.structural_development_stage_id
        )
        assert (
            restored_agent.structural_development_effective_gain_by_instance_id
            == source_agent.structural_development_effective_gain_by_instance_id
        )

    source_rng = source.rng_manager.capture_snapshot(generation_index=source.gen)
    restored_rng = restored.rng_manager.capture_snapshot(generation_index=restored.gen)
    assert (
        source_rng["streams"]["policy"]["indices"]
        == restored_rng["streams"]["policy"]["indices"]
    )


def test_latch_follows_instance_through_repack_then_prunes_on_delete_and_not_slot_reuse() -> None:
    root = _module(10, gains=(1.0, 0.0, 0.0))
    dependent = _module(
        11,
        gains=(1.0, 1.0, 1.0),
        prerequisites=(root["homology_id"],),
        latch=True,
    )
    agent = Agent(
        agent_id=1,
        genome=Genome.from_dict(
            {
                "schema_version": "modular_genome_v1",
                "b2_invalid_policy": "reject",
                "genes": {},
                "modules": [deepcopy(root), deepcopy(dependent)],
                "connections": [],
            }
        ),
        policy=object(),
        energy=10.0,
        health=5.0,
    )
    plan = resolve_agent_development_runtime(agent, sim_cfg=_cfg())
    agent.structural_development_latch_by_instance_id = dict(
        plan.latch_by_instance_id
    )
    dependent_instance = dependent["module_instance_id"]
    assert agent.structural_development_latch_by_instance_id == {
        dependent_instance: True
    }

    # Repacking changes only the local slot/module label. Stable owner identity
    # remains the module_instance_id, so the lifetime commitment remains owned.
    repacked = deepcopy(dependent)
    repacked["module_id"] = "m099"
    agent.genome.modules = [deepcopy(root), repacked]
    reconcile_runtime_state_for_genome(agent, orphan_policy="reject")
    assert agent.structural_development_latch_by_instance_id == {
        dependent_instance: True
    }
    assert resolve_agent_development_runtime(
        agent, sim_cfg=_cfg()
    ).latch_by_instance_id == {dependent_instance: True}

    # Deletion is an explicit lifecycle boundary and prunes the old owner.
    agent.genome.modules = [deepcopy(root)]
    pruned = reconcile_runtime_state_for_genome(agent, orphan_policy="prune")
    assert pruned["pruned_development_latch_instance_ids"] == [dependent_instance]
    assert agent.structural_development_latch_by_instance_id == {}

    # Reusing the deleted local module slot with a new stable identity must not
    # inherit the deleted organism-lifetime commitment.
    replacement = _module(
        12,
        gains=(1.0, 1.0, 1.0),
        prerequisites=(root["homology_id"],),
        latch=True,
    )
    replacement["module_id"] = "m099"
    agent.genome.modules = [deepcopy(root), replacement]
    agent.stats.survival_steps = 20
    replacement_plan = resolve_agent_development_runtime(agent, sim_cfg=_cfg())
    replacement_instance = replacement["module_instance_id"]
    assert replacement_instance != dependent_instance
    assert replacement_plan.effective_gain_by_instance_id[replacement_instance] == 0.0
    assert replacement_plan.latch_by_instance_id == {}


class _OneDrawRng:
    def __init__(self) -> None:
        self.count = 0

    def random(self) -> float:
        self.count += 1
        return 0.25


def test_newborn_and_c2_materialized_child_start_with_empty_lifetime_maps() -> None:
    shared_identity = _identity(20)
    parent_a = _module(21, gains=(0.0, 0.5, 1.0), latch=True)
    parent_b = _module(22, gains=(1.0, 0.25, 0.0), latch=False)
    for module in (parent_a, parent_b):
        module.update(deepcopy(shared_identity))
        module["latent_pos"] = list(shared_identity["topology_pos"])

    def payload(module: dict) -> dict:
        return {
            "schema_version": "modular_genome_v1",
            "b2_invalid_policy": "reject",
            "genes": {},
            "modules": [deepcopy(module)],
            "connections": [],
        }

    rng = _OneDrawRng()
    child_payload, summary = recombine_lineage_payloads(
        parent_a_payload=payload(parent_a),
        parent_b_payload=payload(parent_b),
        rng=rng,
    )
    assert summary.budget_draws == rng.count == 1
    assert "structural_development_runtime" not in child_payload

    newborn = Agent(
        agent_id=2,
        genome=Genome.from_dict(child_payload),
        policy=object(),
        energy=10.0,
        health=5.0,
    )
    assert newborn.structural_development_latch_by_instance_id == {}
    assert newborn.structural_development_stage_id == ""
    assert newborn.structural_development_effective_gain_by_instance_id == {}
    assert newborn.structural_live_module_state == {}
    assert newborn.structural_live_module_outputs == {}
