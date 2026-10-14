from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from campaigns.production_engine_components import production_domain_components_v1
from env.env_config import EnvConfig
from env.gridworld import GridWorld
from env.gridworld_pkg.patch_world import (
    EED_RESOURCE_IMPACT_ARRAY_NAMES,
    EED_RESOURCE_IMPACT_OVERLAY_ID,
    EED_RESOURCE_IMPACT_SCHEMA_VERSION,
    PATCH_WORLD_ARRAY_NAMES,
    apply_respec0b_resource_gather,
    eed_fixed_resource_axis,
)
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import (
    _apply_world,
    _export_world,
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.rng_manager import RNGManager
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _module():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "bin"
        / "eed0_rp1_reference_fixture.py"
    )
    spec = importlib.util.spec_from_file_location(
        "eed0_rp1_reference_fixture",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rp1 = _module()


def test_defaults_are_off_and_overlay_is_production_selectable() -> None:
    cfg = EnvConfig()
    assert cfg.eed_resource_impact_enable is False
    assert cfg.eed_resource_impact_apply_enable is False
    assert cfg.eed_ecology_frozen_enable is False
    assert cfg.eed_phenotype_fixed_enable is False
    assert cfg.eed_phenotype_fixed_axis_profile == "neutral"
    assert cfg.eed_resource_a_initial_amount == 1.0
    assert cfg.eed_resource_b_initial_amount == 1.0
    assert cfg.eed_resource_a_capacity == 1.0
    assert cfg.eed_resource_b_capacity == 1.0
    assert cfg.eed_resource_a_renewal_per_step == 0.25
    assert cfg.eed_resource_b_renewal_per_step == 0.25
    assert cfg.eed_resource_impact_per_success == 0.25
    assert cfg.eed_resource_impact_decay_per_step == 0.25
    assert cfg.eed_resource_impact_max_load == 1.0
    assert EED_RESOURCE_IMPACT_OVERLAY_ID in {
        component.component_id
        for component in production_domain_components_v1()
    }


def test_ecology_frozen_preserves_immediate_handling_but_blocks_stock_and_pending() -> None:
    cfg = rp1._cfg(overlay_enabled=True, impact_applied=True)
    cfg.eed_ecology_frozen_enable = True
    env = rp1._env(cfg, seed=47005)
    y, x = rp1._resource_cell(env, rp1.RESOURCES["resource_a"])
    env.add_agent(0, (y, x))
    before_a = env.eed_resource_a_amount.copy()
    before_pending = env.eed_resource_a_pending_impact.copy()
    gained = np.zeros((2,), dtype=np.float32)

    apply_respec0b_resource_gather(
        env,
        np.zeros((2,), dtype=np.int32),
        np.full((2,), rp1.ACT_GATHER, dtype=np.int32),
        np.full((2,), y, dtype=np.int32),
        np.full((2,), x, dtype=np.int32),
        gained,
        np.full((2,), rp1.PHENOTYPES["a_oriented"], dtype=np.float32),
    )

    assert float(np.sum(gained, dtype=np.float64)) > 0.0
    assert np.array_equal(env.eed_resource_a_amount, before_a)
    assert np.array_equal(env.eed_resource_a_pending_impact, before_pending)
    stats = env.respec0b_resource_gather_stats_last
    assert stats["eed_ecology_frozen"] == 1
    assert stats["eed_resource_a_depletion_blocked"] > 0.0
    assert stats["eed_resource_a_pending_impact_blocked"] > 0.0


@pytest.mark.parametrize(
    ("profile", "expected"),
    (
        ("neutral", [0.0, 0.0, 0.0, 0.0]),
        ("a_high", [1.0, 1.0, 1.0, 1.0]),
        ("b_high", [-1.0, -1.0, -1.0, -1.0]),
        ("symmetric_mix", [1.0, -1.0, 1.0, -1.0]),
    ),
)
def test_phenotype_fixed_profiles_project_declared_axis(
    profile: str,
    expected: list[float],
) -> None:
    cfg = rp1._cfg(overlay_enabled=True, impact_applied=True)
    cfg.eed_phenotype_fixed_enable = True
    cfg.eed_phenotype_fixed_axis_profile = profile

    observed = [
        eed_fixed_resource_axis(cfg, slot=index, observed_axis=-0.25)
        for index in range(4)
    ]

    assert observed == expected


def test_overlay_requires_respec_and_allocates_independent_state() -> None:
    invalid = EnvConfig(width=2, height=1)
    invalid.enable_biomes = False
    invalid.patch_world_enable = True
    invalid.eed_resource_impact_enable = True
    with pytest.raises(
        ValueError,
        match="requires respec0b_resource_specialization_enable",
    ):
        GridWorld(
            invalid,
            seed=47,
            run_dir=None,
            rng=RNGManager(global_seed=47).env_np,
        )

    env = rp1._env(
        rp1._cfg(overlay_enabled=True, impact_applied=False),
        seed=47003,
    )
    for name in EED_RESOURCE_IMPACT_ARRAY_NAMES:
        assert hasattr(env, name)
        assert name not in PATCH_WORLD_ARRAY_NAMES
    assert not np.shares_memory(
        env.eed_resource_a_amount,
        env.eed_resource_b_amount,
    )
    before_b = env.eed_resource_b_amount.copy()
    env.eed_resource_a_amount.fill(0.125)
    assert np.array_equal(env.eed_resource_b_amount, before_b)


def test_impact_ablation_preserves_immediate_handling_and_energy() -> None:
    for resource, phenotype, pending_key in (
        ("resource_a", "a_oriented", "pending_a"),
        ("resource_b", "b_oriented", "pending_b"),
    ):
        off = rp1.run_arm(
            resource=resource,
            phenotype=phenotype,
            overlay_enabled=True,
            impact_applied=False,
        )
        on = rp1.run_arm(
            resource=resource,
            phenotype=phenotype,
            overlay_enabled=True,
            impact_applied=True,
        )
        assert off["agent_count"] == on["agent_count"] == 1
        assert off["action_count"] == on["action_count"] == 2
        assert off["immediate_stock"] == on["immediate_stock"]
        assert off["immediate_depletion"] == on["immediate_depletion"]
        assert off["immediate_energy_by_action"] == on["immediate_energy_by_action"]
        assert off["impact_after_action"][pending_key] == 0.0
        assert on["impact_after_action"][pending_key] > 0.0


def test_both_reciprocal_later_reproductive_opportunities_change() -> None:
    built = rp1.build_evidence()
    assert built["reciprocal"]["a_direction_pass"] is True
    assert built["reciprocal"]["b_direction_pass"] is True
    assert (
        built["reciprocal"]["a_impact_on_later_stock"]
        < built["reciprocal"]["a_impact_off_later_stock"]
    )
    assert (
        built["reciprocal"]["b_impact_on_later_stock"]
        < built["reciprocal"]["b_impact_off_later_stock"]
    )
    assert (
        built["reciprocal"]["a_impact_on_try_prob"]
        < built["reciprocal"]["a_impact_off_try_prob"]
    )
    assert (
        built["reciprocal"]["b_impact_on_try_prob"]
        < built["reciprocal"]["b_impact_off_try_prob"]
    )


def test_neutral_and_equalized_sham_do_not_queue_impact() -> None:
    built = rp1.build_evidence()
    for row in built["rows"]:
        if row["phenotype"] not in {"neutral", "equalized_capacity_sham"}:
            continue
        assert row["impact_after_action"]["pending_a"] == 0.0
        assert row["impact_after_action"]["pending_b"] == 0.0


def test_flags_off_repeatability_and_rng_are_exact() -> None:
    built = rp1.build_evidence()
    assert built["flags_off"]["pass"] is True
    assert built["repeatability"]["pass"] is True
    assert built["rng"]["pass"] is True
    assert built["rng"]["new_rng_draws"] == 0
    assert built["decision"]["decision"] == "EED-0-RP1-A"


def test_conditional_checkpoint_plane_round_trips_and_continues_exactly() -> None:
    cfg = rp1._cfg(overlay_enabled=True, impact_applied=True)
    source = rp1._env(cfg, seed=47021)
    source.eed_resource_a_amount[0, 0] = np.float32(0.125)
    source.eed_resource_b_amount[0, 1] = np.float32(0.375)
    source.eed_resource_a_pending_impact[0, 0] = np.float32(0.1875)
    source.eed_resource_b_pending_impact[0, 1] = np.float32(0.0625)
    source.eed_resource_a_impact_load[0, 0] = np.float32(0.25)
    source.eed_resource_b_impact_load[0, 1] = np.float32(0.5)

    payload = json.loads(json.dumps(_export_world(SimpleNamespace(env=source))))
    eed_plane = payload["patch_world"]["eed_resource_impact"]
    assert eed_plane["schema_version"] == EED_RESOURCE_IMPACT_SCHEMA_VERSION
    assert eed_plane["overlay_id"] == EED_RESOURCE_IMPACT_OVERLAY_ID
    assert set(eed_plane["arrays"]) == set(EED_RESOURCE_IMPACT_ARRAY_NAMES)

    restored = rp1._env(cfg, seed=99991)
    _apply_world(SimpleNamespace(env=restored), payload)
    for name in EED_RESOURCE_IMPACT_ARRAY_NAMES:
        assert np.array_equal(getattr(restored, name), getattr(source, name))

    source.patch_world_step(1)
    restored.patch_world_step(1)
    for name in EED_RESOURCE_IMPACT_ARRAY_NAMES:
        assert np.array_equal(getattr(restored, name), getattr(source, name))
    assert np.array_equal(restored.patch_food_amount, source.patch_food_amount)


def test_conditional_checkpoint_plane_is_absent_off_and_required_on() -> None:
    disabled = rp1._env(
        rp1._cfg(overlay_enabled=False, impact_applied=False),
        seed=47031,
    )
    payload = _export_world(SimpleNamespace(env=disabled))
    assert "eed_resource_impact" not in payload["patch_world"]

    enabled_cfg = rp1._cfg(overlay_enabled=True, impact_applied=True)
    enabled = rp1._env(enabled_cfg, seed=47032)
    missing = _export_world(SimpleNamespace(env=enabled))
    del missing["patch_world"]["eed_resource_impact"]
    with pytest.raises(ValueError, match="EED_RESOURCE_IMPACT_CHECKPOINT_MISSING"):
        _apply_world(
            SimpleNamespace(env=rp1._env(enabled_cfg, seed=47033)),
            missing,
        )

    mismatched = _export_world(SimpleNamespace(env=enabled))
    mismatched["patch_world"]["eed_resource_impact"]["overlay_id"] = "wrong"
    with pytest.raises(ValueError, match="EED_RESOURCE_IMPACT_OVERLAY_MISMATCH"):
        _apply_world(
            SimpleNamespace(env=rp1._env(enabled_cfg, seed=47034)),
            mismatched,
        )


def test_checkpoint_bundle_v2_preserves_eed_continuation(tmp_path: Path) -> None:
    def runner(path: Path, *, seed: int) -> SimulationRunner:
        sim_cfg = SimConfig(
            seed=seed,
            generations=2,
            steps_per_generation=3,
            device="cpu",
            bank_device="cpu",
            behavior_log_enable=False,
            policy_deterministic_actions=True,
            policy_tick_stride=1,
        )
        sim_cfg.phase_e_e1_enable = False
        sim_cfg.epi_enable = False
        sim_cfg.checkpoint_bundle_v2_period_gen = 0
        return SimulationRunner(
            env_cfg=rp1._cfg(overlay_enabled=True, impact_applied=True),
            pop_cfg=PopConfig(population_size=2),
            sim_cfg=sim_cfg,
            logger=NoOpMetricsLogger(),
            run_dir=str(path),
            perf=False,
        )

    source = runner(tmp_path / "source", seed=47041)
    source.env.eed_resource_a_amount[0, 0] = np.float32(0.125)
    source.env.eed_resource_a_pending_impact[0, 0] = np.float32(0.1875)
    source.env.eed_resource_a_impact_load[0, 0] = np.float32(0.25)
    bundle = tmp_path / "eed_checkpoint"
    checkpoint_bundle_v2_save(
        source,
        str(bundle),
        generation_index=1,
        strict_replay_mode=True,
    )
    payload = checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        expected_commit_hash=None,
    )

    restored = runner(tmp_path / "restored", seed=99941)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)
    for name in EED_RESOURCE_IMPACT_ARRAY_NAMES:
        assert np.array_equal(
            getattr(restored.env, name),
            getattr(source.env, name),
        )

    source.env.patch_world_step(1)
    restored.env.patch_world_step(1)
    for name in EED_RESOURCE_IMPACT_ARRAY_NAMES:
        assert np.array_equal(
            getattr(restored.env, name),
            getattr(source.env, name),
        )
    assert np.array_equal(
        restored.env.patch_food_amount,
        source.env.patch_food_amount,
    )
