from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from experiments.patch_jungle_eval import scenario_knob_pairs
from knobs_override import apply_knob_overrides
from simulation.metrics import MetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


WASTE_VARIANT_KNOBS: dict[str, tuple[str, ...]] = {
    "waste_low": (
        "env_cfg.patch_waste_regrowth_penalty_scale=0.25",
        "env_cfg.patch_waste_regrowth_penalty_max=0.25",
        "env_cfg.patch_density_regrowth_penalty_scale=0.25",
        "env_cfg.patch_density_regrowth_penalty_max=0.25",
        "env_cfg.patch_regrowth_min_multiplier=0.25",
    ),
    "waste_medium": (
        "env_cfg.patch_waste_regrowth_penalty_scale=0.5",
        "env_cfg.patch_waste_regrowth_penalty_max=0.5",
        "env_cfg.patch_density_regrowth_penalty_scale=0.5",
        "env_cfg.patch_density_regrowth_penalty_max=0.5",
        "env_cfg.patch_regrowth_min_multiplier=0.1",
    ),
    "waste_high": (
        "env_cfg.patch_waste_regrowth_penalty_scale=0.75",
        "env_cfg.patch_waste_regrowth_penalty_max=0.75",
        "env_cfg.patch_density_regrowth_penalty_scale=0.75",
        "env_cfg.patch_density_regrowth_penalty_max=0.75",
        "env_cfg.patch_regrowth_min_multiplier=0.05",
    ),
}


def test_policy_drive_packed_k_staging_flag_is_default_off_and_strictly_opt_in() -> None:
    env_cfg = EnvConfig(width=6, height=6, view_radius=1)
    pop_cfg = PopConfig(population_size=4)
    sim_cfg = SimConfig(device="cpu", bank_device="cpu")

    assert sim_cfg.perf_policy_drive_packed_k_staging_enable is False

    applied, genome_overrides = apply_knob_overrides(
        knob_pairs=["sim_cfg.perf_policy_drive_packed_k_staging_enable=true"],
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
    )

    assert applied == ["sim_cfg.perf_policy_drive_packed_k_staging_enable=true"]
    assert genome_overrides == {}
    assert sim_cfg.perf_policy_drive_packed_k_staging_enable is True


def _run_c2_smoke(tmp_path: Path, *, scenario_id: str, variant: str, seed: int = 101) -> Path:
    run_dir = tmp_path / f"{scenario_id}_{variant}_{seed}"
    env_cfg = EnvConfig(width=8, height=8, view_radius=1)
    pop_cfg = PopConfig(population_size=8)
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=2,
        steps_per_generation=8,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    sim_cfg.phase_e_e1_enable = False
    sim_cfg.epi_enable = False
    sim_cfg.checkpoint_bundle_v2_period_gen = 0

    applied, genome_overrides = apply_knob_overrides(
        knob_pairs=[*scenario_knob_pairs(scenario_id), *WASTE_VARIANT_KNOBS[variant]],
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
    )
    assert applied
    assert genome_overrides == {}

    sim = SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=MetricsLogger(str(run_dir)),
        run_dir=str(run_dir),
        perf=False,
    )
    sim.run(start_gen=0, skip_init=False)
    return run_dir


def _metrics_rows(run_dir: Path) -> list[dict[str, str]]:
    with (run_dir / "metrics.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _assert_finite(row: dict[str, str], keys: tuple[str, ...]) -> None:
    for key in keys:
        assert key in row
        assert math.isfinite(float(row[key]))


@pytest.mark.parametrize("variant", ("waste_low", "waste_medium", "waste_high"))
def test_campaign_c2_full_jungle_recovery_waste_variants_emit_nonempty_metrics(
    tmp_path: Path,
    variant: str,
) -> None:
    run_dir = _run_c2_smoke(tmp_path, scenario_id="s09_full_jungle_recovery", variant=variant)
    rows = _metrics_rows(run_dir)
    assert rows
    final = rows[-1]

    assert int(float(final["patch_world_waste_consequence_enable"])) == 1
    assert int(float(final["jungle_predator_pressure_enable"])) == 1
    assert int(float(final["jungle_ecological_recovery_enable"])) == 1
    _assert_finite(
        final,
        (
            "patch_regrowth_suppressed_total",
            "patch_regrowth_multiplier_mean",
            "jungle_predator_damage_total",
            "cog_capacity_alive_end_mean",
            "cog_capacity_memory_component_mean",
            "cog_capacity_prediction_component_mean",
            "cog_capacity_drive_component_mean",
            "cog_capacity_perception_component_mean",
        ),
    )


@pytest.mark.parametrize("scenario_id", ("s08_full_jungle_no_recovery", "s09_full_jungle_recovery"))
def test_campaign_c2_high_waste_full_jungle_paths_do_not_hit_metrics_schema_typeerror(
    tmp_path: Path,
    scenario_id: str,
) -> None:
    run_dir = _run_c2_smoke(tmp_path, scenario_id=scenario_id, variant="waste_high", seed=202)
    rows = _metrics_rows(run_dir)
    assert rows

    final = rows[-1]
    assert int(float(final["patch_world_waste_consequence_enable"])) == 1
    assert int(float(final["jungle_predator_pressure_enable"])) == 1
    _assert_finite(
        final,
        (
            "patch_waste_mean",
            "patch_density_mean",
            "patch_food_amount_mean",
            "cog_capacity_alive_end_mean",
            "perception_cost_total",
            "sensor_radius_mean",
        ),
    )
