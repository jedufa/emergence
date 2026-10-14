from __future__ import annotations

from pathlib import Path

from tools.performance.eiel_scientific_config_compare import compare_scientific_config


def _write_run_config(run_dir: Path, *, native: bool, food: int = 16) -> None:
    run_dir.mkdir(parents=True)
    rows = [
        "section,key,value",
        "sim,generations,1",
        "sim,steps_per_generation,80",
        "sim,seed,1001",
        "sim,policy_tick_stride,2",
        "sim,policy_weight_mutation_enable,False",
        "sim,policy_weight_mutation_mode,dense",
        "sim,policy_weight_mutation_rate,1.0",
        "sim,policy_weight_mutation_std,0.0",
        "sim,genome_c1_enable,False",
        "sim,genome_c2_enable,False",
        "sim,chemistry_functional_coupling_enable,False",
        "sim,maintenance_enable,False",
        "sim,native_runtime_enable," + ("True" if native else "False"),
        "sim,native_runtime_max_births_per_generation," + ("512" if native else "1"),
        "sim,native_runtime_cxx26_scope_lock_enable," + ("True" if native else "False"),
        "env,width,16",
        "env,height,16",
        "env,view_radius,1",
        f"env,food_spawn_per_step,{food}",
        "env,max_food,96",
        "env,max_energy,20.0",
        "env,food_energy,5.0",
        "env,move_cost,1.0",
        "env,gather_cost,1.0",
        "env,rest_cost,0.25",
        "env,attack_cost,0.6",
        "env,start_health,12.0",
        "env,reproduction_threshold,12.0",
        "env,reproduce_try_prob,1.0",
        "env,reproduce_requires_mate,False",
        "env,reproduce_parent_energy_share,0.5",
        "env,reproduce_min_energy_buffer,1.0",
        "env,reproduce_cooldown_steps,0",
        "env,enable_injury,False",
        "env,enable_scar,False",
        "env,patch_world_enable,False",
        "env,jungle_predator_pressure_enable,False",
        "env,jungle_ecological_recovery_enable,False",
        "env,wdl_delayed_hazard_enable,False",
    ]
    (run_dir / "run_config.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (run_dir / "metrics.csv").write_text(
        "generation,successful_births,deaths\n0,192,0\n",
        encoding="utf-8",
    )


def test_scientific_config_ignores_native_execution_control_differences(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    native = tmp_path / "native"
    _write_run_config(reference, native=False)
    _write_run_config(native, native=True)

    result = compare_scientific_config(reference, native)

    assert result["ok"] is True
    assert result["scientific_differences"] == []
    assert any(row["key"] == "native_runtime_max_births_per_generation" for row in result["native_execution_control_differences"])
    assert result["effective_birth_cap"]["native_raw_cap_binds"] is False


def test_scientific_config_fails_closed_on_effective_food_difference(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    native = tmp_path / "native"
    _write_run_config(reference, native=False, food=16)
    _write_run_config(native, native=True, food=12)

    result = compare_scientific_config(reference, native)

    assert result["ok"] is False
    assert result["fail_closed"] is True
    assert result["scientific_differences"] == [
        {"section": "env", "key": "food_spawn_per_step", "reference": "16", "native": "12"}
    ]
