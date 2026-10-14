from __future__ import annotations

from pathlib import Path

import numpy as np

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from perf.cxxwave1f_cpp_backbone import (
    Cxxwave1fScalarConfig,
    compare_scalar_checkpoints,
    compare_scalar_event_tapes,
    compare_scalar_states,
    compile_scalar_backbone_kernel,
    cxxwave1f_capability_matrix,
    emit_scalar_event_tape_reference,
    live_support_report,
    load_scalar_backbone_kernel,
    pack_scalar_checkpoint_reference,
    run_scalar_backbone_reference,
    sample_scalar_backbone_state,
)
from simulation.metrics import MetricsLogger
from simulation.performance_profiles import FAST_EXPLORATORY_PROFILE_V2, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def test_cxxwave1f_capability_matrix_keeps_live_gate_closed_until_proven() -> None:
    payload = cxxwave1f_capability_matrix()

    assert payload["schema_version"] == "cxxwave1f_support_gate_v1"
    assert payload["summary"]["live_gate_default"] == "closed"
    assert any(row["block"] == "Block 4 chemistry/maintenance" for row in payload["rows"])
    assert any(row["status"] == "native_owned_for_supported_scalar_envelope" for row in payload["rows"])


def test_cxxwave1f_live_support_fails_closed_for_prediction_and_python_grid_scan() -> None:
    report = live_support_report(
        prediction_horizon=2,
        maintenance_opportunity_cost_coeff=0.25,
        library_path="/tmp/libcxxwave1f_scalar_backbone.so",
    )

    assert report["supported"] is False
    assert "prediction_horizon_positive_reads_pre_maintenance_energy" in report["blockers"]
    assert "maintenance_opportunity_score_still_python_grid_scan" in report["blockers"]


def test_cxxwave1f_native_scalar_backbone_matches_reference(tmp_path) -> None:
    compile_info = compile_scalar_backbone_kernel(tmp_path / "build")
    assert compile_info["ok"] is True
    native = load_scalar_backbone_kernel(compile_info["library_path"])
    cfg = Cxxwave1fScalarConfig()
    baseline = sample_scalar_backbone_state(rows=48, capacity=64)
    reference = baseline.copy()
    candidate = baseline.copy()

    reference_result = run_scalar_backbone_reference(reference, cfg)
    native_result = native(candidate, cfg)
    comparison = compare_scalar_states(reference, candidate, atol=3.0e-6)

    assert reference_result["summary"]["drive_rows"] == 48.0
    assert native_result["summary"]["drive_rows"] == 48.0
    assert comparison["ok"] is True


def test_cxxwave1f_native_scalar_backbone_supports_drive_noise_tape(tmp_path) -> None:
    compile_info = compile_scalar_backbone_kernel(tmp_path / "build_noise")
    assert compile_info["ok"] is True
    native = load_scalar_backbone_kernel(compile_info["library_path"])
    cfg = Cxxwave1fScalarConfig(maintenance_enable=False)
    baseline = sample_scalar_backbone_state(rows=24, capacity=32)
    baseline.drive_noise_scale[:] = 0.03
    reference = baseline.copy()
    candidate = baseline.copy()

    run_scalar_backbone_reference(reference, cfg)
    native(candidate, cfg)
    comparison = compare_scalar_states(reference, candidate, atol=3.0e-6)

    assert comparison["ok"] is True


def test_cxxwave1f_native_scalar_event_tape_matches_reference(tmp_path) -> None:
    compile_info = compile_scalar_backbone_kernel(tmp_path / "build_event")
    assert compile_info["ok"] is True
    native = load_scalar_backbone_kernel(compile_info["library_path"])
    state = sample_scalar_backbone_state(rows=40, capacity=48)
    state.alive[3] = 0
    state.slots[7] = -1
    cfg = Cxxwave1fScalarConfig()

    run_scalar_backbone_reference(state, cfg)
    expected = emit_scalar_event_tape_reference(state, tick_index=17)
    actual = native.emit_event_tape(state, tick_index=17)
    comparison = compare_scalar_event_tapes(expected, actual, atol=1.0e-6)

    assert actual["summary"]["emitted_rows"] == expected["summary"]["emitted_rows"]
    assert actual["summary"]["skipped_rows"] == expected["summary"]["skipped_rows"]
    assert comparison["ok"] is True


def test_cxxwave1f_native_scalar_checkpoint_pack_restore_roundtrip(tmp_path) -> None:
    compile_info = compile_scalar_backbone_kernel(tmp_path / "build_checkpoint")
    assert compile_info["ok"] is True
    native = load_scalar_backbone_kernel(compile_info["library_path"])
    source = sample_scalar_backbone_state(rows=36, capacity=44)
    cfg = Cxxwave1fScalarConfig(maintenance_enable=True, maintenance_delay_ticks=1)
    run_scalar_backbone_reference(source, cfg)

    expected = pack_scalar_checkpoint_reference(source)
    actual = native.pack_checkpoint(source)
    comparison = compare_scalar_checkpoints(expected, actual, atol=0.0)
    restored = sample_scalar_backbone_state(rows=36, capacity=44)
    restored.energy[:] = 0.0
    restored.health[:] = 0.0
    restored.injury[:] = 0.0
    restore_result = native.restore_checkpoint(restored, actual)

    assert comparison["ok"] is True
    assert actual["summary"]["capacity"] == float(source.alive.shape[0])
    assert restore_result["summary"]["alive_count"] == float(np.sum(source.alive.astype(np.int32)))
    assert compare_scalar_states(source, restored, atol=0.0)["ok"] is True


def test_cxxwave1f_live_runnercore_scalar_hook_preserves_tiny_fingerprint(tmp_path: Path) -> None:
    compile_info = compile_scalar_backbone_kernel(tmp_path / "live_build")
    assert compile_info["ok"] is True

    control = _prepare_runner(tmp_path / "control", native_library_path="")
    native = _prepare_runner(tmp_path / "native", native_library_path=compile_info["library_path"])

    control_result = _run_tick_loop(control)
    native_result = _run_tick_loop(native)

    assert native_result == control_result
    assert _scalar_fingerprint(native) == _scalar_fingerprint(control)
    summary = native._cxxwave1f_cpp_backbone_summary
    assert summary["decision"] == "CXXWAVE1F-SCALAR-LIVE-NATIVE"
    assert summary["native_ticks"] == 4
    assert summary["native_rows"] > 0
    assert (Path(native.run_dir) / "metrics" / "cxxwave1f_cpp_backbone_summary.json").exists()


def _env_cfg() -> EnvConfig:
    cfg = EnvConfig(width=6, height=6, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 4
    cfg.max_food = 24
    cfg.reproduce_try_prob = 0.0
    cfg.patch_world_enable = False
    cfg.patch_world_observation_enable = False
    cfg.prediction_demand_enable = False
    cfg.prediction_demand_observation_enable = False
    return cfg


def _sim_cfg(*, native_library_path: str) -> SimConfig:
    cfg = SimConfig(
        seed=91051,
        generations=1,
        steps_per_generation=4,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=2,
    )
    apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
    cfg.phase_e_e1_enable = False
    cfg.phase_e_e2_enable = False
    cfg.epi_enable = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.chemistry_functional_coupling_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.maintenance_enable = False
    cfg.maintenance_opportunity_cost_coeff = 0.0
    cfg.pred_horizon = 0
    cfg.perf_aggperf3_timing_buckets_enable = True
    cfg.runnercore1_array_tick_authority_enable = True
    cfg.runnercore1_candidate_enable = True
    cfg.runnercore1_authoritative_enable = False
    cfg.runnercore1_native_enable = False
    if native_library_path:
        cfg.cxxwave1f_scalar_backbone_enable = True
        cfg.cxxwave1f_scalar_backbone_fail_closed = True
        cfg.cxxwave1f_scalar_backbone_library_path = str(native_library_path)
    return cfg


def _prepare_runner(path: Path, *, native_library_path: str) -> SimulationRunner:
    sim = SimulationRunner(
        env_cfg=_env_cfg(),
        pop_cfg=PopConfig(population_size=6),
        sim_cfg=_sim_cfg(native_library_path=native_library_path),
        logger=MetricsLogger(str(path)),
        run_dir=str(path),
        perf=True,
    )
    return sim


def _run_tick_loop(sim: SimulationRunner) -> tuple[int, int, int, float]:
    if not list(sim.pop.alive_agents()):
        sim.pop.init_random()
        sim._ensure_bank()
        sim._place_agents()
        if hasattr(sim, "_mark_bank_dirty"):
            sim._mark_bank_dirty()
    for agent in sim.pop.alive_agents():
        agent.genome.reproduction_threshold = 1.0e9
    sync_fn = getattr(sim, "_phase3_pre_start_generation_sync", None)
    if callable(sync_fn):
        sync_fn(list(sim.pop.alive_agents()))
    return sim._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )


def _fingerprint(sim: SimulationRunner) -> dict[str, object]:
    alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
    ids = [int(a.agent_id) for a in alive]
    hmem_obj = getattr(sim, "_Hmem", None)
    hmem = [] if hmem_obj is None else np.asarray(hmem_obj.detach().cpu().numpy()).round(7).tolist()
    return {
        "alive_ids": ids,
        "energy": [round(float(a.energy), 7) for a in alive],
        "health": [round(float(a.health), 7) for a in alive],
        "injury": [round(float(getattr(a, "injury", 0.0)), 7) for a in alive],
        "scar": [round(float(getattr(a, "scar", 0.0)), 7) for a in alive],
        "pos_y": np.asarray(sim.env.agent_pos_y[ids], dtype=np.int32).tolist() if ids else [],
        "pos_x": np.asarray(sim.env.agent_pos_x[ids], dtype=np.int32).tolist() if ids else [],
        "hmem": hmem,
    }


def _scalar_fingerprint(sim: SimulationRunner) -> dict[str, object]:
    fp = _fingerprint(sim)
    fp.pop("hmem", None)
    return fp
