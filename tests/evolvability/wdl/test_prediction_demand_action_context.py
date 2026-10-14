from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_apply, checkpoint_bundle_v2_load, checkpoint_bundle_v2_save
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.prediction_demand_action_context import (
    ACTION_LABELS,
    EVENTS_FILENAME,
    GENERATION_SUMMARY_FILENAME,
    OUT_DIRNAME,
    PREDICTION_DEMAND_ACTION_CONTEXT_SCHEMA_VERSION,
    RUN_SUMMARY_FILENAME,
    SCHEMA_FILENAME,
    SIGN_GENERATION_SUMMARY_FILENAME,
    SIGN_SUMMARY_FILENAME,
    SIGN_WINDOW_SUMMARY_FILENAME,
    WINDOW_LABELS,
    prediction_demand_action_context_schema,
)
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _env_cfg(*, prediction_enabled: bool = True, observe: bool = True) -> EnvConfig:
    cfg = EnvConfig(width=1, height=1, view_radius=0)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = bool(prediction_enabled)
    cfg.patch_world_observation_enable = bool(observe and prediction_enabled)
    cfg.prediction_demand_enable = bool(prediction_enabled)
    cfg.prediction_demand_observation_enable = bool(observe and prediction_enabled)
    cfg.prediction_demand_delay_min = 2
    cfg.prediction_demand_delay_max = 4
    cfg.prediction_demand_active_duration = 2
    cfg.prediction_demand_event_period_steps = 8
    cfg.prediction_demand_radius = 0
    cfg.prediction_demand_transition_strength = 0.25
    cfg.prediction_demand_cue_noise = 0.30
    return cfg


def _sign_env_cfg(*, prediction_enabled: bool = True, observe: bool = True) -> EnvConfig:
    cfg = EnvConfig(width=3, height=3, view_radius=1)
    cfg.enable_biomes = False
    cfg.food_spawn_per_step = 0
    cfg.max_food = 0
    cfg.patch_world_enable = bool(prediction_enabled)
    cfg.patch_world_observation_enable = bool(observe and prediction_enabled)
    cfg.prediction_demand_enable = bool(prediction_enabled)
    cfg.prediction_demand_observation_enable = bool(observe and prediction_enabled)
    cfg.prediction_demand_delay_min = 4
    cfg.prediction_demand_delay_max = 4
    cfg.prediction_demand_active_duration = 2
    cfg.prediction_demand_event_period_steps = 10
    cfg.prediction_demand_radius = 1
    cfg.prediction_demand_transition_strength = 0.25
    cfg.prediction_demand_cue_noise = 0.30
    return cfg


def _sim_cfg(
    seed: int,
    *,
    action_context_enabled: bool = False,
    write_events: bool = False,
    sign_summary_enabled: bool = False,
    run_id: str = "fixture",
    steps: int = 12,
    sample_tick_period: int = 1,
    sample_first_n_generations: int = 0,
) -> SimConfig:
    cfg = SimConfig(
        seed=int(seed),
        generations=1,
        steps_per_generation=int(steps),
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=1,
    )
    cfg.phase_e_e1_enable = False
    cfg.epi_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.enable_cognitive_trait_registry = False
    cfg.enable_cognitive_status_metrics = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.prediction_demand_action_context_enable = bool(action_context_enabled)
    cfg.prediction_demand_action_context_write_events = bool(write_events)
    cfg.prediction_demand_action_context_sign_summary_enable = bool(sign_summary_enabled)
    cfg.prediction_demand_action_context_run_id = str(run_id)
    cfg.prediction_demand_action_context_memory_arm = "not_configured"
    cfg.prediction_demand_action_context_prediction_arm = "prediction_fixture"
    cfg.prediction_demand_action_context_init_arm = "init_fixture"
    cfg.prediction_demand_action_context_window_radius = 0
    cfg.prediction_demand_action_context_sample_tick_period = int(sample_tick_period)
    cfg.prediction_demand_action_context_sample_generation_period = 1
    cfg.prediction_demand_action_context_sample_first_n_generations = int(sample_first_n_generations)
    return cfg


def _run(
    tmp_path: Path,
    *,
    seed: int = 123,
    prediction_enabled: bool = True,
    action_context_enabled: bool = False,
    write_events: bool = False,
    sign_summary_enabled: bool = False,
    run_id: str = "fixture",
    sample_tick_period: int = 1,
    sample_first_n_generations: int = 0,
) -> SimulationRunner:
    sim = SimulationRunner(
        env_cfg=_env_cfg(prediction_enabled=prediction_enabled),
        pop_cfg=PopConfig(population_size=1),
        sim_cfg=_sim_cfg(
            seed,
            action_context_enabled=action_context_enabled,
            write_events=write_events,
            sign_summary_enabled=sign_summary_enabled,
            run_id=run_id,
            sample_tick_period=sample_tick_period,
            sample_first_n_generations=sample_first_n_generations,
        ),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )
    sim.run()
    return sim


def _run_sign_fixture(
    tmp_path: Path,
    *,
    seed: int = 123,
    prediction_enabled: bool = True,
    write_events: bool = False,
    run_id: str = "sign_fixture",
) -> SimulationRunner:
    sim = SimulationRunner(
        env_cfg=_sign_env_cfg(prediction_enabled=prediction_enabled),
        pop_cfg=PopConfig(population_size=4),
        sim_cfg=_sim_cfg(
            seed,
            action_context_enabled=True,
            write_events=write_events,
            sign_summary_enabled=True,
            run_id=run_id,
            steps=32,
        ),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )
    sim.run()
    return sim


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fingerprint(sim: SimulationRunner) -> dict[str, object]:
    alive = sorted(sim.pop.alive_agents(), key=lambda a: int(a.agent_id))
    return {
        "alive_ids": [int(a.agent_id) for a in alive],
        "health": [round(float(a.health), 8) for a in alive],
        "energy": [round(float(a.energy), 8) for a in alive],
        "pos_y": np.asarray(sim.env.agent_pos_y[:4], dtype=np.int32).tolist(),
        "pos_x": np.asarray(sim.env.agent_pos_x[:4], dtype=np.int32).tolist(),
        "prediction_events": int(getattr(sim, "_prediction_demand_event_count_gen", 0) or 0),
        "prediction_transitions": int(getattr(sim, "_prediction_demand_transition_count_gen", 0) or 0),
        "resource_consequence": round(float(getattr(sim, "_prediction_demand_resource_consequence_total_gen", 0.0)), 8),
    }


def test_prediction_demand_action_context_default_off_no_files_and_equivalent_state(tmp_path: Path) -> None:
    baseline = _run(tmp_path / "baseline", seed=321, action_context_enabled=False)
    repeated = _run(tmp_path / "repeated", seed=321, action_context_enabled=False)

    assert _fingerprint(baseline) == _fingerprint(repeated)
    assert not (tmp_path / "baseline" / OUT_DIRNAME).exists()
    assert not (tmp_path / "repeated" / OUT_DIRNAME).exists()


def test_prediction_demand_action_context_reporting_on_preserves_live_outcomes(tmp_path: Path) -> None:
    reporting_off = _run(tmp_path / "off", seed=456, action_context_enabled=False)
    reporting_on = _run(tmp_path / "on", seed=456, action_context_enabled=True, run_id="parity")

    assert _fingerprint(reporting_off) == _fingerprint(reporting_on)
    assert (tmp_path / "on" / OUT_DIRNAME / SCHEMA_FILENAME).exists()


def test_prediction_demand_action_context_enabled_emits_schema_and_summaries(tmp_path: Path) -> None:
    _run(tmp_path / "enabled", seed=123, action_context_enabled=True, run_id="summary_only")
    out = tmp_path / "enabled" / OUT_DIRNAME

    assert (out / SCHEMA_FILENAME).exists()
    assert (out / GENERATION_SUMMARY_FILENAME).exists()
    assert (out / RUN_SUMMARY_FILENAME).exists()
    assert not (out / EVENTS_FILENAME).exists()
    assert not (out / SIGN_SUMMARY_FILENAME).exists()
    assert not (out / SIGN_WINDOW_SUMMARY_FILENAME).exists()
    assert not (out / SIGN_GENERATION_SUMMARY_FILENAME).exists()

    schema = _json(out / SCHEMA_FILENAME)
    assert schema["schema_version"] == PREDICTION_DEMAND_ACTION_CONTEXT_SCHEMA_VERSION
    assert schema["read_only"] is True
    assert schema["default_off"] is True
    assert set(WINDOW_LABELS).issubset(set(schema["window_labels"]))
    assert set(ACTION_LABELS) == set(schema["action_labels"])

    summary_rows = _jsonl(out / GENERATION_SUMMARY_FILENAME)
    assert len(summary_rows) == 1
    summary = summary_rows[0]
    assert summary["run_id"] == "summary_only"
    assert summary["prediction_demand_enable"] == 1
    assert summary["prediction_demand_action_context_enable"] == 1
    assert int(summary["rows_emitted_total"]) > 0
    assert int(summary["exposure_count_total"]) > 0
    assert summary["reward_no_intervention_audit"]["direct_prediction_reward_active"] == 0
    assert summary["reward_no_intervention_audit"]["direct_error_reward_active"] == 0

    run_summary = _json(out / RUN_SUMMARY_FILENAME)
    assert run_summary["read_only"] is True
    assert run_summary["reward_no_intervention_audit"]["selection_signal_modified"] is False
    assert run_summary["reward_no_intervention_audit"]["reproduction_modified"] is False


def test_prediction_demand_action_context_disabled_prediction_controls_emit_zero_rows(tmp_path: Path) -> None:
    _run(
        tmp_path / "disabled_prediction",
        seed=123,
        prediction_enabled=False,
        action_context_enabled=True,
        run_id="disabled_prediction",
    )
    out = tmp_path / "disabled_prediction" / OUT_DIRNAME

    summary = _jsonl(out / GENERATION_SUMMARY_FILENAME)[0]
    assert summary["prediction_demand_enable"] == 0
    assert summary["rows_emitted_total"] == 0
    assert summary["exposure_count_total"] == 0
    for window in WINDOW_LABELS:
        assert summary["windows"][window]["row_count"] == 0
        assert summary["windows"][window]["exposure_count"] == 0


def test_prediction_demand_sign_summary_emits_compact_rows_without_full_events(tmp_path: Path) -> None:
    _run_sign_fixture(tmp_path / "sign", seed=123, write_events=False, run_id="sign_summary")
    out = tmp_path / "sign" / OUT_DIRNAME

    assert not (out / EVENTS_FILENAME).exists()
    assert (out / SIGN_SUMMARY_FILENAME).exists()
    assert (out / SIGN_WINDOW_SUMMARY_FILENAME).exists()
    assert (out / SIGN_GENERATION_SUMMARY_FILENAME).exists()

    sign_summary = _json(out / SIGN_SUMMARY_FILENAME)
    assert sign_summary["read_only"] is True
    assert sign_summary["default_off"] is True
    assert sign_summary["prediction_arm"] == "prediction_fixture"
    assert sign_summary["init_arm"] == "init_fixture"
    assert sign_summary["reward_no_intervention_audit"]["selection_signal_modified"] is False

    window_rows = _jsonl(out / SIGN_WINDOW_SUMMARY_FILENAME)
    assert window_rows
    signs = {str(row["trend_sign_label"]) for row in window_rows}
    windows = {str(row["window_label"]) for row in window_rows}
    assert {"boom", "bust"}.issubset(signs)
    assert "pre_consequence" in windows
    pre_rows = [row for row in window_rows if row["window_label"] == "pre_consequence"]
    assert pre_rows
    for row in pre_rows:
        assert "pre_consequence_action_entropy" in row
        assert set(row["action_label_counts"]) == set(ACTION_LABELS)
        assert "gather_in_cued_region_rate" in row
        assert "rest_rate" in row
        assert "movement_action_fraction" in row
    boom = [row for row in window_rows if row["trend_sign_label"] == "boom"]
    bust = [row for row in window_rows if row["trend_sign_label"] == "bust"]
    assert any(row["move_toward_before_boom_count"] >= 0 for row in boom)
    assert any(row["move_away_before_bust_count"] >= 0 for row in bust)


def test_prediction_demand_sign_summary_disabled_prediction_controls_emit_zero_rows(tmp_path: Path) -> None:
    _run_sign_fixture(
        tmp_path / "disabled_sign",
        seed=123,
        prediction_enabled=False,
        write_events=False,
        run_id="disabled_sign",
    )
    out = tmp_path / "disabled_sign" / OUT_DIRNAME

    assert (out / SIGN_SUMMARY_FILENAME).exists()
    assert (out / SIGN_WINDOW_SUMMARY_FILENAME).exists()
    sign_summary = _json(out / SIGN_SUMMARY_FILENAME)
    assert sign_summary["prediction_demand_enable"] == 0
    assert sign_summary["rows_emitted_total"] == 0
    assert sign_summary["exposure_count_total"] == 0
    assert _jsonl(out / SIGN_WINDOW_SUMMARY_FILENAME) == []
    generation_rows = _jsonl(out / SIGN_GENERATION_SUMMARY_FILENAME)
    assert len(generation_rows) == 1
    assert generation_rows[0]["rows_emitted_total"] == 0


def test_prediction_demand_action_context_event_rows_are_stable_and_deterministic(tmp_path: Path) -> None:
    _run(tmp_path / "a", seed=789, action_context_enabled=True, write_events=True, run_id="stable")
    _run(tmp_path / "b", seed=789, action_context_enabled=True, write_events=True, run_id="stable")

    events_a = (tmp_path / "a" / OUT_DIRNAME / EVENTS_FILENAME).read_bytes()
    events_b = (tmp_path / "b" / OUT_DIRNAME / EVENTS_FILENAME).read_bytes()
    assert hashlib.sha256(events_a).hexdigest() == hashlib.sha256(events_b).hexdigest()

    rows = _jsonl(tmp_path / "a" / OUT_DIRNAME / EVENTS_FILENAME)
    assert rows
    required = set(prediction_demand_action_context_schema()["required_event_fields"])
    windows = {str(row["window_label"]) for row in rows}
    assert {"cue", "waiting", "pre_consequence", "active_consequence"}.issubset(windows)
    window_ids = [str(row["window_id"]) for row in rows]
    assert window_ids == sorted(window_ids)
    for row in rows:
        assert required.issubset(row)
        assert set(row["action_label_counts"]) == set(ACTION_LABELS)
        assert row["reward_no_intervention_audit"]["direct_prediction_reward_active"] == 0
        assert row["reward_no_intervention_audit"]["direct_error_reward_active"] == 0


def test_prediction_demand_action_context_sampling_limits_events_and_preserves_state(tmp_path: Path) -> None:
    full = _run(
        tmp_path / "full",
        seed=852,
        action_context_enabled=True,
        write_events=True,
        run_id="sampled",
    )
    sampled = _run(
        tmp_path / "sampled",
        seed=852,
        action_context_enabled=True,
        write_events=True,
        run_id="sampled",
        sample_tick_period=2,
        sample_first_n_generations=1,
    )

    assert _fingerprint(full) == _fingerprint(sampled)

    full_rows = _jsonl(tmp_path / "full" / OUT_DIRNAME / EVENTS_FILENAME)
    sampled_rows = _jsonl(tmp_path / "sampled" / OUT_DIRNAME / EVENTS_FILENAME)
    assert 0 < len(sampled_rows) < len(full_rows)

    run_summary = _json(tmp_path / "sampled" / OUT_DIRNAME / RUN_SUMMARY_FILENAME)
    assert run_summary["diagnostic_sampling"]["sample_tick_period"] == 2
    assert run_summary["diagnostic_sampling"]["sample_first_n_generations"] == 1


def test_prediction_demand_sign_summary_is_stable_and_preserves_state(tmp_path: Path) -> None:
    reporting_off = SimulationRunner(
        env_cfg=_sign_env_cfg(prediction_enabled=True),
        pop_cfg=PopConfig(population_size=4),
        sim_cfg=_sim_cfg(246, action_context_enabled=False, run_id="off", steps=32),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path / "off"),
        perf=False,
    )
    reporting_on_a = _run_sign_fixture(tmp_path / "on_a", seed=246, write_events=False, run_id="sign_stable")
    reporting_on_b = _run_sign_fixture(tmp_path / "on_b", seed=246, write_events=False, run_id="sign_stable")
    reporting_off.run()

    assert _fingerprint(reporting_off) == _fingerprint(reporting_on_a)

    files = [SIGN_SUMMARY_FILENAME, SIGN_WINDOW_SUMMARY_FILENAME, SIGN_GENERATION_SUMMARY_FILENAME]
    for name in files:
        a = (tmp_path / "on_a" / OUT_DIRNAME / name).read_bytes()
        b = (tmp_path / "on_b" / OUT_DIRNAME / name).read_bytes()
        assert hashlib.sha256(a).hexdigest() == hashlib.sha256(b).hexdigest()


def test_prediction_demand_action_context_checkpoint_preserves_substrate_state(tmp_path: Path) -> None:
    sim = SimulationRunner(
        env_cfg=_env_cfg(prediction_enabled=True),
        pop_cfg=PopConfig(population_size=1),
        sim_cfg=_sim_cfg(987, action_context_enabled=True, run_id="checkpoint", steps=4),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path / "source"),
        perf=False,
    )
    sim.pop.init_random()
    sim._ensure_bank()
    sim._place_agents()
    sim.env.patch_world_step(0)
    assert (tmp_path / "source" / OUT_DIRNAME / SCHEMA_FILENAME).exists()

    bundle = tmp_path / "source" / "checkpoint_bundles_v2" / "bundle_gen000000"
    checkpoint_bundle_v2_save(sim, str(bundle), generation_index=0, strict_replay_mode=True)
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True, expected_commit_hash=None)

    restored = SimulationRunner(
        env_cfg=_env_cfg(prediction_enabled=True),
        pop_cfg=PopConfig(population_size=1),
        sim_cfg=_sim_cfg(987, action_context_enabled=True, run_id="checkpoint-restored", steps=4),
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path / "restored"),
        perf=False,
    )
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)

    assert np.array_equal(sim.env.patch_prediction_demand_timer, restored.env.patch_prediction_demand_timer)
    assert np.array_equal(sim.env.patch_prediction_demand_event_id, restored.env.patch_prediction_demand_event_id)
    assert np.array_equal(sim.env.patch_prediction_demand_cue, restored.env.patch_prediction_demand_cue)
    assert np.array_equal(sim.env.patch_prediction_demand_transition_sign, restored.env.patch_prediction_demand_transition_sign)
