from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.campaigns.artifact_lifecycle import (
    CHECKPOINT_TRANSIT,
    lifecycle_profile_contract_v1,
)
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics_pkg.lineage_reproduction_funnel import (
    CSV_FIELDS,
    EVENT_FIELDS,
    EVENT_SCHEMA_VERSION,
    SCAN_ACTION_EVENT_SCHEMA_VERSION,
    LineageReproductionFunnelWriter,
    _apply_v3_event_defaults,
    _normalize_event_row,
    _reset_lineage_reproduction_funnel_summary_caches_for_tests,
    _scan_action_rows,
    _stable_json,
    _stable_jsonl_payload,
    build_generation_rows,
    capture_pre_turnover_snapshot,
    classify_amplification,
    classify_run,
    validate_config,
    write_generation,
)
from simulation.metrics_pkg.metrics_logger_genome import parent_child_genome_distance_summary
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runner_pkg.runner_ticks_reproduction import (
    compute_lro1_reproductive_opportunity,
    parse_lro1_reproductive_opportunity_weights,
)


def _agent(
    agent_id: int,
    lineage_id: str,
    *,
    alive: bool = True,
    energy: float | None = None,
    health: float = 11.0,
    repro_cooldown: int = 0,
    reproduction_threshold: float = 40.0,
) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=int(agent_id),
        alive=bool(alive),
        energy=42.0 + float(agent_id) if energy is None else float(energy),
        health=float(health),
        repro_cooldown=int(repro_cooldown),
        genome=SimpleNamespace(reproduction_threshold=float(reproduction_threshold)),
        stats=SimpleNamespace(
            lineage_id=str(lineage_id),
            founder_id=int(agent_id),
            parent_ids=(),
            parent_lineage_ids=(),
            birth_generation=0,
            birth_tick=0,
            survival_steps=17 + int(agent_id),
            effective_fitness=1.5 + float(agent_id),
            offspring_count=0,
        ),
    )


def _enabled_cfg() -> SimConfig:
    cfg = SimConfig(seed=123)
    cfg.lineage_reproduction_funnel_reporting_enable = True
    return cfg


def test_parent_child_genome_distance_summary_reports_components() -> None:
    parent = _agent(1, "L00000001", reproduction_threshold=40.0)
    mate = _agent(2, "L00000001", reproduction_threshold=44.0)
    child = _agent(3, "L00000001", reproduction_threshold=50.0)
    parent.genome.mutation_rate = 0.02
    mate.genome.mutation_rate = 0.04
    child.genome.mutation_rate = 0.08
    parent.genome.memory_dim = 0
    mate.genome.memory_dim = 8
    child.genome.memory_dim = 16
    parent.genome.use_gru = False
    mate.genome.use_gru = True
    child.genome.use_gru = True
    parent.genome.pred_lr = 0.05
    mate.genome.pred_lr = 0.10
    child.genome.pred_lr = 0.20

    summary = parent_child_genome_distance_summary(parent, mate, child)

    assert 0.0 < summary["parent_child_genome_distance_mean"] <= summary["parent_child_genome_distance_max"] <= 1.0
    assert summary["parent_child_genome_memory_field_delta"] > 0.0
    assert summary["parent_child_genome_prediction_field_delta"] > 0.0
    assert "parent_child_genome_module_gate_delta" in summary


def _tiny_sim(tmp_path: Path, *, seed: int, reporting: bool, patch_world: bool = False) -> SimulationRunner:
    env_cfg = EnvConfig(width=8, height=8)
    env_cfg.patch_world_enable = bool(patch_world)
    env_cfg.patch_world_observation_enable = bool(patch_world)
    pop_cfg = PopConfig(population_size=8, parent_fraction=0.5, cull_fraction=0.25)
    sim_cfg = SimConfig(
        seed=int(seed),
        generations=2,
        steps_per_generation=2,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
    )
    sim_cfg.lineage_reproduction_funnel_reporting_enable = bool(reporting)
    return SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=pop_cfg,
        sim_cfg=sim_cfg,
        logger=NoOpMetricsLogger(),
        run_dir=str(tmp_path),
        perf=False,
    )


def _fingerprint(sim: SimulationRunner) -> list[tuple[int, str, bool]]:
    rows = []
    for agent in sorted(sim.pop.agents.values(), key=lambda item: int(item.agent_id)):
        rows.append((int(agent.agent_id), str(agent.stats.lineage_id), bool(agent.alive)))
    return rows


def test_lineage_reproduction_funnel_config_defaults_off() -> None:
    cfg = SimConfig()

    assert cfg.lineage_reproduction_funnel_reporting_enable is False
    assert cfg.lineage_reproduction_funnel_schema_version == 1
    assert cfg.lineage_reproduction_funnel_event_schema_version == 3
    assert cfg.lineage_reproduction_funnel_write_scan_action_events is False
    assert cfg.lineage_reproduction_funnel_event_output_profile == "full"
    assert cfg.lineage_reproduction_funnel_incremental_summary_cache_enable is False
    assert cfg.path2_roz_reporting_enable is False
    assert cfg.path2_roz_schema_version == 1
    assert cfg.path2_roz_zone_mode == "static_center_manhattan"
    assert cfg.path2_roz_window_period_steps == 40
    assert cfg.path2_roz_window_duration_steps == 8
    assert cfg.path2_roz_zone_radius == 0
    assert cfg.path2_roz_schedule_mode == "static_periodic_compat"
    assert cfg.path2_roz_schedule_id == ""
    assert cfg.path2_roz_schedule_phase_offset_steps == 0
    assert cfg.path2_roz_schedule_zones_json == ""
    assert cfg.lro1_reproductive_opportunity_enable is False
    assert cfg.lro1_reproductive_opportunity_mode == "local_patch_carrying_capacity_v1"
    assert cfg.lro1_reproductive_opportunity_apply_stage == "pre_attempt_try_prob"
    assert cfg.lro1_reproductive_opportunity_min_multiplier == 0.35
    assert cfg.lro1_reproductive_opportunity_max_multiplier == 1.0
    assert cfg.lro1_reproductive_opportunity_report_fields_enable is True
    assert cfg.lro4_offspring_dispersal_enable is False
    assert cfg.lro4_offspring_dispersal_mode == "crowding_conditioned_radius_v1"
    assert cfg.lro4_offspring_dispersal_base_radius == 1
    assert cfg.lro4_offspring_dispersal_max_radius == 3
    assert cfg.lro4_offspring_dispersal_density_threshold == 4.0
    assert cfg.lro4_offspring_dispersal_waste_threshold == 0.25
    assert cfg.lro4_offspring_dispersal_no_space_trigger_enable is True
    assert cfg.lro4_offspring_dispersal_deterministic_ring_order is True
    assert cfg.lro4_offspring_dispersal_report_fields_enable is True
    assert cfg.lineage_founder_spatial_separation_enable is False
    assert EVENT_SCHEMA_VERSION == 3
    assert SCAN_ACTION_EVENT_SCHEMA_VERSION == 4
    assert "lro4_offspring_dispersal_enable" in EVENT_FIELDS
    assert "lro4_offspring_dispersal_selected_from_expanded" in EVENT_FIELDS
    assert "lro4_offspring_dispersal_triggered_rows" in CSV_FIELDS
    assert "lro4_offspring_dispersal_disabled_leakage_rows" in CSV_FIELDS


def test_path2_roz_reporting_requires_scan_v4_events() -> None:
    cfg = _enabled_cfg()
    cfg.path2_roz_reporting_enable = True

    with pytest.raises(ValueError, match="path2_roz_reporting_enable requires"):
        validate_config(cfg)

    cfg.lineage_reproduction_funnel_write_events = True
    cfg.lineage_reproduction_funnel_event_schema_version = 4
    cfg.lineage_reproduction_funnel_write_scan_action_events = True
    validate_config(cfg)


def test_funnel_path2_roz_summary_only_event_profile_preserves_summary_without_event_file(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    cfg.lineage_reproduction_funnel_write_events = True
    cfg.lineage_reproduction_funnel_event_schema_version = 4
    cfg.lineage_reproduction_funnel_write_scan_action_events = True
    cfg.lineage_reproduction_funnel_event_output_profile = "path2_roz_summary_only_v1"
    cfg.lineage_reproduction_funnel_incremental_summary_cache_enable = True
    cfg.path2_roz_reporting_enable = True
    cfg.path2_roz_window_period_steps = 1
    cfg.path2_roz_window_duration_steps = 1
    cfg.path2_roz_zone_radius = 1
    validate_config(cfg)

    invalid = _enabled_cfg()
    invalid.lineage_reproduction_funnel_event_output_profile = "unknown"
    with pytest.raises(ValueError, match="lineage_reproduction_funnel_event_output_profile"):
        validate_config(invalid)
    invalid_summary_only = _enabled_cfg()
    invalid_summary_only.lineage_reproduction_funnel_event_output_profile = "path2_roz_summary_only_v1"
    with pytest.raises(ValueError, match="incremental_summary_cache_enable"):
        validate_config(invalid_summary_only)

    agents = [_agent(1, "L00000001")]
    common_roz = {
        "scan_id": "scan-1",
        "parent_id": 1,
        "parent_agent_id": 1,
        "parent_lineage_id": "L00000001",
        "parent_biological_lineage_id": "L00000001",
        "parent_energy": 43.0,
        "path2_roz_reporting_enable": 1,
        "path2_roz_schema_version": 1,
        "path2_roz_zone_mode": "static_center_manhattan",
        "path2_roz_window_active": 1,
        "path2_roz_window_id": "roz-1",
        "path2_roz_zone_id": "resource_exposed",
        "path2_roz_zone_count": 3,
        "path2_roz_parent_in_zone": 1,
        "path2_roz_selected_parent_scan_in_zone": 1,
        "parent_patch_food": "x" * 2000,
        "agent_patch_food": "y" * 2000,
    }
    birth = {
        **common_roz,
        "event": "successful_birth",
        "event_type": "birth_success",
        "child_id": 7,
        "child_agent_id": 7,
        "child_lineage_id": "L00000001",
        "birth_attempted": 1,
        "birth_success": 1,
        "path2_roz_child_in_zone": 1,
        "path2_roz_birth_in_zone": 1,
    }
    scan = {
        **common_roz,
        "event": "reproduction_scan_exposure",
        "event_type": "reproduction_scan_exposure",
    }
    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[1],
        selected_parent_agents=agents,
        successful_births=1,
        failed_births=0,
        birth_failure_counts={},
        birth_contexts=[birth],
        birth_event_contexts=[birth],
        scan_action_event_contexts=[scan],
        sim_cfg=cfg,
    )

    writer = LineageReproductionFunnelWriter(str(tmp_path), incremental_summary_cache_enable=True)
    writer.write_generation(
        generation=1,
        pre_turnover_snapshot=snapshot,
        post_refill_agents=agents,
        turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1}},
        write_events=True,
        event_schema_version=4,
        write_scan_action_events=True,
        event_output_profile="path2_roz_summary_only_v1",
    )

    events_path = tmp_path / "metrics" / "lineage_reproduction_funnel_events.jsonl"
    assert not events_path.exists()

    summary = json.loads((tmp_path / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(encoding="utf-8"))
    assert summary["path2_roz_event_rows"] > 0
    assert summary["path2_roz_scan_exposure_rows"] > 0
    assert summary["path2_roz_birth_success_rows"] > 0
    assert summary["path2_roz_parent_in_zone_rows"] > 0


def test_funnel_row_pipeline_golden_defaults_and_serialization_bytes() -> None:
    agent = _agent(7, "L00000007", energy=44.5)
    source = {
        "generation": 3,
        "tick": 9,
        "event": "successful_birth",
        "lineage_id": "L00000007",
        "parent_id": 7,
        "child_id": 17,
        "child_lineage_id": "L00000007",
        "birth_attempted": 1,
        "birth_success": 1,
        "candidate_free_spawn_cell_count": "2",
        "debug_nested_extra_ignored": {"z": [3, 2, 1]},
    }

    normalized = _normalize_event_row(
        source,
        generation=3,
        event_index=4,
        run_id="golden_run",
        selected_meta={
            7: {
                "parent_was_selected_parent_this_generation": 1,
                "parent_selected_rank": 2,
                "parent_selected_rank_bucket": "top3_selected_parent",
                "selected_parent_pool_size": 5,
            }
        },
        agent_by_id={7: agent},
        parent_fraction=0.5,
        event_schema_version=4,
    )

    assert set(normalized) == set(EVENT_FIELDS)
    assert normalized["schema_version"] == 4
    assert normalized["event_schema_version"] == 4
    assert normalized["event_id"] == "g000003:e00000004"
    assert normalized["event_type"] == "birth_success"
    assert normalized["parent_id"] == 7
    assert normalized["parent_energy"] == 44.5
    assert normalized["candidate_free_spawn_cell_count"] == 2
    assert "debug_nested_extra_ignored" not in normalized
    assert _stable_jsonl_payload([normalized]) == _stable_json(normalized) + "\n"

    explicit_none_window = {
        "event_type": "reproduction_scan_exposure",
        "accounting_window_rule": None,
        "exposure_phase": None,
        "parent_selected_rank": 1,
    }
    _apply_v3_event_defaults(
        explicit_none_window,
        generation=5,
        event_type="reproduction_scan_exposure",
        event_schema_version=4,
    )
    assert explicit_none_window["schema_version"] == 4
    assert explicit_none_window["accounting_window_start"] == 5
    assert explicit_none_window["accounting_window_end"] == 5
    assert explicit_none_window["accounting_window_rule"] is None
    assert explicit_none_window["exposure_phase"] is None
    assert explicit_none_window["selected_exposure_id"] == "g000005:selected:000001"


def test_funnel_row_pipeline_golden_scan_action_order_and_links() -> None:
    scan_a = "g000006:t000002:scan:00000007:000001"
    scan_b = "g000006:t000002:scan:00000008:000001"
    birth_rows = [
        {
            "event_id": f"{scan_b}:birth",
            "event_type": "birth_fail_non_space",
            "scan_id": scan_b,
            "parent_id": 8,
            "birth_attempted": 1,
            "birth_failed": 1,
        },
        {
            "event_id": f"{scan_a}:birth",
            "event_type": "birth_success",
            "scan_id": scan_a,
            "parent_id": 7,
            "birth_attempted": 1,
            "birth_success": 1,
        },
    ]
    scan_events = [
        "not a mapping",
        {
            "event_type": "ignored_event",
            "scan_id": "ignored",
        },
        {
            "event": "reproduction_scan_exposure",
            "event_type": "reproduction_scan_exposure",
            "event_id": f"{scan_a}:scan_exposure",
            "scan_id": scan_a,
            "parent_id": 7,
            "path2_roz_reporting_enable": 1,
            "path2_roz_parent_in_zone": 1,
        },
        {
            "event": "reproduction_scan_eligible",
            "event_type": "reproduction_scan_eligible",
            "event_id": f"{scan_a}:scan_eligible",
            "scan_id": scan_a,
            "parent_id": 7,
        },
    ]

    rows = _scan_action_rows(
        generation=6,
        scan_events=scan_events,
        birth_events=birth_rows,
        selected_meta={
            7: {
                "parent_was_selected_parent_this_generation": 1,
                "parent_selected_rank": 1,
                "parent_selected_rank_bucket": "top1_selected_parent",
                "selected_parent_pool_size": 2,
            }
        },
        event_schema_version=4,
    )

    assert [row["event_type"] for row in rows] == [
        "reproduction_scan_exposure",
        "reproduction_scan_eligible",
        "reproduction_scan_attempt_recorded",
        "reproduction_scan_attempt_recorded",
    ]
    assert [row["scan_id"] for row in rows[-2:]] == sorted([scan_a, scan_b])
    assert rows[0]["linked_success_event_id"] == f"{scan_a}:birth"
    assert rows[0]["path2_roz_selected_parent_scan_in_zone"] == 1
    assert rows[-1]["linked_failure_event_id"] == f"{scan_b}:birth"
    assert all(set(row) == set(EVENT_FIELDS) for row in rows)
    assert _stable_jsonl_payload(rows) == "".join(_stable_json(row) + "\n" for row in rows)


def test_path2_roz_schedule_config_fails_closed() -> None:
    cfg = _enabled_cfg()
    cfg.lineage_reproduction_funnel_write_events = True
    cfg.lineage_reproduction_funnel_event_schema_version = 4
    cfg.lineage_reproduction_funnel_write_scan_action_events = True
    cfg.path2_roz_reporting_enable = True

    cfg.path2_roz_schedule_mode = "patch_derived"
    with pytest.raises(ValueError, match="path2_roz_schedule_mode unsupported"):
        validate_config(cfg)

    cfg.path2_roz_schedule_mode = "multi_zone_round_robin"
    cfg.path2_roz_schedule_zones_json = "{bad"
    with pytest.raises(ValueError, match="path2_roz_schedule_zones_json must be valid JSON"):
        validate_config(cfg)

    cfg.path2_roz_schedule_zones_json = json.dumps([{"id": "bad", "center_y": 1, "center_x": 1, "radius": -1}])
    with pytest.raises(ValueError, match="radius must be non-negative"):
        validate_config(cfg)

    cfg.path2_roz_schedule_zones_json = json.dumps(
        [
            {"id": "z0", "center_y": 1, "center_x": 1, "radius": 1},
            {"id": "z1", "center_y": 2, "center_x": 2, "radius": 1},
        ]
    )
    validate_config(cfg)


def test_lro1_reproductive_opportunity_config_fails_closed() -> None:
    cfg = SimConfig()
    cfg.lro1_reproductive_opportunity_enable = True
    validate_config(cfg)

    cfg.lro1_reproductive_opportunity_mode = "lineage_quota"
    with pytest.raises(ValueError, match="mode unsupported"):
        validate_config(cfg)

    cfg.lro1_reproductive_opportunity_mode = "local_patch_carrying_capacity_v1"
    cfg.lro1_reproductive_opportunity_apply_stage = "post_birth_bonus"
    with pytest.raises(ValueError, match="apply_stage unsupported"):
        validate_config(cfg)

    cfg.lro1_reproductive_opportunity_apply_stage = "pre_attempt_try_prob"
    cfg.lro1_reproductive_opportunity_min_multiplier = 1.1
    with pytest.raises(ValueError, match="multipliers"):
        validate_config(cfg)

    cfg.lro1_reproductive_opportunity_min_multiplier = 0.35
    cfg.lro1_reproductive_opportunity_weights_json = "{bad"
    with pytest.raises(ValueError, match="must be valid JSON"):
        validate_config(cfg)

    cfg.lro1_reproductive_opportunity_weights_json = json.dumps({"lineage_id": 1.0})
    with pytest.raises(ValueError, match="unsupported LRO1"):
        validate_config(cfg)


def test_lro4_offspring_dispersal_config_fails_closed() -> None:
    cfg = SimConfig()
    cfg.lro4_offspring_dispersal_enable = True
    validate_config(cfg)

    cfg.lro4_offspring_dispersal_mode = "lineage_aware_radius"
    with pytest.raises(ValueError, match="mode unsupported"):
        validate_config(cfg)

    cfg.lro4_offspring_dispersal_mode = "crowding_conditioned_radius_v1"
    cfg.lro4_offspring_dispersal_base_radius = 0
    with pytest.raises(ValueError, match="radii"):
        validate_config(cfg)

    cfg.lro4_offspring_dispersal_base_radius = 2
    cfg.lro4_offspring_dispersal_max_radius = 1
    with pytest.raises(ValueError, match="radii"):
        validate_config(cfg)

    cfg.lro4_offspring_dispersal_base_radius = 1
    cfg.lro4_offspring_dispersal_max_radius = 3
    cfg.lro4_offspring_dispersal_density_threshold = -0.1
    with pytest.raises(ValueError, match="thresholds"):
        validate_config(cfg)

    cfg.lro4_offspring_dispersal_density_threshold = 4.0
    cfg.lro4_offspring_dispersal_deterministic_ring_order = False
    with pytest.raises(ValueError, match="deterministic_ring_order"):
        validate_config(cfg)


def test_lro1_opportunity_default_off_fields_do_not_leak() -> None:
    fields = compute_lro1_reproductive_opportunity(
        {
            "density": 9.0,
            "waste": 1.0,
            "scent": 1.0,
            "shelter_cover": 0.0,
            "food": 0.0,
            "hazard": 1.0,
        },
        enabled=False,
        base_try_prob=0.75,
    )

    assert fields["lro1_reproductive_opportunity_enable"] == 0
    assert fields["lro1_opportunity_score_parent_site"] == 0.0
    assert fields["lro1_opportunity_multiplier"] == 1.0
    assert fields["lro1_opportunity_effective_try_prob"] == 0.75


def test_lro1_opportunity_bounds_direction_and_neutral_context() -> None:
    weights = parse_lro1_reproductive_opportunity_weights("")
    neutral = compute_lro1_reproductive_opportunity({}, enabled=True, weights=weights, base_try_prob=0.8)
    crowded = compute_lro1_reproductive_opportunity(
        {"density": 9.0, "waste": 1.0, "scent": 1.0, "hazard": 1.0},
        enabled=True,
        weights=weights,
        base_try_prob=0.8,
    )
    buffered = compute_lro1_reproductive_opportunity(
        {"density": 3.0, "waste": 0.2, "scent": 0.1, "hazard": 0.0, "shelter_cover": 1.0, "food": 1.0},
        enabled=True,
        weights=weights,
        base_try_prob=0.8,
    )

    assert neutral["lro1_opportunity_neutral_context"] == 1
    assert neutral["lro1_opportunity_multiplier"] == 1.0
    assert 0.35 <= crowded["lro1_opportunity_multiplier"] < buffered["lro1_opportunity_multiplier"] <= 1.0
    assert crowded["lro1_opportunity_effective_try_prob"] < buffered["lro1_opportunity_effective_try_prob"] <= 0.8


def test_lro1_opportunity_repeatable_and_lineage_blind() -> None:
    context = {
        "density": 4.0,
        "waste": 0.4,
        "scent": 0.2,
        "shelter_cover": 0.7,
        "food": 0.5,
        "hazard": 0.1,
        "lineage_id": "L00000099",
    }
    a = compute_lro1_reproductive_opportunity(context, enabled=True, base_try_prob=0.5)
    context["lineage_id"] = "L00000001"
    b = compute_lro1_reproductive_opportunity(context, enabled=True, base_try_prob=0.5)
    c = compute_lro1_reproductive_opportunity(dict(context), enabled=True, base_try_prob=0.5)

    assert a == b == c


def test_lro1_opportunity_invalid_context_is_neutral_and_serializes() -> None:
    fields = compute_lro1_reproductive_opportunity(
        {"density": "bad", "waste": None},
        enabled=True,
        base_try_prob=0.5,
    )

    assert fields["lro1_opportunity_invalid_context"] == 1
    assert fields["lro1_opportunity_effective_try_prob"] == 0.5
    json.loads(json.dumps(asdict(SimConfig()), sort_keys=True))


def test_capture_snapshot_disabled_returns_empty() -> None:
    cfg = SimConfig()

    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=[_agent(1, "L00000001")],
        alive_after_agents=[_agent(1, "L00000001")],
        eligible_ids=[1],
        selected_parent_agents=[_agent(1, "L00000001")],
        successful_births=0,
        failed_births=0,
        birth_failure_counts={},
        birth_contexts=[],
        sim_cfg=cfg,
    )

    assert snapshot == {}


def test_generation_rows_include_schema_and_degrade_optional_context() -> None:
    cfg = _enabled_cfg()
    alive = [_agent(1, "L00000001"), _agent(2, "L00000001"), _agent(3, "L00000002")]
    snapshot = capture_pre_turnover_snapshot(
        generation=3,
        alive_start_agents=alive,
        alive_after_agents=alive,
        eligible_ids=[1, 2, 3],
        selected_parent_agents=[alive[0], alive[1]],
        successful_births=1,
        failed_births=2,
        birth_failure_counts={"birth_fail_no_space": 2},
        birth_contexts=[{"lineage_id": "L00000001", "child_y": 2, "child_x": 3, "birth_attempt_count": 1}],
        sim_cfg=cfg,
    )

    rows = build_generation_rows(
        generation=3,
        pre_turnover_snapshot=snapshot,
        post_refill_agents=[_agent(1, "L00000001"), _agent(4, "L00000001"), _agent(5, "L00000002")],
        turnover_snapshot={
            "pre_cull_alive_counts": {"L00000001": 2, "L00000002": 1},
            "post_cull_alive_counts": {"L00000001": 1, "L00000002": 1},
            "culled_counts": {"L00000001": 1},
            "refill_offspring_counts": {"L00000001": 1},
        },
    )

    assert {row["lineage_id"] for row in rows} == {"L00000001", "L00000002"}
    top = [row for row in rows if row["lineage_id"] == "L00000001"][0]
    assert set(CSV_FIELDS) <= set(top)
    assert top["successful_births"] == 1
    assert top["child_patch_id_or_coordinates"] == "2,3"
    assert top["child_basin_id"] == ""
    assert top["birth_failure_due_to_no_local_space"] == 2


def test_amplification_classifier_and_run_summary() -> None:
    assert classify_amplification(None) == "denominator unavailable"
    assert classify_amplification(1.0) == "no amplification"
    assert classify_amplification(1.2) == "mild amplification"
    assert classify_amplification(1.6) == "strong amplification"
    assert classify_amplification(2.1) == "collapse-stage amplification"

    rows = [
        {
            "generation": 1,
            "lineage_id": "L00000001",
            "post_refill_alive_share": 0.9,
            "selected_parent_amplification_class": "strong amplification",
            "refill_amplification_class": "no amplification",
        }
    ]
    summary = classify_run(rows)

    assert summary["collapse_stage_classification"] == "selected-parent compression"
    assert summary["decision"].startswith("Decision 2")


def test_writer_output_order_is_repeatable(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    agents = [_agent(1, "L00000002"), _agent(2, "L00000001")]
    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[1, 2],
        selected_parent_agents=agents,
        successful_births=0,
        failed_births=0,
        birth_failure_counts={},
        birth_contexts=[],
        sim_cfg=cfg,
    )

    for root in (tmp_path / "a", tmp_path / "b"):
        writer = LineageReproductionFunnelWriter(str(root))
        writer.write_generation(
            generation=1,
            pre_turnover_snapshot=snapshot,
            post_refill_agents=agents,
            turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1, "L00000002": 1}},
            write_events=False,
        )

    assert (tmp_path / "a" / "metrics" / "lineage_reproduction_funnel_generation_summary.csv").read_text(
        encoding="utf-8"
    ) == (tmp_path / "b" / "metrics" / "lineage_reproduction_funnel_generation_summary.csv").read_text(
        encoding="utf-8"
    )
    with (tmp_path / "a" / "metrics" / "lineage_reproduction_funnel_generation_summary.csv").open(
        "r", newline="", encoding="utf-8"
    ) as f:
        rows = list(csv.DictReader(f))
    assert [row["lineage_id"] for row in rows] == ["L00000001", "L00000002"]


def test_lifecycle_bound_funnel_event_writer_fails_closed_when_trace_forbidden(
    tmp_path: Path,
) -> None:
    contract = lifecycle_profile_contract_v1(
        CHECKPOINT_TRANSIT,
        total_max_bytes=10_000,
        checkpoint_schedule=[1],
    )
    (tmp_path / "artifact_lifecycle_contract.json").write_text(
        json.dumps(contract, sort_keys=True), encoding="utf-8"
    )
    writer = LineageReproductionFunnelWriter(str(tmp_path))

    with pytest.raises(RuntimeError, match="forbids"):
        writer._check_lifecycle_event_write('{"event":"diagnostic"}\n')

    violation = json.loads(
        (tmp_path / "lifecycle_storage_violation_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert violation["code"] == "undeclared_event_trace"
    assert violation["artifact_family"] == "lineage_reproduction_funnel.events"
    assert violation["owner_writer"] == "LineageReproductionFunnelWriter"
    assert violation["automatic_prune_allowed"] is False


def _cache_parity_snapshot(generation: int, agents: list[SimpleNamespace]) -> dict:
    cfg = _enabled_cfg()
    event = {
        "schema_version": 1,
        "generation": int(generation),
        "tick": 3 + int(generation),
        "event": "failed_birth" if generation == 1 else "successful_birth",
        "failure_reason": "birth_fail_no_space" if generation == 1 else "",
        "lineage_id": "L00000001",
        "child_lineage_id": "L00000001",
        "parent_lineage_id": "L00000001",
        "parent_id": 1,
        "child_id": -1 if generation == 1 else 9 + int(generation),
        "child_y": 2,
        "child_x": 3 + int(generation),
        "birth_attempt_count": 1,
        "local_free_neighbor_count": 0 if generation == 1 else 2,
        "candidate_free_spawn_cell_count": 0 if generation == 1 else 2,
        "failed_birth_attempt_count": 1 if generation == 1 else 0,
        "no_local_space": 1 if generation == 1 else 0,
        "path2_roz_reporting_enable": 1,
        "path2_roz_window_active": 1,
        "path2_roz_zone_id": f"z{generation}",
        "path2_roz_parent_in_zone": 1,
    }
    return capture_pre_turnover_snapshot(
        generation=int(generation),
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[1],
        selected_parent_agents=agents,
        successful_births=0 if generation == 1 else 1,
        failed_births=1 if generation == 1 else 0,
        birth_failure_counts={"birth_fail_no_space": 1} if generation == 1 else {},
        birth_contexts=[] if generation == 1 else [event],
        sim_cfg=cfg,
        birth_event_contexts=[event],
    )


def test_incremental_summary_cache_preserves_writer_outputs(tmp_path: Path) -> None:
    _reset_lineage_reproduction_funnel_summary_caches_for_tests()
    agents = [_agent(1, "L00000001")]

    def run_case(root: Path, *, cache: bool) -> None:
        cfg = _enabled_cfg()
        cfg.lineage_reproduction_funnel_write_events = True
        cfg.lineage_reproduction_funnel_incremental_summary_cache_enable = bool(cache)
        for generation in (1, 2):
            write_generation(
                run_dir=str(root),
                sim_cfg=cfg,
                generation=generation,
                pre_turnover_snapshot=_cache_parity_snapshot(generation, agents),
                post_refill_agents=agents,
                turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1}},
            )

    off_root = tmp_path / "off" / "stable"
    on_root = tmp_path / "on" / "stable"
    run_case(off_root, cache=False)
    run_case(on_root, cache=True)

    for filename in (
        "lineage_reproduction_funnel_generation_summary.csv",
        "lineage_reproduction_funnel_generation_summary.jsonl",
        "lineage_reproduction_funnel_events.jsonl",
        "lineage_reproduction_funnel_run_summary.json",
    ):
        assert (on_root / "metrics" / filename).read_bytes() == (off_root / "metrics" / filename).read_bytes()

    sidecar = json.loads(
        (on_root / "metrics" / "lineage_reproduction_funnel_summary_cache.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["enabled"] is True
    assert sidecar["diagnostics"]["events_applied"] > 0
    assert sidecar["diagnostics"]["hits"] > 0
    assert sidecar["diagnostics"]["json_decodes_avoided"] > 0
    assert sidecar["diagnostics"]["full_scan_fallbacks"] == 0
    assert sidecar["diagnostics"]["event_writer_rows"] > 0
    assert sidecar["diagnostics"]["event_writer_batches"] == 2
    assert sidecar["diagnostics"]["event_writer_writes"] == sidecar["diagnostics"]["event_writer_batches"]
    assert sidecar["event_writer_diagnostics"]["event_writer_rows"] > 0
    assert sidecar["event_writer_diagnostics"]["event_writer_writes"] == 1


def test_incremental_summary_cache_matches_full_scan_with_malformed_lines(tmp_path: Path) -> None:
    _reset_lineage_reproduction_funnel_summary_caches_for_tests()
    valid_rows = [
        {
            "event_schema_version": 4,
            "event_type": "reproduction_scan_exposure",
            "path2_roz_reporting_enable": 1,
            "path2_roz_window_active": 1,
            "path2_roz_zone_id": "z0",
        },
        {
            "event_schema_version": 4,
            "event_type": "reproduction_action_taken",
            "failure_reason_group": "non_space",
            "path2_roz_reporting_enable": 1,
            "path2_roz_window_active": 1,
            "path2_roz_zone_id": "z1",
        },
    ]
    seeded_text = (
        json.dumps(valid_rows[0], sort_keys=True)
        + "\n\n{bad json\n"
        + json.dumps(valid_rows[1], sort_keys=True)
        + "\n"
    )

    for name, cache in (("off", False), ("on", True)):
        writer = LineageReproductionFunnelWriter(
            str(tmp_path / name),
            incremental_summary_cache_enable=cache,
        )
        Path(writer.events_path).write_text(seeded_text, encoding="utf-8")
        writer._write_run_summary()

    assert (tmp_path / "on" / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_bytes() == (
        tmp_path / "off" / "metrics" / "lineage_reproduction_funnel_run_summary.json"
    ).read_bytes()
    sidecar = json.loads(
        (tmp_path / "on" / "metrics" / "lineage_reproduction_funnel_summary_cache.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["diagnostics"]["blank_lines"] == 1
    assert sidecar["diagnostics"]["malformed_lines"] == 1


def test_incremental_summary_cache_falls_back_after_external_append(tmp_path: Path) -> None:
    _reset_lineage_reproduction_funnel_summary_caches_for_tests()
    agents = [_agent(1, "L00000001")]
    cfg = _enabled_cfg()
    cfg.lineage_reproduction_funnel_write_events = True
    cfg.lineage_reproduction_funnel_incremental_summary_cache_enable = True
    write_generation(
        run_dir=str(tmp_path),
        sim_cfg=cfg,
        generation=1,
        pre_turnover_snapshot=_cache_parity_snapshot(1, agents),
        post_refill_agents=agents,
        turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1}},
    )

    events_path = tmp_path / "metrics" / "lineage_reproduction_funnel_events.jsonl"
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event_schema_version": 4, "event_type": "reproduction_action_taken"}) + "\n")

    writer = LineageReproductionFunnelWriter(str(tmp_path), incremental_summary_cache_enable=True)
    writer._write_run_summary()

    summary = json.loads(
        (tmp_path / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(encoding="utf-8")
    )
    sidecar = json.loads(
        (tmp_path / "metrics" / "lineage_reproduction_funnel_summary_cache.json").read_text(encoding="utf-8")
    )
    assert summary["reproduction_action_taken_rows"] == 1
    assert sidecar["diagnostics"]["invalidations"] >= 1


def test_incremental_summary_cache_merges_schedule_and_weather_accumulators(tmp_path: Path) -> None:
    _reset_lineage_reproduction_funnel_summary_caches_for_tests()
    writer = LineageReproductionFunnelWriter(str(tmp_path), incremental_summary_cache_enable=True)
    batches = [
        [
            {
                "event_schema_version": 4,
                "event_type": "birth_success",
                "generation": 1,
                "path2_roz_reporting_enable": 1,
                "path2_roz_window_active": 1,
                "path2_roz_zone_id": "z0",
                "child_mutation_weather_active": 1,
                "mutation_weather_generation": 1,
                "child_mutation_weather_event_id": "weather-a",
                "parent_child_genome_distance_mean": 2.0,
                "parent_child_genome_distance_max": 3.0,
            }
        ],
        [
            {
                "event_schema_version": 4,
                "event_type": "birth_success",
                "generation": 2,
                "path2_roz_reporting_enable": 1,
                "path2_roz_window_active": 1,
                "path2_roz_zone_id": "z1",
                "child_mutation_weather_active": 1,
                "mutation_weather_generation": 1,
                "child_mutation_weather_event_id": "weather-b",
                "parent_child_genome_distance_mean": 4.0,
                "parent_child_genome_distance_max": 5.0,
            }
        ],
    ]
    for rows in batches:
        ready = writer._prepare_event_summary_cache_for_append()
        with open(writer.events_path, "a", encoding="utf-8") as f:
            f.write(_stable_jsonl_payload(rows))
        writer._apply_event_summary_cache_events(rows, cache_ready_for_append=ready)

    cached = writer._cached_event_summary()
    full_scan = writer._event_summary()
    sidecar = json.loads(Path(writer.summary_cache_path).read_text(encoding="utf-8"))

    assert cached == full_scan
    assert cached["path2_roz_schedule_zone_switch_count"] == 1
    assert cached["path2_roz_schedule_distinct_zone_count"] == 2
    assert cached["weather_active_generation_count"] == 1
    assert cached["weather_event_count"] == 2
    assert cached["parent_child_genome_distance_mean_weather_on"] == pytest.approx(3.0)
    assert cached["parent_child_genome_distance_max_weather_on"] == pytest.approx(5.0)
    assert sidecar["diagnostics"]["full_scan_fallbacks"] == 0


def test_writer_emits_failed_no_space_event_context(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    agents = [_agent(1, "L00000001")]
    event = {
        "schema_version": 1,
        "generation": 1,
        "tick": 3,
        "event": "failed_birth",
        "failure_reason": "birth_fail_no_space",
        "lineage_id": "L00000001",
        "parent_lineage_id": "L00000001",
        "parent_id": 1,
        "child_id": -1,
        "local_free_neighbor_count": 0,
        "candidate_free_spawn_cell_count": 0,
        "failed_birth_attempt_count": 1,
        "no_local_space": 1,
    }
    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[1],
        selected_parent_agents=agents,
        successful_births=0,
        failed_births=1,
        birth_failure_counts={"birth_fail_no_space": 1},
        birth_contexts=[],
        sim_cfg=cfg,
        birth_event_contexts=[event],
    )

    writer = LineageReproductionFunnelWriter(str(tmp_path))
    writer.write_generation(
        generation=1,
        pre_turnover_snapshot=snapshot,
        post_refill_agents=agents,
        turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1}},
        write_events=True,
    )

    events_path = tmp_path / "metrics" / "lineage_reproduction_funnel_events.jsonl"
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    selected_rows = [row for row in rows if row["event_type"] == "selected_parent_exposure"]
    fail_rows = [row for row in rows if row["event_type"] == "birth_fail_no_space"]
    assert len(selected_rows) == 1
    assert len(fail_rows) == 1
    attempt_rows = [row for row in rows if row["event_type"] == "birth_attempt_recorded"]
    assert len(attempt_rows) == 1
    assert fail_rows[0]["schema_version"] == 3
    assert fail_rows[0]["event_schema_version"] == 3
    assert fail_rows[0]["parent_biological_lineage_id"] == "L00000001"
    assert fail_rows[0]["failure_reason_group"] == "no_space"
    assert fail_rows[0]["no_local_space_marker"] == 1
    assert fail_rows[0]["no_space_marker"] == 1
    assert fail_rows[0]["candidate_free_spawn_cells"] == 0
    assert fail_rows[0]["parent_energy"] == 43.0
    assert fail_rows[0]["parent_selected_rank"] == 1
    assert fail_rows[0]["selected_exposure_id"] == selected_rows[0]["selected_exposure_id"]
    assert attempt_rows[0]["linked_attempt_event_id"] == fail_rows[0]["event_id"]


def test_writer_emits_success_event_with_parent_condition(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    agents = [_agent(1, "L00000001")]
    event = {
        "schema_version": 1,
        "generation": 1,
        "tick": 4,
        "event": "successful_birth",
        "lineage_id": "L00000001",
        "child_lineage_id": "L00000001",
        "parent_lineage_id": "L00000001",
        "mate_lineage_id": "",
        "parent_id": 1,
        "mate_id": -1,
        "child_id": 9,
        "parent_y": 2,
        "parent_x": 3,
        "child_y": 2,
        "child_x": 4,
        "birth_attempt_count": 1,
        "parent_child_genome_distance_mean": 0.125,
        "parent_child_genome_distance_max": 0.25,
        "parent_child_genome_scalar_distance_mean": 0.10,
        "parent_child_genome_memory_field_delta": 0.20,
        "parent_child_genome_prediction_field_delta": 0.30,
        "parent_child_genome_module_gate_delta": 0.40,
    }
    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[1],
        selected_parent_agents=agents,
        successful_births=1,
        failed_births=0,
        birth_failure_counts={},
        birth_contexts=[event],
        sim_cfg=cfg,
        birth_event_contexts=[event],
        parent_fraction=0.5,
    )

    writer = LineageReproductionFunnelWriter(str(tmp_path))
    writer.write_generation(
        generation=1,
        pre_turnover_snapshot=snapshot,
        post_refill_agents=agents,
        turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1}},
        write_events=True,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "lineage_reproduction_funnel_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    success = [row for row in rows if row["event_type"] == "birth_success"][0]
    attempt = [row for row in rows if row["event_type"] == "birth_attempt_recorded"][0]
    assert success["parent_biological_lineage_id"] == "L00000001"
    assert success["child_biological_lineage_id"] == "L00000001"
    assert success["birth_attempted"] == 1
    assert success["birth_success"] == 1
    assert attempt["linked_success_event_id"] == success["event_id"]
    assert success["parent_energy"] == 43.0
    assert success["reproduction_threshold"] == 40.0
    assert success["energy_threshold_margin"] == 3.0
    assert success["parent_health"] == 11.0
    assert success["parent_age"] == 18
    assert success["parent_fraction"] == 0.5
    assert success["parent_child_genome_distance_mean"] == 0.125
    assert success["parent_child_genome_distance_max"] == 0.25
    assert success["parent_child_genome_memory_field_delta"] == 0.20


def test_writer_emits_non_space_failure_event_context(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    agents = [_agent(2, "L00000002")]
    event = {
        "schema_version": 2,
        "event_schema_version": 2,
        "generation": 1,
        "tick": 5,
        "event": "failed_birth",
        "event_type": "birth_fail_non_space",
        "failure_reason": "birth_fail_no_mate",
        "lineage_id": "L00000002",
        "parent_lineage_id": "L00000002",
        "parent_id": 2,
        "mate_id": -1,
        "child_id": -1,
        "birth_attempted": 0,
        "birth_failed": 1,
    }
    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[2],
        selected_parent_agents=agents,
        successful_births=0,
        failed_births=1,
        birth_failure_counts={"birth_fail_no_mate": 1},
        birth_contexts=[],
        sim_cfg=cfg,
        birth_event_contexts=[event],
    )

    writer = LineageReproductionFunnelWriter(str(tmp_path))
    writer.write_generation(
        generation=1,
        pre_turnover_snapshot=snapshot,
        post_refill_agents=agents,
        turnover_snapshot={"pre_cull_alive_counts": {"L00000002": 1}},
        write_events=True,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "lineage_reproduction_funnel_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    failure = [row for row in rows if row["event_type"] == "birth_fail_non_space"][0]
    no_attempt = [row for row in rows if row["event_type"] == "selected_parent_no_attempt"][0]
    assert failure["failure_reason"] == "birth_fail_no_mate"
    assert failure["failure_reason_group"] == "mate"
    assert failure["birth_attempted"] == 0
    assert failure["birth_failed"] == 1
    assert failure["parent_biological_lineage_id"] == "L00000002"
    assert no_attempt["no_attempt_marker"] == 1


def test_writer_emits_exposure_eligibility_and_no_attempt_rows(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    eligible = _agent(1, "L00000001")
    ineligible = _agent(2, "L00000002", repro_cooldown=4)
    agents = [eligible, ineligible]
    event = {
        "schema_version": 2,
        "event_schema_version": 2,
        "generation": 1,
        "tick": 6,
        "event": "successful_birth",
        "event_type": "birth_success",
        "lineage_id": "L00000001",
        "child_lineage_id": "L00000001",
        "parent_lineage_id": "L00000001",
        "parent_id": 1,
        "child_id": 11,
        "birth_attempted": 1,
        "birth_success": 1,
    }
    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[1],
        selected_parent_agents=agents,
        successful_births=1,
        failed_births=0,
        birth_failure_counts={},
        birth_contexts=[event],
        sim_cfg=cfg,
        birth_event_contexts=[event],
    )

    writer = LineageReproductionFunnelWriter(str(tmp_path))
    writer.write_generation(
        generation=1,
        pre_turnover_snapshot=snapshot,
        post_refill_agents=agents,
        turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1, "L00000002": 1}},
        write_events=True,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "lineage_reproduction_funnel_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    types = [row["event_type"] for row in rows]
    assert types.count("selected_parent_exposure") == 2
    assert types.count("selected_parent_eligible") == 1
    assert types.count("selected_parent_ineligible") == 1
    assert types.count("selected_parent_no_attempt") == 1
    assert types.count("birth_attempt_recorded") == 1
    ineligible_row = [row for row in rows if row["event_type"] == "selected_parent_ineligible"][0]
    assert ineligible_row["parent_id"] == 2
    assert ineligible_row["parent_reproduction_eligible"] == 0
    assert ineligible_row["parent_ineligible_reason_group"] == "cooldown_age"
    assert ineligible_row["cooldown_blocked"] == 1
    no_attempt = [row for row in rows if row["event_type"] == "selected_parent_no_attempt"][0]
    assert no_attempt["parent_id"] == 2
    assert no_attempt["selected_exposure_id"] == ineligible_row["selected_exposure_id"]
    summary = json.loads((tmp_path / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(encoding="utf-8"))
    assert summary["event_schema_version"] == 3
    assert summary["selected_parent_exposure_rows"] == 2
    assert summary["selected_parent_eligible_rows"] == 1
    assert summary["selected_parent_ineligible_rows"] == 1
    assert summary["selected_parent_no_attempt_rows"] == 1
    assert summary["birth_attempt_recorded_rows"] == 1
    assert summary["rows_with_accounting_window"] == len(rows)


def test_reporting_enabled_tiny_runner_preserves_final_population_and_writes_outputs(tmp_path: Path) -> None:
    off = _tiny_sim(tmp_path / "off", seed=707, reporting=False)
    off.run(start_gen=0, skip_init=False)
    on = _tiny_sim(tmp_path / "on", seed=707, reporting=True)
    on.run(start_gen=0, skip_init=False)

    assert _fingerprint(on) == _fingerprint(off)
    metrics_dir = tmp_path / "on" / "metrics"
    assert (metrics_dir / "lineage_reproduction_funnel_generation_summary.csv").exists()
    assert (metrics_dir / "lineage_reproduction_funnel_generation_summary.jsonl").exists()
    assert (metrics_dir / "lineage_reproduction_funnel_run_summary.json").exists()
    assert (metrics_dir / "lineage_reproduction_funnel_schema.json").exists()
    assert not (tmp_path / "off" / "metrics" / "lineage_reproduction_funnel_generation_summary.csv").exists()
    assert not (tmp_path / "off" / "metrics" / "lineage_reproduction_funnel_events.jsonl").exists()

    summary = json.loads((metrics_dir / "lineage_reproduction_funnel_run_summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] > 0


def test_lro1_flags_off_tiny_runner_preserves_final_population(tmp_path: Path) -> None:
    baseline = _tiny_sim(tmp_path / "baseline", seed=717, reporting=False)
    baseline.run(start_gen=0, skip_init=False)

    lro_off = _tiny_sim(tmp_path / "lro_off", seed=717, reporting=False)
    lro_off.sim_cfg.lro1_reproductive_opportunity_enable = False
    lro_off.sim_cfg.lro1_reproductive_opportunity_report_fields_enable = True
    lro_off.sim_cfg.lro1_reproductive_opportunity_weights_json = "{disabled-invalid-json-is-ignored"
    lro_off.run(start_gen=0, skip_init=False)

    assert _fingerprint(lro_off) == _fingerprint(baseline)


def test_writer_event_output_order_is_repeatable(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    agents = [_agent(1, "L00000001"), _agent(2, "L00000002")]
    event = {
        "schema_version": 2,
        "event_schema_version": 2,
        "generation": 1,
        "tick": 7,
        "event": "failed_birth",
        "event_type": "birth_fail_non_space",
        "failure_reason": "birth_fail_conception",
        "lineage_id": "L00000001",
        "parent_lineage_id": "L00000001",
        "parent_id": 1,
        "mate_id": 2,
        "mate_lineage_id": "L00000002",
        "child_id": -1,
    }
    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[1, 2],
        selected_parent_agents=agents,
        successful_births=0,
        failed_births=1,
        birth_failure_counts={"birth_fail_conception": 1},
        birth_contexts=[],
        sim_cfg=cfg,
        birth_event_contexts=[event],
        parent_fraction=0.5,
    )

    for root in (tmp_path / "a" / "stable", tmp_path / "b" / "stable"):
        writer = LineageReproductionFunnelWriter(str(root))
        writer.write_generation(
            generation=1,
            pre_turnover_snapshot=snapshot,
            post_refill_agents=agents,
            turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1, "L00000002": 1}},
            write_events=True,
        )

    a_events = (tmp_path / "a" / "stable" / "metrics" / "lineage_reproduction_funnel_events.jsonl").read_text(
        encoding="utf-8"
    )
    b_events = (tmp_path / "b" / "stable" / "metrics" / "lineage_reproduction_funnel_events.jsonl").read_text(
        encoding="utf-8"
    )
    assert a_events == b_events


def test_writer_incremental_summary_cache_preserves_public_outputs(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    agents = [_agent(1, "L00000001"), _agent(2, "L00000002")]

    def _snapshot(generation: int) -> dict[str, object]:
        event = {
            "schema_version": 2,
            "event_schema_version": 2,
            "generation": generation,
            "tick": generation + 3,
            "event": "failed_birth",
            "event_type": "birth_fail_non_space",
            "failure_reason": "birth_fail_conception",
            "lineage_id": "L00000001",
            "parent_lineage_id": "L00000001",
            "parent_id": 1,
            "mate_id": 2,
            "mate_lineage_id": "L00000002",
            "child_id": -1,
        }
        return capture_pre_turnover_snapshot(
            generation=generation,
            alive_start_agents=agents,
            alive_after_agents=agents,
            eligible_ids=[1, 2],
            selected_parent_agents=agents,
            successful_births=0,
            failed_births=1,
            birth_failure_counts={"birth_fail_conception": 1},
            birth_contexts=[],
            sim_cfg=cfg,
            birth_event_contexts=[event],
            parent_fraction=0.5,
        )

    for root, cache in ((tmp_path / "baseline" / "run", False), (tmp_path / "cached" / "run", True)):
        writer = LineageReproductionFunnelWriter(str(root), incremental_summary_cache_enable=cache)
        for generation in (1, 2):
            writer.write_generation(
                generation=generation,
                pre_turnover_snapshot=_snapshot(generation),
                post_refill_agents=agents,
                turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1, "L00000002": 1}},
                write_events=True,
            )

    baseline_metrics = tmp_path / "baseline" / "run" / "metrics"
    cached_metrics = tmp_path / "cached" / "run" / "metrics"
    for name in (
        "lineage_reproduction_funnel_generation_summary.csv",
        "lineage_reproduction_funnel_generation_summary.jsonl",
        "lineage_reproduction_funnel_events.jsonl",
        "lineage_reproduction_funnel_run_summary.json",
    ):
        assert (cached_metrics / name).read_bytes() == (baseline_metrics / name).read_bytes()

    sidecar = json.loads((cached_metrics / "lineage_reproduction_funnel_summary_cache.json").read_text(encoding="utf-8"))
    assert sidecar["diagnostics"]["summary_hits"] >= 1
    assert sidecar["diagnostics"]["json_decodes_avoided"] > 0
    run_summary = json.loads((baseline_metrics / "lineage_reproduction_funnel_run_summary.json").read_text(encoding="utf-8"))
    for key, value in sidecar["summary"].items():
        assert run_summary[key] == value


def test_writer_emits_default_off_v4_scan_action_rows(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    cfg.lineage_reproduction_funnel_write_events = True
    cfg.lineage_reproduction_funnel_event_schema_version = 4
    cfg.lineage_reproduction_funnel_write_scan_action_events = True
    agents = [_agent(1, "L00000001")]
    scan_id = "g000001:t000002:scan:00000001:000001"
    birth_event = {
        "schema_version": 2,
        "event_schema_version": 2,
        "generation": 1,
        "tick": 2,
        "event": "successful_birth",
        "event_type": "birth_success",
        "scan_id": scan_id,
        "lineage_id": "L00000001",
        "child_lineage_id": "L00000001",
        "parent_lineage_id": "L00000001",
        "parent_id": 1,
        "child_id": 12,
        "birth_attempted": 1,
        "birth_success": 1,
    }
    scan_contexts = [
        {
            "schema_version": 4,
            "event_schema_version": 4,
            "generation": 1,
            "tick": 2,
            "event": "reproduction_scan_exposure",
            "event_type": "reproduction_scan_exposure",
            "event_id": f"{scan_id}:scan_exposure",
            "scan_id": scan_id,
            "scan_phase": "live_tick_reproduction_scan",
            "scan_sequence": 1,
            "parent_id": 1,
            "parent_lineage_id": "L00000001",
            "biological_lineage_id": "L00000001",
            "agent_id": 1,
            "agent_energy": 43.0,
            "agent_health": 11.0,
            "reproduction_scan_seen": 1,
        },
        {
            "schema_version": 4,
            "event_schema_version": 4,
            "generation": 1,
            "tick": 2,
            "event": "reproduction_scan_eligible",
            "event_type": "reproduction_scan_eligible",
            "event_id": f"{scan_id}:scan_eligible",
            "scan_id": scan_id,
            "scan_sequence": 1,
            "parent_id": 1,
            "parent_lineage_id": "L00000001",
            "biological_lineage_id": "L00000001",
            "agent_id": 1,
            "scan_eligible": 1,
        },
    ]
    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[1],
        selected_parent_agents=agents,
        successful_births=1,
        failed_births=0,
        birth_failure_counts={},
        birth_contexts=[birth_event],
        sim_cfg=cfg,
        birth_event_contexts=[birth_event],
        scan_action_event_contexts=scan_contexts,
    )

    writer = LineageReproductionFunnelWriter(str(tmp_path))
    writer.write_generation(
        generation=1,
        pre_turnover_snapshot=snapshot,
        post_refill_agents=agents,
        turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1}},
        write_events=True,
        event_schema_version=4,
        write_scan_action_events=True,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "lineage_reproduction_funnel_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    exposure = [row for row in rows if row["event_type"] == "reproduction_scan_exposure"][0]
    scan_attempt = [row for row in rows if row["event_type"] == "reproduction_scan_attempt_recorded"][0]
    assert exposure["event_schema_version"] == 4
    assert exposure["schema_version"] == 4
    assert exposure["scan_id"] == scan_id
    assert exposure["biological_lineage_id"] == "L00000001"
    assert exposure["linked_attempt_event_id"]
    assert scan_attempt["linked_scan_event_id"] == exposure["event_id"]
    assert scan_attempt["linked_success_event_id"] == exposure["linked_success_event_id"]
    summary = json.loads((tmp_path / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(encoding="utf-8"))
    assert summary["event_schema_version"] == 4
    assert summary["reproduction_scan_exposure_rows"] == 1
    assert summary["reproduction_scan_eligible_rows"] == 1
    assert summary["reproduction_scan_attempt_recorded_rows"] == 1
    assert summary["reproduction_action_offered_rows"] == 0
    assert summary["reproduction_action_taken_rows"] == 0


def test_writer_summarizes_path2_roz_scan_v4_rows(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    cfg.lineage_reproduction_funnel_write_events = True
    cfg.lineage_reproduction_funnel_event_schema_version = 4
    cfg.lineage_reproduction_funnel_write_scan_action_events = True
    cfg.path2_roz_reporting_enable = True
    agents = [_agent(1, "L00000001")]
    scan_id = "g000001:t000002:scan:00000001:000001"
    roz_fields = {
        "path2_roz_reporting_enable": 1,
        "path2_roz_schema_version": 1,
        "path2_roz_zone_mode": "static_center_manhattan",
        "path2_roz_window_period_steps": 40,
        "path2_roz_window_duration_steps": 8,
        "path2_roz_zone_radius": 1,
        "path2_roz_center_y": 4,
        "path2_roz_center_x": 4,
        "path2_roz_window_active": 1,
        "path2_roz_window_id": 0,
        "path2_roz_active_cell_count": 5,
        "path2_roz_parent_distance_to_zone": 0,
        "path2_roz_mate_distance_to_zone": -1,
        "path2_roz_child_distance_to_zone": 0,
        "path2_roz_parent_in_zone": 1,
        "path2_roz_mate_in_zone": 0,
        "path2_roz_child_in_zone": 1,
        "path2_roz_birth_in_zone": 1,
        "path2_roz_schedule_mode": "periodic_single_zone",
        "path2_roz_schedule_id": "unit_schedule",
        "path2_roz_zone_id": "unit_zone",
        "path2_roz_zone_index": 0,
        "path2_roz_zone_count": 1,
        "path2_roz_window_index": 0,
        "path2_roz_schedule_phase_step": 2,
    }
    birth_event = {
        "schema_version": 2,
        "event_schema_version": 2,
        "generation": 1,
        "tick": 2,
        "event": "successful_birth",
        "event_type": "birth_success",
        "scan_id": scan_id,
        "lineage_id": "L00000001",
        "child_lineage_id": "L00000001",
        "parent_lineage_id": "L00000001",
        "parent_id": 1,
        "child_id": 12,
        "child_y": 4,
        "child_x": 4,
        "birth_attempted": 1,
        "birth_success": 1,
        **roz_fields,
    }
    scan_contexts = [
        {
            "schema_version": 4,
            "event_schema_version": 4,
            "generation": 1,
            "tick": 2,
            "event": "reproduction_scan_exposure",
            "event_type": "reproduction_scan_exposure",
            "event_id": f"{scan_id}:scan_exposure",
            "scan_id": scan_id,
            "scan_phase": "live_tick_reproduction_scan",
            "scan_sequence": 1,
            "parent_id": 1,
            "parent_lineage_id": "L00000001",
            "biological_lineage_id": "L00000001",
            "agent_id": 1,
            "parent_y": 4,
            "parent_x": 4,
            "reproduction_scan_seen": 1,
            **roz_fields,
        },
        {
            "schema_version": 4,
            "event_schema_version": 4,
            "generation": 1,
            "tick": 2,
            "event": "reproduction_scan_eligible",
            "event_type": "reproduction_scan_eligible",
            "event_id": f"{scan_id}:scan_eligible",
            "scan_id": scan_id,
            "scan_sequence": 1,
            "parent_id": 1,
            "parent_lineage_id": "L00000001",
            "biological_lineage_id": "L00000001",
            "agent_id": 1,
            "parent_y": 4,
            "parent_x": 4,
            "scan_eligible": 1,
            **roz_fields,
        },
    ]
    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[1],
        selected_parent_agents=agents,
        successful_births=1,
        failed_births=0,
        birth_failure_counts={},
        birth_contexts=[birth_event],
        sim_cfg=cfg,
        birth_event_contexts=[birth_event],
        scan_action_event_contexts=scan_contexts,
    )

    writer = LineageReproductionFunnelWriter(str(tmp_path))
    writer.write_generation(
        generation=1,
        pre_turnover_snapshot=snapshot,
        post_refill_agents=agents,
        turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1}},
        write_events=True,
        event_schema_version=4,
        write_scan_action_events=True,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "metrics" / "lineage_reproduction_funnel_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    exposure = [row for row in rows if row["event_type"] == "reproduction_scan_exposure"][0]
    scan_attempt = [row for row in rows if row["event_type"] == "reproduction_scan_attempt_recorded"][0]
    assert exposure["path2_roz_reporting_enable"] == 1
    assert exposure["path2_roz_parent_in_zone"] == 1
    assert exposure["parent_selected_rank"] == 1
    assert exposure["path2_roz_selected_parent_scan_in_zone"] == 1
    assert scan_attempt["path2_roz_selected_parent_scan_in_zone"] == 1
    summary = json.loads((tmp_path / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(encoding="utf-8"))
    assert summary["path2_roz_scan_exposure_rows"] == 1
    assert summary["path2_roz_scan_eligible_rows"] == 1
    assert summary["path2_roz_scan_attempt_recorded_rows"] == 1
    assert summary["path2_roz_birth_success_rows"] == 1
    assert summary["path2_roz_parent_in_zone_rows"] >= 3
    assert summary["path2_roz_selected_parent_scan_in_zone_rows"] == 3
    assert summary["path2_roz_schedule_window_active_rows"] >= 3
    assert summary["path2_roz_schedule_distinct_zone_count"] == 1
    assert summary["path2_roz_schedule_parent_in_zone_rows"] >= 3
    assert summary["path2_roz_schedule_selected_parent_scan_in_zone_rows"] == 3


def test_writer_summarizes_lro1_scan_v4_rows_and_disabled_leakage(tmp_path: Path) -> None:
    cfg = _enabled_cfg()
    cfg.lineage_reproduction_funnel_write_events = True
    cfg.lineage_reproduction_funnel_event_schema_version = 4
    cfg.lineage_reproduction_funnel_write_scan_action_events = True
    agents = [_agent(1, "L00000001")]
    scan_id = "g000001:t000002:scan:00000001:000001"
    lro_enabled = {
        "lro1_reproductive_opportunity_enable": 1,
        "lro1_reproductive_opportunity_mode": "local_patch_carrying_capacity_v1",
        "lro1_opportunity_score_parent_site": -0.45,
        "lro1_opportunity_multiplier": 0.55,
        "lro1_opportunity_context_density": 9.0,
        "lro1_opportunity_context_waste": 0.4,
        "lro1_opportunity_context_scent": 0.2,
        "lro1_opportunity_context_shelter_cover": 0.0,
        "lro1_opportunity_context_food": 0.0,
        "lro1_opportunity_context_hazard": 0.5,
        "lro1_opportunity_invalid_context": 0,
        "lro1_opportunity_neutral_context": 0,
        "lro1_opportunity_effective_try_prob": 0.55,
    }
    lro_disabled = {
        "lro1_reproductive_opportunity_enable": 0,
        "lro1_reproductive_opportunity_mode": "",
        "lro1_opportunity_score_parent_site": 0.0,
        "lro1_opportunity_multiplier": 1.0,
        "lro1_opportunity_context_density": 0.0,
        "lro1_opportunity_context_waste": 0.0,
        "lro1_opportunity_context_scent": 0.0,
        "lro1_opportunity_context_shelter_cover": 0.0,
        "lro1_opportunity_context_food": 0.0,
        "lro1_opportunity_context_hazard": 0.0,
        "lro1_opportunity_invalid_context": 0,
        "lro1_opportunity_neutral_context": 1,
        "lro1_opportunity_effective_try_prob": 1.0,
    }
    birth_event = {
        "schema_version": 2,
        "event_schema_version": 2,
        "generation": 1,
        "tick": 2,
        "event": "successful_birth",
        "event_type": "birth_success",
        "scan_id": scan_id,
        "lineage_id": "L00000001",
        "child_lineage_id": "L00000001",
        "parent_lineage_id": "L00000001",
        "parent_id": 1,
        "child_id": 12,
        "birth_attempted": 1,
        "birth_success": 1,
        **lro_enabled,
    }
    scan_contexts = [
        {
            "schema_version": 4,
            "event_schema_version": 4,
            "generation": 1,
            "tick": 2,
            "event": "reproduction_scan_exposure",
            "event_type": "reproduction_scan_exposure",
            "event_id": f"{scan_id}:scan_exposure",
            "scan_id": scan_id,
            "scan_sequence": 1,
            "parent_id": 1,
            "parent_lineage_id": "L00000001",
            "biological_lineage_id": "L00000001",
            "agent_id": 1,
            "reproduction_scan_seen": 1,
            **lro_disabled,
        },
        {
            "schema_version": 4,
            "event_schema_version": 4,
            "generation": 1,
            "tick": 2,
            "event": "reproduction_scan_eligible",
            "event_type": "reproduction_scan_eligible",
            "event_id": f"{scan_id}:scan_eligible",
            "scan_id": scan_id,
            "scan_sequence": 1,
            "parent_id": 1,
            "parent_lineage_id": "L00000001",
            "biological_lineage_id": "L00000001",
            "agent_id": 1,
            "scan_eligible": 1,
            **lro_enabled,
        },
        {
            "schema_version": 4,
            "event_schema_version": 4,
            "generation": 1,
            "tick": 2,
            "event": "reproduction_scan_no_attempt",
            "event_type": "reproduction_scan_no_attempt",
            "event_id": f"{scan_id}:scan_no_attempt",
            "scan_id": scan_id,
            "scan_sequence": 1,
            "parent_id": 1,
            "parent_lineage_id": "L00000001",
            "biological_lineage_id": "L00000001",
            "agent_id": 1,
            "action_not_taken": 1,
            **lro_enabled,
        },
    ]
    snapshot = capture_pre_turnover_snapshot(
        generation=1,
        alive_start_agents=agents,
        alive_after_agents=agents,
        eligible_ids=[1],
        selected_parent_agents=agents,
        successful_births=1,
        failed_births=0,
        birth_failure_counts={},
        birth_contexts=[birth_event],
        sim_cfg=cfg,
        birth_event_contexts=[birth_event],
        scan_action_event_contexts=scan_contexts,
    )

    writer = LineageReproductionFunnelWriter(str(tmp_path))
    writer.write_generation(
        generation=1,
        pre_turnover_snapshot=snapshot,
        post_refill_agents=agents,
        turnover_snapshot={"pre_cull_alive_counts": {"L00000001": 1}},
        write_events=True,
        event_schema_version=4,
        write_scan_action_events=True,
    )

    summary = json.loads((tmp_path / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(encoding="utf-8"))
    assert summary["lro1_event_rows"] >= 4
    assert summary["lro1_enabled_rows"] >= 3
    assert summary["lro1_disabled_rows"] >= 1
    assert summary["lro1_disabled_leakage_rows"] == 0
    assert summary["lro1_multiplier_lt1_rows"] >= 3
    assert summary["lro1_scan_exposure_rows"] == 1
    assert summary["lro1_scan_eligible_rows"] == 1
    assert summary["lro1_scan_no_attempt_rows"] == 1
    assert summary["lro1_birth_success_rows"] == 1


def test_v4_scan_action_tiny_runner_preserves_final_population_and_writes_scan_rows(tmp_path: Path) -> None:
    off = _tiny_sim(tmp_path / "off_scan", seed=808, reporting=False)
    off.run(start_gen=0, skip_init=False)
    on = _tiny_sim(tmp_path / "on_scan", seed=808, reporting=True)
    on.sim_cfg.lineage_reproduction_funnel_write_events = True
    on.sim_cfg.lineage_reproduction_funnel_event_schema_version = 4
    on.sim_cfg.lineage_reproduction_funnel_write_scan_action_events = True
    on.run(start_gen=0, skip_init=False)

    assert _fingerprint(on) == _fingerprint(off)
    events_path = tmp_path / "on_scan" / "metrics" / "lineage_reproduction_funnel_events.jsonl"
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert any(row["event_type"] == "reproduction_scan_exposure" for row in rows)
    assert all(row["event_schema_version"] == 4 for row in rows if str(row["event_type"]).startswith("reproduction_scan"))
    summary = json.loads((tmp_path / "on_scan" / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(encoding="utf-8"))
    assert summary["reproduction_scan_exposure_rows"] > 0
    assert summary["reproduction_action_offered_rows"] == 0
    assert summary["reproduction_action_taken_rows"] == 0


def test_path2_roz_periodic_schedule_runtime_fields_are_reporting_only(tmp_path: Path) -> None:
    off = _tiny_sim(tmp_path / "off_periodic", seed=910, reporting=False)
    off.run(start_gen=0, skip_init=False)
    on = _tiny_sim(tmp_path / "on_periodic", seed=910, reporting=True)
    on.sim_cfg.lineage_reproduction_funnel_write_events = True
    on.sim_cfg.lineage_reproduction_funnel_event_schema_version = 4
    on.sim_cfg.lineage_reproduction_funnel_write_scan_action_events = True
    on.sim_cfg.path2_roz_reporting_enable = True
    on.sim_cfg.path2_roz_schedule_mode = "periodic_single_zone"
    on.sim_cfg.path2_roz_schedule_id = "periodic_unit"
    on.sim_cfg.path2_roz_window_period_steps = 2
    on.sim_cfg.path2_roz_window_duration_steps = 1
    on.sim_cfg.path2_roz_zone_radius = 8
    on.run(start_gen=0, skip_init=False)

    assert _fingerprint(on) == _fingerprint(off)
    rows = [
        json.loads(line)
        for line in (tmp_path / "on_periodic" / "metrics" / "lineage_reproduction_funnel_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    roz_scan_rows = [
        row
        for row in rows
        if str(row["event_type"]).startswith("reproduction_scan")
        and int(row.get("path2_roz_reporting_enable", 0) or 0) == 1
    ]
    assert roz_scan_rows
    assert {row["path2_roz_schedule_mode"] for row in roz_scan_rows} == {"periodic_single_zone"}
    assert {row["path2_roz_schedule_id"] for row in roz_scan_rows} == {"periodic_unit"}
    assert {int(row["path2_roz_zone_count"]) for row in roz_scan_rows} == {1}
    assert {int(row["path2_roz_schedule_phase_step"]) for row in roz_scan_rows} <= {0, 1}
    assert any(int(row["path2_roz_window_active"]) == 1 for row in roz_scan_rows)
    assert any(int(row["path2_roz_window_active"]) == 0 for row in roz_scan_rows)
    summary = json.loads(
        (tmp_path / "on_periodic" / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["path2_roz_schedule_window_active_rows"] > 0
    assert summary["path2_roz_schedule_distinct_zone_count"] == 1
    assert summary["reproduction_action_offered_rows"] == 0
    assert summary["reproduction_action_taken_rows"] == 0


def test_path2_roz_round_robin_schedule_runtime_fields_are_reporting_only(tmp_path: Path) -> None:
    off = _tiny_sim(tmp_path / "off_round_robin", seed=911, reporting=False)
    off.run(start_gen=0, skip_init=False)
    on = _tiny_sim(tmp_path / "on_round_robin", seed=911, reporting=True)
    on.sim_cfg.lineage_reproduction_funnel_write_events = True
    on.sim_cfg.lineage_reproduction_funnel_event_schema_version = 4
    on.sim_cfg.lineage_reproduction_funnel_write_scan_action_events = True
    on.sim_cfg.path2_roz_reporting_enable = True
    on.sim_cfg.path2_roz_schedule_mode = "multi_zone_round_robin"
    on.sim_cfg.path2_roz_schedule_id = "round_robin_unit"
    on.sim_cfg.path2_roz_window_period_steps = 1
    on.sim_cfg.path2_roz_window_duration_steps = 1
    on.sim_cfg.path2_roz_schedule_zones_json = json.dumps(
        [
            {"id": "center", "center_y": 4, "center_x": 4, "radius": 8},
            {"id": "corner", "center_y": 0, "center_x": 0, "radius": 8},
        ]
    )
    on.run(start_gen=0, skip_init=False)

    assert _fingerprint(on) == _fingerprint(off)
    rows = [
        json.loads(line)
        for line in (tmp_path / "on_round_robin" / "metrics" / "lineage_reproduction_funnel_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    roz_scan_rows = [
        row
        for row in rows
        if str(row["event_type"]).startswith("reproduction_scan")
        and int(row.get("path2_roz_reporting_enable", 0) or 0) == 1
    ]
    assert roz_scan_rows
    assert {row["path2_roz_schedule_mode"] for row in roz_scan_rows} == {"multi_zone_round_robin"}
    assert {row["path2_roz_zone_id"] for row in roz_scan_rows} == {"center", "corner"}
    assert {int(row["path2_roz_zone_count"]) for row in roz_scan_rows} == {2}
    assert {int(row["path2_roz_zone_index"]) for row in roz_scan_rows} == {0, 1}
    summary = json.loads(
        (tmp_path / "on_round_robin" / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["path2_roz_schedule_distinct_zone_count"] == 2
    assert summary["path2_roz_schedule_zone_switch_count"] > 0
    assert summary["reproduction_action_offered_rows"] == 0
    assert summary["reproduction_action_taken_rows"] == 0


def test_path2_roz_patch_derived_static_schedule_runtime_fields_are_reporting_only(tmp_path: Path) -> None:
    off = _tiny_sim(tmp_path / "off_patch_derived", seed=912, reporting=False, patch_world=True)
    off.env.patch_food_amount.fill(0.0)
    off.env.patch_cover.fill(0.0)
    off.env.patch_shelter.fill(0.0)
    off.env.patch_waste.fill(0.0)
    off.env.patch_recovery_activity.fill(0.0)
    off.env.patch_food_amount[1, 2] = 1.0
    off.env.patch_shelter[1, 2] = 1.0
    off.run(start_gen=0, skip_init=False)

    on = _tiny_sim(tmp_path / "on_patch_derived", seed=912, reporting=True, patch_world=True)
    on.env.patch_food_amount.fill(0.0)
    on.env.patch_cover.fill(0.0)
    on.env.patch_shelter.fill(0.0)
    on.env.patch_waste.fill(0.0)
    on.env.patch_recovery_activity.fill(0.0)
    on.env.patch_food_amount[1, 2] = 1.0
    on.env.patch_shelter[1, 2] = 1.0
    on.sim_cfg.lineage_reproduction_funnel_write_events = True
    on.sim_cfg.lineage_reproduction_funnel_event_schema_version = 4
    on.sim_cfg.lineage_reproduction_funnel_write_scan_action_events = True
    on.sim_cfg.path2_roz_reporting_enable = True
    on.sim_cfg.path2_roz_schedule_mode = "patch_derived_static"
    on.sim_cfg.path2_roz_schedule_id = "patch_static_unit"
    on.sim_cfg.path2_roz_window_period_steps = 1
    on.sim_cfg.path2_roz_window_duration_steps = 1
    on.sim_cfg.path2_roz_zone_radius = 1
    on.run(start_gen=0, skip_init=False)

    assert _fingerprint(on) == _fingerprint(off)
    rows = [
        json.loads(line)
        for line in (tmp_path / "on_patch_derived" / "metrics" / "lineage_reproduction_funnel_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    roz_scan_rows = [
        row
        for row in rows
        if str(row["event_type"]).startswith("reproduction_scan")
        and int(row.get("path2_roz_reporting_enable", 0) or 0) == 1
    ]
    assert roz_scan_rows
    assert {row["path2_roz_schedule_mode"] for row in roz_scan_rows} == {"patch_derived_static"}
    assert {row["path2_roz_schedule_id"] for row in roz_scan_rows} == {"patch_static_unit"}
    assert len({row["path2_roz_zone_id"] for row in roz_scan_rows}) == 1
    assert all(str(row["path2_roz_zone_id"]).startswith("patch_static_y") for row in roz_scan_rows)
    assert len({int(row["path2_roz_center_y"]) for row in roz_scan_rows}) == 1
    assert len({int(row["path2_roz_center_x"]) for row in roz_scan_rows}) == 1
    summary = json.loads(
        (tmp_path / "on_patch_derived" / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["path2_roz_schedule_distinct_zone_count"] == 1
    assert summary["path2_roz_schedule_window_active_rows"] > 0
    assert summary["reproduction_action_offered_rows"] == 0
    assert summary["reproduction_action_taken_rows"] == 0


def test_path2_roz_tiny_runner_preserves_final_population_and_writes_zone_rows(tmp_path: Path) -> None:
    off = _tiny_sim(tmp_path / "off_roz", seed=909, reporting=False)
    off.run(start_gen=0, skip_init=False)
    on = _tiny_sim(tmp_path / "on_roz", seed=909, reporting=True)
    on.sim_cfg.lineage_reproduction_funnel_write_events = True
    on.sim_cfg.lineage_reproduction_funnel_event_schema_version = 4
    on.sim_cfg.lineage_reproduction_funnel_write_scan_action_events = True
    on.sim_cfg.path2_roz_reporting_enable = True
    on.sim_cfg.path2_roz_window_period_steps = 10
    on.sim_cfg.path2_roz_window_duration_steps = 10
    on.sim_cfg.path2_roz_zone_radius = 8
    on.run(start_gen=0, skip_init=False)

    assert _fingerprint(on) == _fingerprint(off)
    events_path = tmp_path / "on_roz" / "metrics" / "lineage_reproduction_funnel_events.jsonl"
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    roz_scan_rows = [
        row
        for row in rows
        if str(row["event_type"]).startswith("reproduction_scan")
        and int(row.get("path2_roz_reporting_enable", 0) or 0) == 1
    ]
    assert roz_scan_rows
    assert all(row["path2_roz_zone_mode"] == "static_center_manhattan" for row in roz_scan_rows)
    assert any(int(row.get("path2_roz_parent_in_zone", 0) or 0) == 1 for row in roz_scan_rows)
    summary = json.loads((tmp_path / "on_roz" / "metrics" / "lineage_reproduction_funnel_run_summary.json").read_text(encoding="utf-8"))
    assert summary["path2_roz_scan_exposure_rows"] > 0
    assert summary["path2_roz_parent_in_zone_rows"] > 0
    assert summary["reproduction_action_offered_rows"] == 0
    assert summary["reproduction_action_taken_rows"] == 0
