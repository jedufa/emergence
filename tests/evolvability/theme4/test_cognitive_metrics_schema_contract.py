from __future__ import annotations

import math
from dataclasses import asdict, fields

from agents.cognition import compute_cognitive_budget_ledger, express_cognitive_phenotype
from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from env.env_config import EnvConfig
from simulation.metrics_pkg.cognitive_status import (
    cognitive_status_metric_defaults,
    compute_cognitive_status_metrics,
)
from simulation.metrics_pkg.rows import MetricsRow
from simulation.runner_pkg.runner_config import SimConfig


JSON_METRIC_FIELDS = {
    "mem_evo_population_memory_dim_histogram",
    "mem_evo_selected_parent_memory_dim_histogram",
    "mem_evo_descendant_memory_dim_histogram",
    "mem_evo_lineage_memory_dim_summary_json",
    "mem_evo_effective_memory_dim_histogram",
    "trait_capacity_saturation_summary_json",
    "trait_capacity_saturation_lineage_json",
    "trait_capacity_saturation_structural_payload_json",
    "structural_live_semantics_summary_json",
}


def _genome(*, memory_dim: int = 16) -> Genome:
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
        drive_noise=0.02,
        drive_logit_gain=1.1,
        pred_lr=0.08,
        developmental_program=canonicalize_developmental_program(None),
    )


def _metrics_row_payload() -> dict[str, object]:
    payload: dict[str, object] = {}
    for field in fields(MetricsRow):
        if field.name.endswith("_json"):
            payload[field.name] = "[]"
        elif field.name == "dev_stage_eval_stage_counts_json":
            payload[field.name] = "{}"
        else:
            payload[field.name] = 0.0
    return payload


def test_cognitive_status_metrics_match_metrics_row_schema() -> None:
    defaults = cognitive_status_metric_defaults()
    computed = compute_cognitive_status_metrics([], env_cfg=None, sim_cfg=None)
    row_fields = {field.name for field in fields(MetricsRow)}

    assert set(computed) == set(defaults)
    assert set(defaults) <= row_fields

    payload = _metrics_row_payload()
    payload.update(computed)
    row = MetricsRow(**payload)

    assert row.cog_capacity_alive_end_mean == computed["cog_capacity_alive_end_mean"]
    for key in computed:
        value = computed[key]
        if key in JSON_METRIC_FIELDS:
            assert isinstance(value, str)
        else:
            assert isinstance(value, (int, float))
            assert math.isfinite(float(value))


def test_cog_capacity_fields_are_emitted_and_in_metrics_row_consistently() -> None:
    defaults = cognitive_status_metric_defaults()
    row_fields = {field.name for field in fields(MetricsRow)}
    emitted_capacity_fields = {key for key in defaults if key.startswith("cog_capacity_")}
    row_capacity_fields = {key for key in row_fields if key.startswith("cog_capacity_")}

    assert {
        "cog_capacity_alive_end_mean",
        "cog_capacity_memory_component_mean",
        "cog_capacity_prediction_component_mean",
        "cog_capacity_drive_component_mean",
        "cog_capacity_perception_component_mean",
    } <= emitted_capacity_fields
    assert emitted_capacity_fields <= row_capacity_fields


def test_cognitive_budget_ledger_constructs_for_active_and_disabled_paths() -> None:
    env_cfg = EnvConfig(width=8, height=8, view_radius=3)
    genome = _genome(memory_dim=64)
    ledger_fields = {field.name for field in fields(type(compute_cognitive_budget_ledger(
        express_cognitive_phenotype(genome, env_cfg=env_cfg, sim_cfg=SimConfig(seed=1)),
        env_cfg=env_cfg,
        sim_cfg=SimConfig(seed=1),
    )))}

    for enabled in (True, False):
        sim_cfg = SimConfig(seed=1)
        sim_cfg.enable_cognitive_budget_ledger = bool(enabled)
        phenotype = express_cognitive_phenotype(genome, env_cfg=env_cfg, sim_cfg=sim_cfg)
        ledger = compute_cognitive_budget_ledger(phenotype, env_cfg=env_cfg, sim_cfg=sim_cfg)
        values = asdict(ledger)

        assert set(values) == ledger_fields
        for key in (
            "prediction_update_cost_hypothetical",
            "prediction_update_cost_charged",
            "prediction_state_cost_hypothetical",
            "prediction_state_cost_charged",
            "prediction_cost_per_tick",
            "prediction_cost_total_component",
            "prediction_cost_fraction_of_cog_cost",
            "prediction_capacity_component",
            "prediction_cost_charging_enabled",
            "prediction_cost_telemetry_only",
            "perception_cost_total",
            "active_sensor_channel_count",
            "active_sensor_radius",
            "cell_channel_capacity",
        ):
            assert key in values

        for value in values.values():
            if isinstance(value, (int, float)):
                assert math.isfinite(float(value))

        if enabled:
            assert values["prediction_capacity_component"] == phenotype.cognitive_capacity_prediction_component
        else:
            assert values["prediction_capacity_component"] == 0.0
            assert values["perception_cost_total"] == 0.0
