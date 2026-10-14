from __future__ import annotations

import json
from types import SimpleNamespace

from agents.developmental_program_v1 import canonicalize_developmental_program
from agents.genome import Genome
from env.env_config import EnvConfig
from simulation.metrics_pkg.cognitive_status import compute_cognitive_status_metrics
from simulation.metrics_pkg.trait_capacity_saturation import (
    SCHEMA_VERSION,
    trait_capacity_saturation_payloads,
)
from simulation.runner_pkg.runner_config import SimConfig


def _record(
    lineage: int,
    value: float,
    *,
    offspring: float = 0.0,
    module_count: int = 0,
    connection_count: int = 0,
) -> dict[str, object]:
    return {
        "lineage": int(lineage),
        "effective_memory": float(value),
        "mutation_rate": float(value),
        "effective_pred_lr": float(value),
        "sensor_radius": float(value),
        "sensor_channel_count": float(value),
        "module_count": float(module_count),
        "connection_count": float(connection_count),
        "action_temperature": float(value),
        "drive_gain": float(value),
        "reproduction_threshold": float(value),
        "offspring": float(offspring),
    }


def _payload(summary_json: str) -> dict:
    return json.loads(summary_json)


def _agent(agent_id: int, genome: Genome, *, offspring: int = 0, lineage: int | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        agent_id=int(agent_id),
        genome=genome,
        energy=50.0,
        health=10.0,
        alive=True,
        drives=[0.0, 0.0, 0.0, 0.0],
        stats=SimpleNamespace(
            offspring_count=int(offspring),
            lineage_id=int(lineage if lineage is not None else agent_id),
            pred_err_count=0,
            pred_err_sum=0.0,
            prediction_update_count=0,
        ),
    )


def _genome(
    *,
    memory_dim: int = 16,
    mutation_rate: float = 0.03,
    pred_lr: float = 0.08,
    sensor_radius: int = 2,
    action_temperature: float = 1.0,
    drive_logit_gain: float = 1.0,
    reproduction_threshold: float = 30.0,
    modules: list[dict] | None = None,
    connections: list[dict] | None = None,
) -> Genome:
    return Genome(
        hidden_size=64,
        exploration_bias=0.1,
        aggression_bias=0.2,
        reproduction_threshold=float(reproduction_threshold),
        mutation_rate=float(mutation_rate),
        action_temperature=float(action_temperature),
        use_gru=bool(memory_dim > 0),
        memory_dim=int(memory_dim),
        drive_inertia=0.85,
        drive_noise=0.02,
        drive_logit_gain=float(drive_logit_gain),
        pred_lr=float(pred_lr),
        sensor_radius=int(sensor_radius),
        modules=list(modules or []),
        connections=list(connections or []),
        developmental_program=canonicalize_developmental_program(None),
    )


def test_reporting_disabled_keeps_compact_fields_empty() -> None:
    cfg = SimConfig(seed=1)
    cfg.trait_capacity_saturation_reporting_enable = False
    metrics = compute_cognitive_status_metrics(
        [_agent(1, _genome(memory_dim=64))],
        selected_parent_agents=[],
        env_cfg=EnvConfig(width=8, height=8, view_radius=3),
        sim_cfg=cfg,
        generation=1,
    )
    assert metrics["trait_capacity_saturation_summary_json"] == "{}"
    assert metrics["trait_capacity_saturation_lineage_json"] == "{}"
    assert metrics["trait_capacity_saturation_structural_payload_json"] == "{}"


def test_inactive_sensor_radius_is_not_active_not_below_cap() -> None:
    cfg = SimConfig(seed=1)
    cfg.trait_capacity_saturation_reporting_enable = True
    cfg.enable_evolvable_sensor_radius = False
    metrics = compute_cognitive_status_metrics(
        [_agent(1, _genome(sensor_radius=3))],
        env_cfg=EnvConfig(width=8, height=8, view_radius=3),
        sim_cfg=cfg,
        generation=2,
    )
    trait = _payload(metrics["trait_capacity_saturation_summary_json"])["traits"]["sensor_radius"]
    assert trait["status"] == "not_active"
    assert trait["active"] is False
    assert trait["evaluable"] is False


def test_below_cap_and_cap_touch_are_separate_states() -> None:
    cfg = SimConfig(seed=1)
    cfg.trait_capacity_saturation_reporting_enable = True
    below = compute_cognitive_status_metrics(
        [_agent(1, _genome(memory_dim=32))],
        env_cfg=EnvConfig(width=8, height=8),
        sim_cfg=cfg,
        generation=1,
    )
    touched = compute_cognitive_status_metrics(
        [_agent(1, _genome(memory_dim=64))],
        env_cfg=EnvConfig(width=8, height=8),
        sim_cfg=cfg,
        generation=1,
    )
    assert _payload(below["trait_capacity_saturation_summary_json"])["traits"]["memory_dim"]["status"] == "below_cap"
    mem = _payload(touched["trait_capacity_saturation_summary_json"])["traits"]["memory_dim"]
    assert mem["status"] == "cap_touch"
    assert mem["cap_touch"] is True
    assert mem["cap_persistent"] is False
    assert mem["cap_selected"] is False
    assert mem["cap_binding"] is False


def test_persistent_lineage_and_selected_enrichment_rules_are_explicit() -> None:
    alive = [_record(1, 64), _record(2, 16), _record(3, 16), _record(4, 16)]
    selected = [_record(1, 64), _record(1, 64)]
    summary_json, _, _ = trait_capacity_saturation_payloads(
        alive,
        selected,
        lineage_streaks={"memory_dim": {1: 3}},
        sim_cfg=SimConfig(seed=1),
    )
    mem = _payload(summary_json)["traits"]["memory_dim"]
    assert mem["cap_touch"] is True
    assert mem["cap_persistent"] is True
    assert mem["persistent_lineage_count"] == 1
    assert mem["longest_lineage_cap_streak"] == 3
    assert mem["cap_selected"] is True
    assert mem["selected_minus_alive_at_cap_fraction"] == 0.75


def test_binding_requires_blocked_attempt_evidence() -> None:
    alive = [
        _record(1, 64, module_count=12, connection_count=0),
        _record(2, 16, module_count=1, connection_count=0),
    ]
    selected = [_record(1, 64, module_count=12), _record(1, 64, module_count=12)]
    no_block, _, _ = trait_capacity_saturation_payloads(
        alive,
        selected,
        lineage_streaks={"module_count": {1: 3}},
        sim_cfg=SimConfig(seed=1),
    )
    with_block, _, _ = trait_capacity_saturation_payloads(
        alive,
        selected,
        lineage_streaks={"module_count": {1: 3}},
        structural_context={"module_cap_drop_count": 2},
        sim_cfg=SimConfig(seed=1),
    )
    assert _payload(no_block)["traits"]["module_count"]["cap_binding"] is False
    mod = _payload(with_block)["traits"]["module_count"]
    assert mod["cap_binding"] is True
    assert mod["binding_evaluable"] is True
    assert mod["status"] == "cap_binding"


def test_cap_touch_without_blocked_evidence_is_binding_not_evaluable() -> None:
    summary_json, _, _ = trait_capacity_saturation_payloads(
        [_record(1, 64), _record(2, 32)],
        [_record(1, 64), _record(1, 64)],
        lineage_streaks={"memory_dim": {1: 3}},
        sim_cfg=SimConfig(seed=1),
    )
    mem = _payload(summary_json)["traits"]["memory_dim"]
    assert mem["cap_touch"] is True
    assert mem["binding_evaluable"] is False
    assert mem["cap_binding"] is False


def test_structural_empty_is_not_below_cap() -> None:
    cfg = SimConfig(seed=1)
    summary_json, _, structural_json = trait_capacity_saturation_payloads(
        [_record(1, 0, module_count=0, connection_count=0), _record(2, 0, module_count=0, connection_count=0)],
        [],
        sim_cfg=cfg,
    )
    payload = _payload(summary_json)
    structural = _payload(structural_json)
    assert payload["traits"]["module_count"]["status"] == "structural_payload_empty"
    assert payload["traits"]["connection_count"]["status"] == "structural_payload_empty"
    assert structural["structural_status"] == "structural_payload_empty"


def test_structural_non_empty_below_cap_and_exact_cap() -> None:
    below_json, _, _ = trait_capacity_saturation_payloads(
        [_record(1, 1, module_count=2, connection_count=3)],
        [],
        sim_cfg=SimConfig(seed=1),
    )
    cap_json, _, structural_json = trait_capacity_saturation_payloads(
        [_record(1, 1, module_count=12, connection_count=32)],
        [],
        structural_context={"module_cap_drop_count": 1, "connection_cap_drop_count": 1},
        sim_cfg=SimConfig(seed=1),
    )
    assert _payload(below_json)["traits"]["module_count"]["status"] == "below_cap"
    payload = _payload(cap_json)
    assert payload["traits"]["module_count"]["cap_touch"] is True
    assert payload["traits"]["connection_count"]["cap_touch"] is True
    structural = _payload(structural_json)
    assert structural["module_cap_touch_fraction"] == 1.0
    assert structural["connection_cap_touch_fraction"] == 1.0


def test_rep1a_json_is_stable_and_versioned() -> None:
    args = (
        [_record(2, 32), _record(1, 64, offspring=1.0)],
        [_record(1, 64), _record(1, 64)],
    )
    a = trait_capacity_saturation_payloads(*args, lineage_streaks={"memory_dim": {1: 3}}, sim_cfg=SimConfig(seed=1))
    b = trait_capacity_saturation_payloads(*args, lineage_streaks={"memory_dim": {1: 3}}, sim_cfg=SimConfig(seed=1))
    assert a == b
    assert _payload(a[0])["schema_version"] == SCHEMA_VERSION
