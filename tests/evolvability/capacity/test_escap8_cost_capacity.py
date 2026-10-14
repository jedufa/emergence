from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from agents.structural_identity_v1 import new_native_identity
from evolution.capacity_contract import build_capacity_contract, capacity_headroom_summary
from evolution.structural_gated_recurrent_v1 import (
    GATED_OPERATOR,
    neutral_gated_recurrent_record,
)
from evolution.structural_live_modules_v1 import (
    execute_agent_live_modules,
    reconcile_runtime_state_for_genome,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_RECORD_KEY,
    ACTIVITY_SCHEMA_VERSION,
    COST_COEFFICIENTS,
    COST_MODE,
    COST_PROFILE,
    COST_SCHEMA_VERSION,
)
from simulation.metrics_pkg.trait_capacity_saturation import (
    trait_capacity_saturation_payloads,
)


def _identity(index: int) -> dict:
    return new_native_identity(
        context={
            "campaign_id": "escap8-cost-capacity-test",
            "branch_id": "sustainable-growth",
            "generation": 1,
            "tick": 0,
            "child_agent_id": index,
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap8-cost-capacity-lineage",
        },
        operator_id="fixture",
        attempt_index=index,
        operator_role=f"module-{index}",
    )


def _module(
    index: int,
    *,
    dormant: bool = False,
    gated: bool = False,
    output_gain: float = 1.0,
    with_bindings: bool = False,
    prerequisite_homology_ids: tuple[str, ...] = (),
) -> dict:
    identity = _identity(index)
    params: dict = {
        "live_operator": "bounded_scalar_integrator_v1",
        "expression_scale": 1.0,
        "live_retention": 0.5,
        "live_input_gain": 1.0,
        "live_output_gain": float(output_gain),
    }
    if gated:
        record = neutral_gated_recurrent_record()
        record["output_gain"] = float(output_gain)
        params = {
            "live_operator": GATED_OPERATOR,
            "expression_scale": 1.0,
            "gated_recurrent": record,
        }
    if dormant:
        params[ACTIVITY_RECORD_KEY] = {
            "schema_version": ACTIVITY_SCHEMA_VERSION,
            "dormant": True,
        }
    if with_bindings:
        params["observation_routing"] = {
            "bindings": [
                {
                    "semantic_id": "self_energy",
                    "target_slot": "in",
                    "gain": 1.0,
                    "enabled": True,
                }
            ]
        }
        params["output_routing"] = {
            "bindings": [
                {
                    "semantic_id": "rest",
                    "source_slot": "out",
                    "gain": 1.0,
                    "enabled": True,
                }
            ]
        }
    if prerequisite_homology_ids:
        params["developmental_expression"] = {
            "prerequisite_homology_ids": list(prerequisite_homology_ids)
        }
    return {
        "module_id": f"m{index:03d}",
        "module_type": "generic",
        "params": params,
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _connection(source: dict, target: dict) -> dict:
    return {
        "connection_id": "c000",
        "src_module_id": source["module_id"],
        "dst_module_id": target["module_id"],
        "src_slot": "out",
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "typed",
        "weight": 1.0,
    }


def _agent(modules: list[dict], connections: list[dict] | None = None) -> SimpleNamespace:
    genome = SimpleNamespace(
        modules=deepcopy(modules),
        connections=deepcopy(connections or []),
        canonical_bytes=lambda: b"escap8-canonical-genome",
    )
    return SimpleNamespace(
        genome=genome,
        structural_live_module_state={},
        structural_live_module_outputs={},
        structural_live_module_telemetry={},
        structural_development_latch_by_instance_id={},
        structural_development_effective_gain_by_instance_id={},
        structural_development_stage_id="",
    )


def _plan(modules: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        effective_gain_by_instance_id={
            str(module["module_instance_id"]): 1.0 for module in modules
        },
        latch_by_instance_id={},
        stage_id="mature",
        telemetry={"stage_id": "mature", "random_draws": 0},
    )


def test_dormant_source_and_target_freeze_state_zero_outputs_and_skip_cost() -> None:
    source = _module(1, dormant=True)
    target = _module(2, dormant=True)
    source_key = str(source["module_instance_id"])
    target_key = str(target["module_instance_id"])
    agent = _agent([source, target], [_connection(source, target)])
    agent.structural_live_module_state = {source_key: 0.75, target_key: -0.25}
    agent.structural_live_module_outputs = {source_key: 0.8, target_key: -0.5}
    agent.structural_development_latch_by_instance_id = {source_key: True}

    step = execute_agent_live_modules(
        agent,
        fatigue=1.0,
        sustainable_growth_enabled=True,
    )

    assert agent.structural_live_module_state == {source_key: 0.75, target_key: -0.25}
    assert agent.structural_live_module_outputs == {source_key: 0.0, target_key: 0.0}
    assert agent.structural_development_latch_by_instance_id == {source_key: True}
    assert step.rest_residual == 0.0
    assert step.summary["executing_modules"] == 0
    assert step.summary["dormant_modules"] == 2
    assert step.summary["evaluated_connections"] == 0
    assert step.summary["state_updates"] == 0
    assert step.summary["reported_compute_cost"] == 0
    assert step.summary["structural_cost"]["counts"]["dormant_module_maintenance"] == 2
    assert step.summary["structural_cost_total"] == pytest.approx(2.0)
    assert step.summary["bloat_classification"]["classification"] == "dormant_reserve"
    assert all(row["structurally_dormant"] for row in step.summary["state_rows"])
    assert not any(row["executed_live_operator"] for row in step.summary["state_rows"])


def test_enabled_cost_formula_counts_execution_bindings_dependencies_and_bytes() -> None:
    source = _module(3, with_bindings=True)
    target = _module(
        4,
        gated=True,
        output_gain=1.0,
        prerequisite_homology_ids=(str(source["homology_id"]),),
    )
    agent = _agent([source, target], [_connection(source, target)])
    agent.energy = 17.5
    agent.stats = SimpleNamespace(
        survival_steps=12,
        energy_collected=8.0,
        offspring_count=1,
        attacks=0,
    )
    stats_before = deepcopy(vars(agent.stats))

    step = execute_agent_live_modules(
        agent,
        fatigue=0.5,
        developmental_plan=_plan([source, target]),
        gated_recurrent_enabled=True,
        sustainable_growth_enabled=True,
    )
    cost = step.summary["structural_cost"]

    assert cost["schema_version"] == COST_SCHEMA_VERSION
    assert cost["profile"] == COST_PROFILE
    assert cost["mode"] == COST_MODE == "reporting_only"
    assert cost["reporting_only"] is True
    assert cost["coefficients"] == COST_COEFFICIENTS
    assert cost["counts"] == {
        "allocated_module": 2,
        "executing_integrator": 1,
        "executing_gated": 1,
        "evaluated_connection": 1,
        "active_input_binding": 1,
        "active_output_binding": 1,
        "active_dependency_edge": 1,
        "dormant_module_maintenance": 0,
        "serialized_genome_byte": len(b"escap8-canonical-genome"),
        "saturated_or_rejected_activity": 0,
    }
    assert cost["total"] == pytest.approx(44.0)
    assert step.summary["structural_cost_total"] == pytest.approx(44.0)
    assert step.summary["reported_compute_cost"] == 38
    assert step.summary["serialized_genome_cap"] is None
    assert step.summary["serialized_genome_capacity_bounded"] is False
    assert agent.energy == 17.5
    assert vars(agent.stats) == stats_before


def test_feature_off_keeps_escap7_summary_and_state_exact() -> None:
    module = _module(5)
    first = _agent([module])
    second = _agent([module])
    expected = execute_agent_live_modules(first, fatigue=0.4)
    observed = execute_agent_live_modules(
        second,
        fatigue=0.4,
        sustainable_growth_enabled=False,
    )

    assert expected == observed
    assert first.structural_live_module_state == second.structural_live_module_state
    assert first.structural_live_module_outputs == second.structural_live_module_outputs
    assert "sustainable_growth_enabled" not in observed.summary
    assert "structural_cost" not in observed.summary


def test_prune_reconcile_clears_every_runtime_surface_before_local_slot_reuse() -> None:
    kept = _module(6)
    kept_key = str(kept["module_instance_id"])
    removed = _module(7)
    removed_key = str(removed["module_instance_id"])
    agent = _agent([kept])
    agent.structural_live_module_state = {kept_key: 0.1, removed_key: 0.2}
    agent.structural_live_module_outputs = {kept_key: 0.3, removed_key: 0.4}
    agent.structural_development_latch_by_instance_id = {
        kept_key: True,
        removed_key: True,
    }
    agent.structural_development_effective_gain_by_instance_id = {
        kept_key: 1.0,
        removed_key: 0.5,
    }
    agent.structural_live_module_telemetry = {
        "state_rows": [{"module_instance_id": removed_key}],
        "state_instance_map": {"m999": removed_key},
    }

    result = reconcile_runtime_state_for_genome(
        agent,
        orphan_policy="prune",
        sustainable_growth_enabled=True,
    )

    assert agent.structural_live_module_state == {kept_key: 0.1}
    assert agent.structural_live_module_outputs == {kept_key: 0.3}
    assert agent.structural_development_latch_by_instance_id == {kept_key: True}
    assert agent.structural_development_effective_gain_by_instance_id == {kept_key: 1.0}
    assert agent.structural_live_module_telemetry == {}
    assert result["pruned_instance_ids"] == [removed_key]
    assert result["pruned_development_latch_instance_ids"] == [removed_key]
    assert result["pruned_development_effective_gain_instance_ids"] == [removed_key]
    assert result["stale_telemetry_cleared"] is True

    replacement = _module(8)
    replacement["module_id"] = removed["module_id"]
    replacement_key = str(replacement["module_instance_id"])
    agent.genome.modules = [deepcopy(kept), deepcopy(replacement)]
    execute_agent_live_modules(
        agent,
        fatigue=0.0,
        sustainable_growth_enabled=True,
    )
    assert removed_key not in agent.structural_live_module_state
    assert removed_key not in agent.structural_live_module_outputs
    assert agent.structural_live_module_state[replacement_key] == 0.0
    assert agent.structural_live_module_outputs[replacement_key] == 0.0


def test_capacity_contract_and_headroom_report_activity_without_new_caps() -> None:
    cfg = SimpleNamespace(structural_sustainable_growth_enable=True)
    contract = build_capacity_contract(sim_cfg=cfg)
    growth = contract["structure"]["sustainable_growth"]

    assert contract["structure"]["limits"] == {
        "max_connections_per_agent": 32,
        "max_module_fanin": 6,
        "max_module_fanout": 6,
        "max_modules_per_agent": 12,
    }
    assert growth["enabled"] is True
    assert growth["fixed_shape_contract"]["new_runtime_arrays"] == 0
    assert growth["cost"]["mode"] == "reporting_only"
    assert growth["cost"]["organism_energy_debit"] is False
    assert growth["classification_is_selection_input"] is False

    headroom = capacity_headroom_summary(
        [
            {
                "module_count": 12,
                "connection_count": 4,
                "executing_modules": 7,
                "dormant_modules": 3,
                "causally_active_modules": 2,
                "neutral_duplicate_modules": 2,
                "redundant_modules": 2,
                "prunable_modules": 2,
                "structural_cost_total": 99.0,
            }
        ],
        contract=contract,
        structural_context={
            "cap_pressure_streak": 3,
            "blocked_duplication_attempts": 1,
            "blocked_prune_attempts": 2,
        },
    )["sustainable_growth"]

    assert headroom["allocated_module_count_max"] == 12
    assert headroom["executing_module_count_max"] == 7
    assert headroom["dormant_module_count_max"] == 3
    assert headroom["causally_active_module_count_max"] == 2
    assert headroom["prunable_module_count_max"] == 2
    assert headroom["module_cap_contact"] is True
    assert headroom["sustained_cap_pressure"] is True
    assert headroom["structural_cost_max"] == pytest.approx(99.0)
    assert headroom["bloat_classification"]["classification"] == "possible_bloat"
    assert headroom["serialized_genome_cap"] is None
    assert headroom["reporting_only"] is True
    assert headroom["selection_input"] is False


def test_production_reporting_aggregates_growth_blocks_pressure_and_cost_from_rows() -> None:
    cfg = SimpleNamespace(structural_sustainable_growth_enable=True)
    row = {
        "module_count": 12,
        "connection_count": 2,
        "executing_modules": 3,
        "dormant_modules": 2,
        "nonzero_output_modules": 1,
        "neutral_duplicate_modules": 2,
        "redundant_modules": 2,
        "prunable_modules": 2,
        "structural_cost_total": 55.0,
        "escap8_growth_attempt_count": 4,
        "escap8_blocked_duplication_attempt_count": 2,
        "escap8_blocked_prune_attempt_count": 1,
        "escap8_cap_pressure_streak": 3,
        "escap8_specialization_metrics_json": '{"specialized":true}',
    }

    summary_json, _, structural_json = trait_capacity_saturation_payloads(
        [row],
        sim_cfg=cfg,
    )
    summary = json.loads(summary_json)
    structural = json.loads(structural_json)
    headroom = summary["capacity_headroom"]["sustainable_growth"]

    assert headroom["blocked_duplication_attempts"] == 2
    assert headroom["blocked_prune_attempts"] == 1
    assert headroom["cap_pressure_streak"] == 3
    assert headroom["sustained_cap_pressure"] is True
    assert headroom["structural_cost_max"] == pytest.approx(55.0)
    assert headroom["causally_active_module_count_max"] == 0
    assert structural["executing_module_count_max"] == pytest.approx(3.0)
    assert structural["dormant_module_count_max"] == pytest.approx(2.0)
    assert structural["specialization_metrics_record_count"] == 1
    assert structural["reporting_only"] is True
    assert structural["selection_input"] is False
