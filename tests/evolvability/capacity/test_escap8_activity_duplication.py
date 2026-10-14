from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from agents.structural_identity_v1 import new_native_identity
from evolution.operators_c1 import (
    _op_duplicate_with_decay,
    _op_motif_duplication,
    apply_operator_pipeline,
)
from evolution.structural_live_modules_v1 import (
    execute_agent_live_modules,
    reconcile_runtime_state_for_genome,
)
from simulation.rng_manager import RNGManager


class SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return float(next(self.values))


def _module(index: int, *, output_gain: float) -> dict:
    identity = new_native_identity(
        context={
            "campaign_id": "escap8-duplication-test",
            "branch_id": "sustainable-growth",
            "generation": 1,
            "tick": 0,
            "child_agent_id": index,
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap8-duplication-lineage",
        },
        operator_id="fixture",
        attempt_index=index,
        operator_role=f"module-{index}",
    )
    return {
        "module_id": f"m{index:03d}",
        "module_type": "generic",
        "params": {
            "live_operator": "bounded_scalar_integrator_v1",
            "expression_scale": 1.0,
            "live_retention": 0.5,
            "live_input_gain": 0.75,
            "live_output_gain": float(output_gain),
            "observation_routing": {
                "schema_version": "escap4_observation_routing_v1",
                "registry_version": "perception_routing_contract_v1",
                "bindings": [],
            },
            "output_routing": {
                "schema_version": "escap5_output_routing_v1",
                "registry_version": "eiel_action_semantic_routing_registry_v1",
                "bindings": [],
            },
        },
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **identity,
        "latent_pos": list(identity["topology_pos"]),
    }


def _payload(modules: list[dict]) -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": deepcopy(modules),
        "connections": [],
    }


def _context(index: int, *, enabled: bool) -> dict:
    return {
        "operator_id": "duplicate_with_decay",
        "attempt_index": index,
        "sustainable_growth_enabled": bool(enabled),
        "identity_context": {
            "campaign_id": "escap8-duplication-test",
            "branch_id": "sustainable-growth",
            "generation": 2,
            "tick": 0,
            "child_agent_id": 100 + index,
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap8-duplicate-child",
        },
    }


def test_integrator_duplicate_is_exact_output_neutral_only_when_escap8_enabled() -> None:
    source = _module(1, output_gain=1.5)

    historical, historical_changed = _op_duplicate_with_decay(
        _payload([source]), [0.0, 0.5, 0.5], _context(1, enabled=False)
    )
    treatment, treatment_changed = _op_duplicate_with_decay(
        _payload([source]), [0.0, 0.5, 0.5], _context(1, enabled=True)
    )

    assert historical_changed is treatment_changed is True
    assert historical["modules"][1]["params"]["live_output_gain"] != 0.0
    assert treatment["modules"][0] == source
    duplicate = treatment["modules"][1]
    assert duplicate["params"]["live_operator"] == "bounded_scalar_integrator_v1"
    assert duplicate["params"]["live_output_gain"] == 0.0
    assert duplicate["module_instance_id"] != source["module_instance_id"]
    assert duplicate["homology_id"] != source["homology_id"]
    assert duplicate["ancestry"]["parent_module_instance_ids"] == [
        source["module_instance_id"]
    ]
    duplicate["params"]["observation_routing"]["bindings"].append({"probe": True})
    assert source["params"]["observation_routing"]["bindings"] == []


def test_integrator_motif_neutralizes_every_new_output_without_aliasing() -> None:
    first = _module(2, output_gain=0.75)
    second = _module(3, output_gain=-0.5)

    child, changed = _op_motif_duplication(
        _payload([first, second]), [0.0, 0.5, 0.5], _context(2, enabled=True)
    )

    assert changed is True
    assert [row["params"]["live_output_gain"] for row in child["modules"][:2]] == [
        0.75,
        -0.5,
    ]
    assert [row["params"]["live_output_gain"] for row in child["modules"][2:]] == [
        0.0,
        0.0,
    ]
    child["modules"][2]["params"]["output_routing"]["bindings"].append(
        {"probe": True}
    )
    assert child["modules"][0]["params"]["output_routing"]["bindings"] == []


def test_runtime_rng_contract_locks_escap8_to_two_supplied_draws() -> None:
    manager = RNGManager(global_seed=8)

    assert manager.expected_delta_for_boundary(
        "mutation.c1.operator.sustainable_growth_mutation"
    ) == {"mutation.py": 2}


def test_pipeline_disabled_is_exact_and_enabled_spends_exactly_two_draws() -> None:
    payload = _payload([_module(4, output_gain=0.25)])
    historical_rng = SequenceRng([])
    historical = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=historical_rng,
        max_ops=0,
    )
    disabled_rng = SequenceRng([])
    disabled = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=disabled_rng,
        max_ops=0,
        structural_sustainable_growth_enable=False,
        structural_sustainable_growth_operator_attempts=1,
    )

    assert disabled == historical
    assert disabled_rng.draw_count == historical_rng.draw_count == 0
    assert "structural_sustainable_growth_mutation" not in disabled[1]

    enabled_rng = SequenceRng([0.1, 0.5])
    enabled_payload, telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=enabled_rng,
        max_ops=0,
        structural_sustainable_growth_enable=True,
        structural_sustainable_growth_operator_attempts=1,
    )

    assert enabled_rng.draw_count == 2
    growth = telemetry["structural_sustainable_growth_mutation"]
    assert growth["enabled"] is True
    assert growth["attempts"] == 1
    assert growth["draws_per_attempt"] == 2
    assert enabled_payload["genes"]["escap8_growth_attempt_count"] == 1
    assert enabled_payload["genes"]["escap8_growth_last_operation"] == "toggle_dormancy"
    assert enabled_payload["genes"]["escap8_growth_last_status"] == "accepted"
    assert enabled_payload["genes"]["escap8_blocked_prune_attempt_count"] == 0


def test_reporting_cost_does_not_change_active_behavior_energy_or_selection_inputs() -> None:
    module = _module(5, output_gain=0.75)

    def build_agent() -> SimpleNamespace:
        return SimpleNamespace(
            genome=SimpleNamespace(
                modules=[deepcopy(module)],
                connections=[],
                canonical_bytes=lambda: b"escap8-reporting-isolation",
            ),
            structural_live_module_state={},
            structural_live_module_outputs={},
            structural_live_module_telemetry={},
            structural_development_latch_by_instance_id={},
            structural_development_effective_gain_by_instance_id={},
            structural_development_stage_id="",
            energy=17.5,
            parent_eligible=True,
            selection_inputs={"fitness": 3.25, "lineage": "fixture"},
        )

    historical_agent = build_agent()
    reporting_agent = build_agent()
    historical = execute_agent_live_modules(historical_agent, fatigue=0.4)
    reporting = execute_agent_live_modules(
        reporting_agent,
        fatigue=0.4,
        sustainable_growth_enabled=True,
    )

    assert reporting.rest_residual == historical.rest_residual
    assert reporting_agent.structural_live_module_state == historical_agent.structural_live_module_state
    assert reporting_agent.structural_live_module_outputs == historical_agent.structural_live_module_outputs
    assert reporting_agent.energy == historical_agent.energy == 17.5
    assert reporting_agent.parent_eligible is historical_agent.parent_eligible is True
    assert reporting_agent.selection_inputs == historical_agent.selection_inputs
    assert reporting.summary["random_draws"] == historical.summary["random_draws"] == 0
    assert reporting.summary["structural_cost"]["reporting_only"] is True


def test_every_growth_lane_outcome_spends_the_same_two_draws() -> None:
    expected_operations = {
        0.10: "toggle_dormancy",
        0.30: "prune_eligible_module",
        0.60: "refine_descendant_parameter",
        0.90: "refine_descendant_routing_or_expression",
    }
    payload = _payload([_module(6, output_gain=0.5)])

    for operation_draw, expected_operation in expected_operations.items():
        rng = SequenceRng([operation_draw, 0.5])
        _, telemetry = apply_operator_pipeline(
            payload=deepcopy(payload),
            rng=rng,
            max_ops=0,
            structural_sustainable_growth_enable=True,
            structural_sustainable_growth_operator_attempts=1,
        )
        row = telemetry["structural_sustainable_growth_mutation"]["attempt_rows"][0]
        assert rng.draw_count == 2
        assert row["operation"] == expected_operation
        assert row["draw_budget"] == {
            "declared_supplied_draws": 2,
            "internal_rng_draws": 0,
        }


def test_prune_boundary_clears_stale_telemetry_even_without_runtime_map_orphans() -> None:
    kept = _module(7, output_gain=0.5)
    agent = SimpleNamespace(
        genome=SimpleNamespace(modules=[kept]),
        structural_live_module_state={},
        structural_live_module_outputs={},
        structural_development_latch_by_instance_id={},
        structural_development_effective_gain_by_instance_id={},
        structural_live_module_telemetry={
            "state_rows": [{"module_instance_id": "already_pruned"}]
        },
    )

    result = reconcile_runtime_state_for_genome(
        agent,
        orphan_policy="prune",
        sustainable_growth_enabled=True,
    )

    assert agent.structural_live_module_telemetry == {}
    assert result["pruned_instance_ids"] == []
    assert result["stale_telemetry_cleared"] is True


def test_enabled_pipeline_persists_blocked_growth_and_pressure_reporting_only() -> None:
    payload = _payload([_module(index, output_gain=0.5) for index in range(20, 32)])
    rng = SequenceRng([0.2, 0.2, 0.3, 0.4, 0.30, 0.5])

    child, telemetry = apply_operator_pipeline(
        payload=payload,
        rng=rng,
        max_ops=1,
        structural_sustainable_growth_enable=True,
        structural_sustainable_growth_operator_attempts=1,
        structural_identity_context={"lineage_token": "escap8-pressure-fixture"},
    )

    assert rng.draw_count == 6
    assert len(child["modules"]) == 12
    genes = child["genes"]
    assert genes["escap8_blocked_duplication_attempt_count"] == 1
    assert genes["escap8_blocked_prune_attempt_count"] == 1
    assert genes["escap8_cap_contact_count"] == 1
    assert genes["escap8_cap_pressure_streak"] == 1
    assert telemetry["structural_sustainable_growth_mutation"]["accepted"] == 0
