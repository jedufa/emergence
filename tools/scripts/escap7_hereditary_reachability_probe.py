#!/usr/bin/env python3
from __future__ import annotations

"""Bounded deterministic ESCAP-7 hereditary reachability/persistence probe.

This is a CPU/reference mechanics probe.  It has no environment, reward,
fitness, tuning, or selection input and makes no ecological or adaptive claim.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_ROOT = PROJECT_ROOT / "engine"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from agents.agent import ACTION_VOCABULARY  # noqa: E402
from agents.genome import Genome  # noqa: E402
from env.env_config import EnvConfig  # noqa: E402
from evolution.capacity_contract import build_capacity_contract  # noqa: E402
from evolution.operators_c1 import (  # noqa: E402
    _op_duplicate_with_decay,
    _op_promoter_expression_mutation,
    apply_gated_recurrent_operator_attempt,
    apply_operator_pipeline,
)
from evolution.recombination_c2 import (  # noqa: E402
    recombine_lineage_payloads,
    recombine_self_lineage_payload,
)
from evolution.structural_development_v1 import (  # noqa: E402
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    neutral_developmental_expression_record,
    normalize_agent_developmental_modules,
    validate_agent_developmental_graph,
)
from evolution.structural_gated_recurrent_v1 import (  # noqa: E402
    GATED_OPERATOR,
    GATED_RECORD_KEY,
    GATED_STATE_WIDTH,
)
from evolution.structural_live_modules_v1 import execute_agent_live_modules  # noqa: E402
from evolution.structural_output_routing_v1 import (  # noqa: E402
    ACTION_SEMANTIC_REGISTRY_VERSION,
    OUTPUT_ROUTING_BLOCK_KEY,
    OUTPUT_ROUTING_SCHEMA_VERSION,
    compile_agent_output_routing,
    normalize_agent_output_routing_modules,
)
from simulation.runner_pkg.runner_config import SimConfig  # noqa: E402


PROBE_SCHEMA_VERSION = "escap7_hereditary_reachability_persistence_probe_v1"


class SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self.values = [float(value) for value in values]
        self.draw_count = 0

    def random(self) -> float:
        if self.draw_count >= len(self.values):
            raise AssertionError("unexpected hidden RNG draw")
        value = self.values[self.draw_count]
        self.draw_count += 1
        return float(value)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _empty_payload() -> dict[str, Any]:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": [],
        "connections": [],
    }


def _identity_context(*, generation: int, child_id: int) -> dict[str, Any]:
    return {
        "campaign_id": "escap7-hereditary-probe",
        "branch_id": "fixed-width-gated",
        "generation": int(generation),
        "tick": 0,
        "child_agent_id": int(child_id),
        "parent_agent_ids": [max(0, int(child_id) - 2), max(0, int(child_id) - 1)],
        "lineage_token": "escap7-hereditary-probe-lineage",
    }


def _canonical(payload: dict[str, Any]) -> dict[str, Any]:
    return Genome.from_dict(deepcopy(payload)).to_dict()


def _identity_map(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        str(module["homology_id"]): {
            "module_instance_id": str(module["module_instance_id"]),
            "ancestry": deepcopy(module["ancestry"]),
            "topology_pos": list(module["topology_pos"]),
        }
        for module in payload["modules"]
    }


def _architecture_map(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "identities": _identity_map(payload),
        "operators": {
            str(module["homology_id"]): str(module["params"].get("live_operator", ""))
            for module in payload["modules"]
        },
        "development": {
            str(module["homology_id"]): deepcopy(
                module["params"].get(DEVELOPMENTAL_EXPRESSION_BLOCK_KEY)
            )
            for module in payload["modules"]
        },
        "output_routing": {
            str(module["homology_id"]): deepcopy(
                module["params"].get(OUTPUT_ROUTING_BLOCK_KEY)
            )
            for module in payload["modules"]
        },
        "connections": [
            {
                key: deepcopy(connection.get(key))
                for key in (
                    "src_module_id",
                    "dst_module_id",
                    "src_slot",
                    "dst_slot",
                    "value_type",
                    "channel",
                    "weight",
                )
            }
            for connection in payload["connections"]
        ],
    }


def _agent(
    payload: dict[str, Any],
    *,
    state: dict[str, float] | None = None,
    outputs: dict[str, float] | None = None,
) -> SimpleNamespace:
    state_map = deepcopy(state or {})
    output_map = (
        deepcopy(outputs)
        if outputs is not None
        else {str(key): 0.0 for key in state_map}
    )
    return SimpleNamespace(
        genome=SimpleNamespace(
            modules=deepcopy(payload["modules"]),
            connections=deepcopy(payload["connections"]),
        ),
        structural_live_module_state=state_map,
        structural_live_module_outputs=output_map,
        structural_live_module_telemetry={},
        structural_development_latch_by_instance_id={},
        structural_development_stage_id="",
        structural_development_effective_gain_by_instance_id={},
    )


def _development_plan(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        effective_gain_by_instance_id={
            str(module["module_instance_id"]): 1.0 for module in payload["modules"]
        },
        latch_by_instance_id={},
        stage_id="mature",
        telemetry={"stage_id": "mature", "random_draws": 0},
    )


def _shape_contract() -> dict[str, Any]:
    cfg = SimConfig()
    contract = build_capacity_contract(sim_cfg=cfg, env_cfg=EnvConfig())
    live = contract["structure"]["live_computation"]
    gated = next(
        row
        for row in live["supported_operators"]
        if row["operator"] == GATED_OPERATOR
    )
    fixed = dict(contract["policy"]["escap7_fixed_shape_contract"])
    return {
        "generic_module_cap": int(
            contract["structure"]["limits"]["max_modules_per_agent"]
        ),
        "connection_cap": int(
            contract["structure"]["limits"]["max_connections_per_agent"]
        ),
        "state_width_per_module": int(gated["state_width"]),
        "runtime_state_storage": "structural_live_module_state scalar map",
        "runtime_output_storage": "structural_live_module_outputs scalar map",
        "new_physical_arrays": int(live["fixed_storage_contract"]["new_arrays"]),
        "observation_width": contract["observations"]["active_observation_width"],
        "observation_width_change": int(
            contract["observations"]["escap7_observation_width_change"]
        ),
        "action_width": int(contract["actions"]["count"]),
        "action_width_change": int(contract["actions"]["escap7_action_width_change"]),
        "mlp_hidden_width": int(fixed["mlp_hidden_width"]),
        "central_gru_hidden_width": int(fixed["central_gru_hidden_width"]),
        "central_gru_unchanged": bool(fixed["central_gru_unchanged"]),
        "policy_tensor_shape_change": int(fixed["policy_tensor_shape_change"]),
    }


def _event(
    index: int,
    name: str,
    *,
    attempts: int = 1,
    accepted: int = 0,
    rejected: int = 0,
    noops: int = 0,
    blocked: int = 0,
    draw_delta: int = 0,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sequence_index": int(index),
        "event": str(name),
        "attempts": int(attempts),
        "outcomes": {
            "accepted": int(accepted),
            "rejected": int(rejected),
            "noop": int(noops),
            "blocked": int(blocked),
        },
        "exact_draw_delta": int(draw_delta),
        "details": deepcopy(details or {}),
    }


def run_sequence() -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    initial_shape = _shape_contract()
    initial_shape_digest = _hash(initial_shape)

    # Matched ESCAP-2 control: same four scheduled draws, bootstrap gate off.
    bootstrap_control_rng = SequenceRng([0.20, 0.10, 0.20, 0.30])
    bootstrap_control, bootstrap_control_telemetry = apply_operator_pipeline(
        payload=_empty_payload(),
        rng=bootstrap_control_rng,
        max_ops=1,
        structural_bootstrap_enable=False,
        structural_identity_context=_identity_context(generation=1, child_id=100),
    )

    # 1. Ordinary ESCAP-2 C1 scheduling creates the first generic scalar.
    bootstrap_rng = SequenceRng([0.20, 0.10, 0.20, 0.30])
    payload, bootstrap_telemetry = apply_operator_pipeline(
        payload=_empty_payload(),
        rng=bootstrap_rng,
        max_ops=1,
        structural_bootstrap_enable=True,
        structural_identity_context=_identity_context(generation=1, child_id=100),
    )
    payload = _canonical(payload)
    founder = payload["modules"][0]
    founder_identity = _identity_map(payload)
    events.append(
        _event(
            1,
            "ordinary_escap2_bootstrap_generic_scalar",
            accepted=1,
            draw_delta=bootstrap_rng.draw_count,
            details={
                "module_count": len(payload["modules"]),
                "module_type": founder["module_type"],
                "operator": founder["params"].get(
                    "live_operator", "bounded_scalar_integrator_v1"
                ),
                "telemetry": bootstrap_telemetry,
            },
        )
    )

    # One existing scalar lifetime value is retained across neutral conversion.
    founder_instance = str(founder["module_instance_id"])
    runtime = _agent(payload, state={founder_instance: 0.75})
    neutral_before_state = deepcopy(runtime.structural_live_module_state)

    # Matched operator-disabled control still consumes the fixed wrapper budget.
    acquisition_parent_modules = deepcopy(payload["modules"])
    acquisition_control_rng = SequenceRng([0.10, 0.50])
    acquisition_control = apply_gated_recurrent_operator_attempt(
        deepcopy(acquisition_parent_modules),
        rng=acquisition_control_rng,
        enabled=False,
    )

    # 2. The post-ESCAP-6 two-draw lane acquires the output-neutral gated type.
    acquisition_rng = SequenceRng([0.10, 0.50])
    acquired, acquisition_telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=acquisition_rng,
        max_ops=0,
        structural_gated_recurrent_mutation_enable=True,
        structural_gated_recurrent_operator_attempts=1,
    )
    payload = _canonical(acquired)
    runtime.genome.modules = deepcopy(payload["modules"])
    neutral_step = execute_agent_live_modules(
        runtime,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )
    acquired_row = acquisition_telemetry["structural_gated_recurrent_mutation"][
        "attempt_rows"
    ][0]
    first_acquisition = {
        "sequence_index": 2,
        "homology_id": str(payload["modules"][0]["homology_id"]),
        "module_instance_id": str(payload["modules"][0]["module_instance_id"]),
        "public_output": float(neutral_step.summary["state_rows"][0]["typed_output"]),
    }
    events.append(
        _event(
            2,
            "neutral_gated_type_acquisition",
            accepted=int(acquired_row["status"] == "accepted"),
            rejected=int(acquired_row["status"] == "rejected"),
            noops=int(acquired_row["status"] in {"noop", "ineligible"}),
            draw_delta=acquisition_rng.draw_count,
            details={
                "mutation": acquired_row,
                "state_before_conversion": neutral_before_state,
                "state_after_neutral_tick": deepcopy(runtime.structural_live_module_state),
                "public_output": first_acquisition["public_output"],
            },
        )
    )

    # Existing promoter mutation expresses the module without changing its
    # gated type or record.  It is support for the later output-gain mutation.
    promoted, promoted_ok = _op_promoter_expression_mutation(
        deepcopy(payload), [0.0, 0.5], {}
    )
    payload = _canonical(promoted)

    # Matched feature-gate-off control consumes no post-ESCAP-6 lane draws.
    parameter_parent = deepcopy(payload)
    parameter_control_rng = SequenceRng([])
    parameter_control, parameter_control_telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=parameter_control_rng,
        max_ops=0,
        structural_gated_recurrent_mutation_enable=False,
        structural_gated_recurrent_operator_attempts=0,
    )
    parameter_control_unchanged = _canonical(parameter_control) == _canonical(
        parameter_parent
    )

    # 3. One later legal two-draw output-gain mutation makes output nonzero.
    parameter_rng = SequenceRng([0.90, 0.75])
    activated, parameter_telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=parameter_rng,
        max_ops=0,
        structural_gated_recurrent_mutation_enable=True,
        structural_gated_recurrent_operator_attempts=1,
    )
    payload = _canonical(activated)
    runtime.genome.modules = deepcopy(payload["modules"])
    active_step = execute_agent_live_modules(
        runtime,
        fatigue=0.0,
        gated_recurrent_enabled=True,
    )
    parameter_row = parameter_telemetry["structural_gated_recurrent_mutation"][
        "attempt_rows"
    ][0]
    active_output = float(active_step.summary["state_rows"][0]["typed_output"])
    first_nonzero_output = {
        "sequence_index": 3,
        "module_instance_id": founder_instance,
        "output": active_output,
        "state": float(runtime.structural_live_module_state[founder_instance]),
    }
    events.append(
        _event(
            3,
            "later_legal_gated_parameter_activation",
            attempts=2,
            accepted=int(bool(promoted_ok)) + int(parameter_row["status"] == "accepted"),
            rejected=int(parameter_row["status"] == "rejected"),
            noops=int(parameter_row["status"] in {"noop", "ineligible"}),
            draw_delta=2 + parameter_rng.draw_count,
            details={
                "promoter_supplied_draws": 2,
                "gated_mutation": parameter_row,
                "first_nonzero_output": first_nonzero_output,
            },
        )
    )

    # 4. Ordinary duplication deep-copies heredity, assigns new identity and
    # homology, neutralizes only the duplicate's output gain, and owns no state.
    duplicate_context = {
        "operator_id": "duplicate_with_decay",
        "attempt_index": 0,
        "identity_context": _identity_context(generation=2, child_id=101),
    }
    duplicated, duplicated_ok = _op_duplicate_with_decay(
        deepcopy(payload), [0.0, 0.5, 0.5], duplicate_context
    )
    payload = _canonical(duplicated)
    original, duplicate = payload["modules"]
    duplicate_identity = {
        "original_instance": str(original["module_instance_id"]),
        "duplicate_instance": str(duplicate["module_instance_id"]),
        "original_homology": str(original["homology_id"]),
        "duplicate_homology": str(duplicate["homology_id"]),
        "duplicate_ancestry": deepcopy(duplicate["ancestry"]),
        "duplicate_output_gain": float(
            duplicate["params"][GATED_RECORD_KEY]["output_gain"]
        ),
    }
    events.append(
        _event(
            4,
            "gated_duplication_new_identity_and_neutral_output",
            accepted=int(bool(duplicated_ok)),
            draw_delta=3,
            details=duplicate_identity,
        )
    )

    # 5. Add one valid acyclic typed connection from the parent to duplicate.
    payload["connections"] = [
        {
            "connection_id": "c000",
            "src_module_id": str(original["module_id"]),
            "dst_module_id": str(duplicate["module_id"]),
            "src_slot": "out",
            "dst_slot": "in",
            "value_type": "scalar",
            "channel": "typed",
            "weight": 0.5,
        }
    ]
    payload = _canonical(payload)
    events.append(
        _event(
            5,
            "valid_internal_connection",
            accepted=1,
            details={"connection": deepcopy(payload["connections"][0])},
        )
    )

    # 6. Allocate valid bounded ESCAP-6 records without dependencies.
    modules = deepcopy(payload["modules"])
    for module in modules:
        module["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = (
            neutral_developmental_expression_record()
        )
    modules = normalize_agent_developmental_modules(modules)
    developmental_order = validate_agent_developmental_graph(modules)
    payload["modules"] = modules
    payload = _canonical(payload)
    events.append(
        _event(
            6,
            "valid_escap6_developmental_records",
            accepted=1,
            details={
                "record_count": len(modules),
                "topological_homology_order": [
                    str(module["homology_id"]) for module in developmental_order
                ],
            },
        )
    )

    # 7. Allocate one valid ESCAP-5 semantic output binding.
    modules = deepcopy(payload["modules"])
    modules[0]["params"][OUTPUT_ROUTING_BLOCK_KEY] = {
        "schema_version": OUTPUT_ROUTING_SCHEMA_VERSION,
        "registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
        "bindings": [
            {
                "semantic_id": "rest",
                "source_slot": "out",
                "gain": 1.0,
                "enabled": True,
            }
        ],
    }
    modules = normalize_agent_output_routing_modules(modules)
    payload["modules"] = modules
    payload = _canonical(payload)
    route_agent = _agent(payload, state=runtime.structural_live_module_state)
    execute_agent_live_modules(
        route_agent,
        fatigue=0.0,
        gated_recurrent_enabled=True,
        developmental_plan=_development_plan(payload),
    )
    routed = compile_agent_output_routing(route_agent, legacy_rest_residual=0.0)
    events.append(
        _event(
            7,
            "valid_escap5_output_binding",
            accepted=1,
            details={
                "binding": deepcopy(
                    payload["modules"][0]["params"][OUTPUT_ROUTING_BLOCK_KEY]
                ),
                "binding_count": int(routed.summary["allocated_bindings"]),
                "random_draws": int(routed.summary["random_draws"]),
            },
        )
    )

    architecture_before_inheritance = _architecture_map(payload)
    identity_before_inheritance = _identity_map(payload)

    # 8. Ordinary self-lineage inheritance consumes its declared C2 budget.
    inheritance_rng = SequenceRng([0.11, 0.22, 0.33])
    inherited, inheritance_summary = recombine_self_lineage_payload(
        parent_payload=deepcopy(payload),
        rng=inheritance_rng,
    )
    inherited = _canonical(inherited)
    architecture_after_inheritance = _architecture_map(inherited)
    events.append(
        _event(
            8,
            "ordinary_inheritance_preserves_architecture",
            accepted=1,
            draw_delta=inheritance_rng.draw_count,
            details={
                "budget_draws": int(inheritance_summary.budget_draws),
                "architecture_preserved": (
                    architecture_after_inheritance == architecture_before_inheritance
                ),
                "identity_continuity": (
                    _identity_map(inherited) == identity_before_inheritance
                ),
            },
        )
    )

    # 9. A compatible second parent differs in one legal gated value.  C2
    # chooses whole module donors and preserves the complete gated architecture.
    parent_b = deepcopy(inherited)
    parent_b["modules"][0]["params"][GATED_RECORD_KEY]["reset_bias"] = 1.0
    parent_b = _canonical(parent_b)
    c2_rng = SequenceRng([0.91, 0.82, 0.73])
    c2_child, c2_summary = recombine_lineage_payloads(
        parent_a_payload=deepcopy(inherited),
        parent_b_payload=deepcopy(parent_b),
        rng=c2_rng,
    )
    c2_child = _canonical(c2_child)
    parent_a_records = {
        str(module["homology_id"]): deepcopy(module["params"][GATED_RECORD_KEY])
        for module in inherited["modules"]
    }
    parent_b_records = {
        str(module["homology_id"]): deepcopy(module["params"][GATED_RECORD_KEY])
        for module in parent_b["modules"]
    }
    whole_donor_records = all(
        module["params"][GATED_RECORD_KEY]
        in (
            parent_a_records[str(module["homology_id"])],
            parent_b_records[str(module["homology_id"])],
        )
        for module in c2_child["modules"]
    )
    events.append(
        _event(
            9,
            "compatible_c2_preserves_gated_architecture",
            accepted=1,
            draw_delta=c2_rng.draw_count,
            details={
                "summary": vars(c2_summary),
                "whole_donor_records": bool(whole_donor_records),
                "module_count": len(c2_child["modules"]),
                "connection_count": len(c2_child["connections"]),
            },
        )
    )

    # 10. Lifetime state maps are outside hereditary payloads.  A new child
    # starts empty, including the duplicate, even though the parent had state.
    parent_runtime_state = deepcopy(route_agent.structural_live_module_state)
    newborn = _agent(c2_child)
    lifetime_nonhereditary = (
        bool(parent_runtime_state)
        and newborn.structural_live_module_state == {}
        and newborn.structural_live_module_outputs == {}
        and "structural_live_module_state" not in c2_child
        and "structural_live_module_outputs" not in c2_child
    )
    events.append(
        _event(
            10,
            "lifetime_state_nonhereditary",
            accepted=1,
            details={
                "parent_state_entry_count": len(parent_runtime_state),
                "newborn_state_entry_count": len(newborn.structural_live_module_state),
                "newborn_output_entry_count": len(newborn.structural_live_module_outputs),
                "payload_has_runtime_state": "structural_live_module_state" in c2_child,
            },
        )
    )

    # Checkpoint-like continuation: save the fixed scalar maps, restore them,
    # and require the next CPU/reference tick to match exactly.
    first_tick = execute_agent_live_modules(
        newborn,
        fatigue=0.25,
        gated_recurrent_enabled=True,
        developmental_plan=_development_plan(c2_child),
    )
    saved_state = deepcopy(newborn.structural_live_module_state)
    saved_outputs = deepcopy(newborn.structural_live_module_outputs)
    restored = _agent(c2_child, state=saved_state, outputs=saved_outputs)
    source_next = execute_agent_live_modules(
        newborn,
        fatigue=0.5,
        gated_recurrent_enabled=True,
        developmental_plan=_development_plan(c2_child),
    )
    restored_next = execute_agent_live_modules(
        restored,
        fatigue=0.5,
        gated_recurrent_enabled=True,
        developmental_plan=_development_plan(c2_child),
    )
    checkpoint_continuation = {
        "seed_tick_rng_draws": int(first_tick.summary["random_draws"]),
        "source_next_rng_draws": int(source_next.summary["random_draws"]),
        "restored_next_rng_draws": int(restored_next.summary["random_draws"]),
        "state_equal": (
            newborn.structural_live_module_state
            == restored.structural_live_module_state
        ),
        "outputs_equal": (
            newborn.structural_live_module_outputs
            == restored.structural_live_module_outputs
        ),
        "summary_equal": source_next.summary == restored_next.summary,
        "state_entry_count": len(restored.structural_live_module_state),
        "state_width_per_entry": 1,
    }

    # 11 is evaluated by run_probe(), which runs this complete sequence twice.
    events.append(
        _event(
            11,
            "exact_repeat_parity",
            accepted=1,
            details={"evaluated_by_outer_repeat": True},
        )
    )

    # 12. The declared physical controller and module widths never changed.
    final_shape = _shape_contract()
    final_shape_digest = _hash(final_shape)
    shape_unchanged = initial_shape == final_shape
    events.append(
        _event(
            12,
            "physical_shape_digest_unchanged",
            accepted=1,
            details={
                "initial_digest": initial_shape_digest,
                "final_digest": final_shape_digest,
                "shape_contract": final_shape,
                "runtime_state_entries": len(restored.structural_live_module_state),
                "runtime_state_width_per_module": int(GATED_STATE_WIDTH),
            },
        )
    )

    controls = {
        "bootstrap_disabled": {
            "draw_delta": bootstrap_control_rng.draw_count,
            "module_count": len(bootstrap_control["modules"]),
            "connection_count": len(bootstrap_control["connections"]),
            "noop_count": int(bootstrap_control_telemetry["noop_count"]),
            "blocked_count": int(bootstrap_control_telemetry["capacity_blocked_count"]),
        },
        "gated_attempt_wrapper_disabled": {
            "draw_delta": acquisition_control_rng.draw_count,
            "status": acquisition_control.telemetry["status"],
            "modules_unchanged": (
                acquisition_control.modules == acquisition_parent_modules
            ),
        },
        "gated_feature_gate_disabled": {
            "draw_delta": parameter_control_rng.draw_count,
            "payload_unchanged": bool(parameter_control_unchanged),
            "telemetry_lane_absent": (
                "structural_gated_recurrent_mutation" not in parameter_control_telemetry
            ),
        },
    }

    checks = {
        "01_bootstrap_reaches_generic_scalar": (
            len(founder_identity) == 1
            and founder["module_type"] == "generic"
            and bootstrap_rng.draw_count == 4
            and bootstrap_control_rng.draw_count == 4
            and len(bootstrap_control["modules"]) == 0
        ),
        "02_neutral_gated_acquisition": (
            acquired_row["status"] == "accepted"
            and acquisition_rng.draw_count == 2
            and first_acquisition["public_output"] == 0.0
            and _identity_map(payload)[str(original["homology_id"])]["module_instance_id"]
            == str(original["module_instance_id"])
        ),
        "03_later_parameter_mutation_nonzero_output": (
            bool(promoted_ok)
            and parameter_row["status"] == "accepted"
            and parameter_row["selected_field"] == "output_gain"
            and parameter_rng.draw_count == 2
            and active_output != 0.0
        ),
        "04_duplication_new_identity_homology_and_ancestry": (
            bool(duplicated_ok)
            and duplicate_identity["original_instance"]
            != duplicate_identity["duplicate_instance"]
            and duplicate_identity["original_homology"]
            != duplicate_identity["duplicate_homology"]
            and duplicate_identity["duplicate_output_gain"] == 0.0
            and duplicate_identity["duplicate_ancestry"][
                "parent_module_instance_ids"
            ]
            == [duplicate_identity["original_instance"]]
        ),
        "05_internal_connection_valid": (
            len(c2_child["connections"]) == 1
            and c2_child["connections"][0]["value_type"] == "scalar"
        ),
        "06_developmental_records_valid": (
            len(validate_agent_developmental_graph(c2_child["modules"])) == 2
            and all(
                DEVELOPMENTAL_EXPRESSION_BLOCK_KEY in module["params"]
                for module in c2_child["modules"]
            )
        ),
        "07_output_binding_valid": (
            routed.summary["allocated_bindings"] == 1
            and routed.summary["random_draws"] == 0
            and any(
                module["params"].get(OUTPUT_ROUTING_BLOCK_KEY, {}).get("bindings")
                for module in c2_child["modules"]
            )
        ),
        "08_ordinary_inheritance_preserves_architecture": (
            architecture_after_inheritance == architecture_before_inheritance
            and inheritance_rng.draw_count == inheritance_summary.budget_draws == 3
        ),
        "09_compatible_c2_preserves_atomic_gated_architecture": (
            c2_rng.draw_count == c2_summary.budget_draws == 3
            and c2_summary.matched_modules == 2
            and c2_summary.matched_connections == 1
            and whole_donor_records
            and all(
                module["params"]["live_operator"] == GATED_OPERATOR
                for module in c2_child["modules"]
            )
        ),
        "10_lifetime_state_nonhereditary": lifetime_nonhereditary,
        "11_repeat_parity_exact": True,
        "12_physical_shape_digest_unchanged": (
            shape_unchanged
            and initial_shape_digest == final_shape_digest
            and final_shape["state_width_per_module"] == 1
            and final_shape["new_physical_arrays"] == 0
            and final_shape["action_width"] == len(ACTION_VOCABULARY) == 7
            and final_shape["observation_width_change"] == 0
            and final_shape["action_width_change"] == 0
            and final_shape["central_gru_hidden_width"] == 64
            and final_shape["central_gru_unchanged"]
            and final_shape["policy_tensor_shape_change"] == 0
        ),
    }

    return {
        "lane": "python_reference_cpu_stride1",
        "required_sequence": events,
        "matched_disabled_controls": controls,
        "measurements": {
            "first_acquisition": first_acquisition,
            "first_nonzero_gated_output": first_nonzero_output,
            "founder_identity": founder_identity,
            "identity_after_inheritance": _identity_map(inherited),
            "duplication": duplicate_identity,
            "c2": {
                "summary": vars(c2_summary),
                "whole_donor_records": bool(whole_donor_records),
            },
            "checkpoint_like_continuation": checkpoint_continuation,
            "physical_shape_contract": final_shape,
            "physical_shape_digest": final_shape_digest,
            "execution_rng_draws": {
                "neutral_tick": int(neutral_step.summary["random_draws"]),
                "active_tick": int(active_step.summary["random_draws"]),
                "routing_tick": int(
                    route_agent.structural_live_module_telemetry["random_draws"]
                ),
                "checkpoint_seed_tick": int(first_tick.summary["random_draws"]),
                "checkpoint_source_next": int(source_next.summary["random_draws"]),
                "checkpoint_restored_next": int(restored_next.summary["random_draws"]),
            },
        },
        "checks": checks,
    }


def run_probe() -> dict[str, Any]:
    first = run_sequence()
    second = run_sequence()
    repeat_parity = first == second
    first["checks"]["11_repeat_parity_exact"] = bool(repeat_parity)
    repeat_event = next(
        event
        for event in first["required_sequence"]
        if event["sequence_index"] == 11
    )
    repeat_event["details"]["exact_parity"] = bool(repeat_parity)
    repeat_event["details"]["second_run_digest"] = _hash(second)
    payload: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "status": "passed"
        if repeat_parity and all(first["checks"].values())
        else "failed",
        "purpose": "bounded hereditary reachability and persistence mechanics only",
        **first,
        "repeat": {
            "runs": 2,
            "exact_parity": bool(repeat_parity),
            "run_digest": _hash(first),
        },
        "interpretation_boundary": (
            "No ecology, tuning, reward, fitness, selection, adaptiveness, "
            "cognition, intelligence, learning, or emergence claim."
        ),
    }
    payload["probe_hash"] = _hash(payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = run_probe()
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
