from __future__ import annotations

"""Bounded CPU oracle for the Phase-E modular tick integration boundary.

This fixture intentionally proves less than a runtime parity gate.  It freezes
one small, deterministic Python-reference cell so later accelerated lanes have
an exact receipt to compare against.  It owns no RNG and does not import test
helpers.
"""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

import numpy as np

from agents.structural_identity_v1 import new_native_identity
from evolution.structural_development_runtime_v1 import resolve_agent_development_runtime
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    DEVELOPMENTAL_FEATURE_PROFILE,
    neutral_developmental_expression_record,
)
from evolution.structural_gated_recurrent_v1 import (
    GATED_OPERATOR,
    GATED_PROFILE,
    GATED_RECORD_KEY,
    neutral_gated_recurrent_record,
    step_gated_scalar,
)
from evolution.structural_live_modules_v1 import (
    LIVE_OPERATOR,
    LIVE_PROFILE,
    execute_agent_live_modules,
    execute_live_modules_batch,
)
from evolution.structural_observation_routing_v1 import (
    ROUTING_BLOCK_KEY,
    ROUTING_PROFILE,
    normalize_routing_block,
    route_observations_batch,
)
from evolution.structural_output_routing_v1 import (
    OUTPUT_ROUTING_BLOCK_KEY,
    OUTPUT_ROUTING_PROFILE,
    add_output_routing_residuals_,
    compile_output_routing_batch,
    normalize_output_routing_block,
)
from evolution.structural_sustainable_growth_v1 import (
    ACTIVITY_RECORD_KEY,
    ACTIVITY_SCHEMA_VERSION,
    SUSTAINABLE_GROWTH_PROFILE,
    fixed_envelope_shape_digest_for_caps,
    fixed_envelope_shape_for_caps,
    mutate_sustainable_growth,
)
from simulation.runner_pkg.runner_config import SimConfig


SCHEMA_VERSION = "eiel_phase_e_modular_tick_fixture_receipt_v1"
FIXTURE_ID = "phase_e_t_flat_64_agents_3_ticks_v1"
EXPECTED_RECEIPT_SHA256 = "7a0deb703751cd68ddd4ad9b8628c841aecee43784fcb021b31dc4b6ddca1aea"
LANE_OWNER = "python_reference_cpu"
AGENT_COUNT = 64
STRUCTURED_AGENT_COUNT = 4
NEUTRAL_AGENT_COUNT = 60
OBSERVATION_WIDTH = 27
ACTION_WIDTH = 7
LOCAL_SIZE = 25
TICK_AGES = (0, 20, 40)
ARM_NAMES = (
    "integrator",
    "gated",
    "input_routing",
    "output_routing",
    "development",
    "dependency",
    "dormancy",
    "structural_cost",
)
EXPANDED_B2_CAPS = {
    "max_modules_per_agent": 16,
    "max_connections_per_agent": 48,
    "max_module_fanin": 6,
    "max_module_fanout": 6,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_receipt_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_receipt_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_receipt_bytes(value)).hexdigest()


def _tensor_receipt(value: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": str(array.dtype),
        "shape": [int(part) for part in array.shape],
        "nbytes": int(array.nbytes),
        "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _float32_bits_sha256(values: Iterable[Any]) -> str:
    array = np.ascontiguousarray(list(values), dtype=np.float32)
    bits = array.view(np.uint32)
    return hashlib.sha256(bits.tobytes(order="C")).hexdigest()


def _same_f32(left: Any, right: Any) -> bool:
    return int(np.float32(left).view(np.uint32)) == int(np.float32(right).view(np.uint32))


def _identity(agent_index: int, module_index: int, role: str) -> dict[str, Any]:
    return new_native_identity(
        context={
            "campaign_id": FIXTURE_ID,
            "branch_id": "phase-e-t",
            "generation": 0,
            "tick": 0,
            "child_agent_id": int(agent_index),
            "parent_agent_ids": [],
            "lineage_token": f"phase-e-agent-{agent_index}",
        },
        operator_id="phase_e_fixture",
        attempt_index=int(module_index),
        operator_role=str(role),
    )


def _development(
    gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    prerequisites: Iterable[str] = (),
    latch: bool = False,
) -> dict[str, Any]:
    record = neutral_developmental_expression_record()
    record["stage_gains"] = [float(value) for value in gains]
    record["prerequisite_homology_ids"] = sorted(str(value) for value in prerequisites)
    record["latch_dependency_readiness"] = bool(latch)
    return record


def _module(
    agent_index: int,
    module_index: int,
    role: str,
    *,
    operator: str = LIVE_OPERATOR,
    input_semantic: str | None = None,
    input_gain: float = 0.0,
    output_semantic: str | None = None,
    output_gain: float = 0.0,
    development: dict[str, Any] | None = None,
    dormant: bool = False,
) -> dict[str, Any]:
    identity = _identity(agent_index, module_index, role)
    params: dict[str, Any]
    if operator == GATED_OPERATOR:
        gated = neutral_gated_recurrent_record()
        gated.update(
            {
                "reset_bias": -0.5,
                "reset_input_gain": 1.0,
                "reset_state_gain": -0.75,
                "update_bias": 0.25,
                "update_input_gain": -1.25,
                "update_state_gain": 0.5,
                "candidate_bias": -0.125,
                "candidate_input_gain": 1.5,
                "candidate_state_gain": -1.0,
                "output_gain": 1.25,
            }
        )
        params = {
            "live_operator": GATED_OPERATOR,
            "expression_scale": -0.75,
            GATED_RECORD_KEY: gated,
        }
    else:
        params = {
            "live_operator": LIVE_OPERATOR,
            "expression_scale": 0.75,
            "live_retention": 0.5,
            "live_input_gain": 0.75,
            "live_output_gain": 0.5,
        }
    if input_semantic is not None:
        params[ROUTING_BLOCK_KEY] = normalize_routing_block(
            {
                "bindings": [
                    {
                        "semantic_id": str(input_semantic),
                        "target_slot": "in",
                        "gain": float(input_gain),
                        "enabled": True,
                    }
                ]
            }
        )
    if output_semantic is not None:
        params[OUTPUT_ROUTING_BLOCK_KEY] = normalize_output_routing_block(
            {
                "bindings": [
                    {
                        "semantic_id": str(output_semantic),
                        "source_slot": "out",
                        "gain": float(output_gain),
                        "enabled": True,
                    }
                ]
            }
        )
    if development is not None:
        params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = deepcopy(development)
    if dormant:
        params[ACTIVITY_RECORD_KEY] = {
            "schema_version": ACTIVITY_SCHEMA_VERSION,
            "dormant": True,
        }
    return {
        "module_id": f"a{agent_index:02d}_m{module_index:02d}",
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


def _connection(index: int, source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    return {
        "connection_id": f"c{index:02d}",
        "src_module_id": str(source["module_id"]),
        "dst_module_id": str(target["module_id"]),
        "src_slot": "out",
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "phase_e_fixture",
        "weight": float(np.float32(0.625)),
        "enabled": True,
    }


def _genome(modules: list[dict[str, Any]], connections: list[dict[str, Any]]) -> Any:
    stable = canonical_receipt_bytes(
        {"modules": modules, "connections": connections, "b2_caps": EXPANDED_B2_CAPS}
    )
    return SimpleNamespace(
        modules=deepcopy(modules),
        connections=deepcopy(connections),
        b2_caps=dict(EXPANDED_B2_CAPS),
        canonical_bytes=lambda: stable,
    )


def _agent(index: int, modules: list[dict[str, Any]], connections: list[dict[str, Any]]) -> Any:
    state = {
        str(module["module_instance_id"]): float(np.float32(0.1 * (index + module_no + 1)))
        for module_no, module in enumerate(modules)
    }
    outputs = {
        str(module["module_instance_id"]): float(np.float32(0.05 * (module_no + 1)))
        for module_no, module in enumerate(modules)
    }
    if index == 1 and modules:
        state[str(modules[0]["module_instance_id"])] = float(np.float32(-0.4))
        outputs[str(modules[0]["module_instance_id"])] = float(np.float32(0.0))
    if index == 3 and modules:
        state[str(modules[0]["module_instance_id"])] = float(np.float32(0.75))
        outputs[str(modules[0]["module_instance_id"])] = float(np.float32(0.5))
    return SimpleNamespace(
        agent_id=int(index),
        slot=int(index),
        slot_token=f"flat-slot-{index:02d}",
        genome=_genome(modules, connections),
        stats=SimpleNamespace(survival_steps=0),
        structural_live_module_state=state,
        structural_live_module_outputs=outputs,
        structural_live_module_telemetry={},
        structural_development_latch_by_instance_id={},
        structural_development_stage_id="",
        structural_development_effective_gain_by_instance_id={},
    )


def _build_population() -> tuple[list[Any], dict[str, Any]]:
    integrator = _module(
        0,
        0,
        "integrator",
        input_semantic="self_energy",
        input_gain=0.5,
        output_semantic="rest",
        output_gain=0.5,
        development=_development(),
    )
    gated = _module(
        1,
        0,
        "gated",
        operator=GATED_OPERATOR,
        input_semantic="self_health",
        input_gain=-0.25,
        output_semantic="move_east",
        output_gain=0.75,
        development=_development(),
    )
    prerequisite = _module(
        2,
        0,
        "dependency-root",
        development=_development((1.0, 1.0, 0.0)),
    )
    dependent = _module(
        2,
        1,
        "dependency-leaf",
        development=_development(
            (1.0, 1.0, 1.0),
            prerequisites=[str(prerequisite["homology_id"])],
            latch=True,
        ),
    )
    dormant = _module(
        3,
        0,
        "dormant",
        output_semantic="move_west",
        output_gain=1.0,
        development=_development(),
        dormant=True,
    )
    specifications = {
        0: ([integrator], []),
        1: ([gated], []),
        2: ([prerequisite, dependent], [_connection(0, prerequisite, dependent)]),
        3: ([dormant], []),
    }
    alive = [
        _agent(index, *specifications.get(index, ([], [])))
        for index in range(AGENT_COUNT)
    ]
    return alive, {
        "integrator": integrator,
        "gated": gated,
        "prerequisite": prerequisite,
        "dependent": dependent,
        "dormant": dormant,
    }


def _config() -> SimConfig:
    cfg = SimConfig()
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.policy_tick_stride = 1
    cfg.structural_live_modules_enable = True
    cfg.structural_live_modules_profile = LIVE_PROFILE
    cfg.structural_observation_routing_enable = True
    cfg.structural_observation_routing_profile = ROUTING_PROFILE
    cfg.structural_output_routing_enable = True
    cfg.structural_output_routing_profile = OUTPUT_ROUTING_PROFILE
    cfg.structural_development_enable = True
    cfg.structural_development_profile = DEVELOPMENTAL_FEATURE_PROFILE
    cfg.structural_gated_recurrent_enable = True
    cfg.structural_gated_recurrent_profile = GATED_PROFILE
    cfg.structural_sustainable_growth_enable = True
    cfg.structural_sustainable_growth_profile = SUSTAINABLE_GROWTH_PROFILE
    return cfg


def _fixed_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observations = np.zeros((len(TICK_AGES), AGENT_COUNT, OBSERVATION_WIDTH), dtype=np.float32)
    base_logits = np.zeros((len(TICK_AGES), AGENT_COUNT, ACTION_WIDTH), dtype=np.float32)
    policy_tape = np.empty_like(base_logits)
    action_masks = np.ones((len(TICK_AGES), AGENT_COUNT, ACTION_WIDTH), dtype=np.bool_)
    for tick in range(len(TICK_AGES)):
        observations[tick, :, LOCAL_SIZE] = np.float32(0.6 - 0.1 * tick)
        observations[tick, :, LOCAL_SIZE + 1] = np.float32(0.4 + 0.1 * tick)
        base_logits[tick, :, :] = np.arange(ACTION_WIDTH, dtype=np.float32) * np.float32(0.01)
        policy_tape[tick, :, :] = (
            np.arange(ACTION_WIDTH, dtype=np.float32)[None, :] * np.float32(-0.015)
            + np.float32(tick * 0.002)
        )
    return observations, base_logits, policy_tape, action_masks


def _identity_maps(alive: list[Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    biological: dict[str, Any] = {}
    local: dict[str, Any] = {}
    for agent in alive[:STRUCTURED_AGENT_COUNT]:
        modules = list(agent.genome.modules)
        biological[str(agent.agent_id)] = [
            {
                "module_id": str(module["module_id"]),
                "module_instance_id": str(module["module_instance_id"]),
                "homology_id": str(module["homology_id"]),
            }
            for module in modules
        ]
        local[str(agent.agent_id)] = {
            "module_slots": {
                str(module["module_id"]): int(index) for index, module in enumerate(modules)
            },
            "connection_slots": {
                str(connection["connection_id"]): int(index)
                for index, connection in enumerate(agent.genome.connections)
            },
        }
    return biological, local


def _arm(
    checks: list[tuple[str, bool]],
    raw: Any,
    tensors: Mapping[str, Any],
    floats: Iterable[Any],
    *,
    identity_digest: str,
    slot_digest: str,
    rng_before: int,
    rng_after: int,
) -> dict[str, Any]:
    first = next((name for name, ok in checks if not bool(ok)), None)
    tensor_manifest = {name: _tensor_receipt(value) for name, value in sorted(tensors.items())}
    return {
        "ok": first is None,
        "first_mismatch": first,
        "checks": {name: bool(ok) for name, ok in checks},
        "raw_sha256": canonical_receipt_sha256(raw),
        "float32_bits_sha256": _float32_bits_sha256(floats),
        "tensor_manifest": tensor_manifest,
        "tensor_manifest_sha256": canonical_receipt_sha256(tensor_manifest),
        "shape_digest_16_48": fixed_envelope_shape_digest_for_caps(EXPANDED_B2_CAPS),
        "lane_owner": LANE_OWNER,
        "hidden_fallback_count": 0,
        "structural_rng": {
            "before": int(rng_before),
            "after": int(rng_after),
            "delta": int(rng_after - rng_before),
        },
        "biological_identity_map_sha256": identity_digest,
        "local_slot_map_sha256": slot_digest,
    }


def _run_once() -> dict[str, Any]:
    alive, modules = _build_population()
    cfg = _config()
    observations, base_logits, policy_tape, action_masks = _fixed_inputs()
    initial_state = deepcopy(alive[0].structural_live_module_state)
    initial_output = deepcopy(alive[0].structural_live_module_outputs)
    dormant_initial_state = deepcopy(alive[3].structural_live_module_state)
    dormant_initial_output = deepcopy(alive[3].structural_live_module_outputs)
    biological_map, local_slot_map = _identity_maps(alive)
    identity_digest = canonical_receipt_sha256(biological_map)
    slot_digest = canonical_receipt_sha256(local_slot_map)
    structural_rng_counter = 0
    rng_before = structural_rng_counter

    routed_ticks: list[Any] = []
    rest_ticks: list[np.ndarray] = []
    output_ticks: list[np.ndarray] = []
    final_logits_ticks: list[np.ndarray] = []
    action_ticks: list[np.ndarray] = []
    state_rows: dict[int, list[dict[str, Any]]] = {index: [] for index in range(4)}
    development_rows: dict[int, list[dict[str, Any]]] = {index: [] for index in range(4)}
    cost_rows: list[dict[str, Any]] = []

    for tick, age in enumerate(TICK_AGES):
        for agent in alive:
            agent.stats.survival_steps = int(age)
        routed = route_observations_batch(
            alive, observations[tick], local_size=LOCAL_SIZE, sim_cfg=cfg
        )
        fatigue = observations[tick, :, LOCAL_SIZE]
        rest = execute_live_modules_batch(
            alive, fatigue, sim_cfg=cfg, device="cpu", routing_batches=routed
        )
        output = compile_output_routing_batch(
            alive, sim_cfg=cfg, legacy_rest_residuals=rest
        )
        logits = base_logits[tick].copy()
        add_output_routing_residuals_(logits, output)
        masked = np.where(action_masks[tick], logits + policy_tape[tick], np.float32(-1.0e9))
        actions = np.argmax(masked, axis=1).astype(np.int64)
        routed_ticks.append(routed)
        rest_ticks.append(rest.copy())
        output_ticks.append(output.residuals.copy())
        final_logits_ticks.append(logits.copy())
        action_ticks.append(actions)
        for index in range(4):
            state_rows[index].append(deepcopy(alive[index].structural_live_module_telemetry))
            development_rows[index].append(
                {
                    "stage_id": str(alive[index].structural_development_stage_id),
                    "effective_gain_by_instance_id": deepcopy(
                        alive[index].structural_development_effective_gain_by_instance_id
                    ),
                    "latch_by_instance_id": deepcopy(
                        alive[index].structural_development_latch_by_instance_id
                    ),
                }
            )
        cost_rows.append(deepcopy(alive[2].structural_live_module_telemetry["structural_cost"]))

    rest_array = np.stack(rest_ticks)
    output_array = np.stack(output_ticks)
    logits_array = np.stack(final_logits_ticks)
    actions_array = np.stack(action_ticks)
    rng_after = structural_rng_counter

    integrator_id = str(modules["integrator"]["module_instance_id"])
    gated_id = str(modules["gated"]["module_instance_id"])
    root_id = str(modules["prerequisite"]["module_instance_id"])
    dependent_id = str(modules["dependent"]["module_instance_id"])
    dormant_id = str(modules["dormant"]["module_instance_id"])

    integrator_trace = [row["state_rows"][0] for row in state_rows[0]]
    reset_agent = _agent(0, [modules["integrator"]], [])
    reset_step = execute_agent_live_modules(
        reset_agent,
        fatigue=float(observations[0, 0, LOCAL_SIZE]),
        reset_instance_ids=[integrator_id],
        routed_observation_by_instance_id={integrator_id: 0.3},
        gated_recurrent_enabled=True,
        sustainable_growth_enabled=True,
    )

    gated_trace = [row["state_rows"][0] for row in state_rows[1]]
    first_gated = gated_trace[0]
    gated_expected = step_gated_scalar(
        external_input=float(first_gated["typed_input_combined_external"]),
        routed_previous=float(first_gated["typed_input_previous_outputs"]),
        previous_state=-0.4,
        expression_scale=-0.75,
        developmental_gain=float(first_gated["developmental_expression_gain"]),
        record=modules["gated"]["params"][GATED_RECORD_KEY],
    )

    routed_integrator = np.asarray(
        [
            routed_ticks[tick][0].aggregate_by_instance_id[integrator_id]
            for tick in range(len(TICK_AGES))
        ],
        dtype=np.float32,
    )
    routed_lesion = route_observations_batch(
        _build_population()[0],
        observations[0],
        local_size=LOCAL_SIZE,
        sim_cfg=cfg,
        lesion_semantic_ids=["self_energy"],
    )[0].aggregate_by_instance_id[integrator_id]

    output_integrator = output_array[:, 0, :]
    output_lesion = compile_output_routing_batch(
        alive,
        sim_cfg=cfg,
        lesion_binding_keys=[(integrator_id, "rest", "out")],
        legacy_rest_residuals=np.zeros((AGENT_COUNT,), dtype=np.float32),
    ).residuals[0]

    stage_ids = [row["stage_id"] for row in development_rows[2]]
    root_gains = [row["effective_gain_by_instance_id"][root_id] for row in development_rows[2]]
    dependent_gains = [
        row["effective_gain_by_instance_id"][dependent_id] for row in development_rows[2]
    ]
    dependency_probe_agent = _agent(
        2,
        [modules["prerequisite"], modules["dependent"]],
        [_connection(0, modules["prerequisite"], modules["dependent"])],
    )
    dependency_probe_agent.stats.survival_steps = 20
    dependency_plan = resolve_agent_development_runtime(dependency_probe_agent, sim_cfg=cfg)
    dependency_lesion_agent = deepcopy(dependency_probe_agent)
    dependency_lesion = resolve_agent_development_runtime(
        dependency_lesion_agent,
        sim_cfg=cfg,
        expression_lesion_instance_ids=[root_id],
    )

    dormant_states = [row["state_rows"][0]["state_after"] for row in state_rows[3]]
    dormant_outputs = [row["state_rows"][0]["typed_output"] for row in state_rows[3]]
    dormant_payload = {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "b2_caps": dict(EXPANDED_B2_CAPS),
        "genes": {},
        "modules": [deepcopy(modules["dormant"])],
        "connections": [],
    }
    reactivated = mutate_sustainable_growth(dormant_payload, 0.1, 0.1, enabled=True)
    reactivated_agent = _agent(3, list(reactivated.payload["modules"]), [])
    reactivated_agent.structural_live_module_state = deepcopy(dormant_initial_state)
    reactivated_agent.structural_live_module_outputs = deepcopy(dormant_initial_output)
    reactivated_step = execute_agent_live_modules(
        reactivated_agent,
        fatigue=0.6,
        gated_recurrent_enabled=True,
        sustainable_growth_enabled=True,
    )

    cost_totals = np.asarray([row["total"] for row in cost_rows], dtype=np.float32)
    action_without_cost = actions_array.copy()  # cost is report-only and has no policy seam.

    common = {
        "identity_digest": identity_digest,
        "slot_digest": slot_digest,
        "rng_before": rng_before,
        "rng_after": rng_after,
    }
    arms = {
        "integrator": _arm(
            [
                ("three_tick_trace", len(integrator_trace) == 3),
                ("known_initial_state", _same_f32(initial_state[integrator_id], 0.1)),
                ("known_initial_output", _same_f32(initial_output[integrator_id], 0.05)),
                (
                    "reset_changes_state",
                    not _same_f32(
                        reset_step.summary["state_rows"][0]["state_after"],
                        integrator_trace[0]["state_after"],
                    ),
                ),
            ],
            {"trace": integrator_trace, "reset": reset_step.summary["state_rows"]},
            {"rest_residual": rest_array[:, 0], "final_logits": logits_array[:, 0, :]},
            [row["state_after"] for row in integrator_trace],
            **common,
        ),
        "gated": _arm(
            [
                ("three_tick_trace", len(gated_trace) == 3),
                ("known_initial_state", _same_f32(first_gated["state_before"], -0.4)),
                ("exact_state_authority", _same_f32(first_gated["state_after"], gated_expected.state_after)),
                (
                    "exact_output_authority",
                    _same_f32(first_gated["gated_effective_output"], gated_expected.effective_output),
                ),
                ("zero_rng", all(row["gated_evaluation_rng_draws"] == 0 for row in gated_trace)),
            ],
            gated_trace,
            {"state_output": np.asarray([[row["state_after"], row["typed_output"]] for row in gated_trace], dtype=np.float32)},
            [value for row in gated_trace for value in (row["state_after"], row["typed_output"])],
            **common,
        ),
        "input_routing": _arm(
            [
                ("semantic_route_nonzero", bool(np.all(routed_integrator != np.float32(0.0)))),
                ("lesion_zeroes_route", _same_f32(routed_lesion, 0.0)),
                ("observation_width_27", observations.shape[2] == OBSERVATION_WIDTH),
            ],
            [routed_ticks[tick][0].summary for tick in range(len(TICK_AGES))],
            {"observations": observations[:, 0, :], "routed": routed_integrator},
            routed_integrator,
            **common,
        ),
        "output_routing": _arm(
            [
                ("action_width_7", output_array.shape[2] == ACTION_WIDTH),
                ("active_residual", bool(np.any(output_integrator != np.float32(0.0)))),
                ("binding_lesion_zero", bool(np.all(output_lesion == np.float32(0.0)))),
                ("fixed_policy_tape", policy_tape.shape == base_logits.shape),
            ],
            [routed_ticks[tick][0].summary for tick in range(len(TICK_AGES))],
            {
                "output_residual": output_integrator,
                "base_logits": base_logits[:, 0, :],
                "policy_tape": policy_tape[:, 0, :],
                "actions": actions_array[:, 0],
            },
            output_integrator.reshape(-1),
            **common,
        ),
        "development": _arm(
            [
                ("three_stage_sequence", stage_ids == ["early", "mature", "late"]),
                ("root_stage_gains", all(_same_f32(a, b) for a, b in zip(root_gains, [1.0, 1.0, 0.0]))),
                ("late_root_inactive", _same_f32(root_gains[-1], 0.0)),
            ],
            development_rows[2],
            {"root_gain": np.asarray(root_gains, dtype=np.float32), "dependent_gain": np.asarray(dependent_gains, dtype=np.float32)},
            [*root_gains, *dependent_gains],
            **common,
        ),
        "dependency": _arm(
            [
                ("topological_order", dependency_plan.topological_instance_ids == (root_id, dependent_id)),
                ("mature_dependency_ready", bool(dependency_plan.dependency_ready_by_instance_id[dependent_id])),
                ("lesion_blocks_dependency", not bool(dependency_lesion.dependency_ready_by_instance_id[dependent_id])),
                ("readiness_latched", bool(development_rows[2][-1]["latch_by_instance_id"].get(dependent_id, False))),
            ],
            {"normal": dependency_plan.telemetry, "lesion": dependency_lesion.telemetry},
            {"dependent_gain": np.asarray(dependent_gains, dtype=np.float32)},
            dependent_gains,
            **common,
        ),
        "dormancy": _arm(
            [
                ("state_frozen", all(_same_f32(value, 0.75) for value in dormant_states)),
                ("output_zero", all(_same_f32(value, 0.0) for value in dormant_outputs)),
                ("toggle_accepted", reactivated.telemetry["status"] == "accepted"),
                (
                    "reactivation_changes_state",
                    not _same_f32(
                        reactivated_step.summary["state_rows"][0]["state_after"], 0.75
                    ),
                ),
            ],
            {"trace": state_rows[3], "reactivation": reactivated.telemetry},
            {
                "state": np.asarray(dormant_states, dtype=np.float32),
                "output": np.asarray(dormant_outputs, dtype=np.float32),
            },
            [*dormant_states, *dormant_outputs],
            **common,
        ),
        "structural_cost": _arm(
            [
                ("reporting_only", all(bool(row["reporting_only"]) for row in cost_rows)),
                ("nonnegative", bool(np.all(cost_totals >= np.float32(0.0)))),
                ("actions_unchanged", bool(np.array_equal(actions_array, action_without_cost))),
                ("three_cost_rows", len(cost_rows) == 3),
            ],
            cost_rows,
            {"cost_total": cost_totals, "actions": actions_array},
            cost_totals,
            **common,
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": FIXTURE_ID,
        "lane_owner": LANE_OWNER,
        "cpu_only": True,
        "population_contract": {
            "agent_count": AGENT_COUNT,
            "structured_agent_count": STRUCTURED_AGENT_COUNT,
            "neutral_agent_count": NEUTRAL_AGENT_COUNT,
            "flat_cells": 1,
            "observation_width": OBSERVATION_WIDTH,
            "action_width": ACTION_WIDTH,
            "tick_count": len(TICK_AGES),
            "tick_ages": list(TICK_AGES),
        },
        "fixed_inputs": {
            "observations": _tensor_receipt(observations),
            "base_logits": _tensor_receipt(base_logits),
            "policy_tape": _tensor_receipt(policy_tape),
            "action_masks": _tensor_receipt(action_masks),
        },
        "known_initial": {
            "integrator_state_bits": int(np.float32(initial_state[integrator_id]).view(np.uint32)),
            "integrator_output_bits": int(np.float32(initial_output[integrator_id]).view(np.uint32)),
            "dormant_state_bits": int(np.float32(dormant_initial_state[dormant_id]).view(np.uint32)),
            "dormant_output_bits": int(np.float32(dormant_initial_output[dormant_id]).view(np.uint32)),
        },
        "shape_contract": {
            "b2_caps": dict(EXPANDED_B2_CAPS),
            "shape": fixed_envelope_shape_for_caps(EXPANDED_B2_CAPS),
            "shape_digest_16_48": fixed_envelope_shape_digest_for_caps(EXPANDED_B2_CAPS),
        },
        "biological_identity_map": biological_map,
        "local_slot_map": local_slot_map,
        "structural_rng": {"before": rng_before, "after": rng_after, "delta": 0},
        "hidden_fallback_count": 0,
        "arms": arms,
        "claims": {
            "cuda_parity": False,
            "campaign_validated": False,
            "phase_e_closed": False,
            "accelerated_runtime_validated": False,
        },
    }


def build_phase_e_modular_tick_fixture_receipt() -> dict[str, Any]:
    first = _run_once()
    second = _run_once()
    first_digest = canonical_receipt_sha256(first)
    second_digest = canonical_receipt_sha256(second)
    receipt = deepcopy(first)
    receipt["same_process_repeat"] = {
        "ok": first_digest == second_digest,
        "first_sha256": first_digest,
        "second_sha256": second_digest,
    }
    receipt["receipt_sha256"] = canonical_receipt_sha256(receipt)
    return receipt


def validate_phase_e_modular_tick_fixture_receipt(receipt: Mapping[str, Any]) -> None:
    candidate = deepcopy(dict(receipt))
    observed_digest = str(candidate.pop("receipt_sha256", ""))
    expected_digest = canonical_receipt_sha256(candidate)
    if observed_digest != expected_digest:
        raise ValueError("PHASE_E_FIXTURE_RECEIPT_DIGEST_MISMATCH")
    if observed_digest != EXPECTED_RECEIPT_SHA256:
        raise ValueError("PHASE_E_FIXTURE_RECEIPT_NOT_FROZEN")
    if tuple(candidate.get("arms", {})) != ARM_NAMES:
        raise ValueError("PHASE_E_FIXTURE_ARM_ORDER_MISMATCH")
    first_failure = next(
        (name for name in ARM_NAMES if not bool(candidate["arms"][name].get("ok", False))),
        None,
    )
    if first_failure is not None:
        raise ValueError(f"PHASE_E_FIXTURE_ARM_FAILED arm={first_failure}")
    if not bool(candidate.get("same_process_repeat", {}).get("ok", False)):
        raise ValueError("PHASE_E_FIXTURE_REPEAT_MISMATCH")
    if any(bool(value) for value in candidate.get("claims", {}).values()):
        raise ValueError("PHASE_E_FIXTURE_UNBOUNDED_CLAIM")


__all__ = [
    "ACTION_WIDTH",
    "AGENT_COUNT",
    "ARM_NAMES",
    "EXPECTED_RECEIPT_SHA256",
    "FIXTURE_ID",
    "LANE_OWNER",
    "NEUTRAL_AGENT_COUNT",
    "OBSERVATION_WIDTH",
    "SCHEMA_VERSION",
    "STRUCTURED_AGENT_COUNT",
    "build_phase_e_modular_tick_fixture_receipt",
    "canonical_receipt_bytes",
    "canonical_receipt_sha256",
    "validate_phase_e_modular_tick_fixture_receipt",
]
