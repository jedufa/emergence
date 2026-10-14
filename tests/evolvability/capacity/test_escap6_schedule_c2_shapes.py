from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from agents.agent_policy import MLPPolicy
from agents.gru_policy import GRUPolicy
from agents.modular_genome_v1 import B2_FROZEN_CAPS
from agents.structural_identity_v1 import new_native_identity
from env.gridworld_pkg.constants import N_ACTIONS
from evolution.operators_c1 import apply_operator_pipeline
from evolution.recombination_c2 import recombine_lineage_payloads
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
    resolve_developmental_expression,
    validate_agent_developmental_graph,
)
from evolution.structural_observation_routing_v1 import (
    MAX_BINDINGS_PER_AGENT,
    MAX_BINDINGS_PER_MODULE,
    ROUTING_BLOCK_KEY,
    ROUTING_REGISTRY_VERSION,
    ROUTING_SCHEMA_VERSION,
)
from evolution.structural_output_routing_v1 import (
    ACTION_SEMANTIC_REGISTRY_VERSION,
    MAX_OUTPUT_BINDINGS_PER_AGENT,
    MAX_OUTPUT_BINDINGS_PER_MODULE,
    OUTPUT_ROUTING_BLOCK_KEY,
    OUTPUT_ROUTING_SCHEMA_VERSION,
)
from simulation.checkpoint_bundle_v2 import _build_metadata
from simulation.runner_pkg.runner_config import SimConfig


class _SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self.draw_count = 0

    def random(self) -> float:
        self.draw_count += 1
        return float(next(self._values))


def _identity(index: int) -> dict:
    return new_native_identity(
        context={
            "campaign_id": "escap6-expanded-test",
            "branch_id": "developmental-growth",
            "generation": 1,
            "tick": 0,
            "child_agent_id": index,
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap6-expanded-lineage",
        },
        operator_id="fixture",
        attempt_index=index,
        operator_role=f"module-{index}",
    )


def _development(
    gains: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    prerequisites: tuple[str, ...] = (),
    latch: bool = False,
) -> dict:
    return {
        "schema_version": DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
        "stage_profile_version": DEVELOPMENTAL_STAGE_PROFILE_VERSION,
        "stage_gains": list(gains),
        "prerequisite_homology_ids": list(prerequisites),
        "latch_dependency_readiness": bool(latch),
    }


def _module(
    index: int,
    *,
    identity: dict | None = None,
    development: dict | None = None,
    observation_binding: bool = False,
    output_binding: bool = False,
) -> dict:
    stable_identity = deepcopy(identity if identity is not None else _identity(index))
    params: dict = {
        "live_operator": "bounded_scalar_integrator_v1",
        "expression_scale": 1.0,
        "live_retention": 0.5,
        "live_input_gain": 1.0,
        "live_output_gain": 1.0,
    }
    if development is not None:
        params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = deepcopy(development)
    if observation_binding:
        params[ROUTING_BLOCK_KEY] = {
            "schema_version": ROUTING_SCHEMA_VERSION,
            "registry_version": ROUTING_REGISTRY_VERSION,
            "bindings": [
                {
                    "semantic_id": "self_energy",
                    "target_slot": "in",
                    "gain": 0.5,
                    "enabled": True,
                }
            ],
        }
    if output_binding:
        params[OUTPUT_ROUTING_BLOCK_KEY] = {
            "schema_version": OUTPUT_ROUTING_SCHEMA_VERSION,
            "registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
            "bindings": [
                {
                    "semantic_id": "rest",
                    "source_slot": "out",
                    "gain": 0.5,
                    "enabled": True,
                }
            ],
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
        **stable_identity,
        "latent_pos": list(stable_identity["topology_pos"]),
    }


def _connection(index: int, source: dict, target: dict) -> dict:
    return {
        "connection_id": f"c{index:03d}",
        "src_module_id": source["module_id"],
        "src_slot": "out",
        "dst_module_id": target["module_id"],
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "generic",
        "weight": 0.25,
        "enabled": True,
    }


def _payload(modules: list[dict], connections: list[dict] | None = None) -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": deepcopy(modules),
        "connections": deepcopy(connections or []),
    }


def _repack(payload: dict, order: list[int], *, prefix: str) -> dict:
    chosen = [deepcopy(payload["modules"][index]) for index in order]
    old_to_new: dict[str, str] = {}
    for index, module in enumerate(chosen):
        old_id = str(module["module_id"])
        new_id = f"{prefix}{index:02d}"
        module["module_id"] = new_id
        old_to_new[old_id] = new_id
    connections: list[dict] = []
    for raw in payload.get("connections", []):
        connection = deepcopy(raw)
        connection["src_module_id"] = old_to_new[str(connection["src_module_id"])]
        connection["dst_module_id"] = old_to_new[str(connection["dst_module_id"])]
        connections.append(connection)
    return {**deepcopy(payload), "modules": chosen, "connections": connections}


def _metadata(cfg: object) -> dict:
    return _build_metadata(
        SimpleNamespace(sim_cfg=cfg),
        7,
        True,
        "world",
        "population",
        "rng",
        "runner",
        "bank",
        "resident",
        "native",
        "events",
        "wal",
    )


def test_disabled_escap6_is_exactly_absent_from_c1_schedule_draws_genome_and_telemetry() -> None:
    module = _module(1, observation_binding=True, output_binding=True)
    parent = _payload([module])
    # Schedule promoter mutation, then execute the already-existing ESCAP-4
    # and ESCAP-5 attempts.  An explicitly disabled ESCAP-6 lane must not add
    # even one draw or one telemetry key to this historical path.
    draws = [0.95, 0.2, 0.4, 0.3, 0.6, 0.7, 0.8]
    historical_rng = _SequenceRng(draws)
    historical, historical_telemetry = apply_operator_pipeline(
        payload=deepcopy(parent),
        rng=historical_rng,
        max_ops=1,
        observation_routing_mutation_enable=True,
        observation_routing_operator_attempts=1,
        output_routing_mutation_enable=True,
        output_routing_operator_attempts=1,
    )
    disabled_rng = _SequenceRng(draws)
    disabled, disabled_telemetry = apply_operator_pipeline(
        payload=deepcopy(parent),
        rng=disabled_rng,
        max_ops=1,
        observation_routing_mutation_enable=True,
        observation_routing_operator_attempts=1,
        output_routing_mutation_enable=True,
        output_routing_operator_attempts=1,
        structural_development_mutation_enable=False,
        structural_development_operator_attempts=0,
    )
    assert historical_rng.draw_count == disabled_rng.draw_count == 7
    assert historical_telemetry["schedule_ids"] == disabled_telemetry["schedule_ids"] == [
        "promoter_expression_mutation"
    ]
    assert historical == disabled
    assert historical_telemetry == disabled_telemetry
    assert "structural_development_mutation" not in disabled_telemetry
    assert json.dumps(historical, sort_keys=True, separators=(",", ":")) == json.dumps(
        disabled, sort_keys=True, separators=(",", ":")
    )


def test_disabled_checkpoint_identity_matches_a_pre_escap6_shaped_config() -> None:
    disabled = SimConfig()
    disabled.structural_development_enable = False
    disabled.structural_development_profile = "none"
    pre_escap6 = SimpleNamespace(
        **{
            name: value
            for name, value in vars(disabled).items()
            if not name.startswith("structural_development_")
        }
    )
    disabled_metadata = _metadata(disabled)
    historical_metadata = _metadata(pre_escap6)
    assert "structural_development" not in disabled_metadata
    assert "structural_development" not in historical_metadata
    assert disabled_metadata == historical_metadata
    assert disabled_metadata["checkpoint_id"] == historical_metadata["checkpoint_id"]


def _shared_developmental_parents() -> tuple[dict, dict]:
    identities = [_identity(20 + index) for index in range(3)]
    root_h = identities[0]["homology_id"]
    middle_h = identities[1]["homology_id"]
    modules = [
        _module(20, identity=identities[0], development=_development((1.0, 1.0, 1.0))),
        _module(
            21,
            identity=identities[1],
            development=_development((1.0, 0.5, 1.0), prerequisites=(root_h,)),
        ),
        _module(
            22,
            identity=identities[2],
            development=_development(
                (0.0, 1.0, 1.0), prerequisites=(root_h, middle_h), latch=True
            ),
        ),
    ]
    base = _payload(modules, [_connection(0, modules[0], modules[1])])
    return (
        _repack(base, [2, 0, 1], prefix="a_slot_"),
        _repack(base, [1, 2, 0], prefix="b_slot_"),
    )


def test_c2_rebuilds_chain_and_two_prerequisite_dag_across_parent_reversal_and_repacking() -> None:
    parent_a, parent_b = _shared_developmental_parents()
    rng_ab = _SequenceRng([0.1] * 8)
    child_ab, summary_ab = recombine_lineage_payloads(
        parent_a_payload=parent_a,
        parent_b_payload=parent_b,
        rng=rng_ab,
    )
    rng_ba = _SequenceRng([0.1] * 8)
    child_ba, summary_ba = recombine_lineage_payloads(
        parent_a_payload=parent_b,
        parent_b_payload=parent_a,
        rng=rng_ba,
    )
    assert child_ab == child_ba
    assert summary_ab == summary_ba
    assert summary_ab.matched_modules == 3
    assert summary_ab.matched_connections == 1
    assert rng_ab.draw_count == rng_ba.draw_count == summary_ab.budget_draws == 4
    ordered = validate_agent_developmental_graph(child_ab["modules"])
    dependencies = {
        module["homology_id"]: module["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY][
            "prerequisite_homology_ids"
        ]
        for module in child_ab["modules"]
    }
    assert sorted(len(values) for values in dependencies.values()) == [0, 1, 2]
    position = {module["homology_id"]: index for index, module in enumerate(ordered)}
    for owner, prerequisites in dependencies.items():
        assert all(position[target] < position[owner] for target in prerequisites)
    assert [module["module_id"] for module in child_ab["modules"]] == [
        "m000",
        "m001",
        "m002",
    ]
    valid_slots = {module["module_id"] for module in child_ab["modules"]}
    assert child_ab["connections"][0]["src_module_id"] in valid_slots
    assert child_ab["connections"][0]["dst_module_id"] in valid_slots

    # Repack both parents a second time. Stable homology, rather than either
    # recyclable local slot layout, remains the dependency authority.
    parent_a_twice = _repack(parent_a, [1, 2, 0], prefix="a_again_")
    parent_b_twice = _repack(parent_b, [2, 0, 1], prefix="b_again_")
    repacked_rng = _SequenceRng([0.1] * 8)
    child_repacked, summary_repacked = recombine_lineage_payloads(
        parent_a_payload=parent_a_twice,
        parent_b_payload=parent_b_twice,
        rng=repacked_rng,
    )
    assert child_repacked == child_ab
    assert summary_repacked == summary_ab
    assert repacked_rng.draw_count == 4


@pytest.mark.parametrize(
    ("fault", "message"),
    (("missing", "MISSING_PREREQUISITE"), ("ambiguous", "AMBIGUOUS_HOMOLOGY")),
)
def test_c2_invalid_parent_dependency_targets_fail_before_donor_draws(
    fault: str, message: str
) -> None:
    first = _module(30, development=_development())
    if fault == "missing":
        second = _module(31, development=_development(prerequisites=("missing-target",)))
        parent_a = _payload([first, second])
    else:
        second = _module(31, development=_development())
        second["homology_id"] = first["homology_id"]
        parent_a = _payload([first, second])
    parent_b = _payload([_module(32, development=_development())])
    rng = _SequenceRng([0.1] * 8)
    with pytest.raises(ValueError, match=message):
        recombine_lineage_payloads(
            parent_a_payload=parent_a,
            parent_b_payload=parent_b,
            rng=rng,
        )
    assert rng.draw_count == 0


def _opposed_valid_parent_dags() -> tuple[dict, dict]:
    identity_a = _identity(40)
    identity_b = _identity(41)
    homology_a = identity_a["homology_id"]
    homology_b = identity_b["homology_id"]
    parent_a = _payload(
        [
            _module(
                40,
                identity=identity_a,
                development=_development(prerequisites=(homology_b,)),
            ),
            _module(41, identity=identity_b, development=_development()),
        ]
    )
    parent_b = _payload(
        [
            _module(40, identity=identity_a, development=_development()),
            _module(
                41,
                identity=identity_b,
                development=_development(prerequisites=(homology_a,)),
            ),
        ]
    )
    validate_agent_developmental_graph(parent_a["modules"])
    validate_agent_developmental_graph(parent_b["modules"])
    return parent_a, parent_b


def test_c2_cycle_created_only_by_parent_record_combination_fails_after_fixed_draw_budget() -> None:
    parent_a, parent_b = _opposed_valid_parent_dags()
    cycle_pattern: list[float] | None = None
    for first_draw in (0.1, 0.9):
        for second_draw in (0.1, 0.9):
            rng = _SequenceRng([first_draw, second_draw])
            try:
                recombine_lineage_payloads(
                    parent_a_payload=parent_a,
                    parent_b_payload=parent_b,
                    rng=rng,
                )
            except ValueError as exc:
                if "DEPENDENCY_CYCLE" in str(exc):
                    cycle_pattern = [first_draw, second_draw]
                    assert rng.draw_count == 2
                    break
        if cycle_pattern is not None:
            break
    assert cycle_pattern is not None

    # Canonical parent ordering makes both the failure and the consumed donor
    # budget independent of caller order.
    reversed_rng = _SequenceRng(cycle_pattern)
    with pytest.raises(ValueError, match="DEPENDENCY_CYCLE"):
        recombine_lineage_payloads(
            parent_a_payload=parent_b,
            parent_b_payload=parent_a,
            rng=reversed_rng,
        )
    assert reversed_rng.draw_count == 2


def _shape_digest(shape_record: dict) -> str:
    encoded = json.dumps(shape_record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_development_changes_expression_count_without_resizing_any_storage_or_policy_shape() -> None:
    root = _module(
        50,
        development=_development((1.0, 1.0, 1.0)),
        observation_binding=True,
        output_binding=True,
    )
    child = _module(
        51,
        development=_development(
            (0.0, 1.0, 1.0), prerequisites=(root["homology_id"],)
        ),
        observation_binding=True,
        output_binding=True,
    )
    modules = [root, child]
    connections = [_connection(0, root, child)]
    state_by_instance = {
        root["module_instance_id"]: 0.25,
        child["module_instance_id"]: -0.25,
    }
    early = resolve_developmental_expression(modules, stage_id="early")
    mature = resolve_developmental_expression(modules, stage_id="mature")
    assert sum(row["effective_gain"] > 0.0 for row in early.rows_by_module_instance_id.values()) == 1
    assert sum(row["effective_gain"] > 0.0 for row in mature.rows_by_module_instance_id.values()) == 2

    observation_width = 13
    batch_size = 2
    with torch.random.fork_rng():
        torch.manual_seed(18606)
        mlp = MLPPolicy(obs_size=observation_width, hidden_size=8, n_actions=N_ACTIONS)
        gru = GRUPolicy(obs_size=observation_width, hidden_dim=64, n_actions=N_ACTIONS)
    policy_input = torch.zeros((batch_size, observation_width), dtype=torch.float32)
    gru_hidden = torch.zeros((batch_size, gru.hidden_dim), dtype=torch.float32)
    with torch.inference_mode():
        mlp_output = mlp(policy_input)
        gru_output, gru_hidden_after = gru(policy_input, gru_hidden)

    def record(resolution: object) -> dict:
        return {
            "module_storage": {
                "allocated": len(modules),
                "capacity": B2_FROZEN_CAPS["max_modules_per_agent"],
            },
            "module_state_storage": [len(state_by_instance)],
            "development_runtime_storage": [
                len(resolution.rows_by_module_instance_id),
                len(resolution.latch_state_by_module_instance_id),
            ],
            "connection_storage": {
                "allocated": len(connections),
                "capacity": B2_FROZEN_CAPS["max_connections_per_agent"],
            },
            "input_binding_storage": {
                "per_module": [
                    len(module["params"][ROUTING_BLOCK_KEY]["bindings"]) for module in modules
                ],
                "per_module_capacity": MAX_BINDINGS_PER_MODULE,
                "per_agent_capacity": MAX_BINDINGS_PER_AGENT,
            },
            "output_binding_storage": {
                "per_module": [
                    len(module["params"][OUTPUT_ROUTING_BLOCK_KEY]["bindings"])
                    for module in modules
                ],
                "per_module_capacity": MAX_OUTPUT_BINDINGS_PER_MODULE,
                "per_agent_capacity": MAX_OUTPUT_BINDINGS_PER_AGENT,
            },
            "policy_input_tensor": list(policy_input.shape),
            "mlp_output_tensor": list(mlp_output.shape),
            "gru_output_tensor": list(gru_output.shape),
            "gru_hidden_before": list(gru_hidden.shape),
            "gru_hidden_after": list(gru_hidden_after.shape),
            "gru_weight_tensors": {
                name: list(parameter.shape) for name, parameter in gru.named_parameters()
            },
            "observation_width": observation_width,
            "action_width": N_ACTIONS,
        }

    before = record(early)
    after = record(mature)
    assert before == after
    assert _shape_digest(before) == _shape_digest(after)
    assert before["policy_input_tensor"] == [2, 13]
    assert before["mlp_output_tensor"] == before["gru_output_tensor"] == [2, 7]
    assert before["gru_hidden_before"] == before["gru_hidden_after"] == [2, 64]
    assert before["observation_width"] == 13
    assert before["action_width"] == N_ACTIONS == 7
