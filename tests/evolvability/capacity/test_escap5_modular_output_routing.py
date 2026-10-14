from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import evolution.structural_output_routing_v1 as output_routing_module

from agents.agent import ACTION_VOCABULARY, ACTION_VOCABULARY_SCHEMA_VERSION
from agents.agent_policy import MLPPolicy
from agents.genome import Genome
from agents.gru_policy import GRUPolicy
from agents.structural_identity_v1 import new_native_identity
from env.gridworld_pkg.constants import ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W, ACT_REST, N_ACTIONS
from env.env_config import EnvConfig
from evolution.capacity_contract import build_capacity_contract
from evolution.operators_c1 import apply_operator_pipeline, apply_output_routing_operator_attempt
from evolution.operators_output_routing_v1 import OUTPUT_ROUTING_OPERATOR_DRAWS, mutate_output_bindings
from evolution.policy_family_head_authority_runtime import authority_log_probs_from_logits_torch
from evolution.recombination_c2 import recombine_lineage_payloads
from evolution.structural_live_modules_v1 import LIVE_PROFILE, execute_agent_live_modules
from evolution.structural_output_routing_probe import run_technical_probe
from evolution.structural_output_routing_v1 import (
    ACTION_SEMANTIC_REGISTRY_VERSION,
    ADMITTED_ACTION_SEMANTIC_IDS,
    COMBINED_REST_COMPATIBILITY_VERSION,
    MAX_ACTION_RESIDUAL,
    MAX_OUTPUT_BINDINGS_PER_AGENT,
    MAX_OUTPUT_BINDINGS_PER_MODULE,
    OUTPUT_ROUTING_ARBITRATION_VERSION,
    OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION,
    OUTPUT_ROUTING_BLOCK_KEY,
    OUTPUT_ROUTING_OPERATOR_PROFILE,
    OUTPUT_ROUTING_PROFILE,
    OUTPUT_ROUTING_SCHEMA_VERSION,
    add_output_routing_residuals_,
    compile_agent_output_routing,
    compile_output_routing_batch,
    normalize_agent_output_routing_modules,
    normalize_output_routing_block,
    validate_output_routing_config,
)
from simulation.checkpoint_bundle_v2 import (
    CheckpointStrictError,
    _build_metadata,
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.control_plane_pkg.util_json import canonical_json_bytes
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_action_opportunity import _valid_actions_for_agent
from simulation.runner_pkg.runner_config import SimConfig
from evolution.population import PopConfig


ACTION_INDEX = {
    "move_north": ACT_MOVE_N,
    "move_south": ACT_MOVE_S,
    "move_east": ACT_MOVE_E,
    "move_west": ACT_MOVE_W,
    "rest": ACT_REST,
}


def _identity(child: int, role: str) -> dict:
    return new_native_identity(
        context={
            "campaign_id": "escap5-test",
            "branch_id": "output-routing",
            "generation": 1,
            "tick": 0,
            "child_agent_id": child,
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap5-lineage",
        },
        operator_id="fixture",
        attempt_index=child,
        operator_role=role,
    )


def _binding(semantic_id: str = "move_north", gain: float = 0.5, enabled: bool = True) -> dict:
    return {"semantic_id": semantic_id, "source_slot": "out", "gain": gain, "enabled": enabled}


def _module(mid: str = "m000", *, child: int = 1, bindings: list[dict] | None = None) -> dict:
    identity = _identity(child, mid)
    return {
        "module_id": mid,
        "module_type": "generic",
        "params": {
            "live_operator": "bounded_scalar_integrator_v1",
            "expression_scale": 1.0,
            "live_retention": 0.5,
            "live_input_gain": 1.0,
            "live_output_gain": 0.5,
            OUTPUT_ROUTING_BLOCK_KEY: {
                "schema_version": OUTPUT_ROUTING_SCHEMA_VERSION,
                "registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
                "bindings": deepcopy(bindings or []),
            },
        },
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **deepcopy(identity),
        "latent_pos": list(identity["topology_pos"]),
    }


def _agent(modules: list[dict], outputs: dict[str, float] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        genome=SimpleNamespace(modules=deepcopy(modules), connections=[]),
        structural_live_module_state={},
        structural_live_module_outputs=deepcopy(outputs or {}),
        structural_live_module_telemetry={},
    )


def _cfg(enabled: bool = True) -> SimConfig:
    cfg = SimConfig()
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.policy_tick_stride = 1
    cfg.structural_live_modules_enable = True
    cfg.structural_live_modules_profile = LIVE_PROFILE
    cfg.structural_output_routing_enable = enabled
    cfg.structural_output_routing_profile = OUTPUT_ROUTING_PROFILE if enabled else "none"
    cfg.structural_output_routing_schema_version = OUTPUT_ROUTING_SCHEMA_VERSION
    cfg.structural_output_routing_registry_version = ACTION_SEMANTIC_REGISTRY_VERSION
    cfg.structural_output_routing_max_bindings_per_module = MAX_OUTPUT_BINDINGS_PER_MODULE
    cfg.structural_output_routing_max_bindings_per_agent = MAX_OUTPUT_BINDINGS_PER_AGENT
    cfg.structural_output_routing_operator_profile = OUTPUT_ROUTING_OPERATOR_PROFILE
    cfg.structural_output_routing_operator_attempts = 1 if enabled else 0
    cfg.structural_output_routing_max_binding_gain = 1.0
    cfg.structural_output_routing_max_action_residual = MAX_ACTION_RESIDUAL
    cfg.structural_output_routing_arbitration_version = OUTPUT_ROUTING_ARBITRATION_VERSION
    cfg.structural_output_routing_authority_order_version = OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION
    cfg.structural_output_routing_combined_rest_compatibility_version = COMBINED_REST_COMPATIBILITY_VERSION
    return cfg


def _payload(module: dict) -> dict:
    return {"schema_version": "modular_genome_v1", "b2_invalid_policy": "reject", "genes": {}, "modules": [deepcopy(module)], "connections": []}


def test_action_registry_is_existing_fixed_vocabulary_and_directionally_symmetric() -> None:
    assert ADMITTED_ACTION_SEMANTIC_IDS == ("move_north", "move_south", "move_east", "move_west", "rest")
    assert tuple(ACTION_VOCABULARY) == ("move_north", "move_south", "move_east", "move_west", "gather", "rest", "attack")
    assert ACTION_VOCABULARY_SCHEMA_VERSION == "eiel_action_vocabulary_v1"
    assert {semantic: ACTION_VOCABULARY.index(semantic) for semantic in ADMITTED_ACTION_SEMANTIC_IDS} == ACTION_INDEX
    assert N_ACTIONS == len(ACTION_VOCABULARY) == 7


def test_binding_canonicalization_unknown_duplicate_caps_and_nonfinite_fail_closed() -> None:
    block = normalize_output_routing_block({"bindings": [_binding("rest", -0.25), _binding("move_north", 0.75)]})
    assert [row["semantic_id"] for row in block["bindings"]] == ["move_north", "rest"]
    with pytest.raises(RuntimeError, match="DUPLICATE_BINDING"):
        normalize_output_routing_block({"bindings": [_binding(), _binding()]})
    with pytest.raises(RuntimeError, match="UNKNOWN_ACTION_SEMANTIC_ID"):
        normalize_output_routing_block({"bindings": [_binding("invented_action")]})
    with pytest.raises(RuntimeError, match="ACTION_SEMANTIC_INELIGIBLE"):
        normalize_output_routing_block({"bindings": [_binding("gather")]})
    with pytest.raises(RuntimeError, match="NONFINITE"):
        normalize_output_routing_block({"bindings": [_binding(gain=float("nan"))]})
    with pytest.raises(RuntimeError, match="BINDING_GAIN_OUT_OF_RANGE"):
        normalize_output_routing_block({"bindings": [_binding(gain=1.01)]})
    with pytest.raises(RuntimeError, match="MODULE_BINDING_CAP_EXCEEDED"):
        normalize_output_routing_block({"bindings": [_binding("move_north"), _binding("rest"), _binding("move_south")]})
    with pytest.raises(RuntimeError, match="BINDING_SCHEMA_UNSUPPORTED"):
        normalize_output_routing_block({"schema_version": "unknown_schema", "bindings": []})
    with pytest.raises(RuntimeError, match="REGISTRY_VERSION_MISMATCH"):
        normalize_output_routing_block({"registry_version": "unknown_registry", "bindings": []})


def test_semantic_identity_rejects_deliberately_changed_raw_action_mapping(monkeypatch) -> None:
    reordered = list(ACTION_VOCABULARY)
    reordered[ACT_MOVE_N], reordered[ACT_MOVE_S] = reordered[ACT_MOVE_S], reordered[ACT_MOVE_N]
    monkeypatch.setattr(output_routing_module, "ACTION_VOCABULARY", tuple(reordered))
    module = _module(bindings=[_binding("move_north")])
    with pytest.raises(RuntimeError, match="ACTION_SEMANTIC_INDEX_MISMATCH"):
        compile_agent_output_routing(
            modules=[module],
            outputs_by_instance_id={module["module_instance_id"]: 0.5},
        )


def test_agent_cap_and_nonfinite_or_out_of_range_module_output_reject() -> None:
    modules = [_module(f"m{i}", child=10 + i, bindings=[_binding("move_north"), _binding("rest")]) for i in range(5)]
    with pytest.raises(RuntimeError, match="AGENT_BINDING_CAP_EXCEEDED"):
        normalize_agent_output_routing_modules(modules)
    module = _module(bindings=[_binding()])
    iid = module["module_instance_id"]
    with pytest.raises(RuntimeError, match="NONFINITE"):
        compile_agent_output_routing(modules=[module], outputs_by_instance_id={iid: float("nan")})
    with pytest.raises(RuntimeError, match="MODULE_OUTPUT_OUT_OF_RANGE"):
        compile_agent_output_routing(modules=[module], outputs_by_instance_id={iid: 1.01})


@pytest.mark.parametrize("semantic_id", ADMITTED_ACTION_SEMANTIC_IDS)
def test_one_module_routes_only_to_declared_target(semantic_id: str) -> None:
    module = _module(bindings=[_binding(semantic_id, 0.5)])
    iid = module["module_instance_id"]
    result = compile_agent_output_routing(modules=[module], outputs_by_instance_id={iid: 0.4}, legacy_rest_residual=0.1)
    nonzero = set(np.flatnonzero(result.residual))
    expected = {ACT_REST, ACTION_INDEX[semantic_id]}
    assert nonzero == expected
    assert result.residual[ACTION_INDEX[semantic_id]] == pytest.approx(0.2 if semantic_id != "rest" else 0.25)


def test_two_modules_same_target_and_one_module_two_targets_arbitrate_canonically() -> None:
    a = _module("m000", child=20, bindings=[_binding("move_north", 0.5)])
    b = _module("m001", child=21, bindings=[_binding("move_north", -0.25)])
    result = compile_agent_output_routing(
        modules=[b, a],
        outputs_by_instance_id={a["module_instance_id"]: 0.4, b["module_instance_id"]: 0.4},
        legacy_rest_residual=0.1,
    )
    assert result.residual[ACT_MOVE_N] == pytest.approx(0.1)
    dual = _module(bindings=[_binding("move_west", 0.5), _binding("move_east", -0.25)])
    routed = compile_agent_output_routing(modules=[dual], outputs_by_instance_id={dual["module_instance_id"]: 0.4})
    assert routed.residual[ACT_MOVE_W] == pytest.approx(0.2)
    assert routed.residual[ACT_MOVE_E] == pytest.approx(-0.1)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_cancellation_and_positive_negative_saturation_are_order_invariant(sign: float) -> None:
    a = _module("m000", child=30, bindings=[_binding("move_south", sign)])
    b = _module("m001", child=31, bindings=[_binding("move_south", sign)])
    outputs = {a["module_instance_id"]: 1.0, b["module_instance_id"]: 1.0}
    first = compile_agent_output_routing(modules=[a, b], outputs_by_instance_id=outputs)
    second = compile_agent_output_routing(modules=[b, a], outputs_by_instance_id=outputs)
    assert first.residual.tobytes() == second.residual.tobytes()
    assert first.residual[ACT_MOVE_S] == pytest.approx(sign * MAX_ACTION_RESIDUAL)
    cancel = deepcopy(b)
    cancel["params"][OUTPUT_ROUTING_BLOCK_KEY]["bindings"][0]["gain"] = -sign
    neutral = compile_agent_output_routing(modules=[a, cancel], outputs_by_instance_id=outputs)
    assert neutral.residual[ACT_MOVE_S] == 0.0


def test_binding_lesion_and_state_reset_are_distinct_causal_interventions() -> None:
    module = _module(bindings=[_binding("move_north", 1.0)])
    iid = module["module_instance_id"]
    active = compile_agent_output_routing(modules=[module], outputs_by_instance_id={iid: 0.4})
    lesioned = compile_agent_output_routing(
        modules=[module], outputs_by_instance_id={iid: 0.4}, lesion_binding_keys=[(iid, "move_north", "out")]
    )
    assert active.residual[ACT_MOVE_N] != 0.0 and lesioned.residual[ACT_MOVE_N] == 0.0
    agent = _agent([module])
    agent.structural_live_module_state[iid] = 0.8
    agent.structural_live_module_outputs[iid] = 0.8
    control = deepcopy(agent)
    execute_agent_live_modules(control, fatigue=0.0)
    execute_agent_live_modules(agent, fatigue=0.0, reset_instance_ids=[iid])
    assert control.structural_live_module_state != agent.structural_live_module_state


def test_combined_rest_is_one_final_clip_one_add_and_neutral_equals_legacy_bytes() -> None:
    legacy = np.asarray([0.2], dtype=np.float32)
    module = _module(bindings=[_binding("rest", 1.0)])
    iid = module["module_instance_id"]
    active_agent = _agent([module], {iid: 0.2})
    active = compile_output_routing_batch([active_agent], sim_cfg=_cfg(), legacy_rest_residuals=legacy)
    assert active.residuals[0, ACT_REST] == MAX_ACTION_RESIDUAL
    logits = np.zeros((1, N_ACTIONS), dtype=np.float32)
    add_output_routing_residuals_(logits, active)
    assert logits[0, ACT_REST] == MAX_ACTION_RESIDUAL
    zero = deepcopy(module)
    zero["params"][OUTPUT_ROUTING_BLOCK_KEY]["bindings"][0]["gain"] = 0.0
    neutral = compile_output_routing_batch([_agent([zero], {iid: 0.2})], sim_cfg=_cfg(), legacy_rest_residuals=legacy)
    disabled = compile_output_routing_batch([_agent([zero], {iid: 0.2})], sim_cfg=_cfg(False), legacy_rest_residuals=legacy)
    assert neutral.residuals[:, ACT_REST].tobytes() == legacy.tobytes() == disabled.residuals[:, ACT_REST].tobytes()


def test_mlp_gru_numpy_and_torch_seams_use_same_target_only_convention_and_existing_mask_wins() -> None:
    residual = np.zeros((1, N_ACTIONS), dtype=np.float32)
    residual[0, ACT_MOVE_N] = MAX_ACTION_RESIDUAL
    numpy_logits = np.zeros_like(residual)
    torch_logits = torch.zeros((1, N_ACTIONS), dtype=torch.float32)
    add_output_routing_residuals_(numpy_logits, residual)
    add_output_routing_residuals_(torch_logits, torch.from_numpy(residual))
    assert np.array_equal(numpy_logits, torch_logits.numpy())
    obs = torch.zeros((1, 2), dtype=torch.float32)
    mlp_base = MLPPolicy(obs_size=2, hidden_size=2, n_actions=N_ACTIONS)(obs)
    gru_base, _ = GRUPolicy(obs_size=2, hidden_dim=2, n_actions=N_ACTIONS)(
        obs, torch.zeros((1, 64), dtype=torch.float32)
    )
    for base in (mlp_base, gru_base):
        routed = base.detach().clone()
        add_output_routing_residuals_(routed, torch.from_numpy(residual))
        delta = routed - base
        assert float(delta[0, ACT_MOVE_N].detach().item()) == pytest.approx(MAX_ACTION_RESIDUAL, abs=2e-7)
        assert torch.count_nonzero(delta).item() == 1
    valid = torch.ones_like(torch_logits, dtype=torch.bool)
    valid[:, ACT_MOVE_N] = False
    scores = authority_log_probs_from_logits_torch(torch_logits, torch.zeros((1, 4)), valid_action_mask=valid)
    assert torch.isneginf(scores[0, ACT_MOVE_N])
    with pytest.raises(RuntimeError, match="base_logits"):
        add_output_routing_residuals_(
            np.full((1, N_ACTIONS), np.nan, dtype=np.float32),
            residual,
        )
    with pytest.raises(RuntimeError, match="logit_residuals"):
        add_output_routing_residuals_(
            np.zeros((1, N_ACTIONS), dtype=np.float32),
            np.full((1, N_ACTIONS), np.inf, dtype=np.float32),
        )


@pytest.mark.parametrize(("draw1", "operation"), [(0.1, "add"), (0.3, "remove"), (0.6, "gain"), (0.8, "retarget")])
def test_output_operator_has_fixed_two_draw_symmetric_mapping(draw1: float, operation: str) -> None:
    initial = [] if operation == "add" else [_binding("move_north", 0.5)]
    result = mutate_output_bindings([_module(bindings=initial)], 0.7, draw1, enabled=True)
    assert OUTPUT_ROUTING_OPERATOR_DRAWS == 2
    assert result.telemetry["operation"] == operation
    assert result.telemetry["draw_budget"] == {"declared_supplied_draws": 2, "internal_rng_draws": 0}
    assert result.telemetry["admitted_semantic_ids"] == list(ADMITTED_ACTION_SEMANTIC_IDS)


@pytest.mark.parametrize(
    ("draw0", "semantic_id"),
    [(0.1, "move_north"), (0.3, "move_south"), (0.5, "move_east"), (0.7, "move_west"), (0.9, "rest")],
)
def test_output_operator_admitted_semantic_bins_are_equal_and_reachable(draw0: float, semantic_id: str) -> None:
    result = mutate_output_bindings([_module(bindings=[])], draw0, 0.1, enabled=True)
    assert result.telemetry["status"] == "accepted"
    binding = result.modules[0]["params"][OUTPUT_ROUTING_BLOCK_KEY]["bindings"][0]
    assert binding["semantic_id"] == semantic_id
    assert binding["gain"] == 0.0


class _SequenceRng:
    def __init__(self, values: list[float]):
        self.values = iter(values)
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return float(next(self.values))


class _SeededCountingRng:
    def __init__(self, seed: int):
        self.inner = random.Random(seed)
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return float(self.inner.random())


def test_c1_disabled_schedule_parity_and_enabled_wrapper_two_draws() -> None:
    payload = _payload(_module(bindings=[]))
    legacy_rng = _SequenceRng([])
    legacy = apply_operator_pipeline(payload=deepcopy(payload), rng=legacy_rng, max_ops=0)
    disabled_rng = _SequenceRng([])
    disabled = apply_operator_pipeline(
        payload=deepcopy(payload), rng=disabled_rng, max_ops=0,
        output_routing_mutation_enable=False, output_routing_operator_attempts=0,
    )
    assert legacy == disabled and legacy_rng.draws == disabled_rng.draws == 0
    rng = _SequenceRng([0.1, 0.1])
    result = apply_output_routing_operator_attempt(payload["modules"], rng=rng, enabled=True)
    assert rng.draws == 2 and result.telemetry["status"] == "accepted"


def test_disabled_output_lane_preserves_real_c1_and_escap4_schedule_and_draws() -> None:
    legacy_module = _module(bindings=[])
    legacy_module["params"].pop(OUTPUT_ROUTING_BLOCK_KEY)
    payload = _payload(legacy_module)
    baseline_rng = _SeededCountingRng(51805)
    baseline_payload, baseline_telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=baseline_rng,
        max_ops=3,
        observation_routing_mutation_enable=True,
        observation_routing_operator_attempts=1,
    )
    disabled_rng = _SeededCountingRng(51805)
    disabled_payload, disabled_telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=disabled_rng,
        max_ops=3,
        observation_routing_mutation_enable=True,
        observation_routing_operator_attempts=1,
        output_routing_mutation_enable=False,
        output_routing_operator_attempts=0,
    )
    assert disabled_payload == baseline_payload
    assert disabled_telemetry == baseline_telemetry
    assert disabled_rng.draws == baseline_rng.draws
    assert disabled_telemetry["schedule_ids"] == baseline_telemetry["schedule_ids"]
    assert disabled_telemetry["observation_routing_mutation"] == baseline_telemetry["observation_routing_mutation"]
    assert "output_routing_mutation" not in disabled_telemetry

    enabled_rng = _SeededCountingRng(51805)
    _, enabled_telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=enabled_rng,
        max_ops=3,
        observation_routing_mutation_enable=True,
        observation_routing_operator_attempts=1,
        output_routing_mutation_enable=True,
        output_routing_operator_attempts=1,
    )
    assert enabled_telemetry["schedule_ids"] == baseline_telemetry["schedule_ids"]
    schedule_digests = {
        name: hashlib.sha256(
            json.dumps(rows, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        for name, rows in {
            "legacy": baseline_telemetry["schedule_ids"],
            "disabled": disabled_telemetry["schedule_ids"],
            "enabled": enabled_telemetry["schedule_ids"],
        }.items()
    }
    assert len(set(schedule_digests.values())) == 1
    assert schedule_digests["enabled"] == "d6a8d76a9c17692ba92f8487e042d5112ca16769028073ceaffa022b466a6fd2"
    assert enabled_rng.draws == disabled_rng.draws + OUTPUT_ROUTING_OPERATOR_DRAWS


@pytest.mark.parametrize(
    ("modules", "enabled", "status", "reason"),
    [
        ([_module(bindings=[])], True, "accepted", "binding_mutation_applied"),
        ([_module(bindings=[_binding("move_north")])], True, "rejected", "duplicate_binding"),
        (
            [_module(bindings=[_binding("move_north"), _binding("rest")])],
            True,
            "cap_blocked",
            "module_binding_cap",
        ),
        ([], True, "ineligible", "no_generic_module"),
        ([_module(bindings=[])], False, "disabled", "operator_disabled"),
    ],
)
def test_every_output_attempt_outcome_consumes_two_draws_and_invalid_preserves_parent(
    modules: list[dict], enabled: bool, status: str, reason: str
) -> None:
    canonical_parent = normalize_agent_output_routing_modules(modules)
    rng = _SequenceRng([0.1, 0.1])
    result = apply_output_routing_operator_attempt(deepcopy(modules), rng=rng, enabled=enabled)
    assert rng.draws == 2
    assert result.telemetry["status"] == status
    assert result.telemetry["reason"] == reason
    if status != "accepted":
        assert result.modules == canonical_parent


class _CountingRng:
    def __init__(self):
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return 0.25


def test_c2_inherits_whole_output_binding_record_with_no_extra_draw() -> None:
    identity = _identity(50, "shared")
    a = _module("slot_a", child=50, bindings=[_binding("move_north", 0.25)])
    b = _module("slot_b", child=51, bindings=[_binding("move_west", 0.75)])
    for module in (a, b):
        module.update(deepcopy(identity))
        module["latent_pos"] = list(identity["topology_pos"])
    rng = _CountingRng()
    child, summary = recombine_lineage_payloads(parent_a_payload=_payload(a), parent_b_payload=_payload(b), rng=rng)
    assert summary.budget_draws == rng.draws == 1
    inherited = child["modules"][0]["params"][OUTPUT_ROUTING_BLOCK_KEY]
    assert inherited in (normalize_output_routing_block(a["params"][OUTPUT_ROUTING_BLOCK_KEY]), normalize_output_routing_block(b["params"][OUTPUT_ROUTING_BLOCK_KEY]))


def test_c2_output_schema_or_registry_incompatibility_rejects_before_donor_draw() -> None:
    identity = _identity(60, "shared")
    a = _module("slot_a", child=60, bindings=[_binding("move_north")])
    b = _module("slot_b", child=61, bindings=[_binding("move_west")])
    for module in (a, b):
        module.update(deepcopy(identity))
        module["latent_pos"] = list(identity["topology_pos"])
    b["params"][OUTPUT_ROUTING_BLOCK_KEY]["registry_version"] = "unknown_registry"
    rng = _CountingRng()
    with pytest.raises(RuntimeError, match="REGISTRY_VERSION_MISMATCH"):
        recombine_lineage_payloads(parent_a_payload=_payload(a), parent_b_payload=_payload(b), rng=rng)
    assert rng.draws == 0


def test_capacity_checkpoint_metadata_and_legacy_disabled_representation() -> None:
    contract = build_capacity_contract(sim_cfg=_cfg())
    output = contract["structure"]["output_routing"]
    assert contract["actions"]["output_routing_admitted_semantic_ids"] == list(ADMITTED_ACTION_SEMANTIC_IDS)
    assert output["maximum_bindings_per_module"] == 2 and output["maximum_bindings_per_agent"] == 8
    assert output["execution_rng_draws"] == 0

    def metadata(cfg: SimConfig) -> dict:
        return _build_metadata(
            SimpleNamespace(sim_cfg=cfg), 1, True, "world", "population", "rng", "runner", "bank",
            "resident", "native", "events", "wal",
        )

    enabled = metadata(_cfg())
    disabled = metadata(_cfg(False))
    legacy_shape = SimpleNamespace(
        **{
            key: value
            for key, value in vars(_cfg(False)).items()
            if not key.startswith("structural_output_routing_")
        }
    )
    legacy_metadata = metadata(legacy_shape)
    row = enabled["structural_output_routing"]
    assert row["admitted_semantic_ids"] == list(ADMITTED_ACTION_SEMANTIC_IDS)
    assert row["combined_rest_compatibility_version"] == COMBINED_REST_COMPATIBILITY_VERSION
    assert "structural_output_routing" not in disabled
    assert canonical_json_bytes(disabled) == canonical_json_bytes(legacy_metadata)
    assert disabled["checkpoint_id"] == legacy_metadata["checkpoint_id"]
    assert enabled["checkpoint_id"] != disabled["checkpoint_id"]


def test_strict_checkpoint_missing_or_mismatched_enabled_metadata_rejects_before_apply() -> None:
    cfg = _cfg()
    live_meta = {
        "strict_replay_mode": True,
        "structural_live_modules_enable": True,
        "structural_live_modules_profile": LIVE_PROFILE,
        "structural_live_modules_schema_version": "escap3_live_module_runtime_v1",
    }
    with pytest.raises(CheckpointStrictError, match="metadata missing"):
        checkpoint_bundle_v2_apply(SimpleNamespace(sim_cfg=cfg), {"metadata": live_meta})
    disabled_cfg = _cfg(False)
    with pytest.raises(CheckpointStrictError, match="must be omitted"):
        checkpoint_bundle_v2_apply(
            SimpleNamespace(sim_cfg=disabled_cfg),
            {"metadata": {**live_meta, "structural_output_routing": {
                "enabled": False,
                "profile": "none",
                "schema_version": OUTPUT_ROUTING_SCHEMA_VERSION,
                "registry_version": ACTION_SEMANTIC_REGISTRY_VERSION,
                "admitted_semantic_ids": list(ADMITTED_ACTION_SEMANTIC_IDS),
                "max_bindings_per_module": 2, "max_bindings_per_agent": 8,
                "operator_profile": OUTPUT_ROUTING_OPERATOR_PROFILE, "operator_attempts": 0,
                "max_binding_gain": 1.0, "max_action_residual": 0.25,
                "arbitration_version": OUTPUT_ROUTING_ARBITRATION_VERSION,
                "authority_order_version": OUTPUT_ROUTING_AUTHORITY_ORDER_VERSION,
                "combined_rest_compatibility_version": COMBINED_REST_COMPATIBILITY_VERSION,
                "migration_rng_draws": 0,
            }}},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile", "wrong_profile"),
        ("schema_version", "wrong_schema"),
        ("registry_version", "wrong_registry"),
        ("admitted_semantic_ids", ["move_north", "rest"]),
        ("max_bindings_per_module", 1),
        ("max_bindings_per_agent", 7),
        ("operator_profile", "wrong_operator"),
        ("operator_attempts", 2),
        ("max_binding_gain", 0.5),
        ("max_action_residual", 0.5),
        ("arbitration_version", "wrong_arbitration"),
        ("authority_order_version", "wrong_authority_order"),
        ("combined_rest_compatibility_version", "wrong_rest_rule"),
        ("migration_rng_draws", 1),
    ],
)
def test_enabled_checkpoint_contract_mismatch_rejects_before_continuation(field: str, value: object) -> None:
    cfg = _cfg()
    metadata = _build_metadata(
        SimpleNamespace(sim_cfg=cfg), 1, True, "world", "population", "rng", "runner", "bank",
        "resident", "native", "events", "wal",
    )
    metadata["structural_output_routing"][field] = value
    with pytest.raises(CheckpointStrictError, match="structural_output_routing config mismatch"):
        checkpoint_bundle_v2_apply(SimpleNamespace(sim_cfg=cfg), {"metadata": metadata})


def test_config_preflight_and_reporting_are_isolated_and_probe_is_green() -> None:
    validate_output_routing_config(_cfg(), device="cpu", policy_stride=1)
    with pytest.raises(RuntimeError, match="CPU_REFERENCE_ONLY"):
        validate_output_routing_config(_cfg(), device="cuda")
    with pytest.raises(RuntimeError, match="POLICY_STRIDE_UNSUPPORTED"):
        validate_output_routing_config(_cfg(), policy_stride=2)
    module = _module(bindings=[_binding("move_north", 0.5)])
    iid = module["module_instance_id"]
    agent = _agent([module], {iid: 0.4})
    before = deepcopy((agent.genome.modules, agent.structural_live_module_outputs))
    compile_agent_output_routing(agent, legacy_rest_residual=0.1)
    assert (agent.genome.modules, agent.structural_live_module_outputs) == before
    probe = run_technical_probe()
    assert len(probe["arms"]) == 10
    assert all(probe["checks"].values())


def _real_runner(root, *, output_routing: bool = True) -> SimulationRunner:
    cfg = _cfg(output_routing)
    cfg.seed = 18105
    cfg.generations = 1
    cfg.steps_per_generation = 1
    cfg.behavior_log_enable = False
    cfg.policy_deterministic_actions = True
    cfg.mem_max_live_level4_instrumentation_enable = True
    sim = SimulationRunner(
        EnvConfig(width=8, height=8),
        PopConfig(population_size=4),
        cfg,
        NoOpMetricsLogger(),
        str(root),
        perf=False,
    )
    sim.run(start_gen=0, skip_init=False)
    return sim


def _install_real_module(sim: SimulationRunner, module: dict) -> None:
    for agent in sim.pop.agents.values():
        raw = agent.genome.to_dict()
        raw["modules"] = [deepcopy(module)]
        raw["connections"] = []
        raw["b2_invalid_policy"] = "reject"
        agent.genome = Genome.from_dict(raw)
        agent.structural_live_module_state = {}
        agent.structural_live_module_outputs = {}
        agent.structural_live_module_telemetry = {}
        agent.drives[2] = np.float32(0.6)
        sim._drive_fatigue[int(agent.slot)] = np.float32(0.6)


@pytest.mark.parametrize("semantic_id", ("move_north", "move_south", "move_east", "move_west"))
def test_real_runner_mlp_and_gru_directional_symmetry_before_sampling(tmp_path, semantic_id: str) -> None:
    zero = _real_runner(tmp_path / f"zero-{semantic_id}")
    active = _real_runner(tmp_path / f"active-{semantic_id}")
    zero_module = _module(bindings=[_binding(semantic_id, 0.0)])
    active_module = _module(bindings=[_binding(semantic_id, 1.0)])
    _install_real_module(zero, zero_module)
    _install_real_module(active, active_module)
    zero_alive = [agent for agent in zero.pop.agents.values() if agent.alive]
    active_alive = [agent for agent in active.pop.agents.values() if agent.alive]
    zero_masks = [_valid_actions_for_agent(zero, zero_alive, index) for index in range(len(zero_alive))]
    active_masks = [_valid_actions_for_agent(active, active_alive, index) for index in range(len(active_alive))]
    assert zero_masks == active_masks
    assert [agent.genome.action_temperature for agent in zero_alive] == [
        agent.genome.action_temperature for agent in active_alive
    ]
    assert zero.sim_cfg.policy_family_head_residual_authority_enable is False
    assert active.sim_cfg.policy_family_head_residual_authority_enable is False
    zero.sim_cfg.policy_deterministic_actions = False
    active.sim_cfg.policy_deterministic_actions = False
    zero_rng_before = zero.rng_manager.capture_snapshot(generation_index=zero.gen)
    active_rng_before = active.rng_manager.capture_snapshot(generation_index=active.gen)
    assert zero_rng_before["streams"]["policy"]["indices"] == active_rng_before["streams"]["policy"]["indices"]
    zero._batched_actions(zero_alive)
    active._batched_actions(active_alive)
    zero_rng_after = zero.rng_manager.capture_snapshot(generation_index=zero.gen)
    active_rng_after = active.rng_manager.capture_snapshot(generation_index=active.gen)
    assert zero_rng_after["streams"]["policy"]["indices"] == active_rng_after["streams"]["policy"]["indices"]
    zero_logits = zero._mem_max_live_level4_last_policy["logits_by_agent_id"]
    active_logits = active._mem_max_live_level4_last_policy["logits_by_agent_id"]
    families: set[str] = set()
    for zero_agent, active_agent in zip(zero_alive, active_alive, strict=True):
        assert zero_agent.agent_id == active_agent.agent_id
        families.add("gru" if bool(active_agent.genome.use_gru) else "mlp")
        before = np.asarray(zero_logits[zero_agent.agent_id], dtype=np.float32)
        after = np.asarray(active_logits[active_agent.agent_id], dtype=np.float32)
        delta = after - before
        target_index = ACTION_INDEX[semantic_id]
        assert delta[target_index] == pytest.approx(MAX_ACTION_RESIDUAL, abs=2e-7)
        assert np.delete(delta, target_index).tobytes() == np.zeros((N_ACTIONS - 1,), dtype=np.float32).tobytes()
        assert active_agent.structural_live_module_telemetry["random_draws"] == 0
        telemetry = active_agent.structural_live_module_telemetry["output_routing"]
        assert telemetry["active_bindings"] == 1
        assert telemetry["per_action_residual_totals"][semantic_id] == pytest.approx(MAX_ACTION_RESIDUAL)
        assert semantic_id in telemetry["saturated_action_semantic_ids"]
        assert telemetry["residual_base_logit_l1_ratio"] >= 0.0
        assert telemetry["reporting_only"] is True
    assert families == {"mlp", "gru"}


def test_enabled_checkpoint_roundtrip_continues_state_logits_actions_and_rng(tmp_path) -> None:
    source = _real_runner(tmp_path / "source")
    module = _module(bindings=[_binding("move_north", 0.5), _binding("rest", -0.25)])
    _install_real_module(source, module)
    source_alive = [agent for agent in source.pop.agents.values() if agent.alive]
    source._batched_actions(source_alive)
    bundle = checkpoint_bundle_v2_save(
        source,
        str(tmp_path / "bundle"),
        generation_index=1,
        strict_replay_mode=True,
    )
    loaded = checkpoint_bundle_v2_load(bundle, strict=True)
    metadata = loaded["metadata"]["structural_output_routing"]
    assert metadata["admitted_semantic_ids"] == list(ADMITTED_ACTION_SEMANTIC_IDS)
    assert metadata["combined_rest_compatibility_version"] == COMBINED_REST_COMPATIBILITY_VERSION

    restored = _real_runner(tmp_path / "restored")
    checkpoint_bundle_v2_apply(restored, loaded, restore_rng=True)
    for agent_id, source_agent in source.pop.agents.items():
        restored_agent = restored.pop.agents[agent_id]
        assert restored_agent.genome.modules == source_agent.genome.modules
        assert restored_agent.structural_live_module_state == source_agent.structural_live_module_state
        assert restored_agent.structural_live_module_outputs == source_agent.structural_live_module_outputs

    source_alive = [agent for agent in source.pop.agents.values() if agent.alive]
    restored_alive = [agent for agent in restored.pop.agents.values() if agent.alive]
    assert [
        _valid_actions_for_agent(source, source_alive, index)
        for index in range(len(source_alive))
    ] == [
        _valid_actions_for_agent(restored, restored_alive, index)
        for index in range(len(restored_alive))
    ]

    source_actions = source._batched_actions(source_alive)
    restored_actions = restored._batched_actions(restored_alive)
    assert source_actions.tobytes() == restored_actions.tobytes()
    for agent_id, source_logits in source._mem_max_live_level4_last_policy["logits_by_agent_id"].items():
        restored_logits = restored._mem_max_live_level4_last_policy["logits_by_agent_id"][agent_id]
        assert np.asarray(source_logits, dtype=np.float32).tobytes() == np.asarray(
            restored_logits, dtype=np.float32
        ).tobytes()
    for source_agent, restored_agent in zip(source_alive, restored_alive, strict=True):
        source_output = source_agent.structural_live_module_telemetry["output_routing"]
        restored_output = restored_agent.structural_live_module_telemetry["output_routing"]
        assert source_output["per_action_residual_totals"] == restored_output["per_action_residual_totals"]
        assert source_output["action_rows"] == restored_output["action_rows"]
        rest_row = next(row for row in source_output["action_rows"] if row["semantic_id"] == "rest")
        expected_rest = np.float32(
            np.clip(
                rest_row["legacy_rest_residual"]
                + rest_row["routed_aggregate_before_legacy_rest"],
                -MAX_ACTION_RESIDUAL,
                MAX_ACTION_RESIDUAL,
            )
        )
        assert rest_row["action_residual"] == pytest.approx(float(expected_rest), abs=2e-7)
    source_rng = source.rng_manager.capture_snapshot(generation_index=source.gen)
    restored_rng = restored.rng_manager.capture_snapshot(generation_index=restored.gen)
    assert source_rng["streams"]["policy"]["indices"] == restored_rng["streams"]["policy"]["indices"]


def test_disabled_legacy_checkpoint_omits_escap5_and_first_resume_is_exact(tmp_path) -> None:
    source = _real_runner(tmp_path / "legacy-source", output_routing=False)
    legacy_module = _module(bindings=[])
    legacy_module["params"].pop(OUTPUT_ROUTING_BLOCK_KEY)
    _install_real_module(source, legacy_module)
    source._batched_actions([agent for agent in source.pop.agents.values() if agent.alive])
    bundle = checkpoint_bundle_v2_save(
        source,
        str(tmp_path / "legacy-bundle"),
        generation_index=1,
        strict_replay_mode=True,
    )
    loaded = checkpoint_bundle_v2_load(bundle, strict=True)
    assert "structural_output_routing" not in loaded["metadata"]
    assert all(
        OUTPUT_ROUTING_BLOCK_KEY not in module.get("params", {})
        for agent in source.pop.agents.values()
        for module in agent.genome.modules
    )

    restored = _real_runner(tmp_path / "legacy-restored", output_routing=False)
    checkpoint_bundle_v2_apply(restored, loaded, restore_rng=True)
    source_actions = source._batched_actions(
        [agent for agent in source.pop.agents.values() if agent.alive]
    )
    restored_actions = restored._batched_actions(
        [agent for agent in restored.pop.agents.values() if agent.alive]
    )
    assert source_actions.tobytes() == restored_actions.tobytes()
    for agent_id, source_logits in source._mem_max_live_level4_last_policy["logits_by_agent_id"].items():
        restored_logits = restored._mem_max_live_level4_last_policy["logits_by_agent_id"][agent_id]
        assert np.asarray(source_logits, dtype=np.float32).tobytes() == np.asarray(
            restored_logits, dtype=np.float32
        ).tobytes()
    source_rng = source.rng_manager.capture_snapshot(generation_index=source.gen)
    restored_rng = restored.rng_manager.capture_snapshot(generation_index=restored.gen)
    assert source_rng["streams"]["policy"]["indices"] == restored_rng["streams"]["policy"]["indices"]
