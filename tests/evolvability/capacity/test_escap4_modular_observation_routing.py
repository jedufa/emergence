from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from agents.perception import perception_channel_registry, perception_channel_registry_dict
from agents.genome import Genome
from agents.structural_identity_v1 import new_native_identity
from evolution.capacity_contract import build_capacity_contract
from evolution.operators_c1 import (
    _op_promoter_expression_mutation,
    apply_observation_routing_operator_attempt,
    apply_operator_pipeline,
)
from evolution.operators_routing_v1 import ROUTING_OPERATOR_DRAWS, mutate_observation_bindings
from evolution.recombination_c2 import recombine_lineage_payloads
from evolution.structural_live_modules_v1 import (
    LIVE_PROFILE,
    add_rest_logit_residual_,
    execute_agent_live_modules,
)
from evolution.structural_observation_routing_probe import build_stage0_audit, run_technical_probe
from evolution.structural_observation_routing_v1 import (
    ADMITTED_SEMANTIC_IDS,
    MAX_BINDINGS_PER_AGENT,
    MAX_BINDINGS_PER_MODULE,
    ROUTING_EXPOSURE_MODE,
    ROUTING_PROFILE,
    ROUTING_REGISTRY_VERSION,
    ROUTING_SCHEMA_VERSION,
    compile_module_routing,
    normalize_agent_routing_modules,
    normalize_routing_block,
    resolve_routing_values,
    route_observations_batch,
    validate_routing_config,
)
from simulation.checkpoint_bundle_v2 import _build_metadata
from simulation.checkpoint_bundle_v2 import (
    CheckpointStrictError,
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _identity(child: int, role: str) -> dict:
    return new_native_identity(
        context={
            "campaign_id": "escap4-test",
            "branch_id": "routing",
            "generation": 1,
            "tick": 0,
            "child_agent_id": int(child),
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap4-lineage",
        },
        operator_id="fixture",
        attempt_index=int(child),
        operator_role=str(role),
    )


def _binding(semantic_id: str = "self_energy", *, gain: float = 0.5, enabled: bool = True) -> dict:
    return {
        "semantic_id": semantic_id,
        "target_slot": "in",
        "gain": gain,
        "enabled": enabled,
    }


def _module(
    mid: str = "m000",
    *,
    child: int = 1,
    bindings: list[dict] | None = None,
    retention: float = 0.5,
    input_gain: float = 1.0,
    output_gain: float = 0.5,
) -> dict:
    identity = _identity(child, mid)
    return {
        "module_id": mid,
        "module_type": "generic",
        "params": {
            "live_operator": "bounded_scalar_integrator_v1",
            "expression_scale": 1.0,
            "live_retention": retention,
            "live_input_gain": input_gain,
            "live_output_gain": output_gain,
            "observation_routing": {
                "schema_version": ROUTING_SCHEMA_VERSION,
                "registry_version": ROUTING_REGISTRY_VERSION,
                "bindings": deepcopy(bindings if bindings is not None else [_binding()]),
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


def _agent(modules: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        genome=SimpleNamespace(modules=deepcopy(modules), connections=[]),
        structural_live_module_state={},
        structural_live_module_outputs={},
        structural_live_module_telemetry={},
    )


def _obs(*, local_size: int = 9, energy: float = 0.8, health: float = 0.4) -> np.ndarray:
    row = np.arange(local_size + 2, dtype=np.float32) / np.float32(max(1, local_size + 1))
    row[local_size] = np.float32(energy)
    row[local_size + 1] = np.float32(health)
    return row


def _cfg(*, routing: bool = True) -> SimConfig:
    cfg = SimConfig()
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.policy_tick_stride = 1
    cfg.structural_live_modules_enable = True
    cfg.structural_live_modules_profile = LIVE_PROFILE
    cfg.structural_observation_routing_enable = bool(routing)
    cfg.structural_observation_routing_profile = ROUTING_PROFILE if routing else "none"
    cfg.structural_observation_routing_schema_version = ROUTING_SCHEMA_VERSION
    cfg.structural_observation_routing_registry_version = ROUTING_REGISTRY_VERSION
    cfg.structural_observation_routing_max_bindings_per_module = MAX_BINDINGS_PER_MODULE
    cfg.structural_observation_routing_max_bindings_per_agent = MAX_BINDINGS_PER_AGENT
    cfg.structural_observation_routing_operator_profile = "escap4_routing_binding_operator_v1"
    cfg.structural_observation_routing_operator_attempts = 1 if routing else 0
    return cfg


def _payload(module: dict) -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": [deepcopy(module)],
        "connections": [],
    }


def test_registry_admits_only_existing_self_state_scalars_and_marks_spatial_ineligible() -> None:
    assert ADMITTED_SEMANTIC_IDS == ("self_energy", "self_health")
    registry = perception_channel_registry()
    eligible = tuple(sorted(row.channel_id for row in registry if row.routing_eligible))
    assert eligible == ADMITTED_SEMANTIC_IDS
    spatial = [row for row in registry if row.spatial and row.active_now]
    assert spatial
    assert all(row.value_type == "vector" and row.value_shape == "view_cells" for row in spatial)
    assert all(not row.routing_eligible and row.routing_ineligible_reason for row in spatial)
    machine = perception_channel_registry_dict()
    assert machine["routing_contract_version"] == ROUTING_REGISTRY_VERSION
    assert all("routing_eligible" in row and "routing_ineligible_reason" in row for row in machine["channels"])


def test_binding_canonical_order_and_fail_closed_contracts() -> None:
    block = normalize_routing_block(
        {
            "bindings": [
                _binding("self_health", gain=-0.25),
                _binding("self_energy", gain=0.75),
            ]
        }
    )
    assert [row["semantic_id"] for row in block["bindings"]] == ["self_energy", "self_health"]
    assert block["schema_version"] == ROUTING_SCHEMA_VERSION
    assert block["registry_version"] == ROUTING_REGISTRY_VERSION

    with pytest.raises(RuntimeError, match="DUPLICATE_BINDING"):
        normalize_routing_block({"bindings": [_binding(), _binding()]})
    with pytest.raises(RuntimeError, match="UNKNOWN_SEMANTIC_ID"):
        normalize_routing_block({"bindings": [_binding("unknown_global_metric")]})
    with pytest.raises(RuntimeError, match="SEMANTIC_INELIGIBLE"):
        normalize_routing_block({"bindings": [_binding("local_grid_state")]})
    with pytest.raises(RuntimeError, match="NONFINITE"):
        normalize_routing_block({"bindings": [_binding(gain=float("nan"))]})
    with pytest.raises(RuntimeError, match="BINDING_GAIN_OUT_OF_RANGE"):
        normalize_routing_block({"bindings": [_binding(gain=1.01)]})
    with pytest.raises(RuntimeError, match="MODULE_BINDING_CAP_EXCEEDED"):
        normalize_routing_block({"bindings": [_binding(), _binding("self_health"), _binding()]})


def test_agent_binding_cap_and_target_type_reject() -> None:
    modules = [
        _module(f"m{i:03d}", child=10 + i, bindings=[_binding(), _binding("self_health")])
        for i in range(5)
    ]
    with pytest.raises(RuntimeError, match="AGENT_BINDING_CAP_EXCEEDED"):
        normalize_agent_routing_modules(modules)
    bad_slot = _binding()
    bad_slot["target_slot"] = "raw_obs_index"
    with pytest.raises(RuntimeError, match="TARGET_SLOT_UNSUPPORTED"):
        normalize_routing_block({"bindings": [bad_slot]})


def test_final_vector_resolution_preserves_bytes_and_source_lesion_is_explicit() -> None:
    row = _obs()
    before = row.tobytes()
    resolved = resolve_routing_values(row, 9, ("self_health", "self_energy"))
    assert row.tobytes() == before
    assert resolved["self_energy"]["source_value"] == pytest.approx(0.8)
    assert resolved["self_health"]["resolved_value"] == pytest.approx(0.4)
    assert all(value["direct_policy_exposure"] is True for value in resolved.values())

    lesioned = resolve_routing_values(row, 9, ("self_energy",), lesion_semantic_ids=("self_energy",))
    assert lesioned["self_energy"]["source_value"] == pytest.approx(0.8)
    assert lesioned["self_energy"]["resolved_value"] == 0.0
    assert lesioned["self_energy"]["source_lesioned"] is True
    assert row.tobytes() == before


def test_routing_disabled_and_zero_gain_are_exactly_neutral() -> None:
    zero_module = _module(bindings=[_binding(gain=0.0)])
    zero_agent = _agent([zero_module])
    row = _obs()
    zero_batch = compile_module_routing(
        zero_agent.genome.modules,
        resolve_routing_values(row, 9, ("self_energy",)),
    )
    zero = execute_agent_live_modules(
        zero_agent,
        fatigue=0.0,
        routed_observation_by_instance_id=zero_batch.aggregate_by_instance_id,
        routing_telemetry_by_instance_id=zero_batch.telemetry_by_instance_id,
        routing_summary=zero_batch.summary,
    )
    disabled_agent = _agent([_module()])
    disabled_batches = route_observations_batch(
        [disabled_agent], row.reshape(1, -1), local_size=9, sim_cfg=_cfg(routing=False)
    )
    disabled = execute_agent_live_modules(disabled_agent, fatigue=0.0)
    assert disabled_batches[0].summary == {"execution": "disabled", "random_draws": 0}
    assert zero.rest_residual == disabled.rest_residual == 0.0
    assert zero_agent.structural_live_module_state == disabled_agent.structural_live_module_state
    assert zero.summary["random_draws"] == disabled.summary["random_draws"] == 0
    assert zero_batch.summary["allocated_bindings"] == 1
    assert zero_batch.summary["active_bindings"] == 0


def test_active_routed_input_enters_existing_state_equation() -> None:
    module = _module(retention=0.5, input_gain=1.0, output_gain=0.5)
    agent = _agent([module])
    row = _obs(energy=0.8)
    batch = compile_module_routing(
        agent.genome.modules,
        resolve_routing_values(row, 9, ("self_energy",)),
    )
    assert list(batch.aggregate_by_instance_id.values()) == [pytest.approx(0.4)]
    step0 = execute_agent_live_modules(
        agent,
        fatigue=0.0,
        routed_observation_by_instance_id=batch.aggregate_by_instance_id,
        routing_telemetry_by_instance_id=batch.telemetry_by_instance_id,
        routing_summary=batch.summary,
    )
    state0 = step0.summary["state_rows"][0]
    assert state0["typed_input_routed_observation"] == pytest.approx(0.4)
    assert state0["state_after"] == pytest.approx(0.4)
    assert state0["typed_output"] == pytest.approx(0.2)
    assert step0.rest_residual == pytest.approx(0.2)
    step1 = execute_agent_live_modules(
        agent,
        fatigue=0.0,
        routed_observation_by_instance_id=batch.aggregate_by_instance_id,
        routing_telemetry_by_instance_id=batch.telemetry_by_instance_id,
        routing_summary=batch.summary,
    )
    assert step1.summary["state_rows"][0]["state_after"] == pytest.approx(0.6)


def test_two_binding_cancellation_and_single_aggregate_clipping_are_order_invariant() -> None:
    row = _obs(energy=1.0, health=1.0)
    resolved = resolve_routing_values(row, 9, ADMITTED_SEMANTIC_IDS)

    def batch(gain_energy: float, gain_health: float, *, reverse: bool = False):
        bindings = [
            _binding("self_energy", gain=gain_energy),
            _binding("self_health", gain=gain_health),
        ]
        if reverse:
            bindings.reverse()
        return compile_module_routing([_module(bindings=bindings)], resolved)

    positive = batch(1.0, 1.0)
    positive_reversed = batch(1.0, 1.0, reverse=True)
    negative = batch(-1.0, -1.0)
    cancelled = batch(1.0, -1.0)
    instance_id = next(iter(positive.aggregate_by_instance_id))
    assert positive.telemetry_by_instance_id[instance_id]["aggregate_before_clip"] == 2.0
    assert positive.aggregate_by_instance_id[instance_id] == 1.0
    assert positive.telemetry_by_instance_id[instance_id]["aggregate_clipped"] is True
    assert positive.aggregate_by_instance_id == positive_reversed.aggregate_by_instance_id
    assert positive.telemetry_by_instance_id == positive_reversed.telemetry_by_instance_id
    assert negative.telemetry_by_instance_id[instance_id]["aggregate_before_clip"] == -2.0
    assert negative.aggregate_by_instance_id[instance_id] == -1.0
    assert cancelled.telemetry_by_instance_id[instance_id]["aggregate_before_clip"] == 0.0
    assert cancelled.aggregate_by_instance_id[instance_id] == 0.0
    assert cancelled.telemetry_by_instance_id[instance_id]["aggregate_clipped"] is False

    routed_agent = _agent([_module(bindings=[_binding("self_energy", gain=1.0), _binding("self_health", gain=-1.0)])])
    baseline_agent = _agent([_module(bindings=[])])
    routed = execute_agent_live_modules(
        routed_agent,
        fatigue=0.4,
        routed_observation_by_instance_id=cancelled.aggregate_by_instance_id,
        routing_telemetry_by_instance_id=cancelled.telemetry_by_instance_id,
        routing_summary=cancelled.summary,
    )
    baseline = execute_agent_live_modules(baseline_agent, fatigue=0.4)
    assert routed_agent.structural_live_module_state == baseline_agent.structural_live_module_state
    assert routed.rest_residual == baseline.rest_residual
    logits = np.arange(7, dtype=np.float32).reshape(1, 7)
    changed = logits.copy()
    add_rest_logit_residual_(
        changed,
        np.asarray([positive.aggregate_by_instance_id[instance_id]], dtype=np.float32),
        rest_index=5,
    )
    assert np.delete(changed - logits, 5).tobytes() == np.zeros((6,), dtype=np.float32).tobytes()


def test_source_and_binding_lesions_are_observably_distinct() -> None:
    module = _module()
    instance_id = str(module["module_instance_id"])
    row = _obs(energy=0.8)
    source_batch = compile_module_routing(
        [module],
        resolve_routing_values(row, 9, ("self_energy",), lesion_semantic_ids=("self_energy",)),
    )
    binding_batch = compile_module_routing(
        [module],
        resolve_routing_values(row, 9, ("self_energy",)),
        lesion_binding_keys=((instance_id, "self_energy", "in"),),
    )
    source = source_batch.telemetry_by_instance_id[instance_id]["bindings"][0]
    binding = binding_batch.telemetry_by_instance_id[instance_id]["bindings"][0]
    assert source_batch.aggregate_by_instance_id[instance_id] == 0.0
    assert binding_batch.aggregate_by_instance_id[instance_id] == 0.0
    assert source["source_lesioned"] is True and source["binding_lesioned"] is False
    assert binding["source_lesioned"] is False and binding["binding_lesioned"] is True
    assert source["source_value"] == binding["source_value"] == pytest.approx(0.8)


def test_module_and_binding_repacking_preserve_instance_owned_execution() -> None:
    a = _module(
        "m000",
        child=21,
        bindings=[_binding("self_health", gain=0.25), _binding("self_energy", gain=0.5)],
    )
    b = _module("m001", child=22, bindings=[_binding("self_energy", gain=-0.25)])
    resolved = resolve_routing_values(_obs(energy=0.8, health=0.4), 9, ADMITTED_SEMANTIC_IDS)
    first = compile_module_routing([a, b], resolved)
    a_repacked = deepcopy(a)
    a_repacked["params"]["observation_routing"]["bindings"].reverse()
    second = compile_module_routing([b, a_repacked], resolved)
    assert first.aggregate_by_instance_id == second.aggregate_by_instance_id
    agent_a = _agent([a, b])
    agent_b = _agent([b, a_repacked])
    step_a = execute_agent_live_modules(
        agent_a,
        fatigue=0.0,
        routed_observation_by_instance_id=first.aggregate_by_instance_id,
        routing_telemetry_by_instance_id=first.telemetry_by_instance_id,
        routing_summary=first.summary,
    )
    step_b = execute_agent_live_modules(
        agent_b,
        fatigue=0.0,
        routed_observation_by_instance_id=second.aggregate_by_instance_id,
        routing_telemetry_by_instance_id=second.telemetry_by_instance_id,
        routing_summary=second.summary,
    )
    assert agent_a.structural_live_module_state == agent_b.structural_live_module_state
    assert step_a.rest_residual == step_b.rest_residual


@pytest.mark.parametrize(
    ("operation", "draw1", "initial", "expected_status"),
    [
        ("add", 0.10, [_binding("self_health")], "accepted"),
        ("remove", 0.30, [_binding()], "accepted"),
        ("gain", 0.70, [_binding(gain=0.5)], "accepted"),
        ("retarget", 0.80, [_binding("self_health")], "accepted"),
    ],
)
def test_routing_operator_fixed_two_draw_mapping(
    operation: str, draw1: float, initial: list[dict], expected_status: str
) -> None:
    module = _module(bindings=initial)
    identity_before = {
        key: deepcopy(module[key])
        for key in ("module_instance_id", "homology_id", "ancestry", "topology_pos")
    }
    result = mutate_observation_bindings([module], 0.10, draw1, enabled=True)
    assert ROUTING_OPERATOR_DRAWS == 2
    assert result.telemetry["draw_budget"] == {"declared_supplied_draws": 2, "internal_rng_draws": 0}
    assert result.telemetry["operation"] == operation
    assert result.telemetry["status"] == expected_status
    assert {key: result.modules[0][key] for key in identity_before} == identity_before


def test_routing_operator_disabled_cap_ineligible_and_promoter_independent() -> None:
    parent = [_module(bindings=[_binding()])]
    disabled = mutate_observation_bindings(parent, 0.1, 0.1, enabled=False)
    assert disabled.telemetry["status"] == "disabled"
    assert disabled.modules == normalize_agent_routing_modules(parent)

    capped_parent = [_module(bindings=[_binding(), _binding("self_health")])]
    capped = mutate_observation_bindings(capped_parent, 0.1, 0.1, enabled=True)
    assert capped.telemetry["status"] == "cap_blocked"
    assert capped.telemetry["reason"] == "module_binding_cap"

    ineligible = mutate_observation_bindings([], 0.1, 0.1, enabled=True)
    assert ineligible.telemetry["status"] == "ineligible"

    promoter_payload = {"modules": deepcopy(parent), "connections": []}
    routing_before = deepcopy(promoter_payload["modules"][0]["params"]["observation_routing"])
    out, changed = _op_promoter_expression_mutation(promoter_payload, [0.0, 0.5], {})
    assert changed is True
    assert out["modules"][0]["params"]["observation_routing"] == routing_before


class _CountingRng:
    def __init__(self, value: float = 0.25) -> None:
        self.value = float(value)
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return self.value


def test_c2_inherits_whole_module_binding_record_and_excludes_lifetime_state() -> None:
    identity = _identity(40, "shared")
    module_a = _module("slot_a", child=40, bindings=[_binding("self_energy", gain=0.25)])
    module_b = _module("slot_b", child=41, bindings=[_binding("self_health", gain=0.75)])
    for module in (module_a, module_b):
        for key, value in identity.items():
            module[key] = deepcopy(value)
        module["latent_pos"] = list(identity["topology_pos"])
    pa = _payload(module_a)
    pb = _payload(module_b)
    pa["structural_live_module_state"] = {str(identity["module_instance_id"]): 0.75}
    pb["structural_live_module_state"] = {str(identity["module_instance_id"]): -0.25}
    rng = _CountingRng()
    child, summary = recombine_lineage_payloads(parent_a_payload=pa, parent_b_payload=pb, rng=rng)
    assert summary.matched_modules == 1 and summary.budget_draws == 1 and rng.draws == 1
    inherited = child["modules"][0]["params"]["observation_routing"]
    parent_records = {
        repr(normalize_routing_block(module_a["params"]["observation_routing"])),
        repr(normalize_routing_block(module_b["params"]["observation_routing"])),
    }
    assert repr(inherited) in parent_records
    assert "structural_live_module_state" not in child


def test_c2_bad_routing_schema_rejects_before_donor_draw() -> None:
    identity = _identity(50, "bad-schema")
    module_a = _module("m000", child=50)
    module_b = deepcopy(module_a)
    for module in (module_a, module_b):
        for key, value in identity.items():
            module[key] = deepcopy(value)
        module["latent_pos"] = list(identity["topology_pos"])
    module_b["params"]["observation_routing"]["schema_version"] = "unsupported_v999"
    rng = _CountingRng()
    with pytest.raises(RuntimeError, match="BINDING_SCHEMA_UNSUPPORTED"):
        recombine_lineage_payloads(
            parent_a_payload=_payload(module_a),
            parent_b_payload=_payload(module_b),
            rng=rng,
            trusted_canonical_parent_payloads=True,
        )
    assert rng.draws == 0


def test_config_preflight_rejects_dependency_profile_lane_stride_and_attempts() -> None:
    valid = _cfg()
    validate_routing_config(valid, device="cpu", policy_stride=1)
    missing_live = _cfg()
    missing_live.structural_live_modules_enable = False
    with pytest.raises(RuntimeError, match="REQUIRES_ESCAP3"):
        validate_routing_config(missing_live)
    bad_profile = _cfg()
    bad_profile.structural_observation_routing_profile = "unsupported"
    with pytest.raises(RuntimeError, match="CONFIG_CONTRACT_MISMATCH"):
        validate_routing_config(bad_profile)
    with pytest.raises(RuntimeError, match="CPU_REFERENCE_ONLY"):
        validate_routing_config(valid, device="cuda")
    with pytest.raises(RuntimeError, match="POLICY_STRIDE_UNSUPPORTED"):
        validate_routing_config(valid, policy_stride=2)
    native = _cfg()
    native.native_runtime_enable = True
    with pytest.raises(RuntimeError, match="NATIVE_LANE_UNSUPPORTED"):
        validate_routing_config(native)
    attempts = _cfg()
    attempts.structural_observation_routing_operator_attempts = 3
    with pytest.raises(RuntimeError, match="OPERATOR_BUDGET_MISMATCH"):
        validate_routing_config(attempts)


def test_checkpoint_metadata_and_capacity_contract_are_versioned_and_bounded() -> None:
    def metadata(cfg: SimConfig) -> dict:
        sim = SimpleNamespace(sim_cfg=cfg)
        return _build_metadata(
            sim,
            generation_index=1,
            strict_replay_mode=True,
            world_hash="world",
            population_hash="population",
            rng_snapshot_hash="rng",
            runner_state_hash="runner",
            bank_state_hash="bank",
            resident_runtime_state_hash="resident",
            native_runtime_checkpoint_hash="native",
            event_log_hash="events",
            wal_hash="wal",
        )

    enabled = metadata(_cfg())
    disabled = metadata(_cfg(routing=False))
    routing = enabled["structural_observation_routing"]
    assert routing["enabled"] is True
    assert routing["profile"] == ROUTING_PROFILE
    assert routing["schema_version"] == ROUTING_SCHEMA_VERSION
    assert routing["registry_version"] == ROUTING_REGISTRY_VERSION
    assert routing["admitted_semantic_ids"] == ["self_energy", "self_health"]
    assert "structural_observation_routing" not in disabled
    assert enabled["checkpoint_id"] != disabled["checkpoint_id"]

    contract = build_capacity_contract(sim_cfg=_cfg())
    observation = contract["observations"]
    structural = contract["structure"]["observation_routing"]
    assert observation["routing_eligible_semantic_ids"] == ["self_energy", "self_health"]
    assert structural["maximum_bindings_per_module"] == MAX_BINDINGS_PER_MODULE
    assert structural["maximum_bindings_per_agent"] == MAX_BINDINGS_PER_AGENT
    assert structural["execution_rng_draws"] == 0
    assert structural["exposure_mode"] == ROUTING_EXPOSURE_MODE


def test_machine_audit_and_six_arm_probe_are_matched_causal_and_rng_free() -> None:
    audit = build_stage0_audit()
    assert audit["routing_contract"]["eligible_semantic_ids"] == ["self_energy", "self_health"]
    assert audit["routing_contract"]["direct_observation_shape_change"] is False
    assert all(
        row["routing_ineligible_reason"]
        for row in audit["channels"]
        if row["spatial"] and row["active_now"]
    )
    probe = run_technical_probe()
    assert [arm["arm"] for arm in probe["arms"]] == [
        "routing_disabled",
        "enabled_zero_gain",
        "active",
        "source_channel_lesion",
        "binding_lesion",
        "state_reset_tick2",
    ]
    assert all(bool(value) for value in probe["checks"].values())
    assert probe["checkpoint_continuation"]["migration_random_draws"] == 0
    assert "no ecology campaign" in probe["interpretation_boundary"]


class _SequenceRng:
    def __init__(self, values: list[float]) -> None:
        self.values = iter(values)
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return float(next(self.values))


def test_c1_wrapper_keeps_existing_profile_unchanged_and_routes_with_two_draws() -> None:
    payload = _payload(_module(bindings=[]))
    legacy_rng = _SequenceRng([])
    legacy, legacy_telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=legacy_rng,
        max_ops=0,
    )
    disabled_rng = _SequenceRng([])
    disabled, disabled_telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=disabled_rng,
        max_ops=0,
        observation_routing_mutation_enable=False,
        observation_routing_operator_attempts=0,
    )
    assert legacy == disabled
    assert legacy_telemetry == disabled_telemetry
    assert legacy_rng.draws == disabled_rng.draws == 0

    active_rng = _SequenceRng([0.1, 0.1])
    active, telemetry = apply_operator_pipeline(
        payload=deepcopy(payload),
        rng=active_rng,
        max_ops=0,
        observation_routing_mutation_enable=True,
        observation_routing_operator_attempts=1,
    )
    assert active_rng.draws == 2
    routing = telemetry["observation_routing_mutation"]
    assert routing["draws_per_attempt"] == 2
    assert routing["accepted"] == 1
    assert active["modules"][0]["params"]["observation_routing"]["bindings"] == [
        _binding("self_energy", gain=0.0)
    ]


@pytest.mark.parametrize(
    ("modules", "draws", "enabled", "status", "reason"),
    [
        ([_module(bindings=[])], [0.1, 0.1], True, "accepted", "binding_mutation_applied"),
        ([_module(bindings=[_binding()])], [0.1, 0.1], True, "rejected", "duplicate_binding"),
        (
            [_module(bindings=[_binding(), _binding("self_health")])],
            [0.1, 0.1],
            True,
            "cap_blocked",
            "module_binding_cap",
        ),
        ([], [0.1, 0.1], True, "ineligible", "no_generic_module"),
        ([_module(bindings=[])], [0.1, 0.1], False, "disabled", "operator_disabled"),
    ],
)
def test_every_routing_attempt_outcome_consumes_two_draws_and_invalid_preserves_parent(
    modules: list[dict],
    draws: list[float],
    enabled: bool,
    status: str,
    reason: str,
) -> None:
    canonical_parent = normalize_agent_routing_modules(modules)
    rng = _SequenceRng(draws)
    result = apply_observation_routing_operator_attempt(
        deepcopy(modules),
        rng=rng,
        enabled=enabled,
    )
    assert rng.draws == 2
    assert result.telemetry["draw_budget"] == {
        "declared_supplied_draws": 2,
        "internal_rng_draws": 0,
    }
    assert result.telemetry["status"] == status
    assert result.telemetry["reason"] == reason
    if status != "accepted":
        assert result.modules == canonical_parent


def _runner(root, *, routing: bool = True) -> SimulationRunner:
    cfg = _cfg(routing=routing)
    cfg.seed = 18074
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


def _install_module(sim: SimulationRunner, module: dict) -> None:
    for agent in sim.pop.agents.values():
        raw = agent.genome.to_dict()
        raw["modules"] = [deepcopy(module)]
        raw["connections"] = []
        raw["b2_invalid_policy"] = "reject"
        agent.genome = Genome.from_dict(raw)
        agent.drives[2] = np.float32(0.0)
        sim._drive_fatigue[int(agent.slot)] = np.float32(0.0)


def _direct_observation_bytes(sim: SimulationRunner) -> bytes:
    alive = [agent for agent in sim.pop.agents.values() if agent.alive]
    ids = np.asarray([agent.agent_id for agent in alive], dtype=np.int32)
    local = np.asarray(sim.env.get_local_observations(ids), dtype=np.float32)
    obs = np.empty((len(alive), local.shape[1] + 2), dtype=np.float32)
    obs[:, : local.shape[1]] = local
    obs[:, local.shape[1]] = np.asarray(
        [float(agent.energy) / float(sim.env_cfg.max_energy) for agent in alive],
        dtype=np.float32,
    )
    obs[:, local.shape[1] + 1] = np.asarray(
        [float(agent.health) / float(sim.env_cfg.start_health) for agent in alive],
        dtype=np.float32,
    )
    return obs.tobytes()


def test_real_runner_mlp_and_gru_route_only_to_rest_once(tmp_path) -> None:
    zero = _runner(tmp_path / "zero")
    active = _runner(tmp_path / "active")
    zero_module = _module(bindings=[_binding(gain=0.0)], retention=0.0, output_gain=1.0)
    active_module = _module(bindings=[_binding(gain=0.5)], retention=0.0, output_gain=1.0)
    _install_module(zero, zero_module)
    _install_module(active, active_module)
    zero_alive = [agent for agent in zero.pop.agents.values() if agent.alive]
    active_alive = [agent for agent in active.pop.agents.values() if agent.alive]
    zero._batched_actions(zero_alive)
    active._batched_actions(active_alive)
    zero_logits = zero._mem_max_live_level4_last_policy["logits_by_agent_id"]
    active_logits = active._mem_max_live_level4_last_policy["logits_by_agent_id"]
    families: set[str] = set()
    for zero_agent, active_agent in zip(zero_alive, active_alive, strict=True):
        assert zero_agent.agent_id == active_agent.agent_id
        families.add("gru" if bool(active_agent.genome.use_gru) else "mlp")
        before = np.asarray(zero_logits[zero_agent.agent_id], dtype=np.float32)
        after = np.asarray(active_logits[active_agent.agent_id], dtype=np.float32)
        delta = after - before
        assert delta[5] > 0.0
        assert np.delete(delta, 5).tobytes() == np.zeros((6,), dtype=np.float32).tobytes()
        routing = active_agent.structural_live_module_telemetry["observation_routing"]
        assert routing["active_bindings"] == 1
        assert active_agent.structural_live_module_telemetry["random_draws"] == 0
    assert families == {"mlp", "gru"}


def test_full_checkpoint_roundtrip_preserves_bindings_telemetry_and_strict_profile(tmp_path) -> None:
    source = _runner(tmp_path / "source")
    module = _module(bindings=[_binding(gain=0.5)], retention=0.5, output_gain=0.5)
    _install_module(source, module)
    alive = [agent for agent in source.pop.agents.values() if agent.alive]
    source._batched_actions(alive)
    source_agent = alive[0]
    bundle = checkpoint_bundle_v2_save(
        source,
        str(tmp_path / "bundle"),
        generation_index=1,
        strict_replay_mode=True,
    )
    loaded = checkpoint_bundle_v2_load(bundle, strict=True)
    assert loaded["metadata"]["structural_observation_routing"]["operator_attempts"] == 1

    restored_sim = _runner(tmp_path / "restored")
    checkpoint_bundle_v2_apply(restored_sim, loaded, restore_rng=True)
    restored = restored_sim.pop.agents[source_agent.agent_id]
    assert restored.genome.modules == source_agent.genome.modules
    assert restored.structural_live_module_state == source_agent.structural_live_module_state
    assert restored.structural_live_module_outputs == source_agent.structural_live_module_outputs
    assert restored.structural_live_module_telemetry == source_agent.structural_live_module_telemetry
    assert _direct_observation_bytes(restored_sim) == _direct_observation_bytes(source)
    source_actions = source._batched_actions(
        [agent for agent in source.pop.agents.values() if agent.alive]
    )
    restored_actions = restored_sim._batched_actions(
        [agent for agent in restored_sim.pop.agents.values() if agent.alive]
    )
    assert source_actions.tobytes() == restored_actions.tobytes()
    for agent_id, source_logits in source._mem_max_live_level4_last_policy["logits_by_agent_id"].items():
        restored_logits = restored_sim._mem_max_live_level4_last_policy["logits_by_agent_id"][agent_id]
        assert np.asarray(source_logits, dtype=np.float32).tobytes() == np.asarray(
            restored_logits, dtype=np.float32
        ).tobytes()
    for agent_id, source_live in source.pop.agents.items():
        restored_live = restored_sim.pop.agents[agent_id]
        assert source_live.structural_live_module_state == restored_live.structural_live_module_state
        assert source_live.structural_live_module_outputs == restored_live.structural_live_module_outputs
        assert source_live.structural_live_module_telemetry == restored_live.structural_live_module_telemetry

    routing_disabled = _runner(tmp_path / "mismatch", routing=False)
    with pytest.raises(CheckpointStrictError, match="structural_observation_routing config mismatch"):
        checkpoint_bundle_v2_apply(routing_disabled, loaded, restore_rng=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registry_version", "different_registry"),
        ("admitted_semantic_ids", ["self_energy"]),
        ("max_bindings_per_module", 1),
        ("max_bindings_per_agent", 7),
        ("operator_attempts", 2),
        ("exposure_mode", "structural_only"),
    ],
)
def test_enabled_checkpoint_contract_mismatches_reject_before_continuation(
    tmp_path, field: str, value: object
) -> None:
    source = _runner(tmp_path / f"source-{field}")
    bundle = checkpoint_bundle_v2_save(
        source,
        str(tmp_path / f"bundle-{field}"),
        generation_index=1,
        strict_replay_mode=True,
    )
    loaded = checkpoint_bundle_v2_load(bundle, strict=True)
    loaded["metadata"]["structural_observation_routing"][field] = value
    destination = _runner(tmp_path / f"destination-{field}")
    before = _direct_observation_bytes(destination)
    with pytest.raises(CheckpointStrictError, match="structural_observation_routing config mismatch"):
        checkpoint_bundle_v2_apply(destination, loaded, restore_rng=False)
    assert _direct_observation_bytes(destination) == before


def test_legacy_escap3_checkpoint_omits_routing_and_continues_only_disabled(tmp_path) -> None:
    source = _runner(tmp_path / "legacy-source", routing=False)
    legacy_module = _module(bindings=[], retention=0.5, input_gain=1.0, output_gain=0.5)
    legacy_module["params"].pop("observation_routing")
    _install_module(source, legacy_module)
    alive = [agent for agent in source.pop.agents.values() if agent.alive]
    source._batched_actions(alive)
    direct_before = _direct_observation_bytes(source)
    bundle = checkpoint_bundle_v2_save(
        source,
        str(tmp_path / "legacy-bundle"),
        generation_index=1,
        strict_replay_mode=True,
    )
    loaded = checkpoint_bundle_v2_load(bundle, strict=True)
    assert "structural_observation_routing" not in loaded["metadata"]

    restored = _runner(tmp_path / "legacy-restored", routing=False)
    checkpoint_bundle_v2_apply(restored, loaded, restore_rng=True)
    assert _direct_observation_bytes(restored) == direct_before
    assert all(
        "observation_routing" not in module.get("params", {})
        for agent in restored.pop.agents.values()
        for module in agent.genome.modules
    )
    source._batched_actions([agent for agent in source.pop.agents.values() if agent.alive])
    restored._batched_actions([agent for agent in restored.pop.agents.values() if agent.alive])
    for agent_id, source_logits in source._mem_max_live_level4_last_policy["logits_by_agent_id"].items():
        restored_logits = restored._mem_max_live_level4_last_policy["logits_by_agent_id"][agent_id]
        assert np.asarray(source_logits, dtype=np.float32).tobytes() == np.asarray(
            restored_logits, dtype=np.float32
        ).tobytes()
    continued = checkpoint_bundle_v2_save(
        restored,
        str(tmp_path / "legacy-continued-bundle"),
        generation_index=2,
        strict_replay_mode=True,
    )
    assert "structural_observation_routing" not in checkpoint_bundle_v2_load(continued, strict=True)["metadata"]

    enabled_destination = _runner(tmp_path / "legacy-enabled-destination", routing=True)
    with pytest.raises(CheckpointStrictError, match="structural_observation_routing config mismatch"):
        checkpoint_bundle_v2_apply(enabled_destination, loaded, restore_rng=False)
