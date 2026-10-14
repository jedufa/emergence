from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.agent import Agent
from agents.genome import Genome
from agents.structural_identity_v1 import new_native_identity
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from evolution.operators_c1 import _op_promoter_expression_mutation
from evolution.capacity_contract import build_capacity_contract
from evolution.recombination_c2 import recombine_lineage_payloads
from evolution.structural_live_modules_v1 import (
    LIVE_MODULE_SCHEMA_VERSION,
    LIVE_PROFILE,
    add_rest_logit_residual_,
    execute_agent_live_modules,
    execute_live_modules_batch,
    reconcile_runtime_state_for_genome,
    validate_live_modules_config,
)
from evolution.structural_live_modules_probe import build_stage0_audit, run_technical_probe
from simulation.checkpoint_bundle_v2 import (
    CheckpointStrictError,
    _agent_to_dict,
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig


def _identity(child: int, role: str) -> dict:
    return new_native_identity(
        context={
            "campaign_id": "escap3-test",
            "branch_id": "capacity",
            "generation": 1,
            "tick": 0,
            "child_agent_id": int(child),
            "parent_agent_ids": [1, 2],
            "lineage_token": "test-lineage",
        },
        operator_id="fixture",
        attempt_index=int(child),
        operator_role=str(role),
    )


def _module(mid: str, identity: dict, **params: float | str) -> dict:
    return {
        "module_id": mid,
        "module_type": "generic",
        "params": {
            "live_operator": "bounded_scalar_integrator_v1",
            "expression_scale": 1.0,
            "live_retention": 0.5,
            "live_input_gain": 0.5,
            "live_output_gain": 0.5,
            **params,
        },
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **deepcopy(identity),
        "latent_pos": list(identity["topology_pos"]),
    }


def _connection(src: str, dst: str, *, cid: str = "c000", weight: float = 0.25) -> dict:
    return {
        "connection_id": cid,
        "src_module_id": src,
        "dst_module_id": dst,
        "src_slot": "out",
        "dst_slot": "in",
        "value_type": "scalar",
        "channel": "typed",
        "weight": weight,
    }


def _agent(modules: list[dict], connections: list[dict] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        genome=SimpleNamespace(modules=deepcopy(modules), connections=deepcopy(connections or [])),
        structural_live_module_state={},
        structural_live_module_outputs={},
        structural_live_module_telemetry={},
    )


def _cfg(enabled: bool = True) -> SimConfig:
    cfg = SimConfig()
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.policy_tick_stride = 1
    cfg.structural_live_modules_enable = bool(enabled)
    cfg.structural_live_modules_profile = LIVE_PROFILE if enabled else "none"
    return cfg


def test_disabled_and_zero_bootstrap_are_neutral_and_rng_free() -> None:
    ident = _identity(1, "neutral")
    module = _module(
        "m000",
        ident,
        expression_scale=0.0,
        live_retention=0.0,
        live_input_gain=0.0,
        live_output_gain=0.0,
    )
    agent = _agent([module])
    disabled = execute_live_modules_batch([agent], np.array([0.8], dtype=np.float32), sim_cfg=_cfg(False), device="cpu")
    assert disabled.tobytes() == np.zeros((1,), dtype=np.float32).tobytes()
    assert agent.structural_live_module_state == {}

    step = execute_agent_live_modules(agent, fatigue=0.8)
    assert step.rest_residual == 0.0
    assert step.summary["random_draws"] == 0
    assert step.summary["active_modules"] == 0
    assert step.summary["reported_compute_cost"] == 4


def test_state_equation_previous_tick_connection_and_order_invariance() -> None:
    a = _module("m000", _identity(2, "source"), live_retention=0.0, live_input_gain=1.0, live_output_gain=1.0)
    b = _module("m001", _identity(3, "sink"), live_retention=0.0, live_input_gain=0.0, live_output_gain=1.0)
    conn = _connection("m000", "m001", weight=0.5)
    first = _agent([a, b], [conn])
    step0 = execute_agent_live_modules(first, fatigue=0.4)
    rows0 = {row["module_id"]: row for row in step0.summary["state_rows"]}
    assert rows0["m000"]["typed_output"] == pytest.approx(0.4)
    assert rows0["m001"]["typed_input_previous_outputs"] == 0.0
    assert rows0["m001"]["typed_output"] == 0.0
    step1 = execute_agent_live_modules(first, fatigue=0.4)
    rows1 = {row["module_id"]: row for row in step1.summary["state_rows"]}
    assert rows1["m001"]["typed_input_previous_outputs"] == pytest.approx(0.2)
    assert rows1["m001"]["typed_output"] == pytest.approx(0.2)

    repacked = _agent([b, a], [conn])
    execute_agent_live_modules(repacked, fatigue=0.4)
    reordered = execute_agent_live_modules(repacked, fatigue=0.4)
    assert reordered.rest_residual == step1.rest_residual
    assert repacked.structural_live_module_state == first.structural_live_module_state


def test_connection_permutation_aggregation_and_lesions_are_causal() -> None:
    a = _module("m000", _identity(4, "source-a"), live_retention=0.0, live_input_gain=1.0, live_output_gain=1.0)
    b = _module("m001", _identity(5, "source-b"), live_retention=0.0, live_input_gain=1.0, live_output_gain=1.0)
    c = _module("m002", _identity(6, "sink"), live_retention=0.0, live_input_gain=0.0, live_output_gain=1.0)
    conns = [_connection("m000", "m002", cid="c000", weight=0.25), _connection("m001", "m002", cid="c001", weight=0.5)]
    x = _agent([a, b, c], conns)
    y = _agent([a, b, c], list(reversed(conns)))
    execute_agent_live_modules(x, fatigue=0.8)
    execute_agent_live_modules(y, fatigue=0.8)
    control_x = execute_agent_live_modules(x, fatigue=0.8)
    control_y = execute_agent_live_modules(y, fatigue=0.8)
    assert control_x.rest_residual == control_y.rest_residual
    assert x.structural_live_module_state == y.structural_live_module_state

    reset_key = str(c["module_instance_id"])
    reset = execute_agent_live_modules(x, fatigue=0.8, reset_instance_ids=[reset_key])
    connection_lesion = execute_agent_live_modules(y, fatigue=0.8, lesion_connection_ids=["c001"])
    expression_lesion = execute_agent_live_modules(_agent([a, b, c], conns), fatigue=0.8, lesion_expression=True)
    assert reset.summary["reset_instance_ids"] == [reset_key]
    control_sink = next(row for row in control_y.summary["state_rows"] if row["module_id"] == "m002")
    lesioned_sink = next(row for row in connection_lesion.summary["state_rows"] if row["module_id"] == "m002")
    assert lesioned_sink["typed_output"] != control_sink["typed_output"]
    assert expression_lesion.rest_residual == 0.0


def test_controller_seam_changes_only_rest_and_applies_same_tensor_convention() -> None:
    base = torch.arange(21, dtype=torch.float32).reshape(3, 7)
    residual = torch.tensor([0.1, -0.2, 0.25], dtype=torch.float32)
    actual = base.clone()
    add_rest_logit_residual_(actual, residual, rest_index=5)
    for column in range(7):
        if column == 5:
            assert torch.equal(actual[:, column], base[:, column] + residual)
        else:
            assert torch.equal(actual[:, column], base[:, column])
    # The helper is shared by the MLP and GRU call sites; its output is not
    # retained externally, so repeated calls add only when explicitly invoked.
    again = base.clone()
    add_rest_logit_residual_(again, residual, rest_index=5)
    assert torch.equal(actual, again)


def test_runtime_identity_orphans_duplicates_and_slot_reuse_fail_closed() -> None:
    original = _module("m000", _identity(7, "original"))
    agent = _agent([original])
    execute_agent_live_modules(agent, fatigue=0.5)
    old_key = str(original["module_instance_id"])

    replacement = _module("m000", _identity(8, "replacement"))
    agent.genome.modules = [replacement]
    with pytest.raises(RuntimeError, match="ESCAP3_ORPHAN_RUNTIME_STATE"):
        execute_agent_live_modules(agent, fatigue=0.5)
    result = reconcile_runtime_state_for_genome(agent, orphan_policy="prune")
    assert result["pruned_instance_ids"] == [old_key]
    assert agent.structural_live_module_state == {}
    execute_agent_live_modules(agent, fatigue=0.5)
    assert old_key not in agent.structural_live_module_state

    duplicate = _agent([original, {**deepcopy(replacement), "module_instance_id": old_key}])
    with pytest.raises(RuntimeError, match="ESCAP3_DUPLICATE_INSTANCE_ID"):
        execute_agent_live_modules(duplicate, fatigue=0.5)


@pytest.mark.parametrize(
    ("surface", "value", "pattern"),
    [
        ("parameter", float("nan"), "ESCAP3_NONFINITE"),
        ("state", float("inf"), "ESCAP3_NONFINITE"),
        ("output", float("nan"), "ESCAP3_NONFINITE"),
        ("input", float("inf"), "ESCAP3_NONFINITE"),
        ("weight", float("nan"), "ESCAP3_NONFINITE"),
    ],
)
def test_nonfinite_surfaces_fail_closed(surface: str, value: float, pattern: str) -> None:
    module = _module("m000", _identity(9, surface))
    sink = _module("m001", _identity(10, f"{surface}-sink"))
    conn = _connection("m000", "m001")
    agent = _agent([module, sink], [conn])
    if surface == "parameter":
        agent.genome.modules[0]["params"]["live_input_gain"] = value
    elif surface == "state":
        agent.structural_live_module_state = {str(module["module_instance_id"]): value}
        agent.structural_live_module_outputs = {str(module["module_instance_id"]): 0.0}
    elif surface == "output":
        agent.structural_live_module_state = {str(module["module_instance_id"]): 0.0}
        agent.structural_live_module_outputs = {str(module["module_instance_id"]): value}
    elif surface == "weight":
        agent.genome.connections[0]["weight"] = value
    with pytest.raises(RuntimeError, match=pattern):
        execute_agent_live_modules(agent, fatigue=(value if surface == "input" else 0.5))


def test_extreme_legal_values_clip_at_state_output_and_final_residual() -> None:
    modules = [
        _module(f"m{i:03d}", _identity(20 + i, f"clip-{i}"), expression_scale=2.0, live_retention=1.0, live_input_gain=2.0, live_output_gain=2.0)
        for i in range(3)
    ]
    step = execute_agent_live_modules(_agent(modules), fatigue=1.0)
    assert all(row["state_after"] == 1.0 and row["typed_output"] == 1.0 for row in step.summary["state_rows"])
    assert step.rest_residual == 0.25
    assert step.summary["clipped_rest_residual"] is True


def test_promoter_live_parameter_mapping_is_fixed_budget_and_identity_preserving() -> None:
    identity = _identity(30, "mutation")
    base = _module("m000", identity, expression_scale=0.0, live_retention=0.0, live_input_gain=0.0, live_output_gain=0.0)
    for draw, expected in ((0.1, "live_retention"), (0.5, "live_input_gain"), (0.9, "live_output_gain")):
        payload = {"modules": [deepcopy(base)], "connections": []}
        before_identity = {k: deepcopy(base[k]) for k in ("module_instance_id", "homology_id", "ancestry", "topology_pos")}
        out, changed = _op_promoter_expression_mutation(payload, [0.0, draw], {})
        assert changed is True
        assert expected in out["modules"][0]["params"]
        assert {k: out["modules"][0][k] for k in before_identity} == before_identity


class _CountingRng:
    def __init__(self) -> None:
        self.draws = 0

    def random(self) -> float:
        self.draws += 1
        return 0.25


def test_c2_incompatible_live_operator_rejects_before_donor_draw_and_state_is_external() -> None:
    identity = _identity(40, "shared")
    a = _module("m000", identity)
    b = deepcopy(a)
    b["params"]["live_operator"] = "unsupported_v2"
    pa = {"schema_version": "modular_genome_v1", "b2_invalid_policy": "reject", "genes": {}, "modules": [a], "connections": []}
    pb = {"schema_version": "modular_genome_v1", "b2_invalid_policy": "reject", "genes": {}, "modules": [b], "connections": []}
    rng = _CountingRng()
    with pytest.raises(RuntimeError, match="DET_C2_INCOMPATIBLE_LIVE_MODULE"):
        recombine_lineage_payloads(parent_a_payload=pa, parent_b_payload=pb, rng=rng)
    assert rng.draws == 0
    assert "structural_live_module_state" not in pa and "structural_live_module_state" not in pb


def test_c2_compatible_homolog_recombines_architecture_but_child_lifetime_state_is_empty() -> None:
    identity = _identity(41, "shared-compatible")
    module_a = _module("slot_a", identity, live_input_gain=0.25)
    module_b = _module("slot_b", identity, live_input_gain=0.75)
    pa = {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": [module_a],
        "connections": [],
    }
    pb = {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": [module_b],
        "connections": [],
    }
    parent_a_runtime = {
        "state": {str(identity["module_instance_id"]): 0.75},
        "outputs": {str(identity["module_instance_id"]): 0.5},
    }
    parent_b_runtime = {
        "state": {str(identity["module_instance_id"]): -0.25},
        "outputs": {str(identity["module_instance_id"]): -0.125},
    }
    rng = _CountingRng()
    child_payload, summary = recombine_lineage_payloads(
        parent_a_payload=pa,
        parent_b_payload=pb,
        rng=rng,
    )
    assert summary.matched_modules == 1
    assert summary.budget_draws == 1
    assert rng.draws == 1
    assert child_payload["modules"][0]["homology_id"] == identity["homology_id"]
    child = Agent(
        agent_id=99,
        genome=Genome.from_dict(child_payload),
        policy=object(),
        energy=10.0,
        health=5.0,
    )
    assert parent_a_runtime != parent_b_runtime
    assert child.structural_live_module_state == {}
    assert child.structural_live_module_outputs == {}


def test_checkpoint_payload_contains_versioned_runtime_and_newborn_starts_empty() -> None:
    identity = _identity(50, "checkpoint")
    genome = Genome.from_dict(
        {
            "schema_version": "modular_genome_v1",
            "b2_invalid_policy": "reject",
            "genes": {},
            "modules": [_module("m000", identity)],
            "connections": [],
        }
    )
    agent = Agent(agent_id=1, genome=genome, policy=object(), energy=10.0, health=5.0)
    execute_agent_live_modules(agent, fatigue=0.5)
    payload = _agent_to_dict(agent)["structural_live_module_runtime"]
    assert payload["schema_version"] == LIVE_MODULE_SCHEMA_VERSION
    assert payload["state_by_instance_id"] == agent.structural_live_module_state
    assert payload["previous_outputs_by_instance_id"] == agent.structural_live_module_outputs
    newborn = Agent(agent_id=2, genome=genome, policy=object(), energy=10.0, health=5.0)
    assert newborn.structural_live_module_state == {}
    assert newborn.structural_live_module_outputs == {}
    assert "structural_live_module_runtime" not in _agent_to_dict(newborn)


def test_checkpoint_roundtrip_first_continuation_and_strict_flags(tmp_path) -> None:
    def make_sim(root, *, enabled: bool = True) -> SimulationRunner:
        cfg = _cfg(enabled)
        cfg.seed = 18073
        cfg.generations = 1
        cfg.steps_per_generation = 1
        cfg.behavior_log_enable = False
        cfg.checkpoint_bundle_v2_period_gen = 0
        return SimulationRunner(
            EnvConfig(width=8, height=8),
            PopConfig(population_size=2),
            cfg,
            NoOpMetricsLogger(),
            str(root),
            perf=False,
        )

    identity = _identity(55, "checkpoint-full")
    genome = Genome.from_dict(
        {
            "schema_version": "modular_genome_v1",
            "b2_invalid_policy": "reject",
            "genes": {},
            "modules": [_module("m000", identity)],
            "connections": [],
        }
    )
    sim_a = make_sim(tmp_path / "source")
    sim_a.run(start_gen=0, skip_init=False)
    source_agent = next(iter(sim_a.pop.agents.values()))
    source_agent.genome = genome
    execute_agent_live_modules(source_agent, fatigue=0.4)
    execute_agent_live_modules(source_agent, fatigue=0.8)
    bundle = checkpoint_bundle_v2_save(
        sim_a,
        str(tmp_path / "bundle"),
        generation_index=1,
        strict_replay_mode=True,
    )
    loaded = checkpoint_bundle_v2_load(bundle, strict=True)
    assert loaded["metadata"]["structural_live_modules_enable"] is True
    assert loaded["metadata"]["structural_live_modules_profile"] == LIVE_PROFILE

    sim_b = make_sim(tmp_path / "restored")
    checkpoint_bundle_v2_apply(sim_b, loaded, restore_rng=True)
    restored = sim_b.pop.agents[source_agent.agent_id]
    assert restored.structural_live_module_state == source_agent.structural_live_module_state
    assert restored.structural_live_module_outputs == source_agent.structural_live_module_outputs
    next_a = execute_agent_live_modules(source_agent, fatigue=0.2)
    next_b = execute_agent_live_modules(restored, fatigue=0.2)
    assert next_a.rest_residual == next_b.rest_residual
    assert next_a.summary == next_b.summary

    mismatched = make_sim(tmp_path / "mismatch", enabled=False)
    with pytest.raises(CheckpointStrictError, match="structural_live_modules config mismatch"):
        checkpoint_bundle_v2_apply(mismatched, loaded, restore_rng=False)

    profile_mismatch = make_sim(tmp_path / "profile-mismatch")
    profile_mismatch.sim_cfg.structural_live_modules_profile = "different_profile"
    with pytest.raises(CheckpointStrictError, match="structural_live_modules config mismatch"):
        checkpoint_bundle_v2_apply(profile_mismatch, loaded, restore_rng=False)

    bad_schema_payload = deepcopy(loaded)
    live_agent = next(
        row
        for row in bad_schema_payload["population"]["agents"]
        if int(row["agent_id"]) == int(source_agent.agent_id)
    )
    live_agent["structural_live_module_runtime"]["schema_version"] = "unsupported_runtime_v999"
    bad_schema_destination = make_sim(tmp_path / "bad-schema")
    with pytest.raises(RuntimeError, match="CHECKPOINT_RUNTIME_SCHEMA_UNSUPPORTED"):
        checkpoint_bundle_v2_apply(bad_schema_destination, bad_schema_payload, restore_rng=False)


def test_lane_profile_and_stride_rejections_happen_in_preflight() -> None:
    cfg = _cfg(True)
    validate_live_modules_config(cfg, device="cpu", policy_stride=1)
    with pytest.raises(RuntimeError, match="CPU_REFERENCE_ONLY"):
        validate_live_modules_config(cfg, device="cuda", policy_stride=1)
    with pytest.raises(RuntimeError, match="POLICY_STRIDE_UNSUPPORTED"):
        validate_live_modules_config(cfg, device="cpu", policy_stride=2)
    cfg.native_runtime_enable = True
    with pytest.raises(RuntimeError, match="NATIVE_LANE_UNSUPPORTED"):
        validate_live_modules_config(cfg, device="cpu", policy_stride=1)
    cfg.native_runtime_enable = False
    cfg.structural_live_modules_profile = "unsupported"
    with pytest.raises(RuntimeError, match="UNSUPPORTED_PROFILE"):
        validate_live_modules_config(cfg, device="cpu", policy_stride=1)

    contract = build_capacity_contract(sim_cfg=_cfg(True))["structure"]["live_computation"]
    assert contract["storable"] is True
    assert contract["executable"] is True
    assert contract["active"] is True
    assert contract["owned_state_dimensions_per_executable_module"] == 1
    assert contract["supported_lanes"] == ["python_reference_cpu"]


def test_real_runner_mlp_and_gru_paths_change_only_rest_logit(tmp_path) -> None:
    def make_sim(root, *, enabled: bool) -> SimulationRunner:
        cfg = _cfg(enabled)
        cfg.seed = 1
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

    control = make_sim(tmp_path / "control", enabled=False)
    active = make_sim(tmp_path / "active", enabled=True)
    modules = [
        _module(
            "m000",
            _identity(70, "runner-live"),
            live_retention=0.0,
            live_input_gain=0.5,
            live_output_gain=0.5,
        )
    ]
    for sim in (control, active):
        for agent in sim.pop.agents.values():
            raw = agent.genome.to_dict()
            raw["modules"] = deepcopy(modules)
            raw["connections"] = []
            raw["b2_invalid_policy"] = "reject"
            agent.genome = Genome.from_dict(raw)
            agent.drives[2] = np.float32(0.8)
            sim._drive_fatigue[int(agent.slot)] = np.float32(0.8)

    control_alive = [agent for agent in control.pop.agents.values() if agent.alive]
    active_alive = [agent for agent in active.pop.agents.values() if agent.alive]
    control._batched_actions(control_alive)
    active._batched_actions(active_alive)
    control_logits = control._mem_max_live_level4_last_policy["logits_by_agent_id"]
    active_logits = active._mem_max_live_level4_last_policy["logits_by_agent_id"]
    families_seen: set[str] = set()
    for control_agent, active_agent in zip(control_alive, active_alive, strict=True):
        assert control_agent.agent_id == active_agent.agent_id
        family = "gru" if bool(active_agent.genome.use_gru) else "mlp"
        families_seen.add(family)
        before = np.asarray(control_logits[control_agent.agent_id], dtype=np.float32)
        after = np.asarray(active_logits[active_agent.agent_id], dtype=np.float32)
        delta = after - before
        assert delta[5] == pytest.approx(0.2)
        assert np.delete(delta, 5).tobytes() == np.zeros((6,), dtype=np.float32).tobytes()
        assert active_agent.structural_live_module_telemetry["rest_logit_residual"] == pytest.approx(0.2)
        assert control_agent.structural_live_module_state == {}
    assert families_seen == {"mlp", "gru"}


def test_type_cycle_missing_endpoint_and_runtime_map_mismatch_reject() -> None:
    a = _module("m000", _identity(60, "guard-a"))
    b = _module("m001", _identity(61, "guard-b"))
    bad_type = _connection("m000", "m001")
    bad_type["value_type"] = "gate"
    with pytest.raises(RuntimeError, match="TYPE_MISMATCH"):
        execute_agent_live_modules(_agent([a, b], [bad_type]), fatigue=0.5)
    with pytest.raises(RuntimeError, match="MISSING_ENDPOINT"):
        execute_agent_live_modules(_agent([a], [_connection("m000", "absent")]), fatigue=0.5)
    cycle = [_connection("m000", "m001", cid="c0"), _connection("m001", "m000", cid="c1")]
    with pytest.raises(RuntimeError, match="CYCLE_REJECTED"):
        execute_agent_live_modules(_agent([a, b], cycle), fatigue=0.5)
    mismatch = _agent([a])
    mismatch.structural_live_module_state = {str(a["module_instance_id"]): 0.0}
    with pytest.raises(RuntimeError, match="RUNTIME_MAP_KEY_MISMATCH"):
        execute_agent_live_modules(mismatch, fatigue=0.5)


def test_stage0_audit_and_five_arm_probe_are_machine_readable_and_bounded() -> None:
    audit = build_stage0_audit()
    assert audit["execution_contract"]["rng_draws"] == 0
    assert audit["lane_support"]["cuda"] == "fail closed in preflight"
    assert audit["field_classification"]["executing"]
    probe = run_technical_probe()
    assert [arm["arm"] for arm in probe["arms"]] == [
        "disabled",
        "enabled_zero_neutral",
        "active",
        "active_state_reset_lesion",
        "active_connection_lesion",
    ]
    assert probe["checks"]["active_first_expression_tick"] == 0
    assert all(
        bool(value)
        for key, value in probe["checks"].items()
        if key != "active_first_expression_tick"
    )
    assert probe["checkpoint_continuation"]["migration_random_draws"] == 0
    assert "no ecology" in probe["interpretation_boundary"]
