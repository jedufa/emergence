from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from agents.genome import Genome
from agents.structural_identity_v1 import new_native_identity
from env.env_config import EnvConfig
from evolution.operators_c1 import apply_development_operator_attempt
from evolution.operators_development_v1 import mutate_developmental_expression
from evolution.capacity_contract import build_capacity_contract
from evolution.population import PopConfig
from evolution.recombination_c2 import recombine_lineage_payloads
from evolution.structural_development_runtime_v1 import resolve_agent_development_runtime
from evolution.structural_development_v1 import (
    DEVELOPMENTAL_EXPRESSION_BLOCK_KEY,
    DEVELOPMENTAL_FEATURE_PROFILE,
    DEVELOPMENTAL_MODULE_SCHEMA_VERSION,
    DEVELOPMENTAL_STAGE_PROFILE_VERSION,
    CYCLE_POLICY,
    DEPENDENCY_IDENTITY_AUTHORITY,
    INACTIVE_OUTPUT_POLICY,
    INACTIVE_STATE_POLICY,
    LATCH_SEMANTICS,
    LATCH_VERSION,
    MAX_DEPENDENCY_EDGES_PER_AGENT,
    MAX_PREREQUISITES_PER_MODULE,
    SUPPORTED_LANE,
    canonical_stage_program,
    neutral_developmental_expression_record,
    normalize_agent_developmental_modules,
    resolve_developmental_expression,
    resolve_structural_stage,
    validate_agent_developmental_graph,
    validate_structural_development_config,
)
from evolution.structural_live_modules_v1 import LIVE_PROFILE, execute_agent_live_modules
from simulation.runner_pkg.runner_config import SimConfig
from simulation.checkpoint_bundle_v2 import (
    _build_metadata,
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.control_plane_pkg.util_json import canonical_json_bytes
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner


def _identity(index: int) -> dict:
    return new_native_identity(
        context={
            "campaign_id": "escap6-test",
            "branch_id": "development",
            "generation": 1,
            "tick": 0,
            "child_agent_id": index,
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap6-lineage",
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
        "latch_dependency_readiness": latch,
    }


def _module(
    index: int,
    *,
    development: dict | None = None,
    expression_scale: float = 1.0,
    retention: float = 0.5,
    input_gain: float = 1.0,
    output_gain: float = 1.0,
) -> dict:
    identity = _identity(index)
    params = {
        "live_operator": "bounded_scalar_integrator_v1",
        "expression_scale": expression_scale,
        "live_retention": retention,
        "live_input_gain": input_gain,
        "live_output_gain": output_gain,
    }
    if development is not None:
        params[DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = deepcopy(development)
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


def _connection(index: int, source: dict, target: dict, weight: float = 1.0) -> dict:
    return {
        "connection_id": f"c{index:03d}",
        "src_module_id": source["module_id"],
        "src_slot": "out",
        "dst_module_id": target["module_id"],
        "dst_slot": "in",
        "value_type": "scalar",
        "weight": weight,
        "enabled": True,
    }


def _agent(modules: list[dict], *, age: int = 0, connections: list[dict] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        genome=SimpleNamespace(modules=deepcopy(modules), connections=deepcopy(connections or [])),
        stats=SimpleNamespace(survival_steps=age),
        structural_live_module_state={},
        structural_live_module_outputs={},
        structural_live_module_telemetry={},
        structural_development_latch_by_instance_id={},
        structural_development_stage_id="",
        structural_development_effective_gain_by_instance_id={},
    )


def _cfg(enabled: bool = True) -> SimConfig:
    cfg = SimConfig()
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.policy_tick_stride = 1
    cfg.structural_live_modules_enable = True
    cfg.structural_live_modules_profile = LIVE_PROFILE
    cfg.structural_development_enable = enabled
    cfg.structural_development_profile = DEVELOPMENTAL_FEATURE_PROFILE if enabled else "none"
    return cfg


def test_three_stage_boundaries_use_survival_steps_and_existing_resolver() -> None:
    cfg = _cfg()
    assert canonical_stage_program()["stages"] == [
        {"stage_id": "early", "start_step": 0, "end_step": 19},
        {"stage_id": "mature", "start_step": 20, "end_step": 39},
        {"stage_id": "late", "start_step": 40, "end_step": -1},
    ]
    assert [resolve_structural_stage(age, cfg)[0] for age in (0, 19, 20, 39, 40, 100)] == [
        "early", "early", "mature", "mature", "late", "late"
    ]


def test_enabled_config_is_frozen_cpu_stride_one_and_requires_live_modules() -> None:
    cfg = _cfg()
    validate_structural_development_config(cfg, device="cpu", policy_stride=1)
    assert cfg.structural_development_dependency_identity_authority == DEPENDENCY_IDENTITY_AUTHORITY
    assert cfg.structural_development_cycle_policy == CYCLE_POLICY
    assert cfg.structural_development_inactive_state_policy == INACTIVE_STATE_POLICY
    assert cfg.structural_development_inactive_output_policy == INACTIVE_OUTPUT_POLICY
    assert cfg.structural_development_latch_version == LATCH_VERSION
    assert cfg.structural_development_latch_semantics == LATCH_SEMANTICS
    assert cfg.structural_development_supported_lane == SUPPORTED_LANE
    with pytest.raises(RuntimeError, match="CPU_REFERENCE_ONLY"):
        validate_structural_development_config(cfg, device="cuda", policy_stride=1)
    with pytest.raises(RuntimeError, match="POLICY_STRIDE_UNSUPPORTED"):
        validate_structural_development_config(cfg, device="cpu", policy_stride=2)
    cfg.structural_live_modules_enable = False
    with pytest.raises(RuntimeError, match="REQUIRES_ESCAP3"):
        validate_structural_development_config(cfg, device="cpu", policy_stride=1)


def test_absent_record_is_preserved_and_neutral_record_is_exactly_neutral() -> None:
    absent = _module(1)
    neutral = _module(2, development=neutral_developmental_expression_record())
    normalized = normalize_agent_developmental_modules([absent, neutral])
    assert DEVELOPMENTAL_EXPRESSION_BLOCK_KEY not in normalized[0]["params"]
    for stage in ("early", "mature", "late"):
        resolved = resolve_developmental_expression(normalized, stage_id=stage)
        assert {row["effective_gain"] for row in resolved.rows_by_module_instance_id.values()} == {1.0}


def test_record_validation_rejects_non_generic_invalid_gain_self_and_caps() -> None:
    invalid = _module(1, development=_development())
    invalid["module_type"] = "probe"
    with pytest.raises(ValueError, match="REQUIRES_GENERIC_MODULE"):
        normalize_agent_developmental_modules([invalid])

    bad_gain = _module(2, development=_development((0.0, float("nan"), 1.0)))
    with pytest.raises(ValueError, match="GAIN_OUT_OF_BOUNDS"):
        normalize_agent_developmental_modules([bad_gain])

    self_edge = _module(3)
    self_edge["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = _development(
        prerequisites=(self_edge["homology_id"],)
    )
    with pytest.raises(ValueError, match="SELF_EDGE"):
        normalize_agent_developmental_modules([self_edge])

    too_many = _module(4, development=_development(prerequisites=("a", "b", "c")))
    with pytest.raises(ValueError, match="PREREQUISITE_CAP_EXCEEDED"):
        normalize_agent_developmental_modules([too_many])


def test_dependency_endpoints_duplicates_cycles_and_agent_edge_cap_fail_closed() -> None:
    first = _module(1, development=_development())
    second = _module(2, development=_development(prerequisites=(first["homology_id"],)))
    validate_agent_developmental_graph([second, first])

    missing = _module(3, development=_development(prerequisites=("not-present",)))
    with pytest.raises(ValueError, match="MISSING_PREREQUISITE"):
        validate_agent_developmental_graph([missing])

    duplicate = deepcopy(second)
    duplicate["module_instance_id"] = first["module_instance_id"]
    with pytest.raises(ValueError, match="DUPLICATE_MODULE_INSTANCE"):
        validate_agent_developmental_graph([first, duplicate])

    cycle_a = _module(10)
    cycle_b = _module(11)
    cycle_a["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = _development(prerequisites=(cycle_b["homology_id"],))
    cycle_b["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = _development(prerequisites=(cycle_a["homology_id"],))
    with pytest.raises(ValueError, match="DEPENDENCY_CYCLE"):
        validate_agent_developmental_graph([cycle_a, cycle_b])

    roots = [_module(20 + index) for index in range(MAX_DEPENDENCY_EDGES_PER_AGENT + 2)]
    consumers: list[dict] = []
    for index in range((MAX_DEPENDENCY_EDGES_PER_AGENT // MAX_PREREQUISITES_PER_MODULE) + 1):
        consumer = _module(50 + index)
        consumer["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY] = _development(
            prerequisites=(roots[index * 2]["homology_id"], roots[index * 2 + 1]["homology_id"])
        )
        consumers.append(consumer)
    with pytest.raises(ValueError, match="EDGE_CAP_EXCEEDED"):
        validate_agent_developmental_graph(roots + consumers)


def test_topological_resolution_uses_homology_and_two_parent_readiness() -> None:
    first = _module(1, development=_development((1.0, 0.0, 1.0)))
    second = _module(2, development=_development((1.0, 1.0, 1.0)))
    third = _module(
        3,
        development=_development(
            prerequisites=(second["homology_id"], first["homology_id"])
        ),
    )
    early = resolve_developmental_expression([third, second, first], stage_id="early")
    assert early.rows_by_module_instance_id[third["module_instance_id"]]["effective_gain"] == 1.0
    mature = resolve_developmental_expression([third, second, first], stage_id="mature")
    assert mature.rows_by_module_instance_id[first["module_instance_id"]]["effective_gain"] == 0.0
    assert mature.rows_by_module_instance_id[third["module_instance_id"]]["effective_gain"] == 0.0
    assert mature.topological_homology_ids[-1] == third["homology_id"]


def test_latch_opens_once_but_stage_gain_still_controls_expression() -> None:
    prerequisite = _module(1, development=_development((1.0, 0.0, 0.0)))
    dependent = _module(
        2,
        development=_development(
            (1.0, 0.5, 0.0),
            prerequisites=(prerequisite["homology_id"],),
            latch=True,
        ),
    )
    early = resolve_developmental_expression([dependent, prerequisite], stage_id="early")
    assert early.rows_by_module_instance_id[dependent["module_instance_id"]]["effective_gain"] == 1.0
    assert early.latch_state_by_module_instance_id[dependent["module_instance_id"]] is True
    mature = resolve_developmental_expression(
        [dependent, prerequisite],
        stage_id="mature",
        latch_state_by_module_instance_id=early.latch_state_by_module_instance_id,
    )
    assert mature.rows_by_module_instance_id[dependent["module_instance_id"]]["effective_gain"] == 0.5
    late = resolve_developmental_expression(
        [dependent, prerequisite],
        stage_id="late",
        latch_state_by_module_instance_id=mature.latch_state_by_module_instance_id,
    )
    assert late.rows_by_module_instance_id[dependent["module_instance_id"]]["effective_gain"] == 0.0


def test_runtime_uses_agent_age_and_keys_latches_by_instance_identity() -> None:
    root = _module(1, development=_development((1.0, 0.0, 1.0)))
    child = _module(2, development=_development(prerequisites=(root["homology_id"],), latch=True))
    agent = _agent([child, root], age=0)
    plan = resolve_agent_development_runtime(agent, sim_cfg=_cfg())
    assert plan.stage_id == "early"
    assert tuple(plan.topological_instance_ids) == (root["module_instance_id"], child["module_instance_id"])
    assert set(plan.latch_by_instance_id) == {child["module_instance_id"]}
    assert plan.telemetry["random_draws"] == 0


def test_zero_gain_freezes_state_and_zeroes_output_without_stale_connection_leak() -> None:
    source = _module(1, development=_development((0.0, 1.0, 1.0)), retention=1.0, input_gain=1.0)
    target = _module(2, development=_development(), retention=1.0, input_gain=0.0)
    agent = _agent([source, target], age=0, connections=[_connection(1, source, target)])
    source_id = source["module_instance_id"]
    target_id = target["module_instance_id"]
    agent.structural_live_module_state = {source_id: 0.75, target_id: 0.0}
    agent.structural_live_module_outputs = {source_id: 0.75, target_id: 0.0}
    plan = resolve_agent_development_runtime(agent, sim_cfg=_cfg())
    step = execute_agent_live_modules(agent, fatigue=1.0, developmental_plan=plan)
    assert agent.structural_live_module_state[source_id] == 0.75
    assert agent.structural_live_module_outputs[source_id] == 0.0
    assert agent.structural_live_module_state[target_id] == 0.0
    assert step.summary["evaluated_connections"] == 1


def test_reactivation_continues_retained_state_and_gain_multiplies_after_base_scale_once() -> None:
    module = _module(
        1,
        development=_development((0.0, 0.5, 1.0)),
        expression_scale=0.5,
        retention=1.0,
        input_gain=1.0,
        output_gain=1.0,
    )
    agent = _agent([module], age=0)
    instance = module["module_instance_id"]
    agent.structural_live_module_state = {instance: 0.4}
    agent.structural_live_module_outputs = {instance: 0.4}
    execute_agent_live_modules(agent, fatigue=1.0, developmental_plan=resolve_agent_development_runtime(agent, sim_cfg=_cfg()))
    assert agent.structural_live_module_state[instance] == 0.4
    agent.stats.survival_steps = 20
    step = execute_agent_live_modules(agent, fatigue=0.0, developmental_plan=resolve_agent_development_runtime(agent, sim_cfg=_cfg()))
    row = step.summary["state_rows"][0]
    assert row["state_after"] == pytest.approx(0.4)
    assert row["base_live_output"] == pytest.approx(0.2)
    assert row["typed_output"] == pytest.approx(0.1)
    assert agent.structural_development_stage_id == "mature"
    assert agent.structural_development_effective_gain_by_instance_id == {instance: 0.5}


def test_disabled_live_execution_is_legacy_exact_and_adds_no_escap6_telemetry() -> None:
    module = _module(1, expression_scale=0.5, retention=0.5, input_gain=1.0, output_gain=1.0)
    legacy = _agent([module])
    control = deepcopy(legacy)
    first = execute_agent_live_modules(legacy, fatigue=0.5)
    second = execute_agent_live_modules(control, fatigue=0.5, developmental_plan=None)
    assert first == second
    assert legacy.structural_live_module_state == control.structural_live_module_state
    assert legacy.structural_live_module_outputs == control.structural_live_module_outputs
    assert "structural_development" not in first.summary


class _CountingRng:
    def __init__(self, values: tuple[float, ...]) -> None:
        self.values = iter(values)
        self.count = 0

    def random(self) -> float:
        self.count += 1
        return next(self.values)


@pytest.mark.parametrize(
    ("draw0", "operation"),
    ((0.0, "add_neutral_record"), (0.25, "mutate_stage_gain"), (0.5, "add_or_remove_dependency"), (0.75, "toggle_latch")),
)
def test_development_operator_has_four_frozen_bins_and_exactly_two_draws(draw0: float, operation: str) -> None:
    modules = [_module(1), _module(2)]
    rng = _CountingRng((draw0, 0.2))
    result = apply_development_operator_attempt(modules, rng=rng, enabled=True)
    assert rng.count == 2
    assert result.telemetry["operation"] == operation
    assert result.telemetry["draw_budget"] == {"declared_supplied_draws": 2, "internal_rng_draws": 0}


def test_disabled_operator_still_consumes_two_draws_and_preserves_parent() -> None:
    modules = [_module(1)]
    rng = _CountingRng((0.1, 0.9))
    result = apply_development_operator_attempt(modules, rng=rng, enabled=False)
    assert rng.count == 2
    assert result.modules == modules
    assert result.telemetry["status"] == "disabled"


def _payload(module: dict) -> dict:
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": [deepcopy(module)],
        "connections": [],
    }


def test_c2_inherits_whole_development_record_without_extra_draw() -> None:
    shared = _identity(90)
    parent_a = _module(91, development=_development((0.0, 0.5, 1.0), latch=True))
    parent_b = _module(92, development=_development((1.0, 0.25, 0.0), latch=False))
    for module in (parent_a, parent_b):
        module.update(deepcopy(shared))
        module["latent_pos"] = list(shared["topology_pos"])
    rng = _CountingRng((0.25,))
    child, summary = recombine_lineage_payloads(
        parent_a_payload=_payload(parent_a),
        parent_b_payload=_payload(parent_b),
        rng=rng,
    )
    assert summary.budget_draws == rng.count == 1
    inherited = child["modules"][0]["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY]
    assert inherited in (
        parent_a["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY],
        parent_b["params"][DEVELOPMENTAL_EXPRESSION_BLOCK_KEY],
    )


def test_c2_invalid_parent_dag_rejects_before_any_donor_draw() -> None:
    parent_a = _module(101, development=_development(prerequisites=("missing",)))
    parent_b = _module(102, development=_development())
    rng = _CountingRng((0.25,))
    with pytest.raises(ValueError, match="MISSING_PREREQUISITE"):
        recombine_lineage_payloads(
            parent_a_payload=_payload(parent_a),
            parent_b_payload=_payload(parent_b),
            rng=rng,
        )
    assert rng.count == 0


def _metadata(cfg: object) -> dict:
    return _build_metadata(
        SimpleNamespace(sim_cfg=cfg),
        1,
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


def test_capacity_checkpoint_metadata_and_disabled_legacy_representation() -> None:
    enabled_cfg = _cfg()
    contract = build_capacity_contract(sim_cfg=enabled_cfg)
    development = contract["development"]["structural_development"]
    assert development["stage_semantic_ids"] == ["early", "mature", "late"]
    assert development["developmental_records"]["maximum_per_agent"] == 12
    assert development["dependency_edges"]["maximum_per_module"] == 2
    assert development["dependency_edges"]["maximum_per_agent"] == 8
    assert development["execution_rng_draws"] == 0

    enabled = _metadata(enabled_cfg)
    disabled_cfg = _cfg(False)
    disabled = _metadata(disabled_cfg)
    legacy_cfg = SimpleNamespace(
        **{
            key: value
            for key, value in vars(disabled_cfg).items()
            if not key.startswith("structural_development_")
        }
    )
    legacy = _metadata(legacy_cfg)
    assert enabled["structural_development"]["mature_start_step"] == 20
    assert enabled["structural_development"]["late_start_step"] == 40
    assert "structural_development" not in disabled
    assert canonical_json_bytes(disabled) == canonical_json_bytes(legacy)
    assert disabled["checkpoint_id"] == legacy["checkpoint_id"]
    assert enabled["checkpoint_id"] != disabled["checkpoint_id"]


def _real_runner(root, *, enabled: bool = True) -> SimulationRunner:
    cfg = _cfg(enabled)
    cfg.seed = 18606
    cfg.generations = 1
    cfg.steps_per_generation = 1
    cfg.behavior_log_enable = False
    cfg.policy_deterministic_actions = True
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


def _install_modules(sim: SimulationRunner, modules: list[dict]) -> None:
    for agent in sim.pop.agents.values():
        raw = agent.genome.to_dict()
        raw["modules"] = deepcopy(modules)
        raw["connections"] = []
        raw["b2_invalid_policy"] = "reject"
        agent.genome = Genome.from_dict(raw)
        agent.structural_live_module_state = {}
        agent.structural_live_module_outputs = {}
        agent.structural_live_module_telemetry = {}
        agent.structural_development_latch_by_instance_id = {}
        agent.structural_development_stage_id = ""
        agent.structural_development_effective_gain_by_instance_id = {}


def test_enabled_checkpoint_roundtrip_preserves_stage_gain_latch_and_continuation(tmp_path) -> None:
    source = _real_runner(tmp_path / "source")
    module = _module(201, development=_development((0.5, 0.25, 0.0), latch=True))
    _install_modules(source, [module])
    source_alive = [agent for agent in source.pop.agents.values() if agent.alive]
    source._batched_actions(source_alive)
    assert all(agent.structural_development_stage_id == "early" for agent in source_alive)
    assert all(
        agent.structural_development_effective_gain_by_instance_id[module["module_instance_id"]] == 0.5
        for agent in source_alive
    )
    assert all(
        agent.structural_development_latch_by_instance_id[module["module_instance_id"]] is True
        for agent in source_alive
    )

    bundle = checkpoint_bundle_v2_save(
        source,
        str(tmp_path / "bundle"),
        generation_index=1,
        strict_replay_mode=True,
    )
    loaded = checkpoint_bundle_v2_load(bundle, strict=True)
    assert loaded["metadata"]["structural_development"]["stage_semantic_ids"] == [
        "early", "mature", "late"
    ]
    restored = _real_runner(tmp_path / "restored")
    checkpoint_bundle_v2_apply(restored, loaded, restore_rng=True)
    for agent_id, source_agent in source.pop.agents.items():
        restored_agent = restored.pop.agents[agent_id]
        assert restored_agent.structural_development_stage_id == source_agent.structural_development_stage_id
        assert (
            restored_agent.structural_development_effective_gain_by_instance_id
            == source_agent.structural_development_effective_gain_by_instance_id
        )
        assert (
            restored_agent.structural_development_latch_by_instance_id
            == source_agent.structural_development_latch_by_instance_id
        )
        assert restored_agent.structural_live_module_state == source_agent.structural_live_module_state
        assert restored_agent.structural_live_module_outputs == source_agent.structural_live_module_outputs

    restored_alive = [agent for agent in restored.pop.agents.values() if agent.alive]
    source_actions = source._batched_actions(source_alive)
    restored_actions = restored._batched_actions(restored_alive)
    assert source_actions.tobytes() == restored_actions.tobytes()
    assert source.rng_manager.capture_snapshot(generation_index=source.gen)["streams"]["policy"]["indices"] == (
        restored.rng_manager.capture_snapshot(generation_index=restored.gen)["streams"]["policy"]["indices"]
    )


def test_dependency_mutation_cycle_proposal_is_rejected_whole_module() -> None:
    first = _module(1, development=_development())
    second = _module(2, development=_development(prerequisites=(first["homology_id"],)))
    parent = [first, second]
    # The dependency bin enumerates deterministic remove/add proposals. Scan
    # bounded draw positions and prove any cycle proposal cannot alter parent.
    saw_cycle_block = False
    for ordinal in range(64):
        result = mutate_developmental_expression(parent, 0.5, (ordinal + 0.5) / 64.0, enabled=True)
        if result.telemetry["status"] == "cycle_blocked":
            saw_cycle_block = True
            expected = sorted(
                normalize_agent_developmental_modules(parent),
                key=lambda module: str(module["homology_id"]),
            )
            assert result.modules == expected
    assert saw_cycle_block
