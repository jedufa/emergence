from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents.agent import ACTION_VOCABULARY
from agents.genome import Genome
from agents.structural_identity_v1 import new_native_identity
from env.env_config import EnvConfig
from evolution.capacity_contract import build_capacity_contract
from evolution.population_pkg.population_config import PopConfig
from evolution.structural_gated_recurrent_v1 import (
    GATED_DRAW1_BINS,
    GATED_NEUTRAL_TEMPLATE_VERSION,
    GATED_OPERATOR,
    neutral_gated_recurrent_record,
)
from evolution.structural_live_modules_v1 import (
    LIVE_MODULE_SCHEMA_VERSION,
    LIVE_PROFILE,
    add_rest_logit_residual_,
    execute_agent_live_modules,
)
from simulation.checkpoint_bundle_v2 import (
    CheckpointStrictError,
    _build_metadata,
    _escap7_gated_recurrent_metadata,
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.metrics_pkg.noop_loggers import NoOpMetricsLogger
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_base import _validate_escap7_gated_recurrent_config
from simulation.runner_pkg.runner_config import SimConfig


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


def _cfg(*, enabled: bool) -> SimConfig:
    cfg = SimConfig()
    cfg.structural_gated_recurrent_enable = bool(enabled)
    cfg.structural_gated_recurrent_profile = (
        "escap7_fixed_width_gated_recurrent_v1" if enabled else "none"
    )
    cfg.structural_live_modules_enable = True
    cfg.structural_live_modules_profile = LIVE_PROFILE
    cfg.structural_live_modules_schema_version = LIVE_MODULE_SCHEMA_VERSION
    cfg.device = "cpu"
    cfg.bank_device = "cpu"
    cfg.policy_tick_stride = 1
    return cfg


def test_disabled_escap7_checkpoint_representation_is_exactly_pre_escap7() -> None:
    disabled = _cfg(enabled=False)
    historical = SimpleNamespace(
        **{
            name: deepcopy(value)
            for name, value in vars(disabled).items()
            if not name.startswith("structural_gated_recurrent_")
        }
    )

    disabled_metadata = _metadata(disabled)
    historical_metadata = _metadata(historical)

    assert "structural_gated_recurrent" not in disabled_metadata
    assert "structural_gated_recurrent" not in historical_metadata
    assert disabled_metadata == historical_metadata
    assert disabled_metadata["checkpoint_id"] == historical_metadata["checkpoint_id"]


def test_enabled_checkpoint_metadata_freezes_complete_replay_contract() -> None:
    cfg = _cfg(enabled=True)
    metadata = _metadata(cfg)
    gated = metadata["structural_gated_recurrent"]

    assert gated == _escap7_gated_recurrent_metadata(cfg)
    assert gated["operator"] == "gated_scalar_recurrent_v1"
    assert gated["parameter_schema_version"] == "escap7_gated_scalar_recurrent_params_v1"
    assert gated["equation_version"] == "escap7_reset_update_candidate_scalar_v1"
    assert gated["numeric_policy_version"] == "escap7_stable_f32_clip16_v1"
    assert gated["state_width"] == 1
    assert gated["operator_attempts"] == 1
    assert gated["operator_draws_per_attempt"] == 2
    assert gated["draw_bins"] == [
        "acquire_neutral",
        "mutate_reset_parameter",
        "mutate_update_parameter",
        "mutate_candidate_parameter",
        "mutate_output_gain",
    ]
    assert gated["neutral_template_version"] == GATED_NEUTRAL_TEMPLATE_VERSION
    assert gated["neutral_template"] == {
        "source_operator": "bounded_scalar_integrator_v1",
        "gated_record": neutral_gated_recurrent_record(),
        "public_output_neutral": True,
        "latent_state_frozen": False,
    }
    assert gated["draw_bin_map"] == [
        {
            "lower_inclusive": float(lower),
            "upper_exclusive": float(upper),
            "operation": operation,
        }
        for lower, upper, operation in GATED_DRAW1_BINS
    ]
    assert gated["supported_module_types"] == [
        {
            "module_type": "generic",
            "operators": [
                "bounded_scalar_integrator_v1",
                "gated_scalar_recurrent_v1",
            ],
        }
    ]
    assert gated["execution_rng_draws"] == 0
    assert gated["migration_rng_draws"] == 0
    assert gated["fixed_shape_policy"] == {
        "module_state_width": 1,
        "new_arrays": 0,
        "observation_width_change": 0,
        "action_width_change": 0,
        "central_gru_width_change": 0,
    }


def test_enabled_checkpoint_id_changes_for_every_versioned_contract_field() -> None:
    control = _cfg(enabled=True)
    control_metadata = _metadata(control)
    cases = {
        "structural_gated_recurrent_equation_version": "different-equation",
        "structural_gated_recurrent_numeric_policy_version": "different-numeric",
        "structural_gated_recurrent_state_width": 2,
        "structural_gated_recurrent_decomposition_version": "different-decomposition",
        "structural_gated_recurrent_c2_compatibility_version": "different-c2",
        "structural_gated_recurrent_supported_lane": "different-lane",
    }
    for field, value in cases.items():
        changed = _cfg(enabled=True)
        setattr(changed, field, value)
        changed_metadata = _metadata(changed)
        assert changed_metadata["checkpoint_id"] != control_metadata["checkpoint_id"]
        assert changed_metadata["structural_gated_recurrent"] != control_metadata[
            "structural_gated_recurrent"
        ]


def test_enabled_lane_accepts_only_cpu_reference_stride_one() -> None:
    cfg = _cfg(enabled=True)
    _validate_escap7_gated_recurrent_config(
        cfg,
        policy_device=torch.device("cpu"),
        bank_device=torch.device("cpu"),
    )

    cases = (
        ("policy_cuda", {"policy_device": torch.device("cuda"), "bank_device": torch.device("cpu")}),
        ("bank_cuda", {"policy_device": torch.device("cpu"), "bank_device": torch.device("cuda")}),
    )
    for _name, devices in cases:
        with pytest.raises(RuntimeError, match="UNSUPPORTED_EXECUTION_LANE"):
            _validate_escap7_gated_recurrent_config(cfg, **devices)

    stride = _cfg(enabled=True)
    stride.policy_tick_stride = 2
    with pytest.raises(RuntimeError, match="UNSUPPORTED_EXECUTION_LANE"):
        _validate_escap7_gated_recurrent_config(
            stride,
            policy_device=torch.device("cpu"),
            bank_device=torch.device("cpu"),
        )

    for field in ("perf_sync_cuda", "native_runtime_enable"):
        alternate = _cfg(enabled=True)
        setattr(alternate, field, True)
        with pytest.raises(RuntimeError, match="UNSUPPORTED_EXECUTION_LANE"):
            _validate_escap7_gated_recurrent_config(
                alternate,
                policy_device=torch.device("cpu"),
                bank_device=torch.device("cpu"),
            )


def test_every_discovered_runnercore_enable_boolean_fails_closed() -> None:
    base = _cfg(enabled=True)
    fields = sorted(
        name
        for name, field in type(base).__dataclass_fields__.items()
        if "runnercore" in "".join(ch for ch in name.lower() if ch.isalnum())
        and "".join(ch for ch in name.lower() if ch.isalnum()).endswith("enable")
        and field.default is False
    )
    assert fields

    for name in fields:
        cfg = _cfg(enabled=True)
        setattr(cfg, name, True)
        with pytest.raises(RuntimeError, match="UNSUPPORTED_EXECUTION_LANE") as exc:
            _validate_escap7_gated_recurrent_config(
                cfg,
                policy_device=torch.device("cpu"),
                bank_device=torch.device("cpu"),
            )
        assert name in str(exc.value)


@pytest.mark.parametrize(
    "name",
    (
        "cxxrewrite_array_backend_authoritative_enable",
        "cxxrewrite_native_tick_enable",
        "cxxbackbone3_policy_aten_gru_authority_enable",
        "cxxcore6_authoritative_pilot_enable",
        "torch_compile_policy",
    ),
)
def test_audited_cxx_native_authority_and_compile_flags_fail_closed(name: str) -> None:
    cfg = _cfg(enabled=True)
    assert hasattr(cfg, name)
    setattr(cfg, name, True)
    with pytest.raises(RuntimeError, match="UNSUPPORTED_EXECUTION_LANE") as exc:
        _validate_escap7_gated_recurrent_config(
            cfg,
            policy_device=torch.device("cpu"),
            bank_device=torch.device("cpu"),
        )
    assert name in str(exc.value)


@pytest.mark.parametrize(
    "name",
    (
        "runnercore_supported_profile",
        "runnercore1_supported_profile",
        "runnercore2_supported_profile",
        "runnercore14_native_c1_projection_library_path",
    ),
)
def test_known_nonboolean_runnercore_selectors_fail_when_nondefault(name: str) -> None:
    cfg = _cfg(enabled=True)
    setattr(cfg, name, "different")
    with pytest.raises(RuntimeError, match="UNSUPPORTED_EXECUTION_LANE"):
        _validate_escap7_gated_recurrent_config(
            cfg,
            policy_device=torch.device("cpu"),
            bank_device=torch.device("cpu"),
        )


def test_authority_and_authoritative_boolean_spellings_are_both_discovered() -> None:
    fields = type(_cfg(enabled=True)).__dataclass_fields__
    assert "runnercore1_array_tick_authority_enable" in fields
    assert "runnercore_authoritative_enable" in fields
    for name in (
        "runnercore1_array_tick_authority_enable",
        "runnercore_authoritative_enable",
    ):
        cfg = _cfg(enabled=True)
        setattr(cfg, name, True)
        with pytest.raises(RuntimeError, match="UNSUPPORTED_EXECUTION_LANE"):
            _validate_escap7_gated_recurrent_config(
                cfg,
                policy_device=torch.device("cpu"),
                bank_device=torch.device("cpu"),
            )


def test_disabled_escap7_adds_no_lane_rejection() -> None:
    cfg = _cfg(enabled=False)
    cfg.policy_tick_stride = 8
    cfg.perf_sync_cuda = True
    cfg.native_runtime_enable = True
    cfg.runnercore_native_enable = True
    _validate_escap7_gated_recurrent_config(
        cfg,
        policy_device=torch.device("cuda"),
        bank_device=torch.device("cuda"),
    )


def test_capacity_contract_keeps_one_scalar_fixed_width_and_seven_actions() -> None:
    cfg = _cfg(enabled=True)
    contract = build_capacity_contract(sim_cfg=cfg)
    live = contract["structure"]["live_computation"]
    gated = next(
        row for row in live["supported_operators"] if row["operator"] == "gated_scalar_recurrent_v1"
    )

    assert gated["state_width"] == 1
    assert live["owned_state_dimensions_per_executable_module"] == 1
    assert live["fixed_storage_contract"] == {
        "state_map": "structural_live_module_state",
        "output_map": "structural_live_module_outputs",
        "state_width_per_module": 1,
        "new_arrays": 0,
        "observation_width_change": 0,
        "action_width_change": 0,
        "central_gru_width_change": 0,
    }
    assert contract["policy"]["escap7_fixed_shape_contract"] == {
        "central_gru_unchanged": True,
        "central_gru_hidden_width": 64,
        "mlp_hidden_width": 64,
        "policy_tensor_shape_change": 0,
    }
    assert contract["actions"]["count"] == len(ACTION_VOCABULARY) == 7
    assert contract["observations"]["escap7_observation_width_change"] == 0


def test_enabled_real_checkpoint_bundle_roundtrip_continues_exact_gated_tick(tmp_path) -> None:
    def make_sim(root) -> SimulationRunner:
        cfg = _cfg(enabled=True)
        cfg.seed = 18707
        cfg.generations = 1
        cfg.steps_per_generation = 1
        cfg.behavior_log_enable = False
        cfg.policy_deterministic_actions = True
        cfg.checkpoint_bundle_v2_period_gen = 0
        sim = SimulationRunner(
            EnvConfig(width=8, height=8),
            PopConfig(population_size=2),
            cfg,
            NoOpMetricsLogger(),
            str(root),
            perf=False,
        )
        sim.run(start_gen=0, skip_init=False)
        return sim

    identity = new_native_identity(
        context={
            "campaign_id": "escap7-checkpoint-test",
            "branch_id": "fixed-width-gated",
            "generation": 1,
            "tick": 0,
            "child_agent_id": 707,
            "parent_agent_ids": [1, 2],
            "lineage_token": "escap7-checkpoint-lineage",
        },
        operator_id="fixture",
        attempt_index=0,
        operator_role="checkpoint-gated-module",
    )
    gated_record = neutral_gated_recurrent_record()
    gated_record.update(
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
    module = {
        "module_id": "m000",
        "module_type": "generic",
        "params": {
            "live_operator": GATED_OPERATOR,
            "expression_scale": -0.75,
            "gated_recurrent": gated_record,
        },
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **deepcopy(identity),
        "latent_pos": list(identity["topology_pos"]),
    }
    genome = Genome.from_dict(
        {
            "schema_version": "modular_genome_v1",
            "b2_invalid_policy": "reject",
            "genes": {},
            "modules": [module],
            "connections": [],
        }
    )

    source = make_sim(tmp_path / "source")
    source_agent = next(iter(source.pop.agents.values()))
    source_agent.genome = genome
    instance_id = str(module["module_instance_id"])
    source_agent.structural_live_module_state = {instance_id: -0.4}
    source_agent.structural_live_module_outputs = {instance_id: 0.0}
    execute_agent_live_modules(
        source_agent,
        fatigue=0.6,
        gated_recurrent_enabled=True,
    )
    execute_agent_live_modules(
        source_agent,
        fatigue=0.35,
        gated_recurrent_enabled=True,
    )

    bundle = checkpoint_bundle_v2_save(
        source,
        str(tmp_path / "bundle"),
        generation_index=1,
        strict_replay_mode=True,
    )
    loaded = checkpoint_bundle_v2_load(bundle, strict=True)
    metadata = loaded["metadata"]["structural_gated_recurrent"]
    assert metadata == _escap7_gated_recurrent_metadata(source.sim_cfg)
    assert metadata["state_width"] == 1
    assert metadata["execution_rng_draws"] == 0

    restored = make_sim(tmp_path / "restored")
    checkpoint_bundle_v2_apply(restored, loaded, restore_rng=True)
    restored_agent = restored.pop.agents[source_agent.agent_id]
    assert restored_agent.genome.modules == source_agent.genome.modules
    assert restored_agent.structural_live_module_state == source_agent.structural_live_module_state
    assert restored_agent.structural_live_module_outputs == source_agent.structural_live_module_outputs
    assert restored_agent.structural_live_module_telemetry == source_agent.structural_live_module_telemetry

    source_rng = source.rng_manager.capture_snapshot(generation_index=source.gen)
    restored_rng = restored.rng_manager.capture_snapshot(generation_index=restored.gen)
    assert source_rng["streams"]["policy"]["indices"] == restored_rng["streams"]["policy"]["indices"]

    next_source = execute_agent_live_modules(
        source_agent,
        fatigue=0.2,
        gated_recurrent_enabled=True,
    )
    next_restored = execute_agent_live_modules(
        restored_agent,
        fatigue=0.2,
        gated_recurrent_enabled=True,
    )
    assert next_source.rest_residual == next_restored.rest_residual
    assert next_source.summary == next_restored.summary
    source_row = next_source.summary["state_rows"][0]
    restored_row = next_restored.summary["state_rows"][0]
    for field in (
        "reset_gate",
        "update_gate",
        "candidate",
        "state_after",
        "gated_effective_output",
    ):
        assert source_row[field] == restored_row[field]

    source_logits = np.zeros((1, len(ACTION_VOCABULARY)), dtype=np.float32)
    restored_logits = np.zeros_like(source_logits)
    add_rest_logit_residual_(
        source_logits,
        np.asarray([next_source.rest_residual], dtype=np.float32),
        rest_index=ACTION_VOCABULARY.index("rest"),
    )
    add_rest_logit_residual_(
        restored_logits,
        np.asarray([next_restored.rest_residual], dtype=np.float32),
        rest_index=ACTION_VOCABULARY.index("rest"),
    )
    assert source_logits.tobytes() == restored_logits.tobytes()
    assert int(np.argmax(source_logits[0])) == int(np.argmax(restored_logits[0]))

    mismatch = make_sim(tmp_path / "mismatch")
    mismatch.sim_cfg.structural_gated_recurrent_equation_version = "wrong-equation"
    with pytest.raises(CheckpointStrictError, match="gated_recurrent metadata/config mismatch"):
        checkpoint_bundle_v2_apply(mismatch, loaded, restore_rng=False)
