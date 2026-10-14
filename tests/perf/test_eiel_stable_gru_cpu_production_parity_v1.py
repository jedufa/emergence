from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from agents.batched_gru import forward_batched_gru
from agents.gru_policy import GRUPolicy
from engine.perf.recurrent_native.stable_gru_cpu_artifact_v1 import (
    CheckpointRowV1,
    DEFAULT_BUILD_MANIFEST,
    StableGruCpuArtifactError,
    StepConfigV1,
    build_prebuilt_stable_gru_cpu_oracle,
    load_prebuilt_stable_gru_cpu_oracle,
    repack_production_gru_parameter_tape_v1,
)
from engine.perf.recurrent_native_contract_v1 import load_recurrent_native_contract
from env.env_config import EnvConfig
from evolution.population_pkg.population_config import PopConfig
from perf.cxxbackbone2_policy_core_live_probe import _build_policy_input
from perf.cxxbackbone10_resident_state_owner import (
    ResidentRuntimeState,
    build_import_projection,
    compare_projection_digest,
    load_resident_state_owner_extension,
)
from simulation.metrics import MetricsLogger
from simulation.checkpoint_bundle_v2 import (
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.performance_profiles import FAST_EXPLORATORY_PROFILE_V2, apply_performance_profile
from simulation.runner import SimulationRunner
from simulation.runner_pkg.runner_config import SimConfig
from simulation.runnercore.action_cache import runnercore1_actions_strided
from simulation.runnercore.tick_state import export_to_agents, import_generation_state


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = PROJECT_ROOT / "engine/perf/native/recurrent"
SANITIZER_PROBE = Path(__file__).with_name("stable_gru_cpu_oracle_probe_v1.cpp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def oracle(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("stable_gru_cpu_prebuilt")
    first = build_prebuilt_stable_gru_cpu_oracle(root / "build1")
    second = build_prebuilt_stable_gru_cpu_oracle(root / "build2")
    assert first.read_bytes() == second.read_bytes()
    assert _sha256(first) == _sha256(second)
    return load_prebuilt_stable_gru_cpu_oracle(first)


def _float_pointer(values: np.ndarray):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _int32_pointer(values: np.ndarray):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _uint64_pointer(values: np.ndarray):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _pack_policy(policy: GRUPolicy) -> np.ndarray:
    tensors = (
        policy.encoder.weight,
        policy.encoder.bias,
        policy.gru.weight_ih,
        policy.gru.bias_ih,
        policy.gru.weight_hh,
        policy.gru.bias_hh,
        policy.head.weight,
        policy.head.bias,
    )
    return np.ascontiguousarray(
        torch.cat([value.detach().cpu().reshape(-1) for value in tensors]).numpy(),
        dtype=np.float32,
    )


def _unpack(packs: np.ndarray, observation_width: int) -> dict[str, torch.Tensor]:
    n = packs.shape[0]
    h = 64
    a = 7
    tensor = torch.from_numpy(np.ascontiguousarray(packs, dtype=np.float32))
    offset = 0

    def take(count: int, shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal offset
        value = tensor[:, offset : offset + count].reshape(n, *shape)
        offset += count
        return value

    result = {
        "encW": take(h * observation_width, (h, observation_width)),
        "encb": take(h, (h,)),
        "Wih": take(3 * h * h, (3 * h, h)),
        "bih": take(3 * h, (3 * h,)),
        "Whh": take(3 * h * h, (3 * h, h)),
        "bhh": take(3 * h, (3 * h,)),
        "headW": take(a * h, (a, h)),
        "headb": take(a, (a,)),
    }
    assert offset == packs.shape[1]
    return result


def _production_step(
    observations: np.ndarray,
    hidden: np.ndarray,
    packs: np.ndarray,
    widths: np.ndarray,
    aggression: np.ndarray,
    temperature: np.ndarray,
    drives: np.ndarray,
    gain: np.ndarray,
    weights: tuple[float, float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.from_numpy(np.ascontiguousarray(observations, dtype=np.float32))
    h_prev = torch.from_numpy(np.ascontiguousarray(hidden, dtype=np.float32))
    width_t = torch.from_numpy(np.asarray(widths, dtype=np.int64))
    mask = (torch.arange(64).unsqueeze(0) < width_t.unsqueeze(1)).to(torch.float32)
    h_prev = h_prev * mask
    unpacked = _unpack(packs, observations.shape[1])
    logits, raw_hidden = forward_batched_gru(
        x=x,
        h_prev=h_prev,
        **unpacked,
        attack_bias=torch.from_numpy(np.ascontiguousarray(aggression, dtype=np.float32)),
        action_temperature=torch.from_numpy(np.ascontiguousarray(temperature, dtype=np.float32)),
    )
    d = torch.from_numpy(np.ascontiguousarray(drives, dtype=np.float32))
    g = torch.from_numpy(np.ascontiguousarray(gain, dtype=np.float32)).clamp(min=0.0)
    logits[:, 4] += g * (weights[0] * d[:, 0])
    logits[:, 5] += g * (weights[1] * d[:, 1] + weights[2] * d[:, 2])
    logits[:, 6] -= g * (weights[3] * d[:, 1] + weights[4] * d[:, 3])
    settled = raw_hidden * mask
    actions = torch.argmax(logits, dim=1)
    return (
        settled.detach().cpu().numpy().copy(),
        logits.detach().cpu().numpy().copy(),
        actions.detach().cpu().numpy().astype(np.int32, copy=True),
    )


def _native_step(
    oracle,
    observations: np.ndarray,
    hidden: np.ndarray,
    packs: np.ndarray,
    widths: np.ndarray,
    aggression: np.ndarray,
    temperature: np.ndarray,
    drives: np.ndarray,
    gain: np.ndarray,
    weights: tuple[float, float, float, float, float],
    *,
    output_sentinel: float = -900.0,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    observations = np.ascontiguousarray(observations, dtype=np.float32)
    hidden = np.ascontiguousarray(hidden, dtype=np.float32)
    packs = np.ascontiguousarray(packs, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.int64)
    aggression = np.asarray(aggression, dtype=np.float32)
    temperature = np.asarray(temperature, dtype=np.float32)
    drives = np.ascontiguousarray(drives, dtype=np.float32)
    gain = np.asarray(gain, dtype=np.float32)
    rows, observation_width = observations.shape
    configs = (StepConfigV1 * rows)()
    for row in range(rows):
        configs[row].contract_version = 1
        configs[row].observation_size = observation_width
        configs[row].active_memory_size = int(widths[row])
        configs[row].aggression = float(aggression[row])
        configs[row].temperature = float(temperature[row])
        configs[row].drives[:] = tuple(float(value) for value in drives[row])
        configs[row].gain = float(gain[row])
        configs[row].hunger_gather_weight = weights[0]
        configs[row].pain_rest_weight = weights[1]
        configs[row].fatigue_rest_weight = weights[2]
        configs[row].pain_attack_weight = weights[3]
        configs[row].fear_attack_weight = weights[4]
    scratch_hidden = np.empty((rows, 64), dtype=np.float32)
    scratch_logits = np.empty((rows, 7), dtype=np.float32)
    scratch_actions = np.empty((rows,), dtype=np.int32)
    output_hidden = np.full((rows, 64), output_sentinel, dtype=np.float32)
    output_logits = np.full((rows, 7), output_sentinel - 1.0, dtype=np.float32)
    output_actions = np.full((rows,), -902, dtype=np.int32)
    status = int(oracle.library.eiel_stable_gru_forward_batch_v1(
        configs,
        rows,
        _float_pointer(observations), observations.size,
        _float_pointer(hidden), hidden.size,
        _float_pointer(packs), packs.size,
        rows,
        _float_pointer(scratch_hidden), scratch_hidden.size,
        _float_pointer(scratch_logits), scratch_logits.size,
        _int32_pointer(scratch_actions), scratch_actions.size,
        _float_pointer(output_hidden), output_hidden.size,
        _float_pointer(output_logits), output_logits.size,
        _int32_pointer(output_actions), output_actions.size,
    ))
    return status, output_hidden, output_logits, output_actions


def _fixture(rows: int = 5, observation_width: int = 11):
    policies: list[GRUPolicy] = []
    for seed in range(rows):
        torch.manual_seed(1000 + seed)
        policies.append(GRUPolicy(observation_width, 64, 7).eval())
    packs = np.stack([_pack_policy(policy) for policy in policies])
    rng = np.random.default_rng(30303)
    observations = rng.normal(0.0, 0.5, size=(rows, observation_width)).astype(np.float32)
    hidden = rng.normal(0.0, 0.25, size=(rows, 64)).astype(np.float32)
    widths = np.asarray([0, 8, 16, 32, 64][:rows], dtype=np.int64)
    aggression = rng.normal(0.0, 0.2, size=rows).astype(np.float32)
    temperature = np.asarray([0.01, 0.05, 0.051, 0.7, 1.3][:rows], dtype=np.float32)
    drives = rng.uniform(0.0, 1.0, size=(rows, 4)).astype(np.float32)
    gain = np.asarray([-1.0, 0.0, 0.25, 0.75, 1.5][:rows], dtype=np.float32)
    weights = (1.2, 0.9, 1.0, 1.2, 1.4)
    return policies, packs, observations, hidden, widths, aggression, temperature, drives, gain, weights


def _live_gru_runner(
    path: Path,
    *,
    stride: int = 2,
    patch: bool = False,
    wdl: bool = False,
) -> tuple[SimulationRunner, list[object]]:
    env_cfg = EnvConfig(width=8, height=8, view_radius=1)
    env_cfg.enable_biomes = False
    env_cfg.food_spawn_per_step = 0
    env_cfg.max_food = 80
    env_cfg.reproduce_try_prob = 0.0
    env_cfg.patch_world_enable = bool(patch)
    env_cfg.patch_world_observation_enable = bool(patch)
    env_cfg.wdl_delayed_hazard_enable = bool(wdl)
    env_cfg.wdl_delayed_hazard_cue_enable = bool(wdl)
    env_cfg.wdl_delayed_hazard_observation_enable = bool(wdl)
    env_cfg.wdl_delayed_hazard_damage_enable = bool(wdl)
    env_cfg.wdl_delayed_hazard_deterministic_schedule_enable = bool(wdl)
    env_cfg.wdl_delayed_hazard_observation_dropout_rate = 0.0
    env_cfg.wdl_delayed_hazard_delay_steps = 2
    env_cfg.wdl_delayed_hazard_cue_duration_steps = 1
    env_cfg.wdl_delayed_hazard_active_duration_steps = 1
    env_cfg.wdl_delayed_hazard_event_period_steps = 5
    env_cfg.wdl_delayed_hazard_radius = 1
    env_cfg.wdl_delayed_hazard_intensity = 0.1
    env_cfg.reproduce_requires_mate = False
    env_cfg.prediction_demand_enable = False
    env_cfg.prediction_demand_observation_enable = False
    cfg = SimConfig(
        seed=4242,
        generations=2,
        steps_per_generation=3,
        device="cpu",
        bank_device="cpu",
        behavior_log_enable=False,
        policy_deterministic_actions=True,
        policy_tick_stride=stride,
    )
    apply_performance_profile(cfg, FAST_EXPLORATORY_PROFILE_V2)
    cfg.phase_e_e1_enable = False
    cfg.phase_e_e2_enable = False
    cfg.epi_enable = False
    cfg.enable_cognitive_budget_ledger = False
    cfg.chemistry_functional_coupling_enable = False
    cfg.checkpoint_bundle_v2_period_gen = 0
    cfg.maintenance_enable = False
    cfg.shock_pulsed = False
    cfg.shock_start_prob = 0.0
    cfg.shock_period = 0
    cfg.shock_duration = 0
    cfg.cxxbackbone2_policy_core_live_probe_enable = True
    cfg.cxxbackbone2_policy_core_live_probe_write_summary = False
    cfg.cxxbackbone3_policy_aten_gru_authority_enable = False
    cfg.cxxbackbone4_policy_aten_mlp_authority_enable = False
    cfg.native_runtime_cxx26_scope_lock_enable = True
    cfg.genome_c1_enable = False
    cfg.genome_c2_enable = False
    cfg.policy_weight_mutation_enable = False
    runner = SimulationRunner(
        env_cfg=env_cfg,
        pop_cfg=PopConfig(population_size=5),
        sim_cfg=cfg,
        logger=MetricsLogger(str(path)),
        run_dir=str(path),
        perf=True,
    )
    runner.sim_cfg.native_runtime_cxx26_scope_lock_enable = True
    runner.sim_cfg.genome_c1_enable = False
    runner.sim_cfg.genome_c2_enable = False
    runner.sim_cfg.policy_weight_mutation_enable = False
    runner.pop.init_random()
    agents = sorted(runner.pop.alive_agents(), key=lambda item: int(item.agent_id))
    for index, agent in enumerate(agents):
        agent.genome.use_gru = True
        agent.genome.memory_dim = [0, 8, 16, 32, 64][index]
        torch.manual_seed(5000 + index)
        agent.policy = GRUPolicy(int(runner._obs_size), 64, int(runner._out)).eval()
    runner._ensure_bank()
    runner._place_agents()
    runner._mark_bank_dirty()
    runner._sync_gru_bank_for_agents(agents)
    return runner, agents


def _native_multigeneration_schedule() -> dict[str, object]:
    return {
        "schema_version": "eiel_native_snapshot_schedule_v1",
        "progress_every_generations": 1,
        "report_every_generations": 0,
        "checkpoint_every_generations": 0,
        "checkpoint_on_cancel": True,
        "explicit_next_generations": [2],
        "emit_final_snapshot": True,
        "typed_lifecycle_mode": "cursor_bounded_delta",
        "mailbox_capacity": 1,
    }


def _assert_captured_live_refill_matches_native(oracle, runner, captured: dict) -> None:
    inputs = _build_policy_input(
        runner,
        captured["state"],
        slots=captured["slots"],
        obs_np=captured["obs_np"],
        hmem_before=captured["hmem_before"],
        k=captured["k"],
    )
    slots = np.asarray(inputs.row_slots, dtype=np.int64)
    weights = (
        inputs.w_hunger_gather,
        inputs.w_pain_rest,
        inputs.w_fatigue_rest,
        inputs.w_pain_attack,
        inputs.w_fear_attack,
    )
    status, native_hidden, native_logits, native_actions = _native_step(
        oracle,
        inputs.observations,
        inputs.hmem[slots],
        inputs.gru_pack[slots],
        inputs.effective_memory_dim,
        inputs.aggression,
        inputs.temperature,
        inputs.drives,
        inputs.gain,
        weights,
    )
    assert status == 0
    expected_hidden = runner._Hmem.detach().cpu().numpy()[slots]
    np.testing.assert_allclose(native_hidden, expected_hidden, rtol=0.0, atol=5.0e-7)
    np.testing.assert_array_equal(native_actions, captured["produced_actions"][:, 0])
    production = _production_step(
        inputs.observations,
        inputs.hmem[slots],
        inputs.gru_pack[slots],
        inputs.effective_memory_dim,
        inputs.aggression,
        inputs.temperature,
        inputs.drives,
        inputs.gain,
        weights,
    )
    np.testing.assert_allclose(native_logits, production[1], rtol=0.0, atol=5.0e-6)


def test_prebuilt_artifact_is_reproducible_hash_verified_and_not_production_routed(oracle) -> None:
    manifest = oracle.manifest
    assert oracle.artifact_sha256 == manifest["artifact"]["sha256"]
    assert manifest["runtime_jit"] is False
    assert manifest["production_routing_qualified"] is False
    assert "production routing" in manifest["qualification"]["does_not_authorize"]
    assert oracle.library.eiel_stable_gru_cpu_oracle_abi_version_v1() == 1


def test_loader_rejects_tampered_artifact_and_manifest_drift(tmp_path: Path, oracle) -> None:
    tampered = tmp_path / oracle.artifact_path.name
    data = bytearray(oracle.artifact_path.read_bytes())
    data[len(data) // 2] ^= 1
    tampered.write_bytes(data)
    with pytest.raises(StableGruCpuArtifactError, match="artifact hash mismatch"):
        load_prebuilt_stable_gru_cpu_oracle(tampered)

    payload = json.loads(DEFAULT_BUILD_MANIFEST.read_text(encoding="utf-8"))
    payload["sources"][0]["sha256"] = "0" * 64
    drifted = tmp_path / "drifted_manifest.json"
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StableGruCpuArtifactError, match="source hash mismatch"):
        load_prebuilt_stable_gru_cpu_oracle(oracle.artifact_path, manifest_path=drifted)

    payload = json.loads(DEFAULT_BUILD_MANIFEST.read_text(encoding="utf-8"))
    payload["c_abi"]["version"] = 2
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StableGruCpuArtifactError, match="C ABI receipt mismatch"):
        load_prebuilt_stable_gru_cpu_oracle(oracle.artifact_path, manifest_path=drifted)

    payload = json.loads(DEFAULT_BUILD_MANIFEST.read_text(encoding="utf-8"))
    payload["qualification"]["float_gate"]["one_step_hidden_atol"] = 1.0
    drifted.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StableGruCpuArtifactError, match="float gate drifted"):
        load_prebuilt_stable_gru_cpu_oracle(oracle.artifact_path, manifest_path=drifted)

    symlink = tmp_path / "symlink" / oracle.artifact_path.name
    symlink.parent.mkdir()
    symlink.symlink_to(oracle.artifact_path)
    with pytest.raises(StableGruCpuArtifactError, match="must not be a symlink"):
        load_prebuilt_stable_gru_cpu_oracle(symlink)


def test_real_gru_policy_pack_order_and_width_match_production_bank(oracle) -> None:
    policy = GRUPolicy(11, 64, 7).eval()
    packed = _pack_policy(policy)
    expected = torch.cat([
        policy.encoder.weight.detach().reshape(-1), policy.encoder.bias.detach().reshape(-1),
        policy.gru.weight_ih.detach().reshape(-1), policy.gru.bias_ih.detach().reshape(-1),
        policy.gru.weight_hh.detach().reshape(-1), policy.gru.bias_hh.detach().reshape(-1),
        policy.head.weight.detach().reshape(-1), policy.head.bias.detach().reshape(-1),
    ]).numpy()
    np.testing.assert_array_equal(packed, expected)
    count = ctypes.c_int64(-1)
    assert oracle.library.eiel_stable_gru_packed_parameter_count_v1(11, ctypes.byref(count)) == 0
    assert count.value == packed.size == load_recurrent_native_contract().packed_width(11)


def test_production_parameter_iteration_tape_maps_to_native_mutation_order(oracle) -> None:
    torch.manual_seed(777)
    policy = GRUPolicy(11, 64, 7).eval()
    production_parent = np.ascontiguousarray(
        torch.cat([value.detach().reshape(-1) for value in policy.parameters()]).numpy(),
        dtype=np.float32,
    )
    draws = np.linspace(-0.75, 0.75, production_parent.size, dtype=np.float32)
    mask = (np.arange(production_parent.size) % 3 != 0).astype(np.uint8)
    scale = np.float32(0.03125)
    production_child = production_parent + scale * draws * mask.astype(np.float32)
    packed_parent = np.ascontiguousarray(
        repack_production_gru_parameter_tape_v1(
            production_parent, observation_width=11
        ), dtype=np.float32,
    )
    packed_draws = np.ascontiguousarray(
        repack_production_gru_parameter_tape_v1(draws, observation_width=11),
        dtype=np.float32,
    )
    packed_mask = np.ascontiguousarray(
        repack_production_gru_parameter_tape_v1(mask, observation_width=11),
        dtype=np.uint8,
    )
    expected = np.ascontiguousarray(
        repack_production_gru_parameter_tape_v1(
            production_child, observation_width=11
        ), dtype=np.float32,
    )
    scratch = np.full_like(packed_parent, -80.0)
    child = np.full_like(packed_parent, -81.0)
    status = oracle.library.eiel_stable_gru_mutate_v1(
        _float_pointer(packed_parent), packed_parent.size,
        _float_pointer(packed_draws),
        packed_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        scale,
        _float_pointer(scratch), scratch.size,
        _float_pointer(child), child.size,
    )
    assert status == 0
    np.testing.assert_array_equal(child, expected)


def test_mixed_production_batch_matches_native_all_memory_widths_and_permutation(oracle) -> None:
    _, packs, obs, hidden, widths, aggression, temperature, drives, gain, weights = _fixture()
    for order in (np.arange(5), np.asarray([4, 0, 3, 1, 2])):
        args = tuple(value[order] for value in (
            obs, hidden, packs, widths, aggression, temperature, drives, gain
        ))
        reference = _production_step(*args, weights)
        status, native_hidden, native_logits, native_actions = _native_step(oracle, *args, weights)
        assert status == 0
        np.testing.assert_allclose(native_hidden, reference[0], rtol=0.0, atol=5.0e-7)
        np.testing.assert_allclose(native_logits, reference[1], rtol=0.0, atol=5.0e-6)
        np.testing.assert_array_equal(native_actions, reference[2])
        for row, width in enumerate(widths[order]):
            assert np.all(native_hidden[row, int(width):] == np.float32(0.0))


def test_production_sequences_match_native_for_every_memory_width(oracle) -> None:
    _, packs, _, hidden, widths, aggression, temperature, drives, gain, weights = _fixture()
    native_hidden = hidden.copy()
    production_hidden = hidden.copy()
    rng = np.random.default_rng(40404)
    for _ in range(32):
        observations = rng.normal(0.0, 0.5, size=(5, 11)).astype(np.float32)
        expected_hidden, expected_logits, expected_actions = _production_step(
            observations, production_hidden, packs, widths, aggression, temperature,
            drives, gain, weights,
        )
        status, actual_hidden, actual_logits, actual_actions = _native_step(
            oracle, observations, native_hidden, packs, widths, aggression,
            temperature, drives, gain, weights,
        )
        assert status == 0
        np.testing.assert_allclose(actual_hidden, expected_hidden, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(actual_logits, expected_logits, rtol=0.0, atol=1.0e-5)
        np.testing.assert_array_equal(actual_actions, expected_actions)
        native_hidden = actual_hidden
        production_hidden = expected_hidden


def test_temperature_boundaries_and_zero_tie_match_production(oracle) -> None:
    rows = 6
    observation_width = 3
    count = load_recurrent_native_contract().packed_width(observation_width)
    packs = np.zeros((rows, count), dtype=np.float32)
    observations = np.zeros((rows, observation_width), dtype=np.float32)
    hidden = np.zeros((rows, 64), dtype=np.float32)
    widths = np.asarray([0, 8, 16, 32, 64, 64], dtype=np.int64)
    aggression = np.zeros(rows, dtype=np.float32)
    temperature = np.asarray([-1.0, 0.0, 0.049, 0.05, 0.051, 1.0], dtype=np.float32)
    drives = np.zeros((rows, 4), dtype=np.float32)
    gain = np.full(rows, -2.0, dtype=np.float32)
    weights = (1.2, 0.9, 1.0, 1.2, 1.4)
    reference = _production_step(
        observations, hidden, packs, widths, aggression, temperature, drives, gain, weights)
    status, actual_hidden, actual_logits, actual_actions = _native_step(
        oracle, observations, hidden, packs, widths, aggression, temperature, drives, gain, weights)
    assert status == 0
    np.testing.assert_array_equal(actual_hidden, reference[0])
    np.testing.assert_array_equal(actual_logits, reference[1])
    np.testing.assert_array_equal(actual_actions, np.zeros(rows, dtype=np.int32))


def test_multirow_failure_is_transactional(oracle) -> None:
    _, packs, obs, hidden, widths, aggression, temperature, drives, gain, weights = _fixture()
    obs[2, 3] = np.nan
    status, output_hidden, output_logits, output_actions = _native_step(
        oracle, obs, hidden, packs, widths, aggression, temperature, drives, gain, weights)
    assert status == 11
    assert np.all(output_hidden == np.float32(-900.0))
    assert np.all(output_logits == np.float32(-901.0))
    assert np.all(output_actions == np.int32(-902))


def test_native_artifact_lifecycle_and_fresh_owner_checkpoint_are_exact(oracle) -> None:
    hidden = np.arange(4 * 64, dtype=np.float32).reshape(4, 64)
    tokens = np.asarray([101, 0, 303, 404], dtype=np.uint64)
    original_death = hidden[2].copy()
    lib = oracle.library
    assert lib.eiel_stable_gru_lifecycle_v1(
        _float_pointer(hidden), hidden.size, _uint64_pointer(tokens), tokens.size,
        4, 1, 0, 202, 2,
    ) == 0
    assert lib.eiel_stable_gru_lifecycle_v1(
        _float_pointer(hidden), hidden.size, _uint64_pointer(tokens), tokens.size,
        4, 2, 303, 0, 3,
    ) == 0
    assert lib.eiel_stable_gru_lifecycle_v1(
        _float_pointer(hidden), hidden.size, _uint64_pointer(tokens), tokens.size,
        4, 3, 404, 405, 4,
    ) == 0
    assert np.all(hidden[1] == 0.0) and np.all(hidden[3] == 0.0)
    np.testing.assert_array_equal(hidden[2], original_death)
    np.testing.assert_array_equal(tokens, [101, 202, 0, 405])

    checkpoint = CheckpointRowV1()
    assert lib.eiel_stable_gru_checkpoint_project_v1(
        _float_pointer(hidden), hidden.size, _uint64_pointer(tokens), tokens.size,
        4, 1, 202, ctypes.byref(checkpoint),
    ) == 0
    destination = np.full((4, 64), -7.0, dtype=np.float32)
    destination_tokens = np.asarray([11, 22, 999, 44], dtype=np.uint64)
    checkpoint.agent_index = 2
    assert lib.eiel_stable_gru_checkpoint_restore_v1(
        ctypes.byref(checkpoint), _float_pointer(destination), destination.size,
        _uint64_pointer(destination_tokens), destination_tokens.size, 4, 999,
    ) == 0
    np.testing.assert_array_equal(destination[2], hidden[1])
    assert destination_tokens.tolist() == [11, 22, 202, 44]
    assert np.all(destination[[0, 1, 3]] == -7.0)


def test_real_runner_pack_refill_cache_hit_and_exhaustion_match_native(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oracle,
) -> None:
    runner, agents = _live_gru_runner(tmp_path / "live_cache")
    slots = np.asarray([int(agent.slot) for agent in agents], dtype=np.int64)
    for agent in agents:
        np.testing.assert_array_equal(
            runner._gru_pack[int(agent.slot)].detach().cpu().numpy(),
            _pack_policy(agent.policy),
        )
    state = import_generation_state(runner, generation=0, profile="supported_fast_runner_v2")
    ids = np.asarray([int(agent.agent_id) for agent in agents], dtype=np.int32)
    captures: list[dict] = []

    def capture(_runner, _state, **kwargs):
        captures.append({"state": _state, **kwargs})
        return {"supported": False, "executed_native": False, "hidden_fallback": False}

    monkeypatch.setattr(
        "perf.cxxbackbone2_policy_core_live_probe.run_policy_core_live_probe", capture
    )
    hmem_initial = runner._Hmem.detach().cpu().numpy().copy()
    first = runnercore1_actions_strided(runner, state, ids, buckets={})
    assert len(captures) == 1
    _assert_captured_live_refill_matches_native(oracle, runner, captures[-1])
    hmem_after_refill = runner._Hmem.detach().cpu().numpy().copy()
    assert not np.array_equal(hmem_after_refill[slots[1:]], hmem_initial[slots[1:]])
    assert state.cache_cursor[slots].tolist() == [1] * 5

    second = runnercore1_actions_strided(runner, state, ids, buckets={})
    assert len(captures) == 1
    np.testing.assert_array_equal(second, first)
    np.testing.assert_array_equal(runner._Hmem.detach().cpu().numpy(), hmem_after_refill)
    assert state.cache_cursor[slots].tolist() == [2] * 5

    third = runnercore1_actions_strided(runner, state, ids, buckets={})
    assert len(captures) == 2
    _assert_captured_live_refill_matches_native(oracle, runner, captures[-1])
    np.testing.assert_array_equal(third, first)
    assert state.cache_cursor[slots].tolist() == [1] * 5
    assert state.counters.action_cache_hits == 5
    assert state.counters.action_cache_misses == 10


def test_resident_cpu_owner_refill_uses_stable_gru_core_exact_actions(
    tmp_path: Path,
) -> None:
    runner, agents = _live_gru_runner(tmp_path / "resident_owner")
    runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
    owner = ResidentRuntimeState.from_runner(
        runner,
        {"flat": True, "patch": False, "gru": True, "churn": False},
        generation=0,
        tick=0,
        build_directory=tmp_path / "owner_build",
    )
    state = import_generation_state(
        runner, generation=0, profile="supported_fast_runner_v2"
    )
    agent_ids = np.asarray(
        [int(agent.agent_id) for agent in agents], dtype=np.int32
    )
    slots = np.asarray([int(agent.slot) for agent in agents], dtype=np.int64)

    expected_actions = runnercore1_actions_strided(
        runner, state, agent_ids, buckets={}
    )
    expected_hidden = runner._Hmem.detach().cpu().numpy().copy()
    result = owner.refill_flat_policy_cpu()
    checkpoint = owner.export_checkpoint_projection()

    assert result["executed"] is True
    assert result["hidden_fallback"] is False
    assert int(result["mlp_rows"]) == 0
    assert int(result["gru_rows"]) == len(agents)
    np.testing.assert_array_equal(
        result["actions"].detach().cpu().numpy(), expected_actions
    )
    np.testing.assert_allclose(
        checkpoint["tensors"]["hmem"].detach().cpu().numpy()[slots],
        expected_hidden[slots],
        rtol=0.0,
        atol=5.0e-7,
    )


def test_resident_patch_cpu_owner_refill_uses_stable_gru_core_exact_actions(
    tmp_path: Path,
) -> None:
    runner, agents = _live_gru_runner(
        tmp_path / "resident_patch_owner",
        patch=True,
    )
    runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
    owner = ResidentRuntimeState.from_runner(
        runner,
        {"flat": False, "patch": True, "gru": True, "churn": False},
        generation=0,
        tick=0,
        build_directory=tmp_path / "owner_build",
    )
    state = import_generation_state(
        runner, generation=0, profile="supported_fast_runner_v2"
    )
    agent_ids = np.asarray(
        [int(agent.agent_id) for agent in agents], dtype=np.int32
    )
    slots = np.asarray([int(agent.slot) for agent in agents], dtype=np.int64)

    expected_actions = runnercore1_actions_strided(
        runner, state, agent_ids, buckets={}
    )
    expected_hidden = runner._Hmem.detach().cpu().numpy().copy()
    result = owner.refill_patch_policy_cpu()
    checkpoint = owner.export_checkpoint_projection()

    assert result["executed"] is True
    assert result["hidden_fallback"] is False
    assert int(result["mlp_rows"]) == 0
    assert int(result["gru_rows"]) == len(agents)
    np.testing.assert_array_equal(
        result["actions"].detach().cpu().numpy(), expected_actions
    )
    np.testing.assert_allclose(
        checkpoint["tensors"]["hmem"].detach().cpu().numpy()[slots],
        expected_hidden[slots],
        rtol=0.0,
        atol=5.0e-7,
    )


def test_resident_patch_cpu_owner_gru_g1_matches_reference_trajectory(
    tmp_path: Path,
) -> None:
    runner, _agents = _live_gru_runner(
        tmp_path / "resident_patch_g1",
        patch=True,
    )
    runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
    capability = {"flat": False, "patch": True, "gru": True, "churn": False}
    owner = ResidentRuntimeState.from_runner(
        runner,
        capability,
        generation=0,
        tick=0,
        build_directory=tmp_path / "owner_build",
    )

    runner._run_ticks(
        gen=0,
        buckets={},
        injury_recover_mult=1.0,
        deaths_this_gen=0,
        successful_births=0,
        failed_births=0,
        gen_peak_injury=0.0,
        rest_streaks={},
    )
    reference = build_import_projection(
        runner,
        capability,
        generation=0,
        tick=3,
    )
    result = owner.execute_generation_patch_cpu(0, 3)
    native = owner.export_checkpoint_projection()

    assert result["executed"] is True
    assert result["hidden_fallback"] is False
    assert int(result["native_tick_loop_count"]) == 3
    for name in (
        "slot_to_agent_id",
        "slot_tokens",
        "alive_mask",
        "pos_y",
        "pos_x",
        "agent_grid",
        "food_grid",
        "patch_food_amount",
        "patch_scent",
        "patch_waste",
        "patch_movement_cost",
        "policy_use_gru",
        "policy_mem_dim",
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
    ):
        torch.testing.assert_close(
            native["tensors"][name],
            reference.tensors[name],
            rtol=0.0,
            atol=0.0,
            msg=lambda message, field=name: f"{field}: {message}",
        )
    for name in ("energy", "health", "gru_upkeep_cost"):
        torch.testing.assert_close(
            native["tensors"][name],
            reference.tensors[name],
            rtol=0.0,
            atol=0.0,
            msg=lambda message, field=name: f"{field}: {message}",
        )
    torch.testing.assert_close(
        native["tensors"]["hmem"],
        reference.tensors["hmem"],
        rtol=0.0,
        atol=2.0e-6,
    )


def test_resident_patch_cpu_owner_memory_ablation_is_clean_and_lifecycle_stable(
    tmp_path: Path,
) -> None:
    normal_runner, _normal_agents = _live_gru_runner(
        tmp_path / "memory_normal",
        patch=True,
    )
    ablated_runner, _ablated_agents = _live_gru_runner(
        tmp_path / "memory_ablated",
        patch=True,
    )
    capability = {"flat": False, "patch": True, "gru": True, "churn": True}
    for runner in (normal_runner, ablated_runner):
        runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
        runner.pop_cfg.cull_fraction = 0.4
        runner.pop_cfg.parent_fraction = 1.0
    ablated_runner.sim_cfg.force_memory_disabled = True
    ablated_runner.sim_cfg.reset_memory_each_tick = True

    normal_initial = build_import_projection(
        normal_runner,
        capability,
        generation=0,
        tick=0,
    )
    ablated_initial = build_import_projection(
        ablated_runner,
        capability,
        generation=0,
        tick=0,
    )
    for name in (
        "slot_to_agent_id",
        "alive_mask",
        "pos_y",
        "pos_x",
        "agent_grid",
        "food_grid",
        "patch_food_amount",
        "patch_scent",
        "patch_waste",
        "policy_use_gru",
        "genome_memory_dim",
        "gru_pack",
    ):
        torch.testing.assert_close(
            normal_initial.tensors[name],
            ablated_initial.tensors[name],
            rtol=0.0,
            atol=0.0,
            msg=lambda message, field=name: f"matched initial {field}: {message}",
        )
    assert ablated_initial.metadata["force_memory_disabled"] is True
    assert ablated_initial.metadata["reset_memory_each_tick"] is True
    assert normal_initial.metadata["force_memory_disabled"] is False
    assert normal_initial.metadata["reset_memory_each_tick"] is False
    assert torch.count_nonzero(ablated_initial.tensors["policy_mem_dim"]) == 0
    assert torch.count_nonzero(ablated_initial.tensors["gru_upkeep_cost"]) == 0

    build_directory = tmp_path / "owner_build"
    normal_probe = ResidentRuntimeState.from_runner(
        normal_runner,
        capability,
        generation=0,
        tick=0,
        build_directory=build_directory,
    )
    ablated_probe = ResidentRuntimeState.from_runner(
        ablated_runner,
        capability,
        generation=0,
        tick=0,
        build_directory=build_directory,
    )
    ablated_initial_checkpoint = ablated_probe.export_checkpoint_projection()
    normal_probe.refill_patch_policy_cpu()
    ablated_probe.refill_patch_policy_cpu()
    normal_after_refill = normal_probe.export_checkpoint_projection()
    ablated_after_refill = ablated_probe.export_checkpoint_projection()
    live = ablated_after_refill["tensors"]["alive_mask"]
    assert torch.count_nonzero(normal_after_refill["tensors"]["hmem"][live]) > 0
    assert torch.count_nonzero(ablated_after_refill["tensors"]["hmem"][live]) == 0
    assert (
        normal_after_refill["metadata"]["native_runtime_numpy_policy_state"]
        == ablated_after_refill["metadata"]["native_runtime_numpy_policy_state"]
    )

    ablated_owner = ResidentRuntimeState.from_checkpoint_projection(
        ablated_initial_checkpoint,
        build_directory=build_directory,
    )
    ablated_runner.pop._eiel_current_generation_index = 0
    generation_start = ablated_runner._start_generation(0)
    ablated_runner._run_ticks(
        gen=0,
        buckets=generation_start[0],
        injury_recover_mult=generation_start[2],
        deaths_this_gen=generation_start[3],
        successful_births=generation_start[4],
        failed_births=generation_start[10],
        gen_peak_injury=generation_start[9],
        rest_streaks=generation_start[11],
    )
    reference_before_turnover = build_import_projection(
        ablated_runner,
        capability,
        generation=0,
        tick=3,
    )
    ablated_owner.execute_generation_patch_with_population_birth_commit_cpu(
        0,
        3,
        max_births=32,
    )
    native_before_turnover = ablated_owner.export_checkpoint_projection()
    for name in (
        "slot_to_agent_id",
        "alive_mask",
        "pos_y",
        "pos_x",
        "agent_grid",
        "action_cache_actions",
        "action_cache_cursor",
        "action_cache_next",
        "policy_mem_dim",
        "gru_upkeep_cost",
        "energy",
        "health",
    ):
        torch.testing.assert_close(
            native_before_turnover["tensors"][name],
            reference_before_turnover.tensors[name],
            rtol=0.0,
            atol=0.0,
            msg=lambda message, field=name: f"ablated trajectory {field}: {message}",
        )
    assert torch.count_nonzero(native_before_turnover["tensors"]["hmem"]) == 0

    ablated_owner.execute_population_generation_turnover_cpu()
    native_after_turnover = ablated_owner.export_checkpoint_projection()
    alive_after = native_after_turnover["tensors"]["alive_mask"]
    assert torch.all(native_after_turnover["tensors"]["policy_use_gru"][alive_after])
    assert torch.count_nonzero(
        native_after_turnover["tensors"]["genome_memory_dim"][alive_after]
    ) > 0
    assert torch.count_nonzero(
        native_after_turnover["tensors"]["policy_mem_dim"][alive_after]
    ) == 0
    assert torch.count_nonzero(
        native_after_turnover["tensors"]["gru_upkeep_cost"][alive_after]
    ) == 0
    assert torch.count_nonzero(native_after_turnover["tensors"]["hmem"][alive_after]) == 0


@pytest.mark.parametrize(
    "force_memory_disabled",
    [False, True],
    ids=["normal", "memory_disabled"],
)
def test_native_multigeneration_controller_owns_wdl_gru_g2_deterministically(
    tmp_path: Path,
    force_memory_disabled: bool,
) -> None:
    module = load_resident_state_owner_extension(
        build_directory=tmp_path / "owner_build"
    )
    assert module.EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_CAPABILITIES[
        "stable_gru_multigeneration_cpu_v1"
    ] is True
    capability = {
        "flat": False,
        "patch": True,
        "wdl": True,
        "gru": True,
        "churn": True,
    }
    final_projections = []
    for repetition in range(2):
        runner, _agents = _live_gru_runner(
            tmp_path / f"controller_{force_memory_disabled}_{repetition}",
            patch=True,
            wdl=True,
        )
        runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
        runner.sim_cfg.force_memory_disabled = force_memory_disabled
        runner.sim_cfg.reset_memory_each_tick = force_memory_disabled
        runner.pop_cfg.cull_fraction = 0.4
        runner.pop_cfg.parent_fraction = 1.0
        projection = build_import_projection(
            runner,
            capability,
            generation=0,
            tick=0,
        )
        projection.metadata[
            "native_runtime_architecture_b_compact_reporting_only_v1"
        ] = True
        projection.metadata["native_runtime_memory_science_reporting_v1"] = True
        controller = module.FlatCpuArchitectureBControllerV1(
            projection.tensors,
            copy.deepcopy(projection.metadata),
            f"stable-memory-{force_memory_disabled}-{repetition}",
            False,
        )
        result = controller.run_generations(
            0,
            2,
            3,
            32,
            _native_multigeneration_schedule(),
        )

        assert result["completed_generations"] == 2, result.get(
            "root_failure", {}
        ).get("message", result)
        assert result["controller_invocations"] == 1
        assert result["python_generation_dispatches"] == 0
        assert result["native_generation_transactions"] == 2
        assert result["hidden_fallback"] is False
        assert len(result["boundaries"]) == 2
        for generation, boundary in enumerate(result["boundaries"]):
            science = boundary["native_memory_science_summary"]
            assert science["generation"] == generation
            assert science["wdl_delayed_hazard_enable"] == 1
            assert science["memory_science"]["behavior_row_count"] > 0
            assert science["behavior"]["row_count"] > 0
            assert science["lineage"]["record_count"] > 0
            assert "recurrent_upkeep_charge_total" in science, sorted(science)
            assert "recurrent_upkeep_charged_agent_ticks" in science, sorted(science)
            if force_memory_disabled:
                assert science["recurrent_upkeep_charge_total"] == 0.0
                assert science["recurrent_upkeep_charged_agent_ticks"] == 0
            else:
                assert science["recurrent_upkeep_charge_total"] > 0.0
                assert science["recurrent_upkeep_charged_agent_ticks"] > 0
        final = result["boundaries"][-1]["checkpoint_projection"]
        assert final["metadata"]["generation"] == 2
        assert final["metadata"]["tick"] == 0
        alive = final["tensors"]["alive_mask"]
        if force_memory_disabled:
            assert torch.count_nonzero(final["tensors"]["hmem"][alive]) == 0
            assert torch.count_nonzero(final["tensors"]["policy_mem_dim"][alive]) == 0
            assert torch.count_nonzero(final["tensors"]["gru_upkeep_cost"][alive]) == 0
            assert torch.count_nonzero(final["tensors"]["genome_memory_dim"][alive]) > 0
        else:
            assert torch.count_nonzero(final["tensors"]["hmem"][alive]) > 0
        final_projections.append(final)

    comparison = compare_projection_digest(final_projections[0], final_projections[1])
    assert comparison["ok"], comparison


def test_native_multigeneration_controller_rejects_gru_policy_mutation(
    tmp_path: Path,
) -> None:
    runner, _agents = _live_gru_runner(
        tmp_path / "unsupported_gru_policy_mutation",
        patch=True,
        wdl=True,
    )
    runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
    runner.sim_cfg.policy_weight_mutation_enable = True
    projection = build_import_projection(
        runner,
        {
            "flat": False,
            "patch": True,
            "wdl": True,
            "gru": True,
            "churn": True,
        },
        generation=0,
        tick=0,
    )
    module = load_resident_state_owner_extension(
        build_directory=tmp_path / "owner_build"
    )

    with pytest.raises(
        RuntimeError,
        match="stable GRU multigeneration control is not qualified when policy mutation is enabled",
    ):
        module.FlatCpuArchitectureBControllerV1(
            projection.tensors,
            copy.deepcopy(projection.metadata),
            "unsupported-gru-policy-mutation",
            False,
        )


def test_resident_flat_cpu_owner_gru_turnover_preserves_survivors_and_resets_children(
    tmp_path: Path,
) -> None:
    runner, _agents = _live_gru_runner(tmp_path / "resident_flat_gru_turnover")
    runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
    runner.pop_cfg.cull_fraction = 0.4
    runner.pop_cfg.parent_fraction = 1.0
    runner.pop._eiel_current_generation_index = 0
    for agent in runner.pop.alive_agents():
        agent.stats.survival_steps = 2
        agent.stats.energy_collected = float(100 - int(agent.agent_id))
        agent.stats.early_life_weight = 1.0
        agent.stats.early_survival_steps = 1
        agent.stats.early_energy_collected = 0.0
        agent.stats.early_injury_gained = 0.0

    capability = {"flat": True, "patch": False, "gru": True, "churn": True}
    owner = ResidentRuntimeState.from_runner(
        runner,
        capability,
        generation=0,
        tick=0,
        build_directory=tmp_path / "owner_build",
    )
    before = owner.export_checkpoint_projection()
    before_hmem = before["tensors"]["hmem"].clone()
    before_ids = before["tensors"]["slot_to_agent_id"].clone()
    global_torch_rng_before = torch.get_rng_state().clone()

    runner.pop.reproduce_and_cull()
    runner._sync_env_with_population()
    runner._mark_bank_dirty()
    runner._sync_gru_bank_for_agents(runner.pop.alive_agents())
    reference_global_torch_rng_after = torch.get_rng_state().clone()
    reference = build_import_projection(
        runner,
        capability,
        generation=1,
        tick=0,
    )

    torch.set_rng_state(global_torch_rng_before)
    result = owner.execute_population_generation_turnover_cpu()
    native_global_torch_rng_after = torch.get_rng_state().clone()
    native = owner.export_checkpoint_projection()

    assert result["branch"] == "cull_refill"
    assert int(result["refill_rows"]) == 2
    assert int(result["committed_refill_rows"]) == 2
    assert torch.equal(native_global_torch_rng_after, reference_global_torch_rng_after)
    for name in (
        "slot_to_agent_id",
        "agent_id_to_slot",
        "slot_tokens",
        "slot_free_queue",
        "alive_mask",
        "pos_y",
        "pos_x",
        "agent_grid",
        "energy",
        "health",
        "policy_use_gru",
        "policy_mem_dim",
        "genome_memory_dim",
        "gru_pack",
        "hmem",
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
    ):
        torch.testing.assert_close(
            native["tensors"][name],
            reference.tensors[name],
            rtol=0.0,
            atol=0.0,
            msg=lambda message, field=name: f"{field}: {message}",
        )
    for name in (
        "native_runtime_numpy_env_state",
        "native_runtime_numpy_policy_state",
        "native_runtime_numpy_population_state",
        "native_runtime_numpy_runner_state",
        "native_runtime_python_random_state",
    ):
        assert native["metadata"][name] == reference.metadata[name], name

    surviving_slots = torch.nonzero(
        torch.logical_and(before_ids >= 0, native["tensors"]["slot_to_agent_id"] == before_ids),
        as_tuple=False,
    ).view(-1)
    child_slots = torch.tensor(
        [int(slot) for slot in result["refill_result"]["child_slots"]],
        dtype=torch.int64,
    )
    assert surviving_slots.numel() == 3
    assert child_slots.numel() == 2
    torch.testing.assert_close(
        native["tensors"]["hmem"].index_select(0, surviving_slots),
        before_hmem.index_select(0, surviving_slots),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.count_nonzero(
        native["tensors"]["hmem"].index_select(0, child_slots)
    ).item() == 0


def test_resident_flat_cpu_owner_gru_birth_after_death_clears_recurrent_slot(
    tmp_path: Path,
) -> None:
    runner, agents = _live_gru_runner(tmp_path / "resident_flat_gru_birth_death")
    runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
    runner.sim_cfg.steps_per_generation = 1
    runner.env_cfg.reproduce_try_prob = 1.0
    runner.env_cfg.reproduce_parent_energy_share = 0.5
    runner.env_cfg.reproduce_min_energy_buffer = 1.0
    runner.env_cfg.reproduce_cooldown_steps = 99
    runner.env_cfg.gru_cost_base = 0.0
    runner.env_cfg.gru_cost_per_dim = 0.0
    generation_start = runner._start_generation(0)
    for agent in agents:
        agent.energy = 10.0
        agent.health = 1.0
        agent.injury = 0.0
        agent.scar = 0.0
        agent.repro_cooldown = 0
        agent.genome.reproduction_threshold = 200.0
    parent = agents[0]
    parent.energy = 60.0
    parent.genome.reproduction_threshold = 20.0
    dying = agents[-1]
    dying.health = 0.0
    dying_slot = int(dying.slot)
    runner._Hmem[dying_slot].fill_(0.75)

    capability = {"flat": True, "patch": False, "gru": True, "churn": True}
    owner = ResidentRuntimeState.from_runner(
        runner,
        capability,
        generation=0,
        tick=0,
        build_directory=tmp_path / "owner_build",
    )
    next_child_id = int(runner.pop.next_id)
    global_torch_rng_before = torch.get_rng_state().clone()

    runner._run_ticks(
        gen=0,
        buckets=generation_start[0],
        injury_recover_mult=generation_start[2],
        deaths_this_gen=generation_start[3],
        successful_births=generation_start[4],
        failed_births=generation_start[10],
        gen_peak_injury=generation_start[9],
        rest_streaks=generation_start[11],
    )
    reference_global_torch_rng_after = torch.get_rng_state().clone()
    reference = build_import_projection(runner, capability, generation=0, tick=1)

    torch.set_rng_state(global_torch_rng_before)
    result = owner.execute_generation_flat_with_population_birth_commit_cpu(
        0,
        1,
        max_births=1,
    )
    native_global_torch_rng_after = torch.get_rng_state().clone()
    native = owner.export_checkpoint_projection()

    assert int(result["flat_generation_core"]["died"]) == 1
    assert int(result["committed_birth_rows"]) == 1
    assert [int(value) for value in result["child_agent_ids"]] == [next_child_id]
    assert torch.equal(native_global_torch_rng_after, reference_global_torch_rng_after)
    for name in (
        "slot_to_agent_id",
        "agent_id_to_slot",
        "slot_tokens",
        "slot_free_queue",
        "alive_mask",
        "pos_y",
        "pos_x",
        "agent_grid",
        "energy",
        "health",
        "policy_use_gru",
        "policy_mem_dim",
        "genome_memory_dim",
        "gru_pack",
        "action_cache_actions",
        "action_cache_token",
        "action_cache_cursor",
        "action_cache_next",
    ):
        torch.testing.assert_close(
            native["tensors"][name],
            reference.tensors[name],
            rtol=0.0,
            atol=0.0,
            msg=lambda message, field=name: f"{field}: {message}",
        )
    torch.testing.assert_close(
        native["tensors"]["hmem"],
        reference.tensors["hmem"],
        rtol=0.0,
        atol=2.0e-6,
    )
    for name in (
        "native_runtime_numpy_env_state",
        "native_runtime_numpy_policy_state",
        "native_runtime_numpy_population_state",
        "native_runtime_numpy_runner_state",
        "native_runtime_python_random_state",
    ):
        assert native["metadata"][name] == reference.metadata[name], name

    child_slot = int(native["tensors"]["agent_id_to_slot"][next_child_id].item())
    assert child_slot >= 0
    assert torch.count_nonzero(native["tensors"]["hmem"][child_slot]).item() == 0
    if child_slot != dying_slot:
        assert torch.count_nonzero(native["tensors"]["hmem"][dying_slot]).item() == 0


@pytest.mark.parametrize("patch", [False, True], ids=["flat", "patch"])
def test_resident_cpu_owner_gru_fresh_owner_cache_continuation_is_exact(
    tmp_path: Path,
    patch: bool,
) -> None:
    runner, _agents = _live_gru_runner(
        tmp_path / ("resident_patch_gru_continuation" if patch else "resident_flat_gru_continuation"),
        patch=patch,
    )
    runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
    capability = {
        "flat": not patch,
        "patch": patch,
        "gru": True,
        "churn": False,
    }
    source = ResidentRuntimeState.from_runner(
        runner,
        capability,
        generation=0,
        tick=0,
        build_directory=tmp_path / "owner_build",
    )
    initial = source.export_checkpoint_projection()
    uninterrupted = ResidentRuntimeState.from_checkpoint_projection(
        initial,
        build_directory=tmp_path / "owner_build",
    )
    split = ResidentRuntimeState.from_checkpoint_projection(
        initial,
        build_directory=tmp_path / "owner_build",
    )

    execute_uninterrupted = (
        uninterrupted.execute_generation_patch_cpu
        if patch
        else uninterrupted.execute_generation_flat_cpu
    )
    execute_split = (
        split.execute_generation_patch_cpu
        if patch
        else split.execute_generation_flat_cpu
    )
    execute_uninterrupted(0, 4)
    execute_split(0, 1)
    boundary = split.export_checkpoint_projection()
    alive = boundary["tensors"]["alive_mask"]
    assert bool(boundary["tensors"]["action_cache_cursor"][alive].eq(1).all().item())

    restored = ResidentRuntimeState.from_checkpoint_projection(
        boundary,
        build_directory=tmp_path / "owner_build",
    )
    restored_boundary = restored.export_checkpoint_projection()
    boundary_compare = compare_projection_digest(boundary, restored_boundary)
    assert boundary_compare["ok"], boundary_compare

    execute_restored = (
        restored.execute_generation_patch_cpu
        if patch
        else restored.execute_generation_flat_cpu
    )
    execute_restored(0, 3)
    final_compare = compare_projection_digest(
        uninterrupted.export_checkpoint_projection(),
        restored.export_checkpoint_projection(),
    )
    assert final_compare["ok"], final_compare


@pytest.mark.parametrize("patch", [False, True], ids=["flat", "patch"])
def test_resident_cpu_owner_gru_g32_lifecycle_matches_reference(
    tmp_path: Path,
    patch: bool,
) -> None:
    runner, _agents = _live_gru_runner(
        tmp_path / ("resident_patch_gru_g32" if patch else "resident_flat_gru_g32"),
        patch=patch,
    )
    runner.sim_cfg.native_runtime_fixed_policy_family = "gru"
    runner.sim_cfg.steps_per_generation = 3
    runner.sim_cfg.native_runtime_use_precomputed_drive_noise = False
    runner.pop_cfg.cull_fraction = 0.4
    runner.pop_cfg.parent_fraction = 1.0
    capability = {
        "flat": not patch,
        "patch": patch,
        "gru": True,
        "churn": True,
    }
    owner = ResidentRuntimeState.from_runner(
        runner,
        capability,
        generation=0,
        tick=0,
        build_directory=tmp_path / "owner_build",
    )
    for generation in range(32):
        runner.pop._eiel_current_generation_index = generation
        generation_start = runner._start_generation(generation)
        global_torch_rng_before = torch.get_rng_state().clone()
        runner._run_ticks(
            gen=generation,
            buckets=generation_start[0],
            injury_recover_mult=generation_start[2],
            deaths_this_gen=generation_start[3],
            successful_births=generation_start[4],
            failed_births=generation_start[10],
            gen_peak_injury=generation_start[9],
            rest_streaks=generation_start[11],
        )
        reference_before_turnover = build_import_projection(
            runner,
            capability,
            generation=generation,
            tick=3,
        )
        runner.pop.reproduce_and_cull()
        runner._sync_env_with_population()
        runner._mark_bank_dirty()
        runner._sync_gru_bank_for_agents(runner.pop.alive_agents())
        reference_global_torch_rng_after = torch.get_rng_state().clone()
        reference = build_import_projection(
            runner,
            capability,
            generation=generation + 1,
            tick=0,
        )

        torch.set_rng_state(global_torch_rng_before)
        if patch:
            owner.execute_generation_patch_with_population_birth_commit_cpu(
                generation,
                3,
                max_births=512,
            )
        else:
            owner.execute_generation_flat_with_population_birth_commit_cpu(
                generation,
                3,
                max_births=512,
            )
        native_before_turnover = owner.export_checkpoint_projection()
        assert (
            native_before_turnover["metadata"]["native_runtime_numpy_runner_state"]
            == reference_before_turnover.metadata["native_runtime_numpy_runner_state"]
        ), (
            native_before_turnover["metadata"]["native_runtime_numpy_runner_state"],
            reference_before_turnover.metadata["native_runtime_numpy_runner_state"],
        )
        pre_turnover_fields = (
            "slot_to_agent_id",
            "alive_mask",
            "last_action",
            "last_last_action",
            "action_cache_actions",
            "action_cache_cursor",
            "action_cache_next",
            "gru_upkeep_cost",
            "energy",
            "health",
            "survival_steps",
            "energy_collected",
            "early_survival_steps",
            "early_energy_collected",
            "early_injury_gained",
            "offspring_count",
            "attacks",
            "action_switches",
            "action_oscillations",
            "drives",
            "prev_drive_delta",
            "drive_volatility",
            "drive_jitter",
            "selfharm_events",
            "pred_err_sum",
            "pred_lr_exact",
            "pred_err_count",
            "episode0_pred_err_sum",
            "episode0_pred_err_count",
            "late_pred_err_sum",
            "late_pred_err_count",
            "prediction_update_count",
            "prediction_reset_count",
            "prediction_state_delta_abs_sum",
            "prediction_state_delta_abs_max",
            "adult_pred_err_ema",
            "exp_dE",
            "exp_dInj",
            "pred_lr_exact",
            "prediction_queue_due",
            "prediction_queue_action",
            "prediction_queue_exp_de",
            "prediction_queue_exp_dinj",
            "prediction_queue_energy",
            "prediction_queue_injury",
            "prediction_queue_count",
            "pred_err_sum_exact",
            "episode0_pred_err_sum_exact",
            "late_pred_err_sum_exact",
            "prediction_state_delta_abs_sum_exact",
            "prediction_state_delta_abs_max_exact",
            "adult_pred_err_ema_exact",
        ) + (
            (
                "patch_food_amount",
                "patch_scent",
                "patch_waste",
                "patch_movement_cost",
            )
            if patch
            else ()
        )
        for name in pre_turnover_fields:
            torch.testing.assert_close(
                native_before_turnover["tensors"][name],
                reference_before_turnover.tensors[name],
                rtol=0.0,
                atol=0.0,
                msg=lambda message, field=name, gen=generation: (
                    f"generation={gen} pre-turnover field={field}: {message}"
                ),
            )
        turnover = owner.execute_population_generation_turnover_cpu()
        native_global_torch_rng_after = torch.get_rng_state().clone()
        native = owner.export_checkpoint_projection()
        assert int(turnover["next_generation"]) == generation + 1
        assert torch.equal(
            native_global_torch_rng_after,
            reference_global_torch_rng_after,
        ), generation
        post_turnover_fields = (
            "slot_to_agent_id",
            "agent_id_to_slot",
            "slot_tokens",
            "slot_free_queue",
            "alive_mask",
            "pos_y",
            "pos_x",
            "agent_grid",
            "food_grid",
            "energy",
            "health",
            "policy_use_gru",
            "policy_mem_dim",
            "genome_memory_dim",
            "gru_pack",
            "action_cache_actions",
            "action_cache_token",
            "action_cache_cursor",
            "action_cache_next",
            "gru_upkeep_cost",
            "exp_dE",
            "exp_dInj",
            "prediction_queue_due",
            "prediction_queue_action",
            "prediction_queue_exp_de",
            "prediction_queue_exp_dinj",
            "prediction_queue_energy",
            "prediction_queue_injury",
            "prediction_queue_count",
            "pred_err_sum_exact",
            "episode0_pred_err_sum_exact",
            "late_pred_err_sum_exact",
            "prediction_state_delta_abs_sum_exact",
            "prediction_state_delta_abs_max_exact",
            "adult_pred_err_ema_exact",
        ) + (
            (
                "patch_food_amount",
                "patch_scent",
                "patch_waste",
                "patch_movement_cost",
            )
            if patch
            else ()
        )
        for name in post_turnover_fields:
            torch.testing.assert_close(
                native["tensors"][name],
                reference.tensors[name],
                rtol=0.0,
                atol=0.0,
                msg=lambda message, field=name, gen=generation: (
                    f"generation={gen} field={field}: {message}"
                ),
            )
        torch.testing.assert_close(
            native["tensors"]["hmem"],
            reference.tensors["hmem"],
            rtol=0.0,
            atol=2.0e-6,
            msg=f"generation={generation} hmem",
        )
        for name in (
            "prediction_tick",
            "native_runtime_numpy_env_state",
            "native_runtime_numpy_policy_state",
            "native_runtime_numpy_population_state",
            "native_runtime_numpy_runner_state",
            "native_runtime_python_random_state",
        ):
            assert native["metadata"][name] == reference.metadata[name], (
                generation,
                name,
            )


def test_real_runner_stride_cache_and_hmem_continue_exactly_in_fresh_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, oracle,
) -> None:
    captures: list[dict] = []

    def capture(_runner, _state, **kwargs):
        captures.append({"runner": _runner, "state": _state, **kwargs})
        return {"supported": False, "executed_native": False, "hidden_fallback": False}

    monkeypatch.setattr(
        "perf.cxxbackbone2_policy_core_live_probe.run_policy_core_live_probe", capture
    )
    source, source_agents = _live_gru_runner(tmp_path / "source")
    source_state = import_generation_state(
        source, generation=0, profile="supported_fast_runner_v2"
    )
    source_ids = np.asarray(
        [int(agent.agent_id) for agent in source_agents], dtype=np.int32
    )
    source_slots = np.asarray(
        [int(agent.slot) for agent in source_agents], dtype=np.int64
    )
    first = runnercore1_actions_strided(source, source_state, source_ids, buckets={})
    assert len(captures) == 1
    _assert_captured_live_refill_matches_native(oracle, source, captures[-1])
    hmem_at_checkpoint = source._Hmem.detach().cpu().numpy().copy()
    assert source_state.cache_cursor[source_slots].tolist() == [1] * 5
    export_to_agents(source_state, source, boundary_reason="gru_cpu_parity_checkpoint")
    bundle = tmp_path / "checkpoint_bundles_v2" / "bundle_gen000001"
    checkpoint_bundle_v2_save(
        source, str(bundle), generation_index=1, strict_replay_mode=True
    )

    uninterrupted_hit = runnercore1_actions_strided(
        source, source_state, source_ids, buckets={}
    )
    uninterrupted_refill = runnercore1_actions_strided(
        source, source_state, source_ids, buckets={}
    )
    uninterrupted_hmem = source._Hmem.detach().cpu().numpy().copy()
    assert len(captures) == 2

    restored, _ = _live_gru_runner(tmp_path / "restored")
    payload = checkpoint_bundle_v2_load(str(bundle), strict=True)
    checkpoint_bundle_v2_apply(restored, payload, restore_rng=True)
    restored_agents = sorted(
        restored.pop.alive_agents(), key=lambda item: int(item.agent_id)
    )
    restored_ids = np.asarray(
        [int(agent.agent_id) for agent in restored_agents], dtype=np.int32
    )
    restored_slots = np.asarray(
        [int(agent.slot) for agent in restored_agents], dtype=np.int64
    )
    restored_state = import_generation_state(
        restored, generation=1, profile="supported_fast_runner_v2"
    )
    np.testing.assert_array_equal(
        restored._Hmem.detach().cpu().numpy(), hmem_at_checkpoint
    )
    assert restored_state.cache_cursor[restored_slots].tolist() == [1] * 5
    assert restored_state.cache_token[restored_slots].tolist() == (
        source_state.cache_token[source_slots].tolist()
    )

    capture_count = len(captures)
    restored_hit = runnercore1_actions_strided(
        restored, restored_state, restored_ids, buckets={}
    )
    assert len(captures) == capture_count
    np.testing.assert_array_equal(restored_hit, uninterrupted_hit)
    np.testing.assert_array_equal(
        restored._Hmem.detach().cpu().numpy(), hmem_at_checkpoint
    )
    restored_refill = runnercore1_actions_strided(
        restored, restored_state, restored_ids, buckets={}
    )
    assert len(captures) == capture_count + 1
    _assert_captured_live_refill_matches_native(oracle, restored, captures[-1])
    np.testing.assert_array_equal(restored_refill, uninterrupted_refill)
    np.testing.assert_array_equal(
        restored._Hmem.detach().cpu().numpy(), uninterrupted_hmem
    )
    np.testing.assert_array_equal(first, restored_hit)


@pytest.mark.parametrize("compiler", ["g++", "clang++"])
def test_stable_gru_cpu_c_abi_is_sanitizer_clean(tmp_path: Path, compiler: str) -> None:
    executable = shutil.which(compiler)
    if executable is None:
        pytest.skip(f"{compiler} is unavailable")
    output = tmp_path / f"stable_gru_cpu_oracle_probe_{compiler.replace('+', 'x')}"
    command = [
        executable,
        "-std=c++17",
        "-O1",
        "-g",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-pedantic",
        "-fno-omit-frame-pointer",
        "-fsanitize=address,undefined",
        f"-I{NATIVE_ROOT}",
        str(NATIVE_ROOT / "stable_gru_v1.cpp"),
        str(NATIVE_ROOT / "stable_gru_cpu_oracle_c_api_v1.cpp"),
        str(SANITIZER_PROBE),
        "-o",
        str(output),
    ]
    compiled = subprocess.run(
        command, cwd=PROJECT_ROOT, check=False, capture_output=True, text=True
    )
    assert compiled.returncode == 0, compiled.stderr
    environment = dict(os.environ)
    environment["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"
    environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    executed = subprocess.run(
        [str(output)], cwd=PROJECT_ROOT, env=environment,
        check=False, capture_output=True, text=True,
    )
    assert executed.returncode == 0, executed.stderr
