from __future__ import annotations

"""RUNNERCORE-1 array-owned policy/action cache."""

import time
from typing import Any

import numpy as np
import torch

from agents.batched_gru import forward_batched_gru
from agents.batched_policy import forward_batched_mlp
from agents.chemistry import (
    CHEM_SLOT_ACUTE_DAMAGE,
    CHEM_SLOT_CHRONIC_DAMAGE,
    CHEM_SLOT_REGULATOR_1,
    CHEM_SLOT_REGULATOR_2,
    CHEM_SLOT_REPAIR_RESERVE,
)
from agents.chemistry_coupling import observation_fidelity_and_bias_from_chem
from agents.cognition import memory_ablation_flags
from agents.perception import perception_channel_registry, sensor_channel_mask_contains
from env.gridworld_pkg.patch_world import (
    habacclim_observation_enabled,
    habacclim_observation_scalar_index,
    patch_world_local_observation_size,
    patch_world_total_observation_size,
)
from simulation.runner_pkg.runner_ticks_common import ACT_ATTACK, ACT_GATHER, ACT_REST
from simulation.runnercore.tick_state import (
    RunnerCore1TickState,
    runnercore3_profile_add,
    runnercore3_profile_count,
    slots_for_agent_ids,
)


SCHEMA_VERSION = "runnercore1_action_cache_v1"


class RunnerCore1ActionError(RuntimeError):
    """Raised when the array action path cannot safely produce actions."""


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _torch_index(values: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.int64)).to(device=device, dtype=torch.long, non_blocking=True)


def _runnercore3_profile_enabled(runner: Any) -> bool:
    return _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore3_profile_enable", False))


def _runnercore3_vectorized_action_cache_enabled(runner: Any) -> bool:
    return _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore3_vectorized_action_cache_enable", False))


def _runnercore33_stride1_direct_action_enabled(runner: Any) -> bool:
    return _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore33_stride1_direct_action_enable", False))


def _cxxbackbone6_vertical_slice_enabled(runner: Any) -> bool:
    return _as_bool(
        getattr(getattr(runner, "sim_cfg", None), "cxxbackbone6_resident_policy_vertical_slice_enable", False)
    )


def _cxxbackbone8r_flat_cpu_policy_inputs_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    if not _as_bool(getattr(cfg, "cxxbackbone8r_flat_cpu_native_policy_inputs_enable", False)):
        return False
    if str(getattr(cfg, "device", "cpu") or "cpu").startswith("cuda"):
        raise RunnerCore1ActionError("CXXBACKBONE8R native policy inputs are CPU-only in this stage")
    if _as_bool(getattr(env_cfg, "patch_world_enable", False)) or _as_bool(getattr(env_cfg, "patch_world_observation_enable", False)):
        raise RunnerCore1ActionError("CXXBACKBONE8R native policy inputs are flat-only in this stage")
    if _as_bool(getattr(cfg, "chemistry_functional_coupling_enable", False)):
        raise RunnerCore1ActionError("CXXBACKBONE8R native policy inputs do not yet own chemistry observation coupling")
    if _policy_observation_masks_requested(runner):
        raise RunnerCore1ActionError("CXXBACKBONE8R native policy inputs do not yet own perception masks")
    return True


def _cxxbackbone8r_patch_cpu_policy_inputs_enabled(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    env_cfg = getattr(runner, "env_cfg", None)
    if not _as_bool(getattr(cfg, "cxxbackbone8r_patch_cpu_native_policy_inputs_enable", False)):
        return False
    if str(getattr(cfg, "device", "cpu") or "cpu").startswith("cuda"):
        raise RunnerCore1ActionError("CXXBACKBONE8R Patch native policy inputs are CPU-only in this stage")
    if not _as_bool(getattr(env_cfg, "patch_world_enable", False)):
        raise RunnerCore1ActionError("CXXBACKBONE8R Patch native policy inputs require patch_world_enable")
    if not _as_bool(getattr(env_cfg, "patch_world_observation_enable", False)):
        raise RunnerCore1ActionError("CXXBACKBONE8R Patch native policy inputs require patch_world_observation_enable")
    unsupported = {
        "wdl_delayed_hazard_enable": False,
        "wdl_delayed_hazard_observation_enable": False,
        "prediction_demand_enable": False,
        "prediction_demand_observation_enable": False,
        "delayed_resource_depletion_enable": False,
        "delayed_resource_depletion_observation_enable": False,
        "delayed_crowding_waste_enable": False,
        "delayed_crowding_waste_observation_enable": False,
        "mem_max_delayed_choice_enable": False,
        "mem_max_delayed_choice_observation_enable": False,
    }
    for name in unsupported:
        if _as_bool(getattr(env_cfg, name, False)):
            raise RunnerCore1ActionError(f"CXXBACKBONE8R Patch native policy inputs do not yet own {name}")
    if _as_bool(getattr(cfg, "chemistry_functional_coupling_enable", False)):
        raise RunnerCore1ActionError("CXXBACKBONE8R Patch native policy inputs do not yet own chemistry observation coupling")
    if _policy_observation_masks_requested(runner):
        raise RunnerCore1ActionError("CXXBACKBONE8R Patch native policy inputs do not yet own perception masks")
    return True


def _runnercore48_vectorized_chemistry_observation_enabled(runner: Any) -> bool:
    return _as_bool(
        getattr(getattr(runner, "sim_cfg", None), "runnercore48_vectorized_chemistry_observation_enable", False)
    )


def _runnercore20_policy_observation_enabled(runner: Any) -> bool:
    return _as_bool(getattr(getattr(runner, "sim_cfg", None), "runnercore20_policy_observation_arrays_enable", False))


def _cxxbackbone5_policy_profile_enabled(runner: Any) -> bool:
    return _as_bool(getattr(getattr(runner, "sim_cfg", None), "cxxbackbone5_policy_envelope_profile_enable", False))


def _cxxbackbone5_bucket_add(buckets: dict[str, float] | None, name: str, value: float = 1.0) -> None:
    if buckets is None:
        return
    key = "cxxbackbone5_" + str(name)
    buckets[key] = float(buckets.get(key, 0.0) + float(value))


def _policy_observation_masks_requested(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(
        _as_bool(getattr(cfg, "enable_evolvable_sensor_channel_mask", False))
        or _as_bool(getattr(cfg, "ablate_sensor_channel_mask", False))
        or _as_bool(getattr(cfg, "force_all_sensor_channels_enabled", False))
        or _as_bool(getattr(cfg, "force_minimal_sensor_channels_enabled", False))
        or _as_bool(getattr(cfg, "enable_evolvable_sensor_radius", False))
        or _as_bool(getattr(cfg, "ablate_sensor_radius", False))
        or _as_bool(getattr(cfg, "force_min_sensor_radius", False))
        or _as_bool(getattr(cfg, "force_max_sensor_radius", False))
        or _as_bool(getattr(cfg, "force_default_sensor_radius", False))
    )


def _apply_resident_chemistry_observation(
    runner: Any,
    state: RunnerCore1TickState,
    slots: np.ndarray,
    local: np.ndarray,
    energy_norm: np.ndarray,
    health_norm: np.ndarray,
) -> None:
    if not _as_bool(getattr(getattr(runner, "sim_cfg", None), "chemistry_functional_coupling_enable", False)):
        return
    cfg = runner.sim_cfg
    fidelity_min = float(getattr(cfg, "chemistry_observation_fidelity_min", 0.70))
    fidelity_max = float(getattr(cfg, "chemistry_observation_fidelity_max", 1.00))
    noise_abs_max = float(getattr(cfg, "chemistry_observation_noise_abs_max", 0.15))
    fidelity_strength = float(getattr(cfg, "chemistry_observation_fidelity_strength", 0.9))
    noise_strength = float(getattr(cfg, "chemistry_observation_noise_strength", 0.8))
    if _runnercore48_vectorized_chemistry_observation_enabled(runner):
        chem = np.asarray(state.chemistry)
        required_cols = max(
            CHEM_SLOT_REPAIR_RESERVE,
            CHEM_SLOT_ACUTE_DAMAGE,
            CHEM_SLOT_CHRONIC_DAMAGE,
            CHEM_SLOT_REGULATOR_1,
            CHEM_SLOT_REGULATOR_2,
        ) + 1
        if int(chem.ndim) != 2 or int(chem.shape[1]) < int(required_cols):
            raise RunnerCore1ActionError("runnercore48 requires resident chemistry rows with chemistry slots")
        slot_rows = np.asarray(slots, dtype=np.int64)
        c = np.asarray(chem[slot_rows], dtype=np.float32).astype(np.float64, copy=False)
        c = np.clip(c, 0.0, 1.0)
        damage = (
            0.60 * c[:, CHEM_SLOT_ACUTE_DAMAGE]
            + 0.40 * c[:, CHEM_SLOT_CHRONIC_DAMAGE]
            - 0.25 * c[:, CHEM_SLOT_REPAIR_RESERVE]
            - 0.15 * c[:, CHEM_SLOT_REGULATOR_1]
        )
        damage = np.clip(damage, 0.0, 1.0)

        fidelity_scale = min(1.0, max(0.0, float(fidelity_strength)))
        fidelity_span = max(0.0, float(fidelity_max) - float(fidelity_min))
        fidelity = float(fidelity_min) + fidelity_span * (1.0 - fidelity_scale * damage)
        fidelity = np.where(
            fidelity < float(fidelity_min),
            float(fidelity_min),
            np.where(fidelity > float(fidelity_max), float(fidelity_max), fidelity),
        )

        signed = (c[:, CHEM_SLOT_REGULATOR_2] - 0.5) * 2.0
        noise_scale = min(1.0, max(0.0, float(noise_strength)))
        bias = float(noise_abs_max) * noise_scale * damage * signed
        bias_limit = abs(float(noise_abs_max))
        bias = np.where(bias < -bias_limit, -bias_limit, np.where(bias > bias_limit, bias_limit, bias))

        fidelity32 = fidelity.astype(np.float32, copy=False)
        bias32 = bias.astype(np.float32, copy=False)
        local[:, :] = np.clip(
            local * fidelity32.reshape((-1, 1)) + bias32.reshape((-1, 1)),
            np.float32(-1.0),
            np.float32(1.0),
        )

        energy_updated = energy_norm.astype(np.float64, copy=False) * fidelity + bias
        energy_updated = np.where(energy_updated > 0.0, energy_updated, 0.0)
        energy_updated = np.where(energy_updated < 1.0, energy_updated, 1.0)
        health_updated = health_norm.astype(np.float64, copy=False) * fidelity + bias
        health_updated = np.where(health_updated > 0.0, health_updated, 0.0)
        health_updated = np.where(health_updated < 1.0, health_updated, 1.0)
        energy_norm[:] = energy_updated.astype(np.float32, copy=False)
        health_norm[:] = health_updated.astype(np.float32, copy=False)
        state.chemistry_observation_fidelity[slot_rows] = fidelity32
        state.chemistry_observation_bias[slot_rows] = bias32
        if _runnercore3_profile_enabled(runner):
            runnercore3_profile_count(state, "runnercore48_chemistry_observation_vectorized_rows", int(slot_rows.shape[0]))
        return

    for i, slot_raw in enumerate(slots.tolist()):
        slot = int(slot_raw)
        fidelity, bias = observation_fidelity_and_bias_from_chem(
            state.chemistry[slot],
            fidelity_min=fidelity_min,
            fidelity_max=fidelity_max,
            noise_abs_max=noise_abs_max,
            fidelity_strength=fidelity_strength,
            noise_strength=noise_strength,
        )
        local[i, :] = np.clip(local[i, :] * np.float32(fidelity) + np.float32(bias), -1.0, 1.0)
        energy_norm[i] = np.float32(min(1.0, max(0.0, float(energy_norm[i]) * float(fidelity) + float(bias))))
        health_norm[i] = np.float32(min(1.0, max(0.0, float(health_norm[i]) * float(fidelity) + float(bias))))
        state.chemistry_observation_fidelity[slot] = np.float32(float(fidelity))
        state.chemistry_observation_bias[slot] = np.float32(float(bias))


def _apply_resident_perception_masks(
    runner: Any,
    state: RunnerCore1TickState,
    slots: np.ndarray,
    local: np.ndarray,
    energy_norm: np.ndarray,
    health_norm: np.ndarray,
) -> None:
    if not _policy_observation_masks_requested(runner):
        return
    if local.size <= 0:
        return
    max_radius = int(max(1, getattr(runner.env_cfg, "view_radius", 2)))
    k = int(2 * max_radius + 1)
    cells = int(k * k)
    if cells <= 0 or int(local.shape[1]) % cells != 0:
        return
    channel_count = int(local.shape[1]) // int(cells)
    cell_visible_cache: dict[int, np.ndarray] = {}
    channels = perception_channel_registry()

    for i, slot_raw in enumerate(slots.tolist()):
        slot = int(slot_raw)
        mask = int(state.sensor_channel_mask[slot])
        row = local[i, :]
        for channel in channels:
            if not bool(channel.active_now) or bool(channel.future_reserved):
                continue
            if channel.local_channel_index is not None:
                ci = int(channel.local_channel_index)
                if ci >= channel_count:
                    continue
                if not sensor_channel_mask_contains(mask, str(channel.channel_id)):
                    start = int(ci * cells)
                    row[start : start + cells] = np.float32(float(channel.neutral_masked_value))
            elif channel.self_state_index == 0:
                if not sensor_channel_mask_contains(mask, str(channel.channel_id)):
                    energy_norm[i] = np.float32(float(channel.neutral_masked_value))
            elif channel.self_state_index == 1:
                if not sensor_channel_mask_contains(mask, str(channel.channel_id)):
                    health_norm[i] = np.float32(float(channel.neutral_masked_value))

        active_radius = int(state.sensor_radius[slot])
        if active_radius < max_radius:
            cell_visible = cell_visible_cache.get(active_radius)
            if cell_visible is None:
                visible = np.zeros((k, k), dtype=bool)
                center = int(max_radius)
                for yy in range(k):
                    for xx in range(k):
                        if max(abs(int(yy) - center), abs(int(xx) - center)) <= active_radius:
                            visible[yy, xx] = True
                cell_visible = visible.reshape(cells)
                cell_visible_cache[active_radius] = cell_visible
            for ci in range(channel_count):
                start = int(ci * cells)
                block = row[start : start + cells]
                block[~cell_visible] = np.float32(0.0)


def _compute_actions_k(
    runner: Any,
    state: RunnerCore1TickState,
    agent_ids: np.ndarray,
    *,
    k: int,
    buckets: dict[str, float] | None = None,
    precomputed_slots: np.ndarray | None = None,
    precomputed_local_observations: np.ndarray | None = None,
) -> np.ndarray:
    k = max(1, int(k))
    n = int(agent_ids.shape[0])
    if n <= 0:
        return np.empty((0, k), dtype=np.int32)
    if not _as_bool(getattr(runner.sim_cfg, "policy_deterministic_actions", False)):
        raise RunnerCore1ActionError("runnercore1 requires deterministic policy actions")

    profile_enabled = _runnercore3_profile_enabled(runner)
    cxxbackbone5_profile = _cxxbackbone5_policy_profile_enabled(runner)
    if cxxbackbone5_profile:
        _cxxbackbone5_bucket_add(buckets, "policy_python_entry_calls")
        _cxxbackbone5_bucket_add(buckets, "policy_requested_rows", float(n))
        _cxxbackbone5_bucket_add(buckets, "policy_requested_k", float(k))
    native_flat_policy_inputs = bool(
        precomputed_slots is None
        and precomputed_local_observations is None
        and _cxxbackbone8r_flat_cpu_policy_inputs_enabled(runner)
    )
    native_patch_policy_inputs = bool(
        precomputed_slots is None
        and precomputed_local_observations is None
        and not native_flat_policy_inputs
        and _cxxbackbone8r_patch_cpu_policy_inputs_enabled(runner)
    )
    native_policy_inputs = bool(native_flat_policy_inputs or native_patch_policy_inputs)
    native_obs_np = None
    native_groups = None
    if native_policy_inputs:
        from perf.cxxbackbone8r_generation_island import (  # noqa: WPS433
            flat_observation_frame_cpu,
            group_rows_cpu,
            patch_observation_frame_cpu,
        )
        from simulation.runnercore.tick_state import (  # noqa: WPS433
            legacy_dense_agent_id_to_slot,
        )

        build_dir = str(getattr(runner.sim_cfg, "cxxbackbone3_policy_aten_build_dir", "") or "")
        try:
            legacy_id_to_slot = legacy_dense_agent_id_to_slot(
                state,
                max_width=max(65_536, int(state.capacity) * 64),
            )
        except ValueError as exc:
            raise RunnerCore1ActionError(
                "legacy native policy grouping does not admit sparse high agent IDs"
            ) from exc
        native_groups = group_rows_cpu(
            agent_ids=np.asarray(agent_ids, dtype=np.int32),
            agent_id_to_slot=legacy_id_to_slot,
            use_gru=np.asarray(state.use_gru, dtype=np.bool_),
            build_directory=build_dir or None,
        )
        slots = np.asarray(native_groups["slots"], dtype=np.int32)
        if native_patch_policy_inputs:
            native_obs_np = patch_observation_frame_cpu(
                agent_ids=np.asarray(agent_ids, dtype=np.int32),
                slots=slots,
                agent_pos_y=np.asarray(getattr(runner.env, "agent_pos_y"), dtype=np.int32),
                agent_pos_x=np.asarray(getattr(runner.env, "agent_pos_x"), dtype=np.int32),
                agent_grid=np.asarray(getattr(runner.env, "agent_grid"), dtype=np.int32),
                food_grid=np.asarray(getattr(runner.env, "grid"), dtype=np.int32),
                energy_by_slot=np.asarray(state.energy, dtype=np.float32),
                health_by_slot=np.asarray(state.health, dtype=np.float32),
                patch_food_amount=np.asarray(getattr(runner.env, "patch_food_amount"), dtype=np.float32),
                patch_cover=np.asarray(getattr(runner.env, "patch_cover"), dtype=np.float32),
                patch_shelter=np.asarray(getattr(runner.env, "patch_shelter"), dtype=np.float32),
                patch_scent=np.asarray(getattr(runner.env, "patch_scent"), dtype=np.float32),
                patch_waste=np.asarray(getattr(runner.env, "patch_waste"), dtype=np.float32),
                patch_movement_cost=np.asarray(getattr(runner.env, "patch_movement_cost"), dtype=np.float32),
                view_radius=int(getattr(runner.env_cfg, "view_radius", 1)),
                max_energy=float(getattr(runner.env_cfg, "max_energy", 1.0) or 1.0),
                start_health=float(getattr(runner.env_cfg, "start_health", 1.0) or 1.0),
                patch_food_amount_max=float(getattr(runner.env_cfg, "patch_food_amount_max", 1.0) or 1.0),
                patch_scent_max=float(getattr(runner.env_cfg, "patch_scent_max", 1.0) or 1.0),
                patch_waste_max=float(getattr(runner.env_cfg, "patch_waste_max", 1.0) or 1.0),
                patch_density_radius=int(getattr(runner.env_cfg, "patch_density_radius", 1)),
                patch_density_include_self=_as_bool(getattr(runner.env_cfg, "patch_density_include_self", True)),
                patch_density_normalize=_as_bool(getattr(runner.env_cfg, "patch_density_normalize", False)),
                patch_movement_cost_min=float(getattr(runner.env_cfg, "patch_movement_cost_min", 0.0)),
                patch_movement_cost_max=float(getattr(runner.env_cfg, "patch_movement_cost_max", 4.0)),
                build_directory=build_dir or None,
            )
        else:
            native_obs_np = flat_observation_frame_cpu(
                agent_ids=np.asarray(agent_ids, dtype=np.int32),
                slots=slots,
                agent_pos_y=np.asarray(getattr(runner.env, "agent_pos_y"), dtype=np.int32),
                agent_pos_x=np.asarray(getattr(runner.env, "agent_pos_x"), dtype=np.int32),
                agent_grid=np.asarray(getattr(runner.env, "agent_grid"), dtype=np.int32),
                food_grid=np.asarray(getattr(runner.env, "grid"), dtype=np.int32),
                energy_by_slot=np.asarray(state.energy, dtype=np.float32),
                health_by_slot=np.asarray(state.health, dtype=np.float32),
                view_radius=int(getattr(runner.env_cfg, "view_radius", 1)),
                max_energy=float(getattr(runner.env_cfg, "max_energy", 1.0) or 1.0),
                start_health=float(getattr(runner.env_cfg, "start_health", 1.0) or 1.0),
                build_directory=build_dir or None,
            )
        if cxxbackbone5_profile:
            _cxxbackbone5_bucket_add(buckets, "cxxbackbone8r_native_group_row_map_calls")
            _cxxbackbone5_bucket_add(buckets, "cxxbackbone8r_native_observation_frame_calls")
            _cxxbackbone5_bucket_add(buckets, "cxxbackbone8r_native_policy_input_rows", float(n))
            if native_patch_policy_inputs:
                _cxxbackbone5_bucket_add(buckets, "cxxbackbone8r_native_patch_observation_frame_calls")
    elif precomputed_slots is None:
        slots = np.asarray([state.slot_for_agent_id(int(aid)) for aid in agent_ids.tolist()], dtype=np.int32)
    else:
        slots = np.asarray(precomputed_slots, dtype=np.int32)
        if int(slots.shape[0]) != n:
            raise RunnerCore1ActionError("precomputed slot row count does not match agent ids")
    if cxxbackbone5_profile:
        if precomputed_slots is None and native_groups is None:
            _cxxbackbone5_bucket_add(buckets, "slot_lookup_python_calls", float(n))
        _cxxbackbone5_bucket_add(buckets, "slot_row_allocations")
    if bool(np.any(slots < 0)):
        raise RunnerCore1ActionError("runnercore1 action request contains unknown agent id")

    local_size = patch_world_local_observation_size(runner.env_cfg)
    obs_size = patch_world_total_observation_size(runner.env_cfg)
    t_obs = time.perf_counter() if profile_enabled else 0.0
    if native_obs_np is not None:
        local = np.asarray(native_obs_np[:, :local_size], dtype=np.float32)
    elif precomputed_local_observations is None:
        local = np.asarray(runner.env.get_local_observations(np.asarray(agent_ids, dtype=np.int32)), dtype=np.float32)
    else:
        local = np.asarray(precomputed_local_observations, dtype=np.float32)
        if tuple(local.shape) != (n, local_size):
            raise RunnerCore1ActionError("precomputed local observation shape does not match runner config")
    if cxxbackbone5_profile:
        if precomputed_local_observations is None and native_obs_np is None:
            _cxxbackbone5_bucket_add(buckets, "observation_python_callbacks")
        _cxxbackbone5_bucket_add(buckets, "observation_rows", float(n))
        _cxxbackbone5_bucket_add(buckets, "observation_local_bytes", float(local.nbytes))
    if profile_enabled:
        runnercore3_profile_add(state, "observation_buffer_update_ms", time.perf_counter() - t_obs)

    t_frame = time.perf_counter() if profile_enabled else 0.0
    if native_obs_np is not None:
        obs_np = np.asarray(native_obs_np, dtype=np.float32)
    else:
        obs_np = np.empty((n, obs_size), dtype=np.float32)
    if cxxbackbone5_profile:
        _cxxbackbone5_bucket_add(buckets, "numpy_allocations")
        _cxxbackbone5_bucket_add(buckets, "observation_frame_bytes", float(obs_np.nbytes))
    if native_obs_np is not None:
        energy_norm = obs_np[:, local_size]
        health_norm = obs_np[:, local_size + 1]
    elif _runnercore20_policy_observation_enabled(runner):
        local = np.asarray(local, dtype=np.float32).copy()
        energy_norm = (np.asarray(state.energy[slots], dtype=np.float32) / np.float32(float(runner.env_cfg.max_energy) or 1.0)).astype(
            np.float32,
            copy=False,
        )
        health_norm = (
            np.asarray(state.health[slots], dtype=np.float32) / np.float32(float(runner.env_cfg.start_health) or 1.0)
        ).astype(np.float32, copy=False)
        _apply_resident_chemistry_observation(runner, state, slots, local, energy_norm, health_norm)
        _apply_resident_perception_masks(runner, state, slots, local, energy_norm, health_norm)
    else:
        energy_norm = np.asarray(state.energy[slots], dtype=np.float32) / np.float32(float(runner.env_cfg.max_energy) or 1.0)
        health_norm = np.asarray(state.health[slots], dtype=np.float32) / np.float32(float(runner.env_cfg.start_health) or 1.0)
    if habacclim_observation_enabled(runner.env_cfg):
        _ = habacclim_observation_scalar_index(runner.env_cfg)
        raise RunnerCore1ActionError("runnercore1 action cache does not support habacclim observation scalar")
    if native_obs_np is None:
        obs_np[:, :local_size] = local
        obs_np[:, local_size] = energy_norm
        obs_np[:, local_size + 1] = health_norm

    device = runner.device
    x_all = torch.from_numpy(obs_np).to(device=device, dtype=torch.float32, non_blocking=True)
    aggr_all = torch.from_numpy(np.asarray(state.aggr[slots], dtype=np.float32)).to(device=device, non_blocking=True)
    temp_all = torch.from_numpy(np.asarray(state.temp[slots], dtype=np.float32)).to(device=device, non_blocking=True)
    drives_all = torch.from_numpy(np.asarray(state.drives[slots], dtype=np.float32)).to(device=device, non_blocking=True)
    gain_all = torch.from_numpy(np.asarray(state.gain[slots], dtype=np.float32)).to(device=device, non_blocking=True)
    if cxxbackbone5_profile:
        _cxxbackbone5_bucket_add(buckets, "torch_tensor_constructions", 5.0)
        if str(device).startswith("cuda"):
            _cxxbackbone5_bucket_add(
                buckets,
                "h2d_bytes",
                float(
                    obs_np.nbytes
                    + np.asarray(state.aggr[slots], dtype=np.float32).nbytes
                    + np.asarray(state.temp[slots], dtype=np.float32).nbytes
                    + np.asarray(state.drives[slots], dtype=np.float32).nbytes
                    + np.asarray(state.gain[slots], dtype=np.float32).nbytes
                ),
            )

    if native_groups is not None:
        mlp_idx = np.asarray(native_groups["mlp_idx"], dtype=np.int64)
        gru_idx = np.asarray(native_groups["gru_idx"], dtype=np.int64)
    else:
        mlp_idx = np.nonzero(~state.use_gru[slots])[0].astype(np.int64)
        gru_idx = np.nonzero(state.use_gru[slots])[0].astype(np.int64)
    actions_all = torch.empty((n, k), dtype=torch.int64, device=device)
    if cxxbackbone5_profile:
        if native_groups is None:
            _cxxbackbone5_bucket_add(buckets, "policy_grouping_python_calls")
            _cxxbackbone5_bucket_add(buckets, "row_map_allocations", 2.0)
        _cxxbackbone5_bucket_add(buckets, "mlp_group_rows", float(mlp_idx.shape[0]))
        _cxxbackbone5_bucket_add(buckets, "gru_group_rows", float(gru_idx.shape[0]))
        _cxxbackbone5_bucket_add(buckets, "torch_output_allocations")
    if profile_enabled:
        runnercore3_profile_add(state, "policy_input_frame_ms", time.perf_counter() - t_frame)

    w_hunger_gather = float(getattr(runner.sim_cfg, "drive_w_hunger_gather", 1.2))
    w_pain_rest = float(getattr(runner.sim_cfg, "drive_w_pain_rest", 0.9))
    w_fatigue_rest = float(getattr(runner.sim_cfg, "drive_w_fatigue_rest", 1.0))
    w_pain_attack = float(getattr(runner.sim_cfg, "drive_w_pain_attack", 1.2))
    w_fear_attack = float(getattr(runner.sim_cfg, "drive_w_fear_attack", 1.4))
    cxxbackbone2_policy_probe_before_hmem = None
    if bool(getattr(runner.sim_cfg, "cxxbackbone2_policy_core_live_probe_enable", False)):
        hmem_probe_source = getattr(runner, "_Hmem", None)
        if hmem_probe_source is not None:
            if hasattr(hmem_probe_source, "detach"):
                cxxbackbone2_policy_probe_before_hmem = (
                    hmem_probe_source.detach().cpu().numpy().astype(np.float32, copy=True)
                )
            else:
                cxxbackbone2_policy_probe_before_hmem = np.asarray(hmem_probe_source, dtype=np.float32).copy()

    t_forward = time.perf_counter() if profile_enabled else 0.0
    with torch.no_grad():
        if int(mlp_idx.shape[0]) > 0:
            rows = _torch_index(mlp_idx, device=device)
            bank_slots_np = np.asarray(slots[mlp_idx], dtype=np.int64)
            bank_slots = _torch_index(bank_slots_np, device=getattr(runner, "_bank_device", device))
            pack = runner._mlp_pack.index_select(0, bank_slots)
            pack = runner._eiel_to_compute(pack, name="runnercore1_mlp_pack_rows", stage="runnercore1.bank_select.mlp")
            h = int(runner._hidden)
            o = int(runner._obs_size)
            a = int(runner._out)
            off = 0
            w1 = pack[:, off : off + (h * o)].view(-1, h, o)
            off += h * o
            b1 = pack[:, off : off + h].view(-1, h)
            off += h
            w2 = pack[:, off : off + (h * h)].view(-1, h, h)
            off += h * h
            b2 = pack[:, off : off + h].view(-1, h)
            off += h
            w3 = pack[:, off : off + (a * h)].view(-1, a, h)
            off += a * h
            b3 = pack[:, off : off + a].view(-1, a)

            x_rows = x_all.index_select(0, rows)
            aggr_rows = aggr_all.index_select(0, rows)
            temp_rows = temp_all.index_select(0, rows)
            d = drives_all.index_select(0, rows)
            gain = gain_all.index_select(0, rows).clamp(min=0.0)
            if bool(getattr(runner.sim_cfg, "cxxbackbone4_policy_aten_mlp_authority_enable", False)):
                from perf.cxxbackbone3_policy_aten import run_mlp_policy_aten_torch  # noqa: WPS433

                build_dir = str(getattr(runner.sim_cfg, "cxxbackbone3_policy_aten_build_dir", "") or "")
                if cxxbackbone5_profile:
                    _cxxbackbone5_bucket_add(buckets, "mlp_aten_authority_calls")
                aten_result = run_mlp_policy_aten_torch(
                    x=x_rows,
                    pack=pack,
                    aggr=aggr_rows,
                    temp=temp_rows,
                    drives=d,
                    gain=gain,
                    weights={
                        "hunger_gather": w_hunger_gather,
                        "pain_rest": w_pain_rest,
                        "fatigue_rest": w_fatigue_rest,
                        "pain_attack": w_pain_attack,
                        "fear_attack": w_fear_attack,
                    },
                    obs=int(runner._obs_size),
                    hidden=int(runner._hidden),
                    actions=int(runner._out),
                    build_directory=build_dir or None,
                )
                logits = aten_result["logits_after_drives"]
                if buckets is not None:
                    buckets["cxxbackbone4_policy_aten_mlp_rows"] = float(
                        buckets.get("cxxbackbone4_policy_aten_mlp_rows", 0.0) + float(mlp_idx.shape[0])
                    )
            else:
                logits = forward_batched_mlp(
                    x_rows,
                    w1,
                    b1,
                    w2,
                    b2,
                    w3,
                    b3,
                    attack_bias=aggr_rows,
                    action_temperature=temp_rows,
                )
                if logits.numel() != 0:
                    logits[:, ACT_GATHER] = logits[:, ACT_GATHER] + gain * (w_hunger_gather * d[:, 0])
                    logits[:, ACT_REST] = logits[:, ACT_REST] + gain * (w_pain_rest * d[:, 1] + w_fatigue_rest * d[:, 2])
                    logits[:, ACT_ATTACK] = logits[:, ACT_ATTACK] - gain * (w_pain_attack * d[:, 1] + w_fear_attack * d[:, 3])
            action = torch.argmax(logits, dim=1)
            actions_all.index_copy_(0, rows, action.unsqueeze(1).expand(-1, k))
            if cxxbackbone5_profile:
                _cxxbackbone5_bucket_add(buckets, "action_selection_aten_calls")
                _cxxbackbone5_bucket_add(buckets, "output_scatter_aten_calls")
            if buckets is not None:
                buckets["runnercore1_policy_forward_mlp_rows"] = float(
                    buckets.get("runnercore1_policy_forward_mlp_rows", 0.0) + float(mlp_idx.shape[0])
                )

        if int(gru_idx.shape[0]) > 0:
            rows = _torch_index(gru_idx, device=device)
            bank_slots_np = np.asarray(slots[gru_idx], dtype=np.int64)
            bank_slots = _torch_index(bank_slots_np, device=getattr(runner, "_bank_device", device))
            mem_dims = torch.from_numpy(np.asarray(state.mem_dim[slots[gru_idx]], dtype=np.int64)).to(
                device=device, non_blocking=True
            )
            h_prev = runner._Hmem.index_select(0, bank_slots)
            h_prev = runner._eiel_to_compute(h_prev, name="runnercore1_gru_h_prev", stage="runnercore1.bank_select.hmem")
            hm = int(h_prev.shape[1])
            mem_dims = torch.clamp(mem_dims, 0, hm)
            idx = torch.arange(hm, device=device, dtype=torch.long).unsqueeze(0)
            mask = (idx < mem_dims.unsqueeze(1)).to(h_prev.dtype)
            h_prev = h_prev * mask
            reset_count = 0
            if bool(memory_ablation_flags(runner.sim_cfg).get("reset_each_tick", False)):
                reset_count = int(h_prev.shape[0])
                h_prev = torch.zeros_like(h_prev)

            pack = runner._gru_pack.index_select(0, bank_slots)
            pack = runner._eiel_to_compute(pack, name="runnercore1_gru_pack_rows", stage="runnercore1.bank_select.gru")
            o = int(runner._obs_size)
            a = int(runner._out)
            hm = int(runner._gru_hidden)
            off = 0
            enc_w = pack[:, off : off + (hm * o)].view(-1, hm, o)
            off += hm * o
            enc_b = pack[:, off : off + hm].view(-1, hm)
            off += hm
            wih = pack[:, off : off + (3 * hm * hm)].view(-1, 3 * hm, hm)
            off += 3 * hm * hm
            bih = pack[:, off : off + (3 * hm)].view(-1, 3 * hm)
            off += 3 * hm
            whh = pack[:, off : off + (3 * hm * hm)].view(-1, 3 * hm, hm)
            off += 3 * hm * hm
            bhh = pack[:, off : off + (3 * hm)].view(-1, 3 * hm)
            off += 3 * hm
            head_w = pack[:, off : off + (a * hm)].view(-1, a, hm)
            off += a * hm
            head_b = pack[:, off : off + a].view(-1, a)

            x_rows = x_all.index_select(0, rows)
            aggr_rows = aggr_all.index_select(0, rows)
            temp_rows = temp_all.index_select(0, rows)
            d = drives_all.index_select(0, rows)
            gain = gain_all.index_select(0, rows).clamp(min=0.0)
            if bool(getattr(runner.sim_cfg, "cxxbackbone3_policy_aten_gru_authority_enable", False)):
                from perf.cxxbackbone3_policy_aten import run_gru_policy_aten_torch  # noqa: WPS433

                build_dir = str(getattr(runner.sim_cfg, "cxxbackbone3_policy_aten_build_dir", "") or "")
                if cxxbackbone5_profile:
                    _cxxbackbone5_bucket_add(buckets, "gru_aten_authority_calls")
                aten_result = run_gru_policy_aten_torch(
                    x=x_rows,
                    h_prev=h_prev,
                    pack=pack,
                    mem_dims=mem_dims,
                    aggr=aggr_rows,
                    temp=temp_rows,
                    drives=d,
                    gain=gain,
                    weights={
                        "hunger_gather": w_hunger_gather,
                        "pain_rest": w_pain_rest,
                        "fatigue_rest": w_fatigue_rest,
                        "pain_attack": w_pain_attack,
                        "fear_attack": w_fear_attack,
                    },
                    obs=int(runner._obs_size),
                    hidden=int(runner._gru_hidden),
                    actions=int(runner._out),
                    build_directory=build_dir or None,
                )
                logits = aten_result["logits_after_drives"]
                h_next = aten_result["hmem_after"]
                if buckets is not None:
                    buckets["cxxbackbone3_policy_aten_gru_rows"] = float(
                        buckets.get("cxxbackbone3_policy_aten_gru_rows", 0.0) + float(gru_idx.shape[0])
                    )
            else:
                logits, h_next = forward_batched_gru(
                    x=x_rows,
                    h_prev=h_prev,
                    encW=enc_w,
                    encb=enc_b,
                    Wih=wih,
                    bih=bih,
                    Whh=whh,
                    bhh=bhh,
                    headW=head_w,
                    headb=head_b,
                    attack_bias=aggr_rows,
                    action_temperature=temp_rows,
                )
                if logits.numel() != 0:
                    logits[:, ACT_GATHER] = logits[:, ACT_GATHER] + gain * (w_hunger_gather * d[:, 0])
                    logits[:, ACT_REST] = logits[:, ACT_REST] + gain * (w_pain_rest * d[:, 1] + w_fatigue_rest * d[:, 2])
                    logits[:, ACT_ATTACK] = logits[:, ACT_ATTACK] - gain * (w_pain_attack * d[:, 1] + w_fear_attack * d[:, 3])
            h_next = h_next * mask
            runner._record_memory_runtime_telemetry(h_prev=h_prev, h_next=h_next, mask=mask, reset_count=reset_count)
            h_next_bank = runner._eiel_to_bank(h_next, like=runner._Hmem, name="runnercore1_gru_h_next", stage="runnercore1.bank_update.hmem")
            runner._Hmem.index_copy_(0, bank_slots, h_next_bank)
            action = torch.argmax(logits, dim=1)
            actions_all.index_copy_(0, rows, action.unsqueeze(1).expand(-1, k))
            if cxxbackbone5_profile:
                _cxxbackbone5_bucket_add(buckets, "hmem_bank_updates")
                _cxxbackbone5_bucket_add(buckets, "action_selection_aten_calls")
                _cxxbackbone5_bucket_add(buckets, "output_scatter_aten_calls")
            if buckets is not None:
                buckets["runnercore1_policy_forward_gru_rows"] = float(
                    buckets.get("runnercore1_policy_forward_gru_rows", 0.0) + float(gru_idx.shape[0])
                )

    result = actions_all.to("cpu").numpy().astype(np.int32, copy=False)
    if cxxbackbone5_profile:
        if str(device).startswith("cuda"):
            _cxxbackbone5_bucket_add(buckets, "d2h_bytes", float(result.nbytes))
            _cxxbackbone5_bucket_add(buckets, "cuda_to_cpu_action_copies")
        _cxxbackbone5_bucket_add(buckets, "policy_result_rows", float(result.shape[0]))
    if cxxbackbone2_policy_probe_before_hmem is not None:
        from perf.cxxbackbone2_policy_core_live_probe import run_policy_core_live_probe  # noqa: WPS433

        run_policy_core_live_probe(
            runner,
            state,
            agent_ids=np.asarray(agent_ids, dtype=np.int32),
            slots=np.asarray(slots, dtype=np.int32),
            obs_np=np.asarray(obs_np, dtype=np.float32),
            produced_actions=result,
            hmem_before=cxxbackbone2_policy_probe_before_hmem,
            k=int(k),
            buckets=buckets,
        )
    if profile_enabled:
        runnercore3_profile_add(state, "policy_forward_ms", time.perf_counter() - t_forward)
    return result


def runnercore1_actions_strided(
    runner: Any,
    state: RunnerCore1TickState,
    agent_ids: np.ndarray,
    *,
    buckets: dict[str, float] | None = None,
) -> np.ndarray:
    stride = int(state.stride)
    n = int(agent_ids.shape[0])
    out = np.empty((n,), dtype=np.int32)
    if n <= 0:
        return out
    profile_enabled = _runnercore3_profile_enabled(runner)
    cxxbackbone5_profile = _cxxbackbone5_policy_profile_enabled(runner)
    if cxxbackbone5_profile:
        _cxxbackbone5_bucket_add(buckets, "cache_service_python_calls")
        _cxxbackbone5_bucket_add(buckets, "cache_service_rows", float(n))
    if _cxxbackbone6_vertical_slice_enabled(runner):
        from perf.cxxbackbone6_policy_service import execute_tick_flat_policy_service_v1  # noqa: WPS433

        result = execute_tick_flat_policy_service_v1(runner, state, agent_ids, buckets=buckets)
        if result is not None:
            return result
        if _as_bool(getattr(runner.sim_cfg, "cxxbackbone6_resident_policy_vertical_slice_fail_closed", True)):
            raise RunnerCore1ActionError("CXXBACKBONE6 resident policy vertical slice unsupported for current run")
    if int(stride) == 1 and _runnercore33_stride1_direct_action_enabled(runner):
        t_refill = time.perf_counter() if profile_enabled else 0.0
        order = np.argsort(np.asarray(agent_ids, dtype=np.int64), kind="stable")
        sorted_ids = np.asarray(agent_ids, dtype=np.int32)[order].astype(np.int32, copy=False)
        actions_k = _compute_actions_k(runner, state, sorted_ids, k=1, buckets=buckets)
        out[order] = actions_k[:, 0]
        state.counters.action_cache_misses += int(n)
        if buckets is not None:
            buckets["runnercore33_stride1_direct_action_rows"] = float(
                buckets.get("runnercore33_stride1_direct_action_rows", 0.0) + float(n)
            )
        if cxxbackbone5_profile:
            _cxxbackbone5_bucket_add(buckets, "cache_stride1_direct_rows", float(n))
            _cxxbackbone5_bucket_add(buckets, "cache_refill_rows", float(n))
        if profile_enabled:
            runnercore3_profile_add(state, "action_cache_refill_ms", time.perf_counter() - t_refill)
        return out
    vectorized_cache = _runnercore3_vectorized_action_cache_enabled(runner)
    refill_ids: list[int] = []
    refill_out_idx: list[int] = []
    refill_slots: list[int] = []

    t_serve = time.perf_counter() if profile_enabled else 0.0
    if vectorized_cache:
        ids_i64 = np.asarray(agent_ids, dtype=np.int64)
        slots = slots_for_agent_ids(state, ids_i64)
        if bool(np.any(slots < 0)):
            bad = int(ids_i64[int(np.nonzero(slots < 0)[0][0])])
            raise RunnerCore1ActionError(f"unknown agent id in action request: {bad}")
        cursors = np.asarray(state.cache_cursor[slots], dtype=np.int32)
        hit_mask = (
            (np.asarray(state.cache_agent_id[slots], dtype=np.int64) == ids_i64)
            & (np.asarray(state.cache_token[slots], dtype=np.int64) == np.asarray(state.slot_tokens[slots], dtype=np.int64))
            & (cursors >= np.int32(0))
            & (cursors < np.int32(stride))
        )
        hit_idx = np.nonzero(hit_mask)[0].astype(np.int64)
        if int(hit_idx.shape[0]) > 0:
            hit_slots = slots[hit_idx]
            hit_cursors = cursors[hit_idx]
            out[hit_idx] = state.cache_actions[hit_slots, hit_cursors]
            state.cache_cursor[hit_slots] = (hit_cursors + np.int32(1)).astype(np.int32, copy=False)
            state.counters.action_cache_hits += int(hit_idx.shape[0])
        miss_idx = np.nonzero(~hit_mask)[0].astype(np.int64)
        if int(miss_idx.shape[0]) > 0:
            refill_ids_arr = ids_i64[miss_idx].astype(np.int32, copy=False)
            refill_out_idx_arr = miss_idx.astype(np.int64, copy=False)
            refill_slots_arr = slots[miss_idx].astype(np.int32, copy=False)
            state.counters.action_cache_misses += int(miss_idx.shape[0])
        else:
            refill_ids_arr = np.empty((0,), dtype=np.int32)
            refill_out_idx_arr = np.empty((0,), dtype=np.int64)
            refill_slots_arr = np.empty((0,), dtype=np.int32)
    else:
        for i, aid_raw in enumerate(agent_ids.tolist()):
            aid = int(aid_raw)
            slot = state.slot_for_agent_id(aid)
            if slot < 0:
                raise RunnerCore1ActionError(f"unknown agent id in action request: {aid}")
            token = int(state.slot_tokens[slot])
            if (
                int(state.cache_agent_id[slot]) == aid
                and int(state.cache_token[slot]) == token
                and 0 <= int(state.cache_cursor[slot]) < stride
            ):
                cursor = int(state.cache_cursor[slot])
                out[i] = np.int32(state.cache_actions[slot, cursor])
                state.cache_cursor[slot] = np.int32(cursor + 1)
                state.counters.action_cache_hits += 1
                if cxxbackbone5_profile:
                    _cxxbackbone5_bucket_add(buckets, "cache_hits")
                continue
            refill_ids.append(aid)
            refill_out_idx.append(i)
            refill_slots.append(slot)
            state.counters.action_cache_misses += 1
            if cxxbackbone5_profile:
                _cxxbackbone5_bucket_add(buckets, "cache_misses")
        refill_ids_arr = np.asarray(refill_ids, dtype=np.int32)
        refill_out_idx_arr = np.asarray(refill_out_idx, dtype=np.int64)
        refill_slots_arr = np.asarray(refill_slots, dtype=np.int32)

    if profile_enabled:
        runnercore3_profile_add(state, "action_cache_serve_ms", time.perf_counter() - t_serve)
    if cxxbackbone5_profile and vectorized_cache:
        _cxxbackbone5_bucket_add(buckets, "cache_hits", float(hit_idx.shape[0]))
        _cxxbackbone5_bucket_add(buckets, "cache_misses", float(miss_idx.shape[0]))

    if int(refill_ids_arr.shape[0]) > 0:
        t_refill = time.perf_counter() if profile_enabled else 0.0
        order = np.argsort(refill_ids_arr.astype(np.int64, copy=False), kind="stable")
        sorted_ids = refill_ids_arr[order].astype(np.int32, copy=False)
        actions_k = _compute_actions_k(runner, state, sorted_ids, k=stride, buckets=buckets)
        if vectorized_cache:
            out_idx = refill_out_idx_arr[order]
            slot_idx = refill_slots_arr[order]
            out[out_idx] = actions_k[:, 0]
            state.cache_agent_id[slot_idx] = sorted_ids.astype(np.int64, copy=False)
            state.cache_token[slot_idx] = state.slot_tokens[slot_idx]
            state.cache_cursor[slot_idx] = np.int32(1)
            state.cache_actions[slot_idx, :stride] = actions_k[:, :stride]
            state.counters.action_cache_refill_rows += int(slot_idx.shape[0])
        else:
            for row_i, src_i in enumerate(order.tolist()):
                out_i = int(refill_out_idx_arr[int(src_i)])
                slot = int(refill_slots_arr[int(src_i)])
                aid = int(refill_ids_arr[int(src_i)])
                seq = np.asarray(actions_k[int(row_i)], dtype=np.int32)
                out[out_i] = np.int32(seq[0])
                state.cache_agent_id[slot] = np.int64(aid)
                state.cache_token[slot] = np.int64(state.slot_tokens[slot])
                state.cache_cursor[slot] = np.int32(1)
                state.cache_actions[slot, :stride] = seq[:stride]
                state.counters.action_cache_refill_rows += 1
        if profile_enabled:
            runnercore3_profile_add(state, "action_cache_refill_ms", time.perf_counter() - t_refill)
        if cxxbackbone5_profile:
            _cxxbackbone5_bucket_add(buckets, "cache_refill_rows", float(refill_ids_arr.shape[0]))
            _cxxbackbone5_bucket_add(buckets, "cache_python_writeback_calls")
    return out


def action_cache_contract_payload() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "owned_arrays": [
            "cache_agent_id",
            "cache_token",
            "cache_cursor",
            "cache_actions",
        ],
        "legacy_forbidden_calls": [
            "PolicyMixin._batched_actions",
            "PolicyMixin._batched_actions_k",
            "TickLoopMixin._batched_actions_strided",
        ],
        "status": "array_authoritative_for_supported_candidate",
    }
