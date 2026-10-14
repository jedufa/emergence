from __future__ import annotations

"""CXXBACKBONE-8R native generation-island binding."""

from pathlib import Path
from typing import Any
import time

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "engine" / "perf" / "native" / "cxxbackbone8r_generation_island.cpp"
SOURCE_CUDA = ROOT / "engine" / "perf" / "native" / "cxxbackbone8r_generation_island_cuda.cu"
_MODULE: Any | None = None
_CUDA_MODULE: Any | None = None
ONECALL_ENVELOPE_SCHEMA_VERSION = "cxxbackbone8r_onecall_generation_envelope_v1"


def load_generation_island_extension(*, build_directory: str | Path | None = None, verbose: bool = False) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    source = SOURCE.read_text(encoding="utf-8")
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load_inline(
        name="cxxbackbone8r_generation_island_ext",
        cpp_sources=[source],
        functions=[
            "cxxbackbone8r_flat_apply_actions_cpu_v1",
            "cxxbackbone8r_patch_apply_actions_cpu_v1",
            "cxxbackbone8r_flat_observation_frame_cpu_v1",
            "cxxbackbone8r_patch_observation_frame_cpu_v1",
            "cxxbackbone8r_group_rows_cpu_v1",
        ],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2"],
        with_cuda=False,
        verbose=bool(verbose),
    )
    return _MODULE


def load_generation_island_cuda_extension(*, build_directory: str | Path | None = None, verbose: bool = False) -> Any:
    global _CUDA_MODULE
    if _CUDA_MODULE is not None:
        return _CUDA_MODULE
    cuda_source = SOURCE_CUDA.read_text(encoding="utf-8")
    cpp_source = """
    #include <torch/extension.h>

    torch::Tensor cxxbackbone8r_patch_observation_frame_cuda_v1(
        torch::Tensor agent_ids,
        torch::Tensor slots,
        torch::Tensor agent_pos_y,
        torch::Tensor agent_pos_x,
        torch::Tensor agent_grid,
        torch::Tensor food_grid,
        torch::Tensor energy_by_slot,
        torch::Tensor health_by_slot,
        torch::Tensor patch_food_amount,
        torch::Tensor patch_cover,
        torch::Tensor patch_shelter,
        torch::Tensor patch_scent,
        torch::Tensor patch_waste,
        torch::Tensor patch_movement_cost,
        const int64_t view_radius,
        const double max_energy,
        const double start_health,
        const double patch_food_amount_max,
        const double patch_scent_max,
        const double patch_waste_max,
        const int64_t patch_density_radius,
        const bool patch_density_include_self,
        const bool patch_density_normalize,
        const double patch_movement_cost_min,
        const double patch_movement_cost_max
    );
    """
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _CUDA_MODULE = load_inline(
        name="cxxbackbone8r_generation_island_cuda_ext",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["cxxbackbone8r_patch_observation_frame_cuda_v1"],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _CUDA_MODULE


def flat_apply_actions_cpu(
    *,
    agent_ids: np.ndarray,
    actions: np.ndarray,
    agent_pos_y: np.ndarray,
    agent_pos_x: np.ndarray,
    agent_grid: np.ndarray,
    food_grid: np.ndarray,
    energies: np.ndarray,
    healths: np.ndarray,
    injuries: np.ndarray,
    scars: np.ndarray,
    move_cost: float,
    gather_cost: float,
    rest_cost: float,
    attack_cost: float,
    action_cost_mult: float,
    rest_cost_mult: float,
    max_energy: float,
    food_energy: float,
    build_directory: str | Path | None = None,
) -> dict[str, np.ndarray]:
    mod = load_generation_island_extension(build_directory=build_directory)
    pos_y_t = torch.from_numpy(np.ascontiguousarray(agent_pos_y, dtype=np.int32))
    pos_x_t = torch.from_numpy(np.ascontiguousarray(agent_pos_x, dtype=np.int32))
    grid_t = torch.from_numpy(np.ascontiguousarray(agent_grid, dtype=np.int32))
    food_t = torch.from_numpy(np.ascontiguousarray(food_grid, dtype=np.int32))
    outputs = mod.cxxbackbone8r_flat_apply_actions_cpu_v1(
        torch.from_numpy(np.ascontiguousarray(agent_ids, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(actions, dtype=np.int32)),
        pos_y_t,
        pos_x_t,
        grid_t,
        food_t,
        torch.from_numpy(np.ascontiguousarray(energies, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(healths, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(injuries, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(scars, dtype=np.float32)),
        float(move_cost),
        float(gather_cost),
        float(rest_cost),
        float(attack_cost),
        float(action_cost_mult),
        float(rest_cost_mult),
        float(max_energy),
        float(food_energy),
    )
    keys = (
        "died_mask",
        "new_energy",
        "new_health",
        "food_gained",
        "new_injury",
        "new_scar",
        "moved_mask",
    )
    result = {key: tensor.detach().cpu().numpy().copy() for key, tensor in zip(keys, outputs, strict=True)}
    result["agent_pos_y"] = pos_y_t.numpy().copy()
    result["agent_pos_x"] = pos_x_t.numpy().copy()
    result["agent_grid"] = grid_t.numpy().copy()
    result["food_grid"] = food_t.numpy().copy()
    result["executed"] = np.asarray([1], dtype=np.int32)
    return result


def patch_apply_actions_cpu(
    *,
    agent_ids: np.ndarray,
    actions: np.ndarray,
    agent_pos_y: np.ndarray,
    agent_pos_x: np.ndarray,
    agent_grid: np.ndarray,
    food_grid: np.ndarray,
    patch_food_amount: np.ndarray,
    patch_scent_emit_pending: np.ndarray,
    patch_waste_emit_pending: np.ndarray,
    patch_movement_cost: np.ndarray,
    energies: np.ndarray,
    healths: np.ndarray,
    injuries: np.ndarray,
    scars: np.ndarray,
    move_cost: float,
    gather_cost: float,
    rest_cost: float,
    attack_cost: float,
    action_cost_mult: float,
    rest_cost_mult: float,
    max_energy: float,
    food_energy: float,
    patch_food_gather_amount: float,
    patch_food_presence_threshold: float,
    patch_scent_move_emit: float,
    patch_waste_move_emit: float,
    patch_scent_gather_emit: float,
    patch_waste_consume_emit: float,
    patch_movement_cost_apply_enable: bool,
    patch_movement_cost_energy_scale: float,
    build_directory: str | Path | None = None,
) -> dict[str, np.ndarray]:
    mod = load_generation_island_extension(build_directory=build_directory)
    pos_y_t = torch.from_numpy(np.ascontiguousarray(agent_pos_y, dtype=np.int32))
    pos_x_t = torch.from_numpy(np.ascontiguousarray(agent_pos_x, dtype=np.int32))
    grid_t = torch.from_numpy(np.ascontiguousarray(agent_grid, dtype=np.int32))
    food_t = torch.from_numpy(np.ascontiguousarray(food_grid, dtype=np.int32))
    patch_food_t = torch.from_numpy(np.ascontiguousarray(patch_food_amount, dtype=np.float32))
    scent_pending_t = torch.from_numpy(np.ascontiguousarray(patch_scent_emit_pending, dtype=np.float32))
    waste_pending_t = torch.from_numpy(np.ascontiguousarray(patch_waste_emit_pending, dtype=np.float32))
    outputs = mod.cxxbackbone8r_patch_apply_actions_cpu_v1(
        torch.from_numpy(np.ascontiguousarray(agent_ids, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(actions, dtype=np.int32)),
        pos_y_t,
        pos_x_t,
        grid_t,
        food_t,
        patch_food_t,
        scent_pending_t,
        waste_pending_t,
        torch.from_numpy(np.ascontiguousarray(patch_movement_cost, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(energies, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(healths, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(injuries, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(scars, dtype=np.float32)),
        float(move_cost),
        float(gather_cost),
        float(rest_cost),
        float(attack_cost),
        float(action_cost_mult),
        float(rest_cost_mult),
        float(max_energy),
        float(food_energy),
        float(patch_food_gather_amount),
        float(patch_food_presence_threshold),
        float(patch_scent_move_emit),
        float(patch_waste_move_emit),
        float(patch_scent_gather_emit),
        float(patch_waste_consume_emit),
        bool(patch_movement_cost_apply_enable),
        float(patch_movement_cost_energy_scale),
    )
    keys = (
        "died_mask",
        "new_energy",
        "new_health",
        "food_gained",
        "new_injury",
        "new_scar",
        "moved_mask",
        "patch_food_harvest_total",
        "patch_scent_move_queued_total",
        "patch_waste_move_queued_total",
        "patch_scent_gather_queued_total",
        "patch_waste_gather_queued_total",
        "patch_movement_energy_extra",
        "patch_movement_energy_event_count",
    )
    result = {key: tensor.detach().cpu().numpy().copy() for key, tensor in zip(keys, outputs, strict=True)}
    result["agent_pos_y"] = pos_y_t.numpy().copy()
    result["agent_pos_x"] = pos_x_t.numpy().copy()
    result["agent_grid"] = grid_t.numpy().copy()
    result["food_grid"] = food_t.numpy().copy()
    result["patch_food_amount"] = patch_food_t.numpy().copy()
    result["patch_scent_emit_pending"] = scent_pending_t.numpy().copy()
    result["patch_waste_emit_pending"] = waste_pending_t.numpy().copy()
    result["executed"] = np.asarray([1], dtype=np.int32)
    return result


def flat_observation_frame_cpu(
    *,
    agent_ids: np.ndarray,
    slots: np.ndarray,
    agent_pos_y: np.ndarray,
    agent_pos_x: np.ndarray,
    agent_grid: np.ndarray,
    food_grid: np.ndarray,
    energy_by_slot: np.ndarray,
    health_by_slot: np.ndarray,
    view_radius: int,
    max_energy: float,
    start_health: float,
    build_directory: str | Path | None = None,
) -> np.ndarray:
    mod = load_generation_island_extension(build_directory=build_directory)
    out = mod.cxxbackbone8r_flat_observation_frame_cpu_v1(
        torch.from_numpy(np.ascontiguousarray(agent_ids, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(slots, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(agent_pos_y, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(agent_pos_x, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(agent_grid, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(food_grid, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(energy_by_slot, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(health_by_slot, dtype=np.float32)),
        int(view_radius),
        float(max_energy),
        float(start_health),
    )
    return out.detach().cpu().numpy().copy()


def patch_observation_frame_cpu(
    *,
    agent_ids: np.ndarray,
    slots: np.ndarray,
    agent_pos_y: np.ndarray,
    agent_pos_x: np.ndarray,
    agent_grid: np.ndarray,
    food_grid: np.ndarray,
    energy_by_slot: np.ndarray,
    health_by_slot: np.ndarray,
    patch_food_amount: np.ndarray,
    patch_cover: np.ndarray,
    patch_shelter: np.ndarray,
    patch_scent: np.ndarray,
    patch_waste: np.ndarray,
    patch_movement_cost: np.ndarray,
    view_radius: int,
    max_energy: float,
    start_health: float,
    patch_food_amount_max: float,
    patch_scent_max: float,
    patch_waste_max: float,
    patch_density_radius: int,
    patch_density_include_self: bool,
    patch_density_normalize: bool,
    patch_movement_cost_min: float,
    patch_movement_cost_max: float,
    build_directory: str | Path | None = None,
) -> np.ndarray:
    mod = load_generation_island_extension(build_directory=build_directory)
    out = mod.cxxbackbone8r_patch_observation_frame_cpu_v1(
        torch.from_numpy(np.ascontiguousarray(agent_ids, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(slots, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(agent_pos_y, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(agent_pos_x, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(agent_grid, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(food_grid, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(energy_by_slot, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(health_by_slot, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(patch_food_amount, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(patch_cover, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(patch_shelter, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(patch_scent, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(patch_waste, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(patch_movement_cost, dtype=np.float32)),
        int(view_radius),
        float(max_energy),
        float(start_health),
        float(patch_food_amount_max),
        float(patch_scent_max),
        float(patch_waste_max),
        int(patch_density_radius),
        bool(patch_density_include_self),
        bool(patch_density_normalize),
        float(patch_movement_cost_min),
        float(patch_movement_cost_max),
    )
    return out.detach().cpu().numpy().copy()


def patch_observation_frame_cuda(
    *,
    agent_ids: torch.Tensor,
    slots: torch.Tensor,
    agent_pos_y: torch.Tensor,
    agent_pos_x: torch.Tensor,
    agent_grid: torch.Tensor,
    food_grid: torch.Tensor,
    energy_by_slot: torch.Tensor,
    health_by_slot: torch.Tensor,
    patch_food_amount: torch.Tensor,
    patch_cover: torch.Tensor,
    patch_shelter: torch.Tensor,
    patch_scent: torch.Tensor,
    patch_waste: torch.Tensor,
    patch_movement_cost: torch.Tensor,
    view_radius: int,
    max_energy: float,
    start_health: float,
    patch_food_amount_max: float,
    patch_scent_max: float,
    patch_waste_max: float,
    patch_density_radius: int,
    patch_density_include_self: bool,
    patch_density_normalize: bool,
    patch_movement_cost_min: float,
    patch_movement_cost_max: float,
    build_directory: str | Path | None = None,
) -> torch.Tensor:
    mod = load_generation_island_cuda_extension(build_directory=build_directory)
    return mod.cxxbackbone8r_patch_observation_frame_cuda_v1(
        agent_ids.contiguous().to(dtype=torch.int32),
        slots.contiguous().to(dtype=torch.int32),
        agent_pos_y.contiguous().to(dtype=torch.int32),
        agent_pos_x.contiguous().to(dtype=torch.int32),
        agent_grid.contiguous().to(dtype=torch.int32),
        food_grid.contiguous().to(dtype=torch.int32),
        energy_by_slot.contiguous().to(dtype=torch.float32),
        health_by_slot.contiguous().to(dtype=torch.float32),
        patch_food_amount.contiguous().to(dtype=torch.float32),
        patch_cover.contiguous().to(dtype=torch.float32),
        patch_shelter.contiguous().to(dtype=torch.float32),
        patch_scent.contiguous().to(dtype=torch.float32),
        patch_waste.contiguous().to(dtype=torch.float32),
        patch_movement_cost.contiguous().to(dtype=torch.float32),
        int(view_radius),
        float(max_energy),
        float(start_health),
        float(patch_food_amount_max),
        float(patch_scent_max),
        float(patch_waste_max),
        int(patch_density_radius),
        bool(patch_density_include_self),
        bool(patch_density_normalize),
        float(patch_movement_cost_min),
        float(patch_movement_cost_max),
    )


def execute_onecall_generation_envelope(
    runner: Any,
    *,
    generation_index: int,
    buckets: dict[str, float],
    injury_recover_mult: float,
    deaths_this_gen: int,
    successful_births: int,
    failed_births: int,
    gen_peak_injury: float,
    rest_streaks: dict[int, int],
) -> dict[str, Any]:
    started = time.perf_counter()
    before = dict(buckets)
    result = {
        "schema_version": ONECALL_ENVELOPE_SCHEMA_VERSION,
        "generation": int(generation_index),
        "executed": False,
        "hidden_fallback": False,
        "monolithic_cpp_loop": False,
        "cuda_resident_generation_loop": False,
        "native_boundary_route": "none",
        "elapsed_s": 0.0,
        "counters_before": before,
        "counters_after": {},
        "counter_delta": {},
        "checks": {},
    }
    flat_native = bool(getattr(getattr(runner, "sim_cfg", None), "cxxbackbone8r_flat_cpu_native_action_apply_enable", False))
    patch_native = bool(getattr(getattr(runner, "sim_cfg", None), "cxxbackbone8r_patch_cpu_native_action_apply_enable", False))
    flat_inputs = bool(getattr(getattr(runner, "sim_cfg", None), "cxxbackbone8r_flat_cpu_native_policy_inputs_enable", False))
    patch_inputs = bool(getattr(getattr(runner, "sim_cfg", None), "cxxbackbone8r_patch_cpu_native_policy_inputs_enable", False))
    if patch_native and patch_inputs:
        result["native_boundary_route"] = "patch_cpu"
    elif flat_native and flat_inputs:
        result["native_boundary_route"] = "flat_cpu"
    else:
        result["native_boundary_route"] = "unsupported"
        result["elapsed_s"] = float(time.perf_counter() - started)
        return result

    runner._run_ticks(
        gen=int(generation_index),
        buckets=buckets,
        injury_recover_mult=float(injury_recover_mult),
        deaths_this_gen=int(deaths_this_gen),
        successful_births=int(successful_births),
        failed_births=int(failed_births),
        gen_peak_injury=float(gen_peak_injury),
        rest_streaks=rest_streaks,
    )
    after = dict(buckets)
    delta = {
        key: float(after.get(key, 0.0) or 0.0) - float(before.get(key, 0.0) or 0.0)
        for key in sorted(set(before) | set(after))
    }
    policy_grouping = float(after.get("cxxbackbone5_policy_grouping_python_calls", 0.0) or 0.0)
    row_maps = float(after.get("cxxbackbone5_row_map_allocations", 0.0) or 0.0)
    slot_lookup = float(after.get("cxxbackbone5_slot_lookup_python_calls", 0.0) or 0.0)
    obs_callbacks = float(after.get("cxxbackbone5_observation_python_callbacks", 0.0) or 0.0)
    native_actions = float(delta.get("cxxbackbone8r_native_action_application_calls", 0.0) or 0.0)
    native_obs = float(delta.get("cxxbackbone5_cxxbackbone8r_native_observation_frame_calls", 0.0) or 0.0)
    result.update(
        {
            "executed": True,
            "elapsed_s": float(time.perf_counter() - started),
            "counters_after": after,
            "counter_delta": delta,
            "checks": {
                "native_action_calls_positive": native_actions > 0.0,
                "native_observation_calls_positive": native_obs > 0.0,
                "python_policy_grouping_zero": policy_grouping == 0.0,
                "python_row_maps_zero": row_maps == 0.0,
                "python_slot_lookup_zero": slot_lookup == 0.0,
                "python_observation_callbacks_zero": obs_callbacks == 0.0,
            },
        }
    )
    result["hidden_fallback"] = not all(bool(value) for value in result["checks"].values())
    buckets["cxxbackbone8r_onecall_generation_envelope_calls"] = float(
        buckets.get("cxxbackbone8r_onecall_generation_envelope_calls", 0.0) + 1.0
    )
    buckets["cxxbackbone8r_onecall_generation_envelope_wall_s"] = float(
        buckets.get("cxxbackbone8r_onecall_generation_envelope_wall_s", 0.0) + float(result["elapsed_s"])
    )
    return result


def group_rows_cpu(
    *,
    agent_ids: np.ndarray,
    agent_id_to_slot: np.ndarray,
    use_gru: np.ndarray,
    build_directory: str | Path | None = None,
) -> dict[str, np.ndarray]:
    mod = load_generation_island_extension(build_directory=build_directory)
    outputs = mod.cxxbackbone8r_group_rows_cpu_v1(
        torch.from_numpy(np.ascontiguousarray(agent_ids, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(agent_id_to_slot, dtype=np.int32)),
        torch.from_numpy(np.ascontiguousarray(use_gru, dtype=np.bool_)),
    )
    keys = ("slots", "mlp_idx", "gru_idx")
    return {key: tensor.detach().cpu().numpy().copy() for key, tensor in zip(keys, outputs, strict=True)}


__all__ = [
    "flat_apply_actions_cpu",
    "flat_observation_frame_cpu",
    "group_rows_cpu",
    "load_generation_island_cuda_extension",
    "load_generation_island_extension",
    "patch_apply_actions_cpu",
    "patch_observation_frame_cuda",
    "patch_observation_frame_cpu",
    "execute_onecall_generation_envelope",
]
