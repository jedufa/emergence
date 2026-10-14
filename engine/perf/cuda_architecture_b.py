from __future__ import annotations

"""Bounded resident-CUDA bring-up surface for Architecture B."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
CUDA_SOURCE = ROOT / "engine" / "perf" / "native" / "cxxbackbone8r_generation_island_cuda.cu"
_MODULE: Any | None = None

CUDA_STAGE1_ACTION_CAPABILITIES = {
    "schema_version": "eiel_cuda_architecture_b_stage1_action_capabilities_v1",
    "decision": "ARCHB-CUDA-STAGE1A/RESIDENT-SCRIPTED-ACTION-SLICE",
    "cuda_state_resident_between_generation_calls": True,
    "native_whole_scripted_generation_call": True,
    "python_tick_loop": False,
    "flat_base_action_state_mutation": True,
    "patch_base_action_state_mutation": True,
    "slot_indexed_observation_resident": True,
    "fixed_capacity_counted_active_rows": True,
    "device_resident_active_count": True,
    "fixed_mlp_policy_resident": False,
    "birth_death_slot_lifecycle_resident": False,
    "checkpoint_continuation_qualified": False,
    "multi_cell_resident": False,
    "promotion_qualified": False,
}


def load_cuda_architecture_b_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B CUDA Stage 1 requires an available CUDA device")
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_cuda_scripted_action_generation_v1(
        torch::Tensor agent_ids,
        torch::Tensor actions_by_tick_agent,
        torch::Tensor agent_pos_y,
        torch::Tensor agent_pos_x,
        torch::Tensor agent_grid,
        torch::Tensor food_grid,
        torch::Tensor patch_food_amount,
        torch::Tensor patch_scent_emit_pending,
        torch::Tensor patch_waste_emit_pending,
        torch::Tensor patch_movement_cost,
        torch::Tensor energies,
        torch::Tensor healths,
        const bool patch_enabled,
        const double move_cost,
        const double gather_cost,
        const double rest_cost,
        const double attack_cost,
        const double action_cost_mult,
        const double rest_cost_mult,
        const double max_energy,
        const double food_energy,
        const double patch_food_gather_amount,
        const double patch_food_presence_threshold,
        const double patch_scent_move_emit,
        const double patch_waste_move_emit,
        const double patch_scent_gather_emit,
        const double patch_waste_consume_emit,
        const bool patch_movement_cost_apply_enable,
        const double patch_movement_cost_energy_scale
    );

    std::vector<torch::Tensor> eiel_cuda_scripted_owner_action_generation_v1(
        torch::Tensor agent_ids_by_alive_order,
        torch::Tensor agent_id_to_slot,
        torch::Tensor actions_by_tick_slot,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor food_grid,
        torch::Tensor patch_food_amount,
        torch::Tensor patch_scent_emit_pending,
        torch::Tensor patch_waste_emit_pending,
        torch::Tensor patch_movement_cost,
        torch::Tensor energy,
        torch::Tensor health,
        const bool patch_enabled,
        const double move_cost,
        const double gather_cost,
        const double rest_cost,
        const double attack_cost,
        const double action_cost_mult,
        const double rest_cost_mult,
        const double max_energy,
        const double food_energy,
        const double patch_food_gather_amount,
        const double patch_food_presence_threshold,
        const double patch_scent_move_emit,
        const double patch_waste_move_emit,
        const double patch_scent_gather_emit,
        const double patch_waste_consume_emit,
        const bool patch_movement_cost_apply_enable,
        const double patch_movement_cost_energy_scale
    );

    std::vector<torch::Tensor> eiel_cuda_scripted_owner_action_generation_counted_v1(
        torch::Tensor agent_ids_by_alive_order,
        torch::Tensor active_count,
        torch::Tensor agent_id_to_slot,
        torch::Tensor actions_by_tick_slot,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor food_grid,
        torch::Tensor patch_food_amount,
        torch::Tensor patch_scent_emit_pending,
        torch::Tensor patch_waste_emit_pending,
        torch::Tensor patch_movement_cost,
        torch::Tensor energy,
        torch::Tensor health,
        const bool patch_enabled,
        const double move_cost,
        const double gather_cost,
        const double rest_cost,
        const double attack_cost,
        const double action_cost_mult,
        const double rest_cost_mult,
        const double max_energy,
        const double food_energy,
        const double patch_food_gather_amount,
        const double patch_food_presence_threshold,
        const double patch_scent_move_emit,
        const double patch_waste_move_emit,
        const double patch_scent_gather_emit,
        const double patch_waste_consume_emit,
        const bool patch_movement_cost_apply_enable,
        const double patch_movement_cost_energy_scale
    );

    torch::Tensor eiel_cuda_owner_observation_frame_v1(
        torch::Tensor agent_ids_by_alive_order,
        torch::Tensor agent_id_to_slot,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor food_grid,
        torch::Tensor energy,
        torch::Tensor health,
        torch::Tensor patch_food_amount,
        torch::Tensor patch_cover,
        torch::Tensor patch_shelter,
        torch::Tensor patch_scent,
        torch::Tensor patch_waste,
        torch::Tensor patch_movement_cost,
        const bool patch_enabled,
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

    torch::Tensor eiel_cuda_owner_observation_frame_counted_v1(
        torch::Tensor agent_ids_by_alive_order,
        torch::Tensor active_count,
        torch::Tensor agent_id_to_slot,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor food_grid,
        torch::Tensor energy,
        torch::Tensor health,
        torch::Tensor patch_food_amount,
        torch::Tensor patch_cover,
        torch::Tensor patch_shelter,
        torch::Tensor patch_scent,
        torch::Tensor patch_waste,
        torch::Tensor patch_movement_cost,
        const bool patch_enabled,
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

    torch::Tensor eiel_cuda_owner_observation_frame_batched_counted_v1(
        torch::Tensor agent_ids_by_alive_order,
        torch::Tensor active_count,
        torch::Tensor agent_id_to_slot,
        torch::Tensor pos_y,
        torch::Tensor pos_x,
        torch::Tensor agent_grid,
        torch::Tensor food_grid,
        torch::Tensor energy,
        torch::Tensor health,
        torch::Tensor patch_food_amount,
        torch::Tensor patch_cover,
        torch::Tensor patch_shelter,
        torch::Tensor patch_scent,
        torch::Tensor patch_waste,
        torch::Tensor patch_movement_cost,
        const bool patch_enabled,
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
    _MODULE = load_inline(
        name="eiel_cuda_architecture_b_stage1_ext",
        cpp_sources=[cpp_source],
        cuda_sources=[CUDA_SOURCE.read_text(encoding="utf-8")],
        functions=[
            "eiel_cuda_scripted_action_generation_v1",
            "eiel_cuda_scripted_owner_action_generation_v1",
            "eiel_cuda_scripted_owner_action_generation_counted_v1",
            "eiel_cuda_owner_observation_frame_v1",
            "eiel_cuda_owner_observation_frame_counted_v1",
            "eiel_cuda_owner_observation_frame_batched_counted_v1",
        ],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


@dataclass(frozen=True)
class CudaActionConfig:
    move_cost: float
    gather_cost: float
    rest_cost: float
    attack_cost: float
    action_cost_mult: float = 1.0
    rest_cost_mult: float = 1.0
    max_energy: float = 100.0
    food_energy: float = 20.0
    patch_food_gather_amount: float = 1.0
    patch_food_presence_threshold: float = 0.5
    patch_scent_move_emit: float = 0.0
    patch_waste_move_emit: float = 0.0
    patch_scent_gather_emit: float = 0.0
    patch_waste_consume_emit: float = 0.0
    patch_movement_cost_apply_enable: bool = False
    patch_movement_cost_energy_scale: float = 0.0


@dataclass(frozen=True)
class CudaObservationConfig:
    view_radius: int = 1
    max_energy: float = 100.0
    start_health: float = 1.0
    patch_food_amount_max: float = 1.0
    patch_scent_max: float = 1.0
    patch_waste_max: float = 1.0
    patch_density_radius: int = 1
    patch_density_include_self: bool = True
    patch_density_normalize: bool = False
    patch_movement_cost_min: float = 0.0
    patch_movement_cost_max: float = 4.0


class ResidentCudaActionState:
    """Owns the bounded Stage 1A action state on one CUDA device."""

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        patch_enabled: bool,
        device: torch.device,
        imported_state_bytes: int,
    ) -> None:
        self.tensors = dict(tensors)
        self.patch_enabled = bool(patch_enabled)
        self.device = device
        self.imported_state_bytes = int(imported_state_bytes)
        self.generation_calls = 0
        self.action_payload_imports = 0
        self.action_payload_bytes = 0
        self._last_outputs: tuple[torch.Tensor, ...] | None = None

    @classmethod
    def from_arrays(
        cls,
        *,
        agent_ids: np.ndarray,
        agent_pos_y: np.ndarray,
        agent_pos_x: np.ndarray,
        agent_grid: np.ndarray,
        food_grid: np.ndarray,
        energies: np.ndarray,
        healths: np.ndarray,
        patch_enabled: bool,
        patch_food_amount: np.ndarray | None = None,
        patch_scent_emit_pending: np.ndarray | None = None,
        patch_waste_emit_pending: np.ndarray | None = None,
        patch_movement_cost: np.ndarray | None = None,
        device: str | torch.device = "cuda",
    ) -> "ResidentCudaActionState":
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("ResidentCudaActionState requires a CUDA device")
        grid = np.ascontiguousarray(agent_grid, dtype=np.int32)
        shape = grid.shape

        def patch_array(value: np.ndarray | None, fill: float) -> np.ndarray:
            if value is None:
                return np.full(shape, fill, dtype=np.float32)
            return np.ascontiguousarray(value, dtype=np.float32)

        arrays = {
            "agent_ids": np.ascontiguousarray(agent_ids, dtype=np.int32),
            "agent_pos_y": np.ascontiguousarray(agent_pos_y, dtype=np.int32),
            "agent_pos_x": np.ascontiguousarray(agent_pos_x, dtype=np.int32),
            "agent_grid": grid,
            "food_grid": np.ascontiguousarray(food_grid, dtype=np.int32),
            "patch_food_amount": patch_array(patch_food_amount, 0.0),
            "patch_scent_emit_pending": patch_array(patch_scent_emit_pending, 0.0),
            "patch_waste_emit_pending": patch_array(patch_waste_emit_pending, 0.0),
            "patch_movement_cost": patch_array(patch_movement_cost, 1.0),
            "energies": np.ascontiguousarray(energies, dtype=np.float32),
            "healths": np.ascontiguousarray(healths, dtype=np.float32),
        }
        if any(array.shape != shape for name, array in arrays.items() if name.startswith("patch_") or name == "food_grid"):
            raise ValueError("all world and Patch grids must have the same shape")
        tensors = {
            name: torch.from_numpy(array).to(device=resolved).contiguous()
            for name, array in arrays.items()
        }
        imported_state_bytes = sum(int(array.nbytes) for array in arrays.values())
        return cls(
            tensors,
            patch_enabled=patch_enabled,
            device=resolved,
            imported_state_bytes=imported_state_bytes,
        )

    def execute_scripted_generation(
        self,
        actions_by_tick_agent: np.ndarray | torch.Tensor,
        config: CudaActionConfig,
        *,
        build_directory: str | Path | None = None,
    ) -> dict[str, Any]:
        if isinstance(actions_by_tick_agent, torch.Tensor):
            actions = actions_by_tick_agent.detach().to(device=self.device, dtype=torch.int32).contiguous()
            action_bytes = int(actions.numel() * actions.element_size())
        else:
            action_array = np.ascontiguousarray(actions_by_tick_agent, dtype=np.int32)
            actions = torch.from_numpy(action_array).to(device=self.device).contiguous()
            action_bytes = int(action_array.nbytes)
        if actions.dim() != 2 or actions.size(1) != self.tensors["agent_ids"].size(0):
            raise ValueError("actions_by_tick_agent must be [ticks, agents]")
        mod = load_cuda_architecture_b_extension(build_directory=build_directory)
        outputs = mod.eiel_cuda_scripted_action_generation_v1(
            self.tensors["agent_ids"],
            actions,
            self.tensors["agent_pos_y"],
            self.tensors["agent_pos_x"],
            self.tensors["agent_grid"],
            self.tensors["food_grid"],
            self.tensors["patch_food_amount"],
            self.tensors["patch_scent_emit_pending"],
            self.tensors["patch_waste_emit_pending"],
            self.tensors["patch_movement_cost"],
            self.tensors["energies"],
            self.tensors["healths"],
            self.patch_enabled,
            config.move_cost,
            config.gather_cost,
            config.rest_cost,
            config.attack_cost,
            config.action_cost_mult,
            config.rest_cost_mult,
            config.max_energy,
            config.food_energy,
            config.patch_food_gather_amount,
            config.patch_food_presence_threshold,
            config.patch_scent_move_emit,
            config.patch_waste_move_emit,
            config.patch_scent_gather_emit,
            config.patch_waste_consume_emit,
            config.patch_movement_cost_apply_enable,
            config.patch_movement_cost_energy_scale,
        )
        self._last_outputs = tuple(outputs)
        self.generation_calls += 1
        self.action_payload_imports += 1
        self.action_payload_bytes += action_bytes
        return {
            "schema_version": "eiel_cuda_architecture_b_stage1_action_result_v1",
            "decision": "ARCHB-CUDA-STAGE1A/RESIDENT-SCRIPTED-ACTION-SLICE",
            "executed": True,
            "hidden_fallback": False,
            "native_generation_calls": self.generation_calls,
            "native_generation_kernel_launches": self.generation_calls,
            "python_tick_loops": 0,
            "ticks": int(actions.size(0)),
            "state_device": str(self.device),
            "state_import_count": 1,
            "state_import_bytes": self.imported_state_bytes,
            "action_payload_import_count": self.action_payload_imports,
            "action_payload_import_bytes": self.action_payload_bytes,
            "full_stage1_qualified": False,
        }

    def export_boundary(self) -> dict[str, Any]:
        if self._last_outputs is None:
            raise RuntimeError("no scripted generation has executed")
        energy, died, food_gained, moved_count, metrics_float, metrics_int = self._last_outputs
        torch.cuda.synchronize(self.device)
        floats = metrics_float.detach().cpu().numpy().copy()
        ints = metrics_int.detach().cpu().numpy().copy()
        return {
            "agent_pos_y": self.tensors["agent_pos_y"].detach().cpu().numpy().copy(),
            "agent_pos_x": self.tensors["agent_pos_x"].detach().cpu().numpy().copy(),
            "agent_grid": self.tensors["agent_grid"].detach().cpu().numpy().copy(),
            "food_grid": self.tensors["food_grid"].detach().cpu().numpy().copy(),
            "patch_food_amount": self.tensors["patch_food_amount"].detach().cpu().numpy().copy(),
            "patch_scent_emit_pending": self.tensors["patch_scent_emit_pending"].detach().cpu().numpy().copy(),
            "patch_waste_emit_pending": self.tensors["patch_waste_emit_pending"].detach().cpu().numpy().copy(),
            "energy": energy.detach().cpu().numpy().copy(),
            "died_mask": died.detach().cpu().numpy().copy(),
            "food_gained": food_gained.detach().cpu().numpy().copy(),
            "moved_count": moved_count.detach().cpu().numpy().copy(),
            "patch_food_harvest_total": float(floats[0]),
            "patch_scent_move_queued_total": float(floats[1]),
            "patch_waste_move_queued_total": float(floats[2]),
            "patch_scent_gather_queued_total": float(floats[3]),
            "patch_waste_gather_queued_total": float(floats[4]),
            "patch_movement_energy_extra": float(floats[5]),
            "patch_movement_energy_event_count": int(ints[0]),
            "native_tick_loop_count": int(ints[1]),
        }


class ResidentCudaOwnerActionState:
    """Stage 1A action owner using the accepted slot-indexed tensor contract."""

    _REQUIRED = (
        "agent_ids_by_alive_order",
        "agent_id_to_slot",
        "slot_to_agent_id",
        "alive_mask",
        "pos_y",
        "pos_x",
        "agent_grid",
        "food_grid",
        "energy",
        "health",
    )
    _PATCH = (
        "patch_food_amount",
        "patch_cover",
        "patch_shelter",
        "patch_scent",
        "patch_waste",
        "patch_scent_emit_pending",
        "patch_waste_emit_pending",
        "patch_movement_cost",
    )

    def __init__(
        self,
        tensors: Mapping[str, torch.Tensor],
        *,
        patch_enabled: bool,
        device: torch.device,
        imported_state_bytes: int,
    ) -> None:
        self.tensors = dict(tensors)
        self.patch_enabled = bool(patch_enabled)
        self.device = device
        self.imported_state_bytes = int(imported_state_bytes)
        self.generation_calls = 0
        self.action_payload_imports = 0
        self.action_payload_bytes = 0
        self.observation_calls = 0
        self._last_outputs: tuple[torch.Tensor, ...] | None = None

    @classmethod
    def from_projection(
        cls,
        projection: Mapping[str, Any],
        *,
        patch_enabled: bool,
        device: str | torch.device = "cuda",
    ) -> "ResidentCudaOwnerActionState":
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError("ResidentCudaOwnerActionState requires a CUDA device")
        source = dict(projection.get("tensors") or {})
        missing = [name for name in cls._REQUIRED if name not in source]
        if missing:
            raise ValueError(f"owner projection is missing required tensors: {missing}")
        grid = source["agent_grid"]
        if not hasattr(grid, "detach") or grid.dim() != 2:
            raise ValueError("Stage 1A requires one 2D world cell")
        tensors: dict[str, torch.Tensor] = {}
        for name in cls._REQUIRED:
            value = source[name]
            if not hasattr(value, "detach"):
                raise ValueError(f"owner projection tensor {name!r} is malformed")
            tensors[name] = value.detach().to(device=resolved).contiguous()
        for name in cls._PATCH:
            value = source.get(name)
            if value is None:
                fill = 1.0 if name == "patch_movement_cost" else 0.0
                value = torch.full_like(grid, fill, dtype=torch.float32)
            tensors[name] = value.detach().to(device=resolved, dtype=torch.float32).contiguous()
        capacity = int(tensors["energy"].size(0))
        for name in ("slot_to_agent_id", "alive_mask", "pos_y", "pos_x", "health"):
            if tensors[name].dim() != 1 or int(tensors[name].size(0)) != capacity:
                raise ValueError(f"owner projection slot tensor {name!r} has the wrong shape")
        if tensors["agent_id_to_slot"].dim() != 1:
            raise ValueError("agent_id_to_slot must be rank 1")
        imported_state_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in tensors.values()
        )
        return cls(
            tensors,
            patch_enabled=patch_enabled,
            device=resolved,
            imported_state_bytes=imported_state_bytes,
        )

    def _validate_counted_active_count(self, active_count: torch.Tensor) -> None:
        if not active_count.is_cuda or active_count.device != self.device:
            raise ValueError("active_count must be resident on the owner CUDA device")
        if active_count.dtype != torch.int64 or tuple(active_count.shape) != (1,):
            raise ValueError("active_count must be contiguous CUDA int64[1]")
        if not active_count.is_contiguous():
            raise ValueError("active_count must be contiguous CUDA int64[1]")

    def execute_scripted_generation(
        self,
        actions_by_tick_slot: np.ndarray | torch.Tensor,
        config: CudaActionConfig,
        *,
        build_directory: str | Path | None = None,
    ) -> dict[str, Any]:
        if isinstance(actions_by_tick_slot, torch.Tensor):
            actions = actions_by_tick_slot.detach().to(device=self.device, dtype=torch.int32).contiguous()
            action_bytes = int(actions.numel() * actions.element_size())
        else:
            action_array = np.ascontiguousarray(actions_by_tick_slot, dtype=np.int32)
            actions = torch.from_numpy(action_array).to(device=self.device).contiguous()
            action_bytes = int(action_array.nbytes)
        capacity = int(self.tensors["energy"].size(0))
        if actions.dim() != 2 or int(actions.size(1)) != capacity:
            raise ValueError("actions_by_tick_slot must be [ticks, slot_capacity]")
        mod = load_cuda_architecture_b_extension(build_directory=build_directory)
        outputs = mod.eiel_cuda_scripted_owner_action_generation_v1(
            self.tensors["agent_ids_by_alive_order"],
            self.tensors["agent_id_to_slot"],
            actions,
            self.tensors["pos_y"],
            self.tensors["pos_x"],
            self.tensors["agent_grid"],
            self.tensors["food_grid"],
            self.tensors["patch_food_amount"],
            self.tensors["patch_scent_emit_pending"],
            self.tensors["patch_waste_emit_pending"],
            self.tensors["patch_movement_cost"],
            self.tensors["energy"],
            self.tensors["health"],
            self.patch_enabled,
            config.move_cost,
            config.gather_cost,
            config.rest_cost,
            config.attack_cost,
            config.action_cost_mult,
            config.rest_cost_mult,
            config.max_energy,
            config.food_energy,
            config.patch_food_gather_amount,
            config.patch_food_presence_threshold,
            config.patch_scent_move_emit,
            config.patch_waste_move_emit,
            config.patch_scent_gather_emit,
            config.patch_waste_consume_emit,
            config.patch_movement_cost_apply_enable,
            config.patch_movement_cost_energy_scale,
        )
        self._last_outputs = tuple(outputs)
        self.generation_calls += 1
        self.action_payload_imports += 1
        self.action_payload_bytes += action_bytes
        return {
            "schema_version": "eiel_cuda_architecture_b_stage1_owner_action_result_v1",
            "decision": "ARCHB-CUDA-STAGE1A/OWNER-SLOT-ACTION-SLICE",
            "executed": True,
            "hidden_fallback": False,
            "accepted_owner_projection_layout": True,
            "native_generation_calls": self.generation_calls,
            "native_generation_kernel_launches": self.generation_calls,
            "python_tick_loops": 0,
            "ticks": int(actions.size(0)),
            "state_import_count": 1,
            "state_import_bytes": self.imported_state_bytes,
            "action_payload_import_count": self.action_payload_imports,
            "action_payload_import_bytes": self.action_payload_bytes,
            "full_stage1_qualified": False,
        }

    def execute_scripted_generation_counted(
        self,
        actions_by_tick_slot: np.ndarray | torch.Tensor,
        active_count: torch.Tensor,
        config: CudaActionConfig,
        *,
        build_directory: str | Path | None = None,
    ) -> dict[str, Any]:
        self._validate_counted_active_count(active_count)
        if isinstance(actions_by_tick_slot, torch.Tensor):
            actions = actions_by_tick_slot.detach().to(device=self.device, dtype=torch.int32).contiguous()
            action_bytes = int(actions.numel() * actions.element_size())
        else:
            action_array = np.ascontiguousarray(actions_by_tick_slot, dtype=np.int32)
            actions = torch.from_numpy(action_array).to(device=self.device).contiguous()
            action_bytes = int(action_array.nbytes)
        capacity = int(self.tensors["energy"].size(0))
        if actions.dim() != 2 or int(actions.size(1)) != capacity:
            raise ValueError("actions_by_tick_slot must be [ticks, slot_capacity]")
        mod = load_cuda_architecture_b_extension(build_directory=build_directory)
        outputs = mod.eiel_cuda_scripted_owner_action_generation_counted_v1(
            self.tensors["agent_ids_by_alive_order"],
            active_count,
            self.tensors["agent_id_to_slot"],
            actions,
            self.tensors["pos_y"],
            self.tensors["pos_x"],
            self.tensors["agent_grid"],
            self.tensors["food_grid"],
            self.tensors["patch_food_amount"],
            self.tensors["patch_scent_emit_pending"],
            self.tensors["patch_waste_emit_pending"],
            self.tensors["patch_movement_cost"],
            self.tensors["energy"],
            self.tensors["health"],
            self.patch_enabled,
            config.move_cost,
            config.gather_cost,
            config.rest_cost,
            config.attack_cost,
            config.action_cost_mult,
            config.rest_cost_mult,
            config.max_energy,
            config.food_energy,
            config.patch_food_gather_amount,
            config.patch_food_presence_threshold,
            config.patch_scent_move_emit,
            config.patch_waste_move_emit,
            config.patch_scent_gather_emit,
            config.patch_waste_consume_emit,
            config.patch_movement_cost_apply_enable,
            config.patch_movement_cost_energy_scale,
        )
        self._last_outputs = tuple(outputs)
        self.generation_calls += 1
        self.action_payload_imports += 1
        self.action_payload_bytes += action_bytes
        return {
            "schema_version": "eiel_cuda_architecture_b_counted_owner_action_result_v1",
            "decision": "ARCHB-CUDA-STAGE1A/COUNTED-OWNER-SLOT-ACTION-SLICE",
            "executed": True,
            "hidden_fallback": False,
            "accepted_owner_projection_layout": True,
            "fixed_capacity_outputs": True,
            "device_resident_active_count": True,
            "native_generation_calls": self.generation_calls,
            "native_generation_kernel_launches": self.generation_calls,
            "python_tick_loops": 0,
            "ticks": int(actions.size(0)),
            "state_import_count": 1,
            "state_import_bytes": self.imported_state_bytes,
            "action_payload_import_count": self.action_payload_imports,
            "action_payload_import_bytes": self.action_payload_bytes,
            "full_stage1_qualified": False,
        }

    def build_observation_frame(
        self,
        config: CudaObservationConfig,
        *,
        build_directory: str | Path | None = None,
    ) -> torch.Tensor:
        mod = load_cuda_architecture_b_extension(build_directory=build_directory)
        observation = mod.eiel_cuda_owner_observation_frame_v1(
            self.tensors["agent_ids_by_alive_order"],
            self.tensors["agent_id_to_slot"],
            self.tensors["pos_y"],
            self.tensors["pos_x"],
            self.tensors["agent_grid"],
            self.tensors["food_grid"],
            self.tensors["energy"],
            self.tensors["health"],
            self.tensors["patch_food_amount"],
            self.tensors["patch_cover"],
            self.tensors["patch_shelter"],
            self.tensors["patch_scent"],
            self.tensors["patch_waste"],
            self.tensors["patch_movement_cost"],
            self.patch_enabled,
            config.view_radius,
            config.max_energy,
            config.start_health,
            config.patch_food_amount_max,
            config.patch_scent_max,
            config.patch_waste_max,
            config.patch_density_radius,
            config.patch_density_include_self,
            config.patch_density_normalize,
            config.patch_movement_cost_min,
            config.patch_movement_cost_max,
        )
        self.observation_calls += 1
        return observation

    def build_observation_frame_counted(
        self,
        active_count: torch.Tensor,
        config: CudaObservationConfig,
        *,
        build_directory: str | Path | None = None,
    ) -> torch.Tensor:
        self._validate_counted_active_count(active_count)
        mod = load_cuda_architecture_b_extension(build_directory=build_directory)
        observation = mod.eiel_cuda_owner_observation_frame_counted_v1(
            self.tensors["agent_ids_by_alive_order"],
            active_count,
            self.tensors["agent_id_to_slot"],
            self.tensors["pos_y"],
            self.tensors["pos_x"],
            self.tensors["agent_grid"],
            self.tensors["food_grid"],
            self.tensors["energy"],
            self.tensors["health"],
            self.tensors["patch_food_amount"],
            self.tensors["patch_cover"],
            self.tensors["patch_shelter"],
            self.tensors["patch_scent"],
            self.tensors["patch_waste"],
            self.tensors["patch_movement_cost"],
            self.patch_enabled,
            config.view_radius,
            config.max_energy,
            config.start_health,
            config.patch_food_amount_max,
            config.patch_scent_max,
            config.patch_waste_max,
            config.patch_density_radius,
            config.patch_density_include_self,
            config.patch_density_normalize,
            config.patch_movement_cost_min,
            config.patch_movement_cost_max,
        )
        self.observation_calls += 1
        return observation

    def export_boundary(self) -> dict[str, Any]:
        if self._last_outputs is None:
            raise RuntimeError("no scripted owner generation has executed")
        energy, died, food_gained, moved_count, metrics_float, metrics_int, slots = self._last_outputs
        torch.cuda.synchronize(self.device)
        return {
            "tensors": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.tensors.items()
            },
            "alive_slots_by_alive_order": slots.detach().cpu().clone(),
            "energy": energy.detach().cpu().clone(),
            "died_mask_by_alive_order": died.detach().cpu().clone(),
            "food_gained_by_alive_order": food_gained.detach().cpu().clone(),
            "moved_count_by_alive_order": moved_count.detach().cpu().clone(),
            "metrics_float": metrics_float.detach().cpu().clone(),
            "metrics_int": metrics_int.detach().cpu().clone(),
        }


__all__ = [
    "CUDA_STAGE1_ACTION_CAPABILITIES",
    "CudaActionConfig",
    "CudaObservationConfig",
    "ResidentCudaActionState",
    "ResidentCudaOwnerActionState",
    "load_cuda_architecture_b_extension",
]
