"""Tensor World C2 backend: resident Patch/Jungle/WDL/EED expansion."""

from __future__ import annotations

from dataclasses import dataclass, fields
import secrets
from typing import Any, Mapping

import torch

from .c1_backend import clone_bundle_to_device_v1
from .c2_native_loader import load_tensor_world_c2_extension
from .c2_state import TensorWorldStateBundleV2
from .contracts import (
    BackendCapabilitiesV1,
    BackendRunStateV1,
    TensorWorldContractError,
    TickReceiptV1,
    TickRequestV1,
)


@dataclass(frozen=True, slots=True)
class TensorWorldC2ConfigV1:
    max_energy: float = 160.0
    start_health: float = 12.0
    scent_decay_factor: float = 0.95
    waste_decay_factor: float = 0.98
    scent_presence_emit: float = 0.01
    waste_presence_emit: float = 0.005
    waste_density_emit_scale: float = 0.0
    scent_max: float = 1.0
    waste_max: float = 1.0
    food_max: float = 1.0
    food_presence_threshold: float = 0.5
    density_radius: int = 1
    density_include_self: bool = True
    density_normalize: bool = False
    food_regrow_use_fertility: bool = True
    food_regrow_cap_by_fertility: bool = False
    waste_consequence_enable: bool = True
    waste_penalty_enable: bool = True
    waste_penalty_scale: float = 0.5
    waste_penalty_max: float = 0.5
    density_penalty_enable: bool = True
    density_penalty_scale: float = 0.5
    density_penalty_max: float = 0.5
    regrowth_min_multiplier: float = 0.1
    recovery_enabled: bool = True
    recovery_activity_min: float = 0.0
    recovery_activity_max: float = 1.0
    fertility_min: float = 0.0
    fertility_max: float = 1.0
    recovery_decay_factor: float = 0.98
    recovery_base_growth: float = 0.01
    recovery_cover_weight: float = 0.02
    recovery_fertility_weight: float = 0.02
    recovery_waste_penalty_weight: float = 0.05
    recovery_density_penalty_weight: float = 0.05
    recovery_scent_penalty_weight: float = 0.02
    recovery_boost_enabled: bool = True
    recovery_boost_scale: float = 0.1
    recovery_boost_max: float = 0.25
    jungle_enabled: bool = True
    jungle_base_risk: float = 0.02
    jungle_scent_weight: float = 0.01
    jungle_waste_weight: float = 0.01
    jungle_density_weight: float = 0.01
    jungle_visibility_weight: float = 0.01
    jungle_cover_weight: float = 0.02
    jungle_shelter_weight: float = 0.02
    jungle_risk_min: float = 0.0
    jungle_risk_cap: float = 0.05
    jungle_damage: float = 0.02
    jungle_juvenile_enabled: bool = True
    jungle_juvenile_age: int = 10
    jungle_juvenile_multiplier: float = 1.5
    jungle_terrain_enabled: bool = True
    jungle_cover_damage_weight: float = 0.25
    jungle_shelter_damage_weight: float = 0.25
    jungle_terrain_min: float = 0.25
    wdl_delay_steps: int = 12
    wdl_cue_steps: int = 1
    wdl_active_steps: int = 1
    wdl_event_period_steps: int = 20
    wdl_radius: int = 1
    wdl_intensity: float = 0.1
    wdl_dropout_rate: float = 0.0
    eed_impact_applied: bool = True
    eed_ecology_frozen: bool = False
    eed_equalized_sham: bool = False
    eed_impact_decay: float = 0.25
    eed_max_load: float = 1.0
    eed_impact_per_success: float = 0.25
    eed_gather_amount: float = 1.0
    eed_handling_threshold_a: float = 1.5
    eed_handling_threshold_b: float = 1.75
    eed_return_a: float = 0.75
    eed_return_b: float = 1.5
    move_cost: float = 0.2
    gather_cost: float = 0.08
    rest_cost: float = 0.05
    attack_cost: float = 0.6
    action_cost_mult: float = 1.0
    rest_cost_mult: float = 1.0
    movement_cost_energy_scale: float = 1.0
    scent_move_emit: float = 0.02
    waste_move_emit: float = 0.0
    scent_gather_emit: float = 0.05
    waste_gather_emit: float = 0.05
    w_hunger_gather: float = 1.2
    w_pain_rest: float = 0.8
    w_fatigue_rest: float = 1.0
    w_pain_attack: float = 0.8
    w_fear_attack: float = 1.0

    def __post_init__(self) -> None:
        if self.wdl_event_period_steps < self.wdl_delay_steps + self.wdl_active_steps + 1:
            raise TensorWorldContractError("C2 WDL event period violates the temporal contract")
        if self.wdl_cue_steps < 1 or self.wdl_cue_steps > self.wdl_delay_steps:
            raise TensorWorldContractError("C2 WDL cue duration is invalid")
        if self.max_energy <= 0 or self.start_health <= 0:
            raise TensorWorldContractError("C2 normalization maxima must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TensorWorldC2ConfigV1":
        known = {field.name for field in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise TensorWorldContractError(f"unknown C2 configuration fields: {sorted(unknown)}")
        return cls(**value)


def _copy_owner(owner: Any, device: torch.device) -> Any:
    return type(owner)(*(getattr(owner, f.name).detach().to(device=device, copy=True).contiguous() for f in fields(owner)))


def clone_c2_bundle_to_device_v1(source: TensorWorldStateBundleV2, device: torch.device) -> TensorWorldStateBundleV2:
    source.validate()
    result = TensorWorldStateBundleV2(
        core=clone_bundle_to_device_v1(source.core, device),
        patch=_copy_owner(source.patch, device), wdl=_copy_owner(source.wdl, device),
        eed=_copy_owner(source.eed, device), policy=_copy_owner(source.policy, device),
        rng=_copy_owner(source.rng, device), effects=_copy_owner(source.effects, device),
        world_events=_copy_owner(source.world_events, device),
        transactions=_copy_owner(source.transactions, device),
        hot_metrics=_copy_owner(source.hot_metrics, device),
    )
    result.validate()
    return result


def _native_state_v1(s: TensorWorldStateBundleV2) -> list[torch.Tensor]:
    c, p, w, e = s.core, s.patch, s.wdl, s.eed
    return [
        c.agents.alive, c.agents.agent_id, c.agents.pos_y, c.agents.pos_x,
        c.agents.energy, c.agents.health, c.agents.age, c.agents.survival_steps,
        c.agents.energy_collected, c.occupancy.agent_id_grid, c.occupancy.slot_grid,
        c.transactions.actions, c.transactions.moved, c.transactions.gathered,
        c.transactions.food_gained, c.hot_events.moves_attempted,
        c.hot_events.moves_committed, c.hot_events.gathers_committed,
        c.rng.action_uniforms, c.rng.action_cursor, c.world.cover, c.world.shelter,
        c.world.movement_cost, c.world.food_amount, c.world.scent, c.world.waste,
        c.world.density, p.fertility, p.scent_emit_pending, p.waste_emit_pending,
        p.recovery_activity, p.visibility, p.agent_grid_i32, p.food_grid_i32,
        p.multiplier_scratch, p.waste_penalty_scratch, p.density_penalty_scratch,
        p.sorted_multiplier_scratch, p.recovery_boost_scratch,
        p.sorted_recovery_boost_scratch, w.cue, w.timer, w.active, w.intensity,
        w.age, w.event_id, w.prior_cue, w.base_seed, w.next_absolute_step,
        e.resource_a_amount, e.resource_b_amount, e.resource_a_capacity,
        e.resource_b_capacity, e.resource_a_renewal, e.resource_b_renewal,
        e.resource_a_pending_impact, e.resource_b_pending_impact,
        e.resource_a_impact_load, e.resource_b_impact_load, e.resource_class,
        e.active_y, e.active_x, e.active_class, e.handling_progress, e.handling_axis,
        s.policy.packed_weights, s.rng.jungle_uniforms,
        s.effects.pre_policy_health_delta, s.effects.post_action_health_delta,
        s.effects.hazard_flags, s.world_events.scent_emission,
        s.world_events.waste_emission, s.world_events.gather_amount,
        s.world_events.gather_resource_class, s.transactions.observations,
        s.transactions.logits, s.hot_metrics.patch_stats,
        s.hot_metrics.recovery_stats, s.hot_metrics.jungle_integers,
        s.hot_metrics.jungle_doubles, s.hot_metrics.wdl_integers,
        s.hot_metrics.wdl_delay_sum, s.hot_metrics.wdl_action_integers,
        s.hot_metrics.eed_receipt, c.policy.drives, c.policy.attack_bias,
        c.policy.action_temperature, c.policy.drive_logit_gain,
    ]


def _config_tensors_v1(config: TensorWorldC2ConfigV1, state: TensorWorldStateBundleV2) -> tuple[torch.Tensor, torch.Tensor]:
    ints = [
        state.grid_size, state.capacity, config.wdl_delay_steps, config.wdl_cue_steps,
        config.wdl_active_steps, config.wdl_event_period_steps, config.wdl_radius,
        config.density_radius, config.density_include_self, config.density_normalize,
        config.food_regrow_use_fertility, config.food_regrow_cap_by_fertility,
        config.waste_consequence_enable, config.waste_penalty_enable,
        config.density_penalty_enable, config.recovery_enabled,
        config.recovery_boost_enabled, config.jungle_enabled,
        config.jungle_juvenile_enabled, config.jungle_terrain_enabled,
        config.eed_impact_applied, config.eed_ecology_frozen, config.eed_equalized_sham,
    ]
    floats = [
        config.max_energy, config.start_health, config.scent_decay_factor,
        config.waste_decay_factor, config.scent_presence_emit, config.waste_presence_emit,
        config.waste_density_emit_scale, config.scent_max, config.waste_max,
        config.food_max, 0.0, config.food_presence_threshold,
        config.waste_penalty_scale, config.waste_penalty_max,
        config.density_penalty_scale, config.density_penalty_max,
        config.regrowth_min_multiplier, config.recovery_activity_min,
        config.recovery_activity_max, config.fertility_min, config.fertility_max,
        config.recovery_decay_factor, config.recovery_base_growth,
        config.recovery_cover_weight, config.recovery_fertility_weight,
        config.recovery_waste_penalty_weight, config.recovery_density_penalty_weight,
        config.recovery_scent_penalty_weight, config.recovery_boost_scale,
        config.recovery_boost_max, config.jungle_base_risk,
        config.jungle_scent_weight, config.jungle_waste_weight,
        config.jungle_density_weight, config.jungle_visibility_weight,
        config.jungle_cover_weight, config.jungle_shelter_weight,
        config.jungle_risk_min, config.jungle_risk_cap, config.jungle_damage,
        float(config.jungle_juvenile_age), config.jungle_juvenile_multiplier,
        config.jungle_cover_damage_weight, config.jungle_shelter_damage_weight,
        config.jungle_terrain_min, config.wdl_intensity, config.wdl_dropout_rate,
        config.eed_impact_decay, config.eed_max_load, config.eed_impact_per_success,
        config.eed_gather_amount, config.eed_handling_threshold_a,
        config.eed_handling_threshold_b, config.eed_return_a, config.eed_return_b,
        config.move_cost, config.gather_cost, config.rest_cost, config.attack_cost,
        config.action_cost_mult, config.rest_cost_mult, config.movement_cost_energy_scale,
        config.scent_move_emit, config.waste_move_emit, config.scent_gather_emit,
        config.waste_gather_emit, config.w_hunger_gather, config.w_pain_rest,
        config.w_fatigue_rest, config.w_pain_attack, config.w_fear_attack,
    ]
    return (
        torch.tensor(ints, dtype=torch.int64, device=state.device),
        torch.tensor(floats, dtype=torch.float64, device=state.device),
    )


class TensorWorldC2BackendV1:
    _capabilities = BackendCapabilitiesV1(
        backend_id="tensor_world_c2_v1",
        mechanics=("patch", "jungle", "wdl", "eed", "observation", "fixed_mlp", "movement", "gather", "energy"),
        batch_sizes=(1, 4, 48), grid_sizes=(32,), initial_populations=(256,),
        slot_capacities=(512,), policy_families=("fixed_mlp",),
        observation_abi="patch_wdl_radius1_observation_9x9_plus2_v1",
        policy_abi="fixed_mlp_83x64x64x7_fp32_pack_v1",
        determinism="fixture_trajectory_exact_c2", rng_mode="device_resident_fixture_injection_c2",
        residency_mode="gpu_resident_static_mechanics_cuda_graph_v1",
        checkpoint_replay=False, status="experimental",
    )

    def __init__(self, *, device: str | torch.device = "cuda:0", extension_path: str | None = None) -> None:
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise TensorWorldContractError("Tensor World C2 is a CUDA-only backend")
        self._native = load_tensor_world_c2_extension(extension_path)
        self._runs: dict[str, dict[str, Any]] = {}

    @property
    def capabilities(self) -> BackendCapabilitiesV1:
        return self._capabilities

    def initialize(self, canonical_state: Mapping[str, Any]) -> BackendRunStateV1:
        source = canonical_state.get("bundle")
        if not isinstance(source, TensorWorldStateBundleV2):
            raise TensorWorldContractError("canonical C2 state requires TensorWorldStateBundleV2")
        config = TensorWorldC2ConfigV1.from_mapping(canonical_state.get("config", {}))
        fixture_mode = canonical_state.get("fixture_mode", False)
        with torch.cuda.device(self._device):
            state = clone_c2_bundle_to_device_v1(source, self._device)
            active = state.core.agents.alive.sum(dim=1)
            if not bool(torch.all(active == active[0]).item()):
                raise TensorWorldContractError("C2 requires equal active counts across worlds")
            shape = (state.batch, state.grid_size, int(active[0]), state.capacity)
            admitted = shape in {(1, 32, 256, 512), (4, 32, 256, 512), (48, 32, 256, 512)}
            if not admitted and (not fixture_mode or shape not in {(1, 8, 4, 8), (4, 8, 4, 8)}):
                raise TensorWorldContractError("C2 shape is outside the fail-closed capability")
            native_state = _native_state_v1(state)
            ci, cf = _config_tensors_v1(config, state)
            mutable = [native_state[i] for i in (
                *range(2, 20), 23, 24, 25, 26, *range(28, 64), *range(67, 84)
            )]
            snapshot = tuple(t.clone() for t in mutable)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(self._device)
            self._execute(native_state, ci, cf)
            torch.cuda.synchronize(self._device)
            with torch.cuda.graph(graph):
                self._execute(native_state, ci, cf)
            torch.cuda.synchronize(self._device)
            for destination, saved in zip(mutable, snapshot, strict=True):
                destination.copy_(saved)
        token = secrets.token_hex(16)
        self._runs[token] = {"bundle": state, "config": config, "graph": graph,
            "native_state": native_state, "ci": ci, "cf": cf,
            "steady_state_counters": {name: 0 for name in (
                "host_to_device_bytes", "device_to_host_bytes", "host_callback_count",
                "synchronization_count", "materialization_count", "cuda_allocator_call_count",
                "python_per_world_loop_count", "python_per_agent_loop_count")}}
        return BackendRunStateV1(self._capabilities.backend_id, token)

    def _execute(self, state: list[torch.Tensor], ci: torch.Tensor, cf: torch.Tensor) -> None:
        self._native.wdl_step(state, ci, cf)
        self._native.eed_step(state, ci, cf)
        self._native.patch_step(state, ci, cf)
        self._native.jungle_step(state, ci, cf)
        self._native.observe(state, ci, cf)
        self._native.policy_logits(state, ci, cf)
        self._native.select_actions(state, ci, cf)
        self._native.settle(state, ci, cf)

    def _owned(self, state: BackendRunStateV1) -> dict[str, Any]:
        if state.closed or state.backend_id != self._capabilities.backend_id or state.token not in self._runs:
            raise TensorWorldContractError("Tensor World C2 run handle is invalid or closed")
        return self._runs[state.token]

    def advance(self, state: BackendRunStateV1, request: TickRequestV1) -> TickReceiptV1:
        if not isinstance(request, TickRequestV1):
            raise TensorWorldContractError("C2 advance requires TickRequestV1")
        owned = self._owned(state)
        before = (torch.cuda.memory_allocated(self._device), torch.cuda.memory_reserved(self._device))
        owned["graph"].replay()
        after = (torch.cuda.memory_allocated(self._device), torch.cuda.memory_reserved(self._device))
        if after != before:
            raise RuntimeError("C2 CUDA Graph replay changed allocator state")
        owned["bundle"].core.tick += 1
        return TickReceiptV1(self._capabilities.backend_id, 1, 0, 0, 0, 0, 0)

    def export_state(self, state: BackendRunStateV1) -> Mapping[str, Any]:
        owned = self._owned(state)
        torch.cuda.synchronize(self._device)
        return {"bundle": clone_c2_bundle_to_device_v1(owned["bundle"], torch.device("cpu")),
                "config": owned["config"]}

    def residency_counters(self, state: BackendRunStateV1) -> Mapping[str, int]:
        return dict(self._owned(state)["steady_state_counters"])

    def close(self, state: BackendRunStateV1) -> None:
        if not state.closed:
            self._runs.pop(state.token, None)
            state.closed = True
