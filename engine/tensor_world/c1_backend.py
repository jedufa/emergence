"""Production Tensor World C1 backend: resident Patch/fixed-MLP vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, fields
import secrets
from typing import Any, Mapping

import torch

from .contracts import (
    BackendCapabilitiesV1,
    BackendRunStateV1,
    TensorWorldContractError,
    TickReceiptV1,
    TickRequestV1,
)
from .native_loader import load_tensor_world_c1_extension
from .state import (
    AgentTensorStateV1,
    HotEventAccumulatorStateV1,
    IdentityTensorStateV1,
    OccupancyTensorStateV1,
    PolicyTensorStateV1,
    RngTensorStateV1,
    TensorWorldStateBundleV1,
    TransactionTensorStateV1,
    WorldTensorStateV1,
)


@dataclass(frozen=True, slots=True)
class TensorWorldC1ConfigV1:
    max_energy: float = 160.0
    start_health: float = 12.0
    food_amount_max: float = 1.0
    scent_max: float = 1.0
    waste_max: float = 1.0
    density_radius: int = 1
    density_include_self: bool = True
    density_normalize: bool = False
    movement_cost_min: float = 0.0
    movement_cost_max: float = 4.0
    move_cost: float = 0.20
    gather_cost: float = 0.08
    rest_cost: float = 0.05
    attack_cost: float = 0.60
    action_cost_mult: float = 1.0
    rest_cost_mult: float = 1.0
    food_energy: float = 20.0
    gather_amount: float = 1.0
    food_presence_threshold: float = 0.5
    movement_cost_apply: bool = True
    movement_cost_energy_scale: float = 1.0
    w_hunger_gather: float = 1.2
    w_pain_rest: float = 0.8
    w_fatigue_rest: float = 1.0
    w_pain_attack: float = 0.8
    w_fear_attack: float = 1.0

    def __post_init__(self) -> None:
        if self.max_energy <= 0 or self.start_health <= 0:
            raise TensorWorldContractError(
                "C1 energy and health normalization maxima must be positive"
            )
        if self.food_amount_max <= 0 or self.scent_max <= 0 or self.waste_max <= 0:
            raise TensorWorldContractError(
                "C1 Patch normalization maxima must be positive"
            )
        if self.gather_amount <= 0 or self.density_radius < 0:
            raise TensorWorldContractError(
                "C1 gather amount and density radius are invalid"
            )
        if self.movement_cost_max <= self.movement_cost_min:
            raise TensorWorldContractError(
                "C1 movement-cost normalization range is invalid"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TensorWorldC1ConfigV1:
        known = {field.name for field in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise TensorWorldContractError(
                f"unknown C1 configuration fields: {sorted(unknown)}"
            )
        return cls(**value)


def _to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.detach().to(device=device, copy=True).contiguous()


def clone_bundle_to_device_v1(
    source: TensorWorldStateBundleV1, device: torch.device
) -> TensorWorldStateBundleV1:
    """Deterministic one-time state ingestion; never used in a steady tick."""

    source.validate()
    bundle = TensorWorldStateBundleV1(
        world=WorldTensorStateV1(
            *(
                _to_device(getattr(source.world, field.name), device)
                for field in fields(WorldTensorStateV1)
            )
        ),
        agents=AgentTensorStateV1(
            *(
                _to_device(getattr(source.agents, field.name), device)
                for field in fields(AgentTensorStateV1)
            )
        ),
        occupancy=OccupancyTensorStateV1(
            *(
                _to_device(getattr(source.occupancy, field.name), device)
                for field in fields(OccupancyTensorStateV1)
            )
        ),
        policy=PolicyTensorStateV1(
            *(
                _to_device(getattr(source.policy, field.name), device)
                for field in fields(PolicyTensorStateV1)
            )
        ),
        identity=IdentityTensorStateV1(
            *(
                _to_device(getattr(source.identity, field.name), device)
                for field in fields(IdentityTensorStateV1)
            )
        ),
        rng=RngTensorStateV1(
            *(
                _to_device(getattr(source.rng, field.name), device)
                for field in fields(RngTensorStateV1)
            )
        ),
        transactions=TransactionTensorStateV1(
            *(
                _to_device(getattr(source.transactions, field.name), device)
                for field in fields(TransactionTensorStateV1)
            )
        ),
        hot_events=HotEventAccumulatorStateV1(
            *(
                _to_device(getattr(source.hot_events, field.name), device)
                for field in fields(HotEventAccumulatorStateV1)
            )
        ),
        tick=source.tick,
    )
    bundle.validate()
    # C1 deliberately admits a fixed active prefix in canonical agent-id order.
    for world in range(bundle.batch):
        active_ids = bundle.agents.agent_id[world, bundle.agents.alive[world]].cpu()
        if active_ids.numel() and not bool(torch.all(active_ids[1:] > active_ids[:-1])):
            raise TensorWorldContractError(
                "C1 active rows must be strictly ascending by agent_id"
            )
        alive_cpu = bundle.agents.alive[world].cpu()
        count = int(alive_cpu.sum())
        if not bool(alive_cpu[:count].all()) or bool(alive_cpu[count:].any()):
            raise TensorWorldContractError(
                "C1 requires one fixed active prefix per world"
            )
    return bundle


class TensorWorldC1BackendV1:
    """One-tick CUDA Graph backend with no steady-state host dependency."""

    _capabilities = BackendCapabilitiesV1(
        backend_id="tensor_world_c1_v1",
        mechanics=(
            "observation",
            "fixed_mlp",
            "action_selection",
            "movement",
            "occupancy_conflict",
            "gather",
            "energy",
        ),
        batch_sizes=(1, 4, 48),
        grid_sizes=(32,),
        initial_populations=(256,),
        slot_capacities=(512,),
        policy_families=("fixed_mlp",),
        observation_abi="patch_radius1_observation_8x9_plus2_v1",
        policy_abi="fixed_mlp_74x64x64x7_fp32_pack_v1",
        determinism="fixture_trajectory_exact_c1",
        rng_mode="device_resident_fixture_injection_c1",
        residency_mode="gpu_resident_cross_world_cuda_graph_v1",
        checkpoint_replay=False,
        status="experimental",
    )

    def __init__(
        self,
        *,
        device: str | torch.device = "cuda:0",
        extension_path: str | None = None,
    ) -> None:
        self._device = torch.device(device)
        if self._device.type != "cuda":
            raise TensorWorldContractError("Tensor World C1 is a CUDA-only backend")
        self._native = load_tensor_world_c1_extension(extension_path)
        self._runs: dict[str, dict[str, Any]] = {}

    @property
    def capabilities(self) -> BackendCapabilitiesV1:
        return self._capabilities

    def initialize(self, canonical_state: Mapping[str, Any]) -> BackendRunStateV1:
        source = canonical_state.get("bundle")
        if not isinstance(source, TensorWorldStateBundleV1):
            raise TensorWorldContractError(
                "canonical state requires a TensorWorldStateBundleV1 under 'bundle'"
            )
        config_value = canonical_state.get("config", {})
        if not isinstance(config_value, Mapping):
            raise TensorWorldContractError("canonical C1 config must be a mapping")
        config = TensorWorldC1ConfigV1.from_mapping(config_value)
        with torch.cuda.device(self._device):
            bundle = clone_bundle_to_device_v1(source, self._device)
            active_counts = bundle.agents.alive.sum(dim=1)
            if not bool(torch.all(active_counts == active_counts[0]).item()):
                raise TensorWorldContractError(
                    "C1 cross-world policy batching requires equal active counts"
                )
            active = int(active_counts[0].item())
            if active <= 0:
                raise TensorWorldContractError("C1 requires at least one active agent")
            shape = (bundle.batch, bundle.grid_size, active, bundle.capacity)
            fixture_mode = canonical_state.get("fixture_mode", False)
            if type(fixture_mode) is not bool:
                raise TensorWorldContractError("C1 fixture_mode must be boolean")
            if not self._is_admitted_shape(*shape):
                fixture_shapes = {(1, 8, 4, 8), (4, 8, 4, 8)}
                if not fixture_mode or shape not in fixture_shapes:
                    raise TensorWorldContractError(
                        "C1 shape is outside the fail-closed production capability"
                    )
            policy = self._prepare_policy(bundle, active, config)
            mutable_snapshot = self._snapshot_mutable_state(bundle)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(self._device)
            self._execute_pipeline(bundle, policy, config)
            torch.cuda.synchronize(self._device)
            with torch.cuda.graph(graph):
                self._execute_pipeline(bundle, policy, config)
            torch.cuda.synchronize(self._device)
            self._restore_mutable_state(bundle, mutable_snapshot)
        token = secrets.token_hex(16)
        self._runs[token] = {
            "bundle": bundle,
            "config": config,
            "graph": graph,
            "policy": policy,
            "steady_state_counters": {
                "host_to_device_bytes": 0,
                "device_to_host_bytes": 0,
                "host_callback_count": 0,
                "synchronization_count": 0,
                "materialization_count": 0,
                "cuda_allocator_call_count": 0,
                "python_per_world_loop_count": 0,
                "python_per_agent_loop_count": 0,
            },
        }
        return BackendRunStateV1(self._capabilities.backend_id, token)

    @classmethod
    def _is_admitted_shape(
        cls, batch: int, grid_size: int, active: int, capacity: int
    ) -> bool:
        return cls._capabilities.supports(
            batch=batch,
            grid_size=grid_size,
            initial_population=active,
            slot_capacity=capacity,
            policy_family="fixed_mlp",
            mechanics=cls._capabilities.mechanics,
        )

    @staticmethod
    def _snapshot_mutable_state(
        bundle: TensorWorldStateBundleV1,
    ) -> tuple[torch.Tensor, ...]:
        return tuple(
            tensor.clone()
            for tensor in (
                bundle.world.food_amount,
                bundle.agents.pos_y,
                bundle.agents.pos_x,
                bundle.agents.energy,
                bundle.agents.energy_collected,
                bundle.occupancy.agent_id_grid,
                bundle.occupancy.slot_grid,
                bundle.rng.action_cursor,
            )
        )

    @staticmethod
    def _restore_mutable_state(
        bundle: TensorWorldStateBundleV1, snapshot: tuple[torch.Tensor, ...]
    ) -> None:
        for destination, source in zip(
            (
                bundle.world.food_amount,
                bundle.agents.pos_y,
                bundle.agents.pos_x,
                bundle.agents.energy,
                bundle.agents.energy_collected,
                bundle.occupancy.agent_id_grid,
                bundle.occupancy.slot_grid,
                bundle.rng.action_cursor,
            ),
            snapshot,
            strict=True,
        ):
            destination.copy_(source)

    def _prepare_policy(
        self,
        bundle: TensorWorldStateBundleV1,
        active: int,
        config: TensorWorldC1ConfigV1,
    ) -> dict[str, torch.Tensor | int]:
        batch = bundle.batch
        pack = (
            bundle.policy.packed_weights[:, :active]
            .contiguous()
            .reshape(batch * active, 9415)
        )
        return {
            "active": active,
            "pack": pack,
            "logits": torch.empty((batch * active, 7), device=bundle.device),
            "drives": bundle.policy.drives[:, :active]
            .contiguous()
            .reshape(batch * active, 4),
            "attack": bundle.policy.attack_bias[:, :active]
            .contiguous()
            .reshape(batch * active),
            "temperature": bundle.policy.action_temperature[:, :active]
            .contiguous()
            .reshape(batch * active),
            "gain": bundle.policy.drive_logit_gain[:, :active]
            .contiguous()
            .reshape(batch * active),
        }

    def _execute_pipeline(
        self,
        bundle: TensorWorldStateBundleV1,
        policy: Mapping[str, Any],
        config: TensorWorldC1ConfigV1,
    ) -> None:
        self._native.observe(
            bundle.transactions.observations,
            bundle.agents.alive,
            bundle.agents.pos_y,
            bundle.agents.pos_x,
            bundle.occupancy.agent_id_grid,
            bundle.agents.energy,
            bundle.agents.health,
            bundle.world.food_amount,
            bundle.world.cover,
            bundle.world.shelter,
            bundle.world.scent,
            bundle.world.waste,
            bundle.world.movement_cost,
            config.max_energy,
            config.start_health,
            config.food_amount_max,
            config.food_presence_threshold,
            config.scent_max,
            config.waste_max,
            config.density_radius,
            config.density_include_self,
            config.density_normalize,
            config.movement_cost_min,
            config.movement_cost_max,
        )
        active = int(policy["active"])
        pack = policy["pack"]
        logits = policy["logits"]
        drives = policy["drives"]
        attack = policy["attack"]
        temperature = policy["temperature"]
        gain = policy["gain"]
        assert all(
            isinstance(value, torch.Tensor)
            for value in (pack, logits, drives, attack, temperature, gain)
        )
        rows = bundle.batch * active
        x = bundle.transactions.observations[:, :active].reshape(rows, 74)
        self._native.policy_logits(
            logits,
            pack,
            x,
            drives,
            attack,
            temperature,
            gain,
            config.w_hunger_gather,
            config.w_pain_rest,
            config.w_fatigue_rest,
            config.w_pain_attack,
            config.w_fear_attack,
        )
        bundle.transactions.logits[:, :active].copy_(
            logits.view(bundle.batch, active, 7)
        )
        self._native.select_actions(
            bundle.transactions.actions,
            bundle.transactions.logits,
            bundle.rng.action_uniforms,
            bundle.agents.alive,
            bundle.rng.action_cursor,
        )
        self._native.settle(
            bundle.transactions.actions,
            bundle.agents.alive,
            bundle.agents.agent_id,
            bundle.agents.pos_y,
            bundle.agents.pos_x,
            bundle.occupancy.agent_id_grid,
            bundle.occupancy.slot_grid,
            bundle.world.food_amount,
            bundle.world.movement_cost,
            bundle.agents.energy,
            bundle.agents.energy_collected,
            bundle.transactions.moved,
            bundle.transactions.gathered,
            bundle.transactions.food_gained,
            bundle.hot_events.moves_attempted,
            bundle.hot_events.moves_committed,
            bundle.hot_events.gathers_committed,
            config.move_cost,
            config.gather_cost,
            config.rest_cost,
            config.attack_cost,
            config.action_cost_mult,
            config.rest_cost_mult,
            config.max_energy,
            config.food_energy,
            config.gather_amount,
            config.food_presence_threshold,
            config.movement_cost_apply,
            config.movement_cost_energy_scale,
        )

    def _owned(self, state: BackendRunStateV1) -> dict[str, Any]:
        if (
            state.closed
            or state.backend_id != self._capabilities.backend_id
            or state.token not in self._runs
        ):
            raise TensorWorldContractError(
                "Tensor World C1 run handle is invalid or closed"
            )
        return self._runs[state.token]

    def advance(
        self, state: BackendRunStateV1, request: TickRequestV1
    ) -> TickReceiptV1:
        if not isinstance(request, TickRequestV1):
            raise TensorWorldContractError("C1 advance requires TickRequestV1")
        owned = self._owned(state)
        allocated_before = torch.cuda.memory_allocated(self._device)
        reserved_before = torch.cuda.memory_reserved(self._device)
        owned["graph"].replay()
        if (
            torch.cuda.memory_allocated(self._device) != allocated_before
            or torch.cuda.memory_reserved(self._device) != reserved_before
        ):
            raise RuntimeError("CUDA Graph replay changed allocator state")
        owned["bundle"].tick += 1
        return TickReceiptV1(self._capabilities.backend_id, 1, 0, 0, 0, 0, 0)

    def export_state(self, state: BackendRunStateV1) -> Mapping[str, Any]:
        owned = self._owned(state)
        torch.cuda.synchronize(self._device)
        bundle = owned["bundle"]
        return {
            "bundle": clone_bundle_to_device_v1(bundle, torch.device("cpu")),
            "config": owned["config"],
        }

    def residency_counters(self, state: BackendRunStateV1) -> Mapping[str, int]:
        return dict(self._owned(state)["steady_state_counters"])

    def close(self, state: BackendRunStateV1) -> None:
        if state.closed:
            return
        self._runs.pop(state.token, None)
        state.closed = True
