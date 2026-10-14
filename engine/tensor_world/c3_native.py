"""Source-current resident C3 native execution contract.

The C3 vectorized backend is useful as a development comparator, but the
production route is the Architecture-B persistent full-tick entry point.  This
module owns the ordered tensor ABI and the allocation/mapping boundary for
that entry point.  It deliberately does not claim C3 qualification: C5 RNG
state must be explicitly bound before native execution is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Final, Mapping, Any

import torch
from torch.utils.cpp_extension import load

from .c2_backend import TensorWorldC2ConfigV1
from .c3_contract import (
    C3_ACTION_COUNT_V1,
    C3_OBSERVATION_CHANNELS_V1,
    C3_OBSERVATION_COLUMNS_V1,
    C3_POLICY_HIDDEN_WIDTH_V1,
    C3_POLICY_PACKED_WIDTH_V1,
)
from .c3_state import TensorWorldStateBundleV3
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
PERF_ROOT = ROOT / "engine" / "perf"
NATIVE_BINDING = PERF_ROOT / "cuda_architecture_b_persistent_generation_binding.cpp"
NATIVE_KERNEL = PERF_ROOT / "cuda_architecture_b_persistent_generation_kernel.cu"
NATIVE_HEADER = PERF_ROOT / "cuda_architecture_b_persistent_generation.hpp"

C3_PERSISTENT_NATIVE_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c3_persistent_native_state_v1"
)
C3_PERSISTENT_NATIVE_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_persistent_flat_dynamic_policy_food_birth_full_tick_v7"
)
C3_PERSISTENT_NATIVE_RESOURCE_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_persistent_flat_dynamic_policy_food_birth_full_tick_resource_receipt_v7"
)


# These tuples mirror the current binding/kernel order.  They are part of the
# state contract, not cosmetic labels: changing one requires a source audit.
FULL_TICK_STATE_FIELDS_V1: Final[tuple[str, ...]] = (
    "age",
    "gru_upkeep_cost",
    "cost_charge_scalar",
    "cost_charge_maturation_scalar",
    "drive_inertia",
    "drive_noise",
    "prev_drive_delta",
    "drive_volatility",
    "drive_jitter",
    "episode0_drive_jitter_sum",
    "last_energy_delta",
    "last_health_delta",
    "last_injury_delta",
    "selfharm_events",
    "episode0_selfharm_events",
    "chemistry",
    "runner_pcg64_state_words",
    "runner_pcg64_buffer_words",
    "shared_env_policy_pcg64_state_words",
    "shared_env_policy_pcg64_buffer_words",
    "drive_noise_samples_by_tick_slot",
    "drive_noise_samples_valid_by_tick_slot",
    "slot_free_queue",
    "slot_free_queue_count",
    "residual_pending_agent_ids",
    "residual_pending_slots",
    "residual_pending_count",
    "policy_np_semantic_draws",
    "metrics_np_semantic_draws",
)

FULL_TICK_OUTPUT_FIELDS_V1: Final[tuple[str, ...]] = (
    "chemistry_order_slots",
    "runner_shuffle_draws",
    "runner_shuffle_status",
    "drive_noise_draws",
    "drive_noise_status",
    "chemistry_draws",
    "chemistry_status",
    "active_mechanics_status",
    "active_mechanics_counts",
    "residual_death_agent_ids",
    "residual_death_slots",
    "residual_death_reason_codes",
    "residual_death_count",
    "residual_release_count",
    "residual_release_status",
    "policy_pcg64_state_after_action",
    "policy_pcg64_buffer_after_action",
    "food_spawn_stats",
)

EVENT_STATE_FIELDS_V1: Final[tuple[str, ...]] = (
    "event_type",
    "generation",
    "tick",
    "actor_id",
    "target_id",
    "value",
    "typed_event_type",
    "typed_generation",
    "typed_tick",
    "typed_sequence",
    "typed_actor_id",
    "typed_target_id",
    "typed_actor_slot",
    "typed_target_slot",
    "typed_parent_id",
    "typed_child_id",
    "typed_founder_id",
    "typed_lineage_depth",
    "typed_subtype",
    "typed_reason",
    "typed_success",
    "typed_value",
    "active_count",
    "status",
    "lineage_founder_id",
    "lineage_depth",
    "initial_id_to_row",
    "terminal_survival",
    "terminal_rows",
    "status_by_tick",
)

TOP_INPUT_FIELDS_V1: Final[tuple[str, ...]] = (
    "active_agent_ids",
    "active_count_state",
    "slot_to_agent_id",
    "agent_id_to_slot",
    "alive_mask",
    "slot_tokens",
    "action_cache_actions",
    "action_cache_token",
    "action_cache_cursor",
    "action_cache_next",
    "pos_y",
    "pos_x",
    "agent_grid",
    "food_grid",
    "energy",
    "health",
    "injury",
    "policy_energy_observation",
    "rest_streak",
    "pending_free_agent_ids",
    "pending_free_slots",
    "pending_free_count",
    "energy_collected",
    "energy_collected_exact",
    "last_action",
    "last_last_action",
    "action_switches",
    "action_oscillations",
    "episode0_switches",
    "episode0_oscillations",
    "survival_steps",
    "mlp_pack",
    "drives",
    "attack_bias",
    "action_temperature",
    "drive_logit_gain",
    "food_gained_scratch",
    "ordered_agent_ids",
    "ordered_slots",
    "observations",
    "logits",
    "row_actions",
    "actions_by_slot",
    "pre_energy",
    "pre_health",
    "pre_injury",
    "pre_survival_steps",
    "active_count",
    "policy_refill_rows",
    "policy_cache_hit_rows",
    "death_agent_ids",
    "death_slots",
    "death_reason_codes",
    "death_count",
    "survivor_agent_ids",
    "survivor_count",
    "lifecycle_status",
    "status",
)


@dataclass(frozen=True, slots=True)
class TensorWorldC3PersistentConfigV1:
    """Scalar arguments frozen for the fixed-MLP C3 native route."""

    view_radius: int = 4
    hidden_size: int = 64
    action_count: int = 7
    cache_width: int = 2
    deterministic_actions: bool = True
    max_food: int = 1
    food_spawn_per_step: float = 0.0
    food_spawn_mult: float = 1.0
    food_energy: float = 1.0
    episode0_k: int = 10
    w_hunger_gather: float = 1.2
    w_pain_rest: float = 0.8
    w_fatigue_rest: float = 1.0
    w_pain_attack: float = 0.8
    w_fear_attack: float = 1.0
    enable_rest_charge: bool = False
    rest_streak_free: int = 3
    rest_charge_energy: float = 0.0
    rest_charge_growth: float = 0.0
    rest_deficit_to_health: float = 0.0
    developmental_maturity_age: int = 20
    drive_hunger_gain: float = 0.1
    drive_hunger_decay: float = 0.01
    drive_pain_gain: float = 0.2
    drive_pain_decay: float = 0.01
    drive_fatigue_gain: float = 0.05
    drive_fatigue_rest_relief: float = 0.1
    drive_fatigue_decay: float = 0.01
    drive_fear_gain: float = 0.2
    drive_fear_decay: float = 0.01
    generation: int = 0
    start_tick: int = 0
    max_births_per_tick: int = 0
    reproduce_parent_energy_share: float = 0.5
    reproduce_min_energy_buffer: float = 0.0
    reproduce_cooldown_steps: int = 0
    genome_sensor_radius_max: int = 1
    injury_repro_k_injury: float = 0.0
    injury_repro_k_scar: float = 0.0
    injury_birth_fail_scar: float = 0.0
    injury_birth_fail_injury: float = 0.0
    env_policy_alias: bool = False
    reproduce_requires_mate: bool = False
    max_energy: float = 160.0
    start_health: float = 12.0
    move_cost: float = 0.2
    gather_cost: float = 0.08
    rest_cost: float = 0.05
    attack_cost: float = 0.6
    action_cost_mult: float = 1.0
    rest_cost_mult: float = 1.0


@dataclass(slots=True)
class TensorWorldC3PersistentNativeStateV1:
    """All tensors passed to one persistent native invocation."""

    tensors: dict[str, torch.Tensor]
    full_tick_state: tuple[torch.Tensor, ...]
    full_tick_outputs: tuple[torch.Tensor, ...]
    event_state: tuple[torch.Tensor, ...]
    ticks: int
    event_capacity: int
    rng_bound: bool = False
    schema_version: str = C3_PERSISTENT_NATIVE_SCHEMA_VERSION_V1

    @property
    def device(self) -> torch.device:
        return self.tensors["active_agent_ids"].device

    @property
    def cells(self) -> int:
        return int(self.tensors["active_agent_ids"].shape[0])

    @property
    def capacity(self) -> int:
        return int(self.tensors["slot_to_agent_id"].shape[1])

    def validate(self, *, require_rng: bool = False) -> None:
        if self.schema_version != C3_PERSISTENT_NATIVE_SCHEMA_VERSION_V1:
            raise TensorWorldContractError(
                "C3 persistent native state schema version is unsupported"
            )
        if type(self.ticks) is not int or not 0 < self.ticks <= 80:
            raise TensorWorldContractError("C3 native ticks must be in [1, 80]")
        if type(self.event_capacity) is not int or self.event_capacity <= 0:
            raise TensorWorldContractError("C3 native event capacity must be positive")
        if require_rng and not self.rng_bound:
            raise TensorWorldContractError(
                "C3 native execution requires an explicit C5 PCG64 state binding"
            )
        missing = [name for name in TOP_INPUT_FIELDS_V1 if name not in self.tensors]
        if missing:
            raise TensorWorldContractError(
                f"C3 native state is missing top-level tensors: {missing}"
            )
        if len(self.full_tick_state) != len(FULL_TICK_STATE_FIELDS_V1):
            raise TensorWorldContractError("C3 native full-tick state order is incomplete")
        if len(self.full_tick_outputs) != len(FULL_TICK_OUTPUT_FIELDS_V1):
            raise TensorWorldContractError("C3 native full-tick output order is incomplete")
        if len(self.event_state) != len(EVENT_STATE_FIELDS_V1):
            raise TensorWorldContractError("C3 native event state order is incomplete")
        reference = self.tensors["active_agent_ids"]
        if not reference.is_contiguous():
            raise TensorWorldContractError("C3 native active IDs must be contiguous")
        devices = {reference.device}
        for name in TOP_INPUT_FIELDS_V1:
            tensor = self.tensors[name]
            if not isinstance(tensor, torch.Tensor) or not tensor.is_contiguous():
                raise TensorWorldContractError(f"C3 native tensor {name} must be contiguous")
            devices.add(tensor.device)
        for group in (self.full_tick_state, self.full_tick_outputs, self.event_state):
            for tensor in group:
                if not isinstance(tensor, torch.Tensor) or not tensor.is_contiguous():
                    raise TensorWorldContractError("C3 native ABI tensors must be contiguous")
                devices.add(tensor.device)
        if len(devices) != 1:
            raise TensorWorldContractError("C3 native ABI tensors must share one device")
        cells, capacity = self.cells, self.capacity
        if reference.ndim != 2 or reference.shape[0] != cells or not 0 < reference.shape[1] <= capacity:
            raise TensorWorldContractError("C3 active ID shape is inconsistent")
        if not bool(torch.all(self.tensors["active_count_state"] == reference.shape[1]).item()):
            raise TensorWorldContractError("C3 active ID rows disagree with active count")
        if self.tensors["active_count_state"].shape != (cells,):
            raise TensorWorldContractError("C3 active count state shape is invalid")
        if self.tensors["slot_to_agent_id"].shape != (cells, capacity):
            raise TensorWorldContractError("C3 slot identity shape is invalid")
        if self.tensors["alive_mask"].shape != (cells, capacity):
            raise TensorWorldContractError("C3 alive mask shape is invalid")
        if self.tensors["agent_grid"].shape[0] != cells or self.tensors["agent_grid"].ndim != 3:
            raise TensorWorldContractError("C3 agent grid shape is invalid")
        if self.tensors["mlp_pack"].shape[:2] != (cells, capacity):
            raise TensorWorldContractError("C3 policy pack shape is invalid")
        if self.tensors["action_cache_actions"].shape != (cells, capacity, 2):
            raise TensorWorldContractError("C3 action cache shape is invalid")
        if self.tensors["observations"].shape[:3] != (cells, self.ticks, capacity):
            raise TensorWorldContractError("C3 observation output shape is invalid")
        if self.tensors["logits"].shape[:3] != (cells, self.ticks, capacity):
            raise TensorWorldContractError("C3 logits output shape is invalid")
        if self.event_state[0].shape != (cells, self.event_capacity):
            raise TensorWorldContractError("C3 event capacity shape is invalid")
        if self.event_state[22].shape != (cells,) or self.event_state[23].shape != (cells,):
            raise TensorWorldContractError("C3 event cursor/status shape is invalid")
        if self.full_tick_state[20].shape[:3] != (cells, self.ticks, capacity):
            raise TensorWorldContractError("C3 drive-noise sample shape is invalid")
        if self.full_tick_state[21].shape != (cells, self.ticks, capacity):
            raise TensorWorldContractError("C3 drive-noise validity shape is invalid")
        for index in (16, 18):
            if (
                self.full_tick_state[index].dtype != torch.uint64
                or self.full_tick_state[index].shape != (cells, 4)
            ):
                raise TensorWorldContractError(
                    "C3 native PCG64 state words require uint64[cells,4]"
                )
        for index in (17, 19):
            if (
                self.full_tick_state[index].dtype != torch.int64
                or self.full_tick_state[index].shape != (cells, 2)
            ):
                raise TensorWorldContractError(
                    "C3 native PCG64 buffer words require int64[cells,2]"
                )


def c3_fixed_mlp_policy_cpu_oracle_v1(
    observations: torch.Tensor,
    packed_weights: torch.Tensor,
    drives: torch.Tensor,
    attack_bias: torch.Tensor,
    action_temperature: torch.Tensor,
    drive_logit_gain: torch.Tensor,
    *,
    alive: torch.Tensor | None = None,
    action_uniforms: torch.Tensor | None = None,
    w_hunger_gather: float = 1.2,
    w_pain_rest: float = 0.8,
    w_fatigue_rest: float = 1.0,
    w_pain_attack: float = 0.8,
    w_fear_attack: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the source fixed-MLP policy on CPU for parity preparation.

    This is a comparator oracle only.  It follows the source pack layout.  If
    ``action_uniforms`` is supplied, action selection uses the source Gumbel
    fixture rule; otherwise it returns the deterministic argmax used by the
    bounded policy-only fixture.  ``alive`` makes source inactive-row zeroing
    explicit.  It is not a steady-state backend path.
    """

    expected_rows = tuple(observations.shape[:2]) if isinstance(observations, torch.Tensor) and observations.ndim == 3 else None
    if expected_rows is None or observations.shape[-1] != 83:
        raise TensorWorldContractError("C3 policy observations require shape [batch,capacity,83]")
    if (
        not isinstance(packed_weights, torch.Tensor)
        or packed_weights.shape[:2] != expected_rows
        or packed_weights.shape[-1] != C3_POLICY_PACKED_WIDTH_V1
        or packed_weights.dtype != torch.float32
    ):
        raise TensorWorldContractError("C3 policy pack requires source shape [batch,capacity,9991] float32")
    for name, tensor, trailing in (
        ("drives", drives, (4,)),
        ("attack_bias", attack_bias, ()),
        ("action_temperature", action_temperature, ()),
        ("drive_logit_gain", drive_logit_gain, ()),
    ):
        if not isinstance(tensor, torch.Tensor) or tensor.shape[:2] != expected_rows or tuple(tensor.shape[2:]) != trailing:
            raise TensorWorldContractError(f"C3 policy {name} shape is incompatible with observations")
    if observations.dtype != torch.float32 or drives.dtype != torch.float32:
        raise TensorWorldContractError("C3 policy observations and drives require float32")
    if alive is not None:
        if alive.shape != expected_rows or alive.dtype != torch.bool:
            raise TensorWorldContractError("C3 policy alive mask requires bool [batch,capacity]")
        alive_mask = alive
    else:
        alive_mask = torch.ones(expected_rows, dtype=torch.bool, device=observations.device)
    if action_uniforms is not None:
        if (
            not isinstance(action_uniforms, torch.Tensor)
            or action_uniforms.shape != (*expected_rows, C3_ACTION_COUNT_V1)
            or not action_uniforms.is_floating_point()
        ):
            raise TensorWorldContractError(
                "C3 policy action uniforms require floating [batch,capacity,7]"
            )
        uniforms = action_uniforms.to(dtype=torch.float32).clamp(1.0e-6, 1.0 - 1.0e-6)
    else:
        uniforms = None
    hidden = C3_POLICY_HIDDEN_WIDTH_V1
    observation_width = observations.shape[-1]
    first_end = hidden * observation_width
    first_bias_end = first_end + hidden
    second_weights_end = first_bias_end + hidden * hidden
    second_bias_end = second_weights_end + hidden
    output_weights_end = second_bias_end + C3_ACTION_COUNT_V1 * hidden
    if output_weights_end + C3_ACTION_COUNT_V1 != C3_POLICY_PACKED_WIDTH_V1:
        raise TensorWorldContractError("C3 policy pack constants are inconsistent")

    first_weights = packed_weights[..., :first_end].reshape(*expected_rows, hidden, observation_width)
    first_bias = packed_weights[..., first_end:first_bias_end]
    second_weights = packed_weights[..., first_bias_end:second_weights_end].reshape(*expected_rows, hidden, hidden)
    second_bias = packed_weights[..., second_weights_end:second_bias_end]
    output_weights = packed_weights[..., second_bias_end:output_weights_end].reshape(
        *expected_rows, C3_ACTION_COUNT_V1, hidden
    )
    output_bias = packed_weights[..., output_weights_end:]
    first = torch.tanh(torch.einsum("bso,bsho->bsh", observations, first_weights) + first_bias)
    second = torch.tanh(torch.einsum("bsh,bskh->bsk", first, second_weights) + second_bias)
    logits = torch.einsum("bsh,bsah->bsa", second, output_weights) + output_bias
    temperature = action_temperature.clamp_min(0.05).unsqueeze(-1)
    gain = drive_logit_gain.clamp_min(0.0).unsqueeze(-1)
    logits = logits.clone()
    logits[..., 6] = logits[..., 6] + attack_bias
    logits = logits / temperature
    logits[..., 4] = logits[..., 4] + gain[..., 0] * w_hunger_gather * drives[..., 0]
    logits[..., 5] = logits[..., 5] + gain[..., 0] * (
        w_pain_rest * drives[..., 1] + w_fatigue_rest * drives[..., 2]
    )
    logits[..., 6] = logits[..., 6] - gain[..., 0] * (
        w_pain_attack * drives[..., 1] + w_fear_attack * drives[..., 3]
    )
    logits = torch.where(alive_mask.unsqueeze(-1), logits, torch.zeros_like(logits))
    if uniforms is None:
        actions = torch.argmax(logits, dim=-1).to(torch.int32)
    else:
        scores = logits - torch.log(-torch.log(uniforms))
        actions = torch.argmax(scores, dim=-1).to(torch.int32)
    actions = torch.where(alive_mask, actions, torch.full_like(actions, -1))
    return logits.contiguous(), actions.contiguous()


def c3_observation_cpu_oracle_v1(
    bundle: TensorWorldStateBundleV3,
    config: TensorWorldC2ConfigV1 | None = None,
) -> torch.Tensor:
    """Materialize the source C2 83-column observation ABI on a CPU fixture.

    This is intentionally an oracle, not a steady-state backend path.  The
    channel order and normalization mirror ``observe_kernel`` in the bound C2
    source so C3 parity reports can identify observation mismatches before a
    CUDA route is admitted.
    """

    bundle.validate()
    if bundle.device.type != "cpu":
        raise TensorWorldContractError("C3 observation oracle requires a CPU fixture")
    config = config or TensorWorldC2ConfigV1()
    if not isinstance(config, TensorWorldC2ConfigV1):
        raise TensorWorldContractError("C3 observation oracle requires a C2 config")
    core = bundle.base.core
    width = bundle.grid_size
    output = torch.zeros(
        (bundle.batch, bundle.capacity, 83), dtype=torch.float32, device="cpu"
    )
    density_denom = float((2 * config.density_radius + 1) ** 2)
    if not config.density_include_self:
        density_denom -= 1.0
    density_denom = max(density_denom, 1.0)

    def safe_norm(value: float, maximum: float) -> float:
        if maximum <= 0.0:
            return 0.0
        return min(1.0, max(0.0, value / maximum))

    def cue_value(value: float, base_seed: int, y: int, x: int) -> float:
        cue = min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0
        dropout = min(1.0, max(0.0, float(config.wdl_dropout_rate)))
        if dropout <= 0.0:
            return cue
        if dropout >= 1.0:
            return 0.0
        mixed = (
            y * 1_103_515_245
            + x * 12_345
            + (int(base_seed) & 0xFFFF_FFFF)
            + 0x9E37_79B9
        ) & 0xFFFF_FFFF
        return cue if mixed / 4_294_967_296.0 >= dropout else 0.0

    for world in range(bundle.batch):
        base_seed = int(bundle.base.wdl.base_seed[world])
        for slot in range(bundle.capacity):
            if not bool(core.agents.alive[world, slot]):
                continue
            center_y = int(core.agents.pos_y[world, slot])
            center_x = int(core.agents.pos_x[world, slot])
            for offset_y, dy in enumerate((-1, 0, 1)):
                for offset_x, dx in enumerate((-1, 0, 1)):
                    y = center_y + dy
                    x = center_x + dx
                    offset = offset_y * 3 + offset_x
                    values = [0.0] * 9
                    if 0 <= y < width and 0 <= x < width:
                        grid_index = (world, y, x)
                        occupied = int(core.occupancy.agent_id_grid[grid_index]) != -1
                        food = float(core.world.food_amount[grid_index])
                        values = [
                            -1.0 if occupied else (1.0 if food > config.food_presence_threshold else 0.0),
                            safe_norm(food, config.food_max),
                            min(1.0, max(0.0, float(core.world.cover[grid_index]))),
                            min(1.0, max(0.0, float(core.world.shelter[grid_index]))),
                            safe_norm(float(core.world.scent[grid_index]), config.scent_max),
                            safe_norm(float(core.world.waste[grid_index]), config.waste_max),
                            (
                                min(1.0, max(0.0, float(core.world.density[grid_index])))
                                if config.density_normalize
                                else safe_norm(float(core.world.density[grid_index]), density_denom)
                            ),
                            min(1.0, max(0.0, float(core.world.movement_cost[grid_index]) / 4.0)),
                            cue_value(float(bundle.base.wdl.cue[grid_index]), base_seed, y, x),
                        ]
                    for channel, value in enumerate(values):
                        output[world, slot, channel * 9 + offset] = value
            output[world, slot, 81] = float(core.agents.energy[world, slot]) / config.max_energy
            output[world, slot, 82] = float(core.agents.health[world, slot]) / config.start_health
    return output.contiguous()


def compare_c3_identity_surfaces_v1(
    expected_agent_ids: torch.Tensor,
    expected_surfaces: Mapping[str, torch.Tensor],
    actual_agent_ids: torch.Tensor,
    actual_surfaces: Mapping[str, torch.Tensor],
    *,
    atol: float = 1.0e-6,
    rtol: float = 0.0,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Compare non-hot-path C3 surfaces by world and biological identity.

    Native and oracle routes may use different resident slots after C4
    settlement.  This preparation-only comparator canonicalizes each world by
    active agent ID before comparing observations, logits, actions, movement,
    or eligibility tensors.  It intentionally materializes CPU views and must
    not be called from a steady-state tick.
    """

    if (
        not isinstance(expected_agent_ids, torch.Tensor)
        or not isinstance(actual_agent_ids, torch.Tensor)
        or expected_agent_ids.ndim != 2
        or actual_agent_ids.ndim != 2
        or tuple(expected_agent_ids.shape) != tuple(actual_agent_ids.shape)
    ):
        raise TensorWorldContractError("C3 identity comparator requires matching [cells,capacity] ID tensors")
    expected_ids = expected_agent_ids.detach().to(device="cpu", dtype=torch.int64)
    actual_ids = actual_agent_ids.detach().to(device="cpu", dtype=torch.int64)
    cells, capacity = expected_ids.shape

    def identity_order(ids: torch.Tensor, world: int) -> tuple[int, ...]:
        row = ids[world].tolist()
        active = [int(agent_id) for agent_id in row if int(agent_id) >= 0]
        if len(active) != len(set(active)):
            raise TensorWorldContractError("C3 identity comparator found duplicate biological IDs")
        return tuple(sorted(active))

    expected_order = tuple(identity_order(expected_ids, world) for world in range(cells))
    actual_order = tuple(identity_order(actual_ids, world) for world in range(cells))
    if expected_order != actual_order:
        raise TensorWorldContractError("C3 identity comparator found an identity-set mismatch")

    names = tuple(expected_surfaces)
    if set(names) != set(actual_surfaces):
        raise TensorWorldContractError("C3 identity comparator surface names do not match")
    expected_rows: dict[str, torch.Tensor] = {}
    actual_rows: dict[str, torch.Tensor] = {}
    for name in names:
        left = expected_surfaces[name]
        right = actual_surfaces[name]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TensorWorldContractError(f"C3 comparator surface {name} must be a tensor")
        if left.ndim < 2 or right.ndim < 2 or tuple(left.shape[2:]) != tuple(right.shape[2:]):
            raise TensorWorldContractError(f"C3 comparator surface {name} has incompatible trailing shape")
        if tuple(left.shape[:2]) != (cells, capacity) or tuple(right.shape[:2]) != (cells, capacity):
            raise TensorWorldContractError(f"C3 comparator surface {name} has incompatible row shape")
        expected_rows[name] = left.detach().to(device="cpu")
        actual_rows[name] = right.detach().to(device="cpu")

    mismatches: dict[str, tuple[tuple[int, int], ...]] = {}
    for name in names:
        rows: list[tuple[int, int]] = []
        for world in range(cells):
            expected_slots = {
                int(expected_ids[world, slot]): slot
                for slot in range(capacity)
                if int(expected_ids[world, slot]) >= 0
            }
            actual_slots = {
                int(actual_ids[world, slot]): slot
                for slot in range(capacity)
                if int(actual_ids[world, slot]) >= 0
            }
            for agent_id in expected_order[world]:
                expected_slot = expected_slots[agent_id]
                actual_slot = actual_slots[agent_id]
                expected_row = expected_rows[name][world, expected_slot]
                actual_row = actual_rows[name][world, actual_slot]
                if expected_row.is_floating_point() or actual_row.is_floating_point():
                    equal = torch.allclose(
                        expected_row,
                        actual_row,
                        atol=atol,
                        rtol=rtol,
                        equal_nan=False,
                    )
                else:
                    equal = torch.equal(expected_row, actual_row)
                if not equal:
                    rows.append((world, agent_id))
        if rows:
            mismatches[name] = tuple(rows)
    return mismatches


def compare_c3_observation_channels_v1(
    expected_agent_ids: torch.Tensor,
    expected_observations: torch.Tensor,
    actual_agent_ids: torch.Tensor,
    actual_observations: torch.Tensor,
    *,
    atol: float = 1.0e-6,
    rtol: float = 0.0,
) -> dict[str, tuple[tuple[int, int, str], ...]]:
    """Compare 83-column observations by channel and biological identity.

    This preparation-only comparator preserves the slot-independent identity
    mapping of :func:`compare_c3_identity_surfaces_v1` while reporting the
    exact source registry column that differs.  It intentionally materializes
    CPU views and must not be used in a steady-state tick.
    """

    if (
        not isinstance(expected_observations, torch.Tensor)
        or not isinstance(actual_observations, torch.Tensor)
        or expected_observations.ndim != 3
        or actual_observations.ndim != 3
        or tuple(expected_observations.shape) != tuple(actual_observations.shape)
        or expected_observations.shape[-1] != len(C3_OBSERVATION_COLUMNS_V1)
    ):
        raise TensorWorldContractError(
            "C3 observation comparator requires matching [cells,capacity,83] tensors"
        )
    if (
        expected_observations.dtype != torch.float32
        or actual_observations.dtype != torch.float32
    ):
        raise TensorWorldContractError("C3 observation comparator requires float32 tensors")
    if (
        not isinstance(expected_agent_ids, torch.Tensor)
        or not isinstance(actual_agent_ids, torch.Tensor)
        or expected_agent_ids.ndim != 2
        or tuple(expected_agent_ids.shape) != tuple(actual_agent_ids.shape)
        or tuple(expected_agent_ids.shape) != tuple(expected_observations.shape[:2])
    ):
        raise TensorWorldContractError(
            "C3 observation comparator requires matching identity tensors"
        )
    expected_ids = expected_agent_ids.detach().to(device="cpu", dtype=torch.int64)
    actual_ids = actual_agent_ids.detach().to(device="cpu", dtype=torch.int64)
    expected = expected_observations.detach().to(device="cpu")
    actual = actual_observations.detach().to(device="cpu")
    cells, capacity = expected_ids.shape

    def slots_by_id(ids: torch.Tensor, world: int) -> dict[int, int]:
        result: dict[int, int] = {}
        for slot in range(capacity):
            agent_id = int(ids[world, slot])
            if agent_id < 0:
                continue
            if agent_id in result:
                raise TensorWorldContractError(
                    "C3 observation comparator found duplicate biological IDs"
                )
            result[agent_id] = slot
        return result

    mismatches: dict[str, list[tuple[int, int, str]]] = {
        channel: [] for channel in C3_OBSERVATION_CHANNELS_V1
    }
    mismatches["agent_state"] = []
    for world in range(cells):
        expected_slots = slots_by_id(expected_ids, world)
        actual_slots = slots_by_id(actual_ids, world)
        if set(expected_slots) != set(actual_slots):
            raise TensorWorldContractError("C3 observation comparator found an identity-set mismatch")
        for agent_id, expected_slot in expected_slots.items():
            actual_slot = actual_slots[agent_id]
            left = expected[world, expected_slot]
            right = actual[world, actual_slot]
            for column, column_name in enumerate(C3_OBSERVATION_COLUMNS_V1):
                if not bool(torch.isclose(left[column], right[column], atol=atol, rtol=rtol, equal_nan=False)):
                    channel = (
                        C3_OBSERVATION_CHANNELS_V1[column // 9]
                        if column < 81
                        else "agent_state"
                    )
                    mismatches[channel].append((world, agent_id, column_name))
    return {
        channel: tuple(rows)
        for channel, rows in mismatches.items()
        if rows
    }


_MODULE: Any | None = None


def load_c3_persistent_native_extension(
    *, build_directory: str | Path | None = None, verbose: bool = False
) -> Any:
    """Build/load the current persistent full-tick source entry point."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise TensorWorldContractError("C3 persistent native execution requires CUDA")
    sources = (NATIVE_BINDING, NATIVE_KERNEL)
    source_hash = sha256(
        b"\0".join(
            source.read_bytes()
            for source in (
                *sources,
                NATIVE_HEADER,
                PERF_ROOT / "cuda_architecture_b_persistent_active_mechanics.cuh",
                PERF_ROOT / "cuda_architecture_b_persistent_chemistry.cuh",
                PERF_ROOT / "cuda_architecture_b_persistent_stochastic_policy.cuh",
                PERF_ROOT / "cuda_architecture_b_torch_rng_state_pack_v1.cuh",
            )
        )
    ).hexdigest()[:12]
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load(
        name=f"tensor_world_c3_persistent_native_{source_hash}",
        sources=[str(source) for source in sources],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(PERF_ROOT), str(PERF_ROOT / "native")],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false", "--prec-div=true", "--prec-sqrt=true"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def _owned(tensor: torch.Tensor, device: torch.device, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is None and tensor.device == device and tensor.is_contiguous():
        return tensor
    return tensor.to(device=device, dtype=dtype, copy=True).contiguous()


def _zeros(shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, dtype=dtype, device=device)


def _full(shape: tuple[int, ...], value: int | float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.full(shape, value, dtype=dtype, device=device)


def _seeded_pcg64(cells: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create valid but explicitly unqualified PCG64 state for preparation."""

    state = _zeros((cells, 4), torch.uint64, device)
    state_bits = state.view(torch.int64)
    for index, value in enumerate(
        (0x853C49E6748FEA9B, 0xDA3E39CB94B95BDB, 0x14057B7EF767814F, 0x5851F42D4C957F2D)
    ):
        signed = value if value < (1 << 63) else value - (1 << 64)
        state_bits[:, index] = signed
    return state, _zeros((cells, 2), torch.int64, device)


def _dense_agent_id_to_slot(
    slot_to_agent_id: torch.Tensor, *, capacity: int, device: torch.device
) -> torch.Tensor:
    cells = int(slot_to_agent_id.shape[0])
    max_id = int(slot_to_agent_id.max().item()) if slot_to_agent_id.numel() else -1
    id_capacity = max(capacity, max_id + 1)
    result = _full((cells, id_capacity), -1, torch.int32, device)
    slots = torch.arange(capacity, dtype=torch.int32, device=device).expand(cells, -1)
    valid = (slot_to_agent_id >= 0) & (slot_to_agent_id < id_capacity)
    safe_ids = slot_to_agent_id.clamp(0, id_capacity - 1)
    result.scatter_(1, safe_ids.to(torch.int64), torch.where(valid, slots, torch.full_like(slots, -1)))
    return result


def _sorted_active_ids(slot_to_agent_id: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
    sentinel = torch.iinfo(torch.int64).max
    ordered = torch.where(alive & (slot_to_agent_id >= 0), slot_to_agent_id, sentinel)
    ordered = torch.sort(ordered, dim=1).values
    return torch.where(ordered == sentinel, torch.full_like(ordered, -1), ordered)


def _free_queue(alive: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cells, capacity = alive.shape
    slots = torch.arange(capacity, dtype=torch.int64, device=alive.device).expand(cells, -1)
    ordered = torch.sort(torch.where(~alive, slots, slots + capacity), dim=1).values
    queue = torch.where(ordered < capacity, ordered, torch.full_like(ordered, -1))
    return queue.contiguous(), (~alive).sum(dim=1, dtype=torch.int64).contiguous()


def make_c3_persistent_native_state_v1(
    bundle: TensorWorldStateBundleV3,
    *,
    ticks: int = 1,
    event_capacity: int | None = None,
    device: str | torch.device | None = None,
    rng_state: Mapping[str, torch.Tensor] | None = None,
) -> TensorWorldC3PersistentNativeStateV1:
    """Project a C3 bundle into the exact persistent native ABI.

    The projection is initialization-time only.  When ``rng_state`` is not
    supplied, valid PCG64 words are materialized for shape/ABI rehearsal but
    ``rng_bound`` remains false and execution fails closed until C5 binds the
    streams.
    """

    bundle.validate()
    target = torch.device(device) if device is not None else bundle.device
    source = bundle.base.core
    c3 = bundle
    cells, capacity = bundle.batch, bundle.capacity
    if type(ticks) is not int or not 0 < ticks <= 80:
        raise TensorWorldContractError("C3 native ticks must be in [1, 80]")
    alive = _owned(source.agents.alive, target)
    slot_to_agent_id = _owned(source.agents.agent_id, target)
    id_to_slot = _dense_agent_id_to_slot(slot_to_agent_id, capacity=capacity, device=target)
    active_order = _sorted_active_ids(slot_to_agent_id, alive)
    active_count = (active_order >= 0).sum(dim=1, dtype=torch.int64)
    if not bool(torch.all(active_count == active_count[0]).item()):
        raise TensorWorldContractError("C3 native route requires equal active counts across cells")
    active_agent_ids = active_order[:, : int(active_count[0].item())].contiguous()
    queue, queue_count = _free_queue(alive)
    cache_actions = _zeros((cells, capacity, 2), torch.int32, target)
    cache_actions.copy_(_owned(c3.policy_runtime.action_cache_actions, target))
    pending_ids = _full((cells, capacity), -1, torch.int64, target)
    pending_slots = _full((cells, capacity), -1, torch.int64, target)

    runner_state, runner_buffer = _seeded_pcg64(cells, target)
    shared_state, shared_buffer = _seeded_pcg64(cells, target)
    bound = False
    if rng_state is not None:
        required = (
            "runner_pcg64_state_words",
            "runner_pcg64_buffer_words",
            "shared_env_policy_pcg64_state_words",
            "shared_env_policy_pcg64_buffer_words",
        )
        missing = [name for name in required if name not in rng_state]
        if missing:
            raise TensorWorldContractError(f"C3 native RNG binding is missing {missing}")
        runner_state = _owned(rng_state[required[0]], target)
        runner_buffer = _owned(rng_state[required[1]], target)
        shared_state = _owned(rng_state[required[2]], target)
        shared_buffer = _owned(rng_state[required[3]], target)
        bound = True

    drive_noise = _owned(c3.transactions.drive_noise_samples, target)
    drive_noise = drive_noise.unsqueeze(1).expand(cells, ticks, capacity, drive_noise.shape[-1]).contiguous()
    drive_valid = _owned(c3.transactions.drive_noise_samples_valid, target)
    drive_valid = drive_valid.unsqueeze(1).expand(cells, ticks, capacity).contiguous()
    residual_ids = _full((cells, capacity), -1, torch.int64, target)
    residual_slots = _full((cells, capacity), -1, torch.int64, target)
    residual_count = _zeros((cells,), torch.int64, target)

    tensors: dict[str, torch.Tensor] = {
        "active_agent_ids": active_agent_ids,
        "active_count_state": active_count,
        "slot_to_agent_id": slot_to_agent_id,
        "agent_id_to_slot": id_to_slot,
        "alive_mask": alive,
        "slot_tokens": _owned(source.identity.slot_ids, target),
        "action_cache_actions": cache_actions,
        "action_cache_token": _owned(c3.policy_runtime.action_cache_token, target),
        "action_cache_cursor": _owned(c3.policy_runtime.action_cache_cursor, target),
        "action_cache_next": _owned(c3.policy_runtime.action_cache_next, target),
        "pos_y": _owned(source.agents.pos_y, target, dtype=torch.int32),
        "pos_x": _owned(source.agents.pos_x, target, dtype=torch.int32),
        "agent_grid": _owned(c3.base.patch.agent_grid_i32, target),
        "food_grid": _owned(c3.base.patch.food_grid_i32, target),
        "energy": _owned(source.agents.energy, target),
        "health": _owned(source.agents.health, target),
        "injury": _owned(c3.agent.injury, target),
        "policy_energy_observation": _owned(c3.agent.policy_energy_observation, target),
        "rest_streak": _owned(c3.agent.rest_streak, target),
        "pending_free_agent_ids": pending_ids,
        "pending_free_slots": pending_slots,
        "pending_free_count": _zeros((cells,), torch.int64, target),
        "energy_collected": _owned(c3.agent.energy_collected, target),
        "energy_collected_exact": _owned(c3.agent.energy_collected_exact, target),
        "last_action": _owned(c3.agent.last_action, target),
        "last_last_action": _owned(c3.agent.last_last_action, target),
        "action_switches": _owned(c3.agent.action_switches, target),
        "action_oscillations": _owned(c3.agent.action_oscillations, target),
        "episode0_switches": _owned(c3.agent.episode0_switches, target),
        "episode0_oscillations": _owned(c3.agent.episode0_oscillations, target),
        "survival_steps": _owned(c3.agent.survival_steps, target),
        # C2 owns the source-current 83-feature fixed-MLP pack.  The
        # inherited C1 core still retains its 74-feature compatibility pack;
        # routing that tensor here would fail the native ABI boundary.
        "mlp_pack": _owned(bundle.base.policy.packed_weights, target),
        "drives": _owned(c3.agent.drives, target),
        "attack_bias": _owned(c3.genome.aggression_bias, target),
        "action_temperature": _owned(c3.genome.action_temperature, target),
        "drive_logit_gain": _owned(c3.genome.drive_logit_gain, target),
        "food_gained_scratch": _zeros((cells, capacity), torch.float32, target),
    }

    def output(name: str, shape: tuple[int, ...], dtype: torch.dtype, value: int | float = 0) -> None:
        tensors[name] = _full(shape, value, dtype, target)

    output("ordered_agent_ids", (cells, ticks, capacity), torch.int64, -1)
    output("ordered_slots", (cells, ticks, capacity), torch.int32, -1)
    output("observations", (cells, ticks, capacity, 83), torch.float32)
    output("logits", (cells, ticks, capacity, 7), torch.float32)
    output("row_actions", (cells, ticks, capacity), torch.int32, -1)
    output("actions_by_slot", (cells, ticks, capacity), torch.int32, -1)
    output("pre_energy", (cells, ticks, capacity), torch.float32)
    output("pre_health", (cells, ticks, capacity), torch.float32)
    output("pre_injury", (cells, ticks, capacity), torch.float32)
    output("pre_survival_steps", (cells, ticks, capacity), torch.int32)
    output("active_count", (cells, ticks), torch.int64)
    output("policy_refill_rows", (cells, ticks), torch.int64)
    output("policy_cache_hit_rows", (cells, ticks), torch.int64)
    output("death_agent_ids", (cells, ticks, capacity), torch.int64, -1)
    output("death_slots", (cells, ticks, capacity), torch.int64, -1)
    output("death_reason_codes", (cells, ticks, capacity), torch.int32)
    output("death_count", (cells, ticks), torch.int64)
    output("survivor_agent_ids", (cells, ticks, capacity), torch.int64, -1)
    output("survivor_count", (cells, ticks), torch.int64)
    output("lifecycle_status", (cells, ticks), torch.int32)
    output("status", (cells,), torch.int32)

    full_tick: dict[str, torch.Tensor] = {
        "age": _owned(c3.agent.age, target),
        "gru_upkeep_cost": _owned(c3.agent.gru_upkeep_cost, target),
        "cost_charge_scalar": _owned(c3.agent.cost_charge_scalar, target),
        "cost_charge_maturation_scalar": _owned(c3.agent.cost_charge_maturation_scalar, target),
        "drive_inertia": _owned(c3.agent.drive_inertia, target),
        "drive_noise": _owned(c3.agent.drive_noise, target),
        "prev_drive_delta": _owned(c3.agent.prev_drive_delta, target),
        "drive_volatility": _owned(c3.agent.drive_volatility, target),
        "drive_jitter": _owned(c3.agent.drive_jitter, target),
        "episode0_drive_jitter_sum": _owned(c3.agent.episode0_drive_jitter_sum, target),
        "last_energy_delta": _owned(c3.agent.last_energy_delta, target),
        "last_health_delta": _owned(c3.agent.last_health_delta, target),
        "last_injury_delta": _owned(c3.agent.last_injury_delta, target),
        "selfharm_events": _owned(c3.agent.selfharm_events, target),
        "episode0_selfharm_events": _owned(c3.agent.episode0_selfharm_events, target),
        "chemistry": _owned(c3.adaptation.chemistry, target),
        "runner_pcg64_state_words": runner_state,
        "runner_pcg64_buffer_words": runner_buffer,
        "shared_env_policy_pcg64_state_words": shared_state,
        "shared_env_policy_pcg64_buffer_words": shared_buffer,
        "drive_noise_samples_by_tick_slot": drive_noise,
        "drive_noise_samples_valid_by_tick_slot": drive_valid,
        "slot_free_queue": queue,
        "slot_free_queue_count": queue_count,
        "residual_pending_agent_ids": residual_ids,
        "residual_pending_slots": residual_slots,
        "residual_pending_count": residual_count,
        "policy_np_semantic_draws": _zeros((cells,), torch.int64, target),
        "metrics_np_semantic_draws": _zeros((cells,), torch.int64, target),
    }
    full_tick_state = tuple(full_tick[name] for name in FULL_TICK_STATE_FIELDS_V1)

    for name, shape, dtype, value in (
        ("chemistry_order_slots", (cells, ticks, capacity), torch.int32, -1),
        ("runner_shuffle_draws", (cells, ticks), torch.int64, 0),
        ("runner_shuffle_status", (cells, ticks), torch.int32, 0),
        ("drive_noise_draws", (cells, ticks), torch.int64, 0),
        ("drive_noise_status", (cells, ticks), torch.int32, 0),
        ("chemistry_draws", (cells, ticks), torch.int64, 0),
        ("chemistry_status", (cells, ticks), torch.int32, 0),
        ("active_mechanics_status", (cells, ticks), torch.int32, 0),
        ("active_mechanics_counts", (cells, ticks, 5), torch.int64, 0),
        ("residual_death_agent_ids", (cells, ticks, capacity), torch.int64, -1),
        ("residual_death_slots", (cells, ticks, capacity), torch.int64, -1),
        ("residual_death_reason_codes", (cells, ticks, capacity), torch.int32, 0),
        ("residual_death_count", (cells, ticks), torch.int64, 0),
        ("residual_release_count", (cells, ticks), torch.int64, 0),
        ("residual_release_status", (cells, ticks), torch.int32, 0),
        ("policy_pcg64_state_after_action", (cells, ticks, 4), torch.uint64, 0),
        ("policy_pcg64_buffer_after_action", (cells, ticks, 2), torch.int64, 0),
        ("food_spawn_stats", (cells, ticks, 3), torch.int64, 0),
    ):
        tensors[name] = _full(shape, value, dtype, target)
    full_tick_outputs = tuple(tensors[name] for name in FULL_TICK_OUTPUT_FIELDS_V1)

    event_capacity = max(capacity, int(event_capacity or 2 * capacity))
    event_dtype: tuple[torch.dtype, ...] = (
        torch.int32, torch.int64, torch.int64, torch.int64, torch.int64, torch.float32,
        torch.int32, torch.int64, torch.int64, torch.int64, torch.int64, torch.int64,
        torch.int64, torch.int64, torch.int64, torch.int64, torch.int64, torch.int32,
        torch.int32, torch.int32, torch.int8, torch.float32,
    )
    event_tensors: dict[str, torch.Tensor] = {}
    for index, name in enumerate(EVENT_STATE_FIELDS_V1[:22]):
        event_tensors[name] = _zeros((cells, event_capacity), event_dtype[index], target)
    event_tensors["active_count"] = _zeros((cells,), torch.int64, target)
    event_tensors["status"] = _zeros((cells,), torch.int32, target)
    event_tensors["lineage_founder_id"] = _full((cells, capacity), -1, torch.int32, target)
    event_tensors["lineage_depth"] = _zeros((cells, capacity), torch.int32, target)
    initial_id_to_row = _full((cells, id_to_slot.shape[1]), -1, torch.int64, target)
    initial_id_to_row.scatter_(1, active_agent_ids.clamp(0, id_to_slot.shape[1] - 1), torch.arange(active_agent_ids.shape[1], device=target, dtype=torch.int64).expand(cells, -1))
    event_tensors["initial_id_to_row"] = initial_id_to_row
    event_tensors["terminal_survival"] = _zeros((cells, capacity), torch.int32, target)
    event_tensors["terminal_rows"] = _zeros((cells, capacity), torch.int32, target)
    event_tensors["status_by_tick"] = _zeros((cells, ticks), torch.int32, target)
    event_state = tuple(event_tensors[name] for name in EVENT_STATE_FIELDS_V1)

    result = TensorWorldC3PersistentNativeStateV1(
        tensors={**tensors, **full_tick},
        full_tick_state=full_tick_state,
        full_tick_outputs=full_tick_outputs,
        event_state=event_state,
        ticks=ticks,
        event_capacity=event_capacity,
        rng_bound=bound,
    )
    result.validate()
    return result


def execute_c3_persistent_native_v1(
    state: TensorWorldC3PersistentNativeStateV1,
    config: TensorWorldC3PersistentConfigV1 | None = None,
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> dict[str, torch.Tensor]:
    """Execute one preallocated persistent C3 call; C5 RNG is mandatory."""

    state.validate(require_rng=True)
    config = config or TensorWorldC3PersistentConfigV1()
    module = load_c3_persistent_native_extension(
        build_directory=build_directory, verbose=verbose
    )
    top = state.tensors
    args = [top[name] for name in TOP_INPUT_FIELDS_V1]
    args.extend(
        (
            list(state.full_tick_state),
            list(state.full_tick_outputs),
            list(state.event_state),
            [],
            [],
            state.ticks,
            config.view_radius,
            config.hidden_size,
            config.action_count,
            config.cache_width,
            config.deterministic_actions,
            config.max_food,
            config.food_spawn_per_step,
            config.food_spawn_mult,
            config.max_energy,
            config.start_health,
            config.move_cost,
            config.gather_cost,
            config.rest_cost,
            config.attack_cost,
            config.action_cost_mult,
            config.rest_cost_mult,
            config.food_energy,
            config.episode0_k,
            config.w_hunger_gather,
            config.w_pain_rest,
            config.w_fatigue_rest,
            config.w_pain_attack,
            config.w_fear_attack,
            config.enable_rest_charge,
            config.rest_streak_free,
            config.rest_charge_energy,
            config.rest_charge_growth,
            config.rest_deficit_to_health,
            config.developmental_maturity_age,
            config.drive_hunger_gain,
            config.drive_hunger_decay,
            config.drive_pain_gain,
            config.drive_pain_decay,
            config.drive_fatigue_gain,
            config.drive_fatigue_rest_relief,
            config.drive_fatigue_decay,
            config.drive_fear_gain,
            config.drive_fear_decay,
            config.generation,
            config.start_tick,
            config.max_births_per_tick,
            config.reproduce_parent_energy_share,
            config.reproduce_min_energy_buffer,
            config.reproduce_cooldown_steps,
            config.genome_sensor_radius_max,
            config.injury_repro_k_injury,
            config.injury_repro_k_scar,
            config.injury_birth_fail_scar,
            config.injury_birth_fail_injury,
            config.env_policy_alias,
            config.reproduce_requires_mate,
        )
    )
    module_args = getattr(module, C3_PERSISTENT_NATIVE_ENTRYPOINT_V1)
    module_args(*args)
    result = {name: top[name] for name in TOP_INPUT_FIELDS_V1 if name in top}
    result.update(dict(zip(FULL_TICK_OUTPUT_FIELDS_V1, state.full_tick_outputs, strict=True)))
    for name, tensor in zip(EVENT_STATE_FIELDS_V1, state.event_state, strict=True):
        # ``active_count`` and ``status`` are also top-level tick outputs;
        # preserve those authoritative output names and expose event cursors
        # under an explicit namespace instead of silently overwriting them.
        result[f"event_{name}" if name in result else name] = tensor
    return result


__all__ = [
    "C3_PERSISTENT_NATIVE_ENTRYPOINT_V1",
    "C3_PERSISTENT_NATIVE_RESOURCE_ENTRYPOINT_V1",
    "C3_PERSISTENT_NATIVE_SCHEMA_VERSION_V1",
    "EVENT_STATE_FIELDS_V1",
    "FULL_TICK_OUTPUT_FIELDS_V1",
    "FULL_TICK_STATE_FIELDS_V1",
    "TOP_INPUT_FIELDS_V1",
    "TensorWorldC3PersistentConfigV1",
    "TensorWorldC3PersistentNativeStateV1",
    "c3_observation_cpu_oracle_v1",
    "c3_fixed_mlp_policy_cpu_oracle_v1",
    "compare_c3_identity_surfaces_v1",
    "compare_c3_observation_channels_v1",
    "execute_c3_persistent_native_v1",
    "load_c3_persistent_native_extension",
    "make_c3_persistent_native_state_v1",
]
