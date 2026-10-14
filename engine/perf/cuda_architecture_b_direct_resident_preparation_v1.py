"""Prepare an exact Patch P2 configurable-birth direct-resident trajectory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import struct
from typing import Any, Final

from emergence_contracts.direct_resident_field_surface import (
    BASE_BATCHED_FIELDS as BASE_BATCHED_FIELDS,
    ORDINARY_BIRTH_DRIVE_FIELDS as ORDINARY_BIRTH_DRIVE_FIELDS,
    ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS as ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS,
    ORDINARY_BIRTH_GENOME_INT32_FIELDS as ORDINARY_BIRTH_GENOME_INT32_FIELDS,
    ORDINARY_BIRTH_IDENTITY_FIELDS as ORDINARY_BIRTH_IDENTITY_FIELDS,
    ORDINARY_BIRTH_POLICY_FIELDS as ORDINARY_BIRTH_POLICY_FIELDS,
    ORDINARY_BIRTH_SCALAR_RNG_FIELDS as ORDINARY_BIRTH_SCALAR_RNG_FIELDS,
    PATCH_P2_STATE_FIELDS_V1,
    PATCH_RESIDENT_FIELDS as PATCH_RESIDENT_FIELDS,
    TURNOVER_STATE_FIELDS as TURNOVER_STATE_FIELDS,
)

import torch

from engine.perf import cuda_architecture_b_resident_chunk_runtime_v1 as chunk_runtime
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    DIRECT_ARGUMENT_COUNT,
    GENERATION_COUNT_ARGUMENT_INDEX,
    GENERATION_ARGUMENT_INDEX,
    PHYSICAL_WINDOW_GENERATIONS,
)
from engine.perf.cuda_architecture_b_flattened_birth_scratch_layout_v2 import (
    make_legacy_tail_birth_scratch_layout_v2,
)
from engine.perf.cuda_diploid_state_bank_v1 import (
    CANONICAL_DIGEST_TEMPLATE_PACK_BYTES,
)


LOGICAL_GENERATIONS_V1: Final[int] = 4
BATCH_CELLS_V1: Final[int] = 1
TICKS_PER_GENERATION_V1: Final[int] = 80
ADMITTED_TICKS_PER_GENERATION_V1: Final[tuple[int, ...]] = (80, 160)
MAX_TICKS_PER_GENERATION_V1: Final[int] = 160
GENERATION_SUMMARY_WIDTH_V1: Final[int] = 31
PATCH_STAT_COUNT_V1: Final[int] = 25
ORDINARY_BIRTH_OUTPUT_COUNT_V1: Final[int] = 63
TURNOVER_OUTPUT_COUNT_V1: Final[int] = 16
ORDINARY_BIRTH_STATE_COUNTS_V1: Final[frozenset[int]] = frozenset({113, 122, 139})
ADMITTED_INITIAL_LIVE_COUNTS_V1: Final[frozenset[int]] = frozenset(
    {64, 128, 192, 256, 512}
)
LIVE_STATE_ARGUMENT_FIELDS_V1: Final[tuple[str, ...]] = (
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
)

PREFIX_OUTPUT_ARGUMENT_FIELDS_V1: Final[tuple[str, ...]] = (
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
)

POST_BOOKKEEPING_OUTPUT_ARGUMENT_FIELDS_V1: Final[tuple[str, ...]] = (
    "death_agent_ids",
    "death_slots",
    "death_reason_codes",
    "death_count",
    "survivor_agent_ids",
    "survivor_count",
    "lifecycle_status",
    "status",
)

PRE_GENERATION_CONFIG_ARGUMENTS_V1: Final[tuple[tuple[str, type], ...]] = (
    ("view_radius", int),
    ("hidden_size", int),
    ("action_count", int),
    ("cache_width", int),
    ("max_food", int),
    ("food_spawn_per_step", float),
    ("food_spawn_mult", float),
    ("max_energy", float),
    ("start_energy", float),
    ("start_health", float),
    ("move_cost", float),
    ("gather_cost", float),
    ("rest_cost", float),
    ("attack_cost", float),
    ("action_cost_mult", float),
    ("rest_cost_mult", float),
    ("food_energy", float),
    ("episode0_k", int),
    ("w_hunger_gather", float),
    ("w_pain_rest", float),
    ("w_fatigue_rest", float),
    ("w_pain_attack", float),
    ("w_fear_attack", float),
    ("enable_rest_charge", bool),
    ("rest_streak_free", int),
    ("rest_charge_energy", float),
    ("rest_charge_growth", float),
    ("rest_deficit_to_health", float),
    ("developmental_maturity_age", int),
    ("life_history_mode", int),
    ("life_history_maturity_start_ticks", int),
    ("life_history_maturity_rise_ticks", int),
    ("life_history_fertility_decline_start_ticks", int),
    ("life_history_fertility_decline_ticks", int),
    ("life_history_elder_fertility_floor", float),
    ("life_history_senescence_start_ticks", int),
    ("life_history_senescence_ramp_ticks", int),
    ("life_history_senescence_energy_cost", float),
    ("life_history_senescence_health_cost", float),
    ("life_history_max_age_ticks", int),
    ("drive_hunger_gain", float),
    ("drive_hunger_decay", float),
    ("drive_pain_gain", float),
    ("drive_pain_decay", float),
    ("drive_fatigue_gain", float),
    ("drive_fatigue_rest_relief", float),
    ("drive_fatigue_decay", float),
    ("drive_fear_gain", float),
    ("drive_fear_decay", float),
)

POST_GENERATION_CONFIG_ARGUMENTS_V1: Final[tuple[tuple[str, type], ...]] = (
    ("start_tick", int),
    ("ordinary_births_enabled", bool),
    ("max_births_per_tick", int),
    ("reproduce_parent_energy_share", float),
    ("reproduce_min_energy_buffer", float),
    ("reproduce_cooldown_steps", int),
    ("genome_sensor_radius_max", int),
    ("injury_repro_k_injury", float),
    ("injury_repro_k_scar", float),
    ("injury_birth_fail_scar", float),
    ("injury_birth_fail_injury", float),
    ("env_policy_alias", bool),
    ("reproduce_requires_mate", bool),
    ("generation_turnover_enabled", bool),
    ("lifecycle_mode", int),
    ("population_size", int),
    ("cull_fraction", float),
    ("parent_fraction", float),
    ("turnover_selection_mode", int),
    ("coherence_thrashing_weight", float),
    ("coherence_drive_volatility_weight", float),
    ("coherence_selfharm_weight", float),
    ("coherence_pred_err_weight", float),
)

REPORTING_FITNESS_ARGUMENTS_V1: Final[tuple[tuple[str, type], ...]] = (
    ("parent_fraction", float),
    ("episode0_fitness_weight", float),
    ("episode0_pred_err_fitness_weight", float),
    ("episode0_thrashing_fitness_weight", float),
    ("episode0_drive_jitter_fitness_weight", float),
    ("episode0_selfharm_fitness_weight", float),
    ("coherence_thrashing_weight", float),
    ("coherence_drive_jitter_weight", float),
    ("coherence_selfharm_weight", float),
    ("coherence_pred_err_weight", float),
    ("coherence_reset_shock_weight", float),
    ("coherence_episode0_pred_err_weight", float),
    ("late_pred_cost_weight", float),
    ("late_pred_cost_enable", bool),
    ("disable_prediction_error_selection_pressure", bool),
)

PATCH_P2_CONFIG_ARGUMENTS_V1: Final[tuple[tuple[str, type], ...]] = (
    ("patch_scent_decay_per_step", float),
    ("patch_waste_decay_per_step", float),
    ("patch_scent_presence_emit", float),
    ("patch_waste_presence_emit", float),
    ("patch_waste_density_emit_scale", float),
    ("patch_scent_max", float),
    ("patch_waste_max", float),
    ("patch_food_amount_max", float),
    ("patch_food_regrow_per_step", float),
    ("patch_food_regrow_use_fertility", bool),
    ("patch_food_regrow_cap_by_fertility", bool),
    ("patch_food_presence_threshold", float),
    ("patch_density_radius", int),
    ("patch_density_include_self", bool),
    ("patch_density_normalize", bool),
    ("patch_world_waste_consequence_enable", bool),
    ("patch_waste_regrowth_penalty_enable", bool),
    ("patch_waste_regrowth_penalty_scale", float),
    ("patch_waste_regrowth_penalty_max", float),
    ("patch_density_regrowth_penalty_enable", bool),
    ("patch_density_regrowth_penalty_scale", float),
    ("patch_density_regrowth_penalty_max", float),
    ("patch_regrowth_min_multiplier", float),
    ("patch_food_amount_initial_food_cell_value", float),
    ("patch_food_gather_amount", float),
    ("patch_scent_move_emit", float),
    ("patch_waste_move_emit", float),
    ("patch_scent_gather_emit", float),
    ("patch_waste_consume_emit", float),
    ("patch_movement_cost_apply_enable", bool),
    ("patch_movement_cost_energy_scale", float),
    ("patch_movement_cost_min", float),
    ("patch_movement_cost_max", float),
)

_PREFIX_ARGUMENT_NAMES = (
    "active_agent_ids",
    "active_agent_slots",
    "active_count_state",
    *(f"live_state.{name}" for name in LIVE_STATE_ARGUMENT_FIELDS_V1),
    *(f"outputs.{name}" for name in PREFIX_OUTPUT_ARGUMENT_FIELDS_V1),
    "physical_runtime_bookkeeping",
    *(f"outputs.{name}" for name in POST_BOOKKEEPING_OUTPUT_ARGUMENT_FIELDS_V1),
    "full_tick_state_tensors",
    "physical_full_tick_output_tensors",
    "compact_summary_reporting_tensors",
    "compact_summary_checkpoint_source_tensors",
    "compact_summary_checkpoint_descriptor_specs",
    "compact_summary_checkpoint_output_tensors",
    "compact_summary_settlement_tensors",
    "event_state_tensors",
    "ordinary_birth_state_tensors",
    "ordinary_birth_output_tensors",
    "ordinary_birth_scratch_tensors",
    "turnover_selection_seed_by_cell",
    "turnover_state_tensors",
    "turnover_shadow_tensors",
    "turnover_output_tensors",
    "turnover_pointer_table_tensors",
    "patch_p2_state_tensors",
    "patch_p2_scratch_tensors",
    "physical_patch_stats_tensors",
    "patch_p2_config",
    "ticks_per_generation",
)
_PRE_CONFIG_ARGUMENT_NAMES = tuple(
    f"config.{name}" for name, _value_type in PRE_GENERATION_CONFIG_ARGUMENTS_V1
)
_POST_CONFIG_ARGUMENT_NAMES = tuple(
    f"config.{name}" for name, _value_type in POST_GENERATION_CONFIG_ARGUMENTS_V1
)
_REPORTING_ARGUMENT_NAMES = tuple(
    f"reporting_fitness.{name}" for name, _value_type in REPORTING_FITNESS_ARGUMENTS_V1
)

DIRECT_RESIDENT_ARGUMENT_NAMES_V1: Final[tuple[str, ...]] = (
    *_PREFIX_ARGUMENT_NAMES,
    *_PRE_CONFIG_ARGUMENT_NAMES[:4],
    "deterministic",
    *_PRE_CONFIG_ARGUMENT_NAMES[4:],
    "generation_base",
    *_POST_CONFIG_ARGUMENT_NAMES,
    "physical_generation_count",
    *_REPORTING_ARGUMENT_NAMES,
    "outputs.first_failure_records",
)


def _argument_index(name: str) -> int:
    try:
        return DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(name)
    except ValueError as exc:
        raise RuntimeError(f"direct resident ABI is missing {name}") from exc


EVENT_STATE_ARGUMENT_INDEX_V1: Final[int] = _argument_index("event_state_tensors")
ORDINARY_BIRTH_STATE_ARGUMENT_INDEX_V1: Final[int] = _argument_index(
    "ordinary_birth_state_tensors"
)
TURNOVER_SELECTION_SEED_BY_CELL_ARGUMENT_INDEX_V1: Final[int] = _argument_index(
    "turnover_selection_seed_by_cell"
)
GENERATION_ARGUMENT_INDEX_V1: Final[int] = _argument_index("generation_base")
GENERATION_COUNT_ARGUMENT_INDEX_V1: Final[int] = _argument_index(
    "physical_generation_count"
)

if len(DIRECT_RESIDENT_ARGUMENT_NAMES_V1) != DIRECT_ARGUMENT_COUNT:
    raise RuntimeError(
        f"direct resident ABI schema must contain exactly {DIRECT_ARGUMENT_COUNT} "
        "arguments"
    )
if GENERATION_ARGUMENT_INDEX_V1 != GENERATION_ARGUMENT_INDEX:
    raise RuntimeError("direct resident ABI generation marker index changed")
if GENERATION_COUNT_ARGUMENT_INDEX_V1 != GENERATION_COUNT_ARGUMENT_INDEX:
    raise RuntimeError("direct resident ABI generation count index changed")


class DirectResidentPreparationError(ValueError):
    """The direct resident source or its dispatch ABI is incomplete."""


@dataclass(frozen=True, slots=True)
class DirectResidentSourceV1:
    """All live authorities required to prepare one exact production launch."""

    configuration: object
    active_agent_ids: Any
    active_agent_slots: Any
    active_count_state: Any
    live_state: Mapping[str, Any]
    outputs: Mapping[str, Any]
    full_tick_state_tensors: Sequence[Any]
    full_tick_output_tensors: Sequence[Any]
    compact_summary_reporting_tensors: Sequence[Any]
    compact_summary_checkpoint_source_tensors: Sequence[Any]
    compact_summary_checkpoint_descriptor_specs: Any
    compact_summary_checkpoint_output_tensors: Sequence[Any]
    compact_summary_settlement_tensors: Sequence[Any]
    event_state_tensors: Sequence[Any]
    ordinary_birth_state_tensors: Sequence[Any]
    ordinary_birth_output_tensors: Sequence[Any]
    ordinary_birth_scratch_tensors: Sequence[Any]
    turnover_selection_seed_by_cell: Any
    turnover_state_tensors: Sequence[Any]
    turnover_shadow_tensors: Sequence[Any]
    turnover_output_tensors: Sequence[Any]
    turnover_pointer_table_tensors: Sequence[Any]
    patch_p2_state_tensors: Sequence[Any]
    patch_p2_scratch_tensors: Sequence[Any]
    patch_p2_stats_tensors: Sequence[Any]
    patch_p2_config: Mapping[str, Any]
    live_banks: Mapping[str, Any]
    checkpoint_projection_inputs: Mapping[str, Any]
    deterministic: bool
    generation_base: int
    logical_generation_count: int = LOGICAL_GENERATIONS_V1
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS
    ticks_per_generation: int = TICKS_PER_GENERATION_V1
    batch_cells: int = BATCH_CELLS_V1
    allow_short_technical_gate_generation_count: bool = False
    allow_partial_generation_source_cursor: bool = False
    move_cost_by_cell: Any | None = None
    time_authority: str = "legacy_generation_v1"
    resident_owner_span_ticks: int | None = None
    native_dispatch_window_ticks: int | None = None
    refill_chunk_ticks: int | None = None
    source_window_ticks: int | None = None
    resident_event_capacity: int | None = None


@dataclass(frozen=True, slots=True)
class PreparedDirectResidentTrajectoryV1:
    """A fully validated direct ABI plus retained state authorities."""

    source: DirectResidentSourceV1
    arguments: tuple[Any, ...]
    physical_generation_summary: Any
    physical_runtime_bookkeeping: Any
    physical_patch_stats: Any
    physical_full_tick_output_tensors: Sequence[Any]
    live_banks: Mapping[str, Any]
    checkpoint_projection_inputs: Mapping[str, Any]

    @property
    def physical_window_arguments(self) -> tuple[Any, ...]:
        return self.arguments


def _fail(message: str) -> DirectResidentPreparationError:
    return DirectResidentPreparationError(message)


def _validated_initial_live_count_v1(value: Any, *, path: str) -> int:
    if type(value) is not int or value not in ADMITTED_INITIAL_LIVE_COUNTS_V1:
        raise _fail(
            f"{path} must be an admitted initial live count: "
            f"{sorted(ADMITTED_INITIAL_LIVE_COUNTS_V1)!r}"
        )
    return int(value)


def _require_admitted_ticks_per_generation_v1(value: object, *, path: str) -> int:
    if type(value) is not int or value not in ADMITTED_TICKS_PER_GENERATION_V1:
        raise _fail(
            f"{path} must be one of {list(ADMITTED_TICKS_PER_GENERATION_V1)!r}"
        )
    return value


def _require_exact_int(value: object, *, path: str, expected: int | None = None) -> int:
    if type(value) is not int:
        raise _fail(f"{path} must be an exact integer")
    if expected is not None and value != expected:
        raise _fail(f"{path} must equal {expected}")
    return value


def _require_physical_window_generations_v1(value: object, *, path: str) -> int:
    if type(value) is not int or value < 1 or value > 2:
        raise _fail(f"{path} must be an exact integer in [1, 2]")
    return value


def validate_asexual_birth_bound_v1(
    max_births_per_tick: object,
    slot_capacity: object,
    *,
    path: str = "configuration.max_births_per_tick",
) -> int:
    """Validate the configurable asexual birth bound against resident slots."""

    births = _require_exact_int(max_births_per_tick, path=path)
    capacity = _require_exact_int(slot_capacity, path="slot_capacity")
    if capacity < 1:
        raise _fail("slot_capacity must be positive")
    if births < 1 or births > capacity:
        raise _fail(f"{path} must be in [1, slot_capacity={capacity}], got {births}")
    return births


def required_lifecycle_scratch_bytes_v1(
    max_births_per_tick: object,
    slot_capacity: object,
    packed_width: object,
    hmem_width: object,
    torch_state_bytes: object,
) -> int:
    """Return the checked per-cell scratch extent for one production launch."""

    births = validate_asexual_birth_bound_v1(max_births_per_tick, slot_capacity)
    capacity = int(slot_capacity)
    widths = {
        "packed_width": packed_width,
        "hmem_width": hmem_width,
        "torch_state_bytes": torch_state_bytes,
    }
    normalized: dict[str, int] = {}
    for name, value in widths.items():
        normalized[name] = _require_exact_int(value, path=name)
        if normalized[name] < 1:
            raise _fail(f"{name} must be positive")
    layout = make_legacy_tail_birth_scratch_layout_v2(
        births,
        capacity,
        normalized["packed_width"],
        normalized["hmem_width"],
        normalized["torch_state_bytes"],
    )
    if not layout.ok():
        raise _fail(
            "runtime birth scratch layout v2 failed closed: "
            f"status={layout.status.name} max_births_per_tick={births} "
            f"slot_capacity={capacity}"
        )
    return layout.bytes_per_cell


def _require_sequence(
    value: object,
    *,
    path: str,
    length: int,
) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _fail(f"{path} must be a sequence of length {length}")
    if len(value) != length:
        raise _fail(f"{path} must contain exactly {length} entries")
    if any(item is None for item in value):
        raise _fail(f"{path} must not contain null entries")
    return value


def _require_sequence_length_in(
    value: object,
    *,
    path: str,
    lengths: frozenset[int],
) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise _fail(f"{path} must be a sequence with one of lengths {sorted(lengths)}")
    if len(value) not in lengths:
        raise _fail(f"{path} must contain one of {sorted(lengths)} entries")
    if any(item is None for item in value):
        raise _fail(f"{path} must not contain null entries")
    return value


def _shape(value: object, *, path: str) -> tuple[int, ...]:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        raise _fail(f"{path} must expose a tensor-like shape")
    try:
        return tuple(int(dimension) for dimension in raw_shape)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{path} has an invalid tensor-like shape") from exc


def _require_shape(value: object, expected: tuple[int, ...], *, path: str) -> None:
    actual = _shape(value, path=path)
    if actual != expected:
        raise _fail(f"{path} shape must be {expected}, got {actual}")


def _mapping_value(mapping: Mapping[str, Any], name: str, *, path: str) -> Any:
    if name not in mapping:
        raise _fail(f"{path} is missing {name}")
    value = mapping[name]
    if value is None:
        raise _fail(f"{path}.{name} must not be null")
    return value


def _require_finite_positive_float32(value: float, *, path: str) -> float:
    try:
        narrowed = struct.unpack("!f", struct.pack("!f", value))[0]
    except (OverflowError, struct.error):
        narrowed = math.inf
    if (
        not math.isfinite(value)
        or value <= 0.0
        or not math.isfinite(narrowed)
        or narrowed <= 0.0
    ):
        raise _fail(f"{path} must be finite, positive, and representable as float32")
    return narrowed


def _config_value(configuration: object, name: str, value_type: type) -> Any:
    if isinstance(configuration, Mapping):
        if name not in configuration:
            raise _fail(f"configuration is missing {name}")
        value = configuration[name]
    else:
        try:
            value = getattr(configuration, name)
        except AttributeError as exc:
            raise _fail(f"configuration is missing {name}") from exc
    if name == "ordinary_births_enabled":
        if value is None:
            if isinstance(configuration, Mapping):
                max_births = configuration.get("max_births_per_tick")
            else:
                max_births = getattr(configuration, "max_births_per_tick", None)
            if type(max_births) is not int:
                raise _fail(
                    "configuration.max_births_per_tick must be an exact integer "
                    "when ordinary_births_enabled is omitted"
                )
            return max_births > 0
        if type(value) is not bool:
            raise _fail("configuration.ordinary_births_enabled must be exact bool")
        return value
    if name == "turnover_selection_mode":
        if type(value) is not int:
            raise _fail("configuration.turnover_selection_mode must be exact int")
        return value
    try:
        converted = value_type(value)
    except (TypeError, ValueError) as exc:
        raise _fail(
            f"configuration.{name} cannot be converted to {value_type.__name__}"
        ) from exc
    if name in {"max_energy", "start_health"}:
        return _require_finite_positive_float32(converted, path=f"configuration.{name}")
    return converted


def _require_turnover_selection_seed_by_cell(
    value: object,
    *,
    batch_cells: int,
    path: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise _fail(f"{path} must be a CUDA uint64 tensor")
    if value.device.type != "cuda":
        raise _fail(f"{path} must be CUDA-resident")
    if value.dtype is not torch.uint64:
        raise _fail(f"{path} dtype must be torch.uint64")
    if value.shape != (batch_cells,):
        raise _fail(f"{path} shape must be ({batch_cells},), got {tuple(value.shape)}")
    if not value.is_contiguous():
        raise _fail(f"{path} must be contiguous")
    if value.requires_grad:
        raise _fail(f"{path} must not require gradients")
    return value


def _typed_config_values(
    configuration: object,
    schema: Sequence[tuple[str, type]],
) -> tuple[Any, ...]:
    return tuple(
        _config_value(configuration, name, value_type) for name, value_type in schema
    )


_PATCH_P2_FLOAT64_BY_CELL_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {
        "patch_food_gather_amount_by_cell",
        "patch_food_regrow_per_step_by_cell",
        "patch_movement_cost_energy_scale_by_cell",
        "reproduce_min_energy_buffer_by_cell",
        "reproduce_parent_energy_share_by_cell",
        "reproduce_try_prob_by_cell",
    }
)
_PATCH_P2_INT32_BY_CELL_FIELDS_V1: Final[frozenset[str]] = frozenset(
    {"reproduce_cooldown_steps_by_cell"}
)


def _validate_patch_p2_config(
    value: object,
    *,
    path: str,
    batch_cells: int | None = None,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{path} must be a mapping")
    expected_fields = {name for name, _value_type in PATCH_P2_CONFIG_ARGUMENTS_V1}
    actual_fields = set(value)
    optional_fields = (
        _PATCH_P2_FLOAT64_BY_CELL_FIELDS_V1 | _PATCH_P2_INT32_BY_CELL_FIELDS_V1
    )
    optional_fields = optional_fields | {
        "event_history_sink_path",
        "tw_rngv2_semantics_version",
        "tw_rngv2_world_uid",
        "tw_rngv2_world_uid_by_cell",
        "tw_rngv2_namespace_key64",
        "tw_rngv2_semantic_epoch",
    }
    if not expected_fields <= actual_fields or not actual_fields <= expected_fields | optional_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise _fail(
            f"{path} must contain the exact production fields: "
            f"missing={missing} extra={extra}"
        )
    for name, expected_type in PATCH_P2_CONFIG_ARGUMENTS_V1:
        if type(value[name]) is not expected_type:
            raise _fail(f"{path}.{name} must be exact {expected_type.__name__}")
    if "event_history_sink_path" in actual_fields and (
        type(value["event_history_sink_path"]) is not str
        or not value["event_history_sink_path"].strip()
    ):
        raise _fail(f"{path}.event_history_sink_path must be non-empty text")
    rng_fields = {
        "tw_rngv2_semantics_version",
        "tw_rngv2_world_uid",
        "tw_rngv2_world_uid_by_cell",
        "tw_rngv2_namespace_key64",
        "tw_rngv2_semantic_epoch",
    }
    present_rng_fields = actual_fields & rng_fields
    if present_rng_fields and present_rng_fields != rng_fields:
        raise _fail(
            f"{path} must provide the complete TW RNG coordinate contract"
        )
    if "tw_rngv2_semantics_version" in actual_fields and (
        type(value["tw_rngv2_semantics_version"]) is not int
        or value["tw_rngv2_semantics_version"] not in {1, 2}
    ):
        raise _fail(f"{path}.tw_rngv2_semantics_version must be exact 1 or 2")
    if "tw_rngv2_world_uid" in actual_fields and (
        type(value["tw_rngv2_world_uid"]) is not int
        or not 0 <= value["tw_rngv2_world_uid"] <= 0xFFFFFFFF
    ):
        raise _fail(f"{path}.tw_rngv2_world_uid must be an exact uint32")
    if "tw_rngv2_world_uid_by_cell" in actual_fields:
        world_uids = value["tw_rngv2_world_uid_by_cell"]
        if (
            not torch.is_tensor(world_uids)
            or world_uids.dtype != torch.int64
            or world_uids.ndim != 1
            or (batch_cells is not None and int(world_uids.numel()) != batch_cells)
        ):
            raise _fail(
                f"{path}.tw_rngv2_world_uid_by_cell must be int64[batch_cells]"
            )
    if "tw_rngv2_namespace_key64" in actual_fields and (
        type(value["tw_rngv2_namespace_key64"]) is not int
        or not 0 <= value["tw_rngv2_namespace_key64"] <= 0xFFFFFFFFFFFFFFFF
    ):
        raise _fail(f"{path}.tw_rngv2_namespace_key64 must be an exact uint64")
    if "tw_rngv2_semantic_epoch" in actual_fields and (
        type(value["tw_rngv2_semantic_epoch"]) is not int
        or value["tw_rngv2_semantic_epoch"] != 1
    ):
        raise _fail(f"{path}.tw_rngv2_semantic_epoch must equal 1")
    for name in sorted(_PATCH_P2_FLOAT64_BY_CELL_FIELDS_V1 & actual_fields):
        per_cell_scale = value[name]
        if (
            not torch.is_tensor(per_cell_scale)
            or per_cell_scale.dtype != torch.float64
            or per_cell_scale.ndim != 1
            or not per_cell_scale.is_contiguous()
            or not bool(torch.isfinite(per_cell_scale).all().item())
            or bool(torch.any(per_cell_scale < 0.0).item())
        ):
            raise _fail(
                f"{path}.{name} must be "
                "a contiguous finite nonnegative float64 vector"
            )
        if batch_cells is not None and tuple(per_cell_scale.shape) != (batch_cells,):
            raise _fail(
                f"{path}.{name} must have "
                "one value for each physical cell"
            )
    for name in sorted(_PATCH_P2_INT32_BY_CELL_FIELDS_V1 & actual_fields):
        per_cell_integer = value[name]
        if (
            not torch.is_tensor(per_cell_integer)
            or per_cell_integer.dtype != torch.int32
            or per_cell_integer.ndim != 1
            or not per_cell_integer.is_contiguous()
            or bool(torch.any(per_cell_integer < 0).item())
        ):
            raise _fail(
                f"{path}.{name} must be a contiguous nonnegative int32 vector"
            )
        if batch_cells is not None and tuple(per_cell_integer.shape) != (batch_cells,):
            raise _fail(f"{path}.{name} must have one value for each physical cell")
    return value


def _validate_source_contract(source: DirectResidentSourceV1) -> None:
    logical_generation_count = _require_exact_int(
        source.logical_generation_count,
        path="logical_generation_count",
    )
    minimum_generation_count = (
        1
        if source.allow_short_technical_gate_generation_count is True
        else 3
    )
    if (
        logical_generation_count < minimum_generation_count
        or logical_generation_count > 128
    ):
        raise _fail(
            "logical_generation_count must be an integer in "
            f"[{minimum_generation_count}, 128]"
        )
    physical_window_generations = _require_physical_window_generations_v1(
        source.physical_window_generations,
        path="physical_window_generations",
    )
    ticks_per_generation = _require_admitted_ticks_per_generation_v1(
        source.ticks_per_generation,
        path="ticks_per_generation",
    )
    batch_cells = _require_exact_int(source.batch_cells, path="batch_cells")
    if batch_cells < 1 or batch_cells > 256:
        raise _fail("batch_cells must be in [1,256]")
    generation = _require_exact_int(source.generation_base, path="generation_base")
    if generation < 0:
        raise _fail("generation_base must be nonnegative")
    if type(source.deterministic) is not bool or source.deterministic:
        raise _fail("the production source requires stochastic fixed-MLP policy")
    if not isinstance(source.live_state, Mapping):
        raise _fail("live_state must be a mapping")
    if not isinstance(source.outputs, Mapping):
        raise _fail("outputs must be a mapping")
    if not isinstance(source.live_banks, Mapping):
        raise _fail("live_banks must be a mapping")
    if not isinstance(source.checkpoint_projection_inputs, Mapping):
        raise _fail("checkpoint_projection_inputs must be a mapping")

    for name in LIVE_STATE_ARGUMENT_FIELDS_V1:
        _mapping_value(source.live_state, name, path="live_state")
    for name in (
        *PREFIX_OUTPUT_ARGUMENT_FIELDS_V1,
        *POST_BOOKKEEPING_OUTPUT_ARGUMENT_FIELDS_V1,
        "generation_summary_scalar",
        "runtime_bookkeeping_by_tick",
        "first_failure_records",
    ):
        _mapping_value(source.outputs, name, path="outputs")

    _require_shape(
        source.active_count_state,
        (batch_cells,),
        path="active_count_state",
    )
    active_id_shape = _shape(source.active_agent_ids, path="active_agent_ids")
    _require_shape(
        source.active_agent_slots,
        active_id_shape,
        path="active_agent_slots",
    )
    _require_shape(
        source.outputs["generation_summary_scalar"],
        (
            physical_window_generations,
            batch_cells,
            GENERATION_SUMMARY_WIDTH_V1,
        ),
        path="outputs.generation_summary_scalar",
    )
    active_tick_capacity = physical_window_generations * ticks_per_generation
    maximum_tick_capacity = (
        physical_window_generations * MAX_TICKS_PER_GENERATION_V1
    )
    runtime_bookkeeping_shape = _shape(
        source.outputs["runtime_bookkeeping_by_tick"],
        path="outputs.runtime_bookkeeping_by_tick",
    )
    if (
        len(runtime_bookkeeping_shape) != 3
        or runtime_bookkeeping_shape[0] != batch_cells
        or runtime_bookkeeping_shape[1] < active_tick_capacity
        or runtime_bookkeeping_shape[1] > maximum_tick_capacity
        or runtime_bookkeeping_shape[2] != 5
    ):
        raise _fail(
            "outputs.runtime_bookkeeping_by_tick shape must be "
            f"(batch_cells={batch_cells}, active_ticks>={active_tick_capacity}, "
            f"max_ticks<={maximum_tick_capacity}, 5), got "
            f"{runtime_bookkeeping_shape}"
        )

    _require_sequence(
        source.full_tick_state_tensors,
        path="full_tick_state_tensors",
        length=29,
    )
    full_tick_outputs = _require_sequence_length_in(
        source.full_tick_output_tensors,
        path="full_tick_output_tensors",
        lengths=frozenset({20, 27}),
    )
    if full_tick_outputs[19] is not source.outputs["generation_summary_scalar"]:
        raise _fail("full_tick_output_tensors[19] must be generation_summary_scalar")
    _require_sequence(
        source.compact_summary_reporting_tensors,
        path="compact_summary_reporting_tensors",
        length=0,
    )
    _require_sequence(
        source.compact_summary_checkpoint_source_tensors,
        path="compact_summary_checkpoint_source_tensors",
        length=0,
    )
    _require_shape(
        source.compact_summary_checkpoint_descriptor_specs,
        (0, 13),
        path="compact_summary_checkpoint_descriptor_specs",
    )
    _require_sequence(
        source.compact_summary_checkpoint_output_tensors,
        path="compact_summary_checkpoint_output_tensors",
        length=0,
    )
    _require_sequence(
        source.compact_summary_settlement_tensors,
        path="compact_summary_settlement_tensors",
        length=0,
    )
    _require_sequence(source.event_state_tensors, path="event_state_tensors", length=31)
    ordinary_birth_state = source.ordinary_birth_state_tensors
    if (
        not isinstance(ordinary_birth_state, Sequence)
        or isinstance(ordinary_birth_state, (str, bytes, bytearray))
        or len(ordinary_birth_state) not in ORDINARY_BIRTH_STATE_COUNTS_V1
        or any(item is None for item in ordinary_birth_state)
    ):
        raise _fail(
            "ordinary_birth_state_tensors must contain exactly 113 legacy, "
            "122 diploid, or 139 Genome V2 entries"
        )
    ordinary_births_enabled = _config_value(
        source.configuration, "ordinary_births_enabled", bool
    )
    generation_turnover_enabled = _config_value(
        source.configuration, "generation_turnover_enabled", bool
    )
    lifecycle_mode = _config_value(source.configuration, "lifecycle_mode", int)
    if lifecycle_mode not in {0, 1}:
        raise _fail("production configuration.lifecycle_mode must be 0 or 1")
    if lifecycle_mode == 0 and generation_turnover_enabled is not True:
        raise _fail("legacy lifecycle configuration requires generation turnover")
    if lifecycle_mode == 1 and generation_turnover_enabled is not False:
        raise _fail("hybrid lifecycle configuration requires no generation turnover")
    _require_sequence(
        source.ordinary_birth_output_tensors,
        path="ordinary_birth_output_tensors",
        length=(
            ORDINARY_BIRTH_OUTPUT_COUNT_V1
            if ordinary_births_enabled
            else 0
        ),
    )
    _require_sequence(
        source.ordinary_birth_scratch_tensors,
        path="ordinary_birth_scratch_tensors",
        length=1,
    )
    _require_turnover_selection_seed_by_cell(
        source.turnover_selection_seed_by_cell,
        batch_cells=batch_cells,
        path="turnover_selection_seed_by_cell",
    )
    _require_sequence(
        source.turnover_state_tensors,
        path="turnover_state_tensors",
        length=122 if generation_turnover_enabled else 0,
    )
    _require_sequence(
        source.turnover_shadow_tensors,
        path="turnover_shadow_tensors",
        length=122 if generation_turnover_enabled else 0,
    )
    _require_sequence(
        source.turnover_output_tensors,
        path="turnover_output_tensors",
        length=TURNOVER_OUTPUT_COUNT_V1,
    )
    _require_sequence(
        source.turnover_pointer_table_tensors,
        path="turnover_pointer_table_tensors",
        length=1 if generation_turnover_enabled else 0,
    )
    _require_sequence(
        source.patch_p2_state_tensors,
        path="patch_p2_state_tensors",
        length=len(PATCH_P2_STATE_FIELDS_V1),
    )
    patch_state_mismatches = [
        name
        for name, tensor in zip(
            PATCH_P2_STATE_FIELDS_V1,
            source.patch_p2_state_tensors,
            strict=True,
        )
        if source.live_state.get(name) is not tensor
    ]
    if patch_state_mismatches:
        raise _fail(
            "patch_p2_state_tensors do not retain live_state authority for: "
            + ", ".join(patch_state_mismatches)
        )
    _require_sequence(
        source.patch_p2_scratch_tensors,
        path="patch_p2_scratch_tensors",
        length=1,
    )
    patch_stats = _require_sequence(
        source.patch_p2_stats_tensors,
        path="patch_p2_stats_tensors",
        length=1,
    )
    patch_stats_shape = _shape(
        patch_stats[0],
        path="patch_p2_stats_tensors[0]",
    )
    if (
        len(patch_stats_shape) != 3
        or patch_stats_shape[0] != batch_cells
        or patch_stats_shape[1] < active_tick_capacity
        or patch_stats_shape[1] > maximum_tick_capacity
        or patch_stats_shape[2] != PATCH_STAT_COUNT_V1
    ):
        raise _fail(
            "patch_p2_stats_tensors[0] shape must be "
            f"(batch_cells={batch_cells}, active_ticks>={active_tick_capacity}, "
            f"max_ticks<={maximum_tick_capacity}, {PATCH_STAT_COUNT_V1}), "
            f"got {patch_stats_shape}"
        )

    _validate_patch_p2_config(
        source.patch_p2_config,
        path="patch_p2_config",
        batch_cells=batch_cells,
    )

    start_tick = _config_value(source.configuration, "start_tick", int)
    turnover_selection_mode = _config_value(
        source.configuration, "turnover_selection_mode", int
    )
    if isinstance(source.configuration, Mapping):
        raw_max_births = source.configuration.get("max_births_per_tick")
        requires_mate = source.configuration.get("reproduce_requires_mate")
    else:
        raw_max_births = getattr(source.configuration, "max_births_per_tick", None)
        requires_mate = getattr(source.configuration, "reproduce_requires_mate", None)
    energy_shape = _shape(source.live_state["energy"], path="live_state.energy")
    if len(energy_shape) != 2 or energy_shape[0] != batch_cells:
        raise _fail(
            "live_state.energy must have shape (batch_cells, slot_capacity), "
            f"got {energy_shape}"
    )
    slot_capacity = energy_shape[1]
    max_births = validate_asexual_birth_bound_v1(raw_max_births, slot_capacity)
    population = _config_value(source.configuration, "population_size", int)
    inheritance_profile = (
        source.configuration.get("inheritance_profile", "legacy_midpoint_v1")
        if isinstance(source.configuration, Mapping)
        else getattr(
            source.configuration,
            "inheritance_profile",
            "legacy_midpoint_v1",
        )
    )
    diploid_route = inheritance_profile == "diploid_linkage_v1"
    if type(source.allow_partial_generation_source_cursor) is not bool:
        raise _fail("partial-generation source authority must be an exact bool")
    if (
        not 0 <= start_tick < ticks_per_generation
        or (start_tick != 0 and not source.allow_partial_generation_source_cursor)
    ):
        raise _fail("production configuration.start_tick must equal 0")
    if type(ordinary_births_enabled) is not bool:
        raise _fail(
            "production configuration.ordinary_births_enabled must be exact bool"
        )
    life_history_mode = _config_value(source.configuration, "life_history_mode", int)
    if life_history_mode not in {0, 1}:
        raise _fail("production configuration.life_history_mode must be 0 or 1")
    for name in (
        "life_history_maturity_start_ticks",
        "life_history_maturity_rise_ticks",
        "life_history_fertility_decline_start_ticks",
        "life_history_fertility_decline_ticks",
        "life_history_senescence_start_ticks",
        "life_history_senescence_ramp_ticks",
        "life_history_max_age_ticks",
    ):
        if _config_value(source.configuration, name, int) < 0:
            raise _fail(f"production configuration.{name} must be non-negative")
    elder_floor = _config_value(
        source.configuration, "life_history_elder_fertility_floor", float
    )
    if not (math.isfinite(elder_floor) and 0.0 <= elder_floor <= 1.0):
        raise _fail(
            "production configuration.life_history_elder_fertility_floor "
            "must be finite in [0, 1]"
        )
    for name in (
        "life_history_senescence_energy_cost",
        "life_history_senescence_health_cost",
    ):
        value = _config_value(source.configuration, name, float)
        if not (math.isfinite(value) and value >= 0.0):
            raise _fail(f"production configuration.{name} must be finite and non-negative")
    if turnover_selection_mode not in {0, 1}:
        raise _fail(
            "production configuration.turnover_selection_mode must be 0 or 1"
        )
    if type(requires_mate) is not bool:
        raise _fail("production configuration.reproduce_requires_mate must be boolean")
    if diploid_route and not requires_mate:
        raise _fail("diploid direct resident births require profile-matched mate mode")
    if requires_mate and max_births <= 2:
        raise _fail(
            "mate-required direct resident births require runtime-sized "
            "three-role scratch (max_births_per_tick > 2)"
        )
    if population not in {64, 128, 192, 256, 512}:
        raise _fail(
            "production configuration.population_size must equal 64, 128, 192, 256, or 512"
        )
    if population > slot_capacity:
        raise _fail(
            "production configuration.population_size must not exceed "
            "resident slot capacity"
        )

    scratch = source.ordinary_birth_scratch_tensors[0]
    for name in (
        "mlp_pack",
        "hmem",
        "rng_manager_mutation_torch_cpu_state",
    ):
        if name not in source.live_banks:
            raise _fail(
                f"live_banks is missing {name} required by birth scratch layout v2"
            )
    mlp_shape = _shape(source.live_banks["mlp_pack"], path="live_banks.mlp_pack")
    hmem_shape = _shape(source.live_banks["hmem"], path="live_banks.hmem")
    torch_state_shape = _shape(
        source.live_banks["rng_manager_mutation_torch_cpu_state"],
        path="live_banks.rng_manager_mutation_torch_cpu_state",
    )
    if len(mlp_shape) != 3 or len(hmem_shape) != 3 or len(torch_state_shape) != 2:
        raise _fail("birth scratch layout v2 source tensors have invalid ranks")
    scratch_bytes = required_lifecycle_scratch_bytes_v1(
        max_births,
        slot_capacity,
        mlp_shape[-1],
        hmem_shape[-1],
        torch_state_shape[-1],
    )
    if len(source.ordinary_birth_state_tensors) in {122, 139}:
        scratch_bytes += CANONICAL_DIGEST_TEMPLATE_PACK_BYTES
    _require_shape(
        scratch,
        (batch_cells, scratch_bytes),
        path="ordinary_birth_scratch_tensors[0]",
    )


def _physical_views(source: DirectResidentSourceV1) -> tuple[Any, Any, Any, list[Any]]:
    generation_window = chunk_runtime.generation_slice(
        source.physical_window_generations,
        0,
        source.physical_window_generations,
    )
    tick_window = chunk_runtime.tick_slice(
        source.physical_window_generations,
        source.ticks_per_generation,
        0,
        source.physical_window_generations,
    )
    physical_generation_summary = source.outputs["generation_summary_scalar"][
        generation_window
    ]
    physical_runtime_bookkeeping = source.outputs["runtime_bookkeeping_by_tick"][
        :, tick_window
    ]
    physical_patch_stats = source.patch_p2_stats_tensors[0][:, tick_window]
    physical_full_tick_outputs = list(source.full_tick_output_tensors)
    physical_full_tick_outputs[19] = physical_generation_summary
    return (
        physical_generation_summary,
        physical_runtime_bookkeeping,
        physical_patch_stats,
        physical_full_tick_outputs,
    )


def _assemble_arguments(
    source: DirectResidentSourceV1,
    *,
    physical_runtime_bookkeeping: Any,
    physical_patch_stats: Any,
    physical_full_tick_outputs: Sequence[Any],
) -> tuple[Any, ...]:
    pre_config = list(
        _typed_config_values(
            source.configuration,
            PRE_GENERATION_CONFIG_ARGUMENTS_V1,
        )
    )
    move_cost_index = tuple(name for name, _ in PRE_GENERATION_CONFIG_ARGUMENTS_V1).index(
        "move_cost"
    )
    move_cost_by_cell = source.move_cost_by_cell
    if move_cost_by_cell is None:
        energy = _mapping_value(source.live_state, "energy", path="live_state")
        device = getattr(energy, "device", "cpu")
        move_cost_by_cell = torch.full(
            (source.batch_cells,),
            pre_config[move_cost_index],
            dtype=torch.float64,
            device=device,
        )
    pre_config[move_cost_index] = move_cost_by_cell
    post_config = _typed_config_values(
        source.configuration,
        POST_GENERATION_CONFIG_ARGUMENTS_V1,
    )
    reporting_config = _typed_config_values(
        source.configuration,
        REPORTING_FITNESS_ARGUMENTS_V1,
    )
    return (
        source.active_agent_ids,
        source.active_agent_slots,
        source.active_count_state,
        *(
            _mapping_value(source.live_state, name, path="live_state")
            for name in LIVE_STATE_ARGUMENT_FIELDS_V1
        ),
        *(
            _mapping_value(source.outputs, name, path="outputs")
            for name in PREFIX_OUTPUT_ARGUMENT_FIELDS_V1
        ),
        physical_runtime_bookkeeping,
        *(
            _mapping_value(source.outputs, name, path="outputs")
            for name in POST_BOOKKEEPING_OUTPUT_ARGUMENT_FIELDS_V1
        ),
        source.full_tick_state_tensors,
        physical_full_tick_outputs,
        source.compact_summary_reporting_tensors,
        source.compact_summary_checkpoint_source_tensors,
        source.compact_summary_checkpoint_descriptor_specs,
        source.compact_summary_checkpoint_output_tensors,
        source.compact_summary_settlement_tensors,
        source.event_state_tensors,
        source.ordinary_birth_state_tensors,
        source.ordinary_birth_output_tensors,
        source.ordinary_birth_scratch_tensors,
        source.turnover_selection_seed_by_cell,
        source.turnover_state_tensors,
        source.turnover_shadow_tensors,
        source.turnover_output_tensors,
        source.turnover_pointer_table_tensors,
        source.patch_p2_state_tensors,
        source.patch_p2_scratch_tensors,
        [physical_patch_stats],
        source.patch_p2_config,
        source.ticks_per_generation,
        *pre_config[:4],
        source.deterministic,
        *pre_config[4:],
        source.generation_base,
        *post_config,
        source.physical_window_generations,
        *reporting_config,
        _mapping_value(source.outputs, "first_failure_records", path="outputs"),
    )


_SCALAR_ARGUMENT_TYPES: Final[dict[int, type]] = {
    _argument_index("ticks_per_generation"): int,
    _argument_index("deterministic"): bool,
    GENERATION_ARGUMENT_INDEX_V1: int,
    GENERATION_COUNT_ARGUMENT_INDEX_V1: int,
}
for _name, _value_type in PRE_GENERATION_CONFIG_ARGUMENTS_V1:
    if _name != "move_cost":
        _SCALAR_ARGUMENT_TYPES[_argument_index(f"config.{_name}")] = _value_type
for _name, _value_type in POST_GENERATION_CONFIG_ARGUMENTS_V1:
    _SCALAR_ARGUMENT_TYPES[_argument_index(f"config.{_name}")] = _value_type
for _name, _value_type in REPORTING_FITNESS_ARGUMENTS_V1:
    _SCALAR_ARGUMENT_TYPES[_argument_index(f"reporting_fitness.{_name}")] = _value_type


def validate_direct_resident_argument_abi_v1(
    arguments: Sequence[Any],
    *,
    expected_arguments: Sequence[Any] | None = None,
) -> tuple[Any, ...]:
    """Validate the exact outer and nested production ABI."""

    normalized = tuple(arguments)
    if len(normalized) != DIRECT_ARGUMENT_COUNT:
        raise _fail(
            f"direct resident ABI requires exactly {DIRECT_ARGUMENT_COUNT} arguments"
        )
    if any(value is None for value in normalized):
        null_indices = [
            index for index, value in enumerate(normalized) if value is None
        ]
        raise _fail(f"direct resident ABI contains null arguments at {null_indices}")

    for index, expected_type in _SCALAR_ARGUMENT_TYPES.items():
        if type(normalized[index]) is not expected_type:
            name = DIRECT_RESIDENT_ARGUMENT_NAMES_V1[index]
            raise _fail(
                f"argument {index} ({name}) must be exact {expected_type.__name__}"
            )

    ordinary_births_enabled = normalized[
        _argument_index("config.ordinary_births_enabled")
    ]
    generation_turnover_enabled = normalized[
        _argument_index("config.generation_turnover_enabled")
    ]
    sequence_lengths = {
        _argument_index("full_tick_state_tensors"): 29,
        _argument_index("compact_summary_reporting_tensors"): 0,
        _argument_index("compact_summary_checkpoint_source_tensors"): 0,
        _argument_index("compact_summary_checkpoint_output_tensors"): 0,
        _argument_index("compact_summary_settlement_tensors"): 0,
        EVENT_STATE_ARGUMENT_INDEX_V1: 31,
        _argument_index("ordinary_birth_output_tensors"): (
            ORDINARY_BIRTH_OUTPUT_COUNT_V1
            if ordinary_births_enabled
            else 0
        ),
        _argument_index("ordinary_birth_scratch_tensors"): 1,
        _argument_index("turnover_state_tensors"): (
            122 if generation_turnover_enabled else 0
        ),
        _argument_index("turnover_shadow_tensors"): (
            122 if generation_turnover_enabled else 0
        ),
        _argument_index("turnover_output_tensors"): TURNOVER_OUTPUT_COUNT_V1,
        _argument_index("turnover_pointer_table_tensors"): (
            1 if generation_turnover_enabled else 0
        ),
        _argument_index("patch_p2_state_tensors"): len(PATCH_P2_STATE_FIELDS_V1),
        _argument_index("patch_p2_scratch_tensors"): 1,
        _argument_index("physical_patch_stats_tensors"): 1,
    }
    for index, length in sequence_lengths.items():
        _require_sequence(
            normalized[index],
            path=f"arguments[{index}] ({DIRECT_RESIDENT_ARGUMENT_NAMES_V1[index]})",
            length=length,
        )
    _require_sequence_length_in(
        normalized[_argument_index("physical_full_tick_output_tensors")],
        path=(
            f"arguments[{_argument_index('physical_full_tick_output_tensors')}] "
            "(physical_full_tick_output_tensors)"
        ),
        lengths=frozenset({20, 27}),
    )
    ordinary_birth_state = normalized[ORDINARY_BIRTH_STATE_ARGUMENT_INDEX_V1]
    if (
        not isinstance(ordinary_birth_state, Sequence)
        or isinstance(ordinary_birth_state, (str, bytes, bytearray))
        or len(ordinary_birth_state) not in ORDINARY_BIRTH_STATE_COUNTS_V1
        or any(item is None for item in ordinary_birth_state)
    ):
        raise _fail(
            f"arguments[{ORDINARY_BIRTH_STATE_ARGUMENT_INDEX_V1}] "
            "(ordinary_birth_state_tensors) must contain "
            "exactly 113 legacy, 122 diploid, or 139 Genome V2 entries"
        )
    checkpoint_descriptor_index = _argument_index(
        "compact_summary_checkpoint_descriptor_specs"
    )
    _require_shape(
        normalized[checkpoint_descriptor_index],
        (0, 13),
        path=(
            f"arguments[{checkpoint_descriptor_index}] "
            "(compact_summary_checkpoint_descriptor_specs)"
        ),
    )
    energy_index = _argument_index("live_state.energy")
    energy_shape = _shape(
        normalized[energy_index],
        path=f"arguments[{energy_index}] (live_state.energy)",
    )
    if len(energy_shape) != 2 or not 1 <= energy_shape[0] <= 256:
        raise _fail(
            f"arguments[{energy_index}] (live_state.energy) must have shape "
            "(batch_cells, slot_capacity) with batch_cells in [1,256]"
        )
    batch_cells = energy_shape[0]
    move_cost_index = _argument_index("config.move_cost")
    move_cost_by_cell = normalized[move_cost_index]
    if (
        not isinstance(move_cost_by_cell, torch.Tensor)
        or move_cost_by_cell.dtype != torch.float64
        or move_cost_by_cell.ndim != 1
        or tuple(move_cost_by_cell.shape) != (batch_cells,)
        or not move_cost_by_cell.is_contiguous()
        or not bool(torch.isfinite(move_cost_by_cell).all().item())
        or bool(torch.any(move_cost_by_cell < 0.0).item())
    ):
        raise _fail(
            f"argument {move_cost_index} (config.move_cost) must be a contiguous "
            "finite nonnegative float64[batch_cells] tensor"
        )
    active_count_index = _argument_index("active_count_state")
    _require_shape(
        normalized[active_count_index],
        (batch_cells,),
        path=f"arguments[{active_count_index}] (active_count_state)",
    )
    active_ids_index = _argument_index("active_agent_ids")
    active_id_shape = _shape(
        normalized[active_ids_index],
        path=f"arguments[{active_ids_index}] (active_agent_ids)",
    )
    active_slots_index = _argument_index("active_agent_slots")
    _require_shape(
        normalized[active_slots_index],
        active_id_shape,
        path=f"arguments[{active_slots_index}] (active_agent_slots)",
    )
    ticks_index = _argument_index("ticks_per_generation")
    _require_admitted_ticks_per_generation_v1(
        normalized[ticks_index],
        path=f"arguments[{ticks_index}]",
    )
    physical_generation_count = _require_physical_window_generations_v1(
        normalized[GENERATION_COUNT_ARGUMENT_INDEX_V1],
        path=f"arguments[{GENERATION_COUNT_ARGUMENT_INDEX_V1}]",
    )
    physical_ticks = physical_generation_count * normalized[ticks_index]
    runtime_bookkeeping_index = _argument_index("physical_runtime_bookkeeping")
    _require_shape(
        normalized[runtime_bookkeeping_index],
        (batch_cells, physical_ticks, 5),
        path=(f"arguments[{runtime_bookkeeping_index}] (physical_runtime_bookkeeping)"),
    )
    full_tick_output_index = _argument_index("physical_full_tick_output_tensors")
    _require_shape(
        normalized[full_tick_output_index][19],
        (physical_generation_count, batch_cells, GENERATION_SUMMARY_WIDTH_V1),
        path=(f"arguments[{full_tick_output_index}][19] (physical_generation_summary)"),
    )
    patch_stats_index = _argument_index("physical_patch_stats_tensors")
    _require_shape(
        normalized[patch_stats_index][0],
        (
            batch_cells,
            physical_ticks,
            PATCH_STAT_COUNT_V1,
        ),
        path=f"arguments[{patch_stats_index}][0] (physical_patch_stats)",
    )
    patch_config_index = _argument_index("patch_p2_config")
    _validate_patch_p2_config(
        normalized[patch_config_index],
        path=f"arguments[{patch_config_index}] (patch_p2_config)",
        batch_cells=batch_cells,
    )
    deterministic_index = _argument_index("deterministic")
    if normalized[deterministic_index] is not False:
        raise _fail(
            f"arguments[{deterministic_index}] must select stochastic fixed-MLP policy"
        )
    if normalized[GENERATION_ARGUMENT_INDEX_V1] < 0:
        raise _fail(
            f"arguments[{GENERATION_ARGUMENT_INDEX_V1}] generation_base "
            "must be nonnegative"
        )
    if normalized[GENERATION_COUNT_ARGUMENT_INDEX_V1] != physical_generation_count:
        raise _fail(
            f"arguments[{GENERATION_COUNT_ARGUMENT_INDEX_V1}] "
            "physical_generation_count must equal its validated window"
        )
    _require_turnover_selection_seed_by_cell(
        normalized[TURNOVER_SELECTION_SEED_BY_CELL_ARGUMENT_INDEX_V1],
        batch_cells=batch_cells,
        path=(
            f"arguments[{TURNOVER_SELECTION_SEED_BY_CELL_ARGUMENT_INDEX_V1}] "
            "(turnover_selection_seed_by_cell)"
        ),
    )
    max_births_index = _argument_index("config.max_births_per_tick")
    abi_max_births = validate_asexual_birth_bound_v1(
        normalized[max_births_index],
        energy_shape[1],
        path=f"argument {max_births_index} (config.max_births_per_tick)",
    )
    requires_mate_index = _argument_index("config.reproduce_requires_mate")
    if type(normalized[requires_mate_index]) is not bool:
        raise _fail(
            f"argument {requires_mate_index} "
            "(config.reproduce_requires_mate) must be exact bool"
        )
    if normalized[requires_mate_index] and abi_max_births <= 2:
        raise _fail(
            "mate-required direct resident births require runtime-sized "
            "three-role scratch (max_births_per_tick > 2)"
        )
    selection_mode_index = _argument_index("config.turnover_selection_mode")
    lifecycle_mode_index = _argument_index("config.lifecycle_mode")
    if normalized[lifecycle_mode_index] not in {0, 1}:
        raise _fail(
            f"argument {lifecycle_mode_index} "
            "(config.lifecycle_mode) must be 0 or 1"
        )
    life_history_mode_index = _argument_index("config.life_history_mode")
    if normalized[life_history_mode_index] not in {0, 1}:
        raise _fail(
            f"argument {life_history_mode_index} "
            "(config.life_history_mode) must be 0 or 1"
        )
    for name in (
        "life_history_maturity_start_ticks",
        "life_history_maturity_rise_ticks",
        "life_history_fertility_decline_start_ticks",
        "life_history_fertility_decline_ticks",
        "life_history_senescence_start_ticks",
        "life_history_senescence_ramp_ticks",
        "life_history_max_age_ticks",
    ):
        index = _argument_index(f"config.{name}")
        if normalized[index] < 0:
            raise _fail(f"argument {index} (config.{name}) must be non-negative")
    elder_floor_index = _argument_index("config.life_history_elder_fertility_floor")
    if not (
        math.isfinite(normalized[elder_floor_index])
        and 0.0 <= normalized[elder_floor_index] <= 1.0
    ):
        raise _fail(
            f"argument {elder_floor_index} "
            "(config.life_history_elder_fertility_floor) must be finite in [0, 1]"
        )
    for name in (
        "life_history_senescence_energy_cost",
        "life_history_senescence_health_cost",
    ):
        index = _argument_index(f"config.{name}")
        if not (math.isfinite(normalized[index]) and normalized[index] >= 0.0):
            raise _fail(
                f"argument {index} (config.{name}) must be finite and non-negative"
            )
    if normalized[selection_mode_index] not in {0, 1}:
        raise _fail(
            f"argument {selection_mode_index} "
            "(config.turnover_selection_mode) must be 0 or 1"
        )
    population_index = _argument_index("config.population_size")
    if normalized[population_index] not in ADMITTED_INITIAL_LIVE_COUNTS_V1:
        raise _fail(
            f"argument {population_index} "
            "(config.population_size) must equal 64, 128, 192, 256, or 512"
        )
    start_tick_index = _argument_index("config.start_tick")
    if not 0 <= normalized[start_tick_index] < normalized[ticks_index]:
        raise _fail(
            f"argument {start_tick_index} (config.start_tick) must be within "
            "the configured lifecycle interval"
        )
    expected_production_config = {
        _argument_index("config.lifecycle_mode"): normalized[lifecycle_mode_index],
    }
    for index, expected_value in expected_production_config.items():
        if normalized[index] != expected_value:
            name = DIRECT_RESIDENT_ARGUMENT_NAMES_V1[index]
            raise _fail(f"argument {index} ({name}) must equal {expected_value!r}")
    population_index = _argument_index("config.population_size")
    argument_population = _validated_initial_live_count_v1(
        normalized[population_index],
        path=f"argument {population_index} (config.population_size)",
    )
    if argument_population > energy_shape[1]:
        raise _fail(
            f"argument {population_index} (config.population_size) must not "
            "exceed resident slot capacity"
        )
    turnover_index = _argument_index("config.generation_turnover_enabled")
    expected_turnover = normalized[lifecycle_mode_index] == 0
    if normalized[turnover_index] is not expected_turnover:
        mode_name = "legacy" if expected_turnover else "hybrid"
        raise _fail(
            f"argument {turnover_index} "
            "(config.generation_turnover_enabled) must equal "
            f"{expected_turnover!r} for {mode_name} lifecycle mode"
        )

    if expected_arguments is not None:
        expected = tuple(expected_arguments)
        if len(expected) != DIRECT_ARGUMENT_COUNT:
            raise _fail(
                f"expected_arguments must contain exactly {DIRECT_ARGUMENT_COUNT} "
                "entries"
            )
        mismatches: list[str] = []
        for index, (actual, expected_value) in enumerate(
            zip(normalized, expected, strict=True)
        ):
            if index in _SCALAR_ARGUMENT_TYPES:
                matches = (
                    type(actual) is type(expected_value) and actual == expected_value
                )
            elif index == move_cost_index:
                matches = (
                    isinstance(actual, torch.Tensor)
                    and isinstance(expected_value, torch.Tensor)
                    and torch.equal(actual, expected_value)
                )
            elif index == patch_stats_index:
                matches = (
                    isinstance(actual, Sequence)
                    and isinstance(expected_value, Sequence)
                    and len(actual) == len(expected_value)
                    and all(
                        actual_item is expected_item
                        for actual_item, expected_item in zip(
                            actual,
                            expected_value,
                            strict=True,
                        )
                    )
                )
            else:
                matches = actual is expected_value
            if not matches:
                mismatches.append(f"{index}:{DIRECT_RESIDENT_ARGUMENT_NAMES_V1[index]}")
        if mismatches:
            raise _fail(
                "direct resident ABI authority mismatch at " + ", ".join(mismatches)
            )
    return normalized


def prepare_direct_resident_trajectory_v1(
    source: DirectResidentSourceV1,
) -> PreparedDirectResidentTrajectoryV1:
    """Build and validate one dispatch-ready physical-g2 production ABI."""

    if not isinstance(source, DirectResidentSourceV1):
        raise _fail("source must be DirectResidentSourceV1")
    _validate_source_contract(source)
    (
        physical_generation_summary,
        physical_runtime_bookkeeping,
        physical_patch_stats,
        physical_full_tick_outputs,
    ) = _physical_views(source)
    arguments = _assemble_arguments(
        source,
        physical_runtime_bookkeeping=physical_runtime_bookkeeping,
        physical_patch_stats=physical_patch_stats,
        physical_full_tick_outputs=physical_full_tick_outputs,
    )
    validated_arguments = validate_direct_resident_argument_abi_v1(
        arguments,
        expected_arguments=arguments,
    )
    return PreparedDirectResidentTrajectoryV1(
        source=source,
        arguments=validated_arguments,
        physical_generation_summary=physical_generation_summary,
        physical_runtime_bookkeeping=physical_runtime_bookkeeping,
        physical_patch_stats=physical_patch_stats,
        physical_full_tick_output_tensors=physical_full_tick_outputs,
        live_banks=source.live_banks,
        checkpoint_projection_inputs=source.checkpoint_projection_inputs,
    )


def validate_prepared_direct_resident_trajectory_v1(
    prepared: PreparedDirectResidentTrajectoryV1,
) -> PreparedDirectResidentTrajectoryV1:
    """Revalidate a prepared trajectory against every source authority."""

    if not isinstance(prepared, PreparedDirectResidentTrajectoryV1):
        raise _fail("prepared must be PreparedDirectResidentTrajectoryV1")
    _validate_source_contract(prepared.source)
    expected_arguments = _assemble_arguments(
        prepared.source,
        physical_runtime_bookkeeping=prepared.physical_runtime_bookkeeping,
        physical_patch_stats=prepared.physical_patch_stats,
        physical_full_tick_outputs=prepared.physical_full_tick_output_tensors,
    )
    validate_direct_resident_argument_abi_v1(
        prepared.arguments,
        expected_arguments=expected_arguments,
    )
    if prepared.live_banks is not prepared.source.live_banks:
        raise _fail("prepared live_banks no longer retain the source authority")
    if (
        prepared.checkpoint_projection_inputs
        is not prepared.source.checkpoint_projection_inputs
    ):
        raise _fail(
            "prepared checkpoint_projection_inputs no longer retain source authority"
        )
    return prepared


__all__ = [
    "BATCH_CELLS_V1",
    "DIRECT_RESIDENT_ARGUMENT_NAMES_V1",
    "DirectResidentPreparationError",
    "DirectResidentSourceV1",
    "EVENT_STATE_ARGUMENT_INDEX_V1",
    "GENERATION_ARGUMENT_INDEX_V1",
    "GENERATION_COUNT_ARGUMENT_INDEX_V1",
    "LIVE_STATE_ARGUMENT_FIELDS_V1",
    "LOGICAL_GENERATIONS_V1",
    "ORDINARY_BIRTH_OUTPUT_COUNT_V1",
    "TURNOVER_OUTPUT_COUNT_V1",
    "ORDINARY_BIRTH_STATE_ARGUMENT_INDEX_V1",
    "ORDINARY_BIRTH_STATE_COUNTS_V1",
    "PATCH_P2_CONFIG_ARGUMENTS_V1",
    "PATCH_P2_STATE_FIELDS_V1",
    "POST_BOOKKEEPING_OUTPUT_ARGUMENT_FIELDS_V1",
    "POST_GENERATION_CONFIG_ARGUMENTS_V1",
    "PREFIX_OUTPUT_ARGUMENT_FIELDS_V1",
    "PRE_GENERATION_CONFIG_ARGUMENTS_V1",
    "PreparedDirectResidentTrajectoryV1",
    "REPORTING_FITNESS_ARGUMENTS_V1",
    "TICKS_PER_GENERATION_V1",
    "ADMITTED_TICKS_PER_GENERATION_V1",
    "MAX_TICKS_PER_GENERATION_V1",
    "prepare_direct_resident_trajectory_v1",
    "required_lifecycle_scratch_bytes_v1",
    "validate_asexual_birth_bound_v1",
    "validate_direct_resident_argument_abi_v1",
    "validate_prepared_direct_resident_trajectory_v1",
]
