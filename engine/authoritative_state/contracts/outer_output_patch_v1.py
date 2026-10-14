"""Exact metadata for the closed outer, birth-output, scratch, and Patch ABIs."""

from __future__ import annotations

from typing import Any, Final

from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    PATCH_P2_STATE_FIELDS_V1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1,
    GENOME_V2_CAUSAL_RECORD_WIDTH_V1,
    ORDINARY_BIRTH_OUTPUT_FIELDS,
    TURNOVER_OUTPUT_FIELDS,
)


SOURCE_ALLOCATOR: Final[str] = (
    "engine/perf/cuda_architecture_b_direct_resident_source_allocator_v1.py"
)
SOURCE_PREPARATION: Final[str] = (
    "engine/perf/cuda_architecture_b_direct_resident_preparation_v1.py"
)
SOURCE_BINDING: Final[str] = (
    "engine/perf/cuda_architecture_b_flattened_controller_v1_binding.cpp"
)

_CONTAINERS: Final[dict[str, tuple[str, int]]] = {
    "full_tick_state_tensors": ("full_tick_state", 29),
    "physical_full_tick_output_tensors": ("full_tick_output", 20),
    "compact_summary_reporting_tensors": ("compact_summary_reporting", 0),
    "compact_summary_checkpoint_source_tensors": (
        "compact_summary_checkpoint_source",
        0,
    ),
    "compact_summary_checkpoint_output_tensors": (
        "compact_summary_checkpoint_output",
        0,
    ),
    "compact_summary_settlement_tensors": ("compact_summary_settlement", 0),
    "event_state_tensors": ("event_state", 31),
    "ordinary_birth_state_tensors": ("ordinary_birth_state", 113),
    "ordinary_birth_output_tensors": (
        "ordinary_birth_output",
        len(ORDINARY_BIRTH_OUTPUT_FIELDS),
    ),
    "ordinary_birth_scratch_tensors": ("ordinary_birth_scratch", 1),
    "turnover_state_tensors": ("turnover_mutable_state", 122),
    "turnover_shadow_tensors": ("turnover_shadow", 122),
    "turnover_output_tensors": (
        "turnover_output",
        len(TURNOVER_OUTPUT_FIELDS),
    ),
    "turnover_pointer_table_tensors": ("turnover_pointer_table", 1),
    "patch_p2_state_tensors": ("patch_p2_state", 10),
    "patch_p2_scratch_tensors": ("patch_p2_scratch", 1),
    "physical_patch_stats_tensors": ("patch_p2_stats", 1),
}

_SCALAR_TYPES: Final[dict[str, str]] = {
    "deterministic": "bool",
    "generation_base": "int64",
    "physical_generation_count": "int64",
    "ticks_per_generation": "int64",
}

_BOOL_CONFIGS: Final[frozenset[str]] = frozenset(
    {
        "config.enable_rest_charge",
        "config.env_policy_alias",
        "config.reproduce_requires_mate",
        "config.generation_turnover_enabled",
        "config.ordinary_births_enabled",
        "reporting_fitness.late_pred_cost_enable",
        "reporting_fitness.disable_prediction_error_selection_pressure",
    }
)
_INT_CONFIGS: Final[frozenset[str]] = frozenset(
    {
        "config.view_radius",
        "config.hidden_size",
        "config.action_count",
        "config.cache_width",
        "config.max_food",
        "config.episode0_k",
        "config.rest_streak_free",
        "config.developmental_maturity_age",
        "config.life_history_mode",
        "config.life_history_maturity_start_ticks",
        "config.life_history_maturity_rise_ticks",
        "config.life_history_fertility_decline_start_ticks",
        "config.life_history_fertility_decline_ticks",
        "config.life_history_senescence_start_ticks",
        "config.life_history_senescence_ramp_ticks",
        "config.life_history_max_age_ticks",
        "config.start_tick",
        "config.max_births_per_tick",
        "config.reproduce_cooldown_steps",
        "config.genome_sensor_radius_max",
        "config.population_size",
        "config.turnover_selection_mode",
    }
)


def _row(
    *,
    name: str,
    kind: str,
    dtype: str,
    rank: int | None,
    shape: str,
    capacity: str,
    valid: str,
    mutability: str,
    stale_tail: str,
    checkpoint: str,
    chronicle: str,
    provider: str,
    analysis: str,
    profile: str,
    owner: str,
    source: str,
    boundary: str,
    sentinels: tuple[Any, ...] = (),
    **extra: Any,
) -> dict[str, Any]:
    physical = _physical_metadata(dtype)
    return {
        "name": name,
        "kind": kind,
        "dtype": dtype,
        "rank": rank,
        "shape_formula": shape,
        "capacity_formula": capacity,
        "valid_row_count_formula": valid,
        "mutability": mutability,
        "stale_tail_semantics": stale_tail,
        "checkpoint_participation": checkpoint,
        "chronicle_participation": chronicle,
        "provider_participation": provider,
        "analysis_participation": analysis,
        "profile_availability": profile,
        "semantic_owner": owner,
        "source": source,
        "boundary": boundary,
        **physical,
        "sentinels": list(sentinels),
        "metadata_status": "supported_exact",
        **extra,
    }


def _physical_metadata(dtype: str) -> dict[str, Any]:
    """Return representation bounds; semantic restrictions remain row-local."""

    integer = {
        "int32": ("signed", 32, "[-2147483648,2147483647]"),
        "int64": (
            "signed",
            64,
            "[-9223372036854775808,9223372036854775807]",
        ),
        "uint8": ("unsigned", 8, "[0,255]"),
        "uint64": ("unsigned", 64, "[0,18446744073709551615]"),
    }
    if dtype in integer:
        signedness, bit_width, valid_range = integer[dtype]
    elif dtype == "bool":
        signedness, bit_width, valid_range = "unsigned", 8, "{false,true}"
    elif dtype == "float32":
        signedness, bit_width, valid_range = (
            "signed",
            32,
            "IEEE-754 binary32 finite values and explicitly listed sentinels",
        )
    elif dtype == "float64":
        signedness, bit_width, valid_range = (
            "signed",
            64,
            "IEEE-754 binary64 finite values and explicitly listed sentinels",
        )
    else:
        signedness, bit_width, valid_range = (
            "not_applicable",
            None,
            "structural contract only",
        )
    return {
        "signedness": signedness,
        "bit_width": bit_width,
        "valid_range": valid_range,
    }


_LIVE_TYPES: Final[dict[str, tuple[str, str]]] = {
    "slot_to_agent_id": ("int64", "[B,S]"),
    "agent_id_to_slot": ("int32", "[B,I]"),
    "alive_mask": ("bool", "[B,S]"),
    "slot_tokens": ("int64", "[B,S]"),
    "action_cache_actions": ("int32", "[B,S]"),
    "action_cache_token": ("int64", "[B,S]"),
    "action_cache_cursor": ("int64", "[B,S]"),
    "action_cache_next": ("int64", "[B,S]"),
    "pos_y": ("int32", "[B,S]"),
    "pos_x": ("int32", "[B,S]"),
    "agent_grid": ("int32", "[B,H,W]"),
    "food_grid": ("int32", "[B,H,W]"),
    "energy": ("float32", "[B,S]"),
    "health": ("float32", "[B,S]"),
    "injury": ("float32", "[B,S]"),
    "policy_energy_observation": ("float32", "[B,S]"),
    "rest_streak": ("int32", "[B,S]"),
    "pending_free_agent_ids": ("int64", "[B,S]"),
    "pending_free_slots": ("int64", "[B,S]"),
    "pending_free_count": ("int64", "[B]"),
    "energy_collected": ("float32", "[B,S]"),
    "energy_collected_exact": ("float64", "[B,S]"),
    "last_action": ("int32", "[B,S]"),
    "last_last_action": ("int32", "[B,S]"),
    "action_switches": ("int32", "[B,S]"),
    "action_oscillations": ("int32", "[B,S]"),
    "episode0_switches": ("int32", "[B,S]"),
    "episode0_oscillations": ("int32", "[B,S]"),
    "survival_steps": ("int32", "[B,S]"),
    "mlp_pack": ("float32", "[B,S,P]"),
    "drives": ("float32", "[B,S,4]"),
    "attack_bias": ("float32", "[B,S]"),
    "action_temperature": ("float32", "[B,S]"),
    "drive_logit_gain": ("float32", "[B,S]"),
}

_OUTPUT_TYPES: Final[dict[str, tuple[str, str]]] = {
    "food_gained_scratch": ("float32", "[B,S]"),
    "ordered_agent_ids": ("int64", "[B,O,S]"),
    "ordered_slots": ("int32", "[B,O,S]"),
    "observations": ("float32", "[B,O,S,Q]"),
    "logits": ("float32", "[B,O,S,A]"),
    "row_actions": ("int32", "[B,O,S]"),
    "actions_by_slot": ("int32", "[B,O,S]"),
    "pre_energy": ("float32", "[B,O,S]"),
    "pre_health": ("float32", "[B,O,S]"),
    "pre_injury": ("float32", "[B,O,S]"),
    "pre_survival_steps": ("int32", "[B,O,S]"),
    "active_count": ("int64", "[B,O]"),
    "policy_refill_rows": ("int64", "[B,O]"),
    "policy_cache_hit_rows": ("int64", "[B,O]"),
    "death_agent_ids": ("int64", "[B,O,S]"),
    "death_slots": ("int64", "[B,O,S]"),
    "death_reason_codes": ("int32", "[B,O,S]"),
    "death_count": ("int64", "[B,O]"),
    "survivor_agent_ids": ("int64", "[B,O,S]"),
    "survivor_count": ("int64", "[B,O]"),
    "lifecycle_status": ("int32", "[B,O]"),
    "status": ("int32", "[B]"),
    "first_failure_records": ("int64", "[B,10]"),
}

_OUTER_OUTPUT_SENTINELS: Final[dict[str, tuple[Any, ...]]] = {
    "ordered_agent_ids": (-1,),
    "ordered_slots": (-1,),
    "row_actions": (-1,),
    "actions_by_slot": (5,),
    "pre_energy": ("uninitialized_until_native_write",),
    "pre_health": ("uninitialized_until_native_write",),
    "pre_injury": ("uninitialized_until_native_write",),
    "pre_survival_steps": ("uninitialized_until_native_write",),
    "death_agent_ids": (-1,),
    "death_slots": (-1,),
    "survivor_agent_ids": (-1,),
    "first_failure_records": (-1,),
}


def _outer_contract() -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for ordinal, name in enumerate(DIRECT_RESIDENT_ARGUMENT_NAMES_V1):
        common = dict(
            checkpoint="boundary_control_or_projection_input",
            chronicle="not_directly_serialized",
            provider="dispatch_input",
            analysis="not_directly_consumed",
            profile="patch_fixed_mlp_direct_resident_v1",
            owner="direct_resident_native_invocation",
            source=SOURCE_PREPARATION,
            boundary="pre_dispatch",
            ordinal=ordinal,
        )
        if name in _CONTAINERS:
            child, count = _CONTAINERS[name]
            birth_output_conditional = name == "ordinary_birth_output_tensors"
            rows.append(
                _row(
                    name=name,
                    kind="ordered_tensor_container",
                    dtype="ordered_tensor_references",
                    rank=None,
                    shape=(
                        "ordered[63] when config.ordinary_births_enabled "
                        "else ordered[0]"
                        if birth_output_conditional
                        else f"ordered[{count}]"
                    ),
                    capacity=(
                        "63 when config.ordinary_births_enabled else 0"
                        if birth_output_conditional
                        else str(count)
                    ),
                    valid=(
                        "63 when config.ordinary_births_enabled else 0"
                        if birth_output_conditional
                        else str(count)
                    ),
                    mutability="container_immutable_children_route_defined",
                    stale_tail="no_container_tail",
                    child_contract_family=child,
                    ordered_count=count,
                    activation=(
                        "config.ordinary_births_enabled"
                        if birth_output_conditional
                        else "always"
                    ),
                    **common,
                )
            )
            continue
        if name == "turnover_selection_seed_by_cell":
            rows.append(
                _row(
                    name=name,
                    kind="tensor",
                    dtype="uint64",
                    rank=1,
                    shape="[B]",
                    capacity="B",
                    valid="B",
                    mutability="read_only",
                    stale_tail="no tail; one immutable seed per physical cell",
                    **{
                        **common,
                        "checkpoint": "excluded_reconstructed_route_control",
                        "chronicle": "configuration_identity_only",
                        "provider": "dispatch_input_and_route_identity",
                        "owner": "production_lifecycle_selection_control",
                    },
                )
            )
            continue
        if name == "patch_p2_config":
            rows.append(
                _row(
                    name=name,
                    kind="ordered_mapping",
                    dtype="typed_scalar_mapping",
                    rank=None,
                    shape="mapping[33]",
                    capacity="33",
                    valid="33",
                    mutability="read_only",
                    stale_tail="no_tail",
                    **common,
                )
            )
            continue
        if name == "config.move_cost":
            rows.append(
                _row(
                    name=name,
                    kind="tensor",
                    dtype="float64",
                    rank=1,
                    shape="[B]",
                    capacity="B",
                    valid="B",
                    mutability="read_only",
                    stale_tail="no tail; one exact movement cost per physical cell",
                    **common,
                )
            )
            continue
        if name in _SCALAR_TYPES or name.startswith(("config.", "reporting_fitness.")):
            dtype = _SCALAR_TYPES.get(
                name,
                "bool" if name in _BOOL_CONFIGS else "int64" if name in _INT_CONFIGS else "float64",
            )
            rows.append(
                _row(
                    name=name,
                    kind="scalar",
                    dtype=dtype,
                    rank=0,
                    shape="[]",
                    capacity="1",
                    valid="1",
                    mutability="read_only",
                    stale_tail="no_tail",
                    **common,
                )
            )
            continue
        if name == "compact_summary_checkpoint_descriptor_specs":
            dtype, shape, mutability = "int64", "[0,13]", "read_only"
        elif name == "physical_runtime_bookkeeping":
            dtype, shape, mutability = "int64", "[B,2*T,5]", "write_only"
        elif name == "active_agent_ids":
            dtype, shape, mutability = "int64", "[B,S]", "read_write"
        elif name == "active_agent_slots":
            dtype, shape, mutability = "int32", "[B,S]", "read_write"
        elif name == "active_count_state":
            dtype, shape, mutability = "int64", "[B]", "read_write"
        elif name.startswith("live_state."):
            dtype, shape = _LIVE_TYPES[name.removeprefix("live_state.")]
            mutability = "read_write"
        elif name.startswith("outputs."):
            dtype, shape = _OUTPUT_TYPES[name.removeprefix("outputs.")]
            mutability = "write_only"
        else:  # The list above is intentionally exhaustive.
            raise RuntimeError(f"missing direct-resident outer contract for {name}")
        rows.append(
            _row(
                name=name,
                kind="tensor",
                dtype=dtype,
                rank=shape.count(",") + 1 if shape != "[]" else 0,
                shape=shape,
                capacity=f"product({shape})",
                valid="0" if mutability == "write_only" else "route_active_prefix_or_full_storage",
                mutability=mutability,
                stale_tail=(
                    "ignored_until_written; sentinel preserved"
                    if mutability == "write_only"
                    else "inactive capacity is non-semantic"
                ),
                sentinels=(
                    _OUTER_OUTPUT_SENTINELS.get(
                        name.removeprefix("outputs."), ()
                    )
                    if name.startswith("outputs.")
                    else ()
                ),
                **common,
            )
        )
    return tuple(rows)


_BIRTH_INT32: Final[frozenset[str]] = frozenset(
    {
        "birth_planned_repro_cooldown",
        "birth_planned_spawn_positions",
        "birth_decision_reason_codes",
        "birth_decision_candidate_counts",
        "birth_decision_selected_positions",
        "birth_plan_status",
        "birth_identity_status",
        "birth_scalar_status",
        "birth_drive_status",
        "birth_chemistry_status",
        "birth_policy_status",
        "birth_combined_status",
        "birth_decision_adjacent_counts",
        "birth_decision_spawn_candidate_counts",
        "birth_mate_decision_reason_codes",
    }
)
_BIRTH_FLOAT32: Final[frozenset[str]] = frozenset(
    {"birth_planned_child_energy", "birth_policy_mutation_delta"}
)
_BIRTH_FLOAT64: Final[frozenset[str]] = frozenset(
    {
        "birth_policy_mutation_delta_total",
        "birth_mate_planned_energy",
        "birth_mate_planned_child_energy_exact",
        "birth_mate_planned_parent_energy_after",
        "birth_mate_planned_mate_energy_after",
    }
)
_BIRTH_BOOL: Final[frozenset[str]] = frozenset(
    {
        "birth_decision_selected_reserved",
        "birth_policy_mutation_occurred",
        "birth_genome_v2_causal_record_valid",
        "birth_genome_v2_causal_capture_enabled",
    }
)
_BIRTH_MINUS_ONE_SENTINELS: Final[frozenset[str]] = frozenset(
    {
        "birth_planned_parent_slots",
        "birth_planned_child_slots",
        "birth_planned_spawn_positions",
        "birth_decision_parent_slots",
        "birth_decision_reason_codes",
        "birth_decision_selected_positions",
        "birth_committed_parent_agent_ids",
        "birth_committed_child_agent_ids",
        "birth_committed_child_slots",
        "birth_planned_mate_slots",
        "birth_committed_mate_agent_ids",
        "birth_decision_mate_slots",
        "birth_mate_decision_reason_codes",
    }
)
_BIRTH_NAN_SENTINELS: Final[frozenset[str]] = frozenset(
    {
        "birth_planned_child_energy",
        "birth_mate_planned_child_energy_exact",
        "birth_mate_planned_parent_energy_after",
        "birth_mate_planned_mate_energy_after",
    }
)


def _birth_shape(name: str) -> str:
    if name == "birth_slot_free_queue_head":
        return "[B]"
    if name == "birth_genome_v2_causal_record_rows":
        return (
            f"[B,{GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1},"
            f"{GENOME_V2_CAUSAL_RECORD_WIDTH_V1}]"
        )
    if name == "birth_genome_v2_causal_record_valid":
        return f"[B,{GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1}]"
    if name in {
        "birth_genome_v2_causal_record_count",
        "birth_genome_v2_causal_record_dropped_count",
        "birth_genome_v2_causal_capture_enabled",
    }:
        return "[B]"
    if name in {"birth_planned_repro_cooldown", "birth_mate_planned_energy"}:
        return "[B,S]"
    if name == "birth_planned_spawn_positions":
        return "[B,O,M,2]"
    if name == "birth_decision_selected_positions":
        return "[B,O,S,2]"
    if name in {"birth_reason_counts", "birth_mate_reason_counts"}:
        return "[B,O,10]" if name == "birth_reason_counts" else "[B,O,12]"
    if name in {"birth_planner_pcg64_state_before", "birth_planner_pcg64_buffer_before"}:
        return "[B,O,4]" if name.endswith("state_before") else "[B,O,2]"
    if name in {
        "birth_planned_parent_slots",
        "birth_planned_child_slots",
        "birth_planned_child_energy",
        "birth_committed_parent_agent_ids",
        "birth_committed_child_agent_ids",
        "birth_committed_child_slots",
        "birth_policy_mutation_occurred",
        "birth_policy_mutation_delta",
        "birth_planned_mate_slots",
        "birth_committed_mate_agent_ids",
        "birth_mate_planned_child_energy_exact",
        "birth_mate_planned_parent_energy_after",
        "birth_mate_planned_mate_energy_after",
    }:
        return "[B,O,M]"
    if name in {
        "birth_decision_parent_slots",
        "birth_decision_reason_codes",
        "birth_decision_candidate_counts",
        "birth_decision_draw_starts",
        "birth_decision_draw_ends",
        "birth_decision_selected_reserved",
        "birth_decision_mate_slots",
        "birth_decision_adjacent_counts",
        "birth_decision_spawn_candidate_counts",
        "birth_mate_decision_reason_codes",
    }:
        return "[B,O,S]"
    return "[B,O]"


def _birth_contract() -> tuple[dict[str, Any], ...]:
    rows = []
    for ordinal, name in enumerate(ORDINARY_BIRTH_OUTPUT_FIELDS):
        dtype = (
            "bool"
            if name in _BIRTH_BOOL
            else "float32"
            if name in _BIRTH_FLOAT32
            else "float64"
            if name in _BIRTH_FLOAT64
            else "int32"
            if name in _BIRTH_INT32
            else "uint64"
            if name == "birth_planner_pcg64_state_before"
            else "int64"
        )
        shape = _birth_shape(name)
        semantic_rows = (
            (
                "sum(birth_genome_v2_causal_record_count_by_cell)"
                f"*{GENOME_V2_CAUSAL_RECORD_WIDTH_V1}"
            )
            if name == "birth_genome_v2_causal_record_rows"
            else "sum(birth_genome_v2_causal_record_count_by_cell)"
            if name == "birth_genome_v2_causal_record_valid"
            else "B"
            if name
            in {
                "birth_genome_v2_causal_record_count",
                "birth_genome_v2_causal_record_dropped_count",
                "birth_genome_v2_causal_capture_enabled",
            }
            else
            "sum(birth_committed_count_by_cell_tick)"
            if "committed" in name or "mutation_" in name
            else "sum(birth_planned_count_by_cell_tick)"
            if "planned_" in name
            else "sum(active_count_by_cell)*O"
            if "decision_" in name
            else "B*O"
        )
        rows.append(
            _row(
                name=name,
                kind="tensor",
                dtype=dtype,
                rank=shape.count(",") + 1,
                shape=shape,
                capacity=f"product({shape})",
                valid=semantic_rows,
                mutability="write_only",
                stale_tail="only the count-defined prefix is semantic; tail retains allocator sentinel",
                checkpoint="projection_input_not_persisted_as_output_plane",
                chronicle="birth_attempt_or_commit_projection_source",
                provider="reproduction_attempt_projection_source",
                analysis="derived_reproduction_metrics_source",
                profile="ordinary_birth_enabled_or_compact_reporting",
                owner="native_ordinary_birth_lifecycle",
                source=SOURCE_ALLOCATOR,
                boundary="post_tick_or_post_physical_window",
                ordinal=ordinal,
                sentinels=(
                    ("INT64_MIN",)
                    if name == "birth_genome_v2_causal_record_rows"
                    else
                    (-1,)
                    if name in _BIRTH_MINUS_ONE_SENTINELS
                    else ("NaN",)
                    if name in _BIRTH_NAN_SENTINELS
                    else ()
                ),
            )
        )
    return tuple(rows)


def _turnover_output_contract() -> tuple[dict[str, Any], ...]:
    rows = []
    for ordinal, name in enumerate(TURNOVER_OUTPUT_FIELDS):
        if name == "turnover_counts":
            dtype, shape, valid = "int64", "[B,8]", "B*8"
        elif name == "turnover_status":
            dtype, shape, valid = "int32", "[B]", "B"
        elif name in {"turnover_next_generation", "turnover_next_tick"}:
            dtype, shape, valid = "int64", "[B]", "B"
        elif name == "turnover_genome_v2_causal_record_rows":
            dtype, shape, valid = (
                "int64",
                (
                    f"[B,{GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1},"
                    f"{GENOME_V2_CAUSAL_RECORD_WIDTH_V1}]"
                ),
                (
                    "sum(turnover_genome_v2_causal_record_count_by_cell)"
                    f"*{GENOME_V2_CAUSAL_RECORD_WIDTH_V1}"
                ),
            )
        elif name == "turnover_genome_v2_causal_record_valid":
            dtype, shape, valid = (
                "bool",
                f"[B,{GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1}]",
                "sum(turnover_genome_v2_causal_record_count_by_cell)",
            )
        elif name in {
            "turnover_genome_v2_causal_record_count",
            "turnover_genome_v2_causal_record_dropped_count",
            "turnover_genome_v2_causal_capture_enabled",
        }:
            dtype = (
                "bool"
                if name == "turnover_genome_v2_causal_capture_enabled"
                else "int64"
            )
            shape, valid = "[B]", "B"
        else:
            dtype, shape, valid = (
                "int64",
                "[B,S]",
                "sum(turnover_committed_count_by_cell)",
            )
        rows.append(
            _row(
                name=name,
                kind="tensor",
                dtype=dtype,
                rank=shape.count(",") + 1,
                shape=shape,
                capacity=f"product({shape})",
                valid=valid,
                mutability="write_only",
                stale_tail=(
                    "only the count-defined prefix is semantic; tail retains "
                    "allocator sentinel"
                ),
                checkpoint="projection_input_not_persisted_as_output_plane",
                chronicle="turnover_commit_projection_source",
                provider="generation_turnover_projection_source",
                analysis="derived_turnover_metrics_source",
                profile="generation_turnover_enabled",
                owner="native_generation_turnover_lifecycle",
                source=SOURCE_ALLOCATOR,
                boundary="post_generation_turnover",
                ordinal=ordinal,
                sentinels=(
                    ("INT64_MIN",)
                    if name == "turnover_genome_v2_causal_record_rows"
                    else (-1,)
                    if ordinal < 7
                    else ()
                ),
            )
        )
    return tuple(rows)


def _scratch_contract() -> tuple[dict[str, Any], ...]:
    return (
        _row(
            name="lifecycle_scratch_storage",
            kind="tensor",
            dtype="uint8",
            rank=2,
            shape="[B,lifecycle_scratch_bytes+digest_template_bytes]",
            capacity="B*(lifecycle_scratch_bytes+digest_template_bytes)",
            valid="B*(lifecycle_scratch_bytes+digest_template_bytes)",
            mutability="read_write_opaque_workspace",
            stale_tail="all bytes are opaque native workspace; no semantic tail",
            checkpoint="excluded",
            chronicle="excluded",
            provider="excluded",
            analysis="excluded",
            profile="birth_or_turnover_or_full_patch_lifecycle",
            owner="native_lifecycle_scratch_layout_v2",
            source=SOURCE_ALLOCATOR,
            boundary="invocation_local",
            ordinal=0,
            metadata_status="supported_exact",
        ),
    )


def _patch_contract() -> tuple[dict[str, Any], ...]:
    rows = []
    for ordinal, name in enumerate(PATCH_P2_STATE_FIELDS_V1):
        rows.append(
            _row(
                name=name,
                kind="tensor",
                dtype="float32",
                rank=3,
                shape="[B,H,W]",
                capacity="B*H*W",
                valid="B*H*W",
                mutability="read_write",
                stale_tail="no tail; every grid cell is authoritative",
                checkpoint="required_exact_tensor",
                chronicle="derived_patch_measurement_source",
                provider="resident_checkpoint_projection",
                analysis="patch_reporting_source",
                profile=(
                    "patch_p2_context_7,patch_p2_full_10"
                    if ordinal < 7
                    else "patch_p2_full_10"
                ),
                owner="native_patch_p2_resident_state",
                source=SOURCE_PREPARATION,
                boundary="pre_dispatch_and_terminal_checkpoint",
                ordinal=ordinal,
                variants=("patch_p2_state_context_7", "patch_p2_state_full_10")
                if ordinal < 7
                else ("patch_p2_state_full_10",),
            )
        )
    return tuple(rows)


def build_outer_output_patch_contracts_v1() -> dict[str, tuple[dict[str, Any], ...]]:
    """Return immutable-order, JSON-compatible production contract rows."""

    contracts = {
        "direct_resident_outer": _outer_contract(),
        "ordinary_birth_output": _birth_contract(),
        "turnover_output": _turnover_output_contract(),
        "ordinary_birth_scratch": _scratch_contract(),
        "patch_p2_state": _patch_contract(),
    }
    assert len(contracts["direct_resident_outer"]) == len(
        DIRECT_RESIDENT_ARGUMENT_NAMES_V1
    )
    assert len(contracts["ordinary_birth_output"]) == len(
        ORDINARY_BIRTH_OUTPUT_FIELDS
    )
    assert len(contracts["turnover_output"]) == len(TURNOVER_OUTPUT_FIELDS)
    assert len(contracts["ordinary_birth_scratch"]) == 1
    assert len(contracts["patch_p2_state"]) == 10
    return contracts


__all__ = ["build_outer_output_patch_contracts_v1"]
