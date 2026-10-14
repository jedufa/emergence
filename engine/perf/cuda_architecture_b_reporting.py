from __future__ import annotations

"""Strict post-generation materialization for the CUDA Architecture B sidecar."""

from dataclasses import dataclass
import math
from typing import Any, Final

import torch

from native_runtime.architecture_b_reporting import _public_output_counts
from native_runtime.engine import _compact_raw_result
from native_runtime.flat_population_reporting import (
    validated_native_flat_behavior_rows,
)
from perf.cuda_architecture_b_generation import (
    CudaGenerationConfig,
    CudaGenerationResult,
)


REPORTING_SCHEMA_VERSION: Final = (
    "eiel_cuda_architecture_b_generation_reporting_v1"
)

GENERATION_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "native_tick_loop_count",
    "death_rows",
    "committed_birth_rows",
    "failed_birth_rows",
    "public_failed_birth_rows",
    "birth_attempt_rows",
    "injury_blocked_births",
    "applied_actions",
    "action_move_total",
    "action_move_n",
    "action_move_s",
    "action_move_e",
    "action_move_w",
    "action_gather",
    "action_rest",
    "action_attack",
    "moved",
    "gathered",
    "movement_attempts",
    "movement_blocked_occupied",
    "movement_blocked_bounds",
    "birth_pre_cooldown",
    "birth_pre_no_energy",
    "birth_pre_injury_gate",
    "birth_pre_no_space",
    "birth_pre_no_mate",
    "birth_fail_no_space",
    "birth_fail_no_mate",
    "birth_fail_buffer",
    "birth_fail_injury_gate",
    "birth_fail_scar_gate",
    "birth_fail_conception",
    "birth_fail_gestation",
    "alive_count",
    "capacity",
    "free_slots",
    "typed_event_rows",
    "policy_loop_tick_calls",
    "policy_refill_calls",
    "policy_refill_rows",
    "cache_hit_rows",
    "policy_cache_hit_calls",
    "policy_cache_rows_requested",
    "policy_cache_rows_hit",
    "policy_cache_rows_missed",
    "policy_cache_rows_refilled",
    "policy_cache_full_refill_ticks",
    "policy_cache_partial_refill_ticks",
)

REPORTING_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "generation_turnover_disabled",
    2: "ordinary_births_not_supported",
    3: "partial_generation_not_supported",
    4: "policy_cache_failed",
    5: "generation_failed",
    6: "turnover_failed",
    7: "seven_action_policy_required",
}

_PER_AGENT_DTYPES: Final[dict[str, torch.dtype]] = {
    "reporting_agent_ids": torch.int64,
    "reporting_cohort_codes": torch.int32,
    "reporting_action_counts": torch.int64,
    "reporting_start_injury": torch.float32,
    "reporting_survival_steps": torch.int32,
    "reporting_energy_collected_exact": torch.float64,
    "reporting_offspring_count": torch.int32,
    "reporting_attacks": torch.int32,
    "reporting_alive_end": torch.bool,
    "reporting_injury_end": torch.float32,
    "reporting_aggression_bias": torch.float32,
    "reporting_early_survival_steps": torch.int32,
    "reporting_early_energy_collected_exact": torch.float64,
    "reporting_early_injury_gained": torch.float32,
    "reporting_early_life_weight": torch.float32,
    "reporting_action_switches": torch.int32,
    "reporting_action_oscillations": torch.int32,
    "reporting_drive_jitter": torch.float32,
    "reporting_selfharm_events": torch.int32,
    "reporting_pred_err_sum": torch.float32,
    "reporting_pred_err_count": torch.int32,
    "reporting_birth_reset_shock": torch.float32,
    "reporting_episode0_pred_err_sum": torch.float32,
    "reporting_episode0_pred_err_count": torch.int32,
    "reporting_episode0_selfharm_events": torch.int32,
    "reporting_late_pred_err_sum": torch.float32,
    "reporting_late_pred_err_count": torch.int32,
}


class CudaArchitectureBReportingError(RuntimeError):
    """Raised when the detached CUDA reporting sidecar is not exact."""


@dataclass(frozen=True)
class CudaArchitectureBReportingPayload:
    generation_result: dict[str, Any]
    compact_result: dict[str, Any]
    public_output_counts: dict[str, Any]
    behavior_rows: tuple[dict[str, Any], ...]
    selected_parent_agent_ids: tuple[int, ...]


def _fail(message: str) -> CudaArchitectureBReportingError:
    return CudaArchitectureBReportingError(
        f"CUDA Architecture B generation reporting: {message}"
    )


def _tensor(
    result: CudaGenerationResult,
    name: str,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    value = getattr(result, name, None)
    if not isinstance(value, torch.Tensor):
        raise _fail(f"{name} is missing")
    if value.dtype != dtype or tuple(value.shape) != shape:
        raise _fail(
            f"{name} must be {dtype} with shape {shape}, got "
            f"{value.dtype} {tuple(value.shape)}"
        )
    return value.detach().cpu()


def _scalar(
    result: CudaGenerationResult,
    name: str,
    *,
    dtype: torch.dtype,
) -> int | float | bool:
    return _tensor(result, name, dtype=dtype, shape=(1,)).tolist()[0]


def _base_fitness(
    *,
    survival: int,
    energy: float,
    offspring: int,
    attacks: int,
    early_survival: int,
    early_energy: float,
    early_injury: float,
    early_weight: float,
    alive_end: bool,
) -> float:
    score = min(float(survival), 120.0) * 0.05
    if offspring > 0:
        score += 180.0
    if offspring > 1:
        score += 120.0
    if offspring > 2:
        score += 60.0 * float(offspring - 2)
    score += (float(energy) / max(1.0, float(survival))) * 80.0
    score -= float(attacks) * 2.5
    weight = float(early_weight) if math.isfinite(early_weight) else 1.0
    early_component = (
        0.10 * float(early_survival)
        + 80.0 * (float(early_energy) / max(1.0, float(early_survival)))
        - 10.0 * float(early_injury)
    )
    score += weight * early_component
    if alive_end:
        score += 10.0
    return float(score)


def _effective_fitness(
    base: float,
    row: dict[str, int | float | bool],
    config: CudaGenerationConfig,
) -> float:
    episode0_pred = float(row["episode0_pred_err_sum"]) / max(
        1.0, float(row["episode0_pred_err_count"])
    )
    thrash = 0.5 * float(row["action_switches"]) + float(
        row["action_oscillations"]
    )
    drive_jitter = float(row["drive_jitter"])
    selfharm = float(row["selfharm_events"])
    episode0_selfharm = float(row["episode0_selfharm_events"])
    effective = float(base)

    split_episode0 = any(
        weight != 0.0
        for weight in (
            config.episode0_pred_err_fitness_weight,
            config.episode0_thrashing_fitness_weight,
            config.episode0_drive_jitter_fitness_weight,
            config.episode0_selfharm_fitness_weight,
        )
    )
    if split_episode0:
        pred_weight = config.episode0_pred_err_fitness_weight
        if config.disable_prediction_error_selection_pressure:
            pred_weight = 0.0
        effective += pred_weight * episode0_pred
        effective += config.episode0_thrashing_fitness_weight * thrash
        effective += config.episode0_drive_jitter_fitness_weight * drive_jitter
        effective += config.episode0_selfharm_fitness_weight * episode0_selfharm
    else:
        effective += config.episode0_fitness_weight * (
            episode0_pred + thrash + drive_jitter + episode0_selfharm
        )

    pred_weight = config.coherence_pred_err_weight
    episode0_pred_weight = config.coherence_episode0_pred_err_weight
    late_enabled = config.late_pred_cost_enable
    if config.disable_prediction_error_selection_pressure:
        pred_weight = 0.0
        episode0_pred_weight = 0.0
        late_enabled = False
    pred_mean = float(row["pred_err_sum"]) / max(
        1.0, float(row["pred_err_count"])
    )
    legacy_penalty = (
        config.coherence_thrashing_weight * thrash
        + config.coherence_drive_jitter_weight * drive_jitter
        + config.coherence_selfharm_weight * selfharm
        + pred_weight * pred_mean
        + config.coherence_reset_shock_weight * float(row["birth_reset_shock"])
        + episode0_pred_weight * episode0_pred
    )
    late_cost = 0.0
    late_count = float(row["late_pred_err_count"])
    if late_enabled and config.late_pred_cost_weight > 0.0 and late_count > 0.0:
        late_mean = float(row["late_pred_err_sum"]) / late_count
        if late_mean > 0.0:
            late_cost = config.late_pred_cost_weight * late_mean
    return max(0.0, effective - legacy_penalty - late_cost)


def _canonical_aggression(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return round(float(value) * 100_000_000.0) / 100_000_000.0


def _strict_scope(
    config: CudaGenerationConfig,
    *,
    phase_e_no_turnover: bool,
) -> None:
    if type(phase_e_no_turnover) is not bool:
        raise TypeError("phase_e_no_turnover must be an exact bool")
    if phase_e_no_turnover:
        if config.generation_turnover_enabled:
            raise _fail("Phase E reporting forbids generation turnover")
    elif not config.generation_turnover_enabled:
        raise _fail("generation turnover must be enabled")
    if config.max_births_per_tick != 0:
        raise _fail("ordinary births are outside the bounded reporting contract")
    if config.start_tick != 0:
        raise _fail("only complete generations starting at tick zero are supported")
    if config.action_count != 7:
        raise _fail("the canonical behavior schema requires seven actions")
    if not 0.0 <= config.parent_fraction <= 1.0:
        raise _fail("parent_fraction must be in [0, 1]")


def materialize_cuda_architecture_b_reporting(
    result: CudaGenerationResult,
    config: CudaGenerationConfig,
    *,
    phase_e_no_turnover: bool = False,
) -> CudaArchitectureBReportingPayload:
    """Synchronize and validate the immutable sidecar after native turnover."""

    _strict_scope(config, phase_e_no_turnover=phase_e_no_turnover)
    status = int(_scalar(result, "reporting_status", dtype=torch.int32))
    valid = bool(_scalar(result, "reporting_valid", dtype=torch.bool))
    if status not in REPORTING_STATUS:
        raise _fail(f"unknown native reporting status {status}")
    if status != 0 or not valid:
        raise _fail(f"native sidecar is invalid: {REPORTING_STATUS[status]}")

    count_tensor = _tensor(
        result,
        "reporting_generation_counts",
        dtype=torch.int64,
        shape=(len(GENERATION_COUNT_FIELDS),),
    )
    counts = {
        name: int(value)
        for name, value in zip(
            GENERATION_COUNT_FIELDS, count_tensor.tolist(), strict=True
        )
    }
    runtime_bookkeeping = _tensor(
        result,
        "runtime_bookkeeping_by_tick",
        dtype=torch.int64,
        shape=(config.steps, 5),
    )
    runtime_totals = runtime_bookkeeping.sum(0).tolist()
    for index, name in enumerate(
        (
            "applied_actions",
            "moved",
            "gathered",
            "policy_refill_calls",
            "policy_refill_rows",
        )
    ):
        if int(runtime_totals[index]) != counts[name]:
            raise _fail(f"runtime bookkeeping does not sum to {name}")
    food_spawn_stats = _tensor(
        result,
        "food_spawn_stats_by_tick",
        dtype=torch.int64,
        shape=(config.steps, 3),
    )
    if bool((food_spawn_stats < 0).any().item()):
        raise _fail("food spawn counters must be nonnegative")
    food_spawn_attempts = int(food_spawn_stats[:, 0].sum().item())
    food_spawned_rows = int(food_spawn_stats[:, 1].sum().item())
    patch_spawn_sync_rows = int(food_spawn_stats[:, 2].sum().item())
    if food_spawned_rows > food_spawn_attempts:
        raise _fail("food spawn placements exceed attempts")
    if not config.patch_enabled and patch_spawn_sync_rows != 0:
        raise _fail("Flat generation reported Patch spawn synchronization")
    if any(value < 0 for value in counts.values()):
        raise _fail("generation summary counters must be nonnegative")
    if counts["native_tick_loop_count"] != config.steps:
        raise _fail("native tick loop count does not match configured steps")

    agent_ids_tensor = getattr(result, "reporting_agent_ids", None)
    if not isinstance(agent_ids_tensor, torch.Tensor) or agent_ids_tensor.ndim != 1:
        raise _fail("reporting_agent_ids must be rank one")
    row_count = int(agent_ids_tensor.numel())
    values: dict[str, list[Any]] = {}
    for name, dtype in _PER_AGENT_DTYPES.items():
        shape = (row_count, 7) if name == "reporting_action_counts" else (row_count,)
        values[name] = _tensor(
            result, name, dtype=dtype, shape=shape
        ).tolist()

    agent_ids = [int(value) for value in values["reporting_agent_ids"]]
    if any(agent_id < 0 for agent_id in agent_ids):
        raise _fail("reporting agent IDs must be nonnegative")
    if agent_ids != sorted(set(agent_ids)):
        raise _fail("reporting agent IDs must be unique and strictly increasing")
    cohort_codes = [int(value) for value in values["reporting_cohort_codes"]]
    if any(code != 0 for code in cohort_codes):
        raise _fail("ordinary born-cohort rows are outside the bounded contract")

    action_counts = values["reporting_action_counts"]
    if any(int(value) < 0 for row in action_counts for value in row):
        raise _fail("per-agent action counters must be nonnegative")
    action_columns = [
        sum(int(row[column]) for row in action_counts) for column in range(7)
    ]
    expected_action_columns = [
        counts["action_move_n"],
        counts["action_move_s"],
        counts["action_move_e"],
        counts["action_move_w"],
        counts["action_gather"],
        counts["action_rest"],
        counts["action_attack"],
    ]
    if action_columns != expected_action_columns:
        raise _fail("per-agent and generation action counters disagree")
    if sum(action_columns) != counts["applied_actions"]:
        raise _fail("applied action count does not match the seven action bins")
    if sum(action_columns[:4]) != counts["action_move_total"]:
        raise _fail("directional movement counts do not match action_move_total")
    if counts["movement_attempts"] != counts["action_move_total"]:
        raise _fail("movement attempts do not match movement action rows")
    if (
        counts["moved"]
        + counts["movement_blocked_occupied"]
        + counts["movement_blocked_bounds"]
        != counts["movement_attempts"]
    ):
        raise _fail("movement success and blockers do not partition attempts")
    if counts["capacity"] != counts["alive_count"] + counts["free_slots"]:
        raise _fail("capacity does not equal alive plus free slots")

    birth_fields = (
        "committed_birth_rows",
        "failed_birth_rows",
        "public_failed_birth_rows",
        "birth_attempt_rows",
        "injury_blocked_births",
        "birth_pre_cooldown",
        "birth_pre_no_energy",
        "birth_pre_injury_gate",
        "birth_pre_no_space",
        "birth_pre_no_mate",
        "birth_fail_no_space",
        "birth_fail_no_mate",
        "birth_fail_buffer",
        "birth_fail_injury_gate",
        "birth_fail_scar_gate",
        "birth_fail_conception",
        "birth_fail_gestation",
    )
    if any(counts[name] != 0 for name in birth_fields):
        raise _fail("birth counters must be zero when ordinary births are disabled")

    if counts["policy_loop_tick_calls"] != config.steps:
        raise _fail("policy loop calls do not match configured steps")
    if counts["policy_cache_rows_requested"] != counts["applied_actions"]:
        raise _fail("policy cache requests do not match applied actions")
    if (
        counts["policy_cache_rows_requested"]
        != counts["policy_cache_rows_hit"] + counts["policy_cache_rows_missed"]
    ):
        raise _fail("policy cache hits and misses do not partition requests")
    if counts["cache_hit_rows"] != counts["policy_cache_rows_hit"]:
        raise _fail("legacy and public policy cache hit rows disagree")
    if (
        counts["policy_cache_rows_missed"]
        != counts["policy_cache_rows_refilled"]
        or counts["policy_cache_rows_refilled"] != counts["policy_refill_rows"]
    ):
        raise _fail("policy cache misses and refilled rows disagree")
    if (
        counts["policy_cache_full_refill_ticks"]
        + counts["policy_cache_partial_refill_ticks"]
        != counts["policy_refill_calls"]
    ):
        raise _fail("full and partial refill ticks do not partition refill calls")
    if (
        counts["policy_cache_hit_calls"] + counts["policy_refill_calls"]
        > counts["policy_loop_tick_calls"]
    ):
        raise _fail("policy hit/refill calls exceed loop calls")

    row_stats: list[dict[str, int | float | bool]] = []
    for index in range(row_count):
        row_stats.append(
            {
                "action_switches": int(values["reporting_action_switches"][index]),
                "action_oscillations": int(
                    values["reporting_action_oscillations"][index]
                ),
                "drive_jitter": float(values["reporting_drive_jitter"][index]),
                "selfharm_events": int(values["reporting_selfharm_events"][index]),
                "pred_err_sum": float(values["reporting_pred_err_sum"][index]),
                "pred_err_count": int(values["reporting_pred_err_count"][index]),
                "birth_reset_shock": float(
                    values["reporting_birth_reset_shock"][index]
                ),
                "episode0_pred_err_sum": float(
                    values["reporting_episode0_pred_err_sum"][index]
                ),
                "episode0_pred_err_count": int(
                    values["reporting_episode0_pred_err_count"][index]
                ),
                "episode0_selfharm_events": int(
                    values["reporting_episode0_selfharm_events"][index]
                ),
                "late_pred_err_sum": float(
                    values["reporting_late_pred_err_sum"][index]
                ),
                "late_pred_err_count": int(
                    values["reporting_late_pred_err_count"][index]
                ),
            }
        )

    base_fitness: list[float] = []
    effective_fitness: list[float] = []
    for index, stats in enumerate(row_stats):
        base = _base_fitness(
            survival=int(values["reporting_survival_steps"][index]),
            energy=float(values["reporting_energy_collected_exact"][index]),
            offspring=int(values["reporting_offspring_count"][index]),
            attacks=int(values["reporting_attacks"][index]),
            early_survival=int(
                values["reporting_early_survival_steps"][index]
            ),
            early_energy=float(
                values["reporting_early_energy_collected_exact"][index]
            ),
            early_injury=float(values["reporting_early_injury_gained"][index]),
            early_weight=float(values["reporting_early_life_weight"][index]),
            alive_end=bool(values["reporting_alive_end"][index]),
        )
        base_fitness.append(base)
        effective_fitness.append(_effective_fitness(base, stats, config))

    eligible = [
        index
        for index in range(row_count)
        if cohort_codes[index] == 0
        and bool(values["reporting_alive_end"][index])
    ]
    ranked = sorted(
        eligible, key=lambda index: effective_fitness[index], reverse=True
    )
    parent_count = min(
        len(eligible),
        max(2, int(float(len(eligible)) * config.parent_fraction)),
    )
    selected = set(ranked[:parent_count])

    behavior_rows_raw: list[list[Any]] = []
    for index, agent_id in enumerate(agent_ids):
        actions = [int(value) for value in action_counts[index]]
        alive_end = bool(values["reporting_alive_end"][index])
        injury_end = float(values["reporting_injury_end"][index])
        behavior_rows_raw.append(
            [
                config.generation,
                agent_id,
                "start",
                *actions,
                int(values["reporting_survival_steps"][index]),
                float(values["reporting_energy_collected_exact"][index]),
                int(values["reporting_offspring_count"][index]),
                int(values["reporting_attacks"][index]),
                base_fitness[index],
                int(alive_end),
                int(alive_end),
                int(index in selected),
                injury_end,
                injury_end - float(values["reporting_start_injury"][index]),
                _canonical_aggression(
                    float(values["reporting_aggression_bias"][index])
                ),
            ]
        )
    try:
        behavior_rows = validated_native_flat_behavior_rows(
            behavior_rows_raw, expected_generation=config.generation
        )
    except ValueError as exc:
        raise _fail(str(exc)) from exc

    alive_rows = sum(int(row["alive_end"]) for row in behavior_rows)
    if alive_rows != counts["alive_count"]:
        raise _fail("behavior alive rows disagree with pre-turnover alive count")
    if row_count - alive_rows != counts["death_rows"]:
        raise _fail("behavior terminal rows disagree with generation death count")

    peak_injury = float(
        _scalar(result, "reporting_peak_injury", dtype=torch.float64)
    )
    if not math.isfinite(peak_injury) or peak_injury < 0.0:
        raise _fail("peak injury must be finite and nonnegative")
    observed_injury = [
        *[float(value) for value in values["reporting_start_injury"]],
        *[float(value) for value in values["reporting_injury_end"]],
    ]
    if observed_injury and peak_injury < max(observed_injury):
        raise _fail("peak injury is below an observed start/end injury")

    generation_result: dict[str, Any] = {
        "schema_version": REPORTING_SCHEMA_VERSION,
        "decision": "ARCHB-CUDA-STAGE1-BOUNDED-REPORTING",
        "executed": True,
        "hidden_fallback": False,
        "native_owner": True,
        "single_native_generation_call": True,
        "generation_index": config.generation,
        "steps_per_generation": config.steps,
        "peak_injury": peak_injury,
        "behavior_rows": behavior_rows_raw,
        "food_spawn_attempts": food_spawn_attempts,
        "food_spawned_rows": food_spawned_rows,
        **counts,
    }
    if config.patch_enabled:
        generation_result["patch_spawn_sync_rows"] = patch_spawn_sync_rows
    try:
        compact_result = _compact_raw_result(generation_result)
        public_output_counts = _public_output_counts(
            generation_result, compact_result
        )
    except (RuntimeError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    return CudaArchitectureBReportingPayload(
        generation_result=generation_result,
        compact_result=compact_result,
        public_output_counts=public_output_counts,
        behavior_rows=behavior_rows,
        selected_parent_agent_ids=tuple(
            sorted(agent_ids[index] for index in selected)
        ),
    )


__all__ = [
    "CudaArchitectureBReportingError",
    "CudaArchitectureBReportingPayload",
    "GENERATION_COUNT_FIELDS",
    "REPORTING_SCHEMA_VERSION",
    "REPORTING_STATUS",
    "materialize_cuda_architecture_b_reporting",
]
