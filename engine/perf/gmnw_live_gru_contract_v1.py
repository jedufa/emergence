"""Frozen contract and data split for live GMNW stable-GRU qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final, Mapping

from engine.evolution.gmnw_v1 import (
    GMNW_CUE_ACTIONS,
    GmnwConfig,
    balanced_cue_identity,
)
from engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1 import (
    GMNW_GRU_OBSERVATION_COLUMNS_V1,
    GMNW_GRU_OBSERVATION_REGISTRY_SHA256_V1,
    gmnw_gru_observation_contract_v1,
)


LIVE_CONTRACT_SCHEMA_V1: Final[str] = "eiel_gmnw_live_gru_contract_v1"
DATA_SPLIT_SCHEMA_V1: Final[str] = "eiel_gmnw_live_gru_data_split_v1"
DEVELOPMENT_SEEDS_V1: Final[tuple[int, ...]] = (101, 202, 7001)
QUALIFICATION_SEEDS_V1: Final[tuple[int, ...]] = (9101, 9102)
QUALIFICATION_EPISODES_PER_STREAM_V1: Final[int] = 48
INITIAL_STREAM_COUNT_V1: Final[int] = 64
TASK_CONFIGURATIONS_V1: Final[tuple[tuple[int, int, bool], ...]] = tuple(
    (cue_alphabet, delay_ticks, distractors_enabled)
    for cue_alphabet in (2, 4)
    for delay_ticks in (4, 12)
    for distractors_enabled in (False, True)
)


class GmnwLiveGruContractError(RuntimeError):
    """The frozen live capability contract or split is invalid."""


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _bind(payload: dict[str, Any], field: str) -> dict[str, Any]:
    if field in payload:
        raise GmnwLiveGruContractError(f"{field} is reserved for the canonical hash")
    return {**payload, field: canonical_sha256(payload)}


def live_contract_v1() -> dict[str, Any]:
    observation = gmnw_gru_observation_contract_v1()
    if (
        tuple(observation["columns"]) != GMNW_GRU_OBSERVATION_COLUMNS_V1
        or observation["registry_sha256"]
        != GMNW_GRU_OBSERVATION_REGISTRY_SHA256_V1
    ):
        raise GmnwLiveGruContractError("the 86-column observation registry drifted")
    payload: dict[str, Any] = {
        "schema_version": LIVE_CONTRACT_SCHEMA_V1,
        "decision": "GMNW-LIVE-CONTRACT-A/FROZEN-AND-HASHED",
        "world": "GRU_MEMORY_NECESSITY_WORLD_V1",
        "policy_family": "stable_gru",
        "observation": observation,
        "observation_roles": {
            "cue_columns": [74, 75, 76, 77],
            "phase_columns": [78, 79, 80, 81],
            "delay_state": "private delay_remaining; public phase_delay at column 79",
            "distractor_columns": [82, 83, 84, 85],
            "decision_mask": "alive active rows with phase_decision column 80 equal to 1",
            "witness_effective_columns": list(range(74, 86)),
            "witness_zero_padded_columns": list(range(74)),
        },
        "task_timing": {
            "cue_ticks": 1,
            "delay_ticks": [4, 12],
            "decision_ticks": 1,
            "consequence_ticks": 1,
            "episode_policy_steps": "delay_ticks + 3",
            "persistent_hidden_crosses_episode_boundaries": True,
            "generation_boundary_resets_hidden": False,
        },
        "actions": {
            "action_count": 7,
            "cue_to_private_action": list(GMNW_CUE_ACTIONS),
            "correctness": "selected_private_action equals cue_to_private_action[cue_identity] on decision rows",
            "public_action": 5,
            "public_action_name": "REST",
            "public_rest_neutralization_after_private_grading": True,
        },
        "hidden_state": {
            "dtype": "float32",
            "width": 64,
            "initialization": "zero for every new stream identity",
            "persistent_gru": "retain after every inference, episode, and generation",
            "reset_gru": "zero immediately before every inference",
            "clear_on_death_or_slot_reuse": True,
            "hidden_fallback": False,
        },
        "action_sampling": {
            "temperature": 0.05,
            "scaled_logits": "raw_logits / 0.05",
            "distribution": "argmax(scaled_logits + gumbel)",
            "gumbel_uniform_dtype": "numpy_float32",
            "gumbel_uniform_clamp": [1e-6, 1.0 - 1e-6],
            "draws_per_active_row": 7,
            "rng": "resident policy PCG64 in active-agent order",
            "cache_width": 1,
        },
        "qualification_rows": {
            "include": "completed decision rows emitted by alive active streams",
            "exclude": [
                "cue rows",
                "delay rows",
                "consequence rows",
                "dead rows",
                "episodes truncated before a decision",
                "rows beyond the first 48 completed decisions per initial stream",
            ],
            "accuracy": "correct qualifying decisions / qualifying decisions",
            "cue_balancing": "report per-cue accuracy and macro mean",
        },
        "pressure_schedule": {
            "grace_generations": [0, 1, 2, 3],
            "grace_incorrect_damage_fraction": 0.01,
            "grace_incorrect_reproduction_scale": 0.5,
            "severe_generations": [4, 5, 6, 7],
            "severe_incorrect_damage_fraction": 0.01,
            "severe_incorrect_reproduction_scale": 0.0,
            "lethal_generation_start": 8,
            "lethal_incorrect_damage_fraction": 1.0,
            "lethal_incorrect_reproduction_scale": 0.0,
            "correct_decision_reward_or_bonus": False,
        },
        "episode_settlement": {
            "order": [
                "private decision grading",
                "task health and reproduction update",
                "task death settlement",
                "public REST neutralization",
                "ordinary survivor lifecycle",
                "birth and turnover settlement",
            ],
            "next_episode_initialized_after_consequence": True,
            "new_stream_starts_fresh_balanced_cue_block": True,
        },
        "checkpoint": {
            "authority": "canonical Patch/GRU checkpoint plus GMNW sidecar",
            "gmnw_sidecar_alone_is_complete": False,
            "persistent_fields": [
                "alive and identity planes",
                "hidden state",
                "policy PCG64 state and buffer",
                "action cache",
                "GMNW private task tensors",
                "GMNW counters and damage totals",
                "per-cell task configuration and policy arm",
            ],
            "fresh_owner_continuation_required": True,
        },
        "causal_interventions": [
            "normal_hidden",
            "reset_before_decision",
            "shuffle_between_agents_with_different_cues",
            "shuffle_between_episodes",
            "inject_wrong_cue_hidden_before_decision",
        ],
        "thresholds": {
            "persistent_accuracy_minimum": 0.90,
            "reset_accuracy_maximum_above_chance": 0.10,
            "matched_mlp_accuracy_maximum_above_chance": 0.10,
            "observation_probe_accuracy_maximum_above_chance": 0.05,
            "persistent_survival_minimum": 0.80,
            "control_survival_maximum": 0.30,
            "intervention_accuracy_drop_minimum": 0.20,
            "wrong_cue_redirect_rate_minimum": 0.80,
        },
        "ownership": {
            "python_tick_ownership": False,
            "python_generation_ownership": False,
            "runtime_jit": False,
            "cpu_cuda_exactness_required": True,
        },
    }
    return _bind(payload, "contract_sha256")


def _cue_sequence_digest(*, seeds: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for seed in seeds:
        for cue_alphabet, delay_ticks, distractors_enabled in TASK_CONFIGURATIONS_V1:
            config = GmnwConfig(
                cue_alphabet=cue_alphabet,
                delay_ticks=delay_ticks,
                distractors_enabled=distractors_enabled,
                task_seed=seed,
            )
            for stream_id in range(INITIAL_STREAM_COUNT_V1):
                for episode_index in range(QUALIFICATION_EPISODES_PER_STREAM_V1):
                    row = (
                        seed,
                        cue_alphabet,
                        delay_ticks,
                        int(distractors_enabled),
                        stream_id,
                        episode_index,
                        balanced_cue_identity(
                            config,
                            stream_id=stream_id,
                            episode_index=episode_index,
                        ),
                    )
                    digest.update(json.dumps(row, separators=(",", ":")).encode("ascii"))
                    digest.update(b"\n")
    return digest.hexdigest()


def data_split_v1() -> dict[str, Any]:
    if set(DEVELOPMENT_SEEDS_V1) & set(QUALIFICATION_SEEDS_V1):
        raise GmnwLiveGruContractError("development and qualification seeds overlap")
    payload: dict[str, Any] = {
        "schema_version": DATA_SPLIT_SCHEMA_V1,
        "decision": "GMNW-DATA-SPLIT-A/DEVELOPMENT-AND-QUALIFICATION-FROZEN",
        "development": {
            "simulation_and_task_seeds": list(DEVELOPMENT_SEEDS_V1),
            "permitted_uses": [
                "diagnosis",
                "trace construction",
                "deployment repair",
                "causal-tool development",
                "at most one preregistered live-native witness training campaign",
            ],
            "cue_sequence_sha256": _cue_sequence_digest(seeds=DEVELOPMENT_SEEDS_V1),
        },
        "qualification": {
            "simulation_and_task_seeds": list(QUALIFICATION_SEEDS_V1),
            "opened_only_after_code_witness_and_threshold_freeze": True,
            "task_configurations": [
                {
                    "cue_alphabet": cue_alphabet,
                    "delay_ticks": delay_ticks,
                    "distractors_enabled": distractors_enabled,
                }
                for cue_alphabet, delay_ticks, distractors_enabled in TASK_CONFIGURATIONS_V1
            ],
            "policy_arms": ["persistent_gru", "reset_gru", "matched_mlp"],
            "generation_count": 9,
            "ticks_per_generation": 80,
            "initial_stream_ids": [0, INITIAL_STREAM_COUNT_V1 - 1],
            "completed_episodes_per_initial_stream": QUALIFICATION_EPISODES_PER_STREAM_V1,
            "cue_sequence_sha256": _cue_sequence_digest(seeds=QUALIFICATION_SEEDS_V1),
            "threshold_source": "eiel_gmnw_live_gru_contract_v1.thresholds",
            "causal_intervention_source": "eiel_gmnw_live_gru_contract_v1.causal_interventions",
        },
        "selection_rule": (
            "qualification cases never move into development and no post-qualification "
            "witness selection is permitted"
        ),
    }
    return _bind(payload, "data_split_sha256")


def validate_bound_payload(
    value: Mapping[str, Any], *, schema_version: str, hash_field: str
) -> dict[str, Any]:
    payload = dict(value)
    if payload.get("schema_version") != schema_version:
        raise GmnwLiveGruContractError("schema_version mismatch")
    actual = payload.pop(hash_field, None)
    expected = canonical_sha256(payload)
    if actual != expected:
        raise GmnwLiveGruContractError(f"{hash_field} mismatch")
    payload[hash_field] = expected
    return payload


def write_live_contract_artifacts_v1(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "contract": root / "gmnw_live_contract_v1.json",
        "data_split": root / "gmnw_live_data_split_v1.json",
    }
    values = {"contract": live_contract_v1(), "data_split": data_split_v1()}
    for name, path in outputs.items():
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(values[name], allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
        temporary.replace(path)
    return outputs


__all__ = [
    "DATA_SPLIT_SCHEMA_V1",
    "DEVELOPMENT_SEEDS_V1",
    "GmnwLiveGruContractError",
    "LIVE_CONTRACT_SCHEMA_V1",
    "QUALIFICATION_SEEDS_V1",
    "canonical_sha256",
    "data_split_v1",
    "live_contract_v1",
    "validate_bound_payload",
    "write_live_contract_artifacts_v1",
]
