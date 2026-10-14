"""Device-neutral contract for the GRU memory-necessity world (GMNW-V1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import IntEnum
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Final, Mapping

import numpy as np

from env.gridworld_pkg.constants import (
    ACT_MOVE_E,
    ACT_MOVE_N,
    ACT_MOVE_S,
    ACT_MOVE_W,
)


GMNW_WORLD_NAME: Final[str] = "GRU_MEMORY_NECESSITY_WORLD_V1"
GMNW_WORLD_SHORT_NAME: Final[str] = "GMNW-V1"
GMNW_SCHEMA_VERSION: Final[str] = "eiel_gmnw_private_task_state_v1"
GMNW_INFORMATION_CONTRACT_VERSION: Final[str] = "eiel_gmnw_information_contract_v1"
GMNW_VENV1_PROFILE_CONTRACT_VERSION: Final[str] = (
    "eiel_gmnw_venv1_profile_contract_v1"
)
GMNW_VENV1_PROFILE_CONFIG_VERSION: Final[str] = "eiel_gmnw_venv1_profile_config_v1"
GMNW_VENV1_ACCEPTED_ONE_HIT_LETHAL_DAMAGE: Final[float] = 1.0
GMNW_VENV1_GENERATION_TIERS: Final[tuple[str, ...]] = (
    "grace",
    "severe",
    "lethal",
)
GMNW_VENV1_PROFILE_IDS: Final[tuple[str, ...]] = ("legacy", "P1", "P2", "P3")
GMNW_VENV1_PROFILE_CODES: Final[Mapping[str, int]] = MappingProxyType(
    {"legacy": 0, "P1": 1, "P2": 2, "P3": 3}
)
GMNW_CUE_ACTIONS: Final[tuple[int, ...]] = (
    ACT_MOVE_N,
    ACT_MOVE_E,
    ACT_MOVE_S,
    ACT_MOVE_W,
)
GMNW_TASK_OBSERVATION_COLUMNS: Final[tuple[str, ...]] = (
    "gmnw_cue_0",
    "gmnw_cue_1",
    "gmnw_cue_2",
    "gmnw_cue_3",
    "gmnw_phase_cue",
    "gmnw_phase_delay",
    "gmnw_phase_decision",
    "gmnw_phase_consequence",
    "gmnw_distractor_0",
    "gmnw_distractor_1",
    "gmnw_distractor_2",
    "gmnw_distractor_3",
)
GMNW_PRIVATE_STATE_FIELDS: Final[tuple[str, ...]] = (
    "episode_index",
    "cue_identity",
    "cue_phase",
    "phase_tick",
    "delay_remaining",
    "decision_due",
    "correct_action",
    "last_response",
    "correctness",
    "consequence_status",
    "task_rng_state",
    "task_rng_cursor",
)

_MASK64: Final[int] = (1 << 64) - 1


class GmnwPhase(IntEnum):
    CUE = 0
    DELAY = 1
    DECISION = 2
    CONSEQUENCE = 3


class GmnwConsequenceStatus(IntEnum):
    NONE = 0
    CORRECT = 1
    INCORRECT = 2


@dataclass(frozen=True, slots=True)
class GmnwConfig:
    cue_alphabet: int = 4
    cue_ticks: int = 1
    delay_ticks: int = 4
    consequence_ticks: int = 1
    distractors_enabled: bool = False
    cue_visible_at_decision: bool = False
    task_seed: int = 101
    grace_generation_count: int = 4
    severe_generation_count: int = 4
    grace_damage_fraction: float = 0.15
    severe_damage_fraction: float = 0.45
    lethal_damage_fraction: float = 1.0
    grace_reproduction_scale: float = 0.5

    def __post_init__(self) -> None:
        if type(self.cue_alphabet) is not int or self.cue_alphabet not in {2, 4}:
            raise ValueError("cue_alphabet must be exactly 2 or 4")
        for name in ("cue_ticks", "delay_ticks", "consequence_ticks"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        if type(self.distractors_enabled) is not bool:
            raise ValueError("distractors_enabled must be an exact bool")
        if type(self.cue_visible_at_decision) is not bool:
            raise ValueError("cue_visible_at_decision must be an exact bool")
        if type(self.task_seed) is not int or self.task_seed < 0:
            raise ValueError("task_seed must be a non-negative exact integer")
        if self.task_seed > (1 << 63) - 1:
            raise ValueError("task_seed must fit in a signed 64-bit integer")
        for name in ("grace_generation_count", "severe_generation_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive exact integer")
        for name in (
            "grace_damage_fraction",
            "severe_damage_fraction",
            "lethal_damage_fraction",
            "grace_reproduction_scale",
        ):
            value = getattr(self, name)
            if type(value) is not float or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite exact float")
        if not 0.0 < self.grace_damage_fraction <= self.severe_damage_fraction:
            raise ValueError("grace damage must be positive and not exceed severe damage")
        if not self.severe_damage_fraction < self.lethal_damage_fraction:
            raise ValueError("severe damage must be below lethal damage")
        if self.lethal_damage_fraction < 1.0:
            raise ValueError("lethal damage must cover at least one full health unit")
        if not 0.0 < self.grace_reproduction_scale < 1.0:
            raise ValueError(
                "grace reproduction scale must be strictly between zero and one"
            )

    @property
    def chance_accuracy(self) -> float:
        return 1.0 / float(self.cue_alphabet)


@dataclass(frozen=True, slots=True)
class GmnwPrivateTaskState:
    episode_index: int
    cue_identity: int
    cue_phase: int
    phase_tick: int
    delay_remaining: int
    decision_due: bool
    correct_action: int
    last_response: int
    correctness: int
    consequence_status: int
    task_rng_state: int
    task_rng_cursor: int

    @property
    def phase(self) -> GmnwPhase:
        return GmnwPhase(self.cue_phase)


@dataclass(frozen=True, slots=True)
class GmnwStepResult:
    state: GmnwPrivateTaskState
    action_applied_to_public_world: bool
    decision_recorded: bool
    decision_correct: bool | None


@dataclass(frozen=True, slots=True)
class GmnwPressure:
    tier: str
    damage_fraction: float
    reproduction_scale: float
    lethal: bool


@dataclass(frozen=True, slots=True)
class GmnwVenv1Profile:
    profile_id: str
    profile_code: int
    generation_tier_independent: bool
    incorrect_reproduction_scales: tuple[float, float, float]
    incorrect_lethal_damage_multipliers: tuple[float, float, float]

    @property
    def incorrect_reproduction_scale(self) -> float | None:
        if not self.generation_tier_independent:
            return None
        return self.incorrect_reproduction_scales[0]

    @property
    def incorrect_lethal_damage_multiplier(self) -> float | None:
        if not self.generation_tier_independent:
            return None
        return self.incorrect_lethal_damage_multipliers[0]


GMNW_VENV1_PROFILES: Final[Mapping[str, GmnwVenv1Profile]] = MappingProxyType(
    {
        "legacy": GmnwVenv1Profile(
            profile_id="legacy",
            profile_code=0,
            generation_tier_independent=False,
            incorrect_reproduction_scales=(0.5, 0.0, 0.0),
            incorrect_lethal_damage_multipliers=(0.15, 0.45, 1.0),
        ),
        "P1": GmnwVenv1Profile(
            profile_id="P1",
            profile_code=1,
            generation_tier_independent=True,
            incorrect_reproduction_scales=(0.0, 0.0, 0.0),
            incorrect_lethal_damage_multipliers=(0.0, 0.0, 0.0),
        ),
        "P2": GmnwVenv1Profile(
            profile_id="P2",
            profile_code=2,
            generation_tier_independent=True,
            incorrect_reproduction_scales=(0.0, 0.0, 0.0),
            incorrect_lethal_damage_multipliers=(0.25, 0.25, 0.25),
        ),
        "P3": GmnwVenv1Profile(
            profile_id="P3",
            profile_code=3,
            generation_tier_independent=True,
            incorrect_reproduction_scales=(0.0, 0.0, 0.0),
            incorrect_lethal_damage_multipliers=(0.5, 0.5, 0.5),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class GmnwOutcomeState:
    health: float = 1.0
    reproduction_scale: float = 1.0
    alive: bool = True
    decision_count: int = 0
    correct_count: int = 0
    incorrect_count: int = 0
    damage_total: float = 0.0


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _episode_rng_state(
    config: GmnwConfig, *, stream_id: int, episode_index: int
) -> int:
    mixed = (
        int(config.task_seed)
        ^ ((int(stream_id) + 1) * 0xD6E8FEB86659FD93)
        ^ ((int(episode_index) + 1) * 0xA5A3564E27F8862F)
    ) & _MASK64
    return _splitmix64(mixed)


def pressure_for_generation(config: GmnwConfig, generation_index: int) -> GmnwPressure:
    if type(generation_index) is not int or generation_index < 0:
        raise ValueError("generation_index must be a non-negative exact integer")
    if generation_index < config.grace_generation_count:
        return GmnwPressure(
            tier="grace",
            damage_fraction=config.grace_damage_fraction,
            reproduction_scale=config.grace_reproduction_scale,
            lethal=False,
        )
    if (
        generation_index
        < config.grace_generation_count + config.severe_generation_count
    ):
        return GmnwPressure(
            tier="severe",
            damage_fraction=config.severe_damage_fraction,
            reproduction_scale=0.0,
            lethal=False,
        )
    return GmnwPressure(
        tier="lethal",
        damage_fraction=config.lethal_damage_fraction,
        reproduction_scale=0.0,
        lethal=True,
    )


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_gmnw_venv1_profile(profile: str | int) -> GmnwVenv1Profile:
    """Resolve one exact VENV-1 ID or code without aliases or fallback."""

    if type(profile) is str:
        resolved = GMNW_VENV1_PROFILES.get(profile)
    elif type(profile) is int:
        resolved = next(
            (
                candidate
                for candidate in GMNW_VENV1_PROFILES.values()
                if candidate.profile_code == profile
            ),
            None,
        )
    else:
        resolved = None
    if resolved is None:
        raise ValueError(
            "VENV-1 profile must be an exact canonical ID "
            "('legacy', 'P1', 'P2', 'P3') or exact code (0, 1, 2, 3)"
        )
    return resolved


def _gmnw_venv1_profile_payload(profile: GmnwVenv1Profile) -> dict[str, object]:
    resolved_by_tier = {
        tier: {
            "incorrect_reproduction_scale": profile.incorrect_reproduction_scales[
                index
            ],
            "incorrect_lethal_damage_multiplier": (
                profile.incorrect_lethal_damage_multipliers[index]
            ),
            "incorrect_damage_fraction": (
                GMNW_VENV1_ACCEPTED_ONE_HIT_LETHAL_DAMAGE
                * profile.incorrect_lethal_damage_multipliers[index]
            ),
        }
        for index, tier in enumerate(GMNW_VENV1_GENERATION_TIERS)
    }
    return {
        "profile_id": profile.profile_id,
        "profile_code": profile.profile_code,
        "generation_tier_independent": profile.generation_tier_independent,
        "resolved_by_generation_tier": resolved_by_tier,
    }


def gmnw_venv1_profile_contract() -> dict[str, object]:
    material: dict[str, object] = {
        "schema_version": GMNW_VENV1_PROFILE_CONTRACT_VERSION,
        "profile_ids": list(GMNW_VENV1_PROFILE_IDS),
        "profile_codes": dict(GMNW_VENV1_PROFILE_CODES),
        "accepted_one_hit_lethal_damage": (
            GMNW_VENV1_ACCEPTED_ONE_HIT_LETHAL_DAMAGE
        ),
        "incorrect_decision_semantics": {
            "P1_P2_P3_reproduction_opportunity_scale": 0.0,
            "damage_basis": "accepted_one_hit_lethal_damage",
            "generation_tier_independent_profiles": ["P1", "P2", "P3"],
            "legacy_behavior": "unchanged_generation_tier_schedule",
        },
        "profiles": [
            _gmnw_venv1_profile_payload(GMNW_VENV1_PROFILES[profile_id])
            for profile_id in GMNW_VENV1_PROFILE_IDS
        ],
        "resolution": {
            "accepted_selectors": "exact_canonical_profile_id_or_exact_integer_code",
            "aliases": False,
            "case_folding": False,
            "fallback": False,
        },
        "hash": {
            "algorithm": "sha256",
            "canonical_json": "utf8_sorted_keys_compact_ascii_finite",
        },
    }
    return {**material, "contract_hash": _canonical_sha256(material)}


def gmnw_venv1_profile_contract_hash() -> str:
    return str(gmnw_venv1_profile_contract()["contract_hash"])


def _gmnw_venv1_effective_config_payload(
    profile: GmnwVenv1Profile,
    config: GmnwConfig,
) -> dict[str, object]:
    if profile.profile_id == "legacy":
        reproduction_scales = (
            config.grace_reproduction_scale,
            0.0,
            0.0,
        )
        damage_fractions = (
            config.grace_damage_fraction,
            config.severe_damage_fraction,
            config.lethal_damage_fraction,
        )
        tier_boundaries: dict[str, int] | None = {
            "grace_generation_count": config.grace_generation_count,
            "severe_generation_count": config.severe_generation_count,
        }
    else:
        reproduction_scales = profile.incorrect_reproduction_scales
        damage_fractions = tuple(
            float(config.lethal_damage_fraction * multiplier)
            for multiplier in profile.incorrect_lethal_damage_multipliers
        )
        tier_boundaries = None
    return {
        "schema_version": GMNW_VENV1_PROFILE_CONFIG_VERSION,
        "contract_hash": gmnw_venv1_profile_contract_hash(),
        "profile_id": profile.profile_id,
        "profile_code": profile.profile_code,
        "generation_tier_independent": profile.generation_tier_independent,
        "generation_tier_boundaries": tier_boundaries,
        "accepted_one_hit_lethal_damage": config.lethal_damage_fraction,
        "resolved_by_generation_tier": {
            tier: {
                "incorrect_reproduction_scale": reproduction_scales[index],
                "incorrect_damage_fraction": damage_fractions[index],
            }
            for index, tier in enumerate(GMNW_VENV1_GENERATION_TIERS)
        },
    }


def gmnw_venv1_profile_config(
    profile: str | int,
    *,
    config: GmnwConfig | None = None,
) -> dict[str, object]:
    """Return the canonical effective configuration for one VENV-1 profile."""

    resolved_profile = resolve_gmnw_venv1_profile(profile)
    resolved_config = GmnwConfig() if config is None else config
    if type(resolved_config) is not GmnwConfig:
        raise ValueError("config must be an exact GmnwConfig")
    material = _gmnw_venv1_effective_config_payload(
        resolved_profile,
        resolved_config,
    )
    return {**material, "config_hash": _canonical_sha256(material)}


def resolve_gmnw_venv1_profile_config(
    profile: str | int,
    *,
    generation_index: int,
    config: GmnwConfig | None = None,
) -> dict[str, object]:
    """Resolve exact effective VENV-1 consequence values for one generation."""

    resolved_profile = resolve_gmnw_venv1_profile(profile)
    resolved_config = GmnwConfig() if config is None else config
    if type(resolved_config) is not GmnwConfig:
        raise ValueError("config must be an exact GmnwConfig")
    legacy_pressure = pressure_for_generation(resolved_config, generation_index)
    tier_index = GMNW_VENV1_GENERATION_TIERS.index(legacy_pressure.tier)
    if resolved_profile.profile_id == "legacy":
        reproduction_scale = legacy_pressure.reproduction_scale
        damage_fraction = legacy_pressure.damage_fraction
        lethal_damage_multiplier = float(
            damage_fraction / resolved_config.lethal_damage_fraction
        )
        lethal = legacy_pressure.lethal
    else:
        reproduction_scale = resolved_profile.incorrect_reproduction_scales[tier_index]
        lethal_damage_multiplier = (
            resolved_profile.incorrect_lethal_damage_multipliers[tier_index]
        )
        damage_fraction = float(
            resolved_config.lethal_damage_fraction * lethal_damage_multiplier
        )
        lethal = damage_fraction >= resolved_config.lethal_damage_fraction
    material: dict[str, object] = {
        "schema_version": GMNW_VENV1_PROFILE_CONFIG_VERSION,
        "contract_hash": gmnw_venv1_profile_contract_hash(),
        "profile_id": resolved_profile.profile_id,
        "profile_code": resolved_profile.profile_code,
        "generation_index": generation_index,
        "generation_tier": legacy_pressure.tier,
        "generation_tier_independent": (
            resolved_profile.generation_tier_independent
        ),
        "accepted_one_hit_lethal_damage": resolved_config.lethal_damage_fraction,
        "incorrect_reproduction_scale": reproduction_scale,
        "incorrect_lethal_damage_multiplier": lethal_damage_multiplier,
        "incorrect_damage_fraction": damage_fraction,
        "incorrect_decision_lethal": lethal,
    }
    config_hash = gmnw_venv1_profile_config(
        resolved_profile.profile_id,
        config=resolved_config,
    )["config_hash"]
    return {**material, "config_hash": config_hash}


def gmnw_venv1_profile_config_hash(
    profile: str | int,
    *,
    config: GmnwConfig | None = None,
) -> str:
    return str(gmnw_venv1_profile_config(profile, config=config)["config_hash"])


def apply_decision_consequence(
    outcome: GmnwOutcomeState,
    *,
    decision_correct: bool,
    pressure: GmnwPressure,
    start_health: float = 1.0,
) -> GmnwOutcomeState:
    if type(decision_correct) is not bool:
        raise ValueError("decision_correct must be an exact bool")
    if (
        type(start_health) is not float
        or not np.isfinite(start_health)
        or start_health <= 0.0
    ):
        raise ValueError("start_health must be a positive finite exact float")
    if not outcome.alive:
        return outcome
    if decision_correct:
        return replace(
            outcome,
            decision_count=outcome.decision_count + 1,
            correct_count=outcome.correct_count + 1,
        )
    damage = float(start_health * pressure.damage_fraction)
    health = float(outcome.health - damage)
    return replace(
        outcome,
        health=health,
        reproduction_scale=min(
            outcome.reproduction_scale,
            pressure.reproduction_scale,
        ),
        alive=health > 0.0,
        decision_count=outcome.decision_count + 1,
        incorrect_count=outcome.incorrect_count + 1,
        damage_total=outcome.damage_total + damage,
    )


def reset_episode_reproduction_opportunity(
    outcome: GmnwOutcomeState,
) -> GmnwOutcomeState:
    if not outcome.alive:
        return outcome
    return replace(outcome, reproduction_scale=1.0)


def balanced_cue_identity(
    config: GmnwConfig,
    *,
    stream_id: int,
    episode_index: int,
) -> int:
    """Return a shuffled cue while keeping every alphabet-sized block balanced."""

    if type(stream_id) is not int or stream_id < 0:
        raise ValueError("stream_id must be a non-negative exact integer")
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("episode_index must be a non-negative exact integer")
    alphabet = config.cue_alphabet
    block_index, ordinal = divmod(episode_index, alphabet)
    values = list(range(alphabet))
    state = _episode_rng_state(config, stream_id=stream_id, episode_index=block_index)
    for upper in range(alphabet - 1, 0, -1):
        state = _splitmix64(state)
        swap = int(state % (upper + 1))
        values[upper], values[swap] = values[swap], values[upper]
    return int(values[ordinal])


def initial_private_task_state(
    config: GmnwConfig,
    *,
    stream_id: int,
    episode_index: int = 0,
) -> GmnwPrivateTaskState:
    cue = balanced_cue_identity(
        config,
        stream_id=stream_id,
        episode_index=episode_index,
    )
    return GmnwPrivateTaskState(
        episode_index=episode_index,
        cue_identity=cue,
        cue_phase=int(GmnwPhase.CUE),
        phase_tick=0,
        delay_remaining=0,
        decision_due=False,
        correct_action=int(GMNW_CUE_ACTIONS[cue]),
        last_response=-1,
        correctness=-1,
        consequence_status=int(GmnwConsequenceStatus.NONE),
        task_rng_state=_episode_rng_state(
            config,
            stream_id=stream_id,
            episode_index=episode_index,
        ),
        task_rng_cursor=episode_index + 1,
    )


def task_observation(
    state: GmnwPrivateTaskState,
    config: GmnwConfig,
) -> np.ndarray:
    """Return the fixed-width public task overlay for one policy observation."""

    observation = np.zeros((len(GMNW_TASK_OBSERVATION_COLUMNS),), dtype=np.float32)
    phase = state.phase
    if phase is GmnwPhase.CUE or (
        phase is GmnwPhase.DECISION and config.cue_visible_at_decision
    ):
        observation[state.cue_identity] = np.float32(1.0)
    observation[4 + int(phase)] = np.float32(1.0)
    if phase is GmnwPhase.DELAY and config.distractors_enabled:
        elapsed = config.delay_ticks - state.delay_remaining
        distractor_state = _splitmix64(
            state.task_rng_state
            ^ ((elapsed + 1) * 0x8CB92BA72F3D8DD7)
            ^ 0x47524D4E575F4453
        )
        observation[8 + int(distractor_state % 4)] = np.float32(1.0)
    return observation


def append_task_observation(
    base_observation: np.ndarray,
    state: GmnwPrivateTaskState,
    config: GmnwConfig,
) -> np.ndarray:
    base = np.asarray(base_observation, dtype=np.float32)
    if base.ndim != 1 or not np.isfinite(base).all():
        raise ValueError("base_observation must be one finite float32 row")
    return np.concatenate((base, task_observation(state, config))).astype(
        np.float32,
        copy=False,
    )


def advance_private_task_state(
    state: GmnwPrivateTaskState,
    config: GmnwConfig,
    *,
    stream_id: int,
    action: int,
) -> GmnwStepResult:
    """Advance one tick; cue and delay actions never alter the public world."""

    phase = state.phase
    if phase is GmnwPhase.CUE:
        next_tick = state.phase_tick + 1
        if next_tick < config.cue_ticks:
            next_state = replace(state, phase_tick=next_tick)
        else:
            next_state = replace(
                state,
                cue_phase=int(GmnwPhase.DELAY),
                phase_tick=0,
                delay_remaining=config.delay_ticks,
            )
        return GmnwStepResult(next_state, False, False, None)

    if phase is GmnwPhase.DELAY:
        remaining = state.delay_remaining - 1
        if remaining > 0:
            next_state = replace(
                state,
                phase_tick=state.phase_tick + 1,
                delay_remaining=remaining,
            )
        else:
            next_state = replace(
                state,
                cue_phase=int(GmnwPhase.DECISION),
                phase_tick=0,
                delay_remaining=0,
                decision_due=True,
            )
        return GmnwStepResult(next_state, False, False, None)

    if phase is GmnwPhase.DECISION:
        correct = int(action) == state.correct_action
        next_state = replace(
            state,
            cue_phase=int(GmnwPhase.CONSEQUENCE),
            phase_tick=0,
            decision_due=False,
            last_response=int(action),
            correctness=1 if correct else 0,
            consequence_status=int(
                GmnwConsequenceStatus.CORRECT
                if correct
                else GmnwConsequenceStatus.INCORRECT
            ),
        )
        return GmnwStepResult(next_state, False, True, correct)

    next_tick = state.phase_tick + 1
    if next_tick < config.consequence_ticks:
        next_state = replace(state, phase_tick=next_tick)
    else:
        next_state = initial_private_task_state(
            config,
            stream_id=stream_id,
            episode_index=state.episode_index + 1,
        )
    return GmnwStepResult(next_state, False, False, None)


def decision_observation_equivalence(config: GmnwConfig) -> dict[str, object]:
    rows: list[bytes] = []
    digests: list[str] = []
    base = np.zeros((17,), dtype=np.float32)
    for cue in range(config.cue_alphabet):
        state = replace(
            initial_private_task_state(config, stream_id=0),
            cue_identity=cue,
            cue_phase=int(GmnwPhase.DECISION),
            decision_due=True,
            correct_action=int(GMNW_CUE_ACTIONS[cue]),
        )
        payload = append_task_observation(base, state, config).tobytes(order="C")
        rows.append(payload)
        digests.append(hashlib.sha256(payload).hexdigest())
    return {
        "schema_version": GMNW_INFORMATION_CONTRACT_VERSION,
        "cue_alphabet": config.cue_alphabet,
        "decision_observation_bytes_identical": len(set(rows)) == 1,
        "decision_observation_sha256_by_cue": digests,
        "chance_accuracy": config.chance_accuracy,
        "majority_observation_only_accuracy": config.chance_accuracy,
        "leakage_threshold": config.chance_accuracy + 0.05,
    }


def information_contract() -> dict[str, object]:
    return {
        "schema_version": GMNW_INFORMATION_CONTRACT_VERSION,
        "world": GMNW_WORLD_NAME,
        "short_name": GMNW_WORLD_SHORT_NAME,
        "private_state_schema_version": GMNW_SCHEMA_VERSION,
        "private_state_fields": list(GMNW_PRIVATE_STATE_FIELDS),
        "task_observation_columns": list(GMNW_TASK_OBSERVATION_COLUMNS),
        "cue_actions": list(GMNW_CUE_ACTIONS),
        "public_action_application": {
            "cue": False,
            "delay": False,
            "decision": False,
            "consequence": False,
        },
        "decision_rule": "correct action depends only on the earlier private cue",
        "decision_observation_rule": "cue and distractor channels are zero; all public bytes are cue-invariant",
        "excluded_external_memory": [
            "position",
            "terrain",
            "food",
            "scent",
            "waste",
            "density",
            "previous_action",
            "other_agents",
            "agent_identity",
        ],
        "pressure_schedule": consequence_schedule_contract(),
    }


def consequence_schedule_contract() -> dict[str, object]:
    config = GmnwConfig()
    return {
        "schema_version": "eiel_gmnw_consequence_schedule_v1",
        "health_unit": 1.0,
        "correct_decision": {
            "damage": 0.0,
            "direct_reward": 0.0,
            "bonus_energy": 0.0,
            "bonus_fitness": 0.0,
            "reproduction_scale": 1.0,
        },
        "tiers": [
            {
                "name": "grace",
                "generation_start": 0,
                "generation_end_inclusive": config.grace_generation_count - 1,
                "incorrect_damage_fraction": config.grace_damage_fraction,
                "reproduction_scale": config.grace_reproduction_scale,
                "lethal": False,
            },
            {
                "name": "severe",
                "generation_start": config.grace_generation_count,
                "generation_end_inclusive": (
                    config.grace_generation_count + config.severe_generation_count - 1
                ),
                "incorrect_damage_fraction": config.severe_damage_fraction,
                "reproduction_scale": 0.0,
                "lethal": False,
            },
            {
                "name": "lethal",
                "generation_start": (
                    config.grace_generation_count + config.severe_generation_count
                ),
                "generation_end_inclusive": None,
                "incorrect_damage_fraction": config.lethal_damage_fraction,
                "reproduction_scale": 0.0,
                "lethal": True,
            },
        ],
        "tick_order": [
            "observation",
            "policy_action",
            "task_decision",
            "task_consequence",
            "task_health_and_reproduction_update",
            "task_death_settlement",
            "ordinary_survivor_upkeep",
            "residual_upkeep_death_settlement",
            "task_and_lifecycle_event_publication",
            "birth_candidate_scan",
            "successful_birth_commit",
            "generation_turnover_if_boundary",
        ],
        "atomicity": {
            "health_updates_per_decision": 1,
            "reproduction_updates_per_decision": 1,
            "death_settlements_per_agent": 1,
            "birth_scan_after_consequence_and_death": True,
            "checkpoint_after_settlement": True,
        },
    }


def reproduction_opportunity_contract() -> dict[str, object]:
    return {
        "schema_version": "eiel_gmnw_reproduction_opportunity_contract_v1",
        "scale_semantics": "absolute_multiplier_of_the_ordinary_attempt_probability",
        "multiplicative_compounding_within_episode": False,
        "one_decision_per_episode": True,
        "incorrect_decision_effect": "replace_current_episode_scale_with_tier_scale",
        "correct_decision_effect": "preserve_ordinary_scale_without_bonus",
        "reset": "only_when_the_next_private_episode_is_initialized",
        "generation_boundary_reset": False,
        "selected_parent_requirements": [
            "alive",
            "ordinary_parent_eligibility",
            "current_episode_reproduction_scale_greater_than_zero",
        ],
        "ordinary_birth_limits_unchanged": True,
        "required_compact_counters": [
            "decision_count",
            "correct_count",
            "incorrect_count",
            "reproduction_scale_half_count",
            "reproduction_scale_zero_count",
            "task_damage_total",
            "task_death_count",
        ],
    }


def turnover_extinction_contract() -> dict[str, object]:
    return {
        "schema_version": "eiel_gmnw_turnover_extinction_contract_v1",
        "refill_parent_source": "surviving_selected_eligible_parents_only",
        "generic_founder_template_reuse": False,
        "random_replacement": False,
        "extinct_lineage_reintroduction": False,
        "unearned_policy_family_creation": False,
        "target_population_restoration_guaranteed": False,
        "zero_eligible_parent_result": "extinct_no_eligible_parents",
        "zero_parent_world_terminal": True,
        "one_parent_refill": "exact_asexual_inheritance_subject_to_ordinary_limits",
        "mixed_parent_refill": "only_from_the_frozen_selected_surviving_parent_set",
        "lineage_invariant": "every_refill_child_has_an_alive_eligible_parent_in_the_same_world",
    }


def episode_balance_contract() -> dict[str, object]:
    return {
        "schema_version": "eiel_gmnw_episode_balance_contract_v1",
        "balance_unit": "private_per_agent_alphabet_sized_shuffled_block",
        "complete_block": "contains_each_cue_exactly_once",
        "new_agent": "starts_a_fresh_private_block",
        "checkpoint": "preserves_episode_index_block_position_and_task_rng_state",
        "death": "discards_the_unfinished_private_block",
        "slot_reuse": "new_stream_identity_never_inherits_the_previous_agents_sequence",
        "analysis_weighting": "completed_decisions_only",
        "partial_block_reporting": True,
        "cue_independent_of": [
            "policy_family",
            "lineage",
            "health",
            "witness_fraction",
            "public_world_state",
        ],
    }


def witness_inheritance_contract() -> dict[str, object]:
    return {
        "schema_version": "eiel_gmnw_witness_inheritance_contract_v1",
        "campaign_claim": "selection_of_an_available_solution",
        "witness_policy_family": "stable_gru",
        "witness_hidden_width": 64,
        "witness_effective_memory_width": 64,
        "witness_weights": "inherited_exactly_without_mutation",
        "policy_weight_mutation_enabled": False,
        "memory_width_mutation_enabled": False,
        "mlp_to_gru_switching_enabled": False,
        "gru_to_mlp_switching_enabled": False,
        "witness_lineage_protection": False,
        "ordinary_death_and_parent_selection_apply": True,
        "unsupported_cross_family_construction": "fail_closed",
    }


def failure_isolation_contract() -> dict[str, object]:
    return {
        "schema_version": "eiel_gmnw_failure_isolation_contract_v1",
        "default_failure_scope": "cell_local_terminal_failure",
        "sibling_world_state_mutation": False,
        "matrix_acceptance_when_any_cell_fails": False,
        "completed_cell_artifact_retention": True,
        "dispatch_global_failure_only_for": [
            "cuda_memory_corruption_risk",
            "shared_allocator_invariant_failure",
            "shared_kernel_contract_failure",
        ],
        "chunking_changes_scientific_configuration": False,
        "mutable_genome_aliasing": False,
        "mutable_policy_parameter_aliasing": False,
        "hidden_state_aliasing": False,
        "task_rng_aliasing": False,
        "lifecycle_rng_aliasing": False,
        "event_identity_or_lineage_aliasing": False,
    }


def write_presimulation_contract_artifacts(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, object]] = {
        "gmnw_consequence_schedule.json": consequence_schedule_contract(),
        "gmnw_reproduction_opportunity_contract.json": (
            reproduction_opportunity_contract()
        ),
        "gmnw_turnover_extinction_contract.json": turnover_extinction_contract(),
        "gmnw_episode_balance_contract.json": episode_balance_contract(),
        "gmnw_witness_inheritance_contract.json": witness_inheritance_contract(),
        "gmnw_failure_isolation_contract.json": failure_isolation_contract(),
    }
    paths: dict[str, Path] = {}
    for filename, payload in payloads.items():
        path = root / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths[filename] = path
    markdown = root / "gmnw_consequence_schedule.md"
    markdown.write_text(
        "# GMNW-V1 Consequence Schedule\n\n"
        "Generations 0-3 apply 15% health damage and reproduction scale 0.5. "
        "Generations 4-7 apply 45% damage and scale 0. Generations 8 onward "
        "apply one full health unit and scale 0. Correct decisions receive no "
        "direct reward, energy, fitness, or reproduction bonus. Consequences "
        "and deaths settle before the birth scan.\n",
        encoding="utf-8",
    )
    paths[markdown.name] = markdown
    return paths


def write_information_contract_artifacts(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    contract = information_contract()
    equivalence = {
        str(alphabet): decision_observation_equivalence(
            GmnwConfig(cue_alphabet=alphabet)
        )
        for alphabet in (2, 4)
    }
    private_state = initial_private_task_state(GmnwConfig(), stream_id=0)
    payloads: dict[str, object] = {
        "gmnw_information_contract.json": contract,
        "gmnw_observation_equivalence.json": equivalence,
        "gmnw_private_state_contract.json": {
            "schema_version": GMNW_SCHEMA_VERSION,
            "fields": list(GMNW_PRIVATE_STATE_FIELDS),
            "example": asdict(private_state),
        },
        "gmnw_chance_baseline.json": {
            "schema_version": "eiel_gmnw_chance_baseline_v1",
            "two_way": 0.5,
            "four_way": 0.25,
            "leakage_ceiling_points": 0.05,
        },
    }
    paths: dict[str, Path] = {}
    for filename, payload in payloads.items():
        path = root / filename
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths[filename] = path
    markdown = root / "gmnw_information_contract.md"
    markdown.write_text(
        "# GMNW-V1 Information Contract\n\n"
        "At the delayed decision tick, every external observation byte is "
        "identical across cue identities. Cue and delay actions are ignored, "
        "so an agent cannot store the cue in position or another public surface.\n\n"
        "Chance accuracy is 50% for two cues and 25% for four cues. The "
        "observation-only leakage ceiling is chance plus five percentage points.\n",
        encoding="utf-8",
    )
    paths[markdown.name] = markdown
    return paths
