from __future__ import annotations

"""Deterministic default-off scalar mutation weather helpers.

MUTRAD-1 policy:
- default off
- scalar sigma multiplier only by default
- deterministic from simulation seed, generation, profile, and salt
- no structural weather, reward, survival, lineage, or behavior changes
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MutationWeatherState:
    enabled: bool = False
    profile: str = ""
    active: bool = False
    event_id: str = ""
    generation: int = 0
    severity: str = "off"
    scalar_multiplier: float = 1.0
    count_multiplier: float = 1.0
    structural_enabled: bool = False
    affected_offspring: int = 0

    def as_event_fields(self, *, child_prefix: bool = False) -> dict[str, Any]:
        fields = {
            "mutation_weather_enabled": int(bool(self.enabled)),
            "mutation_weather_profile": str(self.profile),
            "mutation_weather_active": int(bool(self.active)),
            "mutation_weather_event_id": str(self.event_id),
            "mutation_weather_generation": int(self.generation),
            "mutation_weather_severity": str(self.severity),
            "mutation_weather_scalar_multiplier": float(self.scalar_multiplier),
            "mutation_weather_count_multiplier": float(self.count_multiplier),
            "mutation_weather_structural_enabled": int(bool(self.structural_enabled)),
            "mutation_weather_affected_offspring": int(self.affected_offspring),
        }
        if child_prefix:
            fields.update(
                {
                    "child_mutation_weather_active": int(bool(self.active)),
                    "child_mutation_weather_event_id": str(self.event_id),
                    "child_mutation_weather_scalar_multiplier": float(self.scalar_multiplier),
                }
            )
        return fields


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _fixture_generations(raw: Any) -> set[int]:
    if raw in ("", None, False):
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {_safe_int(v, -10**9) for v in raw}
    text = str(raw).strip()
    if not text:
        return set()
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return {_safe_int(v, -10**9) for v in parsed}
    return {_safe_int(part.strip(), -10**9) for part in text.split(",") if part.strip()}


def _bounded_multiplier(sim_cfg: Any, value: float) -> float:
    max_multiplier = _safe_float(getattr(sim_cfg, "mutation_weather_max_scalar_multiplier", 10.0), 10.0)
    max_multiplier = max(1.0, min(10.0, max_multiplier))
    return float(max(1.0, min(max_multiplier, float(value))))


def _unit_interval_digest(*, seed: int, generation: int, profile: str, salt: str) -> float:
    blob = json.dumps(
        {
            "seed": int(seed),
            "generation": int(generation),
            "profile": str(profile),
            "salt": str(salt),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    return int(digest[:16], 16) / float(16**16 - 1)


def mutation_weather_state(sim_cfg: Any, generation: int) -> MutationWeatherState:
    enabled = bool(getattr(sim_cfg, "mutation_weather_enabled", False)) if sim_cfg is not None else False
    profile = str(getattr(sim_cfg, "mutation_weather_profile", "") or "")
    generation = int(generation)
    if not enabled:
        return MutationWeatherState(enabled=False, profile=profile, generation=generation)

    structural_enabled = bool(getattr(sim_cfg, "mutation_weather_structural_enabled", False))
    if structural_enabled:
        # MUTRAD-1 is scalar-only. Fail closed by reporting structural disabled.
        structural_enabled = False

    fixed = _bounded_multiplier(sim_cfg, _safe_float(getattr(sim_cfg, "mutation_weather_fixed_scalar_multiplier", 1.0), 1.0))
    if fixed > 1.0:
        return MutationWeatherState(
            enabled=True,
            profile=profile or "fixed_scalar_control",
            active=True,
            event_id=f"fixed_scalar_control:g{generation}:x{fixed:g}",
            generation=generation,
            severity="fixed_control",
            scalar_multiplier=fixed,
            count_multiplier=1.0,
            structural_enabled=structural_enabled,
            affected_offspring=1,
        )

    max_multiplier = _bounded_multiplier(sim_cfg, _safe_float(getattr(sim_cfg, "mutation_weather_max_scalar_multiplier", 10.0), 10.0))
    fixture = _fixture_generations(getattr(sim_cfg, "mutation_weather_fixture_generations", ""))
    fixture.update(_fixture_generations(getattr(sim_cfg, "_mutation_weather_fixture_generations", "")))
    if generation in fixture:
        return MutationWeatherState(
            enabled=True,
            profile=profile or "safe_pulse_v0",
            active=True,
            event_id=f"fixture:{profile or 'safe_pulse_v0'}:g{generation}:x{max_multiplier:g}",
            generation=generation,
            severity="fixture",
            scalar_multiplier=max_multiplier,
            count_multiplier=1.0,
            structural_enabled=structural_enabled,
            affected_offspring=1,
        )

    if profile != "safe_pulse_v0":
        return MutationWeatherState(enabled=True, profile=profile, generation=generation, structural_enabled=structural_enabled)

    seed = _safe_int(getattr(sim_cfg, "seed", 0), 0)
    salt = str(getattr(sim_cfg, "mutation_weather_seed_salt", "") or "")
    roll = _unit_interval_digest(seed=seed, generation=generation, profile=profile, salt=salt)
    if roll < 0.02:
        severity = "medium"
        multiplier = min(max_multiplier, 10.0)
    elif roll < 0.08:
        severity = "mild"
        multiplier = min(max_multiplier, 3.0)
    else:
        return MutationWeatherState(enabled=True, profile=profile, generation=generation, structural_enabled=structural_enabled)

    return MutationWeatherState(
        enabled=True,
        profile=profile,
        active=True,
        event_id=f"{profile}:g{generation}:{severity}:x{multiplier:g}",
        generation=generation,
        severity=severity,
        scalar_multiplier=float(multiplier),
        count_multiplier=1.0,
        structural_enabled=structural_enabled,
        affected_offspring=1,
    )


def attach_mutation_weather_metadata(genome: Any, state: MutationWeatherState) -> None:
    if genome is None:
        return
    if not bool(state.enabled):
        return
    for key, value in state.as_event_fields(child_prefix=True).items():
        try:
            setattr(genome, key, value)
        except Exception:
            pass
    payload = getattr(genome, "c1_payload", None)
    if isinstance(payload, dict):
        genes = payload.setdefault("genes", {})
        if isinstance(genes, dict):
            for key, value in state.as_event_fields(child_prefix=False).items():
                genes[key] = value
