"""Read-only scientific sampling projections for the reference tick loop.

TN-2 records observations only after a tick has committed.  The sampler never
calls a simulation RNG, never alters mechanics, and is not checkpoint authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


TICK_NATIVE_SCIENCE_SAMPLE_SCHEMA_VERSION = "eiel-tick-native-science-sample-v1"


class TickNativeSamplingError(RuntimeError):
    """A tick-native scientific sampling request is invalid or unsupported."""


def science_sampling_interval_ticks(sim_cfg: object) -> int | None:
    """Return the configured interval, using ``None``/0 as explicit OFF."""

    value = getattr(sim_cfg, "science_sample_interval_ticks", None)
    if value is None or value == 0:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TickNativeSamplingError(
            "science_sample_interval_ticks must be OFF (None/0) or a positive integer"
        )
    return value


def science_sampling_enabled(sim_cfg: object) -> bool:
    return science_sampling_interval_ticks(sim_cfg) is not None


def _array_digest(value: object) -> bytes:
    array = np.ascontiguousarray(np.asarray(value))
    header = f"{array.dtype.str}:{array.shape}".encode("ascii")
    return header + b"\0" + array.tobytes(order="C")


def _projection_state_digest(sim: object) -> str:
    """Hash a stable read-only projection without invoking simulation methods.

    This digest is diagnostic evidence only; it neither replaces terminal
    parity checks nor participates in simulation or checkpoint identity.
    """

    digest = hashlib.sha256()
    env = getattr(sim, "env", None)
    for name in (
        "grid",
        "agent_grid",
        "agent_alive",
        "agent_pos_x",
        "agent_pos_y",
        "agent_energy",
        "agent_injury",
    ):
        value = getattr(env, name, None)
        if value is None:
            continue
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(_array_digest(value))
        digest.update(b"\0")
    agents = getattr(getattr(sim, "pop", None), "agents", {})
    if isinstance(agents, dict):
        for agent_id, agent in sorted(agents.items(), key=lambda item: int(item[0])):
            row = {
                "agent_id": int(agent_id),
                "alive": bool(getattr(agent, "alive", False)),
                "energy": float(getattr(agent, "energy", 0.0)),
                "health": float(getattr(agent, "health", 0.0)),
                "injury": float(getattr(agent, "injury", 0.0)),
            }
            digest.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def maybe_capture_tick_native_science_sample(
    sim: object,
    *,
    generation: int,
    tick: int,
) -> dict[str, Any] | None:
    """Write one projection at a committed tick schedule boundary, if due."""

    sim_cfg = getattr(sim, "sim_cfg", None)
    interval = science_sampling_interval_ticks(sim_cfg)
    if interval is None:
        return None
    if bool(getattr(sim, "_tick_native_execution_route", False)):
        total_ticks = getattr(sim_cfg, "total_ticks", None)
        if isinstance(total_ticks, bool) or not isinstance(total_ticks, int) or total_ticks <= 0:
            raise TickNativeSamplingError("tick-native sampling requires positive total_ticks")
        absolute_tick = int(tick)
        next_tick = absolute_tick + 1
        if next_tick % interval:
            return None
        run_dir = getattr(sim, "run_dir", "")
        if not isinstance(run_dir, str) or not run_dir:
            raise TickNativeSamplingError("sampling requires a concrete run_dir")
        population = getattr(getattr(sim, "pop", None), "alive_agents", None)
        alive = list(population()) if callable(population) else []
        record: dict[str, Any] = {
            "schema_version": TICK_NATIVE_SCIENCE_SAMPLE_SCHEMA_VERSION,
            "absolute_tick": absolute_tick,
            "next_tick": next_tick,
            "total_ticks": total_ticks,
            "legacy_generation": None,
            "legacy_tick": None,
            "population_alive": len(alive),
            "projection_state_sha256": _projection_state_digest(sim),
        }
        output = Path(run_dir) / "metrics" / "tick_native_science_samples.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")
        return record

    steps_per_generation = getattr(sim_cfg, "steps_per_generation", None)
    generations = getattr(sim_cfg, "generations", None)
    if (
        isinstance(steps_per_generation, bool)
        or not isinstance(steps_per_generation, int)
        or steps_per_generation <= 0
        or isinstance(generations, bool)
        or not isinstance(generations, int)
        or generations <= 0
    ):
        raise TickNativeSamplingError("sampling requires positive generation and tick horizons")
    absolute_tick = int(generation) * int(steps_per_generation) + int(tick)
    next_tick = absolute_tick + 1
    if next_tick % interval:
        return None
    run_dir = getattr(sim, "run_dir", "")
    if not isinstance(run_dir, str) or not run_dir:
        raise TickNativeSamplingError("sampling requires a concrete run_dir")
    population = getattr(getattr(sim, "pop", None), "alive_agents", None)
    alive = list(population()) if callable(population) else []
    record: dict[str, Any] = {
        "schema_version": TICK_NATIVE_SCIENCE_SAMPLE_SCHEMA_VERSION,
        "absolute_tick": absolute_tick,
        "next_tick": next_tick,
        "total_ticks": int(generations) * int(steps_per_generation),
        "legacy_generation": int(generation),
        "legacy_tick": int(tick),
        "population_alive": len(alive),
        "projection_state_sha256": _projection_state_digest(sim),
    }
    output = Path(run_dir) / "metrics" / "tick_native_science_samples.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
        handle.write("\n")
    return record


__all__ = [
    "TICK_NATIVE_SCIENCE_SAMPLE_SCHEMA_VERSION",
    "TickNativeSamplingError",
    "maybe_capture_tick_native_science_sample",
    "science_sampling_enabled",
    "science_sampling_interval_ticks",
]
