"""Explicit reference CPU route for tick-native execution.

This module deliberately does not alter ``SimulationRunner.run``.  Existing
generation-qualified runs continue through that method.  The new route invokes
one initialized reference tick span and never enters generation-final turnover.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TICK_NATIVE_EXECUTION_ROUTE_V1 = "tick_native_execution_route_v1"
LEGACY_GENERATION_ROUTE_V1 = "legacy_generation_route_v1"


class TickNativeRouteError(RuntimeError):
    """The explicit tick-native route was requested outside its current scope."""


def maybe_capture_tick_native_report_v1(sim: Any, *, next_tick: int) -> None:
    """Emit an observational report only at a requested committed tick."""

    if not bool(getattr(sim, "_tick_native_execution_route", False)):
        return
    cadence = int(getattr(getattr(sim, "sim_cfg", None), "report_every_ticks", 0) or 0)
    if cadence <= 0 or next_tick % cadence:
        return
    output = Path(str(getattr(sim, "run_dir", ""))) / "reports" / "tick_native_reports.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"next_tick": int(next_tick)}, sort_keys=True) + "\n")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TickNativeRouteError(f"{name} must be a positive integer")
    return value


def _checkpoint_sidecar_path(bundle_dir: Path) -> Path:
    return bundle_dir / "tick_native_route.json"


def _route_schedule(runner: Any, *, total_ticks: int) -> dict[str, int | None]:
    """Return the durable, non-biological scheduling identity for this route."""

    cfg = runner.sim_cfg
    def optional_cadence(name: str) -> int | None:
        value = getattr(cfg, name, 0)
        if value is None or value == 0:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TickNativeRouteError(f"{name} must be 0 or a positive integer")
        return int(value)

    partition = getattr(cfg, "runtime_partition_ticks", None)
    if partition is None:
        partition = total_ticks
    if isinstance(partition, bool) or not isinstance(partition, int) or partition <= 0:
        raise TickNativeRouteError("runtime_partition_ticks must be a positive integer")
    return {
        "runtime_partition_ticks": int(partition),
        "science_sample_interval_ticks": optional_cadence("science_sample_interval_ticks"),
        "report_every_ticks": optional_cadence("report_every_ticks"),
        "checkpoint_every_ticks": optional_cadence("checkpoint_every_ticks"),
    }


def _write_tick_native_observability_contract(
    runner: Any,
    *,
    total_ticks: int,
    schedule: dict[str, int | None],
) -> None:
    """Describe tick segments for operational consumers without redefining biology.

    Live Simulations can consume committed partition cursors directly.  Canopy
    playback is stricter: it must advance only from an exported, checkpoint-
    backed projection, never from an in-flight heartbeat or dispatch segment.
    The current Canopy relay has no tick-bundle reader, so this route records
    that absence instead of advertising a false live playback link.
    """

    payload = {
        "schema_version": "eiel_tick_native_observability_contract_v1",
        "route_identity": TICK_NATIVE_EXECUTION_ROUTE_V1,
        "total_ticks": int(total_ticks),
        "live_simulations": {
            "cursor": "next_tick",
            "range_semantics": "committed half-open [0,next_tick)",
            "update_boundaries": "runtime partition ends and any checkpoint split",
            "runtime_partition_ticks": schedule["runtime_partition_ticks"],
            "telemetry_precision": "committed_partition_cursor",
        },
        "canopy": {
            "relay_compatible": False,
            "playback_boundary": "checkpoint-backed exported projection only",
            "checkpoint_every_ticks": schedule["checkpoint_every_ticks"],
            "reason": (
                "the current Canopy relay accepts generation-named projection "
                "bundles and must not infer playback state from tick telemetry"
            ),
        },
    }
    output = Path(str(getattr(runner, "run_dir", ""))) / "tick_native_observability.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _reject_unimplemented_native_tick_authority(cfg: Any) -> None:
    """Keep a reference tick route from silently absorbing native requests."""

    requested = tuple(
        name
        for name in (
            "native_runtime_enable",
            "cxxrewrite_native_tick_authoritative_enable",
        )
        if bool(getattr(cfg, name, False))
    )
    if requested:
        raise TickNativeRouteError(
            "tick-native native execution is not implemented or qualified; "
            "reference route rejects native authority request: " + ", ".join(requested)
        )


def _write_tick_native_checkpoint(
    runner: Any,
    *,
    total_ticks: int,
    next_tick: int,
    state: tuple[Any, ...],
    schedule: dict[str, int | None],
) -> Path:
    from simulation.checkpoint_bundle_v2 import checkpoint_bundle_v2_save

    root = Path(str(getattr(runner, "run_dir", ""))) / "checkpoint_bundles_v2"
    bundle = root / f"bundle_tick{next_tick:012d}"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_bundle_v2_save(runner, str(bundle), generation_index=0, strict_replay_mode=True)
    _checkpoint_sidecar_path(bundle).write_text(
        json.dumps(
            {
                "schema_version": "eiel_tick_native_checkpoint_cursor_v1",
                "route_identity": TICK_NATIVE_EXECUTION_ROUTE_V1,
                "total_ticks": total_ticks,
                "next_tick": next_tick,
                "schedule": schedule,
                "continuation": {
                    "injury_recover_mult": state[2],
                    "deaths_this_span": state[3],
                    "successful_births": state[4],
                    "gen_peak_injury": state[9],
                    "failed_births": state[10],
                    "rest_streaks": {str(key): value for key, value in state[11].items()},
                },
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle


def _resume_cursor(
    bundle_dir: str | Path,
    *,
    total_ticks: int,
    schedule: dict[str, int | None],
) -> tuple[int, tuple[Any, ...]]:
    sidecar = _checkpoint_sidecar_path(Path(bundle_dir))
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TickNativeRouteError("tick-native checkpoint cursor sidecar is missing") from exc
    if (
        value.get("schema_version") != "eiel_tick_native_checkpoint_cursor_v1"
        or value.get("route_identity") != TICK_NATIVE_EXECUTION_ROUTE_V1
        or value.get("total_ticks") != total_ticks
        or value.get("schedule") != schedule
    ):
        raise TickNativeRouteError("tick-native checkpoint cursor sidecar contradicts route")
    next_tick = _positive_int(value.get("next_tick"), "checkpoint next_tick")
    if next_tick > total_ticks:
        raise TickNativeRouteError("checkpoint next_tick exceeds total_ticks")
    continuation = value.get("continuation")
    if not isinstance(continuation, dict):
        raise TickNativeRouteError("tick-native checkpoint continuation state is missing")
    rest_raw = continuation.get("rest_streaks")
    if not isinstance(rest_raw, dict):
        raise TickNativeRouteError("tick-native checkpoint rest_streaks are missing")
    try:
        rest_streaks = {int(key): int(item) for key, item in rest_raw.items()}
        state = (
            {}, False, float(continuation["injury_recover_mult"]),
            int(continuation["deaths_this_span"]), int(continuation["successful_births"]),
            [], [], set(), 0, float(continuation["gen_peak_injury"]),
            int(continuation["failed_births"]), rest_streaks,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise TickNativeRouteError("tick-native checkpoint continuation state is invalid") from exc
    return next_tick, state


def execute_tick_native_reference_cpu_v1(
    runner: Any, *, resume_bundle_dir: str | Path | None = None
) -> dict[str, object]:
    """Execute a fresh, no-turnover, absolute-tick reference run on CPU.

    This compatibility entrypoint deliberately keeps the CPU-only public name.
    Call ``execute_tick_native_reference_v1`` for an explicitly selected CUDA
    reference run; neither reference path is a production-native provider.
    """

    cfg = getattr(runner, "sim_cfg", None)
    if str(getattr(cfg, "device", "cpu")) != "cpu" or str(
        getattr(cfg, "bank_device", "cpu")
    ) != "cpu":
        raise TickNativeRouteError("CPU tick-native reference entrypoint requires CPU")
    return execute_tick_native_reference_v1(
        runner, resume_bundle_dir=resume_bundle_dir
    )


def execute_tick_native_reference_v1(
    runner: Any, *, resume_bundle_dir: str | Path | None = None
) -> dict[str, object]:
    """Execute the explicit CPU/CUDA reference route, never a native-provider route.

    The route is reference-owned on CPU or CUDA. Its horizon is
    ``total_ticks`` and tick ``N`` is never executed for a horizon of ``N``.
    Generation-shaped compatibility fields are not consulted for duration or
    turnover authority.
    """

    cfg = getattr(runner, "sim_cfg", None)
    if cfg is None:
        raise TickNativeRouteError("tick-native route requires sim_cfg")
    _reject_unimplemented_native_tick_authority(cfg)
    total_ticks = _positive_int(getattr(cfg, "total_ticks", None), "total_ticks")
    device = str(getattr(cfg, "device", "cpu"))
    bank_device = str(getattr(cfg, "bank_device", "cpu"))
    if device not in {"cpu", "cuda"} or bank_device != device:
        raise TickNativeRouteError(
            "tick-native reference route requires matching cpu/cpu or cuda/cuda devices"
        )
    schedule = _route_schedule(runner, total_ticks=total_ticks)
    _write_tick_native_observability_contract(
        runner, total_ticks=total_ticks, schedule=schedule
    )
    checkpoint_every = schedule["checkpoint_every_ticks"]
    runtime_partition_ticks = int(schedule["runtime_partition_ticks"])

    # This mirrors only the fresh-reference initialization needed before the
    # existing tick body.  It intentionally omits generation-final metrics,
    # selection, synchronized turnover, and generation checkpoint handling.
    runner.pop.sim_cfg = cfg
    runner._current_gen = 0
    runner._tick_native_execution_route = True
    runner._tick_native_total_ticks = total_ticks
    # The shared reference tick body is still parameterized by a local span
    # length.  On this new route that span is derived *from* total_ticks and is
    # never used to synthesize a generation horizon or a turnover boundary.
    cfg.steps_per_generation = total_ticks

    if resume_bundle_dir is None:
        runner.pop.init_random()
        runner._ensure_bank()
        runner._place_agents()
        next_tick = 0
        state = runner._start_tick_native_span()
    else:
        next_tick, state = _resume_cursor(
            resume_bundle_dir, total_ticks=total_ticks, schedule=schedule
        )
        runner._eiel_resume_start_tick = next_tick
    # A partition is a committed native/reference span, so it is the least
    # intrusive truthful cadence for the live cursor. This deliberately adds
    # no callback inside the tick body.
    from monitoring.slot_telemetry import start_tick_runner_slot_lease

    slot_lease = start_tick_runner_slot_lease(
        runner,
        start_absolute_tick=next_tick,
        target_absolute_tick=total_ticks,
    )
    try:
        while next_tick < total_ticks:
            partition_end = min(
                total_ticks,
                ((next_tick // runtime_partition_ticks) + 1) * runtime_partition_ticks,
            )
            checkpoint_end = total_ticks
            if checkpoint_every:
                checkpoint_end = min(
                    total_ticks, ((next_tick // checkpoint_every) + 1) * checkpoint_every
                )
            end_tick = min(partition_end, checkpoint_end)
            runner._tick_native_start_next_tick = next_tick
            runner._tick_native_end_next_tick = end_tick
            if next_tick:
                runner._eiel_resume_start_tick = next_tick
            run = runner._run_ticks(
                0, state[0], state[2], state[3], state[4], state[10], state[9], state[11]
            )
            state = (state[0], state[1], state[2], run[0], run[1], state[5], state[6], state[7], state[8], run[3], run[2], state[11])
            next_tick = int(getattr(runner, "_pred_tick", 0) or 0)
            if next_tick != end_tick:
                raise TickNativeRouteError("tick-native tick span did not commit its requested cursor")
            slot_lease.update(
                next_absolute_tick=next_tick,
                status="running",
                stage=f"tick partition committed at {next_tick}",
            )
            if checkpoint_every and next_tick % checkpoint_every == 0:
                _write_tick_native_checkpoint(
                    runner,
                    total_ticks=total_ticks,
                    next_tick=next_tick,
                    state=state,
                    schedule=schedule,
                )
        if next_tick != total_ticks:
            raise TickNativeRouteError(
                f"tick-native route cursor mismatch: expected {total_ticks}, got {next_tick}"
            )
        result: dict[str, object] = {
            "schema_version": "eiel_tick_native_execution_manifest_v1",
            "route_identity": TICK_NATIVE_EXECUTION_ROUTE_V1,
            "execution_backend": f"reference_{device}",
            "total_ticks": total_ticks,
            "next_tick": next_tick,
            "absolute_tick": None,
            "legacy_turnover_interval_ticks": None,
            **schedule,
        }
        output = Path(str(getattr(runner, "run_dir", ""))) / "tick_native_manifest.json"
        output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        runner._tick_native_route_state = dict(result)
        slot_lease.close(status="complete")
        return result
    except Exception as exc:
        slot_lease.close(status="failed", error=str(exc))
        raise


__all__ = [
    "LEGACY_GENERATION_ROUTE_V1",
    "TICK_NATIVE_EXECUTION_ROUTE_V1",
    "TickNativeRouteError",
    "execute_tick_native_reference_v1",
    "execute_tick_native_reference_cpu_v1",
    "maybe_capture_tick_native_report_v1",
]
