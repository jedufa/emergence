from __future__ import annotations

"""Live runner gate for the RUNNERCORE-0 supported-profile contract."""

import json
from pathlib import Path
import time
from typing import Any

from simulation.runnercore.profile import (
    RUNNERCORE_DECISION_BLOCKED_BY_BOUNDARY,
    candidate_requested,
    classify_supported_profile,
    profile_contract_payload,
    runnercore_requested,
)
from simulation.runnercore.state import (
    import_generation_start,
    state_summary_payload,
    validate_state_against_runner,
)


SUMMARY_SCHEMA_VERSION = "runnercore0_supported_profile_summary_v1"


class RunnerCoreUnsupportedError(RuntimeError):
    """Raised when a candidate/native RunnerCore path would cross an unsupported boundary."""


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _summary(runner: Any) -> dict[str, Any]:
    current = getattr(runner, "_runnercore0_supported_profile_summary", None)
    if isinstance(current, dict):
        return current
    current = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "requested_ticks": 0,
        "shadow_import_ticks": 0,
        "candidate_fail_closed_ticks": 0,
        "profile_blocked_ticks": 0,
        "shadow_validation_failures": 0,
        "per_tick_agent_sync_rows": 0,
        "first_divergence": None,
        "last_support_report": None,
        "last_state_summary": None,
        "last_validation": None,
        "contract": profile_contract_payload(),
    }
    setattr(runner, "_runnercore0_supported_profile_summary", current)
    return current


def _write_summary_if_requested(runner: Any, summary: dict[str, Any]) -> Path | None:
    cfg = getattr(runner, "sim_cfg", None)
    if not _as_bool(getattr(cfg, "runnercore_write_summary", True)):
        return None
    run_dir = str(getattr(runner, "run_dir", "") or "")
    if not run_dir:
        return None
    filename = str(getattr(cfg, "runnercore_summary_filename", "runnercore0_supported_profile_summary.json") or "")
    if not filename:
        return None
    out = Path(run_dir) / "metrics" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out


def build_runnercore_summary(runner: Any) -> dict[str, Any]:
    return dict(_summary(runner))


def maybe_runnercore_supported_tick(
    runner: Any,
    *,
    alive: Any,
    generation: int,
    tick: int,
    buckets: dict[str, float] | None = None,
) -> dict[str, Any] | None:
    cfg = getattr(runner, "sim_cfg", None)
    if cfg is None or not runnercore_requested(cfg):
        return None

    started = time.perf_counter()
    summary = _summary(runner)
    summary["requested_ticks"] = int(summary.get("requested_ticks", 0)) + 1
    report = classify_supported_profile(cfg)
    summary["last_support_report"] = report

    fail_closed = _as_bool(getattr(cfg, "runnercore_fail_closed_on_unsupported", True))
    if not bool(report.get("shadow_contract_supported", False)):
        summary["profile_blocked_ticks"] = int(summary.get("profile_blocked_ticks", 0)) + 1
        _write_summary_if_requested(runner, summary)
        if fail_closed:
            msg = "RUNNERCORE0-D/PROFILE-CONTRACT-MISMATCH: " + ",".join(report.get("profile_blockers", []))
            raise RunnerCoreUnsupportedError(msg)
        return report

    if candidate_requested(cfg):
        summary["candidate_fail_closed_ticks"] = int(summary.get("candidate_fail_closed_ticks", 0)) + 1
        _write_summary_if_requested(runner, summary)
        if fail_closed:
            blockers = ",".join(report.get("candidate_blockers", []))
            raise RunnerCoreUnsupportedError(f"{RUNNERCORE_DECISION_BLOCKED_BY_BOUNDARY}: {blockers}")
        return report

    if _as_bool(getattr(cfg, "runnercore_shadow_contract_enable", False)) or _as_bool(
        getattr(cfg, "runnercore_supported_profile_enable", False)
    ):
        state = import_generation_start(
            runner,
            alive=alive,
            generation=int(generation),
            tick=int(tick),
            profile=str(report.get("supported_profile", "")),
        )
        validation = validate_state_against_runner(state, runner, alive=alive)
        state_summary = state_summary_payload(state)
        summary["shadow_import_ticks"] = int(summary.get("shadow_import_ticks", 0)) + 1
        summary["per_tick_agent_sync_rows"] = int(summary.get("per_tick_agent_sync_rows", 0)) + int(state.alive_count)
        summary["last_state_summary"] = state_summary
        summary["last_validation"] = validation
        if not bool(validation.get("ok", False)):
            summary["shadow_validation_failures"] = int(summary.get("shadow_validation_failures", 0)) + 1
            if summary.get("first_divergence") is None:
                summary["first_divergence"] = validation
        _write_summary_if_requested(runner, summary)

    if buckets is not None:
        buckets["runnercore0_contract_total"] = float(
            buckets.get("runnercore0_contract_total", 0.0) + (time.perf_counter() - started)
        )
    return report
