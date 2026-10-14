from __future__ import annotations

"""Campaign output storage profile and preflight helpers.

This module is deliberately policy-only. It classifies and controls diagnostics,
checkpoint cadence, and storage metadata for campaign launches without changing
simulation mechanics.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


COMPACT_DEFAULT = "compact_default"
COMPACT_FUNNEL = "compact_funnel"
FAST_SCAN = "fast_scan"
REPEAT_PARITY = "repeat_parity"
DEBUG_EVENT_TRACE = "debug_event_trace"
FULL_REPLAY = "full_replay"

OUTPUT_PROFILES = frozenset(
    {
        COMPACT_DEFAULT,
        COMPACT_FUNNEL,
        FAST_SCAN,
        REPEAT_PARITY,
        DEBUG_EVENT_TRACE,
        FULL_REPLAY,
    }
)

EVENT_TRACE_PROFILES = frozenset({DEBUG_EVENT_TRACE, FULL_REPLAY})


PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
    COMPACT_DEFAULT: {
        "checkpoint_policy": "periodic checkpoints disabled; compact summaries/configs/fingerprints retained",
        "event_jsonl_policy": "event-level funnel JSONL disabled",
        "large_output_allowed": False,
    },
    COMPACT_FUNNEL: {
        "checkpoint_policy": "periodic checkpoints disabled; per-generation lineage funnel summaries retained",
        "event_jsonl_policy": "event-level funnel JSONL disabled",
        "large_output_allowed": False,
    },
    FAST_SCAN: {
        "checkpoint_policy": "periodic checkpoints disabled; minimal output profile",
        "event_jsonl_policy": "event-level funnel JSONL disabled",
        "large_output_allowed": False,
    },
    REPEAT_PARITY: {
        "checkpoint_policy": "periodic checkpoints disabled unless explicit sparse/final replay is requested",
        "event_jsonl_policy": "event-level funnel JSONL disabled unless explicitly elevated",
        "large_output_allowed": False,
    },
    DEBUG_EVENT_TRACE: {
        "checkpoint_policy": "sparse checkpoints only unless separately acknowledged",
        "event_jsonl_policy": "event-level funnel JSONL may be enabled explicitly",
        "large_output_allowed": True,
    },
    FULL_REPLAY: {
        "checkpoint_policy": "per-generation checkpoints may be enabled with large-output acknowledgement",
        "event_jsonl_policy": "event-level funnel JSONL may be enabled explicitly",
        "large_output_allowed": True,
    },
}


@dataclass(frozen=True)
class OutputSettings:
    profile: str
    checkpoint_period_gen: int
    behavior_log_enable: bool
    lineage_funnel_reporting_enable: bool
    lineage_funnel_write_per_generation: bool
    lineage_funnel_write_events: bool
    lineage_funnel_include_patch_context: bool
    lineage_funnel_include_basin_context: bool
    runnercore47_developmental_trace_buffered_emit_enable: bool
    allow_large_output_profile: bool
    allow_generation_checkpointing: bool
    generations: int | None
    chronicle_enable: bool
    chronicle_profile: str
    chronicle_max_stored_events_per_generation: int
    chronicle_estimated_event_bytes: int


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)


def _cli_flag_value(cli: Sequence[str], names: Iterable[str]) -> str | None:
    name_set = {str(name) for name in names}
    for idx, token in enumerate(cli):
        if str(token) in name_set and idx + 1 < len(cli):
            return str(cli[idx + 1])
    return None


def _cli_knobs(cli: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    idx = 0
    while idx < len(cli):
        token = str(cli[idx])
        if token == "--knob" and idx + 1 < len(cli):
            raw = str(cli[idx + 1]).strip()
            if "=" in raw:
                key, value = raw.split("=", 1)
                out[key.strip()] = value.strip()
            idx += 2
            continue
        idx += 1
    return out


def validate_output_profile(profile: str | None) -> str:
    selected = str(profile or COMPACT_DEFAULT).strip() or COMPACT_DEFAULT
    if selected not in OUTPUT_PROFILES:
        raise ValueError(
            f"unknown campaign output profile: {selected!r}; expected one of {sorted(OUTPUT_PROFILES)}"
        )
    return selected


def campaign_output_profile_knobs(
    *,
    profile: str | None,
    allow_large_output: bool = False,
    allow_generation_checkpointing: bool = False,
) -> list[str]:
    selected = validate_output_profile(profile)
    knobs = [
        f"sim_cfg.campaign_output_profile={selected}",
        f"sim_cfg.allow_large_output_profile={'true' if allow_large_output else 'false'}",
        f"sim_cfg.allow_generation_checkpointing={'true' if allow_generation_checkpointing else 'false'}",
    ]
    if selected in {COMPACT_DEFAULT, FAST_SCAN}:
        knobs.extend(
            [
                "sim_cfg.behavior_log_enable=false",
                "sim_cfg.checkpoint_bundle_v2_period_gen=0",
                "sim_cfg.lineage_reproduction_funnel_write_events=false",
                "sim_cfg.lineage_reproduction_funnel_write_per_generation=false",
                "sim_cfg.lineage_reproduction_funnel_include_patch_context=false",
                "sim_cfg.lineage_reproduction_funnel_include_basin_context=false",
            ]
        )
    elif selected == COMPACT_FUNNEL:
        knobs.extend(
            [
                "sim_cfg.behavior_log_enable=false",
                "sim_cfg.checkpoint_bundle_v2_period_gen=0",
                "sim_cfg.lineage_reproduction_funnel_write_events=false",
                "sim_cfg.lineage_reproduction_funnel_write_per_generation=true",
                "sim_cfg.lineage_reproduction_funnel_include_patch_context=true",
                "sim_cfg.lineage_reproduction_funnel_include_basin_context=true",
                "sim_cfg.runnercore47_developmental_trace_buffered_emit_enable=false",
            ]
        )
    elif selected == REPEAT_PARITY:
        knobs.extend(
            [
                "sim_cfg.behavior_log_enable=false",
                "sim_cfg.lineage_reproduction_funnel_write_events=false",
            ]
        )
    return knobs


def resolve_output_settings(cli: Sequence[str], *, default_profile: str = COMPACT_DEFAULT) -> OutputSettings:
    knobs = _cli_knobs(cli)
    profile = validate_output_profile(knobs.get("sim_cfg.campaign_output_profile", default_profile))
    v2_flag = _cli_flag_value(cli, ["--checkpoint_bundle_v2_period_gen"])
    legacy_flag = _cli_flag_value(cli, ["--checkpoint_period_gen"])
    checkpoint_period = _as_int(v2_flag if v2_flag is not None else legacy_flag, 0)
    if "sim_cfg.checkpoint_bundle_v2_period_gen" in knobs:
        checkpoint_period = _as_int(knobs.get("sim_cfg.checkpoint_bundle_v2_period_gen"), checkpoint_period)

    generations_raw = _cli_flag_value(cli, ["--generations", "--gen"])
    if "sim_cfg.generations" in knobs:
        generations_raw = knobs["sim_cfg.generations"]
    generations = _as_int(generations_raw, -1)
    generations_value = generations if generations >= 0 else None

    return OutputSettings(
        profile=profile,
        checkpoint_period_gen=checkpoint_period,
        behavior_log_enable=_as_bool(knobs.get("sim_cfg.behavior_log_enable"), True),
        lineage_funnel_reporting_enable=_as_bool(
            knobs.get("sim_cfg.lineage_reproduction_funnel_reporting_enable"), False
        ),
        lineage_funnel_write_per_generation=_as_bool(
            knobs.get("sim_cfg.lineage_reproduction_funnel_write_per_generation"), True
        ),
        lineage_funnel_write_events=_as_bool(knobs.get("sim_cfg.lineage_reproduction_funnel_write_events"), False),
        lineage_funnel_include_patch_context=_as_bool(
            knobs.get("sim_cfg.lineage_reproduction_funnel_include_patch_context"), True
        ),
        lineage_funnel_include_basin_context=_as_bool(
            knobs.get("sim_cfg.lineage_reproduction_funnel_include_basin_context"), True
        ),
        runnercore47_developmental_trace_buffered_emit_enable=_as_bool(
            knobs.get("sim_cfg.runnercore47_developmental_trace_buffered_emit_enable"), False
        ),
        allow_large_output_profile=_as_bool(knobs.get("sim_cfg.allow_large_output_profile"), False),
        allow_generation_checkpointing=_as_bool(knobs.get("sim_cfg.allow_generation_checkpointing"), False),
        generations=generations_value,
        chronicle_enable=_as_bool(knobs.get("sim_cfg.chronicle_enable"), False),
        chronicle_profile=str(knobs.get("sim_cfg.chronicle_profile", "notable") or "notable"),
        chronicle_max_stored_events_per_generation=max(
            1, _as_int(knobs.get("sim_cfg.chronicle_max_stored_events_per_generation"), 32)
        ),
        chronicle_estimated_event_bytes=max(
            256, _as_int(knobs.get("sim_cfg.chronicle_estimated_event_bytes"), 2048)
        ),
    )


def classify_storage_risk(settings: OutputSettings, *, planned_cells: int = 1) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    risk = "compact"

    if settings.checkpoint_period_gen == 1:
        risk = "full_replay_heavy"
        if settings.profile != FULL_REPLAY:
            errors.append("checkpoint_period_1_requires_full_replay_profile")
        if not settings.allow_generation_checkpointing:
            errors.append("checkpoint_period_1_requires_allow_generation_checkpointing")
        if not settings.allow_large_output_profile:
            errors.append("checkpoint_period_1_requires_allow_large_output")
    elif settings.checkpoint_period_gen > 1:
        risk = "moderate"
        warnings.append("periodic_checkpoints_enabled")

    if settings.lineage_funnel_write_events:
        risk = "large" if risk != "full_replay_heavy" else risk
        if settings.profile not in EVENT_TRACE_PROFILES:
            errors.append("event_jsonl_requires_debug_event_trace_or_full_replay_profile")
        if not settings.allow_large_output_profile:
            errors.append("event_jsonl_requires_allow_large_output")
        warnings.append("lineage_funnel_event_jsonl_enabled")

    if settings.lineage_funnel_write_per_generation and settings.lineage_funnel_reporting_enable:
        if risk == "compact":
            risk = "moderate"
        warnings.append("per_generation_funnel_summary_enabled")

    if settings.behavior_log_enable:
        if risk == "compact":
            risk = "moderate"
        warnings.append("behavior_log_enabled")

    if settings.runnercore47_developmental_trace_buffered_emit_enable:
        if risk == "compact":
            risk = "moderate"
        warnings.append("developmental_trace_jsonl_enabled")

    chronicle_estimated_events = 0
    chronicle_estimated_bytes = 0
    if settings.chronicle_enable and settings.chronicle_profile != "off":
        generations = max(0, int(settings.generations or 0))
        chronicle_estimated_events = (
            generations * max(1, int(planned_cells)) * settings.chronicle_max_stored_events_per_generation
        )
        chronicle_estimated_bytes = chronicle_estimated_events * settings.chronicle_estimated_event_bytes
        warnings.append("scientific_chronicle_enabled")
        if settings.chronicle_profile in {"scientific_full", "debug_trace"}:
            risk = "large" if risk != "full_replay_heavy" else risk
            if not settings.allow_large_output_profile:
                errors.append("heavy_chronicle_profile_requires_allow_large_output")
        elif risk == "compact":
            risk = "moderate"

    if planned_cells >= 32 and risk in {"moderate", "large"}:
        warnings.append("many_cells_amplify_output_size")

    return {
        "schema_version": "eiel_campaign_storage_preflight_v1",
        "profile": settings.profile,
        "risk": risk,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "settings": settings.__dict__,
        "planned_cells": int(planned_cells),
        "chronicle_estimated_events": int(chronicle_estimated_events),
        "chronicle_estimated_bytes": int(chronicle_estimated_bytes),
    }


def build_storage_plan(
    *,
    registry_name: str,
    resolved_rows: Sequence[Mapping[str, Any]],
    output_profile: str,
) -> dict[str, Any]:
    rows = []
    worst_rank = {"compact": 0, "moderate": 1, "large": 2, "full_replay_heavy": 3}
    worst = "compact"
    errors: list[str] = []
    warnings: list[str] = []
    for row in resolved_rows:
        cli = list(row.get("cli", []) or [])
        settings = resolve_output_settings(cli, default_profile=output_profile)
        risk = classify_storage_risk(settings, planned_cells=len(resolved_rows))
        if worst_rank[str(risk["risk"])] > worst_rank[worst]:
            worst = str(risk["risk"])
        for err in risk["errors"]:
            label = f"{row.get('run_id', '')}: {err}"
            errors.append(label)
        for warning in risk["warnings"]:
            label = f"{row.get('run_id', '')}: {warning}"
            warnings.append(label)
        rows.append(
            {
                "run_id": row.get("run_id", ""),
                "runtime_run_id": row.get("runtime_run_id", ""),
                "run_tag": row.get("run_tag", ""),
                "risk": risk["risk"],
                "ok": risk["ok"],
                "errors": risk["errors"],
                "warnings": risk["warnings"],
                "settings": risk["settings"],
            }
        )
    return {
        "schema_version": "eiel_campaign_storage_plan_v1",
        "registry_name": str(registry_name),
        "output_profile": validate_output_profile(output_profile),
        "planned_cells": len(resolved_rows),
        "risk": worst,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "profile_definitions": PROFILE_DEFINITIONS,
        "rows": rows,
    }


def write_storage_plan(path: Path, plan: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_runtime_storage(root: Path, *, top_n: int = 20, output_profile: str = "") -> dict[str, Any]:
    root = Path(root)
    categories = {
        "checkpoints": 0,
        "metrics": 0,
        "event_jsonl": 0,
        "chronicle": 0,
        "config_manifests": 0,
        "logs": 0,
        "other": 0,
    }
    file_count = 0
    total_bytes = 0
    checkpoint_bundle_count = 0
    event_jsonl_count = 0
    event_jsonl_bytes = 0
    largest: list[tuple[int, str]] = []
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            try:
                size = int(path.stat().st_size)
            except OSError:
                continue
            rel = str(path.relative_to(root)).replace("\\", "/")
            file_count += 1
            total_bytes += size
            if "checkpoint_bundles_v2/" in rel or rel.startswith("checkpoint_bundles_v2/"):
                categories["checkpoints"] += size
                if path.name == "manifest.json":
                    checkpoint_bundle_count += 1
            elif rel.startswith("chronicle/") or "/chronicle/" in rel:
                categories["chronicle"] += size
            elif rel.endswith(".jsonl") and (
                "funnel" in rel or "developmental_stage_trace" in rel or "developmental_lock_trace" in rel
            ):
                categories["event_jsonl"] += size
                event_jsonl_count += 1
                event_jsonl_bytes += size
            elif rel.startswith("metrics/") or rel == "metrics.csv" or rel.endswith("/metrics.csv"):
                categories["metrics"] += size
            elif path.name in {"config.json", "manifest.json", "run_config.csv", "storage_plan.json"}:
                categories["config_manifests"] += size
            elif rel.endswith(".log") or "/logs/" in rel or rel.startswith("logs/"):
                categories["logs"] += size
            else:
                categories["other"] += size
            largest.append((size, rel))
    largest.sort(reverse=True)
    risk = "compact"
    if event_jsonl_bytes or checkpoint_bundle_count:
        risk = "moderate"
    return {
        "schema_version": "eiel_runtime_storage_summary_v1",
        "root": str(root),
        "exists": root.exists(),
        "output_profile": str(output_profile or ""),
        "evidence_compatibility": risk,
        "total_bytes": total_bytes,
        "total_files": file_count,
        "bytes_by_category": categories,
        "top_largest_files": [
            {"bytes": int(size), "path": rel} for size, rel in largest[: max(0, int(top_n))]
        ],
        "checkpoint_bundle_count": checkpoint_bundle_count,
        "event_jsonl_file_count": event_jsonl_count,
        "event_jsonl_bytes": event_jsonl_bytes,
    }


def write_storage_summary(path: Path, root: Path, *, output_profile: str = "", top_n: int = 20) -> dict[str, Any]:
    summary = summarize_runtime_storage(root, top_n=top_n, output_profile=output_profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
