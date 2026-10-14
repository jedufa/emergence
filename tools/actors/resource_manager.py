"""Advisory EIEL Resource Manager built on the read-only ERM."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.canonical import canonical_sha256
from tools import resource_monitor as erm
from tools.actors.common import (
    ActorArgumentParser,
    ActorError,
    actor_state_root,
    emit_actor_error,
    emit_payload,
    forbidden_operation_error,
    gpu_uuid_equal_v1,
    new_record,
    read_json,
    read_structured,
    require_explicit_schema_version,
    stable_id,
    utc_now,
    with_guidance,
)
from tools.actors.reference_integrity import actor_source_ref
from tools.actors.store import ActorRecordStore
from tools.actors.release import operator_metadata

ACTOR_NAME = "resourceManager"
STORE_NAMESPACE = "resources"
OBSERVATION_RECORD_TYPE = "resource_observation"
ASSESSMENT_RECORD_TYPE = "resource_admission_recommendation"
DEMAND_SCHEMA_VERSION = "eiel_resource_demand_v1"
ASSESSMENT_SCHEMA_VERSION = "eiel_resource_admission_recommendation_v1_2"
DECISIONS = ("ADMIT", "ADMIT_WITH_LIMITS", "WAIT", "REJECT", "MANUAL_REVIEW")
GPU_ADMISSION_MODES = ("shared_functional", "exclusive")
EXCLUSIVE_WORKLOAD_MARKERS = frozenset(
    {
        "benchmark",
        "checkpoint",
        "critical",
        "formal",
        "performance",
        "qualification",
        "qualifying",
        "residency",
        "resume",
        "timing",
        "vram",
    }
)
SHARED_FUNCTIONAL_LIMITS = (
    "shared GPU execution is suitable for functional or determinism diagnostics only",
    "bandwidth, SM contention, VRAM-margin, residency-timing, and performance measurements are non-gating while shared",
    "formal qualification and critical checkpoint/resume or timing evidence require an exclusive GPU rerun",
)


def _store(args: argparse.Namespace) -> ActorRecordStore:
    return ActorRecordStore(actor_state_root(args.state_root), STORE_NAMESPACE)


def _common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root")
    parser.add_argument("--generated-at")
    parser.add_argument("--output")
    parser.add_argument("--human", action="store_true")


def _monitor_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repo-root",
        "--worktree",
        dest="repo_root",
        type=Path,
        default=Path.cwd(),
        help="Repository/worktree root to inspect (--worktree is a compatibility alias).",
    )
    parser.add_argument("--runtime-root", type=Path, default=erm.DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--live-slots-root", type=Path)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--ste-path", type=Path, action="append", default=[])
    parser.add_argument("--cpu-sample-interval-s", type=float, default=0.1)
    parser.add_argument("--stale-heartbeat-s", type=float, default=300.0)
    parser.add_argument("--min-safe-ram-gib", type=float, default=4.0)
    parser.add_argument("--min-safe-vram-gib", type=float, default=2.0)
    parser.add_argument(
        "--usable-vram-gib",
        type=float,
        help="Optional host/GPU capacity profile used for peak shared-admission accounting.",
    )
    parser.add_argument("--report", help="Use a saved ERM report instead of inspecting the host.")
    parser.add_argument("--max-observation-age-s", type=float, default=300.0)


def _monitor_config(args: argparse.Namespace) -> erm.MonitorConfig:
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    return erm.MonitorConfig(
        repo_root=Path(args.repo_root).expanduser().resolve(),
        runtime_root=runtime_root,
        proc_root=Path(args.proc_root).expanduser().resolve(),
        live_slots_root=(Path(args.live_slots_root).expanduser().resolve() if args.live_slots_root else runtime_root / "live/simulation-slots"),
        cpu_sample_interval_s=max(0.0, float(args.cpu_sample_interval_s)),
        stale_heartbeat_s=max(1.0, float(args.stale_heartbeat_s)),
        min_safe_ram_bytes=int(float(args.min_safe_ram_gib) * 1024**3),
        min_safe_vram_bytes=int(float(args.min_safe_vram_gib) * 1024**3),
        usable_vram_gib=(
            None
            if args.usable_vram_gib is None
            else max(0.0, float(args.usable_vram_gib))
        ),
        ste_paths=tuple(Path(path).expanduser().resolve() for path in args.ste_path),
    )


def _report(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(args.report, name="ERM report") if args.report else erm.collect_report(_monitor_config(args))
    if report.get("schema_version") != erm.SCHEMA_VERSION:
        raise ActorError(
            "unsupported ERM report schema_version",
            code="unsupported_resource_observation_schema",
            next_actions=[{"description": "generate a fresh observation", "command": "python -m tools.eiel resourceManager inspect"}],
        )
    return report


def _observation_record(report: Mapping[str, Any], *, generated_at: str) -> dict[str, Any]:
    observation_id = stable_id("ROBS", report)
    return new_record(
        actor=ACTOR_NAME,
        record_type=OBSERVATION_RECORD_TYPE,
        record_id=observation_id,
        status="OBSERVED",
        payload={
            "observation_id": observation_id,
            "report": dict(report),
            "advisory_only": True,
            "non_goals": ["no_schedule", "no_reservation", "no_launch", "no_cancel", "no_process_kill"],
        },
        generated_at=generated_at,
        action="inspect",
    )


def _parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(dt.timezone.utc)


def observation_age_seconds(report: Mapping[str, Any], *, now_utc: str) -> float | None:
    observed = _parse_time(report.get("generated_at_utc"))
    now = _parse_time(now_utc)
    if observed is None or now is None:
        return None
    return (now - observed).total_seconds()


def _gib(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value) / float(1024**3)
    except (TypeError, ValueError):
        return None


def _storage_available_gib(report: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for row in dict(report.get("storage") or {}).get("roots") or []:
        if isinstance(row, Mapping):
            parsed = _gib(row.get("free_bytes", row.get("available_bytes")))
            if parsed is not None:
                values.append(parsed)
    return min(values) if values else None


def _max_vram_free_gib(report: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for row in dict(report.get("gpu") or {}).get("gpus") or []:
        if isinstance(row, Mapping):
            parsed = _gib(row.get("vram_free_bytes"))
            if parsed is not None:
                values.append(parsed)
    return max(values) if values else None


def _nonnegative_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed >= 0 else None


def _normalize_gpu_admission_mode(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "shared": "shared_functional",
        "nonexclusive": "shared_functional",
        "non_exclusive": "shared_functional",
        "exclusive_critical": "exclusive",
    }
    text = aliases.get(text, text)
    return text if text in GPU_ADMISSION_MODES else None


def _workload_class_requires_exclusive(value: Any) -> bool:
    tokens = {
        token
        for token in str(value or "").lower().replace("-", "_").split("_")
        if token
    }
    return bool(tokens.intersection(EXCLUSIVE_WORKLOAD_MARKERS))


def _default_gpu_admission_mode(source: Mapping[str, Any]) -> str:
    if source.get("exclusive_gpu") is True:
        return "exclusive"
    raw_explicit = source.get("gpu_admission_mode")
    if raw_explicit not in (None, ""):
        explicit = _normalize_gpu_admission_mode(raw_explicit)
        return explicit or str(raw_explicit).strip().lower()
    if _workload_class_requires_exclusive(source.get("workload_class")):
        return "exclusive"
    return "shared_functional"


def _pid(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _gpu_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("device") or "").strip().lower() in {"gpu", "cuda"}


def _occupant_mode(row: Mapping[str, Any]) -> str:
    explicit = row.get("gpu_admission_mode")
    if explicit not in (None, ""):
        return _normalize_gpu_admission_mode(explicit) or "unknown"
    if isinstance(row.get("exclusive_gpu"), bool):
        return "exclusive" if row["exclusive_gpu"] else "shared_functional"
    if _workload_class_requires_exclusive(row.get("workload_class")):
        return "exclusive"
    return "unknown"


def _row_peak_vram_gib(row: Mapping[str, Any]) -> float | None:
    peak = _nonnegative_number(
        row.get("peak_vram_gib", row.get("declared_peak_vram_gib"))
    )
    if peak is not None:
        return peak
    peak_bytes = row.get("peak_vram_bytes")
    return _gib(peak_bytes) if peak_bytes is not None else None


def _row_observed_vram_gib(row: Mapping[str, Any]) -> float | None:
    for field in ("vram_used_bytes", "used_memory_bytes", "used_vram_bytes"):
        if row.get(field) is not None:
            return _gib(row.get(field))
    return None


def _gpu_capacity_gib(report: Mapping[str, Any]) -> float | None:
    values: list[float] = []
    for row in dict(report.get("gpu") or {}).get("gpus") or []:
        if not isinstance(row, Mapping):
            continue
        usable = row.get("usable_vram_bytes", row.get("vram_total_bytes"))
        parsed = _gib(usable)
        if parsed is not None:
            values.append(parsed)
        elif _nonnegative_number(row.get("usable_vram_gib")) is not None:
            values.append(float(row["usable_vram_gib"]))
    return max(values) if values else None


def _gpu_occupants(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return compute occupants with explicit sharing metadata when available."""

    metadata_by_pid: dict[int, dict[str, Any]] = {}
    metadata_without_pid: list[dict[str, Any]] = []
    for source_name in ("running_eiel_workloads", "slot_heartbeats"):
        for row in report.get(source_name) or []:
            if not isinstance(row, Mapping) or not _gpu_row(row):
                continue
            copied = dict(row)
            pid = _pid(row.get("pid"))
            if pid is None:
                metadata_without_pid.append(copied)
                continue
            target = metadata_by_pid.setdefault(pid, {})
            for key, value in copied.items():
                if value not in (None, "", "UNKNOWN") or isinstance(value, bool):
                    target[key] = value

    occupants: list[dict[str, Any]] = []
    seen_pids: set[int] = set()
    for process in dict(report.get("gpu") or {}).get("processes") or []:
        if not isinstance(process, Mapping):
            continue
        pid = _pid(process.get("pid"))
        merged = dict(process)
        if pid is not None:
            seen_pids.add(pid)
            merged.update(metadata_by_pid.get(pid) or {})
        occupants.append(
            {
                "pid": pid or "UNKNOWN",
                "mode": _occupant_mode(merged),
                "peak_vram_gib": _row_peak_vram_gib(merged),
                "observed_vram_gib": _row_observed_vram_gib(process),
                "campaign": merged.get("campaign") or merged.get("campaign_id") or "UNKNOWN",
                "run_id": merged.get("run_id") or "UNKNOWN",
            }
        )

    for pid, metadata in metadata_by_pid.items():
        if pid in seen_pids:
            continue
        occupants.append(
            {
                "pid": pid,
                "mode": _occupant_mode(metadata),
                "peak_vram_gib": _row_peak_vram_gib(metadata),
                "observed_vram_gib": None,
                "campaign": metadata.get("campaign") or metadata.get("campaign_id") or "UNKNOWN",
                "run_id": metadata.get("run_id") or "UNKNOWN",
            }
        )

    for metadata in metadata_without_pid:
        occupants.append(
            {
                "pid": "UNKNOWN",
                "mode": _occupant_mode(metadata),
                "peak_vram_gib": _row_peak_vram_gib(metadata),
                "observed_vram_gib": None,
                "campaign": metadata.get("campaign") or metadata.get("campaign_id") or "UNKNOWN",
                "run_id": metadata.get("run_id") or "UNKNOWN",
            }
        )
    return occupants


def normalize_demand(raw: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(raw)
    if source.get("record_type") == "mission_specification" and isinstance(source.get("payload"), Mapping):
        source = dict(source["payload"])
    mission_id = source.get("mission_id")
    if isinstance(source.get("resource_demand"), Mapping):
        source = {**dict(source["resource_demand"]), "mission_id": mission_id or dict(source["resource_demand"]).get("mission_id")}
    if "schema_version" in source and source.get("schema_version") != DEMAND_SCHEMA_VERSION:
        require_explicit_schema_version(
            source,
            expected=DEMAND_SCHEMA_VERSION,
            label="resource demand",
        )
    demand = dict(source)
    demand["schema_version"] = DEMAND_SCHEMA_VERSION
    demand["device"] = str(demand.get("device") or "").lower()
    demand.setdefault("workload_class", "unspecified")
    demand.setdefault("exclusive_gpu", False)
    demand["gpu_admission_mode"] = _default_gpu_admission_mode(demand)
    if demand["gpu_admission_mode"] == "exclusive":
        demand["exclusive_gpu"] = True
    demand.setdefault("peak_vram_gib", None)
    demand.setdefault("minimum_ram_available_gib", 0.0)
    demand.setdefault("minimum_vram_available_gib", 0.0)
    demand.setdefault("minimum_storage_available_gib", 0.0)
    demand.setdefault("cpu_threads", 0)
    demand.setdefault("expected_duration_minutes", 0)
    demand.setdefault("mission_id", None)
    return demand


def validate_demand(demand: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if demand.get("schema_version") != DEMAND_SCHEMA_VERSION:
        errors.append("unsupported resource demand schema_version")
    if demand.get("device") not in {"cpu", "cuda", "either", "none"}:
        errors.append("device must be cpu, cuda, either, or none")
    for field in ("minimum_ram_available_gib", "minimum_vram_available_gib", "minimum_storage_available_gib", "cpu_threads", "expected_duration_minutes"):
        value = demand.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
            errors.append(f"{field} must be a non-negative number")
    if not isinstance(demand.get("exclusive_gpu"), bool):
        errors.append("exclusive_gpu must be boolean")
    if demand.get("gpu_admission_mode") not in GPU_ADMISSION_MODES:
        errors.append("gpu_admission_mode must be shared_functional or exclusive")
    peak = demand.get("peak_vram_gib")
    if peak is not None and _nonnegative_number(peak) is None:
        errors.append("peak_vram_gib must be a non-negative number or null")
    return errors


def load_resource_demand(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Load a persisted resource demand without defaulting its version."""

    require_explicit_schema_version(
        raw,
        expected=DEMAND_SCHEMA_VERSION,
        label="resource demand",
    )
    demand = normalize_demand(raw)
    errors = validate_demand(demand)
    if errors:
        raise ActorError(
            "resource demand is malformed",
            code="schema_record_malformed",
            problems=errors,
        )
    return demand


def load_resource_admission_recommendation(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Load one retained advisory recommendation and validate its nested demand."""

    require_explicit_schema_version(
        raw,
        expected=ASSESSMENT_SCHEMA_VERSION,
        label="resource admission recommendation",
    )
    assessment = dict(raw)
    errors: list[str] = []
    if assessment.get("decision") not in DECISIONS:
        errors.append("resource recommendation decision is invalid")
    if assessment.get("recommended_route") not in {"cpu", "cuda", "none"}:
        errors.append("resource recommendation route is invalid")
    if not isinstance(assessment.get("advisory_only"), bool):
        errors.append("resource recommendation advisory_only must be boolean")
    demand = assessment.get("demand")
    if not isinstance(demand, Mapping):
        errors.append("resource recommendation demand must be an object")
    else:
        try:
            assessment["demand"] = load_resource_demand(demand)
        except ActorError as exc:
            if exc.code.startswith("schema_version_"):
                raise
            errors.extend(exc.problems)
    for field in ("reason_codes", "reasons", "limits", "next_actions", "non_goals"):
        if not isinstance(assessment.get(field), list):
            errors.append(f"resource recommendation {field} must be a list")
    if not isinstance(assessment.get("observed"), Mapping):
        errors.append("resource recommendation observed must be an object")
    if not isinstance(assessment.get("operating_policy"), Mapping):
        errors.append("resource recommendation operating_policy must be an object")
    if errors:
        raise ActorError(
            "resource admission recommendation is malformed",
            code="schema_record_malformed",
            problems=errors,
        )
    return assessment


def _load_demand(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source_refs: list[dict[str, Any]] = []
    if args.demand:
        raw = read_structured(args.demand, name="resource demand")
        source_refs.append({"kind": "resource_demand", "uri": str(Path(args.demand).resolve())})
    elif args.mission:
        mission_store = ActorRecordStore(actor_state_root(args.state_root), "missions")
        mission = mission_store.get(args.mission)
        raw = dict(mission.get("payload") or {})
        source_refs.append(actor_source_ref("mission", mission))
    elif args.device:
        raw = {
            "device": args.device,
            "workload_class": args.workload_class or "ad_hoc",
            "minimum_ram_available_gib": args.minimum_ram_available_gib,
            "minimum_vram_available_gib": args.minimum_vram_available_gib,
            "minimum_storage_available_gib": args.minimum_storage_available_gib,
            "exclusive_gpu": args.exclusive_gpu,
            "gpu_admission_mode": args.gpu_admission_mode,
            "peak_vram_gib": args.peak_vram_gib,
        }
    else:
        raise ActorError(
            "resource assessment requires --mission, --demand, or an explicit --device",
            code="missing_resource_demand",
            next_actions=[{"description": "assess a mission", "command": "python -m tools.eiel resourceManager assess --mission MIS-..."}],
        )
    demand = normalize_demand(raw)
    if args.device:
        demand["device"] = args.device
    if args.gpu_admission_mode:
        demand["gpu_admission_mode"] = args.gpu_admission_mode
        if args.gpu_admission_mode == "exclusive":
            demand["exclusive_gpu"] = True
    if args.exclusive_gpu:
        demand["exclusive_gpu"] = True
        demand["gpu_admission_mode"] = "exclusive"
    if args.peak_vram_gib is not None:
        demand["peak_vram_gib"] = args.peak_vram_gib
    demand = normalize_demand(demand)
    errors = validate_demand(demand)
    if errors:
        raise ActorError(
            "invalid resource demand",
            code="invalid_resource_demand",
            problems=errors,
            next_actions=[{"description": "repair the demand file or mission resource_demand", "command": "python -m tools.eiel missionPlanner inspect MIS-..."}],
        )
    return demand, source_refs


def _evaluate_cuda_admission(
    demand: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    gpu_available: bool,
    unknown: bool,
    gpu_blocked: bool,
    vram_short: bool,
) -> dict[str, Any]:
    mode = str(demand.get("gpu_admission_mode") or "shared_functional")
    occupants = _gpu_occupants(report)
    capacity = _gpu_capacity_gib(report)
    candidate_peak = _nonnegative_number(demand.get("peak_vram_gib"))
    observed_peak = sum(
        float(row["observed_vram_gib"])
        for row in occupants
        if row.get("observed_vram_gib") is not None
    )
    declared_peaks = [
        float(row["peak_vram_gib"])
        for row in occupants
        if row.get("peak_vram_gib") is not None
    ]
    all_existing_peaks_known = len(declared_peaks) == len(occupants)
    existing_peak = sum(declared_peaks) if all_existing_peaks_known else None
    projected_peak = (
        existing_peak + candidate_peak
        if existing_peak is not None and candidate_peak is not None
        else None
    )
    result: dict[str, Any] = {
        "decision": "MANUAL_REVIEW",
        "route": "none",
        "state": "UNKNOWN",
        "reason_codes": [],
        "reasons": [],
        "limits": [],
        "occupants": occupants,
        "capacity_gib": capacity,
        "candidate_peak_vram_gib": candidate_peak,
        "observed_vram_used_gib": observed_peak,
        "existing_declared_peak_vram_gib": existing_peak,
        "projected_peak_vram_gib": projected_peak,
    }

    def finish(
        decision: str,
        state: str,
        code: str,
        reason: str,
        *,
        limits: Sequence[str] = (),
    ) -> dict[str, Any]:
        result.update(
            {
                "decision": decision,
                "route": "cuda" if decision in {"ADMIT", "ADMIT_WITH_LIMITS"} else "none",
                "state": state,
                "reason_codes": [code],
                "reasons": [reason],
                "limits": list(limits),
            }
        )
        return result

    if not gpu_available or unknown:
        return finish(
            "MANUAL_REVIEW",
            "UNKNOWN",
            "gpu_state_unknown",
            "CUDA safety cannot be attributed with sufficient confidence",
        )

    if mode not in GPU_ADMISSION_MODES:
        return finish(
            "MANUAL_REVIEW",
            "UNKNOWN",
            "gpu_admission_mode_unknown",
            "GPU admission mode is missing or unsupported",
        )

    if mode == "exclusive":
        if occupants or gpu_blocked:
            return finish(
                "WAIT",
                "BUSY_EXCLUSIVE_REQUIRED",
                "exclusive_gpu_conflict",
                "exclusive GPU admission requires no observed compute occupant",
            )
        if vram_short:
            return finish(
                "WAIT",
                "VRAM_HEADROOM_LOW",
                "vram_headroom_insufficient",
                "CUDA route lacks the requested current VRAM headroom",
            )
        if capacity is None:
            return finish(
                "MANUAL_REVIEW",
                "UNKNOWN",
                "vram_capacity_unknown",
                "usable GPU capacity is unknown; exclusive launch safety cannot be assessed",
            )
        if candidate_peak is not None and candidate_peak > capacity:
            return finish(
                "WAIT",
                "VRAM_CAPACITY_BLOCKED",
                "peak_vram_exceeds_capacity",
                f"declared peak VRAM {candidate_peak:.3f} GiB exceeds usable capacity {capacity:.3f} GiB",
            )
        if str(dict(report.get("recommendation") or {}).get("value") or "") == "CAUTION":
            return finish(
                "ADMIT_WITH_LIMITS",
                "IDLE_EXCLUSIVE_CAUTION",
                "erm_caution",
                "exclusive CUDA route is available but ERM reports caution",
                limits=("preserve the declared VRAM headroom",),
            )
        return finish(
            "ADMIT",
            "IDLE_EXCLUSIVE",
            "exclusive_gpu_available",
            "exclusive CUDA route is available with no observed compute occupant",
        )

    if vram_short:
        return finish(
            "WAIT",
            "VRAM_HEADROOM_LOW",
            "vram_headroom_insufficient",
            "CUDA route lacks the requested current VRAM headroom",
        )
    if not occupants:
        if gpu_blocked:
            return finish(
                "WAIT",
                "BUSY_UNKNOWN",
                "gpu_blocked",
                "an active GPU workload is reported without enough owner metadata to prove safe sharing",
            )
        if capacity is None:
            return finish(
                "MANUAL_REVIEW",
                "UNKNOWN",
                "vram_capacity_unknown",
                "usable GPU capacity is unknown; shared launch safety cannot be assessed",
            )
        if candidate_peak is None:
            return finish(
                "WAIT",
                "SHARED_PEAK_UNKNOWN",
                "shared_candidate_peak_unknown",
                "shared admission requires peak_vram_gib even on an idle GPU",
            )
        if candidate_peak is not None and candidate_peak > capacity:
            return finish(
                "WAIT",
                "VRAM_CAPACITY_BLOCKED",
                "peak_vram_exceeds_capacity",
                f"declared peak VRAM {candidate_peak:.3f} GiB exceeds usable capacity {capacity:.3f} GiB",
            )
        limits = list(SHARED_FUNCTIONAL_LIMITS)
        if str(dict(report.get("recommendation") or {}).get("value") or "") == "CAUTION":
            return finish(
                "ADMIT_WITH_LIMITS",
                "IDLE_SHARED_CAUTION",
                "erm_caution",
                "shared-functional CUDA route is available but ERM reports caution",
                limits=limits,
            )
        return finish(
            "ADMIT_WITH_LIMITS",
            "IDLE_SHARED",
            "shared_functional_gpu_available",
            "shared-functional CUDA route is available; no compute occupant is observed",
            limits=limits,
        )

    if any(row.get("mode") == "exclusive" for row in occupants):
        return finish(
            "WAIT",
            "BUSY_EXCLUSIVE",
            "exclusive_gpu_conflict",
            "an exclusive or critical CUDA workload currently owns the GPU",
        )
    if any(row.get("mode") == "unknown" for row in occupants):
        return finish(
            "WAIT",
            "BUSY_UNKNOWN",
            "gpu_blocked",
            "a current CUDA owner has no explicit sharing policy; do not infer that it is shareable",
        )
    if candidate_peak is None:
        return finish(
            "WAIT",
            "SHARED_PEAK_UNKNOWN",
            "shared_candidate_peak_unknown",
            "shared admission requires the candidate peak_vram_gib declaration",
        )
    if not all_existing_peaks_known:
        return finish(
            "WAIT",
            "SHARED_PEAK_UNKNOWN",
            "existing_peak_vram_unknown",
            "shared admission requires a peak_vram_gib declaration for every current CUDA owner",
        )
    if capacity is None:
        return finish(
            "MANUAL_REVIEW",
            "UNKNOWN",
            "vram_capacity_unknown",
            "usable GPU capacity is unknown; shared peak accounting cannot be assessed",
        )
    if any(
        row.get("observed_vram_gib") is not None
        and row["observed_vram_gib"] > float(row["peak_vram_gib"]) + 1e-6
        for row in occupants
    ):
        return finish(
            "WAIT",
            "SHARED_PEAK_UNDERREPORTED",
            "existing_peak_underreported",
            "an observed CUDA allocation already exceeds its owner's declared peak VRAM",
        )
    if projected_peak is not None and projected_peak > capacity:
        return finish(
            "WAIT",
            "VRAM_CAPACITY_BLOCKED",
            "shared_peak_vram_exceeds_capacity",
            f"projected shared peak VRAM {projected_peak:.3f} GiB exceeds usable capacity {capacity:.3f} GiB",
        )
    return finish(
        "ADMIT_WITH_LIMITS",
        "CONTENDED_SHARED",
        "shared_functional_gpu_fits",
        f"shared-functional peak VRAM {projected_peak:.3f} GiB fits usable capacity {capacity:.3f} GiB",
        limits=SHARED_FUNCTIONAL_LIMITS,
    )


def assess_dead_cuda_owner_reclaim_v1(
    *,
    attempt_id: str,
    resource_lease_digest: Mapping[str, Any],
    fencing_token_digest: Mapping[str, Any],
    continuation_lease_digest: Mapping[str, Any],
    continuation_fence_state: str,
    continuation_terminal_event_digest: Mapping[str, Any] | None,
    resource_started_event_digest: Mapping[str, Any],
    device_uuid: str,
    selected_resource_ids: Sequence[str],
    owner_pid: int,
    owner_process_start_identity: str,
    process_identity_observation: Mapping[str, Any],
    report: Mapping[str, Any],
    now_utc: str,
    max_observation_age_s: float = 5.0,
) -> dict[str, Any]:
    """Issue a fail-closed advisory proof for one crash-dead CUDA owner.

    The Scheduler remains the authority that may consume this proof.  This
    function only decides whether fresh procfs and device-specific NVIDIA
    telemetry jointly prove that the exact owner is gone.
    """

    age = observation_age_seconds(report, now_utc=now_utc)
    gpu = dict(report.get("gpu") or {})
    processes = [
        dict(row)
        for row in gpu.get("processes") or []
        if isinstance(row, Mapping)
    ]
    target_rows = [
        row
        for row in processes
        if row.get("pid") == owner_pid
        and gpu_uuid_equal_v1(row.get("gpu_uuid"), device_uuid)
    ]
    same_pid_unknown_device = [
        row
        for row in processes
        if row.get("pid") == owner_pid and not row.get("gpu_uuid")
    ]
    observed_devices = [
        row.get("uuid")
        for row in gpu.get("gpus") or []
        if isinstance(row, Mapping) and row.get("uuid")
    ]
    process_observation = dict(process_identity_observation)
    process_observed = _parse_time(process_observation.get("observed_at_utc"))
    assessed_at = _parse_time(now_utc)
    process_observation_age = (
        None
        if process_observed is None or assessed_at is None
        else (assessed_at - process_observed).total_seconds()
    )
    resource_observed = _parse_time(report.get("generated_at_utc"))
    reasons: list[str] = []
    if continuation_fence_state not in {"EXPIRED", "OWNER_DEAD_REQUIRED"}:
        reasons.append("CONTINUATION_FENCE_INVALID")
    elif continuation_fence_state == "EXPIRED" and not isinstance(
        continuation_terminal_event_digest, Mapping
    ):
        reasons.append("CONTINUATION_FENCE_INVALID")
    elif (
        continuation_fence_state == "OWNER_DEAD_REQUIRED"
        and continuation_terminal_event_digest is not None
    ):
        reasons.append("CONTINUATION_FENCE_INVALID")
    if age is None or age < 0 or age > max(1.0, float(max_observation_age_s)):
        reasons.append("STALE_RESOURCE_OBSERVATION")
    if gpu.get("available") is not True or gpu.get("process_query_complete") is not True:
        reasons.append("CUDA_PROCESS_TELEMETRY_INCOMPLETE")
    if not any(gpu_uuid_equal_v1(value, device_uuid) for value in observed_devices):
        reasons.append("DEVICE_UUID_UNOBSERVED")
    if (
        process_observation.get("schema_version")
        != "eiel_process_identity_observation_v1"
        or process_observation.get("pid") != owner_pid
        or process_observation.get("expected_process_start_identity")
        != owner_process_start_identity
        or process_observation.get("state") not in {"ABSENT", "REUSED"}
    ):
        reasons.append("OWNER_PROCESS_NOT_PROVEN_DEAD")
    if (
        process_observation_age is None
        or process_observation_age < 0
        or process_observation_age > max(1.0, float(max_observation_age_s))
    ):
        reasons.append("STALE_PROCESS_IDENTITY_OBSERVATION")
    if (
        process_observed is None
        or resource_observed is None
        or assessed_at is None
        or process_observed > resource_observed
        or resource_observed > assessed_at
    ):
        reasons.append("CAUSAL_EVIDENCE_ORDER_INVALID")
    if target_rows:
        reasons.append("OWNER_PID_STILL_HAS_TARGET_CUDA_CONTEXT")
    if same_pid_unknown_device:
        reasons.append("OWNER_PID_CUDA_DEVICE_UNKNOWN")
    decision = "RECLAIM" if not reasons else "WAIT"
    core = {
        "schema_version": "eiel_dead_cuda_owner_reclaim_proof_v2",
        "decision": decision,
        "attempt_id": attempt_id,
        "resource_lease_digest": dict(resource_lease_digest),
        "fencing_token_digest": dict(fencing_token_digest),
        "continuation_lease_digest": dict(continuation_lease_digest),
        "continuation_fence_state": continuation_fence_state,
        "continuation_terminal_event_digest": (
            None
            if continuation_terminal_event_digest is None
            else dict(continuation_terminal_event_digest)
        ),
        "resource_started_event_digest": dict(resource_started_event_digest),
        "device_uuid": device_uuid,
        "selected_resource_ids": list(selected_resource_ids),
        "owner_pid": owner_pid,
        "owner_process_start_identity": owner_process_start_identity,
        "process_identity_observation": process_observation,
        "gpu_process_query_complete": gpu.get("process_query_complete") is True,
        "matching_owner_gpu_processes": target_rows,
        "resource_monitor_report_sha256": canonical_sha256(dict(report)),
        "resource_observed_at_utc": report.get("generated_at_utc"),
        "assessed_at_utc": now_utc,
        "observation_age_seconds": age,
        "max_observation_age_seconds": float(max_observation_age_s),
        "reason_codes": reasons,
    }
    return {**core, "proof_sha256": canonical_sha256(core)}


def assess_legacy_dead_cuda_owner_reclaim_v1(
    *,
    attempt_id: str,
    resource_lease_digest: Mapping[str, Any],
    fencing_token_digest: Mapping[str, Any],
    device_uuid: str,
    selected_resource_ids: Sequence[str],
    start_effect_digest: Mapping[str, Any],
    resource_started_event_digest: Mapping[str, Any],
    report: Mapping[str, Any],
    now_utc: str,
    max_observation_age_s: float = 5.0,
) -> dict[str, Any]:
    """Assess one PID-less row for an explicit, host-drained migration only.

    Historical Launchpad start effects predate the exact Linux process identity
    binding.  They cannot use :func:`assess_dead_cuda_owner_reclaim_v1`.
    This deliberately stronger fallback is therefore limited to a completely
    idle exact CUDA device: telemetry must be complete and there may be no
    process row for the device (nor any row whose device is unknown).  This
    proof is intentionally insufficient for normal automatic admission;
    only a separately authorized migration boundary may consume it.
    """

    age = observation_age_seconds(report, now_utc=now_utc)
    gpu = dict(report.get("gpu") or {})
    processes = [
        dict(row)
        for row in gpu.get("processes") or []
        if isinstance(row, Mapping)
    ]
    target_rows = [
        row
        for row in processes
        if gpu_uuid_equal_v1(row.get("gpu_uuid"), device_uuid)
    ]
    unknown_device_rows = [row for row in processes if not row.get("gpu_uuid")]
    observed_devices = [
        row.get("uuid")
        for row in gpu.get("gpus") or []
        if isinstance(row, Mapping) and row.get("uuid")
    ]
    reasons: list[str] = []
    if age is None or age < 0 or age > max(1.0, float(max_observation_age_s)):
        reasons.append("STALE_RESOURCE_OBSERVATION")
    if gpu.get("available") is not True or gpu.get("process_query_complete") is not True:
        reasons.append("CUDA_PROCESS_TELEMETRY_INCOMPLETE")
    if not any(gpu_uuid_equal_v1(value, device_uuid) for value in observed_devices):
        reasons.append("DEVICE_UUID_UNOBSERVED")
    if target_rows:
        reasons.append("TARGET_CUDA_DEVICE_NOT_IDLE")
    if unknown_device_rows:
        reasons.append("CUDA_PROCESS_DEVICE_UNKNOWN")
    decision = "RECLAIM" if not reasons else "WAIT"
    core = {
        "schema_version": "eiel_legacy_dead_cuda_owner_reclaim_proof_v1",
        "decision": decision,
        "attempt_id": attempt_id,
        "resource_lease_digest": dict(resource_lease_digest),
        "fencing_token_digest": dict(fencing_token_digest),
        "device_uuid": device_uuid,
        "selected_resource_ids": list(selected_resource_ids),
        "start_effect_digest": dict(start_effect_digest),
        "resource_started_event_digest": dict(resource_started_event_digest),
        "gpu_process_query_complete": gpu.get("process_query_complete") is True,
        "target_device_gpu_processes": target_rows,
        "unknown_device_gpu_processes": unknown_device_rows,
        "resource_monitor_report_sha256": canonical_sha256(dict(report)),
        "resource_observed_at_utc": report.get("generated_at_utc"),
        "assessed_at_utc": now_utc,
        "observation_age_seconds": age,
        "max_observation_age_seconds": float(max_observation_age_s),
        "reason_codes": reasons,
    }
    return {**core, "proof_sha256": canonical_sha256(core)}


def assess_legacy_gpu_migration_reclaim_v1(
    *,
    migration_manifest_sha256: str,
    migration_row_sha256: str,
    classification: str,
    attempt_id: str,
    resource_lease_digest: Mapping[str, Any],
    fencing_token_digest: Mapping[str, Any],
    continuation_lease_digest: Mapping[str, Any],
    continuation_fence_code: str,
    continuation_terminal_event_digest: Mapping[str, Any] | None,
    continuation_successor_event_digest: Mapping[str, Any] | None,
    device_uuid: str,
    selected_resource_ids: Sequence[str],
    start_effect_digest: Mapping[str, Any],
    resource_started_event_digest: Mapping[str, Any],
    owner_pid: int | None,
    owner_process_start_ticks: int | None,
    process_start_ticks_observation: Mapping[str, Any] | None,
    report: Mapping[str, Any],
    now_utc: str,
    max_observation_age_s: float = 5.0,
) -> dict[str, Any]:
    """Assess one exact row in the digest-pinned PID-less GPU migration.

    ``SEALED_PROCESS_CONTEXT`` proves that the historical PID/start-ticks
    owner is gone and allows unrelated GPU occupants.  ``TRUE_PIDLESS_DEVICE_IDLE``
    has no process identity and therefore requires an already-expired
    continuation plus a completely idle exact device.  The returned record is
    advisory; only the Scheduler migration entrypoint may consume it.
    """

    age = observation_age_seconds(report, now_utc=now_utc)
    gpu = dict(report.get("gpu") or {})
    processes = [
        dict(row)
        for row in gpu.get("processes") or []
        if isinstance(row, Mapping)
    ]
    target_device_rows = [
        row
        for row in processes
        if gpu_uuid_equal_v1(row.get("gpu_uuid"), device_uuid)
    ]
    unknown_device_rows = [row for row in processes if not row.get("gpu_uuid")]
    matching_owner_rows = [
        row
        for row in target_device_rows
        if owner_pid is not None and row.get("pid") == owner_pid
    ]
    owner_pid_unknown_device_rows = [
        row
        for row in unknown_device_rows
        if owner_pid is not None and row.get("pid") == owner_pid
    ]
    observed_devices = [
        row.get("uuid")
        for row in gpu.get("gpus") or []
        if isinstance(row, Mapping) and row.get("uuid")
    ]
    assessed_at = _parse_time(now_utc)
    resource_observed = _parse_time(report.get("generated_at_utc"))
    process_observation = (
        None
        if process_start_ticks_observation is None
        else dict(process_start_ticks_observation)
    )
    process_observed = _parse_time(
        None if process_observation is None else process_observation.get("observed_at_utc")
    )
    process_observation_age = (
        None
        if process_observed is None or assessed_at is None
        else (assessed_at - process_observed).total_seconds()
    )
    reasons: list[str] = []
    if classification not in {
        "SEALED_PROCESS_CONTEXT",
        "TRUE_PIDLESS_DEVICE_IDLE",
    }:
        reasons.append("MIGRATION_CLASSIFICATION_INVALID")
    if continuation_fence_code not in {"ACTIVE", "EXPIRED"}:
        reasons.append("CONTINUATION_FENCE_INVALID")
    elif continuation_fence_code == "ACTIVE" and (
        continuation_terminal_event_digest is not None
        or continuation_successor_event_digest is not None
    ):
        reasons.append("CONTINUATION_FENCE_INVALID")
    elif continuation_fence_code == "EXPIRED" and not isinstance(
        continuation_terminal_event_digest, Mapping
    ):
        reasons.append("CONTINUATION_FENCE_INVALID")
    if age is None or age < 0 or age > max(1.0, float(max_observation_age_s)):
        reasons.append("STALE_RESOURCE_OBSERVATION")
    if gpu.get("available") is not True or gpu.get("process_query_complete") is not True:
        reasons.append("CUDA_PROCESS_TELEMETRY_INCOMPLETE")
    if not any(gpu_uuid_equal_v1(value, device_uuid) for value in observed_devices):
        reasons.append("DEVICE_UUID_UNOBSERVED")

    if classification == "SEALED_PROCESS_CONTEXT":
        if (
            type(owner_pid) is not int
            or owner_pid <= 0
            or type(owner_process_start_ticks) is not int
            or owner_process_start_ticks <= 0
            or process_observation is None
            or process_observation.get("schema_version")
            != "eiel_legacy_process_start_ticks_observation_v1"
            or process_observation.get("pid") != owner_pid
            or process_observation.get("expected_process_start_ticks")
            != owner_process_start_ticks
            or process_observation.get("state") not in {"ABSENT", "REUSED"}
        ):
            reasons.append("OWNER_PROCESS_NOT_PROVEN_DEAD")
        if (
            process_observation_age is None
            or process_observation_age < 0
            or process_observation_age > max(1.0, float(max_observation_age_s))
        ):
            reasons.append("STALE_PROCESS_IDENTITY_OBSERVATION")
        if (
            process_observed is None
            or resource_observed is None
            or assessed_at is None
            or process_observed > resource_observed
            or resource_observed > assessed_at
        ):
            reasons.append("CAUSAL_EVIDENCE_ORDER_INVALID")
        if matching_owner_rows:
            reasons.append("OWNER_PID_STILL_HAS_TARGET_CUDA_CONTEXT")
        if owner_pid_unknown_device_rows:
            reasons.append("OWNER_PID_CUDA_DEVICE_UNKNOWN")
    elif classification == "TRUE_PIDLESS_DEVICE_IDLE":
        if continuation_fence_code != "EXPIRED":
            reasons.append("PIDLESS_CONTINUATION_NOT_EXPIRED")
        if owner_pid is not None or owner_process_start_ticks is not None:
            reasons.append("PIDLESS_ROW_HAS_SYNTHETIC_PROCESS_IDENTITY")
        if process_observation is not None:
            reasons.append("PIDLESS_ROW_HAS_PROCESS_OBSERVATION")
        if target_device_rows:
            reasons.append("TARGET_CUDA_DEVICE_NOT_IDLE")
        if unknown_device_rows:
            reasons.append("CUDA_PROCESS_DEVICE_UNKNOWN")

    decision = "RECLAIM" if not reasons else "WAIT"
    core = {
        "schema_version": "eiel_legacy_gpu_migration_reclaim_proof_v1",
        "decision": decision,
        "migration_manifest_sha256": migration_manifest_sha256,
        "migration_row_sha256": migration_row_sha256,
        "classification": classification,
        "attempt_id": attempt_id,
        "resource_lease_digest": dict(resource_lease_digest),
        "fencing_token_digest": dict(fencing_token_digest),
        "continuation_lease_digest": dict(continuation_lease_digest),
        "continuation_fence_code": continuation_fence_code,
        "continuation_terminal_event_digest": (
            None
            if continuation_terminal_event_digest is None
            else dict(continuation_terminal_event_digest)
        ),
        "continuation_successor_event_digest": (
            None
            if continuation_successor_event_digest is None
            else dict(continuation_successor_event_digest)
        ),
        "device_uuid": device_uuid,
        "selected_resource_ids": list(selected_resource_ids),
        "start_effect_digest": dict(start_effect_digest),
        "resource_started_event_digest": dict(resource_started_event_digest),
        "owner_pid": owner_pid,
        "owner_process_start_ticks": owner_process_start_ticks,
        "process_start_ticks_observation": process_observation,
        "gpu_process_query_complete": gpu.get("process_query_complete") is True,
        "matching_owner_gpu_processes": matching_owner_rows,
        "owner_pid_unknown_device_processes": owner_pid_unknown_device_rows,
        "target_device_gpu_processes": target_device_rows,
        "unknown_device_gpu_processes": unknown_device_rows,
        "resource_monitor_report_sha256": canonical_sha256(dict(report)),
        "resource_observed_at_utc": report.get("generated_at_utc"),
        "assessed_at_utc": now_utc,
        "observation_age_seconds": age,
        "max_observation_age_seconds": float(max_observation_age_s),
        "reason_codes": reasons,
    }
    return {**core, "proof_sha256": canonical_sha256(core)}


def observe_legacy_gpu_migration_reclaim_v1(
    *,
    repo_root: str | Path,
    migration_manifest_sha256: str,
    migration_row_sha256: str,
    classification: str,
    attempt_id: str,
    resource_lease_digest: Mapping[str, Any],
    fencing_token_digest: Mapping[str, Any],
    continuation_lease_digest: Mapping[str, Any],
    continuation_fence_code: str,
    continuation_terminal_event_digest: Mapping[str, Any] | None,
    continuation_successor_event_digest: Mapping[str, Any] | None,
    device_uuid: str,
    selected_resource_ids: Sequence[str],
    start_effect_digest: Mapping[str, Any],
    resource_started_event_digest: Mapping[str, Any],
    owner_pid: int | None,
    owner_process_start_ticks: int | None,
) -> dict[str, Any]:
    """Observe and assess one migration row inside Resource Manager ownership.

    The Scheduler invokes a dedicated Resource Manager subprocess while it
    holds its execution lock.  Keeping both procfs and CUDA collection here
    prevents the subprocess wrapper (or any caller) from manufacturing the
    observation payload consumed by the advisory assessment.
    """

    resolved_repo_root = Path(repo_root).expanduser().resolve()
    if not resolved_repo_root.is_dir():
        raise ActorError(
            "legacy GPU migration repository root is unavailable",
            code="launchpad_legacy_gpu_migration_denied",
        )
    process_observation = None
    if owner_pid is not None or owner_process_start_ticks is not None:
        if type(owner_pid) is not int or type(owner_process_start_ticks) is not int:
            raise ActorError(
                "legacy GPU migration process identity is malformed",
                code="launchpad_legacy_gpu_migration_denied",
            )
        process_observation = erm.observe_legacy_process_start_ticks_v1(
            owner_pid,
            owner_process_start_ticks,
        )
    report = erm.collect_report(erm.MonitorConfig(repo_root=resolved_repo_root))
    return assess_legacy_gpu_migration_reclaim_v1(
        migration_manifest_sha256=migration_manifest_sha256,
        migration_row_sha256=migration_row_sha256,
        classification=classification,
        attempt_id=attempt_id,
        resource_lease_digest=resource_lease_digest,
        fencing_token_digest=fencing_token_digest,
        continuation_lease_digest=continuation_lease_digest,
        continuation_fence_code=continuation_fence_code,
        continuation_terminal_event_digest=continuation_terminal_event_digest,
        continuation_successor_event_digest=continuation_successor_event_digest,
        device_uuid=device_uuid,
        selected_resource_ids=selected_resource_ids,
        start_effect_digest=start_effect_digest,
        resource_started_event_digest=resource_started_event_digest,
        owner_pid=owner_pid,
        owner_process_start_ticks=owner_process_start_ticks,
        process_start_ticks_observation=process_observation,
        report=report,
        now_utc=utc_now(),
    )


def assess_resources(
    demand: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    now_utc: str | None = None,
    max_observation_age_s: float = 300.0,
) -> dict[str, Any]:
    demand = normalize_demand(demand)
    device = str(demand.get("device") or "either")
    blockers = list(report.get("blockers") or [])
    classes = {str(row.get("class") or "") for row in blockers if isinstance(row, Mapping)}
    kinds = {str(row.get("kind") or "") for row in blockers if isinstance(row, Mapping)}
    ram_available = _gib(dict(report.get("memory") or {}).get("ram_available_bytes"))
    vram_available = _max_vram_free_gib(report)
    storage_available = _storage_available_gib(report)
    reasons: list[str] = []
    reason_codes: list[str] = []
    limits: list[str] = []
    next_actions: list[dict[str, Any]] = []
    now = now_utc or utc_now()
    age = observation_age_seconds(report, now_utc=now)

    def add_reason(code: str, text: str) -> None:
        reason_codes.append(code)
        reasons.append(text)

    def below(observed: float | None, required: Any, label: str) -> bool:
        required_value = float(required or 0.0)
        if required_value <= 0:
            return False
        if observed is None:
            add_reason(f"{label.lower()}_unknown", f"{label} availability is unknown; required {required_value:.3f} GiB")
            return True
        if observed < required_value:
            add_reason(f"{label.lower()}_insufficient", f"{label} available {observed:.3f} GiB is below required {required_value:.3f} GiB")
            return True
        return False

    stale = age is None or age < 0 or age > max(1.0, float(max_observation_age_s))
    if stale:
        add_reason("stale_resource_observation", "resource observation is missing a valid timestamp or is older than the allowed age")
        next_actions.append({"description": "collect a fresh read-only ERM observation", "command": "python -m tools.eiel resourceManager inspect"})
    ram_short = below(ram_available, demand.get("minimum_ram_available_gib"), "RAM")
    storage_short = below(storage_available, demand.get("minimum_storage_available_gib"), "storage")
    vram_short = below(vram_available, demand.get("minimum_vram_available_gib"), "VRAM") if device in {"cuda", "either"} else False
    gpu_available = bool(dict(report.get("gpu") or {}).get("available"))
    gpu_blocked = "GPU_BLOCKER" in classes or "VRAM_CAUTION" in classes
    unknown = "UNKNOWN" in classes or "gpu_telemetry_unavailable" in kinds or "unattributed_eiel_process" in kinds
    cpu_caution = bool(classes.intersection({"CPU_CAUTION", "RAM_CAUTION", "STORAGE_CAUTION"}))
    cuda_eval = _evaluate_cuda_admission(
        demand,
        report,
        gpu_available=gpu_available,
        unknown=unknown,
        gpu_blocked=gpu_blocked,
        vram_short=vram_short,
    )

    if stale:
        decision, route = "MANUAL_REVIEW", "none"
    elif device == "none":
        decision, route = "ADMIT", "none"
        add_reason("no_runtime_compute_required", "mission declares no runtime compute demand")
    elif ram_short or storage_short:
        decision, route = ("WAIT" if ram_available is not None or storage_available is not None else "MANUAL_REVIEW"), "none"
        next_actions.append({"description": "wait for memory or storage pressure to clear and inspect again", "command": "python -m tools.eiel resourceManager inspect"})
    elif device == "cpu":
        route = "cpu"
        if cpu_caution:
            decision = "ADMIT_WITH_LIMITS"
            limits.append("run CPU-only and avoid overlapping compiles or large memory jobs")
            add_reason("cpu_capacity_caution", "CPU/RAM/storage caution evidence exists")
        else:
            decision = "ADMIT"
            add_reason("cpu_capacity_available", "no CPU, RAM, or storage blocker detected")
    elif device == "cuda":
        decision, route = cuda_eval["decision"], cuda_eval["route"]
        for code, reason in zip(cuda_eval["reason_codes"], cuda_eval["reasons"]):
            add_reason(str(code), str(reason))
        limits.extend(str(limit) for limit in cuda_eval["limits"])
        if decision == "MANUAL_REVIEW":
            next_actions.append({"description": "inspect GPU attribution before launch", "command": "python -m tools.eiel resourceManager campaigns"})
    else:
        if cuda_eval["decision"] in {"ADMIT", "ADMIT_WITH_LIMITS"}:
            route = "cuda"
            decision = "ADMIT_WITH_LIMITS" if cpu_caution or cuda_eval["decision"] == "ADMIT_WITH_LIMITS" else "ADMIT"
            for code, reason in zip(cuda_eval["reason_codes"], cuda_eval["reasons"]):
                add_reason(str(code), str(reason))
            limits.extend(str(limit) for limit in cuda_eval["limits"])
            add_reason("cuda_route_preferred", "CUDA route is currently available")
            if cpu_caution:
                limits.append("avoid additional CPU-heavy work while CUDA mission runs")
        elif not ram_short and not storage_short:
            route = "cpu"
            decision = "ADMIT_WITH_LIMITS"
            limits.append("use CPU-only route; CUDA is unavailable or blocked")
            add_reason("cpu_fallback", "CPU fallback is available while CUDA is not")
        else:
            route, decision = "none", "WAIT"
    if decision not in DECISIONS:
        decision = "MANUAL_REVIEW"
    poll_seconds = int(
        report.get("next_check_after_seconds")
        or report.get("recommended_poll_interval_seconds")
        or 30
    )
    poll_seconds = max(1, poll_seconds)
    contested_resource = "cuda" if device in {"cuda", "either"} and (
        cuda_eval["state"] not in {"IDLE_EXCLUSIVE", "IDLE_SHARED", "IDLE_SHARED_CAUTION"}
        or gpu_blocked
        or unknown
        or vram_short
    ) else (
        "memory_or_storage" if ram_short or storage_short else "none"
    )
    safe_non_contending_actions = [
        "continue bounded source inspection or implementation that does not contend for the blocked resource",
        "run lightweight CPU-only tests when RAM and CPU evidence permit",
        "prepare documentation, qualification plans, or evidence reports",
    ]
    if decision == "WAIT":
        next_actions = [
            {
                "action": "do_not_launch_contested_work",
                "description": f"Do not launch the contested {contested_resource} workload while ERM reports WAIT or an equivalent blocker.",
            },
            {
                "action": "continue_non_contending_work",
                "description": "Continue safe non-contending work such as source inspection, documentation, CPU-only tests when admitted, qualification planning, or evidence review; do not start a background scheduler.",
            },
            {
                "action": "rerun_fresh_inspection",
                "description": f"Rerun a fresh read-only inspection after approximately {poll_seconds} seconds if the resource is still needed.",
                "command": "python -m tools.eiel resourceManager inspect",
                "retry_after_seconds": poll_seconds,
            },
        ]
    elif decision == "MANUAL_REVIEW":
        next_actions = [
            *next_actions,
            {
                "action": "do_not_guess",
                "description": "Do not infer missing or unattributed resource state. Inspect the blocker or ask the operator.",
            },
        ]
    return {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "decision": decision,
        "recommended_route": route,
        "advisory_only": True,
        "binding_authority": "none",
        "demand": dict(demand),
        "observed": {
            "observation_generated_at_utc": report.get("generated_at_utc"),
            "observation_age_seconds": age,
            "max_observation_age_seconds": float(max_observation_age_s),
            "observation_stale": stale,
            "ram_available_gib": ram_available,
            "vram_available_gib": vram_available,
            "gpu_state": cuda_eval["state"],
            "gpu_admission_mode": demand.get("gpu_admission_mode"),
            "gpu_compute_process_count": len(cuda_eval["occupants"]),
            "gpu_occupants": cuda_eval["occupants"],
            "usable_vram_gib": cuda_eval["capacity_gib"],
            "observed_vram_used_gib": cuda_eval["observed_vram_used_gib"],
            "existing_declared_peak_vram_gib": cuda_eval["existing_declared_peak_vram_gib"],
            "candidate_peak_vram_gib": cuda_eval["candidate_peak_vram_gib"],
            "projected_peak_vram_gib": cuda_eval["projected_peak_vram_gib"],
            "storage_available_gib": storage_available,
            "erm_recommendation": dict(report.get("recommendation") or {}),
            "blocker_classes": sorted(classes),
        },
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "reasons": list(dict.fromkeys(reasons)),
        "limits": list(dict.fromkeys(limits)),
        "next_actions": next_actions,
        "operating_policy": {
            "retry_after_seconds": poll_seconds if decision == "WAIT" else None,
            "background_polling": False,
            "contested_resource": contested_resource,
            "gpu_state": cuda_eval["state"],
            "gpu_admission_mode": demand.get("gpu_admission_mode"),
            "shared_functional_run": demand.get("gpu_admission_mode") == "shared_functional",
            "contended_measurements_non_gating": (
                demand.get("gpu_admission_mode") == "shared_functional"
                and bool(cuda_eval["occupants"])
            ),
            "formal_qualification_requires_exclusive_rerun": (
                demand.get("gpu_admission_mode") == "shared_functional"
            ),
            "safe_non_contending_actions": safe_non_contending_actions,
            "continue_non_contending_work": decision == "WAIT",
            "wait_recommendation_is_advisory": decision == "WAIT",
            "erm_wait_is_binding_for_codex_launch": decision == "WAIT",
            "dispatch_request_owns_launch_authorization": True,
            "resource_manager_remains_advisory": True,
        },
        "non_goals": ["no_schedule", "no_reservation", "no_launch", "no_cancel", "no_process_kill"],
    }


def command_inspect(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report = _report(args)
    generated_at = utc_now(args.generated_at)
    record = _observation_record(report, generated_at=generated_at)
    path = None if args.no_store else _store(args).put(record)
    age = observation_age_seconds(report, now_utc=generated_at)
    warnings = [] if age is not None and 0 <= age <= args.max_observation_age_s else ["observation timestamp is stale or unavailable"]
    return with_guidance(
        {"ok": True, "observation_id": record["record_id"], "record_path": str(path) if path else None, "report": report, "observation_age_seconds": age},
        status="OBSERVED", warnings=warnings,
        next_actions=[{"description": "assess a mission against this observation", "command": "python -m tools.eiel resourceManager assess --mission MIS-..."}],
        related_records=[record["record_id"]],
    ), 0


def _assess(args: argparse.Namespace, *, recommendation_only: bool) -> tuple[dict[str, Any], int]:
    demand, refs = _load_demand(args)
    report = _report(args)
    generated_at = utc_now(args.generated_at)
    observation = _observation_record(report, generated_at=generated_at)
    observation_path = _store(args).put(observation)
    assessment = assess_resources(demand, report, now_utc=generated_at, max_observation_age_s=args.max_observation_age_s)
    observation_hash = canonical_sha256(report)
    assessment_id = stable_id("RADM", {"demand": demand, "observation_sha256": observation_hash})
    if demand.get("mission_id"):
        assessment["next_actions"] = [
            *list(assessment.get("next_actions") or []),
            {
                "action": "attach_resource_record",
                "description": "Attach this advisory resource record to the mission.",
                "command": (
                    f"python -m tools.eiel missionPlanner update {demand['mission_id']} "
                    f"--resource-ref {assessment_id}"
                ),
            },
        ]
    payload = {"assessment_id": assessment_id, "observation_id": observation["record_id"], "observation_sha256": observation_hash, **assessment}
    record = new_record(
        actor=ACTOR_NAME,
        record_type=ASSESSMENT_RECORD_TYPE,
        record_id=assessment_id,
        status=str(assessment["decision"]),
        payload=payload,
        generated_at=generated_at,
        source_refs=[*refs, actor_source_ref("resource_observation", observation)],
        action="recommend" if recommendation_only else "assess",
    )
    path = _store(args).put(record)
    positive = assessment["decision"] in {"ADMIT", "ADMIT_WITH_LIMITS"}
    return with_guidance(
        {
            "ok": positive,
            "assessment_id": assessment_id,
            "record_path": str(path),
            "observation_record_path": str(observation_path),
            "assessment": payload,
        },
        status=str(assessment["decision"]),
        reason_codes=assessment["reason_codes"],
        problems=assessment["reasons"] if not positive else [],
        warnings=assessment["limits"],
        next_actions=assessment["next_actions"],
        related_records=[assessment_id, observation["record_id"]],
    ), 0 if positive else 1


def command_assess(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    return _assess(args, recommendation_only=False)


def command_recommend(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    return _assess(args, recommendation_only=True)


def command_campaigns(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report = _report(args)
    workloads = report.get("running_eiel_workloads") or []
    heartbeats = report.get("slot_heartbeats") or []
    gpu_state = erm.gpu_occupancy_state(report)
    return with_guidance(
        {
            "ok": True,
            "running_eiel_workloads": workloads,
            "slot_heartbeats": heartbeats,
            "campaign_manifests": report.get("campaign_manifests") or [],
            "gpu": report.get("gpu") or {},
            "gpu_state": gpu_state,
            "recommendation": report.get("recommendation") or {},
        },
        status="WORKLOADS_PRESENT" if workloads or heartbeats or gpu_state == "CONTENDED" else "IDLE",
        warnings=["review stale or unattributed workloads"] if report.get("blockers") else [],
        next_actions=[{"description": "collect a fresh resource observation", "command": "python -m tools.eiel resourceManager inspect"}],
    ), 0


def command_history(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    store = _store(args)
    if getattr(args, "record_id", None):
        record = store.get(args.record_id)
        rows = store.history(args.record_id)
        return with_guidance(
            {"ok": True, "record_id": args.record_id, "revision_count": len(rows), "history": rows, "record": record},
            status=str(record.get("status") or "UNKNOWN"), related_records=[args.record_id],
        ), 0
    rows = store.list()
    if args.kind != "all":
        expected = OBSERVATION_RECORD_TYPE if args.kind == "observations" else ASSESSMENT_RECORD_TYPE
        rows = [row for row in rows if row.get("record_type") == expected]
    counts = Counter(str(row.get("status") or "UNKNOWN") for row in rows)
    return with_guidance(
        {"ok": not any(row.get("status") == "INVALID" for row in rows), "kind": args.kind, "count": len(rows), "status_counts": dict(sorted(counts.items())), "records": rows},
        status="HISTORY", warnings=["invalid stored resource record exists"] if any(row.get("status") == "INVALID" for row in rows) else [],
    ), 0


def command_explain(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.assessment_id)
    if record.get("record_type") != ASSESSMENT_RECORD_TYPE:
        raise ActorError("record is not a resource assessment", code="wrong_record_type", related_records=[args.assessment_id])
    assessment = dict(record.get("payload") or {})
    positive = assessment.get("decision") in {"ADMIT", "ADMIT_WITH_LIMITS"}
    return with_guidance(
        {"ok": positive, "assessment": assessment, "record": record},
        status=str(assessment.get("decision") or "UNKNOWN"),
        reason_codes=assessment.get("reason_codes") or [],
        problems=[] if positive else assessment.get("reasons") or [],
        warnings=assessment.get("limits") or [],
        next_actions=assessment.get("next_actions") or [],
        related_records=[args.assessment_id, str(assessment.get("observation_id") or "")],
    ), 0


def command_show(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.record_id)
    if record.get("record_type") == ASSESSMENT_RECORD_TYPE:
        args.assessment_id = args.record_id
        return command_explain(args)
    report = dict(dict(record.get("payload") or {}).get("report") or {})
    return with_guidance(
        {"ok": True, "record": record, "report": report}, status=str(record.get("status") or "OBSERVED"),
        next_actions=[{"description": "assess a mission against a fresh observation", "command": "python -m tools.eiel resourceManager assess --mission MIS-..."}],
        related_records=[args.record_id],
    ), 0


def command_next(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    args.assessment_id = args.assessment_id
    return command_explain(args)


def command_handoff(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    record = _store(args).get(args.assessment_id)
    if record.get("record_type") != ASSESSMENT_RECORD_TYPE:
        raise ActorError("record is not a resource assessment", code="wrong_record_type", related_records=[args.assessment_id])
    assessment = dict(record.get("payload") or {})
    positive = assessment.get("decision") in {"ADMIT", "ADMIT_WITH_LIMITS"}
    handoff = {
        "schema_version": "eiel_codex_resource_handoff_v1",
        **operator_metadata(),
        "assessment_id": args.assessment_id,
        "observation_id": assessment.get("observation_id"),
        "decision": assessment.get("decision"),
        "recommended_route": assessment.get("recommended_route"),
        "reason_codes": assessment.get("reason_codes") or [],
        "reasons": assessment.get("reasons") or [],
        "limits": assessment.get("limits") or [],
        "next_actions": assessment.get("next_actions") or [],
        "operating_policy": assessment.get("operating_policy") or {},
        "demand": assessment.get("demand") or {},
        "observed": assessment.get("observed") or {},
        "record_sha256": record.get("record_sha256"),
        "advisory_only": True,
        "binding_authority": "none",
    }
    return with_guidance(
        {"ok": positive, "handoff": handoff}, status=str(assessment.get("decision") or "UNKNOWN"),
        reason_codes=handoff["reason_codes"], problems=[] if positive else handoff["reasons"],
        warnings=handoff["limits"], next_actions=handoff["next_actions"],
        related_records=[args.assessment_id, str(assessment.get("observation_id") or "")],
    ), 0


def _human(payload: Mapping[str, Any]) -> str:
    if isinstance(payload.get("assessment"), Mapping):
        row = payload["assessment"]
        observed = dict(row.get("observed") or {})
        return (
            f"Resource recommendation: {row.get('decision')}\n"
            f"Route: {row.get('recommended_route')}\n"
            f"GPU state: {observed.get('gpu_state', 'UNKNOWN')}\n"
            f"Peak VRAM projected/usable: {observed.get('projected_peak_vram_gib', 'UNKNOWN')} / {observed.get('usable_vram_gib', 'UNKNOWN')} GiB\n"
            + "\n".join(f"- {reason}" for reason in row.get("reasons") or [])
        )
    if isinstance(payload.get("report"), Mapping):
        return erm.render_human(payload["report"])
    if "running_eiel_workloads" in payload:
        return f"GPU state: {payload.get('gpu_state', 'UNKNOWN')}\nRunning workloads: {len(payload.get('running_eiel_workloads') or [])}\nHeartbeats: {len(payload.get('slot_heartbeats') or [])}\nRecommendation: {dict(payload.get('recommendation') or {}).get('value', 'UNKNOWN')}"
    return f"Resource records: {payload.get('count', 0)}"


def _assessment_options(parser: argparse.ArgumentParser) -> None:
    _common_options(parser)
    _monitor_options(parser)
    parser.add_argument("--mission")
    parser.add_argument("--demand")
    parser.add_argument("--device", choices=("cpu", "cuda", "either", "none"))
    parser.add_argument("--workload-class")
    parser.add_argument("--minimum-ram-available-gib", type=float, default=0.0)
    parser.add_argument("--minimum-vram-available-gib", type=float, default=0.0)
    parser.add_argument("--minimum-storage-available-gib", type=float, default=0.0)
    parser.add_argument("--exclusive-gpu", action="store_true")
    parser.add_argument("--gpu-admission-mode", choices=GPU_ADMISSION_MODES)
    parser.add_argument(
        "--peak-vram-gib",
        type=float,
        help="Declared peak CUDA allocation, including checkpoint/resume transients.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(prog="python -m tools.eiel resourceManager", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ActorArgumentParser)

    inspect = sub.add_parser("inspect", help="Run a fresh read-only ERM inspection and optionally retain it.")
    _common_options(inspect)
    _monitor_options(inspect)
    inspect.add_argument("--no-store", action="store_true")

    assess = sub.add_parser("assess", help="Assess a mission or resource demand against fresh or saved ERM evidence.")
    _assessment_options(assess)

    recommend = sub.add_parser("recommend", help="Recommend CPU, CUDA, wait, or manual review.")
    _assessment_options(recommend)

    campaigns = sub.add_parser("campaigns", help="Show currently observed workloads and heartbeats.")
    _common_options(campaigns)
    _monitor_options(campaigns)

    history = sub.add_parser("history", help="List retained observations and advisory assessments.")
    _common_options(history)
    history.add_argument("record_id", nargs="?")
    history.add_argument("--kind", choices=("all", "observations", "assessments"), default="all")

    explain = sub.add_parser("explain", help="Explain a retained resource recommendation and its next action.")
    _common_options(explain)
    explain.add_argument("assessment_id")

    show = sub.add_parser("show", help="Show one retained resource observation or assessment record.")
    _common_options(show)
    show.add_argument("record_id")

    next_cmd = sub.add_parser("next", help="Show the next safe action for a retained resource assessment.")
    _common_options(next_cmd)
    next_cmd.add_argument("assessment_id")

    handoff = sub.add_parser("handoff", help="Produce a compact Codex resource handoff.")
    _common_options(handoff)
    handoff.add_argument("assessment_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    forbidden = {
        "reserve": "Resource Manager is advisory only and cannot reserve resources.",
        "launch": "Use Codex and existing campaign tooling after reviewing the advisory decision.",
        "kill": "Resource Manager never kills or alters processes.",
        "schedule": "No scheduler or background service exists in this actor.",
        "cancel": "Resource Manager never cancels workloads.",
    }
    if raw_args and raw_args[0] in forbidden:
        exc = forbidden_operation_error(f"eiel {ACTOR_NAME}", raw_args[0], forbidden[raw_args[0]])
        emit_actor_error(exc, actor=ACTOR_NAME, command=raw_args[0], raw_args=raw_args)
        return 2
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(raw_args)
        handlers = {
            "inspect": command_inspect,
            "assess": command_assess,
            "recommend": command_recommend,
            "campaigns": command_campaigns,
            "history": command_history,
            "explain": command_explain,
            "show": command_show,
            "next": command_next,
            "handoff": command_handoff,
        }
        payload, code = handlers[args.command](args)
        emit_payload(payload, output=args.output, human_text=_human(payload), json_output=not args.human)
        return code
    except ActorError as exc:
        command = raw_args[0] if raw_args else "unknown"
        emit_actor_error(
            exc,
            actor=ACTOR_NAME,
            command=command,
            raw_args=raw_args,
            parsed_args=args,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
