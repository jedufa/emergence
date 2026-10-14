from __future__ import annotations

"""Selective Simulation Trace F0 contracts and offline fixtures.

TRACE1-F0 deliberately does not hook the live simulator. It defines deterministic
requests, planning, storage estimates, and fixture artifacts that future runtime
instrumentation must satisfy.
"""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from simulation.chronicle.schema import canonical_payload, stable_hash, event_order_key
from simulation.chronicle.storage import artifact_inventory, read_json, read_jsonl, sha256_file, write_json, write_jsonl


TRACE_REQUEST_SCHEMA = "eiel_selective_trace_request_v1"
TRACE_STEP_RECORD_SCHEMA = "eiel_selective_trace_step_record_v1"
TRACE_WINDOW_SCHEMA = "eiel_selective_trace_window_v1"
TRACE_MANIFEST_SCHEMA = "eiel_selective_trace_manifest_v1"
TRACE_STORAGE_SUMMARY_SCHEMA = "eiel_selective_trace_storage_summary_v1"
TRACE_SOURCE_COMPLETENESS_SCHEMA = "eiel_selective_trace_source_completeness_v1"
TRACE_REQUEST_STATUS_SCHEMA = "eiel_selective_trace_request_status_v1"
TRACE_COMPANION_INDEX_SCHEMA = "eiel_selective_trace_companion_index_v1"

TRACE_PROFILES = ("trace_off", "trace_triggered", "trace_subjects", "trace_sampled", "trace_diagnostic_full")
TRACE_FIELD_GROUPS = (
    "identity",
    "action",
    "state_core",
    "local_ecology",
    "observation_summary",
    "policy_summary",
    "memory_summary",
    "genome_reference",
    "world_delta",
)
DEFAULT_FIELD_GROUPS = ("identity", "action", "state_core", "local_ecology", "world_delta")
DIAGNOSTIC_FIELD_GROUPS = TRACE_FIELD_GROUPS
HIGH_VALUE_PRIORITIES = {"critical", "high"}
HIGH_VALUE_SEVERITIES = {"critical", "major"}
TRIGGER_EVENT_TYPES = {
    "notable_death",
    "lineage_near_extinction",
    "lineage_extinction",
    "population_bottleneck",
    "predator_wave",
    "delayed_hazard_cycle",
    "food_depletion",
    "rare_mutation",
}
TRACE_ARTIFACTS = {
    "requests": "trace_requests.jsonl",
    "windows": "trace_windows.jsonl",
    "steps": "trace_step_records.jsonl",
    "manifest": "trace_manifest.json",
    "storage": "trace_storage_summary.json",
    "source_completeness": "trace_source_completeness.json",
    "status": "trace_request_status.json",
    "index": "trace_companion_index.json",
}


class TraceError(ValueError):
    """Raised when TRACE1-F0 contracts are violated."""


def _strings(values: Iterable[Any] | None) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in (values or ()) if str(value)}))


def _ints(values: Iterable[Any] | None) -> tuple[int, ...]:
    out: set[int] = set()
    for value in values or ():
        try:
            out.add(int(value))
        except Exception:
            continue
    return tuple(sorted(out))


def _ranges(values: Iterable[Sequence[int]] | None) -> tuple[tuple[int, int], ...]:
    rows: set[tuple[int, int]] = set()
    for value in values or ():
        if len(value) != 2:
            raise TraceError("ranges must contain exactly two integers")
        start = int(value[0])
        end = int(value[1])
        if start > end:
            raise TraceError("range start must be <= end")
        rows.add((start, end))
    return tuple(sorted(rows))


def _field_groups(values: Iterable[str] | None, *, profile: str) -> tuple[str, ...]:
    requested = _strings(values)
    if not requested:
        requested = DIAGNOSTIC_FIELD_GROUPS if profile == "trace_diagnostic_full" else DEFAULT_FIELD_GROUPS
    unknown = sorted(set(requested).difference(TRACE_FIELD_GROUPS))
    if unknown:
        raise TraceError(f"unknown trace field groups: {unknown}")
    if profile != "trace_diagnostic_full":
        forbidden = {"observation_summary", "policy_summary", "memory_summary"}.intersection(requested)
        # Compact summaries are allowed, full tensors/hidden states are not. The
        # F0 schema exposes summaries only, so these groups remain valid.
        _ = forbidden
    return requested


@dataclass(frozen=True)
class TraceBudgets:
    max_windows: int = 32
    max_windows_per_generation: int = 8
    max_subjects: int = 16
    max_records_per_generation: int = 512
    max_records_per_run: int = 10_000
    max_output_bytes_per_generation: int = 2_000_000
    max_output_bytes_per_run: int = 32_000_000
    max_ring_buffer_bytes: int = 8_000_000
    diagnostic_full_max_generations: int = 2
    diagnostic_full_max_steps: int = 300

    def to_dict(self) -> dict[str, Any]:
        return canonical_payload(asdict(self))


@dataclass(frozen=True)
class TraceRequest:
    schema_version: str = TRACE_REQUEST_SCHEMA
    request_id: str = ""
    run_id: str = ""
    campaign_id: str | None = None
    branch_id: str | None = None
    profile: str = "trace_off"
    generation_ranges: tuple[tuple[int, int], ...] = ()
    step_ranges: tuple[tuple[int, int], ...] = ()
    selected_agents: tuple[int, ...] = ()
    selected_lineages: tuple[str, ...] = ()
    selected_regions: tuple[str, ...] = ()
    selected_patches: tuple[str, ...] = ()
    selected_episodes: tuple[str, ...] = ()
    selected_event_ids: tuple[str, ...] = ()
    trigger_rules: tuple[str, ...] = ()
    steps_before: int = 0
    steps_after: int = 0
    sampling_stride: int = 1
    captured_field_groups: tuple[str, ...] = DEFAULT_FIELD_GROUPS
    budgets: TraceBudgets = field(default_factory=TraceBudgets)
    output_profile: str = "compact"
    allow_large_output: bool = False
    source_commit: str = ""
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return canonical_payload(asdict(self))


def make_trace_request(
    *,
    run_id: str,
    profile: str = "trace_off",
    campaign_id: str | None = None,
    branch_id: str | None = None,
    generation_ranges: Iterable[Sequence[int]] = (),
    step_ranges: Iterable[Sequence[int]] = (),
    selected_agents: Iterable[Any] = (),
    selected_lineages: Iterable[Any] = (),
    selected_regions: Iterable[Any] = (),
    selected_patches: Iterable[Any] = (),
    selected_episodes: Iterable[Any] = (),
    selected_event_ids: Iterable[Any] = (),
    trigger_rules: Iterable[Any] = (),
    steps_before: int = 4,
    steps_after: int = 4,
    sampling_stride: int = 1,
    captured_field_groups: Iterable[str] = (),
    budgets: TraceBudgets | None = None,
    output_profile: str = "compact",
    allow_large_output: bool = False,
    source_commit: str = "",
    limitations: Iterable[Any] = (),
) -> TraceRequest:
    if profile not in TRACE_PROFILES:
        raise TraceError(f"unsupported trace profile: {profile}")
    if not run_id:
        raise TraceError("run_id is required")
    budget = budgets or TraceBudgets()
    request = TraceRequest(
        request_id="",
        run_id=str(run_id),
        campaign_id=str(campaign_id) if campaign_id else None,
        branch_id=str(branch_id) if branch_id else None,
        profile=profile,
        generation_ranges=_ranges(generation_ranges),
        step_ranges=_ranges(step_ranges),
        selected_agents=_ints(selected_agents),
        selected_lineages=_strings(selected_lineages),
        selected_regions=_strings(selected_regions),
        selected_patches=_strings(selected_patches),
        selected_episodes=_strings(selected_episodes),
        selected_event_ids=_strings(selected_event_ids),
        trigger_rules=_strings(trigger_rules),
        steps_before=max(0, int(steps_before)),
        steps_after=max(0, int(steps_after)),
        sampling_stride=max(1, int(sampling_stride)),
        captured_field_groups=_field_groups(captured_field_groups, profile=profile),
        budgets=budget,
        output_profile=str(output_profile or "compact"),
        allow_large_output=bool(allow_large_output),
        source_commit=str(source_commit or ""),
        limitations=_strings(limitations),
    )
    _validate_request(request)
    payload = request.to_dict()
    payload.pop("request_id", None)
    request_id = "trq_" + stable_hash("eiel-trace-request-v1", payload)
    return replace(request, request_id=request_id)


def _validate_request(request: TraceRequest) -> None:
    if request.profile == "trace_off":
        return
    if request.profile == "trace_diagnostic_full":
        if not request.allow_large_output:
            raise TraceError("trace_diagnostic_full requires allow_large_output=true")
        total_generations = sum(end - start + 1 for start, end in request.generation_ranges) if request.generation_ranges else 1
        total_steps = sum(end - start + 1 for start, end in request.step_ranges) if request.step_ranges else request.steps_before + request.steps_after + 1
        if total_generations > request.budgets.diagnostic_full_max_generations:
            raise TraceError("trace_diagnostic_full exceeds generation cap")
        if total_generations * total_steps > request.budgets.diagnostic_full_max_steps:
            raise TraceError("trace_diagnostic_full exceeds step cap")
    selected_subjects = (
        len(request.selected_agents)
        + len(request.selected_lineages)
        + len(request.selected_regions)
        + len(request.selected_patches)
        + len(request.selected_episodes)
    )
    if selected_subjects > request.budgets.max_subjects:
        raise TraceError("selected subjects exceed trace budget")


def _in_ranges(value: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return not ranges or any(start <= value <= end for start, end in ranges)


def _event_triggers(request: TraceRequest, event: Mapping[str, Any]) -> list[str]:
    if request.profile == "trace_off":
        return []
    event_id = str(event.get("event_id", ""))
    event_type = str(event.get("event_type", ""))
    priority = str(event.get("live_priority", ""))
    severity = str(event.get("severity", ""))
    generation = int(event.get("generation", 0) or 0)
    if not _in_ranges(generation, request.generation_ranges):
        return []
    triggers: list[str] = []
    if event_id in request.selected_event_ids:
        triggers.append("operator_selected_event")
    if request.profile == "trace_triggered":
        rules = set(request.trigger_rules) or {"high_value_chronicle_event", "notable_event_type"}
        if "high_value_chronicle_event" in rules and (
            priority in HIGH_VALUE_PRIORITIES
            or severity in HIGH_VALUE_SEVERITIES
            or int(event.get("scientific_importance", 0) or 0) >= 4
            or int(event.get("historical_importance", 0) or 0) >= 4
        ):
            triggers.append("high_value_chronicle_event")
        if "notable_event_type" in rules and event_type in TRIGGER_EVENT_TYPES:
            triggers.append(event_type)
    if request.profile == "trace_subjects":
        if request.selected_agents and set(request.selected_agents).intersection(int(value) for value in event.get("agent_ids", []) or []):
            triggers.append("selected_agent")
        if request.selected_lineages and set(request.selected_lineages).intersection(str(value) for value in event.get("lineage_ids", []) or []):
            triggers.append("selected_lineage")
        if request.selected_regions and str(event.get("region_id") or "") in request.selected_regions:
            triggers.append("selected_region")
        patch = event.get("patch_coordinates")
        if request.selected_patches and patch is not None and f"{patch[0]},{patch[1]}" in request.selected_patches:
            triggers.append("selected_patch")
    if request.profile == "trace_sampled":
        if generation % request.sampling_stride == 0:
            triggers.append("sampled_generation")
    if request.profile == "trace_diagnostic_full":
        triggers.append("diagnostic_full")
    return sorted(set(triggers))


def _subject_key(event: Mapping[str, Any]) -> str:
    if event.get("subject_id"):
        return f"{event.get('subject_type') or 'subject'}:{event.get('subject_id')}"
    if event.get("lineage_ids"):
        return "lineage:" + str(sorted(str(value) for value in event.get("lineage_ids", []) or [])[0])
    if event.get("region_id"):
        return "region:" + str(event.get("region_id"))
    return "global"


def _window_from_event(request: TraceRequest, event: Mapping[str, Any], triggers: list[str]) -> dict[str, Any]:
    generation = int(event.get("generation", 0) or 0)
    step = int(event.get("step", 0) or 0)
    start_step = max(0, step - request.steps_before)
    end_step = max(start_step, step + request.steps_after)
    subject_key = _subject_key(event)
    payload = {
        "request_id": request.request_id,
        "trigger_type": triggers[0],
        "trigger_source_ids": [str(event.get("event_id", ""))],
        "subject_key": subject_key,
        "generation": generation,
        "start_step": start_step,
        "end_step": end_step,
        "requested_steps_before": request.steps_before,
        "requested_steps_after": request.steps_after,
        "captured_field_groups": list(request.captured_field_groups),
    }
    return {
        "schema_version": TRACE_WINDOW_SCHEMA,
        "window_id": "trw_" + stable_hash("eiel-trace-window-v1", payload),
        **payload,
        "actual_steps_before": step - start_step,
        "actual_steps_after": end_step - step,
        "step_record_ids": [],
        "overlap_trigger_source_ids": [],
        "truncation_reason": None,
        "source_completeness": "fixture_complete",
        "limitations": ["TRACE1-F0 fixture window; not captured from live tick loop."],
        "byte_count": 0,
    }


def plan_trace_windows(
    request: TraceRequest,
    events: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    if request.profile == "trace_off":
        return canonical_payload(
            {
                "schema_version": "eiel_selective_trace_plan_v1",
                "request": request.to_dict(),
                "windows": [],
                "dropped_windows": [],
                "estimated_records": 0,
                "estimated_bytes": 0,
                "limitations": ["trace_off produces no trace output."],
            }
        )
    raw: list[dict[str, Any]] = []
    for event in sorted((dict(row) for row in events), key=event_order_key):
        triggers = _event_triggers(request, event)
        if triggers:
            raw.append(_window_from_event(request, event, triggers))
    merged = _merge_windows(raw)
    accepted: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    per_generation: dict[int, int] = {}
    for window in merged:
        generation = int(window["generation"])
        per_generation[generation] = per_generation.get(generation, 0) + 1
        if len(accepted) >= request.budgets.max_windows:
            dropped.append({**window, "drop_reason": "max_windows"})
        elif per_generation[generation] > request.budgets.max_windows_per_generation:
            dropped.append({**window, "drop_reason": "max_windows_per_generation"})
        else:
            accepted.append(window)
    estimated_records = sum(int(row["end_step"]) - int(row["start_step"]) + 1 for row in accepted)
    estimated_bytes = estimate_trace_storage(
        field_groups=request.captured_field_groups,
        step_records=estimated_records,
        windows=len(accepted),
    )["estimated_bytes"]
    return canonical_payload(
        {
            "schema_version": "eiel_selective_trace_plan_v1",
            "request": request.to_dict(),
            "windows": accepted,
            "dropped_windows": dropped,
            "estimated_records": estimated_records,
            "estimated_bytes": estimated_bytes,
            "limitations": ["TRACE1-F0 planner uses Chronicle events and fixture estimates only."],
        }
    )


def _merge_windows(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(windows, key=lambda row: (int(row["generation"]), str(row["subject_key"]), int(row["start_step"]), int(row["end_step"]), str(row["window_id"])))
    merged: list[dict[str, Any]] = []
    for window in rows:
        if not merged:
            merged.append(dict(window))
            continue
        previous = merged[-1]
        compatible = (
            int(previous["generation"]) == int(window["generation"])
            and str(previous["subject_key"]) == str(window["subject_key"])
            and tuple(previous["captured_field_groups"]) == tuple(window["captured_field_groups"])
            and int(window["start_step"]) <= int(previous["end_step"]) + 1
        )
        if not compatible:
            merged.append(dict(window))
            continue
        trigger_ids = sorted(set(previous["trigger_source_ids"]) | set(window["trigger_source_ids"]))
        previous["end_step"] = max(int(previous["end_step"]), int(window["end_step"]))
        previous["trigger_source_ids"] = trigger_ids
        previous["overlap_trigger_source_ids"] = sorted(set(previous.get("overlap_trigger_source_ids", [])) | set(window["trigger_source_ids"]))
        previous["window_id"] = "trw_" + stable_hash(
            "eiel-trace-window-merged-v1",
            {
                "request_id": previous["request_id"],
                "generation": previous["generation"],
                "subject_key": previous["subject_key"],
                "start_step": previous["start_step"],
                "end_step": previous["end_step"],
                "trigger_source_ids": trigger_ids,
            },
        )
    return merged


FIELD_GROUP_ESTIMATED_BYTES = {
    "identity": 96,
    "action": 96,
    "state_core": 128,
    "local_ecology": 192,
    "observation_summary": 160,
    "policy_summary": 192,
    "memory_summary": 96,
    "genome_reference": 96,
    "world_delta": 128,
}


def estimate_trace_storage(
    *,
    field_groups: Iterable[str],
    step_records: int,
    windows: int,
    agents_per_step: int = 1,
) -> dict[str, Any]:
    groups = _field_groups(field_groups, profile="trace_triggered")
    bytes_per_agent_step = sum(FIELD_GROUP_ESTIMATED_BYTES[group] for group in groups)
    estimated_step_bytes = int(step_records) * max(1, int(agents_per_step)) * bytes_per_agent_step
    estimated_window_bytes = int(windows) * 512
    estimated_bytes = estimated_step_bytes + estimated_window_bytes + 4096
    return canonical_payload(
        {
            "schema_version": TRACE_STORAGE_SUMMARY_SCHEMA,
            "field_groups": list(groups),
            "bytes_per_agent_step": bytes_per_agent_step,
            "step_records": int(step_records),
            "agents_per_step": int(agents_per_step),
            "windows": int(windows),
            "estimated_step_bytes": estimated_step_bytes,
            "estimated_window_bytes": estimated_window_bytes,
            "estimated_bytes": estimated_bytes,
            "estimated_mebibytes": round(estimated_bytes / (1024 * 1024), 6),
        }
    )


def storage_projection_table(*, steps_per_generation: int = 150) -> list[dict[str, Any]]:
    shapes = [
        ("full_trace_pop32_g24", 32, 24),
        ("full_trace_pop64_g36", 64, 36),
        ("full_trace_pop128_g36", 128, 36),
        ("full_trace_pop64_g1000", 64, 1000),
        ("full_trace_pop64_g10000", 64, 10000),
        ("triggered_8_windows_g36", 1, 36),
        ("subjects_4_agents_g36", 4, 36),
        ("sampled_every_10th_g1000", 64, 100),
        ("diagnostic_full_one_generation_pop64", 64, 1),
    ]
    rows: list[dict[str, Any]] = []
    for label, agents, generations in shapes:
        records = int(steps_per_generation) * int(generations)
        if label.startswith("triggered"):
            records = 8 * 9
        elif label.startswith("subjects"):
            records = int(steps_per_generation) * int(generations)
        estimate = estimate_trace_storage(
            field_groups=DEFAULT_FIELD_GROUPS if not label.startswith("full_trace") else TRACE_FIELD_GROUPS,
            step_records=records,
            windows=8 if label.startswith("triggered") else max(1, generations),
            agents_per_step=agents,
        )
        rows.append(
            {
                "shape": label,
                "steps_per_generation": int(steps_per_generation),
                "population_or_subjects": int(agents),
                "generations": int(generations),
                "estimated_bytes": estimate["estimated_bytes"],
                "estimated_mebibytes": estimate["estimated_mebibytes"],
                "preflight_risk": "rejected_default" if label.startswith("full_trace") and generations >= 1000 else ("large" if estimate["estimated_bytes"] > 32_000_000 else "bounded"),
            }
        )
    return canonical_payload(rows)


@dataclass
class RingBuffer:
    capacity: int
    rows: list[dict[str, Any]] = field(default_factory=list)

    def append(self, row: Mapping[str, Any]) -> None:
        self.rows.append(dict(row))
        if len(self.rows) > int(self.capacity):
            self.rows = self.rows[-int(self.capacity) :]

    def window(self, *, generation: int, start_step: int, end_step: int, subject_key: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.rows
            if int(row.get("generation", -1)) == int(generation)
            and int(start_step) <= int(row.get("step", -1)) <= int(end_step)
            and str(row.get("subject_key", "")) == str(subject_key)
        ]


def _fixture_step_record(request: TraceRequest, window: Mapping[str, Any], step: int) -> dict[str, Any]:
    subject_key = str(window["subject_key"])
    generation = int(window["generation"])
    identity = {
        "run_id": request.run_id,
        "generation": generation,
        "step": int(step),
        "phase": "fixture_step_boundary",
        "subject_key": subject_key,
        "source_owner": "TRACE1-F0 synthetic fixture",
    }
    record: dict[str, Any] = {
        "schema_version": TRACE_STEP_RECORD_SCHEMA,
        "record_id": "trs_" + stable_hash("eiel-trace-step-record-v1", {**identity, "request_id": request.request_id}),
        "request_id": request.request_id,
        "window_id": str(window["window_id"]),
        "subject_key": subject_key,
        "generation": generation,
        "step": int(step),
        "field_groups": list(request.captured_field_groups),
        "source_completeness": "fixture_complete",
        "limitations": ["Synthetic TRACE1-F0 record; not captured from live runtime."],
    }
    if "identity" in request.captured_field_groups:
        record["identity"] = identity
    if "action" in request.captured_field_groups:
        actions = ("north", "east", "gather", "rest")
        record["action"] = {
            "selected_action": actions[step % len(actions)],
            "applied_action": actions[step % len(actions)],
            "action_success": True,
            "target": None,
            "action_cost": round(0.1 + (step % 3) * 0.05, 3),
        }
    if "state_core" in request.captured_field_groups:
        record["state_core"] = {
            "alive": True,
            "energy": round(20.0 + generation - step * 0.1, 3),
            "injury": round((step % 5) * 0.02, 3),
            "age": generation * 150 + step,
            "position": [step % 12, (step + generation) % 12],
        }
    if "local_ecology" in request.captured_field_groups:
        record["local_ecology"] = {
            "food": round(1.0 - (step % 4) * 0.1, 3),
            "cover": 0.25,
            "shelter": 0.1,
            "movement_cost": 1.0,
            "scent": round((step % 6) * 0.01, 3),
            "waste": round((step % 7) * 0.01, 3),
            "density": step % 3,
            "predator_risk": {"value": None, "source_completeness": "unavailable_in_fixture"},
            "delayed_hazard": {"active": False, "source_completeness": "fixture_complete"},
        }
    if "observation_summary" in request.captured_field_groups:
        record["observation_summary"] = {"schema": "fixture_compact_summary_v1", "visible_food_cells": step % 5, "source_completeness": "fixture_summary_only"}
    if "policy_summary" in request.captured_field_groups:
        record["policy_summary"] = {"top_action": "gather", "second_action": "rest", "margin": 0.1, "policy_hash": "fixture_policy_hash"}
    if "memory_summary" in request.captured_field_groups:
        record["memory_summary"] = {"memory_state_hash": "fixture_memory_hash", "memory_norm": 0.0, "remembered_content_claim": "not_assessed"}
    if "genome_reference" in request.captured_field_groups:
        record["genome_reference"] = {"genome_hash": "fixture_genome_hash", "mutation_companion_ids": []}
    if "world_delta" in request.captured_field_groups:
        record["world_delta"] = {"energy_delta": -0.1, "injury_delta": 0.0, "position_delta": [0, 1], "local_food_delta": -0.1}
    return canonical_payload(record)


def produce_fixture_trace(request: TraceRequest, plan: Mapping[str, Any]) -> dict[str, Any]:
    if request.profile == "trace_off":
        return {"windows": [], "step_records": [], "status": _status(request, "trace_off")}
    ring = RingBuffer(capacity=max(1, request.steps_before + request.steps_after + 1))
    step_records: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    dropped = list(plan.get("dropped_windows", []) or [])
    for source_window in list(plan.get("windows", []) or []):
        window = dict(source_window)
        window_records: list[dict[str, Any]] = []
        for step in range(int(window["start_step"]), int(window["end_step"]) + 1):
            record = _fixture_step_record(request, window, step)
            ring.append(record)
            window_records.append(record)
        if len(step_records) + len(window_records) > request.budgets.max_records_per_run:
            window["truncation_reason"] = "max_records_per_run"
            allowed = max(0, request.budgets.max_records_per_run - len(step_records))
            window_records = window_records[:allowed]
            dropped.append({**window, "drop_reason": "truncated_records"})
        step_records.extend(window_records)
        window["step_record_ids"] = [str(row["record_id"]) for row in window_records]
        window["byte_count"] = len("\n".join(stable_hash("byte-proxy", row, length=64) for row in window_records).encode("utf-8"))
        windows.append(canonical_payload(window))
    return canonical_payload(
        {
            "windows": windows,
            "step_records": step_records,
            "status": _status(request, "ready" if not dropped else "ready_with_truncation", dropped=dropped),
        }
    )


def _status(request: TraceRequest, status: str, *, dropped: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    return canonical_payload(
        {
            "schema_version": TRACE_REQUEST_STATUS_SCHEMA,
            "request_id": request.request_id,
            "profile": request.profile,
            "status": status,
            "dropped_window_count": len(dropped),
            "dropped_windows": list(dropped),
            "source_writes_attempted": False,
            "rng_consumed": False,
            "runtime_hooks_used": False,
        }
    )


def query_trace(
    records: Iterable[Mapping[str, Any]],
    *,
    window_id: str | None = None,
    event_id: str | None = None,
    agent: int | None = None,
    lineage: str | None = None,
    region: str | None = None,
    patch: str | None = None,
    generation: int | None = None,
    step: int | None = None,
    action: str | None = None,
    field_group: str | None = None,
    source_completeness: str | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        if window_id and str(row.get("window_id")) != str(window_id):
            continue
        if generation is not None and int(row.get("generation", -1)) != int(generation):
            continue
        if step is not None and int(row.get("step", -1)) != int(step):
            continue
        if field_group and field_group not in (row.get("field_groups", []) or []):
            continue
        if source_completeness and str(row.get("source_completeness")) != source_completeness:
            continue
        if action and str(dict(row.get("action", {}) or {}).get("applied_action")) != action:
            continue
        text = " ".join(str(value) for value in (row.get("subject_key"), row.get("identity"), row.get("genome_reference")))
        if event_id and event_id not in text:
            continue
        if agent is not None and f"agent:{int(agent)}" not in text and str(int(agent)) not in text:
            continue
        if lineage and str(lineage) not in text:
            continue
        if region and str(region) not in text:
            continue
        if patch and str(patch) not in text:
            continue
        selected.append(row)
    return sorted(selected, key=lambda row: (int(row.get("generation", 0)), int(row.get("step", 0)), str(row.get("record_id", ""))))


def write_trace_outputs(
    output_dir: Path | str,
    request: TraceRequest,
    plan: Mapping[str, Any],
    fixture: Mapping[str, Any],
    *,
    source_root: Path | str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve()
    if output.is_symlink():
        raise TraceError("trace output must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    windows = list(fixture.get("windows", []) or [])
    records = list(fixture.get("step_records", []) or [])
    write_jsonl(output / TRACE_ARTIFACTS["requests"], [request.to_dict()])
    write_jsonl(output / TRACE_ARTIFACTS["windows"], windows)
    write_jsonl(output / TRACE_ARTIFACTS["steps"], records)
    storage = estimate_trace_storage(
        field_groups=request.captured_field_groups,
        step_records=len(records),
        windows=len(windows),
    )
    write_json(output / TRACE_ARTIFACTS["storage"], storage)
    completeness = {
        "schema_version": TRACE_SOURCE_COMPLETENESS_SCHEMA,
        "request_id": request.request_id,
        "runtime_capture": False,
        "fixture_capture": True,
        "unavailable_fields_are_not_zero": True,
        "limitations": ["TRACE1-F0 fixture does not prove live runtime capture."],
    }
    write_json(output / TRACE_ARTIFACTS["source_completeness"], completeness)
    write_json(output / TRACE_ARTIFACTS["status"], fixture.get("status", _status(request, "unknown")))
    index = {
        "schema_version": TRACE_COMPANION_INDEX_SCHEMA,
        "label": "Selective trace window, not a complete simulation replay.",
        "request_id": request.request_id,
        "run_id": request.run_id,
        "query_filters": [
            "trace_window_id",
            "trigger",
            "event_id",
            "episode_id",
            "agent",
            "lineage",
            "region",
            "patch",
            "generation",
            "step",
            "action",
            "field_group",
            "source_completeness",
        ],
        "artifacts": dict(TRACE_ARTIFACTS),
        "source_root": str(source_root) if source_root is not None else None,
    }
    write_json(output / TRACE_ARTIFACTS["index"], index)
    manifest = {
        "schema_version": TRACE_MANIFEST_SCHEMA,
        "request_id": request.request_id,
        "run_id": request.run_id,
        "profile": request.profile,
        "source_root": str(source_root) if source_root is not None else None,
        "source_writes_attempted": False,
        "rng_consumed": False,
        "runtime_hooks_used": False,
        "window_count": len(windows),
        "step_record_count": len(records),
        "artifact_inventory": artifact_inventory(output, logical_names=()),
    }
    write_json(output / TRACE_ARTIFACTS["manifest"], manifest)
    artifacts: list[dict[str, Any]] = []
    for name, filename in sorted(TRACE_ARTIFACTS.items()):
        path = output / filename
        artifacts.append({"artifact": name, "relative_path": filename, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest["artifact_inventory"] = artifacts
    write_json(output / TRACE_ARTIFACTS["manifest"], manifest)
    return canonical_payload({"output_dir": str(output), "request_id": request.request_id, "files": sorted(TRACE_ARTIFACTS.values())})


def trace_from_chronicle_fixture(
    run_dir: Path | str,
    output_dir: Path | str,
    *,
    profile: str = "trace_triggered",
    steps_before: int = 4,
    steps_after: int = 4,
    max_windows: int = 8,
) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    chronicle = root / "chronicle" if (root / "chronicle" / "formal_events.jsonl").is_file() else root
    events = read_jsonl(chronicle / "formal_events.jsonl")
    manifest_path = chronicle / "chronicle_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else {}
    request = make_trace_request(
        run_id=str(manifest.get("run_id") or root.name),
        campaign_id=manifest.get("campaign_id"),
        branch_id=manifest.get("branch_id"),
        profile=profile,
        steps_before=steps_before,
        steps_after=steps_after,
        trigger_rules=("high_value_chronicle_event", "notable_event_type"),
        budgets=TraceBudgets(max_windows=max_windows),
        source_commit=str(manifest.get("source_commit", "")),
    )
    plan = plan_trace_windows(request, events)
    fixture = produce_fixture_trace(request, plan)
    return write_trace_outputs(output_dir, request, plan, fixture, source_root=root)


__all__ = [
    "TRACE_ARTIFACTS",
    "TRACE_COMPANION_INDEX_SCHEMA",
    "TRACE_FIELD_GROUPS",
    "TRACE_MANIFEST_SCHEMA",
    "TRACE_PROFILES",
    "TRACE_REQUEST_SCHEMA",
    "TRACE_REQUEST_STATUS_SCHEMA",
    "TRACE_SOURCE_COMPLETENESS_SCHEMA",
    "TRACE_STEP_RECORD_SCHEMA",
    "TRACE_STORAGE_SUMMARY_SCHEMA",
    "TRACE_WINDOW_SCHEMA",
    "RingBuffer",
    "TraceBudgets",
    "TraceError",
    "TraceRequest",
    "estimate_trace_storage",
    "make_trace_request",
    "plan_trace_windows",
    "produce_fixture_trace",
    "query_trace",
    "storage_projection_table",
    "trace_from_chronicle_fixture",
    "write_trace_outputs",
]
