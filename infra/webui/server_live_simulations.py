"""Read-only Emergence live-simulation projection for the main WebUI."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import os
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any


DEFAULT_TELEMETRY_ROOT = Path(public_runtime_path('live/simulation-slots'))
LIVE_CHRONICLE_SCHEMA = "eiel_native_live_chronicle_event_v1"
_LIVE_MAX_RUNS = 256
_LIVE_MAX_EVENTS = 256
_LIVE_TTL_SECONDS = 6 * 60 * 60
_live_lock = threading.Lock()
_live_runs: OrderedDict[str, dict[str, Any]] = OrderedDict()


def ingest_live_chronicle_event(value: Any) -> dict[str, Any]:
    """Accept one bounded native boundary event into process memory only."""

    if not isinstance(value, dict) or value.get("schema_version") != LIVE_CHRONICLE_SCHEMA:
        raise ValueError("unsupported live Chronicle event schema")
    required_strings = (
        "run_id", "controller_id", "event_id", "event_type", "headline",
        "scientific_summary", "source_completeness", "observed_at",
    )
    for field in required_strings:
        if not isinstance(value.get(field), str) or not str(value[field]).strip():
            raise ValueError(f"live Chronicle {field} must be a non-empty string")
        if len(str(value[field])) > 2048:
            raise ValueError(f"live Chronicle {field} exceeds its bound")
    for field in ("owner_epoch", "generation", "step", "boundary_ordinal"):
        if type(value.get(field)) is not int or int(value[field]) < 0:
            raise ValueError(f"live Chronicle {field} must be a non-negative integer")
    if value["event_type"] != "native_generation_boundary":
        raise ValueError("unsupported live Chronicle event type")

    event = {key: value[key] for key in (
        "schema_version", "run_id", "controller_id", "owner_epoch", "event_id",
        "event_type", "generation", "step", "boundary_ordinal",
        "headline", "scientific_summary", "source_completeness", "observed_at",
    )}
    event["trajectory_commit_frontier"] = value.get("trajectory_commit_frontier")
    measurements = value.get("measurements")
    if isinstance(measurements, dict) and type(measurements.get("alive_count")) is int:
        event["measurements"] = {"alive_count": int(measurements["alive_count"])}
    now = time.monotonic()
    with _live_lock:
        expired = [
            run_id for run_id, row in _live_runs.items()
            if now - float(row["received_monotonic"]) > _LIVE_TTL_SECONDS
        ]
        for run_id in expired:
            _live_runs.pop(run_id, None)
        run_id = str(event["run_id"])
        row = _live_runs.get(run_id)
        if (
            row is None
            or row["controller_id"] != event["controller_id"]
            or row["owner_epoch"] != event["owner_epoch"]
        ):
            row = {
                "controller_id": event["controller_id"],
                "owner_epoch": event["owner_epoch"],
                "events": deque(maxlen=_LIVE_MAX_EVENTS),
                "event_ids": set(),
                "last_boundary_ordinal": -1,
                "received_monotonic": now,
            }
            _live_runs[run_id] = row
        if (
            event["event_id"] not in row["event_ids"]
            and event["boundary_ordinal"] > row["last_boundary_ordinal"]
        ):
            if len(row["events"]) == _LIVE_MAX_EVENTS:
                removed = row["events"][0]
                row["event_ids"].discard(removed["event_id"])
            row["events"].append(event)
            row["event_ids"].add(event["event_id"])
            row["last_boundary_ordinal"] = event["boundary_ordinal"]
        row["received_monotonic"] = now
        _live_runs.move_to_end(run_id)
        while len(_live_runs) > _LIVE_MAX_RUNS:
            _live_runs.popitem(last=False)
        count = len(row["events"])
    return {
        "schema_version": "eiel_live_chronicle_ingest_receipt_v1",
        "accepted": True,
        "run_id": event["run_id"],
        "event_id": event["event_id"],
        "retained_event_count": count,
        "storage": "process_memory_only",
    }


def _live_chronicle_for_slot(slot: dict[str, Any]) -> dict[str, Any] | None:
    identifiers = [
        str(value).strip()
        for value in (slot.get("cell_id"), slot.get("run_id"))
        if str(value or "").strip()
    ]
    now = time.monotonic()
    with _live_lock:
        match = next(
            (
                _live_runs[identifier]
                for identifier in identifiers
                if identifier in _live_runs
                and now - float(_live_runs[identifier]["received_monotonic"])
                <= _LIVE_TTL_SECONDS
            ),
            None,
        )
        if match is None:
            return None
        events = [dict(event) for event in match["events"]]
    items = [
        {
            "kind": "event",
            "formal_event": {
                "event_id": event["event_id"],
                "event_type": event["event_type"],
                "generation": event["generation"],
                "step": event["step"],
                "event_category": "native_boundary",
                "severity": "informational",
                "live_priority": "medium",
            },
            "entry": {
                "headline": event["headline"],
                "scientific_summary": (
                    f"{event['scientific_summary']} Alive: {event['measurements']['alive_count']}."
                    if "measurements" in event
                    else event["scientific_summary"]
                ),
            },
        }
        for event in events[-12:]
    ]
    return {
        "chronicle_status": "live",
        "message": "Live native boundary Chronicle. Provisional until terminal materialization.",
        "overview": {
            "run_id": slot.get("run_id") or slot.get("cell_id"),
            "profile": "native_boundary_live",
            "source_completeness": "provisional_native_boundary",
            "finalized": False,
            "event_count": len(events),
            "episode_count": 0,
            "generation_range": {
                "first": {"generation": events[0]["generation"], "step": 0},
                "last": {"generation": events[-1]["generation"], "step": 0},
            },
        },
        "timeline": {"total": len(events), "items": items},
        "transport": "ram_only_local_http",
        "authoritative": False,
    }


def telemetry_root() -> Path:
    raw = str(os.environ.get("EIEL_SLOT_TELEMETRY_ROOT") or "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_TELEMETRY_ROOT


def snapshot(
    *, root: Path | None = None, include_system: bool = True
) -> dict[str, Any]:
    from tools.slot_monitor.server import build_snapshot

    selected = (root or telemetry_root()).resolve()
    selected.mkdir(parents=True, exist_ok=True)
    return build_snapshot(
        selected,
        include_system=include_system,
        observatory_route_prefix="/api/emergence/live_simulations/observatory",
    )


def observatory_response(
    *,
    path: str,
    root: Path | None = None,
) -> tuple[int, dict[str, Any] | bytes, str]:
    from tools.slot_monitor.server import (
        observatory_route_response,
    )

    selected = (root or telemetry_root()).resolve()
    selected.mkdir(parents=True, exist_ok=True)
    status, payload, content_type = observatory_route_response(
        selected,
        path=path,
        route_prefix="/api/emergence/live_simulations/observatory",
    )
    return int(status), payload, content_type


def canopy_asset(path: str) -> tuple[bytes, str]:
    from tools.slot_monitor.server import (
        living_lineage_canopy_asset,
    )

    return living_lineage_canopy_asset(path)


def chronicle_snapshot(
    *,
    cfg: Any,
    device: str,
    position: int,
    root: Path | None = None,
    include_system: bool = True,
) -> dict[str, Any]:
    """Return a bounded Chronicle preview for one trusted live slot."""

    selected_device = str(device or "").strip().lower()
    if selected_device not in {"cpu", "gpu"}:
        raise ValueError("device must be cpu or gpu")
    selected_position = int(position)
    payload = snapshot(root=root, include_system=include_system)
    slots = payload[f"{selected_device}_slots"]
    if selected_position < 1 or selected_position > len(slots):
        raise ValueError("slot position is outside device capacity")
    slot = slots[selected_position - 1]
    response: dict[str, Any] = {
        "schema_version": "eiel_live_simulation_chronicle_preview_v1",
        "device": selected_device,
        "position": selected_position,
        "cell_id": slot.get("cell_id"),
        "run_id": slot.get("run_id"),
        "simulation_status": slot.get("status"),
        "chronicle_status": "unavailable",
        "message": "No world occupies this slot.",
    }
    if slot.get("status") == "idle":
        return response

    live = _live_chronicle_for_slot(slot)
    active = str(slot.get("status") or "") in {
        "starting", "running", "finishing", "stale", "conflict", "untracked"
    }
    if live is not None and active:
        response.update(live)
        return response

    from infra.webui import server_chronicle

    ready = [row for row in server_chronicle.discover_runs(cfg) if row.get("status") == "ready"]
    run_root_text = str(slot.get("run_root") or "").strip()
    exact_root_matches: list[dict[str, Any]] = []
    if run_root_text:
        selected_root = Path(run_root_text).expanduser().resolve()
        for row in ready:
            try:
                discovered_root, _chronicle_root, _manifest, _summary = server_chronicle._resolve_run(
                    cfg, str(row["run_key"])
                )
            except (OSError, ValueError, KeyError):
                continue
            if discovered_root.resolve() == selected_root:
                exact_root_matches.append(row)

    identifiers = {
        str(value).strip()
        for value in (slot.get("cell_id"), slot.get("run_id"))
        if str(value or "").strip()
    }
    identity_matches = [
        row
        for row in ready
        if str(row.get("run_id") or "") in identifiers
        and (not active or row.get("storage_kind") != "verified_zip")
    ]
    matches = exact_root_matches or identity_matches
    if not matches:
        if live is not None:
            response.update(live)
            return response
        response["chronicle_status"] = (
            "finalizing"
            if str(slot.get("status") or "") == "finishing"
            else ("pending" if active else "unavailable")
        )
        response["message"] = (
            "Chronicle is finalizing and will link to permanent history after verified archival."
            if response["chronicle_status"] == "finalizing"
            else (
                "Chronicle is buffered during GPU execution and will appear after terminal materialization."
                if active
                else "This world completed without a discoverable Chronicle in the approved roots."
            )
        )
        return response
    if len(matches) > 1:
        response["chronicle_status"] = "ambiguous"
        response["message"] = "Multiple Chronicle runs match this world; open the full explorer to choose one."
        return response

    selected = matches[0]
    run_key = str(selected["run_key"])
    overview = server_chronicle.run_overview(cfg, run_key)
    first_page = server_chronicle.timeline(cfg, run_key, {"limit": ["1"]})
    total = int(first_page.get("total", 0) or 0)
    offset = max(0, total - 12)
    preview = server_chronicle.timeline(
        cfg,
        run_key,
        {"limit": ["12"], "offset": [str(offset)]},
    )
    response.update(
        {
            "chronicle_status": (
                "archived" if selected.get("storage_kind") == "verified_zip" else "ready"
            ),
            "message": (
                "Chronicle is finalized in verified permanent history."
                if selected.get("storage_kind") == "verified_zip"
                else "Chronicle ready."
            ),
            "run_key": run_key,
            "archive_key": selected.get("archive_key"),
            "overview": {
                "run_id": overview.get("identity", {}).get("run_id"),
                "profile": overview.get("profile"),
                "source_completeness": overview.get("source_completeness"),
                "finalized": overview.get("finalized"),
                "event_count": overview.get("event_count"),
                "episode_count": overview.get("episode_count"),
                "generation_range": overview.get("generation_range"),
            },
            "timeline": preview,
        }
    )
    return response
