from __future__ import annotations

"""Read-only LIVE1 sidecar over generation-boundary Scientific Chronicle files."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from simulation.chronicle.observer import (
    AUDIENCE_PROFILES,
    ObserverError,
    _ScientificSource,
    render_observer_episode,
    render_observer_event,
)
from simulation.chronicle.schema import (
    CHRONICLE_ENTRY_SCHEMA_VERSION,
    EPISODE_SCHEMA_VERSION,
    EVENT_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    canonical_payload,
    stable_hash,
)
from simulation.chronicle.storage import sha256_file, write_json, write_jsonl


LIVE1_VERSION = "eiel_chronicle_live1_v1"
LIVE1_CURSOR_SCHEMA = "eiel_chronicle_live_cursor_v1"
LIVE1_ITEM_SCHEMA = "eiel_chronicle_live_observer_item_v1"
LIVE1_EPISODE_SCHEMA = "eiel_chronicle_live_observer_episode_v1"
LIVE1_STATUS_SCHEMA = "eiel_chronicle_live_status_v1"
LIVE1_HEALTH_SCHEMA = "eiel_chronicle_live_health_v1"
LIVE1_INDEX_SCHEMA = "eiel_chronicle_live_index_v1"
LIVE1_MANIFEST_SCHEMA = "eiel_chronicle_live_manifest_v1"

LIVE1_ARTIFACTS = {
    "feed": "live_observer_feed.jsonl",
    "episodes": "live_observer_episodes.jsonl",
    "cursor": "live_observer_cursor.json",
    "status": "live_observer_status.json",
    "health": "live_observer_health.json",
    "index": "live_observer_index.json",
    "manifest": "live_observer_manifest.json",
}

_HASHED_ARTIFACTS = ("feed", "episodes", "cursor", "index")
_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_LINE_BYTES = 4 * 1024 * 1024
_PRIORITY_DURATION_MS = {
    "critical": None,
    "high": 12_000,
    "medium": 8_000,
    "low": 5_000,
    "background": 3_000,
}


class LiveObserverError(ValueError):
    """Raised when LIVE1 cannot safely consume or publish a source."""


class LiveObserverRetry(LiveObserverError):
    """Raised for a transient source boundary that should be polled again."""


@dataclass(frozen=True)
class LiveObserverConfig:
    audience_profiles: tuple[str, ...] = AUDIENCE_PROFILES
    poll_interval_seconds: float = 1.0
    maximum_source_records: int = 100_000
    maximum_simultaneous_bubbles: int = 8
    maximum_transport_page: int = 200

    def validate(self) -> None:
        profiles = tuple(dict.fromkeys(str(value) for value in self.audience_profiles))
        if not profiles or profiles != self.audience_profiles:
            raise LiveObserverError("audience profiles must be non-empty and unique")
        unsupported = sorted(set(profiles).difference(AUDIENCE_PROFILES))
        if unsupported:
            raise LiveObserverError(f"unsupported LIVE1 audience profiles: {unsupported}")
        if self.poll_interval_seconds < 0.1 or self.poll_interval_seconds > 300.0:
            raise LiveObserverError("poll interval must be between 0.1 and 300 seconds")
        if self.maximum_source_records < 1 or self.maximum_source_records > 1_000_000:
            raise LiveObserverError("maximum_source_records must be between 1 and 1000000")
        if self.maximum_simultaneous_bubbles < 1 or self.maximum_simultaneous_bubbles > 100:
            raise LiveObserverError("maximum_simultaneous_bubbles must be between 1 and 100")
        if self.maximum_transport_page < 1 or self.maximum_transport_page > 1_000:
            raise LiveObserverError("maximum_transport_page must be between 1 and 1000")

    def to_dict(self) -> dict[str, Any]:
        return canonical_payload(asdict(self))


@dataclass(frozen=True)
class _SourceSnapshot:
    run_root: Path
    chronicle_root: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    events: list[dict[str, Any]]
    entries: list[dict[str, Any]]
    episodes: list[dict[str, Any]]
    manifest_sha256: str
    event_prefix_digest: str
    partial_event_tail: bool
    partial_entry_tail: bool
    available_count: int


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any] | list[Any]:
    if path.is_symlink() or not path.is_file():
        raise LiveObserverError(f"missing or unsafe Chronicle artifact: {path.name}")
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise LiveObserverError(f"Chronicle artifact exceeds bounded reader: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveObserverRetry(f"Chronicle artifact is not yet readable: {path.name}: {exc}") from exc
    if not isinstance(value, (dict, list)):
        raise LiveObserverError(f"unexpected Chronicle artifact shape: {path.name}")
    return value


def _read_complete_jsonl(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if path.is_symlink() or not path.is_file():
        raise LiveObserverError(f"missing or unsafe Chronicle feed: {path.name}")
    rows: list[dict[str, Any]] = []
    partial_tail = False
    try:
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if len(raw) > _MAX_LINE_BYTES:
                    raise LiveObserverError(f"Chronicle feed line exceeds bound: {path.name}:{line_number}")
                if not raw.endswith(b"\n"):
                    partial_tail = bool(raw.strip())
                    break
                if not raw.strip():
                    continue
                value = json.loads(raw.decode("utf-8"))
                if not isinstance(value, dict):
                    raise LiveObserverError(f"expected object in {path.name}:{line_number}")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveObserverError(f"invalid complete Chronicle feed line in {path.name}: {exc}") from exc
    return rows, partial_tail


def _source_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return canonical_payload(
        {
            "run_id": str(manifest.get("run_id") or ""),
            "campaign_id": manifest.get("campaign_id"),
            "branch_id": manifest.get("branch_id"),
            "parent_checkpoint_id": manifest.get("parent_checkpoint_id"),
            "source_commit": manifest.get("source_commit"),
            "config_hash": manifest.get("config_hash"),
            "manifest_schema": manifest.get("schema_version"),
        }
    )


def _event_prefix_digest(events: Sequence[Mapping[str, Any]]) -> str:
    return "sha256:" + stable_hash("eiel-live1-formal-event-prefix-v1", list(events), length=64)


def _resolve_source(source_root: Path, approved_roots: Sequence[Path]) -> tuple[Path, Path]:
    raw = source_root.expanduser()
    if raw.is_symlink():
        raise LiveObserverError(f"source root must not be a symlink: {source_root}")
    selected = raw.resolve()
    if not selected.is_dir():
        raise LiveObserverError(f"source root is not a directory: {source_root}")
    approved = tuple(Path(value).expanduser().resolve() for value in approved_roots)
    if not approved or not any(_inside(selected, root) for root in approved):
        raise LiveObserverError("source root is outside the configured approved roots")
    nested = selected / "chronicle"
    chronicle = nested if (nested / "chronicle_manifest.json").is_file() else selected
    if chronicle.is_symlink() or not _inside(chronicle, selected):
        raise LiveObserverError("Chronicle source path is unsafe")
    run_root = chronicle.parent if chronicle.name == "chronicle" else chronicle
    return run_root, chronicle


def _resolve_output(output_root: Path, run_root: Path, chronicle_root: Path) -> Path:
    raw = output_root.expanduser()
    if raw.is_symlink():
        raise LiveObserverError(f"LIVE1 output must not be a symlink: {output_root}")
    output = raw.resolve()
    if output.exists() and not output.is_dir():
        raise LiveObserverError("LIVE1 output path exists and is not a directory")
    if _inside(output, run_root) or _inside(output, chronicle_root) or _inside(run_root, output):
        raise LiveObserverError("LIVE1 output must be separate from the source run tree")
    return output


def _load_snapshot(source_root: Path, approved_roots: Sequence[Path], maximum_records: int) -> _SourceSnapshot:
    run_root, chronicle = _resolve_source(source_root, approved_roots)
    manifest_path = chronicle / "chronicle_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise LiveObserverError("missing or unsafe Chronicle artifact: chronicle_manifest.json")
    try:
        manifest_before_raw = manifest_path.read_bytes()
    except OSError as exc:
        raise LiveObserverRetry(f"Scientific Chronicle manifest is not yet readable: {exc}") from exc
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or str(manifest.get("schema_version")) != MANIFEST_SCHEMA_VERSION:
        raise LiveObserverError("unsupported Scientific Chronicle manifest schema")
    summary = _read_json(chronicle / "scientific_chronicle_summary.json")
    episodes = _read_json(chronicle / "episodes.json")
    if not isinstance(summary, dict) or not isinstance(episodes, list):
        raise LiveObserverError("unexpected Scientific Chronicle summary or episode shape")
    events, partial_event = _read_complete_jsonl(chronicle / "formal_events.jsonl")
    entries, partial_entry = _read_complete_jsonl(chronicle / "scientific_chronicle.jsonl")
    manifest_after_raw = manifest_path.read_bytes()
    if manifest_before_raw != manifest_after_raw:
        raise LiveObserverRetry("Scientific Chronicle manifest changed while the sidecar was reading")
    declared = int(manifest.get("event_count", 0) or 0)
    available = min(declared, len(events), len(entries))
    if available > maximum_records:
        raise LiveObserverError("Scientific Chronicle exceeds LIVE1 maximum_source_records")
    events = events[:available]
    entries = entries[:available]
    seen_events: set[str] = set()
    seen_entries: set[str] = set()
    for index, (event, entry) in enumerate(zip(events, entries)):
        event_id = str(event.get("event_id") or "")
        entry_id = str(entry.get("chronicle_entry_id") or "")
        if str(event.get("schema_version")) != EVENT_SCHEMA_VERSION or not event_id:
            raise LiveObserverError(f"unsupported or missing formal event identity at index {index}")
        if str(entry.get("schema_version")) != CHRONICLE_ENTRY_SCHEMA_VERSION or not entry_id:
            raise LiveObserverError(f"unsupported or missing Chronicle entry identity at index {index}")
        if event_id in seen_events or entry_id in seen_entries:
            raise LiveObserverError("duplicate Scientific Chronicle identity")
        if event_id not in {str(value) for value in entry.get("source_event_ids", []) or []}:
            raise LiveObserverError("paired Chronicle entry does not reference its formal event")
        seen_events.add(event_id)
        seen_entries.add(entry_id)
    episode_ids: set[str] = set()
    available_event_ids = {str(event["event_id"]) for event in events}
    safe_episodes: list[dict[str, Any]] = []
    for episode in episodes:
        if not isinstance(episode, dict) or str(episode.get("schema_version")) != EPISODE_SCHEMA_VERSION:
            raise LiveObserverError("unsupported Scientific Chronicle episode schema")
        episode_id = str(episode.get("episode_id") or "")
        if not episode_id or episode_id in episode_ids:
            raise LiveObserverError("missing or duplicate Scientific Chronicle episode identity")
        source_ids = {str(value) for value in episode.get("source_event_ids", []) or []}
        if source_ids.issubset(available_event_ids):
            safe_episodes.append(episode)
        episode_ids.add(episode_id)
    manifest_sha = "sha256:" + stable_hash("eiel-live1-manifest-snapshot-v1", manifest, length=64)
    return _SourceSnapshot(
        run_root=run_root,
        chronicle_root=chronicle,
        manifest=manifest,
        summary=summary,
        events=events,
        entries=entries,
        episodes=safe_episodes,
        manifest_sha256=manifest_sha,
        event_prefix_digest=_event_prefix_digest(events),
        partial_event_tail=partial_event or declared > len(events),
        partial_entry_tail=partial_entry or declared > len(entries),
        available_count=available,
    )


def _load_json_file(path: Path, expected: type) -> Any:
    if path.is_symlink() or not path.is_file():
        raise LiveObserverError(f"missing or unsafe LIVE1 artifact: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveObserverError(f"invalid LIVE1 artifact {path.name}: {exc}") from exc
    if not isinstance(value, expected):
        raise LiveObserverError(f"unexpected LIVE1 artifact shape: {path.name}")
    return value


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows, partial = _read_complete_jsonl(path)
    if partial:
        raise LiveObserverError(f"LIVE1 artifact has a partial final line: {path.name}")
    return rows


def _load_existing(output: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not output.exists():
        return None, []
    names = {path.name for path in output.iterdir()}
    if names.issubset({LIVE1_ARTIFACTS["health"]}):
        return None, []
    cursor = _load_json_file(output / LIVE1_ARTIFACTS["cursor"], dict)
    if str(cursor.get("schema_version")) != LIVE1_CURSOR_SCHEMA:
        raise LiveObserverError("unsupported LIVE1 cursor schema")
    feed = _load_jsonl_file(output / LIVE1_ARTIFACTS["feed"])
    emitted = [str(value) for value in cursor.get("emitted_observer_ids", []) or []]
    observed = [str(row.get("observer_item_id") or "") for row in feed]
    if len(observed) != len(set(observed)) or observed != emitted:
        raise LiveObserverError("LIVE1 cursor and feed identities differ")
    return cursor, feed


def _map_anchor(event: Mapping[str, Any]) -> dict[str, Any] | None:
    coordinates = event.get("patch_coordinates")
    if isinstance(coordinates, (list, tuple)) and len(coordinates) == 2:
        try:
            return canonical_payload(
                {
                    "kind": "patch",
                    "coordinates": [int(coordinates[0]), int(coordinates[1])],
                    "authoritative": True,
                    "source_field": "formal_event.patch_coordinates",
                }
            )
        except (TypeError, ValueError):
            return None
    return None


def _checkpoint_segment(event: Mapping[str, Any], run_id: str) -> str:
    return "current_run" if str(event.get("run_id")) == run_id else "inherited_checkpoint_prefix"


def _live_event_row(
    event: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    profile: str,
    episode_family: str | None,
    source_completeness: str,
    run_id: str,
) -> dict[str, Any]:
    observer = dict(
        render_observer_event(
            event,
            entry,
            audience_profile=profile,
            episode_family=episode_family,
            source_completeness=source_completeness,
        )
    )
    observer_schema = observer.pop("schema_version")
    anchor = _map_anchor(event)
    priority = str(dict(observer.get("presentation_dimensions", {}) or {}).get("live_priority", "background"))
    return canonical_payload(
        {
            **observer,
            "schema_version": LIVE1_ITEM_SCHEMA,
            "observer_schema_version": observer_schema,
            "live1_version": LIVE1_VERSION,
            "map_anchor": anchor,
            "global_fallback": anchor is None,
            "bubble_policy": {
                "duration_ms": _PRIORITY_DURATION_MS.get(priority, 3_000),
                "persistent_until_ui_acknowledgement": priority == "critical",
            },
            "source_provenance": {
                **dict(observer.get("source_provenance", {}) or {}),
                "checkpoint_segment": _checkpoint_segment(event, run_id),
            },
        }
    )


def _live_episode_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(row)
    observer_schema = value.pop("schema_version")
    return canonical_payload(
        {
            **value,
            "schema_version": LIVE1_EPISODE_SCHEMA,
            "observer_schema_version": observer_schema,
            "live1_version": LIVE1_VERSION,
        }
    )


def _render_snapshot(snapshot: _SourceSnapshot, profiles: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes_by_id = {str(episode["episode_id"]): episode for episode in snapshot.episodes}
    source_completeness = str(
        snapshot.manifest.get("source_completeness", snapshot.summary.get("source_completeness", "unknown"))
    )
    run_id = str(snapshot.manifest.get("run_id") or "")
    feed: list[dict[str, Any]] = []
    for event, entry in zip(snapshot.events, snapshot.entries):
        episode = episodes_by_id.get(str(entry.get("episode_id") or ""))
        family = str(episode.get("episode_type")) if episode else None
        for profile in profiles:
            feed.append(
                _live_event_row(
                    event,
                    entry,
                    profile=profile,
                    episode_family=family,
                    source_completeness=source_completeness,
                    run_id=run_id,
                )
            )
    scientific_source = _ScientificSource(
        run_root=snapshot.run_root,
        chronicle_root=snapshot.chronicle_root,
        manifest=snapshot.manifest,
        summary=snapshot.summary,
        events=snapshot.events,
        entries=snapshot.entries,
        episodes=snapshot.episodes,
        hashes=[],
    )
    rendered_episodes: list[dict[str, Any]] = []
    for episode in sorted(
        snapshot.episodes,
        key=lambda row: (int(row.get("start_generation", 0)), str(row.get("episode_id", ""))),
    ):
        for profile in profiles:
            rendered_episodes.append(
                _live_episode_row(
                    render_observer_episode(scientific_source, episode, audience_profile=profile)
                )
            )
    return feed, rendered_episodes


def _artifact_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for logical_name in _HASHED_ARTIFACTS:
        filename = LIVE1_ARTIFACTS[logical_name]
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise LiveObserverError(f"LIVE1 publication is missing: {filename}")
        rows.append(
            {
                "artifact": logical_name,
                "relative_path": filename,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return canonical_payload(rows)


def _require_reserved_path_available(path: Path) -> None:
    if path.is_symlink():
        raise LiveObserverError(f"reserved LIVE1 publication path must not be a symlink: {path}")
    if path.exists():
        raise LiveObserverError(f"reserved LIVE1 publication path already exists: {path}")


def _publish_tree(staging: Path, output: Path) -> None:
    backup = output.with_name(f"{output.name}.live1-previous")
    _require_reserved_path_available(backup)
    moved = False
    try:
        if output.exists():
            os.replace(output, backup)
            moved = True
        os.replace(staging, output)
    except Exception:
        if moved and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup.exists():
        import shutil

        shutil.rmtree(backup)


def _write_failure_health(output: Path, exc: Exception) -> None:
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        return
    try:
        write_json(
            output / LIVE1_ARTIFACTS["health"],
            {
                "schema_version": LIVE1_HEALTH_SCHEMA,
                "watcher_version": LIVE1_VERSION,
                "status": "retry" if isinstance(exc, LiveObserverRetry) else "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "observed_at_utc": _utc_now(),
                "source_writes_attempted": False,
            },
        )
    except Exception:
        pass


def catch_up_live_observer(
    source_root: Path,
    output_root: Path,
    *,
    approved_roots: Sequence[Path],
    config: LiveObserverConfig = LiveObserverConfig(),
) -> dict[str, Any]:
    """Publish one manifest-consistent LIVE1 snapshot and crash-safe cursor."""

    config.validate()
    run_root, chronicle_root = _resolve_source(source_root, approved_roots)
    output = _resolve_output(output_root, run_root, chronicle_root)
    try:
        existing_cursor, _existing_feed = _load_existing(output)
        snapshot = _load_snapshot(source_root, approved_roots, config.maximum_source_records)
        identity = _source_identity(snapshot.manifest)
        if existing_cursor is not None:
            if dict(existing_cursor.get("source_identity", {}) or {}) != identity:
                raise LiveObserverError("Scientific Chronicle source identity changed incompatibly")
            consumed = int(existing_cursor.get("source_event_count", 0) or 0)
            if consumed > snapshot.available_count:
                raise LiveObserverError("Scientific Chronicle source was truncated")
            observed_prefix = _event_prefix_digest(snapshot.events[:consumed])
            if observed_prefix != str(existing_cursor.get("consumed_event_prefix_digest") or ""):
                raise LiveObserverError("previously consumed formal-event history was replaced")
            expected_profiles = list(config.audience_profiles)
            if list(existing_cursor.get("audience_profiles", []) or []) != expected_profiles:
                raise LiveObserverError("LIVE1 audience profiles changed across restart")
            if (
                str(existing_cursor.get("source_manifest_sha256")) == snapshot.manifest_sha256
                and str(existing_cursor.get("health_status")) == "ready"
                and not snapshot.partial_event_tail
                and not snapshot.partial_entry_tail
            ):
                return canonical_payload(
                    {
                        "schema_version": LIVE1_STATUS_SCHEMA,
                        "status": "idle",
                        "run_id": identity["run_id"],
                        "source_event_count": consumed,
                        "emitted_item_count": len(existing_cursor.get("emitted_observer_ids", []) or []),
                        "output_root": str(output),
                    }
                )

        feed, episodes = _render_snapshot(snapshot, config.audience_profiles)
        emitted_ids = [str(row["observer_item_id"]) for row in feed]
        if len(emitted_ids) != len(set(emitted_ids)):
            raise LiveObserverError("LIVE1 rendering produced duplicate Observer item IDs")
        expected_count = snapshot.available_count * len(config.audience_profiles)
        if len(feed) != expected_count:
            raise LiveObserverError("LIVE1 rendering skipped completed source events")
        last_event = snapshot.events[-1] if snapshot.events else {}
        last_entry = snapshot.entries[-1] if snapshot.entries else {}
        cursor = canonical_payload(
            {
                "schema_version": LIVE1_CURSOR_SCHEMA,
                "watcher_version": LIVE1_VERSION,
                "run_id": identity["run_id"],
                "source_identity": identity,
                "source_chronicle_root": str(snapshot.chronicle_root),
                "source_artifacts": {
                    "events": "formal_events.jsonl",
                    "entries": "scientific_chronicle.jsonl",
                    "episodes": "episodes.json",
                    "manifest": "chronicle_manifest.json",
                },
                "source_manifest_sha256": snapshot.manifest_sha256,
                "source_event_count": snapshot.available_count,
                "source_episode_count": len(snapshot.episodes),
                "last_complete_event_id": last_event.get("event_id"),
                "last_complete_entry_id": last_entry.get("chronicle_entry_id"),
                "last_generation": last_event.get("generation"),
                "last_step": last_event.get("step"),
                "last_checkpoint_segment": (
                    _checkpoint_segment(last_event, str(identity["run_id"])) if last_event else None
                ),
                "consumed_event_prefix_digest": snapshot.event_prefix_digest,
                "emitted_observer_ids": emitted_ids,
                "audience_profiles": list(config.audience_profiles),
                "source_schema_versions": {
                    "manifest": MANIFEST_SCHEMA_VERSION,
                    "event": EVENT_SCHEMA_VERSION,
                    "entry": CHRONICLE_ENTRY_SCHEMA_VERSION,
                    "episode": EPISODE_SCHEMA_VERSION,
                },
                "health_status": "partial_tail_wait" if snapshot.partial_event_tail or snapshot.partial_entry_tail else "ready",
            }
        )
        index = canonical_payload(
            {
                "schema_version": LIVE1_INDEX_SCHEMA,
                "run_id": identity["run_id"],
                "read_only": True,
                "replay_not_simulation_replay": True,
                "feeds": dict(sorted(LIVE1_ARTIFACTS.items())),
                "counts": {"feed": len(feed), "episodes": len(episodes)},
                "audience_profiles": list(config.audience_profiles),
                "filter_fields": [
                    "audience_profile",
                    "event_category",
                    "live_priority",
                    "public_interest",
                    "visual_interest",
                    "generation",
                    "agent_id",
                    "lineage_id",
                    "region_id",
                    "episode_id",
                    "source_completeness",
                    "global_fallback",
                ],
                "maximum_transport_page": config.maximum_transport_page,
                "maximum_simultaneous_bubbles": config.maximum_simultaneous_bubbles,
            }
        )
        status = canonical_payload(
            {
                "schema_version": LIVE1_STATUS_SCHEMA,
                "watcher_version": LIVE1_VERSION,
                "status": "partial_tail_wait" if snapshot.partial_event_tail or snapshot.partial_entry_tail else "ready",
                "run_id": identity["run_id"],
                "source_finalized": bool(snapshot.manifest.get("finalized")),
                "source_event_count": snapshot.available_count,
                "source_declared_event_count": int(snapshot.manifest.get("event_count", 0) or 0),
                "source_episode_count": len(snapshot.episodes),
                "emitted_item_count": len(feed),
                "map_anchored_item_count": sum(1 for row in feed if row.get("map_anchor")),
                "global_fallback_item_count": sum(1 for row in feed if row.get("global_fallback")),
                "source_writes_attempted": False,
                "simulation_control_available": False,
                "last_poll_utc": _utc_now(),
            }
        )
        health = canonical_payload(
            {
                "schema_version": LIVE1_HEALTH_SCHEMA,
                "watcher_version": LIVE1_VERSION,
                "status": status["status"],
                "run_id": identity["run_id"],
                "partial_event_tail": snapshot.partial_event_tail,
                "partial_entry_tail": snapshot.partial_entry_tail,
                "source_writes_attempted": False,
                "observed_at_utc": _utc_now(),
            }
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        staging = output.with_name(f"{output.name}.live1-staging")
        _require_reserved_path_available(staging)
        staging.mkdir()
        try:
            write_jsonl(staging / LIVE1_ARTIFACTS["feed"], feed)
            write_jsonl(staging / LIVE1_ARTIFACTS["episodes"], episodes)
            write_json(staging / LIVE1_ARTIFACTS["cursor"], cursor)
            write_json(staging / LIVE1_ARTIFACTS["index"], index)
            write_json(staging / LIVE1_ARTIFACTS["status"], status)
            write_json(staging / LIVE1_ARTIFACTS["health"], health)
            artifacts = _artifact_inventory(staging)
            manifest = canonical_payload(
                {
                    "schema_version": LIVE1_MANIFEST_SCHEMA,
                    "watcher_version": LIVE1_VERSION,
                    "run_id": identity["run_id"],
                    "source_identity": identity,
                    "source_chronicle_root": str(snapshot.chronicle_root),
                    "source_manifest_sha256": snapshot.manifest_sha256,
                    "source_event_prefix_digest": snapshot.event_prefix_digest,
                    "source_event_count": snapshot.available_count,
                    "source_episode_count": len(snapshot.episodes),
                    "audience_profiles": list(config.audience_profiles),
                    "read_only_external_consumer": True,
                    "source_writes_attempted": False,
                    "simulation_control_available": False,
                    "observer_renderer": "OBS1 render_observer_event/render_observer_episode",
                    "map_anchor_policy": "exact_patch_coordinates_only_v1",
                    "global_fallback_for_unanchored": True,
                    "gen1_boundary": {
                        "maximum_evidence_strength": "observed",
                        "no_matching_record_text": "No matching genome-attribution record is available for this run.",
                        "inheritance": "unavailable",
                        "persistence": "unavailable",
                        "effect": "unavailable",
                    },
                    "config": config.to_dict(),
                    "artifacts": artifacts,
                }
            )
            write_json(staging / LIVE1_ARTIFACTS["manifest"], manifest)
            _publish_tree(staging, output)
        except Exception:
            if staging.exists() and not staging.is_symlink():
                import shutil

                shutil.rmtree(staging, ignore_errors=True)
            raise
        return status
    except Exception as exc:
        _write_failure_health(output, exc)
        raise


def discover_chronicle_runs(approved_root: Path, *, maximum_depth: int = 8) -> list[Path]:
    root = approved_root.expanduser()
    if root.is_symlink() or not root.is_dir():
        raise LiveObserverError("approved discovery root is missing or unsafe")
    resolved = root.resolve()
    rows: list[Path] = []
    for manifest in resolved.rglob("chronicle_manifest.json"):
        try:
            relative = manifest.relative_to(resolved)
        except ValueError:
            continue
        if len(relative.parts) > maximum_depth + 1 or manifest.is_symlink():
            continue
        chronicle = manifest.parent
        if chronicle.is_symlink() or not _inside(chronicle, resolved):
            continue
        rows.append(chronicle.parent if chronicle.name == "chronicle" else chronicle)
    return sorted(set(rows), key=lambda path: path.as_posix())


def catch_up_approved_root(
    source_root: Path,
    output_root: Path,
    *,
    config: LiveObserverConfig = LiveObserverConfig(),
    maximum_depth: int = 8,
) -> list[dict[str, Any]]:
    source = source_root.expanduser().resolve()
    output = output_root.expanduser().resolve()
    results: list[dict[str, Any]] = []
    for run_root in discover_chronicle_runs(source, maximum_depth=maximum_depth):
        relative = run_root.relative_to(source).as_posix()
        key = "live_" + stable_hash("eiel-live1-discovered-run-v1", relative)
        try:
            result = catch_up_live_observer(
                run_root,
                output / key,
                approved_roots=(source,),
                config=config,
            )
            results.append({"run_root": str(run_root), "live_key": key, "result": result})
        except LiveObserverError as exc:
            results.append(
                {
                    "run_root": str(run_root),
                    "live_key": key,
                    "result": {"status": "failed_closed", "error": str(exc)},
                }
            )
    return canonical_payload(results)


def follow_live_observer(
    source_root: Path,
    output_root: Path,
    *,
    approved_roots: Sequence[Path],
    config: LiveObserverConfig = LiveObserverConfig(),
    maximum_polls: int | None = None,
) -> dict[str, Any]:
    """Poll one source until interrupted; maximum_polls is for controlled fixtures."""

    polls = 0
    last: dict[str, Any] = {}
    try:
        while maximum_polls is None or polls < maximum_polls:
            polls += 1
            try:
                last = catch_up_live_observer(
                    source_root,
                    output_root,
                    approved_roots=approved_roots,
                    config=config,
                )
            except LiveObserverRetry as exc:
                last = {"schema_version": LIVE1_STATUS_SCHEMA, "status": "retry", "error": str(exc)}
            if maximum_polls is not None and polls >= maximum_polls:
                break
            time.sleep(config.poll_interval_seconds)
    except KeyboardInterrupt:
        last = {**last, "status": "closed_cleanly"}
    return canonical_payload({"poll_count": polls, "last_status": last})


__all__ = [
    "LIVE1_ARTIFACTS",
    "LIVE1_CURSOR_SCHEMA",
    "LIVE1_EPISODE_SCHEMA",
    "LIVE1_HEALTH_SCHEMA",
    "LIVE1_INDEX_SCHEMA",
    "LIVE1_ITEM_SCHEMA",
    "LIVE1_MANIFEST_SCHEMA",
    "LIVE1_STATUS_SCHEMA",
    "LIVE1_VERSION",
    "LiveObserverConfig",
    "LiveObserverError",
    "LiveObserverRetry",
    "catch_up_approved_root",
    "catch_up_live_observer",
    "discover_chronicle_runs",
    "follow_live_observer",
]
