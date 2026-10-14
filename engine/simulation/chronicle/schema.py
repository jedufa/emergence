from __future__ import annotations

"""Canonical schemas and deterministic identifiers for Scientific Chronicle V1."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from typing import Any, Iterable, Mapping


EVENT_SCHEMA_VERSION = "eiel_formal_event_v1"
CHRONICLE_ENTRY_SCHEMA_VERSION = "eiel_scientific_chronicle_entry_v1"
EPISODE_SCHEMA_VERSION = "eiel_scientific_episode_v1"
CONTEXT_SCHEMA_VERSION = "eiel_event_context_window_v1"
MANIFEST_SCHEMA_VERSION = "eiel_scientific_chronicle_manifest_v1"
COMPARISON_SCHEMA_VERSION = "eiel_scientific_chronicle_comparison_v1"

EVENT_CATEGORIES = (
    "summary",
    "lifecycle",
    "lineage",
    "ecology",
    "hazard",
    "population",
    "inheritance",
    "integrity",
)
LIVE_PRIORITIES = ("critical", "high", "medium", "low", "background")
CONFIDENCE_CLASSES = ("measured", "derived", "partial", "unknown")
CAUSALITY_CLASSES = ("direct_mechanism", "temporal_association", "association", "not_assessed")
SEVERITIES = ("critical", "major", "moderate", "minor", "informational")
TIME_SCOPES = ("step", "generation", "generation_range", "run")


class ChronicleSchemaError(ValueError):
    """Raised when a formal Chronicle payload violates the V1 contract."""


def _canonical_value(value: Any, *, path: str = "root") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ChronicleSchemaError(f"non-finite value at {path}")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item, path=f"{path}[]") for item in value]
        return sorted(normalized, key=canonical_json)
    if hasattr(value, "item"):
        try:
            return _canonical_value(value.item(), path=path)
        except Exception:
            pass
    raise ChronicleSchemaError(f"unsupported value at {path}: {type(value).__name__}")


def canonical_payload(value: Any) -> Any:
    return _canonical_value(value)


def canonical_json(value: Any) -> str:
    return json.dumps(canonical_payload(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(namespace: str, value: Any, *, length: int = 32) -> str:
    digest = hashlib.sha256((str(namespace) + "\n" + canonical_json(value)).encode("utf-8")).hexdigest()
    return digest[: max(12, int(length))]


def _strings(values: Iterable[Any] | None) -> tuple[str, ...]:
    return tuple(sorted({str(item) for item in (values or ()) if str(item)}))


def _ints(values: Iterable[Any] | None) -> tuple[int, ...]:
    out: set[int] = set()
    for item in values or ():
        try:
            out.add(int(item))
        except Exception:
            continue
    return tuple(sorted(out))


@dataclass(frozen=True)
class FormalEvent:
    schema_version: str
    event_id: str
    run_id: str
    campaign_id: str | None
    branch_id: str | None
    generation: int
    step: int
    phase_order: int
    intra_step_sequence: int
    event_time_scope: str
    event_type: str
    event_category: str
    subject_type: str
    subject_id: str | None
    agent_ids: tuple[int, ...] = field(default_factory=tuple)
    lineage_ids: tuple[str, ...] = field(default_factory=tuple)
    patch_coordinates: tuple[int, int] | None = None
    region_id: str | None = None
    severity: str = "informational"
    live_priority: str = "background"
    scientific_importance: int = 0
    historical_importance: int = 0
    public_interest: int = 0
    visual_interest: int = 0
    novelty_class: str = "routine"
    confidence_class: str = "measured"
    causality_class: str = "not_assessed"
    source_component: str = ""
    source_record_refs: tuple[str, ...] = field(default_factory=tuple)
    parent_event_ids: tuple[str, ...] = field(default_factory=tuple)
    related_event_ids: tuple[str, ...] = field(default_factory=tuple)
    episode_hint: str | None = None
    measurements: Mapping[str, Any] = field(default_factory=dict)
    state_before_summary: Mapping[str, Any] | None = None
    state_after_summary: Mapping[str, Any] | None = None
    factual_summary_fields: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return canonical_payload(asdict(self))


def _bounded_importance(value: Any, name: str) -> int:
    try:
        result = int(value)
    except Exception as exc:
        raise ChronicleSchemaError(f"{name} must be an integer") from exc
    if result < 0 or result > 5:
        raise ChronicleSchemaError(f"{name} must be between 0 and 5")
    return result


def event_order_key(event: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
    return (
        int(event.get("generation", 0)),
        int(event.get("step", 0)),
        int(event.get("phase_order", 0)),
        int(event.get("intra_step_sequence", 0)),
        str(event.get("event_id", "")),
    )


def make_event(
    *,
    run_id: str,
    generation: int,
    step: int,
    phase_order: int,
    intra_step_sequence: int,
    event_type: str,
    event_category: str,
    subject_type: str,
    subject_id: str | None = None,
    campaign_id: str | None = None,
    branch_id: str | None = None,
    event_time_scope: str = "generation",
    agent_ids: Iterable[Any] = (),
    lineage_ids: Iterable[Any] = (),
    patch_coordinates: tuple[int, int] | None = None,
    region_id: str | None = None,
    severity: str = "informational",
    live_priority: str = "background",
    scientific_importance: int = 0,
    historical_importance: int = 0,
    public_interest: int = 0,
    visual_interest: int = 0,
    novelty_class: str = "routine",
    confidence_class: str = "measured",
    causality_class: str = "not_assessed",
    source_component: str,
    source_record_refs: Iterable[Any] = (),
    parent_event_ids: Iterable[Any] = (),
    related_event_ids: Iterable[Any] = (),
    episode_hint: str | None = None,
    measurements: Mapping[str, Any] | None = None,
    state_before_summary: Mapping[str, Any] | None = None,
    state_after_summary: Mapping[str, Any] | None = None,
    factual_summary_fields: Mapping[str, Any] | None = None,
    tags: Iterable[Any] = (),
) -> dict[str, Any]:
    identity = {
        "run_id": str(run_id),
        "campaign_id": str(campaign_id) if campaign_id else None,
        "branch_id": str(branch_id) if branch_id else None,
        "generation": int(generation),
        "step": int(step),
        "phase_order": int(phase_order),
        "intra_step_sequence": int(intra_step_sequence),
        "event_type": str(event_type),
        "event_category": str(event_category),
        "subject_type": str(subject_type),
        "subject_id": str(subject_id) if subject_id is not None else None,
        "agent_ids": _ints(agent_ids),
        "lineage_ids": _strings(lineage_ids),
        "patch_coordinates": tuple(int(v) for v in patch_coordinates) if patch_coordinates is not None else None,
        "region_id": str(region_id) if region_id else None,
        "source_component": str(source_component),
        "source_record_refs": _strings(source_record_refs),
        "measurements": canonical_payload(measurements or {}),
        "factual_summary_fields": canonical_payload(factual_summary_fields or {}),
    }
    event_id = "evt_" + stable_hash("eiel-formal-event-v1", identity)
    event = FormalEvent(
        schema_version=EVENT_SCHEMA_VERSION,
        event_id=event_id,
        run_id=str(run_id),
        campaign_id=str(campaign_id) if campaign_id else None,
        branch_id=str(branch_id) if branch_id else None,
        generation=int(generation),
        step=int(step),
        phase_order=int(phase_order),
        intra_step_sequence=int(intra_step_sequence),
        event_time_scope=str(event_time_scope),
        event_type=str(event_type),
        event_category=str(event_category),
        subject_type=str(subject_type),
        subject_id=str(subject_id) if subject_id is not None else None,
        agent_ids=_ints(agent_ids),
        lineage_ids=_strings(lineage_ids),
        patch_coordinates=tuple(int(v) for v in patch_coordinates) if patch_coordinates is not None else None,
        region_id=str(region_id) if region_id else None,
        severity=str(severity),
        live_priority=str(live_priority),
        scientific_importance=_bounded_importance(scientific_importance, "scientific_importance"),
        historical_importance=_bounded_importance(historical_importance, "historical_importance"),
        public_interest=_bounded_importance(public_interest, "public_interest"),
        visual_interest=_bounded_importance(visual_interest, "visual_interest"),
        novelty_class=str(novelty_class),
        confidence_class=str(confidence_class),
        causality_class=str(causality_class),
        source_component=str(source_component),
        source_record_refs=_strings(source_record_refs),
        parent_event_ids=_strings(parent_event_ids),
        related_event_ids=_strings(related_event_ids),
        episode_hint=str(episode_hint) if episode_hint else None,
        measurements=canonical_payload(measurements or {}),
        state_before_summary=canonical_payload(state_before_summary) if state_before_summary is not None else None,
        state_after_summary=canonical_payload(state_after_summary) if state_after_summary is not None else None,
        factual_summary_fields=canonical_payload(factual_summary_fields or {}),
        tags=_strings(tags),
    ).to_dict()
    validate_event(event)
    return event


def validate_event(event: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "event_id",
        "run_id",
        "generation",
        "step",
        "phase_order",
        "intra_step_sequence",
        "event_time_scope",
        "event_type",
        "event_category",
        "subject_type",
        "severity",
        "live_priority",
        "scientific_importance",
        "historical_importance",
        "public_interest",
        "visual_interest",
        "confidence_class",
        "causality_class",
        "source_component",
        "source_record_refs",
        "measurements",
        "factual_summary_fields",
        "tags",
    }
    missing = sorted(required.difference(event.keys()))
    if missing:
        raise ChronicleSchemaError(f"missing event fields: {', '.join(missing)}")
    if str(event["schema_version"]) != EVENT_SCHEMA_VERSION:
        raise ChronicleSchemaError(f"unsupported event schema: {event['schema_version']}")
    if not str(event["event_id"]).startswith("evt_") or not str(event["run_id"]):
        raise ChronicleSchemaError("event_id and run_id must be populated")
    if str(event["event_time_scope"]) not in TIME_SCOPES:
        raise ChronicleSchemaError(f"invalid event_time_scope: {event['event_time_scope']}")
    if str(event["event_category"]) not in EVENT_CATEGORIES:
        raise ChronicleSchemaError(f"invalid event_category: {event['event_category']}")
    if str(event["severity"]) not in SEVERITIES:
        raise ChronicleSchemaError(f"invalid severity: {event['severity']}")
    if str(event["live_priority"]) not in LIVE_PRIORITIES:
        raise ChronicleSchemaError(f"invalid live_priority: {event['live_priority']}")
    if str(event["confidence_class"]) not in CONFIDENCE_CLASSES:
        raise ChronicleSchemaError(f"invalid confidence_class: {event['confidence_class']}")
    if str(event["causality_class"]) not in CAUSALITY_CLASSES:
        raise ChronicleSchemaError(f"invalid causality_class: {event['causality_class']}")
    for name in ("scientific_importance", "historical_importance", "public_interest", "visual_interest"):
        _bounded_importance(event[name], name)
    canonical_payload(event)


def assert_unique_events(events: Iterable[Mapping[str, Any]]) -> None:
    seen: dict[str, str] = {}
    last_key: tuple[int, int, int, int, str] | None = None
    for event in events:
        validate_event(event)
        event_id = str(event["event_id"])
        encoded = canonical_json(event)
        if event_id in seen:
            if seen[event_id] != encoded:
                raise ChronicleSchemaError(f"event ID collision: {event_id}")
            raise ChronicleSchemaError(f"duplicate event ID: {event_id}")
        key = event_order_key(event)
        if last_key is not None and key < last_key:
            raise ChronicleSchemaError(f"events are not in total order at {event_id}")
        seen[event_id] = encoded
        last_key = key
