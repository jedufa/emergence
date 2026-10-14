from __future__ import annotations

"""Deterministic filtering helpers for formal Chronicle events."""

from typing import Any, Iterable, Mapping

from simulation.chronicle.schema import event_order_key


def filter_events(
    events: Iterable[Mapping[str, Any]],
    *,
    event_types: Iterable[str] = (),
    categories: Iterable[str] = (),
    priorities: Iterable[str] = (),
    lineage_ids: Iterable[str] = (),
    subject_ids: Iterable[str] = (),
    region_ids: Iterable[str] = (),
    generation_start: int | None = None,
    generation_end: int | None = None,
    minimum_scientific_importance: int = 0,
) -> list[dict[str, Any]]:
    """Return a stable ordered subset without mutating the input records."""

    type_set = {str(value) for value in event_types}
    category_set = {str(value) for value in categories}
    priority_set = {str(value) for value in priorities}
    lineage_set = {str(value) for value in lineage_ids}
    subject_set = {str(value) for value in subject_ids}
    region_set = {str(value) for value in region_ids}
    selected: list[dict[str, Any]] = []
    for source in events:
        event = dict(source)
        generation = int(event.get("generation", 0))
        if type_set and str(event.get("event_type")) not in type_set:
            continue
        if category_set and str(event.get("event_category")) not in category_set:
            continue
        if priority_set and str(event.get("live_priority")) not in priority_set:
            continue
        if lineage_set and not lineage_set.intersection(str(value) for value in event.get("lineage_ids", []) or []):
            continue
        if subject_set and str(event.get("subject_id")) not in subject_set:
            continue
        if region_set and str(event.get("region_id")) not in region_set:
            continue
        if generation_start is not None and generation < int(generation_start):
            continue
        if generation_end is not None and generation > int(generation_end):
            continue
        if int(event.get("scientific_importance", 0)) < int(minimum_scientific_importance):
            continue
        selected.append(event)
    return sorted(selected, key=event_order_key)
