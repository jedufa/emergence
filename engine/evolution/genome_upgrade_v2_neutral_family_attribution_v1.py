"""Neutral hereditary-family attribution for bounded Genome V2 bridge runs.

The marker is intentionally non-phenotypic.  It is advanced from authoritative
new-organism and death events only; it never feeds policy, resources, survival,
fertility, reward, or Genome V2 expression.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final, Literal


SCHEMA_VERSION: Final[str] = "genome_upgrade_v2_neutral_family_attribution_v1"
SCHEDULE_SCHEMA_VERSION: Final[str] = "genome_upgrade_v2_neutral_family_schedule_v1"
MARKER_ID: Final[str] = "genome_upgrade_v2.neutral_family_marker_v1"
EventKind = Literal["ordinary_birth", "turnover_refill_birth", "death"]


class NeutralFamilyAttributionError(ValueError):
    """Neutral-family attribution input or transition is contradictory."""


def _fail(message: str) -> NeutralFamilyAttributionError:
    return NeutralFamilyAttributionError(message)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class NeutralFamilyWindowV1:
    family_id: str
    opens_at_tick: int
    closes_at_tick: int
    eligible_event_kind: Literal["ordinary_birth", "turnover_refill_birth"]
    maximum_introductions: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "opens_at_tick": self.opens_at_tick,
            "closes_at_tick": self.closes_at_tick,
            "eligible_event_kind": self.eligible_event_kind,
            "maximum_introductions": self.maximum_introductions,
        }


@dataclass(frozen=True, slots=True)
class NeutralFamilyEventV1:
    cell_id: str
    generation: int
    tick: int
    event_kind: EventKind
    child_agent_id: int | None = None
    parent_agent_ids: tuple[int, ...] = ()
    slot: int | None = None
    reason: str | None = None

    @property
    def physical_tick(self) -> int:
        return self.generation * 80 + self.tick

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "generation": self.generation,
            "tick": self.tick,
            "physical_tick": self.physical_tick,
            "event_kind": self.event_kind,
            "child_agent_id": self.child_agent_id,
            "parent_agent_ids": list(self.parent_agent_ids),
            "slot": self.slot,
            "reason": self.reason,
        }


def _validate_window(window: NeutralFamilyWindowV1) -> None:
    if type(window.family_id) is not str or not window.family_id:
        raise _fail("neutral family window requires a non-empty family_id")
    if (
        type(window.opens_at_tick) is not int
        or type(window.closes_at_tick) is not int
        or window.opens_at_tick < 0
        or window.closes_at_tick < window.opens_at_tick
    ):
        raise _fail("neutral family window has invalid physical tick bounds")
    if window.eligible_event_kind not in {"ordinary_birth", "turnover_refill_birth"}:
        raise _fail("neutral family window has an invalid eligible event kind")
    if type(window.maximum_introductions) is not int or window.maximum_introductions != 1:
        raise _fail("neutral family schedule v1 admits exactly one introduction")


def _validate_event(event: NeutralFamilyEventV1) -> None:
    if type(event.cell_id) is not str or not event.cell_id:
        raise _fail("neutral family event requires a non-empty cell_id")
    if type(event.generation) is not int or event.generation < 0:
        raise _fail("neutral family event generation must be non-negative")
    if type(event.tick) is not int or event.tick < 0:
        raise _fail("neutral family event tick must be non-negative")
    if event.event_kind not in {"ordinary_birth", "turnover_refill_birth", "death"}:
        raise _fail("neutral family event kind is invalid")
    if event.event_kind == "death":
        if event.child_agent_id is None:
            raise _fail("death events must name the removed biological ID")
    else:
        if event.child_agent_id is None or event.child_agent_id < 0:
            raise _fail("new-organism events must name the child biological ID")
    if len(set(event.parent_agent_ids)) != len(event.parent_agent_ids):
        raise _fail("neutral family event has duplicate parent IDs")
    if any(type(parent) is not int or parent < 0 for parent in event.parent_agent_ids):
        raise _fail("neutral family event parent IDs must be non-negative integers")
    if event.slot is not None and (type(event.slot) is not int or event.slot < 0):
        raise _fail("neutral family event slot must be non-negative when present")


def build_schedule_payload_v1(
    *,
    cell_id: str,
    paired_seed: int,
    windows: Sequence[NeutralFamilyWindowV1],
) -> dict[str, Any]:
    if type(cell_id) is not str or not cell_id:
        raise _fail("neutral family schedule requires a non-empty cell_id")
    if type(paired_seed) is not int or paired_seed < 0:
        raise _fail("neutral family schedule requires a non-negative paired seed")
    if isinstance(windows, (str, bytes)) or not windows:
        raise _fail("neutral family schedule requires at least one window")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for window in windows:
        _validate_window(window)
        if window.family_id in seen:
            raise _fail("neutral family schedule has duplicate family IDs")
        seen.add(window.family_id)
        rows.append(window.to_dict())
    payload = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "cell_id": cell_id,
        "paired_seed": paired_seed,
        "marker_id": MARKER_ID,
        "introduction_rule": "first_eligible_new_organism_after_window_opens_v1",
        "windows": rows,
    }
    payload["schedule_sha256"] = _canonical_sha256(payload)
    return payload


def attribute_neutral_families_v1(
    *,
    cell_id: str,
    paired_seed: int,
    windows: Sequence[NeutralFamilyWindowV1],
    events: Iterable[NeutralFamilyEventV1],
    initial_carriers: Mapping[int, Sequence[str]] | None = None,
    fail_on_missing_introduction: bool = True,
) -> dict[str, Any]:
    """Apply neutral hereditary-family transitions to authoritative events.

    A new child inherits the union of both parents' carried families.  If the
    child is the first eligible newborn in an open scheduled window, that
    scheduled family is introduced on the child in addition to inherited
    families.  Death removes the biological ID from the carrier table, so slot
    reuse cannot preserve a marker by accident.
    """

    schedule = build_schedule_payload_v1(
        cell_id=cell_id,
        paired_seed=paired_seed,
        windows=windows,
    )
    windows_by_id = {window.family_id: window for window in windows}
    introduced_counts = {family_id: 0 for family_id in windows_by_id}
    carriers: dict[int, frozenset[str]] = {}
    if initial_carriers is not None:
        for raw_agent_id, raw_families in initial_carriers.items():
            if type(raw_agent_id) is not int or raw_agent_id < 0:
                raise _fail("initial carrier IDs must be non-negative integers")
            family_set = frozenset(str(family) for family in raw_families)
            unknown = sorted(family_set - set(windows_by_id))
            if unknown:
                raise _fail("initial carriers reference unknown neutral families")
            if family_set:
                carriers[raw_agent_id] = family_set
    carrier_depths: dict[int, dict[str, int]] = {
        agent_id: {family_id: 0 for family_id in families}
        for agent_id, families in carriers.items()
    }

    ordered_events = list(events)
    for event in ordered_events:
        _validate_event(event)
        if event.cell_id != cell_id:
            raise _fail("neutral family event cell_id contradicts schedule")
    ordered_events.sort(
        key=lambda event: (
            event.physical_tick,
            0 if event.event_kind == "death" else 1,
            event.child_agent_id if event.child_agent_id is not None else -1,
        )
    )

    introduction_records: list[dict[str, Any]] = []
    transmission_records: list[dict[str, Any]] = []
    extinction_records: list[dict[str, Any]] = []
    carrier_counts: dict[str, int] = {family_id: 0 for family_id in windows_by_id}

    def refresh_counts() -> None:
        for family_id in carrier_counts:
            carrier_counts[family_id] = sum(
                1 for families in carriers.values() if family_id in families
            )

    refresh_counts()
    for event in ordered_events:
        if event.event_kind == "death":
            removed = carriers.pop(int(event.child_agent_id), frozenset())
            carrier_depths.pop(int(event.child_agent_id), None)
            refresh_counts()
            for family_id in sorted(removed):
                if carrier_counts[family_id] == 0:
                    extinction_records.append(
                        {
                            **event.to_dict(),
                            "family_id": family_id,
                            "extinction_reason": event.reason or "death",
                        }
                    )
            continue

        inherited: set[str] = set()
        for parent_id in event.parent_agent_ids:
            inherited.update(carriers.get(parent_id, frozenset()))
        introduced: list[str] = []
        for family_id, window in windows_by_id.items():
            if introduced_counts[family_id] >= window.maximum_introductions:
                continue
            if event.event_kind != window.eligible_event_kind:
                continue
            if window.opens_at_tick <= event.physical_tick <= window.closes_at_tick:
                introduced_counts[family_id] += 1
                introduced.append(family_id)
        child_families = frozenset((*inherited, *introduced))
        child_depths: dict[str, int] = {}
        for family_id in child_families:
            inherited_depths = [
                carrier_depths.get(parent_id, {}).get(family_id)
                for parent_id in event.parent_agent_ids
                if family_id in carrier_depths.get(parent_id, {})
            ]
            if inherited_depths:
                child_depths[family_id] = max(int(value) for value in inherited_depths) + 1
            else:
                child_depths[family_id] = 0
        if child_families:
            carriers[int(event.child_agent_id)] = child_families
            carrier_depths[int(event.child_agent_id)] = child_depths
        else:
            carriers.pop(int(event.child_agent_id), None)
            carrier_depths.pop(int(event.child_agent_id), None)
        refresh_counts()
        for family_id in introduced:
            introduction_records.append(
                {
                    **event.to_dict(),
                    "family_id": family_id,
                    "marker_id": MARKER_ID,
                    "introduced_count": introduced_counts[family_id],
                }
            )
        for family_id in sorted(child_families):
            transmission_records.append(
                {
                    **event.to_dict(),
                    "family_id": family_id,
                    "inherited": family_id in inherited,
                    "introduced": family_id in introduced,
                    "parent_carriers": [
                        parent_id
                        for parent_id in event.parent_agent_ids
                        if family_id in carriers.get(parent_id, frozenset())
                    ],
                    "hereditary_depth": child_depths[family_id],
                }
            )

    not_introduced = sorted(
        family_id
        for family_id, count in introduced_counts.items()
        if count != windows_by_id[family_id].maximum_introductions
    )
    if fail_on_missing_introduction and not_introduced:
        raise _fail(
            "neutral family schedule did not receive all introductions: "
            + ", ".join(not_introduced)
        )
    terminal_carriers = {
        str(agent_id): sorted(families)
        for agent_id, families in sorted(carriers.items())
        if families
    }
    family_summaries: dict[str, dict[str, Any]] = {}
    for family_id in sorted(windows_by_id):
        family_transmissions = [
            record for record in transmission_records if record["family_id"] == family_id
        ]
        introduced_children = {
            int(record["child_agent_id"])
            for record in family_transmissions
            if record["introduced"]
        }
        descendant_children = {
            int(record["child_agent_id"])
            for record in family_transmissions
            if record["inherited"]
        }
        living_carriers = sorted(
            agent_id for agent_id, families in carriers.items() if family_id in families
        )
        family_summaries[family_id] = {
            "introduced": introduced_counts[family_id],
            "introduced_child_agent_ids": sorted(int(item) for item in introduced_children),
            "unique_descendant_agent_ids": sorted(descendant_children),
            "unique_descendant_count": len(descendant_children),
            "terminal_living_carrier_agent_ids": living_carriers,
            "terminal_living_carrier_count": len(living_carriers),
            "maximum_hereditary_depth": max(
                (int(record["hereditary_depth"]) for record in family_transmissions),
                default=0,
            ),
            "extinct": not living_carriers,
            "extinction_records": [
                record for record in extinction_records if record["family_id"] == family_id
            ],
        }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "paired_seed": paired_seed,
        "marker_id": MARKER_ID,
        "neutrality_contract": {
            "phenotypic": False,
            "policy_input": False,
            "resource_input": False,
            "lifecycle_input": False,
            "reward_input": False,
            "selection_input": False,
            "state_transition_source": (
                "authoritative_new_organism_and_death_events_v1"
            ),
        },
        "schedule": schedule,
        "introduction_records": introduction_records,
        "transmission_records": transmission_records,
        "extinction_records": extinction_records,
        "terminal_carriers_by_agent_id": terminal_carriers,
        "family_summaries": family_summaries,
        "introduced_family_count": len(introduction_records),
        "terminal_carrier_count": len(terminal_carriers),
        "not_introduced_family_ids": not_introduced,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


__all__ = [
    "MARKER_ID",
    "SCHEMA_VERSION",
    "SCHEDULE_SCHEMA_VERSION",
    "NeutralFamilyAttributionError",
    "NeutralFamilyEventV1",
    "NeutralFamilyWindowV1",
    "attribute_neutral_families_v1",
    "build_schedule_payload_v1",
]
