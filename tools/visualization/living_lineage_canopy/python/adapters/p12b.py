from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .. import SCHEMA_VERSION
from ..schema import validate_playback


def classify_p12b(root: Path) -> dict[str, Any]:
    events = root / "07_successor_science_reduction_ab5b5c659" / "p12b_science_birth_events.csv"
    settlement = root / "07_successor_science_reduction_ab5b5c659" / "p12b_science_birth_settlement.csv"
    if not events.exists():
        return {
            "classification": "D",
            "semantics_mode": "insufficient",
            "reason": f"missing birth event table: {events}",
            "missing_fields": ["agent identity", "birth time", "death time", "parent relationships"],
        }
    with events.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
    required = {"child_id", "parent_id", "lineage_founder_id", "generation", "event_sequence", "cell_id"}
    missing = sorted(required - fields)
    if missing:
        return {
            "classification": "D",
            "semantics_mode": "insufficient",
            "reason": "birth event table lacks required fields",
            "missing_fields": missing,
        }
    return {
        "classification": "B",
        "semantics_mode": "partial_agents",
        "reason": "P12B event reductions expose individual birth receipts and parents, but not complete per-agent death timing or full initial population lifecycles.",
        "event_path": str(events),
        "settlement_path": str(settlement) if settlement.exists() else None,
        "missing_fields": [
            "authoritative death tick for every agent",
            "complete initial population birth records",
            "full branch semantics beyond founder lineage and source cell",
        ],
    }


def adapt_p12b(root: Path, *, limit_rows: int = 800) -> dict[str, Any]:
    classification = classify_p12b(root)
    if classification["classification"] == "D":
        raise ValueError(classification["reason"])
    events = Path(classification["event_path"])
    lineages: dict[str, dict[str, Any]] = {}
    branches: dict[str, dict[str, Any]] = {}
    agents: list[dict[str, Any]] = []
    seen_agents: set[str] = set()
    max_tick = 0
    with events.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index >= limit_rows:
                break
            founder = row["lineage_founder_id"]
            cell = row["cell_id"]
            lineage_id = f"p12b_founder_{founder}"
            branch_id = f"{lineage_id}_{cell[-6:]}"
            tick = int(row.get("event_sequence") or row.get("generation") or index)
            max_tick = max(max_tick, tick)
            lineages.setdefault(
                lineage_id,
                {
                    "lineage_id": lineage_id,
                    "parent_lineage_id": None,
                    "founder_lineage_id": lineage_id,
                    "creation_tick": 0,
                    "extinction_tick": None,
                    "display_label": f"P12B founder {founder}",
                },
            )
            branches.setdefault(
                branch_id,
                {
                    "branch_id": branch_id,
                    "lineage_id": lineage_id,
                    "parent_branch_id": None,
                    "creation_tick": 0,
                    "extinction_tick": None,
                    "display_label": cell,
                },
            )
            child = f"p12b_{cell}_{row['child_id']}"
            if child in seen_agents:
                continue
            seen_agents.add(child)
            parents = []
            for field in ("parent_id", "mate_parent_id"):
                value = row.get(field)
                if value not in {None, "", "-1"}:
                    parents.append(f"p12b_external_{cell}_{value}")
            agents.append(
                {
                    "agent_id": child,
                    "lineage_id": lineage_id,
                    "branch_id": branch_id,
                    "birth_tick": tick,
                    "death_tick": None,
                    "parent_agent_ids": parents[:2],
                    "generation_born": int(row["generation"]) if row.get("generation") else None,
                    "generation_died": None,
                }
            )
    data = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "p12b_partial_birth_events",
            "source_kind": "real_adapter",
            "source_path": str(root),
            "provenance_classification": classification["reason"],
            "visualization_seed": 8701,
            "time_unit": "event_sequence",
            "first_tick": 0,
            "last_tick": max_tick,
            "generation_mapping": None,
            "semantics_mode": "partial_agents",
            "parent_reference_policy": "allow_missing_external",
            "notes": "Partial P12B adapter: particles represent captured birth-event children only; missing parents are marked external and death timing is unavailable.",
            "adapter_classification": classification,
        },
        "lineages": sorted(lineages.values(), key=lambda item: item["lineage_id"]),
        "branches": sorted(branches.values(), key=lambda item: item["branch_id"]),
        "agents": agents,
    }
    validate_playback(data)
    return data
