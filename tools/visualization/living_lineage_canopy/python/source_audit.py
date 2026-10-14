from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifact_readers.checkpoint_bundle_v2 import choose_cell, projection_paths_for_cell, snapshot_from_projection, typed_lifecycle_events


@dataclass(frozen=True)
class SourceCandidate:
    source_root: str
    classification: str
    semantics_mode: str
    confidence: str
    supporting_files: list[str]
    available_fields: list[str]
    missing_fields: list[str]
    timing_resolution: str
    identity_continuity: str
    parent_edge_status: str
    lineage_authority_status: str
    death_coverage: str
    branch_semantics: str
    coverage_interval: str
    restrictions: list[str]


P12B_EVENT_FIELDS = {
    "cell_id",
    "child_id",
    "parent_id",
    "mate_parent_id",
    "lineage_founder_id",
    "generation",
    "event_sequence",
}


def _csv_fields(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return set(csv.DictReader(handle).fieldnames or [])


def _json_keys(path: Path) -> set[str]:
    try:
        value = json.loads(path.read_text())
    except Exception:
        return set()
    return set(value) if isinstance(value, dict) else set()


def _lineage_metrics_nonempty(root: Path) -> bool:
    for path in root.rglob("metrics/lineage_events.jsonl"):
        try:
            if path.stat().st_size > 0:
                return True
        except OSError:
            continue
    return False


def _checkpoint_population_count(root: Path) -> int:
    best = 0
    for path in root.rglob("checkpoint_bundles_v2/**/population.json"):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and isinstance(data.get("agents"), list):
            best = max(best, len(data["agents"]))
    return best


def _gv2_projected_disappearance_count(snapshots: list[Any], events: list[dict[str, int]]) -> int:
    if not snapshots:
        return 0
    birth_ticks = {agent_id: 0 for agent_id in snapshots[0].alive_agent_ids}
    for event in events:
        if event["event_type"] == 1 and event["success"] == 1 and event["child_agent_id"] >= 0:
            birth_ticks.setdefault(event["child_agent_id"], event["sequence"] + 1)
    death_ticks = {
        event["target_agent_id"]: event["sequence"] + 1
        for event in events
        if event["event_type"] == 3 and event["success"] == 1 and event["target_agent_id"] >= 0
    }
    projected = 0
    projected_seen: set[int] = set()
    for snapshot in snapshots[1:]:
        boundary = snapshot.event_boundary_tick + 1
        current_alive = set(snapshot.alive_agent_ids)
        expected_alive = {
            agent_id
            for agent_id, birth_tick in birth_ticks.items()
            if birth_tick <= boundary and (death_ticks.get(agent_id) is None or death_ticks[agent_id] > boundary)
        }
        for missing in expected_alive - current_alive:
            if missing not in projected_seen:
                projected_seen.add(missing)
                projected += 1
    return projected


def _gv2_unknown_exact_death_targets(snapshots: list[Any], events: list[dict[str, int]]) -> list[int]:
    if not snapshots:
        return []
    known = set(snapshots[0].alive_agent_ids)
    for event in events:
        if event["event_type"] == 1 and event["success"] == 1 and event["child_agent_id"] >= 0:
            known.add(event["child_agent_id"])
    return sorted(
        {
            event["target_agent_id"]
            for event in events
            if event["event_type"] == 3
            and event["success"] == 1
            and event["target_agent_id"] >= 0
            and event["target_agent_id"] not in known
        }
    )


def audit_source(root: Path) -> SourceCandidate:
    root = root.resolve()
    projection_paths = projection_paths_for_cell(root)
    if projection_paths:
        try:
            cell_id = choose_cell(root)
            cell_paths = projection_paths_for_cell(root, cell_id)
            snapshots = [snapshot_from_projection(path) for path in cell_paths]
            final = snapshots[-1]
            events = typed_lifecycle_events(final.projection)
            event_names = {
                "agent_ids_by_alive_order",
                "slot_to_agent_id",
                "alive_mask",
                "lineage_founder_id",
                "lineage_depth",
                "lineage_birth_generation",
                "lineage_birth_tick",
                "typed_lifecycle_event_parent_agent_id",
                "typed_lifecycle_event_child_agent_id",
                "typed_lifecycle_event_generation",
                "typed_lifecycle_event_tick",
                "typed_lifecycle_event_sequence",
            }
            death_events = [event for event in events if event["event_type"] == 3 and event["success"] == 1]
            unknown_deaths = _gv2_unknown_exact_death_targets(snapshots, events)
            if unknown_deaths:
                return SourceCandidate(
                    source_root=str(root),
                    classification="EXP-D",
                    semantics_mode="insufficient",
                    confidence="high",
                    supporting_files=[str(path) for path in cell_paths],
                    available_fields=sorted(event_names),
                    missing_fields=["consistent exact death target identity coverage"],
                    timing_resolution="blocked",
                    identity_continuity="blocked by exact death target outside known initial/birth identities",
                    parent_edge_status="typed lifecycle parent_agent_id for successful birth events",
                    lineage_authority_status="lineage_founder_id and lineage_depth from production projection tensors",
                    death_coverage=f"identity conflict for death targets {unknown_deaths[:8]}",
                    branch_semantics="lineage_root_projection",
                    coverage_interval=f"G0..G{final.generation}; {len(events)} typed lifecycle events",
                    restrictions=["Exact death records reference identities absent from initial checkpoint and successful births."],
                )
            projected_disappearances = _gv2_projected_disappearance_count(snapshots, events)
            classification = "EXP-A" if projected_disappearances == 0 else "EXP-B"
            death_coverage = (
                "complete exact coverage across checkpoint boundaries"
                if classification == "EXP-A"
                else "mixed exact death events and checkpoint-boundary disappearance projections"
                if death_events
                else "checkpoint-boundary disappearance only; typed death events are not successful in selected projection"
            )
            missing = [] if classification == "EXP-A" else ["authoritative exact death tick for every checkpoint-disappeared agent"]
            return SourceCandidate(
                source_root=str(root),
                classification=classification,
                semantics_mode="actual_agents" if classification == "EXP-A" else "partial_agents",
                confidence="high",
                supporting_files=[str(path) for path in cell_paths],
                available_fields=sorted(event_names),
                missing_fields=missing,
                timing_resolution="event sequence with generation/tick fields; checkpoint-boundary intervals for disappearance when exact death events are absent",
                identity_continuity=f"stable agent ids for selected cell {cell_id}",
                parent_edge_status="typed lifecycle parent_agent_id for successful birth events",
                lineage_authority_status="lineage_founder_id and lineage_depth from production projection tensors",
                death_coverage=death_coverage,
                branch_semantics="lineage_root_projection",
                coverage_interval=f"G0..G{final.generation}; {len(events)} typed lifecycle events",
                restrictions=[
                    "Exporter reads one selected cell from the run root in V0.",
                    "When exact death events are absent, deaths are rendered as checkpoint-boundary disappearance projections.",
                    "Branch lobes are neutral lineage-root display projections, not source-authored descendant branch ids.",
                ],
            )
        except Exception as exc:
            return SourceCandidate(
                source_root=str(root),
                classification="EXP-D",
                semantics_mode="insufficient",
                confidence="high",
                supporting_files=[str(path) for path in projection_paths[:8]],
                available_fields=["checkpoint_bundles_v2/native_runtime_checkpoint_projection.json.gz"],
                missing_fields=["decodable checkpoint projection"],
                timing_resolution="blocked",
                identity_continuity="blocked",
                parent_edge_status="blocked",
                lineage_authority_status="blocked",
                death_coverage="blocked",
                branch_semantics="blocked",
                coverage_interval="blocked",
                restrictions=[f"Checkpoint projection decode failed: {exc}"],
            )

    birth_events = sorted(root.rglob("p12b_science_birth_events.csv"))
    settlement = sorted(root.rglob("p12b_science_birth_settlement.csv"))
    campaign_manifest = next(root.rglob("campaign_manifest.json"), None)
    checkpoint_count = _checkpoint_population_count(root)
    has_lineage_metrics = _lineage_metrics_nonempty(root)

    if birth_events:
        fields = _csv_fields(birth_events[0])
        missing = sorted(P12B_EVENT_FIELDS - fields)
        files = [str(birth_events[0])]
        if settlement:
            files.append(str(settlement[0]))
        if campaign_manifest:
            files.append(str(campaign_manifest))
        if missing:
            return SourceCandidate(
                source_root=str(root),
                classification="EXP-D",
                semantics_mode="insufficient",
                confidence="high",
                supporting_files=files,
                available_fields=sorted(fields),
                missing_fields=missing,
                timing_resolution="unsupported",
                identity_continuity="birth table lacks required identity fields",
                parent_edge_status="unusable",
                lineage_authority_status="unusable",
                death_coverage="unusable",
                branch_semantics="unavailable",
                coverage_interval="unknown",
                restrictions=["Required P12B birth-event columns are absent."],
            )
        death_coverage = "terminal settlement retained/lost flags" if settlement else "absent"
        missing_fields = [
            "complete initial population identities",
            "authoritative per-agent death tick",
            "complete lifecycle coverage for non-birth-receipt agents",
            "authoritative descendant branch ids",
        ]
        if not settlement:
            missing_fields.append("terminal settlement retained/lost table")
        return SourceCandidate(
            source_root=str(root),
            classification="EXP-B",
            semantics_mode="partial_agents",
            confidence="high",
            supporting_files=files,
            available_fields=sorted(fields | ({"retained", "retention_relation"} if settlement else set())),
            missing_fields=missing_fields,
            timing_resolution="event_sequence plus generation; terminal losses are interval-censored",
            identity_continuity="stable child ids within cell-scoped birth receipts",
            parent_edge_status="parent and mate_parent ids recorded; some parents are external to exported subset",
            lineage_authority_status="lineage_founder_id recorded in birth receipts",
            death_coverage=death_coverage,
            branch_semantics="source_group_projection",
            coverage_interval="birth-event rows plus terminal settlement boundary where present",
            restrictions=[
                "Particles are captured birth-event children only.",
                "Death timing is not exact; lost terminal particles receive a documented boundary projection.",
                "Particle count is not current living production population.",
            ],
        )

    if checkpoint_count > 0:
        return SourceCandidate(
            source_root=str(root),
            classification="EXP-B",
            semantics_mode="partial_agents",
            confidence="medium",
            supporting_files=[str(path) for path in sorted(root.rglob("checkpoint_bundles_v2/**/population.json"))[:8]],
            available_fields=["checkpoint population agents", "generation boundary"],
            missing_fields=["birth events", "death events", "parent relationships"],
            timing_resolution="checkpoint boundary",
            identity_continuity="checkpoint-local identities require source-specific reader",
            parent_edge_status="absent",
            lineage_authority_status="unknown",
            death_coverage="generation-boundary absence only",
            branch_semantics="unavailable",
            coverage_interval=f"sampled checkpoint population up to {checkpoint_count} agents",
            restrictions=["Requires a source-specific checkpoint reader before export."],
        )

    if has_lineage_metrics:
        return SourceCandidate(
            source_root=str(root),
            classification="EXP-C",
            semantics_mode="representative_particles",
            confidence="medium",
            supporting_files=[str(path) for path in sorted(root.rglob("metrics/lineage_events.jsonl"))[:8]],
            available_fields=["lineage metrics/events"],
            missing_fields=["agent identities", "birth events", "death events", "parent relationships"],
            timing_resolution="generation or metrics event",
            identity_continuity="aggregate lineage-only",
            parent_edge_status="absent",
            lineage_authority_status="aggregate metric records only",
            death_coverage="aggregate only",
            branch_semantics="unavailable",
            coverage_interval="metrics file coverage",
            restrictions=["Representative playback only; no individual lifecycle export."],
        )

    manifests = sorted(root.rglob("campaign_manifest.json"))[:8]
    keys = sorted(_json_keys(manifests[0])) if manifests else []
    return SourceCandidate(
        source_root=str(root),
        classification="EXP-D",
        semantics_mode="insufficient",
        confidence="medium",
        supporting_files=[str(path) for path in manifests],
        available_fields=keys,
        missing_fields=["agent identities", "lineage identities", "birth timing", "death timing"],
        timing_resolution="none",
        identity_continuity="unavailable",
        parent_edge_status="unavailable",
        lineage_authority_status="unavailable",
        death_coverage="unavailable",
        branch_semantics="unavailable",
        coverage_interval="none",
        restrictions=["No load-bearing lifecycle source found by the exporter audit."],
    )


def audit_many(roots: list[Path]) -> dict[str, Any]:
    candidates = [asdict(audit_source(root)) for root in roots]
    order = {"EXP-A": 0, "EXP-B": 1, "EXP-C": 2, "EXP-D": 3}
    candidates.sort(key=lambda item: (order.get(item["classification"], 9), item["source_root"]))
    return {
        "schema_version": "eiel_lineage_source_audit_v1",
        "candidates": candidates,
        "selected_source": candidates[0] if candidates else None,
    }
