from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION
from .artifact_readers.checkpoint_bundle_v2 import (
    ProjectionSnapshot,
    choose_cell,
    projection_paths_for_cell,
    snapshot_boundary,
    snapshot_from_projection,
    snapshot_order_key,
    typed_lifecycle_events,
)
from .provenance import NO_CLAIMS_STATEMENT, exporter_git_commit, output_hash, source_file_manifest
from .schema import validate_playback
from .source_audit import audit_source


def _read_settlement(path: Path | None) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    if path is None or not path.exists():
        return rows
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["cell_id"], row["terminal_agent_id"])] = row
    return rows


def export_p12b_partial(
    source: Path,
    *,
    limit_rows: int | None = None,
    generated_at: str | None = None,
    command: str = "",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_supplied = str(source)
    source = source.resolve()
    audit = audit_source(source)
    if audit.classification != "EXP-B" or "p12b_science_birth_events.csv" not in " ".join(audit.supporting_files):
        raise ValueError(f"unsupported source for export: {source} classified {audit.classification}")
    events_path = next(source.rglob("p12b_science_birth_events.csv"))
    settlement_path = next(source.rglob("p12b_science_birth_settlement.csv"), None)
    settlement = _read_settlement(settlement_path)

    raw_rows: list[dict[str, str]] = []
    with events_path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit_rows is not None and index >= limit_rows:
                break
            raw_rows.append(row)
    if not raw_rows:
        raise ValueError(f"no birth rows found in {events_path}")

    event_ticks = [int(row["event_sequence"]) for row in raw_rows]
    first_tick = min(event_ticks)
    terminal_tick = max(event_ticks) + 1
    lineages: dict[str, dict[str, Any]] = {}
    branches: dict[str, dict[str, Any]] = {}
    agents: list[dict[str, Any]] = []
    seen: set[str] = set()
    retained_count = 0
    projected_death_count = 0

    for row in sorted(raw_rows, key=lambda item: (int(item["event_sequence"]), item["cell_id"], int(item["child_id"]))):
        founder = row["lineage_founder_id"]
        cell = row["cell_id"]
        lineage_id = f"p12b_founder_{founder}"
        branch_id = f"{lineage_id}_{cell[-6:]}"
        child_id = row["child_id"]
        agent_id = f"p12b_{cell}_{child_id}"
        if agent_id in seen:
            raise ValueError(f"duplicate exported agent id: {agent_id}")
        seen.add(agent_id)
        tick = int(row["event_sequence"])
        lineages.setdefault(
            lineage_id,
            {
                "lineage_id": lineage_id,
                "parent_lineage_id": None,
                "founder_lineage_id": lineage_id,
                "creation_tick": first_tick,
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
                "creation_tick": first_tick,
                "extinction_tick": None,
                "display_label": cell,
                "branch_semantics": "source_group_projection",
            },
        )
        parents = []
        for field in ("parent_id", "mate_parent_id"):
            value = row.get(field)
            if value not in {None, "", "-1"}:
                parents.append(f"p12b_external_{cell}_{value}")
        status = settlement.get((cell, child_id))
        retained = None if status is None else status.get("retained") == "True"
        death_tick = None
        death_interval = None
        death_tick_source = "not_recorded"
        if retained is False:
            death_tick = terminal_tick
            death_interval = [tick, terminal_tick]
            death_tick_source = "terminal_boundary_projection"
            projected_death_count += 1
        elif retained is True:
            retained_count += 1
        agents.append(
            {
                "agent_id": agent_id,
                "lineage_id": lineage_id,
                "branch_id": branch_id,
                "birth_tick": tick,
                "death_tick": death_tick,
                "parent_agent_ids": parents[:2],
                "generation_born": int(row["generation"]),
                "generation_died": None,
                "death_interval": death_interval,
                "death_tick_source": death_tick_source,
                "source_cell_id": cell,
                "source_child_id": child_id,
                "source_transaction_id": row.get("transaction_id"),
                "source_event_receipt_sha256": row.get("event_receipt_sha256"),
                "terminal_retained": retained,
                "retention_relation": status.get("retention_relation") if status else None,
            }
        )

    playback = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "p12b_partial_real_export",
            "source_kind": "real_export",
            "source_path": str(source),
            "source_classification": audit.classification,
            "semantics_mode": "partial_agents",
            "time_unit": "event_sequence",
            "time_semantics": "birth events use event_sequence; terminal losses use interval-censored terminal boundary projection",
            "first_tick": first_tick,
            "last_tick": terminal_tick,
            "generation_mapping": None,
            "parent_reference_policy": "allow_missing_external",
            "branch_semantics": "source_group_projection",
            "visualization_seed": 9713,
            "generated_at": generated_at,
            "birth_record_semantics": "birth-event children only",
            "death_record_semantics": "terminal or unavailable death timing",
            "initial_population_semantics": "incomplete initial population",
            "population_count_semantics": "recorded particle count, not living population",
            "share_semantics": "recorded particle share over captured birth-event children",
            "display_semantics": {
                "metric_label": "Recorded particles",
                "share_label": "Recorded share",
                "notice_title": "Partial Real Adapter",
                "notice_summary": "Birth receipts only. Death timing and the complete initial population are unavailable. Particle count is not current living population.",
            },
            "notes": "EXP-B partial real export. Particles are captured birth-event children only; death timing and the complete initial population are unavailable.",
            "warnings": [
                "Partial real export: particle count is not current living production population.",
                "Terminal losses are interval-censored and rendered with a documented terminal boundary projection.",
                "Branch lobes are source cell group projections, not authoritative evolved branches.",
                NO_CLAIMS_STATEMENT,
            ],
        },
        "lineages": sorted(lineages.values(), key=lambda item: item["lineage_id"]),
        "branches": sorted(branches.values(), key=lambda item: item["branch_id"]),
        "agents": sorted(agents, key=lambda item: (item["birth_tick"], item["agent_id"])),
    }
    validate_playback(playback)

    source_files = [events_path]
    if settlement_path:
        source_files.append(settlement_path)
    manifest = {
        "schema_version": "eiel_lineage_playback_export_provenance_v1",
        "exporter_schema_version": "eiel_lineage_playback_exporter_v0",
        "exporter_git_commit": exporter_git_commit(),
        "playback_schema_version": SCHEMA_VERSION,
        "dataset_id": playback["metadata"]["dataset_id"],
        "source_root": str(source),
        "source_root_supplied": source_supplied,
        "source_root_resolved": str(source),
        "source_files": source_file_manifest(source_files),
        "source_run_id": source.name,
        "source_cell_id": None,
        "source_classification": audit.classification,
        "semantics_mode": "partial_agents",
        "source_code_commit": None,
        "production_provider_identity": None,
        "production_runtime_identity": None,
        "source_schema_versions": ["p12b_science_birth_events.csv"],
        "campaign_manifest_identity": None,
        "unavailable_fields": [
            "source_code_commit",
            "production_provider_identity",
            "production_runtime_identity",
            "complete initial population identities",
            "authoritative per-agent death tick",
        ],
        "missing_fields": audit.missing_fields,
        "transformations_applied": [
            "cell-scoped child ids normalized into playback agent ids",
            "lineage_founder_id normalized into lineage ids",
            "cell id normalized into display branch ids",
        ],
        "projections_applied": [
            "terminal settlement losses rendered as interval-censored terminal boundary deaths",
            "source cell grouping rendered as display branch projection",
        ],
        "output_sha256": output_hash(playback),
        "command": command,
        "generated_at": generated_at,
        "warnings": playback["metadata"]["warnings"],
        "no_claims_statement": NO_CLAIMS_STATEMENT,
    }
    summary = {
        "schema_version": "eiel_lineage_playback_export_summary_v1",
        "source_root": str(source),
        "classification": audit.classification,
        "semantics_mode": "partial_agents",
        "agent_count": len(agents),
        "lineage_count": len(lineages),
        "branch_count": len(branches),
        "first_tick": first_tick,
        "last_tick": terminal_tick,
        "retained_terminal_count": retained_count,
        "projected_terminal_death_count": projected_death_count,
        "parent_edge_count": sum(len(agent["parent_agent_ids"]) for agent in agents),
        "output_sha256": manifest["output_sha256"],
    }
    return playback, manifest, summary


def _lineage_id(founder: int) -> str:
    return f"gv2_founder_{founder}"


def _agent_id(agent_id: int) -> str:
    return f"gv2_agent_{agent_id}"


def _checkpoint_boundary(snapshot: Any) -> int:
    """Use event sequence when retained, otherwise the exact checkpoint generation."""

    return snapshot_boundary(snapshot)


def export_gv2_checkpoint_run(
    source: Path,
    *,
    cell_id: str | None = None,
    generated_at: str | None = None,
    command: str = "",
    snapshots: Sequence[ProjectionSnapshot] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_supplied = str(source)
    source = source.resolve()
    selected_cell = cell_id or choose_cell(source)
    paths = (
        projection_paths_for_cell(source, selected_cell)
        if snapshots is None
        else [snapshot.path for snapshot in snapshots]
    )
    if not paths:
        raise ValueError(f"no checkpoint projections for cell {selected_cell} under {source}")
    ordered_snapshots = sorted(
        [snapshot_from_projection(path) for path in paths]
        if snapshots is None
        else snapshots,
        key=snapshot_order_key,
    )
    initial = ordered_snapshots[0]
    final = ordered_snapshots[-1]
    tick_native = all(snapshot.absolute_tick is not None for snapshot in ordered_snapshots)
    if any(snapshot.absolute_tick is not None for snapshot in ordered_snapshots) and not tick_native:
        raise ValueError("mixed generation- and tick-named checkpoint histories are invalid")
    initial_boundary = _checkpoint_boundary(initial) if tick_native else 0
    ticks_per_generation = final.ticks_per_generation if tick_native else None
    events = typed_lifecycle_events(final.projection)
    successful_births = [event for event in events if event["event_type"] == 1 and event["success"] == 1 and event["child_agent_id"] >= 0]
    successful_deaths = [event for event in events if event["event_type"] == 3 and event["success"] == 1 and event["target_agent_id"] >= 0]

    lineages: dict[str, dict[str, Any]] = {}
    branches: dict[str, dict[str, Any]] = {}
    agents: dict[int, dict[str, Any]] = {}

    def ensure_lineage(founder: int) -> tuple[str, str]:
        lineage_id = _lineage_id(founder)
        branch_id = f"{lineage_id}_root"
        lineages.setdefault(
            lineage_id,
            {
                "lineage_id": lineage_id,
                "parent_lineage_id": None,
                "founder_lineage_id": lineage_id,
                "creation_tick": initial_boundary,
                "extinction_tick": None,
                "display_label": f"GV2 founder {founder}",
            },
        )
        branches.setdefault(
            branch_id,
            {
                "branch_id": branch_id,
                "lineage_id": lineage_id,
                "parent_branch_id": None,
                "creation_tick": initial_boundary,
                "extinction_tick": None,
                "display_label": f"Founder {founder} root",
                "branch_semantics": "lineage_root_projection",
            },
        )
        return lineage_id, branch_id

    for agent_id in initial.alive_agent_ids:
        founder = initial.lineage_by_agent_id.get(agent_id, agent_id)
        lineage_id, branch_id = ensure_lineage(founder)
        agents[agent_id] = {
            "agent_id": _agent_id(agent_id),
            "lineage_id": lineage_id,
            "branch_id": branch_id,
            "birth_tick": initial_boundary,
            "death_tick": None,
            "parent_agent_ids": [],
            "generation_born": 0,
            "generation_died": None,
            "source_agent_id": agent_id,
            "source_birth_kind": "initial_population_snapshot",
            "source_lineage_depth": initial.depth_by_agent_id.get(agent_id, 0),
        }

    for event in sorted(successful_births, key=lambda item: (item["sequence"], item["child_agent_id"])):
        child = event["child_agent_id"]
        if child in agents:
            if agents[child].get("source_birth_kind") == "initial_population_snapshot":
                continue
            raise ValueError(f"duplicate GV2 birth for agent {child}")
        event_boundary = (
            event["generation"] * ticks_per_generation + event["tick"]
            if ticks_per_generation is not None
            else event["sequence"] + 1
        )
        # A tick-native relay can begin at a late first checkpoint.  Retained
        # lifecycle history may then contain agents born and gone before the
        # visible interval.  They are source history, not playback members;
        # adding them would falsely keep them alive at the first checkpoint.
        if tick_native and event_boundary < initial_boundary:
            continue
        lineage_id, branch_id = ensure_lineage(event["lineage_founder_id"])
        parents = []
        if event["parent_agent_id"] >= 0:
            parents.append(_agent_id(event["parent_agent_id"]))
        agents[child] = {
            "agent_id": _agent_id(child),
            "lineage_id": lineage_id,
            "branch_id": branch_id,
            "birth_tick": event_boundary,
            "death_tick": None,
            "parent_agent_ids": parents,
            "generation_born": event["generation"],
            "generation_died": None,
            "source_agent_id": child,
            "source_birth_kind": "typed_lifecycle_birth_event",
            "source_event_sequence": event["sequence"],
            "source_event_generation": event["generation"],
            "source_event_tick": event["tick"],
            "source_lineage_depth": event["lineage_depth"],
        }

    # Some checkpoint-bundle-v2 families retain exact membership identities at
    # each boundary but no lifecycle-event history.  A new identity can then
    # be represented truthfully only as *first observed at this checkpoint*,
    # never as an inferred birth or parentage.  This preserves the strict
    # membership equation while keeping the output EXP-B/partial.
    death_events_by_agent = {event["target_agent_id"]: event for event in successful_deaths}
    for agent_id, event in death_events_by_agent.items():
        if agent_id not in agents:
            event_boundary = (
                event["generation"] * ticks_per_generation + event["tick"]
                if ticks_per_generation is not None
                else event["sequence"] + 1
            )
            if tick_native and event_boundary < initial_boundary:
                continue
            raise ValueError(f"exact death event references unknown GV2 agent {agent_id}")
        agents[agent_id]["death_tick"] = (
            event["generation"] * ticks_per_generation + event["tick"]
            if ticks_per_generation is not None
            else event["sequence"] + 1
        )
        agents[agent_id]["generation_died"] = event["generation"]
        agents[agent_id]["death_tick_source"] = "typed_lifecycle_death_event"
        agents[agent_id]["source_death_event_sequence"] = event["sequence"]

    previous_alive = set(initial.alive_agent_ids)
    previous_boundary = initial_boundary
    projected_deaths = 0
    checkpoint_observed_identity_count = 0
    for snapshot in ordered_snapshots[1:]:
        current_alive = set(snapshot.alive_agent_ids)
        boundary = _checkpoint_boundary(snapshot)
        for source_id in sorted(current_alive):
            existing = agents.get(source_id)
            if existing is not None:
                if existing.get("death_tick") is not None:
                    raise ValueError(f"GV2 identity {source_id} reappeared after its recorded death")
                continue
            founder = snapshot.lineage_by_agent_id.get(source_id)
            if founder is None or founder < 0:
                raise ValueError(f"checkpoint-observed GV2 identity {source_id} has no lineage founder")
            lineage_id, branch_id = ensure_lineage(founder)
            agents[source_id] = {
                "agent_id": _agent_id(source_id),
                "lineage_id": lineage_id,
                "branch_id": branch_id,
                "birth_tick": boundary,
                "death_tick": None,
                "parent_agent_ids": [],
                "generation_born": snapshot.generation,
                "generation_died": None,
                "source_agent_id": source_id,
                "source_birth_kind": "checkpoint_first_observed_identity",
                "source_first_observed_generation": snapshot.generation,
                "source_first_observed_boundary": boundary,
                "source_lineage_depth": snapshot.depth_by_agent_id.get(source_id, 0),
            }
            checkpoint_observed_identity_count += 1
        born_before_boundary = {
            source_id
            for source_id, record in agents.items()
            if record["birth_tick"] <= boundary and record.get("death_tick") is None
        }
        for disappeared in sorted(born_before_boundary - current_alive):
            record = agents.get(disappeared)
            if record and record.get("death_tick") is None:
                record["death_tick"] = boundary
                record["generation_died"] = snapshot.generation
                record["death_interval"] = [previous_boundary, boundary]
                record["death_tick_source"] = "checkpoint_boundary_projection"
                projected_deaths += 1
        previous_alive = current_alive
        previous_boundary = boundary

    final_alive = set(final.alive_agent_ids)
    reconstructed_alive = {
        source_id
        for source_id, record in agents.items()
        if record["birth_tick"] <= previous_boundary and (record.get("death_tick") is None or record["death_tick"] > previous_boundary)
    }
    if reconstructed_alive != final_alive:
        missing = sorted(final_alive - reconstructed_alive)[:8]
        extra = sorted(reconstructed_alive - final_alive)[:8]
        raise ValueError(f"checkpoint reconstruction mismatch missing={missing} extra={extra}")

    classification = "EXP-A" if projected_deaths == 0 and checkpoint_observed_identity_count == 0 else "EXP-B"
    semantics_mode = "actual_agents" if classification == "EXP-A" else "partial_agents"
    warnings = [
        "Real GV2 checkpoint export from one selected cell; it is not a multi-cell campaign aggregate.",
        "Branch lobes are neutral lineage-root projections because source descendant branch ids are unavailable.",
        NO_CLAIMS_STATEMENT,
    ]
    parent_reference_policy = (
        "allow_missing_external"
        if initial.generation > 0 or initial_boundary > 0
        else "strict"
    )
    if parent_reference_policy == "allow_missing_external":
        warnings.insert(
            1,
            "The export starts from the first available checkpoint, so some typed birth parent ids may refer to pre-window agents unavailable in the playback.",
        )
    if semantics_mode != "actual_agents":
        warnings.insert(
            1,
            "Exact successful death events are unavailable in the selected projection; deaths are checkpoint-boundary disappearance projections.",
        )
    playback = {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": "gv2_r1_checkpoint_lineage_export",
            "source_kind": "real_export",
            "source_path": str(source),
            "source_cell_id": selected_cell,
            "source_classification": classification,
            "semantics_mode": semantics_mode,
            "time_unit": "absolute_tick" if tick_native else "event_sequence",
            "time_semantics": (
                "typed events use recorded absolute generation/tick coordinates; "
                "projected deaths use exact checkpoint next-tick boundaries"
                if tick_native
                else "births use typed lifecycle event sequence; projected deaths use checkpoint boundary sequence"
            ),
            "first_tick": initial_boundary,
            "last_tick": previous_boundary,
            "generation_mapping": None,
            "parent_reference_policy": parent_reference_policy,
            "branch_semantics": "lineage_root_projection",
            "visualization_seed": 10403,
            "generated_at": generated_at,
            "birth_record_semantics": "authoritative typed lifecycle birth events; otherwise first observed checkpoint identity",
            "death_record_semantics": "checkpoint-boundary disappearance projection with intervals"
            if semantics_mode != "actual_agents"
            else "authoritative typed lifecycle death events with complete coverage",
            "initial_population_semantics": f"complete selected-cell first exported checkpoint identities at G{initial.generation}",
            "population_count_semantics": "projected active particles between checkpoint boundaries; exact at represented checkpoint boundaries after source cross-check"
            if semantics_mode != "actual_agents"
            else "actual active agent identities across exact lifecycle records",
            "share_semantics": "projected active lineage share between checkpoint boundaries"
            if semantics_mode != "actual_agents"
            else "actual active lineage share",
            "display_semantics": {
                "metric_label": "Living" if semantics_mode == "actual_agents" else "Projected active particles",
                "share_label": "Largest share" if semantics_mode == "actual_agents" else "Projected active share",
                "notice_title": "Checkpoint-Projected Real Export",
                "notice_summary": (
                    "Initial checkpoint identities and typed birth events are available; exact successful death events are unavailable "
                    "for checkpoint-disappeared agents, so active counts use interval-censored checkpoint-boundary projections."
                )
                if semantics_mode != "actual_agents"
                else "Initial checkpoint identities, typed births, and complete exact death coverage are available for the exported interval.",
            },
            "notes": "Offline read-only export from GV2 checkpoint bundle v2 production projection tensors.",
            "warnings": warnings,
        },
        "lineages": sorted(lineages.values(), key=lambda item: item["lineage_id"]),
        "branches": sorted(branches.values(), key=lambda item: item["branch_id"]),
        "agents": sorted(agents.values(), key=lambda item: (item["birth_tick"], item["agent_id"])),
    }
    validate_playback(playback)
    source_files = [*paths]
    manifest = {
        "schema_version": "eiel_lineage_playback_export_provenance_v1",
        "exporter_schema_version": "eiel_lineage_playback_exporter_v0",
        "playback_schema_version": SCHEMA_VERSION,
        "dataset_id": playback["metadata"]["dataset_id"],
        "source_root": str(source),
        "source_root_supplied": source_supplied,
        "source_root_resolved": str(source),
        "source_files": source_file_manifest(source_files),
        "source_run_id": source.name,
        "source_cell_id": selected_cell,
        "source_classification": classification,
        "semantics_mode": semantics_mode,
        "exporter_git_commit": exporter_git_commit(),
        "source_code_commit": None,
        "production_provider_identity": "eiel-production-provider-v1",
        "production_runtime_identity": "eiel-production-runtime-v1",
        "source_schema_versions": ["native_runtime_checkpoint_projection_v1", "cxxbackbone10_checkpoint_projection_v2"],
        "campaign_manifest_identity": None,
        "unavailable_fields": [
            "source_code_commit",
            "campaign_manifest_identity",
            *(["complete pre-window parent identities"] if parent_reference_policy == "allow_missing_external" else []),
            *([] if semantics_mode == "actual_agents" else ["authoritative exact death tick for checkpoint-disappeared agents"]),
        ],
        "missing_fields": [] if semantics_mode == "actual_agents" else ["authoritative exact death tick for checkpoint-disappeared agents"],
        "transformations_applied": [
            "first exported checkpoint alive ids normalized into initial playback agents",
            "pre-window birth rows for agents already alive in the first exported checkpoint are retained as source history but not duplicated as playback births",
            "typed lifecycle birth events normalized into playback births",
            "checkpoint-only identity appearances normalized as first-observed identities without inferred parentage",
            "lineage_founder_id normalized into lineage ids",
        ],
        "projections_applied": [
            "lineage roots rendered as neutral display branches",
            "checkpoint disappearance rendered as boundary death projection" if semantics_mode != "actual_agents" else "none",
        ],
        "output_sha256": output_hash(playback),
        "command": command,
        "generated_at": generated_at,
        "warnings": warnings,
        "no_claims_statement": NO_CLAIMS_STATEMENT,
    }
    summary = {
        "schema_version": "eiel_lineage_playback_export_summary_v1",
        "source_root": str(source),
        "source_cell_id": selected_cell,
        "classification": classification,
        "semantics_mode": semantics_mode,
        "agent_count": len(playback["agents"]),
        "lineage_count": len(playback["lineages"]),
        "branch_count": len(playback["branches"]),
        "first_tick": initial_boundary,
        "last_tick": previous_boundary,
        "actual_birth_event_count": len(successful_births),
        "exported_typed_birth_event_count": sum(1 for agent in playback["agents"] if agent.get("source_birth_kind") == "typed_lifecycle_birth_event"),
        "checkpoint_first_observed_identity_count": checkpoint_observed_identity_count,
        "pre_window_birth_event_count": len(successful_births)
        - sum(1 for agent in playback["agents"] if agent.get("source_birth_kind") == "typed_lifecycle_birth_event"),
        "actual_death_event_count": len(successful_deaths),
        "checkpoint_projected_death_count": projected_deaths,
        "final_alive_count": len(final_alive),
        "parent_edge_count": sum(len(agent["parent_agent_ids"]) for agent in playback["agents"]),
        "output_sha256": manifest["output_sha256"],
    }
    return playback, manifest, summary


def cross_check_p12b(source: Path, playback: dict[str, Any]) -> dict[str, Any]:
    source = source.resolve()
    events_path = next(source.rglob("p12b_science_birth_events.csv"))
    settlement_path = next(source.rglob("p12b_science_birth_settlement.csv"), None)
    with events_path.open(newline="") as handle:
        event_rows = list(csv.DictReader(handle))
    settlement_rows: list[dict[str, str]] = []
    if settlement_path:
        with settlement_path.open(newline="") as handle:
            settlement_rows = list(csv.DictReader(handle))
    exported_agents = playback["agents"]
    source_child_keys = {(row["cell_id"], row["child_id"]) for row in event_rows}
    exported_child_keys = {(agent["source_cell_id"], agent["source_child_id"]) for agent in exported_agents}
    mismatches = []
    if source_child_keys != exported_child_keys:
        mismatches.append("exported child key set differs from source birth-event child key set")
    projected_deaths = sum(1 for agent in exported_agents if agent.get("death_tick_source") == "terminal_boundary_projection")
    terminal_losses = sum(1 for row in settlement_rows if row.get("retained") == "False")
    if settlement_rows and projected_deaths != terminal_losses:
        mismatches.append("projected terminal death count differs from settlement terminal losses")
    return {
        "schema_version": "eiel_lineage_playback_cross_check_v1",
        "source_birth_rows": len(event_rows),
        "exported_agent_count": len(exported_agents),
        "source_lineage_count": len({row["lineage_founder_id"] for row in event_rows}),
        "exported_lineage_count": len(playback["lineages"]),
        "source_parent_edge_count": sum(1 for row in event_rows for key in ("parent_id", "mate_parent_id") if row.get(key) not in {None, "", "-1"}),
        "exported_parent_edge_count": sum(len(agent["parent_agent_ids"]) for agent in exported_agents),
        "source_terminal_loss_count": terminal_losses,
        "exported_projected_death_count": projected_deaths,
        "coverage_interval": [playback["metadata"]["first_tick"], playback["metadata"]["last_tick"]],
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def cross_check_gv2(
    source: Path,
    playback: dict[str, Any],
    *,
    cell_id: str | None = None,
    snapshots: Sequence[ProjectionSnapshot] | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    selected_cell = cell_id or playback["metadata"].get("source_cell_id") or choose_cell(source)
    ordered_snapshots = sorted(
        [snapshot_from_projection(path) for path in projection_paths_for_cell(source, selected_cell)]
        if snapshots is None
        else snapshots,
        key=snapshot_order_key,
    )
    final = ordered_snapshots[-1]
    final_alive = set(final.alive_agent_ids)
    last_tick = playback["metadata"]["last_tick"]
    exported_final_alive = {
        int(agent["source_agent_id"])
        for agent in playback["agents"]
        if agent["birth_tick"] <= last_tick and (agent.get("death_tick") is None or agent["death_tick"] > last_tick)
    }
    events = typed_lifecycle_events(final.projection)
    births = [event for event in events if event["event_type"] == 1 and event["success"] == 1]
    mismatches = []
    agents_by_source_id = {int(agent["source_agent_id"]): agent for agent in playback["agents"]}
    per_boundary_checks = []
    exported_typed_births = sum(1 for agent in playback["agents"] if agent.get("source_birth_kind") == "typed_lifecycle_birth_event")
    checkpoint_first_observed = sum(1 for agent in playback["agents"] if agent.get("source_birth_kind") == "checkpoint_first_observed_identity")
    if len(playback["agents"]) != exported_typed_births + checkpoint_first_observed + len(ordered_snapshots[0].alive_agent_ids):
        mismatches.append("exported agent count differs from initial alive plus typed births plus checkpoint first-observed identities")
    if exported_final_alive != final_alive:
        mismatches.append("exported final alive set differs from final checkpoint alive set")
    for index, snapshot in enumerate(ordered_snapshots):
        boundary = (
            0
            if index == 0 and snapshot.absolute_tick is None
            else _checkpoint_boundary(snapshot)
        )
        source_alive = set(snapshot.alive_agent_ids)
        exported_alive = {
            source_id
            for source_id, agent in agents_by_source_id.items()
            if agent["birth_tick"] <= boundary and (agent.get("death_tick") is None or agent["death_tick"] > boundary)
        }
        source_lineage_count = len(set(snapshot.lineage_by_agent_id.values()))
        exported_lineage_count = len({agents_by_source_id[source_id]["lineage_id"] for source_id in exported_alive})
        projected_disappearances = sum(
            1
            for agent in playback["agents"]
            if agent.get("death_tick_source") == "checkpoint_boundary_projection" and agent.get("death_tick") is not None and agent["death_tick"] <= boundary
        )
        cumulative_births = sum(1 for agent in playback["agents"] if agent.get("source_birth_kind") == "typed_lifecycle_birth_event" and agent["birth_tick"] <= boundary)
        ok = source_alive == exported_alive and source_lineage_count == exported_lineage_count
        if not ok:
            mismatches.append(f"checkpoint boundary mismatch at generation {snapshot.generation}")
        exact_deaths = sum(
            1
            for agent in playback["agents"]
            if agent.get("death_tick_source") == "typed_lifecycle_death_event" and agent.get("death_tick") is not None and agent["death_tick"] <= boundary
        )
        lineage_counts: dict[str, int] = {}
        for source_id in exported_alive:
            lineage_id = agents_by_source_id[source_id]["lineage_id"]
            lineage_counts[lineage_id] = lineage_counts.get(lineage_id, 0) + 1
        per_boundary_checks.append(
            {
                "generation": snapshot.generation,
                "boundary_tick": boundary,
                "source_alive_count": len(source_alive),
                "exported_alive_count": len(exported_alive),
                "source_alive_identity_set_sha256": output_hash(sorted(source_alive)),
                "exported_alive_identity_set_sha256": output_hash(sorted(exported_alive)),
                "source_lineage_count": source_lineage_count,
                "exported_lineage_count": exported_lineage_count,
                "exported_lineage_counts": dict(sorted(lineage_counts.items())),
                "cumulative_births": cumulative_births,
                "exact_deaths": exact_deaths,
                "projected_disappearances": projected_disappearances,
                "ok": ok,
            }
        )
    return {
        "schema_version": "eiel_lineage_playback_cross_check_v1",
        "source_cell_id": selected_cell,
        "source_initial_alive_count": len(ordered_snapshots[0].alive_agent_ids),
        "source_successful_birth_count": len(births),
        "exported_successful_birth_count": exported_typed_births,
        "checkpoint_first_observed_identity_count": checkpoint_first_observed,
        "pre_window_birth_event_count": len(births) - exported_typed_births,
        "exported_agent_count": len(playback["agents"]),
        "source_final_alive_count": len(final_alive),
        "exported_final_alive_count": len(exported_final_alive),
        "source_lineage_count_final": len(set(final.lineage_by_agent_id.values())),
        "exported_lineage_count": len(playback["lineages"]),
        "per_boundary_checks": per_boundary_checks,
        "checkpoint_boundary_checks": per_boundary_checks,
        "coverage_interval": [playback["metadata"]["first_tick"], playback["metadata"]["last_tick"]],
        "mismatches": mismatches,
        "ok": not mismatches,
    }
