from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .lifecycle_events import (
    FIELD_DTYPES,
    LifecycleEventType,
    SCHEMA_VERSION,
    typed_lifecycle_event_digest,
    typed_lifecycle_events_from_projection,
    typed_lifecycle_events_to_rows,
)


@dataclass(frozen=True)
class TypedLifecycleArtifact:
    source: str
    active_count: int
    cursor: int
    digest: str
    rows: tuple[dict[str, Any], ...]
    jsonl_rows: tuple[dict[str, Any], ...] = ()
    csv_rows: tuple[dict[str, Any], ...] = ()

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": "eiel_typed_lifecycle_event_artifact_v1",
            "source": self.source,
            "active_count": int(self.active_count),
            "cursor": int(self.cursor),
            "digest": self.digest,
            "jsonl_rows": len(self.jsonl_rows),
            "csv_rows": len(self.csv_rows),
        }


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"schema_version": str(row.get("schema_version", SCHEMA_VERSION) or SCHEMA_VERSION)}
    for name, dtype in FIELD_DTYPES.items():
        value = row.get(name)
        out[name] = float(value) if dtype.is_floating_point else int(value)
    return out


def _load_jsonl_rows(path: str | Path | None) -> tuple[dict[str, Any], ...]:
    if path is None:
        return ()
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(_normalize_row(json.loads(line)))
    return tuple(rows)


def _load_csv_rows(path: str | Path | None) -> tuple[dict[str, Any], ...]:
    if path is None:
        return ()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return tuple(_normalize_row(row) for row in csv.DictReader(handle))


def typed_lifecycle_artifact_from_projection(
    projection: Mapping[str, Any],
    *,
    source: str,
    jsonl_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> TypedLifecycleArtifact:
    metadata = dict(projection.get("metadata") or {})
    events = typed_lifecycle_events_from_projection(projection)
    rows = tuple(typed_lifecycle_events_to_rows(events))
    active_count = len(rows)
    return TypedLifecycleArtifact(
        source=str(source),
        active_count=active_count,
        cursor=int(metadata.get("typed_lifecycle_event_active_count", metadata.get("event_tape_cursor", active_count)) or 0),
        digest=typed_lifecycle_event_digest(events),
        rows=rows,
        jsonl_rows=_load_jsonl_rows(jsonl_path),
        csv_rows=_load_csv_rows(csv_path),
    )


def compare_typed_lifecycle_artifacts(
    expected: TypedLifecycleArtifact,
    actual: TypedLifecycleArtifact,
    *,
    require_public_rows: bool = False,
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    for field in ("active_count", "cursor", "digest", "rows"):
        if getattr(expected, field) != getattr(actual, field):
            mismatches.append({"field": field, "expected": getattr(expected, field), "actual": getattr(actual, field)})
    for artifact in (expected, actual):
        if artifact.jsonl_rows and artifact.jsonl_rows != artifact.rows:
            mismatches.append({"field": f"{artifact.source}.jsonl_rows", "expected": artifact.rows, "actual": artifact.jsonl_rows})
        if artifact.csv_rows and artifact.csv_rows != artifact.rows:
            mismatches.append({"field": f"{artifact.source}.csv_rows", "expected": artifact.rows, "actual": artifact.csv_rows})
        if require_public_rows and (not artifact.jsonl_rows or not artifact.csv_rows):
            mismatches.append({"field": f"{artifact.source}.public_rows", "expected": "jsonl_and_csv", "actual": "missing"})
    if expected.jsonl_rows and actual.jsonl_rows and expected.jsonl_rows != actual.jsonl_rows:
        mismatches.append({"field": "jsonl_rows", "expected": expected.jsonl_rows, "actual": actual.jsonl_rows})
    if expected.csv_rows and actual.csv_rows and expected.csv_rows != actual.csv_rows:
        mismatches.append({"field": "csv_rows", "expected": expected.csv_rows, "actual": actual.csv_rows})
    return {
        "schema_version": "eiel_typed_lifecycle_event_artifact_compare_v1",
        "ok": not mismatches,
        "expected": expected.to_summary(),
        "actual": actual.to_summary(),
        "mismatches": mismatches,
    }


def lifecycle_semantic_rows_from_artifact(artifact: TypedLifecycleArtifact) -> tuple[dict[str, Any], ...]:
    rows = []
    for row in artifact.rows:
        rows.append(
            {
                "schema_version": "eiel_lifecycle_semantic_row_v1",
                "generation": int(row["generation"]),
                "tick": int(row["tick"]),
                "sequence": int(row["sequence"]),
                "event_type": int(row["event_type"]),
                "actor_agent_id": int(row["actor_agent_id"]),
                "target_agent_id": int(row["target_agent_id"]),
                "parent_agent_id": int(row["parent_agent_id"]),
                "child_agent_id": int(row["child_agent_id"]),
                "lineage_founder_id": int(row["lineage_founder_id"]),
                "lineage_depth": int(row["lineage_depth"]),
                "reason_code": int(row["reason_code"]),
                "success": int(row["success"]),
            }
        )
    return tuple(rows)


def lifecycle_counts_from_artifact(artifact: TypedLifecycleArtifact) -> dict[str, Any]:
    counts = {
        "schema_version": "eiel_lifecycle_counts_from_typed_events_v1",
        "source": artifact.source,
        "event_rows": int(artifact.active_count),
        "event_cursor": int(artifact.cursor),
        "successful_births": 0,
        "failed_births": 0,
        "deaths": 0,
        "mutations": 0,
        "slot_reuse": 0,
    }
    for row in artifact.rows:
        event_type = int(row["event_type"])
        if event_type == int(LifecycleEventType.BIRTH_COMMITTED):
            counts["successful_births"] += 1
        elif event_type == int(LifecycleEventType.BIRTH_FAILED):
            counts["failed_births"] += 1
        elif event_type == int(LifecycleEventType.DEATH_COMMITTED):
            counts["deaths"] += 1
        elif event_type == int(LifecycleEventType.MUTATION):
            counts["mutations"] += 1
    return counts


def final_lifecycle_summary_from_artifact(
    artifact: TypedLifecycleArtifact,
    *,
    final_alive_count: int | None = None,
    lifecycle_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = dict(
        lifecycle_counts
        if lifecycle_counts is not None
        else lifecycle_counts_from_artifact(artifact)
    )
    summary["schema_version"] = "eiel_final_lifecycle_summary_from_typed_events_v1"
    if final_alive_count is not None:
        summary["final_alive_count"] = int(final_alive_count)
    summary["event_schema_version"] = SCHEMA_VERSION
    summary["event_digest"] = artifact.digest
    return summary


def lineage_funnel_semantic_rows_from_artifact(artifact: TypedLifecycleArtifact) -> tuple[dict[str, Any], ...]:
    rows = []
    for row in artifact.rows:
        event_type = int(row["event_type"])
        if event_type not in {
            int(LifecycleEventType.BIRTH_COMMITTED),
            int(LifecycleEventType.BIRTH_FAILED),
        }:
            continue
        success = event_type == int(LifecycleEventType.BIRTH_COMMITTED)
        rows.append(
            {
                "schema_version": "eiel_lineage_funnel_semantic_row_from_typed_events_v1",
                "generation": int(row["generation"]),
                "tick": int(row["tick"]),
                "sequence": int(row["sequence"]),
                "event_type": "birth_success" if success else "birth_failed",
                "parent_id": int(row["parent_agent_id"] if success else row["actor_agent_id"]),
                "child_id": int(row["child_agent_id"]) if success else -1,
                "lineage_id": int(row["lineage_founder_id"]),
                "lineage_depth": int(row["lineage_depth"]),
                "birth_success": 1 if success else 0,
                "birth_failed": 0 if success else 1,
                "failure_reason_code": -1 if success else int(row["reason_code"]),
            }
        )
    return tuple(rows)


def checkpoint_sidecar_semantics_from_projection(
    projection: Mapping[str, Any],
    *,
    artifact: TypedLifecycleArtifact | None = None,
) -> dict[str, Any]:
    metadata = dict(projection.get("metadata") or {})
    if artifact is None:
        artifact = typed_lifecycle_artifact_from_projection(projection, source="checkpoint_sidecar")
    return {
        "schema_version": "eiel_checkpoint_sidecar_semantics_v1",
        "event_schema_version": str(metadata.get("typed_lifecycle_event_schema_version", SCHEMA_VERSION) or SCHEMA_VERSION),
        "active_event_count": int(metadata.get("event_tape_active_count", artifact.active_count) or 0),
        "event_cursor": int(metadata.get("event_tape_cursor", artifact.cursor) or 0),
        "typed_lifecycle_event_active_count": int(
            metadata.get("typed_lifecycle_event_active_count", artifact.active_count) or 0
        ),
        "agent_grid_phase_contract_schema_version": str(
            metadata.get("agent_grid_phase_contract_schema_version", "") or ""
        ),
        "agent_grid_authority_epoch": str(metadata.get("agent_grid_authority_epoch", "") or ""),
        "digest": artifact.digest,
    }


def final_lineage_summary_semantics_from_projection(
    projection: Mapping[str, Any],
    *,
    lifecycle_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tensors = dict(projection.get("tensors") or {})

    def _array(name: str):
        value = tensors.get(name)
        if value is None or not hasattr(value, "detach"):
            return None
        return value.detach().cpu().numpy()

    alive = _array("alive_mask")
    slot_to_agent_id = _array("slot_to_agent_id")
    if alive is None or slot_to_agent_id is None:
        raise KeyError("final lineage summary requires alive_mask and slot_to_agent_id")
    founder = _array("lineage_founder_id")
    depth = _array("lineage_depth")
    offspring = _array("offspring_count")
    missing_lineage = [
        name
        for name, value in (
            ("lineage_founder_id", founder),
            ("lineage_depth", depth),
            ("offspring_count", offspring),
        )
        if value is None
    ]
    if missing_lineage:
        raise KeyError(f"final lineage summary requires canonical lineage tensors: {missing_lineage}")

    lineage_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for slot, aid_raw in enumerate(slot_to_agent_id.tolist()):
        if int(aid_raw) < 0 or not bool(alive[slot]):
            continue
        lineage_id = int(founder[slot])
        if lineage_id < 0:
            raise ValueError(
                "final lineage summary has invalid founder identity for "
                f"agent_id={int(aid_raw)} slot={int(slot)}"
            )
        row = {
            "agent_id": int(aid_raw),
            "slot": int(slot),
            "lineage_id": lineage_id,
            "lineage_depth": int(depth[slot]),
            "offspring_count": int(offspring[slot]),
        }
        member_rows.append(row)

    by_lineage: dict[int, dict[str, Any]] = {}
    for row in member_rows:
        lineage_id = int(row["lineage_id"])
        bucket = by_lineage.setdefault(
            lineage_id,
            {
                "lineage_id": lineage_id,
                "member_ids": [],
                "member_count": 0,
                "max_depth": 0,
                "cumulative_births": 0,
            },
        )
        bucket["member_ids"].append(int(row["agent_id"]))
        bucket["member_count"] += 1
        bucket["max_depth"] = max(int(bucket["max_depth"]), int(row["lineage_depth"]))
        bucket["cumulative_births"] += int(row["offspring_count"])

    for lineage_id in sorted(by_lineage):
        bucket = dict(by_lineage[lineage_id])
        bucket["member_ids"] = sorted(int(x) for x in bucket["member_ids"])
        lineage_rows.append(bucket)

    counts = dict(lifecycle_counts or {})
    return {
        "schema_version": "eiel_final_lineage_summary_semantics_v1",
        "alive_count": int(bool(alive.size) and alive.astype(bool).sum()),
        "surviving_lineage_ids": [int(row["lineage_id"]) for row in lineage_rows],
        "lineage_count": int(len(lineage_rows)),
        "lineages": tuple(lineage_rows),
        "members": tuple(sorted(member_rows, key=lambda row: (int(row["lineage_id"]), int(row["agent_id"])))),
        "failed_births": int(counts.get("failed_births", 0) or 0),
        "deaths": int(counts.get("deaths", 0) or 0),
        "mutations": int(counts.get("mutations", 0) or 0),
    }


def final_summary_json_semantics_from_projection(
    projection: Mapping[str, Any],
    *,
    artifact: TypedLifecycleArtifact | None = None,
    lifecycle_counts: Mapping[str, Any] | None = None,
    final_lineage_summary: Mapping[str, Any] | None = None,
    checkpoint_sidecar_semantics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if artifact is None:
        artifact = typed_lifecycle_artifact_from_projection(projection, source="final_summary")
    final_summary_lifecycle_counts = dict(
        lifecycle_counts
        if lifecycle_counts is not None
        else lifecycle_counts_from_artifact(artifact)
    )
    final_summary_lifecycle_counts["source"] = "final_summary"
    final_lifecycle = final_lifecycle_summary_from_artifact(
        artifact,
        lifecycle_counts=final_summary_lifecycle_counts,
    )
    final_lineage = (
        final_lineage_summary
        if final_lineage_summary is not None
        else final_lineage_summary_semantics_from_projection(
            projection,
            lifecycle_counts=final_summary_lifecycle_counts,
        )
    )
    sidecar = (
        checkpoint_sidecar_semantics
        if checkpoint_sidecar_semantics is not None
        else checkpoint_sidecar_semantics_from_projection(projection, artifact=artifact)
    )
    metadata = dict(projection.get("metadata") or {})
    return {
        "schema_version": "eiel_final_summary_json_semantics_v1",
        "authoritative_semantic_fields": {
            "generation": int(metadata.get("generation", 0) or 0),
            "tick": int(metadata.get("tick", 0) or 0),
            "world": str(metadata.get("world", "") or ""),
            "typed_event_active_count": int(artifact.active_count),
            "typed_event_cursor": int(artifact.cursor),
            "typed_event_digest": artifact.digest,
            "lifecycle_counts": final_summary_lifecycle_counts,
            "final_lifecycle_summary": final_lifecycle,
            "final_lineage_summary": final_lineage,
            "checkpoint_sidecar_semantics": sidecar,
        },
        "deterministic_storage_fields": {
            "checkpoint_schema_version": str(projection.get("schema_version", "") or ""),
            "agent_grid_phase_contract_schema_version": str(
                metadata.get("agent_grid_phase_contract_schema_version", "") or ""
            ),
            "agent_grid_authority_epoch": str(metadata.get("agent_grid_authority_epoch", "") or ""),
        },
        "diagnostic_only_fields": {},
        "implementation_specific_fields": {},
    }
