from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import SCHEMA_VERSION


class SchemaError(ValueError):
    """Raised when a playback document violates the V0 schema."""


PROJECTED_DEATH_SOURCES = {"checkpoint_boundary_projection", "terminal_boundary_projection"}
DEATH_TICK_SOURCES = PROJECTED_DEATH_SOURCES | {"typed_lifecycle_death_event", "implicit_lifecycle_record", "not_recorded"}


@dataclass(frozen=True)
class ValidationResult:
    warnings: tuple[str, ...] = ()


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaError(f"{label} must be an object")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaError(f"{label} must be a list")
    return value


def _require_id(record: dict[str, Any], field: str, label: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{label}.{field} must be a non-empty string")
    return value


def validate_playback(data: dict[str, Any]) -> ValidationResult:
    root = _require_mapping(data, "playback")
    metadata = _require_mapping(root.get("metadata"), "metadata")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(
            f"unsupported schema_version {metadata.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    if metadata.get("semantics_mode") not in {
        "actual_agents",
        "partial_agents",
        "representative_particles",
    }:
        raise SchemaError("metadata.semantics_mode must be a supported value")
    if metadata.get("source_classification") is not None and metadata.get("source_classification") not in {"EXP-A", "EXP-B", "EXP-C", "EXP-D"}:
        raise SchemaError("metadata.source_classification must be EXP-A, EXP-B, EXP-C, or EXP-D")
    first_tick = metadata.get("first_tick")
    last_tick = metadata.get("last_tick")
    if not isinstance(first_tick, int) or not isinstance(last_tick, int):
        raise SchemaError("metadata.first_tick and metadata.last_tick must be integers")
    if last_tick < first_tick:
        raise SchemaError("metadata.last_tick must be >= metadata.first_tick")

    lineages = _require_list(root.get("lineages"), "lineages")
    branches = _require_list(root.get("branches"), "branches")
    agents = _require_list(root.get("agents"), "agents")

    lineage_ids: set[str] = set()
    for raw in lineages:
        item = _require_mapping(raw, "lineage")
        lineage_id = _require_id(item, "lineage_id", "lineage")
        if lineage_id in lineage_ids:
            raise SchemaError(f"duplicate lineage_id {lineage_id}")
        lineage_ids.add(lineage_id)

    for raw in lineages:
        item = _require_mapping(raw, "lineage")
        lineage_id = item["lineage_id"]
        parent_id = item.get("parent_lineage_id")
        founder_id = _require_id(item, "founder_lineage_id", "lineage")
        if parent_id is not None and parent_id not in lineage_ids:
            raise SchemaError(f"lineage {lineage_id} references missing parent {parent_id}")
        if founder_id not in lineage_ids:
            raise SchemaError(f"lineage {lineage_id} references missing founder {founder_id}")
        created = item.get("creation_tick")
        extinct = item.get("extinction_tick")
        if not isinstance(created, int):
            raise SchemaError(f"lineage {lineage_id} creation_tick must be an integer")
        if extinct is not None and (not isinstance(extinct, int) or extinct < created):
            raise SchemaError(f"lineage {lineage_id} has invalid extinction_tick")

    branch_ids: set[str] = set()
    branch_lineage: dict[str, str] = {}
    for raw in branches:
        item = _require_mapping(raw, "branch")
        branch_id = _require_id(item, "branch_id", "branch")
        if branch_id in branch_ids:
            raise SchemaError(f"duplicate branch_id {branch_id}")
        lineage_id = _require_id(item, "lineage_id", "branch")
        if lineage_id not in lineage_ids:
            raise SchemaError(f"branch {branch_id} references missing lineage {lineage_id}")
        branch_ids.add(branch_id)
        branch_lineage[branch_id] = lineage_id

    for raw in branches:
        item = _require_mapping(raw, "branch")
        branch_id = item["branch_id"]
        parent_id = item.get("parent_branch_id")
        if parent_id is not None:
            if parent_id not in branch_ids:
                raise SchemaError(f"branch {branch_id} references missing parent {parent_id}")
            if branch_lineage[parent_id] != branch_lineage[branch_id]:
                raise SchemaError(f"branch {branch_id} parent belongs to another lineage")
        created = item.get("creation_tick")
        extinct = item.get("extinction_tick")
        if not isinstance(created, int):
            raise SchemaError(f"branch {branch_id} creation_tick must be an integer")
        if extinct is not None and (not isinstance(extinct, int) or extinct < created):
            raise SchemaError(f"branch {branch_id} has invalid extinction_tick")

    agent_ids: set[str] = set()
    for raw in agents:
        item = _require_mapping(raw, "agent")
        agent_id = _require_id(item, "agent_id", "agent")
        if agent_id in agent_ids:
            raise SchemaError(f"duplicate agent_id {agent_id}")
        agent_ids.add(agent_id)

    warnings: list[str] = []
    parent_policy = metadata.get("parent_reference_policy", "strict")
    for raw in agents:
        item = _require_mapping(raw, "agent")
        agent_id = item["agent_id"]
        lineage_id = _require_id(item, "lineage_id", "agent")
        branch_id = _require_id(item, "branch_id", "agent")
        if lineage_id not in lineage_ids:
            raise SchemaError(f"agent {agent_id} references missing lineage {lineage_id}")
        if branch_id not in branch_ids:
            raise SchemaError(f"agent {agent_id} references missing branch {branch_id}")
        if branch_lineage[branch_id] != lineage_id:
            raise SchemaError(f"agent {agent_id} branch belongs to another lineage")
        birth_tick = item.get("birth_tick")
        death_tick = item.get("death_tick")
        if not isinstance(birth_tick, int):
            raise SchemaError(f"agent {agent_id} birth_tick must be an integer")
        if death_tick is not None and (not isinstance(death_tick, int) or death_tick < birth_tick):
            raise SchemaError(f"agent {agent_id} has invalid death_tick")
        raw_death_source = item.get("death_tick_source")
        death_source = raw_death_source if raw_death_source is not None else ("implicit_lifecycle_record" if death_tick is not None else "not_recorded")
        if death_source not in DEATH_TICK_SOURCES:
            raise SchemaError(f"agent {agent_id} has unsupported death_tick_source {death_source!r}")
        interval = item.get("death_interval")
        if interval is not None:
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not all(isinstance(value, int) for value in interval)
            ):
                raise SchemaError(f"agent {agent_id} death_interval must be two integer bounds")
            lower, upper = interval
            if upper < lower:
                raise SchemaError(f"agent {agent_id} death_interval upper bound must be >= lower bound")
            if upper < birth_tick:
                raise SchemaError(f"agent {agent_id} death_interval must end after birth_tick")
            if death_tick is not None and death_tick != upper:
                raise SchemaError(f"agent {agent_id} death_tick must match death_interval upper bound")
        if death_source in PROJECTED_DEATH_SOURCES:
            if interval is None:
                raise SchemaError(f"agent {agent_id} projected death requires death_interval")
            if death_tick is None:
                raise SchemaError(f"agent {agent_id} projected death requires death_tick")
            if metadata.get("semantics_mode") == "actual_agents" or metadata.get("source_classification") == "EXP-A":
                raise SchemaError("projected deaths require non-EXP-A partial semantics")
        if death_source == "typed_lifecycle_death_event" and death_tick is None:
            raise SchemaError(f"agent {agent_id} typed death source requires death_tick")
        if raw_death_source == "not_recorded" and death_tick is not None:
            raise SchemaError(f"agent {agent_id} cannot have death_tick_source not_recorded with a death_tick")
        parents = item.get("parent_agent_ids", [])
        if not isinstance(parents, list) or len(parents) > 2:
            raise SchemaError(f"agent {agent_id} parent_agent_ids must contain zero to two ids")
        for parent_id in parents:
            if not isinstance(parent_id, str) or not parent_id:
                raise SchemaError(f"agent {agent_id} parent_agent_ids contains an invalid id")
            if parent_id not in agent_ids:
                if parent_policy == "allow_missing_external":
                    warnings.append(f"agent {agent_id} references unavailable parent {parent_id}")
                else:
                    raise SchemaError(f"agent {agent_id} references missing parent {parent_id}")
    return ValidationResult(tuple(warnings))


def peak_living_count(data: dict[str, Any]) -> int:
    validate_playback(data)
    deltas: dict[int, int] = {}
    for agent in data["agents"]:
        birth_tick = agent["birth_tick"]
        death_tick = agent.get("death_tick")
        deltas[birth_tick] = deltas.get(birth_tick, 0) + 1
        if death_tick is not None:
            deltas[death_tick] = deltas.get(death_tick, 0) - 1
    living = 0
    peak = 0
    for tick in sorted(deltas):
        living += deltas[tick]
        peak = max(peak, living)
    return peak
