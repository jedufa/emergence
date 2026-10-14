from __future__ import annotations

"""Passive genome attribution over authoritative reproduction observations.

This module is deliberately offline. It consumes no RNG, imports no runner,
and never changes reproduction, mutation, policy, selection, or Chronicle data.
"""

from collections import Counter, defaultdict, deque
import csv
import io
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from agents.modular_genome_v1 import canonical_hash_from_genome_dict, canonical_payload_from_genome_dict
from simulation.chronicle.schema import canonical_json, canonical_payload, stable_hash
from simulation.chronicle.storage import atomic_write_text, sha256_file, write_json, write_jsonl


MUTATION_EVENT_SCHEMA = "eiel_mutation_event_v1"
GENOME_DELTA_SCHEMA = "eiel_genome_delta_v1"
INHERITANCE_SCHEMA = "eiel_mutation_inheritance_v1"
PERSISTENCE_SCHEMA = "eiel_mutation_persistence_summary_v1"
CARRIER_SCHEMA = "eiel_mutation_carrier_summary_v1"
ASSOCIATION_SCHEMA = "eiel_mutation_outcome_association_v1"
MANIFEST_SCHEMA = "eiel_mutation_attribution_manifest_v1"
SUCCESS_PROFILE_SCHEMA = "eiel_genome_success_profile_v1"
STATE_SCHEMA = "eiel_genome_attribution_accumulator_state_v1"

EVIDENCE_STRENGTHS = (
    "observed",
    "inherited",
    "persisted",
    "associated",
    "enriched",
    "branch_supported",
    "experimentally_supported",
)
EVIDENCE_RANK = {value: index for index, value in enumerate(EVIDENCE_STRENGTHS)}

OUTPUT_NAMES = (
    "mutation_events.jsonl",
    "genome_deltas.jsonl",
    "mutation_inheritance.jsonl",
    "mutation_persistence_summary.json",
    "mutation_carrier_summary.json",
    "mutation_outcome_associations.csv",
    "mutation_attribution_manifest.json",
    "mutation_attribution_report.md",
)

DEFAULT_MATCH_FIELDS = (
    "generation",
    "lineage_id",
    "parent_or_sibling_group",
    "age",
    "scenario",
    "seed",
    "region_id",
    "opportunity_exposure",
    "starting_state",
)

AUTHORITATIVE_TRAIT_LABELS = {
    "/genes/exploration_bias": "exploration_bias",
    "/genes/aggression_bias": "aggression_bias",
    "/genes/reproduction_threshold": "reproduction_threshold",
    "/genes/mutation_rate": "mutation_rate",
    "/genes/action_temperature": "action_temperature",
    "/genes/memory_dim": "memory_dim",
    "/genes/sensor_channel_mask": "sensor_channel_mask",
    "/genes/sensor_radius": "sensor_radius",
    "/genes/introspection_channel_mask": "introspection_channel_mask",
    "/genes/self_regulation_control_mask": "self_regulation_control_mask",
    "/genes/self_modification_target_mask": "self_modification_target_mask",
    "/genes/directed_inheritance_channel_mask": "directed_inheritance_channel_mask",
    "/genes/drive_inertia": "drive_inertia",
    "/genes/drive_noise": "drive_noise",
    "/genes/drive_logit_gain": "drive_logit_gain",
    "/genes/pred_lr": "pred_lr",
    "/genes/resource_handling_axis": "resource_handling_axis",
}

_FORBIDDEN_CAUSAL_WORDING = re.compile(
    r"\b(caused|beneficial|learned|adopted\s+(?:a\s+)?strategy|likely\s+cause|proved\s+effective)\b",
    re.IGNORECASE,
)
_MISSING = object()


class GenomeAttributionError(ValueError):
    """Raised when an attribution record cannot retain exact source identity."""


def _normalized_hash(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text[7:]
    if not text:
        return None
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise GenomeAttributionError(f"invalid SHA-256 hash: {value}")
    return text


def _canonical_genome(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return canonical_payload_from_genome_dict(dict(payload))
    except Exception as exc:
        raise GenomeAttributionError(f"invalid canonical genome payload: {exc}") from exc


def _genome_hash(payload: Mapping[str, Any]) -> str:
    try:
        return canonical_hash_from_genome_dict(dict(payload))
    except Exception as exc:
        raise GenomeAttributionError(f"canonical genome hashing failed: {exc}") from exc


def _stage(
    record: Mapping[str, Any],
    payload_field: str,
    hash_field: str,
    *,
    required: bool,
) -> dict[str, Any]:
    raw_payload = record.get(payload_field)
    payload = _canonical_genome(raw_payload) if isinstance(raw_payload, Mapping) else None
    supplied = _normalized_hash(record.get(hash_field))
    computed = _genome_hash(payload) if payload is not None else None
    if supplied and computed and supplied != computed:
        raise GenomeAttributionError(
            f"{hash_field} does not reconcile with {payload_field}: supplied={supplied} computed={computed}"
        )
    resolved = supplied or computed
    if required and not resolved:
        raise GenomeAttributionError(f"{payload_field} or {hash_field} is required")
    return {
        "payload": payload,
        "hash": resolved,
        "hash_verification": "verified_from_payload" if computed else ("source_hash_only" if supplied else "unavailable"),
    }


def _parent_stages(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_payloads = list(record.get("parent_genomes", []) or [])
    raw_hashes = list(record.get("parent_genome_hashes", []) or [])
    parent_ids = [int(value) for value in record.get("parent_agent_ids", []) or []]
    count = max(len(raw_payloads), len(raw_hashes), len(parent_ids))
    if count == 0:
        raise GenomeAttributionError("parent_agent_ids and parent genome identity are required")
    if len(parent_ids) != count:
        raise GenomeAttributionError("parent agent and parent genome counts do not match")
    rows: list[dict[str, Any]] = []
    for index in range(count):
        raw_payload = raw_payloads[index] if index < len(raw_payloads) else None
        payload = _canonical_genome(raw_payload) if isinstance(raw_payload, Mapping) else None
        supplied = _normalized_hash(raw_hashes[index]) if index < len(raw_hashes) else None
        computed = _genome_hash(payload) if payload is not None else None
        if supplied and computed and supplied != computed:
            raise GenomeAttributionError(f"parent genome hash mismatch at parent index {index}")
        resolved = supplied or computed
        if not resolved:
            raise GenomeAttributionError(f"parent genome identity unavailable at parent index {index}")
        rows.append(
            {
                "agent_id": parent_ids[index],
                "payload": payload,
                "hash": resolved,
                "hash_verification": "verified_from_payload" if computed else "source_hash_only",
            }
        )
    return rows


def _source_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    refs = sorted({str(value) for value in record.get("source_record_refs", []) or [] if str(value)})
    if not refs:
        raise GenomeAttributionError("source_record_refs are required for exact attribution identity")
    run_id = str(record.get("run_id") or "")
    if not run_id:
        raise GenomeAttributionError("run_id is required")
    child_id = record.get("child_agent_id")
    if child_id is None:
        raise GenomeAttributionError("child_agent_id is required")
    return {
        "run_id": run_id,
        "campaign_id": str(record.get("campaign_id")) if record.get("campaign_id") is not None else None,
        "branch_id": str(record.get("branch_id")) if record.get("branch_id") is not None else None,
        "source_commit": str(record.get("source_commit") or "unknown"),
        "generation": int(record.get("generation", 0)),
        "step": int(record.get("step", -1)),
        "reproduction_phase": str(record.get("reproduction_phase") or "reproduction"),
        "parent_agent_ids": sorted(int(value) for value in record.get("parent_agent_ids", []) or []),
        "child_agent_id": int(child_id),
        "lineage_id": str(record.get("lineage_id")) if record.get("lineage_id") is not None else None,
        "source_record_refs": refs,
    }


def normalize_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    identity = _source_identity(record)
    parents = sorted(_parent_stages(record), key=lambda value: int(value["agent_id"]))
    if identity["parent_agent_ids"] != sorted(int(value["agent_id"]) for value in parents):
        raise GenomeAttributionError("parent IDs do not reconcile with parent genome records")
    pre = _stage(record, "child_pre_mutation_genome", "child_pre_mutation_genome_hash", required=True)
    post = _stage(record, "child_post_mutation_genome", "child_post_mutation_genome_hash", required=True)
    final = _stage(record, "final_attached_child_genome", "final_attached_child_genome_hash", required=True)
    source_id = "gsrc_" + stable_hash(
        "eiel-genome-attribution-source-v1",
        {
            **identity,
            "parent_hashes": [value["hash"] for value in parents],
            "pre_hash": pre["hash"],
            "post_hash": post["hash"],
            "final_hash": final["hash"],
            "operator": str(record.get("mutation_operator") or "unknown_unavailable"),
        },
    )
    return canonical_payload(
        {
            "source_record_id": source_id,
            **identity,
            "parents": parents,
            "child_pre": pre,
            "child_post": post,
            "child_final": final,
            "mutation_operator": str(record.get("mutation_operator") or "unknown_unavailable"),
            "source_completeness": str(record.get("source_completeness") or "unknown"),
            "chronicle_event_id": str(record.get("chronicle_event_id")) if record.get("chronicle_event_id") else None,
            "birth_chronicle_event_id": str(record.get("birth_chronicle_event_id")) if record.get("birth_chronicle_event_id") else None,
            "authoritative_trait_labels": dict(record.get("authoritative_trait_labels", {}) or {}),
            "population_size": int(record["population_size"]) if record.get("population_size") is not None else None,
            "scenario": str(record.get("scenario") or "unknown"),
            "seed": record.get("seed", "unknown"),
        }
    )


def _escape_pointer(value: str) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _flatten(value: Any, *, path: str = "") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    if isinstance(value, Mapping):
        if not value:
            rows[path or "/"] = {}
        for key in sorted(value, key=str):
            rows.update(_flatten(value[key], path=f"{path}/{_escape_pointer(str(key))}"))
    elif isinstance(value, (list, tuple)):
        if not value:
            rows[path or "/"] = []
        for index, item in enumerate(value):
            rows.update(_flatten(item, path=f"{path}/{index}"))
    else:
        rows[path or "/"] = canonical_payload(value)
    return rows


def _numeric_delta(before: Any, after: Any) -> int | float | None:
    if isinstance(before, bool) or isinstance(after, bool):
        return None
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        value = after - before
        return int(value) if isinstance(before, int) and isinstance(after, int) else float(value)
    return None


def _path_parts(path: str) -> tuple[str | None, str | None, str | None]:
    parts = [value for value in path.split("/") if value]
    locus = parts[0] if parts else None
    module = None
    if locus in {"modules", "connections"} and len(parts) > 1:
        module = f"{locus}:{parts[1]}"
    field = parts[-1] if parts else None
    return locus, module, field


def _raw_deltas(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    before_flat = _flatten(before)
    after_flat = _flatten(after)
    rows: list[dict[str, Any]] = []
    for path in sorted(set(before_flat) | set(after_flat)):
        prior = before_flat.get(path, _MISSING)
        current = after_flat.get(path, _MISSING)
        if prior is not _MISSING and current is not _MISSING and canonical_json(prior) == canonical_json(current):
            continue
        locus, module, field = _path_parts(path)
        rows.append(
            {
                "path": path,
                "changed_locus": locus,
                "changed_module": module,
                "changed_field": field,
                "before_available": prior is not _MISSING,
                "before_value": None if prior is _MISSING else prior,
                "after_available": current is not _MISSING,
                "after_value": None if current is _MISSING else current,
                "delta": None if prior is _MISSING or current is _MISSING else _numeric_delta(prior, current),
                "change_kind": "added" if prior is _MISSING else ("removed" if current is _MISSING else "changed"),
            }
        )
    return rows


def _trait_label(path: str, overrides: Mapping[str, Any]) -> str | None:
    explicit = overrides.get(path)
    if explicit:
        return str(explicit)
    return AUTHORITATIVE_TRAIT_LABELS.get(path)


def _get_pointer(payload: Mapping[str, Any] | None, path: str) -> tuple[bool, Any]:
    if payload is None:
        return False, None
    current: Any = payload
    for raw_part in [value for value in path.split("/") if value]:
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if part not in current:
                return False, None
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, canonical_payload(current)


def _source_events(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pre = source["child_pre"]
    post = source["child_post"]
    raw_deltas = _raw_deltas(pre["payload"], post["payload"]) if pre.get("payload") is not None and post.get("payload") is not None else []
    if not raw_deltas:
        raw_deltas = [
            {
                "path": None,
                "changed_locus": None,
                "changed_module": None,
                "changed_field": None,
                "before_available": False,
                "before_value": None,
                "after_available": False,
                "after_value": None,
                "delta": None,
                "change_kind": "hash_only_unavailable" if pre.get("payload") is None or post.get("payload") is None else "no_canonical_genome_change",
            }
        ]
    events: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    for raw_delta in raw_deltas:
        delta_identity = {
            "path": raw_delta["path"],
            "before_available": raw_delta["before_available"],
            "before_value": raw_delta["before_value"],
            "after_available": raw_delta["after_available"],
            "after_value": raw_delta["after_value"],
            "change_kind": raw_delta["change_kind"],
        }
        event_id = "mut_" + stable_hash(
            "eiel-mutation-event-v1",
            {
                "source_record_id": source["source_record_id"],
                "operator": source["mutation_operator"],
                "delta": delta_identity,
            },
        )
        similar_key = "sim_" + stable_hash(
            "eiel-independent-similar-genome-delta-v1",
            {"operator": source["mutation_operator"], "delta": delta_identity},
        )
        path = raw_delta["path"]
        label = _trait_label(str(path), source.get("authoritative_trait_labels", {})) if path else None
        event = {
            "schema_version": MUTATION_EVENT_SCHEMA,
            "mutation_event_id": event_id,
            "source_record_id": source["source_record_id"],
            "run_id": source["run_id"],
            "campaign_id": source.get("campaign_id"),
            "branch_id": source.get("branch_id"),
            "source_commit": source.get("source_commit", "unknown"),
            "generation": source["generation"],
            "step": source["step"],
            "reproduction_phase": source["reproduction_phase"],
            "parent_agent_ids": source["parent_agent_ids"],
            "child_agent_id": source["child_agent_id"],
            "lineage_id": source.get("lineage_id"),
            "parent_genome_hashes": [value["hash"] for value in source["parents"]],
            "child_pre_mutation_genome_hash": pre["hash"],
            "child_post_mutation_genome_hash": post["hash"],
            "final_attached_child_genome_hash": source["child_final"]["hash"],
            "final_stage_relation": "same_as_post_mutation" if post["hash"] == source["child_final"]["hash"] else "different_from_post_mutation",
            "mutation_operator": source["mutation_operator"],
            "changed_locus": raw_delta["changed_locus"],
            "changed_module": raw_delta["changed_module"],
            "changed_field": raw_delta["changed_field"],
            "changed_path": path,
            "before_available": raw_delta["before_available"],
            "before_value": raw_delta["before_value"],
            "after_available": raw_delta["after_available"],
            "after_value": raw_delta["after_value"],
            "delta": raw_delta["delta"],
            "change_kind": raw_delta["change_kind"],
            "authoritative_trait_label": label,
            "opaque_parameter_semantics": path is not None and label is None,
            "source_completeness": source["source_completeness"],
            "source_record_refs": source["source_record_refs"],
            "chronicle_event_id": source.get("chronicle_event_id"),
            "birth_chronicle_event_id": source.get("birth_chronicle_event_id"),
            "confidence": "measured" if raw_delta["before_available"] and raw_delta["after_available"] else "partial",
            "evidence_strength": "observed",
            "causal_status": "not_assessed",
            "similar_delta_key": similar_key,
            "mutation_identity_relation": "distinct_recorded_event",
            "independent_similar_mutation_event_ids": [],
        }
        delta_id = "gdel_" + stable_hash(
            "eiel-genome-delta-v1",
            {"mutation_event_id": event_id, "delta": delta_identity},
        )
        delta_record = {
            "schema_version": GENOME_DELTA_SCHEMA,
            "genome_delta_id": delta_id,
            "mutation_event_id": event_id,
            "run_id": source["run_id"],
            "child_agent_id": source["child_agent_id"],
            "generation": source["generation"],
            "path": path,
            "stable_order_key": [str(path or ""), event_id],
            "change_kind": raw_delta["change_kind"],
            "before_available": raw_delta["before_available"],
            "before_value": raw_delta["before_value"],
            "after_available": raw_delta["after_available"],
            "after_value": raw_delta["after_value"],
            "delta": raw_delta["delta"],
            "pre_genome_hash": pre["hash"],
            "post_genome_hash": post["hash"],
            "hash_verification": {
                "pre": pre["hash_verification"],
                "post": post["hash_verification"],
                "final": source["child_final"]["hash_verification"],
                "reconstructable_from_source_payloads": pre.get("payload") is not None and post.get("payload") is not None,
            },
            "authoritative_trait_label": label,
            "semantic_meaning": label if label else "unknown_unavailable",
            "source_record_refs": source["source_record_refs"],
        }
        event["genome_delta_id"] = delta_id
        events.append(canonical_payload(event))
        deltas.append(canonical_payload(delta_record))
    return events, deltas


def _mark_independent_similar(events: list[dict[str, Any]]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("changed_path") and event.get("before_available") and event.get("after_available"):
            groups[str(event["similar_delta_key"])].append(event)
    for values in groups.values():
        source_ids = {str(value["source_record_id"]) for value in values}
        if len(source_ids) <= 1:
            continue
        ids = sorted(str(value["mutation_event_id"]) for value in values)
        for event in values:
            event["mutation_identity_relation"] = "independent_similar_mutation"
            event["independent_similar_mutation_event_ids"] = [value for value in ids if value != event["mutation_event_id"]]


def _inheritance(
    sources: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, set[int]], dict[str, int]]:
    by_child = {int(source["child_agent_id"]): source for source in sources}
    children: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for source in sources:
        for parent_id in source["parent_agent_ids"]:
            children[int(parent_id)].append(source)
    for values in children.values():
        values.sort(key=lambda value: (int(value["generation"]), int(value["child_agent_id"]), str(value["source_record_id"])))
    rows: list[dict[str, Any]] = []
    carriers: dict[str, set[int]] = defaultdict(set)
    carrier_generation: dict[str, int] = {}
    for event in sorted(events, key=lambda value: str(value["mutation_event_id"])):
        mutation_id = str(event["mutation_event_id"])
        origin = int(event["child_agent_id"])
        origin_generation = int(event["generation"])
        carriers[mutation_id].add(origin)
        carrier_generation[f"{mutation_id}:{origin}"] = origin_generation
        source = by_child.get(origin)
        if source is None:
            continue
        event_path = str(event.get("changed_path") or "")
        if not event_path or not bool(event.get("after_available")):
            continue
        queue: deque[tuple[int, list[int]]] = deque([(origin, [origin])])
        visited: set[int] = {origin}
        while queue:
            parent_id, chain = queue.popleft()
            for child_source in children.get(parent_id, []):
                child_id = int(child_source["child_agent_id"])
                if child_id in visited:
                    continue
                visited.add(child_id)
                final_payload = child_source["child_final"].get("payload")
                available, value = _get_pointer(final_payload, event_path)
                expected = event.get("after_value")
                retained = available and canonical_json(value) == canonical_json(expected)
                status = "inherited" if retained and parent_id == origin else ("present_in_descendant" if retained else ("lost" if available else "unavailable"))
                evidence_strength = "inherited" if retained else "observed"
                ancestry = chain + [child_id]
                inheritance_id = "inh_" + stable_hash(
                    "eiel-mutation-inheritance-v1",
                    {"mutation_event_id": mutation_id, "carrier_agent_id": child_id, "ancestry_chain": ancestry, "status": status},
                )
                row = {
                    "schema_version": INHERITANCE_SCHEMA,
                    "inheritance_record_id": inheritance_id,
                    "mutation_event_id": mutation_id,
                    "origin_agent_id": origin,
                    "inherited_from_agent_id": parent_id,
                    "carrier_agent_id": child_id,
                    "generation": int(child_source["generation"]),
                    "lineage_id": child_source.get("lineage_id"),
                    "status": status,
                    "path": event_path,
                    "expected_value": expected,
                    "observed_value_available": available,
                    "observed_value": value if available else None,
                    "ancestry_chain": ancestry,
                    "evidence_strength": evidence_strength,
                    "source_record_refs": child_source["source_record_refs"],
                    "causal_status": "not_assessed",
                }
                rows.append(canonical_payload(row))
                if retained:
                    carriers[mutation_id].add(child_id)
                    carrier_generation[f"{mutation_id}:{child_id}"] = int(child_source["generation"])
                    queue.append((child_id, ancestry))
    rows.sort(key=lambda value: (str(value["mutation_event_id"]), int(value["generation"]), int(value["carrier_agent_id"]), str(value["inheritance_record_id"])))
    return rows, carriers, carrier_generation


def _persistence_and_carriers(
    sources: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    inheritance: Sequence[Mapping[str, Any]],
    carriers: Mapping[str, set[int]],
    carrier_generation: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_by_child = {int(value["child_agent_id"]): value for value in sources}
    persistence_rows: list[dict[str, Any]] = []
    carrier_rows: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda value: str(value["mutation_event_id"])):
        mutation_id = str(event["mutation_event_id"])
        origin_generation = int(event["generation"])
        agent_ids = sorted(carriers.get(mutation_id, {int(event["child_agent_id"])}))
        generations = [int(carrier_generation.get(f"{mutation_id}:{agent_id}", origin_generation)) for agent_id in agent_ids]
        max_generation = max(generations, default=origin_generation)
        duration = max_generation - origin_generation
        evidence = "persisted" if duration > 0 else "observed"
        population_sizes = [
            int(source_by_child[agent_id]["population_size"])
            for agent_id in agent_ids
            if agent_id in source_by_child and source_by_child[agent_id].get("population_size")
        ]
        denominator = max(population_sizes) if population_sizes else None
        lineage_counts = Counter(
            str(source_by_child[agent_id].get("lineage_id") or "unknown")
            for agent_id in agent_ids
            if agent_id in source_by_child
        )
        persistence_rows.append(
            {
                "mutation_event_id": mutation_id,
                "origin_generation": origin_generation,
                "last_recorded_carrier_generation": max_generation,
                "persisted_generations": duration,
                "carrier_count": len(agent_ids),
                "evidence_strength": evidence,
                "status": "persisted" if duration > 0 else "observed_only",
                "limitations": [] if duration > 0 else ["No later descendant carrier was available in the supplied source records."],
            }
        )
        carrier_rows.append(
            {
                "mutation_event_id": mutation_id,
                "carrier_agent_ids": agent_ids,
                "carrier_count": len(agent_ids),
                "carrier_fraction": (len(agent_ids) / denominator) if denominator else None,
                "carrier_fraction_denominator": denominator,
                "lineage_carrier_counts": dict(sorted(lineage_counts.items())),
                "lineage_enrichment": {
                    "status": "unavailable",
                    "reason": "lineage-specific carrier and non-carrier denominators were not supplied",
                },
                "cross_lineage_occurrence": len(lineage_counts) > 1,
                "causal_status": "not_assessed",
            }
        )
    return (
        canonical_payload({"schema_version": PERSISTENCE_SCHEMA, "mutations": persistence_rows}),
        canonical_payload({"schema_version": CARRIER_SCHEMA, "mutations": carrier_rows}),
    )


def _association_rows(
    events: Sequence[Mapping[str, Any]],
    carriers: Mapping[str, set[int]],
    outcomes: Sequence[Mapping[str, Any]],
    *,
    required_match_fields: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    outcome_rows = [dict(value) for value in outcomes]
    for event in sorted(events, key=lambda value: str(value["mutation_event_id"])):
        mutation_id = str(event["mutation_event_id"])
        carrier_ids = carriers.get(mutation_id, set())
        metrics = sorted(
            {
                str(key)
                for outcome in outcome_rows
                for key in dict(outcome.get("outcomes", {}) or {})
            }
        )
        for metric in metrics or ["unavailable"]:
            matched_carriers: list[float] = []
            matched_controls: list[float] = []
            missing_fields = sorted(
                {
                    field
                    for outcome in outcome_rows
                    for field in required_match_fields
                    if field not in outcome or outcome.get(field) is None
                }
            )
            status = "unavailable"
            reason = "no matched carrier and non-carrier stratum"
            if not missing_fields and metric != "unavailable":
                by_key: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"carrier": [], "control": []})
                for outcome in outcome_rows:
                    values = dict(outcome.get("outcomes", {}) or {})
                    if metric not in values:
                        continue
                    try:
                        measurement = float(values[metric])
                    except (TypeError, ValueError):
                        continue
                    key = canonical_json([outcome.get(field) for field in required_match_fields])
                    group = "carrier" if int(outcome.get("agent_id", -1)) in carrier_ids else "control"
                    by_key[key][group].append(measurement)
                for values in by_key.values():
                    if values["carrier"] and values["control"]:
                        matched_carriers.extend(values["carrier"])
                        matched_controls.extend(values["control"])
                if matched_carriers and matched_controls:
                    status = "available"
                    reason = "exact matched strata available"
            if missing_fields:
                reason = "required matching fields unavailable: " + ", ".join(missing_fields)
            carrier_mean = sum(matched_carriers) / len(matched_carriers) if matched_carriers else None
            control_mean = sum(matched_controls) / len(matched_controls) if matched_controls else None
            association_id = "assoc_" + stable_hash(
                "eiel-mutation-outcome-association-v1",
                {"mutation_event_id": mutation_id, "metric": metric, "match_fields": list(required_match_fields)},
            )
            rows.append(
                {
                    "schema_version": ASSOCIATION_SCHEMA,
                    "association_id": association_id,
                    "mutation_event_id": mutation_id,
                    "outcome_metric": metric,
                    "status": status,
                    "reason": reason,
                    "required_match_fields": list(required_match_fields),
                    "carrier_count": len(matched_carriers),
                    "matched_non_carrier_count": len(matched_controls),
                    "carrier_mean": carrier_mean,
                    "matched_non_carrier_mean": control_mean,
                    "difference_carrier_minus_control": (
                        carrier_mean - control_mean
                        if carrier_mean is not None and control_mean is not None
                        else None
                    ),
                    "evidence_strength": "associated" if status == "available" else "observed",
                    "causal_status": "association_only" if status == "available" else "not_assessed",
                    "limitations": ["A matched passive association does not establish mutation benefit or causation."],
                }
            )
    return canonical_payload(rows)


def validate_evidence_strength(record: Mapping[str, Any]) -> None:
    strength = str(record.get("evidence_strength") or "")
    if strength not in EVIDENCE_RANK:
        raise GenomeAttributionError(f"invalid evidence strength: {strength}")
    if strength == "inherited" and not record.get("inheritance_record_id"):
        raise GenomeAttributionError("inherited evidence requires an inheritance record")
    if strength == "persisted" and int(record.get("persisted_generations", 0) or 0) <= 0:
        raise GenomeAttributionError("persisted evidence requires a positive generation duration")
    if strength == "associated" and not record.get("association_id"):
        raise GenomeAttributionError("associated evidence requires an association record")
    if strength == "enriched" and not record.get("enrichment_record_id"):
        raise GenomeAttributionError("enriched evidence requires a bounded enrichment record")
    if strength == "branch_supported" and not record.get("branch_comparison_refs"):
        raise GenomeAttributionError("branch-supported evidence requires branch comparison references")
    if strength == "experimentally_supported" and not record.get("experiment_refs"):
        raise GenomeAttributionError("experimentally-supported evidence requires formal experiment references")


def make_success_profile(
    *,
    subject_type: str,
    subject_id: str,
    run_id: str,
    measured_outcomes: Mapping[str, Any],
    strongest_deviations: Sequence[Mapping[str, Any]] = (),
    mutation_event_ids: Sequence[str] = (),
    ecology: Mapping[str, Any] | None = None,
    opportunities: Mapping[str, Any] | None = None,
    associations: Sequence[Mapping[str, Any]] = (),
    limitations: Sequence[str] = (),
    causal_status: str = "not_assessed",
) -> dict[str, Any]:
    if causal_status not in {"not_assessed", "association_only", "branch_supported"}:
        raise GenomeAttributionError("unsupported success-profile causal status")
    profile = {
        "schema_version": SUCCESS_PROFILE_SCHEMA,
        "success_profile_id": "sprof_" + stable_hash(
            "eiel-genome-success-profile-v1",
            {"run_id": run_id, "subject_type": subject_type, "subject_id": subject_id},
        ),
        "run_id": str(run_id),
        "subject_type": str(subject_type),
        "subject_id": str(subject_id),
        "measured_outcomes": dict(measured_outcomes),
        "strongest_deviations": [dict(value) for value in strongest_deviations],
        "mutation_event_ids": sorted({str(value) for value in mutation_event_ids}),
        "ecology": dict(ecology or {}),
        "opportunities": dict(opportunities or {}),
        "associations": [dict(value) for value in associations],
        "limitations": [str(value) for value in limitations],
        "causal_status": causal_status,
    }
    if _FORBIDDEN_CAUSAL_WORDING.search(canonical_json(profile)):
        raise GenomeAttributionError("success profile contains unsupported causal or benefit wording")
    return canonical_payload(profile)


def storage_preflight(
    *,
    source_record_count: int,
    estimated_delta_count: int,
    include_full_genomes: bool = False,
    acknowledge_large_output: bool = False,
) -> dict[str, Any]:
    records = max(0, int(source_record_count))
    deltas = max(0, int(estimated_delta_count))
    if records > 1_000_000 or deltas > 20_000_000:
        raise GenomeAttributionError("attribution output exceeds compact safety bounds")
    if include_full_genomes and not acknowledge_large_output:
        raise GenomeAttributionError("full-genome output requires acknowledge_large_output=true")
    estimated_bytes = records * 2_048 + deltas * 1_024
    if include_full_genomes:
        estimated_bytes += records * 128_000
    return canonical_payload(
        {
            "schema_version": "eiel_mutation_attribution_storage_preflight_v1",
            "accepted": True,
            "source_record_count": records,
            "estimated_delta_count": deltas,
            "include_full_genomes": bool(include_full_genomes),
            "acknowledge_large_output": bool(acknowledge_large_output),
            "estimated_bytes_upper_bound": estimated_bytes,
            "risk_classification": "large" if include_full_genomes or estimated_bytes > 512 * 1024 * 1024 else "compact",
        }
    )


class GenomeAttributionAccumulator:
    """Restart-safe, duplicate-rejecting collector for passive source records."""

    def __init__(self) -> None:
        self._sources: dict[str, dict[str, Any]] = {}

    @property
    def source_records(self) -> list[dict[str, Any]]:
        return [self._sources[key] for key in sorted(self._sources)]

    def add(self, record: Mapping[str, Any]) -> str:
        normalized = normalize_source_record(record)
        source_id = str(normalized["source_record_id"])
        encoded = canonical_json(normalized)
        if source_id in self._sources:
            if canonical_json(self._sources[source_id]) != encoded:
                raise GenomeAttributionError(f"source record ID collision: {source_id}")
            return source_id
        self._sources[source_id] = normalized
        return source_id

    def export_state(self) -> dict[str, Any]:
        return canonical_payload(
            {
                "schema_version": STATE_SCHEMA,
                "source_records": self.source_records,
                "source_record_count": len(self._sources),
            }
        )

    def import_state(self, state: Mapping[str, Any]) -> None:
        if str(state.get("schema_version")) != STATE_SCHEMA:
            raise GenomeAttributionError("unsupported attribution accumulator state")
        imported: dict[str, dict[str, Any]] = {}
        for value in state.get("source_records", []) or []:
            if not isinstance(value, Mapping):
                raise GenomeAttributionError("invalid attribution accumulator source record")
            source_id = str(value.get("source_record_id") or "")
            if not source_id:
                raise GenomeAttributionError("attribution accumulator source ID missing")
            parent_hashes = [str(parent.get("hash") or "") for parent in value.get("parents", []) or []]
            expected_id = "gsrc_" + stable_hash(
                "eiel-genome-attribution-source-v1",
                {
                    "run_id": value.get("run_id"),
                    "campaign_id": value.get("campaign_id"),
                    "branch_id": value.get("branch_id"),
                    "source_commit": value.get("source_commit", "unknown"),
                    "generation": value.get("generation"),
                    "step": value.get("step"),
                    "reproduction_phase": value.get("reproduction_phase"),
                    "parent_agent_ids": value.get("parent_agent_ids", []),
                    "child_agent_id": value.get("child_agent_id"),
                    "lineage_id": value.get("lineage_id"),
                    "source_record_refs": value.get("source_record_refs", []),
                    "parent_hashes": parent_hashes,
                    "pre_hash": dict(value.get("child_pre", {}) or {}).get("hash"),
                    "post_hash": dict(value.get("child_post", {}) or {}).get("hash"),
                    "final_hash": dict(value.get("child_final", {}) or {}).get("hash"),
                    "operator": value.get("mutation_operator"),
                },
            )
            if source_id != expected_id:
                raise GenomeAttributionError(f"attribution accumulator source identity mismatch: {source_id}")
            normalized = canonical_payload(dict(value))
            if source_id in imported and canonical_json(imported[source_id]) != canonical_json(normalized):
                raise GenomeAttributionError(f"attribution accumulator source ID collision: {source_id}")
            imported[source_id] = normalized
        if int(state.get("source_record_count", len(imported))) != len(imported):
            raise GenomeAttributionError("attribution accumulator state count mismatch")
        self._sources = imported

    def build(
        self,
        *,
        outcomes: Sequence[Mapping[str, Any]] = (),
        required_match_fields: Sequence[str] = DEFAULT_MATCH_FIELDS,
    ) -> dict[str, Any]:
        sources = self.source_records
        events: list[dict[str, Any]] = []
        deltas: list[dict[str, Any]] = []
        for source in sources:
            source_events, source_deltas = _source_events(source)
            events.extend(source_events)
            deltas.extend(source_deltas)
        events.sort(key=lambda value: (
            int(value["generation"]),
            int(value["step"]),
            int(value["child_agent_id"]),
            str(value.get("changed_path") or ""),
            str(value["mutation_event_id"]),
        ))
        deltas.sort(key=lambda value: (int(value["generation"]), int(value["child_agent_id"]), str(value.get("path") or ""), str(value["genome_delta_id"])))
        _mark_independent_similar(events)
        inheritance, carriers, carrier_generation = _inheritance(sources, events)
        persistence, carrier_summary = _persistence_and_carriers(
            sources, events, inheritance, carriers, carrier_generation
        )
        associations = _association_rows(
            events, carriers, outcomes, required_match_fields=tuple(required_match_fields)
        )
        return canonical_payload(
            {
                "source_records": sources,
                "mutation_events": events,
                "genome_deltas": deltas,
                "mutation_inheritance": inheritance,
                "mutation_persistence_summary": persistence,
                "mutation_carrier_summary": carrier_summary,
                "mutation_outcome_associations": associations,
            }
        )


def _association_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "schema_version", "association_id", "mutation_event_id", "outcome_metric", "status", "reason",
        "carrier_count", "matched_non_carrier_count", "carrier_mean", "matched_non_carrier_mean",
        "difference_carrier_minus_control", "evidence_strength", "causal_status", "required_match_fields", "limitations",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        encoded = dict(row)
        encoded["required_match_fields"] = canonical_json(row.get("required_match_fields", []))
        encoded["limitations"] = canonical_json(row.get("limitations", []))
        writer.writerow({key: encoded.get(key) for key in fields})
    return output.getvalue()


def _report(result: Mapping[str, Any], preflight: Mapping[str, Any], limitations: Sequence[str]) -> str:
    sources = result["source_records"]
    completeness = Counter(str(value.get("source_completeness", "unknown")) for value in sources)
    hash_only = sum(
        1
        for value in sources
        if value["child_pre"].get("payload") is None or value["child_post"].get("payload") is None
    )
    lines = [
        "# Chronicle GEN1-F0 Mutation Attribution",
        "",
        "Win/Loss: Passive attribution foundation implemented; causal mutation effects remain unclaimed.",
        "",
        f"- Authoritative source records: {len(sources)}",
        f"- Mutation-event records: {len(result['mutation_events'])}",
        f"- Canonical genome-delta records: {len(result['genome_deltas'])}",
        f"- Inheritance records: {len(result['mutation_inheritance'])}",
        f"- Hash-only source records: {hash_only}",
        f"- Storage risk: `{preflight['risk_classification']}`",
        "",
        "## Source Completeness",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(completeness.items()))
    lines.extend(
        [
            "",
            "## Evidence Strength",
            "",
            "Records use the ordered ladder: `observed`, `inherited`, `persisted`, `associated`, `enriched`, `branch_supported`, and `experimentally_supported`.",
            "F0 emits only levels supported by supplied passive records. Branch and experimental levels require separate formal references.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in limitations)
    return "\n".join(lines) + "\n"


def write_attribution_outputs(
    accumulator: GenomeAttributionAccumulator,
    output_dir: Path | str,
    *,
    outcomes: Sequence[Mapping[str, Any]] = (),
    required_match_fields: Sequence[str] = DEFAULT_MATCH_FIELDS,
    include_full_genomes: bool = False,
    acknowledge_large_output: bool = False,
) -> dict[str, Any]:
    result = accumulator.build(outcomes=outcomes, required_match_fields=required_match_fields)
    preflight = storage_preflight(
        source_record_count=len(result["source_records"]),
        estimated_delta_count=len(result["genome_deltas"]),
        include_full_genomes=include_full_genomes,
        acknowledge_large_output=acknowledge_large_output,
    )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "mutation_events.jsonl", result["mutation_events"])
    write_jsonl(output / "genome_deltas.jsonl", result["genome_deltas"])
    write_jsonl(output / "mutation_inheritance.jsonl", result["mutation_inheritance"])
    write_json(output / "mutation_persistence_summary.json", result["mutation_persistence_summary"])
    write_json(output / "mutation_carrier_summary.json", result["mutation_carrier_summary"])
    atomic_write_text(output / "mutation_outcome_associations.csv", _association_csv(result["mutation_outcome_associations"]))
    if include_full_genomes:
        write_jsonl(
            output / "genome_snapshots.jsonl",
            [
                {
                    "schema_version": "eiel_genome_attribution_source_snapshot_v1",
                    "source_record_id": source["source_record_id"],
                    "run_id": source["run_id"],
                    "generation": source["generation"],
                    "child_agent_id": source["child_agent_id"],
                    "parents": source["parents"],
                    "child_pre": source["child_pre"],
                    "child_post": source["child_post"],
                    "child_final": source["child_final"],
                    "source_record_refs": source["source_record_refs"],
                }
                for source in result["source_records"]
            ],
        )
    limitations = [
        "Only supplied authoritative genome snapshots or explicitly labelled canonical genome hashes are used.",
        "Native child hash-chain and mutation rows are policy/resident-row diagnostics, not canonical modular-genome hashes, and are not promoted to genome deltas.",
        "Identical numeric deltas from different source records remain independent similar mutations unless ancestry records establish inheritance.",
        "Passive associations do not establish that a mutation caused, benefited, taught, or strategically changed an agent or lineage.",
        "Full-genome artifacts are excluded by default and require explicit large-output acknowledgement.",
    ]
    atomic_write_text(output / "mutation_attribution_report.md", _report(result, preflight, limitations))
    artifact_rows = []
    for name in (*OUTPUT_NAMES, "genome_snapshots.jsonl"):
        path = output / name
        if path.name == "mutation_attribution_manifest.json" or not path.exists():
            continue
        artifact_rows.append({"relative_path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "source_record_count": len(result["source_records"]),
        "mutation_event_count": len(result["mutation_events"]),
        "genome_delta_count": len(result["genome_deltas"]),
        "inheritance_record_count": len(result["mutation_inheritance"]),
        "source_run_ids": sorted({str(value["run_id"]) for value in result["source_records"]}),
        "source_commits": sorted({str(value.get("source_commit", "unknown")) for value in result["source_records"]}),
        "source_completeness_counts": dict(sorted(Counter(str(value["source_completeness"]) for value in result["source_records"]).items())),
        "schemas": {
            "mutation_event": MUTATION_EVENT_SCHEMA,
            "genome_delta": GENOME_DELTA_SCHEMA,
            "inheritance": INHERITANCE_SCHEMA,
            "persistence": PERSISTENCE_SCHEMA,
            "carrier": CARRIER_SCHEMA,
            "association": ASSOCIATION_SCHEMA,
        },
        "evidence_strength_ladder": list(EVIDENCE_STRENGTHS),
        "canonical_genome_hash": "sha256_over_modular_genome_v1_canonical_bytes",
        "include_full_genomes": bool(include_full_genomes),
        "storage_preflight": preflight,
        "artifacts": artifact_rows,
        "limitations": limitations,
        "causal_status": "not_assessed_or_passive_association_only",
    }
    write_json(output / "mutation_attribution_manifest.json", manifest)
    return canonical_payload(
        {
            "output_dir": str(output),
            "manifest": manifest,
            "files": sorted(path.name for path in output.iterdir() if path.is_file()),
        }
    )


def load_source_jsonl(path: Path | str) -> GenomeAttributionAccumulator:
    source = Path(path).expanduser().resolve()
    accumulator = GenomeAttributionAccumulator()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GenomeAttributionError(f"invalid source JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise GenomeAttributionError(f"source JSONL line {line_number} is not an object")
            accumulator.add(value)
    return accumulator


def load_outcome_jsonl(path: Path | str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    source = Path(path).expanduser().resolve()
    rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GenomeAttributionError(f"invalid outcome JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, Mapping):
                raise GenomeAttributionError(f"outcome JSONL line {line_number} is not an object")
            rows.append(dict(value))
    return rows
