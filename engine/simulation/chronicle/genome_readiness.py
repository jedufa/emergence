from __future__ import annotations

"""Deterministic, read-only GEN1 retrospective attribution readiness.

The reader intentionally treats canonical genomes, policy parameters, structural
graphs, and native aggregates as separate identity scopes. It never imports a
runner or infers equivalence between those scopes.
"""

from collections import Counter, defaultdict
import csv
import io
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

from simulation.chronicle.genome_attribution import EVIDENCE_STRENGTHS
from simulation.chronicle.schema import canonical_json, canonical_payload, stable_hash
from simulation.chronicle.storage import atomic_write_text, sha256_file, write_json, write_jsonl


READINESS_SCHEMA = "eiel_chronicle_gen1_readiness_v1"
EVENT_SCHEMA = "eiel_gen1_mutation_event_inventory_v1"
IDENTITY_SCHEMA = "eiel_gen1_mutation_identity_map_v1"
CARRIER_SCHEMA = "eiel_gen1_carrier_record_v1"
PERSISTENCE_SCHEMA = "eiel_gen1_mutation_persistence_summary_v1"
ENRICHMENT_SCHEMA = "eiel_gen1_mutation_lineage_enrichment_v1"
ASSOCIATION_SCHEMA = "eiel_gen1_mutation_outcome_association_v1"
AGENT_PROFILE_SCHEMA = "eiel_gen1_agent_success_profile_v1"
LINEAGE_PROFILE_SCHEMA = "eiel_gen1_lineage_success_profile_v1"
COMPLETENESS_SCHEMA = "eiel_gen1_source_completeness_v1"
CONFOUND_SCHEMA = "eiel_gen1_confound_matrix_v1"
BRANCH_SCHEMA = "eiel_gen1_branch_test_candidates_v1"
DECISION_SCHEMA = "eiel_gen1_readiness_decision_v1"
COMPANION_SCHEMA = "eiel_gen1_chronicle_companion_index_v1"

DECISION_A = "CHRONICLE-GEN1-R1-A"
DECISION_B = "CHRONICLE-GEN1-R1-B"
DECISION_C = "CHRONICLE-GEN1-R1-C"
DECISION_D = "CHRONICLE-GEN1-R1-D"

OUTPUT_NAMES = (
    "mutation_event_inventory.jsonl",
    "mutation_identity_map.json",
    "carrier_records.jsonl",
    "mutation_persistence_summary.json",
    "mutation_lineage_enrichment.csv",
    "mutation_outcome_associations.csv",
    "agent_success_profiles.jsonl",
    "lineage_success_profiles.jsonl",
    "source_completeness.json",
    "confound_matrix.json",
    "branch_test_candidates.json",
    "gen1_readiness_decision.json",
    "gen1_readiness_report.md",
    "chronicle_companion_index.json",
    "gen1_readiness_manifest.json",
)

MATCH_HIERARCHY = (
    "sibling_or_same_parent",
    "same_lineage_and_generation",
    "same_scenario_and_seed",
    "similar_age_or_start_generation",
    "similar_initial_state",
    "similar_region_or_patch_exposure",
    "similar_opportunity_exposure",
    "similar_linked_mutation_burden",
)

_RECOGNIZED_NAMES = {
    "active_policy_stage_lineage_events.jsonl",
    "mutation_events.jsonl",
    "mutation_inheritance.jsonl",
    "mutation_persistence_summary.json",
    "mutation_carrier_summary.json",
    "mutation_outcome_associations.csv",
    "mutation_attribution_manifest.json",
    "rep2m_mutation_delta_events.csv",
    "polarch4_mutation_events.csv",
    "lineage_reproduction_funnel_events.jsonl",
    "formal_events.jsonl",
    "chronicle_manifest.json",
    "lineage_metrics.jsonl",
    "lineage_descriptors.jsonl",
}
_MAX_FILES = 10_000
_MAX_FILE_BYTES = 512 * 1024 * 1024
_MAX_JSONL_LINE_BYTES = 8 * 1024 * 1024
_FORBIDDEN_CAUSAL_WORDING = re.compile(
    r"\b(caused|beneficial|learned|adopted\s+(?:a\s+)?strategy|likely\s+cause|proved\s+effective)\b",
    re.IGNORECASE,
)


class GenomeReadinessError(ValueError):
    """Raised when retrospective selection or source identity is unsafe."""


def validate_no_causal_wording(value: Any) -> None:
    if _FORBIDDEN_CAUSAL_WORDING.search(canonical_json(value)):
        raise GenomeReadinessError("unsupported causal or benefit wording")


def _safe_explicit_path(value: Path | str) -> Path:
    raw = str(value)
    if not raw or "\x00" in raw or ".." in Path(raw).parts:
        raise GenomeReadinessError(f"unsafe source path: {value}")
    path = Path(raw).expanduser()
    if path.is_symlink():
        raise GenomeReadinessError(f"symlink sources are not accepted: {value}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GenomeReadinessError(f"source does not exist: {value}") from exc
    if not (resolved.is_file() or resolved.is_dir()):
        raise GenomeReadinessError(f"source is not a regular file or directory: {value}")
    return resolved


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _iter_tree_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    visited = 0
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirnames[:] = sorted(
            name for name in dirnames
            if not (current_path / name).is_symlink() and not name.startswith(".")
        )
        for name in sorted(filenames):
            visited += 1
            if visited > _MAX_FILES:
                raise GenomeReadinessError(f"source root exceeds {_MAX_FILES} files: {root}")
            path = current_path / name
            if path.is_symlink():
                continue
            if not path.is_file() or not _inside(path, root):
                raise GenomeReadinessError(f"source tree contains an unsafe file: {path}")
            yield path


def _selected_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return [path for path in _iter_tree_files(root) if path.name in _RECOGNIZED_NAMES]


def _tree_fingerprint(root: Path) -> dict[str, Any]:
    rows = []
    for path in _iter_tree_files(root):
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise GenomeReadinessError(f"source file exceeds bounded reader: {path}")
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        rows.append({"relative_path": relative, "bytes": size, "sha256": sha256_file(path)})
    return canonical_payload(
        {
            "file_count": len(rows),
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "files": rows,
            "tree_sha256": stable_hash("eiel-gen1-source-tree-v1", rows),
        }
    )


def _read_json(path: Path) -> Any:
    if path.stat().st_size > _MAX_FILE_BYTES:
        raise GenomeReadinessError(f"source file exceeds bounded reader: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenomeReadinessError(f"invalid JSON source {path}: {exc}") from exc


def _iter_jsonl(path: Path, maximum: int) -> Iterable[tuple[int, dict[str, Any]]]:
    if path.stat().st_size > _MAX_FILE_BYTES:
        raise GenomeReadinessError(f"source file exceeds bounded reader: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            record_count = 0
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                record_count += 1
                if record_count > maximum:
                    return
                if len(raw.encode("utf-8")) > _MAX_JSONL_LINE_BYTES:
                    raise GenomeReadinessError(f"oversized JSONL record: {path}:{line_number}")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise GenomeReadinessError(f"non-object JSONL record: {path}:{line_number}")
                yield line_number, value
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenomeReadinessError(f"invalid JSONL source {path}: {exc}") from exc


def _iter_csv(path: Path, maximum: int) -> Iterable[tuple[int, dict[str, str]]]:
    if path.stat().st_size > _MAX_FILE_BYTES:
        raise GenomeReadinessError(f"source file exceeds bounded reader: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=2):
                if index - 1 > maximum:
                    return
                yield index, dict(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise GenomeReadinessError(f"invalid CSV source {path}: {exc}") from exc


def _int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _json_cell(value: Any) -> Any:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _selection_id(path: Path, role: str) -> str:
    return "sel_" + stable_hash("eiel-gen1-selection-v1", {"path": str(path), "role": role})


def _source_ref(selection_id: str, path: Path, line: int | None = None) -> str:
    return f"{selection_id}:{path.name}" + (f":{line}" if line is not None else "")


def _readiness_event_id(selection_id: str, source_event_id: str) -> str:
    return "rmut_" + stable_hash(
        "eiel-gen1-readiness-mutation-v1",
        {"selection_id": selection_id, "source_event_id": source_event_id},
    )


def _base_event(
    *,
    selection_id: str,
    source_event_id: str,
    source_ref: str,
    source_role: str,
    source_scope: str,
    run_id: str,
    generation: int,
    parent_agent_ids: Sequence[Any],
    child_agent_id: Any,
    lineage_id: Any,
    mutation_operator: str,
    source_completeness: str,
) -> dict[str, Any]:
    return {
        "schema_version": EVENT_SCHEMA,
        "mutation_event_id": _readiness_event_id(selection_id, source_event_id),
        "source_mutation_event_id": source_event_id,
        "selection_id": selection_id,
        "source_role": source_role,
        "source_scope": source_scope,
        "run_id": run_id,
        "generation": generation,
        "parent_agent_ids": [str(value) for value in parent_agent_ids if str(value) not in {"", "-1"}],
        "child_agent_id": str(child_agent_id) if str(child_agent_id) not in {"", "-1"} else None,
        "lineage_id": str(lineage_id) if lineage_id not in {None, ""} else None,
        "mutation_operator": mutation_operator,
        "source_record_refs": [source_ref],
        "source_completeness": source_completeness,
        "evidence_strength": "observed",
        "causal_status": "not_assessed",
    }


def _f0_event(row: Mapping[str, Any], *, selection_id: str, role: str, ref: str) -> dict[str, Any]:
    if str(row.get("schema_version")) != "eiel_mutation_event_v1":
        raise GenomeReadinessError(f"unsupported F0 mutation schema at {ref}")
    source_event_id = str(row.get("mutation_event_id") or "")
    if not source_event_id:
        raise GenomeReadinessError(f"F0 mutation event identity missing at {ref}")
    event = _base_event(
        selection_id=selection_id,
        source_event_id=source_event_id,
        source_ref=ref,
        source_role=role,
        source_scope="canonical_modular_genome",
        run_id=str(row.get("run_id") or "unknown"),
        generation=_int(row.get("generation"), 0),
        parent_agent_ids=row.get("parent_agent_ids", []) or [],
        child_agent_id=row.get("child_agent_id"),
        lineage_id=row.get("lineage_id"),
        mutation_operator=str(row.get("mutation_operator") or "unknown"),
        source_completeness=str(row.get("source_completeness") or "unknown"),
    )
    event.update(
        {
            "campaign_id": row.get("campaign_id"),
            "branch_id": row.get("branch_id"),
            "step": _int(row.get("step")),
            "reproduction_phase": row.get("reproduction_phase"),
            "parent_genome_hashes": list(row.get("parent_genome_hashes", []) or []),
            "child_pre_mutation_hash": row.get("child_pre_mutation_genome_hash"),
            "child_post_mutation_hash": row.get("child_post_mutation_genome_hash"),
            "final_attached_child_hash": row.get("final_attached_child_genome_hash"),
            "changed_fields": [row.get("changed_path")] if row.get("changed_path") else [],
            "before_value": row.get("before_value") if row.get("before_available") else None,
            "after_value": row.get("after_value") if row.get("after_available") else None,
            "delta": row.get("delta"),
            "identity_relation": str(row.get("mutation_identity_relation") or "distinct_recorded_event"),
            "independent_similar_source_ids": list(row.get("independent_similar_mutation_event_ids", []) or []),
            "chronicle_event_id": row.get("chronicle_event_id"),
            "birth_chronicle_event_id": row.get("birth_chronicle_event_id"),
            "hash_reconciliation": "canonical_parent_pre_post_final_recorded",
        }
    )
    event["source_record_refs"] = sorted(
        {ref, *[str(value) for value in row.get("source_record_refs", []) or []]}
    )
    return canonical_payload(event)


def _policy_event(row: Mapping[str, Any], *, selection_id: str, role: str, ref: str) -> dict[str, Any] | None:
    if str(row.get("schema_version")) != "eiel_policy_lineage_event_v1" or not _bool(row.get("mutation_mutated")):
        return None
    seed = row.get("seed", "unknown")
    source_event_id = str(row.get("mutation_trace_id") or row.get("checkpoint_replay_sequence_id") or row.get("event_id"))
    event = _base_event(
        selection_id=selection_id,
        source_event_id=f"{seed}:{source_event_id}",
        source_ref=ref,
        source_role=role,
        source_scope="policy_parameters",
        run_id=str(row.get("run_id") or f"policy-lineage-seed-{seed}"),
        generation=_int(row.get("generation"), 0),
        parent_agent_ids=row.get("biological_parent_ids", []) or [],
        child_agent_id=row.get("child_agent_id"),
        lineage_id=row.get("child_biological_lineage_id"),
        mutation_operator=str(row.get("policy_mutation_mode") or "unknown"),
        source_completeness="policy_stage_hash_chain_without_parameter_values_or_descendant_carrier_join",
    )
    event.update(
        {
            "seed": seed,
            "parent_genome_hashes": [row.get("parent_policy_hash")] if row.get("parent_policy_hash") else [],
            "child_pre_mutation_hash": row.get("child_pre_mutation_policy_hash"),
            "child_post_mutation_hash": row.get("child_post_mutation_policy_hash"),
            "final_attached_child_hash": row.get("child_final_policy_hash"),
            "changed_fields": ["/policy/parameters"],
            "before_value": None,
            "after_value": None,
            "delta": {
                "mutated_parameter_count": _int(row.get("mutation_mutated_parameter_count"), 0),
                "parameter_count": _int(row.get("mutation_parameter_count"), 0),
                "full_policy_tensor_delta_norm": row.get("full_policy_tensor_delta_norm"),
            },
            "identity_relation": "distinct_recorded_event",
            "independent_similar_source_ids": [],
            "hash_reconciliation": (
                "policy_pre_post_final_reconciled"
                if row.get("child_post_mutation_policy_hash") == row.get("child_final_policy_hash")
                else "policy_final_stage_mismatch"
            ),
            "limitations": [
                "Policy hashes are not canonical modular-genome hashes.",
                "Aggregate parameter counts do not identify changed tensor values.",
            ],
        }
    )
    return canonical_payload(event)


def _rep2m_event(row: Mapping[str, Any], *, selection_id: str, role: str, ref: str) -> dict[str, Any] | None:
    if str(row.get("schema_version")) != "structural_mutation_delta_event_v1":
        return None
    if str(row.get("operator_status")) != "applied" or not _bool(row.get("semantic_changed")):
        return None
    source_event_id = str(row.get("mutation_attempt_event_id") or "")
    if not source_event_id:
        raise GenomeReadinessError(f"REP2M mutation attempt identity missing at {ref}")
    local = str(row.get("local_birth_event_id") or "")
    match = re.match(r"g([0-9]+)", local)
    event = _base_event(
        selection_id=selection_id,
        source_event_id=source_event_id,
        source_ref=ref,
        source_role=role,
        source_scope="structural_policy_graph",
        run_id=str(row.get("run_id") or "unknown"),
        generation=int(match.group(1)) if match else -1,
        parent_agent_ids=[],
        child_agent_id=row.get("child_id"),
        lineage_id=None,
        mutation_operator=str(row.get("operator_name") or "unknown"),
        source_completeness="structural_pre_post_digest_without_parent_ancestry_or_descendant_carrier_join",
    )
    changed = [
        name for name in ("expression_changed", "historical_changed", "parameter_changed", "topology_changed")
        if _bool(row.get(name))
    ]
    event.update(
        {
            "parent_genome_hashes": [],
            "child_pre_mutation_hash": row.get("pre_graph_semantic_digest"),
            "child_post_mutation_hash": row.get("post_graph_semantic_digest"),
            "final_attached_child_hash": None,
            "changed_fields": [f"/structural_graph/{name}" for name in changed],
            "before_value": row.get("pre_graph_identity_digest"),
            "after_value": row.get("post_graph_identity_digest"),
            "delta": {"rng_draw_delta": _int(row.get("rng_draw_delta"), 0)},
            "identity_relation": "distinct_recorded_event",
            "independent_similar_source_ids": [],
            "birth_event_id": row.get("global_birth_event_id"),
            "hash_reconciliation": "structural_pre_post_digest_only",
            "limitations": [
                "Structural graph digests are not canonical modular-genome hashes.",
                "The selected mutation row lacks parent identity and final attached child identity.",
            ],
        }
    )
    return canonical_payload(event)


def _polarch_event(row: Mapping[str, Any], *, selection_id: str, role: str, ref: str) -> dict[str, Any] | None:
    if not row.get("mutation_event_id") or not row.get("pre_mutation_vector"):
        return None
    if not _bool(row.get("semantic_change")) or _bool(row.get("no_op")):
        return None
    source_event_id = str(row["mutation_event_id"])
    event = _base_event(
        selection_id=selection_id,
        source_event_id=source_event_id,
        source_ref=ref,
        source_role=role,
        source_scope="policy_family_head_vector",
        run_id=str(row.get("run_id") or "unknown"),
        generation=_int(row.get("generation"), -1),
        parent_agent_ids=[row.get("parent_id"), row.get("mate_id")],
        child_agent_id=row.get("child_id"),
        lineage_id=None,
        mutation_operator=str(row.get("operator_id") or "unknown"),
        source_completeness="exact_family_head_vector_delta_without_biological_lineage_carrier_join",
    )
    event.update(
        {
            "seed": row.get("seed"),
            "parent_genome_hashes": [
                value for value in (row.get("parent_a_semantic_digest"), row.get("parent_b_semantic_digest")) if value
            ],
            "child_pre_mutation_hash": row.get("post_recombination_semantic_digest"),
            "child_post_mutation_hash": row.get("semantic_digest"),
            "final_attached_child_hash": row.get("canonical_digest"),
            "changed_fields": ["/policy/family_head_vector"],
            "before_value": _json_cell(row.get("pre_mutation_vector")),
            "after_value": _json_cell(row.get("final_effective_vector")),
            "delta": _json_cell(row.get("candidate_raw_vector")),
            "identity_relation": "distinct_recorded_event",
            "independent_similar_source_ids": [],
            "reproduction_event_id": row.get("reproduction_event_id"),
            "hash_reconciliation": "family_head_pre_post_final_vector_recorded",
            "limitations": [
                "Family-head semantic digests are not canonical modular-genome hashes.",
                "Biological lineage and descendant carrier identity are not present in this selected artifact.",
            ],
        }
    )
    return canonical_payload(event)


def _source_kind(path: Path, first: Mapping[str, Any] | None = None) -> str:
    if path.name == "mutation_events.jsonl":
        return "gen1_f0_mutation_events"
    if path.name == "mutation_inheritance.jsonl":
        return "gen1_f0_inheritance"
    if path.name == "mutation_persistence_summary.json":
        return "gen1_f0_persistence"
    if path.name == "mutation_carrier_summary.json":
        return "gen1_f0_carriers"
    if path.name == "mutation_outcome_associations.csv":
        return "gen1_f0_associations"
    if path.name == "mutation_attribution_manifest.json":
        return "gen1_f0_manifest"
    if path.name == "rep2m_mutation_delta_events.csv":
        return "rep2m_structural_mutation_delta"
    if path.name == "polarch4_mutation_events.csv":
        return "polarch_family_head_mutation"
    schema = str((first or {}).get("schema_version") or "")
    if schema == "eiel_policy_lineage_event_v1" or path.name == "active_policy_stage_lineage_events.jsonl":
        return "policy_stage_hash_chain"
    if schema == "eiel_formal_event_v1" or path.name == "formal_events.jsonl":
        return "chronicle_formal_events"
    if path.name == "chronicle_manifest.json":
        return "chronicle_manifest"
    if path.name in {"lineage_metrics.jsonl", "lineage_descriptors.jsonl"}:
        return "native_or_lineage_aggregate"
    if path.name == "lineage_reproduction_funnel_events.jsonl":
        return "lineage_outcome_context"
    return "unrecognized_selected_source"


def _rng_fingerprints() -> dict[str, str | None]:
    values: dict[str, str | None] = {
        "python": stable_hash("python-rng-state", repr(random.getstate())),
        "numpy": None,
        "torch": None,
    }
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        state = numpy.random.get_state()
        values["numpy"] = stable_hash(
            "numpy-rng-state",
            [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])],
        )
    torch = sys.modules.get("torch")
    if torch is not None:
        values["torch"] = stable_hash("torch-rng-state", torch.random.get_rng_state().tolist())
    return values


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        encoded = {
            key: canonical_json(value) if isinstance(value, (dict, list, tuple)) else value
            for key, value in row.items()
        }
        writer.writerow({key: encoded.get(key) for key in fields})
    return output.getvalue()


def _unavailable_association(event: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        "schema_version": ASSOCIATION_SCHEMA,
        "association_id": "rassoc_" + stable_hash(
            "eiel-gen1-unavailable-association-v1", {"mutation_event_id": event["mutation_event_id"], "reason": reason}
        ),
        "mutation_event_id": event["mutation_event_id"],
        "selection_id": event["selection_id"],
        "source_role": event["source_role"],
        "outcome_metric": "unavailable",
        "status": "unavailable",
        "reason": reason,
        "carrier_group": [],
        "control_group": [],
        "matching_dimensions": list(MATCH_HIERARCHY),
        "unmatched_dimensions": list(MATCH_HIERARCHY),
        "carrier_outcome": None,
        "control_outcome": None,
        "difference_carrier_minus_control": None,
        "coverage": 0.0,
        "evidence_strength": "observed",
        "causal_status": "not_assessed",
        "limitations": ["Matched carrier and non-carrier controls were not available in the selected source."],
    }


def _profile_rows(events: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_agent: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_lineage: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        agent = event.get("child_agent_id")
        lineage = event.get("lineage_id")
        if agent is not None:
            by_agent[(str(event["selection_id"]), str(event["run_id"]), str(agent))].append(event)
        if lineage is not None:
            by_lineage[(str(event["selection_id"]), str(event["run_id"]), str(lineage))].append(event)
    agent_rows = []
    for (selection_id, run_id, agent_id), values in sorted(by_agent.items()):
        mutation_ids = sorted(str(value["mutation_event_id"]) for value in values)
        mutation_burden = sum(
            max(1, _int(dict(value.get("delta") or {}).get("mutated_parameter_count"), 1))
            if isinstance(value.get("delta"), Mapping) else 1
            for value in values
        )
        row = {
            "schema_version": AGENT_PROFILE_SCHEMA,
            "success_profile_id": "ragent_" + stable_hash(
                "eiel-gen1-agent-profile-v1",
                {"selection_id": selection_id, "run_id": run_id, "agent_id": agent_id},
            ),
            "selection_id": selection_id,
            "source_role": values[0].get("source_role"),
            "run_id": run_id,
            "agent_id": agent_id,
            "lineage_id": next((value.get("lineage_id") for value in values if value.get("lineage_id")), None),
            "status": "partial_no_success_outcome_measure",
            "notability_rule": "top_five_recorded_mutation_burden_within_explicit_selection_and_run",
            "notability_score": mutation_burden,
            "measured_outcomes": {},
            "mutation_event_ids": mutation_ids,
            "ecology": {},
            "opportunities": {},
            "association_ids": [],
            "confounds": ["parent_quality", "opportunity", "ecological_exposure", "linked_mutations"],
            "causal_status": "not_assessed",
            "limitations": ["No matched success-outcome record was joined for this agent."],
        }
        validate_no_causal_wording(row)
        agent_rows.append(canonical_payload(row))
    lineage_rows = []
    for (selection_id, run_id, lineage_id), values in sorted(by_lineage.items()):
        mutation_ids = sorted(str(value["mutation_event_id"]) for value in values)
        row = {
            "schema_version": LINEAGE_PROFILE_SCHEMA,
            "success_profile_id": "rlineage_" + stable_hash(
                "eiel-gen1-lineage-profile-v1",
                {"selection_id": selection_id, "run_id": run_id, "lineage_id": lineage_id},
            ),
            "selection_id": selection_id,
            "source_role": values[0].get("source_role"),
            "run_id": run_id,
            "lineage_id": lineage_id,
            "status": "partial_no_success_outcome_measure",
            "notability_rule": "top_three_recorded_mutation_count_within_explicit_selection_and_run",
            "notability_score": len(mutation_ids),
            "measured_outcomes": {},
            "mutation_event_ids": mutation_ids,
            "ecology": {},
            "opportunities": {},
            "association_ids": [],
            "confounds": ["lineage_sweep", "founder_effect", "opportunity", "ecological_exposure"],
            "causal_status": "not_assessed",
            "limitations": ["No lineage denominator or matched success-outcome record was joined."],
        }
        validate_no_causal_wording(row)
        lineage_rows.append(canonical_payload(row))

    selected_agents = []
    grouped_agents: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in agent_rows:
        grouped_agents[(str(row["selection_id"]), str(row["run_id"]))].append(row)
    for key in sorted(grouped_agents):
        ranked = sorted(
            grouped_agents[key],
            key=lambda value: (-int(value["notability_score"]), str(value["agent_id"]), str(value["success_profile_id"])),
        )
        for rank, row in enumerate(ranked[:5], start=1):
            selected_agents.append(canonical_payload({**row, "notability_rank_within_run": rank}))

    selected_lineages = []
    grouped_lineages: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in lineage_rows:
        grouped_lineages[(str(row["selection_id"]), str(row["run_id"]))].append(row)
    for key in sorted(grouped_lineages):
        ranked = sorted(
            grouped_lineages[key],
            key=lambda value: (-int(value["notability_score"]), str(value["lineage_id"]), str(value["success_profile_id"])),
        )
        for rank, row in enumerate(ranked[:3], start=1):
            selected_lineages.append(canonical_payload({**row, "notability_rank_within_run": rank}))
    return selected_agents, selected_lineages


def _confound_matrix() -> dict[str, Any]:
    rows = [
        ("geographic_luck", "required", "region or patch exposure plus matched controls"),
        ("parent_quality", "required", "sibling or same-parent control"),
        ("opportunity", "required", "reproduction and survival opportunity exposure"),
        ("linked_mutations", "required", "complete mutation burden at the comparison boundary"),
        ("lineage_sweep", "required", "lineage share and ancestry-aware non-carriers"),
        ("initial_state", "required", "matched energy, health, age, and start generation"),
        ("scenario_seed", "required", "same scenario and seed or declared branch pair"),
        ("ecological_exposure", "required", "food, movement cost, predator, hazard, injury, and region exposure"),
    ]
    return {
        "schema_version": CONFOUND_SCHEMA,
        "matching_hierarchy": list(MATCH_HIERARCHY),
        "confounds": [
            {"confound": name, "status": status, "required_evidence": evidence}
            for name, status, evidence in rows
        ],
        "silent_matching_degradation_allowed": False,
    }


def _branch_candidates(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, tuple[str, ...]], list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("source_role") != "evidence":
            continue
        key = (
            str(event.get("source_scope")),
            str(event.get("mutation_operator")),
            tuple(str(value) for value in event.get("changed_fields", []) or []),
        )
        groups[key].append(event)
    rows = []
    for key, values in sorted(groups.items())[:50]:
        scope, operator, changed_fields = key
        ids = sorted(str(value["mutation_event_id"]) for value in values)
        row = {
            "branch_test_candidate_id": "btest_" + stable_hash(
                "eiel-gen1-branch-candidate-v1", {"scope": scope, "operator": operator, "fields": changed_fields, "ids": ids}
            ),
            "status": "not_executed",
            "source_scope": scope,
            "mutation_operator": operator,
            "changed_fields": list(changed_fields),
            "source_mutation_event_ids": ids[:25],
            "checkpoint_boundary": "immediately_before_the_recorded_reproduction_or_mutation_boundary",
            "observed_branch": "preserve_the_recorded_child_state",
            "matched_branch": "restore_only_the_recorded_pre_mutation_value_or_parameter_state",
            "required_controls": list(MATCH_HIERARCHY),
            "required_repeats": "bounded_declared_seed_set_with_repeat_parity",
            "candidate_outcomes": ["survival", "successful_births", "surviving_offspring", "ecological_exposure", "lineage_persistence"],
            "evidence_strength": "observed",
            "causal_status": "candidate_only_not_branch_supported",
            "limitations": [
                "No branch was executed in GEN1-R1.",
                "Post-intervention RNG and population divergence must be recorded rather than assumed equivalent.",
            ],
        }
        validate_no_causal_wording(row)
        rows.append(canonical_payload(row))
    return {"schema_version": BRANCH_SCHEMA, "candidates": rows}


def _companion_index(
    events: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    lineages: Sequence[Mapping[str, Any]],
    branch_candidates: Mapping[str, Any],
) -> dict[str, Any]:
    by_run: dict[str, dict[str, Any]] = {}
    for event in events:
        run = by_run.setdefault(
            str(event.get("run_id") or "unknown"),
            {"chronicle_events": defaultdict(list), "birth_events": defaultdict(list), "agents": defaultdict(list), "lineages": defaultdict(list)},
        )
        event_id = str(event["mutation_event_id"])
        if event.get("chronicle_event_id"):
            run["chronicle_events"][str(event["chronicle_event_id"])].append(event_id)
        if event.get("birth_chronicle_event_id"):
            run["birth_events"][str(event["birth_chronicle_event_id"])].append(event_id)
        if event.get("child_agent_id") is not None:
            run["agents"][str(event["child_agent_id"])].append(event_id)
        if event.get("lineage_id") is not None:
            run["lineages"][str(event["lineage_id"])].append(event_id)
    agent_profiles = defaultdict(list)
    for row in agents:
        agent_profiles[(str(row["run_id"]), str(row["agent_id"]))].append(str(row["success_profile_id"]))
    lineage_profiles = defaultdict(list)
    for row in lineages:
        lineage_profiles[(str(row["run_id"]), str(row["lineage_id"]))].append(str(row["success_profile_id"]))
    normalized_runs = {}
    for run_id, indexes in sorted(by_run.items()):
        normalized_runs[run_id] = {
            "chronicle_event_mutations": {key: sorted(set(value)) for key, value in sorted(indexes["chronicle_events"].items())},
            "birth_event_mutations": {key: sorted(set(value)) for key, value in sorted(indexes["birth_events"].items())},
            "agent_mutations": {key: sorted(set(value)) for key, value in sorted(indexes["agents"].items())},
            "lineage_mutations": {key: sorted(set(value)) for key, value in sorted(indexes["lineages"].items())},
            "agent_profiles": {
                key[1]: sorted(value) for key, value in sorted(agent_profiles.items()) if key[0] == run_id
            },
            "lineage_profiles": {
                key[1]: sorted(value) for key, value in sorted(lineage_profiles.items()) if key[0] == run_id
            },
        }
    return {
        "schema_version": COMPANION_SCHEMA,
        "attachment_mode": "external_read_only_companion",
        "frozen_chronicle_artifacts_modified": False,
        "runs": normalized_runs,
        "record_artifacts": {
            "mutation_events": "mutation_event_inventory.jsonl",
            "carriers": "carrier_records.jsonl",
            "persistence": "mutation_persistence_summary.json",
            "enrichment": "mutation_lineage_enrichment.csv",
            "associations": "mutation_outcome_associations.csv",
            "agent_profiles": "agent_success_profiles.jsonl",
            "lineage_profiles": "lineage_success_profiles.jsonl",
            "branch_candidates": "branch_test_candidates.json",
            "source_completeness": "source_completeness.json",
        },
        "branch_test_candidate_ids": sorted(
            str(value["branch_test_candidate_id"]) for value in branch_candidates.get("candidates", []) or []
        ),
    }


def _decision(
    events: Sequence[Mapping[str, Any]],
    carriers: Sequence[Mapping[str, Any]],
    persistence: Sequence[Mapping[str, Any]],
    associations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    real_events = [value for value in events if value.get("source_role") == "evidence"]
    real_ids = {str(value["mutation_event_id"]) for value in real_events}
    inherited = [
        value for value in carriers
        if value.get("evidence_strength") == "inherited" and str(value.get("mutation_event_id")) in real_ids
    ]
    persisted = [
        value for value in persistence
        if value.get("status") == "persisted" and str(value.get("mutation_event_id")) in real_ids
    ]
    available_associations = [
        value for value in associations
        if value.get("status") == "available" and str(value.get("mutation_event_id")) in real_ids
    ]
    canonical_ids = {
        str(value["mutation_event_id"])
        for value in real_events
        if value.get("source_scope") == "canonical_modular_genome"
    }
    inherited_ids = {str(value["mutation_event_id"]) for value in inherited}
    persisted_ids = {str(value["mutation_event_id"]) for value in persisted}
    associated_ids = {str(value["mutation_event_id"]) for value in available_associations}
    complete_chain_ids = canonical_ids.intersection(inherited_ids, persisted_ids)
    associated_chain_ids = complete_chain_ids.intersection(associated_ids)
    if associated_chain_ids:
        code = DECISION_A
        reason = "Real selected artifacts support canonical occurrence, inheritance, persistence, and bounded matched association."
    elif complete_chain_ids:
        code = DECISION_B
        reason = "Real selected artifacts support canonical occurrence, inheritance, and persistence, but matched associations are insufficient."
    elif real_events:
        code = DECISION_C
        reason = "Real selected artifacts support partial mutation reconstruction, but canonical genome scope or descendant ancestry is incomplete."
    else:
        code = DECISION_D
        reason = "No reliable mutation occurrence with stable event identity was present in the real selected artifacts."
    return {
        "schema_version": DECISION_SCHEMA,
        "decision": code,
        "reason": reason,
        "counts_excluding_technical_fixtures": {
            "mutation_events": len(real_events),
            "canonical_modular_genome_events": len(canonical_ids),
            "inheritance_links": len(inherited),
            "persistence_chains": len(persisted),
            "available_matched_associations": len(available_associations),
            "complete_canonical_chain_mutations": len(complete_chain_ids),
            "associated_complete_chain_mutations": len(associated_chain_ids),
        },
        "technical_fixtures_excluded_from_decision": True,
        "evidence_strength_levels_reached": [
            strength for strength in EVIDENCE_STRENGTHS
            if strength in {
                *({"observed"} if real_events else set()),
                *({"inherited"} if inherited else set()),
                *({"persisted"} if persisted else set()),
                *({"associated"} if available_associations else set()),
            }
        ],
        "runtime_instrumentation_prerequisite": (
            "Emit one passive versioned birth-boundary record containing canonical modular-genome parent payload/hash, "
            "child pre-mutation payload/hash, child post-mutation payload/hash, final attached child payload/hash, "
            "stable mutation-event ID, parent/child/lineage ancestry IDs, Chronicle/birth source IDs, and checkpoint-"
            "restored continuation identity; separately retain per-agent outcome and exposure rows needed by every "
            "declared matching dimension. Prove the capture path is RNG-neutral and trajectory-neutral in the owner branch."
        ),
        "causal_status": "not_assessed",
    }


def _report(
    decision: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    carriers: Sequence[Mapping[str, Any]],
    persistence: Sequence[Mapping[str, Any]],
    associations: Sequence[Mapping[str, Any]],
    agents: Sequence[Mapping[str, Any]],
    lineages: Sequence[Mapping[str, Any]],
    branches: Mapping[str, Any],
) -> str:
    real_events = [value for value in events if value.get("source_role") == "evidence"]
    unavailable = sum(1 for value in associations if value.get("status") != "available")
    return "\n".join(
        [
            "# Chronicle GEN1-R1 Attribution Readiness",
            "",
            f"Decision: `{decision['decision']}`",
            "",
            "Win/Loss: bounded retrospective reconstruction completed; canonical live ancestry remains incomplete.",
            "",
            "## Selected Evidence",
            "",
            f"- Explicit source selections: {len(sources)}",
            f"- Real mutation occurrences reconstructed: {len(real_events)}",
            f"- Technical-fixture mutation occurrences: {len(events) - len(real_events)}",
            f"- Inherited carrier links: {sum(1 for value in carriers if value.get('evidence_strength') == 'inherited')}",
            f"- Persisted chains: {sum(1 for value in persistence if value.get('status') == 'persisted')}",
            f"- Available matched associations: {sum(1 for value in associations if value.get('status') == 'available')}",
            f"- Unavailable associations: {unavailable}",
            f"- Agent profiles: {len(agents)}",
            f"- Lineage profiles: {len(lineages)}",
            f"- Branch-test candidates: {len(branches.get('candidates', []))}",
            "",
            "## Interpretation",
            "",
            str(decision["reason"]),
            "Each source selection remains separate. Canonical modular genomes, policy parameters, structural graphs, "
            "family-head vectors, and native aggregates are never pooled as one identity space.",
            "Unavailable association rows retain every required matching dimension instead of weakening controls.",
            "",
            "## Missing Instrumentation",
            "",
            str(decision["runtime_instrumentation_prerequisite"]),
            "",
            "## No-Claims Statement",
            "",
            "These retrospective records do not establish mutation benefit, causation, learning, strategy, intelligence, "
            "emergence, consciousness, or adaptation. Branch-supported and experimentally-supported evidence remain unused.",
            "",
        ]
    )


def write_readiness_outputs(
    source_roots: Sequence[Path | str],
    output_dir: Path | str,
    *,
    selection_reason: str,
    fixture_roots: Sequence[Path | str] = (),
    maximum_records_per_source: int = 100_000,
) -> dict[str, Any]:
    if not str(selection_reason).strip():
        raise GenomeReadinessError("selection_reason is required; implicit pooling is forbidden")
    maximum = int(maximum_records_per_source)
    if maximum < 1 or maximum > 1_000_000:
        raise GenomeReadinessError("maximum_records_per_source must be between 1 and 1000000")
    selections: list[tuple[Path, str]] = []
    for role, values in (("evidence", source_roots), ("technical_fixture", fixture_roots)):
        for value in values:
            path = _safe_explicit_path(value)
            if any(path == existing for existing, _role in selections):
                raise GenomeReadinessError(f"duplicate source selection: {path}")
            selections.append((path, role))
    if not selections:
        raise GenomeReadinessError("at least one explicit source root is required")
    selections.sort(key=lambda value: (value[1], str(value[0])))
    output = Path(output_dir).expanduser().resolve()
    for source, _role in selections:
        if source == output or (source.is_dir() and _inside(output, source)):
            raise GenomeReadinessError("output directory must be outside selected source roots")

    rng_before = _rng_fingerprints()
    before_trees = {str(path): _tree_fingerprint(path) for path, _role in selections}
    events: list[dict[str, Any]] = []
    inherited_rows: list[tuple[str, str, Mapping[str, Any]]] = []
    persistence_rows: list[tuple[str, str, Mapping[str, Any]]] = []
    carrier_summaries: list[tuple[str, str, Mapping[str, Any]]] = []
    source_associations: list[tuple[str, str, Mapping[str, Any]]] = []
    source_rows: list[dict[str, Any]] = []
    original_to_readiness: dict[tuple[str, str], str] = {}

    for root, role in selections:
        selection_id = _selection_id(root, role)
        files = _selected_files(root)
        adapters: Counter[str] = Counter()
        selected_record_count = 0
        emitted_event_count = 0
        capabilities: set[str] = set()
        limitations: set[str] = set()
        truncated_files: list[str] = []
        for path in files:
            first: Mapping[str, Any] | None = None
            if path.suffix == ".jsonl":
                materialized = list(_iter_jsonl(path, maximum + 1))
                if len(materialized) > maximum:
                    truncated_files.append(path.name)
                    materialized = materialized[:maximum]
                first = materialized[0][1] if materialized else None
                kind = _source_kind(path, first)
                adapters[kind] += 1
                selected_record_count += len(materialized)
                for line_number, row in materialized:
                    ref = _source_ref(selection_id, path, line_number)
                    event = None
                    if kind == "gen1_f0_mutation_events":
                        event = _f0_event(row, selection_id=selection_id, role=role, ref=ref)
                        capabilities.add("canonical_mutation_occurrence")
                    elif kind == "policy_stage_hash_chain":
                        event = _policy_event(row, selection_id=selection_id, role=role, ref=ref)
                        capabilities.add("policy_mutation_occurrence")
                        limitations.add("policy_hashes_are_not_canonical_modular_genome_hashes")
                    elif kind in {"native_or_lineage_aggregate", "lineage_outcome_context"}:
                        capabilities.add("aggregate_or_outcome_context_only")
                        limitations.add("aggregate_rows_lack_exact_mutation_event_and_descendant_carrier_identity")
                    elif kind == "chronicle_formal_events":
                        capabilities.add("chronicle_event_context")
                        limitations.add("frozen_chronicle_events_do_not_contain_canonical_parent_pre_post_final_genomes")
                    if event is not None:
                        events.append(event)
                        emitted_event_count += 1
                        original_to_readiness[(selection_id, str(event["source_mutation_event_id"]))] = str(event["mutation_event_id"])
                    if kind == "gen1_f0_inheritance":
                        inherited_rows.append((selection_id, role, row))
                        capabilities.add("mutation_inheritance")
            elif path.suffix == ".csv":
                materialized_csv = list(_iter_csv(path, maximum + 1))
                if len(materialized_csv) > maximum:
                    truncated_files.append(path.name)
                    materialized_csv = materialized_csv[:maximum]
                first = materialized_csv[0][1] if materialized_csv else None
                kind = _source_kind(path, first)
                adapters[kind] += 1
                selected_record_count += len(materialized_csv)
                for line_number, row in materialized_csv:
                    ref = _source_ref(selection_id, path, line_number)
                    event = None
                    if kind == "rep2m_structural_mutation_delta":
                        event = _rep2m_event(row, selection_id=selection_id, role=role, ref=ref)
                        capabilities.add("structural_mutation_occurrence")
                        limitations.add("structural_digests_are_not_canonical_modular_genome_hashes")
                    elif kind == "polarch_family_head_mutation":
                        event = _polarch_event(row, selection_id=selection_id, role=role, ref=ref)
                        capabilities.add("family_head_mutation_occurrence")
                        limitations.add("family_head_digests_are_not_canonical_modular_genome_hashes")
                    elif kind == "gen1_f0_associations":
                        source_associations.append((selection_id, role, row))
                        capabilities.add("matched_outcome_association")
                    if event is not None:
                        events.append(event)
                        emitted_event_count += 1
                        original_to_readiness[(selection_id, str(event["source_mutation_event_id"]))] = str(event["mutation_event_id"])
            elif path.suffix == ".json":
                payload = _read_json(path)
                first = payload if isinstance(payload, Mapping) else None
                kind = _source_kind(path, first)
                adapters[kind] += 1
                selected_record_count += 1
                if kind == "gen1_f0_persistence" and isinstance(payload, Mapping):
                    for row in payload.get("mutations", []) or []:
                        if isinstance(row, Mapping):
                            persistence_rows.append((selection_id, role, row))
                    capabilities.add("mutation_persistence")
                elif kind == "gen1_f0_carriers" and isinstance(payload, Mapping):
                    for row in payload.get("mutations", []) or []:
                        if isinstance(row, Mapping):
                            carrier_summaries.append((selection_id, role, row))
                    capabilities.add("carrier_summary")
                elif kind == "chronicle_manifest":
                    capabilities.add("chronicle_provenance")
                elif kind == "gen1_f0_manifest":
                    capabilities.add("gen1_f0_provenance")
        if not files:
            limitations.add("no_recognized_attribution_artifact_in_selected_root")
        source_rows.append(
            canonical_payload(
                {
                    "selection_id": selection_id,
                    "role": role,
                    "path": str(root),
                    "selection_reason": str(selection_reason),
                    "selected_file_count": len(files),
                    "selected_record_count": selected_record_count,
                    "emitted_mutation_event_count": emitted_event_count,
                    "bounded_reader_truncated": bool(truncated_files),
                    "truncated_files": sorted(set(truncated_files)),
                    "adapters": dict(sorted(adapters.items())),
                    "capabilities": sorted(capabilities),
                    "limitations": sorted(limitations),
                    "tree_before": before_trees[str(root)],
                }
            )
        )

    events.sort(
        key=lambda value: (
            str(value["selection_id"]), str(value["run_id"]), _int(value.get("generation")),
            str(value.get("child_agent_id") or ""), str(value["mutation_event_id"]),
        )
    )
    by_id = {str(value["mutation_event_id"]): value for value in events}
    identity_rows = []
    for event in events:
        related = [
            original_to_readiness[(str(event["selection_id"]), str(source_id))]
            for source_id in event.get("independent_similar_source_ids", []) or []
            if (str(event["selection_id"]), str(source_id)) in original_to_readiness
        ]
        identity_rows.append(
            {
                "mutation_event_id": event["mutation_event_id"],
                "selection_id": event["selection_id"],
                "source_scope": event["source_scope"],
                "source_mutation_event_id": event["source_mutation_event_id"],
                "identity_relation": event.get("identity_relation", "distinct_recorded_event"),
                "independent_similar_mutation_event_ids": sorted(related),
                "same_locus_different_value_event_ids": [],
                "unknown_identity_event_ids": [],
                "ancestry_required_for_merge": True,
            }
        )

    carriers = []
    for event in events:
        carriers.append(
            {
                "schema_version": CARRIER_SCHEMA,
                "carrier_record_id": "rcar_" + stable_hash(
                    "eiel-gen1-origin-carrier-v1", {"mutation_event_id": event["mutation_event_id"], "agent": event.get("child_agent_id")}
                ),
                "mutation_event_id": event["mutation_event_id"],
                "selection_id": event["selection_id"],
                "run_id": event["run_id"],
                "agent_id": event.get("child_agent_id"),
                "lineage_id": event.get("lineage_id"),
                "generation": event.get("generation"),
                "inherited_from_agent_id": None,
                "genome_hash": event.get("final_attached_child_hash"),
                "carrier_status": "observed_origin",
                "source_completeness": event.get("source_completeness"),
                "evidence_strength": "observed",
            }
        )
    for selection_id, _role, row in inherited_rows:
        mutation_id = original_to_readiness.get((selection_id, str(row.get("mutation_event_id"))))
        if not mutation_id or mutation_id not in by_id:
            continue
        event = by_id[mutation_id]
        status = str(row.get("status") or "unavailable")
        strength = "inherited" if status == "inherited" else "observed"
        carriers.append(
            {
                "schema_version": CARRIER_SCHEMA,
                "carrier_record_id": "rcar_" + stable_hash(
                    "eiel-gen1-inherited-carrier-v1",
                    {"selection_id": selection_id, "source_id": row.get("inheritance_record_id")},
                ),
                "mutation_event_id": mutation_id,
                "selection_id": selection_id,
                "run_id": event["run_id"],
                "agent_id": str(row.get("carrier_agent_id")),
                "lineage_id": row.get("lineage_id"),
                "generation": _int(row.get("generation")),
                "inherited_from_agent_id": str(row.get("inherited_from_agent_id")) if row.get("inherited_from_agent_id") is not None else None,
                "genome_hash": None,
                "carrier_status": status,
                "source_completeness": event.get("source_completeness"),
                "evidence_strength": strength,
            }
        )
    carriers.sort(key=lambda value: (str(value["mutation_event_id"]), _int(value.get("generation")), str(value.get("agent_id"))))

    persistence_by_id = {}
    for selection_id, role, row in persistence_rows:
        mutation_id = original_to_readiness.get((selection_id, str(row.get("mutation_event_id"))))
        if mutation_id:
            persistence_by_id[mutation_id] = {
                "mutation_event_id": mutation_id,
                "selection_id": selection_id,
                "source_role": role,
                "status": row.get("status"),
                "first_observed_generation": row.get("origin_generation"),
                "last_observed_generation": row.get("last_recorded_carrier_generation"),
                "duration_generations": row.get("persisted_generations"),
                "carrier_counts_by_generation": {},
                "lineage_fractions": {},
                "loss_or_extinction": "not_observed" if row.get("status") == "persisted" else "unavailable",
                "independent_recurrences": [],
                "confidence": "measured" if row.get("status") == "persisted" else "partial",
                "evidence_strength": row.get("evidence_strength", "observed"),
                "limitations": list(row.get("limitations", []) or []),
            }
    persistence = []
    for event in events:
        persistence.append(
            persistence_by_id.get(
                str(event["mutation_event_id"]),
                {
                    "mutation_event_id": event["mutation_event_id"],
                    "selection_id": event["selection_id"],
                    "source_role": event["source_role"],
                    "status": "unavailable",
                    "first_observed_generation": event.get("generation"),
                    "last_observed_generation": event.get("generation"),
                    "duration_generations": 0,
                    "carrier_counts_by_generation": {str(event.get("generation")): 1},
                    "lineage_fractions": {},
                    "loss_or_extinction": "unavailable",
                    "independent_recurrences": [],
                    "confidence": "identity_or_ancestry_incomplete",
                    "evidence_strength": "observed",
                    "limitations": ["No complete descendant ancestry and carrier observation chain was selected."],
                },
            )
        )

    carrier_summary_by_id = {}
    for selection_id, _role, row in carrier_summaries:
        mutation_id = original_to_readiness.get((selection_id, str(row.get("mutation_event_id"))))
        if mutation_id:
            carrier_summary_by_id[mutation_id] = row
    enrichment = []
    for event in events:
        summary = carrier_summary_by_id.get(str(event["mutation_event_id"]), {})
        detail = dict(summary.get("lineage_enrichment", {}) or {})
        enrichment.append(
            {
                "schema_version": ENRICHMENT_SCHEMA,
                "mutation_event_id": event["mutation_event_id"],
                "selection_id": event["selection_id"],
                "lineage_id": event.get("lineage_id"),
                "status": detail.get("status", "unavailable"),
                "reason": detail.get("reason", "lineage carrier and non-carrier denominators were not selected"),
                "carrier_count": summary.get("carrier_count", 1),
                "lineage_population": None,
                "carrier_fraction": summary.get("carrier_fraction"),
                "baseline_fraction": None,
                "enrichment_difference": None,
                "evidence_strength": "observed",
                "causal_status": "not_assessed",
            }
        )

    associations = []
    associated_ids = set()
    inherited_event_ids = {
        str(value["mutation_event_id"])
        for value in carriers
        if value.get("evidence_strength") == "inherited"
    }
    persisted_event_ids = {
        str(value["mutation_event_id"])
        for value in persistence
        if value.get("status") == "persisted"
    }
    for selection_id, role, row in source_associations:
        mutation_id = original_to_readiness.get((selection_id, str(row.get("mutation_event_id"))))
        if not mutation_id:
            continue
        status = str(row.get("status") or "unavailable")
        ladder_complete = mutation_id in inherited_event_ids and mutation_id in persisted_event_ids
        matching_dimensions = _json_cell(row.get("required_match_fields")) or []
        missing_matching_dimensions = [
            value for value in MATCH_HIERARCHY if value not in matching_dimensions
        ]
        if status == "available" and not ladder_complete:
            status = "unavailable"
        if status == "available" and missing_matching_dimensions:
            status = "unavailable"
        associated_ids.add(mutation_id)
        association = {
            "schema_version": ASSOCIATION_SCHEMA,
            "association_id": "rassoc_" + stable_hash(
                "eiel-gen1-imported-association-v1", {"selection_id": selection_id, "source_id": row.get("association_id")}
            ),
            "mutation_event_id": mutation_id,
            "selection_id": selection_id,
            "source_role": role,
            "outcome_metric": row.get("outcome_metric"),
            "status": status,
            "reason": (
                row.get("reason")
                if status == str(row.get("status") or "unavailable")
                else (
                    "evidence ladder incomplete: inherited and persisted records are required before association"
                    if not ladder_complete
                    else "required R1 matching hierarchy incomplete; matching was not weakened"
                )
            ),
            "carrier_group": {"count": _int(row.get("carrier_count"), 0)},
            "control_group": {"count": _int(row.get("matched_non_carrier_count"), 0)},
            "matching_dimensions": matching_dimensions,
            "unmatched_dimensions": [] if status == "available" else missing_matching_dimensions or list(MATCH_HIERARCHY),
            "carrier_outcome": row.get("carrier_mean"),
            "control_outcome": row.get("matched_non_carrier_mean"),
            "difference_carrier_minus_control": row.get("difference_carrier_minus_control"),
            "coverage": 1.0 if status == "available" else 0.0,
            "evidence_strength": "associated" if status == "available" else "observed",
            "causal_status": "association_only" if status == "available" else "not_assessed",
            "limitations": (
                _json_cell(row.get("limitations")) or ["Matched controls unavailable."]
            )
            + ([] if ladder_complete else ["Evidence strength was not advanced past an incomplete inheritance or persistence level."])
            + ([] if not missing_matching_dimensions else ["One or more required R1 matching dimensions were absent."]),
        }
        validate_no_causal_wording(association)
        associations.append(canonical_payload(association))
    for event in events:
        if str(event["mutation_event_id"]) not in associated_ids:
            associations.append(
                _unavailable_association(
                    event,
                    "complete matched outcome, exposure, opportunity, and linked-mutation controls were not selected",
                )
            )
    associations.sort(key=lambda value: (str(value["mutation_event_id"]), str(value["association_id"])))
    agent_profiles, lineage_profiles = _profile_rows(events)
    confounds = _confound_matrix()
    branches = _branch_candidates(events)
    decision = _decision(events, carriers, persistence, associations)
    companion = _companion_index(events, agent_profiles, lineage_profiles, branches)

    after_trees = {str(path): _tree_fingerprint(path) for path, _role in selections}
    for row in source_rows:
        row["tree_after"] = after_trees[str(row["path"])]
        row["source_immutable"] = row["tree_before"] == row["tree_after"]
    rng_after = _rng_fingerprints()
    rng_unchanged = rng_before == rng_after
    completeness = {
        "schema_version": COMPLETENESS_SCHEMA,
        "selection_reason": str(selection_reason),
        "pooling_policy": "each explicit source selection retains a selection_id; cross-selection identities are never merged",
        "sources": source_rows,
        "all_selected_sources_immutable": all(bool(row["source_immutable"]) for row in source_rows),
        "rng_generators_checked": sorted(key for key, value in rng_before.items() if value is not None),
        "rng_unchanged": rng_unchanged,
        "simulation_launched": False,
        "canonical_scope_rule": "policy, structural, family-head, native, and aggregate hashes are not canonical modular-genome hashes",
    }
    if not completeness["all_selected_sources_immutable"]:
        raise GenomeReadinessError("selected source bytes changed during retrospective processing")
    if not rng_unchanged:
        raise GenomeReadinessError("RNG state changed during retrospective processing")

    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "mutation_event_inventory.jsonl", events)
    write_json(output / "mutation_identity_map.json", {"schema_version": IDENTITY_SCHEMA, "mutations": identity_rows})
    write_jsonl(output / "carrier_records.jsonl", carriers)
    write_json(output / "mutation_persistence_summary.json", {"schema_version": PERSISTENCE_SCHEMA, "mutations": persistence})
    enrichment_fields = (
        "schema_version", "mutation_event_id", "selection_id", "lineage_id", "status", "reason",
        "carrier_count", "lineage_population", "carrier_fraction", "baseline_fraction", "enrichment_difference",
        "evidence_strength", "causal_status",
    )
    atomic_write_text(output / "mutation_lineage_enrichment.csv", _csv_text(enrichment, enrichment_fields))
    association_fields = (
        "schema_version", "association_id", "mutation_event_id", "selection_id", "source_role", "outcome_metric",
        "status", "reason", "carrier_group", "control_group", "matching_dimensions", "unmatched_dimensions",
        "carrier_outcome", "control_outcome", "difference_carrier_minus_control", "coverage", "evidence_strength",
        "causal_status", "limitations",
    )
    atomic_write_text(output / "mutation_outcome_associations.csv", _csv_text(associations, association_fields))
    write_jsonl(output / "agent_success_profiles.jsonl", agent_profiles)
    write_jsonl(output / "lineage_success_profiles.jsonl", lineage_profiles)
    write_json(output / "source_completeness.json", completeness)
    write_json(output / "confound_matrix.json", confounds)
    write_json(output / "branch_test_candidates.json", branches)
    write_json(output / "gen1_readiness_decision.json", decision)
    write_json(output / "chronicle_companion_index.json", companion)
    atomic_write_text(
        output / "gen1_readiness_report.md",
        _report(decision, source_rows, events, carriers, persistence, associations, agent_profiles, lineage_profiles, branches),
    )

    artifacts = []
    for name in OUTPUT_NAMES:
        path = output / name
        if name == "gen1_readiness_manifest.json" or not path.is_file():
            continue
        artifacts.append({"relative_path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = {
        "schema_version": READINESS_SCHEMA,
        "selection_reason": str(selection_reason),
        "selection_ids": [str(row["selection_id"]) for row in source_rows],
        "source_count": len(source_rows),
        "input_run_ids": sorted({str(value["run_id"]) for value in events}),
        "mutation_event_count": len(events),
        "real_mutation_event_count": sum(1 for value in events if value.get("source_role") == "evidence"),
        "carrier_record_count": len(carriers),
        "association_record_count": len(associations),
        "agent_profile_count": len(agent_profiles),
        "lineage_profile_count": len(lineage_profiles),
        "decision": decision["decision"],
        "evidence_strength_ladder": list(EVIDENCE_STRENGTHS),
        "matching_hierarchy": list(MATCH_HIERARCHY),
        "source_immutability": completeness["all_selected_sources_immutable"],
        "rng_immutability": completeness["rng_unchanged"],
        "simulation_launched": False,
        "artifacts": artifacts,
        "output_bytes_excluding_manifest": sum(int(value["bytes"]) for value in artifacts),
        "full_genomes_duplicated": False,
        "causal_status": "not_assessed",
    }
    write_json(output / "gen1_readiness_manifest.json", manifest)
    return canonical_payload(
        {
            "output_dir": str(output),
            "decision": decision["decision"],
            "mutation_event_count": len(events),
            "real_mutation_event_count": manifest["real_mutation_event_count"],
            "carrier_record_count": len(carriers),
            "files": sorted(path.name for path in output.iterdir() if path.is_file()),
        }
    )


__all__ = [
    "GenomeReadinessError",
    "MATCH_HIERARCHY",
    "OUTPUT_NAMES",
    "validate_no_causal_wording",
    "write_readiness_outputs",
]
