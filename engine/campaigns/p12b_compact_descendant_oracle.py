"""P12B compact ordinary-birth descendant oracle.

The oracle is post-hoc over authoritative checkpoint projections and complete
bounded causal segments.  It emits the decision-bearing ancestry surface needed
for P12B without requiring one monolithic G30 causal projection.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import p12a_gv2_persistence_reducer as p12a
from . import p12b_ob_state_persistence_reducer as p12b
from .canonical import canonical_sha256, pretty_json


SCHEMA_VERSION = "p12b_compact_descendant_oracle_schema_v1"
RESULT_SCHEMA_VERSION = "p12b_compact_descendant_oracle_result_v1"
ORACLE_ID = "p12b_ob_compact_descendant_oracle_v1"
ORACLE_VERSION = "p12b_ob_compact_descendant_oracle_v1"
SUPPORTED_IDENTITY_TYPES = p12a.SUPPORTED_IDENTITY_TYPES
AUTHORITY_KEYS = (
    "provider_id",
    "runtime_profile",
    "cuda_artifact_sha256",
    "cuda_manifest_sha256",
    "execution_route_sha256",
    "execution_route_receipt_sha256",
    "package_id_sha256",
    "artifact_source_commit",
)


class P12BCompactDescendantOracleError(RuntimeError):
    """Compact oracle input is not authoritative enough for P12B."""


def _fail(message: str) -> None:
    raise P12BCompactDescendantOracleError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P12BCompactDescendantOracleError(
            f"cannot read JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"JSON document must be an object: {path}")
    return value


def _validate_authority(authority: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(authority, Mapping):
        _fail("authority binding must be an object")
    result: dict[str, str] = {}
    for key in AUTHORITY_KEYS:
        value = authority.get(key)
        if type(value) is not str or not value:
            _fail(f"authority binding lacks {key}")
        result[key] = value
    if result["provider_id"] != "eiel-production-provider-v1":
        _fail("authority binding provider_id is not production provider")
    if result["runtime_profile"] != "eiel-production-runtime-v1":
        _fail("authority binding runtime_profile is not production runtime")
    for key in (
        "cuda_artifact_sha256",
        "cuda_manifest_sha256",
        "execution_route_sha256",
        "execution_route_receipt_sha256",
        "package_id_sha256",
    ):
        value = result[key]
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            _fail(f"authority binding {key} must be a lowercase SHA-256")
    return result


def _projection(path: Path, *, milestone: int) -> dict[str, Any]:
    try:
        return p12a._projection(_read_json(path), milestone=milestone)
    except p12a.P12APersistenceReducerError as exc:
        raise P12BCompactDescendantOracleError(str(exc)) from exc


def _gv2_causal(projection: Mapping[str, Any]) -> Mapping[str, Any]:
    gv2 = projection.get("genome_upgrade_v2_checkpoint_projection")
    if not isinstance(gv2, Mapping):
        _fail("checkpoint projection missing Genome V2 projection")
    causal = gv2.get("causal_transaction_projection")
    if not isinstance(causal, Mapping):
        _fail("checkpoint projection missing causal transaction projection")
    if causal.get("capture_status") != "capture_enabled_joined":
        _fail("segment causal projection is not capture_enabled_joined")
    if causal.get("evaluable") is not True:
        _fail("segment causal projection is not evaluable")
    dropped = causal.get("dropped_counts")
    if not isinstance(dropped, Mapping) or int(dropped.get("total", -1)) != 0:
        _fail("segment causal projection has dropped records")
    return causal


def _segment_event_records(
    projection: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    cell_id: str,
    left: int,
    right: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    causal = _gv2_causal(projection)
    receipts = causal.get("event_receipts")
    if not isinstance(receipts, list):
        _fail("segment causal event receipts must be an array")
    rows: list[dict[str, Any]] = []
    counts = {"ordinary_birth": 0, "turnover_refill": 0, "other": 0}
    seen_children: set[int] = set()
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            _fail("segment event receipt must be an object")
        record = receipt.get("transaction_record")
        if not isinstance(record, Mapping):
            _fail("segment event receipt lacks transaction_record")
        identity = record.get("identity")
        if not isinstance(identity, Mapping):
            _fail("segment event receipt lacks identity")
        p12b._receipt_sha(receipt)
        generation = identity.get("generation")
        if type(generation) is not int or not left <= generation < right:
            _fail(
                "segment event generation outside declared window: "
                f"left={left} right={right} generation={generation!r}"
            )
        if identity.get("cell_id") != cell_id:
            _fail("segment event cell_id mismatch")
        expected = {
            "provider_id": authority["provider_id"],
            "runtime_profile_id": authority["runtime_profile"],
            "artifact_sha256": authority["cuda_artifact_sha256"],
            "package_id_sha256": authority["package_id_sha256"],
            "execution_route_sha256": authority["execution_route_sha256"],
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                _fail(f"segment event authority mismatch for {key}")
        event_kind = str(identity.get("event_kind"))
        if event_kind == "ordinary_birth":
            counts["ordinary_birth"] += 1
        elif event_kind == "turnover_refill":
            counts["turnover_refill"] += 1
        else:
            counts["other"] += 1
        child_id = identity.get("child_id")
        if type(child_id) is not int or child_id < 0:
            _fail("segment event lacks child stable ID")
        if child_id in seen_children:
            _fail("segment event contains duplicate child stable ID")
        seen_children.add(child_id)
        parent_id = identity.get("parent_id")
        mate_parent_id = identity.get("mate_parent_id")
        parent_slot = identity.get("parent_slot")
        mate_parent_slot = identity.get("mate_parent_slot")
        if type(parent_id) is not int or parent_id < 0:
            _fail("segment event lacks parent stable ID")
        if type(parent_slot) is not int or parent_slot < 0:
            _fail("segment event lacks parent slot")
        if event_kind == "ordinary_birth" and (
            type(mate_parent_id) is not int or mate_parent_id < 0
        ):
            _fail("ordinary-birth event lacks mate stable ID")
        if event_kind == "ordinary_birth" and (
            type(mate_parent_slot) is not int or mate_parent_slot < 0
        ):
            _fail("ordinary-birth event lacks mate slot")
        hereditary_digest = canonical_sha256(
            {
                "schema_version": "p12b_event_hereditary_digest_v1",
                "genotype_summary": record.get("genotype_summary"),
                "product_summary": record.get("product_summary"),
                "phenotype_summary": record.get("phenotype_summary"),
                "regulatory_summary": record.get("regulatory_summary"),
            }
        )
        child_identity_fields = _event_child_identity_fields(record)
        rows.append(
            {
                "receipt_sha256": str(receipt["receipt_sha256"]),
                "transaction_id": str(identity.get("transaction_id")),
                "event_kind": event_kind,
                "generation": generation,
                "tick": int(identity.get("tick")),
                "event_sequence": int(identity.get("event_sequence")),
                "parent_stable_id": parent_id,
                "parent_slot": parent_slot,
                "mate_stable_id": int(mate_parent_id)
                if type(mate_parent_id) is int
                else None,
                "mate_slot": int(mate_parent_slot)
                if type(mate_parent_slot) is int
                else None,
                "child_stable_id": child_id,
                "child_slot": int(identity.get("destination_slot")),
                "lineage_founder_id": int(identity.get("lineage_founder_id")),
                "lineage_depth": int(identity.get("lineage_depth")),
                "child_hereditary_digest_sha256": hereditary_digest,
                "child_identity_fields": child_identity_fields,
                "child_state_sha256_by_identity": {
                    identity_type: canonical_sha256(
                        {
                            "schema_version": (
                                "p12b_event_child_state_identity_v1"
                            ),
                            "identity_type": identity_type,
                            "matched_fields": fields,
                        }
                    )
                    for identity_type, fields in child_identity_fields.items()
                },
                "event_record_sha256": canonical_sha256(record),
            }
        )
    return rows, counts


def _flatten_ints(value: Any, *, path: str) -> list[int]:
    if isinstance(value, list):
        result: list[int] = []
        for index, item in enumerate(value):
            result.extend(_flatten_ints(item, path=f"{path}[{index}]"))
        return result
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"{path} must contain only integer Genome V2 values")
    return [int(value)]


def _event_child_identity_fields(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    genotype = record.get("genotype_summary")
    product = record.get("product_summary")
    phenotype = record.get("phenotype_summary")
    if (
        not isinstance(genotype, Mapping)
        or not isinstance(product, Mapping)
        or not isinstance(phenotype, Mapping)
    ):
        _fail("segment event lacks complete Genome V2 child summaries")
    allele_fields = {
        "genome_v2_alleles_q10": _flatten_ints(
            genotype.get("alleles_q10"), path="genotype_summary.alleles_q10"
        ),
        "genome_v2_modifiers_q10": _flatten_ints(
            genotype.get("modifiers_q10"), path="genotype_summary.modifiers_q10"
        ),
        "genome_v2_products_q10": _flatten_ints(
            product.get("products_q10"), path="product_summary.products_q10"
        ),
        "genome_v2_phenotypes_q10": _flatten_ints(
            phenotype.get("phenotypes_q10"), path="phenotype_summary.phenotypes_q10"
        ),
    }
    active_product_count = product.get("active_product_count")
    saturated_product_count = product.get("saturated_product_count")
    if type(active_product_count) is not int or type(saturated_product_count) is not int:
        _fail("segment event product summary lacks exact product counts")
    package_fields = {
        "genome_v2_phenotypes_q10": list(allele_fields["genome_v2_phenotypes_q10"]),
        "genome_v2_active_product_count": [active_product_count],
        "genome_v2_saturated_product_count": [saturated_product_count],
    }
    return {
        "allele_genotype_state_identity": allele_fields,
        "preregistered_package_identity": package_fields,
        "event_time_hereditary_payload_summary_identity": {
            "child_hereditary_digest_sha256": canonical_sha256(
                {
                    "schema_version": "p12b_event_hereditary_digest_v1",
                    "genotype_summary": genotype,
                    "product_summary": product,
                    "phenotype_summary": phenotype,
                    "regulatory_summary": record.get("regulatory_summary"),
                }
            )
        },
    }


def _validate_segments(
    *,
    segment_paths: Sequence[Path],
    target_milestone: int,
) -> list[tuple[int, int, Path]]:
    windows: list[tuple[int, int, Path]] = []
    left = 0
    for path in segment_paths:
        doc = _read_json(path)
        projection = doc.get("projection", doc)
        if not isinstance(projection, Mapping):
            _fail("segment checkpoint document lacks projection")
        metadata = projection.get("metadata")
        if not isinstance(metadata, Mapping) or type(metadata.get("generation")) is not int:
            _fail("segment checkpoint projection lacks generation metadata")
        right = int(metadata["generation"])
        if right <= left:
            _fail("segment checkpoints must be strictly increasing")
        windows.append((left, right, path))
        left = right
    if left != target_milestone:
        _fail(
            f"segment windows do not seal target milestone: expected G{target_milestone}, got G{left}"
        )
    return windows


def _event_chain(
    *,
    agent_id: int,
    founder_id: int,
    birth_by_child: Mapping[int, Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]], int]:
    if agent_id == founder_id:
        return "founder", [], 0

    def walk(current: int, seen: frozenset[int]) -> list[list[dict[str, Any]]]:
        if current == founder_id:
            return [[]]
        if current in seen:
            return []
        event = birth_by_child.get(current)
        if event is None:
            return []
        if event["event_kind"] != "ordinary_birth":
            return []
        found: list[list[dict[str, Any]]] = []
        next_seen = frozenset((*seen, current))
        for ancestor in (
            int(event["parent_stable_id"]),
            int(event["mate_stable_id"]),
        ):
            for suffix in walk(ancestor, next_seen):
                found.append([dict(event), *suffix])
        return found

    chains = walk(agent_id, frozenset())
    if not chains:
        return "missing_ancestry_root", [], 0
    chains.sort(
        key=lambda chain: (
            len(chain),
            [
                (
                    int(row["generation"]),
                    int(row["tick"]),
                    int(row["event_sequence"]),
                    int(row["child_stable_id"]),
                )
                for row in chain
            ],
        )
    )
    chain = chains[0]
    if not chain:
        return "founder", chain, len(chains)
    return (
        "ordinary_birth" if len(chain) == 1 else "mixed_descendant_path",
        chain,
        len(chains),
    )


def _chain_digest(chain: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        {
            "schema_version": "p12b_compact_ordinary_birth_chain_digest_v1",
            "events": [
                {
                    "transaction_id": row["transaction_id"],
                    "receipt_sha256": row["receipt_sha256"],
                    "parent_stable_id": row["parent_stable_id"],
                    "parent_slot": row["parent_slot"],
                    "mate_stable_id": row["mate_stable_id"],
                    "mate_slot": row["mate_slot"],
                    "child_stable_id": row["child_stable_id"],
                    "child_slot": row["child_slot"],
                    "event_sequence": row["event_sequence"],
                    "generation": row["generation"],
                    "tick": row["tick"],
                    "event_record_sha256": row["event_record_sha256"],
                }
                for row in chain
            ],
        }
    )


def reduce_compact_descendant_oracle(
    *,
    g0_path: Path,
    target_path: Path,
    segment_paths: Sequence[Path],
    target_milestone: int,
    authority: Mapping[str, Any],
    source_commit: str,
    campaign_id: str,
    cell_id: str,
    seed: int,
    evidence_root: Path,
) -> dict[str, Any]:
    if type(target_milestone) is not int or target_milestone <= 0:
        _fail("target milestone must be a positive integer")
    schema = {
        "schema_version": SCHEMA_VERSION,
        "oracle_id": ORACLE_ID,
        "oracle_version": ORACLE_VERSION,
        "source_commit": source_commit,
        "authority": _validate_authority(authority),
        "identity_types": list(SUPPORTED_IDENTITY_TYPES),
        "positive_rule": {
            "requires_terminal_checkpoint_state_match": True,
            "requires_complete_bounded_causal_segments": True,
            "requires_ordinary_birth_event_chain_to_founder": True,
            "counts": ["ordinary_birth", "mixed_descendant_path"],
        },
    }
    g0 = _projection(g0_path, milestone=0)
    target = _projection(target_path, milestone=target_milestone)
    capacity = len(p12a._tensor(g0, "alive_mask")[0])
    if len(p12a._tensor(target, "alive_mask")[0]) != capacity:
        _fail(f"G0/G{target_milestone} tensor capacity mismatch")
    g0_live = p12a._live_slots(g0)
    target_live = p12a._live_slots(target)
    founders = [
        row
        for row in g0_live
        if row["lineage_depth"] == 0 and row["lineage_founder_id"] == row["agent_id"]
    ]
    if not founders:
        _fail("G0 contains no authoritative founders")

    event_rows: list[dict[str, Any]] = []
    segment_receipts: list[dict[str, Any]] = []
    for left, right, path in _validate_segments(
        segment_paths=segment_paths,
        target_milestone=target_milestone,
    ):
        segment = _projection(path, milestone=right)
        rows, counts = _segment_event_records(
            segment,
            authority=schema["authority"],
            cell_id=cell_id,
            left=left,
            right=right,
        )
        event_rows.extend(rows)
        segment_receipts.append(
            {
                "left_generation": left,
                "right_generation": right,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "event_counts": counts,
                "compact_event_rows_observed": len(rows),
            }
        )
    birth_by_child: dict[int, dict[str, Any]] = {}
    for row in event_rows:
        child = int(row["child_stable_id"])
        if child in birth_by_child:
            _fail("compact event ledger contains duplicate child stable ID")
        if row["event_kind"] == "ordinary_birth":
            birth_by_child[child] = row

    state_registry: list[dict[str, Any]] = []
    carrier_rows: list[dict[str, Any]] = []
    reducer_rows: list[dict[str, Any]] = []
    ambiguity_count = 0
    lineage_candidate_unresolved_count = 0
    for founder in founders:
        for identity_type in SUPPORTED_IDENTITY_TYPES:
            state = p12a._state_definition(
                g0,
                founder,
                identity_type=identity_type,
                capacity=capacity,
            )
            state_id = f"state-{state['state_definition_sha256'][:16]}-{identity_type}"
            state_registry.append(
                {
                    "campaign_id": campaign_id,
                    "cell_id": cell_id,
                    "seed": seed,
                    "preregistered_state_id": state_id,
                    "identity_type": identity_type,
                    "state_definition_sha256": state["state_definition_sha256"],
                    "founder_stable_id": founder["agent_id"],
                    "founder_slot": founder["slot"],
                    "founder_slot_token": founder["slot_token"],
                    "exact_hereditary_digest": state["exact_hereditary_digest"],
                    "matched_fields": state["matched_fields"],
                }
            )
            matches: list[dict[str, Any]] = []
            for carrier in target_live:
                try:
                    matched, selected, exact = p12a._matches(
                        state, target, carrier["slot"], capacity=capacity
                    )
                except p12a.P12APersistenceReducerError as exc:
                    raise P12BCompactDescendantOracleError(str(exc)) from exc
                if not matched:
                    continue
                origin_class, chain, chain_path_count = _event_chain(
                    agent_id=int(carrier["agent_id"]),
                    founder_id=int(founder["agent_id"]),
                    birth_by_child=birth_by_child,
                )
                if origin_class in {
                    "ambiguous_cycle",
                    "missing_ancestry_root",
                    "non_ordinary_birth_path",
                }:
                    ambiguity_count += 1
                    if (
                        int(carrier["lineage_depth"]) > 0
                        and int(carrier["lineage_founder_id"]) == int(founder["agent_id"])
                    ):
                        lineage_candidate_unresolved_count += 1
                    origin_for_row = "unresolved"
                else:
                    origin_for_row = origin_class
                positive = (
                    origin_for_row in {"ordinary_birth", "mixed_descendant_path"}
                    and int(carrier["lineage_depth"]) > 0
                    and int(carrier["lineage_founder_id"]) == int(founder["agent_id"])
                )
                row = {
                    "campaign_id": campaign_id,
                    "cell_id": cell_id,
                    "seed": seed,
                    "milestone": target_milestone,
                    "preregistered_state_id": state_id,
                    "identity_type": identity_type,
                    "state_definition_sha256": state["state_definition_sha256"],
                    "founder_stable_id": founder["agent_id"],
                    "carrier_stable_id": carrier["agent_id"],
                    "carrier_slot": carrier["slot"],
                    "carrier_slot_token": carrier["slot_token"],
                    "lineage_depth": carrier["lineage_depth"],
                    "lineage_founder_id": carrier["lineage_founder_id"],
                    "carrier_origin_class": origin_for_row,
                    "origin_failure": None
                    if origin_for_row == origin_class
                    else origin_class,
                    "positive_ordinary_birth_persistence": positive,
                    "selected_values": selected,
                    "exact_values": exact,
                    "ordinary_birth_chain_length": len(chain),
                    "ordinary_birth_chain_path_count": chain_path_count,
                    "ordinary_birth_chain_digest_sha256": _chain_digest(chain),
                    "ordinary_birth_chain": chain,
                }
                matches.append(row)
                carrier_rows.append(row)
            positive_rows = [
                row for row in matches if row["positive_ordinary_birth_persistence"]
            ]
            ordinary_direct = sum(
                1 for row in positive_rows if row["carrier_origin_class"] == "ordinary_birth"
            )
            mixed_path = sum(
                1
                for row in positive_rows
                if row["carrier_origin_class"] == "mixed_descendant_path"
            )
            founder_count = sum(
                1 for row in matches if row["carrier_origin_class"] == "founder"
            )
            unresolved_count = sum(
                1 for row in matches if row["carrier_origin_class"] == "unresolved"
            )
            reducer_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "oracle_id": ORACLE_ID,
                    "oracle_version": ORACLE_VERSION,
                    "source_commit": source_commit,
                    "provider_runtime": {
                        "provider_id": schema["authority"]["provider_id"],
                        "runtime_profile": schema["authority"]["runtime_profile"],
                    },
                    "artifact_package_route_identities": {
                        key: schema["authority"][key]
                        for key in (
                            "cuda_artifact_sha256",
                            "cuda_manifest_sha256",
                            "execution_route_sha256",
                            "execution_route_receipt_sha256",
                            "package_id_sha256",
                        )
                    },
                    "campaign_id": campaign_id,
                    "cell_id": cell_id,
                    "seed": seed,
                    "milestone": target_milestone,
                    "preregistered_state_id": state_id,
                    "identity_type": identity_type,
                    "state_definition_sha256": state["state_definition_sha256"],
                    "founder_stable_id": founder["agent_id"],
                    "match_result": "match" if matches else "no_match",
                    "carrier_count": len(matches),
                    "founder_carrier_count": founder_count,
                    "positive_ordinary_birth_carrier_count": len(positive_rows),
                    "ordinary_birth_direct_child_carrier_count": ordinary_direct,
                    "mixed_descendant_path_carrier_count": mixed_path,
                    "unresolved_carrier_count": unresolved_count,
                    "alive_denominator": len(target_live),
                    "frequency": (len(matches) / len(target_live)) if target_live else 0.0,
                    "positive_ordinary_birth_frequency": (
                        len(positive_rows) / len(target_live) if target_live else 0.0
                    ),
                    "classification": {
                        "state_present": len(matches) > 0,
                        "founder_present": founder_count > 0,
                        "positive_ordinary_birth_persistence": len(positive_rows) > 0,
                        "ordinary_birth_direct_child_persistence": ordinary_direct > 0,
                        "ordinary_birth_descendant_persistence": mixed_path > 0,
                        "unresolved_present": unresolved_count > 0,
                        "state_lost": len(matches) == 0,
                    },
                }
            )

    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "oracle_id": ORACLE_ID,
        "oracle_version": ORACLE_VERSION,
        "schema": schema,
        "source_commit": source_commit,
        "campaign_id": campaign_id,
        "cell_id": cell_id,
        "seed": seed,
        "target_milestone": target_milestone,
        "checkpoint_identity": {
            "g0": {"path": str(g0_path), "sha256": hashlib.sha256(g0_path.read_bytes()).hexdigest()},
            f"g{target_milestone}": {
                "path": str(target_path),
                "sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            },
        },
        "segment_receipts": segment_receipts,
        "evidence_root": str(evidence_root),
        "state_registry": state_registry,
        "compact_event_rows": [
            row
            for row in {
                event["transaction_id"]: event
                for carrier in carrier_rows
                for event in carrier["ordinary_birth_chain"]
            }.values()
        ],
        "source_event_rows": event_rows,
        "target_carrier_table": carrier_rows,
        "reducer_rows": reducer_rows,
        "summary": {
            "g0_founder_count": len(founders),
            "g0_alive_denominator": len(g0_live),
            "target_milestone": target_milestone,
            "target_alive_denominator": len(target_live),
            "segment_count": len(segment_receipts),
            "source_event_row_count": len(event_rows),
            "compact_event_row_count": len(
                {
                    event["transaction_id"]
                    for carrier in carrier_rows
                    for event in carrier["ordinary_birth_chain"]
                }
            ),
            "ordinary_birth_event_row_count": sum(
                int(receipt["event_counts"]["ordinary_birth"])
                for receipt in segment_receipts
            ),
            "carrier_row_count": len(carrier_rows),
            "reducer_row_count": len(reducer_rows),
            "positive_ordinary_birth_carrier_rows": sum(
                1 for row in carrier_rows if row["positive_ordinary_birth_persistence"]
            ),
            "ambiguous_or_unresolved_chain_count": ambiguity_count,
            "lineage_candidate_unresolved_chain_count": lineage_candidate_unresolved_count,
            "evaluable": True,
            "supported_identity_types": list(SUPPORTED_IDENTITY_TYPES),
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def write_oracle_outputs(result: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": output_dir / "p12b_compact_descendant_oracle_result_v1.json",
        "schema": output_dir / "p12b_compact_descendant_oracle_schema_v1.json",
        "state_registry": output_dir / "p12b_compact_state_registry_v1.json",
        "compact_event_rows": output_dir / "p12b_compact_event_rows_v1.json",
        "source_event_rows": output_dir / "p12b_source_event_rows_v1.json",
        "target_carrier_table": output_dir / "p12b_compact_target_carrier_table_v1.json",
        "reducer_rows": output_dir / "p12b_compact_reducer_rows_v1.json",
        "summary": output_dir / "p12b_compact_descendant_oracle_summary_v1.json",
    }
    payloads = {
        "result": dict(result),
        "schema": result["schema"],
        "state_registry": {"rows": result["state_registry"]},
        "compact_event_rows": {"rows": result["compact_event_rows"]},
        "source_event_rows": {"rows": result["source_event_rows"]},
        "target_carrier_table": {"rows": result["target_carrier_table"]},
        "reducer_rows": {"rows": result["reducer_rows"]},
        "summary": result["summary"],
    }
    written: dict[str, str] = {}
    for key, path in paths.items():
        path.write_text(pretty_json(payloads[key]), encoding="utf-8")
        written[key] = str(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--segment", type=Path, action="append", required=True)
    parser.add_argument("--target-milestone", type=int, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = reduce_compact_descendant_oracle(
        g0_path=args.g0,
        target_path=args.target,
        segment_paths=tuple(args.segment),
        target_milestone=args.target_milestone,
        authority=_read_json(args.authority),
        source_commit=args.source_commit,
        campaign_id=args.campaign_id,
        cell_id=args.cell_id,
        seed=args.seed,
        evidence_root=args.evidence_root,
    )
    written = write_oracle_outputs(result, args.output_dir)
    print(
        pretty_json(
            {"ok": True, "result_sha256": result["result_sha256"], "paths": written}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
