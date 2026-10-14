"""P12A Genome V2 single-arm persistence reducer.

The reducer is intentionally post-hoc over authoritative checkpoint projections:
it never accepts caller-supplied founder states, carrier classes, or reporting
summaries as evidence.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

from .canonical import canonical_sha256, pretty_json


SCHEMA_VERSION = "p12a_gv2_single_arm_persistence_reducer_schema_v1"
STATE_SCHEMA_VERSION = "p12a_gv2_persistence_state_definition_v1"
RESULT_SCHEMA_VERSION = "p12a_gv2_persistence_reducer_result_v1"
REDUCER_ID = "p12a_gv2_single_arm_nonfounder_carrier_persistence_v1"
REDUCER_VERSION = "p12a_gv2_single_arm_persistence_reducer_v1"
SUPPORTED_IDENTITY_TYPES = (
    "exact_hereditary_payload_identity",
    "allele_genotype_state_identity",
    "preregistered_package_identity",
)
CLASSIFICATIONS = (
    "founder_present",
    "nonfounder_carrier_present",
    "state_present_in_direct_child",
    "state_present_after_2_generations",
    "state_present_after_5_generations",
    "state_lost",
    "state_fixed",
    "state_coexisting",
    "founder_only_persistence",
    "reproductive_persistence_unobserved",
)

EXPECTED_AUTHORITY = {
    "handoff_decision": "P12A-WDL-TURNOVER-PARENTAGE-HANDOFF-B",
    "handoff_receipt_sha256": "2bd31077d85d115e0e0e141154f51bbdf61e6f2190c018399cc6f9ad84ed0b85",
    "persistence_design_decision": "P12A-GV2-PERSISTENCE-DESIGN-D",
    "persistence_decision_json_sha256": "786ff1b5bbf137d18a003d293d21506e82281b9a4c97388dc07722d9c07f2c63",
    "metric_schema_sha256": "1bfb9b830b89febd9cf347f086f35341b0311e564451ac26e03217911737f48a",
    "provider_id": "eiel-production-provider-v1",
    "runtime_profile": "eiel-production-runtime-v1",
    "native_binding": "wdl_fixed_mlp_native_binding_v1",
    "cuda_artifact_sha256": "371dcf21aeaad912f19900221cab32882cd0a0c6255ff36b55ead81c0b1a3430",
    "cuda_manifest_sha256": "5a1d6c75c7d713806d228532f8fb0bdf3a96058cb9334630e4fa598078fb9d1f",
    "execution_route_sha256": "4e0b87d0604e8c8ab1329a6b09091e3b734a1cbf85c92b6b21d71c3bb377d17a",
    "package_id_sha256": "593dbc112f29dda5397be44f11be49b9eaa2dec6ab51638aae54a62853b7a7ae",
}
STALE_HANDOFFS = {"HANDOFF-A", "P12A-WDL-TURNOVER-PARENTAGE-HANDOFF-A"}
STALE_RECEIPT_PREFIXES = ("845e9e32",)


class P12APersistenceReducerError(RuntimeError):
    """Reducer input is not authoritative enough for P12A qualification."""


def _fail(message: str) -> None:
    raise P12APersistenceReducerError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                value = json.load(handle)
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise P12APersistenceReducerError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"JSON document must be an object: {path}")
    return value


def _hex_to_values(tensor: Mapping[str, Any], *, name: str) -> list[Any]:
    dtype = tensor.get("dtype")
    shape = tensor.get("shape")
    data_hex = tensor.get("data_hex")
    if not isinstance(dtype, str) or not isinstance(shape, list):
        _fail(f"tensor {name} must include dtype and shape")
    if not all(type(item) is int and item >= 0 for item in shape):
        _fail(f"tensor {name} shape is malformed")
    if not isinstance(data_hex, str):
        _fail(f"tensor {name} must include data_hex")
    raw = bytes.fromhex(data_hex)
    count = math.prod(shape)
    if dtype == "bool":
        if len(raw) != count:
            _fail(f"tensor {name} byte count does not match bool shape")
        return [bool(item) for item in raw]
    if dtype == "uint8":
        if len(raw) != count:
            _fail(f"tensor {name} byte count does not match uint8 shape")
        return list(raw)
    if dtype == "int32":
        if len(raw) != count * 4:
            _fail(f"tensor {name} byte count does not match int32 shape")
        return list(struct.unpack("<" + "i" * count, raw))
    if dtype == "int64":
        if len(raw) != count * 8:
            _fail(f"tensor {name} byte count does not match int64 shape")
        return list(struct.unpack("<" + "q" * count, raw))
    _fail(f"tensor {name} has unsupported dtype {dtype!r}")


def _tensor(projection: Mapping[str, Any], name: str) -> tuple[list[Any], list[int]]:
    tensors = projection.get("tensors")
    if not isinstance(tensors, Mapping):
        _fail("checkpoint projection requires tensors")
    raw = tensors.get(name)
    if not isinstance(raw, Mapping):
        _fail(f"checkpoint projection missing tensor {name}")
    shape = raw.get("shape")
    if not isinstance(shape, list) or not all(type(item) is int for item in shape):
        _fail(f"tensor {name} shape is malformed")
    return _hex_to_values(raw, name=name), list(shape)


def _projection(document: Mapping[str, Any], *, milestone: int) -> dict[str, Any]:
    if "classification" in document or "reducer_rows" in document:
        _fail("reporting-derived or reducer-derived input is not authoritative")
    projection = document.get("projection", document)
    if not isinstance(projection, dict):
        _fail("checkpoint document does not contain a projection object")
    metadata = projection.get("metadata")
    if not isinstance(metadata, Mapping):
        _fail("checkpoint projection missing metadata")
    generation = metadata.get("generation", document.get("generation_index"))
    if generation is None:
        generation = document.get("source_generation")
    if type(generation) is not int or generation != milestone:
        _fail(f"checkpoint milestone mismatch: expected G{milestone}, got {generation!r}")
    if "tensors" not in projection:
        _fail("reporting-only input lacks checkpoint tensors")
    return projection


def _digest_rows(values: Sequence[Any], slot: int, *, capacity: int) -> str:
    start = slot * 32
    end = start + 32
    if len(values) != capacity * 32:
        _fail("canonical_hereditary_digest must be [capacity,32]")
    return bytes(int(item) for item in values[start:end]).hex()


def _slot_values(value: Any, slot: int, *, capacity: int) -> list[int]:
    if isinstance(value, list):
        if len(value) == capacity and all(not isinstance(item, list) for item in value):
            return [int(value[slot])]
        result: list[int] = []
        for item in value:
            result.extend(_slot_values(item, slot, capacity=capacity))
        return result
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("Genome V2 tensor values must be integers")
    return [int(value)]


def _gv2_slot_fields(
    projection: Mapping[str, Any], slot: int, *, capacity: int
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    gv2 = projection.get("genome_upgrade_v2_checkpoint_projection")
    if not isinstance(gv2, Mapping):
        _fail("checkpoint projection missing Genome V2 projection")
    tensors = gv2.get("tensors")
    if not isinstance(tensors, Mapping):
        _fail("Genome V2 projection missing tensors")
    allele_names = (
        "genome_v2_alleles_q10",
        "genome_v2_modifiers_q10",
        "genome_v2_products_q10",
        "genome_v2_phenotypes_q10",
    )
    package_names = (
        "genome_v2_phenotypes_q10",
        "genome_v2_active_product_count",
        "genome_v2_saturated_product_count",
    )
    allele_fields: dict[str, list[int]] = {}
    package_fields: dict[str, list[int]] = {}
    for name in allele_names:
        if name not in tensors:
            _fail(f"Genome V2 projection missing tensor {name}")
        allele_fields[name] = _slot_values(tensors[name], slot, capacity=capacity)
    for name in package_names:
        if name not in tensors:
            _fail(f"Genome V2 projection missing tensor {name}")
        package_fields[name] = _slot_values(tensors[name], slot, capacity=capacity)
    return allele_fields, package_fields


def _projection_identity(path: Path, projection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema_version": projection.get("schema_version"),
        "source_generation": projection.get("metadata", {}).get("generation"),
        "alive_count": projection.get("metadata", {}).get("alive_count"),
    }


def _validate_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(authority, Mapping):
        _fail("authority binding must be an object")
    handoff = authority.get("handoff_decision")
    receipt = str(authority.get("handoff_receipt_sha256", ""))
    if handoff in STALE_HANDOFFS or any(receipt.startswith(p) for p in STALE_RECEIPT_PREFIXES):
        _fail("stale handoff or receipt is rejected")
    result: dict[str, Any] = {}
    for key, expected in EXPECTED_AUTHORITY.items():
        actual = authority.get(key)
        if actual != expected:
            _fail(f"authority binding mismatch for {key}: expected {expected!r}, got {actual!r}")
        result[key] = actual
    return result


def reducer_schema(*, source_commit: str, authority: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_id": REDUCER_ID,
        "reducer_version": REDUCER_VERSION,
        "source_commit": source_commit,
        "authority": _validate_authority(authority),
        "identity_types": list(SUPPORTED_IDENTITY_TYPES),
        "classifications": list(CLASSIFICATIONS),
        "unit_of_observation": "cell x preregistered hereditary state x milestone",
        "milestones": [0, 5, 10, 20, 30],
        "current_milestones": [0, 5],
    }


def _live_slots(projection: Mapping[str, Any]) -> list[dict[str, int]]:
    alive, alive_shape = _tensor(projection, "alive_mask")
    agent_ids, id_shape = _tensor(projection, "slot_to_agent_id")
    slot_tokens, token_shape = _tensor(projection, "slot_tokens")
    depth, depth_shape = _tensor(projection, "lineage_depth")
    founder_ids, founder_shape = _tensor(projection, "lineage_founder_id")
    shapes = {tuple(alive_shape), tuple(id_shape), tuple(token_shape), tuple(depth_shape), tuple(founder_shape)}
    if len(shapes) != 1 or len(alive_shape) != 1:
        _fail("runtime identity tensors must share one-dimensional capacity")
    seen_ids: set[int] = set()
    rows: list[dict[str, int]] = []
    for slot, is_alive in enumerate(alive):
        if not is_alive:
            continue
        agent_id = int(agent_ids[slot])
        token = int(slot_tokens[slot])
        if agent_id in seen_ids:
            _fail("duplicate stable agent ID among alive slots")
        if int(depth[slot]) < 0:
            _fail("lineage depth is negative")
        seen_ids.add(agent_id)
        rows.append(
            {
                "slot": slot,
                "agent_id": agent_id,
                "slot_token": token,
                "lineage_depth": int(depth[slot]),
                "lineage_founder_id": int(founder_ids[slot]),
            }
        )
    return rows


def _state_definition(
    projection: Mapping[str, Any],
    founder: Mapping[str, int],
    *,
    identity_type: str,
    capacity: int,
) -> dict[str, Any]:
    digest_values, digest_shape = _tensor(projection, "canonical_hereditary_digest")
    if digest_shape != [capacity, 32]:
        _fail("canonical_hereditary_digest must be [capacity,32]")
    slot = int(founder["slot"])
    digest = _digest_rows(digest_values, slot, capacity=capacity)
    allele_fields, package_fields = _gv2_slot_fields(projection, slot, capacity=capacity)
    if identity_type == "exact_hereditary_payload_identity":
        matched = {"canonical_hereditary_digest": digest}
    elif identity_type == "allele_genotype_state_identity":
        matched = allele_fields
    elif identity_type == "preregistered_package_identity":
        matched = package_fields
    else:
        _fail(f"unknown identity type: {identity_type}")
    source = {
        "schema_version": STATE_SCHEMA_VERSION,
        "identity_type": identity_type,
        "founder_stable_id": int(founder["agent_id"]),
        "founder_slot": slot,
        "matched_fields": matched,
    }
    state_hash = canonical_sha256(source)
    return {
        **source,
        "state_definition_sha256": state_hash,
        "exact_hereditary_digest": digest,
        "all_allele_genotype_fields": allele_fields,
        "all_package_fields": package_fields,
    }


def _matches(
    state: Mapping[str, Any],
    projection: Mapping[str, Any],
    slot: int,
    *,
    capacity: int,
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    identity_type = state["identity_type"]
    digest_values, _ = _tensor(projection, "canonical_hereditary_digest")
    digest = _digest_rows(digest_values, slot, capacity=capacity)
    allele_fields, package_fields = _gv2_slot_fields(projection, slot, capacity=capacity)
    if identity_type == "exact_hereditary_payload_identity":
        matched = {"canonical_hereditary_digest": digest}
        expected = state["matched_fields"]
    elif identity_type == "allele_genotype_state_identity":
        matched = allele_fields
        expected = state["matched_fields"]
    elif identity_type == "preregistered_package_identity":
        matched = package_fields
        expected = state["matched_fields"]
    else:
        _fail(f"unknown identity type: {identity_type}")
    return matched == expected, matched, {"exact_hereditary_digest": digest}


def _carrier_class(carrier: Mapping[str, int], founder_id: int) -> str:
    if carrier["agent_id"] == founder_id and carrier["lineage_depth"] == 0:
        return "original_founder"
    if carrier["lineage_founder_id"] == founder_id and carrier["lineage_depth"] == 1:
        return "direct_nonfounder_child"
    if carrier["lineage_founder_id"] == founder_id and carrier["lineage_depth"] > 1:
        return "later_nonfounder_carrier"
    if carrier["lineage_founder_id"] != founder_id:
        return "unrelated_carrier"
    return "invalid_unresolved"


def _classification_flags(
    *, total: int, founder_count: int, nonfounder_depths: Sequence[int], denominator: int
) -> dict[str, bool | str]:
    nonfounder_count = len(nonfounder_depths)
    return {
        "founder_present": founder_count > 0,
        "nonfounder_carrier_present": nonfounder_count > 0,
        "state_present_in_direct_child": any(depth == 1 for depth in nonfounder_depths),
        "state_present_after_2_generations": any(depth >= 2 for depth in nonfounder_depths),
        "state_present_after_5_generations": any(depth >= 5 for depth in nonfounder_depths),
        "state_lost": total == 0,
        "state_fixed": denominator > 0 and total == denominator,
        "state_coexisting": 0 < total < denominator,
        "founder_only_persistence": founder_count > 0 and nonfounder_count == 0,
        "reproductive_persistence_unobserved": nonfounder_count == 0,
    }


def reduce_persistence(
    *,
    g0_path: Path,
    g5_path: Path,
    target_milestone: int = 5,
    authority: Mapping[str, Any],
    source_commit: str,
    campaign_id: str,
    cell_id: str,
    seed: int,
    evidence_root: Path,
) -> dict[str, Any]:
    if type(target_milestone) is not int or target_milestone <= 0:
        _fail("target milestone must be a positive integer")
    schema = reducer_schema(source_commit=source_commit, authority=authority)
    g0_doc = _read_json(g0_path)
    target_doc = _read_json(g5_path)
    g0 = _projection(g0_doc, milestone=0)
    target = _projection(target_doc, milestone=target_milestone)
    g0_live = _live_slots(g0)
    target_live = _live_slots(target)
    capacity = len(_tensor(g0, "alive_mask")[0])
    if len(_tensor(target, "alive_mask")[0]) != capacity:
        _fail(f"G0/G{target_milestone} tensor capacity mismatch")
    founders = [
        row
        for row in g0_live
        if row["lineage_depth"] == 0 and row["lineage_founder_id"] == row["agent_id"]
    ]
    if not founders:
        _fail("G0 contains no authoritative founders")
    founder_states: list[dict[str, Any]] = []
    reducer_rows: list[dict[str, Any]] = []
    carrier_rows: list[dict[str, Any]] = []
    for founder in founders:
        for identity_type in SUPPORTED_IDENTITY_TYPES:
            state = _state_definition(g0, founder, identity_type=identity_type, capacity=capacity)
            state_id = (
                f"state-{state['state_definition_sha256'][:16]}-"
                f"{identity_type}"
            )
            founder_state = {
                "metric_id": REDUCER_ID,
                "campaign_id": campaign_id,
                "cell_id": cell_id,
                "seed": seed,
                "milestone": 0,
                "preregistered_state_id": state_id,
                "identity_type": identity_type,
                "state_definition_sha256": state["state_definition_sha256"],
                "founder_stable_id": founder["agent_id"],
                "founder_slot": founder["slot"],
                "founder_slot_token": founder["slot_token"],
                "exact_hereditary_digest": state["exact_hereditary_digest"],
                "matched_fields": state["matched_fields"],
                "canonical_source_representation": state,
            }
            founder_states.append(founder_state)
            baseline_flags = _classification_flags(
                total=1, founder_count=1, nonfounder_depths=(), denominator=len(g0_live)
            )
            reducer_rows.append(
                {
                    **_row_base(schema, campaign_id, cell_id, seed, 0, state_id, state, founder, g0_path, evidence_root),
                    "carrier_stable_id": founder["agent_id"],
                    "carrier_slot_token": founder["slot_token"],
                    "alive_state": True,
                    "lineage_depth": founder["lineage_depth"],
                    "lineage_founder_id": founder["lineage_founder_id"],
                    "founder_nonfounder_classification": "original_founder",
                    "match_result": "match",
                    "carrier_count": 1,
                    "founder_carrier_count": 1,
                    "nonfounder_carrier_count": 0,
                    "alive_denominator": len(g0_live),
                    "frequency": 1 / len(g0_live),
                    "classification": baseline_flags,
                }
            )
            matches: list[dict[str, Any]] = []
            for carrier in target_live:
                matched, selected, exact = _matches(
                    state, target, carrier["slot"], capacity=capacity
                )
                if not matched:
                    continue
                carrier_class = _carrier_class(carrier, founder["agent_id"])
                row = {
                    "preregistered_state_id": state_id,
                    "identity_type": identity_type,
                    "state_definition_sha256": state["state_definition_sha256"],
                    "founder_stable_id": founder["agent_id"],
                    "carrier_stable_id": carrier["agent_id"],
                    "carrier_slot": carrier["slot"],
                    "carrier_slot_token": carrier["slot_token"],
                    "lineage_depth": carrier["lineage_depth"],
                    "lineage_founder_id": carrier["lineage_founder_id"],
                    "carrier_classification": carrier_class,
                    "selected_values": selected,
                    "exact_values": exact,
                }
                matches.append(row)
                carrier_rows.append(row)
            founder_count = sum(1 for item in matches if item["carrier_classification"] == "original_founder")
            nonfounder_depths = [
                int(item["lineage_depth"])
                for item in matches
                if item["carrier_classification"] in {"direct_nonfounder_child", "later_nonfounder_carrier"}
            ]
            flags = _classification_flags(
                total=len(matches),
                founder_count=founder_count,
                nonfounder_depths=nonfounder_depths,
                denominator=len(target_live),
            )
            reducer_rows.append(
                {
                    **_row_base(
                        schema,
                        campaign_id,
                        cell_id,
                        seed,
                        target_milestone,
                        state_id,
                        state,
                        founder,
                        g5_path,
                        evidence_root,
                    ),
                    "carrier_stable_id": None,
                    "carrier_slot_token": None,
                    "alive_state": None,
                    "lineage_depth": None,
                    "lineage_founder_id": None,
                    "founder_nonfounder_classification": "summary",
                    "match_result": "match" if matches else "no_match",
                    "carrier_count": len(matches),
                    "founder_carrier_count": founder_count,
                    "nonfounder_carrier_count": len(nonfounder_depths),
                    "alive_denominator": len(target_live),
                    "frequency": (
                        (len(matches) / len(target_live)) if target_live else 0.0
                    ),
                    "classification": flags,
                }
            )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "metric_id": REDUCER_ID,
        "reducer_version": REDUCER_VERSION,
        "schema": schema,
        "source_commit": source_commit,
        "campaign_id": campaign_id,
        "cell_id": cell_id,
        "seed": seed,
        "checkpoint_identity": {
            "g0": _projection_identity(g0_path, g0),
            f"g{target_milestone}": _projection_identity(g5_path, target),
        },
        "evidence_root_identity": {
            "checkpoint_pair_sha256": canonical_sha256(
                {
                    "g0": _projection_identity(g0_path, g0),
                    f"g{target_milestone}": _projection_identity(g5_path, target),
                }
            ),
        },
        "target_milestone": target_milestone,
        "founder_state_table": founder_states,
        "target_carrier_table": carrier_rows,
        "g5_carrier_table": carrier_rows,
        "reducer_rows": reducer_rows,
        "summary": {
            "g0_founder_count": len(founders),
            "g0_alive_denominator": len(g0_live),
            "target_milestone": target_milestone,
            "target_alive_denominator": len(target_live),
            f"g{target_milestone}_alive_denominator": len(target_live),
            "reducer_row_count": len(reducer_rows),
            "carrier_row_count": len(carrier_rows),
            "supported_identity_types": list(SUPPORTED_IDENTITY_TYPES),
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def _row_base(
    schema: Mapping[str, Any],
    campaign_id: str,
    cell_id: str,
    seed: int,
    milestone: int,
    state_id: str,
    state: Mapping[str, Any],
    founder: Mapping[str, int],
    checkpoint_path: Path,
    evidence_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_id": REDUCER_ID,
        "source_commit": schema["source_commit"],
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
                "package_id_sha256",
            )
        },
        "campaign_id": campaign_id,
        "cell_id": cell_id,
        "seed": seed,
        "milestone": milestone,
        "preregistered_state_id": state_id,
        "identity_type": state["identity_type"],
        "state_definition_sha256": state["state_definition_sha256"],
        "founder_stable_id": founder["agent_id"],
        "exact_hereditary_digest": state["exact_hereditary_digest"],
        "selected_allele_genotype_package_values": state["matched_fields"],
        "matched_fields": sorted(state["matched_fields"]),
        "unmatched_fields": [],
        "canonical_source_representation": state,
        "exact_vs_preregistered_partial_match": (
            "exact"
            if state["identity_type"] == "exact_hereditary_payload_identity"
            else "preregistered_partial"
        ),
        "checkpoint_identity": {
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        },
        "reducer_version_hash": canonical_sha256(
            {"id": REDUCER_ID, "version": REDUCER_VERSION}
        ),
        "evidence_root_identity": "content_addressed_checkpoint_projection_v1",
    }


def write_reducer_outputs(result: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": output_dir / "p12a_gv2_persistence_reducer_result_v1.json",
        "schema": output_dir / "p12a_gv2_persistence_reducer_schema_v1.json",
        "founder_state_table": output_dir / "p12a_gv2_founder_state_table_v1.json",
        "target_carrier_table": output_dir / "p12a_gv2_carrier_table_v1.json",
        "g5_carrier_table": output_dir / "p12a_gv2_g5_carrier_table_v1.json",
        "reducer_rows": output_dir / "p12a_gv2_persistence_reducer_rows_v1.json",
        "summary": output_dir / "p12a_gv2_persistence_reducer_summary_v1.json",
    }
    payloads = {
        "result": dict(result),
        "schema": result["schema"],
        "founder_state_table": {"rows": result["founder_state_table"]},
        "target_carrier_table": {"rows": result["target_carrier_table"]},
        "g5_carrier_table": {"rows": result["g5_carrier_table"]},
        "reducer_rows": {"rows": result["reducer_rows"]},
        "summary": result["summary"],
    }
    written: dict[str, str] = {}
    for name, path in paths.items():
        path.write_text(pretty_json(payloads[name]), encoding="utf-8")
        written[name] = str(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0", type=Path, required=True)
    parser.add_argument("--g5", type=Path, required=True)
    parser.add_argument("--target-milestone", type=int, default=5)
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
    result = reduce_persistence(
        g0_path=args.g0,
        g5_path=args.g5,
        target_milestone=args.target_milestone,
        authority=_read_json(args.authority),
        source_commit=args.source_commit,
        campaign_id=args.campaign_id,
        cell_id=args.cell_id,
        seed=args.seed,
        evidence_root=args.evidence_root,
    )
    written = write_reducer_outputs(result, args.output_dir)
    print(pretty_json({"ok": True, "result_sha256": result["result_sha256"], "paths": written}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
