"""Immutable raw-state semantic evidence and independent host derivation.

No caller-provided PASS boolean is authority here.  The writer persists exact
state, scratch, event, and count bytes for a fixed set of tiny executions.  The
reader reopens those bytes and derives each transition-family result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import struct
from typing import Any, Final

from .contract import (
    WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
    canonical_sha256,
    workhorse_shape_v2,
)
from .layout import WorkhorseLayoutV2, workhorse_layout_v2
from .native_loader import file_sha256
from .semantic_cases import (
    WORKHORSE_SEMANTIC_CASE_COUNT_V2,
    reopen_terminalized_workhorse_semantic_case_receipt_v2,
    reopen_workhorse_semantic_case_fragment_v2,
    workhorse_semantic_case_table_sha256_v2,
    workhorse_semantic_case_table_v2,
)

WORKHORSE_SEMANTIC_EVIDENCE_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_raw_semantic_evidence_bundle_v2"
)
WORKHORSE_SEMANTIC_CASE_AGGREGATE_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_offline_semantic_case_aggregate_v2"
)

WORKHORSE_SEMANTIC_DATASETS_V2: Final[tuple[str, ...]] = (
    "parity_reference",
    "parity_candidate",
    "b_whole",
    "b_split",
    "k_split",
    "causal_baseline",
    "causal_patch_disabled",
    "causal_wdl_disabled",
    "causal_jungle_disabled",
    "causal_eed_disabled",
    "causal_birth_disabled",
    "causal_mutation_disabled",
    "causal_recombination_disabled",
    "eed_enabled",
    "eed_disabled",
    "wdl_tick1",
    "wdl_tick2",
    "wdl_tick3",
    "overflow_reference",
    "overflow_candidate",
    "middle_survivor_reference",
    "middle_survivor_candidate",
    "restore_checkpoint",
    "restore_reconstructed",
    "restore_same_continuation",
    "restore_fresh_continuation",
    "restore_invalid_before",
    "restore_invalid_after",
    "transaction_after",
)

_SEMANTIC_COMPONENTS_V2: Final[tuple[str, ...]] = (
    "input_state",
    "input_scratch",
    "input_events",
    "input_counts",
    "state",
    "scratch",
    "events",
    "counts",
)

_SEMANTIC_DATASET_EXECUTION_LAWS_V2: Final[dict[str, tuple[str, int, str]]] = {
    "parity_reference": ("REFERENCE", 0, "K1_SEQUENCE"),
    "parity_candidate": ("CANDIDATE", 0, "K_DISPATCH"),
    "b_whole": ("CANDIDATE", 0, "K_DISPATCH"),
    "b_split": ("CANDIDATE", 0, "B_SPLIT_K_DISPATCH"),
    "k_split": ("CANDIDATE", 0, "K1_SEQUENCE"),
    "causal_baseline": ("CANDIDATE", 0, "SINGLE_DISPATCH"),
    "causal_patch_disabled": ("CANDIDATE", 1 << 0, "SINGLE_DISPATCH"),
    "causal_wdl_disabled": ("CANDIDATE", 1 << 1, "SINGLE_DISPATCH"),
    "causal_jungle_disabled": ("CANDIDATE", 1 << 2, "SINGLE_DISPATCH"),
    "causal_eed_disabled": ("CANDIDATE", 1 << 3, "SINGLE_DISPATCH"),
    "causal_birth_disabled": ("CANDIDATE", 1 << 4, "SINGLE_DISPATCH"),
    "causal_mutation_disabled": ("CANDIDATE", 1 << 5, "SINGLE_DISPATCH"),
    "causal_recombination_disabled": (
        "CANDIDATE",
        (1 << 5) | (1 << 6),
        "SINGLE_DISPATCH",
    ),
    "eed_enabled": ("CANDIDATE", 1 << 5, "K1_SEQUENCE"),
    "eed_disabled": ("CANDIDATE", (1 << 5) | (1 << 3), "K1_SEQUENCE"),
    "wdl_tick1": ("CANDIDATE", 0, "WDL_PREFIX_TICK_1"),
    "wdl_tick2": ("CANDIDATE", 0, "WDL_PREFIX_TICK_2"),
    "wdl_tick3": ("CANDIDATE", 0, "WDL_PREFIX_TICK_3"),
    "overflow_reference": ("REFERENCE", 0, "OVERFLOW_FIXTURE_K1"),
    "overflow_candidate": ("CANDIDATE", 0, "OVERFLOW_FIXTURE_K1"),
    "middle_survivor_reference": (
        "REFERENCE",
        0,
        "MIDDLE_SURVIVOR_DEATH_FIXTURE_K1",
    ),
    "middle_survivor_candidate": (
        "CANDIDATE",
        0,
        "MIDDLE_SURVIVOR_DEATH_FIXTURE_K1",
    ),
    "restore_checkpoint": ("CANDIDATE", 0, "CHECKPOINT_DISPATCH"),
    "restore_reconstructed": ("CANDIDATE", 0, "RESTORE_DERIVED"),
    "restore_same_continuation": ("CANDIDATE", 0, "RESTORE_CONTINUATION"),
    "restore_fresh_continuation": (
        "CANDIDATE",
        0,
        "FRESH_PROCESS_RESTORE_CONTINUATION",
    ),
    "restore_invalid_before": ("CANDIDATE", 0, "INVALID_RESTORE_INPUT"),
    "restore_invalid_after": ("CANDIDATE", 0, "INVALID_RESTORE_FAIL_STOP"),
    "transaction_after": ("CANDIDATE", 0, "TRANSACTION_ERROR_FIXTURE_K2"),
}

WORKHORSE_EVIDENCE_DERIVED_CHECKS_V2: Final[frozenset[str]] = frozenset(
    {
        "same_shape_reference_state_byte_parity",
        "same_shape_reference_scratch_byte_parity",
        "same_shape_reference_event_byte_parity",
        "same_shape_reference_event_count_parity",
        "B_partition_state_invariance",
        "B_partition_scratch_invariance",
        "B_partition_event_invariance",
        "K_partition_state_invariance",
        "K_partition_scratch_invariance",
        "K_partition_event_invariance",
        "event_inactive_rows_zero",
        "event_count_exact",
        "event_id_codec_exact",
        "event_full_key_unique",
        "event_absolute_ticks_exact",
        "event_u32_sites_in_shape",
        "birth_event_state_identity_exact",
        "death_event_present",
        "birth_event_present",
        "expressed_cache_matches_diploid_genome",
        "mutation_isolated_to_one_child_locus_and_cache",
        "mutation_disabled_genome_b_unchanged",
        "recombination_full_32_locus_vector_valid",
        "recombination_parent_a_both_homologs",
        "recombination_parent_b_both_homologs",
        "recombination_high_loci_witness",
        "patch_paired_state_effect",
        "patch_paired_downstream_resource_or_agent_effect",
        "wdl_paired_physiology_effect",
        "jungle_paired_physiology_effect",
        "eed_paired_downstream_energy_effect",
        "ordinary_birth_paired_alive_effect",
        "recombination_paired_genome_effect",
        "eed_is_heritable_and_changes_child_outcome",
        "wdl_new_emission_delay_is_exactly_two_ticks",
        "active_order_is_strict_biological_uid_order",
        "birth_reuses_low_slot_but_appends_uid_order",
        "restore_reconstructs_after_arbitrary_derived_corruption",
        "restore_continuation_state_exact",
        "restore_continuation_scratch_exact",
        "restore_continuation_events_exact",
        "fresh_process_restore_continuation_exact",
        "invalid_restore_status_preserved_and_dispatch_fail_stops",
        "overflow_reference_candidate_state_parity",
        "overflow_reference_candidate_scratch_parity",
        "overflow_reference_candidate_event_parity",
        "overflow_truncates_to_single_free_slot",
        "overflow_is_not_status_error",
        "overflow_publishes_one_tick",
        "middle_survivor_reference_candidate_byte_parity",
        "birth_pairs_adjacent_compacted_survivor_uid_order",
        "first_error_status_is_deterministic",
        "failed_dispatch_does_not_publish_frontier",
        "remaining_temporal_ticks_fail_stop",
        "failed_event_buffer_cannot_be_reused",
    }
)

_EVENT = struct.Struct("<QQQQQIHBB")
_DTYPE_FORMAT = {"u8": "B", "u16": "H", "u32": "I", "i32": "i", "u64": "Q"}


class WorkhorseSemanticEvidenceError(RuntimeError):
    """Raised when raw semantic evidence is absent, forged, or contradictory."""


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _valid_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_component_bytes(
    layout: WorkhorseLayoutV2, *, B: int, temporal_k: int
) -> dict[str, int]:
    output = {
        "state": B * layout.state_bytes_per_world,
        "scratch": B * layout.scratch_bytes_per_world,
        "events": B * temporal_k * layout.event_rows_per_world_tick * 48,
        "counts": B * 4,
    }
    return {
        **{f"input_{name}": size for name, size in output.items()},
        **output,
    }


def write_workhorse_semantic_evidence_bundle_v2(
    output_directory: str | Path,
    *,
    shape_id: str,
    temporal_k: int,
    reference_artifact_sha256: str,
    candidate_artifact_sha256: str,
    datasets: Mapping[str, Mapping[str, Any]],
    governing_contracts: Mapping[str, Any],
) -> Path:
    """Persist all raw tiny-run inputs/outputs before a semantic verdict."""
    shape = workhorse_shape_v2(shape_id)
    layout = workhorse_layout_v2(shape)
    if temporal_k <= 1 or temporal_k not in shape.temporal_k_candidates:
        raise WorkhorseSemanticEvidenceError("semantic evidence requires compiled K>1")
    if set(datasets) != set(WORKHORSE_SEMANTIC_DATASETS_V2):
        raise WorkhorseSemanticEvidenceError(
            "semantic evidence dataset inventory is incomplete or contains extras"
        )
    if not _valid_sha(reference_artifact_sha256) or not _valid_sha(
        candidate_artifact_sha256
    ):
        raise WorkhorseSemanticEvidenceError("semantic artifact hash is invalid")
    root = Path(output_directory)
    if root.exists() or root.is_symlink():
        raise WorkhorseSemanticEvidenceError("semantic evidence root must be fresh")
    root.mkdir(parents=True, mode=0o700)
    rows: dict[str, Any] = {}
    for name in WORKHORSE_SEMANTIC_DATASETS_V2:
        dataset = datasets[name]
        if type(dataset) is not dict or set(dataset) != {
            "B",
            "temporal_k",
            "start_next_absolute_tick",
            *_SEMANTIC_COMPONENTS_V2,
        }:
            raise WorkhorseSemanticEvidenceError(f"dataset {name} fields are inexact")
        B = dataset["B"]
        K = dataset["temporal_k"]
        start = dataset["start_next_absolute_tick"]
        if (
            type(B) is not int
            or B <= 0
            or type(K) is not int
            or K <= 0
            or type(start) is not int
            or start < 0
        ):
            raise WorkhorseSemanticEvidenceError(f"dataset {name} coordinates invalid")
        expected = _expected_component_bytes(layout, B=B, temporal_k=K)
        component_rows: dict[str, Any] = {}
        for component, expected_bytes in expected.items():
            payload = dataset[component]
            if type(payload) is not bytes or len(payload) != expected_bytes:
                raise WorkhorseSemanticEvidenceError(
                    f"dataset {name}.{component} byte length is invalid"
                )
            path = root / f"{name}.{component}.bin"
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            path.chmod(0o444)
            component_rows[component] = {
                "file_name": path.name,
                "bytes": len(payload),
                "sha256": _sha_bytes(payload),
            }
        rows[name] = {
            "B": B,
            "temporal_k": K,
            "start_next_absolute_tick": start,
            "execution_law": {
                "artifact_role": _SEMANTIC_DATASET_EXECUTION_LAWS_V2[name][0],
                "disable_mask": _SEMANTIC_DATASET_EXECUTION_LAWS_V2[name][1],
                "partition": _SEMANTIC_DATASET_EXECUTION_LAWS_V2[name][2],
            },
            "components": component_rows,
        }
    body = {
        "schema_version": WORKHORSE_SEMANTIC_EVIDENCE_SCHEMA_V2,
        "shape_id": shape_id,
        "layout_sha256": layout.layout_sha256,
        "semantic_transition_sha256": WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
        "temporal_k": temporal_k,
        "reference_artifact_sha256": reference_artifact_sha256,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "governing_contracts": dict(governing_contracts),
        "datasets": rows,
    }
    manifest = {**body, "bundle_body_sha256": canonical_sha256(body)}
    path = root / "semantic_evidence.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    _fsync_directory(root)
    root.chmod(0o555)
    _fsync_directory(root.parent)
    reopen_workhorse_semantic_evidence_bundle_v2(path)
    return path


def reopen_workhorse_semantic_evidence_bundle_v2(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(manifest_path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.name != "semantic_evidence.json"
        or path.stat().st_mode & 0o222
        or path.parent.stat().st_mode & 0o222
    ):
        raise WorkhorseSemanticEvidenceError(
            "semantic evidence manifest/root must be immutable regular content"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkhorseSemanticEvidenceError("semantic evidence is unreadable") from exc
    if type(value) is not dict:
        raise WorkhorseSemanticEvidenceError("semantic evidence must be an object")
    body = {key: item for key, item in value.items() if key != "bundle_body_sha256"}
    shape = workhorse_shape_v2(value.get("shape_id"))
    layout = workhorse_layout_v2(shape)
    rows = value.get("datasets")
    if (
        value.get("schema_version") != WORKHORSE_SEMANTIC_EVIDENCE_SCHEMA_V2
        or value.get("bundle_body_sha256") != canonical_sha256(body)
        or value.get("layout_sha256") != layout.layout_sha256
        or value.get("semantic_transition_sha256")
        != WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2
        or set(rows or {}) != set(WORKHORSE_SEMANTIC_DATASETS_V2)
    ):
        raise WorkhorseSemanticEvidenceError(
            "semantic evidence identity, inventory, or seal is invalid"
        )
    reopened: dict[str, dict[str, Any]] = {}
    for name in WORKHORSE_SEMANTIC_DATASETS_V2:
        row = rows[name]
        if type(row) is not dict or set(row) != {
            "B",
            "temporal_k",
            "start_next_absolute_tick",
            "execution_law",
            "components",
        }:
            raise WorkhorseSemanticEvidenceError(f"dataset {name} metadata is inexact")
        B, K, start = row["B"], row["temporal_k"], row["start_next_absolute_tick"]
        if (
            type(B) is not int
            or B <= 0
            or type(K) is not int
            or K <= 0
            or type(start) is not int
            or start < 0
        ):
            raise WorkhorseSemanticEvidenceError(f"dataset {name} coordinates invalid")
        expected = _expected_component_bytes(layout, B=B, temporal_k=K)
        expected_law = _SEMANTIC_DATASET_EXECUTION_LAWS_V2[name]
        if row["execution_law"] != {
            "artifact_role": expected_law[0],
            "disable_mask": expected_law[1],
            "partition": expected_law[2],
        }:
            raise WorkhorseSemanticEvidenceError(
                f"dataset {name} execution law is not canonical"
            )
        components = row["components"]
        if type(components) is not dict or set(components) != set(expected):
            raise WorkhorseSemanticEvidenceError(f"dataset {name} components inexact")
        payloads: dict[str, Any] = {
            "B": B,
            "temporal_k": K,
            "start_next_absolute_tick": start,
        }
        for component, expected_bytes in expected.items():
            item = components[component]
            if type(item) is not dict or set(item) != {
                "file_name",
                "bytes",
                "sha256",
            }:
                raise WorkhorseSemanticEvidenceError(
                    f"dataset {name}.{component} identity is inexact"
                )
            file_name = item["file_name"]
            target = path.parent / file_name
            if (
                type(file_name) is not str
                or Path(file_name).name != file_name
                or target.is_symlink()
                or not target.is_file()
                or target.stat().st_mode & 0o222
            ):
                raise WorkhorseSemanticEvidenceError(
                    f"dataset {name}.{component} locator is unsafe"
                )
            payload = target.read_bytes()
            if (
                item["bytes"] != expected_bytes
                or len(payload) != expected_bytes
                or item["sha256"] != _sha_bytes(payload)
            ):
                raise WorkhorseSemanticEvidenceError(
                    f"dataset {name}.{component} exact-content seal failed"
                )
            payloads[component] = payload
        reopened[name] = payloads
    return value, reopened


def _write_streamed_component_v2(
    target: Path, payload_parts: Sequence[bytes]
) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with target.open("xb") as handle:
        for payload in payload_parts:
            if type(payload) is not bytes:
                raise WorkhorseSemanticEvidenceError(
                    "semantic aggregate source component is not exact bytes"
                )
            handle.write(payload)
            digest.update(payload)
            size += len(payload)
        handle.flush()
        os.fsync(handle.fileno())
    target.chmod(0o444)
    return {"file_name": target.name, "bytes": size, "sha256": digest.hexdigest()}


def workhorse_semantic_dataset_projection_v2(
    semantic_temporal_k: int,
) -> dict[str, tuple[str, str]]:
    """Map every final dataset to one physical case/dataset or the B merger."""

    final_sources: dict[str, tuple[str, str]] = {}
    for case in workhorse_semantic_case_table_v2(semantic_temporal_k):
        for dataset in case.fragment_datasets:
            if not dataset.startswith("__"):
                if dataset in final_sources:
                    raise WorkhorseSemanticEvidenceError(
                        f"semantic dataset {dataset} has multiple physical producers"
                    )
                final_sources[dataset] = (case.case_id, dataset)
    final_sources["b_whole"] = (
        "parity_candidate_k_dispatch",
        "parity_candidate",
    )
    final_sources["b_split"] = ("__B_SPLIT__", "__B_SPLIT__")
    if set(final_sources) != set(WORKHORSE_SEMANTIC_DATASETS_V2):
        raise WorkhorseSemanticEvidenceError(
            "24-case projection does not close the exact 29-dataset inventory"
        )
    return final_sources


def aggregate_workhorse_semantic_case_receipts_v2(
    output_directory: str | Path,
    *,
    terminal_receipt_paths: Sequence[str | Path],
    governing_contracts: Mapping[str, Any],
) -> Path:
    """Stream 24 terminal case receipts into the exact 29-dataset V2 bundle.

    This phase is GPU-free.  It reopens one tiny case fragment at a time, makes
    the parity-candidate alias explicit, concatenates the two B=1 shards in
    world order, and seals the supplemental reused-buffer P2 witness beside the
    legacy evidence manifest.
    """

    if len(terminal_receipt_paths) != WORKHORSE_SEMANTIC_CASE_COUNT_V2:
        raise WorkhorseSemanticEvidenceError(
            "semantic aggregate requires exactly 24 terminal receipts"
        )
    receipts: dict[str, dict[str, Any]] = {}
    case_fragments: dict[str, dict[str, Any]] = {}
    semantic_temporal_k: int | None = None
    shape_id: str | None = None
    attempt_ids: set[str] = set()
    for supplied_path in terminal_receipt_paths:
        try:
            receipt = reopen_terminalized_workhorse_semantic_case_receipt_v2(
                supplied_path
            )
            fragment, _ = reopen_workhorse_semantic_case_fragment_v2(
                receipt["case_fragment_path"]
            )
        except Exception as exc:
            raise WorkhorseSemanticEvidenceError(
                f"semantic terminal receipt cannot be aggregated: {exc}"
            ) from exc
        case_id = receipt["case_id"]
        attempt_id = receipt["canonical_parent_binding"]["attempt_id"]
        if case_id in receipts or attempt_id in attempt_ids:
            raise WorkhorseSemanticEvidenceError(
                "semantic cases must have one distinct Scheduler Attempt each"
            )
        attempt_ids.add(attempt_id)
        receipts[case_id] = receipt
        case_fragments[case_id] = fragment
        if semantic_temporal_k is None:
            semantic_temporal_k = receipt["semantic_temporal_k"]
            shape_id = receipt["shape_id"]
        elif (
            receipt["semantic_temporal_k"] != semantic_temporal_k
            or receipt["shape_id"] != shape_id
        ):
            raise WorkhorseSemanticEvidenceError(
                "semantic receipts do not share one shape/K table"
            )
    assert semantic_temporal_k is not None and shape_id is not None
    table = workhorse_semantic_case_table_v2(semantic_temporal_k)
    if set(receipts) != {case.case_id for case in table}:
        raise WorkhorseSemanticEvidenceError(
            "semantic receipt case inventory is incomplete or contains extras"
        )
    if (
        case_fragments["fresh_restore_continuation"]["producer_pid"]
        == case_fragments["checkpoint_same_restore_sequence"]["producer_pid"]
    ):
        raise WorkhorseSemanticEvidenceError(
            "fresh restore continuation was not produced in a fresh process"
        )
    reference_hashes = {
        receipts[case.case_id]["artifact_sha256"]
        for case in table
        if case.artifact_role == "REFERENCE"
    }
    candidate_hashes = {
        receipts[case.case_id]["artifact_sha256"]
        for case in table
        if case.artifact_role == "CANDIDATE"
    }
    device_uuids = {
        receipt["canonical_parent_binding"]["device_uuid"]
        for receipt in receipts.values()
    }
    reference_backends = {
        receipts[case.case_id]["canonical_parent_binding"]["backend_id"]
        for case in table
        if case.artifact_role == "REFERENCE"
    }
    candidate_backends = {
        receipts[case.case_id]["canonical_parent_binding"]["backend_id"]
        for case in table
        if case.artifact_role == "CANDIDATE"
    }
    if (
        len(reference_hashes) != 1
        or len(candidate_hashes) != 1
        or len(device_uuids) != 1
        or reference_backends != {"reference_phased_v2"}
        or len(candidate_backends) != 1
        or next(iter(candidate_backends))
        not in {"phase_graph_v2", "fused_cta_temporal_v2"}
    ):
        raise WorkhorseSemanticEvidenceError(
            "semantic cases do not bind one reference/candidate artifact and device"
        )
    reference_artifact_sha256 = next(iter(reference_hashes))
    candidate_artifact_sha256 = next(iter(candidate_hashes))
    root = Path(output_directory)
    if root.exists() or root.is_symlink():
        raise WorkhorseSemanticEvidenceError("semantic aggregate root must be fresh")
    root.mkdir(parents=True, mode=0o700)
    final_sources = workhorse_semantic_dataset_projection_v2(semantic_temporal_k)

    rows: dict[str, Any] = {}
    for name in WORKHORSE_SEMANTIC_DATASETS_V2:
        source_case, source_dataset = final_sources[name]
        if source_case == "__B_SPLIT__":
            fragments = []
            for shard_case in (
                "b_split_world_0_k_dispatch",
                "b_split_world_1_k_dispatch",
            ):
                _, data = reopen_workhorse_semantic_case_fragment_v2(
                    receipts[shard_case]["case_fragment_path"]
                )
                fragments.append(data[f"__b_split_world_{len(fragments)}"])
            if any(
                fragment["B"] != 1
                or fragment["temporal_k"] != semantic_temporal_k
                or fragment["start_next_absolute_tick"] != 71
                for fragment in fragments
            ):
                raise WorkhorseSemanticEvidenceError(
                    "B-split fragments differ from the exact shard law"
                )
            B, dataset_k, start = 2, semantic_temporal_k, 71
            payload_parts = {
                component: [fragment[component] for fragment in fragments]
                for component in _SEMANTIC_COMPONENTS_V2
            }
        else:
            _, data = reopen_workhorse_semantic_case_fragment_v2(
                receipts[source_case]["case_fragment_path"]
            )
            dataset = data[source_dataset]
            B = dataset["B"]
            dataset_k = dataset["temporal_k"]
            start = dataset["start_next_absolute_tick"]
            payload_parts = {
                component: [dataset[component]]
                for component in _SEMANTIC_COMPONENTS_V2
            }
        expected = _expected_component_bytes(
            workhorse_layout_v2(workhorse_shape_v2(shape_id)),
            B=B,
            temporal_k=dataset_k,
        )
        component_rows: dict[str, Any] = {}
        for component in _SEMANTIC_COMPONENTS_V2:
            row = _write_streamed_component_v2(
                root / f"{name}.{component}.bin", payload_parts[component]
            )
            if row["bytes"] != expected[component]:
                raise WorkhorseSemanticEvidenceError(
                    f"aggregated {name}.{component} byte length is invalid"
                )
            component_rows[component] = row
        law = _SEMANTIC_DATASET_EXECUTION_LAWS_V2[name]
        rows[name] = {
            "B": B,
            "temporal_k": dataset_k,
            "start_next_absolute_tick": start,
            "execution_law": {
                "artifact_role": law[0],
                "disable_mask": law[1],
                "partition": law[2],
            },
            "components": component_rows,
        }

    layout = workhorse_layout_v2(workhorse_shape_v2(shape_id))
    evidence_body = {
        "schema_version": WORKHORSE_SEMANTIC_EVIDENCE_SCHEMA_V2,
        "shape_id": shape_id,
        "layout_sha256": layout.layout_sha256,
        "semantic_transition_sha256": WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
        "temporal_k": semantic_temporal_k,
        "reference_artifact_sha256": reference_artifact_sha256,
        "candidate_artifact_sha256": candidate_artifact_sha256,
        "governing_contracts": dict(governing_contracts),
        "datasets": rows,
    }
    evidence = {
        **evidence_body,
        "bundle_body_sha256": canonical_sha256(evidence_body),
    }
    evidence_path = root / "semantic_evidence.json"
    with evidence_path.open("x", encoding="ascii") as handle:
        handle.write(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    evidence_path.chmod(0o444)

    p2_fragment = case_fragments["event_reuse_canonicalization_regression"]
    p2_body = {
        "schema_version": "eiel_workhorse_supplemental_p2_projection_v2",
        "claim_class": "TECHNICAL_ONLY_RAW_REGRESSION",
        "qualification_authority_granted": False,
        "science_authority_granted": False,
        "case_id": "event_reuse_canonicalization_regression",
        "terminal_receipt_content_sha256": receipts[
            "event_reuse_canonicalization_regression"
        ]["receipt_content_sha256"],
        "case_fragment_body_sha256": p2_fragment["fragment_body_sha256"],
        "witness": p2_fragment["p2_witness"],
    }
    p2_value = {**p2_body, "projection_sha256": canonical_sha256(p2_body)}
    p2_path = root / "supplemental_p2_witness.json"
    with p2_path.open("x", encoding="ascii") as handle:
        handle.write(json.dumps(p2_value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    p2_path.chmod(0o444)

    receipt_rows = [
        {
            "case_id": case.case_id,
            "attempt_id": receipts[case.case_id]["canonical_parent_binding"][
                "attempt_id"
            ],
            "result_class": receipts[case.case_id]["result_class"],
            "receipt_path": receipts[case.case_id]["receipt_path"],
            "receipt_file_sha256": receipts[case.case_id]["receipt_file_sha256"],
            "receipt_content_sha256": receipts[case.case_id][
                "receipt_content_sha256"
            ],
        }
        for case in table
    ]
    aggregate_body = {
        "schema_version": WORKHORSE_SEMANTIC_CASE_AGGREGATE_SCHEMA_V2,
        "claim_class": "TECHNICAL_ONLY_RAW_EVIDENCE",
        "qualification_authority_granted": False,
        "science_authority_granted": False,
        "observation": {
            "mode": "OFF",
            "science_records_emitted": 0,
            "science_observation_schedule": None,
            "population_change_detection": False,
        },
        "case_count": WORKHORSE_SEMANTIC_CASE_COUNT_V2,
        "case_table_sha256": workhorse_semantic_case_table_sha256_v2(
            semantic_temporal_k
        ),
        "shape_id": shape_id,
        "semantic_temporal_k": semantic_temporal_k,
        "terminal_receipts": receipt_rows,
        "semantic_evidence": {
            "manifest_path": str(evidence_path.resolve()),
            "manifest_file_sha256": file_sha256(evidence_path),
            "bundle_body_sha256": evidence["bundle_body_sha256"],
            "dataset_count": len(rows),
        },
        "supplemental_p2_witness": {
            "path": str(p2_path.resolve()),
            "file_sha256": file_sha256(p2_path),
            "projection_sha256": p2_value["projection_sha256"],
        },
        "aggregation": "OFFLINE_STREAMING_NO_GPU",
    }
    aggregate = {
        **aggregate_body,
        "aggregate_content_sha256": canonical_sha256(aggregate_body),
    }
    aggregate_path = root / "semantic_case_aggregate.json"
    with aggregate_path.open("x", encoding="ascii") as handle:
        handle.write(json.dumps(aggregate, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    aggregate_path.chmod(0o444)
    _fsync_directory(root)
    root.chmod(0o555)
    _fsync_directory(root.parent)
    reopen_workhorse_semantic_evidence_bundle_v2(evidence_path)
    reopen_workhorse_semantic_case_aggregate_v2(aggregate_path)
    return aggregate_path


def reopen_workhorse_semantic_case_aggregate_v2(
    aggregate_path: str | Path,
) -> dict[str, Any]:
    path = Path(aggregate_path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.name != "semantic_case_aggregate.json"
        or path.stat().st_mode & 0o222
        or path.parent.stat().st_mode & 0o222
    ):
        raise WorkhorseSemanticEvidenceError(
            "semantic case aggregate/root must be immutable regular content"
        )
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkhorseSemanticEvidenceError(
            "semantic case aggregate is unreadable"
        ) from exc
    if type(value) is not dict:
        raise WorkhorseSemanticEvidenceError("semantic case aggregate is not an object")
    body = {
        key: item for key, item in value.items() if key != "aggregate_content_sha256"
    }
    receipts = value.get("terminal_receipts")
    semantic = value.get("semantic_evidence")
    p2 = value.get("supplemental_p2_witness")
    if (
        value.get("schema_version") != WORKHORSE_SEMANTIC_CASE_AGGREGATE_SCHEMA_V2
        or value.get("claim_class") != "TECHNICAL_ONLY_RAW_EVIDENCE"
        or value.get("qualification_authority_granted") is not False
        or value.get("science_authority_granted") is not False
        or value.get("aggregation") != "OFFLINE_STREAMING_NO_GPU"
        or value.get("case_count") != WORKHORSE_SEMANTIC_CASE_COUNT_V2
        or value.get("case_table_sha256")
        != workhorse_semantic_case_table_sha256_v2(value.get("semantic_temporal_k"))
        or type(receipts) is not list
        or len(receipts) != WORKHORSE_SEMANTIC_CASE_COUNT_V2
        or type(semantic) is not dict
        or type(p2) is not dict
        or value.get("aggregate_content_sha256") != canonical_sha256(body)
    ):
        raise WorkhorseSemanticEvidenceError("semantic case aggregate seal changed")
    table = workhorse_semantic_case_table_v2(value["semantic_temporal_k"])
    if [row.get("case_id") for row in receipts] != [case.case_id for case in table]:
        raise WorkhorseSemanticEvidenceError("semantic aggregate case order changed")
    reopened_receipts = [
        reopen_terminalized_workhorse_semantic_case_receipt_v2(row["receipt_path"])
        for row in receipts
    ]
    for row, reopened, case in zip(receipts, reopened_receipts, table):
        if row != {
            "case_id": case.case_id,
            "attempt_id": reopened["canonical_parent_binding"]["attempt_id"],
            "result_class": reopened["result_class"],
            "receipt_path": reopened["receipt_path"],
            "receipt_file_sha256": reopened["receipt_file_sha256"],
            "receipt_content_sha256": reopened["receipt_content_sha256"],
        }:
            raise WorkhorseSemanticEvidenceError(
                "semantic aggregate terminal receipt projection changed"
            )
    evidence_manifest, _ = reopen_workhorse_semantic_evidence_bundle_v2(
        semantic.get("manifest_path")
    )
    if semantic != {
        "manifest_path": str(Path(semantic["manifest_path"]).resolve()),
        "manifest_file_sha256": file_sha256(Path(semantic["manifest_path"])),
        "bundle_body_sha256": evidence_manifest["bundle_body_sha256"],
        "dataset_count": len(WORKHORSE_SEMANTIC_DATASETS_V2),
    }:
        raise WorkhorseSemanticEvidenceError(
            "semantic aggregate evidence projection changed"
        )
    p2_path = Path(p2.get("path", ""))
    try:
        p2_value = json.loads(p2_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkhorseSemanticEvidenceError(
            "semantic aggregate P2 witness cannot reopen"
        ) from exc
    p2_body = {
        key: item for key, item in p2_value.items() if key != "projection_sha256"
    }
    if (
        p2_path.is_symlink()
        or not p2_path.is_file()
        or p2_path.stat().st_mode & 0o222
        or p2.get("file_sha256") != file_sha256(p2_path)
        or p2.get("projection_sha256") != p2_value.get("projection_sha256")
        or p2_value.get("projection_sha256") != canonical_sha256(p2_body)
        or p2_body.get("case_id") != "event_reuse_canonicalization_regression"
    ):
        raise WorkhorseSemanticEvidenceError(
            "semantic aggregate supplemental P2 projection changed"
        )
    p2_index = next(
        index
        for index, case in enumerate(table)
        if case.case_id == "event_reuse_canonicalization_regression"
    )
    p2_receipt = reopened_receipts[p2_index]
    p2_fragment, _ = reopen_workhorse_semantic_case_fragment_v2(
        p2_receipt["case_fragment_path"]
    )
    if (
        p2_body.get("terminal_receipt_content_sha256")
        != p2_receipt["receipt_content_sha256"]
        or p2_body.get("case_fragment_body_sha256")
        != p2_fragment["fragment_body_sha256"]
        or p2_body.get("witness") != p2_fragment["p2_witness"]
    ):
        raise WorkhorseSemanticEvidenceError(
            "semantic aggregate P2 projection differs from its terminal case"
        )
    return value


def _field_values(
    dataset: Mapping[str, Any],
    layout: WorkhorseLayoutV2,
    name: str,
    world: int = 0,
    *,
    component: str = "state",
) -> tuple[int, ...]:
    field = layout.field(name)
    start = world * layout.state_bytes_per_world + field.offset
    payload = dataset[component][start : start + field.nbytes]
    if field.count == 0:
        return ()
    return struct.unpack(f"<{field.count}{_DTYPE_FORMAT[field.dtype]}", payload)


def _events(
    dataset: Mapping[str, Any], layout: WorkhorseLayoutV2
) -> tuple[list[dict[str, int]], tuple[int, ...]]:
    B, K = dataset["B"], dataset["temporal_k"]
    rows_per_tick = layout.event_rows_per_world_tick
    rows: list[dict[str, int]] = []
    payload = dataset["events"]
    for world in range(B):
        for local_tick in range(K):
            for row_index in range(rows_per_tick):
                offset = ((world * K + local_tick) * rows_per_tick + row_index) * 48
                values = _EVENT.unpack_from(payload, offset)
                rows.append(
                    dict(
                        zip(
                            (
                                "tick",
                                "event_id",
                                "agent_id",
                                "parent_a_id",
                                "parent_b_id",
                                "site",
                                "kind",
                                "status",
                                "reserved",
                            ),
                            values,
                        ),
                        world=world,
                        local_tick=local_tick,
                        row_index=row_index,
                        raw_zero=int(payload[offset : offset + 48] == bytes(48)),
                    )
                )
    counts = struct.unpack(f"<{B}i", dataset["counts"])
    return rows, counts


def _field_changed(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    layout: WorkhorseLayoutV2,
    name: str,
) -> bool:
    return _field_values(left, layout, name) != _field_values(right, layout, name)


def _inputs_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left[f"input_{component}"] == right[f"input_{component}"]
        for component in ("state", "scratch", "events", "counts")
    )


def _without_state_field(
    payload: bytes, layout: WorkhorseLayoutV2, field_name: str
) -> bytes:
    field = layout.field(field_name)
    return payload[: field.offset] + payload[field.offset + field.nbytes :]


def derive_workhorse_semantic_checks_v2(
    manifest_path: str | Path,
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Derive transition checks solely from reopened persisted raw bytes."""
    manifest_path = Path(manifest_path)
    manifest, data = reopen_workhorse_semantic_evidence_bundle_v2(manifest_path)
    shape = workhorse_shape_v2(manifest["shape_id"])
    layout = workhorse_layout_v2(shape)
    temporal_k = manifest["temporal_k"]
    parity_ref, parity_cand = data["parity_reference"], data["parity_candidate"]
    b_whole, b_split = data["b_whole"], data["b_split"]
    k_split = data["k_split"]
    baseline = data["causal_baseline"]
    if (
        parity_ref["B"] != 2
        or parity_cand["B"] != 2
        or parity_ref["temporal_k"] != temporal_k
        or parity_cand["temporal_k"] != temporal_k
        or b_whole["B"] != 2
        or b_split["B"] != 2
        or k_split["B"] != 1
        or k_split["temporal_k"] != temporal_k
        or baseline["B"] != 1
        or baseline["temporal_k"] != 1
    ):
        raise WorkhorseSemanticEvidenceError(
            "semantic parity/partition dataset coordinates are invalid"
        )
    checks: dict[str, bool] = {
        "same_shape_reference_state_byte_parity": parity_ref["input_state"]
        == parity_cand["input_state"]
        and parity_ref["state"] == parity_cand["state"],
        "same_shape_reference_scratch_byte_parity": parity_ref["input_scratch"]
        == parity_cand["input_scratch"]
        and parity_ref["scratch"] == parity_cand["scratch"],
        "same_shape_reference_event_byte_parity": parity_ref["input_events"]
        == parity_cand["input_events"]
        and parity_ref["events"] == parity_cand["events"],
        "same_shape_reference_event_count_parity": parity_ref["input_counts"]
        == parity_cand["input_counts"]
        and parity_ref["counts"] == parity_cand["counts"],
        "B_partition_state_invariance": b_whole["input_state"] == b_split["input_state"]
        and b_whole["state"] == b_split["state"],
        "B_partition_scratch_invariance": b_whole["input_scratch"]
        == b_split["input_scratch"]
        and b_whole["scratch"] == b_split["scratch"],
        "B_partition_event_invariance": b_whole["input_events"]
        == b_split["input_events"]
        and b_whole["input_counts"] == b_split["input_counts"]
        and b_whole["events"] == b_split["events"]
        and b_whole["counts"] == b_split["counts"],
        "K_partition_state_invariance": parity_cand["input_state"][
            : layout.state_bytes_per_world
        ]
        == k_split["input_state"]
        and parity_cand["state"][: layout.state_bytes_per_world] == k_split["state"],
        "K_partition_scratch_invariance": parity_cand["input_scratch"][
            : layout.scratch_bytes_per_world
        ]
        == k_split["input_scratch"]
        and parity_cand["scratch"][: layout.scratch_bytes_per_world]
        == k_split["scratch"],
        "K_partition_event_invariance": parity_cand["input_events"][
            : temporal_k * layout.event_bytes_per_world_tick
        ]
        == k_split["input_events"]
        and parity_cand["input_counts"][:4] == k_split["input_counts"]
        and parity_cand["events"][: temporal_k * layout.event_bytes_per_world_tick]
        == k_split["events"]
        and parity_cand["counts"][:4] == k_split["counts"],
    }

    rows, counts = _events(baseline, layout)
    header = _field_values(baseline, layout, "header")
    world_uid = header[0]
    agent_ids = _field_values(baseline, layout, "agent_id")
    parent_a_ids = _field_values(baseline, layout, "parent_a_id")
    parent_b_ids = _field_values(baseline, layout, "parent_b_id")
    positions = _field_values(baseline, layout, "position")
    active = [row for row in rows if row["status"] != 0]
    inactive = [row for row in rows if row["status"] == 0]
    death_rows = [row for row in active if row["kind"] == 1]
    birth_rows = [row for row in active if row["kind"] == 2]
    birth_slots = [row["row_index"] - shape.capacity for row in birth_rows]
    full_keys = {(world_uid, row["tick"], row["event_id"]) for row in active}
    codec = True
    identity = True
    for row in active:
        slot = (
            row["row_index"] if row["kind"] == 1 else row["row_index"] - shape.capacity
        )
        codec = codec and (
            row["status"] == 1
            and row["reserved"] == 0
            and row["kind"] in (1, 2)
            and 0 <= slot < shape.capacity
            and row["event_id"] == (row["kind"] << 48) | slot
        )
        if row["kind"] == 2 and 0 <= slot < shape.capacity:
            identity = identity and (
                agent_ids[slot] == row["agent_id"]
                and parent_a_ids[slot] == row["parent_a_id"]
                and parent_b_ids[slot] == row["parent_b_id"]
                and positions[slot] == row["site"]
            )
    checks.update(
        {
            "event_inactive_rows_zero": all(row["raw_zero"] for row in inactive)
            and baseline["input_events"] == bytes(len(baseline["input_events"]))
            and baseline["input_counts"] == bytes(len(baseline["input_counts"])),
            "event_count_exact": counts == (len(active),),
            "event_id_codec_exact": codec,
            "event_full_key_unique": len(full_keys) == len(active),
            "event_absolute_ticks_exact": all(
                row["tick"] == baseline["start_next_absolute_tick"] + row["local_tick"]
                for row in active
            ),
            "event_u32_sites_in_shape": all(
                row["site"] < shape.sites for row in active
            ),
            "birth_event_state_identity_exact": identity,
            "death_event_present": bool(death_rows),
            "birth_event_present": bool(birth_rows),
        }
    )

    mutation_disabled = data["causal_mutation_disabled"]
    recombination_disabled = data["causal_recombination_disabled"]
    ga = _field_values(baseline, layout, "genome_a")
    gb = _field_values(baseline, layout, "genome_b")
    expressed = _field_values(baseline, layout, "expressed")
    alive = _field_values(baseline, layout, "alive")
    ga_nomut = _field_values(mutation_disabled, layout, "genome_a")
    gb_nomut = _field_values(mutation_disabled, layout, "genome_b")
    expressed_nomut = _field_values(mutation_disabled, layout, "expressed")
    ids_nomut = _field_values(mutation_disabled, layout, "agent_id")
    pa_ids_nomut = _field_values(mutation_disabled, layout, "parent_a_id")
    pb_ids_nomut = _field_values(mutation_disabled, layout, "parent_b_id")
    mutation_isolated = bool(birth_slots)
    donor_full = bool(birth_slots)
    donor_a_both = donor_b_both = high_locus = False
    for slot in birth_slots:
        offset = slot * 32
        diffs = [
            locus
            for locus in range(32)
            if ga[offset + locus] != ga_nomut[offset + locus]
        ]
        cache_diffs = [
            locus
            for locus in range(32)
            if expressed[offset + locus] != expressed_nomut[offset + locus]
        ]
        mutation_isolated = (
            mutation_isolated and len(diffs) == 1 and diffs == cache_diffs
        )
        mutation_isolated = (
            mutation_isolated
            and gb[offset : offset + 32] == gb_nomut[offset : offset + 32]
        )
        try:
            pa = ids_nomut.index(pa_ids_nomut[slot])
            pb = ids_nomut.index(pb_ids_nomut[slot])
        except ValueError:
            donor_full = False
            continue
        child_a = ga_nomut[offset : offset + 32]
        child_b = gb_nomut[offset : offset + 32]
        pa_a = ga_nomut[pa * 32 : pa * 32 + 32]
        pa_b = gb_nomut[pa * 32 : pa * 32 + 32]
        pb_a = ga_nomut[pb * 32 : pb * 32 + 32]
        pb_b = gb_nomut[pb * 32 : pb * 32 + 32]
        donor_full = donor_full and all(
            child_a[i] in (pa_a[i], pa_b[i]) and child_b[i] in (pb_a[i], pb_b[i])
            for i in range(32)
        )
        donor_a_both = donor_a_both or (
            any(pa_a[i] != pa_b[i] and child_a[i] == pa_a[i] for i in range(32))
            and any(pa_a[i] != pa_b[i] and child_a[i] == pa_b[i] for i in range(32))
        )
        donor_b_both = donor_b_both or (
            any(pb_a[i] != pb_b[i] and child_b[i] == pb_a[i] for i in range(32))
            and any(pb_a[i] != pb_b[i] and child_b[i] == pb_b[i] for i in range(32))
        )
        high_locus = high_locus or all(
            child_a[i] in (pa_a[i], pa_b[i]) and child_b[i] in (pb_a[i], pb_b[i])
            for i in range(16, 32)
        )
    cache_exact = all(
        not alive[slot]
        or all(
            expressed[slot * 32 + locus]
            == (ga[slot * 32 + locus] + gb[slot * 32 + locus]) // 2
            for locus in range(32)
        )
        for slot in range(shape.capacity)
    )
    checks.update(
        {
            "expressed_cache_matches_diploid_genome": cache_exact,
            "mutation_isolated_to_one_child_locus_and_cache": mutation_isolated
            and _inputs_equal(baseline, mutation_disabled),
            "mutation_disabled_genome_b_unchanged": gb == gb_nomut
            and _inputs_equal(baseline, mutation_disabled),
            "recombination_full_32_locus_vector_valid": donor_full
            and _inputs_equal(baseline, mutation_disabled),
            "recombination_parent_a_both_homologs": donor_a_both
            and _inputs_equal(baseline, mutation_disabled),
            "recombination_parent_b_both_homologs": donor_b_both
            and _inputs_equal(baseline, mutation_disabled),
            "recombination_high_loci_witness": high_locus
            and _inputs_equal(baseline, mutation_disabled),
            "patch_paired_state_effect": _field_changed(
                baseline, data["causal_patch_disabled"], layout, "patch_field"
            )
            and _inputs_equal(baseline, data["causal_patch_disabled"]),
            "patch_paired_downstream_resource_or_agent_effect": (
                _field_changed(
                    baseline, data["causal_patch_disabled"], layout, "resource"
                )
                or _field_changed(
                    baseline, data["causal_patch_disabled"], layout, "energy"
                )
            )
            and _inputs_equal(baseline, data["causal_patch_disabled"]),
            "wdl_paired_physiology_effect": _field_changed(
                baseline, data["causal_wdl_disabled"], layout, "health"
            )
            and _inputs_equal(baseline, data["causal_wdl_disabled"]),
            "jungle_paired_physiology_effect": _field_changed(
                baseline, data["causal_jungle_disabled"], layout, "health"
            )
            and _inputs_equal(baseline, data["causal_jungle_disabled"]),
            "eed_paired_downstream_energy_effect": _field_changed(
                baseline, data["causal_eed_disabled"], layout, "energy"
            )
            and _inputs_equal(baseline, data["causal_eed_disabled"]),
            "ordinary_birth_paired_alive_effect": _field_changed(
                baseline, data["causal_birth_disabled"], layout, "alive"
            )
            and _inputs_equal(baseline, data["causal_birth_disabled"]),
            "recombination_paired_genome_effect": _field_changed(
                mutation_disabled, recombination_disabled, layout, "genome_a"
            )
            and _inputs_equal(mutation_disabled, recombination_disabled),
        }
    )
    eed_enabled = data["eed_enabled"]
    eed_disabled = data["eed_disabled"]
    eed_expressed = _field_values(eed_enabled, layout, "expressed")
    eed_energy = _field_values(eed_enabled, layout, "energy")
    no_eed_energy = _field_values(eed_disabled, layout, "energy")
    checks["eed_is_heritable_and_changes_child_outcome"] = (
        bool(birth_slots)
        and all(eed_expressed[slot * 32 + 31] > 0 for slot in birth_slots)
        and any(eed_energy[slot] != no_eed_energy[slot] for slot in birth_slots)
        and _inputs_equal(eed_enabled, eed_disabled)
    )
    wdl_sums = [
        sum(_field_values(data[f"wdl_tick{tick}"], layout, "wdl_active"))
        for tick in (1, 2, 3)
    ]
    wdl_chain_exact = all(
        data[f"wdl_tick{tick + 1}"][f"input_{component}"]
        == data[f"wdl_tick{tick}"][component]
        for tick in (1, 2)
        for component in ("state", "scratch")
    )
    checks["wdl_new_emission_delay_is_exactly_two_ticks"] = (
        wdl_sums[0] == 0 and wdl_sums[1] == 0 and wdl_sums[2] > 0 and wdl_chain_exact
    )

    middle_ref = data["middle_survivor_reference"]
    middle = data["middle_survivor_candidate"]
    middle_input_header = _field_values(
        middle, layout, "header", component="input_state"
    )
    middle_input_order = _field_values(
        middle, layout, "active_slots_by_uid", component="input_state"
    )[: middle_input_header[3]]
    middle_input_ids = _field_values(
        middle, layout, "agent_id", component="input_state"
    )
    middle_input_health = _field_values(
        middle, layout, "health", component="input_state"
    )
    middle_input_energy = _field_values(
        middle, layout, "energy", component="input_state"
    )
    middle_output_header = _field_values(middle, layout, "header")
    middle_output_order = _field_values(middle, layout, "active_slots_by_uid")[
        : middle_output_header[3]
    ]
    middle_output_ids = _field_values(middle, layout, "agent_id")
    middle_rows, middle_counts = _events(middle, layout)
    middle_births = [row for row in middle_rows if row["status"] and row["kind"] == 2]
    middle_deaths = [row for row in middle_rows if row["status"] and row["kind"] == 1]
    expected_survivor_ids = [
        middle_input_ids[middle_input_order[index]] for index in (0, 2, 3)
    ]
    expected_parent_ids = tuple(expected_survivor_ids[:2])
    checks.update(
        {
            "middle_survivor_reference_candidate_byte_parity": all(
                middle_ref[component] == middle[component]
                and middle_ref[f"input_{component}"] == middle[f"input_{component}"]
                for component in ("state", "scratch", "events", "counts")
            ),
            "birth_pairs_adjacent_compacted_survivor_uid_order": (
                len(middle_input_order) == 4
                and middle_input_health[middle_input_order[1]] == 0
                and middle_input_energy[middle_input_order[3]] < 256
                and len(middle_deaths) == 1
                and middle_deaths[0]["agent_id"]
                == middle_input_ids[middle_input_order[1]]
                and len(middle_births) == 1
                and (
                    middle_births[0]["parent_a_id"],
                    middle_births[0]["parent_b_id"],
                )
                == expected_parent_ids
                and middle_counts == (2,)
                and [middle_output_ids[slot] for slot in middle_output_order[:3]]
                == expected_survivor_ids
                and middle_output_ids[middle_output_order[-1]]
                == middle_births[0]["agent_id"]
            ),
        }
    )

    checkpoint = data["restore_checkpoint"]
    checkpoint_header = _field_values(checkpoint, layout, "header")
    active_count = checkpoint_header[3]
    order = _field_values(checkpoint, layout, "active_slots_by_uid")[:active_count]
    checkpoint_ids = _field_values(checkpoint, layout, "agent_id")
    ordered_ids = [checkpoint_ids[slot] for slot in order]
    checkpoint_rows, _ = _events(checkpoint, layout)
    checkpoint_birth_slots = [
        row["row_index"] - shape.capacity
        for row in checkpoint_rows
        if row["status"] and row["kind"] == 2
    ]
    invalid_before = _field_values(data["restore_invalid_before"], layout, "header")
    invalid_after = _field_values(data["restore_invalid_after"], layout, "header")
    checks.update(
        {
            "active_order_is_strict_biological_uid_order": bool(ordered_ids)
            and all(left < right for left, right in zip(ordered_ids, ordered_ids[1:])),
            "birth_reuses_low_slot_but_appends_uid_order": bool(checkpoint_birth_slots)
            and any(slot < shape.initial_population for slot in checkpoint_birth_slots)
            and tuple(order[-len(checkpoint_birth_slots) :])
            == tuple(checkpoint_birth_slots),
            "restore_reconstructs_after_arbitrary_derived_corruption": checkpoint[
                "state"
            ]
            == data["restore_reconstructed"]["state"]
            and data["restore_reconstructed"]["input_state"] != checkpoint["state"]
            and _without_state_field(
                data["restore_reconstructed"]["input_state"], layout, "occupancy"
            )
            == _without_state_field(checkpoint["state"], layout, "occupancy")
            and data["restore_reconstructed"]["input_scratch"] != checkpoint["scratch"],
            "restore_continuation_state_exact": data["restore_same_continuation"][
                "state"
            ]
            == data["restore_fresh_continuation"]["state"]
            and data["restore_same_continuation"]["input_state"]
            == data["restore_fresh_continuation"]["input_state"],
            "restore_continuation_scratch_exact": data["restore_same_continuation"][
                "scratch"
            ]
            == data["restore_fresh_continuation"]["scratch"]
            and data["restore_same_continuation"]["input_scratch"]
            == data["restore_fresh_continuation"]["input_scratch"],
            "restore_continuation_events_exact": data["restore_same_continuation"][
                "events"
            ]
            == data["restore_fresh_continuation"]["events"]
            and data["restore_same_continuation"]["counts"]
            == data["restore_fresh_continuation"]["counts"],
            "fresh_process_restore_continuation_exact": all(
                data["restore_same_continuation"][component]
                == data["restore_fresh_continuation"][component]
                for component in ("state", "scratch", "events", "counts")
            ),
            "invalid_restore_status_preserved_and_dispatch_fail_stops": invalid_before[
                4
            ]
            != 0
            and invalid_after[4] == invalid_before[4]
            and invalid_after[1] == invalid_before[1]
            and data["restore_invalid_after"]["input_state"]
            == data["restore_invalid_before"]["state"],
        }
    )

    overflow_ref, overflow_cand = data["overflow_reference"], data["overflow_candidate"]
    overflow_rows, overflow_counts = _events(overflow_cand, layout)
    overflow_births = [
        row for row in overflow_rows if row["status"] and row["kind"] == 2
    ]
    overflow_header = _field_values(overflow_cand, layout, "header")
    checks.update(
        {
            "overflow_reference_candidate_state_parity": overflow_ref["state"]
            == overflow_cand["state"]
            and overflow_ref["input_state"] == overflow_cand["input_state"],
            "overflow_reference_candidate_scratch_parity": overflow_ref["scratch"]
            == overflow_cand["scratch"]
            and overflow_ref["input_scratch"] == overflow_cand["input_scratch"],
            "overflow_reference_candidate_event_parity": overflow_ref["events"]
            == overflow_cand["events"]
            and overflow_ref["counts"] == overflow_cand["counts"]
            and overflow_ref["input_events"] == overflow_cand["input_events"]
            and overflow_ref["input_counts"] == overflow_cand["input_counts"],
            "overflow_truncates_to_single_free_slot": len(overflow_births) == 1,
            "overflow_is_not_status_error": overflow_header[4] == 0,
            "overflow_publishes_one_tick": overflow_header[1]
            == overflow_cand["start_next_absolute_tick"] + 1,
        }
    )
    transaction = data["transaction_after"]
    transaction_header = _field_values(transaction, layout, "header")
    transaction_input_header = struct.unpack_from("<32Q", transaction["input_state"])
    rows_per_tick_bytes = layout.event_bytes_per_world_tick
    second_tick = transaction["events"][rows_per_tick_bytes : 2 * rows_per_tick_bytes]
    checks.update(
        {
            "first_error_status_is_deterministic": transaction_input_header[4] == 0
            and transaction_header[4] == 1,
            "failed_dispatch_does_not_publish_frontier": transaction_header[1]
            == transaction["start_next_absolute_tick"]
            and transaction_input_header[1] == transaction["start_next_absolute_tick"],
            "remaining_temporal_ticks_fail_stop": second_tick
            == bytes(len(second_tick)),
            # The owner law forbids reuse whenever status is nonzero and the
            # frontier did not publish; this is derived from those persisted
            # state facts, not a caller assertion.
            "failed_event_buffer_cannot_be_reused": transaction_header[4] != 0
            and transaction_header[1] == transaction["start_next_absolute_tick"],
        }
    )
    if set(checks) != WORKHORSE_EVIDENCE_DERIVED_CHECKS_V2:
        raise WorkhorseSemanticEvidenceError(
            "semantic evidence derivation coverage changed"
        )
    witnesses = {
        "bundle_path": str(Path(manifest_path).resolve()),
        "bundle_file_sha256": file_sha256(manifest_path),
        "bundle_body_sha256": manifest["bundle_body_sha256"],
        "birth_slots": birth_slots,
        "death_rows": len(death_rows),
        "birth_rows": len(birth_rows),
        "wdl_active_sum_after_ticks_1_2_3": wdl_sums,
        "overflow_birth_rows": len(overflow_births),
        "overflow_event_count": overflow_counts[0],
    }
    return checks, witnesses


__all__ = [
    "WORKHORSE_EVIDENCE_DERIVED_CHECKS_V2",
    "WORKHORSE_SEMANTIC_CASE_AGGREGATE_SCHEMA_V2",
    "WORKHORSE_SEMANTIC_DATASETS_V2",
    "WORKHORSE_SEMANTIC_EVIDENCE_SCHEMA_V2",
    "WorkhorseSemanticEvidenceError",
    "aggregate_workhorse_semantic_case_receipts_v2",
    "derive_workhorse_semantic_checks_v2",
    "reopen_workhorse_semantic_case_aggregate_v2",
    "reopen_workhorse_semantic_evidence_bundle_v2",
    "workhorse_semantic_dataset_projection_v2",
    "write_workhorse_semantic_evidence_bundle_v2",
]
