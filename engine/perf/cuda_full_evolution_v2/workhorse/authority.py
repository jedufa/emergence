"""Reopenable evidence authorities for the CUDA workhorse V2.

This module intentionally treats paths and exact bytes as authority.  Caller
mappings and standalone digest strings are reporting inputs, never sufficient
to acknowledge events, reopen checkpoints, or grant a performance verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import zlib
from typing import Any, Final, Mapping, Sequence

try:
    from compression import zstd as _zstd
except ImportError:  # pragma: no cover - supported interpreter provides zstd
    _zstd = None

from .contract import (
    WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
    WORKHORSE_TIME_AUTHORITY_V2,
    WORKHORSE_TRANSITION_ID_V2,
    FullEvolutionCudaWorkhorseContractV2,
    FullEvolutionWorkhorseContractError,
    canonical_sha256,
    workhorse_artifact_identity_v2,
)
from .native_loader import file_sha256, validate_workhorse_prebuilt_v2
from .attempt_authority import validate_workhorse_attempt_authority_binding_v2
from .semantic_evidence import (
    derive_workhorse_semantic_checks_v2,
    reopen_workhorse_semantic_case_aggregate_v2,
    reopen_workhorse_semantic_evidence_bundle_v2,
)
from .semantic_cases import (
    WORKHORSE_SEMANTIC_CASE_TERMINAL_RECEIPT_SCHEMA_V2,
    reopen_terminalized_workhorse_semantic_case_receipt_v2,
    reopen_workhorse_semantic_case_fragment_v2,
    workhorse_semantic_case_table_v2,
    write_terminalized_workhorse_semantic_case_receipt_v2,
)

PROJECT_ROOT_V2: Final[Path] = Path(__file__).resolve().parents[4]
ARCHITECTURE_CONTRACT_RELATIVE_PATH_V2: Final[str] = (
    "_artifacts/tw-cuda-full-evolution-reform-20260825/architecture_contract.json"
)
WORKHORSE_SEMANTIC_CONTRACT_RELATIVE_PATH_V2: Final[str] = (
    "engine/perf/cuda_full_evolution_v2/workhorse/contract.py"
)
PERFORMANCE_CONTRACT_RELATIVE_PATH_V2: Final[str] = (
    "_artifacts/tw-cuda-full-evolution-reform-20260825/"
    "performance_restructure_contract.json"
)
VALIDATION_MATRIX_RELATIVE_PATH_V2: Final[str] = (
    "_artifacts/tw-cuda-full-evolution-reform-20260825/validation_matrix.json"
)
MICRO_BENCHMARK_RELATIVE_PATH_V2: Final[str] = (
    "engine/perf/cuda_full_evolution_v2/benchmark.py"
)
WORKHORSE_EVENT_DISPATCH_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_workhorse_event_dispatch_v2"
)
WORKHORSE_EVENT_SINK_RECEIPT_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_workhorse_event_sink_receipt_v2"
)
WORKHORSE_SPARSE_EVENT_DISPATCH_SCHEMA_V3: Final[str] = (
    "eiel_full_evolution_cuda_workhorse_sparse_event_dispatch_v3"
)
WORKHORSE_SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3: Final[str] = (
    "eiel_full_evolution_cuda_workhorse_sparse_event_sink_receipt_v3"
)
WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4: Final[str] = (
    "eiel_full_evolution_cuda_workhorse_sparse_event_consumer_receipt_v4"
)
WORKHORSE_TECHNICAL_DIGEST_CONSUMER_PROFILE_V1: Final[str] = (
    "TECHNICAL_DIGEST_CONSUMER_V1"
)
WORKHORSE_CHECKPOINT_BUNDLE_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_workhorse_checkpoint_bundle_v2"
)
WORKHORSE_SEMANTIC_GATE_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_workhorse_semantic_gate_v2"
)
WORKHORSE_FRESH_PROCESS_RAW_COMPONENTS_V2: Final[tuple[str, ...]] = (
    "input_state",
    "input_scratch",
    "input_events",
    "input_counts",
    "state",
    "scratch",
    "events",
    "counts",
)
WORKHORSE_REQUIRED_SEMANTIC_CHECKS_V2: Final[frozenset[str]] = frozenset(
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
        "checkpoint_binds_external_exact_content_sink_ack",
        "sealed_checkpoint_reopens_exact_state_bytes",
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
        "native_runtime_inspection_present",
    }
)


class FullEvolutionWorkhorseAuthorityError(RuntimeError):
    """Raised when evidence cannot be independently reopened and recomputed."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_dict(value: object, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise FullEvolutionWorkhorseAuthorityError(f"{label} must be an object")
    return value


def _historical_source_path_matches_relative_v2(
    value: object,
    relative_path: str,
) -> bool:
    """Accept an exact repository-relative source path from any old checkout."""

    if type(value) is not str:
        return False
    source_path = Path(value)
    relative_parts = Path(relative_path).parts
    return (
        source_path.is_absolute()
        and ".." not in source_path.parts
        and len(source_path.parts) >= len(relative_parts)
        and source_path.parts[-len(relative_parts) :] == relative_parts
    )


def _regular_file(path: str | Path, label: str, *, read_only: bool) -> Path:
    supplied = Path(path)
    if supplied.is_symlink() or not supplied.is_file():
        raise FullEvolutionWorkhorseAuthorityError(
            f"{label} must be a regular non-symlink"
        )
    resolved = supplied.resolve()
    if read_only and resolved.stat().st_mode & 0o222:
        raise FullEvolutionWorkhorseAuthorityError(f"{label} must be read-only")
    return resolved


def _read_json(
    path: str | Path, label: str, *, read_only: bool = True
) -> tuple[Path, dict[str, Any]]:
    resolved = _regular_file(path, label, read_only=read_only)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"cannot reopen {label}: {exc}"
        ) from exc
    return resolved, _require_exact_dict(value, label)


def workhorse_governing_contract_bindings_v2(
    project_root: Path = PROJECT_ROOT_V2,
) -> dict[str, dict[str, str]]:
    """Rehash the truthful workhorse semantic law and performance contract."""
    semantic_path = project_root / WORKHORSE_SEMANTIC_CONTRACT_RELATIVE_PATH_V2
    performance_path = project_root / PERFORMANCE_CONTRACT_RELATIVE_PATH_V2
    for label, path in (
        ("workhorse semantic contract", semantic_path),
        ("performance contract", performance_path),
    ):
        if path.is_symlink() or not path.is_file():
            raise FullEvolutionWorkhorseAuthorityError(
                f"frozen {label} is missing or a symlink"
            )
    return {
        "workhorse_semantic_contract": {
            "path": WORKHORSE_SEMANTIC_CONTRACT_RELATIVE_PATH_V2,
            "source_sha256": file_sha256(semantic_path),
            "semantic_transition_id": WORKHORSE_TRANSITION_ID_V2,
            "semantic_transition_sha256": (WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2),
        },
        "performance_contract": {
            "path": PERFORMANCE_CONTRACT_RELATIVE_PATH_V2,
            "sha256": file_sha256(performance_path),
        },
    }


def _micro_governing_contract_bindings_v2(
    project_root: Path = PROJECT_ROOT_V2,
) -> dict[str, dict[str, str]]:
    """Return only contracts which govern the independently built micro route."""
    bindings: dict[str, dict[str, str]] = {}
    for label, relative in (
        ("architecture_contract", ARCHITECTURE_CONTRACT_RELATIVE_PATH_V2),
        ("performance_contract", PERFORMANCE_CONTRACT_RELATIVE_PATH_V2),
    ):
        path = project_root / relative
        if path.is_symlink() or not path.is_file():
            raise FullEvolutionWorkhorseAuthorityError(
                f"frozen micro {label} is missing or a symlink"
            )
        bindings[label] = {"path": relative, "sha256": file_sha256(path)}
    return bindings


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values or not 0.0 <= quantile <= 1.0:
        raise FullEvolutionWorkhorseAuthorityError("percentile input is invalid")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    )


_HISTORICAL_DIRECT_SPARSE_AUTOTUNE_SCHEMA_V3: Final[str] = (
    "eiel_cuda_full_evolution_direct_sparse_autotune_v3"
)
_HISTORICAL_DIRECT_SPARSE_RESULT_SCHEMA_V2: Final[str] = (
    "eiel_cuda_full_evolution_direct_sparse_sustained_benchmark_v2"
)
_HISTORICAL_DIRECT_SPARSE_ENVELOPE_SCHEMA_V3: Final[str] = (
    "eiel_cuda_full_evolution_direct_sparse_durable_envelope_v3"
)
_HISTORICAL_DIRECT_SPARSE_DISPATCH_SCHEMA_V3: Final[str] = (
    "eiel_full_evolution_cuda_sparse_event_dispatch_v3"
)
_HISTORICAL_DIRECT_SPARSE_SINK_SCHEMA_V3: Final[str] = (
    "eiel_cuda_full_evolution_direct_sparse_durable_sink_receipt_v3"
)
_HISTORICAL_DIRECT_SPARSE_EXECUTION_SCHEMA_V3: Final[str] = (
    "eiel_full_evolution_cuda_direct_sparse_execution_receipt_v3"
)
_HISTORICAL_DIRECT_SPARSE_OWNER_SCHEMA_V3: Final[str] = (
    "eiel_full_evolution_cuda_direct_sparse_owner_v3"
)
_HISTORICAL_DIRECT_SPARSE_WIRE_V3: Final[str] = (
    "eiel_full_evolution_cuda_sparse_lifecycle_event_v3"
)
_HISTORICAL_DIRECT_SPARSE_FIXTURE_V2: Final[str] = (
    "eiel_full_evolution_cuda_full_capacity_benchmark_fixture_v2"
)
_HISTORICAL_DIRECT_SPARSE_ENVELOPE_MAGIC_V3: Final[bytes] = b"EIELDSV3"
_HISTORICAL_DIRECT_SPARSE_ROW_BYTES_V3: Final[int] = 32
_HISTORICAL_FULL_EVOLUTION_CAPABILITIES_V2: Final[tuple[str, ...]] = (
    "patch",
    "wdl_delayed_hazard",
    "jungle",
    "eed",
    "genome_v2_ordinary_birth",
    "genome_v2_mutation",
    "recombination_profile",
)


def _read_sealed_regular_bytes_v2(
    path: str | Path, label: str
) -> tuple[Path, bytes, os.stat_result]:
    """Read one immutable inode without following or accepting a replacement."""
    supplied = Path(path).expanduser()
    descriptor: int | None = None
    try:
        before_path = supplied.lstat()
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(supplied, flags)
        before = os.fstat(descriptor)
        if (
            supplied.is_symlink()
            or not stat.S_ISREG(before_path.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (before_path.st_dev, before_path.st_ino)
            != (before.st_dev, before.st_ino)
            or before.st_nlink != 1
            or before_path.st_nlink != 1
            or before.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            or before_path.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                f"{label} must be one read-only single-link regular file"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        after_path = supplied.lstat()
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mode, after.st_nlink)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mode, 1)
            or (
                after_path.st_dev,
                after_path.st_ino,
                after_path.st_size,
                after_path.st_mode,
                after_path.st_nlink,
            )
            != (before.st_dev, before.st_ino, before.st_size, before.st_mode, 1)
            or len(payload) != before.st_size
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                f"{label} changed while it was reopened"
            )
        return supplied.resolve(strict=True), payload, before
    except FullEvolutionWorkhorseAuthorityError:
        raise
    except OSError as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"cannot reopen {label}: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_sealed_json_object_v2(
    path: str | Path, label: str
) -> tuple[Path, dict[str, Any], bytes, os.stat_result]:
    resolved, payload, metadata = _read_sealed_regular_bytes_v2(path, label)
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"cannot decode {label}: {exc}"
        ) from exc
    return resolved, _require_exact_dict(value, label), payload, metadata


def _historical_source_identities_v2(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = manifest.get("source_inventory")
    if type(inventory) is not list or not inventory:
        raise FullEvolutionWorkhorseAuthorityError(
            "historical manifest source inventory is missing"
        )
    historical_rows: list[dict[str, str]] = []
    current_rows: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    for raw_row in inventory:
        row = _require_exact_dict(raw_row, "historical source inventory row")
        relative_text = row.get("path")
        digest = row.get("sha256")
        if type(relative_text) is not str or not _valid_sha256(digest):
            raise FullEvolutionWorkhorseAuthorityError(
                "historical source inventory row is malformed"
            )
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative_text in observed_paths
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "historical source inventory path is unsafe or duplicated"
            )
        observed_paths.add(relative_text)
        historical_rows.append({"path": relative_text, "sha256": digest})
        current = PROJECT_ROOT_V2 / relative
        try:
            current_metadata = current.lstat()
            current_regular = (
                stat.S_ISREG(current_metadata.st_mode) and not current.is_symlink()
            )
            current_sha256 = file_sha256(current) if current_regular else None
        except OSError:
            current_regular = False
            current_sha256 = None
        current_rows.append(
            {
                "path": relative_text,
                "regular_non_symlink": current_regular,
                "sha256": current_sha256,
                "matches_historical": current_sha256 == digest,
            }
        )
    inventory_sha256 = canonical_sha256(historical_rows)
    if (
        manifest.get("source_inventory_sha256") != inventory_sha256
        or manifest.get("post_build_live_source_inventory_sha256") != inventory_sha256
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical manifest source inventory seal is invalid"
        )
    historical = {
        "schema_version": "eiel_historical_source_identity_v2",
        "identity_role": "immutable_benchmark_build_identity",
        "source_inventory": historical_rows,
        "source_inventory_sha256": inventory_sha256,
    }
    current = {
        "schema_version": "eiel_current_source_observation_v2",
        "identity_role": "informational_current_checkout_identity_no_execution_authority",
        "source_inventory": current_rows,
        "source_inventory_sha256": canonical_sha256(current_rows),
        "matches_historical_identity": all(
            row["matches_historical"] for row in current_rows
        ),
    }
    return historical, current


def _validate_historical_direct_sparse_envelope_v2(
    row: Mapping[str, Any],
    *,
    batch_size: int,
    temporal_k: int,
    result_directory: Path,
    artifact_sha256: str,
    manifest_sha256: str,
) -> None:
    envelope = _require_exact_dict(row.get("durable_envelope"), "durable envelope")
    runtime = _require_exact_dict(
        row.get("runtime_execution_receipt"), "direct sparse execution receipt"
    )
    durable = _require_exact_dict(
        runtime.get("durable_envelope"), "runtime durable envelope"
    )
    envelope_path, payload, metadata = _read_sealed_regular_bytes_v2(
        envelope.get("path"), "direct sparse durable envelope"
    )
    if envelope_path.parent != result_directory:
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse durable envelope is outside its sealed result root"
        )
    content_sha256 = _sha_bytes(payload)
    if (
        envelope.get("path") != str(envelope_path)
        or envelope.get("bytes") != metadata.st_size
        or envelope.get("device") != metadata.st_dev
        or envelope.get("inode") != metadata.st_ino
        or envelope.get("link_count") != 1
        or envelope.get("mode") != oct(stat.S_IMODE(metadata.st_mode))
        or envelope.get("sha256") != content_sha256
        or durable.get("durable_path") != str(envelope_path)
        or durable.get("durable_device") != metadata.st_dev
        or durable.get("durable_inode") != metadata.st_ino
        or durable.get("durable_link_count") != 1
        or durable.get("durable_mode") != stat.S_IMODE(metadata.st_mode)
        or durable.get("durable_nbytes") != metadata.st_size
        or durable.get("durable_content_sha256") != content_sha256
        or durable.get("durable_mtime_ns") != metadata.st_mtime_ns
        or durable.get("durable_ctime_ns") != metadata.st_ctime_ns
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse durable envelope identity changed"
        )
    if payload[:8] != _HISTORICAL_DIRECT_SPARSE_ENVELOPE_MAGIC_V3 or len(payload) < 16:
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse durable envelope header is invalid"
        )
    header_nbytes = int.from_bytes(payload[8:16], byteorder="little", signed=False)
    header_end = 16 + header_nbytes
    if header_end > len(payload):
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse durable envelope header is truncated"
        )
    try:
        header = _require_exact_dict(
            json.loads(payload[16:header_end].decode("ascii")),
            "direct sparse durable envelope header",
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"direct sparse durable envelope header cannot be decoded: {exc}"
        ) from exc
    if payload[16:header_end] != _json_bytes(header):
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse durable envelope header is not canonical"
        )
    offset_count = header.get("world_offset_count")
    if type(offset_count) is not int or offset_count != batch_size + 1:
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse durable envelope has another batch shape"
        )
    offsets_end = header_end + 8 * offset_count
    if offsets_end > len(payload):
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse durable envelope offsets are truncated"
        )
    offsets = tuple(
        int.from_bytes(
            payload[header_end + 8 * index : header_end + 8 * (index + 1)],
            byteorder="little",
            signed=False,
        )
        for index in range(offset_count)
    )
    packed_events = payload[offsets_end:]
    counts = header.get("event_counts")
    final_ids = header.get("final_next_agent_ids")
    world_uids = header.get("world_uids")
    packed_count = header.get("packed_event_count")
    if (
        type(counts) is not list
        or type(final_ids) is not list
        or type(world_uids) is not list
        or len(counts) != batch_size
        or len(final_ids) != batch_size
        or len(world_uids) != batch_size
        or any(type(value) is not int or value < 0 for value in counts)
        or offsets[0] != 0
        or tuple(offsets[index + 1] - offsets[index] for index in range(batch_size))
        != tuple(counts)
        or type(packed_count) is not int
        or packed_count != offsets[-1]
        or header.get("packed_event_nbytes") != len(packed_events)
        or len(packed_events) != packed_count * _HISTORICAL_DIRECT_SPARSE_ROW_BYTES_V3
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse durable envelope payload shape is invalid"
        )
    sample_id = row.get("sample_id")
    if (
        header.get("schema_version") != _HISTORICAL_DIRECT_SPARSE_ENVELOPE_SCHEMA_V3
        or header.get("claim_class") != "TECHNICAL_ONLY"
        or header.get("sample_id") != sample_id
        or header.get("dispatch_schema_version")
        != _HISTORICAL_DIRECT_SPARSE_DISPATCH_SCHEMA_V3
        or header.get("dispatch_time_authority") != WORKHORSE_TIME_AUTHORITY_V2
        or header.get("wire_abi") != _HISTORICAL_DIRECT_SPARSE_WIRE_V3
        or header.get("start_next_absolute_tick") != 0
        or header.get("end_next_absolute_tick") != temporal_k
        or header.get("tick_count") != temporal_k
        or header.get("artifact_sha256") != artifact_sha256
        or header.get("manifest_sha256") != manifest_sha256
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse durable envelope authority is inconsistent"
        )
    dispatch_metadata = {
        "schema_version": _HISTORICAL_DIRECT_SPARSE_DISPATCH_SCHEMA_V3,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "wire_abi": _HISTORICAL_DIRECT_SPARSE_WIRE_V3,
        "wire_endianness": "little",
        "packed_event_row_bytes": _HISTORICAL_DIRECT_SPARSE_ROW_BYTES_V3,
        "packed_event_count": packed_count,
        "packed_event_nbytes": len(packed_events),
        "world_offsets": list(offsets),
        "event_counts": counts,
        "final_next_agent_ids": final_ids,
        "world_uids": world_uids,
        "start_next_absolute_tick": 0,
        "end_next_absolute_tick": temporal_k,
        "tick_count": temporal_k,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
    }
    dispatch_digest = hashlib.sha256()
    dispatch_digest.update(b"EIEL-CUDA-FULL-EVOLUTION-V3-SPARSE-EVENTS\0")
    dispatch_digest.update(_json_bytes(dispatch_metadata))
    dispatch_digest.update(b"\0PACKED-ROWS\0")
    dispatch_digest.update(packed_events)
    dispatch_sha256 = dispatch_digest.hexdigest()
    if (
        header.get("dispatch_content_sha256") != dispatch_sha256
        or row.get("event_content_sha256") != dispatch_sha256
        or row.get("event_count") != packed_count
        or row.get("sparse_wire_bytes") != len(packed_events) + 8 * len(offsets)
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse dispatch seal does not recompute"
        )
    sink_body = {
        "schema_version": _HISTORICAL_DIRECT_SPARSE_SINK_SCHEMA_V3,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "wire_abi": _HISTORICAL_DIRECT_SPARSE_WIRE_V3,
        "envelope_schema_version": _HISTORICAL_DIRECT_SPARSE_ENVELOPE_SCHEMA_V3,
        "dispatch_content_sha256": dispatch_sha256,
        "sink_id": "immutable-local-direct-sparse-envelope-v3",
        "sink_receipt_id": f"{envelope_path.name}:{content_sha256}",
        "sample_id": sample_id,
        "world_uids": world_uids,
        "start_next_absolute_tick": 0,
        "end_next_absolute_tick": temporal_k,
        "tick_count": temporal_k,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        **durable,
    }
    sink_sha256 = canonical_sha256(sink_body)
    if (
        row.get("sink_receipt_sha256") != sink_sha256
        or runtime.get("sink_receipt_sha256") != sink_sha256
        or runtime.get("schema_version")
        != _HISTORICAL_DIRECT_SPARSE_EXECUTION_SCHEMA_V3
        or runtime.get("owner_schema_version")
        != _HISTORICAL_DIRECT_SPARSE_OWNER_SCHEMA_V3
        or runtime.get("profile_id") != "eiel_full_evolution_cuda_native_v2"
        or runtime.get("event_wire_abi") != _HISTORICAL_DIRECT_SPARSE_WIRE_V3
        or runtime.get("event_content_sha256") != dispatch_sha256
        or runtime.get("artifact_sha256") != artifact_sha256
        or runtime.get("manifest_sha256") != manifest_sha256
        or runtime.get("B") != batch_size
        or runtime.get("start_next_absolute_tick") != 0
        or runtime.get("end_next_absolute_tick") != temporal_k
        or runtime.get("committed_next_absolute_tick") != temporal_k
        or runtime.get("tick_count") != temporal_k
        or runtime.get("committed_world_ticks") != batch_size * temporal_k
        or runtime.get("runtime_status") != "committed"
        or runtime.get("event_status")
        != "exact_direct_sparse_content_sink_acknowledged"
        or runtime.get("selected_capacity_rows_per_world") != 512
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "direct sparse nested sink/execution seal is invalid"
        )


def _validate_historical_direct_sparse_result_v2(
    result_path: str | Path,
    *,
    summary: Mapping[str, Any],
    planned_candidate: Mapping[str, Any],
    expected_device_identity: Mapping[str, Any],
    artifact_path: Path,
    artifact_metadata: os.stat_result,
    artifact_sha256: str,
    manifest_path: Path,
    manifest_metadata: os.stat_result,
    manifest_file_sha256: str,
    manifest_sha256: str,
    historical_source_sha256_by_path: Mapping[str, str],
    probe_samples: int,
    probe_warmups: int,
) -> dict[str, Any]:
    resolved, value, _, _ = _read_sealed_json_object_v2(
        result_path, "historical direct sparse result"
    )
    body = {key: item for key, item in value.items() if key != "result_body_sha256"}
    candidate = _require_exact_dict(value.get("candidate"), "direct sparse candidate")
    batch_size = candidate.get("batch_size")
    temporal_k = candidate.get("temporal_k")
    if (
        value.get("schema_version") != _HISTORICAL_DIRECT_SPARSE_RESULT_SCHEMA_V2
        or value.get("claim_class") != "TECHNICAL_ONLY"
        or value.get("result_body_sha256") != canonical_sha256(body)
        or summary.get("result_path") != str(resolved)
        or summary.get("result_body_sha256") != value.get("result_body_sha256")
        or type(batch_size) is not int
        or batch_size <= 0
        or type(temporal_k) is not int
        or temporal_k <= 0
        or candidate
        != {
            "batch_size": batch_size,
            "fixture_id": _HISTORICAL_DIRECT_SPARSE_FIXTURE_V2,
            "initial_capacity_rows_per_world": 512,
            "initial_population": 256,
            "resident_capacity": 512,
            "temporal_k": temporal_k,
            "world_width": 32,
        }
        or planned_candidate.get("batch_size") != batch_size
        or planned_candidate.get("temporal_k") != temporal_k
        or planned_candidate.get("initial_capacity_rows_per_world") != 512
        or value.get("samples") != probe_samples
        or value.get("warmups") != probe_warmups
        or value.get("device_identity") != dict(expected_device_identity)
        or value.get("absolute_tick_authority") is not True
        or value.get("horizon_dependent_resident_bytes") != 0
        or value.get("all_seven_capabilities_positive_each_sample") is not True
        or value.get("full_evolution_capabilities")
        != list(_HISTORICAL_FULL_EVOLUTION_CAPABILITIES_V2)
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical direct sparse result seal or W32/P256/C512 shape is invalid"
        )
    event_transport = _require_exact_dict(
        value.get("event_transport"), "direct sparse event transport"
    )
    if (
        event_transport.get("production_wire") != _HISTORICAL_DIRECT_SPARSE_WIRE_V3
        or event_transport.get("production_emission")
        != "direct_per_world_bounded_staging"
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical result is not direct-sparse-v3 production transport"
        )
    artifact = _require_exact_dict(value.get("artifact"), "historical artifact")
    manifest_binding = _require_exact_dict(
        artifact.get("manifest_binding"), "historical manifest binding"
    )
    if (
        artifact.get("path") != str(artifact_path)
        or artifact.get("bytes") != artifact_metadata.st_size
        or artifact.get("device") != artifact_metadata.st_dev
        or artifact.get("inode") != artifact_metadata.st_ino
        or artifact.get("link_count") != 1
        or artifact.get("mode") != oct(stat.S_IMODE(artifact_metadata.st_mode))
        or artifact.get("sha256") != artifact_sha256
        or artifact.get("manifest_path") != str(manifest_path)
        or artifact.get("manifest_sha256") != manifest_sha256
        or artifact.get("manifest_file_sha256") != manifest_file_sha256
        or manifest_binding.get("bytes") != manifest_metadata.st_size
        or manifest_binding.get("device") != manifest_metadata.st_dev
        or manifest_binding.get("inode") != manifest_metadata.st_ino
        or manifest_binding.get("link_count") != 1
        or manifest_binding.get("mode") != oct(stat.S_IMODE(manifest_metadata.st_mode))
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical nested artifact/manifest identity changed"
        )
    benchmark_source = _require_exact_dict(
        value.get("benchmark_source"), "historical benchmark source"
    )
    benchmark_relative = benchmark_source.get("relative_path")
    if (
        type(benchmark_relative) is not str
        or benchmark_source.get("binding")
        != "manifest_row_plus_open_descriptor_start_end_equality"
        or benchmark_source.get("sha256")
        != historical_source_sha256_by_path.get(benchmark_relative)
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical benchmark source is not manifest-bound"
        )
    fixture = _require_exact_dict(
        value.get("fixture_checkpoint"), "historical fixture checkpoint"
    )
    if (
        fixture.get("population_fixture_id") != _HISTORICAL_DIRECT_SPARSE_FIXTURE_V2
        or fixture.get("artifact_sha256") != artifact_sha256
        or fixture.get("manifest_sha256") != manifest_sha256
        or not _valid_sha256(fixture.get("canonical_state_sha256"))
        or not _valid_sha256(fixture.get("metadata_seal_sha256"))
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical fixture checkpoint identity is invalid"
        )
    measurement_rows = value.get("measurement_rows")
    if type(measurement_rows) is not list or len(measurement_rows) != (
        probe_samples + probe_warmups
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical direct sparse sample set is incomplete"
        )
    rates: list[float] = []
    expected_ids = [f"warmup-{index:03d}" for index in range(probe_warmups)] + [
        f"sample-{index:03d}" for index in range(probe_samples)
    ]
    for index, raw_row in enumerate(measurement_rows):
        row = _require_exact_dict(raw_row, "historical direct sparse sample")
        measured = index >= probe_warmups
        elapsed = row.get("wall_elapsed_seconds")
        throughput = row.get("committed_world_ticks_per_second")
        committed = batch_size * temporal_k
        counters = _require_exact_dict(
            row.get("capability_counters"), "direct sparse capability counters"
        )
        if (
            row.get("sample_id") != expected_ids[index]
            or row.get("measured") is not measured
            or row.get("committed_world_ticks") != committed
            or type(elapsed) not in (int, float)
            or not math.isfinite(float(elapsed))
            or elapsed <= 0
            or type(throughput) not in (int, float)
            or not math.isfinite(float(throughput))
            or throughput != committed / float(elapsed)
            or row.get("artifact_sha256") != artifact_sha256
            or row.get("manifest_sha256") != manifest_sha256
            or row.get("selected_capacity_rows_per_world") != 512
            or set(counters) != set(_HISTORICAL_FULL_EVOLUTION_CAPABILITIES_V2)
            or any(type(count) is not int or count <= 0 for count in counters.values())
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "historical direct sparse timing row is invalid"
            )
        _validate_historical_direct_sparse_envelope_v2(
            row,
            batch_size=batch_size,
            temporal_k=temporal_k,
            result_directory=resolved.parent,
            artifact_sha256=artifact_sha256,
            manifest_sha256=manifest_sha256,
        )
        if measured:
            rates.append(float(throughput))
    ordered = sorted(rates)
    p10 = ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)]
    median = (
        ordered[len(ordered) // 2]
        if len(ordered) % 2
        else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    )
    target = value.get("target_world_ticks_per_second")
    expected_status = (
        "FULL_TARGET_PASS"
        if type(target) in (int, float) and min(ordered) >= float(target)
        else "FULL_TARGET_NOT_DEMONSTRATED"
    )
    if (
        value.get("minimum_committed_world_ticks_per_second") != min(ordered)
        or value.get("p10_committed_world_ticks_per_second") != p10
        or value.get("median_committed_world_ticks_per_second") != median
        or value.get("maximum_committed_world_ticks_per_second") != max(ordered)
        or value.get("status") != expected_status
        or summary.get("status") != expected_status
        or summary.get("minimum_committed_world_ticks_per_second") != min(ordered)
        or summary.get("p10_committed_world_ticks_per_second") != p10
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical direct sparse statistics do not recompute"
        )
    return {
        "summary": dict(summary),
        "result_path": str(resolved),
        "result_body_sha256": value["result_body_sha256"],
        "batch_size": batch_size,
        "temporal_k": temporal_k,
        "p10_world_ticks_per_second": p10,
        "minimum_world_ticks_per_second": min(ordered),
        "status": expected_status,
    }


def _validate_historical_direct_sparse_micro_denominator_v2(
    receipt_path: str | Path,
    *,
    expected_device_identity: Mapping[str, Any],
) -> dict[str, Any]:
    receipt, value, receipt_payload, _ = _read_sealed_json_object_v2(
        receipt_path, "historical direct sparse autotune receipt"
    )
    body = {key: item for key, item in value.items() if key != "result_sha256"}
    plan = _require_exact_dict(value.get("plan"), "historical autotune plan")
    plan_body = {key: item for key, item in plan.items() if key != "plan_sha256"}
    if (
        value.get("schema_version") != _HISTORICAL_DIRECT_SPARSE_AUTOTUNE_SCHEMA_V3
        or value.get("claim_class") != "TECHNICAL_ONLY"
        or value.get("selection_statistic")
        != "p10_then_minimum_committed_world_ticks_per_second"
        or value.get("result_sha256") != canonical_sha256(body)
        or plan.get("schema_version")
        != "eiel_cuda_full_evolution_direct_sparse_autotune_plan_v3"
        or plan.get("claim_class") != "TECHNICAL_ONLY"
        or plan.get("plan_sha256") != canonical_sha256(plan_body)
        or plan.get("scaling_law", {}).get("horizon_dependent_resident_bytes") != 0
        or value.get("device_identity") != dict(expected_device_identity)
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical direct sparse autotune receipt or plan seal is invalid"
        )
    artifact_row = _require_exact_dict(
        value.get("artifact"), "historical autotune artifact"
    )
    artifact_path, artifact_payload, artifact_metadata = _read_sealed_regular_bytes_v2(
        artifact_row.get("path"), "historical direct sparse artifact"
    )
    artifact_sha256 = _sha_bytes(artifact_payload)
    manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
    (
        reopened_manifest_path,
        manifest,
        manifest_payload,
        manifest_metadata,
    ) = _read_sealed_json_object_v2(
        manifest_path, "historical direct sparse build manifest"
    )
    manifest_body = {
        key: item for key, item in manifest.items() if key != "manifest_sha256"
    }
    manifest_sha256 = manifest.get("manifest_sha256")
    output = _require_exact_dict(manifest.get("output"), "historical manifest output")
    if (
        artifact_row.get("path") != str(artifact_path)
        or artifact_row.get("sha256") != artifact_sha256
        or manifest.get("schema_version")
        != "eiel_full_evolution_cuda_native_build_manifest_v2"
        or manifest_sha256 != canonical_sha256(manifest_body)
        or manifest.get("immutable_output") is not True
        or manifest.get("compiled_from_sealed_snapshot") is not True
        or manifest.get("sealed_snapshot")
        != {
            "files_mode": "0444",
            "directories_mode": "0555",
            "post_compile_inventory_verified": True,
        }
        or manifest.get("profile") != "EIEL_FULL_EVOLUTION_CUDA_NATIVE_V2"
        or manifest.get("route_id") != "tensor_world_cuda_full_evolution_v2"
        or output.get("file_name") != artifact_path.name
        or output.get("bytes") != artifact_metadata.st_size
        or output.get("sha256") != artifact_sha256
        or manifest.get("layout_manifest_sha256")
        != canonical_sha256(manifest.get("layout_manifest"))
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical direct sparse artifact/manifest seal is invalid"
        )
    layout = _require_exact_dict(
        manifest.get("layout_manifest"), "historical direct sparse layout"
    )
    fields = layout.get("fields")
    if (
        type(fields) is not list
        or next(
            (row.get("shape") for row in fields if row.get("name") == "occupancy"),
            None,
        )
        != [1024]
        or next(
            (row.get("shape") for row in fields if row.get("name") == "agent_id"),
            None,
        )
        != [512]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical direct sparse artifact is not exact W32/C512"
        )
    historical_source, current_source = _historical_source_identities_v2(manifest)
    historical_source_sha256_by_path = {
        row["path"]: row["sha256"] for row in historical_source["source_inventory"]
    }
    autotune_source = _require_exact_dict(
        value.get("autotune_source"), "historical autotune source"
    )
    autotune_relative = "engine/perf/cuda_full_evolution_v2/direct_sparse_autotune.py"
    # This is immutable historical benchmark evidence, not authority for the
    # current checkout.  Its sealed receipt legitimately names the checkout in
    # which the benchmark ran.  Bind the exact repository-relative source path
    # and manifest hash without requiring that historical checkout to equal the
    # checkout performing the audit.
    if (
        not _historical_source_path_matches_relative_v2(
            autotune_source.get("path"), autotune_relative
        )
        or autotune_source.get("sha256")
        != historical_source_sha256_by_path.get(autotune_relative)
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical autotune source is not manifest-bound"
        )
    planned = plan.get("candidates")
    summaries = value.get("measurements")
    probe_samples = value.get("probe_samples")
    probe_warmups = value.get("probe_warmups")
    if (
        type(planned) is not list
        or type(summaries) is not list
        or not planned
        or len(planned) != len(summaries)
        or type(probe_samples) is not int
        or probe_samples < 1
        or type(probe_warmups) is not int
        or probe_warmups < 1
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical autotune measurement set is incomplete"
        )
    validated_rows = [
        _validate_historical_direct_sparse_result_v2(
            _require_exact_dict(summary, "historical autotune summary").get(
                "result_path"
            ),
            summary=summary,
            planned_candidate=_require_exact_dict(
                planned[index], "historical planned candidate"
            ),
            expected_device_identity=expected_device_identity,
            artifact_path=artifact_path,
            artifact_metadata=artifact_metadata,
            artifact_sha256=artifact_sha256,
            manifest_path=reopened_manifest_path,
            manifest_metadata=manifest_metadata,
            manifest_file_sha256=_sha_bytes(manifest_payload),
            manifest_sha256=str(manifest_sha256),
            historical_source_sha256_by_path=historical_source_sha256_by_path,
            probe_samples=probe_samples,
            probe_warmups=probe_warmups,
        )
        for index, summary in enumerate(summaries)
    ]
    selected = max(
        validated_rows,
        key=lambda row: (
            row["p10_world_ticks_per_second"],
            row["minimum_world_ticks_per_second"],
        ),
    )
    if value.get("selected") != selected["summary"]:
        raise FullEvolutionWorkhorseAuthorityError(
            "historical direct sparse autotune selection does not recompute"
        )
    current_contracts = _micro_governing_contract_bindings_v2()
    historical_contracts = manifest.get("normative_execution_contracts")
    if type(historical_contracts) is not list or any(
        type(row) is not dict
        or row.get("sha256")
        != historical_source_sha256_by_path.get(str(row.get("path")))
        for row in historical_contracts
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical governing contract identity is invalid"
        )
    historical_contract_bindings = {
        label: {
            "path": binding["path"],
            "sha256": historical_source_sha256_by_path[binding["path"]],
        }
        for label, binding in current_contracts.items()
        if binding["path"] in historical_source_sha256_by_path
    }
    if set(historical_contract_bindings) != set(current_contracts):
        raise FullEvolutionWorkhorseAuthorityError(
            "historical micro governing contract set is incomplete"
        )
    return {
        "schema_version": "eiel_workhorse_micro_denominator_binding_v2",
        "claim_class": "TECHNICAL_ONLY",
        "evidence_role": "historical_immutable_micro_denominator_only",
        "execution_authority_granted": False,
        "qualification_authority_granted": False,
        "science_authority_granted": False,
        "current_execution_compatibility_granted": False,
        "receipt_path": str(receipt),
        "receipt_file_sha256": _sha_bytes(receipt_payload),
        "receipt_content_sha256": value["result_sha256"],
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "manifest_path": str(reopened_manifest_path),
        "manifest_file_sha256": _sha_bytes(manifest_payload),
        "manifest_sha256": manifest_sha256,
        "source_inventory_sha256": manifest["source_inventory_sha256"],
        "historical_benchmark_identity": {
            "schema_version": "eiel_historical_direct_sparse_benchmark_identity_v2",
            "autotune_result_sha256": value["result_sha256"],
            "selected_result_body_sha256": selected["result_body_sha256"],
            "artifact_sha256": artifact_sha256,
            "manifest_sha256": manifest_sha256,
            "source_identity": historical_source,
            "governing_contracts": historical_contracts,
        },
        "current_source_identity": {
            **current_source,
            "governing_contracts": current_contracts,
        },
        "historical_source_matches_current": current_source[
            "matches_historical_identity"
        ],
        "device_identity": dict(expected_device_identity),
        "sites": 1024,
        "initial_population": 256,
        "capacity": 512,
        "p10_world_ticks_per_second": selected["p10_world_ticks_per_second"],
        "selected_execution_mode": (
            "historical_direct_sparse_v3_full_commit_wall_clock"
        ),
        "selected_measurement_sha256": selected["result_body_sha256"],
        "denominator_criterion": (
            "historical immutable direct-sparse-v3 full-commit wall-clock p10"
        ),
        "micro_governing_contracts": historical_contract_bindings,
        "micro_governing_contract_role": "historical_benchmark_identity_only",
    }


def validate_authoritative_micro_denominator_v2(
    receipt_path: str | Path,
    *,
    expected_device_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen a current or historical micro receipt and derive its p10.

    Current receipts remain live-source/device-bound.  The historical r11
    direct-sparse-v3 form is accepted only as an immutable ``TECHNICAL_ONLY``
    denominator with explicit no-authority flags and separate historical/current
    source identities.
    """
    supplied = Path(receipt_path).expanduser()
    try:
        initial_value = json.loads(supplied.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"cannot inspect micro benchmark receipt schema: {exc}"
        ) from exc
    if (
        type(initial_value) is dict
        and initial_value.get("schema_version")
        == _HISTORICAL_DIRECT_SPARSE_AUTOTUNE_SCHEMA_V3
    ):
        return _validate_historical_direct_sparse_micro_denominator_v2(
            receipt_path,
            expected_device_identity=expected_device_identity,
        )
    # The owning micro writer does not currently chmod its final projection.
    # Reopen and rehash the exact file now, then bind that file hash into the
    # immutable workhorse receipt.  All nested authority still comes from the
    # sealed artifact, source inventory, replay, and CUDA-event samples.
    receipt, value = _read_json(
        receipt_path, "micro benchmark receipt", read_only=False
    )
    body = {key: item for key, item in value.items() if key != "result_sha256"}
    if (
        value.get("schema_version")
        != "eiel_cuda_event_autotune_authoritative_result_v2"
        or value.get("authoritative") is not True
        or value.get("status") != "FULL_TARGET_PASS"
        or value.get("authority_class")
        != "AUTHORITATIVE_FRESH_LIVE_REPLAY_WITH_NON_AUTHORITATIVE_ORIGINAL_SWEEP"
        or value.get("result_sha256") != canonical_sha256(body)
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "micro receipt is not the sealed authoritative result"
        )
    artifact_row = _require_exact_dict(value.get("artifact"), "micro artifact row")
    artifact_path = artifact_row.get("path")
    if type(artifact_path) is not str:
        raise FullEvolutionWorkhorseAuthorityError("micro artifact path is missing")
    try:
        from ..native_loader import validate_prebuilt_cuda_full_evolution_v2

        artifact, _, manifest = validate_prebuilt_cuda_full_evolution_v2(artifact_path)
    except Exception as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"cannot reopen source-bound micro artifact: {exc}"
        ) from exc
    if (
        artifact_row.get("path") != str(artifact)
        or artifact_row.get("sha256") != file_sha256(artifact)
        or artifact_row.get("sha256") != manifest.get("output", {}).get("sha256")
        or artifact_row.get("manifest_sha256") != manifest.get("manifest_sha256")
        or manifest.get("route_id") != "tensor_world_cuda_full_evolution_v2"
        or manifest.get("profile") != "EIEL_FULL_EVOLUTION_CUDA_NATIVE_V2"
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "micro artifact/manifest identity is inconsistent"
        )
    layout = _require_exact_dict(manifest.get("layout_manifest"), "micro layout")
    fields = layout.get("fields")
    if (
        type(fields) is not list
        or next(
            (row.get("shape") for row in fields if row.get("name") == "occupancy"),
            None,
        )
        != [1024]
        or next(
            (row.get("shape") for row in fields if row.get("name") == "agent_id"),
            None,
        )
        != [512]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "micro denominator is not exact W32/C512"
        )
    target = _require_exact_dict(value.get("target_verdict"), "micro target verdict")
    target_body = {
        key: item for key, item in target.items() if key != "target_verdict_sha256"
    }
    validation = _require_exact_dict(
        target.get("authoritative_validation"), "micro authoritative validation"
    )
    validation_body = {
        key: item for key, item in validation.items() if key != "validation_sha256"
    }
    replay = _require_exact_dict(validation.get("live_replay"), "micro live replay")
    replay_body = {key: item for key, item in replay.items() if key != "replay_sha256"}
    current_bindings = _micro_governing_contract_bindings_v2()
    if (
        target.get("target_verdict_sha256") != canonical_sha256(target_body)
        or validation.get("validation_sha256") != canonical_sha256(validation_body)
        or replay.get("replay_sha256") != canonical_sha256(replay_body)
        or validation.get("artifact_sha256") != file_sha256(artifact)
        or validation.get("manifest_sha256") != manifest.get("manifest_sha256")
        or validation.get("artifact_path") != str(artifact)
        or validation.get("harness_sha256")
        != file_sha256(PROJECT_ROOT_V2 / MICRO_BENCHMARK_RELATIVE_PATH_V2)
        or validation.get("performance_restructure_contract_sha256")
        != current_bindings["performance_contract"]["sha256"]
        or validation.get("validation_matrix_sha256")
        != file_sha256(PROJECT_ROOT_V2 / VALIDATION_MATRIX_RELATIVE_PATH_V2)
        or validation.get("device_identity") != dict(expected_device_identity)
        or value.get("device_identity") != dict(expected_device_identity)
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "micro live-source/device binding is invalid"
        )
    normative = manifest.get("normative_execution_contracts")
    expected_contracts = {
        (binding["path"], binding["sha256"]) for binding in current_bindings.values()
    }
    observed_contracts = (
        {(row.get("path"), row.get("sha256")) for row in normative if type(row) is dict}
        if type(normative) is list
        else set()
    )
    if type(normative) is not list or not expected_contracts <= observed_contracts:
        raise FullEvolutionWorkhorseAuthorityError(
            "micro artifact omits the current governing contract bindings"
        )
    measurements = replay.get("measurements")
    selected = replay.get("selected")
    if type(measurements) is not list or type(selected) is not dict:
        raise FullEvolutionWorkhorseAuthorityError(
            "micro replay measurement set is incomplete"
        )
    selected_mode = replay.get("selected_execution_mode")
    expected_modes = {"eager_monolith", "cuda_graph_monolith"}
    selected_rows = []
    observed_modes = set()
    for measurement in measurements:
        measurement = _require_exact_dict(measurement, "micro measurement")
        measurement_body = {
            key: item
            for key, item in measurement.items()
            if key != "measurement_sha256"
        }
        if (
            measurement.get("authoritative") is not True
            or measurement.get("authority_class") != "AUTHORITATIVE_FRESH_LIVE_REPLAY"
            or measurement.get("measurement_sha256")
            != canonical_sha256(measurement_body)
            or measurement.get("artifact_sha256") != file_sha256(artifact)
            or measurement.get("manifest_sha256") != manifest.get("manifest_sha256")
            or measurement.get("device_identity") != dict(expected_device_identity)
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "micro measurement seal/source binding is invalid"
            )
        observed_modes.add(measurement.get("execution_mode"))
        if measurement.get("execution_mode") == selected_mode:
            selected_rows.append(measurement)
    if (
        len(measurements) != len(expected_modes)
        or observed_modes != expected_modes
        or selected_mode not in expected_modes
        or len(selected_rows) != 1
        or selected_rows[0] != selected
        or value.get("measurements") != measurements
        or value.get("selected") != selected
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "micro selected measurement is not unique"
        )
    candidate = _require_exact_dict(selected.get("candidate"), "micro candidate")
    population = _require_exact_dict(
        selected.get("initial_population"), "micro initial population"
    )
    batch_size = candidate.get("batch_size")
    temporal_k = candidate.get("temporal_k")
    if (
        type(batch_size) is not int
        or batch_size <= 0
        or type(temporal_k) is not int
        or temporal_k <= 0
        or population
        != {
            "world_count": batch_size,
            "total_active_population": batch_size * 256,
            "minimum_active_population": 256,
            "maximum_active_population": 256,
        }
        or selected.get("status") != "VALID"
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "micro denominator is not the exact live P256 fixture"
        )
    samples = selected.get("samples")
    if type(samples) is not list or not samples:
        raise FullEvolutionWorkhorseAuthorityError("micro selected samples are missing")
    rates: list[float] = []
    for sample in samples:
        sample = _require_exact_dict(sample, "micro timing sample")
        ticks = sample.get("committed_world_ticks")
        elapsed_ms = sample.get("cuda_event_elapsed_ms")
        if (
            type(ticks) is not int
            or ticks != batch_size * temporal_k
            or type(elapsed_ms) not in (int, float)
            or not math.isfinite(float(elapsed_ms))
            or elapsed_ms <= 0
            or sample.get("start_population") != population
            or not _valid_sha256(sample.get("event_content_sha256"))
            or not _valid_sha256(sample.get("final_state_sha256"))
        ):
            raise FullEvolutionWorkhorseAuthorityError("micro timing sample is invalid")
        rates.append(ticks * 1000.0 / float(elapsed_ms))
    recomputed_p10 = _percentile(rates, 0.10)
    if (
        selected.get("selected_statistic_p10_world_ticks_per_second") != recomputed_p10
        or replay.get("selected_p10_world_ticks_per_second") != recomputed_p10
        or value.get("selected_p10_world_ticks_per_second") != recomputed_p10
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "micro p10 does not recompute from sealed CUDA-event samples"
        )
    return {
        "schema_version": "eiel_workhorse_micro_denominator_binding_v2",
        "receipt_path": str(receipt),
        "receipt_file_sha256": file_sha256(receipt),
        "artifact_path": str(artifact),
        "artifact_sha256": file_sha256(artifact),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_inventory_sha256": manifest["source_inventory_sha256"],
        "device_identity": dict(expected_device_identity),
        "sites": 1024,
        "initial_population": 256,
        "capacity": 512,
        "p10_world_ticks_per_second": recomputed_p10,
        "selected_execution_mode": selected_mode,
        "selected_measurement_sha256": selected["measurement_sha256"],
        "micro_governing_contracts": current_bindings,
    }


def workhorse_event_dispatch_content_sha256_v2(
    *, metadata: Mapping[str, Any], event_rows: bytes, event_counts: bytes
) -> str:
    digest = hashlib.sha256()
    digest.update(b"EIEL-WORKHORSE-V2-EVENT-DISPATCH\0")
    digest.update(_json_bytes(dict(metadata)))
    digest.update(b"\0ROWS\0")
    digest.update(event_rows)
    digest.update(b"\0COUNTS\0")
    digest.update(event_counts)
    return digest.hexdigest()


def workhorse_sparse_event_dispatch_content_sha256_v3(
    *, metadata: Mapping[str, Any], packed_events: bytes
) -> str:
    digest = hashlib.sha256()
    digest.update(b"EIEL-WORKHORSE-V3-SPARSE-EVENT-DISPATCH\0")
    digest.update(_json_bytes(dict(metadata)))
    digest.update(b"\0PACKED-ROWS\0")
    digest.update(packed_events)
    return digest.hexdigest()


def _event_attempt_view_v2(
    value: Mapping[str, Any], *, historical: bool
) -> dict[str, Any]:
    """Normalize historical callback and canonical-parent row bindings.

    The canonical-parent branch validates only the exact sealed projection
    supplied by the outer Launchpad owner; it does not manufacture Scheduler
    authority inside the Workhorse engine.
    """
    if (
        isinstance(value, Mapping)
        and value.get("schema_version")
        == "eiel_workhorse_canonical_parent_attempt_binding_v2"
    ):
        from .canonical_rows import (
            validate_workhorse_canonical_parent_authority_binding_v2,
        )

        operation = value.get("operation")
        if operation not in {
            "EXECUTION_ROW",
            "SEMANTIC_EVIDENCE_CASE",
            "SCIENCE_COMPATIBILITY_DEMO",
            "INDEXED_ACS_CONTINUATION",
        }:
            raise FullEvolutionWorkhorseAuthorityError(
                "event sink canonical parent operation is unsupported"
            )
        binding = validate_workhorse_canonical_parent_authority_binding_v2(
            value, expected_operation=operation
        )
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            binding["contract"]
        )
        return {
            "binding": binding,
            "contract": contract,
            "binding_sha256": binding["binding_sha256"],
            "lineage_sha256": binding["attempt_lineage_sha256"],
            "owner_pid": binding["owner_pid"],
            "owner_process_start_ticks": binding["owner_process_start_ticks"],
            "device_uuid": binding["device_uuid"],
            "artifact_sha256": binding["artifact_sha256"],
            "B": contract.B,
            "start_next_tick": contract.start_next_tick,
            "target_next_tick": contract.start_next_tick + contract.total_ticks,
        }
    binding = validate_workhorse_attempt_authority_binding_v2(
        value, historical=historical
    )
    request = binding["request"]
    response = binding["response"]
    contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        request.get("contract", {})
    )
    return {
        "binding": binding,
        "contract": contract,
        "binding_sha256": binding["authority_binding_sha256"],
        "lineage_sha256": binding["attempt_lineage_sha256"],
        "owner_pid": response["owner_pid"],
        "owner_process_start_ticks": response["owner_process_start_ticks"],
        "device_uuid": response["device_uuid"],
        "artifact_sha256": request["artifact_sha256"],
        "B": request["B"],
        "start_next_tick": contract.start_next_tick,
        "target_next_tick": contract.start_next_tick + contract.total_ticks,
    }


def _validate_sparse_physical_segments_v3(
    segments: object,
    *,
    world_count: int,
    start_tick: int,
    end_tick: int,
    device_page_rows_per_world: int,
) -> list[dict[str, Any]]:
    if type(segments) is not list or not segments:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse event physical segments are absent"
        )
    expected_start = start_tick
    per_world_totals = [0] * world_count
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(segments):
        segment = _require_exact_dict(raw, f"sparse physical segment {index}")
        if set(segment) != {
            "segment_index",
            "start_next_absolute_tick",
            "end_next_absolute_tick",
            "tick_count",
            "event_counts",
            "packed_event_count",
        }:
            raise FullEvolutionWorkhorseAuthorityError(
                "sparse physical segment fields changed"
            )
        counts = segment.get("event_counts")
        segment_start = segment.get("start_next_absolute_tick")
        segment_end = segment.get("end_next_absolute_tick")
        if (
            segment.get("segment_index") != index
            or type(segment_start) is not int
            or type(segment_end) is not int
            or segment_start != expected_start
            or segment_end <= segment_start
            or segment.get("tick_count") != segment_end - segment_start
            or type(counts) is not list
            or len(counts) != world_count
            or any(
                type(count) is not int
                or count < 0
                or count > device_page_rows_per_world
                for count in counts
            )
            or segment.get("packed_event_count") != sum(counts)
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "sparse physical segment coverage or capacity is invalid"
            )
        for world, count in enumerate(counts):
            per_world_totals[world] += count
        expected_start = segment_end
        validated.append(dict(segment))
    if expected_start != end_tick:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse physical segments do not cover the dispatch interval"
        )
    return validated


def _validate_sparse_packed_rows_v3(
    *,
    packed_events: bytes,
    world_offsets: Sequence[int],
    event_counts: Sequence[int],
    start_tick: int,
    end_tick: int,
    capacity: int,
    sites: int,
    physical_segments: Sequence[Mapping[str, Any]],
) -> None:
    if len(packed_events) != sum(event_counts) * 48:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse packed event byte extent is invalid"
        )
    event_struct = struct.Struct("<QQQQQIHBB")
    segment_world_counts = [[0] * len(event_counts) for _segment in physical_segments]
    for world, count in enumerate(event_counts):
        begin = world_offsets[world]
        end = world_offsets[world + 1]
        if end - begin != count:
            raise FullEvolutionWorkhorseAuthorityError(
                "sparse world offsets differ from event counts"
            )
        prior_key: tuple[int, int, int] | None = None
        for row_index in range(begin, end):
            values = event_struct.unpack_from(packed_events, row_index * 48)
            (
                absolute_tick,
                event_id,
                agent_id,
                parent_a_id,
                parent_b_id,
                site_index,
                event_kind,
                status,
                reserved,
            ) = values
            slot = event_id & ((1 << 48) - 1)
            key = (absolute_tick, event_kind, slot)
            if (
                not start_tick <= absolute_tick < end_tick
                or event_kind not in (1, 2)
                or event_id != (event_kind << 48) | slot
                or slot >= capacity
                or site_index >= sites
                or agent_id <= 0
                or (event_kind == 2 and (parent_a_id <= 0 or parent_b_id <= 0))
                or status != 1
                or reserved != 0
                or (prior_key is not None and key <= prior_key)
            ):
                raise FullEvolutionWorkhorseAuthorityError(
                    "sparse event row identity or canonical order is invalid"
                )
            segment_index = next(
                (
                    index
                    for index, segment in enumerate(physical_segments)
                    if segment["start_next_absolute_tick"]
                    <= absolute_tick
                    < segment["end_next_absolute_tick"]
                ),
                None,
            )
            if segment_index is None:
                raise FullEvolutionWorkhorseAuthorityError(
                    "sparse event row is outside its physical segments"
                )
            segment_world_counts[segment_index][world] += 1
            prior_key = key
    expected_segment_world_counts = [
        list(segment["event_counts"]) for segment in physical_segments
    ]
    if segment_world_counts != expected_segment_world_counts:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse physical segment counts differ from packed event ticks"
        )


def publish_workhorse_sparse_event_sink_receipt_v3(
    sink_root: str | Path,
    *,
    sink_id: str,
    packed_events: bytes,
    event_counts: Sequence[int],
    physical_segments: Sequence[Mapping[str, Any]],
    overflow_recovery_count: int,
    world_uids: Sequence[int],
    start_next_absolute_tick: int,
    end_next_absolute_tick: int,
    artifact_identity_sha256: str,
    artifact_sha256: str,
    manifest_sha256: str,
    attempt_authority_binding: Mapping[str, Any],
    rng_keys: Sequence[int],
    pre_state_sha256: str,
    post_state_sha256: str,
    retain_packed_events: bool = True,
) -> Path:
    """Publish one canonical sparse dispatch after exact external consumption.

    The default exact-payload profile remains the semantic oracle.  Technical
    timing rows may use the compact consumer profile: every packed row is
    validated before acknowledgement, while the durable receipt retains the
    exact content digest and causal/frontier metadata instead of the raw
    payload.  The compact profile grants neither science nor qualification
    authority.
    """
    if type(packed_events) is not bytes:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse event sink requires immutable packed bytes"
        )
    if type(retain_packed_events) is not bool:
        raise FullEvolutionWorkhorseAuthorityError(
            "retain_packed_events must be boolean"
        )
    if type(sink_id) is not str or not sink_id.strip() or sink_id != sink_id.strip():
        raise FullEvolutionWorkhorseAuthorityError("sink_id must be a trimmed string")
    if type(overflow_recovery_count) is not int or overflow_recovery_count < 0:
        raise FullEvolutionWorkhorseAuthorityError(
            "overflow recovery count must be nonnegative"
        )
    if not all(
        _valid_sha256(value)
        for value in (
            artifact_identity_sha256,
            artifact_sha256,
            manifest_sha256,
            pre_state_sha256,
            post_state_sha256,
        )
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse event sink artifact hashes are invalid"
        )
    uids = tuple(world_uids)
    keys = tuple(rng_keys)
    counts = tuple(event_counts)
    if (
        not uids
        or len(set(uids)) != len(uids)
        or any(type(uid) is not int or uid <= 0 for uid in uids)
        or len(keys) != len(uids)
        or any(type(key) is not int or key < 0 or key >= 1 << 64 for key in keys)
        or len(counts) != len(uids)
        or any(type(count) is not int or count < 0 for count in counts)
        or type(start_next_absolute_tick) is not int
        or type(end_next_absolute_tick) is not int
        or end_next_absolute_tick <= start_next_absolute_tick
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse event sink world/count/tick frontier is invalid"
        )
    try:
        attempt_view = _event_attempt_view_v2(
            attempt_authority_binding, historical=False
        )
    except Exception as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"sparse event sink Attempt authority is invalid: {exc}"
        ) from exc
    attempt_binding = attempt_view["binding"]
    attempt_contract = attempt_view["contract"]
    if (
        attempt_view["B"] != len(uids)
        or attempt_view["artifact_sha256"] != artifact_sha256
        or workhorse_artifact_identity_v2(
            attempt_contract.shape, attempt_contract.backend_id
        )["identity_sha256"]
        != artifact_identity_sha256
        or start_next_absolute_tick < attempt_view["start_next_tick"]
        or end_next_absolute_tick > attempt_view["target_next_tick"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse event sink does not bind the exact started Attempt"
        )
    page_rows = 2 * attempt_contract.shape.capacity
    segments = _validate_sparse_physical_segments_v3(
        list(physical_segments),
        world_count=len(uids),
        start_tick=start_next_absolute_tick,
        end_tick=end_next_absolute_tick,
        device_page_rows_per_world=page_rows,
    )
    segment_totals = [
        sum(segment["event_counts"][world] for segment in segments)
        for world in range(len(uids))
    ]
    if tuple(segment_totals) != counts:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse segment counts do not sum to dispatch counts"
        )
    if overflow_recovery_count != len(segments) - 1:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse overflow recovery count differs from its binary split leaves"
        )
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    _validate_sparse_packed_rows_v3(
        packed_events=packed_events,
        world_offsets=offsets,
        event_counts=counts,
        start_tick=start_next_absolute_tick,
        end_tick=end_next_absolute_tick,
        capacity=attempt_contract.shape.capacity,
        sites=attempt_contract.shape.sites,
        physical_segments=segments,
    )
    # Preserve a compact per-world event-trajectory identity even when the
    # technical consumer releases the packed payload.  The digest is computed
    # over each world's canonical row slice, so it is invariant to physical B
    # repacking while remaining bound to the exact sparse event bytes.
    world_event_rows_sha256 = [
        _sha_bytes(
            packed_events[offsets[index] * 48 : offsets[index + 1] * 48]
        )
        for index in range(len(uids))
    ]
    benchmark_selectable = (
        overflow_recovery_count == 0
        and len(segments) == 1
        and segments[0]["start_next_absolute_tick"] == start_next_absolute_tick
        and segments[0]["end_next_absolute_tick"] == end_next_absolute_tick
    )
    rng_identity = {
        "schema_version": "eiel_workhorse_rng_identity_v2",
        "rng_epoch": 4,
        "world_uids": list(uids),
        "rng_keys": list(keys),
    }
    rng_identity_sha256 = canonical_sha256(rng_identity)
    dispatch_identity_body = {
        "schema_version": "eiel_workhorse_state_dispatch_identity_v3",
        "attempt_authority_binding_sha256": attempt_view["binding_sha256"],
        "attempt_lineage_sha256": attempt_view["lineage_sha256"],
        "artifact_identity_sha256": artifact_identity_sha256,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "start_next_absolute_tick": start_next_absolute_tick,
        "end_next_absolute_tick": end_next_absolute_tick,
        "rng_identity_sha256": rng_identity_sha256,
        "pre_state_sha256": pre_state_sha256,
        "post_state_sha256": post_state_sha256,
        "overflow_recovery_count": overflow_recovery_count,
        "benchmark_selectable": benchmark_selectable,
    }
    dispatch_identity_sha256 = canonical_sha256(dispatch_identity_body)
    metadata = {
        "schema_version": WORKHORSE_SPARSE_EVENT_DISPATCH_SCHEMA_V3,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "wire_abi": "eiel_full_evolution_workhorse_sparse_lifecycle_event_v3",
        "wire_endianness": "little",
        "packed_event_row_bytes": 48,
        "packed_event_count": offsets[-1],
        "packed_event_nbytes": len(packed_events),
        "world_offsets": offsets,
        "event_counts": list(counts),
        "world_event_rows_sha256": world_event_rows_sha256,
        "device_page_rows_per_world": page_rows,
        "resident_event_growth_with_temporal_k_bytes": 0,
        "physical_segments": segments,
        "overflow_recovery_count": overflow_recovery_count,
        "benchmark_selectable": benchmark_selectable,
        "world_uids": list(uids),
        "start_next_absolute_tick": start_next_absolute_tick,
        "end_next_absolute_tick": end_next_absolute_tick,
        "tick_count": end_next_absolute_tick - start_next_absolute_tick,
        "artifact_identity_sha256": artifact_identity_sha256,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "attempt_authority_binding": attempt_binding,
        "attempt_authority_binding_sha256": attempt_view["binding_sha256"],
        "attempt_lineage_sha256": attempt_view["lineage_sha256"],
        "owner_pid": attempt_view["owner_pid"],
        "owner_process_start_ticks": attempt_view["owner_process_start_ticks"],
        "device_uuid": attempt_view["device_uuid"],
        "rng_identity": rng_identity,
        "rng_identity_sha256": rng_identity_sha256,
        "pre_state_sha256": pre_state_sha256,
        "post_state_sha256": post_state_sha256,
        "dispatch_identity": dispatch_identity_body,
        "dispatch_identity_sha256": dispatch_identity_sha256,
    }
    packed_events_sha256 = _sha_bytes(packed_events)
    if retain_packed_events:
        content_sha = workhorse_sparse_event_dispatch_content_sha256_v3(
            metadata=metadata, packed_events=packed_events
        )
    else:
        compact_content = {
            "schema_version": WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4,
            "consumer_profile": WORKHORSE_TECHNICAL_DIGEST_CONSUMER_PROFILE_V1,
            "dispatch": metadata,
            "packed_events": {
                "bytes": len(packed_events),
                "sha256": packed_events_sha256,
            },
        }
        content_sha = canonical_sha256(compact_content)
    root = Path(sink_root)
    if root.is_symlink():
        raise FullEvolutionWorkhorseAuthorityError("event sink root is a symlink")
    root.mkdir(parents=True, exist_ok=True)
    dispatch_dir = root / content_sha
    try:
        dispatch_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse event sink dispatch content was already published"
        ) from exc
    if retain_packed_events:
        events_path = dispatch_dir / "packed_events.bin"
        with events_path.open("xb") as handle:
            handle.write(packed_events)
            handle.flush()
            os.fsync(handle.fileno())
        events_path.chmod(0o444)
        if events_path.read_bytes() != packed_events:
            raise FullEvolutionWorkhorseAuthorityError(
                "sparse event sink readback changed packed content"
            )
        body = {
            "schema_version": WORKHORSE_SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3,
            "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
            "sink_id": sink_id,
            "sink_kind": "immutable_file_exact_sparse_content_v3",
            "dispatch": metadata,
            "dispatch_content_sha256": content_sha,
            "packed_events": {
                "file_name": events_path.name,
                "bytes": len(packed_events),
                "sha256": packed_events_sha256,
            },
        }
    else:
        body = {
            "schema_version": WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4,
            "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
            "sink_id": sink_id,
            "sink_kind": "immutable_compact_sparse_consumer_v4",
            "consumer_profile": WORKHORSE_TECHNICAL_DIGEST_CONSUMER_PROFILE_V1,
            "dispatch": metadata,
            "dispatch_content_sha256": content_sha,
            "consumed_packed_events": {
                "bytes": len(packed_events),
                "sha256": packed_events_sha256,
                "raw_payload_retained": False,
            },
            "science_authority_granted": False,
            "qualification_authority_granted": False,
        }
    receipt_value = {**body, "receipt_body_sha256": canonical_sha256(body)}
    receipt_path = dispatch_dir / "sink_receipt.json"
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt_value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    receipt_path.chmod(0o444)
    _fsync_directory(dispatch_dir)
    dispatch_dir.chmod(0o555)
    _fsync_directory(root)
    validate_workhorse_event_sink_receipt_v2(
        receipt_path,
        expected_artifact_identity_sha256=artifact_identity_sha256,
        expected_end_next_absolute_tick=end_next_absolute_tick,
    )
    return receipt_path


def _validate_workhorse_sparse_event_sink_receipt_v3(
    receipt: Path,
    value: Mapping[str, Any],
    *,
    expected_artifact_identity_sha256: str | None,
    expected_end_next_absolute_tick: int | None,
    expected_attempt_authority_binding_sha256: str | None,
    expected_rng_identity_sha256: str | None,
    expected_post_state_sha256: str | None,
) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != "receipt_body_sha256"}
    dispatch = _require_exact_dict(value.get("dispatch"), "sparse event dispatch")
    compact_consumer = (
        value.get("schema_version") == WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4
    )
    packed = _require_exact_dict(
        value.get("consumed_packed_events" if compact_consumer else "packed_events"),
        "consumed packed event identity" if compact_consumer else "packed event file",
    )
    if (
        value.get("schema_version")
        not in {
            WORKHORSE_SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3,
            WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4,
        }
        or value.get("time_authority") != WORKHORSE_TIME_AUTHORITY_V2
        or (
            compact_consumer
            and (
                value.get("sink_kind") != "immutable_compact_sparse_consumer_v4"
                or value.get("consumer_profile")
                != WORKHORSE_TECHNICAL_DIGEST_CONSUMER_PROFILE_V1
                or value.get("science_authority_granted") is not False
                or value.get("qualification_authority_granted") is not False
            )
        )
        or (
            not compact_consumer
            and value.get("sink_kind") != "immutable_file_exact_sparse_content_v3"
        )
        or value.get("receipt_body_sha256") != canonical_sha256(body)
        or dispatch.get("schema_version") != WORKHORSE_SPARSE_EVENT_DISPATCH_SCHEMA_V3
        or dispatch.get("time_authority") != WORKHORSE_TIME_AUTHORITY_V2
        or dispatch.get("wire_abi")
        != "eiel_full_evolution_workhorse_sparse_lifecycle_event_v3"
        or dispatch.get("wire_endianness") != "little"
        or dispatch.get("packed_event_row_bytes") != 48
        or dispatch.get("resident_event_growth_with_temporal_k_bytes") != 0
        or receipt.name != "sink_receipt.json"
        or receipt.parent.name != value.get("dispatch_content_sha256")
        or receipt.parent.stat().st_mode & 0o222
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse event sink receipt schema or seal is invalid"
        )
    if (
        expected_artifact_identity_sha256 is not None
        and dispatch.get("artifact_identity_sha256")
        != expected_artifact_identity_sha256
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse event sink artifact identity does not match"
        )
    if (
        expected_end_next_absolute_tick is not None
        and dispatch.get("end_next_absolute_tick") != expected_end_next_absolute_tick
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse event sink frontier does not match"
        )
    try:
        attempt_view = _event_attempt_view_v2(
            dispatch.get("attempt_authority_binding"), historical=True
        )
    except Exception as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"sparse event sink Attempt authority no longer validates: {exc}"
        ) from exc
    attempt_binding = attempt_view["binding"]
    attempt_contract = attempt_view["contract"]
    rng_identity = _require_exact_dict(
        dispatch.get("rng_identity"), "sparse event RNG identity"
    )
    dispatch_identity = _require_exact_dict(
        dispatch.get("dispatch_identity"), "sparse state dispatch identity"
    )
    expected_dispatch_identity = {
        "schema_version": "eiel_workhorse_state_dispatch_identity_v3",
        "attempt_authority_binding_sha256": attempt_view["binding_sha256"],
        "attempt_lineage_sha256": attempt_view["lineage_sha256"],
        "artifact_identity_sha256": dispatch.get("artifact_identity_sha256"),
        "artifact_sha256": dispatch.get("artifact_sha256"),
        "manifest_sha256": dispatch.get("manifest_sha256"),
        "start_next_absolute_tick": dispatch.get("start_next_absolute_tick"),
        "end_next_absolute_tick": dispatch.get("end_next_absolute_tick"),
        "rng_identity_sha256": dispatch.get("rng_identity_sha256"),
        "pre_state_sha256": dispatch.get("pre_state_sha256"),
        "post_state_sha256": dispatch.get("post_state_sha256"),
        "overflow_recovery_count": dispatch.get("overflow_recovery_count"),
        "benchmark_selectable": dispatch.get("benchmark_selectable"),
    }
    world_uids = dispatch.get("world_uids")
    counts = dispatch.get("event_counts")
    offsets = dispatch.get("world_offsets")
    world_event_rows_sha256 = dispatch.get("world_event_rows_sha256")
    start_tick = dispatch.get("start_next_absolute_tick")
    end_tick = dispatch.get("end_next_absolute_tick")
    tick_count = dispatch.get("tick_count")
    page_rows = dispatch.get("device_page_rows_per_world")
    overflow_count = dispatch.get("overflow_recovery_count")
    if (
        dispatch.get("attempt_authority_binding_sha256")
        != attempt_view["binding_sha256"]
        or dispatch.get("attempt_lineage_sha256") != attempt_view["lineage_sha256"]
        or dispatch.get("owner_pid") != attempt_view["owner_pid"]
        or dispatch.get("owner_process_start_ticks")
        != attempt_view["owner_process_start_ticks"]
        or dispatch.get("device_uuid") != attempt_view["device_uuid"]
        or attempt_view["artifact_sha256"] != dispatch.get("artifact_sha256")
        or workhorse_artifact_identity_v2(
            attempt_contract.shape, attempt_contract.backend_id
        )["identity_sha256"]
        != dispatch.get("artifact_identity_sha256")
        or dispatch.get("start_next_absolute_tick") < attempt_view["start_next_tick"]
        or dispatch.get("end_next_absolute_tick") > attempt_view["target_next_tick"]
        or rng_identity.get("schema_version") != "eiel_workhorse_rng_identity_v2"
        or rng_identity.get("rng_epoch") != 4
        or rng_identity.get("world_uids") != world_uids
        or type(rng_identity.get("rng_keys")) is not list
        or len(rng_identity["rng_keys"]) != attempt_view["B"]
        or any(
            type(key) is not int or key < 0 or key >= 1 << 64
            for key in rng_identity["rng_keys"]
        )
        or dispatch.get("rng_identity_sha256") != canonical_sha256(rng_identity)
        or not _valid_sha256(dispatch.get("pre_state_sha256"))
        or not _valid_sha256(dispatch.get("post_state_sha256"))
        or dispatch_identity != expected_dispatch_identity
        or dispatch.get("dispatch_identity_sha256")
        != canonical_sha256(dispatch_identity)
        or (
            expected_attempt_authority_binding_sha256 is not None
            and dispatch.get("attempt_authority_binding_sha256")
            != expected_attempt_authority_binding_sha256
        )
        or (
            expected_rng_identity_sha256 is not None
            and dispatch.get("rng_identity_sha256") != expected_rng_identity_sha256
        )
        or (
            expected_post_state_sha256 is not None
            and dispatch.get("post_state_sha256") != expected_post_state_sha256
        )
        or type(world_uids) is not list
        or len(world_uids) != attempt_view["B"]
        or len(set(world_uids)) != len(world_uids)
        or any(type(uid) is not int or uid <= 0 for uid in world_uids)
        or type(counts) is not list
        or len(counts) != len(world_uids)
        or any(type(count) is not int or count < 0 for count in counts)
        or type(offsets) is not list
        or len(offsets) != len(world_uids) + 1
        or offsets[0] != 0
        or any(type(offset) is not int or offset < 0 for offset in offsets)
        or tuple(offsets[index + 1] - offsets[index] for index in range(len(counts)))
        != tuple(counts)
        or type(start_tick) is not int
        or type(end_tick) is not int
        or type(tick_count) is not int
        or end_tick <= start_tick
        or tick_count != end_tick - start_tick
        or page_rows != 2 * attempt_contract.shape.capacity
        or type(overflow_count) is not int
        or overflow_count < 0
        or dispatch.get("packed_event_count") != offsets[-1]
        or dispatch.get("packed_event_nbytes") != offsets[-1] * 48
        or (
            world_event_rows_sha256 is not None
            and (
                type(world_event_rows_sha256) is not list
                or len(world_event_rows_sha256) != len(world_uids)
                or any(not _valid_sha256(item) for item in world_event_rows_sha256)
            )
        )
        or type(value.get("sink_id")) is not str
        or not value["sink_id"].strip()
        or value["sink_id"] != value["sink_id"].strip()
        or not all(
            _valid_sha256(dispatch.get(key))
            for key in (
                "artifact_identity_sha256",
                "artifact_sha256",
                "manifest_sha256",
            )
        )
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse event owner/identity/shape binding is invalid"
        )
    segments = _validate_sparse_physical_segments_v3(
        dispatch.get("physical_segments"),
        world_count=len(world_uids),
        start_tick=start_tick,
        end_tick=end_tick,
        device_page_rows_per_world=page_rows,
    )
    if [
        sum(segment["event_counts"][world] for segment in segments)
        for world in range(len(world_uids))
    ] != counts:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse physical segment counts do not recompute"
        )
    if overflow_count != len(segments) - 1:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse overflow recovery count differs from its binary split leaves"
        )
    expected_selectable = (
        overflow_count == 0
        and len(segments) == 1
        and segments[0]["start_next_absolute_tick"] == start_tick
        and segments[0]["end_next_absolute_tick"] == end_tick
    )
    if dispatch.get("benchmark_selectable") is not expected_selectable:
        raise FullEvolutionWorkhorseAuthorityError(
            "sparse overflow recovery selection policy changed"
        )
    expected_packed_fields = (
        {"bytes", "sha256", "raw_payload_retained"}
        if compact_consumer
        else {"file_name", "bytes", "sha256"}
    )
    if set(packed) != expected_packed_fields:
        raise FullEvolutionWorkhorseAuthorityError(
            "packed event file identity fields changed"
        )
    if compact_consumer:
        compact_content = {
            "schema_version": WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4,
            "consumer_profile": WORKHORSE_TECHNICAL_DIGEST_CONSUMER_PROFILE_V1,
            "dispatch": dispatch,
            "packed_events": {
                "bytes": packed.get("bytes"),
                "sha256": packed.get("sha256"),
            },
        }
        if (
            packed.get("bytes") != dispatch.get("packed_event_nbytes")
            or not _valid_sha256(packed.get("sha256"))
            or packed.get("raw_payload_retained") is not False
            or value.get("dispatch_content_sha256") != canonical_sha256(compact_content)
            or list(receipt.parent.iterdir()) != [receipt]
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "compact event consumer seal does not recompute"
            )
        packed_events_sha256 = packed["sha256"]
    else:
        file_name = packed.get("file_name")
        if (
            type(file_name) is not str
            or Path(file_name).name != file_name
            or file_name in {"", ".", ".."}
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "packed event file locator is unsafe"
            )
        packed_path = _regular_file(
            receipt.parent / file_name, "packed event rows", read_only=True
        )
        packed_events = packed_path.read_bytes()
        if (
            packed.get("bytes") != len(packed_events)
            or packed.get("sha256") != _sha_bytes(packed_events)
            or dispatch.get("packed_event_nbytes") != len(packed_events)
            or value.get("dispatch_content_sha256")
            != workhorse_sparse_event_dispatch_content_sha256_v3(
                metadata=dispatch, packed_events=packed_events
            )
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "sparse event exact-content seal does not recompute"
            )
        _validate_sparse_packed_rows_v3(
            packed_events=packed_events,
            world_offsets=offsets,
            event_counts=counts,
            start_tick=start_tick,
            end_tick=end_tick,
            capacity=attempt_contract.shape.capacity,
            sites=attempt_contract.shape.sites,
            physical_segments=segments,
        )
        if world_event_rows_sha256 is not None:
            expected_world_event_rows = [
                _sha_bytes(
                    packed_events[offsets[index] * 48 : offsets[index + 1] * 48]
                )
                for index in range(len(world_uids))
            ]
            if world_event_rows_sha256 != expected_world_event_rows:
                raise FullEvolutionWorkhorseAuthorityError(
                    "sparse per-world event trajectory seal does not recompute"
                )
        packed_events_sha256 = packed["sha256"]
    return {
        "schema_version": value["schema_version"],
        "receipt_path": str(receipt),
        "receipt_file_sha256": file_sha256(receipt),
        "receipt_body_sha256": value["receipt_body_sha256"],
        "dispatch_content_sha256": value["dispatch_content_sha256"],
        "event_rows_file_sha256": packed_events_sha256,
        "event_counts_file_sha256": canonical_sha256(counts),
        "packed_events_file_sha256": packed_events_sha256,
        "raw_payload_retained": not compact_consumer,
        "event_count": offsets[-1],
        "event_counts": counts,
        "world_event_rows_sha256": world_event_rows_sha256,
        "world_offsets": offsets,
        "device_page_rows_per_world": page_rows,
        "resident_event_growth_with_temporal_k_bytes": 0,
        "physical_segments": segments,
        "overflow_recovery_count": overflow_count,
        "benchmark_selectable": expected_selectable,
        "world_uids": world_uids,
        "start_next_absolute_tick": start_tick,
        "end_next_absolute_tick": end_tick,
        "tick_count": tick_count,
        "artifact_identity_sha256": dispatch["artifact_identity_sha256"],
        "artifact_sha256": dispatch["artifact_sha256"],
        "manifest_sha256": dispatch["manifest_sha256"],
        "attempt_authority_binding": attempt_binding,
        "attempt_authority_binding_sha256": dispatch[
            "attempt_authority_binding_sha256"
        ],
        "attempt_lineage_sha256": dispatch["attempt_lineage_sha256"],
        "owner_pid": dispatch["owner_pid"],
        "owner_process_start_ticks": dispatch["owner_process_start_ticks"],
        "device_uuid": dispatch["device_uuid"],
        "rng_identity": rng_identity,
        "rng_identity_sha256": dispatch["rng_identity_sha256"],
        "pre_state_sha256": dispatch["pre_state_sha256"],
        "post_state_sha256": dispatch["post_state_sha256"],
        "dispatch_identity_sha256": dispatch["dispatch_identity_sha256"],
    }


def publish_workhorse_event_sink_receipt_v2(
    sink_root: str | Path,
    *,
    sink_id: str,
    event_rows: bytes,
    event_counts: bytes,
    event_rows_shape: Sequence[int],
    event_counts_shape: Sequence[int],
    world_uids: Sequence[int],
    start_next_absolute_tick: int,
    end_next_absolute_tick: int,
    artifact_identity_sha256: str,
    artifact_sha256: str,
    manifest_sha256: str,
    attempt_authority_binding: Mapping[str, Any],
    rng_keys: Sequence[int],
    pre_state_sha256: str,
    post_state_sha256: str,
    compress_payloads: bool = False,
) -> Path:
    """External file sink: persist, fsync, reopen, then issue an exact ack."""
    if (
        type(event_rows) is not bytes
        or type(event_counts) is not bytes
        or not event_rows
        or not event_counts
        or type(compress_payloads) is not bool
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event sink requires immutable nonempty row/count bytes"
        )
    if type(sink_id) is not str or not sink_id.strip() or sink_id != sink_id.strip():
        raise FullEvolutionWorkhorseAuthorityError("sink_id must be a trimmed string")
    if not all(
        _valid_sha256(value)
        for value in (
            artifact_identity_sha256,
            artifact_sha256,
            manifest_sha256,
            pre_state_sha256,
            post_state_sha256,
        )
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event sink artifact hashes are invalid"
        )
    uids = tuple(world_uids)
    keys = tuple(rng_keys)
    if (
        not uids
        or len(set(uids)) != len(uids)
        or any(type(uid) is not int or uid <= 0 for uid in uids)
        or len(keys) != len(uids)
        or any(type(key) is not int or key < 0 or key >= 1 << 64 for key in keys)
        or type(start_next_absolute_tick) is not int
        or type(end_next_absolute_tick) is not int
        or end_next_absolute_tick <= start_next_absolute_tick
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event sink world/tick frontier is invalid"
        )
    try:
        attempt_view = _event_attempt_view_v2(
            attempt_authority_binding, historical=False
        )
    except Exception as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"event sink Attempt authority is invalid: {exc}"
        ) from exc
    attempt_binding = attempt_view["binding"]
    attempt_contract = attempt_view["contract"]
    if (
        attempt_view["B"] != len(uids)
        or attempt_view["artifact_sha256"] != artifact_sha256
        or start_next_absolute_tick < attempt_view["start_next_tick"]
        or end_next_absolute_tick > attempt_view["target_next_tick"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event sink does not bind an exact started B/artifact Attempt"
        )
    rng_identity = {
        "schema_version": "eiel_workhorse_rng_identity_v2",
        "rng_epoch": 4,
        "world_uids": list(uids),
        "rng_keys": list(keys),
    }
    rng_identity_sha256 = canonical_sha256(rng_identity)
    dispatch_identity_body = {
        "schema_version": "eiel_workhorse_state_dispatch_identity_v2",
        "attempt_authority_binding_sha256": attempt_view["binding_sha256"],
        "attempt_lineage_sha256": attempt_view["lineage_sha256"],
        "artifact_identity_sha256": artifact_identity_sha256,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "start_next_absolute_tick": start_next_absolute_tick,
        "end_next_absolute_tick": end_next_absolute_tick,
        "rng_identity_sha256": rng_identity_sha256,
        "pre_state_sha256": pre_state_sha256,
        "post_state_sha256": post_state_sha256,
    }
    dispatch_identity_sha256 = canonical_sha256(dispatch_identity_body)
    rows_shape = tuple(event_rows_shape)
    counts_shape = tuple(event_counts_shape)
    if (
        len(rows_shape) != 4
        or rows_shape[0] != len(uids)
        or rows_shape[1] != end_next_absolute_tick - start_next_absolute_tick
        or rows_shape[2] != 2 * attempt_contract.shape.capacity
        or any(type(dimension) is not int or dimension <= 0 for dimension in rows_shape)
        or rows_shape[-1] != 48
        or counts_shape != (len(uids),)
        or math.prod(rows_shape) != len(event_rows)
        or 4 * math.prod(counts_shape) != len(event_counts)
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event sink byte lengths do not match the exact tensor shapes"
        )
    metadata = {
        "schema_version": WORKHORSE_EVENT_DISPATCH_SCHEMA_V2,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "world_uids": list(uids),
        "start_next_absolute_tick": start_next_absolute_tick,
        "end_next_absolute_tick": end_next_absolute_tick,
        "tick_count": end_next_absolute_tick - start_next_absolute_tick,
        "event_rows_shape": list(rows_shape),
        "event_counts_shape": list(counts_shape),
        "artifact_identity_sha256": artifact_identity_sha256,
        "artifact_sha256": artifact_sha256,
        "manifest_sha256": manifest_sha256,
        "attempt_authority_binding": attempt_binding,
        "attempt_authority_binding_sha256": attempt_view["binding_sha256"],
        "attempt_lineage_sha256": attempt_view["lineage_sha256"],
        "owner_pid": attempt_view["owner_pid"],
        "owner_process_start_ticks": attempt_view["owner_process_start_ticks"],
        "device_uuid": attempt_view["device_uuid"],
        "rng_identity": rng_identity,
        "rng_identity_sha256": rng_identity_sha256,
        "pre_state_sha256": pre_state_sha256,
        "post_state_sha256": post_state_sha256,
        "dispatch_identity": dispatch_identity_body,
        "dispatch_identity_sha256": dispatch_identity_sha256,
    }
    content_sha = workhorse_event_dispatch_content_sha256_v2(
        metadata=metadata, event_rows=event_rows, event_counts=event_counts
    )
    root = Path(sink_root)
    if root.is_symlink():
        raise FullEvolutionWorkhorseAuthorityError("event sink root is a symlink")
    root.mkdir(parents=True, exist_ok=True)
    dispatch_dir = root / content_sha
    try:
        dispatch_dir.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            "event sink dispatch content was already published"
        ) from exc
    payload_rows: list[tuple[Path, bytes, bytes, str]] = []
    for stem, payload in (("event_rows", event_rows), ("event_counts", event_counts)):
        stored = payload
        encoding = "identity_v1"
        suffix = ".bin"
        if compress_payloads:
            if _zstd is None:
                compressed = zlib.compress(payload, level=1)
                compressed_encoding = "zlib_level_1_v1"
                compressed_suffix = ".bin.zlib"
            else:
                compressed = _zstd.compress(payload, level=1)
                compressed_encoding = "zstd_level_1_v1"
                compressed_suffix = ".bin.zst"
            if len(compressed) < len(payload):
                stored = compressed
                encoding = compressed_encoding
                suffix = compressed_suffix
        payload_rows.append(
            (dispatch_dir / f"{stem}{suffix}", payload, stored, encoding)
        )
    for path, _decoded, stored, _encoding in payload_rows:
        with path.open("xb") as handle:
            handle.write(stored)
            handle.flush()
            os.fsync(handle.fileno())
        path.chmod(0o444)
    rows_path, _rows_decoded, rows_stored, rows_encoding = payload_rows[0]
    counts_path, _counts_decoded, counts_stored, counts_encoding = payload_rows[1]
    if (
        rows_path.read_bytes() != rows_stored
        or counts_path.read_bytes() != counts_stored
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "external event sink readback changed stored content"
        )

    def payload_identity(
        path: Path, decoded: bytes, stored: bytes, encoding: str
    ) -> dict[str, Any]:
        identity: dict[str, Any] = {
            "file_name": path.name,
            "bytes": len(decoded),
            "sha256": _sha_bytes(decoded),
        }
        if compress_payloads:
            identity.update(
                {
                    "storage_encoding": encoding,
                    "stored_bytes": len(stored),
                    "stored_sha256": _sha_bytes(stored),
                }
            )
        return identity

    body = {
        "schema_version": WORKHORSE_EVENT_SINK_RECEIPT_SCHEMA_V2,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "sink_id": sink_id,
        "sink_kind": (
            "immutable_encoded_file_exact_content_v2"
            if compress_payloads
            else "immutable_file_exact_content_v2"
        ),
        "dispatch": metadata,
        "dispatch_content_sha256": content_sha,
        "event_rows": payload_identity(
            rows_path, event_rows, rows_stored, rows_encoding
        ),
        "event_counts": payload_identity(
            counts_path, event_counts, counts_stored, counts_encoding
        ),
    }
    receipt_value = {**body, "receipt_body_sha256": canonical_sha256(body)}
    receipt_path = dispatch_dir / "sink_receipt.json"
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(receipt_value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    receipt_path.chmod(0o444)
    _fsync_directory(dispatch_dir)
    dispatch_dir.chmod(0o555)
    _fsync_directory(root)
    validate_workhorse_event_sink_receipt_v2(
        receipt_path,
        expected_artifact_identity_sha256=artifact_identity_sha256,
        expected_end_next_absolute_tick=end_next_absolute_tick,
    )
    return receipt_path


def validate_workhorse_event_sink_receipt_v2(
    receipt_path: str | Path,
    *,
    expected_artifact_identity_sha256: str | None = None,
    expected_end_next_absolute_tick: int | None = None,
    expected_attempt_authority_binding_sha256: str | None = None,
    expected_rng_identity_sha256: str | None = None,
    expected_post_state_sha256: str | None = None,
) -> dict[str, Any]:
    """Reopen an exact payload or publisher-validated compact consumer receipt."""
    receipt, value = _read_json(receipt_path, "workhorse event sink receipt")
    if value.get("schema_version") in {
        WORKHORSE_SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3,
        WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4,
    }:
        return _validate_workhorse_sparse_event_sink_receipt_v3(
            receipt,
            value,
            expected_artifact_identity_sha256=(expected_artifact_identity_sha256),
            expected_end_next_absolute_tick=expected_end_next_absolute_tick,
            expected_attempt_authority_binding_sha256=(
                expected_attempt_authority_binding_sha256
            ),
            expected_rng_identity_sha256=expected_rng_identity_sha256,
            expected_post_state_sha256=expected_post_state_sha256,
        )
    body = {key: item for key, item in value.items() if key != "receipt_body_sha256"}
    dispatch = _require_exact_dict(value.get("dispatch"), "event dispatch identity")
    rows = _require_exact_dict(value.get("event_rows"), "event row file identity")
    counts = _require_exact_dict(value.get("event_counts"), "event count file identity")
    if (
        value.get("schema_version") != WORKHORSE_EVENT_SINK_RECEIPT_SCHEMA_V2
        or value.get("time_authority") != WORKHORSE_TIME_AUTHORITY_V2
        or value.get("sink_kind")
        not in {
            "immutable_file_exact_content_v2",
            "immutable_encoded_file_exact_content_v2",
        }
        or value.get("receipt_body_sha256") != canonical_sha256(body)
        or dispatch.get("schema_version") != WORKHORSE_EVENT_DISPATCH_SCHEMA_V2
        or dispatch.get("time_authority") != WORKHORSE_TIME_AUTHORITY_V2
        or receipt.name != "sink_receipt.json"
        or receipt.parent.name != value.get("dispatch_content_sha256")
        or receipt.parent.stat().st_mode & 0o222
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event sink receipt schema or seal is invalid"
        )
    if (
        expected_artifact_identity_sha256 is not None
        and dispatch.get("artifact_identity_sha256")
        != expected_artifact_identity_sha256
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event sink artifact identity does not match"
        )
    if (
        expected_end_next_absolute_tick is not None
        and dispatch.get("end_next_absolute_tick") != expected_end_next_absolute_tick
    ):
        raise FullEvolutionWorkhorseAuthorityError("event sink frontier does not match")
    try:
        attempt_view = _event_attempt_view_v2(
            dispatch.get("attempt_authority_binding"), historical=True
        )
    except Exception as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"event sink Attempt authority no longer validates: {exc}"
        ) from exc
    attempt_binding = attempt_view["binding"]
    attempt_contract = attempt_view["contract"]
    rng_identity = _require_exact_dict(
        dispatch.get("rng_identity"), "event RNG identity"
    )
    dispatch_identity = _require_exact_dict(
        dispatch.get("dispatch_identity"), "event state dispatch identity"
    )
    expected_dispatch_identity = {
        "schema_version": "eiel_workhorse_state_dispatch_identity_v2",
        "attempt_authority_binding_sha256": attempt_view["binding_sha256"],
        "attempt_lineage_sha256": attempt_view["lineage_sha256"],
        "artifact_identity_sha256": dispatch.get("artifact_identity_sha256"),
        "artifact_sha256": dispatch.get("artifact_sha256"),
        "manifest_sha256": dispatch.get("manifest_sha256"),
        "start_next_absolute_tick": dispatch.get("start_next_absolute_tick"),
        "end_next_absolute_tick": dispatch.get("end_next_absolute_tick"),
        "rng_identity_sha256": dispatch.get("rng_identity_sha256"),
        "pre_state_sha256": dispatch.get("pre_state_sha256"),
        "post_state_sha256": dispatch.get("post_state_sha256"),
    }
    if (
        dispatch.get("attempt_authority_binding_sha256")
        != attempt_view["binding_sha256"]
        or dispatch.get("attempt_lineage_sha256") != attempt_view["lineage_sha256"]
        or dispatch.get("owner_pid") != attempt_view["owner_pid"]
        or dispatch.get("owner_process_start_ticks")
        != attempt_view["owner_process_start_ticks"]
        or dispatch.get("device_uuid") != attempt_view["device_uuid"]
        or attempt_view["B"] != len(dispatch.get("world_uids", ()))
        or attempt_view["artifact_sha256"] != dispatch.get("artifact_sha256")
        or dispatch.get("start_next_absolute_tick") < attempt_view["start_next_tick"]
        or dispatch.get("end_next_absolute_tick") > attempt_view["target_next_tick"]
        or rng_identity.get("schema_version") != "eiel_workhorse_rng_identity_v2"
        or rng_identity.get("rng_epoch") != 4
        or rng_identity.get("world_uids") != dispatch.get("world_uids")
        or type(rng_identity.get("rng_keys")) is not list
        or len(rng_identity["rng_keys"]) != len(dispatch.get("world_uids", ()))
        or any(
            type(key) is not int or key < 0 or key >= 1 << 64
            for key in rng_identity["rng_keys"]
        )
        or dispatch.get("rng_identity_sha256") != canonical_sha256(rng_identity)
        or not _valid_sha256(dispatch.get("pre_state_sha256"))
        or not _valid_sha256(dispatch.get("post_state_sha256"))
        or dispatch_identity != expected_dispatch_identity
        or dispatch.get("dispatch_identity_sha256")
        != canonical_sha256(dispatch_identity)
        or (
            expected_attempt_authority_binding_sha256 is not None
            and dispatch.get("attempt_authority_binding_sha256")
            != expected_attempt_authority_binding_sha256
        )
        or (
            expected_rng_identity_sha256 is not None
            and dispatch.get("rng_identity_sha256") != expected_rng_identity_sha256
        )
        or (
            expected_post_state_sha256 is not None
            and dispatch.get("post_state_sha256") != expected_post_state_sha256
        )
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event owner/Attempt/RNG/state dispatch binding is invalid"
        )
    encoded_storage = (
        value.get("sink_kind") == "immutable_encoded_file_exact_content_v2"
    )
    rows_shape = dispatch.get("event_rows_shape")
    counts_shape = dispatch.get("event_counts_shape")
    world_uids = dispatch.get("world_uids")
    start_tick = dispatch.get("start_next_absolute_tick")
    end_tick = dispatch.get("end_next_absolute_tick")
    tick_count = dispatch.get("tick_count")
    if (
        type(rows_shape) is not list
        or len(rows_shape) != 4
        or any(type(dimension) is not int or dimension <= 0 for dimension in rows_shape)
        or rows_shape[-1] != 48
        or rows_shape[2] != 2 * attempt_contract.shape.capacity
        or type(counts_shape) is not list
        or len(counts_shape) != 1
        or any(
            type(dimension) is not int or dimension <= 0 for dimension in counts_shape
        )
        or rows.get("bytes") != math.prod(rows_shape)
        or counts.get("bytes") != 4 * math.prod(counts_shape)
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event receipt shape/byte binding is invalid"
        )

    paths: list[Path] = []
    payloads: list[bytes] = []
    for label, row in (("event rows", rows), ("event counts", counts)):
        file_name = row.get("file_name")
        if (
            type(file_name) is not str
            or Path(file_name).name != file_name
            or file_name in {"", ".", ".."}
        ):
            raise FullEvolutionWorkhorseAuthorityError(f"{label} locator is unsafe")
        path = _regular_file(receipt.parent / file_name, label, read_only=True)
        stored = path.read_bytes()
        if encoded_storage:
            if set(row) != {
                "file_name",
                "bytes",
                "sha256",
                "storage_encoding",
                "stored_bytes",
                "stored_sha256",
            }:
                raise FullEvolutionWorkhorseAuthorityError(
                    f"{label} encoded identity fields are invalid"
                )
            if row.get("stored_bytes") != len(stored) or row.get(
                "stored_sha256"
            ) != _sha_bytes(stored):
                raise FullEvolutionWorkhorseAuthorityError(
                    f"{label} stored-content seal mismatch"
                )
            encoding = row.get("storage_encoding")
            if encoding == "identity_v1":
                payload = stored
            elif encoding == "zstd_level_1_v1":
                if _zstd is None:
                    raise FullEvolutionWorkhorseAuthorityError(
                        f"{label} zstd decoder is unavailable"
                    )
                try:
                    frame = _zstd.get_frame_info(stored)
                    if frame.decompressed_size != row["bytes"]:
                        raise FullEvolutionWorkhorseAuthorityError(
                            f"{label} encoded payload boundary is invalid"
                        )
                    decoder = _zstd.ZstdDecompressor()
                    payload = decoder.decompress(stored, int(row["bytes"]) + 1)
                except (ValueError, _zstd.ZstdError) as exc:
                    raise FullEvolutionWorkhorseAuthorityError(
                        f"{label} encoded payload cannot be decoded"
                    ) from exc
                if (
                    not decoder.eof
                    or decoder.unused_data
                    or len(payload) > row["bytes"]
                ):
                    raise FullEvolutionWorkhorseAuthorityError(
                        f"{label} encoded payload boundary is invalid"
                    )
            elif encoding == "zlib_level_1_v1":
                try:
                    decoder = zlib.decompressobj()
                    payload = decoder.decompress(stored, int(row["bytes"]) + 1)
                    payload += decoder.flush()
                except (ValueError, zlib.error) as exc:
                    raise FullEvolutionWorkhorseAuthorityError(
                        f"{label} encoded payload cannot be decoded"
                    ) from exc
                if (
                    not decoder.eof
                    or decoder.unused_data
                    or decoder.unconsumed_tail
                    or len(payload) > row["bytes"]
                ):
                    raise FullEvolutionWorkhorseAuthorityError(
                        f"{label} encoded payload boundary is invalid"
                    )
            else:
                raise FullEvolutionWorkhorseAuthorityError(
                    f"{label} storage encoding is unsupported"
                )
        else:
            if set(row) != {"file_name", "bytes", "sha256"}:
                raise FullEvolutionWorkhorseAuthorityError(
                    f"{label} exact-content identity fields are invalid"
                )
            payload = stored
        if row.get("bytes") != len(payload) or row.get("sha256") != _sha_bytes(payload):
            raise FullEvolutionWorkhorseAuthorityError(
                f"{label} exact-content seal mismatch"
            )
        paths.append(path)
        payloads.append(payload)
    if value.get(
        "dispatch_content_sha256"
    ) != workhorse_event_dispatch_content_sha256_v2(
        metadata=dispatch, event_rows=payloads[0], event_counts=payloads[1]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event dispatch combined exact-content seal mismatch"
        )
    if (
        type(rows_shape) is not list
        or len(rows_shape) != 4
        or any(type(dimension) is not int or dimension <= 0 for dimension in rows_shape)
        or rows_shape[-1] != 48
        or rows_shape[2] != 2 * attempt_contract.shape.capacity
        or math.prod(rows_shape) != len(payloads[0])
        or type(counts_shape) is not list
        or len(counts_shape) != 1
        or any(
            type(dimension) is not int or dimension <= 0 for dimension in counts_shape
        )
        or math.prod(counts_shape) * 4 != len(payloads[1])
        or type(world_uids) is not list
        or not world_uids
        or any(type(uid) is not int or uid <= 0 for uid in world_uids)
        or len(set(world_uids)) != len(world_uids)
        or rows_shape[0] != len(world_uids)
        or counts_shape[0] != len(world_uids)
        or type(start_tick) is not int
        or type(end_tick) is not int
        or type(tick_count) is not int
        or end_tick <= start_tick
        or tick_count != end_tick - start_tick
        or rows_shape[1] != tick_count
        or type(value.get("sink_id")) is not str
        or not value["sink_id"].strip()
        or value["sink_id"] != value["sink_id"].strip()
        or not all(
            _valid_sha256(dispatch.get(key))
            for key in (
                "artifact_identity_sha256",
                "artifact_sha256",
                "manifest_sha256",
            )
        )
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "event receipt shape/byte/world binding is invalid"
        )
    event_counts = struct.unpack(f"<{counts_shape[0]}i", payloads[1])
    maximum_events_per_world = rows_shape[1] * rows_shape[2]
    if any(count < 0 or count > maximum_events_per_world for count in event_counts):
        raise FullEvolutionWorkhorseAuthorityError(
            "event receipt count exceeds its exact staging capacity"
        )
    return {
        "schema_version": WORKHORSE_EVENT_SINK_RECEIPT_SCHEMA_V2,
        "receipt_path": str(receipt),
        "receipt_file_sha256": file_sha256(receipt),
        "receipt_body_sha256": value["receipt_body_sha256"],
        "dispatch_content_sha256": value["dispatch_content_sha256"],
        "event_rows_file_sha256": rows["sha256"],
        "event_counts_file_sha256": counts["sha256"],
        "world_uids": world_uids,
        "start_next_absolute_tick": dispatch["start_next_absolute_tick"],
        "end_next_absolute_tick": dispatch["end_next_absolute_tick"],
        "tick_count": dispatch["tick_count"],
        "artifact_identity_sha256": dispatch["artifact_identity_sha256"],
        "artifact_sha256": dispatch["artifact_sha256"],
        "manifest_sha256": dispatch["manifest_sha256"],
        "attempt_authority_binding": attempt_binding,
        "attempt_authority_binding_sha256": dispatch[
            "attempt_authority_binding_sha256"
        ],
        "attempt_lineage_sha256": dispatch["attempt_lineage_sha256"],
        "owner_pid": dispatch["owner_pid"],
        "owner_process_start_ticks": dispatch["owner_process_start_ticks"],
        "device_uuid": dispatch["device_uuid"],
        "rng_identity": rng_identity,
        "rng_identity_sha256": dispatch["rng_identity_sha256"],
        "pre_state_sha256": dispatch["pre_state_sha256"],
        "post_state_sha256": dispatch["post_state_sha256"],
        "dispatch_identity_sha256": dispatch["dispatch_identity_sha256"],
    }


def workhorse_checkpoint_metadata_from_state_identity_v2(
    *,
    contract: FullEvolutionCudaWorkhorseContractV2,
    checkpoint_at_tick: int,
    state_nbytes: int,
    state_sha256: str,
    state_headers: Sequence[Sequence[int]],
    artifact_path: str | Path,
    event_sink_receipt_path: str | Path,
    attempt_authority_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build canonical checkpoint metadata from a bounded state identity.

    The synchronous writer and the detached ACS2 writer share this function so
    Attempt, event-ACK, artifact, world/RNG, and checkpoint identity law cannot
    diverge. ``state_headers`` contains only the 32-u64 header of each world;
    callers may hold the canonical state in immutable bytes or a host-only
    capture file.
    """
    if type(contract) is not FullEvolutionCudaWorkhorseContractV2:
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint requires the exact workhorse contract"
        )
    if type(state_nbytes) is not int or state_nbytes <= 0:
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint state byte length is invalid"
        )
    if not _valid_sha256(state_sha256):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint state digest is invalid"
        )
    if (
        not isinstance(state_headers, Sequence)
        or isinstance(state_headers, (str, bytes, bytearray))
        or len(state_headers) != contract.B
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint state headers do not exactly cover B"
        )
    artifact, _, manifest = validate_workhorse_prebuilt_v2(artifact_path)
    expected_nbytes = contract.B * manifest["native_shape"]["state_bytes_per_world"]
    if state_nbytes != expected_nbytes:
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint state byte length does not match exact B/layout"
        )
    try:
        attempt_view = _event_attempt_view_v2(
            attempt_authority_binding,
            historical=False,
        )
        attempt_binding = attempt_view["binding"]
    except Exception as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"checkpoint Attempt authority is invalid: {exc}"
        ) from exc
    if attempt_view["contract"] != contract:
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint contract differs from its Attempt authority"
        )
    if not (
        attempt_view["start_next_tick"]
        <= checkpoint_at_tick
        <= attempt_view["target_next_tick"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint frontier is outside its Attempt absolute-tick span"
        )
    artifact_identity_sha = manifest["identity"]["identity_sha256"]
    sink = validate_workhorse_event_sink_receipt_v2(
        event_sink_receipt_path,
        expected_artifact_identity_sha256=artifact_identity_sha,
        expected_end_next_absolute_tick=checkpoint_at_tick,
        expected_attempt_authority_binding_sha256=attempt_view["binding_sha256"],
        expected_post_state_sha256=state_sha256,
    )
    if (
        sink["artifact_sha256"] != file_sha256(artifact)
        or sink["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint event sink does not bind the exact sealed artifact"
        )
    state_bytes_per_world = manifest["native_shape"]["state_bytes_per_world"]
    world_uids: list[int] = []
    rng_keys: list[int] = []
    for raw_header in state_headers:
        if (
            not isinstance(raw_header, Sequence)
            or isinstance(raw_header, (str, bytes, bytearray))
            or len(raw_header) != 32
            or any(type(value) is not int for value in raw_header)
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "checkpoint state header is malformed"
            )
        header = tuple(int(value) for value in raw_header)
        if (
            header[1] != checkpoint_at_tick
            or header[4] != 0
            or header[14] != 4
            or header[0] <= 0
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "checkpoint header frontier/status/RNG identity is invalid"
            )
        world_uids.append(header[0])
        rng_keys.append(header[13])
    if len(set(world_uids)) != contract.B or world_uids != sink["world_uids"]:
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint world UIDs do not match acknowledged events"
        )
    expected_rng_identity = {
        "schema_version": "eiel_workhorse_rng_identity_v2",
        "rng_epoch": 4,
        "world_uids": world_uids,
        "rng_keys": rng_keys,
    }
    if sink["rng_identity"] != expected_rng_identity or sink[
        "rng_identity_sha256"
    ] != canonical_sha256(expected_rng_identity):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint RNG vector differs from its acknowledged dispatch"
        )
    from .layout import workhorse_checkpoint_identity_v2, workhorse_layout_v2

    checkpoint_identity = workhorse_checkpoint_identity_v2(
        contract,
        checkpoint_at_tick=checkpoint_at_tick,
        artifact_identity_sha256=artifact_identity_sha,
        state_content_sha256=state_sha256,
        event_sink_receipt_path=str(event_sink_receipt_path),
        attempt_authority_binding=attempt_binding,
    )
    body = {
        "schema_version": WORKHORSE_CHECKPOINT_BUNDLE_SCHEMA_V2,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "contract": contract.to_dict(),
        "checkpoint_identity": checkpoint_identity,
        "checkpoint_at_tick": checkpoint_at_tick,
        "canonical_state": {
            "file_name": "canonical_state.bin",
            "bytes": state_nbytes,
            "sha256": state_sha256,
            "B": contract.B,
            "state_bytes_per_world": state_bytes_per_world,
            "layout_sha256": workhorse_layout_v2(contract.shape).layout_sha256,
        },
        "world_uids": world_uids,
        "rng_keys": rng_keys,
        "rng_epoch": 4,
        "attempt_authority_binding": attempt_binding,
        "attempt_authority_binding_sha256": attempt_view["binding_sha256"],
        "attempt_lineage_sha256": attempt_view["lineage_sha256"],
        "dispatch_identity_sha256": sink["dispatch_identity_sha256"],
        "artifact": {
            "path": str(artifact),
            "sha256": file_sha256(artifact),
            "manifest_sha256": manifest["manifest_sha256"],
            "identity_sha256": artifact_identity_sha,
            "sealed_artifact_identity_sha256": manifest["sealed_artifact_identity"][
                "sealed_artifact_identity_sha256"
            ],
        },
        "event_sink_ack": sink,
        "governing_contracts": workhorse_governing_contract_bindings_v2(),
        "checkpoint_host_staging_slabs": 2,
        "checkpoint_device_staging_bytes": 0,
    }
    return {**body, "bundle_body_sha256": canonical_sha256(body)}


def write_workhorse_checkpoint_bundle_v2(
    output_directory: str | Path,
    *,
    contract: FullEvolutionCudaWorkhorseContractV2,
    checkpoint_at_tick: int,
    state_bytes: bytes,
    artifact_path: str | Path,
    event_sink_receipt_path: str | Path,
    attempt_authority_binding: Mapping[str, Any],
) -> Path:
    """Write an immutable state/metadata bundle eligible for fresh reopen."""
    from .layout import workhorse_layout_v2

    if type(state_bytes) is not bytes:
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint state must be immutable host bytes"
        )
    state_sha = _sha_bytes(state_bytes)
    state_bytes_per_world = workhorse_layout_v2(
        contract.shape
    ).state_bytes_per_world
    state_headers = tuple(
        struct.unpack_from("<32Q", state_bytes, world * state_bytes_per_world)
        for world in range(contract.B)
    )
    metadata = workhorse_checkpoint_metadata_from_state_identity_v2(
        contract=contract,
        checkpoint_at_tick=checkpoint_at_tick,
        state_nbytes=len(state_bytes),
        state_sha256=state_sha,
        state_headers=state_headers,
        artifact_path=artifact_path,
        event_sink_receipt_path=event_sink_receipt_path,
        attempt_authority_binding=attempt_authority_binding,
    )
    output = Path(output_directory)
    if output.is_symlink():
        raise FullEvolutionWorkhorseAuthorityError("checkpoint root is a symlink")
    try:
        output.mkdir(parents=True)
    except FileExistsError as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint output must be a fresh directory"
        ) from exc
    state_path = output / "canonical_state.bin"
    with state_path.open("xb") as handle:
        handle.write(state_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    state_path.chmod(0o444)
    # The second bounded host slab is this independent readback, never a device
    # clone.  It is compared and then released before publication.
    verified_state = state_path.read_bytes()
    if verified_state != state_bytes:
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint state changed during durable readback"
        )
    metadata_path = output / "checkpoint.json"
    with metadata_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    metadata_path.chmod(0o444)
    _fsync_directory(output)
    output.chmod(0o555)
    _fsync_directory(output.parent)
    reopen_workhorse_checkpoint_bundle_v2(metadata_path)
    return metadata_path


def inspect_workhorse_checkpoint_bundle_v2(
    metadata_path: str | Path,
    *,
    require_current_source_match: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Validate a checkpoint by streaming its state; do not materialize full B."""
    path, value = _read_json(metadata_path, "workhorse checkpoint bundle")
    body = {key: item for key, item in value.items() if key != "bundle_body_sha256"}
    if (
        value.get("schema_version") != WORKHORSE_CHECKPOINT_BUNDLE_SCHEMA_V2
        or value.get("time_authority") != WORKHORSE_TIME_AUTHORITY_V2
        or value.get("bundle_body_sha256") != canonical_sha256(body)
        or path.name != "checkpoint.json"
        or path.parent.stat().st_mode & 0o222
        or value.get("checkpoint_host_staging_slabs") != 2
        or value.get("checkpoint_device_staging_bytes") != 0
        or value.get("rng_epoch") != 4
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint bundle schema or seal is invalid"
        )
    try:
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(value["contract"])
    except (KeyError, FullEvolutionWorkhorseContractError) as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"checkpoint contract is invalid: {exc}"
        ) from exc
    artifact_row = _require_exact_dict(value.get("artifact"), "checkpoint artifact")
    artifact, _, manifest = validate_workhorse_prebuilt_v2(
        artifact_row.get("path"),
        require_current_source_match=require_current_source_match,
    )
    if (
        manifest["shape_id"] != contract.shape_id
        or manifest["backend_id"] != contract.backend_id
        or artifact_row.get("sha256") != file_sha256(artifact)
        or artifact_row.get("manifest_sha256") != manifest["manifest_sha256"]
        or artifact_row.get("identity_sha256")
        != manifest["identity"]["identity_sha256"]
        or artifact_row.get("sealed_artifact_identity_sha256")
        != manifest["sealed_artifact_identity"]["sealed_artifact_identity_sha256"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint sealed artifact identity is invalid"
        )
    if value.get("governing_contracts") != workhorse_governing_contract_bindings_v2():
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint governing contract binding changed"
        )
    state_row = _require_exact_dict(
        value.get("canonical_state"), "checkpoint state identity"
    )
    file_name = state_row.get("file_name")
    if type(file_name) is not str or Path(file_name).name != file_name:
        raise FullEvolutionWorkhorseAuthorityError("checkpoint state locator is unsafe")
    state_path = _regular_file(
        path.parent / file_name, "checkpoint state", read_only=True
    )
    state_nbytes = int(state_path.stat().st_size)
    state_sha256 = file_sha256(state_path)
    if (
        state_row.get("bytes") != state_nbytes
        or state_row.get("sha256") != state_sha256
        or state_row.get("B") != contract.B
        or state_row.get("state_bytes_per_world")
        != manifest["native_shape"]["state_bytes_per_world"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint canonical state exact-content seal is invalid"
        )
    sink_row = _require_exact_dict(value.get("event_sink_ack"), "checkpoint sink ack")
    sink = validate_workhorse_event_sink_receipt_v2(
        sink_row.get("receipt_path"),
        expected_artifact_identity_sha256=manifest["identity"]["identity_sha256"],
        expected_end_next_absolute_tick=value.get("checkpoint_at_tick"),
        expected_attempt_authority_binding_sha256=value.get(
            "attempt_authority_binding_sha256"
        ),
        expected_post_state_sha256=state_row.get("sha256"),
    )
    if sink != sink_row:
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint sink acknowledgement no longer reopens exactly"
        )
    from .layout import workhorse_checkpoint_identity_v2, workhorse_layout_v2

    expected_checkpoint_identity = workhorse_checkpoint_identity_v2(
        contract,
        checkpoint_at_tick=value.get("checkpoint_at_tick"),
        artifact_identity_sha256=manifest["identity"]["identity_sha256"],
        state_content_sha256=state_row["sha256"],
        event_sink_receipt_path=sink["receipt_path"],
        attempt_authority_binding=value.get("attempt_authority_binding"),
    )
    if (
        value.get("checkpoint_identity") != expected_checkpoint_identity
        or value.get("attempt_authority_binding") != sink["attempt_authority_binding"]
        or value.get("attempt_authority_binding_sha256")
        != sink["attempt_authority_binding_sha256"]
        or value.get("attempt_lineage_sha256") != sink["attempt_lineage_sha256"]
        or value.get("dispatch_identity_sha256") != sink["dispatch_identity_sha256"]
        or state_row.get("layout_sha256")
        != workhorse_layout_v2(contract.shape).layout_sha256
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint identity/layout does not recompute"
        )
    state_bytes_per_world = state_row["state_bytes_per_world"]
    observed_uids = []
    observed_keys = []
    with state_path.open("rb") as state_handle:
        for world in range(contract.B):
            state_handle.seek(world * state_bytes_per_world)
            raw_header = state_handle.read(32 * 8)
            if len(raw_header) != 32 * 8:
                raise FullEvolutionWorkhorseAuthorityError(
                    "checkpoint state contains a truncated world header"
                )
            header = struct.unpack("<32Q", raw_header)
            if (
                header[1] != value.get("checkpoint_at_tick")
                or header[4] != 0
                or header[14] != 4
            ):
                raise FullEvolutionWorkhorseAuthorityError(
                    "checkpoint state contains an uncommitted/error frontier"
                )
            observed_uids.append(header[0])
            observed_keys.append(header[13])
    if (
        any(uid <= 0 for uid in observed_uids)
        or len(set(observed_uids)) != contract.B
        or observed_uids != value.get("world_uids")
        or observed_keys != value.get("rng_keys")
        or sink.get("rng_identity")
        != {
            "schema_version": "eiel_workhorse_rng_identity_v2",
            "rng_epoch": 4,
            "world_uids": observed_uids,
            "rng_keys": observed_keys,
        }
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint state world/RNG identity changed"
        )
    return value, state_path


def reopen_workhorse_checkpoint_bundle_v2(
    metadata_path: str | Path,
    *,
    require_current_source_match: bool = True,
) -> tuple[dict[str, Any], bytes]:
    """Reopen all checkpoint sources and materialize the validated complete set."""

    value, state_path = inspect_workhorse_checkpoint_bundle_v2(
        metadata_path,
        require_current_source_match=require_current_source_match,
    )
    state_bytes = state_path.read_bytes()
    state_row = value["canonical_state"]
    if (
        len(state_bytes) != state_row["bytes"]
        or _sha_bytes(state_bytes) != state_row["sha256"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "checkpoint state changed after streaming validation"
        )
    return value, state_bytes


_AGGREGATE_SEMANTIC_GATE_AUTHORITY_V2: Final[str] = (
    "reopened_24_case_aggregate_derivation_v2"
)
_OBSERVATION_OFF_V2: Final[dict[str, Any]] = {
    "mode": "OFF",
    "science_records_emitted": 0,
    "science_observation_schedule": None,
    "population_change_detection": False,
}


def _semantic_artifact_projection_v2(
    artifact_path: str | Path,
    *,
    require_current_source_match: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact, _, manifest = validate_workhorse_prebuilt_v2(
        artifact_path,
        require_current_source_match=require_current_source_match,
    )
    projection = {
        "artifact": str(artifact),
        "artifact_sha256": file_sha256(artifact),
        "manifest_sha256": manifest["manifest_sha256"],
        "identity": manifest["identity"],
        "build_identity": manifest["build_identity"],
        "sealed_artifact_identity": manifest["sealed_artifact_identity"],
    }
    return projection, manifest


def _validate_semantic_artifact_projection_v2(
    value: object,
    *,
    label: str,
    require_current_source_match: bool = True,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    row = _require_exact_dict(value, f"semantic {label}")
    try:
        artifact, _, manifest = validate_workhorse_prebuilt_v2(
            row.get("artifact"),
            require_current_source_match=require_current_source_match,
        )
    except Exception as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"cannot reopen semantic {label} artifact: {exc}"
        ) from exc
    expected, _ = _semantic_artifact_projection_v2(
        artifact,
        require_current_source_match=require_current_source_match,
    )
    if row != expected:
        raise FullEvolutionWorkhorseAuthorityError(
            f"semantic {label} source binding is invalid"
        )
    return row, artifact, manifest


def _semantic_case_aggregate_projection_v2(
    aggregate_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(aggregate_path).expanduser().resolve()
    aggregate = reopen_workhorse_semantic_case_aggregate_v2(path)
    projection = {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "aggregate_content_sha256": aggregate["aggregate_content_sha256"],
        "case_count": aggregate["case_count"],
        "case_table_sha256": aggregate["case_table_sha256"],
    }
    return projection, aggregate


def _derive_aggregate_semantic_support_v2(
    aggregate: Mapping[str, Any],
    *,
    reference_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
    """Derive non-dataset gate checks from the 24 terminalized cases."""

    table = workhorse_semantic_case_table_v2(aggregate["semantic_temporal_k"])
    receipt_rows = aggregate["terminal_receipts"]
    receipts = {
        row["case_id"]: reopen_terminalized_workhorse_semantic_case_receipt_v2(
            row["receipt_path"]
        )
        for row in receipt_rows
    }
    reference_sha = reference_manifest["output"]["sha256"]
    candidate_sha = candidate_manifest["output"]["sha256"]
    expected_results = {
        case.case_id: (
            "SUCCESS"
            if case.expected_attempt_terminal_status == "COMPLETED"
            else "VALID_NEGATIVE"
        )
        for case in table
    }
    device_identities = [
        receipts[case.case_id]["canonical_parent_binding"]["device_identity"]
        for case in table
    ]
    device = device_identities[0]
    artifacts_exact = all(
        receipts[case.case_id]["artifact_sha256"]
        == (reference_sha if case.artifact_role == "REFERENCE" else candidate_sha)
        for case in table
    )
    results_exact = all(
        receipts[case.case_id]["result_class"] == expected_results[case.case_id]
        for case in table
    )
    device_exact = all(item == device for item in device_identities)

    checkpoint_receipt = receipts["checkpoint_same_restore_sequence"]
    checkpoint_fragment, checkpoint_datasets = (
        reopen_workhorse_semantic_case_fragment_v2(
            checkpoint_receipt["case_fragment_path"]
        )
    )
    base = _require_exact_dict(
        checkpoint_fragment.get("base_checkpoint"), "aggregate base checkpoint"
    )
    base_identity = _require_exact_dict(
        base.get("identity"), "aggregate base checkpoint identity"
    )
    base_state_path = Path(checkpoint_receipt["case_fragment_path"]).parent / str(
        base.get("state_file_name")
    )
    base_state = base_state_path.read_bytes()
    acknowledgement = _require_exact_dict(
        base_identity.get("last_acknowledged_event_frontier"),
        "aggregate checkpoint event acknowledgement",
    )
    reopened_acknowledgement = validate_workhorse_event_sink_receipt_v2(
        acknowledgement.get("receipt_path"),
        expected_artifact_identity_sha256=base_identity.get(
            "source_artifact_identity_sha256"
        ),
        expected_end_next_absolute_tick=base_identity.get("checkpoint_at_tick"),
        expected_post_state_sha256=base_identity.get("state_sha256"),
    )
    acknowledgement_exact = all(
        reopened_acknowledgement.get(key) == item
        for key, item in acknowledgement.items()
    )
    produced_source = _require_exact_dict(
        checkpoint_receipt.get("produced_external_source_checkpoint_binding"),
        "aggregate produced checkpoint source",
    )

    fresh_receipt = receipts["fresh_restore_continuation"]
    fresh_fragment, _ = reopen_workhorse_semantic_case_fragment_v2(
        fresh_receipt["case_fragment_path"]
    )
    fresh_parent = fresh_receipt["canonical_parent_binding"]
    fresh_source_exact = (
        fresh_fragment["external_source_checkpoint_binding"] == produced_source
        and fresh_parent["source_checkpoint_identity_sha256"]
        == base_identity["base_checkpoint_identity_sha256"]
        and fresh_fragment["producer_pid"] != checkpoint_fragment["producer_pid"]
    )

    evidence_manifest, evidence_datasets = reopen_workhorse_semantic_evidence_bundle_v2(
        aggregate["semantic_evidence"]["manifest_path"]
    )
    checkpoint_state_exact = (
        base_state == checkpoint_datasets["restore_checkpoint"]["state"]
        and base_state == evidence_datasets["restore_checkpoint"]["state"]
        and len(base_state) == base_identity["state_bytes"]
        and _sha_bytes(base_state) == base_identity["state_sha256"]
    )

    p2_receipt = receipts["event_reuse_canonicalization_regression"]
    p2_fragment, _ = reopen_workhorse_semantic_case_fragment_v2(
        p2_receipt["case_fragment_path"]
    )
    p2 = _require_exact_dict(p2_fragment.get("p2_witness"), "aggregate P2 witness")
    p2_exact = (
        all(
            p2.get(field) is expected
            for field, expected in {
                "failed_event_acknowledged": False,
                "failed_dispatch_pending_preserved": True,
                "post_failure_reuse_rejected": True,
                "failed_event_buffer_cannot_be_reused": True,
                "active_rows_byte_preserved": True,
                "same_owner_storage_data_ptr": True,
                "successful_ack_preceded_failed_reuse": True,
                "inactive_rows_exact_zero_after_canonicalization": True,
                "later_k_slices_exact_zero_after_canonicalization": True,
            }.items()
        )
        and p2.get("failed_event_publication_path") is None
    )

    runtime_exact = (
        artifacts_exact
        and results_exact
        and device_exact
        and candidate_manifest["shape_id"] == aggregate["shape_id"]
        and reference_manifest["shape_id"] == aggregate["shape_id"]
        and reference_manifest["backend_id"] == "reference_phased_v2"
        and candidate_manifest["backend_id"]
        in {"phase_graph_v2", "fused_cta_temporal_v2"}
        and candidate_manifest["target_sm"] == device.get("sm")
        and candidate_manifest["identity"]["profile_id"]
        == receipts["parity_candidate_k_dispatch"]["canonical_parent_binding"][
            "profile_id"
        ]
        and candidate_manifest["identity"]["route_id"]
        == receipts["parity_candidate_k_dispatch"]["canonical_parent_binding"][
            "route_id"
        ]
    )
    runtime = {
        "runtime_inspected": runtime_exact,
        "inspection_authority": "24_CANONICAL_TERMINALIZED_PHYSICAL_CASES_V2",
        "shape_id": aggregate["shape_id"],
        "backend_id": candidate_manifest["backend_id"],
        "profile_id": candidate_manifest["identity"]["profile_id"],
        "route_id": candidate_manifest["identity"]["route_id"],
        "device_uuid": device["uuid"],
        "target_sm": device["sm"],
        "multiprocessor_count": device["multiprocessor_count"],
        "total_device_bytes": device["total_memory_bytes"],
        "case_count": len(receipts),
        "distinct_attempt_count": len(
            {
                receipt["canonical_parent_binding"]["attempt_id"]
                for receipt in receipts.values()
            }
        ),
    }
    supporting = {
        "checkpoint_binds_external_exact_content_sink_ack": acknowledgement_exact,
        "sealed_checkpoint_reopens_exact_state_bytes": checkpoint_state_exact,
        "fresh_process_restore_continuation_exact": fresh_source_exact,
        "failed_event_buffer_cannot_be_reused": p2_exact,
        "native_runtime_inspection_present": runtime_exact,
    }
    witness = {
        "aggregate_content_sha256": aggregate["aggregate_content_sha256"],
        "evidence_bundle_body_sha256": evidence_manifest["bundle_body_sha256"],
        "terminal_receipt_count": len(receipts),
        "success_count": sum(value == "SUCCESS" for value in expected_results.values()),
        "valid_negative_count": sum(
            value == "VALID_NEGATIVE" for value in expected_results.values()
        ),
        "device_identity": device,
        "base_checkpoint_identity_sha256": base_identity[
            "base_checkpoint_identity_sha256"
        ],
        "checkpoint_occurrence_identity_sha256": produced_source[
            "checkpoint_occurrence_identity_sha256"
        ],
        "fresh_process_producer_pid": fresh_fragment["producer_pid"],
        "checkpoint_producer_pid": checkpoint_fragment["producer_pid"],
        "p2_witness_sha256": p2["witness_sha256"],
    }
    return supporting, runtime, witness


def write_workhorse_semantic_case_aggregate_gate_receipt_v2(
    output_path: str | Path,
    *,
    aggregate_path: str | Path,
    reference_artifact_path: str | Path,
    candidate_artifact_path: str | Path,
) -> Path:
    """Seal a semantic gate derived only from the 24 terminalized case aggregate."""

    reference, reference_manifest = _semantic_artifact_projection_v2(
        reference_artifact_path
    )
    candidate, candidate_manifest = _semantic_artifact_projection_v2(
        candidate_artifact_path
    )
    aggregate_projection, aggregate = _semantic_case_aggregate_projection_v2(
        aggregate_path
    )
    evidence_projection = {
        key: aggregate["semantic_evidence"][key]
        for key in ("manifest_path", "manifest_file_sha256", "bundle_body_sha256")
    }
    evidence_checks, evidence_witness = derive_workhorse_semantic_checks_v2(
        Path(evidence_projection["manifest_path"])
    )
    supporting, runtime, supporting_witness = _derive_aggregate_semantic_support_v2(
        aggregate,
        reference_manifest=reference_manifest,
        candidate_manifest=candidate_manifest,
    )
    checks = dict(evidence_checks)
    for name, passed in supporting.items():
        checks[name] = checks.get(name, True) and passed
    if (
        set(checks) != WORKHORSE_REQUIRED_SEMANTIC_CHECKS_V2
        or aggregate["shape_id"] != candidate_manifest["shape_id"]
        or aggregate["shape_id"] != reference_manifest["shape_id"]
        or aggregate["semantic_evidence"]["dataset_count"] != 29
        or reference["artifact_sha256"]
        != reopen_workhorse_semantic_evidence_bundle_v2(
            evidence_projection["manifest_path"]
        )[0]["reference_artifact_sha256"]
        or candidate["artifact_sha256"]
        != reopen_workhorse_semantic_evidence_bundle_v2(
            evidence_projection["manifest_path"]
        )[0]["candidate_artifact_sha256"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "24-case aggregate does not bind the exact semantic artifacts/checks"
        )
    body = {
        "schema_version": WORKHORSE_SEMANTIC_GATE_SCHEMA_V2,
        "claim_class": "TECHNICAL_ONLY",
        "qualification_authority_granted": False,
        "science_authority_granted": False,
        "observation": dict(_OBSERVATION_OFF_V2),
        "shape_id": aggregate["shape_id"],
        "semantic_transition_sha256": candidate_manifest["identity"][
            "semantic_transition_sha256"
        ],
        "reference": reference,
        "candidate": candidate,
        "semantic_case_aggregate": aggregate_projection,
        "semantic_evidence": evidence_projection,
        "runtime": runtime,
        "temporal_k": aggregate["semantic_temporal_k"],
        "checks": checks,
        "derived_evidence_witness": evidence_witness,
        "supporting_evidence_witness": supporting_witness,
        "decision": ("SEMANTIC_GATE_SATISFIED" if all(checks.values()) else "REJECTED"),
        "decision_authority": _AGGREGATE_SEMANTIC_GATE_AUTHORITY_V2,
        "governing_contracts": workhorse_governing_contract_bindings_v2(),
    }
    value = {**body, "semantic_gate_receipt_sha256": canonical_sha256(body)}
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic receipt output must be fresh"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    output.chmod(0o444)
    if value["decision"] == "SEMANTIC_GATE_SATISFIED":
        validate_workhorse_semantic_gate_receipt_v2(
            output,
            expected_shape_id=aggregate["shape_id"],
            expected_candidate_artifact_sha256=candidate["artifact_sha256"],
            expected_candidate_backend_id=candidate_manifest["backend_id"],
        )
    return output


def write_workhorse_semantic_gate_receipt_v2(
    receipt: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Seal a just-measured semantic gate without accepting pre-sealed input."""
    value = dict(receipt)
    if "semantic_gate_receipt_sha256" in value:
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic receipt input must not be pre-sealed"
        )
    if "checks" in value or "decision" in value:
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic receipt writer does not accept caller verdicts or checks"
        )
    value["governing_contracts"] = workhorse_governing_contract_bindings_v2()
    evidence = _require_exact_dict(
        value.get("semantic_evidence"), "raw semantic evidence projection"
    )
    evidence_checks, evidence_witness = derive_workhorse_semantic_checks_v2(
        evidence.get("manifest_path")
    )
    _, _, candidate_manifest = validate_workhorse_prebuilt_v2(
        _require_exact_dict(value.get("candidate"), "semantic candidate").get(
            "artifact"
        )
    )
    supporting_checks = _validate_semantic_supporting_evidence_v2(
        value, candidate_manifest=candidate_manifest
    )
    checks = dict(evidence_checks)
    for name, passed in supporting_checks.items():
        checks[name] = checks.get(name, True) and passed
    if set(checks) != WORKHORSE_REQUIRED_SEMANTIC_CHECKS_V2:
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic derivation does not cover every required check"
        )
    value["checks"] = checks
    value["derived_evidence_witness"] = evidence_witness
    value["decision"] = (
        "SEMANTIC_GATE_SATISFIED" if all(checks.values()) else "REJECTED"
    )
    value["decision_authority"] = "reopened_raw_state_scratch_event_derivation_v2"
    value["semantic_gate_receipt_sha256"] = canonical_sha256(value)
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic receipt output must be fresh"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    output.chmod(0o444)
    if value.get("decision") == "SEMANTIC_GATE_SATISFIED":
        candidate = _require_exact_dict(value.get("candidate"), "semantic candidate")
        validate_workhorse_semantic_gate_receipt_v2(
            output,
            expected_shape_id=value.get("shape_id"),
            expected_candidate_artifact_sha256=candidate.get("artifact_sha256"),
            expected_candidate_backend_id=_require_exact_dict(
                candidate.get("identity"), "semantic candidate identity"
            ).get("backend_id"),
        )
    return output


def _reopen_fresh_worker_raw_continuation_v2(
    worker_path: Path,
    worker: Mapping[str, Any],
    *,
    expected_dataset: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen every fresh-worker transition input/output byte independently."""
    raw = _require_exact_dict(
        worker.get("raw_continuation"), "fresh-process raw continuation"
    )
    if set(raw) != {
        "B",
        "temporal_k",
        "start_next_absolute_tick",
        "components",
    }:
        raise FullEvolutionWorkhorseAuthorityError(
            "fresh-process raw continuation metadata is inexact"
        )
    components = _require_exact_dict(
        raw.get("components"), "fresh-process raw component inventory"
    )
    if (
        set(components) != set(WORKHORSE_FRESH_PROCESS_RAW_COMPONENTS_V2)
        or raw.get("B") != expected_dataset.get("B")
        or raw.get("temporal_k") != expected_dataset.get("temporal_k")
        or raw.get("start_next_absolute_tick")
        != expected_dataset.get("start_next_absolute_tick")
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "fresh-process raw coordinates/inventory changed"
        )
    reopened: dict[str, Any] = {
        "B": raw["B"],
        "temporal_k": raw["temporal_k"],
        "start_next_absolute_tick": raw["start_next_absolute_tick"],
    }
    for component in WORKHORSE_FRESH_PROCESS_RAW_COMPONENTS_V2:
        row = _require_exact_dict(
            components[component], f"fresh-process raw {component} identity"
        )
        expected_path = worker_path.with_name(
            f"{worker_path.stem}.{component}.bin"
        ).resolve()
        if set(row) != {"path", "bytes", "sha256"} or row.get("path") != str(
            expected_path
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                f"fresh-process raw {component} locator/identity is inexact"
            )
        path = _regular_file(
            row["path"], f"fresh-process raw {component}", read_only=True
        )
        if path != expected_path or path.parent != worker_path.parent:
            raise FullEvolutionWorkhorseAuthorityError(
                f"fresh-process raw {component} escaped its receipt directory"
            )
        payload = path.read_bytes()
        if (
            row.get("bytes") != len(payload)
            or row.get("sha256") != _sha_bytes(payload)
            or payload != expected_dataset.get(component)
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                f"fresh-process raw {component} exact content changed"
            )
        reopened[component] = payload
    return reopened


def _validate_semantic_attempt_lifecycles_v2(
    value: Mapping[str, Any],
    *,
    runtime: Mapping[str, Any],
    source_checkpoint_identity_sha256: str,
) -> None:
    """Require every tiny physical execution to retain its exact closeout."""
    expected_roles = {
        "parity_candidate",
        "parity_reference",
        "b_whole",
        "b_split_0",
        "b_split_1",
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
        "wdl_ticks_1_2_3",
        "restore_checkpoint_dispatch",
        "restore_reconstruction",
        "restore_same_process_continuation",
        "restore_clean_continuation",
        "restore_fresh_process_continuation",
        "restore_invalid_fail_stop",
        "overflow_reference",
        "overflow_candidate",
        "middle_survivor_reference",
        "middle_survivor_candidate",
        "transaction_fail_stop",
    }
    rows = value.get("attempt_lifecycles")
    if type(rows) is not list or len(rows) != len(expected_roles):
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic Attempt lifecycle inventory is incomplete"
        )
    reference_sha = _require_exact_dict(
        value.get("reference"), "semantic reference"
    ).get("artifact_sha256")
    candidate_sha = _require_exact_dict(
        value.get("candidate"), "semantic candidate"
    ).get("artifact_sha256")
    observed_roles: set[str] = set()
    execution_bindings: set[str] = set()
    closeout_bindings: set[str] = set()
    checkpoint_bound_roles = {
        "restore_reconstruction",
        "restore_same_process_continuation",
        "restore_clean_continuation",
        "restore_fresh_process_continuation",
        "restore_invalid_fail_stop",
    }
    common_request_fields = (
        "contract",
        "profile_id",
        "route_id",
        "backend_id",
        "B",
        "temporal_k",
        "artifact_sha256",
        "sealed_artifact_identity_sha256",
        "device_uuid",
        "owner_pid",
        "owner_process_start_ticks",
        "candidate_envelope_sha256",
        "governing_contracts_sha256",
    )
    for row in rows:
        if type(row) is not dict or set(row) != {
            "schema_version",
            "role",
            "execution_attempt_authority",
            "closeout_attempt_authority",
            "attempt_lifecycle_sha256",
        }:
            raise FullEvolutionWorkhorseAuthorityError(
                "semantic Attempt lifecycle row is inexact"
            )
        body = {
            key: item for key, item in row.items() if key != "attempt_lifecycle_sha256"
        }
        role = row.get("role")
        if (
            row.get("schema_version") != "eiel_workhorse_attempt_lifecycle_v2"
            or role not in expected_roles
            or role in observed_roles
            or row.get("attempt_lifecycle_sha256") != canonical_sha256(body)
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "semantic Attempt lifecycle role or seal is invalid"
            )
        try:
            execution = validate_workhorse_attempt_authority_binding_v2(
                row["execution_attempt_authority"], historical=True
            )
            closeout = validate_workhorse_attempt_authority_binding_v2(
                row["closeout_attempt_authority"], historical=True
            )
        except Exception as exc:
            raise FullEvolutionWorkhorseAuthorityError(
                f"semantic Attempt lifecycle does not reopen: {exc}"
            ) from exc
        execution_request = execution["request"]
        closeout_request = closeout["request"]
        expected_artifact = (
            reference_sha
            if role
            in {
                "parity_reference",
                "overflow_reference",
                "middle_survivor_reference",
            }
            else candidate_sha
        )
        if (
            execution_request.get("phase") != "EXECUTION_ROW"
            or closeout_request.get("phase") != "CLOSEOUT"
            or execution["attempt_lineage_sha256"] != closeout["attempt_lineage_sha256"]
            or closeout_request.get("prior_lineage_sha256")
            != execution["attempt_lineage_sha256"]
            or any(
                closeout_request.get(field) != execution_request.get(field)
                for field in common_request_fields
            )
            or closeout_request.get("source_checkpoint_identity_sha256") is not None
            or execution_request.get("source_checkpoint_identity_sha256")
            != (
                source_checkpoint_identity_sha256
                if role in checkpoint_bound_roles
                else None
            )
            or execution_request.get("artifact_sha256") != expected_artifact
            or execution_request.get("device_uuid") != runtime.get("device_uuid")
            or execution["response"].get("device_total_bytes")
            != runtime.get("total_device_bytes")
            or closeout["response"].get("device_total_bytes")
            != runtime.get("total_device_bytes")
            or closeout["response"].get("owner_pid")
            != execution["response"].get("owner_pid")
            or closeout["response"].get("owner_process_start_ticks")
            != execution["response"].get("owner_process_start_ticks")
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "semantic execution/closeout lineage or owner was substituted"
            )
        observed_roles.add(role)
        execution_bindings.add(execution["authority_binding_sha256"])
        closeout_bindings.add(closeout["authority_binding_sha256"])
    if (
        observed_roles != expected_roles
        or len(execution_bindings) != len(expected_roles)
        or len(closeout_bindings) != len(expected_roles)
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic Attempt lifecycles were reused or omitted"
        )


def _validate_semantic_supporting_evidence_v2(
    value: Mapping[str, Any],
    *,
    candidate_manifest: Mapping[str, Any],
) -> dict[str, bool]:
    """Reopen supporting files and derive the repaired authority checks."""
    runtime = _require_exact_dict(value.get("runtime"), "semantic runtime")
    if (
        runtime.get("runtime_inspected") is not True
        or runtime.get("shape_id", candidate_manifest["shape_id"])
        != candidate_manifest["shape_id"]
        or runtime.get("backend_id") != candidate_manifest["backend_id"]
        or runtime.get("profile_id") != candidate_manifest["identity"]["profile_id"]
        or runtime.get("route_id") != candidate_manifest["identity"]["route_id"]
        or type(runtime.get("device_uuid")) is not str
        or not runtime["device_uuid"]
        or type(runtime.get("target_sm")) is not str
        or not runtime["target_sm"]
        or type(runtime.get("multiprocessor_count")) is not int
        or runtime["multiprocessor_count"] <= 0
        or type(runtime.get("total_device_bytes")) is not int
        or runtime["total_device_bytes"] <= 0
        or type(runtime.get("compiled_kernel_resources")) is not list
        or not runtime["compiled_kernel_resources"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic runtime inspection is not bound to the candidate"
        )
    temporal_k = value.get("temporal_k")
    if type(temporal_k) is not int or temporal_k <= 1 or value.get("B") != 2:
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic gate did not use the exact tiny B2/K>1 audit"
        )
    graph = value.get("graph_inspection")
    if candidate_manifest["backend_id"] == "phase_graph_v2":
        graph = _require_exact_dict(graph, "semantic graph inspection")
        expected_kernel_nodes = (
            len(candidate_manifest["identity"]["backend_topology"]["kernels_per_tick"])
            * temporal_k
        )
        if (
            graph.get("runtime_inspected") is not True
            or graph.get("shape_id") != candidate_manifest["shape_id"]
            or graph.get("backend_id") != candidate_manifest["backend_id"]
            or graph.get("B") != 2
            or graph.get("temporal_k") != temporal_k
            or graph.get("linear_chain") is not True
            or graph.get("kernel_sequence_exact") is not True
            or graph.get("kernel_geometry_exact") is not True
            or graph.get("kernel_node_count") != expected_kernel_nodes
            or graph.get("memset_node_count") != 1
            or graph.get("node_count") != expected_kernel_nodes + 1
            or graph.get("edge_count") != expected_kernel_nodes
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                "semantic graph evidence does not recompute"
            )
    elif graph is not None:
        raise FullEvolutionWorkhorseAuthorityError(
            "fused semantic receipt must not self-attest a CUDA graph"
        )

    witnesses = _require_exact_dict(value.get("witnesses"), "semantic witnesses")
    restore = _require_exact_dict(witnesses.get("restore"), "restore witness")
    checkpoint_path = restore.get("checkpoint_bundle_receipt_path")
    checkpoint, _ = reopen_workhorse_checkpoint_bundle_v2(checkpoint_path)
    checkpoint_value, checkpoint_state = reopen_workhorse_checkpoint_bundle_v2(
        checkpoint_path
    )
    if (
        restore.get("checkpoint_bundle_receipt_file_sha256")
        != file_sha256(checkpoint_path)
        or restore.get("checkpoint_bundle") != checkpoint
        or checkpoint["artifact"]["sha256"] != candidate_manifest["output"]["sha256"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic checkpoint witness does not reopen exactly"
        )
    worker_path, worker = _read_json(
        restore.get("fresh_process_receipt_path"),
        "fresh-process restore receipt",
    )
    worker_body = {
        key: item for key, item in worker.items() if key != "worker_receipt_sha256"
    }
    expected_worker_fields = {
        "schema_version",
        "source_pid",
        "checkpoint_bundle_sha256",
        "artifact_sha256",
        "state_sha256",
        "scratch_sha256",
        "events_sha256",
        "counts_sha256",
        "event_sink_ack",
        "attempt_authority_binding",
        "closeout_attempt_authority",
        "raw_continuation",
        "next_absolute_tick",
        "worker_receipt_sha256",
    }
    worker_sink = _require_exact_dict(
        worker.get("event_sink_ack"), "fresh-process sink acknowledgement"
    )
    reopened_worker_sink = validate_workhorse_event_sink_receipt_v2(
        worker_sink.get("receipt_path"),
        expected_artifact_identity_sha256=candidate_manifest["identity"][
            "identity_sha256"
        ],
        expected_end_next_absolute_tick=worker.get("next_absolute_tick"),
    )
    same_process = _require_exact_dict(
        restore.get("same_process_continuation"),
        "same-process continuation witness",
    )
    continuation_fields = {
        "state_sha256",
        "scratch_sha256",
        "events_sha256",
        "counts_sha256",
        "next_absolute_tick",
    }
    semantic_source_pid = value.get("source_pid")
    worker_source_pid = worker.get("source_pid")
    if (
        set(worker) != expected_worker_fields
        or worker.get("schema_version")
        != "eiel_workhorse_fresh_process_restore_receipt_v2"
        or worker.get("worker_receipt_sha256") != canonical_sha256(worker_body)
        or restore.get("fresh_process_receipt_file_sha256") != file_sha256(worker_path)
        or restore.get("fresh_process_receipt") != worker
        or worker.get("checkpoint_bundle_sha256") != checkpoint["bundle_body_sha256"]
        or worker.get("artifact_sha256") != candidate_manifest["output"]["sha256"]
        or worker_sink != reopened_worker_sink
        or type(semantic_source_pid) is not int
        or semantic_source_pid <= 0
        or type(worker_source_pid) is not int
        or worker_source_pid <= 0
        or worker_source_pid == semantic_source_pid
        or set(same_process) != continuation_fields
        or not all(
            _valid_sha256(same_process[field])
            for field in continuation_fields
            if field.endswith("_sha256")
        )
        or any(
            worker.get(field) != same_process.get(field)
            for field in continuation_fields
        )
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "fresh-process continuation witness does not reopen/recompute"
        )

    transaction = _require_exact_dict(
        witnesses.get("transaction_fail_stop"), "transaction witness"
    )
    forensic_sink = _require_exact_dict(
        transaction.get("forensic_sink_ack"), "forensic sink acknowledgement"
    )
    if (
        validate_workhorse_event_sink_receipt_v2(
            forensic_sink.get("receipt_path"),
            expected_artifact_identity_sha256=candidate_manifest["identity"][
                "identity_sha256"
            ],
        )
        != forensic_sink
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "failed-dispatch forensic sink does not reopen exactly"
        )
    causal = _require_exact_dict(witnesses.get("causal"), "causal witness")
    patch = _require_exact_dict(
        causal.get("patch_downstream_hashes"), "Patch downstream witness"
    )
    required_patch_hashes = {
        "baseline_resource_sha256",
        "patch_disabled_resource_sha256",
        "baseline_energy_sha256",
        "patch_disabled_energy_sha256",
    }
    if (
        set(patch) != required_patch_hashes
        or not all(_valid_sha256(item) for item in patch.values())
        or (
            patch["baseline_resource_sha256"] == patch["patch_disabled_resource_sha256"]
            and patch["baseline_energy_sha256"] == patch["patch_disabled_energy_sha256"]
        )
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "Patch has no downstream paired causal witness"
        )
    evidence_projection = _require_exact_dict(
        value.get("semantic_evidence"), "raw semantic evidence projection"
    )
    evidence_path = evidence_projection.get("manifest_path")
    evidence_manifest, datasets = reopen_workhorse_semantic_evidence_bundle_v2(
        evidence_path
    )
    if (
        evidence_projection
        != {
            "manifest_path": str(Path(evidence_path).resolve()),
            "manifest_file_sha256": file_sha256(evidence_path),
            "bundle_body_sha256": evidence_manifest["bundle_body_sha256"],
        }
        or evidence_manifest.get("shape_id") != candidate_manifest["shape_id"]
        or evidence_manifest.get("temporal_k") != temporal_k
        or evidence_manifest.get("candidate_artifact_sha256")
        != candidate_manifest["output"]["sha256"]
        or evidence_manifest.get("reference_artifact_sha256")
        != _require_exact_dict(value.get("reference"), "semantic reference").get(
            "artifact_sha256"
        )
        or evidence_manifest.get("governing_contracts")
        != workhorse_governing_contract_bindings_v2()
        or checkpoint_state != datasets["restore_checkpoint"]["state"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "raw semantic evidence does not bind the exact artifacts/checkpoint"
        )
    same = datasets["restore_same_continuation"]
    fresh = datasets["restore_fresh_continuation"]
    try:
        worker_authority = validate_workhorse_attempt_authority_binding_v2(
            worker.get("attempt_authority_binding"), historical=True
        )
        worker_closeout_authority = validate_workhorse_attempt_authority_binding_v2(
            worker.get("closeout_attempt_authority"), historical=True
        )
    except Exception as exc:
        raise FullEvolutionWorkhorseAuthorityError(
            f"fresh-process Attempt authority no longer validates: {exc}"
        ) from exc
    checkpoint_contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
        checkpoint["contract"]
    )
    expected_worker_contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=checkpoint_contract.shape_id,
        backend_id=checkpoint_contract.backend_id,
        B=checkpoint_contract.B,
        temporal_k=1,
        total_ticks=1,
        start_next_tick=checkpoint["checkpoint_at_tick"],
    )
    worker_request = worker_authority["request"]
    worker_response = worker_authority["response"]
    worker_closeout_request = worker_closeout_authority["request"]
    checkpoint_identity_sha256 = checkpoint["checkpoint_identity"][
        "checkpoint_identity_sha256"
    ]
    worker_raw = _reopen_fresh_worker_raw_continuation_v2(
        worker_path, worker, expected_dataset=fresh
    )
    if (
        worker_request.get("phase") != "EXECUTION_ROW"
        or worker_request.get("contract") != expected_worker_contract.to_dict()
        or worker_request.get("artifact_sha256") != checkpoint["artifact"]["sha256"]
        or worker_request.get("sealed_artifact_identity_sha256")
        != checkpoint["artifact"]["sealed_artifact_identity_sha256"]
        or worker_request.get("device_uuid")
        != checkpoint["event_sink_ack"]["device_uuid"]
        or worker_request.get("candidate_envelope_sha256") is not None
        or worker_request.get("prior_lineage_sha256") is not None
        or worker_request.get("source_checkpoint_identity_sha256")
        != checkpoint_identity_sha256
        or worker_request.get("governing_contracts_sha256")
        != canonical_sha256(workhorse_governing_contract_bindings_v2())
        or worker_response.get("owner_pid") != worker_source_pid
        or worker_closeout_request.get("phase") != "CLOSEOUT"
        or worker_closeout_request.get("contract") != expected_worker_contract.to_dict()
        or worker_closeout_request.get("artifact_sha256")
        != checkpoint["artifact"]["sha256"]
        or worker_closeout_request.get("sealed_artifact_identity_sha256")
        != checkpoint["artifact"]["sealed_artifact_identity_sha256"]
        or worker_closeout_request.get("device_uuid")
        != checkpoint["event_sink_ack"]["device_uuid"]
        or worker_closeout_request.get("candidate_envelope_sha256") is not None
        or worker_closeout_request.get("prior_lineage_sha256")
        != worker_authority["attempt_lineage_sha256"]
        or worker_closeout_request.get("source_checkpoint_identity_sha256") is not None
        or worker_closeout_authority["attempt_lineage_sha256"]
        != worker_authority["attempt_lineage_sha256"]
        or worker_closeout_authority["response"].get("owner_pid") != worker_source_pid
        or worker.get("closeout_attempt_authority") != worker_closeout_authority
        or worker_response.get("device_total_bytes") != runtime["total_device_bytes"]
        or checkpoint["attempt_authority_binding"]["response"].get("device_total_bytes")
        != runtime["total_device_bytes"]
        or worker_sink.get("owner_pid") != worker_source_pid
        or worker.get("attempt_authority_binding") != worker_authority
        or worker_sink.get("attempt_authority_binding") != worker_authority
        or worker_sink.get("attempt_authority_binding_sha256")
        != worker_authority["authority_binding_sha256"]
        or worker_sink.get("attempt_lineage_sha256")
        != worker_authority["attempt_lineage_sha256"]
        or worker_sink.get("world_uids") != checkpoint["world_uids"]
        or worker_sink.get("rng_identity")
        != {
            "schema_version": "eiel_workhorse_rng_identity_v2",
            "rng_epoch": 4,
            "world_uids": checkpoint["world_uids"],
            "rng_keys": checkpoint["rng_keys"],
        }
        or worker_sink.get("start_next_absolute_tick")
        != checkpoint["checkpoint_at_tick"]
        or worker_sink.get("end_next_absolute_tick")
        != checkpoint["checkpoint_at_tick"] + 1
        or worker_sink.get("pre_state_sha256") != _sha_bytes(worker_raw["input_state"])
        or worker_sink.get("post_state_sha256") != _sha_bytes(worker_raw["state"])
        or worker_sink.get("event_rows_file_sha256") != _sha_bytes(worker_raw["events"])
        or worker_sink.get("event_counts_file_sha256")
        != _sha_bytes(worker_raw["counts"])
        or worker_raw["input_state"] != checkpoint_state
        or worker_raw["input_events"] != bytes(len(worker_raw["input_events"]))
        or worker_raw["input_counts"] != bytes(len(worker_raw["input_counts"]))
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "fresh-process checkpoint/Attempt/RNG/state lineage is invalid"
        )
    expected_same_hashes = {
        f"{component}_sha256": _sha_bytes(same[component])
        for component in ("state", "scratch", "events", "counts")
    }
    expected_fresh_hashes = {
        f"{component}_sha256": _sha_bytes(fresh[component])
        for component in ("state", "scratch", "events", "counts")
    }
    if any(same_process.get(key) != item for key, item in expected_same_hashes.items()):
        raise FullEvolutionWorkhorseAuthorityError(
            "same-process restore witness differs from persisted raw bytes"
        )
    fresh_exact = all(
        worker.get(key) == item for key, item in expected_fresh_hashes.items()
    ) and all(
        worker_raw[component] == fresh[component]
        for component in WORKHORSE_FRESH_PROCESS_RAW_COMPONENTS_V2
    )
    checkpoint_ack_exact = (
        checkpoint_value["event_sink_ack"]
        == checkpoint_value["checkpoint_identity"]["event_sink_receipt"]
        and checkpoint_value["checkpoint_identity"]["event_sink_ack_sha256"]
        == checkpoint_value["event_sink_ack"]["receipt_file_sha256"]
    )
    _validate_semantic_attempt_lifecycles_v2(
        value,
        runtime=runtime,
        source_checkpoint_identity_sha256=checkpoint_identity_sha256,
    )
    return {
        "checkpoint_binds_external_exact_content_sink_ack": checkpoint_ack_exact,
        "sealed_checkpoint_reopens_exact_state_bytes": checkpoint_state
        == datasets["restore_checkpoint"]["state"],
        "fresh_process_restore_continuation_exact": fresh_exact
        and datasets["restore_same_continuation"]["state"]
        == datasets["restore_fresh_continuation"]["state"],
        "native_runtime_inspection_present": runtime.get("runtime_inspected") is True,
    }


def _validate_aggregate_semantic_gate_receipt_v2(
    path: Path,
    value: Mapping[str, Any],
    *,
    expected_shape_id: str,
    expected_candidate_artifact_sha256: str,
    expected_candidate_backend_id: str,
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "claim_class",
        "qualification_authority_granted",
        "science_authority_granted",
        "observation",
        "shape_id",
        "semantic_transition_sha256",
        "reference",
        "candidate",
        "semantic_case_aggregate",
        "semantic_evidence",
        "runtime",
        "temporal_k",
        "checks",
        "derived_evidence_witness",
        "supporting_evidence_witness",
        "decision",
        "decision_authority",
        "governing_contracts",
        "semantic_gate_receipt_sha256",
    }
    body = {
        key: item
        for key, item in value.items()
        if key != "semantic_gate_receipt_sha256"
    }
    if (
        set(value) != fields
        or value.get("schema_version") != WORKHORSE_SEMANTIC_GATE_SCHEMA_V2
        or value.get("claim_class") != "TECHNICAL_ONLY"
        or value.get("qualification_authority_granted") is not False
        or value.get("science_authority_granted") is not False
        or value.get("observation") != _OBSERVATION_OFF_V2
        or value.get("decision") != "SEMANTIC_GATE_SATISFIED"
        or value.get("decision_authority") != _AGGREGATE_SEMANTIC_GATE_AUTHORITY_V2
        or value.get("semantic_gate_receipt_sha256") != canonical_sha256(body)
        or value.get("governing_contracts")
        != workhorse_governing_contract_bindings_v2()
        or value.get("shape_id") != expected_shape_id
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "aggregate semantic gate decision/seal/authority is invalid"
        )
    reference, _, reference_manifest = _validate_semantic_artifact_projection_v2(
        value.get("reference"),
        label="reference",
        require_current_source_match=require_current_source_match,
    )
    candidate, candidate_artifact, candidate_manifest = (
        _validate_semantic_artifact_projection_v2(
            value.get("candidate"),
            label="candidate",
            require_current_source_match=require_current_source_match,
        )
    )
    if (
        reference_manifest["backend_id"] != "reference_phased_v2"
        or reference_manifest["shape_id"] != expected_shape_id
        or candidate_manifest["backend_id"] != expected_candidate_backend_id
        or candidate_manifest["shape_id"] != expected_shape_id
        or file_sha256(candidate_artifact) != expected_candidate_artifact_sha256
        or value.get("semantic_transition_sha256")
        != candidate_manifest["identity"]["semantic_transition_sha256"]
        or reference_manifest["identity"]["semantic_transition_sha256"]
        != candidate_manifest["identity"]["semantic_transition_sha256"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "aggregate semantic gate is for different artifacts or semantics"
        )
    aggregate_projection = _require_exact_dict(
        value.get("semantic_case_aggregate"), "semantic case aggregate projection"
    )
    expected_projection, aggregate = _semantic_case_aggregate_projection_v2(
        aggregate_projection.get("path")
    )
    evidence_projection = _require_exact_dict(
        value.get("semantic_evidence"), "aggregate semantic evidence projection"
    )
    expected_evidence_projection = {
        key: aggregate["semantic_evidence"][key]
        for key in ("manifest_path", "manifest_file_sha256", "bundle_body_sha256")
    }
    evidence_manifest, _ = reopen_workhorse_semantic_evidence_bundle_v2(
        expected_evidence_projection["manifest_path"]
    )
    if (
        aggregate_projection != expected_projection
        or evidence_projection != expected_evidence_projection
        or aggregate["shape_id"] != expected_shape_id
        or aggregate["semantic_temporal_k"] != value.get("temporal_k")
        or evidence_manifest["reference_artifact_sha256"]
        != reference["artifact_sha256"]
        or evidence_manifest["candidate_artifact_sha256"]
        != candidate["artifact_sha256"]
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "aggregate semantic evidence/artifact projection changed"
        )
    evidence_checks, evidence_witness = derive_workhorse_semantic_checks_v2(
        Path(evidence_projection["manifest_path"])
    )
    supporting, runtime, supporting_witness = _derive_aggregate_semantic_support_v2(
        aggregate,
        reference_manifest=reference_manifest,
        candidate_manifest=candidate_manifest,
    )
    checks = dict(evidence_checks)
    for name, passed in supporting.items():
        checks[name] = checks.get(name, True) and passed
    if (
        set(checks) != WORKHORSE_REQUIRED_SEMANTIC_CHECKS_V2
        or value.get("checks") != checks
        or not all(checks.values())
        or value.get("runtime") != runtime
        or value.get("derived_evidence_witness") != evidence_witness
        or value.get("supporting_evidence_witness") != supporting_witness
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "aggregate semantic PASS does not derive from terminalized evidence"
        )
    return {
        "schema_version": WORKHORSE_SEMANTIC_GATE_SCHEMA_V2,
        "receipt_path": str(path),
        "receipt_file_sha256": file_sha256(path),
        "semantic_gate_receipt_sha256": value["semantic_gate_receipt_sha256"],
        "shape_id": expected_shape_id,
        "candidate_backend_id": expected_candidate_backend_id,
        "candidate_artifact_sha256": expected_candidate_artifact_sha256,
        "decision": value["decision"],
        "runtime_device_identity": {
            "device_uuid": runtime["device_uuid"],
            "target_sm": runtime["target_sm"],
            "multiprocessor_count": runtime["multiprocessor_count"],
            "total_device_bytes": runtime["total_device_bytes"],
        },
        "governing_contracts": value["governing_contracts"],
    }


def validate_workhorse_semantic_gate_receipt_v2(
    receipt_path: str | Path,
    *,
    expected_shape_id: str,
    expected_candidate_artifact_sha256: str,
    expected_candidate_backend_id: str,
    require_current_source_match: bool = True,
) -> dict[str, Any]:
    """Reopen a passed semantic receipt and both exact sealed artifacts."""
    if expected_candidate_backend_id not in {
        "phase_graph_v2",
        "fused_cta_temporal_v2",
    }:
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic candidate backend must be a performance backend"
        )
    path, value = _read_json(receipt_path, "workhorse semantic gate receipt")
    if value.get("decision_authority") == _AGGREGATE_SEMANTIC_GATE_AUTHORITY_V2:
        return _validate_aggregate_semantic_gate_receipt_v2(
            path,
            value,
            expected_shape_id=expected_shape_id,
            expected_candidate_artifact_sha256=(expected_candidate_artifact_sha256),
            expected_candidate_backend_id=expected_candidate_backend_id,
            require_current_source_match=require_current_source_match,
        )
    body = {
        key: item
        for key, item in value.items()
        if key != "semantic_gate_receipt_sha256"
    }
    checks = value.get("checks")
    candidate = _require_exact_dict(value.get("candidate"), "semantic candidate")
    reference = _require_exact_dict(value.get("reference"), "semantic reference")
    if (
        value.get("schema_version") != WORKHORSE_SEMANTIC_GATE_SCHEMA_V2
        or value.get("decision") != "SEMANTIC_GATE_SATISFIED"
        or value.get("decision_authority")
        != "reopened_raw_state_scratch_event_derivation_v2"
        or value.get("semantic_gate_receipt_sha256") != canonical_sha256(body)
        or value.get("governing_contracts")
        != workhorse_governing_contract_bindings_v2()
        or value.get("shape_id") != expected_shape_id
        or type(checks) is not dict
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic gate decision/seal/checks are invalid"
        )
    reopened = []
    for label, row in (("reference", reference), ("candidate", candidate)):
        try:
            artifact, _, manifest = validate_workhorse_prebuilt_v2(
                row.get("artifact"),
                require_current_source_match=require_current_source_match,
            )
        except Exception as exc:
            raise FullEvolutionWorkhorseAuthorityError(
                f"cannot reopen semantic {label} artifact: {exc}"
            ) from exc
        if (
            row.get("artifact_sha256") != file_sha256(artifact)
            or row.get("manifest_sha256") != manifest["manifest_sha256"]
            or row.get("identity") != manifest["identity"]
            or row.get("build_identity") != manifest["build_identity"]
            or row.get("sealed_artifact_identity")
            != manifest["sealed_artifact_identity"]
        ):
            raise FullEvolutionWorkhorseAuthorityError(
                f"semantic {label} source binding is invalid"
            )
        reopened.append((artifact, manifest))
    _, reference_manifest = reopened[0]
    candidate_artifact, candidate_manifest = reopened[1]
    if (
        reference_manifest["backend_id"] != "reference_phased_v2"
        or reference_manifest["shape_id"] != expected_shape_id
        or reference_manifest["identity"]["semantic_transition_sha256"]
        != candidate_manifest["identity"]["semantic_transition_sha256"]
        or file_sha256(candidate_artifact) != expected_candidate_artifact_sha256
        or candidate_manifest["backend_id"] != expected_candidate_backend_id
        or candidate_manifest["shape_id"] != expected_shape_id
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic receipt is for a different candidate"
        )
    evidence_projection = _require_exact_dict(
        value.get("semantic_evidence"), "raw semantic evidence projection"
    )
    evidence_checks, evidence_witness = derive_workhorse_semantic_checks_v2(
        evidence_projection.get("manifest_path")
    )
    supporting_checks = _validate_semantic_supporting_evidence_v2(
        value, candidate_manifest=candidate_manifest
    )
    derived_checks = dict(evidence_checks)
    for name, passed in supporting_checks.items():
        derived_checks[name] = derived_checks.get(name, True) and passed
    if (
        set(derived_checks) != WORKHORSE_REQUIRED_SEMANTIC_CHECKS_V2
        or value.get("checks") != derived_checks
        or not all(derived_checks.values())
        or value.get("derived_evidence_witness") != evidence_witness
    ):
        raise FullEvolutionWorkhorseAuthorityError(
            "semantic PASS does not independently derive from persisted evidence"
        )
    return {
        "schema_version": WORKHORSE_SEMANTIC_GATE_SCHEMA_V2,
        "receipt_path": str(path),
        "receipt_file_sha256": file_sha256(path),
        "semantic_gate_receipt_sha256": value["semantic_gate_receipt_sha256"],
        "shape_id": expected_shape_id,
        "candidate_backend_id": expected_candidate_backend_id,
        "candidate_artifact_sha256": expected_candidate_artifact_sha256,
        "decision": value["decision"],
        "runtime_device_identity": {
            "device_uuid": value["runtime"]["device_uuid"],
            "target_sm": value["runtime"]["target_sm"],
            "multiprocessor_count": value["runtime"]["multiprocessor_count"],
            "total_device_bytes": value["runtime"]["total_device_bytes"],
        },
        "governing_contracts": value["governing_contracts"],
    }


def seal_workhorse_terminalized_semantic_case_receipt_v2(
    output_path: str | Path,
    *,
    case_fragment_path: str | Path,
    canonical_terminal_binding: Mapping[str, Any],
    produced_external_source_checkpoint_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal one canonical semantic case after Scheduler terminal reopen."""

    path = write_terminalized_workhorse_semantic_case_receipt_v2(
        output_path,
        case_fragment_path=case_fragment_path,
        canonical_terminal_binding=canonical_terminal_binding,
        produced_external_source_checkpoint_binding=(
            produced_external_source_checkpoint_binding
        ),
    )
    return reopen_terminalized_workhorse_semantic_case_receipt_v2(path)


def validate_workhorse_terminalized_semantic_case_receipt_v2(
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Reopen one terminal semantic result without granting broader authority."""

    return reopen_terminalized_workhorse_semantic_case_receipt_v2(receipt_path)


__all__ = [
    "ARCHITECTURE_CONTRACT_RELATIVE_PATH_V2",
    "FullEvolutionWorkhorseAuthorityError",
    "PERFORMANCE_CONTRACT_RELATIVE_PATH_V2",
    "PROJECT_ROOT_V2",
    "VALIDATION_MATRIX_RELATIVE_PATH_V2",
    "WORKHORSE_SEMANTIC_CONTRACT_RELATIVE_PATH_V2",
    "WORKHORSE_CHECKPOINT_BUNDLE_SCHEMA_V2",
    "WORKHORSE_EVENT_DISPATCH_SCHEMA_V2",
    "WORKHORSE_EVENT_SINK_RECEIPT_SCHEMA_V2",
    "WORKHORSE_SPARSE_EVENT_DISPATCH_SCHEMA_V3",
    "WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4",
    "WORKHORSE_SPARSE_EVENT_SINK_RECEIPT_SCHEMA_V3",
    "WORKHORSE_TECHNICAL_DIGEST_CONSUMER_PROFILE_V1",
    "WORKHORSE_REQUIRED_SEMANTIC_CHECKS_V2",
    "WORKHORSE_SEMANTIC_CASE_TERMINAL_RECEIPT_SCHEMA_V2",
    "publish_workhorse_event_sink_receipt_v2",
    "publish_workhorse_sparse_event_sink_receipt_v3",
    "inspect_workhorse_checkpoint_bundle_v2",
    "reopen_workhorse_checkpoint_bundle_v2",
    "seal_workhorse_terminalized_semantic_case_receipt_v2",
    "validate_authoritative_micro_denominator_v2",
    "validate_workhorse_event_sink_receipt_v2",
    "validate_workhorse_semantic_gate_receipt_v2",
    "validate_workhorse_terminalized_semantic_case_receipt_v2",
    "workhorse_checkpoint_metadata_from_state_identity_v2",
    "workhorse_event_dispatch_content_sha256_v2",
    "workhorse_sparse_event_dispatch_content_sha256_v3",
    "workhorse_governing_contract_bindings_v2",
    "write_workhorse_checkpoint_bundle_v2",
    "write_workhorse_semantic_case_aggregate_gate_receipt_v2",
    "write_workhorse_semantic_gate_receipt_v2",
]
