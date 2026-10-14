"""Authenticated host-boundary observations for selected-world continuations.

The indexed continuation runtime deliberately remains a technical execution
surface with ``observation_mode=OFF``.  This module is an additive adapter over
the already committed terminal checkpoint and acknowledged event receipts.  It
does not observe or mutate the resident owner and it does not infer event kinds
whose raw rows were intentionally released.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from typing import Any, Final

from .contract import FullEvolutionCudaWorkhorseContractV2, canonical_sha256
from .continuation import validate_workhorse_indexed_continuation_result_v2
from .indexed_checkpoint import inspect_workhorse_indexed_checkpoint_set_v1
from .layout import WorkhorseFieldV2, workhorse_layout_v2


WORKHORSE_SELECTED_WORLD_OBSERVATION_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_selected_world_host_observation_v1"
)
WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1: Final[str] = (
    "WORKHORSE_SELECTED_WORLD_HOST_BOUNDARY_COMPACT_V1"
)
WORKHORSE_SELECTED_WORLD_OBSERVATION_CLAIM_CLASS_V1: Final[str] = (
    "AUTHENTICATED_SCIENTIFIC_OBSERVATION_NO_EXECUTION_OR_QUALIFICATION_AUTHORITY"
)

_MAXIMUM_OBSERVATION_BYTES: Final[int] = 4 * 1024 * 1024
_MAXIMUM_ACTIVE_AGENT_EXAMPLES: Final[int] = 32
_SHA256_FIELDS: Final[tuple[str, ...]] = (
    "source_checkpoint_occurrence_sha256",
    "source_checkpoint_occurrence_projection_sha256",
)
_HEADER_COUNTERS: Final[tuple[tuple[str, int], ...]] = (
    ("ticks_completed", 5),
    ("patch_engagement_count", 6),
    ("wdl_engagement_count", 7),
    ("jungle_engagement_count", 8),
    ("eed_engagement_count", 9),
    ("birth_capability_count", 10),
    ("mutation_capability_count", 11),
    ("recombination_capability_count", 12),
    ("birth_count", 15),
    ("death_count", 16),
    ("mutation_count", 17),
    ("recombination_count", 18),
    ("last_dispatch_event_count", 19),
)
_SITE_FIELDS: Final[tuple[str, ...]] = (
    "resource",
    "patch_field",
    "patch_pending",
    "wdl_delay_0",
    "wdl_delay_1",
    "wdl_active",
    "jungle",
    "eed_field",
)


class WorkhorseSelectedWorldObservationError(ValueError):
    """The selected-world host observation is absent or identity-inexact."""


def _fail(message: str) -> None:
    raise WorkhorseSelectedWorldObservationError(message)


def _valid_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _field_values(state: bytes, field: WorkhorseFieldV2) -> memoryview:
    format_by_dtype = {
        "u8": "B",
        "u16": "H",
        "u32": "I",
        "i32": "i",
        "u64": "Q",
    }
    try:
        wire = format_by_dtype[field.dtype]
    except KeyError as exc:  # pragma: no cover - layout validation owns this law.
        raise WorkhorseSelectedWorldObservationError(
            f"unsupported Workhorse observation field dtype: {field.dtype}"
        ) from exc
    view = memoryview(state)[field.offset : field.offset + field.nbytes]
    if field.count == 0:
        return view
    # Workhorse CUDA artifacts and their checkpoint ABI are little-endian.
    # Every supported deployment host is little-endian; struct verifies the
    # same interpretation at the exact offset before exposing the zero-copy
    # view used for large selected worlds.
    if struct.unpack_from(f"<{wire}", state, field.offset)[0] != view.cast(wire)[0]:
        _fail("host byte order differs from the Workhorse checkpoint ABI")
    return view.cast(wire)


def _binary_digest(prefix: bytes, rows: list[bytes]) -> str:
    digest = hashlib.sha256(prefix + b"\0")
    for row in rows:
        digest.update(row)
    return digest.hexdigest()


def _summary(values: list[int]) -> dict[str, int | None]:
    return {
        "count": len(values),
        "sum": sum(values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _selected_result_identity(result: Mapping[str, Any]) -> tuple[dict[str, Any], FullEvolutionCudaWorkhorseContractV2]:
    continuation = result.get("continuation_contract")
    logical = (
        continuation.get("science_continuation_identity")
        if isinstance(continuation, Mapping)
        else None
    )
    physical = (
        continuation.get("physical_continuation_plan")
        if isinstance(continuation, Mapping)
        else None
    )
    if not isinstance(logical, dict) or not isinstance(physical, Mapping):
        _fail("indexed continuation lacks its logical/physical identity")
    required = {
        "source_checkpoint_occurrence_sha256",
        "source_parent_occurrence_sha256",
        "source_semantic_world_sha256",
        "source_control_history_sha256",
        "source_rng_state_sha256",
        "source_event_prefix_sha256",
        "source_active_control_state_sha256",
    }
    if not required.issubset(logical) or any(
        not _valid_sha256(logical[field]) for field in required
    ):
        _fail("host observation requires one authenticated selected-world source")
    try:
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            physical["target_contract"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise WorkhorseSelectedWorldObservationError(
            f"selected-world target contract is invalid: {exc}"
        ) from exc
    if (
        contract.B != 1
        or logical.get("world_count") != 1
        or result.get("outcome") != "CONTINUED"
        or result.get("absolute_frontiers", {}).get("committed_next_tick")
        != contract.start_next_tick + contract.total_ticks
    ):
        _fail("host observation is restricted to one completed selected world")
    return logical, contract


def _event_coverage(result: Mapping[str, Any], *, world_uid: int) -> dict[str, Any]:
    measurements = result.get("dispatch_measurements")
    if not isinstance(measurements, list) or not measurements:
        _fail("selected-world observation lacks acknowledged dispatch coverage")
    segments: list[dict[str, Any]] = []
    total = 0
    previous_end: int | None = None
    for ordinal, measurement in enumerate(measurements):
        ack = measurement.get("event_sink_receipt") if isinstance(measurement, Mapping) else None
        if not isinstance(ack, Mapping):
            _fail("selected-world dispatch lacks its event acknowledgement")
        counts = ack.get("event_counts")
        start = ack.get("start_next_absolute_tick")
        end = ack.get("end_next_absolute_tick")
        if (
            ack.get("world_uids") != [world_uid]
            or not isinstance(counts, list)
            or len(counts) != 1
            or type(counts[0]) is not int
            or counts[0] < 0
            or type(start) is not int
            or type(end) is not int
            or end <= start
            or (previous_end is not None and start != previous_end)
            or any(
                not _valid_sha256(ack.get(field))
                for field in (
                    "receipt_file_sha256",
                    "receipt_body_sha256",
                    "dispatch_content_sha256",
                    "post_state_sha256",
                )
            )
        ):
            _fail("selected-world event acknowledgement coverage is inexact")
        count = counts[0]
        total += count
        segments.append(
            {
                "dispatch_ordinal": ordinal,
                "start_next_absolute_tick": start,
                "end_next_absolute_tick": end,
                "event_count": count,
                "sink_receipt_file_sha256": ack["receipt_file_sha256"],
                "sink_receipt_body_sha256": ack["receipt_body_sha256"],
                "dispatch_content_sha256": ack["dispatch_content_sha256"],
                "post_state_sha256": ack["post_state_sha256"],
            }
        )
        previous_end = end
    frontiers = result["absolute_frontiers"]
    if (
        segments[0]["start_next_absolute_tick"] != frontiers["start_next_tick"]
        or segments[-1]["end_next_absolute_tick"] != frontiers["committed_next_tick"]
    ):
        _fail("selected-world event acknowledgements do not cover the continuation")
    body = {
        "coverage": "ACKNOWLEDGED_COUNTS_AND_DIGESTS_ONLY",
        "event_kind_rows_available": False,
        "event_kind_rows_unavailable_reason": (
            "TECHNICAL_DIGEST_CONSUMER_RELEASED_RAW_PACKED_EVENTS"
        ),
        "birth_death_causal_rows_claimed": False,
        "total_event_count": total,
        "segments": segments,
    }
    return {**body, "coverage_sha256": canonical_sha256(body)}


def _world_observation(
    state: bytes,
    *,
    contract: FullEvolutionCudaWorkhorseContractV2,
    world_uid: int,
    expected_state_sha256: str,
) -> dict[str, Any]:
    layout = workhorse_layout_v2(contract.shape)
    if len(state) != layout.state_bytes_per_world:
        _fail("selected-world state has another Workhorse layout size")
    state_sha256 = hashlib.sha256(state).hexdigest()
    if state_sha256 != expected_state_sha256:
        _fail("selected-world state changed before host observation")
    fields = {
        field.name: _field_values(state, field) for field in layout.state_fields
    }
    header = fields["header"]
    if (
        header[0] != world_uid
        or header[1] != contract.start_next_tick + contract.total_ticks
        or header[4] != 0
        or header[14] != 4
    ):
        _fail("selected-world host observation has another identity/frontier")
    alive = fields["alive"]
    if any(value not in {0, 1} for value in alive):
        _fail("selected-world alive registry is not binary")
    active_slots = [slot for slot, value in enumerate(alive) if value == 1]
    if len(active_slots) != header[3]:
        _fail("selected-world active population differs from alive state")
    agent_ids = fields["agent_id"]
    active_ids = [agent_ids[slot] for slot in active_slots]
    if any(value <= 0 for value in active_ids) or len(set(active_ids)) != len(active_ids):
        _fail("selected-world active agent identities are invalid")

    genome_a = fields["genome_a"]
    genome_b = fields["genome_b"]
    expressed = fields["expressed"]
    ordered_active_slots = sorted(active_slots, key=lambda slot: agent_ids[slot])
    genotype_counts: Counter[str] = Counter()
    parent_pairs: set[tuple[int, int]] = set()
    founder_count = 0
    allele_histograms = [
        {
            "haplotype_a": Counter(),
            "haplotype_b": Counter(),
            "expressed": Counter(),
        }
        for _ in range(32)
    ]
    examples: list[dict[str, Any]] = []
    identity_digest_rows: list[bytes] = []
    lineage_digest_rows: list[bytes] = []
    for example_ordinal, slot in enumerate(ordered_active_slots):
        base = slot * 32
        haplotype_a = list(genome_a[base : base + 32])
        haplotype_b = list(genome_b[base : base + 32])
        expressed_row = list(expressed[base : base + 32])
        genotype_sha256 = _binary_digest(
            b"eiel_workhorse_active_genotype_u8_v1",
            [bytes(haplotype_a), bytes(haplotype_b), bytes(expressed_row)],
        )
        genotype_counts[genotype_sha256] += 1
        parent_pair = (
            int(fields["parent_a_id"][slot]),
            int(fields["parent_b_id"][slot]),
        )
        parent_pairs.add(parent_pair)
        founder_count += int(parent_pair == (0, 0))
        for locus in range(32):
            allele_histograms[locus]["haplotype_a"][haplotype_a[locus]] += 1
            allele_histograms[locus]["haplotype_b"][haplotype_b[locus]] += 1
            allele_histograms[locus]["expressed"][expressed_row[locus]] += 1
        agent_id = int(agent_ids[slot])
        birth_tick = int(fields["birth_tick"][slot])
        identity_digest_rows.append(struct.pack("<Q", agent_id))
        lineage_digest_rows.append(
            struct.pack(
                "<4Q", agent_id, parent_pair[0], parent_pair[1], birth_tick
            )
        )
        if example_ordinal < _MAXIMUM_ACTIVE_AGENT_EXAMPLES:
            examples.append({
                "agent_id": agent_ids[slot],
                "parent_a_id": parent_pair[0],
                "parent_b_id": parent_pair[1],
                "absolute_birth_tick": birth_tick,
                "position": fields["position"][slot],
                "energy": fields["energy"][slot],
                "health": fields["health"][slot],
                "age": fields["age"][slot],
                "genotype_sha256": genotype_sha256,
                "haplotype_a_u8": haplotype_a,
                "haplotype_b_u8": haplotype_b,
                "expressed_u8": expressed_row,
            })
    genotype_multiset_rows = [
        bytes.fromhex(digest) + struct.pack("<Q", count)
        for digest, count in sorted(genotype_counts.items())
    ]
    locus_histograms = []
    for locus, histograms in enumerate(allele_histograms):
        locus_histograms.append(
            {
                "locus": locus,
                **{
                    name: [
                        {"value_u8": value, "count": count}
                        for value, count in sorted(histogram.items())
                    ]
                    for name, histogram in histograms.items()
                },
            }
        )
    counters = {name: header[index] for name, index in _HEADER_COUNTERS}
    site_totals = {f"{name}_total": sum(fields[name]) for name in _SITE_FIELDS}
    occupied_site_count = sum(value != 0xFFFFFFFF for value in fields["occupancy"])
    body = {
        "world_uid": world_uid,
        "next_absolute_tick": header[1],
        "rng_key": header[13],
        "rng_epoch": header[14],
        "status": header[4],
        "active_population": header[3],
        "alive_count": len(active_slots),
        "occupied_site_count": occupied_site_count,
        "cumulative_counters": counters,
        "site_field_totals": site_totals,
        "active_agent_aggregates": {
            "energy": _summary([fields["energy"][slot] for slot in active_slots]),
            "health": _summary([fields["health"][slot] for slot in active_slots]),
            "age": _summary([fields["age"][slot] for slot in active_slots]),
        },
        "active_lineage_summary": {
            "active_agent_identity_digest_domain": (
                "eiel_workhorse_active_agent_identity_set_v1"
            ),
            "active_agent_identity_set_encoding": (
                "agent_id_u64le_in_ascending_agent_id_order_v1"
            ),
            "active_agent_identity_set_sha256": _binary_digest(
                b"eiel_workhorse_active_agent_identity_set_v1",
                identity_digest_rows,
            ),
            "active_lineage_digest_domain": (
                "eiel_workhorse_active_lineage_v1"
            ),
            "lineage_encoding": (
                "agent_parent_a_parent_b_birth_tick_u64le_in_ascending_agent_id_order_v1"
            ),
            "active_lineage_sha256": _binary_digest(
                b"eiel_workhorse_active_lineage_v1", lineage_digest_rows
            ),
            "unique_parent_pair_count": len(parent_pairs),
            "active_founder_count": founder_count,
        },
        "active_genotype_summary": {
            "genotype_encoding": (
                "haplotype_a_32_u8_then_haplotype_b_32_u8_then_expressed_32_u8_v1"
            ),
            "genotype_digest_domain": "eiel_workhorse_active_genotype_u8_v1",
            "unique_genotype_count": len(genotype_counts),
            "genotype_multiset_encoding": (
                "genotype_sha256_bytes_then_count_u64le_sorted_by_digest_v1"
            ),
            "genotype_multiset_digest_domain": (
                "eiel_workhorse_active_genotype_multiset_v1"
            ),
            "genotype_multiset_sha256": _binary_digest(
                b"eiel_workhorse_active_genotype_multiset_v1",
                genotype_multiset_rows,
            ),
            "locus_value_histograms": locus_histograms,
        },
        "bounded_active_agent_examples": {
            "selection_rule": "LOWEST_AGENT_ID_ASCENDING",
            "maximum_rows": _MAXIMUM_ACTIVE_AGENT_EXAMPLES,
            "represented_rows": len(examples),
            "total_active_rows": len(active_slots),
            "complete": len(examples) == len(active_slots),
            "rows": examples,
        },
        "canonical_world_state_sha256": state_sha256,
    }
    return {**body, "world_observation_sha256": canonical_sha256(body)}


def project_workhorse_exact_state_observation_v1(
    state: bytes,
    *,
    contract: FullEvolutionCudaWorkhorseContractV2,
    world_uid: int,
    expected_state_sha256: str,
) -> dict[str, Any]:
    """Project authenticated facts from one already-reopened exact state row."""

    if type(state) is not bytes:
        _fail("exact-state observation input must be immutable bytes")
    if not isinstance(contract, FullEvolutionCudaWorkhorseContractV2):
        _fail("exact-state observation requires a Workhorse contract")
    return _world_observation(
        state,
        contract=contract,
        world_uid=world_uid,
        expected_state_sha256=expected_state_sha256,
    )


def _build_observation(
    *,
    execution_result_path: str | Path,
    terminal_indexed_checkpoint_manifest_path: str | Path,
    source_checkpoint_occurrence_sha256: str,
    source_checkpoint_occurrence_projection_sha256: str,
    cell_id: str,
) -> dict[str, Any]:
    if not isinstance(cell_id, str) or not cell_id or cell_id != cell_id.strip():
        _fail("selected-world observation cell identity is invalid")
    for field, value in zip(
        _SHA256_FIELDS,
        (
            source_checkpoint_occurrence_sha256,
            source_checkpoint_occurrence_projection_sha256,
        ),
        strict=True,
    ):
        if not _valid_sha256(value):
            _fail(f"{field} is not a lowercase SHA-256")
    result = validate_workhorse_indexed_continuation_result_v2(
        execution_result_path
    )
    logical, contract = _selected_result_identity(result)
    indexed, state_path = inspect_workhorse_indexed_checkpoint_set_v1(
        terminal_indexed_checkpoint_manifest_path
    )
    identity = indexed.get("world_state_set_identity")
    rows = indexed.get("world_index_by_uid")
    restored = result.get("restored_world_uids")
    terminal_tick = contract.start_next_tick + contract.total_ticks
    if (
        not isinstance(identity, Mapping)
        or identity.get("world_count") != 1
        or identity.get("checkpoint_at_tick") != terminal_tick
        or identity.get("shape_id") != contract.shape_id
        or identity.get("semantic_transition_sha256")
        != contract.semantic_transition_sha256
        or not isinstance(rows, list)
        or len(rows) != 1
        or not isinstance(rows[0], Mapping)
        or not isinstance(restored, list)
        or len(restored) != 1
        or rows[0].get("world_uid") != restored[0]
        or source_checkpoint_occurrence_sha256
        == logical["source_checkpoint_occurrence_sha256"]
    ):
        # A child terminal occurrence must never alias the parent occurrence.
        _fail("selected-world terminal checkpoint identity is not exact")
    row = rows[0]
    with state_path.open("rb") as handle:
        handle.seek(int(row["state_offset"]))
        state = handle.read(int(row["state_bytes"]))
    world = _world_observation(
        state,
        contract=contract,
        world_uid=int(row["world_uid"]),
        expected_state_sha256=str(row["state_sha256"]),
    )
    event_coverage = _event_coverage(result, world_uid=int(row["world_uid"]))
    control = result.get("control_application")
    if not isinstance(control, Mapping):
        _fail("selected-world observation lacks authenticated control readback")
    control_projection = {
        "control_contract_sha256": control.get("control_contract_sha256"),
        "applied_next_absolute_tick": control.get("applied_next_absolute_tick"),
        "first_permitted_divergence_tick": control.get(
            "first_permitted_divergence_tick"
        ),
        "ordered_native_fields": control.get("ordered_native_fields"),
        "requested_native_rows": control.get("requested_native_rows"),
        "readback_native_rows": control.get("readback_native_rows"),
        "control_application_sha256": control.get("control_application_sha256"),
    }
    if any(
        not _valid_sha256(control_projection[field])
        for field in ("control_contract_sha256", "control_application_sha256")
    ):
        _fail("selected-world control readback identity is invalid")
    result_path = Path(str(result["receipt_path"])).resolve()
    indexed_path = Path(terminal_indexed_checkpoint_manifest_path).resolve()
    body = {
        "schema_version": WORKHORSE_SELECTED_WORLD_OBSERVATION_SCHEMA_V1,
        "observation_mode": WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1,
        "claim_class": WORKHORSE_SELECTED_WORLD_OBSERVATION_CLAIM_CLASS_V1,
        "time_authority": "absolute_tick_v1",
        "observation_boundary": (
            "HOST_PROJECTION_AFTER_ACKNOWLEDGED_TERMINAL_CHECKPOINT_COMMIT"
        ),
        "execution_authority_granted": False,
        "qualification_authority_granted": False,
        "named_phenotype_reward_authority_granted": False,
        "cell_id": cell_id,
        "source_continuation": {
            "execution_result_path": str(result_path),
            "execution_result_file_sha256": result["receipt_file_sha256"],
            "execution_result_content_sha256": result["receipt_content_sha256"],
            "science_continuation_identity_sha256": logical[
                "science_continuation_identity_sha256"
            ],
            "source_semantic_world_sha256": logical[
                "source_semantic_world_sha256"
            ],
            "source_control_history_sha256": logical[
                "source_control_history_sha256"
            ],
            "source_rng_state_sha256": logical["source_rng_state_sha256"],
            "source_event_prefix_sha256": logical["source_event_prefix_sha256"],
        },
        "terminal_checkpoint": {
            "indexed_manifest_path": str(indexed_path),
            "indexed_manifest_file_sha256": _file_sha256(indexed_path),
            "indexed_manifest_body_sha256": indexed["manifest_body_sha256"],
            "world_state_set_sha256": identity["world_state_set_sha256"],
            "checkpoint_at_tick": terminal_tick,
            "checkpoint_occurrence_sha256": (
                source_checkpoint_occurrence_sha256
            ),
            "checkpoint_occurrence_projection_sha256": (
                source_checkpoint_occurrence_projection_sha256
            ),
        },
        "control_readback": control_projection,
        "event_observation_coverage": event_coverage,
        "world_observations": [world],
        "integrity": {
            "world_count": 1,
            "dropped_record_count": 0,
            "checkpoint_state_reopened_exact": True,
            "event_acknowledgements_reopened_exact": True,
            "control_readback_authenticated": True,
            "runtime_observation_mode_remains_off": True,
            "host_boundary_adapter_is_observation_only": True,
        },
    }
    return {**body, "observation_content_sha256": canonical_sha256(body)}


def seal_workhorse_selected_world_observation_v1(
    output_path: str | Path,
    *,
    execution_result_path: str | Path,
    terminal_indexed_checkpoint_manifest_path: str | Path,
    source_checkpoint_occurrence_sha256: str,
    source_checkpoint_occurrence_projection_sha256: str,
    cell_id: str,
) -> dict[str, Any]:
    """Seal one compact observation from an exact terminal selected world."""

    observation = _build_observation(
        execution_result_path=execution_result_path,
        terminal_indexed_checkpoint_manifest_path=(
            terminal_indexed_checkpoint_manifest_path
        ),
        source_checkpoint_occurrence_sha256=source_checkpoint_occurrence_sha256,
        source_checkpoint_occurrence_projection_sha256=(
            source_checkpoint_occurrence_projection_sha256
        ),
        cell_id=cell_id,
    )
    requested = Path(output_path).expanduser()
    requested.parent.mkdir(parents=True, exist_ok=True)
    if requested.parent.is_symlink():
        _fail("selected-world observation parent is a symbolic link")
    path = requested.parent.resolve() / requested.name
    if path.is_symlink():
        _fail("selected-world observation destination is a symbolic link")
    if path.exists():
        existing = validate_workhorse_selected_world_observation_v1(
            path,
            expected_checkpoint_occurrence_sha256=(
                source_checkpoint_occurrence_sha256
            ),
            expected_checkpoint_occurrence_projection_sha256=(
                source_checkpoint_occurrence_projection_sha256
            ),
        )
        source = existing["source_continuation"]
        terminal = existing["terminal_checkpoint"]
        if (
            existing["cell_id"] != cell_id
            or Path(source["execution_result_path"]).resolve()
            != Path(execution_result_path).resolve()
            or Path(terminal["indexed_manifest_path"]).resolve()
            != Path(terminal_indexed_checkpoint_manifest_path).resolve()
        ):
            _fail("existing selected-world observation names another source")
        return existing
    payload = _canonical_json_bytes(observation)
    if len(payload) > _MAXIMUM_OBSERVATION_BYTES:
        _fail("selected-world compact observation exceeds its bounded size")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    path.chmod(0o444)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return validate_workhorse_selected_world_observation_v1(path)


def validate_workhorse_selected_world_observation_v1(
    observation_path: str | Path,
    *,
    expected_checkpoint_occurrence_sha256: str | None = None,
    expected_checkpoint_occurrence_projection_sha256: str | None = None,
) -> dict[str, Any]:
    """Reopen the observation and rederive it from retained source authority."""

    requested = Path(observation_path).expanduser()
    if requested.is_symlink():
        _fail("selected-world observation is a symbolic link")
    path = requested.resolve()
    info = path.stat()
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_mode & 0o222
        or info.st_size <= 0
        or info.st_size > _MAXIMUM_OBSERVATION_BYTES
    ):
        _fail("selected-world observation file identity is invalid")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("ascii"))
    except (UnicodeError, ValueError) as exc:
        raise WorkhorseSelectedWorldObservationError(
            f"selected-world observation JSON is invalid: {exc}"
        ) from exc
    if not isinstance(value, dict) or payload != _canonical_json_bytes(value):
        _fail("selected-world observation is not canonical JSON")
    expected_fields = {
        "schema_version",
        "observation_mode",
        "claim_class",
        "time_authority",
        "observation_boundary",
        "execution_authority_granted",
        "qualification_authority_granted",
        "named_phenotype_reward_authority_granted",
        "cell_id",
        "source_continuation",
        "terminal_checkpoint",
        "control_readback",
        "event_observation_coverage",
        "world_observations",
        "integrity",
        "observation_content_sha256",
    }
    body = {key: item for key, item in value.items() if key != "observation_content_sha256"}
    if (
        set(value) != expected_fields
        or value.get("schema_version")
        != WORKHORSE_SELECTED_WORLD_OBSERVATION_SCHEMA_V1
        or value.get("observation_mode")
        != WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1
        or value.get("claim_class")
        != WORKHORSE_SELECTED_WORLD_OBSERVATION_CLAIM_CLASS_V1
        or value.get("time_authority") != "absolute_tick_v1"
        or value.get("execution_authority_granted") is not False
        or value.get("qualification_authority_granted") is not False
        or value.get("named_phenotype_reward_authority_granted") is not False
        or value.get("observation_content_sha256") != canonical_sha256(body)
    ):
        _fail("selected-world observation contract or authority changed")
    source = value.get("source_continuation")
    terminal = value.get("terminal_checkpoint")
    if not isinstance(source, Mapping) or not isinstance(terminal, Mapping):
        _fail("selected-world observation source binding is absent")
    if (
        expected_checkpoint_occurrence_sha256 is not None
        and terminal.get("checkpoint_occurrence_sha256")
        != expected_checkpoint_occurrence_sha256
    ) or (
        expected_checkpoint_occurrence_projection_sha256 is not None
        and terminal.get("checkpoint_occurrence_projection_sha256")
        != expected_checkpoint_occurrence_projection_sha256
    ):
        _fail("selected-world observation names another terminal occurrence")
    rebuilt = _build_observation(
        execution_result_path=str(source.get("execution_result_path")),
        terminal_indexed_checkpoint_manifest_path=str(
            terminal.get("indexed_manifest_path")
        ),
        source_checkpoint_occurrence_sha256=str(
            terminal.get("checkpoint_occurrence_sha256")
        ),
        source_checkpoint_occurrence_projection_sha256=str(
            terminal.get("checkpoint_occurrence_projection_sha256")
        ),
        cell_id=str(value.get("cell_id")),
    )
    if rebuilt != value:
        _fail("selected-world observation does not rederive from its source")
    return {
        **value,
        "observation_path": str(path),
        "observation_file_sha256": _file_sha256(path),
        "observation_file_bytes": info.st_size,
    }


__all__ = (
    "WORKHORSE_SELECTED_WORLD_OBSERVATION_CLAIM_CLASS_V1",
    "WORKHORSE_SELECTED_WORLD_OBSERVATION_MODE_V1",
    "WORKHORSE_SELECTED_WORLD_OBSERVATION_SCHEMA_V1",
    "WorkhorseSelectedWorldObservationError",
    "project_workhorse_exact_state_observation_v1",
    "seal_workhorse_selected_world_observation_v1",
    "validate_workhorse_selected_world_observation_v1",
)
