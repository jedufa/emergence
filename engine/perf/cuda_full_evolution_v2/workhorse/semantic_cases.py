"""Canonical, callback-free Workhorse semantic-case contracts and raw fragments.

The 24 cases in this module are deliberately closed.  A physical case consumes
one parent Attempt binding created by the canonical provider; this module never
creates, subdivides, or refreshes Scheduler authority.  Terminalization is a
separate reopen step so an expected native failure cannot be confused with an
infrastructure failure or self-attested as a valid negative.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from typing import Any, Final

from .contract import (
    BACKEND_REFERENCE_PHASED_V2,
    WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
    WORKHORSE_TIME_AUTHORITY_V2,
    FullEvolutionCudaWorkhorseContractV2,
    canonical_sha256,
    workhorse_shape_v2,
)
from .layout import WorkhorseLayoutV2, workhorse_layout_v2
from .native_loader import file_sha256


WORKHORSE_SEMANTIC_CASE_COUNT_V2: Final[int] = 24
WORKHORSE_SEMANTIC_CASE_OPERATION_V2: Final[str] = "SEMANTIC_EVIDENCE_CASE"
WORKHORSE_SEMANTIC_CASE_FRAGMENT_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_semantic_case_raw_fragment_v2"
)
WORKHORSE_SEMANTIC_CASE_TERMINAL_RECEIPT_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_terminalized_semantic_case_receipt_v2"
)
WORKHORSE_SEMANTIC_P2_WITNESS_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_reused_event_buffer_canonicalization_witness_v2"
)
WORKHORSE_SCIENCE_BASE_CHECKPOINT_IDENTITY_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_science_compatible_base_checkpoint_identity_v2"
)
WORKHORSE_EXTERNAL_CHECKPOINT_SOURCE_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_external_checkpoint_source_binding_v2"
)
WORKHORSE_EXTERNAL_CHECKPOINT_AUTHORIZATION_SOURCE_V2: Final[str] = (
    "CRCA_WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_V2"
)
WORKHORSE_SCIENCE_CHECKPOINT_COMPATIBILITY_SCHEMA_V2: Final[str] = (
    "eiel_workhorse_science_checkpoint_compatibility_binding_v2"
)
SCIENCE_CONSUMER_EXTENSION_CONTRACT_RELATIVE_PATH_V2: Final[str] = (
    "_artifacts/tw-cuda-full-evolution-reform-20260825/"
    "science_consumer_extension_contract.json"
)

_RAW_COMPONENTS: Final[tuple[str, ...]] = (
    "input_state",
    "input_scratch",
    "input_events",
    "input_counts",
    "state",
    "scratch",
    "events",
    "counts",
)
_NO_CLAIM_AUTHORITY: Final[dict[str, bool]] = {
    "qualification_authority_granted": False,
    "science_authority_granted": False,
}
_OBSERVATION_OFF: Final[dict[str, Any]] = {
    "mode": "OFF",
    "science_records_emitted": 0,
    "science_observation_schedule": None,
    "population_change_detection": False,
}


class WorkhorseSemanticCaseError(RuntimeError):
    """A semantic case, fragment, or terminal result is not exact."""


@dataclass(frozen=True, slots=True)
class WorkhorseSemanticCaseLawV2:
    ordinal: int
    case_id: str
    artifact_role: str
    B: int
    dispatch_temporal_k: int
    total_ticks: int
    start_next_absolute_tick: int
    sequence_law: str
    fixture_law: str
    disable_mask: int
    fragment_datasets: tuple[str, ...]
    expected_native_outcome: str
    expected_attempt_terminal_status: str
    terminal_frontier_law: str
    requires_source_checkpoint: bool = False
    produces_base_checkpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["fragment_datasets"] = list(self.fragment_datasets)
        return value


def _case(
    ordinal: int,
    case_id: str,
    artifact_role: str,
    B: int,
    dispatch_temporal_k: int,
    total_ticks: int,
    start: int,
    sequence_law: str,
    fixture_law: str,
    disable_mask: int,
    datasets: tuple[str, ...],
    *,
    expected_native_outcome: str = "SUCCESS",
    expected_attempt_terminal_status: str = "COMPLETED",
    terminal_frontier_law: str = "ALL_TARGET",
    requires_source_checkpoint: bool = False,
    produces_base_checkpoint: bool = False,
) -> WorkhorseSemanticCaseLawV2:
    return WorkhorseSemanticCaseLawV2(
        ordinal=ordinal,
        case_id=case_id,
        artifact_role=artifact_role,
        B=B,
        dispatch_temporal_k=dispatch_temporal_k,
        total_ticks=total_ticks,
        start_next_absolute_tick=start,
        sequence_law=sequence_law,
        fixture_law=fixture_law,
        disable_mask=disable_mask,
        fragment_datasets=datasets,
        expected_native_outcome=expected_native_outcome,
        expected_attempt_terminal_status=expected_attempt_terminal_status,
        terminal_frontier_law=terminal_frontier_law,
        requires_source_checkpoint=requires_source_checkpoint,
        produces_base_checkpoint=produces_base_checkpoint,
    )


def workhorse_semantic_case_table_v2(
    semantic_temporal_k: int,
) -> tuple[WorkhorseSemanticCaseLawV2, ...]:
    """Return the exact 24-case semantic execution table for one compiled K."""

    if type(semantic_temporal_k) is not int or semantic_temporal_k <= 1:
        raise WorkhorseSemanticCaseError("semantic_temporal_k must be an integer >1")
    k = semantic_temporal_k
    cases = (
        _case(1, "parity_reference_k1_sequence", "REFERENCE", 2, 1, k, 71,
              "K1_SEQUENCE_COMBINED", "DETERMINISTIC_SEMANTIC", 0,
              ("parity_reference",)),
        _case(2, "parity_candidate_k_dispatch", "CANDIDATE", 2, k, k, 71,
              "ONE_K_DISPATCH", "DETERMINISTIC_SEMANTIC", 0,
              ("parity_candidate",)),
        _case(3, "b_split_world_0_k_dispatch", "CANDIDATE", 1, k, k, 71,
              "ONE_K_DISPATCH", "DETERMINISTIC_SEMANTIC_WORLD_0", 0,
              ("__b_split_world_0",)),
        _case(4, "b_split_world_1_k_dispatch", "CANDIDATE", 1, k, k, 71,
              "ONE_K_DISPATCH", "DETERMINISTIC_SEMANTIC_WORLD_1", 0,
              ("__b_split_world_1",)),
        _case(5, "k_split_candidate_k1_sequence", "CANDIDATE", 1, 1, k, 71,
              "K1_SEQUENCE_COMBINED", "DETERMINISTIC_SEMANTIC_WORLD_0", 0,
              ("k_split",)),
        _case(6, "causal_baseline", "CANDIDATE", 1, 1, 1, 71,
              "ONE_K1_DISPATCH", "DETERMINISTIC_SEMANTIC", 0,
              ("causal_baseline",)),
        _case(7, "causal_patch_disabled", "CANDIDATE", 1, 1, 1, 71,
              "ONE_K1_DISPATCH", "DETERMINISTIC_SEMANTIC", 1 << 0,
              ("causal_patch_disabled",)),
        _case(8, "causal_wdl_disabled", "CANDIDATE", 1, 1, 1, 71,
              "ONE_K1_DISPATCH", "DETERMINISTIC_SEMANTIC", 1 << 1,
              ("causal_wdl_disabled",)),
        _case(9, "causal_jungle_disabled", "CANDIDATE", 1, 1, 1, 71,
              "ONE_K1_DISPATCH", "DETERMINISTIC_SEMANTIC", 1 << 2,
              ("causal_jungle_disabled",)),
        _case(10, "causal_eed_disabled", "CANDIDATE", 1, 1, 1, 71,
              "ONE_K1_DISPATCH", "DETERMINISTIC_SEMANTIC", 1 << 3,
              ("causal_eed_disabled",)),
        _case(11, "causal_birth_disabled", "CANDIDATE", 1, 1, 1, 71,
              "ONE_K1_DISPATCH", "DETERMINISTIC_SEMANTIC", 1 << 4,
              ("causal_birth_disabled",)),
        _case(12, "causal_recombination_disabled", "CANDIDATE", 1, 1, 1, 71,
              "ONE_K1_DISPATCH", "DETERMINISTIC_SEMANTIC", (1 << 5) | (1 << 6),
              ("causal_recombination_disabled",)),
        _case(13, "eed_enabled_k1_sequence", "CANDIDATE", 1, 1, 2, 71,
              "K1_SEQUENCE_PREFIX_AND_COMBINED", "DETERMINISTIC_SEMANTIC", 1 << 5,
              ("causal_mutation_disabled", "eed_enabled")),
        _case(14, "eed_disabled_k1_sequence", "CANDIDATE", 1, 1, 2, 71,
              "K1_SEQUENCE_COMBINED", "DETERMINISTIC_SEMANTIC", (1 << 5) | (1 << 3),
              ("eed_disabled",)),
        _case(15, "wdl_k1_sequence", "CANDIDATE", 1, 1, 3, 0,
              "K1_SEQUENCE_PREFIXES", "ZERO_WORLD", 0,
              ("wdl_tick1", "wdl_tick2", "wdl_tick3")),
        _case(16, "overflow_reference", "REFERENCE", 1, 1, 1, 9,
              "ONE_K1_DISPATCH", "BIRTH_OVERFLOW", 0,
              ("overflow_reference",)),
        _case(17, "overflow_candidate", "CANDIDATE", 1, 1, 1, 9,
              "ONE_K1_DISPATCH", "BIRTH_OVERFLOW", 0,
              ("overflow_candidate",)),
        _case(18, "middle_survivor_reference", "REFERENCE", 1, 1, 1, 31,
              "ONE_K1_DISPATCH", "MIDDLE_SURVIVOR_DEATH", 0,
              ("middle_survivor_reference",)),
        _case(19, "middle_survivor_candidate", "CANDIDATE", 1, 1, 1, 31,
              "ONE_K1_DISPATCH", "MIDDLE_SURVIVOR_DEATH", 0,
              ("middle_survivor_candidate",)),
        _case(20, "checkpoint_same_restore_sequence", "CANDIDATE", 1, 1, 2, 21,
              "CHECKPOINT_RECONSTRUCT_AND_CONTINUE", "DETERMINISTIC_SEMANTIC", 0,
              ("restore_checkpoint", "restore_reconstructed", "restore_same_continuation"),
              produces_base_checkpoint=True),
        _case(21, "fresh_restore_continuation", "CANDIDATE", 1, 1, 1, 22,
              "EXTERNAL_CHECKPOINT_RESTORE_AND_CONTINUE", "EXTERNAL_CHECKPOINT", 0,
              ("restore_fresh_continuation",), requires_source_checkpoint=True),
        _case(22, "invalid_restore_failstop", "CANDIDATE", 1, 1, 1, 22,
              "INVALID_RESTORE_THEN_FAIL_STOP", "CORRUPTED_ACTIVE_ORDER", 0,
              ("restore_invalid_before", "restore_invalid_after"),
              expected_native_outcome="EXPECTED_INVALID_RESTORE_FAILURE",
              expected_attempt_terminal_status="FAILED",
              terminal_frontier_law="ALL_START", requires_source_checkpoint=True),
        _case(23, "transaction_failstop", "CANDIDATE", 1, 2, 2, 13,
              "ONE_K2_FAILED_DISPATCH", "TRANSACTION_ERROR", 0,
              ("transaction_after",),
              expected_native_outcome="EXPECTED_TRANSACTION_FAILURE",
              expected_attempt_terminal_status="FAILED",
              terminal_frontier_law="DISPATCHED_TARGET_OTHERS_START"),
        _case(24, "event_reuse_canonicalization_regression", "CANDIDATE", 1, k, 2 * k, 41,
              "REUSED_OWNER_SUCCESS_ACK_THEN_FAILED_K_DISPATCH", "TRANSACTION_ERROR_AFTER_SUCCESS", 0,
              (), expected_native_outcome="EXPECTED_REUSED_OWNER_TRANSACTION_FAILURE",
              expected_attempt_terminal_status="FAILED",
              terminal_frontier_law="FIRST_DISPATCH_COMMITTED_SECOND_DISPATCH_FAILED"),
    )
    if (
        len(cases) != WORKHORSE_SEMANTIC_CASE_COUNT_V2
        or tuple(case.ordinal for case in cases) != tuple(range(1, 25))
        or len({case.case_id for case in cases}) != 24
    ):
        raise RuntimeError("Workhorse semantic case inventory is not exact")
    return cases


def workhorse_semantic_case_law_v2(
    case_id: str, *, semantic_temporal_k: int
) -> WorkhorseSemanticCaseLawV2:
    matches = [
        case
        for case in workhorse_semantic_case_table_v2(semantic_temporal_k)
        if case.case_id == case_id
    ]
    if len(matches) != 1:
        raise WorkhorseSemanticCaseError(f"unknown semantic case {case_id!r}")
    return matches[0]


def workhorse_semantic_case_table_sha256_v2(semantic_temporal_k: int) -> str:
    return canonical_sha256(
        [case.to_dict() for case in workhorse_semantic_case_table_v2(semantic_temporal_k)]
    )


def workhorse_semantic_case_terminal_contract_v2(
    law: WorkhorseSemanticCaseLawV2,
) -> dict[str, Any]:
    """Derive the exact Scheduler terminal projection from one closed case law."""

    start = law.start_next_absolute_tick
    target = start + law.total_ticks
    if law.terminal_frontier_law == "ALL_TARGET":
        values = (target, target, target, target)
    elif law.terminal_frontier_law == "ALL_START":
        values = (start, start, start, start)
    elif law.terminal_frontier_law == "DISPATCHED_TARGET_OTHERS_START":
        values = (target, start, start, start)
    elif law.terminal_frontier_law == "FIRST_DISPATCH_COMMITTED_SECOND_DISPATCH_FAILED":
        midpoint = start + law.dispatch_temporal_k
        values = (target, midpoint, midpoint, midpoint)
    else:  # pragma: no cover - closed table construction prevents this
        raise WorkhorseSemanticCaseError("unknown semantic terminal frontier law")
    return {
        "attempt_terminal_status": law.expected_attempt_terminal_status,
        "final_frontiers": {
            "schema_version": "crca_attempt_frontiers_v1",
            "last_dispatched_next_tick": values[0],
            "last_computed_next_tick": values[1],
            "durable_checkpoint_next_tick": values[2],
            "published_artifact_next_tick": values[3],
        },
    }


def workhorse_semantic_case_operation_contract_v2(
    *, case_id: str, shape_id: str, semantic_temporal_k: int
) -> dict[str, Any]:
    shape = workhorse_shape_v2(shape_id)
    if semantic_temporal_k not in shape.temporal_k_candidates:
        raise WorkhorseSemanticCaseError("semantic K is not compiled for the shape")
    law = workhorse_semantic_case_law_v2(
        case_id, semantic_temporal_k=semantic_temporal_k
    )
    body = {
        "schema_version": "eiel_workhorse_semantic_case_operation_contract_v2",
        "operation": WORKHORSE_SEMANTIC_CASE_OPERATION_V2,
        "case_law": law.to_dict(),
        "case_table_sha256": workhorse_semantic_case_table_sha256_v2(
            semantic_temporal_k
        ),
        "attempt_terminal_contract": workhorse_semantic_case_terminal_contract_v2(
            law
        ),
        "shape_id": shape.shape_id,
        "semantic_temporal_k": semantic_temporal_k,
        "semantic_transition_sha256": WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "observation": dict(_OBSERVATION_OFF),
        **_NO_CLAIM_AUTHORITY,
    }
    return {**body, "operation_contract_sha256": canonical_sha256(body)}


def validate_workhorse_semantic_case_parent_binding_v2(
    value: Mapping[str, Any],
    *,
    case_id: str,
    semantic_temporal_k: int,
    enforce_current_process: bool = True,
) -> dict[str, Any]:
    """Reopen a provider-created parent binding; never manufacture authority."""

    try:
        from .canonical_rows import (
            validate_workhorse_canonical_parent_authority_binding_v2,
        )

        parent = validate_workhorse_canonical_parent_authority_binding_v2(
            value, expected_operation=WORKHORSE_SEMANTIC_CASE_OPERATION_V2
        )
        contract = FullEvolutionCudaWorkhorseContractV2.from_mapping(
            parent["contract"]
        )
    except Exception as exc:
        raise WorkhorseSemanticCaseError(
            f"semantic case parent is not canonical: {exc}"
        ) from exc
    law = workhorse_semantic_case_law_v2(
        case_id, semantic_temporal_k=semantic_temporal_k
    )
    operation_contract = workhorse_semantic_case_operation_contract_v2(
        case_id=case_id,
        shape_id=contract.shape_id,
        semantic_temporal_k=semantic_temporal_k,
    )
    expected_backend = (
        BACKEND_REFERENCE_PHASED_V2
        if law.artifact_role == "REFERENCE"
        else parent["backend_id"]
    )
    source = parent["source_checkpoint_identity_sha256"]
    if (
        contract.B != law.B
        or contract.temporal_k != law.dispatch_temporal_k
        or contract.total_ticks != law.total_ticks
        or contract.start_next_tick != law.start_next_absolute_tick
        or parent["backend_id"] != expected_backend
        or parent["operation_contract_sha256"]
        != operation_contract["operation_contract_sha256"]
        or (source is not None) != law.requires_source_checkpoint
    ):
        raise WorkhorseSemanticCaseError(
            "semantic case parent differs from the closed operation law"
        )
    if enforce_current_process and parent["owner_pid"] != os.getpid():
        raise WorkhorseSemanticCaseError(
            "semantic case parent belongs to a different process"
        )
    return parent


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exact_mapping(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise WorkhorseSemanticCaseError(f"{label} fields are inexact")
    return value


def _expected_component_bytes(
    layout: WorkhorseLayoutV2, *, B: int, temporal_k: int
) -> dict[str, int]:
    output = {
        "state": B * layout.state_bytes_per_world,
        "scratch": B * layout.scratch_bytes_per_world,
        "events": B * temporal_k * layout.event_rows_per_world_tick * 48,
        "counts": B * 4,
    }
    return {**{f"input_{key}": size for key, size in output.items()}, **output}


def _dataset_coordinates(
    name: str, *, law: WorkhorseSemanticCaseLawV2, semantic_temporal_k: int
) -> tuple[int, int, int]:
    if name in {
        "parity_reference",
        "parity_candidate",
        "__b_split_world_0",
        "__b_split_world_1",
        "k_split",
    }:
        return law.B, semantic_temporal_k, 71
    if name in {"causal_mutation_disabled"}:
        return 1, 1, 71
    if name in {"eed_enabled", "eed_disabled"}:
        return 1, 2, 71
    if name.startswith("wdl_tick"):
        return 1, 1, int(name[-1]) - 1
    if name.startswith("overflow_"):
        return 1, 1, 9
    if name.startswith("middle_survivor_"):
        return 1, 1, 31
    if name in {"restore_checkpoint", "restore_reconstructed"}:
        return 1, 1, 21
    if name in {
        "restore_same_continuation",
        "restore_fresh_continuation",
        "restore_invalid_before",
        "restore_invalid_after",
    }:
        return 1, 1, 22
    if name == "transaction_after":
        return 1, 2, 13
    return law.B, law.dispatch_temporal_k, law.start_next_absolute_tick


def science_compatible_base_checkpoint_identity_v2(
    *,
    shape_id: str,
    source_artifact_identity_sha256: str,
    checkpoint_at_tick: int,
    state_bytes: bytes,
    world_uids: Sequence[int],
    rng_keys: Sequence[int],
    event_sink_acknowledgement: Mapping[str, Any],
    science_checkpoint_compatibility_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an exact event-occurrence-bound identity reusable by child Attempts."""

    layout = workhorse_layout_v2(workhorse_shape_v2(shape_id))
    uids, keys = list(world_uids), list(rng_keys)
    if (
        not _valid_sha(source_artifact_identity_sha256)
        or type(checkpoint_at_tick) is not int
        or checkpoint_at_tick < 0
        or type(state_bytes) is not bytes
        or len(state_bytes) != len(uids) * layout.state_bytes_per_world
        or not uids
        or len(uids) != len(keys)
        or len(set(uids)) != len(uids)
        or any(type(uid) is not int or uid <= 0 for uid in uids)
        or any(type(key) is not int or key < 0 or key >= 1 << 64 for key in keys)
    ):
        raise WorkhorseSemanticCaseError("base checkpoint identity input is invalid")
    science_compatibility = validate_workhorse_science_checkpoint_compatibility_binding_v2(
        science_checkpoint_compatibility_binding
    )
    ack_required = {
        "receipt_path",
        "receipt_file_sha256",
        "receipt_body_sha256",
        "dispatch_content_sha256",
        "event_rows_file_sha256",
        "event_counts_file_sha256",
        "world_uids",
        "start_next_absolute_tick",
        "end_next_absolute_tick",
        "artifact_identity_sha256",
        "rng_identity_sha256",
        "post_state_sha256",
        "dispatch_identity_sha256",
    }
    if not ack_required.issubset(event_sink_acknowledgement):
        raise WorkhorseSemanticCaseError(
            "base checkpoint lacks the exact acknowledged event frontier"
        )
    ack = {key: event_sink_acknowledgement[key] for key in sorted(ack_required)}
    ack_path = Path(ack["receipt_path"])
    try:
        from .authority import validate_workhorse_event_sink_receipt_v2

        reopened_ack = validate_workhorse_event_sink_receipt_v2(
            ack_path,
            expected_artifact_identity_sha256=source_artifact_identity_sha256,
            expected_end_next_absolute_tick=checkpoint_at_tick,
            expected_post_state_sha256=_sha_bytes(state_bytes),
        )
    except Exception as exc:
        raise WorkhorseSemanticCaseError(
            f"base checkpoint event ACK cannot be reopened: {exc}"
        ) from exc
    reopened_projection = {key: reopened_ack[key] for key in sorted(ack_required)}
    if (
        ack_path.is_symlink()
        or not ack_path.is_file()
        or ack_path.stat().st_mode & 0o222
        or file_sha256(ack_path) != ack["receipt_file_sha256"]
        or any(
            not _valid_sha(ack[field])
            for field in (
                "receipt_file_sha256",
                "receipt_body_sha256",
                "dispatch_content_sha256",
                "event_rows_file_sha256",
                "event_counts_file_sha256",
                "artifact_identity_sha256",
                "rng_identity_sha256",
                "post_state_sha256",
                "dispatch_identity_sha256",
            )
        )
        or ack["world_uids"] != uids
        or ack["end_next_absolute_tick"] != checkpoint_at_tick
        or type(ack["start_next_absolute_tick"]) is not int
        or ack["start_next_absolute_tick"] >= checkpoint_at_tick
        or ack["artifact_identity_sha256"] != source_artifact_identity_sha256
        or ack["post_state_sha256"] != _sha_bytes(state_bytes)
        or ack != reopened_projection
    ):
        raise WorkhorseSemanticCaseError(
            "base checkpoint event-frontier acknowledgement is not exact"
        )
    observed_uids: list[int] = []
    observed_ticks: list[int] = []
    observed_keys: list[int] = []
    for world in range(len(uids)):
        offset = world * layout.state_bytes_per_world
        header = struct.unpack_from("<32Q", state_bytes, offset)
        observed_uids.append(header[0])
        observed_ticks.append(header[1])
        observed_keys.append(header[13])
        if header[4] != 0:
            raise WorkhorseSemanticCaseError(
                "base checkpoint cannot bind an error status"
            )
    if (
        observed_uids != uids
        or observed_keys != keys
        or observed_ticks != [checkpoint_at_tick] * len(uids)
    ):
        raise WorkhorseSemanticCaseError(
            "base checkpoint state identity/frontier differs from metadata"
        )
    body = {
        "schema_version": WORKHORSE_SCIENCE_BASE_CHECKPOINT_IDENTITY_SCHEMA_V2,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "shape_id": shape_id,
        "layout_sha256": layout.layout_sha256,
        "semantic_transition_sha256": WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
        "source_artifact_identity_sha256": source_artifact_identity_sha256,
        "checkpoint_at_tick": checkpoint_at_tick,
        "B": len(uids),
        "world_uids": uids,
        "rng_keys": keys,
        "state_bytes": len(state_bytes),
        "state_sha256": _sha_bytes(state_bytes),
        "last_acknowledged_event_frontier": ack,
        "science_checkpoint_compatibility": science_compatibility,
        "observation": dict(_OBSERVATION_OFF),
        **_NO_CLAIM_AUTHORITY,
    }
    return {**body, "base_checkpoint_identity_sha256": canonical_sha256(body)}


def _validate_base_checkpoint_identity_v2(value: object) -> dict[str, Any]:
    identity = _exact_mapping(
        value,
        {
            "schema_version",
            "time_authority",
            "shape_id",
            "layout_sha256",
            "semantic_transition_sha256",
            "source_artifact_identity_sha256",
            "checkpoint_at_tick",
            "B",
            "world_uids",
            "rng_keys",
            "state_bytes",
            "state_sha256",
            "last_acknowledged_event_frontier",
            "science_checkpoint_compatibility",
            "observation",
            "qualification_authority_granted",
            "science_authority_granted",
            "base_checkpoint_identity_sha256",
        },
        "base checkpoint identity",
    )
    shape = workhorse_shape_v2(identity["shape_id"])
    layout = workhorse_layout_v2(shape)
    body = {
        key: item
        for key, item in identity.items()
        if key != "base_checkpoint_identity_sha256"
    }
    science_compatibility = validate_workhorse_science_checkpoint_compatibility_binding_v2(
        identity["science_checkpoint_compatibility"]
    )
    ack = identity["last_acknowledged_event_frontier"]
    if type(ack) is not dict:
        raise WorkhorseSemanticCaseError("base checkpoint event frontier is absent")
    ack_path = Path(ack.get("receipt_path", ""))
    if (
        identity["schema_version"]
        != WORKHORSE_SCIENCE_BASE_CHECKPOINT_IDENTITY_SCHEMA_V2
        or identity["time_authority"] != WORKHORSE_TIME_AUTHORITY_V2
        or identity["layout_sha256"] != layout.layout_sha256
        or identity["semantic_transition_sha256"]
        != WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2
        or not _valid_sha(identity["source_artifact_identity_sha256"])
        or type(identity["checkpoint_at_tick"]) is not int
        or identity["checkpoint_at_tick"] < 0
        or type(identity["B"]) is not int
        or identity["B"] <= 0
        or type(identity["world_uids"]) is not list
        or type(identity["rng_keys"]) is not list
        or len(identity["world_uids"]) != identity["B"]
        or len(identity["rng_keys"]) != identity["B"]
        or len(set(identity["world_uids"])) != identity["B"]
        or any(type(uid) is not int or uid <= 0 for uid in identity["world_uids"])
        or any(
            type(key) is not int or key < 0 or key >= 1 << 64
            for key in identity["rng_keys"]
        )
        or identity["state_bytes"] != identity["B"] * layout.state_bytes_per_world
        or not _valid_sha(identity["state_sha256"])
        or ack.get("world_uids") != identity["world_uids"]
        or ack.get("end_next_absolute_tick") != identity["checkpoint_at_tick"]
        or ack.get("artifact_identity_sha256")
        != identity["source_artifact_identity_sha256"]
        or ack.get("post_state_sha256") != identity["state_sha256"]
        or not _valid_sha(ack.get("receipt_file_sha256"))
        or ack_path.is_symlink()
        or not ack_path.is_file()
        or ack_path.stat().st_mode & 0o222
        or file_sha256(ack_path) != ack["receipt_file_sha256"]
        or identity["science_checkpoint_compatibility"] != science_compatibility
        or identity["observation"] != _OBSERVATION_OFF
        or identity["qualification_authority_granted"] is not False
        or identity["science_authority_granted"] is not False
        or identity["base_checkpoint_identity_sha256"] != canonical_sha256(body)
    ):
        raise WorkhorseSemanticCaseError("base checkpoint identity changed")
    return identity


def workhorse_science_checkpoint_compatibility_binding_v2(
    project_root: str | Path,
) -> dict[str, Any]:
    """Bind the existing opaque science-continuation consumer contract."""

    root = Path(project_root).resolve()
    contract = root / SCIENCE_CONSUMER_EXTENSION_CONTRACT_RELATIVE_PATH_V2
    if contract.is_symlink() or not contract.is_file():
        raise WorkhorseSemanticCaseError(
            "science consumer extension contract is missing or a symlink"
        )
    body = {
        "schema_version": WORKHORSE_SCIENCE_CHECKPOINT_COMPATIBILITY_SCHEMA_V2,
        "science_consumer_extension_contract_path": (
            SCIENCE_CONSUMER_EXTENSION_CONTRACT_RELATIVE_PATH_V2
        ),
        "science_consumer_extension_contract_sha256": file_sha256(contract),
        "science_extension_contract_schema": (
            "eiel_cuda_full_evolution_science_extension_contract_v1"
        ),
        "science_checkpoint_extension_schema": (
            "eiel_cuda_full_evolution_science_checkpoint_extension_v1"
        ),
        "science_checkpoint_identity_schema": (
            "eiel_cuda_full_evolution_science_checkpoint_identity_v1"
        ),
        "science_restore_gate_schema": (
            "eiel_cuda_full_evolution_science_restore_gate_v1"
        ),
        "science_route_binding_adapter_schema": (
            "eiel_cuda_full_evolution_science_route_binding_adapter_v1"
        ),
        "science_route_binding_wire_schema": "eiel_science_route_binding_v1",
        "science_observation_record_schema": "eiel_science_observation_record_v1",
        "science_observation_prefix_cursor_schema": (
            "eiel_science_observation_prefix_cursor_v1"
        ),
        "continuation_adapter_version": (
            "eiel_cuda_full_evolution_science_checkpoint_continuation_adapter_v1"
        ),
        "observation_adapter_version": (
            "eiel_cuda_full_evolution_science_observation_adapter_v1"
        ),
        "runtime_observation_mode": "OFF",
        "native_capture_runtime_integrated": False,
        "science_route_qualified": False,
    }
    return {**body, "binding_sha256": canonical_sha256(body)}


def validate_workhorse_science_checkpoint_compatibility_binding_v2(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _exact_mapping(
        value,
        {
            "schema_version",
            "science_consumer_extension_contract_path",
            "science_consumer_extension_contract_sha256",
            "science_extension_contract_schema",
            "science_checkpoint_extension_schema",
            "science_checkpoint_identity_schema",
            "science_restore_gate_schema",
            "science_route_binding_adapter_schema",
            "science_route_binding_wire_schema",
            "science_observation_record_schema",
            "science_observation_prefix_cursor_schema",
            "continuation_adapter_version",
            "observation_adapter_version",
            "runtime_observation_mode",
            "native_capture_runtime_integrated",
            "science_route_qualified",
            "binding_sha256",
        },
        "science checkpoint compatibility binding",
    )
    core = {key: item for key, item in binding.items() if key != "binding_sha256"}
    contract_path = (
        Path(__file__).resolve().parents[4]
        / SCIENCE_CONSUMER_EXTENSION_CONTRACT_RELATIVE_PATH_V2
    )
    if (
        binding["schema_version"]
        != WORKHORSE_SCIENCE_CHECKPOINT_COMPATIBILITY_SCHEMA_V2
        or binding["science_consumer_extension_contract_path"]
        != SCIENCE_CONSUMER_EXTENSION_CONTRACT_RELATIVE_PATH_V2
        or not _valid_sha(binding["science_consumer_extension_contract_sha256"])
        or contract_path.is_symlink()
        or not contract_path.is_file()
        or file_sha256(contract_path)
        != binding["science_consumer_extension_contract_sha256"]
        or binding["science_extension_contract_schema"]
        != "eiel_cuda_full_evolution_science_extension_contract_v1"
        or binding["science_checkpoint_extension_schema"]
        != "eiel_cuda_full_evolution_science_checkpoint_extension_v1"
        or binding["science_checkpoint_identity_schema"]
        != "eiel_cuda_full_evolution_science_checkpoint_identity_v1"
        or binding["science_restore_gate_schema"]
        != "eiel_cuda_full_evolution_science_restore_gate_v1"
        or binding["science_route_binding_adapter_schema"]
        != "eiel_cuda_full_evolution_science_route_binding_adapter_v1"
        or binding["science_route_binding_wire_schema"]
        != "eiel_science_route_binding_v1"
        or binding["science_observation_record_schema"]
        != "eiel_science_observation_record_v1"
        or binding["science_observation_prefix_cursor_schema"]
        != "eiel_science_observation_prefix_cursor_v1"
        or binding["continuation_adapter_version"]
        != "eiel_cuda_full_evolution_science_checkpoint_continuation_adapter_v1"
        or binding["observation_adapter_version"]
        != "eiel_cuda_full_evolution_science_observation_adapter_v1"
        or binding["runtime_observation_mode"] != "OFF"
        or binding["native_capture_runtime_integrated"] is not False
        or binding["science_route_qualified"] is not False
        or binding["binding_sha256"] != canonical_sha256(core)
    ):
        raise WorkhorseSemanticCaseError(
            "science checkpoint compatibility binding changed"
        )
    return binding


def validate_external_checkpoint_source_binding_v2(
    value: Mapping[str, Any], *, expected_identity_sha256: str
) -> dict[str, Any]:
    binding = _exact_mapping(
        value,
        {
            "schema_version",
            "authorization_source",
            "checkpoint_occurrence_record_path",
            "checkpoint_occurrence_record_file_sha256",
            "checkpoint_occurrence_identity_sha256",
            "base_checkpoint_manifest_path",
            "base_checkpoint_manifest_file_sha256",
            "base_checkpoint_identity_sha256",
            "binding_sha256",
        },
        "external checkpoint source binding",
    )
    if (
        binding["schema_version"] != WORKHORSE_EXTERNAL_CHECKPOINT_SOURCE_SCHEMA_V2
        or binding["authorization_source"]
        != WORKHORSE_EXTERNAL_CHECKPOINT_AUTHORIZATION_SOURCE_V2
        or binding["base_checkpoint_identity_sha256"] != expected_identity_sha256
    ):
        raise WorkhorseSemanticCaseError(
            "external checkpoint source identity/authority changed"
        )
    for field in (
        "checkpoint_occurrence_record_file_sha256",
        "checkpoint_occurrence_identity_sha256",
        "base_checkpoint_manifest_file_sha256",
        "base_checkpoint_identity_sha256",
    ):
        if not _valid_sha(binding[field]):
            raise WorkhorseSemanticCaseError(
                f"external checkpoint {field} is invalid"
            )
    for path_field, hash_field in (
        ("checkpoint_occurrence_record_path", "checkpoint_occurrence_record_file_sha256"),
        ("base_checkpoint_manifest_path", "base_checkpoint_manifest_file_sha256"),
    ):
        path = Path(binding[path_field])
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_mode & 0o222
            or file_sha256(path) != binding[hash_field]
        ):
            raise WorkhorseSemanticCaseError(
                f"external checkpoint {path_field} cannot be reopened exactly"
            )
    core = {key: item for key, item in binding.items() if key != "binding_sha256"}
    if binding["binding_sha256"] != canonical_sha256(core):
        raise WorkhorseSemanticCaseError("external checkpoint binding seal changed")
    return binding


def _validate_p2_witness(value: object, *, semantic_temporal_k: int) -> dict[str, Any]:
    witness = _exact_mapping(
        value,
        {
            "schema_version",
            "B",
            "temporal_k",
            "successful_dispatch_start_tick",
            "successful_dispatch_end_tick",
            "successful_ack_receipt_path",
            "successful_ack_receipt_file_sha256",
            "failed_event_acknowledged",
            "failed_event_publication_path",
            "failed_dispatch_pending_preserved",
            "poisoned_status",
            "post_failure_reuse_rejected",
            "failed_event_buffer_cannot_be_reused",
            "canonicalization_law",
            "failed_dispatch_start_tick",
            "failed_dispatch_requested_end_tick",
            "failed_status",
            "failed_frontier",
            "active_rows_before_canonicalization",
            "active_rows_after_canonicalization",
            "active_row_bytes_sha256_before_canonicalization",
            "active_row_bytes_sha256_after_canonicalization",
            "active_row_bytes_hex_before_canonicalization",
            "active_row_bytes_hex_after_canonicalization",
            "active_row_absolute_ticks",
            "active_rows_match_failed_absolute_tick",
            "inactive_rows_exact_zero_after_canonicalization",
            "later_k_slices_exact_zero_after_canonicalization",
            "active_rows_byte_preserved",
            "same_owner_storage_data_ptr",
            "successful_ack_preceded_failed_reuse",
            "witness_sha256",
        },
        "P2 reused-buffer witness",
    )
    body = {key: item for key, item in witness.items() if key != "witness_sha256"}
    try:
        before_bytes = bytes.fromhex(
            witness["active_row_bytes_hex_before_canonicalization"]
        )
        after_bytes = bytes.fromhex(
            witness["active_row_bytes_hex_after_canonicalization"]
        )
    except (TypeError, ValueError) as exc:
        raise WorkhorseSemanticCaseError("P2 active-row bytes are invalid") from exc
    successful_ack = Path(witness["successful_ack_receipt_path"])
    try:
        from .authority import validate_workhorse_event_sink_receipt_v2

        reopened_successful_ack = validate_workhorse_event_sink_receipt_v2(
            successful_ack,
            expected_end_next_absolute_tick=witness[
                "successful_dispatch_end_tick"
            ],
        )
    except Exception as exc:
        raise WorkhorseSemanticCaseError(
            f"P2 successful dispatch ACK cannot be reopened: {exc}"
        ) from exc
    if (
        witness["schema_version"] != WORKHORSE_SEMANTIC_P2_WITNESS_SCHEMA_V2
        or witness["B"] != 1
        or witness["temporal_k"] != semantic_temporal_k
        or witness["successful_dispatch_end_tick"]
        != witness["successful_dispatch_start_tick"] + semantic_temporal_k
        or witness["failed_dispatch_start_tick"]
        != witness["successful_dispatch_end_tick"]
        or witness["failed_dispatch_requested_end_tick"]
        != witness["failed_dispatch_start_tick"] + semantic_temporal_k
        or type(witness["failed_status"]) is not int
        or witness["failed_status"] == 0
        or witness["failed_frontier"] != witness["failed_dispatch_start_tick"]
        or type(witness["active_rows_before_canonicalization"]) is not int
        or witness["active_rows_before_canonicalization"] <= 0
        or witness["active_rows_after_canonicalization"]
        != witness["active_rows_before_canonicalization"]
        or witness["active_row_bytes_sha256_before_canonicalization"]
        != witness["active_row_bytes_sha256_after_canonicalization"]
        or not _valid_sha(witness["successful_ack_receipt_file_sha256"])
        or successful_ack.is_symlink()
        or not successful_ack.is_file()
        or successful_ack.stat().st_mode & 0o222
        or file_sha256(successful_ack)
        != witness["successful_ack_receipt_file_sha256"]
        or reopened_successful_ack.get("receipt_path")
        != str(successful_ack.resolve())
        or reopened_successful_ack.get("receipt_file_sha256")
        != witness["successful_ack_receipt_file_sha256"]
        or reopened_successful_ack.get("start_next_absolute_tick")
        != witness["successful_dispatch_start_tick"]
        or reopened_successful_ack.get("end_next_absolute_tick")
        != witness["successful_dispatch_end_tick"]
        or reopened_successful_ack.get("tick_count") != semantic_temporal_k
        or len(reopened_successful_ack.get("world_uids", ())) != witness["B"]
        or witness["failed_event_acknowledged"] is not False
        or witness["failed_event_publication_path"] is not None
        or witness["failed_dispatch_pending_preserved"] is not True
        or type(witness["poisoned_status"]) is not list
        or witness["poisoned_status"] != [witness["failed_status"]]
        or witness["post_failure_reuse_rejected"] is not True
        or witness["failed_event_buffer_cannot_be_reused"] is not True
        or witness["canonicalization_law"]
        != "NATIVE_CANONICALIZATION_WITHOUT_ACK_OR_PUBLICATION"
        or not _valid_sha(
            witness["active_row_bytes_sha256_before_canonicalization"]
        )
        or len(before_bytes)
        != 48 * witness["active_rows_before_canonicalization"]
        or before_bytes != after_bytes
        or _sha_bytes(before_bytes)
        != witness["active_row_bytes_sha256_before_canonicalization"]
        or _sha_bytes(after_bytes)
        != witness["active_row_bytes_sha256_after_canonicalization"]
        or type(witness["active_row_absolute_ticks"]) is not list
        or len(witness["active_row_absolute_ticks"])
        != witness["active_rows_after_canonicalization"]
        or any(
            tick != witness["failed_dispatch_start_tick"]
            for tick in witness["active_row_absolute_ticks"]
        )
        or witness["active_rows_match_failed_absolute_tick"] is not True
        or any(
            witness[field] is not True
            for field in (
                "inactive_rows_exact_zero_after_canonicalization",
                "later_k_slices_exact_zero_after_canonicalization",
                "active_rows_byte_preserved",
                "same_owner_storage_data_ptr",
                "successful_ack_preceded_failed_reuse",
            )
        )
        or witness["witness_sha256"] != canonical_sha256(body)
    ):
        raise WorkhorseSemanticCaseError(
            "P2 reused-buffer witness does not prove the physical invariant"
        )
    return witness


def write_workhorse_semantic_case_fragment_v2(
    output_directory: str | Path,
    *,
    case_id: str,
    semantic_temporal_k: int,
    shape_id: str,
    artifact_sha256: str,
    canonical_parent_binding: Mapping[str, Any],
    native_outcome: str,
    datasets: Mapping[str, Mapping[str, Any]],
    base_checkpoint: Mapping[str, Any] | None = None,
    p2_witness: Mapping[str, Any] | None = None,
    external_source_checkpoint_binding: Mapping[str, Any] | None = None,
) -> Path:
    """Write, seal, and reopen one physical case's raw result."""

    law = workhorse_semantic_case_law_v2(
        case_id, semantic_temporal_k=semantic_temporal_k
    )
    parent = validate_workhorse_semantic_case_parent_binding_v2(
        canonical_parent_binding,
        case_id=case_id,
        semantic_temporal_k=semantic_temporal_k,
    )
    shape = workhorse_shape_v2(shape_id)
    layout = workhorse_layout_v2(shape)
    if (
        shape_id != parent["shape_id"]
        or not _valid_sha(artifact_sha256)
        or artifact_sha256 != parent["artifact_sha256"]
        or native_outcome != law.expected_native_outcome
        or set(datasets) != set(law.fragment_datasets)
        or (p2_witness is not None) != (case_id == "event_reuse_canonicalization_regression")
        or (base_checkpoint is not None) != law.produces_base_checkpoint
        or (external_source_checkpoint_binding is not None)
        != law.requires_source_checkpoint
    ):
        raise WorkhorseSemanticCaseError(
            "semantic case result differs from its parent/law"
        )
    if p2_witness is not None:
        _validate_p2_witness(p2_witness, semantic_temporal_k=semantic_temporal_k)
    source_checkpoint_projection = None
    if external_source_checkpoint_binding is not None:
        source_checkpoint_projection = validate_external_checkpoint_source_binding_v2(
            external_source_checkpoint_binding,
            expected_identity_sha256=parent[
                "source_checkpoint_identity_sha256"
            ],
        )
    root = Path(output_directory)
    if root.exists() or root.is_symlink():
        raise WorkhorseSemanticCaseError("semantic case fragment root must be fresh")
    root.mkdir(parents=True, mode=0o700)
    dataset_rows: dict[str, Any] = {}
    for name in law.fragment_datasets:
        raw = datasets[name]
        if type(raw) is not dict or set(raw) != {
            "B", "temporal_k", "start_next_absolute_tick", *_RAW_COMPONENTS
        }:
            raise WorkhorseSemanticCaseError(f"case dataset {name} fields are inexact")
        B, K, start = _dataset_coordinates(
            name, law=law, semantic_temporal_k=semantic_temporal_k
        )
        if (raw["B"], raw["temporal_k"], raw["start_next_absolute_tick"]) != (
            B, K, start
        ):
            raise WorkhorseSemanticCaseError(
                f"case dataset {name} coordinates differ from the closed law"
            )
        components: dict[str, Any] = {}
        for component, expected_bytes in _expected_component_bytes(
            layout, B=B, temporal_k=K
        ).items():
            payload = raw[component]
            if type(payload) is not bytes or len(payload) != expected_bytes:
                raise WorkhorseSemanticCaseError(
                    f"case dataset {name}.{component} byte length is invalid"
                )
            target = root / f"{name}.{component}.bin"
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            target.chmod(0o444)
            components[component] = {
                "file_name": target.name,
                "bytes": len(payload),
                "sha256": _sha_bytes(payload),
            }
        dataset_rows[name] = {
            "B": B,
            "temporal_k": K,
            "start_next_absolute_tick": start,
            "components": components,
        }
    checkpoint_projection = None
    if base_checkpoint is not None:
        base = _exact_mapping(
            base_checkpoint,
            {"identity", "state"},
            "base checkpoint payload",
        )
        identity = _validate_base_checkpoint_identity_v2(base["identity"])
        state = base["state"]
        if (
            type(state) is not bytes
            or len(state) != identity["state_bytes"]
            or _sha_bytes(state) != identity["state_sha256"]
            or identity["shape_id"] != shape_id
        ):
            raise WorkhorseSemanticCaseError(
                "base checkpoint bytes differ from their reusable identity"
            )
        checkpoint_path = root / "base_checkpoint_state.bin"
        with checkpoint_path.open("xb") as handle:
            handle.write(state)
            handle.flush()
            os.fsync(handle.fileno())
        checkpoint_path.chmod(0o444)
        checkpoint_projection = {
            "identity": identity,
            "state_file_name": checkpoint_path.name,
            "state_bytes": len(state),
            "state_sha256": _sha_bytes(state),
        }
    body = {
        "schema_version": WORKHORSE_SEMANTIC_CASE_FRAGMENT_SCHEMA_V2,
        "claim_class": "TECHNICAL_ONLY_RAW_EVIDENCE",
        **_NO_CLAIM_AUTHORITY,
        "observation": dict(_OBSERVATION_OFF),
        "case_id": case_id,
        "case_law": law.to_dict(),
        "case_table_sha256": workhorse_semantic_case_table_sha256_v2(
            semantic_temporal_k
        ),
        "semantic_temporal_k": semantic_temporal_k,
        "shape_id": shape_id,
        "layout_sha256": layout.layout_sha256,
        "semantic_transition_sha256": WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
        "artifact_sha256": artifact_sha256,
        "canonical_parent_binding": dict(parent),
        "canonical_parent_binding_sha256": parent["binding_sha256"],
        "producer_pid": os.getpid(),
        "native_outcome": native_outcome,
        "datasets": dataset_rows,
        "base_checkpoint": checkpoint_projection,
        "p2_witness": None if p2_witness is None else dict(p2_witness),
        "external_source_checkpoint_binding": source_checkpoint_projection,
    }
    manifest = {**body, "fragment_body_sha256": canonical_sha256(body)}
    path = root / "semantic_case_fragment.json"
    with path.open("x", encoding="ascii") as handle:
        handle.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o444)
    _fsync_directory(root)
    root.chmod(0o555)
    _fsync_directory(root.parent)
    reopen_workhorse_semantic_case_fragment_v2(path)
    return path


def reopen_workhorse_semantic_case_fragment_v2(
    manifest_path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(manifest_path)
    if (
        path.is_symlink()
        or not path.is_file()
        or path.name != "semantic_case_fragment.json"
        or path.stat().st_mode & 0o222
        or path.parent.stat().st_mode & 0o222
    ):
        raise WorkhorseSemanticCaseError(
            "semantic case fragment/root must be immutable regular content"
        )
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkhorseSemanticCaseError("semantic case fragment is unreadable") from exc
    fields = {
        "schema_version", "claim_class", "qualification_authority_granted",
        "science_authority_granted", "observation", "case_id", "case_law",
        "case_table_sha256", "semantic_temporal_k", "shape_id", "layout_sha256",
        "semantic_transition_sha256", "artifact_sha256",
        "canonical_parent_binding", "canonical_parent_binding_sha256",
        "producer_pid", "native_outcome", "datasets", "base_checkpoint",
        "p2_witness", "fragment_body_sha256",
        "external_source_checkpoint_binding",
    }
    value = _exact_mapping(value, fields, "semantic case fragment")
    body = {key: item for key, item in value.items() if key != "fragment_body_sha256"}
    K = value["semantic_temporal_k"]
    law = workhorse_semantic_case_law_v2(value["case_id"], semantic_temporal_k=K)
    shape = workhorse_shape_v2(value["shape_id"])
    layout = workhorse_layout_v2(shape)
    if (
        value["schema_version"] != WORKHORSE_SEMANTIC_CASE_FRAGMENT_SCHEMA_V2
        or value["claim_class"] != "TECHNICAL_ONLY_RAW_EVIDENCE"
        or value["qualification_authority_granted"] is not False
        or value["science_authority_granted"] is not False
        or value["observation"] != _OBSERVATION_OFF
        or value["case_law"] != law.to_dict()
        or value["case_table_sha256"] != workhorse_semantic_case_table_sha256_v2(K)
        or value["layout_sha256"] != layout.layout_sha256
        or value["semantic_transition_sha256"]
        != WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2
        or not _valid_sha(value["artifact_sha256"])
        or value["native_outcome"] != law.expected_native_outcome
        or value["fragment_body_sha256"] != canonical_sha256(body)
        or set(value["datasets"] or {}) != set(law.fragment_datasets)
    ):
        raise WorkhorseSemanticCaseError("semantic case fragment identity/seal changed")
    parent = validate_workhorse_semantic_case_parent_binding_v2(
        value["canonical_parent_binding"],
        case_id=law.case_id,
        semantic_temporal_k=K,
        enforce_current_process=False,
    )
    if (
        value["canonical_parent_binding_sha256"] != parent["binding_sha256"]
        or value["artifact_sha256"] != parent["artifact_sha256"]
        or (value["p2_witness"] is not None)
        != (law.case_id == "event_reuse_canonicalization_regression")
        or (value["base_checkpoint"] is not None) != law.produces_base_checkpoint
        or (value["external_source_checkpoint_binding"] is not None)
        != law.requires_source_checkpoint
    ):
        raise WorkhorseSemanticCaseError("semantic case fragment parent/payload changed")
    if value["p2_witness"] is not None:
        _validate_p2_witness(value["p2_witness"], semantic_temporal_k=K)
    if value["external_source_checkpoint_binding"] is not None:
        validate_external_checkpoint_source_binding_v2(
            value["external_source_checkpoint_binding"],
            expected_identity_sha256=parent[
                "source_checkpoint_identity_sha256"
            ],
        )
    if value["base_checkpoint"] is not None:
        checkpoint = _exact_mapping(
            value["base_checkpoint"],
            {"identity", "state_file_name", "state_bytes", "state_sha256"},
            "base checkpoint projection",
        )
        identity = _validate_base_checkpoint_identity_v2(checkpoint["identity"])
        state_path = path.parent / checkpoint["state_file_name"]
        if (
            type(checkpoint["state_file_name"]) is not str
            or Path(checkpoint["state_file_name"]).name
            != checkpoint["state_file_name"]
            or state_path.is_symlink()
            or not state_path.is_file()
            or state_path.stat().st_mode & 0o222
        ):
            raise WorkhorseSemanticCaseError("base checkpoint locator is unsafe")
        state = state_path.read_bytes()
        if (
            checkpoint["state_bytes"] != identity["state_bytes"]
            or checkpoint["state_sha256"] != identity["state_sha256"]
            or len(state) != identity["state_bytes"]
            or _sha_bytes(state) != identity["state_sha256"]
        ):
            raise WorkhorseSemanticCaseError("base checkpoint content seal failed")
    reopened: dict[str, dict[str, Any]] = {}
    for name in law.fragment_datasets:
        B, dataset_k, start = _dataset_coordinates(
            name, law=law, semantic_temporal_k=K
        )
        row = _exact_mapping(
            value["datasets"][name],
            {"B", "temporal_k", "start_next_absolute_tick", "components"},
            f"case dataset {name}",
        )
        if (row["B"], row["temporal_k"], row["start_next_absolute_tick"]) != (
            B, dataset_k, start
        ):
            raise WorkhorseSemanticCaseError(f"case dataset {name} coordinates changed")
        expected = _expected_component_bytes(layout, B=B, temporal_k=dataset_k)
        components = row["components"]
        if type(components) is not dict or set(components) != set(expected):
            raise WorkhorseSemanticCaseError(f"case dataset {name} components changed")
        payloads: dict[str, Any] = {
            "B": B, "temporal_k": dataset_k,
            "start_next_absolute_tick": start,
        }
        for component, expected_bytes in expected.items():
            item = _exact_mapping(
                components[component], {"file_name", "bytes", "sha256"},
                f"case dataset {name}.{component}",
            )
            target = path.parent / item["file_name"]
            if (
                type(item["file_name"]) is not str
                or Path(item["file_name"]).name != item["file_name"]
                or target.is_symlink()
                or not target.is_file()
                or target.stat().st_mode & 0o222
            ):
                raise WorkhorseSemanticCaseError(
                    f"case dataset {name}.{component} locator is unsafe"
                )
            payload = target.read_bytes()
            if (
                len(payload) != expected_bytes
                or item["bytes"] != expected_bytes
                or item["sha256"] != _sha_bytes(payload)
            ):
                raise WorkhorseSemanticCaseError(
                    f"case dataset {name}.{component} exact-content seal failed"
                )
            payloads[component] = payload
        reopened[name] = payloads
    return value, reopened


def write_terminalized_workhorse_semantic_case_receipt_v2(
    output_path: str | Path,
    *,
    case_fragment_path: str | Path,
    canonical_terminal_binding: Mapping[str, Any],
    produced_external_source_checkpoint_binding: Mapping[str, Any] | None = None,
) -> Path:
    """Classify one raw case only after the Scheduler terminal is reopened."""

    fragment, _ = reopen_workhorse_semantic_case_fragment_v2(case_fragment_path)
    parent = fragment["canonical_parent_binding"]
    try:
        from .canonical_rows import validate_workhorse_canonical_terminal_binding_v2

        terminal = validate_workhorse_canonical_terminal_binding_v2(
            canonical_terminal_binding, parent=parent
        )
    except Exception as exc:
        raise WorkhorseSemanticCaseError(
            f"semantic case terminal is not canonical: {exc}"
        ) from exc
    law = workhorse_semantic_case_law_v2(
        fragment["case_id"], semantic_temporal_k=fragment["semantic_temporal_k"]
    )
    terminal_contract = workhorse_semantic_case_terminal_contract_v2(law)
    if (
        terminal["attempt_terminal_status"]
        != terminal_contract["attempt_terminal_status"]
        or terminal["final_frontiers"] != terminal_contract["final_frontiers"]
    ):
        raise WorkhorseSemanticCaseError(
            "semantic case terminal status/frontiers differ from the case law"
        )
    result_class = (
        "SUCCESS"
        if law.expected_attempt_terminal_status == "COMPLETED"
        else "VALID_NEGATIVE"
    )
    produced_source: dict[str, Any] | None = None
    if law.produces_base_checkpoint:
        base_checkpoint = fragment.get("base_checkpoint")
        base_identity = (
            base_checkpoint.get("identity")
            if isinstance(base_checkpoint, Mapping)
            else None
        )
        if (
            not isinstance(base_identity, Mapping)
            or not _valid_sha(base_identity.get("base_checkpoint_identity_sha256"))
            or not isinstance(produced_external_source_checkpoint_binding, Mapping)
        ):
            raise WorkhorseSemanticCaseError(
                "base-checkpoint terminal receipt requires its continuation source binding"
            )
        produced_source = validate_external_checkpoint_source_binding_v2(
            produced_external_source_checkpoint_binding,
            expected_identity_sha256=base_identity[
                "base_checkpoint_identity_sha256"
            ],
        )
    elif produced_external_source_checkpoint_binding is not None:
        raise WorkhorseSemanticCaseError(
            "non-checkpoint semantic terminal gained a continuation source binding"
        )
    body = {
        "schema_version": WORKHORSE_SEMANTIC_CASE_TERMINAL_RECEIPT_SCHEMA_V2,
        "claim_class": "TECHNICAL_ONLY",
        **_NO_CLAIM_AUTHORITY,
        "case_id": law.case_id,
        "case_law": law.to_dict(),
        "case_table_sha256": fragment["case_table_sha256"],
        "semantic_temporal_k": fragment["semantic_temporal_k"],
        "shape_id": fragment["shape_id"],
        "artifact_sha256": fragment["artifact_sha256"],
        "case_fragment_path": str(Path(case_fragment_path).resolve()),
        "case_fragment_file_sha256": file_sha256(case_fragment_path),
        "case_fragment_body_sha256": fragment["fragment_body_sha256"],
        "canonical_parent_binding": parent,
        "canonical_terminal_binding": dict(terminal),
        "produced_external_source_checkpoint_binding": produced_source,
        "result_class": result_class,
        "observation": dict(_OBSERVATION_OFF),
    }
    value = {**body, "receipt_content_sha256": canonical_sha256(body)}
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise WorkhorseSemanticCaseError("semantic terminal receipt must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    output.chmod(0o444)
    _fsync_directory(output.parent)
    reopen_terminalized_workhorse_semantic_case_receipt_v2(output)
    return output


def reopen_terminalized_workhorse_semantic_case_receipt_v2(
    receipt_path: str | Path,
) -> dict[str, Any]:
    path = Path(receipt_path)
    try:
        metadata = path.lstat()
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkhorseSemanticCaseError(
            f"cannot reopen semantic terminal receipt: {exc}"
        ) from exc
    fields = {
        "schema_version", "claim_class", "qualification_authority_granted",
        "science_authority_granted", "case_id", "case_law", "case_table_sha256",
        "semantic_temporal_k", "shape_id", "artifact_sha256",
        "case_fragment_path", "case_fragment_file_sha256",
        "case_fragment_body_sha256", "canonical_parent_binding",
        "canonical_terminal_binding", "produced_external_source_checkpoint_binding",
        "result_class", "observation",
        "receipt_content_sha256",
    }
    value = _exact_mapping(value, fields, "semantic terminal receipt")
    body = {key: item for key, item in value.items() if key != "receipt_content_sha256"}
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_mode & 0o222
        or value["schema_version"]
        != WORKHORSE_SEMANTIC_CASE_TERMINAL_RECEIPT_SCHEMA_V2
        or value["claim_class"] != "TECHNICAL_ONLY"
        or value["qualification_authority_granted"] is not False
        or value["science_authority_granted"] is not False
        or value["observation"] != _OBSERVATION_OFF
        or value["receipt_content_sha256"] != canonical_sha256(body)
    ):
        raise WorkhorseSemanticCaseError("semantic terminal receipt seal changed")
    fragment_path = Path(value["case_fragment_path"])
    if (
        fragment_path.is_symlink()
        or not fragment_path.is_file()
        or file_sha256(fragment_path) != value["case_fragment_file_sha256"]
    ):
        raise WorkhorseSemanticCaseError("semantic terminal fragment cannot reopen")
    fragment, _ = reopen_workhorse_semantic_case_fragment_v2(fragment_path)
    law = workhorse_semantic_case_law_v2(
        value["case_id"], semantic_temporal_k=value["semantic_temporal_k"]
    )
    produced_source = value["produced_external_source_checkpoint_binding"]
    if law.produces_base_checkpoint:
        base_checkpoint = fragment.get("base_checkpoint")
        base_identity = (
            base_checkpoint.get("identity")
            if isinstance(base_checkpoint, Mapping)
            else None
        )
        if (
            not isinstance(base_identity, Mapping)
            or not _valid_sha(base_identity.get("base_checkpoint_identity_sha256"))
            or not isinstance(produced_source, Mapping)
            or validate_external_checkpoint_source_binding_v2(
                produced_source,
                expected_identity_sha256=base_identity[
                    "base_checkpoint_identity_sha256"
                ],
            )
            != produced_source
        ):
            raise WorkhorseSemanticCaseError(
                "base-checkpoint terminal continuation source binding changed"
            )
    elif produced_source is not None:
        raise WorkhorseSemanticCaseError(
            "non-checkpoint semantic terminal has a continuation source binding"
        )
    if (
        value["case_law"] != law.to_dict()
        or value["case_table_sha256"] != fragment["case_table_sha256"]
        or value["shape_id"] != fragment["shape_id"]
        or value["artifact_sha256"] != fragment["artifact_sha256"]
        or value["case_fragment_body_sha256"] != fragment["fragment_body_sha256"]
        or value["canonical_parent_binding"]
        != fragment["canonical_parent_binding"]
        or value["result_class"]
        != (
            "SUCCESS"
            if law.expected_attempt_terminal_status == "COMPLETED"
            else "VALID_NEGATIVE"
        )
    ):
        raise WorkhorseSemanticCaseError("semantic terminal/fragment identity changed")
    try:
        from .canonical_rows import validate_workhorse_canonical_terminal_binding_v2

        validate_workhorse_canonical_terminal_binding_v2(
            value["canonical_terminal_binding"],
            parent=value["canonical_parent_binding"],
        )
    except Exception as exc:
        raise WorkhorseSemanticCaseError(
            f"semantic terminal authority cannot reopen: {exc}"
        ) from exc
    value["receipt_path"] = str(path.resolve())
    value["receipt_file_sha256"] = _sha_bytes(raw)
    return value


__all__ = [
    "WORKHORSE_EXTERNAL_CHECKPOINT_AUTHORIZATION_SOURCE_V2",
    "WORKHORSE_EXTERNAL_CHECKPOINT_SOURCE_SCHEMA_V2",
    "WORKHORSE_SCIENCE_BASE_CHECKPOINT_IDENTITY_SCHEMA_V2",
    "WORKHORSE_SCIENCE_CHECKPOINT_COMPATIBILITY_SCHEMA_V2",
    "WORKHORSE_SEMANTIC_CASE_COUNT_V2",
    "WORKHORSE_SEMANTIC_CASE_FRAGMENT_SCHEMA_V2",
    "WORKHORSE_SEMANTIC_CASE_OPERATION_V2",
    "WORKHORSE_SEMANTIC_CASE_TERMINAL_RECEIPT_SCHEMA_V2",
    "WORKHORSE_SEMANTIC_P2_WITNESS_SCHEMA_V2",
    "WorkhorseSemanticCaseError",
    "WorkhorseSemanticCaseLawV2",
    "reopen_terminalized_workhorse_semantic_case_receipt_v2",
    "reopen_workhorse_semantic_case_fragment_v2",
    "science_compatible_base_checkpoint_identity_v2",
    "validate_workhorse_science_checkpoint_compatibility_binding_v2",
    "validate_external_checkpoint_source_binding_v2",
    "validate_workhorse_semantic_case_parent_binding_v2",
    "workhorse_semantic_case_law_v2",
    "workhorse_semantic_case_operation_contract_v2",
    "workhorse_semantic_case_table_sha256_v2",
    "workhorse_semantic_case_terminal_contract_v2",
    "workhorse_semantic_case_table_v2",
    "workhorse_science_checkpoint_compatibility_binding_v2",
    "write_terminalized_workhorse_semantic_case_receipt_v2",
    "write_workhorse_semantic_case_fragment_v2",
]
