from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import struct

import pytest

from engine.perf.cuda_full_evolution_v2.workhorse import canonical_rows
from engine.perf.cuda_full_evolution_v2.workhorse import authority
from engine.perf.cuda_full_evolution_v2.workhorse import semantic_cases
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    SUPPORTED_WORKHORSE_SHAPES_V2,
    FullEvolutionCudaWorkhorseContractV2,
    canonical_sha256,
)
from engine.perf.cuda_full_evolution_v2.workhorse.layout import workhorse_layout_v2
from engine.perf.cuda_full_evolution_v2.workhorse.probe import (
    run_workhorse_semantic_case_v2,
    run_workhorse_semantic_gate_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.semantic_cases import (
    WorkhorseSemanticCaseError,
    science_compatible_base_checkpoint_identity_v2,
    workhorse_science_checkpoint_compatibility_binding_v2,
    workhorse_semantic_case_operation_contract_v2,
    workhorse_semantic_case_table_v2,
    workhorse_semantic_case_terminal_contract_v2,
    write_terminalized_workhorse_semantic_case_receipt_v2,
    write_workhorse_semantic_case_fragment_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.semantic_evidence import (
    WORKHORSE_SEMANTIC_DATASETS_V2,
    aggregate_workhorse_semantic_case_receipts_v2,
    reopen_workhorse_semantic_case_aggregate_v2,
    reopen_workhorse_semantic_evidence_bundle_v2,
    workhorse_semantic_dataset_projection_v2,
)


ROOT = Path(__file__).resolve().parents[1]
SHAPE = SUPPORTED_WORKHORSE_SHAPES_V2[0]
K = 2
SHA = "a" * 64


def _raw_dataset(*, B: int, temporal_k: int, start: int) -> dict[str, object]:
    layout = workhorse_layout_v2(SHAPE)
    output = {
        "state": bytes(B * layout.state_bytes_per_world),
        "scratch": bytes(B * layout.scratch_bytes_per_world),
        "events": bytes(B * temporal_k * layout.event_rows_per_world_tick * 48),
        "counts": bytes(B * 4),
    }
    return {
        "B": B,
        "temporal_k": temporal_k,
        "start_next_absolute_tick": start,
        **{f"input_{name}": payload for name, payload in output.items()},
        **output,
    }


def _parent(
    *,
    case_id: str,
    status: str = "COMPLETED",
) -> dict[str, object]:
    law = next(case for case in workhorse_semantic_case_table_v2(K) if case.case_id == case_id)
    parent = {
        "shape_id": SHAPE.shape_id,
        "backend_id": (
            "reference_phased_v2"
            if law.artifact_role == "REFERENCE"
            else "fused_cta_temporal_v2"
        ),
        "artifact_sha256": SHA,
        "binding_sha256": "b" * 64,
        "attempt_id": f"attempt-{case_id}",
        "device_uuid": "GPU-test",
        "source_checkpoint_identity_sha256": (
            "c" * 64 if law.requires_source_checkpoint else None
        ),
        "contract": {
            "B": law.B,
            "temporal_k": law.dispatch_temporal_k,
            "total_ticks": law.total_ticks,
            "start_next_tick": law.start_next_absolute_tick,
        },
        "absolute_tick_span": {
            "start_next_tick": law.start_next_absolute_tick,
            "target_next_tick": law.start_next_absolute_tick + law.total_ticks,
        },
        "test_terminal_status": status,
    }
    return parent


def _trust_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        semantic_cases,
        "validate_workhorse_semantic_case_parent_binding_v2",
        lambda value, **_kwargs: dict(value),
    )


def _trust_p2_success_ack(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        authority,
        "validate_workhorse_event_sink_receipt_v2",
        lambda path, **_kwargs: {
            "receipt_path": str(Path(path).resolve()),
            "receipt_file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "start_next_absolute_tick": 41,
            "end_next_absolute_tick": 43,
            "tick_count": 2,
            "world_uids": [98_000],
        },
    )


def _terminal(parent: dict[str, object], *, status: str) -> dict[str, object]:
    start = parent["absolute_tick_span"]["start_next_tick"]  # type: ignore[index]
    target = parent["absolute_tick_span"]["target_next_tick"]  # type: ignore[index]
    if status == "COMPLETED":
        values = (target, target, target, target)
    else:
        values = (target, start, start, start)
    return {
        "attempt_terminal_status": status,
        "final_frontiers": {
            "schema_version": "crca_attempt_frontiers_v1",
            "last_dispatched_next_tick": values[0],
            "last_computed_next_tick": values[1],
            "durable_checkpoint_next_tick": values[2],
            "published_artifact_next_tick": values[3],
        },
    }


def test_closed_case_table_and_dataset_projection_are_exact() -> None:
    assert (
        semantic_cases.WORKHORSE_EXTERNAL_CHECKPOINT_AUTHORIZATION_SOURCE_V2
        == "CRCA_WORKHORSE_SEMANTIC_CHECKPOINT_OCCURRENCE_V2"
    )
    table = workhorse_semantic_case_table_v2(K)
    assert len(table) == 24
    assert [case.ordinal for case in table] == list(range(1, 25))
    assert len({case.case_id for case in table}) == 24
    assert sum(case.artifact_role == "REFERENCE" for case in table) == 3
    assert sum(case.artifact_role == "CANDIDATE" for case in table) == 21
    assert {
        case.case_id
        for case in table
        if case.expected_attempt_terminal_status == "FAILED"
    } == {
        "invalid_restore_failstop",
        "transaction_failstop",
        "event_reuse_canonicalization_regression",
    }
    assert next(case for case in table if case.ordinal == 13).fragment_datasets == (
        "causal_mutation_disabled",
        "eed_enabled",
    )
    projection = workhorse_semantic_dataset_projection_v2(K)
    assert set(projection) == set(WORKHORSE_SEMANTIC_DATASETS_V2)
    assert len(projection) == 29
    assert projection["b_whole"] == (
        "parity_candidate_k_dispatch",
        "parity_candidate",
    )
    assert projection["b_split"] == ("__B_SPLIT__", "__B_SPLIT__")


def test_operation_contract_is_tick_native_observation_off_and_callback_free() -> None:
    contract = workhorse_semantic_case_operation_contract_v2(
        case_id="transaction_failstop",
        shape_id=SHAPE.shape_id,
        semantic_temporal_k=K,
    )
    encoded = json.dumps(contract, sort_keys=True)
    assert "generation" not in encoded.lower()
    assert contract["time_authority"] == "absolute_tick_v1"
    assert contract["observation"] == {
        "mode": "OFF",
        "science_records_emitted": 0,
        "science_observation_schedule": None,
        "population_change_detection": False,
    }
    assert contract["science_authority_granted"] is False
    assert contract["attempt_terminal_contract"] == {
        "attempt_terminal_status": "FAILED",
        "final_frontiers": {
            "schema_version": "crca_attempt_frontiers_v1",
            "last_dispatched_next_tick": 15,
            "last_computed_next_tick": 13,
            "durable_checkpoint_next_tick": 13,
            "published_artifact_next_tick": 13,
        },
    }
    signature = inspect.signature(run_workhorse_semantic_case_v2)
    assert "attempt_authority_validator" not in signature.parameters
    assert "callback" not in signature.parameters
    source = inspect.getsource(run_workhorse_semantic_case_v2)
    assert "subprocess" not in source
    assert "Popen" not in source
    with pytest.raises(Exception, match="direct Workhorse semantic GPU execution is retired"):
        run_workhorse_semantic_gate_v2(
            "a",
            "b",
            event_sink_root="sink",
            checkpoint_root="checkpoint",
            semantic_evidence_root="evidence",
            receipt_output_path="receipt",
            attempt_authority_validator=lambda _request: {},
        )


def test_science_base_identity_binds_consumer_abi_and_last_event_ack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout = workhorse_layout_v2(SHAPE)
    state = bytearray(layout.state_bytes_per_world)
    header = [0] * 32
    header[0], header[1], header[13] = 97_000, 22, 53
    struct.pack_into("<32Q", state, 0, *header)
    state_bytes = bytes(state)
    contract = FullEvolutionCudaWorkhorseContractV2(
        shape_id=SHAPE.shape_id,
        backend_id="fused_cta_temporal_v2",
        B=1,
        temporal_k=1,
        total_ticks=1,
        start_next_tick=21,
    )
    attempt_binding = {"test": "binding"}
    monkeypatch.setattr(
        authority,
        "_event_attempt_view_v2",
        lambda _value, **_kwargs: {
            "binding": attempt_binding,
            "contract": contract,
            "binding_sha256": "8" * 64,
            "lineage_sha256": "9" * 64,
            "owner_pid": 123,
            "owner_process_start_ticks": 456,
            "device_uuid": "GPU-test",
            "artifact_sha256": "b" * 64,
            "B": 1,
            "start_next_tick": 21,
            "target_next_tick": 22,
        },
    )
    ack_path = authority.publish_workhorse_event_sink_receipt_v2(
        tmp_path / "ack",
        sink_id="science-checkpoint-test",
        event_rows=bytes(layout.event_rows_per_world_tick * 48),
        event_counts=bytes(4),
        event_rows_shape=(1, 1, layout.event_rows_per_world_tick, 48),
        event_counts_shape=(1,),
        world_uids=[97_000],
        start_next_absolute_tick=21,
        end_next_absolute_tick=22,
        artifact_identity_sha256=SHA,
        artifact_sha256="b" * 64,
        manifest_sha256="c" * 64,
        attempt_authority_binding=attempt_binding,
        rng_keys=[53],
        pre_state_sha256="d" * 64,
        post_state_sha256=hashlib.sha256(state_bytes).hexdigest(),
    )
    ack = authority.validate_workhorse_event_sink_receipt_v2(ack_path)
    science = workhorse_science_checkpoint_compatibility_binding_v2(ROOT)
    identity = science_compatible_base_checkpoint_identity_v2(
        shape_id=SHAPE.shape_id,
        source_artifact_identity_sha256=SHA,
        checkpoint_at_tick=22,
        state_bytes=state_bytes,
        world_uids=[97_000],
        rng_keys=[53],
        event_sink_acknowledgement=ack,
        science_checkpoint_compatibility_binding=science,
    )
    assert identity["last_acknowledged_event_frontier"]["receipt_file_sha256"] == ack[
        "receipt_file_sha256"
    ]
    assert identity["science_checkpoint_compatibility"] == science
    assert identity["observation"]["mode"] == "OFF"
    assert identity["science_authority_granted"] is False

    omitted = dict(science)
    omitted.pop("continuation_adapter_version")
    with pytest.raises(WorkhorseSemanticCaseError, match="fields are inexact"):
        science_compatible_base_checkpoint_identity_v2(
            shape_id=SHAPE.shape_id,
            source_artifact_identity_sha256=SHA,
            checkpoint_at_tick=22,
            state_bytes=state_bytes,
            world_uids=[97_000],
            rng_keys=[53],
            event_sink_acknowledgement=ack,
            science_checkpoint_compatibility_binding=omitted,
        )
    substituted = dict(science)
    substituted["observation_adapter_version"] = "hostile-v2"
    substituted["binding_sha256"] = canonical_sha256(
        {key: value for key, value in substituted.items() if key != "binding_sha256"}
    )
    with pytest.raises(WorkhorseSemanticCaseError, match="binding changed"):
        science_compatible_base_checkpoint_identity_v2(
            shape_id=SHAPE.shape_id,
            source_artifact_identity_sha256=SHA,
            checkpoint_at_tick=22,
            state_bytes=state_bytes,
            world_uids=[97_000],
            rng_keys=[53],
            event_sink_acknowledgement=ack,
            science_checkpoint_compatibility_binding=substituted,
        )
    missing_ack = dict(ack)
    missing_ack.pop("receipt_body_sha256")
    with pytest.raises(WorkhorseSemanticCaseError, match="lacks the exact"):
        science_compatible_base_checkpoint_identity_v2(
            shape_id=SHAPE.shape_id,
            source_artifact_identity_sha256=SHA,
            checkpoint_at_tick=22,
            state_bytes=state_bytes,
            world_uids=[97_000],
            rng_keys=[53],
            event_sink_acknowledgement=missing_ack,
            science_checkpoint_compatibility_binding=science,
        )
    fabricated_path = tmp_path / "fabricated-ack.json"
    fabricated_path.write_text("{}\n", encoding="ascii")
    fabricated_path.chmod(0o444)
    fabricated_ack = {
        **ack,
        "receipt_path": str(fabricated_path.resolve()),
        "receipt_file_sha256": hashlib.sha256(
            fabricated_path.read_bytes()
        ).hexdigest(),
    }
    with pytest.raises(WorkhorseSemanticCaseError, match="cannot be reopened"):
        science_compatible_base_checkpoint_identity_v2(
            shape_id=SHAPE.shape_id,
            source_artifact_identity_sha256=SHA,
            checkpoint_at_tick=22,
            state_bytes=state_bytes,
            world_uids=[97_000],
            rng_keys=[53],
            event_sink_acknowledgement=fabricated_ack,
            science_checkpoint_compatibility_binding=science,
        )
    substituted_ack = dict(ack)
    substituted_ack["dispatch_identity_sha256"] = "7" * 64
    with pytest.raises(WorkhorseSemanticCaseError, match="not exact"):
        science_compatible_base_checkpoint_identity_v2(
            shape_id=SHAPE.shape_id,
            source_artifact_identity_sha256=SHA,
            checkpoint_at_tick=22,
            state_bytes=state_bytes,
            world_uids=[97_000],
            rng_keys=[53],
            event_sink_acknowledgement=substituted_ack,
            science_checkpoint_compatibility_binding=science,
        )
    with pytest.raises(WorkhorseSemanticCaseError, match="cannot be reopened"):
        science_compatible_base_checkpoint_identity_v2(
            shape_id=SHAPE.shape_id,
            source_artifact_identity_sha256="e" * 64,
            checkpoint_at_tick=22,
            state_bytes=state_bytes,
            world_uids=[97_000],
            rng_keys=[53],
            event_sink_acknowledgement=ack,
            science_checkpoint_compatibility_binding=science,
        )


def test_raw_fragments_are_fresh_sealed_and_reopened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _trust_parent(monkeypatch)
    parent = _parent(case_id="causal_baseline")
    path = write_workhorse_semantic_case_fragment_v2(
        tmp_path / "fragment",
        case_id="causal_baseline",
        semantic_temporal_k=K,
        shape_id=SHAPE.shape_id,
        artifact_sha256=SHA,
        canonical_parent_binding=parent,
        native_outcome="SUCCESS",
        datasets={
            "causal_baseline": _raw_dataset(B=1, temporal_k=1, start=71)
        },
    )
    value, data = semantic_cases.reopen_workhorse_semantic_case_fragment_v2(path)
    assert value["case_id"] == "causal_baseline"
    assert set(data) == {"causal_baseline"}
    assert path.stat().st_mode & 0o222 == 0
    assert path.parent.stat().st_mode & 0o222 == 0
    with pytest.raises(WorkhorseSemanticCaseError, match="must be fresh"):
        write_workhorse_semantic_case_fragment_v2(
            tmp_path / "fragment",
            case_id="causal_baseline",
            semantic_temporal_k=K,
            shape_id=SHAPE.shape_id,
            artifact_sha256=SHA,
            canonical_parent_binding=parent,
            native_outcome="SUCCESS",
            datasets={
                "causal_baseline": _raw_dataset(B=1, temporal_k=1, start=71)
            },
        )


def test_terminal_classification_derives_valid_negative_and_rejects_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _trust_parent(monkeypatch)
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_canonical_terminal_binding_v2",
        lambda value, **_kwargs: dict(value),
    )
    completed_parent = _parent(case_id="causal_baseline")
    completed_fragment = write_workhorse_semantic_case_fragment_v2(
        tmp_path / "completed-fragment",
        case_id="causal_baseline",
        semantic_temporal_k=K,
        shape_id=SHAPE.shape_id,
        artifact_sha256=SHA,
        canonical_parent_binding=completed_parent,
        native_outcome="SUCCESS",
        datasets={
            "causal_baseline": _raw_dataset(B=1, temporal_k=1, start=71)
        },
    )
    completed_receipt = write_terminalized_workhorse_semantic_case_receipt_v2(
        tmp_path / "completed.json",
        case_fragment_path=completed_fragment,
        canonical_terminal_binding=_terminal(completed_parent, status="COMPLETED"),
    )
    assert semantic_cases.reopen_terminalized_workhorse_semantic_case_receipt_v2(
        completed_receipt
    )["result_class"] == "SUCCESS"
    assert semantic_cases.reopen_terminalized_workhorse_semantic_case_receipt_v2(
        completed_receipt
    )["produced_external_source_checkpoint_binding"] is None

    failed_parent = _parent(case_id="transaction_failstop", status="FAILED")
    failed_fragment = write_workhorse_semantic_case_fragment_v2(
        tmp_path / "failed-fragment",
        case_id="transaction_failstop",
        semantic_temporal_k=K,
        shape_id=SHAPE.shape_id,
        artifact_sha256=SHA,
        canonical_parent_binding=failed_parent,
        native_outcome="EXPECTED_TRANSACTION_FAILURE",
        datasets={
            "transaction_after": _raw_dataset(B=1, temporal_k=2, start=13)
        },
    )
    failed_receipt = write_terminalized_workhorse_semantic_case_receipt_v2(
        tmp_path / "failed.json",
        case_fragment_path=failed_fragment,
        canonical_terminal_binding=_terminal(failed_parent, status="FAILED"),
    )
    assert semantic_cases.reopen_terminalized_workhorse_semantic_case_receipt_v2(
        failed_receipt
    )["result_class"] == "VALID_NEGATIVE"
    with pytest.raises(WorkhorseSemanticCaseError, match="status/frontiers"):
        write_terminalized_workhorse_semantic_case_receipt_v2(
            tmp_path / "mismatch.json",
            case_fragment_path=completed_fragment,
            canonical_terminal_binding=_terminal(completed_parent, status="FAILED"),
        )


def test_p2_witness_proves_no_failed_ack_or_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _trust_parent(monkeypatch)
    _trust_p2_success_ack(monkeypatch)
    ack_path = tmp_path / "success-ack.json"
    ack_path.write_text("{}\n", encoding="ascii")
    ack_path.chmod(0o444)
    active = struct.pack("<Q", 43) + bytes(range(40))
    witness_body = {
        "schema_version": "eiel_workhorse_reused_event_buffer_canonicalization_witness_v2",
        "B": 1,
        "temporal_k": K,
        "successful_dispatch_start_tick": 41,
        "successful_dispatch_end_tick": 43,
        "successful_ack_receipt_path": str(ack_path.resolve()),
        "successful_ack_receipt_file_sha256": hashlib.sha256(
            ack_path.read_bytes()
        ).hexdigest(),
        "failed_event_acknowledged": False,
        "failed_event_publication_path": None,
        "failed_dispatch_pending_preserved": True,
        "poisoned_status": [1],
        "post_failure_reuse_rejected": True,
        "failed_event_buffer_cannot_be_reused": True,
        "canonicalization_law": "NATIVE_CANONICALIZATION_WITHOUT_ACK_OR_PUBLICATION",
        "failed_dispatch_start_tick": 43,
        "failed_dispatch_requested_end_tick": 45,
        "failed_status": 1,
        "failed_frontier": 43,
        "active_rows_before_canonicalization": 1,
        "active_rows_after_canonicalization": 1,
        "active_row_bytes_sha256_before_canonicalization": hashlib.sha256(
            active
        ).hexdigest(),
        "active_row_bytes_sha256_after_canonicalization": hashlib.sha256(
            active
        ).hexdigest(),
        "active_row_bytes_hex_before_canonicalization": active.hex(),
        "active_row_bytes_hex_after_canonicalization": active.hex(),
        "active_row_absolute_ticks": [43],
        "active_rows_match_failed_absolute_tick": True,
        "inactive_rows_exact_zero_after_canonicalization": True,
        "later_k_slices_exact_zero_after_canonicalization": True,
        "active_rows_byte_preserved": True,
        "same_owner_storage_data_ptr": True,
        "successful_ack_preceded_failed_reuse": True,
    }
    witness = {**witness_body, "witness_sha256": canonical_sha256(witness_body)}
    write_workhorse_semantic_case_fragment_v2(
        tmp_path / "p2-fragment",
        case_id="event_reuse_canonicalization_regression",
        semantic_temporal_k=K,
        shape_id=SHAPE.shape_id,
        artifact_sha256=SHA,
        canonical_parent_binding=_parent(
            case_id="event_reuse_canonicalization_regression", status="FAILED"
        ),
        native_outcome="EXPECTED_REUSED_OWNER_TRANSACTION_FAILURE",
        datasets={},
        p2_witness=witness,
    )
    hostile_body = {**witness_body, "failed_event_acknowledged": True}
    hostile = {**hostile_body, "witness_sha256": canonical_sha256(hostile_body)}
    with pytest.raises(WorkhorseSemanticCaseError, match="does not prove"):
        write_workhorse_semantic_case_fragment_v2(
            tmp_path / "hostile-p2-fragment",
            case_id="event_reuse_canonicalization_regression",
            semantic_temporal_k=K,
            shape_id=SHAPE.shape_id,
            artifact_sha256=SHA,
            canonical_parent_binding=_parent(
                case_id="event_reuse_canonicalization_regression", status="FAILED"
            ),
            native_outcome="EXPECTED_REUSED_OWNER_TRANSACTION_FAILURE",
            datasets={},
            p2_witness=hostile,
        )
    unpoisoned_body = {**witness_body, "poisoned_status": [0]}
    unpoisoned = {
        **unpoisoned_body,
        "witness_sha256": canonical_sha256(unpoisoned_body),
    }
    with pytest.raises(WorkhorseSemanticCaseError, match="does not prove"):
        write_workhorse_semantic_case_fragment_v2(
            tmp_path / "unpoisoned-p2-fragment",
            case_id="event_reuse_canonicalization_regression",
            semantic_temporal_k=K,
            shape_id=SHAPE.shape_id,
            artifact_sha256=SHA,
            canonical_parent_binding=_parent(
                case_id="event_reuse_canonicalization_regression", status="FAILED"
            ),
            native_outcome="EXPECTED_REUSED_OWNER_TRANSACTION_FAILURE",
            datasets={},
            p2_witness=unpoisoned,
        )


def test_offline_streaming_aggregate_maps_24_receipts_to_29_datasets_and_p2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _trust_parent(monkeypatch)
    _trust_p2_success_ack(monkeypatch)
    monkeypatch.setattr(
        canonical_rows,
        "validate_workhorse_canonical_terminal_binding_v2",
        lambda value, **_kwargs: dict(value),
    )
    monkeypatch.setattr(
        semantic_cases,
        "_validate_base_checkpoint_identity_v2",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        semantic_cases,
        "validate_external_checkpoint_source_binding_v2",
        lambda value, **_kwargs: dict(value),
    )
    layout = workhorse_layout_v2(SHAPE)
    ack_path = tmp_path / "p2-success-ack.json"
    ack_path.write_text("{}\n", encoding="ascii")
    ack_path.chmod(0o444)
    active = struct.pack("<Q", 43) + bytes(range(40))
    p2_body = {
        "schema_version": "eiel_workhorse_reused_event_buffer_canonicalization_witness_v2",
        "B": 1,
        "temporal_k": K,
        "successful_dispatch_start_tick": 41,
        "successful_dispatch_end_tick": 43,
        "successful_ack_receipt_path": str(ack_path.resolve()),
        "successful_ack_receipt_file_sha256": hashlib.sha256(
            ack_path.read_bytes()
        ).hexdigest(),
        "failed_event_acknowledged": False,
        "failed_event_publication_path": None,
        "failed_dispatch_pending_preserved": True,
        "poisoned_status": [1],
        "post_failure_reuse_rejected": True,
        "failed_event_buffer_cannot_be_reused": True,
        "canonicalization_law": "NATIVE_CANONICALIZATION_WITHOUT_ACK_OR_PUBLICATION",
        "failed_dispatch_start_tick": 43,
        "failed_dispatch_requested_end_tick": 45,
        "failed_status": 1,
        "failed_frontier": 43,
        "active_rows_before_canonicalization": 1,
        "active_rows_after_canonicalization": 1,
        "active_row_bytes_sha256_before_canonicalization": hashlib.sha256(
            active
        ).hexdigest(),
        "active_row_bytes_sha256_after_canonicalization": hashlib.sha256(
            active
        ).hexdigest(),
        "active_row_bytes_hex_before_canonicalization": active.hex(),
        "active_row_bytes_hex_after_canonicalization": active.hex(),
        "active_row_absolute_ticks": [43],
        "active_rows_match_failed_absolute_tick": True,
        "inactive_rows_exact_zero_after_canonicalization": True,
        "later_k_slices_exact_zero_after_canonicalization": True,
        "active_rows_byte_preserved": True,
        "same_owner_storage_data_ptr": True,
        "successful_ack_preceded_failed_reuse": True,
    }
    p2_witness = {**p2_body, "witness_sha256": canonical_sha256(p2_body)}
    receipt_paths = []
    real_getpid = semantic_cases.os.getpid
    real_pid = real_getpid()
    try:
        for law in workhorse_semantic_case_table_v2(K):
            parent = _parent(
                case_id=law.case_id,
                status=law.expected_attempt_terminal_status,
            )
            parent["artifact_sha256"] = (
                "e" * 64 if law.artifact_role == "REFERENCE" else SHA
            )
            datasets = {}
            for name in law.fragment_datasets:
                B, dataset_k, start = semantic_cases._dataset_coordinates(
                    name, law=law, semantic_temporal_k=K
                )
                dataset = _raw_dataset(B=B, temporal_k=dataset_k, start=start)
                fill = bytes([law.ordinal])
                dataset = {
                    key: (
                        fill * len(value)
                        if key in {
                            "input_state",
                            "input_scratch",
                            "input_events",
                            "input_counts",
                            "state",
                            "scratch",
                            "events",
                            "counts",
                        }
                        else value
                    )
                    for key, value in dataset.items()
                }
                datasets[name] = dataset
            base_checkpoint = None
            if law.produces_base_checkpoint:
                checkpoint_state = bytes(layout.state_bytes_per_world)
                base_checkpoint = {
                    "identity": {
                        "shape_id": SHAPE.shape_id,
                        "state_bytes": len(checkpoint_state),
                        "state_sha256": hashlib.sha256(checkpoint_state).hexdigest(),
                        "base_checkpoint_identity_sha256": "d" * 64,
                    },
                    "state": checkpoint_state,
                }
            semantic_cases.os.getpid = (
                (lambda: real_pid + 1)
                if law.case_id == "fresh_restore_continuation"
                else (lambda: real_pid)
            )
            fragment = write_workhorse_semantic_case_fragment_v2(
                tmp_path / f"fragment-{law.ordinal:02d}",
                case_id=law.case_id,
                semantic_temporal_k=K,
                shape_id=SHAPE.shape_id,
                artifact_sha256=parent["artifact_sha256"],
                canonical_parent_binding=parent,
                native_outcome=law.expected_native_outcome,
                datasets=datasets,
                base_checkpoint=base_checkpoint,
                p2_witness=(p2_witness if law.ordinal == 24 else None),
                external_source_checkpoint_binding=(
                    {"test_source": law.case_id}
                    if law.requires_source_checkpoint
                    else None
                ),
            )
            terminal_contract = workhorse_semantic_case_terminal_contract_v2(law)
            produced_source = (
                {"test_source": law.case_id}
                if law.produces_base_checkpoint
                else None
            )
            if law.produces_base_checkpoint:
                with pytest.raises(
                    WorkhorseSemanticCaseError,
                    match="requires its continuation source binding",
                ):
                    write_terminalized_workhorse_semantic_case_receipt_v2(
                        tmp_path / "terminal-20-missing-source.json",
                        case_fragment_path=fragment,
                        canonical_terminal_binding=terminal_contract,
                    )
            receipt_paths.append(
                write_terminalized_workhorse_semantic_case_receipt_v2(
                    tmp_path / f"terminal-{law.ordinal:02d}.json",
                    case_fragment_path=fragment,
                    canonical_terminal_binding=terminal_contract,
                    produced_external_source_checkpoint_binding=produced_source,
                )
            )
    finally:
        semantic_cases.os.getpid = real_getpid
    aggregate_path = aggregate_workhorse_semantic_case_receipts_v2(
        tmp_path / "aggregate",
        terminal_receipt_paths=receipt_paths,
        governing_contracts={"test_contract_sha256": "f" * 64},
    )
    aggregate = reopen_workhorse_semantic_case_aggregate_v2(aggregate_path)
    manifest, datasets = reopen_workhorse_semantic_evidence_bundle_v2(
        aggregate["semantic_evidence"]["manifest_path"]
    )
    assert len(aggregate["terminal_receipts"]) == 24
    assert aggregate["aggregation"] == "OFFLINE_STREAMING_NO_GPU"
    assert len(manifest["datasets"]) == 29
    assert set(datasets) == set(WORKHORSE_SEMANTIC_DATASETS_V2)
    assert datasets["b_whole"] == datasets["parity_candidate"]
    assert datasets["b_split"]["state"][: layout.state_bytes_per_world] == bytes(
        [3]
    ) * layout.state_bytes_per_world
    assert datasets["b_split"]["state"][layout.state_bytes_per_world :] == bytes(
        [4]
    ) * layout.state_bytes_per_world
    assert aggregate["supplemental_p2_witness"]["projection_sha256"]
