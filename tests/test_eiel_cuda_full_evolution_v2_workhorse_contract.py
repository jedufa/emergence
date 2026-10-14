from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import ast
import hashlib
import inspect
import json
import os
import struct
import sys
from types import ModuleType, SimpleNamespace

try:
    from compression import zstd as _test_zstd
except ImportError:  # pragma: no cover - compatibility interpreter
    _test_zstd = None

import pytest

from engine.perf.cuda_full_evolution_v2.workhorse import (
    BACKEND_FUSED_CTA_TEMPORAL_V2,
    BACKEND_PHASE_GRAPH_V2,
    BACKEND_REFERENCE_PHASED_V2,
    FULL_EVOLUTION_CAPABILITIES_V2,
    SUPPORTED_WORKHORSE_SHAPES_V2,
    WORKHORSE_BACKENDS_V2,
    WORKHORSE_EVIDENCE_DERIVED_CHECKS_V2,
    WORKHORSE_REQUIRED_SEMANTIC_CHECKS_V2,
    WORKHORSE_SEMANTIC_DATASETS_V2,
    WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
    FullEvolutionCudaWorkhorseContractV2,
    FullEvolutionWorkhorseAuthorityError,
    FullEvolutionWorkhorseContractError,
    WorkhorseOwnerComponentsV2,
    WorkhorseAttemptAuthorityError,
    admit_workhorse_vram_v2,
    derive_workhorse_semantic_checks_v2,
    estimate_workhorse_resources_v2,
    publish_workhorse_event_sink_receipt_v2,
    publish_workhorse_sparse_event_sink_receipt_v3,
    validate_authoritative_micro_denominator_v2,
    validate_workhorse_event_sink_receipt_v2,
    validate_workhorse_attempt_authority_binding_v2,
    validate_workhorse_semantic_gate_receipt_v2,
    validate_workhorse_world_identity_binding_v2,
    validate_injected_workhorse_attempt_authority_v2,
    workhorse_attempt_authority_request_v2,
    workhorse_attempt_lineage_sha256_v2,
    workhorse_attempt_identity_v2,
    workhorse_artifact_identity_v2,
    workhorse_backend_temporal_k_candidates_v2,
    workhorse_b_candidates_v2,
    workhorse_candidate_envelope_v2,
    workhorse_checkpoint_identity_v2,
    workhorse_governing_contract_bindings_v2,
    workhorse_layout_v2,
    workhorse_resource_observation_sha256_v2,
    workhorse_semantic_contract_v2,
    workhorse_topology_resource_binding_v2,
    workhorse_world_identity_binding_v2,
    write_workhorse_semantic_evidence_bundle_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.attempt_authority import (
    WORKHORSE_INJECTED_ATTEMPT_AUTHORITY_SCHEMA_V2,
    resolve_injected_workhorse_attempt_authority_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.authority import (
    WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4,
    workhorse_event_dispatch_content_sha256_v2,
    workhorse_sparse_event_dispatch_content_sha256_v3,
)
from engine.perf.cuda_full_evolution_v2.workhorse.canonical_rows import (
    run_workhorse_canonical_admitted_row_v2,
)
from engine.perf.cuda_full_evolution_v2.workhorse.contract import (
    MAX_SIGNED_INT64,
    WORKHORSE_REJECTED_SEMANTIC_HASHES_V2,
    WorkhorseCompileShapeV2,
    canonical_sha256,
    checked_add_i64,
    checked_multiply_i64,
)


def _contract(
    shape_index: int = 0,
    *,
    backend: str = BACKEND_FUSED_CTA_TEMPORAL_V2,
    B: int = 3,
    temporal_k: int = 1,
    total_ticks: int = 17,
) -> FullEvolutionCudaWorkhorseContractV2:
    return FullEvolutionCudaWorkhorseContractV2(
        shape_id=SUPPORTED_WORKHORSE_SHAPES_V2[shape_index].shape_id,
        backend_id=backend,
        B=B,
        temporal_k=temporal_k,
        total_ticks=total_ticks,
        start_next_tick=11,
    )


def test_world_identity_binding_is_ordered_and_packing_independent() -> None:
    binding = workhorse_world_identity_binding_v2(
        world_uids=[9003, 9001, 9002],
        rng_keys=[71, 72, 73],
    )

    assert validate_workhorse_world_identity_binding_v2(
        binding, expected_world_count=3
    ) == binding
    assert binding["packing_independent_fields"] == ["world_uids", "rng_keys"]
    assert {"physical_slot", "B", "K", "device"}.issubset(
        binding["forbidden_identity_inputs"]
    )
    reordered = workhorse_world_identity_binding_v2(
        world_uids=[9001, 9002, 9003],
        rng_keys=[72, 73, 71],
    )
    assert reordered["binding_sha256"] != binding["binding_sha256"]


@pytest.mark.parametrize(
    ("world_uids", "rng_keys", "message"),
    [
        ([1, 1], [2, 3], "unique"),
        ([0], [2], "positive"),
        ([1], [-1], "nonnegative"),
        ([1, 2], [3], "align"),
    ],
)
def test_world_identity_binding_rejects_ambiguous_carriers(
    world_uids: list[int], rng_keys: list[int], message: str
) -> None:
    with pytest.raises(FullEvolutionWorkhorseContractError, match=message):
        workhorse_world_identity_binding_v2(
            world_uids=world_uids,
            rng_keys=rng_keys,
        )


def _owner_components(
    contract: FullEvolutionCudaWorkhorseContractV2,
    *,
    graph_object_bytes: int = 0,
    graph_exec_bytes: int = 0,
) -> WorkhorseOwnerComponentsV2:
    return WorkhorseOwnerComponentsV2(
        backend_id=contract.backend_id,
        temporal_k=contract.temporal_k,
        graph_object_bytes=graph_object_bytes,
        graph_exec_bytes=graph_exec_bytes,
        allocator_owner_overhead_bytes=4096,
        driver_unattributed_owner_bytes=2048,
        process_baseline_bytes=8192,
        checkpoint_device_staging_bytes_per_world=0,
        measurement_receipt_sha256="e" * 64,
    )


def _authority_response(
    request: dict[str, object],
    *,
    effect_kind: str | None = None,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build an exact fake outer-adapter projection for host contract tests."""
    started = (
        request["phase"] != "PLAN_REJECTION"
        if effect_kind is None
        else effect_kind == "started"
    )
    now = datetime.now(timezone.utc)
    response: dict[str, object] = {
        "schema_version": WORKHORSE_INJECTED_ATTEMPT_AUTHORITY_SCHEMA_V2,
        "phase": request["phase"],
        "authority_request_sha256": request["authority_request_sha256"],
        "effect_state": ("STARTED_LIVE_EXCLUSIVE" if started else "PREPARED_NO_EFFECT"),
        "scheduler_validation_status": (
            "AUTHORITATIVE_BUNDLE_VALID" if started else "PREPARED_CLAIM_VALID"
        ),
        "scheduler_authority_source": (
            "SCHEDULER_ISSUED_BUNDLE" if started else "SCHEDULER_PREPARED_NO_EFFECT"
        ),
        "scheduler_owner_id": "scheduler-owner-v2",
        "scheduler_owner_digest": "2" * 64,
        "execution_owner_id": "execution-owner-v2",
        "execution_owner_digest": "3" * 64,
        "run_id": "RUN-WORKHORSE-V2" if started else None,
        "attempt_id": "ATTEMPT-WORKHORSE-V2" if started else None,
        "claim_digest": "4" * 64,
        "prepared_claim_path": None if started else "/authority/prepared.json",
        "prepared_claim_file_sha256": None if started else "5" * 64,
        "authoritative_bundle_path": ("/authority/started.json" if started else None),
        "authoritative_bundle_file_sha256": "6" * 64 if started else None,
        "authoritative_bundle_digest": "7" * 64 if started else None,
        "attempt_start_present": started,
        "execution_effect_present": started,
        "execution_authority_granted": started,
        "resource_start_lease_digest": "8" * 64 if started else None,
        "runtime_resource_lease_digest": "9" * 64 if started else None,
        "runtime_resource_head_digest": "a" * 64 if started else None,
        "runtime_resource_lease_expires_at_utc": (
            (now + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
            if started
            else None
        ),
        "continuation_lease_digest": "b" * 64 if started else None,
        "fencing_token_digest": "c" * 64 if started else None,
        "continuation_lease_epoch": 1 if started else None,
        "continuation_head_digest": "d" * 64 if started else None,
        "lease_expires_at_utc": (
            (now + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
            if started
            else None
        ),
        "profile_id": request["profile_id"],
        "route_id": request["route_id"],
        "backend_id": request["backend_id"],
        "B": request["B"],
        "temporal_k": request["temporal_k"],
        "artifact_sha256": request["artifact_sha256"],
        "sealed_artifact_identity_sha256": request["sealed_artifact_identity_sha256"],
        "device_uuid": request["device_uuid"],
        "owner_pid": request["owner_pid"],
        "owner_process_start_ticks": request["owner_process_start_ticks"],
        "visible_compute_pids": [request["owner_pid"]] if started else [],
        "foreign_compute_pids": [],
        "accounting_mode": "NVML_PER_PROCESS" if started else None,
        "owner_process_bytes": 1024 if started else None,
        "global_device_bytes": 2048 if started else None,
        "device_free_bytes": 7 * 1024**3 if started else None,
        "device_total_bytes": 8 * 1024**3 if started else None,
        "resource_observation_digest": None,
        "observed_at_utc": now.isoformat().replace("+00:00", "Z"),
        "outer_adapter_id": "test-outer-adapter-v2",
        "outer_adapter_source_sha256": "e" * 64,
    }
    response.update(overrides or {})
    if not overrides or "resource_observation_digest" not in overrides:
        response["resource_observation_digest"] = (
            workhorse_resource_observation_sha256_v2(response)
        )
    return {**response, "response_body_sha256": canonical_sha256(response)}


def _attempt_binding(
    contract: FullEvolutionCudaWorkhorseContractV2,
    *,
    phase: str = "EXECUTION_ROW",
    artifact_sha256: str = "a" * 64,
    sealed_artifact_identity_sha256: str = "b" * 64,
    device_uuid: str = "GPU-test",
    candidate_envelope_sha256: str | None = "c" * 64,
    prior_lineage_sha256: str | None = None,
    source_checkpoint_identity_sha256: str | None = None,
    response_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    request = workhorse_attempt_authority_request_v2(
        phase=phase,
        contract=contract,
        artifact_sha256=artifact_sha256,
        sealed_artifact_identity_sha256=sealed_artifact_identity_sha256,
        device_uuid=device_uuid,
        candidate_envelope_sha256=(
            candidate_envelope_sha256
            if phase in {"EXECUTION_ROW", "PLAN_REJECTION", "CLOSEOUT"}
            else None
        ),
        governing_contracts=workhorse_governing_contract_bindings_v2(),
        prior_lineage_sha256=prior_lineage_sha256,
        source_checkpoint_identity_sha256=source_checkpoint_identity_sha256,
    )
    return _historical_binding_from_request(
        request, response_overrides=response_overrides
    )


def _historical_binding_from_request(
    request: dict[str, object],
    *,
    response_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Seal test-only historical evidence without executing an adapter."""
    response = validate_injected_workhorse_attempt_authority_v2(
        _authority_response(request, overrides=response_overrides),
        request=request,
    )
    lineage = (
        None
        if request["phase"] == "PLAN_REJECTION"
        else workhorse_attempt_lineage_sha256_v2(response)
    )
    body = {
        "schema_version": "eiel_workhorse_attempt_authority_binding_v2",
        "request": request,
        "response": response,
        "attempt_lineage_sha256": lineage,
    }
    return {**body, "authority_binding_sha256": canonical_sha256(body)}


def _runtime_resource_fixture(
    shape,
    backend_id: str,
    *,
    active_blocks_per_sm: int = 4,
    registers_per_thread: int = 32,
) -> dict[str, object]:
    backend = next(
        item for item in WORKHORSE_BACKENDS_V2 if item.backend_id == backend_id
    )
    topology = backend.topology_payload(shape)
    kernels = topology.get("kernels_per_dispatch") or topology["kernels_per_tick"]
    return {
        "runtime_inspected": True,
        "shape_id": shape.shape_id,
        "profile_id": shape.profile_id,
        "route_id": shape.route_id,
        "backend_id": backend_id,
        "multiprocessor_count": 22,
        "maximum_grid_x": 2_147_483_647,
        "maximum_grid_y": 65_535,
        "maximum_threads_per_block": 1024,
        "maximum_threads_per_sm": 2048,
        "registers_per_sm": 65_536,
        "shared_memory_per_sm": 65_536,
        "maximum_blocks_per_sm": 16,
        "warp_size": 32,
        "compiled_kernel_resources": [
            {
                "kernel": kernel,
                "registers_per_thread": registers_per_thread,
                "static_shared_bytes": 0,
                "local_bytes_per_thread": 0,
                "maximum_threads_per_block": 1024,
                "active_blocks_per_sm_at_compiled_block": active_blocks_per_sm,
            }
            for kernel in kernels
        ],
    }


def _graph_resource_fixture(shape, temporal_k: int) -> dict[str, object]:
    backend = next(
        item
        for item in WORKHORSE_BACKENDS_V2
        if item.backend_id == BACKEND_PHASE_GRAPH_V2
    )
    sequence = backend.topology_payload(shape)["kernels_per_tick"] * temporal_k
    nodes: list[dict[str, object]] = [
        {
            "topological_rank": 0,
            "node_type": "memset",
            "name": "event_count_memset_v2",
        }
    ]
    nodes.extend(
        {
            "topological_rank": rank,
            "node_type": "kernel",
            "kernel": kernel,
            "grid": (
                (
                    1
                    if kernel
                    in {
                        "compact_birth_events_v3",
                        "compact_death_events_v3",
                        "rank_prefix_phase_v2",
                        "mate_spawn_phase_v2",
                        "publish_phase_v2",
                    }
                    else (
                        shape.site_tiles
                        if kernel == "environment_phase_v2"
                        else shape.agent_tiles
                    )
                ),
                1,
                1,
            ),
            "block": (256, 1, 1),
        }
        for rank, kernel in enumerate(sequence, start=1)
    )
    return {
        "runtime_inspected": True,
        "shape_id": shape.shape_id,
        "backend_id": BACKEND_PHASE_GRAPH_V2,
        "B": 1,
        "temporal_k": temporal_k,
        "kernel_node_count": len(sequence),
        "memset_node_count": 1,
        "node_count": len(sequence) + 1,
        "edge_count": len(sequence),
        "linear_chain": True,
        "kernel_sequence_exact": True,
        "kernel_geometry_exact": True,
        "graph_object_device_bytes": 64 * 1024,
        "graph_exec_device_bytes": 128 * 1024,
        "nodes": nodes,
    }


def _topology_binding(shape, backend_id: str, temporal_k: int, **resource_changes):
    runtime = _runtime_resource_fixture(shape, backend_id, **resource_changes)
    graph = (
        _graph_resource_fixture(shape, temporal_k)
        if backend_id == BACKEND_PHASE_GRAPH_V2
        else None
    )
    return workhorse_topology_resource_binding_v2(
        shape,
        backend_id=backend_id,
        temporal_k=temporal_k,
        runtime_inspection=runtime,
        graph_inspection=graph,
    )


_FIELD_FORMATS = {"u8": "B", "u16": "H", "u32": "I", "i32": "i", "u64": "Q"}
_EVENT = struct.Struct("<QQQQQIHBB")


def _blank_raw_dataset(layout, *, B: int, temporal_k: int, start: int):
    sizes = {
        "state": B * layout.state_bytes_per_world,
        "scratch": B * layout.scratch_bytes_per_world,
        "events": B * temporal_k * layout.event_bytes_per_world_tick,
        "counts": B * 4,
    }
    return {
        "B": B,
        "temporal_k": temporal_k,
        "start_next_absolute_tick": start,
        **{f"input_{name}": bytearray(size) for name, size in sizes.items()},
        **{name: bytearray(size) for name, size in sizes.items()},
    }


def _set_field(dataset, layout, name: str, values, *, world: int = 0) -> None:
    field = layout.field(name)
    materialized = tuple(values)
    assert len(materialized) <= field.count
    offset = world * layout.state_bytes_per_world + field.offset
    if field.dtype == "u8":
        dataset["state"][offset : offset + len(materialized)] = bytes(materialized)
    elif materialized:
        struct.pack_into(
            f"<{len(materialized)}{_FIELD_FORMATS[field.dtype]}",
            dataset["state"],
            offset,
            *materialized,
        )


def _set_event(
    dataset,
    layout,
    *,
    row: int,
    tick: int,
    kind: int,
    slot: int,
    agent_id: int,
    parent_a_id: int = 0,
    parent_b_id: int = 0,
    site: int = 0,
) -> None:
    offset = row * 48
    _EVENT.pack_into(
        dataset["events"],
        offset,
        tick,
        (kind << 48) | slot,
        agent_id,
        parent_a_id,
        parent_b_id,
        site,
        kind,
        1,
        0,
    )


def _freeze_dataset(dataset):
    return {
        **dataset,
        **{
            component: bytes(dataset[component])
            for component in (
                "input_state",
                "input_scratch",
                "input_events",
                "input_counts",
                "state",
                "scratch",
                "events",
                "counts",
            )
        },
    }


def _valid_semantic_datasets():
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    layout = workhorse_layout_v2(shape)

    parity = _blank_raw_dataset(layout, B=2, temporal_k=2, start=71)
    for world in range(2):
        header = [0] * 15
        header[0] = 90_001 + world
        header[1] = 73
        header[13] = 17
        header[14] = 4
        _set_field(parity, layout, "header", header, world=world)
    parity = _freeze_dataset(parity)
    k_split = {
        "B": 1,
        "temporal_k": 2,
        "start_next_absolute_tick": 71,
        "input_state": parity["input_state"][: layout.state_bytes_per_world],
        "input_scratch": parity["input_scratch"][: layout.scratch_bytes_per_world],
        "input_events": parity["input_events"][: 2 * layout.event_bytes_per_world_tick],
        "input_counts": parity["input_counts"][:4],
        "state": parity["state"][: layout.state_bytes_per_world],
        "scratch": parity["scratch"][: layout.scratch_bytes_per_world],
        "events": parity["events"][: 2 * layout.event_bytes_per_world_tick],
        "counts": parity["counts"][:4],
    }

    baseline = _blank_raw_dataset(layout, B=1, temporal_k=1, start=91)
    header = [0] * 15
    header[0], header[1], header[3], header[13], header[14] = 92_001, 92, 3, 19, 4
    _set_field(baseline, layout, "header", header)
    _set_field(baseline, layout, "resource", [2])
    _set_field(baseline, layout, "patch_field", [4])
    _set_field(baseline, layout, "wdl_active", [1])
    _set_field(baseline, layout, "jungle", [1])
    _set_field(baseline, layout, "eed_field", [2])
    _set_field(baseline, layout, "alive", [1, 1, 1])
    _set_field(baseline, layout, "active_slots_by_uid", [0, 1, 2])
    _set_field(baseline, layout, "position", [1, 2, 5])
    _set_field(baseline, layout, "energy", [300, 300, 128])
    _set_field(baseline, layout, "health", [250, 251, 256])
    _set_field(baseline, layout, "agent_id", [10, 20, 30])
    _set_field(baseline, layout, "parent_a_id", [0, 0, 10])
    _set_field(baseline, layout, "parent_b_id", [0, 0, 20])
    genome_a = bytearray(shape.capacity * 32)
    genome_b = bytearray(shape.capacity * 32)
    genome_a[0:32] = bytes([10] * 32)
    genome_b[0:32] = bytes([20] * 32)
    genome_a[32:64] = bytes([30] * 32)
    genome_b[32:64] = bytes([40] * 32)
    child_a_nomut = bytes(10 if locus % 2 == 0 else 20 for locus in range(32))
    child_b = bytes(30 if locus % 2 == 0 else 40 for locus in range(32))
    child_a = bytearray(child_a_nomut)
    child_a[0] ^= 0x80
    genome_a[64:96] = child_a
    genome_b[64:96] = child_b
    expressed = bytearray(shape.capacity * 32)
    for slot in range(3):
        for locus in range(32):
            index = slot * 32 + locus
            expressed[index] = (genome_a[index] + genome_b[index]) // 2
    _set_field(baseline, layout, "genome_a", genome_a)
    _set_field(baseline, layout, "genome_b", genome_b)
    _set_field(baseline, layout, "expressed", expressed)
    _set_event(
        baseline,
        layout,
        row=3,
        tick=91,
        kind=1,
        slot=3,
        agent_id=40,
        site=6,
    )
    _set_event(
        baseline,
        layout,
        row=shape.capacity + 2,
        tick=91,
        kind=2,
        slot=2,
        agent_id=30,
        parent_a_id=10,
        parent_b_id=20,
        site=5,
    )
    struct.pack_into("<i", baseline["counts"], 0, 2)
    baseline = _freeze_dataset(baseline)

    mutation_disabled = {**baseline, "state": bytearray(baseline["state"])}
    child_offset = layout.field("genome_a").offset + 2 * 32
    mutation_disabled["state"][child_offset : child_offset + 32] = child_a_nomut
    expressed_offset = layout.field("expressed").offset + 2 * 32
    mutation_disabled["state"][expressed_offset : expressed_offset + 32] = bytes(
        (child_a_nomut[locus] + child_b[locus]) // 2 for locus in range(32)
    )
    mutation_disabled = _freeze_dataset(mutation_disabled)

    recombination_disabled = {
        **mutation_disabled,
        "state": bytearray(mutation_disabled["state"]),
    }
    recombination_disabled["state"][child_offset : child_offset + 32] = bytes([10] * 32)
    recombination_disabled = _freeze_dataset(recombination_disabled)

    def changed_copy(field: str, values):
        row = {**baseline, "state": bytearray(baseline["state"])}
        _set_field(row, layout, field, values)
        return _freeze_dataset(row)

    patch_disabled = {**baseline, "state": bytearray(baseline["state"])}
    _set_field(patch_disabled, layout, "patch_field", [0])
    _set_field(patch_disabled, layout, "resource", [1])
    patch_disabled = _freeze_dataset(patch_disabled)
    wdl_disabled = changed_copy("health", [249, 251, 256])
    jungle_disabled = changed_copy("health", [250, 249, 256])
    eed_causal_disabled = changed_copy("energy", [300, 300, 120])
    birth_disabled = changed_copy("alive", [1, 1, 0])

    eed_enabled = baseline
    eed_disabled = changed_copy("energy", [300, 300, 100])
    wdl_rows = []
    for active_value in (0, 0, 1):
        row = _blank_raw_dataset(layout, B=1, temporal_k=1, start=95)
        _set_field(row, layout, "wdl_active", [active_value])
        wdl_rows.append(_freeze_dataset(row))
    for index in (1, 2):
        wdl_rows[index] = {
            **wdl_rows[index],
            **{
                f"input_{component}": wdl_rows[index - 1][component]
                for component in ("state", "scratch", "events", "counts")
            },
        }

    overflow = _blank_raw_dataset(layout, B=1, temporal_k=1, start=101)
    overflow_header = [0] * 15
    overflow_header[0], overflow_header[1], overflow_header[13], overflow_header[14] = (
        93_001,
        102,
        23,
        4,
    )
    _set_field(overflow, layout, "header", overflow_header)
    _set_event(
        overflow,
        layout,
        row=shape.capacity + 4,
        tick=101,
        kind=2,
        slot=4,
        agent_id=50,
        site=8,
    )
    struct.pack_into("<i", overflow["counts"], 0, 1)
    overflow = _freeze_dataset(overflow)

    middle = _blank_raw_dataset(layout, B=1, temporal_k=1, start=31)
    middle_header = [0] * 20
    middle_header[0], middle_header[1], middle_header[2], middle_header[3] = (
        95_500,
        31,
        105,
        4,
    )
    middle_header[13], middle_header[14] = 45, 4
    _set_field(middle, layout, "header", middle_header)
    _set_field(middle, layout, "alive", [1, 1, 1, 1])
    _set_field(middle, layout, "active_slots_by_uid", [0, 1, 2, 3])
    _set_field(middle, layout, "energy", [600, 600, 600, 64])
    _set_field(middle, layout, "health", [256, 0, 256, 256])
    _set_field(middle, layout, "agent_id", [101, 102, 103, 104])
    middle["input_state"] = bytearray(middle["state"])
    middle_output_header = list(middle_header)
    middle_output_header[1] = 32
    middle_output_header[2] = 106
    middle_output_header[15] = 1
    middle_output_header[16] = 1
    middle_output_header[19] = 2
    _set_field(middle, layout, "header", middle_output_header)
    _set_field(middle, layout, "active_slots_by_uid", [0, 2, 3, 1])
    _set_field(middle, layout, "agent_id", [101, 105, 103, 104])
    _set_field(middle, layout, "parent_a_id", [0, 101, 0, 0])
    _set_field(middle, layout, "parent_b_id", [0, 103, 0, 0])
    _set_event(
        middle,
        layout,
        row=1,
        tick=31,
        kind=1,
        slot=1,
        agent_id=102,
        site=130,
    )
    _set_event(
        middle,
        layout,
        row=shape.capacity + 1,
        tick=31,
        kind=2,
        slot=1,
        agent_id=105,
        parent_a_id=101,
        parent_b_id=103,
        site=1,
    )
    struct.pack_into("<i", middle["counts"], 0, 2)
    middle = _freeze_dataset(middle)

    checkpoint = _blank_raw_dataset(layout, B=1, temporal_k=1, start=110)
    checkpoint_header = [0] * 15
    checkpoint_header[0], checkpoint_header[1], checkpoint_header[3] = 94_001, 111, 3
    checkpoint_header[13], checkpoint_header[14] = 29, 4
    _set_field(checkpoint, layout, "header", checkpoint_header)
    _set_field(checkpoint, layout, "alive", [0, 0, 1, 0, 0, 1, 0, 1])
    _set_field(checkpoint, layout, "active_slots_by_uid", [5, 7, 2])
    ids = [0] * 8
    ids[2], ids[5], ids[7] = 30, 10, 20
    _set_field(checkpoint, layout, "agent_id", ids)
    _set_event(
        checkpoint,
        layout,
        row=shape.capacity + 2,
        tick=110,
        kind=2,
        slot=2,
        agent_id=30,
        parent_a_id=10,
        parent_b_id=20,
        site=5,
    )
    struct.pack_into("<i", checkpoint["counts"], 0, 1)
    checkpoint = _freeze_dataset(checkpoint)
    corrupted_state = bytearray(checkpoint["state"])
    occupancy = layout.field("occupancy")
    corrupted_state[occupancy.offset : occupancy.offset + occupancy.nbytes] = bytes(
        occupancy.nbytes
    )
    corrupted_state[occupancy.offset] = 0xA5
    reconstructed = {
        **checkpoint,
        "input_state": bytes(corrupted_state),
        "input_scratch": bytes([0xA5]) * layout.scratch_bytes_per_world,
        "input_events": checkpoint["events"],
        "input_counts": checkpoint["counts"],
    }
    continuation = _freeze_dataset(
        _blank_raw_dataset(layout, B=1, temporal_k=1, start=111)
    )
    invalid_before = _blank_raw_dataset(layout, B=1, temporal_k=1, start=120)
    invalid_header = [0] * 15
    invalid_header[0], invalid_header[1], invalid_header[4] = 95_001, 120, 7
    invalid_header[13], invalid_header[14] = 31, 4
    _set_field(invalid_before, layout, "header", invalid_header)
    invalid_before = _freeze_dataset(invalid_before)
    invalid_after = {
        **invalid_before,
        "input_state": invalid_before["state"],
        "input_scratch": invalid_before["scratch"],
        "input_events": invalid_before["events"],
        "input_counts": invalid_before["counts"],
    }
    transaction = _blank_raw_dataset(layout, B=1, temporal_k=2, start=130)
    transaction_header = [0] * 15
    transaction_header[0], transaction_header[1], transaction_header[4] = 96_001, 130, 1
    transaction_header[13], transaction_header[14] = 37, 4
    _set_field(transaction, layout, "header", transaction_header)
    input_header = list(transaction_header)
    input_header[4] = 0
    struct.pack_into("<15Q", transaction["input_state"], 0, *input_header)
    transaction = _freeze_dataset(transaction)

    return {
        "parity_reference": parity,
        "parity_candidate": parity,
        "b_whole": parity,
        "b_split": parity,
        "k_split": k_split,
        "causal_baseline": baseline,
        "causal_patch_disabled": patch_disabled,
        "causal_wdl_disabled": wdl_disabled,
        "causal_jungle_disabled": jungle_disabled,
        "causal_eed_disabled": eed_causal_disabled,
        "causal_birth_disabled": birth_disabled,
        "causal_mutation_disabled": mutation_disabled,
        "causal_recombination_disabled": recombination_disabled,
        "eed_enabled": eed_enabled,
        "eed_disabled": eed_disabled,
        "wdl_tick1": wdl_rows[0],
        "wdl_tick2": wdl_rows[1],
        "wdl_tick3": wdl_rows[2],
        "overflow_reference": overflow,
        "overflow_candidate": overflow,
        "middle_survivor_reference": middle,
        "middle_survivor_candidate": middle,
        "restore_checkpoint": checkpoint,
        "restore_reconstructed": reconstructed,
        "restore_same_continuation": continuation,
        "restore_fresh_continuation": continuation,
        "restore_invalid_before": invalid_before,
        "restore_invalid_after": invalid_after,
        "transaction_after": transaction,
    }


def _corrupt_semantic_family(datasets, family: str):
    rows = {name: dict(dataset) for name, dataset in datasets.items()}
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    layout = workhorse_layout_v2(shape)
    if family == "parity":
        payload = bytearray(rows["parity_candidate"]["state"])
        payload[layout.field("resource").offset] ^= 1
        rows["parity_candidate"]["state"] = bytes(payload)
    elif family == "parity_input":
        payload = bytearray(rows["parity_candidate"]["input_state"])
        payload[layout.field("resource").offset] ^= 1
        rows["parity_candidate"]["input_state"] = bytes(payload)
    elif family == "B_partition":
        payload = bytearray(rows["b_split"]["scratch"])
        payload[0] ^= 1
        rows["b_split"]["scratch"] = bytes(payload)
    elif family == "K_partition":
        payload = bytearray(rows["k_split"]["events"])
        payload[0] ^= 1
        rows["k_split"]["events"] = bytes(payload)
    elif family == "events":
        payload = bytearray(rows["causal_baseline"]["events"])
        struct.pack_into("<Q", payload, 3 * 48 + 8, 0)
        rows["causal_baseline"]["events"] = bytes(payload)
    elif family == "WDL":
        payload = bytearray(rows["wdl_tick3"]["state"])
        struct.pack_into("<H", payload, layout.field("wdl_active").offset, 0)
        rows["wdl_tick3"]["state"] = bytes(payload)
    elif family == "Jungle":
        rows["causal_jungle_disabled"]["state"] = rows["causal_baseline"]["state"]
    elif family == "EED":
        rows["causal_eed_disabled"]["state"] = rows["causal_baseline"]["state"]
    elif family == "Patch":
        rows["causal_patch_disabled"]["state"] = rows["causal_baseline"]["state"]
    elif family == "Patch_input":
        payload = bytearray(rows["causal_patch_disabled"]["input_scratch"])
        payload[0] ^= 1
        rows["causal_patch_disabled"]["input_scratch"] = bytes(payload)
    elif family == "birth":
        rows["causal_birth_disabled"]["state"] = rows["causal_baseline"]["state"]
    elif family == "mutation":
        rows["causal_mutation_disabled"]["state"] = rows["causal_baseline"]["state"]
    elif family == "recombination":
        rows["causal_recombination_disabled"]["state"] = rows[
            "causal_mutation_disabled"
        ]["state"]
    elif family == "overflow":
        rows["overflow_candidate"]["events"] = bytes(
            len(rows["overflow_candidate"]["events"])
        )
        rows["overflow_candidate"]["counts"] = bytes(4)
    elif family == "middle_birth":
        payload = bytearray(rows["middle_survivor_candidate"]["events"])
        struct.pack_into("<Q", payload, (shape.capacity + 1) * 48 + 32, 104)
        rows["middle_survivor_candidate"]["events"] = bytes(payload)
    else:  # pragma: no cover - test helper guard
        raise AssertionError(f"unknown semantic corruption family {family}")
    return rows


def test_exact_shape_matrix_has_distinct_executable_profiles_and_routes() -> None:
    scalable_k = (1, 2, 4, 8, 16, 32, 64)
    expected = (
        ("w64h64p1024c2048", 64, 64, 1024, 2048, 544, scalable_k),
        ("w128h80p2560c5120", 128, 80, 2560, 5120, 256, scalable_k),
        ("w160h160p6400c12800", 160, 160, 6400, 12800, 128, scalable_k),
        ("w256h200p12800c25600", 256, 200, 12800, 25600, 64, scalable_k),
        ("w320h320p25600c51200", 320, 320, 25600, 51200, 32, scalable_k),
    )
    actual = tuple(
        (
            shape.shape_id,
            shape.width,
            shape.height,
            shape.initial_population,
            shape.capacity,
            shape.batch_seed,
            shape.temporal_k_candidates,
        )
        for shape in SUPPORTED_WORKHORSE_SHAPES_V2
    )
    assert actual == expected
    assert len({shape.profile_id for shape in SUPPORTED_WORKHORSE_SHAPES_V2}) == 5
    assert len({shape.route_id for shape in SUPPORTED_WORKHORSE_SHAPES_V2}) == 5
    assert all(
        shape.shape_id in shape.profile_id for shape in SUPPORTED_WORKHORSE_SHAPES_V2
    )
    assert all(
        shape.shape_id in shape.route_id for shape in SUPPORTED_WORKHORSE_SHAPES_V2
    )


def test_backend_supported_calibration_k_is_exact() -> None:
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    assert workhorse_backend_temporal_k_candidates_v2(
        shape, BACKEND_REFERENCE_PHASED_V2
    ) == (1,)
    assert workhorse_backend_temporal_k_candidates_v2(
        shape, BACKEND_PHASE_GRAPH_V2
    ) == shape.temporal_k_candidates
    assert workhorse_backend_temporal_k_candidates_v2(
        shape, BACKEND_FUSED_CTA_TEMPORAL_V2
    ) == shape.temporal_k_candidates


def test_semantic_hash_is_shared_but_backend_topology_is_not() -> None:
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    contracts = [
        FullEvolutionCudaWorkhorseContractV2(
            shape_id=shape.shape_id,
            backend_id=backend.backend_id,
            B=2,
            temporal_k=1,
            total_ticks=2,
        )
        for backend in WORKHORSE_BACKENDS_V2
    ]
    assert {contract.semantic_transition_sha256 for contract in contracts} == {
        WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2
    }
    assert len({contract.backend_topology_sha256 for contract in contracts}) == 3
    assert all(
        contract.capabilities == FULL_EVOLUTION_CAPABILITIES_V2
        for contract in contracts
    )


def test_truthful_workhorse_semantic_law_invalidates_all_fifteen_identities() -> None:
    semantic = workhorse_semantic_contract_v2()
    law = semantic["contract"]
    assert semantic["semantic_transition_sha256"] == (
        "f5abe56a2519f5b9b16b52559c27b19d34a1ad7a8bbe2c9f87dfb09e3c4773c8"
    )
    assert (
        "046e45859028e2fd2d966e68bcda1527b60506145dc23cdbf95c49177b13e854"
        in WORKHORSE_REJECTED_SEMANTIC_HASHES_V2
    )
    assert law["authority_scope"] == "workhorse_v2_only_no_micro_semantic_inheritance"
    assert law["policy_and_observation"]["present"] is False
    assert "even survivor ordinal" in law["ordinary_birth_law"]
    assert "fixed_direction_spawn" in law["tick_order"][6]
    assert law["wdl_delay_ticks"] == 2
    assert tuple(law["capabilities"]) == FULL_EVOLUTION_CAPABILITIES_V2
    assert WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2 not in (
        WORKHORSE_REJECTED_SEMANTIC_HASHES_V2
    )
    governing = workhorse_governing_contract_bindings_v2()
    assert set(governing) == {"workhorse_semantic_contract", "performance_contract"}

    identities = []
    for shape in SUPPORTED_WORKHORSE_SHAPES_V2:
        for backend in WORKHORSE_BACKENDS_V2:
            identity = workhorse_artifact_identity_v2(shape, backend.backend_id)
            current_hash = identity["identity_sha256"]
            current_body = {
                key: value
                for key, value in identity.items()
                if key != "identity_sha256"
            }
            for rejected_hash in WORKHORSE_REJECTED_SEMANTIC_HASHES_V2:
                old_body = {**current_body, "semantic_transition_sha256": rejected_hash}
                assert canonical_sha256(old_body) != current_hash
            identities.append(current_hash)
    assert len(identities) == len(set(identities)) == 15


def test_raw_semantic_evidence_derives_every_family_without_caller_booleans(
    tmp_path: Path,
) -> None:
    datasets = _valid_semantic_datasets()
    assert set(datasets) == set(WORKHORSE_SEMANTIC_DATASETS_V2)
    manifest = write_workhorse_semantic_evidence_bundle_v2(
        tmp_path / "evidence",
        shape_id=SUPPORTED_WORKHORSE_SHAPES_V2[0].shape_id,
        temporal_k=2,
        reference_artifact_sha256="a" * 64,
        candidate_artifact_sha256="b" * 64,
        datasets=datasets,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
    )
    checks, witness = derive_workhorse_semantic_checks_v2(manifest)
    assert set(checks) == WORKHORSE_EVIDENCE_DERIVED_CHECKS_V2
    assert all(checks.values()), {
        name: value for name, value in checks.items() if not value
    }
    assert witness["birth_rows"] == 1
    assert witness["death_rows"] == 1


@pytest.mark.parametrize(
    ("family", "expected_failed_check"),
    (
        ("parity", "same_shape_reference_state_byte_parity"),
        ("parity_input", "same_shape_reference_state_byte_parity"),
        ("B_partition", "B_partition_scratch_invariance"),
        ("K_partition", "K_partition_event_invariance"),
        ("events", "event_id_codec_exact"),
        ("WDL", "wdl_new_emission_delay_is_exactly_two_ticks"),
        ("Jungle", "jungle_paired_physiology_effect"),
        ("EED", "eed_paired_downstream_energy_effect"),
        ("Patch", "patch_paired_state_effect"),
        ("Patch_input", "patch_paired_state_effect"),
        ("birth", "ordinary_birth_paired_alive_effect"),
        ("mutation", "mutation_isolated_to_one_child_locus_and_cache"),
        ("recombination", "recombination_paired_genome_effect"),
        ("overflow", "overflow_truncates_to_single_free_slot"),
        (
            "middle_birth",
            "birth_pairs_adjacent_compacted_survivor_uid_order",
        ),
    ),
)
def test_resealed_raw_evidence_corruption_cannot_grant_family_pass(
    tmp_path: Path, family: str, expected_failed_check: str
) -> None:
    datasets = _corrupt_semantic_family(_valid_semantic_datasets(), family)
    manifest = write_workhorse_semantic_evidence_bundle_v2(
        tmp_path / family,
        shape_id=SUPPORTED_WORKHORSE_SHAPES_V2[0].shape_id,
        temporal_k=2,
        reference_artifact_sha256="a" * 64,
        candidate_artifact_sha256="b" * 64,
        datasets=datasets,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
    )
    checks, _ = derive_workhorse_semantic_checks_v2(manifest)
    assert checks[expected_failed_check] is False


def test_semantic_writer_rejects_caller_supplied_pass_bits(tmp_path: Path) -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse import (
        write_workhorse_semantic_gate_receipt_v2,
    )

    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="caller verdicts"):
        write_workhorse_semantic_gate_receipt_v2(
            {"checks": {"forged": True}, "decision": "SEMANTIC_GATE_SATISFIED"},
            tmp_path / "forged.json",
        )


def test_contract_roundtrip_and_generation_firewall() -> None:
    contract = _contract(B=68, temporal_k=4, total_ticks=16)
    assert (
        FullEvolutionCudaWorkhorseContractV2.from_mapping(contract.to_dict())
        == contract
    )
    with pytest.raises(FullEvolutionWorkhorseContractError, match="generation-shaped"):
        FullEvolutionCudaWorkhorseContractV2.from_mapping(
            {
                "shape_id": contract.shape_id,
                "backend_id": contract.backend_id,
                "B": 1,
                "temporal_k": 1,
                "total_ticks": 1,
                "generation_count": 1,
            }
        )
    with pytest.raises(FullEvolutionWorkhorseContractError, match="profile_id"):
        replace(contract, profile_id="common-profile")


def test_reference_is_k1_only_and_shape_k_envelopes_are_exact() -> None:
    with pytest.raises(FullEvolutionWorkhorseContractError, match="reference backend"):
        _contract(
            backend=BACKEND_REFERENCE_PHASED_V2,
            temporal_k=2,
        )
    assert _contract(shape_index=4, temporal_k=64).temporal_k == 64
    with pytest.raises(FullEvolutionWorkhorseContractError, match="temporal_k"):
        _contract(shape_index=4, temporal_k=128)


def test_u32_indices_reserve_empty_and_i64_arithmetic_fails_closed() -> None:
    with pytest.raises(FullEvolutionWorkhorseContractError, match="reserve UINT32_MAX"):
        WorkhorseCompileShapeV2("w4294967295h1p1c2", 4_294_967_295, 1, 1, 2, (1,), 1)
    with pytest.raises(FullEvolutionWorkhorseContractError, match="signed int64"):
        checked_add_i64(MAX_SIGNED_INT64, 1, "overflow")
    with pytest.raises(FullEvolutionWorkhorseContractError, match="signed int64"):
        checked_multiply_i64(MAX_SIGNED_INT64, 2, "overflow")


@pytest.mark.parametrize("shape", SUPPORTED_WORKHORSE_SHAPES_V2)
def test_layout_is_shape_specialized_linear_and_has_full_genome_cache(shape) -> None:
    layout = workhorse_layout_v2(shape)
    assert layout.shape is shape
    assert layout.field("occupancy").dtype == "u32"
    assert layout.field("position").dtype == "u32"
    assert layout.field("active_slots_by_uid").count == shape.capacity
    assert layout.field("genome_a").count == shape.capacity * 32
    assert layout.field("genome_b").count == shape.capacity * 32
    assert layout.field("expressed").count == shape.capacity * 32
    assert layout.field("move_winner_uid", scratch=True).count == shape.sites
    assert layout.state_bytes_per_world + layout.scratch_bytes_per_world < (
        40 * shape.sites + 176 * shape.capacity + 4096
    )


def test_event_staging_is_exact_single_2_c_page_and_k_horizon_independent() -> None:
    short = _contract(B=7, temporal_k=4, total_ticks=4)
    widest_k = replace(short, temporal_k=64, total_ticks=64)
    long = replace(short, total_ticks=100_000)
    owner = _owner_components(short)
    short_ledger = estimate_workhorse_resources_v2(short, owner_components=owner)
    widest_k_ledger = estimate_workhorse_resources_v2(
        widest_k, owner_components=_owner_components(widest_k)
    )
    long_ledger = estimate_workhorse_resources_v2(long, owner_components=owner)
    assert short_ledger == long_ledger
    assert short_ledger.event_bytes == 7 * 2 * short.shape.capacity * 48
    assert widest_k_ledger.event_bytes == short_ledger.event_bytes
    assert widest_k_ledger.event_count_bytes == short_ledger.event_count_bytes == 7 * 8
    assert short_ledger.to_dict()["horizon_dependent_resident_bytes"] == 0
    assert short_ledger.checkpoint_host_staging_slabs == 2
    assert short_ledger.checkpoint_host_staging_bytes == 2 * short_ledger.state_bytes
    assert short_ledger.checkpoint_host_peak_bytes == 3 * short_ledger.state_bytes
    assert short_ledger.checkpoint_device_staging_bytes == 0
    assert short_ledger.to_dict()["arbitrary_allowance_bytes"] == 0


def test_vram_admission_enforces_live_25_percent_headroom() -> None:
    contract = _contract(B=2)
    owner = _owner_components(contract)
    required = estimate_workhorse_resources_v2(
        contract, owner_components=owner
    ).incremental_device_bytes
    total = required * 4
    admitted = admit_workhorse_vram_v2(
        contract,
        free_bytes=total,
        total_device_bytes=total,
        owner_components=owner,
    )
    assert admitted.admitted
    assert admitted.reserve_bytes == total // 4
    rejected = admit_workhorse_vram_v2(
        contract,
        free_bytes=required,
        total_device_bytes=total,
        owner_components=owner,
    )
    assert not rejected.admitted
    assert rejected.reason == "WAIT_VRAM_HEADROOM"


def test_attempt_identity_binds_exact_live_admission_and_frozen_contracts() -> None:
    contract = _contract(B=2)
    owner = _owner_components(contract)
    ledger = estimate_workhorse_resources_v2(contract, owner_components=owner)
    total = ledger.incremental_device_bytes * 8
    admission = admit_workhorse_vram_v2(
        contract,
        free_bytes=total,
        total_device_bytes=total,
        owner_components=owner,
    )
    governing = workhorse_governing_contract_bindings_v2()
    authority = _attempt_binding(
        contract,
        response_overrides={
            "device_free_bytes": total,
            "device_total_bytes": total,
        },
    )
    identity = workhorse_attempt_identity_v2(
        contract,
        artifact_sha256="a" * 64,
        sealed_artifact_identity_sha256="b" * 64,
        device_uuid="GPU-test",
        target_sm="sm_75",
        resource_ledger=ledger,
        admission=admission,
        governing_contracts=governing,
        attempt_authority_binding=authority,
    )
    assert identity["live_vram_admission"] == admission.to_dict()
    assert identity["governing_contracts"] == governing
    changed_admission = admit_workhorse_vram_v2(
        contract,
        free_bytes=total - 1,
        total_device_bytes=total,
        owner_components=owner,
    )
    changed_authority = _attempt_binding(
        contract,
        response_overrides={
            "device_free_bytes": total - 1,
            "device_total_bytes": total,
        },
    )
    with pytest.raises(
        FullEvolutionWorkhorseContractError, match="Scheduler/lease authority"
    ):
        workhorse_attempt_identity_v2(
            contract,
            artifact_sha256="a" * 64,
            sealed_artifact_identity_sha256="b" * 64,
            device_uuid="GPU-test",
            target_sm="sm_75",
            resource_ledger=ledger,
            admission=changed_admission,
            governing_contracts=governing,
            attempt_authority_binding=authority,
        )
    changed = workhorse_attempt_identity_v2(
        contract,
        artifact_sha256="a" * 64,
        sealed_artifact_identity_sha256="b" * 64,
        device_uuid="GPU-test",
        target_sm="sm_75",
        resource_ledger=ledger,
        admission=changed_admission,
        governing_contracts=governing,
        attempt_authority_binding=changed_authority,
    )
    assert changed["attempt_identity_sha256"] != identity["attempt_identity_sha256"]
    with pytest.raises(
        FullEvolutionWorkhorseContractError, match="semantic and performance"
    ):
        workhorse_attempt_identity_v2(
            contract,
            artifact_sha256="a" * 64,
            sealed_artifact_identity_sha256="b" * 64,
            device_uuid="GPU-test",
            target_sm="sm_75",
            resource_ledger=ledger,
            admission=admission,
            governing_contracts={},
            attempt_authority_binding=authority,
        )
    forged_governing = json.loads(json.dumps(governing))
    forged_governing["performance_contract"]["sha256"] = "f" * 64
    with pytest.raises(FullEvolutionWorkhorseContractError, match="not current"):
        workhorse_attempt_identity_v2(
            contract,
            artifact_sha256="a" * 64,
            sealed_artifact_identity_sha256="b" * 64,
            device_uuid="GPU-test",
            target_sm="sm_75",
            resource_ledger=ledger,
            admission=admission,
            governing_contracts=forged_governing,
            attempt_authority_binding=authority,
        )


def test_injected_attempt_authority_separates_prepared_rejection_from_execution() -> (
    None
):
    contract = _contract(B=2)
    started = _attempt_binding(contract, phase="EXECUTION_ROW")
    reopened = validate_workhorse_attempt_authority_binding_v2(started)
    assert (
        reopened["request"]["authority_request_sha256"]
        == reopened["response"]["authority_request_sha256"]
    )
    assert reopened["authority_binding_sha256"] == canonical_sha256(
        {
            key: value
            for key, value in reopened.items()
            if key != "authority_binding_sha256"
        }
    )
    assert reopened["response"]["execution_authority_granted"] is True

    rejection = _attempt_binding(contract, phase="PLAN_REJECTION")
    rejection = validate_workhorse_attempt_authority_binding_v2(rejection)
    assert rejection["attempt_lineage_sha256"] is None
    assert rejection["response"]["effect_state"] == "PREPARED_NO_EFFECT"
    assert rejection["response"]["attempt_start_present"] is False
    assert rejection["response"]["execution_authority_granted"] is False
    assert rejection["response"]["visible_compute_pids"] == []

    plan_request = workhorse_attempt_authority_request_v2(
        phase="PLAN_REJECTION",
        contract=contract,
        artifact_sha256="a" * 64,
        sealed_artifact_identity_sha256="b" * 64,
        device_uuid="GPU-test",
        candidate_envelope_sha256="c" * 64,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
    )
    with pytest.raises(WorkhorseAttemptAuthorityError, match="retired"):
        resolve_injected_workhorse_attempt_authority_v2(
            lambda request: _authority_response(dict(request), effect_kind="started"),
            request=plan_request,
        )
    with pytest.raises(WorkhorseAttemptAuthorityError, match="retired"):
        resolve_injected_workhorse_attempt_authority_v2(
            lambda request: _authority_response(
                dict(request),
                effect_kind="prepared",
                overrides={"device_free_bytes": 1},
            ),
            request=plan_request,
        )

    execution_request = workhorse_attempt_authority_request_v2(
        phase="EXECUTION_ROW",
        contract=contract,
        artifact_sha256="a" * 64,
        sealed_artifact_identity_sha256="b" * 64,
        device_uuid="GPU-test",
        candidate_envelope_sha256="c" * 64,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
    )
    with pytest.raises(WorkhorseAttemptAuthorityError):
        resolve_injected_workhorse_attempt_authority_v2(
            lambda request: _authority_response(dict(request), effect_kind="prepared"),
            request=execution_request,
        )


@pytest.mark.parametrize(
    "overrides",
    (
        {"device_uuid": "GPU-foreign"},
        {"owner_pid": os.getpid() + 10_000},
        {"foreign_compute_pids": [os.getpid() + 1]},
        {
            "lease_expires_at_utc": "2000-01-01T00:00:00Z",
            "runtime_resource_lease_expires_at_utc": "2000-01-01T00:00:00Z",
        },
        {"observed_at_utc": "2000-01-01T00:00:00Z"},
        {"device_free_bytes": 9 * 1024**3},
        {"device_total_bytes": 1024, "global_device_bytes": 2048},
        {"resource_observation_digest": "0" * 64},
    ),
)
def test_injected_attempt_authority_rejects_wrong_stale_or_foreign_projection(
    overrides: dict[str, object],
) -> None:
    contract = _contract(B=2)
    request = workhorse_attempt_authority_request_v2(
        phase="EXECUTION_ROW",
        contract=contract,
        artifact_sha256="a" * 64,
        sealed_artifact_identity_sha256="b" * 64,
        device_uuid="GPU-test",
        candidate_envelope_sha256="c" * 64,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
    )
    with pytest.raises(WorkhorseAttemptAuthorityError):
        resolve_injected_workhorse_attempt_authority_v2(
            lambda supplied: _authority_response(dict(supplied), overrides=overrides),
            request=request,
        )


def test_retired_attempt_callback_cannot_execute_or_substitute_request() -> None:
    contract = _contract(B=2)
    request = workhorse_attempt_authority_request_v2(
        phase="EXECUTION_ROW",
        contract=contract,
        artifact_sha256="a" * 64,
        sealed_artifact_identity_sha256="b" * 64,
        device_uuid="GPU-test",
        candidate_envelope_sha256="c" * 64,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
    )
    foreign_request = workhorse_attempt_authority_request_v2(
        phase="EXECUTION_ROW",
        contract=replace(contract, B=3),
        artifact_sha256="a" * 64,
        sealed_artifact_identity_sha256="b" * 64,
        device_uuid="GPU-test",
        candidate_envelope_sha256="c" * 64,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
    )
    foreign_response = _authority_response(foreign_request)
    with pytest.raises(WorkhorseAttemptAuthorityError, match="request binding"):
        validate_injected_workhorse_attempt_authority_v2(
            foreign_response,
            request=request,
        )

    invoked = False

    def hostile_adapter(_supplied):
        nonlocal invoked
        invoked = True
        return foreign_response

    with pytest.raises(WorkhorseAttemptAuthorityError, match="retired"):
        resolve_injected_workhorse_attempt_authority_v2(
            hostile_adapter,
            request=request,
        )
    assert invoked is False

    started = _attempt_binding(contract)
    closeout_request = workhorse_attempt_authority_request_v2(
        phase="CLOSEOUT",
        contract=contract,
        artifact_sha256="a" * 64,
        sealed_artifact_identity_sha256="b" * 64,
        device_uuid="GPU-test",
        candidate_envelope_sha256="c" * 64,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
        prior_lineage_sha256=started["attempt_lineage_sha256"],
    )
    closeout = _historical_binding_from_request(closeout_request)
    assert closeout["attempt_lineage_sha256"] == started["attempt_lineage_sha256"]
    substituted_request = {
        **closeout_request,
        "prior_lineage_sha256": "f" * 64,
    }
    substituted_body = {
        key: value
        for key, value in substituted_request.items()
        if key != "authority_request_sha256"
    }
    substituted_request["authority_request_sha256"] = canonical_sha256(substituted_body)
    with pytest.raises(WorkhorseAttemptAuthorityError, match="retired"):
        resolve_injected_workhorse_attempt_authority_v2(
            lambda supplied: _authority_response(dict(supplied)),
            request=substituted_request,
        )


def test_restore_execution_attempt_binds_exact_source_checkpoint_request() -> None:
    contract = _contract(B=1, temporal_k=1, total_ticks=1)
    checkpoint_a = "1" * 64
    checkpoint_b = "2" * 64
    request_a = workhorse_attempt_authority_request_v2(
        phase="EXECUTION_ROW",
        contract=contract,
        artifact_sha256="a" * 64,
        sealed_artifact_identity_sha256="b" * 64,
        device_uuid="GPU-test",
        candidate_envelope_sha256=None,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
        source_checkpoint_identity_sha256=checkpoint_a,
    )
    binding = _historical_binding_from_request(request_a)
    assert binding["request"]["source_checkpoint_identity_sha256"] == checkpoint_a
    assert (
        binding["response"]["authority_request_sha256"]
        == request_a["authority_request_sha256"]
    )

    request_b = workhorse_attempt_authority_request_v2(
        phase="EXECUTION_ROW",
        contract=contract,
        artifact_sha256="a" * 64,
        sealed_artifact_identity_sha256="b" * 64,
        device_uuid="GPU-test",
        candidate_envelope_sha256=None,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
        source_checkpoint_identity_sha256=checkpoint_b,
    )
    response_a = _authority_response(request_a)
    with pytest.raises(WorkhorseAttemptAuthorityError, match="request binding"):
        validate_injected_workhorse_attempt_authority_v2(
            response_a,
            request=request_b,
        )
    with pytest.raises(WorkhorseAttemptAuthorityError, match="only for"):
        workhorse_attempt_authority_request_v2(
            phase="PLAN_REJECTION",
            contract=contract,
            artifact_sha256="a" * 64,
            sealed_artifact_identity_sha256="b" * 64,
            device_uuid="GPU-test",
            candidate_envelope_sha256="c" * 64,
            governing_contracts=workhorse_governing_contract_bindings_v2(),
            source_checkpoint_identity_sha256=checkpoint_a,
        )


def test_semantic_attempt_lifecycle_rejects_cross_closeout_substitution() -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse.authority import (
        _validate_semantic_attempt_lifecycles_v2,
    )
    from engine.perf.cuda_full_evolution_v2.workhorse.probe import (
        _attempt_lifecycle_row_v2,
    )

    roles = (
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
    )
    checkpoint_identity = "9" * 64
    checkpoint_roles = {
        "restore_reconstruction",
        "restore_same_process_continuation",
        "restore_clean_continuation",
        "restore_fresh_process_continuation",
        "restore_invalid_fail_stop",
    }
    contract = _contract(B=1, temporal_k=1, total_ticks=3)
    lifecycles = []
    for index, role in enumerate(roles):
        overrides = {"attempt_id": f"ATTEMPT-WORKHORSE-V2-{index}"}
        execution = _attempt_binding(
            contract,
            phase="EXECUTION_ROW",
            source_checkpoint_identity_sha256=(
                checkpoint_identity if role in checkpoint_roles else None
            ),
            response_overrides=overrides,
        )
        closeout = _attempt_binding(
            contract,
            phase="CLOSEOUT",
            prior_lineage_sha256=execution["attempt_lineage_sha256"],
            response_overrides=overrides,
        )
        lifecycles.append(_attempt_lifecycle_row_v2(role, execution, closeout))
    value = {
        "reference": {"artifact_sha256": "a" * 64},
        "candidate": {"artifact_sha256": "a" * 64},
        "attempt_lifecycles": lifecycles,
    }
    runtime = {
        "device_uuid": "GPU-test",
        "total_device_bytes": 8 * 1024**3,
    }
    _validate_semantic_attempt_lifecycles_v2(
        value,
        runtime=runtime,
        source_checkpoint_identity_sha256=checkpoint_identity,
    )

    forged = dict(lifecycles[0])
    forged["closeout_attempt_authority"] = lifecycles[1]["closeout_attempt_authority"]
    forged_body = {
        key: item for key, item in forged.items() if key != "attempt_lifecycle_sha256"
    }
    forged["attempt_lifecycle_sha256"] = canonical_sha256(forged_body)
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="substituted"):
        _validate_semantic_attempt_lifecycles_v2(
            {**value, "attempt_lifecycles": [forged, *lifecycles[1:]]},
            runtime=runtime,
            source_checkpoint_identity_sha256=checkpoint_identity,
        )

    checkpoint_index = roles.index("restore_reconstruction")
    missing_execution = _attempt_binding(
        contract,
        phase="EXECUTION_ROW",
        response_overrides={"attempt_id": "ATTEMPT-MISSING-CHECKPOINT"},
    )
    missing_closeout = _attempt_binding(
        contract,
        phase="CLOSEOUT",
        prior_lineage_sha256=missing_execution["attempt_lineage_sha256"],
        response_overrides={"attempt_id": "ATTEMPT-MISSING-CHECKPOINT"},
    )
    missing_row = _attempt_lifecycle_row_v2(
        "restore_reconstruction", missing_execution, missing_closeout
    )
    missing_rows = list(lifecycles)
    missing_rows[checkpoint_index] = missing_row
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="substituted"):
        _validate_semantic_attempt_lifecycles_v2(
            {**value, "attempt_lifecycles": missing_rows},
            runtime=runtime,
            source_checkpoint_identity_sha256=checkpoint_identity,
        )


def test_same_process_restore_call_sites_bind_reopened_checkpoint_identity() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/workhorse/probe.py"
    ).read_text(encoding="utf-8")
    restore = source[
        source.index("def _restore_and_checkpoint_check(") : source.index(
            "def run_workhorse_semantic_gate_v2("
        )
    ]
    assert (
        restore.count(
            "source_checkpoint_identity_sha256=source_checkpoint_identity_sha256"
        )
        == 4
    )
    assert (
        'source_checkpoint_identity_sha256 = checkpoint["checkpoint_identity"]['
        in restore
    )


def test_b_candidates_include_exact_per_k_vram_maximum_and_regimes() -> None:
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    contract = _contract(B=1, temporal_k=1)
    owner = _owner_components(contract)
    total = 2 * 1024**3
    candidates = workhorse_b_candidates_v2(
        shape,
        backend_id=contract.backend_id,
        temporal_k=1,
        free_bytes=total,
        total_device_bytes=total,
        owner_components=owner,
        topology_resource_binding=_topology_binding(
            shape, contract.backend_id, temporal_k=1
        ),
    )
    admitted = [
        B
        for B in candidates
        if admit_workhorse_vram_v2(
            replace(contract, B=B),
            free_bytes=total,
            total_device_bytes=total,
            owner_components=owner,
        ).admitted
    ]
    maximum = max(admitted)
    one_ledger = estimate_workhorse_resources_v2(contract, owner_components=owner)
    fixed = (
        owner.graph_object_bytes
        + owner.graph_exec_bytes
        + owner.allocator_owner_overhead_bytes
        + owner.driver_unattributed_owner_bytes
    )
    per_world = one_ledger.incremental_device_bytes - fixed
    usable = total - total // 4
    expected_maximum = min(65_535, (usable - fixed) // per_world)
    assert maximum == expected_maximum
    assert {1, 2, 4, 8, 16, 32, 68, 136} <= set(candidates)
    assert {11, 22, 44, 88, 272, 544, 1088, expected_maximum} <= set(candidates)
    if maximum < 65_535:
        assert maximum + 1 in candidates


@pytest.mark.parametrize("shape_index", range(len(SUPPORTED_WORKHORSE_SHAPES_V2)))
def test_complete_candidate_envelope_has_every_compiled_k_and_exact_maximum(
    shape_index: int,
) -> None:
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[shape_index]
    owners = {
        temporal_k: _owner_components(
            _contract(shape_index=shape_index, B=1, temporal_k=temporal_k)
        )
        for temporal_k in shape.temporal_k_candidates
    }
    envelope = workhorse_candidate_envelope_v2(
        shape,
        backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
        free_bytes=6 * 1024**3,
        total_device_bytes=6 * 1024**3,
        owner_components_by_k=owners,
        topology_resource_bindings_by_k={
            temporal_k: _topology_binding(
                shape, BACKEND_FUSED_CTA_TEMPORAL_V2, temporal_k
            )
            for temporal_k in shape.temporal_k_candidates
        },
        live_observation_authority_binding_sha256="f" * 64,
    )
    assert envelope["complete"] is True
    assert [row["temporal_k"] for row in envelope["per_k"]] == list(
        shape.temporal_k_candidates
    )
    for row in envelope["per_k"]:
        admitted = [item["B"] for item in row["rows"] if item["admission"]["admitted"]]
        assert row["maximum_vram_admitted_B"] == max(admitted)


def test_reference_candidate_envelope_is_exactly_k1() -> None:
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    owner = _owner_components(
        _contract(backend=BACKEND_REFERENCE_PHASED_V2, temporal_k=1)
    )
    envelope = workhorse_candidate_envelope_v2(
        shape,
        backend_id=BACKEND_REFERENCE_PHASED_V2,
        free_bytes=4 * 1024**3,
        total_device_bytes=4 * 1024**3,
        owner_components_by_k={1: owner},
        topology_resource_bindings_by_k={
            1: _topology_binding(shape, BACKEND_REFERENCE_PHASED_V2, 1)
        },
        live_observation_authority_binding_sha256="f" * 64,
    )

    assert envelope["temporal_k"] == [1]
    assert [row["temporal_k"] for row in envelope["per_k"]] == [1]


def test_compiled_resource_and_backend_topology_change_the_sealed_b_plan() -> None:
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    total = 4 * 1024**3
    owners = {
        temporal_k: _owner_components(_contract(B=1, temporal_k=temporal_k))
        for temporal_k in shape.temporal_k_candidates
    }

    def envelope(
        active_blocks: int, registers: int = 32, authority_sha256: str = "f" * 64
    ):
        topology = {
            temporal_k: _topology_binding(
                shape,
                BACKEND_FUSED_CTA_TEMPORAL_V2,
                temporal_k,
                active_blocks_per_sm=active_blocks,
                registers_per_thread=registers,
            )
            for temporal_k in shape.temporal_k_candidates
        }
        return workhorse_candidate_envelope_v2(
            shape,
            backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
            free_bytes=total,
            total_device_bytes=total,
            owner_components_by_k=owners,
            topology_resource_bindings_by_k=topology,
            live_observation_authority_binding_sha256=authority_sha256,
        )

    four_blocks = envelope(4)
    two_blocks = envelope(2)
    changed_registers = envelope(4, registers=64)
    changed_authority = envelope(4, authority_sha256="e" * 64)
    assert (
        four_blocks["candidate_envelope_sha256"]
        != two_blocks["candidate_envelope_sha256"]
    )
    assert (
        four_blocks["per_k"][0]["B_candidates"]
        != two_blocks["per_k"][0]["B_candidates"]
    )
    assert (
        four_blocks["candidate_envelope_sha256"]
        != changed_registers["candidate_envelope_sha256"]
    )
    assert (
        four_blocks["candidate_envelope_sha256"]
        != changed_authority["candidate_envelope_sha256"]
    )
    for row in four_blocks["per_k"]:
        assert (
            row["exact_launch_admissible_maximum_B"] == row["maximum_vram_admitted_B"]
        )
        assert row["exact_launch_admissible_maximum_B"] in row["B_candidates"]
        assert (
            row["vram_maximum_adjacent_B"][-1]
            <= row["topology_resource_binding"]["maximum_launch_B"]
        )

    graph_owners = {
        temporal_k: _owner_components(
            _contract(
                B=1,
                temporal_k=temporal_k,
                backend=BACKEND_PHASE_GRAPH_V2,
            ),
            graph_object_bytes=64 * 1024,
            graph_exec_bytes=128 * 1024,
        )
        for temporal_k in shape.temporal_k_candidates
    }
    graph_envelope = workhorse_candidate_envelope_v2(
        shape,
        backend_id=BACKEND_PHASE_GRAPH_V2,
        free_bytes=total,
        total_device_bytes=total,
        owner_components_by_k=graph_owners,
        topology_resource_bindings_by_k={
            temporal_k: _topology_binding(shape, BACKEND_PHASE_GRAPH_V2, temporal_k)
            for temporal_k in shape.temporal_k_candidates
        },
        live_observation_authority_binding_sha256="f" * 64,
    )
    assert (
        graph_envelope["per_k"][0]["B_candidates"]
        != four_blocks["per_k"][0]["B_candidates"]
    )


def test_topology_envelope_rejects_missing_occupancy_and_graph_owner_mismatch() -> None:
    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    runtime = _runtime_resource_fixture(shape, BACKEND_FUSED_CTA_TEMPORAL_V2)
    runtime["compiled_kernel_resources"][0][
        "active_blocks_per_sm_at_compiled_block"
    ] = 0
    with pytest.raises(FullEvolutionWorkhorseContractError, match="blocks-per-SM"):
        workhorse_topology_resource_binding_v2(
            shape,
            backend_id=BACKEND_FUSED_CTA_TEMPORAL_V2,
            temporal_k=1,
            runtime_inspection=runtime,
        )

    topology = _topology_binding(shape, BACKEND_PHASE_GRAPH_V2, 1)
    contract = _contract(B=1, backend=BACKEND_PHASE_GRAPH_V2)
    with pytest.raises(FullEvolutionWorkhorseContractError, match="graph owner bytes"):
        workhorse_b_candidates_v2(
            shape,
            backend_id=BACKEND_PHASE_GRAPH_V2,
            temporal_k=1,
            free_bytes=4 * 1024**3,
            total_device_bytes=4 * 1024**3,
            owner_components=_owner_components(contract),
            topology_resource_binding=topology,
        )


@pytest.mark.parametrize("shape", SUPPORTED_WORKHORSE_SHAPES_V2)
@pytest.mark.parametrize(
    "backend_id",
    (BACKEND_PHASE_GRAPH_V2, BACKEND_FUSED_CTA_TEMPORAL_V2),
)
def test_every_shape_mode_has_exact_live_per_k_maximum(shape, backend_id: str) -> None:
    total = 24 * 1024**3
    free = 23 * 1024**3
    owners = {}
    for temporal_k in shape.temporal_k_candidates:
        contract = FullEvolutionCudaWorkhorseContractV2(
            shape_id=shape.shape_id,
            backend_id=backend_id,
            B=1,
            temporal_k=temporal_k,
            total_ticks=temporal_k,
        )
        owners[temporal_k] = _owner_components(
            contract,
            graph_object_bytes=(
                64 * 1024 if backend_id == BACKEND_PHASE_GRAPH_V2 else 0
            ),
            graph_exec_bytes=(
                128 * 1024 if backend_id == BACKEND_PHASE_GRAPH_V2 else 0
            ),
        )
    envelope = workhorse_candidate_envelope_v2(
        shape,
        backend_id=backend_id,
        free_bytes=free,
        total_device_bytes=total,
        owner_components_by_k=owners,
        topology_resource_bindings_by_k={
            temporal_k: _topology_binding(shape, backend_id, temporal_k)
            for temporal_k in shape.temporal_k_candidates
        },
        live_observation_authority_binding_sha256="f" * 64,
    )
    maxima = []
    for k_row in envelope["per_k"]:
        temporal_k = k_row["temporal_k"]
        owner = owners[temporal_k]
        contract = FullEvolutionCudaWorkhorseContractV2(
            shape_id=shape.shape_id,
            backend_id=backend_id,
            B=1,
            temporal_k=temporal_k,
            total_ticks=temporal_k,
        )
        one = estimate_workhorse_resources_v2(contract, owner_components=owner)
        fixed = (
            owner.graph_object_bytes
            + owner.graph_exec_bytes
            + owner.allocator_owner_overhead_bytes
            + owner.driver_unattributed_owner_bytes
        )
        expected_maximum = min(
            65_535,
            (free - total // 4 - fixed) // (one.incremental_device_bytes - fixed),
        )
        assert k_row["maximum_vram_admitted_B"] == expected_maximum
        assert expected_maximum in k_row["B_candidates"]
        if expected_maximum < 65_535:
            assert expected_maximum + 1 in k_row["B_candidates"]
            plus_one = next(
                row for row in k_row["rows"] if row["B"] == expected_maximum + 1
            )
            assert plus_one["admission"]["admitted"] is False
        maxima.append(expected_maximum)
    assert maxima == sorted(maxima, reverse=True)


def test_artifact_identity_binds_exact_shape_backend_and_layout() -> None:
    identities = {
        (shape.shape_id, backend.backend_id): workhorse_artifact_identity_v2(
            shape, backend.backend_id
        )
        for shape in SUPPORTED_WORKHORSE_SHAPES_V2
        for backend in WORKHORSE_BACKENDS_V2
    }
    assert len({value["identity_sha256"] for value in identities.values()}) == 15
    for (shape_id, backend_id), value in identities.items():
        assert value["shape_id"] == shape_id
        assert value["backend_id"] == backend_id
        assert value["profile_id"].endswith(shape_id)
        assert value["route_id"].endswith(shape_id)


def test_checkpoint_binds_reopened_external_sink_ack_and_absolute_frontier(
    tmp_path: Path,
) -> None:
    contract = _contract(total_ticks=17)
    rows = bytes(contract.B * 1 * 2 * contract.shape.capacity * 48)
    counts = bytes(contract.B * 4)
    authority = _attempt_binding(contract, artifact_sha256="1" * 64)
    receipt = publish_workhorse_event_sink_receipt_v2(
        tmp_path / "sink-a",
        sink_id="test-external-sink",
        event_rows=rows,
        event_counts=counts,
        event_rows_shape=(contract.B, 1, 2 * contract.shape.capacity, 48),
        event_counts_shape=(contract.B,),
        world_uids=tuple(range(100, 100 + contract.B)),
        start_next_absolute_tick=18,
        end_next_absolute_tick=19,
        artifact_identity_sha256="a" * 64,
        artifact_sha256="1" * 64,
        manifest_sha256="2" * 64,
        attempt_authority_binding=authority,
        rng_keys=tuple(range(200, 200 + contract.B)),
        pre_state_sha256="c" * 64,
        post_state_sha256="b" * 64,
    )
    identity = workhorse_checkpoint_identity_v2(
        contract,
        checkpoint_at_tick=19,
        artifact_identity_sha256="a" * 64,
        state_content_sha256="b" * 64,
        event_sink_receipt_path=str(receipt),
        attempt_authority_binding=authority,
    )
    assert identity["checkpoint_at_tick"] == 19
    assert identity["next_absolute_tick"] == 19
    assert "generation" not in str(identity).lower()
    assert (
        identity["event_sink_ack_sha256"]
        == validate_workhorse_event_sink_receipt_v2(receipt)["receipt_file_sha256"]
    )
    with pytest.raises((TypeError, FullEvolutionWorkhorseAuthorityError)):
        workhorse_checkpoint_identity_v2(
            contract,
            checkpoint_at_tick=19,
            artifact_identity_sha256="a" * 64,
            state_content_sha256="b" * 64,
            event_sink_receipt_path="c" * 64,
            attempt_authority_binding=authority,
        )


def test_encoded_event_sink_reopens_exact_content_and_rejects_stored_tamper(
    tmp_path: Path,
) -> None:
    contract = _contract(B=1, total_ticks=2)
    authority = _attempt_binding(contract, artifact_sha256="b" * 64)
    event_rows = bytes(2 * contract.shape.capacity * 48)
    event_counts = bytes(4)
    receipt = publish_workhorse_event_sink_receipt_v2(
        tmp_path / "encoded",
        sink_id="encoded-exact-content-test-v2",
        event_rows=event_rows,
        event_counts=event_counts,
        event_rows_shape=(1, 1, 2 * contract.shape.capacity, 48),
        event_counts_shape=(1,),
        world_uids=(7,),
        start_next_absolute_tick=11,
        end_next_absolute_tick=12,
        artifact_identity_sha256="a" * 64,
        artifact_sha256="b" * 64,
        manifest_sha256="c" * 64,
        attempt_authority_binding=authority,
        rng_keys=(17,),
        pre_state_sha256="d" * 64,
        post_state_sha256="e" * 64,
        compress_payloads=True,
    )
    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["sink_kind"] == "immutable_encoded_file_exact_content_v2"
    assert value["event_rows"]["storage_encoding"] == (
        "zstd_level_1_v1" if _test_zstd is not None else "zlib_level_1_v1"
    )
    assert value["event_rows"]["stored_bytes"] < value["event_rows"]["bytes"]
    reopened = validate_workhorse_event_sink_receipt_v2(receipt)
    assert reopened["event_rows_file_sha256"] == hashlib.sha256(event_rows).hexdigest()
    assert reopened["event_counts_file_sha256"] == hashlib.sha256(
        event_counts
    ).hexdigest()

    rows_path = receipt.parent / value["event_rows"]["file_name"]
    rows_path.chmod(0o644)
    stored = bytearray(rows_path.read_bytes())
    stored[-1] ^= 1
    rows_path.write_bytes(stored)
    rows_path.chmod(0o444)
    with pytest.raises(
        FullEvolutionWorkhorseAuthorityError, match="stored-content"
    ):
        validate_workhorse_event_sink_receipt_v2(receipt)


def test_external_sink_rejects_forged_ack_and_tampered_exact_content(
    tmp_path: Path,
) -> None:
    forged = tmp_path / "forged.json"
    forged.write_text(
        json.dumps(
            {
                "schema_version": (
                    "eiel_full_evolution_cuda_workhorse_event_sink_receipt_v2"
                ),
                "receipt_body_sha256": "f" * 64,
                "dispatch_content_sha256": "e" * 64,
            }
        ),
        encoding="utf-8",
    )
    forged.chmod(0o444)
    with pytest.raises(FullEvolutionWorkhorseAuthorityError):
        validate_workhorse_event_sink_receipt_v2(forged)

    contract = _contract(B=1, total_ticks=2)
    authority = _attempt_binding(contract, artifact_sha256="b" * 64)
    rows_per_tick = 2 * contract.shape.capacity
    receipt = publish_workhorse_event_sink_receipt_v2(
        tmp_path / "real",
        sink_id="test-external-sink",
        event_rows=bytes(rows_per_tick * 48),
        event_counts=bytes(4),
        event_rows_shape=(1, 1, rows_per_tick, 48),
        event_counts_shape=(1,),
        world_uids=(7,),
        start_next_absolute_tick=11,
        end_next_absolute_tick=12,
        artifact_identity_sha256="a" * 64,
        artifact_sha256="b" * 64,
        manifest_sha256="c" * 64,
        attempt_authority_binding=authority,
        rng_keys=(17,),
        pre_state_sha256="d" * 64,
        post_state_sha256="e" * 64,
    )
    rows_path = receipt.parent / "event_rows.bin"
    rows_path.chmod(0o644)
    rows_path.write_bytes(b"x" + rows_path.read_bytes()[1:])
    rows_path.chmod(0o444)
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="exact-content"):
        validate_workhorse_event_sink_receipt_v2(receipt)


def _sparse_event_row(
    *,
    tick: int,
    kind: int,
    slot: int,
    site: int,
    agent: int,
    parent_a: int = 0,
    parent_b: int = 0,
) -> bytes:
    return _EVENT.pack(
        tick,
        (kind << 48) | slot,
        agent,
        parent_a,
        parent_b,
        site,
        kind,
        1,
        0,
    )


def test_sparse_event_sink_reopens_recovered_segments_and_rejects_resealed_lies(
    tmp_path: Path,
) -> None:
    contract = _contract(B=2, temporal_k=4, total_ticks=4)
    authority = _attempt_binding(contract, artifact_sha256="b" * 64)
    artifact_identity = workhorse_artifact_identity_v2(
        contract.shape, contract.backend_id
    )["identity_sha256"]
    rows = [
        [
            _sparse_event_row(tick=11, kind=1, slot=3, site=7, agent=101),
            _sparse_event_row(
                tick=12,
                kind=2,
                slot=4,
                site=8,
                agent=102,
                parent_a=91,
                parent_b=92,
            ),
            _sparse_event_row(tick=14, kind=1, slot=5, site=9, agent=103),
        ],
        [
            _sparse_event_row(
                tick=13,
                kind=2,
                slot=2,
                site=10,
                agent=201,
                parent_a=191,
                parent_b=192,
            )
        ],
    ]
    segments = [
        {
            "segment_index": 0,
            "start_next_absolute_tick": 11,
            "end_next_absolute_tick": 13,
            "tick_count": 2,
            "event_counts": [2, 0],
            "packed_event_count": 2,
        },
        {
            "segment_index": 1,
            "start_next_absolute_tick": 13,
            "end_next_absolute_tick": 15,
            "tick_count": 2,
            "event_counts": [1, 1],
            "packed_event_count": 2,
        },
    ]
    receipt = publish_workhorse_sparse_event_sink_receipt_v3(
        tmp_path / "sparse-valid",
        sink_id="sparse-recovered-v3",
        packed_events=b"".join(row for world in rows for row in world),
        event_counts=(3, 1),
        physical_segments=segments,
        overflow_recovery_count=1,
        world_uids=(7001, 7002),
        start_next_absolute_tick=11,
        end_next_absolute_tick=15,
        artifact_identity_sha256=artifact_identity,
        artifact_sha256="b" * 64,
        manifest_sha256="c" * 64,
        attempt_authority_binding=authority,
        rng_keys=(17, 18),
        pre_state_sha256="d" * 64,
        post_state_sha256="e" * 64,
    )
    reopened = validate_workhorse_event_sink_receipt_v2(receipt)
    assert reopened["event_counts"] == [3, 1]
    assert reopened["overflow_recovery_count"] == 1
    assert reopened["benchmark_selectable"] is False
    assert reopened["physical_segments"] == segments

    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["dispatch"]["physical_segments"][0]["event_counts"] = [1, 1]
    value["dispatch"]["physical_segments"][1]["event_counts"] = [2, 0]
    packed_events = b"".join(row for world in rows for row in world)
    value["dispatch_content_sha256"] = (
        workhorse_sparse_event_dispatch_content_sha256_v3(
            metadata=value["dispatch"], packed_events=packed_events
        )
    )
    body = {key: item for key, item in value.items() if key != "receipt_body_sha256"}
    value["receipt_body_sha256"] = canonical_sha256(body)
    forged_dir = tmp_path / "sparse-forged" / value["dispatch_content_sha256"]
    forged_dir.mkdir(parents=True)
    (forged_dir / "packed_events.bin").write_bytes(packed_events)
    forged_receipt = forged_dir / "sink_receipt.json"
    forged_receipt.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    for path in forged_dir.iterdir():
        path.chmod(0o444)
    forged_dir.chmod(0o555)
    with pytest.raises(
        FullEvolutionWorkhorseAuthorityError,
        match="segment counts differ from packed event ticks",
    ):
        validate_workhorse_event_sink_receipt_v2(forged_receipt)


def test_sparse_event_sink_rejects_order_identity_and_split_count_substitution(
    tmp_path: Path,
) -> None:
    contract = _contract(B=1, temporal_k=4, total_ticks=4)
    authority = _attempt_binding(contract, artifact_sha256="b" * 64)
    artifact_identity = workhorse_artifact_identity_v2(
        contract.shape, contract.backend_id
    )["identity_sha256"]
    first = _sparse_event_row(tick=11, kind=1, slot=3, site=7, agent=101)
    second = _sparse_event_row(tick=12, kind=1, slot=4, site=8, agent=102)
    common = {
        "sink_id": "sparse-hostile-v3",
        "event_counts": (2,),
        "physical_segments": (
            {
                "segment_index": 0,
                "start_next_absolute_tick": 11,
                "end_next_absolute_tick": 15,
                "tick_count": 4,
                "event_counts": [2],
                "packed_event_count": 2,
            },
        ),
        "overflow_recovery_count": 0,
        "world_uids": (7001,),
        "start_next_absolute_tick": 11,
        "end_next_absolute_tick": 15,
        "artifact_identity_sha256": artifact_identity,
        "artifact_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "attempt_authority_binding": authority,
        "rng_keys": (17,),
        "pre_state_sha256": "d" * 64,
        "post_state_sha256": "e" * 64,
    }
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="canonical order"):
        publish_workhorse_sparse_event_sink_receipt_v3(
            tmp_path / "bad-order", packed_events=second + first, **common
        )
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="exact started Attempt"):
        publish_workhorse_sparse_event_sink_receipt_v3(
            tmp_path / "bad-identity",
            packed_events=first + second,
            **{**common, "artifact_identity_sha256": "f" * 64},
        )
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="split leaves"):
        publish_workhorse_sparse_event_sink_receipt_v3(
            tmp_path / "bad-split-count",
            packed_events=first + second,
            **{**common, "overflow_recovery_count": 1},
        )


def test_compact_sparse_event_consumer_validates_before_ack_without_raw_payload(
    tmp_path: Path,
) -> None:
    contract = _contract(B=2, temporal_k=4, total_ticks=4)
    authority = _attempt_binding(contract, artifact_sha256="b" * 64)
    artifact_identity = workhorse_artifact_identity_v2(
        contract.shape, contract.backend_id
    )["identity_sha256"]
    rows = (
        _sparse_event_row(tick=11, kind=1, slot=3, site=7, agent=101),
        _sparse_event_row(
            tick=12,
            kind=2,
            slot=4,
            site=8,
            agent=102,
            parent_a=91,
            parent_b=92,
        ),
    )
    receipt = publish_workhorse_sparse_event_sink_receipt_v3(
        tmp_path / "compact-consumer",
        sink_id="technical-digest-consumer-v1",
        packed_events=b"".join(rows),
        event_counts=(1, 1),
        physical_segments=(
            {
                "segment_index": 0,
                "start_next_absolute_tick": 11,
                "end_next_absolute_tick": 15,
                "tick_count": 4,
                "event_counts": [1, 1],
                "packed_event_count": 2,
            },
        ),
        overflow_recovery_count=0,
        world_uids=(7001, 7002),
        start_next_absolute_tick=11,
        end_next_absolute_tick=15,
        artifact_identity_sha256=artifact_identity,
        artifact_sha256="b" * 64,
        manifest_sha256="c" * 64,
        attempt_authority_binding=authority,
        rng_keys=(17, 18),
        pre_state_sha256="d" * 64,
        post_state_sha256="e" * 64,
        retain_packed_events=False,
    )
    assert {path.name for path in receipt.parent.iterdir()} == {"sink_receipt.json"}
    reopened = validate_workhorse_event_sink_receipt_v2(receipt)
    assert reopened["schema_version"] == (
        WORKHORSE_SPARSE_EVENT_CONSUMER_RECEIPT_SCHEMA_V4
    )
    assert reopened["event_counts"] == [1, 1]
    assert reopened["raw_payload_retained"] is False
    assert reopened["benchmark_selectable"] is True

    receipt.chmod(0o644)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["consumed_packed_events"]["raw_payload_retained"] = True
    body = {key: item for key, item in value.items() if key != "receipt_body_sha256"}
    value["receipt_body_sha256"] = canonical_sha256(body)
    receipt.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    receipt.chmod(0o444)
    with pytest.raises(
        FullEvolutionWorkhorseAuthorityError,
        match="compact event consumer seal",
    ):
        validate_workhorse_event_sink_receipt_v2(receipt)


def test_canonical_rows_use_compact_event_consumers_for_all_dispatches() -> None:
    source = inspect.getsource(run_workhorse_canonical_admitted_row_v2)
    assert source.count("retain_sink_payload=False") == 2
    assert "_initialize_owner_exact" in source
    assert "world_identity_binding=world_identity" in source
    assert "_initialize_owner(" not in source


@pytest.mark.parametrize("failure_stage", ("drain", "publish"))
def test_sparse_ack_failure_restores_prestate_and_poisons_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse import probe as probe_module

    restored: list[bytes] = []
    owner = SimpleNamespace(
        pending={
            "pre_state_bytes": b"exact-prestate",
            "pre_state_sha256": "d" * 64,
            "attempt_authority_binding_sha256": "f" * 64,
        },
        poisoned_status=(),
        loaded=SimpleNamespace(
            artifact=tmp_path / "unused.so",
            manifest={
                "identity": {"identity_sha256": "a" * 64},
                "manifest_sha256": "c" * 64,
            },
        ),
        world_uids=(7001,),
        rng_keys=(17,),
        attempt_authority_binding={"placeholder": True},
    )
    monkeypatch.setattr(
        probe_module,
        "_restore_owner_state_bytes_v3",
        lambda _owner, _torch, state: restored.append(state),
    )
    if failure_stage == "drain":
        monkeypatch.setattr(
            probe_module,
            "_prepare_sparse_ack_payload_v3",
            lambda *_args: (_ for _ in ()).throw(RuntimeError("drain failed")),
        )
    else:
        monkeypatch.setattr(
            probe_module,
            "_prepare_sparse_ack_payload_v3",
            lambda *_args: {
                "start_tick": 11,
                "requested_end_tick": 12,
                "counts": (0,),
                "statuses": (0,),
                "next_ticks": (12,),
                "physical_segments": [
                    {
                        "segment_index": 0,
                        "start_next_absolute_tick": 11,
                        "end_next_absolute_tick": 12,
                        "tick_count": 1,
                        "event_counts": [0],
                        "packed_event_count": 0,
                    }
                ],
                "recovery_count": 0,
                "packed_events": b"",
                "post_state_sha256": "e" * 64,
            },
        )
        monkeypatch.setattr(probe_module, "file_sha256", lambda _path: "b" * 64)
        monkeypatch.setattr(
            probe_module,
            "publish_workhorse_sparse_event_sink_receipt_v3",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("publish failed")
            ),
        )
    with pytest.raises(RuntimeError, match=failure_stage):
        probe_module._acknowledge_events(
            owner,
            None,
            sink_root=tmp_path / "sink",
            sink_id="rollback-test-v3",
        )
    assert restored == [b"exact-prestate"]
    assert owner.poisoned_status == (-1,)
    assert owner.pending is not None


def test_event_sink_requires_exact_2c_row_dimension_on_publish_and_reopen(
    tmp_path: Path,
) -> None:
    contract = _contract(B=1, total_ticks=2)
    authority = _attempt_binding(contract, artifact_sha256="b" * 64)
    common = {
        "sink_id": "exact-capacity-sentinel-v2",
        "event_counts": bytes(4),
        "event_counts_shape": (1,),
        "world_uids": (7,),
        "start_next_absolute_tick": 11,
        "end_next_absolute_tick": 12,
        "artifact_identity_sha256": "a" * 64,
        "artifact_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "attempt_authority_binding": authority,
        "rng_keys": (17,),
        "pre_state_sha256": "d" * 64,
        "post_state_sha256": "e" * 64,
    }
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="tensor shapes"):
        publish_workhorse_event_sink_receipt_v2(
            tmp_path / "bad-publish",
            event_rows=bytes(48),
            event_rows_shape=(1, 1, 1, 48),
            **common,
        )

    rows_per_tick = 2 * contract.shape.capacity
    valid_receipt = publish_workhorse_event_sink_receipt_v2(
        tmp_path / "valid",
        event_rows=bytes(rows_per_tick * 48),
        event_rows_shape=(1, 1, rows_per_tick, 48),
        **common,
    )
    value = json.loads(valid_receipt.read_text(encoding="utf-8"))
    forged_rows = bytes(48)
    forged_counts = bytes(4)
    value["dispatch"]["event_rows_shape"] = [1, 1, 1, 48]
    value["event_rows"] = {
        "file_name": "event_rows.bin",
        "bytes": len(forged_rows),
        "sha256": hashlib.sha256(forged_rows).hexdigest(),
    }
    value["event_counts"] = {
        "file_name": "event_counts.bin",
        "bytes": len(forged_counts),
        "sha256": hashlib.sha256(forged_counts).hexdigest(),
    }
    value["dispatch_content_sha256"] = workhorse_event_dispatch_content_sha256_v2(
        metadata=value["dispatch"],
        event_rows=forged_rows,
        event_counts=forged_counts,
    )
    body = {key: item for key, item in value.items() if key != "receipt_body_sha256"}
    value["receipt_body_sha256"] = canonical_sha256(body)
    forged_dir = tmp_path / "forged-reopen" / value["dispatch_content_sha256"]
    forged_dir.mkdir(parents=True)
    (forged_dir / "event_rows.bin").write_bytes(forged_rows)
    (forged_dir / "event_counts.bin").write_bytes(forged_counts)
    forged_receipt = forged_dir / "sink_receipt.json"
    forged_receipt.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    for path in forged_dir.iterdir():
        path.chmod(0o444)
    forged_dir.chmod(0o555)
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="shape/byte"):
        validate_workhorse_event_sink_receipt_v2(forged_receipt)


def test_event_sink_and_checkpoint_reject_cross_attempt_or_rng_substitution(
    tmp_path: Path,
) -> None:
    contract = _contract(B=1, total_ticks=2)
    request = workhorse_attempt_authority_request_v2(
        phase="EXECUTION_ROW",
        contract=contract,
        artifact_sha256="b" * 64,
        sealed_artifact_identity_sha256="c" * 64,
        device_uuid="GPU-test",
        candidate_envelope_sha256="d" * 64,
        governing_contracts=workhorse_governing_contract_bindings_v2(),
    )
    owner_a = _historical_binding_from_request(request)
    owner_b = _historical_binding_from_request(
        request,
        response_overrides={"attempt_id": "ATTEMPT-WORKHORSE-V2-B"},
    )
    assert owner_a["attempt_lineage_sha256"] != owner_b["attempt_lineage_sha256"]
    receipt = publish_workhorse_event_sink_receipt_v2(
        tmp_path / "sink",
        sink_id="cross-owner-test-v2",
        event_rows=bytes(2 * contract.shape.capacity * 48),
        event_counts=bytes(4),
        event_rows_shape=(1, 1, 2 * contract.shape.capacity, 48),
        event_counts_shape=(1,),
        world_uids=(7,),
        start_next_absolute_tick=11,
        end_next_absolute_tick=12,
        artifact_identity_sha256="a" * 64,
        artifact_sha256="b" * 64,
        manifest_sha256="e" * 64,
        attempt_authority_binding=owner_a,
        rng_keys=(17,),
        pre_state_sha256="f" * 64,
        post_state_sha256="1" * 64,
    )
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="Attempt"):
        validate_workhorse_event_sink_receipt_v2(
            receipt,
            expected_attempt_authority_binding_sha256=owner_b[
                "authority_binding_sha256"
            ],
        )
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="RNG"):
        validate_workhorse_event_sink_receipt_v2(
            receipt,
            expected_rng_identity_sha256="2" * 64,
        )
    with pytest.raises(FullEvolutionWorkhorseAuthorityError, match="Attempt"):
        workhorse_checkpoint_identity_v2(
            contract,
            checkpoint_at_tick=12,
            artifact_identity_sha256="a" * 64,
            state_content_sha256="1" * 64,
            event_sink_receipt_path=str(receipt),
            attempt_authority_binding=owner_b,
        )


def test_fabricated_micro_mapping_cannot_be_used_as_denominator(tmp_path: Path) -> None:
    forged = tmp_path / "micro.json"
    body = {
        "schema_version": "eiel_cuda_event_autotune_authoritative_result_v2",
        "authoritative": True,
        "status": "FULL_TARGET_PASS",
        "authority_class": (
            "AUTHORITATIVE_FRESH_LIVE_REPLAY_WITH_NON_AUTHORITATIVE_ORIGINAL_SWEEP"
        ),
        "selected_p10_world_ticks_per_second": 1e30,
        "artifact": {"path": str(tmp_path / "forged.so")},
    }
    from engine.perf.cuda_full_evolution_v2.workhorse.contract import canonical_sha256

    forged.write_text(
        json.dumps({**body, "result_sha256": canonical_sha256(body)}),
        encoding="utf-8",
    )
    forged.chmod(0o444)
    with pytest.raises(FullEvolutionWorkhorseAuthorityError):
        validate_authoritative_micro_denominator_v2(
            forged, expected_device_identity={"uuid": "forged"}
        )


def test_all_true_self_attested_semantic_pass_is_rejected(tmp_path: Path) -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse.contract import canonical_sha256

    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    body = {
        "schema_version": "eiel_full_evolution_cuda_workhorse_semantic_gate_v2",
        "decision": "SEMANTIC_GATE_SATISFIED",
        "decision_authority": "runtime_inspected_exact_checks",
        "shape_id": shape.shape_id,
        "source_pid": os.getpid(),
        "reference": {"artifact": str(tmp_path / "forged-reference.so")},
        "candidate": {"artifact": str(tmp_path / "forged-candidate.so")},
        "runtime": {"runtime_inspected": True},
        "B": 2,
        "temporal_k": 2,
        "checks": {check: True for check in WORKHORSE_REQUIRED_SEMANTIC_CHECKS_V2},
        "witnesses": {},
        "governing_contracts": workhorse_governing_contract_bindings_v2(),
    }
    receipt = tmp_path / "forged-semantic-pass.json"
    receipt.write_text(
        json.dumps(
            {
                **body,
                "semantic_gate_receipt_sha256": canonical_sha256(body),
            }
        ),
        encoding="utf-8",
    )
    receipt.chmod(0o444)
    with pytest.raises(FullEvolutionWorkhorseAuthorityError):
        validate_workhorse_semantic_gate_receipt_v2(
            receipt,
            expected_shape_id=shape.shape_id,
            expected_candidate_artifact_sha256="a" * 64,
            expected_candidate_backend_id=BACKEND_PHASE_GRAPH_V2,
        )


def test_explicit_candidate_override_is_always_preliminary() -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse.probe import (
        classify_workhorse_tuning_plan_v2,
    )

    assert (
        classify_workhorse_tuning_plan_v2(B_candidates=None, temporal_k_candidates=None)
        == "CANONICAL_LIVE_DERIVED_COMPLETE"
    )
    assert (
        classify_workhorse_tuning_plan_v2(
            B_candidates=(544,), temporal_k_candidates=None
        )
        == "EXPLICIT_PARTIAL_PRELIMINARY"
    )
    assert (
        classify_workhorse_tuning_plan_v2(
            B_candidates=None, temporal_k_candidates=(1, 2, 4, 8, 16)
        )
        == "EXPLICIT_PARTIAL_PRELIMINARY"
    )


def test_self_attested_shape_viability_receipt_is_rejected(tmp_path: Path) -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse.contract import canonical_sha256
    from engine.perf.cuda_full_evolution_v2.workhorse.probe import (
        FullEvolutionWorkhorseProbeError,
        validate_workhorse_shape_scaling_receipt_v2,
    )

    body = {
        "schema_version": "eiel_workhorse_shape_mode_scaling_receipt_v2",
        "shape_id": SUPPORTED_WORKHORSE_SHAPES_V2[0].shape_id,
        "decision": "VIABLE",
        "performance_viability_threshold": 0.25,
        "mode_envelope_complete": True,
        "mode_receipt_files": {},
        "mode_receipts": {},
        "rows": [],
        "winner": {"normalized_site_throughput_efficiency": 1e30},
        "governing_contracts": workhorse_governing_contract_bindings_v2(),
    }
    receipt = tmp_path / "forged-shape-viability.json"
    receipt.write_text(
        json.dumps({**body, "receipt_content_sha256": canonical_sha256(body)}),
        encoding="utf-8",
    )
    receipt.chmod(0o444)
    with pytest.raises(FullEvolutionWorkhorseProbeError):
        validate_workhorse_shape_scaling_receipt_v2(receipt)


def test_native_source_is_v2_shape_specialized_and_batched() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu"
    ).read_text(encoding="utf-8")
    for required in (
        "EIEL_WORKHORSE_WIDTH",
        "EIEL_WORKHORSE_HEIGHT",
        "EIEL_WORKHORSE_INITIAL_POPULATION",
        "EIEL_WORKHORSE_CAPACITY",
        "EIEL_WORKHORSE_BACKEND_KIND",
        "blockIdx.y",
        "fused_world_temporal_v2",
        "environment_phase_v2",
        "semantic_transition_sha256",
        "backend_topology_sha256",
    ):
        assert required in source
    assert "uint16_t site_index" not in source
    assert "constexpr int kEventRowsPerTick = 2 * kCapacity;" in source
    assert "events->dim() == 3" in source
    assert "events->size(1) == kEventRowsPerTick" in source
    assert "2 * kCapacity * temporal_k * 48" not in source
    assert "if (kMaximumTemporalK >= 32) values.append(32);" in source
    assert "if (kMaximumTemporalK >= 64) values.append(64);" in source


def test_historical_prebuilt_inventory_reopen_keeps_seals_without_live_match(
    tmp_path: Path,
) -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse.native_loader import (
        FullEvolutionWorkhorseLoadError,
        _validate_source_inventory_rows_v2,
    )

    source = tmp_path / "bound-source.py"
    source.write_bytes(b"sealed source v1\n")
    inventory = [
        {
            "path": source.name,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    assert _validate_source_inventory_rows_v2(
        inventory,
        project_root=tmp_path,
        require_current_source_match=True,
    ) == inventory

    source.write_bytes(b"later source v2\n")
    with pytest.raises(FullEvolutionWorkhorseLoadError, match="source binding mismatch"):
        _validate_source_inventory_rows_v2(
            inventory,
            project_root=tmp_path,
            require_current_source_match=True,
        )
    assert _validate_source_inventory_rows_v2(
        inventory,
        project_root=tmp_path,
        require_current_source_match=False,
    ) == inventory

    with pytest.raises(FullEvolutionWorkhorseLoadError, match="unsafe"):
        _validate_source_inventory_rows_v2(
            [{"path": "../escape.py", "sha256": "0" * 64}],
            project_root=tmp_path,
            require_current_source_match=False,
        )


def test_native_scalability_patch_preserves_sparse_exact_ack_boundary() -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse.native_loader import (
        WORKHORSE_REQUIRED_EXPORTS_V2,
    )

    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu"
    ).read_text(encoding="utf-8")
    probe = (
        root / "engine/perf/cuda_full_evolution_v2/workhorse/probe.py"
    ).read_text(encoding="utf-8")
    invalidation = source[
        source.index("void invalidate_event_status(") :
        source.index("__device__ __forceinline__ void write_event(")
    ]
    environment = source[
        source.index("__global__ void environment_phase_v2(") :
        source.index("__device__ __forceinline__ void propose_move(")
    ]
    fused = source[
        source.index("__global__ void fused_world_temporal_v2(") :
        source.index("__global__ void force_causal_fixture_v2(")
    ]
    canonicalize = source[
        source.index("__global__ void canonicalize_inactive_events_v2(") :
        source.index("__global__ void initialize_worlds_v2(")
    ]
    acknowledge = probe[
        probe.index("def _acknowledge_events(") :
        probe.index("def _owner_cpu_snapshot_v2(")
    ]
    prepare_ack = probe[
        probe.index("def _prepare_sparse_ack_payload_v3(") :
        probe.index("def _acknowledge_events(")
    ]
    drain = probe[
        probe.index("def _drain_sparse_page_v3(") :
        probe.index("def _restore_owner_state_bytes_v3(")
    ]
    native_topology = source[
        source.index("py::dict kernel_topology()") :
        source.index("py::dict compiled_kernel_resource_v2(")
    ]

    assert "canonicalize_events" in WORKHORSE_REQUIRED_EXPORTS_V2
    assert "row->status = 0;" in invalidation
    assert environment.index("kBackendKind != 0 && local_tick == 0") < (
        environment.index("[kStatus] != 0) return;")
    )
    assert "static_cast<std::int64_t>(world) * kEventRowsPerTick" in environment
    assert "event_tick * kEventRowsPerTick" not in environment
    assert environment.index("kBackendKind == 0") > environment.index(
        "[kStatus] != 0) return;"
    )
    assert "clear_event(base + index);" in environment
    fused_prefix, fused_ticks = fused.split(
        "for (int local_tick = 0; local_tick < temporal_k; ++local_tick)", 1
    )
    assert fused_prefix.index("dispatch_event_rows") < fused_prefix.index(
        "[kStatus] != 0) return;"
    )
    assert "invalidate_event_status(dispatch_events + index);" in fused_prefix
    assert "invalidate_event_status" not in fused_ticks
    assert "prepare_event_for_transition" not in source
    assert "events[row_index].status == 0" in canonicalize
    assert "clear_event(events + row_index);" in canonicalize
    assert drain.index("canonicalize_events(owner.events)") < drain.index(
        "owner.events.detach().cpu()"
    )
    assert "publish_workhorse_sparse_event_sink_receipt_v3" in acknowledge
    assert "_recover_sparse_overflow_v3" in prepare_ack
    assert "_restore_owner_state_bytes_v3" in acknowledge
    assert "host_monotonic_synchronized_post_transition_tail" in probe
    preparations = {
        backend.backend_id: backend.topology_payload(SUPPORTED_WORKHORSE_SHAPES_V2[0])[
            "event_transition_preparation"
        ]
        for backend in WORKHORSE_BACKENDS_V2
    }
    assert preparations[BACKEND_REFERENCE_PHASED_V2] == (
        "reference_k1_single_page_dense_zero"
    )
    assert "single_2C_sparse_page_status_invalidation_once_inside_" in (
        preparations[BACKEND_PHASE_GRAPH_V2]
    )
    assert "single_2C_sparse_page_status_invalidation_once_inside_" in preparations[
        BACKEND_FUSED_CTA_TEMPORAL_V2
    ]
    assert native_topology.count('value["event_transition_preparation"]') == 2
    for preparation in preparations.values():
        assert f'"{preparation}"' in native_topology


def test_native_scalability_patch_aggregates_counters_and_parallelizes_fused_rank() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu"
    ).read_text(encoding="utf-8")
    physiology = source[
        source.index("interact_and_physiology(") :
        source.index("__global__ void interaction_physiology_phase_v2(")
    ]
    fused = source[
        source.index("__global__ void fused_world_temporal_v2(") :
        source.index("__global__ void force_causal_fixture_v2(")
    ]

    assert "atomic_add_u64" not in physiology
    assert "commit_physiology_cta_totals" in source
    assert "reduce_cta_u32" in source
    assert "reduce_cta_birth" in source
    assert "PhysiologyCounterDeltasV2 physiology_deltas" in fused
    assert "for (int tile = 0; tile < kAgentTiles; ++tile)" in fused
    assert "active_scan[threadIdx.x]" in fused
    assert "free_scan[threadIdx.x]" in fused
    assert "accepted_scan[threadIdx.x]" in fused
    assert "for (int lane = 0; lane < kThreads; ++lane)" not in fused
    assert "if (threadIdx.x == 0) {\n            std::uint32_t birth_total" not in fused


def test_native_birth_pairs_compacted_survivors_after_middle_death() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu"
    ).read_text(encoding="utf-8")
    enqueue = source[source.index("void enqueue_dispatch(") :]
    ordered_phases = (
        "death_phase_v2<<<",
        "rank_tiles_phase_v2<<<",
        "rank_prefix_phase_v2<<<",
        "scatter_ranks_phase_v2<<<",
        "mate_spawn_phase_v2<<<",
        "birth_commit_phase_v2<<<",
    )
    positions = [enqueue.index(phase) for phase in ordered_phases]
    assert positions == sorted(positions)
    runtime = source[source.index("py::dict runtime_inspect()") :]
    resource_positions = [
        runtime.index(f'"{phase.removesuffix("<<<")}"') for phase in ordered_phases
    ]
    assert resource_positions == sorted(resource_positions)
    assert "propose_birth_from_survivor_order" in source
    assert "survivor_order[survivor_ordinal + 1]" in source
    assert "force_middle_survivor_death_fixture_v2" in source
    assert "slot == 1 ? 0 : 256" in source
    assert "slot == 3 ? 64 : 600" in source
    for backend in WORKHORSE_BACKENDS_V2[:2]:
        topology = backend.topology_payload(SUPPORTED_WORKHORSE_SHAPES_V2[0])
        rank_start = topology["kernels_per_tick"].index("rank_tiles_phase_v2")
        assert topology["kernels_per_tick"][rank_start : rank_start + 4] == [
            "rank_tiles_phase_v2",
            "rank_prefix_phase_v2",
            "scatter_ranks_phase_v2",
            "mate_spawn_phase_v2",
        ]
        assert "mate_spawn_phase_v2" in topology["single_cta_world_phases"]


def test_native_p2_fixture_emits_death_before_real_birth_commit_failure() -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse.native_loader import (
        WORKHORSE_REQUIRED_EXPORTS_V2,
    )

    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu"
    ).read_text(encoding="utf-8")
    probe = (
        root / "engine/perf/cuda_full_evolution_v2/workhorse/probe.py"
    ).read_text(encoding="utf-8")
    fixture = source[
        source.index("void force_event_then_transaction_error_fixture(") :
        source.index("void restore_derived(")
    ]
    child = source[
        source.index("__device__ __forceinline__ BirthCounterDeltasV2 construct_child(") :
        source.index("__global__ void birth_commit_phase_v2(")
    ]

    assert "force_event_then_transaction_error_fixture" in (
        WORKHORSE_REQUIRED_EXPORTS_V2
    )
    assert "force_middle_survivor_death_fixture_v2<<<" in fixture
    assert "enable_event_then_transaction_error_fixture_v2<<<" in fixture
    assert "kFixtureBirthCommitConflictAfterDeath" in child
    assert "rank == 0" in child
    assert "atomicCAS(occupancy + spawn, kEmpty, child_slot)" in child
    assert "force_event_then_transaction_error_fixture" in probe
    assert "P2 reused-buffer law failed:" in probe


def test_build_inventory_binds_workhorse_semantics_authority_and_performance() -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse.build import (
        BOUND_SOURCE_RELATIVE_PATHS_V2,
    )

    assert BOUND_SOURCE_RELATIVE_PATHS_V2 == (
        "engine/perf/cuda_full_evolution_v2/native_loader.py",
        "engine/perf/cuda_full_evolution_v2/benchmark.py",
        "engine/perf/cuda_full_evolution_v2/science_extension.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/__init__.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/attempt_authority.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/authority.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/canonical_rows.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/contract.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/continuation.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/dispatch_plan.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/indexed_checkpoint.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/layout.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/native_loader.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/build.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/probe.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/science_demo.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/semantic_cases.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/semantic_evidence.py",
        "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu",
        "_artifacts/tw-cuda-full-evolution-reform-20260825/science_consumer_extension_contract.json",
        "_artifacts/tw-cuda-full-evolution-reform-20260825/architecture_contract.json",
        "_artifacts/tw-cuda-full-evolution-reform-20260825/performance_restructure_contract.json",
        "_artifacts/tw-cuda-full-evolution-reform-20260825/validation_matrix.json",
    )
    workhorse_root = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/workhorse"
    )
    direct_parent_imports = {
        (path.name, node.module)
        for path in workhorse_root.glob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.level == 2
    }
    assert direct_parent_imports == {
        ("authority.py", "native_loader"),
            ("canonical_rows.py", "benchmark"),
            ("continuation.py", "benchmark"),
            ("probe.py", "benchmark"),
        ("science_demo.py", "science_extension"),
    }


def test_build_rejects_bound_source_mutation_before_artifact_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse import build as build_module

    source = tmp_path / "bound-source.py"
    source.write_text("before\n", encoding="utf-8")
    monkeypatch.setattr(build_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        build_module, "BOUND_SOURCE_RELATIVE_PATHS_V2", (source.name,)
    )
    monkeypatch.setattr(
        build_module,
        "workhorse_governing_contract_bindings_v2",
        lambda _root: {"contract": {"sha256": "a" * 64}},
    )
    inventory, _, governing = build_module._bound_source_snapshot_v2()
    source.write_text("after\n", encoding="utf-8")

    with pytest.raises(
        build_module.FullEvolutionWorkhorseBuildError,
        match="bound source changed during compilation",
    ):
        build_module._require_bound_source_snapshot_unchanged_v2(
            expected_inventory=inventory,
            expected_governing_contracts=governing,
        )


def _fake_workhorse_build_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    introspection_mutation=None,
):
    from engine.perf.cuda_full_evolution_v2.workhorse import build as build_module

    project_root = tmp_path / "repo"
    project_root.mkdir()
    bound_source = project_root / "bound-source.py"
    bound_source.write_text("before\n", encoding="utf-8")
    governing_contract = project_root / "governing-contract.json"
    governing_contract.write_text('{"contract":"before"}\n', encoding="utf-8")
    compiled_artifact = tmp_path / "compiled-workhorse.so"
    compiled_artifact.write_bytes(b"fake-compiled-workhorse")
    output = tmp_path / "output"

    shape = build_module.workhorse_shape_v2(SUPPORTED_WORKHORSE_SHAPES_V2[0].shape_id)
    backend = build_module.workhorse_backend_v2(BACKEND_PHASE_GRAPH_V2)
    layout = build_module.workhorse_layout_v2(shape)
    module = SimpleNamespace(__file__=str(compiled_artifact))
    for name in build_module.WORKHORSE_REQUIRED_EXPORTS_V2:
        setattr(module, name, object())
    module.compile_shape = lambda: build_module.expected_native_shape_v2(shape.shape_id)
    module.profile_id = lambda: shape.profile_id
    module.route_id = lambda: shape.route_id
    module.backend_id = lambda: backend.backend_id
    module.layout_sha256 = lambda: layout.layout_sha256
    module.semantic_transition_sha256 = (
        lambda: build_module.WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2
    )
    module.backend_topology_sha256 = lambda: backend.topology_sha256(shape)

    def kernel_topology():
        if introspection_mutation is not None:
            introspection_mutation(bound_source, governing_contract)
        return {"schema_version": "fake_workhorse_kernel_topology_v2"}

    module.kernel_topology = kernel_topology

    fake_torch = ModuleType("torch")
    fake_torch.__path__ = []
    fake_torch.__version__ = "fake-torch"
    fake_torch.version = SimpleNamespace(cuda="fake-cuda")
    fake_torch._C = SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False)
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_capability=lambda: (7, 5),
    )
    fake_utils = ModuleType("torch.utils")
    fake_utils.__path__ = []
    fake_extension = ModuleType("torch.utils.cpp_extension")
    fake_extension.load = lambda **_kwargs: module
    fake_utils.cpp_extension = fake_extension
    fake_torch.utils = fake_utils
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", fake_extension)

    monkeypatch.setattr(build_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        build_module, "BOUND_SOURCE_RELATIVE_PATHS_V2", (bound_source.name,)
    )

    def governing_bindings(_root: Path) -> dict[str, object]:
        return {
            "governing_contract": {
                "path": governing_contract.name,
                "sha256": build_module.file_sha256(governing_contract),
            }
        }

    monkeypatch.setattr(
        build_module,
        "workhorse_governing_contract_bindings_v2",
        governing_bindings,
    )
    monkeypatch.setattr(
        build_module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="fake nvcc"),
    )
    return SimpleNamespace(
        build_module=build_module,
        bound_source=bound_source,
        governing_contract=governing_contract,
        output=output,
        shape=shape,
        backend=backend,
    )


def _run_fake_workhorse_build(environment) -> Path:
    return environment.build_module.build_workhorse_prebuilt_v2(
        environment.output,
        shape_id=environment.shape.shape_id,
        backend_id=environment.backend.backend_id,
        target_sm="sm_75",
    )


def _assert_no_public_workhorse_build(output: Path) -> None:
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_build_rechecks_bound_source_after_final_runtime_introspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate_source(bound_source: Path, _governing_contract: Path) -> None:
        bound_source.write_text("changed during introspection\n", encoding="utf-8")

    environment = _fake_workhorse_build_environment(
        tmp_path,
        monkeypatch,
        introspection_mutation=mutate_source,
    )
    with pytest.raises(
        environment.build_module.FullEvolutionWorkhorseBuildError,
        match="bound source changed during compilation",
    ):
        _run_fake_workhorse_build(environment)
    _assert_no_public_workhorse_build(environment.output)


def test_build_rechecks_governing_contract_after_final_runtime_introspection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate_governing_contract(
        _bound_source: Path, governing_contract: Path
    ) -> None:
        governing_contract.write_text(
            '{"contract":"changed during introspection"}\n',
            encoding="utf-8",
        )

    environment = _fake_workhorse_build_environment(
        tmp_path,
        monkeypatch,
        introspection_mutation=mutate_governing_contract,
    )
    with pytest.raises(
        environment.build_module.FullEvolutionWorkhorseBuildError,
        match="bound source changed during compilation",
    ):
        _run_fake_workhorse_build(environment)
    _assert_no_public_workhorse_build(environment.output)


def test_build_rechecks_bound_source_after_staged_artifact_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _fake_workhorse_build_environment(tmp_path, monkeypatch)
    real_copy = environment.build_module.shutil.copy2

    def mutating_copy(source, destination, *args, **kwargs):
        result = real_copy(source, destination, *args, **kwargs)
        environment.bound_source.write_text(
            "changed during staged copy\n", encoding="utf-8"
        )
        return result

    monkeypatch.setattr(environment.build_module.shutil, "copy2", mutating_copy)
    with pytest.raises(
        environment.build_module.FullEvolutionWorkhorseBuildError,
        match="bound source changed during compilation",
    ):
        _run_fake_workhorse_build(environment)
    _assert_no_public_workhorse_build(environment.output)


def test_build_stages_on_output_filesystem_for_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _fake_workhorse_build_environment(tmp_path, monkeypatch)
    observed: dict[str, Path] = {}
    real_temporary_directory = environment.build_module.tempfile.TemporaryDirectory

    def same_filesystem_temporary_directory(*args, **kwargs):
        observed["dir"] = Path(kwargs["dir"]).resolve()
        return real_temporary_directory(*args, **kwargs)

    monkeypatch.setattr(
        environment.build_module.tempfile,
        "TemporaryDirectory",
        same_filesystem_temporary_directory,
    )

    artifact = _run_fake_workhorse_build(environment)

    assert observed["dir"] == environment.output.parent.resolve()
    assert artifact.is_file()


def test_phase_graph_is_real_native_capture_with_runtime_inspection() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu"
    ).read_text(encoding="utf-8")
    for required in (
        "cudaStreamCreateWithFlags",
        "cudaStreamNonBlocking",
        "cudaStreamBeginCapture",
        "cudaStreamEndCapture",
        "cudaStreamDestroy",
        "cudaGraphInstantiate",
        "cudaGraphLaunch",
        "cudaGraphGetNodes",
        "cudaGraphGetEdges",
        "cudaGraphKernelNodeGetParams",
        "cudaFuncGetAttributes",
        "graph_object_device_bytes",
        "graph_exec_device_bytes",
        "cudaMemGetInfo",
        "prepare_graph before advance",
    ):
        assert required in source
    prepare = source[
        source.index("void prepare_graph(") : source.index(
            "const char* graph_kernel_name_v2"
        )
    ]
    assert prepare.index("cudaStreamSynchronize(stream)") < prepare.index(
        "cudaStreamCreateWithFlags"
    )
    assert "cudaStreamBeginCapture(\n            capture_stream" in prepare
    assert "disable_mask, capture_stream" in prepare
    graph = next(
        backend
        for backend in WORKHORSE_BACKENDS_V2
        if backend.backend_id == BACKEND_PHASE_GRAPH_V2
    ).topology_payload(SUPPORTED_WORKHORSE_SHAPES_V2[0])
    reference = next(
        backend
        for backend in WORKHORSE_BACKENDS_V2
        if backend.backend_id == BACKEND_REFERENCE_PHASED_V2
    ).topology_payload(SUPPORTED_WORKHORSE_SHAPES_V2[0])
    assert graph["capture_owner"] == "native_prepared_graph_v2"
    assert graph["graph_kernel_nodes_per_temporal_k"] == 15
    assert "capture_owner" not in reference
    assert reference["capture"] is False


def test_native_transaction_rng_event_and_uid_laws_are_not_declarations_only() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu"
    ).read_text(encoding="utf-8")
    assert "philox4x32_10" in source
    assert "mix64" not in source
    assert "set_first_status" in source and "atomicCAS" in source
    assert "reference_phased_v2 requires K=1" in source
    assert "(static_cast<std::uint64_t>(kind) << 48) | slot" in source
    assert source.count("control[2] = min(") == 1
    assert source.count("kWorldControlOffset)[2] = min(") == 1
    assert "kActiveSlotsByUidOffset" in source
    assert "cudaMemset2DAsync" not in source
    assert "advance must never erase a pending restore/owner error" in source
    assert "cudaStreamSynchronize(stream)" in source
    # Only initialization and the two explicit new-owner audit fixtures reset
    # status. Restore and every dispatch preserve a pending error.
    assert source.count("header[kStatus] = 0") == 3
    assert source.count("[kStatus] = 0") == 3


def test_phase_graph_event_compaction_stops_after_transaction_failure() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/workhorse/native/full_evolution_workhorse_v2.cu"
    ).read_text(encoding="utf-8")
    compaction = source[
        source.index("__global__ void compact_phase_events_v3(") :
        source.index("__device__ __forceinline__ void propose_birth_from_survivor_order(")
    ]

    status_guard = (
        "if (field<std::uint64_t>(row, kHeaderOffset)[kStatus] != 0) return;"
    )
    assert status_guard in compaction
    assert compaction.index(status_guard) < compaction.index(
        "append_sparse_page_events_v3<EventKind>("
    )


def test_probe_is_v2_only_and_retains_scaling_regime_evidence() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_full_evolution_v2/workhorse/probe.py"
    ).read_text(encoding="utf-8")
    assert "_v1" not in source.lower()
    for required in (
        "run_workhorse_semantic_gate_v2",
        "run_workhorse_tuning_v2",
        "allocator_peak_delta_bytes",
        "normalized_site_throughput_efficiency",
        "p10_site_ticks_per_second",
        "p10_agent_capacity_ticks_per_second",
        "compiled_kernel_resources",
        "owner_component_measurements",
        "candidate_envelope_attempt_authority",
        "attempt_admission",
        "device_free_bytes",
        "validate_workhorse_scaling_receipt_v2",
        "validate_workhorse_shape_scaling_receipt_v2",
        "complete_cartesian_rows",
        "CANONICAL_LIVE_DERIVED_COMPLETE",
        "validate_authoritative_micro_denominator_v2",
        "validate_workhorse_semantic_gate_receipt_v2",
        "event buffer reuse requires an external exact-content sink acknowledgement",
        "warmup_sink_acknowledgements",
        "measurement_samples",
        "cuda_event_elapsed_ms",
        "micro, semantic gate, and tuning runtime are not the same device",
        "PERFORMANCE_RECOVERY",
        "0.25",
        "attempt_lifecycles",
        "restore_fresh_process_continuation",
        "--worker-closeout-request",
    ):
        assert required in source
    tree = ast.parse(source)
    tuning = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_workhorse_tuning_v2"
    )
    attempt_admission = next(
        node.value
        for node in ast.walk(tuning)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "attempt_admission"
            for target in node.targets
        )
    )
    assert isinstance(attempt_admission, ast.Call)
    owner_keyword = next(
        keyword
        for keyword in attempt_admission.keywords
        if keyword.arg == "owner_components"
    )
    assert isinstance(owner_keyword.value, ast.Name)
    assert owner_keyword.value.id == "owner_components"


def test_runtime_graph_inspection_is_recomputed_from_exact_nodes() -> None:
    from engine.perf.cuda_full_evolution_v2.workhorse.probe import (
        FullEvolutionWorkhorseProbeError,
        _validate_graph_inspection_v2,
    )

    shape = SUPPORTED_WORKHORSE_SHAPES_V2[0]
    backend = next(
        item
        for item in WORKHORSE_BACKENDS_V2
        if item.backend_id == BACKEND_PHASE_GRAPH_V2
    )
    temporal_k = 2
    B = 7
    tick_sequence = backend.topology_payload(shape)["kernels_per_tick"]
    sequence = tick_sequence * temporal_k
    nodes = [
        {
            "topological_rank": 0,
            "node_type": "memset",
            "name": "event_count_memset_v2",
        }
    ]
    for rank, kernel in enumerate(sequence, start=1):
        grid_x = (
            1
            if kernel
            in {
                "compact_birth_events_v3",
                "compact_death_events_v3",
                "rank_prefix_phase_v2",
                "mate_spawn_phase_v2",
                "publish_phase_v2",
            }
            else (
                shape.site_tiles
                if kernel == "environment_phase_v2"
                else shape.agent_tiles
            )
        )
        nodes.append(
            {
                "topological_rank": rank,
                "node_type": "kernel",
                "kernel": kernel,
                "grid": [grid_x, B, 1],
                "block": [256, 1, 1],
                "dynamic_shared_bytes": 0,
                "registers_per_thread": 32,
                "static_shared_bytes": 0,
                "local_bytes_per_thread": 0,
                "maximum_threads_per_block": 1024,
            }
        )
    inspection = {
        "schema_version": "eiel_full_evolution_cuda_graph_inspection_v2",
        "runtime_inspected": True,
        "shape_id": shape.shape_id,
        "backend_id": BACKEND_PHASE_GRAPH_V2,
        "B": B,
        "temporal_k": temporal_k,
        "disable_mask": 0,
        "graph_object_device_bytes": 0,
        "graph_exec_device_bytes": 0,
        "node_count": len(sequence) + 1,
        "edge_count": len(sequence),
        "kernel_node_count": len(sequence),
        "memset_node_count": 1,
        "linear_chain": True,
        "kernel_sequence": sequence,
        "expected_kernel_sequence": sequence,
        "kernel_sequence_exact": True,
        "kernel_geometry_exact": True,
        "nodes": nodes,
    }
    _validate_graph_inspection_v2(
        inspection,
        shape=shape,
        backend_id=BACKEND_PHASE_GRAPH_V2,
        B=B,
        temporal_k=temporal_k,
        disable_mask=0,
    )
    nodes[1]["grid"][1] = B + 1
    with pytest.raises(FullEvolutionWorkhorseProbeError, match="geometry"):
        _validate_graph_inspection_v2(
            inspection,
            shape=shape,
            backend_id=BACKEND_PHASE_GRAPH_V2,
            B=B,
            temporal_k=temporal_k,
            disable_mask=0,
        )
