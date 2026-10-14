from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from engine.perf.cuda_full_evolution_v2.native_loader import (
    BOUND_SOURCE_RELATIVE_PATHS_V2,
    NORMATIVE_EXECUTION_CONTRACT_PATHS_V2,
    PROJECT_ROOT,
)
from engine.perf.cuda_full_evolution_v2.contract import (
    CudaFullEvolutionNativeContractV2,
)
from engine.perf.cuda_full_evolution_v2.layout import state_layout_v2
from engine.perf.cuda_full_evolution_v2.runtime import (
    CudaFullEvolutionCheckpointV2,
    _canonical_sha256,
    _canonical_state_layout_sha256,
    _checkpoint_metadata_body,
)
from engine.perf.cuda_full_evolution_v2.science_extension import (
    SCIENCE_CHECKPOINT_EXTENSION_SCHEMA_V1,
    SCIENCE_CONTRACT_RELATIVE_PATH_V1,
    SCIENCE_EXTENSION_CONTRACT_SCHEMA_V1,
    SCIENCE_EXTENSION_STATUS_V1,
    SCIENCE_OBSERVATION_OFF_V1,
    SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_V1,
    SCIENCE_OBSERVATION_RECORD_SCHEMA_V1,
    SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_V1,
    SCIENCE_OBSERVATION_TYPED_SIDECAR_V1,
    SCIENCE_ROUTE_BINDING_SCHEMA_V1,
    SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1,
    SCIENCE_STATIC_CONTROL_TABLE_V1,
    SCIENCE_TYPED_RECORD_KINDS_V1,
    CudaFullEvolutionScienceExtensionError,
    ScienceTypedCaptureBufferV1,
    admit_science_checkpoint_restore_v1,
    admit_science_extension_request_v1,
    empty_science_observation_prefix_sha256_v1,
    extend_science_observation_prefix_sha256_v1,
    observation_off_projection_v1,
    project_base_checkpoint_identity_v1,
    science_observation_cursor_sha256_v1,
    science_semantic_record_sha256_v1,
    seal_empty_science_observation_cursor_v1,
    seal_science_checkpoint_extension_v1,
    seal_science_route_binding_adapter_v1,
    seal_science_world_mapping_v1,
    seal_static_science_control_table_v1,
    validate_science_checkpoint_extension_v1,
    validate_science_observation_prefix_cursor_v1,
    validate_science_observation_record_projection_v1,
    validate_science_route_binding_v1,
    validate_static_science_control_table_v1,
    validate_typed_science_capture_inventory_v1,
)


def _sha(character: str) -> str:
    return character * 64


def _digest(character: str) -> dict[str, str]:
    return {"algorithm": "sha256", "hex": _sha(character)}


def _canonical_digest(value: object) -> dict[str, str]:
    return {"algorithm": "sha256", "hex": _canonical_sha256(value)}


def _reseal_checkpoint_extension(envelope: dict[str, object]) -> None:
    body = {
        key: value
        for key, value in envelope.items()
        if key not in {"extension_sha256", "science_checkpoint_identity_sha256"}
    }
    envelope["extension_sha256"] = _canonical_digest(body)
    envelope["science_checkpoint_identity_sha256"] = _canonical_digest(
        {
            "schema_version": "eiel_cuda_full_evolution_science_checkpoint_identity_v1",
            "base_checkpoint_identity_sha256": envelope[
                "base_checkpoint_identity_sha256"
            ],
            "science_extension_sha256": envelope["extension_sha256"],
        }
    )


def _reseal_capture_inventory(inventory: dict[str, object]) -> None:
    body = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    inventory["inventory_sha256"] = _canonical_digest(body)


def _route_binding(instrument: dict[str, str]) -> dict[str, object]:
    fields = (
        "route_profile_sha256",
        "capability_projection_sha256",
        "native_build_closure_sha256",
        "execution_adapter_sha256",
        "control_adapter_sha256",
        "observation_adapter_sha256",
        "checkpoint_adapter_sha256",
        "semantic_conformance_sha256",
        "resource_envelope_sha256",
        "qualified_envelope_sha256",
        "qualification_authority_sha256",
    )
    return {
        "schema_version": SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1,
        "instrument_manifest_sha256": instrument,
        "route_id": "cuda-full-evolution-v2-host-adapter",
        "machine_profile_id": "EIEL_PRODUCTION_BASELINE_V1",
        **{
            field: _digest(character)
            for field, character in zip(fields, "123456789ab", strict=True)
        },
    }


def _empty_cursor(*, cuda_world_uid: int, logical_world_id: int) -> dict[str, object]:
    cursor = seal_empty_science_observation_cursor_v1(
        instrument_manifest_sha256=_digest("a"),
        crca_semantic_run_digest=_digest("b"),
        science_semantic_acquisition_sha256=_digest("c"),
        cell_id=f"cell-{logical_world_id:03d}",
        logical_world_id=logical_world_id,
        world_uid=_digest("d" if logical_world_id == 0 else "e"),
    )
    return {
        "cuda_world_uid": cuda_world_uid,
        "cursor": cursor,
        "cursor_sha256": science_observation_cursor_sha256_v1(cursor),
    }


def _root_cursors() -> list[dict[str, object]]:
    return [
        _empty_cursor(cuda_world_uid=101, logical_world_id=0),
        _empty_cursor(cuda_world_uid=102, logical_world_id=1),
    ]


def _mapping(cursors: list[dict[str, object]]) -> dict[str, object]:
    rows = []
    for projection in reversed(cursors):
        cursor = projection["cursor"]
        assert isinstance(cursor, dict)
        logical_world_id = cursor["logical_world_id"]
        rows.append(
            {
                "cuda_world_uid": projection["cuda_world_uid"],
                "science_world_uid": cursor["world_uid"],
                "semantic_world_id": f"study-a:test-a:arm-a:r1:w{logical_world_id}",
                "cell_id": cursor["cell_id"],
                "logical_world_id": logical_world_id,
                "study_id": "study-a",
                "test_id": "test-a",
                "version_id": "v1",
                "arm_id": "ARM-A",
                "replicate_id": "r1",
                "reducer_id": "reducer-v1",
            }
        )
    return seal_science_world_mapping_v1(rows)


def _controls(*, terminal_tick: int = 11) -> dict[str, object]:
    rows = []
    for cuda_uid, character in ((102, "2"), (101, "1")):
        intervals = []
        if terminal_tick:
            intervals = [
                {
                    "start_absolute_tick": 0,
                    "end_absolute_tick": terminal_tick,
                    "control_payload_sha256": _digest(character),
                }
            ]
        rows.append({"cuda_world_uid": cuda_uid, "intervals": intervals})
    return seal_static_science_control_table_v1(
        rows, terminal_next_absolute_tick=terminal_tick
    )


def _base_checkpoint(*, next_tick: int) -> CudaFullEvolutionCheckpointV2:
    import hashlib

    contract = CudaFullEvolutionNativeContractV2(
        B=2, total_ticks=11, start_next_tick=0, temporal_k=1
    )
    bytes_per_world = state_layout_v2().canonical_checkpoint_bytes_per_world
    canonical_state = bytes(2 * bytes_per_world)
    state_sha256 = hashlib.sha256(canonical_state).hexdigest()
    shape = (2, bytes_per_world)
    provisional = CudaFullEvolutionCheckpointV2(
        contract=contract,
        canonical_state=canonical_state,
        canonical_state_shape=shape,
        canonical_state_nbytes=len(canonical_state),
        canonical_state_sha256=state_sha256,
        canonical_state_layout_sha256=_canonical_state_layout_sha256(
            shape=shape,
            nbytes=len(canonical_state),
            content_sha256=state_sha256,
        ),
        initial_next_absolute_tick=0,
        advanced_ticks=next_tick,
        acknowledged_event_next_tick=next_tick,
        last_sink_receipt_sha256=None if next_tick == 0 else _sha("6"),
        world_uids=(101, 102),
        rng_keys=(11, 12),
        rng_semantic_epoch=3,
        artifact_path=Path("/nonexistent/host-test-artifact.so"),
        artifact_sha256=_sha("7"),
        manifest_sha256=_sha("8"),
        device_identity={
            "index": 0,
            "name": "host-test-device",
            "uuid": "host-test-uuid",
            "pci_domain_id": 0,
            "pci_bus_id": 1,
            "pci_device_id": 2,
            "total_memory_bytes": 1,
            "multiprocessor_count": 1,
            "sm": "sm75",
            "cuda_runtime": "host-test",
        },
        population_fixture_id=None,
        metadata_seal_sha256="",
    )
    return replace(
        provisional,
        metadata_seal_sha256=_canonical_sha256(_checkpoint_metadata_body(provisional)),
    )


def _record(
    cursor_projection: dict[str, object],
    *,
    absolute_tick: int,
    sequence: int,
    identity_character: str,
) -> dict[str, object]:
    cursor = cursor_projection["cursor"]
    assert isinstance(cursor, dict)
    return {
        "schema_version": SCIENCE_OBSERVATION_RECORD_SCHEMA_V1,
        "instrument_manifest_sha256": cursor["instrument_manifest_sha256"],
        "producer_component_id": "cuda-full-evolution-v2-host-adapter",
        "producer_contract_version": "cuda_full_evolution_science_adapter_v1",
        "semantic_execution_id": cursor["semantic_execution_id"],
        "crca_semantic_run_digest": cursor["crca_semantic_run_digest"],
        "science_semantic_acquisition_sha256": cursor[
            "science_semantic_acquisition_sha256"
        ],
        "cell_id": cursor["cell_id"],
        "logical_world_id": cursor["logical_world_id"],
        "world_uid": cursor["world_uid"],
        "absolute_tick": absolute_tick,
        "semantic_source_address": (
            f"tick:{absolute_tick}/phase:0/world:{cursor['logical_world_id']}"
        ),
        "pre_sequence_key_sha256": _digest(identity_character),
        "world_local_sequence": sequence,
        "deterministic_event_id": _digest(identity_character),
        "record_kind": "control_binding",
        "record_kind_version": 1,
        "control_binding_sha256": _digest("f"),
        "payload": {
            "schema_version": "eiel_science_control_binding_payload_v1",
            "payload": {
                "binding_id": f"binding-{identity_character}",
                "semantic_cell_id": cursor["cell_id"],
                "seed_bindings": {},
                "control_values": {},
                "control_schedules": [],
                "registered_profile_ids": [],
                "semantic_envelope_resolution_sha256": _digest("1"),
                "semantic_applied_readback_sha256": _digest("2"),
                "continuation_parent_execution_identity": None,
            },
        },
        "source_state_before_sha256": None,
        "source_state_after_sha256": None,
        "cause_event_ids": [],
        "lifecycle_epoch": 0,
        "legacy_compatibility": None,
    }


def _world_completion_record(
    cursor_projection: dict[str, object], *, completion_class: str
) -> dict[str, object]:
    record = _record(
        cursor_projection, absolute_tick=7, sequence=0, identity_character="6"
    )
    record["record_kind"] = "world_completion"
    record["payload"] = {
        "schema_version": "eiel_science_world_completion_payload_v1",
        "payload": {
            "completion_id": "completion-001",
            "completion_class": completion_class,
            "terminal_state_sha256": _digest("7"),
            "terminal_population_count": 0,
            "terminal_absolute_tick": 7,
            "next_absolute_tick": 7,
            "terminal_record_count": 1,
            "canonical_record_stream_sha256": _digest("8"),
            "checkpoint_occurrence_ids": [],
            "reason_code": "host-test",
        },
    }
    return record


def _native_agent_state_record(
    cursor_projection: dict[str, object],
) -> dict[str, object]:
    record = _record(
        cursor_projection, absolute_tick=7, sequence=0, identity_character="6"
    )
    record["record_kind"] = "agent_state_snapshot"
    record["payload"] = {
        "schema_version": SCIENCE_NATIVE_AGENT_STATE_SNAPSHOT_PAYLOAD_SCHEMA_V1,
        "payload": {
            "agent_state_id": "agent-state-001",
            "agent_id": 11,
            "alive": True,
            "position": 5,
            "energy": 9,
            "health": 8,
            "injury": 0,
            "age_ticks": 6,
            "absolute_birth_tick": 1,
            "genome_state_id": "genome-state-001",
            "policy_id": "policy-001",
            "founder_id": "founder-001",
            "lineage_id": "lineage-001",
            "lineage_depth": 1,
            "hereditary_package_state_ids": [],
            "linkage_state_id": "linkage-001",
        },
    }
    return record


def _root_bundle() -> dict[str, object]:
    cursors = _root_cursors()
    capture = ScienceTypedCaptureBufferV1(
        capacity_rows=0, starting_cursor_projections=cursors
    ).seal(checkpoint_at_tick=0)
    mapping = _mapping(cursors)
    controls = _controls()
    route = _route_binding(_digest("a"))
    base = _base_checkpoint(next_tick=0)
    envelope = seal_science_checkpoint_extension_v1(
        base_checkpoint=base,
        science_route_binding=route,
        world_mapping=mapping,
        control_table=controls,
        typed_capture_inventory=capture,
        continuation_parent_checkpoint_sha256=None,
    )
    return {
        "base": base,
        "route": route,
        "mapping": mapping,
        "controls": controls,
        "capture": capture,
        "envelope": envelope,
    }


def _continuation_bundle(root: dict[str, object]) -> dict[str, object]:
    root_capture = root["capture"]
    assert isinstance(root_capture, dict)
    buffer = ScienceTypedCaptureBufferV1(
        capacity_rows=4,
        starting_cursor_projections=root_capture["final_cursors"],
    )
    starting = root_capture["final_cursors"]
    assert isinstance(starting, list)
    buffer.append(
        _record(starting[1], absolute_tick=2, sequence=0, identity_character="7")
    )
    buffer.append(
        _record(starting[0], absolute_tick=1, sequence=0, identity_character="6")
    )
    capture = buffer.seal(checkpoint_at_tick=7)
    envelope = seal_science_checkpoint_extension_v1(
        base_checkpoint=_base_checkpoint(next_tick=7),
        science_route_binding=root["route"],
        world_mapping=root["mapping"],
        control_table=root["controls"],
        typed_capture_inventory=capture,
        continuation_parent_checkpoint_sha256=root["envelope"][
            "science_checkpoint_identity_sha256"
        ],
    )
    return {
        **root,
        "base": _base_checkpoint(next_tick=7),
        "capture": capture,
        "envelope": envelope,
    }


def test_science_contract_is_normative_pinned_and_not_a_readiness_claim() -> None:
    contract_path = PROJECT_ROOT / SCIENCE_CONTRACT_RELATIVE_PATH_V1
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["schema_version"] == SCIENCE_EXTENSION_CONTRACT_SCHEMA_V1
    assert contract["claim_boundary"] == {
        "current_b136_standard_supported": False,
        "native_capture_runtime_integrated": False,
        "science_route_qualified": False,
        "status": SCIENCE_EXTENSION_STATUS_V1,
        "technical_performance_is_science_evidence": False,
    }
    assert contract["canonical_science_contract_provenance"]["source_commit"] == (
        "b6a38af9c752771c3918ff28064824cff3d2c830"
    )
    assert tuple(contract["observation_contract"]["typed_record_kinds"]) == (
        SCIENCE_TYPED_RECORD_KINDS_V1
    )
    assert contract["route_binding"] == {
        "adapter_class": SCIENCE_ROUTE_BINDING_SCHEMA_V1,
        "complete_wire_projection_required": True,
        "numeric_uid_to_semantic_world_mapping_required": True,
        "semantic_dimensions": [
            "study_id",
            "test_id",
            "version_id",
            "arm_id",
            "replicate_id",
            "reducer_id",
        ],
        "wire_schema": SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1,
    }
    assert SCIENCE_CONTRACT_RELATIVE_PATH_V1 in BOUND_SOURCE_RELATIVE_PATHS_V2
    assert SCIENCE_CONTRACT_RELATIVE_PATH_V1 in NORMATIVE_EXECUTION_CONTRACT_PATHS_V2
    assert (
        "engine/perf/cuda_full_evolution_v2/science_extension.py"
        in BOUND_SOURCE_RELATIVE_PATHS_V2
    )


def test_fast_route_is_explicit_off_zero_allocation_and_runtime_hooks() -> None:
    projection = observation_off_projection_v1(
        base_checkpoint_state_sha256=_sha("a"),
        base_checkpoint_metadata_sha256=_sha("b"),
        next_absolute_tick=64,
    )
    assert projection["observation_mode"] == SCIENCE_OBSERVATION_OFF_V1
    assert projection["science_allocation_bytes"] == 0
    assert projection["science_hot_path_hooks"] == 0
    assert projection["trajectory_semantics"] == "unchanged"
    assert projection["science_route_qualified"] is False
    assert admit_science_extension_request_v1(
        {"observation_mode": "OFF", "control_schedule_kind": None}
    ) == {
        "observation_mode": "OFF",
        "control_schedule_kind": None,
        "science_allocation_bytes": 0,
        "science_hot_path_hooks": 0,
        "science_route_qualified": False,
    }
    runtime_source = (
        PROJECT_ROOT / "engine/perf/cuda_full_evolution_v2/runtime.py"
    ).read_text(encoding="utf-8")
    assert "science_extension" not in runtime_source


@pytest.mark.parametrize(
    "request_payload",
    (
        {
            "observation_mode": SCIENCE_OBSERVATION_TYPED_SIDECAR_V1,
            "control_schedule_kind": SCIENCE_STATIC_CONTROL_TABLE_V1,
        },
        {"observation_mode": "OFF", "control_schedule_kind": "DYNAMIC_PULSE_V1"},
    ),
)
def test_runtime_admission_rejects_observation_on_and_control_schedules(
    request_payload: dict[str, object],
) -> None:
    with pytest.raises(
        CudaFullEvolutionScienceExtensionError, match="host-adapter-only"
    ):
        admit_science_extension_request_v1(request_payload)


def test_complete_science_route_binding_is_validated_not_a_free_hash() -> None:
    route = _route_binding(_digest("a"))
    adapter = seal_science_route_binding_adapter_v1(route)
    assert validate_science_route_binding_v1(route) == route
    assert adapter["adapter_class"] == SCIENCE_ROUTE_BINDING_SCHEMA_V1
    assert adapter["source_schema_version"] == SCIENCE_ROUTE_BINDING_WIRE_SCHEMA_V1
    missing = deepcopy(route)
    del missing["checkpoint_adapter_sha256"]
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="missing"):
        validate_science_route_binding_v1(missing)
    bad_digest = deepcopy(route)
    bad_digest["observation_adapter_sha256"] = _sha("0")
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="digest object"):
        validate_science_route_binding_v1(bad_digest)


def test_cursor_wire_shape_and_prefix_identity_are_digest_object_exact() -> None:
    cursor = _root_cursors()[0]["cursor"]
    assert isinstance(cursor, dict)
    assert cursor["schema_version"] == SCIENCE_OBSERVATION_PREFIX_CURSOR_SCHEMA_V1
    assert cursor["prefix_chain_sha256"] == (
        empty_science_observation_prefix_sha256_v1()
    )
    assert cursor["next_absolute_tick"] == cursor["checkpoint_at_tick"] == 0
    assert cursor["next_world_local_sequence"] == cursor["prefix_record_count"] == 0
    assert cursor["effects_performed"] is False
    assert set(cursor["science_semantic_acquisition_sha256"]) == {
        "algorithm",
        "hex",
    }
    tampered = deepcopy(cursor)
    tampered["semantic_execution_id"] = _digest("0")
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="semantic"):
        validate_science_observation_prefix_cursor_v1(tampered)


def test_world_mapping_and_absolute_tick_controls_are_order_neutral() -> None:
    cursors = _root_cursors()
    first = _mapping(cursors)
    second = seal_science_world_mapping_v1(list(reversed(first["rows"])))
    assert first == second
    assert [row["cuda_world_uid"] for row in first["rows"]] == [101, 102]
    controls = _controls()
    assert [row["cuda_world_uid"] for row in controls["rows"]] == [101, 102]
    assert controls["time_authority"] == "absolute_tick_v1"
    assert controls["rows"][0]["intervals"][0]["end_absolute_tick"] == 11
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="contiguous"):
        seal_static_science_control_table_v1(
            [
                {
                    "cuda_world_uid": 101,
                    "intervals": [
                        {
                            "start_absolute_tick": 1,
                            "end_absolute_tick": 11,
                            "control_payload_sha256": _digest("1"),
                        }
                    ],
                }
            ],
            terminal_next_absolute_tick=11,
        )


@pytest.mark.parametrize("invalid_uid", (False, 1.0, 0, -1, 1 << 63))
def test_all_public_cuda_uid_seams_reject_out_of_domain_values(
    invalid_uid: object,
) -> None:
    cursors = _root_cursors()
    mapping_row = deepcopy(_mapping(cursors)["rows"][0])
    mapping_row["cuda_world_uid"] = invalid_uid
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="positive"):
        seal_science_world_mapping_v1([mapping_row])

    control_row = deepcopy(_controls()["rows"][0])
    control_row["cuda_world_uid"] = invalid_uid
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="positive"):
        seal_static_science_control_table_v1(
            [control_row], terminal_next_absolute_tick=11
        )

    projection = deepcopy(cursors[0])
    projection["cuda_world_uid"] = invalid_uid
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="positive"):
        ScienceTypedCaptureBufferV1(
            capacity_rows=0, starting_cursor_projections=[projection]
        )

    checkpoint = _base_checkpoint(next_tick=0)
    base_projection = {
        field: getattr(checkpoint, field)
        for field in (
            "schema_version",
            "time_authority",
            "canonical_state_sha256",
            "metadata_seal_sha256",
            "acknowledged_event_next_tick",
            "world_uids",
        )
    }
    base_projection["world_uids"] = [invalid_uid]
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="positive"):
        project_base_checkpoint_identity_v1(base_projection)


def test_all_public_cuda_uid_seams_accept_int64_max() -> None:
    int64_max = (1 << 63) - 1
    cursors = _root_cursors()
    mapping_row = deepcopy(_mapping(cursors)["rows"][0])
    mapping_row["cuda_world_uid"] = int64_max
    assert (
        seal_science_world_mapping_v1([mapping_row])["rows"][0]["cuda_world_uid"]
        == int64_max
    )

    control_row = deepcopy(_controls()["rows"][0])
    control_row["cuda_world_uid"] = int64_max
    assert (
        seal_static_science_control_table_v1(
            [control_row], terminal_next_absolute_tick=11
        )["rows"][0]["cuda_world_uid"]
        == int64_max
    )

    projection = deepcopy(cursors[0])
    projection["cuda_world_uid"] = int64_max
    inventory = ScienceTypedCaptureBufferV1(
        capacity_rows=0, starting_cursor_projections=[projection]
    ).seal(checkpoint_at_tick=0)
    assert inventory["final_cursors"][0]["cuda_world_uid"] == int64_max

    checkpoint = _base_checkpoint(next_tick=0)
    base_projection = {
        field: getattr(checkpoint, field)
        for field in (
            "schema_version",
            "time_authority",
            "canonical_state_sha256",
            "metadata_seal_sha256",
            "acknowledged_event_next_tick",
            "world_uids",
        )
    }
    base_projection["world_uids"] = [int64_max]
    assert project_base_checkpoint_identity_v1(base_projection)["world_uids"] == [
        int64_max
    ]


def test_static_control_start_tick_rejects_boolean_with_recomputed_seal() -> None:
    controls = deepcopy(_controls())
    controls["start_next_absolute_tick"] = False
    body = {
        key: value for key, value in controls.items() if key != "control_table_sha256"
    }
    controls["control_table_sha256"] = _canonical_digest(body)
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="envelope"):
        validate_static_science_control_table_v1(controls)


def test_typed_capture_stores_exact_records_and_extends_canonical_cursor() -> None:
    cursors = _root_cursors()
    first_record = _record(
        cursors[0], absolute_tick=1, sequence=0, identity_character="6"
    )
    original = deepcopy(first_record)
    buffer = ScienceTypedCaptureBufferV1(
        capacity_rows=2, starting_cursor_projections=cursors
    )
    buffer.append(first_record)
    inventory = buffer.seal(checkpoint_at_tick=7)
    assert first_record == original
    assert validate_typed_science_capture_inventory_v1(inventory) == inventory
    assert inventory["committed_rows"] == 1
    assert inventory["records"][0] == (
        validate_science_observation_record_projection_v1(first_record)
    )
    assert inventory["records"][0]["record_kind"] == "control_binding"
    assert science_semantic_record_sha256_v1(first_record)["algorithm"] == "sha256"
    final_cursor = inventory["final_cursors"][0]["cursor"]
    assert final_cursor["next_absolute_tick"] == 7
    assert final_cursor["next_world_local_sequence"] == 1
    assert final_cursor["prefix_record_count"] == 1
    assert final_cursor["previous_cursor_sha256"] == cursors[0]["cursor_sha256"]


def test_public_prefix_extension_rejects_mixed_worlds_and_sequence_gaps() -> None:
    cursors = _root_cursors()
    first = _record(cursors[0], absolute_tick=1, sequence=0, identity_character="6")
    other_world = _record(
        cursors[1], absolute_tick=1, sequence=1, identity_character="7"
    )
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="mixes worlds"):
        extend_science_observation_prefix_sha256_v1(
            empty_science_observation_prefix_sha256_v1(),
            [first, other_world],
        )

    sequence_gap = _record(
        cursors[0], absolute_tick=2, sequence=2, identity_character="8"
    )
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="contiguous"):
        extend_science_observation_prefix_sha256_v1(
            empty_science_observation_prefix_sha256_v1(),
            [first, sequence_gap],
        )


def test_typed_capture_is_partition_and_cross_world_packing_neutral() -> None:
    cursors = _root_cursors()
    records = [
        _record(cursors[0], absolute_tick=1, sequence=0, identity_character="6"),
        _record(cursors[1], absolute_tick=1, sequence=0, identity_character="7"),
    ]
    first = ScienceTypedCaptureBufferV1(
        capacity_rows=2, starting_cursor_projections=cursors
    )
    first.extend(records)
    second = ScienceTypedCaptureBufferV1(
        capacity_rows=2, starting_cursor_projections=list(reversed(cursors))
    )
    for record in reversed(records):
        second.extend([record])
    assert first.seal(checkpoint_at_tick=7) == second.seal(checkpoint_at_tick=7)


def test_typed_capture_overflow_and_sequence_gap_fail_before_frontier_commit() -> None:
    cursors = _root_cursors()
    buffer = ScienceTypedCaptureBufferV1(
        capacity_rows=0, starting_cursor_projections=cursors
    )
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="overflow"):
        buffer.append(
            _record(cursors[0], absolute_tick=1, sequence=0, identity_character="6")
        )
    assert buffer.overflowed is True
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="overflow"):
        buffer.seal(checkpoint_at_tick=7)
    gap = ScienceTypedCaptureBufferV1(
        capacity_rows=1, starting_cursor_projections=cursors
    )
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="contiguous"):
        gap.append(
            _record(cursors[0], absolute_tick=1, sequence=1, identity_character="6")
        )


def test_typed_capture_rejects_per_world_canonical_order_regression() -> None:
    cursor = _root_cursors()[0]
    buffer = ScienceTypedCaptureBufferV1(
        capacity_rows=2, starting_cursor_projections=[cursor]
    )
    buffer.append(_record(cursor, absolute_tick=5, sequence=0, identity_character="6"))
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="order"):
        buffer.append(
            _record(cursor, absolute_tick=1, sequence=1, identity_character="7")
        )


def test_typed_capture_rejects_uppercase_alias_and_generation_authority() -> None:
    record = _record(
        _root_cursors()[0], absolute_tick=1, sequence=0, identity_character="6"
    )
    uppercase = deepcopy(record)
    uppercase["record_kind"] = "CONTROL_BINDING"
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="canonical"):
        validate_science_observation_record_projection_v1(uppercase)
    generation = deepcopy(record)
    generation["payload"]["payload"]["generation"] = 1
    with pytest.raises(CudaFullEvolutionScienceExtensionError):
        validate_science_observation_record_projection_v1(generation)

    padded_producer = deepcopy(record)
    padded_producer["producer_component_id"] += " "
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="unpadded"):
        validate_science_observation_record_projection_v1(padded_producer)


def test_native_agent_state_variant_carries_absolute_birth_tick() -> None:
    record = _native_agent_state_record(_root_cursors()[0])

    assert validate_science_observation_record_projection_v1(record) == record

    generic = deepcopy(record)
    generic["payload"]["schema_version"] = (
        "eiel_science_agent_state_snapshot_payload_v1"
    )
    with pytest.raises(
        CudaFullEvolutionScienceExtensionError,
        match="frozen record-kind contract",
    ):
        validate_science_observation_record_projection_v1(generic)


def test_schema_version_fields_obey_the_pinned_ascii_token_domain() -> None:
    record = _record(
        _root_cursors()[0], absolute_tick=1, sequence=0, identity_character="6"
    )
    valid = deepcopy(record)
    valid["producer_contract_version"] = "A._+-9"
    valid["payload"]["schema_version"] = "P._+-9"
    valid["legacy_compatibility"] = {
        "schema_version": "L._+-9",
        "payload": {},
    }
    assert validate_science_observation_record_projection_v1(valid) == valid

    padded_or_spaced = deepcopy(record)
    padded_or_spaced["producer_contract_version"] = "science adapter_v1"
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="schema-version"):
        validate_science_observation_record_projection_v1(padded_or_spaced)

    invalid_payload_token = deepcopy(record)
    invalid_payload_token["payload"]["schema_version"] = "science/payload_v1"
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="schema-version"):
        validate_science_observation_record_projection_v1(invalid_payload_token)

    invalid_legacy_token = deepcopy(record)
    invalid_legacy_token["legacy_compatibility"] = {
        "schema_version": "legacy schema_v1",
        "payload": {},
    }
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="schema-version"):
        validate_science_observation_record_projection_v1(invalid_legacy_token)


def test_pinned_integer_fields_accept_int64_max_and_reject_larger_values() -> None:
    int64_max = (1 << 63) - 1
    record = _record(
        _root_cursors()[0], absolute_tick=1, sequence=0, identity_character="6"
    )
    for field in (
        "logical_world_id",
        "absolute_tick",
        "world_local_sequence",
        "lifecycle_epoch",
    ):
        boundary = deepcopy(record)
        boundary[field] = int64_max
        assert validate_science_observation_record_projection_v1(boundary)[field] == (
            int64_max
        )
        overflow = deepcopy(record)
        overflow[field] = int64_max + 1
        with pytest.raises(CudaFullEvolutionScienceExtensionError, match="integer"):
            validate_science_observation_record_projection_v1(overflow)
    overflow_version = deepcopy(record)
    overflow_version["record_kind_version"] = int64_max + 1
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="integer"):
        validate_science_observation_record_projection_v1(overflow_version)
    boolean_version = deepcopy(record)
    boolean_version["record_kind_version"] = True
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="integer"):
        validate_science_observation_record_projection_v1(boolean_version)

    cursor = deepcopy(_root_cursors()[0]["cursor"])
    assert isinstance(cursor, dict)
    boundary_cursor = deepcopy(cursor)
    boundary_cursor["logical_world_id"] = int64_max
    boundary_cursor["checkpoint_at_tick"] = int64_max
    boundary_cursor["next_absolute_tick"] = int64_max
    boundary_cursor["next_world_local_sequence"] = int64_max
    boundary_cursor["prefix_record_count"] = int64_max
    boundary_cursor["last_deterministic_event_id"] = _digest("6")
    assert validate_science_observation_prefix_cursor_v1(boundary_cursor) == (
        boundary_cursor
    )
    for field in (
        "logical_world_id",
        "checkpoint_at_tick",
        "next_absolute_tick",
        "next_world_local_sequence",
        "prefix_record_count",
    ):
        overflow_cursor = deepcopy(cursor)
        overflow_cursor[field] = int64_max + 1
        with pytest.raises(CudaFullEvolutionScienceExtensionError, match="integer"):
            validate_science_observation_prefix_cursor_v1(overflow_cursor)


def test_generation_authority_is_rejected_except_inside_legacy_subtrees() -> None:
    record = _record(
        _root_cursors()[0], absolute_tick=1, sequence=0, identity_character="6"
    )
    authoritative = deepcopy(record)
    authoritative["payload"]["payload"]["control_values"] = {"generation": 3}
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="generation"):
        validate_science_observation_record_projection_v1(authoritative)

    nested_legacy = deepcopy(record)
    nested_legacy["payload"]["payload"]["control_values"] = {
        "legacy_compatibility": {
            "generation": 3,
            "nested": [{"ticks_per_generation": 160}],
        }
    }
    assert validate_science_observation_record_projection_v1(nested_legacy) == (
        nested_legacy
    )
    top_level_legacy = deepcopy(record)
    top_level_legacy["legacy_compatibility"] = {
        "schema_version": "legacy_display_v1",
        "payload": {"generation": 3},
    }
    assert validate_science_observation_record_projection_v1(top_level_legacy) == (
        top_level_legacy
    )


def test_world_completion_class_is_exactly_the_pinned_closed_domain() -> None:
    cursor = _root_cursors()[0]
    for completion_class in (
        "EMPTY_INITIAL",
        "EXTINCT",
        "NO_BIRTHS",
        "TERMINAL_HORIZON",
    ):
        record = _world_completion_record(cursor, completion_class=completion_class)
        assert validate_science_observation_record_projection_v1(record) == record
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="completion"):
        validate_science_observation_record_projection_v1(
            _world_completion_record(cursor, completion_class="UNKNOWN")
        )


def test_checkpoint_extension_requires_one_shared_run_identity() -> None:
    root = _root_bundle()
    final_cursors = root["capture"]["final_cursors"]
    assert (
        len(
            {
                projection["cursor"]["crca_semantic_run_digest"]["hex"]
                for projection in final_cursors
            }
        )
        == 1
    )
    assert (
        len(
            {
                projection["cursor"]["semantic_execution_id"]["hex"]
                for projection in final_cursors
            }
        )
        == 1
    )
    assert root["envelope"]["continuation_kind"] == "ROOT"

    cursors = _root_cursors()
    mixed_cursor = seal_empty_science_observation_cursor_v1(
        instrument_manifest_sha256=_digest("a"),
        crca_semantic_run_digest=_digest("f"),
        science_semantic_acquisition_sha256=_digest("c"),
        cell_id="cell-001",
        logical_world_id=1,
        world_uid=_digest("e"),
    )
    cursors[1] = {
        "cuda_world_uid": 102,
        "cursor": mixed_cursor,
        "cursor_sha256": science_observation_cursor_sha256_v1(mixed_cursor),
    }
    mixed_capture = ScienceTypedCaptureBufferV1(
        capacity_rows=0, starting_cursor_projections=cursors
    ).seal(checkpoint_at_tick=0)
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="CRCA run"):
        seal_science_checkpoint_extension_v1(
            base_checkpoint=_base_checkpoint(next_tick=0),
            science_route_binding=_route_binding(_digest("a")),
            world_mapping=_mapping(cursors),
            control_table=_controls(),
            typed_capture_inventory=mixed_capture,
            continuation_parent_checkpoint_sha256=None,
        )


def test_root_checkpoint_seals_but_only_restore_gate_completes_identity() -> None:
    root = _root_bundle()
    envelope = root["envelope"]
    assert envelope["schema_version"] == SCIENCE_CHECKPOINT_EXTENSION_SCHEMA_V1
    assert envelope["continuation_kind"] == "ROOT"
    assert envelope["science_continuable"] is False
    assert validate_science_checkpoint_extension_v1(envelope) == envelope
    gate = admit_science_checkpoint_restore_v1(
        envelope,
        base_checkpoint=root["base"],
        science_route_binding=root["route"],
        world_mapping=root["mapping"],
        control_table=root["controls"],
        typed_capture_inventory=root["capture"],
    )
    assert gate["opaque_extension_restore_admitted"] is True
    assert gate["science_continuation_identity_complete"] is True
    assert gate["science_continuable"] is False
    assert gate["science_route_qualified"] is False


def test_continuation_restore_requires_exact_parent_and_cursor_frontier() -> None:
    root = _root_bundle()
    child = _continuation_bundle(root)
    envelope = child["envelope"]
    assert envelope["continuation_kind"] == "CONTINUATION"
    assert (
        envelope["continuation_parent_checkpoint_sha256"]
        == root["envelope"]["science_checkpoint_identity_sha256"]
    )
    gate = admit_science_checkpoint_restore_v1(
        envelope,
        base_checkpoint=child["base"],
        science_route_binding=child["route"],
        world_mapping=child["mapping"],
        control_table=child["controls"],
        typed_capture_inventory=child["capture"],
        parent_envelope=root["envelope"],
        parent_typed_capture_inventory=root["capture"],
    )
    assert gate["next_absolute_tick"] == 7
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="requires parent"):
        admit_science_checkpoint_restore_v1(
            envelope,
            base_checkpoint=child["base"],
            science_route_binding=child["route"],
            world_mapping=child["mapping"],
            control_table=child["controls"],
            typed_capture_inventory=child["capture"],
        )


def test_continuation_restore_rejects_parent_cursor_discontinuity() -> None:
    root = _root_bundle()
    child = _continuation_bundle(root)
    wrong_parent_capture = deepcopy(root["capture"])
    wrong_parent_capture["final_cursors"][0]["cursor_sha256"] = _digest("0")
    with pytest.raises(CudaFullEvolutionScienceExtensionError):
        admit_science_checkpoint_restore_v1(
            child["envelope"],
            base_checkpoint=child["base"],
            science_route_binding=child["route"],
            world_mapping=child["mapping"],
            control_table=child["controls"],
            typed_capture_inventory=child["capture"],
            parent_envelope=root["envelope"],
            parent_typed_capture_inventory=wrong_parent_capture,
        )


@pytest.mark.parametrize(
    "field",
    (
        "science_semantic_acquisition_sha256",
        "observation_cursor_set_sha256",
        "control_table_sha256",
        "continuation_parent_checkpoint_sha256",
        "world_mapping_sha256",
        "typed_capture_inventory_sha256",
        "science_route_binding_sha256",
        "extension_sha256",
        "science_checkpoint_identity_sha256",
    ),
)
def test_checkpoint_extension_rejects_identity_tampering(field: str) -> None:
    envelope = deepcopy(_continuation_bundle(_root_bundle())["envelope"])
    envelope[field] = _digest("0")
    with pytest.raises(CudaFullEvolutionScienceExtensionError):
        validate_science_checkpoint_extension_v1(envelope)


@pytest.mark.parametrize(
    "field",
    (
        "capture_overflowed",
        "native_capture_runtime_integrated",
        "science_route_qualified",
        "science_continuable",
    ),
)
def test_checkpoint_boolean_constants_reject_integer_with_recomputed_seal(
    field: str,
) -> None:
    envelope = deepcopy(_root_bundle()["envelope"])
    envelope[field] = 0
    _reseal_checkpoint_extension(envelope)
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="authority"):
        validate_science_checkpoint_extension_v1(envelope)


def test_checkpoint_control_cursor_rejects_boolean_with_recomputed_seal() -> None:
    envelope = deepcopy(_root_bundle()["envelope"])
    envelope["control_cursor_next_absolute_tick"] = False
    _reseal_checkpoint_extension(envelope)
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="integer"):
        validate_science_checkpoint_extension_v1(envelope)


@pytest.mark.parametrize("field", ("capture_overflowed", "capture_effects_performed"))
def test_capture_inventory_booleans_reject_integer_with_recomputed_seal(
    field: str,
) -> None:
    inventory = deepcopy(_root_bundle()["capture"])
    inventory[field] = 0
    _reseal_capture_inventory(inventory)
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="authority"):
        validate_typed_science_capture_inventory_v1(inventory)


@pytest.mark.parametrize("invalid_count", (False, 0.0, 1 << 63))
@pytest.mark.parametrize("field", ("committed_rows", "record_kind_counts"))
def test_capture_inventory_counts_reject_noncanonical_integer_domains(
    field: str, invalid_count: object
) -> None:
    inventory = deepcopy(_root_bundle()["capture"])
    if field == "committed_rows":
        inventory[field] = invalid_count
    else:
        inventory[field]["control_binding"] = invalid_count
    _reseal_capture_inventory(inventory)
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="integer"):
        validate_typed_science_capture_inventory_v1(inventory)


def test_cursor_effects_boolean_rejects_integer_with_recomputed_seal() -> None:
    projection = deepcopy(_root_cursors()[0])
    projection["cursor"]["effects_performed"] = 0
    projection["cursor_sha256"] = _canonical_digest(projection["cursor"])
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="effect-free"):
        ScienceTypedCaptureBufferV1(
            capacity_rows=0, starting_cursor_projections=[projection]
        )


def test_checkpoint_extension_cannot_be_sealed_from_hashes_without_sidecars() -> None:
    root = _root_bundle()
    missing_route = deepcopy(root["route"])
    del missing_route["checkpoint_adapter_sha256"]
    with pytest.raises(CudaFullEvolutionScienceExtensionError, match="missing"):
        seal_science_checkpoint_extension_v1(
            base_checkpoint=root["base"],
            science_route_binding=missing_route,
            world_mapping=root["mapping"],
            control_table=root["controls"],
            typed_capture_inventory=root["capture"],
            continuation_parent_checkpoint_sha256=None,
        )
