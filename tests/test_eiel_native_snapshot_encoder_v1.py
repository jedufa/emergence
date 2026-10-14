from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import stat
import struct
import threading

import pytest
import torch

import engine.native_runtime.cuda_v2_science_capture_v1 as capture_adapter
from emergence_contracts.science_compact_capture import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
)
from engine.native_runtime.cuda_snapshot_encoder_v1 import (
    compile_native_snapshot_encoder_plan_v1,
    encode_native_snapshot_to_stage_file_v1,
    encode_native_snapshot_v1,
    encode_owned_native_snapshot_to_stage_file_v1,
)
from engine.native_runtime.cuda_snapshot_encoder_v1.contract import (
    NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
    NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1,
)
from engine.native_runtime.cuda_snapshot_encoder_v1.build import (
    build_native_snapshot_encoder_v1,
)
from engine.native_runtime.cuda_snapshot_encoder_v1.native_loader import (
    NativeSnapshotEncoderLoadError,
    load_native_snapshot_encoder_v1,
    validate_native_snapshot_encoder_manifest_v1,
)
from engine.native_runtime.cuda_snapshot_plan_v1 import (
    NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1,
    NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1,
    NATIVE_SNAPSHOT_STATE_CAPTURE_TRANSPORT_SCHEMA_V1,
    compile_cuda_snapshot_plan_v1,
)
from engine.native_runtime.cuda_v2_compact_science_capture_v1 import (
    write_compact_science_snapshot_v1,
)


CONTROL_SET_SHA256 = "3" * 64
PUBLICATION_CONTEXT_SHA256 = "4" * 64


@pytest.fixture(scope="session")
def native_encoder(tmp_path_factory: pytest.TempPathFactory):
    output = tmp_path_factory.mktemp("native-snapshot-encoder") / "sealed"
    artifact = build_native_snapshot_encoder_v1(output)
    return load_native_snapshot_encoder_v1(artifact)


def _pack_field(
    raw: bytearray,
    field: str,
    wire_type: str,
    index: int,
    value: int,
) -> None:
    size = struct.calcsize(wire_type)
    struct.pack_into(
        wire_type,
        raw,
        capture_adapter._field_offset(field) + size * index,
        value,
    )


def _world_state(
    *,
    world_uid: int,
    absolute_tick: int,
    agents: list[tuple[int, int, bool]],
    extrema: bool = False,
) -> bytes:
    raw = bytearray(capture_adapter._STATE_BYTES)
    _pack_field(raw, "scalar_header_counters", "<Q", 0, world_uid)
    _pack_field(raw, "scalar_header_counters", "<Q", 1, absolute_tick)
    _pack_field(
        raw,
        "scalar_header_counters",
        "<Q",
        3,
        sum(int(alive) for _, _, alive in agents),
    )
    _pack_field(raw, "scalar_header_counters", "<Q", 15, 2**64 - 1 if extrema else 17)
    _pack_field(raw, "scalar_header_counters", "<Q", 16, 2**63 if extrema else 5)
    _pack_field(raw, "scalar_header_counters", "<Q", 17, 2**32 if extrema else 3)
    _pack_field(raw, "scalar_header_counters", "<Q", 18, 2**16 if extrema else 2)

    if extrema:
        for index in range(1024):
            _pack_field(
                raw,
                "resource",
                "<H",
                index,
                65535 if index % 2 else 0,
            )
    else:
        for index, value in enumerate((7, 5, 3, 1)):
            _pack_field(raw, "resource", "<H", index, value + world_uid % 3)

    alive_words = [0] * capture_adapter._ALIVE_WORDS
    policy_offset = capture_adapter._field_offset("policy_alleles")
    trait_offset = capture_adapter._field_offset("trait_alleles")
    for slot, agent_id, alive in agents:
        if alive:
            alive_words[slot // 32] |= 1 << (slot % 32)
        _pack_field(raw, "agent_id", "<Q", slot, agent_id)
        _pack_field(
            raw,
            "parent_a_id",
            "<Q",
            slot,
            (2**64 - 2) if extrema else agent_id // 2,
        )
        _pack_field(raw, "parent_b_id", "<Q", slot, 2**63 if extrema else 0)
        _pack_field(raw, "position", "<H", slot, 65535 if extrema else slot * 7)
        _pack_field(raw, "energy", "<h", slot, -32768 if extrema else 100 - slot)
        _pack_field(raw, "health", "<h", slot, 32767 if extrema else 80 - slot)
        _pack_field(raw, "age", "<H", slot, 65535 if extrema else slot + 1)
        _pack_field(
            raw,
            "absolute_birth_tick",
            "<Q",
            slot,
            2**64 - 1 if extrema else max(0, absolute_tick - slot),
        )
        raw[policy_offset + slot] = 0x80 if extrema else agent_id % 101
        raw[policy_offset + capture_adapter._CAPACITY + slot] = (
            0x7F if extrema else (-(agent_id % 31)) & 0xFF
        )
        struct.pack_into(
            "<h",
            raw,
            trait_offset + 2 * slot,
            -32768 if extrema else -(agent_id % 1000),
        )
        struct.pack_into(
            "<h",
            raw,
            trait_offset + 2 * (capture_adapter._CAPACITY + slot),
            32767 if extrema else agent_id % 1000,
        )
    for index, value in enumerate(alive_words):
        _pack_field(raw, "alive_words", "<I", index, value)
    return bytes(raw)


def _metrics(
    world_count: int,
    agents_by_world: tuple[list[tuple[int, int, bool]], ...],
    *,
    extrema: bool = False,
) -> torch.Tensor:
    tensor = torch.zeros(
        (
            world_count,
            capture_adapter._CAPACITY,
            len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
        ),
        dtype=torch.int64,
    )
    for world_index, agents in enumerate(agents_by_world):
        for slot, agent_id, _alive in agents:
            tensor[world_index, slot, 0] = -(2**63) if extrema else agent_id * 10
            tensor[world_index, slot, 1] = 2**63 - 1 if extrema else slot
            tensor[world_index, slot, -1] = -1 if extrema else slot + 1000
    return tensor


def _metrics_projection(
    tensor: torch.Tensor,
    *,
    world_uids: tuple[int, ...],
    absolute_tick: int,
) -> dict[str, object]:
    raw = tensor.numpy().astype("<i8", copy=False).tobytes(order="C")
    return {
        "schema_version": "eiel_full_evolution_cuda_active_science_metrics_v1",
        "next_absolute_tick": absolute_tick,
        "world_uids": list(world_uids),
        "metric_fields": list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
        "shape": list(tensor.shape),
        "content_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor": tensor,
    }


def _world_cells(
    world_uids: tuple[int, ...],
    schedules: tuple[tuple[int, ...], ...],
    *,
    cell_ids: tuple[str, ...] | None = None,
) -> dict[str, dict[str, object]]:
    resolved_cell_ids = cell_ids or tuple(
        f"cell-{world_uid}" for world_uid in world_uids
    )
    rows = list(zip(world_uids, schedules, resolved_cell_ids, strict=True))
    return {
        cell_id: {
            "science_logical_world_id": world_uid,
            "science_ownership": {
                "study_id": f"study-{world_uid}",
                "arm_id": f"arm-{world_uid}",
            },
            "science_world_sha256": {
                "algorithm": "sha256",
                "hex": f"{world_uid % 16:x}" * 64,
            },
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": f"{(world_uid + 1) % 16:x}" * 64,
            },
            "science_acquisition": {"snapshot_ticks": list(ticks)},
        }
        for world_uid, ticks, cell_id in reversed(rows)
    }


def _old_and_native(
    tmp_path: Path,
    *,
    loaded_extension,
    world_uids: tuple[int, ...],
    schedules: tuple[tuple[int, ...], ...],
    absolute_tick: int,
    agents_by_world: tuple[list[tuple[int, int, bool]], ...],
    catalog_tick: int | None,
    extrema: bool = False,
    cell_ids: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], dict[str, object], bytes, bytes]:
    cells = _world_cells(world_uids, schedules, cell_ids=cell_ids)
    plan = compile_cuda_snapshot_plan_v1(
        world_uids=world_uids,
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SET_SHA256,
        execution_publication_context_sha256=PUBLICATION_CONTEXT_SHA256,
        catalog_absolute_tick=catalog_tick,
    )
    native_plan = compile_native_snapshot_encoder_plan_v1(
        plan, loaded_extension=loaded_extension
    )
    state_bytes = b"".join(
        _world_state(
            world_uid=world_uid,
            absolute_tick=absolute_tick,
            agents=agents,
            extrema=extrema,
        )
        for world_uid, agents in zip(world_uids, agents_by_world, strict=True)
    )
    state = torch.frombuffer(bytearray(state_bytes), dtype=torch.uint8).reshape(
        len(world_uids), capture_adapter._STATE_BYTES
    )
    metrics = _metrics(len(world_uids), agents_by_world, extrema=extrema)
    projection = _metrics_projection(
        metrics,
        world_uids=world_uids,
        absolute_tick=absolute_tick,
    )
    native = encode_native_snapshot_v1(
        native_plan,
        state=state,
        active_metrics_projection=projection,
        absolute_tick=absolute_tick,
    )
    expected = write_compact_science_snapshot_v1(
        tmp_path / f"legacy-{absolute_tick}",
        state=state_bytes,
        active_metrics_projection=projection,
        world_cells=cells,
        absolute_tick=absolute_tick,
        native_world_control_set_sha256=CONTROL_SET_SHA256,
        include_genome_catalog=catalog_tick == absolute_tick,
    )
    legacy_gzip = (
        Path(str(expected["path"])) / "snapshot.jsonl.gz"
    ).read_bytes()
    legacy_jsonl = gzip.decompress(legacy_gzip)
    return native, expected, legacy_jsonl, legacy_gzip


def _native_case(
    loaded_extension,
    *,
    world_uids: tuple[int, ...],
    schedules: tuple[tuple[int, ...], ...],
    absolute_tick: int,
    agents_by_world: tuple[list[tuple[int, int, bool]], ...],
    catalog_tick: int | None = None,
):
    cells = _world_cells(world_uids, schedules)
    plan = compile_cuda_snapshot_plan_v1(
        world_uids=world_uids,
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SET_SHA256,
        execution_publication_context_sha256=PUBLICATION_CONTEXT_SHA256,
        catalog_absolute_tick=catalog_tick,
    )
    native_plan = compile_native_snapshot_encoder_plan_v1(
        plan, loaded_extension=loaded_extension
    )
    state_bytes = b"".join(
        _world_state(
            world_uid=world_uid,
            absolute_tick=absolute_tick,
            agents=agents,
        )
        for world_uid, agents in zip(world_uids, agents_by_world, strict=True)
    )
    state = torch.frombuffer(bytearray(state_bytes), dtype=torch.uint8).reshape(
        len(world_uids), capture_adapter._STATE_BYTES
    )
    metrics = _metrics(len(world_uids), agents_by_world)
    projection = _metrics_projection(
        metrics,
        world_uids=world_uids,
        absolute_tick=absolute_tick,
    )
    return plan, native_plan, state, metrics, projection


def _assert_exact_v1_parity(
    native: dict[str, object],
    expected: dict[str, object],
    legacy_jsonl: bytes,
    legacy_gzip: bytes,
) -> None:
    assert native["canonical_jsonl"] == legacy_jsonl
    assert native["deterministic_gzip_level_9"] == legacy_gzip
    assert native["uncompressed_content_sha256"] == expected[
        "uncompressed_content_sha256"
    ]
    assert native["compressed_file_sha256"] == expected["compressed_file_sha256"]
    assert native["row_count"] == expected["row_count"]
    assert native["captured_world_count"] == expected["captured_world_count"]
    assert native["genome_catalog_included"] == expected["genome_catalog_included"]
    assert native["uncompressed_bytes"] == len(legacy_jsonl)
    assert native["compressed_bytes"] == len(legacy_gzip)
    assert hashlib.sha256(bytes(native["canonical_jsonl"])).hexdigest() == native[
        "uncompressed_content_sha256"
    ]
    assert hashlib.sha256(
        bytes(native["deterministic_gzip_level_9"])
    ).hexdigest() == native["compressed_file_sha256"]
    assert native["host_only"] is True
    assert native["synchronous"] is True
    assert native["filesystem_publication_performed"] is False
    assert native["native_world_control_set_sha256"] == CONTROL_SET_SHA256
    assert native["execution_publication_context_sha256"] == (
        PUBLICATION_CONTEXT_SHA256
    )
    assert native["metric_fields"] == list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
    assert native["input_ownership_contract"] == (
        "caller_immutable_for_full_synchronous_call_v1"
    )
    assert native["gil_held_for_full_synchronous_call"] is True
    assert native["input_versions_unchanged"] is True
    assert native["state_completion_version"] == native["state_input_version"]
    assert native["metrics_completion_version"] == native["metrics_input_version"]
    assert native["active_metrics_authentication"] == (
        "validated_external_projection_descriptor_v1"
    )
    assert len(native["active_metrics_content_sha256"]) == 64
    assert "total_native_work_ns" not in native["timing"]
    assert native["timing"]["external_full_call_elapsed_ns"] >= native["timing"][
        "core_encode_compress_hash_ns"
    ]


@pytest.mark.parametrize("distribution", ("zero", "all", "sparse"))
def test_native_encoder_exact_v1_alive_and_order_parity(
    tmp_path: Path,
    native_encoder,
    distribution: str,
) -> None:
    if distribution == "zero":
        agents = [(9, 91, False), (2, 22, False)]
    elif distribution == "all":
        agents = [
            (slot, capture_adapter._CAPACITY - slot, True)
            for slot in range(capture_adapter._CAPACITY)
        ]
    else:
        agents = [(7, 30, True), (2, 10, False), (12, 20, True)]
    native, expected, jsonl, compressed = _old_and_native(
        tmp_path,
        loaded_extension=native_encoder,
        world_uids=(101,),
        schedules=((13,),),
        absolute_tick=13,
        agents_by_world=(agents,),
        catalog_tick=None,
    )
    _assert_exact_v1_parity(native, expected, jsonl, compressed)


@pytest.mark.parametrize("absolute_tick", (0, 9))
def test_native_encoder_exact_v1_heterogeneous_world_and_catalog_parity(
    tmp_path: Path,
    native_encoder,
    absolute_tick: int,
) -> None:
    native, expected, jsonl, compressed = _old_and_native(
        tmp_path,
        loaded_extension=native_encoder,
        world_uids=(202, 101),
        schedules=((0, 9), (0,)),
        absolute_tick=absolute_tick,
        agents_by_world=(
            [(7, 30, True), (2, 10, False), (12, 20, True)],
            [(3, 5, True)],
        ),
        catalog_tick=0,
    )
    _assert_exact_v1_parity(native, expected, jsonl, compressed)
    expected_worlds = 2 if absolute_tick == 0 else 1
    assert native["captured_world_count"] == expected_worlds
    assert native["selected_world_indices"] == (
        [0, 1] if absolute_tick == 0 else [0]
    )
    assert native["selected_world_uids"] == (
        [202, 101] if absolute_tick == 0 else [202]
    )


def test_native_encoder_exact_v1_integer_extrema_parity(
    tmp_path: Path,
    native_encoder,
) -> None:
    maximum_tick = 2**63 - 1
    native, expected, jsonl, compressed = _old_and_native(
        tmp_path,
        loaded_extension=native_encoder,
        world_uids=(2**64 - 1,),
        schedules=((maximum_tick,),),
        absolute_tick=maximum_tick,
        agents_by_world=([(511, 2**64 - 1, True)],),
        catalog_tick=maximum_tick,
        extrema=True,
    )
    _assert_exact_v1_parity(native, expected, jsonl, compressed)
    assert b'"energy":-32768' in jsonl
    assert b'"health":32767' in jsonl
    assert b'"action_0_count":-9223372036854775808' in jsonl
    assert b'"action_1_count":9223372036854775807' in jsonl


def test_native_encoder_rejects_duplicate_identity_and_invalid_boundary(
    native_encoder,
) -> None:
    cells = _world_cells((101,), ((4,),))
    plan = compile_cuda_snapshot_plan_v1(
        world_uids=(101,),
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SET_SHA256,
        execution_publication_context_sha256=PUBLICATION_CONTEXT_SHA256,
        catalog_absolute_tick=None,
    )
    native_plan = compile_native_snapshot_encoder_plan_v1(
        plan, loaded_extension=native_encoder
    )
    state_bytes = _world_state(
        world_uid=101,
        absolute_tick=4,
        agents=[(0, 77, True), (1, 77, False)],
    )
    state = torch.frombuffer(bytearray(state_bytes), dtype=torch.uint8).reshape(
        1, capture_adapter._STATE_BYTES
    )
    metrics = _metrics(1, ([(0, 77, True), (1, 77, False)],))
    projection = _metrics_projection(
        metrics, world_uids=(101,), absolute_tick=4
    )
    with pytest.raises(RuntimeError, match="duplicate agent identity 77"):
        native_plan.encode(state, projection, 4)
    with pytest.raises(RuntimeError, match="not declared by the bound plan"):
        future = {**projection, "next_absolute_tick": 5}
        native_plan.encode(state, future, 5)
    with pytest.raises(RuntimeError, match="CPU-contiguous uint8"):
        native_plan.encode(state.to(torch.int16), projection, 4)


def test_native_encoder_cold_bind_rejects_mismatched_plan_mapping(
    native_encoder,
) -> None:
    cells = _world_cells((101,), ((4,),))
    plan = compile_cuda_snapshot_plan_v1(
        world_uids=(101,),
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SET_SHA256,
        execution_publication_context_sha256=PUBLICATION_CONTEXT_SHA256,
        catalog_absolute_tick=None,
    )
    mismatched = plan.to_mapping()
    mismatched["execution_publication_context_sha256"] = "5" * 64
    with pytest.raises(RuntimeError, match="canonical plan bytes differ"):
        native_encoder.module.compile_snapshot_encoder_plan_v1(
            plan.canonical_plan_bytes,
            mismatched,
        )

    transport_mismatch = plan.to_mapping()
    transport_mismatch["state_capture_transport"]["accepted_modes"][0][
        "world_stride_bytes"
    ] += 1
    transport_mismatch_bytes = json.dumps(
        transport_mismatch,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    with pytest.raises(RuntimeError, match="capture mode geometry differs"):
        native_encoder.module.compile_snapshot_encoder_plan_v1(
            transport_mismatch_bytes,
            transport_mismatch,
        )


def test_native_encoder_validates_projection_and_selected_state_frontiers(
    tmp_path: Path,
    native_encoder,
) -> None:
    _plan, native_plan, state, metrics, projection = _native_case(
        native_encoder,
        world_uids=(202, 101),
        schedules=((9,), (9,)),
        absolute_tick=9,
        agents_by_world=([(0, 20, True)], [(1, 30, True)]),
    )

    wrong_frontier = {**projection, "next_absolute_tick": 8}
    with pytest.raises(RuntimeError, match="frontier differs"):
        encode_native_snapshot_v1(
            native_plan,
            state=state,
            active_metrics_projection=wrong_frontier,
            absolute_tick=9,
        )

    wrong_world_order = {**projection, "world_uids": [101, 202]}
    with pytest.raises(RuntimeError, match="world order differs"):
        encode_native_snapshot_v1(
            native_plan,
            state=state,
            active_metrics_projection=wrong_world_order,
            absolute_tick=9,
        )

    wrong_digest = {**projection, "content_sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="content digest differs"):
        encode_native_snapshot_v1(
            native_plan,
            state=state,
            active_metrics_projection=wrong_digest,
            absolute_tick=9,
        )

    swapped_state = state.flip(0).contiguous()
    with pytest.raises(RuntimeError, match="header world identity differs"):
        encode_native_snapshot_v1(
            native_plan,
            state=swapped_state,
            active_metrics_projection=projection,
            absolute_tick=9,
        )
    swapped_owned_stage = (tmp_path / "swapped-owned.stage").resolve()
    with pytest.raises(RuntimeError, match="header world identity differs"):
        encode_owned_native_snapshot_to_stage_file_v1(
            native_plan,
            state=swapped_state,
            metrics=metrics,
            absolute_tick=9,
            stage_path=swapped_owned_stage,
            encoding_policy=NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
        )
    assert not swapped_owned_stage.exists()

    wrong_header_bytes = bytearray(state.numpy().tobytes(order="C"))
    _pack_field(
        wrong_header_bytes,
        "scalar_header_counters",
        "<Q",
        1,
        8,
    )
    wrong_header = torch.frombuffer(
        wrong_header_bytes, dtype=torch.uint8
    ).reshape_as(state)
    with pytest.raises(RuntimeError, match="header frontier differs"):
        encode_native_snapshot_v1(
            native_plan,
            state=wrong_header,
            active_metrics_projection=projection,
            absolute_tick=9,
        )
    wrong_frontier_owned_stage = (tmp_path / "wrong-frontier-owned.stage").resolve()
    with pytest.raises(RuntimeError, match="header frontier differs"):
        encode_owned_native_snapshot_to_stage_file_v1(
            native_plan,
            state=wrong_header,
            metrics=metrics,
            absolute_tick=9,
            stage_path=wrong_frontier_owned_stage,
            encoding_policy=NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
        )
    assert not wrong_frontier_owned_stage.exists()

    assert metrics.shape == projection["tensor"].shape


def test_native_encoder_unicode_control_cell_and_catalog_dedup_parity(
    tmp_path: Path,
    native_encoder,
) -> None:
    repeated_genome_delta = 101 * 31 * 1000
    native, expected, jsonl, compressed = _old_and_native(
        tmp_path,
        loaded_extension=native_encoder,
        world_uids=(101,),
        schedules=((0,),),
        absolute_tick=0,
        agents_by_world=(
            [
                (2, 1, True),
                (7, 1 + repeated_genome_delta, True),
            ],
        ),
        catalog_tick=0,
        cell_ids=('cell-"\\\n\t\u2603-\x01',),
    )
    _assert_exact_v1_parity(native, expected, jsonl, compressed)
    rows = [json.loads(line) for line in jsonl.splitlines()]
    assert sum(row["row_kind"] == "genome_catalog" for row in rows) == 1
    assert native["row_count"] == 4
    assert b'"cell_id":"cell-\\\"\\\\\\n\\t\\u2603-\\u0001"' in jsonl


def test_native_encoder_tensor_contract_bool_tick_and_determinism(
    native_encoder,
) -> None:
    _plan, native_plan, state, metrics, projection = _native_case(
        native_encoder,
        world_uids=(101,),
        schedules=((4,),),
        absolute_tick=4,
        agents_by_world=([(2, 7, True)],),
    )
    first = encode_native_snapshot_v1(
        native_plan,
        state=state,
        active_metrics_projection=projection,
        absolute_tick=4,
    )
    second = encode_native_snapshot_v1(
        native_plan,
        state=state,
        active_metrics_projection=projection,
        absolute_tick=4,
    )
    assert first["canonical_jsonl"] == second["canonical_jsonl"]
    assert first["deterministic_gzip_level_9"] == second[
        "deterministic_gzip_level_9"
    ]
    assert first["compressed_file_sha256"] == second["compressed_file_sha256"]
    assert first["state_version_guard_enabled"] is True
    assert first["metrics_version_guard_enabled"] is True
    assert first["state_input_version"] == state._version
    assert first["metrics_input_version"] == metrics._version
    with pytest.raises(ValueError, match="exact integer"):
        encode_native_snapshot_v1(
            native_plan,
            state=state,
            active_metrics_projection=projection,
            absolute_tick=True,
        )

    wrong_state_shape = state[:, :-1].contiguous()
    with pytest.raises(RuntimeError, match="CPU-contiguous uint8"):
        encode_native_snapshot_v1(
            native_plan,
            state=wrong_state_shape,
            active_metrics_projection=projection,
            absolute_tick=4,
        )
    wrong_state_dtype = state.to(torch.int16)
    with pytest.raises(RuntimeError, match="CPU-contiguous uint8"):
        encode_native_snapshot_v1(
            native_plan,
            state=wrong_state_dtype,
            active_metrics_projection=projection,
            absolute_tick=4,
        )
    state_backing = torch.empty(
        (1, capture_adapter._STATE_BYTES * 2), dtype=torch.uint8
    )
    noncontiguous_state = state_backing[:, ::2]
    noncontiguous_state.copy_(state)
    assert not noncontiguous_state.is_contiguous()
    with pytest.raises(RuntimeError, match="CPU-contiguous uint8"):
        encode_native_snapshot_v1(
            native_plan,
            state=noncontiguous_state,
            active_metrics_projection=projection,
            absolute_tick=4,
        )

    wrong_metrics_dtype = metrics.to(torch.int32)
    wrong_dtype_projection = {
        **projection,
        "tensor": wrong_metrics_dtype,
        "shape": list(wrong_metrics_dtype.shape),
    }
    with pytest.raises(RuntimeError, match="CPU-contiguous int64"):
        encode_native_snapshot_v1(
            native_plan,
            state=state,
            active_metrics_projection=wrong_dtype_projection,
            absolute_tick=4,
        )
    wrong_metrics_shape = metrics[:, :, :-1].contiguous()
    wrong_shape_projection = {
        **projection,
        "tensor": wrong_metrics_shape,
        "shape": list(wrong_metrics_shape.shape),
    }
    with pytest.raises(RuntimeError, match="declared shape differs"):
        encode_native_snapshot_v1(
            native_plan,
            state=state,
            active_metrics_projection=wrong_shape_projection,
            absolute_tick=4,
        )
    metrics_backing = torch.empty(
        (
            1,
            capture_adapter._CAPACITY,
            len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1) * 2,
        ),
        dtype=torch.int64,
    )
    noncontiguous_metrics = metrics_backing[:, :, ::2]
    noncontiguous_metrics.copy_(metrics)
    assert not noncontiguous_metrics.is_contiguous()
    noncontiguous_projection = _metrics_projection(
        noncontiguous_metrics,
        world_uids=(101,),
        absolute_tick=4,
    )
    with pytest.raises(RuntimeError, match="CPU-contiguous int64"):
        encode_native_snapshot_v1(
            native_plan,
            state=state,
            active_metrics_projection=noncontiguous_projection,
            absolute_tick=4,
        )


def _copy_encoder_artifact(tmp_path: Path, native_encoder):
    root = tmp_path / "sealed-copy"
    root.mkdir(parents=True)
    artifact = root / native_encoder.artifact_path.name
    manifest = artifact.with_suffix(artifact.suffix + ".manifest.json")
    shutil.copy2(native_encoder.artifact_path, artifact)
    shutil.copy2(native_encoder.manifest_path, manifest)
    return artifact, manifest


def _rewrite_manifest_with_digest(path: Path, value: dict[str, object]) -> None:
    body = {key: child for key, child in value.items() if key != "manifest_sha256"}
    value["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path.chmod(0o644)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def test_native_encoder_loader_rejects_artifact_manifest_and_source_tamper(
    tmp_path: Path,
    native_encoder,
) -> None:
    artifact, _manifest = _copy_encoder_artifact(tmp_path / "artifact", native_encoder)
    artifact.chmod(0o644)
    payload = bytearray(artifact.read_bytes())
    payload[-1] ^= 0x01
    artifact.write_bytes(payload)
    with pytest.raises(NativeSnapshotEncoderLoadError, match="artifact digest"):
        validate_native_snapshot_encoder_manifest_v1(artifact)

    artifact, manifest = _copy_encoder_artifact(tmp_path / "manifest", native_encoder)
    decoded = json.loads(manifest.read_text())
    decoded["supports_cuda"] = True
    manifest.chmod(0o644)
    manifest.write_text(json.dumps(decoded, sort_keys=True, indent=2) + "\n")
    with pytest.raises(NativeSnapshotEncoderLoadError, match="build identity"):
        validate_native_snapshot_encoder_manifest_v1(artifact)

    artifact, manifest = _copy_encoder_artifact(tmp_path / "source", native_encoder)
    decoded = json.loads(manifest.read_text())
    decoded["source_inventory"][0]["sha256"] = "0" * 64
    decoded["source_inventory_sha256"] = hashlib.sha256(
        json.dumps(
            decoded["source_inventory"], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    _rewrite_manifest_with_digest(manifest, decoded)
    with pytest.raises(NativeSnapshotEncoderLoadError, match="bound source differs"):
        validate_native_snapshot_encoder_manifest_v1(artifact)


def test_native_encoder_loader_rejects_artifact_symlink(
    tmp_path: Path,
    native_encoder,
) -> None:
    artifact_link = tmp_path / native_encoder.artifact_path.name
    artifact_link.symlink_to(native_encoder.artifact_path)
    with pytest.raises(NativeSnapshotEncoderLoadError, match="symlink"):
        validate_native_snapshot_encoder_manifest_v1(artifact_link)


def test_native_encoder_binds_exact_zlib_runtime(native_encoder) -> None:
    zlib_identity = native_encoder.manifest["dependencies"]["zlib"]
    assert native_encoder.module.zlib_compile_version() == zlib_identity[
        "compile_header_version"
    ]
    assert native_encoder.module.zlib_runtime_version() == zlib_identity[
        "runtime_version"
    ]
    assert zlib_identity["compile_header_version"] == zlib_identity[
        "runtime_version"
    ]
    assert native_encoder.manifest["filesystem_capability"] == {
        "stage_file_output": True,
        "stage_file_open_policy": "EXCLUSIVE_CREATE_NO_OVERWRITE_V1",
        "stage_file_durability": "FILE_FSYNC_AND_CLOSE_V1",
        "owned_tensor_stage_method": True,
        "owned_tensor_native_work_releases_gil": True,
        "atomic_final_publication": False,
        "directory_fsync": False,
    }


@pytest.mark.parametrize("distribution", ("zero", "all", "sparse"))
def test_native_stage_file_exact_v1_parity_without_python_payloads(
    tmp_path: Path,
    native_encoder,
    distribution: str,
) -> None:
    if distribution == "zero":
        agents = [(9, 91, False), (2, 22, False)]
    elif distribution == "all":
        agents = [
            (slot, capture_adapter._CAPACITY - slot, True)
            for slot in range(capture_adapter._CAPACITY)
        ]
    else:
        agents = [(7, 30, True), (2, 10, False), (12, 20, True)]
    _memory, expected, legacy_jsonl, legacy_gzip = _old_and_native(
        tmp_path,
        loaded_extension=native_encoder,
        world_uids=(101,),
        schedules=((13,),),
        absolute_tick=13,
        agents_by_world=(agents,),
        catalog_tick=None,
    )
    plan, native_plan, state, metrics_tensor, projection = _native_case(
        native_encoder,
        world_uids=(101,),
        schedules=((13,),),
        absolute_tick=13,
        agents_by_world=(agents,),
    )
    stage_path = (tmp_path / f"{distribution}.snapshot.jsonl.gz.stage").resolve()
    descriptor = encode_native_snapshot_to_stage_file_v1(
        native_plan,
        state=state,
        active_metrics_projection=projection,
        absolute_tick=13,
        stage_path=stage_path,
    )
    staged = stage_path.read_bytes()
    assert staged == legacy_gzip
    assert gzip.decompress(staged) == legacy_jsonl
    assert descriptor["schema_version"] == NATIVE_SNAPSHOT_STAGE_FILE_RECEIPT_SCHEMA_V1
    assert descriptor["stage_path"] == str(stage_path)
    assert descriptor["stage_encoding_policy"] == (
        NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1
    )
    assert descriptor["stage_file_state"] == "FSYNCED_STAGE_ONLY_NOT_PUBLISHED"
    assert descriptor["stage_file_fsync_completed"] is True
    assert descriptor["stage_file_close_completed"] is True
    assert descriptor["atomic_publication_performed"] is False
    assert descriptor["directory_fsync_performed"] is False
    assert descriptor["stage_bytes_written"] == len(staged)
    assert descriptor["compressed_bytes"] == expected["compressed_bytes"]
    assert descriptor["compressed_file_sha256"] == expected[
        "compressed_file_sha256"
    ]
    assert descriptor["uncompressed_content_sha256"] == expected[
        "uncompressed_content_sha256"
    ]
    assert descriptor["row_count"] == expected["row_count"]
    assert "canonical_jsonl" not in descriptor
    assert "deterministic_gzip_level_9" not in descriptor
    assert stat.S_IMODE(stage_path.stat().st_mode) == 0o600
    timing = descriptor["timing"]
    for field in (
        "core_encode_compress_hash_ns",
        "native_stage_file_open_ns",
        "native_stage_file_write_ns",
        "native_stage_file_fsync_and_fstat_ns",
        "native_stage_file_close_ns",
        "native_encode_write_fsync_close_ns",
        "external_full_call_elapsed_ns",
    ):
        assert type(timing[field]) is int and timing[field] >= 0
    assert timing["native_encode_write_fsync_close_ns"] >= timing[
        "core_encode_compress_hash_ns"
    ]
    assert timing["external_full_call_elapsed_ns"] >= timing[
        "native_encode_write_fsync_close_ns"
    ]

    owned_stage_path = (
        tmp_path / f"{distribution}.owned.snapshot.jsonl.gz.stage"
    ).resolve()
    owned = encode_owned_native_snapshot_to_stage_file_v1(
        native_plan,
        state=state,
        metrics=metrics_tensor,
        absolute_tick=13,
        stage_path=owned_stage_path,
        encoding_policy=NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
    )
    assert owned_stage_path.read_bytes() == legacy_gzip
    assert owned["compressed_file_sha256"] == expected["compressed_file_sha256"]
    assert owned["uncompressed_content_sha256"] == expected[
        "uncompressed_content_sha256"
    ]
    assert owned["active_metrics_content_sha256"] == projection["content_sha256"]
    assert owned["active_metrics_authentication"] == (
        "computed_native_from_plan_bound_resident_tensor_v1"
    )
    assert owned["input_ownership_contract"] == (
        "native_service_exclusive_tensor_lease_until_completion_v1"
    )
    assert owned["gil_held_for_full_synchronous_call"] is False
    assert owned["state_input_version"] == state._version
    assert owned["state_completion_version"] == state._version
    assert owned["metrics_input_version"] == metrics_tensor._version
    assert owned["metrics_completion_version"] == metrics_tensor._version
    assert owned["input_versions_unchanged"] is True
    assert "canonical_jsonl" not in owned
    assert "deterministic_gzip_level_9" not in owned
    assert owned["timing"]["active_metrics_content_hashing_ns"] >= 0
    assert owned["state_capture_transport_schema_version"] == (
        NATIVE_SNAPSHOT_STATE_CAPTURE_TRANSPORT_SCHEMA_V1
    )
    assert owned["state_capture_mode"] == (
        NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1
    )
    assert owned["state_capture_preferred_mode"] == (
        NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1
    )
    assert owned["state_capture_mode_is_preferred"] is False

    canonical_row_bytes = int(
        plan.to_mapping()["layout"]["canonical_state_bytes_per_world"]
    )
    canonical_state = state[:, :canonical_row_bytes].contiguous()
    canonical_stage_path = (
        tmp_path / f"{distribution}.canonical.owned.snapshot.jsonl.gz.stage"
    ).resolve()
    canonical_owned = encode_owned_native_snapshot_to_stage_file_v1(
        native_plan,
        state=canonical_state,
        metrics=metrics_tensor,
        absolute_tick=13,
        stage_path=canonical_stage_path,
        encoding_policy=NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
    )
    assert canonical_stage_path.read_bytes() == legacy_gzip
    assert canonical_owned["compressed_file_sha256"] == owned[
        "compressed_file_sha256"
    ]
    assert canonical_owned["uncompressed_content_sha256"] == owned[
        "uncompressed_content_sha256"
    ]
    assert canonical_owned["state_input_row_bytes"] == canonical_row_bytes
    assert canonical_owned["state_input_bytes"] == canonical_state.numel()
    assert canonical_owned["canonical_state_bytes_selected"] == (
        canonical_state.numel()
    )
    assert canonical_owned["resident_state_row_bytes"] == state.shape[1]
    assert canonical_owned["state_capture_mode"] == (
        NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1
    )
    assert canonical_owned["state_capture_preferred_mode"] == (
        NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1
    )
    assert canonical_owned["state_capture_mode_is_preferred"] is True
    assert owned["state_input_row_bytes"] == state.shape[1]
    assert owned["state_input_bytes"] == state.numel()
    assert owned["resident_state_row_bytes"] == state.shape[1]


def test_native_encoder_plan_bound_canonical_transport_uses_compact_world_stride(
    native_encoder,
) -> None:
    plan, native_plan, state, _metrics_tensor, projection = _native_case(
        native_encoder,
        world_uids=(2**64 - 1, 202, 101),
        schedules=((13,), (17,), (13, 17)),
        absolute_tick=13,
        agents_by_world=(
            [(0, 91, True)],
            [(1, 7, False)],
            [(511, 99, True)],
        ),
    )
    canonical_row_bytes = int(
        plan.to_mapping()["layout"]["canonical_state_bytes_per_world"]
    )
    state[:, canonical_row_bytes:].fill_(0xA5)
    canonical_state = state[:, :canonical_row_bytes].contiguous()

    full = encode_native_snapshot_v1(
        native_plan,
        state=state,
        active_metrics_projection=projection,
        absolute_tick=13,
    )
    canonical = encode_native_snapshot_v1(
        native_plan,
        state=canonical_state,
        active_metrics_projection=projection,
        absolute_tick=13,
    )

    assert canonical["canonical_jsonl"] == full["canonical_jsonl"]
    assert canonical["deterministic_gzip_level_9"] == full[
        "deterministic_gzip_level_9"
    ]
    assert canonical["compressed_file_sha256"] == full["compressed_file_sha256"]
    assert canonical["selected_world_indices"] == [0, 2]
    assert canonical["selected_world_uids"] == [2**64 - 1, 101]
    assert canonical["canonical_state_bytes_selected"] == 2 * canonical_row_bytes
    assert canonical["state_input_row_bytes"] == canonical_row_bytes
    assert canonical["state_input_bytes"] == 3 * canonical_row_bytes
    assert canonical["state_capture_mode"] == (
        NATIVE_SNAPSHOT_STATE_CAPTURE_CANONICAL_PREFIX_V1
    )
    assert canonical["state_capture_mode_is_preferred"] is True
    assert full["state_capture_mode"] == (
        NATIVE_SNAPSHOT_STATE_CAPTURE_FULL_RESIDENT_V1
    )
    assert full["state_capture_mode_is_preferred"] is False


def test_native_stage_file_is_deterministic_and_exclusive(
    tmp_path: Path,
    native_encoder,
) -> None:
    _plan, native_plan, state, _metrics_tensor, projection = _native_case(
        native_encoder,
        world_uids=(202, 101),
        schedules=((0,), (0,)),
        absolute_tick=0,
        agents_by_world=(
            [(7, 30, True), (12, 20, True)],
            [(3, 5, True)],
        ),
        catalog_tick=0,
    )
    paths = [
        (tmp_path / "repeat-a.snapshot.jsonl.gz.stage").resolve(),
        (tmp_path / "repeat-b.snapshot.jsonl.gz.stage").resolve(),
    ]
    descriptors = [
        encode_native_snapshot_to_stage_file_v1(
            native_plan,
            state=state,
            active_metrics_projection=projection,
            absolute_tick=0,
            stage_path=path,
        )
        for path in paths
    ]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert descriptors[0]["compressed_file_sha256"] == descriptors[1][
        "compressed_file_sha256"
    ]
    original = paths[0].read_bytes()
    with pytest.raises(RuntimeError, match="exclusive create failed"):
        encode_native_snapshot_to_stage_file_v1(
            native_plan,
            state=state,
            active_metrics_projection=projection,
            absolute_tick=0,
            stage_path=paths[0],
        )
    assert paths[0].read_bytes() == original


def test_owned_native_stage_file_releases_gil_and_guards_tensor_versions(
    tmp_path: Path,
    native_encoder,
) -> None:
    world_uids = (101, 202, 303, 404)
    all_alive = [
        (slot, capture_adapter._CAPACITY - slot, True)
        for slot in range(capture_adapter._CAPACITY)
    ]
    _plan, native_plan, state, metrics, projection = _native_case(
        native_encoder,
        world_uids=world_uids,
        schedules=((23,),) * len(world_uids),
        absolute_tick=23,
        agents_by_world=tuple(all_alive.copy() for _ in world_uids),
    )
    stop = threading.Event()
    started = threading.Event()
    progress = [0]

    def advance_python() -> None:
        started.set()
        while not stop.is_set():
            progress[0] += 1

    worker = threading.Thread(target=advance_python)
    worker.start()
    assert started.wait(timeout=2)
    before = progress[0]
    try:
        descriptor = encode_owned_native_snapshot_to_stage_file_v1(
            native_plan,
            state=state,
            metrics=metrics,
            absolute_tick=23,
            stage_path=(tmp_path / "gil-release.snapshot.stage").resolve(),
            encoding_policy=NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
        )
        after = progress[0]
    finally:
        stop.set()
        worker.join(timeout=2)
    assert not worker.is_alive()
    assert after > before
    assert descriptor["active_metrics_content_sha256"] == projection[
        "content_sha256"
    ]
    assert descriptor["state_input_version"] == descriptor[
        "state_completion_version"
    ]
    assert descriptor["metrics_input_version"] == descriptor[
        "metrics_completion_version"
    ]
    assert descriptor["gil_held_for_full_synchronous_call"] is False


def test_native_stage_file_failures_leave_no_completion_descriptor(
    tmp_path: Path,
    native_encoder,
) -> None:
    _plan, native_plan, state, metrics_tensor, projection = _native_case(
        native_encoder,
        world_uids=(101,),
        schedules=((4,),),
        absolute_tick=4,
        agents_by_world=([(2, 7, True)],),
    )
    missing_parent_stage = (
        tmp_path / "missing-parent" / "snapshot.jsonl.gz.stage"
    ).resolve()
    with pytest.raises(RuntimeError, match="exclusive create failed"):
        encode_native_snapshot_to_stage_file_v1(
            native_plan,
            state=state,
            active_metrics_projection=projection,
            absolute_tick=4,
            stage_path=missing_parent_stage,
        )
    assert not missing_parent_stage.exists()

    partial_stage = (tmp_path / "partial.snapshot.jsonl.gz.stage").resolve()
    with pytest.raises(RuntimeError, match="injected.*write failure"):
        encode_native_snapshot_to_stage_file_v1(
            native_plan,
            state=state,
            active_metrics_projection=projection,
            absolute_tick=4,
            stage_path=partial_stage,
            _test_fail_after_compressed_bytes=64,
        )
    assert partial_stage.exists()
    assert partial_stage.stat().st_size == 64
    with pytest.raises((EOFError, gzip.BadGzipFile)):
        gzip.decompress(partial_stage.read_bytes())
    partial_stage.unlink()

    unsupported_stage = (tmp_path / "unsupported.stage").resolve()
    with pytest.raises(ValueError, match="policy is unsupported"):
        encode_native_snapshot_to_stage_file_v1(
            native_plan,
            state=state,
            active_metrics_projection=projection,
            absolute_tick=4,
            stage_path=unsupported_stage,
            encoding_policy="ORDERED_MULTI_MEMBER_PARALLEL_V1",
        )
    assert not unsupported_stage.exists()
    with pytest.raises(ValueError, match="absolute filesystem path"):
        encode_native_snapshot_to_stage_file_v1(
            native_plan,
            state=state,
            active_metrics_projection=projection,
            absolute_tick=4,
            stage_path="relative.snapshot.stage",
        )

    owned_invalid = (tmp_path / "owned-invalid.snapshot.stage").resolve()
    with pytest.raises(ValueError, match="exact integer"):
        encode_owned_native_snapshot_to_stage_file_v1(
            native_plan,
            state=state,
            metrics=metrics_tensor,
            absolute_tick=True,
            stage_path=owned_invalid,
            encoding_policy=NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
        )
    assert not owned_invalid.exists()
    with pytest.raises(RuntimeError, match="CPU-contiguous int64 projection"):
        encode_owned_native_snapshot_to_stage_file_v1(
            native_plan,
            state=state,
            metrics=metrics_tensor[:, :, :-1].contiguous(),
            absolute_tick=4,
            stage_path=owned_invalid,
            encoding_policy=NATIVE_SNAPSHOT_STAGE_ENCODING_POLICY_V1,
        )
    assert not owned_invalid.exists()
