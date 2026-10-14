from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct

import pytest
import torch

import engine.native_runtime.cuda_v2_science_capture_v1 as capture_adapter
from engine.campaigns.cuda_full_evolution_v2_provider import (
    COMPACT_SCIENCE_SNAPSHOT_TIMING_ENV_V1,
    _compact_snapshot_timing_enabled_v1,
    _compact_snapshot_timing_result_fields_v1,
    _compact_snapshot_timing_summary_v1,
)
from engine.campaigns.providers import CampaignProviderError
from engine.native_runtime.cuda_v2_compact_science_capture_v1 import (
    canonical_sha256,
    write_compact_science_snapshot_v1,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
)


CONTROL_SET_SHA256 = "3" * 64
SNAPSHOT_SCHEMA_V1 = "eiel_cuda_v2_compact_science_snapshot_artifact_v1"


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
) -> bytes:
    """Build one exact native-layout state without invoking snapshot decoding."""

    raw = bytearray(capture_adapter._STATE_BYTES)
    _pack_field(raw, "scalar_header_counters", "<Q", 0, world_uid)
    _pack_field(raw, "scalar_header_counters", "<Q", 1, absolute_tick)
    _pack_field(
        raw,
        "scalar_header_counters",
        "<Q",
        3,
        sum(alive for _, _, alive in agents),
    )
    _pack_field(raw, "scalar_header_counters", "<Q", 15, 17)
    _pack_field(raw, "scalar_header_counters", "<Q", 16, 5)
    _pack_field(raw, "scalar_header_counters", "<Q", 17, 3)
    _pack_field(raw, "scalar_header_counters", "<Q", 18, 2)
    for index, value in enumerate((7, 5, 3, 1)):
        _pack_field(raw, "resource", "<H", index, value + world_uid % 3)

    alive_words = [0] * capture_adapter._ALIVE_WORDS
    policy_offset = capture_adapter._field_offset("policy_alleles")
    trait_offset = capture_adapter._field_offset("trait_alleles")
    for slot, agent_id, alive in agents:
        assert 0 <= slot < capture_adapter._CAPACITY
        if alive:
            alive_words[slot // 32] |= 1 << (slot % 32)
        _pack_field(raw, "agent_id", "<Q", slot, agent_id)
        _pack_field(raw, "parent_a_id", "<Q", slot, agent_id // 2)
        _pack_field(raw, "parent_b_id", "<Q", slot, 0)
        _pack_field(raw, "position", "<H", slot, (slot * 7) % 1024)
        _pack_field(raw, "energy", "<h", slot, 100 - slot)
        _pack_field(raw, "health", "<h", slot, 80 - slot)
        _pack_field(raw, "age", "<H", slot, slot + 1)
        _pack_field(
            raw,
            "absolute_birth_tick",
            "<Q",
            slot,
            max(0, absolute_tick - slot),
        )

        # Native allele storage is [haplotype, locus, slot].  Two nonzero
        # values make each fixture genome deterministic and agent-specific.
        raw[policy_offset + slot] = (agent_id % 101) & 0xFF
        raw[policy_offset + capture_adapter._CAPACITY + slot] = (
            -(agent_id % 31)
        ) & 0xFF
        struct.pack_into(
            "<h",
            raw,
            trait_offset + 2 * slot,
            -(agent_id % 1000),
        )
        struct.pack_into(
            "<h",
            raw,
            trait_offset + 2 * (capture_adapter._CAPACITY + slot),
            agent_id % 1000,
        )
    for index, value in enumerate(alive_words):
        _pack_field(raw, "alive_words", "<I", index, value)
    return bytes(raw)


def _metrics_projection(
    *,
    world_uids: tuple[int, ...],
    absolute_tick: int,
    agents_by_world: tuple[list[tuple[int, int, bool]], ...],
) -> dict[str, object]:
    tensor = torch.zeros(
        (
            len(world_uids),
            capture_adapter._CAPACITY,
            len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
        ),
        dtype=torch.int64,
    )
    for world_index, agents in enumerate(agents_by_world):
        for slot, agent_id, _alive in agents:
            tensor[world_index, slot, 0] = agent_id * 10
            tensor[world_index, slot, -1] = slot + 1000
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
    *, world_uids: tuple[int, ...], snapshot_ticks: tuple[int, ...]
) -> dict[str, dict[str, object]]:
    # Reverse insertion order deliberately: snapshot order is the native
    # projection's world order, not Python mapping insertion order.
    return {
        f"cell-{world_uid}": {
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
            "science_acquisition": {"snapshot_ticks": list(snapshot_ticks)},
        }
        for world_uid in reversed(world_uids)
    }


def _write_snapshot(
    root: Path,
    *,
    world_uids: tuple[int, ...],
    absolute_tick: int,
    agents_by_world: tuple[list[tuple[int, int, bool]], ...],
    include_genome_catalog: bool,
    timing_diagnostics: dict[str, object] | None = None,
    state_as_tensor: bool = False,
) -> dict[str, object]:
    state = b"".join(
        _world_state(
            world_uid=world_uid,
            absolute_tick=absolute_tick,
            agents=agents,
        )
        for world_uid, agents in zip(
            world_uids, agents_by_world, strict=True
        )
    )
    return write_compact_science_snapshot_v1(
        root,
        state=(
            torch.frombuffer(bytearray(state), dtype=torch.uint8)
            if state_as_tensor
            else state
        ),
        active_metrics_projection=_metrics_projection(
            world_uids=world_uids,
            absolute_tick=absolute_tick,
            agents_by_world=agents_by_world,
        ),
        world_cells=_world_cells(
            world_uids=world_uids,
            snapshot_ticks=(0, absolute_tick),
        ),
        absolute_tick=absolute_tick,
        native_world_control_set_sha256=CONTROL_SET_SHA256,
        include_genome_catalog=include_genome_catalog,
        timing_diagnostics=timing_diagnostics,
    )


def _assert_canonical_artifact(
    receipt: dict[str, object],
) -> tuple[bytes, list[dict[str, object]]]:
    root = Path(str(receipt["path"]))
    compressed = (root / "snapshot.jsonl.gz").read_bytes()
    content = gzip.decompress(compressed)
    rows = [json.loads(line) for line in content.splitlines()]
    canonical = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    assert content == canonical
    assert hashlib.sha256(content).hexdigest() == receipt[
        "uncompressed_content_sha256"
    ]
    assert hashlib.sha256(compressed).hexdigest() == receipt[
        "compressed_file_sha256"
    ]
    assert len(compressed) == receipt["compressed_bytes"]

    manifest = {
        key: value
        for key, value in receipt.items()
        if key not in {"path", "timing_diagnostics"}
    }
    body = {
        key: value for key, value in manifest.items() if key != "artifact_sha256"
    }
    assert manifest["artifact_sha256"] == canonical_sha256(body)
    assert json.loads((root / "manifest.json").read_bytes()) == manifest
    assert (root / "manifest.json").read_bytes() == (
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    return content, rows


@pytest.mark.parametrize("distribution", ("zero", "all", "sparse"))
def test_snapshot_v1_freezes_alive_filtering_and_agent_order(
    tmp_path: Path, distribution: str
) -> None:
    if distribution == "zero":
        agents = [(9, 91, False), (2, 22, False)]
        expected_agent_ids: list[int] = []
    elif distribution == "all":
        agents = [
            (slot, capture_adapter._CAPACITY - slot, True)
            for slot in range(capture_adapter._CAPACITY)
        ]
        expected_agent_ids = list(range(1, capture_adapter._CAPACITY + 1))
    else:
        agents = [(7, 30, True), (2, 10, False), (12, 20, True)]
        expected_agent_ids = [20, 30]

    receipt = _write_snapshot(
        tmp_path / distribution,
        world_uids=(101,),
        absolute_tick=13,
        agents_by_world=(agents,),
        include_genome_catalog=False,
    )
    _content, rows = _assert_canonical_artifact(receipt)
    active_rows = [row for row in rows if row["row_kind"] == "active_agent_state"]
    assert [row["agent_id"] for row in active_rows] == expected_agent_ids
    assert all(row["alive"] is True for row in active_rows)
    assert receipt["row_count"] == 1 + len(expected_agent_ids)
    assert receipt["captured_worlds"][0]["active_agent_count"] == len(
        expected_agent_ids
    )
    assert receipt["genome_catalog_included"] is False


def test_snapshot_v1_freezes_world_order_catalog_policy_and_canonical_hashes(
    tmp_path: Path,
) -> None:
    world_uids = (202, 101)
    agents_by_world = (
        [(7, 30, True), (2, 10, False), (12, 20, True)],
        [(3, 5, True)],
    )
    initial = _write_snapshot(
        tmp_path / "initial",
        world_uids=world_uids,
        absolute_tick=0,
        agents_by_world=agents_by_world,
        include_genome_catalog=True,
    )
    initial_content, initial_rows = _assert_canonical_artifact(initial)
    assert hashlib.sha256(initial_content).hexdigest() == (
        "3098f670c0c687b9dc71edefc7834d872541d9cbe14d5eab8306f4d905a6f601"
    )
    assert [row["row_kind"] for row in initial_rows] == [
        "world_state",
        "active_agent_state",
        "active_agent_state",
        "genome_catalog",
        "genome_catalog",
        "world_state",
        "active_agent_state",
        "genome_catalog",
    ]
    assert [
        row["world_uid"]
        for row in initial_rows
        if row["row_kind"] == "world_state"
    ] == [202, 101]
    assert [
        row["agent_id"]
        for row in initial_rows
        if row["row_kind"] == "active_agent_state"
    ] == [20, 30, 5]
    assert [row["world_uid"] for row in initial["captured_worlds"]] == [202, 101]
    assert initial["schema_version"] == SNAPSHOT_SCHEMA_V1
    assert initial["absolute_tick"] == 0
    assert initial["row_count"] == 8
    assert initial["captured_world_count"] == 2
    assert initial["metric_fields"] == list(
        COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
    )
    assert initial["genome_catalog_included"] is True
    assert initial["data_file"] == "snapshot.jsonl.gz"
    assert initial["native_world_control_set_sha256"] == CONTROL_SET_SHA256
    assert initial["dropped_record_count"] == 0

    later = _write_snapshot(
        tmp_path / "later",
        world_uids=world_uids,
        absolute_tick=9,
        agents_by_world=agents_by_world,
        include_genome_catalog=False,
    )
    later_content, later_rows = _assert_canonical_artifact(later)
    assert hashlib.sha256(later_content).hexdigest() == (
        "7eee84de05482bac98b773daf6bf74e5c887d5c056f9dce812db10224e4f892f"
    )
    assert [row["row_kind"] for row in later_rows] == [
        "world_state",
        "active_agent_state",
        "active_agent_state",
        "world_state",
        "active_agent_state",
    ]
    assert later["absolute_tick"] == 9
    assert later["row_count"] == 5
    assert later["captured_world_count"] == 2
    assert later["genome_catalog_included"] is False


def test_snapshot_v1_rejects_duplicate_nonzero_agent_identity(
    tmp_path: Path,
) -> None:
    agents = [(0, 77, True), (1, 77, False)]
    with pytest.raises(ValueError, match="duplicate agent identity 77"):
        _write_snapshot(
            tmp_path / "duplicate",
            world_uids=(101,),
            absolute_tick=4,
            agents_by_world=(agents,),
            include_genome_catalog=False,
        )


def test_snapshot_v1_timing_is_default_off_and_byte_neutral(
    tmp_path: Path,
) -> None:
    world_uids = (202, 101)
    agents_by_world = (
        [(7, 30, True), (2, 10, False), (12, 20, True)],
        [(3, 5, True)],
    )
    ordinary = _write_snapshot(
        tmp_path / "ordinary",
        world_uids=world_uids,
        absolute_tick=9,
        agents_by_world=agents_by_world,
        include_genome_catalog=False,
    )
    diagnostics: dict[str, object] = {}
    instrumented = _write_snapshot(
        tmp_path / "instrumented",
        world_uids=world_uids,
        absolute_tick=9,
        agents_by_world=agents_by_world,
        include_genome_catalog=False,
        timing_diagnostics=diagnostics,
        state_as_tensor=True,
    )

    assert "timing_diagnostics" not in ordinary
    assert instrumented["timing_diagnostics"] == diagnostics
    assert diagnostics["success"] is True
    assert diagnostics["implementation"] == "python_v1"
    counters = diagnostics["counters"]
    assert counters["planned_world_count"] == 2
    assert counters["captured_world_count"] == 2
    assert counters["active_agent_count"] == 3
    assert counters["row_count"] == 5
    assert counters["state_host_bytes"] == 2 * capture_adapter._STATE_BYTES
    assert counters["metrics_input_bytes"] == (
        2
        * capture_adapter._CAPACITY
        * len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
        * 8
    )
    phases = diagnostics["phases"]
    for phase in (
        "state_d2h_transfer_and_wait",
        "state_host_contiguous",
        "state_host_bytes_conversion",
        "agent_decode_and_semantic_hashing",
        "metrics_tolist",
        "canonical_json_encoding_pass_1",
        "canonical_json_encoding_pass_2",
        "gzip_compression_write",
        "uncompressed_content_hashing",
        "compressed_content_hashing",
        "data_file_write",
        "data_file_flush",
        "data_file_fsync",
        "data_file_replace",
        "manifest_file_encode",
        "manifest_file_fsync",
        "writer_total",
    ):
        assert phases[phase]["wall_time_ns"] >= 0
        assert phases[phase]["cpu_time_ns"] >= 0
    assert phases["state_d2h_transfer_and_wait"]["source_device"] == "cpu"
    assert diagnostics["derived_accounting"] == {
        "categories_are_non_additive": True,
        "note": "derived duration categories can overlap and must not be summed",
    }
    assert all(type(value) is int for value in diagnostics["derived"].values())

    ordinary_root = Path(str(ordinary["path"]))
    instrumented_root = Path(str(instrumented["path"]))
    assert (ordinary_root / "snapshot.jsonl.gz").read_bytes() == (
        instrumented_root / "snapshot.jsonl.gz"
    ).read_bytes()
    assert (ordinary_root / "manifest.json").read_bytes() == (
        instrumented_root / "manifest.json"
    ).read_bytes()


def test_snapshot_timing_summary_uses_nearest_rank_and_exact_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(COMPACT_SCIENCE_SNAPSHOT_TIMING_ENV_V1, raising=False)
    assert _compact_snapshot_timing_enabled_v1() is False
    monkeypatch.setenv(COMPACT_SCIENCE_SNAPSHOT_TIMING_ENV_V1, "1")
    assert _compact_snapshot_timing_enabled_v1() is True
    monkeypatch.setenv(COMPACT_SCIENCE_SNAPSHOT_TIMING_ENV_V1, "yes")
    with pytest.raises(CampaignProviderError, match="must be exactly 0 or 1"):
        _compact_snapshot_timing_enabled_v1()

    samples = [
        {
            "producer_critical_path_time_ns": value,
            "full_durable_completion_time_ns": value,
            "publication_lag_time_ns": 0,
            "gpu_sync_host_wait_upper_bound_ns": value // 4,
            "evolution_producer_blocking_time_ns": value,
            "phases": {
                "writer": {
                    "wall_time_ns": value,
                    "cpu_time_ns": value // 2,
                    "cuda_event_time_ns": value // 5,
                }
            },
            "counters": {
                "row_count": value,
                "captured_world_count": 1,
                "active_agent_count": value,
                "d2h_bytes": value * 2,
                "uncompressed_bytes": value * 3,
                "compressed_bytes": value,
            },
        }
        for value in (10, 20, 30, 40, 50)
    ]
    summary = _compact_snapshot_timing_summary_v1(samples)
    distribution = summary["duration_distributions"][
        "producer_critical_path_time_ns"
    ]
    assert distribution == {
        "sample_count": 5,
        "minimum_ns": 10,
        "p50_ns": 30,
        "p95_ns": 50,
        "maximum_ns": 50,
        "sum_ns": 150,
        "mean_ns": 30.0,
    }
    assert summary["phase_distributions"]["writer"]["p95_ns"] == 50
    assert summary["phase_cpu_time_distributions"]["writer"]["p50_ns"] == 15
    assert summary["phase_cpu_time_distributions"]["writer"]["p95_ns"] == 25
    assert summary["phase_cuda_event_time_distributions"]["writer"]["p50_ns"] == 6
    assert summary["phase_cuda_event_time_distributions"]["writer"]["p95_ns"] == 10
    assert summary["aggregate_counters"]["row_count_sum"] == 150
    assert _compact_snapshot_timing_result_fields_v1(
        enabled=False,
        samples=samples,
    ) == {}
    assert set(
        _compact_snapshot_timing_result_fields_v1(
            enabled=True,
            samples=samples,
        )
    ) == {"compact_science_snapshot_timing"}
