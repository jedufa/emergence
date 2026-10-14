from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

import pytest
import torch

from engine.native_runtime.cuda_v2_compact_science_capture_v1 import (
    CompactScienceCaptureError,
    load_compact_science_checkpoint_frontier_v1,
    reconstruct_compact_birth_genome_v1,
    write_compact_science_checkpoint_frontier_v1,
    write_compact_science_dispatch_v1,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
    COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1,
    CudaFullEvolutionCompactScienceCaptureV1,
)
from engine.perf.cuda_full_evolution_v2.runtime import (
    CudaFullEvolutionSparseEventDispatchV3,
)


CONTROL_SET = "3" * 64


def test_birth_provenance_reconstructs_generic_child_genome_exactly() -> None:
    parent_a_policy = [10] * 112 + [20] * 112
    parent_a_policy[112 + 1] = 127
    parent_b_policy = [30] * 112 + [40] * 112
    parent_a_trait = [100] * 16 + [200] * 16
    parent_b_trait = [300] * 16 + [400] * 16
    provenance = bytearray(132)
    provenance[0] = 1 | (2 << 2)
    provenance[1] = 1
    provenance[128:132] = (1).to_bytes(4, "little")
    result = reconstruct_compact_birth_genome_v1(
        parent_a_genome={"policy": parent_a_policy, "trait": parent_a_trait},
        parent_b_genome={"policy": parent_b_policy, "trait": parent_b_trait},
        birth_provenance=provenance,
        mutation_step=3,
    )
    assert result["pre_mutation_genome"]["policy"][0] == 20
    assert result["pre_mutation_genome"]["policy"][112] == 30
    assert result["child_genome"]["policy"][0] == 23
    assert result["child_genome"]["policy"][112] == 27
    assert result["child_genome"]["policy"][1] == 127
    assert result["mutation_origin_count"] == 3
    assert result["saturated_noop_count"] == 1

    negative_edge = reconstruct_compact_birth_genome_v1(
        parent_a_genome={
            "policy": [-128] * 224,
            "trait": [-32768] * 32,
        },
        parent_b_genome={
            "policy": [-128] * 224,
            "trait": [-32768] * 32,
        },
        birth_provenance=bytes([2] + [0] * 127 + [0] * 4),
        mutation_step=3,
    )
    assert negative_edge["child_genome"]["policy"][0] == -127
    assert negative_edge["saturated_noop_count"] == 0


def _projection() -> dict[str, object]:
    tensor = torch.zeros((2, 512, 21), dtype=torch.int64)
    tensor[0, 7, 3] = 19
    tensor[1, 11, 20] = 23
    raw = tensor.numpy().astype("<i8", copy=False).tobytes(order="C")
    return {
        "schema_version": "eiel_full_evolution_cuda_active_science_metrics_v1",
        "next_absolute_tick": 40,
        "world_uids": [101, 202],
        "metric_fields": list(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1),
        "shape": [2, 512, 21],
        "content_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor": tensor,
    }


def test_checkpoint_frontier_round_trips_generic_world_order(tmp_path: Path) -> None:
    receipt = write_compact_science_checkpoint_frontier_v1(
        tmp_path / "frontier",
        active_metrics_projection=_projection(),
        absolute_tick=40,
        native_world_control_set_sha256=CONTROL_SET,
    )
    loaded = load_compact_science_checkpoint_frontier_v1(
        receipt,
        expected_root=tmp_path,
        expected_absolute_tick=40,
        expected_world_uids=(101, 202),
        expected_native_world_control_set_sha256=CONTROL_SET,
    )
    assert torch.equal(loaded["tensor"], _projection()["tensor"])
    assert loaded["world_uids"] == [101, 202]


def test_checkpoint_frontier_rejects_world_reownership(tmp_path: Path) -> None:
    receipt = write_compact_science_checkpoint_frontier_v1(
        tmp_path / "frontier",
        active_metrics_projection=_projection(),
        absolute_tick=40,
        native_world_control_set_sha256=CONTROL_SET,
    )
    with pytest.raises(CompactScienceCaptureError, match="identity differs"):
        load_compact_science_checkpoint_frontier_v1(
            receipt,
            expected_root=tmp_path,
            expected_absolute_tick=40,
            expected_world_uids=(202, 101),
            expected_native_world_control_set_sha256=CONTROL_SET,
        )


def test_sparse_dispatch_artifact_preserves_logical_world_ownership(
    tmp_path: Path,
) -> None:
    dispatch = CudaFullEvolutionSparseEventDispatchV3(
        packed_events=b"",
        world_offsets=(0, 0, 0),
        event_counts=(0, 0),
        final_next_agent_ids=(9, 13),
        world_uids=(101, 202),
        start_next_absolute_tick=8,
        end_next_absolute_tick=16,
        tick_count=8,
        drain_elapsed_seconds=0.0,
        count_projection_elapsed_seconds=0.0,
        compaction_and_transfer_elapsed_seconds=0.0,
        content_hash_elapsed_seconds=0.0,
        content_sha256="5" * 64,
        artifact_sha256="6" * 64,
        manifest_sha256="7" * 64,
    )
    capture = CudaFullEvolutionCompactScienceCaptureV1(
        event_agent_metrics=torch.empty((0, 21), dtype=torch.int64),
        event_birth_provenance=torch.empty((0, 132), dtype=torch.uint8),
        metric_fields=COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
        birth_provenance_encoding=(
            COMPACT_SCIENCE_BIRTH_PROVENANCE_ENCODING_V1
        ),
        start_next_absolute_tick=8,
        end_next_absolute_tick=16,
        world_uids=(101, 202),
        world_offsets=(0, 0, 0),
        event_content_sha256="5" * 64,
    )
    cells = {
        "alpha": {
            "science_logical_world_id": 101,
            "science_ownership": {"study_id": "GENERIC-A"},
            "science_world_sha256": {"algorithm": "sha256", "hex": "8" * 64},
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": "9" * 64,
            },
        },
        "beta": {
            "science_logical_world_id": 202,
            "science_ownership": {"study_id": "GENERIC-B"},
            "science_world_sha256": {"algorithm": "sha256", "hex": "a" * 64},
            "science_acquisition_sha256": {
                "algorithm": "sha256",
                "hex": "b" * 64,
            },
        },
    }
    receipt = write_compact_science_dispatch_v1(
        tmp_path / "dispatch",
        dispatch=dispatch,
        capture=capture,
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SET,
    )
    assert [row["cell_id"] for row in receipt["logical_worlds"]] == [
        "alpha",
        "beta",
    ]
    assert receipt["world_offsets"] == [0, 0, 0]
    assert receipt["birth_provenance_source_shape"] == [0, 132]
    assert receipt["birth_provenance_shape"] == [0, 68]
    assert receipt["birth_provenance_contract"]["death_rows"] == (
        "omitted_after_exact_zero_validation"
    )
    assert receipt["dropped_record_count"] == 0
    assert receipt["dispatch_payload_storage_encoding"] == (
        "gzip_mtime0_level1_v1"
    )
    repeated = write_compact_science_dispatch_v1(
        tmp_path / "dispatch-repeated",
        dispatch=dispatch,
        capture=capture,
        world_cells=cells,
        native_world_control_set_sha256=CONTROL_SET,
    )
    for prefix in ("event", "metric", "birth_provenance"):
        first_path = Path(receipt["path"]) / receipt[f"{prefix}_file"]
        repeated_path = Path(repeated["path"]) / repeated[f"{prefix}_file"]
        stored = first_path.read_bytes()
        assert first_path.name.endswith(".bin.gz")
        assert stored == repeated_path.read_bytes()
        assert len(stored) == receipt[f"{prefix}_stored_bytes"]
        uncompressed = gzip.decompress(stored)
        assert len(uncompressed) == receipt[f"{prefix}_bytes"]
        assert hashlib.sha256(uncompressed).hexdigest() == receipt[
            f"{prefix}_uncompressed_sha256"
        ]
