from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from engine.native_runtime.compact_summary_digests import (
    behavior_rows_digest_sha256,
    lineage_records_digest_sha256,
)
from engine.native_runtime.compact_summary_reporting_projection import (
    BEHAVIOR_ROW_BYTES,
    DYNAMIC_ROW_CAPACITY,
    LINEAGE_ROW_BYTES,
    REPORTING_HEADER_BYTES,
    _BEHAVIOR_DTYPE,
    _HEADER_DTYPE,
    _LINEAGE_DTYPE,
    project_retained_reporting_authority,
    reporting_memory_budget,
    retained_bytes_per_generation_cell,
)


def _fixture(*, generations: int = 2, cells: int = 2) -> np.ndarray:
    capacity = DYNAMIC_ROW_CAPACITY
    stride = retained_bytes_per_generation_cell(capacity)
    retained = np.zeros((generations, cells, stride), dtype=np.uint8)
    behavior_offset = REPORTING_HEADER_BYTES
    lineage_offset = behavior_offset + capacity * BEHAVIOR_ROW_BYTES
    for generation in range(generations):
        for cell in range(cells):
            raw = retained[generation, cell]
            header = np.frombuffer(raw, dtype=_HEADER_DTYPE, count=1)[0]
            header["start_id_ceiling"] = 10
            header["behavior_row_count"] = 2
            header["lineage_record_count"] = 1
            header["status"] = 2
            behavior = np.frombuffer(
                raw[behavior_offset:], dtype=_BEHAVIOR_DTYPE, count=capacity
            )
            behavior[0]["agent_id"] = 3
            behavior[0]["energy_collected_exact"] = 2.5 + generation
            behavior[0]["base_fitness"] = 4.0
            behavior[0]["injury_start"] = np.float32(0.0)
            behavior[0]["injury_end"] = np.float32(-0.0)
            behavior[0]["aggression_bias_f32"] = np.float32(0.25)
            behavior[0]["lifespan_steps"] = 80
            behavior[0]["offspring"] = 1
            behavior[0]["attacks_made"] = 2
            behavior[0]["action_counts"] = [1, 2, 3, 4, 5, 6, 7]
            behavior[0]["flags"] = 0b1111
            behavior[1]["agent_id"] = 12
            behavior[1]["energy_collected_exact"] = 1.0
            behavior[1]["base_fitness"] = 0.5
            behavior[1]["injury_end"] = np.float32(0.5)
            behavior[1]["aggression_bias_f32"] = np.float32(-0.25)
            behavior[1]["lifespan_steps"] = 4
            behavior[1]["flags"] = 0b1000
            lineage = np.frombuffer(
                raw[lineage_offset:], dtype=_LINEAGE_DTYPE, count=capacity
            )
            lineage[0]["agent_id"] = 3
            lineage[0]["energy"] = 40.0
            lineage[0]["energy_collected_exact"] = 2.5 + generation
            lineage[0]["episode0_pred_err_mean"] = 0.125
            lineage[0]["identity_drift_ema"] = 0.0
            lineage[0]["health"] = np.float32(0.75)
            lineage[0]["injury"] = np.float32(-0.0)
            lineage[0]["drives"] = [0.1, 0.2, 0.3, 0.4]
            lineage[0]["pred_err_sum"] = np.float32(0.5)
            lineage[0]["drive_jitter"] = np.float32(0.03125)
            lineage[0]["aggression_bias_f32"] = np.float32(0.25)
            lineage[0]["exploration_bias_f32"] = np.float32(-0.125)
            lineage[0]["y"] = 2 + cell
            lineage[0]["x"] = 4
            lineage[0]["founder_id"] = 7
            lineage[0]["birth_generation"] = 0
            lineage[0]["birth_tick"] = 3
            lineage[0]["lineage_depth"] = 1
            lineage[0]["offspring_count"] = 1
            lineage[0]["survival_steps"] = 80
            lineage[0]["first_repro_step"] = -1
            lineage[0]["pred_err_count"] = 4
            lineage[0]["patch_habitat_lineage_id"] = 0
            lineage[0]["hidden_size"] = 8
            lineage[0]["memory_dim"] = 4
            lineage[0]["use_gru"] = 0
    return retained


def test_struct_sizes_and_memory_budget_are_frozen() -> None:
    assert _HEADER_DTYPE.itemsize == 16
    assert _BEHAVIOR_DTYPE.itemsize == 56
    assert _LINEAGE_DTYPE.itemsize == 136
    assert retained_bytes_per_generation_cell(64) == 12_304
    assert retained_bytes_per_generation_cell(224) == 43_024
    budget = reporting_memory_budget(
        generation_count=32, cells=16, slot_capacity=128
    )
    assert budget[
        "retained_bytes"
    ] == 22_028_288
    assert budget["scratch_bytes"] == 1_179_648
    assert budget["total_bytes"] == 23_207_936


def test_machine_readable_layout_matches_host_decoder() -> None:
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (
            root
            / "docs/300_architecture-specifications/370_specifications/EIEL-DOC-000852--compact-summary-reporting-retained-layout-v2.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["sizes"]["header_bytes"] == _HEADER_DTYPE.itemsize
    assert contract["sizes"]["behavior_row_bytes"] == _BEHAVIOR_DTYPE.itemsize
    assert contract["sizes"]["lineage_row_bytes"] == _LINEAGE_DTYPE.itemsize
    assert contract["sizes"]["dynamic_bytes_per_generation_cell"] == (
        retained_bytes_per_generation_cell(DYNAMIC_ROW_CAPACITY)
    )
    for field, (_dtype, offset) in contract["layout"]["behavior_row"].items():
        assert _BEHAVIOR_DTYPE.fields[field][1] == offset
    for field, (_dtype, offset) in contract["layout"]["lineage_row"].items():
        assert _LINEAGE_DTYPE.fields[field][1] == offset


def test_projection_is_exact_generation_major_and_digest_ready() -> None:
    projected = project_retained_reporting_authority(
        _fixture(), first_generation=5
    )
    assert len(projected) == 2
    assert len(projected[0]) == 2
    for generation_index, rows in enumerate(projected):
        for cell, result in enumerate(rows):
            assert result["generation"] == 5 + generation_index
            assert result["cell"] == cell
            assert result["behavior_rows_digest_sha256"] == (
                behavior_rows_digest_sha256(result["behavior_rows"])
            )
            assert result["lineage_records_digest_sha256"] == (
                lineage_records_digest_sha256(
                    5 + generation_index, result["lineage_records"]
                )
            )
            assert math.copysign(
                1.0, result["behavior_rows"][0]["injury_delta"]
            ) == -1.0
            assert result["lineage_records"][0]["stats"]["lineage_id"] == (
                "L00000007"
            )


def test_projection_canonicalizes_behavior_aggression_bias_to_eight_decimals() -> None:
    retained = _fixture(generations=1, cells=1)
    behavior = np.frombuffer(
        retained[0, 0, REPORTING_HEADER_BYTES:],
        dtype=_BEHAVIOR_DTYPE,
        count=DYNAMIC_ROW_CAPACITY,
    )
    behavior[0]["aggression_bias_f32"] = np.float32(0.42961761)
    assert float(behavior[0]["aggression_bias_f32"]) == 0.42961761355400085

    projected = project_retained_reporting_authority(retained, first_generation=0)

    assert projected[0][0]["behavior_rows"][0]["aggression_bias"] == 0.42961761


def test_projection_canonicalizes_lineage_genome_biases_to_eight_decimals() -> None:
    retained = _fixture(generations=1, cells=1)
    lineage_offset = (
        REPORTING_HEADER_BYTES + DYNAMIC_ROW_CAPACITY * BEHAVIOR_ROW_BYTES
    )
    lineage = np.frombuffer(
        retained[0, 0, lineage_offset:],
        dtype=_LINEAGE_DTYPE,
        count=DYNAMIC_ROW_CAPACITY,
    )
    lineage[0]["aggression_bias_f32"] = np.float32(0.42961761)
    lineage[0]["exploration_bias_f32"] = np.float32(-0.42961761)

    projected = project_retained_reporting_authority(retained, first_generation=0)
    genome = projected[0][0]["lineage_records"][0]["genome"]

    assert genome["aggression_bias"] == 0.42961761
    assert genome["exploration_bias"] == -0.42961761


def test_projection_rejects_device_transfer_ambiguity_and_inactive_bytes() -> None:
    retained = _fixture(generations=1, cells=1)
    retained[0, 0, -1] = 1
    with pytest.raises(ValueError, match="inactive reporting authority"):
        project_retained_reporting_authority(retained, first_generation=0)


def test_projection_rejects_lineage_alive_identity_mismatch() -> None:
    retained = _fixture(generations=1, cells=1)
    capacity = DYNAMIC_ROW_CAPACITY
    lineage_offset = REPORTING_HEADER_BYTES + capacity * BEHAVIOR_ROW_BYTES
    lineage = np.frombuffer(
        retained[0, 0, lineage_offset:], dtype=_LINEAGE_DTYPE, count=capacity
    )
    lineage[0]["agent_id"] = 4
    with pytest.raises(ValueError, match="exactly match"):
        project_retained_reporting_authority(retained, first_generation=0)


def test_projection_rejects_corrupt_negative_lineage_counter() -> None:
    retained = _fixture(generations=1, cells=1)
    capacity = DYNAMIC_ROW_CAPACITY
    lineage_offset = REPORTING_HEADER_BYTES + capacity * BEHAVIOR_ROW_BYTES
    lineage = np.frombuffer(
        retained[0, 0, lineage_offset:], dtype=_LINEAGE_DTYPE, count=capacity
    )
    lineage[0]["survival_steps"] = -1
    with pytest.raises(ValueError, match="negative fields"):
        project_retained_reporting_authority(retained, first_generation=0)
