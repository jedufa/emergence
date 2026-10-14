from __future__ import annotations

import json
from pathlib import Path

from tools.scripts.eiel_hapb_gate8_measurement_from_h4_v1 import (
    measurement_row_from_h4_receipt_v1,
)
from tools.scripts.eiel_hapb_gate8_measurements_table_v1 import (
    assemble_measurements_table_v1,
    resident_row_from_receipt_v1,
)


def test_assemble_table_keeps_resident_control_and_frozen_high_b() -> None:
    resident_receipt = {
        "candidate_id": "resident_b48",
        "batch_cells": 48,
        "amortized_production_seconds_per_generation": 142.108223859,
        "receipt_sha256": "res",
        "memory": {
            "device_total_bytes": 6027608064,
            "minimum_free_at_peak_bytes": 3196059648,
            "peak_used_bytes": 2831548416,
            "rule_pass": True,
        },
    }
    hapb_receipt = {
        "physical_cells": 48,
        "generations": 6,
        "continuation_exact": True,
        "checkpoint_restore_boundary_mismatch_count": 0,
        "acs_checkpoint_slots": 0,
        "receipt_sha256": "hapb",
        "secondary_boundary_timing": {
            "samples": [{"label": "g0_to_g6", "native_execute_wall_ns": 852_000_000_000}]
        },
        "memory": {
            "device_total_bytes": 6027608064,
            "minimum_free_at_peak_bytes": 2000,
            "rule_pass": True,
        },
    }
    resident = resident_row_from_receipt_v1(
        resident_receipt, receipt_path=Path("/tmp/resident.json")
    )
    hapb = measurement_row_from_h4_receipt_v1(
        hapb_receipt,
        candidate_id="hapb_b48",
        route="hapb",
        receipt_path=Path("/tmp/hapb.json"),
    )
    table = assemble_measurements_table_v1(
        resident=resident,
        hapb_rows=[hapb],
        high_b_abi="frozen",
        ladder_complete=True,
        ladder_stop_reason="next_candidate_cannot_beat_winner",
    )
    assert table["schema_version"] == "eiel_hapb_gate8_benchmark_measurements_v1"
    assert table["measurements"][0]["route"] == "resident"
    assert table["measurements"][1]["candidate_id"] == "hapb_b48"
    assert table["measurements"][1]["sustained_rehearsal_pass"] is False
    assert table["implemented_batch_cells"] == [48]
    assert table["ladder_complete"] is True


def test_official_g6_row_can_be_marked_sustained() -> None:
    hapb_receipt = {
        "physical_cells": 48,
        "generations": 6,
        "continuation_exact": True,
        "checkpoint_restore_boundary_mismatch_count": 0,
        "acs_checkpoint_slots": 2,
        "receipt_sha256": "hapb",
        "secondary_boundary_timing": {
            "samples": [{"label": "g0_to_g6", "native_execute_wall_ns": 852_000_000_000}]
        },
        "memory": {
            "device_total_bytes": 6027608064,
            "minimum_free_at_peak_bytes": 2000,
            "rule_pass": True,
        },
    }
    row = measurement_row_from_h4_receipt_v1(
        hapb_receipt,
        candidate_id="hapb_b48",
        route="hapb",
        receipt_path=Path("/tmp/hapb.json"),
        sustained_rehearsal_pass=True,
    )
    assert row["sustained_rehearsal_pass"] is True


def test_resident_row_from_executed_219_receipt() -> None:
    path = Path(
        "/tmp/emergence-test/runtime/evidence_raw/hapb_g6_frag_rebind_20260816/"
        "219_gate8_resident_b48_g0_g6/gate8_resident_b48_receipt.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    row = resident_row_from_receipt_v1(receipt, receipt_path=path)
    assert row["batch_cells"] == 48
    assert row["route"] == "resident"
    assert row["amortized_production_seconds_per_generation"] == 142.108223859
    assert row["memory"]["rule_pass"] is True
