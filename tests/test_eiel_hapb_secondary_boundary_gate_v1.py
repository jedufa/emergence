from __future__ import annotations

import json
import sys
from pathlib import Path

from tools.scripts import eiel_hapb_secondary_boundary_gate_v1 as gate
from tools.scripts import eiel_hapb_secondary_seam_decision_v1 as seam_decision


def _write_case(root: Path, *, tile_rows: int) -> Path:
    root.mkdir()
    counters = {
        "policy_row_bytes": 39_964,
        "policy_h2d_rows": 256,
        "policy_h2d_bytes": 256 * 39_964,
        "policy_d2h_rows": 0,
        "policy_d2h_bytes": 0,
        "control_d2h_bytes": 0,
        "control_h2d_bytes": 0,
        "sync_count": 1,
        "gpu_stall_ns": 17,
    }
    timing_samples = []
    for label in ("first", "second"):
        timing_samples.append(
            {
                "schema_version": (
                    "eiel_b48_wdl_jungle_eed_hapb_h4_continuation_v1_"
                    "secondary_boundary_timing_sample_v1"
                ),
                "label": label,
                "tile_rows": tile_rows,
                "host_classification_map_construction_wall_ns": 100,
                "native_execute_wall_ns": 200,
                "counters": counters,
                "control_d2h_bytes": 0,
                "control_h2d_bytes": 0,
                "sync_count": 1,
                "gpu_stall_ns": 17,
                "policy_byte_reduction_credit": False,
                "scope": "bounded secondary timing only; wall clock is not a policy-byte reduction measurement",
            }
        )
        (root / f"repeat_tick_{label}_birth_transaction_trace.json").write_text(
            json.dumps(
                {
                    "trace_label": label,
                    "events": [
                        {
                            "active_cells": [
                                {
                                    "policy_birth_tape_cursor_start": 3,
                                    "policy_birth_tape_cursor_end": 4,
                                    "birth_count": 1,
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    receipt = {
        "schema_version": gate.H4_SCHEMA_PREFIX,
        "host_harness_source_commit": "harness",
        "artifact_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "reused_generation_zero_source_root": "/immutable/source",
        "tile_rows": tile_rows,
        "content_alias_enabled": True,
        "production_execution_admission": False,
        "qualification_claim": False,
        "repeat_count": 2,
        "terminal_projection_digests_exact": True,
        "mismatch_count": 0,
        "terminal_projection_digests": {"first": ["digest"], "second": ["digest"]},
        "policy_traffic_observations": {
            label: {
                "content_alias_enabled": True,
                "content_alias_relation_sha256": "relation",
                "counters": counters,
            }
            for label in ("first", "second")
        },
        "process_rng_traces": {
            label: {
                "after_prepare": {"state": "prepare"},
                "after_execute": {"state": "execute"},
            }
            for label in ("first", "second")
        },
        "secondary_boundary_timing": {
            "schema_version": gate.TIMING_SCHEMA_PREFIX,
            "samples": timing_samples,
            "policy_byte_reduction_credit": False,
        },
    }
    receipt_path = root / "repeat_tick_diagnostic_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path


def test_secondary_gate_accepts_exact_alias_tile_pair(tmp_path: Path, monkeypatch) -> None:
    tile_512 = _write_case(tmp_path / "tile512", tile_rows=512)
    tile_4096 = _write_case(tmp_path / "tile4096", tile_rows=4096)
    output_root = tmp_path / "gate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eiel_hapb_secondary_boundary_gate_v1.py",
            "--tile-512-receipt",
            str(tile_512),
            "--tile-4096-receipt",
            str(tile_4096),
            "--output-root",
            str(output_root),
        ],
    )
    assert gate.main() == 0
    result = json.loads(
        (output_root / "hapb_secondary_boundary_gate_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["decision"] == "PASS"
    assert result["policy_byte_reduction_credit"] is False
    assert result["gate6_authorization"] is False


def test_secondary_seam_decision_freezes_512_and_rejects_unmaterial_seams(
    tmp_path: Path, monkeypatch
) -> None:
    def samples(tile_rows: int, native: tuple[int, int], host: tuple[int, int]):
        return [
            {
                "native_execute_wall_ns": native[0],
                "host_classification_map_construction_wall_ns": host[0],
                "control_d2h_bytes": 4_838_400,
                "control_h2d_bytes": 0,
                "sync_count": 0,
                "gpu_stall_ns": 0,
                "policy_byte_reduction_credit": False,
                "tile_rows": tile_rows,
            },
            {
                "native_execute_wall_ns": native[1],
                "host_classification_map_construction_wall_ns": host[1],
                "control_d2h_bytes": 4_838_400,
                "control_h2d_bytes": 0,
                "sync_count": 0,
                "gpu_stall_ns": 0,
                "policy_byte_reduction_credit": False,
                "tile_rows": tile_rows,
            },
        ]

    secondary = tmp_path / "secondary.json"
    secondary.write_text(
        json.dumps(
            {
                "decision": "PASS",
                "terminal_projection_parity": True,
                "rng_after_prepare_and_execute_parity": True,
                "birth_recombination_cursor_parity": True,
                "timing_comparison": {
                    "tile_512": samples(512, (100, 100), (4, 4)),
                    "tile_4096": samples(4096, (110, 110), (5, 5)),
                },
            }
        ),
        encoding="utf-8",
    )
    primary = tmp_path / "primary.json"
    primary.write_text(
        json.dumps(
            {
                "decision": "PASS",
                "policy_weight_h2d_bytes_reduction_ratio": 11.0,
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "decision"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eiel_hapb_secondary_seam_decision_v1.py",
            "--secondary-gate-receipt",
            str(secondary),
            "--primary-gate-receipt",
            str(primary),
            "--output-root",
            str(output_root),
        ],
    )
    assert seam_decision.main() == 0
    result = json.loads(
        (output_root / "hapb_secondary_seam_decision_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["canonical_b48_architecture"]["policy_tile_rows"] == 512
    assert result["tile_selection"]["decision"] == "KEEP_512"
    assert (
        result["secondary_seams"]["gpu_resident_request_map_version_reuse"][
            "decision"
        ]
        == "REJECT"
    )
    assert (
        result["secondary_seams"]["clone_birth_descriptor_alias"]["decision"]
        == "REJECT"
    )
