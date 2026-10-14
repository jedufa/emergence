from __future__ import annotations

import json
import math
from pathlib import Path

import engine.native_runtime.cuda_v2_science_capture_v1 as capture_adapter
from engine.native_runtime.cuda_v2_compact_science_capture_v1 import (
    canonical_sha256,
)
from engine.perf.cuda_full_evolution_v2.direct_sparse_runtime import (
    COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1,
)
from tools.performance.eiel_snapshot_v1_baseline import (
    CLAIM_CLASS,
    SCHEMA_VERSION,
    main,
)


def _load_sealed_report(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    report = json.loads(raw)
    expected = report.pop("report_sha256")
    assert canonical_sha256(report) == expected
    report["report_sha256"] = expected
    return report


def test_tiny_phase_timed_matrix_reports_exact_host_boundary(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "reports" / "baseline.json"
    result = main(
        [
            "--output",
            str(output),
            "--batch-size",
            "2",
            "--repetitions",
            "2",
            "--warmups",
            "1",
            "--sparse-stride",
            "17",
            "--phase-timing",
        ]
    )
    assert result == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["ok"] is True
    assert stdout["output"] == str(output.resolve())

    report = _load_sealed_report(output)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["claim_class"] == CLAIM_CLASS == "TECHNICAL_ONLY"
    assert report["benchmark_scope"] == "HOST_MICRO_WRITER_V1"
    assert report["science_result"] is False
    assert report["qualification_authority_effect"] is False
    assert report["prequalification_started"] is False
    assert report["qualification_started"] is False
    assert report["gpu_used"] is False
    assert report["measurement_boundary"]["T1"] == ("BASELINE_ONLY_NO_SPEEDUP_CLAIM")
    assert report["measurement_boundary"]["T2"] == "NOT_MEASURED"
    assert report["measurement_boundary"]["T3"] == "NOT_MEASURED"

    capacity = int(capture_adapter._STATE_LAYOUT.field("agent_id").shape[0])
    state_bytes = int(capture_adapter._STATE_BYTES)
    metric_count = len(COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1)
    assert report["configuration"] == {
        "absolute_tick": 1,
        "batch_size": 2,
        "denominator_phase_timing": False,
        "include_genome_catalog": False,
        "occupancies": ["zero", "sparse", "all"],
        "occupancy_semantics": (
            "alive-bit distribution; every native slot carries one "
            "deterministic nonzero agent identity"
        ),
        "output": str(output.resolve()),
        "phase_interleave_order": "alternating_exact_input_pairs",
        "phase_timing": True,
        "phase_timed_repetitions": 2,
        "phase_timed_warmups": 1,
        "repetitions": 2,
        "sparse_stride": 17,
        "warmups": 1,
    }
    assert report["layout"]["agent_capacity_per_world"] == capacity
    assert report["layout"]["state_bytes_per_world"] == state_bytes
    assert report["layout"]["metric_fields"] == list(
        COMPACT_SCIENCE_AGENT_METRIC_FIELDS_V1
    )
    assert report["source_identity"]["git"]["commit"]
    assert report["source_identity"]["tool_file_sha256"]
    assert report["command"]["argv"][-1] == "--phase-timing"

    expected_active = {
        "zero": 0,
        "sparse": 2 * math.ceil(capacity / 17),
        "all": 2 * capacity,
    }
    cells = {cell["occupancy"]: cell for cell in report["cells"]}
    assert set(cells) == set(expected_active)
    for occupancy, cell in cells.items():
        assert cell["expected_active_agent_count"] == expected_active[occupancy]
        assert cell["nonzero_agent_id_count"] == 2 * capacity
        assert cell["warmup_repetitions"] == 1
        assert cell["measured_repetitions"] == 2
        assert len(cell["warmup_samples"]) == 1
        assert len(cell["measured_samples"]) == 2
        assert len(cell["phase_timed_warmup_samples"]) == 1
        assert len(cell["phase_timed_samples"]) == 2
        assert cell["warmup_samples"][0]["warmup"] is True
        assert cell["warmup_samples"][0]["measurement_role"] == ("denominator_untimed")
        assert "timing_diagnostics" not in cell["warmup_samples"][0]
        assert (
            cell["phase_timed_warmup_samples"][0]["measurement_role"]
            == "phase_diagnostic_timed"
        )
        assert (
            cell["phase_timed_warmup_samples"][0]["pair_execution_order"]
            == "untimed_then_timed"
        )
        assert all(
            sample["measurement_role"] == "denominator_untimed"
            and "timing_diagnostics" not in sample
            for sample in cell["measured_samples"]
        )
        assert all(
            sample["measurement_role"] == "phase_diagnostic_timed"
            and "timing_diagnostics" in sample
            for sample in cell["phase_timed_samples"]
        )
        assert [
            sample["pair_execution_order"] for sample in cell["measured_samples"]
        ] == ["timed_then_untimed", "untimed_then_timed"]
        assert [
            sample["pair_execution_order"] for sample in cell["phase_timed_samples"]
        ] == ["timed_then_untimed", "untimed_then_timed"]
        assert cell["measured_summary"]["writer_elapsed_seconds"]["count"] == 2
        assert cell["measured_summary"]["row_count"]["min"] == (
            2 + expected_active[occupancy]
        )
        assert cell["measured_summary"]["captured_world_count"]["min"] == 2
        assert cell["measured_summary"]["d2h_bytes"]["max"] == 0
        assert cell["measured_summary"]["state_input_bytes"]["min"] == (2 * state_bytes)
        assert cell["measured_summary"]["metrics_input_bytes"]["min"] == (
            2 * capacity * metric_count * 8
        )
        assert cell["measured_summary"]["uncompressed_bytes"]["min"] > 0
        assert cell["measured_summary"]["compressed_bytes"]["min"] > 0
        assert cell["measured_summary"]["records_per_second"]["min"] > 0
        assert cell["phase_aggregates"]["enabled"] is True
        assert cell["phase_aggregates"]["sample_count"] == 2
        assert cell["phase_aggregates"]["sample_stream"] == ("phase_timed_samples")
        assert cell["phase_aggregates"]["eligible_as_denominator"] is False
        assert "writer_total" in cell["phase_aggregates"]["phases"]
        assert all(
            sample["artifact_sha256"] == cell["measured_samples"][0]["artifact_sha256"]
            for sample in cell["measured_samples"]
        )
        assert all(
            untimed["artifact_sha256"] == timed["artifact_sha256"]
            for untimed, timed in zip(
                cell["measured_samples"],
                cell["phase_timed_samples"],
                strict=True,
            )
        )
        observer = cell["observer_overhead"]
        assert observer["enabled"] is True
        assert observer["sample_pair_count"] == 2
        assert observer["denominator_observer_timing"] is False
        assert observer["phase_measurements_eligible_as_denominator"] is False
        assert (
            observer["untimed_writer_elapsed_ns"]["median"]
            == cell["measured_summary"]["writer_elapsed_ns"]["median"]
        )
        assert (
            observer["median_observer_overhead_ns"]
            == observer["paired_observer_overhead_ns"]["median"]
        )
        assert (
            observer["median_timed_over_untimed_ratio"]
            == observer["paired_timed_over_untimed_ratio"]["median"]
        )
        assert observer["median_timed_over_untimed_ratio"] > 0
    assert report["phase_aggregates_all_measured_cells"]["sample_count"] == 6
    assert report["observer_overhead_all_measured_cells"]["sample_pair_count"] == 6
    assert not list(output.parent.glob(".eiel-snapshot-v1-host-baseline-*"))
    assert not list(output.parent.glob(".*.tmp"))


def test_tiny_single_cell_can_disable_optional_phase_timing(
    tmp_path: Path, capsys
) -> None:
    output = tmp_path / "baseline-no-phases.json"
    assert (
        main(
            [
                "--output",
                str(output),
                "--batch-size",
                "1",
                "--repetitions",
                "1",
                "--warmups",
                "0",
                "--occupancy",
                "zero",
            ]
        )
        == 0
    )
    capsys.readouterr()
    report = _load_sealed_report(output)
    assert report["configuration"]["occupancies"] == ["zero"]
    assert report["configuration"]["phase_timing"] is False
    assert report["configuration"]["denominator_phase_timing"] is False
    assert report["configuration"]["phase_timed_repetitions"] == 0
    assert report["configuration"]["phase_timed_warmups"] == 0
    assert report["configuration"]["phase_interleave_order"] == "untimed_only"
    assert len(report["cells"]) == 1
    cell = report["cells"][0]
    assert cell["warmup_samples"] == []
    assert cell["phase_timed_warmup_samples"] == []
    assert cell["phase_timed_samples"] == []
    assert cell["warmup_summary"] is None
    assert "timing_diagnostics" not in cell["measured_samples"][0]
    assert cell["phase_aggregates"] == {
        "derived": {},
        "eligible_as_denominator": False,
        "enabled": False,
        "phases": {},
        "sample_count": 0,
        "sample_stream": "phase_timed_samples",
        "schema_version": "eiel_compact_science_snapshot_timing_v1",
    }
    observer = cell["observer_overhead"]
    assert observer["enabled"] is False
    assert observer["sample_pair_count"] == 0
    assert observer["denominator_sample_count"] == 1
    assert observer["phase_sample_count"] == 0
    assert observer["untimed_writer_elapsed_ns"]["count"] == 1
    assert observer["timed_writer_elapsed_ns"] is None
    assert report["observer_overhead_all_measured_cells"]["enabled"] is False
