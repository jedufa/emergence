from __future__ import annotations

import json
from pathlib import Path

from engine.native_runtime import scientific_throughput_efficiency_v1 as ste
from experiments.harness import run_experiments
from tools.scripts import eiel_campaign_tui as tui
from tools.scripts import eiel_scientific_throughput_efficiency_v1 as cli


def test_feature_flag_disables_telemetry() -> None:
    assert ste.scientific_throughput_efficiency_enabled({"EIEL_STE_TELEMETRY": "1"})
    assert not ste.scientific_throughput_efficiency_enabled({"EIEL_STE_TELEMETRY": "0"})


def test_cuda_report_keeps_sm_occupancy_as_gpu_submetric_only() -> None:
    report = ste.build_cuda_scientific_throughput_efficiency_report(
        direct_result={
            "grid_blocks": 48,
            "threads_per_block": 128,
            "persistent_hot_loop_launch_count": 12,
            "cuda_event_elapsed_ns": 2_000_000_000,
            "bounded_device_ticks": 3840,
            "torch_observed_interval_peak_allocated_bytes": 4_000,
        },
        production_route_receipt={},
        completed_generations=4,
        compact_summary_rows=4,
        hardware_descriptor={
            "schema_version": "eiel_cuda_hardware_descriptor_v1",
            "backend": "cuda",
            "detector_status": "fixture",
            "gpu_model": "fixture",
            "sm_count": 24,
            "warp_size": 32,
            "max_resident_blocks_per_sm": 16,
            "max_resident_warps_per_sm": 64,
            "max_threads_per_sm": 2048,
            "total_vram_bytes": 8_000,
        },
    )

    assert report["metric_name"] == "Scientific Throughput Efficiency"
    assert report["backend"] == "cuda"
    assert "sm_occupancy" not in report
    gpu = report["backend_specific"]["gpu_resource_utilization"]
    assert gpu["sm_occupancy"] == 1.0
    assert gpu["worlds_per_sm"] == 2.0
    assert gpu["block_saturation"] == 0.125
    assert gpu["average_kernel_duration_seconds"] == 0.166666667
    assert report["science_throughput"]["worlds_completed_per_second"] == 24.0
    assert report["memory_utilization"]["vram_efficiency"] == 0.5


def test_cuda_route_hardware_uses_known_model_defaults() -> None:
    report = ste.build_cuda_scientific_throughput_efficiency_report(
        direct_result={"grid_blocks": 48, "threads_per_block": 128},
        production_route_receipt={
            "hardware": {
                "detector_status": "detected",
                "name": "NVIDIA GeForce GTX 1660 Ti",
                "compute_capability": [7, 5],
                "vram_bytes": 6_024_855_552,
            }
        },
        completed_generations=1,
        compact_summary_rows=1,
    )

    hardware = report["hardware_descriptor"]
    assert hardware["sm_count"] == 24
    assert hardware["cuda_cores"] == 1536
    assert hardware["memory_bandwidth_gbps"] == 288.0
    assert (
        report["backend_specific"]["gpu_resource_utilization"]["worlds_per_sm"] == 2.0
    )


def test_cpu_report_uses_common_top_level_shape() -> None:
    report = ste.build_cpu_scientific_throughput_efficiency_report(
        resident_worlds=6,
        active_worlds=4,
        completed_worlds=3,
        elapsed_seconds=2.0,
        ticks_completed=960,
        generations_completed=12,
        parameter_evaluations=6,
        storage_bytes_written=200,
        hardware_descriptor={
            "schema_version": "eiel_cpu_hardware_descriptor_v1",
            "backend": "cpu",
            "detector_status": "fixture",
            "cpu_model": "fixture",
            "physical_core_count": 2,
            "logical_thread_count": 4,
            "cache_sizes": {},
            "ram_bytes": 10_000,
            "platform": "fixture",
        },
        process_sample={"rss_bytes": 1000, "peak_rss_bytes": 2000},
    )

    assert report["backend"] == "cpu"
    assert (
        report["engineering_model"]["schema_version"] == "eiel_ste_engineering_model_v2"
    )
    assert report["resource_efficiency"]["level"] == 1
    assert report["scientific_efficiency"]["level"] == 2
    assert report["scientific_discovery_efficiency"]["level"] == 3
    assert report["science_throughput"]["worlds_completed_per_second"] == 1.5
    assert report["science_throughput"]["ticks_per_second"] == 480.0
    assert report["scientific_throughput"]["world_generations_per_hour"] == 21600.0
    assert (
        report["primary_science_speed"]["inference_status"]
        == "unknown_multi_trajectory_or_not_supplied"
    )
    assert (
        report["legacy_metric_aliases"]["generations_per_hour"]
        == "scientific_throughput.world_generations_per_hour"
    )
    cpu = report["backend_specific"]["cpu_resource_utilization"]
    assert cpu["worlds_per_physical_core"] == 3.0
    assert cpu["worlds_per_logical_thread"] == 1.5
    assert report["memory_utilization"]["working_set_bytes"] == 1000


def test_discovery_efficiency_reports_productivity_not_scientific_score() -> None:
    report = ste.build_cpu_scientific_throughput_efficiency_report(
        resident_worlds=340,
        active_worlds=0,
        completed_worlds=340,
        elapsed_seconds=3600.0,
        ticks_completed=1_088_000,
        generations_completed=13_600,
        checkpoint_branches=340,
        parameter_evaluations=340,
        hardware_descriptor={"backend": "cpu"},
        process_sample={},
        discovery_metrics={
            "evidence_packages": 340,
            "evidence_volume_bytes": 1_000_000,
            "completed_qualification_cells": 340,
            "campaign_completion_fraction": 1.0,
            "replay_validation": "PASS",
            "evidence_reproducibility": "PASS",
            "successful_replay_validations": 12,
            "completed_deterministic_replay_checks": 12,
        },
    )

    discovery = report["scientific_discovery_efficiency"]
    assert discovery["schema_version"] == "eiel_scientific_discovery_efficiency_v1"
    assert discovery["worlds_completed"] == 340
    assert discovery["generations_completed"] == 13_600
    assert discovery["checkpoint_branches"] == 340
    assert discovery["evidence_packages"] == 340
    assert discovery["completed_qualification_cells"] == 340
    assert discovery["campaign_completion_fraction"] == 1.0
    assert discovery["replay_validation"] == "PASS"
    assert discovery["evidence_reproducibility"] == "PASS"
    assert discovery["trend_metrics"]["worlds_per_hour"] == 340.0
    assert discovery["trend_metrics"]["scientific_worlds_completed_per_hour"] == 340.0
    assert discovery["trend_metrics"]["qualification_generations_per_hour"] == 13600.0
    assert discovery["trend_metrics"]["evidence_packages_per_hour"] == 340.0
    assert "deprecated_legacy_aliases" in discovery["trend_metrics"]
    assert "intelligence" in discovery["excluded_score_metrics"]
    assert "scientific scoring" in discovery["claims_boundary"]


def test_markdown_renders_scientific_discovery_efficiency_section() -> None:
    report = ste.build_cpu_scientific_throughput_efficiency_report(
        resident_worlds=1,
        active_worlds=1,
        completed_worlds=1,
        elapsed_seconds=1.0,
        checkpoint_branches=1,
        parameter_evaluations=1,
        hardware_descriptor={"backend": "cpu"},
        process_sample={},
        discovery_metrics={
            "evidence_packages": 1,
            "completed_qualification_cells": 1,
            "replay_validation": "PASS",
            "evidence_reproducibility": "PASS",
        },
    )

    markdown = ste.render_markdown(report)

    assert "## Resource Efficiency" in markdown
    assert "## Primary Science Speed" in markdown
    assert "## Scientific Throughput" in markdown
    assert "## Qualification Throughput" in markdown
    assert "## Scientific Efficiency (Legacy Aggregate Aliases)" in markdown
    assert "## Scientific Discovery Efficiency" in markdown
    assert "Worlds completed: 1" in markdown
    assert "Evidence packages: 1" in markdown
    assert "Replay validation: PASS" in markdown


def test_dashboard_payload_and_time_series_are_trend_ready(tmp_path: Path) -> None:
    sample = ste.build_resource_utilization_time_series_sample(
        timestamp_seconds=3.0,
        cpu_utilization_percent=25.0,
        rss_bytes=1000,
        ram_used_bytes=2000,
        ram_total_bytes=4000,
        gpu_utilization_percent=50.0,
        vram_used_bytes=3000,
        vram_total_bytes=6000,
        source="fixture",
    )
    report = ste.build_cpu_scientific_throughput_efficiency_report(
        resident_worlds=2,
        active_worlds=2,
        completed_worlds=2,
        elapsed_seconds=2.0,
        hardware_descriptor={"backend": "cpu"},
        process_sample={},
        discovery_metrics={"evidence_packages": 1},
        time_series_samples=[sample],
    )

    dashboard = ste.build_scientific_throughput_efficiency_dashboard_payload(report)
    paths = ste.write_scientific_throughput_efficiency_report_files(report, tmp_path)

    assert report["resource_utilization_time_series"]["sample_count"] == 1
    assert (
        report["resource_utilization_time_series"]["samples"][0]["source"] == "fixture"
    )
    assert dashboard["schema_version"] == "eiel_ste_dashboard_payload_v1"
    assert dashboard["cards"]["resource_efficiency"]["label"] == "Resource Efficiency"
    assert (
        dashboard["cards"]["primary_science_speed"]["label"] == "Primary Science Speed"
    )
    assert (
        dashboard["cards"]["scientific_throughput"]["label"] == "Scientific Throughput"
    )
    assert (
        dashboard["cards"]["qualification_throughput"]["label"]
        == "Qualification Throughput"
    )
    assert dashboard["scheduler_action"] == "none"
    assert Path(paths["report_json"]).is_file()
    assert (
        Path(paths["report_markdown"])
        .read_text(encoding="utf-8")
        .startswith("## Scientific Throughput Efficiency")
    )
    assert Path(paths["dashboard_json"]).is_file()


def test_campaign_aggregate_preserves_backend_independent_semantics() -> None:
    first = ste.build_cpu_scientific_throughput_efficiency_report(
        resident_worlds=1,
        active_worlds=1,
        completed_worlds=1,
        elapsed_seconds=1.0,
        hardware_descriptor={"backend": "cpu"},
        process_sample={},
    )
    second = {**first, "backend": "cuda"}

    aggregate = ste.aggregate_scientific_throughput_efficiency_reports([first, second])

    assert aggregate is not None
    assert aggregate["backend"] == "mixed"
    assert aggregate["backend_reports"] == 2
    assert aggregate["science_throughput"]["mean_worlds_completed_per_second"] == 1.0
    assert aggregate["resource_efficiency"]["label"] == "Resource Efficiency"
    assert aggregate["scientific_efficiency"]["label"] == "Scientific Efficiency"
    discovery = aggregate["scientific_discovery_efficiency"]
    assert discovery["source_reports"] == 2
    assert discovery["worlds_completed"] == 2
    assert "worlds_per_hour" in discovery["mean_trend_metrics"]
    assert "scientific_worlds_completed_per_hour" in discovery["mean_trend_metrics"]
    assert "deprecated_legacy_aliases" in discovery["mean_trend_metrics"]


def test_harness_collects_run_manifest_ste(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = ste.build_cpu_scientific_throughput_efficiency_report(
        resident_worlds=1,
        active_worlds=1,
        completed_worlds=1,
        elapsed_seconds=1.0,
        hardware_descriptor={"backend": "cpu"},
        process_sample={},
    )
    (run_dir / "cuda_architecture_b_output_manifest.json").write_text(
        json.dumps({"scientific_throughput_efficiency": report}, sort_keys=True),
        encoding="ascii",
    )

    reports = ste.collect_run_scientific_throughput_efficiency_reports([run_dir])
    assert (
        run_experiments.aggregate_scientific_throughput_efficiency_reports(reports)[
            "backend_reports"
        ]
        == 1
    )


def test_scaling_estimator_is_backend_independent() -> None:
    cuda = ste.estimate_backend_scaling(
        backend="cuda",
        hardware_descriptor={"gpu_model": "NVIDIA GeForce GTX 1660 Ti"},
        resident_world_counts=[24, 48, 96],
    )
    cpu = ste.estimate_backend_scaling(
        backend="cpu",
        hardware_descriptor={"physical_core_count": 8, "logical_thread_count": 16},
        resident_world_counts=[8, 16],
    )

    assert cuda[0]["classification"] == "near_saturation"
    assert cuda[1]["worlds_per_sm"] == 2.0
    assert cpu[0]["worlds_per_physical_core"] == 1.0


def test_offline_cli_estimator_is_input_driven() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--backend",
            "cuda",
            "--resident-worlds",
            "24",
            "--completed-worlds",
            "24",
            "--elapsed-seconds",
            "3",
            "--sm-count",
            "24",
            "--generations-completed",
            "4",
            "--evidence-packages",
            "24",
            "--completed-qualification-cells",
            "24",
            "--replay-validation",
            "PASS",
            "--cpu-utilization-percent",
            "10",
            "--rss-bytes",
            "1000",
            "--scaling-worlds",
            "24,48",
        ]
    )

    report = cli.estimate(args)
    scaling = ste.estimate_backend_scaling(
        backend=args.backend,
        hardware_descriptor=report["hardware_descriptor"],
        resident_world_counts=[24, 48],
    )

    assert report["backend"] == "cuda"
    assert report["hardware_descriptor"]["detector_status"] == "offline_input"
    assert report["scientific_discovery_efficiency"]["evidence_packages"] == 24
    assert report["scientific_discovery_efficiency"]["replay_validation"] == "PASS"
    assert report["resource_utilization_time_series"]["sample_count"] == 1
    assert (
        report["backend_specific"]["gpu_resource_utilization"][
            "occupancy_classification"
        ]
        == "near_saturation"
    )
    assert scaling[1]["resident_worlds"] == 48


def test_report_json_schema_exposes_three_level_model() -> None:
    schema = ste.scientific_throughput_efficiency_json_schema()

    assert schema["$id"] == "eiel_scientific_throughput_efficiency_report_v2"
    assert "resource_efficiency" in schema["required"]
    assert "scientific_efficiency" in schema["required"]
    assert "scientific_discovery_efficiency" in schema["required"]
    assert "resource_utilization_time_series" in schema["properties"]
    assert "dashboard_payload" in schema["properties"]
    assert "primary_science_speed" in schema["properties"]
    assert "scientific_throughput" in schema["properties"]
    assert "qualification_throughput" in schema["properties"]
    assert "evidence_throughput" in schema["properties"]


def test_campaign_tui_renders_ste_dashboard_summary() -> None:
    report = ste.build_cpu_scientific_throughput_efficiency_report(
        resident_worlds=1,
        active_worlds=1,
        completed_worlds=1,
        elapsed_seconds=1.0,
        hardware_descriptor={"backend": "cpu"},
        process_sample={},
        discovery_metrics={"evidence_packages": 2},
    )
    rendered = tui.render_plain(
        {
            "children": [],
            "planned_runs": 0,
            "runs_root": "/tmp/eiel-fixture",
            "overall_eta_s": None,
            "summary": {"scientific_throughput_efficiency": report},
        }
    )

    assert "STE | resource occupancy" in rendered
    assert "primary gen/h" in rendered
    assert "qualification gen/h" in rendered
    assert "evidence packages 2" in rendered


def test_primary_science_speed_requires_single_world_or_explicit_primary_counter() -> (
    None
):
    aggregate = ste.build_cpu_scientific_throughput_efficiency_report(
        resident_worlds=1,
        active_worlds=1,
        completed_worlds=3,
        elapsed_seconds=422.92,
        generations_completed=600,
        ticks_completed=96_000,
        hardware_descriptor={"backend": "cpu"},
        process_sample={},
        discovery_metrics={"completed_qualification_cells": 3},
    )
    explicit = ste.build_cpu_scientific_throughput_efficiency_report(
        resident_worlds=1,
        active_worlds=1,
        completed_worlds=3,
        elapsed_seconds=422.92,
        generations_completed=600,
        ticks_completed=96_000,
        hardware_descriptor={"backend": "cpu"},
        process_sample={},
        discovery_metrics={
            "completed_qualification_cells": 3,
            "primary_world_generations_completed": 200,
            "primary_world_ticks_completed": 32_000,
            "primary_world_elapsed_seconds": 422.92,
        },
    )

    assert (
        aggregate["primary_science_speed"]["primary_world_generations_per_hour"] is None
    )
    assert (
        aggregate["qualification_throughput"]["qualification_generations_per_hour"]
        == 5107.348908
    )
    assert (
        explicit["primary_science_speed"]["primary_world_generations_per_hour"]
        == 1702.449636
    )
