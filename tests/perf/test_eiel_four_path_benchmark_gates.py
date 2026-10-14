from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from native_runtime.lifecycle_events import typed_lifecycle_events_from_reference_contexts
from native_runtime.runner_integration import emit_route_neutral_typed_lifecycle_public_outputs
from tools.performance.run_historical_current_four_path_benchmark import (
    Arm,
    BENCHMARK_WALL_TIME_CV_MAX,
    Cell,
    CURRENT_ENGINE_ONLY_KNOBS,
    CURRENT_MATCHED_PUBLIC_OUTPUT_KNOBS,
    DEFAULT_REGISTRY,
    EXACT_CURRENT_PUBLIC_OUTPUT_FILES,
    MINIMUM_W1_REPETITIONS,
    MINIMUM_W6_REPETITIONS,
    NATIVE_PEAK_RSS_MAX_REFERENCE_RATIO,
    PROJECT_ROOT,
    PROMOTION_CURRENT_W1_SPEEDUP_MIN,
    PROMOTION_CURRENT_W6_SPEEDUP_MIN,
    REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES,
    _aggregate_output_inventory,
    _analyze,
    _build_native_extension,
    _cell_command,
    _extension_integrity_check,
    _load_registry,
    _matched_output_checkpoint_profile,
    _metrics_summary_pair_fairness,
    _metrics_summary_profile,
    _native_peak_rss_fairness,
    _native_worker_command,
    _native_worker_probe_main,
    _native_worker_promotion_gate,
    _output_burden_fairness,
    _prepare_native_extension,
    _promotion_policy_from_registry,
    _python_interpreter_path,
    _run_output_profile,
    _repetition_plan,
    _shared_metric_parity,
    _timed_subprocess_env,
    main as benchmark_main,
)


def _result(arm: str, *, cell_id: str = "s1001_high_g2_ckptoff", births: float = 10.0) -> dict[str, object]:
    return {
        "rep": "rep1",
        "arm": arm,
        "cell_id": cell_id,
        "metrics": {
            "final_alive": 256,
            "total_successful_births": births,
            "total_failed_births": 2.0,
            "total_deaths": 1.0,
        },
    }


def _campaign(arm: str, result: dict[str, object]) -> dict[str, object]:
    return {"arm": arm, "results": [result]}


def _cell() -> Cell:
    return Cell(
        cell_id="s101_moderate_g2_ckptoff",
        seed=101,
        generations=2,
        steps=80,
        pop=64,
        grid_size=16,
        food_spawn_per_step=10,
        max_food=64,
        churn="moderate",
        checkpoint_mode="off",
        checkpoint_period=0,
    )


def _write_metrics_summary(
    run_dir: Path,
    rows: list[tuple[int, int, int, int, str]],
    *,
    header: str = "generation,alive,births,deaths,avg_gen_time_s",
) -> None:
    path = run_dir / "viz" / "metrics_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header, *(",".join(str(value) for value in row) for row in rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metrics_rows = [
        (generation, alive, births, 0, deaths, f"lineage-{generation}")
        for generation, alive, births, deaths, _timing in rows
    ]
    (run_dir / "metrics.csv").write_text(
        "generation,alive,successful_births,failed_births,deaths,lineage_semantic\n"
        + "".join(",".join(str(value) for value in row) + "\n" for row in metrics_rows),
        encoding="utf-8",
    )
    for relative_path in REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES:
        if relative_path in {"metrics.csv", "viz/metrics_summary.csv"}:
            continue
        required = run_dir / relative_path
        required.parent.mkdir(parents=True, exist_ok=True)
        required.write_text("fixture\n", encoding="utf-8")


def _profile_result(arm: str, run_dir: Path) -> dict[str, object]:
    output_profile = _run_output_profile(run_dir)
    return {
        "rep": "rep1",
        "arm": arm,
        "cell_id": "s101_moderate_g2_ckptoff",
        "checkpoint_mode": "off",
        "checkpoint_period": 0,
        "checkpoint_files": output_profile["checkpoint_files"],
        "checkpoint_bytes": output_profile["checkpoint_bytes"],
        "output_bytes": output_profile["output_bytes"],
        "output_profile": output_profile,
    }


def _write_canonical_checkpoint_metadata(
    run_dir: Path,
    *,
    digest: str,
    route_only_filename: str,
) -> None:
    bundle = run_dir / "checkpoint_bundles_v2" / "bundle_gen000001"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "checkpoint_bundle_v2",
                "generation_index": 1,
                "canonical_checkpoint_storage_schema_version": (
                    "eiel_checkpoint_digest_layers_v1"
                ),
                "canonical_checkpoint_storage_digest": digest,
                "canonical_checkpoint_storage_field_count": 3,
            }
        ),
        encoding="utf-8",
    )
    (bundle / route_only_filename).write_text("route-support\n", encoding="utf-8")


def _native_worker_audit() -> dict[str, object]:
    return {
        "schema_version": "eiel_native_worker_promotion_audit_v1",
        "extension_path": "/tmp/accepted-native.so",
        "extension_sha256": "a" * 64,
        "extension_abi_version": "eiel_native_runtime_abi_v1",
        "extension_version": "eiel_native_runtime_flat_v1",
        "comparator_source_commit": "d" * 40,
        "comparator_module_path": str(
            (PROJECT_ROOT / "engine/native_runtime/four_way_interchange.py").resolve()
        ),
        "comparator_module_sha256": "c" * 64,
        "runtime_jit": False,
        "native_ownership_active": True,
        "mixed_ownership": False,
        "stale_extension": False,
        "checks": {"fixture": True},
        "first_failure_cause": None,
        "ok": True,
    }


def _analysis_campaigns(
    tmp_path: Path,
    wall_times: list[tuple[float, float]],
) -> list[dict[str, object]]:
    campaigns: list[dict[str, object]] = []
    w1_wall_times = [
        (7.50, 5.00),
        (7.55, 5.03),
        (7.45, 4.97),
        (7.52, 5.01),
        (7.48, 4.99),
    ]
    arm_wall_times: list[tuple[str, int, list[tuple[float, float]]]] = [
        ("w1", 1, w1_wall_times),
        ("w6", 6, wall_times),
    ]
    for lane, workers, lane_wall_times in arm_wall_times:
        for rep_index, (reference_wall_s, native_wall_s) in enumerate(
            lane_wall_times, start=1
        ):
            rep = f"rep{rep_index}"
            for arm, wall_s, timing in (
                (f"current_{lane}_reference", reference_wall_s, "2.5302305620280094"),
                (f"current_{lane}_native", native_wall_s, "0.7745103650377132"),
            ):
                run_dir = tmp_path / lane / rep / arm
                _write_metrics_summary(run_dir, [(0, 64, 7, 2, timing)])
                result = _profile_result(arm, run_dir)
                result["rep"] = rep
                result["metrics"] = _result(arm)["metrics"]
                result["time_v"] = {
                    "max_rss_kb": 120_000 + rep_index,
                    "user_time_s": wall_s * 0.8,
                    "system_time_s": wall_s * 0.1,
                    "cpu_percent_value": 95.0,
                }
                is_native = arm.endswith("_native")
                result["native_worker_audit"] = _native_worker_audit() if is_native else None
                result["promotion_eligible"] = True
                result["complete"] = True
                output_profile = result["output_profile"]
                campaigns.append(
                    {
                        "rep": rep,
                        "arm": arm,
                        "workers": workers,
                        "native": is_native,
                        "wall_s": wall_s,
                        "cells_per_second": 1.0 / wall_s,
                        "steps_per_second": 160.0 / wall_s,
                        "output_bytes": output_profile["output_bytes"],
                        "checkpoint_bytes": result["checkpoint_bytes"],
                        "peak_cell_rss_kb": 120_000 + rep_index,
                        "sum_cell_peak_rss_kb": 120_000 + rep_index,
                        "child_user_time_s": wall_s * 0.8,
                        "child_system_time_s": wall_s * 0.1,
                        "child_cpu_time_s": wall_s * 0.9,
                        "worker_utilization_ratio": 0.15,
                        "resource_metrics_complete": True,
                        "attempted_cells": 1,
                        "complete_cells": 1,
                        "incomplete_cells": 0,
                        "any_hidden_fallback": False,
                        "any_native_fail_closed": False,
                        "promotion_eligible": True,
                        "results": [result],
                        "excluded_results": [],
                    }
                )
    return campaigns


def test_python_interpreter_path_preserves_venv_launcher_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    system_python = tmp_path / "usr" / "bin" / "python3.14"
    system_python.parent.mkdir(parents=True)
    system_python.touch()
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(system_python)
    monkeypatch.chdir(tmp_path)

    selected = _python_interpreter_path(Path(".venv/bin/python"))

    assert selected == venv_python
    assert selected.is_symlink()
    assert selected.resolve() == system_python


def test_current_reference_command_uses_same_fixed_mlp_scope_as_native() -> None:
    reference = Arm("current_w1_reference", Path("/current"), 1, False, reference="current")
    native = Arm("current_w1_native", Path("/current"), 1, True, reference="current")

    reference_cmd = _cell_command(Path("/python"), reference, _cell(), "ref", Path("/runs"))
    native_cmd = _cell_command(Path("/python"), native, _cell(), "native", Path("/runs"))

    assert CURRENT_MATCHED_PUBLIC_OUTPUT_KNOBS == (
        "sim_cfg.typed_lifecycle_public_output_route_neutral_enable=true",
        "sim_cfg.lineage_observability_enable=true",
        "sim_cfg.runnercore1_write_summary=false",
        "sim_cfg.native_runtime_write_summary=false",
        "sim_cfg.developmental_trace_write_enable=false",
        "sim_cfg.phase_e_e1_enable=false",
        "sim_cfg.a3_recovery_taxonomy_enable=false",
        "sim_cfg.lineage_qd_archive_enable=false",
    )
    for cmd in (reference_cmd, native_cmd):
        assert "--native_runtime_cxx26_scope_lock_enable" in cmd
        family_index = cmd.index("--native_runtime_fixed_policy_family")
        assert cmd[family_index + 1] == "mlp"
        for knob in CURRENT_ENGINE_ONLY_KNOBS:
            assert knob in cmd
        for knob in CURRENT_MATCHED_PUBLIC_OUTPUT_KNOBS:
            key = knob.split("=", 1)[0]
            command_knobs = [
                cmd[index + 1]
                for index, argument in enumerate(cmd[:-1])
                if argument == "--knob"
            ]
            assert [value for value in command_knobs if value.startswith(f"{key}=")] == [knob]
    assert "metrics/lineage_descriptors.jsonl" in REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES
    assert EXACT_CURRENT_PUBLIC_OUTPUT_FILES == (
        "behaviors.csv",
        "metrics/lineage_descriptors.jsonl",
        "metrics/lineage_summary.csv",
        "metrics/typed_lifecycle_events.csv",
        "metrics/typed_lifecycle_events.jsonl",
    )
    assert set(EXACT_CURRENT_PUBLIC_OUTPUT_FILES) <= set(REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES)
    assert "sim_cfg.native_safe_default_enable=false" in reference_cmd
    assert "sim_cfg.native_safe_default_enable=true" in native_cmd
    assert "--native_runtime_enable" not in reference_cmd
    assert "--native_runtime_enable" in native_cmd


def test_historical_command_does_not_receive_current_scope_flags() -> None:
    historical = Arm("historical_w1", Path("/historical"), 1, False, reference="historical")

    cmd = _cell_command(Path("/python"), historical, _cell(), "historical", Path("/runs"))

    assert "--native_runtime_cxx26_scope_lock_enable" not in cmd
    assert "--native_runtime_fixed_policy_family" not in cmd
    assert not any("native_safe_default_enable" in arg for arg in cmd)
    assert not any(knob in cmd for knob in CURRENT_ENGINE_ONLY_KNOBS)
    assert not any(knob in cmd for knob in CURRENT_MATCHED_PUBLIC_OUTPUT_KNOBS)


def test_route_neutral_lifecycle_writer_emits_identical_required_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = typed_lifecycle_events_from_reference_contexts(
        [
            {
                "generation": 0,
                "tick": 3,
                "event_type": "birth_success",
                "parent_id": 10,
                "child_id": 12,
            },
            {
                "generation": 0,
                "tick": 5,
                "event_type": "death_committed",
                "agent_id": 11,
                "reason_code": 1,
                "value": 1.0,
            },
        ]
    )
    prefixed = {f"typed_lifecycle_event_{name}": tensor for name, tensor in events.items()}
    projection = {
        "metadata": {"typed_lifecycle_event_active_count": 2},
        "tensors": prefixed,
    }

    from perf import cxxbackbone10_resident_state_owner as resident_owner

    monkeypatch.setattr(
        resident_owner,
        "_typed_lifecycle_event_tensors_for_runner",
        lambda _runner: prefixed,
    )

    reference = SimpleNamespace(run_dir=str(tmp_path / "reference"))
    native = SimpleNamespace(
        run_dir=str(tmp_path / "native"),
        _native_eiel_engine=SimpleNamespace(export_checkpoint_projection=lambda: projection),
    )
    reference_summary = emit_route_neutral_typed_lifecycle_public_outputs(reference)
    native_summary = emit_route_neutral_typed_lifecycle_public_outputs(native)
    reference_profile = _run_output_profile(Path(reference.run_dir))
    native_profile = _run_output_profile(Path(native.run_dir))

    assert reference_summary["total_event_rows"] == 2
    assert native_summary["total_event_rows"] == 2
    assert reference_profile["relative_files"] == [
        "metrics/typed_lifecycle_events.csv",
        "metrics/typed_lifecycle_events.jsonl",
    ]
    assert native_profile["relative_files"] == reference_profile["relative_files"]
    assert native_profile["output_bytes"] == reference_profile["output_bytes"]
    assert not (Path(native.run_dir) / "metrics" / "native_typed_lifecycle_events.csv").exists()


def test_output_inventory_classifies_purpose_requirement_and_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_metrics_summary(run_dir, [(0, 64, 7, 2, "1.0"), (1, 60, 5, 4, "0.9")])
    diagnostic = run_dir / "diagnostics" / "policy_cache_trace.jsonl"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text('{"hit": true}\n{"hit": false}\n', encoding="utf-8")

    profile = _run_output_profile(run_dir)
    inventory = {entry["relative_path"]: entry for entry in profile["output_inventory"]}

    assert inventory["metrics.csv"] == {
        "relative_path": "metrics.csv",
        "enabled": True,
        "purpose": "generation_metrics",
        "requirement_class": "required",
        "required": True,
        "debug_only": False,
        "count": 1,
        "rows": 2,
        "row_count_error": None,
        "bytes": inventory["metrics.csv"]["bytes"],
        "sha256": inventory["metrics.csv"]["sha256"],
    }
    assert inventory["diagnostics/policy_cache_trace.jsonl"]["purpose"] == "cache_trace"
    assert inventory["diagnostics/policy_cache_trace.jsonl"]["requirement_class"] == "debug_only"
    assert inventory["diagnostics/policy_cache_trace.jsonl"]["rows"] == 2


def test_aggregate_output_inventory_records_counts_and_enabled_state(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_metrics_summary(run_dir, [(0, 64, 7, 2, "1.0")])
    result = _profile_result("current_w6_reference", run_dir)

    inventory = _aggregate_output_inventory([_campaign("current_w6_reference", result)])
    metrics_entry = next(entry for entry in inventory["entries"] if entry["relative_path"] == "metrics.csv")

    assert metrics_entry["purpose"] == "generation_metrics"
    assert metrics_entry["requirement_class"] == "required"
    assert metrics_entry["count"] == 1
    assert metrics_entry["row_count"]["median"] == 1.0
    assert metrics_entry["enabled_for_all_runs"] is True


def test_aggregate_output_inventory_reports_reference_native_equivalence(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "2.0")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "1.0")])

    inventory = _aggregate_output_inventory(
        [
            _campaign("current_w6_reference", _profile_result("current_w6_reference", reference_run)),
            _campaign("current_w6_native", _profile_result("current_w6_native", native_run)),
        ]
    )
    metrics_equivalence = next(
        entry
        for entry in inventory["reference_native_equivalence"]
        if entry["relative_path"] == "metrics.csv"
    )

    assert inventory["reference_native_equivalence_ok"] is True
    assert metrics_equivalence["purpose"] == "generation_metrics"
    assert metrics_equivalence["row_count_equal"] is True
    assert metrics_equivalence["equivalent"] is True


def test_shared_metric_parity_passes_exact_current_reference_native_pair() -> None:
    summary = _shared_metric_parity(
        [
            _campaign("current_w6_reference", _result("current_w6_reference", births=10.0)),
            _campaign("current_w6_native", _result("current_w6_native", births=10.0)),
        ]
    )

    assert summary["ok"] is True
    assert summary["paired_cells"] == 1
    assert summary["mismatched_cells"] == 0


def test_shared_metric_parity_reports_first_current_reference_native_mismatch() -> None:
    summary = _shared_metric_parity(
        [
            _campaign("current_w6_reference", _result("current_w6_reference", births=467.0)),
            _campaign("current_w6_native", _result("current_w6_native", births=193.0)),
        ]
    )

    assert summary["ok"] is False
    assert summary["paired_cells"] == 1
    assert summary["mismatched_cells"] == 1
    assert summary["first_mismatch"]["cell_id"] == "s1001_high_g2_ckptoff"
    assert summary["first_mismatch"]["differences"]["total_successful_births"] == {
        "reference": 467.0,
        "native": 193.0,
    }


def test_output_burden_fairness_fails_native_output_excess() -> None:
    summary = _output_burden_fairness(
        {
            "current_w6_reference": {"output_bytes": {"median": 100.0}},
            "current_w6_native": {"output_bytes": {"median": 126.0}},
        },
        max_native_to_reference_ratio=1.25,
    )

    assert summary["ok"] is False
    assert summary["pairs"][0]["native_to_reference_ratio"] == 1.26


def test_output_burden_fairness_fails_native_output_deficit() -> None:
    summary = _output_burden_fairness(
        {
            "current_w6_reference": {"output_bytes": {"median": 100.0}},
            "current_w6_native": {"output_bytes": {"median": 79.0}},
        },
    )

    assert summary["ok"] is False
    assert summary["pairs"][0]["native_to_reference_ratio"] == 0.79
    assert summary["pairs"][0]["min_allowed_ratio"] == 0.80


@pytest.mark.parametrize("ratio", [0.80, 1.25])
def test_output_burden_fairness_accepts_inclusive_symmetric_boundaries(ratio: float) -> None:
    summary = _output_burden_fairness(
        {
            "current_w6_reference": {"output_bytes": {"median": 100.0}},
            "current_w6_native": {"output_bytes": {"median": 100.0 * ratio}},
        },
    )

    assert summary["ok"] is True


def test_output_burden_fairness_fails_closed_on_first_offending_paired_cell() -> None:
    reference_results = [
        _result("current_w6_reference", cell_id="cell-b") | {"output_bytes": 100},
        _result("current_w6_reference", cell_id="cell-a") | {"output_bytes": 100},
    ]
    native_results = [
        _result("current_w6_native", cell_id="cell-b") | {"output_bytes": 126},
        _result("current_w6_native", cell_id="cell-a") | {"output_bytes": 79},
    ]
    summary = _output_burden_fairness(
        {
            "current_w6_reference": {"output_bytes": {"median": 200.0}},
            "current_w6_native": {"output_bytes": {"median": 200.0}},
        },
        [
            {"arm": "current_w6_reference", "results": reference_results},
            {"arm": "current_w6_native", "results": native_results},
        ],
    )

    assert summary["ok"] is False
    assert summary["pairs"][0]["aggregate_median_ok"] is True
    assert summary["paired_cells"] == 2
    assert summary["out_of_bounds_cells"] == 2
    assert summary["first_offending_cell"] == {
        "reference_arm": "current_w6_reference",
        "native_arm": "current_w6_native",
        "rep": "rep1",
        "cell_id": "cell-a",
        "reference_output_bytes": 100,
        "native_output_bytes": 79,
        "native_to_reference_ratio": 0.79,
        "min_allowed_ratio": 0.80,
        "max_allowed_ratio": 1.25,
        "reason": "ratio_out_of_bounds",
    }


def test_metrics_summary_allows_observed_timing_inequality_diagnostically(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "2.5302305620280094")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "0.7745103650377132")])

    summary = _metrics_summary_pair_fairness(
        _metrics_summary_profile(reference_run / "viz" / "metrics_summary.csv"),
        _metrics_summary_profile(native_run / "viz" / "metrics_summary.csv"),
    )

    assert summary["ok"] is True
    assert summary["semantic_rows_equal"] is True
    assert summary["reference_avg_gen_time_s_values"] == [2.5302305620280094]
    assert summary["native_avg_gen_time_s_values"] == [0.7745103650377132]
    assert summary["raw_bytes_equal"] is False
    assert summary["raw_byte_inequality"] is True


def test_metrics_summary_rejects_semantic_birth_and_death_mismatch(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "1.0")])
    _write_metrics_summary(native_run, [(0, 64, 6, 3, "1.0")])

    summary = _metrics_summary_pair_fairness(
        _metrics_summary_profile(reference_run / "viz" / "metrics_summary.csv"),
        _metrics_summary_profile(native_run / "viz" / "metrics_summary.csv"),
    )

    assert summary["ok"] is False
    assert summary["first_semantic_mismatch"]["differences"] == {
        "births": {"reference": 7, "native": 6},
        "deaths": {"reference": 2, "native": 3},
    }


def test_matched_profile_rejects_nonaggregate_metrics_csv_field_mismatch(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "1.0")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "1.0")])
    (native_run / "metrics.csv").write_text(
        "generation,alive,successful_births,failed_births,deaths,lineage_semantic\n"
        "0,64,7,0,2,lineage-different\n",
        encoding="utf-8",
    )

    summary = _matched_output_checkpoint_profile(
        [
            _campaign("current_w6_reference", _profile_result("current_w6_reference", reference_run)),
            _campaign("current_w6_native", _profile_result("current_w6_native", native_run)),
        ]
    )

    assert summary["ok"] is False
    assert summary["first_mismatch"]["reasons"] == ["metrics_csv_full_semantic_mismatch"]
    assert summary["first_mismatch"]["metrics_csv"]["semantic_scope"] == "all_emitted_columns_and_rows"
    assert summary["first_mismatch"]["metrics_csv"]["first_mismatch"] == {
        "surface": "data",
        "row_index": 0,
        "column_index": 5,
        "column": "lineage_semantic",
        "reference": "lineage-0",
        "native": "lineage-different",
    }


@pytest.mark.parametrize("invalid_timing", ["nan", "inf", "-0.0001"])
def test_metrics_summary_rejects_non_finite_or_negative_timing(
    tmp_path: Path,
    invalid_timing: str,
) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, invalid_timing)])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "0.5")])

    reference_profile = _metrics_summary_profile(reference_run / "viz" / "metrics_summary.csv")
    summary = _metrics_summary_pair_fairness(
        reference_profile,
        _metrics_summary_profile(native_run / "viz" / "metrics_summary.csv"),
    )

    assert reference_profile["valid"] is False
    assert "row_1_avg_gen_time_s_not_finite_nonnegative" in reference_profile["errors"]
    assert summary["ok"] is False


def test_metrics_summary_rejects_header_order_and_generation_order(tmp_path: Path) -> None:
    wrong_header_run = tmp_path / "wrong-header"
    wrong_generation_run = tmp_path / "wrong-generation"
    _write_metrics_summary(
        wrong_header_run,
        [(0, 64, 7, 2, "0.5")],
        header="generation,alive,deaths,births,avg_gen_time_s",
    )
    _write_metrics_summary(
        wrong_generation_run,
        [(1, 64, 7, 2, "0.5"), (0, 64, 7, 2, "0.4")],
    )

    wrong_header = _metrics_summary_profile(wrong_header_run / "viz" / "metrics_summary.csv")
    wrong_generation = _metrics_summary_profile(wrong_generation_run / "viz" / "metrics_summary.csv")

    assert wrong_header["valid"] is False
    assert "header_mismatch" in wrong_header["errors"]
    assert wrong_generation["valid"] is False
    assert "row_2_generation_order" in wrong_generation["errors"]


def test_matched_output_profile_rejects_reference_only_output(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "2.0")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "1.0")])
    (reference_run / "reference-only.jsonl").write_text("{}\n", encoding="utf-8")

    summary = _matched_output_checkpoint_profile(
        [
            _campaign("current_w6_reference", _profile_result("current_w6_reference", reference_run)),
            _campaign("current_w6_native", _profile_result("current_w6_native", native_run)),
        ]
    )

    assert summary["ok"] is False
    assert summary["first_mismatch"]["reasons"] == ["output_file_profile_mismatch"]
    assert summary["first_mismatch"]["reference_only_files"] == ["reference-only.jsonl"]


def test_matched_output_profile_rejects_symmetric_debug_only_output(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "2.0")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "1.0")])
    for run_dir in (reference_run, native_run):
        trace = run_dir / "diagnostics" / "policy_cache_trace.jsonl"
        trace.parent.mkdir(parents=True)
        trace.write_text('{"hit": true}\n', encoding="utf-8")

    summary = _matched_output_checkpoint_profile(
        [
            _campaign("current_w6_reference", _profile_result("current_w6_reference", reference_run)),
            _campaign("current_w6_native", _profile_result("current_w6_native", native_run)),
        ]
    )

    assert summary["ok"] is False
    assert summary["first_mismatch"]["reasons"] == ["debug_only_output_enabled"]
    assert summary["first_mismatch"]["debug_only_outputs_enabled"] == [
        "diagnostics/policy_cache_trace.jsonl"
    ]


def test_matched_output_profile_rejects_symmetrically_missing_required_output(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "2.0")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "1.0")])
    relative_path = "metrics/typed_lifecycle_events.jsonl"
    (reference_run / relative_path).unlink()
    (native_run / relative_path).unlink()

    summary = _matched_output_checkpoint_profile(
        [
            _campaign("current_w6_reference", _profile_result("current_w6_reference", reference_run)),
            _campaign("current_w6_native", _profile_result("current_w6_native", native_run)),
        ]
    )

    assert summary["ok"] is False
    assert summary["first_mismatch"]["reasons"] == ["required_public_output_missing"]
    assert summary["first_mismatch"]["missing_required_public_outputs"] == [relative_path]


def test_matched_output_profile_rejects_typed_lifecycle_byte_mismatch(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "2.0")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "1.0")])
    relative_path = "metrics/typed_lifecycle_events.jsonl"
    (native_run / relative_path).write_text("different\n", encoding="utf-8")

    summary = _matched_output_checkpoint_profile(
        [
            _campaign("current_w6_reference", _profile_result("current_w6_reference", reference_run)),
            _campaign("current_w6_native", _profile_result("current_w6_native", native_run)),
        ]
    )

    assert summary["ok"] is False
    assert summary["first_mismatch"]["reasons"] == ["exact_public_output_byte_mismatch"]
    assert summary["first_mismatch"]["exact_public_output_mismatches"] == [relative_path]


def test_matched_output_profile_accepts_semantics_with_raw_timing_inequality(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "2.5302305620280094")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "0.7745103650377132")])

    summary = _matched_output_checkpoint_profile(
        [
            _campaign("current_w6_reference", _profile_result("current_w6_reference", reference_run)),
            _campaign("current_w6_native", _profile_result("current_w6_native", native_run)),
        ]
    )

    assert summary["ok"] is True
    assert summary["raw_metrics_summary_inequalities"] == 1
    assert summary["first_raw_metrics_summary_inequality"]["cell_id"] == "s101_moderate_g2_ckptoff"


def test_matched_output_profile_rejects_checkpoint_configuration_mismatch(tmp_path: Path) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "2.0")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "1.0")])
    reference_result = _profile_result("current_w6_reference", reference_run)
    native_result = _profile_result("current_w6_native", native_run)
    native_result["checkpoint_period"] = 1

    summary = _matched_output_checkpoint_profile(
        [
            _campaign("current_w6_reference", reference_result),
            _campaign("current_w6_native", native_result),
        ]
    )

    assert summary["ok"] is False
    assert summary["first_mismatch"]["reasons"] == ["checkpoint_profile_mismatch"]
    assert summary["first_mismatch"]["checkpoint_configuration_equal"] is False


def test_matched_output_profile_uses_canonical_checkpoint_storage_not_raw_layout(
    tmp_path: Path,
) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "2.0")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "1.0")])
    digest = "a" * 64
    _write_canonical_checkpoint_metadata(
        reference_run,
        digest=digest,
        route_only_filename="bank_state.json",
    )
    _write_canonical_checkpoint_metadata(
        native_run,
        digest=digest,
        route_only_filename="bundle_complete.json",
    )
    reference_result = _profile_result("current_w6_reference", reference_run)
    native_result = _profile_result("current_w6_native", native_run)
    for result in (reference_result, native_result):
        result["checkpoint_mode"] = "every_gen"
        result["checkpoint_period"] = 1

    summary = _matched_output_checkpoint_profile(
        [
            _campaign("current_w6_reference", reference_result),
            _campaign("current_w6_native", native_result),
        ]
    )

    assert summary["ok"] is True
    assert summary["first_mismatch"] is None


def test_matched_output_profile_rejects_canonical_checkpoint_digest_mismatch(
    tmp_path: Path,
) -> None:
    reference_run = tmp_path / "reference"
    native_run = tmp_path / "native"
    _write_metrics_summary(reference_run, [(0, 64, 7, 2, "2.0")])
    _write_metrics_summary(native_run, [(0, 64, 7, 2, "1.0")])
    _write_canonical_checkpoint_metadata(
        reference_run,
        digest="a" * 64,
        route_only_filename="bank_state.json",
    )
    _write_canonical_checkpoint_metadata(
        native_run,
        digest="b" * 64,
        route_only_filename="bundle_complete.json",
    )
    reference_result = _profile_result("current_w6_reference", reference_run)
    native_result = _profile_result("current_w6_native", native_run)
    for result in (reference_result, native_result):
        result["checkpoint_mode"] = "every_gen"
        result["checkpoint_period"] = 1

    summary = _matched_output_checkpoint_profile(
        [
            _campaign("current_w6_reference", reference_result),
            _campaign("current_w6_native", native_result),
        ]
    )

    assert summary["ok"] is False
    assert summary["first_mismatch"]["reasons"] == ["checkpoint_profile_mismatch"]
    assert summary["first_mismatch"]["canonical_checkpoint_storage_equal"] is False


def test_analysis_speedup_uses_repeated_external_campaign_wall_time(tmp_path: Path) -> None:
    campaigns = _analysis_campaigns(
        tmp_path / "campaign",
        [(6.0, 3.0), (6.1, 3.05), (5.9, 2.95)],
    )

    analysis = _analyze(campaigns, tmp_path / "analysis-output")

    assert analysis["fail_closed"] is False
    assert analysis["speedups"]["native_contribution_c_over_d"] == 2.0
    assert analysis["performance_measurement"] == {
        "source": "external_time.perf_counter_around_repeated_arm_subprocess_campaigns",
        "metrics_summary_avg_gen_time_s_role": "finite_nonnegative_output_validity_only",
    }
    assert analysis["repetition_gate"]["ok"] is True
    assert analysis["output_burden_fairness"]["ok"] is True
    assert analysis["output_burden_fairness"]["paired_cells"] == 8
    assert analysis["output_burden_fairness"]["out_of_bounds_cells"] == 0
    assert analysis["stability_cv_max"] == BENCHMARK_WALL_TIME_CV_MAX
    assert analysis["stability_flags"] == []
    assert analysis["resource_metrics"]["ok"] is True
    assert analysis["resource_metrics"]["arms"]["current_w6_native"]["peak_cell_rss_kb"]["count"] == 3
    promotion = (tmp_path / "analysis-output" / "analysis" / "promotion_evidence.json").read_text(
        encoding="utf-8"
    )
    assert '"promotion_safe": true' in promotion
    assert '"resource_metrics"' in promotion


def test_analysis_fails_closed_on_cell_output_ratio_hidden_by_aggregate_median(
    tmp_path: Path,
) -> None:
    campaigns = _analysis_campaigns(
        tmp_path / "campaign",
        [(6.0, 3.0), (6.1, 3.05), (5.9, 2.95)],
    )
    native_rep1 = next(
        campaign
        for campaign in campaigns
        if campaign["arm"] == "current_w6_native" and campaign["rep"] == "rep1"
    )
    native_result = native_rep1["results"][0]
    reference_rep1 = next(
        campaign
        for campaign in campaigns
        if campaign["arm"] == "current_w6_reference" and campaign["rep"] == "rep1"
    )
    reference_result = reference_rep1["results"][0]
    native_result["output_bytes"] = int(reference_result["output_bytes"]) * 2

    analysis = _analyze(campaigns, tmp_path / "analysis-output")

    fairness = analysis["output_burden_fairness"]
    assert fairness["pairs"][0]["aggregate_median_ok"] is True
    assert fairness["out_of_bounds_cells"] == 1
    assert fairness["first_offending_cell"]["rep"] == "rep1"
    assert fairness["first_offending_cell"]["cell_id"] == "s101_moderate_g2_ckptoff"
    assert fairness["first_offending_cell"]["native_to_reference_ratio"] == 2.0
    assert analysis["fail_closed"] is True
    assert "native_reference_output_burden_ratio_out_of_bounds" in analysis["fail_closed_reasons"]


def test_analysis_propagates_stability_flags_to_fail_closed_promotion(
    tmp_path: Path,
) -> None:
    campaigns = _analysis_campaigns(
        tmp_path / "campaign",
        [(9.0, 4.5), (6.0, 3.0), (3.0, 1.5)],
    )

    analysis = _analyze(campaigns, tmp_path / "analysis-output")

    assert analysis["fail_closed"] is True
    assert analysis["fail_closed_reasons"] == ["measured_repetition_stability_cv_exceeded"]
    assert {flag["arm"] for flag in analysis["stability_flags"]} == {
        "current_w6_reference",
        "current_w6_native",
    }
    assert all(flag["cv"] == 0.5 for flag in analysis["stability_flags"])
    promotion = json.loads(
        (tmp_path / "analysis-output" / "analysis" / "promotion_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    assert promotion["promotion_safe"] is False
    assert promotion["stability_ok"] is False
    assert promotion["stability_flags"] == analysis["stability_flags"]
    assert promotion["fail_closed_reasons"] == analysis["fail_closed_reasons"]


@pytest.mark.parametrize(
    ("native_arm", "scale", "lane"),
    [
        ("current_w1_native", 1.50 / 1.49, "current_w1_reference_to_native"),
        ("current_w6_native", 2.00 / 1.14, "current_w6_reference_to_native"),
    ],
)
def test_analysis_enforces_preregistered_w1_and_w6_speedup_thresholds(
    tmp_path: Path,
    native_arm: str,
    scale: float,
    lane: str,
) -> None:
    campaigns = _analysis_campaigns(
        tmp_path / "campaign",
        [(6.0, 3.0), (6.1, 3.05), (5.9, 2.95)],
    )
    for campaign in campaigns:
        if campaign["arm"] == native_arm:
            campaign["wall_s"] = float(campaign["wall_s"]) * scale

    analysis = _analyze(campaigns, tmp_path / "analysis-output")

    gate_row = next(
        row for row in analysis["performance_threshold_gate"]["rows"] if row["lane"] == lane
    )
    assert gate_row["ok"] is False
    assert gate_row["observed_speedup"] < gate_row["minimum_speedup"]
    assert "preregistered_native_speedup_threshold_not_met" in analysis["fail_closed_reasons"]


def test_same_cell_peak_rss_gate_accepts_boundary_and_only_preregistered_bounded_exception(
    tmp_path: Path,
) -> None:
    campaigns = _analysis_campaigns(
        tmp_path / "campaign",
        [(6.0, 3.0), (6.1, 3.05), (5.9, 2.95)],
    )
    reference = next(
        campaign
        for campaign in campaigns
        if campaign["arm"] == "current_w6_reference" and campaign["rep"] == "rep1"
    )["results"][0]
    native = next(
        campaign
        for campaign in campaigns
        if campaign["arm"] == "current_w6_native" and campaign["rep"] == "rep1"
    )["results"][0]
    reference["time_v"]["max_rss_kb"] = 100_000
    native["time_v"]["max_rss_kb"] = 125_000
    policy = _promotion_policy_from_registry({})

    boundary = _native_peak_rss_fairness(campaigns, policy)

    assert boundary["ok"] is True
    boundary_row = next(
        row
        for row in boundary["rows"]
        if row["native_arm"] == "current_w6_native" and row["rep"] == "rep1"
    )
    assert boundary_row["native_to_reference_ratio"] == NATIVE_PEAK_RSS_MAX_REFERENCE_RATIO

    native["time_v"]["max_rss_kb"] = 129_000
    rejected = _native_peak_rss_fairness(campaigns, policy)
    assert rejected["ok"] is False
    assert rejected["first_failure"]["exception_id"] is None

    exception_policy = copy.deepcopy(policy)
    exception_policy["native_peak_rss_exceptions"] = [
        {
            "id": "w6-rep1-fixture",
            "reference_arm": "current_w6_reference",
            "native_arm": "current_w6_native",
            "rep": "rep1",
            "cell_id": "s101_moderate_g2_ckptoff",
            "max_native_to_reference_ratio": 1.30,
            "rationale": "bounded deterministic fixture",
            "approval_record": "test-preregistration",
            "approved_before_authoritative_run": True,
        }
    ]
    accepted = _native_peak_rss_fairness(campaigns, exception_policy)
    accepted_row = next(
        row
        for row in accepted["rows"]
        if row["native_arm"] == "current_w6_native" and row["rep"] == "rep1"
    )
    assert accepted["ok"] is True
    assert accepted_row["exception_id"] == "w6-rep1-fixture"
    assert accepted_row["allowed_max_ratio"] == 1.30


@pytest.mark.parametrize(
    ("field", "bad_value", "summary_field"),
    [
        ("runtime_jit", True, "runtime_jit_false"),
        ("native_ownership_active", False, "expected_native_ownership"),
        ("mixed_ownership", True, "mixed_ownership_false"),
        ("stale_extension", True, "stale_extension_false"),
    ],
)
def test_native_worker_gate_rejects_runtime_and_ownership_contract_failures(
    tmp_path: Path,
    field: str,
    bad_value: object,
    summary_field: str,
) -> None:
    campaigns = _analysis_campaigns(
        tmp_path / "campaign",
        [(6.0, 3.0), (6.1, 3.05), (5.9, 2.95)],
    )
    target = next(campaign for campaign in campaigns if campaign["arm"] == "current_w6_native")
    target["results"][0]["native_worker_audit"][field] = bad_value

    gate = _native_worker_promotion_gate(
        campaigns,
        {"path": "/tmp/accepted-native.so", "expected_sha256": "a" * 64, "ok": True},
    )

    assert gate[summary_field] is False
    assert gate["ok"] is False


def test_w6_native_worker_gate_rejects_path_sha_abi_or_version_heterogeneity(
    tmp_path: Path,
) -> None:
    campaigns = _analysis_campaigns(
        tmp_path / "campaign",
        [(6.0, 3.0), (6.1, 3.05), (5.9, 2.95)],
    )
    target = next(
        campaign
        for campaign in campaigns
        if campaign["arm"] == "current_w6_native" and campaign["rep"] == "rep1"
    )
    second_result = copy.deepcopy(target["results"][0])
    second_result["cell_id"] = "s202_moderate_g2_ckptoff"
    second_result["native_worker_audit"]["extension_version"] = "stale-version"
    target["results"].append(second_result)
    target["attempted_cells"] = 2
    target["complete_cells"] = 2

    gate = _native_worker_promotion_gate(
        campaigns,
        {"path": "/tmp/accepted-native.so", "expected_sha256": "a" * 64, "ok": True},
    )

    campaign_gate = next(
        row
        for row in gate["campaigns"]
        if row["arm"] == "current_w6_native" and row["rep"] == "rep1"
    )
    assert campaign_gate["identity_homogeneous"] is False
    assert campaign_gate["w6_identity_homogeneous"] is False
    assert gate["w6_identity_homogeneous"] is False
    assert gate["ok"] is False


def test_w6_native_worker_gate_rejects_comparator_commit_heterogeneity(
    tmp_path: Path,
) -> None:
    campaigns = _analysis_campaigns(
        tmp_path / "campaign",
        [(6.0, 3.0), (6.1, 3.05), (5.9, 2.95)],
    )
    target = next(
        campaign
        for campaign in campaigns
        if campaign["arm"] == "current_w6_native" and campaign["rep"] == "rep1"
    )
    second_result = copy.deepcopy(target["results"][0])
    second_result["cell_id"] = "s202_moderate_g2_ckptoff"
    second_result["native_worker_audit"]["comparator_source_commit"] = "e" * 40
    target["results"].append(second_result)
    target["attempted_cells"] = 2
    target["complete_cells"] = 2

    gate = _native_worker_promotion_gate(
        campaigns,
        {"path": "/tmp/accepted-native.so", "expected_sha256": "a" * 64, "ok": True},
    )

    campaign_gate = next(
        row
        for row in gate["campaigns"]
        if row["arm"] == "current_w6_native" and row["rep"] == "rep1"
    )
    assert campaign_gate["identity_homogeneous"] is False
    assert campaign_gate["w6_identity_homogeneous"] is False
    assert gate["ok"] is False


def test_failed_cell_is_excluded_from_statistics_with_durable_first_cause(
    tmp_path: Path,
) -> None:
    campaigns = _analysis_campaigns(
        tmp_path / "campaign",
        [(6.0, 3.0), (6.1, 3.05), (5.9, 2.95)],
    )
    failed_campaign = next(
        campaign
        for campaign in campaigns
        if campaign["arm"] == "current_w6_native" and campaign["rep"] == "rep1"
    )
    failed_result = failed_campaign["results"].pop()
    failed_result.update(
        {
            "complete": False,
            "promotion_eligible": False,
            "first_failure_cause": "subprocess_returncode:9",
            "exclusion_reasons": ["subprocess_returncode:9", "run_finalization_incomplete"],
        }
    )
    failed_campaign.update(
        {
            "complete_cells": 0,
            "incomplete_cells": 1,
            "promotion_eligible": False,
            "first_failure_cause": "subprocess_returncode:9",
            "exclusion_reasons": ["failed_or_incomplete_cell"],
            "excluded_results": [failed_result],
        }
    )

    analysis = _analyze(campaigns, tmp_path / "analysis-output")

    exclusion = analysis["cell_exclusion_audit"]
    assert exclusion["ok"] is False
    assert exclusion["excluded_cells"] == 1
    assert exclusion["inventory_excludes_failed_cells"] is True
    assert exclusion["first_failure"]["first_failure_cause"] == "subprocess_returncode:9"
    assert analysis["arms"]["current_w6_native"]["wall_s"]["count"] == 2
    assert "failed_or_incomplete_cells_excluded" in analysis["fail_closed_reasons"]
    assert "insufficient_w1_or_w6_repetitions" in analysis["fail_closed_reasons"]


def test_native_worker_command_wraps_exact_extension_and_external_audit(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "accepted.so"
    extension.write_bytes(b"accepted")
    audit = tmp_path / "audits" / "cell.json"
    command = ["/venv/python", "/project/engine/main.py", "--seed", "101"]

    wrapped = _native_worker_command(
        command,
        audit_path=audit,
        native_extension=extension,
        expected_sha256="b" * 64,
        expected_source_commit="d" * 40,
    )

    assert wrapped[0] == command[0]
    assert wrapped[2] == "--native-worker-probe"
    assert wrapped[3] == str(audit)
    assert wrapped[4] == str(extension.resolve())
    assert wrapped[5] == "b" * 64
    assert wrapped[6] == "d" * 40
    assert wrapped[7:] == [command[1], "--", *command[2:]]


def test_native_worker_probe_captures_exact_runtime_identity_and_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import native_runtime.bindings as bindings
    import tools.performance.run_historical_current_four_path_benchmark as benchmark_module

    extension = tmp_path / "accepted.so"
    extension.write_bytes(b"accepted")
    audit_path = tmp_path / "worker-audit.json"
    fake_main = tmp_path / "fake_engine_main.py"
    fake_main.write_text(
        """
from types import SimpleNamespace

class Payload:
    def __init__(self, payload):
        self.payload = payload
    def to_dict(self):
        return dict(self.payload)

def execute_run_context(ctx):
    return 0

def main(argv):
    counters = Payload({
        'runtime_jit_build_requests': 0,
        'normal_python_agent_traversals': 0,
        'normal_apply_owner_checkpoint_to_runner_calls': 0,
    })
    capability = Payload({
        'extension_abi_version': 'eiel_native_runtime_abi_v1',
        'extension_version': 'eiel_native_runtime_flat_v1',
        'runtime_jit_in_backend': False,
        'production_jit_free': True,
        'supports_python_agent_sync_in_normal_lane': False,
    })
    engine = SimpleNamespace(
        counters=counters,
        capability=capability,
        _owner=SimpleNamespace(_native_owner=object()),
    )
    runner = SimpleNamespace(
        sim_cfg=SimpleNamespace(native_runtime_allow_runtime_jit_build=False),
        _native_safe_default_decision={
            'native_execution_active': True,
            'fail_closed': False,
        },
        _native_runtime_summary={
            'schema_version': 'eiel_native_generation_result_v1',
            'decision': 'EIEL-NATIVE-RUNTIME-PERSISTENT-GENERATION',
            'executed': True,
            'compact_result': {'python_agent_sync': False},
        },
        _native_eiel_engine=engine,
    )
    execute_run_context(SimpleNamespace(runner=runner))
    return 0
""".lstrip(),
        encoding="utf-8",
    )
    status_payload = {
        "extension_path": str(extension.resolve()),
        "manifest_path": f"{extension.resolve()}.manifest.json",
        "extension_abi_version": "eiel_native_runtime_abi_v1",
        "extension_version": "eiel_native_runtime_flat_v1",
        "abi_compatible": True,
        "output_sha256_verified": True,
        "available": True,
        "runtime_jit": False,
        "loaded": True,
        "error": "",
    }

    def fake_status(_path: object = None) -> SimpleNamespace:
        return SimpleNamespace(**status_payload, to_dict=lambda: dict(status_payload))

    loaded_module = SimpleNamespace(
        __file__=str(extension.resolve()),
        EIEL_NATIVE_RUNTIME_ABI_VERSION="eiel_native_runtime_abi_v1",
        EIEL_NATIVE_RUNTIME_EXTENSION_VERSION="eiel_native_runtime_flat_v1",
    )
    monkeypatch.setattr(bindings, "native_extension_status", fake_status)
    monkeypatch.setattr(bindings, "load_prebuilt_extension", lambda _path: loaded_module)
    monkeypatch.setattr(benchmark_module, "_sha256", lambda _path: "c" * 64)
    monkeypatch.setattr(benchmark_module, "_git", lambda _args, _cwd: "d" * 40)

    returncode = _native_worker_probe_main(
        [
            str(audit_path),
            str(extension),
            "c" * 64,
            "d" * 40,
            str(fake_main),
            "--",
            "--seed",
            "101",
        ]
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))

    assert returncode == 0
    assert audit["ok"] is True
    assert audit["extension_path"] == str(extension.resolve())
    assert audit["extension_sha256"] == "c" * 64
    assert audit["comparator_source_commit"] == "d" * 40
    assert audit["comparator_module_path"] == str(
        (PROJECT_ROOT / "engine/native_runtime/four_way_interchange.py").resolve()
    )
    assert audit["runtime_jit"] is False
    assert audit["native_ownership_active"] is True
    assert audit["mixed_ownership"] is False
    assert audit["stale_extension"] is False
    assert audit["first_failure_cause"] is None


def test_preregistered_policy_rejects_weakened_threshold_or_unbounded_exception() -> None:
    with pytest.raises(ValueError, match="must exactly match validated promotion envelope value 1.5"):
        _promotion_policy_from_registry(
            {"promotion_gates": {"current_w1_reference_to_native_min_speedup": 1.49}}
        )
    with pytest.raises(ValueError, match="must bind one exact rep and cell_id"):
        _promotion_policy_from_registry(
            {
                "promotion_gates": {
                    "current_w1_reference_to_native_min_speedup": 1.50,
                    "current_w6_reference_to_native_min_speedup": 1.15,
                    "measured_arm_wall_time_cv_max": 0.05,
                    "output_native_to_reference_min_ratio": 0.80,
                    "output_native_to_reference_max_ratio": 1.25,
                    "native_peak_rss_to_same_cell_reference_max_ratio": 1.25,
                    "native_peak_rss_exceptions": [
                        {
                            "id": "unbounded",
                            "reference_arm": "current_w6_reference",
                            "native_arm": "current_w6_native",
                            "max_native_to_reference_ratio": 1.30,
                            "rationale": "missing exact cell binding",
                            "approval_record": "test",
                            "approved_before_authoritative_run": True,
                        }
                    ]
                }
            }
        )
    with pytest.raises(ValueError, match="IDs must exactly match"):
        _promotion_policy_from_registry(
            {
                "promotion_gates": {
                    "current_w1_reference_to_native_min_speedup": 1.50,
                    "current_w6_reference_to_native_min_speedup": 1.15,
                    "measured_arm_wall_time_cv_max": 0.05,
                    "output_native_to_reference_min_ratio": 0.80,
                    "output_native_to_reference_max_ratio": 1.25,
                    "native_peak_rss_to_same_cell_reference_max_ratio": 1.25,
                    "native_peak_rss_exceptions": [
                        {
                            "id": "registry-only-exception",
                            "reference_arm": "current_w6_reference",
                            "native_arm": "current_w6_native",
                            "rep": "rep1",
                            "cell_id": "s101_moderate_g2_ckptoff",
                            "max_native_to_reference_ratio": 1.30,
                            "rationale": "not accepted by the strict contract",
                            "approval_record": "test",
                            "approved_before_authoritative_run": True,
                        }
                    ],
                }
            }
        )


def test_repetition_plan_enforces_five_w1_and_three_w6() -> None:
    arms = {
        "current_w1_reference": Arm("current_w1_reference", Path("/current"), 1, False),
        "current_w1_native": Arm("current_w1_native", Path("/current"), 1, True),
        "current_w6_reference": Arm("current_w6_reference", Path("/current"), 6, False),
        "current_w6_native": Arm("current_w6_native", Path("/current"), 6, True),
    }

    plan = _repetition_plan(
        {"benchmark_repetitions": {"w1": 5, "w6": 3}},
        arms,
        list(arms),
    )

    assert plan == {
        "current_w1_reference": MINIMUM_W1_REPETITIONS,
        "current_w1_native": MINIMUM_W1_REPETITIONS,
        "current_w6_reference": MINIMUM_W6_REPETITIONS,
        "current_w6_native": MINIMUM_W6_REPETITIONS,
    }
    with pytest.raises(ValueError, match="W1 repetitions must be >= 5"):
        _repetition_plan({}, arms, list(arms), w1_override=4)
    with pytest.raises(ValueError, match="W6 repetitions must be >= 3"):
        _repetition_plan({}, arms, list(arms), w6_override=2)


def test_registry_records_authoritative_w1_w6_repetition_policy() -> None:
    registry = _load_registry(DEFAULT_REGISTRY)
    policy = _promotion_policy_from_registry(registry)

    assert registry["benchmark_repetitions"]["w1"] == MINIMUM_W1_REPETITIONS
    assert registry["benchmark_repetitions"]["w6"] == MINIMUM_W6_REPETITIONS
    assert "current_w1_native" in {name for order in registry["repetition_order"] for name in order}
    assert policy["current_w1_reference_to_native_min_speedup"] == PROMOTION_CURRENT_W1_SPEEDUP_MIN
    assert policy["current_w6_reference_to_native_min_speedup"] == PROMOTION_CURRENT_W6_SPEEDUP_MIN
    assert policy["measured_arm_wall_time_cv_max"] == BENCHMARK_WALL_TIME_CV_MAX
    assert policy["native_peak_rss_to_same_cell_reference_max_ratio"] == (
        NATIVE_PEAK_RSS_MAX_REFERENCE_RATIO
    )
    assert policy["native_peak_rss_exceptions"] == []
    assert policy["source"]["contract_id"] == "flat_native_promotion_envelope_v1"
    assert policy["source"]["contract_version"] == 1
    assert policy["source"]["contract_path"].endswith(
        "registries/performance/flat_native_promotion_envelope_v1.json"
    )
    assert len(policy["source"]["contract_sha256"]) == 64
    assert policy["source"]["registry_mirror_checked"] is True
    assert policy["w1_repetitions_min"] == MINIMUM_W1_REPETITIONS
    assert policy["w6_repetitions_min"] == MINIMUM_W6_REPETITIONS
    assert policy["warmup_repetitions"] == 1

    matrix = registry["matrix"]
    cell_count = (
        len(matrix["seeds"])
        * len(matrix["churn_levels"])
        * len(matrix["workload_lengths"])
        * len(matrix["checkpoint_modes"])
    )
    measured_count = sum(
        cell_count
        * (
            registry["benchmark_repetitions"]["w1"]
            if int(spec["workers"]) == 1
            else registry["benchmark_repetitions"]["w6"]
        )
        for spec in registry["arms"].values()
    )
    assert cell_count == 96
    assert measured_count == 2_016
    assert measured_count + len(registry["arms"]) == 2_021


def test_extension_integrity_check_detects_post_campaign_mutation(tmp_path: Path) -> None:
    extension = tmp_path / "eiel_native_runtime_ext.so"
    extension.write_bytes(b"immutable-v1")
    expected = _extension_integrity_check(extension, "wrong", "precondition")["observed_sha256"]

    baseline = _extension_integrity_check(extension, str(expected), "post_build")
    extension.write_bytes(b"mutated-v2")
    post_campaign = _extension_integrity_check(extension, str(expected), "post_all_campaigns")

    assert baseline["ok"] is True
    assert post_campaign["ok"] is False
    assert post_campaign["observed_sha256"] != post_campaign["expected_sha256"]


def test_external_native_extension_is_smoked_in_place_without_build_or_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    extension = external_dir / "accepted_native_extension.so"
    original_bytes = b"accepted-external-extension-v1"
    extension.write_bytes(original_bytes)
    original_mtime_ns = extension.stat().st_mtime_ns
    selected_path = tmp_path / "selected-extension.so"
    selected_path.symlink_to(extension)
    expected_sha256 = hashlib.sha256(original_bytes).hexdigest()
    capture_calls: list[dict[str, object]] = []

    def fail_if_build_invoked(*_args: object, **_kwargs: object) -> Path:
        pytest.fail("external native extension selection invoked the build path")

    def fake_run_capture(
        cmd: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, object]:
        capture_calls.append(
            {"cmd": list(cmd), "cwd": cwd, "env": dict(env or {}), "timeout": timeout}
        )
        return {
            "command": list(cmd),
            "cwd": str(cwd),
            "returncode": 0,
            "stdout": "loaded\n",
            "stderr": "",
            "wall_s": 0.01,
        }

    monkeypatch.setattr(
        "tools.performance.run_historical_current_four_path_benchmark._build_native_extension",
        fail_if_build_invoked,
    )
    monkeypatch.setattr(
        "tools.performance.run_historical_current_four_path_benchmark._run_capture",
        fake_run_capture,
    )

    exact_path, immutable_manifest = _prepare_native_extension(
        Path("/venv/bin/python"),
        tmp_path / "evidence",
        external_path=selected_path,
        expected_external_sha256=expected_sha256.upper(),
    )

    assert exact_path == extension.resolve()
    assert extension.read_bytes() == original_bytes
    assert extension.stat().st_mtime_ns == original_mtime_ns
    assert selected_path.is_symlink()
    assert len(capture_calls) == 1
    assert capture_calls[0]["cmd"][-1] == str(exact_path)  # type: ignore[index]
    assert "load_prebuilt_extension(p)" in capture_calls[0]["cmd"][2]  # type: ignore[index]
    assert capture_calls[0]["env"]["EIEL_NATIVE_RUNTIME_EXTENSION"] == str(  # type: ignore[index]
        exact_path
    )
    assert immutable_manifest["artifact_source"] == "external_in_place"
    assert immutable_manifest["build_invoked"] is False
    assert immutable_manifest["copy_or_rewrite_permitted"] is False
    assert immutable_manifest["path"] == str(exact_path)
    assert immutable_manifest["required_sha256"] == expected_sha256
    assert immutable_manifest["expected_sha256"] == expected_sha256
    assert immutable_manifest["checks"][0]["phase"] == "post_external_load_smoke"
    environment_dir = tmp_path / "evidence" / "environment"
    persisted_manifest = json.loads(
        (environment_dir / "immutable_native_extension_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted_manifest == immutable_manifest
    smoke = json.loads((environment_dir / "native_load_smoke.json").read_text(encoding="utf-8"))
    assert smoke["extension_path"] == str(exact_path)
    assert smoke["artifact_unchanged"] is True
    assert not (environment_dir / "eiel_native_runtime_ext_four_path.so").exists()


def test_external_native_extension_missing_or_hash_mismatch_fails_before_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_invoked(*_args: object, **_kwargs: object) -> None:
        pytest.fail("invalid external artifact reached smoke or build")

    monkeypatch.setattr(
        "tools.performance.run_historical_current_four_path_benchmark._build_native_extension",
        fail_if_invoked,
    )
    monkeypatch.setattr(
        "tools.performance.run_historical_current_four_path_benchmark._smoke_native_extension",
        fail_if_invoked,
    )

    with pytest.raises(ValueError, match="external native extension does not exist"):
        _prepare_native_extension(
            Path("/python"),
            tmp_path / "missing-evidence",
            external_path=tmp_path / "missing.so",
        )

    extension = tmp_path / "existing.so"
    extension.write_bytes(b"hash-mismatch")
    with pytest.raises(ValueError, match="external native extension SHA-256 mismatch"):
        _prepare_native_extension(
            Path("/python"),
            tmp_path / "mismatch-evidence",
            external_path=extension,
            expected_external_sha256="0" * 64,
        )


def test_external_native_extension_rejects_ambiguous_skip_build_option(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        benchmark_main(
            [
                "--native-extension",
                str(tmp_path / "external.so"),
                "--skip-native-build",
            ]
        )

    assert exc_info.value.code == 2
    assert "--native-extension and --skip-native-build are mutually exclusive" in (
        capsys.readouterr().err
    )


def test_skip_native_build_requires_existing_evidence_root_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_build_invoked(*_args: object, **_kwargs: object) -> None:
        pytest.fail("--skip-native-build invoked subprocess build")

    monkeypatch.setattr(
        "tools.performance.run_historical_current_four_path_benchmark._run_capture",
        fail_if_build_invoked,
    )

    with pytest.raises(RuntimeError, match="requires an existing reusable evidence-root extension"):
        _build_native_extension(Path("/python"), tmp_path / "evidence", skip_build=True)

    reusable = (
        tmp_path
        / "evidence"
        / "environment"
        / "eiel_native_runtime_ext_four_path.so"
    )
    reusable.parent.mkdir(parents=True)
    reusable.write_bytes(b"reusable-evidence-root-extension")
    smoke_calls: list[tuple[Path, str]] = []

    def fake_smoke(
        _py: Path,
        extension: Path,
        _evidence_root: Path,
        *,
        artifact_source: str,
    ) -> dict[str, object]:
        smoke_calls.append((extension, artifact_source))
        return {"returncode": 0}

    monkeypatch.setattr(
        "tools.performance.run_historical_current_four_path_benchmark._smoke_native_extension",
        fake_smoke,
    )

    selected = _build_native_extension(Path("/python"), tmp_path / "evidence", skip_build=True)

    assert selected == reusable.resolve()
    assert smoke_calls == [(reusable, "legacy_evidence_root_reuse")]


def test_timed_subprocess_env_clears_inherited_trace_debug_and_parity_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EIEL_TRACE_ACTION_ROWS", "1")
    monkeypatch.setenv("EIEL_NATIVE_RUNTIME_DEBUG_ACTION_ROWS", "0")
    monkeypatch.setenv("EIEL_MEMBERSHIP_PARITY", "1")
    monkeypatch.setenv("EIEL_FUTURE_PARITY_PROBE", "enabled")
    monkeypatch.setenv("EIEL_STATE_DUMP", "1")
    monkeypatch.setenv("EIEL_EVOPY_LEDGER", "1")
    monkeypatch.setenv("EIEL_GUARD_ENV_ALIVE", "1")

    env, record = _timed_subprocess_env(Path("/current"))

    assert "EIEL_TRACE_ACTION_ROWS" not in env
    assert "EIEL_NATIVE_RUNTIME_DEBUG_ACTION_ROWS" not in env
    assert "EIEL_FUTURE_PARITY_PROBE" not in env
    assert "EIEL_STATE_DUMP" not in env
    assert "EIEL_EVOPY_LEDGER" not in env
    assert "EIEL_GUARD_ENV_ALIVE" not in env
    assert env["DEBUG_DETERMINISM"] == "0"
    assert env["EIEL_STRICT_DETERMINISM"] == "0"
    assert env["EIEL_MEMBERSHIP_PARITY"] == "0"
    assert record["ok"] is True
    assert record["enabled_control_values"] == {}
    assert "EIEL_TRACE_ACTION_ROWS" in record["cleared_inherited_keys"]
