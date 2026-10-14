from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simulation.output_cost_timing import FlatOutputCostTiming
from tools.performance.run_historical_current_four_path_benchmark import (
    Arm,
    Cell,
    CURRENT_MATCHED_PUBLIC_OUTPUT_KNOBS,
    EXACT_CURRENT_PUBLIC_OUTPUT_FILES,
    FLAT_OUTPUT_COST_PHASES,
    FLAT_OUTPUT_COST_SAMPLE_SCHEMA,
    FLAT_OUTPUT_COST_SIDECAR_ENV,
    REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES,
    _analyze,
    _arm_specs,
    _cell_command,
    _cells_from_registry,
    _flat_output_cost_attribution_analysis,
    _load_flat_output_cost_sample,
    _load_registry,
    _run_cell,
    _timed_subprocess_env,
    _validate_flat_output_cost_sample,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FOUR_CELL_REGISTRY = (
    PROJECT_ROOT / "registries" / "performance" / "eiel_flat_output_fairness_four_cell.yaml"
)


def _sample(*, scale: int = 1) -> dict[str, object]:
    return {
        "schema_version": FLAT_OUTPUT_COST_SAMPLE_SCHEMA,
        "exclusive_ns": {
            phase: scale * (index + 1) * 10
            for index, phase in enumerate(FLAT_OUTPUT_COST_PHASES)
        },
        "call_counts": {
            phase: index + 1
            for index, phase in enumerate(FLAT_OUTPUT_COST_PHASES)
        },
        "run_total_ns": scale * 1_000,
        "complete": True,
        "errors": [],
    }


def _cell() -> Cell:
    return Cell(
        cell_id="s101_moderate_g4_ckptoff",
        seed=101,
        generations=4,
        steps=80,
        pop=64,
        grid_size=16,
        food_spawn_per_step=10,
        max_food=64,
        churn="moderate",
        checkpoint_mode="off",
        checkpoint_period=0,
    )


def _write_metrics(
    run_dir: Path,
    *,
    generations: int = 4,
    timed_cmd: list[str] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"{generation},64,0,0,0\n" for generation in range(generations))
    (run_dir / "metrics.csv").write_text(
        "generation,alive,successful_births,failed_births,deaths\n" + rows,
        encoding="utf-8",
    )
    for relative_path in REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES:
        path = run_dir / relative_path
        if path == run_dir / "metrics.csv":
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("row\n" if path.suffix == ".csv" else "{}\n", encoding="utf-8")
    status_path = run_dir / "viz" / "run_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    if timed_cmd is not None and "-o" in timed_cmd:
        time_path = Path(timed_cmd[timed_cmd.index("-o") + 1])
        time_path.parent.mkdir(parents=True, exist_ok=True)
        time_path.write_text(
            "User time (seconds): 1.0\n"
            "System time (seconds): 0.1\n"
            "Percent of CPU this job got: 100%\n"
            "Maximum resident set size (kbytes): 1024\n",
            encoding="utf-8",
        )


def test_strict_cost_sample_schema_accepts_only_exact_complete_payload() -> None:
    assert _validate_flat_output_cost_sample(_sample()) == []

    unexpected = _sample()
    unexpected["metadata"] = {}
    assert "unexpected_top_level_keys:metadata" in _validate_flat_output_cost_sample(unexpected)

    missing_phase = _sample()
    del missing_phase["exclusive_ns"]["flush"]  # type: ignore[index]
    errors = _validate_flat_output_cost_sample(missing_phase)
    assert "exclusive_ns_missing_phases:flush" in errors
    assert "exclusive_ns_flush_not_nonnegative_integer" in errors

    over_attributed = _sample()
    over_attributed["run_total_ns"] = 1
    assert "exclusive_ns_sum_exceeds_run_total_ns" in _validate_flat_output_cost_sample(
        over_attributed
    )


def test_producer_and_benchmark_validator_share_exact_phase_contract(tmp_path: Path) -> None:
    timing = FlatOutputCostTiming(tmp_path / "sample.json")
    timing.begin()
    for phase in FLAT_OUTPUT_COST_PHASES:
        with timing.phase(phase):
            pass

    payload = timing.finalize()

    assert set(payload["exclusive_ns"]) == set(FLAT_OUTPUT_COST_PHASES)
    assert set(payload["call_counts"]) == set(FLAT_OUTPUT_COST_PHASES)
    assert _validate_flat_output_cost_sample(payload) == []


def test_incomplete_producer_sample_is_valid_but_fails_required_completion(tmp_path: Path) -> None:
    sample = _sample()
    sample["complete"] = False
    sample["errors"] = ["flush_not_observed"]
    sidecar = tmp_path / "sample.json"
    sidecar.write_text(json.dumps(sample), encoding="utf-8")

    loaded = _load_flat_output_cost_sample(sidecar)

    assert loaded["loaded"] is True
    assert loaded["valid"] is True
    assert loaded["complete"] is False
    assert loaded["errors"] == ["sample_incomplete", "sample_error:flush_not_observed"]


def test_required_current_cell_passes_external_sidecar_and_excludes_it_from_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = Arm("current_w1_reference", tmp_path, 1, False, reference="current")
    cell = _cell()
    rep = "rep1"
    run_id = f"{rep}_{arm.name}_{cell.cell_id}"
    run_dir = tmp_path / "campaign_results" / rep / arm.name / "runs" / run_id
    captured_env: dict[str, str] = {}

    def fake_run(_cmd: list[str], **kwargs: object) -> SimpleNamespace:
        env = dict(kwargs["env"])  # type: ignore[arg-type]
        captured_env.update(env)
        sidecar = Path(env[FLAT_OUTPUT_COST_SIDECAR_ENV])
        assert run_dir not in sidecar.parents
        sidecar.write_text(json.dumps(_sample()), encoding="utf-8")
        _write_metrics(run_dir, timed_cmd=_cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "tools.performance.run_historical_current_four_path_benchmark.subprocess.run",
        fake_run,
    )

    result = _run_cell(
        Path("/python"),
        arm,
        cell,
        rep,
        tmp_path,
        None,
        flat_output_cost_attribution="required",
    )

    sidecar = Path(captured_env[FLAT_OUTPUT_COST_SIDECAR_ENV])
    assert sidecar.is_file()
    assert result["complete"] is True
    assert result["flat_output_cost_attribution"]["complete"] is True
    assert sidecar.name not in result["output_profile"]["relative_files"]
    assert all(
        FLAT_OUTPUT_COST_SIDECAR_ENV not in entry["relative_path"]
        for entry in result["output_profile"]["output_inventory"]
    )


def test_required_current_cell_fails_closed_when_sidecar_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = Arm("current_w1_reference", tmp_path, 1, False, reference="current")
    cell = _cell()
    rep = "rep1"
    run_id = f"{rep}_{arm.name}_{cell.cell_id}"
    run_dir = tmp_path / "campaign_results" / rep / arm.name / "runs" / run_id

    def fake_run(_cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        _write_metrics(run_dir, timed_cmd=_cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "tools.performance.run_historical_current_four_path_benchmark.subprocess.run",
        fake_run,
    )

    result = _run_cell(
        Path("/python"),
        arm,
        cell,
        rep,
        tmp_path,
        None,
        flat_output_cost_attribution="required",
    )

    assert result["returncode"] == 0
    assert result["complete"] is False
    assert result["flat_output_cost_attribution"]["errors"] == ["missing_sidecar"]


def test_historical_required_cell_is_not_applicable_and_receives_no_sidecar_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arm = Arm("historical_w1", tmp_path, 1, False, reference="historical")
    cell = _cell()
    rep = "rep1"
    run_id = f"{rep}_{arm.name}_{cell.cell_id}"
    run_dir = tmp_path / "campaign_results" / rep / arm.name / "runs" / run_id

    def fake_run(_cmd: list[str], **kwargs: object) -> SimpleNamespace:
        assert FLAT_OUTPUT_COST_SIDECAR_ENV not in kwargs["env"]  # type: ignore[operator]
        _write_metrics(run_dir, timed_cmd=_cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        "tools.performance.run_historical_current_four_path_benchmark.subprocess.run",
        fake_run,
    )

    result = _run_cell(
        Path("/python"),
        arm,
        cell,
        rep,
        tmp_path,
        None,
        flat_output_cost_attribution="required",
    )

    assert result["complete"] is True
    assert result["flat_output_cost_attribution"]["applicability"] == "not_applicable"


def test_timed_environment_clears_inherited_cost_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(FLAT_OUTPUT_COST_SIDECAR_ENV, "/tmp/stale-cost-sidecar.json")

    env, record = _timed_subprocess_env(Path("/current"))

    assert FLAT_OUTPUT_COST_SIDECAR_ENV not in env
    assert FLAT_OUTPUT_COST_SIDECAR_ENV in record["cleared_inherited_keys"]


def test_cost_analysis_emits_raw_samples_phase_stats_and_historical_not_applicable() -> None:
    def current_result(rep: str, scale: int) -> dict[str, object]:
        sample = _sample(scale=scale)
        return {
            "rep": rep,
            "cell_id": f"cell-{rep}",
            "flat_output_cost_attribution": {
                "loaded": True,
                "valid": True,
                "complete": True,
                "errors": [],
                "sample": sample,
                "sidecar_path": f"/outside/{rep}.json",
            },
        }

    analysis = _flat_output_cost_attribution_analysis(
        [
            {
                "arm": "historical_w1",
                "reference": "historical",
                "results": [{"rep": "rep1", "cell_id": "historical"}],
            },
            {
                "arm": "current_w1_reference",
                "reference": "current",
                "results": [current_result("rep1", 1), current_result("rep2", 2)],
            },
        ],
        mode="required",
    )

    assert analysis["complete"] is True
    assert analysis["current_samples_expected"] == 2
    assert len(analysis["raw_samples"]) == 2
    assert analysis["arms"]["historical_w1"]["applicability"] == "not_applicable"
    simulation = analysis["arms"]["current_w1_reference"]["phases"]["simulation"]
    assert simulation["exclusive_ns"]["median"] == 15.0
    assert simulation["call_counts"]["median"] == 1.0


def test_required_analysis_fails_closed_on_missing_current_sample(tmp_path: Path) -> None:
    campaign = {
        "rep": "rep1",
        "arm": "current_w1_reference",
        "reference": "current",
        "wall_s": 1.0,
        "cells_per_second": 1.0,
        "steps_per_second": 1.0,
        "output_bytes": 0,
        "checkpoint_bytes": 0,
        "peak_cell_rss_kb": 1,
        "sum_cell_peak_rss_kb": 1,
        "child_user_time_s": 1.0,
        "child_system_time_s": 0.0,
        "child_cpu_time_s": 1.0,
        "worker_utilization_ratio": 1.0,
        "resource_metrics_complete": True,
        "incomplete_cells": 0,
        "any_hidden_fallback": False,
        "any_native_fail_closed": False,
        "results": [{"rep": "rep1", "cell_id": "missing", "output_profile": {}}],
    }

    analysis = _analyze(
        [campaign],
        tmp_path,
        required_repetitions_by_arm={"current_w1_reference": 1},
        flat_output_cost_attribution="required",
    )

    assert "current_flat_output_cost_attribution_incomplete" in analysis["fail_closed_reasons"]
    artifact = json.loads(
        (tmp_path / "analysis" / "flat_output_cost_attribution.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact["complete"] is False
    assert artifact["errors"][0]["errors"] == ["missing_cell_attribution_record", "sample_not_object"]


def test_explicit_four_cell_registry_is_exact_and_non_cartesian() -> None:
    registry = _load_registry(FOUR_CELL_REGISTRY)
    cells = _cells_from_registry(registry)

    assert registry["matrix"] == {
        "cells": [
            {
                "seed": 101,
                "churn": {
                    "name": "moderate",
                    "food_spawn_per_step": 10,
                    "max_food": 64,
                },
                "workload": {"name": "g4", "generations": 4},
                "checkpoint": {
                    "name": "off",
                    "checkpoint_bundle_v2_period_gen": 0,
                },
            },
            {
                "seed": 202,
                "churn": {
                    "name": "high",
                    "food_spawn_per_step": 16,
                    "max_food": 96,
                },
                "workload": {"name": "g4", "generations": 4},
                "checkpoint": {
                    "name": "off",
                    "checkpoint_bundle_v2_period_gen": 0,
                },
            },
            {
                "seed": 303,
                "churn": {
                    "name": "moderate",
                    "food_spawn_per_step": 10,
                    "max_food": 64,
                },
                "workload": {"name": "g4", "generations": 4},
                "checkpoint": {
                    "name": "every_gen",
                    "checkpoint_bundle_v2_period_gen": 1,
                },
            },
            {
                "seed": 1001,
                "churn": {
                    "name": "high",
                    "food_spawn_per_step": 16,
                    "max_food": 96,
                },
                "workload": {"name": "g4", "generations": 4},
                "checkpoint": {
                    "name": "every_gen",
                    "checkpoint_bundle_v2_period_gen": 1,
                },
            },
        ]
    }
    assert [cell.__dict__ for cell in cells] == [
        {
            "cell_id": "s101_moderate_g4_ckptoff",
            "seed": 101,
            "generations": 4,
            "steps": 80,
            "pop": 64,
            "grid_size": 16,
            "food_spawn_per_step": 10,
            "max_food": 64,
            "churn": "moderate",
            "checkpoint_mode": "off",
            "checkpoint_period": 0,
        },
        {
            "cell_id": "s202_high_g4_ckptoff",
            "seed": 202,
            "generations": 4,
            "steps": 80,
            "pop": 64,
            "grid_size": 16,
            "food_spawn_per_step": 16,
            "max_food": 96,
            "churn": "high",
            "checkpoint_mode": "off",
            "checkpoint_period": 0,
        },
        {
            "cell_id": "s303_moderate_g4_ckptevery_gen",
            "seed": 303,
            "generations": 4,
            "steps": 80,
            "pop": 64,
            "grid_size": 16,
            "food_spawn_per_step": 10,
            "max_food": 64,
            "churn": "moderate",
            "checkpoint_mode": "every_gen",
            "checkpoint_period": 1,
        },
        {
            "cell_id": "s1001_high_g4_ckptevery_gen",
            "seed": 1001,
            "generations": 4,
            "steps": 80,
            "pop": 64,
            "grid_size": 16,
            "food_spawn_per_step": 16,
            "max_food": 96,
            "churn": "high",
            "checkpoint_mode": "every_gen",
            "checkpoint_period": 1,
        },
    ]
    assert set(_arm_specs(registry)) == {
        "current_w1_reference",
        "current_w6_reference",
        "current_w1_native",
        "current_w6_native",
    }


def test_four_cell_registry_applies_identical_accepted_route_knobs_to_every_cell() -> None:
    registry = _load_registry(FOUR_CELL_REGISTRY)
    cells = _cells_from_registry(registry)
    arms = _arm_specs(registry)
    route_knobs = tuple(registry["benchmark_envelope"]["current_matched_route_knobs"])

    assert route_knobs == CURRENT_MATCHED_PUBLIC_OUTPUT_KNOBS
    assert tuple(registry["benchmark_envelope"]["required_public_outputs"]) == (
        REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES
    )
    assert tuple(registry["benchmark_envelope"]["byte_exact_public_outputs"]) == (
        EXACT_CURRENT_PUBLIC_OUTPUT_FILES
    )
    assert {
        "sim_cfg.developmental_trace_write_enable=false",
        "sim_cfg.phase_e_e1_enable=false",
        "sim_cfg.a3_recovery_taxonomy_enable=false",
    } <= set(route_knobs)
    assert "sim_cfg.lineage_observability_enable=true" in route_knobs
    assert "metrics/lineage_descriptors.jsonl" in REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES
    assert "metrics/lineage_descriptors.jsonl" in EXACT_CURRENT_PUBLIC_OUTPUT_FILES
    assert set(EXACT_CURRENT_PUBLIC_OUTPUT_FILES) <= set(REQUIRED_CURRENT_PUBLIC_OUTPUT_FILES)

    observed_profiles: set[tuple[str, ...]] = set()
    for arm in arms.values():
        assert arm.reference == "current"
        for cell in cells:
            command = _cell_command(
                Path("/python"),
                arm,
                cell,
                f"{arm.name}_{cell.cell_id}",
                Path("/runs"),
            )
            command_knobs = [
                command[index + 1]
                for index, argument in enumerate(command[:-1])
                if argument == "--knob"
            ]
            for expected in route_knobs:
                key = expected.split("=", 1)[0]
                assert [knob for knob in command_knobs if knob.startswith(f"{key}=")] == [
                    expected
                ]
            observed_profiles.add(
                tuple(knob for knob in command_knobs if knob in route_knobs)
            )

    assert observed_profiles == {route_knobs}


def test_cartesian_registry_generation_remains_supported() -> None:
    registry = {
        "benchmark_envelope": {
            "population": 64,
            "grid_size": 16,
            "steps_per_generation": 80,
        },
        "matrix": {
            "seeds": [1, 2],
            "churn_levels": [
                {"name": "moderate", "food_spawn_per_step": 10, "max_food": 64}
            ],
            "workload_lengths": [{"name": "g4", "generations": 4}],
            "checkpoint_modes": [
                {"name": "off", "checkpoint_bundle_v2_period_gen": 0},
                {"name": "every_gen", "checkpoint_bundle_v2_period_gen": 1},
            ],
        },
    }

    cells = _cells_from_registry(copy.deepcopy(registry))

    assert [cell.cell_id for cell in cells] == [
        "s1_moderate_g4_ckptoff",
        "s1_moderate_g4_ckptevery_gen",
        "s2_moderate_g4_ckptoff",
        "s2_moderate_g4_ckptevery_gen",
    ]
