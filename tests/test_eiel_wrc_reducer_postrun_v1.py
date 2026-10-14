from __future__ import annotations

import csv
import json
from pathlib import Path

from engine.wrc.reducer import ReducerConfig
from engine.wrc.reducer.aggregate import classify_response_curve
from engine.wrc.reducer.campaign import reduce_input


def _write_metrics(path: Path, *, offset: float, readback: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["generation", "test_enable", "mechanism", "alive", "text"],
        )
        writer.writeheader()
        for generation in range(5):
            writer.writerow(
                {
                    "generation": generation,
                    "test_enable": readback,
                    "mechanism": offset + generation,
                    "alive": 100 + offset - generation,
                    "text": "ignored",
                }
            )


def _atlas_root(tmp_path: Path) -> Path:
    root = tmp_path / "atlas"
    row_root = root / "rows" / "20260807-P001"
    launcher = row_root / "_launcher"
    launcher.mkdir(parents=True)

    levels = [False, True]
    seeds = [101, 202]
    jobs = []
    ordinal = 0
    for level in levels:
        for seed in seeds:
            ordinal += 1
            run_id = f"ptest-level-{str(level).lower()}-seed-{seed}"
            _write_metrics(
                row_root / "full" / run_id / "metrics.csv",
                offset=10.0 if level else 0.0,
                readback=1.0 if level else 0.0,
            )
            jobs.append(
                {
                    "ordinal": ordinal,
                    "level": level,
                    "seed": seed,
                    "run_id": run_id,
                }
            )

    plan = {
        "schema_version": "eiel_wrc_serial_cpu_launch_v1",
        "id": "P001",
        "parameter": "env_cfg.test_enable",
        "runtime_parameter": "env_cfg.test_enable",
        "type": "bool",
        "levels": levels,
        "seeds": seeds,
        "planned_runs": 4,
        "launcher_profile": "STANDARD_CUDA",
        "world_overlays": [],
        "fixed_required_overrides": {},
        "fixed_activation_overrides": {},
        "implicit_context_overrides": {},
        "challenge": {
            "assertions": [
                {
                    "metric": "mechanism",
                    "aggregate": "max",
                    "operator": ">",
                    "value": 0,
                }
            ]
        },
        "full": {
            "generations": 5,
            "ticks_per_generation": 10,
            "jobs": jobs,
        },
    }
    selected = {
        "ID": "P001",
        "Parameter": "env_cfg.test_enable",
        "Family": "Test ecology",
        "Type": "bool",
        "Reference default": "false",
        "Levels JSON": '["false","true"]',
        "Seeds JSON": "[101,202]",
        "Planned runs": "4",
        "Primary response metrics": "mechanism; alive",
        "Expected timescale": "SHORT",
        "Launcher profile": "STANDARD_CUDA",
        "World base": "patch",
        "World overlays JSON": "[]",
        "Preflight assertions JSON": json.dumps(plan["challenge"]["assertions"]),
    }
    (launcher / "launch_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (launcher / "selected_row.json").write_text(
        json.dumps(selected), encoding="utf-8"
    )
    return root


def test_campaign_reduction_reduces_paired_levels_and_classifies_response(
    tmp_path: Path,
) -> None:
    root = _atlas_root(tmp_path)
    output = tmp_path / "reduced"

    atlas = reduce_input(
        root,
        output_dir=output,
        config=ReducerConfig(),
    )

    assert atlas["row_result_count"] == 1
    assert atlas["failed_row_count"] == 0
    parameter = json.loads(
        (output / "rows" / "P001" / "parameter_summary.json").read_text()
    )
    assert parameter["coverage"]["complete_worlds"] == 4
    assert parameter["screening"]["classification"] == "ACTIVE"
    assert parameter["screening"]["top_metric"] == "mechanism"
    with (output / "rows" / "P001" / "metric_effects.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        mechanism = next(
            row
            for row in csv.DictReader(handle)
            if row["metric"] == "mechanism" and row["feature"] == "terminal"
    )
    assert float(mechanism["mean_delta"]) == 10.0
    assert float(mechanism["seed_direction_consistency"]) == 1.0
    assert int(mechanism["paired_seed_count"]) == 2
    assert 0.0 < float(mechanism["scientific_confidence_weight"]) < 1.0


def test_output_bundle_is_compact_hashed_and_points_back_to_raw_rows(
    tmp_path: Path,
) -> None:
    source = _atlas_root(tmp_path)
    output = tmp_path / "reduced"

    reduce_input(
        source,
        output_dir=output,
        config=ReducerConfig(),
    )

    for name in (
        "atlas_reduction_manifest.json",
        "atlas_reduction_status.json",
        "wrc_master.csv",
        "wrc_master.md",
        "wrc_family_summaries.json",
        "rows/P001/parameter_summary.json",
        "rows/P001/parameter_card.md",
        "rows/P001/raw_row_index.csv",
        "rows/P001/selected_points.csv",
    ):
        assert (output / name).is_file(), name

    manifest = json.loads((output / "atlas_reduction_manifest.json").read_text())
    assert manifest["manifest_sha256"]
    with (output / "rows" / "P001" / "selected_points.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    mechanism_rows = [row for row in rows if row["metric"] == "mechanism"]
    assert mechanism_rows[0]["source_row_number"] == "2"
    assert mechanism_rows[-1]["source_row_number"] == "6"


def test_numeric_response_curve_vocabulary_is_deterministic() -> None:
    def shape(values: list[float], *, config: ReducerConfig | None = None) -> str:
        rows = [
            {
                "level": float(index),
                "mean": float(value),
                "standard_deviation": 0.0,
                "seed_count": 2,
            }
            for index, value in enumerate(values)
        ]
        return classify_response_curve(
            rows,
            domain="other",
            config=config or ReducerConfig(),
        )["shape"]

    assert shape([0.0, 1.0, 2.0, 3.0]) == "MONOTONIC_POSITIVE"
    assert (
        shape(
            [0.0, 1.0, 1.3, 1.35],
            config=ReducerConfig(threshold_step_fraction=1.1),
        )
        == "SATURATING_POSITIVE"
    )
    assert shape([1.0, 0.2, 0.3, 1.1]) == "U_SHAPED"
