from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from tests.evidence.test_schema_checker import METRIC_COLUMNS, build_cogp1_fixture


def rewrite_metrics_for_signal(manifest_path: Path, *, signal: str) -> Path:
    run_root = manifest_path.parent / "runs"
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    for run in summary["runs"]:
        axes = run["axes"]
        seed = int(run["seed"])
        ecology = axes["ecology_context"]
        perception = axes["perception_condition"]
        pressure = ecology != "cogp1_flat_control_clean"
        if signal == "supported":
            alive_delta = 6 if pressure else 1
            births_delta = 0.6 if pressure else 0.1
        elif signal == "none":
            alive_delta = 0
            births_delta = 0.0
        else:
            raise ValueError(signal)
        base_alive = 100 + seed % 13
        base_births = 2.0
        normal_bonus = alive_delta if perception == "perception_normal" else 0
        birth_bonus = births_delta if perception == "perception_normal" else 0.0
        row = {column: "0" for column in METRIC_COLUMNS}
        row.update(
            {
                "generation": "15",
                "alive": str(base_alive + normal_bonus),
                "mean_offspring_total": str(base_births + birth_bonus),
                "lineage_effective_alive_lineages": "4.0",
                "lineage_alive_share_entropy": "1.0",
                "lineage_active_count": "4",
                "founder_initial_effective_lineages": "8",
                "founder_initial_top1_share": "0.2",
                "perception_cost_charged_mean": "0.01",
                "sensor_radius_cost_charged_mean": "0.01",
                "sensor_channel_active_count_mean": "3",
                "sensor_channel_force_minimal_active_flag": "1" if perception == "perception_ablated" else "0",
                "sensor_radius_force_default_active_flag": "1",
                "behavior_action_total": "10",
                "behavior_patch_interaction_proxy": str(2.0 if pressure else 1.0),
                "behavior_risk_exposure_proxy": str(3.0 if pressure else 1.0),
                "prediction_direct_selection_pressure_disabled_flag": "1",
            }
        )
        metrics = Path(run["run_dir"]) / "metrics.csv"
        with metrics.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS)
            writer.writeheader()
            writer.writerow(row)
    return manifest_path


def cogp1_manifest(tmp_path: Path, *, signal: str = "supported") -> Path:
    manifest_path, _runs = build_cogp1_fixture(tmp_path)
    return rewrite_metrics_for_signal(manifest_path, signal=signal)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
