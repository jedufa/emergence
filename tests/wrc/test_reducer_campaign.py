from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

from engine.wrc.reducer.campaign import reduce_input


def _write_metrics(path: Path, values: list[float], *, readback: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generation", "switch", "activation_signal", "alive"])
        for generation, value in enumerate(values):
            writer.writerow([generation, readback, value, 100.0 - value])


def _row_tree(root: Path) -> Path:
    row = root / "rows" / "run-PTEST"
    launcher = row / "_launcher"
    launcher.mkdir(parents=True)
    selected = {
        "ID": "PTEST",
        "Parameter": "env_cfg.switch",
        "Type": "bool",
        "Reference default": "false",
        "Family": "test",
        "World base": "patch",
        "World overlays JSON": "[]",
        "Launcher profile": "STANDARD_CUDA",
        "Primary response metrics": "activation signal; alive",
    }
    jobs = []
    for level in (False, True):
        for seed in (101, 202):
            run_id = f"ptest-{str(level).lower()}-{seed}"
            jobs.append({"level": level, "seed": seed, "run_id": run_id})
            _write_metrics(
                row / "full" / run_id / "metrics.csv",
                [0.0, 0.0, 0.0] if not level else [1.0, 2.0, 3.0],
                readback=1.0 if level else 0.0,
            )
    plan = {
        "schema_version": "eiel_wrc_serial_cpu_launch_v1",
        "id": "PTEST",
        "parameter": "env_cfg.switch",
        "runtime_parameter": "env_cfg.switch",
        "type": "bool",
        "levels": [False, True],
        "seeds": [101, 202],
        "planned_runs": 4,
        "launcher_profile": "STANDARD_CUDA",
        "world_overlays": [],
        "challenge": {"assertions": [{"metric": "activation_signal"}]},
        "full": {"generations": 3, "ticks_per_generation": 10, "jobs": jobs},
    }
    (launcher / "selected_row.json").write_text(json.dumps(selected))
    (launcher / "launch_plan.json").write_text(json.dumps(plan))
    return row


def test_directory_and_zip_campaign_reduction_match(tmp_path: Path) -> None:
    atlas = tmp_path / "atlas"
    _row_tree(atlas)

    directory_output = tmp_path / "out-directory"
    directory_result = reduce_input(atlas, output_dir=directory_output)

    archive = tmp_path / "atlas.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in atlas.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(atlas.parent).as_posix())
    zip_output = tmp_path / "out-zip"
    zip_result = reduce_input(archive, output_dir=zip_output)

    assert directory_result["failed_row_count"] == 0
    assert zip_result["failed_row_count"] == 0
    directory_summary = json.loads(
        (directory_output / "rows" / "PTEST" / "parameter_summary.json").read_text()
    )
    zip_summary = json.loads(
        (zip_output / "rows" / "PTEST" / "parameter_summary.json").read_text()
    )
    assert directory_summary["result_sha256"] == zip_summary["result_sha256"]
    assert (directory_output / "wrc_master.csv").is_file()
    assert (directory_output / "wrc_master.md").is_file()
    assert (directory_output / "rows" / "PTEST" / "parameter_card.json").is_file()
    assert (directory_output / "rows" / "PTEST" / "parameter_card.md").is_file()
    assert (directory_output / "rows" / "PTEST" / "raw_row_index.csv").is_file()
    card = json.loads(
        (directory_output / "rows" / "PTEST" / "parameter_card.json").read_text()
    )
    assert card["summary"]["control_transport"]["status"] == "CONFIRMED"
    with (directory_output / "wrc_master.csv").open(newline="", encoding="utf-8") as handle:
        master = next(csv.DictReader(handle))
    assert master["control_transport_status"] == "CONFIRMED"
    assert master["top_metric"] == "activation_signal"
    with (directory_output / "wrc_family_summaries.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        family = next(csv.DictReader(handle))
    assert family["family"] == "test"
    assert family["parameter_count"] == "1"
    assert family["active_count"] == "1"


def test_atlas_artifacts_do_not_depend_on_output_root(tmp_path: Path) -> None:
    atlas = tmp_path / "atlas"
    _row_tree(atlas)

    output_a = tmp_path / "output-a"
    output_b = tmp_path / "somewhere-else" / "output-b"
    reduce_input(atlas, output_dir=output_a)
    reduce_input(atlas, output_dir=output_b)

    deterministic_artifacts = (
        "wrc_master.csv",
        "wrc_master.md",
        "atlas_reduction_status.json",
        "atlas_reduction_manifest.json",
        "wrc_family_summaries.csv",
        "wrc_family_summaries.json",
        "wrc_family_summaries.md",
    )
    for name in deterministic_artifacts:
        assert (output_a / name).read_bytes() == (output_b / name).read_bytes()

    with (output_a / "wrc_master.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        master = next(csv.DictReader(handle))
    assert master["parameter_card_path"] == "rows/PTEST/parameter_card.json"
    assert master["parameter_card_markdown_path"] == "rows/PTEST/parameter_card.md"
    assert master["parameter_summary_path"] == "rows/PTEST/parameter_summary.json"

    status = json.loads((output_a / "atlas_reduction_status.json").read_text())
    assert status["rows"][0]["output_dir"] == "rows/PTEST"
