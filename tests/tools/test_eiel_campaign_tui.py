from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path


def _load_tui_module():
    path = Path("tools/scripts/eiel_campaign_tui.py").resolve()
    spec = importlib.util.spec_from_file_location("eiel_campaign_tui", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_progress_line_reads_elapsed_avg_and_eta() -> None:
    tui = _load_tui_module()

    sample = tui.parse_progress_line("[gen 1/8] elapsed 00:00:02.96 | avg/gen 2.96s | ETA 00:00:20")

    assert sample.done == 1
    assert sample.total == 8
    assert sample.elapsed_s == 2.96
    assert sample.avg_gen_s == 2.96
    assert sample.eta_s == 20


def test_parse_progress_line_clamps_negative_time_to_unknown() -> None:
    tui = _load_tui_module()

    sample = tui.parse_progress_line("[gen 2/8] elapsed -00:00:01 | avg/gen 2.50s | ETA -00:00:03")

    assert sample.done == 2
    assert sample.elapsed_s is None
    assert sample.eta_s is None


def test_snapshot_aggregates_manifest_command_log_and_stdout(tmp_path: Path) -> None:
    tui = _load_tui_module()
    runs_root = tmp_path / "campaign"
    logs = runs_root / "logs"
    logs.mkdir(parents=True)
    stdout1 = logs / "child_0001_a.stdout.txt"
    stderr1 = logs / "child_0001_a.stderr.txt"
    stdout2 = logs / "child_0002_b.stdout.txt"
    stderr2 = logs / "child_0002_b.stderr.txt"
    stdout1.write_text("[gen 3/8] elapsed 00:00:09 | avg/gen 3.00s | ETA 00:00:15\n", encoding="utf-8")
    stdout2.write_text("[gen 8/8] elapsed 00:00:24 | avg/gen 3.00s | ETA 00:00:00\nRun complete.\n", encoding="utf-8")
    manifest = {
        "schema_version": "eiel_parallel_campaign_live_manifest_v1",
        "runs_root": str(runs_root),
        "workers": 2,
        "planned_runs": 3,
        "command_log_path": str(runs_root / "parallel_command_log.jsonl"),
        "status_csv_path": str(runs_root / "parallel_child_status.csv"),
        "children": [
            {
                "index": 1,
                "worker_slot": 0,
                "label": "a",
                "run_id": "run_a",
                "stdout_path": str(stdout1),
                "stderr_path": str(stderr1),
            },
            {
                "index": 2,
                "worker_slot": 1,
                "label": "b",
                "run_id": "run_b",
                "stdout_path": str(stdout2),
                "stderr_path": str(stderr2),
            },
            {
                "index": 3,
                "worker_slot": 0,
                "label": "c",
                "run_id": "run_c",
                "stdout_path": str(logs / "child_0003_c.stdout.txt"),
                "stderr_path": str(logs / "child_0003_c.stderr.txt"),
            },
        ],
    }
    (runs_root / "parallel_live_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (runs_root / "parallel_command_log.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"kind": "child_start", "index": 1}),
                json.dumps({"kind": "child_start", "index": 2}),
                json.dumps({"kind": "child_end", "index": 2, "returncode": 0, "wall_s": 24.0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (runs_root / "parallel_child_status.csv").write_text("index,status,wall_s,returncode\n2,ok,24.0,0\n", encoding="utf-8")
    now = time.time()
    for path in (stdout1, stdout2):
        path.touch()

    snapshot = tui.load_campaign_snapshot(runs_root, stale_after_s=120.0)
    children = {child.index: child for child in snapshot["children"]}

    assert now <= time.time()
    assert children[1].status == "running"
    assert children[1].progress.done == 3
    assert children[2].status == "complete"
    assert children[2].wall_s == 24.0
    assert children[3].status == "queued"
    assert snapshot["planned_runs"] == 3
    assert snapshot["overall_eta_s"] is not None


def test_render_plain_includes_worker_bars_eta_and_avg(tmp_path: Path) -> None:
    tui = _load_tui_module()
    runs_root = tmp_path / "campaign"
    logs = runs_root / "logs"
    logs.mkdir(parents=True)
    stdout = logs / "child_0001_a.stdout.txt"
    stdout.write_text("[gen 1/4] elapsed 00:00:02.50 | avg/gen 2.50s | ETA 00:00:07\n", encoding="utf-8")
    manifest = {
        "schema_version": "eiel_parallel_campaign_live_manifest_v1",
        "runs_root": str(runs_root),
        "workers": 1,
        "planned_runs": 1,
        "command_log_path": str(runs_root / "parallel_command_log.jsonl"),
        "status_csv_path": str(runs_root / "parallel_child_status.csv"),
        "children": [
            {
                "index": 1,
                "worker_slot": 0,
                "label": "a",
                "run_id": "run_a",
                "stdout_path": str(stdout),
                "stderr_path": str(logs / "child_0001_a.stderr.txt"),
            }
        ],
    }
    (runs_root / "parallel_live_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (runs_root / "parallel_command_log.jsonl").write_text(json.dumps({"kind": "child_start", "index": 1}) + "\n", encoding="utf-8")

    rendered = tui.render_plain(tui.load_campaign_snapshot(runs_root))

    assert "w00" in rendered
    assert "gen   1/4" in rendered
    assert "avg/gen 2.50s" in rendered
    assert "ETA 00:00:07" in rendered


def test_snapshot_can_fall_back_to_command_log_without_manifest(tmp_path: Path) -> None:
    tui = _load_tui_module()
    runs_root = tmp_path / "campaign"
    logs = runs_root / "logs"
    logs.mkdir(parents=True)
    stdout = logs / "child_0001_a.stdout.txt"
    stderr = logs / "child_0001_a.stderr.txt"
    stdout.write_text("[gen 2/4] elapsed 00:00:05 | avg/gen 2.50s | ETA 00:00:05\n", encoding="utf-8")
    (runs_root / "parallel_command_log.jsonl").write_text(
        json.dumps(
            {
                "kind": "child_start",
                "index": 1,
                "worker_slot": 0,
                "label": "a",
                "run_id": "run_a",
                "stdout_path": str(stdout),
                "stderr_path": str(stderr),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (runs_root / "parallel_campaign_summary.json").write_text(
        json.dumps({"planned_runs": 1, "workers": 1}),
        encoding="utf-8",
    )

    snapshot = tui.load_campaign_snapshot(runs_root)

    assert len(snapshot["children"]) == 1
    assert snapshot["children"][0].status == "running"
    assert snapshot["children"][0].progress.done == 2
