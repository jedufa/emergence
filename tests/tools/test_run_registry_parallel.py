from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path


def _load_wrapper_module():
    path = Path("tools/scripts/run_registry_parallel.py").resolve()
    spec = importlib.util.spec_from_file_location("run_registry_parallel", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_registry(path: Path) -> str:
    run_id = "parallel_fixture__scenario_s00__seed000111"
    path.write_text(
        """
harness_api: 1
version: 1
name: parallel_fixture
description: "parallel wrapper fixture"
intent:
  question: "fixture"
  axes:
    - scenario:
        - s00
matrix_order:
  - scenario
  - seed
axes:
  scenario:
    values:
      - s00
    flags:
      s00:
        - --knob
        - env_cfg.patch_world_enable=false
seeds:
  - 111
cli_defaults:
  - - --generations
    - "1"
  - - --steps
    - "1"
  - - --pop
    - "4"
  - - --grid_size
    - "4"
  - - --device
    - cpu
  - - --bank_device
    - cpu
naming:
  run_id_format: "{name}__scenario_{scenario}__seed{seed:06d}"
  run_tag_prefix: "parallel_fixture"
  run_tag_format: "{run_tag_prefix}_{scenario}_seed{seed:06d}"
expected_artifacts:
  - manifest.json
  - run_config.csv
""".lstrip(),
        encoding="utf-8",
    )
    return run_id


def test_parallel_wrapper_dry_run_plans_selected_run(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    run_id = _write_registry(registry)
    runs_root = tmp_path / "runs"

    proc = subprocess.run(
        [
            sys.executable,
            "tools/scripts/run_registry_parallel.py",
            "--registry",
            str(registry),
            "--runs-root",
            str(runs_root),
            "--batch-tag",
            "fixture",
            "--workers",
            "2",
            "--only-run-id",
            run_id,
            "--performance-profile",
            "accepted_w6_fast_v1",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads((runs_root / "parallel_campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "dry_run"
    assert summary["workers"] == 2
    assert summary["worker_affinity_mode"] == "none"
    assert summary["performance_profile"] == "accepted_w6_fast_v1"
    assert summary["planned_runs"] == 1
    assert summary["selected_run_ids"] == [run_id]
    assert Path(summary["plan"]["matrix_path"]).exists()
    command_log = (runs_root / "parallel_command_log.jsonl").read_text(encoding="utf-8")
    assert "--performance_profile" in command_log
    assert "accepted_w6_fast_v1" in command_log


def test_parallel_wrapper_invalid_run_id_fails_with_summary(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    _write_registry(registry)
    runs_root = tmp_path / "runs"

    proc = subprocess.run(
        [
            sys.executable,
            "tools/scripts/run_registry_parallel.py",
            "--registry",
            str(registry),
            "--runs-root",
            str(runs_root),
            "--batch-tag",
            "fixture",
            "--only-run-id",
            "missing_run_id",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    summary = json.loads((runs_root / "parallel_campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed_planning"
    assert summary["invalid_run_ids"] == ["missing_run_id"]
    assert summary["failures"][0]["status"] == "invalid_run_id"


def test_worker_affinity_compact_and_spread_plans_cpu_sets() -> None:
    wrapper = _load_wrapper_module()

    compact = wrapper._resolve_worker_cpu_sets(
        mode="compact",
        workers=3,
        manual_sets=[],
        available_cpus=tuple(range(8)),
    )
    spread = wrapper._resolve_worker_cpu_sets(
        mode="spread",
        workers=3,
        manual_sets=[],
        available_cpus=tuple(range(8)),
    )

    assert compact["worker_cpu_sets"] == [[0, 1], [2, 3], [4, 5]]
    assert spread["worker_cpu_sets"] == [[0, 4], [1, 5], [2, 6]]


def test_worker_affinity_manual_parses_ranges_and_rejects_bad_counts() -> None:
    wrapper = _load_wrapper_module()

    manual = wrapper._resolve_worker_cpu_sets(
        mode="manual",
        workers=3,
        manual_sets=["0-1", "2,3", "4"],
        available_cpus=tuple(range(8)),
    )
    assert manual["worker_cpu_sets"] == [[0, 1], [2, 3], [4]]

    try:
        wrapper._resolve_worker_cpu_sets(
            mode="manual",
            workers=3,
            manual_sets=["0", "1"],
            available_cpus=tuple(range(8)),
        )
    except ValueError as exc:
        assert "exactly one CPU set per worker" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("manual affinity should reject missing worker CPU sets")


def test_worker_affinity_spread_uses_single_cpus_when_workers_exceed_core_pairs() -> None:
    wrapper = _load_wrapper_module()

    spread = wrapper._resolve_worker_cpu_sets(
        mode="spread",
        workers=5,
        manual_sets=[],
        available_cpus=tuple(range(8)),
    )

    assert spread["worker_cpu_sets"] == [[0], [4], [1], [5], [2]]


def test_live_manifest_records_child_log_paths(tmp_path: Path) -> None:
    wrapper = _load_wrapper_module()
    runs_root = tmp_path / "runs"
    logs_dir = runs_root / "logs"
    command_log = runs_root / "parallel_command_log.jsonl"
    summary_json = runs_root / "parallel_campaign_summary.json"
    status_csv = runs_root / "parallel_child_status.csv"
    manifest_path = runs_root / "parallel_live_manifest.json"
    args = type("Args", (), {"batch_tag": "fixture", "workers": 2})()
    spec = wrapper.ChildSpec(
        index=1,
        label="Fixture Run",
        run_id="fixture_run_id",
        registry="registry.yaml",
        child_root=str(runs_root / "children" / "0001_fixture"),
        batch_tag="fixture_0001",
        knobs=(),
        performance_profile="",
        campaign_output_profile="compact_default",
        allow_large_output=False,
        allow_generation_checkpointing=False,
        perf=False,
        overwrite=False,
        command_log=str(command_log),
        logs_dir=str(logs_dir),
        worker_slot=0,
    )

    wrapper._write_live_manifest(
        manifest_path,
        args=args,
        runs_root=runs_root,
        command_log=command_log,
        summary_json=summary_json,
        status_csv=status_csv,
        specs=[spec],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    child = manifest["children"][0]
    assert manifest["schema_version"] == "eiel_parallel_campaign_live_manifest_v1"
    assert manifest["planned_runs"] == 1
    assert child["worker_slot"] == 0
    assert child["stdout_path"].endswith("child_0001_fixture_run.stdout.txt")


def test_aggregate_only_rebuilds_summary_without_registry_or_child_execution(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    child_root = runs_root / "children" / "0001_fixture"
    run_dir = child_root / "e" / "commit" / "batch" / "r" / "run001"
    run_dir.mkdir(parents=True)
    summary_path = child_root / "e" / "commit" / "batch" / "summary.json"
    summary_path.write_text(json.dumps({"runs": [{"run_dir": str(run_dir), "status": "ok"}]}) + "\n", encoding="utf-8")
    (run_dir / "metrics.csv").write_text(
        "generation,alive,deaths,successful_births,failed_births,lineage_active_count,lineage_top1_alive_share,lineage_effective_alive_lineages\n"
        "0,4,0,1,0,2,0.5,2\n"
        "1,7,0,3,0,3,0.4,2.5\n",
        encoding="utf-8",
    )
    (run_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    (run_dir / "run_config.csv").write_text("key,value\n", encoding="utf-8")
    (run_dir / "viz").mkdir()
    (run_dir / "viz" / "checkpoints_index.json").write_text("{}\n", encoding="utf-8")
    extra_root = runs_root / "children" / "0002_extra"
    extra_run_dir = extra_root / "e" / "commit" / "batch" / "r" / "run002"
    extra_run_dir.mkdir(parents=True)
    (extra_root / "e" / "commit" / "batch" / "summary.json").write_text(
        json.dumps({"runs": [{"run_dir": str(extra_run_dir), "status": "ok"}]}) + "\n",
        encoding="utf-8",
    )
    (extra_run_dir / "metrics.csv").write_text(
        "generation,alive,deaths,successful_births,failed_births,lineage_active_count,lineage_top1_alive_share,lineage_effective_alive_lineages\n"
        "0,2,0,1,0,1,1.0,1\n",
        encoding="utf-8",
    )
    (runs_root / "parallel_live_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "eiel_parallel_campaign_live_manifest_v1",
                "children": [
                    {
                        "index": 1,
                        "worker_slot": 0,
                        "label": "fixture",
                        "run_id": "fixture_run",
                        "registry": "registry.yaml",
                        "child_root": str(child_root),
                        "batch_tag": "fixture_0001",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            "tools/scripts/run_registry_parallel.py",
            "--runs-root",
            str(runs_root),
            "--batch-tag",
            "fixture",
            "--aggregate-only",
            "--campaign-index",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    summary = json.loads((runs_root / "parallel_campaign_summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate_mode"]["enabled"] is True
    assert summary["status"] == "complete"
    assert summary["planned_runs"] == 2
    assert summary["results"][0]["status"] == "ok"
    assert summary["results"][0]["final_metrics"]["alive"] == "7"
    assert summary["results"][0]["metrics_row_count"] == 2
    assert summary["results"][1]["run_id"] == "0002_extra"
    assert summary["campaign_index"]["counters"]["campaign_index_entries_added"] == 2
    index = json.loads((runs_root / "campaign_run_index_v1.json").read_text(encoding="utf-8"))
    assert index["schema_version"] == "eiel_campaign_run_index_v1"
    assert index["entries"][0]["run_id"] == "fixture_run"


def test_child_start_command_log_includes_live_log_paths(tmp_path: Path, monkeypatch) -> None:
    wrapper = _load_wrapper_module()
    command_log = tmp_path / "parallel_command_log.jsonl"
    logs_dir = tmp_path / "logs"
    child_root = tmp_path / "child"
    monkeypatch.setattr(wrapper, "_run_capture", lambda *args, **kwargs: (0, 1.25))
    monkeypatch.setattr(wrapper, "_find_latest_summary", lambda *args, **kwargs: None)

    result = wrapper._run_child(
        wrapper.ChildSpec(
            index=1,
            label="Fixture Run",
            run_id="fixture_run_id",
            registry="registry.yaml",
            child_root=str(child_root),
            batch_tag="fixture_0001",
            knobs=(),
            performance_profile="",
            campaign_output_profile="compact_default",
            allow_large_output=False,
            allow_generation_checkpointing=False,
            perf=False,
            overwrite=False,
            command_log=str(command_log),
            logs_dir=str(logs_dir),
            worker_slot=0,
        )
    )

    rows = [json.loads(line) for line in command_log.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "child_start"
    assert rows[0]["stdout_path"] == result["stdout_path"]
    assert rows[0]["stderr_path"] == result["stderr_path"]


def test_progress_ui_off_does_not_launch_tmux(tmp_path: Path) -> None:
    wrapper = _load_wrapper_module()
    args = SimpleNamespace(progress_ui="off")

    payload = wrapper._launch_progress_ui(
        args=args,
        runs_root=tmp_path / "runs",
        command_log=tmp_path / "parallel_command_log.jsonl",
    )

    assert payload["launched"] is False
    assert payload["kind"] == "off"
    assert payload["attach_command"] == ""


def test_progress_ui_auto_launches_detached_tmux(tmp_path: Path, monkeypatch) -> None:
    wrapper = _load_wrapper_module()
    calls = []

    class Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return Proc()

    monkeypatch.setattr(wrapper.shutil, "which", lambda name: "/usr/bin/tmux" if name == "tmux" else None)
    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    args = SimpleNamespace(
        progress_ui="auto",
        progress_ui_session="fixture_ui",
        progress_ui_refresh_s=0.5,
        batch_tag="fixture",
    )

    payload = wrapper._launch_progress_ui(
        args=args,
        runs_root=tmp_path / "runs",
        command_log=tmp_path / "parallel_command_log.jsonl",
    )

    assert payload["launched"] is True
    assert payload["kind"] == "tmux"
    assert payload["session"] == "fixture_ui"
    assert payload["attach_command"] == "tmux attach -t fixture_ui"
    assert calls[0][0][:5] == ["/usr/bin/tmux", "new-session", "-d", "-s", "fixture_ui"]
    assert "eiel_campaign_tui.py" in calls[0][0][5]
    assert "--runs-root" in calls[0][0][5]


def test_metrics_final_summary_streams_last_complete_row(tmp_path: Path) -> None:
    wrapper = _load_wrapper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text(
        "generation,alive,deaths,successful_births,failed_births,lineage_active_count,lineage_top1_alive_share,lineage_effective_alive_lineages\n"
        "0,4,0,1,0,2,0.5,2\n"
        "1,5,1,2,0,3,0.6,3\n",
        encoding="utf-8",
    )

    summary = wrapper._metrics_final_summary(run_dir)

    assert summary["metrics_row_count"] == 2
    assert summary["final_metrics"]["alive"] == "5"
    assert summary["final_metrics"]["lineage_effective_alive_lineages"] == "3"


def test_metrics_final_summary_ignores_incomplete_trailing_row(tmp_path: Path) -> None:
    wrapper = _load_wrapper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text(
        "generation,alive,deaths,successful_births,failed_births,lineage_active_count,lineage_top1_alive_share,lineage_effective_alive_lineages\n"
        "0,4,0,1,0,2,0.5,2\n"
        "1,",
        encoding="utf-8",
    )

    summary = wrapper._metrics_final_summary(run_dir)

    assert summary["metrics_row_count"] == 1
    assert summary["final_metrics"]["alive"] == "4"


def test_metrics_final_summary_handles_no_trailing_newline_and_blank_lines(tmp_path: Path) -> None:
    wrapper = _load_wrapper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text(
        "generation,alive,deaths,successful_births,failed_births,lineage_active_count,lineage_top1_alive_share,lineage_effective_alive_lineages\n"
        "0,4,0,1,0,2,0.5,2\n"
        "\n"
        "1,8,0,3,0,4,0.25,3",
        encoding="utf-8",
    )

    summary = wrapper._metrics_final_summary(run_dir)

    assert summary["metrics_row_count"] == 2
    assert summary["final_metrics"]["alive"] == "8"


def test_metrics_final_summary_skips_missing_field_row_and_ignores_extra_field(tmp_path: Path) -> None:
    wrapper = _load_wrapper_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.csv").write_text(
        "generation,alive,deaths,successful_births,failed_births,lineage_active_count,lineage_top1_alive_share,lineage_effective_alive_lineages\n"
        "0,4,0,1,0,2,0.5,2\n"
        "1,9,0,3,0,4,0.25\n"
        "2,10,0,4,0,5,0.2,4,ignored_extra\n",
        encoding="utf-8",
    )

    summary = wrapper._metrics_final_summary(run_dir)

    assert summary["metrics_row_count"] == 2
    assert summary["final_metrics"]["alive"] == "10"


def test_metrics_final_summary_header_only_and_empty_file(tmp_path: Path) -> None:
    wrapper = _load_wrapper_module()
    header_only = tmp_path / "header_only"
    empty = tmp_path / "empty"
    header_only.mkdir()
    empty.mkdir()
    (header_only / "metrics.csv").write_text(
        "generation,alive,deaths,successful_births,failed_births,lineage_active_count,lineage_top1_alive_share,lineage_effective_alive_lineages\n",
        encoding="utf-8",
    )
    (empty / "metrics.csv").write_text("", encoding="utf-8")

    assert wrapper._metrics_final_summary(header_only)["metrics_row_count"] == 0
    assert wrapper._metrics_final_summary(empty)["metrics_row_count"] == 0


def test_campaign_index_reuses_unchanged_summary_and_falls_back_when_stale(tmp_path: Path) -> None:
    wrapper = _load_wrapper_module()
    child_root = tmp_path / "child"
    run_dir = child_root / "batch" / "run"
    run_dir.mkdir(parents=True)
    summary_path = child_root / "batch" / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    summary_path.write_text(json.dumps({"runs": [{"run_dir": str(run_dir), "status": "ok"}]}) + "\n", encoding="utf-8")
    metrics_path.write_text("generation,alive\n0,4\n", encoding="utf-8")
    row = {
        "index": 1,
        "label": "fixture",
        "run_id": "fixture_run",
        "registry": "registry.yaml",
        "child_root": str(child_root),
        "summary_path": str(summary_path),
        "run_dir": str(run_dir),
        "status": "ok",
        "returncode": 0,
    }
    entry = wrapper._campaign_index_entry_from_result(row)
    counters = wrapper._campaign_index_counters()

    assert wrapper._find_latest_summary(child_root, entry, counters) == summary_path
    assert counters["campaign_index_entries_reused"] == 1

    summary_path.write_text(json.dumps({"runs": [{"run_dir": str(run_dir), "status": "failed"}]}) + "\n", encoding="utf-8")
    assert wrapper._find_latest_summary(child_root, entry, counters) == summary_path
    assert counters["campaign_index_entries_refreshed"] == 1
    assert counters["campaign_index_full_scan_fallbacks"] == 1


def test_campaign_index_falls_back_when_newer_summary_sibling_exists(tmp_path: Path) -> None:
    wrapper = _load_wrapper_module()
    child_root = tmp_path / "child"
    old_run_dir = child_root / "e" / "commit" / "batch_old" / "r" / "run"
    new_run_dir = child_root / "e" / "commit" / "batch_new" / "r" / "run"
    old_run_dir.mkdir(parents=True)
    old_summary = child_root / "e" / "commit" / "batch_old" / "summary.json"
    new_summary = child_root / "e" / "commit" / "batch_new" / "summary.json"
    old_summary.write_text(json.dumps({"runs": [{"run_dir": str(old_run_dir), "status": "old"}]}) + "\n", encoding="utf-8")
    entry = wrapper._campaign_index_entry_from_result(
        {
            "index": 1,
            "label": "fixture",
            "run_id": "fixture_run",
            "registry": "registry.yaml",
            "child_root": str(child_root),
            "summary_path": str(old_summary),
            "run_dir": str(old_run_dir),
            "status": "ok",
            "returncode": 0,
        }
    )
    new_run_dir.mkdir(parents=True)
    new_summary.write_text(json.dumps({"runs": [{"run_dir": str(new_run_dir), "status": "new"}]}) + "\n", encoding="utf-8")
    counters = wrapper._campaign_index_counters()

    assert wrapper._find_latest_summary(child_root, entry, counters) == new_summary
    assert counters["campaign_index_entries_refreshed"] == 1
    assert counters["campaign_index_full_scan_fallbacks"] == 1


def test_campaign_index_writes_atomic_versioned_sidecar(tmp_path: Path) -> None:
    wrapper = _load_wrapper_module()
    args = SimpleNamespace(batch_tag="fixture", registry="registry.yaml", case=[])
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    summary_path = tmp_path / "summary.json"
    summary_path.write_text('{"runs": []}\n', encoding="utf-8")
    (run_dir / "metrics.csv").write_text("generation,alive\n0,4\n", encoding="utf-8")
    payload = wrapper._build_campaign_index_payload(
        runs_root=tmp_path,
        args=args,
        results=[
            {
                "index": 1,
                "label": "fixture",
                "run_id": "fixture_run",
                "registry": "registry.yaml",
                "child_root": str(tmp_path),
                "summary_path": str(summary_path),
                "run_dir": str(run_dir),
                "status": "ok",
                "returncode": 0,
            }
        ],
        counters=wrapper._campaign_index_counters(),
    )
    path = tmp_path / "campaign_run_index_v1.json"

    wrapper._write_campaign_index(path, payload)
    loaded = wrapper._load_campaign_index(path)

    assert loaded["schema_version"] == "eiel_campaign_run_index_v1"
    assert loaded["entries"][0]["run_id"] == "fixture_run"
