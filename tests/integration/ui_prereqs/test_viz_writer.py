"""tests/integration/ui_prereqs/test_viz_writer.py

Phase 5d (UI prerequisites) engine-side instrumentation tests.

These tests validate that the run-local <RUN_DIR>/viz/ artifacts:
- are created,
- are parseable and written atomically (JSON),
- are append-only and well-formed (CSV/JSONL),
- can be regenerated from the filesystem (checkpoint index).

IMPORTANT: viz/ is explicitly outside the determinism substrate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from simulation.viz_pkg.viz_writer import VizWriter


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _viz_env_defaults(monkeypatch: pytest.MonkeyPatch):
    """Force viz/ instrumentation on for these unit tests."""
    monkeypatch.setenv("VIZ_ENABLE", "1")
    monkeypatch.setenv("VIZ_SUMMARY_EVERY_N", "1")
    monkeypatch.setenv("VIZ_ENABLE_BEHAVIORS_AGG", "0")


def test_viz_initialize_creates_required_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Determinism mode flag should be surfaced in run_determinism_status.json
    monkeypatch.setenv("DEBUG_DETERMINISM", "1")

    vw = VizWriter.from_env(run_dir)
    vw.initialize_run(start_generation=0, target_generation=5, run_kind="base")

    viz_dir = run_dir / "viz"
    assert viz_dir.is_dir()

    assert (viz_dir / "run_status.json").exists()
    assert (viz_dir / "run_determinism_status.json").exists()
    assert (viz_dir / "checkpoints_index.json").exists()
    assert (viz_dir / "metrics_summary.csv").exists()
    assert (viz_dir / "applied_commands_log.jsonl").exists()

    run_status = _read_json(viz_dir / "run_status.json")
    assert run_status["schema_version"] == "run_status_v1"
    assert run_status["status"] == "running"
    assert run_status["current_generation"] == 0
    assert run_status["target_generation"] == 5

    det = _read_json(viz_dir / "run_determinism_status.json")
    assert det["schema_version"] == "run_det_status_v1"
    assert det["strict_mode"] is True
    assert det["phase4_last_result"] in ("UNKNOWN", "PASS", "FAIL")
    assert det["last_verified_generation"] == -1

    ckpt = _read_json(viz_dir / "checkpoints_index.json")
    assert ckpt["schema_version"] == "checkpoints_index_v1"
    assert isinstance(ckpt["checkpoints"], list)

    metrics_lines = (viz_dir / "metrics_summary.csv").read_text(encoding="utf-8").splitlines()
    assert metrics_lines[0] == "generation,alive,births,deaths,avg_gen_time_s"


def test_run_status_atomic_no_tmp_leftovers(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    vw = VizWriter.from_env(run_dir)
    vw.initialize_run(start_generation=0, target_generation=2, run_kind="base")

    viz_dir = run_dir / "viz"

    # Re-write a few times.
    vw.write_run_status(status="running", current_generation=1, target_generation=2, last_error=None, branch_state="RUNNING")
    vw.write_run_status(status="complete", current_generation=2, target_generation=2, last_error=None, branch_state="COMPLETE")

    # Parseable JSON, and no *.tmp leftovers.
    _ = _read_json(viz_dir / "run_status.json")
    tmp_files = list(viz_dir.glob("*.tmp"))
    assert tmp_files == []


def test_metrics_summary_append_only_and_newline_discipline(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    vw = VizWriter.from_env(run_dir)
    vw.initialize_run(start_generation=0, target_generation=3, run_kind="base")

    p = run_dir / "viz" / "metrics_summary.csv"

    vw.append_metrics_summary(generation=0, alive=10, births=2, deaths=1, avg_gen_time_s=0.5)
    vw.append_metrics_summary(generation=1, alive=11, births=1, deaths=0, avg_gen_time_s=0.4)

    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "generation,alive,births,deaths,avg_gen_time_s"
    assert lines[1].startswith("0,10,2,1,")
    assert lines[2].startswith("1,11,1,0,")

    # Simulate a partial trailing line (no newline) and ensure the next append doesn't crash
    # or concatenate onto the partial line.
    with p.open("ab") as f:
        f.write(b"partial_no_newline")

    vw.append_metrics_summary(generation=2, alive=9, births=0, deaths=2, avg_gen_time_s=0.6)

    new_lines = p.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("2,9,0,2,") for line in new_lines)


def test_applied_commands_log_jsonl_is_parseable(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    vw = VizWriter.from_env(run_dir)
    vw.initialize_run(start_generation=0, target_generation=1, run_kind="base")

    log_path = run_dir / "viz" / "applied_commands_log.jsonl"

    vw.append_applied_command_event(
        command_uuid="cmd-1",
        command_type="CheckpointRequestCommand",
        apply_phase="submit",
        applied_generation=0,
        payload_hash="deadbeef",
        ack_status="REJECTED",
        error_reason="nope",
        timestamp="2026-01-01T00:00:00Z",
    )

    vw.append_applied_command_event(
        command_uuid="cmd-2",
        command_type="PressureUpdateCommand",
        apply_phase="safeboundary_step6",
        applied_generation=1,
        payload_hash="cafebabe",
        ack_status="APPLIED",
        error_reason=None,
        timestamp="2026-01-01T00:00:01Z",
    )

    raw = log_path.read_text(encoding="utf-8")
    assert raw.endswith("\n")

    lines = [ln for ln in raw.splitlines() if ln.strip()]
    assert len(lines) == 2

    for ln in lines:
        obj = json.loads(ln)
        assert obj["schema_version"] == "applied_cmd_v1"
        assert obj["ack_status"] in ("APPLIED", "REJECTED")


def test_checkpoints_index_scans_filesystem(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create a fake checkpoint bundle v2 directory.
    bundle_dir = run_dir / "checkpoint_bundles_v2" / "bundle_gen000010"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "schema_version": "checkpoint_bundle_v2",
        "checkpoint_id": "ckpt-10",
        "generation_index": 10,
        "bundle_dir": str(bundle_dir),
        "created_at": "2026-01-01T00:00:00Z",
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (bundle_dir / "dummy.bin").write_bytes(b"1234")

    vw = VizWriter.from_env(run_dir)
    vw.initialize_run(start_generation=0, target_generation=20, run_kind="base")

    # Rescan
    vw.update_checkpoints_index()

    idx = _read_json(run_dir / "viz" / "checkpoints_index.json")
    assert idx["schema_version"] == "checkpoints_index_v1"

    cps = idx["checkpoints"]
    assert len(cps) == 1
    assert cps[0]["generation"] == 10
    assert cps[0]["checkpoint_id"] == "ckpt-10"
    assert cps[0]["relative_path"].startswith("checkpoint_bundles_v2/")
    assert isinstance(cps[0]["size_bytes"], int)
    assert cps[0]["size_bytes"] >= 4
