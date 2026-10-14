from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.eiel_actor_v11_helpers import REPO_ROOT, run_eiel
from tools import resource_monitor as erm


def test_resource_manager_live_read_only_inspection_conforms_to_erm_contract(tmp_path: Path) -> None:
    if not Path("/proc").is_dir():
        pytest.skip("live ERM inspection requires a Linux /proc filesystem")
    runtime = tmp_path / "runtime"
    live = runtime / "live" / "simulation-slots"
    live.mkdir(parents=True)
    result = run_eiel(
        [
            "resourceManager", "inspect",
            "--repo-root", str(REPO_ROOT),
            "--runtime-root", str(runtime),
            "--live-slots-root", str(live),
            "--proc-root", "/proc",
            "--cpu-sample-interval-s", "0",
            "--no-store",
        ],
        state_root=tmp_path / "state",
        expected=0,
        extra_env={"PATH": os.environ.get("PATH", "")},
    )
    report = result["report"]
    assert result["status"] == "OBSERVED"
    assert report["schema_version"] == erm.SCHEMA_VERSION
    assert report["mode"] == "read_only_stateless_fresh_inspection"
    assert set(report["recommendation"]) >= {"value", "confidence", "reason"}
    assert result["record_path"] is None
    assert "no_scheduler" in report["non_goals"]


def test_resource_manager_unknown_gpu_and_conflicting_exclusive_gpu_fail_closed(tmp_path: Path) -> None:
    from tests.eiel_actor_v11_helpers import write_safe_erm_report

    state = tmp_path / "state"
    unknown_report = write_safe_erm_report(tmp_path / "unknown.json", gpu_available=False)
    unknown = run_eiel(
        [
            "resourceManager", "assess", "--device", "cuda", "--report", str(unknown_report),
            "--generated-at", "2026-08-03T12:00:00Z",
        ],
        state_root=state,
        expected=1,
    )
    assert unknown["status"] == "MANUAL_REVIEW"
    assert "gpu_state_unknown" in unknown["reason_codes"]

    conflict_report = write_safe_erm_report(tmp_path / "conflict.json", blocked=True)
    conflict = run_eiel(
        [
            "resourceManager", "assess", "--device", "cuda", "--exclusive-gpu",
            "--report", str(conflict_report), "--generated-at", "2026-08-03T12:00:00Z",
        ],
        state_root=state,
        expected=1,
    )
    assert conflict["status"] in {"WAIT", "REJECT"}
    assert any(code in conflict["reason_codes"] for code in ("exclusive_gpu_conflict", "gpu_blocked"))
