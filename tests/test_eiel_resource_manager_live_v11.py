from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").is_dir(), reason="live ERM inspection requires a Linux-like /proc")
def test_live_resource_inspection_is_read_only_and_schema_valid(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    state = tmp_path / "state"
    proc = subprocess.run(
        [
            sys.executable, "-m", "tools.eiel", "resourceManager", "inspect",
            "--no-store", "--cpu-sample-interval-s", "0",
            "--repo-root", str(ROOT), "--runtime-root", str(tmp_path / "runtime"),
            "--state-root", str(state),
        ],
        cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "OBSERVED"
    assert payload["report"]["schema_version"] == "eiel_resource_monitor_report_v1"
    assert payload["report"]["mode"] == "read_only_stateless_fresh_inspection"
    assert payload["record_path"] is None
    assert not state.exists()
