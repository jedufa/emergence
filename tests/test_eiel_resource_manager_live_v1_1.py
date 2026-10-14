from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tools import resource_monitor as erm

ROOT = Path(__file__).resolve().parents[1]
EIEL = ROOT / "tools" / "bin" / "eiel"


@pytest.mark.skipif(
    os.environ.get("EIEL_RUN_LIVE_RESOURCE_TESTS") != "1",
    reason="set EIEL_RUN_LIVE_RESOURCE_TESTS=1 to perform a read-only host inspection",
)
def test_live_resource_inspection_is_read_only_and_schema_valid(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            str(EIEL), "resourceManager", "inspect", "--no-store",
            "--repo-root", str(ROOT), "--state-root", str(tmp_path / "state"),
            "--cpu-sample-interval-s", "0",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)
    payload = json.loads(proc.stdout)
    assert payload["report"]["schema_version"] == erm.SCHEMA_VERSION
    assert payload["report"]["mode"] == "read_only_stateless_fresh_inspection"
    text = json.dumps(payload["report"]).lower()
    for forbidden in ("reservation_created", "process_killed", "campaign_launched"):
        assert forbidden not in text
