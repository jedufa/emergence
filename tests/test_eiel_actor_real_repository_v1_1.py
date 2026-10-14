from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
EIEL = ROOT / "tools" / "bin" / "eiel"
REGISTER = ROOT / "docs/100_governance-authority/110_authority-registers/EIEL-DOC-000005--eiel-qualification-authority-register.json"

pytestmark = pytest.mark.skipif(not REGISTER.is_file(), reason="published qualification register is not included in this core transfer")


def _run(args: Sequence[str], expected: int | set[int] = 0) -> dict[str, Any]:
    proc = subprocess.run(
        [str(EIEL), *args], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
        text=True, capture_output=True, check=False,
    )
    allowed = {expected} if isinstance(expected, int) else expected
    assert proc.returncode in allowed, (proc.args, proc.returncode, proc.stdout, proc.stderr)
    return json.loads(proc.stdout)


def test_real_authority_register_qualified_blocked_superseded_and_inconsistent(tmp_path: Path) -> None:
    assert REGISTER.is_file()
    state = tmp_path / "state"

    qualified = _run([
        "qualificationManager", "check", "--track-label", "b1w32p256c512gNt160cpu",
        "--store", "--state-root", str(state),
    ])
    assert qualified["decision"] == "QUALIFIED"
    assert qualified["record_id"] == "b1w32p256c512gNt160cpu"
    assert qualified["check_id"].startswith("QCHK-")

    blocked = _run([
        "qualificationManager", "check", "--track-label", "b24w32p512c1024gNt160gpu",
        "--state-root", str(state),
    ], expected=1)
    assert blocked["decision"] == "BLOCKED"
    assert "authority_blocked" in blocked["reason_codes"]
    assert blocked["next_actions"][0]["command"].startswith("python -m tools.eiel qualificationManager plan")

    superseded = _run([
        "qualificationManager", "check", "--track-label", "b72w16p64c128gNt160gpu",
        "--state-root", str(state),
    ], expected=1)
    assert superseded["decision"] == "SUPERSEDED"
    assert "authority_superseded" in superseded["reason_codes"]

    inconsistent = _run([
        "qualificationManager", "check", "--track-label", "b48w32p512c1024gNt160gpu",
        "--state-root", str(state),
    ], expected=1)
    assert inconsistent["decision"] == "INCONSISTENT_AUTHORITY_RECORD"
    assert inconsistent["ok"] is False

    audit = _run(["qualificationManager", "audit", "--state-root", str(state)])
    assert audit["ok"] is True
    assert any(row["label"] == "b48w32p512c1024gNt160gpu" for row in audit["semantic_warnings"])


def test_real_register_source_mismatch_fails_closed(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({
        "track_label": "b1w32p256c512gNt160cpu",
        "source_commit": "definitely-not-the-register-commit",
    }), encoding="utf-8")
    result = _run([
        "qualificationManager", "check", "--request", str(request),
        "--state-root", str(tmp_path / "state"),
    ], expected=1)
    assert result["decision"] == "REVALIDATION_REQUIRED"
    assert result["ok"] is False
    assert any("source_commit differs" in problem for problem in result["problems"])
