from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from tools import eiel


def test_eiel_dispatcher_exposes_exact_actor_names(capsys) -> None:
    assert eiel.main(["--help"]) == 0
    text = capsys.readouterr().out
    for name in ("evidenceManager", "qualificationManager", "resourceManager", "missionPlanner"):
        assert name in text


def test_eiel_dispatcher_rejects_unknown_actor(capsys) -> None:
    assert eiel.main(["imaginaryManager"]) == 2
    assert "Unknown EIEL actor" in capsys.readouterr().err


def test_pyproject_exposes_installable_eiel_entry_point() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert payload["project"]["scripts"]["eiel"] == "tools.eiel:main"


def test_repository_local_eiel_launcher_works_outside_the_repository(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [str(root / "tools/bin/eiel"), "--version"],
        cwd=tmp_path,
        env={**os.environ, "EMERGENCE_PYTHON": sys.executable},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "EIEL operator tooling" in result.stdout
