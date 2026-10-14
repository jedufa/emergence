from __future__ import annotations

import subprocess
import sys


def test_engine_main_help_is_a_supported_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "engine.main", "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage: python -m engine.main" in result.stdout
