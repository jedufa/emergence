from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_phase_b_strict_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("EIEL_STRICT_DETERMINISM", "DEBUG_DETERMINISM", "EIEL_DETERMINISM_STRICT"):
        monkeypatch.delenv(key, raising=False)
