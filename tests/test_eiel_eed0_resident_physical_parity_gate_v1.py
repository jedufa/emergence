from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = (
    PROJECT
    / "tools/scripts/eiel_eed0_resident_physical_parity_gate_v1.py"
)


def _subject():
    spec = importlib.util.spec_from_file_location(
        "eed_resident_physical_gate_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_padded_cursor_preserves_prefix_and_fills_retained_capacity() -> None:
    subject = _subject()
    source = np.asarray([4, 5, 6], dtype=np.int32)

    result = subject._padded_cursor(source, width=5, default=-1)

    assert result.tolist() == [4, 5, 6, -1, -1]


def test_physical_gate_is_explicitly_unpromoted() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'parser.add_argument("--ticks", type=int, choices=(80,), default=80)' in source
    assert 'if ticks != 80:' in source
    assert 'raise ValueError("resident physical EED parity requires t80")' in source
    assert '"production_route_integrated": False' in source
    assert '"campaign_authorized": False' in source
    assert "selection, route qualification, and EED-1 remain unauthorized" in source
