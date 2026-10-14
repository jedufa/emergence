from __future__ import annotations

import pytest

from tools.scripts.eiel_gpu_activity_monitor_v1 import _parse_row


def test_gpu_activity_row_parser_preserves_exact_query_order() -> None:
    assert _parse_row("2026/08/22 20:00:00.000, 0, 73, 41, 5120, 6144, 55.2, 1493") == [
        "2026/08/22 20:00:00.000",
        "0",
        "73",
        "41",
        "5120",
        "6144",
        "55.2",
        "1493",
    ]


def test_gpu_activity_row_parser_rejects_incomplete_samples() -> None:
    with pytest.raises(RuntimeError, match="field count"):
        _parse_row("2026/08/22 20:00:00.000, 0, 73")
