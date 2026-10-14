from __future__ import annotations

from tools.scripts.eiel_b48_hapb_fixed_mlp_policy_callback_profile_v1 import _summary


def test_callback_profile_summary_reports_stable_order_statistics() -> None:
    assert _summary([30, 10, 20]) == {
        "count": 3,
        "min_ns": 10,
        "median_ns": 20,
        "p95_ns": 30,
        "max_ns": 30,
    }
