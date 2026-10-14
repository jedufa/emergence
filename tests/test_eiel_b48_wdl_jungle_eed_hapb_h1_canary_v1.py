from __future__ import annotations

import pytest

from tools.scripts import eiel_b48_wdl_jungle_eed_hapb_h1_canary_v1 as subject


def test_scaled_indices_replay_observed_low_slot_p256_across_b48() -> None:
    observed = [list(range(256)) for _ in range(160)]
    pure = subject._scaled_indices(observed, active_rows=256, representation="pure")
    hot = subject._scaled_indices(observed, active_rows=512, representation="p256_hot_prefix")

    assert len(pure) == 160
    assert len(pure[0]) == 48 * 256
    assert len(hot[0]) == 48 * 256
    assert pure[0][:3] == (0, 1, 2)
    assert hot[0][:3] == (256, 257, 258)


def test_scaled_indices_rejects_unproven_low_slot_premise() -> None:
    with pytest.raises(subject.H1CanaryError, match="slot ordering"):
        subject._scaled_indices([[1] * 256 for _ in range(160)], active_rows=256, representation="pure")


def test_pressure_gate_is_fail_closed_on_major_fault_or_swap() -> None:
    baseline = {
        "rss_bytes": 0,
        "hwm_bytes": 0,
        "major_fault_delta": 0,
        "swap_free_delta_bytes": 0,
        "psi_memory_avg10_delta": 0.0,
    }
    assert subject._pressure_passed(baseline)
    assert not subject._pressure_passed({**baseline, "major_fault_delta": 1})
    assert not subject._pressure_passed({**baseline, "swap_free_delta_bytes": -1})
