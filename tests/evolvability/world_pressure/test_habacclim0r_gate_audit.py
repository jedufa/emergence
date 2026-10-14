from __future__ import annotations

from tools.bin.habacclim0r_gate_audit import _ordinary_observation_fixture


def test_habacclim0r_acclimation_state_is_required_but_absent_from_ordinary_fixture() -> None:
    fixture = _ordinary_observation_fixture()

    assert fixture["ordinary_observations_identical_without_acclimation_scalar"] is True
    assert fixture["required_behaviors_differ"] is True
    assert fixture["candidate_required_but_absent_scalar"] == "habitat_acclimation_state_norm"
    assert fixture["candidate_required_but_absent_scalar"] not in fixture["ordinary_observation_channels"]
