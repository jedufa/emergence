from __future__ import annotations

import numpy as np

from simulation.corridor_cp0_fixed_probe import ARM_NAMES, controller_contract, validate_controller_contract
from tools.bin.corridor_cp1_probe_construct_search import _maps, _obs, _run_direct, _symmetry_blocker


def test_corridor_cp1_probe_contract_excludes_hidden_zone() -> None:
    contract = controller_contract()
    validation = validate_controller_contract()

    assert tuple(contract["arms"]) == ARM_NAMES
    assert contract["obs_size"] == 74
    assert validation["ok"] is True
    assert validation["zone_label_excluded"] is True
    assert "corridor_cp0_zone_class" in contract["hidden_state_excluded_from_action"]


def test_corridor_cp1_home_interior_reflection_is_memoryless_blocker() -> None:
    normal = _maps(reflect=False, width=14, home_width=4, remote_width=4)
    reflected = _maps(reflect=True, width=14, home_width=4, remote_width=4)
    y = normal["zones"].shape[0] // 2
    normal_obs = _obs(normal, y, 1, 3.0)
    reflected_obs = _obs(reflected, y, reflected["zones"].shape[1] - 2, 3.0)
    blocker = _symmetry_blocker()

    assert np.array_equal(normal_obs, reflected_obs)
    assert blocker["ordinary_observations_identical"] is True
    assert blocker["normal_home_interior_required_first_route_action"] != blocker["reflected_home_interior_required_first_route_action"]


def test_corridor_cp1_direct_construct_does_not_pass_commuter_cycle() -> None:
    normal = _run_direct("remote_commuter", reflect=False)
    reflected = _run_direct("remote_commuter", reflect=True)

    assert normal["completed_commute_cycles"] == 0
    assert reflected["completed_commute_cycles"] == 0
