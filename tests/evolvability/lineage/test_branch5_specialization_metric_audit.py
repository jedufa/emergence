from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "tools/bin/branch5_specialization_metric_audit.py"
SPEC = importlib.util.spec_from_file_location("branch5_specialization_metric_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
branch5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(branch5)


ZONE_A = "food_high|shelter_low|waste_low|density_low|move_cost_base|recovery_low"
ZONE_B = "food_low|shelter_high|waste_high|density_high|move_cost_base|recovery_high"


def _row(agent_id: int, exposure: dict[str, int], *, births: dict[str, int] | None = None, food_scale: float = 1.0) -> dict:
    return {
        "agent_id": agent_id,
        "birth_generation": 0,
        "final_generation": 4,
        "zone_exposure_counts": exposure,
        "movement_actions_by_zone": {zone: max(1, int(count * 0.25)) for zone, count in exposure.items()},
        "gather_actions_by_zone": {zone: max(1, int(count * 0.10)) for zone, count in exposure.items()},
        "stay_actions_by_zone": {zone: max(1, int(count * 0.05)) for zone, count in exposure.items()},
        "food_collected_by_zone": {zone: count * food_scale for zone, count in exposure.items()},
        "damage_by_zone": {},
        "injury_by_zone": {},
        "recovery_by_zone": {},
        "successful_births_by_zone": births or {},
        "transitions_between_zone_classes": {},
    }


def _profile(strategy_id: str, exposure: dict[str, int], *, copies: int = 8, births: dict[str, int] | None = None) -> dict:
    rows = [_row(i + 1, exposure, births=births) for i in range(copies)]
    return branch5.build_strategy_profile("fixture", strategy_id, rows, profile="nominal")


def test_identical_full_profiles_have_zero_distance() -> None:
    left = _profile("left", {ZONE_A: 60, ZONE_B: 40})
    right = _profile("right", {ZONE_A: 60, ZONE_B: 40})

    distance = branch5.profile_distance(left, right)

    assert distance["full_profile_jsd"] == 0.0
    assert distance["hellinger"] == 0.0


def test_strict_specialists_score_above_weak_mixed_preferences() -> None:
    weak_a = _profile("weak_a", {ZONE_A: 55, ZONE_B: 45})
    weak_b = _profile("weak_b", {ZONE_A: 45, ZONE_B: 55})
    strict_a = _profile("strict_a", {ZONE_A: 100})
    strict_b = _profile("strict_b", {ZONE_B: 100})

    weak = branch5.profile_distance(weak_a, weak_b)
    strict = branch5.profile_distance(strict_a, strict_b)

    assert 0.0 < weak["full_profile_jsd"] < strict["full_profile_jsd"] < 1.0
    assert 0.0 < weak["hellinger"] < strict["hellinger"] <= 1.0


def test_low_exposure_strategy_fails_low_confidence() -> None:
    low = branch5.build_strategy_profile("fixture", "low", [_row(1, {ZONE_A: 12})], profile="nominal")

    assert low["confidence"] == "low_exposure"


def test_population_scaling_preserves_normalized_profile() -> None:
    small = _profile("small", {ZONE_A: 70, ZONE_B: 30}, copies=8)
    large = _profile("large", {ZONE_A: 70, ZONE_B: 30}, copies=16)

    distance = branch5.profile_distance(small, large)

    assert distance["full_profile_jsd"] == 0.0
    assert small["dominant_zone_share"] == large["dominant_zone_share"]


def test_birth_scaling_preserves_zone_profile_but_changes_reproductive_component() -> None:
    base = _profile("base", {ZONE_A: 70, ZONE_B: 30}, births={ZONE_A: 1})
    scaled = _profile("scaled", {ZONE_A: 70, ZONE_B: 30}, births={ZONE_A: 2})

    distance = branch5.profile_distance(base, scaled)

    assert distance["full_profile_jsd"] == 0.0
    assert distance["reproductive_jsd"] == 0.0


def test_row_order_does_not_change_profile() -> None:
    rows = [_row(1, {ZONE_A: 70, ZONE_B: 30}), _row(2, {ZONE_A: 30, ZONE_B: 70})]

    forward = branch5.build_strategy_profile("fixture", "ordered", rows, profile="permissive")
    reverse = branch5.build_strategy_profile("fixture", "ordered", list(reversed(rows)), profile="permissive")

    assert forward["zone_exposure_distribution"] == reverse["zone_exposure_distribution"]
    assert forward["profile_vector"] == reverse["profile_vector"]
