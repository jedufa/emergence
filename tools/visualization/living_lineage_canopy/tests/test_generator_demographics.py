from __future__ import annotations

import json

from tools.visualization.living_lineage_canopy.python.generator import (
    balanced,
    boom_collapse,
    branch_split,
    cyclic_fortunes,
    dominance_reversal,
    dominant_sweep,
    founder_bottleneck,
    late_bloomer,
    scale_stress,
    sibling_branch_competition,
    weak_extinction,
)
from tools.visualization.living_lineage_canopy.python.playback_model import PlaybackModel
from tools.visualization.living_lineage_canopy.python.schema import peak_living_count, validate_playback


def counts(data: dict, tick: int) -> dict[str, int]:
    return PlaybackModel(data).state_at(tick).lineage_counts


def branch_counts(data: dict, tick: int) -> dict[str, int]:
    return PlaybackModel(data).state_at(tick).branch_counts


def leader(data: dict, tick: int) -> str:
    values = counts(data, tick)
    return max(values, key=values.get)


def test_same_seed_reproduces_identical_output() -> None:
    assert json.dumps(balanced(1234), sort_keys=True) == json.dumps(balanced(1234), sort_keys=True)
    assert json.dumps(balanced(1234), sort_keys=True) != json.dumps(balanced(1235), sort_keys=True)


def test_all_agents_have_valid_lifecycle_references() -> None:
    for data in [
        balanced(),
        dominant_sweep(),
        weak_extinction(),
        late_bloomer(),
        branch_split(),
        sibling_branch_competition(),
        boom_collapse(),
        dominance_reversal(),
        cyclic_fortunes(),
        founder_bottleneck(),
        scale_stress(),
    ]:
        validate_playback(data)
        assert data["metadata"]["synthetic_profile_label"] == "synthetic demographic dynamics; not scientific fitness"
        for agent in data["agents"]:
            if agent["death_tick"] is not None:
                assert agent["death_tick"] >= agent["birth_tick"]


def test_stress_dataset_reaches_particle_target() -> None:
    assert peak_living_count(scale_stress()) >= 5000


def test_lineage_becomes_dominant_and_takeover_is_gradual() -> None:
    data = dominant_sweep()
    early = counts(data, 20)
    late = counts(data, 240)
    assert late["lineage_1"] / sum(late.values()) > 0.72
    assert early["lineage_1"] / sum(early.values()) < 0.35


def test_lineage_declines_substantially_and_extinction_occurs() -> None:
    data = dominant_sweep()
    c60 = counts(data, 60)
    c260 = counts(data, 260)
    assert c260.get("lineage_5", 0) <= c60["lineage_5"] * 0.15
    assert c260.get("lineage_7", 0) == 0


def test_weak_lineage_extinction_scenario_extinguishes_a_lineage() -> None:
    data = weak_extinction()
    c30 = counts(data, 30)
    c220 = counts(data, 220)
    assert c30["lineage_6"] > 40
    assert c220.get("lineage_6", 0) == 0


def test_late_bloomer_scenario_blooms_late() -> None:
    data = late_bloomer()
    c80 = counts(data, 80)
    c260 = counts(data, 260)
    assert c80.get("lineage_4", 0) < 30
    assert c260["lineage_4"] > 300


def test_late_bloom_and_boom_collapse_are_present() -> None:
    data = boom_collapse()
    c20 = counts(data, 20)
    c80 = counts(data, 80)
    c155 = counts(data, 155)
    c245 = counts(data, 245)
    assert c155["lineage_2"] > c20["lineage_2"] * 3.0
    assert c245.get("lineage_2", 0) < c155["lineage_2"] * 0.2
    assert c245["lineage_5"] > c80["lineage_5"] * 1.4


def test_descendant_branch_becomes_stronger_than_sibling() -> None:
    data = branch_split()
    late = branch_counts(data, 220)
    assert late["lineage_1_branch_b"] > late["lineage_1_branch_a"] * 1.25


def test_dominance_reversal_occurs() -> None:
    data = boom_collapse()
    assert leader(data, 150) == "lineage_2"
    assert leader(data, 245) != "lineage_2"


def test_explicit_dominance_reversal_scenario_changes_leader() -> None:
    data = dominance_reversal()
    assert leader(data, 90) == "lineage_2"
    assert leader(data, 265) == "lineage_5"


def test_cyclic_fortunes_rotate_lineage_leaders() -> None:
    data = cyclic_fortunes()
    leaders = {leader(data, tick) for tick in [60, 120, 180, 240, 300]}
    assert len(leaders) >= 3


def test_founder_bottleneck_contracts_then_recovers() -> None:
    data = founder_bottleneck()
    c20 = counts(data, 20)
    c115 = counts(data, 115)
    c260 = counts(data, 260)
    assert c115["lineage_1"] < c20["lineage_1"] * 0.2
    assert c260["lineage_1"] > c115["lineage_1"] * 8
