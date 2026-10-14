from __future__ import annotations

import torch

from env.gridworld_pkg.constants import ACT_MOVE_E, ACT_MOVE_N, ACT_MOVE_S, ACT_MOVE_W
from evolution.target_alignment import action_class_map, score_logits_for_targets


def test_position_aware_action_classes_horizontal_targets() -> None:
    classes = action_class_map(agent_pos=(3, 3), target_a=(3, 1), target_b=(3, 5), height=7, width=7)

    assert classes[ACT_MOVE_W]["target_class"] == "toward_a"
    assert classes[ACT_MOVE_E]["target_class"] == "toward_b"
    assert classes[ACT_MOVE_N]["target_class"] == "away_or_blocked"
    assert classes[ACT_MOVE_S]["target_class"] == "away_or_blocked"


def test_position_aware_score_uses_actual_geometry_not_fixed_action_assumption() -> None:
    logits_a = torch.zeros(7)
    logits_b = torch.zeros(7)
    logits_a[ACT_MOVE_W] = 3.0
    logits_b[ACT_MOVE_E] = 3.0

    score = score_logits_for_targets(
        logits_a,
        logits_b,
        agent_pos=(3, 3),
        target_a=(3, 1),
        target_b=(3, 5),
        height=7,
        width=7,
    )

    assert score["position_aware_level4"] is True
    assert score["top_action_cue_a_target_class"] == "toward_a"
    assert score["top_action_cue_b_target_class"] == "toward_b"


def test_position_aware_score_mirrors_target_layout() -> None:
    logits_a = torch.zeros(7)
    logits_b = torch.zeros(7)
    logits_a[ACT_MOVE_E] = 3.0
    logits_b[ACT_MOVE_W] = 3.0

    score = score_logits_for_targets(
        logits_a,
        logits_b,
        agent_pos=(3, 3),
        target_a=(3, 5),
        target_b=(3, 1),
        height=7,
        width=7,
    )

    assert score["position_aware_level4"] is True
    assert score["top_action_cue_a_target_class"] == "toward_a"
    assert score["top_action_cue_b_target_class"] == "toward_b"
