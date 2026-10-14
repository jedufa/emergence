from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from simulation.runner_pkg.runner_action_opportunity import (
    decision_margin_payload,
    record_counterfactual_action_response,
    shadow_margin_rows,
)


def _runner(*, deterministic: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        sim_cfg=SimpleNamespace(
            structural_counterfactual_action_response_reporting_enable=True,
            structural_decision_margin_reporting_enable=False,
            structural_decision_margin_event_sample_limit=4096,
            structural_variant_identity_reporting_enable=False,
            structural_live_semantics_enable=False,
            policy_deterministic_actions=bool(deterministic),
        ),
        env=None,
        gen=0,
        tick=0,
    )


def test_rep2j_deterministic_counterfactual_records_action_change() -> None:
    runner = _runner(deterministic=True)
    alive = [SimpleNamespace(agent_id=1), SimpleNamespace(agent_id=2)]
    record_counterfactual_action_response(
        runner,
        alive=alive,
        row_indices=[0, 1],
        actual_logits=np.asarray([[0.0, 3.0], [2.0, 0.0]], dtype=np.float32),
        counterfactual_logits=np.asarray([[4.0, 1.0], [2.0, 0.0]], dtype=np.float32),
        actual_actions=np.asarray([1, 0], dtype=np.int64),
        status_by_row=["ok", "ok"],
    )
    summary = runner._structural_action_opportunity_total
    assert summary["actual_counterfactual_evaluated_count"] == 2
    assert summary["actual_counterfactual_action_change_count"] == 1
    assert summary["chosen_action_change_count"] == 1
    assert summary["actual_counterfactual_changed_action_pairs"] == {"1->0": 1}


def test_rep2j_stochastic_counterfactual_reuses_same_noise() -> None:
    runner = _runner(deterministic=False)
    alive = [SimpleNamespace(agent_id=1)]
    same_noise = np.asarray([[0.0, 3.0]], dtype=np.float32)
    record_counterfactual_action_response(
        runner,
        alive=alive,
        row_indices=[0],
        actual_logits=np.asarray([[2.0, 0.0]], dtype=np.float32),
        counterfactual_logits=np.asarray([[2.0, -4.0]], dtype=np.float32),
        actual_actions=np.asarray([1], dtype=np.int64),
        sample_noise=same_noise,
        status_by_row=["ok"],
    )
    summary = runner._structural_action_opportunity_total
    assert summary["actual_counterfactual_evaluated_count"] == 1
    assert summary["actual_counterfactual_action_change_count"] == 1
    assert summary["chosen_action_change_count"] == 1
    assert summary["actual_counterfactual_changed_action_pairs"] == {"1->0": 1}


def test_rep2j_stochastic_counterfactual_fails_closed_without_noise() -> None:
    runner = _runner(deterministic=False)
    record_counterfactual_action_response(
        runner,
        alive=[SimpleNamespace(agent_id=1)],
        row_indices=[0],
        actual_logits=np.asarray([[0.0, 1.0]], dtype=np.float32),
        counterfactual_logits=np.asarray([[1.0, 0.0]], dtype=np.float32),
        actual_actions=np.asarray([1], dtype=np.int64),
        status_by_row=["ok"],
    )
    summary = runner._structural_action_opportunity_total
    assert summary["actual_counterfactual_evaluated_count"] == 0
    assert summary["actual_counterfactual_not_evaluable_count"] == 1
    assert summary["actual_counterfactual_not_evaluable_reasons"] == {"same_sample_noise_missing": 1}


def test_rep2n_valid_action_margin_uses_only_valid_actions() -> None:
    margin = decision_margin_payload(
        np.asarray([10.0, 0.5, 0.25, 0.0, 9.0, -1.0, 8.0], dtype=np.float32),
        [1, 2, 4],
    )
    assert margin["top_action"] == 4
    assert margin["second_action"] == 1
    assert margin["top_two_margin"] == 8.5


def test_rep2n_one_valid_action_is_not_evaluable() -> None:
    margin = decision_margin_payload(np.asarray([0.0, 1.0], dtype=np.float32), [1])
    assert margin["evaluable"] is False
    assert margin["top_two_margin"] == 0.0


def test_rep2n_decision_margin_sampler_records_same_sample_event() -> None:
    runner = _runner(deterministic=False)
    runner.sim_cfg.structural_decision_margin_reporting_enable = True
    runner.env = None
    record_counterfactual_action_response(
        runner,
        alive=[SimpleNamespace(agent_id=1)],
        row_indices=[0],
        actual_logits=np.asarray([[0.0, 2.0, 1.0]], dtype=np.float32),
        counterfactual_logits=np.asarray([[0.0, 0.5, 3.0]], dtype=np.float32),
        actual_actions=np.asarray([1], dtype=np.int64),
        sample_noise=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        status_by_row=["ok"],
    )
    assert runner._structural_action_opportunity_total["actual_counterfactual_action_change_count"] == 1
    events = runner._structural_decision_margin_events
    assert len(events) == 1
    assert events[0]["same_sample_noise_present"] is True
    assert events[0]["chosen_action_changed"] == 1


def test_rep2n_shadow_margin_rows_are_read_only_recomputations() -> None:
    event = {
        "actual_score_vector_json": "[0.0,2.0]",
        "ablated_score_vector_json": "[2.0,0.0]",
        "valid_actions": "0,1",
        "actual_chosen_action": 1,
        "ablated_top_action": 0,
    }
    rows = shadow_margin_rows(event, multipliers=(0.0, 0.5, 1.0))
    assert [row["shadow_top_action"] for row in rows] == [0, 0, 1]
    assert rows[0]["multiplier"] == 0.0
