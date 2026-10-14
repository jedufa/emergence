import json

import numpy as np
from types import SimpleNamespace

from engine.simulation.runner_pkg.runner_action_opportunity import (
    _empty_family_head_choice_margin_summary,
    _empty_family_head_magnitude_surface_summary,
    _argmax_stable,
    _authority_scores_for_alpha,
    _find_crossing_alpha,
    _empty_opportunity_conditioned_candidate_summary,
    _merge_family_head_choice_margin_delta,
    _update_opportunity_conditioned_candidate_summary,
    _update_family_head_choice_margin_summary,
    _update_family_head_magnitude_surface_summary,
    decision_margin_payload,
    record_family_head_authority_choice_margin,
)


def test_polarch9_same_sample_alpha_crossing_reconstructs_choice_change():
    logits = np.array([0.0, 0.05, -0.2, -0.3, 0.0, -0.4, -0.5], dtype=np.float64)
    valid_actions = list(range(7))
    residual = {"movement": 0.0, "gather": 0.1, "rest": 0.0, "attack": 0.0}
    noise = np.zeros_like(logits)

    neutral_choice = _argmax_stable(logits + noise)
    directional_scores = _authority_scores_for_alpha(logits, valid_actions, residual=residual, alpha=1.0)
    directional_choice = _argmax_stable(directional_scores + noise)
    crossing_alpha = _find_crossing_alpha(
        logits,
        valid_actions,
        residual=residual,
        alpha_cap=1.0,
        noise=noise,
        mode="primitive",
    )

    assert neutral_choice == 1
    assert directional_choice == 4
    assert crossing_alpha is not None
    assert 0.49 <= crossing_alpha <= 0.51


def test_polarch9_neutral_residual_has_no_crossing():
    logits = np.array([0.2, 0.1, -0.3, -0.4, -0.2, -0.5, -0.6], dtype=np.float64)
    valid_actions = list(range(7))
    residual = {"movement": 0.0, "gather": 0.0, "rest": 0.0, "attack": 0.0}

    crossing_alpha = _find_crossing_alpha(
        logits,
        valid_actions,
        residual=residual,
        alpha_cap=5.0,
        noise=np.zeros_like(logits),
        mode="primitive",
    )

    assert crossing_alpha is None
    assert np.allclose(
        _authority_scores_for_alpha(logits, valid_actions, residual=residual, alpha=0.0),
        _authority_scores_for_alpha(logits, valid_actions, residual=residual, alpha=3.0),
    )


def test_polarch10_magnitude_surface_records_alpha_dose_response():
    summary = _empty_family_head_magnitude_surface_summary(enabled=True)
    summary["_event_sample_limit"] = 16
    base_event = {
        "actual_sampled_action": 4,
        "action_source_class": "test",
        "policy_kind": "mlp",
        "gather_resource_available": 1,
        "legal_move_count": 3,
    }
    logits = np.array([0.0, 0.05, -0.2, -0.3, 0.0, -0.4, -0.5], dtype=np.float64)
    residual = {"movement": 0.0, "gather": 0.1, "rest": 0.0, "attack": 0.0}

    _update_family_head_magnitude_surface_summary(
        summary,
        base_event,
        logits,
        list(range(7)),
        residual,
        np.zeros_like(logits),
    )

    assert summary["event_count"] == 1
    assert summary["choice_reconstruction_mismatch_count_at_alpha_1"] == 0
    assert summary["alpha_surface"]["0"]["family_choice_change_count"] == 0
    assert summary["alpha_surface"]["1"]["family_choice_change_count"] == 1
    assert summary["alpha_surface"]["1"]["meaningful_promoted_opportunity_count"] == 1


def test_polarch10_magnitude_surface_can_run_without_choice_margin_reporting():
    runner = SimpleNamespace(
        sim_cfg=SimpleNamespace(
            policy_deterministic_actions=True,
            policy_family_head_authority_choice_margin_reporting_enable=False,
            policy_family_head_authority_magnitude_surface_reporting_enable=True,
            policy_family_head_authority_magnitude_surface_event_sample_limit=4,
        )
    )

    record_family_head_authority_choice_margin(
        runner,
        alive=[],
        row_indices=[0],
        neutral_logits=np.zeros((1, 7), dtype=np.float64),
        directional_logits=np.zeros((1, 7), dtype=np.float64),
        actions=np.array([0], dtype=np.int64),
        sample_noise=np.zeros((1, 7), dtype=np.float64),
    )

    assert not hasattr(runner, "_policy_family_head_authority_choice_margin_total")
    assert runner._policy_family_head_authority_magnitude_surface_total["not_evaluable_reasons"] == {
        "row_out_of_range": 1
    }


def test_polarch12_choice_margin_summary_rejects_nonfinite_values():
    delta = _empty_family_head_choice_margin_summary(enabled=True)
    _update_family_head_choice_margin_summary(
        delta,
        {
            "deterministic_choice": True,
            "reconstruction_parity": True,
            "directional_primitive_margin": float("inf"),
            "neutral_primitive_margin": 0.25,
            "neutral_family_margin": 0.125,
            "directional_family_margin": 0.5,
        },
    )
    runner = SimpleNamespace()
    _merge_family_head_choice_margin_delta(runner, delta)
    summary = runner._policy_family_head_authority_choice_margin_total

    assert summary["event_count"] == 1
    assert summary["directional_primitive_margin_sum"] == 0.0
    assert summary["directional_primitive_margin_mean"] == 0.0
    assert summary["not_evaluable_reasons"] == {"nonfinite_directional_primitive_margin": 1}
    json.dumps(summary, sort_keys=True, allow_nan=False)


def test_polarch12_singleton_choice_margin_is_finite():
    payload = decision_margin_payload(
        np.array([-np.inf, -np.inf, 0.25, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float64),
        [2],
    )

    assert payload["evaluable"] is False
    assert payload["top_action"] == 2
    assert payload["second_action"] == -1
    assert payload["top_two_margin"] == 0.0
    json.dumps(payload, sort_keys=True, allow_nan=False)


def test_polarch11_binary_gate_suppresses_no_opportunity_family():
    summary = _empty_opportunity_conditioned_candidate_summary(enabled=True)
    summary["_event_sample_limit"] = 16
    base_event = {
        "actual_sampled_action": 4,
        "action_source_class": "test",
        "policy_kind": "mlp",
        "gather_resource_available": 0,
        "legal_move_count": 3,
    }
    logits = np.array([0.0, 0.05, -0.2, -0.3, 0.0, -0.4, -0.5], dtype=np.float64)
    residual = {"movement": 0.0, "gather": 0.1, "rest": -0.1, "attack": 0.0}

    _update_opportunity_conditioned_candidate_summary(
        summary,
        base_event,
        logits,
        list(range(7)),
        residual,
        np.zeros_like(logits),
    )

    unconditioned = summary["candidate_surface"]["current_unconditioned|0.2"]
    binary = summary["candidate_surface"]["binary_executable_gate|0.2"]
    assert unconditioned["family_choice_change_count"] == 1
    assert binary["family_choice_change_count"] == 0
    assert binary["no_opportunity_suppression_count"] == 1
