from __future__ import annotations

from tools.bin.habacclim1r_construct_search import (
    ARM_NAMES,
    freeze_fixture_contract,
    run_search,
    spec_to_policy,
    target_contract,
    _base_candidate,
)


def test_habacclim1r_fixture_contract_is_frozen_and_excludes_hidden_inputs() -> None:
    contract = freeze_fixture_contract()

    assert contract["schema_version"] == "habacclim1r_construct_fixture_contract_v1"
    assert contract["frozen_before_search"] is True
    assert contract["policy_observations"]["input_dimension"] == 75
    assert contract["policy_observations"]["scalar_index"] == 74
    assert contract["policy_observations"]["hidden_habitat_label_available_to_policy"] is False
    assert contract["policy_observations"]["global_coordinates_available_to_policy"] is False
    assert len(contract["fixtures"]) >= 16
    assert {fx["control"] for fx in contract["fixtures"]} >= {"zero_lag", "true_flat", "acclimation_tradeoff"}


def test_habacclim1r_target_contract_uses_zero_denominator_construct_failure() -> None:
    contract = target_contract()

    assert "switcher_success_denominator_null" not in contract["hard_constraints"]["balanced_switcher"]
    assert contract["zero_denominator_policy"]["intended_behavior_absent"] == "CONSTRUCT_FAILURE"
    assert "transitions_gt_both_specialists" in contract["hard_constraints"]["balanced_switcher"]


def test_habacclim1r_candidate_specs_build_ordinary_75_input_mlp() -> None:
    candidate = _base_candidate()

    for arm in ARM_NAMES:
        policy = spec_to_policy(getattr(candidate, arm))
        assert policy.net[0].in_features == 75
        assert policy.net[0].out_features == 16
        assert policy.net[4].out_features == 7


def test_habacclim1r_mini_search_writes_compact_result(tmp_path) -> None:
    result = run_search(12, tmp_path)

    assert result["schema_version"] == "habacclim1r_construct_search_result_v1"
    assert result["total_candidate_fixture_evaluations"] == 12
    assert result["selected_candidate"] is None
    assert (tmp_path / "habacclim1r_fixture_contract.json").exists()
    assert (tmp_path / "habacclim1r_target_contract.json").exists()
    assert (tmp_path / "habacclim1r_construct_search_result.json").exists()
    assert (tmp_path / "habacclim1r_construct_search_records.csv").exists()
