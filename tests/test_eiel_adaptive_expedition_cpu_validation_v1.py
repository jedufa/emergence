from __future__ import annotations

import json
from pathlib import Path

import tests.eiel_adaptive_expedition_cpu_validation_v1 as validation


def test_development_corpus_covers_declared_families_without_oracle_leakage() -> None:
    fixtures = validation.build_development_fixtures()
    assert len(fixtures) == 22
    assert tuple(fixture.family for fixture in fixtures) == validation.DEVELOPMENT_FAMILIES
    assert all(not fixture.holdout for fixture in fixtures)

    policy = validation.AdaptiveIlluminationPolicyV1()
    for fixture in fixtures:
        decision_input = validation.build_policy_input_v1(policy, fixture)
        encoded = json.dumps(decision_input, sort_keys=True)
        hidden = (
            *fixture.hidden_informative_regions,
            *fixture.hidden_dead_regions,
            *fixture.hidden_traps,
        )
        assert all(label not in encoded for label in hidden)
        assert validation._find_forbidden_keys(decision_input) == ()


def test_p1_policy_suite_is_deterministic_and_has_no_forbidden_decisions() -> None:
    result = validation.run_policy_suite(
        validation.AdaptiveIlluminationPolicyV1(),
        validation.build_development_fixtures(),
    )
    assert result["fixture_count"] == 22
    assert result["determinism_failures"] == 0
    assert result["forbidden_action_count"] == 0
    assert result["forbidden_authority_field_count"] == 0
    assert result["hidden_oracle_leak_count"] == 0
    assert result["capacity_violation_count"] == 0
    assert result["status_counts"]["FORBIDDEN"] == 0
    assert result["status_counts"]["QUESTIONABLE"] >= 1


def test_p2_holdout_is_distinct_and_scored_after_frozen_contract() -> None:
    development = validation.build_development_fixtures()
    holdout = validation.build_holdout_fixtures()
    assert len(holdout) == 8
    assert all(fixture.holdout for fixture in holdout)
    assert not {fixture.case_id for fixture in development} & {
        fixture.case_id for fixture in holdout
    }
    assert validation.fixture_set_digest(development) != validation.fixture_set_digest(holdout)

    report = validation.run_cpu_validation_programme()
    assert report["holdout"]["holdout_set_frozen_before_scoring"] is True
    assert report["frozen_policy_contract"]["policy_version"] == "1"
    assert report["holdout"]["status_counts"]["FORBIDDEN"] == 0
    assert report["holdout"]["determinism_failures"] == 0


def test_p3_metamorphic_suite_uses_semantic_projection_and_explicit_packing_limit() -> None:
    result = validation.run_metamorphic_suite()
    assert result["case_count"] == 10
    assert result["failure_count"] == 0
    assert result["packing_semantics"] == validation.PACKING_PROJECTION_ONLY
    assert {
        row["case_id"] for row in result["cases"]
    } >= {
        "world-order-permutation",
        "result-arrival-permutation",
        "action-candidate-enumeration-permutation",
        "archive-iteration-permutation",
        "packing-projection-B16",
        "packing-projection-B64",
        "packing-projection-B1024",
        "logical-world-id-renaming",
        "nonsemantic-path-relocation",
        "decision-restart-replay",
    }
    assert all(
        row["note"] == validation.PACKING_PROJECTION_ONLY
        for row in result["cases"]
        if row["case_id"].startswith("packing-projection-")
    )


def test_p4_atlas_torture_enforces_typed_cap_and_rebuilds() -> None:
    result = validation.run_atlas_torture()
    assert result["input_entries"] == 96
    assert result["raw_append_entries"] == 96
    assert result["terminal_entries"] == result["bounded_test_cap"] == 64
    assert result["typed_overflow_rejected"] is True
    assert result["bounded_cap_respected"] is True
    assert result["raw_rebuild_byte_identical"] is True
    assert result["rebuild_byte_identical"] is True
    assert result["source_result_tamper_rejected"] is True
    assert result["niche_coverage"] == 8
    assert all(
        row["valid"] and row["rebuild_byte_identical"]
        for row in result["descriptor_variants"].values()
    )
    assert result["state_retention_authority"] is False


def test_p4b_parent_bank_torture_preserves_bounds_and_held_parent() -> None:
    result = validation.run_parent_bank_torture()
    assert result["bounded_entries"] is True
    assert result["bounded_bytes"] is True
    assert result["state_identity_unique"] is True
    assert result["protected_drop_rejected"] is True
    assert result["tampered_state_ref_rejected"] is True
    assert result["rebuild_byte_identical"] is True
    assert result["state_availability_authority"] is False
    assert result["deletion_authority"] is False
    assert result["lifecycle_release_authority"] is False


def test_p5a_is_five_family_synthetic_benchmark_and_not_p5b() -> None:
    result = validation.run_cpu_p5a_benchmark()
    assert result["benchmark_kind"] == "P5A_CPU_SYNTHETIC_HIDDEN_LANDSCAPE"
    assert result["row_count"] == 5
    assert result["physical_p5b_assessed"] is False
    assert all(row["physical_execution_assessed"] is False for row in result["rows"])
    assert all(row["gpu_time_seconds"] == 0 for row in result["rows"])
    assert all(row["operator_world_ticks"] <= row["reference_world_ticks"] for row in result["rows"])
    assert any(row["conclusion"] == "BRUTE_FORCE_ADVANTAGE" for row in result["rows"])


def test_p6_scientific_red_team_rejects_hidden_authority() -> None:
    result = validation.run_scientific_red_team()
    assert result["case_count"] == 3
    assert result["failure_count"] == 0
    assert result["no_hidden_reward_violation_count"] == 0
    assert result["forbidden_action_count"] == 0
    assert result["forbidden_authority_field_count"] == 0
    assert result["control_capacity_violations"] == 0
    assert result["control_domain_violations"] == 0


def test_p6b_fault_suite_fails_closed_without_duplicate_effects() -> None:
    result = validation.run_ledger_fault_suite()
    assert result["valid_ledger_replay"] is True
    assert result["fault_count"] >= 8
    assert result["all_faults_fail_closed"] is True
    assert result["unrecoverable_orphan_count"] == 0
    assert result["duplicate_action_count"] == 0
    assert result["duplicate_child_count"] == 0
    assert result["rerolled_decision_count"] == 0
    assert result["lost_settled_frontier_count"] == 0


def test_full_report_keeps_partial_verdicts_and_no_execute_gpu_packet() -> None:
    report = validation.run_cpu_validation_programme()
    assert report["no_claims"] is True
    assert report["verdicts"]["INTEGRITY"] == "PASS"
    assert report["verdicts"]["EFFICIENCY"] == "NOT_ASSESSED"
    assert report["verdicts"]["LONG_HORIZON_STABILITY"] == "NOT_ASSESSED"
    assert report["scope"]["P5B_physical_eiel"] == "NOT_ASSESSED"
    packet = report["gpu_followup_packet"]
    assert packet["execution_authority"] is False
    assert packet["no_execute"] is True
    assert packet["cpu_evidence_cannot_substitute_for_physical_gpu_evidence"] is True
    assert packet["active_gpu_run_fence"]["disturb_active_run"] is False
    assert packet["active_gpu_run_fence"]["restart_active_run"] is False
    assert report["recommended_next_step"] == "READY_FOR_MORE_REAL_USE"


def test_compact_report_omits_fixture_payloads(tmp_path: Path) -> None:
    report = validation.run_cpu_validation_programme()
    publication = validation.write_compact_report(tmp_path, report)
    json_path = Path(publication["json_path"])
    markdown_path = Path(publication["markdown_path"])
    assert json_path.is_file()
    assert markdown_path.is_file()
    assert publication["json_bytes"] <= 256 * 1024
    assert publication["markdown_bytes"] <= 256 * 1024
    compact = json.loads(json_path.read_text(encoding="utf-8"))
    assert "rows" not in compact["development"]
    assert "rows" not in compact["holdout"]
    encoded = json_path.read_text(encoding="utf-8")
    assert '"observations":' not in encoded
    assert '"hidden_informative_regions":' not in encoded


def test_cpu_harness_does_not_import_gpu_or_provider_runtime() -> None:
    source = Path(validation.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import subprocess" not in source
    assert "eiel_composition.adaptive_expedition.provider" not in source
