from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from evolution.phase_f_v2.matrix import (
    AUTHORIZED_BASE_COMMIT,
    CASE_COUNT,
    CHECKPOINT_CASE_COUNT,
    EXPECTED_ACCELERATED_INVOCATIONS,
    EXPECTED_CPU_ORACLE_EVALUATIONS,
    EXPECTED_TOTAL_REQUIRED_EVALUATIONS,
    MATRIX_ID,
    SCHEMA_VERSION,
    PhaseFV2MatrixError,
    build_phase_f_v2_matrix,
    load_phase_f_v2_matrix,
    validate_phase_f_v2_matrix,
)


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "_spec/evolutionary_capacity/phase_f/v2/forced_combination_matrix.json"
PROFILE_CODES = ("b12", "b16")
ROLE_IDS = tuple(f"r{index}" for index in range(5))


def _expected_case_ids() -> list[str]:
    return [
        f"pf2_f{family:02d}_{profile}_{role}"
        for family in range(1, 13)
        for profile in PROFILE_CODES
        for role in ROLE_IDS
    ]


def _checkpoint_case_ids() -> set[str]:
    return {
        *(
            f"pf2_f12_{profile}_{role}"
            for profile in PROFILE_CODES
            for role in ROLE_IDS
        ),
        *(f"pf2_f09_{profile}_r4" for profile in PROFILE_CODES),
        *(f"pf2_f11_{profile}_r4" for profile in PROFILE_CODES),
    }


def _keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        found.update(value)
        for child in value.values():
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found


def test_generated_matrix_matches_committed_canonical_spec() -> None:
    generated = build_phase_f_v2_matrix()
    committed = load_phase_f_v2_matrix(SPEC)

    assert committed == generated
    assert committed["bindings"] == {
        "authorized_base_commit": AUTHORIZED_BASE_COMMIT,
        "frozen_execution_commit": None,
        "entry_receipt_sha256": None,
        "cpu_expectation_manifest_sha256": None,
    }


def test_matrix_has_exact_12_by_2_by_5_case_order_and_unique_seeds() -> None:
    matrix = build_phase_f_v2_matrix()
    cases = matrix["cases"]

    assert matrix["case_count"] == len(cases) == CASE_COUNT == 12 * 2 * 5
    assert [case["case_id"] for case in cases] == _expected_case_ids()
    assert len({case["case_id"] for case in cases}) == CASE_COUNT

    seeds = [case["fixture_blueprint"]["construction_seed"] for case in cases]
    assert len(set(seeds)) == CASE_COUNT
    assert seeds[0] == 620_100
    assert seeds[-1] == 621_214


def test_each_case_freezes_eight_target_variants_and_eight_intact_twins() -> None:
    for case in build_phase_f_v2_matrix()["cases"]:
        slots = case["fixture_blueprint"]["slot_plan"]
        assert [slot["slot"] for slot in slots] == list(range(16))
        for variant in range(8):
            target = slots[variant]
            twin = slots[variant + 8]
            assert target == {
                "slot": variant,
                "variant_id": f"v{variant}",
                "twin_slot": variant + 8,
                "intervention_eligible": True,
            }
            assert twin == {
                "slot": variant + 8,
                "variant_id": f"v{variant}",
                "twin_slot": variant,
                "intervention_eligible": False,
            }


def test_checkpoint_set_and_evaluation_arithmetic_are_exact() -> None:
    matrix = build_phase_f_v2_matrix()
    cases = matrix["cases"]
    checkpoint_ids = {
        case["case_id"]
        for case in cases
        if case["checkpoint_schedule"]["kind"] != "none"
    }

    assert checkpoint_ids == _checkpoint_case_ids()
    assert (
        matrix["checkpoint_case_count"] == len(checkpoint_ids) == CHECKPOINT_CASE_COUNT
    )
    assert (
        sum(case["execution_accounting"]["cpu_oracle_evaluations"] for case in cases)
        == matrix["expected_cpu_oracle_evaluations"]
        == EXPECTED_CPU_ORACLE_EVALUATIONS
    )
    assert (
        sum(case["execution_accounting"]["accelerated_invocations"] for case in cases)
        == matrix["expected_accelerated_invocations"]
        == EXPECTED_ACCELERATED_INVOCATIONS
    )
    assert (
        matrix["expected_cpu_oracle_evaluations"]
        + matrix["expected_accelerated_invocations"]
        == matrix["expected_total_required_evaluations"]
        == EXPECTED_TOTAL_REQUIRED_EVALUATIONS
    )


def test_f12_checkpoint_representatives_cover_every_required_class() -> None:
    matrix = build_phase_f_v2_matrix()
    actions = {
        case["role_contract"]["action"]
        for case in matrix["cases"]
        if case["family_id"] == "f12"
    }
    required = {
        "neutral_empty",
        "single_gated",
        "two_module_chain",
        "development_dependency",
        "dormancy",
        "structural_cost",
        "prune_reuse",
        "capacity_migration_post_boundary",
    }
    assert required <= actions
    assert len(actions) == 10


def test_v2_preregistration_resolves_actions_before_execution() -> None:
    matrix = build_phase_f_v2_matrix()

    assert SCHEMA_VERSION == "eiel_phase_f_v2_forced_combination_matrix_v2"
    assert MATRIX_ID == "phase-f-v2-forced-combination-120-v2"
    assert matrix["schema_version"] == SCHEMA_VERSION
    assert matrix["matrix_id"] == MATRIX_ID
    assert all(
        "_or_" not in case["role_contract"]["action"] for case in matrix["cases"]
    )


def test_rejected_or_pre_migration_b12_cases_are_no_op_expectations() -> None:
    cases = {
        case["case_id"]: case
        for case in build_phase_f_v2_matrix()["cases"]
        if case["family_id"] == "f11" and case["profile_id"] == "b2-12x32-v1"
    }

    assert cases["pf2_f11_b12_r2"]["role_contract"]["action"] == (
        "reject_unauthorized_request"
    )
    assert cases["pf2_f11_b12_r3"]["role_contract"]["action"] == (
        "reject_wrong_safe_boundary"
    )
    for case in cases.values():
        assert case["role_contract"]["expected_changed_surfaces"] == []


def test_semantic_thresholds_and_auxiliary_dependency_providers_are_frozen() -> None:
    families = {
        family["family_id"]: family for family in build_phase_f_v2_matrix()["families"]
    }
    f3 = [row["description"] for row in families["f03"]["panel_variants"]]
    f6 = [row["description"] for row in families["f06"]["panel_variants"]]

    assert "update bias at most -4" in f3[0]
    assert "update bias at least +4" in f3[1]
    assert "neutral prerequisite provider" in f3[6]
    assert "neutral prerequisite provider" in f3[7]
    assert f6[2] == "zero-sum signed reciprocal pair"
    assert f6[5] == "zero-sum signed three-module previous-tick loop"


def test_planning_matrix_contains_no_observed_result_fields() -> None:
    forbidden = {"actual", "observed", "passed", "result", "results", "verdict"}
    assert not (_keys(build_phase_f_v2_matrix()) & forbidden)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value.update({"unexpected": True}),
        lambda value: value["bindings"].update({"unexpected": True}),
        lambda value: value["cases"][0].update({"unexpected": True}),
        lambda value: value["cases"][0]["fixture_blueprint"].update(
            {"unexpected": True}
        ),
        lambda value: value["cases"][0]["role_contract"].update({"observed": 1}),
    ),
)
def test_validator_rejects_unknown_or_observed_fields(mutator) -> None:
    candidate = build_phase_f_v2_matrix()
    mutator(candidate)
    with pytest.raises(PhaseFV2MatrixError):
        validate_phase_f_v2_matrix(candidate)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda value: value["cases"].reverse(),
        lambda value: value["cases"][0]["role_contract"].update(
            {"action": "post_result_repair"}
        ),
        lambda value: value["cases"][0]["execution_accounting"].update(
            {"accelerated_invocations": 3}
        ),
        lambda value: value["bindings"].update({"frozen_execution_commit": "a" * 40}),
        lambda value: value.update({"matrix_sha256": "0" * 64}),
    ),
)
def test_validator_fails_closed_on_semantic_or_digest_drift(mutator) -> None:
    candidate = deepcopy(build_phase_f_v2_matrix())
    mutator(candidate)
    with pytest.raises(PhaseFV2MatrixError):
        validate_phase_f_v2_matrix(candidate)


def test_loader_rejects_duplicate_keys_and_non_finite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"x","schema_version":"y"}', encoding="ascii"
    )
    with pytest.raises(PhaseFV2MatrixError, match="duplicate"):
        load_phase_f_v2_matrix(duplicate)

    non_finite = tmp_path / "nan.json"
    non_finite.write_text('{"schema_version":NaN}', encoding="ascii")
    with pytest.raises(PhaseFV2MatrixError, match="non-finite"):
        load_phase_f_v2_matrix(non_finite)


def test_build_returns_detached_mutable_copies() -> None:
    first = build_phase_f_v2_matrix()
    second = build_phase_f_v2_matrix()
    first["cases"][0]["fixture_blueprint"]["slot_plan"][0]["variant_id"] = "changed"
    assert second["cases"][0]["fixture_blueprint"]["slot_plan"][0]["variant_id"] == "v0"
