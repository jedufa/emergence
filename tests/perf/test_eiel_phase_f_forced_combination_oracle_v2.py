from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from evolution.phase_f_v2.canonical import canonical_sha256
from evolution.phase_f_v2.matrix import build_phase_f_v2_matrix
from perf.eiel_phase_f_forced_combination_fixture_v2 import (
    build_phase_f_fixture_registry_v2,
)
from perf.eiel_phase_f_execution_source_v2 import (
    prepare_phase_f_execution_source_v2,
)
from perf.eiel_phase_f_forced_combination_oracle_v2 import (
    EXPECTED_AGENT_COUNT,
    PhaseFV2OracleError,
    build_phase_f_cpu_expectation_v2,
    build_phase_f_cpu_oracle_plan_v2,
    validate_phase_f_cpu_expectation_v2,
)
from perf.eiel_phase_f_forced_combination_source_v2 import (
    materialize_phase_f_source_v2,
)


@pytest.fixture(scope="module")
def matrix() -> dict[str, object]:
    return build_phase_f_v2_matrix()


@pytest.fixture(scope="module")
def registry(matrix: dict[str, object]) -> dict[str, object]:
    return build_phase_f_fixture_registry_v2(matrix)


def _matrix_case(
    matrix: dict[str, object], fixture: dict[str, object]
) -> dict[str, object]:
    return next(
        case for case in matrix["cases"] if case["case_id"] == fixture["case_id"]
    )


def _source(matrix: dict[str, object], fixture: dict[str, object]):
    case = _matrix_case(matrix, fixture)
    return materialize_phase_f_source_v2(
        fixture, source_seed=case["fixture_blueprint"]["construction_seed"]
    )


def test_all_120_cases_build_current_exact_oracle_plans(
    matrix: dict[str, object], registry: dict[str, object]
) -> None:
    plans = [
        build_phase_f_cpu_oracle_plan_v2(matrix=matrix, fixture_case=fixture)
        for fixture in registry["cases"]
    ]

    assert len(plans) == 120
    assert len({plan["case_id"] for plan in plans}) == 120
    assert len({plan["plan_sha256"] for plan in plans}) == 120
    assert {plan["profile_id"] for plan in plans} == {
        "b2-12x32-v1",
        "b2-16x48-v1",
    }
    exact = matrix["comparison_contract"]["exact_fields"]
    admitted = matrix["comparison_contract"]["admitted_float_fields"]
    assert all(plan["exact_fields"] == exact for plan in plans)
    assert all(plan["admitted_float_fields"] == admitted for plan in plans)
    assert all(plan["execution_authorized"] is False for plan in plans)


def _representative_fixtures(
    matrix: dict[str, object], registry: dict[str, object]
) -> list[dict[str, object]]:
    by_case = {fixture["case_id"]: fixture for fixture in registry["cases"]}
    return [
        by_case[case["case_id"]] for case in matrix["cases"] if case["role_id"] == "r0"
    ]


@pytest.fixture(scope="module")
def representative_contracts(
    matrix: dict[str, object], registry: dict[str, object]
) -> list[tuple[dict[str, object], object, dict[str, object]]]:
    result = []
    for fixture in _representative_fixtures(matrix, registry):
        source = _source(matrix, fixture)
        contract = build_phase_f_cpu_expectation_v2(
            matrix=matrix,
            fixture_case=fixture,
            source_state=source,
        )
        result.append((fixture, source, contract))
    return result


def test_real_formal_execution_covers_every_family_and_both_profiles(
    matrix: dict[str, object],
    representative_contracts: list[tuple[dict[str, object], object, dict[str, object]]],
) -> None:
    assert len(representative_contracts) == 24
    fixtures = [fixture for fixture, _, _ in representative_contracts]
    contracts = [contract for _, _, contract in representative_contracts]
    assert {fixture["family_id"] for fixture in fixtures} == {
        f"f{index:02d}" for index in range(1, 13)
    }
    assert {contract["profile_id"] for contract in contracts} == {
        "b2-12x32-v1",
        "b2-16x48-v1",
    }
    required = set(matrix["comparison_contract"]["exact_fields"]) | set(
        matrix["comparison_contract"]["admitted_float_fields"]
    )
    for contract in contracts:
        assert contract["agent_count"] == EXPECTED_AGENT_COUNT
        assert len(contract["per_agent_raw_sha256"]) == EXPECTED_AGENT_COUNT
        assert len(set(contract["per_agent_raw_sha256"])) == EXPECTED_AGENT_COUNT
        assert set(contract["expectation"]) == required
        assert contract["fallback_count"] == 0
        assert contract["modular_rng_draw_count"] == 0
        assert (
            contract["cpu_expectation_evidence"]["expectation_sha256"]
            == (contract["expectation_sha256"])
        )


def test_expectations_are_real_source_dependent_surfaces_not_placeholders(
    representative_contracts: list[tuple[dict[str, object], object, dict[str, object]]],
) -> None:
    contracts = [contract for _, _, contract in representative_contracts]
    assert len({contract["expectation_sha256"] for contract in contracts}) == 24
    for field in (
        "module_output",
        "module_state",
        "previous_output",
        "developmental_effective_gain",
        "routed_observation",
        "structural_cost_components_and_total",
    ):
        assert (
            len(
                {
                    canonical_sha256(contract["expectation"][field])
                    for contract in contracts
                }
            )
            > 1
        ), field
    assert all(
        contract["expectation"]["checkpoint_projection"]["planes_before"]
        and contract["expectation"]["checkpoint_projection"]["planes_after"]
        for contract in contracts
    )


def test_mature_and_late_stage_expectations_are_executed_not_labeled(
    representative_contracts: list[tuple[dict[str, object], object, dict[str, object]]],
) -> None:
    for fixture, _, contract in representative_contracts:
        stages = contract["expectation"]["developmental_stage"]
        expected = [0] * EXPECTED_AGENT_COUNT
        affected_slots: tuple[int, ...] = ()
        if fixture["family_id"] == "f03":
            expected[6] = expected[14] = 1
            affected_slots = (6, 14)
        elif fixture["family_id"] == "f07":
            expected[1] = expected[9] = 1
            expected[2] = expected[10] = 2
            affected_slots = (1, 2, 9, 10)
        assert stages == expected
        for slot in affected_slots:
            assert (
                contract["expectation"]["developmental_effective_gain"][slot][1] == 1.0
            )


def test_f1_disabled_control_has_zero_modular_contribution_and_unchanged_planes(
    matrix: dict[str, object], registry: dict[str, object]
) -> None:
    controls = {}
    for role_id in ("r0", "r1"):
        fixture = next(
            row
            for row in registry["cases"]
            if row["family_id"] == "f01"
            and row["profile_id"] == "b2-12x32-v1"
            and row["role_id"] == role_id
        )
        source = _source(matrix, fixture)
        controls[role_id] = build_phase_f_cpu_expectation_v2(
            matrix=matrix,
            fixture_case=fixture,
            source_state=source,
        )

    disabled = controls["r0"]["expectation"]
    enabled = controls["r1"]["expectation"]
    assert disabled["profile_and_owner_ids"]["modular_enabled"] is False
    assert enabled["profile_and_owner_ids"]["modular_enabled"] is True
    assert all(
        all(value == 0.0 for value in row) for row in disabled["routed_action_residual"]
    )
    assert all(
        before == after
        for before, after in zip(
            disabled["checkpoint_projection"]["planes_before"],
            disabled["checkpoint_projection"]["planes_after"],
            strict=True,
        )
    )
    assert enabled["routed_action_residual"] != disabled["routed_action_residual"]


def test_formal_expectation_repeats_deterministically(
    matrix: dict[str, object], registry: dict[str, object]
) -> None:
    fixture = registry["cases"][2]
    source = _source(matrix, fixture)
    first = build_phase_f_cpu_expectation_v2(
        matrix=matrix, fixture_case=fixture, source_state=source
    )
    second = build_phase_f_cpu_expectation_v2(
        matrix=matrix, fixture_case=fixture, source_state=source
    )

    assert first == second
    assert first["expectation_sha256"] == canonical_sha256(first["expectation"])


def test_checkpoint_case_accepts_distinct_fresh_owner_restored_source(
    matrix: dict[str, object], registry: dict[str, object]
) -> None:
    by_case = {fixture["case_id"]: fixture for fixture in registry["cases"]}
    case = next(
        case
        for case in matrix["cases"]
        if case["checkpoint_schedule"]["kind"] != "none"
    )
    fixture = by_case[case["case_id"]]
    source = _source(matrix, fixture)
    restored = _source(matrix, fixture)

    receipt = build_phase_f_cpu_expectation_v2(
        matrix=matrix,
        fixture_case=fixture,
        source_state=source,
        expectation_kind="resumed",
        restored_source_state=restored,
    )

    assert receipt["expectation_kind"] == "resumed"
    assert receipt["fresh_owner_restored"] is True
    assert receipt["checkpoint_kind"] != "none"
    assert receipt["source_sha256"] == receipt["execution_source_sha256"]


def test_nan_missing_surface_and_tamper_are_rejected(
    matrix: dict[str, object],
    representative_contracts: list[tuple[dict[str, object], object, dict[str, object]]],
) -> None:
    fixture, source, contract = representative_contracts[0]

    nan_contract = deepcopy(contract)
    nan_contract["expectation"]["action_logits"][0][0] = float("nan")
    with pytest.raises(PhaseFV2OracleError, match="nonfinite"):
        validate_phase_f_cpu_expectation_v2(
            nan_contract,
            matrix=matrix,
            fixture_case=fixture,
            source_state=source,
        )

    missing = deepcopy(contract)
    missing["expectation"].pop("module_output")
    with pytest.raises(PhaseFV2OracleError, match="fields are not exact"):
        validate_phase_f_cpu_expectation_v2(
            missing,
            matrix=matrix,
            fixture_case=fixture,
            source_state=source,
        )

    tampered = deepcopy(contract)
    tampered["expectation"]["actions"][0] += 1
    with pytest.raises(PhaseFV2OracleError, match="expectation self digest mismatch"):
        validate_phase_f_cpu_expectation_v2(
            tampered,
            matrix=matrix,
            fixture_case=fixture,
            source_state=source,
        )


def test_noncurrent_fixture_and_source_fail_closed(
    matrix: dict[str, object], registry: dict[str, object]
) -> None:
    fixture = registry["cases"][0]
    source = _source(matrix, fixture)
    noncurrent_fixture = deepcopy(fixture)
    noncurrent_fixture["profile_id"] = "not-current"
    with pytest.raises(PhaseFV2OracleError, match="fixture case is not current"):
        build_phase_f_cpu_oracle_plan_v2(matrix=matrix, fixture_case=noncurrent_fixture)

    noncurrent_source = replace(source, source_sha256="d" * 64)
    with pytest.raises(PhaseFV2OracleError, match="ordinary source is not current"):
        build_phase_f_cpu_expectation_v2(
            matrix=matrix,
            fixture_case=fixture,
            source_state=noncurrent_source,
        )


def test_boundary_prepared_source_uses_receipt_instead_of_raw_fixture(
    matrix: dict[str, object], registry: dict[str, object]
) -> None:
    case_id = "pf2_f09_b12_r2"
    fixture = next(row for row in registry["cases"] if row["case_id"] == case_id)
    prepared = prepare_phase_f_execution_source_v2(
        matrix=matrix,
        fixture_registry=registry,
        case_id=case_id,
    )

    receipt = build_phase_f_cpu_expectation_v2(
        matrix=matrix,
        fixture_case=fixture,
        fixture_registry=registry,
        source_state=prepared.source,
        source_preparation_receipt=prepared.preparation_receipt,
    )

    assert receipt["source_sha256"] == prepared.source.source_sha256
    assert receipt["source_seed"] == prepared.source.source_seed
