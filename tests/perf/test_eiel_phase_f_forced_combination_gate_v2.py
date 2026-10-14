from __future__ import annotations

from copy import deepcopy

import pytest

from evolution.phase_f_v2.matrix import build_phase_f_v2_matrix
from perf.eiel_phase_f_forced_combination_gate_v2 import (
    DECISION_A,
    DECISION_B,
    DECISION_C,
    DECISION_D,
    PhaseFCombinationGateError,
    aggregate_phase_f_terminal_ledger_v2,
    build_cpu_expectation_evidence_v2,
    build_lesion_relationship_evidence_v2,
    compare_phase_f_invocation_v2,
    ordered_phase_f_check_names_v2,
)


SOURCE_COMMIT = "a" * 40
MATRIX_RECEIPT = "b" * 64
ENTRY_RECEIPT = "c" * 64
FIXTURE_SHA256 = "d" * 64
SOURCE_SHA256 = "e" * 64


def _outputs(matrix: dict[str, object]) -> dict[str, object]:
    contract = matrix["comparison_contract"]
    assert isinstance(contract, dict)
    exact = contract["exact_fields"]
    admitted = contract["admitted_float_fields"]
    assert isinstance(exact, list)
    assert isinstance(admitted, list)
    return {
        **{name: {"field": name, "value": 0} for name in exact},
        **{name: [0.0, 1.0] for name in admitted},
    }


def _bindings(
    matrix: dict[str, object],
    case: dict[str, object],
    *,
    kind: str = "uninterrupted",
    repeat_index: int = 0,
    execution_id: str | None = None,
) -> dict[str, object]:
    owner = matrix["owner_contract"]
    assert isinstance(owner, dict)
    fixture = case["fixture_blueprint"]
    checkpoint = case["checkpoint_schedule"]
    assert isinstance(fixture, dict)
    assert isinstance(checkpoint, dict)
    return {
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "execution_id": execution_id or f"{case['case_id']}:{kind}:{repeat_index}",
        "source_commit": SOURCE_COMMIT,
        "matrix_sha256": matrix["matrix_sha256"],
        "matrix_receipt_sha256": MATRIX_RECEIPT,
        "entry_receipt_sha256": ENTRY_RECEIPT,
        "fixture_blueprint_sha256": fixture["fixture_blueprint_sha256"],
        "fixture_sha256": FIXTURE_SHA256,
        "source_sha256": SOURCE_SHA256,
        "owner_id": owner["accelerated_owner"],
        "profile_id": case["profile_id"],
        "lane_id": owner["modular_tick_owner"],
        "invocation_kind": kind,
        "repeat_index": repeat_index,
        "boundary_index": 0,
        "checkpoint_kind": checkpoint["kind"],
        "protocol": deepcopy(matrix["protocol"]),
        "deterministic_action_binding": True,
        "hidden_fallback_count": 0,
        "modular_rng_draw_count": 0,
    }


def _compare(
    matrix: dict[str, object],
    case: dict[str, object],
    *,
    expected: dict[str, object] | None = None,
    observed: dict[str, object] | None = None,
    bindings: dict[str, object] | None = None,
    kind: str = "uninterrupted",
    repeat_index: int = 0,
    execution_id: str | None = None,
) -> dict[str, object]:
    expected_values = expected or _outputs(matrix)
    return compare_phase_f_invocation_v2(
        matrix=matrix,
        bindings=bindings
        or _bindings(
            matrix,
            case,
            kind=kind,
            repeat_index=repeat_index,
            execution_id=execution_id,
        ),
        expected=expected_values,
        observed=observed or deepcopy(expected_values),
        frozen_source_commit=SOURCE_COMMIT,
        matrix_receipt_sha256=MATRIX_RECEIPT,
        entry_receipt_sha256=ENTRY_RECEIPT,
        fixture_sha256=FIXTURE_SHA256,
        source_sha256=SOURCE_SHA256,
    )


def test_float_tolerance_boundary_is_inclusive_and_rtol_is_zero() -> None:
    matrix = build_phase_f_v2_matrix()
    case = matrix["cases"][0]
    expected = _outputs(matrix)
    field = matrix["comparison_contract"]["admitted_float_fields"][0]

    expected[field] = 0.0
    at_boundary = deepcopy(expected)
    at_boundary[field] = 2e-7
    accepted = _compare(matrix, case, expected=expected, observed=at_boundary)
    assert accepted["outcome"] == "no_mismatch"

    expected[field] = 1_000_000.0
    over_boundary = deepcopy(expected)
    over_boundary[field] = 1_000_000.0 + 3e-7
    rejected = _compare(matrix, case, expected=expected, observed=over_boundary)
    assert rejected["outcome"] == "first_difference"
    assert rejected["comparison_receipt"]["check_name"] == f"float:{field}"


def test_nonfinite_float_fails_at_finite_check() -> None:
    matrix = build_phase_f_v2_matrix()
    case = matrix["cases"][0]
    expected = _outputs(matrix)
    observed = deepcopy(expected)
    field = matrix["comparison_contract"]["admitted_float_fields"][0]
    observed[field][0] = float("nan")

    receipt = _compare(matrix, case, expected=expected, observed=observed)

    assert receipt["outcome"] == "first_difference"
    assert receipt["comparison_receipt"]["check_name"] == "finite_float_outputs"


def test_shape_drift_precedes_float_value_comparison() -> None:
    matrix = build_phase_f_v2_matrix()
    case = matrix["cases"][0]
    expected = _outputs(matrix)
    observed = deepcopy(expected)
    field = matrix["comparison_contract"]["admitted_float_fields"][0]
    observed[field] = [[0.0, 1.0]]

    receipt = _compare(matrix, case, expected=expected, observed=observed)

    assert receipt["outcome"] == "first_difference"
    assert receipt["comparison_receipt"]["check_name"] == "float_shape_inventory"


def test_structured_float_fields_are_compared_recursively() -> None:
    matrix = build_phase_f_v2_matrix()
    case = matrix["cases"][0]
    expected = _outputs(matrix)
    field = "state_before_and_after"
    expected[field] = [
        {"before": [0.0, 1.0], "after": [2.0, 3.0]},
    ]

    accepted = _compare(
        matrix,
        case,
        expected=expected,
        observed=deepcopy(expected),
    )

    assert accepted["outcome"] == "no_mismatch"


def test_structured_float_key_drift_fails_at_shape_inventory() -> None:
    matrix = build_phase_f_v2_matrix()
    case = matrix["cases"][0]
    expected = _outputs(matrix)
    field = "state_before_and_after"
    expected[field] = [
        {"before": [0.0, 1.0], "after": [2.0, 3.0]},
    ]
    observed = deepcopy(expected)
    observed[field][0]["after_renamed"] = observed[field][0].pop("after")

    rejected = _compare(
        matrix,
        case,
        expected=expected,
        observed=observed,
    )

    assert rejected["outcome"] == "first_difference"
    assert rejected["comparison_receipt"]["check_name"] == "float_shape_inventory"


def test_checkpoint_projection_uses_named_float_comparison_for_evolving_state() -> None:
    matrix = build_phase_f_v2_matrix()
    case = matrix["cases"][0]
    expected = _outputs(matrix)
    expected["checkpoint_projection"] = {
        "planes_after": [
            {
                "module_previous_output": {
                    "dtype": "torch.float32",
                    "shape": [1, 16, 1],
                    "sha256": "1" * 64,
                },
                "module_state": {
                    "dtype": "torch.float32",
                    "shape": [1, 16, 1],
                    "sha256": "2" * 64,
                },
                "module_weight": {
                    "dtype": "torch.float32",
                    "shape": [1, 16, 1],
                    "sha256": "3" * 64,
                },
            }
        ]
    }
    observed = deepcopy(expected)
    observed["checkpoint_projection"]["planes_after"][0]["module_state"][
        "sha256"
    ] = "4" * 64
    observed["checkpoint_projection"]["planes_after"][0][
        "module_previous_output"
    ]["sha256"] = "5" * 64

    accepted = _compare(matrix, case, expected=expected, observed=observed)

    assert accepted["outcome"] == "no_mismatch"


def test_checkpoint_projection_keeps_nonstate_float_hashes_exact() -> None:
    matrix = build_phase_f_v2_matrix()
    case = matrix["cases"][0]
    expected = _outputs(matrix)
    expected["checkpoint_projection"] = {
        "planes_after": [
            {
                "module_previous_output": {
                    "dtype": "torch.float32",
                    "shape": [1, 16, 1],
                    "sha256": "1" * 64,
                },
                "module_state": {
                    "dtype": "torch.float32",
                    "shape": [1, 16, 1],
                    "sha256": "2" * 64,
                },
                "module_weight": {
                    "dtype": "torch.float32",
                    "shape": [1, 16, 1],
                    "sha256": "3" * 64,
                },
            }
        ]
    }
    observed = deepcopy(expected)
    observed["checkpoint_projection"]["planes_after"][0]["module_weight"][
        "sha256"
    ] = "4" * 64

    rejected = _compare(matrix, case, expected=expected, observed=observed)

    assert rejected["outcome"] == "first_difference"
    assert rejected["comparison_receipt"]["check_name"] == "exact:checkpoint_projection"


def test_first_difference_uses_frozen_order_not_mapping_order() -> None:
    matrix = build_phase_f_v2_matrix()
    case = matrix["cases"][0]
    expected = _outputs(matrix)
    observed = deepcopy(expected)
    exact_field = matrix["comparison_contract"]["exact_fields"][0]
    observed[exact_field] = "wrong"
    bindings = _bindings(matrix, case)
    bindings["source_commit"] = "d" * 40

    receipt = _compare(
        matrix,
        case,
        expected=expected,
        observed=observed,
        bindings=bindings,
    )

    assert receipt["comparison_receipt"]["check_name"] == "source_commit_binding"
    names = ordered_phase_f_check_names_v2(matrix)
    assert names.index("source_commit_binding") < names.index(f"exact:{exact_field}")


def test_missing_or_invented_observed_field_fails_inventory_first() -> None:
    matrix = build_phase_f_v2_matrix()
    case = matrix["cases"][0]
    expected = _outputs(matrix)
    missing = deepcopy(expected)
    missing.pop(matrix["comparison_contract"]["exact_fields"][0])
    missing_receipt = _compare(matrix, case, expected=expected, observed=missing)
    assert missing_receipt["comparison_receipt"]["check_name"] == (
        "output_field_inventory"
    )

    invented = deepcopy(expected)
    invented["invented"] = 1
    invented_receipt = _compare(matrix, case, expected=expected, observed=invented)
    assert invented_receipt["comparison_receipt"]["check_name"] == (
        "output_field_inventory"
    )


def _complete_evidence(
    matrix: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cpu: list[dict[str, object]] = []
    invocations: list[dict[str, object]] = []
    lesions: list[dict[str, object]] = []
    expected = _outputs(matrix)
    cases = matrix["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, dict)
        cpu.append(
            build_cpu_expectation_evidence_v2(
                case=case,
                matrix_sha256=matrix["matrix_sha256"],
                fixture_sha256=FIXTURE_SHA256,
                source_sha256=SOURCE_SHA256,
                expectation=expected,
                expectation_kind="uninterrupted",
                status="passed",
            )
        )
        accounting = case["execution_accounting"]
        assert isinstance(accounting, dict)
        for kind in ("uninterrupted", "resumed"):
            repeats = accounting[f"{kind}_repeats"]
            assert isinstance(repeats, int)
            if kind == "resumed" and repeats:
                cpu.append(
                    build_cpu_expectation_evidence_v2(
                        case=case,
                        matrix_sha256=matrix["matrix_sha256"],
                        fixture_sha256=FIXTURE_SHA256,
                        source_sha256=SOURCE_SHA256,
                        expectation=expected,
                        expectation_kind="resumed",
                        status="passed",
                    )
                )
            for repeat_index in range(repeats):
                invocations.append(
                    _compare(
                        matrix,
                        case,
                        expected=expected,
                        observed=deepcopy(expected),
                        kind=kind,
                        repeat_index=repeat_index,
                    )
                )
        role = case["role_contract"]
        assert isinstance(role, dict)
        if role["claim_kind"] == "matched_causal_lesion":
            lesions.append(
                build_lesion_relationship_evidence_v2(case=case, status="passed")
            )
    return cpu, invocations, lesions


def _aggregate(
    matrix: dict[str, object],
    cpu: list[dict[str, object]],
    invocations: list[dict[str, object]],
    lesions: list[dict[str, object]],
    *,
    blocked: bool = False,
) -> dict[str, object]:
    return aggregate_phase_f_terminal_ledger_v2(
        matrix=matrix,
        frozen_source_commit=SOURCE_COMMIT,
        matrix_receipt_sha256=MATRIX_RECEIPT,
        entry_receipt_sha256=ENTRY_RECEIPT,
        cpu_expectations=cpu,
        invocation_receipts=invocations,
        lesion_relationships=lesions,
        blocked=blocked,
    )


def test_terminal_ledger_exact_counts_and_precedence() -> None:
    matrix = build_phase_f_v2_matrix()
    cpu, invocations, lesions = _complete_evidence(matrix)

    green = _aggregate(matrix, cpu, invocations, lesions)
    assert green["decision"] == DECISION_A
    assert green["complete"] is True
    assert green["observed_counts"] == green["expected_counts"]
    assert green["claims"]["bounded_composition_supported"] is True

    incomplete = _aggregate(matrix, cpu, invocations[:-1], lesions)
    assert incomplete["decision"] == DECISION_D
    assert incomplete["complete"] is False

    failed_lesions = deepcopy(lesions)
    failed_lesions[0] = build_lesion_relationship_evidence_v2(
        case=next(
            case
            for case in matrix["cases"]
            if case["case_id"] == failed_lesions[0]["case_id"]
        ),
        status="failed",
    )
    composition_over_incomplete = _aggregate(
        matrix, cpu, invocations[:-1], failed_lesions, blocked=True
    )
    assert composition_over_incomplete["decision"] == DECISION_B

    failed_invocations = deepcopy(invocations[:-1])
    failed_case = next(
        case
        for case in matrix["cases"]
        if case["case_id"] == failed_invocations[0]["case_id"]
    )
    failed_observed = _outputs(matrix)
    first_exact = matrix["comparison_contract"]["exact_fields"][0]
    failed_observed[first_exact] = "parity-failure"
    failed_invocations[0] = _compare(
        matrix,
        failed_case,
        observed=failed_observed,
        kind=failed_invocations[0]["invocation_kind"],
        repeat_index=failed_invocations[0]["repeat_index"],
    )
    integrity_over_all = _aggregate(
        matrix, cpu, failed_invocations, failed_lesions, blocked=True
    )
    assert integrity_over_all["decision"] == DECISION_C


def test_skips_and_extra_fields_never_enter_a_green_ledger() -> None:
    matrix = build_phase_f_v2_matrix()
    case = matrix["cases"][0]
    with pytest.raises(PhaseFCombinationGateError, match="cannot skip"):
        build_cpu_expectation_evidence_v2(
            case=case,
            matrix_sha256=matrix["matrix_sha256"],
            fixture_sha256=FIXTURE_SHA256,
            source_sha256=SOURCE_SHA256,
            expectation=_outputs(matrix),
            expectation_kind="uninterrupted",
            status="skipped",
        )

    cpu, invocations, lesions = _complete_evidence(matrix)
    invented = deepcopy(cpu[0])
    invented["skip_reason"] = "invented"
    with pytest.raises(PhaseFCombinationGateError, match="fields are not exact"):
        _aggregate(matrix, [invented, *cpu[1:]], invocations, lesions)

    mismatched_cpu = deepcopy(cpu)
    mismatched_case = next(
        row for row in matrix["cases"] if row["case_id"] == cpu[0]["case_id"]
    )
    mismatched_cpu[0] = build_cpu_expectation_evidence_v2(
        case=mismatched_case,
        matrix_sha256=matrix["matrix_sha256"],
        fixture_sha256=FIXTURE_SHA256,
        source_sha256="f" * 64,
        expectation=_outputs(matrix),
        expectation_kind=cpu[0]["expectation_kind"],
        status="passed",
    )
    with pytest.raises(PhaseFCombinationGateError, match="contradicts CPU"):
        _aggregate(matrix, mismatched_cpu, invocations, lesions)
