from __future__ import annotations

from dataclasses import replace

import pytest

from engine.campaigns.admission_kernel_v1 import (
    ADMISSION_CHECK_ORDER_V1,
    AdmissionCheckV1,
    AdmissionKernelError,
    AdmissionKernelRequestV1,
    assert_admission_policy_parity_v1,
    baseline_admission_shadow_v1,
    evaluate_admission_kernel_v1,
)
from engine.campaigns.execution_front_door_v1 import (
    _configuration_admission_allowed_v1,
)
from engine.qrr.production_front_door_v1 import (
    _qrr_baseline_command_allowed_v1,
)


def _checks(
    *,
    states: dict[str, str] | None = None,
    enforced: dict[str, bool] | None = None,
) -> tuple[AdmissionCheckV1, ...]:
    states = states or {}
    enforced = enforced or {}
    return tuple(
        AdmissionCheckV1(
            check_id,
            states.get(check_id, "PASS"),
            enforced.get(check_id, True),
            "a" * 64,
            (
                (f"{check_id.upper()}_{states[check_id]}",)
                if states.get(check_id) in {"FAIL", "WAIT"}
                else ()
            ),
        )
        for check_id in ADMISSION_CHECK_ORDER_V1
    )


def _request(
    *,
    states: dict[str, str] | None = None,
    enforced: dict[str, bool] | None = None,
) -> AdmissionKernelRequestV1:
    return AdmissionKernelRequestV1(
        front_door_id="fixture-front-door-v1",
        intent="execution",
        request_sha256="request-fixture",
        plan_sha256="plan-fixture",
        checks=_checks(states=states, enforced=enforced),
        allow_effect_request="REQUEST_RESOURCE_AND_CONTINUATION_LEASE",
    )


def _baseline_receipt(
    *, configuration_eligible: bool, admitted: bool
) -> dict[str, object]:
    reason_codes = (
        [] if configuration_eligible and admitted else ["UNDECLARED_BASELINE_DEVIATION"]
    )
    return {
        "configuration_eligible": configuration_eligible,
        "admitted": admitted,
        "reason_codes": reason_codes,
        "receipt_sha256": "b" * 64,
        "requirements": [
            {
                "requirement_id": "artifacts.lifecycle_contract",
                "configuration_conformant": True,
            }
        ],
    }


def test_kernel_is_deterministic_content_addressed_and_performs_no_effects() -> None:
    request = _request()
    first = evaluate_admission_kernel_v1(request)
    second = evaluate_admission_kernel_v1(request)

    assert first == second
    assert first["decision"] == {
        "status": "ALLOW",
        "allowed": True,
        "waiting": False,
        "reason_codes": [],
        "effect_request": "REQUEST_RESOURCE_AND_CONTINUATION_LEASE",
    }
    assert first["effects_performed"] is False
    assert len(first["receipt_sha256"]) == 64


@pytest.mark.parametrize("failed_check", ADMISSION_CHECK_ORDER_V1)
def test_every_g4_policy_authority_fails_closed(failed_check: str) -> None:
    receipt = evaluate_admission_kernel_v1(_request(states={failed_check: "FAIL"}))

    assert receipt["decision"] == {
        "status": "REJECT",
        "allowed": False,
        "waiting": False,
        "reason_codes": [f"{failed_check.upper()}_FAIL"],
        "effect_request": "NONE",
    }


def test_rejection_precedes_wait_and_wait_performs_no_effect() -> None:
    waiting = evaluate_admission_kernel_v1(
        _request(states={"resource_admission": "WAIT"})
    )
    rejected = evaluate_admission_kernel_v1(
        _request(
            states={
                "representability_exact": "FAIL",
                "resource_admission": "WAIT",
            }
        )
    )

    assert waiting["decision"]["status"] == "WAIT_RESOURCE"
    assert waiting["decision"]["effect_request"] == "NONE"
    assert rejected["decision"]["status"] == "REJECT"
    assert rejected["decision"]["reason_codes"] == ["REPRESENTABILITY_EXACT_FAIL"]


def test_non_enforced_observation_is_retained_without_regaining_authority() -> None:
    receipt = evaluate_admission_kernel_v1(
        _request(
            states={"artifact_lifecycle": "FAIL"},
            enforced={"artifact_lifecycle": False},
        )
    )

    assert receipt["decision"]["status"] == "ALLOW"
    lifecycle = receipt["request"]["checks"][3]
    assert lifecycle["state"] == "FAIL"
    assert lifecycle["enforced"] is False


def test_malformed_or_incomplete_policy_input_fails_closed() -> None:
    request = _request()
    with pytest.raises(AdmissionKernelError, match="complete, unique"):
        replace(request, checks=request.checks[:-1])
    with pytest.raises(AdmissionKernelError, match="cannot be NOT_APPLICABLE"):
        AdmissionCheckV1("baseline", "NOT_APPLICABLE", True)
    with pytest.raises(AdmissionKernelError, match="requires a reason"):
        AdmissionCheckV1("no_claims", "FAIL", True)


@pytest.mark.parametrize(
    ("configuration_eligible", "admitted", "execution_required", "expected"),
    (
        (True, True, False, True),
        (False, False, False, False),
        (True, True, True, True),
        (True, False, True, False),
    ),
)
def test_baseline_shadow_is_exact_for_launchpad_and_qrr_policy(
    configuration_eligible: bool,
    admitted: bool,
    execution_required: bool,
    expected: bool,
) -> None:
    admission = _baseline_receipt(
        configuration_eligible=configuration_eligible,
        admitted=admitted,
    )
    receipt = baseline_admission_shadow_v1(
        front_door_id="fixture-front-door-v1",
        intent="execution" if execution_required else "configuration",
        request_sha256="request-fixture",
        plan_sha256="plan-fixture",
        admission_receipt=admission,
        execution_required=execution_required,
    )

    assert receipt["decision"]["allowed"] is expected
    assert receipt["decision"]["waiting"] is False


def test_launchpad_configuration_intent_is_cut_over_with_direct_rollback() -> None:
    admitted = _baseline_receipt(configuration_eligible=True, admitted=False)
    rejected = _baseline_receipt(configuration_eligible=False, admitted=False)

    assert (
        _configuration_admission_allowed_v1(
            admission_receipt=admitted,
            request_sha256="request",
            plan_sha256="plan",
        )
        is True
    )
    assert (
        _configuration_admission_allowed_v1(
            admission_receipt=rejected,
            request_sha256="request",
            plan_sha256="plan",
        )
        is False
    )


def test_qrr_retains_direct_policy_while_exercising_shared_shadow() -> None:
    admitted = _baseline_receipt(configuration_eligible=True, admitted=True)
    execution_rejected = _baseline_receipt(configuration_eligible=True, admitted=False)

    assert (
        _qrr_baseline_command_allowed_v1(
            admission_receipt=admitted,
            command="check-representability",
            request_sha256="request",
            plan_sha256="plan",
        )
        is True
    )
    assert (
        _qrr_baseline_command_allowed_v1(
            admission_receipt=execution_rejected,
            command="run",
            request_sha256="request",
            plan_sha256="plan",
        )
        is False
    )


def test_parity_assertion_rejects_any_direct_policy_divergence() -> None:
    receipt = evaluate_admission_kernel_v1(_request())
    with pytest.raises(AdmissionKernelError, match="shadow diverged"):
        assert_admission_policy_parity_v1(
            receipt,
            legacy_allowed=False,
            legacy_waiting=False,
            legacy_reason_codes=("LEGACY_REJECT",),
            legacy_effect_request="NONE",
        )
