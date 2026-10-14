"""Receipt assertions and expected-negative validation."""

from __future__ import annotations

from ..io import InputStore, ReviewInputError, assertion_matches, json_pointer
from ..model import CheckResult


def _validate_assertion(document: object, assertion: dict) -> str | None:
    pointer = assertion["pointer"]
    operator = assertion["operator"]
    if operator == "exists":
        try:
            json_pointer(document, pointer)
        except ReviewInputError:
            return f"expected JSON pointer to exist: {pointer}"
        return None
    try:
        actual = json_pointer(document, pointer)
        matched = assertion_matches(actual, operator, assertion.get("expected"))
    except ReviewInputError as exc:
        return str(exc)
    if matched:
        return None
    return (
        f"assertion failed at {pointer}: operator={operator}, "
        f"expected={assertion.get('expected')!r}, actual={actual!r}"
    )


def validate_json_assertion(spec: dict, store: InputStore) -> CheckResult:
    result = CheckResult(spec["id"], "json_assertion")
    try:
        document = store.read_json(spec["path"])
        error = _validate_assertion(document, spec)
        if error:
            result.fail(error)
        result.details = {
            "path": spec["path"],
            "pointer": spec["pointer"],
            "operator": spec["operator"],
        }
    except ReviewInputError as exc:
        result.fail(str(exc))
    return result


def validate_expected_negative(spec: dict, store: InputStore) -> CheckResult:
    result = CheckResult(spec["id"], "expected_negative")
    try:
        receipt = store.read_json(spec["path"])
        actual_code = json_pointer(receipt, spec.get("exit_code_pointer", "/exit_code"))
        expected_code = spec["expected_exit_code"]
        if actual_code != expected_code:
            result.fail(
                f"expected negative exit code {expected_code}, got {actual_code}"
            )
        reasons_pointer = spec.get("reasons_pointer", "/reason_codes")
        actual_reasons = json_pointer(receipt, reasons_pointer)
        expected_reasons = spec["expected_reasons"]
        if spec.get("ordered_reasons", True):
            reasons_match = actual_reasons == expected_reasons
        else:
            reasons_match = (
                isinstance(actual_reasons, list)
                and len(actual_reasons) == len(expected_reasons)
                and set(actual_reasons) == set(expected_reasons)
            )
        if not reasons_match:
            result.fail(
                f"expected exact reasons {expected_reasons!r}, got {actual_reasons!r}"
            )
        for assertion in spec.get("state_assertions", []):
            error = _validate_assertion(receipt, assertion)
            if error:
                result.fail(error)
        result.details = {
            "path": spec["path"],
            "exit_code": actual_code,
            "reason_count": len(actual_reasons) if isinstance(actual_reasons, list) else None,
            "state_assertion_count": len(spec.get("state_assertions", [])),
        }
    except ReviewInputError as exc:
        result.fail(str(exc))
    return result


validate_inline_assertion = _validate_assertion
