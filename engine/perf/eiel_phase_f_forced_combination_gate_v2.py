"""Data-only Phase F v2 comparison gate and terminal evidence ledger.

This module compares already-produced evidence.  It has no execution, dispatch,
CUDA, checkpoint, or authorization path.  Every receipt it creates is explicitly
evidence-only and keeps ``execution_authorized`` false.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import re
from typing import Any, Final

from evolution.phase_f_v2.canonical import canonical_bytes, canonical_sha256
from evolution.phase_f_v2.first_difference import (
    build_first_difference_record,
    validate_first_difference_record,
)
from evolution.phase_f_v2.matrix import validate_phase_f_v2_matrix


INVOCATION_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_phase_f_v2_invocation_comparison_receipt_v1"
)
CPU_EXPECTATION_SCHEMA_VERSION: Final[str] = (
    "eiel_phase_f_v2_cpu_expectation_evidence_v1"
)
LESION_RELATIONSHIP_SCHEMA_VERSION: Final[str] = (
    "eiel_phase_f_v2_lesion_relationship_evidence_v1"
)
TERMINAL_LEDGER_SCHEMA_VERSION: Final[str] = "eiel_phase_f_v2_terminal_ledger_v1"
EVIDENCE_ONLY_AUTHORITY: Final[str] = "evidence_only"

DECISION_A: Final[str] = "PHASE-F-V2-A"
DECISION_B: Final[str] = "PHASE-F-V2-B"
DECISION_C: Final[str] = "PHASE-F-V2-C"
DECISION_D: Final[str] = "PHASE-F-V2-D"

EXPECTED_LOGICAL_CASES: Final[int] = 120
EXPECTED_CPU_EVALUATIONS: Final[int] = 134
EXPECTED_UNINTERRUPTED_INVOCATIONS: Final[int] = 240
EXPECTED_RESUMED_INVOCATIONS: Final[int] = 28
EXPECTED_ACCELERATED_INVOCATIONS: Final[int] = 268
EXPECTED_TOTAL_EVALUATIONS: Final[int] = 402
EXPECTED_CHECKPOINT_CASES: Final[int] = 14
EXPECTED_LESION_RELATIONSHIPS: Final[int] = 42

_SHA256_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{40}\Z")
_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,255}\Z")
_VALIDATED_MATRIX_CACHE: dict[str, dict[str, Any]] = {}
_CROSS_BACKEND_ADMITTED_CHECKPOINT_PLANES: Final[frozenset[str]] = frozenset(
    {"module_previous_output", "module_state"}
)

_BINDING_FIELDS = frozenset(
    {
        "case_id",
        "case_sha256",
        "execution_id",
        "source_commit",
        "matrix_sha256",
        "matrix_receipt_sha256",
        "entry_receipt_sha256",
        "fixture_blueprint_sha256",
        "fixture_sha256",
        "source_sha256",
        "owner_id",
        "profile_id",
        "lane_id",
        "invocation_kind",
        "repeat_index",
        "boundary_index",
        "checkpoint_kind",
        "protocol",
        "deterministic_action_binding",
        "hidden_fallback_count",
        "modular_rng_draw_count",
    }
)
_INVOCATION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "authority",
        "execution_authorized",
        "case_id",
        "case_sha256",
        "execution_id",
        "source_commit",
        "matrix_sha256",
        "matrix_receipt_sha256",
        "entry_receipt_sha256",
        "fixture_blueprint_sha256",
        "fixture_sha256",
        "source_sha256",
        "expected_sha256",
        "observed_sha256",
        "owner_id",
        "profile_id",
        "lane_id",
        "invocation_kind",
        "repeat_index",
        "boundary_index",
        "checkpoint_kind",
        "outcome",
        "comparison_receipt",
        "receipt_sha256",
    }
)
_CPU_EXPECTATION_FIELDS = frozenset(
    {
        "schema_version",
        "authority",
        "execution_authorized",
        "case_id",
        "case_sha256",
        "matrix_sha256",
        "fixture_sha256",
        "source_sha256",
        "expectation_sha256",
        "expectation_kind",
        "status",
        "receipt_sha256",
    }
)
_LESION_RELATIONSHIP_FIELDS = frozenset(
    {
        "schema_version",
        "authority",
        "execution_authorized",
        "case_id",
        "case_sha256",
        "base_case_id",
        "neutral_case_id",
        "status",
        "receipt_sha256",
    }
)

_ORDERED_PREFIX_CHECKS: Final[tuple[str, ...]] = (
    "execution_identity",
    "source_commit_binding",
    "entry_receipt_binding",
    "matrix_receipt_binding",
    "case_digest_binding",
    "fixture_blueprint_binding",
    "fixture_digest_binding",
    "source_state_binding",
    "owner_identity",
    "profile_identity",
    "single_cell_protocol",
    "fixed_envelope_shapes",
    "deterministic_action_binding",
    "lifecycle_exclusions",
    "hidden_fallback_zero",
    "modular_rng_zero",
    "output_field_inventory",
    "float_shape_inventory",
    "finite_float_outputs",
)


class PhaseFCombinationGateError(ValueError):
    """Phase F evidence is malformed, ambiguous, or outside the frozen gate."""


def _exact_mapping(value: Any, *, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise PhaseFCombinationGateError(f"{label} must be a string-keyed mapping")
    result = dict(value)
    missing = sorted(fields - set(result))
    extra = sorted(set(result) - fields)
    if missing or extra:
        raise PhaseFCombinationGateError(
            f"{label} fields are not exact; missing={missing!r} extra={extra!r}"
        )
    return result


def _sha256(value: Any, *, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise PhaseFCombinationGateError(f"{field} must be a lowercase SHA-256")
    return value


def _commit(value: Any, *, field: str) -> str:
    if type(value) is not str or _COMMIT_RE.fullmatch(value) is None:
        raise PhaseFCombinationGateError(f"{field} must be a full lowercase commit")
    return value


def _token(value: Any, *, field: str) -> str:
    if type(value) is not str or _TOKEN_RE.fullmatch(value) is None:
        raise PhaseFCombinationGateError(f"{field} must be a stable token")
    return value


def _nonnegative_index(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise PhaseFCombinationGateError(f"{field} must be a nonnegative integer")
    return value


def _checked_matrix(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        claimed_digest = value.get("matrix_sha256")
        if type(claimed_digest) is str and _SHA256_RE.fullmatch(claimed_digest):
            unsigned = dict(value)
            unsigned.pop("matrix_sha256", None)
            try:
                digest_matches = canonical_sha256(unsigned) == claimed_digest
            except (TypeError, ValueError):
                digest_matches = False
            if digest_matches and claimed_digest in _VALIDATED_MATRIX_CACHE:
                return _VALIDATED_MATRIX_CACHE[claimed_digest]
    try:
        matrix = validate_phase_f_v2_matrix(value)
    except (TypeError, ValueError) as exc:
        raise PhaseFCombinationGateError(
            "matrix is not the current canonical matrix"
        ) from exc
    cases = matrix.get("cases")
    if type(cases) is not list:
        raise PhaseFCombinationGateError("matrix cases must be a list")
    checkpoint_cases = [
        case for case in cases if case["checkpoint_schedule"]["kind"] != "none"
    ]
    lesions = [
        case
        for case in cases
        if case["role_contract"]["claim_kind"] == "matched_causal_lesion"
    ]
    expected = {
        "case_count": EXPECTED_LOGICAL_CASES,
        "expected_cpu_oracle_evaluations": EXPECTED_CPU_EVALUATIONS,
        "expected_accelerated_invocations": EXPECTED_ACCELERATED_INVOCATIONS,
        "expected_total_required_evaluations": EXPECTED_TOTAL_EVALUATIONS,
    }
    for field, required in expected.items():
        if matrix.get(field) != required:
            raise PhaseFCombinationGateError(f"matrix {field} must equal {required}")
    if len(checkpoint_cases) != EXPECTED_CHECKPOINT_CASES:
        raise PhaseFCombinationGateError("matrix checkpoint count is not exactly 14")
    if len(lesions) != EXPECTED_LESION_RELATIONSHIPS:
        raise PhaseFCombinationGateError(
            "matrix lesion relationship count is not exactly 42"
        )
    _VALIDATED_MATRIX_CACHE[matrix["matrix_sha256"]] = matrix
    return matrix


def _case_map(matrix: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in matrix["cases"]:
        case = dict(raw)
        case_id = _token(case.get("case_id"), field="case_id")
        if case_id in result:
            raise PhaseFCombinationGateError(f"duplicate matrix case: {case_id}")
        result[case_id] = case
    return result


def _comparison_fields(
    matrix: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    contract = matrix.get("comparison_contract")
    if not isinstance(contract, Mapping):
        raise PhaseFCombinationGateError("matrix comparison_contract is missing")
    exact = contract.get("exact_fields")
    admitted = contract.get("admitted_float_fields")
    if (
        type(exact) is not list
        or type(admitted) is not list
        or not exact
        or not admitted
        or any(type(name) is not str or not name for name in (*exact, *admitted))
        or len(set((*exact, *admitted))) != len((*exact, *admitted))
    ):
        raise PhaseFCombinationGateError("matrix comparison fields are not exact lists")
    float_rule = contract.get("float_rule")
    if not isinstance(float_rule, Mapping) or dict(float_rule) != {
        "absolute_tolerance": 2e-7,
        "relative_tolerance": 0.0,
    }:
        raise PhaseFCombinationGateError("matrix float rule is not atol=2e-7, rtol=0")
    if contract.get("categorical_rule") != "exact":
        raise PhaseFCombinationGateError("matrix categorical rule is not exact")
    return tuple(exact), tuple(admitted)


def ordered_phase_f_check_names_v2(matrix: Any) -> tuple[str, ...]:
    """Return the frozen first-difference order bound by the current matrix."""

    checked = _checked_matrix(matrix)
    exact, admitted = _comparison_fields(checked)
    return (
        *_ORDERED_PREFIX_CHECKS,
        *(f"exact:{name}" for name in exact),
        *(f"float:{name}" for name in admitted),
        "evidence_digest_integrity",
    )


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_bytes(left) == canonical_bytes(right)
    except (TypeError, ValueError) as exc:
        raise PhaseFCombinationGateError(
            "comparison values must be canonical JSON"
        ) from exc


def _safe_comparison(expected: Any, observed: Any) -> dict[str, Any]:
    scalar_types = {bool, int, float, str}
    if (
        (expected is None or type(expected) in scalar_types)
        and (observed is None or type(observed) in scalar_types)
        and not (isinstance(expected, float) and not math.isfinite(expected))
        and not (isinstance(observed, float) and not math.isfinite(observed))
    ):
        return {"kind": "canonical_scalar", "expected": expected, "observed": observed}
    return {
        "kind": "digest",
        "expected_sha256": canonical_sha256(expected),
        "observed_sha256": canonical_sha256(observed),
    }


def _check(name: str, expected: Any, observed: Any) -> dict[str, Any]:
    if _canonical_equal(expected, observed):
        return {"check_name": name, "matches": True}
    return {
        "check_name": name,
        "matches": False,
        "comparison": _safe_comparison(expected, observed),
    }


def _checkpoint_projection_for_cross_backend_exactness(value: Any) -> Any:
    """Remove only float hashes covered by the named tolerance comparisons.

    CPU and CUDA may differ by accepted float32 rounding in recurrent state.
    Their raw hashes therefore cannot be compared across backends.  The full
    values remain in the evidence and are compared by the named float fields;
    uninterrupted/resumed exactness still compares the raw checkpoint hashes
    within each backend in the relationship analysis.
    """

    if not isinstance(value, Mapping) or "planes_after" not in value:
        return value
    normalized = deepcopy(dict(value))
    rows = normalized["planes_after"]
    if not isinstance(rows, list) or not rows:
        raise PhaseFCombinationGateError(
            "checkpoint_projection.planes_after must be a nonempty array"
        )
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PhaseFCombinationGateError(
                f"checkpoint_projection.planes_after[{row_index}] must be a mapping"
            )
        for plane in _CROSS_BACKEND_ADMITTED_CHECKPOINT_PLANES:
            manifest = row.get(plane)
            if (
                not isinstance(manifest, Mapping)
                or set(manifest) != {"dtype", "shape", "sha256"}
                or manifest.get("dtype") != "torch.float32"
                or not isinstance(manifest.get("shape"), list)
            ):
                raise PhaseFCombinationGateError(
                    "checkpoint projection admitted float manifest is malformed"
                )
            _sha256(
                manifest.get("sha256"),
                field=f"checkpoint_projection.planes_after.{plane}.sha256",
            )
            manifest["sha256"] = "cross_backend_named_float_comparison"
    return normalized


def _numeric_shape_and_values(
    value: Any, *, field: str
) -> tuple[tuple[Any, ...], list[float]]:
    if type(value) in {int, float}:
        return (), [float(value)]
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise PhaseFCombinationGateError(
                f"{field} must use string keys for structured numeric values"
            )
        children = [
            (
                key,
                _numeric_shape_and_values(
                    value[key],
                    field=f"{field}.{key}",
                ),
            )
            for key in sorted(value)
        ]
        signature = (
            "mapping",
            tuple((key, shape) for key, (shape, _numbers) in children),
        )
        flattened = [
            number
            for _key, (_shape, numbers) in children
            for number in numbers
        ]
        return signature, flattened
    if not isinstance(value, (list, tuple)):
        raise PhaseFCombinationGateError(f"{field} must be a numeric scalar or array")
    children = [
        _numeric_shape_and_values(child, field=f"{field}[{index}]")
        for index, child in enumerate(value)
    ]
    if not children:
        return (0,), []
    child_shape = children[0][0]
    if any(shape != child_shape for shape, _ in children[1:]):
        raise PhaseFCombinationGateError(f"{field} must be a rectangular numeric array")
    flattened = [number for _, numbers in children for number in numbers]
    return (len(children), *child_shape), flattened


def _float_descriptors(
    values: Mapping[str, Any], fields: Sequence[str], *, label: str
) -> tuple[
    dict[str, list[Any]],
    dict[str, bool],
    dict[str, tuple[tuple[Any, ...], list[float]]],
]:
    shapes: dict[str, list[Any]] = {}
    finite: dict[str, bool] = {}
    parsed: dict[str, tuple[tuple[Any, ...], list[float]]] = {}
    for field in fields:
        try:
            shape, numbers = _numeric_shape_and_values(
                values[field], field=f"{label}.{field}"
            )
        except (KeyError, PhaseFCombinationGateError):
            shapes[field] = [-1]
            finite[field] = False
            continue
        shapes[field] = list(shape)
        finite[field] = all(math.isfinite(number) for number in numbers)
        parsed[field] = (shape, numbers)
    return shapes, finite, parsed


def _float_values_match(
    expected: tuple[tuple[Any, ...], list[float]] | None,
    observed: tuple[tuple[Any, ...], list[float]] | None,
) -> bool:
    if expected is None or observed is None or expected[0] != observed[0]:
        return False
    if not all(math.isfinite(number) for number in (*expected[1], *observed[1])):
        return False
    return all(
        abs(expected_value - observed_value) <= 2e-7
        for expected_value, observed_value in zip(expected[1], observed[1], strict=True)
    )


def _float_check(
    name: str,
    expected: tuple[tuple[Any, ...], list[float]] | None,
    observed: tuple[tuple[Any, ...], list[float]] | None,
) -> dict[str, Any]:
    if _float_values_match(expected, observed):
        return {"check_name": name, "matches": True}
    expected_digest_value: Any = (
        {"shape": list(expected[0]), "values": expected[1]}
        if expected is not None and all(math.isfinite(item) for item in expected[1])
        else "invalid_expected_float_field"
    )
    observed_digest_value: Any = (
        {"shape": list(observed[0]), "values": observed[1]}
        if observed is not None and all(math.isfinite(item) for item in observed[1])
        else "invalid_observed_float_field"
    )
    return {
        "check_name": name,
        "matches": False,
        "comparison": _safe_comparison(expected_digest_value, observed_digest_value),
    }


def _subset(value: Any, names: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {name: None for name in names}
    return {name: value.get(name) for name in names}


def compare_phase_f_invocation_v2(
    *,
    matrix: Any,
    bindings: Any,
    expected: Any,
    observed: Any,
    frozen_source_commit: str,
    matrix_receipt_sha256: str,
    entry_receipt_sha256: str,
    fixture_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Compare one already-produced invocation and return evidence-only data."""

    checked_matrix = _checked_matrix(matrix)
    cases = _case_map(checked_matrix)
    actual = _exact_mapping(bindings, fields=_BINDING_FIELDS, label="bindings")
    case_id = _token(actual["case_id"], field="bindings.case_id")
    if case_id not in cases:
        raise PhaseFCombinationGateError(f"unknown Phase F case: {case_id}")
    case = cases[case_id]
    source_commit = _commit(frozen_source_commit, field="frozen_source_commit")
    matrix_receipt = _sha256(matrix_receipt_sha256, field="matrix_receipt_sha256")
    entry_receipt = _sha256(entry_receipt_sha256, field="entry_receipt_sha256")
    expected_fixture_sha256 = _sha256(fixture_sha256, field="fixture_sha256")
    expected_source_sha256 = _sha256(source_sha256, field="source_sha256")
    execution_id = _token(actual["execution_id"], field="bindings.execution_id")
    repeat_index = _nonnegative_index(
        actual["repeat_index"], field="bindings.repeat_index"
    )
    boundary_index = _nonnegative_index(
        actual["boundary_index"], field="bindings.boundary_index"
    )

    exact_fields, float_fields = _comparison_fields(checked_matrix)
    required_output_fields = frozenset((*exact_fields, *float_fields))
    expected_values = _exact_mapping(
        expected, fields=required_output_fields, label="expected"
    )
    if not isinstance(observed, Mapping) or any(
        type(key) is not str for key in observed
    ):
        raise PhaseFCombinationGateError("observed must be a string-keyed mapping")
    observed_values = dict(observed)

    expected_shapes, expected_finite, expected_parsed = _float_descriptors(
        expected_values, float_fields, label="expected"
    )
    if not all(expected_finite.values()):
        raise PhaseFCombinationGateError(
            "CPU expectation contains invalid or nonfinite floats"
        )
    observed_shapes, observed_finite, observed_parsed = _float_descriptors(
        observed_values, float_fields, label="observed"
    )
    try:
        canonical_sha256(observed_values)
        observed_evidence_digestible = True
    except (TypeError, ValueError):
        observed_evidence_digestible = False

    owner = checked_matrix["owner_contract"]
    expected_owner = owner["accelerated_owner"]
    expected_lane = owner["modular_tick_owner"]
    checkpoint_kind = case["checkpoint_schedule"]["kind"]
    fixture_blueprint_digest = case["fixture_blueprint"]["fixture_blueprint_sha256"]
    protocol = checked_matrix["protocol"]
    envelope_fields = (
        "population",
        "grid_width",
        "grid_height",
        "action_width",
        "physical_module_slots",
        "physical_connection_slots",
        "module_state_width",
    )
    lifecycle_fields = (
        "births",
        "deaths",
        "cull",
        "refill",
        "turnover",
        "selection",
        "reproduction",
        "in_tick_topology_growth",
        "multi_cell_owner",
    )
    expected_identity = {
        "case_id": case_id,
        "invocation_kind": actual["invocation_kind"],
        "repeat_index": repeat_index,
        "boundary_index": boundary_index,
        "checkpoint_kind": checkpoint_kind,
    }
    actual_identity = {
        "case_id": actual["case_id"],
        "invocation_kind": actual["invocation_kind"],
        "repeat_index": actual["repeat_index"],
        "boundary_index": actual["boundary_index"],
        "checkpoint_kind": actual["checkpoint_kind"],
    }
    checks = [
        _check("execution_identity", expected_identity, actual_identity),
        _check("source_commit_binding", source_commit, actual["source_commit"]),
        _check("entry_receipt_binding", entry_receipt, actual["entry_receipt_sha256"]),
        _check(
            "matrix_receipt_binding", matrix_receipt, actual["matrix_receipt_sha256"]
        ),
        _check("case_digest_binding", case["case_sha256"], actual["case_sha256"]),
        _check(
            "fixture_blueprint_binding",
            fixture_blueprint_digest,
            actual["fixture_blueprint_sha256"],
        ),
        _check(
            "fixture_digest_binding",
            expected_fixture_sha256,
            actual["fixture_sha256"],
        ),
        _check(
            "source_state_binding",
            expected_source_sha256,
            actual["source_sha256"],
        ),
        _check("owner_identity", expected_owner, actual["owner_id"]),
        _check("profile_identity", case["profile_id"], actual["profile_id"]),
        _check("single_cell_protocol", protocol, actual["protocol"]),
        _check(
            "fixed_envelope_shapes",
            _subset(protocol, envelope_fields),
            _subset(actual["protocol"], envelope_fields),
        ),
        _check(
            "deterministic_action_binding",
            True,
            actual["deterministic_action_binding"],
        ),
        _check(
            "lifecycle_exclusions",
            _subset(protocol, lifecycle_fields),
            _subset(actual["protocol"], lifecycle_fields),
        ),
        _check("hidden_fallback_zero", 0, actual["hidden_fallback_count"]),
        _check("modular_rng_zero", 0, actual["modular_rng_draw_count"]),
        _check(
            "output_field_inventory",
            sorted(required_output_fields),
            sorted(observed_values),
        ),
        _check("float_shape_inventory", expected_shapes, observed_shapes),
        _check(
            "finite_float_outputs",
            {field: True for field in float_fields},
            observed_finite,
        ),
    ]
    for field in exact_fields:
        observed_value = observed_values.get(field, "__missing_observed_field__")
        if field == "checkpoint_projection":
            expected_value = _checkpoint_projection_for_cross_backend_exactness(
                expected_values[field]
            )
            observed_value = _checkpoint_projection_for_cross_backend_exactness(
                observed_value
            )
        else:
            expected_value = expected_values[field]
        checks.append(_check(f"exact:{field}", expected_value, observed_value))
    for field in float_fields:
        checks.append(
            _float_check(
                f"float:{field}",
                expected_parsed.get(field),
                observed_parsed.get(field),
            )
        )
    checks.append(
        _check(
            "evidence_digest_integrity",
            True,
            observed_evidence_digestible,
        )
    )
    expected_names = ordered_phase_f_check_names_v2(checked_matrix)
    if tuple(check["check_name"] for check in checks) != expected_names:
        raise PhaseFCombinationGateError("internal ordered check inventory drifted")

    comparison = build_first_difference_record(
        case_id=case_id,
        execution_id=execution_id,
        source_commit=source_commit,
        matrix_receipt_sha256=matrix_receipt,
        owner_id=expected_owner,
        profile_id=case["profile_id"],
        lane_id=expected_lane,
        repeat_index=repeat_index,
        boundary_index=boundary_index,
        checks=checks,
    )
    receipt: dict[str, Any] = {
        "schema_version": INVOCATION_RECEIPT_SCHEMA_VERSION,
        "authority": EVIDENCE_ONLY_AUTHORITY,
        "execution_authorized": False,
        "case_id": case_id,
        "case_sha256": case["case_sha256"],
        "execution_id": execution_id,
        "source_commit": source_commit,
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "matrix_receipt_sha256": matrix_receipt,
        "entry_receipt_sha256": entry_receipt,
        "fixture_blueprint_sha256": fixture_blueprint_digest,
        "fixture_sha256": expected_fixture_sha256,
        "source_sha256": expected_source_sha256,
        "expected_sha256": canonical_sha256(expected_values),
        "observed_sha256": canonical_sha256(observed_values)
        if observed_evidence_digestible
        else canonical_sha256("noncanonical_observed_evidence"),
        "owner_id": expected_owner,
        "profile_id": case["profile_id"],
        "lane_id": expected_lane,
        "invocation_kind": actual["invocation_kind"],
        "repeat_index": repeat_index,
        "boundary_index": boundary_index,
        "checkpoint_kind": checkpoint_kind,
        "outcome": comparison["outcome"],
        "comparison_receipt": comparison,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return validate_phase_f_invocation_receipt_v2(
        receipt,
        matrix=checked_matrix,
        frozen_source_commit=source_commit,
        matrix_receipt_sha256=matrix_receipt,
        entry_receipt_sha256=entry_receipt,
    )


def _validate_self_digest(receipt: Mapping[str, Any]) -> None:
    observed = _sha256(receipt["receipt_sha256"], field="receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256")
    if canonical_sha256(unsigned) != observed:
        raise PhaseFCombinationGateError("receipt self digest mismatch")


def validate_phase_f_invocation_receipt_v2(
    value: Any,
    *,
    matrix: Any,
    frozen_source_commit: str,
    matrix_receipt_sha256: str,
    entry_receipt_sha256: str,
) -> dict[str, Any]:
    checked_matrix = _checked_matrix(matrix)
    cases = _case_map(checked_matrix)
    receipt = _exact_mapping(
        value, fields=_INVOCATION_RECEIPT_FIELDS, label="invocation receipt"
    )
    if receipt["schema_version"] != INVOCATION_RECEIPT_SCHEMA_VERSION:
        raise PhaseFCombinationGateError("invocation receipt schema mismatch")
    if receipt["authority"] != EVIDENCE_ONLY_AUTHORITY:
        raise PhaseFCombinationGateError(
            "invocation authority must remain evidence_only"
        )
    if receipt["execution_authorized"] is not False:
        raise PhaseFCombinationGateError(
            "invocation receipt cannot authorize execution"
        )
    case_id = _token(receipt["case_id"], field="case_id")
    if case_id not in cases:
        raise PhaseFCombinationGateError("invocation receipt has unknown case")
    case = cases[case_id]
    exact_bindings = {
        "case_sha256": case["case_sha256"],
        "source_commit": _commit(frozen_source_commit, field="frozen_source_commit"),
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "matrix_receipt_sha256": _sha256(
            matrix_receipt_sha256, field="matrix_receipt_sha256"
        ),
        "entry_receipt_sha256": _sha256(
            entry_receipt_sha256, field="entry_receipt_sha256"
        ),
        "fixture_blueprint_sha256": case["fixture_blueprint"][
            "fixture_blueprint_sha256"
        ],
        "owner_id": checked_matrix["owner_contract"]["accelerated_owner"],
        "profile_id": case["profile_id"],
        "lane_id": checked_matrix["owner_contract"]["modular_tick_owner"],
        "checkpoint_kind": case["checkpoint_schedule"]["kind"],
    }
    for field, expected in exact_bindings.items():
        if receipt[field] != expected:
            raise PhaseFCombinationGateError(f"invocation {field} binding mismatch")
    for field in (
        "fixture_sha256",
        "source_sha256",
        "expected_sha256",
        "observed_sha256",
    ):
        _sha256(receipt[field], field=field)
    _token(receipt["execution_id"], field="execution_id")
    repeat_index = _nonnegative_index(receipt["repeat_index"], field="repeat_index")
    _nonnegative_index(receipt["boundary_index"], field="boundary_index")
    kind = receipt["invocation_kind"]
    if kind not in {"uninterrupted", "resumed"}:
        raise PhaseFCombinationGateError("invocation_kind is unsupported")
    limit = case["execution_accounting"][f"{kind}_repeats"]
    if repeat_index >= limit:
        raise PhaseFCombinationGateError(
            "invocation repeat index is outside the case plan"
        )
    comparison = validate_first_difference_record(receipt["comparison_receipt"])
    identity = {
        "case_id": case_id,
        "execution_id": receipt["execution_id"],
        "source_commit": receipt["source_commit"],
        "matrix_receipt_sha256": receipt["matrix_receipt_sha256"],
        "owner_id": receipt["owner_id"],
        "profile_id": receipt["profile_id"],
        "lane_id": receipt["lane_id"],
        "repeat_index": repeat_index,
        "boundary_index": receipt["boundary_index"],
        "ordered_check_names_sha256": canonical_sha256(
            list(ordered_phase_f_check_names_v2(checked_matrix))
        ),
    }
    for field, expected in identity.items():
        if comparison[field] != expected:
            raise PhaseFCombinationGateError(
                f"comparison receipt identity mismatch: {field}"
            )
    if receipt["outcome"] != comparison["outcome"]:
        raise PhaseFCombinationGateError("invocation outcome differs from comparison")
    _validate_self_digest(receipt)
    return receipt


def build_cpu_expectation_evidence_v2(
    *,
    case: Mapping[str, Any],
    matrix_sha256: str,
    fixture_sha256: str,
    source_sha256: str,
    expectation: Any,
    expectation_kind: str,
    status: str,
) -> dict[str, Any]:
    if expectation_kind not in {"uninterrupted", "resumed"}:
        raise PhaseFCombinationGateError("CPU expectation_kind is unsupported")
    if status not in {"passed", "failed"}:
        raise PhaseFCombinationGateError("CPU expectation status cannot skip")
    receipt: dict[str, Any] = {
        "schema_version": CPU_EXPECTATION_SCHEMA_VERSION,
        "authority": EVIDENCE_ONLY_AUTHORITY,
        "execution_authorized": False,
        "case_id": _token(case.get("case_id"), field="case.case_id"),
        "case_sha256": _sha256(case.get("case_sha256"), field="case.case_sha256"),
        "matrix_sha256": _sha256(matrix_sha256, field="matrix_sha256"),
        "fixture_sha256": _sha256(fixture_sha256, field="fixture_sha256"),
        "source_sha256": _sha256(source_sha256, field="source_sha256"),
        "expectation_sha256": canonical_sha256(expectation),
        "expectation_kind": expectation_kind,
        "status": status,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def build_lesion_relationship_evidence_v2(
    *, case: Mapping[str, Any], status: str
) -> dict[str, Any]:
    if status not in {"passed", "failed"}:
        raise PhaseFCombinationGateError("lesion relationship status cannot skip")
    relationship = case.get("relationship")
    role = case.get("role_contract")
    if (
        not isinstance(relationship, Mapping)
        or not isinstance(role, Mapping)
        or role.get("claim_kind") != "matched_causal_lesion"
        or relationship.get("kind") != "matched_intervention"
    ):
        raise PhaseFCombinationGateError("case is not a matched causal lesion")
    receipt: dict[str, Any] = {
        "schema_version": LESION_RELATIONSHIP_SCHEMA_VERSION,
        "authority": EVIDENCE_ONLY_AUTHORITY,
        "execution_authorized": False,
        "case_id": _token(case.get("case_id"), field="case.case_id"),
        "case_sha256": _sha256(case.get("case_sha256"), field="case.case_sha256"),
        "base_case_id": _token(
            relationship.get("base_case_id"), field="relationship.base_case_id"
        ),
        "neutral_case_id": _token(
            relationship.get("neutral_case_id"), field="relationship.neutral_case_id"
        ),
        "status": status,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _validate_cpu_record(
    value: Any, case: Mapping[str, Any], *, matrix_sha256: str
) -> dict[str, Any]:
    receipt = _exact_mapping(
        value, fields=_CPU_EXPECTATION_FIELDS, label="CPU expectation evidence"
    )
    if receipt["schema_version"] != CPU_EXPECTATION_SCHEMA_VERSION:
        raise PhaseFCombinationGateError("CPU expectation schema mismatch")
    if (
        receipt["authority"] != EVIDENCE_ONLY_AUTHORITY
        or receipt["execution_authorized"] is not False
    ):
        raise PhaseFCombinationGateError("CPU expectation overclaims authority")
    if (
        receipt["case_id"] != case["case_id"]
        or receipt["case_sha256"] != case["case_sha256"]
    ):
        raise PhaseFCombinationGateError("CPU expectation case binding mismatch")
    if receipt["matrix_sha256"] != matrix_sha256:
        raise PhaseFCombinationGateError("CPU expectation matrix binding mismatch")
    for field in ("fixture_sha256", "source_sha256", "expectation_sha256"):
        _sha256(receipt[field], field=field)
    if receipt["expectation_kind"] not in {"uninterrupted", "resumed"}:
        raise PhaseFCombinationGateError("CPU expectation kind is unsupported")
    if receipt["status"] not in {"passed", "failed"}:
        raise PhaseFCombinationGateError("CPU expectation cannot be skipped")
    _validate_self_digest(receipt)
    return receipt


def _validate_lesion_record(value: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _exact_mapping(
        value,
        fields=_LESION_RELATIONSHIP_FIELDS,
        label="lesion relationship evidence",
    )
    if receipt["schema_version"] != LESION_RELATIONSHIP_SCHEMA_VERSION:
        raise PhaseFCombinationGateError("lesion relationship schema mismatch")
    if (
        receipt["authority"] != EVIDENCE_ONLY_AUTHORITY
        or receipt["execution_authorized"] is not False
    ):
        raise PhaseFCombinationGateError("lesion relationship overclaims authority")
    relationship = case["relationship"]
    expected = {
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "base_case_id": relationship["base_case_id"],
        "neutral_case_id": relationship["neutral_case_id"],
    }
    for field, required in expected.items():
        if receipt[field] != required:
            raise PhaseFCombinationGateError(f"lesion {field} binding mismatch")
    if receipt["status"] not in {"passed", "failed"}:
        raise PhaseFCombinationGateError("lesion relationship cannot be skipped")
    _validate_self_digest(receipt)
    return receipt


def aggregate_phase_f_terminal_ledger_v2(
    *,
    matrix: Any,
    frozen_source_commit: str,
    matrix_receipt_sha256: str,
    entry_receipt_sha256: str,
    cpu_expectations: Sequence[Any],
    invocation_receipts: Sequence[Any],
    lesion_relationships: Sequence[Any],
    blocked: bool = False,
) -> dict[str, Any]:
    """Validate complete evidence inventories and choose C > B > D > A."""

    checked_matrix = _checked_matrix(matrix)
    cases = _case_map(checked_matrix)
    source_commit = _commit(frozen_source_commit, field="frozen_source_commit")
    matrix_receipt = _sha256(matrix_receipt_sha256, field="matrix_receipt_sha256")
    entry_receipt = _sha256(entry_receipt_sha256, field="entry_receipt_sha256")
    if type(blocked) is not bool:
        raise PhaseFCombinationGateError("blocked must be an exact boolean")
    for name, values in (
        ("cpu_expectations", cpu_expectations),
        ("invocation_receipts", invocation_receipts),
        ("lesion_relationships", lesion_relationships),
    ):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise PhaseFCombinationGateError(f"{name} must be a sequence")

    expected_cpu_slots: set[tuple[str, str]] = set()
    expected_invocation_slots: set[tuple[str, str, int]] = set()
    expected_lesion_cases: set[str] = set()
    for case in cases.values():
        expected_cpu_slots.add((case["case_id"], "uninterrupted"))
        if case["execution_accounting"]["resumed_repeats"]:
            expected_cpu_slots.add((case["case_id"], "resumed"))
        for kind in ("uninterrupted", "resumed"):
            for repeat_index in range(case["execution_accounting"][f"{kind}_repeats"]):
                expected_invocation_slots.add((case["case_id"], kind, repeat_index))
        if case["role_contract"]["claim_kind"] == "matched_causal_lesion":
            expected_lesion_cases.add(case["case_id"])

    cpu_by_slot: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in cpu_expectations:
        if not isinstance(raw, Mapping) or raw.get("case_id") not in cases:
            raise PhaseFCombinationGateError("CPU expectation contains an unknown case")
        case = cases[str(raw["case_id"])]
        receipt = _validate_cpu_record(
            raw, case, matrix_sha256=checked_matrix["matrix_sha256"]
        )
        slot = (receipt["case_id"], receipt["expectation_kind"])
        if slot not in expected_cpu_slots:
            raise PhaseFCombinationGateError("invented CPU expectation slot")
        if slot in cpu_by_slot:
            raise PhaseFCombinationGateError("duplicate CPU expectation slot")
        cpu_by_slot[slot] = receipt

    invocation_by_slot: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw in invocation_receipts:
        receipt = validate_phase_f_invocation_receipt_v2(
            raw,
            matrix=checked_matrix,
            frozen_source_commit=source_commit,
            matrix_receipt_sha256=matrix_receipt,
            entry_receipt_sha256=entry_receipt,
        )
        slot = (
            receipt["case_id"],
            receipt["invocation_kind"],
            receipt["repeat_index"],
        )
        if slot not in expected_invocation_slots:
            raise PhaseFCombinationGateError("invented accelerated invocation slot")
        if slot in invocation_by_slot:
            raise PhaseFCombinationGateError("duplicate accelerated invocation slot")
        cpu_binding = cpu_by_slot.get((receipt["case_id"], receipt["invocation_kind"]))
        if cpu_binding is not None:
            for field, cpu_field in (
                ("fixture_sha256", "fixture_sha256"),
                ("source_sha256", "source_sha256"),
                ("expected_sha256", "expectation_sha256"),
            ):
                if receipt[field] != cpu_binding[cpu_field]:
                    raise PhaseFCombinationGateError(
                        f"accelerated invocation {field} contradicts CPU expectation"
                    )
        invocation_by_slot[slot] = receipt

    lesions_by_case: dict[str, dict[str, Any]] = {}
    for raw in lesion_relationships:
        if not isinstance(raw, Mapping) or raw.get("case_id") not in cases:
            raise PhaseFCombinationGateError("lesion evidence contains an unknown case")
        case_id = str(raw["case_id"])
        if case_id not in expected_lesion_cases:
            raise PhaseFCombinationGateError("invented lesion relationship")
        if case_id in lesions_by_case:
            raise PhaseFCombinationGateError("duplicate lesion relationship")
        lesions_by_case[case_id] = _validate_lesion_record(raw, cases[case_id])

    missing_cpu = expected_cpu_slots - set(cpu_by_slot)
    missing_invocations = expected_invocation_slots - set(invocation_by_slot)
    missing_lesions = expected_lesion_cases - set(lesions_by_case)
    integrity_failed = any(
        receipt["status"] == "failed" for receipt in cpu_by_slot.values()
    ) or any(
        receipt["outcome"] == "first_difference"
        for receipt in invocation_by_slot.values()
    )
    composition_failed = any(
        receipt["status"] == "failed" for receipt in lesions_by_case.values()
    )
    incomplete = bool(missing_cpu or missing_invocations or missing_lesions or blocked)
    if integrity_failed:
        decision = DECISION_C
    elif composition_failed:
        decision = DECISION_B
    elif incomplete:
        decision = DECISION_D
    else:
        decision = DECISION_A

    uninterrupted_count = sum(
        kind == "uninterrupted" for _, kind, _ in invocation_by_slot
    )
    resumed_count = sum(kind == "resumed" for _, kind, _ in invocation_by_slot)
    observed_accelerated = len(invocation_by_slot)
    observed_cpu = len(cpu_by_slot)
    complete = not incomplete
    if decision == DECISION_A and (
        observed_cpu != EXPECTED_CPU_EVALUATIONS
        or uninterrupted_count != EXPECTED_UNINTERRUPTED_INVOCATIONS
        or resumed_count != EXPECTED_RESUMED_INVOCATIONS
        or observed_accelerated != EXPECTED_ACCELERATED_INVOCATIONS
        or observed_cpu + observed_accelerated != EXPECTED_TOTAL_EVALUATIONS
        or len(lesions_by_case) != EXPECTED_LESION_RELATIONSHIPS
    ):
        raise PhaseFCombinationGateError("green decision count invariant failed")

    ledger: dict[str, Any] = {
        "schema_version": TERMINAL_LEDGER_SCHEMA_VERSION,
        "authority": EVIDENCE_ONLY_AUTHORITY,
        "execution_authorized": False,
        "matrix_id": checked_matrix["matrix_id"],
        "matrix_sha256": checked_matrix["matrix_sha256"],
        "source_commit": source_commit,
        "matrix_receipt_sha256": matrix_receipt,
        "entry_receipt_sha256": entry_receipt,
        "decision": decision,
        "complete": complete,
        "blocked": blocked,
        "integrity_failed": integrity_failed,
        "composition_failed": composition_failed,
        "expected_counts": {
            "logical_cases": EXPECTED_LOGICAL_CASES,
            "cpu_evaluations": EXPECTED_CPU_EVALUATIONS,
            "uninterrupted_invocations": EXPECTED_UNINTERRUPTED_INVOCATIONS,
            "resumed_invocations": EXPECTED_RESUMED_INVOCATIONS,
            "accelerated_invocations": EXPECTED_ACCELERATED_INVOCATIONS,
            "total_evaluations": EXPECTED_TOTAL_EVALUATIONS,
            "checkpoint_cases": EXPECTED_CHECKPOINT_CASES,
            "lesion_relationships": EXPECTED_LESION_RELATIONSHIPS,
            "skips": 0,
        },
        "observed_counts": {
            "logical_cases": len(cases),
            "cpu_evaluations": observed_cpu,
            "uninterrupted_invocations": uninterrupted_count,
            "resumed_invocations": resumed_count,
            "accelerated_invocations": observed_accelerated,
            "total_evaluations": observed_cpu + observed_accelerated,
            "checkpoint_cases": sum(
                case["checkpoint_schedule"]["kind"] != "none" for case in cases.values()
            ),
            "lesion_relationships": len(lesions_by_case),
            "skips": 0,
        },
        "missing_counts": {
            "cpu_evaluations": len(missing_cpu),
            "accelerated_invocations": len(missing_invocations),
            "lesion_relationships": len(missing_lesions),
        },
        "claims": {
            "bounded_composition_supported": decision == DECISION_A,
            "campaign_qualified": False,
            "speed_claimed": False,
        },
    }
    ledger["receipt_sha256"] = canonical_sha256(ledger)
    return ledger


__all__ = [
    "CPU_EXPECTATION_SCHEMA_VERSION",
    "DECISION_A",
    "DECISION_B",
    "DECISION_C",
    "DECISION_D",
    "EVIDENCE_ONLY_AUTHORITY",
    "INVOCATION_RECEIPT_SCHEMA_VERSION",
    "LESION_RELATIONSHIP_SCHEMA_VERSION",
    "PhaseFCombinationGateError",
    "TERMINAL_LEDGER_SCHEMA_VERSION",
    "aggregate_phase_f_terminal_ledger_v2",
    "build_cpu_expectation_evidence_v2",
    "build_lesion_relationship_evidence_v2",
    "compare_phase_f_invocation_v2",
    "ordered_phase_f_check_names_v2",
    "validate_phase_f_invocation_receipt_v2",
]
