from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from engine.evolution.phase_f_v2.canonical import canonical_sha256
from engine.evolution.phase_f_v2.first_difference import (
    FIRST_DIFFERENCE_SCHEMA_VERSION,
    NO_MISMATCH_SCHEMA_VERSION,
    PhaseFFirstDifferenceError,
    build_first_difference_record,
    select_first_difference,
    validate_first_difference_record,
    verify_first_difference_record,
)


SOURCE_COMMIT = "1" * 40
MATRIX_RECEIPT_SHA256 = "a" * 64


def _identity() -> dict[str, object]:
    return {
        "case_id": "phase-f-case-001",
        "execution_id": "phase-f-case-001-r0",
        "source_commit": SOURCE_COMMIT,
        "matrix_receipt_sha256": MATRIX_RECEIPT_SHA256,
        "owner_id": "phase-f-v2-owner",
        "profile_id": "phase-e-authorized-profile-v1",
        "lane_id": "cpu_reference",
        "repeat_index": 0,
        "boundary_index": 2,
    }


def _digest_checks() -> list[dict[str, object]]:
    return [
        {"check_name": "shape", "matches": True},
        {
            "check_name": "state-digest",
            "matches": False,
            "comparison": {
                "kind": "digest",
                "expected_sha256": "b" * 64,
                "observed_sha256": "c" * 64,
            },
        },
        {
            "check_name": "later-digest",
            "matches": False,
            "comparison": {
                "kind": "digest",
                "expected_sha256": "d" * 64,
                "observed_sha256": "e" * 64,
            },
        },
    ]


def test_selects_first_mismatch_in_frozen_sequence_order() -> None:
    selected = select_first_difference(_digest_checks())

    assert selected is not None
    ordinal, row = selected
    assert ordinal == 1
    assert row["check_name"] == "state-digest"

    receipt = build_first_difference_record(**_identity(), checks=_digest_checks())
    assert receipt["schema_version"] == FIRST_DIFFERENCE_SCHEMA_VERSION
    assert receipt["outcome"] == "first_difference"
    assert receipt["authority"] == "evidence_only"
    assert receipt["execution_authorized"] is False
    assert receipt["check_ordinal"] == 1
    assert receipt["check_name"] == "state-digest"
    assert receipt["ordered_check_names_sha256"] == canonical_sha256(
        ["shape", "state-digest", "later-digest"]
    )
    assert receipt["comparison"] == _digest_checks()[1]["comparison"]
    verify_first_difference_record(receipt)


def test_scalar_difference_preserves_exact_canonical_values() -> None:
    checks = [
        {
            "check_name": "scalar-output",
            "matches": False,
            "comparison": {
                "kind": "canonical_scalar",
                "expected": -0.0,
                "observed": 0.0,
            },
        }
    ]

    receipt = build_first_difference_record(**_identity(), checks=checks)

    assert receipt["comparison"] == checks[0]["comparison"]
    assert receipt == validate_first_difference_record(receipt)


def test_no_mismatch_uses_separate_schema_without_fabricated_values() -> None:
    checks = [
        {"check_name": "shape", "matches": True},
        {"check_name": "state-digest", "matches": True},
    ]

    assert select_first_difference(checks) is None
    receipt = build_first_difference_record(**_identity(), checks=checks)

    assert receipt["schema_version"] == NO_MISMATCH_SCHEMA_VERSION
    assert receipt["outcome"] == "no_mismatch"
    assert receipt["completed_check_count"] == 2
    assert not {
        "check_name",
        "check_ordinal",
        "comparison",
        "expected",
        "observed",
        "expected_sha256",
        "observed_sha256",
    } & set(receipt)
    verify_first_difference_record(receipt)


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        ([], "nonempty ordered sequence"),
        (
            [
                {"check_name": "same", "matches": True},
                {"check_name": "same", "matches": True},
            ],
            "duplicate check_name",
        ),
        (
            [{"check_name": "missing-comparison", "matches": False}],
            "fields are not exact",
        ),
        (
            [
                {
                    "check_name": "fabricated-match",
                    "matches": True,
                    "comparison": {
                        "kind": "digest",
                        "expected_sha256": "b" * 64,
                        "observed_sha256": "c" * 64,
                    },
                }
            ],
            "fields are not exact",
        ),
        (
            [
                {
                    "check_name": "equal-digest",
                    "matches": False,
                    "comparison": {
                        "kind": "digest",
                        "expected_sha256": "b" * 64,
                        "observed_sha256": "b" * 64,
                    },
                }
            ],
            "requires different",
        ),
        (
            [
                {
                    "check_name": "equal-scalar",
                    "matches": False,
                    "comparison": {
                        "kind": "canonical_scalar",
                        "expected": "same",
                        "observed": "same",
                    },
                }
            ],
            "requires different",
        ),
    ],
)
def test_invalid_ordered_checks_fail_closed(
    checks: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(PhaseFFirstDifferenceError, match=message):
        build_first_difference_record(**_identity(), checks=checks)


@pytest.mark.parametrize(
    "scalar",
    [float("nan"), float("inf"), [], {}, object()],
)
def test_noncanonical_scalars_fail_closed(scalar: object) -> None:
    checks = [
        {
            "check_name": "scalar",
            "matches": False,
            "comparison": {
                "kind": "canonical_scalar",
                "expected": 0,
                "observed": scalar,
            },
        }
    ]

    with pytest.raises(PhaseFFirstDifferenceError):
        build_first_difference_record(**_identity(), checks=checks)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("authority", "execution", "evidence_only"),
        ("execution_authorized", True, "remain false"),
        ("source_commit", "2" * 39, "full lowercase Git commit"),
        ("matrix_receipt_sha256", "A" * 64, "lowercase SHA-256"),
        ("owner_id", "contains a space", "stable token"),
        ("repeat_index", True, "nonnegative integer"),
        ("boundary_index", -1, "nonnegative integer"),
    ],
)
def test_identity_and_authority_tampering_fails_closed(
    field: str, value: object, message: str
) -> None:
    receipt = build_first_difference_record(
        **_identity(), checks=_digest_checks()
    )
    receipt[field] = value

    with pytest.raises(PhaseFFirstDifferenceError, match=message):
        validate_first_difference_record(receipt)


def test_record_self_digest_and_exact_field_set_fail_closed() -> None:
    receipt = build_first_difference_record(**_identity(), checks=_digest_checks())
    changed = copy.deepcopy(receipt)
    changed["check_name"] = "changed-check"
    with pytest.raises(PhaseFFirstDifferenceError, match="self digest mismatch"):
        validate_first_difference_record(changed)

    extra = {**receipt, "command": ["forbidden"]}
    with pytest.raises(PhaseFFirstDifferenceError, match="fields are not exact"):
        validate_first_difference_record(extra)

    missing = dict(receipt)
    missing.pop("profile_id")
    with pytest.raises(PhaseFFirstDifferenceError, match="fields are not exact"):
        validate_first_difference_record(missing)


def test_record_build_is_byte_stable() -> None:
    first = build_first_difference_record(**_identity(), checks=_digest_checks())
    second = build_first_difference_record(**_identity(), checks=_digest_checks())

    assert first == second
    assert canonical_sha256(first) == canonical_sha256(second)


def test_module_has_no_execution_accelerator_or_command_authority() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "engine/evolution/phase_f_v2/first_difference.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"__import__", "compile", "eval", "exec"}
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {
                    "Popen",
                    "execv",
                    "popen",
                    "run",
                    "spawnv",
                    "system",
                }

    forbidden = ("cuda", "native_runtime", "orchestration", "providers", "subprocess", "torch")
    assert not [name for name in imports if any(item in name for item in forbidden)]
