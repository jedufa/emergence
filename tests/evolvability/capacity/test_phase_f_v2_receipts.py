from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from engine.evolution.phase_f_v2.canonical import (
    PhaseFCanonicalJsonError,
    canonical_bytes,
    canonical_sha256,
    pretty_json,
)
from engine.evolution.phase_f_v2.receipts import (
    EXPECTED_ACCELERATED_INVOCATION_COUNT,
    EXPECTED_CHECKPOINT_CASE_COUNT,
    EXPECTED_CPU_EXPECTATION_COUNT,
    EXPECTED_LOGICAL_CASE_COUNT,
    EXPECTED_TOTAL_REQUIRED_EVALUATION_COUNT,
    MATRIX_RECEIPT_SCHEMA_VERSION,
    PhaseFReceiptError,
    create_matrix_registry_receipt,
    validate_matrix_registry_receipt,
    verify_matrix_registry_receipt,
)
from engine.evolution.phase_f_v2.storage import (
    atomic_write_canonical_json,
    atomic_write_pretty_json,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SOURCE_COMMIT = "1" * 40


def _receipt() -> dict[str, object]:
    return create_matrix_registry_receipt(
        source_commit=SOURCE_COMMIT,
        preregistration_sha256=SHA_A,
        phase_e_authorization_sha256=SHA_B,
        matrix_spec_sha256=SHA_C,
        fixture_manifest_sha256=SHA_D,
        cell_registry_sha256=SHA_E,
    )


def test_canonical_json_is_sorted_compact_ascii_finite_and_exact() -> None:
    value = {"z": "caf\N{LATIN SMALL LETTER E WITH ACUTE}", "a": [1, -0.0, True]}

    encoded = canonical_bytes(value)

    assert encoded == b'{"a":[1,-0.0,true],"z":"caf\\u00e9"}'
    assert encoded.isascii()
    assert b" " not in encoded
    assert b"\n" not in encoded
    assert canonical_sha256(value) == hashlib.sha256(encoded).hexdigest()
    assert pretty_json(value).endswith("\n")
    assert json.loads(pretty_json(value)) == value


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_values(nonfinite: float) -> None:
    with pytest.raises(PhaseFCanonicalJsonError, match="finite canonical JSON"):
        canonical_bytes({"value": nonfinite})
    with pytest.raises(PhaseFCanonicalJsonError, match="finite canonical JSON"):
        pretty_json({"value": nonfinite})


def test_registry_receipt_has_exact_nonexecuting_authority_and_self_digest() -> None:
    receipt = _receipt()

    assert receipt == validate_matrix_registry_receipt(receipt)
    verify_matrix_registry_receipt(receipt)
    assert receipt["schema_version"] == MATRIX_RECEIPT_SCHEMA_VERSION
    assert receipt["status"] == "compiled_verified"
    assert receipt["authority"] == "registry_only"
    assert receipt["offline_verification_scope"] == "self_consistency_only"
    assert receipt["no_launch"] is True
    assert receipt["dispatch_started"] is False
    assert receipt["execution_started"] is False
    assert receipt["logical_case_count"] == EXPECTED_LOGICAL_CASE_COUNT
    assert receipt["checkpoint_case_count"] == EXPECTED_CHECKPOINT_CASE_COUNT
    assert receipt["accelerated_invocation_count"] == EXPECTED_ACCELERATED_INVOCATION_COUNT
    assert receipt["cpu_expectation_count"] == EXPECTED_CPU_EXPECTATION_COUNT
    assert receipt["total_required_evaluation_count"] == EXPECTED_TOTAL_REQUIRED_EVALUATION_COUNT
    unsigned = dict(receipt)
    observed = unsigned.pop("receipt_sha256")
    assert observed == canonical_sha256(unsigned)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("authority", "execution", "registry_only"),
        ("no_launch", False, "registry-only authority"),
        ("dispatch_started", True, "registry-only authority"),
        ("execution_started", True, "registry-only authority"),
        ("logical_case_count", 119, "must equal 120"),
        ("accelerated_invocation_count", 267, "must equal 268"),
        ("matrix_spec_sha256", "A" * 64, "lowercase SHA-256"),
    ],
)
def test_registry_receipt_fails_closed_on_authority_shape_or_hash_tampering(
    field: str, value: object, message: str
) -> None:
    receipt = _receipt()
    receipt[field] = value

    with pytest.raises(PhaseFReceiptError, match=message):
        validate_matrix_registry_receipt(receipt)


def test_registry_receipt_fails_closed_on_self_digest_or_field_set_tampering() -> None:
    changed = _receipt()
    changed["source_commit"] = "2" * 40
    with pytest.raises(PhaseFReceiptError, match="self digest mismatch"):
        validate_matrix_registry_receipt(changed)

    extra = {**_receipt(), "command": ["should", "not", "exist"]}
    with pytest.raises(PhaseFReceiptError, match="fields are not exact"):
        validate_matrix_registry_receipt(extra)

    missing = _receipt()
    missing.pop("fixture_manifest_sha256")
    with pytest.raises(PhaseFReceiptError, match="fields are not exact"):
        validate_matrix_registry_receipt(missing)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"source_commit": "short"}, "full lowercase Git commit"),
        ({"matrix_spec_sha256": "f" * 63}, "lowercase SHA-256"),
        ({"logical_case_count": True}, "must equal 120"),
        ({"repeat_count": 3}, "must equal 2"),
    ],
)
def test_registry_receipt_creation_rejects_invalid_frozen_inputs(
    updates: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "source_commit": SOURCE_COMMIT,
        "preregistration_sha256": SHA_A,
        "phase_e_authorization_sha256": SHA_B,
        "matrix_spec_sha256": SHA_C,
        "fixture_manifest_sha256": SHA_D,
        "cell_registry_sha256": SHA_E,
    }
    arguments.update(updates)

    with pytest.raises(PhaseFReceiptError, match=message):
        create_matrix_registry_receipt(**arguments)  # type: ignore[arg-type]


def test_atomic_json_writes_replace_complete_files_without_temp_residue(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "matrix_receipt.json"
    first = _receipt()
    second = copy.deepcopy(first)
    second["receipt_sha256"] = "0" * 64

    assert atomic_write_canonical_json(target, first) == target
    assert target.read_bytes() == canonical_bytes(first) + b"\n"
    assert atomic_write_pretty_json(target, second) == target
    assert target.read_text(encoding="ascii") == pretty_json(second)
    assert not list(target.parent.glob(f".{target.name}.*.tmp"))


def test_failed_serialization_does_not_replace_an_existing_receipt(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    original = canonical_bytes(_receipt()) + b"\n"
    target.write_bytes(original)

    with pytest.raises(PhaseFCanonicalJsonError):
        atomic_write_canonical_json(target, {"not_finite": float("nan")})

    assert target.read_bytes() == original
    assert not list(tmp_path.glob(f".{target.name}.*.tmp"))
