from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from native_runtime import safe_default
from native_runtime.promotion_certificate import (
    CERTIFICATE_REQUIRED_FIELDS,
    CERTIFICATE_SCHEMA_VERSION,
    PromotionCertificateValidationError,
    canonical_promotion_certificate_bytes,
    checkpoint_qualification_claim,
    generate_promotion_certificate,
    load_promotion_certificate,
    promotion_certificate_sha256,
    require_promotable_certificate,
    supported_envelope_claim,
    validate_promotion_certificate,
    write_promotion_certificate,
)
from native_runtime.promotion_envelope import load_promotion_envelope


COMMIT_SHA = "1" * 40
EXTENSION_SHA = "2" * 64


def _fields() -> dict[str, Any]:
    hashes = iter(character * 64 for character in "3456789abcdef")
    return {
        "supported_envelope": supported_envelope_claim(),
        "candidate_release_commit_sha": COMMIT_SHA,
        "extension_sha256": EXTENSION_SHA,
        "extension_abi_version": "eiel_native_runtime_abi_v1",
        "extension_version": "eiel_native_runtime_flat_v1",
        "reporting_schema_version": "eiel_flat_native_reporting_v1",
        "checkpoint_qualification": checkpoint_qualification_claim(),
        "parity_evidence_version": "flat-native-candidate-1",
        "g2_result_sha256": next(hashes),
        "g4_result_sha256": next(hashes),
        "g8_result_sha256": next(hashes),
        "checkpoint_result_sha256": next(hashes),
        "long_horizon_a_result_sha256": next(hashes),
        "long_horizon_b_result_sha256": next(hashes),
        "long_horizon_c_result_sha256": next(hashes),
        "long_horizon_d_result_sha256": next(hashes),
        "full_regression_sha256": next(hashes),
        "output_fairness_result_sha256": next(hashes),
        "output_fairness_ratio": 1.25,
        "benchmark_result_sha256": next(hashes),
        "current_reference_w1_median_wall_s": 15.0,
        "current_native_w1_median_wall_s": 10.0,
        "w1_native_speedup": 1.5,
        "current_reference_w6_median_wall_s": 11.5,
        "current_native_w6_median_wall_s": 10.0,
        "w6_native_speedup": 1.15,
        "measured_arm_coefficients_of_variation": {
            "current_w1_reference": 0.05,
            "current_w1_native": 0.04,
            "current_w6_reference": 0.03,
            "current_w6_native": 0.02,
        },
        "peak_memory_result": {
            "current_reference_peak_rss_bytes": 100,
            "current_native_peak_rss_bytes": 125,
            "native_to_reference_ratio": 1.25,
            "bounded_exception": None,
        },
        "hidden_fallback_false": True,
        "mixed_ownership_false": True,
        "runtime_jit_false": True,
        "decision_timestamp": "2026-07-15T20:00:00Z",
    }


def _certificate() -> dict[str, Any]:
    return generate_promotion_certificate(_fields())


def test_generator_covers_envelope_fields_and_accepts_inclusive_thresholds() -> None:
    envelope = load_promotion_envelope()
    certificate = _certificate()

    assert tuple(envelope.evidence_requirements.certificate_required_fields) == (
        CERTIFICATE_REQUIRED_FIELDS
    )
    assert certificate["schema_version"] == CERTIFICATE_SCHEMA_VERSION
    assert certificate["verdict"] == {
        "schema_version": "flat_native_promotion_verdict_v1",
        "decision": "promote",
        "eligible": True,
        "gates": {gate: True for gate in certificate["verdict"]["gates"]},
        "reasons": [],
    }
    assert certificate["benchmark_evidence"]["source_commit_sha"] == COMMIT_SHA
    assert certificate["benchmark_evidence"]["extension_sha256"] == EXTENSION_SHA
    assert certificate["benchmark_evidence"]["benchmark_result_sha256"] == (
        certificate["benchmark_result_sha256"]
    )
    assert certificate["benchmark_evidence"]["verdict"] == "passed"


def test_generator_and_canonical_hash_ignore_input_mapping_order() -> None:
    fields = _fields()
    reversed_fields = dict(reversed(tuple(fields.items())))

    first = generate_promotion_certificate(fields)
    second = generate_promotion_certificate(reversed_fields)

    assert first == second
    assert canonical_promotion_certificate_bytes(first) == canonical_promotion_certificate_bytes(
        second
    )
    assert promotion_certificate_sha256(first) == promotion_certificate_sha256(second)
    assert canonical_promotion_certificate_bytes(first).endswith(b"\n")
    assert b"\n" not in canonical_promotion_certificate_bytes(first)[:-1]


def test_writer_and_loader_verify_exact_canonical_file_and_sidecar(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "flat_native_promotion_certificate_v1.json"

    written, sidecar, digest = write_promotion_certificate(path, certificate)

    assert written == path
    assert sidecar == Path(f"{path}.sha256")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert sidecar.read_text(encoding="ascii") == f"{digest}  {path.name}\n"
    assert load_promotion_certificate(path) == certificate
    assert load_promotion_certificate(path, expected_sha256=digest) == certificate


@pytest.mark.parametrize("field", CERTIFICATE_REQUIRED_FIELDS)
def test_generator_rejects_every_missing_required_field(field: str) -> None:
    fields = _fields()
    del fields[field]

    with pytest.raises(PromotionCertificateValidationError, match="missing required fields"):
        generate_promotion_certificate(fields)


def test_generator_rejects_unknown_input_field() -> None:
    fields = _fields()
    fields["unreviewed_override"] = True

    with pytest.raises(PromotionCertificateValidationError, match="unknown fields"):
        generate_promotion_certificate(fields)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda fields: fields.__setitem__("candidate_release_commit_sha", "A" * 40),
            "lowercase hexadecimal",
        ),
        (
            lambda fields: fields.__setitem__("extension_sha256", "2" * 63),
            "64 lowercase hexadecimal",
        ),
        (
            lambda fields: fields.__setitem__("w1_native_speedup", 99.0),
            "does not match the recorded W1 medians",
        ),
        (
            lambda fields: fields["peak_memory_result"].__setitem__(
                "native_to_reference_ratio", 1.0
            ),
            "does not match the recorded byte counts",
        ),
        (
            lambda fields: fields.__setitem__("decision_timestamp", "2026-07-15T22:00:00+02:00"),
            "canonical UTC form",
        ),
        (
            lambda fields: fields["measured_arm_coefficients_of_variation"].__setitem__(
                "current_w1_native", float("nan")
            ),
            "finite number",
        ),
        (
            lambda fields: fields["measured_arm_coefficients_of_variation"].__setitem__(
                "current_w1_native", -0.01
            ),
            "nonnegative",
        ),
    ],
)
def test_generator_rejects_malformed_or_inconsistent_evidence(mutate: Any, message: str) -> None:
    fields = _fields()
    mutate(fields)

    with pytest.raises(PromotionCertificateValidationError, match=message):
        generate_promotion_certificate(fields)


def test_generator_emits_reject_verdict_when_a_preregistered_gate_fails() -> None:
    fields = _fields()
    fields["current_reference_w1_median_wall_s"] = 14.9
    fields["w1_native_speedup"] = 1.49

    certificate = generate_promotion_certificate(fields)

    assert certificate["benchmark_evidence"]["verdict"] == "failed"
    assert certificate["verdict"]["decision"] == "reject"
    assert certificate["verdict"]["eligible"] is False
    assert certificate["verdict"]["gates"]["w1_performance"] is False
    assert certificate["verdict"]["gates"]["benchmark_qualified"] is False
    assert certificate["verdict"]["reasons"] == ["w1_performance", "benchmark_qualified"]
    assert validate_promotion_certificate(certificate) == certificate
    with pytest.raises(PromotionCertificateValidationError, match="verdict is reject"):
        require_promotable_certificate(certificate)


def test_checkpoint_classification_failure_is_a_reject_verdict() -> None:
    fields = _fields()
    fields["checkpoint_qualification"] = checkpoint_qualification_claim(qualified=False)

    certificate = generate_promotion_certificate(fields)

    assert certificate["verdict"]["gates"]["checkpoint_qualified"] is False
    assert certificate["verdict"]["reasons"] == ["checkpoint_qualified"]


def test_validator_rejects_tampered_derived_verdict() -> None:
    certificate = _certificate()
    certificate["verdict"]["eligible"] = False

    with pytest.raises(PromotionCertificateValidationError, match="does not match the verdict"):
        validate_promotion_certificate(certificate)


@pytest.mark.parametrize("field", ["source_commit_sha", "extension_sha256", "verdict"])
def test_validator_rejects_tampered_benchmark_binding(field: str) -> None:
    certificate = _certificate()
    certificate["benchmark_evidence"][field] = "failed" if field == "verdict" else "f" * 64

    with pytest.raises(
        PromotionCertificateValidationError, match="benchmark_evidence does not match"
    ):
        validate_promotion_certificate(certificate)


def test_loader_rejects_noncanonical_json_even_when_payload_is_valid(tmp_path: Path) -> None:
    certificate = _certificate()
    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    digest = promotion_certificate_sha256(certificate)

    with pytest.raises(PromotionCertificateValidationError, match="not exact canonical"):
        load_promotion_certificate(path, expected_sha256=digest)


def test_loader_rejects_duplicate_fields_and_nonfinite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"first","schema_version":"second"}\n')
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":NaN}\n')

    with pytest.raises(PromotionCertificateValidationError, match="duplicate JSON field"):
        load_promotion_certificate(duplicate, expected_sha256="0" * 64)
    with pytest.raises(PromotionCertificateValidationError, match="non-finite JSON number"):
        load_promotion_certificate(nonfinite, expected_sha256="0" * 64)


def test_loader_rejects_missing_or_mismatched_hash_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "certificate.json"
    _, sidecar, digest = write_promotion_certificate(path, _certificate())
    sidecar.unlink()

    with pytest.raises(PromotionCertificateValidationError, match="cannot read.*sidecar"):
        load_promotion_certificate(path)

    sidecar.write_text(f"{'0' * 64}  {path.name}\n", encoding="ascii")
    with pytest.raises(PromotionCertificateValidationError, match="SHA-256 mismatch"):
        load_promotion_certificate(path)
    assert load_promotion_certificate(path, expected_sha256=digest)["verdict"]["eligible"] is True


def test_expected_hash_rejects_canonical_payload_tampering() -> None:
    certificate = _certificate()
    digest = promotion_certificate_sha256(certificate)
    tampered = copy.deepcopy(certificate)
    tampered["decision_timestamp"] = "2026-07-15T20:00:01Z"

    with pytest.raises(PromotionCertificateValidationError, match="SHA-256 mismatch"):
        validate_promotion_certificate(tampered, expected_sha256=digest)


def test_safe_default_binds_promoted_certificate_to_observed_extension_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    certificate = _certificate()
    path = tmp_path / "flat_native_promotion_certificate_v1.json"
    _, _, digest = write_promotion_certificate(path, certificate)
    monkeypatch.setattr(safe_default, "NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED", True)
    monkeypatch.setattr(safe_default, "FLAT_NATIVE_PROMOTION_CERTIFICATE_PATH", str(path))
    monkeypatch.setattr(safe_default, "FLAT_NATIVE_PROMOTION_CERTIFICATE_SHA256", digest)
    monkeypatch.setattr(
        safe_default,
        "PARITY_EVIDENCE_VERSION",
        certificate["parity_evidence_version"],
    )

    accepted = safe_default._qualify_promotion_certificate(
        safe_default.NativeExtensionQualification(
            supported=True,
            sha256=certificate["extension_sha256"],
            abi_version=certificate["extension_abi_version"],
            extension_version=certificate["extension_version"],
        )
    )
    mismatched = safe_default._qualify_promotion_certificate(
        safe_default.NativeExtensionQualification(
            supported=True,
            sha256="f" * 64,
            abi_version=certificate["extension_abi_version"],
            extension_version=certificate["extension_version"],
        )
    )

    assert accepted.accepted is True
    assert accepted.candidate_commit_sha == certificate["candidate_release_commit_sha"]
    assert mismatched.accepted is False
    assert mismatched.reasons == ("promotion_certificate_extension_hash_mismatch",)
