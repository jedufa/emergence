"""Deterministic generation and validation for Flat native promotion certificates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .promotion_envelope import FlatNativePromotionEnvelope, load_promotion_envelope


CERTIFICATE_SCHEMA_VERSION = "flat_native_promotion_certificate_v1"
BENCHMARK_EVIDENCE_SCHEMA_VERSION = "flat_native_promotion_benchmark_evidence_v1"
VERDICT_SCHEMA_VERSION = "flat_native_promotion_verdict_v1"
CHECKPOINT_QUALIFICATION_SCHEMA_VERSION = "flat_native_checkpoint_qualification_v1"
HASH_ALGORITHM = "sha256"

CERTIFICATE_REQUIRED_FIELDS = (
    "supported_envelope",
    "candidate_release_commit_sha",
    "extension_sha256",
    "extension_abi_version",
    "extension_version",
    "reporting_schema_version",
    "checkpoint_qualification",
    "parity_evidence_version",
    "g2_result_sha256",
    "g4_result_sha256",
    "g8_result_sha256",
    "checkpoint_result_sha256",
    "long_horizon_a_result_sha256",
    "long_horizon_b_result_sha256",
    "long_horizon_c_result_sha256",
    "long_horizon_d_result_sha256",
    "full_regression_sha256",
    "output_fairness_result_sha256",
    "output_fairness_ratio",
    "benchmark_result_sha256",
    "current_reference_w1_median_wall_s",
    "current_native_w1_median_wall_s",
    "w1_native_speedup",
    "current_reference_w6_median_wall_s",
    "current_native_w6_median_wall_s",
    "w6_native_speedup",
    "measured_arm_coefficients_of_variation",
    "peak_memory_result",
    "hidden_fallback_false",
    "mixed_ownership_false",
    "runtime_jit_false",
    "decision_timestamp",
)

REQUIRED_MEASURED_ARMS = (
    "current_w1_reference",
    "current_w1_native",
    "current_w6_reference",
    "current_w6_native",
)

VERDICT_GATES = (
    "supported_envelope",
    "committed_source_bound",
    "extension_identity",
    "reporting_qualified",
    "checkpoint_qualified",
    "parity_evidence_bound",
    "result_evidence_bound",
    "w1_performance",
    "w6_performance",
    "stability",
    "output_fairness",
    "peak_memory",
    "hidden_fallback_absent",
    "mixed_ownership_absent",
    "runtime_jit_absent",
    "benchmark_qualified",
)

_RESULT_HASH_FIELDS = (
    "g2_result_sha256",
    "g4_result_sha256",
    "g8_result_sha256",
    "checkpoint_result_sha256",
    "long_horizon_a_result_sha256",
    "long_horizon_b_result_sha256",
    "long_horizon_c_result_sha256",
    "long_horizon_d_result_sha256",
    "full_regression_sha256",
    "output_fairness_result_sha256",
    "benchmark_result_sha256",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


class PromotionCertificateValidationError(ValueError):
    """Raised when a promotion certificate is malformed or internally inconsistent."""


def _contract(envelope: FlatNativePromotionEnvelope | None) -> FlatNativePromotionEnvelope:
    return envelope if envelope is not None else load_promotion_envelope()


def _object(value: object, path: str, fields: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromotionCertificateValidationError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise PromotionCertificateValidationError(f"{path} keys must be strings")
    expected = set(fields)
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise PromotionCertificateValidationError(f"{path} missing required fields: {missing}")
    if unknown:
        raise PromotionCertificateValidationError(f"{path} has unknown fields: {unknown}")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise PromotionCertificateValidationError(f"{path} must be a non-empty string")
    return value


def _literal_string(value: object, path: str, expected: str) -> str:
    observed = _string(value, path)
    if observed != expected:
        raise PromotionCertificateValidationError(
            f"{path} must be {expected!r}, observed {observed!r}"
        )
    return observed


def _boolean(value: object, path: str) -> bool:
    if type(value) is not bool:
        raise PromotionCertificateValidationError(f"{path} must be a boolean")
    return value


def _finite_number(value: object, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromotionCertificateValidationError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise PromotionCertificateValidationError(f"{path} must be a finite number")
    if positive and number <= 0.0:
        raise PromotionCertificateValidationError(f"{path} must be positive")
    return number


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PromotionCertificateValidationError(f"{path} must be a positive integer")
    return value


def _sha256(value: object, path: str) -> str:
    digest = _string(value, path)
    if _SHA256_RE.fullmatch(digest) is None:
        raise PromotionCertificateValidationError(
            f"{path} must be exactly 64 lowercase hexadecimal characters"
        )
    return digest


def _commit_sha(value: object, path: str) -> str:
    commit = _string(value, path)
    if _COMMIT_RE.fullmatch(commit) is None:
        raise PromotionCertificateValidationError(
            f"{path} must be exactly 40 lowercase hexadecimal characters"
        )
    return commit


def _decision_timestamp(value: object, path: str) -> str:
    timestamp = _string(value, path)
    if _TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise PromotionCertificateValidationError(
            f"{path} must use canonical UTC form YYYY-MM-DDTHH:MM:SSZ"
        )
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise PromotionCertificateValidationError(f"{path} is not a valid UTC timestamp") from exc
    return timestamp


def supported_envelope_claim(
    envelope: FlatNativePromotionEnvelope | None = None,
) -> dict[str, Any]:
    """Return the exact promotion-envelope identity bound into a certificate."""

    contract = _contract(envelope)
    profile = contract.supported_profile
    return {
        "schema_version": contract.schema_version,
        "contract_id": contract.contract_id,
        "contract_version": contract.contract_version,
        "world": profile.world,
        "policy_family": profile.policy.family,
        "policy_selection": profile.policy.selection,
        "simulation_device": profile.device.simulation,
        "tensor_bank_device": profile.device.tensor_bank,
        "reporting_schema_version": profile.reporting.schema_version,
        "output_profile": profile.output.profile_id,
    }


def checkpoint_qualification_claim(
    *,
    qualified: bool = True,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> dict[str, Any]:
    """Return an explicit qualification row for every checkpoint mode in the envelope."""

    contract = _contract(envelope)
    modes = [
        {
            "name": mode.name,
            "supported": mode.supported,
            "qualification": mode.qualification,
            "qualified": bool(qualified),
        }
        for mode in contract.supported_profile.checkpoint.modes
    ]
    return {
        "schema_version": CHECKPOINT_QUALIFICATION_SCHEMA_VERSION,
        "all_modes_qualified": bool(qualified),
        "modes": modes,
    }


def _validate_supported_envelope(
    value: object,
    envelope: FlatNativePromotionEnvelope,
) -> bool:
    path = "$.supported_envelope"
    expected = supported_envelope_claim(envelope)
    row = _object(value, path, tuple(expected))
    return dict(row) == expected


def _validate_checkpoint_qualification(
    value: object,
    envelope: FlatNativePromotionEnvelope,
) -> bool:
    path = "$.checkpoint_qualification"
    row = _object(value, path, ("schema_version", "all_modes_qualified", "modes"))
    _literal_string(
        row["schema_version"],
        f"{path}.schema_version",
        CHECKPOINT_QUALIFICATION_SCHEMA_VERSION,
    )
    all_modes_qualified = _boolean(
        row["all_modes_qualified"], f"{path}.all_modes_qualified"
    )
    raw_modes = row["modes"]
    if not isinstance(raw_modes, list):
        raise PromotionCertificateValidationError(f"{path}.modes must be an array")
    expected_modes = envelope.supported_profile.checkpoint.modes
    if len(raw_modes) != len(expected_modes):
        raise PromotionCertificateValidationError(
            f"{path}.modes must contain exactly {len(expected_modes)} rows"
        )
    qualified_rows: list[bool] = []
    for index, (raw_mode, expected_mode) in enumerate(zip(raw_modes, expected_modes, strict=True)):
        mode_path = f"{path}.modes[{index}]"
        mode = _object(
            raw_mode,
            mode_path,
            ("name", "supported", "qualification", "qualified"),
        )
        _literal_string(mode["name"], f"{mode_path}.name", expected_mode.name)
        if _boolean(mode["supported"], f"{mode_path}.supported") is not expected_mode.supported:
            raise PromotionCertificateValidationError(
                f"{mode_path}.supported contradicts the promotion envelope"
            )
        _literal_string(
            mode["qualification"],
            f"{mode_path}.qualification",
            expected_mode.qualification,
        )
        qualified_rows.append(_boolean(mode["qualified"], f"{mode_path}.qualified"))
    if all_modes_qualified != all(qualified_rows):
        raise PromotionCertificateValidationError(
            f"{path}.all_modes_qualified contradicts its mode rows"
        )
    return all_modes_qualified


def _validate_coefficients(value: object) -> dict[str, float]:
    path = "$.measured_arm_coefficients_of_variation"
    row = _object(value, path, REQUIRED_MEASURED_ARMS)
    coefficients = {
        arm: _finite_number(row[arm], f"{path}.{arm}")
        for arm in REQUIRED_MEASURED_ARMS
    }
    if any(coefficient < 0.0 for coefficient in coefficients.values()):
        raise PromotionCertificateValidationError(f"{path} values must be nonnegative")
    return coefficients


def _validate_peak_memory(value: object) -> dict[str, Any]:
    path = "$.peak_memory_result"
    row = _object(
        value,
        path,
        (
            "current_reference_peak_rss_bytes",
            "current_native_peak_rss_bytes",
            "native_to_reference_ratio",
            "bounded_exception",
        ),
    )
    reference = _positive_integer(
        row["current_reference_peak_rss_bytes"],
        f"{path}.current_reference_peak_rss_bytes",
    )
    native = _positive_integer(
        row["current_native_peak_rss_bytes"],
        f"{path}.current_native_peak_rss_bytes",
    )
    ratio = _finite_number(
        row["native_to_reference_ratio"],
        f"{path}.native_to_reference_ratio",
        positive=True,
    )
    if not math.isclose(ratio, native / reference, rel_tol=1e-9, abs_tol=1e-12):
        raise PromotionCertificateValidationError(
            f"{path}.native_to_reference_ratio does not match the recorded byte counts"
        )
    if row["bounded_exception"] is not None:
        raise PromotionCertificateValidationError(
            f"{path}.bounded_exception must be null for certificate v1"
        )
    return {
        "current_reference_peak_rss_bytes": reference,
        "current_native_peak_rss_bytes": native,
        "native_to_reference_ratio": ratio,
        "bounded_exception": None,
    }


def _threshold_passes(value: float, threshold: float, *, minimum: bool, inclusive: bool) -> bool:
    if minimum:
        return value >= threshold if inclusive else value > threshold
    return value <= threshold if inclusive else value < threshold


def _derive_gate_results(
    fields: Mapping[str, Any],
    envelope: FlatNativePromotionEnvelope,
    *,
    supported_envelope_ok: bool,
    checkpoint_ok: bool,
    coefficients: Mapping[str, float],
    peak_memory: Mapping[str, Any],
) -> dict[str, bool]:
    thresholds = envelope.promotion_thresholds
    w1_ok = _threshold_passes(
        float(fields["w1_native_speedup"]),
        thresholds.primary_performance.minimum_speedup,
        minimum=True,
        inclusive=thresholds.primary_performance.inclusive,
    )
    w6_ok = _threshold_passes(
        float(fields["w6_native_speedup"]),
        thresholds.secondary_performance.minimum_speedup,
        minimum=True,
        inclusive=thresholds.secondary_performance.inclusive,
    )
    stability_ok = all(
        _threshold_passes(
            coefficient,
            thresholds.stability.maximum,
            minimum=False,
            inclusive=thresholds.stability.inclusive,
        )
        for coefficient in coefficients.values()
    )
    fairness_ratio = float(fields["output_fairness_ratio"])
    fairness = thresholds.output_fairness
    if fairness.inclusive:
        fairness_ok = fairness.minimum <= fairness_ratio <= fairness.maximum
    else:
        fairness_ok = fairness.minimum < fairness_ratio < fairness.maximum
    memory_ok = _threshold_passes(
        float(peak_memory["native_to_reference_ratio"]),
        thresholds.memory.maximum,
        minimum=False,
        inclusive=thresholds.memory.inclusive,
    )
    hidden_fallback_ok = bool(fields["hidden_fallback_false"])
    mixed_ownership_ok = bool(fields["mixed_ownership_false"])
    runtime_jit_ok = bool(fields["runtime_jit_false"])
    benchmark_ok = all(
        (
            w1_ok,
            w6_ok,
            stability_ok,
            fairness_ok,
            memory_ok,
            hidden_fallback_ok,
            mixed_ownership_ok,
            runtime_jit_ok,
        )
    )
    return {
        "supported_envelope": supported_envelope_ok,
        "committed_source_bound": True,
        "extension_identity": (
            fields["extension_abi_version"] == envelope.binary_contract.extension_abi_version
            and fields["extension_version"] == envelope.binary_contract.extension_version
        ),
        "reporting_qualified": (
            fields["reporting_schema_version"]
            == envelope.supported_profile.reporting.schema_version
        ),
        "checkpoint_qualified": checkpoint_ok,
        "parity_evidence_bound": bool(fields["parity_evidence_version"]),
        "result_evidence_bound": True,
        "w1_performance": w1_ok,
        "w6_performance": w6_ok,
        "stability": stability_ok,
        "output_fairness": fairness_ok,
        "peak_memory": memory_ok,
        "hidden_fallback_absent": hidden_fallback_ok,
        "mixed_ownership_absent": mixed_ownership_ok,
        "runtime_jit_absent": runtime_jit_ok,
        "benchmark_qualified": benchmark_ok,
    }


def _benchmark_evidence(fields: Mapping[str, Any], verdict: str) -> dict[str, Any]:
    return {
        "schema_version": BENCHMARK_EVIDENCE_SCHEMA_VERSION,
        "source_commit_sha": fields["candidate_release_commit_sha"],
        "extension_sha256": fields["extension_sha256"],
        "benchmark_result_sha256": fields["benchmark_result_sha256"],
        "current_reference_w1_median_wall_s": fields[
            "current_reference_w1_median_wall_s"
        ],
        "current_native_w1_median_wall_s": fields["current_native_w1_median_wall_s"],
        "w1_native_speedup": fields["w1_native_speedup"],
        "current_reference_w6_median_wall_s": fields[
            "current_reference_w6_median_wall_s"
        ],
        "current_native_w6_median_wall_s": fields["current_native_w6_median_wall_s"],
        "w6_native_speedup": fields["w6_native_speedup"],
        "measured_arm_coefficients_of_variation": fields[
            "measured_arm_coefficients_of_variation"
        ],
        "output_fairness_ratio": fields["output_fairness_ratio"],
        "peak_memory_result": fields["peak_memory_result"],
        "hidden_fallback_false": fields["hidden_fallback_false"],
        "mixed_ownership_false": fields["mixed_ownership_false"],
        "runtime_jit_false": fields["runtime_jit_false"],
        "verdict": verdict,
    }


def _verdict(gates: Mapping[str, bool]) -> dict[str, Any]:
    reasons = [gate for gate in VERDICT_GATES if not gates[gate]]
    eligible = not reasons
    return {
        "schema_version": VERDICT_SCHEMA_VERSION,
        "decision": "promote" if eligible else "reject",
        "eligible": eligible,
        "gates": dict(gates),
        "reasons": reasons,
    }


def _validate_required_fields(
    fields: Mapping[str, Any],
    envelope: FlatNativePromotionEnvelope,
) -> tuple[dict[str, float], dict[str, Any], bool, bool]:
    envelope_fields = tuple(envelope.evidence_requirements.certificate_required_fields)
    if envelope_fields != CERTIFICATE_REQUIRED_FIELDS:
        raise PromotionCertificateValidationError(
            "promotion envelope certificate field contract drifted from certificate v1"
        )
    if envelope.evidence_requirements.certificate_schema_version != CERTIFICATE_SCHEMA_VERSION:
        raise PromotionCertificateValidationError(
            "promotion envelope certificate schema version does not match certificate v1"
        )
    if envelope.evidence_requirements.hash_algorithm != HASH_ALGORITHM:
        raise PromotionCertificateValidationError(
            "promotion envelope certificate hash algorithm must be sha256"
        )
    supported_ok = _validate_supported_envelope(fields["supported_envelope"], envelope)
    _commit_sha(fields["candidate_release_commit_sha"], "$.candidate_release_commit_sha")
    _sha256(fields["extension_sha256"], "$.extension_sha256")
    _string(fields["extension_abi_version"], "$.extension_abi_version")
    _string(fields["extension_version"], "$.extension_version")
    _string(fields["reporting_schema_version"], "$.reporting_schema_version")
    checkpoint_ok = _validate_checkpoint_qualification(
        fields["checkpoint_qualification"], envelope
    )
    _string(fields["parity_evidence_version"], "$.parity_evidence_version")
    for field in _RESULT_HASH_FIELDS:
        _sha256(fields[field], f"$.{field}")
    output_fairness_ratio = _finite_number(
        fields["output_fairness_ratio"], "$.output_fairness_ratio", positive=True
    )
    measurements = {
        name: _finite_number(fields[name], f"$.{name}", positive=True)
        for name in (
            "current_reference_w1_median_wall_s",
            "current_native_w1_median_wall_s",
            "w1_native_speedup",
            "current_reference_w6_median_wall_s",
            "current_native_w6_median_wall_s",
            "w6_native_speedup",
        )
    }
    expected_w1 = (
        measurements["current_reference_w1_median_wall_s"]
        / measurements["current_native_w1_median_wall_s"]
    )
    expected_w6 = (
        measurements["current_reference_w6_median_wall_s"]
        / measurements["current_native_w6_median_wall_s"]
    )
    if not math.isclose(
        measurements["w1_native_speedup"], expected_w1, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise PromotionCertificateValidationError(
            "$.w1_native_speedup does not match the recorded W1 medians"
        )
    if not math.isclose(
        measurements["w6_native_speedup"], expected_w6, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise PromotionCertificateValidationError(
            "$.w6_native_speedup does not match the recorded W6 medians"
        )
    coefficients = _validate_coefficients(fields["measured_arm_coefficients_of_variation"])
    peak_memory = _validate_peak_memory(fields["peak_memory_result"])
    _boolean(fields["hidden_fallback_false"], "$.hidden_fallback_false")
    _boolean(fields["mixed_ownership_false"], "$.mixed_ownership_false")
    _boolean(fields["runtime_jit_false"], "$.runtime_jit_false")
    _decision_timestamp(fields["decision_timestamp"], "$.decision_timestamp")
    # Keep these values evaluated before verdict derivation so bool/NaN coercion cannot pass a gate.
    _ = output_fairness_ratio
    return coefficients, peak_memory, supported_ok, checkpoint_ok


def generate_promotion_certificate(
    fields: Mapping[str, Any],
    *,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> dict[str, Any]:
    """Generate a deterministic certificate and derive its fail-closed verdict."""

    contract = _contract(envelope)
    required = _object(fields, "$", CERTIFICATE_REQUIRED_FIELDS)
    coefficients, peak_memory, supported_ok, checkpoint_ok = _validate_required_fields(
        required, contract
    )
    gates = _derive_gate_results(
        required,
        contract,
        supported_envelope_ok=supported_ok,
        checkpoint_ok=checkpoint_ok,
        coefficients=coefficients,
        peak_memory=peak_memory,
    )
    verdict = _verdict(gates)
    payload = {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        **dict(required),
        "benchmark_evidence": _benchmark_evidence(
            required, "passed" if gates["benchmark_qualified"] else "failed"
        ),
        "verdict": verdict,
    }
    return validate_promotion_certificate(payload, envelope=contract)


def validate_promotion_certificate(
    payload: object,
    *,
    expected_sha256: str | None = None,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> dict[str, Any]:
    """Validate certificate schema, evidence bindings, derived verdict, and optional hash."""

    contract = _contract(envelope)
    fields = (
        "schema_version",
        *CERTIFICATE_REQUIRED_FIELDS,
        "benchmark_evidence",
        "verdict",
    )
    row = _object(payload, "$", fields)
    _literal_string(row["schema_version"], "$.schema_version", CERTIFICATE_SCHEMA_VERSION)
    required = {field: row[field] for field in CERTIFICATE_REQUIRED_FIELDS}
    coefficients, peak_memory, supported_ok, checkpoint_ok = _validate_required_fields(
        required, contract
    )
    gates = _derive_gate_results(
        required,
        contract,
        supported_envelope_ok=supported_ok,
        checkpoint_ok=checkpoint_ok,
        coefficients=coefficients,
        peak_memory=peak_memory,
    )
    expected_verdict = _verdict(gates)
    observed_verdict = _object(
        row["verdict"],
        "$.verdict",
        ("schema_version", "decision", "eligible", "gates", "reasons"),
    )
    observed_gates = _object(observed_verdict["gates"], "$.verdict.gates", VERDICT_GATES)
    for gate in VERDICT_GATES:
        _boolean(observed_gates[gate], f"$.verdict.gates.{gate}")
    reasons = observed_verdict["reasons"]
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise PromotionCertificateValidationError("$.verdict.reasons must be an array of strings")
    _literal_string(
        observed_verdict["schema_version"],
        "$.verdict.schema_version",
        VERDICT_SCHEMA_VERSION,
    )
    _string(observed_verdict["decision"], "$.verdict.decision")
    _boolean(observed_verdict["eligible"], "$.verdict.eligible")
    if dict(observed_verdict) != expected_verdict:
        raise PromotionCertificateValidationError(
            "$.verdict does not match the verdict derived from certificate evidence"
        )
    expected_benchmark = _benchmark_evidence(
        required, "passed" if gates["benchmark_qualified"] else "failed"
    )
    observed_benchmark = _object(
        row["benchmark_evidence"],
        "$.benchmark_evidence",
        tuple(expected_benchmark),
    )
    if dict(observed_benchmark) != expected_benchmark:
        raise PromotionCertificateValidationError(
            "$.benchmark_evidence does not match top-level source, extension, or measurements"
        )
    normalized = json.loads(_canonical_json_bytes(dict(row)).decode("utf-8"))
    if expected_sha256 is not None:
        expected = _sha256(expected_sha256, "expected_sha256")
        observed = hashlib.sha256(_canonical_json_bytes(normalized)).hexdigest()
        if observed != expected:
            raise PromotionCertificateValidationError(
                f"certificate SHA-256 mismatch: expected {expected}, observed {observed}"
            )
    return normalized


def require_promotable_certificate(
    payload: object,
    *,
    expected_sha256: str | None = None,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> dict[str, Any]:
    """Validate a certificate and reject a structurally valid non-promotion verdict."""

    certificate = validate_promotion_certificate(
        payload, expected_sha256=expected_sha256, envelope=envelope
    )
    verdict = certificate["verdict"]
    if not verdict["eligible"]:
        reasons = ",".join(verdict["reasons"])
        raise PromotionCertificateValidationError(
            f"promotion certificate verdict is reject: {reasons}"
        )
    return certificate


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PromotionCertificateValidationError(
            f"certificate is not canonical-JSON serializable: {exc}"
        ) from exc
    return f"{encoded}\n".encode("utf-8")


def canonical_promotion_certificate_bytes(
    payload: object,
    *,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> bytes:
    """Return the exact newline-terminated canonical bytes used for certificate hashing."""

    validated = validate_promotion_certificate(payload, envelope=envelope)
    return _canonical_json_bytes(validated)


def promotion_certificate_sha256(
    payload: object,
    *,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> str:
    """Hash the exact canonical certificate bytes written to disk."""

    return hashlib.sha256(
        canonical_promotion_certificate_bytes(payload, envelope=envelope)
    ).hexdigest()


def write_promotion_certificate(
    path: str | Path,
    payload: object,
    *,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> tuple[Path, Path, str]:
    """Write canonical certificate bytes and a sha256sum-compatible sidecar."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    canonical = canonical_promotion_certificate_bytes(payload, envelope=envelope)
    digest = hashlib.sha256(canonical).hexdigest()
    sidecar = Path(f"{target}.sha256")
    target.write_bytes(canonical)
    sidecar.write_text(f"{digest}  {target.name}\n", encoding="ascii")
    return target, sidecar, digest


def _reject_json_constant(value: str) -> None:
    raise PromotionCertificateValidationError(
        f"non-finite JSON number {value!r} is not permitted in a promotion certificate"
    )


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise PromotionCertificateValidationError(f"duplicate JSON field {key!r}")
        out[key] = value
    return out


def _read_sidecar(sidecar: Path, certificate_name: str) -> str:
    try:
        contents = sidecar.read_text(encoding="ascii")
    except OSError as exc:
        raise PromotionCertificateValidationError(
            f"cannot read promotion certificate SHA-256 sidecar {sidecar}: {exc}"
        ) from exc
    parts = contents.strip().split()
    if len(parts) != 2 or parts[1].lstrip("*") != certificate_name:
        raise PromotionCertificateValidationError(
            "promotion certificate SHA-256 sidecar must contain one digest and certificate filename"
        )
    return _sha256(parts[0], "certificate sidecar SHA-256")


def load_promotion_certificate(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    envelope: FlatNativePromotionEnvelope | None = None,
) -> dict[str, Any]:
    """Load an exact canonical certificate and verify its required SHA-256."""

    target = Path(path)
    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise PromotionCertificateValidationError(
            f"cannot read promotion certificate {target}: {exc}"
        ) from exc
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_json_constant,
        )
    except PromotionCertificateValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionCertificateValidationError(
            f"malformed promotion certificate JSON in {target}: {exc}"
        ) from exc
    certificate = validate_promotion_certificate(payload, envelope=envelope)
    canonical = _canonical_json_bytes(certificate)
    if raw != canonical:
        raise PromotionCertificateValidationError(
            "promotion certificate file is not exact canonical newline-terminated JSON"
        )
    expected = (
        _sha256(expected_sha256, "expected_sha256")
        if expected_sha256 is not None
        else _read_sidecar(Path(f"{target}.sha256"), target.name)
    )
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise PromotionCertificateValidationError(
            f"certificate SHA-256 mismatch: expected {expected}, observed {observed}"
        )
    return certificate


__all__ = [
    "BENCHMARK_EVIDENCE_SCHEMA_VERSION",
    "CERTIFICATE_REQUIRED_FIELDS",
    "CERTIFICATE_SCHEMA_VERSION",
    "CHECKPOINT_QUALIFICATION_SCHEMA_VERSION",
    "HASH_ALGORITHM",
    "PromotionCertificateValidationError",
    "REQUIRED_MEASURED_ARMS",
    "VERDICT_GATES",
    "VERDICT_SCHEMA_VERSION",
    "canonical_promotion_certificate_bytes",
    "checkpoint_qualification_claim",
    "generate_promotion_certificate",
    "load_promotion_certificate",
    "promotion_certificate_sha256",
    "require_promotable_certificate",
    "supported_envelope_claim",
    "validate_promotion_certificate",
    "write_promotion_certificate",
]
