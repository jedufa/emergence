from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from native_runtime.promotion_envelope import (
    CONTRACT_SCHEMA_VERSION,
    DEFAULT_CONTRACT_RELATIVE_PATH,
    FlatNativePromotionEnvelope,
    PromotionEnvelopeValidationError,
    load_promotion_envelope,
    promotion_envelope_contract_path,
    validate_promotion_envelope,
)
from native_runtime.safe_default import (
    NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED,
    PARITY_EVIDENCE_VERSION,
)


CONTRACT_PATH = promotion_envelope_contract_path()


def _payload() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _write_payload(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "flat_native_promotion_envelope.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _nested(payload: object, path: tuple[str | int, ...]) -> Any:
    value = payload
    for part in path:
        value = value[part]  # type: ignore[index]
    return value


def test_committed_contract_loads_as_immutable_typed_data() -> None:
    contract = load_promotion_envelope()

    assert isinstance(contract, FlatNativePromotionEnvelope)
    assert contract.schema_version == CONTRACT_SCHEMA_VERSION
    assert contract.supported_profile.world == "flat"
    assert contract.supported_profile.policy.family == "mlp"
    assert contract.supported_profile.policy.selection == "fixed"
    assert contract.supported_profile.device.simulation == "cpu"
    assert contract.supported_profile.device.tensor_bank == "cpu"
    assert contract.supported_profile.lifecycle.branch == "flat_generation_birth_commit"
    assert contract.supported_profile.chemistry.mode == "disabled"
    assert contract.supported_profile.structural_operators.c1_mode == "disabled"
    assert contract.supported_profile.checkpoint.mode("off").supported is True
    assert contract.supported_profile.checkpoint.mode("native_save_resume").supported is True
    assert contract.supported_profile.checkpoint.mode("partial_generation").supported is False
    assert contract.supported_profile.reporting.schema_version == "eiel_flat_native_reporting_v1"
    assert contract.supported_profile.reporting.profile == "compact_route_neutral_exact"
    assert contract.supported_profile.reporting.qualification == "route_neutral_exact"
    assert contract.supported_profile.reporting.metrics_csv_exact is True
    assert contract.supported_profile.reporting.behaviors_exact is True
    assert contract.supported_profile.reporting.lineage_exact is True
    assert contract.supported_profile.reporting.typed_lifecycle_exact is True
    assert contract.supported_profile.output.campaign_profile == "compact_default"
    assert contract.binary_contract.extension_abi_version == "eiel_native_runtime_abi_v1"
    assert contract.binary_contract.extension_version == "eiel_native_runtime_flat_v1"
    assert contract.promotion_thresholds.primary_performance.minimum_speedup == 1.5
    assert contract.promotion_thresholds.secondary_performance.minimum_speedup == 1.15
    assert contract.promotion_thresholds.stability.maximum == 0.05
    assert contract.promotion_thresholds.output_fairness.minimum == 0.8
    assert contract.promotion_thresholds.output_fairness.maximum == 1.25
    assert contract.promotion_thresholds.memory.maximum == 1.25
    assert contract.promotion_thresholds.memory.accepted_exceptions == ()
    assert "decision_timestamp" in contract.evidence_requirements.certificate_required_fields
    assert len(contract.post_promotion_proof.cases) == 12
    assert isinstance(contract.disallowed_features, tuple)
    assert isinstance(contract.failure_semantics.routing_rules, tuple)

    with pytest.raises(FrozenInstanceError):
        contract.contract_version = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        contract.supported_profile.policy.family = "gru"  # type: ignore[misc]


def test_contract_slice_matches_current_promotion_constants() -> None:
    assert NATIVE_SAFE_DEFAULT_PROMOTION_ACCEPTED is True
    assert PARITY_EVIDENCE_VERSION == "eiel-flat-native-authoritative-promotion-20260716"


def test_default_and_repo_relative_paths_are_independent_of_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    default_contract = load_promotion_envelope()
    relative_contract = load_promotion_envelope(DEFAULT_CONTRACT_RELATIVE_PATH)

    assert default_contract == relative_contract
    assert promotion_envelope_contract_path(DEFAULT_CONTRACT_RELATIVE_PATH) == CONTRACT_PATH


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "contract_id",
        "contract_version",
        "supported_profile",
        "disallowed_features",
        "binary_contract",
        "evidence_requirements",
        "failure_semantics",
        "promotion_thresholds",
        "post_promotion_proof",
    ],
)
def test_validator_rejects_every_missing_top_level_field(field: str) -> None:
    payload = _payload()
    del payload[field]

    with pytest.raises(PromotionEnvelopeValidationError, match="missing required fields"):
        validate_promotion_envelope(payload)


@pytest.mark.parametrize(
    ("container_path", "field"),
    [
        (("supported_profile", "policy"), "family"),
        (("supported_profile", "reporting", "schemas"), "typed_lifecycle"),
        (("supported_profile", "output"), "required_files"),
        (("binary_contract",), "extension_abi_version"),
        (("evidence_requirements",), "required_artifacts"),
        (("failure_semantics", "failure_atomicity"), "durable_first_cause_required"),
        (("promotion_thresholds", "memory"), "maximum"),
        (("post_promotion_proof", "expected_promotion_state"), "certificate_pointer_required"),
    ],
)
def test_validator_rejects_missing_nested_fields(
    container_path: tuple[str | int, ...],
    field: str,
) -> None:
    payload = _payload()
    del _nested(payload, container_path)[field]

    with pytest.raises(PromotionEnvelopeValidationError, match="missing required fields"):
        validate_promotion_envelope(payload)


@pytest.mark.parametrize(
    "container_path",
    [
        (),
        ("supported_profile",),
        ("supported_profile", "checkpoint", "modes", 0),
        ("binary_contract",),
        ("failure_semantics", "routing_rules", 0),
        ("promotion_thresholds", "output_fairness"),
        ("post_promotion_proof", "expected_promotion_state"),
    ],
)
def test_validator_rejects_unknown_fields_at_every_shape(
    container_path: tuple[str | int, ...],
) -> None:
    payload = _payload()
    _nested(payload, container_path)["unexpected"] = "not permitted"

    with pytest.raises(PromotionEnvelopeValidationError, match="unknown fields"):
        validate_promotion_envelope(payload)


@pytest.mark.parametrize(
    ("path", "tampered_value", "message"),
    [
        (("schema_version",), "eiel_flat_native_promotion_envelope_v2", "schema_version"),
        (("contract_version",), True, "must be an integer"),
        (("supported_profile", "world"), "patch", "supported_profile.world"),
        (("supported_profile", "policy", "resident_gru_allowed"), True, "resident_gru_allowed"),
        (("supported_profile", "device", "cuda_allowed"), True, "cuda_allowed"),
        (("supported_profile", "reporting", "metrics_csv_exact"), False, "metrics_csv_exact"),
        (("binary_contract", "extension_abi_version"), "wrong_abi", "extension_abi_version"),
        (("binary_contract", "extension_version"), "wrong_version", "extension_version"),
        (("binary_contract", "runtime_jit_allowed"), True, "runtime_jit_allowed"),
        (("promotion_thresholds", "primary_performance", "minimum_speedup"), 1.49, "minimum_speedup"),
        (("promotion_thresholds", "stability", "maximum"), 0.051, "stability.maximum"),
        (("promotion_thresholds", "output_fairness", "minimum"), 0.79, "output_fairness.minimum"),
        (("promotion_thresholds", "memory", "maximum"), 1.26, "memory.maximum"),
        (
            ("post_promotion_proof", "expected_promotion_state", "native_safe_default_promotion_accepted"),
            False,
            "native_safe_default_promotion_accepted",
        ),
    ],
)
def test_validator_rejects_scalar_tampering(
    path: tuple[str | int, ...],
    tampered_value: object,
    message: str,
) -> None:
    payload = _payload()
    parent = _nested(payload, path[:-1])
    parent[path[-1]] = tampered_value

    with pytest.raises(PromotionEnvelopeValidationError, match=message):
        validate_promotion_envelope(payload)


def test_validator_rejects_checkpoint_support_contradiction() -> None:
    payload = _payload()
    partial_mode = payload["supported_profile"]["checkpoint"]["modes"][-1]
    assert partial_mode["name"] == "partial_generation"
    partial_mode["supported"] = True

    with pytest.raises(PromotionEnvelopeValidationError, match="checkpoint matrix"):
        validate_promotion_envelope(payload)


def test_validator_rejects_failure_routing_contradiction() -> None:
    payload = _payload()
    runtime_failure = payload["failure_semantics"]["routing_rules"][-1]
    assert runtime_failure["trigger"] == "native_runtime_failure"
    runtime_failure["auto_behavior"] = "route_reference_with_structured_reason"

    with pytest.raises(PromotionEnvelopeValidationError, match="post-selection failure semantics"):
        validate_promotion_envelope(payload)


@pytest.mark.parametrize(
    ("sequence_path", "removed_value"),
    [
        (("disallowed_features",), "pulsed_shock"),
        (("binary_contract", "worker_identity_fields"), "extension_sha256"),
        (("evidence_requirements", "required_artifacts"), "promotion_certificate"),
        (("evidence_requirements", "certificate_required_fields"), "checkpoint_qualification"),
        (("failure_semantics", "native_result_rejections"), "hidden_fallback"),
        (("failure_semantics", "rejection_enforcement_scopes"), "promotion_gate"),
        (("failure_semantics", "checkpoint_write_failures"), "sidecar_write_failure"),
        (("promotion_thresholds", "fairness_requirements"), "no_mixed_ownership"),
        (("post_promotion_proof", "cases"), "checkpoint_write_error_rejected"),
    ],
)
def test_validator_rejects_required_sequence_tampering(
    sequence_path: tuple[str | int, ...],
    removed_value: str,
) -> None:
    payload = _payload()
    sequence = _nested(payload, sequence_path)
    sequence.remove(removed_value)

    with pytest.raises(PromotionEnvelopeValidationError, match="must exactly match"):
        validate_promotion_envelope(payload)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{not-json\n", "malformed promotion envelope JSON"),
        ('{"schema_version": "first", "schema_version": "second"}\n', "duplicate JSON field"),
        ('{"contract_version": NaN}\n', "non-finite JSON number"),
    ],
)
def test_loader_rejects_malformed_json(
    tmp_path: Path,
    contents: str,
    message: str,
) -> None:
    path = tmp_path / "malformed.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(PromotionEnvelopeValidationError, match=message):
        load_promotion_envelope(path)


def test_loader_rejects_unreadable_or_missing_contract(tmp_path: Path) -> None:
    with pytest.raises(PromotionEnvelopeValidationError, match="cannot read promotion envelope"):
        load_promotion_envelope(tmp_path / "missing.json")


def test_file_loader_rejects_tampered_payload(tmp_path: Path) -> None:
    payload = copy.deepcopy(_payload())
    payload["supported_profile"]["output"]["required_files"].append("partial-success.json")

    with pytest.raises(PromotionEnvelopeValidationError, match="must exactly match"):
        load_promotion_envelope(_write_payload(tmp_path, payload))
