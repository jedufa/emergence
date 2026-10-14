from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from engine.native_runtime.cuda_architecture_b_routing import (
    AUTHORIZATION_DECISION_ID,
    AUTHORIZATION_SCHEMA_VERSION,
    AUTHORIZATION_SOURCE,
    CAPABILITY_SCHEMA_VERSION,
    CPP_CPU_BACKEND,
    CUDA_BACKEND,
    LEGACY_PYTHON_BACKEND,
    NO_BACKEND,
    QUALIFIED_ARTIFACT_SHA256,
    QUALIFIED_ENVELOPE_ID,
    QUALIFIED_EXTENSION_ABI_VERSION,
    QUALIFIED_EXTENSION_VERSION,
    QUALIFIED_MANIFEST_SHA256,
    QUALIFIED_OUTPUT_SCHEMA,
    QUALIFIED_RECEIPT_SHA256,
    QUALIFIED_REPORTING_RECEIPT_SHA256,
    QUALIFIED_VRAM_BYTES,
    QUALIFIED_VRAM_MAX_BYTES,
    QUALIFIED_VRAM_MIN_BYTES,
    RECEIPT_SCHEMA_VERSION,
    REQUEST_SCHEMA_VERSION,
    resolve_cuda_architecture_b_production_route,
    explicit_legacy_python_pre_dispatch_receipt,
)


def test_qualified_vram_matches_live_torch_device_identity() -> None:
    assert QUALIFIED_VRAM_BYTES == 6_024_855_552


@pytest.mark.parametrize(
    "vram_bytes",
    [QUALIFIED_VRAM_BYTES, 6 * 1024 * 1024 * 1024],
)
def test_qualified_vram_range_accepts_live_detector_variants(
    vram_bytes: int,
) -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(hardware={"vram_bytes": vram_bytes})
    )

    assert receipt["resolved_backend"] == CUDA_BACKEND


@pytest.mark.parametrize(
    "vram_bytes",
    [QUALIFIED_VRAM_MIN_BYTES - 1, QUALIFIED_VRAM_MAX_BYTES + 1],
)
def test_qualified_vram_range_rejects_outside_device_class(
    vram_bytes: int,
) -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(request_mode="explicit_cuda", hardware={"vram_bytes": vram_bytes})
    )

    assert receipt["resolved_backend"] == NO_BACKEND
    assert receipt["reason"] == "vram_mismatch"


def _request(**overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "request_mode": "auto",
        "allow_cpp_cpu_fallback": True,
        "allow_explicit_cuda_exactness_override": False,
        "default_authorization": {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "automatic_default_authorized": True,
            "authorization_source": AUTHORIZATION_SOURCE,
            "authorization_decision_id": AUTHORIZATION_DECISION_ID,
            "promotion_registry_digest": "a" * 64,
            "artifact_bindings": {
                "artifact_sha256": QUALIFIED_ARTIFACT_SHA256,
                "manifest_sha256": QUALIFIED_MANIFEST_SHA256,
                "qualification_receipt_sha256": QUALIFIED_RECEIPT_SHA256,
                "reporting_receipt_sha256": QUALIFIED_REPORTING_RECEIPT_SHA256,
            },
            "selection_basis": "exactness",
            "performance_claimed": False,
            "extension_self_promotion_claimed": False,
        },
        "dispatch_started": False,
        "hardware": {
            "detector_status": "detected",
            "device": "cuda:0",
            "name": "NVIDIA GeForce GTX 1660 Ti",
            "compute_capability": [7, 5],
            "vram_bytes": QUALIFIED_VRAM_BYTES,
        },
        "workload": {
            "world": "patch",
            "profile": "P2",
            "population": 64,
            "grid": [16, 16],
            "policy_family": "stochastic_fixed_mlp",
            "generation_count": 4,
            "max_births_per_tick": 1,
            "batch_cells": 1,
            "checkpoint_schedule": "terminal_only",
            "output_schema": QUALIFIED_OUTPUT_SCHEMA,
        },
        "extension": {
            "available": True,
            "artifact_sha256": QUALIFIED_ARTIFACT_SHA256,
            "manifest_sha256": QUALIFIED_MANIFEST_SHA256,
            "qualification_receipt_sha256": QUALIFIED_RECEIPT_SHA256,
            "reporting_receipt_sha256": QUALIFIED_REPORTING_RECEIPT_SHA256,
            "abi_version": QUALIFIED_EXTENSION_ABI_VERSION,
            "extension_version": QUALIFIED_EXTENSION_VERSION,
            "capability": {
                "schema_version": CAPABILITY_SCHEMA_VERSION,
                "envelope_id": QUALIFIED_ENVELOPE_ID,
                "qualified": True,
                "runtime_jit": False,
                "hidden_fallback": False,
                "production_promotion": False,
                "entrypoints_verified": True,
                "patch_p2_qualification": {
                    "population": 64,
                    "grid_size": 16,
                    "supported_geometry_envelopes": [
                        {"population": 64, "grid_size": 16},
                        {"population": 128, "grid_size": 16},
                        {"population": 256, "grid_size": 32},
                        {"population": 512, "grid_size": 32},
                    ],
                },
            },
        },
        "unsupported_mechanic_flags": {
            "jungle": False,
            "wdl": False,
            "gru": False,
            "mate_required_reproduction": False,
        },
        "performance_decision": "A",
        "performance_decision_measured": True,
    }
    for key, value in overrides.items():
        if key in {"hardware", "workload", "extension"}:
            request[key].update(value)
        else:
            request[key] = value
    return request


def test_external_registry_authorized_exact_auto_route_selects_cuda() -> None:
    receipt = resolve_cuda_architecture_b_production_route(_request())

    assert receipt == {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract_status": "resolver_contract_pending_parent_integration",
        "routing_integrated": False,
        "production_promotion_claimed": False,
        "automatic_default_authorized": True,
        "authorization_source": AUTHORIZATION_SOURCE,
        "authorization_decision_id": AUTHORIZATION_DECISION_ID,
        "promotion_registry_digest": "a" * 64,
        "selection_basis": "exactness",
        "performance_claimed": False,
        "extension_self_promotion_claimed": False,
        "request_mode": "auto",
        "requested_backend": "auto",
        "resolved_backend": CUDA_BACKEND,
        "reason": "external_registry_authorized_exact_cuda_default",
        "extension_artifact_sha256": QUALIFIED_ARTIFACT_SHA256,
        "hardware": {
            "detector_status": "detected",
            "device": "cuda:0",
            "name": "NVIDIA GeForce GTX 1660 Ti",
            "compute_capability": [7, 5],
            "vram_bytes": QUALIFIED_VRAM_BYTES,
        },
        "dispatch_started": False,
        "fallback_performed": False,
        "fallback_phase": "none",
        "no_hidden_fallback": True,
        "post_dispatch_fallback_allowed": False,
    }


@pytest.mark.parametrize("max_births_per_tick", [2, 64])
def test_auto_route_accepts_qualified_multi_birth_contract(
    max_births_per_tick: int,
) -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(workload={"max_births_per_tick": max_births_per_tick})
    )

    assert receipt["resolved_backend"] == CUDA_BACKEND


def test_auto_route_accepts_w32p256_geometry_when_capability_advertises_it() -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(workload={"population": 256, "grid": [32, 32]})
    )

    assert receipt["resolved_backend"] == CUDA_BACKEND


def test_auto_route_accepts_w16p128_geometry_when_capability_advertises_it() -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(workload={"population": 128, "grid": [16, 16]})
    )

    assert receipt["resolved_backend"] == CUDA_BACKEND


def test_external_registry_binds_extension_hashes_without_global_promotion() -> None:
    request = _request()
    bindings = {
        "artifact_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
        "qualification_receipt_sha256": "d" * 64,
        "reporting_receipt_sha256": "e" * 64,
    }
    request["default_authorization"] = dict(request["default_authorization"])
    request["default_authorization"]["artifact_bindings"] = bindings
    request["extension"] = dict(request["extension"])
    request["extension"].update(bindings)

    receipt = resolve_cuda_architecture_b_production_route(request)

    assert receipt["resolved_backend"] == CUDA_BACKEND
    assert receipt["reason"] == "external_registry_authorized_exact_cuda_default"
    assert receipt["extension_artifact_sha256"] == "b" * 64


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        (
            "automatic_default_authorized",
            False,
            "default_authorization_automatic_default_authorized_mismatch",
        ),
        (
            "authorization_source",
            "extension",
            "default_authorization_authorization_source_mismatch",
        ),
        (
            "selection_basis",
            "performance",
            "default_authorization_selection_basis_mismatch",
        ),
        (
            "performance_claimed",
            True,
            "default_authorization_performance_claimed_mismatch",
        ),
        (
            "promotion_registry_digest",
            "invalid",
            "default_authorization_registry_digest_invalid",
        ),
    ],
)
def test_tampered_default_authorization_fails_closed(
    field: str,
    value: object,
    reason: str,
) -> None:
    request = _request()
    request["default_authorization"] = dict(request["default_authorization"])
    request["default_authorization"][field] = value

    receipt = resolve_cuda_architecture_b_production_route(request)

    assert receipt["resolved_backend"] == NO_BACKEND
    assert receipt["reason"] == reason
    assert receipt["fallback_performed"] is False
    assert receipt["dispatch_started"] is False


@pytest.mark.parametrize("max_births_per_tick", [0, 65])
def test_auto_route_rejects_birth_contract_outside_qualified_range(
    max_births_per_tick: int,
) -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(workload={"max_births_per_tick": max_births_per_tick})
    )

    assert receipt["resolved_backend"] == CPP_CPU_BACKEND
    assert receipt["reason"] == "max_births_per_tick_outside_qualified_range"
    assert receipt["fallback_performed"] is True


def test_unpromoted_measured_decision_b_defaults_to_cpp_cpu() -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(
            default_authorization=None,
            performance_decision="B",
        )
    )

    assert receipt["resolved_backend"] == CPP_CPU_BACKEND
    assert receipt["reason"] == "measured_decision_b_gtx1660ti_cpp_cpu_default"
    assert receipt["fallback_performed"] is True
    assert receipt["fallback_phase"] == "pre_dispatch"
    assert receipt["no_hidden_fallback"] is True


def test_explicit_decision_b_override_is_allowed_only_inside_exact_envelope() -> None:
    exact = resolve_cuda_architecture_b_production_route(
        _request(
            default_authorization=None,
            request_mode="explicit_cuda",
            performance_decision="B",
            allow_explicit_cuda_exactness_override=True,
        )
    )
    outside = resolve_cuda_architecture_b_production_route(
        _request(
            default_authorization=None,
            request_mode="explicit_cuda",
            performance_decision="B",
            allow_explicit_cuda_exactness_override=True,
            workload={"generation_count": 8},
        )
    )

    assert exact["requested_backend"] == CUDA_BACKEND
    assert exact["resolved_backend"] == CUDA_BACKEND
    assert exact["reason"] == (
        "explicit_cuda_measured_decision_b_override_exact_envelope"
    )
    assert outside["resolved_backend"] == NO_BACKEND
    assert outside["reason"] == "generation_count_mismatch"
    assert outside["fallback_performed"] is False


def test_explicit_decision_b_requires_override_policy() -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(
            default_authorization=None,
            request_mode="explicit_cuda",
            performance_decision="B",
        )
    )

    assert receipt["resolved_backend"] == NO_BACKEND
    assert receipt["reason"] == "explicit_cuda_exactness_override_not_allowed"
    assert receipt["fallback_performed"] is False


def test_explicit_unmeasured_route_requires_exactness_override() -> None:
    rejected = resolve_cuda_architecture_b_production_route(
        _request(
            default_authorization=None,
            request_mode="explicit_cuda",
            performance_decision="unmeasured",
            performance_decision_measured=False,
        )
    )
    accepted = resolve_cuda_architecture_b_production_route(
        _request(
            default_authorization=None,
            request_mode="explicit_cuda",
            allow_explicit_cuda_exactness_override=True,
            performance_decision="unmeasured",
            performance_decision_measured=False,
        )
    )

    assert rejected["resolved_backend"] == NO_BACKEND
    assert rejected["reason"] == "explicit_cuda_exactness_override_not_allowed"
    assert accepted["resolved_backend"] == CUDA_BACKEND
    assert accepted["reason"] == (
        "explicit_cuda_unmeasured_exactness_override_exact_envelope"
    )


def test_unmeasured_auto_route_stays_on_cpp_cpu_before_dispatch() -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(
            default_authorization=None,
            performance_decision="unmeasured",
            performance_decision_measured=False,
        )
    )

    assert receipt["resolved_backend"] == CPP_CPU_BACKEND
    assert receipt["reason"] == "performance_decision_unknown_or_unmeasured"
    assert receipt["fallback_performed"] is True


@pytest.mark.parametrize(
    ("extension_override", "reason"),
    [
        ({"available": False}, "extension_missing"),
        ({"artifact_sha256": "b" * 64}, "extension_hash_mismatch"),
        ({"manifest_sha256": "b" * 64}, "extension_manifest_hash_mismatch"),
        (
            {"qualification_receipt_sha256": "b" * 64},
            "extension_qualification_receipt_hash_mismatch",
        ),
        (
            {"reporting_receipt_sha256": "b" * 64},
            "extension_reporting_receipt_hash_mismatch",
        ),
        ({"manifest_sha256": ""}, "extension_hash_missing_or_invalid"),
        ({"abi_version": "wrong-abi"}, "extension_abi_mismatch"),
        ({"extension_version": "wrong-version"}, "extension_version_mismatch"),
        ({"capability": None}, "extension_capability_missing"),
    ],
)
def test_explicit_cuda_extension_failures_have_one_reason_and_no_fallback(
    extension_override: dict[str, Any],
    reason: str,
) -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(request_mode="explicit_cuda", extension=extension_override)
    )

    assert receipt["resolved_backend"] == NO_BACKEND
    assert receipt["reason"] == reason
    assert receipt["fallback_performed"] is False
    assert receipt["no_hidden_fallback"] is True


@pytest.mark.parametrize(
    ("workload_override", "reason"),
    [
        ({"world": "flat"}, "world_mismatch"),
        ({"profile": "P1"}, "profile_mismatch"),
        ({"population": 63}, "geometry_mismatch"),
        ({"grid": [32, 32]}, "geometry_mismatch"),
        ({"policy_family": "gru"}, "policy_family_mismatch"),
        ({"generation_count": 8}, "generation_count_mismatch"),
        ({"batch_cells": 48}, "batch_cells_mismatch"),
        (
            {"checkpoint_schedule": "midpoint_and_terminal"},
            "checkpoint_schedule_mismatch",
        ),
        (
            {"output_schema": "eiel_compact_generation_summary_v1"},
            "output_schema_mismatch",
        ),
    ],
)
def test_exact_workload_schema_mismatches_fail_closed_for_explicit_cuda(
    workload_override: dict[str, Any],
    reason: str,
) -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(request_mode="explicit_cuda", workload=workload_override)
    )

    assert receipt["resolved_backend"] == NO_BACKEND
    assert receipt["reason"] == reason
    assert receipt["fallback_performed"] is False


def test_unknown_hardware_detector_fails_closed() -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(
            request_mode="explicit_cuda",
            hardware={
                "detector_status": "unknown",
                "device": "unknown",
                "name": "unknown",
                "compute_capability": None,
                "vram_bytes": 0,
            },
        )
    )

    assert receipt["resolved_backend"] == NO_BACKEND
    assert receipt["reason"] == "hardware_detector_unknown"


def test_unsupported_auto_routes_to_explicit_cpp_cpu_before_dispatch() -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        _request(unsupported_mechanic_flags={"jungle": True})
    )

    assert receipt["resolved_backend"] == CPP_CPU_BACKEND
    assert receipt["reason"] == "unsupported_mechanic:jungle"
    assert receipt["fallback_performed"] is True
    assert receipt["fallback_phase"] == "pre_dispatch"


def test_no_post_dispatch_fallback_is_possible() -> None:
    request = _request(
        dispatch_started=True,
        performance_decision="B",
        unsupported_mechanic_flags={"jungle": True},
    )
    receipt = resolve_cuda_architecture_b_production_route(request)

    assert receipt["resolved_backend"] == NO_BACKEND
    assert receipt["reason"] == "post_dispatch_routing_prohibited"
    assert receipt["fallback_performed"] is False
    assert receipt["fallback_phase"] == "none"
    assert receipt["post_dispatch_fallback_allowed"] is False
    assert receipt["no_hidden_fallback"] is True


def test_exact_schema_rejects_unknown_fields_without_mutating_request() -> None:
    request = _request()
    before = deepcopy(request)
    request["runner_backend"] = "cuda"

    receipt = resolve_cuda_architecture_b_production_route(request)

    assert receipt["resolved_backend"] == NO_BACKEND
    assert receipt["reason"] == "request_schema_unknown_field:request.runner_backend"
    assert request == {**before, "runner_backend": "cuda"}


def test_missing_inputs_default_to_unknown_and_never_select_a_backend() -> None:
    receipt = resolve_cuda_architecture_b_production_route(
        {"schema_version": REQUEST_SCHEMA_VERSION}
    )

    assert receipt["resolved_backend"] == NO_BACKEND
    assert receipt["reason"] == "request_mode_invalid"
    assert receipt["no_hidden_fallback"] is True


def test_visible_pre_dispatch_legacy_receipt_never_claims_cpp_ownership() -> None:
    receipt = explicit_legacy_python_pre_dispatch_receipt(
        reason="outside_qualified_cuda_envelope:generation_count_mismatch"
    )

    assert receipt["resolved_backend"] == LEGACY_PYTHON_BACKEND
    assert receipt["production_promotion_claimed"] is False
    assert receipt["automatic_default_authorized"] is False
    assert receipt["dispatch_started"] is False
    assert receipt["fallback_performed"] is True
    assert receipt["fallback_phase"] == "pre_dispatch"
    assert receipt["no_hidden_fallback"] is True
    assert receipt["post_dispatch_fallback_allowed"] is False


def test_malformed_enum_type_returns_a_schema_receipt_instead_of_raising() -> None:
    request = _request()
    request["request_mode"] = ["auto"]

    receipt = resolve_cuda_architecture_b_production_route(request)

    assert receipt["resolved_backend"] == NO_BACKEND
    assert receipt["reason"] == "request_schema_invalid_field:request.request_mode"
