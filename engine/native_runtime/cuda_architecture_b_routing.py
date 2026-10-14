"""Standalone pre-dispatch routing contract for CUDA Architecture B.

This module does not dispatch a runner and does not promote CUDA to production.
It only resolves a fully described request to a backend, or rejects it, so a
parent integration can make one auditable routing decision before execution.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any, Final


REQUEST_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_production_routing_request_v2"
)
RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_production_routing_receipt_v2"
)
CAPABILITY_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_production_capability_v2"
)
AUTHORIZATION_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_default_authorization_v1"
)
AUTHORIZATION_SOURCE: Final[str] = "external_release_registry"
AUTHORIZATION_DECISION_ID: Final[str] = (
    "ARCHB-CUDA-PATCH-P2-B1-B64-EXACTNESS-DEFAULT-20260720"
)

CUDA_BACKEND: Final[str] = "cuda_architecture_b"
CPP_CPU_BACKEND: Final[str] = "cpp_cpu"
LEGACY_PYTHON_BACKEND: Final[str] = "legacy_python"
NO_BACKEND: Final[str] = "none"

QUALIFIED_ENVELOPE_ID: Final[str] = "patch_p2_g4_b1_to_b64_terminal_compact_v2"
QUALIFIED_MIN_BIRTHS_PER_TICK: Final[int] = 1
QUALIFIED_MAX_BIRTHS_PER_TICK: Final[int] = 64
QUALIFIED_DEVICE: Final[str] = "cuda:0"
QUALIFIED_HARDWARE_NAME: Final[str] = "NVIDIA GeForce GTX 1660 Ti"
QUALIFIED_COMPUTE_CAPABILITY: Final[tuple[int, int]] = (7, 5)
# Match torch.cuda.get_device_properties(0).total_memory on the qualified host.
QUALIFIED_VRAM_BYTES: Final[int] = 6_024_855_552
QUALIFIED_VRAM_MIN_BYTES: Final[int] = 6_000_000_000
QUALIFIED_VRAM_MAX_BYTES: Final[int] = 6 * 1024 * 1024 * 1024
QUALIFIED_EXTENSION_ABI_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_windows_v2"
)
QUALIFIED_EXTENSION_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_flattened_generation_count_capability_v1"
)
QUALIFIED_OUTPUT_SCHEMA: Final[str] = (
    "eiel_patch_p2_production_compact_summary_v1"
)
QUALIFIED_PATCH_GEOMETRIES: Final[frozenset[tuple[int, tuple[int, int]]]] = (
    frozenset(
        {(64, (16, 16)), (128, (16, 16)), (256, (32, 32)), (512, (32, 32))}
    )
)
QUALIFIED_ARTIFACT_PATH: Final[str] = (
    public_runtime_path('artifacts/eiel_cuda_architecture_b_patch_p2_multibirth_v2/eiel_cuda_architecture_b_flattened_controller_v1.so')
)
QUALIFIED_RECEIPT_PATH: Final[str] = (
    public_runtime_path('artifacts/eiel_cuda_architecture_b_patch_p2_multibirth_v2/qualification.json')
)
QUALIFIED_REPORTING_RECEIPT_PATH: Final[str] = (
    public_runtime_path('artifacts/eiel_cuda_architecture_b_patch_p2_multibirth_v2/reporting.json')
)
QUALIFIED_ARTIFACT_SHA256: Final[str] = (
    "667aade97242a8b349537618c825bb7cbf8682ec979e6406e6fd8d44adb1035b"
)
QUALIFIED_MANIFEST_SHA256: Final[str] = (
    "d28c13b50834a0385a66a0c5488db0209b0f69152b86dff4b1cae1d6711d6f92"
)
QUALIFIED_RECEIPT_SHA256: Final[str] = (
    "315deaa1e913525b3a7eaa01e0031d2b6b8215f06fd751e1897e7f50a1b00873"
)
QUALIFIED_REPORTING_RECEIPT_SHA256: Final[str] = (
    "c3beed5a555fa1b48b55971687c35accaaef7ae69efb6ccc2abf7882fc7ac301"
)

_TOP_LEVEL_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "request_mode",
        "allow_cpp_cpu_fallback",
        "allow_explicit_cuda_exactness_override",
        "default_authorization",
        "dispatch_started",
        "hardware",
        "workload",
        "extension",
        "unsupported_mechanic_flags",
        "performance_decision",
        "performance_decision_measured",
    }
)
_HARDWARE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "detector_status",
        "device",
        "name",
        "compute_capability",
        "vram_bytes",
    }
)
_WORKLOAD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "world",
        "profile",
        "population",
        "grid",
        "policy_family",
        "generation_count",
        "max_births_per_tick",
        "batch_cells",
        "checkpoint_schedule",
        "output_schema",
    }
)
_EXTENSION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "available",
        "artifact_sha256",
        "manifest_sha256",
        "qualification_receipt_sha256",
        "reporting_receipt_sha256",
        "abi_version",
        "extension_version",
        "capability",
    }
)
_CAPABILITY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "envelope_id",
        "qualified",
        "runtime_jit",
        "hidden_fallback",
        "production_promotion",
        "entrypoints_verified",
        "patch_p2_qualification",
    }
)
_AUTHORIZATION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "automatic_default_authorized",
        "authorization_source",
        "authorization_decision_id",
        "promotion_registry_digest",
        "artifact_bindings",
        "selection_basis",
        "performance_claimed",
        "extension_self_promotion_claimed",
    }
)

_DEFAULT_HARDWARE: Final[dict[str, Any]] = {
    "detector_status": "unknown",
    "device": "unknown",
    "name": "unknown",
    "compute_capability": None,
    "vram_bytes": 0,
}
_DEFAULT_WORKLOAD: Final[dict[str, Any]] = {
    "world": "unknown",
    "profile": "unknown",
    "population": 0,
    "grid": None,
    "policy_family": "unknown",
    "generation_count": 0,
    "max_births_per_tick": 0,
    "batch_cells": 0,
    "checkpoint_schedule": "unknown",
    "output_schema": "unknown",
}
_DEFAULT_EXTENSION: Final[dict[str, Any]] = {
    "available": False,
    "artifact_sha256": "",
    "manifest_sha256": "",
    "qualification_receipt_sha256": "",
    "reporting_receipt_sha256": "",
    "abi_version": "unknown",
    "extension_version": "unknown",
    "capability": None,
}


def _unknown_field(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    scope: str,
) -> str | None:
    unknown = sorted(str(field) for field in set(value).difference(allowed))
    if unknown:
        return f"request_schema_unknown_field:{scope}.{unknown[0]}"
    return None


def _strict_pair(value: Any) -> tuple[int, int] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
        or any(type(item) is not int for item in value)
    ):
        return None
    return int(value[0]), int(value[1])


def _capability_supports_patch_geometry(
    capability: Mapping[str, Any],
    *,
    population: int,
    grid: tuple[int, int],
) -> bool:
    patch = capability.get("patch_p2_qualification")
    if not isinstance(patch, Mapping):
        return False
    supported = patch.get("supported_geometry_envelopes")
    if isinstance(supported, Sequence) and not isinstance(supported, (str, bytes)):
        for item in supported:
            if not isinstance(item, Mapping):
                continue
            if (
                item.get("population") == population
                and item.get("grid_size") == grid[0]
                and grid[0] == grid[1]
            ):
                return True
    return (
        patch.get("population") == population
        and patch.get("grid_size") == grid[0]
        and grid[0] == grid[1]
    )


def _strict_sha256(value: Any) -> str | None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        return None
    return value


def _invalid_scalar_field(request: Mapping[str, Any]) -> str | None:
    for name in ("schema_version", "request_mode", "performance_decision"):
        if type(request[name]) is not str:
            return f"request_schema_invalid_field:request.{name}"

    hardware = request["hardware"]
    for name in ("detector_status", "device", "name"):
        if type(hardware[name]) is not str:
            return f"request_schema_invalid_field:request.hardware.{name}"
    if (
        hardware["compute_capability"] is not None
        and _strict_pair(hardware["compute_capability"]) is None
    ):
        return (
            "request_schema_invalid_field:"
            "request.hardware.compute_capability"
        )
    if type(hardware["vram_bytes"]) is not int or hardware["vram_bytes"] < 0:
        return "request_schema_invalid_field:request.hardware.vram_bytes"

    workload = request["workload"]
    for name in (
        "world",
        "profile",
        "policy_family",
        "checkpoint_schedule",
        "output_schema",
    ):
        if type(workload[name]) is not str:
            return f"request_schema_invalid_field:request.workload.{name}"
    for name in (
        "population",
        "generation_count",
        "max_births_per_tick",
        "batch_cells",
    ):
        if type(workload[name]) is not int or workload[name] < 0:
            return f"request_schema_invalid_field:request.workload.{name}"
    if workload["grid"] is not None and _strict_pair(workload["grid"]) is None:
        return "request_schema_invalid_field:request.workload.grid"

    extension = request["extension"]
    if type(extension["available"]) is not bool:
        return "request_schema_invalid_field:request.extension.available"
    for name in (
        "artifact_sha256",
        "manifest_sha256",
        "qualification_receipt_sha256",
        "reporting_receipt_sha256",
        "abi_version",
        "extension_version",
    ):
        if type(extension[name]) is not str:
            return f"request_schema_invalid_field:request.extension.{name}"

    capability = extension["capability"]
    if capability is not None:
        for name in ("schema_version", "envelope_id"):
            if name in capability and type(capability[name]) is not str:
                return (
                    "request_schema_invalid_field:"
                    f"request.extension.capability.{name}"
                )
        for name in (
            "qualified",
            "runtime_jit",
            "hidden_fallback",
            "production_promotion",
            "entrypoints_verified",
        ):
            if name in capability and type(capability[name]) is not bool:
                return (
                    "request_schema_invalid_field:"
                    f"request.extension.capability.{name}"
                )
    return None


def _receipt(
    request: Mapping[str, Any],
    *,
    resolved_backend: str,
    reason: str,
    fallback_performed: bool = False,
) -> dict[str, Any]:
    mode = request.get("request_mode", "unknown")
    requested_backend = CUDA_BACKEND if mode == "explicit_cuda" else str(mode)
    extension = request.get("extension")
    artifact_sha256 = (
        extension.get("artifact_sha256", "")
        if isinstance(extension, Mapping)
        else ""
    )
    hardware = request.get("hardware")
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract_status": "resolver_contract_pending_parent_integration",
        "routing_integrated": False,
        "production_promotion_claimed": False,
        "automatic_default_authorized": bool(
            isinstance(request.get("default_authorization"), Mapping)
            and request["default_authorization"].get(
                "automatic_default_authorized"
            )
            is True
        ),
        "authorization_source": (
            request["default_authorization"].get("authorization_source", "")
            if isinstance(request.get("default_authorization"), Mapping)
            else ""
        ),
        "authorization_decision_id": (
            request["default_authorization"].get(
                "authorization_decision_id", ""
            )
            if isinstance(request.get("default_authorization"), Mapping)
            else ""
        ),
        "promotion_registry_digest": (
            request["default_authorization"].get(
                "promotion_registry_digest", ""
            )
            if isinstance(request.get("default_authorization"), Mapping)
            else ""
        ),
        "selection_basis": (
            request["default_authorization"].get("selection_basis", "")
            if isinstance(request.get("default_authorization"), Mapping)
            else ""
        ),
        "performance_claimed": False,
        "extension_self_promotion_claimed": False,
        "request_mode": mode,
        "requested_backend": requested_backend,
        "resolved_backend": resolved_backend,
        "reason": reason,
        "extension_artifact_sha256": artifact_sha256,
        "hardware": deepcopy(dict(hardware)) if isinstance(hardware, Mapping) else {},
        "dispatch_started": request.get("dispatch_started", False),
        "fallback_performed": fallback_performed,
        "fallback_phase": "pre_dispatch" if fallback_performed else "none",
        "no_hidden_fallback": True,
        "post_dispatch_fallback_allowed": False,
    }


def _reject_or_fallback(
    request: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    if (
        request.get("request_mode") == "auto"
        and request.get("allow_cpp_cpu_fallback") is True
        and request.get("dispatch_started") is False
    ):
        return _receipt(
            request,
            resolved_backend=CPP_CPU_BACKEND,
            reason=reason,
            fallback_performed=True,
        )
    return _receipt(request, resolved_backend=NO_BACKEND, reason=reason)


def explicit_legacy_python_pre_dispatch_receipt(
    *,
    reason: str,
    request_mode: str = "auto",
) -> dict[str, Any]:
    """Describe the visible legacy owner used outside the CUDA envelope."""

    if request_mode != "auto":
        raise ValueError("only automatic requests may select the legacy owner")
    if type(reason) is not str or not reason or reason.strip() != reason:
        raise ValueError("pre-dispatch fallback reason must be a stable string")
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract_status": "pre_dispatch_legacy_python_selection",
        "routing_integrated": True,
        "production_promotion_claimed": False,
        "automatic_default_authorized": False,
        "authorization_source": "",
        "authorization_decision_id": "",
        "promotion_registry_digest": "",
        "selection_basis": "",
        "performance_claimed": False,
        "extension_self_promotion_claimed": False,
        "request_mode": request_mode,
        "requested_backend": "auto",
        "resolved_backend": LEGACY_PYTHON_BACKEND,
        "reason": reason,
        "extension_artifact_sha256": "",
        "hardware": {},
        "dispatch_started": False,
        "fallback_performed": True,
        "fallback_phase": "pre_dispatch",
        "no_hidden_fallback": True,
        "post_dispatch_fallback_allowed": False,
    }


def qualified_cuda_pre_dispatch_rejection_receipt(*, reason: str) -> dict[str, Any]:
    """Fail closed when an otherwise qualified default cannot load CUDA."""

    if type(reason) is not str or not reason or reason.strip() != reason:
        raise ValueError("pre-dispatch rejection reason must be a stable string")
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract_status": "qualified_cuda_pre_dispatch_rejected",
        "routing_integrated": True,
        "production_promotion_claimed": False,
        "automatic_default_authorized": True,
        "authorization_source": AUTHORIZATION_SOURCE,
        "authorization_decision_id": AUTHORIZATION_DECISION_ID,
        "promotion_registry_digest": "",
        "selection_basis": "exactness",
        "performance_claimed": False,
        "extension_self_promotion_claimed": False,
        "request_mode": "auto",
        "requested_backend": "auto",
        "resolved_backend": NO_BACKEND,
        "reason": reason,
        "extension_artifact_sha256": "",
        "hardware": {},
        "dispatch_started": False,
        "fallback_performed": False,
        "fallback_phase": "none",
        "no_hidden_fallback": True,
        "post_dispatch_fallback_allowed": False,
    }


def native_engine_pre_dispatch_rejection_receipt(
    *,
    reason: str,
    request_mode: str,
) -> dict[str, Any]:
    """Describe an explicit native-engine rejection before any owner starts."""

    if request_mode not in {"auto", "explicit_cuda"}:
        raise ValueError("native-engine rejection request mode is invalid")
    if type(reason) is not str or not reason or reason.strip() != reason:
        raise ValueError("pre-dispatch rejection reason must be a stable string")
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "contract_status": "native_engine_pre_dispatch_rejected",
        "routing_integrated": True,
        "production_promotion_claimed": False,
        "automatic_default_authorized": False,
        "authorization_source": "",
        "authorization_decision_id": "",
        "promotion_registry_digest": "",
        "selection_basis": "",
        "performance_claimed": False,
        "extension_self_promotion_claimed": False,
        "request_mode": request_mode,
        "requested_backend": "auto" if request_mode == "auto" else CUDA_BACKEND,
        "resolved_backend": NO_BACKEND,
        "reason": reason,
        "extension_artifact_sha256": "",
        "hardware": {},
        "dispatch_started": False,
        "fallback_performed": False,
        "fallback_phase": "none",
        "no_hidden_fallback": True,
        "post_dispatch_fallback_allowed": False,
    }


def _normalize_request(value: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(value, Mapping):
        return {}, "request_schema_not_mapping"
    request = dict(value)
    reason = _unknown_field(request, _TOP_LEVEL_FIELDS, scope="request")
    if reason is not None:
        return request, reason

    request.setdefault("schema_version", "unknown")
    request.setdefault("request_mode", "unknown")
    request.setdefault("allow_cpp_cpu_fallback", False)
    request.setdefault("allow_explicit_cuda_exactness_override", False)
    request.setdefault("default_authorization", None)
    request.setdefault("dispatch_started", False)
    request.setdefault("performance_decision", "unknown")
    request.setdefault("performance_decision_measured", False)
    request.setdefault("unsupported_mechanic_flags", {})

    for name in (
        "allow_cpp_cpu_fallback",
        "allow_explicit_cuda_exactness_override",
        "dispatch_started",
        "performance_decision_measured",
    ):
        if type(request[name]) is not bool:
            return request, f"request_schema_invalid_field:request.{name}"

    authorization = request["default_authorization"]
    if authorization is not None:
        if not isinstance(authorization, Mapping):
            return request, (
                "request_schema_invalid_field:request.default_authorization"
            )
        reason = _unknown_field(
            authorization,
            _AUTHORIZATION_FIELDS,
            scope="request.default_authorization",
        )
        if reason is not None:
            return request, reason
        request["default_authorization"] = dict(authorization)

    nested_specs = (
        ("hardware", _HARDWARE_FIELDS, _DEFAULT_HARDWARE),
        ("workload", _WORKLOAD_FIELDS, _DEFAULT_WORKLOAD),
        ("extension", _EXTENSION_FIELDS, _DEFAULT_EXTENSION),
    )
    for name, fields, defaults in nested_specs:
        raw = request.get(name, {})
        if not isinstance(raw, Mapping):
            return request, f"request_schema_invalid_field:request.{name}"
        reason = _unknown_field(raw, fields, scope=f"request.{name}")
        if reason is not None:
            return request, reason
        normalized = dict(defaults)
        normalized.update(raw)
        request[name] = normalized

    flags = request["unsupported_mechanic_flags"]
    if not isinstance(flags, Mapping):
        return request, (
            "request_schema_invalid_field:request.unsupported_mechanic_flags"
        )
    for name, enabled in sorted(flags.items(), key=lambda item: str(item[0])):
        if type(name) is not str or not name.strip() or type(enabled) is not bool:
            return request, (
                "request_schema_invalid_field:"
                f"request.unsupported_mechanic_flags.{name}"
            )
    request["unsupported_mechanic_flags"] = dict(flags)

    capability = request["extension"]["capability"]
    if capability is not None:
        if not isinstance(capability, Mapping):
            return request, (
                "request_schema_invalid_field:request.extension.capability"
            )
        reason = _unknown_field(
            capability,
            _CAPABILITY_FIELDS,
            scope="request.extension.capability",
        )
        if reason is not None:
            return request, reason
        request["extension"]["capability"] = dict(capability)
    return request, _invalid_scalar_field(request)


def _default_authorization_mismatch(request: Mapping[str, Any]) -> str | None:
    authorization = request.get("default_authorization")
    if authorization is None:
        return None
    if not isinstance(authorization, Mapping):
        return "default_authorization_not_mapping"
    expected = {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "automatic_default_authorized": True,
        "authorization_source": AUTHORIZATION_SOURCE,
        "authorization_decision_id": AUTHORIZATION_DECISION_ID,
        "selection_basis": "exactness",
        "performance_claimed": False,
        "extension_self_promotion_claimed": False,
    }
    for name, value in expected.items():
        if authorization.get(name) != value:
            return f"default_authorization_{name}_mismatch"
    if _strict_sha256(authorization.get("promotion_registry_digest")) is None:
        return "default_authorization_registry_digest_invalid"
    bindings = authorization.get("artifact_bindings")
    if not isinstance(bindings, Mapping):
        return "default_authorization_artifact_bindings_not_mapping"
    for name in (
        "artifact_sha256",
        "manifest_sha256",
        "qualification_receipt_sha256",
        "reporting_receipt_sha256",
    ):
        if _strict_sha256(bindings.get(name)) is None:
            return f"default_authorization_{name}_invalid"
    return None


def _authorized_extension_hashes(request: Mapping[str, Any]) -> dict[str, str]:
    authorization = request.get("default_authorization")
    if isinstance(authorization, Mapping):
        bindings = authorization.get("artifact_bindings")
        if isinstance(bindings, Mapping):
            return {
                name: str(bindings.get(name, ""))
                for name in (
                    "artifact_sha256",
                    "manifest_sha256",
                    "qualification_receipt_sha256",
                    "reporting_receipt_sha256",
                )
            }
    return {
        "artifact_sha256": QUALIFIED_ARTIFACT_SHA256,
        "manifest_sha256": QUALIFIED_MANIFEST_SHA256,
        "qualification_receipt_sha256": QUALIFIED_RECEIPT_SHA256,
        "reporting_receipt_sha256": QUALIFIED_REPORTING_RECEIPT_SHA256,
    }


def _first_envelope_mismatch(request: Mapping[str, Any]) -> str | None:
    hardware = request["hardware"]
    workload = request["workload"]
    extension = request["extension"]

    if hardware["detector_status"] != "detected":
        return "hardware_detector_unknown"
    if hardware["device"] != QUALIFIED_DEVICE:
        return "hardware_device_mismatch"
    if hardware["name"] != QUALIFIED_HARDWARE_NAME:
        return "hardware_name_mismatch"
    if _strict_pair(hardware["compute_capability"]) != QUALIFIED_COMPUTE_CAPABILITY:
        return "compute_capability_mismatch"
    if not (
        type(hardware["vram_bytes"]) is int
        and QUALIFIED_VRAM_MIN_BYTES
        <= hardware["vram_bytes"]
        <= QUALIFIED_VRAM_MAX_BYTES
    ):
        return "vram_mismatch"

    exact_workload = (
        ("world", "patch"),
        ("profile", "P2"),
        ("policy_family", "stochastic_fixed_mlp"),
        ("generation_count", 4),
        ("batch_cells", 1),
        ("checkpoint_schedule", "terminal_only"),
        ("output_schema", QUALIFIED_OUTPUT_SCHEMA),
    )
    workload_grid = _strict_pair(workload["grid"])
    if (workload["population"], workload_grid) not in QUALIFIED_PATCH_GEOMETRIES:
        return "geometry_mismatch"
    for field, expected in exact_workload:
        if workload[field] != expected:
            return f"{field}_mismatch"
    if not (
        type(workload["max_births_per_tick"]) is int
        and QUALIFIED_MIN_BIRTHS_PER_TICK
        <= workload["max_births_per_tick"]
        <= QUALIFIED_MAX_BIRTHS_PER_TICK
    ):
        return "max_births_per_tick_outside_qualified_range"

    enabled_mechanics = sorted(
        name
        for name, enabled in request["unsupported_mechanic_flags"].items()
        if enabled
    )
    if enabled_mechanics:
        return f"unsupported_mechanic:{enabled_mechanics[0]}"

    if extension["available"] is not True:
        return "extension_missing"
    artifact_sha256 = _strict_sha256(extension["artifact_sha256"])
    manifest_sha256 = _strict_sha256(extension["manifest_sha256"])
    qualification_sha256 = _strict_sha256(
        extension["qualification_receipt_sha256"]
    )
    reporting_sha256 = _strict_sha256(extension["reporting_receipt_sha256"])
    if (
        artifact_sha256 is None
        or manifest_sha256 is None
        or qualification_sha256 is None
        or reporting_sha256 is None
    ):
        return "extension_hash_missing_or_invalid"
    authorized_hashes = _authorized_extension_hashes(request)
    if artifact_sha256 != authorized_hashes["artifact_sha256"]:
        return "extension_hash_mismatch"
    if manifest_sha256 != authorized_hashes["manifest_sha256"]:
        return "extension_manifest_hash_mismatch"
    if qualification_sha256 != authorized_hashes["qualification_receipt_sha256"]:
        return "extension_qualification_receipt_hash_mismatch"
    if reporting_sha256 != authorized_hashes["reporting_receipt_sha256"]:
        return "extension_reporting_receipt_hash_mismatch"
    if extension["abi_version"] != QUALIFIED_EXTENSION_ABI_VERSION:
        return "extension_abi_mismatch"
    if extension["extension_version"] != QUALIFIED_EXTENSION_VERSION:
        return "extension_version_mismatch"

    capability = extension["capability"]
    if capability is None:
        return "extension_capability_missing"
    if capability.get("schema_version") != CAPABILITY_SCHEMA_VERSION:
        return "extension_capability_schema_mismatch"
    if capability.get("envelope_id") != QUALIFIED_ENVELOPE_ID:
        return "extension_capability_envelope_mismatch"
    if capability.get("qualified") is not True:
        return "extension_capability_not_qualified"
    if capability.get("runtime_jit") is not False:
        return "extension_capability_runtime_jit_not_prohibited"
    if capability.get("hidden_fallback") is not False:
        return "extension_capability_hidden_fallback_not_prohibited"
    if capability.get("production_promotion") is not False:
        return "extension_capability_claims_production_promotion"
    if capability.get("entrypoints_verified") is not True:
        return "extension_capability_entrypoints_not_verified"
    if not _capability_supports_patch_geometry(
        capability,
        population=workload["population"],
        grid=workload_grid,
    ):
        return "extension_capability_geometry_mismatch"
    return None


def resolve_cuda_architecture_b_production_route(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one Architecture B request before dispatch.

    The resolver returns exactly one reason. Explicit CUDA never falls back.
    Automatic requests may select the explicit C++ CPU backend only before
    dispatch and only when ``allow_cpp_cpu_fallback`` is true.
    """

    normalized, schema_reason = _normalize_request(request)
    if schema_reason is not None:
        return _receipt(
            normalized,
            resolved_backend=NO_BACKEND,
            reason=schema_reason,
        )
    if normalized["schema_version"] != REQUEST_SCHEMA_VERSION:
        return _receipt(
            normalized,
            resolved_backend=NO_BACKEND,
            reason="request_schema_version_mismatch",
        )
    if normalized["request_mode"] not in {"auto", "explicit_cuda"}:
        return _receipt(
            normalized,
            resolved_backend=NO_BACKEND,
            reason="request_mode_invalid",
        )
    if normalized["dispatch_started"]:
        return _receipt(
            normalized,
            resolved_backend=NO_BACKEND,
            reason="post_dispatch_routing_prohibited",
        )

    authorization_mismatch = _default_authorization_mismatch(normalized)
    if authorization_mismatch is not None:
        return _receipt(
            normalized,
            resolved_backend=NO_BACKEND,
            reason=authorization_mismatch,
        )

    mismatch = _first_envelope_mismatch(normalized)
    if mismatch is not None:
        return _reject_or_fallback(normalized, mismatch)

    decision = normalized["performance_decision"]
    measured = normalized["performance_decision_measured"]
    if normalized["request_mode"] == "auto":
        if normalized["default_authorization"] is not None:
            return _receipt(
                normalized,
                resolved_backend=CUDA_BACKEND,
                reason="external_registry_authorized_exact_cuda_default",
            )
        if measured is True and decision == "A":
            return _receipt(
                normalized,
                resolved_backend=CUDA_BACKEND,
                reason="measured_decision_a_exact_cuda_envelope",
            )
        if normalized["allow_cpp_cpu_fallback"]:
            reason = (
                "measured_decision_b_gtx1660ti_cpp_cpu_default"
                if measured is True and decision == "B"
                else "performance_decision_unknown_or_unmeasured"
            )
            return _receipt(
                normalized,
                resolved_backend=CPP_CPU_BACKEND,
                reason=reason,
                fallback_performed=True,
            )
        return _receipt(
            normalized,
            resolved_backend=NO_BACKEND,
            reason="performance_decision_unknown_or_unmeasured",
        )

    if normalized["default_authorization"] is not None:
        return _receipt(
            normalized,
            resolved_backend=CUDA_BACKEND,
            reason="explicit_cuda_external_registry_authorized_exact_envelope",
        )
    if measured is True and decision == "A":
        return _receipt(
            normalized,
            resolved_backend=CUDA_BACKEND,
            reason="explicit_cuda_measured_decision_a_exact_envelope",
        )
    if normalized["allow_explicit_cuda_exactness_override"] and (
        (measured is True and decision == "B")
        or (measured is False and decision == "unmeasured")
    ):
        return _receipt(
            normalized,
            resolved_backend=CUDA_BACKEND,
            reason=(
                "explicit_cuda_measured_decision_b_override_exact_envelope"
                if measured is True
                else "explicit_cuda_unmeasured_exactness_override_exact_envelope"
            ),
        )
    return _receipt(
        normalized,
        resolved_backend=NO_BACKEND,
        reason=(
            "explicit_cuda_exactness_override_not_allowed"
            if decision in {"B", "unmeasured"}
            else "performance_decision_unknown_or_unmeasured"
        ),
    )


resolve_production_route = resolve_cuda_architecture_b_production_route


__all__ = [
    "AUTHORIZATION_DECISION_ID",
    "AUTHORIZATION_SCHEMA_VERSION",
    "AUTHORIZATION_SOURCE",
    "CAPABILITY_SCHEMA_VERSION",
    "CPP_CPU_BACKEND",
    "CUDA_BACKEND",
    "LEGACY_PYTHON_BACKEND",
    "NO_BACKEND",
    "QUALIFIED_ARTIFACT_SHA256",
    "QUALIFIED_COMPUTE_CAPABILITY",
    "QUALIFIED_DEVICE",
    "QUALIFIED_ENVELOPE_ID",
    "QUALIFIED_EXTENSION_ABI_VERSION",
    "QUALIFIED_EXTENSION_VERSION",
    "QUALIFIED_HARDWARE_NAME",
    "QUALIFIED_MANIFEST_SHA256",
    "QUALIFIED_OUTPUT_SCHEMA",
    "QUALIFIED_RECEIPT_SHA256",
    "QUALIFIED_REPORTING_RECEIPT_SHA256",
    "QUALIFIED_ARTIFACT_PATH",
    "QUALIFIED_RECEIPT_PATH",
    "QUALIFIED_REPORTING_RECEIPT_PATH",
    "QUALIFIED_VRAM_BYTES",
    "QUALIFIED_VRAM_MAX_BYTES",
    "QUALIFIED_VRAM_MIN_BYTES",
    "RECEIPT_SCHEMA_VERSION",
    "REQUEST_SCHEMA_VERSION",
    "resolve_cuda_architecture_b_production_route",
    "explicit_legacy_python_pre_dispatch_receipt",
    "qualified_cuda_pre_dispatch_rejection_receipt",
    "resolve_production_route",
]
