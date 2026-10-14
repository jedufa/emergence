"""Artifact-bound production adapter for the qualified Stage 3 CUDA route."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final

from engine.perf.cuda_architecture_b_direct_resident_execution_v2 import (
    DIRECT_ENTRYPOINT_V2,
    RUN_CONTROL_TYPE_V2,
    DirectResidentNativeExecutionV2,
    create_run_control_v2,
    execute_direct_resident_windows_v2,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    DIRECT_ENTRYPOINT,
)

from .cuda_architecture_b_routing import (
    AUTHORIZATION_DECISION_ID,
    AUTHORIZATION_SCHEMA_VERSION,
    AUTHORIZATION_SOURCE,
    CAPABILITY_SCHEMA_VERSION,
    CPP_CPU_BACKEND,
    CUDA_BACKEND,
    QUALIFIED_COMPUTE_CAPABILITY,
    QUALIFIED_DEVICE,
    QUALIFIED_ENVELOPE_ID,
    QUALIFIED_EXTENSION_ABI_VERSION,
    QUALIFIED_EXTENSION_VERSION,
    QUALIFIED_HARDWARE_NAME,
    QUALIFIED_OUTPUT_SCHEMA,
    REQUEST_SCHEMA_VERSION,
    native_engine_pre_dispatch_rejection_receipt,
    resolve_cuda_architecture_b_production_route,
)
from .live_chronicle_stream import NativeRunControlLivePublisher


REGISTRY_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_patch_p2_stage3_qualification_v2"
)
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH: Final[Path] = (
    PROJECT_ROOT / "registries/performance/cuda_architecture_b_patch_p2_stage3_v2.json"
)
PROMOTION_REGISTRY_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_production_promotion_v1"
)
DEFAULT_PROMOTION_REGISTRY_PATH: Final[Path] = (
    PROJECT_ROOT
    / "registries/performance/"
    "cuda_architecture_b_patch_p2_production_promotion_v1.json"
)
MANIFEST_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_flattened_controller_extension_manifest_v1"
)
MODULE_NAME: Final[str] = "eiel_cuda_architecture_b_flattened_controller_v1"
RESOURCE_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_flattened_controller_resource_receipt_v1"
)
CAPABILITY_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_flattened_generation_count_capability_v1"
)
CONTROLLER_ENTRYPOINT: Final[str] = "eiel_cuda_architecture_b_flattened_controller_v1"
REQUIRED_ENTRYPOINTS: Final[tuple[str, ...]] = (
    CONTROLLER_ENTRYPOINT,
    RESOURCE_ENTRYPOINT,
    CAPABILITY_ENTRYPOINT,
    DIRECT_ENTRYPOINT,
    DIRECT_ENTRYPOINT_V2,
)
REQUIRED_ATTRIBUTES: Final[tuple[str, ...]] = (RUN_CONTROL_TYPE_V2,)
QUALIFICATION_DECISION: Final[str] = "CUDA-STAGE3-PATCH-P2-REUSABLE-G2-WINDOW/CLOSED"
REQUIRED_QUALIFICATION_CLAIMS: Final[tuple[str, ...]] = (
    "direct_cpp_window_core_dispatch_qualified",
    "resident_g2_g2_state_continuation_exact",
    "terminal_authoritative_and_canonical_checkpoint_exact",
    "native_run_control_v2_completion_exact",
)
QUALIFIED_PATCH_GEOMETRIES: Final[frozenset[tuple[int, tuple[int, int]]]] = (
    frozenset(
        {(64, (16, 16)), (128, (16, 16)), (256, (32, 32)), (512, (32, 32))}
    )
)


class CudaArchitectureBProductionAdapterError(RuntimeError):
    """The CUDA artifact, route, dispatch, or ownership receipt is invalid."""


class CudaArchitectureBProductionDispatchError(
    CudaArchitectureBProductionAdapterError
):
    """A selected CUDA dispatch failed and exposes its native terminal evidence."""

    def __init__(
        self,
        message: str,
        *,
        run_id: str,
        production_route_receipt: Mapping[str, Any],
        native_failure_receipt: Mapping[str, Any] | None,
        run_control_manifest: Mapping[str, Any] | None,
        native_terminal_result: Mapping[str, Any] | None,
        failure_settlement_error: Mapping[str, Any] | None,
    ) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.production_route_receipt = dict(production_route_receipt)
        self.native_failure_receipt = (
            None
            if native_failure_receipt is None
            else dict(native_failure_receipt)
        )
        self.run_control_manifest = (
            None if run_control_manifest is None else dict(run_control_manifest)
        )
        self.native_terminal_result = (
            None if native_terminal_result is None else dict(native_terminal_result)
        )
        self.failure_settlement_error = (
            None
            if failure_settlement_error is None
            else dict(failure_settlement_error)
        )


def _fail(message: str) -> CudaArchitectureBProductionAdapterError:
    return CudaArchitectureBProductionAdapterError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{field} must be a mapping")
    return dict(value)


def _load_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite number {value}")
            ),
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise _fail(f"cannot load {field}: {exc}") from exc
    return _mapping(payload, field=field)


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if type(actual) is not type(expected) or actual != expected:
        raise _fail(f"{field} mismatch: expected={expected!r} actual={actual!r}")


def _require_admitted_patch_geometry(
    *,
    population: Any,
    grid: Any,
    field: str,
) -> tuple[int, tuple[int, int]]:
    if (
        type(population) is not int
        or not isinstance(grid, Sequence)
        or isinstance(grid, (str, bytes))
        or len(grid) != 2
        or any(type(item) is not int for item in grid)
    ):
        raise _fail(f"{field} geometry is malformed")
    pair = (population, (int(grid[0]), int(grid[1])))
    if pair not in QUALIFIED_PATCH_GEOMETRIES:
        raise _fail(f"{field} geometry is not admitted")
    return pair


def _capability_supports_patch_geometry(
    patch: Mapping[str, Any],
    *,
    population: int,
    grid_size: int,
) -> bool:
    if (population, (grid_size, grid_size)) not in QUALIFIED_PATCH_GEOMETRIES:
        return False
    supported = patch.get("supported_geometry_envelopes")
    if isinstance(supported, Sequence) and not isinstance(supported, (str, bytes)):
        for item in supported:
            if not isinstance(item, Mapping):
                continue
            if (
                item.get("population") == population
                and item.get("grid_size") == grid_size
            ):
                return True
    return patch.get("population") == population and patch.get("grid_size") == grid_size


def _default_module_loader(path: Path, module_name: str) -> ModuleType:
    import torch  # noqa: F401, PLC0415

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise _fail(f"cannot construct extension loader for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_hardware_detector() -> dict[str, Any]:
    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        return {
            "detector_status": "unavailable",
            "device": "unknown",
            "name": "unknown",
            "compute_capability": None,
            "vram_bytes": 0,
            "free_vram_bytes": 0,
        }
    properties = torch.cuda.get_device_properties(0)
    free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
    return {
        "detector_status": "detected",
        "device": "cuda:0",
        "name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "vram_bytes": int(properties.total_memory),
        "free_vram_bytes": int(free_bytes),
    }


@dataclass(frozen=True, slots=True)
class QualifiedCudaArchitectureBArtifact:
    artifact_path: Path
    manifest_path: Path
    qualification_receipt_path: Path
    reporting_receipt_path: Path
    artifact_sha256: str
    manifest_sha256: str
    qualification_receipt_sha256: str
    reporting_receipt_sha256: str
    module: ModuleType
    capability: dict[str, Any]
    resources: dict[str, Any]
    hardware: dict[str, Any]
    promotion_registry_sha256: str

    def default_authorization(self) -> dict[str, Any]:
        return {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "automatic_default_authorized": True,
            "authorization_source": AUTHORIZATION_SOURCE,
            "authorization_decision_id": AUTHORIZATION_DECISION_ID,
            "promotion_registry_digest": self.promotion_registry_sha256,
            "artifact_bindings": {
                "artifact_sha256": self.artifact_sha256,
                "manifest_sha256": self.manifest_sha256,
                "qualification_receipt_sha256": self.qualification_receipt_sha256,
                "reporting_receipt_sha256": self.reporting_receipt_sha256,
            },
            "selection_basis": "exactness",
            "performance_claimed": False,
            "extension_self_promotion_claimed": False,
        }

    def extension_contract(self) -> dict[str, Any]:
        return {
            "available": True,
            "artifact_sha256": self.artifact_sha256,
            "manifest_sha256": self.manifest_sha256,
            "qualification_receipt_sha256": self.qualification_receipt_sha256,
            "reporting_receipt_sha256": self.reporting_receipt_sha256,
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
                "patch_p2_qualification": dict(
                    self.capability["patch_p2_qualification"]
                ),
            },
        }

    def routing_hardware(self) -> dict[str, Any]:
        return {
            name: self.hardware[name]
            for name in (
                "detector_status",
                "device",
                "name",
                "compute_capability",
                "vram_bytes",
            )
        }


@dataclass(frozen=True, slots=True)
class CudaArchitectureBProductionExecutionV2:
    """One selected native trajectory awaiting caller-owned settlement."""

    run_id: str
    owner_epoch: int
    run_control: Any
    direct_result: dict[str, Any]
    run_control_manifest: dict[str, Any]
    completed_generations: int
    production_route_receipt: dict[str, Any]


def _validate_registry(registry: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(registry)
    _require_equal(
        payload.get("schema_version"),
        REGISTRY_SCHEMA_VERSION,
        field="qualification registry schema_version",
    )
    _require_equal(
        payload.get("status"),
        "exact_qualified_not_promoted",
        field="qualification registry status",
    )
    _require_equal(
        payload.get("production_promotion"),
        False,
        field="qualification registry production_promotion",
    )
    _require_equal(
        payload.get("required_entrypoints"),
        list(REQUIRED_ENTRYPOINTS),
        field="qualification registry required_entrypoints",
    )
    _require_equal(
        payload.get("required_attributes"),
        list(REQUIRED_ATTRIBUTES),
        field="qualification registry required_attributes",
    )
    _require_equal(
        payload.get("qualification_decision"),
        QUALIFICATION_DECISION,
        field="qualification registry decision",
    )
    _require_equal(
        payload.get("required_claims"),
        list(REQUIRED_QUALIFICATION_CLAIMS),
        field="qualification registry required_claims",
    )
    envelope = _mapping(
        payload.get("envelope"), field="qualification registry envelope"
    )
    _require_admitted_patch_geometry(
        population=envelope.get("population"),
        grid=envelope.get("grid"),
        field="qualification registry envelope",
    )
    expected_envelope = {
        "world": "patch",
        "profile": "P2",
        "policy_family": "stochastic_fixed_mlp",
        "generation_count": 4,
        "max_births_per_tick": 64,
        "batch_cells": 1,
        "ticks_per_generation": 80,
        "physical_window_generations": 2,
        "checkpoint_schedule": "terminal_only",
        "output_schema": QUALIFIED_OUTPUT_SCHEMA,
    }
    for name, expected in expected_envelope.items():
        _require_equal(
            envelope.get(name), expected, field=f"qualification registry envelope {name}"
        )
    for name in (
        "artifact_sha256",
        "manifest_sha256",
        "qualification_receipt_sha256",
        "reporting_receipt_sha256",
    ):
        value = payload.get(name)
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise _fail(f"qualification registry {name} is not a lowercase SHA-256")
    return payload


def _validate_promotion_registry(
    registry: Mapping[str, Any],
    *,
    qualification_registry_path: Path,
    qualification_registry_sha256: str,
    artifact_bindings: Mapping[str, str],
) -> dict[str, Any]:
    payload = dict(registry)
    _require_equal(
        set(payload),
        {
            "schema_version",
            "status",
            "authority",
            "authorization_decision_id",
            "automatic_backend",
            "unsupported_auto_backend",
            "selection_basis",
            "performance_claimed",
            "extension_self_promotion_required",
            "post_dispatch_fallback_allowed",
            "envelope_id",
            "qualification_registry",
            "artifact_bindings",
            "evidence",
        },
        field="promotion registry fields",
    )
    expected = {
        "schema_version": PROMOTION_REGISTRY_SCHEMA_VERSION,
        "status": "automatic_default_authorized",
        "authority": AUTHORIZATION_SOURCE,
        "authorization_decision_id": AUTHORIZATION_DECISION_ID,
        "automatic_backend": CUDA_BACKEND,
        "unsupported_auto_backend": "legacy_python",
        "selection_basis": "exactness",
        "performance_claimed": False,
        "extension_self_promotion_required": False,
        "post_dispatch_fallback_allowed": False,
        "envelope_id": QUALIFIED_ENVELOPE_ID,
    }
    for name, value in expected.items():
        _require_equal(payload.get(name), value, field=f"promotion registry {name}")

    qualification = _mapping(
        payload.get("qualification_registry"),
        field="promotion qualification registry",
    )
    qualification_path = Path(str(qualification.get("path", ""))).expanduser()
    if not qualification_path.is_absolute():
        qualification_path = PROJECT_ROOT / qualification_path
    _require_equal(
        qualification_path.resolve(),
        qualification_registry_path.resolve(),
        field="promotion qualification registry path",
    )
    _require_equal(
        qualification.get("sha256"),
        qualification_registry_sha256,
        field="promotion qualification registry SHA-256",
    )

    bindings = _mapping(
        payload.get("artifact_bindings"), field="promotion artifact bindings"
    )
    _require_equal(
        bindings,
        dict(artifact_bindings),
        field="promotion artifact bindings",
    )

    evidence = _mapping(payload.get("evidence"), field="promotion evidence")
    expected_evidence = {
        "b64_exact_parity",
        "completion",
        "cancellation_checkpoint",
        "failure_settlement",
        "integrated_owner_completion",
    }
    _require_equal(
        set(evidence), expected_evidence, field="promotion evidence inventory"
    )
    for name in sorted(expected_evidence):
        reference = _mapping(
            evidence[name], field=f"promotion evidence {name}"
        )
        path = Path(str(reference.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise _fail(f"promotion evidence {name} is missing: {path}")
        _require_equal(
            _sha256(path),
            reference.get("sha256"),
            field=f"promotion evidence {name} SHA-256",
        )
        evidence_payload = _load_json(path, field=f"promotion evidence {name}")
        _require_equal(
            evidence_payload.get("ok"), True, field=f"promotion evidence {name} ok"
        )
        terminal = evidence_payload.get("terminal_result")
        claims = evidence_payload.get("claims")
        if name == "b64_exact_parity":
            _require_equal(
                evidence_payload.get("decision"),
                QUALIFICATION_DECISION,
                field="promotion B64 parity decision",
            )
            _require_equal(
                _mapping(claims, field="promotion B64 parity claims").get(
                    "terminal_authoritative_and_canonical_checkpoint_exact"
                ),
                True,
                field="promotion B64 terminal checkpoint parity",
            )
        elif name == "completion":
            _require_equal(
                _mapping(terminal, field="promotion completion terminal").get(
                    "terminal_status"
                ),
                "run_complete",
                field="promotion completion terminal status",
            )
        elif name == "cancellation_checkpoint":
            cancellation_claims = _mapping(
                claims, field="promotion cancellation claims"
            )
            for claim in (
                "checkpoint_on_cancel_exact",
                "exact_cancellation_at_g2_boundary",
                "fresh_owner_continuation_after_cancel_exact",
            ):
                _require_equal(
                    cancellation_claims.get(claim),
                    True,
                    field=f"promotion cancellation claim {claim}",
                )
            _require_equal(
                _mapping(terminal, field="promotion cancellation terminal").get(
                    "terminal_status"
                ),
                "cancelled",
                field="promotion cancellation terminal status",
            )
        elif name == "failure_settlement":
            _require_equal(
                _mapping(terminal, field="promotion failure terminal").get(
                    "terminal_status"
                ),
                "failed",
                field="promotion failure terminal status",
            )
        else:
            _require_equal(
                evidence_payload.get("terminal_status"),
                "run_complete",
                field="promotion integrated owner terminal status",
            )
            _require_equal(
                evidence_payload.get("hidden_fallback"),
                False,
                field="promotion integrated owner hidden fallback",
            )
            _require_equal(
                evidence_payload.get("python_generation_dispatches"),
                0,
                field="promotion integrated owner Python dispatches",
            )
    return payload


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    artifact_path: Path,
    artifact_sha256: str,
) -> None:
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "module_name": MODULE_NAME,
        "artifact": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "source_snapshot_stable": True,
        "immutable_output": True,
        "runtime_jit_required_by_validation_route": False,
        "runtime_jit_compilation": False,
        "sass_only": True,
        "cuda_arch": "sm_75",
    }
    for name, value in expected.items():
        _require_equal(manifest.get(name), value, field=f"CUDA manifest {name}")
    entrypoints = manifest.get("entrypoints")
    if not isinstance(entrypoints, list):
        raise _fail("CUDA manifest entrypoints must be a list")
    missing_entrypoints = [
        name for name in REQUIRED_ENTRYPOINTS if name not in entrypoints
    ]
    if missing_entrypoints:
        raise _fail(
            "CUDA manifest entrypoints missing required entries: "
            f"{missing_entrypoints}"
        )
    required_attributes = manifest.get("required_attributes")
    if not isinstance(required_attributes, list):
        raise _fail("CUDA manifest required_attributes must be a list")
    missing_attributes = [
        name for name in REQUIRED_ATTRIBUTES if name not in required_attributes
    ]
    if missing_attributes:
        raise _fail(
            "CUDA manifest required_attributes missing required entries: "
            f"{missing_attributes}"
        )


def _validate_qualification_receipt(
    receipt: Mapping[str, Any],
    *,
    artifact_sha256: str,
    manifest_sha256: str,
) -> None:
    _require_equal(receipt.get("ok"), True, field="qualification receipt ok")
    _require_equal(
        receipt.get("decision"),
        QUALIFICATION_DECISION,
        field="qualification receipt decision",
    )
    artifacts = _mapping(receipt.get("artifacts"), field="qualification artifacts")
    _require_equal(
        artifacts.get("cuda_extension_sha256"),
        artifact_sha256,
        field="qualification artifact SHA-256",
    )
    _require_equal(
        artifacts.get("cuda_manifest_sha256"),
        manifest_sha256,
        field="qualification manifest SHA-256",
    )
    claims = _mapping(receipt.get("claims"), field="qualification claims")
    for claim in REQUIRED_QUALIFICATION_CLAIMS:
        _require_equal(claims.get(claim), True, field=f"qualification claim {claim}")
    configuration = _mapping(
        receipt.get("configuration"), field="qualification configuration"
    )
    expected = {
        "world": "patch",
        "profile": "P2",
        "batch_cells": 1,
        "total_generations": 4,
        "ticks_per_generation": 80,
        "max_births_per_tick": 64,
        "resident_window_generations": 2,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
    }
    for name, value in expected.items():
        _require_equal(
            configuration.get(name), value, field=f"qualification configuration {name}"
        )


def _validate_reporting_receipt(
    receipt: Mapping[str, Any],
    *,
    artifact_sha256: str,
    manifest_sha256: str,
    qualification_receipt_sha256: str,
) -> None:
    _require_equal(receipt.get("ok"), True, field="reporting receipt ok")
    _require_equal(
        receipt.get("decision"),
        "CUDA-ARCH-B-PHASE1-FAIR-COMPACT/MEASURED",
        field="reporting receipt decision",
    )
    artifacts = _mapping(receipt.get("artifacts"), field="reporting artifacts")
    expected_artifacts = {
        "cuda_extension_sha256": artifact_sha256,
        "cuda_manifest_sha256": manifest_sha256,
        "qualification_receipt_sha256": qualification_receipt_sha256,
        "runtime_jit_allowed": False,
    }
    for name, expected in expected_artifacts.items():
        _require_equal(
            artifacts.get(name), expected, field=f"reporting artifact {name}"
        )
    configuration = _mapping(
        receipt.get("configuration"), field="reporting configuration"
    )
    _require_admitted_patch_geometry(
        population=configuration.get("population"),
        grid=(configuration.get("grid_size"), configuration.get("grid_size")),
        field="reporting configuration",
    )
    expected_configuration = {
        "world": "patch",
        "patch_profile": "P2",
        "generation_count": 4,
        "batch_cells": 1,
        "ticks_per_generation": 80,
        "physical_window_generations": 2,
        "checkpoint_generations": [4],
    }
    for name, expected in expected_configuration.items():
        _require_equal(
            configuration.get(name), expected, field=f"reporting configuration {name}"
        )
    fairness = _mapping(
        receipt.get("fairness_verdict"), field="reporting fairness verdict"
    )
    expected_fairness = {
        "ok": True,
        "logical_output_inventory_exact": True,
        "primary_native_speedup_eligible": False,
        "durable_storage_speedup_eligible": False,
        "first_mismatch": None,
    }
    for name, expected in expected_fairness.items():
        _require_equal(fairness.get(name), expected, field=f"reporting fairness {name}")
    claims = _mapping(receipt.get("claims"), field="reporting claims")
    expected_claims = {
        "production_routing_promoted": False,
        "runtime_jit_used": False,
        "fair_compact_native_speedup_eligible": False,
    }
    for name, expected in expected_claims.items():
        _require_equal(claims.get(name), expected, field=f"reporting claim {name}")
    output_inventory = _mapping(
        receipt.get("output_inventory"), field="reporting output inventory"
    )
    _require_equal(
        output_inventory.get("terminal_checkpoint_count"),
        1,
        field="reporting terminal checkpoint count",
    )
    _require_equal(
        output_inventory.get("compact_summary_rows"),
        4,
        field="reporting compact summary rows",
    )


def _validate_live_capability(value: Any) -> dict[str, Any]:
    capability = _mapping(value, field="live CUDA capability")
    _require_equal(
        capability.get("schema_version"),
        QUALIFIED_EXTENSION_VERSION,
        field="live CUDA capability schema_version",
    )
    _require_equal(
        capability.get("production_routing_enabled"),
        False,
        field="live CUDA capability production_routing_enabled",
    )
    _require_equal(
        capability.get("qualification_metadata_only"),
        True,
        field="live CUDA capability qualification_metadata_only",
    )
    patch = _mapping(
        capability.get("patch_p2_qualification"),
        field="live CUDA Patch P2 capability",
    )
    expected_patch = {
        "qualified": True,
        "world": "patch",
        "profile": "P2",
        "ticks_per_generation": 80,
        "batch_cells": 1,
        "policy": "stochastic_fixed_mlp",
        "generation_turnover": True,
        "mechanically_supported_max_births_per_tick": 64,
        "supported_reproduction_mode": "asexual",
    }
    for name, expected in expected_patch.items():
        _require_equal(patch.get(name), expected, field=f"Patch P2 capability {name}")
    for population, grid in QUALIFIED_PATCH_GEOMETRIES:
        if not _capability_supports_patch_geometry(
            patch,
            population=population,
            grid_size=grid[0],
        ):
            raise _fail(
                "Patch P2 capability does not support every admitted geometry"
            )
    _require_equal(
        patch.get("supported_lifecycle_modes"),
        [0, 1],
        field="Patch P2 capability supported_lifecycle_modes",
    )
    _require_equal(
        capability.get("generation_count_supported"),
        True,
        field="live CUDA generation-count support",
    )
    supported = patch.get("supported_generation_counts")
    if (
        isinstance(supported, (str, bytes))
        or not isinstance(supported, Sequence)
        or 2 not in supported
    ):
        raise _fail(
            "live CUDA capability does not qualify the physical g2 windows "
            "required for logical g4"
        )
    _require_equal(
        capability.get("eiel_compact_generation_summary_v1_supported"),
        True,
        field="live CUDA compact summary support",
    )
    _require_equal(
        capability.get("compact_summary_max_physical_batch_cells"),
        1,
        field="live CUDA compact summary batch limit",
    )
    return capability


def _validate_hardware(
    value: Mapping[str, Any], *, minimum_free_vram_bytes: int
) -> dict[str, Any]:
    hardware = dict(value)
    expected = {
        "detector_status": "detected",
        "device": QUALIFIED_DEVICE,
        "name": QUALIFIED_HARDWARE_NAME,
        "compute_capability": list(QUALIFIED_COMPUTE_CAPABILITY),
    }
    for name, expected_value in expected.items():
        _require_equal(
            hardware.get(name), expected_value, field=f"CUDA hardware {name}"
        )
    vram = hardware.get("vram_bytes")
    if type(vram) is not int or not 6_000_000_000 <= vram <= 6 * 1024**3:
        raise _fail("CUDA hardware VRAM is outside the qualified GTX 1660 Ti range")
    free_vram = hardware.get("free_vram_bytes")
    if type(free_vram) is not int or free_vram < minimum_free_vram_bytes:
        raise _fail("CUDA hardware lacks the qualified free-VRAM headroom")
    return hardware


def _load_qualified_cuda_artifact(
    artifact_path: str | Path,
    qualification_receipt_path: str | Path,
    reporting_receipt_path: str | Path,
    *,
    registry_path: str | Path,
    promotion_registry_path: str | Path,
    module_loader: Callable[[Path, str], ModuleType],
    hardware_detector: Callable[[], Mapping[str, Any]],
) -> QualifiedCudaArchitectureBArtifact:
    registry_target = Path(registry_path).expanduser().resolve()
    registry = _validate_registry(
        _load_json(registry_target, field="CUDA qualification registry")
    )
    registry_sha256 = _sha256(registry_target)
    artifact = Path(artifact_path).expanduser().resolve()
    manifest = artifact.with_suffix(f"{artifact.suffix}.manifest.json")
    qualification = Path(qualification_receipt_path).expanduser().resolve()
    reporting = Path(reporting_receipt_path).expanduser().resolve()
    for path, field in (
        (artifact, "CUDA artifact"),
        (manifest, "CUDA manifest"),
        (qualification, "CUDA qualification receipt"),
        (reporting, "CUDA reporting receipt"),
    ):
        if not path.is_file():
            raise _fail(f"{field} is missing: {path}")
    artifact_sha256 = _sha256(artifact)
    manifest_sha256 = _sha256(manifest)
    qualification_sha256 = _sha256(qualification)
    reporting_sha256 = _sha256(reporting)
    for name, actual in (
        ("artifact_sha256", artifact_sha256),
        ("manifest_sha256", manifest_sha256),
        ("qualification_receipt_sha256", qualification_sha256),
        ("reporting_receipt_sha256", reporting_sha256),
    ):
        _require_equal(actual, registry[name], field=f"qualified {name}")

    promotion_target = Path(promotion_registry_path).expanduser().resolve()
    if not promotion_target.is_file():
        raise _fail(f"CUDA promotion registry is missing: {promotion_target}")
    promotion_sha256 = _sha256(promotion_target)
    _validate_promotion_registry(
        _load_json(promotion_target, field="CUDA promotion registry"),
        qualification_registry_path=registry_target,
        qualification_registry_sha256=registry_sha256,
        artifact_bindings={
            "artifact_sha256": artifact_sha256,
            "manifest_sha256": manifest_sha256,
            "qualification_receipt_sha256": qualification_sha256,
            "reporting_receipt_sha256": reporting_sha256,
        },
    )

    manifest_payload = _load_json(manifest, field="CUDA manifest")
    _validate_manifest(
        manifest_payload,
        artifact_path=artifact,
        artifact_sha256=artifact_sha256,
    )
    qualification_payload = _load_json(
        qualification, field="CUDA qualification receipt"
    )
    _validate_qualification_receipt(
        qualification_payload,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
    )
    reporting_payload = _load_json(reporting, field="CUDA reporting receipt")
    _validate_reporting_receipt(
        reporting_payload,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        qualification_receipt_sha256=qualification_sha256,
    )
    module = module_loader(artifact, MODULE_NAME)
    missing_entrypoints = [
        name
        for name in REQUIRED_ENTRYPOINTS
        if not callable(getattr(module, name, None))
    ]
    if missing_entrypoints:
        raise _fail(
            f"qualified CUDA extension is missing entrypoints: {missing_entrypoints}"
        )
    missing_attributes = [
        name
        for name in REQUIRED_ATTRIBUTES
        if not callable(getattr(module, name, None))
    ]
    if missing_attributes:
        raise _fail(
            "qualified CUDA extension is missing required attributes: "
            f"{missing_attributes}"
        )
    capability = _validate_live_capability(getattr(module, CAPABILITY_ENTRYPOINT)())
    resources = _mapping(
        getattr(module, RESOURCE_ENTRYPOINT)(), field="live CUDA resource receipt"
    )
    hardware = _validate_hardware(
        hardware_detector(),
        minimum_free_vram_bytes=int(registry["minimum_free_vram_bytes"]),
    )
    return QualifiedCudaArchitectureBArtifact(
        artifact_path=artifact,
        manifest_path=manifest,
        qualification_receipt_path=qualification,
        reporting_receipt_path=reporting,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        qualification_receipt_sha256=qualification_sha256,
        reporting_receipt_sha256=reporting_sha256,
        module=module,
        capability=capability,
        resources=resources,
        hardware=hardware,
        promotion_registry_sha256=promotion_sha256,
    )


def load_qualified_cuda_artifact(
    artifact_path: str | Path,
    qualification_receipt_path: str | Path,
    reporting_receipt_path: str | Path,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    promotion_registry_path: str | Path = DEFAULT_PROMOTION_REGISTRY_PATH,
) -> QualifiedCudaArchitectureBArtifact:
    """Load the one tracked exact artifact envelope; no caller claims are trusted."""

    return _load_qualified_cuda_artifact(
        artifact_path,
        qualification_receipt_path,
        reporting_receipt_path,
        registry_path=registry_path,
        promotion_registry_path=promotion_registry_path,
        module_loader=_default_module_loader,
        hardware_detector=_default_hardware_detector,
    )


class CudaArchitectureBProductionAdapter:
    """Resolve once, then own one no-fallback direct CUDA dispatch."""

    def __init__(self, artifact: QualifiedCudaArchitectureBArtifact) -> None:
        self._artifact = artifact
        self._route_receipt: dict[str, Any] | None = None
        self._dispatch_started = False
        self._run_control: Any | None = None
        self._run_control_ready = threading.Event()
        self._dispatch_finished = threading.Event()

    @property
    def dispatch_started(self) -> bool:
        return self._dispatch_started

    def request_cancel(
        self,
        *,
        request_id: str,
        reason: str = "operator_requested",
        timeout_seconds: float = 5.0,
    ) -> dict[str, Any]:
        """Latch cancellation through the native post-tick-0 control window."""

        if (
            type(request_id) is not str
            or not request_id
            or request_id.strip() != request_id
            or not request_id.isascii()
            or not request_id.isprintable()
        ):
            raise _fail("cancellation request_id must be a stable ASCII identifier")
        if reason not in {
            "operator_requested",
            "signal_received",
            "supervisor_requested",
        }:
            raise _fail("CUDA cancellation reason is unsupported")
        if (
            type(timeout_seconds) not in {float, int}
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(float(timeout_seconds))
            or float(timeout_seconds) <= 0.0
        ):
            raise _fail("cancellation timeout_seconds must be finite and positive")

        deadline = time.monotonic() + float(timeout_seconds)
        remaining = deadline - time.monotonic()
        if not self._run_control_ready.wait(max(0.0, remaining)):
            raise _fail("CUDA run control was not created before cancellation timeout")
        control = self._run_control
        if control is None:
            raise _fail("CUDA dispatch ended before a cancellable run control existed")

        last_receipt: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            manifest = _mapping(
                control.manifest(), field="active CUDA run-control manifest"
            )
            receipt = _mapping(
                control.request_cancel(
                    {
                        "api_version": 1,
                        "request_id": request_id,
                        "expected_controller_id": manifest.get("controller_id"),
                        "expected_owner_epoch": manifest.get("owner_epoch"),
                        "reason": reason,
                    }
                ),
                field="native CUDA cancellation receipt",
            )
            last_receipt = receipt
            disposition = receipt.get("disposition")
            if disposition in {"accepted", "already_requested", "already_terminal"}:
                return receipt
            if disposition not in {"not_running", "control_point_unavailable"}:
                raise _fail(
                    "native CUDA cancellation failed closed: "
                    f"disposition={disposition!r}"
                )
            if self._dispatch_finished.is_set():
                break
            time.sleep(0.001)
        raise _fail(
            "native CUDA cancellation was not accepted before timeout: "
            f"last_receipt={last_receipt!r}"
        )

    def resolve_once(
        self,
        *,
        request_mode: str,
        allow_cpp_cpu_fallback: bool,
        workload: Mapping[str, Any],
        allow_explicit_cuda_exactness_override: bool = False,
        unsupported_mechanic_flags: Mapping[str, bool] | None = None,
        performance_decision: str = "unmeasured",
        performance_decision_measured: bool = False,
    ) -> dict[str, Any]:
        if self._route_receipt is not None:
            raise _fail("CUDA production route was already resolved")
        if not isinstance(workload, Mapping):
            raise _fail("CUDA production workload must be derived from the live runner")
        request = {
            "schema_version": REQUEST_SCHEMA_VERSION,
            "request_mode": request_mode,
            "allow_cpp_cpu_fallback": allow_cpp_cpu_fallback,
            "allow_explicit_cuda_exactness_override": (
                allow_explicit_cuda_exactness_override
            ),
            "default_authorization": self._artifact.default_authorization(),
            "dispatch_started": False,
            "hardware": self._artifact.routing_hardware(),
            "workload": dict(workload),
            "extension": self._artifact.extension_contract(),
            "unsupported_mechanic_flags": dict(unsupported_mechanic_flags or {}),
            "performance_decision": performance_decision,
            "performance_decision_measured": performance_decision_measured,
        }
        receipt = resolve_cuda_architecture_b_production_route(request)
        receipt["contract_status"] = "artifact_bound_adapter_resolved"
        receipt["routing_integrated"] = True
        self._route_receipt = receipt
        return dict(receipt)

    def execute_prepared(
        self,
        *,
        run_id: str,
        owner_epoch: int = 1,
        physical_window_arguments: Sequence[Any],
        policy_mutation_enable: Any,
        policy_mutation_mode: Any,
        policy_mutation_rate: Any,
        policy_mutation_std: Any,
    ) -> CudaArchitectureBProductionExecutionV2:
        if self._route_receipt is None:
            raise _fail("CUDA production route must be resolved before dispatch")
        if self._route_receipt.get("resolved_backend") != CUDA_BACKEND:
            backend = self._route_receipt.get("resolved_backend")
            if backend == CPP_CPU_BACKEND:
                raise _fail("pre-dispatch route selected C++ CPU, not CUDA")
            raise _fail("pre-dispatch route rejected CUDA")
        if self._dispatch_started:
            raise _fail("CUDA production adapter is single-use")
        try:
            run_control = create_run_control_v2(
                self._artifact.module,
                run_id=run_id,
                owner_epoch=owner_epoch,
            )
        except BaseException as exc:
            self._run_control_ready.set()
            self._dispatch_finished.set()
            request_mode = str(self._route_receipt.get("request_mode", "auto"))
            rejection = native_engine_pre_dispatch_rejection_receipt(
                reason=(
                    "run_control_construction_failed:"
                    f"{type(exc).__name__}:{exc}"
                ),
                request_mode=request_mode,
            )
            self._route_receipt = dict(rejection)
            raise CudaArchitectureBProductionDispatchError(
                "native CUDA run-control construction failed before dispatch: "
                f"{type(exc).__name__}: {exc}",
                run_id=run_id,
                production_route_receipt=rejection,
                native_failure_receipt=None,
                run_control_manifest=None,
                native_terminal_result=None,
                failure_settlement_error=None,
            ) from exc

        self._dispatch_started = True
        self._run_control = run_control
        self._run_control_ready.set()
        live_chronicle = NativeRunControlLivePublisher(run_control)
        live_chronicle.start()
        try:
            execution: DirectResidentNativeExecutionV2 = (
                execute_direct_resident_windows_v2(
                    self._artifact.module,
                    run_control=run_control,
                    physical_window_arguments=physical_window_arguments,
                    logical_generation_count=4,
                    policy_mutation_enable=policy_mutation_enable,
                    policy_mutation_mode=policy_mutation_mode,
                    policy_mutation_rate=policy_mutation_rate,
                    policy_mutation_std=policy_mutation_std,
                )
            )
        except BaseException as exc:
            try:
                live_chronicle.stop(final_manifest=run_control.manifest())
            except Exception:
                pass
            failure_receipt: dict[str, Any] | None = None
            failure_manifest: dict[str, Any] | None = None
            terminal_result: dict[str, Any] | None = None
            failure_settlement_error: dict[str, Any] | None = None
            try:
                manifest = _mapping(
                    run_control.manifest(),
                    field="failed CUDA run-control manifest",
                )
                boundary = manifest.get("last_boundary")
                frontier = (
                    boundary.get("trajectory_commit_frontier", 0)
                    if isinstance(boundary, Mapping)
                    else 0
                )
                failure_receipt = _mapping(
                    run_control.publish_failure(
                        "CUDA_DISPATCH_FAILED",
                        f"{type(exc).__name__}: {exc}",
                        int(manifest.get("active_generation", -1)),
                        int(frontier),
                    ),
                    field="native CUDA dispatch failure receipt",
                )
                failure_manifest = _mapping(
                    run_control.manifest(),
                    field="settled CUDA run-control manifest",
                )
                terminal_raw = run_control.final_result()
                terminal_result = _mapping(
                    terminal_raw,
                    field="settled CUDA terminal failure result",
                )
                if (
                    failure_receipt.get("terminal_status") != "failed"
                    or failure_manifest.get("status") != "failed"
                    or failure_manifest.get("sealed") is not True
                    or terminal_result.get("terminal_status") != "failed"
                    or terminal_result.get("sealed") is not True
                ):
                    raise _fail(
                        "native CUDA dispatch failure did not seal exact failed state"
                    )
            except BaseException as settlement_exc:
                failure_settlement_error = {
                    "error_type": type(settlement_exc).__name__,
                    "error": str(settlement_exc),
                }
            self._run_control_ready.set()
            self._dispatch_finished.set()
            route_receipt = {
                **self._route_receipt,
                "dispatch_started": True,
                "fallback_performed": False,
                "fallback_phase": "none",
            }
            raise CudaArchitectureBProductionDispatchError(
                "CUDA dispatch failed after backend selection; fallback is prohibited: "
                f"{type(exc).__name__}: {exc}; "
                f"native_failure_receipt={failure_receipt!r}",
                run_id=run_id,
                production_route_receipt=route_receipt,
                native_failure_receipt=failure_receipt,
                run_control_manifest=failure_manifest,
                native_terminal_result=terminal_result,
                failure_settlement_error=failure_settlement_error,
            ) from exc
        try:
            live_chronicle.stop(final_manifest=run_control.manifest())
        except Exception:
            pass
        self._dispatch_finished.set()
        route_receipt = {
            **self._route_receipt,
            "dispatch_started": True,
            "fallback_performed": False,
            "fallback_phase": "none",
        }
        return CudaArchitectureBProductionExecutionV2(
            run_id=run_id,
            owner_epoch=owner_epoch,
            run_control=execution.run_control,
            direct_result=dict(execution.direct_result),
            run_control_manifest=dict(execution.manifest),
            completed_generations=execution.completed_generations,
            production_route_receipt=route_receipt,
        )


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "REGISTRY_SCHEMA_VERSION",
    "REQUIRED_ATTRIBUTES",
    "REQUIRED_ENTRYPOINTS",
    "CudaArchitectureBProductionAdapter",
    "CudaArchitectureBProductionAdapterError",
    "CudaArchitectureBProductionDispatchError",
    "CudaArchitectureBProductionExecutionV2",
    "QualifiedCudaArchitectureBArtifact",
    "load_qualified_cuda_artifact",
]
