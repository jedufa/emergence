from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from engine.native_runtime import (
    cuda_architecture_b_production_adapter as adapter_module,
)
from engine.native_runtime import cuda_architecture_b_routing as routing_module
from engine.native_runtime.cuda_architecture_b_production_adapter import (
    CAPABILITY_ENTRYPOINT,
    CONTROLLER_ENTRYPOINT,
    MANIFEST_SCHEMA_VERSION,
    MODULE_NAME,
    QUALIFICATION_DECISION,
    PROMOTION_REGISTRY_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    REQUIRED_ATTRIBUTES,
    REQUIRED_ENTRYPOINTS,
    REQUIRED_QUALIFICATION_CLAIMS,
    RESOURCE_ENTRYPOINT,
    CudaArchitectureBProductionAdapter,
    CudaArchitectureBProductionAdapterError,
    CudaArchitectureBProductionDispatchError,
    CudaArchitectureBProductionExecutionV2,
    QualifiedCudaArchitectureBArtifact,
    _load_qualified_cuda_artifact,
)
from engine.native_runtime.cuda_architecture_b_routing import (
    AUTHORIZATION_DECISION_ID,
    AUTHORIZATION_SOURCE,
    CUDA_BACKEND,
    QUALIFIED_COMPUTE_CAPABILITY,
    QUALIFIED_DEVICE,
    QUALIFIED_ENVELOPE_ID,
    QUALIFIED_EXTENSION_ABI_VERSION,
    QUALIFIED_EXTENSION_VERSION,
    QUALIFIED_HARDWARE_NAME,
    QUALIFIED_OUTPUT_SCHEMA,
)
from engine.perf.cuda_architecture_b_direct_resident_execution_v2 import (
    DIRECT_ENTRYPOINT_V2,
    RUN_CONTROL_TYPE_V2,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    DIRECT_ARGUMENT_COUNT,
    DIRECT_ENTRYPOINT,
    GENERATION_COUNT_ARGUMENT_INDEX,
    PHYSICAL_WINDOW_GENERATIONS,
    RESULT_SCHEMA_VERSION,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _capability(**patch_overrides: object) -> dict[str, Any]:
    patch = {
        "qualified": True,
        "world": "patch",
        "profile": "P2",
        "population": 64,
        "grid_size": 16,
        "supported_geometry_envelopes": [
            {"population": 64, "grid_size": 16},
            {"population": 256, "grid_size": 32},
            {"population": 512, "grid_size": 32},
        ],
        "ticks_per_generation": 80,
        "batch_cells": 1,
        "policy": "stochastic_fixed_mlp",
        "generation_turnover": True,
        "mechanically_supported_max_births_per_tick": 64,
        "supported_reproduction_mode": "asexual",
        "supported_lifecycle_modes": [0, 1],
        "supported_generation_counts": [1, 2, 8, 32, 64, 128],
    }
    patch.update(patch_overrides)
    return {
        "schema_version": QUALIFIED_EXTENSION_VERSION,
        "generation_count_supported": True,
        "production_routing_enabled": False,
        "qualification_metadata_only": True,
        "patch_p2_qualification": patch,
        "eiel_compact_generation_summary_v1_supported": True,
        "compact_summary_max_physical_batch_cells": 1,
    }


def _hardware(*, free_vram_bytes: int = 2_000_000_000) -> dict[str, Any]:
    return {
        "detector_status": "detected",
        "device": QUALIFIED_DEVICE,
        "name": QUALIFIED_HARDWARE_NAME,
        "compute_capability": list(QUALIFIED_COMPUTE_CAPABILITY),
        "vram_bytes": 6_024_855_552,
        "free_vram_bytes": free_vram_bytes,
    }


def _qualified_workload(*, max_births_per_tick: int = 64) -> dict[str, Any]:
    return {
        "world": "patch",
        "profile": "P2",
        "population": 64,
        "grid": [16, 16],
        "policy_family": "stochastic_fixed_mlp",
        "generation_count": 4,
        "max_births_per_tick": max_births_per_tick,
        "batch_cells": 1,
        "checkpoint_schedule": "terminal_only",
        "output_schema": QUALIFIED_OUTPUT_SCHEMA,
    }


class _FakeRunControl:
    def __init__(self, run_id: str, owner_epoch: int) -> None:
        self.state = {
            "run_id": run_id,
            "controller_id": f"controller:{owner_epoch}:{run_id}",
            "owner_epoch": owner_epoch,
            "status": "initialized",
            "run_started": False,
            "sealed": False,
            "active_generation": -1,
            "requested_generations": 0,
            "target_next_generation": -1,
            "cancel_requested": False,
            "awaiting_cancellation_checkpoint": False,
            "awaiting_final_output_ack": False,
        }
        self.cancel_requests: list[dict[str, Any]] = []
        self.terminal: dict[str, Any] | None = None

    def request_cancel(self, request: Any) -> dict[str, Any]:
        request = dict(request)
        self.cancel_requests.append(request)
        disposition = "control_point_unavailable"
        if self.state["sealed"]:
            disposition = "already_terminal"
        elif not self.state["run_started"]:
            disposition = "not_running"
        elif self.state["cancel_requested"]:
            disposition = "already_requested"
        elif self.state["status"] == "inter_tick_control":
            disposition = "accepted"
            self.state["cancel_requested"] = True
        return {
            "request_id": request["request_id"],
            "disposition": disposition,
            "first_cancel_request_id": (
                request["request_id"] if self.state["cancel_requested"] else None
            ),
            "cancel_sequence": 1 if self.state["cancel_requested"] else None,
            "terminal_status_changed": False,
        }

    def acknowledge_cancellation_checkpoint(self, *_args: Any) -> None: ...

    def acknowledge_run_complete(self, *_args: Any) -> None: ...

    def publish_failure(self, *args: Any) -> dict[str, Any]:
        self.state.update(
            {
                "status": "failed",
                "sealed": True,
                "failure": list(args),
            }
        )
        self.terminal = {
            "terminal_status": "failed",
            "sealed": True,
            "success_marker_allowed": False,
            "promotion_result_allowed": False,
        }
        return {
            "disposition": "accepted",
            "terminal_status": "failed",
            "terminal_status_changed_to_failed": True,
        }

    def final_result(self) -> dict[str, Any] | None:
        return self.terminal

    def manifest(self) -> dict[str, Any]:
        return dict(self.state)


def _direct_result(control: _FakeRunControl) -> dict[str, Any]:
    control.state.update(
        {
            "status": "generation_complete",
            "run_started": True,
            "active_generation": 3,
            "requested_generations": 4,
            "target_next_generation": 4,
            "awaiting_final_output_ack": True,
        }
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "owner": "native_extension",
        "physical_window_generations": 2,
        "logical_generation_count": 4,
        "window_count": 2,
        "completed_generations": 4,
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "native_run_control_v2": control.manifest(),
        "native_run_control_v2_trajectory_owner": True,
        "internal_pybind_trajectory_entry_trampoline": True,
        "python_tick_or_generation_ownership": False,
    }


class _FakeModule(SimpleNamespace):
    def __init__(
        self,
        *,
        capability: dict[str, Any] | None = None,
        direct_v2: Callable[..., object] | None = None,
        include_direct_v1: bool = True,
        include_direct_v2: bool = True,
        include_run_control_v2: bool = True,
    ) -> None:
        super().__init__()
        self.direct_calls: list[tuple[object, ...]] = []
        self.run_controls: list[_FakeRunControl] = []
        setattr(self, CONTROLLER_ENTRYPOINT, lambda *args: {})
        setattr(self, RESOURCE_ENTRYPOINT, lambda: {"owner": "native_extension"})
        setattr(self, CAPABILITY_ENTRYPOINT, lambda: capability or _capability())
        if include_direct_v1:
            setattr(self, DIRECT_ENTRYPOINT, lambda *args: {})
        if include_run_control_v2:

            def create_control(run_id: str, owner_epoch: int) -> _FakeRunControl:
                control = _FakeRunControl(run_id, owner_epoch)
                self.run_controls.append(control)
                return control

            setattr(self, RUN_CONTROL_TYPE_V2, create_control)
        if include_direct_v2:
            handler = direct_v2 or (lambda control, *_args: _direct_result(control))

            def invoke(control: object, *args: object) -> object:
                self.direct_calls.append((control, *args))
                return handler(control, *args)

            setattr(self, DIRECT_ENTRYPOINT_V2, invoke)


@dataclass(frozen=True)
class _ArtifactFiles:
    artifact: Path
    manifest: Path
    qualification: Path
    reporting: Path
    registry: Path
    promotion_registry: Path
    artifact_sha256: str
    manifest_sha256: str
    qualification_sha256: str
    reporting_sha256: str


def _artifact_files(
    tmp_path: Path, *, minimum_free_vram_bytes: int = 1_000
) -> _ArtifactFiles:
    artifact = tmp_path / "qualified_controller.so"
    artifact.write_bytes(b"temporary qualified CUDA extension")
    artifact_sha256 = _sha256(artifact)

    manifest = artifact.with_suffix(f"{artifact.suffix}.manifest.json")
    _write_json(
        manifest,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "module_name": MODULE_NAME,
            "artifact": str(artifact.resolve()),
            "artifact_sha256": artifact_sha256,
            "source_snapshot_stable": True,
            "immutable_output": True,
            "runtime_jit_required_by_validation_route": False,
            "runtime_jit_compilation": False,
            "sass_only": True,
            "cuda_arch": "sm_75",
            "entrypoints": list(REQUIRED_ENTRYPOINTS),
            "required_attributes": list(REQUIRED_ATTRIBUTES),
        },
    )
    manifest_sha256 = _sha256(manifest)

    qualification = tmp_path / "qualification.json"
    _write_json(
        qualification,
        {
            "ok": True,
            "decision": QUALIFICATION_DECISION,
            "artifacts": {
                "cuda_extension_sha256": artifact_sha256,
                "cuda_manifest_sha256": manifest_sha256,
            },
            "claims": {name: True for name in REQUIRED_QUALIFICATION_CLAIMS},
            "configuration": {
                "world": "patch",
                "profile": "P2",
                "batch_cells": 1,
                "total_generations": 4,
                "ticks_per_generation": 80,
                "max_births_per_tick": 64,
                "resident_window_generations": 2,
                "direct_cpp_window_core_dispatch": True,
                "production_native_chunk_loop": True,
            },
        },
    )
    qualification_sha256 = _sha256(qualification)

    reporting = tmp_path / "reporting.json"
    _write_json(
        reporting,
        {
            "ok": True,
            "decision": "CUDA-ARCH-B-PHASE1-FAIR-COMPACT/MEASURED",
            "artifacts": {
                "cuda_extension_sha256": artifact_sha256,
                "cuda_manifest_sha256": manifest_sha256,
                "qualification_receipt_sha256": qualification_sha256,
                "runtime_jit_allowed": False,
            },
            "configuration": {
                "world": "patch",
                "patch_profile": "P2",
                "population": 64,
                "grid_size": 16,
                "generation_count": 4,
                "batch_cells": 1,
                "ticks_per_generation": 80,
                "physical_window_generations": 2,
                "checkpoint_generations": [4],
            },
            "fairness_verdict": {
                "ok": True,
                "logical_output_inventory_exact": True,
                "primary_native_speedup_eligible": False,
                "durable_storage_speedup_eligible": False,
                "first_mismatch": None,
            },
            "claims": {
                "production_routing_promoted": False,
                "runtime_jit_used": False,
                "fair_compact_native_speedup_eligible": False,
            },
            "output_inventory": {
                "terminal_checkpoint_count": 1,
                "compact_summary_rows": 4,
            },
        },
    )
    reporting_sha256 = _sha256(reporting)

    registry = tmp_path / "qualification_registry.json"
    _write_json(
        registry,
        {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "status": "exact_qualified_not_promoted",
            "production_promotion": False,
            "required_entrypoints": list(REQUIRED_ENTRYPOINTS),
            "required_attributes": list(REQUIRED_ATTRIBUTES),
            "qualification_decision": QUALIFICATION_DECISION,
            "required_claims": list(REQUIRED_QUALIFICATION_CLAIMS),
            "envelope": {
                "world": "patch",
                "profile": "P2",
                "population": 64,
                "grid": [16, 16],
                "policy_family": "stochastic_fixed_mlp",
                "generation_count": 4,
                "max_births_per_tick": 64,
                "batch_cells": 1,
                "ticks_per_generation": 80,
                "physical_window_generations": 2,
                "checkpoint_schedule": "terminal_only",
                "output_schema": QUALIFIED_OUTPUT_SCHEMA,
            },
            "artifact_sha256": artifact_sha256,
            "manifest_sha256": manifest_sha256,
            "qualification_receipt_sha256": qualification_sha256,
            "reporting_receipt_sha256": reporting_sha256,
            "minimum_free_vram_bytes": minimum_free_vram_bytes,
        },
    )
    evidence: dict[str, dict[str, str]] = {}
    evidence_payloads = {
        "b64_exact_parity": {
            "ok": True,
            "decision": QUALIFICATION_DECISION,
            "claims": {
                "terminal_authoritative_and_canonical_checkpoint_exact": True
            },
        },
        "completion": {
            "ok": True,
            "terminal_result": {"terminal_status": "run_complete"},
        },
        "cancellation_checkpoint": {
            "ok": True,
            "claims": {
                "checkpoint_on_cancel_exact": True,
                "exact_cancellation_at_g2_boundary": True,
                "fresh_owner_continuation_after_cancel_exact": True,
            },
            "terminal_result": {"terminal_status": "cancelled"},
        },
        "failure_settlement": {
            "ok": True,
            "terminal_result": {"terminal_status": "failed"},
        },
        "integrated_owner_completion": {
            "ok": True,
            "terminal_status": "run_complete",
            "hidden_fallback": False,
            "python_generation_dispatches": 0,
        },
    }
    for name, payload in evidence_payloads.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, payload)
        evidence[name] = {"path": str(path), "sha256": _sha256(path)}
    promotion_registry = tmp_path / "promotion_registry.json"
    _write_json(
        promotion_registry,
        {
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
            "qualification_registry": {
                "path": str(registry),
                "sha256": _sha256(registry),
            },
            "artifact_bindings": {
                "artifact_sha256": artifact_sha256,
                "manifest_sha256": manifest_sha256,
                "qualification_receipt_sha256": qualification_sha256,
                "reporting_receipt_sha256": reporting_sha256,
            },
            "evidence": evidence,
        },
    )
    return _ArtifactFiles(
        artifact=artifact,
        manifest=manifest,
        qualification=qualification,
        reporting=reporting,
        registry=registry,
        promotion_registry=promotion_registry,
        artifact_sha256=artifact_sha256,
        manifest_sha256=manifest_sha256,
        qualification_sha256=qualification_sha256,
        reporting_sha256=reporting_sha256,
    )


def _bind_test_hashes(
    monkeypatch: pytest.MonkeyPatch,
    files: _ArtifactFiles,
) -> None:
    monkeypatch.setattr(
        routing_module, "QUALIFIED_ARTIFACT_SHA256", files.artifact_sha256
    )
    monkeypatch.setattr(
        routing_module, "QUALIFIED_MANIFEST_SHA256", files.manifest_sha256
    )
    monkeypatch.setattr(
        routing_module, "QUALIFIED_RECEIPT_SHA256", files.qualification_sha256
    )
    monkeypatch.setattr(
        routing_module, "QUALIFIED_REPORTING_RECEIPT_SHA256", files.reporting_sha256
    )


def _load(
    files: _ArtifactFiles,
    module: _FakeModule,
    *,
    hardware: dict[str, Any] | None = None,
) -> QualifiedCudaArchitectureBArtifact:
    return _load_qualified_cuda_artifact(
        files.artifact,
        files.qualification,
        files.reporting,
        registry_path=files.registry,
        promotion_registry_path=files.promotion_registry,
        module_loader=lambda _path, _name: module,
        hardware_detector=lambda: hardware or _hardware(),
    )


def _dispatch(
    adapter: CudaArchitectureBProductionAdapter,
) -> CudaArchitectureBProductionExecutionV2:
    physical_window_arguments = [object() for _index in range(DIRECT_ARGUMENT_COUNT)]
    physical_window_arguments[GENERATION_COUNT_ARGUMENT_INDEX] = (
        PHYSICAL_WINDOW_GENERATIONS
    )
    return adapter.execute_prepared(
        run_id="production-run-seed-101",
        owner_epoch=7,
        physical_window_arguments=physical_window_arguments,
        policy_mutation_enable=True,
        policy_mutation_mode="gaussian",
        policy_mutation_rate=0.2,
        policy_mutation_std=0.05,
    )


def test_valid_artifact_binding_derives_extension_contract_and_hardware(
    tmp_path: Path,
) -> None:
    files = _artifact_files(tmp_path)
    artifact = _load(files, _FakeModule())

    assert artifact.artifact_sha256 == files.artifact_sha256
    assert artifact.manifest_sha256 == files.manifest_sha256
    assert artifact.qualification_receipt_sha256 == files.qualification_sha256
    assert artifact.reporting_receipt_sha256 == files.reporting_sha256
    assert artifact.promotion_registry_sha256 == _sha256(files.promotion_registry)
    assert artifact.default_authorization()["automatic_default_authorized"] is True
    assert artifact.default_authorization()["artifact_bindings"] == {
        "artifact_sha256": files.artifact_sha256,
        "manifest_sha256": files.manifest_sha256,
        "qualification_receipt_sha256": files.qualification_sha256,
        "reporting_receipt_sha256": files.reporting_sha256,
    }
    assert artifact.extension_contract() == {
        "available": True,
        "artifact_sha256": files.artifact_sha256,
        "manifest_sha256": files.manifest_sha256,
        "qualification_receipt_sha256": files.qualification_sha256,
        "reporting_receipt_sha256": files.reporting_sha256,
        "abi_version": QUALIFIED_EXTENSION_ABI_VERSION,
        "extension_version": QUALIFIED_EXTENSION_VERSION,
        "capability": {
            "schema_version": routing_module.CAPABILITY_SCHEMA_VERSION,
            "envelope_id": QUALIFIED_ENVELOPE_ID,
            "qualified": True,
            "runtime_jit": False,
            "hidden_fallback": False,
            "production_promotion": False,
            "entrypoints_verified": True,
            "patch_p2_qualification": _capability()["patch_p2_qualification"],
        },
    }
    assert artifact.routing_hardware() == {
        key: _hardware()[key]
        for key in (
            "detector_status",
            "device",
            "name",
            "compute_capability",
            "vram_bytes",
        )
    }


@pytest.mark.parametrize(
    "drifted", ("artifact", "manifest", "qualification", "reporting")
)
def test_hash_drift_rejects_before_module_loader(
    tmp_path: Path,
    drifted: str,
) -> None:
    files = _artifact_files(tmp_path)
    getattr(files, drifted).write_bytes(getattr(files, drifted).read_bytes() + b"\n")
    loader_calls: list[tuple[Path, str]] = []

    def loader(path: Path, name: str) -> _FakeModule:
        loader_calls.append((path, name))
        return _FakeModule()

    with pytest.raises(
        CudaArchitectureBProductionAdapterError, match="sha256 mismatch"
    ):
        _load_qualified_cuda_artifact(
            files.artifact,
            files.qualification,
            files.reporting,
            registry_path=files.registry,
            promotion_registry_path=files.promotion_registry,
            module_loader=loader,
            hardware_detector=_hardware,
        )

    assert loader_calls == []


def test_promotion_registry_policy_tampering_rejects_before_module_load(
    tmp_path: Path,
) -> None:
    files = _artifact_files(tmp_path)
    payload = json.loads(files.promotion_registry.read_text(encoding="ascii"))
    payload["performance_claimed"] = True
    _write_json(files.promotion_registry, payload)

    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="promotion registry performance_claimed mismatch",
    ):
        _load(files, _FakeModule())


def test_bound_cancellation_evidence_hash_drift_rejects_before_module_load(
    tmp_path: Path,
) -> None:
    files = _artifact_files(tmp_path)
    payload = json.loads(files.promotion_registry.read_text(encoding="ascii"))
    evidence_path = Path(payload["evidence"]["cancellation_checkpoint"]["path"])
    evidence_path.write_bytes(evidence_path.read_bytes() + b"\n")

    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="cancellation_checkpoint SHA-256 mismatch",
    ):
        _load(files, _FakeModule())


def test_missing_v2_direct_entrypoint_rejects(tmp_path: Path) -> None:
    files = _artifact_files(tmp_path)

    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="missing entrypoints",
    ):
        _load(files, _FakeModule(include_direct_v2=False))


def test_missing_v2_run_control_type_rejects(tmp_path: Path) -> None:
    files = _artifact_files(tmp_path)

    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="missing required attributes.*RunControlV2",
    ):
        _load(files, _FakeModule(include_run_control_v2=False))


def test_manifest_and_registry_require_v2_control_surface(tmp_path: Path) -> None:
    files = _artifact_files(tmp_path)
    manifest = json.loads(files.manifest.read_text(encoding="utf-8"))
    registry = json.loads(files.registry.read_text(encoding="utf-8"))

    manifest.pop("required_attributes")
    registry.pop("required_attributes")

    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="CUDA manifest required_attributes must be a list",
    ):
        adapter_module._validate_manifest(
            manifest,
            artifact_path=files.artifact.resolve(),
            artifact_sha256=files.artifact_sha256,
        )
    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="qualification registry required_attributes mismatch",
    ):
        adapter_module._validate_registry(registry)


def test_manifest_may_expose_additional_entrypoints_and_attributes(
    tmp_path: Path,
) -> None:
    files = _artifact_files(tmp_path)
    manifest = json.loads(files.manifest.read_text(encoding="utf-8"))
    manifest["entrypoints"].append("future_authoritative_state_entrypoint_v1")
    manifest["required_attributes"].append("FutureRunControlV3")

    adapter_module._validate_manifest(
        manifest,
        artifact_path=files.artifact.resolve(),
        artifact_sha256=files.artifact_sha256,
    )


def test_live_capability_mismatch_rejects(tmp_path: Path) -> None:
    files = _artifact_files(tmp_path)

    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="does not support every admitted geometry",
    ):
        _load(
            files,
            _FakeModule(
                capability=_capability(
                    supported_geometry_envelopes=[
                        {"population": 64, "grid_size": 16},
                    ],
                )
            ),
        )


def test_missing_physical_g2_window_capability_rejects(tmp_path: Path) -> None:
    files = _artifact_files(tmp_path)

    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="physical g2 windows required for logical g4",
    ):
        _load(
            files,
            _FakeModule(capability=_capability(supported_generation_counts=[1, 8, 32])),
        )


def test_free_vram_shortfall_rejects(tmp_path: Path) -> None:
    files = _artifact_files(tmp_path, minimum_free_vram_bytes=1_000)

    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="lacks the qualified free-VRAM headroom",
    ):
        _load(files, _FakeModule(), hardware=_hardware(free_vram_bytes=999))


def test_promoted_auto_unmeasured_selects_exact_cuda_without_speedup_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _artifact_files(tmp_path)
    _bind_test_hashes(monkeypatch, files)
    module = _FakeModule()
    adapter = CudaArchitectureBProductionAdapter(_load(files, module))

    receipt = adapter.resolve_once(
        request_mode="auto",
        allow_cpp_cpu_fallback=True,
        workload=_qualified_workload(),
    )

    assert receipt["resolved_backend"] == CUDA_BACKEND
    assert receipt["reason"] == "external_registry_authorized_exact_cuda_default"
    assert receipt["production_promotion_claimed"] is False
    assert receipt["automatic_default_authorized"] is True
    assert adapter.dispatch_started is False
    assert module.direct_calls == []


def test_explicit_unmeasured_exactness_override_selects_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _artifact_files(tmp_path)
    _bind_test_hashes(monkeypatch, files)
    adapter = CudaArchitectureBProductionAdapter(_load(files, _FakeModule()))

    receipt = adapter.resolve_once(
        request_mode="explicit_cuda",
        allow_cpp_cpu_fallback=False,
        workload=_qualified_workload(),
        allow_explicit_cuda_exactness_override=True,
    )

    assert receipt["resolved_backend"] == CUDA_BACKEND
    assert receipt["reason"] == (
        "explicit_cuda_external_registry_authorized_exact_envelope"
    )
    assert receipt["production_promotion_claimed"] is False
    assert receipt["automatic_default_authorized"] is True


def test_exact_prepared_dispatch_uses_one_fresh_v2_control_and_returns_typed_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _artifact_files(tmp_path)
    _bind_test_hashes(monkeypatch, files)
    module = _FakeModule()
    adapter = CudaArchitectureBProductionAdapter(_load(files, module))
    adapter.resolve_once(
        request_mode="explicit_cuda",
        allow_cpp_cpu_fallback=False,
        workload=_qualified_workload(),
        allow_explicit_cuda_exactness_override=True,
    )

    result = _dispatch(adapter)

    assert isinstance(result, CudaArchitectureBProductionExecutionV2)
    assert result.run_id == "production-run-seed-101"
    assert result.owner_epoch == 7
    assert result.completed_generations == 4
    assert result.direct_result["native_run_control_v2_trajectory_owner"] is True
    assert result.run_control_manifest["awaiting_final_output_ack"] is True
    assert module.run_controls == [result.run_control]
    assert result.run_control_manifest["run_id"] == result.run_id
    assert result.run_control_manifest["owner_epoch"] == result.owner_epoch
    assert len(module.direct_calls) == 1
    assert module.direct_calls[0][0] is result.run_control
    assert len(module.direct_calls[0]) == DIRECT_ARGUMENT_COUNT + 6
    assert adapter.dispatch_started is True
    route = result.production_route_receipt
    assert route["resolved_backend"] == CUDA_BACKEND
    assert route["contract_status"] == "artifact_bound_adapter_resolved"
    assert route["routing_integrated"] is True
    assert route["dispatch_started"] is True
    assert route["fallback_performed"] is False
    assert route["fallback_phase"] == "none"


def test_active_dispatch_accepts_boundary_cancellation_without_python_step_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _artifact_files(tmp_path)
    _bind_test_hashes(monkeypatch, files)
    control_window = threading.Event()
    release_dispatch = threading.Event()

    def cancellable(control: _FakeRunControl, *_args: object) -> object:
        control.state.update(
            {
                "status": "inter_tick_control",
                "run_started": True,
                "active_generation": 1,
                "requested_generations": 4,
                "target_next_generation": 4,
            }
        )
        control_window.set()
        assert release_dispatch.wait(2.0)
        result = _direct_result(control)
        control.state.update(
            {
                "active_generation": 1,
                "cancel_requested": True,
                "awaiting_cancellation_checkpoint": True,
                "awaiting_final_output_ack": False,
            }
        )
        result["completed_generations"] = 2
        result["native_run_control_v2"] = control.manifest()
        return result

    module = _FakeModule(direct_v2=cancellable)
    adapter = CudaArchitectureBProductionAdapter(_load(files, module))
    adapter.resolve_once(
        request_mode="explicit_cuda",
        allow_cpp_cpu_fallback=False,
        workload=_qualified_workload(),
        allow_explicit_cuda_exactness_override=True,
    )
    result_box: dict[str, Any] = {}

    def execute() -> None:
        result_box["execution"] = _dispatch(adapter)

    worker = threading.Thread(target=execute)
    worker.start()
    assert control_window.wait(2.0)
    receipt = adapter.request_cancel(
        request_id="operator-cancel-1",
        reason="operator_requested",
        timeout_seconds=1.0,
    )
    release_dispatch.set()
    worker.join(2.0)

    assert not worker.is_alive()
    assert receipt["disposition"] == "accepted"
    assert receipt["first_cancel_request_id"] == "operator-cancel-1"
    assert result_box["execution"].completed_generations == 2
    assert module.run_controls[0].cancel_requests == [
        {
            "api_version": 1,
            "request_id": "operator-cancel-1",
            "expected_controller_id": module.run_controls[0].state[
                "controller_id"
            ],
            "expected_owner_epoch": 7,
            "reason": "operator_requested",
        }
    ]


def test_post_selection_exception_is_terminal_without_fallback_or_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _artifact_files(tmp_path)
    _bind_test_hashes(monkeypatch, files)

    def fail(*_args: object) -> object:
        raise RuntimeError("device failure")

    module = _FakeModule(direct_v2=fail)
    adapter = CudaArchitectureBProductionAdapter(_load(files, module))
    adapter.resolve_once(
        request_mode="explicit_cuda",
        allow_cpp_cpu_fallback=False,
        workload=_qualified_workload(),
        allow_explicit_cuda_exactness_override=True,
    )

    with pytest.raises(
        CudaArchitectureBProductionDispatchError,
        match="fallback is prohibited.*device failure",
    ) as raised:
        _dispatch(adapter)
    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="single-use",
    ):
        _dispatch(adapter)

    assert adapter.dispatch_started is True
    assert len(module.direct_calls) == 1
    assert len(module.run_controls) == 1
    failed_manifest = module.run_controls[0].manifest()
    assert failed_manifest["run_started"] is False
    assert failed_manifest["status"] == "failed"
    assert failed_manifest["sealed"] is True
    assert failed_manifest["failure"][0] == "CUDA_DISPATCH_FAILED"
    assert raised.value.native_failure_receipt == {
        "disposition": "accepted",
        "terminal_status": "failed",
        "terminal_status_changed_to_failed": True,
    }
    assert raised.value.run_control_manifest["status"] == "failed"
    assert raised.value.native_terminal_result["terminal_status"] == "failed"
    assert raised.value.failure_settlement_error is None


def test_resolve_and_successful_dispatch_are_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = _artifact_files(tmp_path)
    _bind_test_hashes(monkeypatch, files)
    module = _FakeModule()
    adapter = CudaArchitectureBProductionAdapter(_load(files, module))
    adapter.resolve_once(
        request_mode="explicit_cuda",
        allow_cpp_cpu_fallback=False,
        workload=_qualified_workload(),
        allow_explicit_cuda_exactness_override=True,
    )

    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="route was already resolved",
    ):
        adapter.resolve_once(
            request_mode="auto",
            allow_cpp_cpu_fallback=True,
            workload=_qualified_workload(),
        )

    _dispatch(adapter)
    with pytest.raises(
        CudaArchitectureBProductionAdapterError,
        match="single-use",
    ):
        _dispatch(adapter)

    assert len(module.direct_calls) == 1


def test_module_constants_cover_all_required_runtime_entrypoints() -> None:
    assert REQUIRED_ENTRYPOINTS == (
        CONTROLLER_ENTRYPOINT,
        RESOURCE_ENTRYPOINT,
        CAPABILITY_ENTRYPOINT,
        DIRECT_ENTRYPOINT,
        DIRECT_ENTRYPOINT_V2,
    )
    assert REQUIRED_ATTRIBUTES == (RUN_CONTROL_TYPE_V2,)
    assert adapter_module.MODULE_NAME == MODULE_NAME
