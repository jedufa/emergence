from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import native_runtime.run_controller as run_controller_module
from native_runtime.patch_profile import PatchProfileClassification
from native_runtime.run_controller import (
    NATIVE_BOUNDARY_SCHEMA_VERSION,
    NATIVE_CONTROLLER_VERSION,
    NATIVE_INTER_TICK_CONTROL_CAPABILITY,
    NATIVE_PATCH_CAPABILITIES_ATTRIBUTE,
    NATIVE_PATCH_INSTRUMENTATION_RECEIPT_SCHEMA_VERSION,
    NATIVE_PATCH_INSTRUMENTATION_RESULT_FIELD,
    NATIVE_PATCH_INSTRUMENTATION_SCHEMA_VERSION,
    NATIVE_PHASE4_CAPABILITIES_ATTRIBUTE,
    NATIVE_RESULT_SCHEMA_VERSION,
    NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION,
    NativeRunController,
    NativeRunControllerError,
)


RUN_ID = "patch-instrumentation-run"
CONTROLLER_ID = "patch-instrumentation-owner"
OWNER_EPOCH = 17
GENERATION_COUNT = 2
SOURCE_DIGEST = "b" * 64


def _runner(root: Path, *, patch: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        run_dir=root / RUN_ID,
        sim_cfg=SimpleNamespace(
            chronicle_run_id=RUN_ID,
            steps_per_generation=2,
            native_runtime_max_births_per_generation=1,
        ),
        env_cfg=SimpleNamespace(
            patch_world_enable=patch,
            tier4_juvenile_vulnerability_enable=False,
            tier4_juvenile_steps=0,
        ),
    )


def _patch_metadata() -> dict[str, Any]:
    return {
        "world": "patch",
        "native_runtime_patch_profile": "P1",
        "native_runtime_patch_contract_schema_version": (
            "eiel_patch_native_import_contract_v1"
        ),
        "device": "cpu",
        "bank_device": "cpu",
        "native_runtime_allow_runtime_jit_build": False,
        "capability_mask": {
            "flat": False,
            "patch": True,
            "churn": True,
        },
    }


def _import_projection() -> SimpleNamespace:
    return SimpleNamespace(
        tensors={"import_state": torch.arange(4, dtype=torch.int64)},
        metadata=_patch_metadata(),
        pending_generation_free_agent_ids=[],
        pending_generation_free_slots=[],
    )


def _projection(values: list[int]) -> dict[str, Any]:
    tensor = torch.tensor(values, dtype=torch.int64)
    byte_count = tensor.numel() * tensor.element_size()
    return {
        "tensors": {"state": tensor},
        "tensor_manifest": {"state": {"bytes": byte_count}},
    }


def _boundary(offset: int) -> dict[str, Any]:
    reporting = _projection([offset, offset + 1])
    checkpoint = _projection([offset + 10, offset + 11, offset + 12])
    reporting_bytes = 2 * torch.tensor([], dtype=torch.int64).element_size()
    checkpoint_bytes = 3 * torch.tensor([], dtype=torch.int64).element_size()
    return {
        "schema_version": NATIVE_BOUNDARY_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "controller_id": CONTROLLER_ID,
        "owner_epoch": OWNER_EPOCH,
        "snapshot_id": (
            f"{CONTROLLER_ID}:owner:{OWNER_EPOCH}:boundary:{offset + 1}"
        ),
        "snapshot_sequence": offset + 1,
        "completed_generation": offset,
        "next_generation": offset + 1,
        "tick": 0,
        "phase": "generation_complete",
        "controller_status_at_clone": (
            "generation_complete_ready_to_resolve_terminal"
        ),
        "generation_result": {"executed": True, "hidden_fallback": False},
        "generation_end_reporting_frame": reporting,
        "generation_end_reporting_frame_role": (
            "transaction_local_generation_end_values_folded_after_turnover_commit"
        ),
        "turnover_result": {"executed": True, "hidden_fallback": False},
        "post_turnover_checkpoint_projection": checkpoint,
        "generation_end_reporting_frame_tensor_bytes": reporting_bytes,
        "post_turnover_projection_tensor_bytes": checkpoint_bytes,
    }


def _native_result(
    instrumentation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    boundaries = [_boundary(offset) for offset in range(GENERATION_COUNT)]
    projection_bytes = sum(
        boundary["generation_end_reporting_frame_tensor_bytes"]
        + boundary["post_turnover_projection_tensor_bytes"]
        for boundary in boundaries
    )
    result = {
        "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
        "decision": "ARCHB-PATCH-CONTROLLED-DEVELOPMENT-RUN",
        "run_id": RUN_ID,
        "controller_id": CONTROLLER_ID,
        "owner_epoch": OWNER_EPOCH,
        "executed": True,
        "native_owner": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "terminal_status": "awaiting_snapshot_validation",
        "start_generation": 0,
        "requested_generations": GENERATION_COUNT,
        "completed_generations": GENERATION_COUNT,
        "trajectory_commit_frontier": GENERATION_COUNT - 1,
        "snapshot_frontier": -1,
        "next_generation": GENERATION_COUNT,
        "next_tick": 0,
        "controller_invocations": 1,
        "python_generation_dispatches": 0,
        "native_generation_transactions": GENERATION_COUNT,
        "turnover_commits": GENERATION_COUNT,
        "boundary_snapshot_count": GENERATION_COUNT,
        "projection_export_count": GENERATION_COUNT * 2,
        "observer_live_owner_exports": 0,
        "persistence_acknowledgement_count": 0,
        "projection_tensor_bytes": projection_bytes,
        "boundary_wall_time_ns": 997,
        "boundaries": boundaries,
        "final_snapshot_sequence": GENERATION_COUNT,
        "success_marker_allowed": False,
        "promotion_result_allowed": False,
    }
    if instrumentation is not None:
        result[NATIVE_PATCH_INSTRUMENTATION_RESULT_FIELD] = dict(instrumentation)
    return result


class _NativeOwner:
    def __init__(
        self,
        tensors: Mapping[str, Any],
        metadata: Mapping[str, Any],
        run_id: str,
        instrumentation: Mapping[str, Any] | None,
    ) -> None:
        self.tensors = tensors
        self.metadata = metadata
        self.run_id = run_id
        self.instrumentation = instrumentation
        self.status = "initialized"
        self.run_started = False
        self.sealed = False

    def controller_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "eiel_native_architecture_b_controller_manifest_v1",
            "run_id": self.run_id,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "status": self.status,
            "run_started": self.run_started,
            "sealed": self.sealed,
            "native_owner_private": True,
            "python_callback_entry_points": 0,
        }

    def run_generations(
        self,
        start_generation: int,
        generation_count: int,
        steps_per_generation: int,
        max_births: int,
    ) -> dict[str, Any]:
        assert (start_generation, generation_count) == (0, GENERATION_COUNT)
        assert (steps_per_generation, max_births) == (2, 1)
        self.status = "awaiting_snapshot_validation"
        self.run_started = True
        return _native_result(self.instrumentation)

    def acknowledge_snapshot_validation(
        self,
        source_projection_digest: str,
    ) -> dict[str, Any]:
        self.status = "run_complete"
        self.sealed = True
        return {
            "schema_version": NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "acknowledged": True,
            "disposition": "accepted",
            "terminal_status": "run_complete",
            "trajectory_commit_frontier": GENERATION_COUNT - 1,
            "snapshot_frontier": GENERATION_COUNT - 1,
            "final_snapshot_sequence": GENERATION_COUNT,
            "validated_source_projection_digest": source_projection_digest,
            "success_marker_allowed": True,
        }

    def fail_snapshot_validation(self, message: str) -> dict[str, Any]:
        raise AssertionError(f"unexpected snapshot validation failure: {message}")

    def final_result(self) -> dict[str, Any]:
        return {"terminal": False, "status": self.status}


def _module(controller_type: Any) -> SimpleNamespace:
    return SimpleNamespace(
        EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_VERSION=NATIVE_CONTROLLER_VERSION,
        FlatCpuArchitectureBControllerV1=controller_type,
        **{
            NATIVE_PHASE4_CAPABILITIES_ATTRIBUTE: {
                NATIVE_INTER_TICK_CONTROL_CAPABILITY: True,
            },
            NATIVE_PATCH_CAPABILITIES_ATTRIBUTE: {
                "controlled_architecture_b_cpu_v1": True,
                "resident_action_cache_cpu_v1": True,
                "resident_population_birth_commit_cpu_v1": True,
                "exact_parity_qualified": False,
                "promotion_accepted": False,
                "jungle_supported": True,
                "wdl_supported": True,
                "gru_supported": False,
                "cuda_supported": False,
            },
        },
    )


def _install_patch_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    native_instrumentation: list[Mapping[str, Any] | None],
) -> tuple[SimpleNamespace, Path, str, list[_NativeOwner]]:
    extension_path = tmp_path / "patch-instrumentation.so"
    extension_payload = b"patch instrumentation extension fixture"
    extension_path.write_bytes(extension_payload)
    extension_sha256 = hashlib.sha256(extension_payload).hexdigest()
    runner = _runner(tmp_path)
    instances: list[_NativeOwner] = []
    payloads = list(native_instrumentation)

    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        lambda *_args, **_kwargs: PatchProfileClassification(
            supported=True,
            profile="P1",
        ),
    )
    monkeypatch.setattr(
        run_controller_module,
        "validate_native_runtime_supported",
        lambda value: pytest.fail("Patch instrumentation reached Flat qualification"),
    )
    monkeypatch.setattr(
        run_controller_module,
        "_projection_digest_manifests",
        lambda projection: {"source_projection": {"digest": SOURCE_DIGEST}},
    )

    import perf.cxxbackbone10_resident_state_owner as resident_owner_module

    monkeypatch.setattr(
        resident_owner_module,
        "build_import_projection",
        lambda *_args, **_kwargs: _import_projection(),
    )

    def controller_type(
        tensors: Mapping[str, Any],
        metadata: Mapping[str, Any],
        run_id: str,
    ) -> _NativeOwner:
        owner = _NativeOwner(
            tensors,
            metadata,
            run_id,
            payloads[len(instances)],
        )
        instances.append(owner)
        return owner

    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: _module(controller_type),
    )
    return runner, extension_path, extension_sha256, instances


def _controller(
    runner: SimpleNamespace,
    extension_path: Path,
    extension_sha256: str,
    *,
    instrumentation_enabled: bool,
) -> NativeRunController:
    return NativeRunController.from_runner(
        runner,
        extension_path=extension_path,
        expected_extension_sha256=extension_sha256,
        patch_instrumentation_enabled=instrumentation_enabled,
    )


def _assert_exact(left: Any, right: Any, *, path: str = "root") -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        assert isinstance(left, torch.Tensor), path
        assert isinstance(right, torch.Tensor), path
        assert left.dtype == right.dtype, path
        assert tuple(left.shape) == tuple(right.shape), path
        assert torch.equal(left, right), path
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        assert isinstance(left, Mapping), path
        assert isinstance(right, Mapping), path
        assert set(left) == set(right), path
        for key in left:
            _assert_exact(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, list) or isinstance(right, list):
        assert isinstance(left, list), path
        assert isinstance(right, list), path
        assert len(left) == len(right), path
        for index, (left_item, right_item) in enumerate(
            zip(left, right, strict=True)
        ):
            _assert_exact(left_item, right_item, path=f"{path}[{index}]")
        return
    assert left == right, path
    assert type(left) is type(right), path


def _without_instrumentation_receipt(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key != "architecture_b_patch_instrumentation_receipt"
    }


def _native_timings() -> dict[str, Any]:
    return {
        "schema_version": NATIVE_PATCH_INSTRUMENTATION_SCHEMA_VERSION,
        "host_serialization_bytes": 101,
        "host_projection_time_ns": 102,
        "device_transfer_time_ns": 0,
        "device_synchronization_time_ns": 0,
        "allocation_count": 7,
        "allocation_bytes": 103,
        "simulation_time_ns": 104,
        "observation_time_ns": 105,
        "policy_time_ns": 106,
        "world_patch_time_ns": 107,
        "lifecycle_time_ns": 108,
        "boundary_settlement_time_ns": 109,
        "checkpoint_projection_time_ns": 110,
        "reporting_projection_time_ns": 111,
    }


def test_patch_instrumentation_receipt_is_stable_and_trajectory_observational(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_timings = _native_timings()
    runner, extension_path, extension_sha256, instances = _install_patch_factory(
        tmp_path,
        monkeypatch,
        [None, native_timings],
    )
    disabled = _controller(
        runner,
        extension_path,
        extension_sha256,
        instrumentation_enabled=False,
    )
    enabled = _controller(
        runner,
        extension_path,
        extension_sha256,
        instrumentation_enabled=True,
    )

    disabled_result = disabled.run_many(GENERATION_COUNT)
    enabled_result = enabled.run_many(GENERATION_COUNT)

    assert len(instances) == 2
    disabled_receipt = disabled_result[
        "architecture_b_patch_instrumentation_receipt"
    ]
    enabled_receipt = enabled_result["architecture_b_patch_instrumentation_receipt"]
    assert disabled_receipt["schema_version"] == (
        NATIVE_PATCH_INSTRUMENTATION_RECEIPT_SCHEMA_VERSION
    )
    assert disabled_receipt["enabled"] is False
    assert disabled_receipt["status"] == "disabled"
    assert disabled_receipt["observational_only"] is True
    assert disabled_receipt["performance_threshold_applied"] is False
    assert set(disabled_receipt["measurements"]) == set(
        enabled_receipt["measurements"]
    )
    assert set(disabled_receipt["measurement_units"]) == set(
        disabled_receipt["measurements"]
    )
    for field, value in disabled_receipt["measurements"].items():
        if field in {
            "device_transfer_time_ns",
            "device_synchronization_time_ns",
        }:
            assert value == 0
        else:
            assert value is None

    measurements = enabled_receipt["measurements"]
    assert enabled_receipt["enabled"] is True
    assert enabled_receipt["status"] == "enabled"
    assert enabled_receipt["device_transfer_applicability"] == (
        "not_applicable_cpu"
    )
    assert measurements["initial_import_bytes"] > 32
    assert enabled_receipt["initial_import_breakdown"] == {
        "unique_tensor_storage_bytes": 32,
        "unique_tensor_storage_count": 1,
    }
    assert measurements["generation_boundary_projection_bytes"] == 80
    assert measurements["checkpoint_bytes"] == 48
    for field, expected in native_timings.items():
        if field != "schema_version":
            assert measurements[field] == expected
            assert enabled_receipt["measurement_sources"][field] == (
                "native_instrumentation"
            )
    assert enabled_receipt["native_fields_present"] == sorted(
        field for field in native_timings if field != "schema_version"
    )
    assert NATIVE_PATCH_INSTRUMENTATION_RESULT_FIELD not in enabled_result
    assert enabled.manifest()["instrumentation_enabled"] is True

    _assert_exact(
        _without_instrumentation_receipt(enabled_result),
        _without_instrumentation_receipt(disabled_result),
    )


def test_enabled_patch_instrumentation_keeps_unreported_fields_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, extension_path, extension_sha256, _ = _install_patch_factory(
        tmp_path,
        monkeypatch,
        [None],
    )
    controller = _controller(
        runner,
        extension_path,
        extension_sha256,
        instrumentation_enabled=True,
    )

    result = controller.run_many(GENERATION_COUNT)
    receipt = result["architecture_b_patch_instrumentation_receipt"]
    measurements = receipt["measurements"]

    assert receipt["native_fields_present"] == []
    assert measurements["controller_boundary_wall_time_ns"] == 997
    assert receipt["measurement_sources"]["controller_boundary_wall_time_ns"] == (
        "validated_native_result"
    )
    assert measurements["simulation_time_ns"] is None
    assert measurements["host_serialization_bytes"] is None
    assert receipt["measurement_sources"]["host_serialization_bytes"] == (
        "not_reported"
    )
    assert measurements["device_transfer_time_ns"] == 0
    assert measurements["device_synchronization_time_ns"] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("simulation_time_ns", True, "must be an integer"),
        ("allocation_bytes", -1, "must be at least 0"),
        ("host_projection_time_ns", 1.5, "must be an integer"),
        ("device_transfer_time_ns", 1, "must be zero on CPU"),
    ),
)
def test_malformed_native_patch_instrumentation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: Any,
    message: str,
) -> None:
    native_timings = _native_timings()
    native_timings[field] = value
    runner, extension_path, extension_sha256, _ = _install_patch_factory(
        tmp_path,
        monkeypatch,
        [native_timings],
    )
    controller = _controller(
        runner,
        extension_path,
        extension_sha256,
        instrumentation_enabled=True,
    )

    with pytest.raises(NativeRunControllerError, match=message):
        controller.run_many(GENERATION_COUNT)


def test_disabled_patch_instrumentation_rejects_native_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, extension_path, extension_sha256, _ = _install_patch_factory(
        tmp_path,
        monkeypatch,
        [_native_timings()],
    )
    controller = _controller(
        runner,
        extension_path,
        extension_sha256,
        instrumentation_enabled=False,
    )

    with pytest.raises(
        NativeRunControllerError,
        match="native Patch instrumentation was emitted while disabled",
    ):
        controller.run_many(GENERATION_COUNT)


def test_flat_controller_result_has_no_patch_instrumentation_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_controller_module,
        "_projection_digest_manifests",
        lambda projection: {"source_projection": {"digest": SOURCE_DIGEST}},
    )
    owner = _NativeOwner({}, {}, RUN_ID, None)
    controller = NativeRunController(
        owner,
        start_generation=0,
        default_steps_per_generation=2,
        default_max_births=1,
        safe_default_decision={"supported": True},
        extension_path="/not-loaded-in-this-test.so",
        run_id=RUN_ID,
    )

    result = controller.run_many(GENERATION_COUNT)

    assert "architecture_b_patch_instrumentation_receipt" not in result
    assert "instrumentation_enabled" not in controller.manifest()
