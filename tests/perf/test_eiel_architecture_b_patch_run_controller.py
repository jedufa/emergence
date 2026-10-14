from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import native_runtime.run_controller as run_controller_module
from native_runtime.patch_profile import PatchProfileClassification
from native_runtime.run_controller import (
    NATIVE_CONTROLLER_VERSION,
    NATIVE_INTER_TICK_CONTROL_CAPABILITY,
    NATIVE_PATCH_CAPABILITIES_ATTRIBUTE,
    NATIVE_PATCH_CONTROLLER_RECEIPT_SCHEMA_VERSION,
    NATIVE_PHASE4_CAPABILITIES_ATTRIBUTE,
    NATIVE_RESULT_SCHEMA_VERSION,
    NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION,
    NativeRunController,
    NativeRunControllerError,
)


RUN_ID = "patch-controller-run"
CONTROLLER_ID = "patch-controller-owner"
OWNER_EPOCH = 9
SOURCE_DIGEST = "a" * 64


def _runner(root: Path, *, patch: bool) -> SimpleNamespace:
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


def _extension_artifact(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "architecture_b_patch_candidate.so"
    payload = b"mock prebuilt extension"
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def _flat_decision(extension_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        extension_loaded=True,
        extension_runtime_jit=False,
        runtime_jit_prohibited=True,
        extension_path=str(extension_path),
        to_dict=lambda: {"supported": True, "route": "existing_flat"},
    )


def _patch_metadata(profile: str = "P1") -> dict[str, Any]:
    return {
        "world": "patch",
        "native_runtime_patch_profile": profile,
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


def _projection(metadata: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        tensors={},
        metadata=metadata,
        pending_generation_free_agent_ids=[],
        pending_generation_free_slots=[],
    )


class _NativeControllerProbe:
    def __init__(
        self,
        tensors: dict[str, Any],
        metadata: dict[str, Any],
        run_id: str,
    ) -> None:
        self.tensors = tensors
        self.metadata = metadata
        self.run_id = run_id
        self.status = "initialized"
        self.run_started = False
        self.sealed = False
        self.calls: list[tuple[int, int, int, int]] = []
        self.result: dict[str, Any] | None = None

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
        self.calls.append(
            (
                start_generation,
                generation_count,
                steps_per_generation,
                max_births,
            )
        )
        self.run_started = True
        self.status = "awaiting_snapshot_validation"
        self.result = {
            "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
            "decision": "ARCHB-PATCH-CONTROLLED-DEVELOPMENT-RUN",
            "run_id": self.run_id,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "executed": True,
            "native_owner": True,
            "hidden_fallback": False,
            "runtime_jit": False,
            "terminal_status": self.status,
            "start_generation": start_generation,
            "requested_generations": generation_count,
            "completed_generations": generation_count,
            "trajectory_commit_frontier": start_generation + generation_count - 1,
            "snapshot_frontier": start_generation - 1,
            "next_generation": start_generation + generation_count,
            "next_tick": 0,
            "controller_invocations": 1,
            "python_generation_dispatches": 0,
            "native_generation_transactions": generation_count,
            "turnover_commits": generation_count,
            "boundary_snapshot_count": generation_count,
            "projection_export_count": generation_count * 2,
            "observer_live_owner_exports": 0,
            "persistence_acknowledgement_count": 0,
            "projection_tensor_bytes": 0,
            "boundary_wall_time_ns": 0,
            "boundaries": [],
            "final_snapshot_sequence": generation_count,
            "success_marker_allowed": False,
            "promotion_result_allowed": False,
            "final_checkpoint_projection": {},
            "final_checkpoint_projection_digest_manifests": {
                "source_projection": {"digest": SOURCE_DIGEST}
            },
        }
        return self.result

    def acknowledge_snapshot_validation(
        self,
        source_projection_digest: str,
    ) -> dict[str, Any]:
        assert self.result is not None
        self.status = "run_complete"
        self.sealed = True
        self.result["terminal_status"] = self.status
        self.result["snapshot_frontier"] = self.result[
            "trajectory_commit_frontier"
        ]
        self.result["success_marker_allowed"] = True
        return {
            "schema_version": NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "controller_id": CONTROLLER_ID,
            "owner_epoch": OWNER_EPOCH,
            "acknowledged": True,
            "disposition": "accepted",
            "terminal_status": self.status,
            "trajectory_commit_frontier": self.result[
                "trajectory_commit_frontier"
            ],
            "snapshot_frontier": self.result["snapshot_frontier"],
            "final_snapshot_sequence": self.result["final_snapshot_sequence"],
            "validated_source_projection_digest": source_projection_digest,
            "success_marker_allowed": True,
        }

    def fail_snapshot_validation(self, message: str) -> dict[str, Any]:
        raise AssertionError(f"unexpected snapshot validation failure: {message}")

    def final_result(self) -> dict[str, Any]:
        if self.result is None:
            return {"terminal": False, "status": self.status}
        return self.result


def _module(
    controller_type: Any,
    *,
    controlled_patch: bool = True,
) -> SimpleNamespace:
    patch_capabilities = {
        "controlled_architecture_b_cpu_v1": controlled_patch,
        "resident_action_cache_cpu_v1": True,
        "resident_population_birth_commit_cpu_v1": True,
        "exact_parity_qualified": False,
        "promotion_accepted": False,
        "jungle_supported": True,
        "wdl_supported": True,
        "gru_supported": False,
        "cuda_supported": False,
    }
    return SimpleNamespace(
        EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_VERSION=NATIVE_CONTROLLER_VERSION,
        FlatCpuArchitectureBControllerV1=controller_type,
        **{
            NATIVE_PHASE4_CAPABILITIES_ATTRIBUTE: {
                NATIVE_INTER_TICK_CONTROL_CAPABILITY: True,
            },
            NATIVE_PATCH_CAPABILITIES_ATTRIBUTE: patch_capabilities,
        },
    )


def test_base_patch_capability_does_not_require_later_families() -> None:
    module = _module(_NativeControllerProbe)
    capabilities = getattr(module, NATIVE_PATCH_CAPABILITIES_ATTRIBUTE)
    capabilities["jungle_supported"] = False
    capabilities["wdl_supported"] = False

    normalized = run_controller_module._patch_extension_capabilities(module)

    assert normalized["jungle_supported"] is False
    assert normalized["wdl_supported"] is False


def _patch_resident_import(
    monkeypatch: pytest.MonkeyPatch,
    projection: SimpleNamespace,
    scopes: list[dict[str, bool]],
) -> None:
    import perf.cxxbackbone10_resident_state_owner as resident_owner_module

    def build_import_projection(
        runner: Any,
        scope: dict[str, bool],
        *,
        generation: int,
        tick: int,
    ) -> SimpleNamespace:
        assert generation == 0
        assert tick == 0
        scopes.append(dict(scope))
        return projection

    monkeypatch.setattr(
        resident_owner_module,
        "build_import_projection",
        build_import_projection,
    )


def test_flat_from_runner_keeps_existing_qualification_and_import_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_path, extension_sha256 = _extension_artifact(tmp_path)
    runner = _runner(tmp_path, patch=False)
    validation_calls: list[Any] = []
    scopes: list[dict[str, bool]] = []
    instances: list[_NativeControllerProbe] = []

    monkeypatch.setattr(
        run_controller_module,
        "validate_native_runtime_supported",
        lambda value: validation_calls.append(value) or _flat_decision(extension_path),
    )
    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        lambda *_args, **_kwargs: pytest.fail(
            "Flat path reached the Patch classifier"
        ),
    )
    _patch_resident_import(monkeypatch, _projection({}), scopes)

    def controller_type(
        tensors: dict[str, Any],
        metadata: dict[str, Any],
        run_id: str,
    ) -> _NativeControllerProbe:
        instance = _NativeControllerProbe(tensors, metadata, run_id)
        instances.append(instance)
        return instance

    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: _module(controller_type),
    )

    controller = NativeRunController.from_runner(
        runner,
        extension_path=extension_path,
        expected_extension_sha256=extension_sha256,
    )

    assert validation_calls == [runner]
    assert scopes == [{"flat": True, "patch": False, "churn": True}]
    assert len(instances) == 1
    manifest = controller.manifest()
    assert manifest["safe_default_decision"]["route"] == "existing_flat"
    assert "native_runtime_patch_profile" not in manifest
    assert "architecture_b_patch_controller_receipt" not in controller.final_result()


@pytest.mark.parametrize(
    "reason",
    [
        "unsupported_active_jungle",
        "malformed_inspection_field:env_cfg.patch_world_schema_version",
    ],
)
def test_patch_profile_rejection_precedes_import_owner_and_trajectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: str,
) -> None:
    extension_path, extension_sha256 = _extension_artifact(tmp_path)
    runner = _runner(tmp_path, patch=True)
    reached: list[str] = []
    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        lambda *_args, **_kwargs: PatchProfileClassification(
            supported=False,
            unsupported_reasons=(reason,),
        ),
    )
    monkeypatch.setattr(
        run_controller_module,
        "validate_native_runtime_supported",
        lambda runner: reached.append("safe_default"),
    )
    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: reached.append("extension_load"),
    )

    with pytest.raises(NativeRunControllerError, match=reason):
        NativeRunController.from_runner(
            runner,
            extension_path=extension_path,
            expected_extension_sha256=extension_sha256,
        )

    assert reached == []


def test_patch_wdl_route_masks_only_the_accepted_wdl_switches_for_base_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path, patch=True)
    runner.env_cfg.wdl_delayed_hazard_enable = True
    runner.env_cfg.wdl_delayed_hazard_observation_enable = True
    inspected: list[tuple[bool, bool]] = []

    def classify(
        _sim_cfg: Any,
        env_cfg: Any,
        _runner: Any,
        *,
        world: Any,
    ) -> PatchProfileClassification:
        assert world is None
        inspected.append(
            (
                env_cfg.wdl_delayed_hazard_enable,
                env_cfg.wdl_delayed_hazard_observation_enable,
            )
        )
        return PatchProfileClassification(supported=True, profile="P1")

    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        classify,
    )

    world, profile, classification = (
        run_controller_module._classify_architecture_b_world_profile(runner)
    )

    assert (world, profile) == ("patch", "P1")
    assert inspected == [(False, False)]
    assert classification is not None
    assert classification["wdl_native_execution"] is True
    assert classification["wdl_native_observation"] is True


def test_patch_wdl_route_rejects_observation_without_execution(tmp_path: Path) -> None:
    runner = _runner(tmp_path, patch=True)
    runner.env_cfg.wdl_delayed_hazard_enable = False
    runner.env_cfg.wdl_delayed_hazard_observation_enable = True

    with pytest.raises(NativeRunControllerError, match="observations require active WDL"):
        run_controller_module._classify_architecture_b_world_profile(runner)


def test_patch_jungle_route_masks_only_the_accepted_jungle_switches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path, patch=True)
    runner.env_cfg.jungle_predator_pressure_enable = True
    runner.env_cfg.jungle_reproduction_emissions_enable = True
    inspected: list[tuple[bool, bool]] = []

    def classify(
        _sim_cfg: Any,
        env_cfg: Any,
        _runner: Any,
        *,
        world: Any,
    ) -> PatchProfileClassification:
        assert world is None
        inspected.append(
            (
                env_cfg.jungle_predator_pressure_enable,
                env_cfg.jungle_reproduction_emissions_enable,
            )
        )
        return PatchProfileClassification(supported=True, profile="P2")

    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        classify,
    )

    world, profile, classification = (
        run_controller_module._classify_architecture_b_world_profile(runner)
    )

    assert (world, profile) == ("patch", "P2")
    assert inspected == [(False, False)]
    assert classification is not None
    assert classification["jungle_native_execution"] is True
    assert classification["jungle_native_features"][
        "jungle_predator_pressure_enable"
    ] is True


def test_patch_requires_explicit_development_extension_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path, patch=True)
    reached: list[str] = []
    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        lambda *_args, **_kwargs: PatchProfileClassification(
            supported=True,
            profile="P0",
        ),
    )
    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: reached.append("extension_load"),
    )

    with pytest.raises(
        NativeRunControllerError,
        match="requires an explicit development extension",
    ):
        NativeRunController.from_runner(runner)

    assert reached == []


def test_patch_requires_controlled_composer_before_owner_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_path, extension_sha256 = _extension_artifact(tmp_path)
    runner = _runner(tmp_path, patch=True)
    scopes: list[dict[str, bool]] = []
    owner_constructions: list[str] = []
    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        lambda *_args, **_kwargs: PatchProfileClassification(
            supported=True,
            profile="P1",
        ),
    )
    _patch_resident_import(
        monkeypatch,
        _projection(_patch_metadata("P1")),
        scopes,
    )

    def controller_type(
        tensors: dict[str, Any],
        metadata: dict[str, Any],
        run_id: str,
    ) -> _NativeControllerProbe:
        owner_constructions.append(run_id)
        return _NativeControllerProbe(tensors, metadata, run_id)

    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: _module(controller_type, controlled_patch=False),
    )

    with pytest.raises(
        NativeRunControllerError,
        match="controlled_architecture_b_cpu_v1 must be True",
    ):
        NativeRunController.from_runner(
            runner,
            extension_path=extension_path,
            expected_extension_sha256=extension_sha256,
        )

    assert scopes == [{"flat": False, "patch": True, "churn": True}]
    assert owner_constructions == []


def test_patch_wdl_route_imports_with_an_explicit_wdl_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_path, extension_sha256 = _extension_artifact(tmp_path)
    runner = _runner(tmp_path, patch=True)
    runner.env_cfg.wdl_delayed_hazard_enable = True
    runner.env_cfg.wdl_delayed_hazard_observation_enable = False
    scopes: list[dict[str, bool]] = []
    metadata = _patch_metadata("P1")
    metadata["native_runtime_wdl_execution_enable"] = True
    metadata["capability_mask"]["wdl"] = True
    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        lambda *_args, **_kwargs: PatchProfileClassification(
            supported=True,
            profile="P1",
        ),
    )
    _patch_resident_import(monkeypatch, _projection(metadata), scopes)
    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: _module(_NativeControllerProbe, controlled_patch=False),
    )

    with pytest.raises(
        NativeRunControllerError,
        match="controlled_architecture_b_cpu_v1 must be True",
    ):
        NativeRunController.from_runner(
            runner,
            extension_path=extension_path,
            expected_extension_sha256=extension_sha256,
        )

    assert scopes == [
        {"flat": False, "patch": True, "churn": True, "wdl": True}
    ]


def test_patch_jungle_route_imports_with_an_explicit_jungle_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_path, extension_sha256 = _extension_artifact(tmp_path)
    runner = _runner(tmp_path, patch=True)
    runner.env_cfg.jungle_predator_pressure_enable = True
    scopes: list[dict[str, bool]] = []
    metadata = _patch_metadata("P2")
    metadata["native_runtime_jungle_execution_enable"] = True
    metadata["capability_mask"]["jungle"] = True
    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        lambda *_args, **_kwargs: PatchProfileClassification(
            supported=True,
            profile="P2",
        ),
    )
    _patch_resident_import(monkeypatch, _projection(metadata), scopes)
    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: _module(_NativeControllerProbe, controlled_patch=False),
    )

    with pytest.raises(
        NativeRunControllerError,
        match="controlled_architecture_b_cpu_v1 must be True",
    ):
        NativeRunController.from_runner(
            runner,
            extension_path=extension_path,
            expected_extension_sha256=extension_sha256,
        )

    assert scopes == [
        {"flat": False, "patch": True, "churn": True, "jungle": True}
    ]


@pytest.mark.parametrize("profile", ("P0", "P1", "P2"))
def test_patch_single_controller_call_persists_identity_and_callback_counters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    extension_path, extension_sha256 = _extension_artifact(tmp_path)
    runner = _runner(tmp_path, patch=True)
    scopes: list[dict[str, bool]] = []
    instances: list[_NativeControllerProbe] = []
    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        lambda *_args, **_kwargs: PatchProfileClassification(
            supported=True,
            profile=profile,
        ),
    )
    monkeypatch.setattr(
        run_controller_module,
        "validate_native_runtime_supported",
        lambda runner: pytest.fail("Patch route promoted through safe defaults"),
    )
    _patch_resident_import(
        monkeypatch,
        _projection(_patch_metadata(profile)),
        scopes,
    )

    def controller_type(
        tensors: dict[str, Any],
        metadata: dict[str, Any],
        run_id: str,
    ) -> _NativeControllerProbe:
        instance = _NativeControllerProbe(tensors, metadata, run_id)
        instances.append(instance)
        return instance

    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: _module(controller_type),
    )
    monkeypatch.setattr(
        run_controller_module,
        "_projection_digest_manifests",
        lambda projection: {"source_projection": {"digest": SOURCE_DIGEST}},
    )

    controller = NativeRunController.from_runner(
        runner,
        extension_path=extension_path,
        expected_extension_sha256=extension_sha256,
    )
    monkeypatch.setattr(
        controller,
        "_normalize_and_validate_result",
        lambda raw, *, requested, expected_terminal_status: raw,
    )

    result = controller.run_many(
        4,
        steps_per_generation=2,
        max_births=1,
    )

    assert scopes == [{"flat": False, "patch": True, "churn": True}]
    assert len(instances) == 1
    assert instances[0].metadata["world"] == "patch"
    assert instances[0].metadata["native_runtime_patch_profile"] == profile
    assert instances[0].calls == [(0, 4, 2, 1)]

    receipt = result["architecture_b_patch_controller_receipt"]
    assert receipt["schema_version"] == (
        NATIVE_PATCH_CONTROLLER_RECEIPT_SCHEMA_VERSION
    )
    assert receipt["profile"] == profile
    assert receipt["resident_composer"] == (
        "execute_generation_patch_with_population_birth_commit_controlled_cpu"
    )
    assert receipt["controller_capability"] == (
        "controlled_architecture_b_cpu_v1"
    )
    assert receipt["run_id"] == RUN_ID
    assert receipt["controller_id"] == CONTROLLER_ID
    assert receipt["owner_epoch"] == OWNER_EPOCH
    assert receipt["initial_import_count"] == 1
    assert receipt["initial_owner_count"] == 1
    assert receipt["controller_call_count"] == 1
    assert receipt["native_generation_count"] == 4
    assert receipt["python_tick_callback_count"] == 0
    assert receipt["python_generation_trajectory_callback_count"] == 0
    assert receipt["boundary_snapshot_count"] == 4
    assert receipt["boundary_projection_export_count"] == 8
    assert receipt["hidden_fallback_count"] == 0
    assert receipt["runtime_compilation_count"] == 0
    assert receipt["production_routing_enabled"] is False
    assert receipt["safe_default_promoted"] is False
    assert receipt["promotion_accepted"] is False
    assert receipt["cuda_supported"] is False
    assert result["hidden_fallback"] is False
    assert result["runtime_jit"] is False
    assert result["promotion_result_allowed"] is False

    manifest = controller.manifest()
    assert manifest["controller_id"] == CONTROLLER_ID
    assert manifest["owner_epoch"] == OWNER_EPOCH
    assert manifest["initial_import_count"] == 1
    assert manifest["trajectory_dispatch_count"] == 1
    assert manifest["controller_call_count"] == 1
    assert manifest["python_callback_invocation_count"] == 0
    assert manifest["python_tick_callback_invocation_count"] == 0
    assert manifest[
        "python_generation_trajectory_callback_invocation_count"
    ] == 0
    assert manifest["runtime_jit_request_count"] == 0
    assert manifest["hidden_fallback_count"] == 0
    qualification = manifest["safe_default_decision"]
    assert qualification["controller_execution_active"] is True
    assert qualification["production_routing_enabled"] is False
    assert qualification["safe_default_promoted"] is False
    assert qualification["promotion_accepted"] is False
    assert qualification["runtime_jit_prohibited"] is True
    assert qualification["hidden_fallback_allowed"] is False
    assert qualification[
        "architecture_b_extension_explicit_development_artifact"
    ] is True
    assert controller.final_result()[
        "architecture_b_patch_controller_receipt"
    ] == receipt


def test_patch_restored_checkpoint_requires_exact_lowercase_sha_before_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension_path, extension_sha256 = _extension_artifact(tmp_path)
    runner = _runner(tmp_path, patch=True)
    runner._native_runtime_checkpoint_apply = {"status": "restored"}
    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        lambda *_args, **_kwargs: PatchProfileClassification(
            supported=True,
            profile="P2",
        ),
    )
    reached: list[str] = []
    monkeypatch.setattr(
        run_controller_module,
        "_restored_native_checkpoint_projection",
        lambda *_args, **_kwargs: reached.append("restore"),
    )

    with pytest.raises(
        NativeRunControllerError,
        match="must be exactly 64 lowercase hexadecimal characters",
    ):
        NativeRunController.from_runner(
            runner,
            extension_path=extension_path,
            expected_extension_sha256=extension_sha256.upper(),
        )
    assert reached == []


def test_patch_resume_envelope_requires_explicit_development_extension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(tmp_path, patch=True)
    reached: list[str] = []
    monkeypatch.setattr(
        run_controller_module,
        "classify_patch_architecture_b_profile",
        lambda *_args, **_kwargs: PatchProfileClassification(
            supported=True,
            profile="P2",
        ),
    )
    monkeypatch.setattr(
        run_controller_module,
        "validate_architecture_b_resume_envelope_authority",
        lambda envelope: reached.append("resume_envelope"),
    )
    monkeypatch.setattr(
        run_controller_module,
        "validate_native_runtime_supported",
        lambda value: reached.append("flat_qualification"),
    )

    with pytest.raises(
        NativeRunControllerError,
        match="requires an explicit development extension",
    ):
        NativeRunController.from_checkpoint_projection(runner, {})

    assert reached == []
