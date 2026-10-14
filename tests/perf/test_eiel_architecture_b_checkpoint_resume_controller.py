from __future__ import annotations

import copy
import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from native_runtime import NativeEielEngine, NativeEngineCounters
from native_runtime import run_controller as run_controller_module
from native_runtime.checkpoint_digests import (
    checkpoint_source_projection_authority_manifest,
)
from native_runtime.compact_summary_digests import (
    POLICY_MUTATION_TAPE_SCHEMA_VERSION,
)
from native_runtime.run_controller import (
    NATIVE_BOUNDARY_SCHEMA_VERSION,
    NATIVE_CONTROLLER_PROTOCOL_SCHEMA_VERSION,
    NATIVE_CONTROLLER_PROTOCOL_VERSION,
    NATIVE_CONTROLLER_VERSION,
    NATIVE_PHASE4_REQUIRED_CAPABILITIES,
    NATIVE_POLICY_MUTATION_TAPE_BOUNDARY_AUTHORITY_SCHEMA_VERSION,
    NATIVE_RESULT_SCHEMA_VERSION,
    NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION,
    NativeRunController,
    NativeRunControllerError,
)
from simulation.checkpoint_bundle_v2 import (
    ARCHITECTURE_B_BUNDLE_KIND,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save_projection,
    decode_architecture_b_resume_envelope,
)
from tests.perf.test_eiel_native_checkpoint_single_projection_authority import (
    _projection,
)


RUN_ID = "architecture-b-resume"
CURRENT_CONTROLLER_ID = "architecture-b-resumed-controller"
CURRENT_OWNER_EPOCH = 8
POLICY_MUTATION_TAPE_FIELDS = (
    "policy_birth_crossover_mask_tape",
    "policy_birth_delta_tape",
    "policy_birth_mutation_mask_tape",
    "policy_birth_post_rng_state_tape",
    "policy_birth_semantic_draws_tape",
)


def _runner(extension_path: Path) -> Any:
    return SimpleNamespace(
        run_dir=extension_path.parent / "run",
        sim_cfg=SimpleNamespace(
            chronicle_run_id=RUN_ID,
            steps_per_generation=4,
            device="cpu",
            native_runtime_fixed_policy_family="mlp",
            native_runtime_max_births_per_generation=0,
        ),
        env_cfg=SimpleNamespace(
            world="flat",
            tier4_juvenile_vulnerability_enable=False,
            tier4_juvenile_steps=0,
        ),
    )


def _decision(extension_path: Path) -> Any:
    return SimpleNamespace(
        extension_loaded=True,
        extension_runtime_jit=False,
        runtime_jit_prohibited=True,
        extension_path=str(extension_path),
        to_dict=lambda: {"native_supported": True},
    )


def _checkpoint_projection() -> dict[str, Any]:
    projection = _projection(generation=2, tick=0)
    projection["metadata"].update(
        {
            "device": "cpu",
            "native_runtime_fixed_policy_family": "mlp",
        }
    )
    projection["tensor_manifest"] = {
        name: {
            "dtype": str(tensor.dtype),
            "shape": [int(dim) for dim in tensor.shape],
            "device": str(tensor.device),
            "bytes": int(tensor.numel()) * int(tensor.element_size()),
        }
        for name, tensor in projection["tensors"].items()
    }
    return projection


def _continuation() -> dict[str, Any]:
    return {
        "schema_version": "eiel_architecture_b_observer_continuation_v1",
        "checkpoint_next_generation": 2,
        "tick": 0,
        "typed_lifecycle_event_cursor": 0,
        "exact_reporting_boundary_ordinal": 2,
        "behavior_completed_generation_and_row_key": {
            "completed_generation": 1,
            "last_row_key": "generation:1:agent:1",
        },
        "metrics_completed_generation": 1,
        "lineage_completed_generation_and_row_key": {
            "completed_generation": 1,
            "last_row_key": "generation:1:lineage:1",
        },
        "chronicle_observation_ordinal": 2,
        "progress_sequence": 2,
        "schedule_phase": {"origin_generation": 0, "next_generation": 2},
        "deduplication": {
            "acknowledged_record_keys": ["generation:1:exact_reporting"]
        },
        "acknowledged_channel_cursor_ranges": {
            "exact_reporting": [1, 2],
            "behaviors": [1, 2],
            "metrics": [1, 2],
            "lineage": [1, 2],
            "typed_lifecycle": [0, 0],
            "chronicle": [1, 2],
            "checkpoint_generation": [2, 3],
        },
    }


def _policy_mutation_tape_boundary() -> tuple[dict[str, Any], dict[str, Any]]:
    tape = {
        "policy_birth_crossover_mask_tape": torch.zeros((1, 1), dtype=torch.bool),
        "policy_birth_delta_tape": torch.zeros((1, 1), dtype=torch.float32),
        "policy_birth_mutation_mask_tape": torch.zeros((1, 1), dtype=torch.bool),
        "policy_birth_post_rng_state_tape": torch.zeros((1, 1), dtype=torch.uint8),
        "policy_birth_semantic_draws_tape": torch.zeros((1,), dtype=torch.int64),
        "policy_birth_tape_cursor": torch.zeros((1,), dtype=torch.int64),
    }
    prefixes = {
        name: tape[name].narrow(0, 0, 0).contiguous()
        for name in POLICY_MUTATION_TAPE_FIELDS
    }
    authority = {
        "schema_version": (
            NATIVE_POLICY_MUTATION_TAPE_BOUNDARY_AUTHORITY_SCHEMA_VERSION
        ),
        "source_version": POLICY_MUTATION_TAPE_SCHEMA_VERSION,
        "cursor": 0,
        "consumed_shapes": {
            name: [int(size) for size in prefix.shape]
            for name, prefix in prefixes.items()
        },
        "tensor_sha256": {
            name: hashlib.sha256(prefix.numpy().tobytes(order="C")).hexdigest()
            for name, prefix in prefixes.items()
        },
        "authority_complete": True,
        "export_non_consuming": True,
    }
    return tape, authority


def _resume_envelope(tmp_path: Path) -> dict[str, Any]:
    projection = _checkpoint_projection()
    source_digest = str(
        checkpoint_source_projection_authority_manifest(projection)[
            "source_projection_sha256"
        ]
    )
    identity = {
        "run_id": RUN_ID,
        "controller_id": "architecture-b-controller-previous",
        "owner_epoch": 7,
        "snapshot_id": "architecture-b-controller-previous:owner:7:boundary:2",
        "snapshot_sequence": 2,
        "snapshot_digest_sha256": "2" * 64,
        "next_generation": 2,
        "tick": 0,
        "phase": "generation_boundary",
        "source_projection_sha256": source_digest,
    }
    continuation = _continuation()
    bundle = tmp_path / "architecture-b-resume-bundle"
    checkpoint_bundle_v2_save_projection(
        projection,
        str(bundle),
        generation_index=2,
        expected_source_projection_sha256=source_digest,
        architecture_b_identity=identity,
        observer_continuation=continuation,
    )
    payload = checkpoint_bundle_v2_load(
        str(bundle),
        strict=True,
        required_bundle_kind=ARCHITECTURE_B_BUNDLE_KIND,
    )
    return decode_architecture_b_resume_envelope(payload)


class _FactoryOwner:
    constructions: list[dict[str, Any]] = []

    def __init__(
        self,
        tensors: dict[str, Any],
        metadata: dict[str, Any],
        run_id: str,
        persistence_mailbox_required: bool,
        checkpoint_projection: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.status = "initialized"
        self.run_started = False
        self.sealed = False
        self.persistence_mailbox_required = persistence_mailbox_required
        type(self).constructions.append(
            {
                "tensors": tensors,
                "metadata": metadata,
                "run_id": run_id,
                "persistence_mailbox_required": persistence_mailbox_required,
                "checkpoint_projection": checkpoint_projection,
            }
        )

    def controller_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "eiel_native_architecture_b_controller_manifest_v1",
            "run_id": self.run_id,
            "controller_id": CURRENT_CONTROLLER_ID,
            "owner_epoch": CURRENT_OWNER_EPOCH,
            "controller_protocol_version": NATIVE_CONTROLLER_PROTOCOL_VERSION,
            "controller_protocol_schema_version": (
                NATIVE_CONTROLLER_PROTOCOL_SCHEMA_VERSION
            ),
            "controller_protocol_capabilities": {
                name: True for name in NATIVE_PHASE4_REQUIRED_CAPABILITIES
            },
            "status": self.status,
            "run_started": self.run_started,
            "sealed": self.sealed,
            "native_owner_private": True,
            "python_callback_entry_points": 0,
            "persistence_mailbox_required": self.persistence_mailbox_required,
        }

    def run_generations(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("not used by factory tests")

    def read_snapshots(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("not used by factory tests")

    def acknowledge_snapshot_persistence(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("not used by factory tests")

    def fail_snapshot_persistence(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("not used by factory tests")

    def request_cancel(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("not used by factory tests")

    def prepare_snapshot_validation(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("not used by factory tests")

    def commit_snapshot_validation(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("not used by factory tests")

    def acknowledge_snapshot_validation(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("not used by factory tests")

    def fail_snapshot_validation(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("not used by factory tests")

    def final_result(self) -> dict[str, Any]:
        return {"terminal": False, "status": self.status}


def _patch_factory(
    monkeypatch: pytest.MonkeyPatch,
    extension_path: Path,
    *,
    capabilities: dict[str, bool] | None = None,
) -> None:
    monkeypatch.setattr(
        run_controller_module,
        "validate_native_runtime_supported",
        lambda runner: _decision(extension_path),
    )
    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: SimpleNamespace(
            EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_VERSION=NATIVE_CONTROLLER_VERSION,
            EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_CAPABILITIES=(
                {name: True for name in NATIVE_PHASE4_REQUIRED_CAPABILITIES}
                if capabilities is None
                else capabilities
            ),
            FlatCpuArchitectureBControllerV1=_FactoryOwner,
        ),
    )


def _runner_with_restored_native_checkpoint(
    extension_path: Path,
    *,
    projection: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    runner = _runner(extension_path)
    restored_projection = _checkpoint_projection() if projection is None else projection
    source_sha256 = str(
        checkpoint_source_projection_authority_manifest(restored_projection)[
            "source_projection_sha256"
        ]
    )

    class RestoredOwner:
        def __init__(self) -> None:
            self.projection = copy.deepcopy(restored_projection)

        def export_checkpoint_projection(self) -> dict[str, Any]:
            restored = copy.deepcopy(self.projection)
            restored["restore_count"] = int(restored["restore_count"]) + 1
            return restored

    engine = NativeEielEngine.__new__(NativeEielEngine)
    engine.runner = runner
    engine.counters = NativeEngineCounters(
        engine_constructions=1,
        checkpoint_restore_calls=1,
    )
    engine._owner = RestoredOwner()
    runner._native_eiel_engine = engine
    runner._native_runtime_checkpoint_source_projection = restored_projection
    runner._native_runtime_checkpoint_apply = {
        "schema_version": "native_runtime_checkpoint_apply_v1",
        "status": "restored",
        "generation": 2,
        "tick": 0,
        "configured_steps_per_generation": 4,
        "remaining_steps": 4,
        "source": "checkpoint_bundle_v2.native_runtime_checkpoint_projection",
        "source_projection_sha256": source_sha256,
        "verification_target": "restored_native_engine",
    }
    return runner, restored_projection


def test_phase3_default_constructor_remains_compatible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "phase3.so"
    extension.write_bytes(b"phase3")
    runner = _runner(extension)
    runner.sim_cfg.chronicle_run_id = "phase3"
    monkeypatch.setattr(
        run_controller_module,
        "validate_native_runtime_supported",
        lambda current: _decision(extension),
    )
    import perf.cxxbackbone10_resident_state_owner as resident_module

    projection = SimpleNamespace(
        tensors={},
        metadata={},
        pending_generation_free_agent_ids=[],
        pending_generation_free_slots=[],
    )
    monkeypatch.setattr(
        resident_module,
        "build_import_projection",
        lambda *_args, **_kwargs: projection,
    )
    calls: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    class Phase3Owner(_FactoryOwner):
        def __init__(
            self, tensors: dict[str, Any], metadata: dict[str, Any], run_id: str
        ) -> None:
            calls.append((tensors, metadata, run_id))
            self.run_id = run_id
            self.status = "initialized"
            self.run_started = False
            self.sealed = False
            self.persistence_mailbox_required = False

        def controller_manifest(self) -> dict[str, Any]:
            manifest = super().controller_manifest()
            for field in (
                "controller_protocol_version",
                "controller_protocol_schema_version",
                "controller_protocol_capabilities",
                "persistence_mailbox_required",
            ):
                manifest.pop(field, None)
            return manifest

    monkeypatch.setattr(
        run_controller_module,
        "load_prebuilt_extension",
        lambda path: SimpleNamespace(
            EIEL_NATIVE_ARCHITECTURE_B_CONTROLLER_VERSION=NATIVE_CONTROLLER_VERSION,
            FlatCpuArchitectureBControllerV1=Phase3Owner,
        ),
    )
    def uninspectable_pybind_class(_value: Any) -> Any:
        raise ValueError("pybind class has no inspectable signature")

    monkeypatch.setattr(
        run_controller_module.inspect,
        "signature",
        uninspectable_pybind_class,
    )
    controller = NativeRunController.from_runner(
        runner,
        extension_path=extension,
        expected_extension_sha256=hashlib.sha256(b"phase3").hexdigest(),
    )
    assert calls == [
        (
            {},
            {"native_runtime_architecture_b_compact_reporting_only_v1": False},
            "phase3",
        )
    ]
    assert controller.manifest()["persistence_mailbox_required"] is False


def test_from_runner_passes_explicit_persistence_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "phase4.so"
    extension.write_bytes(b"phase4")
    runner = _runner(extension)
    import perf.cxxbackbone10_resident_state_owner as resident_module

    projection = SimpleNamespace(
        tensors={},
        metadata={},
        pending_generation_free_agent_ids=[],
        pending_generation_free_slots=[],
    )
    monkeypatch.setattr(
        resident_module,
        "build_import_projection",
        lambda *_args, **_kwargs: projection,
    )
    _FactoryOwner.constructions.clear()
    _patch_factory(monkeypatch, extension)
    controller = NativeRunController.from_runner(
        runner,
        extension_path=extension,
        expected_extension_sha256=hashlib.sha256(b"phase4").hexdigest(),
        persistence_mailbox_required=True,
    )
    assert _FactoryOwner.constructions[-1]["persistence_mailbox_required"] is True
    assert controller.manifest()["persistence_mailbox_required"] is True


def test_from_runner_passes_default_false_to_capable_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "phase3-capable.so"
    extension.write_bytes(b"phase3-capable")
    runner = _runner(extension)
    import perf.cxxbackbone10_resident_state_owner as resident_module

    monkeypatch.setattr(
        resident_module,
        "build_import_projection",
        lambda *_args, **_kwargs: SimpleNamespace(
            tensors={},
            metadata={},
            pending_generation_free_agent_ids=[],
            pending_generation_free_slots=[],
        ),
    )
    _FactoryOwner.constructions.clear()
    _patch_factory(monkeypatch, extension)
    NativeRunController.from_runner(
        runner,
        extension_path=extension,
        expected_extension_sha256=hashlib.sha256(b"phase3-capable").hexdigest(),
    )
    assert _FactoryOwner.constructions[-1]["persistence_mailbox_required"] is False


def test_from_runner_adapts_bound_restored_native_checkpoint_without_runner_bank_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "phase4-checkpoint-adapter.so"
    extension.write_bytes(b"phase4-checkpoint-adapter")
    runner, restored_projection = _runner_with_restored_native_checkpoint(extension)
    assert not hasattr(runner, "_mlp_pack")

    import perf.cxxbackbone10_resident_state_owner as resident_module

    def forbidden_runner_projection(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("restored checkpoint authority must bypass runner cache rebuild")

    monkeypatch.setattr(
        resident_module,
        "build_import_projection",
        forbidden_runner_projection,
    )
    _FactoryOwner.constructions.clear()
    _patch_factory(monkeypatch, extension)

    controller = NativeRunController.from_runner(
        runner,
        generation=2,
        extension_path=extension,
        expected_extension_sha256=hashlib.sha256(
            b"phase4-checkpoint-adapter"
        ).hexdigest(),
        persistence_mailbox_required=True,
    )

    constructed = _FactoryOwner.constructions[-1]
    restored_mlp_pack = restored_projection["tensors"]["mlp_pack"]
    assert constructed["tensors"]["mlp_pack"].equal(restored_mlp_pack)
    assert constructed["tensors"]["mlp_pack"].data_ptr() != restored_mlp_pack.data_ptr()
    assert checkpoint_source_projection_authority_manifest(
        constructed["checkpoint_projection"]
    )["source_projection_sha256"] == checkpoint_source_projection_authority_manifest(
        restored_projection
    )["source_projection_sha256"]
    assert constructed["checkpoint_projection"]["generation_calls"] == (
        restored_projection["generation_calls"]
    )
    assert constructed["checkpoint_projection"]["restore_count"] == (
        restored_projection["restore_count"]
    )
    decision = controller.manifest()["safe_default_decision"]
    assert decision["architecture_b_checkpoint_resume"] is True
    assert decision["architecture_b_checkpoint_import_source"] == (
        "checkpoint_bundle_v2.native_runtime_checkpoint_projection"
    )
    assert decision["architecture_b_source_projection_sha256"] == (
        runner._native_runtime_checkpoint_apply["source_projection_sha256"]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda runner: runner._native_runtime_checkpoint_apply.update(
                {"generation": 1}
            ),
            "apply cursor contradicts",
        ),
        (
            lambda runner: runner._native_eiel_engine._owner
            .projection
            ["tensors"]
            .pop("mlp_pack"),
            "missing authoritative checkpoint tensor fields",
        ),
    ],
)
def test_from_runner_rejects_invalid_restored_checkpoint_without_runner_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    message: str,
) -> None:
    extension = tmp_path / "invalid-restored-checkpoint.so"
    extension.write_bytes(b"invalid-restored-checkpoint")
    runner, _ = _runner_with_restored_native_checkpoint(extension)
    mutate(runner)

    import perf.cxxbackbone10_resident_state_owner as resident_module

    monkeypatch.setattr(
        resident_module,
        "build_import_projection",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid restored checkpoint must not fall back to runner reconstruction"
        ),
    )
    _FactoryOwner.constructions.clear()
    _patch_factory(monkeypatch, extension)

    with pytest.raises(NativeRunControllerError, match=message):
        NativeRunController.from_runner(
            runner,
            generation=2,
            extension_path=extension,
            expected_extension_sha256=hashlib.sha256(
                b"invalid-restored-checkpoint"
            ).hexdigest(),
            persistence_mailbox_required=True,
        )
    assert _FactoryOwner.constructions == []


def test_strict_resume_import_uses_loader_envelope_without_runner_rebuild(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "resume.so"
    extension.write_bytes(b"resume")
    envelope = _resume_envelope(tmp_path)
    expected_metadata = copy.deepcopy(envelope["controller_import_metadata"])
    original_tensor = envelope["checkpoint_projection"]["tensors"]["alive_mask"]
    _FactoryOwner.constructions.clear()
    _patch_factory(monkeypatch, extension)

    controller = NativeRunController.from_checkpoint_projection(
        _runner(extension),
        envelope,
        extension_path=extension,
        expected_extension_sha256=hashlib.sha256(b"resume").hexdigest(),
    )

    constructed = _FactoryOwner.constructions[-1]
    assert constructed["metadata"] == expected_metadata
    assert constructed["metadata"]["architecture_b_identity"] == envelope[
        "architecture_b_identity"
    ]
    assert constructed["metadata"]["observer_continuation"] == envelope[
        "observer_continuation"
    ]
    assert checkpoint_source_projection_authority_manifest(
        constructed["checkpoint_projection"]
    )["source_projection_sha256"] == envelope["source_projection_sha256"]
    assert constructed["tensors"]["alive_mask"].data_ptr() != original_tensor.data_ptr()
    manifest = controller.manifest()
    assert manifest["initial_boundary_ordinal"] == 2
    assert manifest["schedule_origin_generation"] == 0
    assert manifest["owner_epoch"] == CURRENT_OWNER_EPOCH


@pytest.mark.parametrize(
    "mutate",
    [
        lambda envelope: envelope["checkpoint_projection"].pop("tensors"),
        lambda envelope: envelope["checkpoint_projection"][
            "pending_generation_free_slots"
        ].append(1),
        lambda envelope: envelope["checkpoint_projection"]["metadata"].update(
            {"tick": 1}
        ),
    ],
)
def test_strict_resume_rejects_malformed_or_partial_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
) -> None:
    extension = tmp_path / "resume.so"
    extension.write_bytes(b"resume")
    envelope = _resume_envelope(tmp_path)
    mutate(envelope)
    _FactoryOwner.constructions.clear()
    _patch_factory(monkeypatch, extension)

    with pytest.raises(NativeRunControllerError):
        NativeRunController.from_checkpoint_projection(
            _runner(extension), envelope, extension_path=extension,
            expected_extension_sha256=hashlib.sha256(b"resume").hexdigest(),
        )
    assert _FactoryOwner.constructions == []


def test_strict_resume_rejects_continuation_mismatch_and_missing_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "resume.so"
    extension.write_bytes(b"resume")
    envelope = _resume_envelope(tmp_path)
    envelope["observer_continuation"]["schedule_phase"]["origin_generation"] = 1
    envelope["controller_import_metadata"]["observer_continuation"] = copy.deepcopy(
        envelope["observer_continuation"]
    )
    _patch_factory(monkeypatch, extension)
    with pytest.raises(NativeRunControllerError, match="changed after strict payload decoding"):
        NativeRunController.from_checkpoint_projection(
            _runner(extension), envelope, extension_path=extension,
            expected_extension_sha256=hashlib.sha256(b"resume").hexdigest(),
        )

    valid = _resume_envelope(tmp_path)
    _patch_factory(
        monkeypatch,
        extension,
        capabilities={"persistence_mailbox_v1": True},
    )
    with pytest.raises(NativeRunControllerError, match="capabilities are missing"):
        NativeRunController.from_checkpoint_projection(
            _runner(extension), valid, extension_path=extension,
            expected_extension_sha256=hashlib.sha256(b"resume").hexdigest(),
        )


@pytest.mark.parametrize(
    "missing_capability",
    [
        "resume_observer_continuation_v1",
        "acknowledgement_settlement_gate_v1",
    ],
)
def test_phase4_resume_rejects_each_required_authority_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_capability: str,
) -> None:
    extension = tmp_path / "resume.so"
    extension.write_bytes(b"resume")
    capabilities = {
        name: True
        for name in NATIVE_PHASE4_REQUIRED_CAPABILITIES
        if name != missing_capability
    }
    _patch_factory(monkeypatch, extension, capabilities=capabilities)

    with pytest.raises(NativeRunControllerError, match=missing_capability):
        NativeRunController.from_checkpoint_projection(
            _runner(extension),
            _resume_envelope(tmp_path),
            extension_path=extension,
            expected_extension_sha256=hashlib.sha256(b"resume").hexdigest(),
        )


def test_strict_resume_rejects_copied_or_post_decode_mutated_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = tmp_path / "resume.so"
    extension.write_bytes(b"resume")
    _patch_factory(monkeypatch, extension)
    validated = _resume_envelope(tmp_path)
    with pytest.raises(NativeRunControllerError, match="strict loader"):
        NativeRunController.from_checkpoint_projection(
            _runner(extension),
            dict(validated),
            extension_path=extension,
            expected_extension_sha256=hashlib.sha256(b"resume").hexdigest(),
        )

    identity_mutated = _resume_envelope(tmp_path)
    identity_mutated["architecture_b_identity"]["snapshot_digest_sha256"] = "f" * 64
    identity_mutated["controller_import_metadata"]["architecture_b_identity"] = (
        copy.deepcopy(identity_mutated["architecture_b_identity"])
    )
    with pytest.raises(NativeRunControllerError, match="changed after strict payload decoding"):
        NativeRunController.from_checkpoint_projection(
            _runner(extension),
            identity_mutated,
            extension_path=extension,
            expected_extension_sha256=hashlib.sha256(b"resume").hexdigest(),
        )

    continuation_mutated = _resume_envelope(tmp_path)
    continuation_mutated["observer_continuation"]["progress_sequence"] = 3
    continuation_mutated["controller_import_metadata"]["observer_continuation"] = (
        copy.deepcopy(continuation_mutated["observer_continuation"])
    )
    with pytest.raises(NativeRunControllerError, match="changed after strict payload decoding"):
        NativeRunController.from_checkpoint_projection(
            _runner(extension),
            continuation_mutated,
            extension_path=extension,
            expected_extension_sha256=hashlib.sha256(b"resume").hexdigest(),
        )


def _schedule(
    *,
    report_every: int = 0,
    explicit_next_generations: list[int] | None = None,
    checkpoint_on_cancel: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": "eiel_native_snapshot_schedule_v1",
        "progress_every_generations": 0,
        "report_every_generations": report_every,
        "checkpoint_every_generations": 0,
        "checkpoint_on_cancel": checkpoint_on_cancel,
        "explicit_next_generations": list(explicit_next_generations or []),
        "emit_final_snapshot": True,
        "typed_lifecycle_mode": "cursor_bounded_delta",
        "mailbox_capacity": 1,
    }


@pytest.mark.parametrize(
    "field",
    ["report_every_generations", "checkpoint_every_generations"],
)
@pytest.mark.parametrize("cadence", [0, 1, 2, 8])
def test_snapshot_schedule_accepts_only_supported_cadences(
    field: str,
    cadence: int,
) -> None:
    schedule = _schedule()
    schedule[field] = cadence

    assert run_controller_module._validate_snapshot_schedule(schedule) == schedule


@pytest.mark.parametrize(
    "field",
    ["report_every_generations", "checkpoint_every_generations"],
)
@pytest.mark.parametrize("cadence", [3, 9])
def test_snapshot_schedule_rejects_unsupported_cadences(
    field: str,
    cadence: int,
) -> None:
    schedule = _schedule()
    schedule[field] = cadence

    with pytest.raises(NativeRunControllerError, match="outside the closed"):
        run_controller_module._validate_snapshot_schedule(schedule)


def test_explicit_checkpoint_schedule_is_closed_and_normalized() -> None:
    schedule = _schedule(explicit_next_generations=[4])
    assert run_controller_module._validate_snapshot_schedule(schedule) == schedule


def test_checkpoint_on_cancel_is_a_required_frozen_phase4_schedule_field() -> None:
    schedule = _schedule(checkpoint_on_cancel=True)
    assert run_controller_module._validate_snapshot_schedule(schedule) == schedule

    missing = _schedule()
    missing.pop("checkpoint_on_cancel")
    with pytest.raises(
        NativeRunControllerError,
        match=r"missing=\['checkpoint_on_cancel'\]",
    ):
        run_controller_module._validate_snapshot_schedule(missing)

    with pytest.raises(NativeRunControllerError, match="requires checkpoint_on_cancel=true"):
        run_controller_module._validate_snapshot_schedule(
            _schedule(checkpoint_on_cancel=False)
        )

    non_boolean = _schedule()
    non_boolean["checkpoint_on_cancel"] = 1
    with pytest.raises(NativeRunControllerError, match="must be boolean"):
        run_controller_module._validate_snapshot_schedule(non_boolean)


@pytest.mark.parametrize(
    "explicit",
    [
        (4,),
        [4, 4],
        [5, 4],
        [True],
        [-1],
    ],
)
def test_explicit_checkpoint_schedule_rejects_malformed_coordinates(
    explicit: Any,
) -> None:
    schedule = _schedule()
    schedule["explicit_next_generations"] = explicit
    with pytest.raises(NativeRunControllerError, match="explicit"):
        run_controller_module._validate_snapshot_schedule(schedule)


def test_checkpoint_capability_does_not_mask_missing_boundary_schedule_echo() -> None:
    owner = _DispatchOwner()
    controller = _controller(owner)
    schedule = _schedule()
    expectations, _ = controller._scheduled_boundary_expectations(
        completed=1,
        requested=1,
        schedule=schedule,
    )
    boundary = _boundary(
        owner,
        generation=owner.start_generation,
        sequence=owner.initial_boundary_ordinal + 1,
        final=True,
        report_every=0,
    )
    boundary["schedule_state"].pop("checkpoint_on_cancel")

    with pytest.raises(
        NativeRunControllerError,
        match=r"schedule_state fields mismatch: missing=\['checkpoint_on_cancel'\]",
    ):
        controller._normalize_scheduled_boundary(
            boundary,
            expected=expectations[0],
            schedule=schedule,
        )


def _boundary(
    owner: "_DispatchOwner",
    *,
    generation: int,
    sequence: int,
    final: bool,
    report_every: int,
    checkpoint_on_cancel: bool = True,
    checkpoint_due: bool = False,
    cancellation_observed: bool = False,
) -> dict[str, Any]:
    pre = _checkpoint_projection()
    post = _checkpoint_projection()
    policy_mutation_tape, policy_mutation_tape_authority = (
        _policy_mutation_tape_boundary()
    )
    post["metadata"]["generation"] = generation + 1
    post["metadata"]["agent_grid_authority_epoch"] = (
        f"generation:{generation + 1}:tick:0"
    )
    pre_bytes = sum(row["bytes"] for row in pre["tensor_manifest"].values())
    post_bytes = sum(row["bytes"] for row in post["tensor_manifest"].values())
    boundary = {
        "schema_version": NATIVE_BOUNDARY_SCHEMA_VERSION,
        "run_id": owner.run_id,
        "controller_id": owner.controller_id,
        "owner_epoch": owner.owner_epoch,
        "snapshot_id": (
            f"{owner.controller_id}:owner:{owner.owner_epoch}:boundary:{sequence}"
        ),
        "snapshot_sequence": sequence,
        "boundary_ordinal": sequence,
        "completed_generation": generation,
        "next_generation": generation + 1,
        "tick": 0,
        "phase": "generation_complete",
        "controller_status_at_clone": "generation_complete_waiting_persistence",
        "due_mask": {
            "progress": final,
            "exact_reporting": True,
            "checkpoint": checkpoint_due,
            "final": final,
        },
        "schedule_state": {
            "schema_version": "eiel_native_snapshot_schedule_v1",
            "schedule_origin_generation": owner.schedule_origin,
            "imported_boundary_ordinal": owner.initial_boundary_ordinal,
            "progress_every_generations": 0,
            "report_every_generations": report_every,
            "checkpoint_every_generations": 0,
            "checkpoint_on_cancel": checkpoint_on_cancel,
            "explicit_next_generations": [],
        },
        "compact_progress": {
            "completed_generation": generation,
            "next_generation": generation + 1,
            "tick": 0,
            "alive_count": 1,
        },
        "payload_profile": "canonical_checkpoint" if checkpoint_due else "exact_report",
        "generation_result": {"executed": True, "hidden_fallback": False},
        "generation_end_reporting_frame": pre,
        "generation_end_reporting_frame_role": (
            "transaction_local_generation_end_values_folded_after_turnover_commit"
        ),
        "turnover_result": {"executed": True, "hidden_fallback": False},
        "post_turnover_checkpoint_projection": post,
        "policy_mutation_tape": policy_mutation_tape,
        "policy_mutation_tape_authority": policy_mutation_tape_authority,
        "generation_end_reporting_frame_tensor_bytes": pre_bytes,
        "post_turnover_projection_tensor_bytes": post_bytes,
        "snapshot_digest": "a" * 64,
        "snapshot_digest_sha256": "a" * 64,
        "snapshot_digest_pending_adapter": False,
        "persistence_ack_required": True,
        "cancellation_observed_at_snapshot_cutoff": cancellation_observed,
        "cancellation_checkpoint_requested": cancellation_observed,
        "expected_acknowledged_channel_cursor_ranges": {
            "exact_reporting": [generation, generation + 1]
        },
    }
    if checkpoint_due:
        boundary["checkpoint_coordinate"] = f"generation:{generation + 1}:tick:0"
        boundary["checkpoint_projection"] = copy.deepcopy(post)
        boundary["observer_continuation"] = _continuation()
        boundary["expected_acknowledged_channel_cursor_ranges"][
            "checkpoint_generation"
        ] = [generation + 1, generation + 2]
    return boundary


def _failed_result(owner: "_DispatchOwner", message: str) -> dict[str, Any]:
    return {
        "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
        "decision": "ARCHB-CPU-D/FIRST-AUTHORITATIVE-OWNERSHIP-BLOCKER",
        "run_id": owner.run_id,
        "controller_id": owner.controller_id,
        "owner_epoch": owner.owner_epoch,
        "executed": False,
        "native_owner": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "terminal_status": "failed",
        "start_generation": owner.start_generation,
        "requested_generations": 1,
        "completed_generations": 0,
        "trajectory_commit_frontier": owner.start_generation - 1,
        "snapshot_frontier": owner.start_generation - 1,
        "active_generation": owner.start_generation,
        "controller_invocations": 1,
        "trajectory_dispatch_count": 1,
        "python_generation_dispatches": 0,
        "boundaries": [],
        "root_failure": {
            "error_sequence": 1,
            "error_code": "snapshot_persistence_failed",
            "origin": "FlatCpuArchitectureBControllerV1",
            "message": message,
            "run_id": owner.run_id,
            "controller_id": owner.controller_id,
            "owner_epoch": owner.owner_epoch,
            "active_generation": owner.start_generation,
            "owner_integrity": "suspect_and_sealed",
            "retry_disposition": (
                "new_owner_from_last_durable_complete_boundary_only"
            ),
        },
        "success_marker_allowed": False,
    }


class _DispatchOwner:
    def __init__(
        self,
        *,
        start_generation: int = 2,
        initial_boundary_ordinal: int = 2,
        schedule_origin: int = 0,
        cancelled: bool = False,
    ) -> None:
        self.run_id = RUN_ID
        self.controller_id = CURRENT_CONTROLLER_ID
        self.owner_epoch = CURRENT_OWNER_EPOCH
        self.start_generation = start_generation
        self.initial_boundary_ordinal = initial_boundary_ordinal
        self.schedule_origin = schedule_origin
        self.cancelled = cancelled
        self.status = "initialized"
        self.run_started = False
        self.sealed = False
        self.dispatch_started = threading.Event()
        self.release_dispatch = threading.Event()
        self.dispatch_calls: list[tuple[Any, ...]] = []
        self.validation_acknowledgements = 0
        self.validation_digest = ""
        self.persistence_failures: list[str] = []
        self.result: dict[str, Any] | None = None

    def controller_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "eiel_native_architecture_b_controller_manifest_v1",
            "run_id": self.run_id,
            "controller_id": self.controller_id,
            "owner_epoch": self.owner_epoch,
            "controller_protocol_version": NATIVE_CONTROLLER_PROTOCOL_VERSION,
            "controller_protocol_schema_version": (
                NATIVE_CONTROLLER_PROTOCOL_SCHEMA_VERSION
            ),
            "controller_protocol_capabilities": {
                name: True
                for name in NATIVE_PHASE4_REQUIRED_CAPABILITIES
            },
            "status": self.status,
            "run_started": self.run_started,
            "sealed": self.sealed,
            "native_owner_private": True,
            "python_callback_entry_points": 0,
            "persistence_mailbox_required": True,
        }

    def run_generations(
        self,
        start_generation: int,
        generation_count: int,
        steps: int,
        births: int,
        schedule: dict[str, Any],
    ) -> dict[str, Any]:
        self.dispatch_calls.append(
            (start_generation, generation_count, steps, births, schedule)
        )
        self.run_started = True
        self.status = "generation_start"
        self.dispatch_started.set()
        assert self.release_dispatch.wait(2)
        if self.persistence_failures:
            self.status = "failed"
            self.sealed = True
            self.result = _failed_result(self, self.persistence_failures[0])
            return self.result

        completed = 1 if self.cancelled else generation_count
        report_every = int(schedule["report_every_generations"])
        checkpoint_on_cancel = bool(schedule.get("checkpoint_on_cancel", False))
        boundaries = [
            _boundary(
                self,
                generation=start_generation + offset,
                sequence=self.initial_boundary_ordinal + offset + 1,
                final=(offset + 1 == generation_count),
                report_every=report_every,
                checkpoint_on_cancel=checkpoint_on_cancel,
                checkpoint_due=(
                    self.cancelled
                    and checkpoint_on_cancel
                    and offset + 1 == completed
                ),
                cancellation_observed=(
                    self.cancelled and offset + 1 == completed
                ),
            )
            for offset in range(completed)
            if report_every == 1 or offset + 1 == generation_count
        ]
        projection_bytes = sum(
            row["generation_end_reporting_frame_tensor_bytes"]
            + row["post_turnover_projection_tensor_bytes"]
            for row in boundaries
        )
        terminal_status = "cancelled" if self.cancelled else "awaiting_snapshot_validation"
        self.status = terminal_status
        self.sealed = self.cancelled
        self.result = {
            "schema_version": NATIVE_RESULT_SCHEMA_VERSION,
            "decision": "ARCHB-CPU-VERTICAL-SLICE-NATIVE-RUN-MANY",
            "run_id": self.run_id,
            "controller_id": self.controller_id,
            "owner_epoch": self.owner_epoch,
            "controller_protocol_version": NATIVE_CONTROLLER_PROTOCOL_VERSION,
            "executed": True,
            "native_owner": True,
            "hidden_fallback": False,
            "runtime_jit": False,
            "terminal_status": terminal_status,
            "start_generation": start_generation,
            "requested_generations": generation_count,
            "completed_generations": completed,
            "trajectory_commit_frontier": start_generation + completed - 1,
            "snapshot_frontier": start_generation + completed - 1,
            "next_generation": start_generation + completed,
            "next_tick": 0,
            "controller_invocations": 1,
            "python_generation_dispatches": 0,
            "native_generation_transactions": completed,
            "turnover_commits": completed,
            "boundary_snapshot_count": len(boundaries),
            "projection_export_count": len(boundaries) * 2,
            "observer_live_owner_exports": 0,
            "persistence_acknowledgement_count": len(boundaries),
            "projection_tensor_bytes": projection_bytes,
            "boundary_wall_time_ns": 1,
            "boundaries": boundaries,
            "final_snapshot_sequence": self.initial_boundary_ordinal + completed,
            "success_marker_allowed": False,
            "promotion_result_allowed": False,
        }
        if self.cancelled:
            snapshot_sequence = self.initial_boundary_ordinal + completed
            snapshot_id = (
                f"{self.controller_id}:owner:{self.owner_epoch}:"
                f"boundary:{snapshot_sequence}"
            )
            safe_generation = start_generation + completed
            safe_cursor = f"generation:{safe_generation}:tick:0"
            snapshot_digest = "a" * 64
            self.result.update(
                {
                    "cancel_request_id": "cancel-at-safe-boundary",
                    "cancel_request_observation_cursor": safe_cursor,
                    "final_safe_cursor": safe_cursor,
                    "output_ack_frontier": start_generation + completed - 1,
                    "cancellation_checkpoint_requested": True,
                    "cancellation_checkpoint_persisted": True,
                    "cancelled_after_completed_generation": (
                        start_generation + completed - 1
                    ),
                    "cancelled_at_generation_boundary": start_generation + completed,
                    "cancelled_snapshot_sequence": snapshot_sequence,
                    "cancelled_snapshot_id": snapshot_id,
                    "cancelled_snapshot_completion_disposition": (
                        "cancelled_no_success_or_promotion"
                    ),
                    "cancelled_snapshot_durable": True,
                    "cancel_request_receipt": {
                        "request_id": "cancel-at-safe-boundary",
                        "controller_id": self.controller_id,
                        "owner_epoch": self.owner_epoch,
                        "disposition": "accepted",
                        "first_cancel_request_id": "cancel-at-safe-boundary",
                        "cancel_sequence": 1,
                        "state_observed": "running",
                        "terminal_status_changed": False,
                    },
                    "cancel_observation_receipt": {
                        "schema_version": (
                            "eiel_native_architecture_b_"
                            "cancellation_observation_receipt_v1"
                        ),
                        "request_id": "cancel-at-safe-boundary",
                        "reason": "operator_requested",
                        "controller_id": self.controller_id,
                        "owner_epoch": self.owner_epoch,
                        "cancel_sequence": 1,
                        "observation_cursor": safe_cursor,
                        "observed_after_completed_generation": (
                            start_generation + completed - 1
                        ),
                        "boundary_ordinal": snapshot_sequence,
                        "snapshot_sequence": snapshot_sequence,
                        "snapshot_id": snapshot_id,
                        "snapshot_digest_sha256": snapshot_digest,
                        "checkpoint_requested": True,
                        "checkpoint_persisted": True,
                    },
                    "cancel_terminal_receipt": {
                        "schema_version": (
                            "eiel_native_architecture_b_"
                            "cancellation_terminal_receipt_v1"
                        ),
                        "request_id": "cancel-at-safe-boundary",
                        "controller_id": self.controller_id,
                        "owner_epoch": self.owner_epoch,
                        "terminal_status": "cancelled",
                        "request_observation_cursor": safe_cursor,
                        "final_safe_cursor": safe_cursor,
                        "trajectory_commit_frontier": (
                            start_generation + completed - 1
                        ),
                        "snapshot_frontier": start_generation + completed - 1,
                        "output_ack_frontier": start_generation + completed - 1,
                        "snapshot_sequence": snapshot_sequence,
                        "snapshot_id": snapshot_id,
                        "snapshot_digest_sha256": snapshot_digest,
                        "cancellation_checkpoint_requested": True,
                        "cancellation_checkpoint_persisted": True,
                        "success_marker_allowed": False,
                        "promotion_result_allowed": False,
                    },
                }
            )
        return self.result

    def prepare_snapshot_validation(self, digest: str) -> dict[str, Any]:
        self.validation_acknowledgements += 1
        self.validation_digest = digest
        self.status = "awaiting_snapshot_validation_settlement"
        assert self.result is not None
        return {
            "schema_version": (
                "eiel_native_architecture_b_snapshot_validation_prepare_receipt_v1"
            ),
            "run_id": self.run_id,
            "controller_id": self.controller_id,
            "owner_epoch": self.owner_epoch,
            "prepared": True,
            "disposition": "accepted",
            "terminal_status": "awaiting_snapshot_validation_settlement",
            "trajectory_commit_frontier": self.result["trajectory_commit_frontier"],
            "proposed_snapshot_frontier": self.result["snapshot_frontier"],
            "final_snapshot_sequence": self.result["final_snapshot_sequence"],
            "validated_source_projection_digest": digest,
            "settlement_token": "4" * 64,
            "prepared_receipt_digest_sha256": "5" * 64,
            "success_marker_allowed": False,
        }

    def commit_snapshot_validation(self, request: dict[str, Any]) -> dict[str, Any]:
        assert request["settlement_token"] == "4" * 64
        assert request["prepared_receipt_digest_sha256"] == "5" * 64
        self.status = "run_complete"
        self.sealed = True
        assert self.result is not None
        self.result["terminal_status"] = "run_complete"
        self.result["success_marker_allowed"] = True
        return {
            "schema_version": NATIVE_SNAPSHOT_VALIDATION_RECEIPT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "controller_id": self.controller_id,
            "owner_epoch": self.owner_epoch,
            "acknowledged": True,
            "disposition": "accepted",
            "terminal_status": "run_complete",
            "trajectory_commit_frontier": self.result["trajectory_commit_frontier"],
            "snapshot_frontier": self.result["snapshot_frontier"],
            "final_snapshot_sequence": self.result["final_snapshot_sequence"],
            "validated_source_projection_digest": self.validation_digest,
            "settlement_token": request["settlement_token"],
            "prepared_receipt_digest_sha256": request[
                "prepared_receipt_digest_sha256"
            ],
            "success_marker_allowed": True,
        }

    def acknowledge_snapshot_validation(self, digest: str) -> dict[str, Any]:
        return self.prepare_snapshot_validation(digest)

    def fail_snapshot_persistence(self, message: str) -> dict[str, Any]:
        self.persistence_failures.append(message)
        self.release_dispatch.set()
        return {}

    def fail_snapshot_validation(self, message: str) -> dict[str, Any]:
        self.status = "failed"
        self.sealed = True
        self.result = _failed_result(self, message)
        return self.result

    def read_snapshots(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("fake persistence loop owns this focused test")

    def acknowledge_snapshot_persistence(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("fake persistence loop owns this focused test")

    def request_cancel(self, *_args: Any) -> dict[str, Any]:
        raise AssertionError("not used")

    def final_result(self) -> dict[str, Any]:
        if self.result is None or self.status not in {"run_complete", "cancelled", "failed"}:
            return {"terminal": False, "status": self.status}
        return copy.deepcopy(self.result)


def _controller(owner: _DispatchOwner) -> NativeRunController:
    return NativeRunController(
        owner,
        start_generation=owner.start_generation,
        default_steps_per_generation=4,
        default_max_births=0,
        safe_default_decision={"supported": True},
        extension_path="/fake/architecture-b.so",
        run_id=owner.run_id,
        persistence_mailbox_required=True,
        initial_boundary_ordinal=owner.initial_boundary_ordinal,
        schedule_origin_generation=owner.schedule_origin,
        phase4_capabilities=NATIVE_PHASE4_REQUIRED_CAPABILITIES,
    )


def _persistence_result(
    *,
    terminal: str,
    last_sequence: int,
    checkpoint_paths: list[str] | None = None,
) -> dict[str, Any]:
    paths = list(checkpoint_paths or [])
    return {
        "schema_version": "eiel_architecture_b_persistence_loop_result_v1",
        "stop_reason": (
            "terminal_state_seen" if terminal != "none" else "run_thread_finished"
        ),
        "terminal_state_seen": terminal,
        "snapshots_seen": 1,
        "progress_only_snapshots": 0,
        "skipped_snapshot_sequences": 0,
        "persistence_acknowledgements": 1,
        "identical_acknowledgement_retries": 0,
        "checkpoints_persisted": len(paths),
        "architecture_b_checkpoint_bundle_paths": paths,
        "last_snapshot_sequence": last_sequence,
        "last_progress": None,
    }


def _assert_no_dispatch_thread() -> None:
    assert not any(
        thread.name.startswith("eiel-archb-native-") and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_facade_runs_one_native_dispatch_and_persistence_loop_on_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _DispatchOwner()
    caller_ident = threading.get_ident()
    loop_calls: list[int] = []

    class PersistenceLoop:
        def __init__(self, current_owner: Any, runner: Any, **kwargs: Any) -> None:
            assert current_owner is owner
            assert kwargs["initial_after_snapshot_sequence"] == 2

        def run(self) -> dict[str, Any]:
            loop_calls.append(threading.get_ident())
            assert owner.dispatch_started.wait(1)
            owner.release_dispatch.set()
            return _persistence_result(terminal="none", last_sequence=3)

    monkeypatch.setattr(run_controller_module, "ArchitectureBPersistenceLoop", PersistenceLoop)
    result = _controller(owner).run_with_persistence(
        SimpleNamespace(run_dir="/fake"),
        1,
        snapshot_schedule=_schedule(),
        join_timeout_seconds=1,
    )

    assert result["terminal_status"] == "run_complete"
    assert result["success_marker_allowed"] is True
    assert len(owner.dispatch_calls) == 1
    assert loop_calls == [caller_ident]
    assert owner.validation_acknowledgements == 1
    _assert_no_dispatch_thread()


def test_unobserved_terminal_only_requires_explicit_final_only_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TerminalOnlyProbeOwner(_DispatchOwner):
        def run_generations(self, *args: Any) -> dict[str, Any]:
            self.dispatch_calls.append(tuple(args))
            self.dispatch_started.set()
            raise RuntimeError("terminal-only dispatch probe")

    owner = TerminalOnlyProbeOwner(
        start_generation=0,
        initial_boundary_ordinal=0,
    )

    class PersistenceLoop:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            assert kwargs["snapshot_observer"] is None

        def run(self) -> dict[str, Any]:
            assert owner.dispatch_started.wait(1)
            return _persistence_result(terminal="none", last_sequence=1)

    monkeypatch.setattr(
        run_controller_module,
        "ArchitectureBPersistenceLoop",
        PersistenceLoop,
    )
    schedule = _schedule(
        report_every=0,
        explicit_next_generations=[1],
    )
    with pytest.raises(
        NativeRunControllerError,
        match="terminal-only dispatch probe",
    ):
        _controller(owner).run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            1,
            snapshot_schedule=schedule,
            result_retention="terminal_only",
            allow_unobserved_terminal_only=True,
            join_timeout_seconds=1,
        )

    assert owner.dispatch_calls[0][-1] == "terminal_only"
    _assert_no_dispatch_thread()

    rejected_owner = _DispatchOwner(start_generation=0, initial_boundary_ordinal=0)
    with pytest.raises(
        NativeRunControllerError,
        match="unobserved terminal_only retention requires explicit opt-in",
    ):
        _controller(rejected_owner).run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            1,
            snapshot_schedule=schedule,
            result_retention="terminal_only",
        )


def test_malformed_settlement_receipt_latches_durable_failed_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MalformedSettlementOwner(_DispatchOwner):
        def prepare_snapshot_validation(self, _digest: str) -> dict[str, Any]:
            self.validation_acknowledgements += 1
            assert self.status == "awaiting_snapshot_validation"
            return {"schema_version": "malformed-settlement-receipt"}

    owner = MalformedSettlementOwner()

    class PersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            assert owner.dispatch_started.wait(1)
            owner.release_dispatch.set()
            return _persistence_result(terminal="none", last_sequence=3)

    monkeypatch.setattr(run_controller_module, "ArchitectureBPersistenceLoop", PersistenceLoop)
    controller = _controller(owner)
    with pytest.raises(NativeRunControllerError, match="prepare receipt fields mismatch"):
        controller.run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            1,
            snapshot_schedule=_schedule(),
            join_timeout_seconds=1,
        )

    final = controller.final_result()
    assert final["terminal_status"] == "failed"
    assert final["success_marker_allowed"] is False
    assert owner.status == "failed"
    assert owner.sealed is True
    assert owner.validation_acknowledgements == 1
    _assert_no_dispatch_thread()


def test_facade_allows_only_safe_boundary_partial_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _DispatchOwner(
        start_generation=0,
        initial_boundary_ordinal=0,
        cancelled=True,
    )

    class PersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            assert owner.dispatch_started.wait(1)
            owner.release_dispatch.set()
            return _persistence_result(
                terminal="cancelled",
                last_sequence=1,
                checkpoint_paths=["/runtime/cancel-boundary-1"],
            )

    monkeypatch.setattr(run_controller_module, "ArchitectureBPersistenceLoop", PersistenceLoop)
    result = _controller(owner).run_with_persistence(
        SimpleNamespace(run_dir="/fake"),
        2,
        snapshot_schedule=_schedule(report_every=1),
        join_timeout_seconds=1,
    )
    assert result["terminal_status"] == "cancelled"
    assert result["completed_generations"] == 1
    assert result["requested_generations"] == 2
    assert result["success_marker_allowed"] is False
    assert owner.validation_acknowledgements == 0
    _assert_no_dispatch_thread()


def test_facade_uses_absolute_safe_boundary_after_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _DispatchOwner(
        start_generation=2,
        initial_boundary_ordinal=2,
        cancelled=True,
    )

    class PersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            assert owner.dispatch_started.wait(1)
            owner.release_dispatch.set()
            return _persistence_result(
                terminal="cancelled",
                last_sequence=3,
                checkpoint_paths=["/runtime/cancel-boundary-3"],
            )

    monkeypatch.setattr(run_controller_module, "ArchitectureBPersistenceLoop", PersistenceLoop)
    result = _controller(owner).run_with_persistence(
        SimpleNamespace(run_dir="/fake"),
        2,
        snapshot_schedule=_schedule(report_every=1),
        join_timeout_seconds=1,
    )
    assert result["completed_generations"] == 1
    assert result["cancelled_at_generation_boundary"] == 3
    assert result["next_generation"] == 3
    _assert_no_dispatch_thread()


def test_persistence_setup_failure_starts_no_native_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _DispatchOwner()

    class InvalidPersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ValueError("poll_timeout_ms must be positive")

    monkeypatch.setattr(
        run_controller_module,
        "ArchitectureBPersistenceLoop",
        InvalidPersistenceLoop,
    )
    controller = _controller(owner)
    with pytest.raises(NativeRunControllerError, match="poll_timeout_ms must be positive"):
        controller.run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            1,
            snapshot_schedule=_schedule(),
            poll_timeout_ms=0,
            join_timeout_seconds=1,
        )
    assert owner.dispatch_calls == []
    assert owner.dispatch_started.is_set() is False
    assert controller.manifest()["trajectory_dispatch_count"] == 0
    _assert_no_dispatch_thread()


@pytest.mark.parametrize("explicit", [[2], [5]])
def test_explicit_checkpoint_schedule_rejects_past_or_beyond_target_before_dispatch(
    explicit: list[int],
) -> None:
    owner = _DispatchOwner(start_generation=2, initial_boundary_ordinal=2)
    controller = _controller(owner)
    with pytest.raises(NativeRunControllerError, match="outside the requested run"):
        controller.run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            2,
            snapshot_schedule=_schedule(explicit_next_generations=explicit),
        )
    assert owner.dispatch_calls == []
    assert owner.dispatch_started.is_set() is False
    assert controller.manifest()["trajectory_dispatch_count"] == 0
    _assert_no_dispatch_thread()


def test_persistence_failure_propagates_without_success_or_thread_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _DispatchOwner()

    class FailingPersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            assert owner.dispatch_started.wait(1)
            owner.fail_snapshot_persistence("RuntimeError: durable persistence failed")
            raise RuntimeError("durable persistence failed")

    monkeypatch.setattr(
        run_controller_module,
        "ArchitectureBPersistenceLoop",
        FailingPersistenceLoop,
    )
    controller = _controller(owner)
    expected_root = "RuntimeError: durable persistence failed"
    with pytest.raises(NativeRunControllerError) as raised:
        controller.run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            1,
            snapshot_schedule=_schedule(),
            join_timeout_seconds=1,
        )
    assert str(raised.value) == f"EIEL-NATIVE-RUN-CONTROLLER-FAILED: {expected_root}"
    final = controller.final_result()
    assert final["root_failure"]["message"] == expected_root
    assert final["terminal_status"] == "failed"
    assert final["success_marker_allowed"] is False
    assert owner.persistence_failures == [expected_root]
    assert owner.validation_acknowledgements == 0
    assert owner.status == "failed"
    _assert_no_dispatch_thread()


def test_checkpoint_on_cancel_is_forwarded_and_remains_non_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = _DispatchOwner(
        start_generation=0,
        initial_boundary_ordinal=0,
        cancelled=True,
    )

    class PersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            assert owner.dispatch_started.wait(1)
            owner.release_dispatch.set()
            return _persistence_result(
                terminal="cancelled",
                last_sequence=1,
                checkpoint_paths=["/runtime/checkpoint-on-cancel"],
            )

    monkeypatch.setattr(
        run_controller_module,
        "ArchitectureBPersistenceLoop",
        PersistenceLoop,
    )
    result = _controller(owner).run_with_persistence(
        SimpleNamespace(run_dir="/fake"),
        2,
        snapshot_schedule=_schedule(
            report_every=1,
            checkpoint_on_cancel=True,
        ),
        join_timeout_seconds=1,
    )

    forwarded_schedule = owner.dispatch_calls[0][-1]
    assert forwarded_schedule["checkpoint_on_cancel"] is True
    assert result["terminal_status"] == "cancelled"
    assert result["cancellation_checkpoint_requested"] is True
    assert result["cancellation_checkpoint_persisted"] is True
    assert result["success_marker_allowed"] is False
    assert result["promotion_result_allowed"] is False
    assert "snapshot_validation_receipt" not in result
    assert result["persistence_loop_result"][
        "architecture_b_checkpoint_bundle_paths"
    ] == ["/runtime/checkpoint-on-cancel"]
    boundary = result["boundaries"][0]
    assert boundary["due_mask"]["checkpoint"] is True
    assert boundary["schedule_state"]["checkpoint_on_cancel"] is True
    assert boundary["checkpoint_coordinate"] == "generation:1:tick:0"
    assert boundary["observer_continuation"] == _continuation()
    assert owner.validation_acknowledgements == 0
    _assert_no_dispatch_thread()


def test_cancelled_result_schema_rejects_public_output_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidCancelledOwner(_DispatchOwner):
        def run_generations(self, *args: Any) -> dict[str, Any]:
            result = super().run_generations(*args)
            result["public_output_finalized"] = True
            return result

    owner = InvalidCancelledOwner(
        start_generation=0,
        initial_boundary_ordinal=0,
        cancelled=True,
    )

    class PersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            assert owner.dispatch_started.wait(1)
            owner.release_dispatch.set()
            return _persistence_result(terminal="cancelled", last_sequence=1)

    monkeypatch.setattr(
        run_controller_module,
        "ArchitectureBPersistenceLoop",
        PersistenceLoop,
    )
    controller = _controller(owner)
    with pytest.raises(
        NativeRunControllerError,
        match=r"unexpected=\['public_output_finalized'\]",
    ):
        controller.run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            2,
            snapshot_schedule=_schedule(report_every=1),
            join_timeout_seconds=1,
        )
    with pytest.raises(NativeRunControllerError, match="public_output_finalized"):
        controller.final_result()
    assert owner.status == "cancelled"
    assert owner.validation_acknowledgements == 0
    _assert_no_dispatch_thread()


def test_malformed_cancel_receipt_remains_rejected_by_final_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidReceiptOwner(_DispatchOwner):
        def run_generations(self, *args: Any) -> dict[str, Any]:
            result = super().run_generations(*args)
            result["cancel_observation_receipt"]["unexpected"] = True
            return result

    owner = InvalidReceiptOwner(
        start_generation=0,
        initial_boundary_ordinal=0,
        cancelled=True,
    )

    class PersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            assert owner.dispatch_started.wait(1)
            owner.release_dispatch.set()
            return _persistence_result(
                terminal="cancelled",
                last_sequence=1,
                checkpoint_paths=["/runtime/cancel-malformed-receipt"],
            )

    monkeypatch.setattr(
        run_controller_module,
        "ArchitectureBPersistenceLoop",
        PersistenceLoop,
    )
    controller = _controller(owner)
    with pytest.raises(
        NativeRunControllerError,
        match="cancellation observation receipt fields mismatch",
    ):
        controller.run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            2,
            snapshot_schedule=_schedule(report_every=1),
            join_timeout_seconds=1,
        )
    with pytest.raises(
        NativeRunControllerError,
        match="cancellation observation receipt fields mismatch",
    ):
        controller.final_result()
    _assert_no_dispatch_thread()


def test_commit_receipt_is_consumed_exactly_before_facade_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidCommitReceiptOwner(_DispatchOwner):
        def commit_snapshot_validation(self, request: dict[str, Any]) -> dict[str, Any]:
            receipt = super().commit_snapshot_validation(request)
            receipt["public_output_finalized"] = True
            return receipt

    owner = InvalidCommitReceiptOwner()

    class PersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            assert owner.dispatch_started.wait(1)
            owner.release_dispatch.set()
            return _persistence_result(terminal="none", last_sequence=3)

    monkeypatch.setattr(
        run_controller_module,
        "ArchitectureBPersistenceLoop",
        PersistenceLoop,
    )
    controller = _controller(owner)
    with pytest.raises(NativeRunControllerError, match="receipt fields mismatch"):
        controller.run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            1,
            snapshot_schedule=_schedule(),
            join_timeout_seconds=1,
        )
    final = controller.final_result()
    assert final["terminal_status"] == "failed"
    assert final["success_marker_allowed"] is False
    assert "snapshot_validation_receipt" not in final
    _assert_no_dispatch_thread()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"unexpected": True}, "fields mismatch"),
        ({"cancel_sequence": 2}, "sequence is not the first latch"),
        ({"first_cancel_request_id": "other"}, "acceptance is inconsistent"),
        ({"terminal_status_changed": True}, "must be False"),
    ],
)
def test_cancellation_receipt_is_exact_and_does_not_dispatch_trajectory(
    updates: dict[str, Any],
    message: str,
) -> None:
    class CancellationReceiptOwner(_DispatchOwner):
        def request_cancel(self, request: dict[str, Any]) -> dict[str, Any]:
            receipt = {
                "request_id": request["request_id"],
                "controller_id": self.controller_id,
                "owner_epoch": self.owner_epoch,
                "disposition": "accepted",
                "first_cancel_request_id": request["request_id"],
                "cancel_sequence": 1,
                "state_observed": "running",
                "terminal_status_changed": False,
            }
            receipt.update(updates)
            return receipt

    owner = CancellationReceiptOwner()
    controller = _controller(owner)
    owner.status = "generation_start"
    owner.run_started = True

    with pytest.raises(NativeRunControllerError, match=message):
        controller.request_cancel("cancel-exact")
    assert controller.manifest()["trajectory_dispatch_count"] == 0
    assert owner.status == "generation_start"


def test_repeated_cancellation_receipt_cannot_replace_first_request() -> None:
    class RepeatedCancellationOwner(_DispatchOwner):
        def __init__(self) -> None:
            super().__init__()
            self.requests: list[dict[str, Any]] = []

        def request_cancel(self, request: dict[str, Any]) -> dict[str, Any]:
            self.requests.append(copy.deepcopy(request))
            first = len(self.requests) == 1
            return {
                "request_id": request["request_id"],
                "controller_id": self.controller_id,
                "owner_epoch": self.owner_epoch,
                "disposition": "accepted" if first else "already_requested",
                "first_cancel_request_id": "cancel-first" if first else "changed-first",
                "cancel_sequence": 1,
                "state_observed": "running",
                "terminal_status_changed": False,
            }

    owner = RepeatedCancellationOwner()
    controller = _controller(owner)
    owner.status = "generation_start"
    owner.run_started = True

    accepted = controller.request_cancel(
        "cancel-first",
        reason="signal_received",
    )
    assert accepted["disposition"] == "accepted"
    with pytest.raises(NativeRunControllerError, match="changed the first request"):
        controller.request_cancel("cancel-second")
    assert owner.requests[0] == {
        "api_version": 1,
        "request_id": "cancel-first",
        "expected_controller_id": owner.controller_id,
        "expected_owner_epoch": owner.owner_epoch,
        "reason": "signal_received",
    }
    assert controller.manifest()["trajectory_dispatch_count"] == 0


def test_native_terminal_result_is_the_only_cancelled_status_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ContradictoryFinalOwner(_DispatchOwner):
        def final_result(self) -> dict[str, Any]:
            result = super().final_result()
            if result.get("terminal_status") == "cancelled":
                result["terminal_status"] = "failed"
            return result

    owner = ContradictoryFinalOwner(
        start_generation=0,
        initial_boundary_ordinal=0,
        cancelled=True,
    )

    class PersistenceLoop:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def run(self) -> dict[str, Any]:
            assert owner.dispatch_started.wait(1)
            owner.release_dispatch.set()
            return _persistence_result(terminal="cancelled", last_sequence=1)

    monkeypatch.setattr(
        run_controller_module,
        "ArchitectureBPersistenceLoop",
        PersistenceLoop,
    )
    with pytest.raises(
        NativeRunControllerError,
        match="authoritative terminal status mismatch",
    ):
        _controller(owner).run_with_persistence(
            SimpleNamespace(run_dir="/fake"),
            2,
            snapshot_schedule=_schedule(report_every=1),
            join_timeout_seconds=1,
        )
    assert owner.status == "cancelled"
    assert owner.validation_acknowledgements == 0
    _assert_no_dispatch_thread()
