from __future__ import annotations

import inspect
import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import emergence_contracts as contracts
from emergence_contracts import (
    ActionBatchV1,
    AdaptationComponentV1,
    AgentComponentV1,
    ArtifactId,
    ArtifactIntegrityStatusV1,
    AssemblyId,
    BackendId,
    BranchId,
    CanonicalPayloadV1,
    CellId,
    CheckpointId,
    CommandId,
    ComponentCompatibilityV1,
    ComponentDescriptorV1,
    ComponentId,
    CompositeCheckpointViewV1,
    ContentDigest,
    ContractError,
    ControlCommandV1,
    ControlTargetV1,
    EvidenceEvaluatorV1,
    ExecutionScheduleV1,
    ExperimentCellV1,
    ExperimentExecutorV1,
    ExperimentId,
    ExperimentPlanV1,
    LeaseId,
    LegacyGenerationProjectionV1,
    ModuleId,
    NamedDigestV1,
    NativeBindingV1,
    PairedComparisonV1,
    ParticipantId,
    RNGAuthorityV1,
    RNGLeaseDescriptorV1,
    RNGLeaseV1,
    RunArtifactManifestV1,
    RunArtifactRefV1,
    RunCompletionStatusV1,
    RunCursorV1,
    RunId,
    RuntimeEventV1,
    RuntimeProviderV1,
    RuntimeServicesV1,
    SchemaVersion,
    SnapshotParticipantV1,
    StateMaterialRoleV1,
    StateParticipantDescriptorV1,
    SystemAssemblyV1,
    TelemetrySinkV1,
    TurnoverScheduleV1,
    WorldComponentV1,
    canonical_json_bytes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _digest(character: str = "a") -> ContentDigest:
    return ContentDigest("sha256", character * 64)


def _payload(name: str = "test_payload_v1", value: object | None = None) -> CanonicalPayloadV1:
    return CanonicalPayloadV1.from_value(
        SchemaVersion(name), {"value": 1} if value is None else value
    )


def _artifact(identifier: str, character: str = "a") -> RunArtifactRefV1:
    return RunArtifactRefV1(
        artifact_id=ArtifactId(identifier),
        artifact_kind="test-artifact",
        artifact_schema_version=SchemaVersion("test_artifact_v1"),
        locator=f"artifact:{identifier}",
        content_digest=_digest(character),
        size_bytes=10,
    )


def _component(identifier: str, module: str) -> ComponentDescriptorV1:
    return ComponentDescriptorV1(
        component_id=ComponentId(identifier),
        module_id=ModuleId(module),
        semantic_contract_version=SchemaVersion("semantic_v1"),
        implementation_version=SchemaVersion("implementation_v1"),
        capabilities=("describe",),
        compatibility=ComponentCompatibilityV1(
            semantic_contract_versions=(SchemaVersion("semantic_v1"),)
        ),
    )


def test_registry_completes_all_frozen_contract_ids_with_owner_and_version() -> None:
    expected = ["C-001"] + [f"C-{number:03d}" for number in range(10, 171, 10)]
    assert [item.contract_id for item in contracts.CONTRACT_REGISTRY_V1] == expected
    assert set(contracts.CONTRACT_REGISTRY_BY_ID_V1) == set(expected)
    assert len(contracts.CONTRACT_REGISTRY_V1) == 18
    for definition in contracts.CONTRACT_REGISTRY_V1:
        assert definition.owner
        assert definition.api_version.value == definition.contract_id.lower().replace("-", "") + "_v1"
        assert definition.public_types
        assert all(hasattr(contracts, name) for name in definition.public_types)
    with pytest.raises(TypeError):
        contracts.CONTRACT_REGISTRY_BY_ID_V1["C-999"] = object()  # type: ignore[index]


def test_package_version_and_clean_import_remain_inert() -> None:
    command = [
        sys.executable,
        "-c",
        (
            "import json,sys; import emergence_contracts as c; "
            "forbidden={'engine','tools','experiments','experiments_shared','torch','numpy'}; "
            "print(json.dumps({'version':c.__version__,'loaded':sorted(forbidden & set(sys.modules))}))"
        ),
    ]
    completed = subprocess.run(
        command, cwd=REPO_ROOT, check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout) == {"version": "0.2.0", "loaded": []}


def test_canonical_payload_round_trip_and_noncanonical_rejection() -> None:
    payload = _payload(value={"z": 2, "é": [1, True]})
    assert payload == CanonicalPayloadV1.from_dict(payload.to_dict())
    assert payload.canonical_bytes == canonical_json_bytes({"z": 2, "é": [1, True]})
    with pytest.raises(ContractError):
        CanonicalPayloadV1(SchemaVersion("payload_v1"), b'{"b": 1}')
    with pytest.raises(FrozenInstanceError):
        payload.canonical_bytes = b"{}"  # type: ignore[misc]


def test_schedule_and_cursor_round_trip_preserve_absolute_tick_semantics() -> None:
    schedule = ExecutionScheduleV1(
        total_ticks=12,
        start_next_tick=3,
        runtime_partition_ticks=4,
        science_sample_interval_ticks=3,
        report_every_ticks=6,
        checkpoint_every_ticks=12,
        turnover_schedule=TurnoverScheduleV1((5, 11)),
        legacy_projection=LegacyGenerationProjectionV1(3, 4),
    )
    assert ExecutionScheduleV1.from_dict(schedule.to_dict()) == schedule
    cursor = RunCursorV1(total_ticks=12, next_tick=12)
    assert RunCursorV1.from_dict(cursor.to_dict()) == cursor
    assert cursor.complete
    with pytest.raises(ContractError):
        RunCursorV1(total_ticks=12, next_tick=13)
    with pytest.raises(ContractError):
        ExecutionScheduleV1(total_ticks=12, legacy_projection=LegacyGenerationProjectionV1(2, 4))


def test_component_descriptor_round_trip_is_exact() -> None:
    descriptor = _component("world", "emergence_worldkit")
    assert ComponentDescriptorV1.from_dict(descriptor.to_dict()) == descriptor
    assert descriptor.canonical_bytes() == canonical_json_bytes(descriptor.to_dict())


def test_run_artifact_manifest_round_trip_is_exact() -> None:
    checkpoint = _artifact("checkpoint", "b")
    metric = _artifact("metric", "c")
    event = _artifact("event", "d")
    manifest = RunArtifactManifestV1(
        run_id=RunId("run-1"),
        cell_id=CellId("cell-1"),
        assembly_id=AssemblyId("assembly-1"),
        assembly_digest=_digest("e"),
        runtime_route_id="cpu-patch",
        source_versions=("emergence=0.0.0",),
        config_digests=(NamedDigestV1("runtime", _digest("f")),),
        checkpoint_refs=(checkpoint,),
        metric_refs=(metric,),
        event_refs=(event,),
        completion_status=RunCompletionStatusV1.COMPLETE,
        integrity_status=ArtifactIntegrityStatusV1.VERIFIED,
        output_schema_versions=(SchemaVersion("metrics_v1"),),
    )
    restored = RunArtifactManifestV1.from_dict(manifest.to_dict())
    assert restored == manifest
    assert restored.canonical_bytes() == manifest.canonical_bytes()
    assert restored.content_digest() == manifest.content_digest()


def test_native_binding_and_system_assembly_are_descriptor_only() -> None:
    artifact = _artifact("native", "1")
    native = NativeBindingV1(
        binding_id=ArtifactId("native-binding"),
        semantic_component_id=ComponentId("world"),
        semantic_contract_version=SchemaVersion("semantic_v1"),
        backend_id=BackendId("cpu"),
        abi_version=SchemaVersion("abi_v1"),
        artifact_ref=artifact,
        artifact_sha256=artifact.content_digest,
        manifest_sha256=_digest("2"),
        compiled_source_identity=_digest("3"),
        supported_state_schema_versions=(SchemaVersion("state_v1"),),
        supported_parameter_contract=SchemaVersion("parameters_v1"),
        capability_flags=("fused",),
    )
    assembly = SystemAssemblyV1(
        assembly_id=AssemblyId("assembly"),
        runtime_profile=_component("runtime", "emergence_runtime"),
        world_components=(_component("world", "emergence_worldkit"),),
        agent_component=_component("agent", "emergence_agentkit"),
        adaptation_component=_component("adaptation", "emergence_adaptation"),
        measurement_components=(),
        native_bindings=(native,),
        configuration_fragments=(),
        compatibility_bindings=(),
    )
    assert assembly.identity_digest() == contracts.canonical_sha256(assembly.to_dict())
    assert not hasattr(native, "load")
    assert not hasattr(native, "dispatch")
    assert not hasattr(assembly, "execute")


def test_checkpoint_view_has_stable_participant_order_and_no_payload_copy() -> None:
    payload_ref = _artifact("world-state", "4")
    participant = SnapshotParticipantV1(
        descriptor=StateParticipantDescriptorV1(
            participant_id=ParticipantId("world"),
            module_id=ModuleId("emergence_worldkit"),
            state_schema_version=SchemaVersion("world_state_v1"),
            material_role=StateMaterialRoleV1.AUTHORITATIVE,
            stable_order=1,
        ),
        payload_ref=payload_ref,
        payload_digest=payload_ref.content_digest,
    )
    checkpoint = CompositeCheckpointViewV1(
        checkpoint_id=CheckpointId("checkpoint-1"),
        run_id=RunId("run-1"),
        branch_id=BranchId("branch-1"),
        run_cursor=RunCursorV1(10, 5),
        runtime_state_ref=_artifact("runtime-state", "5"),
        rng_root_snapshot_ref=_artifact("rng-state", "6"),
        control_position=_payload("control_position_v1"),
        participants=(participant,),
        backend_state_ref=None,
        assembly_identity=_digest("7"),
        compatibility_metadata=_payload("checkpoint_compatibility_v1"),
    )
    assert checkpoint.participants[0].payload_ref is payload_ref
    assert not hasattr(checkpoint, "payload")
    with pytest.raises(ContractError):
        replace(checkpoint, participants=(participant, participant))


def test_rng_descriptor_is_zero_draw_metadata_only() -> None:
    descriptor = RNGLeaseDescriptorV1(
        lease_id=LeaseId("lease-1"),
        owner_component_id=ComponentId("adaptation"),
        semantic_domain="mutation",
        stream_identity="mutation/main",
        draw_index=17,
        budget_contract_ref=ArtifactId("rng-budget-v1"),
        device_scope="cpu",
    )
    assert descriptor.draw_index == 17
    assert not hasattr(descriptor, "draw")
    assert not hasattr(descriptor, "generator")


def test_event_and_control_envelopes_fail_closed() -> None:
    cursor = RunCursorV1(10, 3)
    event = RuntimeEventV1(
        run_id=RunId("run"),
        cell_id=CellId("cell"),
        cursor=cursor,
        source_module_id=ModuleId("emergence_runtime"),
        source_component_id=ComponentId("runtime"),
        event_type="heartbeat",
        payload=_payload("event_v1"),
        submission_sequence=2,
    )
    assert event.to_dict()["next_tick_or_event_cursor"]["next_tick"] == 3
    command = ControlCommandV1(
        command_id=CommandId("command"),
        target_run_id=RunId("run"),
        target=ControlTargetV1(boundary_name="safe-boundary"),
        command_type="pause",
        payload=_payload("command_v1"),
        submission_sequence=1,
        expectation_or_reason="operator request",
        idempotency_key=_digest("8"),
    )
    assert command.target.boundary_name == "safe-boundary"
    with pytest.raises(ContractError):
        ControlTargetV1(next_tick=2, boundary_name="also-set")


def test_lab_plan_contains_metadata_not_runtime_implementation() -> None:
    cells = (
        ExperimentCellV1(CellId("a"), 1, "control", AssemblyId("assembly"), _digest("9")),
        ExperimentCellV1(CellId("b"), 1, "treatment", AssemblyId("assembly"), _digest("a")),
    )
    plan = ExperimentPlanV1(
        experiment_id=ExperimentId("experiment"),
        logical_cells=cells,
        paired_comparisons=(PairedComparisonV1(CellId("a"), CellId("b"), "paired-v1"),),
        runtime_requirements=_payload("runtime_requirements_v1"),
        measurement_requirements=_payload("measurement_requirements_v1"),
        output_contract=SchemaVersion("run_artifact_manifest_v1"),
        repeat_parity_plan=_payload("repeat_parity_v1"),
    )
    assert len(plan.logical_cells) == 2
    assert not hasattr(plan, "provider")
    assert not hasattr(plan, "runner")


@pytest.mark.parametrize(
    ("protocol", "methods"),
    [
        (WorldComponentV1, {"descriptor", "configure", "initialize", "observe", "apply_actions", "advance_environment", "state_port", "state_descriptor", "export_state", "import_state", "validate_state"}),
        (AgentComponentV1, {"descriptor", "configure", "instantiate", "act", "apply_world_feedback", "state_port", "export_runtime_state", "import_runtime_state", "state_descriptor"}),
        (AdaptationComponentV1, {"descriptor", "configure", "initialize_population", "active_entities", "agent_blueprints", "record_outcomes", "on_turnover", "lineage_view", "state_port", "export_state", "import_state", "state_descriptor"}),
        (RuntimeServicesV1, {"cursor", "rng_authority", "telemetry_sink", "artifact_output", "control_port", "control_boundaries", "capabilities"}),
        (RuntimeProviderV1, {"capabilities", "validate_assembly", "prepare", "execute", "checkpoint", "resume", "branch", "cancel", "status"}),
        (ExperimentExecutorV1, {"validate_experiment_spec", "compile", "plan_batches", "execute", "resume", "status"}),
        (EvidenceEvaluatorV1, {"ingest", "validate", "compare", "decide", "materialize_report"}),
        (RNGAuthorityV1, {"lease", "root_snapshot_ref", "restore_root_snapshot"}),
        (RNGLeaseV1, {"descriptor", "tracked_handle", "index_snapshot", "assert_budget"}),
        (TelemetrySinkV1, {"emit"}),
    ],
)
def test_protocol_surface_signatures_are_explicit(
    protocol: type[object], methods: set[str]
) -> None:
    assert getattr(protocol, "_is_runtime_protocol", False)
    for method in methods:
        member = getattr(protocol, method)
        signature = inspect.signature(member)
        assert "self" in signature.parameters
        assert signature.return_annotation is not inspect.Signature.empty


def test_distinct_batch_types_prevent_cross_boundary_aliasing() -> None:
    cursor = RunCursorV1(2, 0)
    payload = _payload("batch_v1")
    action = ActionBatchV1(cursor, payload)
    assert not isinstance(action, contracts.ObservationBatchV1)
