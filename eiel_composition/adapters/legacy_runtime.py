"""Dormant runtime, schedule, telemetry, and control compatibility facades.

These adapters project already-authoritative current objects.  Construction and
preparation never create a production lifecycle, translate a campaign batch,
load an artifact, dispatch native code, or draw randomness.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Mapping

from emergence_contracts import (
    BackendId,
    CanonicalPayloadV1,
    CellId,
    CommandId,
    ComponentId,
    ComponentInstanceId,
    ContentDigest,
    ContractError,
    ContractErrorCategory,
    ContractIssueV1,
    ControlCommandV1,
    ControlResultV1,
    ExecutionScheduleV1,
    LegacyGenerationProjectionV1,
    ModuleId,
    PreparedExecutionV1,
    RunCursorV1,
    RunId,
    RuntimeCapabilityProfileV1,
    RuntimeEventV1,
    RuntimeExecutionRequestV1,
    SchemaVersion,
    SystemAssemblyV1,
    ValidationResultV1,
    canonical_sha256,
)


LEGACY_RESIDENT_REQUEST_SCHEMA = SchemaVersion("eiel_resident_batch_request_v1")
LEGACY_CONTROL_EVENT_SCHEMA = SchemaVersion("command_event_v1")
LEGACY_CONTROL_ACK_SCHEMA = SchemaVersion("control_plane_ack_v1")


def _runtime_error(
    category: ContractErrorCategory,
    message: str,
    *,
    details: Mapping[str, object] | None = None,
) -> ContractError:
    return ContractError(
        category,
        owner="eiel_composition.adapters.legacy_runtime",
        message=message,
        details=details,
    )


class LegacyScheduleAdapter:
    """Represent exact legacy G/T authority without inventing tick-native input."""

    @staticmethod
    def from_legacy(
        *,
        generations: int,
        ticks_per_generation: int,
        start_next_tick: int = 0,
        science_sample_interval_ticks: int | None = None,
        report_every_ticks: int | None = None,
        checkpoint_every_ticks: int | None = None,
    ) -> ExecutionScheduleV1:
        projection = LegacyGenerationProjectionV1(
            generations=generations,
            ticks_per_generation=ticks_per_generation,
        )
        # Generation boundaries are represented, not asserted as semantic
        # turnover boundaries.  Current execution remains generation-shaped.
        return ExecutionScheduleV1(
            total_ticks=projection.total_ticks,
            start_next_tick=start_next_tick,
            science_sample_interval_ticks=science_sample_interval_ticks,
            report_every_ticks=report_every_ticks,
            checkpoint_every_ticks=checkpoint_every_ticks,
            turnover_schedule=None,
            legacy_projection=projection,
        )

    @staticmethod
    def from_resident_request(request: object) -> ExecutionScheduleV1:
        return LegacyScheduleAdapter.from_legacy(
            generations=request.generations,
            ticks_per_generation=request.ticks_per_generation,
        )


def _request_identity(request: RuntimeExecutionRequestV1) -> ContentDigest:
    return canonical_sha256(
        {
            "run_id": str(request.run_id),
            "cell_id": str(request.cell_id),
            "schedule": request.schedule.to_dict(),
            "assembly_id": str(request.assembly_id),
            "config_digest": request.config_digest.to_dict(),
            "resume_checkpoint_ref": (
                None
                if request.resume_checkpoint_ref is None
                else request.resume_checkpoint_ref.to_dict()
            ),
            "request_metadata": (
                None
                if request.request_metadata is None
                else request.request_metadata.to_dict()
            ),
        }
    )


class LegacyRuntimeAdapter:
    """C-110 validation/preparation facade around an existing provider identity."""

    def __init__(
        self,
        *,
        provider_id: str,
        profile_id: str,
        backend_id: str,
    ) -> None:
        self._capabilities = RuntimeCapabilityProfileV1(
            provider_id=ComponentId(provider_id),
            profile_id=ComponentId(profile_id),
            backend_id=BackendId(backend_id),
            semantic_capabilities=(),
            capability_flags=("dormant_legacy_facade", "unified_production_lifecycle"),
        )
        self._prepared: dict[str, CanonicalPayloadV1] = {}

    @classmethod
    def from_current_provider(
        cls, provider: object, *, backend_id: str
    ) -> "LegacyRuntimeAdapter":
        return cls(
            provider_id=provider.provider_id,
            profile_id=provider.profile_id,
            backend_id=backend_id,
        )

    def capabilities(self) -> RuntimeCapabilityProfileV1:
        return self._capabilities

    def validate_assembly(self, assembly: SystemAssemblyV1) -> ValidationResultV1:
        if not isinstance(assembly, SystemAssemblyV1):
            return ValidationResultV1(
                valid=False,
                issues=(ContractIssueV1(
                    category=ContractErrorCategory.CONTRACT_INVALID,
                    owner="eiel_composition.adapters.legacy_runtime",
                    message="assembly must be a SystemAssemblyV1",
                ),),
            )
        expected = str(self._capabilities.profile_id)
        if str(assembly.runtime_profile.component_id) != expected:
            return ValidationResultV1(
                valid=False,
                issues=(ContractIssueV1(
                    category=ContractErrorCategory.CAPABILITY_UNSUPPORTED,
                    owner="eiel_composition.adapters.legacy_runtime",
                    message="assembly runtime profile does not match current provider",
                ),),
            )
        return ValidationResultV1(valid=True)

    @staticmethod
    def request_from_resident_batch(
        *,
        resident_request: object,
        assembly: SystemAssemblyV1,
        config_digest: ContentDigest,
        run_id: RunId,
    ) -> RuntimeExecutionRequestV1:
        payload = resident_request.payload()
        return RuntimeExecutionRequestV1(
            run_id=run_id,
            # Current execution ownership is physical-batch shaped.  The
            # batch ID is deliberately preserved rather than pretending a
            # fused request is one independently executable logical cell.
            cell_id=CellId(resident_request.batch_id),
            schedule=LegacyScheduleAdapter.from_resident_request(resident_request),
            assembly_id=assembly.assembly_id,
            config_digest=config_digest,
            request_metadata=CanonicalPayloadV1.from_value(
                LEGACY_RESIDENT_REQUEST_SCHEMA, payload
            ),
        )

    def prepare(
        self, request: RuntimeExecutionRequestV1, assembly: SystemAssemblyV1
    ) -> PreparedExecutionV1:
        validation = self.validate_assembly(assembly)
        if not validation.valid:
            issue = validation.issues[0]
            raise _runtime_error(issue.category, issue.message)
        if request.assembly_id != assembly.assembly_id:
            raise _runtime_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "request assembly identity does not match descriptor",
            )
        if (
            request.request_metadata is None
            or request.request_metadata.schema_version != LEGACY_RESIDENT_REQUEST_SCHEMA
        ):
            raise _runtime_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "dormant facade requires an already-translated resident request",
            )
        request_digest = _request_identity(request)
        opaque = f"legacy-prepared:{request_digest.hex}"
        self._prepared[opaque] = request.request_metadata
        return PreparedExecutionV1(
            provider_id=self._capabilities.provider_id,
            runtime_instance_id=ComponentInstanceId(opaque),
            request_digest=request_digest,
            assembly_digest=assembly.identity_digest(),
            opaque_prepared_ref=opaque,
        )

    def legacy_request_payload(self, prepared: PreparedExecutionV1) -> dict[str, object]:
        try:
            value = self._prepared[prepared.opaque_prepared_ref].to_value()
        except KeyError as exc:
            raise _runtime_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "prepared reference is not owned by this facade",
            ) from exc
        assert isinstance(value, dict)
        return copy.deepcopy(value)

    @staticmethod
    def _dormant(operation: str) -> None:
        raise _runtime_error(
            ContractErrorCategory.CAPABILITY_UNSUPPORTED,
            f"legacy runtime facade operation is dormant: {operation}",
        )

    def execute(self, prepared: PreparedExecutionV1, output_port: object) -> object:
        self._dormant("execute")

    def checkpoint(self, runtime_instance_id: ComponentInstanceId, reason: str) -> object:
        self._dormant("checkpoint")

    def resume(self, checkpoint_ref: object, request_override: object = None) -> object:
        self._dormant("resume")

    def branch(self, checkpoint_ref: object, branch_spec: object) -> object:
        self._dormant("branch")

    def cancel(self, runtime_instance_id: ComponentInstanceId) -> object:
        self._dormant("cancel")

    def status(self, runtime_instance_id: ComponentInstanceId) -> object:
        self._dormant("status")


@dataclass(frozen=True, slots=True)
class LegacyRuntimeServicesAdapter:
    """Dependency container over existing authorities; owns no service state."""

    current_cursor: RunCursorV1
    current_rng_authority: object
    current_telemetry_sink: object
    current_artifact_output: object
    current_control_port: object
    current_control_boundaries: object
    current_capabilities: RuntimeCapabilityProfileV1

    def cursor(self) -> RunCursorV1:
        return self.current_cursor

    def rng_authority(self) -> object:
        return self.current_rng_authority

    def telemetry_sink(self) -> object:
        return self.current_telemetry_sink

    def artifact_output(self) -> object:
        return self.current_artifact_output

    def control_port(self) -> object:
        return self.current_control_port

    def control_boundaries(self) -> object:
        return self.current_control_boundaries

    def capabilities(self) -> RuntimeCapabilityProfileV1:
        return self.current_capabilities


class LegacyTelemetryAdapter:
    """Read-only projection of an existing deterministic control event row."""

    @staticmethod
    def from_control_event(
        row: Mapping[str, object], *, run_id: RunId, cell_id: CellId, cursor: RunCursorV1
    ) -> RuntimeEventV1:
        return RuntimeEventV1(
            run_id=run_id,
            cell_id=cell_id,
            cursor=cursor,
            source_module_id=ModuleId("runtime"),
            source_component_id=ComponentId("legacy-control-plane"),
            event_type="control_command_applied",
            payload=CanonicalPayloadV1.from_value(LEGACY_CONTROL_EVENT_SCHEMA, row),
            submission_sequence=row["submission_sequence_number"],
        )

    @staticmethod
    def to_control_event(event: RuntimeEventV1) -> dict[str, object]:
        if event.payload.schema_version != LEGACY_CONTROL_EVENT_SCHEMA:
            raise _runtime_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "runtime event is not a legacy control event",
            )
        value = event.payload.to_value()
        assert isinstance(value, dict)
        return value


class LegacyControlPortAdapter:
    """Opt-in bridge to an injected current ControlPlane; dormant by default."""

    def __init__(
        self,
        *,
        ticks_per_generation: int,
        submitter: Callable[[object], object] | None = None,
    ) -> None:
        if type(ticks_per_generation) is not int or ticks_per_generation < 1:
            raise _runtime_error(
                ContractErrorCategory.CONTRACT_INVALID,
                "ticks_per_generation must be a positive integer",
            )
        self._ticks_per_generation = ticks_per_generation
        self._submitter = submitter
        self._results: dict[CommandId, ControlResultV1] = {}

    def to_legacy_payload(self, command: ControlCommandV1) -> dict[str, object]:
        if command.target.next_tick is None:
            raise _runtime_error(
                ContractErrorCategory.CAPABILITY_UNSUPPORTED,
                "legacy control plane requires a tick target",
            )
        if command.target.next_tick % self._ticks_per_generation:
            raise _runtime_error(
                ContractErrorCategory.CAPABILITY_UNSUPPORTED,
                "legacy control target must be an exact generation boundary",
            )
        body = command.payload.to_value()
        if not isinstance(body, dict):
            raise _runtime_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "legacy control payload must be an object",
            )
        reserved = {"schema_version", "command_type", "command_uuid", "target_generation"}
        if reserved & set(body):
            raise _runtime_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "legacy control payload repeats envelope-owned fields",
            )
        return {
            "schema_version": str(command.payload.schema_version),
            "command_type": command.command_type,
            "command_uuid": str(command.command_id),
            "target_generation": command.target.next_tick // self._ticks_per_generation,
            **copy.deepcopy(body),
        }

    def submit(self, command: ControlCommandV1) -> ControlResultV1:
        if self._submitter is None:
            raise _runtime_error(
                ContractErrorCategory.CAPABILITY_UNSUPPORTED,
                "legacy control submission is dormant without an injected port",
            )
        payload = self.to_legacy_payload(command)
        from simulation.control_plane_pkg.commands import parse_command

        ack = self._submitter(parse_command(payload))
        ack_payload = ack.to_dict()
        result = ControlResultV1(
            command_id=command.command_id,
            # ACCEPTED is scheduled, not applied.  The legacy event log remains
            # the authority for the eventual application boundary.
            applied=False,
            applied_cursor=None,
            result=CanonicalPayloadV1.from_value(LEGACY_CONTROL_ACK_SCHEMA, ack_payload),
        )
        self._results[command.command_id] = result
        return result

    def status(self, command_id: CommandId) -> ControlResultV1:
        try:
            return self._results[command_id]
        except KeyError as exc:
            raise _runtime_error(
                ContractErrorCategory.CONFIGURATION_INVALID,
                "control command has no result in this facade",
            ) from exc
