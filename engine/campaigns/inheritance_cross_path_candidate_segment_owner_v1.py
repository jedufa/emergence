"""Qualification-only owner for preloaded cross-path CUDA candidates.

The owner reuses the real Patch resident segment loop without granting the
candidate production execution authority.  It accepts only the four upstream
artifact bindings, records every projected exposure segment, and makes a
fresh-owner continuation reload its strict Bundle-v2 source exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Callable, Final, Mapping, Sequence
from uuid import uuid4

from engine.native_runtime.inheritance_cross_path_qualified_artifact_v1 import (
    CandidateInheritanceCrossPathArtifactV1,
    UPSTREAM_BINDING_IDS,
)
from engine.native_runtime.checkpoint_digests import (
    DigestLayer,
    checkpoint_digest_manifest,
)
from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
    required_qualified_reproduction_attempt_capacity_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    required_resident_agent_id_capacity_v1,
)

from .canonical import canonical_sha256
from .inheritance_cross_path_exposure_projection_contract_v1 import (
    AGENT_ID_ALLOCATION_SCHEMA_VERSION,
)
from .production_engine_provider import (
    EielProductionExecutionProviderV1,
)
from .inheritance_cross_path_native_qualification_observer_v1 import (
    CrossPathNativeQualificationObserverV1,
    ResidentAttemptCapacityFaultDirectiveV1,
)
from .inheritance_cross_path_storage_parity_v1 import (
    cross_storage_candidate_dimension_sha256s_v1,
)
from .providers import (
    CampaignProviderError,
    ProductionEngineLifecycleV1,
    ResidentBatchRequestV1,
)


REQUEST_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_candidate_segment_request_v2"
)
RESULT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_candidate_segment_result_v1"
)
SEGMENT_TRACE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_candidate_segment_trace_v2"
)
RESTORE_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_candidate_restore_receipt_v1"
)
CHECKPOINT_SOURCE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_candidate_checkpoint_source_v1"
)

_HEX = frozenset("0123456789abcdef")
_HISTORICAL_TEST_FIXTURE_TOKEN = object()


class InheritanceCrossPathCandidateSegmentOwnerError(RuntimeError):
    """A candidate request cannot be executed without inventing authority."""


def _fail(message: str) -> InheritanceCrossPathCandidateSegmentOwnerError:
    return InheritanceCrossPathCandidateSegmentOwnerError(message)


def _digest(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise _fail(f"{name} must be a lowercase SHA-256 digest")
    return value


def _text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{name} must be non-empty text")
    return value


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise _fail(f"{name} must contain only canonical JSON data") from exc


def _safe_id(name: str, value: Any) -> str:
    text = _text(name, value)
    if Path(text).name != text or text in {".", ".."}:
        raise _fail(f"{name} must be a safe path component")
    return text


def _batch_request_from_payload(value: Any) -> ResidentBatchRequestV1:
    if not isinstance(value, Mapping):
        raise _fail("batch_request must be an object")
    expected = {
        "profile_id",
        "batch_id",
        "cell_ids",
        "generations",
        "ticks_per_generation",
        "checkpoint_milestones",
        "parameter_arrays",
        "seed_arrays",
        "agent_id_allocation_generation_count",
        "checkpoint_storage_mode",
        "reporting_enabled",
        "execution_route_sha256",
        "genome_v2_checkpoint_authority_binding",
    }
    if set(value) != expected:
        raise _fail("batch_request field set is invalid")
    cell_ids = value["cell_ids"]
    milestones = value["checkpoint_milestones"]
    parameters = value["parameter_arrays"]
    seeds = value["seed_arrays"]
    genome_v2_binding = value["genome_v2_checkpoint_authority_binding"]
    if (
        isinstance(cell_ids, (str, bytes))
        or not isinstance(cell_ids, Sequence)
        or isinstance(milestones, (str, bytes))
        or not isinstance(milestones, Sequence)
        or not isinstance(parameters, Mapping)
        or not isinstance(seeds, Mapping)
        or not isinstance(genome_v2_binding, Mapping)
    ):
        raise _fail("batch_request arrays are malformed")
    return ResidentBatchRequestV1(
        profile_id=value["profile_id"],
        batch_id=value["batch_id"],
        cell_ids=tuple(cell_ids),
        generations=value["generations"],
        ticks_per_generation=value["ticks_per_generation"],
        checkpoint_milestones=tuple(milestones),
        parameter_arrays=tuple(
            (name, tuple(rows)) for name, rows in sorted(parameters.items())
        ),
        seed_arrays=tuple((name, tuple(rows)) for name, rows in sorted(seeds.items())),
        agent_id_allocation_generation_count=value[
            "agent_id_allocation_generation_count"
        ],
        checkpoint_storage_mode=value["checkpoint_storage_mode"],
        reporting_enabled=value["reporting_enabled"],
        execution_route_sha256=value["execution_route_sha256"],
        genome_v2_checkpoint_authority_binding=tuple(
            sorted(genome_v2_binding.items())
        ),
    )


@dataclass(frozen=True, slots=True)
class CandidateCheckpointSourceV1:
    """One prior run's immutable checkpoint selected for fresh-owner restore."""

    source_run_id: str
    source_owner_id: str
    source_result_sha256: str
    checkpoint: Mapping[str, Any]
    schema_version: str = CHECKPOINT_SOURCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SOURCE_SCHEMA_VERSION:
            raise _fail("candidate checkpoint source schema mismatch")
        _safe_id("source_run_id", self.source_run_id)
        _safe_id("source_owner_id", self.source_owner_id)
        _digest("source_result_sha256", self.source_result_sha256)
        checkpoint = _json_copy(self.checkpoint, name="checkpoint source")
        if (
            not isinstance(checkpoint, Mapping)
            or type(checkpoint.get("generation")) is not int
            or checkpoint["generation"] < 1
            or not isinstance(checkpoint.get("cell_bundles"), list)
            or not isinstance(checkpoint.get("terminal_cell_digests"), Mapping)
        ):
            raise _fail("candidate checkpoint source is incomplete")
        object.__setattr__(self, "checkpoint", dict(checkpoint))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_run_id": self.source_run_id,
            "source_owner_id": self.source_owner_id,
            "source_result_sha256": self.source_result_sha256,
            "checkpoint": _json_copy(self.checkpoint, name="checkpoint source"),
        }

    @classmethod
    def from_payload(cls, value: Any) -> "CandidateCheckpointSourceV1":
        fields = {
            "schema_version",
            "source_run_id",
            "source_owner_id",
            "source_result_sha256",
            "checkpoint",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise _fail("candidate checkpoint source field set is invalid")
        return cls(**{name: value[name] for name in fields})


@dataclass(frozen=True, slots=True)
class CandidateExposureCapacityV1:
    """Request-local exposure allocation bounded by a candidate's envelope."""

    storage_mode: str
    attempt_capacity: int
    snapshot_capacity: int
    eligible_capacity: int
    aggregate_capacity: int

    def __post_init__(self) -> None:
        if self.storage_mode not in {"exact_full", "compact_complete"}:
            raise _fail("request exposure storage mode is unsupported")
        for name in (
            "attempt_capacity",
            "snapshot_capacity",
            "eligible_capacity",
            "aggregate_capacity",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise _fail(f"request exposure {name} must be positive")
        if (
            self.storage_mode == "exact_full"
            and self.snapshot_capacity < 2 * self.attempt_capacity
        ):
            raise _fail(
                "exact_full request snapshot capacity must cover two roles"
            )

    @classmethod
    def from_payload(cls, value: Any) -> "CandidateExposureCapacityV1":
        fields = {
            "storage_mode",
            "attempt_capacity",
            "snapshot_capacity",
            "eligible_capacity",
            "aggregate_capacity",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise _fail("request exposure capacity field set is invalid")
        return cls(**{name: value[name] for name in fields})

    def payload(self) -> dict[str, Any]:
        return {
            "storage_mode": self.storage_mode,
            "attempt_capacity": self.attempt_capacity,
            "snapshot_capacity": self.snapshot_capacity,
            "eligible_capacity": self.eligible_capacity,
            "aggregate_capacity": self.aggregate_capacity,
        }


@dataclass(frozen=True, slots=True)
class InheritanceCrossPathCandidateSegmentRequestV1:
    """Exact declarative request for one non-production candidate execution."""

    request_id: str
    candidate_authority_sha256: str
    candidate_artifact_sha256s: Mapping[str, str]
    plan: Mapping[str, Any]
    batch_request: ResidentBatchRequestV1
    runtime_root: Path
    exposure_capacity: CandidateExposureCapacityV1
    split_generation: int | None = None
    checkpoint_source: CandidateCheckpointSourceV1 | None = None
    capacity_fault: ResidentAttemptCapacityFaultDirectiveV1 | None = None
    fixed_action_required: bool = False
    storage_observation_required: bool = False
    schema_version: str = REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_SCHEMA_VERSION:
            raise _fail("candidate segment request schema mismatch")
        _safe_id("request_id", self.request_id)
        _digest("candidate_authority_sha256", self.candidate_authority_sha256)
        if not isinstance(self.candidate_artifact_sha256s, Mapping) or set(
            self.candidate_artifact_sha256s
        ) != set(UPSTREAM_BINDING_IDS):
            raise _fail("candidate artifact digest set is incomplete")
        artifact_sha256s = {
            artifact_id: _digest(
                f"candidate_artifact_sha256s.{artifact_id}",
                self.candidate_artifact_sha256s[artifact_id],
            )
            for artifact_id in UPSTREAM_BINDING_IDS
        }
        object.__setattr__(self, "candidate_artifact_sha256s", artifact_sha256s)

        plan = _json_copy(self.plan, name="plan")
        if not isinstance(plan, Mapping):
            raise _fail("plan must be an object")
        plan = dict(plan)
        plan_sha256 = plan.get("plan_sha256")
        _digest("plan.plan_sha256", plan_sha256)
        unsigned_plan = dict(plan)
        unsigned_plan.pop("plan_sha256")
        if canonical_sha256(unsigned_plan) != plan_sha256:
            raise _fail("plan_sha256 does not bind the exact candidate plan")
        for name in ("campaign_id", "qualification_key_sha256"):
            _text(f"plan.{name}", plan.get(name))
        if plan.get("resolved_profile") != (
            EielProductionExecutionProviderV1.profile_id
        ):
            raise _fail("candidate plan uses another runtime profile")

        artifacts = plan.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(
            UPSTREAM_BINDING_IDS
        ):
            raise _fail("candidate plan must contain exactly four upstream artifacts")
        for artifact_id in UPSTREAM_BINDING_IDS:
            row = artifacts[artifact_id]
            if (
                not isinstance(row, Mapping)
                or row.get("sha256") != artifact_sha256s[artifact_id]
                or not isinstance(row.get("path"), str)
                or not row["path"]
                or not isinstance(row.get("abi"), str)
                or not row["abi"]
            ):
                raise _fail(
                    f"candidate plan artifact binding is invalid: {artifact_id}"
                )

        request = self.batch_request
        if not isinstance(request, ResidentBatchRequestV1):
            raise _fail("batch_request must be ResidentBatchRequestV1")
        if (
            request.profile_id != EielProductionExecutionProviderV1.profile_id
            or request.checkpoint_storage_mode != "disk"
            or request.execution_route_sha256 != self.candidate_authority_sha256
            or type(request.reporting_enabled) is not bool
            or type(request.generations) is not int
            or request.generations < 1
            or type(request.ticks_per_generation) is not int
            or request.ticks_per_generation < 1
        ):
            raise _fail("candidate resident request is invalid")
        if (
            not request.cell_ids
            or len(set(request.cell_ids)) != len(request.cell_ids)
            or any(
                _safe_id("batch_request.cell_ids", cell_id) != cell_id
                for cell_id in request.cell_ids
            )
        ):
            raise _fail("candidate resident cell IDs are invalid")
        milestones = request.checkpoint_milestones
        if (
            tuple(sorted(set(milestones))) != milestones
            or not milestones
            or milestones[-1] != request.generations
            or any(
                type(generation) is not int
                or generation < 0
                or generation > request.generations
                for generation in milestones
            )
        ):
            raise _fail("candidate checkpoint milestones are invalid")
        if any(
            len(values) != len(request.cell_ids)
            for _name, values in (
                *request.parameter_arrays,
                *request.seed_arrays,
            )
        ):
            raise _fail("candidate parameter or seed arrays differ from batch size")

        batches = plan.get("physical_batches")
        cells = plan.get("cells")
        if (
            not isinstance(batches, list)
            or len(batches) != 1
            or not isinstance(batches[0], Mapping)
            or batches[0].get("batch_id") != request.batch_id
            or batches[0].get("cell_ids") != list(request.cell_ids)
            or not isinstance(cells, list)
            or [cell.get("cell_id") for cell in cells if isinstance(cell, Mapping)]
            != list(request.cell_ids)
            or len(cells) != len(request.cell_ids)
            or any(
                not isinstance(cell.get("parameters"), Mapping)
                or not isinstance(cell.get("seeds"), Mapping)
                for cell in cells
                if isinstance(cell, Mapping)
            )
        ):
            raise _fail("candidate plan physical order differs from its request")
        execution = plan.get("execution")
        if (
            not isinstance(execution, Mapping)
            or execution.get("generations") != request.generations
            or execution.get("ticks_per_generation") != request.ticks_per_generation
            or execution.get("checkpoints")
            != [
                generation
                for generation in request.checkpoint_milestones
                if generation != 0
            ]
            or execution.get("checkpoint_storage_mode") != "disk"
        ):
            raise _fail("candidate plan execution differs from its request")
        for namespace, arrays in (
            ("parameters", request.parameter_arrays),
            ("seeds", request.seed_arrays),
        ):
            for name, values in arrays:
                try:
                    planned = tuple(dict(cell[namespace])[name] for cell in cells)
                except (KeyError, TypeError, ValueError) as exc:
                    raise _fail(
                        f"candidate plan lacks request {namespace}.{name}"
                    ) from exc
                if values != planned:
                    raise _fail(
                        f"candidate request {namespace}.{name} differs from plan"
                    )

        root = Path(self.runtime_root)
        if not root.is_absolute() or root.is_symlink():
            raise _fail("runtime_root must be an absolute non-symlink path")
        object.__setattr__(self, "runtime_root", root)
        if self.split_generation is not None and (
            type(self.split_generation) is not int
            or not 0 < self.split_generation < request.generations
            or self.split_generation not in milestones
        ):
            raise _fail("split_generation must be an interior declared checkpoint")
        if self.checkpoint_source is not None and (
            not isinstance(self.checkpoint_source, CandidateCheckpointSourceV1)
            or self.split_generation is None
            or self.checkpoint_source.checkpoint.get("generation")
            != self.split_generation
        ):
            raise _fail(
                "checkpoint_source requires the exact declared split generation"
            )
        if self.capacity_fault is not None:
            if (
                not isinstance(
                    self.capacity_fault,
                    ResidentAttemptCapacityFaultDirectiveV1,
                )
                or self.capacity_fault.target_cell_index >= len(request.cell_ids)
                or len(request.cell_ids) < 2
            ):
                raise _fail(
                    "capacity fault must target one cell in a multi-cell batch"
                )
        if (
            type(self.fixed_action_required) is not bool
            or type(self.storage_observation_required) is not bool
        ):
            raise _fail("qualification observation flags must be boolean")
        if not isinstance(self.exposure_capacity, CandidateExposureCapacityV1):
            raise _fail("candidate request requires typed exposure capacity")
        object.__setattr__(self, "plan", plan)

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> "InheritanceCrossPathCandidateSegmentRequestV1":
        if not isinstance(payload, Mapping):
            raise _fail("candidate segment request payload must be an object")
        expected = {
            "schema_version",
            "request_id",
            "candidate_authority_sha256",
            "candidate_artifact_sha256s",
            "plan",
            "batch_request",
            "runtime_root",
            "exposure_capacity",
            "split_generation",
            "checkpoint_source",
            "capacity_fault",
            "fixed_action_required",
            "storage_observation_required",
        }
        if set(payload) != expected:
            raise _fail("candidate segment request field set is invalid")
        return cls(
            schema_version=payload["schema_version"],
            request_id=payload["request_id"],
            candidate_authority_sha256=payload["candidate_authority_sha256"],
            candidate_artifact_sha256s=payload["candidate_artifact_sha256s"],
            plan=payload["plan"],
            batch_request=_batch_request_from_payload(payload["batch_request"]),
            runtime_root=Path(payload["runtime_root"]),
            exposure_capacity=CandidateExposureCapacityV1.from_payload(
                payload["exposure_capacity"]
            ),
            split_generation=payload["split_generation"],
            checkpoint_source=(
                None
                if payload["checkpoint_source"] is None
                else CandidateCheckpointSourceV1.from_payload(
                    payload["checkpoint_source"]
                )
            ),
            capacity_fault=(
                None
                if payload["capacity_fault"] is None
                else ResidentAttemptCapacityFaultDirectiveV1.from_payload(
                    payload["capacity_fault"]
                )
            ),
            fixed_action_required=payload["fixed_action_required"],
            storage_observation_required=payload["storage_observation_required"],
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "candidate_authority_sha256": self.candidate_authority_sha256,
            "candidate_artifact_sha256s": dict(self.candidate_artifact_sha256s),
            "plan": _json_copy(self.plan, name="plan"),
            "batch_request": self.batch_request.payload(),
            "runtime_root": str(self.runtime_root),
            "exposure_capacity": self.exposure_capacity.payload(),
            "split_generation": self.split_generation,
            "checkpoint_source": (
                None
                if self.checkpoint_source is None
                else self.checkpoint_source.payload()
            ),
            "capacity_fault": (
                None
                if self.capacity_fault is None
                else self.capacity_fault.payload()
            ),
            "fixed_action_required": self.fixed_action_required,
            "storage_observation_required": self.storage_observation_required,
        }

    @property
    def request_sha256(self) -> str:
        return canonical_sha256(self.payload())

    def validate_candidate(
        self, candidate: CandidateInheritanceCrossPathArtifactV1
    ) -> None:
        if (
            not isinstance(candidate, CandidateInheritanceCrossPathArtifactV1)
            or candidate.production_authority is not False
            or candidate.candidate_authority_sha256 != self.candidate_authority_sha256
            or set(candidate.bindings) != set(UPSTREAM_BINDING_IDS)
        ):
            raise _fail("loaded candidate differs from request authority")
        artifacts = self.plan["artifacts"]
        for artifact_id in UPSTREAM_BINDING_IDS:
            binding = candidate.bindings[artifact_id]
            row = artifacts[artifact_id]
            if (
                binding.sha256 != self.candidate_artifact_sha256s[artifact_id]
                or str(binding.path) != row["path"]
                or binding.abi != row["abi"]
            ):
                raise _fail(f"loaded candidate binding differs: {artifact_id}")
        registry = candidate.registry
        if (
            registry.get("ticks_per_generation")
            != self.batch_request.ticks_per_generation
            or type(registry.get("max_generations")) is not int
            or self.batch_request.generations > registry["max_generations"]
        ):
            raise _fail("candidate request exceeds its registry execution envelope")
        projection_contract = candidate.telemetry_contract.get(
            "projection_contract"
        )
        registered_capacity = (
            projection_contract.get("envelope")
            if isinstance(projection_contract, Mapping)
            else None
        )
        if not isinstance(registered_capacity, Mapping):
            raise _fail(
                "candidate telemetry lacks its projection capacity envelope"
            )
        requested_capacity = self.exposure_capacity
        agent_id_allocation = registered_capacity.get("agent_id_allocation")
        registered_fields = {
            "attempt_capacity": registered_capacity.get("attempt_capacity"),
            "snapshot_capacity": registered_capacity.get("snapshot_capacity"),
            "eligible_capacity": registered_capacity.get("eligible_capacity"),
            "aggregate_capacity": registered_capacity.get("aggregate_capacity"),
        }
        if (
            requested_capacity.storage_mode
            != registered_capacity.get("storage_mode")
            or any(
                type(value) is not int or value < 1
                for value in registered_fields.values()
            )
            or any(
                getattr(requested_capacity, name) > registered
                for name, registered in registered_fields.items()
            )
            or requested_capacity.eligible_capacity
            != registered_fields["eligible_capacity"]
        ):
            raise _fail(
                "request exposure capacity exceeds or changes its candidate envelope"
            )
        allocation_fields = {
            "current_capacity",
            "population_next_id",
            "max_births_per_tick",
            "slot_capacity",
            "allocation_generation_count",
            "retained_capacity",
            "required_aggregate_capacity",
            "schema_version",
        }
        if (
            not isinstance(agent_id_allocation, Mapping)
            or set(agent_id_allocation) != allocation_fields
            or any(
                type(agent_id_allocation.get(name)) is not int
                or agent_id_allocation[name] < 1
                for name in allocation_fields - {"schema_version"}
            )
            or agent_id_allocation.get("schema_version")
            != AGENT_ID_ALLOCATION_SCHEMA_VERSION
        ):
            raise _fail(
                "candidate telemetry lacks its agent-ID allocation envelope"
            )
        registered_required_aggregate_capacity = (
            required_resident_agent_id_capacity_v1(
                agent_id_allocation["current_capacity"],
                agent_id_allocation["population_next_id"],
                agent_id_allocation["max_births_per_tick"],
                agent_id_allocation["slot_capacity"],
                allocation_generation_count=agent_id_allocation[
                    "allocation_generation_count"
                ],
                retained_capacity=agent_id_allocation["retained_capacity"],
            )
        )
        if (
            agent_id_allocation["required_aggregate_capacity"]
            != registered_required_aggregate_capacity
            or registered_fields["aggregate_capacity"]
            < registered_required_aggregate_capacity
        ):
            raise _fail(
                "candidate agent-ID allocation envelope is internally "
                "inconsistent"
            )
        requested_allocation_generations = (
            self.batch_request.agent_id_allocation_generation_count
        )
        if (
            type(requested_allocation_generations) is not int
            or requested_allocation_generations
            > agent_id_allocation["allocation_generation_count"]
        ):
            raise _fail(
                "candidate request exceeds its registered agent-ID allocation "
                "horizon"
            )
        required_aggregate_capacity = required_resident_agent_id_capacity_v1(
            agent_id_allocation["current_capacity"],
            agent_id_allocation["population_next_id"],
            agent_id_allocation["max_births_per_tick"],
            agent_id_allocation["slot_capacity"],
            allocation_generation_count=requested_allocation_generations,
            retained_capacity=agent_id_allocation["retained_capacity"],
        )
        if requested_capacity.aggregate_capacity < required_aggregate_capacity:
            raise _fail(
                "request exposure aggregate capacity does not bound its "
                "agent-ID allocation horizon"
            )
        prior_generation = 0
        maximum_segment_generations = 0
        for milestone in self.batch_request.checkpoint_milestones:
            maximum_segment_generations = max(
                maximum_segment_generations,
                milestone - prior_generation,
            )
            prior_generation = milestone
        required_attempts = (
            required_qualified_reproduction_attempt_capacity_v1(
                segment_generations=maximum_segment_generations,
                ticks_per_generation=self.batch_request.ticks_per_generation,
                resident_capacity=requested_capacity.eligible_capacity,
            )
        )
        if requested_capacity.attempt_capacity < required_attempts:
            raise _fail(
                "request exposure capacity does not bound its execution dimensions"
            )


@dataclass(slots=True)
class _ExecutionContext:
    run_root: Path
    batch_root: Path
    checkpoints: list[dict[str, Any]]

    def record_checkpoint(self, checkpoint: Mapping[str, Any]) -> None:
        self.checkpoints.append(_json_copy(checkpoint, name="checkpoint record"))


ProviderFactoryV1 = Callable[
    [
        CandidateInheritanceCrossPathArtifactV1,
        InheritanceCrossPathCandidateSegmentRequestV1,
    ],
    ProductionEngineLifecycleV1,
]


class InheritanceCrossPathCandidateSegmentOwnerV1:
    """Tombstone for the retired standalone inheritance candidate owner."""

    def __init__(
        self,
        *,
        provider_factory: ProviderFactoryV1 | None = None,
    ) -> None:
        raise _fail(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "candidate segment owner construction is disabled"
        )

    def _new_provider(
        self,
        candidate: CandidateInheritanceCrossPathArtifactV1,
        request: InheritanceCrossPathCandidateSegmentRequestV1,
    ) -> tuple[ProductionEngineLifecycleV1, str]:
        provider = self._provider_factory(candidate, request)
        if not isinstance(provider, ProductionEngineLifecycleV1):
            raise _fail("provider_factory returned another execution owner")
        return provider, f"candidate-owner-{uuid4().hex}"

    @staticmethod
    def _candidate_paths(
        candidate: CandidateInheritanceCrossPathArtifactV1,
    ) -> tuple[str, str, str]:
        return (
            str(candidate.bindings["cuda_extension"].path),
            str(candidate.bindings["extension_manifest"].path),
            str(candidate.bindings["telemetry_contract"].path),
        )

    @staticmethod
    def _checkpoint_rows(
        checkpoint: Mapping[str, Any],
        *,
        cell_ids: tuple[str, ...],
    ) -> dict[str, Mapping[str, Any]]:
        rows = checkpoint.get("cell_bundles")
        if not isinstance(rows, list):
            raise _fail("candidate checkpoint has no Bundle-v2 rows")
        by_id = {
            row.get("cell_id"): row
            for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
        }
        if tuple(by_id) != cell_ids or len(rows) != len(cell_ids):
            raise _fail("candidate checkpoint cell order drifted")
        return by_id

    def _execute_owner(
        self,
        *,
        provider: ProductionEngineLifecycleV1,
        owner_id: str,
        candidate: CandidateInheritanceCrossPathArtifactV1,
        request: InheritanceCrossPathCandidateSegmentRequestV1,
        batch_request: ResidentBatchRequestV1,
        context: _ExecutionContext,
        start_generation: int,
        resume_projections: Sequence[Mapping[str, Any]] | None,
        prior_metrics_rows: Sequence[Mapping[str, Any]] = (),
        prior_checkpoint_catalog: Mapping[str, Mapping[str, Any]] | None = None,
        prior_native_chronicle_receipt: Mapping[str, Any] | None = None,
        prior_native_continuity: Mapping[str, Any] | None = None,
        prior_chronicle_checkpoint_authority_sha256: str | None = None,
        qualification_observer: CrossPathNativeQualificationObserverV1 | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        observed: list[dict[str, Any]] = []

        def observe(
            *,
            segment_ordinal: int,
            start_generation: int,
            end_generation: int,
            cell_ids: tuple[str, ...],
            segment_artifacts: tuple[Mapping[str, Any], ...],
            checkpoint_projections: tuple[Mapping[str, Any], ...],
        ) -> None:
            if (
                cell_ids != batch_request.cell_ids
                or len(segment_artifacts) != len(cell_ids)
                or len(checkpoint_projections) != len(cell_ids)
                or any(
                    artifact.get("cell_id") != cell_id
                    for artifact, cell_id in zip(
                        segment_artifacts, cell_ids, strict=True
                    )
                )
            ):
                raise _fail("real segment artifacts differ from physical cell order")
            observed.append(
                {
                    "schema_version": SEGMENT_TRACE_SCHEMA_VERSION,
                    "owner_id": owner_id,
                    "owner_segment_ordinal": segment_ordinal,
                    "start_generation": start_generation,
                    "end_generation": end_generation,
                    "cell_ids": list(cell_ids),
                    "segment_artifacts": _json_copy(
                        segment_artifacts, name="segment artifacts"
                    ),
                    "terminal_cell_digests": {
                        cell_id: provider.terminal_projection_digest(projection)
                        for cell_id, projection in zip(
                            cell_ids, checkpoint_projections, strict=True
                        )
                    },
                    "mechanics_checkpoint_digests": {
                        cell_id: str(
                            checkpoint_digest_manifest(
                                projection,
                                layer=DigestLayer.TRAJECTORY_STATE,
                            )["digest"]
                        )
                        for cell_id, projection in zip(
                            cell_ids, checkpoint_projections, strict=True
                        )
                    },
                    "canonical_checkpoint_storage_digests": {
                        cell_id: str(
                            checkpoint_digest_manifest(
                                projection,
                                layer=DigestLayer.CANONICAL_CHECKPOINT_STORAGE,
                            )["digest"]
                        )
                        for cell_id, projection in zip(
                            cell_ids, checkpoint_projections, strict=True
                        )
                    },
                }
            )

        result = ProductionEngineLifecycleV1._execute_segments(
            provider,
            request.plan,
            batch_request,
            context,
            start_generation=start_generation,
            resume_projections=resume_projections,
            prior_metrics_rows=prior_metrics_rows,
            prior_checkpoint_catalog=prior_checkpoint_catalog,
            prior_native_chronicle_receipt=prior_native_chronicle_receipt,
            prior_native_continuity=prior_native_continuity,
            prior_chronicle_checkpoint_authority_sha256=(
                prior_chronicle_checkpoint_authority_sha256
            ),
            preloaded_artifact=candidate,
            execution_artifact_paths=self._candidate_paths(candidate),
            segment_artifact_observer=observe,
            segment_execution_observer=qualification_observer,
        )
        if len(observed) != len(context.checkpoints):
            raise _fail("candidate segment trace/checkpoint count differs")
        for trace, checkpoint in zip(observed, context.checkpoints, strict=True):
            if trace["end_generation"] != checkpoint.get("generation"):
                raise _fail("candidate segment trace/checkpoint generation differs")
            trace["checkpoint"] = checkpoint
        return result, observed

    def execute(
        self,
        candidate: CandidateInheritanceCrossPathArtifactV1,
        request: InheritanceCrossPathCandidateSegmentRequestV1,
    ) -> dict[str, Any]:
        """Reject execution of the retired standalone candidate route."""

        raise _fail(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "candidate segment execution is disabled"
        )
        # The historical implementation below remains temporarily readable
        # while P12 evidence references are reconciled. It is unreachable.
        if not isinstance(request, InheritanceCrossPathCandidateSegmentRequestV1):
            raise _fail("request must be a candidate segment request v1")
        request.validate_candidate(candidate)
        try:
            request.runtime_root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise _fail("candidate runtime_root already exists") from exc

        source_provider, source_owner_id = self._new_provider(candidate, request)
        terminal_provider = source_provider
        terminal_owner_id = source_owner_id
        full_request = request.batch_request
        traces: list[dict[str, Any]] = []
        restore_rows: list[dict[str, Any]] = []
        qualification_observer = (
            CrossPathNativeQualificationObserverV1(
                fault=request.capacity_fault,
                fixed_action_required=request.fixed_action_required,
            )
            if request.capacity_fault is not None
            or request.fixed_action_required
            or request.storage_observation_required
            else None
        )

        try:
            if request.split_generation is None:
                context = _ExecutionContext(
                    request.runtime_root,
                    request.runtime_root / "owner",
                    [],
                )
                context.batch_root.mkdir(parents=True, exist_ok=False)
                terminal_result, traces = self._execute_owner(
                    provider=source_provider,
                    owner_id=source_owner_id,
                    candidate=candidate,
                    request=request,
                    batch_request=full_request,
                    context=context,
                    start_generation=0,
                    resume_projections=None,
                    qualification_observer=qualification_observer,
                )
                checkpoint_rows = self._checkpoint_rows(
                    terminal_result["checkpoint"],
                    cell_ids=full_request.cell_ids,
                )
                restore_rows = [
                    {
                        "schema_version": RESTORE_RECEIPT_SCHEMA_VERSION,
                        "cell_id": cell_id,
                        "source_owner_id": source_owner_id,
                        "terminal_owner_id": terminal_owner_id,
                        "checkpoint_reloaded": False,
                        "checkpoint_source_sha256": checkpoint_rows[cell_id][
                            "source_projection_sha256"
                        ],
                        "restored_source_sha256": checkpoint_rows[cell_id][
                            "source_projection_sha256"
                        ],
                    }
                    for cell_id in full_request.cell_ids
                ]
            else:
                split = request.split_generation
                first_request = replace(
                    full_request,
                    generations=split,
                    checkpoint_milestones=tuple(
                        generation
                        for generation in full_request.checkpoint_milestones
                        if generation <= split
                    ),
                )
                source_context = _ExecutionContext(
                    request.runtime_root,
                    request.runtime_root / "source-owner",
                    [],
                )
                source_context.batch_root.mkdir(parents=True, exist_ok=False)
                first_result, first_traces = self._execute_owner(
                    provider=source_provider,
                    owner_id=source_owner_id,
                    candidate=candidate,
                    request=request,
                    batch_request=first_request,
                    context=source_context,
                    start_generation=0,
                    resume_projections=None,
                    qualification_observer=qualification_observer,
                )
                checkpoint = first_result["checkpoint"]
                if checkpoint.get("generation") != split or checkpoint.get(
                    "checkpoint_binding"
                ) != source_provider._checkpoint_binding(request.plan, full_request):
                    raise _fail("fresh-owner source checkpoint binding differs")
                checkpoint_source = request.checkpoint_source
                if checkpoint_source is not None:
                    external_checkpoint = checkpoint_source.checkpoint
                    generated_rows = self._checkpoint_rows(
                        checkpoint, cell_ids=full_request.cell_ids
                    )
                    external_rows = self._checkpoint_rows(
                        external_checkpoint,
                        cell_ids=full_request.cell_ids,
                    )
                    if (
                        external_checkpoint.get("generation") != split
                        or external_checkpoint.get("terminal_cell_digests")
                        != checkpoint.get("terminal_cell_digests")
                        or {
                            cell_id: generated_rows[cell_id][
                                "source_projection_sha256"
                            ]
                            for cell_id in full_request.cell_ids
                        }
                        != {
                            cell_id: external_rows[cell_id][
                                "source_projection_sha256"
                            ]
                            for cell_id in full_request.cell_ids
                        }
                    ):
                        raise _fail(
                            "freshly reproduced split state differs from the "
                            "preregistered external checkpoint source"
                        )
                    checkpoint = external_checkpoint
                    source_owner_id = checkpoint_source.source_owner_id

                terminal_provider, terminal_owner_id = self._new_provider(
                    candidate, request
                )
                if terminal_owner_id == source_owner_id:
                    raise _fail("fresh-owner identities are not disjoint")
                checkpoint_rows = self._checkpoint_rows(
                    checkpoint, cell_ids=full_request.cell_ids
                )
                projections: list[Mapping[str, Any]] = []
                for cell_id in full_request.cell_ids:
                    row = checkpoint_rows[cell_id]
                    projection = terminal_provider._read_projection_bundle(row["path"])
                    restored = terminal_provider.terminal_projection_digest(projection)
                    if restored != row.get(
                        "source_projection_sha256"
                    ) or restored != checkpoint["terminal_cell_digests"].get(cell_id):
                        raise _fail(
                            "fresh-owner restored source digest differs from checkpoint"
                        )
                    projections.append(projection)
                    restore_rows.append(
                        {
                            "schema_version": RESTORE_RECEIPT_SCHEMA_VERSION,
                            "cell_id": cell_id,
                            "source_owner_id": source_owner_id,
                            "terminal_owner_id": terminal_owner_id,
                            "checkpoint_reloaded": True,
                            "checkpoint_source_sha256": row["source_projection_sha256"],
                            "restored_source_sha256": restored,
                        }
                    )

                terminal_context = _ExecutionContext(
                    request.runtime_root,
                    request.runtime_root / "terminal-owner",
                    [],
                )
                terminal_context.batch_root.mkdir(parents=True, exist_ok=False)
                chronicle_receipt = checkpoint.get(
                    "provider_native_chronicle_receipt"
                )
                native_continuity = checkpoint.get(
                    "provider_native_reproduction_continuity"
                )
                chronicle_authority = checkpoint.get(
                    "provider_native_chronicle_checkpoint_authority_sha256"
                )
                chronicle_values = (
                    chronicle_receipt,
                    native_continuity,
                    chronicle_authority,
                )
                if any(value is not None for value in chronicle_values) and (
                    not isinstance(chronicle_receipt, Mapping)
                    or not isinstance(native_continuity, Mapping)
                    or not isinstance(chronicle_authority, str)
                    or len(chronicle_authority) != 64
                ):
                    raise _fail(
                        "fresh-owner checkpoint Chronicle continuation is incomplete"
                    )
                terminal_result, terminal_traces = self._execute_owner(
                    provider=terminal_provider,
                    owner_id=terminal_owner_id,
                    candidate=candidate,
                    request=request,
                    batch_request=full_request,
                    context=terminal_context,
                    start_generation=split,
                    resume_projections=tuple(projections),
                    prior_metrics_rows=first_result["metrics_rows"],
                    prior_checkpoint_catalog={str(split): checkpoint},
                    prior_native_chronicle_receipt=(
                        chronicle_receipt
                        if isinstance(chronicle_receipt, Mapping)
                        else None
                    ),
                    prior_native_continuity=(
                        native_continuity
                        if isinstance(native_continuity, Mapping)
                        else None
                    ),
                    prior_chronicle_checkpoint_authority_sha256=(
                        chronicle_authority
                        if isinstance(chronicle_authority, str)
                        else None
                    ),
                    qualification_observer=qualification_observer,
                )
                traces = [*first_traces, *terminal_traces]
        except (CampaignProviderError, ValueError, TypeError) as exc:
            raise _fail(f"candidate resident execution failed: {exc}") from exc

        for index, trace in enumerate(traces):
            trace["segment_index"] = index
        result_without_digest = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "request_sha256": request.request_sha256,
            "candidate_authority_sha256": request.candidate_authority_sha256,
            "candidate_artifact_sha256s": dict(request.candidate_artifact_sha256s),
            "parity_dimension_sha256s": (
                cross_storage_candidate_dimension_sha256s_v1(
                    request.plan,
                    cell_ids=list(full_request.cell_ids),
                )
            ),
            "plan_sha256": request.plan["plan_sha256"],
            "batch_id": full_request.batch_id,
            "cell_ids": list(full_request.cell_ids),
            "reporting_enabled": full_request.reporting_enabled,
            "production_authority": False,
            "source_owner_id": source_owner_id,
            "terminal_owner_id": terminal_owner_id,
            "fresh_owner": request.split_generation is not None,
            "restore_receipts": restore_rows,
            "segments": traces,
            "terminal_checkpoint": _json_copy(
                terminal_result["checkpoint"], name="terminal checkpoint"
            ),
            "metrics_rows": _json_copy(
                terminal_result["metrics_rows"], name="metrics rows"
            ),
            "provider_result": _json_copy(
                terminal_result["result"], name="provider result"
            ),
            "qualification_receipts": (
                {
                    "capacity_fault_control": None,
                    "capacity_fault_observation": None,
                    "fixed_action_control": None,
                    "fixed_action_observation": None,
                    "resident_dispatch_observations": [],
                }
                if qualification_observer is None
                else qualification_observer.payload()
            ),
            "direct_projection_evidence": (
                None
                if qualification_observer is None
                else qualification_observer.direct_projection_capture_payload()
            ),
        }
        result = _json_copy(result_without_digest, name="candidate result")
        result["result_sha256"] = canonical_sha256(result)
        return result


__all__ = [
    "CandidateExposureCapacityV1",
    "CHECKPOINT_SOURCE_SCHEMA_VERSION",
    "CandidateCheckpointSourceV1",
    "InheritanceCrossPathCandidateSegmentOwnerError",
    "InheritanceCrossPathCandidateSegmentOwnerV1",
    "InheritanceCrossPathCandidateSegmentRequestV1",
    "REQUEST_SCHEMA_VERSION",
    "RESTORE_RECEIPT_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "SEGMENT_TRACE_SCHEMA_VERSION",
]
