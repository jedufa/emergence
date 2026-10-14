"""Typed bridge from isolation preregistration to candidate segment execution.

The bridge owns no mechanic and grants no production authority.  It derives a
mini-plan and resident request from one registered run, clones only explicitly
mapped compiled-plan template cells, delegates execution to the qualification
candidate owner, and delegates evidence conversion to the strict direct
projection boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Final

from engine.native_runtime.inheritance_cross_path_qualified_artifact_v1 import (
    CandidateInheritanceCrossPathArtifactV1,
    InheritanceCrossPathCandidateArtifactLoaderV1,
    UPSTREAM_BINDING_IDS,
)

from .canonical import canonical_sha256
from .inheritance_cross_path_candidate_direct_projection_v1 import (
    project_candidate_segment_result_to_direct_mechanic_v1,
)
from .inheritance_cross_path_candidate_plan_v1 import (
    InheritanceCrossPathCandidatePlanV1,
)
from .inheritance_cross_path_candidate_segment_owner_v1 import (
    CandidateExposureCapacityV1,
    CandidateCheckpointSourceV1,
    InheritanceCrossPathCandidateSegmentOwnerV1,
    InheritanceCrossPathCandidateSegmentRequestV1,
)
from .production_engine_provider import PROFILE_ID, PROVIDER_ID
from .inheritance_cross_path_mechanic_gate_adapter_v1 import (
    EXACT_FULL_STORAGE_MODE,
    InheritanceCrossPathMechanicGateCallbackV1,
)
from .inheritance_cross_path_native_qualification_observer_v1 import (
    ResidentAttemptCapacityFaultDirectiveV1,
)
from .mixed_mode_resident_isolation_plan_composer_v1 import (
    SEGMENT_REQUEST_SCHEMA_VERSION,
)
from .mixed_mode_resident_isolation_runner import (
    MixedModeResidentIsolationRunnerPlanV1,
    ResidentIsolationExecutionRequestV1,
)
from .providers import ResidentBatchRequestV1


BRIDGE_SCHEMA_VERSION: Final[str] = (
    "eiel_inheritance_cross_path_candidate_isolation_bridge_v1"
)
_SEGMENT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "candidate_artifact_binding_sha256",
        "run_id",
        "run_root",
        "owner_root",
        "checkpoint_root",
        "checkpoint_role",
        "checkpoint_source_run_id",
        "checkpoint_source_root",
        "overflow_control",
        "generations",
        "ticks_per_generation",
        "checkpoint_split_generation",
        "reporting_enabled",
        "exposure",
        "physical_order",
        "segment_request_sha256",
    }
)


class InheritanceCrossPathCandidateIsolationBridgeError(ValueError):
    """A preregistered run cannot be mapped to exact candidate execution."""


def _fail(message: str) -> InheritanceCrossPathCandidateIsolationBridgeError:
    return InheritanceCrossPathCandidateIsolationBridgeError(message)


def _copy(value: Any, *, name: str) -> Any:
    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [thaw(child) for child in item]
        return item

    try:
        return json.loads(
            json.dumps(
                thaw(value),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise _fail(f"{name} must contain canonical JSON data") from exc


def _safe_id(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise _fail(f"{name} must be a safe non-empty identifier")
    return value


def _absolute_path(value: Any, *, name: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or path.is_symlink():
        raise _fail(f"{name} must be an absolute non-symlink path")
    return path


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{name} must be an integer >= {minimum}")
    return value


def _exact_segment_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SEGMENT_FIELDS:
        raise _fail("segment request field set is invalid")
    row = _copy(value, name="segment request")
    supplied = row.pop("segment_request_sha256")
    if (
        row.get("schema_version") != SEGMENT_REQUEST_SCHEMA_VERSION
        or not isinstance(supplied, str)
        or canonical_sha256(row) != supplied
    ):
        raise _fail("segment request digest or schema is invalid")
    return {**row, "segment_request_sha256": supplied}


class InheritanceCrossPathCandidateIsolationBridgeV1:
    """Tombstone for the retired standalone candidate isolation bridge."""

    def __init__(
        self,
        *,
        runner_plan: MixedModeResidentIsolationRunnerPlanV1,
        candidate_plan: InheritanceCrossPathCandidatePlanV1,
        logical_template_cell_ids: Mapping[str, str],
        candidate_loader: InheritanceCrossPathCandidateArtifactLoaderV1 | None = None,
        candidate_owner: InheritanceCrossPathCandidateSegmentOwnerV1 | None = None,
        direct_projector: Callable[..., Mapping[str, Any]] | None = None,
        fixed_action_required: bool = True,
        storage_observation_required: bool = True,
    ) -> None:
        raise _fail(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "candidate isolation bridge construction is disabled"
        )
        # Historical implementation retained temporarily for evidence reading.
        if type(runner_plan) is not MixedModeResidentIsolationRunnerPlanV1:
            raise _fail("runner_plan must be a runner-plan v1 value")
        if type(candidate_plan) is not InheritanceCrossPathCandidatePlanV1:
            raise _fail("candidate_plan must be a candidate-plan v1 value")
        if (
            type(fixed_action_required) is not bool
            or type(storage_observation_required) is not bool
        ):
            raise _fail("qualification observation flags must be boolean")
        if candidate_plan.storage_mode != EXACT_FULL_STORAGE_MODE:
            raise _fail("isolation callback requires an exact_full candidate")
        plan = candidate_plan.plan
        binding = plan.get("candidate_binding")
        artifacts = plan.get("artifacts")
        if (
            plan.get("resolved_profile") != PROFILE_ID
            or not isinstance(binding, Mapping)
            or binding.get("production_authority") is not False
            or binding.get("candidate_authority_sha256")
            != candidate_plan.candidate_authority_sha256
            or binding.get("binding_manifest_sha256")
            != candidate_plan.binding_manifest_sha256
            or not isinstance(artifacts, Mapping)
            or set(artifacts) != set(UPSTREAM_BINDING_IDS)
        ):
            raise _fail("candidate plan binding is incomplete")

        if candidate_loader is None:
            cuda_row = artifacts.get("cuda_extension")
            native_abi = (
                cuda_row.get("abi") if isinstance(cuda_row, Mapping) else None
            )
            if not isinstance(native_abi, str) or not native_abi:
                raise _fail("candidate plan lacks its native ABI")
            candidate_loader = InheritanceCrossPathCandidateArtifactLoaderV1(
                artifacts,
                provider_id=PROVIDER_ID,
                profile_id=PROFILE_ID,
                expected_native_abi=native_abi,
            )
        if not isinstance(
            candidate_loader, InheritanceCrossPathCandidateArtifactLoaderV1
        ):
            raise _fail("candidate_loader must be the qualification-only loader")
        expected_artifacts = {
            artifact_id: str(artifacts[artifact_id]["sha256"])
            for artifact_id in UPSTREAM_BINDING_IDS
        }
        if (
            candidate_loader.candidate_authority_sha256
            != candidate_plan.candidate_authority_sha256
            or candidate_loader.candidate_bindings != expected_artifacts
        ):
            raise _fail("candidate loader differs from the immutable candidate plan")

        owner = candidate_owner or InheritanceCrossPathCandidateSegmentOwnerV1()
        if not callable(getattr(owner, "execute", None)):
            raise _fail("candidate_owner must expose an execute callback")
        projector = (
            direct_projector
            or project_candidate_segment_result_to_direct_mechanic_v1
        )
        if not callable(projector):
            raise _fail("direct_projector must be callable")

        templates = plan.get("cells")
        if not isinstance(templates, list):
            raise _fail("candidate plan lacks compiled template cells")
        by_id = {
            row.get("cell_id"): dict(row)
            for row in templates
            if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
        }
        if len(by_id) != len(templates):
            raise _fail("candidate plan template cells are malformed")
        if (
            not isinstance(logical_template_cell_ids, Mapping)
            or not logical_template_cell_ids
            or any(
                not isinstance(logical_id, str)
                or not logical_id
                or template_id not in by_id
                for logical_id, template_id in logical_template_cell_ids.items()
            )
            or len(set(logical_template_cell_ids.values()))
            != len(logical_template_cell_ids)
        ):
            raise _fail("logical template-cell mapping is invalid")
        declared_logical_ids = {
            cell.config.logical_cell_id
            for run in runner_plan.runs
            for cell in run.cells
        }
        if set(logical_template_cell_ids) != declared_logical_ids:
            raise _fail(
                "logical template-cell mapping must cover the preregistered cells exactly"
            )

        for run in runner_plan.runs:
            payload = run.execution_payload
            if (
                payload.get("candidate_authority_sha256")
                != candidate_plan.candidate_authority_sha256
                or payload.get("exposure_storage_mode")
                != EXACT_FULL_STORAGE_MODE
            ):
                raise _fail(
                    f"registered run candidate binding differs: {run.run_id}"
                )
            segment = _exact_segment_request(payload.get("segment_request"))
            if (
                segment["candidate_artifact_binding_sha256"]
                != candidate_plan.binding_manifest_sha256
            ):
                raise _fail(
                    f"registered run artifact binding differs: {run.run_id}"
                )

        self.runner_plan = runner_plan
        self.candidate_plan = candidate_plan
        self.logical_template_cell_ids = dict(logical_template_cell_ids)
        self._templates = by_id
        self.candidate_loader = candidate_loader
        self.candidate_owner = owner
        self.direct_projector = projector
        self.fixed_action_required = fixed_action_required
        self.storage_observation_required = storage_observation_required
        self._callback_authority_sha256: str | None = None
        self._path_ids: tuple[str, ...] | None = None
        self._checkpoint_sources: dict[str, CandidateCheckpointSourceV1] = {}

    def build_callback(
        self,
        *,
        direct_segment_owner_id: str,
        direct_segment_owner_sha256: str,
        path_ids: Sequence[str],
    ) -> InheritanceCrossPathMechanicGateCallbackV1:
        """Bind this bridge to the already-preregistered callback authority."""

        callback = InheritanceCrossPathMechanicGateCallbackV1(
            candidate_loader=self.candidate_loader,
            direct_segment_owner=self.execute_direct_segment,
            direct_segment_owner_id=direct_segment_owner_id,
            direct_segment_owner_sha256=direct_segment_owner_sha256,
            path_ids=path_ids,
        )
        if (
            callback.execution_authority_sha256
            != self.runner_plan.execution_authority_sha256
        ):
            raise _fail(
                "runner plan execution authority differs from the real bridge callback"
            )
        self._callback_authority_sha256 = callback.execution_authority_sha256
        self._path_ids = tuple(path_ids)
        return callback

    def _cloned_cells(
        self,
        request: ResidentIsolationExecutionRequestV1,
        segment: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        registered = tuple(
            sorted(request.run.cells, key=lambda cell: cell.batch_position)
        )
        physical = segment.get("physical_order")
        if not isinstance(physical, list) or len(physical) != len(registered):
            raise _fail("segment physical order differs from registered cells")
        result: list[dict[str, Any]] = []
        for position, (cell, physical_row) in enumerate(
            zip(registered, physical, strict=True)
        ):
            if not isinstance(physical_row, Mapping):
                raise _fail("segment physical-order row is malformed")
            expected = {
                "occurrence_id": cell.occurrence_id,
                "logical_cell_id": cell.config.logical_cell_id,
                "batch_position": position,
                "path_id": cell.config.reproduction_path,
                "neighbor_path_ids": list(cell.expected_neighbor_path_ids),
            }
            if any(physical_row.get(name) != value for name, value in expected.items()):
                raise _fail(
                    f"segment physical identity differs: {cell.occurrence_id}"
                )
            template_id = self.logical_template_cell_ids[
                cell.config.logical_cell_id
            ]
            template = self._templates[template_id]
            dimensions = template.get("dimensions")
            parameters = template.get("parameters")
            seeds = template.get("seeds")
            if (
                not isinstance(parameters, Mapping)
                or not isinstance(seeds, Mapping)
                or not isinstance(dimensions, Mapping)
                or dimensions.get("reproduction_path")
                != cell.config.reproduction_path
                or dimensions.get("inheritance_mode") != physical_row.get("mode_id")
                or dimensions.get("paired_seed") != physical_row.get("seed")
                or physical_row.get("seed") not in set(seeds.values())
            ):
                raise _fail(
                    f"compiled logical template differs: {cell.config.logical_cell_id}"
                )
            cloned = _copy(template, name=f"template {template_id}")
            cloned["cell_id"] = cell.occurrence_id
            if "ordinal" in cloned:
                cloned["ordinal"] = position
            result.append(cloned)
        return result

    def candidate_request_for(
        self,
        *,
        runner_request: ResidentIsolationExecutionRequestV1,
        segment_request: Mapping[str, Any],
        reporting_enabled: bool,
        overflow_fault: Mapping[str, Any] | None,
        fresh_owner: bool,
        checkpoint_source: CandidateCheckpointSourceV1 | None = None,
    ) -> InheritanceCrossPathCandidateSegmentRequestV1:
        """Compile one registered callback occurrence into an exact mini-plan."""

        if not isinstance(
            runner_request, ResidentIsolationExecutionRequestV1
        ) or runner_request.run not in self.runner_plan.runs:
            raise _fail("runner request is not registered by this bridge plan")
        if (
            runner_request.runner_plan_sha256 != self.runner_plan.digest_sha256
            or runner_request.gate_plan_sha256
            != self.runner_plan.gate_plan.digest_sha256
            or runner_request.execution_callback_id
            != self.runner_plan.execution_callback_id
            or runner_request.execution_authority_sha256
            != self.runner_plan.execution_authority_sha256
        ):
            raise _fail("runner request binding differs from the bridge plan")
        segment = _exact_segment_request(segment_request)
        if (
            segment["run_id"] != runner_request.run.run_id
            or segment["reporting_enabled"] is not reporting_enabled
            or runner_request.run.fresh_owner is not fresh_owner
            or runner_request.run.execution_payload.get("reporting_enabled")
            is not reporting_enabled
            or runner_request.run.execution_payload.get("overflow_fault")
            != overflow_fault
        ):
            raise _fail("callback flags differ from the registered run")

        generations = _integer(
            segment["generations"], name="generations", minimum=1
        )
        ticks = _integer(
            segment["ticks_per_generation"],
            name="ticks_per_generation",
            minimum=1,
        )
        split = _integer(
            segment["checkpoint_split_generation"],
            name="checkpoint_split_generation",
            minimum=1,
        )
        if split >= generations:
            raise _fail("checkpoint split must be interior to the run")
        milestones = (split, generations)
        role = segment.get("checkpoint_role")
        if fresh_owner:
            if role != "fresh_owner_continuation":
                raise _fail("fresh-owner run lacks its continuation role")
            split_generation: int | None = split
            source_run_id = segment.get("checkpoint_source_run_id")
            if (
                not isinstance(checkpoint_source, CandidateCheckpointSourceV1)
                or checkpoint_source.source_run_id != source_run_id
            ):
                raise _fail(
                    "fresh-owner request lacks its preregistered checkpoint source"
                )
        else:
            if role == "fresh_owner_continuation":
                raise _fail("non-fresh run cannot request fresh-owner continuation")
            split_generation = None
            if checkpoint_source is not None:
                raise _fail("non-fresh run cannot carry a checkpoint source")

        cells = self._cloned_cells(runner_request, segment)
        cell_ids = tuple(cell["cell_id"] for cell in cells)
        base = _copy(self.candidate_plan.plan, name="candidate plan")
        base.pop("plan_sha256", None)
        base.pop("comparison_branches", None)
        base["cells"] = cells
        base["physical_batches"] = [
            {
                "ordinal": 0,
                "batch_id": runner_request.run.run_id,
                "cell_ids": list(cell_ids),
            }
        ]
        execution = dict(base.get("execution") or {})
        execution.pop("technical_gate", None)
        execution.update(
            {
                "generations": generations,
                "ticks_per_generation": ticks,
                "checkpoints": list(milestones),
                "checkpoint_storage_mode": "disk",
                "reporting_enabled": reporting_enabled,
            }
        )
        base["execution"] = execution
        base["plan_sha256"] = canonical_sha256(base)

        parameter_names = tuple(sorted(cells[0]["parameters"]))
        seed_names = tuple(sorted(cells[0]["seeds"]))
        if any(
            tuple(sorted(cell["parameters"])) != parameter_names
            or tuple(sorted(cell["seeds"])) != seed_names
            for cell in cells
        ):
            raise _fail("cloned template parameter or seed fields differ")
        batch_request = ResidentBatchRequestV1(
            profile_id=PROFILE_ID,
            batch_id=runner_request.run.run_id,
            cell_ids=cell_ids,
            generations=generations,
            ticks_per_generation=ticks,
            checkpoint_milestones=milestones,
            parameter_arrays=tuple(
                (
                    name,
                    tuple(cell["parameters"][name] for cell in cells),
                )
                for name in parameter_names
            ),
            seed_arrays=tuple(
                (name, tuple(cell["seeds"][name] for cell in cells))
                for name in seed_names
            ),
            agent_id_allocation_generation_count=generations,
            checkpoint_storage_mode="disk",
            reporting_enabled=reporting_enabled,
            execution_route_sha256=(
                self.candidate_plan.candidate_authority_sha256
            ),
        )

        fault_directive = None
        control = segment.get("overflow_control")
        if overflow_fault is not None:
            if (
                not isinstance(control, Mapping)
                or control.get("target_occurrence_id")
                != overflow_fault.get("target_occurrence_id")
                or control.get("error_code") != overflow_fault.get("error_code")
                or control.get("capacity_kind") != "attempt"
            ):
                raise _fail("overflow callback differs from its segment control")
            target_id = str(control["target_occurrence_id"])
            try:
                target_index = cell_ids.index(target_id)
            except ValueError as exc:
                raise _fail("overflow target is absent from the mini-batch") from exc
            target_registered = next(
                cell
                for cell in runner_request.run.cells
                if cell.occurrence_id == target_id
            )
            fault_directive = ResidentAttemptCapacityFaultDirectiveV1(
                target_occurrence_id=target_id,
                target_occurrence_authority_sha256=canonical_sha256(
                    target_registered.payload()
                ),
                target_cell_index=target_index,
                target_cell_authority_sha256=canonical_sha256(
                    cells[target_index]
                ),
                effective_bound=control.get("effective_bound"),
                expected_status_code=control.get("expected_status_code"),
            )
        elif control is not None:
            raise _fail("non-fault run carries an overflow control")

        return InheritanceCrossPathCandidateSegmentRequestV1(
            request_id=runner_request.run.run_id,
            candidate_authority_sha256=(
                self.candidate_plan.candidate_authority_sha256
            ),
            candidate_artifact_sha256s={
                artifact_id: str(
                    self.candidate_plan.plan["artifacts"][artifact_id]["sha256"]
                )
                for artifact_id in UPSTREAM_BINDING_IDS
            },
            plan=base,
            batch_request=batch_request,
            runtime_root=_absolute_path(
                segment["run_root"], name="segment run_root"
            ),
            exposure_capacity=CandidateExposureCapacityV1(
                storage_mode=str(segment["exposure"]["storage_mode"]),
                attempt_capacity=int(segment["exposure"]["A"]),
                snapshot_capacity=int(segment["exposure"]["S"]),
                eligible_capacity=int(segment["exposure"]["C"]),
                aggregate_capacity=int(segment["exposure"]["K"]),
            ),
            split_generation=split_generation,
            checkpoint_source=checkpoint_source,
            capacity_fault=fault_directive,
            fixed_action_required=self.fixed_action_required,
            storage_observation_required=self.storage_observation_required,
        )

    def execute_direct_segment(
        self,
        *,
        candidate: CandidateInheritanceCrossPathArtifactV1,
        runner_request: ResidentIsolationExecutionRequestV1,
        segment_request: Mapping[str, Any],
        reporting_enabled: bool,
        overflow_fault: Mapping[str, Any] | None,
        fresh_owner: bool,
    ) -> Mapping[str, Any]:
        """Execute the typed candidate request and strictly project its evidence."""

        if self._callback_authority_sha256 is None or self._path_ids is None:
            raise _fail("bridge callback authority has not been bound")
        request = self.candidate_request_for(
            runner_request=runner_request,
            segment_request=segment_request,
            reporting_enabled=reporting_enabled,
            overflow_fault=overflow_fault,
            fresh_owner=fresh_owner,
            checkpoint_source=(
                self._checkpoint_sources.get(
                    str(segment_request.get("checkpoint_source_run_id"))
                )
                if fresh_owner
                else None
            ),
        )
        candidate_result = self.candidate_owner.execute(candidate, request)
        segment = _exact_segment_request(segment_request)
        if not fresh_owner and segment.get("checkpoint_role") == (
            "uninterrupted_baseline"
        ):
            split = int(segment["checkpoint_split_generation"])
            source_trace = next(
                (
                    trace
                    for trace in candidate_result.get("segments", ())
                    if isinstance(trace, Mapping)
                    and isinstance(trace.get("checkpoint"), Mapping)
                    and trace["checkpoint"].get("generation") == split
                ),
                None,
            )
            if source_trace is None:
                raise _fail(
                    "checkpoint baseline result lacks the preregistered split"
                )
            self._checkpoint_sources[runner_request.run.run_id] = (
                CandidateCheckpointSourceV1(
                    source_run_id=runner_request.run.run_id,
                    source_owner_id=str(candidate_result["source_owner_id"]),
                    source_result_sha256=str(candidate_result["result_sha256"]),
                    checkpoint=source_trace["checkpoint"],
                )
            )
        return self.direct_projector(
            candidate_result,
            runner_request=runner_request,
            callback_authority_sha256=self._callback_authority_sha256,
            path_ids=self._path_ids,
        )


__all__ = [
    "BRIDGE_SCHEMA_VERSION",
    "InheritanceCrossPathCandidateIsolationBridgeError",
    "InheritanceCrossPathCandidateIsolationBridgeV1",
]
