"""Reusable provider wiring for the Patch cross-path resident overlay.

The campaign supplies only science-specific source materialization.  This
module owns the generic Patch request/preparation/execution/projection adapters
and binds the overlay checkpoint into every canonical Bundle-v2 projection.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Any, Callable, Final, Mapping, Protocol, Sequence
import weakref

import torch

from engine.evolution.inheritance_linkage_digest_table_v1 import (
    LinkageDigestTableV1,
)
from engine.perf.cuda_architecture_b_cross_path_resident_overlay_v1 import (
    PreparedCrossPathDirectResidentPatchMulticellV1,
    execute_cross_path_direct_resident_windows_v1,
    prepare_cross_path_direct_resident_overlay_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    DirectResidentPatchMulticellSettlementV1,
    PreparedDirectResidentPatchMulticellV1,
    prepare_direct_resident_patch_multicell_v1,
    project_direct_resident_patch_multicell_checkpoints_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
    build_direct_resident_source_allocation_request_v1,
)
from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED,
    ResidentReproductionExposureShapeV1,
)

from .production_engine_provider import (
    EielProductionExecutionProviderV1,
)
from .inheritance_cross_path_patch_adapters import (
    CrossPathCellControlV1,
    CrossPathPatchRunnerAdapterV1,
)


SCHEMA_VERSION: Final[str] = "eiel_inheritance_cross_path_patch_execution_adapters_v1"
OVERLAY_FIELD: Final[str] = "inheritance_cross_path_overlay_checkpoint_v1"
OVERLAY_BINDING_FIELD: Final[str] = "inheritance_cross_path_overlay_binding_v1"
EXPOSURE_SEGMENT_FIELD: Final[str] = (
    "inheritance_cross_path_reproduction_exposure_segment_v1"
)
ANALYSIS_POPULATION_SEMANTICS: Final[str] = (
    "surviving_source_cohort_after_turnover_v1"
)


class InheritanceCrossPathPatchExecutionError(RuntimeError):
    """The generic provider wiring cannot preserve exact cross-path state."""


def _projection_tensors(
    projection: Any, *, path: str
) -> Mapping[str, torch.Tensor]:
    if isinstance(projection, Mapping):
        raw = projection.get("tensors")
    else:
        raw = getattr(projection, "tensors", None)
    if not isinstance(raw, Mapping):
        raise InheritanceCrossPathPatchExecutionError(
            f"{path} lacks canonical tensor authority"
        )
    return raw


def _cpu_vector(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    dtype: torch.dtype,
    path: str,
) -> torch.Tensor:
    value = tensors.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != dtype
        or value.layout != torch.strided
        or value.ndim != 1
    ):
        raise InheritanceCrossPathPatchExecutionError(
            f"{path}.{name} must be an exact one-dimensional {dtype} tensor"
        )
    return value.detach().to(device="cpu").contiguous()


def _active_id_slots(
    tensors: Mapping[str, torch.Tensor], *, path: str
) -> dict[int, int]:
    active_ids = _cpu_vector(
        tensors,
        "agent_ids_by_alive_order",
        dtype=torch.int64,
        path=path,
    )
    if active_ids.numel() < 1:
        raise InheritanceCrossPathPatchExecutionError(
            f"{path} has no active analysis rows"
        )
    ids = [int(value) for value in active_ids.tolist()]
    if any(value < 0 for value in ids) or len(set(ids)) != len(ids):
        raise InheritanceCrossPathPatchExecutionError(
            f"{path} active IDs are invalid or duplicated"
        )
    dense = _cpu_vector(
        tensors, "agent_id_to_slot", dtype=torch.int32, path=path
    )
    if any(value >= dense.numel() for value in ids):
        raise InheritanceCrossPathPatchExecutionError(
            f"{path} active ID exceeds dense slot-map capacity"
        )
    slots = [int(dense[value].item()) for value in ids]
    if (
        any(value < 0 for value in slots)
        or len(set(slots)) != len(slots)
    ):
        raise InheritanceCrossPathPatchExecutionError(
            f"{path} active slot mapping is invalid or duplicated"
        )
    return dict(zip(ids, slots, strict=True))


def build_cross_path_analysis_source_rows_v1(
    *,
    cell_ids: Sequence[str],
    source_projections: Sequence[Any],
    terminal_projections: Sequence[Any],
    start_generation: int,
    terminal_generation: int,
    suppressed_cell_indices: Sequence[int] = (),
) -> tuple[Mapping[str, Any], ...]:
    """Project exact endpoint values for the surviving source cohort.

    This route does not expose the compact run-many reporting ledger.  Its
    direct owner does expose canonical pre-dispatch and terminal checkpoint
    tensors, so this projection deliberately admits only agents present at both
    endpoints.  The restriction is explicit and fail-closed.
    """

    if (
        type(start_generation) is not int
        or start_generation < 0
        or type(terminal_generation) is not int
        or terminal_generation <= start_generation
    ):
        raise InheritanceCrossPathPatchExecutionError(
            "analysis endpoint generations are invalid"
        )
    cells = tuple(cell_ids)
    sources = tuple(source_projections)
    terminals = tuple(terminal_projections)
    if not cells or len(sources) != len(cells) or len(terminals) != len(cells):
        raise InheritanceCrossPathPatchExecutionError(
            "analysis endpoint projections differ from physical cell order"
        )
    suppressed = tuple(suppressed_cell_indices)
    if (
        len(set(suppressed)) != len(suppressed)
        or any(type(index) is not int or not 0 <= index < len(cells) for index in suppressed)
    ):
        raise InheritanceCrossPathPatchExecutionError(
            "analysis suppressed cell indices are invalid"
        )
    rows: list[Mapping[str, Any]] = []
    for cell_index, (cell_id, source, terminal) in enumerate(
        zip(cells, sources, terminals, strict=True)
    ):
        if cell_index in suppressed:
            continue
        source_tensors = _projection_tensors(
            source, path=f"source_projections[{cell_index}]"
        )
        terminal_tensors = _projection_tensors(
            terminal, path=f"terminal_projections[{cell_index}]"
        )
        source_slots = _active_id_slots(
            source_tensors, path=f"source_projections[{cell_index}].tensors"
        )
        terminal_slots = _active_id_slots(
            terminal_tensors,
            path=f"terminal_projections[{cell_index}].tensors",
        )
        agent_ids = sorted(set(source_slots).intersection(terminal_slots))
        if not agent_ids:
            raise InheritanceCrossPathPatchExecutionError(
                f"analysis endpoint cohort is empty for {cell_id}"
            )
        source_injury = _cpu_vector(
            source_tensors,
            "injury",
            dtype=torch.float32,
            path=f"source_projections[{cell_index}].tensors",
        )
        terminal_injury = _cpu_vector(
            terminal_tensors,
            "injury",
            dtype=torch.float32,
            path=f"terminal_projections[{cell_index}].tensors",
        )
        terminal_lineage = _cpu_vector(
            terminal_tensors,
            "lineage_founder_id",
            dtype=torch.int32,
            path=f"terminal_projections[{cell_index}].tensors",
        )
        terminal_offspring = _cpu_vector(
            terminal_tensors,
            "offspring_count",
            dtype=torch.int32,
            path=f"terminal_projections[{cell_index}].tensors",
        )
        if any(
            slot >= vector.numel()
            for agent_id in agent_ids
            for slot, vector in (
                (source_slots[agent_id], source_injury),
                (terminal_slots[agent_id], terminal_injury),
                (terminal_slots[agent_id], terminal_lineage),
                (terminal_slots[agent_id], terminal_offspring),
            )
        ):
            raise InheritanceCrossPathPatchExecutionError(
                f"analysis endpoint slot exceeds field capacity for {cell_id}"
            )
        rows.append(
            {
                "cell_id": cell_id,
                "start_generation": start_generation,
                "terminal_generation": terminal_generation,
                "population_semantics": ANALYSIS_POPULATION_SEMANTICS,
                "row_agent_ids": agent_ids,
                "source_values": {
                    "reporting.injury_start.v1": [
                        float(source_injury[source_slots[agent_id]].item())
                        for agent_id in agent_ids
                    ],
                    "reporting.injury_end.v1": [
                        float(terminal_injury[terminal_slots[agent_id]].item())
                        for agent_id in agent_ids
                    ],
                    "identity.lineage_founder_id.v1": [
                        int(terminal_lineage[terminal_slots[agent_id]].item())
                        for agent_id in agent_ids
                    ],
                    "reporting.offspring_count.v1": [
                        int(terminal_offspring[terminal_slots[agent_id]].item())
                        for agent_id in agent_ids
                    ],
                },
            }
        )
    return tuple(rows)


def _derive_cross_path_source_envelope_v1(runner: Any) -> Any:
    """Classify base Patch while retaining mate-required overlay state.

    Patch-P2 correctly rejects mate-required reproduction because its base
    owner has no exact recombination RNG.  The qualified cross-path overlay
    owns that mechanic and RNG, so source allocation classifies an immutable
    view with only the delegated flag disabled.  The live runner and imported
    projection retain the actual mate-required value.
    """

    from engine.native_runtime.cuda_architecture_b_backend_factory import (
        derive_cuda_architecture_b_runner_envelope,
    )

    env_cfg = getattr(runner, "env_cfg", None)
    if getattr(env_cfg, "reproduce_requires_mate", None) is not True:
        raise InheritanceCrossPathPatchExecutionError(
            "cross-path source allocation requires mate-required configuration"
        )
    classification_env_cfg = copy.copy(env_cfg)
    classification_env_cfg.reproduce_requires_mate = False
    classification_runner = copy.copy(runner)
    classification_runner.env_cfg = classification_env_cfg
    return derive_cuda_architecture_b_runner_envelope(classification_runner)


@dataclass(frozen=True, slots=True)
class CrossPathBatchSourceMaterializationV1:
    """Campaign-independent inputs needed to attach the resident overlay."""

    tables: LinkageDigestTableV1 | Sequence[LinkageDigestTableV1]
    source_genotype_codes: torch.Tensor | None
    selection_seed_by_cell: Mapping[str, int]
    neutral_repro_threshold_by_cell: Mapping[str, float]
    birth_capacity: int
    exposure_shape: ResidentReproductionExposureShapeV1 | None = None
    device: str = "cuda"


class CrossPathBatchSourceMaterializerV1(Protocol):
    """Materialize founder/linkage inputs from canonical Patch sources."""

    def __call__(
        self,
        *,
        patch: PreparedDirectResidentPatchMulticellV1,
        controls: tuple[CrossPathCellControlV1, ...],
        source_projections: tuple[Mapping[str, Any], ...],
        checkpoint_projection: Mapping[str, Any] | None,
    ) -> CrossPathBatchSourceMaterializationV1: ...


def _overlay_from_sources(
    projections: tuple[Mapping[str, Any], ...],
    *,
    cell_ids: tuple[str, ...],
) -> Mapping[str, Any] | None:
    bindings = [projection.get(OVERLAY_BINDING_FIELD) for projection in projections]
    if all(value is None for value in bindings):
        if any(OVERLAY_FIELD in projection for projection in projections):
            raise InheritanceCrossPathPatchExecutionError(
                "overlay checkpoint is present without per-cell bindings"
            )
        return None
    if any(not isinstance(value, Mapping) for value in bindings):
        raise InheritanceCrossPathPatchExecutionError(
            "overlay checkpoint bindings are incomplete"
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "cell_ids": list(cell_ids),
    }
    if any(
        value.get("schema_version") != expected["schema_version"]
        or value.get("cell_ids") != expected["cell_ids"]
        or value.get("cell_index") != index
        for index, value in enumerate(bindings)
    ):
        raise InheritanceCrossPathPatchExecutionError(
            "overlay checkpoint bindings differ from physical cell order"
        )
    owner_rows = [
        (index, projection[OVERLAY_FIELD])
        for index, projection in enumerate(projections)
        if OVERLAY_FIELD in projection
    ]
    if (
        len(owner_rows) != 1
        or owner_rows[0][0] != 0
        or not isinstance(owner_rows[0][1], Mapping)
    ):
        raise InheritanceCrossPathPatchExecutionError(
            "fresh-owner continuation requires exactly one cell-0 batch overlay owner"
        )
    owner = owner_rows[0][1]
    digest = owner.get("projection_sha256")
    if any(value.get("projection_sha256") != digest for value in bindings):
        raise InheritanceCrossPathPatchExecutionError(
            "per-cell overlay bindings disagree with their batch owner"
        )
    return owner


class CrossPathPatchExecutionAdaptersV1:
    """Tombstone for the retired standalone Patch execution adapters."""

    def __init__(
        self,
        registry: Mapping[str, Any],
        *,
        source_materializer: CrossPathBatchSourceMaterializerV1,
        runner_constructor: Callable[..., Any] | None = None,
    ) -> None:
        raise InheritanceCrossPathPatchExecutionError(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "cross-path execution-adapter construction is disabled"
        )
        # Historical implementation retained temporarily for evidence reading.
        if not callable(source_materializer):
            raise InheritanceCrossPathPatchExecutionError(
                "source_materializer must be callable"
            )
        self.runner_adapter = CrossPathPatchRunnerAdapterV1(
            registry, runner_constructor=runner_constructor
        )
        self.registry_sha256 = self.runner_adapter.registry_sha256
        self._source_materializer = source_materializer
        self._controls_by_runner: dict[int, CrossPathCellControlV1] = {}
        self._controls_by_request: dict[int, CrossPathCellControlV1] = {}
        self._source_authority_by_cell: dict[str, Mapping[str, Any]] = {}
        self._prepared: weakref.ReferenceType[
            PreparedCrossPathDirectResidentPatchMulticellV1
        ] | None = None

    def runner_builder(self, *args: Any, **kwargs: Any) -> Any:
        built = self.runner_adapter(*args, **kwargs)
        self._controls_by_runner[id(built.runner)] = built.cell_control
        from .inheritance_cross_path_source_materializer_v1 import (
            SOURCE_AUTHORITY_FIELD,
        )

        authority = getattr(built.runner, SOURCE_AUTHORITY_FIELD, None)
        if isinstance(authority, Mapping):
            self._source_authority_by_cell[built.cell_control.cell_id] = authority
        return built

    def source_preparer(
        self,
        provider: EielProductionExecutionProviderV1,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Mapping[str, Any], ...]:
        """Run the canonical source factory and bind its exact founder slots."""

        self._source_authority_by_cell.clear()
        from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
            QUALIFIED_CAPABILITY_MASK_V1,
        )
        from .inheritance_cross_path_source_materializer_v1 import (
            SOURCE_AUTHORITY_FIELD,
            bind_source_authority_to_projection_v1,
            bind_runner_source_authority_to_projection_v1,
        )

        projections = tuple(
            provider._prepare_generation_zero_sources_v2(
                *args,
                **kwargs,
                source_capability_mask={
                    **QUALIFIED_CAPABILITY_MASK_V1,
                    "cross_path_overlay": True,
                },
                source_projection_binder=(
                    bind_runner_source_authority_to_projection_v1
                ),
                source_projection_binding_scope="per_cell",
            )
        )
        request = args[1] if len(args) > 1 else None
        cell_ids = tuple(getattr(request, "cell_ids", ()))
        if len(cell_ids) != len(projections):
            raise InheritanceCrossPathPatchExecutionError(
                "source-preparer cell order differs from its resident request"
            )
        try:
            bound = tuple(
                projection
                if isinstance(projection.get(SOURCE_AUTHORITY_FIELD), Mapping)
                else bind_source_authority_to_projection_v1(
                    projection,
                    self._source_authority_by_cell[cell_id],
                )
                for cell_id, projection in zip(cell_ids, projections, strict=True)
            )
        except (KeyError, ValueError) as exc:
            raise InheritanceCrossPathPatchExecutionError(
                "generation-zero source lacks exact founder authority"
            ) from exc
        self._source_authority_by_cell.clear()
        return bound

    def request_builder(self, runner: Any, **kwargs: Any) -> Any:
        control = self._controls_by_runner.pop(id(runner), None)
        if control is None:
            raise InheritanceCrossPathPatchExecutionError(
                "request runner has no registry-derived cross-path control"
            )
        from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
            QUALIFIED_CAPABILITY_MASK_V1,
        )

        request = build_direct_resident_source_allocation_request_v1(
            runner,
            **kwargs,
            runner_envelope_deriver=_derive_cross_path_source_envelope_v1,
            source_capability_mask={
                **QUALIFIED_CAPABILITY_MASK_V1,
                "cross_path_overlay": True,
            },
        )
        self._controls_by_request[id(request)] = control
        return request

    def preparer(
        self, **kwargs: Any
    ) -> PreparedCrossPathDirectResidentPatchMulticellV1:
        requests = tuple(kwargs.get("per_cell_requests") or ())
        controls = tuple(
            self._controls_by_request.pop(id(request), None) for request in requests
        )
        if any(control is None for control in controls):
            raise InheritanceCrossPathPatchExecutionError(
                "prepared requests lack ordered cross-path controls"
            )
        patch = prepare_direct_resident_patch_multicell_v1(**kwargs)
        sources = tuple(kwargs.get("resume_projections") or patch.source_projections)
        overlay_checkpoint = _overlay_from_sources(sources, cell_ids=patch.cell_ids)
        materialized = self._source_materializer(
            patch=patch,
            controls=controls,
            source_projections=sources,
            checkpoint_projection=overlay_checkpoint,
        )
        if type(materialized) is not CrossPathBatchSourceMaterializationV1:
            raise InheritanceCrossPathPatchExecutionError(
                "source materializer returned the wrong typed result"
            )
        if overlay_checkpoint is None and not isinstance(
            materialized.source_genotype_codes, torch.Tensor
        ):
            raise InheritanceCrossPathPatchExecutionError(
                "fresh source materialization requires genotype codes"
            )
        prepared = prepare_cross_path_direct_resident_overlay_v1(
            patch,
            controls,
            materialized.tables,
            selection_seed_by_cell=materialized.selection_seed_by_cell,
            neutral_repro_threshold_by_cell=(
                materialized.neutral_repro_threshold_by_cell
            ),
            registry_sha256=self.registry_sha256,
            birth_capacity=materialized.birth_capacity,
            device=materialized.device,
            exposure_shape=materialized.exposure_shape,
            source_genotype_codes=(
                materialized.source_genotype_codes
                if overlay_checkpoint is None
                else None
            ),
            checkpoint_projection=overlay_checkpoint,
        )
        self._prepared = weakref.ref(prepared)
        return prepared

    def executor(self, module: Any, **kwargs: Any) -> Mapping[str, Any]:
        prepared = None if self._prepared is None else self._prepared()
        physical = tuple(kwargs.pop("physical_window_arguments"))
        if (
            prepared is None
            or len(physical) != len(prepared.physical_window_arguments)
            or any(
                actual is not expected
                for actual, expected in zip(
                    physical, prepared.physical_window_arguments, strict=True
                )
            )
        ):
            raise InheritanceCrossPathPatchExecutionError(
                "executor arguments do not belong to the current overlay owner"
            )
        return execute_cross_path_direct_resident_windows_v1(
            module, prepared=prepared, **kwargs
        )

    def projector(
        self,
        prepared: PreparedCrossPathDirectResidentPatchMulticellV1,
        **kwargs: Any,
    ) -> DirectResidentPatchMulticellSettlementV1:
        retained = None if self._prepared is None else self._prepared()
        if prepared is not retained:
            raise InheritanceCrossPathPatchExecutionError(
                "projector received a stale overlay owner"
            )
        qualification_enabled = getattr(
            prepared.exposure,
            "qualification_attempt_capacity_enabled",
            None,
        )
        cell_status = getattr(prepared.exposure, "cell_status", None)
        if qualification_enabled is None and cell_status is None:
            reporting_suppressed_cell_indices: tuple[int, ...] = ()
        elif (
            not isinstance(qualification_enabled, torch.Tensor)
            or qualification_enabled.dtype != torch.bool
            or qualification_enabled.ndim != 1
            or qualification_enabled.shape[0] != prepared.batch_cells
            or not isinstance(cell_status, torch.Tensor)
            or cell_status.dtype != torch.int32
            or cell_status.ndim != 1
            or cell_status.shape[0] != prepared.batch_cells
        ):
            raise InheritanceCrossPathPatchExecutionError(
                "qualification failure plane differs from physical cell order"
            )
        else:
            reporting_suppressed_cell_indices = tuple(
                cell
                for cell in range(prepared.batch_cells)
                if bool(qualification_enabled[cell].item())
                and int(cell_status[cell].item())
                == CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED
            )
        settlement = project_direct_resident_patch_multicell_checkpoints_v1(
            prepared.patch,
            reporting_suppressed_cell_indices=(
                reporting_suppressed_cell_indices
            ),
            **kwargs,
        )
        analysis_source_rows = build_cross_path_analysis_source_rows_v1(
            cell_ids=prepared.cell_ids,
            source_projections=prepared.source_projections,
            terminal_projections=settlement.checkpoint_projections,
            start_generation=(
                settlement.terminal_generation
                - settlement.completed_generation_count
            ),
            terminal_generation=settlement.terminal_generation,
            suppressed_cell_indices=reporting_suppressed_cell_indices,
        )
        drained = prepared.exposure.drain_segment()
        segment_output = drained.segment_output
        segment_cells = segment_output.get("cells")
        if (
            not isinstance(segment_cells, list)
            or len(segment_cells) != len(prepared.cell_ids)
            or any(
                not isinstance(row, Mapping)
                or row.get("cell_index") != index
                for index, row in enumerate(segment_cells)
            )
        ):
            raise InheritanceCrossPathPatchExecutionError(
                "reproduction exposure drain differs from physical cell order"
            )
        overlay = prepared.export_checkpoint_projection()
        from .inheritance_cross_path_source_materializer_v1 import (
            SOURCE_AUTHORITY_FIELD,
        )

        projections: list[Mapping[str, Any]] = []
        segment_artifacts: list[Mapping[str, Any]] = []
        for index, projection in enumerate(settlement.checkpoint_projections):
            bound = dict(projection)
            source = prepared.source_projections[index]
            source_authority = (
                source.get(SOURCE_AUTHORITY_FIELD)
                if isinstance(source, Mapping)
                else None
            )
            if not isinstance(source_authority, Mapping):
                raise InheritanceCrossPathPatchExecutionError(
                    "checkpoint projection lost founder source authority"
                )
            bound[SOURCE_AUTHORITY_FIELD] = source_authority
            bound[OVERLAY_BINDING_FIELD] = {
                "schema_version": SCHEMA_VERSION,
                "cell_ids": list(prepared.cell_ids),
                "cell_index": index,
                "projection_sha256": overlay["projection_sha256"],
            }
            artifact = {
                "schema_version": segment_output["schema_version"],
                "segment_index": segment_output["segment_index"],
                "capacity_semantics": segment_output["capacity_semantics"],
                "cell_id": prepared.cell_ids[index],
                "cell": segment_cells[index],
                "segment_output_sha256": segment_output[
                    "segment_output_sha256"
                ],
            }
            bound[EXPOSURE_SEGMENT_FIELD] = artifact
            segment_artifacts.append(artifact)
            if index == 0:
                bound[OVERLAY_FIELD] = overlay
            projections.append(bound)
        self._prepared = None
        return replace(
            settlement,
            checkpoint_projections=tuple(projections),
            segment_artifacts=tuple(segment_artifacts),
            analysis_source_rows=analysis_source_rows,
        )


def build_cross_path_patch_execution_provider_v1(
    registry: Mapping[str, Any],
    *,
    native_api: ModuleType,
    source_materializer: CrossPathBatchSourceMaterializerV1,
    torch_module: ModuleType | None = None,
    runner_constructor: Callable[..., Any] | None = None,
    artifact_loader: Callable[..., Any] | None = None,
    device: str = "cuda",
    require_prebuilt: bool = True,
) -> EielProductionExecutionProviderV1:
    """Reject construction of the retired standalone execution provider."""

    raise InheritanceCrossPathPatchExecutionError(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "cross-path execution-provider construction is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if not isinstance(native_api, ModuleType):
        raise InheritanceCrossPathPatchExecutionError(
            "native_api must be the explicit qualified native contract module"
        )
    adapters = CrossPathPatchExecutionAdaptersV1(
        registry,
        source_materializer=source_materializer,
        runner_constructor=runner_constructor,
    )
    if artifact_loader is None:
        from engine.native_runtime.cuda_architecture_b_production_adapter import (
            load_qualified_cuda_artifact,
        )

        artifact_loader = load_qualified_cuda_artifact
    provider_ref: weakref.ReferenceType[
        EielProductionExecutionProviderV1
    ] | None = None

    def source_preparer(*args: Any, **kwargs: Any) -> Any:
        provider = provider_ref() if provider_ref is not None else None
        if provider is None:
            raise InheritanceCrossPathPatchExecutionError(
                "cross-path provider is unavailable during source preparation"
            )
        return adapters.source_preparer(provider, *args, **kwargs)

    provider = EielProductionExecutionProviderV1(
        native_api=native_api,
        torch_module=torch_module,
        device=device,
        require_prebuilt=require_prebuilt,
        artifact_loader=artifact_loader,
        runner_builder=adapters.runner_builder,
        request_builder=adapters.request_builder,
        preparer=adapters.preparer,
        executor=adapters.executor,
        projector=adapters.projector,
        source_preparer=source_preparer,
        generation_series_required=True,
    )
    provider_ref = weakref.ref(provider)
    setattr(provider, "_cross_path_patch_execution_adapters_v1", adapters)
    return provider


__all__ = [
    "CrossPathBatchSourceMaterializationV1",
    "CrossPathBatchSourceMaterializerV1",
    "CrossPathPatchExecutionAdaptersV1",
    "ANALYSIS_POPULATION_SEMANTICS",
    "EXPOSURE_SEGMENT_FIELD",
    "InheritanceCrossPathPatchExecutionError",
    "OVERLAY_BINDING_FIELD",
    "OVERLAY_FIELD",
    "SCHEMA_VERSION",
    "build_cross_path_analysis_source_rows_v1",
    "build_cross_path_patch_execution_provider_v1",
]
