"""Python owner for typed cross-path controls over the Patch resident batch.

This module extends an already prepared Patch multicell owner.  It allocates no
simulation fallback and does not replace the accepted 160-argument ABI.  A
dedicated native entrypoint receives that ABI followed by the ordered resident
control and linkage-sidecar tensors declared below.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final, Mapping, Sequence

import torch

from engine.campaigns.inheritance_cross_path_patch_adapters import (
    CELL_CONTROL_SCHEMA_VERSION,
    CrossPathCellControlV1,
)
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    PreparedDirectResidentPatchMulticellV1,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    DIRECT_ARGUMENT_COUNT,
    GENERATION_ARGUMENT_INDEX,
    PHYSICAL_WINDOW_GENERATIONS,
    RESULT_SCHEMA_VERSION,
    resident_window_count_v1,
)
from engine.perf.cuda_architecture_b_linkage_resident_sidecar_v1 import (
    CudaLinkageResidentSidecarV1,
    allocate_cuda_linkage_resident_sidecar_v1,
)
from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED,
    CudaResidentReproductionExposureSidecarV1,
    ResidentReproductionExposureShapeV1,
    TENSOR_NAMES_V1 as EXPOSURE_TENSOR_NAMES_V1,
    allocate_cuda_resident_reproduction_exposure_sidecar_v1,
)
from engine.evolution.inheritance_linkage_digest_table_v1 import (
    LinkageDigestTableV1,
)


SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_cross_path_resident_overlay_v1"
)
CONTROL_CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_cross_path_control_checkpoint_v1"
)
OVERLAY_CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_cross_path_overlay_checkpoint_v1"
)
CROSS_PATH_DIRECT_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_cross_path_windows_v1"
)

PATH_GENERATION_BOUNDARY_REFILL: Final[int] = 0
PATH_ORDINARY_LIVE_BIRTH: Final[int] = 1
SELECTION_GENOTYPE_BLIND_NEUTRAL: Final[int] = 0
INHERITANCE_LEGACY_MIDPOINT: Final[int] = 0
INHERITANCE_LINKED_PARENT_COPY: Final[int] = 1

_PATH_CODE: Final[dict[str, int]] = {
    "generation_boundary_refill": PATH_GENERATION_BOUNDARY_REFILL,
    "ordinary_live_birth": PATH_ORDINARY_LIVE_BIRTH,
}
_INHERITANCE_CODE: Final[dict[str, int]] = {
    "legacy_midpoint_v1": INHERITANCE_LEGACY_MIDPOINT,
    "linked_parent_copy_v1": INHERITANCE_LINKED_PARENT_COPY,
}
_BLIND_MASK_ALL: Final[int] = 0b1111
_HEX: Final[frozenset[str]] = frozenset("0123456789abcdef")

CONTROL_TENSOR_NAMES_V1: Final[tuple[str, ...]] = (
    "registry_inheritance_mode_ids",
    "registry_founder_pattern_ids",
    "registry_reproduction_path_ids",
    "reproduction_path_codes",
    "selection_mode_codes",
    "inheritance_mode_codes",
    "inheritance_donor_draw_budgets",
    "generation_boundary_refill_enabled",
    "ordinary_live_birth_enabled",
    "mutation_enabled",
    "neutral_fixture_enabled",
    "mutation_weather_enabled",
    "selective_mating_enabled",
    "ecological_trait_authority_enabled",
    "family_head_authority_enabled",
    "modular_output_authority_enabled",
    "mate_required_for_ordinary_birth",
    "selection_seed_by_cell",
    "neutral_repro_threshold_by_cell",
    "selection_blind_mask_by_cell",
)

LINKAGE_TENSOR_NAMES_V1: Final[tuple[str, ...]] = (
    "genotype_codes",
    "digest_rank_by_code",
    "canonical_parent_0_slots",
    "canonical_parent_1_slots",
    "child_slots",
    "donor_ordinals",
    "donor_statuses",
    "staged_counts",
    "committed_counts",
    "donor_draws_added",
    "transaction_status",
)

CROSS_PATH_NATIVE_ARGUMENT_NAMES_V1: Final[tuple[str, ...]] = (
    *CONTROL_TENSOR_NAMES_V1,
    *LINKAGE_TENSOR_NAMES_V1,
    *EXPOSURE_TENSOR_NAMES_V1,
)
CROSS_PATH_NATIVE_ARGUMENT_COUNT_V1: Final[int] = len(
    CROSS_PATH_NATIVE_ARGUMENT_NAMES_V1
)


class CudaCrossPathResidentOverlayError(RuntimeError):
    """Cross-path state cannot be admitted to the CUDA resident owner."""


def _fail(message: str) -> CudaCrossPathResidentOverlayError:
    return CudaCrossPathResidentOverlayError(message)


def _canonical_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _fail("cross-path identity must be canonical JSON data") from exc
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _cuda_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise _fail("cross-path resident overlay requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise _fail("CUDA is unavailable; cross-path CPU fallback is prohibited")
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _manifest(tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().to(device="cpu").contiguous()
    raw = cpu.numpy().tobytes(order="C")
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _selection_seed_rows(
    cell_ids: tuple[str, ...],
    selection_seed_by_cell: Mapping[str, int],
) -> tuple[int, ...]:
    if not isinstance(selection_seed_by_cell, Mapping) or set(
        selection_seed_by_cell
    ) != set(cell_ids):
        raise _fail("selection seeds must map every physical cell exactly once")
    result: list[int] = []
    for cell_id in cell_ids:
        value = selection_seed_by_cell[cell_id]
        if type(value) is not int or not 0 <= value <= 0xFFFF_FFFF_FFFF_FFFF:
            raise _fail("selection seeds must be unsigned 64-bit integers")
        result.append(value)
    return tuple(result)

def _neutral_threshold_rows(
    cell_ids: tuple[str, ...],
    values_by_cell: Mapping[str, float],
) -> tuple[float, ...]:
    if not isinstance(values_by_cell, Mapping) or set(values_by_cell) != set(
        cell_ids
    ):
        raise _fail(
            "neutral reproduction thresholds must map every physical cell"
        )
    result: list[float] = []
    for cell_id in cell_ids:
        value = values_by_cell[cell_id]
        if type(value) not in {int, float} or not 0.0 <= float(value) < float(
            "inf"
        ):
            raise _fail(
                "neutral reproduction thresholds must be finite nonnegative numbers"
            )
        result.append(float(value))
    return tuple(result)


def _validate_control_rows(
    controls: Sequence[CrossPathCellControlV1],
    *,
    cell_ids: tuple[str, ...],
) -> tuple[CrossPathCellControlV1, ...]:
    values = tuple(controls)
    if (
        len(values) != len(cell_ids)
        or any(type(value) is not CrossPathCellControlV1 for value in values)
        or tuple(value.cell_id for value in values) != cell_ids
    ):
        raise _fail("cross-path controls must match exact physical cell order")
    for value in values:
        path_code = _PATH_CODE.get(value.reproduction_path)
        inheritance_code = _INHERITANCE_CODE.get(value.inheritance_mode)
        if path_code is None or inheritance_code is None:
            raise _fail("cross-path control uses an unsupported native protocol value")
        if (
            value.schema_version != CELL_CONTROL_SCHEMA_VERSION
            or value.reproduction_path_id != path_code
            or value.inheritance_mode_id != inheritance_code
            or value.generation_boundary_refill_enabled
            == value.ordinary_live_birth_enabled
            or value.generation_boundary_refill_enabled
            != (path_code == PATH_GENERATION_BOUNDARY_REFILL)
            or value.ordinary_live_birth_enabled
            != (path_code == PATH_ORDINARY_LIVE_BIRTH)
            or (
                inheritance_code == INHERITANCE_LEGACY_MIDPOINT
                and value.inheritance_donor_draw_budget != 0
            )
            or (
                inheritance_code == INHERITANCE_LINKED_PARENT_COPY
                and value.inheritance_donor_draw_budget < 1
            )
            or value.mutation_enabled
            or not value.neutral_fixture_enabled
            or value.mutation_weather_enabled
            or value.selective_mating_enabled
            or value.ecological_trait_authority_enabled
            or value.family_head_authority_enabled
            or value.modular_output_authority_enabled
            or not value.mate_required_for_ordinary_birth
        ):
            raise _fail("cross-path control contradicts the qualified native ABI")
        selection = value.selection
        if (
            selection.mode != "genotype_blind_neutral"
            or selection.eligible_parent_set_override
            or not selection.uniform_over_eligible_set
            or not selection.deterministic_schedule_required
            or not selection.genotype_blind
            or not selection.lineage_blind
            or not selection.policy_blind
            or not selection.strategy_blind
        ):
            raise _fail("cross-path selection is not exact neutral selection")
    return values


@dataclass(slots=True)
class CudaCrossPathResidentControlsV1:
    """Tombstone owner for retired standalone CUDA cross-path controls."""

    cell_ids: tuple[str, ...]
    registry_sha256: str
    source_controls: tuple[CrossPathCellControlV1, ...]
    selection_seeds: tuple[int, ...]
    neutral_repro_thresholds: tuple[float, ...]
    registry_inheritance_mode_ids: torch.Tensor
    registry_founder_pattern_ids: torch.Tensor
    registry_reproduction_path_ids: torch.Tensor
    reproduction_path_codes: torch.Tensor
    selection_mode_codes: torch.Tensor
    inheritance_mode_codes: torch.Tensor
    inheritance_donor_draw_budgets: torch.Tensor
    generation_boundary_refill_enabled: torch.Tensor
    ordinary_live_birth_enabled: torch.Tensor
    mutation_enabled: torch.Tensor
    neutral_fixture_enabled: torch.Tensor
    mutation_weather_enabled: torch.Tensor
    selective_mating_enabled: torch.Tensor
    ecological_trait_authority_enabled: torch.Tensor
    family_head_authority_enabled: torch.Tensor
    modular_output_authority_enabled: torch.Tensor
    mate_required_for_ordinary_birth: torch.Tensor
    selection_seed_by_cell: torch.Tensor
    neutral_repro_threshold_by_cell: torch.Tensor
    selection_blind_mask_by_cell: torch.Tensor

    def __post_init__(self) -> None:
        raise _fail(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "CUDA cross-path resident-control construction is disabled"
        )

    @property
    def device(self) -> torch.device:
        return self.reproduction_path_codes.device

    @property
    def batch_cells(self) -> int:
        return len(self.cell_ids)

    def tensor_map(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name) for name in CONTROL_TENSOR_NAMES_V1}

    @property
    def native_arguments(self) -> tuple[torch.Tensor, ...]:
        return tuple(self.tensor_map()[name] for name in CONTROL_TENSOR_NAMES_V1)

    def identity(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "registry_sha256": self.registry_sha256,
            "cell_ids": list(self.cell_ids),
            "selection_seeds": list(self.selection_seeds),
            "neutral_repro_thresholds": list(
                self.neutral_repro_thresholds
            ),
            "source_controls": [value.payload() for value in self.source_controls],
        }

    def validate_exact(self) -> None:
        if not _valid_sha256(self.registry_sha256):
            raise _fail("cross-path controls lack a registry SHA-256")
        _validate_control_rows(self.source_controls, cell_ids=self.cell_ids)
        expected_dtype = {
            "registry_inheritance_mode_ids": torch.int64,
            "registry_founder_pattern_ids": torch.int64,
            "registry_reproduction_path_ids": torch.int64,
            "reproduction_path_codes": torch.int8,
            "selection_mode_codes": torch.int8,
            "inheritance_mode_codes": torch.int8,
            "inheritance_donor_draw_budgets": torch.int64,
            "generation_boundary_refill_enabled": torch.bool,
            "ordinary_live_birth_enabled": torch.bool,
            "mutation_enabled": torch.bool,
            "neutral_fixture_enabled": torch.bool,
            "mutation_weather_enabled": torch.bool,
            "selective_mating_enabled": torch.bool,
            "ecological_trait_authority_enabled": torch.bool,
            "family_head_authority_enabled": torch.bool,
            "modular_output_authority_enabled": torch.bool,
            "mate_required_for_ordinary_birth": torch.bool,
            "selection_seed_by_cell": torch.uint64,
            "neutral_repro_threshold_by_cell": torch.float64,
            "selection_blind_mask_by_cell": torch.uint8,
        }
        tensors = self.tensor_map()
        if any(
            not isinstance(tensor, torch.Tensor)
            or tensor.device != self.device
            or tensor.device.type != "cuda"
            or tensor.dtype != expected_dtype[name]
            or tuple(tensor.shape) != (self.batch_cells,)
            or tensor.requires_grad
            for name, tensor in tensors.items()
        ):
            raise _fail("cross-path control tensors have invalid CUDA ABI shape")
        expected = _control_host_values(self.source_controls)
        expected["selection_seed_by_cell"] = self.selection_seeds
        expected["neutral_repro_threshold_by_cell"] = (
            self.neutral_repro_thresholds
        )
        for name, values in expected.items():
            observed = tuple(tensors[name].detach().cpu().tolist())
            if observed != values:
                raise _fail(f"cross-path resident control drifted at {name}")

    def export_checkpoint_projection(self) -> dict[str, Any]:
        self.validate_exact()
        tensors = {
            name: value.detach().to(device="cpu").clone()
            for name, value in self.tensor_map().items()
        }
        manifests = {name: _manifest(value) for name, value in tensors.items()}
        identity = self.identity()
        digest = _canonical_sha256(
            {
                "schema_version": CONTROL_CHECKPOINT_SCHEMA_VERSION,
                "identity": identity,
                "tensor_manifests": manifests,
            }
        )
        return {
            "schema_version": CONTROL_CHECKPOINT_SCHEMA_VERSION,
            "identity": identity,
            "tensors": tensors,
            "tensor_manifests": manifests,
            "projection_sha256": digest,
        }


def _control_host_values(
    controls: tuple[CrossPathCellControlV1, ...],
) -> dict[str, tuple[Any, ...]]:
    return {
        "registry_inheritance_mode_ids": tuple(
            value.inheritance_mode_id for value in controls
        ),
        "registry_founder_pattern_ids": tuple(
            value.founder_pattern_id for value in controls
        ),
        "registry_reproduction_path_ids": tuple(
            value.reproduction_path_id for value in controls
        ),
        "reproduction_path_codes": tuple(
            _PATH_CODE[value.reproduction_path] for value in controls
        ),
        "selection_mode_codes": tuple(
            SELECTION_GENOTYPE_BLIND_NEUTRAL for _value in controls
        ),
        "inheritance_mode_codes": tuple(
            _INHERITANCE_CODE[value.inheritance_mode] for value in controls
        ),
        "inheritance_donor_draw_budgets": tuple(
            value.inheritance_donor_draw_budget for value in controls
        ),
        "generation_boundary_refill_enabled": tuple(
            value.generation_boundary_refill_enabled for value in controls
        ),
        "ordinary_live_birth_enabled": tuple(
            value.ordinary_live_birth_enabled for value in controls
        ),
        "mutation_enabled": tuple(value.mutation_enabled for value in controls),
        "neutral_fixture_enabled": tuple(
            value.neutral_fixture_enabled for value in controls
        ),
        "mutation_weather_enabled": tuple(
            value.mutation_weather_enabled for value in controls
        ),
        "selective_mating_enabled": tuple(
            value.selective_mating_enabled for value in controls
        ),
        "ecological_trait_authority_enabled": tuple(
            value.ecological_trait_authority_enabled for value in controls
        ),
        "family_head_authority_enabled": tuple(
            value.family_head_authority_enabled for value in controls
        ),
        "modular_output_authority_enabled": tuple(
            value.modular_output_authority_enabled for value in controls
        ),
        "mate_required_for_ordinary_birth": tuple(
            value.mate_required_for_ordinary_birth for value in controls
        ),
        "selection_blind_mask_by_cell": tuple(
            _BLIND_MASK_ALL for _value in controls
        ),
    }


def materialize_cuda_cross_path_resident_controls_v1(
    controls: Sequence[CrossPathCellControlV1],
    *,
    cell_ids: Sequence[str],
    selection_seed_by_cell: Mapping[str, int],
    neutral_repro_threshold_by_cell: Mapping[str, float],
    registry_sha256: str,
    device: str | torch.device,
) -> CudaCrossPathResidentControlsV1:
    """Reject retired standalone CUDA control materialization."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "CUDA cross-path resident-control materialization is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    resolved = _cuda_device(device)
    ordered_ids = tuple(cell_ids)
    if (
        not ordered_ids
        or len(set(ordered_ids)) != len(ordered_ids)
        or any(not isinstance(value, str) or not value for value in ordered_ids)
    ):
        raise _fail("physical cell IDs must be unique non-empty strings")
    if not _valid_sha256(registry_sha256):
        raise _fail("registry_sha256 must be a lowercase SHA-256")
    rows = _validate_control_rows(controls, cell_ids=ordered_ids)
    seeds = _selection_seed_rows(ordered_ids, selection_seed_by_cell)
    thresholds = _neutral_threshold_rows(
        ordered_ids, neutral_repro_threshold_by_cell
    )
    host = _control_host_values(rows)
    host["selection_seed_by_cell"] = seeds
    host["neutral_repro_threshold_by_cell"] = thresholds
    dtype = {
        "registry_inheritance_mode_ids": torch.int64,
        "registry_founder_pattern_ids": torch.int64,
        "registry_reproduction_path_ids": torch.int64,
        "reproduction_path_codes": torch.int8,
        "selection_mode_codes": torch.int8,
        "inheritance_mode_codes": torch.int8,
        "inheritance_donor_draw_budgets": torch.int64,
        "generation_boundary_refill_enabled": torch.bool,
        "ordinary_live_birth_enabled": torch.bool,
        "mutation_enabled": torch.bool,
        "neutral_fixture_enabled": torch.bool,
        "mutation_weather_enabled": torch.bool,
        "selective_mating_enabled": torch.bool,
        "ecological_trait_authority_enabled": torch.bool,
        "family_head_authority_enabled": torch.bool,
        "modular_output_authority_enabled": torch.bool,
        "mate_required_for_ordinary_birth": torch.bool,
        "selection_seed_by_cell": torch.uint64,
        "neutral_repro_threshold_by_cell": torch.float64,
        "selection_blind_mask_by_cell": torch.uint8,
    }
    owner = CudaCrossPathResidentControlsV1(
        cell_ids=ordered_ids,
        registry_sha256=registry_sha256,
        source_controls=rows,
        selection_seeds=seeds,
        neutral_repro_thresholds=thresholds,
        **{
            name: torch.tensor(host[name], dtype=dtype[name], device=resolved)
            for name in CONTROL_TENSOR_NAMES_V1
        },
    )
    owner.validate_exact()
    return owner


def restore_cuda_cross_path_resident_controls_v1(
    controls: Sequence[CrossPathCellControlV1],
    projection: Mapping[str, Any],
    *,
    cell_ids: Sequence[str],
    selection_seed_by_cell: Mapping[str, int],
    neutral_repro_threshold_by_cell: Mapping[str, float],
    registry_sha256: str,
    device: str | torch.device,
) -> CudaCrossPathResidentControlsV1:
    """Reject retired standalone CUDA control restoration."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "CUDA cross-path resident-control restoration is disabled"
    )
    owner = materialize_cuda_cross_path_resident_controls_v1(
        controls,
        cell_ids=cell_ids,
        selection_seed_by_cell=selection_seed_by_cell,
        neutral_repro_threshold_by_cell=neutral_repro_threshold_by_cell,
        registry_sha256=registry_sha256,
        device=device,
    )
    if (
        not isinstance(projection, Mapping)
        or projection.get("schema_version") != CONTROL_CHECKPOINT_SCHEMA_VERSION
        or projection.get("identity") != owner.identity()
        or not isinstance(projection.get("tensors"), Mapping)
        or not isinstance(projection.get("tensor_manifests"), Mapping)
    ):
        raise _fail("control checkpoint identity does not match declarations")
    tensors = projection["tensors"]
    if set(tensors) != set(CONTROL_TENSOR_NAMES_V1) or any(
        not isinstance(value, torch.Tensor) or value.device.type != "cpu"
        for value in tensors.values()
    ):
        raise _fail("control checkpoint tensors must be complete CPU transport")
    manifests = {name: _manifest(tensors[name]) for name in CONTROL_TENSOR_NAMES_V1}
    digest = _canonical_sha256(
        {
            "schema_version": CONTROL_CHECKPOINT_SCHEMA_VERSION,
            "identity": owner.identity(),
            "tensor_manifests": manifests,
        }
    )
    if (
        projection.get("tensor_manifests") != manifests
        or projection.get("projection_sha256") != digest
        or any(
            not torch.equal(
                tensors[name],
                owner.tensor_map()[name].detach().to(device="cpu"),
            )
            for name in CONTROL_TENSOR_NAMES_V1
        )
    ):
        raise _fail("control checkpoint bytes differ from declared controls")
    return owner


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PreparedCrossPathDirectResidentPatchMulticellV1:
    """Tombstone owner for the retired standalone CUDA cross-path overlay."""

    schema_version: str
    patch: PreparedDirectResidentPatchMulticellV1
    controls: CudaCrossPathResidentControlsV1
    linkage: CudaLinkageResidentSidecarV1
    exposure: CudaResidentReproductionExposureSidecarV1

    def __post_init__(self) -> None:
        raise _fail(
            "P12_RETIRED_INHERITANCE_AUTHORITY "
            "CUDA cross-path overlay construction is disabled"
        )

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self.patch.cell_ids

    @property
    def batch_cells(self) -> int:
        return self.patch.batch_cells

    @property
    def physical_window_arguments(self) -> tuple[Any, ...]:
        return self.patch.physical_window_arguments

    @property
    def source_projections(self) -> tuple[Any, ...]:
        """Expose the canonical Patch sources expected by provider mutation checks."""

        return self.patch.source_projections

    @property
    def native_overlay_arguments(self) -> tuple[torch.Tensor, ...]:
        linkage = tuple(
            getattr(self.linkage, name) for name in LINKAGE_TENSOR_NAMES_V1
        )
        exposure = tuple(
            getattr(self.exposure, name) for name in EXPOSURE_TENSOR_NAMES_V1
        )
        return (*self.controls.native_arguments, *linkage, *exposure)

    def validate_exact(self) -> None:
        if (
            self.schema_version != SCHEMA_VERSION
            or not isinstance(self.patch, PreparedDirectResidentPatchMulticellV1)
            or self.controls.cell_ids != self.patch.cell_ids
            or self.linkage.shape.batch_cells != self.patch.batch_cells
            or self.exposure.shape.batch_cells != self.patch.batch_cells
            or self.controls.device != self.linkage.device
            or self.controls.device != self.exposure.device
        ):
            raise _fail("prepared cross-path overlay ownership is inconsistent")
        self.controls.validate_exact()
        self.exposure.validate_exact()
        qualification_failures = tuple(
            int(cell)
            for cell in torch.nonzero(
                self.exposure.qualification_attempt_capacity_enabled
                & (
                    self.exposure.cell_status
                    == CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED
                ),
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        self.linkage.validate_exact(
            allowed_failure_cell_indices=qualification_failures
        )
        live_banks = getattr(self.patch.patch, "live_banks", None)
        energy = (
            live_banks.get("energy")
            if isinstance(live_banks, Mapping)
            else None
        )
        if (
            not isinstance(energy, torch.Tensor)
            or energy.device != self.controls.device
            or energy.dtype not in {torch.float32, torch.float64}
            or tuple(energy.shape)
            != (
                self.patch.batch_cells,
                self.linkage.shape.slot_capacity,
            )
        ):
            raise _fail(
                "cross-path sidecar does not match Patch resident slot ownership"
            )
        if (
            len(self.native_overlay_arguments)
            != CROSS_PATH_NATIVE_ARGUMENT_COUNT_V1
        ):
            raise _fail("prepared cross-path native argument count drifted")
        block_count = self.linkage.shape.block_count
        for control in self.controls.source_controls:
            if (
                control.inheritance_mode == "linked_parent_copy_v1"
                and control.inheritance_donor_draw_budget != block_count
            ):
                raise _fail("linked control draw budget differs from sidecar blocks")

    def export_checkpoint_projection(self) -> dict[str, Any]:
        self.validate_exact()
        qualification_failures = tuple(
            int(cell)
            for cell in torch.nonzero(
                self.exposure.qualification_attempt_capacity_enabled
                & (
                    self.exposure.cell_status
                    == CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED
                ),
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        controls = self.controls.export_checkpoint_projection()
        linkage = self.linkage.export_checkpoint_projection(
            allowed_failure_cell_indices=qualification_failures
        )
        exposure = self.exposure.export_checkpoint_projection()
        identity = {
            "schema_version": OVERLAY_CHECKPOINT_SCHEMA_VERSION,
            "registry_sha256": self.controls.registry_sha256,
            "cell_ids": list(self.cell_ids),
            "control_projection_sha256": controls["projection_sha256"],
            "linkage_projection_sha256": linkage["projection_sha256"],
            "exposure_projection_sha256": exposure["projection_sha256"],
        }
        return {
            **identity,
            "controls": controls,
            "linkage": linkage,
            "exposure": exposure,
            "projection_sha256": _canonical_sha256(identity),
        }


def prepare_cross_path_direct_resident_overlay_v1(
    patch: PreparedDirectResidentPatchMulticellV1,
    controls: Sequence[CrossPathCellControlV1],
    tables: LinkageDigestTableV1 | Sequence[LinkageDigestTableV1],
    *,
    selection_seed_by_cell: Mapping[str, int],
    neutral_repro_threshold_by_cell: Mapping[str, float],
    registry_sha256: str,
    birth_capacity: int,
    device: str | torch.device,
    exposure_shape: ResidentReproductionExposureShapeV1 | None = None,
    source_genotype_codes: torch.Tensor | None = None,
    checkpoint_projection: Mapping[str, Any] | None = None,
) -> PreparedCrossPathDirectResidentPatchMulticellV1:
    """Reject retired standalone CUDA overlay preparation."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "CUDA cross-path overlay preparation is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if not isinstance(patch, PreparedDirectResidentPatchMulticellV1):
        raise _fail("patch must be PreparedDirectResidentPatchMulticellV1")
    if (source_genotype_codes is None) == (checkpoint_projection is None):
        raise _fail(
            "supply exactly one of source_genotype_codes or checkpoint_projection"
        )
    if checkpoint_projection is None:
        resident_controls = materialize_cuda_cross_path_resident_controls_v1(
            controls,
            cell_ids=patch.cell_ids,
            selection_seed_by_cell=selection_seed_by_cell,
            neutral_repro_threshold_by_cell=neutral_repro_threshold_by_cell,
            registry_sha256=registry_sha256,
            device=device,
        )
        linkage = allocate_cuda_linkage_resident_sidecar_v1(
            tables,
            source_genotype_codes=source_genotype_codes,
            birth_capacity=birth_capacity,
            device=device,
        )
        if exposure_shape is None:
            exposure_shape = ResidentReproductionExposureShapeV1(
                batch_cells=patch.batch_cells,
                attempt_capacity=1,
                snapshot_capacity=2,
                eligible_capacity=int(source_genotype_codes.shape[1]),
            )
            exposure_enabled = (False,) * patch.batch_cells
        else:
            exposure_enabled = (True,) * patch.batch_cells
        exposure = allocate_cuda_resident_reproduction_exposure_sidecar_v1(
            exposure_shape,
            cell_enabled=exposure_enabled,
            device=device,
        )
    else:
        if (
            checkpoint_projection.get("schema_version")
            != OVERLAY_CHECKPOINT_SCHEMA_VERSION
            or checkpoint_projection.get("registry_sha256") != registry_sha256
            or checkpoint_projection.get("cell_ids") != list(patch.cell_ids)
            or not isinstance(checkpoint_projection.get("controls"), Mapping)
            or not isinstance(checkpoint_projection.get("linkage"), Mapping)
            or not isinstance(checkpoint_projection.get("exposure"), Mapping)
        ):
            raise _fail("overlay checkpoint identity differs from prepared Patch cells")
        identity = {
            name: checkpoint_projection[name]
            for name in (
                "schema_version",
                "registry_sha256",
                "cell_ids",
                "control_projection_sha256",
                "linkage_projection_sha256",
                "exposure_projection_sha256",
            )
        }
        if checkpoint_projection.get("projection_sha256") != _canonical_sha256(
            identity
        ):
            raise _fail("overlay checkpoint projection hash mismatch")
        resident_controls = restore_cuda_cross_path_resident_controls_v1(
            controls,
            checkpoint_projection["controls"],
            cell_ids=patch.cell_ids,
            selection_seed_by_cell=selection_seed_by_cell,
            neutral_repro_threshold_by_cell=neutral_repro_threshold_by_cell,
            registry_sha256=registry_sha256,
            device=device,
        )
        exposure = CudaResidentReproductionExposureSidecarV1.from_checkpoint_projection(
            checkpoint_projection["exposure"],
            device=device,
        )
        qualification_failures = tuple(
            int(cell)
            for cell in torch.nonzero(
                exposure.qualification_attempt_capacity_enabled
                & (
                    exposure.cell_status
                    == CELL_STATUS_ATTEMPT_CAPACITY_EXCEEDED
                ),
                as_tuple=False,
            )
            .flatten()
            .tolist()
        )
        linkage = CudaLinkageResidentSidecarV1.from_checkpoint_projection(
            tables,
            checkpoint_projection["linkage"],
            device=device,
            allowed_failure_cell_indices=qualification_failures,
        )
        if (
            checkpoint_projection["control_projection_sha256"]
            != checkpoint_projection["controls"].get("projection_sha256")
            or checkpoint_projection["linkage_projection_sha256"]
            != checkpoint_projection["linkage"].get("projection_sha256")
            or checkpoint_projection["exposure_projection_sha256"]
            != checkpoint_projection["exposure"].get("projection_sha256")
        ):
            raise _fail("overlay checkpoint child binding mismatch")
    prepared = PreparedCrossPathDirectResidentPatchMulticellV1(
        schema_version=SCHEMA_VERSION,
        patch=patch,
        controls=resident_controls,
        linkage=linkage,
        exposure=exposure,
    )
    prepared.validate_exact()
    if (
        prepared.linkage.shape.birth_capacity != birth_capacity
        or prepared.linkage.shape.slot_capacity
        != int(prepared.linkage.genotype_codes.shape[1])
    ):
        raise _fail("overlay linkage shape differs from requested capacity")
    return prepared


def _mutation_disabled(value: Any) -> bool:
    if type(value) is bool:
        return not value
    if isinstance(value, torch.Tensor):
        return value.device.type == "cuda" and not bool(value.any().item())
    return False


def execute_cross_path_direct_resident_windows_v1(
    module: Any,
    *,
    prepared: PreparedCrossPathDirectResidentPatchMulticellV1,
    logical_generation_count: int,
    policy_mutation_enable: Any,
    policy_mutation_mode: Any,
    policy_mutation_rate: Any,
    policy_mutation_std: Any,
) -> dict[str, Any]:
    """Reject retired standalone CUDA cross-path dispatch."""

    raise _fail(
        "P12_RETIRED_INHERITANCE_AUTHORITY "
        "CUDA cross-path resident dispatch is disabled"
    )
    # Historical implementation retained temporarily for evidence reading.
    if not isinstance(prepared, PreparedCrossPathDirectResidentPatchMulticellV1):
        raise _fail("prepared must be a cross-path Patch resident overlay")
    prepared.validate_exact()
    arguments = tuple(prepared.physical_window_arguments)
    if len(arguments) != DIRECT_ARGUMENT_COUNT:
        raise _fail(
            f"cross-path dispatch requires the exact {DIRECT_ARGUMENT_COUNT}-slot Patch ABI"
        )
    if type(logical_generation_count) is not int or not (
        1 <= logical_generation_count <= 128
    ):
        raise _fail("logical_generation_count must be an integer in [1, 128]")
    initial_generation = arguments[GENERATION_ARGUMENT_INDEX]
    if type(initial_generation) is not int or initial_generation < 0:
        raise _fail("prepared initial generation must be nonnegative")
    if not _mutation_disabled(policy_mutation_enable):
        raise _fail("cross-path mutation-off execution rejects policy mutation")
    entrypoint = getattr(module, CROSS_PATH_DIRECT_ENTRYPOINT, None)
    if not callable(entrypoint):
        raise _fail(
            f"CUDA extension is missing callable {CROSS_PATH_DIRECT_ENTRYPOINT}"
        )
    result = entrypoint(
        *arguments,
        logical_generation_count,
        policy_mutation_enable,
        policy_mutation_mode,
        policy_mutation_rate,
        policy_mutation_std,
        *prepared.native_overlay_arguments,
        prepared.controls.registry_sha256,
    )
    if not isinstance(result, Mapping):
        raise _fail("cross-path native result must be a mapping")
    result = dict(result)
    expected = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "owner": "native_extension",
        "physical_window_generations": PHYSICAL_WINDOW_GENERATIONS,
        "logical_generation_count": logical_generation_count,
        "window_count": resident_window_count_v1(
            initial_generation=initial_generation,
            logical_generation_count=logical_generation_count,
        ),
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "completed_generations": logical_generation_count,
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "cross_path_resident_control_v1": True,
        "cross_path_hidden_fallback": False,
        "cross_path_cell_count": prepared.batch_cells,
        "cross_path_native_argument_count": (
            CROSS_PATH_NATIVE_ARGUMENT_COUNT_V1
        ),
        "cross_path_registry_sha256": prepared.controls.registry_sha256,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"cross-path native result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    prepared.validate_exact()
    return result


__all__ = [
    "CONTROL_CHECKPOINT_SCHEMA_VERSION",
    "CONTROL_TENSOR_NAMES_V1",
    "CROSS_PATH_DIRECT_ENTRYPOINT",
    "CROSS_PATH_NATIVE_ARGUMENT_COUNT_V1",
    "CROSS_PATH_NATIVE_ARGUMENT_NAMES_V1",
    "CudaCrossPathResidentControlsV1",
    "CudaCrossPathResidentOverlayError",
    "LINKAGE_TENSOR_NAMES_V1",
    "OVERLAY_CHECKPOINT_SCHEMA_VERSION",
    "PreparedCrossPathDirectResidentPatchMulticellV1",
    "SCHEMA_VERSION",
    "execute_cross_path_direct_resident_windows_v1",
    "materialize_cuda_cross_path_resident_controls_v1",
    "prepare_cross_path_direct_resident_overlay_v1",
    "restore_cuda_cross_path_resident_controls_v1",
]
