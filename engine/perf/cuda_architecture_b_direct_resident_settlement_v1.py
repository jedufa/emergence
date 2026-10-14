"""Terminal settlement for the qualified Patch P2 b1/g4 direct route."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Final, Protocol

import torch

from engine.perf.cuda_architecture_b_checkpoint import (
    CudaBoundaryCheckpoint,
    project_post_generation_cuda_checkpoint,
)
from perf.cuda_diploid_state_bank_v1 import (
    CudaDiploidStateBankV1,
    checkpoint_metadata_v1,
)
from perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
)
from evolution.genome_upgrade_v2_telemetry import (
    SELECTED_EVENT_LIMIT,
    summarize_genome_upgrade_v2_resident_tensors,
)
from evolution.genome_upgrade_v2 import (
    CONTEXT_COUNT,
    PHENOTYPE_CHANNELS,
    PRODUCT_CHANNELS,
    Q10_SCALE,
    RUNTIME_FACTOR_DELTA_DIVISOR,
    RUNTIME_FACTOR_MAX_Q10,
    RUNTIME_FACTOR_MIN_Q10,
)
from evolution.genome_upgrade_v2_causal_settlement_v1 import (
    project_genome_v2_causal_settlement_v1,
)
from perf.cuda_architecture_b_modular_state_abi import (
    PackedModularStorageV1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    ModularRecurrentMutationReceiptV1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_sequence_v1 import (
    ModularRecurrentMutationSequenceV1,
)
from perf.cuda_architecture_b_boundary import EVENT_COLUMN_NAMES
from engine.perf.cuda_architecture_b_event_sink_v1 import (
    EVENT_SINK_SCHEMA_VERSION,
    EventSinkError,
    validate_event_sink_prefix_reference_v1,
)
from engine.campaigns.production_engine_lifecycle_accounting_v1 import (
    CAPACITY_ACCOUNTING_SCHEMA_VERSION,
    CAPACITY_REASON_AUTHORITY_DIAGNOSTIC_V1,
    CAPACITY_REASON_AUTHORITY_FULL_PREFIX_V1,
    build_lifecycle_population_accounting_v1,
    lifecycle_qacct_receipt_v1,
)
from engine.perf.cuda_architecture_b_resident_reproduction_exposure_sidecar_v1 import (
    CONTINUITY_SCHEMA_VERSION,
    PATH_GENERATION_BOUNDARY_REFILL,
    SEGMENT_OUTPUT_SCHEMA_VERSION,
    STORAGE_MODE_COMPACT_COMPLETE,
)
from engine.perf.cuda_architecture_b_direct_resident_runtime_v1 import (
    RESULT_SCHEMA_VERSION,
    resident_window_count_v1,
)
from perf.cxxbackbone10_resident_state_owner import (
    _with_checkpoint_contract_metadata,
    projection_digest,
)
from perf.inheritance_sidecar_identity_v1 import (
    INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
    inheritance_sidecar_payload_sha256_v1,
)
from simulation.chronicle.schema import canonical_json


SETTLEMENT_SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_direct_resident_settlement_v1"
)
COMPACT_SUMMARY_SCHEMA_VERSION: Final[str] = (
    "eiel_patch_p2_production_compact_summary_v1"
)
GENERATION_SUMMARY_LAYOUT_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_generation_summary_scalar_v1"
)
GENERATION_SUMMARY_COLUMNS: Final[tuple[str, ...]] = (
    "generation",
    "event_cursor_start",
    "event_cursor_pre_turnover",
    "event_cursor_end",
    "alive_after_turnover",
    "energy_sum_after_turnover",
    "births",
    "deaths",
    "failed_births",
    "action_total",
    "move_total",
    "gather_total",
    "moved_total",
    "gathered_total",
    "movement_attempts",
    "movement_blocked_occupied",
    "movement_blocked_bounds",
    "peak_injury",
    "policy_loop_tick_calls",
    "policy_refill_calls",
    "policy_refill_rows",
    "policy_cache_hit_calls",
    "policy_cache_rows_requested",
    "policy_cache_rows_hit",
    "policy_cache_rows_missed",
    "policy_cache_rows_refilled",
    "policy_cache_full_refill_ticks",
    "policy_cache_partial_refill_ticks",
    "patch_food_stock_after_turnover",
    "energy_collected_exact_sum_after_turnover",
    "policy_energy_live_sum_after_turnover",
)

LOGICAL_GENERATION_COUNT: Final[int] = 4
TERMINAL_GENERATION: Final[int] = 4
TICKS_PER_GENERATION: Final[int] = 80
BATCH_CELLS: Final[int] = 1
QUALIFIED_PATCH_GEOMETRIES: Final[frozenset[tuple[int, int]]] = frozenset(
    {(64, 16), (128, 16), (256, 32), (512, 32)}
)
PHYSICAL_WINDOW_GENERATIONS: Final[int] = 2
BIRTH_REASON_NAMES_V1: Final[tuple[str, ...]] = (
    "accepted",
    "no_space",
    "parent_buffer",
    "no_free_slot",
    "unused",
    "conception",
    "gestation",
    "health",
    "cooldown",
    "no_energy",
)
BIRTH_REASON_DENOMINATOR_V1: Final[str] = "ordinary_birth_decision_rows_v1"
BIRTH_ACCEPTED_EVENT_TYPE_V1: Final[int] = 1
BIRTH_FAILED_EVENT_TYPE_V1: Final[int] = 3


class DirectResidentSettlementSourceV1(Protocol):
    """Source fields retained by a prepared direct trajectory."""

    outputs: Mapping[str, Any]
    generation_base: int


class DirectResidentSettlementContextV1(Protocol):
    """Structural subset of ``PreparedDirectResidentTrajectoryV1``."""

    source: DirectResidentSettlementSourceV1
    live_banks: Mapping[str, Any]
    checkpoint_projection_inputs: Mapping[str, Any]


@dataclass(frozen=True)
class DirectResidentTerminalSettlementV1:
    schema_version: str
    compact_summary_rows: tuple[dict[str, Any], ...]
    terminal_checkpoint: CudaBoundaryCheckpoint
    terminal_generation: int
    terminal_tick: int = 0
    segment_artifacts: tuple[Mapping[str, Any], ...] = ()
    lifecycle_qacct_receipt: Mapping[str, Any] | None = None


class CudaArchitectureBDirectResidentSettlementError(RuntimeError):
    """The direct result cannot be settled as the qualified b1/g4 route."""


def _fail(message: str) -> CudaArchitectureBDirectResidentSettlementError:
    return CudaArchitectureBDirectResidentSettlementError(message)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{name} must be a mapping")
    return value


_GENOME_V2_RESIDENT_FIELDS: Final[tuple[str, ...]] = (
    "genome_v2_alleles_q10",
    "genome_v2_modifiers_q10",
    "genome_v2_products_q10",
    "genome_v2_phenotypes_q10",
    "genome_v2_expression_cost_q10",
    "genome_v2_active_product_count",
    "genome_v2_saturated_product_count",
)
_GENOME_V2_CAUSAL_FIELDS: Final[tuple[str, ...]] = tuple(
    f"{producer}_genome_v2_causal_{suffix}"
    for producer in ("birth", "turnover")
    for suffix in (
        "record_rows",
        "record_valid",
        "record_count",
        "record_dropped_count",
        "capture_enabled",
    )
)


def _genome_v2_tensor_sha256_v1(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(str(tuple(int(value) for value in cpu.shape)).encode("ascii"))
    digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _segment_index_chain_and_counts_v1(
    projection: Mapping[str, Any],
) -> tuple[int, str, dict[str, int]]:
    metadata = _mapping(
        projection.get("metadata"),
        name="pre-generation projection metadata",
    )
    segment_index = metadata.get("native_next_segment_index", 0)
    previous_chain = metadata.get("native_chain_sha256", "0" * 64)
    if type(segment_index) is not int or segment_index < 0:
        raise _fail("native segment index is invalid")
    if (
        not isinstance(previous_chain, str)
        or len(previous_chain) != 64
        or any(ch not in "0123456789abcdef" for ch in previous_chain)
    ):
        raise _fail("native segment chain is invalid")
    counts = {
        "attempts": metadata.get("native_cumulative_attempt_count", 0),
        "successes": metadata.get("native_cumulative_success_count", 0),
        "failures": metadata.get("native_cumulative_failure_count", 0),
        "eligible_snapshots": metadata.get("native_cumulative_snapshot_count", 0),
    }
    if (
        any(type(value) is not int or value < 0 for value in counts.values())
        or counts["attempts"] != counts["successes"] + counts["failures"]
    ):
        raise _fail("native cumulative segment counts are invalid")
    return segment_index, previous_chain, counts


def _provider_native_segment_artifact_v1(
    *,
    cell_id: str,
    cell_index: int,
    source_projection: Mapping[str, Any],
    compact_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if type(cell_id) is not str or not cell_id:
        raise _fail("provider-native segment artifact requires cell_id")
    segment_index, previous_chain, prior_counts = (
        _segment_index_chain_and_counts_v1(source_projection)
    )
    attempts: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    for row in compact_rows:
        births = int(row.get("births", 0) or 0)
        failed_births = int(row.get("failed_births", 0) or 0)
        alive = max(0, int(row.get("alive", 0) or 0))
        if births > 0:
            success_count += 1
            attempts.append(
                {
                    "attempt_index": len(attempts),
                    "path": PATH_GENERATION_BOUNDARY_REFILL,
                    "generation": int(row["generation"]),
                    "tick": 0,
                    "eligible_parent_counts": [alive, alive],
                    "selected_parent_ids": [-1, -1],
                    "success": True,
                    "failure_code": 0,
                    "commit_ordinal": int(row["generation"]),
                    "authoritative_commit": None,
                    "compact_generation_summary": dict(row),
                }
            )
        if failed_births > 0 or births == 0:
            failure_count += 1
            attempts.append(
                {
                    "attempt_index": len(attempts),
                    "path": PATH_GENERATION_BOUNDARY_REFILL,
                    "generation": int(row["generation"]),
                    "tick": 0,
                    "eligible_parent_counts": [alive, alive],
                    "selected_parent_ids": [-1, -1],
                    "success": False,
                    "failure_code": 1,
                    "commit_ordinal": None,
                    "authoritative_commit": None,
                    "compact_generation_summary": dict(row),
                }
            )
    payload = {
        "cell_index": cell_index,
        "enabled": True,
        "status": 0,
        "attempt_count": len(attempts),
        "success_count": success_count,
        "failure_count": failure_count,
        "snapshot_count": 0,
        "attempts": attempts,
        "eligible_parent_snapshots": [],
        "selected_parent_counts": [],
        "parent_contribution_counts": [],
        "primary_eligibility_opportunity_counts": [],
        "mate_eligibility_opportunity_counts": [],
        "path_instrumentation": [],
        "cumulative_counts_after_segment": {
            "attempts": prior_counts["attempts"] + len(attempts),
            "successes": prior_counts["successes"] + success_count,
            "failures": prior_counts["failures"] + failure_count,
            "eligible_snapshots": prior_counts["eligible_snapshots"],
        },
    }
    segment_sha256 = _canonical_sha256(
        {
            "schema_version": SEGMENT_OUTPUT_SCHEMA_VERSION,
            "segment_index": segment_index,
            "cell": payload,
        }
    )
    chain_sha256 = _canonical_sha256(
        {
            "schema_version": CONTINUITY_SCHEMA_VERSION,
            "segment_index": segment_index,
            "previous_chain_sha256": previous_chain,
            "segment_sha256": segment_sha256,
        }
    )
    cell_payload = {
        **payload,
        "segment_sha256": segment_sha256,
        "previous_chain_sha256": previous_chain,
        "chain_sha256_after_segment": chain_sha256,
    }
    segment_core = {
        "schema_version": SEGMENT_OUTPUT_SCHEMA_VERSION,
        "segment_index": segment_index,
        "capacity_semantics": {
            "storage_mode": STORAGE_MODE_COMPACT_COMPLETE,
            "scope": (
                "per_execution_segment_compact_generation_summary_v1"
                if compact_rows
                else "partial_tick_segment_no_generation_summary_v1"
            ),
        },
        "cell_id": cell_id,
        "cell": cell_payload,
    }
    artifact = {
        **segment_core,
        "segment_output_sha256": _canonical_sha256(segment_core),
    }
    metadata_updates = {
        "native_next_segment_index": segment_index + 1,
        "native_chain_sha256": chain_sha256,
        "native_cumulative_attempt_count": payload[
            "cumulative_counts_after_segment"
        ]["attempts"],
        "native_cumulative_success_count": payload[
            "cumulative_counts_after_segment"
        ]["successes"],
        "native_cumulative_failure_count": payload[
            "cumulative_counts_after_segment"
        ]["failures"],
        "native_cumulative_snapshot_count": payload[
            "cumulative_counts_after_segment"
        ]["eligible_snapshots"],
    }
    return artifact, metadata_updates


def _normalized_context_plane_v1(
    values: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if not math.isfinite(scale) or scale <= 0.0:
        raise _fail("Genome V2 observability context scale is invalid")
    return (
        (values.to(dtype=torch.float64) * Q10_SCALE / scale)
        .to(dtype=torch.int64)
        .clamp_(0, Q10_SCALE)
    )


def _genome_v2_observability_inputs_v1(
    persistent_state: Mapping[str, Any],
    bank: CudaGenomeUpgradeV2StateBank,
    *,
    alive_mask: torch.Tensor,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    config = _mapping(
        persistent_state.get("genome_v2_observability_config"),
        name="Genome V2 observability config",
    )
    required_config = {
        "max_energy",
        "start_health",
        "developmental_maturity_age",
        "patch_density_radius",
        "patch_density_include_self",
        "patch_density_normalize",
        "patch_food_amount_max",
        "patch_waste_max",
    }
    if set(config) != required_config:
        raise _fail("Genome V2 observability config fields differ")
    positive_scalars = {
        "max_energy",
        "start_health",
        "patch_food_amount_max",
        "patch_waste_max",
    }
    if any(
        type(config[name]) not in {int, float}
        or not math.isfinite(float(config[name]))
        or float(config[name]) <= 0.0
        for name in positive_scalars
    ):
        raise _fail("Genome V2 observability config scale is invalid")
    if (
        type(config["developmental_maturity_age"]) is not int
        or int(config["developmental_maturity_age"]) < 0
        or type(config["patch_density_radius"]) is not int
        or int(config["patch_density_radius"]) < 0
        or type(config["patch_density_include_self"]) is not bool
        or type(config["patch_density_normalize"]) is not bool
    ):
        raise _fail("Genome V2 observability config type is invalid")
    if (
        not isinstance(alive_mask, torch.Tensor)
        or alive_mask.dtype != torch.bool
        or alive_mask.ndim != 2
        or alive_mask.device != bank.device
    ):
        raise _fail("Genome V2 observability alive mask is invalid")
    cells, capacity = alive_mask.shape
    if (cells, capacity) != (bank.cells, bank.capacity):
        raise _fail("Genome V2 observability alive-mask shape differs")
    required_tensors = {
        "energy",
        "health",
        "injury",
        "age",
        "pos_y",
        "pos_x",
        "patch_density",
        "patch_food_amount",
        "patch_fertility",
        "patch_waste",
        "genome_reproduction_threshold",
        "repro_threshold",
        "repro_cooldown",
        "policy_mem_dim",
        "lineage_founder_id",
        "lineage_depth",
    }
    tensors: dict[str, torch.Tensor] = {}
    for name in required_tensors:
        value = persistent_state.get(name)
        if not isinstance(value, torch.Tensor):
            raise _fail(f"Genome V2 observability source tensor is missing: {name}")
        if value.device != bank.device:
            raise _fail(f"Genome V2 observability source tensor device differs: {name}")
        tensors[name] = value
    for name in required_tensors - {
        "patch_density",
        "patch_food_amount",
        "patch_fertility",
        "patch_waste",
    }:
        if tuple(tensors[name].shape) != (cells, capacity):
            raise _fail(f"Genome V2 observability source tensor shape differs: {name}")
    positions_y = tensors["pos_y"].to(dtype=torch.int64)
    positions_x = tensors["pos_x"].to(dtype=torch.int64)
    patch_density = tensors["patch_density"]
    if patch_density.ndim != 3 or patch_density.size(0) != cells:
        raise _fail("Genome V2 observability Patch density shape differs")
    height, width = int(patch_density.size(1)), int(patch_density.size(2))
    grid_index = (positions_y * width + positions_x).clamp_(0, height * width - 1)

    def world_at_slot(name: str) -> torch.Tensor:
        world = tensors[name]
        if tuple(world.shape) != (cells, height, width):
            raise _fail(f"Genome V2 observability Patch tensor shape differs: {name}")
        return world.reshape(cells, -1).gather(1, grid_index)

    density = world_at_slot("patch_density").to(dtype=torch.float64)
    if not bool(config["patch_density_normalize"]):
        radius = int(config["patch_density_radius"])
        denominator = float((2 * radius + 1) ** 2)
        if not bool(config["patch_density_include_self"]):
            denominator -= 1.0
        density = density / max(denominator, 1.0)
    context = torch.empty(
        (cells, capacity, CONTEXT_COUNT),
        dtype=torch.int64,
        device=alive_mask.device,
    )
    context[:, :, 0] = _normalized_context_plane_v1(
        float(config["max_energy"]) - tensors["energy"],
        float(config["max_energy"]),
    )
    context[:, :, 1] = _normalized_context_plane_v1(
        tensors["injury"],
        float(config["start_health"]),
    )
    context[:, :, 2] = _normalized_context_plane_v1(density, 1.0)
    context[:, :, 3] = _normalized_context_plane_v1(
        world_at_slot("patch_food_amount"),
        float(config["patch_food_amount_max"]),
    )
    context[:, :, 4] = _normalized_context_plane_v1(
        world_at_slot("patch_fertility"),
        1.0,
    )
    context[:, :, 5] = _normalized_context_plane_v1(
        world_at_slot("patch_waste"),
        float(config["patch_waste_max"]),
    )
    age = tensors["age"].to(dtype=torch.int64)
    maturity_age = int(config["developmental_maturity_age"])
    context[:, :, 6] = (
        (age < maturity_age).to(dtype=torch.int64) * Q10_SCALE
        if maturity_age > 0
        else 0
    )
    context[:, :, 7] = (
        (age >= maturity_age * 3).to(dtype=torch.int64) * Q10_SCALE
        if maturity_age > 0
        else 0
    )

    phenotype_index = {name: index for index, name in enumerate(PHENOTYPE_CHANNELS)}

    def runtime_factor(name: str, *, inverse: bool = False) -> torch.Tensor:
        phenotype = bank.phenotypes_q10[:, phenotype_index[name], :]
        delta = torch.div(
            phenotype,
            RUNTIME_FACTOR_DELTA_DIVISOR,
            rounding_mode="trunc",
        )
        value = Q10_SCALE - delta if inverse else Q10_SCALE + delta
        return value.clamp(RUNTIME_FACTOR_MIN_Q10, RUNTIME_FACTOR_MAX_Q10)

    body = runtime_factor("body_size")
    metabolic = runtime_factor("metabolic_efficiency")
    expression_debit = (
        bank.expression_cost_q10.to(dtype=torch.float64)
        * body.to(dtype=torch.float64)
        / (metabolic.to(dtype=torch.float64) * Q10_SCALE)
    )
    consequence_rows: list[dict[str, Any]] = []
    selected_events: list[dict[str, Any]] = []
    for cell in range(cells):
        active_slots = alive_mask[cell].nonzero(as_tuple=False).flatten()
        active_count = int(active_slots.numel())
        threshold_delta = tensors["repro_threshold"][cell, active_slots].to(
            dtype=torch.float64
        ) - tensors["genome_reproduction_threshold"][cell, active_slots].to(
            dtype=torch.float64
        )
        consequence_rows.extend(
            (
                {
                    "cell": cell,
                    "surface": "expression_energy_debit_per_tick",
                    "applied_delta": -float(
                        expression_debit[cell, active_slots].sum().item()
                    ),
                    "cap_bound": False,
                    "observed_count": active_count,
                },
                {
                    "cell": cell,
                    "surface": "reproduction_threshold",
                    "applied_delta": float(threshold_delta.sum().item()),
                    "cap_bound": False,
                    "observed_count": active_count,
                },
            )
        )
        slots = [int(value) for value in active_slots.detach().cpu().tolist()]
        slots.sort(
            key=lambda slot: (
                -int(tensors["lineage_depth"][cell, slot].item()),
                slot,
            )
        )
        for slot in slots[:SELECTED_EVENT_LIMIT]:
            alleles = bank.alleles_q10[cell, :, :, :, slot]
            modifiers = bank.modifiers_q10[cell, :, :, slot]
            products = bank.products_q10[cell, :, slot]
            phenotypes = bank.phenotypes_q10[cell, :, slot]
            founder = int(tensors["lineage_founder_id"][cell, slot].item())
            depth = int(tensors["lineage_depth"][cell, slot].item())
            selected_events.append(
                {
                    "cell": cell,
                    "event_kind": (
                        "birth_or_turnover_child_terminal_snapshot"
                        if depth > 0
                        else "founder_terminal_snapshot"
                    ),
                    "lineage_id": f"founder:{founder}:depth:{depth}",
                    "genotype_summary": {
                        "sha256": _genome_v2_tensor_sha256_v1(
                            torch.cat(
                                (
                                    alleles.reshape(-1),
                                    modifiers.reshape(-1),
                                )
                            )
                        ),
                        "heterozygous_parameter_count": int(
                            alleles[0].ne(alleles[1]).sum().item()
                        ),
                        "heterozygous_modifier_count": int(
                            modifiers[0].ne(modifiers[1]).sum().item()
                        ),
                    },
                    "product_summary": {
                        name: int(products[index].item())
                        for index, name in enumerate(PRODUCT_CHANNELS)
                    },
                    "phenotype_summary": {
                        name: int(phenotypes[index].item())
                        for index, name in enumerate(PHENOTYPE_CHANNELS)
                    },
                    "applied_consequence": {
                        "expression_energy_debit_per_tick": float(
                            expression_debit[cell, slot].item()
                        ),
                        "reproduction_threshold_baseline": float(
                            tensors["genome_reproduction_threshold"][cell, slot].item()
                        ),
                        "reproduction_threshold_applied": float(
                            tensors["repro_threshold"][cell, slot].item()
                        ),
                    },
                    "resulting_state": {
                        "energy": float(tensors["energy"][cell, slot].item()),
                        "health": float(tensors["health"][cell, slot].item()),
                        "reproduction_threshold": float(
                            tensors["repro_threshold"][cell, slot].item()
                        ),
                        "reproduction_cooldown": int(
                            tensors["repro_cooldown"][cell, slot].item()
                        ),
                        "memory_dimension": int(
                            tensors["policy_mem_dim"][cell, slot].item()
                        ),
                    },
                }
            )
    return context, consequence_rows, selected_events


def with_resident_inheritance_sidecars_v1(
    checkpoint: CudaBoundaryCheckpoint,
    persistent_state: Mapping[str, Any],
    *,
    cell_id: str | None,
    genome_v2_checkpoint_authority_binding: Any = (),
    genome_v2_causal_physical_cell_ordinal: int = 0,
    include_genome_v2_causal_projection: bool = True,
) -> CudaBoundaryCheckpoint:
    """Attach exact resident inheritance banks to one canonical checkpoint."""

    if type(include_genome_v2_causal_projection) is not bool:
        raise _fail("Genome V2 causal projection selection must be boolean")
    projection = dict(checkpoint.projection)
    projection.pop("diploid_checkpoint_projection", None)
    projection.pop("genome_upgrade_v2_checkpoint_projection", None)
    metadata = _mapping(projection.get("metadata"), name="terminal metadata")
    base_projection_sha256 = projection_digest(projection)
    sidecar_attached = False
    diploid_storage = persistent_state.get("diploid_allele_storage")
    if diploid_storage is not None:
        try:
            if not isinstance(diploid_storage, torch.Tensor):
                raise TypeError("diploid_allele_storage is not a tensor")
            diploid_bank = CudaDiploidStateBankV1(
                storage=diploid_storage,
                genome_resource_handling_axis=persistent_state[
                    "genome_resource_handling_axis"
                ],
                canonical_hereditary_digest=persistent_state[
                    "canonical_hereditary_digest"
                ],
                canonical_hereditary_template=persistent_state[
                    "canonical_hereditary_template"
                ],
                diploid_payload_template=persistent_state["diploid_payload_template"],
                diploid_genotype_template=persistent_state["diploid_genotype_template"],
                metadata=checkpoint_metadata_v1(
                    cells=int(diploid_storage.size(0)),
                    capacity=int(diploid_storage.size(2)),
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _fail(
                f"terminal diploid checkpoint sidecar is incomplete: {exc}"
            ) from exc
        projection["diploid_checkpoint_projection"] = _identified_sidecar_v1(
            diploid_bank.checkpoint_projection(),
            cell_id=cell_id,
            metadata=metadata,
            base_projection_sha256=base_projection_sha256,
        )
        sidecar_attached = True

    genome_v2_present = [
        name in persistent_state for name in _GENOME_V2_RESIDENT_FIELDS
    ]
    if any(genome_v2_present):
        if not all(genome_v2_present):
            missing = [
                name
                for name, present in zip(
                    _GENOME_V2_RESIDENT_FIELDS,
                    genome_v2_present,
                    strict=True,
                )
                if not present
            ]
            raise _fail(
                "terminal GENOME-UPGRADE-V2 checkpoint sidecar is incomplete: "
                f"missing={missing}"
            )
        try:
            genome_v2_bank = CudaGenomeUpgradeV2StateBank(
                alleles_q10=persistent_state["genome_v2_alleles_q10"],
                modifiers_q10=persistent_state["genome_v2_modifiers_q10"],
                products_q10=persistent_state["genome_v2_products_q10"],
                phenotypes_q10=persistent_state["genome_v2_phenotypes_q10"],
                expression_cost_q10=persistent_state["genome_v2_expression_cost_q10"],
                active_product_count=persistent_state["genome_v2_active_product_count"],
                saturated_product_count=persistent_state[
                    "genome_v2_saturated_product_count"
                ],
            )
        except (TypeError, ValueError) as exc:
            raise _fail(
                f"terminal GENOME-UPGRADE-V2 checkpoint sidecar is invalid: {exc}"
            ) from exc
        checkpoint_tensors = genome_v2_bank.checkpoint_cpu_tensors_v1()
        genome_v2_checkpoint = genome_v2_bank.checkpoint_projection(
            cpu_tensors=checkpoint_tensors
        )
        founder_profile = persistent_state.get("genome_v2_founder_profile")
        if not isinstance(founder_profile, Mapping) or set(founder_profile) != {
            "schema_version",
            "profile_id",
            "payload_sha256",
        }:
            raise _fail(
                "terminal GENOME-UPGRADE-V2 founder profile authority is absent"
            )
        genome_v2_checkpoint["founder_profile"] = dict(founder_profile)
        alive_mask = persistent_state.get("alive_mask")
        population_selection = "authoritative_alive_mask"
        if alive_mask is None:
            alive_mask = torch.ones(
                (genome_v2_bank.cells, genome_v2_bank.capacity),
                dtype=torch.bool,
                device=genome_v2_bank.device,
            )
            population_selection = "all_capacity_legacy_fixture"
        context_q10, consequence_rows, selected_events = (
            _genome_v2_observability_inputs_v1(
                persistent_state,
                genome_v2_bank,
                alive_mask=alive_mask,
            )
        )
        compact_summary = summarize_genome_upgrade_v2_resident_tensors(
            checkpoint_tensors,
            alive_mask=alive_mask.detach().to(
                device="cpu",
                dtype=torch.bool,
                copy=True,
            ),
            context_q10=(
                context_q10.detach().to(
                    device="cpu",
                    dtype=torch.int64,
                    copy=True,
                )
                if context_q10 is not None
                else None
            ),
            consequence_rows=consequence_rows,
            selected_events=selected_events,
            population_selection=population_selection,
            founder_profiles=(founder_profile,),
        )
        causal_present = [
            include_genome_v2_causal_projection and name in persistent_state
            for name in _GENOME_V2_CAUSAL_FIELDS
        ]
        if any(causal_present) and not all(causal_present):
            missing = [
                name
                for name, present in zip(
                    _GENOME_V2_CAUSAL_FIELDS,
                    causal_present,
                    strict=True,
                )
                if not present
            ]
            raise _fail(
                "terminal GENOME-UPGRADE-V2 causal outputs are incomplete: "
                f"missing={missing}"
            )
        if all(causal_present):
            causal_projection = project_genome_v2_causal_settlement_v1(
                bank=genome_v2_bank,
                alive_mask=alive_mask,
                terminal_context_q10=context_q10,
                cell_id=cell_id or "",
                physical_cell_ordinal=(genome_v2_causal_physical_cell_ordinal),
                checkpoint_authority_binding=(genome_v2_checkpoint_authority_binding),
                persistent_state=persistent_state,
            )
            genome_v2_checkpoint["causal_transaction_projection"] = causal_projection
            compact_summary = {
                **compact_summary,
                "causal_transaction_projection": {
                    "schema_version": causal_projection["schema_version"],
                    "capture_status": causal_projection["capture_status"],
                    "captured_counts": causal_projection["captured_counts"],
                    "dropped_counts": causal_projection["dropped_counts"],
                    "record_counts": causal_projection["record_counts"],
                    "evaluable": causal_projection["evaluable"],
                    "reasons": causal_projection["reasons"],
                    "projection_sha256": causal_projection["projection_sha256"],
                },
            }
        genome_v2_checkpoint["compact_generation_summary"] = compact_summary
        projection["genome_upgrade_v2_checkpoint_projection"] = _identified_sidecar_v1(
            genome_v2_checkpoint,
            cell_id=cell_id,
            metadata=metadata,
            base_projection_sha256=base_projection_sha256,
        )
        sidecar_attached = True

    if not sidecar_attached:
        return checkpoint
    return dataclasses.replace(
        checkpoint,
        projection=projection,
        projection_digest=projection_digest(projection),
    )


# Retain the focused-test and internal compatibility name while exposing the
# operation through its public production-boundary name.
_with_resident_inheritance_sidecars_v1 = with_resident_inheritance_sidecars_v1


def _identified_sidecar_v1(
    payload: Mapping[str, Any],
    *,
    cell_id: str | None,
    metadata: Mapping[str, Any],
    base_projection_sha256: str,
) -> dict[str, Any]:
    if type(cell_id) is not str or not cell_id or cell_id.strip() != cell_id:
        raise _fail(
            "inheritance checkpoint sidecars require an explicit stable cell_id"
        )
    generation = metadata.get("generation")
    tick = metadata.get("tick")
    if type(generation) is not int or type(tick) is not int:
        raise _fail("inheritance checkpoint sidecar boundary is invalid")
    result = dict(payload)
    result["identity"] = {
        "schema_version": INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
        "cell_id": cell_id,
        "seed": metadata.get("seed"),
        "generation": generation,
        "tick": tick,
        "base_projection_sha256": base_projection_sha256,
        "payload_sha256": inheritance_sidecar_payload_sha256_v1(payload),
    }
    return result


def _context_member(context: Any, name: str) -> Any:
    try:
        return getattr(context, name)
    except AttributeError as exc:
        raise _fail(f"settlement context is missing {name}") from exc


def _exact_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(f"{name} must be an exact integer >= {minimum}")
    return value


def _unbatch(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise _fail(f"settlement source is not a tensor: {name}")
    if value.ndim == 0 or value.size(0) != BATCH_CELLS:
        raise _fail(f"settlement source must have an exact b1 prefix: {name}")
    return value[0]


def _tensor_scalar_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if not isinstance(value, torch.Tensor) or value.numel() != 1:
        raise _fail(f"{name} must be a one-element tensor")
    scalar = value.detach().cpu().item()
    return _exact_int(scalar, name=name, minimum=minimum)


def _capability_supports_patch_geometry(
    qualification: Mapping[str, Any],
    *,
    population: int,
    grid_size: int,
) -> bool:
    if (population, grid_size) not in QUALIFIED_PATCH_GEOMETRIES:
        return False
    supported = qualification.get("supported_geometry_envelopes")
    if isinstance(supported, Sequence) and not isinstance(supported, (str, bytes)):
        for item in supported:
            if not isinstance(item, Mapping):
                continue
            if (
                item.get("population") == population
                and item.get("grid_size") == grid_size
            ):
                return True
    return (
        qualification.get("population") == population
        and qualification.get("grid_size") == grid_size
    )


def _validate_capability(capability: Mapping[str, Any]) -> None:
    if capability.get("generation_summary_scalar_layout_version") != (
        GENERATION_SUMMARY_LAYOUT_VERSION
    ):
        raise _fail("generation-summary scalar layout version is not qualified")
    columns = capability.get("generation_summary_scalar_columns")
    if (
        not isinstance(columns, Sequence)
        or isinstance(columns, (str, bytes, bytearray))
        or tuple(columns) != GENERATION_SUMMARY_COLUMNS
        or capability.get("generation_summary_scalar_layout_width")
        != len(GENERATION_SUMMARY_COLUMNS)
    ):
        raise _fail("generation-summary scalar columns do not match the exact ABI")

    qualification = _mapping(
        capability.get("patch_p2_qualification"),
        name="capability.patch_p2_qualification",
    )
    expected = {
        "qualified": True,
        "profile": "P2",
        "world": "patch",
        "policy": "stochastic_fixed_mlp",
        "batch_cells": BATCH_CELLS,
        "ticks_per_generation": TICKS_PER_GENERATION,
        "generation_turnover": True,
    }
    mismatches = {
        name: {"expected": expected_value, "actual": qualification.get(name)}
        for name, expected_value in expected.items()
        if type(qualification.get(name)) is not type(expected_value)
        or qualification.get(name) != expected_value
    }
    if mismatches:
        raise _fail(f"Patch P2 qualification mismatched: {mismatches}")
    for population, grid_size in QUALIFIED_PATCH_GEOMETRIES:
        if not _capability_supports_patch_geometry(
            qualification,
            population=population,
            grid_size=grid_size,
        ):
            raise _fail("Patch P2 qualification omitted an admitted geometry")
    if capability.get("canonical_final_checkpoint_projection_supported") is not True:
        raise _fail("capability omitted canonical terminal checkpoint support")


def _validate_direct_result(
    result: Mapping[str, Any],
    *,
    initial_generation: int = 0,
    physical_window_generations: int = PHYSICAL_WINDOW_GENERATIONS,
    completed_generation_count: int = LOGICAL_GENERATION_COUNT,
    logical_generation_count: int = LOGICAL_GENERATION_COUNT,
) -> None:
    if (
        type(logical_generation_count) is not int
        or logical_generation_count < 1
        or logical_generation_count > 128
    ):
        raise _fail("logical generation horizon must be an exact integer in [1,128]")
    if not 1 <= completed_generation_count <= logical_generation_count:
        raise _fail("completed generation count exceeds the admitted logical horizon")
    if (
        type(physical_window_generations) is not int
        or physical_window_generations not in {1, 2}
    ):
        raise _fail("physical window generation count must be exact integer 1 or 2")
    expected = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "owner": "native_extension",
        "physical_window_generations": physical_window_generations,
        "logical_generation_count": logical_generation_count,
        "window_count": resident_window_count_v1(
            initial_generation=initial_generation,
            logical_generation_count=logical_generation_count,
            physical_window_generations=physical_window_generations,
        ),
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "completed_generations": completed_generation_count,
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    for name, expected_value in expected.items():
        actual = result.get(name)
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise _fail(
                f"direct result {name} mismatch: "
                f"expected={expected_value!r} actual={actual!r}"
            )
    expected_settled_windows = resident_window_count_v1(
        initial_generation=initial_generation,
        logical_generation_count=completed_generation_count,
        physical_window_generations=physical_window_generations,
    )
    if result.get("settled_window_count", expected_settled_windows) != (
        expected_settled_windows
    ):
        raise _fail("direct result settled_window_count mismatched the complete prefix")


def _direct_result_physical_window_generations_v1(
    result: Mapping[str, Any],
    source: Any,
) -> int:
    source_value = getattr(source, "physical_window_generations", None)
    if source_value is not None:
        return int(source_value)
    result_value = result.get("physical_window_generations")
    if type(result_value) is int and result_value in {1, 2}:
        return result_value
    raise _fail("direct result omitted admitted physical_window_generations")


def _with_materialized_event_history(
    persistent_state: Mapping[str, Any],
    result: Mapping[str, Any],
) -> Mapping[str, Any]:
    if result.get("event_history_storage_mode") == EVENT_SINK_SCHEMA_VERSION:
        out = dict(persistent_state)
        reference = result.get("event_history_reference")
        try:
            validated_reference = validate_event_sink_prefix_reference_v1(reference)
        except EventSinkError as exc:
            raise _fail(f"external event-history reference is invalid: {exc}") from exc
        out["event_history_storage_mode"] = EVENT_SINK_SCHEMA_VERSION
        out["event_history_reference"] = validated_reference
        frontier = result.get("event_history_frontier")
        if frontier != validated_reference["row_count"]:
            raise _fail("external event-history frontier contradicts its prefix identity")
        out["event_history_frontier"] = frontier
        sequence_base = result.get("event_sequence_base")
        if sequence_base is not None:
            out["event_sequence_base"] = sequence_base
        return out
    histories = result.get("event_histories")
    if histories is None:
        return persistent_state
    if (
        not isinstance(histories, Sequence)
        or isinstance(histories, (str, bytes, bytearray))
        or len(histories) != BATCH_CELLS
    ):
        raise _fail("direct result event_histories must contain one cell")
    history = histories[0]
    if (
        not isinstance(history, Sequence)
        or isinstance(history, (str, bytes, bytearray))
        or len(history) != len(EVENT_COLUMN_NAMES)
    ):
        raise _fail("direct result event history does not match the canonical schema")
    columns: list[torch.Tensor] = []
    row_count: int | None = None
    for name, value in zip(EVENT_COLUMN_NAMES, history, strict=True):
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise _fail(f"materialized event history column is invalid: {name}")
        if row_count is None:
            row_count = int(value.numel())
        elif int(value.numel()) != row_count:
            raise _fail("materialized event history column lengths disagree")
        columns.append(value.contiguous().unsqueeze(0))
    if row_count is None:
        raise _fail("materialized event history is empty")
    reported_rows = result.get("event_rows_by_cell")
    if (
        not isinstance(reported_rows, Sequence)
        or isinstance(reported_rows, (str, bytes, bytearray))
        or list(reported_rows) != [row_count]
    ):
        raise _fail("direct result event row count contradicts its history")
    sequence_index = EVENT_COLUMN_NAMES.index("typed_lifecycle_event_sequence")
    sequence = columns[sequence_index][0]
    expected_sequence = torch.arange(
        row_count,
        dtype=sequence.dtype,
        device=sequence.device,
    )
    if not torch.equal(sequence, expected_sequence):
        raise _fail("materialized event history sequence is not zero-based contiguous")
    out = dict(persistent_state)
    out["event_columns"] = columns
    out["event_active_count"] = torch.tensor(
        [row_count], dtype=torch.int64, device=sequence.device
    )
    out["event_status"] = torch.zeros(
        (BATCH_CELLS,), dtype=torch.int32, device=sequence.device
    )
    out["event_sequence_base"] = torch.zeros(
        (BATCH_CELLS,), dtype=torch.int64, device=sequence.device
    )
    out.update(
        {name: column for name, column in zip(EVENT_COLUMN_NAMES, columns, strict=True)}
    )
    return out


def _merge_drained_event_history_v1(
    canonical_projection: Mapping[str, Any],
    persistent_state: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reattach the CPU checkpoint prefix after a drained resident import.

    The direct controller deliberately starts a resumed segment with an empty
    physical event bank.  Its materialized history therefore contains only
    rows emitted by that segment and is locally zero-based.  Canonical
    checkpoint history is immutable here: concatenate the new rows after it
    and republish one absolute zero-based sequence before checkpoint export.
    """

    if persistent_state.get("event_history_storage_mode") == (
        "eiel_resident_event_sink_v1"
    ):
        return persistent_state
    if persistent_state.get("event_history_drained") is not True:
        return persistent_state
    canonical_tensors = canonical_projection.get("tensors")
    if not isinstance(canonical_tensors, Mapping):
        raise _fail("drained event settlement requires canonical tensors")
    canonical_columns: list[torch.Tensor] = []
    prefix_rows: int | None = None
    for name in EVENT_COLUMN_NAMES:
        value = canonical_tensors.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise _fail(
                f"canonical drained event prefix is invalid: {name}"
            )
        if prefix_rows is None:
            prefix_rows = int(value.numel())
        elif int(value.numel()) != prefix_rows:
            raise _fail("canonical drained event prefix columns disagree")
        canonical_columns.append(value.detach().cpu().contiguous())
    if prefix_rows is None:
        raise _fail("canonical drained event prefix is empty or missing")
    prefix_sequence_index = EVENT_COLUMN_NAMES.index(
        "typed_lifecycle_event_sequence"
    )
    if not torch.equal(
        canonical_columns[prefix_sequence_index],
        torch.arange(prefix_rows, dtype=torch.int64),
    ):
        raise _fail("canonical drained event prefix sequence is not exact")

    raw_columns = persistent_state.get("event_columns")
    if (
        not isinstance(raw_columns, (list, tuple))
        or len(raw_columns) != len(EVENT_COLUMN_NAMES)
    ):
        raise _fail("drained resident event columns are incomplete")
    local_count = _tensor_scalar_int(
        persistent_state.get("event_active_count"),
        name="drained resident event_active_count",
    )
    local_columns: list[torch.Tensor] = []
    for index, (name, value) in enumerate(
        zip(EVENT_COLUMN_NAMES, raw_columns, strict=True)
    ):
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or int(value.size(0)) != BATCH_CELLS
            or int(value.size(1)) < local_count
        ):
            raise _fail(f"drained resident event column is invalid: {name}")
        local_columns.append(
            value[0, :local_count].detach().cpu().contiguous()
        )
    local_sequence = local_columns[prefix_sequence_index]
    if not torch.equal(
        local_sequence,
        torch.arange(local_count, dtype=torch.int64),
    ):
        raise _fail("drained resident event sequence is not zero-based contiguous")

    total_rows = prefix_rows + local_count
    merged_columns: list[torch.Tensor] = []
    for index, (prefix, local) in enumerate(
        zip(canonical_columns, local_columns, strict=True)
    ):
        if index == prefix_sequence_index:
            merged = torch.arange(total_rows, dtype=torch.int64)
        elif local_count:
            merged = torch.cat((prefix, local), dim=0).contiguous()
        else:
            merged = prefix.clone()
        merged_columns.append(merged)

    out = dict(persistent_state)
    out["event_columns"] = [
        value.unsqueeze(0).contiguous() for value in merged_columns
    ]
    out["event_active_count"] = torch.tensor(
        [total_rows], dtype=torch.int64
    )
    out["event_status"] = torch.zeros(
        (BATCH_CELLS,), dtype=torch.int32
    )
    out["event_sequence_base"] = torch.zeros(
        (BATCH_CELLS,), dtype=torch.int64
    )
    out.update(
        {
            name: value.unsqueeze(0).contiguous()
            for name, value in zip(
                EVENT_COLUMN_NAMES, merged_columns, strict=True
            )
        }
    )
    return out


def _exact_summary_integer(
    row: torch.Tensor,
    *,
    index: Mapping[str, int],
    name: str,
) -> int:
    value = float(row[index[name]].item())
    if not math.isfinite(value):
        raise _fail(f"generation-summary scalar {name} must be finite")
    integer = int(value)
    if value != float(integer):
        raise _fail(f"generation-summary scalar {name} is not an exact integer")
    return integer


def _finite_summary_float(
    row: torch.Tensor,
    *,
    index: Mapping[str, int],
    name: str,
) -> float:
    value = float(row[index[name]].item())
    if not math.isfinite(value):
        raise _fail(f"generation-summary scalar {name} must be finite")
    return value


def _decode_compact_summary_rows(
    outputs: Mapping[str, Any],
    capability: Mapping[str, Any],
    *,
    completed_generation_count: int = LOGICAL_GENERATION_COUNT,
    source_generation: int = 0,
) -> tuple[dict[str, Any], ...]:
    scalar = outputs.get("generation_summary_scalar")
    expected_shape = (
        completed_generation_count,
        BATCH_CELLS,
        len(GENERATION_SUMMARY_COLUMNS),
    )
    if (
        not isinstance(scalar, torch.Tensor)
        or scalar.dtype != torch.float64
        or scalar.device.type != "cpu"
        or not scalar.is_contiguous()
        or tuple(scalar.shape) != expected_shape
    ):
        raise _fail(
            "generation_summary_scalar must be contiguous CPU "
            f"float64{list(expected_shape)}"
        )
    columns = tuple(capability["generation_summary_scalar_columns"])
    index = {name: offset for offset, name in enumerate(columns)}
    rows: list[dict[str, Any]] = []
    for generation_offset in range(completed_generation_count):
        source = scalar[generation_offset, 0]
        decoded_generation = _exact_summary_integer(
            source,
            index=index,
            name="generation",
        )
        if decoded_generation != source_generation + generation_offset:
            raise _fail(
                "generation-summary scalar rows are not an exact complete prefix"
            )
        rows.append(
            {
                "schema_version": COMPACT_SUMMARY_SCHEMA_VERSION,
                "generation": decoded_generation,
                "world": "patch",
                "alive": _exact_summary_integer(
                    source, index=index, name="alive_after_turnover"
                ),
                "births": _exact_summary_integer(source, index=index, name="births"),
                "failed_births": _exact_summary_integer(
                    source, index=index, name="failed_births"
                ),
                "deaths": _exact_summary_integer(source, index=index, name="deaths"),
                "event_cursor": _exact_summary_integer(
                    source, index=index, name="event_cursor_end"
                ),
                "moved": _exact_summary_integer(
                    source, index=index, name="moved_total"
                ),
                "gathered": _exact_summary_integer(
                    source, index=index, name="gathered_total"
                ),
                "patch_food_stock_after_turnover": _finite_summary_float(
                    source, index=index, name="patch_food_stock_after_turnover"
                ),
                "energy_collected_exact_sum_after_turnover": _finite_summary_float(
                    source,
                    index=index,
                    name="energy_collected_exact_sum_after_turnover",
                ),
                "policy_energy_live_sum_after_turnover": _finite_summary_float(
                    source,
                    index=index,
                    name="policy_energy_live_sum_after_turnover",
                ),
            }
        )
    return tuple(rows)


def _source_config_value(source: Any, name: str, default: Any) -> Any:
    config = getattr(source, "configuration", None)
    if isinstance(config, Mapping) and name in config:
        return config[name]
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return getattr(source, name, default)


def _source_config_sha256_v1(source: Any) -> str:
    config = getattr(source, "configuration", None)
    if config is None:
        return _canonical_sha256({"configuration": "unavailable"})
    try:
        if dataclasses.is_dataclass(config):
            return _canonical_sha256(dataclasses.asdict(config))
    except TypeError:
        pass
    payload = {
        name: getattr(config, name)
        for name in dir(config)
        if (
            not name.startswith("_")
            and type(getattr(config, name)) in {bool, int, float, str, tuple}
        )
    }
    return _canonical_sha256(payload)


def _profile_sha256_v1(source: Any, persistent_state: Mapping[str, Any]) -> str:
    config_value = _source_config_value(
        source,
        "genome_upgrade_v2_founder_profile_sha256",
        "",
    )
    if isinstance(config_value, str) and len(config_value) == 64:
        return config_value
    profile = persistent_state.get("genome_v2_founder_profile")
    if isinstance(profile, Mapping):
        payload_sha256 = profile.get("payload_sha256")
        if isinstance(payload_sha256, str) and len(payload_sha256) == 64:
            return payload_sha256
    return "0" * 64


def _route_lifecycle_control_sha256_v1(
    source: Any,
    *,
    generation_turnover_enabled: bool,
    turnover_counts: torch.Tensor,
) -> str:
    return _canonical_sha256(
        {
            "schema_version": "eiel_gv2_lifecycle_v2_route_control_binding_v1",
            "lifecycle_mode_abi": int(
                _source_config_value(
                    source,
                    "lifecycle_mode",
                    0 if generation_turnover_enabled else 1,
                )
            ),
            "generation_turnover_enabled": generation_turnover_enabled,
            "turnover_count_width": int(turnover_counts.numel()),
        }
    )


def _resolved_lifecycle_mode_abi_v1(
    source: Any,
    *,
    generation_turnover_enabled: bool,
) -> int:
    expected = 0 if generation_turnover_enabled else 1
    raw = _source_config_value(source, "lifecycle_mode", expected)
    if isinstance(raw, bool) or raw not in {0, 1}:
        raise _fail("lifecycle_mode must be ABI value 0 or 1")
    mode = int(raw)
    if mode != expected:
        raise _fail(
            "lifecycle_mode contradicts generation_turnover_enabled: "
            f"lifecycle_mode={mode} generation_turnover_enabled="
            f"{generation_turnover_enabled}"
        )
    return mode


def _resolved_generation_turnover_enabled_v1(
    source: Any,
    *,
    default: bool,
) -> bool:
    raw = _source_config_value(source, "generation_turnover_enabled", default)
    if type(raw) is not bool:
        raise _fail("generation_turnover_enabled must be an exact boolean")
    return raw


def _generation_turnover_default_from_lifecycle_mode_v1(source: Any) -> bool:
    raw = _source_config_value(source, "lifecycle_mode", 0)
    if isinstance(raw, bool) or raw not in {0, 1}:
        return True
    return int(raw) == 0


def _population_authority_payload_v1(
    *,
    generation_turnover_enabled: bool,
    legacy_turnover_final_live_value: int,
) -> dict[str, Any]:
    if generation_turnover_enabled:
        return {
            "population_authority_source": (
                "legacy_turnover_final_live_plus_resident_identity"
            ),
            "legacy_turnover_final_live_authority": (
                "authoritative_turnover_counter"
            ),
            "legacy_turnover_final_live_value": legacy_turnover_final_live_value,
        }
    return {
        "population_authority_source": "resident_identity_plus_qacct",
        "legacy_turnover_final_live_authority": "inactive_zero_sentinel",
        "legacy_turnover_final_live_value": 0,
    }


def _validate_checkpoint_population_authority_v1(
    metadata: Mapping[str, Any],
    *,
    generation_turnover_enabled: bool,
    lifecycle_mode_abi: int,
    resident_live_end: int,
    legacy_turnover_final_live_value: int,
) -> None:
    if metadata.get("generation_turnover_enabled") is not generation_turnover_enabled:
        raise _fail(
            "terminal checkpoint generation_turnover_enabled contradicts "
            "resolved lifecycle route"
        )
    if metadata.get("lifecycle_mode_abi") != lifecycle_mode_abi:
        raise _fail(
            "terminal checkpoint lifecycle_mode_abi contradicts resolved lifecycle route"
        )
    expected = _population_authority_payload_v1(
        generation_turnover_enabled=generation_turnover_enabled,
        legacy_turnover_final_live_value=legacy_turnover_final_live_value,
    )
    if (
        metadata.get("lifecycle_population_authority_source")
        != expected["population_authority_source"]
    ):
        raise _fail(
            "terminal checkpoint lifecycle_population_authority_source "
            "contradicts lifecycle authority"
        )
    for key in (
        "legacy_turnover_final_live_authority",
        "legacy_turnover_final_live_value",
    ):
        value = expected[key]
        if metadata.get(key) != value:
            raise _fail(f"terminal checkpoint {key} contradicts lifecycle authority")
    if (
        _exact_int(
            metadata.get("alive_count"),
            name="terminal checkpoint metadata.alive_count",
        )
        != resident_live_end
    ):
        raise _fail("terminal checkpoint alive_count contradicts QACCT live_end")


def _projection_live_count_v1(
    projection: Mapping[str, Any],
    *,
    fallback_metadata_field: str,
) -> int:
    tensors = projection.get("tensors")
    if isinstance(tensors, Mapping):
        alive_mask = tensors.get("alive_mask")
        if isinstance(alive_mask, torch.Tensor):
            if alive_mask.dtype != torch.bool:
                raise _fail("checkpoint alive_mask must be boolean")
            return int(alive_mask.detach().to("cpu").to(torch.int64).sum().item())
    metadata = _mapping(
        projection.get("metadata"),
        name="checkpoint metadata",
    )
    return _exact_int(
        metadata.get(fallback_metadata_field),
        name=f"checkpoint metadata {fallback_metadata_field}",
    )


def _projection_capacity_v1(projection: Mapping[str, Any], *, name: str) -> int:
    tensors = projection.get("tensors")
    if isinstance(tensors, Mapping):
        alive_mask = tensors.get("alive_mask")
        if isinstance(alive_mask, torch.Tensor):
            if alive_mask.dtype != torch.bool or alive_mask.ndim != 1:
                raise _fail(f"{name} alive_mask must be a one-dimensional boolean tensor")
            return int(alive_mask.numel())
    metadata = _mapping(projection.get("metadata"), name=f"{name} metadata")
    return _exact_int(metadata.get("capacity"), name=f"{name} metadata.capacity")


def _projection_pending_free_count_v1(
    projection: Mapping[str, Any],
    *,
    fallback_free_slots: int,
    name: str,
) -> int:
    tensors = projection.get("tensors")
    if isinstance(tensors, Mapping):
        value = tensors.get("pending_free_count")
        if isinstance(value, torch.Tensor):
            return _tensor_scalar_int(value, name=f"{name}.pending_free_count")
    metadata = projection.get("metadata")
    if isinstance(metadata, Mapping) and "pending_free_count" in metadata:
        return _exact_int(
            metadata.get("pending_free_count"),
            name=f"{name} metadata.pending_free_count",
        )
    return fallback_free_slots


def _birth_reason_totals_v1(
    outputs: Mapping[str, Any],
    persistent_state: Mapping[str, Any],
    *,
    ordinary_admitted_births: int,
    completed_generation_count: int,
    ticks_per_generation: int = TICKS_PER_GENERATION,
    compact_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], int, str, bool]:
    reason_counts = _unbatch(outputs.get("birth_reason_counts"), name="birth_reason_counts")
    if (
        reason_counts.dtype != torch.int64
        or reason_counts.ndim != 2
        or int(reason_counts.size(1)) != len(BIRTH_REASON_NAMES_V1)
    ):
        raise _fail("birth_reason_counts must be route-bound int64[ticks,10]")
    required_ticks = completed_generation_count * ticks_per_generation
    if int(reason_counts.size(0)) >= required_ticks:
        totals_tensor = (
            reason_counts[:required_ticks]
            .detach()
            .to("cpu")
            .reshape(required_ticks, len(BIRTH_REASON_NAMES_V1))
            .sum(dim=0)
        )
        totals: dict[str, int] = {}
        for index, reason in enumerate(BIRTH_REASON_NAMES_V1):
            value = int(totals_tensor[index].item())
            if value < 0:
                raise _fail(f"birth reason count is negative: {reason}")
            totals[reason] = value
        return totals, required_ticks, "birth_reason_counts_full_prefix", True

    return _typed_birth_reason_totals_v1(
        persistent_state,
        ordinary_admitted_births=ordinary_admitted_births,
        compact_rows=compact_rows,
    )


def _typed_birth_reason_totals_v1(
    persistent_state: Mapping[str, Any],
    *,
    ordinary_admitted_births: int,
    compact_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], int, str, bool]:
    event_type = persistent_state.get("typed_lifecycle_event_event_type")
    reason_code = persistent_state.get("typed_lifecycle_event_reason_code")
    generation = persistent_state.get("typed_lifecycle_event_generation")
    success = persistent_state.get("typed_lifecycle_event_success")
    active_count = persistent_state.get("event_active_count")
    if not (
        isinstance(event_type, torch.Tensor)
        and isinstance(reason_code, torch.Tensor)
        and event_type.ndim == 2
        and reason_code.ndim == 2
        and int(event_type.size(0)) == 1
        and int(reason_code.size(0)) == 1
        and int(event_type.size(1)) == int(reason_code.size(1))
    ):
        raise _fail(
            "birth_reason_counts does not cover the completed generation prefix "
            "and typed event reason history is unavailable"
        )
    if generation is not None and not (
        isinstance(generation, torch.Tensor)
        and generation.ndim == 2
        and int(generation.size(0)) == 1
        and int(generation.size(1)) == int(event_type.size(1))
    ):
        raise _fail("typed event generation history is malformed")
    if success is not None and not (
        isinstance(success, torch.Tensor)
        and success.ndim == 2
        and int(success.size(0)) == 1
        and int(success.size(1)) == int(event_type.size(1))
    ):
        raise _fail("typed event success history is malformed")
    rows = _tensor_scalar_int(active_count, name="event_active_count")
    if rows > int(event_type.size(1)):
        raise _fail("event_active_count exceeds typed event reason history")
    totals = {reason: 0 for reason in BIRTH_REASON_NAMES_V1}
    types = event_type[0, :rows].detach().to("cpu")
    reasons = reason_code[0, :rows].detach().to("cpu")
    generation_filtered = False
    if generation is None:
        generation_mask = torch.ones(rows, dtype=torch.bool)
    else:
        compact_generations = {
            int(row["generation"])
            for row in compact_rows
            if type(row.get("generation")) is int
        }
        if len(compact_generations) != len(compact_rows):
            raise _fail("compact rows do not expose exact generations")
        generations = generation[0, :rows].detach().to("cpu")
        generation_mask = torch.zeros(rows, dtype=torch.bool)
        for compact_generation in compact_generations:
            generation_mask |= generations == compact_generation
        generation_filtered = True
    if success is None:
        totals["accepted"] = ordinary_admitted_births
    else:
        successes = success[0, :rows].detach().to("cpu")
        accepted_count = int(
            (
                (types == BIRTH_ACCEPTED_EVENT_TYPE_V1)
                & (successes == 1)
                & generation_mask
            )
            .sum()
            .item()
        )
        if accepted_count != ordinary_admitted_births:
            raise _fail("typed accepted birth count contradicts admitted births")
        totals["accepted"] = accepted_count
    failed_mask = (types == BIRTH_FAILED_EVENT_TYPE_V1) & generation_mask
    for raw_reason in reasons[failed_mask].tolist():
        if type(raw_reason) is not int or not 1 <= raw_reason < len(
            BIRTH_REASON_NAMES_V1
        ):
            raise _fail(f"typed failed-birth reason code is unsupported: {raw_reason}")
        totals[BIRTH_REASON_NAMES_V1[int(raw_reason)]] += 1
    accounted_count = int(generation_mask.sum().item()) if generation_filtered else rows
    source = (
        "typed_lifecycle_event_reason_history_generation_filtered"
        if generation_filtered
        else "typed_lifecycle_event_reason_history"
    )
    return totals, accounted_count, source, True


def _legacy_turnover_culls_from_compact_rows_v1(
    compact_rows: Sequence[Mapping[str, Any]],
    *,
    live_start: int,
    live_end: int,
    compact_deaths_may_include_culls: bool = False,
) -> int:
    """Derive prefix turnover removals from compact generation balances."""

    previous_live = live_start
    legacy_culls = 0
    for row in compact_rows:
        generation = row.get("generation")
        alive = _exact_int(row.get("alive"), name=f"G{generation} alive")
        births = _exact_int(row.get("births", 0), name=f"G{generation} births")
        deaths = _exact_int(row.get("deaths", 0), name=f"G{generation} deaths")
        terminal_removals = previous_live + births - alive
        if terminal_removals < 0:
            raise _fail(
                "compact generation population balance implies negative "
                "terminal removals"
            )
        if terminal_removals < deaths:
            if not compact_deaths_may_include_culls:
                raise _fail(
                    "compact generation deaths exceed population-balance "
                    "terminal removals"
                )
            previous_live = alive
            continue
        legacy_culls += terminal_removals - deaths
        previous_live = alive
    if previous_live != live_end:
        raise _fail("compact generation terminal live count differs from resident identity")
    return legacy_culls


def _capacity_accounting_payload_v1(
    source: Any,
    persistent_state: Mapping[str, Any],
    outputs: Mapping[str, Any],
    compact_rows: Sequence[Mapping[str, Any]],
    checkpoint: CudaBoundaryCheckpoint,
    canonical_projection: Mapping[str, Any],
    *,
    live_start: int,
    live_end: int,
    ecological_deaths: int,
    legacy_culls: int,
    ordinary_admitted_births: int | None = None,
    admitted_births: int,
    ticks_per_generation: int = TICKS_PER_GENERATION,
    terminal_removals: int,
) -> Mapping[str, Any]:
    completed_generation_count = len(compact_rows)
    if ordinary_admitted_births is None:
        ordinary_admitted_births = admitted_births
    failed_births_total = sum(
        int(row.get("failed_births", 0) or 0) for row in compact_rows
    )
    (
        reason_totals,
        accounted_count,
        reason_accounting_source,
        reason_qualification_authority,
    ) = _birth_reason_totals_v1(
        outputs,
        persistent_state,
        ordinary_admitted_births=ordinary_admitted_births,
        completed_generation_count=completed_generation_count,
        ticks_per_generation=ticks_per_generation,
        compact_rows=compact_rows,
    )
    reason_authority_kind = CAPACITY_REASON_AUTHORITY_FULL_PREFIX_V1
    accepted_birth_decisions = reason_totals["accepted"]
    failed_birth_reasons = {
        reason: count
        for reason, count in reason_totals.items()
        if reason != "accepted"
    }
    failed_birth_reason_sum = sum(failed_birth_reasons.values())
    terminal_capacity = _projection_capacity_v1(
        checkpoint.projection,
        name="terminal checkpoint projection",
    )
    source_capacity = _projection_capacity_v1(
        canonical_projection,
        name="source checkpoint projection",
    )
    physical_resident_capacity = max(source_capacity, terminal_capacity)
    initial_population_configured = _exact_int(
        _source_config_value(source, "population_size", live_start),
        name="configured initial population",
    )
    hard_cap_configured = _exact_int(
        _source_config_value(
            source,
            "resident_capacity",
            _source_config_value(source, "hard_cap", physical_resident_capacity),
        ),
        name="configured hard cap",
    )
    free_slots_start = physical_resident_capacity - live_start
    free_slots_end = physical_resident_capacity - live_end
    if free_slots_start < 0 or free_slots_end < 0:
        raise _fail("resident live count exceeds physical resident capacity")
    if failed_birth_reason_sum != failed_births_total:
        if reason_accounting_source != (
            "typed_lifecycle_event_reason_history_generation_filtered"
        ):
            try:
                (
                    typed_reason_totals,
                    typed_accounted_count,
                    typed_reason_accounting_source,
                    typed_reason_qualification_authority,
                ) = _typed_birth_reason_totals_v1(
                    persistent_state,
                    ordinary_admitted_births=ordinary_admitted_births,
                    compact_rows=compact_rows,
                )
                typed_failed_reasons = {
                    reason: count
                    for reason, count in typed_reason_totals.items()
                    if reason != "accepted"
                }
                if sum(typed_failed_reasons.values()) == failed_births_total:
                    reason_totals = typed_reason_totals
                    accepted_birth_decisions = reason_totals["accepted"]
                    failed_birth_reasons = typed_failed_reasons
                    failed_birth_reason_sum = failed_births_total
                    accounted_count = typed_accounted_count
                    reason_accounting_source = typed_reason_accounting_source
                    reason_qualification_authority = (
                        typed_reason_qualification_authority
                    )
            except CudaArchitectureBDirectResidentSettlementError:
                pass
    if failed_birth_reason_sum != failed_births_total:
        saturated_no_space = (
            free_slots_start == 0
            and free_slots_end == 0
            and ordinary_admitted_births == 0
            and ecological_deaths == 0
            and legacy_culls == 0
            and terminal_removals == 0
            and live_start == physical_resident_capacity
            and live_end == physical_resident_capacity
        )
        if not saturated_no_space:
            raise _fail(
                "birth failed-reason counts do not close against compact failed_births"
            )
        failed_birth_reasons = {
            reason: 0 for reason in BIRTH_REASON_NAMES_V1 if reason != "accepted"
        }
        failed_birth_reasons["no_space"] = failed_births_total
        failed_birth_reason_sum = failed_births_total
        reason_qualification_authority = False
        reason_authority_kind = CAPACITY_REASON_AUTHORITY_DIAGNOSTIC_V1
        reason_accounting_source = (
            f"{reason_accounting_source}_plus_saturated_capacity_closed_form"
        )
    if accepted_birth_decisions > ordinary_admitted_births:
        raise _fail("accepted birth decisions contradict admitted births")
    pending_terminal_slots_start = _projection_pending_free_count_v1(
        canonical_projection,
        fallback_free_slots=free_slots_start,
        name="source checkpoint projection",
    )
    pending_terminal_slots_end = (
        _tensor_scalar_int(
            persistent_state.get("pending_free_count"),
            name="resident pending_free_count",
        )
        if "pending_free_count" in persistent_state
        else _projection_pending_free_count_v1(
            checkpoint.projection,
            fallback_free_slots=free_slots_end,
            name="terminal checkpoint projection",
        )
    )
    if pending_terminal_slots_start > physical_resident_capacity:
        raise _fail("pending free slots at start exceed resident capacity")
    if pending_terminal_slots_end > physical_resident_capacity:
        raise _fail("pending free slots at end exceed resident capacity")
    population_conservation_expected_live_end = (
        live_start - terminal_removals + admitted_births
    )
    return {
        "schema_version": CAPACITY_ACCOUNTING_SCHEMA_VERSION,
        "reason_code_schema": "persistent_birth_plan_reason_v1",
        "reason_code_labels": list(BIRTH_REASON_NAMES_V1),
        "reason_accounting_source": reason_accounting_source,
        "reason_authority_kind": reason_authority_kind,
        "reason_qualification_authority": reason_qualification_authority,
        "reason_denominator": BIRTH_REASON_DENOMINATOR_V1,
        "reason_accounted_count": accounted_count,
        "initial_population_configured": initial_population_configured,
        "physical_resident_capacity": physical_resident_capacity,
        "hard_cap_configured": hard_cap_configured,
        "live_start": live_start,
        "live_end": live_end,
        "free_slots_start": free_slots_start,
        "free_slots_end": free_slots_end,
        "pending_terminal_slots_start": pending_terminal_slots_start,
        "pending_terminal_slots_end": pending_terminal_slots_end,
        "birth_attempts_total": accepted_birth_decisions + failed_births_total,
        "admitted_births": admitted_births,
        "ordinary_births_admitted": ordinary_admitted_births,
        "ordinary_birth_decisions_accepted": accepted_birth_decisions,
        "failed_births_total": failed_births_total,
        "failed_birth_reason_sum": failed_birth_reason_sum,
        "failed_birth_reasons": failed_birth_reasons,
        "ecological_deaths": ecological_deaths,
        "senescence_deaths": 0,
        "max_age_deaths": 0,
        "legacy_turnover_culls": legacy_culls,
        "hard_cap_removals": 0,
        "integrity_admin_removals": 0,
        "terminal_removals": terminal_removals,
        "population_conservation_expected_live_end": (
            population_conservation_expected_live_end
        ),
    }


def _lifecycle_qacct_receipt_from_route_v1(
    source: Any,
    persistent_state: Mapping[str, Any],
    outputs: Mapping[str, Any],
    compact_rows: Sequence[Mapping[str, Any]],
    checkpoint: CudaBoundaryCheckpoint,
    canonical_projection: Mapping[str, Any],
    *,
    ticks_per_generation: int = TICKS_PER_GENERATION,
) -> Mapping[str, Any]:
    if not compact_rows:
        raise _fail("lifecycle QACCT requires compact generation rows")
    turnover_counts = _unbatch(outputs.get("turnover_counts"), name="turnover_counts")
    if turnover_counts.dtype != torch.int64 or turnover_counts.numel() != 8:
        raise _fail("lifecycle QACCT requires eight int64 turnover counters")
    turnover_counts_cpu = turnover_counts.detach().cpu().reshape(-1)
    generation_turnover_enabled = _resolved_generation_turnover_enabled_v1(
        source,
        default=bool(turnover_counts_cpu.ne(0).any().item()),
    )
    lifecycle_mode_abi = _resolved_lifecycle_mode_abi_v1(
        source,
        generation_turnover_enabled=generation_turnover_enabled,
    )
    legacy_turnover_final_live = int(turnover_counts_cpu[7].item())
    if not generation_turnover_enabled and bool(turnover_counts_cpu.ne(0).any().item()):
        raise _fail(
            "hybrid lifecycle QACCT observed nonzero turnover counters, "
            "including inactive final-live"
        )
    metadata = _mapping(
        checkpoint.projection.get("metadata"),
        name="terminal checkpoint metadata",
    )
    resident_live_end = _projection_live_count_v1(
        checkpoint.projection,
        fallback_metadata_field="alive_count",
    )
    compact_live_end = int(compact_rows[-1]["alive"])
    if resident_live_end != compact_live_end:
        raise _fail("lifecycle QACCT live_end contradicts resident identity")
    _validate_checkpoint_population_authority_v1(
        metadata,
        generation_turnover_enabled=generation_turnover_enabled,
        lifecycle_mode_abi=lifecycle_mode_abi,
        resident_live_end=resident_live_end,
        legacy_turnover_final_live_value=legacy_turnover_final_live,
    )
    live_start = _projection_live_count_v1(
        canonical_projection,
        fallback_metadata_field="population_size",
    )
    raw_ecological_deaths = sum(int(row.get("deaths", 0) or 0) for row in compact_rows)
    ecological_deaths = raw_ecological_deaths
    ordinary_admitted_births = sum(
        int(row.get("births", 0) or 0) for row in compact_rows
    )
    legacy_turnover_pre_live = (
        int(turnover_counts_cpu[0].item()) if generation_turnover_enabled else 0
    )
    if (
        generation_turnover_enabled
        and lifecycle_mode_abi == 0
        and ordinary_admitted_births == 0
        and legacy_turnover_pre_live > 0
    ):
        legacy_culls = int(turnover_counts_cpu[1].item())
    elif generation_turnover_enabled:
        legacy_culls = _legacy_turnover_culls_from_compact_rows_v1(
            compact_rows,
            live_start=live_start,
            live_end=resident_live_end,
            # HAPB host-authoritative turnover reports culls inside compact
            # deaths while ordinary births omit the refill replacement.
            compact_deaths_may_include_culls=ordinary_admitted_births > 0,
        )
    else:
        legacy_culls = 0
    legacy_turnover_ecological_deaths_reconciled = 0
    if (
        generation_turnover_enabled
        and lifecycle_mode_abi == 0
        and ordinary_admitted_births == 0
    ):
        if legacy_turnover_pre_live > 0:
            effective_ecological_deaths = max(
                0,
                live_start + ordinary_admitted_births - legacy_turnover_pre_live,
            )
            if effective_ecological_deaths > raw_ecological_deaths:
                raise _fail(
                    "legacy turnover pre-live implies more ecological deaths than "
                    "compact rows report: "
                    f"live_start={live_start} "
                    f"ordinary_births={ordinary_admitted_births} "
                    f"pre_live={legacy_turnover_pre_live} "
                    f"effective_ecological_deaths={effective_ecological_deaths} "
                    f"raw_ecological_deaths={raw_ecological_deaths} "
                    f"turnover_counts={tuple(int(v.item()) for v in turnover_counts_cpu)}"
                )
            ecological_deaths = effective_ecological_deaths
            legacy_turnover_ecological_deaths_reconciled = (
                raw_ecological_deaths - effective_ecological_deaths
            )
        else:
            legacy_turnover_ecological_deaths_reconciled = 0
    turnover_admitted_births = (
        int(turnover_counts_cpu[6].item()) if generation_turnover_enabled else 0
    )
    admitted_births = ordinary_admitted_births + turnover_admitted_births
    if not generation_turnover_enabled:
        ecological_deaths = live_start + admitted_births - resident_live_end
        if ecological_deaths < 0:
            raise _fail(
                "hybrid lifecycle population balance implies negative terminal removals"
            )
    elif ordinary_admitted_births > 0:
        # HAPB compact deaths can include host-authoritative culls. Residual
        # identity uses live_start + admitted_births - live_end as removers.
        balanced_removals = live_start + admitted_births - resident_live_end
        if balanced_removals < legacy_culls:
            raise _fail(
                "HAPB compact population balance implies negative ecological deaths"
            )
        ecological_deaths = balanced_removals - legacy_culls
    terminal_removals = ecological_deaths + legacy_culls
    capacity_accounting = _capacity_accounting_payload_v1(
        source,
        persistent_state,
        outputs,
        compact_rows,
        checkpoint,
        canonical_projection,
        live_start=live_start,
        live_end=resident_live_end,
        ecological_deaths=ecological_deaths,
        legacy_culls=legacy_culls,
        ordinary_admitted_births=ordinary_admitted_births,
        admitted_births=admitted_births,
        ticks_per_generation=ticks_per_generation,
        terminal_removals=terminal_removals,
    )
    if generation_turnover_enabled and lifecycle_mode_abi == 0:
        capacity_accounting = {
            **capacity_accounting,
            "raw_ecological_death_events": raw_ecological_deaths,
            "effective_ecological_deaths": ecological_deaths,
            "legacy_turnover_pre_live_count": legacy_turnover_pre_live,
            "legacy_turnover_refill_planned": int(turnover_counts_cpu[5].item()),
            "legacy_turnover_refill_committed": int(turnover_counts_cpu[6].item()),
            "legacy_turnover_ecological_deaths_reconciled": (
                legacy_turnover_ecological_deaths_reconciled
            ),
            "ecological_death_accounting_authority": (
                "legacy_turnover_window_net_live_reduction_with_raw_event_count_retained"
            ),
        }
    if not generation_turnover_enabled:
        capacity_accounting = {
            **capacity_accounting,
            "raw_ecological_death_events": raw_ecological_deaths,
            "effective_ecological_deaths": ecological_deaths,
            "ecological_death_accounting_authority": (
                "hybrid_lifecycle_population_balance_with_raw_event_count_retained"
            ),
        }
    accounting = build_lifecycle_population_accounting_v1(
        live_start=live_start,
        ecological_deaths=ecological_deaths,
        legacy_turnover_culls=legacy_culls,
        admitted_births=admitted_births,
        no_capacity_rejections=(
            capacity_accounting["failed_birth_reasons"]["no_space"]
            + capacity_accounting["failed_birth_reasons"]["no_free_slot"]
        ),
        dead_slots_released=terminal_removals,
        newborn_slot_reuses=min(admitted_births, terminal_removals),
        live_end=resident_live_end,
    )
    return lifecycle_qacct_receipt_v1(
        accounting=accounting,
        lifecycle_control_sha256=_route_lifecycle_control_sha256_v1(
            source,
            generation_turnover_enabled=generation_turnover_enabled,
            turnover_counts=turnover_counts_cpu,
        ),
        configuration_sha256=_source_config_sha256_v1(source),
        profile_sha256=_profile_sha256_v1(source, persistent_state),
        checkpoint_schema_version=str(
            metadata.get("schema_version", "checkpoint_projection_v1")
        ),
        capacity_accounting=capacity_accounting,
        generation_turnover_enabled=generation_turnover_enabled,
        lifecycle_mode_abi=lifecycle_mode_abi,
        **_population_authority_payload_v1(
            generation_turnover_enabled=generation_turnover_enabled,
            legacy_turnover_final_live_value=legacy_turnover_final_live,
        ),
    )


def _project_terminal_checkpoint(
    canonical_projection: Mapping[str, Any],
    birth_state: Any,
    outputs: Mapping[str, Any],
    persistent_state: Mapping[str, Any],
    *,
    completed_generation_count: int = LOGICAL_GENERATION_COUNT,
    ticks_per_generation: int = TICKS_PER_GENERATION,
    source_generation: int,
    source_tick: int = 0,
    completed_tick_count: int | None = None,
    terminal_tick: int | None = None,
    generation_turnover_enabled: bool = True,
    inheritance_sidecar_cell_id: str | None = None,
    genome_v2_checkpoint_authority_binding: Any = (),
    genome_v2_causal_physical_cell_ordinal: int = 0,
    terminal_metadata_updates: Mapping[str, Any] | None = None,
    phase_e_active_modular_authority: PackedModularStorageV1 | None = None,
    phase_e_active_modular_lifecycle_parent_slots: Mapping[int, int] | None = None,
    phase_e_active_modular_lifecycle_mutation_receipt: (
        Mapping[str, object]
        | ModularRecurrentMutationReceiptV1
        | ModularRecurrentMutationSequenceV1
        | None
    ) = None,
    pre_materialized_cpu_fields: frozenset[str] = frozenset(),
) -> CudaBoundaryCheckpoint:
    metadata = _mapping(
        canonical_projection.get("metadata"),
        name="pre_generation_canonical_projection.metadata",
    )
    canonical_tensors = _mapping(
        canonical_projection.get("tensors"),
        name="pre_generation_canonical_projection.tensors",
    )
    canonical_alive_mask = canonical_tensors.get("alive_mask")
    if not isinstance(canonical_alive_mask, torch.Tensor):
        raise _fail("initial canonical projection omitted 'alive_mask'")
    canonical_capacity = int(canonical_alive_mask.numel())
    initial_live_count = _exact_int(
        metadata.get("population_size"),
        name="pre_generation_canonical_projection.metadata.population_size",
    )
    canonical_generation = _exact_int(
        metadata.get("generation"),
        name="pre_generation_canonical_projection.metadata.generation",
    )
    if (
        metadata.get("world") != "patch"
        or canonical_generation != source_generation
        or metadata.get("tick") != source_tick
        or metadata.get("agent_grid_runner_phase")
        != ("generation_boundary" if source_tick == 0 else "partial_generation")
        or initial_live_count
        not in {population for population, _grid in QUALIFIED_PATCH_GEOMETRIES}
        or metadata.get("capacity") != canonical_capacity
        or canonical_capacity < initial_live_count
    ):
        raise _fail(
            "terminal settlement requires the exact Patch P2 source cursor"
        )
    if terminal_tick is None:
        terminal_tick = 0
    partial_terminal = terminal_tick > 0
    terminal_generation = source_generation + (
        completed_generation_count if not partial_terminal else 0
    )

    if birth_state is None:
        raise _fail("checkpoint_projection_inputs omitted resident_birth_state")

    persistent_state = _merge_drained_event_history_v1(
        canonical_projection,
        persistent_state,
    )
    checkpoint_source_projection = canonical_projection
    if persistent_state.get("event_history_storage_mode") == EVENT_SINK_SCHEMA_VERSION:
        reference = persistent_state.get("event_history_reference")
        try:
            validated_reference = validate_event_sink_prefix_reference_v1(reference)
        except EventSinkError as exc:
            raise _fail(f"terminal event-history prefix is invalid: {exc}") from exc
        frontier = persistent_state.get("event_history_frontier")
        if frontier != validated_reference["row_count"]:
            raise _fail("terminal event-history frontier contradicts its identity")
        checkpoint_metadata = dict(metadata)
        checkpoint_metadata.update(
            {
                "event_history_storage_mode": EVENT_SINK_SCHEMA_VERSION,
                "event_history_reference": validated_reference,
                "event_history_frontier": frontier,
                "typed_lifecycle_event_sequence_base": frontier,
                "typed_lifecycle_event_next_sequence": frontier,
                "typed_lifecycle_event_active_count": 0,
                "event_tape_active_count": 0,
                "event_tape_cursor": 0,
            }
        )
        checkpoint_source_projection = dict(canonical_projection)
        checkpoint_source_projection["metadata"] = checkpoint_metadata

    def state_mapping(source: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in source:
            candidate_name = name
            if name == "pcg64_state_words" and source is birth_state.drive_state:
                candidate_name = "population_pcg64_state_words"
            elif name == "pcg64_buffer_words" and source is birth_state.drive_state:
                candidate_name = "population_pcg64_buffer_words"
            if candidate_name not in persistent_state:
                raise _fail(
                    "terminal resident checkpoint bank omitted "
                    f"{candidate_name!r} for {name!r}"
                )
            result[name] = _unbatch(
                persistent_state[candidate_name],
                name=candidate_name,
            )
        return result

    resident_boundary_tensors = {
        name: _unbatch(value, name=name)
        for name, value in persistent_state.items()
        if isinstance(value, torch.Tensor) and value.ndim > 0
    }
    drive_noise_projection_fields = (
        "drive_noise_samples_by_tick_slot",
        "drive_noise_samples_valid_by_tick_slot",
    )
    precomputed_drive_noise = metadata.get(
        "native_runtime_use_precomputed_drive_noise"
    )
    if type(precomputed_drive_noise) is not bool:
        raise _fail(
            "initial canonical projection omitted exact drive-noise mode metadata"
        )
    if precomputed_drive_noise:
        for name in drive_noise_projection_fields:
            canonical_value = canonical_tensors.get(name)
            resident_value = resident_boundary_tensors.get(name)
            if not isinstance(canonical_value, torch.Tensor):
                raise _fail(f"initial canonical projection omitted {name!r}")
            if not isinstance(resident_value, torch.Tensor):
                raise _fail(f"terminal resident checkpoint bank omitted {name!r}")
            resident_boundary_tensors[name] = canonical_value.to(
                device=resident_value.device,
                dtype=resident_value.dtype,
            )
    else:
        for name in drive_noise_projection_fields:
            resident_value = resident_boundary_tensors.pop(name, None)
            if not isinstance(resident_value, torch.Tensor):
                raise _fail(f"terminal resident ABI omitted {name!r}")
            if resident_value.numel() != 0:
                raise _fail(
                    f"RNG V2 terminal resident ABI materialized {name!r}"
                )

    try:
        resident_birth_state = dataclasses.replace(
            birth_state,
            identity_state=state_mapping(birth_state.identity_state),
            scalar_rng_state=state_mapping(birth_state.scalar_rng_state),
            genome_state=state_mapping(birth_state.genome_state),
            policy_mem_dim=_unbatch(
                persistent_state["policy_mem_dim"], name="policy_mem_dim"
            ),
            policy_use_gru=_unbatch(
                persistent_state["policy_use_gru"], name="policy_use_gru"
            ),
            drive_state=state_mapping(birth_state.drive_state),
            policy_state=state_mapping(birth_state.policy_state),
            turnover_policy_state=state_mapping(birth_state.turnover_policy_state),
            planner_pcg64_state_words=_unbatch(
                persistent_state["pcg64_state_words"], name="pcg64_state_words"
            ),
            planner_pcg64_buffer_words=_unbatch(
                persistent_state["pcg64_buffer_words"],
                name="pcg64_buffer_words",
            ),
            policy_np_semantic_draws=_unbatch(
                persistent_state["policy_np_semantic_draws"],
                name="policy_np_semantic_draws",
            ),
            runner_pcg64_state_words=_unbatch(
                persistent_state["runner_pcg64_state_words"],
                name="runner_pcg64_state_words",
            ),
            runner_pcg64_buffer_words=_unbatch(
                persistent_state["runner_pcg64_buffer_words"],
                name="runner_pcg64_buffer_words",
            ),
            metrics_np_semantic_draws=_unbatch(
                persistent_state["metrics_np_semantic_draws"],
                name="metrics_np_semantic_draws",
            ),
            event_tape_columns=tuple(
                _unbatch(column, name=f"event_column_{index}")
                for index, column in enumerate(persistent_state["event_columns"])
            ),
            event_tape_active_count=_unbatch(
                persistent_state["event_active_count"],
                name="event_active_count",
            ),
            event_tape_status=_unbatch(
                persistent_state["event_status"], name="event_status"
            ),
        )
    except (AttributeError, KeyError, TypeError) as exc:
        raise _fail(
            f"terminal resident birth-state banks are incomplete: {exc}"
        ) from exc

    live_mask = _unbatch(persistent_state.get("alive_mask"), name="alive_mask").bool()
    runtime_pred_lr = _unbatch(persistent_state.get("pred_lr"), name="pred_lr")
    genome_pred_lr = _unbatch(
        persistent_state.get("genome_pred_lr"), name="genome_pred_lr"
    )
    if not torch.equal(runtime_pred_lr[live_mask], genome_pred_lr[live_mask]):
        raise _fail(
            "terminal checkpoint cannot preserve distinct live runtime pred_lr "
            "and genome_pred_lr values"
        )

    bookkeeping = _unbatch(
        outputs.get("runtime_bookkeeping_by_tick"),
        name="runtime_bookkeeping_by_tick",
    )
    expected_ticks = (
        completed_generation_count * ticks_per_generation
        if completed_tick_count is None
        else completed_tick_count
    )
    if bookkeeping.dtype != torch.int64 or tuple(bookkeeping.shape) != (
        expected_ticks,
        5,
    ):
        raise _fail(
            "runtime_bookkeeping_by_tick must match the complete generation prefix"
        )
    aggregates = bookkeeping.sum(dim=0)
    active_count = _tensor_scalar_int(
        _unbatch(
            persistent_state.get("active_count_state"),
            name="active_count_state",
        ),
        name="active_count_state",
    )
    birth_status_by_tick = outputs.get("birth_combined_status")
    if birth_status_by_tick is None:
        lifecycle_status = outputs.get("lifecycle_status")
        if not isinstance(lifecycle_status, torch.Tensor):
            raise _fail("outputs omitted lifecycle_status")
        birth_status_by_tick = torch.zeros_like(lifecycle_status)

    if generation_turnover_enabled:
        turnover_next_generation = _unbatch(
            outputs.get("turnover_next_generation"),
            name="turnover_next_generation",
        )
        turnover_next_tick = _unbatch(
            outputs.get("turnover_next_tick"),
            name="turnover_next_tick",
        )
        turnover_boundary_generation = _tensor_scalar_int(
            turnover_next_generation,
            name="turnover_next_generation",
            minimum=1,
        )
        turnover_boundary_tick = _tensor_scalar_int(
            turnover_next_tick, name="turnover_next_tick"
        )
        if (
            turnover_boundary_generation != terminal_generation
            or turnover_boundary_tick != terminal_tick
        ):
            raise _fail(
                "direct resident trajectory did not end at its complete prefix boundary"
            )
    else:
        legacy_turnover_next_generation = outputs.get("turnover_next_generation")
        if legacy_turnover_next_generation is not None:
            _unbatch(
                legacy_turnover_next_generation,
                name="turnover_next_generation",
            )
        legacy_turnover_next_tick = outputs.get("turnover_next_tick")
        if legacy_turnover_next_tick is not None:
            _unbatch(
                legacy_turnover_next_tick,
                name="turnover_next_tick",
            )
        turnover_next_generation = torch.tensor(
            terminal_generation,
            dtype=torch.int64,
            device=bookkeeping.device,
        )
        turnover_next_tick = torch.tensor(
            terminal_tick, dtype=torch.int64, device=bookkeeping.device
        )

    raw_event_sequence_base = persistent_state.get("event_sequence_base")
    if isinstance(raw_event_sequence_base, torch.Tensor):
        event_sequence_base_result = _unbatch(
            raw_event_sequence_base,
            name="event_sequence_base",
        ).reshape(1)
    else:
        event_sequence_base_result = torch.zeros(
            (1,), dtype=torch.int64, device=bookkeeping.device
        )
    event_frontier_source = persistent_state.get("event_history_frontier")
    if isinstance(event_frontier_source, torch.Tensor):
        event_frontier_result = _unbatch(
            event_frontier_source,
            name="event_history_frontier",
        ).reshape(1)
    elif type(event_frontier_source) is int:
        event_frontier_result = torch.tensor(
            [event_frontier_source],
            dtype=torch.int64,
            device=event_sequence_base_result.device,
        )
    else:
        event_frontier_result = (
            event_sequence_base_result
            + _unbatch(
                persistent_state.get("event_active_count"),
                name="event_active_count",
            ).reshape(1)
        )
    generation_result = SimpleNamespace(
        lifecycle_status_by_tick=_unbatch(
            outputs.get("lifecycle_status"), name="lifecycle_status"
        ),
        birth_status_by_tick=_unbatch(
            birth_status_by_tick, name="birth_combined_status"
        ),
        event_tape_status=_unbatch(
            persistent_state.get("event_status"), name="event_status"
        ).reshape(1),
        event_append_status_by_tick=_unbatch(
            outputs.get("event_status_by_tick"), name="event_status_by_tick"
        ),
        turnover_status=_unbatch(
            outputs.get("turnover_status"), name="turnover_status"
        ).reshape(1),
        applied_actions=aggregates[0],
        moved=aggregates[1],
        gathered=aggregates[2],
        policy_refill_calls=aggregates[3],
        policy_refill_rows=aggregates[4],
        runtime_bookkeeping_by_tick=bookkeeping,
        turnover_next_generation=turnover_next_generation.reshape(1),
        turnover_next_tick=turnover_next_tick.reshape(1),
        final_alive_count=torch.tensor(
            [active_count], dtype=torch.int64, device=bookkeeping.device
        ),
        final_alive_agent_ids=_unbatch(
            persistent_state.get("active_agent_ids"), name="active_agent_ids"
        ),
        turnover_counts=_unbatch(
            outputs.get("turnover_counts"), name="turnover_counts"
        ),
        event_tape_active_count=_unbatch(
            persistent_state.get("event_active_count"),
            name="event_active_count",
        ).reshape(1),
        event_lifecycle_sequence_base=event_sequence_base_result,
        event_lifecycle_frontier=event_frontier_result,
        pcg64_state_words=_unbatch(
            persistent_state.get("pcg64_state_words"), name="pcg64_state_words"
        ),
        pcg64_buffer_words=_unbatch(
            persistent_state.get("pcg64_buffer_words"),
            name="pcg64_buffer_words",
        ),
        metrics_np_semantic_draws=_unbatch(
            persistent_state.get("metrics_np_semantic_draws"),
            name="metrics_np_semantic_draws",
        ).reshape(1),
    )
    checkpoint = project_post_generation_cuda_checkpoint(
        pre_generation_canonical_projection=checkpoint_source_projection,
        resident_boundary_tensors=resident_boundary_tensors,
        resident_birth_state=resident_birth_state,
        generation_result=generation_result,
        completed_generation_count=completed_generation_count,
        completed_interval_steps=(terminal_tick if partial_terminal else None),
        phase_e_active_modular_authority=phase_e_active_modular_authority,
        phase_e_active_modular_lifecycle_parent_slots=(
            phase_e_active_modular_lifecycle_parent_slots
        ),
        phase_e_active_modular_lifecycle_mutation_receipt=(
            phase_e_active_modular_lifecycle_mutation_receipt
        ),
        pre_materialized_cpu_fields=pre_materialized_cpu_fields,
        host_authoritative_cpu_policy_fields=(
            frozenset({"mlp_pack"})
            if persistent_state.get("hapb_full_cuda_mlp_pack_absent") is True
            and persistent_state.get("hapb_policy_owner") is not None
            else frozenset()
        ),
        generation_turnover_enabled=generation_turnover_enabled,
    )
    if terminal_metadata_updates is not None:
        projection = dict(checkpoint.projection)
        terminal_metadata = dict(
            _mapping(projection.get("metadata"), name="terminal checkpoint metadata")
        )
        terminal_metadata.update(terminal_metadata_updates)
        projection["metadata"] = terminal_metadata
        projection = _with_checkpoint_contract_metadata(projection)
        checkpoint = dataclasses.replace(
            checkpoint,
            projection=projection,
            projection_digest=projection_digest(projection),
        )
    checkpoint = with_resident_inheritance_sidecars_v1(
        checkpoint,
        persistent_state,
        cell_id=inheritance_sidecar_cell_id,
        genome_v2_checkpoint_authority_binding=(genome_v2_checkpoint_authority_binding),
        genome_v2_causal_physical_cell_ordinal=(genome_v2_causal_physical_cell_ordinal),
    )
    checkpoint_metadata = _mapping(
        checkpoint.projection.get("metadata"),
        name="terminal checkpoint metadata",
    )
    authority = _mapping(
        checkpoint.source_projection_authority,
        name="terminal checkpoint source_projection_authority",
    )
    if (
        type(checkpoint_metadata.get("generation")) is not int
        or checkpoint_metadata.get("generation") != terminal_generation
        or type(checkpoint_metadata.get("tick")) is not int
        or checkpoint_metadata.get("tick") != terminal_tick
        or type(authority.get("source_generation")) is not int
        or authority.get("source_generation") != terminal_generation
        or type(authority.get("source_tick")) is not int
        or authority.get("source_tick") != terminal_tick
        or authority.get("source_phase")
        != ("generation_boundary" if terminal_tick == 0 else "partial_generation")
    ):
        raise _fail("projected checkpoint is not the exact completed-prefix boundary")
    return checkpoint


def settle_patch_p2_b1_g4_direct_resident_v1(
    context: DirectResidentSettlementContextV1,
    *,
    direct_result: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> DirectResidentTerminalSettlementV1:
    """Decode four compact rows and project one terminal resident checkpoint."""

    result = _mapping(direct_result, name="direct_result")
    source = _context_member(context, "source")
    source_outputs = _mapping(
        getattr(source, "outputs", None),
        name="context.source.outputs",
    )
    source_generation = _exact_int(
        getattr(source, "generation_base", None),
        name="context.source.generation_base",
    )
    persistent_state = _with_materialized_event_history(
        _mapping(
            _context_member(context, "live_banks"),
            name="context.live_banks",
        ),
        result,
    )
    projection_inputs = _mapping(
        _context_member(context, "checkpoint_projection_inputs"),
        name="context.checkpoint_projection_inputs",
    )
    capability = _mapping(capability, name="capability")

    outputs = dict(source_outputs)
    for name in (
        "generation_summary_scalar",
        "runtime_bookkeeping_by_tick",
    ):
        value = result.get(name)
        if not isinstance(value, torch.Tensor):
            raise _fail(f"direct result omitted native tensor {name}")
        outputs[name] = value
    canonical_projection = _mapping(
        projection_inputs.get("pre_generation_canonical_projection"),
        name=(
            "context.checkpoint_projection_inputs.pre_generation_canonical_projection"
        ),
    )
    canonical_metadata = _mapping(
        canonical_projection.get("metadata"),
        name=(
            "context.checkpoint_projection_inputs."
            "pre_generation_canonical_projection.metadata"
        ),
    )
    if canonical_metadata.get("generation") != source_generation:
        raise _fail(
            "source-generation canonical boundary differs from the native owner"
        )
    birth_state = projection_inputs.get("resident_birth_state")

    _validate_direct_result(
        result,
        initial_generation=source_generation,
        physical_window_generations=_direct_result_physical_window_generations_v1(
            result,
            source,
        ),
    )
    _validate_capability(capability)
    compact_rows = _decode_compact_summary_rows(
        outputs,
        capability,
        source_generation=source_generation,
    )
    generation_turnover_enabled = _resolved_generation_turnover_enabled_v1(
        source,
        default=_generation_turnover_default_from_lifecycle_mode_v1(source),
    )
    _resolved_lifecycle_mode_abi_v1(
        source,
        generation_turnover_enabled=generation_turnover_enabled,
    )
    checkpoint = _project_terminal_checkpoint(
        canonical_projection,
        birth_state,
        outputs,
        persistent_state,
        source_generation=source_generation,
        generation_turnover_enabled=generation_turnover_enabled,
        inheritance_sidecar_cell_id=projection_inputs.get(
            "inheritance_sidecar_cell_id"
        ),
        genome_v2_checkpoint_authority_binding=projection_inputs.get(
            "genome_v2_checkpoint_authority_binding", ()
        ),
        genome_v2_causal_physical_cell_ordinal=projection_inputs.get(
            "genome_v2_causal_physical_cell_ordinal", 0
        ),
    )
    lifecycle_qacct_receipt = _lifecycle_qacct_receipt_from_route_v1(
        source,
        persistent_state,
        outputs,
        compact_rows,
        checkpoint,
        canonical_projection,
    )
    return DirectResidentTerminalSettlementV1(
        schema_version=SETTLEMENT_SCHEMA_VERSION,
        compact_summary_rows=compact_rows,
        terminal_checkpoint=checkpoint,
        terminal_generation=source_generation + LOGICAL_GENERATION_COUNT,
        lifecycle_qacct_receipt=lifecycle_qacct_receipt,
    )


def settle_patch_p2_b1_direct_resident_prefix_v2(
    context: DirectResidentSettlementContextV1,
    *,
    direct_result: Mapping[str, Any],
    capability: Mapping[str, Any],
    reporting_enabled: bool = True,
) -> DirectResidentTerminalSettlementV1:
    """Settle one exact g1-g4 prefix after completion or safe cancellation."""

    result = _mapping(direct_result, name="direct_result")
    completed = result.get("completed_generations")
    source = _context_member(context, "source")
    logical_generation_count = getattr(
        source, "logical_generation_count", LOGICAL_GENERATION_COUNT
    )
    if (
        type(logical_generation_count) is not int
        or logical_generation_count < 1
        or logical_generation_count > 128
    ):
        raise _fail("context source logical_generation_count must be in [1,128]")
    if type(completed) is not int or not 1 <= completed <= logical_generation_count:
        raise _fail(
            "direct result must contain an exact completed prefix within the "
            "admitted logical generation horizon"
        )
    ticks_per_generation = _exact_int(
        getattr(source, "ticks_per_generation", TICKS_PER_GENERATION),
        name="context.source.ticks_per_generation",
    )
    if ticks_per_generation not in {80, 160}:
        raise _fail("context source ticks_per_generation must be 80 or 160")
    source_outputs = _mapping(
        getattr(source, "outputs", None),
        name="context.source.outputs",
    )
    source_generation = _exact_int(
        getattr(source, "generation_base", None),
        name="context.source.generation_base",
    )
    persistent_state = _with_materialized_event_history(
        _mapping(
            _context_member(context, "live_banks"),
            name="context.live_banks",
        ),
        result,
    )
    projection_inputs = _mapping(
        _context_member(context, "checkpoint_projection_inputs"),
        name="context.checkpoint_projection_inputs",
    )
    capability = _mapping(capability, name="capability")
    tick_span = result.get("absolute_tick_span_native_owner") is True
    outputs = dict(source_outputs)
    generation_summary = result.get("generation_summary_scalar")
    if isinstance(generation_summary, torch.Tensor):
        outputs["generation_summary_scalar"] = generation_summary
    elif not tick_span:
        raise _fail("direct result omitted native tensor generation_summary_scalar")
    runtime_bookkeeping = result.get("runtime_bookkeeping_by_tick")
    if not isinstance(runtime_bookkeeping, torch.Tensor):
        raise _fail("direct result omitted native tensor runtime_bookkeeping_by_tick")
    outputs["runtime_bookkeeping_by_tick"] = runtime_bookkeeping

    _validate_direct_result(
        result,
        initial_generation=source_generation,
        physical_window_generations=_direct_result_physical_window_generations_v1(
            result,
            source,
        ),
        completed_generation_count=completed,
        logical_generation_count=logical_generation_count,
    )
    _validate_capability(capability)
    if type(reporting_enabled) is not bool:
        raise _fail("reporting_enabled must be boolean")
    generation_turnover_enabled = _resolved_generation_turnover_enabled_v1(
        source,
        default=_generation_turnover_default_from_lifecycle_mode_v1(source),
    )
    _resolved_lifecycle_mode_abi_v1(
        source,
        generation_turnover_enabled=generation_turnover_enabled,
    )
    canonical_projection = _mapping(
        projection_inputs.get("pre_generation_canonical_projection"),
        name=(
            "context.checkpoint_projection_inputs."
            "pre_generation_canonical_projection"
        ),
    )
    source_metadata = _mapping(
        canonical_projection.get("metadata"),
        name=(
            "context.checkpoint_projection_inputs."
            "pre_generation_canonical_projection.metadata"
        ),
    )
    if source_metadata.get("generation") != source_generation:
        raise _fail(
            "source-generation canonical boundary differs from the native owner"
        )
    source_tick = _exact_int(source_metadata.get("tick"), name="source tick")
    completed_tick_count: int | None = None
    terminal_tick = 0
    settlement_turnover_enabled = generation_turnover_enabled
    if tick_span:
        span_start = _exact_int(result.get("tick_span_start"), name="tick span start")
        span_count = _exact_int(result.get("tick_span_count"), name="tick span count")
        span_end = _exact_int(result.get("tick_span_end"), name="tick span end")
        terminal_boundary = result.get("terminal_generation_boundary")
        if (
            span_start != source_tick
            or span_count < 1
            or span_end != span_start + span_count
            or span_end > ticks_per_generation
            or type(terminal_boundary) is not bool
            or terminal_boundary != (span_end == ticks_per_generation)
        ):
            raise _fail("native absolute-tick span contradicts its source cursor")
        completed_tick_count = span_count
        terminal_tick = 0 if terminal_boundary else span_end
        settlement_turnover_enabled = (
            generation_turnover_enabled and terminal_boundary
        )
        if reporting_enabled:
            raise _fail(
                "partial lifecycle intervals cannot be reported as complete generations"
            )
    compact_rows = (
        _decode_compact_summary_rows(
            outputs,
            capability,
            completed_generation_count=completed,
            source_generation=source_generation,
        )
        if reporting_enabled
        else ()
    )
    segment_artifacts: tuple[Mapping[str, Any], ...] = ()
    terminal_metadata_updates = None
    cell_id = projection_inputs.get("inheritance_sidecar_cell_id")
    if type(cell_id) is str and cell_id:
        segment_artifact, terminal_metadata_updates = (
            _provider_native_segment_artifact_v1(
                cell_id=cell_id,
                cell_index=int(
                    projection_inputs.get(
                        "genome_v2_causal_physical_cell_ordinal",
                        0,
                    )
                    or 0
                ),
                source_projection=canonical_projection,
                compact_rows=compact_rows,
            )
        )
        segment_artifacts = (segment_artifact,)
    checkpoint = _project_terminal_checkpoint(
        canonical_projection,
        projection_inputs.get("resident_birth_state"),
        outputs,
        persistent_state,
        completed_generation_count=completed,
        ticks_per_generation=ticks_per_generation,
        source_generation=source_generation,
        source_tick=source_tick,
        completed_tick_count=completed_tick_count,
        terminal_tick=terminal_tick,
        generation_turnover_enabled=settlement_turnover_enabled,
        inheritance_sidecar_cell_id=projection_inputs.get(
            "inheritance_sidecar_cell_id"
        ),
        genome_v2_checkpoint_authority_binding=projection_inputs.get(
            "genome_v2_checkpoint_authority_binding", ()
        ),
        genome_v2_causal_physical_cell_ordinal=projection_inputs.get(
            "genome_v2_causal_physical_cell_ordinal", 0
        ),
        terminal_metadata_updates=terminal_metadata_updates,
    )
    lifecycle_qacct_receipt = (
        _lifecycle_qacct_receipt_from_route_v1(
            source,
            persistent_state,
            outputs,
            compact_rows,
            checkpoint,
            canonical_projection,
            ticks_per_generation=ticks_per_generation,
        )
        if compact_rows
        else None
    )
    return DirectResidentTerminalSettlementV1(
        schema_version=SETTLEMENT_SCHEMA_VERSION,
        compact_summary_rows=compact_rows,
        terminal_checkpoint=checkpoint,
        terminal_generation=(
            source_generation + completed if terminal_tick == 0 else source_generation
        ),
        terminal_tick=terminal_tick,
        segment_artifacts=segment_artifacts,
        lifecycle_qacct_receipt=lifecycle_qacct_receipt,
    )


__all__ = [
    "COMPACT_SUMMARY_SCHEMA_VERSION",
    "CudaArchitectureBDirectResidentSettlementError",
    "DirectResidentSettlementContextV1",
    "DirectResidentSettlementSourceV1",
    "DirectResidentTerminalSettlementV1",
    "GENERATION_SUMMARY_COLUMNS",
    "GENERATION_SUMMARY_LAYOUT_VERSION",
    "SETTLEMENT_SCHEMA_VERSION",
    "settle_patch_p2_b1_g4_direct_resident_v1",
    "settle_patch_p2_b1_direct_resident_prefix_v2",
    "with_resident_inheritance_sidecars_v1",
]
