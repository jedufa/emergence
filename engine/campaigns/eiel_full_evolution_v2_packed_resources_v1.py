"""GPU-only VRAM estimator for tape-free V2 packed EIEL_FULL_EVOLUTION_V1.

Calibrated from packed owners on frozen 63cc0cdc9 plus measured B144 occupancy:

- B24 G7 tick reserved = 966787072 (922 MiB)
- B48 prepare reserved = 1788870656 (1706 MiB)
- B144 nvidia-smi tick used = 5704253440 (5440 MiB)
- B144 Q4-B torch reserved = 5744099328 (the high-B calibration point)

B<=48 uses reserved(B) = B24 + (B-24) * (B48-B24) // 24.
B>48 uses reserved(B) = B48 + (B-48) * (B144-B48) // 96 so B144 is not
under-predicted. Host checkpoint bytes are excluded. Campaign declared
VRAM budgets must still fit torch-visible total minus 4% headroom.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from engine.campaigns.resources import (
    CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY,
    MIB,
    PATCH_GATE_SECONDS_PER_CELL_GENERATION,
    PATCH_WALL_PROJECTION_SAFETY_FACTOR,
    RESIDENT_COMPACT_CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY,
    ResourceEstimateError,
    planning_acs_native_fixed_mlp_b136_checkpoint_upper_bound_v1,
    planning_disk_upper_bound_absolute_ticks_v1,
    planning_disk_upper_bound_v1,
)


ESTIMATOR_ID: Final[str] = "eiel_full_evolution_v2_packed_resident_resources_v1"
FE_V2_PACKED_MIN_BATCH_CELLS: Final[int] = 1
FE_V2_PACKED_MAX_BATCH_CELLS: Final[int] = 256
FE_V2_PACKED_RESIDENT_CAPACITY: Final[int] = 512
FE_V2_PACKED_SOURCE_WINDOW_TICKS: Final[int] = 160
FE_V2_PACKED_B24_G7_RESERVED_BYTES: Final[int] = 966_787_072
FE_V2_PACKED_B48_PREPARE_RESERVED_BYTES: Final[int] = 1_788_870_656
# Q4-B Launchpad ACS capture_barrier (batch-0000/manifest.json, hashed):
# max_memory_allocated_bytes=5702117888, memory_reserved_bytes=5744099328.
# nvidia-smi tick used=5704253440. Calibration is the max of those occupancy
# fields. Key max_memory_reserved is absent (uninstrumented).
FE_V2_PACKED_B144_TICK_USED_BYTES: Final[int] = 5_744_099_328
FE_V2_PACKED_B144_NVIDIA_SMI_USED_BYTES: Final[int] = 5_704_253_440
FE_V2_PACKED_B144_Q4B_MAX_ALLOCATED_BYTES: Final[int] = 5_702_117_888
FE_V2_PACKED_B24_BATCH_CELLS: Final[int] = 24
FE_V2_PACKED_B48_BATCH_CELLS: Final[int] = 48
FE_V2_PACKED_B144_BATCH_CELLS: Final[int] = 144
FE_V2_PACKED_PHYSICAL_VRAM_BYTES: Final[int] = 6 * 1024 * 1024 * 1024
FE_V2_PACKED_ALLOCATOR_JITTER_PAD_BYTES: Final[int] = 8 * MIB
FE_V2_PACKED_B136_BATCH_CELLS: Final[int] = 136
FE_V2_PACKED_B136_EVENT_ROWS_PER_CELL: Final[int] = 41_804
FE_V2_PACKED_B136_EVENT_ROW_BYTES: Final[int] = 141
FE_V2_PACKED_B136_EVENT_BYTES: Final[int] = 801_633_504
FE_V2_PACKED_B136_CONSERVATIVE_PEAK_BYTES: Final[int] = 5_441_799_168
FE_V2_PACKED_TORCH_VISIBLE_DEVICE_BYTES: Final[int] = 6_027_608_064
FE_V2_PACKED_FROZEN_RESERVE_BYTES: Final[int] = 413_138_944
FE_V2_PACKED_MINIMUM_FREE_BYTES: Final[int] = 536_870_912
FE_V2_PACKED_ONE_CELL_HOST_SCRATCH_BYTES: Final[int] = (
    CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY
)
FE_V2_PACKED_B48_PREPARED_OWNER_RECEIPT_SHA256: Final[str] = (
    "2cbce6a64acf5d0f376ff2ca3e9ddffe4b3bff8c6a105d567aa949c0441c08b9"
)
FE_V2_PACKED_FROZEN_SOURCE: Final[str] = (
    "63cc0cdc9b985507a3a70bb089dab99d3ce809df"
)


def is_eiel_full_evolution_v2_packed_identity_v1(
    dimensions: Mapping[str, Any],
) -> bool:
    """Return True only for the tape-free V2 FE packed identity."""

    if dimensions.get("cpython_gauss_segment_capacity") != 0:
        return False
    if dimensions.get("policy_birth_segment_refill_enable") is not False:
        return False
    resident_capacity = dimensions.get("resident_capacity")
    if (
        resident_capacity is not None
        and resident_capacity != FE_V2_PACKED_RESIDENT_CAPACITY
    ):
        return False
    source_window = dimensions.get("source_window_ticks")
    if (
        source_window is not None
        and source_window != FE_V2_PACKED_SOURCE_WINDOW_TICKS
    ):
        return False
    ticks = dimensions.get("ticks_per_generation")
    if ticks is not None and ticks != FE_V2_PACKED_SOURCE_WINDOW_TICKS:
        return False
    return True


def eiel_full_evolution_v2_packed_reserved_bytes_v1(batch_cells: int) -> int:
    """Return the first-order GPU reserved prediction for one packed B."""

    if (
        type(batch_cells) is not int
        or not FE_V2_PACKED_MIN_BATCH_CELLS
        <= batch_cells
        <= FE_V2_PACKED_MAX_BATCH_CELLS
    ):
        raise ResourceEstimateError(
            "batch_cells must be an integer in [1, 256]"
        )
    if batch_cells <= FE_V2_PACKED_B48_BATCH_CELLS:
        delta = (
            FE_V2_PACKED_B48_PREPARE_RESERVED_BYTES
            - FE_V2_PACKED_B24_G7_RESERVED_BYTES
        )
        span = FE_V2_PACKED_B48_BATCH_CELLS - FE_V2_PACKED_B24_BATCH_CELLS
        return FE_V2_PACKED_B24_G7_RESERVED_BYTES + (
            (batch_cells - FE_V2_PACKED_B24_BATCH_CELLS) * delta // span
        )
    delta = (
        FE_V2_PACKED_B144_TICK_USED_BYTES
        - FE_V2_PACKED_B48_PREPARE_RESERVED_BYTES
    )
    span = FE_V2_PACKED_B144_BATCH_CELLS - FE_V2_PACKED_B48_BATCH_CELLS
    return FE_V2_PACKED_B48_PREPARE_RESERVED_BYTES + (
        (batch_cells - FE_V2_PACKED_B48_BATCH_CELLS) * delta // span
    )


def estimate_eiel_full_evolution_v2_packed_resident_resources_v1(
    *,
    batch_cells: int,
    logical_trajectories: int,
    checkpoint_boundaries: int,
    logical_generations: int | None = None,
    resident_segment_generations: int | None = None,
    physical_window_generations: int | None = None,
    ticks_per_generation: int | None = None,
    time_authority: str | None = None,
    total_ticks: int | None = None,
    resident_owner_span_ticks: int | None = None,
    native_dispatch_window_ticks: int | None = None,
    source_window_ticks: int | None = None,
    event_window_capacity: int | None = None,
    resident_capacity: int | None = None,
    ordinary_births_enabled: bool | None = None,
    cpython_gauss_segment_capacity: int | None = None,
    policy_birth_segment_refill_enable: bool | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Admit tape-free V2 packed FE using the B24/B48 GPU reserved line.

    Native requests use ``total_ticks`` and explicit resident spans.  The
    optional generation arguments remain only for the historical estimator
    caller; they are not required or emitted for the active tick-native path.
    """

    native = time_authority == "absolute_tick_v1"
    if native:
        retired_dimensions = {
            "logical_generations": logical_generations,
            "resident_segment_generations": resident_segment_generations,
            "physical_window_generations": physical_window_generations,
            "ticks_per_generation": ticks_per_generation,
        }
        contaminated = sorted(
            name for name, value in retired_dimensions.items() if value is not None
        )
        if contaminated:
            raise ResourceEstimateError(
                "absolute-tick packed FE forbids retired generation dimensions: "
                f"{contaminated}"
            )
        if (
            type(total_ticks) is not int
            or total_ticks < 1
            or type(resident_owner_span_ticks) is not int
            or resident_owner_span_ticks < 1
            or type(native_dispatch_window_ticks) is not int
            or native_dispatch_window_ticks < 1
            or native_dispatch_window_ticks > resident_owner_span_ticks
            or type(source_window_ticks) is not int
            or source_window_ticks < 1
            or source_window_ticks > resident_owner_span_ticks
            or type(event_window_capacity) is not int
            or event_window_capacity < 1
        ):
            raise ResourceEstimateError(
                "absolute-tick packed FE resource dimensions are invalid"
            )
    elif (
        type(logical_generations) is not int
        or logical_generations < 1
        or type(resident_segment_generations) is not int
        or resident_segment_generations < 1
    ):
        raise ResourceEstimateError(
            "legacy packed FE logical generations and resident segments must be "
            "integers >= 1"
        )

    identity = {
        "cpython_gauss_segment_capacity": cpython_gauss_segment_capacity,
        "policy_birth_segment_refill_enable": policy_birth_segment_refill_enable,
        "resident_capacity": resident_capacity,
        "ticks_per_generation": None if native else ticks_per_generation,
        "source_window_ticks": source_window_ticks,
    }
    if not is_eiel_full_evolution_v2_packed_identity_v1(identity):
        raise ResourceEstimateError(
            "estimator requires V2 tape-free FE identity "
            "(gauss==0, CPU refill disabled, C512, bounded source window 160)"
        )
    reserved = eiel_full_evolution_v2_packed_reserved_bytes_v1(batch_cells)
    upper_bound = reserved + FE_V2_PACKED_ALLOCATOR_JITTER_PAD_BYTES
    amended_b136_profile = bool(
        native
        and batch_cells == FE_V2_PACKED_B136_BATCH_CELLS
        and event_window_capacity == FE_V2_PACKED_B136_EVENT_ROWS_PER_CELL
        and resident_capacity == FE_V2_PACKED_RESIDENT_CAPACITY
        and source_window_ticks == FE_V2_PACKED_SOURCE_WINDOW_TICKS
        and ordinary_births_enabled is not False
    )
    if amended_b136_profile:
        # CAPACITY-PROOF-0 amendment v1 is the exact physical-B136 authority.
        # It supersedes the generic B24/B48/B144 interpolation only for this
        # fully identified route; the independent reserve remains unchanged.
        reserved = FE_V2_PACKED_B136_CONSERVATIVE_PEAK_BYTES
        upper_bound = FE_V2_PACKED_B136_CONSERVATIVE_PEAK_BYTES
    if (
        type(logical_trajectories) is not int
        or logical_trajectories < 1
        or type(checkpoint_boundaries) is not int
        or checkpoint_boundaries < 1
    ):
        raise ResourceEstimateError(
            "trajectories and checkpoints must be integers >= 1"
        )
    compact_checkpoint_estimate: dict[str, int | str] | None = None
    if native and amended_b136_profile:
        compact_checkpoint_estimate = (
            planning_acs_native_fixed_mlp_b136_checkpoint_upper_bound_v1(
                logical_cells=logical_trajectories,
                batch_cells=batch_cells,
                checkpoint_boundaries=checkpoint_boundaries,
            )
        )
        summary_bytes = logical_trajectories * total_ticks * 32
        disk = summary_bytes + int(
            compact_checkpoint_estimate["checkpoint_bytes_upper_bound"]
        )
        wall_projection = int(
            logical_trajectories
            * total_ticks
            / FE_V2_PACKED_SOURCE_WINDOW_TICKS
            * PATCH_GATE_SECONDS_PER_CELL_GENERATION
            * PATCH_WALL_PROJECTION_SAFETY_FACTOR
            + 0.999999
        )
    elif native:
        disk = planning_disk_upper_bound_absolute_ticks_v1(
            logical_trajectories=logical_trajectories,
            total_ticks=total_ticks,
            checkpoint_boundaries=checkpoint_boundaries,
        )
        wall_projection = int(
            logical_trajectories
            * total_ticks
            / FE_V2_PACKED_SOURCE_WINDOW_TICKS
            * PATCH_GATE_SECONDS_PER_CELL_GENERATION
            * PATCH_WALL_PROJECTION_SAFETY_FACTOR
            + 0.999999
        )
    else:
        disk = planning_disk_upper_bound_v1(
            logical_trajectories=logical_trajectories,
            logical_generations=logical_generations,
            checkpoint_boundaries=checkpoint_boundaries,
            checkpoint_bytes_per_trajectory_boundary=(
                RESIDENT_COMPACT_CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY
            ),
        )
        wall_projection = int(
            logical_trajectories
            * logical_generations
            * PATCH_GATE_SECONDS_PER_CELL_GENERATION
            * PATCH_WALL_PROJECTION_SAFETY_FACTOR
            + 0.999999
        )
    result = {
        "authority": "provider_resource_admission_estimate_v1",
        "vram_bytes_upper_bound": upper_bound,
        "vram_bytes_projection": reserved,
        "vram_bytes_unavailable_reason": None,
        "disk_bytes_upper_bound": disk,
        "wall_seconds_upper_bound": wall_projection,
        "wall_seconds_projection": wall_projection,
        "wall_seconds_bound_authority": "estimated_conservative_not_audited",
        "wall_seconds_unavailable_reason": None,
        "resource_estimator_id": ESTIMATOR_ID,
        "resource_estimator_provenance": (
            "B24 G7 tick reserved and B48 prepare reserved on frozen FE "
            "source 63cc0cdc9; B>48 uses B144 Q4-B torch reserved "
            "5744099328 B (nvidia-smi used 5704253440 B); GPU reserved/used "
            "only; host checkpoint excluded"
        ),
        "admission_note": (
            "V2 packed FE GPU-only piecewise curve; B144 tick used is the "
            "high-B calibration point; B160 CUDA OOM is outside the physical "
            "card; 768 MiB reserve is not stacked here"
        ),
        "resource_estimator_assumptions": {
            "memory_domain": "gpu_reserved_cuda_only",
            "host_projection_counted_as_vram": False,
            "gru_controller_allowance_bytes": 0,
            "per_cell_allocator_margin_bytes": 0,
            "allocator_jitter_pad_bytes": FE_V2_PACKED_ALLOCATOR_JITTER_PAD_BYTES,
            "admitted_batch_range": [
                FE_V2_PACKED_MIN_BATCH_CELLS,
                FE_V2_PACKED_MAX_BATCH_CELLS,
            ],
            "resident_capacity": FE_V2_PACKED_RESIDENT_CAPACITY,
            "cpython_gauss_segment_capacity": 0,
            "policy_birth_segment_refill_enable": False,
            "b24_g7_reserved_bytes": FE_V2_PACKED_B24_G7_RESERVED_BYTES,
            "b48_prepare_reserved_bytes": FE_V2_PACKED_B48_PREPARE_RESERVED_BYTES,
            "b144_tick_used_bytes": FE_V2_PACKED_B144_TICK_USED_BYTES,
            "physical_vram_bytes": FE_V2_PACKED_PHYSICAL_VRAM_BYTES,
            "b48_prepared_owner_receipt_sha256": (
                FE_V2_PACKED_B48_PREPARED_OWNER_RECEIPT_SHA256
            ),
            "frozen_source": FE_V2_PACKED_FROZEN_SOURCE,
            "tick_transient_extrapolated": False,
            "host_checkpoint_import_scratch_bound_bytes": (
                FE_V2_PACKED_ONE_CELL_HOST_SCRATCH_BYTES
            ),
            "host_checkpoint_import_scratch_concurrency": 1,
            "host_checkpoint_import_scratch_horizon_invariant": True,
            "durable_output_checkpoint_bytes_scale": (
                "O(retained_physical_checkpoint_boundaries)"
                if amended_b136_profile
                else "O(logical_trajectories*horizon)"
            ),
        },
        "physical_batch_cells": batch_cells,
        "logical_trajectories": logical_trajectories,
        "peak_resident_gpu_owner_event_bytes": upper_bound,
        "peak_host_checkpoint_import_scratch_bytes": (
            FE_V2_PACKED_ONE_CELL_HOST_SCRATCH_BYTES
        ),
        "total_durable_output_checkpoint_bytes_upper_bound": disk,
        "resident_gpu_owner_event_bytes_horizon_invariant": True,
        "host_checkpoint_import_scratch_bytes_horizon_invariant": True,
        "resource_dimensions_schema_version": (
            "eiel_three_domain_resource_dimensions_v1"
        ),
    }
    if native:
        result["time_authority"] = "absolute_tick_v1"
        result["total_ticks"] = total_ticks
        result["resident_owner_span_ticks"] = resident_owner_span_ticks
        result["native_dispatch_window_ticks"] = native_dispatch_window_ticks
        result["source_window_ticks"] = source_window_ticks
        result["event_window_capacity"] = event_window_capacity
        result["resident_event_bytes_horizon_invariant"] = True
        result["resident_source_bytes_horizon_invariant"] = True
        if compact_checkpoint_estimate is not None:
            result.update(compact_checkpoint_estimate)
            result["checkpoint_representation"] = "aggregate_physical_batch_v1"
            result["explicit_checkpoint_boundaries"] = checkpoint_boundaries
            result["owner_transition_checkpoint_count"] = 0
            result["generation_zero_checkpoint_count"] = 0
        result["resident_event_rows_per_cell"] = (
            FE_V2_PACKED_B136_EVENT_ROWS_PER_CELL
            if amended_b136_profile
            else event_window_capacity
        )
        result["resident_event_row_bytes"] = FE_V2_PACKED_B136_EVENT_ROW_BYTES
        result["resident_event_bytes"] = (
            FE_V2_PACKED_B136_EVENT_BYTES
            if amended_b136_profile
            else event_window_capacity
            * FE_V2_PACKED_B136_EVENT_ROW_BYTES
            * batch_cells
        )
        result["amended_b136_capacity_profile"] = amended_b136_profile
        result["resource_estimator_assumptions"].update(
            {
                "time_authority": "absolute_tick_v1",
                "total_ticks": total_ticks,
                "resident_owner_span_ticks": resident_owner_span_ticks,
                "native_dispatch_window_ticks": native_dispatch_window_ticks,
                "source_window_ticks": source_window_ticks,
                "event_window_capacity": event_window_capacity,
                "resident_event_capacity_horizon_invariant": True,
                "resident_source_capacity_horizon_invariant": True,
                "event_capacity_authority": (
                    "CAPACITY-PROOF-0_amendment_v1"
                    if amended_b136_profile
                    else "route_declared_tick_native_capacity"
                ),
            }
        )
        if amended_b136_profile:
            result["resource_estimator_assumptions"].update(
                {
                    "amended_b136_event_rows_per_cell": (
                        FE_V2_PACKED_B136_EVENT_ROWS_PER_CELL
                    ),
                    "amended_b136_event_row_bytes": FE_V2_PACKED_B136_EVENT_ROW_BYTES,
                    "amended_b136_event_bytes": FE_V2_PACKED_B136_EVENT_BYTES,
                    "frozen_torch_visible_device_bytes": (
                        FE_V2_PACKED_TORCH_VISIBLE_DEVICE_BYTES
                    ),
                    "frozen_reserve_bytes": FE_V2_PACKED_FROZEN_RESERVE_BYTES,
                    "frozen_minimum_free_bytes": FE_V2_PACKED_MINIMUM_FREE_BYTES,
                    "post_reserve_margin_bytes": (
                        FE_V2_PACKED_TORCH_VISIBLE_DEVICE_BYTES
                        - FE_V2_PACKED_B136_CONSERVATIVE_PEAK_BYTES
                        - FE_V2_PACKED_FROZEN_RESERVE_BYTES
                    ),
                    "post_minimum_free_margin_bytes": (
                        FE_V2_PACKED_TORCH_VISIBLE_DEVICE_BYTES
                        - FE_V2_PACKED_B136_CONSERVATIVE_PEAK_BYTES
                        - FE_V2_PACKED_MINIMUM_FREE_BYTES
                    ),
                }
            )
    else:
        result.update(
            {
                "logical_generations": logical_generations,
                "ticks_per_generation": FE_V2_PACKED_SOURCE_WINDOW_TICKS,
                "resident_segment_generations": resident_segment_generations,
            }
        )
        result["resource_estimator_assumptions"].update(
            {
                "ticks_per_generation": FE_V2_PACKED_SOURCE_WINDOW_TICKS,
                "physical_window_generations": physical_window_generations,
                "resident_segment_generations": resident_segment_generations,
            }
        )
    return result


__all__ = [
    "ESTIMATOR_ID",
    "FE_V2_PACKED_B136_CONSERVATIVE_PEAK_BYTES",
    "FE_V2_PACKED_B136_EVENT_BYTES",
    "FE_V2_PACKED_B136_EVENT_ROW_BYTES",
    "FE_V2_PACKED_B136_EVENT_ROWS_PER_CELL",
    "FE_V2_PACKED_FROZEN_RESERVE_BYTES",
    "FE_V2_PACKED_MINIMUM_FREE_BYTES",
    "FE_V2_PACKED_ONE_CELL_HOST_SCRATCH_BYTES",
    "FE_V2_PACKED_TORCH_VISIBLE_DEVICE_BYTES",
    "FE_V2_PACKED_B144_NVIDIA_SMI_USED_BYTES",
    "FE_V2_PACKED_B144_TICK_USED_BYTES",
    "FE_V2_PACKED_MAX_BATCH_CELLS",
    "FE_V2_PACKED_PHYSICAL_VRAM_BYTES",
    "eiel_full_evolution_v2_packed_reserved_bytes_v1",
    "estimate_eiel_full_evolution_v2_packed_resident_resources_v1",
    "is_eiel_full_evolution_v2_packed_identity_v1",
]
