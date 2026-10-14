"""Auditable resource estimates for declarative campaign providers.

The arithmetic here is deliberately independent of qualification gates and
tool scripts.  It carries forward only the conservative Patch resident-memory
accounting constants that are shared by execution admission.  Qualification
still decides whether a particular artifact may execute a request.
"""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Mapping, Sequence
from typing import Any, Final


PATCH_RESIDENT_WINDOW_GENERATIONS: Final[int] = 2
PATCH_MIN_RESIDENT_WINDOW_GENERATIONS: Final[int] = 1
PATCH_MAX_RESIDENT_WINDOW_GENERATIONS: Final[int] = 2
PATCH_RESIDENT_WINDOW_G1_ROUTE_ID: Final[str] = (
    "eiel_cuda_architecture_b_resident_window_g1_v1"
)
PATCH_RESIDENT_WINDOW_G2_ROUTE_ID: Final[str] = (
    "eiel_cuda_architecture_b_resident_window_g2_v1"
)
PATCH_MIN_LOGICAL_GENERATIONS: Final[int] = 3
PATCH_MAX_LOGICAL_GENERATIONS: Final[int] = 128
PATCH_MIN_BATCH_CELLS: Final[int] = 1
PATCH_MAX_BATCH_CELLS: Final[int] = 72
PATCH_AUDITED_VRAM_BATCH_MAX: Final[int] = 2

MIB: Final[int] = 1024 * 1024
SHARED_CONTROLLER_ALLOWANCE_BYTES: Final[int] = 1536 * MIB
PATCH_STATE_AND_RECOVERY_BYTES_PER_CELL: Final[int] = 11_264
PATCH_SCRATCH_BYTES_PER_CELL: Final[int] = 4_096
LIFECYCLE_SCRATCH_BYTES_PER_CELL: Final[int] = 163_840
POLICY_EPISODES_PER_GENERATION: Final[int] = 142
POLICY_RNG_TAPE_BYTES_PER_EPISODE: Final[int] = 45_554
TICKS_PER_GENERATION: Final[int] = 80
PATCH_MAX_ADMITTED_TICKS_PER_GENERATION: Final[int] = 160
PATCH_PHYSICAL_WINDOW_TICKS: Final[int] = (
    PATCH_RESIDENT_WINDOW_GENERATIONS * PATCH_MAX_ADMITTED_TICKS_PER_GENERATION
)
PATCH_MAX_BIRTHS_PER_TICK: Final[int] = 64
PATCH_TURNOVER_EPISODES_PER_GENERATION: Final[int] = 62
PATCH_PHYSICAL_WINDOW_TURNOVER_EPISODES: Final[int] = (
    PATCH_RESIDENT_WINDOW_GENERATIONS * PATCH_TURNOVER_EPISODES_PER_GENERATION
)
PATCH_FIXED_MLP_POLICY_EPISODE_CAPACITY: Final[int] = (
    PATCH_PHYSICAL_WINDOW_TICKS * PATCH_MAX_BIRTHS_PER_TICK
    + PATCH_PHYSICAL_WINDOW_TURNOVER_EPISODES
)
PATCH_FIXED_MLP_BASE_OBSERVATION_SIZE: Final[int] = 74
PATCH_FIXED_MLP_HIDDEN_SIZE: Final[int] = 64
PATCH_FIXED_MLP_ACTION_COUNT: Final[int] = 7
PATCH_FIXED_MLP_WIDTH: Final[int] = (
    PATCH_FIXED_MLP_HIDDEN_SIZE * PATCH_FIXED_MLP_BASE_OBSERVATION_SIZE
    + PATCH_FIXED_MLP_HIDDEN_SIZE
    + PATCH_FIXED_MLP_HIDDEN_SIZE * PATCH_FIXED_MLP_HIDDEN_SIZE
    + PATCH_FIXED_MLP_HIDDEN_SIZE
    + PATCH_FIXED_MLP_ACTION_COUNT * PATCH_FIXED_MLP_HIDDEN_SIZE
    + PATCH_FIXED_MLP_ACTION_COUNT
)
PATCH_TORCH_RNG_STATE_BYTES: Final[int] = 5_056
PATCH_TORCH_RNG_PACKED_STATE_BYTES: Final[int] = 2_560
PATCH_TORCH_RNG_STATE_ENCODING: Final[str] = (
    "torch_cpu_rng_state_5056_u32_middle_v1"
)
PATCH_TORCH_RNG_STATE_ABI: Final[dict[str, object]] = {
    "abi_schema_version": "eiel_torch_cpu_mt19937_state_2560_abi_v1",
    "codec_schema_version": PATCH_TORCH_RNG_STATE_ENCODING,
    "torch_version": "2.10.0+cu128",
    "torch_git_version": "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
    "torch_cuda_build": "12.8",
    "torch_cxx11_abi": True,
    "torch_build_config_sha256": (
        "62a3328b6570991ff3dd369f547254c82ca716de947cd8a66374b52903f6793e"
    ),
    "platform_machine": "x86_64",
    "byteorder": "little",
    "uint64_bytes": 8,
    "uint32_bytes": 4,
    "full_state_bytes": PATCH_TORCH_RNG_STATE_BYTES,
    "packed_state_bytes": PATCH_TORCH_RNG_PACKED_STATE_BYTES,
    "leading_full_uint64_lanes": 3,
    "middle_low_uint32_lanes": 624,
    "trailing_full_uint64_lanes": 5,
    "leading_full_bytes": 24,
    "trailing_full_byte_offset": 5_016,
    "trailing_packed_byte_offset": 2_520,
    "golden_seed": 0x1234_5678_ABCD_EF01,
    "golden_float32_uniform_draws": 9_415,
    "golden_state_sha256": (
        "961fb072efe82dcf2bd9aed93642492b7d3503c66b06adf6771e910620151a1d"
    ),
    "source_identity": "torch.Generator(device=cpu).get_state",
    "physical_profile_identity": (
        "eiel_cuda_architecture_b_fixed_mlp_policy_post_state_v1"
    ),
    "route_binding": "execution_route_sha256",
    "abi_fingerprint_sha256": (
        "dd3966ffb664fc6bb7b6c99a42436a8945e8b0fc4a0adf15238de16d07ddf5b4"
    ),
    "portable_checkpoint_format": False,
    "execution_tape_encoding": True,
}
PATCH_CPYTHON_GAUSS_RAW_CAPACITY_PER_BIRTH: Final[int] = 512
PATCH_POLICY_MUTATION_COMPACT_WIDTH: Final[int] = 1
PATCH_DENSE_OWNER_ALLOCATOR_MARGIN_BYTES_PER_CELL: Final[int] = 64 * MIB
PATCH_CROSSOVER_MASK_ENCODING: Final[str] = "bitpack_lsb0_v1"
DRIVE_NOISE_CHANNEL_COUNT: Final[int] = 17
PATCH_CAPACITY: Final[int] = 384
EVENT_BASE_CAPACITY_ROWS: Final[int] = 65_536
EVENT_ROWS_PER_GENERATION: Final[int] = 41_164
EVENT_ROW_STORAGE_BYTES: Final[int] = 141
IDENTITY_GROWTH_ROWS_PER_GENERATION: Final[int] = 142
IDENTITY_ROW_STORAGE_BYTES: Final[int] = 4
PATCH_STAT_COLUMNS: Final[int] = 25
NONCOMPACT_BOOKKEEPING_SCALARS_PER_TICK: Final[int] = 5
SUMMARY_BYTES_PER_TRAJECTORY_GENERATION: Final[int] = 4_096
CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY: Final[int] = 64 * MIB
# Architecture B's compact resident checkpoint projection is materially
# smaller than the generic planning ceiling.  Keep this estimate explicit so
# bounded physical batches do not inherit the old logical-B136 storage
# reservation.  The 18 MiB ceiling is above the largest retained 5000-tick
# compact projection observed on the bound source route.
RESIDENT_COMPACT_CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY: Final[int] = 18 * MIB

# Narrow source-bound projection for the ACS100X physical-B136 aggregate route.
# These values are used only after the caller proves the full fixed-MLP/tick
# control identity; arbitrary larger batches retain the conservative envelope.
ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS: Final[int] = 136
ACS_NATIVE_FIXED_MLP_B136_ACTIVE_PAYLOAD_BYTES: Final[int] = 3_775_071_136
# PREQUAL-4A v13 measured one 3,884,712,358-byte physical aggregate.  The
# prior 64 MiB metadata ceiling underpredicted that object by 38,333,958
# bytes.  A 128 MiB header allowance raises the per-object bound to
# 3,913,487,264 bytes while keeping two retained physical checkpoints below
# the unchanged 8,000,000,000-byte checkpoint-zone budget.
ACS_NATIVE_FIXED_MLP_B136_MEASURED_AGGREGATE_BYTES: Final[int] = 3_884_712_358
ACS_NATIVE_AGGREGATE_HEADER_MAX_BYTES: Final[int] = 128 * MIB
ACS_NATIVE_AGGREGATE_FOOTER_MAX_BYTES: Final[int] = 4 * MIB
ACS_NATIVE_AGGREGATE_METADATA_MAX_BYTES_PER_OBJECT: Final[int] = (
    ACS_NATIVE_AGGREGATE_HEADER_MAX_BYTES
    + ACS_NATIVE_AGGREGATE_FOOTER_MAX_BYTES
    + 4_096
)
ACS_NATIVE_FIXED_MLP_B136_ACCOUNTING_BASIS: Final[str] = (
    "acs100x_fixed_mlp_b136_aggregate_projection_v1"
)
ACS_NATIVE_FIXED_MLP_B136_ACTIVE_VRAM_BYTES: Final[int] = 5_400_914_432
ACS_NATIVE_FIXED_MLP_B136_TRANSIENT_VRAM_BYTES: Final[int] = 5_429_526_528
ACS_NATIVE_FIXED_MLP_B136_RESERVE_BYTES: Final[int] = 413_138_944
ACS_NATIVE_FIXED_MLP_B136_RESOURCE_ESTIMATOR_ID: Final[str] = (
    "acs100x-physical-b136-g7-t160-v1"
)

WALL_BOUND_BLOCKER: Final[str] = (
    "no audited Patch resident wall-time ceiling exists for arbitrary B1..B72 "
    "and G3..G128; measured qualification timings are observations, not bounds"
)
PATCH_GATE_SECONDS_PER_CELL_GENERATION: Final[float] = 81.518446 / 28
PATCH_WALL_PROJECTION_SAFETY_FACTOR: Final[int] = 11
VRAM_BOUND_BLOCKER: Final[str] = (
    "the current Patch resident memory accounting is audited through B2 only; "
    "larger-batch arithmetic is a planning projection until an allocation-only "
    "audit validates the exact artifact and selected physical-window route"
)
GENOME_V2_NO_BIRTH_B48_BATCH_CELLS: Final[int] = 48
GENOME_V2_NO_BIRTH_B48_RESIDENT_CAPACITY: Final[int] = 256
GENOME_V2_NO_BIRTH_B48_SEGMENT_GENERATIONS_MAX: Final[int] = 10
GENOME_V2_NO_BIRTH_B48_MEASURED_OWNER_PEAK_BYTES: Final[int] = 3_911_188_480
GENOME_V2_NO_BIRTH_B48_CONSERVATIVE_OWNER_PROJECTION_BYTES: Final[int] = (
    GENOME_V2_NO_BIRTH_B48_MEASURED_OWNER_PEAK_BYTES + 384 * MIB
)
GENOME_V2_NO_BIRTH_B48_RECEIPT_SHA256: Final[str] = (
    "797accc35a76f76919614c213e7ea1b462a4743089cc300ac2f5a5cd1834305a"
)
WDL_FIXED_GV2_OB_B48_BATCH_CELLS: Final[int] = 48
WDL_FIXED_GV2_OB_B48_RESIDENT_CAPACITY: Final[int] = 256
WDL_FIXED_GV2_OB_B48_SEGMENT_GENERATIONS_MAX: Final[int] = 5
WDL_FIXED_GV2_OB_B48_LOGICAL_GENERATIONS: Final[int] = 30
WDL_FIXED_GV2_OB_B48_LOGICAL_TRAJECTORIES: Final[int] = 54
WDL_FIXED_GV2_OB_B48_CHECKPOINT_BOUNDARIES: Final[int] = 8
WDL_FIXED_GV2_OB_B48_MEASURED_PREPARED_OWNER_PEAK_BYTES: Final[int] = (
    4_982_833_152
)
WDL_FIXED_GV2_OB_B48_CONSERVATIVE_OWNER_PROJECTION_BYTES: Final[int] = (
    WDL_FIXED_GV2_OB_B48_MEASURED_PREPARED_OWNER_PEAK_BYTES
)
WDL_FIXED_GV2_OB_B48_MINIMUM_RESIDUAL_HEADROOM_BYTES: Final[int] = 64 * MIB
WDL_FIXED_GV2_OB_B48_PREPARED_OWNER_RECEIPT_SHA256: Final[str] = (
    "c183ad6b9fc8c3f03f15e5783a9ea416224b5424ec5ad09d512714e063917e1e"
)
WDL_FIXED_GV2_OB_B48_REQUIRED_RUNTIME_ENVIRONMENT: Final[dict[str, str]] = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}
WDL_FIXED_GV2_OB_B24_BATCH_CELLS: Final[int] = 24
WDL_FIXED_GV2_OB_B24_MEASURED_PREPARED_OWNER_PEAK_BYTES: Final[int] = (
    2_589_982_720
)
WDL_FIXED_GV2_OB_B24_CONSERVATIVE_OWNER_PROJECTION_BYTES: Final[int] = (
    WDL_FIXED_GV2_OB_B24_MEASURED_PREPARED_OWNER_PEAK_BYTES + 128 * MIB
)
WDL_FIXED_GV2_OB_B24_PREPARED_OWNER_RECEIPT_SHA256: Final[str] = (
    "c367e2fed11b159b93ebb6fe2c4a9afc7515b460cb0c94e68cf3236e64f98e16"
)
W32_G1_CPU_STAGED_REQUESTS_B24_BATCH_CELLS: Final[int] = 24
W32_G1_CPU_STAGED_REQUESTS_MAX_BATCH_CELLS: Final[int] = 48
W32_G1_CPU_STAGED_REQUESTS_MAX_RESIDENT_CAPACITY: Final[int] = 512
W32_G1_CPU_STAGED_REQUESTS_MAX_TICKS_PER_GENERATION: Final[int] = 160
W32_G1_CPU_STAGED_REQUESTS_CPYTHON_GAUSS_SEGMENT_CAPACITY: Final[int] = 65_536
W32_G1_CPU_STAGED_REQUESTS_B24_MEASURED_OWNER_PEAK_BYTES: Final[int] = 958_398_464
W32_G1_CPU_STAGED_REQUESTS_B24_RECEIPT_SHA256: Final[str] = (
    "b1ff2a353fbd40e596f4b7516b78ff6609898c728e1dae05985908a74eb6a1d5"
)
W32_G1_CPU_STAGED_REQUESTS_PER_CELL_PEAK_BYTES: Final[int] = (
    (
        W32_G1_CPU_STAGED_REQUESTS_B24_MEASURED_OWNER_PEAK_BYTES
        + W32_G1_CPU_STAGED_REQUESTS_B24_BATCH_CELLS
        - 1
    )
    // W32_G1_CPU_STAGED_REQUESTS_B24_BATCH_CELLS
)
W32_G1_CPU_STAGED_REQUESTS_PROJECTION_MARGIN_BYTES: Final[int] = 512 * MIB

WDL_GRU_AUDITED_BATCH_CELLS: Final[int] = 1
WDL_GRU_AUDITED_SEGMENT_GENERATIONS: Final[int] = 32
WDL_GRU_MEASURED_RESERVED_BYTES_B1_G32: Final[int] = 117_440_512
WDL_GRU_VRAM_SAFETY_FACTOR: Final[int] = 2
WDL_GRU_MEASURED_SECONDS_B1_G32: Final[float] = 405.937623716
WDL_GRU_WALL_SAFETY_FACTOR: Final[int] = 3
WDL_GRU_EVIDENCE_ARTIFACT_SHA256: Final[str] = (
    "7b6b254ac2569a48f72b3d95c1dc4646608afd371eb33249ee16788204493408"
)
WDL_GRU_ADMISSION_BLOCKER: Final[str] = (
    "the retained WDL/stable-GRU resource inventory is B1/G32 only; "
    "larger physical batches or resident segments require a separate allocation audit"
)
PATCH_GRU_ADMISSION_BLOCKER: Final[str] = (
    "the conservative Patch/stable-GRU resource envelope is admitted for B1 "
    "and resident segments through G32 only; larger requests require a separate audit"
)
PATCH_GRU_MEASURED_RESERVED_BYTES_B1_G14: Final[int] = 463_470_592
PATCH_GRU_VRAM_SAFETY_FACTOR: Final[int] = 2
PATCH_GRU_GATE_SECONDS_B1_G14_FOUR_TRAJECTORIES: Final[float] = 902.974143751
PATCH_GRU_WALL_SAFETY_FACTOR: Final[int] = 3
GMNW_ADMITTED_BATCH_MIN: Final[int] = 1
GMNW_ADMITTED_BATCH_MAX: Final[int] = 3
GMNW_ADMITTED_SEGMENT_GENERATION_MIN: Final[int] = 4
GMNW_ADMITTED_SEGMENT_GENERATION_MAX: Final[int] = 9
GMNW_RESOURCE_ADMISSION_BLOCKER: Final[str] = (
    "GMNW resource admission is bounded to the inclusive B1-B3 and "
    "G4-G9 resident segment envelopes; wider planning requires a current live gate"
)


class ResourceEstimateError(ValueError):
    """A provider resource request lies outside the estimator's stated domain."""


def patch_resident_window_route_id_v1(
    *,
    physical_window_generations: int,
) -> str:
    """Return the explicit route identity for a physical resident-window size."""

    if physical_window_generations == 1:
        return PATCH_RESIDENT_WINDOW_G1_ROUTE_ID
    if physical_window_generations == 2:
        return PATCH_RESIDENT_WINDOW_G2_ROUTE_ID
    raise ResourceEstimateError(
        "physical_window_generations must be an integer in [1, 2]"
    )


def fixed_mlp_dense_policy_window_bytes_per_cell_v1() -> dict[str, Any]:
    """Return retained fixed-MLP policy/RNG window bytes for one cell."""

    return fixed_mlp_policy_window_bytes_per_cell_v1(
        physical_window_generations=PATCH_RESIDENT_WINDOW_GENERATIONS,
        cpython_gauss_segment_capacity=None,
    )


def fixed_mlp_policy_window_bytes_per_cell_v1(
    *,
    physical_window_generations: int,
    cpython_gauss_segment_capacity: int | None = None,
    policy_birth_segment_refill_enable: bool | None = None,
    policy_birth_segment_capacity: int | None = None,
    turnover_mlp_shadow_elide_enable: bool | None = None,
    cpu_request_staging_enable: bool | None = None,
) -> dict[str, Any]:
    """Return retained fixed-MLP policy/RNG bytes for one physical window."""

    if (
        type(physical_window_generations) is not int
        or not PATCH_MIN_RESIDENT_WINDOW_GENERATIONS
        <= physical_window_generations
        <= PATCH_MAX_RESIDENT_WINDOW_GENERATIONS
    ):
        raise ResourceEstimateError(
            "physical_window_generations must be an integer in [1, 2]"
        )
    if turnover_mlp_shadow_elide_enable is not None and (
        type(turnover_mlp_shadow_elide_enable) is not bool
    ):
        raise ResourceEstimateError(
            "turnover_mlp_shadow_elide_enable must be a bool"
        )
    if cpu_request_staging_enable is not None and (
        type(cpu_request_staging_enable) is not bool
    ):
        raise ResourceEstimateError("cpu_request_staging_enable must be a bool")
    physical_window_ticks = (
        physical_window_generations * PATCH_MAX_ADMITTED_TICKS_PER_GENERATION
    )
    physical_window_turnover_episodes = (
        physical_window_generations * PATCH_TURNOVER_EPISODES_PER_GENERATION
    )
    episodes = (
        physical_window_ticks * PATCH_MAX_BIRTHS_PER_TICK
        + physical_window_turnover_episodes
    )
    policy_segment_refill_enabled = False
    policy_resident_episode_capacity = episodes
    if policy_birth_segment_refill_enable is not None:
        if type(policy_birth_segment_refill_enable) is not bool:
            raise ResourceEstimateError(
                "policy_birth_segment_refill_enable must be a bool"
            )
        policy_segment_refill_enabled = policy_birth_segment_refill_enable
    if policy_birth_segment_capacity is not None:
        if (
            type(policy_birth_segment_capacity) is not int
            or (
                policy_birth_segment_capacity != -1
                and policy_birth_segment_capacity <= 0
            )
        ):
            raise ResourceEstimateError(
                "policy_birth_segment_capacity must be -1 or a positive integer"
            )
        policy_segment_refill_enabled = True
    if policy_segment_refill_enabled:
        minimum_policy_segment_capacity = max(
            PATCH_MAX_BIRTHS_PER_TICK,
            physical_window_turnover_episodes,
        )
        requested_policy_segment_capacity = (
            minimum_policy_segment_capacity
            if policy_birth_segment_capacity in (None, -1)
            else policy_birth_segment_capacity
        )
        if requested_policy_segment_capacity < minimum_policy_segment_capacity:
            raise ResourceEstimateError(
                "policy_birth_segment_capacity must cover one ordinary tick "
                "and one turnover refill segment"
            )
        policy_resident_episode_capacity = min(
            requested_policy_segment_capacity,
            episodes,
        )
    default_gauss_raw_capacity = episodes * PATCH_CPYTHON_GAUSS_RAW_CAPACITY_PER_BIRTH
    if cpython_gauss_segment_capacity is None:
        gaussian_raw_capacity = default_gauss_raw_capacity
        gaussian_segment_schema = "full_physical_window_gaussian_tape_v1"
    else:
        if (
            type(cpython_gauss_segment_capacity) is not int
            or cpython_gauss_segment_capacity < 0
        ):
            raise ResourceEstimateError(
                "cpython_gauss_segment_capacity must be a nonnegative integer"
            )
        if cpython_gauss_segment_capacity > default_gauss_raw_capacity:
            raise ResourceEstimateError(
                "cpython_gauss_segment_capacity exceeds the default physical-window horizon"
            )
        gaussian_raw_capacity = cpython_gauss_segment_capacity
        gaussian_segment_schema = "host_prepared_exact_gaussian_segment_v1"
    packed_crossover_row_bytes = (
        PATCH_FIXED_MLP_WIDTH + 7
    ) // 8
    rows = {
        "policy_birth_crossover_mask_packed_tape": (
            policy_resident_episode_capacity * packed_crossover_row_bytes
        ),
        "policy_birth_post_rng_state_packed_tape": (
            policy_resident_episode_capacity * PATCH_TORCH_RNG_PACKED_STATE_BYTES
        ),
        "python_random_gauss_primary_tape": (
            gaussian_raw_capacity * 8
        ),
        "python_random_gauss_secondary_tape": (
            gaussian_raw_capacity * 8
        ),
        "policy_birth_delta_tape": (
            policy_resident_episode_capacity * PATCH_POLICY_MUTATION_COMPACT_WIDTH * 4
        ),
        "policy_birth_mutation_mask_tape": (
            policy_resident_episode_capacity * PATCH_POLICY_MUTATION_COMPACT_WIDTH
        ),
        "policy_birth_semantic_draws_tape": policy_resident_episode_capacity * 8,
    }
    rows["total"] = sum(rows.values())
    rows["policy_birth_crossover_mask_encoding"] = PATCH_CROSSOVER_MASK_ENCODING
    rows["policy_birth_crossover_mask_logical_width"] = PATCH_FIXED_MLP_WIDTH
    rows["policy_birth_crossover_mask_packed_row_bytes"] = packed_crossover_row_bytes
    rows["policy_birth_crossover_mask_logical_dense_bytes"] = (
        episodes * PATCH_FIXED_MLP_WIDTH
    )
    rows["policy_birth_crossover_mask_physical_bytes"] = (
        policy_resident_episode_capacity * packed_crossover_row_bytes
    )
    rows["policy_birth_crossover_mask_dense_allocation_materialized"] = False
    rows["policy_birth_crossover_mask_max_logical_temp_bytes"] = PATCH_FIXED_MLP_WIDTH
    rows["policy_birth_post_rng_state_encoding"] = PATCH_TORCH_RNG_STATE_ENCODING
    rows["policy_birth_post_rng_state_logical_row_bytes"] = (
        PATCH_TORCH_RNG_STATE_BYTES
    )
    rows["policy_birth_post_rng_state_packed_row_bytes"] = (
        PATCH_TORCH_RNG_PACKED_STATE_BYTES
    )
    rows["policy_birth_post_rng_state_logical_dense_bytes"] = (
        episodes * PATCH_TORCH_RNG_STATE_BYTES
    )
    rows["policy_birth_post_rng_state_physical_bytes"] = (
        policy_resident_episode_capacity * PATCH_TORCH_RNG_PACKED_STATE_BYTES
    )
    rows["policy_birth_post_rng_state_dense_allocation_materialized"] = False
    rows["policy_birth_post_rng_state_abi"] = dict(PATCH_TORCH_RNG_STATE_ABI)
    rows["policy_birth_post_rng_state_max_logical_temp_bytes"] = (
        PATCH_TORCH_RNG_STATE_BYTES
    )
    rows["physical_window_generations"] = physical_window_generations
    rows["physical_window_ticks"] = physical_window_ticks
    rows["turnover_episodes"] = physical_window_turnover_episodes
    rows["policy_episode_capacity"] = episodes
    rows["policy_resident_episode_capacity"] = policy_resident_episode_capacity
    rows["policy_birth_segment_refill_enabled"] = policy_segment_refill_enabled
    rows["policy_birth_segment_capacity"] = policy_resident_episode_capacity
    rows["policy_birth_segment_boundary"] = (
        "native_birth_boundary_v1"
        if policy_segment_refill_enabled
        else "physical_window_boundary_v1"
    )
    rows["cpython_gauss_default_raw_capacity"] = default_gauss_raw_capacity
    rows["cpython_gauss_segment_schema"] = gaussian_segment_schema
    rows["cpython_gauss_segment_boundary"] = (
        "native_tick_boundary_v1"
        if cpython_gauss_segment_capacity is not None
        else "physical_window_boundary_v1"
    )
    rows["cpython_gauss_segment_capacity"] = gaussian_raw_capacity
    rows["cpython_gauss_active_segment_buffers"] = 1
    rows["cpython_gauss_staging_segment_buffers"] = 0
    rows["cpython_gauss_active_segment_bytes"] = gaussian_raw_capacity * 8 * 2
    rows["cpython_gauss_staging_segment_bytes"] = 0
    rows["cpython_gauss_full_window_tape_materialized"] = (
        cpython_gauss_segment_capacity is None
    )
    return rows


def planning_disk_upper_bound_v1(
    *,
    logical_trajectories: int,
    logical_generations: int,
    checkpoint_boundaries: int,
    checkpoint_bytes_per_trajectory_boundary: int = CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY,
) -> int:
    """Return the compiler's conservative serialized-output envelope."""

    for name, value in (
        ("logical_trajectories", logical_trajectories),
        ("logical_generations", logical_generations),
        ("checkpoint_boundaries", checkpoint_boundaries),
        (
            "checkpoint_bytes_per_trajectory_boundary",
            checkpoint_bytes_per_trajectory_boundary,
        ),
    ):
        if type(value) is not int or value < 1:
            raise ResourceEstimateError(f"{name} must be an integer >= 1")
    summaries = (
        logical_trajectories
        * logical_generations
        * SUMMARY_BYTES_PER_TRAJECTORY_GENERATION
    )
    checkpoints = (
        logical_trajectories
        * checkpoint_boundaries
        * checkpoint_bytes_per_trajectory_boundary
    )
    return summaries + checkpoints


def planning_disk_upper_bound_absolute_ticks_v1(
    *,
    logical_trajectories: int,
    total_ticks: int,
    checkpoint_boundaries: int,
    summary_bytes_per_tick: int = 32,
) -> int:
    """Return a tick-native output envelope without a generation projection.

    Resident allocation is deliberately absent from this calculation: the
    emitted history and checkpoint schedule may grow with the requested
    horizon, while the owner/event/source windows remain separately bounded.
    ``summary_bytes_per_tick`` is a planning constant, not an admission claim.
    """

    for name, value in (
        ("logical_trajectories", logical_trajectories),
        ("total_ticks", total_ticks),
        ("checkpoint_boundaries", checkpoint_boundaries),
        ("summary_bytes_per_tick", summary_bytes_per_tick),
    ):
        if type(value) is not int or value < 1:
            raise ResourceEstimateError(f"{name} must be an integer >= 1")
    summaries = logical_trajectories * total_ticks * summary_bytes_per_tick
    checkpoints = (
        logical_trajectories
        * checkpoint_boundaries
        * CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY
    )
    return summaries + checkpoints


def planning_acs_native_fixed_mlp_b136_checkpoint_upper_bound_v1(
    *,
    logical_cells: int,
    batch_cells: int,
    checkpoint_boundaries: int,
    comparison_trajectories: int = 0,
) -> dict[str, int | str]:
    """Plan the exact physical-B136 aggregate checkpoint envelope.

    The compact payload is the fixed-MLP projection after exact zero-fill
    elision. Aggregate header/footer maxima are charged per physical object;
    comparison trajectories, when present, retain the legacy bound. This
    helper is deliberately narrow and rejects arbitrary aggregate scaling.
    """

    for name, value in (
        ("logical_cells", logical_cells),
        ("batch_cells", batch_cells),
        ("checkpoint_boundaries", checkpoint_boundaries),
        ("comparison_trajectories", comparison_trajectories),
    ):
        if type(value) is not int or value < 0:
            raise ResourceEstimateError(f"{name} must be an integer >= 0")
    if logical_cells != ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS:
        raise ResourceEstimateError(
            "fixed-MLP aggregate projection is bound to logical B136"
        )
    if not 2 <= batch_cells <= logical_cells:
        raise ResourceEstimateError(
            "fixed-MLP aggregate batch_cells must be in [2, logical_cells]"
        )
    if checkpoint_boundaries < 1:
        raise ResourceEstimateError("checkpoint_boundaries must be >= 1")

    physical_batch_count = (
        logical_cells + batch_cells - 1
    ) // batch_cells
    active_payload_bytes = (
        ACS_NATIVE_FIXED_MLP_B136_ACTIVE_PAYLOAD_BYTES * logical_cells
        + ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS
        - 1
    ) // ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS
    metadata_bytes = (
        physical_batch_count
        * ACS_NATIVE_AGGREGATE_METADATA_MAX_BYTES_PER_OBJECT
    )
    comparison_bytes = (
        comparison_trajectories * CHECKPOINT_BYTES_PER_TRAJECTORY_BOUNDARY
    )
    checkpoint_bytes_per_boundary = (
        active_payload_bytes + metadata_bytes + comparison_bytes
    )
    if (
        batch_cells == ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS
        and comparison_trajectories == 0
        and checkpoint_bytes_per_boundary
        < ACS_NATIVE_FIXED_MLP_B136_MEASURED_AGGREGATE_BYTES
    ):
        raise ResourceEstimateError(
            "fixed-MLP aggregate projection is below measured physical B136"
        )
    return {
        "checkpoint_accounting_basis": ACS_NATIVE_FIXED_MLP_B136_ACCOUNTING_BASIS,
        "checkpoint_physical_payload_bytes_per_boundary": active_payload_bytes,
        "checkpoint_physical_metadata_bytes_per_boundary": metadata_bytes,
        "checkpoint_bytes_per_boundary_upper_bound": checkpoint_bytes_per_boundary,
        "checkpoint_bytes_upper_bound": (
            checkpoint_boundaries * checkpoint_bytes_per_boundary
        ),
        "checkpoint_physical_object_count_upper_bound": (
            checkpoint_boundaries * (physical_batch_count + comparison_trajectories)
        ),
        "checkpoint_physical_batch_count_upper_bound": physical_batch_count,
    }


def _is_acs_native_fixed_mlp_b136_resource_shape(
    *,
    batch_cells: Any,
    logical_generations: Any,
    logical_trajectories: Any,
    resident_segment_generations: Any,
    physical_window_generations: Any,
    ticks_per_generation: Any,
    resident_capacity: Any,
    ordinary_births_enabled: Any,
    genome_upgrade_v2_enable: Any,
    genome_upgrade_v2_mutation_rate_q10: Any,
    genome_upgrade_v2_mutation_step_q10: Any,
    cpython_gauss_segment_capacity: Any,
    policy_birth_segment_refill_enable: Any,
    policy_birth_segment_capacity: Any,
    turnover_mlp_shadow_elide_enable: Any,
    cpu_request_staging_enable: Any,
) -> bool:
    """Recognize only the measured physical-B136 resource identity."""

    return (
        type(batch_cells) is int
        and batch_cells == ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS
        and logical_generations == 21
        and logical_trajectories == ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS
        and resident_segment_generations == 7
        and physical_window_generations == 1
        and ticks_per_generation == 160
        and resident_capacity == 512
        and ordinary_births_enabled is True
        and genome_upgrade_v2_enable is True
        and genome_upgrade_v2_mutation_rate_q10 == 20
        and genome_upgrade_v2_mutation_step_q10 == 64
        and cpython_gauss_segment_capacity == 65_536
        and policy_birth_segment_refill_enable is True
        and policy_birth_segment_capacity == -1
        and turnover_mlp_shadow_elide_enable is False
        and cpu_request_staging_enable is False
    )


def estimate_patch_fixed_resident_resources_v1(
    *,
    batch_cells: int,
    logical_generations: int,
    logical_trajectories: int,
    checkpoint_boundaries: int,
    resident_segment_generations: int | None = None,
    physical_window_generations: int = PATCH_RESIDENT_WINDOW_GENERATIONS,
    ticks_per_generation: int = TICKS_PER_GENERATION,
    resident_capacity: int = PATCH_CAPACITY,
    ordinary_births_enabled: bool | None = None,
    genome_upgrade_v2_enable: bool | None = None,
    genome_upgrade_v2_mutation_rate_q10: int | None = None,
    genome_upgrade_v2_mutation_step_q10: int | None = None,
    cpython_gauss_segment_capacity: int | None = None,
    policy_birth_segment_refill_enable: bool | None = None,
    policy_birth_segment_capacity: int | None = None,
    turnover_mlp_shadow_elide_enable: bool | None = None,
    cpu_request_staging_enable: bool | None = None,
) -> dict[str, Any]:
    """Estimate the fixed-MLP Patch resident provider's resource envelope.

    Policy/RNG, drive-noise, event, stat, and bookkeeping storage is bounded by
    the requested one- or two-generation physical window.  Identity allocation
    remains sized to the logical generation horizon so continuation identities
    cannot alias when the window is reused.
    """

    if type(logical_generations) is not int or logical_generations < 3:
        raise ResourceEstimateError("logical_generations must be an integer >= 3")
    if resident_segment_generations is None:
        resident_segment_generations = logical_generations
    if (
        type(resident_segment_generations) is not int
        or not PATCH_MIN_LOGICAL_GENERATIONS
        <= resident_segment_generations
        <= PATCH_MAX_LOGICAL_GENERATIONS
    ):
        raise ResourceEstimateError(
            "resident_segment_generations must be an integer in [3, 128]"
        )
    if type(resident_capacity) is not int or resident_capacity < 1:
        raise ResourceEstimateError("resident_capacity must be an integer >= 1")
    if type(ticks_per_generation) is not int or ticks_per_generation < 1:
        raise ResourceEstimateError("ticks_per_generation must be an integer >= 1")
    if (
        type(physical_window_generations) is not int
        or not PATCH_MIN_RESIDENT_WINDOW_GENERATIONS
        <= physical_window_generations
        <= PATCH_MAX_RESIDENT_WINDOW_GENERATIONS
    ):
        raise ResourceEstimateError(
            "physical_window_generations must be an integer in [1, 2]"
        )
    if turnover_mlp_shadow_elide_enable is not None and (
        type(turnover_mlp_shadow_elide_enable) is not bool
    ):
        raise ResourceEstimateError(
            "turnover_mlp_shadow_elide_enable must be a bool"
        )
    if cpu_request_staging_enable is not None and (
        type(cpu_request_staging_enable) is not bool
    ):
        raise ResourceEstimateError("cpu_request_staging_enable must be a bool")

    physical_b136_route = _is_acs_native_fixed_mlp_b136_resource_shape(
        batch_cells=batch_cells,
        logical_generations=logical_generations,
        logical_trajectories=logical_trajectories,
        resident_segment_generations=resident_segment_generations,
        physical_window_generations=physical_window_generations,
        ticks_per_generation=ticks_per_generation,
        resident_capacity=resident_capacity,
        ordinary_births_enabled=ordinary_births_enabled,
        genome_upgrade_v2_enable=genome_upgrade_v2_enable,
        genome_upgrade_v2_mutation_rate_q10=genome_upgrade_v2_mutation_rate_q10,
        genome_upgrade_v2_mutation_step_q10=genome_upgrade_v2_mutation_step_q10,
        cpython_gauss_segment_capacity=cpython_gauss_segment_capacity,
        policy_birth_segment_refill_enable=policy_birth_segment_refill_enable,
        policy_birth_segment_capacity=policy_birth_segment_capacity,
        turnover_mlp_shadow_elide_enable=turnover_mlp_shadow_elide_enable,
        cpu_request_staging_enable=cpu_request_staging_enable,
    )
    if type(batch_cells) is not int or not (
        PATCH_MIN_BATCH_CELLS <= batch_cells <= PATCH_MAX_BATCH_CELLS
        or physical_b136_route
    ):
        raise ResourceEstimateError(
            "batch_cells must be an integer in [1, 72] or the exact "
            "audited ACS100X physical-B136 route"
        )

    window = physical_window_generations
    fixed_per_cell = (
        PATCH_STATE_AND_RECOVERY_BYTES_PER_CELL
        + PATCH_SCRATCH_BYTES_PER_CELL
        + LIFECYCLE_SCRATCH_BYTES_PER_CELL
    )
    policy_window = fixed_mlp_policy_window_bytes_per_cell_v1(
        physical_window_generations=window,
        cpython_gauss_segment_capacity=cpython_gauss_segment_capacity,
        policy_birth_segment_refill_enable=policy_birth_segment_refill_enable,
        policy_birth_segment_capacity=policy_birth_segment_capacity,
        turnover_mlp_shadow_elide_enable=turnover_mlp_shadow_elide_enable,
        cpu_request_staging_enable=cpu_request_staging_enable,
    )
    route_identity = patch_resident_window_route_id_v1(
        physical_window_generations=window
    )
    physical_window_per_cell = (
        policy_window["total"]
        + resident_capacity * PATCH_FIXED_MLP_WIDTH * 4
        + resident_capacity * PATCH_FIXED_MLP_HIDDEN_SIZE * 4
        + window
        * ticks_per_generation
        * DRIVE_NOISE_CHANNEL_COUNT
        * resident_capacity
        * 4
        + (EVENT_BASE_CAPACITY_ROWS + window * EVENT_ROWS_PER_GENERATION)
        * EVENT_ROW_STORAGE_BYTES
        + window * ticks_per_generation * PATCH_STAT_COLUMNS * 8
        + window
        * ticks_per_generation
        * NONCOMPACT_BOOKKEEPING_SCALARS_PER_TICK
        * 8
        + PATCH_DENSE_OWNER_ALLOCATOR_MARGIN_BYTES_PER_CELL
    )
    logical_identity_per_cell = (
        resident_segment_generations
        * IDENTITY_GROWTH_ROWS_PER_GENERATION
        * IDENTITY_ROW_STORAGE_BYTES
    )
    per_cell = fixed_per_cell + physical_window_per_cell + logical_identity_per_cell
    vram_projection = SHARED_CONTROLLER_ALLOWANCE_BYTES + batch_cells * per_cell
    vram_bound: int | str = (
        vram_projection
        if batch_cells <= PATCH_AUDITED_VRAM_BATCH_MAX
        else "unavailable"
    )
    measured_gv2_no_birth_b48 = (
        batch_cells == GENOME_V2_NO_BIRTH_B48_BATCH_CELLS
        and resident_capacity == GENOME_V2_NO_BIRTH_B48_RESIDENT_CAPACITY
        and resident_segment_generations
        <= GENOME_V2_NO_BIRTH_B48_SEGMENT_GENERATIONS_MAX
        and ordinary_births_enabled is False
        and genome_upgrade_v2_enable is True
        and genome_upgrade_v2_mutation_rate_q10 == 0
        and genome_upgrade_v2_mutation_step_q10 == 0
    )
    measured_w32_g1_cpu_staged_requests = (
        batch_cells <= W32_G1_CPU_STAGED_REQUESTS_MAX_BATCH_CELLS
        and window == 1
        and resident_capacity <= W32_G1_CPU_STAGED_REQUESTS_MAX_RESIDENT_CAPACITY
        and ticks_per_generation <= W32_G1_CPU_STAGED_REQUESTS_MAX_TICKS_PER_GENERATION
        and cpython_gauss_segment_capacity
        == W32_G1_CPU_STAGED_REQUESTS_CPYTHON_GAUSS_SEGMENT_CAPACITY
        and policy_birth_segment_refill_enable is True
        and policy_birth_segment_capacity == -1
        and turnover_mlp_shadow_elide_enable is True
        and cpu_request_staging_enable is True
    )
    estimator_id = f"patch-fixed-resident-window{window}-v1"
    estimator_provenance = (
        "engine-owned extraction of the audited conservative Patch P2 B2/G128 "
        "resident memory declaration; no tool-script import"
    )
    if measured_gv2_no_birth_b48:
        vram_projection = GENOME_V2_NO_BIRTH_B48_CONSERVATIVE_OWNER_PROJECTION_BYTES
        vram_bound = vram_projection
        estimator_id = "patch-gv2-no-birth-fixed-mlp-b48-prepared-owner-v1"
        estimator_provenance = (
            "B48 prepared-owner allocation receipt with frozen Genome V2, "
            "ordinary births disabled, and compact turnover-only event/policy "
            "storage; conservative projection adds 384 MiB to measured peak"
        )
    elif measured_w32_g1_cpu_staged_requests:
        vram_projection = (
            batch_cells * W32_G1_CPU_STAGED_REQUESTS_PER_CELL_PEAK_BYTES
            + W32_G1_CPU_STAGED_REQUESTS_PROJECTION_MARGIN_BYTES
        )
        vram_bound = vram_projection
        estimator_id = "patch-w32-g1-cpu-staged-requests-b48-prepared-owner-v1"
        estimator_provenance = (
            "B24 prepared-owner allocation receipt for exact W32/g1 bounded "
            "Gaussian, policy-segment, turnover-shadow-elided CPU-staged "
            "request route; B48 admission scales the measured per-cell peak "
            "and adds 512 MiB margin pending physical B48 confirmation"
        )
    elif physical_b136_route:
        vram_projection = ACS_NATIVE_FIXED_MLP_B136_TRANSIENT_VRAM_BYTES
        vram_bound = vram_projection
        estimator_id = ACS_NATIVE_FIXED_MLP_B136_RESOURCE_ESTIMATOR_ID
        estimator_provenance = (
            "prior physical-B136/G7 allocation projection; current event-owner "
            "repair still requires a targeted runtime gate before qualification"
        )
        per_cell = (
            vram_projection - SHARED_CONTROLLER_ALLOWANCE_BYTES + batch_cells - 1
        ) // batch_cells
    disk = planning_disk_upper_bound_v1(
        logical_trajectories=logical_trajectories,
        logical_generations=logical_generations,
        checkpoint_boundaries=checkpoint_boundaries,
    )
    wall_projection = int(
        logical_trajectories
        * logical_generations
        * PATCH_GATE_SECONDS_PER_CELL_GENERATION
        * PATCH_WALL_PROJECTION_SAFETY_FACTOR
        + 0.999999
    )
    return {
        "authority": "provider_resource_admission_estimate_v1",
        "vram_bytes_upper_bound": vram_bound,
        "vram_bytes_projection": vram_projection,
        "vram_bytes_unavailable_reason": (
            None if isinstance(vram_bound, int) else VRAM_BOUND_BLOCKER
        ),
        "disk_bytes_upper_bound": disk,
        "wall_seconds_upper_bound": wall_projection,
        "wall_seconds_projection": wall_projection,
        "wall_seconds_bound_authority": "estimated_conservative_not_audited",
        "wall_seconds_unavailable_reason": None,
        "resource_estimator_id": estimator_id,
        "resource_estimator_provenance": estimator_provenance,
        "physical_window_route_identity": route_identity,
        "resource_estimator_assumptions": {
            "logical_generation_range": [3, 128],
            "preparation_batch_range": (
                [
                    ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS,
                    ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS,
                ]
                if physical_b136_route
                else [1, 72]
            ),
            "admitted_batch_range": (
                [
                    ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS,
                    ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS,
                ]
                if physical_b136_route
                else (
                    [GENOME_V2_NO_BIRTH_B48_BATCH_CELLS, GENOME_V2_NO_BIRTH_B48_BATCH_CELLS]
                    if measured_gv2_no_birth_b48
                    else (
                        [1, W32_G1_CPU_STAGED_REQUESTS_MAX_BATCH_CELLS]
                        if measured_w32_g1_cpu_staged_requests
                        else [1, PATCH_AUDITED_VRAM_BATCH_MAX]
                    )
                )
            ),
            "audited_vram_batch_range": (
                [
                    ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS,
                    ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS,
                ]
                if physical_b136_route
                else [1, PATCH_AUDITED_VRAM_BATCH_MAX]
            ),
            "acs_native_physical_b136_route": physical_b136_route,
            "acs_native_physical_b136_active_peak_bytes": (
                ACS_NATIVE_FIXED_MLP_B136_ACTIVE_VRAM_BYTES
                if physical_b136_route
                else None
            ),
            "acs_native_physical_b136_transient_peak_bytes": (
                ACS_NATIVE_FIXED_MLP_B136_TRANSIENT_VRAM_BYTES
                if physical_b136_route
                else None
            ),
            "acs_native_physical_b136_reserve_bytes": (
                ACS_NATIVE_FIXED_MLP_B136_RESERVE_BYTES
                if physical_b136_route
                else None
            ),
            "resident_capacity": resident_capacity,
            "w32_g1_cpu_staged_requests_max_resident_capacity": (
                W32_G1_CPU_STAGED_REQUESTS_MAX_RESIDENT_CAPACITY
                if measured_w32_g1_cpu_staged_requests
                else None
            ),
            "w32_g1_cpu_staged_requests_max_ticks_per_generation": (
                W32_G1_CPU_STAGED_REQUESTS_MAX_TICKS_PER_GENERATION
                if measured_w32_g1_cpu_staged_requests
                else None
            ),
            "ticks_per_generation": ticks_per_generation,
            "w32_g1_cpu_staged_requests_b24_prepared_owner_receipt_sha256": (
                W32_G1_CPU_STAGED_REQUESTS_B24_RECEIPT_SHA256
                if measured_w32_g1_cpu_staged_requests
                else None
            ),
            "w32_g1_cpu_staged_requests_b24_measured_owner_peak_bytes": (
                W32_G1_CPU_STAGED_REQUESTS_B24_MEASURED_OWNER_PEAK_BYTES
                if measured_w32_g1_cpu_staged_requests
                else None
            ),
            "w32_g1_cpu_staged_requests_per_cell_peak_bytes": (
                W32_G1_CPU_STAGED_REQUESTS_PER_CELL_PEAK_BYTES
                if measured_w32_g1_cpu_staged_requests
                else None
            ),
            "w32_g1_cpu_staged_requests_projection_margin_bytes": (
                W32_G1_CPU_STAGED_REQUESTS_PROJECTION_MARGIN_BYTES
                if measured_w32_g1_cpu_staged_requests
                else None
            ),
            "gv2_no_birth_b48_prepared_owner_receipt_sha256": (
                GENOME_V2_NO_BIRTH_B48_RECEIPT_SHA256
                if measured_gv2_no_birth_b48
                else None
            ),
            "gv2_no_birth_b48_measured_owner_peak_bytes": (
                GENOME_V2_NO_BIRTH_B48_MEASURED_OWNER_PEAK_BYTES
                if measured_gv2_no_birth_b48
                else None
            ),
            "gv2_no_birth_b48_conservative_margin_bytes": (
                384 * MIB if measured_gv2_no_birth_b48 else None
            ),
            "gv2_no_birth_b48_admitted_resident_segment_generations": (
                [3, GENOME_V2_NO_BIRTH_B48_SEGMENT_GENERATIONS_MAX]
                if measured_gv2_no_birth_b48
                else None
            ),
            "physical_window_generations": window,
            "physical_window_route_identity": route_identity,
            "physical_window_scoped": [
                "policy_rng_tape",
                "packed_policy_crossover_mask_tape",
                "dense_policy_post_torch_rng_state_tape",
                "dense_cpython_gaussian_tapes",
                "drive_noise",
                "event_storage",
                "patch_stats",
                "runtime_bookkeeping",
            ],
            "logical_horizon_scoped": ["identity_growth"],
            "shared_controller_allowance_bytes": SHARED_CONTROLLER_ALLOWANCE_BYTES,
            "policy_window": {
                "policy_episode_capacity": policy_window["policy_episode_capacity"],
                "policy_resident_episode_capacity": policy_window[
                    "policy_resident_episode_capacity"
                ],
                "policy_birth_segment_refill_enabled": policy_window[
                    "policy_birth_segment_refill_enabled"
                ],
                "policy_birth_segment_capacity": policy_window[
                    "policy_birth_segment_capacity"
                ],
                "policy_birth_segment_boundary": policy_window[
                    "policy_birth_segment_boundary"
                ],
                "max_births_per_tick": PATCH_MAX_BIRTHS_PER_TICK,
                "max_admitted_ticks_per_generation": (
                    PATCH_MAX_ADMITTED_TICKS_PER_GENERATION
                ),
                "physical_window_ticks": policy_window["physical_window_ticks"],
                "turnover_episodes": policy_window["turnover_episodes"],
                "mlp_pack_width": PATCH_FIXED_MLP_WIDTH,
                "crossover_mask_encoding": PATCH_CROSSOVER_MASK_ENCODING,
                "crossover_mask_packed_row_bytes": (
                    (PATCH_FIXED_MLP_WIDTH + 7) // 8
                ),
                "crossover_mask_dense_allocation_materialized": False,
                "crossover_mask_max_logical_temp_bytes": PATCH_FIXED_MLP_WIDTH,
                "torch_rng_state_bytes": PATCH_TORCH_RNG_STATE_BYTES,
                "torch_rng_state_encoding": PATCH_TORCH_RNG_STATE_ENCODING,
                "torch_rng_packed_state_bytes": (
                    PATCH_TORCH_RNG_PACKED_STATE_BYTES
                ),
                "torch_rng_dense_post_state_allocation_materialized": False,
                "torch_rng_max_logical_temp_bytes": PATCH_TORCH_RNG_STATE_BYTES,
                "torch_rng_state_abi": dict(PATCH_TORCH_RNG_STATE_ABI),
                "cpython_gauss_segment_capacity_per_birth": (
                    PATCH_CPYTHON_GAUSS_RAW_CAPACITY_PER_BIRTH
                ),
                "cpython_gauss_default_raw_capacity": (
                    policy_window["cpython_gauss_default_raw_capacity"]
                ),
                "cpython_gauss_segment_capacity": (
                    policy_window["cpython_gauss_segment_capacity"]
                ),
                "cpython_gauss_segment_schema": (
                    policy_window["cpython_gauss_segment_schema"]
                ),
                "cpython_gauss_segment_boundary": (
                    policy_window["cpython_gauss_segment_boundary"]
                ),
                "cpython_gauss_segment_override_enabled": (
                    cpython_gauss_segment_capacity is not None
                ),
                "cpython_gauss_segment_buffering": {
                    "active_buffers": policy_window[
                        "cpython_gauss_active_segment_buffers"
                    ],
                    "staging_buffers": policy_window[
                        "cpython_gauss_staging_segment_buffers"
                    ],
                    "active_bytes_per_cell": policy_window[
                        "cpython_gauss_active_segment_bytes"
                    ],
                    "staging_bytes_per_cell": policy_window[
                        "cpython_gauss_staging_segment_bytes"
                    ],
                    "full_window_tape_materialized": policy_window[
                        "cpython_gauss_full_window_tape_materialized"
                    ],
                },
                "allocator_margin_bytes_per_cell": (
                    PATCH_DENSE_OWNER_ALLOCATOR_MARGIN_BYTES_PER_CELL
                ),
                "turnover_mlp_shadow_elide_enabled": (
                    turnover_mlp_shadow_elide_enable is True
                ),
                "cpu_request_staging_enabled": cpu_request_staging_enable is True,
                "bytes_per_cell": policy_window,
            },
            "disk_formula": (
                "trajectories * generations * 4096 + trajectories * "
                "checkpoint_boundaries * 64MiB"
            ),
            "wall_projection": {
                "authority": "estimated_not_audited",
                "measured_input": (
                    "gate19 primary B2G14 elapsed 81.518446 seconds for "
                    "28 cell-generations"
                ),
                "measured_seconds_per_cell_generation": (
                    PATCH_GATE_SECONDS_PER_CELL_GENERATION
                ),
                "safety_factor": PATCH_WALL_PROJECTION_SAFETY_FACTOR,
                "formula": (
                    "logical_trajectories * logical_generations * "
                    "measured_seconds_per_cell_generation * safety_factor"
                ),
            },
            "wall_bound": WALL_BOUND_BLOCKER,
        },
        "physical_batch_cells": batch_cells,
        "logical_trajectories": logical_trajectories,
        "logical_generations": logical_generations,
        "ticks_per_generation": ticks_per_generation,
        "resident_segment_generations": resident_segment_generations,
        "physical_storage_generations": window,
        "per_cell_vram_bytes_upper_bound": per_cell,
    }


def estimate_patch_wdl_stable_gru_resident_resources_v1(
    *,
    batch_cells: int,
    logical_generations: int,
    logical_trajectories: int,
    checkpoint_boundaries: int,
    resident_segment_generations: int,
    ticks_per_generation: int = TICKS_PER_GENERATION,
) -> dict[str, Any]:
    """Project resources from the retained exact B1/G32 WDL+GRU run.

    B1 with resident segments no longer than G32 is the only resource-admitted
    envelope.  The wider profile range remains useful for planning, but this
    function deliberately returns unavailable bounds outside the measured
    envelope instead of extrapolating B1 tensor ownership to larger batches.
    """

    if (
        type(batch_cells) is not int
        or not PATCH_MIN_BATCH_CELLS <= batch_cells <= PATCH_MAX_BATCH_CELLS
    ):
        raise ResourceEstimateError("batch_cells must be an integer in [1, 72]")
    for name, value in (
        ("logical_generations", logical_generations),
        ("logical_trajectories", logical_trajectories),
        ("checkpoint_boundaries", checkpoint_boundaries),
        ("resident_segment_generations", resident_segment_generations),
        ("ticks_per_generation", ticks_per_generation),
    ):
        if type(value) is not int or value < 1:
            raise ResourceEstimateError(f"{name} must be an integer >= 1")

    within_audited_envelope = (
        batch_cells == WDL_GRU_AUDITED_BATCH_CELLS
        and resident_segment_generations <= WDL_GRU_AUDITED_SEGMENT_GENERATIONS
    )
    vram_projection = (
        WDL_GRU_MEASURED_RESERVED_BYTES_B1_G32 * WDL_GRU_VRAM_SAFETY_FACTOR
    )
    seconds_per_cell_generation = (
        WDL_GRU_MEASURED_SECONDS_B1_G32 / WDL_GRU_AUDITED_SEGMENT_GENERATIONS
    )
    wall_projection = int(
        logical_trajectories
        * logical_generations
        * seconds_per_cell_generation
        * WDL_GRU_WALL_SAFETY_FACTOR
        + 0.999999
    )
    disk = planning_disk_upper_bound_v1(
        logical_trajectories=logical_trajectories,
        logical_generations=logical_generations,
        checkpoint_boundaries=checkpoint_boundaries,
    )
    return {
        "authority": "provider_resource_admission_estimate_v1",
        "vram_bytes_upper_bound": (
            vram_projection if within_audited_envelope else "unavailable"
        ),
        "vram_bytes_projection": vram_projection,
        "vram_bytes_unavailable_reason": (
            None if within_audited_envelope else WDL_GRU_ADMISSION_BLOCKER
        ),
        "disk_bytes_upper_bound": disk,
        "wall_seconds_upper_bound": (
            wall_projection if within_audited_envelope else "unavailable"
        ),
        "wall_seconds_projection": wall_projection,
        "wall_seconds_bound_authority": "estimated_conservative_not_audited",
        "wall_seconds_unavailable_reason": (
            None if within_audited_envelope else WDL_GRU_ADMISSION_BLOCKER
        ),
        "resource_estimator_id": "patch-wdl-stable-gru-b1-g32-v1",
        "resource_estimator_provenance": (
            "retained exact WDL/stable-GRU B1/G32 CUDA resource inventory and "
            "wall-time result for artifact " + WDL_GRU_EVIDENCE_ARTIFACT_SHA256
        ),
        "resource_estimator_assumptions": {
            "preparation_batch_range": [1, PATCH_MAX_BATCH_CELLS],
            "admitted_batch_range": [1, WDL_GRU_AUDITED_BATCH_CELLS],
            "admitted_resident_segment_generations": [
                1,
                WDL_GRU_AUDITED_SEGMENT_GENERATIONS,
            ],
            "measured_reserved_bytes_b1_g32": WDL_GRU_MEASURED_RESERVED_BYTES_B1_G32,
            "vram_safety_factor": WDL_GRU_VRAM_SAFETY_FACTOR,
            "measured_seconds_b1_g32": WDL_GRU_MEASURED_SECONDS_B1_G32,
            "wall_safety_factor": WDL_GRU_WALL_SAFETY_FACTOR,
            "wall_projection": (
                "logical trajectories * logical generations * measured B1/G32 "
                "seconds per generation * safety factor"
            ),
            "evidence_artifact_sha256": WDL_GRU_EVIDENCE_ARTIFACT_SHA256,
        },
        "physical_batch_cells": batch_cells,
        "logical_trajectories": logical_trajectories,
        "logical_generations": logical_generations,
        "ticks_per_generation": ticks_per_generation,
        "resident_segment_generations": resident_segment_generations,
    }


def estimate_patch_stable_gru_resident_resources_v1(
    *,
    batch_cells: int,
    logical_generations: int,
    logical_trajectories: int,
    checkpoint_boundaries: int,
    resident_segment_generations: int,
    ticks_per_generation: int = TICKS_PER_GENERATION,
) -> dict[str, Any]:
    """Admit Patch/GRU under the qualified WDL/GRU B1/G32 upper envelope.

    Patch/GRU uses the same controller artifact and resident GRU state while
    omitting WDL observation/state fields.  The WDL measurement is therefore a
    conservative bound for the narrower execution profile, not a claim that
    the two profiles share qualification identity.
    """

    estimate = estimate_patch_wdl_stable_gru_resident_resources_v1(
        batch_cells=batch_cells,
        logical_generations=logical_generations,
        logical_trajectories=logical_trajectories,
        checkpoint_boundaries=checkpoint_boundaries,
        resident_segment_generations=resident_segment_generations,
        ticks_per_generation=ticks_per_generation,
    )
    within_envelope = batch_cells == 1 and resident_segment_generations <= 32
    vram_bound = (
        PATCH_GRU_MEASURED_RESERVED_BYTES_B1_G14 * PATCH_GRU_VRAM_SAFETY_FACTOR
    )
    measured_seconds_per_trajectory_generation = (
        PATCH_GRU_GATE_SECONDS_B1_G14_FOUR_TRAJECTORIES / (4 * 14)
    )
    wall_bound = int(
        logical_trajectories
        * logical_generations
        * measured_seconds_per_trajectory_generation
        * PATCH_GRU_WALL_SAFETY_FACTOR
        + 0.999999
    )
    estimate.update(
        {
            "vram_bytes_upper_bound": vram_bound if within_envelope else "unavailable",
            "vram_bytes_projection": vram_bound,
            "vram_bytes_unavailable_reason": (
                None if within_envelope else PATCH_GRU_ADMISSION_BLOCKER
            ),
            "wall_seconds_upper_bound": wall_bound if within_envelope else "unavailable",
            "wall_seconds_projection": wall_bound,
            "wall_seconds_unavailable_reason": (
                None if within_envelope else PATCH_GRU_ADMISSION_BLOCKER
            ),
            "resource_estimator_id": "patch-stable-gru-b1-g32-gate36-v1",
            "resource_estimator_provenance": (
                "current-artifact Patch/stable-GRU B1/G14 gate36 peak CUDA inventory "
                "and elapsed time, with WDL B1/G32 evidence limiting segment admission"
            ),
        }
    )
    assumptions = dict(estimate["resource_estimator_assumptions"])
    assumptions["dominance_argument"] = (
        "Patch/stable-GRU is a strict tensor and mechanics subset of the measured "
        "WDL/stable-GRU execution for the same resident controller artifact"
    )
    assumptions.update(
        {
            "measured_reserved_bytes_b1_g14": PATCH_GRU_MEASURED_RESERVED_BYTES_B1_G14,
            "vram_safety_factor": PATCH_GRU_VRAM_SAFETY_FACTOR,
            "measured_gate_seconds_b1_g14_four_trajectories": (
                PATCH_GRU_GATE_SECONDS_B1_G14_FOUR_TRAJECTORIES
            ),
            "wall_safety_factor": PATCH_GRU_WALL_SAFETY_FACTOR,
            "evidence_run_root": (
                public_runtime_path('evidence_raw/eiel_declarative_campaign_20260722/36_patch_gru_b1g14_exact_gate')
            ),
        }
    )
    estimate["resource_estimator_assumptions"] = assumptions
    return estimate


def estimate_patch_wdl_fixed_mlp_resident_resources_v1(
    *,
    batch_cells: int,
    logical_generations: int,
    logical_trajectories: int,
    checkpoint_boundaries: int,
    resident_segment_generations: int,
    physical_window_generations: int = PATCH_RESIDENT_WINDOW_GENERATIONS,
    ticks_per_generation: int = TICKS_PER_GENERATION,
    resident_capacity: int = PATCH_CAPACITY,
    ordinary_births_enabled: bool | None = None,
    genome_upgrade_v2_enable: bool | None = None,
    genome_upgrade_v2_mutation_rate_q10: int | None = None,
    genome_upgrade_v2_mutation_step_q10: int | None = None,
    cpython_gauss_segment_capacity: int | None = None,
    policy_birth_segment_refill_enable: bool | None = None,
    policy_birth_segment_capacity: int | None = None,
    turnover_mlp_shadow_elide_enable: bool | None = None,
    cpu_request_staging_enable: bool | None = None,
) -> dict[str, Any]:
    """Admit WDL/fixed-MLP beneath the measured GRU profile envelopes."""

    if _is_acs_native_fixed_mlp_b136_resource_shape(
        batch_cells=batch_cells,
        logical_generations=logical_generations,
        logical_trajectories=logical_trajectories,
        resident_segment_generations=resident_segment_generations,
        physical_window_generations=physical_window_generations,
        ticks_per_generation=ticks_per_generation,
        resident_capacity=resident_capacity,
        ordinary_births_enabled=ordinary_births_enabled,
        genome_upgrade_v2_enable=genome_upgrade_v2_enable,
        genome_upgrade_v2_mutation_rate_q10=genome_upgrade_v2_mutation_rate_q10,
        genome_upgrade_v2_mutation_step_q10=genome_upgrade_v2_mutation_step_q10,
        cpython_gauss_segment_capacity=cpython_gauss_segment_capacity,
        policy_birth_segment_refill_enable=policy_birth_segment_refill_enable,
        policy_birth_segment_capacity=policy_birth_segment_capacity,
        turnover_mlp_shadow_elide_enable=turnover_mlp_shadow_elide_enable,
        cpu_request_staging_enable=cpu_request_staging_enable,
    ):
        estimate = estimate_patch_fixed_resident_resources_v1(
            batch_cells=batch_cells,
            logical_generations=logical_generations,
            logical_trajectories=logical_trajectories,
            checkpoint_boundaries=checkpoint_boundaries,
            resident_segment_generations=resident_segment_generations,
            physical_window_generations=physical_window_generations,
            ticks_per_generation=ticks_per_generation,
            resident_capacity=resident_capacity,
            ordinary_births_enabled=ordinary_births_enabled,
            genome_upgrade_v2_enable=genome_upgrade_v2_enable,
            genome_upgrade_v2_mutation_rate_q10=genome_upgrade_v2_mutation_rate_q10,
            genome_upgrade_v2_mutation_step_q10=genome_upgrade_v2_mutation_step_q10,
            cpython_gauss_segment_capacity=cpython_gauss_segment_capacity,
            policy_birth_segment_refill_enable=policy_birth_segment_refill_enable,
            policy_birth_segment_capacity=policy_birth_segment_capacity,
            turnover_mlp_shadow_elide_enable=turnover_mlp_shadow_elide_enable,
            cpu_request_staging_enable=cpu_request_staging_enable,
        )
        assumptions = dict(estimate["resource_estimator_assumptions"])
        assumptions["wdl_fixed_mlp_exact_physical_b136_route"] = True
        assumptions["wdl_state_is_included_in_prior_b136_peak"] = True
        estimate["resource_estimator_assumptions"] = assumptions
        return estimate

    estimate = estimate_patch_stable_gru_resident_resources_v1(
        batch_cells=batch_cells,
        logical_generations=logical_generations,
        logical_trajectories=logical_trajectories,
        checkpoint_boundaries=checkpoint_boundaries,
        resident_segment_generations=resident_segment_generations,
        ticks_per_generation=ticks_per_generation,
    )
    fixed_projection = estimate_patch_fixed_resident_resources_v1(
        batch_cells=batch_cells,
        logical_generations=logical_generations,
        logical_trajectories=logical_trajectories,
        checkpoint_boundaries=checkpoint_boundaries,
        resident_segment_generations=resident_segment_generations,
        physical_window_generations=physical_window_generations,
        ticks_per_generation=ticks_per_generation,
        resident_capacity=resident_capacity,
        ordinary_births_enabled=ordinary_births_enabled,
        genome_upgrade_v2_enable=genome_upgrade_v2_enable,
        genome_upgrade_v2_mutation_rate_q10=genome_upgrade_v2_mutation_rate_q10,
        genome_upgrade_v2_mutation_step_q10=genome_upgrade_v2_mutation_step_q10,
        cpython_gauss_segment_capacity=cpython_gauss_segment_capacity,
        policy_birth_segment_refill_enable=policy_birth_segment_refill_enable,
        policy_birth_segment_capacity=policy_birth_segment_capacity,
        turnover_mlp_shadow_elide_enable=turnover_mlp_shadow_elide_enable,
        cpu_request_staging_enable=cpu_request_staging_enable,
    )
    measured_wdl_gv2_ob_b48 = (
        batch_cells == WDL_FIXED_GV2_OB_B48_BATCH_CELLS
        and logical_generations == WDL_FIXED_GV2_OB_B48_LOGICAL_GENERATIONS
        and logical_trajectories == WDL_FIXED_GV2_OB_B48_LOGICAL_TRAJECTORIES
        and checkpoint_boundaries == WDL_FIXED_GV2_OB_B48_CHECKPOINT_BOUNDARIES
        and resident_capacity == WDL_FIXED_GV2_OB_B48_RESIDENT_CAPACITY
        and resident_segment_generations
        <= WDL_FIXED_GV2_OB_B48_SEGMENT_GENERATIONS_MAX
        and ordinary_births_enabled is True
        and genome_upgrade_v2_enable is True
    )
    measured_wdl_gv2_ob_b24 = (
        batch_cells == WDL_FIXED_GV2_OB_B24_BATCH_CELLS
        and logical_generations <= WDL_FIXED_GV2_OB_B48_LOGICAL_GENERATIONS
        and resident_capacity == WDL_FIXED_GV2_OB_B48_RESIDENT_CAPACITY
        and resident_segment_generations
        <= WDL_FIXED_GV2_OB_B48_SEGMENT_GENERATIONS_MAX
        and ordinary_births_enabled is True
        and genome_upgrade_v2_enable is True
    )
    multicell_projection = batch_cells > 1
    if multicell_projection:
        estimate["vram_bytes_projection"] = max(
            int(estimate["vram_bytes_projection"]),
            int(fixed_projection["vram_bytes_projection"]),
        )
        estimate["vram_bytes_upper_bound"] = "unavailable"
        estimate["vram_bytes_unavailable_reason"] = (
            "WDL/fixed-MLP multicell VRAM is a conservative Patch/fixed and "
            "retained recurrent-envelope projection until exact prepared-owner "
            "measurement validates the selected artifact"
        )
        estimate["wall_seconds_projection"] = max(
            int(estimate["wall_seconds_projection"]),
            int(fixed_projection["wall_seconds_projection"]),
        )
        estimate["wall_seconds_upper_bound"] = estimate["wall_seconds_projection"]
        estimate["wall_seconds_unavailable_reason"] = None
    if measured_wdl_gv2_ob_b48:
        estimate["vram_bytes_projection"] = (
            WDL_FIXED_GV2_OB_B48_CONSERVATIVE_OWNER_PROJECTION_BYTES
        )
        estimate["vram_bytes_upper_bound"] = "unavailable"
        estimate["vram_bytes_unavailable_reason"] = (
            "WDL/fixed-MLP Genome V2 ordinary-birth B48 has measured "
            "prepared-owner physical admission, but still requires exact "
            "prepared-owner receipt promotion and a runtime technical gate"
        )
    elif measured_wdl_gv2_ob_b24:
        estimate["vram_bytes_projection"] = (
            WDL_FIXED_GV2_OB_B24_CONSERVATIVE_OWNER_PROJECTION_BYTES
        )
        estimate["vram_bytes_upper_bound"] = estimate["vram_bytes_projection"]
        estimate["vram_bytes_unavailable_reason"] = None
    estimate["resource_estimator_id"] = "patch-wdl-fixed-mlp-b1-g32-v1"
    estimate["resource_estimator_provenance"] = (
        "conservative maximum of current-artifact Patch/stable-GRU peak memory "
        "and retained WDL/stable-GRU B1/G32 evidence; fixed MLP omits resident GRU state"
    )
    if measured_wdl_gv2_ob_b48:
        estimate["resource_estimator_id"] = (
            "patch-wdl-fixed-mlp-gv2-ob-b48-prepared-owner-candidate-v1"
        )
        estimate["resource_estimator_provenance"] = (
            "B48 WDL/fixed-MLP Genome V2 ordinary-birth prepared-owner "
            "measurement after compact policy tape and physical-window event "
            "capacity repair; requires explicit allocator binding, repeated "
            "prepared-owner confirmation, and runtime technical-gate promotion"
        )
    elif measured_wdl_gv2_ob_b24:
        estimate["resource_estimator_id"] = (
            "patch-wdl-fixed-mlp-gv2-ob-b24-prepared-owner-v1"
        )
        estimate["resource_estimator_provenance"] = (
            "B24 WDL/fixed-MLP Genome V2 ordinary-birth prepared-owner "
            "measurement for exact physical partitioning of the frozen B48 "
            "science matrix; projection adds 128 MiB to measured peak"
        )
    assumptions = dict(estimate["resource_estimator_assumptions"])
    assumptions["dominance_argument"] = (
        "WDL/fixed-MLP reuses the measured WDL state contract while omitting the "
        "resident GRU pack covered by the larger current Patch/GRU bound"
    )
    assumptions["admitted_resident_segment_generations"] = [3, 32]
    assumptions["required_runtime_environment"] = (
        dict(WDL_FIXED_GV2_OB_B48_REQUIRED_RUNTIME_ENVIRONMENT)
        if measured_wdl_gv2_ob_b48
        else None
    )
    assumptions["minimum_residual_headroom_bytes"] = (
        WDL_FIXED_GV2_OB_B48_MINIMUM_RESIDUAL_HEADROOM_BYTES
        if measured_wdl_gv2_ob_b48
        else 0
    )
    assumptions["wdl_fixed_gv2_ob_b48_prepared_owner"] = (
        {
            "prepared_owner_receipt_sha256": (
                WDL_FIXED_GV2_OB_B48_PREPARED_OWNER_RECEIPT_SHA256
            ),
            "measured_prepared_owner_peak_bytes": (
                WDL_FIXED_GV2_OB_B48_MEASURED_PREPARED_OWNER_PEAK_BYTES
            ),
            "conservative_projection_bytes": (
                WDL_FIXED_GV2_OB_B48_CONSERVATIVE_OWNER_PROJECTION_BYTES
            ),
            "accepted_scope": (
                "candidate_prepared_owner_authority_only_runtime_gate_pending"
            ),
            "admitted_batch_range": [
                WDL_FIXED_GV2_OB_B48_BATCH_CELLS,
                WDL_FIXED_GV2_OB_B48_BATCH_CELLS,
            ],
            "admitted_logical_generations": (
                WDL_FIXED_GV2_OB_B48_LOGICAL_GENERATIONS
            ),
            "admitted_logical_trajectories": (
                WDL_FIXED_GV2_OB_B48_LOGICAL_TRAJECTORIES
            ),
            "admitted_checkpoint_boundaries": (
                WDL_FIXED_GV2_OB_B48_CHECKPOINT_BOUNDARIES
            ),
            "admitted_resident_segment_generations": [
                3,
                WDL_FIXED_GV2_OB_B48_SEGMENT_GENERATIONS_MAX,
            ],
        }
        if measured_wdl_gv2_ob_b48
        else None
    )
    assumptions["wdl_fixed_gv2_ob_b24_prepared_owner"] = (
        {
            "prepared_owner_receipt_sha256": (
                WDL_FIXED_GV2_OB_B24_PREPARED_OWNER_RECEIPT_SHA256
            ),
            "measured_prepared_owner_peak_bytes": (
                WDL_FIXED_GV2_OB_B24_MEASURED_PREPARED_OWNER_PEAK_BYTES
            ),
            "conservative_projection_bytes": (
                WDL_FIXED_GV2_OB_B24_CONSERVATIVE_OWNER_PROJECTION_BYTES
            ),
            "conservative_margin_bytes": 128 * MIB,
            "accepted_scope": (
                "exact_physical_partition_b24_authority_runtime_gate_pending"
            ),
            "admitted_batch_range": [
                WDL_FIXED_GV2_OB_B24_BATCH_CELLS,
                WDL_FIXED_GV2_OB_B24_BATCH_CELLS,
            ],
            "admitted_logical_generations": [
                1,
                WDL_FIXED_GV2_OB_B48_LOGICAL_GENERATIONS,
            ],
            "admitted_resident_segment_generations": [
                3,
                WDL_FIXED_GV2_OB_B48_SEGMENT_GENERATIONS_MAX,
            ],
        }
        if measured_wdl_gv2_ob_b24
        else None
    )
    assumptions["fixed_mlp_batch_projection"] = (
        {
            "resource_estimator_id": fixed_projection["resource_estimator_id"],
            "vram_bytes_projection": fixed_projection["vram_bytes_projection"],
            "wall_seconds_projection": fixed_projection["wall_seconds_projection"],
            "ordinary_births_enabled": ordinary_births_enabled,
            "genome_upgrade_v2_enable": genome_upgrade_v2_enable,
            "genome_upgrade_v2_mutation_rate_q10": (
                genome_upgrade_v2_mutation_rate_q10
            ),
            "genome_upgrade_v2_mutation_step_q10": (
                genome_upgrade_v2_mutation_step_q10
            ),
        }
        if multicell_projection
        else None
    )
    estimate["resource_estimator_assumptions"] = assumptions
    return estimate


def required_runtime_environment_v1(
    estimate: Mapping[str, Any],
) -> dict[str, str]:
    """Return canonical runtime environment requirements carried by an estimate."""

    assumptions = estimate.get("resource_estimator_assumptions")
    if not isinstance(assumptions, Mapping):
        return {}
    required = assumptions.get("required_runtime_environment")
    if required is None:
        return {}
    if not isinstance(required, Mapping):
        raise ResourceEstimateError("required_runtime_environment must be a mapping")
    result: dict[str, str] = {}
    for key, value in required.items():
        if not isinstance(key, str) or not key:
            raise ResourceEstimateError(
                "required_runtime_environment keys must be non-empty strings"
            )
        if not isinstance(value, str) or not value:
            raise ResourceEstimateError(
                "required_runtime_environment values must be non-empty strings"
            )
        result[key] = value
    return {key: result[key] for key in sorted(result)}


def minimum_residual_headroom_bytes_v1(estimate: Mapping[str, Any]) -> int:
    """Return the route-specific residual headroom required after base headroom."""

    assumptions = estimate.get("resource_estimator_assumptions")
    if not isinstance(assumptions, Mapping):
        return 0
    value = assumptions.get("minimum_residual_headroom_bytes", 0)
    if type(value) is not int or value < 0:
        raise ResourceEstimateError("minimum_residual_headroom_bytes is invalid")
    return value


def estimate_patch_jungle_fixed_mlp_resident_resources_v1(
    *,
    batch_cells: int,
    logical_generations: int,
    logical_trajectories: int,
    checkpoint_boundaries: int,
    resident_segment_generations: int,
    physical_window_generations: int = PATCH_RESIDENT_WINDOW_GENERATIONS,
    ticks_per_generation: int = TICKS_PER_GENERATION,
    resident_capacity: int = PATCH_CAPACITY,
    ordinary_births_enabled: bool | None = None,
    genome_upgrade_v2_enable: bool | None = None,
    genome_upgrade_v2_mutation_rate_q10: int | None = None,
    genome_upgrade_v2_mutation_step_q10: int | None = None,
    cpython_gauss_segment_capacity: int | None = None,
    policy_birth_segment_refill_enable: bool | None = None,
    policy_birth_segment_capacity: int | None = None,
    turnover_mlp_shadow_elide_enable: bool | None = None,
    cpu_request_staging_enable: bool | None = None,
) -> dict[str, Any]:
    """Admit B1 Jungle under the larger measured recurrent controller bound."""

    estimate = estimate_patch_wdl_fixed_mlp_resident_resources_v1(
        batch_cells=batch_cells,
        logical_generations=logical_generations,
        logical_trajectories=logical_trajectories,
        checkpoint_boundaries=checkpoint_boundaries,
        resident_segment_generations=resident_segment_generations,
        physical_window_generations=physical_window_generations,
        ticks_per_generation=ticks_per_generation,
        resident_capacity=resident_capacity,
        ordinary_births_enabled=ordinary_births_enabled,
        genome_upgrade_v2_enable=genome_upgrade_v2_enable,
        genome_upgrade_v2_mutation_rate_q10=genome_upgrade_v2_mutation_rate_q10,
        genome_upgrade_v2_mutation_step_q10=genome_upgrade_v2_mutation_step_q10,
        cpython_gauss_segment_capacity=cpython_gauss_segment_capacity,
        policy_birth_segment_refill_enable=policy_birth_segment_refill_enable,
        policy_birth_segment_capacity=policy_birth_segment_capacity,
        turnover_mlp_shadow_elide_enable=turnover_mlp_shadow_elide_enable,
        cpu_request_staging_enable=cpu_request_staging_enable,
    )
    exact_physical_b136 = _is_acs_native_fixed_mlp_b136_resource_shape(
        batch_cells=batch_cells,
        logical_generations=logical_generations,
        logical_trajectories=logical_trajectories,
        resident_segment_generations=resident_segment_generations,
        physical_window_generations=physical_window_generations,
        ticks_per_generation=ticks_per_generation,
        resident_capacity=resident_capacity,
        ordinary_births_enabled=ordinary_births_enabled,
        genome_upgrade_v2_enable=genome_upgrade_v2_enable,
        genome_upgrade_v2_mutation_rate_q10=genome_upgrade_v2_mutation_rate_q10,
        genome_upgrade_v2_mutation_step_q10=genome_upgrade_v2_mutation_step_q10,
        cpython_gauss_segment_capacity=cpython_gauss_segment_capacity,
        policy_birth_segment_refill_enable=policy_birth_segment_refill_enable,
        policy_birth_segment_capacity=policy_birth_segment_capacity,
        turnover_mlp_shadow_elide_enable=turnover_mlp_shadow_elide_enable,
        cpu_request_staging_enable=cpu_request_staging_enable,
    )
    within_envelope = (
        exact_physical_b136
        or (batch_cells == 1 and 3 <= resident_segment_generations <= 32)
    )
    if not within_envelope:
        estimate["vram_bytes_upper_bound"] = "unavailable"
        estimate["wall_seconds_upper_bound"] = "unavailable"
        estimate["vram_bytes_unavailable_reason"] = (
            "Jungle resource admission is bounded to B1 and G3-G32 resident segments"
        )
        estimate["wall_seconds_unavailable_reason"] = (
            "Jungle resource admission is bounded to B1 and G3-G32 resident segments"
        )
    if not exact_physical_b136:
        estimate["resource_estimator_id"] = "patch-jungle-fixed-mlp-b1-g32-v1"
        estimate["resource_estimator_provenance"] = (
            "conservative current-artifact Patch/stable-GRU B1 bound; the canonical "
            "Jungle fixed-MLP owner omits the resident GRU pack and adds only two "
            "world-grid state tensors plus bounded reporting storage"
        )
    assumptions = dict(estimate["resource_estimator_assumptions"])
    assumptions["dominance_argument"] = (
        "the admitted recurrent bound includes the same Patch owner, policy tape, "
        "lifecycle storage, and a larger GRU pack; Jungle adds two B1 16x16 tensors"
    )
    assumptions["admitted_batch_range"] = (
        [
            ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS,
            ACS_NATIVE_FIXED_MLP_B136_LOGICAL_CELLS,
        ]
        if exact_physical_b136
        else [1, 1]
    )
    assumptions["admitted_resident_segment_generations"] = (
        [7, 7] if exact_physical_b136 else [3, 32]
    )
    assumptions["jungle_grid_state_bytes"] = 16 * 16 * (8 + 8)
    estimate["resource_estimator_assumptions"] = assumptions
    return estimate


def estimate_patch_gmnw_gru_resident_resources_v1(
    *,
    batch_cells: int,
    logical_generations: int,
    logical_trajectories: int,
    checkpoint_boundaries: int,
    resident_segment_generations: int,
    ticks_per_generation: int = TICKS_PER_GENERATION,
) -> dict[str, Any]:
    """Conservatively bound the qualified GMNW B1-B3, G4-G9 live envelope.

    The estimate deliberately reuses the larger audited Patch resident
    declaration. Retained GMNW evidence establishes only dominance and the
    bounded disk-checkpoint shape; it is not treated as a new exact formula.
    """

    estimate = estimate_patch_fixed_resident_resources_v1(
        batch_cells=batch_cells,
        logical_generations=logical_generations,
        logical_trajectories=logical_trajectories,
        checkpoint_boundaries=checkpoint_boundaries,
        resident_segment_generations=resident_segment_generations,
        ticks_per_generation=ticks_per_generation,
    )
    within_envelope = (
        GMNW_ADMITTED_BATCH_MIN <= batch_cells <= GMNW_ADMITTED_BATCH_MAX
        and GMNW_ADMITTED_SEGMENT_GENERATION_MIN
        <= resident_segment_generations
        <= GMNW_ADMITTED_SEGMENT_GENERATION_MAX
    )
    if within_envelope:
        # The qualified GMNW B3 gate and retained B48 allocation evidence admit
        # the same conservative projection that the shared Patch estimator
        # computes, without widening Patch's independent B2 authority.
        estimate["vram_bytes_upper_bound"] = estimate["vram_bytes_projection"]
        estimate["vram_bytes_unavailable_reason"] = None
    else:
        estimate["vram_bytes_upper_bound"] = "unavailable"
        estimate["wall_seconds_upper_bound"] = "unavailable"
        estimate["vram_bytes_unavailable_reason"] = GMNW_RESOURCE_ADMISSION_BLOCKER
        estimate["wall_seconds_unavailable_reason"] = GMNW_RESOURCE_ADMISSION_BLOCKER
    estimate["resource_estimator_id"] = "patch-gmnw-gru-qualified-b1-b3-g4-g9-v1"
    estimate["resource_estimator_provenance"] = (
        "conservative reuse of the audited Patch resident declaration, bounded by "
        "retained GMNW B48/G9 allocation-timing evidence and the current B3/G9 "
        "durable checkpoint and independent-source qualification gates"
    )
    assumptions = dict(estimate["resource_estimator_assumptions"])
    assumptions.update(
        {
            "admitted_batch_range": [
                GMNW_ADMITTED_BATCH_MIN,
                GMNW_ADMITTED_BATCH_MAX,
            ],
            "admitted_resident_segment_generations": [
                GMNW_ADMITTED_SEGMENT_GENERATION_MIN,
                GMNW_ADMITTED_SEGMENT_GENERATION_MAX,
            ],
            "checkpoint_storage_mode": "disk",
            "dominance_argument": (
                "the accepted GMNW B48 declaration uses the same 1536 MiB shared "
                "controller allowance and no larger per-cell declared allocation "
                "than the Patch projection; the current exact B3/G9 execution and "
                "checkpoint gates bind that dominance argument to this artifact"
            ),
            "retained_gmnw_allocation_evidence": (
                public_runtime_path('evidence_raw/eiel_gmnw_lossless_memory_reduction_20260721/47_accepted_baseline_compact_b48_measured/receipt.json')
            ),
            "retained_gmnw_g9_timing_evidence": (
                public_runtime_path('evidence_raw/eiel_gmnw_lossless_memory_reduction_20260721/20_b48_capability_g9/receipt.json')
            ),
            "current_b3_g9_disk_checkpoint_evidence": (
                public_runtime_path('evidence_raw/eiel_declarative_gmnw_closeout_f2b33dc7/03_checkpoint/result.json')
            ),
            "qualification_status": "qualified",
        }
    )
    estimate["resource_estimator_assumptions"] = assumptions
    return estimate


def _uniform_request_parameter_v1(
    requests: Sequence[Any],
    name: str,
) -> Any | None:
    values: set[Any] = set()
    for request in requests:
        parameter_arrays = getattr(request, "parameter_arrays", ())
        if not isinstance(parameter_arrays, Sequence):
            continue
        for parameter_name, parameter_values in parameter_arrays:
            if parameter_name != name:
                continue
            if isinstance(parameter_values, (str, bytes)) or not isinstance(
                parameter_values,
                Sequence,
            ):
                continue
            values.update(parameter_values)
    if len(values) == 1:
        return next(iter(values))
    return None


def resident_request_resource_dimensions_v1(
    plan: Mapping[str, Any],
    requests: Sequence[Any],
    *,
    include_parameter_dimensions: bool = False,
) -> dict[str, Any]:
    """Extract shared dimensions; each provider owns its estimator choice."""

    batch_cells = max((len(request.cell_ids) for request in requests), default=0)
    execution = plan.get("execution")
    resources = plan.get("resource_estimate")
    if not isinstance(execution, Mapping) or not isinstance(resources, Mapping):
        raise ResourceEstimateError("compiled plan lacks execution/resource estimate")
    tick_native = execution.get("execution_route_identity") == (
        "tick_native_execution_route_v1"
    )
    total_ticks = execution.get("total_ticks") if tick_native else None
    tick_native_control = execution.get("tick_native_control")
    ticks = (
        tick_native_control.get("source_window_ticks")
        if tick_native and isinstance(tick_native_control, Mapping)
        else execution.get("ticks_per_generation", TICKS_PER_GENERATION)
    )
    if tick_native and not isinstance(tick_native_control, Mapping):
        raise ResourceEstimateError(
            "tick-native resource dimensions lack tick_native_control"
        )
    native_source_windows = None
    if tick_native:
        if type(ticks) is not int:
            raise ResourceEstimateError("compiled source window is invalid")
        native_source_windows = max(
            1,
            (
                int(tick_native_control["native_dispatch_window_ticks"])
                + ticks
                - 1
            )
            // ticks,
        )
        legacy_generation_count = None
    else:
        legacy_generation_count = (
            (total_ticks + ticks - 1) // ticks
            if type(total_ticks) is int and type(ticks) is int
            else execution.get("generations")
        )
    trajectories = resources.get("logical_trajectories")
    checkpoints = (
        execution.get("checkpoint_ticks", [])
        if tick_native
        else execution.get("checkpoints")
    )
    checkpoint_every = execution.get("checkpoint_every_ticks") if tick_native else None
    if tick_native and checkpoint_every is not None:
        checkpoints = list(range(checkpoint_every, total_ticks + 1, checkpoint_every))
    if not isinstance(checkpoints, list):
        raise ResourceEstimateError("compiled checkpoint schedule is invalid")
    if tick_native:
        if type(total_ticks) is not int or total_ticks < 1:
            raise ResourceEstimateError("compiled total tick horizon is invalid")
    elif type(legacy_generation_count) is not int:
        raise ResourceEstimateError("compiled generation count is invalid")
    if type(ticks) is not int:
        raise ResourceEstimateError("compiled tick count is invalid")
    boundaries = tuple(
        sorted(
            {
                0,
                *checkpoints,
                total_ticks if tick_native else legacy_generation_count,
            }
        )
    )
    segment_ticks = tuple(
        right - left for left, right in zip(boundaries, boundaries[1:])
    )
    segments = (
        (int(native_source_windows),)
        if tick_native and native_source_windows is not None
        else segment_ticks
    )
    if not segments:
        raise ResourceEstimateError(
            "compiled checkpoint schedule has no resident segments"
        )
    if tick_native:
        dimensions = {
            "batch_cells": batch_cells,
            "logical_trajectories": trajectories,
            # Tick 0 is an absolute cursor, not an emitted checkpoint.  Active
            # native routes account only the explicit checkpoint schedule.
            "checkpoint_boundaries": len(checkpoints),
            "time_authority": "absolute_tick_v1",
            "total_ticks": total_ticks,
            "resident_owner_span_ticks": int(
                tick_native_control["resident_owner_span_ticks"]
            ),
            "native_dispatch_window_ticks": int(
                tick_native_control["native_dispatch_window_ticks"]
            ),
            "event_window_capacity": int(tick_native_control["event_window_capacity"]),
            "source_window_ticks": int(tick_native_control["source_window_ticks"]),
            "resident_dispatch_window_ticks": int(
                tick_native_control["native_dispatch_window_ticks"]
            ),
            "resident_event_bytes_horizon_invariant": True,
            "resident_source_bytes_horizon_invariant": True,
        }
    else:
        dimensions = {
            "batch_cells": batch_cells,
            "logical_generations": legacy_generation_count,
            "ticks_per_generation": ticks,
            "logical_trajectories": trajectories,
            "checkpoint_boundaries": len(checkpoints) + 1,
            "resident_segment_generations": max(segments),
            "minimum_resident_segment_generations": min(segments),
        }
    physical_window_generations = execution.get("physical_window_generations")
    if physical_window_generations is not None:
        dimensions["physical_window_generations"] = physical_window_generations
    if not include_parameter_dimensions:
        return dimensions

    parameter_bindings = {
        "patch.capacity": "resident_capacity",
        "production_lifecycle.ordinary_births_enabled": "ordinary_births_enabled",
        "genome_upgrade_v2.enable": "genome_upgrade_v2_enable",
        "genome_upgrade_v2.mutation_rate_q10": "genome_upgrade_v2_mutation_rate_q10",
        "genome_upgrade_v2.mutation_step_q10": "genome_upgrade_v2_mutation_step_q10",
        "cuda_architecture_b.cpython_gauss_segment_capacity": "cpython_gauss_segment_capacity",
        "cuda_architecture_b.policy_birth_segment_refill_enable": "policy_birth_segment_refill_enable",
        "cuda_architecture_b.policy_birth_segment_capacity": "policy_birth_segment_capacity",
        "cuda_architecture_b.turnover_mlp_shadow_elide_enable": "turnover_mlp_shadow_elide_enable",
        "cuda_architecture_b.cpu_request_staging_enable": "cpu_request_staging_enable",
    }
    for parameter_name, dimension_name in parameter_bindings.items():
        value = _uniform_request_parameter_v1(requests, parameter_name)
        if value is not None:
            dimensions[dimension_name] = value
    return dimensions


__all__ = [
    "PATCH_RESIDENT_WINDOW_GENERATIONS",
    "ResourceEstimateError",
    "VRAM_BOUND_BLOCKER",
    "WALL_BOUND_BLOCKER",
    "estimate_patch_fixed_resident_resources_v1",
    "fixed_mlp_dense_policy_window_bytes_per_cell_v1",
    "fixed_mlp_policy_window_bytes_per_cell_v1",
    "patch_resident_window_route_id_v1",
    "estimate_patch_gmnw_gru_resident_resources_v1",
    "estimate_patch_jungle_fixed_mlp_resident_resources_v1",
    "estimate_patch_stable_gru_resident_resources_v1",
    "estimate_patch_wdl_fixed_mlp_resident_resources_v1",
    "estimate_patch_wdl_stable_gru_resident_resources_v1",
    "minimum_residual_headroom_bytes_v1",
    "planning_disk_upper_bound_absolute_ticks_v1",
    "planning_acs_native_fixed_mlp_b136_checkpoint_upper_bound_v1",
    "planning_disk_upper_bound_v1",
    "resident_request_resource_dimensions_v1",
    "required_runtime_environment_v1",
]
