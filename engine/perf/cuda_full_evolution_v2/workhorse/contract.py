"""Strict host contract for the shape-specialized CUDA workhorse V2.

The transition identity is deliberately independent of CUDA topology. Every
executable profile, route, artifact and tuning decision is nevertheless bound
to one exact W/H/P/C shape and one exact backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Final

WORKHORSE_PROFILE_FAMILY_V2: Final[str] = "EIEL_FULL_EVOLUTION_CUDA_WORKHORSE_V2"
WORKHORSE_PROFILE_ID_PREFIX_V2: Final[str] = "eiel_full_evolution_cuda_workhorse_v2"
WORKHORSE_ROUTE_PREFIX_V2: Final[str] = "tensor_world_cuda_full_evolution_workhorse_v2"
WORKHORSE_ABI_V2: Final[str] = "eiel_full_evolution_cuda_workhorse_abi_v2"
WORKHORSE_EVENT_ABI_V2: Final[str] = "eiel_full_evolution_workhorse_event_u32_v2"
WORKHORSE_EVENT_ABI_V3: Final[str] = (
    "eiel_full_evolution_workhorse_sparse_lifecycle_event_v3"
)
WORKHORSE_CHECKPOINT_SCHEMA_V2: Final[str] = (
    "eiel_full_evolution_cuda_workhorse_checkpoint_v2"
)
WORKHORSE_TRANSITION_ID_V2: Final[str] = "eiel_full_evolution_workhorse_transition_v2"
WORKHORSE_TIME_AUTHORITY_V2: Final[str] = "absolute_tick_v1"
WORKHORSE_WORLD_IDENTITY_BINDING_SCHEMA_V1: Final[str] = (
    "eiel_full_evolution_workhorse_world_identity_binding_v1"
)
WORKHORSE_INDEX_DTYPE_V2: Final[str] = "u32"
WORKHORSE_INDEX_WIDTH_BITS_V2: Final[int] = 32
WORKHORSE_EMPTY_INDEX_V2: Final[int] = (1 << 32) - 1
WORKHORSE_EVENT_ROW_BYTES_V2: Final[int] = 48
WORKHORSE_THREADS_V2: Final[int] = 256
WORKHORSE_MAX_GRID_Y_V2: Final[int] = 65_535
WORKHORSE_TEMPORAL_K_CANDIDATES_V2: Final[tuple[int, ...]] = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
)
MAX_SIGNED_INT64: Final[int] = (1 << 63) - 1

BACKEND_REFERENCE_PHASED_V2: Final[str] = "reference_phased_v2"
BACKEND_PHASE_GRAPH_V2: Final[str] = "phase_graph_v2"
BACKEND_FUSED_CTA_TEMPORAL_V2: Final[str] = "fused_cta_temporal_v2"

FULL_EVOLUTION_CAPABILITIES_V2: Final[tuple[str, ...]] = (
    "patch",
    "wdl_delayed_hazard",
    "jungle",
    "eed",
    "genome_v2_ordinary_birth",
    "genome_v2_mutation",
    "recombination_profile",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


# This is the workhorse-specific governing semantic law.  It intentionally
# contains no kernel, launch, batching, temporal blocking, shape, device or
# toolchain fields, and it explicitly does not inherit the broader micro-route
# architecture semantics.  The operator authorized a new semantic family, so
# truthfulness is preferred over compatibility.
WORKHORSE_REJECTED_SEMANTIC_HASHES_V2: Final[frozenset[str]] = frozenset(
    {
        "6def8b2365870ebbfcab99897b3321f40706c76666112d2fc10d37d53370768e",
        "046e45859028e2fd2d966e68bcda1527b60506145dc23cdbf95c49177b13e854",
    }
)
_TRANSITION_SEMANTICS_V2: Final[dict[str, Any]] = {
    "schema_version": "eiel_full_evolution_workhorse_semantics_v2_revision_3",
    "transition_id": WORKHORSE_TRANSITION_ID_V2,
    "authority_scope": "workhorse_v2_only_no_micro_semantic_inheritance",
    "compatibility_obligation": "none",
    "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
    "capabilities": list(FULL_EVOLUTION_CAPABILITIES_V2),
    "tick_order": [
        "settle_patch_then_regrow_resource_and_rotate_two_tick_wdl_ring",
        "decay_eed_field_and_clear_per_tick_scratch",
        "draw_direct_rng_movement_target_and_resolve_uid_minimum_winner",
        "release_then_commit_accepted_moves",
        "gather_resource_emit_patch_wdl_eed_and_apply_physiology",
        "commit_deaths_and_death_events",
        "pair_even_uid_order_ordinal_with_next_ordinal_and_fixed_direction_spawn",
        "rank_uid_order_birth_candidates_and_canonical_free_slots_then_truncate",
        "per_locus_two_parent_homolog_recombination_then_mutation_then_expression",
        "commit_birth_events_then_publish_next_absolute_tick",
    ],
    "policy_and_observation": {
        "present": False,
        "movement_law": (
            "direction=philox(rng_key,world_uid,absolute_tick,movement_domain,"
            "biological_uid,operation=0,draw=0)&3; target is the toroidal neighbor; "
            "the minimum biological UID wins each empty target"
        ),
    },
    "patch_law": (
        "at tick start patch=min(4095,patch+patch_pending), patch_pending=0, and "
        "resource=min(4095,resource+1+patch//64); each live agent later emits "
        "1+expressed_locus_0//64 into patch_pending at its occupied site"
    ),
    "wdl_delay_ticks": 2,
    "wdl_law": (
        "tick start active=delay0, delay0=delay1, delay1=0; live-agent emission "
        "1+expressed_locus_3//64 enters delay1; active applies energy and health cost"
    ),
    "jungle_law": (
        "the checkpointed per-site u8 jungle field adds jungle//32 energy cost and "
        "jungle//64 health cost"
    ),
    "eed_law": (
        "checkpointed eed_field decays by eed_field//16 and receives emission; the "
        "inherited diploid expressed locus 31 gives eed_bonus=expressed//64, changes "
        "gather demand to 1+eed_bonus, and thereby changes resource, energy and later "
        "birth eligibility"
    ),
    "ordinary_birth_law": (
        "survivors remain in prior biological-UID order; each even survivor ordinal "
        "pairs with the immediately following ordinal when both energies are at least "
        "256; spawn is direction 2 from the first parent; minimum first-parent UID "
        "wins a contested empty spawn; accepted births truncate to free-slot count"
    ),
    "mutation_law": (
        "one addressed draw per child selects no mutation unless fixture mode or "
        "draw&7==0; when selected, draw%32 flips bit 7 of child homolog A after "
        "recombination; expressed=floor((homolog_A+homolog_B)/2) is recomputed"
    ),
    "recombination_law": (
        "for every one of 32 loci, child homolog A independently selects parent-A "
        "homolog A or B with the recombination-A addressed draw, while child homolog B "
        "independently selects parent-B homolog A or B with the recombination-B draw"
    ),
    "ordering": (
        "checkpointed active_slots_by_uid; stable survivor compaction in prior UID order; "
        "births append in child-UID rank; canonical free-slot order; immutable UID RNG/events"
    ),
    "rng": {
        "kind": "philox4x32_10_addressed",
        "epoch": 4,
        "address": [
            "rng_key",
            "world_uid",
            "absolute_tick",
            "domain",
            "biological_uid_or_site",
            "operation",
            "draw",
        ],
        "forbidden": ["B", "K", "grid", "block", "warp", "lane", "launch_order"],
    },
    "events": {
        "row_bytes": WORKHORSE_EVENT_ROW_BYTES_V2,
        "device_page_rows_per_world": "2*C independent of temporal K",
        "canonical_order": (
            "world_uid,absolute_tick,event_kind,slot; death precedes birth and "
            "slot is ascending within kind"
        ),
        "event_id": "(event_kind<<48)|slot",
        "wire": "packed active 48-byte rows plus exact per-world offsets and counts",
        "overflow": (
            "count the complete required extent without out-of-bounds writes, restore "
            "the exact pre-dispatch canonical state, and split into smaller absolute-"
            "tick physical segments; any recovered benchmark sample is non-selectable"
        ),
        "frontier": "external_exact_content_ack_before_reuse_or_checkpoint",
    },
    "transaction": "publish next_absolute_tick last; any status error rejects dispatch",
}
WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2: Final[str] = canonical_sha256(
    _TRANSITION_SEMANTICS_V2
)
if WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2 in WORKHORSE_REJECTED_SEMANTIC_HASHES_V2:
    raise RuntimeError(
        "workhorse semantic revision failed to invalidate old identities"
    )


def workhorse_semantic_contract_v2() -> dict[str, Any]:
    """Return a detached exact projection of the frozen governing law."""
    value = json.loads(json.dumps(_TRANSITION_SEMANTICS_V2))
    if canonical_sha256(value) != WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2:
        raise RuntimeError("workhorse semantic contract changed after import")
    return {
        "contract": value,
        "semantic_transition_sha256": WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
    }


class FullEvolutionWorkhorseContractError(ValueError):
    """Raised when a workhorse identity or bounded integer is inexact."""


def checked_add_i64(left: int, right: int, label: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise FullEvolutionWorkhorseContractError(
            f"{label} operands must be nonnegative integers"
        )
    if left > MAX_SIGNED_INT64 - right:
        raise FullEvolutionWorkhorseContractError(f"{label} exceeds signed int64")
    return left + right


def checked_multiply_i64(left: int, right: int, label: str) -> int:
    if type(left) is not int or type(right) is not int or left < 0 or right < 0:
        raise FullEvolutionWorkhorseContractError(
            f"{label} operands must be nonnegative integers"
        )
    if left and right > MAX_SIGNED_INT64 // left:
        raise FullEvolutionWorkhorseContractError(f"{label} exceeds signed int64")
    return left * right


def workhorse_world_identity_binding_v2(
    *,
    world_uids: Sequence[int],
    rng_keys: Sequence[int],
) -> dict[str, Any]:
    """Seal the ordered native world/RNG vector independently of B/K/device.

    These are carrier identities, not physical slot identities.  The order is
    the exact placement for one physical operation, while each UID/key pair is
    stable across later packing changes.
    """

    if isinstance(world_uids, (str, bytes, bytearray)) or isinstance(
        rng_keys, (str, bytes, bytearray)
    ):
        raise FullEvolutionWorkhorseContractError(
            "world identity vectors must be integer sequences"
        )
    uids = tuple(world_uids)
    keys = tuple(rng_keys)
    if not uids or len(uids) != len(keys) or len(uids) > WORKHORSE_MAX_GRID_Y_V2:
        raise FullEvolutionWorkhorseContractError(
            "world identity vectors must align within the Workhorse B limit"
        )
    if any(type(uid) is not int or not 0 < uid <= MAX_SIGNED_INT64 for uid in uids):
        raise FullEvolutionWorkhorseContractError(
            "world UIDs must be positive signed-int64 carrier identities"
        )
    if len(set(uids)) != len(uids):
        raise FullEvolutionWorkhorseContractError(
            "world UIDs must be unique within one physical operation"
        )
    if any(type(key) is not int or not 0 <= key <= MAX_SIGNED_INT64 for key in keys):
        raise FullEvolutionWorkhorseContractError(
            "RNG keys must be nonnegative signed-int64 carrier identities"
        )
    body = {
        "schema_version": WORKHORSE_WORLD_IDENTITY_BINDING_SCHEMA_V1,
        "world_count": len(uids),
        "world_uids": list(uids),
        "rng_keys": list(keys),
        "packing_independent_fields": ["world_uids", "rng_keys"],
        "forbidden_identity_inputs": [
            "physical_batch_index",
            "physical_slot",
            "B",
            "K",
            "device",
        ],
    }
    return {**body, "binding_sha256": canonical_sha256(body)}


def validate_workhorse_world_identity_binding_v2(
    value: Mapping[str, Any],
    *,
    expected_world_count: int | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullEvolutionWorkhorseContractError(
            "world identity binding must be an object"
        )
    expected_fields = {
        "schema_version",
        "world_count",
        "world_uids",
        "rng_keys",
        "packing_independent_fields",
        "forbidden_identity_inputs",
        "binding_sha256",
    }
    if set(value) != expected_fields:
        raise FullEvolutionWorkhorseContractError(
            "world identity binding fields are inexact"
        )
    if type(value.get("world_uids")) is not list or type(
        value.get("rng_keys")
    ) is not list:
        raise FullEvolutionWorkhorseContractError(
            "world identity binding vectors must be lists"
        )
    expected = workhorse_world_identity_binding_v2(
        world_uids=value["world_uids"],
        rng_keys=value["rng_keys"],
    )
    if dict(value) != expected:
        raise FullEvolutionWorkhorseContractError(
            "world identity binding does not recompute"
        )
    if expected_world_count is not None and expected["world_count"] != expected_world_count:
        raise FullEvolutionWorkhorseContractError(
            "world identity binding does not exactly cover B"
        )
    return expected


@dataclass(frozen=True, slots=True)
class WorkhorseCompileShapeV2:
    """One exact compile-time specialization and its bounded K candidates."""

    shape_id: str
    width: int
    height: int
    initial_population: int
    capacity: int
    temporal_k_candidates: tuple[int, ...]
    batch_seed: int

    def __post_init__(self) -> None:
        for name, value in (
            ("width", self.width),
            ("height", self.height),
            ("initial_population", self.initial_population),
            ("capacity", self.capacity),
        ):
            if type(value) is not int or value <= 0:
                raise FullEvolutionWorkhorseContractError(
                    f"{name} must be a positive integer"
                )
        sites = checked_multiply_i64(self.width, self.height, "site count")
        if sites >= WORKHORSE_EMPTY_INDEX_V2:
            raise FullEvolutionWorkhorseContractError(
                "site count must reserve UINT32_MAX as empty"
            )
        if self.capacity >= WORKHORSE_EMPTY_INDEX_V2:
            raise FullEvolutionWorkhorseContractError(
                "capacity must reserve UINT32_MAX as empty"
            )
        if not self.initial_population <= min(sites, self.capacity):
            raise FullEvolutionWorkhorseContractError(
                "initial population exceeds sites or capacity"
            )
        expected = (
            f"w{self.width}h{self.height}p{self.initial_population}c{self.capacity}"
        )
        if self.shape_id != expected:
            raise FullEvolutionWorkhorseContractError(f"shape_id must equal {expected}")
        if (
            not self.temporal_k_candidates
            or tuple(sorted(set(self.temporal_k_candidates)))
            != self.temporal_k_candidates
            or self.temporal_k_candidates[0] != 1
            or any(
                type(value) is not int or value <= 0
                for value in self.temporal_k_candidates
            )
        ):
            raise FullEvolutionWorkhorseContractError(
                "temporal K candidates must be unique ascending positive integers starting at 1"
            )
        if (
            type(self.batch_seed) is not int
            or not 1 <= self.batch_seed <= WORKHORSE_MAX_GRID_Y_V2
        ):
            raise FullEvolutionWorkhorseContractError(
                "batch_seed must fit the grid.y ABI"
            )

    @property
    def sites(self) -> int:
        return checked_multiply_i64(self.width, self.height, "site count")

    @property
    def profile_id(self) -> str:
        return f"{WORKHORSE_PROFILE_ID_PREFIX_V2}__{self.shape_id}"

    @property
    def route_id(self) -> str:
        return f"{WORKHORSE_ROUTE_PREFIX_V2}__{self.shape_id}"

    @property
    def site_tiles(self) -> int:
        return (self.sites + WORKHORSE_THREADS_V2 - 1) // WORKHORSE_THREADS_V2

    @property
    def agent_tiles(self) -> int:
        return (self.capacity + WORKHORSE_THREADS_V2 - 1) // WORKHORSE_THREADS_V2

    @property
    def rank_tiles(self) -> int:
        return self.agent_tiles

    @property
    def maximum_total_ticks(self) -> int:
        worst_counter_delta = max(self.sites, self.capacity * 256)
        return MAX_SIGNED_INT64 // worst_counter_delta

    def identity_fields(self) -> dict[str, Any]:
        return {
            "shape_id": self.shape_id,
            "width": self.width,
            "height": self.height,
            "sites": self.sites,
            "initial_population": self.initial_population,
            "capacity": self.capacity,
            "profile_id": self.profile_id,
            "route_id": self.route_id,
            "index_dtype": WORKHORSE_INDEX_DTYPE_V2,
            "index_width_bits": WORKHORSE_INDEX_WIDTH_BITS_V2,
            "threads": WORKHORSE_THREADS_V2,
            "site_tiles": self.site_tiles,
            "agent_tiles": self.agent_tiles,
            "rank_tiles": self.rank_tiles,
            "temporal_k_candidates": list(self.temporal_k_candidates),
            "batch_seed": self.batch_seed,
        }


SUPPORTED_WORKHORSE_SHAPES_V2: Final[tuple[WorkhorseCompileShapeV2, ...]] = (
    WorkhorseCompileShapeV2(
        "w64h64p1024c2048",
        64,
        64,
        1024,
        2048,
        WORKHORSE_TEMPORAL_K_CANDIDATES_V2,
        544,
    ),
    WorkhorseCompileShapeV2(
        "w128h80p2560c5120",
        128,
        80,
        2560,
        5120,
        WORKHORSE_TEMPORAL_K_CANDIDATES_V2,
        256,
    ),
    WorkhorseCompileShapeV2(
        "w160h160p6400c12800",
        160,
        160,
        6400,
        12800,
        WORKHORSE_TEMPORAL_K_CANDIDATES_V2,
        128,
    ),
    WorkhorseCompileShapeV2(
        "w256h200p12800c25600",
        256,
        200,
        12800,
        25600,
        WORKHORSE_TEMPORAL_K_CANDIDATES_V2,
        64,
    ),
    WorkhorseCompileShapeV2(
        "w320h320p25600c51200",
        320,
        320,
        25600,
        51200,
        WORKHORSE_TEMPORAL_K_CANDIDATES_V2,
        32,
    ),
)
SUPPORTED_WORKHORSE_SHAPES_BY_ID_V2: Final[dict[str, WorkhorseCompileShapeV2]] = {
    shape.shape_id: shape for shape in SUPPORTED_WORKHORSE_SHAPES_V2
}


@dataclass(frozen=True, slots=True)
class WorkhorseBackendV2:
    backend_id: str
    kind: str
    performance_candidate: bool
    temporal_blocking: bool

    def topology_payload(self, shape: WorkhorseCompileShapeV2) -> dict[str, Any]:
        common = {
            "schema_version": "eiel_full_evolution_workhorse_backend_topology_v2",
            "backend_id": self.backend_id,
            "kind": self.kind,
            "shape_id": shape.shape_id,
            "threads": WORKHORSE_THREADS_V2,
            "world_axis": "grid.y",
            "world_isolation": "all offsets begin at blockIdx.y world row",
        }
        if self.backend_id in (BACKEND_REFERENCE_PHASED_V2, BACKEND_PHASE_GRAPH_V2):
            performance_sparse = self.backend_id == BACKEND_PHASE_GRAPH_V2
            kernels_per_tick = [
                "environment_phase_v2",
                "agent_intent_phase_v2",
                "movement_accept_phase_v2",
                "movement_release_phase_v2",
                "movement_commit_phase_v2",
                "interaction_physiology_phase_v2",
                "death_phase_v2",
            ]
            if performance_sparse:
                kernels_per_tick.append("compact_death_events_v3")
            kernels_per_tick.extend(
                [
                    "rank_tiles_phase_v2",
                    "rank_prefix_phase_v2",
                    "scatter_ranks_phase_v2",
                    "mate_spawn_phase_v2",
                    "birth_commit_phase_v2",
                ]
            )
            if performance_sparse:
                kernels_per_tick.append("compact_birth_events_v3")
            kernels_per_tick.append("publish_phase_v2")
            common.update(
                {
                    "kernels_per_tick": kernels_per_tick,
                    "edges_per_tick": len(kernels_per_tick) - 1,
                    "site_tiles": shape.site_tiles,
                    "agent_tiles": shape.agent_tiles,
                    "single_cta_world_phases": [
                        "rank_prefix_phase_v2",
                        "mate_spawn_phase_v2",
                        *(
                            [
                                "compact_death_events_v3",
                                "compact_birth_events_v3",
                            ]
                            if performance_sparse
                            else []
                        ),
                        "publish_phase_v2",
                    ],
                    "event_transition_preparation": (
                        "reference_k1_single_page_dense_zero"
                        if not performance_sparse
                        else (
                            "single_2C_sparse_page_status_invalidation_once_inside_"
                            "first_environment_phase_then_stable_slot_compaction"
                        )
                    ),
                    "capture": self.backend_id == BACKEND_PHASE_GRAPH_V2,
                }
            )
            if self.backend_id == BACKEND_PHASE_GRAPH_V2:
                common.update(
                    {
                        "capture_owner": "native_prepared_graph_v2",
                        "dispatch_boundary_nodes": [
                            "event_count_memset_v2",
                        ],
                        "graph_kernel_nodes_per_temporal_k": 15,
                        "graph_edge_law": "linear_stream_capture_nodes_minus_one",
                        "runtime_inspection": (
                            "cudaGraphGetNodes_edges_kernel_and_function_attributes"
                        ),
                    }
                )
        else:
            common.update(
                {
                    "kernels_per_dispatch": ["fused_world_temporal_v2"],
                    "edges_per_dispatch": 0,
                    "ctas_per_world": 1,
                    "temporal_k_inside_kernel": True,
                    "shape_strided_site_and_agent_loops": True,
                    "event_transition_preparation": (
                        "single_2C_sparse_page_status_invalidation_once_inside_"
                        "fused_kernel_then_stable_slot_compaction_each_tick"
                    ),
                }
            )
        return common

    def topology_sha256(self, shape: WorkhorseCompileShapeV2) -> str:
        return canonical_sha256(self.topology_payload(shape))


WORKHORSE_BACKENDS_V2: Final[tuple[WorkhorseBackendV2, ...]] = (
    WorkhorseBackendV2(
        BACKEND_REFERENCE_PHASED_V2, "correctness_reference", False, False
    ),
    WorkhorseBackendV2(BACKEND_PHASE_GRAPH_V2, "performance", True, False),
    WorkhorseBackendV2(BACKEND_FUSED_CTA_TEMPORAL_V2, "performance", True, True),
)
WORKHORSE_BACKENDS_BY_ID_V2: Final[dict[str, WorkhorseBackendV2]] = {
    backend.backend_id: backend for backend in WORKHORSE_BACKENDS_V2
}


def workhorse_shape_v2(shape_id: str) -> WorkhorseCompileShapeV2:
    if type(shape_id) is not str:
        raise FullEvolutionWorkhorseContractError("shape_id must be a string")
    try:
        return SUPPORTED_WORKHORSE_SHAPES_BY_ID_V2[shape_id]
    except KeyError as exc:
        raise FullEvolutionWorkhorseContractError(
            f"unsupported workhorse compile shape: {shape_id}"
        ) from exc


def workhorse_backend_v2(backend_id: str) -> WorkhorseBackendV2:
    if type(backend_id) is not str:
        raise FullEvolutionWorkhorseContractError("backend_id must be a string")
    try:
        return WORKHORSE_BACKENDS_BY_ID_V2[backend_id]
    except KeyError as exc:
        raise FullEvolutionWorkhorseContractError(
            f"unsupported workhorse backend: {backend_id}"
        ) from exc


def workhorse_backend_temporal_k_candidates_v2(
    shape: str | WorkhorseCompileShapeV2, backend_id: str
) -> tuple[int, ...]:
    """Return the exact dispatch-K envelope implemented by one backend."""
    if isinstance(shape, str):
        shape = workhorse_shape_v2(shape)
    if workhorse_shape_v2(shape.shape_id) != shape:
        raise FullEvolutionWorkhorseContractError("shape must be exact")
    backend = workhorse_backend_v2(backend_id)
    if backend.backend_id == BACKEND_REFERENCE_PHASED_V2:
        return (1,)
    return shape.temporal_k_candidates


_GENERATION_KEY = re.compile(r"(?:generation|(?:^|[_-])g[0-9]+(?:$|[_-]))", re.I)


def _reject_generation_keys(value: object, path: str = "contract") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise FullEvolutionWorkhorseContractError(
                    f"{path}: mapping keys must be strings"
                )
            if _GENERATION_KEY.search(key):
                raise FullEvolutionWorkhorseContractError(
                    f"{path}.{key}: generation-shaped keys are forbidden"
                )
            _reject_generation_keys(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_generation_keys(nested, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class FullEvolutionCudaWorkhorseContractV2:
    """Exact execution identity for one shape/backend/B/K attempt."""

    shape_id: str
    backend_id: str
    B: int
    temporal_k: int
    total_ticks: int
    start_next_tick: int = 0
    profile: str = WORKHORSE_PROFILE_FAMILY_V2
    native_abi: str = WORKHORSE_ABI_V2
    event_abi: str = WORKHORSE_EVENT_ABI_V3
    time_authority: str = WORKHORSE_TIME_AUTHORITY_V2
    semantic_transition_id: str = WORKHORSE_TRANSITION_ID_V2
    semantic_transition_sha256: str = WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2
    capabilities: tuple[str, ...] = FULL_EVOLUTION_CAPABILITIES_V2
    profile_id: str | None = None
    route_id: str | None = None
    backend_topology_sha256: str | None = None

    def __post_init__(self) -> None:
        shape = workhorse_shape_v2(self.shape_id)
        backend = workhorse_backend_v2(self.backend_id)
        expected_profile_id = shape.profile_id
        expected_route_id = shape.route_id
        expected_topology = backend.topology_sha256(shape)
        object.__setattr__(
            self,
            "profile_id",
            expected_profile_id if self.profile_id is None else self.profile_id,
        )
        object.__setattr__(
            self,
            "route_id",
            expected_route_id if self.route_id is None else self.route_id,
        )
        object.__setattr__(
            self,
            "backend_topology_sha256",
            (
                expected_topology
                if self.backend_topology_sha256 is None
                else self.backend_topology_sha256
            ),
        )
        if type(self.B) is not int or not 1 <= self.B <= WORKHORSE_MAX_GRID_Y_V2:
            raise FullEvolutionWorkhorseContractError(
                f"B must be in [1, {WORKHORSE_MAX_GRID_Y_V2}] for grid.y isolation"
            )
        if (
            type(self.temporal_k) is not int
            or self.temporal_k not in shape.temporal_k_candidates
        ):
            raise FullEvolutionWorkhorseContractError(
                f"temporal_k must be one of {shape.temporal_k_candidates} for {shape.shape_id}"
            )
        if backend.backend_id == BACKEND_REFERENCE_PHASED_V2 and self.temporal_k != 1:
            raise FullEvolutionWorkhorseContractError(
                "reference backend requires temporal_k=1"
            )
        if (
            type(self.total_ticks) is not int
            or not 1 <= self.total_ticks <= shape.maximum_total_ticks
        ):
            raise FullEvolutionWorkhorseContractError(
                f"total_ticks must be in [1, {shape.maximum_total_ticks}]"
            )
        if type(self.start_next_tick) is not int or self.start_next_tick < 0:
            raise FullEvolutionWorkhorseContractError(
                "start_next_tick must be a nonnegative integer"
            )
        if self.start_next_tick > MAX_SIGNED_INT64 - self.total_ticks:
            raise FullEvolutionWorkhorseContractError(
                "start_next_tick + total_ticks exceeds signed int64"
            )
        required = (
            ("profile", self.profile, WORKHORSE_PROFILE_FAMILY_V2),
            ("profile_id", self.profile_id, expected_profile_id),
            ("route_id", self.route_id, expected_route_id),
            ("native_abi", self.native_abi, WORKHORSE_ABI_V2),
            ("event_abi", self.event_abi, WORKHORSE_EVENT_ABI_V3),
            ("time_authority", self.time_authority, WORKHORSE_TIME_AUTHORITY_V2),
            (
                "semantic_transition_id",
                self.semantic_transition_id,
                WORKHORSE_TRANSITION_ID_V2,
            ),
            (
                "semantic_transition_sha256",
                self.semantic_transition_sha256,
                WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
            ),
            (
                "backend_topology_sha256",
                self.backend_topology_sha256,
                expected_topology,
            ),
            ("capabilities", self.capabilities, FULL_EVOLUTION_CAPABILITIES_V2),
        )
        for name, actual, expected in required:
            if actual != expected:
                raise FullEvolutionWorkhorseContractError(
                    f"{name} must equal {expected}"
                )

    @property
    def shape(self) -> WorkhorseCompileShapeV2:
        return workhorse_shape_v2(self.shape_id)

    @property
    def backend(self) -> WorkhorseBackendV2:
        return workhorse_backend_v2(self.backend_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "eiel_full_evolution_cuda_workhorse_contract_v2",
            "profile": self.profile,
            "profile_id": self.profile_id,
            "route_id": self.route_id,
            "native_abi": self.native_abi,
            "event_abi": self.event_abi,
            "time_authority": self.time_authority,
            "semantic_transition_id": self.semantic_transition_id,
            "semantic_transition_sha256": self.semantic_transition_sha256,
            "backend_id": self.backend_id,
            "backend_topology_sha256": self.backend_topology_sha256,
            "capabilities": list(self.capabilities),
            **self.shape.identity_fields(),
            "B": self.B,
            "temporal_k": self.temporal_k,
            "total_ticks": self.total_ticks,
            "start_next_tick": self.start_next_tick,
        }

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any]
    ) -> "FullEvolutionCudaWorkhorseContractV2":
        _reject_generation_keys(value)
        allowed = {
            "schema_version",
            "profile",
            "profile_id",
            "route_id",
            "native_abi",
            "event_abi",
            "time_authority",
            "semantic_transition_id",
            "semantic_transition_sha256",
            "backend_id",
            "backend_topology_sha256",
            "capabilities",
            "shape_id",
            "width",
            "height",
            "sites",
            "initial_population",
            "capacity",
            "index_dtype",
            "index_width_bits",
            "threads",
            "site_tiles",
            "agent_tiles",
            "rank_tiles",
            "temporal_k_candidates",
            "batch_seed",
            "B",
            "temporal_k",
            "total_ticks",
            "start_next_tick",
        }
        unknown = set(value) - allowed
        if unknown:
            raise FullEvolutionWorkhorseContractError(
                "unsupported workhorse contract keys: " + ", ".join(sorted(unknown))
            )
        missing = {"shape_id", "backend_id", "B", "temporal_k", "total_ticks"} - set(
            value
        )
        if missing:
            raise FullEvolutionWorkhorseContractError(
                "missing workhorse contract keys: " + ", ".join(sorted(missing))
            )
        shape = workhorse_shape_v2(value["shape_id"])
        for key, expected in shape.identity_fields().items():
            if key in value and value[key] != expected:
                raise FullEvolutionWorkhorseContractError(
                    f"{key} must equal {expected}"
                )
        if (
            "schema_version" in value
            and value["schema_version"]
            != "eiel_full_evolution_cuda_workhorse_contract_v2"
        ):
            raise FullEvolutionWorkhorseContractError(
                "workhorse contract schema mismatch"
            )
        capabilities = value.get("capabilities", FULL_EVOLUTION_CAPABILITIES_V2)
        if type(capabilities) is list:
            capabilities = tuple(capabilities)
        return cls(
            shape_id=value["shape_id"],
            backend_id=value["backend_id"],
            B=value["B"],
            temporal_k=value["temporal_k"],
            total_ticks=value["total_ticks"],
            start_next_tick=value.get("start_next_tick", 0),
            profile=value.get("profile", WORKHORSE_PROFILE_FAMILY_V2),
            profile_id=value.get("profile_id"),
            route_id=value.get("route_id"),
            native_abi=value.get("native_abi", WORKHORSE_ABI_V2),
            event_abi=value.get("event_abi", WORKHORSE_EVENT_ABI_V3),
            time_authority=value.get("time_authority", WORKHORSE_TIME_AUTHORITY_V2),
            semantic_transition_id=value.get(
                "semantic_transition_id", WORKHORSE_TRANSITION_ID_V2
            ),
            semantic_transition_sha256=value.get(
                "semantic_transition_sha256", WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2
            ),
            backend_topology_sha256=value.get("backend_topology_sha256"),
            capabilities=capabilities,
        )


def workhorse_artifact_identity_v2(
    shape: str | WorkhorseCompileShapeV2, backend_id: str
) -> dict[str, Any]:
    if isinstance(shape, str):
        shape = workhorse_shape_v2(shape)
    if workhorse_shape_v2(shape.shape_id) != shape:
        raise FullEvolutionWorkhorseContractError(
            "shape must be an exact reviewed specialization"
        )
    backend = workhorse_backend_v2(backend_id)
    # Local import avoids making the host contract depend on CUDA or creating
    # an import-time cycle while still sealing the concrete byte layout.
    from .layout import workhorse_layout_v2

    layout_sha256 = workhorse_layout_v2(shape).layout_sha256
    payload = {
        "schema_version": "eiel_full_evolution_cuda_workhorse_artifact_identity_v2",
        "profile": WORKHORSE_PROFILE_FAMILY_V2,
        **shape.identity_fields(),
        "native_abi": WORKHORSE_ABI_V2,
        "event_abi": WORKHORSE_EVENT_ABI_V3,
        "semantic_transition_id": WORKHORSE_TRANSITION_ID_V2,
        "semantic_transition_sha256": WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2,
        "backend_id": backend.backend_id,
        "backend_topology": backend.topology_payload(shape),
        "backend_topology_sha256": backend.topology_sha256(shape),
        "layout_sha256": layout_sha256,
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


__all__ = [
    "BACKEND_FUSED_CTA_TEMPORAL_V2",
    "BACKEND_PHASE_GRAPH_V2",
    "BACKEND_REFERENCE_PHASED_V2",
    "FULL_EVOLUTION_CAPABILITIES_V2",
    "FullEvolutionCudaWorkhorseContractV2",
    "FullEvolutionWorkhorseContractError",
    "MAX_SIGNED_INT64",
    "SUPPORTED_WORKHORSE_SHAPES_BY_ID_V2",
    "SUPPORTED_WORKHORSE_SHAPES_V2",
    "WORKHORSE_ABI_V2",
    "WORKHORSE_BACKENDS_BY_ID_V2",
    "WORKHORSE_BACKENDS_V2",
    "WORKHORSE_CHECKPOINT_SCHEMA_V2",
    "WORKHORSE_EMPTY_INDEX_V2",
    "WORKHORSE_EVENT_ABI_V2",
    "WORKHORSE_EVENT_ABI_V3",
    "WORKHORSE_EVENT_ROW_BYTES_V2",
    "WORKHORSE_INDEX_DTYPE_V2",
    "WORKHORSE_INDEX_WIDTH_BITS_V2",
    "WORKHORSE_MAX_GRID_Y_V2",
    "WORKHORSE_PROFILE_FAMILY_V2",
    "WORKHORSE_PROFILE_ID_PREFIX_V2",
    "WORKHORSE_ROUTE_PREFIX_V2",
    "WORKHORSE_REJECTED_SEMANTIC_HASHES_V2",
    "WORKHORSE_SEMANTIC_TRANSITION_SHA256_V2",
    "WORKHORSE_THREADS_V2",
    "WORKHORSE_TEMPORAL_K_CANDIDATES_V2",
    "WORKHORSE_TIME_AUTHORITY_V2",
    "WORKHORSE_TRANSITION_ID_V2",
    "WORKHORSE_WORLD_IDENTITY_BINDING_SCHEMA_V1",
    "WorkhorseBackendV2",
    "WorkhorseCompileShapeV2",
    "canonical_sha256",
    "checked_add_i64",
    "checked_multiply_i64",
    "workhorse_artifact_identity_v2",
    "workhorse_backend_v2",
    "workhorse_backend_temporal_k_candidates_v2",
    "workhorse_shape_v2",
    "workhorse_semantic_contract_v2",
    "workhorse_world_identity_binding_v2",
    "validate_workhorse_world_identity_binding_v2",
]
