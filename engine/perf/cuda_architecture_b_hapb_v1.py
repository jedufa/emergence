"""Fail-closed ownership contract for the B48 WDL/Jungle/EED HAPB route.

This module intentionally owns no native dispatch.  It is the H2 boundary
that makes a full resident CUDA policy pack impossible to pass as HAPB state.
The later CUDA export must consume this contract rather than reinterpret the
resident owner's ``mlp_pack`` argument.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
import hashlib
import os
from typing import Final, Iterator

import torch


HAPB_ABI_VERSION_V1: Final[str] = "architecture_b_wdl_jungle_eed_hapb_fixed_mlp_native_v1"
HAPB_ENTRYPOINT_V1: Final[str] = "eiel_cuda_architecture_b_hapb_compact_policy_logits_v1"
HAPB_DIRECT_ENTRYPOINT_V1: Final[str] = (
    "eiel_cuda_architecture_b_execute_direct_resident_wdl_jungle_eed_hapb_v1"
)
CELL_COUNT_V1: Final[int] = 48
HAPB_PHYSICAL_CELL_COUNT_MIN_V1: Final[int] = 1
HAPB_PHYSICAL_CELL_COUNT_MAX_V1: Final[int] = 256
HAPB_PRODUCTION_ROUTE_RETIRED_PERF_B_V1: Final[bool] = True
HAPB_PRODUCTION_ROUTE_STATUS_V1: Final[str] = "HAPB-PERF-B"
SLOT_CAPACITY_V1: Final[int] = 512
HOT_PREFIX_ROWS_V1: Final[int] = 256
POLICY_WIDTH_V1: Final[int] = 9_991
POLICY_ROW_BYTES_V1: Final[int] = POLICY_WIDTH_V1 * 4
POLICY_CROSSOVER_PACKED_WIDTH_V1: Final[int] = (POLICY_WIDTH_V1 + 7) // 8
CONTENT_POOL_ROWS_PER_CELL_V1: Final[int] = HOT_PREFIX_ROWS_V1
HAPB_TRAFFIC_COUNTER_SCHEMA_V1: Final[str] = (
    "eiel_cuda_architecture_b_hapb_policy_traffic_counters_v1"
)
# The first four counters are the primary, physical policy-byte accounting
# surface.  The remaining slots are reserved for the post-alias boundary
# instrumentation gate and are deliberately kept separate from policy bytes.
HAPB_TRAFFIC_COUNTER_POLICY_H2D_ROWS_V1: Final[int] = 0
HAPB_TRAFFIC_COUNTER_POLICY_H2D_BYTES_V1: Final[int] = 1
HAPB_TRAFFIC_COUNTER_POLICY_D2H_ROWS_V1: Final[int] = 2
HAPB_TRAFFIC_COUNTER_POLICY_D2H_BYTES_V1: Final[int] = 3
HAPB_TRAFFIC_COUNTER_CONTROL_D2H_BYTES_V1: Final[int] = 4
HAPB_TRAFFIC_COUNTER_CONTROL_H2D_BYTES_V1: Final[int] = 5
HAPB_TRAFFIC_COUNTER_SYNC_COUNT_V1: Final[int] = 6
# 213 alias-ON H4: two same-process G0-G3 runs diverged on cell 23
# (G3 ≠ checkpoint). Batched birth copies H2D then mutates without the
# per-cell stream fence. Production stays per-cell until an exact A/B
# re-earns KEEP. Opt in with EIEL_HAPB_FRAG_A_BATCHED_BIRTH=1.
HAPB_FRAG_A_BATCHED_BIRTH_ENABLED_V1: Final[bool] = False
HAPB_FRAG_A_PER_CELL_BIRTH_ENV_V1: Final[str] = "EIEL_HAPB_FRAG_A_PER_CELL_BIRTH"
HAPB_FRAG_A_BATCHED_BIRTH_ENV_V1: Final[str] = "EIEL_HAPB_FRAG_A_BATCHED_BIRTH"
_HAPB_FRAG_A_BIRTH_COUNTERS_V1: dict[str, int] = {
    "launches": 0,
    "device_syncs": 0,
    "active_cells": 0,
    "birth_rows": 0,
    "chunks": 0,
}
HAPB_TRAFFIC_COUNTER_GPU_STALL_NS_V1: Final[int] = 7
HAPB_TRAFFIC_COUNTER_COUNT_V1: Final[int] = 8
HAPB_POLICY_CHECKPOINT_SCHEMA_V1: Final[str] = (
    "eiel_cuda_architecture_b_hapb_policy_bank_checkpoint_v1"
)
# The v2 checkpoint adds only the compact, host-owned alias transaction state
# needed for an exact fresh-owner continuation.  It still serializes the
# canonical policy bank exactly once and never serializes a CUDA cache.
HAPB_POLICY_CHECKPOINT_SCHEMA_V2: Final[str] = (
    "eiel_cuda_architecture_b_hapb_policy_bank_checkpoint_v2"
)


class HAPBContractError(ValueError):
    """The distinct HAPB owner is malformed or would retain a CUDA bank."""


def hapb_physical_cell_count_v1(value: object) -> int:
    """Return a legal physical B. No multiple-of-8 restriction.

    ``CELL_COUNT_V1`` remains the frozen B48 oracle size. High-B uses the
    leading extent of the canonical bank or an explicit integer.
    """

    if isinstance(value, torch.Tensor):
        if value.ndim < 1:
            raise HAPBContractError("HAPB physical cell count requires a batched tensor")
        cells = int(value.size(0))
    elif type(value) is int:
        cells = value
    else:
        raise HAPBContractError("HAPB physical cell count must be an int or batched tensor")
    if (
        cells < HAPB_PHYSICAL_CELL_COUNT_MIN_V1
        or cells > HAPB_PHYSICAL_CELL_COUNT_MAX_V1
    ):
        raise HAPBContractError(
            "HAPB physical cell count must be in "
            f"[{HAPB_PHYSICAL_CELL_COUNT_MIN_V1}, {HAPB_PHYSICAL_CELL_COUNT_MAX_V1}]"
        )
    return cells


def hapb_production_route_admission_v1() -> dict[str, object]:
    """Production HAPB route is retired after Gate 8 HAPB-PERF-B.

    Engineering H4/canary probes remain allowed. Resident B48 stays the
    production route. This does not delete Git history or evidence.
    """

    return {
        "schema_version": "eiel_hapb_production_route_admission_v1",
        "admitted": False,
        "retired": HAPB_PRODUCTION_ROUTE_RETIRED_PERF_B_V1,
        "status": HAPB_PRODUCTION_ROUTE_STATUS_V1,
        "reason": (
            "official exclusive B48/B60/B96 vs Resident 219 is 0.867x; "
            "1.25x is unmet; B96 fails the 10% VRAM reserve so B120 was not launched; "
            "host-gather removable share is 5.0% of wall (NO-GO <30.6%); "
            "B96 Nsight is not required"
        ),
        "keep_resident_b48": True,
        "engineering_probe_allowed": True,
        "qualification_claim": False,
        "production_execution_admission": False,
        "gates_9_to_13_authorized": False,
    }


def hapb_canonical_bank_shape_v1(cells: object) -> tuple[int, int, int]:
    return (
        hapb_physical_cell_count_v1(cells),
        SLOT_CAPACITY_V1,
        POLICY_WIDTH_V1,
    )


def hapb_require_canonical_bank_layout_v1(
    bank: torch.Tensor,
    *,
    name: str = "HAPB canonical policy bank",
) -> int:
    if not isinstance(bank, torch.Tensor):
        raise HAPBContractError(f"{name} must be a tensor")
    if bank.ndim != 3 or tuple(bank.shape[1:]) != (SLOT_CAPACITY_V1, POLICY_WIDTH_V1):
        raise HAPBContractError(f"{name} shape is not exact C512/9991 policy rows")
    return hapb_physical_cell_count_v1(bank)


def _content_row_map_from_canonical_v1(
    canonical_host_bank: torch.Tensor,
) -> torch.Tensor:
    """Build a deterministic logical-slot to hot-pool relationship.

    The B48 alias rescue keeps the existing per-cell P256 CUDA pool.  Prefix
    rows are stable first-occurrence content identities; inactive overflow
    rows are left unbound until an exact ordinary-birth transaction publishes
    their parent identity.  A non-identical active overflow row therefore
    fails closed instead of silently retransferring or evicting content.
    """

    cells = hapb_require_canonical_bank_layout_v1(
        canonical_host_bank, name="HAPB alias map"
    )
    result = torch.full(
        (cells, SLOT_CAPACITY_V1),
        -1,
        dtype=torch.int64,
        device="cpu",
    )
    for cell in range(cells):
        # The physical pool is the existing hot-prefix cache, flattened by
        # cell.  Exact duplicate prefix rows share their first stable identity.
        first_by_digest: dict[bytes, int] = {}
        for slot in range(HOT_PREFIX_ROWS_V1):
            row = canonical_host_bank[cell, slot]
            digest = hashlib.sha256(row.numpy().tobytes(order="C")).digest()
            content_row = cell * CONTENT_POOL_ROWS_PER_CELL_V1 + slot
            first_by_digest.setdefault(digest, content_row)
            result[cell, slot] = content_row
        for slot in range(HOT_PREFIX_ROWS_V1, SLOT_CAPACITY_V1):
            row = canonical_host_bank[cell, slot]
            digest = hashlib.sha256(row.numpy().tobytes(order="C")).digest()
            content_row = first_by_digest.get(digest)
            if content_row is not None:
                result[cell, slot] = content_row
    return result.contiguous()


def _validate_content_row_map_v1(
    content_row_map: torch.Tensor,
) -> None:
    if (
        not isinstance(content_row_map, torch.Tensor)
        or content_row_map.device.type != "cpu"
        or content_row_map.dtype != torch.int64
        or not content_row_map.is_contiguous()
        or content_row_map.ndim != 2
        or content_row_map.shape[1] != SLOT_CAPACITY_V1
    ):
        raise HAPBContractError("HAPB content alias map must be pageable CPU [B,C512]")
    cells = hapb_physical_cell_count_v1(content_row_map)
    values = content_row_map.tolist()
    limit = cells * CONTENT_POOL_ROWS_PER_CELL_V1
    for cell, row in enumerate(values):
        for slot, raw_content_row in enumerate(row):
            content_row = int(raw_content_row)
            if content_row == -1:
                continue
            if not 0 <= content_row < limit:
                raise HAPBContractError("HAPB content alias map contains an invalid pool row")
            if content_row // CONTENT_POOL_ROWS_PER_CELL_V1 != cell:
                raise HAPBContractError("HAPB content alias map crosses a cell pool")


def _new_hapb_traffic_counters_v1() -> torch.Tensor:
    return torch.zeros(
        (HAPB_TRAFFIC_COUNTER_COUNT_V1,), dtype=torch.int64, device="cpu"
    ).contiguous()


def _traffic_snapshot_v1(counters: torch.Tensor) -> dict[str, int | str]:
    if (
        not isinstance(counters, torch.Tensor)
        or counters.device.type != "cpu"
        or counters.dtype != torch.int64
        or tuple(counters.shape) != (HAPB_TRAFFIC_COUNTER_COUNT_V1,)
    ):
        raise HAPBContractError("HAPB policy traffic counters are malformed")
    values = [int(value) for value in counters.tolist()]
    return {
        "schema_version": HAPB_TRAFFIC_COUNTER_SCHEMA_V1,
        "policy_row_bytes": POLICY_ROW_BYTES_V1,
        "policy_h2d_rows": values[HAPB_TRAFFIC_COUNTER_POLICY_H2D_ROWS_V1],
        "policy_h2d_bytes": values[HAPB_TRAFFIC_COUNTER_POLICY_H2D_BYTES_V1],
        "policy_d2h_rows": values[HAPB_TRAFFIC_COUNTER_POLICY_D2H_ROWS_V1],
        "policy_d2h_bytes": values[HAPB_TRAFFIC_COUNTER_POLICY_D2H_BYTES_V1],
        "control_d2h_bytes": values[HAPB_TRAFFIC_COUNTER_CONTROL_D2H_BYTES_V1],
        "control_h2d_bytes": values[HAPB_TRAFFIC_COUNTER_CONTROL_H2D_BYTES_V1],
        "sync_count": values[HAPB_TRAFFIC_COUNTER_SYNC_COUNT_V1],
        "gpu_stall_ns": values[HAPB_TRAFFIC_COUNTER_GPU_STALL_NS_V1],
    }


def _serialize_hapb_int64_state_v1(
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
) -> dict[str, object]:
    """Serialize one compact pageable HAPB transaction tensor exactly once."""

    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.device.type != "cpu"
        or tensor.dtype != torch.int64
        or tensor.is_pinned()
        or not tensor.is_contiguous()
        or tuple(tensor.shape) != shape
    ):
        raise HAPBContractError("HAPB transient checkpoint tensor is malformed")
    payload = tensor.numpy().tobytes(order="C")
    return {
        "shape": list(shape),
        "dtype": "int64",
        "byte_count": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "payload": payload,
    }


def _restore_hapb_int64_state_v1(
    state: Mapping[str, object],
    *,
    name: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Restore one compact HAPB transaction tensor onto pageable CPU."""

    entry = state.get(name)
    if not isinstance(entry, Mapping):
        raise HAPBContractError(f"HAPB checkpoint lacks transient state {name}")
    payload = entry.get("payload")
    expected_bytes = 8
    for dimension in shape:
        expected_bytes *= dimension
    if (
        entry.get("shape") != list(shape)
        or entry.get("dtype") != "int64"
        or entry.get("byte_count") != expected_bytes
        or not isinstance(payload, bytes)
        or len(payload) != expected_bytes
        or entry.get("sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise HAPBContractError(f"HAPB transient state {name} bytes or digest are invalid")
    restored = torch.frombuffer(
        bytearray(payload), dtype=torch.int64, count=expected_bytes // 8
    ).reshape(shape).contiguous()
    if restored.device.type != "cpu" or restored.is_pinned():
        raise HAPBContractError(f"HAPB transient state {name} is not pageable CPU")
    return restored


def _content_alias_relation_sha256_state_v1(
    *,
    enabled: bool,
    content_row_map: torch.Tensor,
    content_row_versions: torch.Tensor,
    slot_generations: torch.Tensor,
) -> str:
    """Digest compact alias state without requiring a CUDA owner instance."""

    if not enabled:
        return "disabled"
    cells = hapb_physical_cell_count_v1(content_row_map)
    h = hashlib.sha256()
    for cell in range(cells):
        for slot in range(SLOT_CAPACITY_V1):
            h.update(cell.to_bytes(2, "little"))
            h.update(slot.to_bytes(2, "little"))
            h.update(int(slot_generations[cell, slot]).to_bytes(8, "little"))
            h.update(int(content_row_map[cell, slot]).to_bytes(8, "little", signed=True))
            h.update(int(content_row_versions[cell, slot]).to_bytes(8, "little"))
    return h.hexdigest()


def _restore_hapb_checkpoint_transient_state_v1(
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    """Validate and restore the compact host-owned state of a v2 checkpoint."""

    state = checkpoint.get("transient_state")
    if not isinstance(state, Mapping):
        raise HAPBContractError("HAPB checkpoint lacks its compact transient state")
    enabled = state.get("content_alias_enabled")
    pool_epoch = state.get("pool_epoch")
    if type(enabled) is not bool or type(pool_epoch) is not int or pool_epoch < 1:
        raise HAPBContractError("HAPB checkpoint alias settings are malformed")
    map_entry = state.get("content_row_map")
    if not isinstance(map_entry, Mapping) or not isinstance(map_entry.get("shape"), list):
        raise HAPBContractError("HAPB checkpoint lacks transient state content_row_map")
    map_shape = tuple(int(dimension) for dimension in map_entry["shape"])
    if len(map_shape) != 2 or map_shape[1] != SLOT_CAPACITY_V1:
        raise HAPBContractError("HAPB checkpoint alias map is not [B,C512]")
    hapb_physical_cell_count_v1(map_shape[0])
    content_row_map = _restore_hapb_int64_state_v1(
        state, name="content_row_map", shape=map_shape
    )
    content_row_versions = _restore_hapb_int64_state_v1(
        state, name="content_row_versions", shape=map_shape
    )
    slot_generations = _restore_hapb_int64_state_v1(
        state, name="slot_generations", shape=map_shape
    )
    _validate_content_row_map_v1(content_row_map)
    if bool(torch.any(content_row_versions < 1).item()):
        raise HAPBContractError("HAPB checkpoint contains an invalid alias version")
    if bool(torch.any(slot_generations < 0).item()):
        raise HAPBContractError("HAPB checkpoint contains an invalid slot generation")
    relation = _content_alias_relation_sha256_state_v1(
        enabled=enabled,
        content_row_map=content_row_map,
        content_row_versions=content_row_versions,
        slot_generations=slot_generations,
    )
    rebuild = checkpoint.get("content_alias_rebuild")
    expected_relation = rebuild.get("relation_sha256") if isinstance(rebuild, Mapping) else None
    if expected_relation != relation:
        raise HAPBContractError("HAPB checkpoint alias relation digest is invalid")
    return {
        "content_alias_enabled": enabled,
        "pool_epoch": pool_epoch,
        "content_row_map": content_row_map,
        "content_row_versions": content_row_versions,
        "slot_generations": slot_generations,
    }


def export_hapb_policy_bank_checkpoint_v1(
    owner: "HostAuthoritativePolicyBankV1",
) -> dict[str, object]:
    """Serialize one drained canonical host bank, excluding all CUDA caches."""

    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise HAPBContractError("HAPB checkpoint requires its sole policy owner")
    if _ACTIVE_POLICY_CONTEXT_V1.get() is not None:
        raise HAPBContractError("HAPB checkpoint refuses an open policy transaction")
    bank = owner.canonical_host_bank
    cells = hapb_require_canonical_bank_layout_v1(bank)
    bank_bytes = bank.numpy().tobytes(order="C")
    payload: dict[str, object] = {
        "schema_version": HAPB_POLICY_CHECKPOINT_SCHEMA_V2,
        "shape": [cells, SLOT_CAPACITY_V1, POLICY_WIDTH_V1],
        "dtype": "float32",
        "byte_count": len(bank_bytes),
        "sha256": hashlib.sha256(bank_bytes).hexdigest(),
        "canonical_host_policy_bank": bank_bytes,
        "excluded_transient_state": [
            "hot_prefix_cache",
            "cuda_tile_arena",
            "pinned_staging_tile",
            "stable_row_map",
            "policy_traffic_counters",
            "jungle_pre_predator_policy_cursor",
        ],
        # These are the compact host transaction coordinates, not a second
        # policy bank.  They must travel with the canonical bytes so a fresh
        # owner resumes the exact alias/version/generation trajectory.
        "transient_state": {
            "content_alias_enabled": owner.content_alias_enabled,
            "pool_epoch": owner.pool_epoch,
            "content_row_map": _serialize_hapb_int64_state_v1(
                owner.content_row_map,
                shape=(cells, SLOT_CAPACITY_V1),
            ),
            "content_row_versions": _serialize_hapb_int64_state_v1(
                owner.content_row_versions,
                shape=(cells, SLOT_CAPACITY_V1),
            ),
            "slot_generations": _serialize_hapb_int64_state_v1(
                owner.slot_generations,
                shape=(cells, SLOT_CAPACITY_V1),
            ),
        },
        "content_alias_rebuild": {
            "enabled": owner.content_alias_enabled,
            "pool_rows_per_cell": CONTENT_POOL_ROWS_PER_CELL_V1,
            "relation_sha256": owner.content_alias_relation_sha256_v1(),
        },
    }
    if owner.hapb_rng_semantics_version == 2 or owner.hapb_rngv2_frontier is not None:
        payload.update(_export_hapb_rngv2_frontier_v1(owner))
    return payload


def bind_hapb_rngv2_frontier_v1(
    owner: "HostAuthoritativePolicyBankV1",
    *,
    identity: object,
    world_uids: tuple[int, ...],
    absolute_tick: int,
) -> "HostAuthoritativePolicyBankV1":
    """Attach a V2 semantic frontier to the live HAPB policy owner."""

    from engine.hapb_rngv2.checkpoint import capture_hapb_rngv2_checkpoint
    from engine.hapb_rngv2.identity import HapbIdentityMap

    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise HAPBContractError("HAPB RNGV2 bind requires the policy owner")
    if not isinstance(identity, HapbIdentityMap):
        raise HAPBContractError("HAPB RNGV2 bind requires HapbIdentityMap")
    frontier = capture_hapb_rngv2_checkpoint(
        identity=identity,
        world_uids=world_uids,
        absolute_tick=absolute_tick,
    ).to_dict()
    return replace(
        owner,
        hapb_rng_semantics_version=2,
        hapb_rngv2_frontier=frontier,
    )


def _export_hapb_rngv2_frontier_v1(
    owner: "HostAuthoritativePolicyBankV1",
) -> dict[str, object]:
    from engine.hapb_rngv2.checkpoint import capture_hapb_rngv2_checkpoint
    from engine.hapb_rngv2.identity import HapbIdentityMap

    frontier = owner.hapb_rngv2_frontier
    if frontier is None:
        cells = hapb_physical_cell_count_v1(owner.canonical_host_bank)
        identity = HapbIdentityMap(cells=cells, slots=SLOT_CAPACITY_V1)
        frontier = capture_hapb_rngv2_checkpoint(
            identity=identity,
            world_uids=tuple(range(cells)),
            absolute_tick=0,
        ).to_dict()
    if int(frontier.get("host_tape_bytes", 0)) != 0:
        raise HAPBContractError("HAPB RNGV2 frontier cannot carry host tape bytes")
    if int(frontier.get("host_gaussian_cache_bytes", 0)) != 0:
        raise HAPBContractError("HAPB RNGV2 frontier cannot carry a CPython gauss cache")
    return {
        "hapb_rng_semantics_version": 2,
        "hapb_rngv2_frontier": frontier,
    }


def restore_hapb_rngv2_frontier_v1(
    checkpoint: Mapping[str, object],
) -> dict[str, object] | None:
    """Restore the V2 frontier from a HAPB policy-bank checkpoint payload."""

    if checkpoint.get("hapb_rng_semantics_version", 1) != 2:
        if checkpoint.get("hapb_rngv2_frontier") is not None:
            raise HAPBContractError("legacy HAPB checkpoint cannot carry an RNGV2 frontier")
        return None
    raw = checkpoint.get("hapb_rngv2_frontier")
    if not isinstance(raw, Mapping):
        raise HAPBContractError("HAPB RNGV2 checkpoint lacks its semantic frontier")
    from engine.hapb_rngv2.checkpoint import restore_hapb_rngv2_checkpoint

    restored = restore_hapb_rngv2_checkpoint(dict(raw))
    return restored.to_dict()


def _resolve_hapb_owner_rngv2_init_v1(
    *,
    cells: int,
    hapb_rng_semantics_version: int,
    hapb_rngv2_frontier: dict[str, object] | None,
) -> tuple[int, dict[str, object] | None]:
    """Select V2 owner state when the HAPB RNGV2 route is active."""

    from engine.hapb_rngv2.route import hapb_rngv2_env_enabled

    version = int(hapb_rng_semantics_version)
    if version == 1 and hapb_rngv2_env_enabled():
        version = 2
    if version != 2:
        if hapb_rngv2_frontier is not None:
            raise HAPBContractError("legacy HAPB owner cannot carry an RNGV2 frontier")
        return 1, None
    if hapb_rngv2_frontier is not None:
        if int(hapb_rngv2_frontier.get("host_tape_bytes", 0)) != 0:
            raise HAPBContractError("HAPB RNGV2 frontier cannot carry host tape bytes")
        return 2, hapb_rngv2_frontier
    from engine.hapb_rngv2.checkpoint import capture_hapb_rngv2_checkpoint
    from engine.hapb_rngv2.identity import HapbIdentityMap

    identity = HapbIdentityMap(cells=cells, slots=SLOT_CAPACITY_V1)
    return 2, capture_hapb_rngv2_checkpoint(
        identity=identity,
        world_uids=tuple(range(cells)),
        absolute_tick=0,
    ).to_dict()


def restore_hapb_policy_bank_checkpoint_v1(
    checkpoint: object,
) -> torch.Tensor:
    """Restore only the canonical pageable bank; no CUDA allocation occurs."""

    if not isinstance(checkpoint, dict):
        raise HAPBContractError("HAPB checkpoint payload must be a mapping")
    if checkpoint.get("schema_version") != HAPB_POLICY_CHECKPOINT_SCHEMA_V2:
        raise HAPBContractError("HAPB checkpoint schema is incompatible")
    raw_shape = checkpoint.get("shape")
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 3
        or raw_shape[1:] != [SLOT_CAPACITY_V1, POLICY_WIDTH_V1]
    ):
        raise HAPBContractError("HAPB checkpoint policy shape is incompatible")
    cells = hapb_physical_cell_count_v1(int(raw_shape[0]))
    if checkpoint.get("dtype") != "float32":
        raise HAPBContractError("HAPB checkpoint policy dtype is incompatible")
    payload = checkpoint.get("canonical_host_policy_bank")
    expected_bytes = cells * SLOT_CAPACITY_V1 * POLICY_WIDTH_V1 * 4
    if (
        not isinstance(payload, bytes)
        or checkpoint.get("byte_count") != expected_bytes
        or len(payload) != expected_bytes
        or checkpoint.get("sha256") != hashlib.sha256(payload).hexdigest()
    ):
        raise HAPBContractError("HAPB checkpoint policy bytes or digest are invalid")
    restored = torch.frombuffer(
        bytearray(payload), dtype=torch.float32, count=expected_bytes // 4
    ).reshape(cells, SLOT_CAPACITY_V1, POLICY_WIDTH_V1)
    if restored.is_pinned() or restored.device.type != "cpu":
        raise HAPBContractError("HAPB checkpoint restore did not produce pageable CPU authority")
    return restored


def restore_hapb_policy_owner_checkpoint_v1(
    checkpoint: object,
    *,
    device: torch.device | str,
) -> "HostAuthoritativePolicyBankV1":
    """Restore host authority and compact alias state with CUDA caches invalid.

    Newly allocated hot-prefix or tile storage is never treated as restored
    state: the direct route must explicitly rehydrate the hot prefix through
    its bounded staged path before it can consume any cached policy row.
    """

    if not isinstance(checkpoint, Mapping):
        raise HAPBContractError("HAPB checkpoint payload must be a mapping")
    canonical = restore_hapb_policy_bank_checkpoint_v1(checkpoint)
    transient = _restore_hapb_checkpoint_transient_state_v1(checkpoint)
    frontier = restore_hapb_rngv2_frontier_v1(checkpoint)
    owner = initialize_selected_hapb_owner_v1(
        canonical,
        device=device,
        populate_hot_prefix=False,
        content_alias_enabled=transient["content_alias_enabled"],
        pool_epoch=transient["pool_epoch"],
        content_row_map=transient["content_row_map"],
        content_row_versions=transient["content_row_versions"],
        slot_generations=transient["slot_generations"],
        hapb_rng_semantics_version=2 if frontier is not None else 1,
        hapb_rngv2_frontier=frontier,
    )
    return owner


def restore_hapb_policy_owner_transient_state_v1(
    owner: "HostAuthoritativePolicyBankV1",
    checkpoint: Mapping[str, object],
) -> "HostAuthoritativePolicyBankV1":
    """Restore compact alias state into an already prepared CUDA owner.

    The CUDA hot-prefix/tile/staging allocations remain the prepared owner's
    allocations.  Only the small host transaction coordinates are replaced;
    the caller then performs the explicit bounded hot-prefix rehydrate.
    """

    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise HAPBContractError("HAPB transient restore requires its policy owner")
    if not isinstance(checkpoint, Mapping):
        raise HAPBContractError("HAPB checkpoint payload must be a mapping")
    transient = _restore_hapb_checkpoint_transient_state_v1(checkpoint)
    frontier = restore_hapb_rngv2_frontier_v1(checkpoint)
    if owner.hapb_rng_semantics_version == 2 and frontier is None:
        raise HAPBContractError("V2 HAPB owner cannot restore a checkpoint without a V2 frontier")
    restored = replace(
        owner,
        content_alias_enabled=transient["content_alias_enabled"],
        pool_epoch=transient["pool_epoch"],
        content_row_map=transient["content_row_map"],
        content_row_versions=transient["content_row_versions"],
        slot_generations=transient["slot_generations"],
        hapb_rng_semantics_version=2 if frontier is not None else 1,
        hapb_rngv2_frontier=frontier,
    )
    if restored.content_alias_relation_sha256_v1() != (
        checkpoint["content_alias_rebuild"]["relation_sha256"]
    ):
        raise HAPBContractError("HAPB transient restore changed alias relation")
    return restored


_ACTIVE_POLICY_CONTEXT_V1: ContextVar["HapbPolicyContextV1 | None"] = ContextVar(
    "active_hapb_policy_context_v1", default=None
)


def canonical_host_policy_bank_from_cell_rows_v1(
    cell_rows: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """Materialize the sole B48/C512 canonical policy authority on CPU.

    This is intentionally separate from the resident source allocator: callers
    must provide the individual checkpoint/source rows while they are still
    CPU tensors.  It rejects CUDA inputs rather than copying a resident bank
    back to host after the forbidden device materialization has occurred.
    """

    if len(cell_rows) < 1:
        raise HAPBContractError("HAPB canonical authority requires at least one cell row")
    cells = hapb_physical_cell_count_v1(len(cell_rows))
    expected = (SLOT_CAPACITY_V1, POLICY_WIDTH_V1)
    for cell, rows in enumerate(cell_rows):
        if (
            not isinstance(rows, torch.Tensor)
            or rows.device.type != "cpu"
            or rows.is_pinned()
            or rows.dtype != torch.float32
            or not rows.is_contiguous()
            or tuple(rows.shape) != expected
        ):
            raise HAPBContractError(
                f"HAPB canonical source row {cell} is not pageable CPU C512/9991"
            )
    canonical = torch.empty(
        hapb_canonical_bank_shape_v1(cells),
        dtype=torch.float32,
        device="cpu",
    )
    for cell, rows in enumerate(cell_rows):
        canonical[cell].copy_(rows)
    return canonical


def stable_overflow_row_map_v1(active_slots_by_cell: torch.Tensor) -> torch.Tensor:
    """Return one host-visible, stable, unique global overflow-row map.

    The order is the resident classifier's action-settlement order.  It must
    survive D2H, host gather, H2D, compact logits, and serial Gumbel settlement
    unchanged; sorting it by policy slot would change RNG/action association.
    ``-1`` is the sole inactive sentinel.
    """

    if (
        not isinstance(active_slots_by_cell, torch.Tensor)
        or active_slots_by_cell.device.type != "cpu"
        or active_slots_by_cell.dtype != torch.int64
        or active_slots_by_cell.ndim != 2
    ):
        raise HAPBContractError("HAPB active-slot discovery must be CPU int64 [B,N]")
    hapb_physical_cell_count_v1(active_slots_by_cell)
    output: list[int] = []
    for cell, raw_slots in enumerate(active_slots_by_cell.tolist()):
        slots = [int(slot) for slot in raw_slots if int(slot) != -1]
        if any(slot < 0 or slot >= SLOT_CAPACITY_V1 for slot in slots):
            raise HAPBContractError("HAPB active-slot discovery contains an invalid slot")
        if len(set(slots)) != len(slots):
            raise HAPBContractError("HAPB active-slot discovery contains duplicate slots")
        output.extend(
            cell * SLOT_CAPACITY_V1 + slot
            for slot in slots
            if slot >= HOT_PREFIX_ROWS_V1
        )
    return torch.tensor(output, dtype=torch.int64, device="cpu")


@dataclass(frozen=True, slots=True)
class HapbStablePolicyMapV1:
    """Stable overflow identity plus the resident action-output row coordinate."""

    global_slots: torch.Tensor
    active_row_indices: torch.Tensor


def stable_overflow_policy_map_v1(
    active_slots_by_cell: torch.Tensor,
) -> HapbStablePolicyMapV1:
    """Retain both coordinates needed by exact compact settlement.

    ``global_slots`` selects canonical policy bytes; ``active_row_indices``
    identifies the corresponding resident ordered-row/action output location.
    Conflating them is safe only for a full fixed C512 row list, not for the
    compact active policy lists produced at the prepare boundary.
    """

    rows = stable_overflow_row_map_v1(active_slots_by_cell)
    active_rows: list[int] = []
    for cell, raw_slots in enumerate(active_slots_by_cell.tolist()):
        for active_row, slot in enumerate(raw_slots):
            if int(slot) >= HOT_PREFIX_ROWS_V1:
                active_rows.append(cell * SLOT_CAPACITY_V1 + active_row)
    indices = torch.tensor(active_rows, dtype=torch.int64, device="cpu")
    if indices.numel() != rows.numel():
        raise HAPBContractError("HAPB stable policy map dimensions diverged")
    return HapbStablePolicyMapV1(rows, indices)


def initialize_selected_hapb_owner_v1(
    canonical_host_bank: torch.Tensor,
    *,
    device: torch.device | str,
    populate_hot_prefix: bool = True,
    content_alias_enabled: bool = True,
    pool_epoch: int = 1,
    tile_rows: int = 512,
    content_row_map: torch.Tensor | None = None,
    content_row_versions: torch.Tensor | None = None,
    slot_generations: torch.Tensor | None = None,
    hapb_rng_semantics_version: int = 1,
    hapb_rngv2_frontier: dict[str, object] | None = None,
) -> "HostAuthoritativePolicyBankV1":
    """Initialize the frozen P256/C512/one-buffer HAPB representation.

    Prefix population is deliberately serialized through the one bounded pinned
    tile.  A direct copy of the entire canonical bank to CUDA is forbidden,
    including during construction.  Restore passes ``False`` so its newly
    allocated cache is explicitly invalid until guarded rehydration.  The
    default remains the qualified 512-row tile; 4096 is exposed only as the
    bounded secondary timing candidate already admitted by the native ABI.
    """

    if type(content_alias_enabled) is not bool:
        raise HAPBContractError("HAPB content alias enable must be an exact bool")
    if type(pool_epoch) is not int or pool_epoch < 1:
        raise HAPBContractError("HAPB content alias pool epoch must be positive")
    if type(tile_rows) is not int or tile_rows not in {512, 4096}:
        raise HAPBContractError("HAPB tile rows must be 512 or 4096")
    if not isinstance(canonical_host_bank, torch.Tensor):
        raise HAPBContractError("HAPB canonical policy bank must be a tensor")
    if (canonical_host_bank.device.type != "cpu" or canonical_host_bank.is_pinned()
            or canonical_host_bank.dtype != torch.float32
            or not canonical_host_bank.is_contiguous()):
        raise HAPBContractError("HAPB canonical bank is not pageable exact C512 authority")
    cells = hapb_require_canonical_bank_layout_v1(
        canonical_host_bank, name="HAPB canonical bank"
    )
    hapb_rng_semantics_version, hapb_rngv2_frontier = (
        _resolve_hapb_owner_rngv2_init_v1(
            cells=cells,
            hapb_rng_semantics_version=hapb_rng_semantics_version,
            hapb_rngv2_frontier=hapb_rngv2_frontier,
        )
    )
    target = torch.device(device)
    if target.type != "cuda":
        raise HAPBContractError("HAPB selected representation requires one CUDA device")
    staging = torch.empty(
        (tile_rows, POLICY_WIDTH_V1), dtype=torch.float32, device="cpu", pin_memory=True
    )
    hot_prefix = torch.empty(
        (cells, HOT_PREFIX_ROWS_V1, POLICY_WIDTH_V1),
        dtype=torch.float32, device=target,
    )
    tile_arena = torch.empty(
        (tile_rows, POLICY_WIDTH_V1), dtype=torch.float32, device=target
    )
    if populate_hot_prefix:
        # Each 256-row prefix is copied through the one owned staging buffer.
        # The second half remains untouched and does not become a second buffer.
        for cell in range(cells):
            staging[:HOT_PREFIX_ROWS_V1].copy_(canonical_host_bank[cell, :HOT_PREFIX_ROWS_V1])
            hot_prefix[cell].copy_(staging[:HOT_PREFIX_ROWS_V1], non_blocking=True)
            torch.cuda.synchronize(target)
    owner = HostAuthoritativePolicyBankV1.create(
        canonical_host_bank,
        pinned_staging_buffers=(staging,),
        hot_prefix_cache=hot_prefix,
        tile_rows=tile_rows,
        cuda_tile_arena=tile_arena,
        hot_prefix_valid=populate_hot_prefix,
        content_alias_enabled=content_alias_enabled,
        pool_epoch=pool_epoch,
        content_row_map=content_row_map,
        content_row_versions=content_row_versions,
        slot_generations=slot_generations,
        hapb_rng_semantics_version=hapb_rng_semantics_version,
        hapb_rngv2_frontier=hapb_rngv2_frontier,
    )
    if populate_hot_prefix:
        owner.record_policy_h2d_rows_v1(cells * HOT_PREFIX_ROWS_V1)
    return owner


@dataclass(frozen=True, slots=True)
class HostAuthoritativePolicyBankV1:
    """One pageable CPU canonical bank plus bounded pinned staging ownership."""

    canonical_host_bank: torch.Tensor
    pinned_staging_buffers: tuple[torch.Tensor, ...]
    hot_prefix_cache: torch.Tensor | None
    cuda_tile_arena: torch.Tensor | None
    tile_rows: int
    hot_prefix_valid: bool
    content_row_map: torch.Tensor
    content_row_versions: torch.Tensor
    slot_generations: torch.Tensor
    pool_epoch: int
    content_alias_enabled: bool
    policy_traffic_counters: torch.Tensor
    hapb_rng_semantics_version: int = 1
    hapb_rngv2_frontier: dict[str, object] | None = None

    @property
    def physical_cell_count(self) -> int:
        return hapb_physical_cell_count_v1(self.canonical_host_bank)

    @classmethod
    def create(
        cls,
        canonical_host_bank: torch.Tensor,
        *,
        pinned_staging_buffers: tuple[torch.Tensor, ...],
        hot_prefix_cache: torch.Tensor | None,
        tile_rows: int,
        cuda_tile_arena: torch.Tensor | None = None,
        hot_prefix_valid: bool = True,
        content_alias_enabled: bool = True,
        pool_epoch: int = 1,
        content_row_map: torch.Tensor | None = None,
        content_row_versions: torch.Tensor | None = None,
        slot_generations: torch.Tensor | None = None,
        policy_traffic_counters: torch.Tensor | None = None,
        hapb_rng_semantics_version: int = 1,
        hapb_rngv2_frontier: dict[str, object] | None = None,
    ) -> "HostAuthoritativePolicyBankV1":
        if not isinstance(canonical_host_bank, torch.Tensor):
            raise HAPBContractError("HAPB canonical policy bank must be a tensor")
        if canonical_host_bank.device.type != "cpu" or canonical_host_bank.is_pinned():
            raise HAPBContractError("HAPB canonical policy bank must be pageable CPU authority")
        if canonical_host_bank.dtype != torch.float32 or not canonical_host_bank.is_contiguous():
            raise HAPBContractError("HAPB canonical policy bank must be contiguous float32")
        cells = hapb_require_canonical_bank_layout_v1(canonical_host_bank)
        if tile_rows not in {512, 4096}:
            raise HAPBContractError("HAPB tile rows must be 512 or 4096")
        if len(pinned_staging_buffers) != 1:
            raise HAPBContractError("frozen H1 policy requires exactly one staging buffer")
        for buffer in pinned_staging_buffers:
            if (not isinstance(buffer, torch.Tensor) or buffer.device.type != "cpu" or not buffer.is_pinned()
                    or buffer.dtype != torch.float32 or not buffer.is_contiguous()
                    or tuple(buffer.shape) != (tile_rows, POLICY_WIDTH_V1)):
                raise HAPBContractError("HAPB staging buffer is not a bounded pinned tile")
        if hot_prefix_cache is None:
            raise HAPBContractError("frozen H1 policy requires the noncanonical P256 hot prefix")
        if (hot_prefix_cache.device.type != "cuda" or hot_prefix_cache.dtype != torch.float32
                or not hot_prefix_cache.is_contiguous()
                or tuple(hot_prefix_cache.shape) != (cells, HOT_PREFIX_ROWS_V1, POLICY_WIDTH_V1)):
            raise HAPBContractError("HAPB hot prefix is not exact noncanonical B/P256 CUDA cache")
        if (not isinstance(cuda_tile_arena, torch.Tensor) or cuda_tile_arena.device.type != "cuda"
                or cuda_tile_arena.dtype != torch.float32 or not cuda_tile_arena.is_contiguous()
                or tuple(cuda_tile_arena.shape) != (tile_rows, POLICY_WIDTH_V1)):
            raise HAPBContractError("HAPB CUDA tile arena is not one bounded exact tile")
        if cuda_tile_arena.data_ptr() == hot_prefix_cache.data_ptr():
            raise HAPBContractError("HAPB tile arena must not alias the hot-prefix cache")
        if not isinstance(hot_prefix_valid, bool):
            raise HAPBContractError("HAPB hot-prefix validity must be explicit")
        if type(content_alias_enabled) is not bool:
            raise HAPBContractError("HAPB content alias enable must be an exact bool")
        if type(pool_epoch) is not int or pool_epoch < 1:
            raise HAPBContractError("HAPB content alias pool epoch must be positive")
        if content_row_map is None:
            content_row_map = _content_row_map_from_canonical_v1(canonical_host_bank)
        _validate_content_row_map_v1(content_row_map)
        if content_row_versions is None:
            content_row_versions = torch.ones(
                (cells, SLOT_CAPACITY_V1),
                dtype=torch.int64,
                device="cpu",
            )
        if (
            not isinstance(content_row_versions, torch.Tensor)
            or content_row_versions.device.type != "cpu"
            or content_row_versions.dtype != torch.int64
            or not content_row_versions.is_contiguous()
            or tuple(content_row_versions.shape) != (cells, SLOT_CAPACITY_V1)
            or bool(torch.any(content_row_versions < 1).item())
        ):
            raise HAPBContractError("HAPB content alias versions are malformed")
        if slot_generations is None:
            slot_generations = torch.zeros(
                (cells, SLOT_CAPACITY_V1),
                dtype=torch.int64,
                device="cpu",
            )
        if (
            not isinstance(slot_generations, torch.Tensor)
            or slot_generations.device.type != "cpu"
            or slot_generations.dtype != torch.int64
            or not slot_generations.is_contiguous()
            or tuple(slot_generations.shape) != (cells, SLOT_CAPACITY_V1)
            or bool(torch.any(slot_generations < 0).item())
        ):
            raise HAPBContractError("HAPB slot generations are malformed")
        if hapb_rng_semantics_version not in {1, 2}:
            raise HAPBContractError("HAPB RNG semantics version must be 1 or 2")
        if hapb_rng_semantics_version == 1 and hapb_rngv2_frontier is not None:
            raise HAPBContractError("legacy HAPB owner cannot carry an RNGV2 frontier")
        if policy_traffic_counters is None:
            policy_traffic_counters = _new_hapb_traffic_counters_v1()
        if (
            not isinstance(policy_traffic_counters, torch.Tensor)
            or policy_traffic_counters.device.type != "cpu"
            or policy_traffic_counters.dtype != torch.int64
            or not policy_traffic_counters.is_contiguous()
            or tuple(policy_traffic_counters.shape) != (HAPB_TRAFFIC_COUNTER_COUNT_V1,)
        ):
            raise HAPBContractError("HAPB policy traffic counters are malformed")
        return cls(
            canonical_host_bank,
            pinned_staging_buffers,
            hot_prefix_cache,
            cuda_tile_arena,
            tile_rows,
            hot_prefix_valid,
            content_row_map,
            content_row_versions,
            slot_generations,
            pool_epoch,
            content_alias_enabled,
            policy_traffic_counters,
            hapb_rng_semantics_version,
            hapb_rngv2_frontier,
        )

    def rehydrate_hot_prefix_cache_v1(self) -> "HostAuthoritativePolicyBankV1":
        """Rebuild the noncanonical P256 cache from sole host authority.

        This is the only legal post-restore cache reconstruction.  It drains
        the single staging tile after every cell so it cannot preserve a hidden
        second policy bank, and it refuses to mutate cache state during an open
        prepare/resume policy transaction.
        """

        if _ACTIVE_POLICY_CONTEXT_V1.get() is not None:
            raise HAPBContractError("HAPB hot-prefix rehydrate refuses an open policy transaction")
        staging = self.pinned_staging_buffers[0]
        cache = self.hot_prefix_cache
        if cache is None:
            raise HAPBContractError("HAPB hot-prefix cache allocation is unavailable")
        for cell in range(self.physical_cell_count):
            staging[:HOT_PREFIX_ROWS_V1].copy_(
                self.canonical_host_bank[cell, :HOT_PREFIX_ROWS_V1]
            )
            cache[cell].copy_(staging[:HOT_PREFIX_ROWS_V1], non_blocking=True)
            torch.cuda.synchronize(cache.device)
        self.record_policy_h2d_rows_v1(self.physical_cell_count * HOT_PREFIX_ROWS_V1)
        return replace(self, hot_prefix_valid=True)

    def require_hot_prefix_cache_v1(self) -> torch.Tensor:
        """Return cache bytes only after construction or explicit rehydration."""

        if not self.hot_prefix_valid or self.hot_prefix_cache is None:
            raise HAPBContractError("HAPB hot-prefix cache is invalid pending guarded rehydrate")
        return self.hot_prefix_cache

    def record_policy_h2d_rows_v1(self, rows: int) -> None:
        """Record physical policy-row H2D traffic at an admitted boundary."""

        if type(rows) is not int or rows < 0:
            raise HAPBContractError("HAPB policy H2D row count must be nonnegative")
        self.policy_traffic_counters[HAPB_TRAFFIC_COUNTER_POLICY_H2D_ROWS_V1] += rows
        self.policy_traffic_counters[HAPB_TRAFFIC_COUNTER_POLICY_H2D_BYTES_V1] += (
            rows * POLICY_ROW_BYTES_V1
        )

    def record_policy_d2h_rows_v1(self, rows: int) -> None:
        if type(rows) is not int or rows < 0:
            raise HAPBContractError("HAPB policy D2H row count must be nonnegative")
        self.policy_traffic_counters[HAPB_TRAFFIC_COUNTER_POLICY_D2H_ROWS_V1] += rows
        self.policy_traffic_counters[HAPB_TRAFFIC_COUNTER_POLICY_D2H_BYTES_V1] += (
            rows * POLICY_ROW_BYTES_V1
        )

    def record_control_d2h_bytes_v1(self, byte_count: int) -> None:
        """Record non-policy control bytes read back at a native boundary."""

        if type(byte_count) is not int or byte_count < 0:
            raise HAPBContractError(
                "HAPB control D2H byte count must be nonnegative"
            )
        self.policy_traffic_counters[HAPB_TRAFFIC_COUNTER_CONTROL_D2H_BYTES_V1] += (
            byte_count
        )

    def record_control_h2d_bytes_v1(self, byte_count: int) -> None:
        """Record non-policy control bytes sent to the native boundary."""

        if type(byte_count) is not int or byte_count < 0:
            raise HAPBContractError(
                "HAPB control H2D byte count must be nonnegative"
            )
        self.policy_traffic_counters[HAPB_TRAFFIC_COUNTER_CONTROL_H2D_BYTES_V1] += (
            byte_count
        )

    def policy_traffic_snapshot_v1(self) -> dict[str, int | str]:
        return _traffic_snapshot_v1(self.policy_traffic_counters)

    def content_alias_relation_sha256_v1(self) -> str:
        """Digest logical slots, generations, and content identities."""

        if not self.content_alias_enabled:
            return "disabled"

        h = hashlib.sha256()
        for cell in range(self.physical_cell_count):
            for slot in range(SLOT_CAPACITY_V1):
                h.update(cell.to_bytes(2, "little"))
                h.update(slot.to_bytes(2, "little"))
                h.update(int(self.slot_generations[cell, slot]).to_bytes(8, "little"))
                h.update(int(self.content_row_map[cell, slot]).to_bytes(8, "little", signed=True))
                h.update(int(self.content_row_versions[cell, slot]).to_bytes(8, "little"))
        return h.hexdigest()

    def require_content_alias_row_v1(self, cell: int, slot: int) -> int:
        if (
            type(cell) is not int
            or type(slot) is not int
            or not 0 <= cell < self.physical_cell_count
            or not 0 <= slot < SLOT_CAPACITY_V1
        ):
            raise HAPBContractError("HAPB content alias lookup coordinates are invalid")
        content_row = int(self.content_row_map[cell, slot].item())
        if content_row < 0:
            raise HAPBContractError(
                "HAPB content alias lookup encountered an unbound logical slot"
            )
        return content_row

    def bind_content_alias_v1(
        self,
        *,
        cell: int,
        parent_slot: int,
        child_slot: int,
    ) -> None:
        """Publish one exact parent-content alias after host commit."""

        parent_content = self.require_content_alias_row_v1(cell, parent_slot)
        parent_bytes = self.canonical_host_bank[cell, parent_slot]
        child_bytes = self.canonical_host_bank[cell, child_slot]
        if not torch.equal(parent_bytes, child_bytes):
            raise HAPBContractError(
                "HAPB alias publication requires exact committed parent bytes"
            )
        self.slot_generations[cell, child_slot] += 1
        self.content_row_map[cell, child_slot] = parent_content
        self.content_row_versions[cell, child_slot] = (
            self.content_row_versions[cell, parent_slot]
        )

    def validate_content_alias_bindings_v1(self) -> None:
        """Fail closed if a currently bound row no longer equals its pool row."""

        flat_pool = self.hot_prefix_cache
        if flat_pool is None:
            raise HAPBContractError("HAPB content alias pool is unavailable")
        for cell in range(self.physical_cell_count):
            for slot in range(SLOT_CAPACITY_V1):
                content_row = int(self.content_row_map[cell, slot].item())
                if content_row < 0:
                    continue
                content_cell = content_row // CONTENT_POOL_ROWS_PER_CELL_V1
                content_slot = content_row % CONTENT_POOL_ROWS_PER_CELL_V1
                if content_cell != cell:
                    raise HAPBContractError("HAPB content alias binding crosses a cell")
                if not torch.equal(
                    self.canonical_host_bank[cell, slot],
                    self.canonical_host_bank[cell, content_slot],
                ):
                    raise HAPBContractError(
                        "HAPB content alias binding diverged from canonical bytes"
                    )

    def _iter_refilled_overflow_tiles_v1(
        self, global_rows: torch.Tensor
    ) -> Iterator[tuple[int, int]]:
        """Yield bounded arena windows after synchronous canonical refills.

        The yielded slices always alias the one retained CUDA tile.  In
        particular, this method must not clone a tile: a clone would silently
        create a second policy arena and would make the H1 one-buffer decision
        untrue.  Consumers must finish each tile before requesting the next.
        """

        if (not isinstance(global_rows, torch.Tensor) or global_rows.device.type != "cpu"
                or global_rows.dtype != torch.int64 or global_rows.ndim != 1):
            raise HAPBContractError("HAPB row map must be a CPU int64 vector")
        values = [int(value) for value in global_rows.tolist()]
        if len(set(values)) != len(values):
            raise HAPBContractError("HAPB row map must be stable and unique")
        limit = self.physical_cell_count * SLOT_CAPACITY_V1
        if any(value < 0 or value >= limit or value % SLOT_CAPACITY_V1 < HOT_PREFIX_ROWS_V1 for value in values):
            raise HAPBContractError("HAPB overflow map contains a non-overflow or invalid row")
        flat = self.canonical_host_bank.reshape(-1, POLICY_WIDTH_V1)
        staging = self.pinned_staging_buffers[0]
        for start in range(0, len(values), self.tile_rows):
            indices = global_rows[start : start + self.tile_rows]
            count = int(indices.numel())
            torch.index_select(flat, 0, indices, out=staging[:count])
            self.cuda_tile_arena[:count].copy_(staging[:count], non_blocking=True)
            torch.cuda.synchronize(self.cuda_tile_arena.device)
            self.record_policy_h2d_rows_v1(count)
            yield start, count

    def compact_logits_v1(
        self,
        compact_policy_export: object,
        global_rows: torch.Tensor,
        observations: torch.Tensor,
        drives: torch.Tensor,
        attack_bias: torch.Tensor,
        action_temperature: torch.Tensor,
        drive_logit_gain: torch.Tensor,
        *,
        hidden_size: int,
        action_count: int = 7,
        w_hunger_gather: float,
        w_pain_rest: float,
        w_fatigue_rest: float,
        w_pain_attack: float,
        w_fear_attack: float,
    ) -> tuple[torch.Tensor, ...]:
        """Run the separate native compact kernel over stable overflow tiles.

        ``global_rows`` remains host-visible until the bounded refill iterator
        has completed its bounded D2H/gather/H2D transaction.  The CUDA map is
        constructed only after the stable CPU map is validated; no resident
        full-bank pointer is accepted by this ABI.
        """

        if not callable(compact_policy_export):
            raise HAPBContractError("HAPB compact policy export is unavailable")
        if observations.device != self.cuda_tile_arena.device or observations.dtype != torch.float32:
            raise HAPBContractError("HAPB observations must be CUDA float32")
        if observations.ndim != 2 or observations.shape[0] != global_rows.numel():
            raise HAPBContractError("HAPB observations do not match the stable row map")
        rows = [int(value) for value in global_rows.tolist()]
        if len(set(rows)) != len(rows):
            raise HAPBContractError("HAPB compact map must be stable and unique")
        outputs: list[torch.Tensor] = []
        for offset, count in self._iter_refilled_overflow_tiles_v1(global_rows):
            tile_rows = global_rows[offset : offset + count].to(
                self.cuda_tile_arena.device, non_blocking=False
            )
            output = compact_policy_export(
                self.cuda_tile_arena[:count],
                observations[offset : offset + count],
                tile_rows,
                drives,
                attack_bias,
                action_temperature,
                drive_logit_gain,
                hidden_size,
                action_count,
                w_hunger_gather,
                w_pain_rest,
                w_fatigue_rest,
                w_pain_attack,
                w_fear_attack,
            )
            # One arena is reusable only after the compact kernel no longer
            # reads it.  The logits are a distinct output authority, not an
            # extra policy-bank tile.
            torch.cuda.synchronize(self.cuda_tile_arena.device)
            outputs.append(output)
        return tuple(outputs)

    @property
    def full_cuda_policy_bank_retained(self) -> bool:
        """The contract makes the forbidden full CUDA policy authority unrepresentable."""

        return False


@dataclass(frozen=True, slots=True)
class HapbPolicyContextV1:
    """One open HAPB exact-policy transaction at the prepare/resume seam.

    This is deliberately an execution context rather than checkpoint state.
    It keeps the only host policy authority and the selected P256/tile cache
    alive while native prepare, compact refill, serial settlement, and resume
    run.  The stable overflow map and pre-predator Jungle cursor snapshot are
    bounded transaction inputs; neither can leak into a generation checkpoint.
    """

    owner: HostAuthoritativePolicyBankV1
    stable_overflow_rows: torch.Tensor
    stable_overflow_active_rows: torch.Tensor
    jungle_pre_predator_policy_pcg64_state: torch.Tensor
    jungle_pre_predator_policy_pcg64_buffer: torch.Tensor

    @classmethod
    def create(
        cls,
        *,
        owner: HostAuthoritativePolicyBankV1,
        stable_overflow_rows: torch.Tensor,
        stable_overflow_active_rows: torch.Tensor,
        jungle_pre_predator_policy_pcg64_state: torch.Tensor,
        jungle_pre_predator_policy_pcg64_buffer: torch.Tensor,
    ) -> "HapbPolicyContextV1":
        if not isinstance(owner, HostAuthoritativePolicyBankV1):
            raise HAPBContractError("HAPB policy context requires its sole owner")
        # The native context receives raw tensors, so validity has to be
        # established before those tensors cross the Python boundary.  This
        # prevents an all-zero post-restore allocation from being mistaken for
        # a hot-prefix policy cache.
        owner.require_hot_prefix_cache_v1()
        rows = stable_overflow_rows
        if (
            not isinstance(rows, torch.Tensor)
            or rows.device.type != "cpu"
            or rows.dtype != torch.int64
            or rows.ndim != 1
        ):
            raise HAPBContractError("HAPB policy context map must be CPU int64")
        values = [int(value) for value in rows.tolist()]
        if len(values) != len(set(values)):
            raise HAPBContractError("HAPB policy context map must be stable and unique")
        if any(
            value < 0
            or value >= owner.physical_cell_count * SLOT_CAPACITY_V1
            or value % SLOT_CAPACITY_V1 < HOT_PREFIX_ROWS_V1
            for value in values
        ):
            raise HAPBContractError("HAPB policy context map contains invalid overflow rows")
        active_rows = stable_overflow_active_rows
        if (
            not isinstance(active_rows, torch.Tensor)
            or active_rows.device.type != "cpu"
            or active_rows.dtype != torch.int64
            or active_rows.ndim != 1
            or active_rows.numel() != rows.numel()
        ):
            raise HAPBContractError("HAPB policy context active-row map is malformed")
        active_values = [int(value) for value in active_rows.tolist()]
        if len(active_values) != len(set(active_values)) or any(
            value < 0 or value >= owner.physical_cell_count * SLOT_CAPACITY_V1
            for value in active_values
        ):
            raise HAPBContractError("HAPB policy context active-row map is invalid")
        if any(
            value // SLOT_CAPACITY_V1 != active // SLOT_CAPACITY_V1
            for value, active in zip(values, active_values, strict=True)
        ):
            raise HAPBContractError("HAPB policy map crosses a resident cell boundary")
        state = jungle_pre_predator_policy_pcg64_state
        buffer = jungle_pre_predator_policy_pcg64_buffer
        device = owner.cuda_tile_arena.device
        if (
            not isinstance(state, torch.Tensor)
            or state.device != device
            or state.dtype not in {torch.int64, torch.uint64}
            or tuple(state.shape) != (owner.physical_cell_count, 4)
            or not state.is_contiguous()
        ):
            raise HAPBContractError("HAPB Jungle policy cursor state must be CUDA [B,4]")
        if (
            not isinstance(buffer, torch.Tensor)
            or buffer.device != device
            or buffer.dtype not in {torch.int64, torch.uint64}
            or tuple(buffer.shape) != (owner.physical_cell_count, 2)
            or not buffer.is_contiguous()
        ):
            raise HAPBContractError("HAPB Jungle policy cursor buffer must be CUDA [B,2]")
        return cls(owner, rows, active_rows, state, buffer)

    @property
    def full_cuda_policy_bank_retained(self) -> bool:
        return self.owner.full_cuda_policy_bank_retained


def hapb_frag_a_birth_counters_snapshot_v1() -> dict[str, int]:
    return dict(_HAPB_FRAG_A_BIRTH_COUNTERS_V1)


def _hapb_frag_a_use_batched_birth_v1() -> bool:
    if os.environ.get(HAPB_FRAG_A_PER_CELL_BIRTH_ENV_V1) == "1":
        return False
    if os.environ.get(HAPB_FRAG_A_BATCHED_BIRTH_ENV_V1) == "1":
        return True
    return HAPB_FRAG_A_BATCHED_BIRTH_ENABLED_V1


def _reset_hapb_frag_a_birth_counters_v1(
    *,
    active_cells: int,
    birth_rows: int,
) -> None:
    _HAPB_FRAG_A_BIRTH_COUNTERS_V1["launches"] = 0
    _HAPB_FRAG_A_BIRTH_COUNTERS_V1["device_syncs"] = 0
    _HAPB_FRAG_A_BIRTH_COUNTERS_V1["active_cells"] = active_cells
    _HAPB_FRAG_A_BIRTH_COUNTERS_V1["birth_rows"] = birth_rows
    _HAPB_FRAG_A_BIRTH_COUNTERS_V1["chunks"] = 0


def _apply_ordinary_birth_mutates_per_cell_v1(
    owner: HostAuthoritativePolicyBankV1,
    *,
    cuda_mutate_export: object,
    plans: list[tuple[int, int, int, torch.Tensor, torch.Tensor]],
    policy_birth_delta_tape: torch.Tensor,
    tape_width: int,
    work_staging: torch.Tensor,
    committed: torch.Tensor,
) -> None:
    for cell, count, cursor, parents, children in plans:
        torch.index_select(
            owner.canonical_host_bank[cell], 0, parents,
            out=work_staging[:count],
        )
        parent_work = owner.cuda_tile_arena[:count]
        delta_work = owner.cuda_tile_arena[count : 2 * count]
        child_work = owner.cuda_tile_arena[2 * count : 3 * count]
        parent_work.copy_(work_staging[:count], non_blocking=True)
        torch.cuda.synchronize(owner.cuda_tile_arena.device)
        delta = policy_birth_delta_tape[cell, cursor : cursor + count]
        if tape_width == 1:
            work_staging[:count].copy_(delta.expand(-1, POLICY_WIDTH_V1))
        else:
            work_staging[:count].copy_(delta)
        delta_work.copy_(work_staging[:count], non_blocking=True)
        torch.cuda.synchronize(owner.cuda_tile_arena.device)
        cuda_mutate_export(parent_work, delta_work, child_work)
        torch.cuda.synchronize(owner.cuda_tile_arena.device)
        work_staging[:count].copy_(child_work, non_blocking=True)
        torch.cuda.synchronize(owner.cuda_tile_arena.device)
        owner.canonical_host_bank[cell].index_copy_(0, children, work_staging[:count])
        committed[cell, :count] = children
        _HAPB_FRAG_A_BIRTH_COUNTERS_V1["launches"] += 1
        _HAPB_FRAG_A_BIRTH_COUNTERS_V1["device_syncs"] += 4
        _HAPB_FRAG_A_BIRTH_COUNTERS_V1["chunks"] += 1


def _apply_ordinary_birth_mutates_batched_v1(
    owner: HostAuthoritativePolicyBankV1,
    *,
    cuda_mutate_export: object,
    plans: list[tuple[int, int, int, torch.Tensor, torch.Tensor]],
    policy_birth_delta_tape: torch.Tensor,
    tape_width: int,
    committed: torch.Tensor,
) -> None:
    total = sum(count for _cell, count, _cursor, _parents, _children in plans)
    if total <= 0:
        return
    parents_cpu = torch.empty((total, POLICY_WIDTH_V1), dtype=torch.float32)
    deltas_cpu = torch.empty((total, POLICY_WIDTH_V1), dtype=torch.float32)
    children_cpu = torch.empty((total, POLICY_WIDTH_V1), dtype=torch.float32)
    offset = 0
    for cell, count, cursor, parents, _children in plans:
        torch.index_select(
            owner.canonical_host_bank[cell],
            0,
            parents,
            out=parents_cpu[offset : offset + count],
        )
        delta = policy_birth_delta_tape[cell, cursor : cursor + count]
        if tape_width == 1:
            deltas_cpu[offset : offset + count].copy_(
                delta.expand(-1, POLICY_WIDTH_V1)
            )
        else:
            deltas_cpu[offset : offset + count].copy_(delta)
        offset += count
    chunk_limit = max(1, owner.tile_rows // 3)
    for start in range(0, total, chunk_limit):
        count = min(chunk_limit, total - start)
        parent_work = owner.cuda_tile_arena[:count]
        delta_work = owner.cuda_tile_arena[count : 2 * count]
        child_work = owner.cuda_tile_arena[2 * count : 3 * count]
        parent_work.copy_(parents_cpu[start : start + count], non_blocking=True)
        delta_work.copy_(deltas_cpu[start : start + count], non_blocking=True)
        torch.cuda.synchronize(owner.cuda_tile_arena.device)
        cuda_mutate_export(parent_work, delta_work, child_work)
        torch.cuda.synchronize(owner.cuda_tile_arena.device)
        children_cpu[start : start + count].copy_(child_work, non_blocking=True)
        torch.cuda.synchronize(owner.cuda_tile_arena.device)
        _HAPB_FRAG_A_BIRTH_COUNTERS_V1["launches"] += 1
        _HAPB_FRAG_A_BIRTH_COUNTERS_V1["device_syncs"] += 3
        _HAPB_FRAG_A_BIRTH_COUNTERS_V1["chunks"] += 1
    offset = 0
    for cell, count, _cursor, _parents, children in plans:
        owner.canonical_host_bank[cell].index_copy_(
            0, children, children_cpu[offset : offset + count]
        )
        committed[cell, :count] = children
        offset += count


def commit_hapb_ordinary_fixed_mlp_birth_rows_v1(
    owner: HostAuthoritativePolicyBankV1,
    *,
    cuda_mutate_export: object,
    parent_slots: torch.Tensor,
    child_slots: torch.Tensor,
    birth_counts: torch.Tensor,
    policy_birth_delta_tape: torch.Tensor,
    policy_birth_mutation_mask_tape: torch.Tensor,
    policy_birth_semantic_draws_tape: torch.Tensor,
    policy_birth_tape_cursor: torch.Tensor,
) -> torch.Tensor:
    """Commit exact ordinary fixed-MLP child rows to host authority first.

    This is the HAPB birth-plan transaction, deliberately separate from the
    resident identity commit.  All plans/tapes are bounded host transfers at
    the existing plan boundary.  The one pinned/CUDA tile is the only
    parent/delta/child workbench.  CUDA performs exact float32 addition;
    child bytes return to pageable authority before this function returns
    eligibility slots.
    """

    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise HAPBContractError("HAPB birth transaction requires its sole owner")
    if not callable(cuda_mutate_export):
        raise HAPBContractError("HAPB birth transaction requires its CUDA mutation export")
    cells = owner.physical_cell_count
    expected_plan = (cells, parent_slots.shape[1]) if (
        isinstance(parent_slots, torch.Tensor) and parent_slots.ndim == 2
    ) else None
    if (
        expected_plan is None
        or not isinstance(child_slots, torch.Tensor)
        or parent_slots.device.type != "cpu"
        or child_slots.device.type != "cpu"
        or parent_slots.dtype != torch.int64
        or child_slots.dtype != torch.int64
        or not parent_slots.is_contiguous()
        or not child_slots.is_contiguous()
        or tuple(child_slots.shape) != expected_plan
        or expected_plan[0] != cells
        or expected_plan[1] * 3 > owner.tile_rows
    ):
        raise HAPBContractError("HAPB birth plans must be bounded CPU int64 B rows")
    if (
        not isinstance(birth_counts, torch.Tensor)
        or birth_counts.device.type != "cpu"
        or birth_counts.dtype != torch.int64
        or not birth_counts.is_contiguous()
        or tuple(birth_counts.shape) != (cells,)
    ):
        raise HAPBContractError("HAPB birth counts must be CPU int64 [B]")
    tape_capacity = (
        int(policy_birth_delta_tape.shape[1])
        if isinstance(policy_birth_delta_tape, torch.Tensor)
        and policy_birth_delta_tape.ndim == 3
        else -1
    )
    tape_width = (
        int(policy_birth_delta_tape.shape[2])
        if tape_capacity >= 0
        else -1
    )
    if (
        not isinstance(policy_birth_delta_tape, torch.Tensor)
        or policy_birth_delta_tape.device.type != "cpu"
        or policy_birth_delta_tape.dtype != torch.float32
        or not policy_birth_delta_tape.is_contiguous()
        or tuple(policy_birth_delta_tape.shape[:2]) != (cells, tape_capacity)
        or tape_width not in {1, POLICY_WIDTH_V1}
        or not isinstance(policy_birth_mutation_mask_tape, torch.Tensor)
        or policy_birth_mutation_mask_tape.device.type != "cpu"
        or policy_birth_mutation_mask_tape.dtype != torch.bool
        or not policy_birth_mutation_mask_tape.is_contiguous()
        or tuple(policy_birth_mutation_mask_tape.shape)
            != tuple(policy_birth_delta_tape.shape)
        or not isinstance(policy_birth_semantic_draws_tape, torch.Tensor)
        or policy_birth_semantic_draws_tape.device.type != "cpu"
        or policy_birth_semantic_draws_tape.dtype != torch.int64
        or not policy_birth_semantic_draws_tape.is_contiguous()
        or tuple(policy_birth_semantic_draws_tape.shape) != (cells, tape_capacity)
        or not isinstance(policy_birth_tape_cursor, torch.Tensor)
        or policy_birth_tape_cursor.device.type != "cpu"
        or policy_birth_tape_cursor.dtype != torch.int64
        or not policy_birth_tape_cursor.is_contiguous()
        or tuple(policy_birth_tape_cursor.shape) != (cells,)
    ):
        raise HAPBContractError("HAPB birth policy tapes must be exact bounded CPU authorities")

    # Validate the whole B48 plan before mutating any canonical row.  Keep the
    # old child bytes in the same frozen staging tile so a CUDA or host error
    # can roll back every already-committed cell without a second policy bank.
    plans: list[tuple[int, int, int, torch.Tensor, torch.Tensor]] = []
    saved_rows = 0
    maximum_cell_births = 0
    for cell in range(cells):
        count = int(birth_counts[cell].item())
        cursor = int(policy_birth_tape_cursor[cell].item())
        if count < 0 or count > expected_plan[1] or cursor < 0 or cursor + count > tape_capacity:
            raise HAPBContractError("HAPB birth plan is invalid or exhausts its policy tape")
        parents = [int(value) for value in parent_slots[cell, :count].tolist()]
        children = [int(value) for value in child_slots[cell, :count].tolist()]
        if (
            any(slot < 0 or slot >= SLOT_CAPACITY_V1 for slot in parents + children)
            or len(set(children)) != len(children)
            or any(parent == child for parent, child in zip(parents, children, strict=True))
            or any(child in parents for child in children)
        ):
            raise HAPBContractError("HAPB birth plan aliases or addresses an invalid policy row")
        if count == 0:
            continue
        parent_index = torch.tensor(parents, dtype=torch.int64, device="cpu")
        child_index = torch.tensor(children, dtype=torch.int64, device="cpu")
        plans.append((cell, count, cursor, parent_index, child_index))
        saved_rows += count
        maximum_cell_births = max(maximum_cell_births, count)
    if saved_rows + 3 * maximum_cell_births > owner.tile_rows:
        raise HAPBContractError(
            "HAPB birth rollback cannot fit in the one bounded staging arena"
        )

    committed = torch.full(expected_plan, -1, dtype=torch.int64, device="cpu")
    staging = owner.pinned_staging_buffers[0]
    backups: list[tuple[int, torch.Tensor, int, int]] = []
    backup_offset = 0
    old_content_row_map = owner.content_row_map.clone()
    old_content_row_versions = owner.content_row_versions.clone()
    old_slot_generations = owner.slot_generations.clone()
    for cell, count, _cursor, _parents, children in plans:
        torch.index_select(
            owner.canonical_host_bank[cell], 0, children,
            out=staging[backup_offset : backup_offset + count],
        )
        backups.append((cell, children, backup_offset, count))
        backup_offset += count

    work_staging = staging[saved_rows:]
    _reset_hapb_frag_a_birth_counters_v1(
        active_cells=len(plans),
        birth_rows=saved_rows,
    )
    try:
        if _hapb_frag_a_use_batched_birth_v1():
            _apply_ordinary_birth_mutates_batched_v1(
                owner,
                cuda_mutate_export=cuda_mutate_export,
                plans=plans,
                policy_birth_delta_tape=policy_birth_delta_tape,
                tape_width=tape_width,
                committed=committed,
            )
        else:
            _apply_ordinary_birth_mutates_per_cell_v1(
                owner,
                cuda_mutate_export=cuda_mutate_export,
                plans=plans,
                policy_birth_delta_tape=policy_birth_delta_tape,
                tape_width=tape_width,
                work_staging=work_staging,
                committed=committed,
            )
        if owner.content_alias_enabled:
            # The rescue route admits only exact parent clones.  Validate every
            # child while the transaction is still inside the rollback fence;
            # no mapping is published until the whole B48 plan is proven.
            for cell, count, _cursor, parents, children in plans:
                for parent, child in zip(
                    parents.tolist(), children.tolist(), strict=True
                ):
                    if not torch.equal(
                        owner.canonical_host_bank[cell, int(parent)],
                        owner.canonical_host_bank[cell, int(child)],
                    ):
                        raise HAPBContractError(
                            "HAPB content alias rescue refuses new child policy bytes"
                        )
            # Update only the touched children. A full B48/C512 digest rebuild
            # here would rescan the entire 10k-wide policy bank for every
            # ordinary-birth callback and make the exact fallback unusably
            # expensive. A replaced hot-prefix pool row invalidates only the
            # overflow aliases that pointed at that physical row; an overflow
            # child can reuse its exact parent's existing pool row or remain
            # explicitly unbound on the tiled fallback.
            for cell, _count, _cursor, parents, children in plans:
                for parent, child in zip(
                    parents.tolist(), children.tolist(), strict=True
                ):
                    parent_slot = int(parent)
                    child_slot = int(child)
                    if child_slot < HOT_PREFIX_ROWS_V1:
                        content_row = (
                            cell * CONTENT_POOL_ROWS_PER_CELL_V1 + child_slot
                        )
                        for overflow_slot in range(
                            HOT_PREFIX_ROWS_V1, SLOT_CAPACITY_V1
                        ):
                            if (
                                int(
                                    owner.content_row_map[
                                        cell, overflow_slot
                                    ].item()
                                )
                                == content_row
                            ):
                                owner.content_row_map[cell, overflow_slot] = -1
                        owner.content_row_map[cell, child_slot] = content_row
                    else:
                        parent_content = int(
                            owner.content_row_map[cell, parent_slot].item()
                        )
                        owner.content_row_map[cell, child_slot] = (
                            parent_content if parent_content >= 0 else -1
                        )
            map_changed = owner.content_row_map.ne(old_content_row_map)
            owner.content_row_versions[map_changed] += 1
        for cell, count, _cursor, _parents, children in plans:
            for child in children.tolist():
                child_slot = int(child)
                owner.slot_generations[cell, child_slot] += 1
                owner.content_row_versions[cell, child_slot] += 1
    except Exception:
        for cell, children, offset, count in backups:
            owner.canonical_host_bank[cell].index_copy_(
                0, children, staging[offset : offset + count]
            )
        owner.content_row_map.copy_(old_content_row_map)
        owner.content_row_versions.copy_(old_content_row_versions)
        owner.slot_generations.copy_(old_slot_generations)
        raise

    # The policy bytes are authoritative before these side effects become
    # visible.  A restored cache intentionally remains invalid and is rebuilt
    # as one complete cache rather than receiving partial child updates.
    if owner.hot_prefix_valid:
        hot_prefix = owner.require_hot_prefix_cache_v1()
        hot_rows_refreshed = 0
        for cell, count, _cursor, _parents, children in plans:
            for child in children.tolist():
                child_slot = int(child)
                if child_slot >= HOT_PREFIX_ROWS_V1:
                    continue
                work_staging[0].copy_(
                    owner.canonical_host_bank[cell, child_slot]
                )
                hot_prefix[cell, child_slot].copy_(
                    work_staging[0], non_blocking=True
                )
                hot_rows_refreshed += 1
        if hot_rows_refreshed:
            torch.cuda.synchronize(hot_prefix.device)
            owner.record_policy_h2d_rows_v1(hot_rows_refreshed)
    for cell, count, cursor, _parents, _children in plans:
        policy_birth_tape_cursor[cell] = cursor + count
    return committed


def commit_hapb_turnover_fixed_mlp_birth_rows_v1(
    owner: HostAuthoritativePolicyBankV1,
    *,
    cuda_mutate_export: object,
    parent_slots: torch.Tensor,
    mate_slots: torch.Tensor,
    child_slots: torch.Tensor,
    committed_counts: torch.Tensor,
    policy_delta_tape: torch.Tensor | tuple[torch.Tensor, ...],
    policy_mutation_mask_tape: torch.Tensor | tuple[torch.Tensor, ...],
    policy_crossover_mask_packed_tape: torch.Tensor | tuple[torch.Tensor, ...],
    policy_absolute_cursor: torch.Tensor,
    policy_window_base: torch.Tensor,
) -> torch.Tensor:
    """Commit exact fixed-MLP turnover rows after native identity commit.

    The native turnover transaction owns cull/refill planning, identity
    assignment, lifecycle state, and RNG advancement.  This bounded callback
    receives the committed child slots and the post-turnover policy cursor,
    reconstructs each crossover/mutation row from the canonical host bank, and
    uses the existing three-row CUDA mutation workbench for the exact float32
    addition.  The host bank is updated before the next native publication and
    remains the only policy authority.
    """

    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise HAPBContractError("HAPB turnover transaction requires its sole owner")
    if not callable(cuda_mutate_export):
        raise HAPBContractError(
            "HAPB turnover transaction requires its CUDA mutation export"
        )

    tensors = (parent_slots, mate_slots, child_slots)
    if any(
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.int64
        or not value.is_contiguous()
        or value.ndim != 2
        for value in tensors
    ):
        raise HAPBContractError(
            "HAPB turnover plans must be contiguous CPU int64 matrices"
        )
    cells = owner.physical_cell_count
    expected_plan_shape = (cells, int(parent_slots.shape[1]))
    if (
        tuple(parent_slots.shape) != expected_plan_shape
        or tuple(mate_slots.shape) != expected_plan_shape
        or tuple(child_slots.shape) != expected_plan_shape
        or expected_plan_shape[1] < 1
        or expected_plan_shape[1] > SLOT_CAPACITY_V1
    ):
        raise HAPBContractError(
            "HAPB turnover plans must cover the exact B/C512 policy capacity"
        )
    if (
        not isinstance(committed_counts, torch.Tensor)
        or committed_counts.device.type != "cpu"
        or committed_counts.dtype != torch.int64
        or not committed_counts.is_contiguous()
        or tuple(committed_counts.shape) != (cells,)
    ):
        raise HAPBContractError(
            "HAPB turnover committed counts must be contiguous CPU int64 [B]"
        )

    def normalize_tape_rows(
        value: torch.Tensor | tuple[torch.Tensor, ...],
        *,
        dtype: torch.dtype,
        width: int | None,
        name: str,
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(value, torch.Tensor):
            if (
                value.device.type != "cpu"
                or value.dtype != dtype
                or not value.is_contiguous()
                or value.ndim != 3
                or value.shape[0] != cells
            ):
                raise HAPBContractError(
                    f"HAPB {name} must be contiguous CPU B tape storage"
                )
            rows = tuple(value[cell] for cell in range(cells))
        elif isinstance(value, tuple) and len(value) == cells:
            rows = value
        else:
            raise HAPBContractError(
                f"HAPB {name} must be one B tensor or one row tuple per cell"
            )
        for row in rows:
            if (
                not isinstance(row, torch.Tensor)
                or row.device.type != "cpu"
                or row.dtype != dtype
                or not row.is_contiguous()
                or row.ndim != 2
                or (width is not None and row.shape[1] != width)
            ):
                raise HAPBContractError(f"HAPB {name} row has an invalid shape or dtype")
        return rows

    delta_rows_by_cell = normalize_tape_rows(
        policy_delta_tape,
        dtype=torch.float32,
        width=None,
        name="turnover policy delta tape",
    )
    tape_width = int(delta_rows_by_cell[0].shape[1])
    if tape_width not in {1, POLICY_WIDTH_V1}:
        raise HAPBContractError("HAPB turnover policy delta tape width is invalid")
    mutation_rows_by_cell = normalize_tape_rows(
        policy_mutation_mask_tape,
        dtype=torch.bool,
        width=tape_width,
        name="turnover policy mutation mask tape",
    )
    crossover_rows_by_cell = normalize_tape_rows(
        policy_crossover_mask_packed_tape,
        dtype=torch.uint8,
        width=POLICY_CROSSOVER_PACKED_WIDTH_V1,
        name="turnover policy crossover tape",
    )
    for cell in range(cells):
        if (
            delta_rows_by_cell[cell].shape[0] != mutation_rows_by_cell[cell].shape[0]
            or delta_rows_by_cell[cell].shape[0]
            != crossover_rows_by_cell[cell].shape[0]
        ):
            raise HAPBContractError("HAPB turnover policy tape row capacities diverged")
    cursor_tensors = (policy_absolute_cursor, policy_window_base)
    if any(
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.int64
        or not value.is_contiguous()
        or tuple(value.shape) != (cells,)
        for value in cursor_tensors
    ):
        raise HAPBContractError(
            "HAPB turnover policy cursors must be contiguous CPU int64 [B]"
        )

    # Validate the complete native plan before changing any canonical row.
    # Native identity commit has already proved liveness and parentage; these
    # checks keep a malformed callback payload from partially mutating HAPB.
    plans: list[tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for cell in range(cells):
        count = int(committed_counts[cell].item())
        absolute_after = int(policy_absolute_cursor[cell].item())
        window_base = int(policy_window_base[cell].item())
        if count < 0 or count > expected_plan_shape[1]:
            raise HAPBContractError("HAPB turnover committed count is invalid")
        absolute_start = absolute_after - count
        physical_start = absolute_start - window_base
        if (
            absolute_after < 0
            or absolute_start < window_base
            or physical_start < 0
            or physical_start + count > delta_rows_by_cell[cell].shape[0]
        ):
            raise HAPBContractError(
                "HAPB turnover policy cursor is outside its resident tape window"
            )
        if count == 0:
            continue
        parents = parent_slots[cell, :count].clone()
        mates = mate_slots[cell, :count].clone()
        children = child_slots[cell, :count].clone()
        parent_values = [int(value) for value in parents.tolist()]
        mate_values = [int(value) for value in mates.tolist()]
        child_values = [int(value) for value in children.tolist()]
        if (
            any(
                slot < 0 or slot >= SLOT_CAPACITY_V1
                for slot in parent_values + mate_values + child_values
            )
            or any(
                parent == mate or parent == child or mate == child
                for parent, mate, child in zip(
                    parent_values, mate_values, child_values, strict=True
                )
            )
            or len(set(child_values)) != len(child_values)
            or any(child in parent_values or child in mate_values for child in child_values)
        ):
            raise HAPBContractError("HAPB turnover plan aliases an invalid policy row")
        plans.append((cell, count, physical_start, parents, mates, children))

    committed = torch.full(
        expected_plan_shape, -1, dtype=torch.int64, device="cpu"
    )
    if not plans:
        return committed

    # The native HAPB tile is three-row workbench slices.  Process a larger
    # turnover plan in bounded chunks rather than allocating a second policy
    # arena or assuming that every turnover fits in one 512-row tile.
    chunk_limit = owner.tile_rows // 3
    if chunk_limit < 1:
        raise HAPBContractError("HAPB turnover tile cannot hold one three-row workbench")
    staging = owner.pinned_staging_buffers[0]
    backups = [
        (
            cell,
            children,
            owner.canonical_host_bank[cell].index_select(0, children),
        )
        for cell, _count, _physical_start, _parents, _mates, children in plans
    ]
    old_content_row_map = owner.content_row_map.clone()
    old_content_row_versions = owner.content_row_versions.clone()
    old_slot_generations = owner.slot_generations.clone()
    try:
        column_indices = torch.arange(POLICY_WIDTH_V1, dtype=torch.int64)
        byte_indices = column_indices // 8
        bit_masks = torch.bitwise_left_shift(
            torch.ones_like(column_indices), column_indices % 8
        )
        for cell, count, physical_start, parents, mates, children in plans:
            for offset in range(0, count, chunk_limit):
                chunk = min(chunk_limit, count - offset)
                parent_index = parents[offset : offset + chunk]
                mate_index = mates[offset : offset + chunk]
                child_index = children[offset : offset + chunk]
                parent_rows = owner.canonical_host_bank[cell].index_select(
                    0, parent_index
                )
                mate_rows = owner.canonical_host_bank[cell].index_select(
                    0, mate_index
                )
                packed_mask = crossover_rows_by_cell[cell][
                    physical_start + offset : physical_start + offset + chunk
                ].to(torch.int64)
                crossover_mask = (
                    packed_mask[:, byte_indices] & bit_masks
                ).ne(0)
                inherited_rows = torch.where(
                    crossover_mask, parent_rows, mate_rows
                )
                delta_rows = delta_rows_by_cell[cell][
                    physical_start + offset : physical_start + offset + chunk
                ]
                if tape_width == 1:
                    delta_rows = delta_rows.expand(-1, POLICY_WIDTH_V1)
                parent_work = owner.cuda_tile_arena[:chunk]
                delta_work = owner.cuda_tile_arena[chunk : 2 * chunk]
                child_work = owner.cuda_tile_arena[2 * chunk : 3 * chunk]
                parent_work.copy_(inherited_rows, non_blocking=False)
                delta_work.copy_(delta_rows, non_blocking=False)
                owner.record_policy_h2d_rows_v1(2 * chunk)
                torch.cuda.synchronize(owner.cuda_tile_arena.device)
                cuda_mutate_export(parent_work, delta_work, child_work)
                torch.cuda.synchronize(owner.cuda_tile_arena.device)
                staging[:chunk].copy_(child_work, non_blocking=True)
                torch.cuda.synchronize(owner.cuda_tile_arena.device)
                owner.record_policy_d2h_rows_v1(chunk)
                owner.canonical_host_bank[cell].index_copy_(
                    0, child_index, staging[:chunk]
                )
            committed[cell, :count] = children

        if owner.content_alias_enabled:
            rebuilt_map = _content_row_map_from_canonical_v1(
                owner.canonical_host_bank
            )
            map_changed = rebuilt_map.ne(owner.content_row_map)
            owner.content_row_map.copy_(rebuilt_map)
            owner.content_row_versions[map_changed] += 1
        for cell, _count, _physical_start, _parents, _mates, children in plans:
            for child in children.tolist():
                child_slot = int(child)
                owner.slot_generations[cell, child_slot] += 1
                owner.content_row_versions[cell, child_slot] += 1
                if owner.content_alias_enabled and child_slot < HOT_PREFIX_ROWS_V1:
                    content_row = cell * CONTENT_POOL_ROWS_PER_CELL_V1 + child_slot
                    aliases = owner.content_row_map[cell] == content_row
                    owner.content_row_versions[cell, aliases] += 1

        # Rebuild the noncanonical P256 cache from the now-authoritative host
        # bank before any next-generation prepare can observe it.  The method
        # returns a replacement only for an invalid restored owner; this owner
        # is already valid, so its shared tensors are updated in place.
        owner.rehydrate_hot_prefix_cache_v1()
        if owner.content_alias_enabled:
            owner.validate_content_alias_bindings_v1()
    except Exception:
        for cell, children, original_rows in backups:
            owner.canonical_host_bank[cell].index_copy_(0, children, original_rows)
        owner.content_row_map.copy_(old_content_row_map)
        owner.content_row_versions.copy_(old_content_row_versions)
        owner.slot_generations.copy_(old_slot_generations)
        owner.rehydrate_hot_prefix_cache_v1()
        raise
    return committed


def execute_hapb_prepared_policy_boundary_v1(
    owner: HostAuthoritativePolicyBankV1,
    *,
    refill_logits_export: object,
    settle_prepared_export: object,
    native_boundary_export: object | None = None,
    ordered_slots: torch.Tensor,
    observations: torch.Tensor,
    stable_miss_rows: torch.Tensor,
    active_row_counts: torch.Tensor,
    miss_row_counts: torch.Tensor,
    drives: torch.Tensor,
    attack_bias: torch.Tensor,
    action_temperature: torch.Tensor,
    drive_logit_gain: torch.Tensor,
    slot_tokens: torch.Tensor,
    action_cache_actions: torch.Tensor,
    action_cache_token: torch.Tensor,
    action_cache_cursor: torch.Tensor,
    action_cache_next: torch.Tensor,
    row_actions: torch.Tensor,
    policy_pcg64_state_words: torch.Tensor,
    policy_pcg64_buffer_words: torch.Tensor,
    w_hunger_gather: float,
    w_pain_rest: float,
    w_fatigue_rest: float,
    w_pain_attack: float,
    w_fear_attack: float,
) -> tuple[torch.Tensor, HapbPolicyContextV1]:
    """Execute one resident prepare/refill/resume policy boundary via HAPB.

    The caller supplies exactly the tensors emitted by the resident *prepare*
    phase.  This function makes the host-visible stable miss map explicit,
    computes only missed policy rows through P256/tiled HAPB, and calls the
    existing serial CUDA settlement before the resident tick resumes.  It does
    not accept an ``mlp_pack`` and therefore cannot silently use the resident
    full CUDA policy bank.
    """

    if not isinstance(owner, HostAuthoritativePolicyBankV1):
        raise HAPBContractError("HAPB prepared boundary requires its sole owner")
    if not callable(refill_logits_export) or not callable(settle_prepared_export):
        raise HAPBContractError("HAPB prepared boundary requires both native exports")
    if native_boundary_export is not None and not callable(native_boundary_export):
        raise HAPBContractError("HAPB prepared boundary export is unavailable")
    device = owner.cuda_tile_arena.device
    cells = owner.physical_cell_count
    expected_cuda: tuple[tuple[torch.Tensor, torch.dtype, tuple[int, ...]], ...] = (
        (ordered_slots, torch.int32, (cells, SLOT_CAPACITY_V1)),
        (observations, torch.float32, (cells, SLOT_CAPACITY_V1, 83)),
        (stable_miss_rows, torch.int64, (cells, SLOT_CAPACITY_V1)),
        (active_row_counts, torch.int64, (cells,)),
        (miss_row_counts, torch.int64, (cells,)),
        (drives, torch.float32, (cells, SLOT_CAPACITY_V1, 4)),
        (attack_bias, torch.float32, (cells, SLOT_CAPACITY_V1)),
        (action_temperature, torch.float32, (cells, SLOT_CAPACITY_V1)),
        (drive_logit_gain, torch.float32, (cells, SLOT_CAPACITY_V1)),
        (slot_tokens, torch.int64, (cells, SLOT_CAPACITY_V1)),
        (action_cache_actions, torch.int32, (cells, SLOT_CAPACITY_V1, 2)),
        (action_cache_token, torch.int64, (cells, SLOT_CAPACITY_V1)),
        (action_cache_cursor, torch.int32, (cells, SLOT_CAPACITY_V1)),
        (action_cache_next, torch.int32, (cells, SLOT_CAPACITY_V1)),
        (row_actions, torch.int32, (cells, SLOT_CAPACITY_V1)),
        (policy_pcg64_state_words, torch.uint64, (cells, 4)),
        (policy_pcg64_buffer_words, torch.int64, (cells, 2)),
    )
    for value, dtype, shape in expected_cuda:
        if (
            not isinstance(value, torch.Tensor)
            or value.device != device
            or value.dtype != dtype
            or not value.is_contiguous()
            or tuple(value.shape) != shape
        ):
            raise HAPBContractError("HAPB prepared boundary tensor ABI is malformed")

    # These are tiny plan coordinates, not policy data.  D2H occurs only at
    # the admitted prepare boundary and preserves resident miss order exactly.
    ordered_cpu = ordered_slots.cpu()
    miss_cpu = stable_miss_rows.cpu()
    active_cpu = active_row_counts.cpu()
    miss_count_cpu = miss_row_counts.cpu()
    hot_rows: list[int] = []
    hot_active_rows: list[int] = []
    overflow_rows: list[int] = []
    overflow_active_rows: list[int] = []
    for cell in range(cells):
        active_count = int(active_cpu[cell].item())
        miss_count = int(miss_count_cpu[cell].item())
        if active_count < 0 or active_count > SLOT_CAPACITY_V1 or not 0 <= miss_count <= active_count:
            raise HAPBContractError("HAPB prepared boundary counts are invalid")
        seen_miss_rows: set[int] = set()
        for ordinal in range(miss_count):
            active_row = int(miss_cpu[cell, ordinal].item())
            if active_row < 0 or active_row >= active_count or active_row in seen_miss_rows:
                raise HAPBContractError("HAPB prepared boundary miss ordering is invalid")
            seen_miss_rows.add(active_row)
            slot = int(ordered_cpu[cell, active_row].item())
            if slot < 0 or slot >= SLOT_CAPACITY_V1:
                raise HAPBContractError("HAPB prepared boundary ordered slot is invalid")
            global_slot = cell * SLOT_CAPACITY_V1 + slot
            global_active_row = cell * SLOT_CAPACITY_V1 + active_row
            if slot < HOT_PREFIX_ROWS_V1:
                hot_rows.append(global_slot)
                hot_active_rows.append(global_active_row)
            else:
                overflow_rows.append(global_slot)
                overflow_active_rows.append(global_active_row)
    context = HapbPolicyContextV1.create(
        owner=owner,
        stable_overflow_rows=torch.tensor(overflow_rows, dtype=torch.int64),
        stable_overflow_active_rows=torch.tensor(overflow_active_rows, dtype=torch.int64),
        jungle_pre_predator_policy_pcg64_state=policy_pcg64_state_words,
        jungle_pre_predator_policy_pcg64_buffer=policy_pcg64_buffer_words,
    )
    hot_rows_tensor = torch.tensor(hot_rows, dtype=torch.int64)
    hot_active_tensor = torch.tensor(hot_active_rows, dtype=torch.int64)
    with hapb_policy_context_guard_v1(context):
        context_arguments = (
            owner.canonical_host_bank, owner.pinned_staging_buffers[0],
            owner.require_hot_prefix_cache_v1(), owner.cuda_tile_arena,
            context.stable_overflow_rows, context.stable_overflow_active_rows,
            policy_pcg64_state_words, policy_pcg64_buffer_words,
        )
        policy_arguments = (
            observations.reshape(-1, 83), hot_rows_tensor, hot_active_tensor,
            drives.reshape(-1, 4), attack_bias.reshape(-1),
            action_temperature.reshape(-1), drive_logit_gain.reshape(-1),
        )
        settlement_arguments = (
            ordered_slots, slot_tokens, action_cache_actions, action_cache_token,
            action_cache_cursor, action_cache_next, row_actions, stable_miss_rows,
            active_row_counts, miss_row_counts,
        )
        weights = (
            w_hunger_gather, w_pain_rest, w_fatigue_rest, w_pain_attack,
            w_fear_attack,
        )
        if native_boundary_export is not None:
            counters = native_boundary_export(
                *context_arguments, *policy_arguments, *settlement_arguments, *weights
            )
        else:
            logits = refill_logits_export(
                *context_arguments, *policy_arguments, 64, 7, *weights
            )
            counters = settle_prepared_export(
                *settlement_arguments[:6], logits, *settlement_arguments[6:],
                policy_pcg64_state_words, policy_pcg64_buffer_words, *weights,
            )
    return counters, context


def active_hapb_policy_context_v1() -> HapbPolicyContextV1:
    """Return the open exact-policy context or fail before resident fallback."""

    context = _ACTIVE_POLICY_CONTEXT_V1.get()
    if context is None:
        raise HAPBContractError("HAPB policy context is not installed")
    return context


@contextmanager
def hapb_policy_context_guard_v1(
    context: HapbPolicyContextV1,
) -> Iterator[HapbPolicyContextV1]:
    """Install one non-nestable HAPB prepare/resume transaction context."""

    if not isinstance(context, HapbPolicyContextV1):
        raise HAPBContractError("HAPB policy context guard requires HAPB context")
    if _ACTIVE_POLICY_CONTEXT_V1.get() is not None:
        raise HAPBContractError("HAPB policy contexts must not nest")
    token: Token[HapbPolicyContextV1 | None] = _ACTIVE_POLICY_CONTEXT_V1.set(context)
    try:
        yield context
    finally:
        _ACTIVE_POLICY_CONTEXT_V1.reset(token)


def resolve_hapb_frag_rng_route_v1(
    *,
    hapb_rng_semantics_version: int = 1,
) -> dict[str, object]:
    """Select legacy HAPB FRAG (default) or explicit HAPB-FRAG-RNGV2."""

    from engine.hapb_rngv2.route import resolve_hapb_rng_route

    return resolve_hapb_rng_route(
        hapb_rng_semantics_version=hapb_rng_semantics_version
    )


def apply_hapb_frag_rng_route_v1(
    sim_cfg: object,
    *,
    hapb_rng_semantics_version: int = 1,
) -> dict[str, object]:
    """Explicit-only binder. The default HAPB start path never calls this."""

    from engine.hapb_rngv2.route import apply_hapb_rng_route

    return apply_hapb_rng_route(
        sim_cfg,
        hapb_rng_semantics_version=hapb_rng_semantics_version,
    )
