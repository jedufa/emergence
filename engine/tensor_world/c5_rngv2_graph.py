"""Default-off live C0–C7 graph fusion for qualified V2 mutation/birth RNG."""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
import time
from typing import Any

import torch

from engine.tw_rngv2.contract import NAMESPACE_DEFAULT, SEMANTIC_EPOCH

from .c4_event_ring_r11 import make_c4_event_ring_r11_v1
from .c4_fixture import make_c4_fixture_v1
from .c4_mating import TensorWorldC4MateBirthConfigV1
from .c4_resident_composed_r11 import (
    C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1,
    bind_c4_resident_composed_r11_v1,
    capture_c4_resident_composed_r11_window_v1,
    execute_c4_resident_composed_r11_window_cuda_v1,
    make_c4_resident_composed_r11_staging_v1,
    prepare_c4_resident_composed_r11_window_control_v1,
)
from .c4_state import make_c4_cpython_random_state_v1
from .c5_determinism import make_c5_pcg64_context_v1
from .c5_rngv2_checkpoint import capture_v2_checkpoint, restore_v2_checkpoint
from .c5_rngv2_route import (
    RNG_SEMANTICS_V2,
    V2_ROUTE_ID,
    assert_v2_no_host_tapes,
    production_contract_hash,
    production_contract_identity,
    reject_v1_v2_mix,
)
from .contracts import TensorWorldContractError


GRAPH_FUSION_ROUTE_ID = "tensor_world_c0_c7_rngv2_graph_fusion_v1"
GRAPH_WINDOW_TICKS = C4_RESIDENT_COMPOSED_R11_GRAPH_WINDOW_TICKS_V1


def _move_in_place(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_in_place(item, device) for item in value)
    if isinstance(value, list):
        return [_move_in_place(item, device) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            setattr(value, field.name, _move_in_place(getattr(value, field.name), device))
    return value


def make_v2_graph_fusion_state(
    *,
    capacity: int,
    active: int,
    width: int,
    seed: int,
    source_commit: str,
    enable_births: bool = True,
) -> tuple[Any, Any]:
    """Build the real R11 B48 owner state with zero mutation/birth tapes."""

    bundle, _ = make_c4_fixture_v1(
        batch=48,
        capacity=capacity,
        active=active,
        width=width,
        seed=seed,
        world_local_identity=True,
    )
    bundle.scalar_rng = make_c4_cpython_random_state_v1(
        batch=48, tape_capacity=0, seed=seed
    )
    assert_v2_no_host_tapes(bundle.scalar_rng)
    reject_v1_v2_mix(rng_semantics_version=RNG_SEMANTICS_V2, tape_capacity=0)
    if enable_births:
        n = min(active, capacity)
        bundle.base.base.core.agents.energy[:, :n] = 160.0
        bundle.base.base.core.agents.health[:, :n] = 12.0
        bundle.base.agent.policy_energy_observation[:, :n] = 160.0
        bundle.base.agent.repro_threshold[:, :n] = 0.0
        bundle.base.genome.reproduction_threshold[:, :n] = 0.0
        bundle.base.genome.mutation_rate[:, :n] = 1.0
    replay = make_c5_pcg64_context_v1(
        bundle, streams=5, seed=seed, typed=True, source_commit=source_commit
    )
    assert bundle.events is not None
    # Long multi-window runs need a bounded resident event bank larger than
    # one historical five-tick window.  This is not a Gaussian tape.
    bundle.event_ring = make_c4_event_ring_r11_v1(
        batch=48,
        capacity=capacity,
        event_capacity=max(int(bundle.events.capacity), 4096),
    )
    bundle.validate()
    replay.validate_typed_continuation_for_bundle(bundle)
    return (
        _move_in_place(bundle, torch.device("cuda")),
        _move_in_place(replay, torch.device("cuda")),
    )


def bind_v2_graph_fusion_owner(
    *,
    capacity: int = 32,
    active: int = 4,
    width: int = 8,
    seed: int = 97,
    source_commit: str,
    max_births: int = 1,
    native_build_directory: str | None = None,
    enable_births: bool = True,
    c2_extension_path: str | None = None,
) -> Any:
    bundle, replay = make_v2_graph_fusion_state(
        capacity=capacity,
        active=active,
        width=width,
        seed=seed,
        source_commit=source_commit,
        enable_births=enable_births,
    )
    staging = make_c4_resident_composed_r11_staging_v1(
        batch=48,
        capacity=capacity,
        grid_size=bundle.grid_size,
        streams=int(replay.rng_state_words.shape[1]),
        device=bundle.device,
    )
    config = TensorWorldC4MateBirthConfigV1(
        max_births=max_births, allow_mutation=True, max_energy=160.0
    )
    owner = bind_c4_resident_composed_r11_v1(
        bundle,
        replay,
        staging,
        enable_native=True,
        population_config=config,
        c2_extension_path=c2_extension_path or os.environ.get("TENSOR_WORLD_C2_PREBUILT"),
        native_build_directory=native_build_directory,
        source_commit=source_commit,
        rng_semantics_version=2,
    )
    capture_c4_resident_composed_r11_window_v1(owner, config)
    return owner


def v2_window_receipt(owner: Any) -> dict[str, object]:
    if owner.rng_semantics_version != 2 or owner.last_v2_window_receipt is None:
        raise TensorWorldContractError("V2 graph window receipt is missing")
    return dict(owner.last_v2_window_receipt)


def run_v2_graph_windows(
    owner: Any,
    *,
    windows: int,
    start_tick: int = 0,
    generation: int = 0,
) -> dict[str, object]:
    if windows < 1:
        raise TensorWorldContractError("V2 graph fusion requires a positive window count")
    recaptures = 0
    refills = 0
    host_gen_ms = 0.0
    tape_bytes = 0
    h2d_bytes = 0
    try:
        prepare_c4_resident_composed_r11_window_control_v1(
            owner, generation=generation, tick=start_tick
        )
    except Exception as exc:
        raise TensorWorldContractError(
            f"V2 graph initial seed tick={start_tick} control failed: {exc}"
        ) from exc
    seed_receipt = v2_window_receipt(owner)
    if seed_receipt.get("graph_recapture"):
        recaptures += 1
    if seed_receipt.get("refill_performed"):
        refills += 1
    host_gen_ms += float(seed_receipt["mutation_birth_host_generation_ms"])
    tape_bytes = max(tape_bytes, int(seed_receipt["mutation_birth_tape_bytes"]))
    h2d_bytes = max(h2d_bytes, int(seed_receipt["mutation_birth_h2d_bytes"]))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(windows)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(windows)]
    last_result = None
    for index in range(windows):
        starts[index].record()
        last_result = execute_c4_resident_composed_r11_window_cuda_v1(owner)
        ends[index].record()
    torch.cuda.synchronize()
    window_ms = [float(start.elapsed_time(end)) for start, end in zip(starts, ends)]
    terminal_status = int(last_result.global_status.max().item()) if last_result is not None else 0
    owner.v2_host_status_d2h += 1
    if terminal_status != 0:
        raise TensorWorldContractError(
            f"V2 graph terminal harvest published nonzero status {terminal_status}"
        )
    return {
        "windows": windows,
        "ticks": windows * GRAPH_WINDOW_TICKS,
        "graph_replays": windows,
        "graph_recaptures": recaptures,
        "refills": refills,
        "mutation_birth_host_generation_ms": host_gen_ms,
        "mutation_birth_tape_bytes": tape_bytes,
        "mutation_birth_h2d_bytes": h2d_bytes,
        "nonzero_status_windows": 0 if terminal_status == 0 else 1,
        "host_prepare_calls": int(owner.v2_host_prepare_calls),
        "host_status_d2h": int(owner.v2_host_status_d2h),
        "host_event_reclaim_copies": int(owner.v2_host_reclaim_copies),
        "terminal_status": terminal_status,
        "route_id": V2_ROUTE_ID,
        "graph_fusion_route_id": GRAPH_FUSION_ROUTE_ID,
        "contract": production_contract_identity(),
        "contract_hash": production_contract_hash(),
        "namespace": NAMESPACE_DEFAULT,
        "semantic_epoch": SEMANTIC_EPOCH,
        "window_ms": window_ms,
    }


def capture_graph_plus_v2_frontier(owner: Any, *, absolute_tick: int) -> dict[str, object]:
    next_ids = owner.bundle.population.next_agent_id
    snap = capture_v2_checkpoint(
        world_uids=tuple(range(owner.bundle.batch)),
        absolute_tick=absolute_tick,
        next_entity_uid=tuple(int(value) for value in next_ids.detach().cpu().tolist()),
    )
    restored = restore_v2_checkpoint(snap.to_dict())
    return {
        "schema": snap.schema_version,
        "digest": snap.digest(),
        "restored_digest": restored.digest(),
        "host_tape_bytes": snap.host_tape_bytes,
        "equal": snap.digest() == restored.digest(),
    }


__all__ = [
    "GRAPH_FUSION_ROUTE_ID",
    "GRAPH_WINDOW_TICKS",
    "bind_v2_graph_fusion_owner",
    "capture_graph_plus_v2_frontier",
    "make_v2_graph_fusion_state",
    "run_v2_graph_windows",
    "v2_window_receipt",
]
