"""Bounded B48 multi-tick driver over the committed resident R9 seam."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Final, Mapping

import torch

from .c4_fixture import make_c4_fixture_v1
from .c4_mating import TensorWorldC4MateBirthConfigV1
from .c4_resident_mutation_r4 import _validate_replay_structure
from .c4_resident_population_r9 import (
    TensorWorldC4ResidentPopulationR9StagingV1,
    execute_c4_resident_population_r9_cuda_v1,
)
from .c4_state import TensorWorldStateBundleV4
from .c4_transaction import (
    C4PopulationTransactionError,
    execute_c4_population_transaction_oracle_v1,
)
from .c5_determinism import (
    TensorWorldC5ReplayContextV1,
    make_c5_pcg64_context_v1,
)
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
R9_ADAPTER = ROOT / "engine/tensor_world/c4_resident_population_r9.py"
R9_KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r9.cu"
CPU_TRANSACTION = ROOT / "engine/tensor_world/c4_transaction.py"

C4_RESIDENT_MULTITICK_R10_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_multitick_r10_v1"
)
C4_RESIDENT_MULTITICK_R10_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_MULTITICK_R10_BATCH_V1: Final[int] = 48
C4_RESIDENT_MULTITICK_R10_CAPACITY_V1: Final[int] = 32
C4_RESIDENT_MULTITICK_R10_TICKS_V1: Final[int] = 5


@dataclass(slots=True)
class C4ResidentMultiTickR10FixtureV1:
    bundle: TensorWorldStateBundleV4
    replay: TensorWorldC5ReplayContextV1
    config: TensorWorldC4MateBirthConfigV1
    reused_slot: int = 1
    generation_start: int = 11
    tick_start: int = 100
    schema_version: str = C4_RESIDENT_MULTITICK_R10_SCHEMA_VERSION_V1

    def validate(self) -> None:
        if self.schema_version != C4_RESIDENT_MULTITICK_R10_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R10 fixture schema is unsupported")
        if (
            self.bundle.batch != C4_RESIDENT_MULTITICK_R10_BATCH_V1
            or self.bundle.capacity != C4_RESIDENT_MULTITICK_R10_CAPACITY_V1
            or self.reused_slot != 1
        ):
            raise TensorWorldContractError("C4 R10 requires B48 C32 and slot 1")
        if (
            not self.config.allow_mutation
            or self.config.max_births != 1
            or self.config.cooldown_steps != 0
        ):
            raise TensorWorldContractError(
                "C4 R10 requires one mutating mate birth without cooldown"
            )
        if self.generation_start < 0 or self.tick_start < 0:
            raise TensorWorldContractError("C4 R10 time origins must be nonnegative")
        self.bundle.validate()
        if self.bundle.device.type == "cpu":
            self.replay.validate_pcg64_for_bundle(self.bundle)
        elif self.bundle.device.type == "cuda":
            _validate_replay_structure(self.bundle, self.replay)
        else:
            raise TensorWorldContractError("C4 R10 requires CPU oracle or CUDA state")


@dataclass(frozen=True, slots=True)
class C4ResidentMultiTickR10TickReceiptV1:
    tick_index: int
    generation: int
    tick: int
    preflight_status: tuple[int, ...]
    dead_agent_ids: tuple[int, ...]
    child_agent_ids: tuple[int, ...]
    slot_tokens: tuple[int, ...]
    next_agent_ids: tuple[int, ...]
    event_active_counts: tuple[int, ...]
    pcg_cursors: tuple[int, ...]
    pcg_semantic_draws: tuple[int, ...]
    python_raw_draws: tuple[int, ...]
    python_semantic_draws: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class C4ResidentMultiTickR10RunReceiptV1:
    status: str
    requested_ticks: int
    completed_ticks: int
    ticks: tuple[C4ResidentMultiTickR10TickReceiptV1, ...]
    failure_tick: int | None = None
    failure_status: tuple[int, ...] | None = None
    rollback_applied: bool = False
    orchestration: str = (
        "host_python_tick_loop_with_per_tick_state_status_d2h_and_synchronization"
    )
    staging_reused: bool = True
    schema_version: str = C4_RESIDENT_MULTITICK_R10_SCHEMA_VERSION_V1
    decision: str = C4_RESIDENT_MULTITICK_R10_DECISION_V1
    gate_claim: bool = False


@dataclass(slots=True)
class _R10BoundarySnapshotV1:
    tensors: Mapping[int, tuple[torch.Tensor, torch.Tensor]]
    tick: int


def make_c4_resident_multitick_r10_fixture_v1(
    *, seed: int = 67, capacity: int = C4_RESIDENT_MULTITICK_R10_CAPACITY_V1
) -> C4ResidentMultiTickR10FixtureV1:
    """Create a dense-map-safe B48 fixture for repeated same-slot reuse."""

    if capacity != C4_RESIDENT_MULTITICK_R10_CAPACITY_V1:
        raise TensorWorldContractError("C4 R10 requires capacity=32")
    bundle, _ = make_c4_fixture_v1(
        batch=C4_RESIDENT_MULTITICK_R10_BATCH_V1,
        capacity=capacity,
        active=4,
        width=8,
        seed=seed,
        world_local_identity=True,
    )
    core = bundle.base.base.core
    patch = bundle.base.base.patch.agent_grid_i32
    positions = ((2, 2), (5, 5), (2, 1), (6, 6))
    core.occupancy.agent_id_grid.fill_(-1)
    core.occupancy.slot_grid.fill_(-1)
    patch.fill_(-1)
    for world in range(bundle.batch):
        for slot, (y, x) in enumerate(positions):
            agent_id = int(core.agents.agent_id[world, slot])
            core.agents.pos_y[world, slot] = y
            core.agents.pos_x[world, slot] = x
            core.occupancy.agent_id_grid[world, y, x] = agent_id
            core.occupancy.slot_grid[world, y, x] = slot
            patch[world, y, x] = agent_id
    bundle.population.free_queue.fill_(-1)
    bundle.population.free_queue_count.zero_()
    core.agents.energy[:, :4] = 1000.0
    bundle.base.agent.policy_energy_observation[:, :4] = 1000.0
    bundle.base.agent.repro_threshold[:, :4] = 0.0
    bundle.base.genome.mutation_rate[:, :4] = 1.0
    scalar = bundle.scalar_rng
    if scalar is None:
        raise TensorWorldContractError("C4 R10 requires CPython scalar RNG state")
    scalar.python_random_gauss_primary_tape.fill_(0.01)
    scalar.python_random_gauss_secondary_tape.zero_()
    fixture = C4ResidentMultiTickR10FixtureV1(
        bundle=bundle,
        replay=make_c5_pcg64_context_v1(bundle, streams=2, seed=seed),
        config=TensorWorldC4MateBirthConfigV1(
            max_births=1,
            parent_energy_share=0.1,
            cooldown_steps=0,
            max_energy=1000.0,
            allow_mutation=True,
        ),
    )
    fixture.validate()
    return fixture


def _snapshot_tensor_leaves_v1(
    value: object, output: dict[int, tuple[torch.Tensor, torch.Tensor]]
) -> None:
    if isinstance(value, torch.Tensor):
        output.setdefault(id(value), (value, value.detach().clone()))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            _snapshot_tensor_leaves_v1(getattr(value, field.name), output)


def _capture_boundary_v1(
    fixture: C4ResidentMultiTickR10FixtureV1,
) -> _R10BoundarySnapshotV1:
    tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    _snapshot_tensor_leaves_v1(fixture.bundle, tensors)
    _snapshot_tensor_leaves_v1(fixture.replay, tensors)
    return _R10BoundarySnapshotV1(
        tensors=tensors, tick=int(fixture.bundle.base.base.core.tick)
    )


def _restore_boundary_v1(
    fixture: C4ResidentMultiTickR10FixtureV1,
    snapshot: _R10BoundarySnapshotV1,
) -> None:
    with torch.no_grad():
        for target, saved in snapshot.tensors.values():
            target.copy_(saved)
    fixture.bundle.base.base.core.tick = snapshot.tick


def _prepare_tick_v1(
    fixture: C4ResidentMultiTickR10FixtureV1, *, tick_index: int
) -> tuple[int, ...]:
    bundle = fixture.bundle
    slot = fixture.reused_slot
    dead_ids = tuple(
        int(value)
        for value in bundle.base.base.core.agents.agent_id[:, slot].detach().cpu().tolist()
    )
    bundle.base.lifecycle.death_candidate.zero_()
    bundle.base.lifecycle.death_candidate[:, slot] = True
    bundle.base.base.core.agents.health[:, slot] = 0.0
    bundle.base.base.core.tick = fixture.tick_start + tick_index
    return dead_ids


def _inject_event_capacity_failure_v1(
    fixture: C4ResidentMultiTickR10FixtureV1, *, world: int
) -> None:
    events = fixture.bundle.events
    if events is None:
        raise TensorWorldContractError("C4 R10 typed event tape is required")
    events.active_count[world] = events.capacity - 1
    fixture.replay.event_sequence.copy_(events.active_count)


def _tick_receipt_v1(
    fixture: C4ResidentMultiTickR10FixtureV1,
    *,
    tick_index: int,
    dead_ids: tuple[int, ...],
    status: torch.Tensor | None,
) -> C4ResidentMultiTickR10TickReceiptV1:
    bundle = fixture.bundle
    population = bundle.population
    events = bundle.events
    scalar = bundle.scalar_rng
    semantic = fixture.replay.semantic_draw_counters
    if events is None or scalar is None or semantic is None:
        raise TensorWorldContractError("C4 R10 receipt state is incomplete")

    def values(tensor: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(value) for value in tensor.detach().cpu().tolist())

    zero_status = torch.zeros(
        bundle.batch, dtype=torch.int32, device=bundle.device
    )
    return C4ResidentMultiTickR10TickReceiptV1(
        tick_index=tick_index,
        generation=fixture.generation_start + tick_index,
        tick=fixture.tick_start + tick_index,
        preflight_status=values(zero_status if status is None else status),
        dead_agent_ids=dead_ids,
        child_agent_ids=values(
            bundle.base.base.core.agents.agent_id[:, fixture.reused_slot]
        ),
        slot_tokens=values(population.slot_tokens[:, fixture.reused_slot]),
        next_agent_ids=values(population.next_agent_id),
        event_active_counts=values(events.active_count),
        pcg_cursors=values(fixture.replay.rng_cursors[:, fixture.config.environment_stream]),
        pcg_semantic_draws=values(semantic[:, fixture.config.environment_stream]),
        python_raw_draws=values(scalar.python_random_raw_draws),
        python_semantic_draws=values(scalar.mutation_py_semantic_draws),
    )


def _validate_run_arguments_v1(
    fixture: C4ResidentMultiTickR10FixtureV1,
    *,
    ticks: int,
    inject_failure_tick: int | None,
    inject_failure_world: int,
) -> None:
    fixture.validate()
    if type(ticks) is not int or not (1 <= ticks <= C4_RESIDENT_MULTITICK_R10_TICKS_V1):
        raise TensorWorldContractError("C4 R10 supports one through five ticks")
    if inject_failure_tick is not None and not (0 <= inject_failure_tick < ticks):
        raise TensorWorldContractError("C4 R10 failure tick is outside the run")
    if not (0 <= inject_failure_world < C4_RESIDENT_MULTITICK_R10_BATCH_V1):
        raise TensorWorldContractError("C4 R10 failure world is outside B48")


def run_c4_resident_multitick_r10_oracle_v1(
    fixture: C4ResidentMultiTickR10FixtureV1,
    *,
    ticks: int = C4_RESIDENT_MULTITICK_R10_TICKS_V1,
    inject_failure_tick: int | None = None,
    inject_failure_world: int = 47,
) -> C4ResidentMultiTickR10RunReceiptV1:
    """Run the exact CPU transaction oracle with a full per-tick rollback."""

    _validate_run_arguments_v1(
        fixture,
        ticks=ticks,
        inject_failure_tick=inject_failure_tick,
        inject_failure_world=inject_failure_world,
    )
    if fixture.bundle.device.type != "cpu":
        raise TensorWorldContractError("C4 R10 oracle driver requires CPU state")
    records: list[C4ResidentMultiTickR10TickReceiptV1] = []
    for tick_index in range(ticks):
        snapshot = _capture_boundary_v1(fixture)
        dead_ids = _prepare_tick_v1(fixture, tick_index=tick_index)
        if inject_failure_tick == tick_index:
            _inject_event_capacity_failure_v1(fixture, world=inject_failure_world)
        try:
            execute_c4_population_transaction_oracle_v1(
                fixture.bundle,
                mate_config=replace(
                    fixture.config, generation=fixture.generation_start + tick_index
                ),
                rng_context=fixture.replay,
            )
        except C4PopulationTransactionError:
            _restore_boundary_v1(fixture, snapshot)
            return C4ResidentMultiTickR10RunReceiptV1(
                status="INJECTED_FAILURE_ROLLED_BACK",
                requested_ticks=ticks,
                completed_ticks=len(records),
                ticks=tuple(records),
                failure_tick=tick_index,
                rollback_applied=True,
            )
        records.append(
            _tick_receipt_v1(
                fixture,
                tick_index=tick_index,
                dead_ids=dead_ids,
                status=None,
            )
        )
    return C4ResidentMultiTickR10RunReceiptV1(
        status="MULTITICK_PASS",
        requested_ticks=ticks,
        completed_ticks=ticks,
        ticks=tuple(records),
    )


def run_c4_resident_multitick_r10_cuda_v1(
    fixture: C4ResidentMultiTickR10FixtureV1,
    staging: TensorWorldC4ResidentPopulationR9StagingV1,
    *,
    ticks: int = C4_RESIDENT_MULTITICK_R10_TICKS_V1,
    inject_failure_tick: int | None = None,
    inject_failure_world: int = 47,
    build_directory: str | Path | None = None,
) -> C4ResidentMultiTickR10RunReceiptV1:
    """Run R9 once per tick with host-observed status and full rollback."""

    _validate_run_arguments_v1(
        fixture,
        ticks=ticks,
        inject_failure_tick=inject_failure_tick,
        inject_failure_world=inject_failure_world,
    )
    if fixture.bundle.device.type != "cuda":
        raise TensorWorldContractError("C4 R10 CUDA driver requires CUDA state")
    records: list[C4ResidentMultiTickR10TickReceiptV1] = []
    for tick_index in range(ticks):
        snapshot = _capture_boundary_v1(fixture)
        dead_ids = _prepare_tick_v1(fixture, tick_index=tick_index)
        if inject_failure_tick == tick_index:
            _inject_event_capacity_failure_v1(fixture, world=inject_failure_world)
        execute_c4_resident_population_r9_cuda_v1(
            fixture.bundle,
            staging,
            replace(fixture.config, generation=fixture.generation_start + tick_index),
            rng_context=fixture.replay,
            build_directory=build_directory,
        )
        failure_status = tuple(
            int(value) for value in staging.preflight_status.detach().cpu().tolist()
        )
        if any(value != 0 for value in failure_status):
            _restore_boundary_v1(fixture, snapshot)
            torch.cuda.synchronize(fixture.bundle.device)
            return C4ResidentMultiTickR10RunReceiptV1(
                status="INJECTED_FAILURE_ROLLED_BACK",
                requested_ticks=ticks,
                completed_ticks=len(records),
                ticks=tuple(records),
                failure_tick=tick_index,
                failure_status=failure_status,
                rollback_applied=True,
            )
        records.append(
            _tick_receipt_v1(
                fixture,
                tick_index=tick_index,
                dead_ids=dead_ids,
                status=staging.preflight_status,
            )
        )
    return C4ResidentMultiTickR10RunReceiptV1(
        status="MULTITICK_PASS",
        requested_ticks=ticks,
        completed_ticks=ticks,
        ticks=tuple(records),
    )


def c4_resident_multitick_r10_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__),
        "r9_adapter": R9_ADAPTER,
        "r9_kernel": R9_KERNEL,
        "cpu_transaction": CPU_TRANSACTION,
    }
    return {
        "schema_version": C4_RESIDENT_MULTITICK_R10_SCHEMA_VERSION_V1,
        "route": "R10_B48_C32_FIVE_TICK_R9_DRIVER",
        "decision": C4_RESIDENT_MULTITICK_R10_DECISION_V1,
        "batch": C4_RESIDENT_MULTITICK_R10_BATCH_V1,
        "capacity": C4_RESIDENT_MULTITICK_R10_CAPACITY_V1,
        "ticks": C4_RESIDENT_MULTITICK_R10_TICKS_V1,
        "execution_classification": (
            "host_python_tick_loop_with_per_tick_state_status_d2h_and_synchronization"
        ),
        "staging_reused": True,
        "rollback_boundary": "all_bundle_and_replay_tensor_leaves_plus_scalar_tick",
        "rollback_scope": "semantic_preflight_failure; runtime_fault_atomicity_not_claimed",
        "persistent_residency_claim": False,
        "cuda_graph_claim": False,
        "gate_claim": False,
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C4_RESIDENT_MULTITICK_R10_BATCH_V1",
    "C4_RESIDENT_MULTITICK_R10_CAPACITY_V1",
    "C4_RESIDENT_MULTITICK_R10_DECISION_V1",
    "C4_RESIDENT_MULTITICK_R10_SCHEMA_VERSION_V1",
    "C4_RESIDENT_MULTITICK_R10_TICKS_V1",
    "C4ResidentMultiTickR10FixtureV1",
    "C4ResidentMultiTickR10RunReceiptV1",
    "C4ResidentMultiTickR10TickReceiptV1",
    "c4_resident_multitick_r10_source_receipt_v1",
    "make_c4_resident_multitick_r10_fixture_v1",
    "run_c4_resident_multitick_r10_cuda_v1",
    "run_c4_resident_multitick_r10_oracle_v1",
]
