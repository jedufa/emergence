"""Bounded B48 multi-generation harness over the committed resident R9 route."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Final

import torch

from .c4_fixture import make_c4_fixture_v1
from .c4_mating import TensorWorldC4MateBirthConfigV1
from .c4_resident_mutation_r4 import _validate_replay_structure
from .c4_resident_population_r9 import (
    TensorWorldC4ResidentPopulationR9StagingV1,
    execute_c4_resident_population_r9_cuda_v1,
)
from .c4_state import TensorWorldStateBundleV4
from .c4_transaction import execute_c4_population_transaction_oracle_v1
from .c5_determinism import (
    TensorWorldC5ReplayContextV1,
    make_c5_pcg64_context_v1,
)
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
R9_ADAPTER = ROOT / "engine/tensor_world/c4_resident_population_r9.py"
R9_KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r9.cu"
CPU_TRANSACTION = ROOT / "engine/tensor_world/c4_transaction.py"
CPU_MATING = ROOT / "engine/tensor_world/c4_mating.py"

C4_RESIDENT_MULTIGEN_R10_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_multigen_r10_harness_v1"
)
C4_RESIDENT_MULTIGEN_R10_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C4_RESIDENT_MULTIGEN_R10_BATCH_V1: Final[int] = 48
C4_RESIDENT_MULTIGEN_R10_CAPACITY_V1: Final[int] = 32
C4_RESIDENT_MULTIGEN_R10_TICKS_V1: Final[int] = 5
C4_RESIDENT_MULTIGEN_R10_FIRST_GENERATION_V1: Final[int] = 10


@dataclass(slots=True)
class C4ResidentMultigenR10FixtureV1:
    bundle: TensorWorldStateBundleV4
    replay: TensorWorldC5ReplayContextV1
    config: TensorWorldC4MateBirthConfigV1
    reused_slot: int = 1
    ticks: int = C4_RESIDENT_MULTIGEN_R10_TICKS_V1
    first_generation: int = C4_RESIDENT_MULTIGEN_R10_FIRST_GENERATION_V1
    schema_version: str = C4_RESIDENT_MULTIGEN_R10_SCHEMA_VERSION_V1

    def validate(self) -> None:
        if self.schema_version != C4_RESIDENT_MULTIGEN_R10_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 R10 fixture schema is unsupported")
        if (
            self.bundle.batch != C4_RESIDENT_MULTIGEN_R10_BATCH_V1
            or self.bundle.capacity != C4_RESIDENT_MULTIGEN_R10_CAPACITY_V1
            or self.bundle.grid_size != 8
        ):
            raise TensorWorldContractError("C4 R10 requires B48/C32/grid8")
        if self.reused_slot != 1 or self.ticks != 5 or self.first_generation != 10:
            raise TensorWorldContractError("C4 R10 turnover envelope diverges")
        if (
            not self.config.allow_mutation
            or self.config.max_births != 1
            or self.config.life_history_mode != 0
        ):
            raise TensorWorldContractError("C4 R10 requires one default mate mutation")
        self.bundle.validate()
        if self.bundle.device.type == "cpu":
            self.replay.validate_pcg64_for_bundle(self.bundle)
        else:
            _validate_replay_structure(self.bundle, self.replay)


def make_c4_resident_multigen_r10_fixture_v1(
    *, seed: int = 79
) -> C4ResidentMultigenR10FixtureV1:
    """Create a dense-map-safe B48 fixture for five same-slot generations."""

    bundle, _ = make_c4_fixture_v1(
        batch=C4_RESIDENT_MULTIGEN_R10_BATCH_V1,
        capacity=C4_RESIDENT_MULTIGEN_R10_CAPACITY_V1,
        active=4,
        width=8,
        seed=seed,
        world_local_identity=True,
    )
    core = bundle.base.base.core
    patch = bundle.base.base.patch.agent_grid_i32
    core.occupancy.agent_id_grid.fill_(-1)
    core.occupancy.slot_grid.fill_(-1)
    patch.fill_(-1)
    y = torch.tensor((2, 5, 2, 6), dtype=torch.int64)
    x = torch.tensor((2, 5, 1, 6), dtype=torch.int64)
    worlds = torch.arange(bundle.batch, dtype=torch.int64)[:, None]
    slots = torch.arange(4, dtype=torch.int64)[None, :]
    core.agents.pos_y[:, :4] = y
    core.agents.pos_x[:, :4] = x
    ids = core.agents.agent_id[:, :4]
    core.occupancy.agent_id_grid[worlds, y[None, :], x[None, :]] = ids
    core.occupancy.slot_grid[worlds, y[None, :], x[None, :]] = slots
    patch[worlds, y[None, :], x[None, :]] = ids.to(torch.int32)
    bundle.population.free_queue.fill_(-1)
    bundle.population.free_queue_count.zero_()
    bundle.base.lifecycle.death_candidate.zero_()
    core.agents.energy[:, :4] = 1000.0
    bundle.base.agent.policy_energy_observation[:, :4] = 1000.0
    bundle.base.agent.repro_threshold[:, :4] = 0.0
    bundle.base.genome.mutation_rate[:, :4] = 1.0
    if bundle.scalar_rng is None:
        raise TensorWorldContractError("C4 R10 requires explicit CPython RNG state")
    bundle.scalar_rng.python_random_gauss_primary_tape.fill_(0.01)
    bundle.scalar_rng.python_random_gauss_secondary_tape.zero_()
    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=seed)
    fixture = C4ResidentMultigenR10FixtureV1(
        bundle=bundle,
        replay=replay,
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


def prepare_c4_resident_multigen_r10_tick_v1(
    fixture: C4ResidentMultigenR10FixtureV1, *, tick_index: int
) -> TensorWorldC4MateBirthConfigV1:
    """Materialize one deterministic death and its generation without world loops."""

    fixture.validate()
    if type(tick_index) is not int or not (0 <= tick_index < fixture.ticks):
        raise TensorWorldContractError("C4 R10 tick index is outside the envelope")
    bundle = fixture.bundle
    bundle.base.base.core.tick = tick_index + 1
    bundle.base.lifecycle.death_candidate.zero_()
    bundle.base.lifecycle.death_candidate[:, fixture.reused_slot] = True
    bundle.base.base.core.agents.health[:, fixture.reused_slot] = 0.0
    return replace(
        fixture.config,
        generation=fixture.first_generation + tick_index,
    )


def execute_c4_resident_multigen_r10_oracle_tick_v1(
    fixture: C4ResidentMultigenR10FixtureV1, *, tick_index: int
) -> None:
    """Advance one CPU transaction through the rollback-owning coordinator."""

    if fixture.bundle.device.type != "cpu":
        raise TensorWorldContractError("C4 R10 oracle tick requires CPU")
    config = prepare_c4_resident_multigen_r10_tick_v1(
        fixture, tick_index=tick_index
    )
    execute_c4_population_transaction_oracle_v1(
        fixture.bundle,
        mate_config=config,
        rng_context=fixture.replay,
    )


def execute_c4_resident_multigen_r10_cuda_tick_v1(
    fixture: C4ResidentMultigenR10FixtureV1,
    staging: TensorWorldC4ResidentPopulationR9StagingV1,
    *,
    tick_index: int,
    build_directory: str | Path | None = None,
) -> TensorWorldC4ResidentPopulationR9StagingV1:
    """Advance one CUDA tick through the frozen R9 resident transaction."""

    if fixture.bundle.device.type != "cuda":
        raise TensorWorldContractError("C4 R10 native tick requires CUDA")
    config = prepare_c4_resident_multigen_r10_tick_v1(
        fixture, tick_index=tick_index
    )
    return execute_c4_resident_population_r9_cuda_v1(
        fixture.bundle,
        staging,
        config,
        rng_context=fixture.replay,
        build_directory=build_directory,
    )


def c4_resident_multigen_r10_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__),
        "frozen_r9_adapter": R9_ADAPTER,
        "frozen_r9_kernel": R9_KERNEL,
        "cpu_transaction": CPU_TRANSACTION,
        "cpu_mating": CPU_MATING,
    }
    return {
        "schema_version": C4_RESIDENT_MULTIGEN_R10_SCHEMA_VERSION_V1,
        "route": "R10_B48_C32_FIVE_TICK_FIVE_GENERATION_R9_HARNESS",
        "decision": C4_RESIDENT_MULTIGEN_R10_DECISION_V1,
        "ticks": C4_RESIDENT_MULTIGEN_R10_TICKS_V1,
        "generations": list(
            range(
                C4_RESIDENT_MULTIGEN_R10_FIRST_GENERATION_V1,
                C4_RESIDENT_MULTIGEN_R10_FIRST_GENERATION_V1
                + C4_RESIDENT_MULTIGEN_R10_TICKS_V1,
            )
        ),
        "batch": C4_RESIDENT_MULTIGEN_R10_BATCH_V1,
        "capacity": C4_RESIDENT_MULTIGEN_R10_CAPACITY_V1,
        "identity_boundary": "dense_world_local_map_with_headroom_through_five_births",
        "rollback_boundary": "one_R9_tick_batch_global_semantic_failure",
        "runtime_fault_atomicity": "inherits_R9_not_claimed_for_multi_kernel_launch_faults",
        "execution_classification": "host_loop_harness_not_single_resident_multi_tick_kernel",
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
    "C4_RESIDENT_MULTIGEN_R10_BATCH_V1",
    "C4_RESIDENT_MULTIGEN_R10_CAPACITY_V1",
    "C4_RESIDENT_MULTIGEN_R10_DECISION_V1",
    "C4_RESIDENT_MULTIGEN_R10_FIRST_GENERATION_V1",
    "C4_RESIDENT_MULTIGEN_R10_SCHEMA_VERSION_V1",
    "C4_RESIDENT_MULTIGEN_R10_TICKS_V1",
    "C4ResidentMultigenR10FixtureV1",
    "c4_resident_multigen_r10_source_receipt_v1",
    "execute_c4_resident_multigen_r10_cuda_tick_v1",
    "execute_c4_resident_multigen_r10_oracle_tick_v1",
    "make_c4_resident_multigen_r10_fixture_v1",
    "prepare_c4_resident_multigen_r10_tick_v1",
]
