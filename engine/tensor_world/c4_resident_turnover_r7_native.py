"""Five-tick native R7 parity harness over the frozen resident R5 route."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Final

from .c4_mating import (
    TensorWorldC4MateBirthConfigV1,
    commit_c4_mate_births_oracle_v1,
    plan_c4_mate_births_oracle_v1,
)
from .c4_population import apply_c4_death_settlement_oracle_v1
from .c4_resident_population_r5 import (
    TensorWorldC4ResidentPopulationR5StagingV1,
    execute_c4_resident_population_r5_cuda_v1,
)
from .c4_resident_mutation_r4 import _validate_replay_structure
from .c4_resident_turnover_r7 import make_c4_resident_turnover_r7_fixture_v1
from .c4_state import TensorWorldStateBundleV4
from .c5_determinism import TensorWorldC5ReplayContextV1
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
R5_ADAPTER = ROOT / "engine/tensor_world/c4_resident_population_r5.py"
R5_KERNEL = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_population_r5.cu"
CPU_DEATH = ROOT / "engine/tensor_world/c4_population.py"
CPU_MATE = ROOT / "engine/tensor_world/c4_mating.py"
CPU_MUTATION = ROOT / "engine/tensor_world/c4_genome.py"

C4_RESIDENT_TURNOVER_R7_NATIVE_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_turnover_r7_native_harness_v1"
)
C4_RESIDENT_TURNOVER_R7_NATIVE_DECISION_V1: Final[str] = "NO_GATE_DECISION"


@dataclass(slots=True)
class C4ResidentTurnoverR7NativeFixtureV1:
    bundle: TensorWorldStateBundleV4
    replay: TensorWorldC5ReplayContextV1
    config: TensorWorldC4MateBirthConfigV1
    reused_slot: int = 1
    schema_version: str = C4_RESIDENT_TURNOVER_R7_NATIVE_SCHEMA_VERSION_V1

    def validate(self) -> None:
        if self.schema_version != C4_RESIDENT_TURNOVER_R7_NATIVE_SCHEMA_VERSION_V1:
            raise TensorWorldContractError("C4 native R7 fixture schema is unsupported")
        if self.bundle.batch != 4 or self.bundle.capacity < 32 or self.reused_slot != 1:
            raise TensorWorldContractError("C4 native R7 requires B4 C>=32 and slot 1")
        if not self.config.allow_mutation or self.config.max_births != 1:
            raise TensorWorldContractError("C4 native R7 requires one mutating mate birth")
        self.bundle.validate()
        if self.bundle.device.type == "cpu":
            self.replay.validate_pcg64_for_bundle(self.bundle)
        else:
            _validate_replay_structure(self.bundle, self.replay)


def make_c4_resident_turnover_r7_native_fixture_v1(
    *, capacity: int = 32, seed: int = 53
) -> C4ResidentTurnoverR7NativeFixtureV1:
    """Create a mate-compatible capacity-safe repeated slot-reuse fixture."""

    prepared = make_c4_resident_turnover_r7_fixture_v1(
        capacity=capacity, width=8, seed=seed
    )
    bundle = prepared.bundle
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
    bundle.base.genome.mutation_rate[:, :4] = 1.0
    scalar = bundle.scalar_rng
    if scalar is None:
        raise TensorWorldContractError("C4 native R7 requires CPython scalar RNG state")
    scalar.python_random_gauss_primary_tape.fill_(0.01)
    scalar.python_random_gauss_secondary_tape.zero_()
    fixture = C4ResidentTurnoverR7NativeFixtureV1(
        bundle=bundle,
        replay=prepared.replay,
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


def prepare_c4_resident_turnover_r7_tick_v1(
    bundle: TensorWorldStateBundleV4, *, reused_slot: int = 1
) -> None:
    """Materialize the next deterministic death input without world loops."""

    if not isinstance(bundle, TensorWorldStateBundleV4):
        raise TensorWorldContractError("C4 native R7 requires a V4 bundle")
    if bundle.batch != 4 or bundle.capacity < 32 or reused_slot != 1:
        raise TensorWorldContractError("C4 native R7 tick requires B4 C>=32 and slot 1")
    bundle.base.lifecycle.death_candidate.zero_()
    bundle.base.lifecycle.death_candidate[:, reused_slot] = True
    bundle.base.base.core.agents.health[:, reused_slot] = 0.0


def execute_c4_resident_turnover_r7_oracle_tick_v1(
    fixture: C4ResidentTurnoverR7NativeFixtureV1, *, tick_index: int
) -> None:
    """Advance one exact CPU death/mate/mutation oracle tick."""

    fixture.validate()
    if fixture.bundle.device.type != "cpu" or type(tick_index) is not int or tick_index < 0:
        raise TensorWorldContractError("C4 native R7 CPU tick input is invalid")
    prepare_c4_resident_turnover_r7_tick_v1(
        fixture.bundle, reused_slot=fixture.reused_slot
    )
    config = replace(fixture.config, generation=tick_index + 1)
    apply_c4_death_settlement_oracle_v1(
        fixture.bundle, event_sequence=fixture.replay.event_sequence
    )
    plan_c4_mate_births_oracle_v1(
        fixture.bundle, config, rng_context=fixture.replay
    )
    commit_c4_mate_births_oracle_v1(
        fixture.bundle, config, rng_context=fixture.replay
    )


def execute_c4_resident_turnover_r7_cuda_tick_v1(
    fixture: C4ResidentTurnoverR7NativeFixtureV1,
    staging: TensorWorldC4ResidentPopulationR5StagingV1,
    *,
    tick_index: int,
    build_directory: str | Path | None = None,
) -> TensorWorldC4ResidentPopulationR5StagingV1:
    """Advance one CUDA tick through R5; this remains a test harness."""

    fixture.validate()
    if fixture.bundle.device.type != "cuda" or type(tick_index) is not int or tick_index < 0:
        raise TensorWorldContractError("C4 native R7 CUDA tick input is invalid")
    prepare_c4_resident_turnover_r7_tick_v1(
        fixture.bundle, reused_slot=fixture.reused_slot
    )
    return execute_c4_resident_population_r5_cuda_v1(
        fixture.bundle,
        staging,
        replace(fixture.config, generation=tick_index + 1),
        rng_context=fixture.replay,
        build_directory=build_directory,
    )


def c4_resident_turnover_r7_native_source_receipt_v1() -> dict[str, object]:
    sources = {
        "adapter": Path(__file__), "r5_adapter": R5_ADAPTER, "r5_kernel": R5_KERNEL,
        "cpu_death": CPU_DEATH, "cpu_mate": CPU_MATE, "cpu_mutation": CPU_MUTATION,
    }
    return {
        "schema_version": C4_RESIDENT_TURNOVER_R7_NATIVE_SCHEMA_VERSION_V1,
        "route": "R7_B4_C32_FIVE_TICK_R5_TEST_HARNESS",
        "decision": C4_RESIDENT_TURNOVER_R7_NATIVE_DECISION_V1,
        "ticks": 5,
        "reused_slot": 1,
        "mutation": True,
        "execution_classification": "host_launched_component_test_not_production_residency",
        "event_failure_rollback": "single_R5_tick_batch_global_unchanged_boundary",
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
    "C4_RESIDENT_TURNOVER_R7_NATIVE_DECISION_V1",
    "C4_RESIDENT_TURNOVER_R7_NATIVE_SCHEMA_VERSION_V1",
    "C4ResidentTurnoverR7NativeFixtureV1",
    "c4_resident_turnover_r7_native_source_receipt_v1",
    "execute_c4_resident_turnover_r7_cuda_tick_v1",
    "execute_c4_resident_turnover_r7_oracle_tick_v1",
    "make_c4_resident_turnover_r7_native_fixture_v1",
    "prepare_c4_resident_turnover_r7_tick_v1",
]
