"""Exact C4 R11 mutation/PCG/MT draw-budget contract.

The admitted geometry is B48/W32/P256/C512 with at most M64 births per
world.  Semantic planner and mutation draws have finite exact upper bounds.
Legacy R2-R9 compilation preserves the source-current unbounded helpers.  R11
defines cap-64 compile-time variants, binds typed raw/output counters, and
admits one five-tick graph window with exact remaining-capacity checks.  That
bounded window is not a long-run continuation claim: the finite resident MT
tape still requires a future device-local refill or ring.
"""

from __future__ import annotations

from hashlib import sha256
from math import ceil
from pathlib import Path
from typing import Final

from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
PCG_SOURCE = ROOT / "engine/perf/cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"
R4_SOURCE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_mutation_r4.cu"
R4_ADAPTER = ROOT / "engine/tensor_world/c4_resident_mutation_r4.py"
MT_SOURCE = ROOT / "engine/perf/cuda_architecture_b_persistent_birth_scalar_genome.cuh"
CPU_GENOME = ROOT / "engine/tensor_world/c4_genome.py"
C4_STATE = ROOT / "engine/tensor_world/c4_state.py"
C5_STATE = ROOT / "engine/tensor_world/c5_determinism.py"
R11_OWNER = ROOT / "engine/tensor_world/c4_resident_composed_r11.py"
R11_ADMISSION = ROOT / "engine/tensor_world/c4_resident_rng_admission_r11.py"
R11_NATIVE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"

C4_RESIDENT_RNG_BUDGET_R11_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c4_resident_rng_budget_r11_v1"
)
C4_RESIDENT_RNG_BUDGET_R11_DECISION_V1: Final[str] = "NO_GATE_DECISION"

BATCH_V1: Final[int] = 48
WORLD_WIDTH_V1: Final[int] = 32
ACTIVE_POPULATION_V1: Final[int] = 256
CAPACITY_V1: Final[int] = 512
MAX_BIRTHS_V1: Final[int] = 64
MT_WORDS_V1: Final[int] = 624
DEFAULT_GAUSSIAN_TAPE_CAPACITY_V1: Final[int] = 2048
R11B_GAUSSIAN_TAPE_CAPACITY_V1: Final[int] = 65_536
R11B_RETRY_CAP_V1: Final[int] = 64
R11B_WINDOW_TICKS_V1: Final[int] = 5


def _pcg_raw64_transitions(output32: int, *, initial_cached32: bool) -> int:
    if type(output32) is not int or output32 < 0:
        raise TensorWorldContractError("C4 R11 PCG output count must be nonnegative")
    cached = 1 if initial_cached32 else 0
    return max(0, (output32 - cached + 1) // 2)


def _mt_twists(raw32_outputs: int, *, initial_index: int) -> int:
    if type(raw32_outputs) is not int or raw32_outputs < 0:
        raise TensorWorldContractError("C4 R11 MT output count must be nonnegative")
    if type(initial_index) is not int or not 0 <= initial_index <= MT_WORDS_V1:
        raise TensorWorldContractError("C4 R11 MT index must be in [0,624]")
    if raw32_outputs == 0:
        return 0
    if initial_index == MT_WORDS_V1:
        return 1 + (raw32_outputs - 1) // MT_WORDS_V1
    return (initial_index + raw32_outputs - 1) // MT_WORDS_V1


def _mt_mutation_schedule(
    *, births: int, randbelow_attempt_cap: int, initial_has_gauss: bool
) -> dict[str, int | bool]:
    if type(births) is not int or births < 0:
        raise TensorWorldContractError("C4 R11 birth count must be nonnegative")
    if type(randbelow_attempt_cap) is not int or randbelow_attempt_cap < 1:
        raise TensorWorldContractError("C4 R11 MT retry cap must be a positive int")
    raw32 = 0
    fresh_tape_indices: list[int] = []
    has_gauss = bool(initial_has_gauss)

    def random_double() -> None:
        nonlocal raw32
        raw32 += 2

    def gauss() -> None:
        nonlocal raw32, has_gauss
        if has_gauss:
            has_gauss = False
            return
        fresh_tape_indices.append(raw32)
        raw32 += 4
        has_gauss = True

    for _birth in range(births):
        random_double()  # hidden-size decision
        for _gene in range(8):
            random_double()
            gauss()
        random_double()  # mutation-rate decision
        gauss()
        random_double()  # memory-dimension decision
        raw32 += randbelow_attempt_cap
    return {
        "raw32_outputs": raw32,
        "fresh_gaussian_tape_reads": len(fresh_tape_indices),
        "last_gaussian_tape_index_offset": (
            max(fresh_tape_indices) if fresh_tape_indices else -1
        ),
        "gaussian_tape_headroom": (
            max(fresh_tape_indices) + 1 if fresh_tape_indices else 0
        ),
        "final_has_gauss": has_gauss,
    }


def c4_resident_rng_budget_r11_v1(
    *,
    pcg_max_raw32_per_bounded_draw: int | None = None,
    mt_max_randbelow_attempts: int | None = None,
) -> dict[str, object]:
    """Return exact finite semantics and conditional capped raw budgets."""

    for name, value in (
        ("pcg_max_raw32_per_bounded_draw", pcg_max_raw32_per_bounded_draw),
        ("mt_max_randbelow_attempts", mt_max_randbelow_attempts),
    ):
        if value is not None and (type(value) is not int or value < 1):
            raise TensorWorldContractError(f"C4 R11 {name} must be a positive int")

    decisions = ACTIVE_POPULATION_V1
    runner_semantic = decisions * 4
    runner_bounded = decisions * 3
    environment_semantic = decisions * 8
    environment_bounded = decisions * 7
    runner_rejection_prone = decisions
    environment_rejection_prone = decisions * 4
    runner_single_output = runner_bounded - runner_rejection_prone
    environment_single_output = environment_bounded - environment_rejection_prone

    if pcg_max_raw32_per_bounded_draw is None:
        runner_raw_upper = None
        environment_raw_upper = None
    else:
        runner_raw_upper = (
            runner_single_output
            + runner_rejection_prone * pcg_max_raw32_per_bounded_draw
        )
        environment_raw_upper = (
            environment_single_output
            + environment_rejection_prone * pcg_max_raw32_per_bounded_draw
        )

    mt_semantic_per_birth = 21
    mt_semantic_per_world = MAX_BIRTHS_V1 * mt_semantic_per_birth
    no_rejection_false = _mt_mutation_schedule(
        births=MAX_BIRTHS_V1,
        randbelow_attempt_cap=1,
        initial_has_gauss=False,
    )
    no_rejection_true = _mt_mutation_schedule(
        births=MAX_BIRTHS_V1,
        randbelow_attempt_cap=1,
        initial_has_gauss=True,
    )
    mt_capped = None
    if mt_max_randbelow_attempts is not None:
        capped_false = _mt_mutation_schedule(
            births=MAX_BIRTHS_V1,
            randbelow_attempt_cap=mt_max_randbelow_attempts,
            initial_has_gauss=False,
        )
        capped_true = _mt_mutation_schedule(
            births=MAX_BIRTHS_V1,
            randbelow_attempt_cap=mt_max_randbelow_attempts,
            initial_has_gauss=True,
        )
        raw_upper = max(
            int(capped_false["raw32_outputs"]), int(capped_true["raw32_outputs"])
        )
        mt_capped = {
            "raw32_outputs_per_world": raw_upper,
            "raw32_outputs_batch": BATCH_V1 * raw_upper,
            "twists_per_world_worst_initial_index": ceil(raw_upper / MT_WORDS_V1),
            "twists_batch_worst_initial_index": (
                BATCH_V1 * ceil(raw_upper / MT_WORDS_V1)
            ),
            "gaussian_tape_headroom_per_world": max(
                int(capped_false["gaussian_tape_headroom"]),
                int(capped_true["gaussian_tape_headroom"]),
            ),
            "initial_has_gauss_false": capped_false,
            "initial_has_gauss_true": capped_true,
        }

    pcg_capped = None
    if runner_raw_upper is not None and environment_raw_upper is not None:
        pcg_capped = {
            "runner_raw32_outputs_per_world": runner_raw_upper,
            "environment_raw32_outputs_per_world": environment_raw_upper,
            "runner_raw64_transitions_per_world_worst_cache": _pcg_raw64_transitions(
                runner_raw_upper, initial_cached32=False
            ),
            "environment_raw64_transitions_per_world_worst_cache": _pcg_raw64_transitions(
                environment_raw_upper, initial_cached32=False
            ),
            "raw32_outputs_per_world": runner_raw_upper + environment_raw_upper,
            "raw32_outputs_batch": BATCH_V1 * (
                runner_raw_upper + environment_raw_upper
            ),
            "raw64_transitions_batch_worst_cache": BATCH_V1
            * (
                _pcg_raw64_transitions(
                    runner_raw_upper, initial_cached32=False
                )
                + _pcg_raw64_transitions(
                    environment_raw_upper, initial_cached32=False
                )
            ),
        }

    sources = {
        "pcg_planner": PCG_SOURCE,
        "r4_mutation": R4_SOURCE,
        "r4_adapter": R4_ADAPTER,
        "mt_gaussian_helper": MT_SOURCE,
        "cpu_genome_oracle": CPU_GENOME,
        "c4_rng_state": C4_STATE,
        "c5_typed_continuation": C5_STATE,
        "r11_owner": R11_OWNER,
        "r11_admission": R11_ADMISSION,
        "r11_native": R11_NATIVE,
    }
    return {
        "schema_version": C4_RESIDENT_RNG_BUDGET_R11_SCHEMA_VERSION_V1,
        "decision": C4_RESIDENT_RNG_BUDGET_R11_DECISION_V1,
        "geometry": {
            "batch": BATCH_V1,
            "world_width": WORLD_WIDTH_V1,
            "active_population": ACTIVE_POPULATION_V1,
            "capacity": CAPACITY_V1,
            "max_births": MAX_BIRTHS_V1,
        },
        "pcg64": {
            "planner_decisions_per_world_upper": decisions,
            "runner_semantic_draws_per_world_upper": runner_semantic,
            "environment_semantic_draws_per_world_upper": environment_semantic,
            "semantic_draws_batch_upper": BATCH_V1
            * (runner_semantic + environment_semantic),
            "runner_bounded_calls_per_world_upper": runner_bounded,
            "environment_bounded_calls_per_world_upper": environment_bounded,
            "bounded_calls_batch_upper": BATCH_V1
            * (runner_bounded + environment_bounded),
            "rejection_prone_calls_per_world_upper": (
                runner_rejection_prone + environment_rejection_prone
            ),
            "raw32_outputs_per_world_upper": None,
            "raw64_transitions_per_world_upper": None,
            "unbounded_reason": "source_bounded_uint32_rejection_loop_has_no_retry_cap",
            "cursor_semantics": "bounded_calls_not_raw32_outputs_or_raw64_transitions",
            "cache_semantics": (
                "one_raw64_transition_returns_low32_and_caches_high32; next_raw32_"
                "consumes_cache_without_transition"
            ),
            "conditional_retry_cap": pcg_max_raw32_per_bounded_draw,
            "conditional_capped_budget": pcg_capped,
        },
        "mt19937_gaussian": {
            "semantic_draws_per_birth_upper": mt_semantic_per_birth,
            "semantic_draws_per_world_upper": mt_semantic_per_world,
            "semantic_draws_batch_upper": BATCH_V1 * mt_semantic_per_world,
            "random_double_calls_per_birth": 11,
            "random_double_calls_per_world": 11 * MAX_BIRTHS_V1,
            "gaussian_calls_per_birth_upper": 9,
            "gaussian_calls_per_world_upper": 9 * MAX_BIRTHS_V1,
            "gaussian_calls_batch_upper": BATCH_V1 * 9 * MAX_BIRTHS_V1,
            "randbelow_calls_per_birth_upper": 1,
            "fresh_gaussian_tape_reads_per_world": 288,
            "fresh_gaussian_tape_reads_batch": BATCH_V1 * 288,
            "raw32_outputs_per_world_upper": None,
            "mt_twists_per_world_upper": None,
            "unbounded_reason": "source_randbelow_2_rejection_loop_has_no_retry_cap",
            "no_rejection_raw32_outputs_per_world": 2624,
            "no_rejection_gaussian_tape_headroom_per_world": max(
                int(no_rejection_false["gaussian_tape_headroom"]),
                int(no_rejection_true["gaussian_tape_headroom"]),
            ),
            "default_gaussian_tape_capacity": DEFAULT_GAUSSIAN_TAPE_CAPACITY_V1,
            "tape_capacity_rule": (
                "tape_capacity-current_python_random_raw_draws_must_be_at_least_"
                "the_bound_headroom_for_every_world"
            ),
            "conditional_randbelow_attempt_cap": mt_max_randbelow_attempts,
            "conditional_capped_budget": mt_capped,
        },
        "counter_headroom": {
            "runner_cursor_per_world": runner_bounded,
            "environment_cursor_per_world": environment_bounded,
            "runner_semantic_per_world": runner_semantic,
            "environment_semantic_per_world": environment_semantic,
            "mutation_semantic_per_world": mt_semantic_per_world,
            "rule": "each_int64_counter_must_be_at_most_INT64_MAX-required_increment",
            "typed_raw_counter_binding": (
                "R11-only eight-tensor replay ABI binds C5 raw64/output32/semantic "
                "counters; R2-R9 retain the frozen six-tensor replay ABI"
            ),
        },
        "capacity_contract": {
            "status": "ADMITTED_R11B_FIVE_TICK_ONLY_LONG_RUN_OPEN",
            "current_route_admitted": True,
            "admitted_window_ticks": R11B_WINDOW_TICKS_V1,
            "frozen_retry_cap": R11B_RETRY_CAP_V1,
            "r11b_gaussian_tape_capacity": R11B_GAUSSIAN_TAPE_CAPACITY_V1,
            "long_run_continuation_admitted": False,
            "long_run_blocker": (
                "absolute MT raw cursor exhausts a finite tape after bounded windows; "
                "device-local refill or a resident tape ring remains required"
            ),
            "pcg_retry_cap_required": True,
            "mt_randbelow_retry_cap_required": True,
            "gaussian_tape_headroom_required": True,
            "no_rejection_tape_deficit_at_raw_draws_zero": max(
                int(no_rejection_false["gaussian_tape_headroom"]),
                int(no_rejection_true["gaussian_tape_headroom"]),
            )
            - DEFAULT_GAUSSIAN_TAPE_CAPACITY_V1,
            "current_default_tape_passes_no_rejection_M64": (
                DEFAULT_GAUSSIAN_TAPE_CAPACITY_V1
                >= max(
                    int(no_rejection_false["gaussian_tape_headroom"]),
                    int(no_rejection_true["gaussian_tape_headroom"]),
                )
            ),
            "native_retry_cap_binding_present": True,
            "required_before_admission": (
                "qualify_one_bounded_native_compile/parity/rollback_probe",
                "add_device-local_MT_tape_refill_or_resident_ring_before_long_run_GO",
            ),
            "gate_claim": False,
        },
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "ACTIVE_POPULATION_V1",
    "BATCH_V1",
    "CAPACITY_V1",
    "C4_RESIDENT_RNG_BUDGET_R11_DECISION_V1",
    "C4_RESIDENT_RNG_BUDGET_R11_SCHEMA_VERSION_V1",
    "DEFAULT_GAUSSIAN_TAPE_CAPACITY_V1",
    "MAX_BIRTHS_V1",
    "MT_WORDS_V1",
    "WORLD_WIDTH_V1",
    "c4_resident_rng_budget_r11_v1",
]
