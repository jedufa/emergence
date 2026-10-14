"""Static proof tests for the admitted C4 R11 RNG budget."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.tensor_world.c4_resident_rng_budget_r11 import (
    C4_RESIDENT_RNG_BUDGET_R11_DECISION_V1,
    C4_RESIDENT_RNG_BUDGET_R11_SCHEMA_VERSION_V1,
    c4_resident_rng_budget_r11_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
PCG = ROOT / "engine/perf/cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"
R4 = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_mutation_r4.cu"
R4_ADAPTER = ROOT / "engine/tensor_world/c4_resident_mutation_r4.py"
STATE = ROOT / "engine/tensor_world/c4_state.py"


def test_r11_exact_finite_semantic_budgets_cover_admitted_geometry() -> None:
    receipt = c4_resident_rng_budget_r11_v1()

    assert receipt["schema_version"] == C4_RESIDENT_RNG_BUDGET_R11_SCHEMA_VERSION_V1
    assert receipt["decision"] == C4_RESIDENT_RNG_BUDGET_R11_DECISION_V1
    assert receipt["geometry"] == {
        "batch": 48,
        "world_width": 32,
        "active_population": 256,
        "capacity": 512,
        "max_births": 64,
    }
    pcg = receipt["pcg64"]
    assert pcg["planner_decisions_per_world_upper"] == 256
    assert pcg["runner_semantic_draws_per_world_upper"] == 1024
    assert pcg["environment_semantic_draws_per_world_upper"] == 2048
    assert pcg["semantic_draws_batch_upper"] == 147_456
    assert pcg["runner_bounded_calls_per_world_upper"] == 768
    assert pcg["environment_bounded_calls_per_world_upper"] == 1792
    assert pcg["bounded_calls_batch_upper"] == 122_880
    assert pcg["rejection_prone_calls_per_world_upper"] == 1280

    mt = receipt["mt19937_gaussian"]
    assert mt["semantic_draws_per_birth_upper"] == 21
    assert mt["semantic_draws_per_world_upper"] == 1344
    assert mt["semantic_draws_batch_upper"] == 64_512
    assert mt["random_double_calls_per_birth"] == 11
    assert mt["gaussian_calls_per_birth_upper"] == 9
    assert mt["fresh_gaussian_tape_reads_per_world"] == 288


def test_r11_legacy_loops_remain_frozen_but_r11b_window_is_bounded() -> None:
    receipt = c4_resident_rng_budget_r11_v1()
    pcg = receipt["pcg64"]
    mt = receipt["mt19937_gaussian"]
    capacity = receipt["capacity_contract"]

    assert pcg["raw32_outputs_per_world_upper"] is None
    assert pcg["raw64_transitions_per_world_upper"] is None
    assert mt["raw32_outputs_per_world_upper"] is None
    assert mt["mt_twists_per_world_upper"] is None
    assert "no_retry_cap" in pcg["unbounded_reason"]
    assert "no_retry_cap" in mt["unbounded_reason"]
    assert capacity["status"] == "ADMITTED_R11B_FIVE_TICK_ONLY_LONG_RUN_OPEN"
    assert capacity["current_route_admitted"] is True
    assert capacity["admitted_window_ticks"] == 5
    assert capacity["frozen_retry_cap"] == 64
    assert capacity["r11b_gaussian_tape_capacity"] == 65_536
    assert capacity["native_retry_cap_binding_present"] is True
    assert capacity["long_run_continuation_admitted"] is False
    assert capacity["gate_claim"] is False

    pcg_source = PCG.read_text(encoding="utf-8")
    r4_source = R4.read_text(encoding="utf-8")
    assert "do {\n        value = pcg64_raw32(rng) & mask;" in pcg_source
    assert "} while (value > max_value);" in pcg_source
    assert "while (value >= upper) value = getrandbits(rng, bit_count);" in r4_source


def test_r11_conditional_caps_prove_cache_transitions_twists_and_tape_headroom() -> None:
    no_rejection = c4_resident_rng_budget_r11_v1(
        pcg_max_raw32_per_bounded_draw=1,
        mt_max_randbelow_attempts=1,
    )
    pcg = no_rejection["pcg64"]["conditional_capped_budget"]
    mt = no_rejection["mt19937_gaussian"]["conditional_capped_budget"]
    assert pcg == {
        "runner_raw32_outputs_per_world": 768,
        "environment_raw32_outputs_per_world": 1792,
        "runner_raw64_transitions_per_world_worst_cache": 384,
        "environment_raw64_transitions_per_world_worst_cache": 896,
        "raw32_outputs_per_world": 2560,
        "raw32_outputs_batch": 122_880,
        "raw64_transitions_batch_worst_cache": 61_440,
    }
    assert mt["raw32_outputs_per_world"] == 2624
    assert mt["raw32_outputs_batch"] == 125_952
    assert mt["twists_per_world_worst_initial_index"] == 5
    assert mt["twists_batch_worst_initial_index"] == 240
    assert mt["gaussian_tape_headroom_per_world"] == 2618
    assert mt["initial_has_gauss_false"]["fresh_gaussian_tape_reads"] == 288
    assert mt["initial_has_gauss_true"]["fresh_gaussian_tape_reads"] == 288

    capped = c4_resident_rng_budget_r11_v1(
        pcg_max_raw32_per_bounded_draw=8,
        mt_max_randbelow_attempts=8,
    )
    pcg8 = capped["pcg64"]["conditional_capped_budget"]
    mt8 = capped["mt19937_gaussian"]["conditional_capped_budget"]
    assert pcg8["runner_raw32_outputs_per_world"] == 2560
    assert pcg8["environment_raw32_outputs_per_world"] == 8960
    assert mt8["raw32_outputs_per_world"] == 3072
    assert mt8["gaussian_tape_headroom_per_world"] == 3059


def test_r11_default_gaussian_tape_cannot_cover_even_no_rejection_m64() -> None:
    receipt = c4_resident_rng_budget_r11_v1()
    mt = receipt["mt19937_gaussian"]

    assert mt["no_rejection_raw32_outputs_per_world"] == 2624
    assert mt["no_rejection_gaussian_tape_headroom_per_world"] == 2618
    assert mt["default_gaussian_tape_capacity"] == 2048
    assert receipt["capacity_contract"][
        "no_rejection_tape_deficit_at_raw_draws_zero"
    ] == 570
    assert receipt["capacity_contract"][
        "current_default_tape_passes_no_rejection_M64"
    ] is False
    assert "tape_capacity: int = 2048" in STATE.read_text(encoding="utf-8")
    assert all(len(row["sha256"]) == 64 for row in receipt["sources"].values())


def test_r11_typed_raw_pcg_counters_are_bound_only_by_the_r11_replay_abi() -> None:
    receipt = c4_resident_rng_budget_r11_v1()
    adapter = R4_ADAPTER.read_text(encoding="utf-8")
    replay_body = adapter.split("def _replay_tensors", 1)[1].split(
        "def _staging_tensors", 1
    )[0]

    assert "raw64_transition_counts" not in replay_body
    assert "output32_return_counts" not in replay_body
    assert "R11-only eight-tensor replay ABI" in receipt["counter_headroom"][
        "typed_raw_counter_binding"
    ]


@pytest.mark.parametrize(
    ("pcg_cap", "mt_cap"),
    ((0, None), (-1, None), (None, 0), (None, -1), (True, None)),
)
def test_r11_retry_capacity_inputs_fail_closed(
    pcg_cap: int | None, mt_cap: int | None
) -> None:
    with pytest.raises(TensorWorldContractError):
        c4_resident_rng_budget_r11_v1(
            pcg_max_raw32_per_bounded_draw=pcg_cap,
            mt_max_randbelow_attempts=mt_cap,
        )
