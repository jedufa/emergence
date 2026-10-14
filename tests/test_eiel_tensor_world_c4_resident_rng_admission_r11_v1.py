"""CPU/static acceptance tests for the Audit-04 R11 RNG boundary."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_resident_rng_admission_r11 import (
    C4_R11_RNG_STATUS_COUNTER_OVERFLOW_V1,
    C4_R11_RNG_STATUS_MT_CAPACITY_V1,
    C4_R11_RNG_STATUS_PEER_FAILED_V1,
    MT_MAX_RANDBELOW_ATTEMPTS_R11_V1,
    PCG_MAX_RAW32_PER_BOUNDED_DRAW_R11_V1,
    R11_GRAPH_WINDOW_TICKS_V1,
    capped_rejection_draw_r11_v1,
    make_c4_resident_rng_native_staging_r11_v1,
    preflight_c4_resident_rng_admission_r11_v1,
)
from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c4_state import make_c4_cpython_random_state_v1
from engine.tensor_world.c5_determinism import make_c5_pcg64_context_v1
from engine.tensor_world.contracts import TensorWorldContractError


INT64_MAX = (1 << 63) - 1
ROOT = Path(__file__).resolve().parents[1]


def _fixture() -> tuple[object, object]:
    bundle, _ = make_c4_fixture_v1(
        batch=48,
        capacity=512,
        active=256,
        width=32,
        seed=107,
        world_local_identity=True,
    )
    bundle.scalar_rng = make_c4_cpython_random_state_v1(
        batch=48, tape_capacity=65536, seed=107
    )
    replay = make_c5_pcg64_context_v1(bundle, streams=5, seed=107, typed=True)
    replay.validate_typed_continuation_for_bundle(bundle)
    return bundle, replay


def _rng_snapshot(bundle: object, replay: object) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for owner_name, owner in (("scalar", bundle.scalar_rng), ("replay", replay)):
        for field in fields(owner):
            value = getattr(owner, field.name)
            if isinstance(value, torch.Tensor):
                values[f"{owner_name}.{field.name}"] = value.clone()
    return values


def test_r11_caps_and_exact_five_tick_m320_headroom_are_frozen() -> None:
    bundle, replay = _fixture()
    receipt = preflight_c4_resident_rng_admission_r11_v1(bundle, replay)

    assert PCG_MAX_RAW32_PER_BOUNDED_DRAW_R11_V1 == 64
    assert MT_MAX_RANDBELOW_ATTEMPTS_R11_V1 == 64
    assert R11_GRAPH_WINDOW_TICKS_V1 == 5
    assert receipt.admitted is True
    assert receipt.local_status.tolist() == [0] * 48
    assert receipt.window_ticks == 5
    assert receipt.max_total_births == 320
    assert receipt.pcg_environment_raw32 == 331_520
    assert receipt.pcg_runner_raw32 == 84_480
    assert receipt.mt_raw32 == 33_280
    assert receipt.mt_semantic == 6_720
    assert receipt.mt_gaussian_tape_headroom == 33_211
    assert receipt.pcg_retry_exhaustion_union_bound < 1.7e-22
    assert receipt.mt_retry_exhaustion_union_bound < 8.4e-16


def test_r11_capped_draw_preserves_first_success_and_fails_atomically() -> None:
    raw = (7, 6, 5, 4, 3, 2, 1, 0)
    # max=5 produces mask=7, so the source-current loop accepts the third row.
    assert capped_rejection_draw_r11_v1(raw, maximum=5, attempts=64) == (5, 3)
    with pytest.raises(TensorWorldContractError, match="retry cap exhausted"):
        capped_rejection_draw_r11_v1((7,) * 64, maximum=5, attempts=64)


def test_r11_c32_zero_birth_component_needs_no_m64_tape_reserve() -> None:
    bundle, _ = make_c4_fixture_v1(
        batch=48, capacity=32, active=4, width=8, seed=109,
        world_local_identity=True,
    )
    bundle.scalar_rng = make_c4_cpython_random_state_v1(
        batch=48, tape_capacity=0, seed=109
    )
    replay = make_c5_pcg64_context_v1(bundle, streams=2, seed=109, typed=True)
    receipt = preflight_c4_resident_rng_admission_r11_v1(
        bundle, replay, max_births=0
    )
    assert receipt.admitted
    assert receipt.max_total_births == 0
    assert receipt.mt_gaussian_tape_headroom == 0
    assert receipt.pcg_environment_raw32 == 0
    assert receipt.pcg_runner_raw32 == 0


def test_r11_m64_tape_failure_marks_peers_and_preserves_whole_batch() -> None:
    bundle, replay = _fixture()
    bundle.scalar_rng.python_random_raw_draws[47] = 32_326
    before = _rng_snapshot(bundle, replay)

    receipt = preflight_c4_resident_rng_admission_r11_v1(bundle, replay)

    assert receipt.admitted is False
    assert receipt.local_status[47].item() == C4_R11_RNG_STATUS_MT_CAPACITY_V1
    assert receipt.local_status[:47].tolist() == [0] * 47
    assert receipt.batch_status[47].item() == C4_R11_RNG_STATUS_MT_CAPACITY_V1
    assert receipt.batch_status[:47].tolist() == [C4_R11_RNG_STATUS_PEER_FAILED_V1] * 47
    after = _rng_snapshot(bundle, replay)
    assert before.keys() == after.keys()
    for name, expected in before.items():
        torch.testing.assert_close(after[name], expected, equal_nan=True, msg=name)


def test_r11_typed_pcg_counter_overflow_is_precommit_and_cache_exact() -> None:
    bundle, replay = _fixture()
    assert replay.typed_role_indices is not None
    assert replay.raw64_transition_counts is not None
    environment = replay.typed_role_indices[0]
    # 331,520 raw32 returns require exactly 165,760 raw64 transitions without a
    # cached high word.  The equality boundary passes; one less headroom fails.
    replay.raw64_transition_counts[:, environment] = INT64_MAX - 165_760
    assert preflight_c4_resident_rng_admission_r11_v1(bundle, replay).admitted

    replay.raw64_transition_counts[12, environment] += 1
    before = _rng_snapshot(bundle, replay)
    receipt = preflight_c4_resident_rng_admission_r11_v1(bundle, replay)
    assert receipt.local_status[12].item() == C4_R11_RNG_STATUS_COUNTER_OVERFLOW_V1
    assert receipt.batch_status[0].item() == C4_R11_RNG_STATUS_PEER_FAILED_V1
    after = _rng_snapshot(bundle, replay)
    for name, expected in before.items():
        torch.testing.assert_close(after[name], expected, equal_nan=True, msg=name)


def test_r11_mt_counter_overflow_is_precommit() -> None:
    bundle, replay = _fixture()
    bundle.scalar_rng.mutation_py_semantic_draws[3] = INT64_MAX - 6719
    before = _rng_snapshot(bundle, replay)

    receipt = preflight_c4_resident_rng_admission_r11_v1(bundle, replay)

    assert receipt.local_status[3].item() == C4_R11_RNG_STATUS_COUNTER_OVERFLOW_V1
    after = _rng_snapshot(bundle, replay)
    for name, expected in before.items():
        torch.testing.assert_close(after[name], expected, equal_nan=True, msg=name)


def test_r11_native_rng_staging_owns_full_typed_counter_projection() -> None:
    staging = make_c4_resident_rng_native_staging_r11_v1(
        batch=48, streams=8, device="cpu"
    )
    staging.validate(batch=48, streams=8, device=torch.device("cpu"))
    assert staging.post_raw64_transition_counts.shape == (48, 8)
    assert staging.post_output32_return_counts.shape == (48, 8)
    assert staging.post_raw64_transition_counts.eq(0).all()


def test_r11_native_source_stages_counters_and_keeps_r2_r9_mode_unchanged() -> None:
    graph = (ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu").read_text()
    binding = (
        ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11_binding.cpp"
    ).read_text()
    r4 = (ROOT / "engine/tensor_world/native/tensor_world_c4_resident_mutation_r4.cu").read_text()
    planner = (
        ROOT / "engine/perf/cuda_architecture_b_persistent_mate_birth_plan_v1.cuh"
    ).read_text()
    owner = (ROOT / "engine/tensor_world/c4_resident_composed_r11.py").read_text()

    assert "#define EIEL_C4_R11_PCG_MAX_RAW32_ATTEMPTS 64" in graph
    assert "#define EIEL_C4_R11_MT_MAX_RANDBELOW_ATTEMPTS 64" in graph
    assert "replay.size() == 8" in graph
    assert "rng_staging.size() == 2" in graph
    assert "requested_ticks.data_ptr<int64_t>()" in graph
    assert "completed_ticks.data_ptr<int64_t>()" in graph
    assert graph.index("preflight_rng_window<<<") < graph.index("r4::preflight_kernel<<<")
    assert graph.index("prepare_rng_counter_shadow<<<") < graph.index("r4::preflight_kernel<<<")
    assert graph.index("r5::merge_all_status<<<") < graph.index("commit_rng_counter_shadow<<<")
    assert "!::all_ok(stage.base, batch)" in graph
    assert "std::vector<torch::Tensor> rng_staging" in binding
    assert "#if defined(EIEL_C4_R11_BOUNDED_RNG_V1)" in r4
    assert "while (value >= upper) value = getrandbits" in r4
    assert "rng.retry_exhausted = true" in planner
    assert "replay.raw64_transition_counts" in owner
    assert "replay.output32_return_counts" in owner
