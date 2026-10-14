"""C5 exact CPython Gaussian-window continuation and native ABI contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engine.tensor_world.c4_fixture import make_c4_fixture_v1
from engine.tensor_world.c5_determinism import (
    C5_GAUSSIAN_WINDOW_AUTHORITATIVE_R11_V1,
    C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1,
    capture_c5_checkpoint_v1,
    make_c5_pcg64_context_v1,
    restore_c5_checkpoint_v1,
)
from engine.tensor_world.c5_gaussian_window_r11 import (
    admit_c5_gaussian_window_r11_v1,
    gaussian_window_tensors_r11_v1,
    refill_c5_gaussian_window_r11_v1,
)
from engine.tensor_world.contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[1]
R11_NATIVE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_graph_r11.cu"
R4_NATIVE = ROOT / "engine/tensor_world/native/tensor_world_c4_resident_mutation_r4.cu"


def test_typed_context_publishes_exact_initial_window_and_native_abi() -> None:
    bundle, _ = make_c4_fixture_v1(
        batch=2, capacity=8, active=3, width=8, seed=211
    )
    context = make_c5_pcg64_context_v1(
        bundle, streams=2, seed=211, typed=True
    )
    window = context.gaussian_window
    assert window is not None and bundle.scalar_rng is not None
    assert window.base_raw_draws.tolist() == [0, 0]
    assert window.valid_raw_draw_ends.tolist() == [
        bundle.scalar_rng.tape_capacity
    ] * 2
    assert window.refill_generations.tolist() == [1, 1]
    assert window.status.tolist() == [
        C5_GAUSSIAN_WINDOW_AUTHORITATIVE_R11_V1
    ] * 2
    assert window.exhaustion_raw_draws.tolist() == [-1, -1]
    assert bool(window.provenance_sha256.any())
    assert len(gaussian_window_tensors_r11_v1(context)) == 6
    context.validate_typed_continuation_for_bundle(bundle)

    second_bundle, _ = make_c4_fixture_v1(
        batch=2, capacity=8, active=3, width=8, seed=211
    )
    second = make_c5_pcg64_context_v1(
        second_bundle,
        streams=2,
        seed=211,
        typed=True,
        source_commit="different-source",
    )
    assert second.gaussian_window is not None
    assert not torch.equal(
        window.provenance_sha256,
        second.gaussian_window.provenance_sha256,
    )


def test_boundary_refill_preserves_counters_and_checkpoint_restores_metadata() -> None:
    bundle, _ = make_c4_fixture_v1(
        batch=1, capacity=8, active=3, width=8, seed=223
    )
    context = make_c5_pcg64_context_v1(
        bundle, streams=2, seed=223, typed=True
    )
    state = bundle.scalar_rng
    window = context.gaussian_window
    assert state is not None and window is not None
    state.python_random_raw_draws.fill_(state.tape_capacity - 1)
    raw_before = state.python_random_raw_draws.clone()
    semantic_before = state.mutation_py_semantic_draws.clone()
    provenance_before = window.provenance_sha256.clone()

    admission = admit_c5_gaussian_window_r11_v1(
        bundle,
        context,
        requested_ticks=5,
        max_births_per_tick=1,
        boundary_kind="control",
        source_commit="c5-test-source",
        boundary_label="generation_4_tick_20",
    )
    assert admission.refill_performed is True
    assert torch.equal(state.python_random_raw_draws, raw_before)
    assert torch.equal(state.mutation_py_semantic_draws, semantic_before)
    assert window.base_raw_draws.tolist() == raw_before.tolist()
    assert window.valid_raw_draw_ends.tolist() == [
        int(raw_before[0]) + state.tape_capacity
    ]
    assert window.refill_generations.tolist() == [2]
    assert not torch.equal(window.provenance_sha256, provenance_before)

    checkpoint = capture_c5_checkpoint_v1(
        bundle,
        context,
        backend_id="r11-test",
        source_commit="c5-test-source",
        provider_contract_id="c5-test-provider",
    )
    expected_base = window.base_raw_draws.clone()
    expected_end = window.valid_raw_draw_ends.clone()
    expected_provenance = window.provenance_sha256.clone()
    window.refill_generations.add_(100)
    window.provenance_sha256.fill_(1)
    restore_c5_checkpoint_v1(
        checkpoint,
        bundle,
        context,
        expected_backend_id="r11-test",
        expected_source_commit="c5-test-source",
        expected_provider_contract_id="c5-test-provider",
    )
    assert torch.equal(window.base_raw_draws, expected_base)
    assert torch.equal(window.valid_raw_draw_ends, expected_end)
    assert torch.equal(window.provenance_sha256, expected_provenance)


def test_refill_rejects_undeclared_boundary_and_exhaustion_is_persisted() -> None:
    bundle, _ = make_c4_fixture_v1(
        batch=1, capacity=8, active=3, width=8, seed=227
    )
    context = make_c5_pcg64_context_v1(
        bundle, streams=2, seed=227, typed=True
    )
    with pytest.raises(TensorWorldContractError, match="boundary"):
        refill_c5_gaussian_window_r11_v1(
            bundle,
            context,
            boundary_kind="normal_tick",  # type: ignore[arg-type]
            source_commit="c5-test-source",
            boundary_label="forbidden",
        )
    with pytest.raises(TensorWorldContractError, match="physical capacity"):
        admit_c5_gaussian_window_r11_v1(
            bundle,
            context,
            requested_ticks=5,
            max_births_per_tick=64,
            boundary_kind="initialization",
            source_commit="c5-test-source",
            boundary_label="capacity_failure",
        )
    assert context.gaussian_window is not None
    assert context.gaussian_window.status.tolist() == [
        C5_GAUSSIAN_WINDOW_EXHAUSTED_R11_V1
    ]
    assert context.gaussian_window.exhaustion_raw_draws.tolist() == [0]


def test_r11_native_extends_only_the_production_scalar_rng_abi() -> None:
    r11 = R11_NATIVE.read_text(encoding="utf-8")
    r4 = R4_NATIVE.read_text(encoding="utf-8")
    assert "#define EIEL_C5_GAUSSIAN_WINDOW_V1 1" in r11
    assert "scalar_rng.size() == 8 || scalar_rng.size() == 14" in r11
    assert "legacy_scalar_rng" in r11
    assert "window_base_raw_draws" in r11
    assert "window_valid_raw_draw_ends" in r11
    assert "window_provenance_sha256" in r11
    assert "state.python.raw_draws[world] - window_base" in r4
    assert "rng.raw_draws + window_base" in r4
