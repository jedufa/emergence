from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
import torch

from engine.option_d.benchmark import (
    ARTIFACT_BINDING,
    CURRENT_C5_TAPE_CAPACITY,
    _bootstrap_median_interval,
    _capture_candidate_checkpoint,
    _current_c5_admission,
    _current_schedule,
    _d6_roi_decision,
    _gaussian_component_schedule,
    _material_host_tensors,
    _paired_opportunity_percent,
    _percentile,
    _raw_control_schedule,
    _sample_summary,
    run_d6_benchmark,
)
from engine.option_d.checkpoint import CheckpointSnapshot
from engine.option_d.contracts import Operator, RNGStateBundle
from engine.option_d.cpu import execute_cpu, state_from_seed


def test_percentile_uses_linear_order_statistic_interpolation() -> None:
    values = [4.0, 1.0, 3.0, 2.0]

    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 0.5) == 2.5
    assert _percentile(values, 1.0) == 4.0
    with pytest.raises(ValueError, match="nonempty"):
        _percentile([], 0.5)


def test_component_schedules_are_source_shaped_and_valid() -> None:
    full = _current_schedule(3, 5)
    raw = _raw_control_schedule(full)
    gaussian = _gaussian_component_schedule(full)

    assert all(
        operation.operator is not Operator.GAUSS
        for world in raw.worlds
        for operation in world
    )
    assert all(
        operation.operator is Operator.GAUSS and operation.predicate is None
        for world in gaussian.worlds
        for operation in world
    )
    assert all(len(world) == 12 * 5 for world in raw.worlds)
    assert all(len(world) == 9 * 5 for world in gaussian.worlds)
    raw.validate(expected_worlds=3)
    gaussian.validate(expected_worlds=3)


def test_host_tape_material_preserves_bits_and_execution_mask() -> None:
    schedule = _current_schedule(2, 2)
    bundle = RNGStateBundle(
        (state_from_seed(101), state_from_seed(202)),
        source_binding="benchmark-contract-test",
    )
    result = execute_cpu(bundle, schedule)

    bits, executed = _material_host_tensors(result)

    assert bits.dtype is torch.uint64
    assert executed.dtype is torch.bool
    for world_index, world in enumerate(result.material.worlds):
        for operation_index, value in enumerate(world):
            assert int(bits[world_index, operation_index]) == value.value_bits
            assert bool(executed[world_index, operation_index]) is value.executed


def test_checkpoint_timer_accepts_a_precomputed_schedule_digest() -> None:
    class Owner:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def capture_checkpoint(
            self,
            *,
            artifact_binding: str,
            schedule_sha256: str,
        ) -> CheckpointSnapshot:
            self.calls.append((artifact_binding, schedule_sha256))
            data = b"checkpoint"
            return CheckpointSnapshot(data, sha256(data).hexdigest())

    owner = Owner()
    assert ARTIFACT_BINDING.endswith("-v2")
    digest, record = _capture_candidate_checkpoint(
        owner,  # type: ignore[arg-type]
        schedule_sha256="a" * 64,
    )

    assert owner.calls == [(ARTIFACT_BINDING, "a" * 64)]
    assert digest == sha256(b"checkpoint").hexdigest()
    assert record["bytes"] == len(b"checkpoint")
    assert record["end_to_end_ms"] >= 0.0


def test_current_c5_admission_matches_frozen_five_tick_headroom() -> None:
    no_cache = state_from_seed(17)
    cached = replace(state_from_seed(19), gauss_next_bits=0)

    no_cache_at_limit = replace(
        no_cache,
        counters=replace(no_cache.counters, raw_words=32_327),
    )
    cached_at_limit = replace(
        cached,
        counters=replace(cached.counters, raw_words=32_325),
    )
    admitted = _current_c5_admission(
        RNGStateBundle(
            (no_cache_at_limit, cached_at_limit),
            source_binding="benchmark-contract-test",
        ),
        base_raw_words=(0, 0),
        capacity=CURRENT_C5_TAPE_CAPACITY,
    )

    assert admitted["headroom_by_world"] == (33_209, 33_211)
    assert admitted["refill_required"] is False

    past_limit = replace(
        no_cache_at_limit,
        counters=replace(no_cache_at_limit.counters, raw_words=32_328),
    )
    rejected = _current_c5_admission(
        RNGStateBundle(
            (past_limit, cached_at_limit),
            source_binding="benchmark-contract-test",
        ),
        base_raw_words=(0, 0),
        capacity=CURRENT_C5_TAPE_CAPACITY,
    )

    assert rejected["refill_required"] is True


def test_paired_roi_uses_direct_complete_cycle_observations() -> None:
    observations = [
        _paired_opportunity_percent(baseline_ns=1_000, candidate_ns=value)
        for value in (960, 940, 900, 930, 950)
    ]

    assert observations == [4.0, 6.0, 10.0, 7.0, 5.0]
    assert _sample_summary(observations)["median"] == 6.0
    assert _bootstrap_median_interval([6.0] * 5, resamples=16) == {
        "confidence": 0.95,
        "high": 6.0,
        "low": 6.0,
        "resamples": 16,
    }


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        ([6.0] * 5, "D6-ROI-GO"),
        ([4.0] * 5, "D6-ROI-NO-GO"),
        ([4.0, 4.0, 5.0, 6.0, 6.0], "D6-ROI-INCONCLUSIVE"),
    ],
)
def test_d6_roi_decision_requires_sufficient_evidence_at_the_threshold(
    observations: list[float],
    expected: str,
) -> None:
    decision, summary, interval = _d6_roi_decision(observations)

    assert decision == expected
    assert summary["samples"] == len(observations)
    assert interval["confidence"] == 0.95


@pytest.mark.parametrize("samples", [30, 32, 64])
def test_d6_rejects_a_non_frozen_decision_sample(
    tmp_path: Path,
    samples: int,
) -> None:
    with pytest.raises(ValueError, match="31 or 63 paired samples"):
        run_d6_benchmark(
            output=tmp_path / "unused.json",
            build_root=tmp_path,
            source_binding="benchmark-contract-test",
            resource_admission="RADM-TEST",
            samples=samples,
        )
