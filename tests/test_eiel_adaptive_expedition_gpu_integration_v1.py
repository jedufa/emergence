"""Physical CUDA integration checks for the Adaptive Expedition boundary.

These tests deliberately prove only physical packing/state/event behavior.  They
do not claim long-horizon expedition efficiency or qualification.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_PREBUILT = os.environ.get("EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT")
_REQUIRE_GPU_GATE = os.environ.get("EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU") == "1"
try:
    import torch
except Exception:  # pragma: no cover - collection without the CUDA environment
    torch = None

pytestmark = pytest.mark.skipif(
    not _PREBUILT or torch is None or not torch.cuda.is_available(),
    reason="requires EIEL_FULL_EVOLUTION_CUDA_V2_PREBUILT and a physical CUDA device",
)

from engine.perf.cuda_full_evolution_v2.contract import (  # noqa: E402
    CudaFullEvolutionNativeContractV2,
)
from engine.perf.cuda_full_evolution_v2.runtime import (  # noqa: E402
    CudaFullEvolutionRuntimeV2,
    seal_event_sink_receipt_v2,
)
from eiel_composition.adaptive_expedition.coordinator import (  # noqa: E402
    prepare_decision_v1,
)
from eiel_composition.adaptive_expedition.observation_adapter import (  # noqa: E402
    project_authenticated_science_request_observations_v1,
)
from eiel_composition.adaptive_expedition.policies.adaptive_illumination_v1 import (  # noqa: E402
    AdaptiveIlluminationPolicyV1,
)
from tests.test_eiel_adaptive_expedition_coordinator_v1 import (  # noqa: E402
    _atlas,
    _bank,
    _manifest,
)

ARTIFACT = Path(_PREBUILT or "")
_OBSERVATION_RUN_ROOT = os.environ.get("EIEL_ADAPTIVE_GPU_OBSERVATION_RUN_ROOT")

if _REQUIRE_GPU_GATE:
    if not ARTIFACT.is_file():
        raise RuntimeError(
            "EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU=1 requires an existing "
            f"prebuilt artifact: {ARTIFACT}"
        )
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError(
            "EIEL_REQUIRE_CUDA_FULL_EVOLUTION_V2_GPU=1 requires a physical CUDA device"
        )


def _run_physical_batch(
    batch: int,
    *,
    world_uids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    assert len(world_uids) == batch
    owner = CudaFullEvolutionRuntimeV2.initialize(
        CudaFullEvolutionNativeContractV2(
            B=batch,
            total_ticks=2,
            temporal_k=2,
        ),
        seed=20260830,
        artifact_path=ARTIFACT,
        world_uids=world_uids,
    )
    try:
        owner.force_test_conditions()
        owner.advance(2)
        dispatch = owner.drain_events(to_cpu=True)
        receipt = seal_event_sink_receipt_v2(
            dispatch,
            sink_id="adaptive-expedition-physical-packing-test",
            sink_receipt_id=f"B{batch}-{dispatch.content_sha256[:24]}",
        )
        owner.acknowledge_event_drain(receipt)
        assert dispatch.world_uids == world_uids
        assert any(int(value) > 0 for value in dispatch.event_counts.tolist())
        return (
            owner.state.detach().cpu().clone(),
            dispatch.event_rows.detach().cpu().clone(),
            dispatch.event_counts.detach().cpu().clone(),
            dispatch.world_uids,
        )
    finally:
        owner.close()


def test_physical_b16_b64_b1024_packing_is_state_and_event_exact() -> None:
    """Actual B16/B64/B1024 launches preserve per-world observations."""

    world_uids = tuple(100_001 + index for index in range(1024))
    observations: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]] = {}
    for batch in (16, 64, 1024):
        observations[batch] = _run_physical_batch(
            batch,
            world_uids=world_uids[:batch],
        )

    reference_state, reference_events, reference_counts, reference_uids = observations[1024]
    for batch in (16, 64):
        state, events, counts, uids = observations[batch]
        assert uids == world_uids[:batch]
        assert reference_uids[:batch] == uids
        assert torch.equal(state, reference_state[:batch])
        assert torch.equal(events, reference_events[:batch])
        assert torch.equal(counts, reference_counts[:batch])


def test_authenticated_gpu_observation_feeds_a_replayable_policy_decision() -> None:
    """A retained GPU Science Request result reaches the policy seam exactly."""

    if not _OBSERVATION_RUN_ROOT:
        pytest.skip(
            "set EIEL_ADAPTIVE_GPU_OBSERVATION_RUN_ROOT to a completed GPU Science Request"
        )

    policy = AdaptiveIlluminationPolicyV1()
    manifest = _manifest(policy)
    projection = project_authenticated_science_request_observations_v1(
        _OBSERVATION_RUN_ROOT,
        manifest=manifest,
    )
    assert projection.frames
    assert all(
        frame.source_state_ref.state_class.value == "OBSERVATION_ONLY"
        for frame in projection.frames
    )

    prepared = prepare_decision_v1(
        manifest=manifest,
        policy=policy,
        frontier_id="frontier-gpu-observation-0000",
        frontier_ordinal=0,
        observation_frames=projection.frames,
        bounded_history=[],
        atlas=_atlas(manifest),
        retained_parent_bank=_bank(manifest),
        logical_capacity=2,
        outer_rng_cursor=0,
    )
    replayed = prepare_decision_v1(
        manifest=manifest,
        policy=policy,
        frontier_id="frontier-gpu-observation-0000",
        frontier_ordinal=0,
        observation_frames=projection.frames,
        bounded_history=[],
        atlas=_atlas(manifest),
        retained_parent_bank=_bank(manifest),
        logical_capacity=2,
        outer_rng_cursor=0,
    )
    assert prepared.decision.to_dict() == replayed.decision.to_dict()
    assert prepared.decision.policy_proposal.to_value()["authority_effect"] == (
        "NONE_PROPOSAL_ONLY"
    )
