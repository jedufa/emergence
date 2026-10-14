from __future__ import annotations

import pytest

from emergence_contracts.adaptive_science_expedition_v1 import (
    ExpeditionManifestV1,
)
from emergence_contracts.science_request_v1 import validate_eiel_science_request_v1
from eiel_composition.adaptive_expedition import founder_loop
from eiel_composition.adaptive_expedition.coordinator import prepare_decision_v1
from eiel_composition.adaptive_expedition.policies.adaptive_illumination_v1 import (
    ADAPTIVE_ILLUMINATION_V1_PROFILE,
    AdaptiveIlluminationPolicyV1,
)
from eiel_composition.adaptive_expedition.policy_protocol import (
    canonical_plain_value_v1,
)
from tests.test_eiel_adaptive_expedition_coordinator_v1 import (
    _atlas,
    _bank,
    _frame,
    _manifest,
)


def _founder_request() -> dict[str, object]:
    return {
        "schema_version": "eiel_science_request_v1",
        "campaign_id": "EIEL-EXPEDITION-FOUNDER-BASIS-V1",
        "test": "EIEL_ADAPTIVE_EXPEDITION_FOUNDER_BASIS_V1",
        "question": {
            "description": "Declare one immutable founder basis",
            "no_claims": True,
        },
        "world": {"width": 32, "height": 32, "population": 16, "capacity": 32},
        "run": {"ticks": 16, "batch": 1, "checkpoint_ticks": [16]},
        "dimensions": {
            "condition": [1],
            "replicate": [7],
            "seed": [1103],
        },
        "world_template": {
            "initialization": {
                "initial_population": 16,
                "initial_live_count": 16,
                "founder_profiles": ["native-broad-diverse-a-v1"],
                "exact_founder_composition": [
                    {"founder_profile_id": "native-broad-diverse-a-v1", "count": 16}
                ],
                "spatial_placement_profile": "deterministic_seeded_shuffle_v1",
                "standing_variation_profile": "broad-standing-variation-v1",
            },
            "controls": {"mutation.enabled": True},
            "seeds": {
                "ecology": "${seed}",
                "founder": "${seed}",
                "initialization": "${seed}",
                "intervention": "${replicate}",
                "mutation": "${seed}",
                "policy": "${condition}",
                "task_environment": "${seed}",
            },
        },
        "replicate_dimension": "replicate",
        "observe": {
            "profile": "SCIENCE_STANDARD_V1",
            "events": ["birth", "death"],
            "state": ["genome_identity", "population_count"],
            "runtime_counters": ["birth_count", "mutation_count"],
            "snapshot_ticks": [0, 16],
        },
        "reduce": {
            "reducer": "science.compact.world_atlas.v1",
            "configuration": {
                "analysis_dimensions": ["condition", "replicate", "seed"],
                "axis_a": "condition",
                "axis_b": "seed",
                "decision_family": "EXPEDITION_FOUNDER_BASIS",
                "reporting_group": "EXPEDITION_FOUNDER_BASIS",
                "tracked_policy_loci": [0],
                "tracked_trait_loci": [1],
            },
        },
        "expect_worlds": 1,
    }


def _founder_policy() -> AdaptiveIlluminationPolicyV1:
    profile = canonical_plain_value_v1(ADAPTIVE_ILLUMINATION_V1_PROFILE)
    profile["profile_version"] = "1-founder-loop-three-world-fixture"
    profile["candidate_generators"]["global_exploration"][
        "founder_execution"
    ]["world_count"] = 3
    return AdaptiveIlluminationPolicyV1(profile)


def _prepared():
    policy = _founder_policy()
    request = validate_eiel_science_request_v1(_founder_request())
    raw_manifest = _manifest(policy).to_dict()
    raw_manifest["founder_basis_refs"] = [
        {
            "schema_version": "run_artifact_ref_v1",
            "artifact_id": "founder-science-request-v1",
            "artifact_kind": "eiel_science_request_v1",
            "artifact_schema_version": "eiel_science_request_v1",
            "locator": "opaque://expedition/founder-request-v1",
            "content_digest": request.request_sha256().to_dict(),
            "size_bytes": None,
        }
    ]
    manifest = ExpeditionManifestV1.from_dict(raw_manifest)
    prepared = prepare_decision_v1(
        manifest=manifest,
        policy=policy,
        frontier_id="frontier-founder-0000",
        frontier_ordinal=0,
        observation_frames=[_frame()],
        bounded_history=[],
        atlas=_atlas(manifest),
        retained_parent_bank=_bank(manifest),
        logical_capacity=4,
        outer_rng_cursor=0,
    )
    return manifest, request, prepared


def _action_id(prepared, action_kind: str) -> str:
    proposal = prepared.decision.policy_proposal.to_value()
    return next(
        row["action_id"]
        for row in proposal["semantic_actions"]
        if row["action_kind"] == action_kind
    )


def test_founder_child_is_deterministic_packed_and_starts_at_tick_zero() -> None:
    manifest, request, prepared = _prepared()
    action_id = _action_id(prepared, "EXPLORE_GLOBAL")
    first = founder_loop.build_founder_child_continuation_v1(
        manifest=manifest,
        decision=prepared.decision,
        action_id=action_id,
        founder_request=request,
        replicate_seeds=[101, 202, 303],
        physical_batch_worlds=3,
    )
    second = founder_loop.build_founder_child_continuation_v1(
        manifest=manifest,
        decision=prepared.decision,
        action_id=action_id,
        founder_request=request,
        replicate_seeds=[101, 202, 303],
        physical_batch_worlds=3,
    )

    assert first.to_dict() == second.to_dict()
    assert first.parent_state_ref is None
    assert first.retry_of_run_id is None
    assert first.effective_next_absolute_tick == 0
    child = validate_eiel_science_request_v1(first.child_request.to_value())
    assert child.expect_worlds == 3
    assert child.batch_plan().batch_sizes == (3,)
    assert child.to_dict()["run"]["ticks"] == manifest.epoch_ticks
    assert child.to_dict()["run"]["checkpoint_ticks"] == [manifest.epoch_ticks]
    assert child.to_dict()["question"]["no_claims"] is True


def test_founder_loop_refuses_historical_branch_semantics() -> None:
    manifest, request, prepared = _prepared()
    branch_action_id = _action_id(prepared, "BRANCH_LOCAL")
    with pytest.raises(founder_loop.FounderLoopError, match="historical/state-local"):
        founder_loop.build_founder_child_continuation_v1(
            manifest=manifest,
            decision=prepared.decision,
            action_id=branch_action_id,
            founder_request=request,
            replicate_seeds=[101, 202, 303],
        )


def test_founder_loop_requires_the_manifest_bound_basis() -> None:
    manifest, _request, prepared = _prepared()
    action_id = _action_id(prepared, "EXPLORE_GLOBAL")
    changed = _founder_request()
    changed["question"]["description"] = "Changed after manifest freeze"
    with pytest.raises(founder_loop.FounderLoopError, match="absent from the frozen"):
        founder_loop.build_founder_child_continuation_v1(
            manifest=manifest,
            decision=prepared.decision,
            action_id=action_id,
            founder_request=changed,
            replicate_seeds=[101, 202, 303],
        )


def test_run_wrapper_uses_only_the_canonical_science_front_door(monkeypatch) -> None:
    manifest, request, prepared = _prepared()
    continuation = founder_loop.build_founder_child_continuation_v1(
        manifest=manifest,
        decision=prepared.decision,
        action_id=_action_id(prepared, "EXPLORE_GLOBAL"),
        founder_request=request,
        replicate_seeds=[101, 202, 303],
    )
    calls = []

    def fake_front_door(request_value, **options):
        calls.append((request_value, options))
        return {"ok": True, "status": "COMPLETED"}

    monkeypatch.setattr(founder_loop, "run_science_request_v1", fake_front_door)
    result = founder_loop.run_founder_child_v1(
        continuation,
        repo_root="/repo",
        runtime_parent="/runtime",
        resume_root="/runtime/existing",
    )
    assert result == {"ok": True, "status": "COMPLETED"}
    assert len(calls) == 1
    assert calls[0][1]["resume_root"] == "/runtime/existing"
    assert calls[0][0] == continuation.child_request.to_value()


def test_founder_child_rejects_seed_count_outside_the_policy_binding() -> None:
    manifest, request, prepared = _prepared()
    with pytest.raises(
        founder_loop.FounderLoopError,
        match="policy-bound founder world_count",
    ):
        founder_loop.build_founder_child_continuation_v1(
            manifest=manifest,
            decision=prepared.decision,
            action_id=_action_id(prepared, "EXPLORE_GLOBAL"),
            founder_request=request,
            replicate_seeds=[101],
        )
