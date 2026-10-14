from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tools.actors.common import ActorError
from tools.scheduler.programme_contracts import (
    ContentDigest,
    LaunchpadResourceStartLeaseV1,
    ProgrammeContractError,
    ResourceStartLeaseEventCodeV1,
    ResourceStartLeaseEventV1,
    canonical_sha256,
    resource_ids_conflict_v1,
    shared_gpu_lane_candidate_sets_v1,
)
from tools.scheduler.programme_driver import ResourceStartLeaseLedgerV1


T0 = "2026-08-27T08:00:00Z"
T1 = "2026-08-27T08:01:00Z"
T2 = "2026-08-27T08:02:00Z"
T4 = "2026-08-27T08:04:00Z"
T5 = "2026-08-27T08:05:00Z"
DEVICE = "GPU-TEST-UUID"


def _digest(label: str) -> ContentDigest:
    return canonical_sha256({"label": label})


def _factory(label: str, *, issued_at_utc: str = T0):
    def build(
        resource_ids: tuple[str, ...],
        epoch: int,
        prior: ContentDigest | None,
    ) -> LaunchpadResourceStartLeaseV1:
        identity = canonical_sha256(
            {
                "label": label,
                "resource_ids": list(resource_ids),
                "epoch": epoch,
                "prior": None if prior is None else prior.to_dict(),
            }
        ).hex[:20]
        return LaunchpadResourceStartLeaseV1(
            lease_id=f"LPRSTART-{identity.upper()}",
            scheduler_owner_id="scheduler-gpu-lane-test",
            scheduler_owner_digest=_digest("scheduler"),
            execution_claim_digest=_digest(f"claim-{label}"),
            run_id=f"RUN-{label}",
            attempt_id=f"ATTEMPT-{label}",
            attempt_admission_request_digest=_digest(f"admission-{label}"),
            resource_ids=resource_ids,
            resource_scope_digest=_digest("shared-gpu-topology"),
            resource_observation_id=f"ROBS-{label}",
            resource_observation_digest=_digest(f"observation-{label}"),
            resource_recommendation_id=f"RADM-{label}",
            resource_recommendation_digest=_digest(f"recommendation-{label}"),
            live_provider_check_digest=_digest(f"provider-{label}"),
            lease_epoch=epoch,
            fencing_token_digest=_digest(f"fence-{label}-{epoch}"),
            prior_lease_digest=prior,
            issued_at_utc=issued_at_utc,
            start_before_utc=T4,
            expires_at_utc=T5,
        )

    return build


def _started(ledger: ResourceStartLeaseLedgerV1, lease) -> dict:
    event = ResourceStartLeaseEventV1(
        lease_digest=lease.lease_digest(),
        code=ResourceStartLeaseEventCodeV1.STARTED,
        scheduler_owner_id=lease.scheduler_owner_id,
        fencing_token_digest=lease.fencing_token_digest,
        event_at_utc=T1,
        reason="test owner started",
        prior_event_digest=None,
        successor_lease_digest=None,
    )
    ledger.append_event(lease.lease_id, event)
    return event.to_dict()


def _released(
    ledger: ResourceStartLeaseLedgerV1,
    lease: LaunchpadResourceStartLeaseV1,
    started: dict,
) -> None:
    ledger.append_event(
        lease.lease_id,
        ResourceStartLeaseEventV1(
            lease_digest=lease.lease_digest(),
            code=ResourceStartLeaseEventCodeV1.RELEASED,
            scheduler_owner_id=lease.scheduler_owner_id,
            fencing_token_digest=lease.fencing_token_digest,
            event_at_utc=T2,
            reason="test owner terminalized",
            prior_event_digest=canonical_sha256(started),
            successor_lease_digest=None,
        ),
    )


def test_gpu_parent_conflicts_with_every_lane_but_lanes_are_independent() -> None:
    lanes = shared_gpu_lane_candidate_sets_v1(DEVICE)
    parent = (f"gpu:{DEVICE}",)

    assert resource_ids_conflict_v1(parent, lanes[0]) is True
    assert resource_ids_conflict_v1(lanes[0], parent) is True
    assert resource_ids_conflict_v1(lanes[0], lanes[0]) is True
    assert resource_ids_conflict_v1(lanes[0], lanes[1]) is False


def test_three_concurrent_grants_are_distinct_and_fourth_waits(tmp_path: Path) -> None:
    ledger = ResourceStartLeaseLedgerV1(tmp_path / "scheduler")
    candidates = shared_gpu_lane_candidate_sets_v1(DEVICE)

    def allocate(label: str):
        try:
            lease, _path = ledger.grant_first_available_launchpad(
                candidate_resource_ids=candidates,
                at_utc=T0,
                lease_factory=_factory(label),
            )
            return lease
        except ActorError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(allocate, ("A", "B", "C", "D")))

    leases = [row for row in results if isinstance(row, LaunchpadResourceStartLeaseV1)]
    waits = [row for row in results if isinstance(row, ActorError)]
    assert len(leases) == 3
    assert len({lease.resource_ids for lease in leases}) == 3
    assert len(waits) == 1
    assert waits[0].code == "launchpad_attempt_resource_wait"


def test_started_lane_remains_occupied_until_release_and_fences_exclusive(
    tmp_path: Path,
) -> None:
    ledger = ResourceStartLeaseLedgerV1(tmp_path / "scheduler")
    candidates = shared_gpu_lane_candidate_sets_v1(DEVICE)
    leases = []
    started_events = []
    for label in ("A", "B", "C"):
        lease, _path = ledger.grant_first_available_launchpad(
            candidate_resource_ids=candidates,
            at_utc=T0,
            lease_factory=_factory(label),
        )
        leases.append(lease)
        started_events.append(_started(ledger, lease))

    assert {lease.lease_id for lease in ledger.active_leases(at_utc=T5)} == {
        lease.lease_id for lease in leases
    }
    with pytest.raises(ActorError) as shared_wait:
        ledger.grant_first_available_launchpad(
            candidate_resource_ids=candidates,
            at_utc=T2,
            lease_factory=_factory("D", issued_at_utc=T2),
        )
    assert shared_wait.value.code == "launchpad_attempt_resource_wait"

    parent_ids = (f"gpu:{DEVICE}",)
    epoch, prior = ledger.next_grant_fields(resource_ids=parent_ids, at_utc=T2)
    with pytest.raises(ProgrammeContractError, match="overlaps"):
        ledger.grant(_factory("EXCLUSIVE", issued_at_utc=T2)(parent_ids, epoch, prior))

    _released(ledger, leases[0], started_events[0])
    replacement, _path = ledger.grant_first_available_launchpad(
        candidate_resource_ids=candidates,
        at_utc=T2,
        lease_factory=_factory("D", issued_at_utc=T2),
    )
    assert replacement.resource_ids == leases[0].resource_ids


def test_active_exclusive_parent_makes_shared_lane_allocation_wait(
    tmp_path: Path,
) -> None:
    ledger = ResourceStartLeaseLedgerV1(tmp_path / "scheduler")
    parent_ids = (f"gpu:{DEVICE}",)
    exclusive = _factory("EXCLUSIVE")(parent_ids, 1, None)
    ledger.grant(exclusive)
    _started(ledger, exclusive)

    with pytest.raises(ActorError) as shared_wait:
        ledger.grant_first_available_launchpad(
            candidate_resource_ids=shared_gpu_lane_candidate_sets_v1(DEVICE),
            at_utc=T2,
            lease_factory=_factory("SHARED", issued_at_utc=T2),
        )

    assert shared_wait.value.code == "launchpad_attempt_resource_wait"
