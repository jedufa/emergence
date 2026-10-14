from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.campaigns.orchestration import (
    CampaignExecutionError,
    _compare_branch,
    _compare_reporting_parity,
    _continuation_transition_sentinels,
    _persist_comparison_result,
)
from engine.campaigns.route_sentinels import RouteTransitionSentinelV1


CELL_ID = "cell-0"
TERMINAL_DIGEST = "terminal-exact"


def _sentinel(
    *,
    transition_name: str = "post_boundary_cache_refill",
    absolute_generation: int = 2,
    absolute_tick: int = 1,
    action_digest: str = "actions-exact",
) -> dict[str, object]:
    return RouteTransitionSentinelV1(
        transition_name=transition_name,
        absolute_generation=absolute_generation,
        absolute_tick=absolute_tick,
        physical_window_position=0,
        rng_cursor=48,
        tape_cursor=96,
        cache_epoch=absolute_generation,
        refill_calls=1,
        refill_row_digest="refill-exact",
        action_digest=action_digest,
        alive_identity_digest="alive-exact",
        runtime_event_cursor=14,
        world_digest="world-exact",
        policy_digest="policy-exact",
    ).payload()


def _manifest(
    *,
    sentinels: list[dict[str, object]] | None,
    terminal_digest: str = TERMINAL_DIGEST,
) -> dict[str, object]:
    result: dict[str, object] = {}
    if sentinels is not None:
        result["transition_sentinels"] = {CELL_ID: sentinels}
    return {
        "checkpoint": {
            "terminal_cell_digests": {CELL_ID: terminal_digest},
        },
        "metrics_rows": [
            {"cell_id": CELL_ID, "metrics": {"population.alive": 1}},
        ],
        "result": result,
    }


def test_exact_transition_sentinels_join_the_comparison_surface() -> None:
    sentinels = [
        _sentinel(
            transition_name="generation_boundary",
            absolute_tick=0,
        ),
        _sentinel(),
    ]

    receipt = _compare_branch(
        baseline=_manifest(sentinels=sentinels),
        candidate=_manifest(sentinels=sentinels),
        cell_id=CELL_ID,
    )

    assert receipt["match"] is True
    assert receipt["first_difference"] is None
    assert "transition_sentinels_sha256" in receipt["compared_surfaces"]
    assert (
        receipt["baseline"]["transition_sentinels_sha256"]
        == receipt["candidate"]["transition_sentinels_sha256"]
    )


def test_continuation_surface_deduplicates_replayed_history() -> None:
    sentinel = _sentinel(transition_name="cold_start", absolute_generation=0)
    rows = _continuation_transition_sentinels(
        _manifest(sentinels=[sentinel, sentinel]),
        CELL_ID,
    )

    assert rows is not None
    assert len(rows) == 1
    assert rows[0]["transition_name"] == "cold_start"


def test_reporting_parity_localizes_transient_mismatch_with_equal_terminal_digest() -> None:
    baseline = [_sentinel(action_digest="actions-enabled")]
    candidate = [_sentinel(action_digest="actions-disabled")]

    receipt = _compare_reporting_parity(
        baseline=_manifest(sentinels=baseline),
        candidate=_manifest(sentinels=candidate),
        cell_id=CELL_ID,
    )

    assert receipt["reporting_enabled_terminal_digest"] == TERMINAL_DIGEST
    assert receipt["reporting_disabled_terminal_digest"] == TERMINAL_DIGEST
    assert receipt["match"] is False
    assert receipt["mismatches"] == ["transition_sentinels_sha256"]
    assert receipt["missing_surfaces"] == []
    assert receipt["first_difference"]["status"] == (
        "localized_transition_sentinel"
    )
    assert receipt["first_difference"]["path"] == "sentinels[0].action_digest"
    assert receipt["first_difference"]["absolute_generation"] == 2
    assert receipt["first_difference"]["absolute_tick"] == 1


def test_one_sided_and_malformed_transition_sentinel_surfaces_fail_closed() -> None:
    one_sided = _compare_reporting_parity(
        baseline=_manifest(sentinels=[_sentinel()]),
        candidate=_manifest(sentinels=None),
        cell_id=CELL_ID,
    )

    assert one_sided["match"] is False
    assert one_sided["mismatches"] == []
    assert one_sided["missing_surfaces"] == ["transition_sentinels"]

    malformed = _manifest(sentinels=None)
    malformed["result"]["transition_sentinels"] = {
        CELL_ID: [{"schema_version": "not-v1"}]
    }
    with pytest.raises(CampaignExecutionError, match="transition sentinel is invalid"):
        _compare_branch(
            baseline=malformed,
            candidate=_manifest(sentinels=[_sentinel()]),
            cell_id=CELL_ID,
        )


def test_continuation_persistence_merges_split_and_resumed_sentinels(
    tmp_path: Path,
) -> None:
    boundary = _sentinel(
        transition_name="generation_boundary",
        absolute_generation=1,
        absolute_tick=0,
    )
    post_restore = _sentinel(
        transition_name="first_post_restore_refill",
        absolute_generation=1,
        absolute_tick=1,
    )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "status": "running",
                "checkpoint": {"generation": 1},
                "metrics_rows": [],
                "result": {
                    "transition_sentinels": {
                        CELL_ID: [boundary],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    persisted = _persist_comparison_result(
        path,
        {
            "status": "completed",
            "checkpoint": {"generation": 2},
            "metrics_rows": [],
            "result": {
                "transition_sentinels": {
                    CELL_ID: [boundary, post_restore],
                }
            },
        },
        phase="terminal_completed",
    )

    assert persisted["result"]["transition_sentinels"][CELL_ID] == [
        boundary,
        post_restore,
    ]
    assert json.loads(path.read_text(encoding="utf-8")) == persisted
