from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.inheritance_generation_series_v1 import (
    InheritanceGenerationSeriesError,
    project_inheritance_generation_series_v1,
)


def _cell(cell_id: str, path: str) -> dict[str, Any]:
    return {
        "cell_id": cell_id,
        "dimensions": {
            "inheritance_mode": "linked",
            "founder_pattern": "packages",
            "reproduction_path": path,
            "paired_seed": 17,
        },
        "parameters": {
            "patch.initial_population": 4,
            "patch.capacity": 8,
            "inheritance_cross_path.mutation_enabled": False,
        },
    }


def _checkpoint(generation: int, cell_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "generation": generation,
        "cell_bundles": [
            {"cell_id": cell_id, "path": f"/runtime/{cell_id}/g{generation}"}
            for cell_id in cell_ids
        ],
        "terminal_cell_digests": {
            cell_id: canonical_sha256([cell_id, generation])
            for cell_id in cell_ids
        },
    }


def _artifact(
    *,
    segment: int,
    cell_id: str,
    attempts: list[dict[str, Any]],
    cumulative: tuple[int, int, int],
) -> dict[str, Any]:
    successes = sum(int(row["success"]) for row in attempts)
    cell = {
        "cell_index": 0,
        "enabled": True,
        "status": 0,
        "attempt_count": len(attempts),
        "success_count": successes,
        "failure_count": len(attempts) - successes,
        "attempts": attempts,
        "cumulative_counts_after_segment": {
            "attempts": cumulative[0],
            "successes": cumulative[1],
            "failures": cumulative[2],
        },
    }
    cell["segment_sha256"] = canonical_sha256(
        {
            "schema_version": "segment-v1",
            "segment_index": segment,
            "cell": cell,
        }
    )
    return {
        "schema_version": "segment-v1",
        "segment_index": segment,
        "cell_id": cell_id,
        "cell": cell,
    }


def _attempt(index: int, generation: int, success: bool) -> dict[str, Any]:
    return {
        "attempt_index": index,
        "generation": generation,
        "success": success,
    }


def _fixture() -> dict[str, Any]:
    cell_ids = ("refill", "ordinary")
    cells = (
        _cell(cell_ids[0], "generation_boundary_refill"),
        _cell(cell_ids[1], "ordinary_live_birth"),
    )
    generic = []
    artifacts = []
    for cell_index, cell_id in enumerate(cell_ids):
        generic.extend(
            [
                {
                    "cell_id": cell_id,
                    "generation": 0,
                    "alive": 4,
                    "births": 1,
                    "failed_births": 1,
                },
                {
                    "cell_id": cell_id,
                    "generation": 1,
                    "alive": 5,
                    "births": 1,
                    "failed_births": 0,
                },
                {
                    "cell_id": cell_id,
                    "generation": 2,
                    "alive": 6,
                    "births": 1,
                    "failed_births": 0,
                },
            ]
        )
        artifacts.extend(
            [
                _artifact(
                    segment=0,
                    cell_id=cell_id,
                    attempts=[
                        _attempt(0, 0, True),
                        _attempt(1, 0, False),
                    ],
                    cumulative=(2, 1, 1),
                ),
                _artifact(
                    segment=1,
                    cell_id=cell_id,
                    attempts=[
                        _attempt(0, 1, True),
                        _attempt(1, 2, True),
                    ],
                    cumulative=(4, 3, 1),
                ),
            ]
        )
        for artifact in artifacts[-2:]:
            artifact["cell"]["cell_index"] = cell_index
            unsigned = dict(artifact["cell"])
            unsigned.pop("segment_sha256")
            artifact["cell"]["segment_sha256"] = canonical_sha256(
                {
                    "schema_version": artifact["schema_version"],
                    "segment_index": artifact["segment_index"],
                    "cell": unsigned,
                }
            )
    return {
        "cells": cells,
        "generic_patch_rows": generic,
        "native_exposure_segment_artifacts": artifacts,
        "checkpoint_catalog": {
            "0": _checkpoint(0, cell_ids),
            "1": _checkpoint(1, cell_ids),
            "3": _checkpoint(3, cell_ids),
        },
        "terminal_generation": 3,
    }


def test_projects_split_segments_and_inclusive_terminal_boundary() -> None:
    rows = project_inheritance_generation_series_v1(**_fixture())

    assert [(row["cell_id"], row["generation"]) for row in rows] == [
        ("refill", 0),
        ("refill", 1),
        ("refill", 2),
        ("refill", 3),
        ("ordinary", 0),
        ("ordinary", 1),
        ("ordinary", 2),
        ("ordinary", 3),
    ]
    assert rows[0]["population"] == {"alive": 4, "capacity": 8}
    assert rows[3]["population"]["alive"] == 6
    assert rows[1]["reproduction"] == {
        "birth_opportunities": 2,
        "successful_births": 1,
        "failed_birth_attempts": 1,
        "refill_child_count": 1,
        "ordinary_birth_count": 0,
    }
    assert rows[5]["reproduction"]["refill_child_count"] == 0
    assert rows[5]["reproduction"]["ordinary_birth_count"] == 1
    assert rows[0]["reproduction"]["birth_opportunities"] == 0
    assert rows[3]["generation"] == 3
    assert rows[3]["identity"] == {
        "inheritance_mode": "linked",
        "founder_pattern": "packages",
        "reproduction_path": "generation_boundary_refill",
        "paired_seed": 17,
    }


def _move_attempt_outside_segment(payload: dict[str, Any]) -> None:
    artifact = payload["native_exposure_segment_artifacts"][1]
    artifact["cell"]["attempts"][0]["generation"] = 0
    unsigned = dict(artifact["cell"])
    unsigned.pop("segment_sha256")
    artifact["cell"]["segment_sha256"] = canonical_sha256(
        {
            "schema_version": artifact["schema_version"],
            "segment_index": artifact["segment_index"],
            "cell": unsigned,
        }
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["generic_patch_rows"].pop(),
            "exact transition prefix",
        ),
        (
            lambda payload: payload["generic_patch_rows"][0].__setitem__(
                "births", 2
            ),
            "reproduction counters differ",
        ),
        (
            _move_attempt_outside_segment,
            "outside its segment",
        ),
        (
            lambda payload: payload["checkpoint_catalog"]["3"].__setitem__(
                "generation", 2
            ),
            "contradicts its catalog key",
        ),
    ],
)
def test_fails_closed_on_gaps_and_contradictions(mutate: Any, message: str) -> None:
    payload = deepcopy(_fixture())
    mutate(payload)

    with pytest.raises(InheritanceGenerationSeriesError, match=message):
        project_inheritance_generation_series_v1(**payload)
