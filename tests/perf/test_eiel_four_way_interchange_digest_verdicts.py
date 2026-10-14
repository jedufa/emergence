from __future__ import annotations

from copy import deepcopy

import torch

from engine.native_runtime.four_way_interchange import (
    _overall_comparison_verdict,
    _projection_parity_verdict,
)


def _projection(*, schema_version: str = "checkpoint_v1", authority_epoch: str = "epoch:0") -> dict:
    return {
        "schema_version": schema_version,
        "metadata": {
            "world": "flat",
            "generation": 0,
            "tick": 0,
            "authority_epoch": authority_epoch,
        },
        "tensors": {
            "alive_mask": torch.tensor([True, False], dtype=torch.bool),
        },
    }


def _artifact() -> dict:
    return {
        "source": "ignored-route-label",
        "semantic_state": {"alive_count": 1},
    }


def _overall(name: str, comparison: dict) -> dict:
    return _overall_comparison_verdict({name: comparison})


def test_digest_only_mismatch_fails_direction_and_overall_comparison() -> None:
    left = _projection(authority_epoch="epoch:left")
    right = _projection(authority_epoch="epoch:right")

    direction = _projection_parity_verdict(left, right, _artifact(), _artifact())

    assert direction["projection_digest_ok"] is False
    assert direction["canonical_checkpoint_storage_ok"] is True
    assert direction["artifact_ok"] is True
    assert direction["ok"] is False
    assert direction["status"] == "failed_closed"
    assert direction["surfaces"]["authoritative_projection_digest"]["first_mismatch"] == {
        "field": "authority_epoch",
        "path": "$.metadata.authority_epoch",
        "left": "epoch:left",
        "right": "epoch:right",
    }
    comparison_name = "native_to_reference_reimported_native_matches_native_uninterrupted"
    overall = _overall(comparison_name, direction)
    assert overall["ok"] is False
    assert overall["first_failed"]["path"] == comparison_name
    assert overall["first_failed"]["details"]["field"] == "authority_epoch"


def test_canonical_storage_only_mismatch_fails_direction_and_overall_comparison() -> None:
    left = _projection(schema_version="checkpoint_left")
    right = _projection(schema_version="checkpoint_right")

    direction = _projection_parity_verdict(left, right, _artifact(), _artifact())

    assert direction["projection_digest_ok"] is True
    assert direction["canonical_checkpoint_storage_ok"] is False
    assert direction["artifact_ok"] is True
    assert direction["ok"] is False
    assert direction["status"] == "failed_closed"
    assert direction["surfaces"]["canonical_checkpoint_storage"]["first_mismatch"] == {
        "field": "schema_version",
        "path": "$.metadata.schema_version",
        "left": "checkpoint_left",
        "right": "checkpoint_right",
    }
    comparison_name = "native_to_reference_reimport_matches_boundary"
    overall = _overall(comparison_name, direction)
    assert overall["ok"] is False
    assert overall["first_failed"]["path"] == comparison_name
    assert overall["first_failed"]["details"]["field"] == "schema_version"


def test_matching_digest_storage_and_artifact_pass_direction_and_overall() -> None:
    left = _projection()
    right = deepcopy(left)

    direction = _projection_parity_verdict(left, right, _artifact(), _artifact())

    assert direction["ok"] is True
    assert direction["status"] == "passed"
    assert direction["first_mismatch"] is None
    assert direction["first_failed"] is None
    assert set(direction["surfaces"]["authoritative_projection_digest"]) == set(
        direction["surfaces"]["canonical_checkpoint_storage"]
    ) == {"ok", "left_digest", "right_digest", "first_mismatch"}
    assert {
        name: surface["ok"]
        for name, surface in direction["surfaces"].items()
    } == {
        "authoritative_projection_digest": True,
        "canonical_checkpoint_storage": True,
        "comparison_artifact": True,
    }
    assert _overall("native_to_reference_reimport_matches_boundary", direction) == {
        "ok": True,
        "first_failed": None,
    }
