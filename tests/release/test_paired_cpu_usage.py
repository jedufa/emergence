from __future__ import annotations

import hashlib

import pytest

from engine.campaigns.canonical import canonical_sha256
from engine.campaigns.artifact_lifecycle import validate_artifact_disposition_manifest_v1
from engine.campaigns import paired_cpu_usage_v1 as paired


def _cell(cell_id: str, probability: float) -> dict:
    return {
        "cell_id": cell_id,
        "parameters": {
            "env_cfg.reproduce_try_prob": probability,
            "patch.capacity": 512,
            "patch.initial_population": 256,
        },
        "seeds": {"ecology": 101, "task": 1001},
        "dimensions": {
            "try_probability": probability,
            "ecology_seed": 101,
            "task_seed": 1001,
        },
    }


def _plan() -> dict:
    plan = {
        "schema_version": "eiel_campaign_plan_v1",
        "campaign_id": "paired-fixture",
        "execution": {
            "generations": 6,
            "ticks_per_generation": 160,
            "checkpoints": [3, 6],
            "tick_schedule": {"total_ticks": 960},
        },
        "cells": [_cell("control", 0.0), _cell("treatment", 0.15)],
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def test_paired_plan_accepts_protocol_and_rejects_control_seed_horizon_or_cell_drift():
    plan = _plan()
    left, right = paired._check_pair_plan(plan)
    assert left["cell_id"] == "control"
    assert right["cell_id"] == "treatment"

    mutations = []
    seed = _plan()
    seed["cells"][1]["seeds"]["task"] = 1002
    mutations.append(seed)
    control = _plan()
    control["cells"][1]["parameters"]["env_cfg.reproduce_try_prob"] = 0.2
    mutations.append(control)
    horizon = _plan()
    horizon["execution"]["tick_schedule"]["total_ticks"] = 800
    mutations.append(horizon)
    missing_cell = _plan()
    missing_cell["cells"].pop()
    mutations.append(missing_cell)
    for changed in mutations:
        changed["plan_sha256"] = canonical_sha256(
            {key: value for key, value in changed.items() if key != "plan_sha256"}
        )
        with pytest.raises(paired.PairedUsageError):
            paired._check_pair_plan(changed)


def test_pair_metrics_preserve_missing_and_null_and_allow_negative_deltas():
    summary = paired._paired_metric_summary(
        [
            {"cell_id": "control", "population.alive": 4, "population.births": 3, "population.deaths": None},
            {"cell_id": "treatment", "population.alive": 2, "population.deaths": 0},
        ],
        "control",
        "treatment",
    )
    assert summary["population.alive"]["treatment_minus_control"] == -2
    assert summary["population.births"] == {
        "control": {"present": True, "value": 3},
        "treatment": {"present": False, "value": None},
        "treatment_minus_control": None,
    }
    assert summary["population.deaths"]["control"] == {"present": True, "value": None}
    assert summary["population.deaths"]["treatment"] == {"present": True, "value": 0}
    assert summary["population.deaths"]["treatment_minus_control"] is None


@pytest.mark.parametrize(
    ("actual", "expected", "label"),
    [
        ("request-tampered", "request-trusted", "request"),
        ("result-tampered", "result-trusted", "result"),
        ("disposition-tampered", "disposition-trusted", "disposition"),
    ],
)
def test_each_external_anchor_rejects_tampering(actual, expected, label):
    with pytest.raises(paired.PairedUsageError, match=label):
        paired._check_trusted_anchor(actual, expected, label)


def test_result_seal_detects_content_mutation():
    result = {"schema_version": "eiel_execution_result_v2", "status": "completed"}
    result["result_sha256"] = canonical_sha256(result)
    assert paired._sealed_hash(result, "result_sha256", "result") == result["result_sha256"]
    result["status"] = "failed"
    with pytest.raises(paired.PairedUsageError, match="seal mismatch"):
        paired._sealed_hash(result, "result_sha256", "result")


def test_inventory_hash_detects_changed_selected_observation(tmp_path):
    path = tmp_path / "batch.json"
    path.write_bytes(b'{"births": 2}\n')
    manifest = {
        "artifacts": [
            {
                "path": "batch.json",
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ]
    }
    assert paired._verify_inventory_file(tmp_path, manifest, "batch.json")["bytes"] == path.stat().st_size
    path.write_bytes(b'{"births": 0}\n')
    with pytest.raises(paired.PairedUsageError, match="SHA-256"):
        paired._verify_inventory_file(tmp_path, manifest, "batch.json")


def test_rehashed_observation_and_manifest_still_fail_fixed_external_disposition_anchor(tmp_path):
    path = tmp_path / "batch.json"
    path.write_bytes(b'{"births": 2}\n')

    def sealed_manifest() -> dict:
        manifest = {
            "schema_version": "eiel_artifact_disposition_manifest_v1",
            "active_release_boundary": False,
            "artifacts": [
                {
                    "path": "batch.json",
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "retention_class": "R5",
                    "exact_object_identity": True,
                    "release_eligible": False,
                }
            ],
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        return validate_artifact_disposition_manifest_v1(manifest)

    initial = sealed_manifest()
    externally_trusted_digest = initial["manifest_sha256"]
    assert paired._verify_inventory_file(tmp_path, initial, "batch.json")["sha256"] == initial["artifacts"][0]["sha256"]

    # An attacker can update both the observation and the self-sealed inventory;
    # only the independent, fixed digest detects the replacement.
    path.write_bytes(b'{"births": 999}\n')
    replacement = sealed_manifest()
    assert replacement["manifest_sha256"] != externally_trusted_digest
    assert paired._verify_inventory_file(tmp_path, replacement, "batch.json")["sha256"] == replacement["artifacts"][0]["sha256"]
    with pytest.raises(paired.PairedUsageError, match="trusted anchor"):
        paired._check_trusted_anchor(
            replacement["manifest_sha256"], externally_trusted_digest, "disposition"
        )


def test_continuation_branch_identity_and_evidence_fail_closed():
    branch = {"branch_id": "resume-a", "kind": "checkpoint_continuation"}
    manifest = {
        "branch_id": "resume-a",
        "kind": "checkpoint_continuation",
        "campaign_id": "paired-fixture",
        "plan_sha256": "a" * 64,
        "status": "completed",
    }
    assert paired._check_branch_identity(
        branch, manifest, campaign_id="paired-fixture", plan_sha256="a" * 64
    ) == "resume-a"
    with pytest.raises(paired.PairedUsageError, match="ID mismatch"):
        paired._check_branch_identity(
            branch,
            {**manifest, "branch_id": "resume-other"},
            campaign_id="paired-fixture",
            plan_sha256="a" * 64,
        )

    receipt = {
        "match": True,
        "mismatches": [],
        "missing_surfaces": [],
        "compared_surfaces": ["metrics_rows_sha256", "terminal_checkpoint_digest"],
        "baseline": {"metrics_rows_sha256": "1", "terminal_checkpoint_digest": "2"},
        "candidate": {"metrics_rows_sha256": "1", "terminal_checkpoint_digest": "2"},
    }
    assert paired._check_comparison_match(
        receipt,
        surfaces={"metrics_rows_sha256", "terminal_checkpoint_digest"},
        label="continuation",
    ) is receipt
    with pytest.raises(paired.PairedUsageError, match="did not match"):
        paired._check_comparison_match(
            {**receipt, "match": False},
            surfaces={"metrics_rows_sha256", "terminal_checkpoint_digest"},
            label="continuation",
        )
    with pytest.raises(paired.PairedUsageError, match="observations differ"):
        paired._check_comparison_match(
            {**receipt, "candidate": {"terminal_checkpoint_digest": "different"}},
            surfaces={"metrics_rows_sha256", "terminal_checkpoint_digest"},
            label="continuation",
        )
