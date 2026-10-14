from __future__ import annotations

import copy
import hashlib
import json

import pytest

from perf.cuda_architecture_b_checkpoint_branch import (
    BRANCH_ID_DOMAIN,
    BRANCH_ID_PREFIX,
    CudaCheckpointBranchCollisionError,
    CudaCheckpointBranchError,
    canonical_branch_json_bytes,
    checkpoint_branch_trajectory_projection,
    checkpoint_branch_trajectory_sha256,
    create_checkpoint_branch,
    create_checkpoint_branch_batch,
    validate_branch_key,
    validate_checkpoint_branch,
    validate_checkpoint_branch_batch,
    validate_checkpoint_branch_collision,
)


CHECKPOINT_SHA256 = "1" * 64
RNG_SHA256 = "2" * 64


def _branch(
    branch_key: str = "control",
    *,
    projection: dict | None = None,
    source_branch_id: str | None = "parent-7",
):
    return create_checkpoint_branch(
        branch_key=branch_key,
        source_checkpoint_projection_sha256=CHECKPOINT_SHA256,
        source_generation=17,
        trajectory_projection=projection
        or {
            "ecology": {"hazard_rate": 0.125, "world": "patch"},
            "policy": {"temperature": 1.0},
        },
        source_rng_state_sha256=RNG_SHA256,
        source_branch_id=source_branch_id,
    )


def test_canonical_json_is_sorted_compact_utf8_and_rejects_nonfinite() -> None:
    assert canonical_branch_json_bytes(
        {"z": [3, 2], "a": {"unicode": "lambda", "ok": True}}
    ) == b'{"a":{"ok":true,"unicode":"lambda"},"z":[3,2]}'

    with pytest.raises(CudaCheckpointBranchError, match="valid canonical branch JSON"):
        canonical_branch_json_bytes({"not_finite": float("nan")})
    with pytest.raises(CudaCheckpointBranchError, match="non-string"):
        canonical_branch_json_bytes({1: "ambiguous"})


@pytest.mark.parametrize(
    "branch_key",
    ("", " Control", "UPPER", "contains space", ".leading", "x" * 129),
)
def test_branch_key_validation_fails_closed_without_normalizing(branch_key: str) -> None:
    with pytest.raises(CudaCheckpointBranchError, match="branch_key"):
        validate_branch_key(branch_key)


def test_branch_identity_is_canonical_domain_separated_and_stable() -> None:
    first = _branch(
        projection={
            "policy": {"temperature": 1.0},
            "ecology": {"world": "patch", "hazard_rate": 0.125},
        }
    )
    second = _branch(
        projection={
            "ecology": {"hazard_rate": 0.125, "world": "patch"},
            "policy": {"temperature": 1.0},
        }
    )
    expected = hashlib.sha256(
        BRANCH_ID_DOMAIN + canonical_branch_json_bytes(first.branch_identity)
    ).hexdigest()

    assert first == second
    assert first.branch_identity_sha256 == expected
    assert first.branch_id == BRANCH_ID_PREFIX + expected
    assert validate_checkpoint_branch(first.to_record()) == first


def test_branch_identity_changes_for_each_trajectory_identity_axis() -> None:
    baseline = _branch()
    changed_key = _branch("treatment")
    changed_projection = _branch(
        projection={
            "ecology": {"hazard_rate": 0.25, "world": "patch"},
            "policy": {"temperature": 1.0},
        }
    )
    changed_checkpoint = create_checkpoint_branch(
        branch_key="control",
        source_checkpoint_projection_sha256="3" * 64,
        source_generation=17,
        trajectory_projection=baseline.trajectory_projection,
        source_rng_state_sha256=RNG_SHA256,
        source_branch_id="parent-7",
    )
    changed_parent = _branch(source_branch_id="parent-8")
    changed_rng = create_checkpoint_branch(
        branch_key="control",
        source_checkpoint_projection_sha256=CHECKPOINT_SHA256,
        source_generation=17,
        trajectory_projection=baseline.trajectory_projection,
        source_rng_state_sha256="4" * 64,
        source_branch_id="parent-7",
    )

    assert len(
        {
            baseline.branch_id,
            changed_key.branch_id,
            changed_projection.branch_id,
            changed_checkpoint.branch_id,
            changed_parent.branch_id,
            changed_rng.branch_id,
        }
    ) == 6


def test_rng_lineage_proves_exact_inheritance_without_seed_derivation() -> None:
    branch = _branch()
    provenance = branch.rng_lineage_provenance

    assert provenance["inheritance_mode"] == "exact_checkpoint_state"
    assert provenance["source_rng_state_sha256"] == RNG_SHA256
    assert provenance["inherited_rng_state_sha256"] == RNG_SHA256
    assert provenance["reseeded"] is False
    assert provenance["seed_derivation"] is None

    with pytest.raises(CudaCheckpointBranchError, match="without reseeding"):
        create_checkpoint_branch(
            branch_key="control",
            source_checkpoint_projection_sha256=CHECKPOINT_SHA256,
            source_generation=17,
            trajectory_projection=branch.trajectory_projection,
            source_rng_state_sha256=RNG_SHA256,
            inherited_rng_state_sha256="3" * 64,
        )
    with pytest.raises(CudaCheckpointBranchError, match="64 lowercase"):
        create_checkpoint_branch(
            branch_key="control",
            source_checkpoint_projection_sha256=CHECKPOINT_SHA256,
            source_generation=17,
            trajectory_projection=branch.trajectory_projection,
            source_rng_state_sha256=RNG_SHA256,
            inherited_rng_state_sha256="",
        )


def test_provenance_is_excluded_from_trajectory_projection() -> None:
    projection = {
        "ecology": {"world": "patch"},
        "branch_id": "caller-owned-id",
        "branch_key": "caller-owned-key",
        "provenance": {"host": "not-trajectory"},
        "rng_lineage_provenance": {"seed": 17},
    }
    branch = _branch(projection=projection)
    expected = {"ecology": {"world": "patch"}}

    assert branch.trajectory_projection == expected
    assert checkpoint_branch_trajectory_projection(branch.to_record()) == expected
    first_digest = checkpoint_branch_trajectory_sha256(branch)

    changed_record = branch.to_record()
    changed_record["rng_lineage_provenance"]["source_branch_id"] = "other-parent"
    assert checkpoint_branch_trajectory_sha256(changed_record) == first_digest


def test_validation_rejects_tampered_identity_and_rng_provenance() -> None:
    branch = _branch()
    tampered_identity = branch.to_record()
    tampered_identity["branch_identity"]["source_checkpoint"]["generation"] = 18
    with pytest.raises(CudaCheckpointBranchError, match="branch_identity_sha256"):
        validate_checkpoint_branch(tampered_identity)

    tampered_rng = branch.to_record()
    tampered_rng["rng_lineage_provenance"]["reseeded"] = True
    with pytest.raises(CudaCheckpointBranchError, match="without reseeding"):
        validate_checkpoint_branch(tampered_rng)

    tampered_parent = branch.to_record()
    tampered_parent["rng_lineage_provenance"]["source_branch_id"] = "other-parent"
    with pytest.raises(CudaCheckpointBranchError, match="parent branch"):
        validate_checkpoint_branch(tampered_parent)


def test_collision_validation_accepts_only_byte_idempotent_same_id() -> None:
    branch = _branch()
    assert validate_checkpoint_branch_collision(branch, branch.to_record()) == (
        "idempotent"
    )
    assert validate_checkpoint_branch_collision(branch, _branch("treatment")) == (
        "distinct"
    )

    distinct_parent = _branch(source_branch_id="equivalent-parent")
    assert distinct_parent.branch_id != branch.branch_id
    assert validate_checkpoint_branch_collision(branch, distinct_parent) == "distinct"


def test_batch_rejects_duplicate_keys_even_when_records_are_idempotent() -> None:
    control = _branch("control")
    treatment = _branch("treatment")
    assert validate_checkpoint_branch_batch((control, treatment)) == (
        control,
        treatment,
    )

    with pytest.raises(CudaCheckpointBranchError, match="duplicate branch_key"):
        validate_checkpoint_branch_batch((control, copy.deepcopy(control)))


def test_batch_builder_rejects_duplicate_specifications() -> None:
    specification = {
        "branch_key": "control",
        "source_checkpoint_projection_sha256": CHECKPOINT_SHA256,
        "source_generation": 17,
        "trajectory_projection": {"ecology": {"world": "patch"}},
        "source_rng_state_sha256": RNG_SHA256,
    }
    with pytest.raises(CudaCheckpointBranchError, match="duplicate branch_key"):
        create_checkpoint_branch_batch((specification, dict(specification)))


def test_record_is_canonical_json_serializable() -> None:
    record = _branch().to_record()
    encoded = canonical_branch_json_bytes(record)

    assert json.loads(encoded) == record
    assert b" " not in encoded
