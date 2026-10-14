from __future__ import annotations

from dataclasses import replace
import json
from math import prod
from typing import Any

import numpy as np
import pytest

from engine.perf.cuda_architecture_b_boundary_controller_v1 import (
    BOUNDARY_PLAN_SCHEMA_VERSION_V1,
    RANK_TIEBREAK_REJECT_V1,
    RANK_TIEBREAK_STABLE_CELL_ID_ASCENDING_V1,
    SEEDED_STABLE_CELL_ID_SHA256_ASSIGNMENT_V1,
    BoundaryControllerContractError,
    BoundaryPruningForkPolicyV1,
    BoundarySourceLineageV1,
    plan_cuda_architecture_b_boundary_controller_v1,
)
from engine.perf.cuda_architecture_b_cell_decision_summary_v1 import (
    freeze_cell_decision_summary_v1,
)
from engine.perf.cuda_architecture_b_checkpoint_clone_v1 import (
    CheckpointCloneFieldDescriptorV1,
    build_accepted_checkpoint_clone_state_manifest_v1,
)


SCORE_POLICY_ID = "test/caller-lexicographic-basis/v1"
SCORE_POLICY_DIGEST = bytes(range(32))
PRUNING_POLICY_DIGEST = "31" * 32
CHECKPOINT_DIGEST = "42" * 32
SHAPE_BINDINGS = {
    "capacity": 8,
    "cell": 4,
    "chemistry_channels": 2,
    "drive_channels": 4,
    "event_capacity": 32,
    "gru_hidden": 4,
    "gru_pack_width": 16,
    "height": 4,
    "mlp_pack_width": 16,
    "mutation_rng_state_bytes": 32,
    "policy_tick_stride": 1,
    "width": 4,
}


def _summary(
    *,
    stable_cell_ids: tuple[int, ...] = (40, 10, 30, 20),
    score_rows: tuple[tuple[float, ...], ...] = (
        (9.0, 1.0),
        (8.0, 2.0),
        (7.0, 3.0),
        (6.0, 4.0),
    ),
):
    return freeze_cell_decision_summary_v1(
        validity=np.ones((1, len(stable_cell_ids)), dtype=np.uint8),
        status=np.zeros((1, len(stable_cell_ids)), dtype=np.int32),
        score_basis=np.array([score_rows], dtype=np.float64),
        stable_cell_ids=np.array(stable_cell_ids, dtype=np.int64),
        score_policy_id=SCORE_POLICY_ID,
        score_policy_digest=SCORE_POLICY_DIGEST,
    )


def _descriptors() -> tuple[CheckpointCloneFieldDescriptorV1, ...]:
    rows: list[CheckpointCloneFieldDescriptorV1] = []
    pointer = 1_000_000
    for field in _state_manifest()["fields"]:
        shape = tuple(field["shape"])
        element_size_bytes = int(field["element_size_bytes"])
        bytes_per_cell = prod(shape[1:]) * element_size_bytes
        rows.append(
            CheckpointCloneFieldDescriptorV1(
                field_name=field["field_name"],
                pointer_identity=pointer,
                shape=shape,
                cell_stride=bytes_per_cell,
                bytes_per_cell=bytes_per_cell,
                element_size_bytes=element_size_bytes,
            )
        )
        pointer += bytes_per_cell * shape[0] + 4096
    return tuple(rows)


def _lineage(stable_cell_id: int) -> BoundarySourceLineageV1:
    return BoundarySourceLineageV1(
        stable_cell_id=stable_cell_id,
        source_rng_state_sha256=f"{stable_cell_id:064x}",
        source_branch_id=f"parent-{stable_cell_id}",
    )


def _state_manifest() -> dict[str, object]:
    return build_accepted_checkpoint_clone_state_manifest_v1(SHAPE_BINDINGS)


def _policy(**overrides: Any) -> BoundaryPruningForkPolicyV1:
    values: dict[str, Any] = {
        "policy_id": "test.prune-bottom-two.seeded-fork.v1",
        "policy_digest": PRUNING_POLICY_DIGEST,
        "generation_index": 0,
        "source_generation": 64,
        "generation_boundary_complete": True,
        "prune_count": 2,
        "planner_seed": 7719,
        "rank_tiebreak": RANK_TIEBREAK_REJECT_V1,
        "fork_source_assignment": SEEDED_STABLE_CELL_ID_SHA256_ASSIGNMENT_V1,
        "branch_key_prefix": "boundary-test",
        "source_checkpoint_projection_sha256": CHECKPOINT_DIGEST,
        "state_manifest": _state_manifest(),
        "source_lineage": (_lineage(40), _lineage(10)),
    }
    values.update(overrides)
    return BoundaryPruningForkPolicyV1(**values)


def _plan(**policy_overrides: Any):
    return plan_cuda_architecture_b_boundary_controller_v1(
        _summary(), _descriptors(), _policy(**policy_overrides)
    )


def _stable_assignments(plan: Any) -> list[tuple[int, int]]:
    return [
        (entry.source_stable_cell_id, entry.target_stable_cell_id)
        for entry in plan.source_to_target_clone_entries
    ]


def test_plan_is_input_order_independent_for_descriptors_and_lineage() -> None:
    first = _plan()
    second = plan_cuda_architecture_b_boundary_controller_v1(
        _summary(),
        tuple(reversed(_descriptors())),
        _policy(source_lineage=tuple(reversed(_policy().source_lineage))),
    )

    assert first.to_dict() == second.to_dict()
    assert first.plan_sha256 == second.plan_sha256
    assert json.loads(json.dumps(first.to_dict())) == first.to_dict()


def test_stable_id_assignment_is_independent_of_summary_cell_axis_order() -> None:
    original = _summary()
    permutation = np.array([3, 2, 0, 1])
    reordered = freeze_cell_decision_summary_v1(
        validity=np.ascontiguousarray(original.validity[:, permutation]),
        status=np.ascontiguousarray(original.status[:, permutation]),
        score_basis=np.ascontiguousarray(original.score_basis[:, permutation, :]),
        stable_cell_ids=np.ascontiguousarray(original.stable_cell_ids[permutation]),
        score_policy_id=SCORE_POLICY_ID,
        score_policy_digest=SCORE_POLICY_DIGEST,
    )

    first = _plan()
    second = plan_cuda_architecture_b_boundary_controller_v1(
        reordered, _descriptors(), _policy()
    )

    assert _stable_assignments(first) == _stable_assignments(second)
    assert [cell.stable_cell_id for cell in first.kept_source_cells] == [40, 10]
    assert [cell.stable_cell_id for cell in second.kept_source_cells] == [40, 10]


def test_same_seed_repeat_is_byte_stable_and_composes_branch_rng_lineage() -> None:
    first = _plan()
    second = _plan()

    assert first.to_dict() == second.to_dict()
    assert first.plan_sha256 == second.plan_sha256
    assert len(first.branch_records) == 2
    for entry, branch in zip(
        first.source_to_target_clone_entries, first.branch_records, strict=True
    ):
        lineage = branch.rng_lineage_provenance
        assert entry.branch_id == branch.branch_id
        assert lineage["source_rng_state_sha256"] == (
            lineage["inherited_rng_state_sha256"]
        )
        assert lineage["reseeded"] is False
        assert lineage["seed_derivation"] is None


def test_rank_ties_fail_closed_unless_stable_cell_id_tiebreak_is_explicit() -> None:
    tied = _summary(score_rows=((5.0,), (5.0,), (5.0,), (5.0,)))
    stable_policy = _policy(
        rank_tiebreak=RANK_TIEBREAK_STABLE_CELL_ID_ASCENDING_V1,
        source_lineage=(_lineage(10), _lineage(20)),
    )

    with pytest.raises(BoundaryControllerContractError, match="ambiguous.*ties"):
        plan_cuda_architecture_b_boundary_controller_v1(
            tied, _descriptors(), replace(stable_policy, rank_tiebreak="reject")
        )

    plan = plan_cuda_architecture_b_boundary_controller_v1(
        tied, _descriptors(), stable_policy
    )
    assert [cell.stable_cell_id for cell in plan.kept_source_cells] == [10, 20]
    assert [cell.stable_cell_id for cell in plan.pruned_source_cells] == [30, 40]


def test_collisions_are_rejected_before_a_plan_is_emitted() -> None:
    duplicate_lineage = _policy(
        source_lineage=(_lineage(40), _lineage(40))
    )
    with pytest.raises(BoundaryControllerContractError, match="collision"):
        plan_cuda_architecture_b_boundary_controller_v1(
            _summary(), _descriptors(), duplicate_lineage
        )

    overlapping = list(_descriptors())
    overlapping[1] = replace(
        overlapping[1],
        pointer_identity=overlapping[0].pointer_identity + 1,
    )
    with pytest.raises(BoundaryControllerContractError, match="overlap"):
        plan_cuda_architecture_b_boundary_controller_v1(
            _summary(), overlapping, _policy()
        )


def test_plan_is_metadata_only_source_preserving_and_zero_d2h() -> None:
    plan = _plan()
    payload = plan.to_dict()

    assert payload["schema_version"] == BOUNDARY_PLAN_SCHEMA_VERSION_V1
    assert payload["boundary_contract"] == {
        "contains_full_simulation_state": False,
        "full_state_payload_accepted": False,
        "device_to_host_operations": 0,
        "device_to_host_bytes": 0,
        "hidden_host_materialization": False,
        "execution_performed": False,
        "source_cells_preserved_by_plan": True,
        "targets_are_distinct_slots": True,
        "live_fork_execution_qualified": False,
        "full_semantic_checkpoint_clone_qualified": False,
    }
    transfer = payload["clone_plan"]["transfer_contract"]
    assert transfer["copy_kind"] == "device_to_device"
    assert transfer["accepted_tensor_state_only"] is True
    assert transfer["full_semantic_checkpoint_clone_qualified"] is False
    assert transfer["device_to_host_operations"] == 0
    assert transfer["device_to_host_bytes"] == 0
    assert transfer["execution_performed"] is False
    assert transfer["zero_d2h_execution_qualified"] is False

    source_slots = {
        entry.source_cell_index
        for entry in plan.source_to_target_clone_entries
    }
    target_slots = {
        entry.target_cell_index
        for entry in plan.source_to_target_clone_entries
    }
    assert source_slots.isdisjoint(target_slots)
    assert len(target_slots) == len(plan.target_cells) == 2
    for source_plan in plan.clone_plan.source_plans:
        clone_payload = source_plan.to_dict()
        assert clone_payload["source_preservation_contract"][
            "source_write_entry_count"
        ] == 0
        assert clone_payload["transfer_contract"]["device_to_host_operations"] == 0


def test_planner_does_not_define_scientific_scoring_semantics() -> None:
    payload = _plan().to_dict()

    assert payload["score_contract"] == {
        "score_policy_id": SCORE_POLICY_ID,
        "score_policy_digest": SCORE_POLICY_DIGEST.hex(),
        "caller_basis_used_verbatim": True,
        "scientific_scoring_semantics_selected": False,
        "rank_tiebreak": RANK_TIEBREAK_REJECT_V1,
    }
    assert "score" not in payload["kept_source_cells"][0]
    assert "score_basis" not in payload["kept_source_cells"][0]


@pytest.mark.parametrize(
    "policy",
    [
        _policy(generation_boundary_complete=False),
        _policy(generation_index=1),
        _policy(prune_count=0),
        _policy(prune_count=4),
        _policy(planner_seed=-1),
        _policy(rank_tiebreak="cell_index"),
        _policy(fork_source_assignment="random"),
        _policy(policy_digest="A" * 64),
        _policy(source_lineage=(_lineage(40),)),
    ],
)
def test_malformed_or_ambiguous_policy_inputs_fail_closed(
    policy: BoundaryPruningForkPolicyV1,
) -> None:
    with pytest.raises(BoundaryControllerContractError):
        plan_cuda_architecture_b_boundary_controller_v1(
            _summary(), _descriptors(), policy
        )


def test_malformed_summary_and_descriptor_bounds_fail_closed() -> None:
    incomplete = freeze_cell_decision_summary_v1(
        validity=np.array([[1, 1, 1, 0]], dtype=np.uint8),
        status=np.array([[0, 0, 0, 9]], dtype=np.int32),
        score_basis=np.array([[[4.0], [3.0], [2.0], [0.0]]], dtype=np.float64),
        stable_cell_ids=np.array([40, 10, 30, 20], dtype=np.int64),
        score_policy_id=SCORE_POLICY_ID,
        score_policy_digest=SCORE_POLICY_DIGEST,
    )
    wrong_cell_count = tuple(
        replace(descriptor, shape=(5, *descriptor.shape[1:]))
        for descriptor in _descriptors()
    )

    with pytest.raises(BoundaryControllerContractError, match="one successful"):
        plan_cuda_architecture_b_boundary_controller_v1(
            incomplete, _descriptors(), _policy()
        )
    with pytest.raises(
        BoundaryControllerContractError,
        match="does not match the frozen manifest layout and semantics",
    ):
        plan_cuda_architecture_b_boundary_controller_v1(
            _summary(), wrong_cell_count, _policy()
        )


def test_full_state_payloads_are_rejected_and_output_has_no_array_payload() -> None:
    policy_payload = _policy().to_dict()
    policy_payload["tensor_state"] = np.ones((4, 4), dtype=np.float32)
    descriptor_payload = _descriptors()[0].to_dict()
    descriptor_payload["full_state"] = [1, 2, 3]

    with pytest.raises(BoundaryControllerContractError, match="unknown"):
        plan_cuda_architecture_b_boundary_controller_v1(
            _summary(), _descriptors(), policy_payload
        )
    with pytest.raises(BoundaryControllerContractError, match="payload"):
        plan_cuda_architecture_b_boundary_controller_v1(
            _summary(), (descriptor_payload, *_descriptors()[1:]), _policy()
        )

    payload = _plan().to_dict()

    def assert_plain_json(value: Any) -> None:
        assert not isinstance(value, np.ndarray)
        if isinstance(value, dict):
            for child in value.values():
                assert_plain_json(child)
        elif isinstance(value, list):
            for child in value:
                assert_plain_json(child)
        else:
            assert value is None or type(value) in {bool, int, float, str}

    assert_plain_json(payload)
    json.dumps(payload, allow_nan=False)
