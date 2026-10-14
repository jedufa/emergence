from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from agents.genome import Genome
from engine.evolution.inheritance_linkage_digest_table_v1 import (
    build_linkage_digest_table_v1,
    founder_codes_v1,
)
from engine.evolution.inheritance_registry_v1 import canonical_parent_order
from engine.perf.cuda_architecture_b_linkage_resident_sidecar_v1 import (
    CudaLinkageResidentSidecarError,
    CudaLinkageResidentSidecarV1,
    NO_DONOR_ORDINAL,
    allocate_cuda_linkage_resident_sidecar_v1,
)


def _founder(marker: int) -> Genome:
    low = marker < 0
    return Genome.from_dict(
        {
            "schema_version": "modular_genome_v1",
            "hidden_size": 16 if low else 32,
            "exploration_bias": -0.8 if low else 0.8,
            "aggression_bias": -1.2 if low else 1.2,
            "reproduction_threshold": 20.0 if low else 70.0,
            "mutation_rate": 0.01 if low else 0.02,
            "action_temperature": 0.5 if low else 2.0,
            "use_gru": False,
            "memory_dim": 0,
            "drive_inertia": 0.2 if low else 0.9,
            "drive_noise": 0.01 if low else 0.08,
            "drive_logit_gain": 0.4 if low else 2.4,
            "pred_lr": 0.1 if low else 0.8,
            "drive_sp_hunger": 0.2 if low else 0.8,
            "drive_sp_pain": 0.1 if low else 0.7,
            "drive_sp_fatigue": 0.3 if low else 0.9,
            "drive_sp_fear": 0.15 if low else 0.65,
            "exp_prior_strength": 0.2 if low else 0.7,
            "exp_prior_noise": 0.02 if low else 0.3,
            "drive_birth_noise": 0.01 if low else 0.15,
            "resource_handling_axis": -0.75 if low else 0.75,
            "family_head_residuals": (
                {"movement": 0.3, "gather": -0.3, "rest": 0.0, "attack": 0.0}
                if low
                else {
                    "movement": 0.0,
                    "gather": 0.0,
                    "rest": -0.3,
                    "attack": 0.3,
                }
            ),
            "developmental_program": {
                "schema_version": "developmental_program_v1",
                "rules": [],
            },
            "modules": [],
            "connections": [],
        }
    )


def _table():
    ordered, _digests = canonical_parent_order(_founder(-1), _founder(1))
    return build_linkage_digest_table_v1(*ordered)


def _alternate_table():
    left = _founder(-1)
    right = _founder(1)
    left.exploration_bias = -0.55
    right.exploration_bias = 0.35
    left.refresh_canonical_payload_cache()
    right.refresh_canonical_payload_cache()
    ordered, _digests = canonical_parent_order(left, right)
    return build_linkage_digest_table_v1(*ordered)


def test_sidecar_fails_closed_instead_of_allocating_cpu_owner() -> None:
    table = _table()
    source = torch.tensor([[0, table.genotype_count - 1, -1]])
    with pytest.raises(
        CudaLinkageResidentSidecarError,
        match="explicit CUDA device",
    ):
        allocate_cuda_linkage_resident_sidecar_v1(
            table,
            source_genotype_codes=source,
            birth_capacity=1,
            device="cpu",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_only_explicit_qualification_failure_cell_may_be_nonzero() -> None:
    table = _table()
    founder_0, founder_1 = founder_codes_v1(table)
    owner = allocate_cuda_linkage_resident_sidecar_v1(
        table,
        source_genotype_codes=torch.tensor(
            [
                [founder_0, founder_1, -1],
                [founder_1, founder_0, -1],
            ],
            dtype=torch.int64,
            device="cuda",
        ),
        birth_capacity=1,
        device="cuda",
    )
    owner.transaction_status[1] = 2
    with pytest.raises(
        CudaLinkageResidentSidecarError,
        match="transaction status",
    ):
        owner.validate_exact()
    owner.validate_exact(allowed_failure_cell_indices=(1,))
    with pytest.raises(
        CudaLinkageResidentSidecarError,
        match="transaction status",
    ):
        owner.export_checkpoint_projection()
    projection = owner.export_checkpoint_projection(
        allowed_failure_cell_indices=(1,)
    )
    with pytest.raises(
        CudaLinkageResidentSidecarError,
        match="transaction status",
    ):
        CudaLinkageResidentSidecarV1.from_checkpoint_projection(
            table, projection, device="cuda"
        )
    restored = CudaLinkageResidentSidecarV1.from_checkpoint_projection(
        table,
        projection,
        device="cuda",
        allowed_failure_cell_indices=(1,),
    )
    restored.validate_exact(allowed_failure_cell_indices=(1,))
    with pytest.raises(
        CudaLinkageResidentSidecarError,
        match="unique in-range",
    ):
        owner.validate_exact(allowed_failure_cell_indices=(2,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_source_stage_commit_projection_and_fresh_owner_are_exact() -> None:
    table = _table()
    alternate = _alternate_table()
    assert alternate.table_sha256 != table.table_sha256
    founder_0, founder_1 = founder_codes_v1(table)
    source = torch.tensor(
        [
            [founder_0, founder_1, -1, -1],
            [founder_1, founder_0, -1, -1],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    owner = allocate_cuda_linkage_resident_sidecar_v1(
        (table, alternate),
        source_genotype_codes=source,
        birth_capacity=2,
        device="cuda",
    )
    assert owner.genotype_codes.device.type == "cuda"
    assert owner.digest_rank_by_code.device.type == "cuda"
    assert owner.digest_rank_by_code.tolist() == [
        list(table.rank_by_code),
        list(alternate.rank_by_code),
    ]

    parents = torch.tensor([[0, -1], [0, -1]], device="cuda")
    mates = torch.tensor([[1, -1], [1, -1]], device="cuda")
    counts = torch.tensor([1, 1], device="cuda")
    owner.stage_canonical_parents(
        parent_slots=parents,
        mate_slots=mates,
        birth_counts=counts,
    )

    # Canonical order follows exact hereditary digest rank, not slot order.
    expected_slots = []
    for cell in range(2):
        cell_table = (table, alternate)[cell]
        code_a = int(source[cell, 0].item())
        code_b = int(source[cell, 1].item())
        expected_codes = cell_table.canonical_pair(code_a, code_b)
        expected_slots.append((0, 1) if expected_codes == (code_a, code_b) else (1, 0))
    assert owner.canonical_parent_0_slots[:, 0].cpu().tolist() == [
        row[0] for row in expected_slots
    ]
    assert owner.canonical_parent_1_slots[:, 0].cpu().tolist() == [
        row[1] for row in expected_slots
    ]

    donors = torch.full(
        (2, 2, table.block_count),
        NO_DONOR_ORDINAL,
        dtype=torch.uint8,
        device="cuda",
    )
    donors[0, 0] = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8, device="cuda")
    donors[1, 0] = torch.tensor([1, 0, 1, 0, 1, 0, 1], dtype=torch.uint8, device="cuda")
    statuses = torch.zeros_like(donors)
    children = torch.tensor([[2, -1], [2, -1]], device="cuda")
    owner.commit_linked_births(
        child_slots=children,
        donor_ordinals=donors,
        donor_statuses=statuses,
        committed_counts=counts,
        donor_draws_added=counts * table.block_count,
        transaction_status=torch.zeros(2, dtype=torch.int32, device="cuda"),
    )

    for cell in range(2):
        cell_table = (table, alternate)[cell]
        canonical_0_slot = expected_slots[cell][0]
        canonical_1_slot = expected_slots[cell][1]
        expected = cell_table.child_code(
            int(source[cell, canonical_0_slot].item()),
            int(source[cell, canonical_1_slot].item()),
            [int(value) for value in donors[cell, 0].cpu().tolist()],
        )
        assert int(owner.genotype_codes[cell, 2].item()) == expected
        assert (
            int(owner.project_digest_ranks()[cell, 2].item())
            == (cell_table.rank_by_code[expected])
        )

    projection = owner.export_checkpoint_projection()
    assert len(projection["identity"]["registry_binding_sha256"]) == 64
    assert projection["identity"]["table_sha256_by_cell"] == [
        table.table_sha256,
        alternate.table_sha256,
    ]
    with pytest.raises(
        CudaLinkageResidentSidecarError,
        match="explicit CUDA device",
    ):
        CudaLinkageResidentSidecarV1.from_checkpoint_projection(
            (table, alternate), projection, device="cpu"
        )
    restored = CudaLinkageResidentSidecarV1.from_checkpoint_projection(
        (table, alternate), projection, device="cuda"
    )
    assert restored is not owner
    assert restored.device.type == "cuda"
    for name in owner._tensor_map():
        assert torch.equal(getattr(owner, name), getattr(restored, name))
    assert (
        restored.export_checkpoint_projection()["projection_sha256"]
        == projection["projection_sha256"]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_turnover_retire_and_second_transaction_survive_checkpoint() -> None:
    table = _table()
    source = torch.tensor(
        [[0, table.genotype_count - 1, -1, -1]],
        dtype=torch.int64,
        device="cuda",
    )
    owner = allocate_cuda_linkage_resident_sidecar_v1(
        table,
        source_genotype_codes=source,
        birth_capacity=2,
        device="cuda",
    )
    rows = torch.tensor([[0, -1]], device="cuda")
    mates = torch.tensor([[1, -1]], device="cuda")
    counts = torch.tensor([1], device="cuda")
    owner.stage_canonical_parents(
        parent_slots=rows, mate_slots=mates, birth_counts=counts
    )
    donors = torch.full(
        (1, 2, table.block_count),
        NO_DONOR_ORDINAL,
        dtype=torch.uint8,
        device="cuda",
    )
    donors[0, 0].fill_(0)
    owner.commit_linked_births(
        child_slots=torch.tensor([[2, -1]], device="cuda"),
        donor_ordinals=donors,
        donor_statuses=torch.zeros_like(donors),
        committed_counts=counts,
        donor_draws_added=counts * table.block_count,
        transaction_status=torch.zeros(1, dtype=torch.int32, device="cuda"),
    )

    owner.retire_slots(
        slots=torch.tensor([[0, -1]], device="cuda"),
        counts=counts,
    )
    owner.stage_canonical_parents(
        parent_slots=torch.tensor([[1, -1]], device="cuda"),
        mate_slots=torch.tensor([[2, -1]], device="cuda"),
        birth_counts=counts,
    )
    donors[0, 0].fill_(1)
    owner.commit_linked_births(
        child_slots=torch.tensor([[0, -1]], device="cuda"),
        donor_ordinals=donors,
        donor_statuses=torch.zeros_like(donors),
        committed_counts=counts,
        donor_draws_added=counts * table.block_count,
        transaction_status=torch.zeros(1, dtype=torch.int32, device="cuda"),
    )
    projection = owner.export_checkpoint_projection()
    fresh = CudaLinkageResidentSidecarV1.from_checkpoint_projection(
        table, projection, device="cuda"
    )
    assert torch.equal(fresh.genotype_codes, owner.genotype_codes)
    assert torch.equal(fresh.donor_ordinals, owner.donor_ordinals)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_checkpoint_and_transactions_reject_wrong_identity_or_partial_rows() -> None:
    table = _table()
    source = torch.tensor(
        [[0, table.genotype_count - 1, -1]],
        dtype=torch.int64,
        device="cuda",
    )
    owner = allocate_cuda_linkage_resident_sidecar_v1(
        table,
        source_genotype_codes=source,
        birth_capacity=1,
        device="cuda",
    )
    counts = torch.tensor([1], device="cuda")
    owner.stage_canonical_parents(
        parent_slots=torch.tensor([[0]], device="cuda"),
        mate_slots=torch.tensor([[1]], device="cuda"),
        birth_counts=counts,
    )
    donors = torch.zeros((1, 1, table.block_count), dtype=torch.uint8, device="cuda")
    failed = torch.zeros_like(donors)
    failed[0, 0, 3] = 1
    before = owner.genotype_codes.clone()
    with pytest.raises(
        CudaLinkageResidentSidecarError,
        match="did not succeed",
    ):
        owner.commit_linked_births(
            child_slots=torch.tensor([[2]], device="cuda"),
            donor_ordinals=donors,
            donor_statuses=failed,
            committed_counts=counts,
            donor_draws_added=counts * table.block_count,
            transaction_status=torch.zeros(1, dtype=torch.int32, device="cuda"),
        )
    assert torch.equal(owner.genotype_codes, before)

    projection = owner.export_checkpoint_projection()
    corrupted = deepcopy(projection)
    corrupted["tensors"]["genotype_codes"][0, 0] = 1
    with pytest.raises(
        CudaLinkageResidentSidecarError,
        match="hash mismatch",
    ):
        CudaLinkageResidentSidecarV1.from_checkpoint_projection(
            table, corrupted, device="cuda"
        )

    stale = deepcopy(projection)
    stale["identity"]["table_sha256"] = "0" * 64
    with pytest.raises(
        CudaLinkageResidentSidecarError,
        match="identity",
    ):
        CudaLinkageResidentSidecarV1.from_checkpoint_projection(
            table, stale, device="cuda"
        )
