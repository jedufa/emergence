from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from evolution.diploid_inheritance_v1 import DIPLOID_SCALAR_LOCI
from perf.cuda_diploid_state_bank_v1 import (
    ALLELE_FIELD_NAMES,
    ALLELE_PLANE_COUNT,
    CHECKPOINT_TENSOR_NAMES,
    CANONICAL_HEREDITARY_DIGEST_BYTES,
    CudaDiploidStateBankError,
    INCREMENTAL_ARRAY_COUNT,
    PHENOTYPE_SOURCE_FIELDS,
    migrate_homozygous_from_phenotype_v1,
    restore_cuda_diploid_state_bank_v1,
)


def _phenotype_state(
    *, cells: int, capacity: int, device: torch.device
) -> dict[str, torch.Tensor]:
    state = {}
    for index, locus in enumerate(DIPLOID_SCALAR_LOCI[:-1]):
        state[PHENOTYPE_SOURCE_FIELDS[locus]] = torch.full(
            (cells, capacity),
            float(index) / 100.0,
            dtype=torch.float32,
            device=device,
        )
    state["canonical_hereditary_digest"] = torch.arange(
        cells * capacity * CANONICAL_HEREDITARY_DIGEST_BYTES,
        dtype=torch.uint8,
        device=device,
    ).reshape(cells, capacity, CANONICAL_HEREDITARY_DIGEST_BYTES)
    state["canonical_hereditary_template"] = torch.arange(
        cells * 11,
        dtype=torch.uint8,
        device=device,
    ).reshape(cells, 11)
    state["diploid_payload_template"] = torch.arange(
        cells * 13,
        dtype=torch.uint8,
        device=device,
    ).reshape(cells, 13)
    state["diploid_genotype_template"] = torch.arange(
        cells * 17,
        dtype=torch.uint8,
        device=device,
    ).reshape(cells, 17)
    return state


@pytest.mark.parametrize("cells,capacity", [(1, 3), (4, 17)])
def test_zero_draw_migration_allocates_34_soa_arrays(
    cells: int, capacity: int
) -> None:
    state = _phenotype_state(
        cells=cells, capacity=capacity, device=torch.device("cpu")
    )
    bank = migrate_homozygous_from_phenotype_v1(
        state, cells=cells, capacity=capacity, device="cpu"
    )

    assert ALLELE_PLANE_COUNT == 34
    assert INCREMENTAL_ARRAY_COUNT == 39
    assert tuple(bank.storage.shape) == (cells, 34, capacity)
    assert bank.incremental_nbytes == (
        cells * capacity * (35 * 4 + 32)
        + cells * (11 + 13 + 17)
    )
    named = bank.named_tensors()
    assert tuple(named) == CHECKPOINT_TENSOR_NAMES
    assert len(ALLELE_FIELD_NAMES) == 34
    assert len({tensor.data_ptr() for tensor in named.values()}) == 39
    for locus in DIPLOID_SCALAR_LOCI:
        left = bank.allele_plane(0, locus)
        right = bank.allele_plane(1, locus)
        assert torch.equal(left, right)
        source_name = PHENOTYPE_SOURCE_FIELDS[locus]
        expected = (
            state[source_name]
            if source_name in state
            else torch.zeros((cells, capacity), dtype=torch.float32)
        )
        assert torch.equal(left, expected)


def test_checkpoint_round_trip_is_exact_and_hash_bound() -> None:
    state = _phenotype_state(cells=2, capacity=5, device=torch.device("cpu"))
    bank = migrate_homozygous_from_phenotype_v1(
        state, cells=2, capacity=5, device="cpu"
    )
    bank.storage[0, 0, 0] = -0.75
    bank.storage[1, -1, -1] = 0.75
    bank.genome_resource_handling_axis[0, 2] = 0.25
    projection = bank.checkpoint_projection()
    restored = restore_cuda_diploid_state_bank_v1(
        projection, device="cpu"
    )

    assert torch.equal(restored.storage, bank.storage)
    assert torch.equal(
        restored.genome_resource_handling_axis,
        bank.genome_resource_handling_axis,
    )
    assert torch.equal(
        restored.canonical_hereditary_digest,
        bank.canonical_hereditary_digest,
    )
    assert restored.metadata == bank.metadata

    malformed = deepcopy(projection)
    malformed["metadata"]["linkage_map_sha256"] = "0" * 64
    with pytest.raises(
        CudaDiploidStateBankError, match="contract or hash mismatch"
    ):
        restore_cuda_diploid_state_bank_v1(malformed, device="cpu")


def test_restore_rejects_missing_or_wrong_tensors() -> None:
    state = _phenotype_state(cells=1, capacity=2, device=torch.device("cpu"))
    projection = migrate_homozygous_from_phenotype_v1(
        state, cells=1, capacity=2, device="cpu"
    ).checkpoint_projection()

    missing = deepcopy(projection)
    missing["tensors"].pop(ALLELE_FIELD_NAMES[0])
    with pytest.raises(CudaDiploidStateBankError, match="tensor set mismatch"):
        restore_cuda_diploid_state_bank_v1(missing, device="cpu")

    wrong = deepcopy(projection)
    wrong["tensors"][ALLELE_FIELD_NAMES[0]] = torch.zeros(
        (1, 2), dtype=torch.float64
    )
    with pytest.raises(CudaDiploidStateBankError, match="incompatible"):
        restore_cuda_diploid_state_bank_v1(wrong, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_state_bank_resides_on_device_and_round_trips() -> None:
    device = torch.device("cuda")
    state = _phenotype_state(cells=3, capacity=11, device=device)
    bank = migrate_homozygous_from_phenotype_v1(
        state, cells=3, capacity=11, device=device
    )
    assert bank.device.type == "cuda"
    assert all(
        tensor.device.type == "cuda"
        for tensor in bank.named_tensors().values()
    )
    restored = restore_cuda_diploid_state_bank_v1(
        bank.checkpoint_projection(), device=device
    )
    assert torch.equal(restored.storage, bank.storage)
