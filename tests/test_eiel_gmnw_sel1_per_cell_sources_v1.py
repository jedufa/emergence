from __future__ import annotations

from dataclasses import dataclass
import inspect

import pytest
import torch

from engine.perf import (
    cuda_architecture_b_direct_resident_source_allocator_v1 as allocator,
)


def _projection(cell: int) -> dict[str, object]:
    return {
        "schema_version": "checkpoint_projection_fixture_v1",
        "metadata": {
            "world": "patch",
            "generation": 4,
            "tick": 0,
            "agent_grid_runner_phase": "generation_boundary",
            "population_size": 64,
            "configured_steps_per_generation": 80,
            "native_runtime_fixed_policy_family": "gru",
            "capability_mask": {"patch": True, "gru": True, "wdl": True},
            "native_runtime_numpy_policy_state": {"cell": cell},
            "native_runtime_rng_state": 10_000 + cell,
            "population_next_id": 100 + cell,
            "import_digest": f"{cell:064x}",
        },
        "tensors": {
            "slot_to_agent_id": torch.tensor(
                [cell * 100 + 1, cell * 100 + 2], dtype=torch.int64
            ),
            "alive_mask": torch.tensor([True, True], dtype=torch.bool),
            "policy_use_gru": torch.tensor([True, True], dtype=torch.bool),
            "hmem": torch.full((2, 3), float(cell), dtype=torch.float32),
            "pcg64_state_words": torch.tensor(
                [cell, cell + 1, cell + 2, cell + 3], dtype=torch.int64
            ),
        },
    }


def test_b72_distinct_checkpoint_sources_preserve_private_cells_on_cpu() -> None:
    projections = [_projection(cell) for cell in range(72)]

    batched = allocator.stack_validated_per_cell_checkpoint_projections_v1(
        projections,
        batch_cells=72,
        device=torch.device("cpu"),
    )

    assert tuple(batched["slot_to_agent_id"].shape) == (72, 2)
    assert tuple(batched["hmem"].shape) == (72, 2, 3)
    assert tuple(batched["pcg64_state_words"].shape) == (72, 4)
    for cell in range(72):
        assert torch.equal(
            batched["slot_to_agent_id"][cell],
            projections[cell]["tensors"]["slot_to_agent_id"],
        )
        assert batched["hmem"][cell].eq(float(cell)).all()
        assert int(batched["pcg64_state_words"][cell, 0]) == cell
    assert batched["hmem"].is_contiguous()
    assert batched["hmem"].untyped_storage().data_ptr() != (
        projections[0]["tensors"]["hmem"].untyped_storage().data_ptr()
    )
    batched["hmem"][0].fill_(-1.0)
    assert batched["hmem"][1].eq(1.0).all()


def test_legacy_one_source_repeat_path_remains_byte_identical() -> None:
    source = torch.arange(24, dtype=torch.int32).reshape(3, 8)
    expected = source.unsqueeze(0).to("cpu").repeat(7, 1, 1).contiguous()

    actual = allocator._stack(
        source,
        device=torch.device("cpu"),
        batch_cells=7,
    )

    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert actual.numpy().tobytes() == expected.numpy().tobytes()
    signature = inspect.signature(allocator.allocate_direct_resident_source_v1)
    assert (
        signature.parameters["per_cell_requests"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert signature.parameters["per_cell_requests"].default is None


def test_ecology_seed_metadata_is_private_per_cell() -> None:
    left = {
        "seed": 101,
        "task_seed": 101,
        "world": "patch",
        "population_size": 64,
    }
    right = {
        "seed": 202,
        "task_seed": 202,
        "world": "patch",
        "population_size": 64,
    }

    assert allocator._shared_projection_metadata_v1(left) == {
        "world": "patch",
        "population_size": 64,
    }
    assert allocator._shared_projection_metadata_v1(right) == {
        "world": "patch",
        "population_size": 64,
    }


def test_genome_v2_mutation_metadata_is_private_per_cell() -> None:
    left = {
        "world": "patch",
        "population_size": 64,
        "genome_upgrade_v2_mutation_rate_q10": 0,
        "genome_upgrade_v2_mutation_step_q10": 0,
    }
    right = {
        "world": "patch",
        "population_size": 64,
        "genome_upgrade_v2_mutation_rate_q10": 16,
        "genome_upgrade_v2_mutation_step_q10": 16,
    }

    assert allocator._shared_projection_metadata_v1(left) == {
        "world": "patch",
        "population_size": 64,
    }
    assert allocator._shared_projection_metadata_v1(right) == {
        "world": "patch",
        "population_size": 64,
    }


def test_legacy_turnover_final_live_value_is_private_per_cell() -> None:
    projections = [_projection(0), _projection(1)]
    projections[0]["metadata"]["legacy_turnover_final_live_value"] = 0
    projections[1]["metadata"]["legacy_turnover_final_live_value"] = 64

    batched = allocator.stack_validated_per_cell_checkpoint_projections_v1(
        projections,
        batch_cells=2,
        device="cpu",
    )

    assert tuple(batched["alive_mask"].shape) == (2, 2)


@dataclass(frozen=True)
class _GenomeV2CellConfig:
    generation: int = 0
    patch_capacity: int = 256
    genome_upgrade_v2_founder_profile_id: str = "neutral"
    genome_upgrade_v2_founder_profile_sha256: str = "neutral-sha"
    genome_upgrade_v2_mutation_rate_q10: int = 0
    genome_upgrade_v2_mutation_step_q10: int = 0
    turnover_selection_seed: int = 0


def test_per_cell_configuration_allows_genome_v2_mutation_regime_difference() -> None:
    left = _GenomeV2CellConfig()
    right = _GenomeV2CellConfig(
        genome_upgrade_v2_founder_profile_id="heterozygous",
        genome_upgrade_v2_founder_profile_sha256="heterozygous-sha",
        genome_upgrade_v2_mutation_rate_q10=16,
        genome_upgrade_v2_mutation_step_q10=16,
        turnover_selection_seed=202,
    )

    assert allocator._same_configuration_except_founder_profile_v1(left, right)


def test_projection_may_preserve_intentional_views_within_one_private_cell() -> None:
    projections = [_projection(0), _projection(1)]
    for projection in projections:
        tensors = projection["tensors"]
        tensors["slot_to_agent_id_view"] = tensors["slot_to_agent_id"].view_as(
            tensors["slot_to_agent_id"]
        )

    batched = allocator.stack_validated_per_cell_checkpoint_projections_v1(
        projections,
        batch_cells=2,
        device="cpu",
    )

    assert torch.equal(batched["slot_to_agent_id"], batched["slot_to_agent_id_view"])


def test_policy_family_and_gru_capability_are_private_per_cell() -> None:
    projections = [_projection(0), _projection(1), _projection(2)]
    projections[1]["metadata"]["native_runtime_fixed_policy_family"] = "mlp"
    projections[1]["metadata"]["capability_mask"]["gru"] = False
    projections[1]["tensors"]["policy_use_gru"].fill_(False)
    projections[2]["metadata"]["native_runtime_fixed_policy_family"] = "mixed"
    projections[2]["tensors"]["policy_use_gru"][1] = False

    batched = allocator.stack_validated_per_cell_checkpoint_projections_v1(
        projections,
        batch_cells=3,
        device="cpu",
    )

    assert tuple(batched["hmem"].shape) == (3, 2, 3)


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda projection: projection["metadata"].pop(
                "native_runtime_fixed_policy_family"
            ),
            "policy family must be exactly",
        ),
        (
            lambda projection: projection["metadata"].__setitem__(
                "native_runtime_fixed_policy_family", "unknown"
            ),
            "policy family must be exactly",
        ),
        (
            lambda projection: projection["metadata"].pop("capability_mask"),
            "capability_mask must be a mapping",
        ),
        (
            lambda projection: projection["metadata"].__setitem__(
                "capability_mask", None
            ),
            "capability_mask must be a mapping",
        ),
        (
            lambda projection: projection["metadata"]["capability_mask"].pop("gru"),
            "GRU capability contradicts",
        ),
        (
            lambda projection: projection["metadata"]["capability_mask"].__setitem__(
                "gru", False
            ),
            "GRU capability contradicts",
        ),
        (
            lambda projection: projection["tensors"].pop("alive_mask"),
            "policy-family tensors must appear together",
        ),
        (
            lambda projection: projection["tensors"].__setitem__(
                "alive_mask", torch.ones(2, dtype=torch.int8)
            ),
            "same-shape CPU bool tensors",
        ),
        (
            lambda projection: projection["tensors"].__setitem__(
                "policy_use_gru", torch.ones(3, dtype=torch.bool)
            ),
            "same-shape CPU bool tensors",
        ),
        (
            lambda projection: projection["tensors"]["policy_use_gru"].__setitem__(
                0, False
            ),
            "GRU family contains a live MLP row",
        ),
    ),
)
def test_per_cell_policy_family_metadata_fails_closed(
    mutate: object,
    message: str,
) -> None:
    projections = [_projection(0), _projection(1)]
    mutate(projections[1])

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match=message,
    ):
        allocator.stack_validated_per_cell_checkpoint_projections_v1(
            projections,
            batch_cells=2,
            device="cpu",
        )


def test_mlp_policy_family_rejects_live_gru_rows() -> None:
    projections = [_projection(0), _projection(1)]
    projections[1]["metadata"]["native_runtime_fixed_policy_family"] = "mlp"
    projections[1]["metadata"]["capability_mask"]["gru"] = False

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="MLP family contains a live GRU row",
    ):
        allocator.stack_validated_per_cell_checkpoint_projections_v1(
            projections,
            batch_cells=2,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("family", "gru_capability"),
    (("mlp", True), ("gru", False), ("mixed", False)),
)
def test_each_policy_family_rejects_contradictory_gru_capability(
    family: str,
    gru_capability: bool,
) -> None:
    projections = [_projection(0), _projection(1)]
    projections[1]["metadata"]["native_runtime_fixed_policy_family"] = family
    projections[1]["metadata"]["capability_mask"]["gru"] = gru_capability
    if family == "mlp":
        projections[1]["tensors"]["policy_use_gru"].fill_(False)

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="GRU capability contradicts",
    ):
        allocator.stack_validated_per_cell_checkpoint_projections_v1(
            projections,
            batch_cells=2,
            device="cpu",
        )


def test_non_gru_capability_difference_still_fails_closed() -> None:
    projections = [_projection(0), _projection(1)]
    projections[1]["metadata"]["capability_mask"]["wdl"] = False

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="metadata/config mismatch",
    ):
        allocator.stack_validated_per_cell_checkpoint_projections_v1(
            projections,
            batch_cells=2,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda projections: projections.pop(),
            "count must exactly equal batch_cells",
        ),
        (
            lambda projections: projections[1]["metadata"].__setitem__(
                "population_size", 32
            ),
            "metadata/config mismatch",
        ),
        (
            lambda projections: projections[1].__setitem__(
                "schema_version", "checkpoint_projection_fixture_v2"
            ),
            "schema_version mismatch",
        ),
        (
            lambda projections: projections[1]["tensors"].__setitem__(
                "hmem", torch.zeros((2, 4), dtype=torch.float32)
            ),
            "shape/dtype/layout mismatch",
        ),
        (
            lambda projections: projections[1]["tensors"].__setitem__(
                "hmem", torch.zeros((2, 3), dtype=torch.float64)
            ),
            "shape/dtype/layout mismatch",
        ),
        (
            lambda projections: projections[1]["tensors"].__setitem__(
                "hmem", projections[0]["tensors"]["hmem"]
            ),
            "storage alias",
        ),
    ),
)
def test_per_cell_checkpoint_sources_fail_closed(
    mutate: object,
    message: str,
) -> None:
    projections = [_projection(0), _projection(1)]
    mutate(projections)

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match=message,
    ):
        allocator.stack_validated_per_cell_checkpoint_projections_v1(
            projections,
            batch_cells=2,
            device="cpu",
        )


def test_per_cell_checkpoint_tensor_schema_order_mismatch_fails_closed() -> None:
    projections = [_projection(0), _projection(1)]
    tensors = projections[1]["tensors"]
    projections[1]["tensors"] = dict(reversed(tuple(tensors.items())))

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="tensor schema/layout mismatch",
    ):
        allocator.stack_validated_per_cell_checkpoint_projections_v1(
            projections,
            batch_cells=2,
            device="cpu",
        )
