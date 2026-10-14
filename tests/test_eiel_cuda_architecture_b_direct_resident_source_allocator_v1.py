from __future__ import annotations

import ast
import inspect
import random
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from evolution.genome_upgrade_v2 import GenomeUpgradeV2V1
from evolution.genome_upgrade_v2_causal_cuda_output_v1 import (
    RECORD_CAPACITY,
    RECORD_WIDTH,
)
from engine.perf import (
    cuda_architecture_b_direct_resident_source_allocator_v1 as allocator,
)
from perf.cxxbackbone10_resident_state_owner import projection_digest
from engine.perf.cuda_architecture_b_lifecycle import (
    pack_cpython_random_state_cuda,
)
from engine.perf.cuda_architecture_b_generation import CudaGenerationConfig
from perf.cuda_architecture_b_flattened_birth_scratch_layout_v2 import (
    BirthScratchSegmentV2,
    make_legacy_tail_birth_scratch_layout_v2,
)
from perf.cuda_genome_upgrade_v2_state_bank import (
    CudaGenomeUpgradeV2StateBank,
    GENOME_V2_FOUNDER_PROFILE_SCHEMA,
    resolve_genome_v2_founder_profile_v1,
)
from perf.cuda_diploid_state_bank_v1 import (
    ALLELE_PLANE_COUNT,
    CANONICAL_HEREDITARY_DIGEST_BYTES,
    CudaDiploidStateBankV1,
    checkpoint_metadata_v1,
)
from perf.inheritance_sidecar_identity_v1 import (
    INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
    inheritance_sidecar_payload_sha256_v1,
)


def _physical_b136_native_capacity_request(
    *,
    event_capacity: int = 41_804,
) -> SimpleNamespace:
    return SimpleNamespace(
        envelope=SimpleNamespace(
            world="patch",
            profile="P2",
            population=256,
            grid=(32, 32),
            max_births_per_tick=3,
            ticks_per_generation=160,
        ),
        tick_native_control={
            "resident_owner_span_ticks": 1_120,
            "native_dispatch_window_ticks": 1_120,
            "source_window_ticks": 160,
            "refill_chunk_ticks": 160,
        },
        import_projection=SimpleNamespace(metadata={"capacity": 512}),
        configuration=SimpleNamespace(
            steps=160,
            ordinary_births_enabled=True,
            generation_turnover_enabled=False,
        ),
        event_capacity=event_capacity,
    )


def test_physical_b136_native_capacity_is_source_derived_and_exact() -> None:
    request = _physical_b136_native_capacity_request()

    allocator._validate_physical_b136_native_event_capacity_v1(
        request,
        batch_cells=136,
        native_event_capacity=41_804,
    )


def test_physical_b136_native_capacity_rejects_generic_default() -> None:
    request = _physical_b136_native_capacity_request(event_capacity=65_536)

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="source-derived exact bound",
    ):
        allocator._validate_physical_b136_native_event_capacity_v1(
            request,
            batch_cells=136,
            native_event_capacity=65_536,
        )


@dataclass(frozen=True)
class _CheckpointTemplateFixture:
    state: dict[str, torch.Tensor]
    columns: tuple[torch.Tensor, ...]
    schema_version: str


def _boundary_projection(
    generation: int,
    *,
    include_runtime_prefix: bool,
) -> SimpleNamespace:
    fields: dict[str, object] = {
        "metadata": {
            "world": "patch",
            "generation": generation,
            "tick": 0,
            "agent_grid_runner_phase": "generation_boundary",
            "event_tape_active_count": 3,
        },
        "tensors": {"sentinel": torch.tensor([generation])},
    }
    if include_runtime_prefix:
        fields.update(
            {
                "restore_count": 2,
                "generation_calls": generation,
                "event_cursor": 31,
                "runtime_event_cursor": 47,
                "last_applied_actions": 80,
                "last_moved": 12,
                "last_gathered": 9,
                "last_policy_refill_calls": 4,
                "last_policy_refill_rows": 64,
                "pending_generation_free_agent_ids": (101,),
                "pending_generation_free_slots": (7,),
                "last_world": "patch",
            }
        )
    return SimpleNamespace(**fields)


def test_exact_native_abi_inventories_are_frozen() -> None:
    assert len(allocator.ORDINARY_BIRTH_OUTPUT_FIELDS) == 63
    assert allocator.ORDINARY_BIRTH_OUTPUT_FIELDS[-5:] == (
        "birth_genome_v2_causal_record_rows",
        "birth_genome_v2_causal_record_valid",
        "birth_genome_v2_causal_record_count",
        "birth_genome_v2_causal_record_dropped_count",
        "birth_genome_v2_causal_capture_enabled",
    )
    assert len(allocator.TURNOVER_OUTPUT_FIELDS) == 16
    assert len(allocator.TURNOVER_MUTABLE_STATE_FIELDS) == 122
    assert allocator.TURNOVER_MUTABLE_STATE_FIELDS[:2] == (
        "active_agent_ids",
        "active_agent_slots",
    )


def test_resource_maintenance_charge_floor_is_idempotent() -> None:
    batched = {
        "cost_charge_scalar": torch.tensor([[0.0, 0.01, 0.04]], dtype=torch.float32)
    }
    config = SimpleNamespace(resource_maintenance_energy_cost_per_tick=0.035)

    allocator._apply_resource_maintenance_cost_charge_floor_v1(batched, config=config)
    once = batched["cost_charge_scalar"].clone()
    allocator._apply_resource_maintenance_cost_charge_floor_v1(batched, config=config)

    expected = torch.tensor([[0.035, 0.035, 0.04]], dtype=torch.float32)
    torch.testing.assert_close(once, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        batched["cost_charge_scalar"], expected, rtol=0.0, atol=0.0
    )


def test_resource_maintenance_charge_fails_closed_for_invalid_values() -> None:
    batched = {"cost_charge_scalar": torch.zeros((1, 2), dtype=torch.float32)}

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="resource_maintenance_energy_cost_per_tick",
    ):
        allocator._apply_resource_maintenance_cost_charge_floor_v1(
            batched,
            config=SimpleNamespace(resource_maintenance_energy_cost_per_tick=-0.1),
        )


def test_disabled_birth_authority_omits_write_only_birth_outputs() -> None:
    source = inspect.getsource(allocator.allocate_direct_resident_source_v1)

    allocation = source[
        source.index("birth_outputs =") : source.index("outputs.update(birth_outputs)")
    ]
    assert "if ordinary_births_enabled" not in allocation
    assignment = source[source.index("birth_output_tensors =") :]
    assert "if ordinary_births_enabled" in assignment
    assert "else []" in assignment


@pytest.mark.parametrize(
    ("cell_requests", "per_cell_ids", "batch_cells", "projection_count", "expected"),
    (
        (None, ("cell-a", "cell-b"), 2, 1, True),
        ((), ("cell-a", "cell-b"), 2, 1, False),
        (None, None, 2, 1, False),
        (None, ("cell-a", "cell-b"), 2, 2, False),
    ),
)
def test_streamed_bwide_seed_is_distinct_from_multicell_restore(
    cell_requests: object,
    per_cell_ids: object,
    batch_cells: int,
    projection_count: int,
    expected: bool,
) -> None:
    assert allocator._is_streamed_bwide_seed_v1(
        cell_requests=cell_requests,  # type: ignore[arg-type]
        per_cell_ids=per_cell_ids,  # type: ignore[arg-type]
        batch_cells=batch_cells,
        canonical_projection_count=projection_count,
    ) is expected


def test_resident_policy_birth_tapes_are_not_demoted_to_cpu() -> None:
    allocation = inspect.getsource(allocator.allocate_direct_resident_source_v1)

    assert 'canonical_tape = source[name].detach().to(device="cpu")' not in allocation
    assert 'device=torch.device("cpu"),' not in allocation


def test_compact_policy_birth_tapes_are_narrowly_admitted_for_cuda_staging() -> None:
    allocator._validate_source_tensor_devices_v1(
        {
            "policy_birth_delta_tape": torch.zeros((1, 1)),
            "policy_birth_mutation_mask_tape": torch.zeros((1, 1), dtype=torch.bool),
            "policy_birth_crossover_mask_packed_tape": torch.zeros(
                (1, 1), dtype=torch.uint8
            ),
            "policy_birth_post_rng_state_packed_tape": torch.zeros(
                (1, 1), dtype=torch.uint8
            ),
            "policy_birth_semantic_draws_tape": torch.zeros((1,), dtype=torch.int64),
            "policy_birth_tape_cursor": torch.zeros((1,), dtype=torch.int64),
        },
        expected_device=torch.device("cuda"),
        name="primary",
    )

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="other.*must be on cuda",
    ):
        allocator._validate_source_tensor_devices_v1(
            {"other": torch.zeros((1,))},
            expected_device=torch.device("cuda"),
            name="primary",
        )


def test_w16p128_patch_geometry_is_admitted_for_source_allocation() -> None:
    assert (128, (16, 16)) in allocator.QUALIFIED_PATCH_GEOMETRIES_V1


def test_capacity_bounded_active_slots_preserve_sorted_id_pairing() -> None:
    active_ids = torch.tensor([[10, 20, -1]], dtype=torch.int64)
    active_count = torch.tensor([2], dtype=torch.int64)
    compact_ids = torch.tensor([[10, 20, -1]], dtype=torch.int64)
    compact_slots = torch.tensor([[2, 0, -1]], dtype=torch.int32)
    slot_to_id = torch.tensor([[20, -1, 10]], dtype=torch.int64)
    alive = torch.tensor([[True, False, True]], dtype=torch.bool)

    assert allocator._capacity_bounded_active_agent_slots_v1(
        active_ids,
        active_count,
        compact_ids,
        compact_slots,
        slot_to_id,
        alive,
    ).tolist() == [[2, 0, -1]]


def test_compact_identity_values_come_from_resident_projections() -> None:
    first = {
        "agent_id_to_slot_agent_ids": torch.tensor([7], dtype=torch.int64),
        "agent_id_to_slot_slots": torch.tensor([2], dtype=torch.int32),
    }
    second = {
        "agent_id_to_slot_agent_ids": torch.tensor([11], dtype=torch.int64),
        "agent_id_to_slot_slots": torch.tensor([0], dtype=torch.int32),
    }

    ids, slots = allocator._compact_identity_values_v1(first, [first, second])

    assert [value.tolist() for value in ids] == [[7], [11]]
    assert [value.tolist() for value in slots] == [[2], [0]]


@pytest.mark.parametrize("corruption", ("compact_slot", "slot_identity", "alive"))
def test_capacity_bounded_active_slots_reject_pair_corruption(
    corruption: str,
) -> None:
    active_ids = torch.tensor([[10, 20, -1]], dtype=torch.int64)
    active_count = torch.tensor([2], dtype=torch.int64)
    compact_ids = torch.tensor([[10, 20, -1]], dtype=torch.int64)
    compact_slots = torch.tensor([[2, 0, -1]], dtype=torch.int32)
    slot_to_id = torch.tensor([[20, -1, 10]], dtype=torch.int64)
    alive = torch.tensor([[True, False, True]], dtype=torch.bool)
    if corruption == "compact_slot":
        compact_slots[0, 0] = 1
    elif corruption == "slot_identity":
        slot_to_id[0, 2] = 99
    else:
        alive[0, 2] = False

    with pytest.raises(RuntimeError, match="contradict resident slot authority"):
        allocator._capacity_bounded_active_agent_slots_v1(
            active_ids,
            active_count,
            compact_ids,
            compact_slots,
            slot_to_id,
            alive,
        )


def test_capacity_bounded_active_slots_accept_int32_max_without_dense_rows() -> None:
    maximum_id = torch.iinfo(torch.int32).max
    active_ids = torch.tensor([[7, maximum_id, -1]], dtype=torch.int64)
    active_count = torch.tensor([2], dtype=torch.int64)
    compact_ids = active_ids.clone()
    compact_slots = torch.tensor([[2, 0, -1]], dtype=torch.int32)
    slot_to_id = torch.tensor([[maximum_id, -1, 7]], dtype=torch.int64)
    alive = torch.tensor([[True, False, True]], dtype=torch.bool)

    active_slots = allocator._capacity_bounded_active_agent_slots_v1(
        active_ids,
        active_count,
        compact_ids,
        compact_slots,
        slot_to_id,
        alive,
    )

    assert active_slots.tolist() == [[2, 0, -1]]
    assert active_slots.numel() == slot_to_id.numel() == 3


def test_agent_id_grid_canonicalization_rebuilds_slot_keyed_import() -> None:
    agent_grid = torch.tensor(
        [
            [
                [0, -1, 1],
                [-1, 2, -1],
            ]
        ],
        dtype=torch.int32,
    ).contiguous()
    slot_to_id = torch.tensor([[10, 20, 30, -1]], dtype=torch.int64)
    alive = torch.tensor([[True, True, True, False]], dtype=torch.bool)
    pos_y = torch.tensor([[0, 0, 1, -1]], dtype=torch.int32)
    pos_x = torch.tensor([[0, 2, 1, -1]], dtype=torch.int32)

    canonical = allocator._canonicalize_agent_id_grid_v1(
        agent_grid,
        slot_to_id,
        alive,
        pos_y,
        pos_x,
    )

    assert canonical.tolist() == [[[10, -1, 20], [-1, 30, -1]]]
    assert canonical.dtype == torch.int32
    assert canonical.is_contiguous()


def test_genome_v2_observability_config_matches_settlement_contract() -> None:
    config = SimpleNamespace(
        max_energy=100,
        start_health=12.5,
        developmental_maturity_age=9,
    )
    patch_config = {
        "patch_density_radius": 2,
        "patch_density_include_self": True,
        "patch_density_normalize": False,
        "patch_food_amount_max": 7,
        "patch_waste_max": 3.5,
    }

    assert allocator._genome_v2_observability_config_v1(config, patch_config) == {
        "max_energy": 100.0,
        "start_health": 12.5,
        "developmental_maturity_age": 9,
        "patch_density_radius": 2,
        "patch_density_include_self": True,
        "patch_density_normalize": False,
        "patch_food_amount_max": 7.0,
        "patch_waste_max": 3.5,
    }


def test_genome_v2_checkpoint_authority_binding_is_validated_and_canonical() -> None:
    binding = {
        "schema_version": "genome_v2_checkpoint_authority_binding_v1",
        "provider_id": "eiel-production-provider-v1",
        "runtime_profile_id": "eiel-production-runtime-v1",
        "global_genome_profile_id": "genome_upgrade_v2",
        "source_commit": "a" * 40,
        "executable_package_core_sha256": "b" * 64,
        "package_id_sha256": "c" * 64,
        "package_receipt_sha256": "d" * 64,
        "artifact_sha256": "e" * 64,
        "manifest_sha256": "f" * 64,
        "state_schema_sha256": "1" * 64,
        "mechanic_contract_sha256": "2" * 64,
        "capability_contract_sha256": "3" * 64,
        "cancellation_abi_sha256": "4" * 64,
        "observability_schema_sha256": "5" * 64,
        "capacity_layout_contract_sha256": "6" * 64,
        "execution_route_sha256": "7" * 64,
    }
    config = SimpleNamespace(
        genome_v2_checkpoint_authority_binding=tuple(reversed(tuple(binding.items())))
    )

    assert allocator._genome_v2_checkpoint_authority_binding_v1(config) == tuple(
        sorted(binding.items())
    )
    assert allocator._genome_v2_checkpoint_authority_binding_v1(SimpleNamespace()) == ()

    binding["provider_id"] = "legacy-provider"
    with pytest.raises(ValueError, match="identity mismatch"):
        allocator._genome_v2_checkpoint_authority_binding_v1(
            SimpleNamespace(
                genome_v2_checkpoint_authority_binding=tuple(binding.items())
            )
        )


@pytest.mark.parametrize("cells", [1, 2])
def test_fresh_owner_disables_genome_v2_qualification_fault(
    cells: int,
) -> None:
    layout = make_legacy_tail_birth_scratch_layout_v2(
        birth_count=2,
        capacity=256,
        packed_width=9415,
        hmem_width=64,
        torch_state_bytes=5056,
    )
    scratch = torch.full(
        (cells, layout.bytes_per_cell),
        0xA5,
        dtype=torch.uint8,
    )

    allocator._disable_genome_v2_qualification_fault_v1(scratch, layout)

    fault_segment = layout.segment(
        BirthScratchSegmentV2.GENOME_UPGRADE_V2_QUALIFICATION_FAULT_STAGE
    )
    assert (
        scratch[:, fault_segment.offset : fault_segment.end_offset]
        .view(torch.int32)
        .tolist()
        == [[0]] * cells
    )
    assert bool((scratch[:, : fault_segment.offset] == 0xA5).all())
    assert bool((scratch[:, fault_segment.end_offset :] == 0xA5).all())


def _identified_projection(
    field: str,
    payload: dict[str, object],
    *,
    cell_id: str,
    generation: int = 4,
) -> dict[str, object]:
    base: dict[str, object] = {
        "metadata": {"generation": generation, "tick": 0, "seed": 101},
        "tensors": {"sentinel": torch.tensor([generation])},
    }
    raw = dict(payload)
    raw["identity"] = {
        "schema_version": INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
        "cell_id": cell_id,
        "seed": 101,
        "generation": generation,
        "tick": 0,
        "base_projection_sha256": projection_digest(base),
        "payload_sha256": inheritance_sidecar_payload_sha256_v1(payload),
    }
    return {**base, field: raw}


def test_genome_v2_multicell_checkpoint_sidecars_restore_by_cell_identity() -> None:
    first = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=3, device="cpu")
    second = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=3, device="cpu")
    first.alleles_q10[0, 0, 0, 0, 1] = 111
    second.alleles_q10[0, 0, 0, 0, 1] = 222
    first.phenotypes_q10[0, 4, 2] = 333
    second.phenotypes_q10[0, 4, 2] = 444

    restored = allocator._genome_v2_bank_from_canonical_projections_v1(
        (
            _identified_projection(
                "genome_upgrade_v2_checkpoint_projection",
                second.checkpoint_projection(),
                cell_id="second",
            ),
            _identified_projection(
                "genome_upgrade_v2_checkpoint_projection",
                first.checkpoint_projection(),
                cell_id="first",
            ),
        ),
        device=torch.device("cpu"),
        cell_ids=("first", "second"),
    )

    assert restored is not None
    assert (restored.cells, restored.capacity) == (2, 3)
    assert restored.alleles_q10[:, 0, 0, 0, 1].tolist() == [111, 222]
    assert restored.phenotypes_q10[:, 4, 2].tolist() == [333, 444]


def test_genome_v2_checkpoint_rejects_mixed_founder_profile_authority() -> None:
    expected = resolve_genome_v2_founder_profile_v1("pleiotropic_tradeoffs")
    wrong = resolve_genome_v2_founder_profile_v1("neutral")
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=3, device="cpu")
    payload = {
        **bank.checkpoint_projection(),
        "founder_profile": {
            "schema_version": GENOME_V2_FOUNDER_PROFILE_SCHEMA,
            "profile_id": wrong["profile_id"],
            "payload_sha256": wrong["payload_sha256"],
        },
    }
    projection = _identified_projection(
        "genome_upgrade_v2_checkpoint_projection",
        payload,
        cell_id="first",
    )

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="founder profile authority mismatch",
    ):
        allocator._genome_v2_bank_from_canonical_projections_v1(
            (projection,),
            device=torch.device("cpu"),
            cell_ids=("first",),
            founder_profile=expected,
        )


def test_per_cell_configuration_diff_is_limited_to_founder_authority() -> None:
    base = CudaGenerationConfig(
        steps=80,
        patch_enabled=True,
        hidden_size=16,
    )
    treatment = replace(
        base,
        genome_upgrade_v2_founder_profile_id="pleiotropic_tradeoffs",
        genome_upgrade_v2_founder_profile_sha256="a" * 64,
    )
    assert allocator._same_configuration_except_founder_profile_v1(base, treatment)
    assert allocator._same_configuration_except_founder_profile_v1(
        base,
        replace(
            treatment,
            move_cost=3.2,
        ),
    )
    assert allocator._same_configuration_except_founder_profile_v1(
        base,
        replace(
            treatment,
            patch_movement_cost_energy_scale=16.0,
        ),
    )
    assert allocator._same_configuration_except_founder_profile_v1(
        base,
        replace(
            treatment,
            reproduce_try_prob=0.25,
        ),
    )
    assert not allocator._same_configuration_except_founder_profile_v1(
        base,
        replace(
            treatment,
            gather_cost=3.2,
        ),
    )
    assert allocator._same_configuration_except_founder_profile_v1(
        base,
        replace(
            treatment,
            diploid_mutation_enable=False,
            recomb72_mutation_off=True,
            genome_upgrade_v2_mutation_rate_q10=0,
            genome_upgrade_v2_mutation_step_q10=0,
        ),
    )
    assert not allocator._same_configuration_except_founder_profile_v1(
        base,
        replace(treatment, max_births_per_tick=2),
    )


def test_b2_checkpoint_profiles_restore_in_cell_identity_order() -> None:
    profiles = (
        resolve_genome_v2_founder_profile_v1("neutral"),
        resolve_genome_v2_founder_profile_v1("pleiotropic_tradeoffs"),
    )
    projections = []
    for cell, (cell_id, profile) in enumerate(
        zip(("neutral-cell", "treatment-cell"), profiles, strict=True)
    ):
        bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=2, device="cpu")
        bank.initialize_founders_by_cell_v1(
            (tuple(profile["genotype_q10"]),),
            alive_mask=torch.tensor([[True, False]], dtype=torch.bool),
        )
        payload = {
            **bank.checkpoint_projection(),
            "founder_profile": {
                "schema_version": GENOME_V2_FOUNDER_PROFILE_SCHEMA,
                "profile_id": profile["profile_id"],
                "payload_sha256": profile["payload_sha256"],
            },
        }
        projections.append(
            _identified_projection(
                "genome_upgrade_v2_checkpoint_projection",
                payload,
                cell_id=cell_id,
            )
        )

    restored = allocator._genome_v2_bank_from_canonical_projections_v1(
        tuple(reversed(projections)),
        device=torch.device("cpu"),
        cell_ids=("neutral-cell", "treatment-cell"),
        founder_profiles=profiles,
    )
    assert restored is not None
    assert restored.genotype(0, 0) == GenomeUpgradeV2V1.neutral()
    assert restored.genotype(1, 0) != GenomeUpgradeV2V1.neutral()

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="founder profile authority mismatch",
    ):
        allocator._genome_v2_bank_from_canonical_projections_v1(
            tuple(reversed(projections)),
            device=torch.device("cpu"),
            cell_ids=("neutral-cell", "treatment-cell"),
            founder_profiles=tuple(reversed(profiles)),
        )


@pytest.mark.parametrize("cells", [1, 4, 72])
def test_genome_v2_checkpoint_sidecars_repeat_fresh_owner_restore(
    cells: int,
) -> None:
    projections = []
    for cell in range(cells):
        bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=3, device="cpu")
        bank.alleles_q10[0, 0, 0, 0, 1] = 1000 + cell
        projections.append(
            _identified_projection(
                "genome_upgrade_v2_checkpoint_projection",
                bank.checkpoint_projection(),
                cell_id=f"cell-{cell:03d}",
            )
        )
    cell_ids = tuple(f"cell-{cell:03d}" for cell in range(cells))
    reordered = tuple(reversed(projections))
    first = allocator._genome_v2_bank_from_canonical_projections_v1(
        reordered,
        device=torch.device("cpu"),
        cell_ids=cell_ids,
    )
    assert first is not None
    second_projections = tuple(
        _identified_projection(
            "genome_upgrade_v2_checkpoint_projection",
            CudaGenomeUpgradeV2StateBank(
                **{
                    {
                        "genome_v2_alleles_q10": "alleles_q10",
                        "genome_v2_modifiers_q10": "modifiers_q10",
                        "genome_v2_products_q10": "products_q10",
                        "genome_v2_phenotypes_q10": "phenotypes_q10",
                        "genome_v2_expression_cost_q10": "expression_cost_q10",
                        "genome_v2_active_product_count": "active_product_count",
                        "genome_v2_saturated_product_count": (
                            "saturated_product_count"
                        ),
                    }[name]: tensor[cell : cell + 1].clone()
                    for name, tensor in first.named_tensors().items()
                }
            ).checkpoint_projection(),
            cell_id=cell_ids[cell],
        )
        for cell in range(cells)
    )
    second = allocator._genome_v2_bank_from_canonical_projections_v1(
        tuple(reversed(second_projections)),
        device=torch.device("cpu"),
        cell_ids=cell_ids,
    )
    assert second is not None
    for name, tensor in first.named_tensors().items():
        assert torch.equal(tensor, second.named_tensors()[name])


@pytest.mark.parametrize("mode", ["duplicate", "extra"])
def test_genome_v2_checkpoint_sidecars_reject_nonexact_identity_set(
    mode: str,
) -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=2, device="cpu")
    first = _identified_projection(
        "genome_upgrade_v2_checkpoint_projection",
        bank.checkpoint_projection(),
        cell_id="first",
    )
    second = _identified_projection(
        "genome_upgrade_v2_checkpoint_projection",
        bank.checkpoint_projection(),
        cell_id="first" if mode == "duplicate" else "extra",
    )
    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="duplicate|identity set differs",
    ):
        allocator._genome_v2_bank_from_canonical_projections_v1(
            (first, second),
            device=torch.device("cpu"),
            cell_ids=("first", "second"),
        )


def test_genome_v2_multicell_checkpoint_sidecars_reject_mixed_presence() -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=3, device="cpu")

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="sidecars differ across cells",
    ):
        allocator._genome_v2_bank_from_canonical_projections_v1(
            (
                {
                    "genome_upgrade_v2_checkpoint_projection": (
                        bank.checkpoint_projection()
                    )
                },
                {},
            ),
            device=torch.device("cpu"),
            cell_ids=("first", "second"),
        )


@pytest.mark.parametrize("corruption", ["cell_id", "generation", "payload"])
def test_genome_v2_checkpoint_sidecar_rejects_identity_corruption(
    corruption: str,
) -> None:
    bank = CudaGenomeUpgradeV2StateBank.allocate(cells=1, capacity=3, device="cpu")
    projection = _identified_projection(
        "genome_upgrade_v2_checkpoint_projection",
        bank.checkpoint_projection(),
        cell_id="first",
    )
    sidecar = projection["genome_upgrade_v2_checkpoint_projection"]
    assert isinstance(sidecar, dict)
    if corruption == "cell_id":
        sidecar["identity"]["cell_id"] = "wrong"
    elif corruption == "generation":
        projection["metadata"]["generation"] = 5
    else:
        sidecar["tensors"]["genome_v2_products_q10"][0][0][0] = 99

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="identity mismatch",
    ):
        allocator._genome_v2_bank_from_canonical_projections_v1(
            (projection,),
            device=torch.device("cpu"),
            cell_ids=("first",),
        )


def test_diploid_multicell_checkpoint_sidecars_restore_in_cell_order() -> None:
    def bank(marker: float) -> CudaDiploidStateBankV1:
        storage = torch.zeros((1, ALLELE_PLANE_COUNT, 3), dtype=torch.float32)
        storage[0, 0, 1] = marker
        template = torch.arange(5, dtype=torch.uint8).reshape(1, 5)
        return CudaDiploidStateBankV1(
            storage=storage,
            genome_resource_handling_axis=torch.full(
                (1, 3), marker, dtype=torch.float32
            ),
            canonical_hereditary_digest=torch.zeros(
                (1, 3, CANONICAL_HEREDITARY_DIGEST_BYTES),
                dtype=torch.uint8,
            ),
            canonical_hereditary_template=template.clone(),
            diploid_payload_template=template.clone(),
            diploid_genotype_template=template.clone(),
            metadata=checkpoint_metadata_v1(cells=1, capacity=3),
        )

    restored = allocator._diploid_bank_from_canonical_projections_v1(
        (
            _identified_projection(
                "diploid_checkpoint_projection",
                bank(1.25).checkpoint_projection(),
                cell_id="first",
            ),
            _identified_projection(
                "diploid_checkpoint_projection",
                bank(2.5).checkpoint_projection(),
                cell_id="second",
            ),
        ),
        device=torch.device("cpu"),
        cell_ids=("first", "second"),
    )

    assert restored is not None
    assert (restored.cells, restored.capacity) == (2, 3)
    assert restored.storage[:, 0, 1].tolist() == [1.25, 2.5]
    assert restored.genome_resource_handling_axis[:, 0].tolist() == [1.25, 2.5]


def test_fresh_diploid_sources_retain_exact_alleles_and_resource_axis() -> None:
    template = torch.arange(5, dtype=torch.uint8)

    def source(marker: float) -> dict[str, torch.Tensor]:
        storage = torch.zeros(
            (ALLELE_PLANE_COUNT, 3),
            dtype=torch.float32,
        )
        storage[0, 1] = marker
        return {
            "energy": torch.ones((3,), dtype=torch.float32),
            "diploid_allele_storage": storage,
            "genome_resource_handling_axis": torch.full(
                (3,), marker, dtype=torch.float32
            ),
            "canonical_hereditary_digest": torch.zeros(
                (3, CANONICAL_HEREDITARY_DIGEST_BYTES),
                dtype=torch.uint8,
            ),
            "canonical_hereditary_template": template.clone(),
            "diploid_payload_template": template.clone(),
            "diploid_genotype_template": template.clone(),
        }

    sources = (source(1.25), source(-2.5))
    restored = allocator._diploid_bank_from_fresh_sources_v1(
        sources[0],
        sources=sources,
        device=torch.device("cpu"),
        batch_cells=2,
        source_groups=None,
    )

    assert restored is not None
    assert (restored.cells, restored.capacity) == (2, 3)
    assert restored.storage[:, 0, 1].tolist() == [1.25, -2.5]
    assert restored.genome_resource_handling_axis[:, 0].tolist() == [
        1.25,
        -2.5,
    ]


def test_rolling_output_allocator_keeps_only_one_tick_and_one_g2_window() -> None:
    outputs = allocator._allocate_tick_outputs(
        capacity=64,
        observation_size=74,
        action_count=7,
        device=torch.device("cpu"),
    )

    assert tuple(outputs["observations"].shape) == (1, 1, 64, 74)
    assert tuple(outputs["runtime_bookkeeping_by_tick"].shape) == (1, 320, 5)
    assert tuple(outputs["generation_summary_scalar"].shape) == (
        2,
        1,
        allocator.GENERATION_SUMMARY_WIDTH_V1,
    )
    assert tuple(outputs["first_failure_records"].shape) == (1, 10)


def test_checkpoint_schema_template_drops_resident_tensor_payloads() -> None:
    source = _CheckpointTemplateFixture(
        state={"weights": torch.ones((384, 9415), dtype=torch.float32)},
        columns=(torch.ones((65536,), dtype=torch.int64),),
        schema_version="fixture_v1",
    )

    template = allocator._checkpoint_schema_template_v1(source)

    assert template is not source
    assert template.schema_version == source.schema_version
    assert set(template.state) == {"weights"}
    assert template.state["weights"].device.type == "cpu"
    assert template.state["weights"].dtype == torch.float32
    assert tuple(template.state["weights"].shape) == (0,)
    assert template.columns[0].dtype == torch.int64
    assert tuple(template.columns[0].shape) == (0,)


def test_fixed_mlp_stack_constructs_only_the_final_observation_layout() -> None:
    source = torch.arange(34, dtype=torch.float32).reshape(2, 17)

    final = allocator._stack_fixed_mlp_final_layout_v1(
        source,
        device=torch.device("cpu"),
        batch_cells=3,
        base_observation_size=3,
        observation_size=4,
        hidden_size=2,
        action_count=1,
    )

    assert tuple(final.shape) == (3, 2, 19)
    assert torch.equal(final[0], final[1])
    assert torch.equal(final[1], final[2])
    expected_prefix = torch.zeros((2, 2, 4), dtype=torch.float32)
    expected_prefix[..., :3] = source[..., :6].reshape(2, 2, 3)
    assert torch.equal(final[0, ..., :8].reshape(2, 2, 4), expected_prefix)
    assert torch.equal(final[0, ..., 8:], source[..., 6:])


def test_rolling_output_allocator_supports_explicit_physical_b3() -> None:
    outputs = allocator._allocate_tick_outputs(
        capacity=64,
        observation_size=86,
        action_count=7,
        device=torch.device("cpu"),
        batch_cells=3,
    )
    births = allocator._allocate_birth_outputs(
        capacity=64,
        max_births_per_tick=1,
        device=torch.device("cpu"),
        batch_cells=3,
    )
    turnover = allocator._allocate_turnover_outputs(
        capacity=64,
        generation=0,
        device=torch.device("cpu"),
        batch_cells=3,
    )

    assert tuple(outputs["observations"].shape) == (3, 1, 64, 86)
    assert tuple(outputs["runtime_bookkeeping_by_tick"].shape) == (3, 320, 5)
    assert tuple(outputs["generation_summary_scalar"].shape) == (
        2,
        3,
        allocator.GENERATION_SUMMARY_WIDTH_V1,
    )
    assert tuple(births["birth_decision_parent_slots"].shape) == (3, 1, 64)
    assert tuple(turnover["turnover_counts"].shape) == (3, 8)


def test_birth_and_turnover_outputs_match_the_exact_argument_order() -> None:
    births = allocator._allocate_birth_outputs(
        capacity=64,
        max_births_per_tick=1,
        device=torch.device("cpu"),
        genome_v2_causal_record_capacity=RECORD_CAPACITY,
    )
    turnover = allocator._allocate_turnover_outputs(
        capacity=64,
        generation=0,
        device=torch.device("cpu"),
        genome_v2_causal_record_capacity=RECORD_CAPACITY,
    )

    assert list(births[name] for name in allocator.ORDINARY_BIRTH_OUTPUT_FIELDS)
    assert list(turnover[name] for name in allocator.TURNOVER_OUTPUT_FIELDS)
    assert tuple(births["birth_decision_parent_slots"].shape) == (1, 1, 64)
    assert tuple(turnover["turnover_counts"].shape) == (1, 8)


def test_genome_v2_causal_outputs_are_bounded_and_default_disabled() -> None:
    births = allocator._allocate_birth_outputs(
        capacity=256,
        max_births_per_tick=2,
        device=torch.device("cpu"),
        batch_cells=72,
    )
    turnover = allocator._allocate_turnover_outputs(
        capacity=256,
        generation=0,
        device=torch.device("cpu"),
        batch_cells=72,
    )
    causal_names = tuple(
        name for name in (*births, *turnover) if "genome_v2_causal" in name
    )
    assert allocator.GENOME_V2_CAUSAL_RECORD_WIDTH_V1 == 182
    assert (
        allocator.GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1
        == RECORD_CAPACITY
        == 1024
    )
    assert allocator.GENOME_V2_CAUSAL_RECORD_WIDTH_V1 == RECORD_WIDTH
    assert len(causal_names) == 10
    assert tuple(births["birth_genome_v2_causal_record_rows"].shape) == (72, 0, 182)
    assert tuple(turnover["turnover_genome_v2_causal_record_rows"].shape) == (
        72,
        0,
        182,
    )
    assert not births["birth_genome_v2_causal_capture_enabled"].any()
    assert not turnover["turnover_genome_v2_causal_capture_enabled"].any()
    for prefix, outputs in (("birth", births), ("turnover", turnover)):
        assert not outputs[f"{prefix}_genome_v2_causal_record_valid"].any()
        assert not outputs[f"{prefix}_genome_v2_causal_record_count"].any()
        assert not outputs[f"{prefix}_genome_v2_causal_record_dropped_count"].any()
    total_bytes = sum(
        (
            births[name].numel() * births[name].element_size()
            if name in births
            else turnover[name].numel() * turnover[name].element_size()
        )
        for name in causal_names
    )
    expected_bytes = 72 * 2 * (2 * torch.int64.itemsize + torch.bool.itemsize)
    assert total_bytes == expected_bytes == 2_448

    enabled_births = allocator._allocate_birth_outputs(
        capacity=256,
        max_births_per_tick=2,
        device=torch.device("cpu"),
        batch_cells=72,
        genome_v2_causal_record_capacity=RECORD_CAPACITY,
    )
    enabled_turnover = allocator._allocate_turnover_outputs(
        capacity=256,
        generation=0,
        device=torch.device("cpu"),
        batch_cells=72,
        genome_v2_causal_record_capacity=RECORD_CAPACITY,
    )
    assert tuple(enabled_births["birth_genome_v2_causal_record_rows"].shape) == (
        72,
        RECORD_CAPACITY,
        RECORD_WIDTH,
    )
    assert tuple(enabled_turnover["turnover_genome_v2_causal_record_rows"].shape) == (
        72,
        RECORD_CAPACITY,
        RECORD_WIDTH,
    )


def test_genome_v2_causal_capture_is_enabled_by_checkpoint_authority() -> None:
    births = allocator._allocate_birth_outputs(
        capacity=64,
        max_births_per_tick=1,
        device=torch.device("cpu"),
    )
    turnover = allocator._allocate_turnover_outputs(
        capacity=64,
        generation=0,
        device=torch.device("cpu"),
    )
    persistent = {**births, **turnover}

    allocator._enable_authority_bound_genome_v2_causal_capture_v1(
        persistent,
        checkpoint_authority=(("schema_version", "test"),),
    )

    assert births["birth_genome_v2_causal_capture_enabled"].tolist() == [True]
    assert turnover["turnover_genome_v2_causal_capture_enabled"].tolist() == [True]
    assert births["birth_genome_v2_causal_record_count"].tolist() == [0]
    assert turnover["turnover_genome_v2_causal_record_count"].tolist() == [0]


def test_generation_zero_canonical_projection_keeps_legacy_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_with_checkpoint_contract_metadata",
        lambda projection: projection,
    )
    projection = _boundary_projection(0, include_runtime_prefix=False)

    generation = allocator._validated_source_generation(
        projection,
        SimpleNamespace(generation=0),
    )
    canonical = allocator._canonical_projection(
        projection,
        source_generation=generation,
    )

    assert generation == 0
    assert canonical["generation_calls"] == 0
    assert canonical["restore_count"] == 0
    assert canonical["event_cursor"] == 3
    assert canonical["runtime_event_cursor"] == 3
    assert canonical["last_world"] == "none"
    assert canonical["pending_generation_free_agent_ids"] == []
    assert canonical["pending_generation_free_slots"] == []


@pytest.mark.parametrize("source_generation", (4, 8))
def test_resumed_generation_boundary_preserves_canonical_runtime_prefix(
    monkeypatch: pytest.MonkeyPatch,
    source_generation: int,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_with_checkpoint_contract_metadata",
        lambda projection: projection,
    )
    projection = _boundary_projection(
        source_generation,
        include_runtime_prefix=True,
    )

    generation = allocator._validated_source_generation(
        projection,
        SimpleNamespace(generation=source_generation),
    )
    canonical = allocator._canonical_projection(
        projection,
        source_generation=generation,
    )
    turnover = allocator._allocate_turnover_outputs(
        capacity=64,
        generation=generation,
        device=torch.device("cpu"),
    )

    assert generation == source_generation
    assert int(turnover["turnover_next_generation"].item()) == source_generation
    assert int(turnover["turnover_next_tick"].item()) == 80
    for name in allocator.CANONICAL_RUNTIME_PREFIX_FIELDS:
        expected = getattr(projection, name)
        if isinstance(expected, tuple):
            expected = list(expected)
        assert canonical[name] == expected


def test_resumed_canonical_projection_preserves_inheritance_sidecars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_with_checkpoint_contract_metadata",
        lambda projection: projection,
    )
    projection = vars(_boundary_projection(4, include_runtime_prefix=True))
    projection["metadata"]["seed"] = 101

    template = torch.arange(5, dtype=torch.uint8).reshape(1, 5)
    diploid_bank = CudaDiploidStateBankV1(
        storage=torch.zeros((1, ALLELE_PLANE_COUNT, 3), dtype=torch.float32),
        genome_resource_handling_axis=torch.full((1, 3), 1.25, dtype=torch.float32),
        canonical_hereditary_digest=torch.zeros(
            (1, 3, CANONICAL_HEREDITARY_DIGEST_BYTES),
            dtype=torch.uint8,
        ),
        canonical_hereditary_template=template.clone(),
        diploid_payload_template=template.clone(),
        diploid_genotype_template=template.clone(),
        metadata=checkpoint_metadata_v1(cells=1, capacity=3),
    )
    diploid_bank.storage[0, 0, 1] = 2.5
    genome_v2_bank = CudaGenomeUpgradeV2StateBank.allocate(
        cells=1, capacity=3, device="cpu"
    )
    genome_v2_bank.alleles_q10[0, 0, 0, 0, 1] = 777
    genome_v2_bank.expression_cost_q10[0, 1] = 19

    base_digest = projection_digest(projection)
    for field, payload in (
        ("diploid_checkpoint_projection", diploid_bank.checkpoint_projection()),
        (
            "genome_upgrade_v2_checkpoint_projection",
            genome_v2_bank.checkpoint_projection(),
        ),
    ):
        sidecar = dict(payload)
        sidecar["identity"] = {
            "schema_version": INHERITANCE_SIDECAR_IDENTITY_SCHEMA_V1,
            "cell_id": "cell-000",
            "seed": 101,
            "generation": 4,
            "tick": 0,
            "base_projection_sha256": base_digest,
            "payload_sha256": inheritance_sidecar_payload_sha256_v1(payload),
        }
        projection[field] = sidecar

    canonical = allocator._canonical_projection(
        projection,
        source_generation=4,
    )
    restored_diploid = allocator._diploid_bank_from_canonical_projections_v1(
        (canonical,),
        device=torch.device("cpu"),
        cell_ids=("cell-000",),
    )
    restored_genome_v2 = allocator._genome_v2_bank_from_canonical_projections_v1(
        (canonical,),
        device=torch.device("cpu"),
        cell_ids=("cell-000",),
    )

    assert restored_diploid is not None
    assert restored_genome_v2 is not None
    assert restored_diploid.storage[0, 0, 1].item() == 2.5
    assert restored_diploid.genome_resource_handling_axis[0, 0].item() == 1.25
    assert restored_genome_v2.alleles_q10[0, 0, 0, 0, 1].item() == 777
    assert restored_genome_v2.expression_cost_q10[0, 1].item() == 19
    assert (
        canonical["diploid_checkpoint_projection"]
        is not projection["diploid_checkpoint_projection"]
    )
    assert (
        canonical["genome_upgrade_v2_checkpoint_projection"]
        is not projection["genome_upgrade_v2_checkpoint_projection"]
    )


def test_source_allocator_cardinality_contract_allows_resident_growth_or_extinction() -> (
    None
):
    assert allocator._source_cardinality_is_qualified_v1(
        source_generation=0,
        source_tick=0,
        expected_population=256,
        capacity=512,
        initial_agent_counts=(256,),
    )
    assert not allocator._source_cardinality_is_qualified_v1(
        source_generation=0,
        source_tick=0,
        expected_population=256,
        capacity=512,
        initial_agent_counts=(255,),
    )
    assert allocator._source_cardinality_is_qualified_v1(
        source_generation=0,
        source_tick=64,
        expected_population=256,
        capacity=512,
        initial_agent_counts=(262,),
    )
    assert allocator._source_cardinality_is_qualified_v1(
        source_generation=1,
        source_tick=0,
        expected_population=256,
        capacity=512,
        initial_agent_counts=(255,),
    )
    assert not allocator._source_cardinality_is_qualified_v1(
        source_generation=0,
        source_tick=64,
        expected_population=256,
        capacity=512,
        initial_agent_counts=(513,),
    )
    source = Path(allocator.__file__).read_text(encoding="utf-8")
    assert "resumed boundary within resident capacity" in source


def test_source_allocator_reserves_fixed_causal_record_abi() -> None:
    source = Path(allocator.__file__).read_text(encoding="utf-8")

    assert (
        "genome_v2_causal_record_capacity = "
        "GENOME_V2_CAUSAL_RECORD_CAPACITY_PER_CELL_V1"
    ) in source
    assert "genome_v2_causal_capture_enabled" in source


def test_resumed_window_tapes_are_anchored_at_absolute_rng_counters() -> None:
    tree = ast.parse(Path(allocator.__file__).read_text(encoding="utf-8"))
    assignments = {
        ast.unparse(node.targets[0]): ast.unparse(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
    }

    assert assignments["batched['python_random_raw_draw_base']"] == (
        "batched['python_random_raw_draws'][:, 0].clone()"
    )
    assert assignments["batched['policy_birth_tape_cursor_base']"] == (
        "batched['policy_birth_tape_cursor'][:, 0].clone()"
    )


def test_resumed_absolute_gaussian_tape_indexes_past_zero_padding() -> None:
    source = random.Random(23)
    state = source.getstate()
    expected = source.gauss(0.0, 1.0)
    packed = pack_cpython_random_state_cuda(
        state,
        device="cpu",
        raw_draws=37,
        gauss_tape_raw_capacity=8,
    )

    base = torch.zeros_like(packed["python_random_raw_draws"])
    physical_cursor = int((packed["python_random_raw_draws"] - base)[0].item())

    assert physical_cursor == 37
    assert float(packed["python_random_gauss_primary_tape"][physical_cursor]) == (
        expected
    )


def test_resumed_local_gaussian_tape_indexes_from_checkpoint_base() -> None:
    source = random.Random(23)
    state = source.getstate()
    expected = source.gauss(0.0, 1.0)
    packed = pack_cpython_random_state_cuda(
        state,
        device="cpu",
        raw_draws=37,
        raw_draw_base=37,
        gauss_tape_raw_capacity=8,
    )

    base = packed["python_random_raw_draws"].clone()
    physical_cursor = int((packed["python_random_raw_draws"] - base)[0].item())

    assert packed["python_random_gauss_primary_tape"].shape == (8,)
    assert physical_cursor == 0
    assert float(packed["python_random_gauss_primary_tape"][physical_cursor]) == (
        expected
    )


def test_typed_lifecycle_next_sequence_is_per_cell_private_metadata() -> None:
    left = {"world": "patch", "typed_lifecycle_event_next_sequence": 1201}
    right = {"world": "patch", "typed_lifecycle_event_next_sequence": 1382}

    assert allocator._shared_projection_metadata_v1(left) == {"world": "patch"}
    assert allocator._shared_projection_metadata_v1(right) == {"world": "patch"}


def test_external_event_history_authority_is_per_cell_private_metadata() -> None:
    left = {
        "world": "patch",
        "event_history_frontier": 4,
        "typed_lifecycle_event_sequence_base": 4,
        "event_history_reference": {"cell_index": 0, "path": "/run/events.bin"},
    }
    right = {
        "world": "patch",
        "event_history_frontier": 7,
        "typed_lifecycle_event_sequence_base": 7,
        "event_history_reference": {"cell_index": 1, "path": "/run/events.bin"},
    }

    assert allocator._shared_projection_metadata_v1(left) == {"world": "patch"}
    assert allocator._shared_projection_metadata_v1(right) == {"world": "patch"}


def test_move_cost_is_per_cell_private_metadata() -> None:
    left = {"world": "patch", "move_cost": 0.2}
    right = {"world": "patch", "move_cost": 3.2}

    assert allocator._shared_projection_metadata_v1(left) == {"world": "patch"}
    assert allocator._shared_projection_metadata_v1(right) == {"world": "patch"}


def test_patch_movement_cost_energy_scale_is_per_cell_private_metadata() -> None:
    key = "patch_movement_cost_energy_scale"
    left = {"world": "patch", key: 0.0}
    right = {"world": "patch", key: 16.0}

    assert allocator._shared_projection_metadata_v1(left) == {"world": "patch"}
    assert allocator._shared_projection_metadata_v1(right) == {"world": "patch"}


def test_wdl_radius_is_per_cell_private_metadata() -> None:
    key = "wdl_delayed_hazard_radius"
    left = {"world": "patch", key: 0}
    right = {"world": "patch", key: 3}

    assert allocator._shared_projection_metadata_v1(left) == {"world": "patch"}
    assert allocator._shared_projection_metadata_v1(right) == {"world": "patch"}


def test_wdl_cued_agent_summary_is_per_cell_private_metadata() -> None:
    key = "wdl_agents_cued_current_generation_agent_ids"
    left = {"world": "patch", key: [46, 141, 156]}
    right = {"world": "patch", key: [83, 145, 173, 195]}

    assert allocator._shared_projection_metadata_v1(left) == {"world": "patch"}
    assert allocator._shared_projection_metadata_v1(right) == {"world": "patch"}


def test_reference_materialization_attestation_is_per_cell_private_metadata() -> None:
    key = "native_runtime_reference_genome_materialization"
    left = {"world": "patch", key: {"supported": True}}
    right = {
        "world": "patch",
        key: {"supported": False, "reasons": ["non_scalar_canonical_payload"]},
    }

    assert allocator._shared_projection_metadata_v1(left) == {"world": "patch"}
    assert allocator._shared_projection_metadata_v1(right) == {"world": "patch"}


def test_multicell_retained_identity_capacity_must_be_exactly_common() -> None:
    key = allocator.RESIDENT_AGENT_ID_CAPACITY_METADATA_V1
    projections = tuple(
        SimpleNamespace(metadata={key: capacity}) for capacity in (1793, 1793)
    )
    assert allocator._common_retained_agent_id_capacity_v1(projections) == 1793

    projections[1].metadata[key] = 2049
    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="retained agent ID capacity mismatch",
    ):
        allocator._common_retained_agent_id_capacity_v1(projections)


def test_per_cell_projection_accepts_ragged_private_event_histories() -> None:
    projections = tuple(
        SimpleNamespace(
            schema_version="checkpoint-v1",
            metadata={
                "world": "patch",
                "native_runtime_fixed_policy_family": "gru",
                "capability_mask": {"gru": True},
            },
            tensors={
                "alive_mask": torch.ones(4, dtype=torch.bool),
                "policy_use_gru": torch.ones(4, dtype=torch.bool),
                "event_tape_actor_id": torch.arange(rows, dtype=torch.int64),
            },
        )
        for rows in (3, 7)
    )

    validated = allocator._validated_per_cell_projection_tensors_v1(
        projections,
        batch_cells=2,
    )

    assert tuple(tensor["event_tape_actor_id"].numel() for tensor in validated) == (
        3,
        7,
    )


def test_per_cell_projection_accepts_divergent_native_cumulative_counters() -> None:
    projections = tuple(
        SimpleNamespace(
            schema_version="checkpoint-v1",
            metadata={
                "world": "patch",
                "native_runtime_fixed_policy_family": "mlp",
                "capability_mask": {"gru": False},
                "native_cumulative_attempt_count": attempts,
                "native_cumulative_success_count": successes,
                "native_cumulative_failure_count": attempts - successes,
            },
            tensors={
                "alive_mask": torch.ones(4, dtype=torch.bool),
                "policy_use_gru": torch.zeros(4, dtype=torch.bool),
                "energy": torch.arange(4, dtype=torch.float32),
            },
        )
        for attempts, successes in ((11, 1), (20, 10))
    )

    validated = allocator._validated_per_cell_projection_tensors_v1(
        projections,
        batch_cells=2,
    )

    assert tuple(tensor["energy"].shape for tensor in validated) == ((4,), (4,))


def test_per_cell_projection_still_rejects_ragged_non_event_state() -> None:
    projections = tuple(
        SimpleNamespace(
            schema_version="checkpoint-v1",
            metadata={
                "world": "patch",
                "native_runtime_fixed_policy_family": "gru",
                "capability_mask": {"gru": True},
            },
            tensors={
                "alive_mask": torch.ones(rows, dtype=torch.bool),
                "policy_use_gru": torch.ones(rows, dtype=torch.bool),
            },
        )
        for rows in (3, 7)
    )

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="shape/dtype/layout mismatch",
    ):
        allocator._validated_per_cell_projection_tensors_v1(
            projections,
            batch_cells=2,
        )


def test_per_cell_projection_accepts_ragged_agent_id_maps() -> None:
    projections = tuple(
        SimpleNamespace(
            schema_version="checkpoint-v1",
            metadata={
                "world": "patch",
                "native_runtime_fixed_policy_family": "mlp",
                "capability_mask": {"gru": False},
            },
            tensors={
                "alive_mask": torch.ones(4, dtype=torch.bool),
                "policy_use_gru": torch.zeros(4, dtype=torch.bool),
                "agent_id_to_slot": torch.arange(rows, dtype=torch.int32),
            },
        )
        for rows in (235, 234)
    )

    validated = allocator._validated_per_cell_projection_tensors_v1(
        projections,
        batch_cells=2,
    )

    assert tuple(row["agent_id_to_slot"].numel() for row in validated) == (235, 234)


def test_per_cell_projection_accepts_aligned_ragged_active_identity_rows() -> None:
    active_rows = (
        (
            torch.tensor([0, 1, 2], dtype=torch.int64),
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([3], dtype=torch.int64),
            torch.tensor([True, True, True, False]),
        ),
        (
            torch.tensor([7], dtype=torch.int64),
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([-1, -1, -1, -1, -1, -1, -1, 1], dtype=torch.int32),
            torch.tensor([0, 2, 3], dtype=torch.int64),
            torch.tensor([False, True, False, False]),
        ),
    )
    projections = tuple(
        SimpleNamespace(
            schema_version="checkpoint-v1",
            metadata={
                "world": "patch",
                "native_runtime_fixed_policy_family": "mlp",
                "capability_mask": {"gru": False},
            },
            tensors={
                "alive_mask": alive,
                "policy_use_gru": torch.zeros(4, dtype=torch.bool),
                "agent_ids_by_alive_order": agent_ids,
                "agent_id_to_slot_agent_ids": agent_ids.clone(),
                "agent_id_to_slot_slots": slots,
                "agent_id_to_slot": dense,
                "slot_free_queue": free_slots,
            },
        )
        for agent_ids, slots, dense, free_slots, alive in active_rows
    )

    validated = allocator._validated_per_cell_projection_tensors_v1(
        projections,
        batch_cells=2,
    )
    stacked = allocator.stack_validated_per_cell_checkpoint_projections_v1(
        projections,
        batch_cells=2,
        device="cpu",
        field_names=tuple(allocator.RAGGED_PROJECTION_FIELDS_V1),
    )

    assert tuple(row["agent_ids_by_alive_order"].numel() for row in validated) == (3, 1)
    assert stacked["agent_ids_by_alive_order"].tolist() == [
        [0, 1, 2],
        [7, -1, -1],
    ]
    assert stacked["agent_id_to_slot_slots"].tolist() == [
        [0, 1, 2],
        [1, -1, -1],
    ]
    assert tuple(stacked["agent_id_to_slot"].shape) == (2, 8)
    assert stacked["slot_free_queue"].tolist() == [
        [3, -1, -1],
        [0, 2, 3],
    ]


def test_per_cell_projection_rejects_misaligned_compact_identity_rows() -> None:
    projections = tuple(
        SimpleNamespace(
            schema_version="checkpoint-v1",
            metadata={
                "world": "patch",
                "native_runtime_fixed_policy_family": "mlp",
                "capability_mask": {"gru": False},
            },
            tensors={
                "alive_mask": torch.tensor([True, True, False, False]),
                "policy_use_gru": torch.zeros(4, dtype=torch.bool),
                "agent_ids_by_alive_order": torch.tensor([0, 1], dtype=torch.int64),
                "agent_id_to_slot_agent_ids": torch.tensor([0], dtype=torch.int64),
                "agent_id_to_slot_slots": torch.tensor([0, 1], dtype=torch.int32),
            },
        )
        for _ in range(2)
    )

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="compact identity lengths mismatch",
    ):
        allocator._validated_per_cell_projection_tensors_v1(
            projections,
            batch_cells=2,
        )


def test_ragged_agent_id_maps_stack_to_private_minus_one_padded_rows() -> None:
    values = (
        torch.tensor([4, 7, 9], dtype=torch.int32),
        torch.tensor([2, 5], dtype=torch.int32),
    )

    stacked = allocator._stack_per_cell_agent_id_to_slot_v1(
        values, device=torch.device("cpu")
    )

    assert torch.equal(
        stacked,
        torch.tensor([[4, 7, 9], [2, 5, -1]], dtype=torch.int32),
    )


def test_ragged_active_ids_pad_to_capacity_and_preserve_counts() -> None:
    active_ids, active_count = allocator._stack_ragged_active_agent_ids_v1(
        (
            torch.tensor([10, 11, 12], dtype=torch.int64),
            torch.tensor([20], dtype=torch.int64),
        ),
        capacity=4,
        device=torch.device("cpu"),
    )

    assert active_ids.tolist() == [
        [10, 11, 12, -1],
        [20, -1, -1, -1],
    ]
    assert active_count.tolist() == [3, 1]
    assert active_count.dtype == torch.int64
    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="exceeds its admitted width",
    ):
        allocator._stack_ragged_active_agent_ids_v1(
            (torch.arange(5, dtype=torch.int64),),
            capacity=4,
            device=torch.device("cpu"),
        )


def test_initial_reporting_rows_use_capacity_bounded_sorted_ids() -> None:
    maximum_id = torch.iinfo(torch.int32).max
    initial_agent_ids_by_row, terminal_survival, terminal_rows = (
        allocator._allocate_initial_id_reporting_state_v1(
            (
                torch.tensor([10, 11, maximum_id], dtype=torch.int64),
                torch.tensor([20], dtype=torch.int64),
            ),
            reporting_capacity=3,
            device=torch.device("cpu"),
        )
    )

    assert initial_agent_ids_by_row.tolist() == [
        [10, 11, maximum_id],
        [20, torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).max],
    ]
    assert initial_agent_ids_by_row.numel() == 6
    assert tuple(terminal_survival.shape) == (2, 3)
    assert torch.equal(terminal_rows, torch.zeros((2, 3), dtype=torch.int32))


def test_initial_reporting_rows_can_use_a_bounded_streamed_override() -> None:
    initial_agent_ids_by_row, terminal_survival, terminal_rows = (
        allocator._allocate_initial_id_reporting_state_v1(
            (
                torch.tensor([10, 11], dtype=torch.int64),
                torch.tensor([20], dtype=torch.int64),
            ),
            reporting_capacity=8,
            reporting_rows_override=8,
            device=torch.device("cpu"),
        )
    )

    assert tuple(initial_agent_ids_by_row.shape) == (2, 8)
    assert initial_agent_ids_by_row[0, :2].tolist() == [10, 11]
    assert initial_agent_ids_by_row[1, 0].item() == 20
    assert initial_agent_ids_by_row[0, 2].item() == torch.iinfo(torch.int64).max
    assert torch.equal(terminal_survival, torch.zeros((2, 8), dtype=torch.int32))
    assert torch.equal(terminal_rows, torch.zeros((2, 8), dtype=torch.int32))


def test_legacy_dense_identity_byte_accounting_is_exact() -> None:
    assert allocator.LEGACY_DENSE_IDENTITY_BYTES_PER_CELL_ID_V1 == 12
    assert allocator.legacy_dense_identity_allocation_bytes_v1(1, 155_585) == (
        1_867_020
    )
    assert allocator.legacy_dense_identity_allocation_bytes_v1(72, 155_585) == (
        134_425_440
    )


def test_identity_width_is_resident_capacity_not_historical_envelope() -> None:
    compatibility_width = allocator.required_resident_agent_id_capacity_v1(
        65,
        torch.iinfo(torch.int32).max,
        64,
        256,
        allocation_generation_count=30,
        retained_capacity=65,
    )

    assert compatibility_width == 256


def test_reporting_width_fails_closed_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocation_attempted = False

    def unexpected_allocation(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal allocation_attempted
        allocation_attempted = True
        raise AssertionError("dense reporting allocation must not be attempted")

    monkeypatch.setattr(allocator.torch, "full", unexpected_allocation)
    initial_ids = (torch.tensor([7, torch.iinfo(torch.int32).max], dtype=torch.int64),)

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="initial reporting rows exceed resident capacity",
    ):
        allocator._allocate_initial_id_reporting_state_v1(
            initial_ids,
            reporting_capacity=1,
            device=torch.device("cpu"),
        )

    assert allocation_attempted is False


@pytest.mark.parametrize(
    ("metadata_update", "configuration_generation", "message"),
    (
        ({"generation": -1}, 0, "nonnegative exact integer"),
        ({"generation": True}, 0, "nonnegative exact integer"),
        ({"tick": 1}, 4, "exact tick 0"),
        ({"agent_grid_runner_phase": "partial_generation"}, 4, "generation boundary"),
        ({}, 5, "must equal"),
    ),
)
def test_source_generation_boundary_disagreement_fails_closed(
    metadata_update: dict[str, object],
    configuration_generation: int,
    message: str,
) -> None:
    projection = _boundary_projection(4, include_runtime_prefix=True)
    projection.metadata.update(metadata_update)

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match=message,
    ):
        allocator._validated_source_generation(
            projection,
            SimpleNamespace(generation=configuration_generation),
        )


def test_resumed_boundary_requires_complete_canonical_runtime_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        allocator,
        "_with_checkpoint_contract_metadata",
        lambda projection: projection,
    )
    projection = _boundary_projection(4, include_runtime_prefix=True)
    del projection.generation_calls

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="generation_calls",
    ):
        allocator._canonical_projection(projection, source_generation=4)


@pytest.mark.parametrize("max_births_per_tick", [1, 3, 64])
def test_birth_outputs_derive_all_birth_axes_and_asexual_sentinels(
    max_births_per_tick: int,
) -> None:
    births = allocator._allocate_birth_outputs(
        capacity=384,
        max_births_per_tick=max_births_per_tick,
        device=torch.device("cpu"),
    )

    for name in (
        "birth_planned_parent_slots",
        "birth_planned_child_slots",
        "birth_committed_parent_agent_ids",
        "birth_committed_child_agent_ids",
        "birth_committed_child_slots",
        "birth_policy_mutation_occurred",
        "birth_policy_mutation_delta",
        "birth_planned_mate_slots",
        "birth_committed_mate_agent_ids",
    ):
        assert births[name].shape[2] == max_births_per_tick
    assert tuple(births["birth_planned_spawn_positions"].shape) == (
        1,
        1,
        max_births_per_tick,
        2,
    )
    assert births["birth_planned_mate_slots"].eq(-1).all()
    assert births["birth_committed_mate_agent_ids"].eq(-1).all()


@pytest.mark.parametrize(
    ("max_births_per_tick", "message"),
    (
        (0, r"max_births_per_tick.*\[1, slot_capacity=384\]"),
        (385, r"max_births_per_tick.*\[1, slot_capacity=384\]"),
        (True, "must be an exact integer"),
        (1.0, "must be an exact integer"),
    ),
)
def test_birth_output_allocator_fails_closed_outside_exact_slot_bound(
    max_births_per_tick: object,
    message: str,
) -> None:
    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match=message,
    ):
        allocator._allocate_birth_outputs(
            capacity=384,
            max_births_per_tick=max_births_per_tick,  # type: ignore[arg-type]
            device=torch.device("cpu"),
        )


def test_physical_window_policy_capacity_is_g2_not_logical_g4() -> None:
    assert allocator.required_physical_window_policy_episodes_v1(1) == 444
    assert allocator.required_physical_window_policy_episodes_v1(3) == 1_084
    assert allocator.required_physical_window_policy_episodes_v1(64) == 20_604
    assert (
        allocator.required_physical_window_policy_episodes_v1(
            64,
            physical_window_generations=1,
        )
        == 10_302
    )


def test_turnover_only_physical_window_capacity_uses_slot_capacity() -> None:
    assert (
        allocator.required_physical_window_turnover_episodes_v1(
            256,
            ordinary_births_enabled=False,
        )
        == 256
    )
    assert (
        allocator.required_physical_window_turnover_episodes_v1(
            256,
            ordinary_births_enabled=True,
        )
        == 124
    )
    assert (
        allocator.required_physical_window_turnover_episodes_v1(
            256,
            ordinary_births_enabled=True,
            physical_window_generations=1,
        )
        == 62
    )


@pytest.mark.parametrize("max_births_per_tick", (1, 3, 64))
def test_agent_identity_width_is_independent_of_birth_bound(
    max_births_per_tick: int,
) -> None:
    assert (
        allocator.required_agent_id_capacity_v1(
            227,
            torch.iinfo(torch.int32).max,
            max_births_per_tick,
            384,
        )
        == 384
    )


def test_agent_identity_width_is_independent_of_resident_horizon() -> None:
    assert (
        allocator.required_agent_id_capacity_v1(
            227, torch.iinfo(torch.int32).max, 1, 384, 32
        )
        == 384
    )


@pytest.mark.parametrize("logical_generation_count", (3, 7, 14, 37))
def test_agent_identity_width_has_no_generation_envelope(
    logical_generation_count: int,
) -> None:
    assert (
        allocator.required_agent_id_capacity_v1(
            227,
            torch.iinfo(torch.int32).max,
            1,
            384,
            logical_generation_count,
        )
        == 384
    )


def test_ragged_agent_identity_rows_pad_to_the_largest_private_width() -> None:
    actual = allocator._stack_ragged_agent_id_to_slot_v1(
        (
            torch.tensor([0, 1, 2], dtype=torch.int32),
            torch.tensor([3, 4, 5, 6, 7], dtype=torch.int32),
        ),
        device=torch.device("cpu"),
    )

    assert actual.tolist() == [[0, 1, 2, -1, -1], [3, 4, 5, 6, 7]]


def test_ragged_agent_identity_alias_is_confined_to_one_source_group() -> None:
    shared = torch.tensor([0, 1, 2], dtype=torch.int32)

    accepted = allocator._stack_ragged_agent_id_to_slot_v1(
        (shared, shared),
        device=torch.device("cpu"),
        source_groups=(0, 0),
    )
    assert accepted.tolist() == [[0, 1, 2], [0, 1, 2]]

    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="storage alias",
    ):
        allocator._stack_ragged_agent_id_to_slot_v1(
            (shared, shared),
            device=torch.device("cpu"),
            source_groups=(0, 1),
        )


def test_checkpoint_identity_width_is_resident_capacity() -> None:
    assert (
        allocator.required_resident_agent_id_capacity_v1(
            227,
            torch.iinfo(torch.int32).max,
            1,
            384,
            allocation_generation_count=6,
            retained_capacity=227,
        )
        == 384
    )


def test_checkpoint_identity_capacity_rejects_unbounded_retained_shape() -> None:
    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="must not exceed resident slot capacity",
    ):
        allocator.required_resident_agent_id_capacity_v1(
            227,
            321,
            1,
            384,
            allocation_generation_count=6,
            retained_capacity=10**9,
        )


@pytest.mark.parametrize("allocation_generation_count", (0, 1, 129, 130, True))
def test_checkpoint_identity_capacity_rejects_invalid_remaining_horizon(
    allocation_generation_count: object,
) -> None:
    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="agent_id_allocation_generation_count",
    ):
        allocator.required_resident_agent_id_capacity_v1(
            227,
            321,
            1,
            384,
            allocation_generation_count=allocation_generation_count,
            retained_capacity=1793,
        )


def test_checkpoint_identity_capacity_accepts_parent_campaign_envelope() -> None:
    assert (
        allocator.required_resident_agent_id_capacity_v1(
            227,
            321,
            1,
            384,
            allocation_generation_count=14,
            retained_capacity=227,
        )
        == 384
    )


def test_public_resident_identity_width_is_capacity_bounded() -> None:
    assert (
        allocator.required_resident_agent_id_capacity_v1(
            65,
            65,
            64,
            256,
            allocation_generation_count=30,
            retained_capacity=65,
        )
        == 256
    )


@pytest.mark.parametrize("logical_generation_count", (0, 1, 2, 129, 130, True))
def test_agent_identity_growth_rejects_invalid_resident_horizon(
    logical_generation_count: object,
) -> None:
    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match=r"logical_generation_count must be an integer in \[3, 128\]",
    ):
        allocator.required_agent_id_capacity_v1(
            500,
            100,
            1,
            384,
            logical_generation_count,
        )


def test_birth_state_receipt_requires_exact_reusable_g2_capacity() -> None:
    valid = SimpleNamespace(
        receipt=SimpleNamespace(
            steps=320,
            max_births_per_tick=64,
            slot_capacity=384,
            turnover_refill_capacity=124,
            birth_episode_capacity=20_604,
            fixed_mlp_policy_tape_episodes=20_604,
            fixed_mlp_policy_full_horizon_episodes=20_604,
            policy_birth_segment_refill_enabled=False,
            torch_rng_post_state_encoding=("torch_cpu_rng_state_5056_u32_middle_v1"),
            torch_rng_post_state_logical_row_bytes=5_056,
            torch_rng_post_state_packed_row_bytes=2_560,
            torch_rng_post_state_dense_allocation_materialized=False,
            torch_rng_post_state_abi=(allocator.torch_cpu_rng_state_abi_receipt_v1()),
        )
    )
    allocator._validate_birth_state_receipt_v1(
        valid,
        max_births_per_tick=64,
        slot_capacity=384,
    )

    g1_valid = SimpleNamespace(
        receipt=SimpleNamespace(
            steps=160,
            max_births_per_tick=64,
            slot_capacity=384,
            turnover_refill_capacity=62,
            birth_episode_capacity=10_302,
            fixed_mlp_policy_tape_episodes=10_302,
            fixed_mlp_policy_full_horizon_episodes=10_302,
            policy_birth_segment_refill_enabled=False,
            torch_rng_post_state_encoding=("torch_cpu_rng_state_5056_u32_middle_v1"),
            torch_rng_post_state_logical_row_bytes=5_056,
            torch_rng_post_state_packed_row_bytes=2_560,
            torch_rng_post_state_dense_allocation_materialized=False,
            torch_rng_post_state_abi=(allocator.torch_cpu_rng_state_abi_receipt_v1()),
        )
    )
    allocator._validate_birth_state_receipt_v1(
        g1_valid,
        max_births_per_tick=64,
        slot_capacity=384,
        physical_window_generations=1,
    )
    tape_free = SimpleNamespace(
        receipt=SimpleNamespace(
            steps=160,
            max_births_per_tick=3,
            slot_capacity=128,
            turnover_refill_capacity=62,
            birth_episode_capacity=0,
            fixed_mlp_policy_tape_episodes=0,
            fixed_mlp_policy_full_horizon_episodes=0,
            cpython_gauss_raw_capacity=0,
            policy_birth_segment_refill_enabled=False,
            torch_rng_post_state_encoding=("torch_cpu_rng_state_5056_u32_middle_v1"),
            torch_rng_post_state_logical_row_bytes=5_056,
            torch_rng_post_state_packed_row_bytes=2_560,
            torch_rng_post_state_dense_allocation_materialized=False,
            torch_rng_post_state_abi=(allocator.torch_cpu_rng_state_abi_receipt_v1()),
        )
    )
    allocator._validate_birth_state_receipt_v1(
        tape_free,
        max_births_per_tick=3,
        slot_capacity=128,
        physical_window_generations=1,
        turnover_refill_capacity=62,
    )
    stub = SimpleNamespace(
        receipt=SimpleNamespace(
            steps=160,
            max_births_per_tick=3,
            slot_capacity=128,
            turnover_refill_capacity=62,
            birth_episode_capacity=1,
            fixed_mlp_policy_tape_episodes=1,
            fixed_mlp_policy_full_horizon_episodes=1,
            cpython_gauss_raw_capacity=0,
            policy_birth_segment_refill_enabled=False,
            torch_rng_post_state_encoding=("torch_cpu_rng_state_5056_u32_middle_v1"),
            torch_rng_post_state_logical_row_bytes=5_056,
            torch_rng_post_state_packed_row_bytes=2_560,
            torch_rng_post_state_dense_allocation_materialized=False,
            torch_rng_post_state_abi=(allocator.torch_cpu_rng_state_abi_receipt_v1()),
        )
    )
    allocator._validate_birth_state_receipt_v1(
        stub,
        max_births_per_tick=3,
        slot_capacity=128,
        physical_window_generations=1,
        turnover_refill_capacity=62,
    )
    abi_receipt = dict(valid.receipt.torch_rng_post_state_abi)
    valid.receipt.torch_rng_post_state_abi = {
        **abi_receipt,
        "byteorder": "big",
    }
    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="torch_rng_post_state_abi",
    ):
        allocator._validate_birth_state_receipt_v1(
            valid,
            max_births_per_tick=64,
            slot_capacity=384,
        )
    valid.receipt.torch_rng_post_state_abi = abi_receipt
    valid.receipt.max_births_per_tick = 1
    valid.receipt.turnover_refill_capacity = 384
    valid.receipt.birth_episode_capacity = 384
    valid.receipt.fixed_mlp_policy_tape_episodes = 384
    valid.receipt.fixed_mlp_policy_full_horizon_episodes = 384
    allocator._validate_birth_state_receipt_v1(
        valid,
        max_births_per_tick=1,
        slot_capacity=384,
        ordinary_births_enabled=False,
    )

    valid.receipt.max_births_per_tick = 64
    valid.receipt.turnover_refill_capacity = 124
    valid.receipt.birth_episode_capacity = 124
    valid.receipt.fixed_mlp_policy_tape_episodes = 124
    valid.receipt.fixed_mlp_policy_full_horizon_episodes = 20_604
    valid.receipt.policy_birth_segment_refill_enabled = True
    allocator._validate_birth_state_receipt_v1(
        valid,
        max_births_per_tick=64,
        slot_capacity=384,
    )

    valid.receipt.birth_episode_capacity = 63
    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match="policy birth segment refill capacity",
    ):
        allocator._validate_birth_state_receipt_v1(
            valid,
            max_births_per_tick=64,
            slot_capacity=384,
        )

    valid.receipt.policy_birth_segment_refill_enabled = False
    valid.receipt.max_births_per_tick = 64
    valid.receipt.birth_episode_capacity = 20_728
    valid.receipt.fixed_mlp_policy_tape_episodes = 20_604
    valid.receipt.fixed_mlp_policy_full_horizon_episodes = 20_604
    with pytest.raises(
        allocator.CudaArchitectureBDirectResidentSourceAllocatorError,
        match=r"birth_episode_capacity:expected=20604,actual=20728",
    ):
        allocator._validate_birth_state_receipt_v1(
            valid,
            max_births_per_tick=64,
            slot_capacity=384,
        )


def test_birth_state_receipt_binds_current_lifecycle_turnover_envelope() -> None:
    receipt = SimpleNamespace(
        steps=160,
        max_births_per_tick=3,
        slot_capacity=512,
        turnover_refill_capacity=256,
        birth_episode_capacity=256,
        fixed_mlp_policy_tape_episodes=256,
        fixed_mlp_policy_full_horizon_episodes=736,
        policy_birth_segment_refill_enabled=True,
        torch_rng_post_state_encoding=("torch_cpu_rng_state_5056_u32_middle_v1"),
        torch_rng_post_state_logical_row_bytes=5_056,
        torch_rng_post_state_packed_row_bytes=2_560,
        torch_rng_post_state_dense_allocation_materialized=False,
        torch_rng_post_state_abi=allocator.torch_cpu_rng_state_abi_receipt_v1(),
    )
    birth_state = SimpleNamespace(receipt=receipt)
    configuration = SimpleNamespace(
        lifecycle_mode=0,
        cull_fraction=0.5,
    )
    turnover_capacity = allocator._runtime_turnover_refill_capacity_v1(
        configuration,
        population_size=256,
        slot_capacity=512,
        ordinary_births_enabled=True,
        physical_window_generations=1,
    )
    assert turnover_capacity == 256
    allocator._validate_birth_state_receipt_v1(
        birth_state,
        max_births_per_tick=3,
        slot_capacity=512,
        physical_window_generations=1,
        ordinary_births_enabled=True,
        turnover_refill_capacity=turnover_capacity,
    )


def test_allocator_imports_no_challenger_or_competing_owner() -> None:
    source_path = Path(allocator.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert not any(
        name == "tools" or name.startswith("tools.") for name in imported_modules
    )
    assert "ResidentRuntimeState" not in imported_names
    assert "NativeRunController" not in imported_names
