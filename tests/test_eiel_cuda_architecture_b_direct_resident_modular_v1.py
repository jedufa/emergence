from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from agents.structural_identity_v1 import new_native_identity
from engine.perf import cuda_architecture_b_direct_resident_modular_v1 as modular
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    PreparedDirectResidentTrajectoryV1,
)
from engine.perf.cuda_architecture_b_modular_state_bank import (
    build_modular_state_bank_manifest,
)
from perf.cuda_architecture_b_modular_state_abi import pack_modular_storage
from tests.perf.test_eiel_cuda_architecture_b_modular_state_abi import (
    _module,
    _record,
)


CELLS = 1
CAPACITY = 4
MUTATION_SEQUENCE_CAPACITY = 3
MUTATION_RECEIPT_WIDTH = 21
ROWV1_RECEIPT_WIDTH = 64
KERNEL_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
)
BINDING_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_binding.cpp"
)


def _ordinary_birth_rowv1_from_raw(row: torch.Tensor) -> torch.Tensor:
    rowv1 = torch.full((ROWV1_RECEIPT_WIDTH,), -1, dtype=torch.int64)
    rowv1[0] = 3
    rowv1[1] = 2
    rowv1[2:22] = row[1:21]
    rowv1[22] = 0
    return rowv1


def _patch() -> PreparedDirectResidentTrajectoryV1:
    arguments: list[Any] = [object() for _ in DIRECT_RESIDENT_ARGUMENT_NAMES_V1]
    arguments[
        DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_runtime_bookkeeping")
    ] = torch.zeros((CELLS, 160, 5), dtype=torch.int64)
    arguments[
        DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_full_tick_output_tensors")
    ] = [torch.zeros((1,), dtype=torch.int64), torch.zeros((2, 1))]
    arguments[
        DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("physical_patch_stats_tensors")
    ] = [torch.zeros((CELLS, 160, 1))]
    arguments[
        DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("ordinary_birth_output_tensors")
    ] = [object() for _ in range(63)]
    arguments[DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("config.max_births_per_tick")] = 2
    arguments[
        DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("config.generation_turnover_enabled")
    ] = True
    arguments[
        DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index("ordinary_birth_scratch_tensors")
    ] = [object()]
    for name, count in (
        ("turnover_state_tensors", 122),
        ("turnover_shadow_tensors", 122),
        ("turnover_output_tensors", 16),
        ("turnover_pointer_table_tensors", 1),
    ):
        arguments[DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index(name)] = [
            object() for _ in range(count)
        ]
    source = SimpleNamespace(
        batch_cells=CELLS,
        generation_base=0,
        live_state={"alive_mask": torch.ones((CELLS, CAPACITY), dtype=torch.bool)},
    )
    return PreparedDirectResidentTrajectoryV1(
        source=source,
        arguments=tuple(arguments),
        physical_generation_summary=None,
        physical_runtime_bookkeeping=None,
        physical_patch_stats=None,
        physical_full_tick_output_tensors=(),
        live_banks={},
        checkpoint_projection_inputs={
            "pre_generation_canonical_projection": {
                "metadata": {
                    "policy_weight_mutation_enable": False,
                    "policy_weight_mutation_mode": "dense",
                    "policy_weight_mutation_rate": 0.125,
                    "policy_weight_mutation_std": 0.25,
                }
            }
        },
    )


def _cell_tensors() -> dict[str, torch.Tensor]:
    manifest = build_modular_state_bank_manifest(CAPACITY)
    return {
        field.projected_name: torch.full(
            field.shape,
            field.fill_value,
            dtype=field.torch_dtype,
        )
        for field in manifest.fields
    }


def _mutation_authority():
    authority = pack_modular_storage(
        [_record([_module(0)], agent_id=10, agent_slot_token=100), None, None, None],
        agent_capacity=CAPACITY,
        profile_id="b2-12x32-v1",
    )
    base_record = authority.records_sidecar[0]
    assert base_record is not None
    source = deepcopy(base_record["genome"]["modules"][0])
    duplicate = deepcopy(source)
    duplicate["module_id"] = "mutation_duplicate_m001"
    duplicate["params"]["live_output_gain"] = 0.0
    duplicate.update(
        new_native_identity(
            context={
                "campaign_id": "modular-sequence-unit",
                "branch_id": "qualification",
                "generation": 0,
                "tick": 0,
                "child_agent_id": 0,
                "parent_agent_ids": [],
                "lineage_token": "unit-duplicate",
            },
            operator_id="duplicate_with_decay",
            attempt_index=0,
            operator_role="neutral_duplicate",
            parent_module=source,
            inherited_homology=False,
        )
    )
    record = deepcopy(base_record)
    record["genome"]["modules"] = [source, duplicate]
    record["genome"]["connections"] = []
    record["structural_live_module_state"] = {
        str(module["module_instance_id"]): 0.0 for module in (source, duplicate)
    }
    record["structural_live_module_outputs"] = {
        str(module["module_instance_id"]): 0.0 for module in (source, duplicate)
    }
    record["structural_development_latch_by_instance_id"] = {}
    record["structural_development_effective_gain_by_instance_id"] = {}
    record.pop("cost_summary", None)
    return pack_modular_storage(
        [record, None, None, None],
        agent_capacity=CAPACITY,
        profile_id="b2-12x32-v1",
    )


def _capability() -> dict[str, Any]:
    return {
        "abi_version": 1,
        "resident_state": True,
        "physical_module_slots": 16,
        "physical_connection_slots": 48,
        "action_width": 7,
        "deterministic_fixed_mlp": True,
        "cache_width": 1,
        "births": False,
        "turnover": False,
        "native_tick_ownership": True,
        "native_generation_ownership": True,
        "runtime_jit_required": False,
        "hidden_fallback": False,
        "production_qualified": False,
    }


def _lifecycle_capability() -> dict[str, Any]:
    return {
        **_capability(),
        "abi_version": 2,
        "births": True,
        "turnover": True,
        "no_mutation_inheritance": True,
        "heterogeneous_parent_structure": True,
    }


def test_prepare_stacks_all_modular_planes_in_manifest_order() -> None:
    prepared = modular.prepare_direct_resident_modular_v1(
        _patch(),
        [_cell_tensors()],
    )
    manifest = build_modular_state_bank_manifest(CAPACITY)

    assert len(prepared.modular_tensors) == len(manifest.fields) == 57
    for tensor, field in zip(prepared.modular_tensors, manifest.fields, strict=True):
        assert tuple(tensor.shape) == (CELLS, *field.shape)
        assert tensor.dtype == field.torch_dtype


def test_action_residual_trace_is_explicit_and_fails_closed() -> None:
    prepared = modular.prepare_direct_resident_modular_v1(
        _patch(),
        [_cell_tensors()],
    )
    traced = modular.with_modular_action_residual_trace_v1(
        prepared,
        output_ticks_per_cell=3,
    )

    assert prepared.action_residual_trace is None
    assert tuple(traced.action_residual_trace.shape) == (1, 3, CAPACITY, 7)
    assert bool(torch.isnan(traced.action_residual_trace).all())
    assert (
        modular._config(traced)["action_residual_trace"] is traced.action_residual_trace
    )

    class OldModule:
        @staticmethod
        def eiel_cuda_architecture_b_flattened_modular_capability_v1():
            return _capability()

    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="does not support modular action residual tracing",
    ):
        modular.execute_direct_resident_modular_physical_v1(
            OldModule(), traced, generation_count=1, ticks_per_generation=1
        )


def test_turnover_mutation_qualification_is_explicit_fixed_shape_storage() -> None:
    lifecycle = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )
    draws = torch.zeros((CELLS, MUTATION_SEQUENCE_CAPACITY, 2), dtype=torch.float64)
    draws[..., 0] = 0.6
    draws[..., 1] = 0.6796875
    duplicate = torch.zeros((CELLS, CAPACITY, 16), dtype=torch.bool)
    duplicate[..., 1] = True

    prepared = modular.with_modular_turnover_mutation_qualification_v1(
        lifecycle,
        mutation_draws=draws,
        duplicate_descendant_mask=duplicate,
    )
    config = modular._config(prepared, lifecycle_enabled=True)

    assert torch.equal(prepared.turnover_mutation_draws, draws)
    assert torch.equal(prepared.duplicate_descendant_mask, duplicate)
    assert tuple(prepared.duplicate_descendant_mask_shadow.shape) == (
        CELLS,
        CAPACITY,
        16,
    )
    assert tuple(prepared.turnover_mutation_receipt_rows.shape) == (
        CELLS,
        MUTATION_SEQUENCE_CAPACITY,
        MUTATION_RECEIPT_WIDTH,
    )
    assert tuple(prepared.turnover_mutation_receipt_counts.shape) == (CELLS,)
    assert tuple(prepared.turnover_mutation_receipt_overflow.shape) == (CELLS,)
    assert prepared.turnover_mutation_receipt_overflow.dtype == torch.int32
    index = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index
    assert prepared.patch.arguments[index("config.max_births_per_tick")] == 0
    assert prepared.patch.arguments[index("ordinary_birth_output_tensors")] == []
    assert prepared.patch.arguments[index("config.cull_fraction")] == pytest.approx(
        1.0 / CAPACITY
    )
    assert set(config) >= {
        "turnover_mutation_draws",
        "duplicate_descendant_mask",
        "duplicate_descendant_mask_shadow",
        "turnover_mutation_receipt_rows",
        "turnover_mutation_receipt_counts",
        "turnover_mutation_receipt_overflow",
    }
    assert config["turnover_mutation_first_generation"] == 1


def test_ordinary_birth_mutation_is_fixed_shape_default_off_and_exclusive() -> None:
    lifecycle = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )
    draws = torch.full(
        (CELLS, MUTATION_SEQUENCE_CAPACITY, 2), 0.6, dtype=torch.float64
    )
    duplicate = torch.zeros((CELLS, CAPACITY, 16), dtype=torch.bool)
    duplicate[..., 1] = True

    prepared = modular.with_modular_ordinary_birth_mutation_qualification_v1(
        lifecycle,
        mutation_draws=draws,
        duplicate_descendant_mask=duplicate,
    )
    config = modular._config(prepared, lifecycle_enabled=True)

    assert prepared.turnover_mutation_draws is None
    assert torch.equal(prepared.ordinary_birth_mutation_draws, draws)
    assert tuple(prepared.ordinary_birth_mutation_receipt_rows.shape) == (
        CELLS,
        MUTATION_SEQUENCE_CAPACITY,
        MUTATION_RECEIPT_WIDTH,
    )
    assert tuple(prepared.ordinary_birth_rowv1_receipt_rows.shape) == (
        CELLS,
        MUTATION_SEQUENCE_CAPACITY,
        ROWV1_RECEIPT_WIDTH,
    )
    assert prepared.ordinary_birth_rowv1_receipt_rows.dtype == torch.int64
    assert tuple(prepared.ordinary_birth_rowv1_receipt_counts.shape) == (CELLS,)
    assert prepared.ordinary_birth_rowv1_receipt_counts.dtype == torch.int64
    assert tuple(prepared.ordinary_birth_rowv1_receipt_overflow.shape) == (CELLS,)
    assert prepared.ordinary_birth_rowv1_receipt_overflow.dtype == torch.int32
    assert set(config) >= {
        "ordinary_birth_mutation_draws",
        "ordinary_birth_mutation_receipt_rows",
        "ordinary_birth_mutation_receipt_counts",
        "ordinary_birth_mutation_receipt_overflow",
        "ordinary_birth_rowv1_receipt_rows",
        "ordinary_birth_rowv1_receipt_counts",
        "ordinary_birth_rowv1_receipt_overflow",
    }
    assert "turnover_mutation_draws" not in config
    assert config["ordinary_birth_mutation_first_generation"] == 0
    assert (
        prepared.patch.arguments[modular.TURNOVER_ARGUMENT_INDEX_V1] is False
    )
    assert all(
        prepared.patch.arguments[index] == []
        for index in modular.TURNOVER_MUTABLE_ARGUMENT_INDICES_V1
    )
    assert prepared.patch.arguments[modular.TURNOVER_OUTPUT_ARGUMENT_INDEX_V1]

    turnover = modular.with_modular_turnover_mutation_qualification_v1(
        lifecycle,
        mutation_draws=draws,
        duplicate_descendant_mask=duplicate,
    )
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="mutually exclusive",
    ):
        modular.with_modular_ordinary_birth_mutation_qualification_v1(
            turnover,
            mutation_draws=draws,
            duplicate_descendant_mask=duplicate,
        )


def test_ordinary_birth_mutation_capability_and_receipt_are_typed() -> None:
    lifecycle = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )
    authority = pack_modular_storage(
        [_record([_module(0)], agent_id=10, agent_slot_token=100), None, None, None],
        agent_capacity=CAPACITY,
        profile_id="b2-12x32-v1",
    )
    prepared = modular.with_modular_ordinary_birth_mutation_qualification_v1(
        replace(lifecycle, modular_authorities=(authority,)),
        mutation_draws=torch.full(
            (CELLS, MUTATION_SEQUENCE_CAPACITY, 2), 0.6, dtype=torch.float64
        ),
        duplicate_descendant_mask=torch.zeros(
            (CELLS, CAPACITY, 16), dtype=torch.bool
        ),
    )
    raw_only_capability = {
        **_lifecycle_capability(),
        "ordinary_birth_mutation": True,
        "ordinary_birth_gated_parameter_refinement": True,
        "ordinary_birth_mutation_receipt_raw_layout_version": 2,
        "ordinary_birth_mutation_receipt_capacity": 3,
        "ordinary_birth_mutation_default_production": False,
    }
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="ordinary_birth_rowv1_receipt_abi_version",
    ):
        modular._require_ordinary_birth_mutation_capability(
            raw_only_capability, prepared
        )
    capability = {
        **raw_only_capability,
        "ordinary_birth_rowv1_receipt_abi_version": 3,
        "ordinary_birth_rowv1_receipt_width": ROWV1_RECEIPT_WIDTH,
        "ordinary_birth_rowv1_receipt_capacity": MUTATION_SEQUENCE_CAPACITY,
        "ordinary_birth_rowv1_receipt_default_production": False,
        "ordinary_birth_rowv1_receipt_kernel_mirror_present": True,
        "ordinary_birth_rowv1_receipt_kernel_write_qualified": True,
        "ordinary_birth_rowv1_receipt_disabled_event_writer": True,
        "ordinary_birth_rowv1_receipt_device_writer_sanitized": True,
    }
    modular._require_ordinary_birth_mutation_capability(capability, prepared)

    row = torch.tensor(
        [
            1, 0, 1, 7, 10, 0, 100, 20, 3, 200, 0,
            int("3fe3333333333333", 16),
            int("3fe5c00000000000", 16),
            2, 1, 1, 0, 0, 0, 5, 1,
        ],
        dtype=torch.int64,
    )
    prepared.ordinary_birth_mutation_receipt_rows[0, 0].copy_(row)
    prepared.ordinary_birth_mutation_receipt_counts[0] = 1
    prepared.ordinary_birth_rowv1_receipt_rows[0, 0].copy_(
        _ordinary_birth_rowv1_from_raw(row)
    )
    prepared.ordinary_birth_rowv1_receipt_counts[0] = 1

    receipt = modular.project_modular_ordinary_birth_mutation_receipt_v1(prepared)

    assert receipt is not None
    transaction = receipt.transactions[0]
    assert transaction.transaction_kind == "ordinary_birth"
    assert transaction.changed_dimension == "reset_bias"
    assert transaction.parent_agent_id == 10
    assert transaction.child_agent_id == 20


def test_ordinary_birth_mutation_receipt_requires_rowv1_mirror() -> None:
    lifecycle = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )
    authority = pack_modular_storage(
        [_record([_module(0)], agent_id=10, agent_slot_token=100), None, None, None],
        agent_capacity=CAPACITY,
        profile_id="b2-12x32-v1",
    )
    prepared = modular.with_modular_ordinary_birth_mutation_qualification_v1(
        replace(lifecycle, modular_authorities=(authority,)),
        mutation_draws=torch.full(
            (CELLS, MUTATION_SEQUENCE_CAPACITY, 2), 0.6, dtype=torch.float64
        ),
        duplicate_descendant_mask=torch.zeros(
            (CELLS, CAPACITY, 16), dtype=torch.bool
        ),
    )
    row = torch.tensor(
        [
            1, 0, 1, 7, 10, 0, 100, 20, 3, 200, 0,
            int("3fe3333333333333", 16),
            int("3fe5c00000000000", 16),
            2, 1, 1, 0, 0, 0, 5, 1,
        ],
        dtype=torch.int64,
    )
    prepared.ordinary_birth_mutation_receipt_rows[0, 0].copy_(row)
    prepared.ordinary_birth_mutation_receipt_counts[0] = 1
    rowv1 = _ordinary_birth_rowv1_from_raw(row)
    prepared.ordinary_birth_rowv1_receipt_rows[0, 0].copy_(rowv1)
    prepared.ordinary_birth_rowv1_receipt_counts[0] = 1

    missing = replace(prepared, ordinary_birth_rowv1_receipt_rows=None)
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="RowV1 receipt tensors are incomplete",
    ):
        modular.project_modular_ordinary_birth_mutation_receipt_v1(missing)

    prepared.ordinary_birth_rowv1_receipt_rows[0, 0, 22] = 9
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="disabled-reason code is invalid",
    ):
        modular.project_modular_ordinary_birth_mutation_receipt_v1(prepared)
    prepared.ordinary_birth_rowv1_receipt_rows[0, 0].copy_(rowv1)

    prepared.ordinary_birth_rowv1_receipt_rows[0, 0, 23] = 0
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="receipt tail is not empty",
    ):
        modular.project_modular_ordinary_birth_mutation_receipt_v1(prepared)
    prepared.ordinary_birth_rowv1_receipt_rows[0, 0].copy_(rowv1)

    prepared.ordinary_birth_rowv1_receipt_rows[0, 0, 5] = 999
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="RowV1 mirror diverged from raw receipt prefix",
    ):
        modular.project_modular_ordinary_birth_mutation_receipt_v1(prepared)


def test_ordinary_birth_mutation_dispatch_is_bounded_to_physical_b1g1t1() -> None:
    lifecycle = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )
    prepared = modular.with_modular_ordinary_birth_mutation_qualification_v1(
        lifecycle,
        mutation_draws=torch.full(
            (CELLS, MUTATION_SEQUENCE_CAPACITY, 2), 0.6, dtype=torch.float64
        ),
        duplicate_descendant_mask=torch.zeros(
            (CELLS, CAPACITY, 16), dtype=torch.bool
        ),
    )

    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="requires b1g1t1",
    ):
        modular.execute_direct_resident_modular_lifecycle_physical_v2(
            object(),
            prepared,
            generation_count=1,
            ticks_per_generation=2,
        )
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="physical b1g1t1",
    ):
        modular.execute_direct_resident_modular_lifecycle_v2(
            object(),
            prepared,
            logical_generation_count=1,
            run_id="must-fail-before-dispatch",
        )


def test_turnover_mutation_qualification_rejects_bad_draws_and_capability() -> None:
    lifecycle = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )
    duplicate = torch.zeros((CELLS, CAPACITY, 16), dtype=torch.bool)
    invalid = torch.zeros((CELLS, MUTATION_SEQUENCE_CAPACITY, 2), dtype=torch.float64)
    invalid[0, 0, 0] = 1.0
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="values in",
    ):
        modular.with_modular_turnover_mutation_qualification_v1(
            lifecycle,
            mutation_draws=invalid,
            duplicate_descendant_mask=duplicate,
        )

    valid = torch.full((CELLS, MUTATION_SEQUENCE_CAPACITY, 2), 0.6, dtype=torch.float64)
    prepared = modular.with_modular_turnover_mutation_qualification_v1(
        lifecycle,
        mutation_draws=valid,
        duplicate_descendant_mask=duplicate,
    )
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="does not support bounded modular turnover mutation",
    ):
        modular._require_turnover_mutation_capability(_lifecycle_capability(), prepared)
    supported = {
        **_lifecycle_capability(),
        "turnover_refine_descendant_parameter_qualification": True,
        "turnover_mutation_receipt_raw_layout_version": 2,
        "turnover_mutation_receipt_sequence_capacity": 3,
        "mutation_receipt_sequence_v1": True,
        "turnover_mutation_qualified_generation_horizon": 3,
        "turnover_mutation_overflow_test_generation": 4,
        "mutated_checkpoint_fresh_owner": True,
        "ordinary_birth_mutation": True,
        "turnover_mutation_g8_g32": False,
        "turnover_mutation_default_production": False,
    }
    modular._require_turnover_mutation_capability(supported, prepared)
    for field in (
        "turnover_mutation_receipt_raw_layout_version",
        "turnover_mutation_receipt_sequence_capacity",
    ):
        malformed = dict(supported)
        malformed[field] = 99
        with pytest.raises(
            modular.CudaArchitectureBDirectResidentModularV1Error,
            match=field,
        ):
            modular._require_turnover_mutation_capability(malformed, prepared)


def test_turnover_mutation_receipt_projects_exact_bits_and_provenance() -> None:
    lifecycle = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )
    assert modular.project_modular_turnover_mutation_receipt_v1(lifecycle) is None
    authority = pack_modular_storage(
        [
            _record(
                [_module(0)],
                agent_id=10,
                agent_slot_token=100,
            ),
            None,
            None,
            None,
        ],
        agent_capacity=CAPACITY,
        profile_id="b2-12x32-v1",
    )
    draws = torch.zeros((CELLS, MUTATION_SEQUENCE_CAPACITY, 2), dtype=torch.float64)
    draws[..., 0] = 0.6
    draws[..., 1] = 0.6796875
    prepared = modular.with_modular_turnover_mutation_qualification_v1(
        replace(lifecycle, modular_authorities=(authority,)),
        mutation_draws=draws,
        duplicate_descendant_mask=torch.zeros((CELLS, CAPACITY, 16), dtype=torch.bool),
    )
    row = torch.tensor(
        [
            1,
            0,
            1,
            0,
            10,
            0,
            100,
            20,
            3,
            200,
            0,
            int("3fe3333333333333", 16),
            int("3fe5c00000000000", 16),
            2,
            1,
            1,
            0,
            0,
            0,
            3,
            1,
        ],
        dtype=torch.int64,
    )
    prepared.turnover_mutation_receipt_rows[0, 0].copy_(row)
    prepared.turnover_mutation_receipt_counts[0] = 1

    receipt = modular.project_modular_turnover_mutation_receipt_v1(prepared)

    assert receipt is not None
    assert receipt.source_packed_sha256 == authority.packed_sha256
    assert receipt.transactions[0].draw0_float64_bits == "3fe3333333333333"
    assert receipt.transactions[0].draw1_float64_bits == "3fe5c00000000000"
    assert receipt.transactions[0].child_slot == 3
    assert (
        receipt.canonical_sha256
        == "9e1b804528791577ed0c1c429a763d465a1294ad917ec19f91cbb67cb4c65653"
    )

    prepared.turnover_mutation_receipt_overflow[0] = 1
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="receipt sequence overflowed",
    ):
        modular.project_modular_turnover_mutation_receipt_v1(prepared)
    prepared.turnover_mutation_receipt_overflow[0] = 0

    prepared.turnover_mutation_receipt_rows[0, 1, 0] = 0
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="inactive.*rows are not empty",
    ):
        modular.project_modular_turnover_mutation_receipt_v1(prepared)


def test_turnover_mutation_sequence_replays_child_of_child_exactly() -> None:
    lifecycle = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )
    authority = _mutation_authority()
    record = authority.records_sidecar[0]
    assert record is not None
    duplicate_slot = next(
        index
        for index, module in enumerate(record["genome"]["modules"])
        if module["module_id"] == "mutation_duplicate_m001"
    )
    duplicate = record["genome"]["modules"][duplicate_slot]
    instance_index = authority.identity_table.index(duplicate["module_instance_id"])
    homology_index = authority.identity_table.index(duplicate["homology_id"])
    draws = torch.tensor(
        [[[0.6, 0.6796875], [0.6, 0.625], [0.6, 0.625]]],
        dtype=torch.float64,
    )
    prepared = modular.with_modular_turnover_mutation_qualification_v1(
        replace(lifecycle, modular_authorities=(authority,)),
        mutation_draws=draws,
        duplicate_descendant_mask=torch.zeros((CELLS, CAPACITY, 16), dtype=torch.bool),
    )
    rows = torch.tensor(
        [
            [
                1,
                0,
                1,
                0,
                10,
                0,
                100,
                20,
                1,
                200,
                0,
                int("3fe3333333333333", 16),
                int("3fe5c00000000000", 16),
                2,
                1,
                1,
                duplicate_slot,
                instance_index,
                homology_index,
                3,
                1,
            ],
            [
                1,
                1,
                2,
                0,
                20,
                1,
                200,
                30,
                2,
                300,
                0,
                int("3fe3333333333333", 16),
                int("3fe4000000000000", 16),
                2,
                1,
                1,
                duplicate_slot,
                instance_index,
                homology_index,
                3,
                1,
            ],
        ],
        dtype=torch.int64,
    )
    prepared.turnover_mutation_receipt_rows[0, :2].copy_(rows)
    prepared.turnover_mutation_receipt_counts[0] = 2

    sequence = modular.project_modular_turnover_mutation_sequence_v1(prepared)

    assert sequence is not None
    assert sequence.count == 2
    first, second = sequence.entries[:2]
    assert first is not None and second is not None
    assert (first.generation, second.generation) == (1, 2)
    assert second.parent_agent_id == first.child_agent_id == 20
    assert second.pre_genome_sha256 == first.post_genome_sha256
    assert first.post_genome_sha256 != second.post_genome_sha256
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="exactly one receipt",
    ):
        modular.project_modular_turnover_mutation_receipt_v1(prepared)


def test_turnover_mutation_executes_inside_shadow_transaction() -> None:
    source = KERNEL_SOURCE.read_text(encoding="utf-8")
    modular_source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_architecture_b_flattened_modular_controller_v1.cuh"
    ).read_text(encoding="utf-8")
    turnover_source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_architecture_b_flattened_turnover_v1_kernels.cuh"
    ).read_text(encoding="utf-8")
    inherit = turnover_source.index("inherit_lifecycle_newborn_v2(")
    mutate = turnover_source.index("mutate_turnover_descendant_parameter_v1(", inherit)
    publish = turnover_source.index("copy_lifecycle_cell_shadow_to_source_v2(", mutate)
    assert inherit < mutate < publish
    assert "turnover_refine_descendant_parameter_qualification" in (
        BINDING_SOURCE.read_text(encoding="utf-8")
    )
    assert "transaction_ordinal != 0 || commit_tick != 0" in modular_source
    assert (
        "context.turnover_mutation_first_generation + sequence_index"
        in modular_source
    )
    mutation_body = modular_source.index("const std::int64_t sequence_index =")
    overflow = modular_source.index(
        "turnover_mutation_receipt_overflow[cell] = 1",
        mutation_body,
    )
    rowv1_overflow = modular_source.index(
        "ordinary_birth_rowv1_receipt_overflow[cell] = 1",
        mutation_body,
    )
    rowv1_count_guard = modular_source.index(
        "ordinary_birth_rowv1_receipt_counts[cell] != sequence_index",
        mutation_body,
    )
    receipt_write = modular_source.index("receipt[0] = 1", mutation_body)
    rowv1_write = modular_source.index(
        "write_ordinary_birth_rowv1_receipt_v3(",
        receipt_write,
    )
    assert overflow < receipt_write
    assert overflow < rowv1_overflow < receipt_write
    assert rowv1_count_guard < receipt_write < rowv1_write
    assert "kOrdinaryBirthRowV1ReceiptWidthV3" in modular_source
    assert "rowv1[1] = 2;" in modular_source
    assert "write_ordinary_birth_rowv1_receipt_v3(" in modular_source
    assert "write_ordinary_birth_rowv1_disabled_event_v3(" in modular_source
    assert "rowv1[22] = raw_disabled_reason;" in modular_source
    assert "kOrdinaryBirthRowV1DisabledReasonAttemptBudgetZero = 6" in modular_source
    assert "ordinary_birth_rowv1_receipt_counts[cell] = sequence_index + 1" in (
        modular_source
    )
    assert "run_flattened_turnover_v1(" in source


def test_turnover_mutation_supports_fixed_width_gated_parameters() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "engine/perf/cuda_architecture_b_flattened_modular_controller_v1.cuh"
    ).read_text(encoding="utf-8")

    assert "operator_code[selected_module] != 2" in source
    assert "const std::int64_t field_count = gated ? 10 : 4;" in source
    assert "gated ? kGatedParametersV1 : kIntegratorParametersV1" in source
    assert "changed_dimension = gated ? field + 5 : field + 1;" in source


def test_lifecycle_builder_rejects_invalid_output_tick_extent() -> None:
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match=r"output_ticks must be in \[1,80\]",
    ):
        modular.build_prepared_direct_resident_modular_lifecycle_v2(
            object(),
            object(),
            qualified_cuda_module=object(),
            output_ticks=0,
        )


def test_lifecycle_builder_completes_fresh_event_cursor_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_projection = SimpleNamespace(
        schema_version="projection-v1",
        metadata={
            "typed_lifecycle_event_next_sequence": 7,
            "generation": 0,
            "tick": 0,
        },
        tensors={
            "alive_mask": torch.zeros((0,), dtype=torch.bool),
            "slot_to_agent_id": torch.zeros((0,), dtype=torch.int64),
            "slot_tokens": torch.zeros((0,), dtype=torch.int64),
        },
    )
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        modular,
        "build_import_projection",
        lambda *_args, **_kwargs: raw_projection,
    )
    monkeypatch.setattr(modular, "validate_modular_storage", lambda _authority: None)
    monkeypatch.setattr(
        modular,
        "modular_storage_projection_payload",
        lambda _authority: {"metadata": {}, "tensors": {}},
    )
    monkeypatch.setattr(modular, "projection_digest", lambda _projection: "digest")

    def capture_request(*_args: Any, **kwargs: Any) -> None:
        captured.update(kwargs["source_checkpoint_projection"])
        raise RuntimeError("captured")

    monkeypatch.setattr(
        modular,
        "build_direct_resident_source_allocation_request_v1",
        capture_request,
    )
    authority = SimpleNamespace(
        layout=SimpleNamespace(agent_capacity=0),
        tensors={
            "agent_occupied": torch.zeros((0,), dtype=torch.bool),
            "agent_id": torch.zeros((0,), dtype=torch.int64),
            "agent_slot_token": torch.zeros((0,), dtype=torch.int64),
        },
    )

    with pytest.raises(RuntimeError, match="captured"):
        modular.build_prepared_direct_resident_modular_lifecycle_v2(
            object(),
            authority,
            qualified_cuda_module=object(),
        )

    assert captured["event_cursor"] == 7
    assert captured["runtime_event_cursor"] == 7


def test_native_failure_record_fails_closed() -> None:
    modular._require_no_native_failure(
        {
            "first_failure_by_cell": [
                {"cell": -1, "generation": -1, "tick": -1, "status": -1}
            ],
            "first_failure_no_failure_sentinel": -1,
        }
    )
    with pytest.raises(
        modular.CudaArchitectureBDirectResidentModularV1Error,
        match="native execution failed.*status=160",
    ):
        modular._require_no_native_failure(
            {
                "first_failure_by_cell": [
                    {"cell": 0, "generation": 0, "tick": 0, "status": 160}
                ]
            }
        )


def test_physical_call_disables_lifecycle_and_keeps_patch_scratch() -> None:
    captured: dict[str, Any] = {}

    class Module:
        @staticmethod
        def eiel_cuda_architecture_b_flattened_modular_capability_v1():
            return _capability()

        @staticmethod
        def eiel_cuda_architecture_b_execute_direct_resident_modular_physical_v1(
            tensors: list[torch.Tensor], config: dict[str, Any], *arguments: Any
        ) -> dict[str, Any]:
            captured.update(tensors=tensors, config=config, arguments=arguments)
            return {
                "modular_native_owner": True,
                "modular_native_controller_abi_version": 1,
                "modular_physical_gate_entrypoint": True,
                "modular_python_tick_or_generation_ownership": False,
                "modular_lifecycle_supported": False,
                "hidden_fallback": False,
                "runtime_jit": False,
            }

    prepared = modular.prepare_direct_resident_modular_v1(
        _patch(),
        [_cell_tensors()],
    )
    modular.execute_direct_resident_modular_physical_v1(
        Module(),
        prepared,
        generation_count=1,
        ticks_per_generation=1,
    )

    arguments = captured["arguments"]
    index = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index
    assert arguments[index("config.cache_width")] == 1
    assert arguments[index("deterministic")] is True
    assert arguments[index("config.max_births_per_tick")] == 0
    assert arguments[index("config.generation_turnover_enabled")] is False
    assert arguments[index("ordinary_birth_output_tensors")] == []
    assert len(arguments[index("ordinary_birth_scratch_tensors")]) == 1
    for name in (
        "turnover_state_tensors",
        "turnover_shadow_tensors",
        "turnover_output_tensors",
        "turnover_pointer_table_tensors",
    ):
        assert arguments[index(name)] == []


def test_kernel_applies_modular_residual_before_action_selection() -> None:
    source = KERNEL_SOURCE.read_text(encoding="utf-8")
    execute = source.index("persistent_modular::action_residuals_exact_v1(")
    apply = source.index("score += modular_residuals[action];", execute)
    select = source.index("if (action == 0 || score > best_score)", apply)

    assert execute < apply < select


def test_versioned_native_entrypoints_fail_closed_on_lifecycle_mode() -> None:
    source = BINDING_SOURCE.read_text(encoding="utf-8")

    assert source.count("context.lifecycle_enabled == expected_lifecycle_enabled") == 2
    assert source.count("modular_config, arguments, false") == 1
    assert source.count("modular_config, arguments, true") == 1
    assert (
        source.count(
            "modular_config,\n                arguments,\n                false"
        )
        == 1
    )
    assert (
        source.count(
            "modular_config,\n                arguments,\n                true"
        )
        == 1
    )


def test_lifecycle_physical_call_preserves_birth_and_turnover_arguments() -> None:
    captured: dict[str, Any] = {}

    class Module:
        @staticmethod
        def eiel_cuda_architecture_b_flattened_modular_lifecycle_capability_v2():
            return _lifecycle_capability()

        @staticmethod
        def eiel_cuda_architecture_b_execute_direct_resident_modular_lifecycle_physical_v2(
            tensors: list[torch.Tensor], config: dict[str, Any], *arguments: Any
        ) -> dict[str, Any]:
            captured.update(tensors=tensors, config=config, arguments=arguments)
            return {
                "modular_native_owner": True,
                "modular_native_controller_abi_version": 2,
                "modular_physical_gate_entrypoint": True,
                "modular_python_tick_or_generation_ownership": False,
                "modular_lifecycle_supported": True,
                "hidden_fallback": False,
                "runtime_jit": False,
            }

    prepared = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )
    modular.execute_direct_resident_modular_lifecycle_physical_v2(
        Module(), prepared, generation_count=2, ticks_per_generation=80
    )

    index = DIRECT_RESIDENT_ARGUMENT_NAMES_V1.index
    arguments = captured["arguments"]
    assert len(captured["tensors"]) == 114
    assert captured["config"]["lifecycle_enabled"] is True
    assert tuple(captured["config"]["lifecycle_pointer_table"].shape) == (57, 5)
    assert tuple(captured["config"]["cold_origin_slots"].shape) == (1, CAPACITY)
    assert captured["config"]["cold_origin_slots"].tolist() == [[-1, -1, -1, -1]]
    assert tuple(captured["config"]["cold_origin_slots_shadow"].shape) == (
        1,
        CAPACITY,
    )
    assert arguments[index("config.max_births_per_tick")] == 2
    assert arguments[index("config.generation_turnover_enabled")] is True
    assert len(arguments[index("turnover_state_tensors")]) == 122


def test_lifecycle_g1_uses_native_owner_and_checkpoint_mutation_contract(
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}
    control = object()

    class Module:
        @staticmethod
        def eiel_cuda_architecture_b_flattened_modular_lifecycle_capability_v2():
            return _lifecycle_capability()

        @staticmethod
        def eiel_cuda_architecture_b_execute_direct_resident_modular_lifecycle_v2(
            run_control: object,
            tensors: list[torch.Tensor],
            config: dict[str, Any],
            *arguments: Any,
        ) -> dict[str, Any]:
            captured.update(
                run_control=run_control,
                tensors=tensors,
                config=config,
                arguments=arguments,
            )
            return {
                "modular_native_owner": True,
                "modular_native_controller_abi_version": 2,
                "modular_python_tick_or_generation_ownership": False,
                "modular_lifecycle_supported": True,
                "completed_generations": 1,
                "logical_generation_count": 1,
                "external_python_window_boundaries": 0,
                "direct_cpp_window_core_dispatch": True,
                "production_native_chunk_loop": True,
                "hidden_fallback": False,
                "runtime_jit": False,
            }

    monkeypatch.setattr(
        modular,
        "create_run_control_v2",
        lambda _module, *, run_id: control,
    )
    prepared = modular.prepare_direct_resident_modular_lifecycle_v2(
        _patch(), [_cell_tensors()]
    )

    modular.execute_direct_resident_modular_lifecycle_v2(
        Module(), prepared, logical_generation_count=1, run_id="test-g1"
    )

    assert captured["run_control"] is control
    assert len(captured["tensors"]) == 114
    assert captured["config"]["lifecycle_enabled"] is True
    assert tuple(captured["config"]["cold_origin_slots"].shape) == (1, CAPACITY)
    assert captured["arguments"][-5:] == (1, False, "dense", 0.125, 0.25)
