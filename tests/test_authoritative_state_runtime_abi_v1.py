from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from engine.authoritative_state.errors import AuthoritativeStateContractError
from engine.authoritative_state.identity import canonical_sha256
from engine.authoritative_state.runtime_abi import (
    RUNTIME_ABI_RECEIPT_SCHEMA_V1,
    _admits_replay_backed_policy_post_rng_state_v1,
    _shape_symbols,
    _small_int_vector,
    _valid_count,
    _validate_observed_variant_metadata_v1,
    observe_prepared_direct_resident_abi_v1,
    resolve_prepared_direct_resident_trajectory_v1,
    validate_runtime_abi_order_v1,
)
from engine.campaigns.production_engine_native_bindings import (
    PreparedNativeMechanicsOwnerV1,
)
from engine.perf import cuda_architecture_b_direct_resident_preparation_v1 as preparation
from engine.perf.cuda_architecture_b_boundary import EVENT_COLUMN_NAMES
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DIRECT_RESIDENT_ARGUMENT_NAMES_V1,
    GENERATION_SUMMARY_WIDTH_V1,
    LIVE_STATE_ARGUMENT_FIELDS_V1,
    PATCH_P2_CONFIG_ARGUMENTS_V1,
    PATCH_P2_STATE_FIELDS_V1,
    POST_BOOKKEEPING_OUTPUT_ARGUMENT_FIELDS_V1,
    POST_GENERATION_CONFIG_ARGUMENTS_V1,
    PREFIX_OUTPUT_ARGUMENT_FIELDS_V1,
    PRE_GENERATION_CONFIG_ARGUMENTS_V1,
    REPORTING_FITNESS_ARGUMENTS_V1,
    DirectResidentSourceV1,
    prepare_direct_resident_trajectory_v1,
    required_lifecycle_scratch_bytes_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_patch_v1 import (
    PreparedDirectResidentPatchMulticellV1,
)
from engine.perf.cuda_architecture_b_cross_path_resident_overlay_v1 import (
    CudaCrossPathResidentOverlayError,
    PreparedCrossPathDirectResidentPatchMulticellV1,
)
from engine.perf.cuda_architecture_b_direct_resident_gmnw_gru_v1 import (
    PreparedDirectResidentGmnwGruV1,
)
from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
    PreparedDirectResidentGruV3,
)
from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
    PreparedDirectResidentEedV1,
)
from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (
    PreparedDirectResidentJungleV4,
)
from engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1 import (
    ORDINARY_BIRTH_OUTPUT_FIELDS,
    TURNOVER_MUTABLE_STATE_FIELDS,
    TURNOVER_OUTPUT_FIELDS,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_gru_v3 import (
    PreparedDirectResidentWdlGruV3,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
    PreparedDirectResidentWdlV2,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_jungle_v1 import (
    PreparedDirectResidentWdlJungleV1,
)


BATCH = 3
CAPACITY = 64
AGENT_ID_CAPACITY = 70
EVENT_CAPACITY = 32
MAX_BIRTHS = 3
MLP_WIDTH = 16
HMEM_WIDTH = 8
TORCH_STATE_WIDTH = 32


def _typed(value_type: type) -> object:
    if value_type is bool:
        return False
    if value_type is int:
        return 1
    if value_type is float:
        return 1.25
    raise AssertionError(value_type)


def _configuration(
    *,
    generation_turnover_enabled: bool = True,
    lifecycle_mode: int = 0,
) -> SimpleNamespace:
    values: dict[str, object] = {}
    for schema in (
        PRE_GENERATION_CONFIG_ARGUMENTS_V1,
        POST_GENERATION_CONFIG_ARGUMENTS_V1,
        REPORTING_FITNESS_ARGUMENTS_V1,
    ):
        for name, value_type in schema:
            values[name] = _typed(value_type)
    values.update(
        start_tick=0,
        ordinary_births_enabled=True,
        max_births_per_tick=MAX_BIRTHS,
        reproduce_requires_mate=False,
        generation_turnover_enabled=generation_turnover_enabled,
        lifecycle_mode=lifecycle_mode,
        population_size=64,
        max_energy=20.0,
        start_health=1.0,
        life_history_mode=0,
        life_history_maturity_start_ticks=0,
        life_history_maturity_rise_ticks=0,
        life_history_fertility_decline_start_ticks=0,
        life_history_fertility_decline_ticks=0,
        life_history_elder_fertility_floor=1.0,
        life_history_senescence_start_ticks=0,
        life_history_senescence_ramp_ticks=0,
        life_history_senescence_energy_cost=0.0,
        life_history_senescence_health_cost=0.0,
        life_history_max_age_ticks=0,
    )
    return SimpleNamespace(**values)


def _tensor(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    fill: int | float = 0,
) -> torch.Tensor:
    return torch.full(shape, fill, dtype=dtype, device=device)


def _prepared(
    device_name: str,
    *,
    generation_turnover_enabled: bool = True,
    lifecycle_mode: int = 0,
):
    device = torch.device(device_name)
    live = {
        name: _tensor((BATCH, CAPACITY), device=device)
        for name in (*LIVE_STATE_ARGUMENT_FIELDS_V1, *PATCH_P2_STATE_FIELDS_V1)
    }
    for name in PATCH_P2_STATE_FIELDS_V1:
        live[name] = _tensor((BATCH, 4, 5), device=device)
    live["energy"] = _tensor((BATCH, CAPACITY), device=device)
    live["agent_id_to_slot"] = _tensor(
        (BATCH, AGENT_ID_CAPACITY),
        device=device,
        dtype=torch.int32,
        fill=-1,
    )
    live["agent_id_to_slot"][:, :4] = torch.arange(
        4, dtype=torch.int32, device=device
    )
    live["mlp_pack"] = _tensor((BATCH, CAPACITY, MLP_WIDTH), device=device)
    live["drives"] = _tensor((BATCH, CAPACITY, 4), device=device)
    live["slot_free_queue"] = _tensor(
        (BATCH, CAPACITY), device=device, dtype=torch.int64, fill=-1
    )
    live["slot_free_queue_count"] = _tensor(
        (BATCH,), device=device, dtype=torch.int64
    )
    live["pending_free_count"] = _tensor(
        (BATCH,), device=device, dtype=torch.int64
    )
    live["pending_free_count"].copy_(
        torch.tensor([1, 0, 2], dtype=torch.int64, device=device)
    )
    live["policy_np_semantic_draws"] = _tensor(
        (BATCH,), device=device, dtype=torch.int64
    )
    live["slot_free_queue_count"].copy_(
        torch.tensor([2, 1, 0], dtype=torch.int64, device=device)
    )

    outputs = {
        name: _tensor((BATCH, CAPACITY), device=device)
        for name in (
            *PREFIX_OUTPUT_ARGUMENT_FIELDS_V1,
            *POST_BOOKKEEPING_OUTPUT_ARGUMENT_FIELDS_V1,
            "first_failure_records",
        )
    }
    outputs["generation_summary_scalar"] = _tensor(
        (2, BATCH, GENERATION_SUMMARY_WIDTH_V1),
        device=device,
        dtype=torch.float64,
    )
    outputs["runtime_bookkeeping_by_tick"] = _tensor(
        (BATCH, 320, 5), device=device, dtype=torch.int64
    )

    active_count = torch.tensor([4, 3, 2], dtype=torch.int64, device=device)
    active_agent_ids = _tensor(
        (BATCH, CAPACITY), device=device, dtype=torch.int64, fill=-1
    )
    active_agent_slots = _tensor(
        (BATCH, CAPACITY), device=device, dtype=torch.int32, fill=-1
    )
    for cell, count in enumerate(active_count.tolist()):
        active_agent_ids[cell, :count] = torch.arange(
            count, dtype=torch.int64, device=device
        )
        active_agent_slots[cell, :count] = torch.arange(
            count, dtype=torch.int32, device=device
        )
    live["active_agent_ids"] = active_agent_ids
    live["active_agent_slots"] = active_agent_slots
    event_count = torch.tensor([3, 2, 1], dtype=torch.int64, device=device)
    event_state = [
        _tensor((BATCH, EVENT_CAPACITY), device=device, dtype=torch.int64)
        for _ in EVENT_COLUMN_NAMES
    ]
    event_state.extend(
        [
            event_count,
            _tensor((BATCH,), device=device, dtype=torch.int32),
            _tensor((BATCH, CAPACITY), device=device, dtype=torch.int32, fill=-1),
            _tensor((BATCH, CAPACITY), device=device, dtype=torch.int32),
            _tensor(
                (BATCH, 4),
                device=device,
                dtype=torch.int64,
                fill=torch.iinfo(torch.int64).max,
            ),
            _tensor((BATCH, 4), device=device, dtype=torch.int32),
            _tensor((BATCH, 4), device=device, dtype=torch.int32),
            _tensor((BATCH, 160), device=device, dtype=torch.int32),
            _tensor((BATCH,), device=device, dtype=torch.int64),
        ]
    )
    for cell, count in enumerate(active_count.tolist()):
        event_state[26][cell, :count] = torch.arange(
            count, dtype=torch.int64, device=device
        )
    assert len(event_state) == 31

    full_tick_state = [
        _tensor((BATCH, CAPACITY), device=device) for _ in range(29)
    ]
    full_tick_state[6] = _tensor((BATCH, CAPACITY, 4), device=device)
    full_tick_state[15] = _tensor((BATCH, CAPACITY, 4), device=device)
    full_tick_state[20] = _tensor((BATCH, 2, CAPACITY, 4), device=device)
    full_tick_state[21] = _tensor(
        (BATCH, 2, CAPACITY),
        device=device,
        dtype=torch.bool,
    )
    full_tick_state[21][:, 0, :4] = True
    full_tick_state[22] = live["slot_free_queue"]
    full_tick_state[23] = live["slot_free_queue_count"]
    full_tick_state[24] = _tensor(
        (BATCH, CAPACITY), device=device, dtype=torch.int64, fill=-1
    )
    full_tick_state[25] = _tensor(
        (BATCH, CAPACITY), device=device, dtype=torch.int64, fill=-1
    )
    full_tick_state[26] = _tensor(
        (BATCH,), device=device, dtype=torch.int64
    )
    full_tick_state[27] = _tensor((BATCH,), device=device, dtype=torch.int64)
    full_tick_state[28] = _tensor((BATCH,), device=device, dtype=torch.int64)
    full_tick_outputs = [
        _tensor((BATCH, 160), device=device) for _ in range(19)
    ]
    full_tick_outputs.append(outputs["generation_summary_scalar"])

    birth_state = [
        _tensor((BATCH, CAPACITY), device=device) for _ in range(113)
    ]
    # Energy is ordinal 7 in the ordinary-birth identity prefix.
    birth_state[7] = live["energy"]
    birth_outputs = [
        _tensor((BATCH, MAX_BIRTHS), device=device)
        for _ in ORDINARY_BIRTH_OUTPUT_FIELDS
    ]
    scratch_bytes = required_lifecycle_scratch_bytes_v1(
        MAX_BIRTHS,
        CAPACITY,
        MLP_WIDTH,
        HMEM_WIDTH,
        TORCH_STATE_WIDTH,
    )
    birth_scratch = [
        _tensor((BATCH, scratch_bytes), device=device, dtype=torch.uint8)
    ]

    turnover_state = [
        live.get(name, _tensor((BATCH, CAPACITY), device=device))
        for name in TURNOVER_MUTABLE_STATE_FIELDS
    ]
    turnover_shadows = [torch.empty_like(tensor) for tensor in turnover_state]
    turnover_outputs = [
        _tensor((BATCH, CAPACITY), device=device)
        for _ in TURNOVER_OUTPUT_FIELDS
    ]
    patch_state = [live[name] for name in PATCH_P2_STATE_FIELDS_V1]
    patch_stats = _tensor((BATCH, 320, 25), device=device, dtype=torch.float64)
    patch_config = {
        name: _typed(value_type)
        for name, value_type in PATCH_P2_CONFIG_ARGUMENTS_V1
    }
    hmem = _tensor((BATCH, CAPACITY, HMEM_WIDTH), device=device)
    torch_state = _tensor(
        (BATCH, TORCH_STATE_WIDTH), device=device, dtype=torch.uint8
    )
    source = DirectResidentSourceV1(
        configuration=_configuration(
            generation_turnover_enabled=generation_turnover_enabled,
            lifecycle_mode=lifecycle_mode,
        ),
        active_agent_ids=active_agent_ids,
        active_agent_slots=active_agent_slots,
        active_count_state=active_count,
        live_state=live,
        outputs=outputs,
        full_tick_state_tensors=full_tick_state,
        full_tick_output_tensors=full_tick_outputs,
        compact_summary_reporting_tensors=(),
        compact_summary_checkpoint_source_tensors=(),
        compact_summary_checkpoint_descriptor_specs=torch.empty(
            (0, 13), dtype=torch.int64
        ),
        compact_summary_checkpoint_output_tensors=(),
        compact_summary_settlement_tensors=(),
        event_state_tensors=event_state,
        ordinary_birth_state_tensors=birth_state,
        ordinary_birth_output_tensors=birth_outputs,
        ordinary_birth_scratch_tensors=birth_scratch,
        turnover_selection_seed_by_cell=torch.tensor(
            tuple(range(BATCH)),
            dtype=torch.uint64,
            device=device,
        ),
        turnover_state_tensors=(
            turnover_state if generation_turnover_enabled else []
        ),
        turnover_shadow_tensors=(
            turnover_shadows if generation_turnover_enabled else []
        ),
        turnover_output_tensors=turnover_outputs,
        turnover_pointer_table_tensors=(
            [_tensor((484,), device=device, dtype=torch.int64)]
            if generation_turnover_enabled
            else []
        ),
        patch_p2_state_tensors=patch_state,
        patch_p2_scratch_tensors=[_tensor((BATCH, 4, 4, 5), device=device)],
        patch_p2_stats_tensors=[patch_stats],
        patch_p2_config=patch_config,
        live_banks={
            "mlp_pack": live["mlp_pack"],
            "hmem": hmem,
            "rng_manager_mutation_torch_cpu_state": torch_state,
        },
        checkpoint_projection_inputs={},
        deterministic=False,
        generation_base=0,
        logical_generation_count=4,
        physical_window_generations=2,
        ticks_per_generation=80,
        batch_cells=BATCH,
    )
    return prepare_direct_resident_trajectory_v1(source)


def _by_path(receipt: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["path"]: row for row in receipt["tensors"]}


def _abi_contract() -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[1]
        / "_spec/authoritative_state/v1/abi_variants.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _allow_cpu_contract_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    original = preparation._require_turnover_selection_seed_by_cell

    def validate(value, **kwargs):
        if value.device.type == "cpu":
            return value
        return original(value, **kwargs)

    monkeypatch.setattr(
        preparation,
        "_require_turnover_selection_seed_by_cell",
        validate,
    )


def test_all_supported_abi_entries_bind_logical_and_physical_widths() -> None:
    entries = [
        entry
        for variant in _abi_contract()["variants"]
        if variant["support_status"] == "supported_exact_metadata"
        for entry in variant["entries"]
    ]
    assert len(entries) == 825
    for entry in entries:
        contract = entry["authority_contract"]
        assert entry["physical_type"] == str(contract.get("kind", "tensor"))
        if entry["dtype"] == "bool":
            assert entry["bit_width"] == 1
            assert contract["bit_width"] == 8
        elif contract.get("kind") in {
            "ordered_tensor_container",
            "ordered_mapping",
        }:
            assert entry["bit_width"] == "container"
            assert contract["bit_width"] is None
        else:
            assert entry["bit_width"] == contract["bit_width"]


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_b3_patch_prepared_owner_emits_deterministic_complete_receipt(
    device_name: str,
) -> None:
    prepared = _prepared(device_name)
    first = observe_prepared_direct_resident_abi_v1(prepared)
    second = observe_prepared_direct_resident_abi_v1(prepared)

    assert first == second
    json.dumps(first, allow_nan=False, sort_keys=True)
    assert first["schema_version"] == RUNTIME_ABI_RECEIPT_SCHEMA_V1
    assert first["phase"] == "pre_dispatch"
    assert first["outer_argument_count"] == len(DIRECT_RESIDENT_ARGUMENT_NAMES_V1)
    assert len(first["outer_arguments"]) == len(DIRECT_RESIDENT_ARGUMENT_NAMES_V1)
    assert first["tensor_count"] == len(first["tensors"])
    assert first["tensor_metadata_sha256"] == canonical_sha256(first["tensors"])
    unsigned = dict(first)
    assert unsigned.pop("receipt_sha256") == canonical_sha256(unsigned)

    shape = first["shape_formula_inputs"]
    assert shape == {
        "batch_cells": 3,
        "slot_capacity": 64,
        "agent_id_capacity": 70,
        "event_capacity": 32,
        "chemistry_width": 4,
        "drive_width": 4,
        "output_tick_capacity": 160,
        "patch_height": 4,
        "patch_width": 5,
        "ticks_per_generation": 80,
        "physical_generation_count": 2,
        "physical_window_ticks": 160,
        "max_births_per_tick": 3,
        "active_count_by_cell": [4, 3, 2],
        "event_active_count_by_cell": [3, 2, 1],
        "slot_free_queue_count_by_cell": [2, 1, 0],
        "pending_free_count_by_cell": [1, 0, 2],
        "residual_count_by_cell": [0, 0, 0],
        "drive_noise_valid_count": 12,
    }
    rows = _by_path(first)
    assert rows["live_state.energy"]["dtype"] == "float32"
    assert rows["live_state.energy"]["rank"] == 2
    assert rows["live_state.energy"]["shape"] == [3, 64]
    assert rows["live_state.energy"]["capacity_count"] == 192
    assert rows["live_state.energy"]["valid_count"] == 9
    assert rows["event_state_tensors.event_tape_type"]["valid_count"] == 6
    assert (
        rows["ordinary_birth_output_tensors.birth_planned_count"]["valid_count"]
        == 0
    )
    assert rows["live_state.agent_id_to_slot"]["valid_count"] == 12
    assert all(
        isinstance(row["capacity_count"], int)
        and isinstance(row["valid_count"], int)
        and "shape_formula_inputs" in row
        and "alias_identity" in row
        for row in first["tensors"]
    )
    # The same energy storage appears in the outer live-state ABI, birth state,
    # and turnover state; deterministic alias labels prove the relationship.
    energy_alias = rows["live_state.energy"]["alias_identity"]["storage_alias"]
    assert (
        rows["ordinary_birth_state_tensors.energy"]["alias_identity"][
            "storage_alias"
        ]
        == energy_alias
    )
    assert (
        rows["turnover_state_tensors.energy"]["alias_identity"]["storage_alias"]
        == energy_alias
    )
    assert not any(
        key in row["alias_identity"]
        for row in first["tensors"]
        for key in ("data_ptr", "storage_ptr", "object_id")
    )
    admitted = validate_runtime_abi_order_v1(first, _abi_contract())
    assert admitted["direct_resident_outer"] == "direct_resident_outer_172"
    assert admitted["event_state"] == "event_state_31"
    assert admitted["full_tick_state"] == "full_tick_state_29"
    assert admitted["ordinary_birth_state"] == "ordinary_birth_legacy_113"


def test_post_dispatch_receipt_exposes_retained_per_cell_scale_vector() -> None:
    prepared = _prepared("cpu")
    prepared.source.patch_p2_config[
        "patch_movement_cost_energy_scale_by_cell"
    ] = torch.tensor([0.0, 1.0, 16.0], dtype=torch.float64)

    receipt = observe_prepared_direct_resident_abi_v1(
        prepared,
        phase="post_dispatch",
    )
    patch_config = next(
        row for row in receipt["outer_arguments"] if row["name"] == "patch_p2_config"
    )
    vector = patch_config["scalar_vectors"][
        "patch_movement_cost_energy_scale_by_cell"
    ]
    assert vector["dtype"] == "float64"
    assert vector["shape"] == [BATCH]
    assert vector["values"] == [0.0, 1.0, 16.0]


def test_runtime_receipt_accepts_hybrid_lifecycle_without_turnover_state() -> None:
    prepared = _prepared(
        "cpu",
        generation_turnover_enabled=False,
        lifecycle_mode=1,
    )

    receipt = observe_prepared_direct_resident_abi_v1(prepared)
    rows = _by_path(receipt)
    outer = {row["name"]: row for row in receipt["outer_arguments"]}

    assert outer["turnover_state_tensors"]["nested_count"] == 0
    assert outer["turnover_shadow_tensors"]["nested_count"] == 0
    assert outer["turnover_pointer_table_tensors"]["nested_count"] == 0
    assert not any(path.startswith("turnover_state_tensors.") for path in rows)
    assert not any(path.startswith("turnover_shadow_tensors.") for path in rows)
    assert not any(
        path.startswith("turnover_pointer_table_tensors.") for path in rows
    )
    assert any(path.startswith("turnover_output_tensors.") for path in rows)
    admitted = validate_runtime_abi_order_v1(receipt, _abi_contract())
    assert admitted["turnover_mutable_state"] == "turnover_disabled_0"


def test_runtime_abi_admits_only_compact_replay_backed_policy_post_state() -> None:
    rows = [{} for _ in range(90)]
    rows[86] = {"shape": [8, 160, 1]}

    assert _admits_replay_backed_policy_post_rng_state_v1(
        family="ordinary_birth_state",
        index=89,
        dimension=2,
        expected=2560,
        actual=0,
        actual_shape=[8, 160, 0],
        observed_rows=rows,
    )

    rows[86] = {"shape": [8, 160, 9415]}
    assert not _admits_replay_backed_policy_post_rng_state_v1(
        family="ordinary_birth_state",
        index=89,
        dimension=2,
        expected=2560,
        actual=0,
        actual_shape=[8, 160, 0],
        observed_rows=rows,
    )


def test_runtime_receipt_fails_closed_on_non_tensor_nested_member() -> None:
    prepared = _prepared("cpu")
    prepared.source.event_state_tensors[0] = object()
    with pytest.raises(AuthoritativeStateContractError) as caught:
        observe_prepared_direct_resident_abi_v1(prepared)
    assert caught.value.code == "E_STATE_RUNTIME_ABI"
    assert caught.value.path == "event_state_tensors.event_tape_type"


def test_runtime_abi_owner_resolution_accepts_only_sanctioned_wrappers() -> None:
    trajectory = _prepared("cpu")
    patch = PreparedDirectResidentPatchMulticellV1(
        schema_version="test",
        patch=trajectory,
        cell_ids=("cell-0",),
        source_projections=({},),
        resumed=False,
    )
    with pytest.raises(
        CudaCrossPathResidentOverlayError,
        match="P12_RETIRED_INHERITANCE_AUTHORITY",
    ):
        PreparedCrossPathDirectResidentPatchMulticellV1(
            schema_version="test",
            patch=patch,
            controls=None,  # type: ignore[arg-type]
            linkage=None,  # type: ignore[arg-type]
            exposure=None,  # type: ignore[arg-type]
        )
    configured_wrappers = (
        PreparedDirectResidentWdlV2(
            patch=trajectory,
            state_tensors=(),
            schedule_tensors=(),
            reporting_tensors=(),
            config={},
            generation_rows=1,
            initial_projection={},
            cell_ids=("cell-0",),
        ),
        PreparedDirectResidentJungleV4(
            patch=trajectory,
            state_tensors=(),
            reporting_tensors=(),
            config={},
            logical_generation_count=1,
            initial_projection={},
        ),
        PreparedDirectResidentGruV3(
            patch=trajectory,
            gru_tensors=(),
            initial_projection={},
            observation_size=1,
        ),
        PreparedDirectResidentEedV1(
            patch=trajectory,
            state_tensors=(),
            config={},
            initial_projection={},
            cell_id="cell-0",
        ),
        PreparedDirectResidentWdlGruV3(
            patch=trajectory,
            gru_tensors=(),
            wdl_state_tensors=(),
            wdl_schedule_tensors=(),
            wdl_reporting_tensors=(),
            wdl_config={},
            initial_projection={},
            observation_size=1,
            generation_rows=1,
        ),
        PreparedDirectResidentGmnwGruV1(
            patch=trajectory,
            gru_tensors=(),
            gmnw_tensors=(),
            gru_config={},
            gmnw_config={},
            initial_projection={},
            base_observation_size=1,
            observation_size=1,
        ),
        PreparedDirectResidentWdlJungleV1(
            wdl=PreparedDirectResidentWdlV2(
                patch=trajectory,
                state_tensors=(),
                schedule_tensors=(),
                reporting_tensors=(),
                config={},
                generation_rows=1,
                initial_projection={},
                cell_ids=("cell-0",),
            ),
            jungle_state_tensors=(),
            jungle_reporting_tensors=(),
            jungle_config={},
        ),
    )

    assert resolve_prepared_direct_resident_trajectory_v1(trajectory) is trajectory
    assert resolve_prepared_direct_resident_trajectory_v1(patch) is trajectory
    assert all(
        resolve_prepared_direct_resident_trajectory_v1(wrapper) is trajectory
        for wrapper in configured_wrappers
    )
    assert (
        resolve_prepared_direct_resident_trajectory_v1(
            PreparedNativeMechanicsOwnerV1(
                physical_window_arguments=configured_wrappers[0],
                source_projections=({},),
                cell_ids=("cell-0",),
            )
        )
        is trajectory
    )

    with pytest.raises(
        AuthoritativeStateContractError,
        match="sanctioned typed wrapper",
    ):
        resolve_prepared_direct_resident_trajectory_v1(
            SimpleNamespace(patch=trajectory)
        )


def test_output_axis_symbol_is_not_the_physical_window_length() -> None:
    symbols = _shape_symbols(
        {
            "shape_formula_inputs": {
                "physical_window_ticks": 160,
                "output_tick_capacity": 1,
            }
        }
    )
    assert symbols["O"] == 1


def test_valid_counts_follow_generated_formula_not_b1_b2_shape_coincidence() -> None:
    grid = torch.zeros((2, 2, 2), dtype=torch.int32)
    count, semantics = _valid_count(
        path="live_state.agent_grid",
        tensor=grid,
        mutability="read_write",
        shape_inputs={
            "batch_cells": 2,
            "slot_capacity": 2,
            "active_count_by_cell": [1, 1],
        },
    )
    assert count == grid.numel()
    assert semantics == "route_active_prefix_or_full_storage"

    energy = torch.zeros((1, 1), dtype=torch.float32)
    count, semantics = _valid_count(
        path="live_state.energy",
        tensor=energy,
        mutability="read_write",
        shape_inputs={
            "batch_cells": 1,
            "slot_capacity": 1,
            "active_count_by_cell": [1],
        },
    )
    assert count == 1
    assert semantics == "route_active_prefix_or_full_storage"


def test_valid_counts_reject_unknown_non_output_contract() -> None:
    with pytest.raises(
        AuthoritativeStateContractError,
        match="has no generated valid-row contract",
    ):
        _valid_count(
            path="live_state.undeclared",
            tensor=torch.zeros((2, 2)),
            mutability="read_write",
            shape_inputs={
                "batch_cells": 2,
                "slot_capacity": 2,
                "active_count_by_cell": [1, 1],
            },
        )


def test_placeholder_prepared_owner_is_not_production_admissible() -> None:
    receipt = observe_prepared_direct_resident_abi_v1(_prepared("cpu"))

    with pytest.raises(
        AuthoritativeStateContractError,
        match=r"full_tick_state_tensors\.age: dtype must be int32",
    ):
        validate_runtime_abi_order_v1(
            receipt,
            _abi_contract(),
            require_production_admission=True,
            prepared=_prepared("cpu"),
        )


def test_unresolved_retained_field_is_not_production_admissible() -> None:
    receipt = observe_prepared_direct_resident_abi_v1(_prepared("cpu"))
    variant = copy.deepcopy(
        next(
            row
            for row in _abi_contract()["variants"]
            if row["variant_id"] == "patch_p2_state_full_10"
        )
    )
    variant["entries"][0]["metadata_status"] = "deferred_unsupported"

    with pytest.raises(
        AuthoritativeStateContractError,
        match="contains a non-exact metadata row",
    ):
        _validate_observed_variant_metadata_v1(
            family="patch_p2_state",
            variant=variant,
            receipt=receipt,
        )


@pytest.mark.parametrize("nested_count", [113, 122, 139])
def test_outer_abi_admits_separately_validated_birth_variants(
    nested_count: int,
) -> None:
    receipt = observe_prepared_direct_resident_abi_v1(_prepared("cpu"))
    receipt["outer_arguments"][68]["nested_count"] = nested_count
    variant = next(
        row
        for row in _abi_contract()["variants"]
        if row["variant_id"] == "direct_resident_outer_172"
    )

    _validate_observed_variant_metadata_v1(
        family="direct_resident_outer",
        variant=variant,
        receipt=receipt,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("semantic_owner", "counterfeit_owner_v1"),
        ("dtype", "float64"),
        ("physical_type", "counterfeit_physical_type"),
        ("bit_width", 64),
        ("rank", 2),
        ("shape_formula", "[B,H,H]"),
        ("valid_row_count_formula", "B"),
        ("capacity_formula", "B"),
        ("valid_range", "counterfeit_range"),
        ("sentinels", ["counterfeit_sentinel"]),
        ("mutability", "read_only"),
        ("checkpoint_participation", "excluded"),
        ("chronicle_participation", "excluded"),
        ("provider_participation", "excluded"),
        ("analysis_participation", "excluded"),
    ],
)
def test_production_metadata_drift_fails_before_dispatch(
    field: str,
    value: object,
) -> None:
    receipt = observe_prepared_direct_resident_abi_v1(_prepared("cpu"))
    variant = copy.deepcopy(
        next(
            row
            for row in _abi_contract()["variants"]
            if row["variant_id"] == "patch_p2_state_full_10"
        )
    )
    variant["entries"][0][field] = value

    with pytest.raises(
        AuthoritativeStateContractError,
        match=rf"patch_p2_state\[0\]\.{field}",
    ):
        _validate_observed_variant_metadata_v1(
            family="patch_p2_state",
            variant=variant,
            receipt=receipt,
        )


@pytest.mark.parametrize(
    ("entry_kind", "physical_bit_width"),
    [
        ("bool", 1),
        ("ordered_tensor_container", 8),
        ("ordered_mapping", 8),
    ],
)
def test_physical_bool_and_container_width_drift_fails_before_dispatch(
    entry_kind: str,
    physical_bit_width: int,
) -> None:
    receipt = observe_prepared_direct_resident_abi_v1(_prepared("cpu"))
    variant = copy.deepcopy(
        next(
            row
            for row in _abi_contract()["variants"]
            if row["variant_id"] == "direct_resident_outer_172"
        )
    )
    entry = next(
        row
        for row in variant["entries"]
        if (
            row["dtype"] == "bool"
            if entry_kind == "bool"
            else row["authority_contract"].get("kind") == entry_kind
        )
    )
    entry["authority_contract"]["bit_width"] = physical_bit_width

    with pytest.raises(
        AuthoritativeStateContractError,
        match=r"direct_resident_outer\[[0-9]+\]\.bit_width",
    ):
        _validate_observed_variant_metadata_v1(
            family="direct_resident_outer",
            variant=variant,
            receipt=receipt,
        )


def test_small_int_vector_admits_b160_and_rejects_b257() -> None:
    assert _small_int_vector(
        torch.arange(160, dtype=torch.int64),
        path="prepared.source.active_count_state",
    ) == list(range(160))
    with pytest.raises(AuthoritativeStateContractError, match="at most 256 values"):
        _small_int_vector(
            torch.arange(257, dtype=torch.int64),
            path="prepared.source.active_count_state",
        )
