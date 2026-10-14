from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import engine.perf.cuda_architecture_b_direct_resident_gru_v3 as gru_v3
from engine.perf.cuda_architecture_b_direct_resident_gru_v3 import (
    GRU_DISABLED_PREDICTION_STATE_FIELDS_V3,
    CudaArchitectureBDirectResidentGruV3Error,
    PreparedDirectResidentGruV3,
    PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V3,
    PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V3,
    PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3,
    TICKS_ARGUMENT_INDEX_V3,
    _canonical_pred_lr_exact_cpu_v3,
    _materialize_disabled_prediction_state_v3,
    execute_direct_resident_gru_v3,
    execute_direct_resident_gru_physical_v3,
    build_prepared_direct_resident_gru_v3,
    project_direct_resident_gru_checkpoint_v3,
    stable_gru_packed_width_v3,
)
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    GENERATION_COUNT_ARGUMENT_INDEX_V1,
)
from engine.perf.cuda_architecture_b_direct_resident_source_builder_v1 import (
    DirectResidentSourceAllocationRequestV1,
)


class _Control:
    def __init__(self, run_id: str, owner_epoch: int) -> None:
        self.state = {
            "run_id": run_id,
            "owner_epoch": owner_epoch,
            "status": "initialized",
            "run_started": False,
            "sealed": False,
        }

    def request_cancel(self, *_args: Any) -> None: ...

    def acknowledge_cancellation_checkpoint(self, *_args: Any) -> None: ...

    def acknowledge_run_complete(self, *_args: Any) -> None: ...

    def publish_failure(self, *_args: Any) -> None: ...

    def final_result(self) -> None: ...

    def manifest(self) -> dict[str, Any]:
        return dict(self.state)


class _Module:
    RunControlV2 = _Control

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def eiel_cuda_architecture_b_execute_direct_resident_gru_v3(
        self,
        control: _Control,
        *_arguments: Any,
    ) -> dict[str, Any]:
        generations = int(_arguments[-5])
        self.calls.append((control, *_arguments))
        control.state.update(
            {
                "status": "generation_complete",
                "run_started": True,
                "active_generation": generations - 1,
                "requested_generations": generations,
                "awaiting_final_output_ack": True,
            }
        )
        return {
            "logical_generation_count": generations,
            "window_count": (generations + 1) // 2,
            "physical_window_alignment": "absolute_generation_lattice_v1",
            "completed_generations": generations,
            "external_python_window_boundaries": 0,
            "direct_cpp_window_core_dispatch": True,
            "production_native_chunk_loop": True,
            "hidden_fallback": False,
            "runtime_jit": False,
            "gru_native_owner": True,
            "gru_native_controller_abi_version": 3,
            "gru_python_tick_or_generation_ownership": False,
        }

    def eiel_cuda_architecture_b_execute_direct_resident_gru_physical_v3(
        self,
        gru_tensors: list[torch.Tensor],
        gru_config: dict[str, float],
        *arguments: Any,
    ) -> dict[str, Any]:
        self.calls.append((gru_tensors, gru_config, *arguments))
        return {
            "gru_native_owner": True,
            "gru_native_controller_abi_version": 3,
            "gru_physical_gate_entrypoint": True,
            "gru_python_tick_or_generation_ownership": False,
            "hidden_fallback": False,
            "runtime_jit": False,
        }


def _prepared(generations: int) -> PreparedDirectResidentGruV3:
    metadata = {
        "policy_weight_mutation_enable": False,
        "policy_weight_mutation_mode": "disabled",
        "policy_weight_mutation_rate": 0.0,
        "policy_weight_mutation_std": 0.0,
        "gru_cost_base": 0.004,
        "gru_cost_per_dim": 0.00008,
    }
    patch = SimpleNamespace(
        source=SimpleNamespace(
            logical_generation_count=generations,
            generation_base=0,
        ),
        arguments=(object(),) * 160,
    )
    return PreparedDirectResidentGruV3(
        patch=patch,
        gru_tensors=(
            torch.empty((1, 1, stable_gru_packed_width_v3(74))),
            torch.empty((1, 1, 519)),
            torch.zeros((1,), dtype=torch.int32),
        ),
        initial_projection=SimpleNamespace(metadata=metadata),
        observation_size=74,
    )


def _physical_prepared() -> PreparedDirectResidentGruV3:
    prepared = _prepared(4)
    arguments: list[Any] = [object()] * 160
    arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3] = torch.empty(
        (1, 320, 5), dtype=torch.int64
    )
    arguments[PHYSICAL_FULL_TICK_OUTPUTS_ARGUMENT_INDEX_V3] = [
        torch.empty((1, 320)),
        torch.empty((4, 28)),
    ]
    arguments[PHYSICAL_PATCH_STATS_ARGUMENT_INDEX_V3] = [
        torch.empty((1, 320, 8))
    ]
    arguments[TICKS_ARGUMENT_INDEX_V3] = 80
    arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] = 2
    return PreparedDirectResidentGruV3(
        patch=SimpleNamespace(
            source=prepared.patch.source,
            arguments=tuple(arguments),
        ),
        gru_tensors=prepared.gru_tensors,
        initial_projection=prepared.initial_projection,
        observation_size=prepared.observation_size,
    )


@pytest.mark.parametrize("generations", (3, 4, 7, 8, 12, 32, 37, 100, 128))
def test_execute_direct_gru_owns_every_resident_window(generations: int) -> None:
    module = _Module()

    result = execute_direct_resident_gru_v3(
        module,
        _prepared(generations),
        logical_generation_count=generations,
        run_id=f"gru-g{generations}",
    )

    assert result["completed_generations"] == generations
    assert result["external_python_window_boundaries"] == 0
    assert result["gru_python_tick_or_generation_ownership"] is False
    assert len(module.calls) == 1
    assert module.calls[0][2] == {
        "upkeep_cost_base": 0.004,
        "upkeep_cost_per_dim": 0.00008,
    }


def test_execute_direct_gru_accepts_durable_mapping_projection() -> None:
    module = _Module()
    prepared = _prepared(7)
    prepared = replace(
        prepared,
        initial_projection={"metadata": dict(prepared.initial_projection.metadata)},
    )

    result = execute_direct_resident_gru_v3(
        module,
        prepared,
        logical_generation_count=7,
        run_id="gru-durable-projection",
    )

    assert result["completed_generations"] == 7


@pytest.mark.parametrize("generations", (0, 2, 129))
def test_execute_direct_gru_rejects_unqualified_horizon(generations: int) -> None:
    module = _Module()

    with pytest.raises(
        CudaArchitectureBDirectResidentGruV3Error,
        match=r"logical_generation_count in \[3,128\]",
    ):
        execute_direct_resident_gru_v3(
            module,
            _prepared(4),
            logical_generation_count=generations,
            run_id="gru-invalid",
        )

    assert module.calls == []


def test_stable_gru_pack_width_matches_frozen_layout() -> None:
    assert stable_gru_packed_width_v3(74) == 30215
    assert stable_gru_packed_width_v3(83) == 30791


def test_public_multicell_requests_preserve_order_and_irregular_horizon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families = ("gru", "mlp", "mixed")
    requests = tuple(
        DirectResidentSourceAllocationRequestV1(
            envelope=object(),
            qualified_cuda_module=object(),
            cuda_capability={},
            import_projection={
                "metadata": {
                    "generation": 9,
                    "tick": 0,
                    "native_runtime_fixed_policy_family": family,
                    "gru_cost_base": 0.004,
                    "gru_cost_per_dim": 0.00008,
                    "policy_weight_mutation_enable": False,
                    "policy_weight_mutation_mode": "disabled",
                    "policy_weight_mutation_rate": 0.0,
                    "policy_weight_mutation_std": 0.0,
                },
                "tensors": {},
            },
            cuda_boundary=object(),
            generation_birth_state=object(),
            configuration=object(),
            cuda_device="cuda",
        )
        for family in families
    )
    captured: dict[str, Any] = {}

    def allocator(first: Any, *, per_cell_requests: Any = None) -> Any:
        captured["first"] = first
        captured["requests"] = per_cell_requests
        return SimpleNamespace(
            batch_cells=3,
            logical_generation_count=37,
            live_state={
                "energy": torch.zeros((3, 2)),
                "gru_pack": torch.zeros((3, 2, stable_gru_packed_width_v3(74))),
            },
            outputs={"observations": torch.zeros((3, 1, 2, 74))},
        )

    monkeypatch.setattr(gru_v3, "_validated_capability", lambda _module: {})
    monkeypatch.setattr(gru_v3, "allocate_direct_resident_source_v1", allocator)
    monkeypatch.setattr(
        gru_v3, "prepare_direct_resident_trajectory_v1", lambda source: SimpleNamespace(source=source)
    )
    shared_pack = torch.zeros((1, 1, stable_gru_packed_width_v3(74)))

    prepared = build_prepared_direct_resident_gru_v3(
        qualified_cuda_module=object(),
        per_cell_requests=requests,
        cell_ids=("a", "b", "c"),
        logical_generation_count=37,
        agent_id_allocation_generation_count=37,
        initial_generation=9,
        observation_extra_columns=9,
        output_ticks=80,
        immutable_shared_gru_pack=shared_pack,
    )

    assert captured["requests"] is not None
    assert tuple(item.import_projection for item in captured["requests"]) == tuple(
        item.import_projection for item in requests
    )
    assert prepared.cell_ids == ("a", "b", "c")
    assert prepared.source_projections == tuple(item.import_projection for item in requests)
    assert tuple(
        projection["metadata"]["native_runtime_fixed_policy_family"]
        for projection in prepared.source_projections
    ) == families
    assert prepared.batch_cells == 3
    assert all(item.observation_extra_columns == 9 for item in captured["requests"])
    assert all(item.output_ticks == 80 for item in captured["requests"])
    assert all(
        item.immutable_shared_gru_pack is shared_pack
        for item in captured["requests"]
    )


def test_physical_multicell_projection_materializes_active_event_rows() -> None:
    columns = tuple(
        torch.arange(12, dtype=torch.int64).reshape(3, 4) + index * 100
        for index, _name in enumerate(gru_v3.EVENT_COLUMN_NAMES)
    )
    prepared = SimpleNamespace(
        patch=SimpleNamespace(
            live_banks={
                "event_columns": columns,
                "event_active_count": torch.tensor([1, 3, 2]),
                "event_status": torch.zeros((3,), dtype=torch.int32),
            }
        )
    )

    histories, counts = gru_v3._materialize_active_event_histories_v3(
        prepared, cell_count=3
    )

    assert counts == (1, 3, 2)
    assert tuple(len(history) for history in histories) == (len(columns),) * 3
    assert tuple(history[0].numel() for history in histories) == counts


@pytest.mark.parametrize("generations", (1, 2))
def test_execute_physical_gru_owns_short_segment(generations: int) -> None:
    module = _Module()

    result = execute_direct_resident_gru_physical_v3(
        module,
        _physical_prepared(),
        generation_count=generations,
    )

    arguments = module.calls[0][2:]
    assert result["gru_physical_gate_entrypoint"] is True
    assert arguments[GENERATION_COUNT_ARGUMENT_INDEX_V1] == generations
    assert arguments[TICKS_ARGUMENT_INDEX_V3] == 80
    assert arguments[PHYSICAL_RUNTIME_BOOKKEEPING_ARGUMENT_INDEX_V3].shape[1] == (
        generations * 80
    )


def test_disabled_prediction_state_preserves_survivors_and_resets_births() -> None:
    prediction_tensors: dict[str, torch.Tensor] = {}
    row_fields = {
        "exp_dE",
        "exp_dInj",
        "prediction_queue_action",
        "prediction_queue_due",
        "prediction_queue_energy",
        "prediction_queue_exp_de",
        "prediction_queue_exp_dinj",
        "prediction_queue_injury",
    }
    int_fields = {
        "episode0_pred_err_count",
        "late_pred_err_count",
        "pred_err_count",
        "prediction_queue_action",
        "prediction_queue_count",
        "prediction_reset_count",
        "prediction_update_count",
    }
    int64_fields = {"prediction_queue_due"}
    exact_fields = {name for name in GRU_DISABLED_PREDICTION_STATE_FIELDS_V3 if name.endswith("_exact")}
    for name in GRU_DISABLED_PREDICTION_STATE_FIELDS_V3:
        shape = (3, 2) if name in row_fields else (3,)
        dtype = (
            torch.int64
            if name in int64_fields
            else torch.int32
            if name in int_fields
            else torch.float64
            if name in exact_fields
            else torch.float32
        )
        prediction_tensors[name] = torch.ones(shape, dtype=dtype)
    prediction_tensors["pred_lr_exact"] = torch.tensor(
        [0.25, 0.5, 0.0], dtype=torch.float64
    )
    prediction_tensors["slot_to_agent_id"] = torch.tensor(
        [10, 11, -1], dtype=torch.int64
    )
    prepared = PreparedDirectResidentGruV3(
        patch=SimpleNamespace(),
        gru_tensors=(),
        initial_projection=SimpleNamespace(
            metadata={
                "ablate_prediction": True,
                "force_prediction_disabled": True,
            },
            tensors=prediction_tensors,
        ),
        observation_size=74,
    )
    persistent = {
        "slot_to_agent_id": torch.tensor([[10, 12, -1]], dtype=torch.int64),
        "alive_mask": torch.tensor([[True, True, False]]),
        "pred_lr": torch.tensor([[0.25, 0.123456789, 0.75]]),
        "survival_steps": torch.tensor([[320, 17, 0]], dtype=torch.int32),
    }

    settled = _materialize_disabled_prediction_state_v3(prepared, persistent)

    assert settled["exp_dE"][0, 0].tolist() == [0.0, 0.0]
    assert settled["exp_dE"][0, 1].tolist() == [0.0, 0.0]
    assert settled["prediction_queue_due"][0, 1].tolist() == [-1, -1]
    assert settled["prediction_queue_action"][0, 2].tolist() == [-1, -1]
    assert settled["prediction_reset_count"].tolist() == [[80, 17, 0]]
    assert settled["pred_lr_exact"].tolist() == [[0.25, 0.12345679, 0.0]]


def test_prediction_rate_checkpoint_canonicalization_matches_decimal_contract() -> None:
    result = _canonical_pred_lr_exact_cpu_v3(
        torch.tensor([0.10669521987438202, 0.25, 0.75]),
        torch.tensor([True, True, False]),
    )

    assert result.tolist() == [0.10669522, 0.25, 0.0]


def test_checkpoint_projection_retains_absolute_source_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, Any] = {}
    patch = SimpleNamespace(
        source=SimpleNamespace(
            logical_generation_count=4,
            generation_base=8,
            outputs={},
        ),
        arguments=(object(),) * 160,
        live_banks={
            "agent_id_to_slot": torch.tensor(
                [[0, -1, -1, -1, -1]], dtype=torch.int32
            )
        },
        checkpoint_projection_inputs={
            "pre_generation_canonical_projection": {},
            "resident_birth_state": object(),
            "genome_v2_checkpoint_authority_binding": (("binding", "value"),),
            "genome_v2_causal_physical_cell_ordinal": 7,
        },
    )
    prepared = PreparedDirectResidentGruV3(
        patch=patch,
        gru_tensors=(),
        initial_projection=SimpleNamespace(metadata={}, tensors={}),
        observation_size=74,
        cell_ids=("causal-cell",),
    )

    monkeypatch.setattr(gru_v3, "_with_materialized_event_history", lambda *_: {})
    monkeypatch.setattr(
        gru_v3,
        "_materialize_disabled_prediction_state_v3",
        lambda *_: {},
    )

    def fake_project(*_args: Any, **kwargs: Any) -> SimpleNamespace:
        recorded.update(kwargs)
        return SimpleNamespace(
            projection={
                "metadata": {},
                "tensors": {
                    "agent_id_to_slot": torch.tensor([0], dtype=torch.int32),
                    "pred_lr": torch.tensor([0.25]),
                    "alive_mask": torch.tensor([True]),
                },
            }
        )

    monkeypatch.setattr(gru_v3, "_project_terminal_checkpoint", fake_project)
    monkeypatch.setattr(
        gru_v3,
        "_with_checkpoint_contract_metadata",
        lambda projection: projection,
    )

    projection = project_direct_resident_gru_checkpoint_v3(
        prepared,
        direct_result={
            "generation_summary_scalar": torch.empty((1, 1, 1)),
            "runtime_bookkeeping_by_tick": torch.empty((1, 80, 1)),
        },
        completed_generation_count=1,
    )

    assert recorded["source_generation"] == 8
    assert recorded["inheritance_sidecar_cell_id"] == "causal-cell"
    assert recorded["genome_v2_checkpoint_authority_binding"] == (
        ("binding", "value"),
    )
    assert recorded["genome_v2_causal_physical_cell_ordinal"] == 7
    assert (
        projection["metadata"][
            gru_v3.RESIDENT_AGENT_ID_CAPACITY_METADATA_V1
        ]
        == 5
    )


def test_multicell_checkpoint_sources_accept_private_ragged_histories() -> None:
    projections = tuple(
        {
            "schema_version": "checkpoint-v1",
            "metadata": {
                "generation": 4,
                "tick": 0,
                "agent_grid_runner_phase": "generation_boundary",
            },
            "tensors": {
                "alive_mask": torch.ones(4, dtype=torch.bool),
                "agent_id_to_slot": torch.arange(ids, dtype=torch.int32),
                "event_tape_actor_id": torch.arange(events, dtype=torch.int64),
            },
        }
        for ids, events in ((235, 562), (234, 585))
    )

    validated = gru_v3._validated_source_projections_v3(
        projections,
        cell_count=2,
        source_generation=4,
        direct_result={},
    )

    assert tuple(row.tensors["agent_id_to_slot"].numel() for row in validated) == (
        235,
        234,
    )
    assert tuple(row.tensors["event_tape_actor_id"].numel() for row in validated) == (
        562,
        585,
    )


@pytest.mark.parametrize("initial_generation", (-1, 1.5, True))
def test_gru_preparation_rejects_invalid_initial_generation(
    initial_generation: Any,
) -> None:
    with pytest.raises(
        CudaArchitectureBDirectResidentGruV3Error,
        match="initial_generation must be a nonnegative exact integer",
    ):
        build_prepared_direct_resident_gru_v3(
            object(),
            qualified_cuda_module=object(),
            initial_generation=initial_generation,
        )


def test_checkpoint_restore_preserves_authoritative_gru_upkeep_cost() -> None:
    target = torch.zeros((1, 4), dtype=torch.float32)
    checkpoint = torch.tensor([0.00912, 0.00656, 0.0, 0.004], dtype=torch.float32)
    source = SimpleNamespace(live_state={"gru_upkeep_cost": target})

    gru_v3._restore_checkpoint_gru_upkeep_cost_v3(
        source,
        {"tensors": {"gru_upkeep_cost": checkpoint}},
    )

    assert torch.equal(target[0], checkpoint)
    assert target.data_ptr() != checkpoint.data_ptr()


def test_checkpoint_restore_rejects_invalid_gru_upkeep_cost() -> None:
    source = SimpleNamespace(
        live_state={"gru_upkeep_cost": torch.zeros((1, 4), dtype=torch.float32)}
    )

    with pytest.raises(
        CudaArchitectureBDirectResidentGruV3Error,
        match="shape or dtype mismatch",
    ):
        gru_v3._restore_checkpoint_gru_upkeep_cost_v3(
            source,
            {"tensors": {"gru_upkeep_cost": torch.zeros(4, dtype=torch.float64)}},
        )


def test_checkpoint_runtime_fields_preserve_manifest_and_event_cursor() -> None:
    checkpoint = {
        name: index
        for index, name in enumerate(gru_v3.CANONICAL_RUNTIME_PREFIX_FIELDS)
    }
    checkpoint.update(
        {
            "tensor_manifest": {"alive_mask": {"shape": [4]}},
            "event_cursor": 17,
            "unrelated": "excluded",
        }
    )

    fields = gru_v3._checkpoint_runtime_fields_v3(checkpoint)

    assert fields["tensor_manifest"] is checkpoint["tensor_manifest"]
    assert fields["event_cursor"] == 17
    assert "unrelated" not in fields
