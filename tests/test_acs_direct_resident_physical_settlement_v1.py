"""CPU contract tests for detached direct-resident settlement."""

from __future__ import annotations

from dataclasses import is_dataclass, replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import engine.campaigns.acs_direct_resident_physical_settlement_v1 as settlement
from engine.campaigns.acs_direct_resident_physical_capture_v1 import (
    SCHEMA_VERSION as CAPTURE_SCHEMA_VERSION,
    SETTLEMENT_OUTPUT_CELL_AXES_V1,
    SETTLEMENT_RECIPE_SCHEMA_VERSION,
)
from engine.perf.cuda_architecture_b_boundary import EVENT_COLUMN_NAMES
from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (
    JUNGLE_STATE_FIELDS_V4,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
    WDL_STATE_FIELDS_V2,
)
from engine.perf.cuda_architecture_b_generation import CudaGenerationConfig
from engine.tw_rngv2.contract import NAMESPACE_DEFAULT as TW_RNGV2_NAMESPACE_DEFAULT
from engine.tw_rngv2.world_identity_v1 import (
    LOGICAL_WORLD_UID_DERIVATION_V1,
    logical_world_uid_v1,
)


def _component_recipe(
    path: Path,
    *,
    cell_ids: tuple[str, ...] = ("cell-0", "cell-1"),
    digest: str = "source-digest",
) -> dict[str, object]:
    file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema_version": "eiel_bounded_source_descriptor_recipe_v1",
        "cell_ids": list(cell_ids),
        "descriptors": [
            {
                "kind": "component",
                "path": str(path),
                "file_sha256": file_digest,
                "source_projection_sha256": digest,
            }
            for _ in cell_ids
        ],
        "source_projection_sha256s": {cell_id: digest for cell_id in cell_ids},
    }


def test_component_source_descriptor_round_trip_and_corruption_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source-component.pt"
    torch.save({"cell": "source", "generation": 3}, path)
    monkeypatch.setattr(settlement, "_source_projection_digest", lambda _: "source-digest")
    recipe = _component_recipe(path)

    loaded = settlement.load_source_projection_from_recipe_v1(
        recipe, cell_index=1, cell_id="cell-1"
    )
    assert loaded == {"cell": "source", "generation": 3}

    corrupted = dict(recipe)
    corrupted["descriptors"] = [dict(row) for row in recipe["descriptors"]]
    corrupted["descriptors"][1]["file_sha256"] = "0" * 64
    with pytest.raises(settlement.DirectResidentPhysicalSettlementError, match="file digest"):
        settlement.load_source_projection_from_recipe_v1(
            corrupted, cell_index=1, cell_id="cell-1"
        )

    corrupted = _component_recipe(path, digest="wrong-source-digest")
    with pytest.raises(settlement.DirectResidentPhysicalSettlementError, match="projection digest"):
        settlement.load_source_projection_from_recipe_v1(
            corrupted, cell_index=0, cell_id="cell-0"
        )

    corrupted = _component_recipe(path)
    corrupted["cell_ids"] = ["cell-1", "cell-1"]
    with pytest.raises(settlement.DirectResidentPhysicalSettlementError, match="cell order"):
        settlement.load_source_projection_from_recipe_v1(
            corrupted, cell_index=0, cell_id="cell-1"
        )


def test_cell_direct_result_reconstructs_selected_cell_and_output_axes() -> None:
    cells = 2
    loaded_fields = {
        "summary": torch.tensor([20, 21, 22], dtype=torch.int64),
        "runtime": torch.tensor([30, 40], dtype=torch.int64),
    }
    layout = {
        "settlement_outputs": {
            "generation_summary_scalar": {"field": "summary"},
            "runtime_bookkeeping_by_tick": {"field": "runtime"},
        }
    }
    recipe = {
        "direct_result_scalars": {
            "completed_generations": 1,
            "event_rows_by_cell": [["row-0"], ["row-1"]],
            "source_projection_cell_ids": ["cell-0", "cell-1"],
        },
        "source": {"logical_generation_count": 1},
    }

    result = settlement._cell_direct_result(
        recipe,
        cell=1,
        cell_id="cell-1",
        cells=cells,
        loaded_fields=loaded_fields,
        layout=layout,
        live_banks={},
    )

    assert result["event_rows_by_cell"] == [["row-1"]]
    assert result["source_projection_cell_ids"] == ["cell-1"]
    assert result["physical_window_generations"] == 1
    assert result["generation_summary_scalar"].shape == (3, 1)
    assert result["runtime_bookkeeping_by_tick"].shape == (1, 2)
    assert SETTLEMENT_OUTPUT_CELL_AXES_V1["generation_summary_scalar"] == 1
    assert SETTLEMENT_OUTPUT_CELL_AXES_V1["runtime_bookkeeping_by_tick"] == 0


def test_field_value_reconstructs_strided_alias_from_selected_storage_owner() -> None:
    selected_storage = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    descriptor = {
        "alias_of": "diploid_allele_storage",
        "view": {
            "storage_offset": 1,
            "shape": [2],
            "stride": [3],
        },
    }

    reconstructed = settlement._field_value(
        {"diploid_allele_storage": selected_storage}, descriptor
    )

    assert torch.equal(reconstructed, torch.tensor([1.0, 4.0]))
    assert reconstructed.stride() == (3,)


def test_partial_tick_runtime_bookkeeping_is_trimmed_to_exact_span() -> None:
    result = settlement._cell_direct_result(
        {
            "direct_result_scalars": {
                "absolute_tick_span_native_owner": True,
                "tick_span_count": 1,
            },
            "source": {"logical_generation_count": 1},
        },
        cell=0,
        cell_id="cell-0",
        cells=1,
        loaded_fields={"runtime": torch.arange(800).reshape(160, 5)},
        layout={
            "settlement_outputs": {
                "runtime_bookkeeping_by_tick": {"field": "runtime"}
            }
        },
        live_banks={},
    )

    assert result["runtime_bookkeeping_by_tick"].shape == (1, 1, 5)


def test_current_tick_native_recipe_verifies_raw_authority_and_private_adapter() -> None:
    scalars = {
        "schema_version": "eiel_cuda_architecture_b_native_resident_window_controller_v1",
        "owner": "native_extension",
        "absolute_tick_span_native_owner": True,
        "physical_window_generations": 1,
        "logical_generation_count": 7,
        "window_count": 7,
        "completed_generations": 1,
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "native_tick_iterations": 161,
        "source_window_ticks": 160,
        "source_window_count": 2,
        "tick_native_start_next_tick": 160,
        "tick_native_end_next_tick": 321,
        "tick_span_start": 0,
        "tick_span_end": 1,
        "tick_span_count": 1,
        "terminal_generation_boundary": False,
        "tick_native_source_prefix_by_cell": [
            {
                "metadata": {
                    "generation": 2,
                    "tick": 0,
                    "prediction_tick": 0,
                    "agent_grid_runner_phase": "generation_boundary",
                },
                "runtime": {},
            }
        ],
    }
    public_scalars = dict(scalars)
    for name in (
        "completed_generations",
        "logical_generation_count",
        "physical_window_generations",
        "terminal_generation_boundary",
        "tick_native_source_prefix_by_cell",
        "window_count",
    ):
        public_scalars.pop(name)
    public_scalars.update(
        {
            "physical_window_alignment": "absolute_tick_v1",
            "tick_span_start": 160,
            "tick_span_end": 321,
            "tick_span_count": 161,
        }
    )
    adapter = {
        "schema_version": "eiel_absolute_tick_private_settlement_adapter_v1",
        "authority": "private_legacy_compatibility_only",
        "derived_from_absolute_next_tick": 321,
        "interval_ticks": 160,
        "source_generation": 2,
        "source_tick": 0,
        "tick_span_start": 0,
        "tick_span_end": 1,
        "tick_span_count": 1,
        "terminal_generation_boundary": False,
        "private_settlement_direct_result_scalars": scalars,
        "private_settlement_source": {
            "generation_base": 0,
            "logical_generation_count": 7,
            "physical_window_generations": 1,
            "ticks_per_generation": 160,
        },
    }
    recipe = {
        "absolute_next_tick": 321,
        "absolute_tick_contract": {
            "schema_version": "eiel_absolute_tick_physical_capture_contract_v1",
            "time_authority": "absolute_tick_v1",
            "segment_start_next_tick": 160,
            "segment_end_next_tick": 321,
            "span_ticks": 161,
            "native_tick_iterations": 161,
            "source_window_ticks": 160,
            "source_window_count": 2,
        },
        "legacy_compatibility": adapter,
        "direct_result_scalars": public_scalars,
        "source": {
            "configuration": {},
            "time_authority": "absolute_tick_v1",
            "absolute_next_tick": 321,
        },
    }
    layout = {
        "settlement_outputs": {
            "runtime_bookkeeping_by_tick": {"field": "runtime"}
        }
    }
    result = settlement._cell_direct_result(
        recipe,
        cell=0,
        cell_id="cell-0",
        cells=1,
        loaded_fields={"runtime": torch.arange(800).reshape(160, 5)},
        layout=layout,
        live_banks={},
    )

    assert result["tick_native_start_next_tick"] == 160
    assert result["tick_native_end_next_tick"] == 321
    assert result["tick_span_start"] == 0
    assert result["tick_span_end"] == 1
    assert result["runtime_bookkeeping_by_tick"].shape == (1, 1, 5)

    with pytest.raises(
        settlement.DirectResidentPhysicalSettlementError,
        match="private coordinates are not tick-derived",
    ):
        settlement._cell_direct_result(
            {
                **recipe,
                "legacy_compatibility": {**adapter, "tick_span_count": 2},
            },
            cell=0,
            cell_id="cell-0",
            cells=1,
            loaded_fields={"runtime": torch.arange(800).reshape(160, 5)},
            layout=layout,
            live_banks={},
        )


@pytest.mark.parametrize(
    (
        "start",
        "end",
        "source_generation",
        "private_start",
        "private_end",
        "boundary",
        "source_window_count",
    ),
    [
        (3359, 3360, 20, 159, 160, True, 1),
        (3358, 3359, 20, 158, 159, False, 1),
        (3200, 3520, 21, 0, 160, True, 2),
        (3199, 3519, 21, 0, 159, False, 3),
    ],
)
def test_restore_validator_derives_boundary_aligned_private_coordinates(
    start: int,
    end: int,
    source_generation: int,
    private_start: int,
    private_end: int,
    boundary: bool,
    source_window_count: int,
) -> None:
    span = end - start
    private_count = private_end - private_start
    phase = "generation_boundary" if private_start == 0 else "partial_generation"
    public_scalars = {
        "absolute_tick_span_native_owner": True,
        "physical_window_alignment": "absolute_tick_v1",
        "tick_native_start_next_tick": start,
        "tick_native_end_next_tick": end,
        "tick_span_start": start,
        "tick_span_end": end,
        "tick_span_count": span,
        "native_tick_iterations": span,
        "source_window_ticks": 160,
        "source_window_count": source_window_count,
    }
    private_scalars = {
        "schema_version": settlement.RESULT_SCHEMA_VERSION,
        "owner": "native_extension",
        "absolute_tick_span_native_owner": True,
        "external_python_window_boundaries": 0,
        "direct_cpp_window_core_dispatch": True,
        "wdl_jungle_native_owner": True,
        "eed_native_owner": True,
        "production_native_chunk_loop": True,
        "hidden_fallback": False,
        "runtime_jit": False,
        "logical_generation_count": 7,
        "physical_window_generations": 1,
        "window_count": 7,
        "completed_generations": 1,
        "physical_window_alignment": "absolute_generation_lattice_v1",
        "tick_span_start": private_start,
        "tick_span_end": private_end,
        "tick_span_count": private_count,
        "terminal_generation_boundary": boundary,
        "tick_native_source_prefix_by_cell": [
            {
                "metadata": {
                    "generation": source_generation,
                    "tick": private_start,
                    "prediction_tick": private_start,
                    "agent_grid_runner_phase": phase,
                }
            }
        ],
    }
    recipe = {
        "absolute_next_tick": end,
        "absolute_tick_contract": {
            "schema_version": "eiel_absolute_tick_physical_capture_contract_v1",
            "time_authority": "absolute_tick_v1",
            "segment_start_next_tick": start,
            "segment_end_next_tick": end,
            "span_ticks": span,
            "native_tick_iterations": span,
            "source_window_ticks": 160,
            "source_window_count": source_window_count,
        },
        "legacy_compatibility": {
            "schema_version": "eiel_absolute_tick_private_settlement_adapter_v1",
            "authority": "private_legacy_compatibility_only",
            "derived_from_absolute_next_tick": end,
            "interval_ticks": 160,
            "source_generation": source_generation,
            "source_tick": private_start,
            "tick_span_start": private_start,
            "tick_span_end": private_end,
            "tick_span_count": private_count,
            "terminal_generation_boundary": boundary,
            "private_settlement_direct_result_scalars": private_scalars,
            "private_settlement_source": {
                "generation_base": 0,
                "logical_generation_count": 7,
                "physical_window_generations": 1,
                "ticks_per_generation": 160,
            },
        },
        "source": {
            "configuration": {},
            "time_authority": "absolute_tick_v1",
            "absolute_next_tick": end,
        },
        "direct_result_scalars": public_scalars,
    }

    settlement._validate_absolute_tick_private_adapter_v1(
        recipe,
        public_scalars=public_scalars,
        private_scalars=private_scalars,
        source_payload={
            "logical_generation_count": 7,
            "physical_window_generations": 1,
        },
    )
    restored = settlement._cell_direct_result(
        recipe,
        cell=0,
        cell_id="cell-0",
        cells=1,
        loaded_fields={
            "runtime": torch.ones((private_count, 5), dtype=torch.int64)
        },
        layout={
            "settlement_outputs": {
                "runtime_bookkeeping_by_tick": {"field": "runtime"}
            }
        },
        live_banks={},
    )
    assert restored["tick_span_start"] == private_start
    assert restored["tick_span_end"] == private_end
    assert restored["runtime_bookkeeping_by_tick"].shape == (1, private_count, 5)


def test_source_configuration_round_trip_preserves_dataclass_replace_authority() -> None:
    from engine.campaigns.acs_direct_resident_physical_capture_v1 import (
        _configuration_payload,
    )

    source = CudaGenerationConfig(
        steps=160,
        patch_enabled=True,
        hidden_size=0,
        generation=7,
        start_tick=13,
    )
    reconstructed = settlement._decode_source_configuration_v1(
        _configuration_payload(source)
    )

    assert reconstructed == source
    assert replace(reconstructed, generation=8).generation == 8


def test_checkpoint_scalar_decoder_restores_nested_tuple_authority() -> None:
    decoded = settlement._decode_checkpoint_scalar_tree_v1(
        {"binding": [["schema_version", "binding-v1"], ["slot", 2]]}
    )

    assert decoded == {
        "binding": (("schema_version", "binding-v1"), ("slot", 2))
    }


def _cursor_fixture(tmp_path: Path) -> tuple[dict[str, object], object, dict[str, torch.Tensor]]:
    cell_ids = ("cell-0", "cell-1")
    source_path = tmp_path / "source.pt"
    torch.save({"canonical": True}, source_path)
    source_recipe = _component_recipe(source_path, cell_ids=cell_ids)
    fields: dict[str, torch.Tensor] = {
        name: torch.full((2, 1), index, dtype=torch.float32)
        for index, name in enumerate(
            tuple(WDL_STATE_FIELDS_V2) + tuple(JUNGLE_STATE_FIELDS_V4)
        )
    }
    fields.update(
        {
            name: torch.ones((2, 1), dtype=torch.int64)
            for name in EVENT_COLUMN_NAMES
        }
    )
    fields.update(
        {
            "next-step": torch.tensor([1120, 1121], dtype=torch.int64),
            "agent-keys": torch.tensor([3, 4], dtype=torch.int64),
            "agent-flags": torch.tensor([1, 0], dtype=torch.int8),
            "summary": torch.tensor([[[5]], [[6]]], dtype=torch.int64),
        }
    )
    layout = {
        "live_bank_tensors": {
            name: {"field": name} for name in WDL_STATE_FIELDS_V2
        },
        "event_columns": {name: {"field": name} for name in EVENT_COLUMN_NAMES},
        "settlement_outputs": {
            "generation_summary_scalar": {"field": "summary"},
        },
        "wdl": {
            "next_absolute_step": {"field": "next-step"},
            "agent_keys": {"field": "agent-keys"},
            "agent_flags": {"field": "agent-flags"},
        },
        "jungle": {name: {"field": name} for name in JUNGLE_STATE_FIELDS_V4},
        "eed": {},
    }
    recipe = {
        "schema_version": SETTLEMENT_RECIPE_SCHEMA_VERSION,
        "cell_ids": list(cell_ids),
        "absolute_next_tick": 1120,
        "reporting_enabled": False,
        "generation_base_override": 0,
        "source_descriptor_recipe": source_recipe,
        "source": {
            "configuration": {"fixture": True, "nested": {"mode": "cpu"}},
            "generation_base": 0,
            "logical_generation_count": 1,
            "ticks_per_generation": 160,
        },
        "checkpoint_projection_inputs": {
            "resident_birth_state": {"kind": "mapping", "items": {}},
            "cuda_capability": {"major": 8},
        },
        "live_bank_scalars": {
            "genome_v2_founder_profiles": [
                {
                    "schema_version": "founder-v1",
                    "profile_id": f"founder-{cell}",
                    "payload_sha256": str(cell) * 64,
                }
                for cell in range(2)
            ]
        },
        "direct_result_scalars": {
            "completed_generations": 1,
            "source_projection_cell_ids": list(cell_ids),
            "tick_native_source_prefix_by_cell": [
                {"metadata": {"generation": 7}, "runtime": {}}
                for _ in cell_ids
            ],
        },
        "field_layout": layout,
    }
    capture = {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "settlement_recipe": recipe,
    }
    return capture, fields, layout


def test_rngv2_restore_rederives_logical_world_identity_before_loading_cells(
    tmp_path: Path,
) -> None:
    capture, _, _ = _cursor_fixture(tmp_path)
    recipe = capture["settlement_recipe"]
    cell_ids = tuple(recipe["cell_ids"])
    configured_world_uid = 17
    recipe["checkpoint_projection_inputs"].update(
        {
            "tw_rngv2_semantics_version": 2,
            "tw_rngv2_configured_world_uid": configured_world_uid,
            "tw_rngv2_namespace_key64": TW_RNGV2_NAMESPACE_DEFAULT,
            "tw_rngv2_semantic_epoch": 1,
            "tw_rngv2_logical_world_uid_derivation": (
                LOGICAL_WORLD_UID_DERIVATION_V1
            ),
            "tw_rngv2_logical_world_uids": [
                logical_world_uid_v1(
                    configured_world_uid=configured_world_uid,
                    cell_id=cell_id,
                )
                for cell_id in cell_ids
            ],
        }
    )
    recipe["checkpoint_projection_inputs"]["tw_rngv2_logical_world_uids"][1] ^= 1

    class Reader:
        def load_selected_cell(self, _index: int):
            raise AssertionError("identity failure must precede physical cell loading")

    with pytest.raises(
        settlement.DirectResidentPhysicalSettlementError,
        match="logical world identity disagrees with restored cell IDs",
    ):
        settlement.build_physical_batch_settlement_cursor_v1(
            capture, reader=Reader()
        )


def test_selected_cell_cursor_reconstructs_one_cell_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, fields, _ = _cursor_fixture(tmp_path)
    monkeypatch.setattr(settlement, "_source_projection_digest", lambda _: "source-digest")
    projected: list[str] = []

    def fake_settlement(combined, *, direct_result, **kwargs):
        cell_id = combined.wdl.cell_ids[0]
        assert is_dataclass(combined.wdl.patch)
        assert combined.wdl.patch.live_banks["genome_v2_founder_profiles"][0][
            "profile_id"
        ] == f"founder-{len(projected)}"
        assert kwargs["generation_base_override"] == 7
        assert len(direct_result["tick_native_source_prefix_by_cell"]) == 1
        projected.append(cell_id)
        return SimpleNamespace(
            checkpoint_projections=({"cell_id": cell_id},),
            compact_result_rows=(),
            segment_artifacts=(),
            lifecycle_qacct_receipts=(),
            terminal_generation=1,
            terminal_tick=1120,
        )

    monkeypatch.setattr(
        settlement,
        "project_direct_resident_wdl_jungle_multicell_settlement_v1",
        fake_settlement,
    )

    class Reader:
        def load_selected_cell(self, index: int):
            return {"fields": {name: value[index] for name, value in fields.items()}}

    cursor = settlement.build_physical_batch_settlement_cursor_v1(
        capture, reader=Reader()
    )
    first = cursor.next_cell()
    assert first is not None
    assert first.cell_id == "cell-0"
    assert first.require_projection() == {"cell_id": "cell-0"}
    cursor.release_cell("cell-0")
    second = cursor.next_cell()
    assert second is not None
    assert second.cell_id == "cell-1"
    assert second.require_projection() == {"cell_id": "cell-1"}
    cursor.release_cell("cell-1")
    assert cursor.next_cell() is None
    receipt = cursor.close()

    assert projected == ["cell-0", "cell-1"]
    assert receipt.covered_cell_ids == ("cell-0", "cell-1")
    assert cursor.aggregate_payload()["terminal_tick"] == 1120


def test_tick_native_cursor_attaches_exact_terminal_metric_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture, fields, _ = _cursor_fixture(tmp_path)
    capture["settlement_recipe"]["direct_result_scalars"].update(
        {
            "absolute_tick_span_native_owner": True,
            "tick_span_count": 1,
            "tick_span_end": 1,
        }
    )
    monkeypatch.setattr(settlement, "_source_projection_digest", lambda _: "source-digest")

    def fake_settlement(combined, *, direct_result, **kwargs):
        cell_id = combined.wdl.cell_ids[0]
        return SimpleNamespace(
            checkpoint_projections=({"cell_id": cell_id},),
            compact_result_rows=(),
            segment_artifacts=(),
            lifecycle_qacct_receipts=(),
            terminal_generation=7,
            terminal_tick=1,
        )

    projected: list[tuple[str, ...]] = []

    def fake_terminal_metrics(projections, *, cell_ids):
        projected.append(tuple(cell_ids))
        assert projections == ({"cell_id": cell_ids[0]},)
        return (
            {
                "cell_id": cell_ids[0],
                "metrics": {"runtime.terminal_absolute_tick": 1121},
            },
        )

    monkeypatch.setattr(
        settlement,
        "project_direct_resident_wdl_jungle_multicell_settlement_v1",
        fake_settlement,
    )
    monkeypatch.setattr(
        settlement,
        "_tick_native_terminal_metric_rows_v1",
        fake_terminal_metrics,
    )

    class Reader:
        def load_selected_cell(self, index: int):
            return {"fields": {name: value[index] for name, value in fields.items()}}

    cursor = settlement.build_physical_batch_settlement_cursor_v1(
        capture, reader=Reader()
    )
    for cell_id in ("cell-0", "cell-1"):
        detached = cursor.next_cell()
        assert detached is not None
        assert detached.compact_result_rows == (
            {
                "cell_id": cell_id,
                "metrics": {"runtime.terminal_absolute_tick": 1121},
            },
        )
        cursor.release_cell(cell_id)
    assert cursor.next_cell() is None
    cursor.close()

    assert projected == [("cell-0",), ("cell-1",)]
