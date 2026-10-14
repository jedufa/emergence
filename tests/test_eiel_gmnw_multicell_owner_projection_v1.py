from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_gmnw_gru_v1 as gmnw
from engine.perf import cuda_architecture_b_direct_resident_gru_v3 as gru
from engine.perf.cuda_architecture_b_direct_resident_preparation_v1 import (
    DirectResidentSourceV1,
    PreparedDirectResidentTrajectoryV1,
)


def _source_projection(cell_id: str, value: int, *, width: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        schema_version=gru.CHECKPOINT_PROJECTION_SCHEMA_VERSION,
        metadata={
            "generation": 0,
            "tick": 0,
            "agent_grid_runner_phase": "generation_boundary",
            "cell_id": cell_id,
        },
        tensors={"identity": torch.full((width,), value, dtype=torch.int64)},
    )


def _patch(
    *, cells: int = 2, generations: int = 2
) -> PreparedDirectResidentTrajectoryV1:
    live_banks = {
        "state": torch.tensor([[10, 11], [20, 21]], dtype=torch.int64)[:cells],
        "event_columns": [
            torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int64)[:cells]
        ],
    }
    outputs = {
        "generation_summary_scalar": torch.zeros(
            (generations, cells, 3), dtype=torch.float64
        ),
        "runtime_bookkeeping_by_tick": torch.zeros(
            (cells, generations * 80, 5), dtype=torch.int64
        ),
        "status": torch.zeros((cells,), dtype=torch.int32),
    }
    source = DirectResidentSourceV1(
        configuration=object(),
        active_agent_ids=live_banks["state"],
        active_agent_slots=torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.int32
        )[:cells],
        active_count_state=torch.full((cells,), 2, dtype=torch.int64),
        live_state=live_banks,
        outputs=outputs,
        full_tick_state_tensors=(),
        full_tick_output_tensors=(),
        compact_summary_reporting_tensors=(),
        compact_summary_checkpoint_source_tensors=(),
        compact_summary_checkpoint_descriptor_specs=(),
        compact_summary_checkpoint_output_tensors=(),
        compact_summary_settlement_tensors=(),
        event_state_tensors=(),
        ordinary_birth_state_tensors=(),
        ordinary_birth_output_tensors=(),
        ordinary_birth_scratch_tensors=(),
        turnover_selection_seed_by_cell=torch.tensor(
            tuple(range(cells)), dtype=torch.uint64
        ),
        turnover_state_tensors=(),
        turnover_shadow_tensors=(),
        turnover_output_tensors=(),
        turnover_pointer_table_tensors=(),
        patch_p2_state_tensors=(),
        patch_p2_scratch_tensors=(),
        patch_p2_stats_tensors=(),
        patch_p2_config={},
        live_banks=live_banks,
        checkpoint_projection_inputs={
            "pre_generation_canonical_projection": {},
            "resident_birth_state": object(),
        },
        deterministic=True,
        generation_base=0,
        logical_generation_count=generations,
        batch_cells=cells,
    )
    return PreparedDirectResidentTrajectoryV1(
        source=source,
        arguments=(),
        physical_generation_summary=outputs["generation_summary_scalar"],
        physical_runtime_bookkeeping=outputs["runtime_bookkeeping_by_tick"],
        physical_patch_stats=(),
        physical_full_tick_output_tensors=(),
        live_banks=live_banks,
        checkpoint_projection_inputs=source.checkpoint_projection_inputs,
    )


def _direct_result(*, cells: int = 2, generations: int = 2) -> dict[str, Any]:
    return {
        "completed_generations": generations,
        "source_projection_cell_ids": [f"cell-{cell}" for cell in range(cells)],
        "generation_summary_scalar": torch.arange(
            generations * cells * 3, dtype=torch.float64
        ).reshape(generations, cells, 3),
        "runtime_bookkeeping_by_tick": torch.arange(
            cells * generations * 80 * 5, dtype=torch.int64
        ).reshape(cells, generations * 80, 5),
        "event_histories": [
            [torch.arange(cell + 1, dtype=torch.int64)] for cell in range(cells)
        ],
        "event_rows_by_cell": [cell + 1 for cell in range(cells)],
    }


def _gru_prepared(*, cells: int = 2) -> gru.PreparedDirectResidentGruV3:
    patch = _patch(cells=cells)
    return gru.PreparedDirectResidentGruV3(
        patch=patch,
        gru_tensors=(),
        initial_projection=_source_projection("cell-0", 0),
        observation_size=1,
    )


def test_multicell_projection_slices_ordered_b1_views_without_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _gru_prepared()
    direct = _direct_result()
    sources = (_source_projection("cell-0", 100), _source_projection("cell-1", 200))
    calls: list[tuple[Any, dict[str, Any]]] = []

    def project_one(
        cell_prepared: Any,
        *,
        direct_result: Any,
        checkpoint_policy_family: str | None,
        **_: Any,
    ) -> dict[str, Any]:
        cell = len(calls)
        calls.append((cell_prepared, direct_result))
        canonical = cell_prepared.patch.checkpoint_projection_inputs[
            "pre_generation_canonical_projection"
        ]
        assert cell_prepared.patch.source.batch_cells == 1
        assert cell_prepared.patch.live_banks["state"].tolist() == [
            [10 + 10 * cell, 11 + 10 * cell]
        ]
        assert direct_result["generation_summary_scalar"].shape == (2, 1, 3)
        assert direct_result["runtime_bookkeeping_by_tick"].shape == (1, 160, 5)
        assert len(direct_result["event_histories"]) == 1
        assert direct_result["event_rows_by_cell"] == [cell + 1]
        assert canonical["metadata"]["cell_id"] == f"cell-{cell}"
        expected_family = ("gru", "mlp")[cell]
        assert checkpoint_policy_family == expected_family
        assert canonical["metadata"]["native_runtime_fixed_policy_family"] == "mixed"
        assert canonical["metadata"]["capability_mask"]["gru"] is True
        assert canonical["tensors"]["identity"].tolist() == [100 + 100 * cell] * 2
        return {
            "metadata": {"cell_id": f"cell-{cell}"},
            "tensors": {"state": cell_prepared.patch.live_banks["state"]},
        }

    monkeypatch.setattr(gru, "project_direct_resident_gru_checkpoint_v3", project_one)
    projected = gru.project_direct_resident_gru_multicell_checkpoints_v3(
        prepared,
        direct_result=direct,
        completed_generation_count=2,
        ordered_source_projections=sources,
        checkpoint_policy_families=("gru", "mlp"),
    )

    assert len(calls) == len(projected) == 2
    first = projected[0]["tensors"]["state"]
    second = projected[1]["tensors"]["state"]
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    first[0, 0] = -999
    assert prepared.patch.live_banks["state"][0, 0].item() == 10
    assert second[0, 0].item() == 20


@pytest.mark.parametrize("failure", ("count", "order", "layout"))
def test_multicell_projection_fails_closed_before_b1_projection(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    prepared = _gru_prepared()
    direct = _direct_result()
    sources: tuple[Any, ...] = (
        _source_projection("cell-0", 100),
        _source_projection("cell-1", 200),
    )
    if failure == "count":
        sources = sources[:1]
    elif failure == "order":
        sources = tuple(reversed(sources))
    else:
        sources = (sources[0], _source_projection("cell-1", 200, width=3))
    calls = 0

    def project_one(*_: Any, **__: Any) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(gru, "project_direct_resident_gru_checkpoint_v3", project_one)
    with pytest.raises(gru.CudaArchitectureBDirectResidentGruV3Error):
        gru.project_direct_resident_gru_multicell_checkpoints_v3(
            prepared,
            direct_result=direct,
            completed_generation_count=2,
            ordered_source_projections=sources,
        )
    assert calls == 0


def test_gmnw_multicell_projection_applies_public_pack_and_cache_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_observation_size = 1
    observation_size = base_observation_size + gmnw.GMNW_OBSERVATION_COLUMNS_V1
    patch = _patch()
    alive = torch.tensor([[True, True], [True, False]])
    policy_use_gru = torch.tensor([[True, False], [True, False]])
    patch.live_banks.update(
        {
            "alive_mask": alive,
            "policy_use_gru": policy_use_gru,
        }
    )
    shared = torch.ones(
        (gmnw.stable_gru_packed_width_v3(observation_size),), dtype=torch.float32
    )
    prepared = gmnw.PreparedDirectResidentGmnwGruV1(
        patch=patch,
        gru_tensors=(shared,),
        gmnw_tensors=(),
        gru_config={"immutable_shared_policy": True},
        gmnw_config={
            "policy_arm_codes": (
                gmnw.GMNW_POLICY_ARM_CODES_V1[("persistent_gru", False)],
                gmnw.GMNW_POLICY_ARM_CODES_V1[("matched_mlp", False)],
            )
        },
        initial_projection=_source_projection("cell-0", 0),
        base_observation_size=base_observation_size,
        observation_size=observation_size,
    )
    original_mlp = torch.arange(
        64 * base_observation_size + 64 + 64 * 64 + 64 + 7 * 64 + 7,
        dtype=torch.float32,
    ).reshape(1, -1)
    expanded_mlp = gmnw._expand_fixed_mlp_pack(
        original_mlp, base_observation_size=base_observation_size
    )
    expanded_gru = torch.zeros(
        (1, gmnw.stable_gru_packed_width_v3(observation_size)), dtype=torch.float32
    )
    raw = tuple(
        {
            "metadata": {"world": "patch", "policy_weight_mutation_enable": False},
            "tensors": {
                "mlp_pack": expanded_mlp.clone(),
                "gru_pack": expanded_gru.clone(),
                "action_cache_token": torch.tensor([4, 5], dtype=torch.int64),
                "action_cache_cursor": torch.tensor([0, 1], dtype=torch.int32),
                "action_cache_next": torch.tensor([2, 3], dtype=torch.int32),
            },
        }
        for _ in range(2)
    )
    observed: dict[str, Any] = {}

    def project_many(owner: Any, **kwargs: Any) -> tuple[dict[str, Any], ...]:
        observed["owner"] = owner
        observed.update(kwargs)
        return raw

    metadata_calls = 0

    def checkpoint_contract(projection: Any) -> dict[str, Any]:
        nonlocal metadata_calls
        metadata_calls += 1
        return dict(projection)

    monkeypatch.setattr(
        gmnw, "project_direct_resident_gru_multicell_checkpoints_v3", project_many
    )
    monkeypatch.setattr(gmnw, "_with_checkpoint_contract_metadata", checkpoint_contract)
    sources = (_source_projection("cell-0", 100), _source_projection("cell-1", 200))
    projected = gmnw.project_direct_resident_gmnw_gru_multicell_checkpoints_v1(
        prepared,
        direct_result={},
        completed_generation_count=2,
        ordered_source_projections=sources,
    )

    assert observed["ordered_source_projections"] is sources
    assert observed["pre_materialized_cpu_fields"] == frozenset({"gru_pack"})
    assert observed["checkpoint_policy_families"] == ("gru", "mlp")
    assert observed["owner"].patch.live_banks["gru_pack"].shape[:2] == (2, 2)
    assert metadata_calls == 2
    for projection, expected_family in zip(
        projected, ("gru", "mlp"), strict=True
    ):
        assert projection["metadata"]["native_runtime_fixed_policy_family"] == (
            expected_family
        )
        assert projection["metadata"]["capability_mask"]["gru"] is (
            expected_family == "gru"
        )
        tensors = projection["tensors"]
        assert torch.equal(tensors["mlp_pack"], original_mlp)
        assert tensors["gru_pack"].shape[-1] == gmnw.stable_gru_packed_width_v3(
            base_observation_size
        )
        assert tensors["action_cache_token"].tolist() == [4, -1]
        assert tensors["action_cache_cursor"].tolist() == [0, -1]
        assert tensors["action_cache_next"].tolist() == [2, -1]
    assert (
        projected[0]["tensors"]["mlp_pack"].untyped_storage().data_ptr()
        != projected[1]["tensors"]["mlp_pack"].untyped_storage().data_ptr()
    )


@pytest.mark.parametrize(
    ("arm", "visible", "expected"),
    [
        ("persistent_gru", False, "gru"),
        ("reset_gru", False, "gru"),
        ("persistent_gru", True, "gru"),
        ("matched_mlp", False, "mlp"),
        ("matched_mlp", True, "mlp"),
        ("mixed_selection", False, "mixed"),
        ("mixed_selection", True, "mixed"),
    ],
)
def test_checkpoint_policy_family_is_exact_per_arm(
    arm: str, visible: bool, expected: str
) -> None:
    code = gmnw.GMNW_POLICY_ARM_CODES_V1[(arm, visible)]

    assert gmnw._checkpoint_policy_family_for_arm_code_v1(code) == expected


@pytest.mark.parametrize(
    ("policy_family", "expected_gru_capability"),
    [("gru", True), ("mlp", False), ("mixed", True)],
)
def test_checkpoint_family_metadata_is_updated_atomically(
    policy_family: str, expected_gru_capability: bool
) -> None:
    metadata = gru._set_checkpoint_policy_family_metadata_v3(
        {
            "native_runtime_fixed_policy_family": "mixed",
            "capability_mask": {"patch": True, "gru": True},
        },
        policy_family,
    )

    assert metadata["native_runtime_fixed_policy_family"] == policy_family
    assert metadata["capability_mask"] == {
        "patch": True,
        "gru": expected_gru_capability,
    }


@pytest.mark.parametrize(
    ("policy_family", "expected_gru_capability"),
    [("gru", True), ("mlp", False), ("mixed", True)],
)
def test_checkpoint_finalizer_keeps_family_and_capability_atomic(
    monkeypatch: pytest.MonkeyPatch,
    policy_family: str,
    expected_gru_capability: bool,
) -> None:
    base_observation_size = 1
    original_mlp = torch.zeros(
        (1, 64 * base_observation_size + 64 + 64 * 64 + 64 + 7 * 64 + 7)
    )
    projection = {
        "metadata": {
            "policy_weight_mutation_enable": False,
            "native_runtime_fixed_policy_family": "mixed",
            "capability_mask": {"patch": True, "gru": True},
        },
        "tensors": {
            "mlp_pack": gmnw._expand_fixed_mlp_pack(
                original_mlp, base_observation_size=base_observation_size
            ),
            "gru_pack": torch.zeros(
                (
                    1,
                    gmnw.stable_gru_packed_width_v3(
                        base_observation_size + gmnw.GMNW_OBSERVATION_COLUMNS_V1
                    ),
                )
            ),
            "action_cache_token": torch.tensor([-1], dtype=torch.int64),
            "action_cache_cursor": torch.tensor([-1], dtype=torch.int32),
            "action_cache_next": torch.tensor([-1], dtype=torch.int32),
        },
    }
    monkeypatch.setattr(
        gmnw,
        "_with_checkpoint_contract_metadata",
        lambda value: dict(value),
    )

    finalized = gmnw._finalize_gmnw_gru_checkpoint_projection_v1(
        projection,
        base_observation_size=base_observation_size,
        policy_family=policy_family,
    )

    assert finalized["metadata"]["native_runtime_fixed_policy_family"] == policy_family
    assert finalized["metadata"]["capability_mask"]["gru"] is expected_gru_capability
