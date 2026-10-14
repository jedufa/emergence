from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_gmnw_checkpoint_v1 as codec
from engine.perf import cuda_architecture_b_direct_resident_gmnw_gru_v1 as gmnw


CELLS = 4
CAPACITY = 5
GENERATIONS = 3
ABSOLUTE_GENERATION = 2
SEL1_ARM_CODES = (3, 6, 3, 6)
SEL1_CUE_VISIBILITY = (False, True, False, True)
LEGACY_VENV_PROFILE_CODES = (0, 0, 0, 0)
CELL_IDS = tuple(f"sel1-cell-{cell:02d}" for cell in range(CELLS))


def _core_config(*, generations: int = GENERATIONS) -> dict[str, Any]:
    return {
        "enabled": True,
        "generation_base": 0,
        "generation_rows_per_cell": generations,
        "cue_ticks": 1,
        "consequence_ticks": 1,
        "grace_generation_count": 4,
        "severe_generation_count": 4,
        "grace_damage_fraction": 0.15,
        "severe_damage_fraction": 0.45,
        "lethal_damage_fraction": 1.0,
        "grace_reproduction_scale": 0.5,
    }


def _sel1_config(*, generations: int = GENERATIONS) -> dict[str, Any]:
    return {
        **_core_config(generations=generations),
        "control_matrix_enabled": True,
        "policy_arm_codes": SEL1_ARM_CODES,
        "cue_visible_at_decision": SEL1_CUE_VISIBILITY,
        "gru_founder_count": (2, 2, 3, 3),
        "venv_consequence_profile_codes": LEGACY_VENV_PROFILE_CODES,
        "qualification_trace_enabled": False,
        "qualification_trace_output_ticks": 0,
        "qualification_decision_tape_enabled": False,
    }


def _modern_b3_config() -> dict[str, Any]:
    return {
        **_core_config(),
        "control_matrix_enabled": True,
        "policy_arm_codes": (0, 2, 0, 2),
        "cue_visible_at_decision": (False, False, False, False),
        "gru_founder_count": (None, None, None, None),
        "venv_consequence_profile_codes": LEGACY_VENV_PROFILE_CODES,
        "qualification_trace_enabled": False,
        "qualification_trace_output_ticks": 0,
        "qualification_decision_tape_enabled": False,
    }


def test_modern_control_arms_require_extended_checkpoint_identity() -> None:
    observed, extended = codec._validated_prepared_config(
        _modern_b3_config(), path="config"
    )

    assert extended is True
    assert observed["policy_arm_codes"] == [0, 2, 0, 2]

    persistent_cell = _modern_b3_config()
    persistent_cell["policy_arm_codes"] = (0, 0, 0, 0)
    _observed, persistent_extended = codec._validated_prepared_config(
        persistent_cell, path="config"
    )
    assert persistent_extended is True


def _selection_metrics(
    *, populated: bool, generations: int, completed_generations: int
) -> dict[str, torch.Tensor]:
    _, metrics = gmnw._allocate_gmnw_selection_metrics_v1(
        cells=CELLS,
        generations=generations,
        device=torch.device("cpu"),
    )
    if populated:
        metric_shape = (CELLS, completed_generations, 2)
        values = torch.arange(
            CELLS * completed_generations * 2, dtype=torch.int64
        ).reshape(metric_shape)
        for ordinal, name in enumerate(gmnw.GMNW_SELECTION_COUNT_METRIC_NAMES_V1, 1):
            metrics[name][:, :completed_generations].copy_(values + ordinal * 100)
        metrics["recurrent_upkeep_charge_total"][:, :completed_generations].copy_(
            values.to(torch.float64) / 8.0
        )
    return metrics


def _prepared(
    *,
    populated: bool,
    generations: int = GENERATIONS,
    selection_metrics: bool = False,
    completed_generations: int = ABSOLUTE_GENERATION,
) -> gmnw.PreparedDirectResidentGmnwGruV1:
    configs = tuple(
        gmnw.GmnwConfigurationV1(
            cue_alphabet=2 if cell < 2 else 4,
            cue_visible_at_decision=SEL1_CUE_VISIBILITY[cell],
            policy_arm="mixed_selection",
            gru_founder_count=2 if cell < 2 else 3,
            task_seed=101 + cell,
            venv_consequence_profile="legacy",
        )
        for cell in range(CELLS)
    )
    tensors = gmnw._allocate_gmnw_tensors(
        cells=CELLS,
        capacity=CAPACITY,
        generations=generations,
        configs=configs,
        device=torch.device("cpu"),
    )
    slot_to_agent_id = (
        torch.arange(CELLS * CAPACITY, dtype=torch.int64).reshape(CELLS, CAPACITY) + 100
    )
    alive_mask = torch.ones((CELLS, CAPACITY), dtype=torch.bool)
    family_tags = torch.tensor(
        [
            [True, False, True, False, False],
            [False, True, False, True, False],
            [True, True, False, False, True],
            [False, False, True, True, False],
        ],
        dtype=torch.bool,
    )
    if populated:
        tensors[0].copy_(slot_to_agent_id)
        cells_capacity = torch.arange(CELLS * CAPACITY).reshape(CELLS, CAPACITY)
        tensors[1].copy_(cells_capacity.to(torch.int64) + 20)
        tensors[2].copy_((cells_capacity % 4).to(torch.int8))
        tensors[3].copy_((cells_capacity % 3).to(torch.int8))
        tensors[4].copy_((cells_capacity + 30).to(torch.int32))
        tensors[5].copy_((cells_capacity + 40).to(torch.int32))
        tensors[6].copy_(cells_capacity % 2 == 0)
        for index in range(7, 11):
            tensors[index].copy_(((cells_capacity + index) % 5 - 1).to(torch.int8))
        tensors[11].copy_(cells_capacity.to(torch.int64) * 17 + 11)
        tensors[12].copy_(cells_capacity.to(torch.int64) * 19 + 12)
        tensors[13].copy_(cells_capacity.to(torch.float32) / 10.0 + 0.25)
        tensors[14].copy_(
            torch.arange(
                CELLS * generations * gmnw.GMNW_COUNTER_COLUMNS_V1,
                dtype=torch.int64,
            ).reshape(CELLS, generations, gmnw.GMNW_COUNTER_COLUMNS_V1)
        )
        tensors[15].copy_(
            torch.arange(CELLS * generations, dtype=torch.float64).reshape(
                CELLS, generations
            )
            / 8.0
        )

    control_tail = (
        torch.tensor(SEL1_ARM_CODES, dtype=torch.int8),
        torch.arange(gmnw.GMNW_MATCHED_MLP_PACKED_WIDTH_V1, dtype=torch.float32),
        torch.full(
            (CELLS, gmnw.GMNW_MATCHED_MLP_SCRATCH_WIDTH_V1),
            -7.0,
            dtype=torch.float32,
        ),
        torch.zeros((CELLS,), dtype=torch.int32),
    )
    patch = SimpleNamespace(
        source=SimpleNamespace(
            live_state={
                "alive_mask": alive_mask,
                "slot_to_agent_id": slot_to_agent_id,
                "policy_use_gru": family_tags.clone() if populated else ~family_tags,
            }
        )
    )
    return gmnw.PreparedDirectResidentGmnwGruV1(
        patch=patch,
        gru_tensors=(),
        gmnw_tensors=(*tensors, *control_tail),
        gru_config={},
        gmnw_config=_sel1_config(generations=generations),
        initial_projection=None,
        base_observation_size=74,
        observation_size=86,
        selection_metrics=(
            _selection_metrics(
                populated=populated,
                generations=generations,
                completed_generations=completed_generations,
            )
            if selection_metrics
            else {}
        ),
    )


def _family_tags(prepared: gmnw.PreparedDirectResidentGmnwGruV1) -> torch.Tensor:
    return prepared.patch.source.live_state["policy_use_gru"]


def _selection_metric_names() -> tuple[str, ...]:
    return (
        *gmnw.GMNW_SELECTION_METRIC_NAMES_V1,
        gmnw.GMNW_SELECTION_METRICS_STATUS_NAME_V1,
    )


def _with_resident_selection_tensors(
    prepared: gmnw.PreparedDirectResidentGmnwGruV1,
) -> gmnw.PreparedDirectResidentGmnwGruV1:
    metrics = prepared.selection_metrics
    counts = torch.stack(
        [metrics[name] for name in gmnw.GMNW_SELECTION_COUNT_METRIC_NAMES_V1],
        dim=-1,
    )
    upkeep = metrics["recurrent_upkeep_charge_total"]
    status = metrics[gmnw.GMNW_SELECTION_METRICS_STATUS_NAME_V1]
    resident_metrics = {
        name: counts[..., column]
        for column, name in enumerate(gmnw.GMNW_SELECTION_COUNT_METRIC_NAMES_V1)
    }
    resident_metrics["recurrent_upkeep_charge_total"] = upkeep
    resident_metrics[gmnw.GMNW_SELECTION_METRICS_STATUS_NAME_V1] = status
    return replace(
        prepared,
        gmnw_tensors=(
            *prepared.gmnw_tensors,
            counts,
            upkeep,
            status,
        ),
        selection_metrics=resident_metrics,
        gmnw_config={**prepared.gmnw_config, "selection_metrics_enabled": True},
    )


def _refresh_sidecar_integrity(sidecar: dict[str, Any]) -> None:
    manifest = [dict(row) for row in sidecar["tensor_manifest"]]
    sidecar["tensor_manifest"] = manifest
    for row in manifest:
        row["sha256"] = codec._tensor_sha256(sidecar["tensors"][row["name"]])
    sidecar["manifest_sha256"] = codec._canonical_sha256(
        manifest, path="sidecar.tensor_manifest"
    )
    sidecar["digest_sha256"] = codec._canonical_sha256(
        codec._digest_payload(sidecar), path="sidecar"
    )


def _refresh_multicell_integrity(checkpoint: dict[str, Any]) -> None:
    checkpoint["digest_sha256"] = codec._canonical_sha256(
        codec._multicell_digest_payload(checkpoint), path="multicell_checkpoint"
    )


def _as_historical_sel1_v1_sidecar(sidecar: dict[str, Any]) -> dict[str, Any]:
    metric_names = tuple(
        name for name in _selection_metric_names() if name in sidecar["tensors"]
    )
    names = (*codec.GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V1, *metric_names)
    tensors = tuple(sidecar["tensors"][name] for name in names)
    manifest = codec._manifest_for_tensors(names, tensors)
    config = dict(sidecar["gmnw_config"])
    config.pop("venv_consequence_profile_codes")
    historical: dict[str, Any] = {
        "schema_version": codec.GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V1,
        "codec_version": codec.GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_SEL1_V1,
        "absolute_generation": sidecar["absolute_generation"],
        "gmnw_config": config,
        "tensor_manifest": manifest,
        "manifest_sha256": codec._canonical_sha256(manifest, path="tensor_manifest"),
        "tensors": dict(zip(names, tensors, strict=True)),
    }
    historical["digest_sha256"] = codec._canonical_sha256(
        codec._digest_payload(historical), path="sidecar"
    )
    return historical


def _prepared_b72(*, populated: bool) -> gmnw.PreparedDirectResidentGmnwGruV1:
    cells = 72
    capacity = 1
    generations = 1
    cue_visibility = tuple(bool(cell % 2) for cell in range(cells))
    arm_codes = tuple(6 if visible else 3 for visible in cue_visibility)
    configs = tuple(
        gmnw.GmnwConfigurationV1(
            cue_alphabet=2,
            cue_visible_at_decision=cue_visibility[cell],
            policy_arm="mixed_selection",
            gru_founder_count=1,
            task_seed=10_000 + cell,
            venv_consequence_profile="legacy",
        )
        for cell in range(cells)
    )
    tensors = gmnw._allocate_gmnw_tensors(
        cells=cells,
        capacity=capacity,
        generations=generations,
        configs=configs,
        device=torch.device("cpu"),
    )
    slot_to_agent_id = torch.arange(cells, dtype=torch.int64).reshape(cells, 1) + 500
    alive_mask = torch.ones((cells, capacity), dtype=torch.bool)
    family_tags = (torch.arange(cells).reshape(cells, 1) % 3) != 0
    if populated:
        tensors[0].copy_(slot_to_agent_id)
        tensors[11].copy_(slot_to_agent_id * 31)
        tensors[12].copy_(slot_to_agent_id * 37)
    control_tail = (
        torch.tensor(arm_codes, dtype=torch.int8),
        torch.zeros(gmnw.GMNW_MATCHED_MLP_PACKED_WIDTH_V1, dtype=torch.float32),
        torch.zeros(
            (cells, gmnw.GMNW_MATCHED_MLP_SCRATCH_WIDTH_V1), dtype=torch.float32
        ),
        torch.zeros((cells,), dtype=torch.int32),
    )
    patch = SimpleNamespace(
        source=SimpleNamespace(
            live_state={
                "alive_mask": alive_mask,
                "slot_to_agent_id": slot_to_agent_id,
                "policy_use_gru": family_tags.clone() if populated else ~family_tags,
            }
        )
    )
    return gmnw.PreparedDirectResidentGmnwGruV1(
        patch=patch,
        gru_tensors=(),
        gmnw_tensors=(*tensors, *control_tail),
        gru_config={},
        gmnw_config={
            **_core_config(generations=generations),
            "control_matrix_enabled": True,
            "policy_arm_codes": arm_codes,
            "cue_visible_at_decision": cue_visibility,
            "gru_founder_count": (1,) * cells,
            "venv_consequence_profile_codes": (0,) * cells,
            "qualification_trace_enabled": False,
            "qualification_trace_output_ticks": 0,
            "qualification_decision_tape_enabled": False,
        },
        initial_projection=None,
        base_observation_size=74,
        observation_size=86,
    )


def test_sel1_multicell_export_restore_is_exact_and_unaliased() -> None:
    source = _prepared(populated=True)
    expected_task = tuple(tensor.clone() for tensor in source.gmnw_tensors[:22])
    expected_profile_codes = source.gmnw_tensors[22].clone()
    expected_arm_codes = source.gmnw_tensors[23].clone()
    expected_family_tags = _family_tags(source).clone()

    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=ABSOLUTE_GENERATION
    )

    assert (
        sidecar["schema_version"]
        == codec.GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V2
    )
    assert list(sidecar["tensors"]) == list(codec.GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2)
    assert sidecar["tensors"]["policy_arm_code"].tolist() == list(SEL1_ARM_CODES)
    assert sidecar["gmnw_config"]["cue_visible_at_decision"] == list(
        SEL1_CUE_VISIBILITY
    )

    assert sidecar["tensors"]["venv_consequence_profile_code"].tolist() == [0] * CELLS
    for tensor in source.gmnw_tensors[:24]:
        tensor.zero_()
    _family_tags(source).logical_not_()
    for index, name in enumerate(codec.GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2[:22]):
        assert torch.equal(sidecar["tensors"][name], expected_task[index])
    assert torch.equal(sidecar["tensors"]["policy_arm_code"], expected_arm_codes)
    assert torch.equal(sidecar["tensors"]["policy_use_gru"], expected_family_tags)

    target = _prepared(populated=False)
    target.gmnw_tensors[17].zero_()
    target.gmnw_tensors[18].fill_(-9)
    target.gmnw_tensors[19].fill_(True)
    target.gmnw_tensors[20].fill_(True)
    target.gmnw_tensors[21].fill_(-1)
    target.gmnw_tensors[23].copy_(torch.tensor(SEL1_ARM_CODES, dtype=torch.int8))
    packed_before = target.gmnw_tensors[24].clone()
    scratch_before = target.gmnw_tensors[25].clone()

    restored_generation = codec.restore_direct_resident_gmnw_checkpoint_v1(
        target, sidecar, absolute_generation=ABSOLUTE_GENERATION
    )

    assert restored_generation == ABSOLUTE_GENERATION
    for observed, expected in zip(target.gmnw_tensors[:22], expected_task, strict=True):
        assert torch.equal(observed, expected)
    assert torch.equal(target.gmnw_tensors[22], expected_profile_codes)
    assert torch.equal(target.gmnw_tensors[23], expected_arm_codes)
    assert torch.equal(_family_tags(target), expected_family_tags)
    assert torch.equal(target.gmnw_tensors[24], packed_before)
    assert torch.equal(target.gmnw_tensors[25], scratch_before)
    for index, name in enumerate(codec.GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2[:22]):
        assert (
            target.gmnw_tensors[index].data_ptr() != sidecar["tensors"][name].data_ptr()
        )
    assert (
        _family_tags(target).data_ptr()
        != sidecar["tensors"]["policy_use_gru"].data_ptr()
    )

    target.gmnw_tensors[11][0, 0] += 1
    _family_tags(target)[0, 0].logical_not_()
    assert torch.equal(sidecar["tensors"]["task_rng_state"], expected_task[11])
    assert torch.equal(sidecar["tensors"]["policy_use_gru"], expected_family_tags)


def test_historical_sel1_v1_sidecar_restores_as_legacy_profile_zero() -> None:
    source = _prepared(populated=True)
    current = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=ABSOLUTE_GENERATION
    )
    historical = _as_historical_sel1_v1_sidecar(current)
    target = _prepared(populated=False)

    restored_generation = codec.restore_direct_resident_gmnw_checkpoint_v1(
        target, historical, absolute_generation=ABSOLUTE_GENERATION
    )

    assert restored_generation == ABSOLUTE_GENERATION
    assert (
        historical["schema_version"]
        == codec.GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V1
    )
    assert "venv_consequence_profile_codes" not in historical["gmnw_config"]
    assert "venv_consequence_profile_code" not in historical["tensors"]
    assert not torch.count_nonzero(target.gmnw_tensors[22])
    assert torch.equal(target.gmnw_tensors[23], source.gmnw_tensors[23])
    assert torch.equal(_family_tags(target), _family_tags(source))


def test_historical_sel1_v1_metrics_restore_at_original_offsets() -> None:
    source = _prepared(
        populated=True,
        generations=2,
        selection_metrics=True,
        completed_generations=2,
    )
    current = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=2
    )
    historical = _as_historical_sel1_v1_sidecar(current)
    target = _prepared(
        populated=False,
        generations=2,
        selection_metrics=True,
        completed_generations=2,
    )

    codec.restore_direct_resident_gmnw_checkpoint_v1(
        target, historical, absolute_generation=2
    )

    assert list(historical["tensors"])[-len(_selection_metric_names()) :] == list(
        _selection_metric_names()
    )
    assert not torch.count_nonzero(target.gmnw_tensors[22])
    for name in _selection_metric_names():
        assert torch.equal(target.selection_metrics[name], source.selection_metrics[name])


def test_sel1_private_multicell_collection_roundtrip_preserves_order_and_state() -> (
    None
):
    source = _prepared(populated=True)
    expected_task = tuple(tensor.clone() for tensor in source.gmnw_tensors[:22])
    expected_profile_codes = source.gmnw_tensors[22].clone()
    expected_arm_codes = source.gmnw_tensors[23].clone()
    expected_family_tags = _family_tags(source).clone()

    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        source,
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )

    assert (
        checkpoint["schema_version"]
        == codec.GMNW_MULTICELL_CHECKPOINT_SCHEMA_VERSION_SEL1_V2
    )
    assert checkpoint["cell_count"] == CELLS
    assert checkpoint["cell_ids"] == list(CELL_IDS)
    assert len(checkpoint["sidecars"]) == CELLS
    occupied_storages: set[int] = set()
    for cell, sidecar in enumerate(checkpoint["sidecars"]):
        assert sidecar["gmnw_config"]["policy_arm_codes"] == [SEL1_ARM_CODES[cell]]
        assert sidecar["gmnw_config"]["cue_visible_at_decision"] == [
            SEL1_CUE_VISIBILITY[cell]
        ]
        assert sidecar["gmnw_config"]["gru_founder_count"] == [2 if cell < 2 else 3]
        assert list(sidecar["tensors"]) == list(
            codec.GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2
        )
        for index, name in enumerate(codec.GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2):
            tensor = sidecar["tensors"][name]
            assert tensor.size(0) == 1
            assert tensor.untyped_storage().data_ptr() not in occupied_storages
            occupied_storages.add(tensor.untyped_storage().data_ptr())
            if index < 22:
                assert torch.equal(tensor, expected_task[index][cell : cell + 1])
            elif name == "policy_arm_code":
                assert torch.equal(tensor, expected_arm_codes[cell : cell + 1])
            elif name == "policy_use_gru":
                assert torch.equal(tensor, expected_family_tags[cell : cell + 1])
            else:
                assert name == "venv_consequence_profile_code"
                assert torch.equal(tensor, expected_profile_codes[cell : cell + 1])
        assert torch.equal(
            sidecar["tensors"]["task_rng_state"],
            expected_task[11][cell : cell + 1],
        )
        assert torch.equal(
            sidecar["tensors"]["task_rng_cursor"],
            expected_task[12][cell : cell + 1],
        )

    for tensor in source.gmnw_tensors[:24]:
        tensor.zero_()
    _family_tags(source).logical_not_()

    target = _prepared(populated=False)
    restored_generation = codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
        target,
        checkpoint,
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )

    assert restored_generation == ABSOLUTE_GENERATION
    for observed, expected in zip(target.gmnw_tensors[:22], expected_task, strict=True):
        assert torch.equal(observed, expected)
    assert torch.equal(target.gmnw_tensors[22], expected_profile_codes)
    assert torch.equal(target.gmnw_tensors[23], expected_arm_codes)
    assert torch.equal(_family_tags(target), expected_family_tags)
    target.gmnw_tensors[11][0, 0] += 1
    assert torch.equal(
        checkpoint["sidecars"][0]["tensors"]["task_rng_state"],
        expected_task[11][0:1],
    )
    assert torch.equal(target.gmnw_tensors[11][1:], expected_task[11][1:])


def test_sel1_b72_private_collection_roundtrip_is_ordered_and_exact_on_cpu() -> None:
    source = _prepared_b72(populated=True)
    cell_ids = tuple(f"sel1-b72-{cell:02d}" for cell in range(72))

    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        source,
        absolute_generation=1,
        cell_ids=cell_ids,
    )

    assert checkpoint["cell_count"] == 72
    assert checkpoint["cell_ids"] == list(cell_ids)
    assert len(checkpoint["sidecars"]) == 72
    for cell, sidecar in enumerate(checkpoint["sidecars"]):
        assert torch.equal(
            sidecar["tensors"]["task_rng_state"],
            source.gmnw_tensors[11][cell : cell + 1],
        )
        assert torch.equal(
            sidecar["tensors"]["task_rng_cursor"],
            source.gmnw_tensors[12][cell : cell + 1],
        )

    target = _prepared_b72(populated=False)
    codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
        target,
        checkpoint,
        absolute_generation=1,
        cell_ids=cell_ids,
    )

    for observed, expected in zip(
        target.gmnw_tensors[:24], source.gmnw_tensors[:24], strict=True
    ):
        assert torch.equal(observed, expected)
    assert torch.equal(_family_tags(target), _family_tags(source))


def test_sel1_multicell_continuation_expands_only_future_aggregate_rows() -> None:
    source_generations = 2
    target_generations = 5
    source = _prepared(populated=True, generations=source_generations)
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        source,
        absolute_generation=source_generations,
        cell_ids=CELL_IDS,
    )
    target = _prepared(populated=False, generations=target_generations)

    restored_generation = codec.continue_direct_resident_gmnw_multicell_checkpoint_v1(
        target,
        checkpoint,
        absolute_generation=source_generations,
        cell_ids=CELL_IDS,
    )

    assert restored_generation == source_generations
    for index in range(14):
        assert torch.equal(target.gmnw_tensors[index], source.gmnw_tensors[index])
    for index in (14, 15):
        assert torch.equal(
            target.gmnw_tensors[index][:, :source_generations],
            source.gmnw_tensors[index],
        )
        assert not torch.count_nonzero(
            target.gmnw_tensors[index][:, source_generations:]
        )
    for index in range(16, 24):
        assert torch.equal(target.gmnw_tensors[index], source.gmnw_tensors[index])
    assert torch.equal(_family_tags(target), _family_tags(source))


def test_sel1_selection_metrics_continue_exact_prefix_and_zero_future_rows() -> None:
    source_generations = 2
    target_generations = 5
    source = _prepared(
        populated=True,
        generations=source_generations,
        selection_metrics=True,
        completed_generations=source_generations,
    )
    expected_metrics = {
        name: tensor.clone() for name, tensor in source.selection_metrics.items()
    }
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        source,
        absolute_generation=source_generations,
        cell_ids=CELL_IDS,
    )

    expected_names = (
        *codec.GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V2,
        *_selection_metric_names(),
    )
    occupied_storages: set[int] = set()
    for cell, sidecar in enumerate(checkpoint["sidecars"]):
        assert tuple(sidecar["tensors"]) == expected_names
        for name in _selection_metric_names():
            tensor = sidecar["tensors"][name]
            assert tensor.untyped_storage().data_ptr() not in occupied_storages
            occupied_storages.add(tensor.untyped_storage().data_ptr())
            assert torch.equal(
                tensor,
                expected_metrics[name][cell : cell + 1],
            )

    target = _prepared(
        populated=False,
        generations=target_generations,
        selection_metrics=True,
        completed_generations=source_generations,
    )
    for name in gmnw.GMNW_SELECTION_METRIC_NAMES_V1:
        target.selection_metrics[name].fill_(99)

    restored_generation = codec.continue_direct_resident_gmnw_multicell_checkpoint_v1(
        target,
        checkpoint,
        absolute_generation=source_generations,
        cell_ids=CELL_IDS,
    )

    assert restored_generation == source_generations
    for name in gmnw.GMNW_SELECTION_METRIC_NAMES_V1:
        observed = target.selection_metrics[name]
        assert torch.equal(
            observed[:, :source_generations], expected_metrics[name]
        )
        assert not torch.count_nonzero(observed[:, source_generations:])
        assert observed.data_ptr() != checkpoint["sidecars"][0]["tensors"][
            name
        ].data_ptr()
    assert not torch.count_nonzero(target.selection_metrics["status"])


def test_sel1_resident_30_tensor_metrics_export_and_continue_exactly() -> None:
    source = _with_resident_selection_tensors(
        _prepared(
            populated=True,
            selection_metrics=True,
            completed_generations=ABSOLUTE_GENERATION,
        )
    )
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        source,
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )
    target = _with_resident_selection_tensors(
        _prepared(
            populated=False,
            selection_metrics=True,
            completed_generations=ABSOLUTE_GENERATION,
        )
    )

    restored_generation = codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
        target,
        checkpoint,
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )

    assert restored_generation == ABSOLUTE_GENERATION
    for name in _selection_metric_names():
        assert torch.equal(target.selection_metrics[name], source.selection_metrics[name])


@pytest.mark.parametrize("mismatch", ("enabled_without_30", "30_without_enabled"))
def test_sel1_metrics_flag_and_resident_abi_must_agree(mismatch: str) -> None:
    if mismatch == "enabled_without_30":
        prepared = _prepared(populated=True, selection_metrics=True)
        prepared = replace(
            prepared,
            gmnw_config={**prepared.gmnw_config, "selection_metrics_enabled": True},
        )
    else:
        prepared = _with_resident_selection_tensors(
            _prepared(populated=True, selection_metrics=True)
        )
        prepared = replace(
            prepared,
            gmnw_config={**prepared.gmnw_config, "selection_metrics_enabled": False},
        )

    with pytest.raises(
        codec.DirectResidentGmnwCheckpointV1Error,
        match="selection_metrics_enabled",
    ):
        codec.export_direct_resident_gmnw_checkpoint_v1(
            prepared,
            absolute_generation=ABSOLUTE_GENERATION,
        )


def test_sel1_selection_metrics_reject_bad_source_status_or_future_rows() -> None:
    nonzero_status = _prepared(
        populated=True,
        selection_metrics=True,
        completed_generations=ABSOLUTE_GENERATION,
    )
    nonzero_status.selection_metrics["status"][1] = 7
    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="status"):
        codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
            nonzero_status,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )

    nonzero_future = _prepared(
        populated=True,
        selection_metrics=True,
        completed_generations=ABSOLUTE_GENERATION,
    )
    nonzero_future.selection_metrics["decision_count"][:, ABSOLUTE_GENERATION] = 1
    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="future rows"):
        codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
            nonzero_future,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "field set mismatch"),
        ("extra", "field set mismatch"),
        ("dtype", "dtype"),
        ("shape", "shape"),
        ("layout", "contiguous"),
        ("digest", "digest"),
    ),
)
def test_sel1_selection_metrics_reject_field_shape_dtype_layout_or_digest_drift(
    mutation: str, message: str
) -> None:
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _prepared(
            populated=True,
            selection_metrics=True,
            completed_generations=ABSOLUTE_GENERATION,
        ),
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )
    changed = dict(checkpoint)
    changed["sidecars"] = list(checkpoint["sidecars"])
    sidecar = dict(changed["sidecars"][0])
    sidecar["tensors"] = dict(sidecar["tensors"])
    changed["sidecars"][0] = sidecar
    name = "decision_count"
    tensor = sidecar["tensors"][name]
    if mutation == "missing":
        del sidecar["tensors"][name]
    elif mutation == "extra":
        sidecar["tensors"]["ordinary_birth_count"] = tensor.clone()
    elif mutation == "dtype":
        sidecar["tensors"][name] = tensor.to(torch.int32)
    elif mutation == "shape":
        sidecar["tensors"][name] = tensor[:, :, :1]
    elif mutation == "layout":
        sidecar["tensors"][name] = torch.zeros(
            (1, GENERATIONS, 4), dtype=torch.int64
        )[:, :, ::2]
    else:
        tensor[0, 0, 0] += 1

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match=message):
        codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
            _prepared(populated=False, selection_metrics=True),
            changed,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )


def test_sel1_selection_metrics_reject_rehashed_nonzero_future_rows() -> None:
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _prepared(populated=True, selection_metrics=True),
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )
    sidecar = checkpoint["sidecars"][0]
    sidecar["tensors"]["correct_count"][0, ABSOLUTE_GENERATION, 1] = 1
    _refresh_sidecar_integrity(sidecar)
    _refresh_multicell_integrity(checkpoint)

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="future rows"):
        codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
            _prepared(populated=False, selection_metrics=True),
            checkpoint,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )


@pytest.mark.parametrize(("source_metrics", "target_metrics"), ((True, False), (False, True)))
def test_sel1_selection_metrics_presence_must_match_exactly(
    source_metrics: bool, target_metrics: bool
) -> None:
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _prepared(populated=True, selection_metrics=source_metrics),
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )
    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="presence"):
        codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
            _prepared(populated=False, selection_metrics=target_metrics),
            checkpoint,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )


def test_sel1_selection_metrics_continuation_rejects_shrinking_rows() -> None:
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _prepared(populated=True, selection_metrics=True),
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )
    target = _prepared(
        populated=False,
        generations=ABSOLUTE_GENERATION,
        selection_metrics=True,
        completed_generations=ABSOLUTE_GENERATION,
    )
    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="cannot shrink"):
        codec.continue_direct_resident_gmnw_multicell_checkpoint_v1(
            target,
            checkpoint,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )


def test_sel1_multicell_restore_rejects_count_order_and_metadata_drift() -> None:
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _prepared(populated=True),
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )
    missing = dict(checkpoint)
    missing["sidecars"] = checkpoint["sidecars"][:-1]
    reordered = dict(checkpoint)
    reordered["sidecars"] = list(reversed(checkpoint["sidecars"]))
    target = _prepared(populated=False)
    before = tuple(tensor.clone() for tensor in target.gmnw_tensors[:24])
    family_before = _family_tags(target).clone()

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="exactly"):
        codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
            target,
            missing,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )
    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="order"):
        codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
            target,
            reordered,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )
    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="count/order"):
        codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
            target,
            checkpoint,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=tuple(reversed(CELL_IDS)),
        )

    drifted_config = dict(target.gmnw_config)
    drifted_config["gru_founder_count"] = (1, 2, 3, 3)
    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="config"):
        codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
            replace(target, gmnw_config=drifted_config),
            checkpoint,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )

    for observed, expected in zip(target.gmnw_tensors[:24], before, strict=True):
        assert torch.equal(observed, expected)
    assert torch.equal(_family_tags(target), family_before)


@pytest.mark.parametrize(
    ("name", "replacement", "message"),
    (
        ("schema_version", "unexpected_sel1_v2", "schema"),
        ("task_seed", lambda tensor: tensor.to(torch.int32), "dtype"),
        ("task_seed", lambda tensor: tensor[:, None], "shape"),
    ),
)
def test_sel1_multicell_restore_rejects_nested_schema_dtype_or_shape_drift(
    name: str, replacement: Any, message: str
) -> None:
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _prepared(populated=True),
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )
    changed = dict(checkpoint)
    changed["sidecars"] = list(checkpoint["sidecars"])
    sidecar = dict(changed["sidecars"][1])
    changed["sidecars"][1] = sidecar
    if name == "schema_version":
        sidecar[name] = replacement
    else:
        sidecar["tensors"] = dict(sidecar["tensors"])
        sidecar["tensors"][name] = replacement(sidecar["tensors"][name])

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match=message):
        codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
            _prepared(populated=False),
            changed,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )


def test_sel1_multicell_restore_rejects_cross_cell_storage_alias() -> None:
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _prepared(populated=True),
        absolute_generation=ABSOLUTE_GENERATION,
        cell_ids=CELL_IDS,
    )
    changed = dict(checkpoint)
    changed["sidecars"] = list(checkpoint["sidecars"])
    sidecar = dict(changed["sidecars"][1])
    sidecar["tensors"] = dict(sidecar["tensors"])
    sidecar["tensors"]["task_status"] = checkpoint["sidecars"][0]["tensors"][
        "task_status"
    ]
    changed["sidecars"][1] = sidecar

    with pytest.raises(
        codec.DirectResidentGmnwCheckpointV1Error, match="storage alias"
    ):
        codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
            _prepared(populated=False),
            changed,
            absolute_generation=ABSOLUTE_GENERATION,
            cell_ids=CELL_IDS,
        )


def test_sel1_export_rejects_cue_or_arm_identity_drift() -> None:
    prepared = _prepared(populated=True)
    drifted_config = dict(prepared.gmnw_config)
    drifted_config["cue_visible_at_decision"] = (True, True, False, True)

    with pytest.raises(
        codec.DirectResidentGmnwCheckpointV1Error,
        match="contradicts the policy arm code",
    ):
        codec.export_direct_resident_gmnw_checkpoint_v1(
            replace(prepared, gmnw_config=drifted_config),
            absolute_generation=ABSOLUTE_GENERATION,
        )

    prepared.gmnw_tensors[23][0] = 6
    with pytest.raises(
        codec.DirectResidentGmnwCheckpointV1Error,
        match="contradict prepared.gmnw_config",
    ):
        codec.export_direct_resident_gmnw_checkpoint_v1(
            prepared, absolute_generation=ABSOLUTE_GENERATION
        )


def test_sel1_restore_rejects_digest_drift_without_mutation() -> None:
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        _prepared(populated=True), absolute_generation=ABSOLUTE_GENERATION
    )
    sidecar["tensors"]["task_seed"][0] += 1
    target = _prepared(populated=False)
    task_before = tuple(tensor.clone() for tensor in target.gmnw_tensors[:24])
    family_before = _family_tags(target).clone()

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="digest"):
        codec.restore_direct_resident_gmnw_checkpoint_v1(
            target, sidecar, absolute_generation=ABSOLUTE_GENERATION
        )

    for observed, expected in zip(target.gmnw_tensors[:24], task_before, strict=True):
        assert torch.equal(observed, expected)
    assert torch.equal(_family_tags(target), family_before)


def test_legacy_v1_sidecar_is_rejected_for_modern_control_target() -> None:
    source = _prepared(populated=True)
    source = replace(source, gmnw_config=_core_config())
    source.gmnw_tensors[23].copy_(torch.tensor([0, 2, 0, 2], dtype=torch.int8))
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=ABSOLUTE_GENERATION
    )
    target = _prepared(populated=False)
    target = replace(target, gmnw_config=_modern_b3_config())
    target.gmnw_tensors[23].copy_(torch.tensor([0, 2, 0, 2], dtype=torch.int8))

    assert sidecar["schema_version"] == codec.GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_V1
    assert len(sidecar["tensor_manifest"]) == 16
    with pytest.raises(
        codec.DirectResidentGmnwCheckpointV1Error,
        match="incomplete for an extended control target",
    ):
        codec.restore_direct_resident_gmnw_checkpoint_v1(
            target, sidecar, absolute_generation=ABSOLUTE_GENERATION
        )


def test_legacy_v1_sidecar_is_rejected_for_sel1_target() -> None:
    legacy_source = replace(_prepared(populated=True), gmnw_config=_core_config())
    legacy_source.gmnw_tensors[23].copy_(torch.tensor([0, 1, 2, 0], dtype=torch.int8))
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        legacy_source, absolute_generation=ABSOLUTE_GENERATION
    )

    with pytest.raises(
        codec.DirectResidentGmnwCheckpointV1Error,
        match="incomplete for an extended control target",
    ):
        codec.restore_direct_resident_gmnw_checkpoint_v1(
            _prepared(populated=False),
            sidecar,
            absolute_generation=ABSOLUTE_GENERATION,
        )
