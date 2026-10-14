from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from engine.perf import cuda_architecture_b_direct_resident_gmnw_checkpoint_v1 as codec
from engine.perf import cuda_architecture_b_direct_resident_gmnw_gru_v1 as gmnw


_ABSOLUTE_GENERATION = 3


def _gmnw_config(
    *, generations: int = 5, grace_reproduction_scale: float = 0.5
) -> dict[str, Any]:
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
        "grace_reproduction_scale": grace_reproduction_scale,
    }


def _prepared(
    *, populated: bool, generations: int = 5
) -> gmnw.PreparedDirectResidentGmnwGruV1:
    task_config = gmnw.GmnwConfigurationV1()
    tensors = gmnw._allocate_gmnw_tensors(
        cells=2,
        capacity=3,
        generations=generations,
        config=task_config,
        device=torch.device("cpu"),
    )
    alive_mask = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )
    slot_to_agent_id = torch.tensor([[10, -1, 12], [-1, 20, -1]], dtype=torch.int64)
    if populated:
        tensors[0].copy_(slot_to_agent_id)
        for index in (1, 11, 12):
            tensors[index].copy_(
                torch.arange(6, dtype=tensors[index].dtype).reshape(2, 3) + index
            )
        for index in (2, 3, 7, 8, 9, 10):
            tensors[index].copy_(
                torch.arange(6, dtype=tensors[index].dtype).reshape(2, 3) - 2
            )
        tensors[4].fill_(7)
        tensors[5].fill_(2)
        tensors[6].copy_(alive_mask)
        tensors[13].copy_(
            torch.tensor([[0.25, 1.0, 0.5], [1.0, 0.75, 1.0]], dtype=torch.float32)
        )
        tensors[14].copy_(
            torch.arange(2 * generations * 6, dtype=torch.int64).reshape(
                2, generations, 6
            )
        )
        tensors[15].copy_(
            torch.arange(2 * generations, dtype=torch.float64).reshape(2, generations)
            / 4.0
        )
        tensors[16].fill_(91)
    patch = SimpleNamespace(
        source=SimpleNamespace(
            live_state={
                "alive_mask": alive_mask,
                "slot_to_agent_id": slot_to_agent_id,
            }
        )
    )
    return gmnw.PreparedDirectResidentGmnwGruV1(
        patch=patch,
        gru_tensors=(),
        gmnw_tensors=tensors,
        gru_config={},
        gmnw_config=_gmnw_config(generations=generations),
        initial_projection=None,
        base_observation_size=74,
        observation_size=86,
    )


def _with_b3_control_tail(
    prepared: gmnw.PreparedDirectResidentGmnwGruV1,
    *,
    arm: torch.Tensor | None = None,
    packed_fill: float = 0.0,
    scratch_fill: float = 0.0,
) -> gmnw.PreparedDirectResidentGmnwGruV1:
    cells = prepared.gmnw_tensors[0].size(0)
    control_tail = (
        arm if arm is not None else torch.arange(cells, dtype=torch.int8),
        torch.full(
            (gmnw.GMNW_MATCHED_MLP_PACKED_WIDTH_V1,),
            packed_fill,
            dtype=torch.float32,
        ),
        torch.full(
            (cells, gmnw.GMNW_MATCHED_MLP_SCRATCH_WIDTH_V1),
            scratch_fill,
            dtype=torch.float32,
        ),
        torch.zeros((cells,), dtype=torch.int32),
    )
    return replace(prepared, gmnw_tensors=(*prepared.gmnw_tensors, *control_tail))


def _sel1_multicell_prepared(
    *, populated: bool = True, profiles: tuple[int, int] = (0, 0)
) -> gmnw.PreparedDirectResidentGmnwGruV1:
    prepared = _with_b3_control_tail(
        _prepared(populated=populated),
        arm=torch.tensor([3, 6], dtype=torch.int8),
    )
    prepared.gmnw_tensors[16].zero_()
    prepared.gmnw_tensors[22].copy_(torch.tensor(profiles, dtype=torch.int32))
    prepared.patch.source.live_state["policy_use_gru"] = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )
    return replace(
        prepared,
        gmnw_config={
            **prepared.gmnw_config,
            "control_matrix_enabled": True,
            "policy_arm_codes": (3, 6),
            "cue_visible_at_decision": (False, True),
            "gru_founder_count": (2, 2),
            "qualification_trace_enabled": False,
            "qualification_trace_output_ticks": 0,
            "qualification_decision_tape_enabled": False,
            "selection_metrics_enabled": False,
            "terminal_low_population_no_reinit": False,
            "venv_consequence_profile_codes": profiles,
        },
    )


def _venv_control_prepared(
    *, populated: bool, profiles: tuple[int, int] = (2, 2)
) -> gmnw.PreparedDirectResidentGmnwGruV1:
    prepared = _with_b3_control_tail(
        _prepared(populated=populated),
        arm=torch.tensor([0, 5], dtype=torch.int8),
    )
    prepared.gmnw_tensors[16].zero_()
    prepared.gmnw_tensors[22].copy_(torch.tensor(profiles, dtype=torch.int32))
    prepared.patch.source.live_state["policy_use_gru"] = torch.tensor(
        [[True, True, True], [False, False, False]], dtype=torch.bool
    )
    return replace(
        prepared,
        gmnw_config={
            **prepared.gmnw_config,
            "control_matrix_enabled": True,
            "policy_arm_codes": (0, 5),
            "cue_visible_at_decision": (False, True),
            "gru_founder_count": (None, None),
            "qualification_trace_enabled": False,
            "qualification_trace_output_ticks": 0,
            "qualification_decision_tape_enabled": False,
            "selection_metrics_enabled": False,
            "terminal_low_population_no_reinit": False,
            "venv_consequence_profile_codes": profiles,
        },
    )


def _as_historical_sel1_v1_sidecar(sidecar: dict[str, Any]) -> dict[str, Any]:
    names = codec.GMNW_CHECKPOINT_TENSOR_NAMES_SEL1_V1
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
        "manifest_sha256": codec._canonical_sha256(
            manifest, path="tensor_manifest"
        ),
        "tensors": dict(zip(names, tensors, strict=True)),
    }
    historical["digest_sha256"] = codec._canonical_sha256(
        codec._digest_payload(historical), path="sidecar"
    )
    return historical


def _as_historical_multicell_v1(checkpoint: dict[str, Any]) -> dict[str, Any]:
    historical = dict(checkpoint)
    historical["schema_version"] = (
        codec.GMNW_MULTICELL_CHECKPOINT_SCHEMA_VERSION_SEL1_V1
    )
    historical["codec_version"] = (
        codec.GMNW_MULTICELL_CHECKPOINT_CODEC_VERSION_SEL1_V1
    )
    historical["sidecars"] = [
        _as_historical_sel1_v1_sidecar(sidecar)
        for sidecar in checkpoint["sidecars"]
    ]
    historical["digest_sha256"] = codec._canonical_sha256(
        codec._multicell_digest_payload(historical), path="multicell_checkpoint"
    )
    return historical


def _replace_sidecar_tensor(
    sidecar: dict[str, Any], name: str, tensor: torch.Tensor
) -> dict[str, Any]:
    changed = dict(sidecar)
    changed["tensors"] = dict(sidecar["tensors"])
    changed["tensors"][name] = tensor
    return changed


def test_profile_aware_sidecar_uses_v2_and_roundtrips_exact_codes() -> None:
    source = _sel1_multicell_prepared(profiles=(2, 3))
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=_ABSOLUTE_GENERATION
    )
    target = _sel1_multicell_prepared(populated=False, profiles=(2, 3))

    codec.restore_direct_resident_gmnw_checkpoint_v1(
        target, sidecar, absolute_generation=_ABSOLUTE_GENERATION
    )

    assert (
        sidecar["schema_version"]
        == codec.GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V2
    )
    assert (
        sidecar["codec_version"]
        == codec.GMNW_CHECKPOINT_SIDECAR_CODEC_VERSION_SEL1_V2
    )
    assert torch.equal(target.gmnw_tensors[22], source.gmnw_tensors[22])
    assert sidecar["tensors"]["venv_consequence_profile_code"].tolist() == [2, 3]


def test_nonselection_venv_control_uses_v2_and_roundtrips_exact_codes() -> None:
    source = _venv_control_prepared(populated=True)
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=_ABSOLUTE_GENERATION
    )
    target = _venv_control_prepared(populated=False)

    codec.restore_direct_resident_gmnw_checkpoint_v1(
        target, sidecar, absolute_generation=_ABSOLUTE_GENERATION
    )

    assert (
        sidecar["schema_version"]
        == codec.GMNW_CHECKPOINT_SIDECAR_SCHEMA_VERSION_SEL1_V2
    )
    assert sidecar["gmnw_config"]["policy_arm_codes"] == [0, 5]
    assert torch.equal(target.gmnw_tensors[22], source.gmnw_tensors[22])


def test_historical_sel1_v1_sidecar_restores_as_legacy_profile_zero() -> None:
    current = codec.export_direct_resident_gmnw_checkpoint_v1(
        _sel1_multicell_prepared(), absolute_generation=_ABSOLUTE_GENERATION
    )
    historical = _as_historical_sel1_v1_sidecar(current)
    target = _sel1_multicell_prepared(populated=False)

    codec.restore_direct_resident_gmnw_checkpoint_v1(
        target, historical, absolute_generation=_ABSOLUTE_GENERATION
    )

    assert torch.count_nonzero(target.gmnw_tensors[22]).item() == 0


def test_cpu_sidecar_roundtrip_preserves_live_gmnw_state_only() -> None:
    source = _prepared(populated=True)
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=_ABSOLUTE_GENERATION
    )
    repeated = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=_ABSOLUTE_GENERATION
    )
    fresh = _prepared(populated=False)
    fresh.gmnw_tensors[16].fill_(17)
    for index in range(17, 22):
        fresh.gmnw_tensors[index].fill_(index + 100)
    frozen_before = tuple(tensor.clone() for tensor in fresh.gmnw_tensors[17:22])

    restored_generation = codec.restore_direct_resident_gmnw_checkpoint_v1(
        fresh, sidecar, absolute_generation=_ABSOLUTE_GENERATION
    )

    assert restored_generation == _ABSOLUTE_GENERATION
    assert sidecar["digest_sha256"] == repeated["digest_sha256"]
    assert sidecar["manifest_sha256"] == repeated["manifest_sha256"]
    assert [row["name"] for row in sidecar["tensor_manifest"]] == list(
        codec.GMNW_CHECKPOINT_TENSOR_NAMES_V1
    )
    assert [row["index"] for row in sidecar["tensor_manifest"]] == list(range(16))
    for index, name in enumerate(codec.GMNW_CHECKPOINT_TENSOR_NAMES_V1):
        exported = sidecar["tensors"][name]
        assert exported.device.type == "cpu"
        assert exported.data_ptr() != source.gmnw_tensors[index].data_ptr()
        assert torch.equal(fresh.gmnw_tensors[index], source.gmnw_tensors[index])
    assert torch.equal(fresh.gmnw_tensors[13], source.gmnw_tensors[13])
    assert torch.equal(fresh.gmnw_tensors[14], source.gmnw_tensors[14])
    assert torch.equal(fresh.gmnw_tensors[15], source.gmnw_tensors[15])
    assert torch.count_nonzero(fresh.gmnw_tensors[16]).item() == 0
    for observed, expected in zip(
        fresh.gmnw_tensors[17:22], frozen_before, strict=True
    ):
        assert torch.equal(observed, expected)


def test_b3_export_preserves_b1_sidecar_schema_and_digest() -> None:
    b1 = _prepared(populated=True)
    b3 = _with_b3_control_tail(
        _prepared(populated=True),
        arm=torch.tensor([2, 0], dtype=torch.int8),
        packed_fill=3.0,
        scratch_fill=4.0,
    )

    b1_sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        b1, absolute_generation=_ABSOLUTE_GENERATION
    )
    b3_sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        b3, absolute_generation=_ABSOLUTE_GENERATION
    )

    assert b3_sidecar.keys() == b1_sidecar.keys()
    assert b3_sidecar["manifest_sha256"] == b1_sidecar["manifest_sha256"]
    assert b3_sidecar["digest_sha256"] == b1_sidecar["digest_sha256"]
    assert b3_sidecar["tensor_manifest"] == b1_sidecar["tensor_manifest"]
    assert set(b3_sidecar["tensors"]) == set(codec.GMNW_CHECKPOINT_TENSOR_NAMES_V1)


def test_b3_roundtrip_restores_only_authoritative_state() -> None:
    source = _with_b3_control_tail(
        _prepared(populated=True),
        arm=torch.tensor([0, 2], dtype=torch.int8),
        packed_fill=11.0,
        scratch_fill=12.0,
    )
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=_ABSOLUTE_GENERATION
    )
    target = _with_b3_control_tail(
        _prepared(populated=False),
        arm=torch.tensor([2, 1], dtype=torch.int8),
        packed_fill=-7.0,
        scratch_fill=-8.0,
    )
    for index in range(17, 22):
        target.gmnw_tensors[index].fill_(index + 40)
    non_authoritative_before = tuple(
        tensor.clone() for tensor in target.gmnw_tensors[17:]
    )

    restored_generation = codec.restore_direct_resident_gmnw_checkpoint_v1(
        target, sidecar, absolute_generation=_ABSOLUTE_GENERATION
    )

    assert restored_generation == _ABSOLUTE_GENERATION
    for index in range(16):
        assert torch.equal(target.gmnw_tensors[index], source.gmnw_tensors[index])
    assert torch.count_nonzero(target.gmnw_tensors[16]).item() == 0
    for observed, expected in zip(
        target.gmnw_tensors[17:], non_authoritative_before, strict=True
    ):
        assert torch.equal(observed, expected)


@pytest.mark.parametrize("tensor_count", [0, 21, 22, 24, 26, 28, 29, 31])
def test_export_rejects_every_unsupported_tensor_count(tensor_count: int) -> None:
    prepared = _prepared(populated=True)
    tensors = prepared.gmnw_tensors
    if tensor_count <= len(tensors):
        changed_tensors = tensors[:tensor_count]
    else:
        changed_tensors = (
            *tensors,
            *(torch.empty(0) for _ in range(tensor_count - len(tensors))),
        )
    changed = replace(prepared, gmnw_tensors=changed_tensors)

    with pytest.raises(
        codec.DirectResidentGmnwCheckpointV1Error,
        match="23-, 27-, or 30-tensor",
    ):
        codec.export_direct_resident_gmnw_checkpoint_v1(
            changed, absolute_generation=_ABSOLUTE_GENERATION
        )


@pytest.mark.parametrize(
    ("index", "replacement", "match"),
    [
        (22, lambda tensor: tensor.to(torch.int64), "dtype"),
        (22, lambda tensor: tensor[:, None], "shape"),
        (22, lambda tensor: torch.tensor([-1, 3], dtype=tensor.dtype), "known"),
        (23, lambda tensor: tensor.to(torch.int32), "dtype"),
        (23, lambda tensor: tensor[:-1], "shape"),
        (23, lambda tensor: torch.tensor([-1, 3], dtype=tensor.dtype), "within"),
        (24, lambda tensor: tensor.to(torch.float64), "dtype"),
        (24, lambda tensor: tensor[:-1], "shape"),
        (25, lambda tensor: tensor.to(torch.float64), "dtype"),
        (25, lambda tensor: tensor[:, :-1], "shape"),
        (26, lambda tensor: tensor.to(torch.int64), "dtype"),
        (26, lambda tensor: tensor[:, None], "shape"),
        (26, lambda tensor: torch.ones_like(tensor), "status"),
    ],
)
def test_export_rejects_malformed_b3_control_tail(
    index: int, replacement: Any, match: str
) -> None:
    prepared = _with_b3_control_tail(_prepared(populated=True))
    tensors = list(prepared.gmnw_tensors)
    tensors[index] = replacement(tensors[index])
    changed = replace(prepared, gmnw_tensors=tuple(tensors))

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match=match):
        codec.export_direct_resident_gmnw_checkpoint_v1(
            changed, absolute_generation=_ABSOLUTE_GENERATION
        )


def test_restore_rejects_malformed_b3_tail_without_mutation() -> None:
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        _prepared(populated=True), absolute_generation=_ABSOLUTE_GENERATION
    )
    target = _with_b3_control_tail(_prepared(populated=False))
    target.gmnw_tensors[26][0] = 1
    before = tuple(tensor.clone() for tensor in target.gmnw_tensors)

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="status"):
        codec.restore_direct_resident_gmnw_checkpoint_v1(
            target, sidecar, absolute_generation=_ABSOLUTE_GENERATION
        )

    for observed, expected in zip(target.gmnw_tensors, before, strict=True):
        assert torch.equal(observed, expected)


def test_restore_rejects_tampered_aggregate_digest() -> None:
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        _prepared(populated=True), absolute_generation=_ABSOLUTE_GENERATION
    )
    sidecar["digest_sha256"] = "0" * 64

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="digest"):
        codec.restore_direct_resident_gmnw_checkpoint_v1(
            _prepared(populated=False),
            sidecar,
            absolute_generation=_ABSOLUTE_GENERATION,
        )


def test_expand_validated_sidecar_adds_only_zero_future_aggregate_rows() -> None:
    source = _prepared(populated=True, generations=4)
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=4
    )

    expanded = codec.expand_direct_resident_gmnw_checkpoint_v1(
        sidecar, generation_rows_per_cell=5
    )
    fresh = _prepared(populated=False, generations=5)
    codec.restore_direct_resident_gmnw_checkpoint_v1(
        fresh, expanded, absolute_generation=4
    )

    assert expanded["gmnw_config"]["generation_rows_per_cell"] == 5
    assert expanded["digest_sha256"] != sidecar["digest_sha256"]
    for index in range(14):
        assert torch.equal(fresh.gmnw_tensors[index], source.gmnw_tensors[index])
    for index in (14, 15):
        assert torch.equal(fresh.gmnw_tensors[index][:, :4], source.gmnw_tensors[index])
        assert torch.count_nonzero(fresh.gmnw_tensors[index][:, 4:]).item() == 0


def test_expand_rejects_tampered_source_before_rebuilding_digests() -> None:
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        _prepared(populated=True, generations=4), absolute_generation=4
    )
    sidecar["tensors"]["damage_total"][0, 0] += 1.0

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="digest"):
        codec.expand_direct_resident_gmnw_checkpoint_v1(
            sidecar, generation_rows_per_cell=5
        )


@pytest.mark.parametrize(
    ("replacement", "match"),
    [
        (lambda tensor: tensor.to(torch.float64), "dtype"),
        (lambda tensor: tensor[:, :2].contiguous(), "shape"),
    ],
)
def test_restore_rejects_tensor_dtype_or_shape(replacement: Any, match: str) -> None:
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        _prepared(populated=True), absolute_generation=_ABSOLUTE_GENERATION
    )
    changed = _replace_sidecar_tensor(
        sidecar,
        "reproduction_scale",
        replacement(sidecar["tensors"]["reproduction_scale"]),
    )

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match=match):
        codec.restore_direct_resident_gmnw_checkpoint_v1(
            _prepared(populated=False),
            changed,
            absolute_generation=_ABSOLUTE_GENERATION,
        )


def test_restore_rejects_schema_and_config_mismatches() -> None:
    source = _prepared(populated=True)
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        source, absolute_generation=_ABSOLUTE_GENERATION
    )
    schema_changed = dict(sidecar, schema_version="unexpected_v2")
    config_changed = replace(
        _prepared(populated=False),
        gmnw_config=_gmnw_config(grace_reproduction_scale=0.25),
    )

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="schema"):
        codec.restore_direct_resident_gmnw_checkpoint_v1(
            _prepared(populated=False),
            schema_changed,
            absolute_generation=_ABSOLUTE_GENERATION,
        )
    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="config"):
        codec.restore_direct_resident_gmnw_checkpoint_v1(
            config_changed,
            sidecar,
            absolute_generation=_ABSOLUTE_GENERATION,
        )


def test_restore_rejects_absolute_generation_mismatch_without_mutation() -> None:
    sidecar = codec.export_direct_resident_gmnw_checkpoint_v1(
        _prepared(populated=True), absolute_generation=_ABSOLUTE_GENERATION
    )
    fresh = _prepared(populated=False)
    before = tuple(tensor.clone() for tensor in fresh.gmnw_tensors)

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="generation"):
        codec.restore_direct_resident_gmnw_checkpoint_v1(
            fresh, sidecar, absolute_generation=_ABSOLUTE_GENERATION + 1
        )

    for observed, expected in zip(fresh.gmnw_tensors, before, strict=True):
        assert torch.equal(observed, expected)


def test_export_rejects_stream_id_occupancy_mismatch() -> None:
    prepared = _prepared(populated=True)
    prepared.gmnw_tensors[0][0, 0] = 999

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="occupancy"):
        codec.export_direct_resident_gmnw_checkpoint_v1(
            prepared, absolute_generation=_ABSOLUTE_GENERATION
        )


def test_public_multicell_validator_is_read_only_and_returns_boundary_identity() -> (
    None
):
    cell_ids = ("durable-cell-00", "durable-cell-01")
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _sel1_multicell_prepared(),
        absolute_generation=_ABSOLUTE_GENERATION,
        cell_ids=cell_ids,
    )
    digest_before = checkpoint["digest_sha256"]
    tensor_snapshots = tuple(
        tensor.clone()
        for sidecar in checkpoint["sidecars"]
        for tensor in sidecar["tensors"].values()
    )

    observed = codec.validate_direct_resident_gmnw_multicell_checkpoint_v1(
        checkpoint,
        expected_cell_ids=cell_ids,
        expected_absolute_generation=_ABSOLUTE_GENERATION,
    )

    assert observed == (cell_ids, _ABSOLUTE_GENERATION)
    assert checkpoint["digest_sha256"] == digest_before
    for observed_tensor, expected_tensor in zip(
        (
            tensor
            for sidecar in checkpoint["sidecars"]
            for tensor in sidecar["tensors"].values()
        ),
        tensor_snapshots,
        strict=True,
    ):
        assert torch.equal(observed_tensor, expected_tensor)


def test_cell_sidecar_digest_selects_validated_state_by_identity() -> None:
    cell_ids = ("durable-cell-00", "durable-cell-01")
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _sel1_multicell_prepared(),
        absolute_generation=_ABSOLUTE_GENERATION,
        cell_ids=cell_ids,
    )

    assert codec.direct_resident_gmnw_cell_sidecar_digest_v1(
        checkpoint, cell_id=cell_ids[0]
    ) == checkpoint["sidecars"][0]["digest_sha256"]
    assert codec.direct_resident_gmnw_cell_sidecar_digest_v1(
        checkpoint, cell_id=cell_ids[1]
    ) == checkpoint["sidecars"][1]["digest_sha256"]
    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="absent"):
        codec.direct_resident_gmnw_cell_sidecar_digest_v1(
            checkpoint, cell_id="missing-cell"
        )


def test_historical_multicell_v1_validates_and_restores_profile_zero() -> None:
    cell_ids = ("durable-cell-00", "durable-cell-01")
    current = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _sel1_multicell_prepared(),
        absolute_generation=_ABSOLUTE_GENERATION,
        cell_ids=cell_ids,
    )
    historical = _as_historical_multicell_v1(current)
    target = _sel1_multicell_prepared(populated=False)

    observed = codec.validate_direct_resident_gmnw_multicell_checkpoint_v1(
        historical,
        expected_cell_ids=cell_ids,
        expected_absolute_generation=_ABSOLUTE_GENERATION,
    )
    codec.restore_direct_resident_gmnw_multicell_checkpoint_v1(
        target,
        historical,
        absolute_generation=_ABSOLUTE_GENERATION,
        cell_ids=cell_ids,
    )

    assert observed == (cell_ids, _ABSOLUTE_GENERATION)
    assert torch.count_nonzero(target.gmnw_tensors[22]).item() == 0


@pytest.mark.parametrize("outer_version", (1, 2))
def test_multicell_outer_and_sidecar_versions_cannot_be_mixed(
    outer_version: int,
) -> None:
    cell_ids = ("durable-cell-00", "durable-cell-01")
    current = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _sel1_multicell_prepared(),
        absolute_generation=_ABSOLUTE_GENERATION,
        cell_ids=cell_ids,
    )
    historical = _as_historical_multicell_v1(current)
    changed = dict(current if outer_version == 2 else historical)
    changed["sidecars"] = list(changed["sidecars"])
    changed["sidecars"][0] = (
        historical["sidecars"][0]
        if outer_version == 2
        else current["sidecars"][0]
    )
    changed["digest_sha256"] = codec._canonical_sha256(
        codec._multicell_digest_payload(changed), path="multicell_checkpoint"
    )

    with pytest.raises(
        codec.DirectResidentGmnwCheckpointV1Error,
        match="does not match the outer",
    ):
        codec.validate_direct_resident_gmnw_multicell_checkpoint_v1(changed)


def test_public_multicell_validator_fails_closed_on_durable_boundary_mismatch() -> None:
    cell_ids = ("durable-cell-00", "durable-cell-01")
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _sel1_multicell_prepared(),
        absolute_generation=_ABSOLUTE_GENERATION,
        cell_ids=cell_ids,
    )

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="count/order"):
        codec.validate_direct_resident_gmnw_multicell_checkpoint_v1(
            checkpoint,
            expected_cell_ids=tuple(reversed(cell_ids)),
        )
    with pytest.raises(
        codec.DirectResidentGmnwCheckpointV1Error,
        match="expected durable checkpoint boundary",
    ):
        codec.validate_direct_resident_gmnw_multicell_checkpoint_v1(
            checkpoint,
            expected_absolute_generation=_ABSOLUTE_GENERATION + 1,
        )


def test_public_multicell_validator_rejects_tampered_reload_digest() -> None:
    cell_ids = ("durable-cell-00", "durable-cell-01")
    checkpoint = codec.export_direct_resident_gmnw_multicell_checkpoint_v1(
        _sel1_multicell_prepared(),
        absolute_generation=_ABSOLUTE_GENERATION,
        cell_ids=cell_ids,
    )
    tampered = dict(checkpoint, digest_sha256="0" * 64)

    with pytest.raises(codec.DirectResidentGmnwCheckpointV1Error, match="digest"):
        codec.validate_direct_resident_gmnw_multicell_checkpoint_v1(tampered)
