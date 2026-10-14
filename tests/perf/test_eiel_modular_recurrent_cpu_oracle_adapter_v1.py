from __future__ import annotations

import copy
import ctypes
from dataclasses import replace
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

from perf.cuda_architecture_b_modular_state_abi import (
    ModularStorageError,
    pack_modular_storage,
)
from engine.perf.eiel_phase_e_modular_tick_fixture_v1 import (
    AGENT_COUNT,
    EXPANDED_B2_CAPS,
    LOCAL_SIZE,
    TICK_AGES,
    _build_population,
    _fixed_inputs,
)
from engine.perf.recurrent_native import modular_recurrent_cpu_artifact_v1
from engine.perf.recurrent_native import (
    modular_recurrent_cpu_oracle_adapter_v1 as oracle_adapter,
)
from engine.perf.recurrent_native.modular_recurrent_cpu_artifact_v1 import (
    ACTION_WIDTH,
    ROW_PLANE_FIELDS,
    ConstOwnerViewV1,
    ModularRecurrentCpuArtifactError,
    RowV1,
    TickReceiptV1,
    build_prebuilt_modular_recurrent_cpu_oracle,
    load_prebuilt_modular_recurrent_cpu_oracle,
)
from engine.perf.recurrent_native.modular_recurrent_cpu_oracle_adapter_v1 import (
    ModularRecurrentCpuOracleAdapterError,
    execute_modular_recurrent_cpu_oracle_newborn_v1,
    execute_modular_recurrent_cpu_oracle_v1,
    production_qualified,
    publication_qualified,
    runtime_build,
)


PROFILE = 1


def _packed_fixture(*, extra_empty_rows: int = 1):
    alive, _modules = _build_population()
    records: list[dict[str, object] | None] = []
    for index, agent in enumerate(alive):
        records.append(
            {
                "agent_id": int(agent.agent_id),
                "agent_slot_token": 10_000 + index,
                "genome": {
                    "schema_version": "modular_genome_v1",
                    "genes": {},
                    "modules": copy.deepcopy(agent.genome.modules),
                    "connections": copy.deepcopy(agent.genome.connections),
                    "b2_caps": copy.deepcopy(EXPANDED_B2_CAPS),
                    "b2_invalid_policy": "reject",
                },
                "structural_live_module_state": copy.deepcopy(
                    agent.structural_live_module_state
                ),
                "structural_live_module_outputs": copy.deepcopy(
                    agent.structural_live_module_outputs
                ),
                "structural_development_latch_by_instance_id": {},
                "structural_development_effective_gain_by_instance_id": {},
                "gated_recurrent_enabled": True,
            }
        )
    records.extend([None] * extra_empty_rows)
    return pack_modular_storage(
        records,
        agent_capacity=AGENT_COUNT + extra_empty_rows,
        profile_id="b2-16x48-v1",
    )


def _tick_inputs(row_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    observations, _base, _policy, _masks = _fixed_inputs()
    tick_count = len(TICK_AGES)
    padded = np.zeros((tick_count, row_count, observations.shape[2]), dtype=np.float32)
    padded[:, :AGENT_COUNT] = observations[:tick_count]
    fatigue = padded[:, :, LOCAL_SIZE].copy()
    stages = np.broadcast_to(
        np.arange(tick_count, dtype=np.int64)[:, None], (tick_count, row_count)
    ).copy()
    return (
        torch.from_numpy(padded),
        torch.from_numpy(fatigue),
        torch.from_numpy(stages),
    )


def _direct_rows(planes: dict[str, torch.Tensor]) -> ctypes.Array[RowV1]:
    row_count = int(planes["agent_occupied"].shape[0])
    rows = (RowV1 * row_count)()
    for row_index in range(row_count):
        row = rows[row_index]
        for name in ROW_PLANE_FIELDS:
            source = planes[name][row_index].reshape(-1).tolist()
            destination = getattr(row, name)
            if isinstance(destination, ctypes.Array):
                assert len(destination) == len(source)
                for index, value in enumerate(source):
                    destination[index] = value
            else:
                setattr(row, name, source[0])
    return rows


def _direct_planes(
    rows: ctypes.Array[RowV1], template: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    result = {name: tensor.clone() for name, tensor in template.items()}
    for row_index, row in enumerate(rows):
        for name in ROW_PLANE_FIELDS:
            target = result[name][row_index]
            source = getattr(row, name)
            values = list(source) if isinstance(source, ctypes.Array) else [source]
            target.copy_(torch.tensor(values, dtype=target.dtype).reshape(target.shape))
    return result


class _DirectOwners:
    def __init__(self, planes: dict[str, torch.Tensor]) -> None:
        occupied = planes["agent_occupied"].tolist()
        agent_ids = planes["agent_id"].tolist()
        tokens = planes["agent_slot_token"].tolist()
        private_tokens = [
            int(token) if bool(occupied[index]) else -(100_000 + index)
            for index, token in enumerate(tokens)
        ]
        row_count = len(agent_ids)
        self.agent_ids = (ctypes.c_int64 * row_count)(*agent_ids)
        self.tokens = (ctypes.c_int64 * row_count)(*private_tokens)
        self.alive = (ctypes.c_uint8 * row_count)(*[int(value) for value in occupied])
        self.view = ConstOwnerViewV1(self.agent_ids, self.tokens, self.alive, row_count)


def _direct_tick(
    library: ctypes.CDLL,
    rows: ctypes.Array[RowV1],
    owners: _DirectOwners,
    observations: torch.Tensor,
    fatigue: torch.Tensor,
    stages: torch.Tensor,
) -> tuple[torch.Tensor, TickReceiptV1]:
    row_count = len(rows)
    scratch_rows = (RowV1 * row_count)()
    scratch_actions = torch.empty((row_count, ACTION_WIDTH), dtype=torch.float32)
    actions = torch.empty((row_count, ACTION_WIDTH), dtype=torch.float32)
    receipt = TickReceiptV1()
    status = library.eiel_modular_recurrent_tick_v1(
        rows,
        row_count,
        PROFILE,
        ctypes.byref(owners.view),
        ctypes.cast(observations.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        observations.numel(),
        ctypes.cast(fatigue.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        fatigue.numel(),
        ctypes.cast(stages.data_ptr(), ctypes.POINTER(ctypes.c_int64)),
        stages.numel(),
        scratch_rows,
        row_count,
        ctypes.cast(scratch_actions.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        scratch_actions.numel(),
        ctypes.cast(actions.data_ptr(), ctypes.POINTER(ctypes.c_float)),
        actions.numel(),
        ctypes.byref(receipt),
    )
    assert status == 0
    return actions, receipt


@pytest.fixture(scope="module")
def oracle(tmp_path_factory: pytest.TempPathFactory):
    artifact = build_prebuilt_modular_recurrent_cpu_oracle(
        tmp_path_factory.mktemp("modular_oracle_adapter")
    )
    return load_prebuilt_modular_recurrent_cpu_oracle(artifact)


def test_exact_direct_c_parity_multi_tick_progression_and_immutability(oracle) -> None:
    packed = _packed_fixture()
    observations, fatigue, stages = _tick_inputs(packed.layout.agent_capacity)
    input_tensors = {name: tensor.clone() for name, tensor in packed.tensors.items()}
    input_canonical = packed.canonical_bytes()

    direct_rows = _direct_rows(packed.tensors)
    direct_owners = _DirectOwners(packed.tensors)
    direct_actions = []
    state_after_first_tick = None
    for tick_index in range(observations.shape[0]):
        actions, receipt = _direct_tick(
            oracle.library,
            direct_rows,
            direct_owners,
            observations[tick_index],
            fatigue[tick_index],
            stages[tick_index],
        )
        direct_actions.append(actions.clone())
        assert receipt.production_routing_qualified == 0
        assert receipt.publication_qualified == 0
        if tick_index == 0:
            state_after_first_tick = _direct_planes(direct_rows, packed.tensors)[
                "module_state"
            ]
    direct_planes = _direct_planes(direct_rows, packed.tensors)

    result = execute_modular_recurrent_cpu_oracle_v1(
        packed, oracle.artifact_path, observations, fatigue, stages
    )

    assert torch.equal(result.action_residuals, torch.stack(direct_actions))
    for name in ROW_PLANE_FIELDS:
        assert torch.equal(result.updated_tensors[name], direct_planes[name]), name
        assert torch.equal(packed.tensors[name], input_tensors[name]), name
        assert (
            result.updated_tensors[name].data_ptr() != packed.tensors[name].data_ptr()
        )
    assert packed.canonical_bytes() == input_canonical
    assert state_after_first_tick is not None
    assert not torch.equal(state_after_first_tick, direct_planes["module_state"])
    assert not torch.equal(
        packed.tensors["module_state"], result.updated_tensors["module_state"]
    )
    assert len(result.tick_receipts) == observations.shape[0]
    assert all(receipt.rng_draws == 0 for receipt in result.tick_receipts)
    assert result.input_identity.input_packed_sha256 == packed.packed_sha256
    assert result.input_identity.input_sidecar_sha256 == packed.sidecar_sha256
    assert not hasattr(result, "output_packed_sha256")


def test_one_tick_shape_and_qualification_boundary(oracle) -> None:
    packed = _packed_fixture()
    observations, fatigue, stages = _tick_inputs(packed.layout.agent_capacity)
    result = execute_modular_recurrent_cpu_oracle_v1(
        packed,
        oracle.artifact_path,
        observations[0],
        fatigue[0],
        stages[0],
    )
    assert result.action_residuals.shape == (
        1,
        packed.layout.agent_capacity,
        ACTION_WIDTH,
    )
    assert production_qualified is False
    assert publication_qualified is False
    assert runtime_build is False
    assert result.production_qualified is False
    assert result.publication_qualified is False
    assert result.runtime_build is False


def _newborn_kwargs(packed) -> dict[str, object]:
    destination = packed.layout.agent_capacity - 1
    return {
        "destination_row": destination,
        "expected_destination_token": 64_000,
        "parent_a_row": 1,
        "expected_parent_a_token": int(
            packed.tensors["agent_slot_token"][1].item()
        ),
        "parent_b_row": 0,
        "expected_parent_b_token": int(
            packed.tensors["agent_slot_token"][0].item()
        ),
        "donor_draw": 0.25,
        "child_agent_id": 64_001,
        "publication_token": 64_002,
        "gated_module_slot": 0,
        "gated_field": "reset_bias",
        "gated_replacement_value": 0.75,
    }


def test_newborn_gated_field_replacement_delta_and_exact_owner_receipt(oracle) -> None:
    packed = _packed_fixture()
    input_canonical = packed.canonical_bytes()
    input_tensors = {name: tensor.clone() for name, tensor in packed.tensors.items()}
    kwargs = _newborn_kwargs(packed)

    first = execute_modular_recurrent_cpu_oracle_newborn_v1(
        packed, oracle.artifact_path, **kwargs
    )
    second = execute_modular_recurrent_cpu_oracle_newborn_v1(
        packed, oracle.artifact_path, **kwargs
    )
    destination = int(kwargs["destination_row"])
    receipt = first.receipt

    assert receipt.value_before_float32 == -0.5
    assert receipt.replacement_value_float32 == 0.75
    assert receipt.additive_delta_float32 == 1.25
    assert receipt.gated_field_index == 0
    assert receipt.donor_parent_row == 1
    assert receipt.expected_destination_token == 64_000
    assert receipt.expected_parent_a_token == 10_001
    assert receipt.expected_parent_b_token == 10_000
    assert receipt.child_agent_id == 64_001
    assert receipt.publication_token == 64_002
    assert receipt.source_packed_sha256 == packed.packed_sha256
    assert receipt.child_tensors_sha256 == second.receipt.child_tensors_sha256
    assert first.child_tensors["gated_parameters"][0, 0].item() == 0.75
    assert first.child_tensors["agent_id"].item() == 64_001
    assert first.child_tensors["agent_slot_token"].item() == 64_002
    assert first.child_tensors["agent_occupied"].item()
    assert not first.child_tensors["module_state_present"].any().item()
    assert not first.child_tensors["module_output_present"].any().item()
    for name in ROW_PLANE_FIELDS:
        assert torch.equal(
            first.child_tensors[name], first.updated_tensors[name][destination]
        )
        assert torch.equal(packed.tensors[name], input_tensors[name])
        assert torch.equal(
            first.updated_tensors[name][:destination],
            packed.tensors[name][:destination],
        )
    assert packed.canonical_bytes() == input_canonical
    assert first.production_qualified is False
    assert first.publication_qualified is False
    assert first.runtime_build is False


def test_newborn_failure_is_atomic_even_if_native_touches_private_stage(
    oracle, monkeypatch: pytest.MonkeyPatch
) -> None:
    packed = _packed_fixture()
    input_canonical = packed.canonical_bytes()
    input_tensors = {name: tensor.clone() for name, tensor in packed.tensors.items()}

    class FailingLibrary:
        @staticmethod
        def eiel_modular_recurrent_newborn_inherit_v1(rows, *_args):
            rows[int(_newborn_kwargs(packed)["destination_row"])].agent_id = 999_999
            return 10

    failed_artifact = replace(oracle, library=FailingLibrary())
    monkeypatch.setattr(
        oracle_adapter,
        "load_prebuilt_modular_recurrent_cpu_oracle",
        lambda _artifact_path: failed_artifact,
    )
    with pytest.raises(
        ModularRecurrentCpuOracleAdapterError,
        match="newborn transaction failed: status=10",
    ):
        execute_modular_recurrent_cpu_oracle_newborn_v1(
            packed, oracle.artifact_path, **_newborn_kwargs(packed)
        )
    assert packed.canonical_bytes() == input_canonical
    for name in ROW_PLANE_FIELDS:
        assert torch.equal(packed.tensors[name], input_tensors[name])


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("unknown", 0.0, "canonical gated numeric field"),
        ("reset_bias", 5.0, "outside the field bounds"),
    ],
)
def test_newborn_invalid_gated_replacement_fails_before_publish(
    oracle, field: str, replacement: float, message: str
) -> None:
    packed = _packed_fixture()
    kwargs = _newborn_kwargs(packed)
    kwargs["gated_field"] = field
    kwargs["gated_replacement_value"] = replacement
    input_canonical = packed.canonical_bytes()
    with pytest.raises(ModularRecurrentCpuOracleAdapterError, match=message):
        execute_modular_recurrent_cpu_oracle_newborn_v1(
            packed, oracle.artifact_path, **kwargs
        )
    assert packed.canonical_bytes() == input_canonical


def test_bad_artifact_identity_fails_closed(oracle, tmp_path: Path) -> None:
    damaged = tmp_path / oracle.artifact_path.name
    shutil.copyfile(oracle.artifact_path, damaged)
    content = bytearray(damaged.read_bytes())
    content[-1] ^= 1
    damaged.write_bytes(content)
    packed = _packed_fixture()
    observations, fatigue, stages = _tick_inputs(packed.layout.agent_capacity)
    with pytest.raises(
        ModularRecurrentCpuArtifactError, match="artifact hash mismatch"
    ):
        execute_modular_recurrent_cpu_oracle_v1(
            packed, damaged, observations[0], fatigue[0], stages[0]
        )


def test_packed_storage_identity_fails_closed(oracle) -> None:
    packed = _packed_fixture()
    tensors = {name: tensor.clone() for name, tensor in packed.tensors.items()}
    tensors["module_state"][0, 0, 0] += 0.25
    damaged = replace(packed, tensors=tensors)
    observations, fatigue, stages = _tick_inputs(packed.layout.agent_capacity)
    with pytest.raises(ModularStorageError, match="packed modular storage digest"):
        execute_modular_recurrent_cpu_oracle_v1(
            damaged, oracle.artifact_path, observations[0], fatigue[0], stages[0]
        )


@pytest.mark.parametrize(
    "damage",
    [
        "observation_shape",
        "observation_dtype",
        "observation_noncontiguous",
        "observation_device",
        "fatigue_shape",
        "fatigue_dtype",
        "stage_shape",
        "stage_dtype",
        "stage_device",
    ],
)
def test_malformed_tick_tensor_rejected(oracle, damage: str) -> None:
    packed = _packed_fixture()
    observations, fatigue, stages = _tick_inputs(packed.layout.agent_capacity)
    if damage == "observation_shape":
        observations = observations[:, :, :-1].contiguous()
    elif damage == "observation_dtype":
        observations = observations.to(torch.float64)
    elif damage == "observation_noncontiguous":
        backing = torch.empty((*observations.shape, 2), dtype=torch.float32)
        observations = backing[..., 0]
    elif damage == "observation_device":
        observations = torch.empty(observations.shape, device="meta")
    elif damage == "fatigue_shape":
        fatigue = fatigue[:, :-1].contiguous()
    elif damage == "fatigue_dtype":
        fatigue = fatigue.to(torch.float64)
    elif damage == "stage_shape":
        stages = stages[:, :-1].contiguous()
    elif damage == "stage_dtype":
        stages = stages.to(torch.int32)
    else:
        stages = torch.empty(stages.shape, dtype=torch.int64, device="meta")
    with pytest.raises(ModularRecurrentCpuOracleAdapterError):
        execute_modular_recurrent_cpu_oracle_v1(
            packed, oracle.artifact_path, observations, fatigue, stages
        )


def test_adapter_execution_does_not_compile(
    oracle, monkeypatch: pytest.MonkeyPatch
) -> None:
    packed = _packed_fixture()
    observations, fatigue, stages = _tick_inputs(packed.layout.agent_capacity)
    real_run = modular_recurrent_cpu_artifact_v1.subprocess.run
    commands: list[tuple[str, ...]] = []

    def guarded_run(command, *args, **kwargs):
        normalized = tuple(str(part) for part in command)
        commands.append(normalized)
        if normalized[0] == "g++" and normalized != ("g++", "--version"):
            raise AssertionError(f"unexpected compiler execution: {normalized}")
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(
        modular_recurrent_cpu_artifact_v1.subprocess, "run", guarded_run
    )
    execute_modular_recurrent_cpu_oracle_v1(
        packed, oracle.artifact_path, observations[0], fatigue[0], stages[0]
    )
    assert commands[0] == ("g++", "--version")
    assert commands[1][0:3] == ("nm", "-D", "--defined-only")
    assert len(commands) == 2
