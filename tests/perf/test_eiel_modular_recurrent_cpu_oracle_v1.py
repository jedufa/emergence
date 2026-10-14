from __future__ import annotations

import copy
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import struct

import numpy as np
import pytest
import torch

from engine.perf.cuda_architecture_b_modular_state_abi import pack_modular_storage
from engine.perf.cuda_architecture_b_modular_tick import execute_modular_tick
from engine.evolution.p12_native_cpu_parity_progress_v1 import (
    NATIVE_CPU_ROUTE_BOUND_ARTIFACT_SHA256,
    NATIVE_CPU_ROUTE_BOUND_BUILD_MANIFEST_SHA256,
    native_cpu_recurrent_lifecycle_owner_progress_v1,
    native_cpu_progress_sha256_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    build_modular_recurrent_mutation_receipt_v1,
    canonical_modular_recurrent_mutation_receipt_bytes_v1,
)
from engine.perf.eiel_phase_e_modular_tick_fixture_v1 import (
    AGENT_COUNT,
    EXPANDED_B2_CAPS,
    LOCAL_SIZE,
    TICK_AGES,
    _build_population,
    _fixed_inputs,
)
from engine.perf.recurrent_native.modular_recurrent_cpu_artifact_v1 import (
    ACTION_WIDTH,
    DEFAULT_BUILD_MANIFEST,
    OBSERVATION_WIDTH,
    ROW_PLANE_FIELDS,
    ConstOwnerViewV1,
    CostCapacityReceiptV1,
    ModularRecurrentCpuArtifactError,
    MutationV1,
    OwnerViewV1,
    RowV1,
    TickReceiptV1,
    build_prebuilt_modular_recurrent_cpu_oracle,
    load_prebuilt_modular_recurrent_cpu_oracle,
)


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "engine/perf/native/recurrent"
PROBE = Path(__file__).with_name("modular_recurrent_cpu_oracle_probe_v1.cpp")
PROFILE = 1
MUTABLE_PLANES = {
    "module_state_present",
    "module_state",
    "module_output_present",
    "module_previous_output",
    "development_latch_present",
    "development_latch_state",
    "development_effective_gain_present",
    "development_effective_gain",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _packed_fixture(*, extra_empty_rows: int = 0):
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


def _rows_from_planes(planes: dict[str, torch.Tensor]) -> ctypes.Array[RowV1]:
    row_count = int(planes["agent_occupied"].shape[0])
    rows = (RowV1 * row_count)()
    for row_index in range(row_count):
        row = rows[row_index]
        for name in ROW_PLANE_FIELDS:
            value = planes[name][row_index].detach().cpu().contiguous().reshape(-1)
            destination = getattr(row, name)
            if isinstance(destination, ctypes.Array):
                assert len(destination) == value.numel(), name
                for index, item in enumerate(value.tolist()):
                    destination[index] = item
            else:
                setattr(row, name, value.item())
    return rows


def _planes_from_rows(
    rows: ctypes.Array[RowV1], template: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    result = {name: value.clone() for name, value in template.items()}
    for row_index, row in enumerate(rows):
        for name in ROW_PLANE_FIELDS:
            target = result[name][row_index]
            source = getattr(row, name)
            values = list(source) if isinstance(source, ctypes.Array) else [source]
            replacement = torch.tensor(values, dtype=target.dtype).reshape(target.shape)
            target.copy_(replacement)
    return result


class _OwnerBundle:
    def __init__(self, planes: dict[str, torch.Tensor]) -> None:
        ids = planes["agent_id"].tolist()
        tokens = [
            int(value) if bool(planes["agent_occupied"][index]) else 90_000 + index
            for index, value in enumerate(planes["agent_slot_token"].tolist())
        ]
        alive = [int(value) for value in planes["agent_occupied"].tolist()]
        count = len(ids)
        self.ids = (ctypes.c_int64 * count)(*ids)
        self.tokens = (ctypes.c_int64 * count)(*tokens)
        self.alive = (ctypes.c_uint8 * count)(*alive)
        self.mutable = OwnerViewV1(self.ids, self.tokens, self.alive, count)
        self.const = ConstOwnerViewV1(self.ids, self.tokens, self.alive, count)


class BirthReceiptV1(ctypes.Structure):
    _fields_ = [
        ("receipt_version", ctypes.c_uint32),
        ("lifecycle_status", ctypes.c_int32),
        ("committed", ctypes.c_uint8),
        ("owner_checked", ctypes.c_uint8),
        ("parent_preserved", ctypes.c_uint8),
        ("child_runtime_reset", ctypes.c_uint8),
        ("canonical_receipt_bound", ctypes.c_uint8),
        ("canonical_receipt_identity_checked", ctypes.c_uint8),
        ("recurrent_receipt_bytes_bound", ctypes.c_uint8),
        ("canonical_receipt_reserved", ctypes.c_uint8),
        ("destination_row", ctypes.c_int64),
        ("parent_a_row", ctypes.c_int64),
        ("parent_b_row", ctypes.c_int64),
        ("selected_parent_row", ctypes.c_int64),
        ("expected_destination_token", ctypes.c_int64),
        ("expected_parent_a_token", ctypes.c_int64),
        ("expected_parent_b_token", ctypes.c_int64),
        ("parent_a_agent_id", ctypes.c_int64),
        ("parent_b_agent_id", ctypes.c_int64),
        ("child_agent_id", ctypes.c_int64),
        ("publication_token", ctypes.c_int64),
        ("donor_draw_float32_bits", ctypes.c_uint32),
        ("mutation_count", ctypes.c_int64),
        ("draw0_float64_bits", ctypes.c_uint64),
        ("draw1_float64_bits", ctypes.c_uint64),
        ("canonical_recurrent_receipt_sha256_words", ctypes.c_uint64 * 4),
        ("canonical_recurrent_receipt_bytes_sha256_words", ctypes.c_uint64 * 4),
        ("transaction_ordinal", ctypes.c_int64),
        ("selected_instance_id_index", ctypes.c_int32),
        ("selected_homology_id_index", ctypes.c_int32),
        ("operation_code", ctypes.c_int32),
        ("status_code", ctypes.c_int32),
        ("reason_code", ctypes.c_int32),
        ("changed", ctypes.c_uint8),
        ("canonical_receipt_reserved_tail", ctypes.c_uint8 * 7),
        ("structural_payload_sha256_words", ctypes.c_uint64 * 4),
        ("structural_payload_sha256_bound", ctypes.c_uint8),
        ("structural_payload_identity_checked", ctypes.c_uint8),
        ("structural_payload_reserved_tail", ctypes.c_uint8 * 6),
    ]


class CanonicalReceiptBindingV1(ctypes.Structure):
    _fields_ = [
        ("binding_version", ctypes.c_uint32),
        ("canonical_receipt_contract_version", ctypes.c_uint32),
        ("canonical_receipt_sha256_words", ctypes.c_uint64 * 4),
        ("canonical_receipt_bytes_sha256_words", ctypes.c_uint64 * 4),
        ("draw0_float64_bits", ctypes.c_uint64),
        ("draw1_float64_bits", ctypes.c_uint64),
        ("transaction_ordinal", ctypes.c_int64),
        ("parent_agent_id", ctypes.c_int64),
        ("parent_slot", ctypes.c_int64),
        ("parent_slot_token", ctypes.c_int64),
        ("child_agent_id", ctypes.c_int64),
        ("child_slot", ctypes.c_int64),
        ("child_slot_token", ctypes.c_int64),
        ("selected_instance_id_index", ctypes.c_int32),
        ("selected_homology_id_index", ctypes.c_int32),
        ("operation_code", ctypes.c_int32),
        ("status_code", ctypes.c_int32),
        ("reason_code", ctypes.c_int32),
        ("changed", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 7),
        ("structural_payload_sha256_words", ctypes.c_uint64 * 4),
        ("changed_dimension_code", ctypes.c_int32),
        ("native_receipt_construction_required", ctypes.c_uint8),
        ("reserved_tail", ctypes.c_uint8 * 3),
    ]


class CanonicalReceiptSourceV1(ctypes.Structure):
    _fields_ = [
        ("source_version", ctypes.c_uint32),
        ("reserved0", ctypes.c_uint32),
        ("source_packed_sha256_words", ctypes.c_uint64 * 4),
        ("ancestry_tape_sha256_words", ctypes.c_uint64 * 4),
    ]


def _float64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _sha256_words(value: str) -> ctypes.Array[ctypes.c_uint64]:
    raw = bytes.fromhex(value)
    assert len(raw) == 32
    return (ctypes.c_uint64 * 4)(
        *[int.from_bytes(raw[index : index + 8], "big") for index in range(0, 32, 8)]
    )


def _words_to_sha256(words: ctypes.Array[ctypes.c_uint64]) -> str:
    return "".join(f"{int(word):016x}" for word in words)


def _native_structural_payload_sha256(library: ctypes.CDLL, row: RowV1) -> str:
    words = (ctypes.c_uint64 * 4)()
    status = library.eiel_modular_recurrent_row_structural_payload_sha256_v1(
        ctypes.byref(row),
        words,
    )
    assert status == 0
    return _words_to_sha256(words)


def _reset_runtime(row: RowV1) -> None:
    for index in range(16):
        row.module_state_present[index] = 0
        row.module_state[index] = ctypes.c_float(0.0).value
        row.module_output_present[index] = 0
        row.module_previous_output[index] = ctypes.c_float(0.0).value


def _apply_mutation(row: RowV1, mutation: MutationV1) -> None:
    if mutation.field == 0:
        row.integrator_parameters[mutation.outer_index * 4 + mutation.inner_index] += (
            mutation.additive_delta
        )
    elif mutation.field == 1:
        row.gated_parameters[mutation.outer_index * 10 + mutation.inner_index] += (
            mutation.additive_delta
        )
    elif mutation.field == 2:
        row.input_gain[mutation.outer_index * 2 + mutation.inner_index] += (
            mutation.additive_delta
        )
    elif mutation.field == 3:
        row.output_gain[mutation.outer_index * 2 + mutation.inner_index] += (
            mutation.additive_delta
        )
    elif mutation.field == 4:
        row.development_stage_gains[
            mutation.outer_index * 3 + mutation.inner_index
        ] += mutation.additive_delta
    elif mutation.field == 5:
        row.connection_weight[mutation.outer_index] += mutation.additive_delta
    else:
        raise AssertionError(f"unsupported fixture mutation field: {mutation.field}")


def _expected_newborn_structural_sha256(
    library: ctypes.CDLL,
    rows: ctypes.Array[RowV1],
    *,
    parent_row: int,
    child_agent_id: int,
    publication_token: int,
    mutation: MutationV1,
) -> str:
    candidate = copy.deepcopy(rows[parent_row])
    candidate.agent_id = child_agent_id
    candidate.agent_slot_token = publication_token
    _reset_runtime(candidate)
    _apply_mutation(candidate, mutation)
    return _native_structural_payload_sha256(library, candidate)


def _canonical_binding(
    *,
    parent_agent_id: int,
    parent_slot: int,
    parent_slot_token: int,
    child_agent_id: int,
    child_slot: int,
    child_slot_token: int,
    draw0: float = 0.625,
    draw1: float = 0.5,
    binding_version: int = 2,
    structural_payload_sha256: str = "33" * 32,
) -> tuple[CanonicalReceiptBindingV1, str, str, str]:
    receipt = build_modular_recurrent_mutation_receipt_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        transactions=[
            {
                "transaction_kind": "ordinary_birth",
                "transaction_ordinal": 0,
                "parent_agent_id": parent_agent_id,
                "parent_slot": parent_slot,
                "parent_slot_token": parent_slot_token,
                "child_agent_id": child_agent_id,
                "child_slot": child_slot,
                "child_slot_token": child_slot_token,
                "attempt_ordinal": 0,
                "draw0_float64_bits": f"{_float64_bits(draw0):016x}",
                "draw1_float64_bits": f"{_float64_bits(draw1):016x}",
                "operation": "refine_descendant_parameter",
                "status": "accepted",
                "reason": "proposal_valid",
                "selected_instance_id_index": 0,
                "selected_homology_id_index": 1,
                "changed_dimension": "live_input_gain",
                "changed": True,
            }
        ],
    )
    receipt_bytes_sha256 = hashlib.sha256(
        canonical_modular_recurrent_mutation_receipt_bytes_v1(receipt)
    ).hexdigest()
    binding = CanonicalReceiptBindingV1()
    binding.binding_version = binding_version
    binding.canonical_receipt_contract_version = 1
    binding.canonical_receipt_sha256_words = _sha256_words(receipt.canonical_sha256)
    binding.canonical_receipt_bytes_sha256_words = _sha256_words(receipt_bytes_sha256)
    binding.structural_payload_sha256_words = _sha256_words(structural_payload_sha256)
    binding.draw0_float64_bits = _float64_bits(draw0)
    binding.draw1_float64_bits = _float64_bits(draw1)
    binding.transaction_ordinal = 0
    binding.parent_agent_id = parent_agent_id
    binding.parent_slot = parent_slot
    binding.parent_slot_token = parent_slot_token
    binding.child_agent_id = child_agent_id
    binding.child_slot = child_slot
    binding.child_slot_token = child_slot_token
    binding.selected_instance_id_index = 0
    binding.selected_homology_id_index = 1
    binding.operation_code = 2
    binding.status_code = 1
    binding.reason_code = 1
    binding.changed = 1
    binding.changed_dimension_code = 3
    binding.native_receipt_construction_required = 1 if binding_version >= 4 else 0
    return binding, receipt.canonical_sha256, receipt_bytes_sha256, structural_payload_sha256


def _canonical_source() -> CanonicalReceiptSourceV1:
    source = CanonicalReceiptSourceV1()
    source.source_version = 1
    source.source_packed_sha256_words = _sha256_words("11" * 32)
    source.ancestry_tape_sha256_words = _sha256_words("22" * 32)
    return source


def _assert_native_canonical_receipt_sha256_matches(
    library: ctypes.CDLL,
    binding: CanonicalReceiptBindingV1,
    canonical_sha256: str,
    canonical_bytes_sha256: str,
) -> None:
    source = _canonical_source()
    canonical_words = (ctypes.c_uint64 * 4)()
    canonical_bytes_words = (ctypes.c_uint64 * 4)()
    status = library.eiel_modular_recurrent_canonical_receipt_sha256_v1(
        PROFILE,
        ctypes.byref(source),
        ctypes.byref(binding),
        canonical_words,
        canonical_bytes_words,
    )
    assert status == 0
    assert list(canonical_words) == list(_sha256_words(canonical_sha256))
    assert list(canonical_bytes_words) == list(_sha256_words(canonical_bytes_sha256))


def _assert_native_canonical_receipt_sha256_rejects(
    library: ctypes.CDLL,
    binding: CanonicalReceiptBindingV1,
) -> None:
    source = _canonical_source()
    canonical_words = (ctypes.c_uint64 * 4)()
    canonical_bytes_words = (ctypes.c_uint64 * 4)()
    status = library.eiel_modular_recurrent_canonical_receipt_sha256_v1(
        PROFILE,
        ctypes.byref(source),
        ctypes.byref(binding),
        canonical_words,
        canonical_bytes_words,
    )
    assert status == 5


def _assert_project_death_restore_round_trip(
    library: ctypes.CDLL,
    rows: ctypes.Array[RowV1],
    owners: _OwnerBundle,
    *,
    row_index: int,
    expected_structural_sha256: str,
) -> str:
    committed = rows[row_index]
    agent_id = int(committed.agent_id)
    slot_token = int(committed.agent_slot_token)
    checkpoint = RowV1()
    assert (
        library.eiel_modular_recurrent_checkpoint_project_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.const),
            row_index,
            slot_token,
            ctypes.byref(checkpoint),
        )
        == 0
    )
    checkpoint_bytes = bytes(checkpoint)
    assert hashlib.sha256(bytes(rows[row_index])).hexdigest() == hashlib.sha256(
        checkpoint_bytes
    ).hexdigest()

    assert (
        library.eiel_modular_recurrent_death_cleanup_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            row_index,
            agent_id,
            slot_token,
        )
        == 0
    )
    assert owners.ids[row_index] == -1
    assert owners.alive[row_index] == 0

    assert (
        library.eiel_modular_recurrent_checkpoint_restore_v1(
            ctypes.byref(checkpoint),
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            row_index,
            owners.tokens[row_index],
        )
        == 0
    )
    assert bytes(rows[row_index]) == checkpoint_bytes
    assert owners.ids[row_index] == agent_id
    assert owners.tokens[row_index] == slot_token
    assert owners.alive[row_index] == 1
    assert _native_structural_payload_sha256(library, rows[row_index]) == (
        expected_structural_sha256
    )
    return hashlib.sha256(checkpoint_bytes).hexdigest()


def _route_bound_generation_turnover_sequence(
    library: ctypes.CDLL,
    *,
    restore_prefix_checkpoint: bool,
) -> dict[str, object]:
    packed = _packed_fixture(extra_empty_rows=1)
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    source = _canonical_source()
    destination = AGENT_COUNT
    parent_a = 0
    parent_b = 1
    prefix_mutation = MutationV1(0, 0, 2, ctypes.c_float(0.125).value)
    prefix_child_agent_id = (1 << 40) + 10_001
    prefix_publication_token = 81_001
    prefix_structural_sha256 = _expected_newborn_structural_sha256(
        library,
        rows,
        parent_row=parent_a,
        child_agent_id=prefix_child_agent_id,
        publication_token=prefix_publication_token,
        mutation=prefix_mutation,
    )
    prefix_binding, prefix_canonical_sha256, prefix_bytes_sha256, _ = (
        _canonical_binding(
            parent_agent_id=int(packed.tensors["agent_id"][parent_a]),
            parent_slot=parent_a,
            parent_slot_token=owners.tokens[parent_a],
            child_agent_id=prefix_child_agent_id,
            child_slot=destination,
            child_slot_token=prefix_publication_token,
            binding_version=4,
            structural_payload_sha256=prefix_structural_sha256,
        )
    )
    _assert_native_canonical_receipt_sha256_matches(
        library,
        prefix_binding,
        prefix_canonical_sha256,
        prefix_bytes_sha256,
    )
    prefix_receipt = BirthReceiptV1()
    assert (
        library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            destination,
            owners.tokens[destination],
            parent_a,
            owners.tokens[parent_a],
            parent_b,
            owners.tokens[parent_b],
            ctypes.c_float(0.25),
            prefix_child_agent_id,
            prefix_publication_token,
            ctypes.byref(prefix_mutation),
            1,
            ctypes.byref(source),
            ctypes.byref(prefix_binding),
            ctypes.byref(prefix_receipt),
        )
        == 0
    )
    assert prefix_receipt.receipt_version == 4
    assert prefix_receipt.transaction_ordinal == 0
    assert prefix_receipt.recurrent_receipt_bytes_bound == 1
    assert prefix_receipt.structural_payload_sha256_bound == 1

    prefix_checkpoint = RowV1()
    assert (
        library.eiel_modular_recurrent_checkpoint_project_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.const),
            destination,
            prefix_publication_token,
            ctypes.byref(prefix_checkpoint),
        )
        == 0
    )
    prefix_checkpoint_sha256 = hashlib.sha256(bytes(prefix_checkpoint)).hexdigest()

    if restore_prefix_checkpoint:
        rows = _rows_from_planes(packed.tensors)
        owners = _OwnerBundle(packed.tensors)
        assert (
            library.eiel_modular_recurrent_checkpoint_restore_v1(
                ctypes.byref(prefix_checkpoint),
                rows,
                len(rows),
                PROFILE,
                ctypes.byref(owners.mutable),
                destination,
                owners.tokens[destination],
            )
            == 0
        )
        assert rows[destination].agent_id == prefix_child_agent_id
        assert rows[destination].agent_slot_token == prefix_publication_token
        assert _native_structural_payload_sha256(library, rows[destination]) == (
            prefix_structural_sha256
        )

    pre_suffix_bank_sha256 = hashlib.sha256(bytes(rows)).hexdigest()
    pre_suffix_owner_state = (
        tuple(int(value) for value in owners.ids),
        tuple(int(value) for value in owners.tokens),
        tuple(int(value) for value in owners.alive),
    )

    suffix_mutation = MutationV1(0, 0, 2, ctypes.c_float(0.0625).value)
    suffix_child_agent_id = (1 << 40) + 10_002
    suffix_publication_token = 81_002
    suffix_structural_sha256 = _expected_newborn_structural_sha256(
        library,
        rows,
        parent_row=parent_a,
        child_agent_id=suffix_child_agent_id,
        publication_token=suffix_publication_token,
        mutation=suffix_mutation,
    )
    suffix_binding, suffix_canonical_sha256, suffix_bytes_sha256, _ = (
        _canonical_binding(
            parent_agent_id=int(rows[parent_a].agent_id),
            parent_slot=parent_a,
            parent_slot_token=owners.tokens[parent_a],
            child_agent_id=suffix_child_agent_id,
            child_slot=destination,
            child_slot_token=suffix_publication_token,
            binding_version=4,
            structural_payload_sha256=suffix_structural_sha256,
        )
    )
    _assert_native_canonical_receipt_sha256_matches(
        library,
        suffix_binding,
        suffix_canonical_sha256,
        suffix_bytes_sha256,
    )
    suffix_receipt = BirthReceiptV1()
    assert (
        library.eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            destination,
            prefix_child_agent_id,
            prefix_publication_token,
            parent_a,
            owners.tokens[parent_a],
            suffix_child_agent_id,
            suffix_publication_token,
            ctypes.byref(suffix_mutation),
            1,
            ctypes.byref(source),
            ctypes.byref(suffix_binding),
            ctypes.byref(suffix_receipt),
        )
        == 0
    )
    assert suffix_receipt.receipt_version == 4
    assert suffix_receipt.transaction_ordinal == 0
    assert suffix_receipt.recurrent_receipt_bytes_bound == 1
    assert suffix_receipt.structural_payload_sha256_bound == 1
    assert _native_structural_payload_sha256(library, rows[destination]) == (
        suffix_structural_sha256
    )

    return {
        "prefix_checkpoint_sha256": prefix_checkpoint_sha256,
        "prefix_receipt_sha256": hashlib.sha256(bytes(prefix_receipt)).hexdigest(),
        "prefix_structural_payload_sha256": prefix_structural_sha256,
        "pre_suffix_bank_sha256": pre_suffix_bank_sha256,
        "pre_suffix_owner_state": pre_suffix_owner_state,
        "suffix_receipt_sha256": hashlib.sha256(bytes(suffix_receipt)).hexdigest(),
        "suffix_canonical_receipt_sha256": suffix_canonical_sha256,
        "suffix_canonical_receipt_bytes_sha256": suffix_bytes_sha256,
        "suffix_structural_payload_sha256": suffix_structural_sha256,
        "final_bank_sha256": hashlib.sha256(bytes(rows)).hexdigest(),
        "final_owner_state": (
            tuple(int(value) for value in owners.ids),
            tuple(int(value) for value in owners.tokens),
            tuple(int(value) for value in owners.alive),
        ),
    }


def _call_tick(
    library: ctypes.CDLL,
    rows: ctypes.Array[RowV1],
    owners: _OwnerBundle,
    observations: np.ndarray,
    fatigue: np.ndarray,
    stages: np.ndarray,
    *,
    output_sentinel: float = -991.0,
) -> tuple[int, np.ndarray, TickReceiptV1]:
    row_count = len(rows)
    observations = np.ascontiguousarray(observations, dtype=np.float32)
    fatigue = np.ascontiguousarray(fatigue, dtype=np.float32)
    stages = np.ascontiguousarray(stages, dtype=np.int64)
    scratch_rows = (RowV1 * row_count)()
    scratch_actions = np.empty((row_count, ACTION_WIDTH), dtype=np.float32)
    actions = np.full((row_count, ACTION_WIDTH), output_sentinel, dtype=np.float32)
    receipt = TickReceiptV1(-991, -992, -993, -994, -995, -996, 0xA5, 0x5A)
    status = library.eiel_modular_recurrent_tick_v1(
        rows,
        row_count,
        PROFILE,
        ctypes.byref(owners.const),
        observations.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        observations.size,
        fatigue.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fatigue.size,
        stages.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        stages.size,
        scratch_rows,
        row_count,
        scratch_actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scratch_actions.size,
        actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        actions.size,
        ctypes.byref(receipt),
    )
    return int(status), actions, receipt


def _configure_birth_receipt_symbols(library: ctypes.CDLL) -> None:
    uint64_p = ctypes.POINTER(ctypes.c_uint64)
    library.eiel_modular_recurrent_row_structural_payload_sha256_v1.argtypes = [
        ctypes.POINTER(RowV1),
        uint64_p,
    ]
    library.eiel_modular_recurrent_canonical_receipt_sha256_v1.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(CanonicalReceiptSourceV1),
        ctypes.POINTER(CanonicalReceiptBindingV1),
        uint64_p,
        uint64_p,
    ]
    library.eiel_modular_recurrent_canonical_receipt_sha256_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_modular_recurrent_row_structural_payload_sha256_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_modular_recurrent_newborn_inherit_with_receipt_v1.argtypes = [
        ctypes.POINTER(RowV1),
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.POINTER(OwnerViewV1),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(MutationV1),
        ctypes.c_int64,
        ctypes.POINTER(BirthReceiptV1),
    ]
    library.eiel_modular_recurrent_newborn_inherit_with_receipt_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1.argtypes = [
        ctypes.POINTER(RowV1),
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.POINTER(OwnerViewV1),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(MutationV1),
        ctypes.c_int64,
        ctypes.POINTER(CanonicalReceiptSourceV1),
        ctypes.POINTER(CanonicalReceiptBindingV1),
        ctypes.POINTER(BirthReceiptV1),
    ]
    library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_modular_recurrent_same_slot_reuse_with_receipt_v1.argtypes = [
        ctypes.POINTER(RowV1),
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.POINTER(OwnerViewV1),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(MutationV1),
        ctypes.c_int64,
        ctypes.POINTER(BirthReceiptV1),
    ]
    library.eiel_modular_recurrent_same_slot_reuse_with_receipt_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1.argtypes = [
        ctypes.POINTER(RowV1),
        ctypes.c_int64,
        ctypes.c_uint32,
        ctypes.POINTER(OwnerViewV1),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(MutationV1),
        ctypes.c_int64,
        ctypes.POINTER(CanonicalReceiptSourceV1),
        ctypes.POINTER(CanonicalReceiptBindingV1),
        ctypes.POINTER(BirthReceiptV1),
    ]
    library.eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1.restype = (
        ctypes.c_int32
    )


@pytest.fixture(scope="module")
def oracle(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("modular_recurrent_cpu_oracle")
    first = build_prebuilt_modular_recurrent_cpu_oracle(root / "build1")
    second = build_prebuilt_modular_recurrent_cpu_oracle(root / "build2")
    assert first.read_bytes() == second.read_bytes()
    assert _sha256(first) == _sha256(second)
    return load_prebuilt_modular_recurrent_cpu_oracle(first)


def test_exact_57_plane_schema_and_three_tick_object_model_parity(oracle) -> None:
    packed = _packed_fixture()
    assert tuple(packed.tensors) == ROW_PLANE_FIELDS
    python_planes = {name: tensor.clone() for name, tensor in packed.tensors.items()}
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    observations, _base, _policy, _masks = _fixed_inputs()

    for tick, _age in enumerate(TICK_AGES):
        expected = execute_modular_tick(
            python_planes,
            torch.from_numpy(observations[tick].copy()),
            torch.from_numpy(observations[tick, :, LOCAL_SIZE].copy()),
            torch.full((AGENT_COUNT,), tick, dtype=torch.int64),
        )
        status, residuals, receipt = _call_tick(
            oracle.library,
            rows,
            owners,
            observations[tick],
            observations[tick, :, LOCAL_SIZE],
            np.full((AGENT_COUNT,), tick, dtype=np.int64),
        )
        assert status == 0
        assert receipt.committed_rows == AGENT_COUNT
        assert receipt.rng_draws == 0
        assert receipt.production_routing_qualified == 0
        assert receipt.publication_qualified == 0
        assert torch.equal(torch.from_numpy(residuals), expected.action_residuals)
        native_planes = _planes_from_rows(rows, packed.tensors)
        for name in ROW_PLANE_FIELDS:
            assert torch.equal(native_planes[name], python_planes[name]), name


def test_lifecycle_isolation_deterministic_inheritance_and_same_slot_reuse(
    oracle,
) -> None:
    def scenario() -> tuple[bytes, list[int], list[int], list[int]]:
        packed = _packed_fixture(extra_empty_rows=1)
        rows = _rows_from_planes(packed.tensors)
        owners = _OwnerBundle(packed.tensors)
        destination = AGENT_COUNT
        mutation = MutationV1(0, 0, 2, ctypes.c_float(0.125).value)
        status = oracle.library.eiel_modular_recurrent_newborn_inherit_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            destination,
            owners.tokens[destination],
            0,
            owners.tokens[0],
            1,
            owners.tokens[1],
            ctypes.c_float(0.25),
            77,
            70_077,
            ctypes.byref(mutation),
            1,
        )
        assert status == 0
        child = rows[destination]
        assert child.integrator_parameters[2] == pytest.approx(
            rows[0].integrator_parameters[2] + 0.125
        )
        assert list(child.module_state_present) == [0] * 16
        assert list(child.module_state) == [0.0] * 16
        before_parent = bytes(rows[0])
        status = oracle.library.eiel_modular_recurrent_same_slot_reuse_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            2,
            owners.ids[2],
            owners.tokens[2],
            0,
            owners.tokens[0],
            88,
            80_088,
            None,
            0,
        )
        assert status == 0
        assert bytes(rows[0]) == before_parent
        assert rows[2].agent_id == 88 and rows[2].agent_slot_token == 80_088
        stale_before = bytes(rows[3])
        status = oracle.library.eiel_modular_recurrent_death_cleanup_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            3,
            owners.ids[3],
            owners.tokens[3] + 1,
        )
        assert status == 5 and bytes(rows[3]) == stale_before
        status = oracle.library.eiel_modular_recurrent_death_cleanup_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            3,
            owners.ids[3],
            owners.tokens[3],
        )
        assert status == 0
        assert owners.alive[3] == 0 and owners.ids[3] == -1
        status = oracle.library.eiel_modular_recurrent_founder_init_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            3,
            owners.tokens[3],
            99,
            90_099,
            ctypes.byref(rows[1]),
        )
        assert status == 0
        assert rows[3].agent_id == 99
        assert list(rows[3].module_previous_output) == [0.0] * 16
        return (
            bytes(rows),
            list(owners.ids),
            list(owners.tokens),
            list(owners.alive),
        )

    assert scenario() == scenario()


def test_p12_native_cpu_recurrent_lifecycle_owner_surfaces_match_answer_key_subset(
    oracle,
) -> None:
    receipt = native_cpu_recurrent_lifecycle_owner_progress_v1()
    assert receipt["native_cpu_a_authorized"] is False
    assert receipt["cuda_promotion_authorized"] is False

    packed = _packed_fixture(extra_empty_rows=1)
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    destination = AGENT_COUNT
    parent_slot = 0
    sibling_slot = 1
    child_agent_id = (1 << 40) + 77
    child_slot_token = 70_077
    mutation = MutationV1(0, 0, 2, ctypes.c_float(0.125).value)
    parent_before = bytes(rows[parent_slot])
    bank_before = bytes(rows)
    owner_before = (
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    )

    status = oracle.library.eiel_modular_recurrent_newborn_inherit_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.mutable),
        destination,
        owners.tokens[destination],
        parent_slot,
        owners.tokens[parent_slot],
        sibling_slot,
        owners.tokens[sibling_slot],
        ctypes.c_float(0.25),
        child_agent_id,
        child_slot_token,
        ctypes.byref(mutation),
        1,
    )

    assert status == 0
    assert bytes(rows[parent_slot]) == parent_before
    assert owners.ids[destination] == child_agent_id
    assert owners.tokens[destination] == child_slot_token
    assert owners.alive[destination] == 1
    assert rows[destination].agent_id == child_agent_id
    assert rows[destination].agent_slot_token == child_slot_token
    assert rows[destination].integrator_parameters[2] == pytest.approx(
        rows[parent_slot].integrator_parameters[2] + 0.125
    )
    assert list(rows[destination].module_state_present) == [0] * 16
    assert list(rows[destination].module_state) == [0.0] * 16

    checkpoint = RowV1()
    assert (
        oracle.library.eiel_modular_recurrent_checkpoint_project_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.const),
            destination,
            child_slot_token,
            ctypes.byref(checkpoint),
        )
        == 0
    )
    child_checkpoint_bytes = bytes(checkpoint)

    before_stale_reuse = (
        bytes(rows),
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    )
    stale_reuse_status = oracle.library.eiel_modular_recurrent_same_slot_reuse_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.mutable),
        destination,
        child_agent_id,
        child_slot_token + 1,
        parent_slot,
        owners.tokens[parent_slot],
        child_agent_id + 1,
        child_slot_token + 1,
        None,
        0,
    )
    assert stale_reuse_status == 5
    assert (
        bytes(rows),
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    ) == before_stale_reuse

    fresh_agent_id = child_agent_id + 1
    fresh_slot_token = child_slot_token + 1
    assert (
        oracle.library.eiel_modular_recurrent_same_slot_reuse_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            destination,
            child_agent_id,
            child_slot_token,
            parent_slot,
            owners.tokens[parent_slot],
            fresh_agent_id,
            fresh_slot_token,
            None,
            0,
        )
        == 0
    )
    assert owners.ids[destination] == fresh_agent_id
    assert owners.tokens[destination] == fresh_slot_token
    assert rows[destination].agent_id == fresh_agent_id
    assert rows[destination].agent_slot_token == fresh_slot_token

    before_stale_death = (
        bytes(rows),
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    )
    assert (
        oracle.library.eiel_modular_recurrent_death_cleanup_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            destination,
            fresh_agent_id,
            fresh_slot_token + 1,
        )
        == 5
    )
    assert (
        bytes(rows),
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    ) == before_stale_death

    assert (
        oracle.library.eiel_modular_recurrent_death_cleanup_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            destination,
            fresh_agent_id,
            fresh_slot_token,
        )
        == 0
    )
    assert owners.ids[destination] == -1
    assert owners.alive[destination] == 0

    assert (
        oracle.library.eiel_modular_recurrent_checkpoint_restore_v1(
            ctypes.byref(checkpoint),
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            destination,
            fresh_slot_token,
        )
        == 0
    )
    assert bytes(rows[destination]) == child_checkpoint_bytes
    assert owners.ids[destination] == child_agent_id
    assert owners.tokens[destination] == child_slot_token
    assert owners.alive[destination] == 1

    parity_payload = {
        "case_id": receipt["case_id"],
        "covered_compare_fields": receipt["covered_compare_fields"],
        "parent_slot": parent_slot,
        "child_slot": destination,
        "parent_slot_token": owner_before[1][parent_slot],
        "child_slot_token": child_slot_token,
        "child_agent_id": child_agent_id,
        "publication_state": "native_child_published",
        "rollback_state": "stale_token_rejected_without_state_change",
        "recurrent_payload_sha256": hashlib.sha256(child_checkpoint_bytes).hexdigest(),
        "checkpoint_bytes": hashlib.sha256(child_checkpoint_bytes).hexdigest(),
        "bank_before_sha256": hashlib.sha256(bank_before).hexdigest(),
    }
    assert native_cpu_progress_sha256_v1(parity_payload)


def test_p12_native_cpu_newborn_inherit_emits_route_bound_birth_receipt(
    oracle,
) -> None:
    _configure_birth_receipt_symbols(oracle.library)
    packed = _packed_fixture(extra_empty_rows=1)
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    destination = AGENT_COUNT
    parent_a = 0
    parent_b = 1
    donor_draw = ctypes.c_float(0.25)
    mutation = MutationV1(0, 0, 2, ctypes.c_float(0.125).value)
    receipt = BirthReceiptV1()
    parent_a_before = bytes(rows[parent_a])
    parent_b_before = bytes(rows[parent_b])
    bank_before = bytes(rows)

    status = oracle.library.eiel_modular_recurrent_newborn_inherit_with_receipt_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.mutable),
        destination,
        owners.tokens[destination],
        parent_a,
        owners.tokens[parent_a],
        parent_b,
        owners.tokens[parent_b],
        donor_draw,
        (1 << 40) + 117,
        71_117,
        ctypes.byref(mutation),
        1,
        ctypes.byref(receipt),
    )

    assert status == 0
    assert receipt.receipt_version == 1
    assert receipt.lifecycle_status == status
    assert receipt.committed == 1
    assert receipt.owner_checked == 1
    assert receipt.parent_preserved == 1
    assert receipt.child_runtime_reset == 1
    assert receipt.destination_row == destination
    assert receipt.parent_a_row == parent_a
    assert receipt.parent_b_row == parent_b
    assert receipt.selected_parent_row == parent_a
    assert receipt.expected_destination_token == 90_000 + destination
    assert receipt.expected_parent_a_token == 10_000 + parent_a
    assert receipt.expected_parent_b_token == 10_000 + parent_b
    assert receipt.parent_a_agent_id == int(packed.tensors["agent_id"][parent_a])
    assert receipt.parent_b_agent_id == int(packed.tensors["agent_id"][parent_b])
    assert receipt.child_agent_id == (1 << 40) + 117
    assert receipt.publication_token == 71_117
    assert receipt.donor_draw_float32_bits == int.from_bytes(
        bytes(donor_draw), "little"
    )
    assert receipt.mutation_count == 1
    assert bytes(rows[parent_a]) == parent_a_before
    assert bytes(rows[parent_b]) == parent_b_before
    assert bytes(rows) != bank_before
    assert rows[destination].agent_id == (1 << 40) + 117
    assert rows[destination].agent_slot_token == 71_117
    assert list(rows[destination].module_state_present) == [0] * 16
    assert list(rows[destination].module_state) == [0.0] * 16

    checkpoint = RowV1()
    assert (
        oracle.library.eiel_modular_recurrent_checkpoint_project_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.const),
            destination,
            71_117,
            ctypes.byref(checkpoint),
        )
        == 0
    )
    assert (
        oracle.library.eiel_modular_recurrent_death_cleanup_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            destination,
            (1 << 40) + 117,
            71_117,
        )
        == 0
    )
    assert (
        oracle.library.eiel_modular_recurrent_checkpoint_restore_v1(
            ctypes.byref(checkpoint),
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            destination,
            owners.tokens[destination],
        )
        == 0
    )
    assert bytes(rows[destination]) == bytes(checkpoint)
    assert owners.ids[destination] == (1 << 40) + 117
    assert owners.tokens[destination] == 71_117
    assert owners.alive[destination] == 1
    route_bound_payload = {
        "receipt_version": int(receipt.receipt_version),
        "lifecycle_status": int(receipt.lifecycle_status),
        "committed": int(receipt.committed),
        "destination_row": int(receipt.destination_row),
        "selected_parent_row": int(receipt.selected_parent_row),
        "child_agent_id": int(receipt.child_agent_id),
        "publication_token": int(receipt.publication_token),
        "donor_draw_float32_bits": f"{int(receipt.donor_draw_float32_bits):08x}",
        "mutation_count": int(receipt.mutation_count),
        "child_checkpoint_sha256": hashlib.sha256(bytes(checkpoint)).hexdigest(),
        "restored_child_checkpoint_sha256": hashlib.sha256(
            bytes(rows[destination])
        ).hexdigest(),
    }
    assert native_cpu_progress_sha256_v1(route_bound_payload)

    before_stale = (
        bytes(rows),
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    )
    stale_receipt = BirthReceiptV1()
    stale_status = oracle.library.eiel_modular_recurrent_newborn_inherit_with_receipt_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.mutable),
        destination,
        owners.tokens[destination],
        parent_a,
        owners.tokens[parent_a] + 1,
        parent_b,
        owners.tokens[parent_b],
        donor_draw,
        (1 << 40) + 118,
        71_118,
        ctypes.byref(mutation),
        1,
        ctypes.byref(stale_receipt),
    )
    assert stale_status == 5
    assert stale_receipt.lifecycle_status == stale_status
    assert stale_receipt.committed == 0
    assert stale_receipt.owner_checked == 1
    assert stale_receipt.parent_preserved == 0
    assert stale_receipt.child_runtime_reset == 0
    assert (
        bytes(rows),
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    ) == before_stale


def test_p12_native_cpu_newborn_inherit_binds_canonical_recurrent_receipt(
    oracle,
) -> None:
    _configure_birth_receipt_symbols(oracle.library)
    packed = _packed_fixture(extra_empty_rows=1)
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    destination = AGENT_COUNT
    parent_a = 0
    parent_b = 1
    child_agent_id = (1 << 40) + 317
    publication_token = 73_317
    mutation = MutationV1(0, 0, 2, ctypes.c_float(0.125).value)
    (
        binding,
        canonical_sha256,
        canonical_bytes_sha256,
        structural_payload_sha256,
    ) = _canonical_binding(
        parent_agent_id=int(packed.tensors["agent_id"][parent_a]),
        parent_slot=parent_a,
        parent_slot_token=owners.tokens[parent_a],
        child_agent_id=child_agent_id,
        child_slot=destination,
        child_slot_token=publication_token,
    )
    receipt = BirthReceiptV1()

    status = oracle.library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.mutable),
        destination,
        owners.tokens[destination],
        parent_a,
        owners.tokens[parent_a],
        parent_b,
        owners.tokens[parent_b],
        ctypes.c_float(0.25),
        child_agent_id,
        publication_token,
        ctypes.byref(mutation),
        1,
        None,
        ctypes.byref(binding),
        ctypes.byref(receipt),
    )

    assert status == 0
    assert receipt.receipt_version == 3
    assert receipt.committed == 1
    assert receipt.canonical_receipt_bound == 1
    assert receipt.canonical_receipt_identity_checked == 1
    assert receipt.recurrent_receipt_bytes_bound == 1
    assert receipt.draw0_float64_bits == _float64_bits(0.625)
    assert receipt.draw1_float64_bits == _float64_bits(0.5)
    assert list(receipt.canonical_recurrent_receipt_sha256_words) == list(
        _sha256_words(canonical_sha256)
    )
    assert list(receipt.canonical_recurrent_receipt_bytes_sha256_words) == list(
        _sha256_words(canonical_bytes_sha256)
    )
    assert receipt.structural_payload_sha256_bound == 1
    assert receipt.structural_payload_identity_checked == 1
    assert list(receipt.structural_payload_sha256_words) == list(
        _sha256_words(structural_payload_sha256)
    )
    assert receipt.transaction_ordinal == 0
    assert receipt.selected_instance_id_index == 0
    assert receipt.selected_homology_id_index == 1
    assert receipt.operation_code == 2
    assert receipt.status_code == 1
    assert receipt.reason_code == 1
    assert receipt.changed == 1
    assert rows[destination].agent_id == child_agent_id
    assert rows[destination].agent_slot_token == publication_token

    stale_rows = _rows_from_planes(packed.tensors)
    stale_owners = _OwnerBundle(packed.tensors)
    stale_before = (
        bytes(stale_rows),
        list(stale_owners.ids),
        list(stale_owners.tokens),
        list(stale_owners.alive),
    )
    bad_binding = copy.deepcopy(binding)
    bad_binding.parent_slot_token += 1
    stale_receipt = BirthReceiptV1()
    stale_status = (
        oracle.library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1(
            stale_rows,
            len(stale_rows),
            PROFILE,
            ctypes.byref(stale_owners.mutable),
            destination,
            stale_owners.tokens[destination],
            parent_a,
            stale_owners.tokens[parent_a],
            parent_b,
            stale_owners.tokens[parent_b],
            ctypes.c_float(0.25),
            child_agent_id,
            publication_token,
            ctypes.byref(mutation),
            1,
            None,
            ctypes.byref(bad_binding),
            ctypes.byref(stale_receipt),
        )
    )
    assert stale_status == 5
    assert stale_receipt.receipt_version == 3
    assert stale_receipt.committed == 0
    assert stale_receipt.canonical_receipt_bound == 0
    assert stale_receipt.structural_payload_sha256_bound == 0
    assert (
        bytes(stale_rows),
        list(stale_owners.ids),
        list(stale_owners.tokens),
        list(stale_owners.alive),
    ) == stale_before


def test_p12_native_cpu_newborn_inherit_v3_binds_native_structural_sha_and_rolls_back(
    oracle,
) -> None:
    _configure_birth_receipt_symbols(oracle.library)
    packed = _packed_fixture(extra_empty_rows=1)
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    destination = AGENT_COUNT
    parent_a = 0
    parent_b = 1
    child_agent_id = (1 << 40) + 517
    publication_token = 75_517
    mutation = MutationV1(0, 0, 2, ctypes.c_float(0.125).value)
    expected_structural_sha256 = _expected_newborn_structural_sha256(
        oracle.library,
        rows,
        parent_row=parent_a,
        child_agent_id=child_agent_id,
        publication_token=publication_token,
        mutation=mutation,
    )
    binding, canonical_sha256, canonical_bytes_sha256, _structural_sha256 = (
        _canonical_binding(
            parent_agent_id=int(packed.tensors["agent_id"][parent_a]),
            parent_slot=parent_a,
            parent_slot_token=owners.tokens[parent_a],
            child_agent_id=child_agent_id,
            child_slot=destination,
            child_slot_token=publication_token,
            binding_version=3,
            structural_payload_sha256=expected_structural_sha256,
        )
    )
    receipt = BirthReceiptV1()

    status = oracle.library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.mutable),
        destination,
        owners.tokens[destination],
        parent_a,
        owners.tokens[parent_a],
        parent_b,
        owners.tokens[parent_b],
        ctypes.c_float(0.25),
        child_agent_id,
        publication_token,
        ctypes.byref(mutation),
        1,
        None,
        ctypes.byref(binding),
        ctypes.byref(receipt),
    )

    assert status == 0
    assert receipt.receipt_version == 4
    assert receipt.committed == 1
    assert receipt.canonical_receipt_bound == 1
    assert receipt.recurrent_receipt_bytes_bound == 1
    assert list(receipt.canonical_recurrent_receipt_sha256_words) == list(
        _sha256_words(canonical_sha256)
    )
    assert list(receipt.canonical_recurrent_receipt_bytes_sha256_words) == list(
        _sha256_words(canonical_bytes_sha256)
    )
    assert receipt.structural_payload_sha256_bound == 1
    assert receipt.structural_payload_identity_checked == 1
    assert list(receipt.structural_payload_sha256_words) == list(
        _sha256_words(expected_structural_sha256)
    )
    assert _native_structural_payload_sha256(
        oracle.library,
        rows[destination],
    ) == expected_structural_sha256

    stale_rows = _rows_from_planes(packed.tensors)
    stale_owners = _OwnerBundle(packed.tensors)
    stale_before = (
        bytes(stale_rows),
        list(stale_owners.ids),
        list(stale_owners.tokens),
        list(stale_owners.alive),
    )
    bad_binding = copy.deepcopy(binding)
    bad_binding.structural_payload_sha256_words = _sha256_words("44" * 32)
    stale_receipt = BirthReceiptV1()
    stale_status = (
        oracle.library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1(
            stale_rows,
            len(stale_rows),
            PROFILE,
            ctypes.byref(stale_owners.mutable),
            destination,
            stale_owners.tokens[destination],
            parent_a,
            stale_owners.tokens[parent_a],
            parent_b,
            stale_owners.tokens[parent_b],
            ctypes.c_float(0.25),
            child_agent_id,
            publication_token,
            ctypes.byref(mutation),
            1,
            None,
            ctypes.byref(bad_binding),
            ctypes.byref(stale_receipt),
        )
    )
    assert stale_status == 5
    assert stale_receipt.receipt_version == 4
    assert stale_receipt.lifecycle_status == stale_status
    assert stale_receipt.committed == 0
    assert stale_receipt.canonical_receipt_bound == 0
    assert stale_receipt.recurrent_receipt_bytes_bound == 0
    assert stale_receipt.structural_payload_sha256_bound == 0
    assert (
        bytes(stale_rows),
        list(stale_owners.ids),
        list(stale_owners.tokens),
        list(stale_owners.alive),
    ) == stale_before


def test_p12_native_cpu_newborn_inherit_v4_binds_native_canonical_receipt_and_rolls_back(
    oracle,
) -> None:
    _configure_birth_receipt_symbols(oracle.library)
    packed = _packed_fixture(extra_empty_rows=1)
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    destination = AGENT_COUNT
    parent_a = 0
    parent_b = 1
    child_agent_id = (1 << 40) + 717
    publication_token = 77_717
    mutation = MutationV1(0, 0, 2, ctypes.c_float(0.125).value)
    expected_structural_sha256 = _expected_newborn_structural_sha256(
        oracle.library,
        rows,
        parent_row=parent_a,
        child_agent_id=child_agent_id,
        publication_token=publication_token,
        mutation=mutation,
    )
    binding, canonical_sha256, canonical_bytes_sha256, _structural_sha256 = (
        _canonical_binding(
            parent_agent_id=int(packed.tensors["agent_id"][parent_a]),
            parent_slot=parent_a,
            parent_slot_token=owners.tokens[parent_a],
            child_agent_id=child_agent_id,
            child_slot=destination,
            child_slot_token=publication_token,
            binding_version=4,
            structural_payload_sha256=expected_structural_sha256,
        )
    )
    _assert_native_canonical_receipt_sha256_matches(
        oracle.library,
        binding,
        canonical_sha256,
        canonical_bytes_sha256,
    )
    invalid_direct_binding = copy.deepcopy(binding)
    invalid_direct_binding.native_receipt_construction_required = 0
    _assert_native_canonical_receipt_sha256_rejects(
        oracle.library,
        invalid_direct_binding,
    )
    source = _canonical_source()
    receipt = BirthReceiptV1()

    status = oracle.library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.mutable),
        destination,
        owners.tokens[destination],
        parent_a,
        owners.tokens[parent_a],
        parent_b,
        owners.tokens[parent_b],
        ctypes.c_float(0.25),
        child_agent_id,
        publication_token,
        ctypes.byref(mutation),
        1,
        ctypes.byref(source),
        ctypes.byref(binding),
        ctypes.byref(receipt),
    )

    assert status == 0
    assert receipt.receipt_version == 4
    assert receipt.committed == 1
    assert receipt.canonical_receipt_bound == 1
    assert receipt.canonical_receipt_identity_checked == 1
    assert receipt.recurrent_receipt_bytes_bound == 1
    assert list(receipt.canonical_recurrent_receipt_sha256_words) == list(
        _sha256_words(canonical_sha256)
    )
    assert list(receipt.canonical_recurrent_receipt_bytes_sha256_words) == list(
        _sha256_words(canonical_bytes_sha256)
    )
    assert receipt.structural_payload_sha256_bound == 1
    assert receipt.structural_payload_identity_checked == 1
    assert _native_structural_payload_sha256(oracle.library, rows[destination]) == (
        expected_structural_sha256
    )
    checkpoint_sha256 = _assert_project_death_restore_round_trip(
        oracle.library,
        rows,
        owners,
        row_index=destination,
        expected_structural_sha256=expected_structural_sha256,
    )
    v4_route_checkpoint_payload = {
        "case_id": "newborn_v4_native_canonical_receipt_checkpoint_restore",
        "receipt_version": int(receipt.receipt_version),
        "canonical_receipt_sha256": canonical_sha256,
        "canonical_receipt_bytes_sha256": canonical_bytes_sha256,
        "structural_payload_sha256": expected_structural_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "child_agent_id": child_agent_id,
        "child_slot_token": publication_token,
    }
    assert native_cpu_progress_sha256_v1(v4_route_checkpoint_payload)

    stale_rows = _rows_from_planes(packed.tensors)
    stale_owners = _OwnerBundle(packed.tensors)
    stale_before = (
        bytes(stale_rows),
        list(stale_owners.ids),
        list(stale_owners.tokens),
        list(stale_owners.alive),
    )
    bad_source = _canonical_source()
    bad_source.source_packed_sha256_words = _sha256_words("44" * 32)
    stale_receipt = BirthReceiptV1()
    stale_status = (
        oracle.library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1(
            stale_rows,
            len(stale_rows),
            PROFILE,
            ctypes.byref(stale_owners.mutable),
            destination,
            stale_owners.tokens[destination],
            parent_a,
            stale_owners.tokens[parent_a],
            parent_b,
            stale_owners.tokens[parent_b],
            ctypes.c_float(0.25),
            child_agent_id,
            publication_token,
            ctypes.byref(mutation),
            1,
            ctypes.byref(bad_source),
            ctypes.byref(binding),
            ctypes.byref(stale_receipt),
        )
    )
    assert stale_status == 5
    assert stale_receipt.receipt_version == 4
    assert stale_receipt.committed == 0
    assert stale_receipt.canonical_receipt_bound == 0
    assert stale_receipt.recurrent_receipt_bytes_bound == 0
    assert stale_receipt.structural_payload_sha256_bound == 0
    assert (
        bytes(stale_rows),
        list(stale_owners.ids),
        list(stale_owners.tokens),
        list(stale_owners.alive),
    ) == stale_before


def test_p12_native_cpu_same_slot_reuse_emits_route_bound_birth_receipt(
    oracle,
) -> None:
    _configure_birth_receipt_symbols(oracle.library)
    packed = _packed_fixture()
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    target = 2
    structural_parent = 0
    mutation = MutationV1(0, 0, 2, ctypes.c_float(0.0625).value)
    target_before = bytes(rows[target])
    parent_before = bytes(rows[structural_parent])
    receipt = BirthReceiptV1()

    status = oracle.library.eiel_modular_recurrent_same_slot_reuse_with_receipt_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.mutable),
        target,
        owners.ids[target],
        owners.tokens[target],
        structural_parent,
        owners.tokens[structural_parent],
        (1 << 40) + 222,
        72_222,
        ctypes.byref(mutation),
        1,
        ctypes.byref(receipt),
    )

    assert status == 0
    assert receipt.receipt_version == 1
    assert receipt.lifecycle_status == status
    assert receipt.committed == 1
    assert receipt.owner_checked == 1
    assert receipt.parent_preserved == 1
    assert receipt.child_runtime_reset == 1
    assert receipt.destination_row == target
    assert receipt.parent_a_row == structural_parent
    assert receipt.parent_b_row == -1
    assert receipt.selected_parent_row == structural_parent
    assert receipt.expected_destination_token == 10_000 + target
    assert receipt.expected_parent_a_token == 10_000 + structural_parent
    assert receipt.expected_parent_b_token == -1
    assert receipt.parent_a_agent_id == int(
        packed.tensors["agent_id"][structural_parent]
    )
    assert receipt.parent_b_agent_id == -1
    assert receipt.child_agent_id == (1 << 40) + 222
    assert receipt.publication_token == 72_222
    assert receipt.donor_draw_float32_bits == 0
    assert receipt.mutation_count == 1
    assert bytes(rows[target]) != target_before
    assert bytes(rows[structural_parent]) == parent_before
    assert owners.ids[target] == (1 << 40) + 222
    assert owners.tokens[target] == 72_222
    assert rows[target].agent_id == (1 << 40) + 222
    assert rows[target].agent_slot_token == 72_222
    assert list(rows[target].module_state_present) == [0] * 16
    assert list(rows[target].module_state) == [0.0] * 16

    checkpoint = RowV1()
    assert (
        oracle.library.eiel_modular_recurrent_checkpoint_project_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.const),
            target,
            72_222,
            ctypes.byref(checkpoint),
        )
        == 0
    )
    route_bound_payload = {
        "receipt_version": int(receipt.receipt_version),
        "lifecycle_status": int(receipt.lifecycle_status),
        "committed": int(receipt.committed),
        "destination_row": int(receipt.destination_row),
        "selected_parent_row": int(receipt.selected_parent_row),
        "child_agent_id": int(receipt.child_agent_id),
        "publication_token": int(receipt.publication_token),
        "donor_draw_float32_bits": f"{int(receipt.donor_draw_float32_bits):08x}",
        "mutation_count": int(receipt.mutation_count),
        "same_slot_child_checkpoint_sha256": hashlib.sha256(
            bytes(checkpoint)
        ).hexdigest(),
    }
    assert native_cpu_progress_sha256_v1(route_bound_payload)

    stale_before = (
        bytes(rows),
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    )
    stale_receipt = BirthReceiptV1()
    stale_status = oracle.library.eiel_modular_recurrent_same_slot_reuse_with_receipt_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.mutable),
        target,
        owners.ids[target],
        owners.tokens[target] + 1,
        structural_parent,
        owners.tokens[structural_parent],
        (1 << 40) + 223,
        72_223,
        ctypes.byref(mutation),
        1,
        ctypes.byref(stale_receipt),
    )
    assert stale_status == 5
    assert stale_receipt.lifecycle_status == stale_status
    assert stale_receipt.committed == 0
    assert stale_receipt.owner_checked == 1
    assert stale_receipt.parent_preserved == 0
    assert stale_receipt.child_runtime_reset == 0
    assert (
        bytes(rows),
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    ) == stale_before


def test_p12_native_cpu_same_slot_reuse_binds_canonical_recurrent_receipt(
    oracle,
) -> None:
    _configure_birth_receipt_symbols(oracle.library)
    packed = _packed_fixture()
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    target = 2
    structural_parent = 0
    child_agent_id = (1 << 40) + 418
    publication_token = 74_418
    mutation = MutationV1(0, 0, 2, ctypes.c_float(0.0625).value)
    (
        binding,
        canonical_sha256,
        canonical_bytes_sha256,
        structural_payload_sha256,
    ) = _canonical_binding(
        parent_agent_id=int(packed.tensors["agent_id"][structural_parent]),
        parent_slot=structural_parent,
        parent_slot_token=owners.tokens[structural_parent],
        child_agent_id=child_agent_id,
        child_slot=target,
        child_slot_token=publication_token,
    )
    receipt = BirthReceiptV1()

    status = (
        oracle.library.eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            target,
            owners.ids[target],
            owners.tokens[target],
            structural_parent,
            owners.tokens[structural_parent],
            child_agent_id,
            publication_token,
            ctypes.byref(mutation),
            1,
            None,
            ctypes.byref(binding),
            ctypes.byref(receipt),
        )
    )

    assert status == 0
    assert receipt.receipt_version == 3
    assert receipt.committed == 1
    assert receipt.canonical_receipt_bound == 1
    assert receipt.canonical_receipt_identity_checked == 1
    assert receipt.recurrent_receipt_bytes_bound == 1
    assert receipt.draw0_float64_bits == _float64_bits(0.625)
    assert receipt.draw1_float64_bits == _float64_bits(0.5)
    assert list(receipt.canonical_recurrent_receipt_sha256_words) == list(
        _sha256_words(canonical_sha256)
    )
    assert list(receipt.canonical_recurrent_receipt_bytes_sha256_words) == list(
        _sha256_words(canonical_bytes_sha256)
    )
    assert receipt.structural_payload_sha256_bound == 1
    assert receipt.structural_payload_identity_checked == 1
    assert list(receipt.structural_payload_sha256_words) == list(
        _sha256_words(structural_payload_sha256)
    )
    assert owners.ids[target] == child_agent_id
    assert owners.tokens[target] == publication_token
    assert rows[target].agent_id == child_agent_id
    assert rows[target].agent_slot_token == publication_token


def test_p12_native_cpu_same_slot_reuse_v3_binds_native_structural_sha_and_rolls_back(
    oracle,
) -> None:
    _configure_birth_receipt_symbols(oracle.library)
    packed = _packed_fixture()
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    target = 2
    structural_parent = 0
    child_agent_id = (1 << 40) + 618
    publication_token = 76_618
    mutation = MutationV1(0, 0, 2, ctypes.c_float(0.0625).value)
    expected_structural_sha256 = _expected_newborn_structural_sha256(
        oracle.library,
        rows,
        parent_row=structural_parent,
        child_agent_id=child_agent_id,
        publication_token=publication_token,
        mutation=mutation,
    )
    binding, _canonical_sha256, _canonical_bytes_sha256, _structural_sha256 = (
        _canonical_binding(
            parent_agent_id=int(packed.tensors["agent_id"][structural_parent]),
            parent_slot=structural_parent,
            parent_slot_token=owners.tokens[structural_parent],
            child_agent_id=child_agent_id,
            child_slot=target,
            child_slot_token=publication_token,
            binding_version=3,
            structural_payload_sha256=expected_structural_sha256,
        )
    )
    receipt = BirthReceiptV1()

    status = (
        oracle.library.eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            target,
            owners.ids[target],
            owners.tokens[target],
            structural_parent,
            owners.tokens[structural_parent],
            child_agent_id,
            publication_token,
            ctypes.byref(mutation),
            1,
            None,
            ctypes.byref(binding),
            ctypes.byref(receipt),
        )
    )

    assert status == 0
    assert receipt.receipt_version == 4
    assert receipt.committed == 1
    assert receipt.structural_payload_sha256_bound == 1
    assert receipt.structural_payload_identity_checked == 1
    assert list(receipt.structural_payload_sha256_words) == list(
        _sha256_words(expected_structural_sha256)
    )
    assert _native_structural_payload_sha256(oracle.library, rows[target]) == (
        expected_structural_sha256
    )

    stale_rows = _rows_from_planes(packed.tensors)
    stale_owners = _OwnerBundle(packed.tensors)
    stale_before = (
        bytes(stale_rows),
        list(stale_owners.ids),
        list(stale_owners.tokens),
        list(stale_owners.alive),
    )
    bad_binding = copy.deepcopy(binding)
    bad_binding.structural_payload_sha256_words = _sha256_words("55" * 32)
    stale_receipt = BirthReceiptV1()
    stale_status = (
        oracle.library.eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1(
            stale_rows,
            len(stale_rows),
            PROFILE,
            ctypes.byref(stale_owners.mutable),
            target,
            stale_owners.ids[target],
            stale_owners.tokens[target],
            structural_parent,
            stale_owners.tokens[structural_parent],
            child_agent_id,
            publication_token,
            ctypes.byref(mutation),
            1,
            None,
            ctypes.byref(bad_binding),
            ctypes.byref(stale_receipt),
        )
    )
    assert stale_status == 5
    assert stale_receipt.receipt_version == 4
    assert stale_receipt.lifecycle_status == stale_status
    assert stale_receipt.committed == 0
    assert stale_receipt.structural_payload_sha256_bound == 0
    assert (
        bytes(stale_rows),
        list(stale_owners.ids),
        list(stale_owners.tokens),
        list(stale_owners.alive),
    ) == stale_before


def test_p12_native_cpu_same_slot_reuse_v4_binds_native_canonical_receipt_and_rolls_back(
    oracle,
) -> None:
    _configure_birth_receipt_symbols(oracle.library)
    packed = _packed_fixture()
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    target = 2
    structural_parent = 0
    child_agent_id = (1 << 40) + 818
    publication_token = 78_818
    mutation = MutationV1(0, 0, 2, ctypes.c_float(0.0625).value)
    expected_structural_sha256 = _expected_newborn_structural_sha256(
        oracle.library,
        rows,
        parent_row=structural_parent,
        child_agent_id=child_agent_id,
        publication_token=publication_token,
        mutation=mutation,
    )
    binding, canonical_sha256, canonical_bytes_sha256, _structural_sha256 = (
        _canonical_binding(
            parent_agent_id=int(packed.tensors["agent_id"][structural_parent]),
            parent_slot=structural_parent,
            parent_slot_token=owners.tokens[structural_parent],
            child_agent_id=child_agent_id,
            child_slot=target,
            child_slot_token=publication_token,
            binding_version=4,
            structural_payload_sha256=expected_structural_sha256,
        )
    )
    _assert_native_canonical_receipt_sha256_matches(
        oracle.library,
        binding,
        canonical_sha256,
        canonical_bytes_sha256,
    )
    source = _canonical_source()
    receipt = BirthReceiptV1()

    status = (
        oracle.library.eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            target,
            owners.ids[target],
            owners.tokens[target],
            structural_parent,
            owners.tokens[structural_parent],
            child_agent_id,
            publication_token,
            ctypes.byref(mutation),
            1,
            ctypes.byref(source),
            ctypes.byref(binding),
            ctypes.byref(receipt),
        )
    )

    assert status == 0
    assert receipt.receipt_version == 4
    assert receipt.committed == 1
    assert receipt.canonical_receipt_bound == 1
    assert receipt.canonical_receipt_identity_checked == 1
    assert receipt.recurrent_receipt_bytes_bound == 1
    assert list(receipt.canonical_recurrent_receipt_sha256_words) == list(
        _sha256_words(canonical_sha256)
    )
    assert list(receipt.canonical_recurrent_receipt_bytes_sha256_words) == list(
        _sha256_words(canonical_bytes_sha256)
    )
    assert receipt.structural_payload_sha256_bound == 1
    assert receipt.structural_payload_identity_checked == 1
    assert _native_structural_payload_sha256(oracle.library, rows[target]) == (
        expected_structural_sha256
    )
    checkpoint_sha256 = _assert_project_death_restore_round_trip(
        oracle.library,
        rows,
        owners,
        row_index=target,
        expected_structural_sha256=expected_structural_sha256,
    )
    v4_route_checkpoint_payload = {
        "case_id": "same_slot_v4_native_canonical_receipt_checkpoint_restore",
        "receipt_version": int(receipt.receipt_version),
        "canonical_receipt_sha256": canonical_sha256,
        "canonical_receipt_bytes_sha256": canonical_bytes_sha256,
        "structural_payload_sha256": expected_structural_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "child_agent_id": child_agent_id,
        "child_slot_token": publication_token,
    }
    assert native_cpu_progress_sha256_v1(v4_route_checkpoint_payload)

    stale_rows = _rows_from_planes(packed.tensors)
    stale_owners = _OwnerBundle(packed.tensors)
    stale_before = (
        bytes(stale_rows),
        list(stale_owners.ids),
        list(stale_owners.tokens),
        list(stale_owners.alive),
    )
    bad_binding = copy.deepcopy(binding)
    bad_binding.canonical_receipt_sha256_words = _sha256_words("55" * 32)
    stale_receipt = BirthReceiptV1()
    stale_status = (
        oracle.library.eiel_modular_recurrent_same_slot_reuse_with_canonical_receipt_v1(
            stale_rows,
            len(stale_rows),
            PROFILE,
            ctypes.byref(stale_owners.mutable),
            target,
            stale_owners.ids[target],
            stale_owners.tokens[target],
            structural_parent,
            stale_owners.tokens[structural_parent],
            child_agent_id,
            publication_token,
            ctypes.byref(mutation),
            1,
            ctypes.byref(source),
            ctypes.byref(bad_binding),
            ctypes.byref(stale_receipt),
        )
    )
    assert stale_status == 5
    assert stale_receipt.receipt_version == 4
    assert stale_receipt.committed == 0
    assert stale_receipt.canonical_receipt_bound == 0
    assert stale_receipt.recurrent_receipt_bytes_bound == 0
    assert stale_receipt.structural_payload_sha256_bound == 0
    assert (
        bytes(stale_rows),
        list(stale_owners.ids),
        list(stale_owners.tokens),
        list(stale_owners.alive),
    ) == stale_before


def test_p12_native_cpu_multicell_v4_checkpoint_collection_restores_canonical_receipts(
    oracle,
) -> None:
    _configure_birth_receipt_symbols(oracle.library)
    packed = _packed_fixture(extra_empty_rows=2)
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    source = _canonical_source()
    cases = (
        {
            "cell_id": "cell0",
            "destination": AGENT_COUNT,
            "parent_a": 0,
            "parent_b": 1,
            "child_agent_id": (1 << 40) + 901,
            "publication_token": 79_901,
            "collection_ordinal": 0,
        },
        {
            "cell_id": "cell1",
            "destination": AGENT_COUNT + 1,
            "parent_a": 2,
            "parent_b": 3,
            "child_agent_id": (1 << 40) + 902,
            "publication_token": 79_902,
            "collection_ordinal": 1,
        },
    )
    mutation = MutationV1(0, 0, 2, ctypes.c_float(0.125).value)
    collection_rows: list[dict[str, object]] = []
    checkpoints: list[RowV1] = []

    for case in cases:
        destination = int(case["destination"])
        parent_a = int(case["parent_a"])
        parent_b = int(case["parent_b"])
        child_agent_id = int(case["child_agent_id"])
        publication_token = int(case["publication_token"])
        collection_ordinal = int(case["collection_ordinal"])
        expected_structural_sha256 = _expected_newborn_structural_sha256(
            oracle.library,
            rows,
            parent_row=parent_a,
            child_agent_id=child_agent_id,
            publication_token=publication_token,
            mutation=mutation,
        )
        binding, canonical_sha256, canonical_bytes_sha256, _structural_sha256 = (
            _canonical_binding(
                parent_agent_id=int(packed.tensors["agent_id"][parent_a]),
                parent_slot=parent_a,
                parent_slot_token=owners.tokens[parent_a],
                child_agent_id=child_agent_id,
                child_slot=destination,
                child_slot_token=publication_token,
                binding_version=4,
                structural_payload_sha256=expected_structural_sha256,
            )
        )
        _assert_native_canonical_receipt_sha256_matches(
            oracle.library,
            binding,
            canonical_sha256,
            canonical_bytes_sha256,
        )
        receipt = BirthReceiptV1()

        status = oracle.library.eiel_modular_recurrent_newborn_inherit_with_canonical_receipt_v1(
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            destination,
            owners.tokens[destination],
            parent_a,
            owners.tokens[parent_a],
            parent_b,
            owners.tokens[parent_b],
            ctypes.c_float(0.25),
            child_agent_id,
            publication_token,
            ctypes.byref(mutation),
            1,
            ctypes.byref(source),
            ctypes.byref(binding),
            ctypes.byref(receipt),
        )

        assert status == 0
        assert receipt.receipt_version == 4
        assert receipt.committed == 1
        assert receipt.transaction_ordinal == 0
        assert receipt.destination_row == destination
        assert receipt.child_agent_id == child_agent_id
        assert receipt.publication_token == publication_token
        assert receipt.canonical_receipt_bound == 1
        assert receipt.recurrent_receipt_bytes_bound == 1
        assert receipt.structural_payload_sha256_bound == 1
        assert list(receipt.canonical_recurrent_receipt_sha256_words) == list(
            _sha256_words(canonical_sha256)
        )
        assert list(receipt.canonical_recurrent_receipt_bytes_sha256_words) == list(
            _sha256_words(canonical_bytes_sha256)
        )
        assert _native_structural_payload_sha256(
            oracle.library, rows[destination]
        ) == expected_structural_sha256

        checkpoint = RowV1()
        assert (
            oracle.library.eiel_modular_recurrent_checkpoint_project_v1(
                rows,
                len(rows),
                PROFILE,
                ctypes.byref(owners.const),
                destination,
                publication_token,
                ctypes.byref(checkpoint),
            )
            == 0
        )
        checkpoint_sha256 = hashlib.sha256(bytes(checkpoint)).hexdigest()
        checkpoints.append(copy.deepcopy(checkpoint))
        collection_rows.append(
            {
                "cell_id": case["cell_id"],
                "collection_ordinal": collection_ordinal,
                "source_receipt_transaction_ordinal": int(receipt.transaction_ordinal),
                "destination_row": destination,
                "child_agent_id": child_agent_id,
                "publication_token": publication_token,
                "canonical_receipt_sha256": canonical_sha256,
                "canonical_receipt_bytes_sha256": canonical_bytes_sha256,
                "structural_payload_sha256": expected_structural_sha256,
                "checkpoint_sha256": checkpoint_sha256,
            }
        )

    bank_after_births = bytes(rows)
    owners_after_births = (
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    )

    for case in cases:
        destination = int(case["destination"])
        assert (
            oracle.library.eiel_modular_recurrent_death_cleanup_v1(
                rows,
                len(rows),
                PROFILE,
                ctypes.byref(owners.mutable),
                destination,
                int(case["child_agent_id"]),
                int(case["publication_token"]),
            )
            == 0
        )
        assert owners.ids[destination] == -1
        assert owners.alive[destination] == 0

    for case, checkpoint in zip(cases, checkpoints, strict=True):
        destination = int(case["destination"])
        assert (
            oracle.library.eiel_modular_recurrent_checkpoint_restore_v1(
                ctypes.byref(checkpoint),
                rows,
                len(rows),
                PROFILE,
                ctypes.byref(owners.mutable),
                destination,
                owners.tokens[destination],
            )
            == 0
        )
        assert bytes(rows[destination]) == bytes(checkpoint)
        assert owners.ids[destination] == int(case["child_agent_id"])
        assert owners.tokens[destination] == int(case["publication_token"])
        assert owners.alive[destination] == 1

    assert bytes(rows) == bank_after_births
    assert (
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    ) == owners_after_births
    collection_payload = {
        "case_id": "multicell_v4_native_canonical_receipt_checkpoint_collection",
        "receipt_version": 4,
        "cell_count": len(cases),
        "artifact_path": oracle.artifact_path.name,
        "artifact_sha256": oracle.artifact_sha256,
        "build_manifest_path": str(DEFAULT_BUILD_MANIFEST.relative_to(ROOT)),
        "build_manifest_sha256": _sha256(DEFAULT_BUILD_MANIFEST),
        "requires_two_build_artifact_byte_identity": True,
        "collection_ordinals": [
            int(row["collection_ordinal"]) for row in collection_rows
        ],
        "source_receipt_transaction_ordinals": [
            int(row["source_receipt_transaction_ordinal"]) for row in collection_rows
        ],
        "checkpoint_rows": collection_rows,
        "bank_after_births_sha256": hashlib.sha256(bank_after_births).hexdigest(),
        "bank_after_restore_sha256": hashlib.sha256(bytes(rows)).hexdigest(),
    }
    assert collection_payload["artifact_sha256"] == NATIVE_CPU_ROUTE_BOUND_ARTIFACT_SHA256
    assert (
        collection_payload["build_manifest_sha256"]
        == NATIVE_CPU_ROUTE_BOUND_BUILD_MANIFEST_SHA256
    )
    assert native_cpu_progress_sha256_v1(collection_payload)


def test_p12_native_cpu_generation_turnover_checkpoint_resume_repeats_v4_birth_receipts(
    oracle,
) -> None:
    _configure_birth_receipt_symbols(oracle.library)
    continuous = _route_bound_generation_turnover_sequence(
        oracle.library,
        restore_prefix_checkpoint=False,
    )
    resumed = _route_bound_generation_turnover_sequence(
        oracle.library,
        restore_prefix_checkpoint=True,
    )

    assert resumed["prefix_checkpoint_sha256"] == continuous["prefix_checkpoint_sha256"]
    assert resumed["prefix_receipt_sha256"] == continuous["prefix_receipt_sha256"]
    assert resumed["pre_suffix_bank_sha256"] == continuous["pre_suffix_bank_sha256"]
    assert resumed["pre_suffix_owner_state"] == continuous["pre_suffix_owner_state"]
    assert resumed["suffix_receipt_sha256"] == continuous["suffix_receipt_sha256"]
    assert resumed["final_bank_sha256"] == continuous["final_bank_sha256"]
    assert resumed["final_owner_state"] == continuous["final_owner_state"]

    turnover_payload = {
        "case_id": "generation_turnover_v4_native_checkpoint_resume_receipt_repeat",
        "artifact_path": oracle.artifact_path.name,
        "artifact_sha256": oracle.artifact_sha256,
        "build_manifest_path": str(DEFAULT_BUILD_MANIFEST.relative_to(ROOT)),
        "build_manifest_sha256": _sha256(DEFAULT_BUILD_MANIFEST),
        "prefix_generation": 0,
        "resume_generation": 1,
        "prefix_checkpoint_sha256": resumed["prefix_checkpoint_sha256"],
        "prefix_receipt_sha256": resumed["prefix_receipt_sha256"],
        "pre_suffix_bank_sha256": resumed["pre_suffix_bank_sha256"],
        "suffix_receipt_transaction_ordinal": 0,
        "suffix_receipt_sha256": resumed["suffix_receipt_sha256"],
        "suffix_canonical_receipt_sha256": resumed["suffix_canonical_receipt_sha256"],
        "suffix_canonical_receipt_bytes_sha256": resumed[
            "suffix_canonical_receipt_bytes_sha256"
        ],
        "suffix_structural_payload_sha256": resumed[
            "suffix_structural_payload_sha256"
        ],
        "final_bank_sha256": resumed["final_bank_sha256"],
    }
    assert turnover_payload["artifact_sha256"] == NATIVE_CPU_ROUTE_BOUND_ARTIFACT_SHA256
    assert (
        turnover_payload["build_manifest_sha256"]
        == NATIVE_CPU_ROUTE_BOUND_BUILD_MANIFEST_SHA256
    )
    assert native_cpu_progress_sha256_v1(turnover_payload)


def test_checkpoint_continuation_cost_capacity_and_disabled_neutrality(oracle) -> None:
    packed = _packed_fixture()
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    observations, _base, _policy, _masks = _fixed_inputs()
    status, _actions, _receipt = _call_tick(
        oracle.library,
        rows,
        owners,
        observations[0],
        observations[0, :, LOCAL_SIZE],
        np.zeros((AGENT_COUNT,), dtype=np.int64),
    )
    assert status == 0
    checkpoint = RowV1()
    status = oracle.library.eiel_modular_recurrent_checkpoint_project_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.const),
        0,
        owners.tokens[0],
        ctypes.byref(checkpoint),
    )
    assert status == 0

    occupied_before = bytes(rows)
    owner_before = (
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    )
    assert (
        oracle.library.eiel_modular_recurrent_checkpoint_restore_v1(
            ctypes.byref(checkpoint),
            rows,
            len(rows),
            PROFILE,
            ctypes.byref(owners.mutable),
            1,
            owners.tokens[1],
        )
        == 5
    )
    assert bytes(rows) == occupied_before
    assert (
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    ) == owner_before

    status, continued_actions, _receipt = _call_tick(
        oracle.library,
        rows,
        owners,
        observations[1],
        observations[1, :, LOCAL_SIZE],
        np.ones((AGENT_COUNT,), dtype=np.int64),
    )
    assert status == 0

    empty_index = AGENT_COUNT
    one = _packed_fixture(extra_empty_rows=1)
    one_rows = _rows_from_planes(one.tensors)
    one_owners = _OwnerBundle(one.tensors)
    # Kill all rows, then restore into one isolated destination.
    for index in range(AGENT_COUNT):
        assert (
            oracle.library.eiel_modular_recurrent_death_cleanup_v1(
                one_rows,
                len(one_rows),
                PROFILE,
                ctypes.byref(one_owners.mutable),
                index,
                one_owners.ids[index],
                one_owners.tokens[index],
            )
            == 0
        )
    assert (
        oracle.library.eiel_modular_recurrent_checkpoint_restore_v1(
            ctypes.byref(checkpoint),
            one_rows,
            len(one_rows),
            PROFILE,
            ctypes.byref(one_owners.mutable),
            empty_index,
            one_owners.tokens[empty_index],
        )
        == 0
    )
    isolated_observations = np.zeros(
        (len(one_rows), OBSERVATION_WIDTH), dtype=np.float32
    )
    isolated_observations[empty_index] = observations[1, 0]
    isolated_fatigue = np.zeros((len(one_rows),), dtype=np.float32)
    isolated_fatigue[empty_index] = observations[1, 0, LOCAL_SIZE]
    status, isolated_actions, _receipt = _call_tick(
        oracle.library,
        one_rows,
        one_owners,
        isolated_observations,
        isolated_fatigue,
        np.ones((len(one_rows),), dtype=np.int64),
    )
    assert status == 0
    assert bytes(one_rows[empty_index]) == bytes(rows[0])
    np.testing.assert_allclose(
        isolated_actions[empty_index], continued_actions[0], rtol=0.0, atol=0.0
    )

    cost = CostCapacityReceiptV1()
    assert (
        oracle.library.eiel_modular_recurrent_cost_capacity_v1(
            ctypes.byref(rows[2]), PROFILE, ctypes.byref(cost)
        )
        == 0
    )
    assert cost.logical_module_slots == 16 and cost.logical_connection_slots == 48
    assert cost.occupied_modules == rows[2].module_count
    assert cost.occupied_connections == rows[2].connection_count
    assert list(cost.counts) == list(rows[2].structural_cost_counts)
    assert list(cost.components) == list(rows[2].structural_cost_components)
    assert cost.total == rows[2].structural_cost_total
    assert cost.reporting_only == 1
    for neutral in (4, 5):
        assert rows[neutral].module_count == 0
        assert continued_actions[neutral].tolist() == [0.0] * ACTION_WIDTH


@pytest.mark.parametrize("damage", ["nan", "order", "owner", "prerequisite_identity"])
def test_malformed_input_is_atomic(oracle, damage: str) -> None:
    packed = _packed_fixture()
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    observations, _base, _policy, _masks = _fixed_inputs()
    inputs = observations[0].copy()
    if damage == "nan":
        inputs[0, 0] = np.nan
    elif damage == "order":
        rows[0].module_execution_order[0] = -1
    elif damage == "owner":
        owners.tokens[0] += 1
    else:
        rows[2].development_prerequisite_homology_index[2] = 12
    before = bytes(rows)
    owner_before = (
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    )
    status, outputs, receipt = _call_tick(
        oracle.library,
        rows,
        owners,
        inputs,
        observations[0, :, LOCAL_SIZE],
        np.zeros((AGENT_COUNT,), dtype=np.int64),
    )
    assert status != 0
    assert bytes(rows) == before
    assert (
        list(owners.ids),
        list(owners.tokens),
        list(owners.alive),
    ) == owner_before
    assert outputs.tolist() == [[-991.0] * ACTION_WIDTH] * AGENT_COUNT
    assert bytes(receipt) == bytes(
        TickReceiptV1(-991, -992, -993, -994, -995, -996, 0xA5, 0x5A)
    )


def test_tick_rejects_overlapping_authoritative_and_scratch_rows(oracle) -> None:
    packed = _packed_fixture()
    rows = _rows_from_planes(packed.tensors)
    owners = _OwnerBundle(packed.tensors)
    observations, _base, _policy, _masks = _fixed_inputs()
    inputs = np.ascontiguousarray(observations[0], dtype=np.float32)
    fatigue = np.ascontiguousarray(observations[0, :, LOCAL_SIZE], dtype=np.float32)
    stages = np.zeros((AGENT_COUNT,), dtype=np.int64)
    scratch_actions = np.full((AGENT_COUNT, ACTION_WIDTH), -111.0, dtype=np.float32)
    actions = np.full((AGENT_COUNT, ACTION_WIDTH), -222.0, dtype=np.float32)
    receipt = TickReceiptV1(-991, -992, -993, -994, -995, -996, 0xA5, 0x5A)
    rows_before = bytes(rows)
    status = oracle.library.eiel_modular_recurrent_tick_v1(
        rows,
        len(rows),
        PROFILE,
        ctypes.byref(owners.const),
        inputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        inputs.size,
        fatigue.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fatigue.size,
        stages.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        stages.size,
        rows,
        len(rows),
        scratch_actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scratch_actions.size,
        actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        actions.size,
        ctypes.byref(receipt),
    )
    assert status == 3
    assert bytes(rows) == rows_before
    assert actions.tolist() == [[-222.0] * ACTION_WIDTH] * AGENT_COUNT
    assert bytes(receipt) == bytes(
        TickReceiptV1(-991, -992, -993, -994, -995, -996, 0xA5, 0x5A)
    )


def test_source_and_binary_identity_fail_closed(oracle, tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_BUILD_MANIFEST.read_text(encoding="utf-8"))
    payload["sources"][0]["sha256"] = "0" * 64
    bad_manifest = tmp_path / "bad-manifest.json"
    bad_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModularRecurrentCpuArtifactError, match="source hash mismatch"):
        load_prebuilt_modular_recurrent_cpu_oracle(
            oracle.artifact_path, manifest_path=bad_manifest
        )

    damaged = tmp_path / oracle.artifact_path.name
    shutil.copyfile(oracle.artifact_path, damaged)
    content = bytearray(damaged.read_bytes())
    content[-1] ^= 1
    damaged.write_bytes(content)
    with pytest.raises(
        ModularRecurrentCpuArtifactError, match="artifact hash mismatch"
    ):
        load_prebuilt_modular_recurrent_cpu_oracle(damaged)

    preserved_directory = tmp_path / "preserved"
    preserved_directory.mkdir()
    preserved = preserved_directory / oracle.artifact_path.name
    shutil.copyfile(oracle.artifact_path, preserved)
    preserved_bytes = preserved.read_bytes()
    bad_artifact_payload = json.loads(
        DEFAULT_BUILD_MANIFEST.read_text(encoding="utf-8")
    )
    bad_artifact_payload["artifact"]["sha256"] = "0" * 64
    bad_artifact_manifest = tmp_path / "bad-artifact-manifest.json"
    bad_artifact_manifest.write_text(json.dumps(bad_artifact_payload), encoding="utf-8")
    with pytest.raises(
        ModularRecurrentCpuArtifactError, match="artifact hash mismatch"
    ):
        build_prebuilt_modular_recurrent_cpu_oracle(
            preserved_directory, manifest_path=bad_artifact_manifest
        )
    assert preserved.read_bytes() == preserved_bytes


def test_probe_is_asan_ubsan_clean(tmp_path: Path) -> None:
    executable = tmp_path / "modular-recurrent-probe"
    command = [
        "g++",
        "-std=c++17",
        "-O1",
        "-g",
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        f"-I{NATIVE}",
        str(PROBE),
        str(NATIVE / "modular_recurrent_cpu_oracle_c_api_v1.cpp"),
        "-o",
        str(executable),
    ]
    build = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    if build.returncode != 0 and "unrecognized" in build.stderr.lower():
        pytest.skip("ASan/UBSan are unsupported by this compiler")
    assert build.returncode == 0, build.stderr
    completed = subprocess.run(
        [str(executable)],
        cwd=ROOT,
        env={"ASAN_OPTIONS": "detect_leaks=1", "UBSAN_OPTIONS": "halt_on_error=1"},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
