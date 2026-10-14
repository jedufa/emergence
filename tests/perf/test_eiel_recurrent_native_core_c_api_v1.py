from __future__ import annotations

import ctypes
from copy import deepcopy
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from agents.structural_identity_v1 import new_native_identity
from engine.evolution.p12_stage2_mutation_oracles_v1 import (
    OrdinaryBirthOwnerV1,
    oracle_sha256_v1,
    recurrent_ordinary_birth_oracle_v1,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    ModularRecurrentMutationReceiptValidationError,
    build_modular_recurrent_mutation_receipt_v1,
    canonical_modular_recurrent_mutation_receipt_bytes_v1,
)
from engine.perf.recurrent_native.modular_recurrent_invalid_input_receipt_v1 import (
    build_modular_recurrent_invalid_input_receipt_v1,
    canonical_modular_recurrent_invalid_input_receipt_bytes_v1,
    invalid_input_digest_sha256_v1,
)
from engine.perf.recurrent_native.native_core_ordinary_birth_receipt_bridge_v1 import (
    DISABLED_REASON_BY_CODE,
    DISABLED_REASON_CODE_BY_REASON,
    DISABLED_REASON_CODE_ORDER,
    NativeCoreOrdinaryBirthBridgeError,
    ORDINARY_BIRTH_TERMINAL_PRECEDENCE_V1,
    build_invalid_input_receipt_from_native_core_ordinary_birth_v1,
    build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1,
    build_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1,
    build_recurrent_receipt_from_native_core_ordinary_birth_v1,
    project_python_payload_from_mapped_native_core_ordinary_birth_v1,
)


ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "engine/perf/native/recurrent"
BUILD = Path("/tmp/emergence-test/runtime/eiel_gru_native_migration_20260720/core_c_api_build")


class MutationReceiptV1(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_int32),
        ("operation", ctypes.c_int32),
        ("selected_slot", ctypes.c_int32),
        ("changed_dimension", ctypes.c_uint32),
        ("changed", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 7),
        ("digest_before", ctypes.c_uint64),
        ("digest_after", ctypes.c_uint64),
    ]


class OrdinaryBirthReceiptV1(ctypes.Structure):
    _fields_ = [
        ("transaction_status", ctypes.c_int32),
        ("native_status", ctypes.c_int32),
        ("operation", ctypes.c_int32),
        ("selected_slot", ctypes.c_int32),
        ("changed_dimension", ctypes.c_uint32),
        ("changed", ctypes.c_uint8),
        ("committed", ctypes.c_uint8),
        ("capacity_available", ctypes.c_uint8),
        ("publish_success", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 4),
        ("parent_agent_id", ctypes.c_int64),
        ("parent_slot", ctypes.c_int64),
        ("parent_slot_token", ctypes.c_int64),
        ("child_agent_id", ctypes.c_int64),
        ("child_slot", ctypes.c_int64),
        ("expected_child_slot_token", ctypes.c_int64),
        ("child_publication_token", ctypes.c_int64),
        ("draw0_float64_bits", ctypes.c_uint64),
        ("draw1_float64_bits", ctypes.c_uint64),
        ("parent_digest_before", ctypes.c_uint64),
        ("child_digest_before", ctypes.c_uint64),
        ("proposal_digest", ctypes.c_uint64),
        ("child_digest_after", ctypes.c_uint64),
        ("selected_module_instance_id", ctypes.c_uint64),
        ("selected_homology_id", ctypes.c_uint64),
        ("value_before_float32", ctypes.c_float),
        ("value_after_float32", ctypes.c_float),
    ]


OrdinaryBirthReceiptV2 = OrdinaryBirthReceiptV1


class OrdinaryBirthReceiptV4(ctypes.Structure):
    _fields_ = [
        *OrdinaryBirthReceiptV1._fields_,
        ("disabled_reason_code", ctypes.c_int32),
        ("reserved_v4", ctypes.c_uint8 * 4),
    ]


def _float64_bits(value: float) -> int:
    return int.from_bytes(struct.pack(">d", value), "big")


def _receipt_dict(receipt: OrdinaryBirthReceiptV1) -> dict[str, int | float]:
    result = {}
    for name, _ in receipt.__class__._fields_:
        if name in {"reserved", "reserved_v4"}:
            continue
        value = getattr(receipt, name)
        result[name] = float(value) if name.endswith("_float32") else int(value)
    return result


def _library() -> ctypes.CDLL:
    BUILD.mkdir(parents=True, exist_ok=True)
    artifact = BUILD / "libeiel_modular_recurrent_core_c_api_v1.so"
    command = [
        "g++",
        "-O0",
        "-g0",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-Wall",
        "-Wextra",
        "-Werror",
        f"-I{NATIVE}",
        str(NATIVE / "modular_recurrent_core_c_api_v1.cpp"),
        str(NATIVE / "modular_recurrent_v1.cpp"),
        "-o",
        str(artifact),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return ctypes.CDLL(str(artifact))


def _configure(library: ctypes.CDLL) -> None:
    library.eiel_modular_recurrent_core_abi_version_v1.restype = ctypes.c_uint32
    library.eiel_modular_recurrent_core_fixed_envelope_size_v1.restype = ctypes.c_int64
    library.eiel_modular_recurrent_core_fixed_envelope_alignment_v1.restype = (
        ctypes.c_int64
    )
    byte_p = ctypes.c_void_p
    library.eiel_modular_recurrent_core_initialize_empty_v1.argtypes = [
        byte_p,
        ctypes.c_int64,
        ctypes.c_uint32,
    ]
    library.eiel_modular_recurrent_core_initialize_empty_v1.restype = ctypes.c_int32
    library.eiel_modular_recurrent_core_validate_v1.argtypes = [
        byte_p,
        ctypes.c_int64,
    ]
    library.eiel_modular_recurrent_core_validate_v1.restype = ctypes.c_int32
    library.eiel_modular_recurrent_core_canonical_digest_v1.argtypes = [
        byte_p,
        ctypes.c_int64,
    ]
    library.eiel_modular_recurrent_core_canonical_digest_v1.restype = (
        ctypes.c_uint64
    )
    library.eiel_modular_recurrent_core_module_count_v1.argtypes = [
        byte_p,
        ctypes.c_int64,
    ]
    library.eiel_modular_recurrent_core_module_count_v1.restype = ctypes.c_int32
    library.eiel_modular_recurrent_core_activate_integrator_v1.argtypes = [
        byte_p,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_uint16),
    ]
    library.eiel_modular_recurrent_core_activate_integrator_v1.restype = ctypes.c_int32
    library.eiel_modular_recurrent_core_neutral_duplicate_v1.argtypes = [
        byte_p,
        ctypes.c_int64,
        ctypes.c_uint16,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint16),
    ]
    library.eiel_modular_recurrent_core_neutral_duplicate_v1.restype = ctypes.c_int32
    library.eiel_modular_recurrent_core_mutate_two_draws_v1.argtypes = [
        byte_p,
        ctypes.c_int64,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(MutationReceiptV1),
    ]
    library.eiel_modular_recurrent_core_mutate_two_draws_v1.restype = ctypes.c_int32
    library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v1.argtypes = [
        byte_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.POINTER(OrdinaryBirthReceiptV1),
    ]
    library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v1.restype = (
        ctypes.c_int32
    )
    library.eiel_modular_recurrent_core_ordinary_birth_receipt_layout_version_v2.restype = (
        ctypes.c_uint32
    )
    library.eiel_modular_recurrent_core_ordinary_birth_receipt_size_v2.restype = (
        ctypes.c_int64
    )
    library.eiel_modular_recurrent_core_ordinary_birth_receipt_alignment_v2.restype = (
        ctypes.c_int64
    )
    library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v2.argtypes = [
        byte_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.POINTER(OrdinaryBirthReceiptV2),
    ]
    library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v2.restype = (
        ctypes.c_int32
    )
    library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v3.argtypes = [
        byte_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.POINTER(OrdinaryBirthReceiptV2),
    ]
    library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v3.restype = (
        ctypes.c_int32
    )
    library.eiel_modular_recurrent_core_ordinary_birth_receipt_layout_version_v4.restype = (
        ctypes.c_uint32
    )
    library.eiel_modular_recurrent_core_ordinary_birth_receipt_size_v4.restype = (
        ctypes.c_int64
    )
    library.eiel_modular_recurrent_core_ordinary_birth_receipt_alignment_v4.restype = (
        ctypes.c_int64
    )
    library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v4.argtypes = [
        byte_p,
        ctypes.c_int64,
        byte_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint32,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.POINTER(OrdinaryBirthReceiptV4),
    ]
    library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v4.restype = (
        ctypes.c_int32
    )


def _fixture(library: ctypes.CDLL) -> tuple[ctypes.Array, int]:
    size = int(library.eiel_modular_recurrent_core_fixed_envelope_size_v1())
    envelope = ctypes.create_string_buffer(size)
    assert library.eiel_modular_recurrent_core_initialize_empty_v1(
        envelope, size, 0
    ) == 0
    source = ctypes.c_uint16(65535)
    assert (
        library.eiel_modular_recurrent_core_activate_integrator_v1(
            envelope,
            size,
            1,
            100,
            10,
            ctypes.c_float(1.0),
            ctypes.c_float(0.25),
            ctypes.c_float(0.5),
            ctypes.c_float(0.5),
            0,
            ctypes.byref(source),
        )
        == 0
    )
    duplicate = ctypes.c_uint16(65535)
    assert (
        library.eiel_modular_recurrent_core_neutral_duplicate_v1(
            envelope,
            size,
            source.value,
            2,
            200,
            20,
            ctypes.byref(duplicate),
        )
        == 0
    )
    assert library.eiel_modular_recurrent_core_validate_v1(envelope, size) == 0
    assert library.eiel_modular_recurrent_core_module_count_v1(envelope, size) == 2
    return envelope, size


def _empty_envelope(library: ctypes.CDLL, size: int) -> ctypes.Array:
    envelope = ctypes.create_string_buffer(size)
    assert library.eiel_modular_recurrent_core_initialize_empty_v1(
        envelope, size, 0
    ) == 0
    return envelope


def _python_payload_matching_native_fixture() -> dict:
    context = {
        "campaign_id": "p12-native-core-payload-observability",
        "branch_id": "native-core-c-api",
        "generation": 0,
        "tick": 0,
        "child_agent_id": (1 << 40) + 7,
        "parent_agent_ids": [100],
        "lineage_token": "p12-native-core-payload-observability",
    }
    source = {
        "module_id": "m000",
        "module_type": "generic",
        "params": {
            "expression_scale": 1.0,
            "live_operator": "bounded_scalar_integrator_v1",
            "live_retention": 0.25,
            "live_input_gain": 0.5,
            "live_output_gain": 0.5,
        },
        "io": {
            "inputs": [{"slot_id": "in", "value_type": "scalar"}],
            "outputs": [{"slot_id": "out", "value_type": "scalar"}],
            "caps": {"max_fanin": 6, "max_fanout": 6},
        },
        **new_native_identity(
            context=context,
            operator_id="founder",
            attempt_index=0,
            operator_role="source",
        ),
    }
    descendant = deepcopy(source)
    descendant.update(
        {
            "module_id": "m001",
            **new_native_identity(
                context=context,
                operator_id="duplicate_with_decay",
                attempt_index=0,
                operator_role="descendant",
                parent_module=source,
            ),
        }
    )
    descendant["params"]["live_output_gain"] = 0.0
    descendant["params"]["observation_routing"] = {
        "schema_version": "escap4_observation_routing_v1",
        "registry_version": "perception_routing_contract_v1",
        "bindings": [
            {
                "semantic_id": "self_energy",
                "target_slot": "in",
                "gain": 0.5,
                "enabled": True,
            }
        ],
    }
    descendant["params"]["output_routing"] = {
        "schema_version": "escap5_output_routing_v1",
        "registry_version": "eiel_action_semantic_routing_registry_v1",
        "bindings": [
            {
                "semantic_id": "rest",
                "source_slot": "out",
                "gain": 0.5,
                "enabled": True,
            }
        ],
    }
    return {
        "schema_version": "modular_genome_v1",
        "b2_invalid_policy": "reject",
        "genes": {},
        "modules": [source, descendant],
        "connections": [],
    }


def _ordinary_birth(
    library: ctypes.CDLL,
    parent: ctypes.Array,
    child: ctypes.Array,
    size: int,
    *,
    parent_slot: int = 2,
    child_slot: int = 4,
    parent_slot_token: int = 12,
    expected_child_slot_token: int = 17,
    child_publication_token: int = 18,
    draw0_bits: int | None = None,
    draw1_bits: int | None = None,
    capacity_available: int = 1,
    publish_success: int = 1,
) -> tuple[int, OrdinaryBirthReceiptV1]:
    receipt = OrdinaryBirthReceiptV1()
    status = library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v1(
        parent,
        size,
        child,
        size,
        100,
        parent_slot,
        parent_slot_token,
        (1 << 40) + 7,
        child_slot,
        expected_child_slot_token,
        child_publication_token,
        _float64_bits(0.625) if draw0_bits is None else draw0_bits,
        _float64_bits(0.5) if draw1_bits is None else draw1_bits,
        capacity_available,
        publish_success,
        ctypes.byref(receipt),
    )
    return status, receipt


def _ordinary_birth_v2(
    library: ctypes.CDLL,
    parent: ctypes.Array,
    child: ctypes.Array,
    size: int,
    *,
    parent_slot: int = 2,
    child_slot: int = 4,
    parent_slot_token: int = 12,
    expected_child_slot_token: int = 17,
    child_publication_token: int = 18,
    draw0_bits: int | None = None,
    draw1_bits: int | None = None,
    capacity_available: int = 1,
    publish_success: int = 1,
) -> tuple[int, OrdinaryBirthReceiptV2]:
    receipt = OrdinaryBirthReceiptV2()
    status = library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v2(
        parent,
        size,
        child,
        size,
        100,
        parent_slot,
        parent_slot_token,
        (1 << 40) + 7,
        child_slot,
        expected_child_slot_token,
        child_publication_token,
        _float64_bits(0.625) if draw0_bits is None else draw0_bits,
        _float64_bits(0.5) if draw1_bits is None else draw1_bits,
        capacity_available,
        publish_success,
        ctypes.byref(receipt),
    )
    return status, receipt


def _ordinary_birth_v3(
    library: ctypes.CDLL,
    parent: ctypes.Array,
    child: ctypes.Array,
    size: int,
    *,
    mutation_enabled: int = 1,
    draw0_bits: int | None = None,
    draw1_bits: int | None = None,
    capacity_available: int = 1,
    publish_success: int = 1,
) -> tuple[int, OrdinaryBirthReceiptV2]:
    receipt = OrdinaryBirthReceiptV2()
    status = library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v3(
        parent,
        size,
        child,
        size,
        100,
        2,
        12,
        (1 << 40) + 7,
        4,
        17,
        18,
        _float64_bits(0.625) if draw0_bits is None else draw0_bits,
        _float64_bits(0.5) if draw1_bits is None else draw1_bits,
        mutation_enabled,
        capacity_available,
        publish_success,
        ctypes.byref(receipt),
    )
    return status, receipt


def _ordinary_birth_v4(
    library: ctypes.CDLL,
    parent: ctypes.Array,
    child: ctypes.Array,
    size: int,
    *,
    disabled_reason_code: int = 0,
    parent_slot: int = 2,
    child_slot: int = 4,
    parent_slot_token: int = 12,
    expected_child_slot_token: int = 17,
    child_publication_token: int = 18,
    draw0_bits: int | None = None,
    draw1_bits: int | None = None,
    capacity_available: int = 1,
    publish_success: int = 1,
) -> tuple[int, OrdinaryBirthReceiptV4]:
    receipt = OrdinaryBirthReceiptV4()
    status = library.eiel_modular_recurrent_core_ordinary_birth_two_draws_v4(
        parent,
        size,
        child,
        size,
        100,
        parent_slot,
        parent_slot_token,
        (1 << 40) + 7,
        child_slot,
        expected_child_slot_token,
        child_publication_token,
        _float64_bits(0.625) if draw0_bits is None else draw0_bits,
        _float64_bits(0.5) if draw1_bits is None else draw1_bits,
        disabled_reason_code,
        capacity_available,
        publish_success,
        ctypes.byref(receipt),
    )
    return status, receipt


def _expected_rejected_replay_receipt(
    *,
    parent_slot: int = 2,
    child_slot: int = 4,
    parent_slot_token: int = 12,
    child_publication_token: int = 18,
    draw0: float = 0.625,
    draw1: float = 0.5,
    operation: str = "refine_descendant_parameter",
    reason: str,
) -> object:
    return build_modular_recurrent_mutation_receipt_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        transactions=[
            {
                "transaction_kind": "ordinary_birth",
                "transaction_ordinal": 0,
                "parent_agent_id": 100,
                "parent_slot": parent_slot,
                "parent_slot_token": parent_slot_token,
                "child_agent_id": (1 << 40) + 7,
                "child_slot": child_slot,
                "child_slot_token": child_publication_token,
                "attempt_ordinal": 0,
                "draw0_float64_bits": f"{_float64_bits(draw0):016x}",
                "draw1_float64_bits": f"{_float64_bits(draw1):016x}",
                "operation": operation,
                "status": "rejected",
                "reason": reason,
                "selected_instance_id_index": -1,
                "selected_homology_id_index": -1,
                "changed_dimension": "none",
                "changed": False,
            }
        ],
    )


def _native_v2_replay_fingerprint(
    *,
    case_draw0: float = 0.625,
    case_draw1: float = 0.5,
    mutation_enabled: int | None = None,
    disabled_reason_code: int | None = None,
    native_kwargs: dict[str, int] | None = None,
) -> tuple[dict, str, bytes, int]:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    python_parent = _python_payload_matching_native_fixture()
    if disabled_reason_code is not None:
        call = _ordinary_birth_v4
    elif mutation_enabled is not None:
        call = _ordinary_birth_v3
    else:
        call = _ordinary_birth_v2
    call_kwargs = dict(native_kwargs or {})
    if mutation_enabled is not None:
        call_kwargs["mutation_enabled"] = mutation_enabled
    if disabled_reason_code is not None:
        call_kwargs["disabled_reason_code"] = disabled_reason_code
    status, native_receipt = call(
        library,
        parent,
        child,
        size,
        draw0_bits=_float64_bits(case_draw0),
        draw1_bits=_float64_bits(case_draw1),
        **call_kwargs,
    )
    assert status == 0
    native_receipt_dict = _receipt_dict(native_receipt)

    identity_table: tuple[str, ...] = ()
    native_instance_map: dict[int, str] = {}
    native_homology_map: dict[int, str] = {}
    if native_receipt.changed:
        owner_parent = OrdinaryBirthOwnerV1(agent_id=100, slot=2, slot_token=12)
        owner_child = OrdinaryBirthOwnerV1(
            agent_id=(1 << 40) + 7,
            slot=4,
            slot_token=18,
        )
        _python_child, python_receipt = recurrent_ordinary_birth_oracle_v1(
            python_parent,
            parent_owner=owner_parent,
            child_owner=owner_child,
            draw0=case_draw0,
            draw1=case_draw1,
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
        )
        identity_table = tuple(
            sorted(
                {
                    value
                    for module in python_parent["modules"]
                    for value in (
                        module["module_instance_id"],
                        module["homology_id"],
                    )
                }
            )
        )
        native_instance_map = {
            native_receipt.selected_module_instance_id: python_receipt[
                "selected_module_instance_id"
            ]
        }
        native_homology_map = {
            native_receipt.selected_homology_id: python_receipt["selected_homology_id"]
        }

    replay_receipt = (
        build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
            native_receipt_dict,
            profile_id="b2-16x48-v1",
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
            identity_table=identity_table,
            native_module_instance_id_to_identity=native_instance_map,
            native_homology_id_to_identity=native_homology_map,
        )
    )
    child_digest = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)
    return (
        native_receipt_dict,
        replay_receipt.canonical_sha256,
        canonical_modular_recurrent_mutation_receipt_bytes_v1(replay_receipt),
        child_digest,
    )


def test_native_core_c_api_exposes_constrained_two_draw_bins() -> None:
    library = _library()
    _configure(library)

    assert library.eiel_modular_recurrent_core_abi_version_v1() == 1
    assert library.eiel_modular_recurrent_core_fixed_envelope_size_v1() > 0
    assert library.eiel_modular_recurrent_core_fixed_envelope_alignment_v1() > 0
    for draw0, draw1, operation, changed_dimension in (
        (0.125, 0.0, 0, 1),
        (0.375, 0.0, 1, 0),
        (0.625, 0.5, 2, 8),
        (0.875, 0.0, 3, 1 << 18),
    ):
        envelope, size = _fixture(library)
        receipt = MutationReceiptV1()
        status = library.eiel_modular_recurrent_core_mutate_two_draws_v1(
            envelope,
            size,
            ctypes.c_float(draw0),
            ctypes.c_float(draw1),
            ctypes.byref(receipt),
        )

        assert status == 0
        assert receipt.status == 0
        assert receipt.operation == operation
        assert receipt.selected_slot in {0, 1}
        assert receipt.changed_dimension == changed_dimension
        assert receipt.changed == 1
        assert receipt.digest_before != 0
        assert receipt.digest_after != 0
        assert receipt.digest_before != receipt.digest_after


def test_native_core_c_api_invalid_draw_is_fail_closed() -> None:
    library = _library()
    _configure(library)
    envelope, size = _fixture(library)
    before = library.eiel_modular_recurrent_core_canonical_digest_v1(envelope, size)
    receipt = MutationReceiptV1()

    status = library.eiel_modular_recurrent_core_mutate_two_draws_v1(
        envelope,
        size,
        ctypes.c_float(-0.1),
        ctypes.c_float(0.5),
        ctypes.byref(receipt),
    )

    assert status == 1
    assert receipt.status == 1
    assert receipt.changed == 0
    assert receipt.digest_before == before
    assert receipt.digest_after == before
    assert library.eiel_modular_recurrent_core_canonical_digest_v1(envelope, size) == before


def test_native_core_c_api_ordinary_birth_commits_with_exact_draw_bits() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    parent_before = library.eiel_modular_recurrent_core_canonical_digest_v1(parent, size)
    child_before = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)
    draw0_bits = _float64_bits(0.625)
    draw1_bits = _float64_bits(0.5)

    status, receipt = _ordinary_birth(
        library,
        parent,
        child,
        size,
        draw0_bits=draw0_bits,
        draw1_bits=draw1_bits,
    )

    assert status == 0
    assert receipt.transaction_status == 0
    assert receipt.native_status == 0
    assert receipt.committed == 1
    assert receipt.capacity_available == 1
    assert receipt.publish_success == 1
    assert receipt.operation == 2
    assert receipt.changed_dimension == 8
    assert receipt.changed == 1
    assert receipt.selected_module_instance_id == 2
    assert receipt.selected_homology_id == 200
    assert receipt.value_before_float32 == ctypes.c_float(0.5).value
    assert receipt.value_after_float32 == ctypes.c_float(-2.0).value
    assert receipt.parent_agent_id == 100
    assert receipt.child_agent_id == (1 << 40) + 7
    assert receipt.draw0_float64_bits == draw0_bits
    assert receipt.draw1_float64_bits == draw1_bits
    assert receipt.parent_digest_before == parent_before
    assert receipt.child_digest_before == child_before
    assert receipt.parent_digest_before != receipt.proposal_digest
    assert receipt.child_digest_before != receipt.child_digest_after
    assert receipt.proposal_digest == receipt.child_digest_after
    assert library.eiel_modular_recurrent_core_canonical_digest_v1(parent, size) == parent_before
    assert library.eiel_modular_recurrent_core_canonical_digest_v1(child, size) == (
        receipt.child_digest_after
    )


def test_native_core_c_api_ordinary_birth_v2_layout_and_receipt_fields() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    draw0_bits = _float64_bits(0.625)
    draw1_bits = _float64_bits(0.5)

    status, receipt = _ordinary_birth_v2(
        library,
        parent,
        child,
        size,
        draw0_bits=draw0_bits,
        draw1_bits=draw1_bits,
    )

    assert library.eiel_modular_recurrent_core_ordinary_birth_receipt_layout_version_v2() == 2
    assert library.eiel_modular_recurrent_core_ordinary_birth_receipt_size_v2() == (
        ctypes.sizeof(OrdinaryBirthReceiptV2)
    )
    assert library.eiel_modular_recurrent_core_ordinary_birth_receipt_alignment_v2() == (
        ctypes.alignment(OrdinaryBirthReceiptV2)
    )
    assert OrdinaryBirthReceiptV2.selected_module_instance_id.offset > (
        OrdinaryBirthReceiptV2.child_digest_after.offset
    )
    assert OrdinaryBirthReceiptV2.value_before_float32.offset > (
        OrdinaryBirthReceiptV2.selected_homology_id.offset
    )
    assert status == 0
    assert receipt.transaction_status == 0
    assert receipt.committed == 1
    assert receipt.draw0_float64_bits == draw0_bits
    assert receipt.draw1_float64_bits == draw1_bits
    assert receipt.selected_module_instance_id == 2
    assert receipt.selected_homology_id == 200
    assert receipt.value_before_float32 == ctypes.c_float(0.5).value
    assert receipt.value_after_float32 == ctypes.c_float(-2.0).value


def test_native_core_c_api_ordinary_birth_capacity_rejection_preserves_child() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    child_before = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)

    status, receipt = _ordinary_birth(
        library,
        parent,
        child,
        size,
        capacity_available=0,
    )

    assert status == 0
    assert receipt.transaction_status == 1
    assert receipt.committed == 0
    assert receipt.proposal_digest == 0
    assert receipt.child_digest_before == child_before
    assert receipt.child_digest_after == child_before
    assert library.eiel_modular_recurrent_core_canonical_digest_v1(child, size) == child_before


def test_native_core_c_api_ordinary_birth_publication_rejection_preserves_child() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    child_before = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)

    status, receipt = _ordinary_birth(
        library,
        parent,
        child,
        size,
        publish_success=0,
    )

    assert status == 0
    assert receipt.transaction_status == 2
    assert receipt.committed == 0
    assert receipt.proposal_digest != 0
    assert receipt.proposal_digest != receipt.parent_digest_before
    assert receipt.child_digest_before == child_before
    assert receipt.child_digest_after == child_before
    assert library.eiel_modular_recurrent_core_canonical_digest_v1(child, size) == child_before


def test_native_core_c_api_ordinary_birth_stale_owner_token_preserves_child() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    child_before = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)

    status, receipt = _ordinary_birth(
        library,
        parent,
        child,
        size,
        parent_slot=4,
        child_slot=4,
        parent_slot_token=12,
        expected_child_slot_token=11,
        child_publication_token=12,
    )

    assert status == 0
    assert receipt.transaction_status == 3
    assert receipt.committed == 0
    assert receipt.proposal_digest == 0
    assert receipt.child_digest_after == child_before
    assert library.eiel_modular_recurrent_core_canonical_digest_v1(child, size) == child_before


def test_native_core_c_api_ordinary_birth_invalid_draw_preserves_child() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    child_before = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)

    status, receipt = _ordinary_birth(
        library,
        parent,
        child,
        size,
        draw0_bits=_float64_bits(1.0),
    )

    assert status == 0
    assert receipt.transaction_status == 4
    assert receipt.native_status == 7
    assert receipt.committed == 0
    assert receipt.proposal_digest == 0
    assert receipt.child_digest_after == child_before
    assert library.eiel_modular_recurrent_core_canonical_digest_v1(child, size) == child_before


@pytest.mark.parametrize(
    ("case_id", "mutation_enabled", "native_kwargs", "expected_status", "expect_digest"),
    (
        ("committed_publication", None, {}, 0, "published_child"),
        ("capacity_unavailable", None, {"capacity_available": 0}, 1, "absent"),
        ("publication_failure", None, {"publish_success": 0}, 2, "proposal_only"),
        (
            "owner_mismatch",
            None,
            {"expected_child_slot_token": 18, "child_publication_token": 18},
            3,
            "absent",
        ),
        ("invalid_draw", None, {"draw0_bits": _float64_bits(1.0)}, 4, "absent"),
        ("mutation_disabled", 0, {}, 6, "absent"),
    ),
)
def test_native_core_proposal_digest_vectors_are_explicit(
    case_id: str,
    mutation_enabled: int | None,
    native_kwargs: dict[str, int],
    expected_status: int,
    expect_digest: str,
) -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    parent_before = library.eiel_modular_recurrent_core_canonical_digest_v1(parent, size)
    child_before = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)

    if mutation_enabled is None:
        status, receipt = _ordinary_birth_v2(
            library,
            parent,
            child,
            size,
            **native_kwargs,
        )
    else:
        status, receipt = _ordinary_birth_v3(
            library,
            parent,
            child,
            size,
            mutation_enabled=mutation_enabled,
            **native_kwargs,
        )

    assert case_id
    assert status == 0
    assert receipt.transaction_status == expected_status
    assert receipt.parent_digest_before == parent_before
    assert receipt.child_digest_before == child_before
    if expect_digest == "published_child":
        assert receipt.committed == 1
        assert receipt.proposal_digest != 0
        assert receipt.proposal_digest != parent_before
        assert receipt.proposal_digest != child_before
        assert receipt.proposal_digest == receipt.child_digest_after
        assert receipt.proposal_digest == library.eiel_modular_recurrent_core_canonical_digest_v1(
            child, size
        )
    elif expect_digest == "proposal_only":
        assert receipt.committed == 0
        assert receipt.proposal_digest != 0
        assert receipt.proposal_digest != parent_before
        assert receipt.proposal_digest != child_before
        assert receipt.child_digest_after == child_before
        assert library.eiel_modular_recurrent_core_canonical_digest_v1(child, size) == child_before
    else:
        assert expect_digest == "absent"
        assert receipt.committed == 0
        assert receipt.proposal_digest == 0
        assert receipt.child_digest_after == child_before
        assert library.eiel_modular_recurrent_core_canonical_digest_v1(child, size) == child_before


def test_native_core_invalid_draw_maps_to_invalid_input_receipt() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    child_before = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)

    status, native_receipt = _ordinary_birth_v2(
        library,
        parent,
        child,
        size,
        draw0_bits=_float64_bits(1.0),
    )
    native_receipt_dict = _receipt_dict(native_receipt)
    invalid_receipt = build_invalid_input_receipt_from_native_core_ordinary_birth_v1(
        native_receipt_dict,
        profile_id="b2-16x48-v1",
    )
    expected_receipt = build_modular_recurrent_invalid_input_receipt_v1(
        profile_id="b2-16x48-v1",
        input_kind="native_core_ordinary_birth",
        transaction_kind="ordinary_birth",
        parent_agent_id=100,
        parent_slot=2,
        parent_slot_token=12,
        child_agent_id=(1 << 40) + 7,
        child_slot=4,
        child_slot_token=18,
        attempt_ordinal=0,
        draw0_float64_bits=f"{_float64_bits(1.0):016x}",
        draw1_float64_bits=f"{_float64_bits(0.5):016x}",
        operation="unknown",
        reason="invalid_draw",
        native_transaction_status=4,
        native_status=7,
        input_digest_sha256=invalid_input_digest_sha256_v1(native_receipt_dict),
    )

    with pytest.raises(
        ModularRecurrentMutationReceiptValidationError,
        match="must encode a finite value",
    ):
        build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
            native_receipt_dict,
            profile_id="b2-16x48-v1",
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
            identity_table=(),
            native_module_instance_id_to_identity={},
            native_homology_id_to_identity={},
        )

    assert status == 0
    assert native_receipt.transaction_status == 4
    assert native_receipt.native_status == 7
    assert native_receipt.committed == 0
    assert native_receipt.proposal_digest == 0
    assert native_receipt.changed == 0
    assert native_receipt.child_digest_before == child_before
    assert native_receipt.child_digest_after == child_before
    assert invalid_receipt.canonical_sha256 == expected_receipt.canonical_sha256
    assert canonical_modular_recurrent_invalid_input_receipt_bytes_v1(
        invalid_receipt
    ) == canonical_modular_recurrent_invalid_input_receipt_bytes_v1(expected_receipt)


def test_native_core_ordinary_birth_can_emit_canonical_recurrent_receipt() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)

    status, native_receipt = _ordinary_birth(library, parent, child, size)
    recurrent_receipt = build_recurrent_receipt_from_native_core_ordinary_birth_v1(
        _receipt_dict(native_receipt),
        profile_id="b2-12x32-v1",
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        selected_instance_id_index=0,
        selected_homology_id_index=1,
    )
    transaction = recurrent_receipt.transactions[0]
    wire = canonical_modular_recurrent_mutation_receipt_bytes_v1(recurrent_receipt)

    assert status == 0
    assert transaction.transaction_kind == "ordinary_birth"
    assert transaction.parent_agent_id == 100
    assert transaction.child_agent_id == (1 << 40) + 7
    assert transaction.child_slot_token == native_receipt.child_publication_token
    assert transaction.draw0_float64_bits == f"{_float64_bits(0.625):016x}"
    assert transaction.draw1_float64_bits == f"{_float64_bits(0.5):016x}"
    assert transaction.operation == "refine_descendant_parameter"
    assert transaction.status == "accepted"
    assert transaction.reason == "native_core_committed"
    assert transaction.changed_dimension == "live_input_gain"
    assert transaction.changed is True
    assert recurrent_receipt.canonical_sha256
    assert b"modular_two_draws_v1" in wire


def test_native_core_ordinary_birth_matches_python_oracle_semantic_payload_fixture() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    python_parent = _python_payload_matching_native_fixture()

    status, native_receipt = _ordinary_birth(library, parent, child, size)
    python_child, python_receipt = recurrent_ordinary_birth_oracle_v1(
        python_parent,
        parent_owner=OrdinaryBirthOwnerV1(agent_id=100, slot=2, slot_token=12),
        child_owner=OrdinaryBirthOwnerV1(
            agent_id=(1 << 40) + 7,
            slot=4,
            slot_token=18,
        ),
        draw0=0.625,
        draw1=0.5,
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
    )
    identity_table = tuple(
        sorted(
            {
                value
                for module in python_parent["modules"]
                for value in (
                    module["module_instance_id"],
                    module["homology_id"],
                )
            }
        )
    )
    selected_instance_index = identity_table.index(
        python_receipt["selected_module_instance_id"]
    )
    selected_homology_index = identity_table.index(
        python_receipt["selected_homology_id"]
    )
    recurrent_receipt = build_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
        _receipt_dict(native_receipt),
        profile_id="b2-12x32-v1",
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        identity_table=identity_table,
        native_module_instance_id_to_identity={
            2: python_receipt["selected_module_instance_id"],
        },
        native_homology_id_to_identity={
            200: python_receipt["selected_homology_id"],
        },
    )
    transaction = recurrent_receipt.transactions[0]
    descendant = next(
        module
        for module in python_child["modules"]
        if module["module_instance_id"] == python_receipt["selected_module_instance_id"]
    )

    assert status == 0
    assert python_receipt["status"] == "accepted"
    assert python_receipt["reason"] == "proposal_valid"
    assert python_receipt["operation"] == transaction.operation
    assert python_receipt["changed_dimension"] == transaction.changed_dimension
    assert python_receipt["draw0_float64_bits"] == transaction.draw0_float64_bits
    assert python_receipt["draw1_float64_bits"] == transaction.draw1_float64_bits
    assert python_receipt["child_owner"]["agent_id"] == transaction.child_agent_id
    assert python_receipt["child_owner"]["slot_token"] == transaction.child_slot_token
    assert python_receipt["selected_module_instance_id"] == str(
        descendant["module_instance_id"]
    )
    assert python_receipt["selected_homology_id"] == str(descendant["homology_id"])
    assert native_receipt.selected_module_instance_id == 2
    assert native_receipt.selected_homology_id == 200
    assert transaction.selected_instance_id_index == selected_instance_index
    assert transaction.selected_homology_id_index == selected_homology_index
    assert float(descendant["params"]["live_input_gain"]) == native_receipt.value_after_float32
    assert native_receipt.value_before_float32 == ctypes.c_float(0.5).value
    assert native_receipt.value_after_float32 == ctypes.c_float(-2.0).value


def test_native_core_mapped_receipt_bridge_rejects_missing_identity_map() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    python_parent = _python_payload_matching_native_fixture()

    _status, native_receipt = _ordinary_birth(library, parent, child, size)
    _python_child, python_receipt = recurrent_ordinary_birth_oracle_v1(
        python_parent,
        parent_owner=OrdinaryBirthOwnerV1(agent_id=100, slot=2, slot_token=12),
        child_owner=OrdinaryBirthOwnerV1(
            agent_id=(1 << 40) + 7,
            slot=4,
            slot_token=18,
        ),
        draw0=0.625,
        draw1=0.5,
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
    )
    identity_table = tuple(
        sorted(
            {
                value
                for module in python_parent["modules"]
                for value in (
                    module["module_instance_id"],
                    module["homology_id"],
                )
            }
        )
    )

    with pytest.raises(NativeCoreOrdinaryBirthBridgeError, match="selected_homology_id"):
        build_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
            _receipt_dict(native_receipt),
            profile_id="b2-16x48-v1",
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
            identity_table=identity_table,
            native_module_instance_id_to_identity={
                2: python_receipt["selected_module_instance_id"],
            },
            native_homology_id_to_identity={},
        )


def test_native_core_mapped_payload_projection_matches_python_oracle_child_sha() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    python_parent = _python_payload_matching_native_fixture()
    owner_parent = OrdinaryBirthOwnerV1(agent_id=100, slot=2, slot_token=12)
    owner_child = OrdinaryBirthOwnerV1(
        agent_id=(1 << 40) + 7,
        slot=4,
        slot_token=18,
    )

    _status, native_receipt = _ordinary_birth(library, parent, child, size)
    canonical_parent, disabled_receipt = recurrent_ordinary_birth_oracle_v1(
        python_parent,
        parent_owner=owner_parent,
        child_owner=owner_child,
        draw0=0.625,
        draw1=0.5,
        mutation_enabled=False,
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
    )
    python_child, python_receipt = recurrent_ordinary_birth_oracle_v1(
        python_parent,
        parent_owner=owner_parent,
        child_owner=owner_child,
        draw0=0.625,
        draw1=0.5,
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
    )
    identity_table = tuple(
        sorted(
            {
                value
                for module in canonical_parent["modules"]
                for value in (
                    module["module_instance_id"],
                    module["homology_id"],
                )
            }
        )
    )
    native_instance_map = {2: python_receipt["selected_module_instance_id"]}
    native_homology_map = {200: python_receipt["selected_homology_id"]}

    projected = project_python_payload_from_mapped_native_core_ordinary_birth_v1(
        canonical_parent,
        _receipt_dict(native_receipt),
        identity_table=identity_table,
        native_module_instance_id_to_identity=native_instance_map,
        native_homology_id_to_identity=native_homology_map,
    )

    assert disabled_receipt["status"] == "disabled"
    assert python_receipt["status"] == "accepted"
    assert projected == python_child
    assert oracle_sha256_v1(projected) == python_receipt["child_payload_sha256"]


@pytest.mark.parametrize(
    ("case_id", "draw0", "draw1"),
    (
        ("toggle_dormancy", 0.125, 0.0),
        ("prune_eligible_module", 0.375, 0.0),
        ("refine_descendant_parameter", 0.625, 0.5),
        ("refine_descendant_routing_or_expression", 0.875, 0.0),
    ),
)
def test_native_core_v2_receipt_bytes_match_python_oracle_replay_fixture(
    case_id: str,
    draw0: float,
    draw1: float,
) -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    python_parent = _python_payload_matching_native_fixture()
    owner_parent = OrdinaryBirthOwnerV1(agent_id=100, slot=2, slot_token=12)
    owner_child = OrdinaryBirthOwnerV1(
        agent_id=(1 << 40) + 7,
        slot=4,
        slot_token=18,
    )

    status, native_receipt = _ordinary_birth_v2(
        library,
        parent,
        child,
        size,
        draw0_bits=_float64_bits(draw0),
        draw1_bits=_float64_bits(draw1),
    )
    _python_child, python_receipt = recurrent_ordinary_birth_oracle_v1(
        python_parent,
        parent_owner=owner_parent,
        child_owner=owner_child,
        draw0=draw0,
        draw1=draw1,
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
    )
    identity_table = tuple(
        sorted(
            {
                value
                for module in python_parent["modules"]
                for value in (
                    module["module_instance_id"],
                    module["homology_id"],
                )
            }
        )
    )
    selected_instance_index = identity_table.index(
        python_receipt["selected_module_instance_id"]
    )
    selected_homology_index = identity_table.index(
        python_receipt["selected_homology_id"]
    )
    source_module, descendant_module = python_parent["modules"]
    native_instance_map = {
        1: source_module["module_instance_id"],
        2: descendant_module["module_instance_id"],
    }
    native_homology_map = {
        100: source_module["homology_id"],
        200: descendant_module["homology_id"],
    }
    native_replay_receipt = (
        build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
            _receipt_dict(native_receipt),
            profile_id="b2-16x48-v1",
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
            identity_table=identity_table,
            native_module_instance_id_to_identity=native_instance_map,
            native_homology_id_to_identity=native_homology_map,
        )
    )
    python_replay_receipt = build_modular_recurrent_mutation_receipt_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        transactions=[
            {
                "transaction_kind": "ordinary_birth",
                "transaction_ordinal": 0,
                "parent_agent_id": python_receipt["parent_owner"]["agent_id"],
                "parent_slot": python_receipt["parent_owner"]["slot"],
                "parent_slot_token": python_receipt["parent_owner"]["slot_token"],
                "child_agent_id": python_receipt["child_owner"]["agent_id"],
                "child_slot": python_receipt["child_owner"]["slot"],
                "child_slot_token": python_receipt["child_owner"]["slot_token"],
                "attempt_ordinal": 0,
                "draw0_float64_bits": python_receipt["draw0_float64_bits"],
                "draw1_float64_bits": python_receipt["draw1_float64_bits"],
                "operation": python_receipt["operation"],
                "status": python_receipt["status"],
                "reason": python_receipt["reason"],
                "selected_instance_id_index": selected_instance_index,
                "selected_homology_id_index": selected_homology_index,
                "changed_dimension": python_receipt["changed_dimension"],
                "changed": python_receipt["changed"],
            }
        ],
    )

    assert status == 0
    assert python_receipt["operation"] == case_id
    assert python_receipt["replay_receipt_sha256"] == python_replay_receipt.canonical_sha256
    assert native_replay_receipt.canonical_sha256 == python_replay_receipt.canonical_sha256
    assert canonical_modular_recurrent_mutation_receipt_bytes_v1(
        native_replay_receipt
    ) == canonical_modular_recurrent_mutation_receipt_bytes_v1(python_replay_receipt)


@pytest.mark.parametrize(
    ("case_id", "native_kwargs", "oracle_kwargs", "reason"),
    (
        (
            "capacity_unavailable",
            {"capacity_available": 0},
            {"capacity_available": False},
            "capacity_unavailable",
        ),
        (
            "publication_failure",
            {"publish_success": 0},
            {"publish_success": False},
            "child_publication_failed",
        ),
    ),
)
def test_native_core_v2_rejected_receipt_bytes_match_python_oracle_fixture(
    case_id: str,
    native_kwargs: dict[str, int],
    oracle_kwargs: dict[str, bool],
    reason: str,
) -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    python_parent = _python_payload_matching_native_fixture()
    owner_parent = OrdinaryBirthOwnerV1(agent_id=100, slot=2, slot_token=12)
    owner_child = OrdinaryBirthOwnerV1(
        agent_id=(1 << 40) + 7,
        slot=4,
        slot_token=18,
    )

    status, native_receipt = _ordinary_birth_v2(
        library,
        parent,
        child,
        size,
        **native_kwargs,
    )
    _python_child, python_receipt = recurrent_ordinary_birth_oracle_v1(
        python_parent,
        parent_owner=owner_parent,
        child_owner=owner_child,
        draw0=0.625,
        draw1=0.5,
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        **oracle_kwargs,
    )
    native_replay_receipt = (
        build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
            _receipt_dict(native_receipt),
            profile_id="b2-16x48-v1",
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
            identity_table=(),
            native_module_instance_id_to_identity={},
            native_homology_id_to_identity={},
        )
    )
    python_replay_receipt = build_modular_recurrent_mutation_receipt_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        transactions=[
            {
                "transaction_kind": "ordinary_birth",
                "transaction_ordinal": 0,
                "parent_agent_id": python_receipt["parent_owner"]["agent_id"],
                "parent_slot": python_receipt["parent_owner"]["slot"],
                "parent_slot_token": python_receipt["parent_owner"]["slot_token"],
                "child_agent_id": python_receipt["child_owner"]["agent_id"],
                "child_slot": python_receipt["child_owner"]["slot"],
                "child_slot_token": python_receipt["child_owner"]["slot_token"],
                "attempt_ordinal": 0,
                "draw0_float64_bits": python_receipt["draw0_float64_bits"],
                "draw1_float64_bits": python_receipt["draw1_float64_bits"],
                "operation": "refine_descendant_parameter",
                "status": "rejected",
                "reason": reason,
                "selected_instance_id_index": -1,
                "selected_homology_id_index": -1,
                "changed_dimension": "none",
                "changed": False,
            }
        ],
    )

    assert case_id
    assert status == 0
    assert python_receipt["status"] == "rejected"
    assert python_receipt["reason"] == reason
    assert native_replay_receipt.canonical_sha256 == python_replay_receipt.canonical_sha256
    assert canonical_modular_recurrent_mutation_receipt_bytes_v1(
        native_replay_receipt
    ) == canonical_modular_recurrent_mutation_receipt_bytes_v1(python_replay_receipt)


def test_native_core_v3_disabled_receipt_bytes_match_canonical_fixture() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    child_before = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)

    status, native_receipt = _ordinary_birth_v3(
        library,
        parent,
        child,
        size,
        mutation_enabled=0,
    )
    native_replay_receipt = (
        build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
            _receipt_dict(native_receipt),
            profile_id="b2-16x48-v1",
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
            identity_table=(),
            native_module_instance_id_to_identity={},
            native_homology_id_to_identity={},
        )
    )
    expected_replay_receipt = build_modular_recurrent_mutation_receipt_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        transactions=[
            {
                "transaction_kind": "ordinary_birth",
                "transaction_ordinal": 0,
                "parent_agent_id": 100,
                "parent_slot": 2,
                "parent_slot_token": 12,
                "child_agent_id": (1 << 40) + 7,
                "child_slot": 4,
                "child_slot_token": 18,
                "attempt_ordinal": 0,
                "draw0_float64_bits": f"{_float64_bits(0.625):016x}",
                "draw1_float64_bits": f"{_float64_bits(0.5):016x}",
                "operation": "refine_descendant_parameter",
                "status": "disabled",
                "reason": "operator_disabled",
                "selected_instance_id_index": -1,
                "selected_homology_id_index": -1,
                "changed_dimension": "none",
                "changed": False,
            }
        ],
    )

    assert status == 0
    assert native_receipt.transaction_status == 6
    assert native_receipt.committed == 0
    assert native_receipt.proposal_digest == 0
    assert native_receipt.selected_slot == -1
    assert native_receipt.selected_module_instance_id == 0
    assert native_receipt.selected_homology_id == 0
    assert native_receipt.changed == 0
    assert native_receipt.changed_dimension == 0
    assert native_receipt.child_digest_before == child_before
    assert native_receipt.child_digest_after == child_before
    assert native_replay_receipt.canonical_sha256 == expected_replay_receipt.canonical_sha256
    assert canonical_modular_recurrent_mutation_receipt_bytes_v1(
        native_replay_receipt
    ) == canonical_modular_recurrent_mutation_receipt_bytes_v1(
        expected_replay_receipt
    )


@pytest.mark.parametrize(
    ("reason_code", "reason"),
    (
        (1, "operator_disabled"),
        (2, "structural_mutation_disabled"),
        (3, "recurrent_mutation_disabled"),
        (4, "operator_family_disabled"),
        (5, "zero_mutation_rate"),
        (6, "mutation_attempt_budget_zero"),
    ),
)
def test_native_core_v4_disabled_reason_receipt_bytes_match_canonical_fixture(
    reason_code: int,
    reason: str,
) -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    child_before = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)

    status, native_receipt = _ordinary_birth_v4(
        library,
        parent,
        child,
        size,
        disabled_reason_code=reason_code,
    )
    native_replay_receipt = (
        build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
            _receipt_dict(native_receipt),
            profile_id="b2-16x48-v1",
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
            identity_table=(),
            native_module_instance_id_to_identity={},
            native_homology_id_to_identity={},
        )
    )
    expected_replay_receipt = build_modular_recurrent_mutation_receipt_v1(
        profile_id="b2-16x48-v1",
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        transactions=[
            {
                "transaction_kind": "ordinary_birth",
                "transaction_ordinal": 0,
                "parent_agent_id": 100,
                "parent_slot": 2,
                "parent_slot_token": 12,
                "child_agent_id": (1 << 40) + 7,
                "child_slot": 4,
                "child_slot_token": 18,
                "attempt_ordinal": 0,
                "draw0_float64_bits": f"{_float64_bits(0.625):016x}",
                "draw1_float64_bits": f"{_float64_bits(0.5):016x}",
                "operation": "refine_descendant_parameter",
                "status": "disabled",
                "reason": reason,
                "selected_instance_id_index": -1,
                "selected_homology_id_index": -1,
                "changed_dimension": "none",
                "changed": False,
            }
        ],
    )

    assert library.eiel_modular_recurrent_core_ordinary_birth_receipt_layout_version_v4() == 4
    assert library.eiel_modular_recurrent_core_ordinary_birth_receipt_size_v4() == (
        ctypes.sizeof(OrdinaryBirthReceiptV4)
    )
    assert library.eiel_modular_recurrent_core_ordinary_birth_receipt_alignment_v4() == (
        ctypes.alignment(OrdinaryBirthReceiptV4)
    )
    assert OrdinaryBirthReceiptV4.disabled_reason_code.offset > (
        OrdinaryBirthReceiptV4.value_after_float32.offset
    )
    assert status == 0
    assert native_receipt.transaction_status == 6
    assert native_receipt.disabled_reason_code == reason_code
    assert native_receipt.committed == 0
    assert native_receipt.proposal_digest == 0
    assert native_receipt.selected_slot == -1
    assert native_receipt.selected_module_instance_id == 0
    assert native_receipt.selected_homology_id == 0
    assert native_receipt.changed == 0
    assert native_receipt.changed_dimension == 0
    assert native_receipt.child_digest_before == child_before
    assert native_receipt.child_digest_after == child_before
    assert canonical_modular_recurrent_mutation_receipt_bytes_v1(
        native_replay_receipt
    ) == canonical_modular_recurrent_mutation_receipt_bytes_v1(
        expected_replay_receipt
    )


def test_native_core_v4_disabled_reason_does_not_override_invalid_draw() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)

    status, native_receipt = _ordinary_birth_v4(
        library,
        parent,
        child,
        size,
        disabled_reason_code=5,
        draw0_bits=_float64_bits(1.0),
    )

    assert status == 0
    assert native_receipt.transaction_status == 4
    assert native_receipt.native_status == 7
    assert native_receipt.disabled_reason_code == 0
    assert native_receipt.proposal_digest == 0


def test_native_core_disabled_reason_canonical_semantics_are_frozen() -> None:
    assert DISABLED_REASON_CODE_ORDER == (1, 2, 3, 4, 5, 6)
    assert DISABLED_REASON_BY_CODE == {
        1: "operator_disabled",
        2: "structural_mutation_disabled",
        3: "recurrent_mutation_disabled",
        4: "operator_family_disabled",
        5: "zero_mutation_rate",
        6: "mutation_attempt_budget_zero",
    }
    assert DISABLED_REASON_CODE_BY_REASON == {
        reason: code for code, reason in DISABLED_REASON_BY_CODE.items()
    }
    assert tuple(row["condition"] for row in ORDINARY_BIRTH_TERMINAL_PRECEDENCE_V1) == (
        "malformed_or_missing_input",
        "invalid_draw_bits",
        "owner_or_slot_token_mismatch",
        "capacity_unavailable",
        "mutation_disabled",
        "proposal_generation",
        "publication_rejected",
    )


def test_native_core_v4_rejects_unknown_disabled_reason_code() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)

    status, native_receipt = _ordinary_birth_v4(
        library,
        parent,
        child,
        size,
        disabled_reason_code=7,
    )

    assert status == 1
    assert native_receipt.transaction_status == 0
    assert native_receipt.disabled_reason_code == 0


@pytest.mark.parametrize(
    ("case_id", "native_kwargs", "expected_transaction_status", "expected_reason"),
    (
        (
            "owner_mismatch_precedes_disabled_reason",
            {"parent_slot_token": 0},
            3,
            "stale_slot_token",
        ),
        (
            "capacity_rejection_precedes_disabled_reason",
            {"capacity_available": 0},
            1,
            "capacity_unavailable",
        ),
    ),
)
def test_native_core_v4_disabled_reason_precedence_for_rejections(
    case_id: str,
    native_kwargs: dict[str, int],
    expected_transaction_status: int,
    expected_reason: str,
) -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)

    status, native_receipt = _ordinary_birth_v4(
        library,
        parent,
        child,
        size,
        disabled_reason_code=5,
        **native_kwargs,
    )
    native_replay_receipt = (
        build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
            _receipt_dict(native_receipt),
            profile_id="b2-16x48-v1",
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
            identity_table=(),
            native_module_instance_id_to_identity={},
            native_homology_id_to_identity={},
        )
    )
    transaction = native_replay_receipt.transactions[0]

    assert case_id
    assert status == 0
    assert native_receipt.transaction_status == expected_transaction_status
    assert native_receipt.disabled_reason_code == 0
    assert native_receipt.proposal_digest == 0
    assert transaction.status == "rejected"
    assert transaction.reason == expected_reason


@pytest.mark.parametrize(
    ("disabled_reason_code", "expected_error"),
    (
        (7, "unknown native disabled reason code"),
        (-1, "disabled_reason_code must be a nonnegative int"),
        ("1", "disabled_reason_code must be a nonnegative int"),
    ),
)
def test_native_core_bridge_fails_closed_for_unknown_disabled_reason_code(
    disabled_reason_code: object,
    expected_error: str,
) -> None:
    native_receipt = {
        "transaction_status": 6,
        "native_status": 0,
        "operation": 2,
        "selected_slot": -1,
        "changed_dimension": 0,
        "changed": 0,
        "parent_agent_id": 100,
        "parent_slot": 2,
        "parent_slot_token": 12,
        "child_agent_id": (1 << 40) + 7,
        "child_slot": 4,
        "child_publication_token": 18,
        "draw0_float64_bits": _float64_bits(0.625),
        "draw1_float64_bits": _float64_bits(0.5),
        "disabled_reason_code": disabled_reason_code,
    }

    with pytest.raises(NativeCoreOrdinaryBirthBridgeError, match=expected_error):
        build_recurrent_receipt_from_native_core_ordinary_birth_v1(
            native_receipt,
            profile_id="b2-16x48-v1",
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
            selected_instance_id_index=-1,
            selected_homology_id_index=-1,
        )


def test_native_core_bridge_defaults_legacy_v3_disabled_reason_only_when_absent() -> None:
    native_receipt = {
        "transaction_status": 6,
        "native_status": 0,
        "operation": 2,
        "selected_slot": -1,
        "changed_dimension": 0,
        "changed": 0,
        "parent_agent_id": 100,
        "parent_slot": 2,
        "parent_slot_token": 12,
        "child_agent_id": (1 << 40) + 7,
        "child_slot": 4,
        "child_publication_token": 18,
        "draw0_float64_bits": _float64_bits(0.625),
        "draw1_float64_bits": _float64_bits(0.5),
    }

    receipt = build_recurrent_receipt_from_native_core_ordinary_birth_v1(
        native_receipt,
        profile_id="b2-16x48-v1",
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        selected_instance_id_index=-1,
        selected_homology_id_index=-1,
    )

    assert receipt.transactions[0].status == "disabled"
    assert receipt.transactions[0].reason == "operator_disabled"


@pytest.mark.parametrize(
    ("case_id", "owner_kwargs"),
    (
        (
            "child_publication_token_not_newer_than_expected",
            {"expected_child_slot_token": 18, "child_publication_token": 18},
        ),
        (
            "same_slot_publication_token_not_newer_than_parent",
            {
                "parent_slot": 4,
                "child_slot": 4,
                "parent_slot_token": 18,
                "expected_child_slot_token": 17,
                "child_publication_token": 18,
            },
        ),
        (
            "invalid_parent_slot_token",
            {"parent_slot_token": 0},
        ),
    ),
)
def test_native_core_v2_owner_rejection_receipt_bytes_match_canonical_fixture(
    case_id: str,
    owner_kwargs: dict[str, int],
) -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)
    child_before = library.eiel_modular_recurrent_core_canonical_digest_v1(child, size)

    status, native_receipt = _ordinary_birth_v2(
        library,
        parent,
        child,
        size,
        **owner_kwargs,
    )
    native_replay_receipt = (
        build_python_oracle_aligned_recurrent_receipt_from_mapped_native_core_ordinary_birth_v1(
            _receipt_dict(native_receipt),
            profile_id="b2-16x48-v1",
            source_packed_sha256="11" * 32,
            ancestry_tape_sha256="22" * 32,
            identity_table=(),
            native_module_instance_id_to_identity={},
            native_homology_id_to_identity={},
        )
    )
    expected_replay_receipt = _expected_rejected_replay_receipt(
        parent_slot=owner_kwargs.get("parent_slot", 2),
        child_slot=owner_kwargs.get("child_slot", 4),
        parent_slot_token=owner_kwargs.get("parent_slot_token", 12),
        child_publication_token=owner_kwargs.get("child_publication_token", 18),
        reason="stale_slot_token",
    )

    assert case_id
    assert status == 0
    assert native_receipt.transaction_status == 3
    assert native_receipt.committed == 0
    assert native_receipt.proposal_digest == 0
    assert native_receipt.selected_slot == -1
    assert native_receipt.selected_module_instance_id == 0
    assert native_receipt.selected_homology_id == 0
    assert native_receipt.changed == 0
    assert native_receipt.changed_dimension == 0
    assert native_receipt.child_digest_before == child_before
    assert native_receipt.child_digest_after == child_before
    assert native_replay_receipt.canonical_sha256 == expected_replay_receipt.canonical_sha256
    assert canonical_modular_recurrent_mutation_receipt_bytes_v1(
        native_replay_receipt
    ) == canonical_modular_recurrent_mutation_receipt_bytes_v1(
        expected_replay_receipt
    )


@pytest.mark.parametrize(
    (
        "case_id",
        "draw0",
        "draw1",
        "mutation_enabled",
        "disabled_reason_code",
        "native_kwargs",
    ),
    (
        ("accepted_toggle_dormancy", 0.125, 0.0, None, None, {}),
        ("accepted_prune_eligible_module", 0.375, 0.0, None, None, {}),
        ("accepted_refine_descendant_parameter", 0.625, 0.5, None, None, {}),
        (
            "accepted_refine_descendant_routing_or_expression",
            0.875,
            0.0,
            None,
            None,
            {},
        ),
        (
            "rejected_capacity_unavailable",
            0.625,
            0.5,
            None,
            None,
            {"capacity_available": 0},
        ),
        ("rejected_publication_failure", 0.625, 0.5, None, None, {"publish_success": 0}),
        ("disabled_operator_flag_off", 0.625, 0.5, 0, None, {}),
        ("v4_disabled_operator_disabled", 0.625, 0.5, None, 1, {}),
        ("v4_disabled_structural_mutation_disabled", 0.625, 0.5, None, 2, {}),
        ("v4_disabled_recurrent_mutation_disabled", 0.625, 0.5, None, 3, {}),
        ("v4_disabled_operator_family_disabled", 0.625, 0.5, None, 4, {}),
        ("v4_disabled_zero_mutation_rate", 0.625, 0.5, None, 5, {}),
        ("v4_disabled_mutation_attempt_budget_zero", 0.625, 0.5, None, 6, {}),
        (
            "owner_rejected_child_publication_token_not_newer_than_expected",
            0.625,
            0.5,
            None,
            None,
            {"expected_child_slot_token": 18, "child_publication_token": 18},
        ),
        (
            "owner_rejected_same_slot_publication_token_not_newer_than_parent",
            0.625,
            0.5,
            None,
            None,
            {
                "parent_slot": 4,
                "child_slot": 4,
                "parent_slot_token": 18,
                "expected_child_slot_token": 17,
                "child_publication_token": 18,
            },
        ),
        (
            "owner_rejected_invalid_parent_slot_token",
            0.625,
            0.5,
            None,
            None,
            {"parent_slot_token": 0},
        ),
    ),
)
def test_native_core_v2_receipt_byte_fixtures_are_deterministic_repeats(
    case_id: str,
    draw0: float,
    draw1: float,
    mutation_enabled: int | None,
    disabled_reason_code: int | None,
    native_kwargs: dict[str, int],
) -> None:
    first = _native_v2_replay_fingerprint(
        case_draw0=draw0,
        case_draw1=draw1,
        mutation_enabled=mutation_enabled,
        disabled_reason_code=disabled_reason_code,
        native_kwargs=native_kwargs,
    )
    second = _native_v2_replay_fingerprint(
        case_draw0=draw0,
        case_draw1=draw1,
        mutation_enabled=mutation_enabled,
        disabled_reason_code=disabled_reason_code,
        native_kwargs=native_kwargs,
    )

    assert case_id
    assert first == second


def test_native_core_ordinary_birth_rejection_maps_to_canonical_receipt() -> None:
    library = _library()
    _configure(library)
    parent, size = _fixture(library)
    child = _empty_envelope(library, size)

    status, native_receipt = _ordinary_birth(
        library,
        parent,
        child,
        size,
        capacity_available=0,
    )
    recurrent_receipt = build_recurrent_receipt_from_native_core_ordinary_birth_v1(
        _receipt_dict(native_receipt),
        profile_id="b2-12x32-v1",
        source_packed_sha256="11" * 32,
        ancestry_tape_sha256="22" * 32,
        selected_instance_id_index=0,
        selected_homology_id_index=1,
    )
    transaction = recurrent_receipt.transactions[0]

    assert status == 0
    assert transaction.status == "rejected"
    assert transaction.reason == "capacity_unavailable"
    assert transaction.changed_dimension == "none"
    assert transaction.changed is False
    assert transaction.selected_instance_id_index == -1
    assert transaction.selected_homology_id_index == -1


def test_native_core_c_api_rejects_wrong_buffer_extent() -> None:
    library = _library()
    _configure(library)
    size = int(library.eiel_modular_recurrent_core_fixed_envelope_size_v1())
    envelope = ctypes.create_string_buffer(size)

    assert library.eiel_modular_recurrent_core_initialize_empty_v1(
        envelope, size - 1, 0
    ) == 1
    assert library.eiel_modular_recurrent_core_validate_v1(envelope, size - 1) == 1
    assert library.eiel_modular_recurrent_core_canonical_digest_v1(
        envelope, size - 1
    ) == 0


def test_native_core_c_api_is_not_production_receipt_or_cuda_surface() -> None:
    library = _library()
    _configure(library)
    exported = subprocess.run(
        ["nm", "-D", "--defined-only", str(BUILD / "libeiel_modular_recurrent_core_c_api_v1.so")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert "eiel_modular_recurrent_core_mutate_two_draws_v1" in exported
    assert "eiel_modular_recurrent_core_ordinary_birth_two_draws_v1" in exported
    assert "canonical_sha256" not in exported
    assert "cuda" not in exported.lower()
    assert "modular_two_draws_v1" not in exported
    assert sys.platform.startswith("linux")
