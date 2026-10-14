"""P1.2 Python/reference ordinary-birth mutation oracles.

These oracles are reference-only. They do not promote native CPU or CUDA
structural/recurrent mutation and they do not publish production children.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Final

from evolution.structural_sustainable_growth_v1 import mutate_sustainable_growth
from engine.perf.recurrent_native.modular_recurrent_mutation_receipt_v1 import (
    build_modular_recurrent_mutation_receipt_v1,
    float64_to_bits,
)
from engine.perf.recurrent_native.modular_recurrent_mutation_replay_v1 import (
    replay_modular_recurrent_mutation_v1,
)

from .p12_stage2_contracts_v1 import (
    RECURRENT_ORACLE_CONTRACT_ID,
    STRUCTURAL_ORACLE_CONTRACT_ID,
)


SCHEMA_VERSION: Final[str] = "p12_stage2_python_reference_mutation_oracle_v1"
RNG_DRAW_BUDGET: Final[int] = 2
SUPPORTED_PROFILE_ID: Final[str] = "b2-16x48-v1"


class P12MutationOracleError(RuntimeError):
    """A P1.2 reference mutation oracle contract failed closed."""


@dataclass(frozen=True, slots=True)
class OrdinaryBirthOwnerV1:
    agent_id: int
    slot: int
    slot_token: int

    def __post_init__(self) -> None:
        if self.agent_id < 0:
            raise P12MutationOracleError("agent_id must be nonnegative")
        if self.slot < 0:
            raise P12MutationOracleError("slot must be nonnegative")
        if self.slot_token <= 0:
            raise P12MutationOracleError("slot_token must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "agent_id": int(self.agent_id),
            "slot": int(self.slot),
            "slot_token": int(self.slot_token),
        }


def canonical_oracle_bytes_v1(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def oracle_sha256_v1(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_oracle_bytes_v1(value)).hexdigest()


def _payload_copy(parent_payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(parent_payload, Mapping):
        raise P12MutationOracleError("parent_payload must be a mapping")
    return deepcopy(dict(parent_payload))


def _unit_draw(value: float, *, field: str) -> float:
    if type(value) is not float or not 0.0 <= value < 1.0:
        raise P12MutationOracleError(f"{field} must be a Python float in [0,1)")
    return float(value)


def _base_receipt(
    *,
    contract_id: str,
    parent_payload: Mapping[str, Any],
    child_payload: Mapping[str, Any],
    parent_owner: OrdinaryBirthOwnerV1,
    child_owner: OrdinaryBirthOwnerV1,
    draw0: float,
    draw1: float,
    operation: str,
    status: str,
    reason: str,
    changed_dimension: str,
    changed: bool,
    committed: bool,
    selected_module_instance_id: str = "",
    selected_homology_id: str = "",
) -> dict[str, Any]:
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": contract_id,
        "profile_id": SUPPORTED_PROFILE_ID,
        "transaction_kind": "ordinary_birth",
        "parent_owner": parent_owner.to_dict(),
        "child_owner": child_owner.to_dict(),
        "rng_draw_budget": RNG_DRAW_BUDGET,
        "draw0_float64_bits": float64_to_bits(draw0),
        "draw1_float64_bits": float64_to_bits(draw1),
        "operation": operation,
        "status": status,
        "reason": reason,
        "changed_dimension": changed_dimension,
        "changed": bool(changed),
        "selected_module_instance_id": selected_module_instance_id,
        "selected_homology_id": selected_homology_id,
        "committed": bool(committed),
        "pre_parent_payload_sha256": oracle_sha256_v1(parent_payload),
        "child_payload_sha256": oracle_sha256_v1(child_payload),
        "parent_preserved": True,
        "native_cuda_promotion_authorized": False,
    }
    receipt["oracle_sha256"] = oracle_sha256_v1(receipt)
    return receipt


def structural_ordinary_birth_oracle_v1(
    parent_payload: Mapping[str, Any],
    *,
    parent_owner: OrdinaryBirthOwnerV1,
    child_owner: OrdinaryBirthOwnerV1,
    draw0: float,
    draw1: float,
    mutation_enabled: bool = True,
    capacity_available: bool = True,
    publish_success: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the structural ordinary-birth reference oracle."""

    draw0 = _unit_draw(draw0, field="draw0")
    draw1 = _unit_draw(draw1, field="draw1")
    before = _payload_copy(parent_payload)
    if child_owner.slot_token <= parent_owner.slot_token and child_owner.slot == parent_owner.slot:
        raise P12MutationOracleError("child slot reuse requires a newer slot token")
    if not capacity_available:
        child = _payload_copy(parent_payload)
        receipt = _base_receipt(
            contract_id=STRUCTURAL_ORACLE_CONTRACT_ID,
            parent_payload=before,
            child_payload=child,
            parent_owner=parent_owner,
            child_owner=child_owner,
            draw0=draw0,
            draw1=draw1,
            operation="none",
            status="rejected",
            reason="capacity_unavailable",
            changed_dimension="none",
            changed=False,
            committed=False,
        )
        return child, receipt

    result = mutate_sustainable_growth(before, draw0, draw1, enabled=mutation_enabled)
    if before != parent_payload:
        raise P12MutationOracleError("oracle mutated parent payload in place")
    child = deepcopy(result.payload)
    telemetry = dict(result.telemetry)
    status = str(telemetry.get("status", "rejected"))
    committed = bool(publish_success and status != "rejected")
    if not publish_success:
        child = _payload_copy(parent_payload)
        status = "rejected"
        reason = "child_publication_failed"
        operation = "none"
        changed_dimension = "none"
        changed = False
    else:
        operation = str(telemetry.get("operation", "unknown"))
        reason = str(telemetry.get("reason", "unknown"))
        changed_dimension = str(telemetry.get("selected_field") or "none")
        changed = not bool(telemetry.get("parent_preserved", True))
        selected_module_instance_id = str(telemetry.get("selected_module_instance_id") or "")
        selected_homology_id = str(telemetry.get("selected_homology_id") or "")
    receipt = _base_receipt(
        contract_id=STRUCTURAL_ORACLE_CONTRACT_ID,
        parent_payload=before,
        child_payload=child,
        parent_owner=parent_owner,
        child_owner=child_owner,
        draw0=draw0,
        draw1=draw1,
        operation=operation,
        status=status,
        reason=reason,
        changed_dimension=changed_dimension,
        changed=changed,
        committed=committed,
        selected_module_instance_id=selected_module_instance_id if publish_success else "",
        selected_homology_id=selected_homology_id if publish_success else "",
    )
    return child, receipt


def _identity_table(parent_payload: Mapping[str, Any]) -> tuple[str, ...]:
    modules = parent_payload.get("modules", ())
    if not isinstance(modules, Sequence) or isinstance(modules, (str, bytes)):
        raise P12MutationOracleError("parent modules must be a sequence")
    values: set[str] = set()
    for module in modules:
        if not isinstance(module, Mapping):
            raise P12MutationOracleError("parent module must be a mapping")
        for field in ("module_instance_id", "homology_id"):
            value = module.get(field)
            if type(value) is not str or not value:
                raise P12MutationOracleError(f"parent module has invalid {field}")
            values.add(value)
    return tuple(sorted(values))


def recurrent_ordinary_birth_oracle_v1(
    parent_payload: Mapping[str, Any],
    *,
    parent_owner: OrdinaryBirthOwnerV1,
    child_owner: OrdinaryBirthOwnerV1,
    draw0: float,
    draw1: float,
    mutation_enabled: bool = True,
    capacity_available: bool = True,
    publish_success: bool = True,
    source_packed_sha256: str = "11" * 32,
    ancestry_tape_sha256: str = "22" * 32,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the recurrent ordinary-birth reference oracle through receipt replay."""

    structural_child, structural_receipt = structural_ordinary_birth_oracle_v1(
        parent_payload,
        parent_owner=parent_owner,
        child_owner=child_owner,
        draw0=draw0,
        draw1=draw1,
        mutation_enabled=mutation_enabled,
        capacity_available=capacity_available,
        publish_success=publish_success,
    )
    if not bool(structural_receipt["committed"]):
        recurrent_receipt = {
            **structural_receipt,
            "contract_id": RECURRENT_ORACLE_CONTRACT_ID,
            "replay_receipt_sha256": "",
        }
        recurrent_receipt["oracle_sha256"] = oracle_sha256_v1(
            {
                key: value
                for key, value in recurrent_receipt.items()
                if key != "oracle_sha256"
            }
        )
        return structural_child, recurrent_receipt

    table = _identity_table(parent_payload)
    selected_instance = -1
    selected_homology = -1
    if structural_receipt["changed"] and structural_receipt["status"] == "accepted":
        selected_instance = table.index(str(structural_receipt["selected_module_instance_id"]))
        selected_homology = table.index(str(structural_receipt["selected_homology_id"]))
    replay_receipt = build_modular_recurrent_mutation_receipt_v1(
        profile_id=SUPPORTED_PROFILE_ID,
        source_packed_sha256=source_packed_sha256,
        ancestry_tape_sha256=ancestry_tape_sha256,
        transactions=[
            {
                "transaction_kind": "ordinary_birth",
                "transaction_ordinal": 0,
                "parent_agent_id": parent_owner.agent_id,
                "parent_slot": parent_owner.slot,
                "parent_slot_token": parent_owner.slot_token,
                "child_agent_id": child_owner.agent_id,
                "child_slot": child_owner.slot,
                "child_slot_token": child_owner.slot_token,
                "attempt_ordinal": 0,
                "draw0_float64_bits": structural_receipt["draw0_float64_bits"],
                "draw1_float64_bits": structural_receipt["draw1_float64_bits"],
                "operation": structural_receipt["operation"],
                "status": structural_receipt["status"],
                "reason": structural_receipt["reason"],
                "selected_instance_id_index": selected_instance,
                "selected_homology_id_index": selected_homology,
                "changed_dimension": structural_receipt["changed_dimension"],
                "changed": structural_receipt["changed"],
            }
        ],
    )
    replayed = replay_modular_recurrent_mutation_v1(
        parent_payload,
        replay_receipt,
        identity_table=table,
    )
    if replayed != structural_child:
        raise P12MutationOracleError("recurrent replay drifted from structural oracle")
    recurrent_receipt = {
        **structural_receipt,
        "contract_id": RECURRENT_ORACLE_CONTRACT_ID,
        "identity_table_sha256": oracle_sha256_v1({"identity_table": table}),
        "replay_receipt_sha256": replay_receipt.canonical_sha256,
    }
    recurrent_receipt["oracle_sha256"] = oracle_sha256_v1(
        {key: value for key, value in recurrent_receipt.items() if key != "oracle_sha256"}
    )
    return replayed, recurrent_receipt


__all__ = [
    "OrdinaryBirthOwnerV1",
    "P12MutationOracleError",
    "canonical_oracle_bytes_v1",
    "oracle_sha256_v1",
    "recurrent_ordinary_birth_oracle_v1",
    "structural_ordinary_birth_oracle_v1",
]
