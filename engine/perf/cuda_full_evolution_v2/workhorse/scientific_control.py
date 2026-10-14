"""Versioned branch-local scientific controls for Workhorse V2.

This is an internal runtime contract folded into selected-world checkpoints and
indexed-continuation receipts.  It is deliberately limited to the two V1
Expedition controls that the native kernel actually implements.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Final

from .contract import WORKHORSE_TIME_AUTHORITY_V2, canonical_sha256


WORKHORSE_SCIENTIFIC_CONTROL_CONTRACT_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_scientific_control_contract_v1"
)
WORKHORSE_ACTIVE_SCIENTIFIC_CONTROL_STATE_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_active_scientific_control_state_v1"
)
WORKHORSE_SCIENTIFIC_CONTROL_APPLICATION_SCHEMA_V1: Final[str] = (
    "eiel_workhorse_scientific_control_application_v1"
)
WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_ABI_V1: Final[str] = (
    "eiel_workhorse_scientific_control_native_abi_v1"
)
WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_EXPORTS_V1: Final[tuple[str, ...]] = (
    "apply_science_controls_v1",
    "science_control_abi_version_v1",
    "science_controls_readback_v1",
)
WORKHORSE_SCIENTIFIC_CONTROL_IDS_V1: Final[frozenset[str]] = frozenset(
    {"environment.resource.multiplier", "mutation.rate"}
)


class WorkhorseScientificControlError(ValueError):
    """A scientific control cannot be represented by the closed native ABI."""


def _native_mapping_v1(
    *, resource_multiplier: float, mutation_rate: float
) -> dict[str, Any]:
    if (
        type(resource_multiplier) not in {int, float}
        or isinstance(resource_multiplier, bool)
        or not math.isfinite(float(resource_multiplier))
        or not 0.0 <= float(resource_multiplier) <= 16.0
    ):
        raise WorkhorseScientificControlError(
            "resource multiplier must be finite in [0,16]"
        )
    if (
        type(mutation_rate) not in {int, float}
        or isinstance(mutation_rate, bool)
        or not math.isfinite(float(mutation_rate))
        or not 0.0 <= float(mutation_rate) <= 1.0
    ):
        raise WorkhorseScientificControlError(
            "mutation rate must be finite in [0,1]"
        )
    return {
        "schema_version": "eiel_workhorse_scientific_control_native_mapping_v1",
        "ordered_fields": [
            "mutation_threshold_u16",
            "resource_multiplier_q16",
        ],
        "ordered_values": [
            int(round(float(mutation_rate) * 65536.0)),
            int(round(float(resource_multiplier) * 65536.0)),
        ],
        "default_preservation": {
            "mutation_threshold_u16": 8192,
            "resource_multiplier_q16": 65536,
            "default_control_path_is_byte_identical": True,
        },
        "runtime_abi": {
            "abi_id": WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_ABI_V1,
            "required_exports": list(WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_EXPORTS_V1),
            "application_rule": "FRESH_RESTORED_OWNER_AT_BOUND_BRANCH_TICK_ONCE",
        },
    }


def workhorse_scientific_control_contract_v1(
    *,
    effective_next_absolute_tick: int,
    resource_multiplier: float,
    mutation_rate: float,
) -> dict[str, Any]:
    """Seal the two V1 branch-local controls and their exact native mapping."""

    if type(effective_next_absolute_tick) is not int or effective_next_absolute_tick < 0:
        raise WorkhorseScientificControlError(
            "scientific controls require a nonnegative absolute branch tick"
        )
    controls = {
        "environment.resource.multiplier": float(resource_multiplier),
        "mutation.rate": float(mutation_rate),
    }
    body = {
        "schema_version": WORKHORSE_SCIENTIFIC_CONTROL_CONTRACT_SCHEMA_V1,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "effective_next_absolute_tick": effective_next_absolute_tick,
        "controls": controls,
        "native_mapping": _native_mapping_v1(
            resource_multiplier=resource_multiplier,
            mutation_rate=mutation_rate,
        ),
        "unsupported_control_policy": "FAIL_CLOSED",
    }
    return {**body, "control_contract_sha256": canonical_sha256(body)}


def validate_workhorse_scientific_control_contract_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "time_authority",
        "effective_next_absolute_tick",
        "controls",
        "native_mapping",
        "unsupported_control_policy",
        "control_contract_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise WorkhorseScientificControlError(
            "scientific control contract fields are inexact"
        )
    controls = value.get("controls")
    if type(controls) is not dict or set(controls) != WORKHORSE_SCIENTIFIC_CONTROL_IDS_V1:
        raise WorkhorseScientificControlError(
            "scientific control contract contains unsupported controls"
        )
    expected = workhorse_scientific_control_contract_v1(
        effective_next_absolute_tick=value.get("effective_next_absolute_tick"),
        resource_multiplier=controls["environment.resource.multiplier"],
        mutation_rate=controls["mutation.rate"],
    )
    if dict(value) != expected:
        raise WorkhorseScientificControlError(
            "scientific control contract does not recompute"
        )
    return expected


def workhorse_active_scientific_control_state_v1(
    *,
    captured_next_absolute_tick: int,
    resource_multiplier: float,
    mutation_rate: float,
) -> dict[str, Any]:
    """Seal the active values omitted from canonical Workhorse state rows."""

    if type(captured_next_absolute_tick) is not int or captured_next_absolute_tick < 0:
        raise WorkhorseScientificControlError(
            "active controls require a nonnegative captured absolute tick"
        )
    controls = {
        "environment.resource.multiplier": float(resource_multiplier),
        "mutation.rate": float(mutation_rate),
    }
    body = {
        "schema_version": WORKHORSE_ACTIVE_SCIENTIFIC_CONTROL_STATE_SCHEMA_V1,
        "time_authority": WORKHORSE_TIME_AUTHORITY_V2,
        "captured_next_absolute_tick": captured_next_absolute_tick,
        "controls": controls,
        "native_mapping": _native_mapping_v1(
            resource_multiplier=resource_multiplier,
            mutation_rate=mutation_rate,
        ),
    }
    return {**body, "active_control_state_sha256": canonical_sha256(body)}


def validate_workhorse_active_scientific_control_state_v1(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "time_authority",
        "captured_next_absolute_tick",
        "controls",
        "native_mapping",
        "active_control_state_sha256",
    }:
        raise WorkhorseScientificControlError(
            "active scientific control state fields are inexact"
        )
    controls = value.get("controls")
    if type(controls) is not dict or set(controls) != WORKHORSE_SCIENTIFIC_CONTROL_IDS_V1:
        raise WorkhorseScientificControlError(
            "active scientific control state contains unsupported controls"
        )
    expected = workhorse_active_scientific_control_state_v1(
        captured_next_absolute_tick=value.get("captured_next_absolute_tick"),
        resource_multiplier=controls["environment.resource.multiplier"],
        mutation_rate=controls["mutation.rate"],
    )
    if dict(value) != expected:
        raise WorkhorseScientificControlError(
            "active scientific control state does not recompute"
        )
    return expected


__all__ = (
    "WORKHORSE_ACTIVE_SCIENTIFIC_CONTROL_STATE_SCHEMA_V1",
    "WORKHORSE_SCIENTIFIC_CONTROL_APPLICATION_SCHEMA_V1",
    "WORKHORSE_SCIENTIFIC_CONTROL_CONTRACT_SCHEMA_V1",
    "WORKHORSE_SCIENTIFIC_CONTROL_IDS_V1",
    "WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_ABI_V1",
    "WORKHORSE_SCIENTIFIC_CONTROL_NATIVE_EXPORTS_V1",
    "WorkhorseScientificControlError",
    "validate_workhorse_active_scientific_control_state_v1",
    "validate_workhorse_scientific_control_contract_v1",
    "workhorse_active_scientific_control_state_v1",
    "workhorse_scientific_control_contract_v1",
)
