"""Science Request integration for exact natural source-state references."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from emergence_contracts import (
    ExactNaturalSourceStateError,
    authenticate_exact_natural_source_state_ref_v1,
    load_exact_natural_source_state_bank_v1,
    materialization_profile_id_v1,
    validate_exact_natural_source_state_ref_v1,
)


EXACT_NATURAL_SOURCE_STATE_CAPABILITY_V1 = (
    "exact_natural_source_state_materialization_v1"
)


class ExactNaturalSourceStateMaterializationError(ValueError):
    """An exact source-state request cannot be resolved fail-closed."""


def _digest_hex(value: object, *, field: str) -> str:
    if isinstance(value, Mapping):
        if set(value) != {"algorithm", "hex"} or value.get("algorithm") != "sha256":
            raise ExactNaturalSourceStateMaterializationError(
                f"{field} is not a sha256 digest"
            )
        value = value.get("hex")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExactNaturalSourceStateMaterializationError(
            f"{field} is not a lowercase sha256 digest"
        )
    return value


def _source_bank_pointer(value: object) -> tuple[Path, str | None, str | None]:
    if not isinstance(value, Mapping):
        raise ExactNaturalSourceStateMaterializationError(
            "exact_natural_source_state_bank must be an object"
        )
    required = {"path", "file_sha256", "bank_sha256"}
    if set(value) != required or not isinstance(value.get("path"), str):
        raise ExactNaturalSourceStateMaterializationError(
            "exact_natural_source_state_bank pointer fields are not exact"
        )
    return (
        Path(value["path"]).expanduser(),
        _digest_hex(value["file_sha256"], field="exact_natural_source_state_bank.file_sha256"),
        _digest_hex(value["bank_sha256"], field="exact_natural_source_state_bank.bank_sha256"),
    )


def load_and_authenticate_source_state_bank_v1(
    pointer: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a bank and authenticate its file and each referenced source."""

    path, expected_file_sha256, expected_bank_sha256 = _source_bank_pointer(pointer)
    try:
        bank, file_sha256 = load_exact_natural_source_state_bank_v1(
            path, expected_file_sha256=expected_file_sha256
        )
    except (ExactNaturalSourceStateError, OSError, ValueError) as error:
        raise ExactNaturalSourceStateMaterializationError(str(error)) from error
    if bank["bank_sha256"]["hex"] != expected_bank_sha256:
        raise ExactNaturalSourceStateMaterializationError(
            "exact source-state bank content digest differs"
        )
    try:
        authenticated = [
            authenticate_exact_natural_source_state_ref_v1(ref)
            for ref in bank["refs"]
        ]
    except (ExactNaturalSourceStateError, OSError, ValueError) as error:
        raise ExactNaturalSourceStateMaterializationError(
            f"exact source-state source authentication failed: {error}"
        ) from error
    authenticated_bank = {**bank, "refs": authenticated}
    binding = {
        "schema_version": "eiel_exact_natural_source_state_bank_binding_v1",
        "bank_id": bank["bank_id"],
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "bank_sha256": bank["bank_sha256"],
        "authenticated_ref_count": len(authenticated),
    }
    return authenticated_bank, binding


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ExactNaturalSourceStateMaterializationError(
            f"{field} must be an integer >= {minimum}"
        )
    return int(value)


def materialize_exact_natural_source_initialization_v1(
    initialization: Mapping[str, Any],
    *,
    bank: Mapping[str, Any],
    bank_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve explicit source refs/roles to exact inline natural states."""

    value = deepcopy(dict(initialization))
    rows = value.get("exact_natural_source_composition")
    if not isinstance(rows, list) or not rows:
        raise ExactNaturalSourceStateMaterializationError(
            "exact natural source composition is required"
        )
    live = _integer(value.get("initial_live_count"), field="initial_live_count", minimum=1)
    resolved_rows: list[dict[str, Any]] = []
    resolved_refs_by_id: dict[str, dict[str, Any]] = {}
    materialization_ids: list[str] = []
    total = 0
    # ``load_and_authenticate_source_state_bank_v1`` has already validated
    # every reference and authenticated every source artifact.  Resolution
    # below must therefore be an in-memory lookup; rehashing historical
    # capture files once per expanded world would make a valid 1024-world
    # request needlessly reread the same source bytes thousands of times.
    refs_by_id = {
        ref["ref_id"]: ref
        for ref in bank.get("refs", [])
        if isinstance(ref, Mapping) and isinstance(ref.get("ref_id"), str)
    }
    panels = bank.get("panels", {})
    if (
        not isinstance(bank_binding, Mapping)
        or bank_binding.get("schema_version")
        != "eiel_exact_natural_source_state_bank_binding_v1"
        or bank_binding.get("bank_id") != bank.get("bank_id")
        or bank_binding.get("authenticated_ref_count") != len(refs_by_id)
        or not isinstance(panels, Mapping)
    ):
        raise ExactNaturalSourceStateMaterializationError(
            "exact source-state bank is not an authenticated materialization input"
        )

    def authenticated_ref(source_ref: str, role: str | None) -> dict[str, Any]:
        candidate = source_ref
        if candidate in panels:
            panel = panels[candidate]
            if (
                role is None
                or not isinstance(panel, Mapping)
                or role not in panel
            ):
                raise ExactNaturalSourceStateMaterializationError(
                    "source-state panel requires an explicit existing role"
                )
            candidate = panel[role]
        elif role is not None:
            raise ExactNaturalSourceStateMaterializationError(
                "a direct exact source reference cannot carry a role selector"
            )
        ref = refs_by_id.get(candidate)
        if ref is None:
            raise ExactNaturalSourceStateMaterializationError(
                "source-state reference is absent from its authenticated bank"
            )
        return dict(ref)

    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise ExactNaturalSourceStateMaterializationError(
                f"exact_natural_source_composition[{index}] must be an object"
            )
        keys = set(raw_row)
        if keys not in ({"source_state_ref", "count"}, {"source_state_ref", "source_state_role", "count"}):
            raise ExactNaturalSourceStateMaterializationError(
                f"exact_natural_source_composition[{index}] fields are not exact"
            )
        source_state_ref = raw_row.get("source_state_ref")
        if not isinstance(source_state_ref, str) or not source_state_ref:
            raise ExactNaturalSourceStateMaterializationError(
                f"exact_natural_source_composition[{index}].source_state_ref is invalid"
            )
        role = raw_row.get("source_state_role")
        if role is not None and (not isinstance(role, str) or not role):
            raise ExactNaturalSourceStateMaterializationError(
                f"exact_natural_source_composition[{index}].source_state_role is invalid"
            )
        count = _integer(
            raw_row.get("count"),
            field=f"exact_natural_source_composition[{index}].count",
        )
        try:
            ref = authenticated_ref(source_state_ref, role)
            # Keep a cheap structural check at the materialization boundary;
            # source bytes were authenticated once when the bank was loaded.
            ref = validate_exact_natural_source_state_ref_v1(ref)
        except (ExactNaturalSourceStateError, OSError, ValueError) as error:
            raise ExactNaturalSourceStateMaterializationError(
                f"exact source-state row {index} cannot be authenticated: {error}"
            ) from error
        ref_id = ref["ref_id"]
        materialization_id = materialization_profile_id_v1(ref["genome_state_id"])
        prior = resolved_refs_by_id.get(ref_id)
        if prior is None:
            resolved_refs_by_id[ref_id] = ref
            materialization_ids.append(materialization_id)
        elif prior != ref:
            raise ExactNaturalSourceStateMaterializationError(
                "one exact source reference resolved to conflicting content"
            )
        resolved_row = {
            "source_state_ref": ref_id,
            "materialization_profile_id": materialization_id,
            "count": count,
        }
        if role is not None:
            resolved_row["source_state_role"] = role
        resolved_rows.append(resolved_row)
        total += count
    if total != live:
        raise ExactNaturalSourceStateMaterializationError(
            "exact natural source composition does not equal initial_live_count"
        )
    if len(materialization_ids) != len(set(materialization_ids)):
        raise ExactNaturalSourceStateMaterializationError(
            "exact natural source references have colliding materialization IDs"
        )
    materialized = {
        **value,
        "founder_profiles": materialization_ids,
        "exact_founder_composition": [
            {
                "founder_profile_id": row["materialization_profile_id"],
                "count": row["count"],
            }
            for row in resolved_rows
        ],
        "exact_natural_source_composition": resolved_rows,
        "exact_natural_source_states": [
            resolved_refs_by_id[ref_id] for ref_id in resolved_refs_by_id
        ],
        "exact_natural_source_state_bank_binding": dict(bank_binding),
    }
    if len(materialized["exact_founder_composition"]) != len(materialization_ids):
        raise ExactNaturalSourceStateMaterializationError(
            "exact natural source composition contains duplicate materialization rows"
        )
    return materialized


def validate_materialized_exact_natural_source_initialization_v1(
    initialization: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Validate a self-contained resolved exact-source initialization."""

    rows = initialization.get("exact_natural_source_composition")
    states = initialization.get("exact_natural_source_states")
    profiles = initialization.get("founder_profiles")
    if not isinstance(rows, list) or not rows or not isinstance(states, list) or not states or not isinstance(profiles, list):
        raise ExactNaturalSourceStateMaterializationError(
            "materialized exact source initialization is incomplete"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for state in states:
        ref = validate_exact_natural_source_state_ref_v1(state)
        if ref["ref_id"] in by_id:
            raise ExactNaturalSourceStateMaterializationError(
                "materialized exact source states contain duplicate refs"
            )
        by_id[ref["ref_id"]] = ref
    seen_profiles: list[str] = []
    resolved: list[dict[str, Any]] = []
    total = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) not in (
            {
                "source_state_ref",
                "materialization_profile_id",
                "count",
            },
            {
                "source_state_ref",
                "source_state_role",
                "materialization_profile_id",
                "count",
            },
        ):
            raise ExactNaturalSourceStateMaterializationError(
                f"materialized source composition row {index} is invalid"
            )
        ref_id = row["source_state_ref"]
        if ref_id not in by_id:
            raise ExactNaturalSourceStateMaterializationError(
                "materialized composition names an absent source ref"
            )
        ref = by_id[ref_id]
        profile_id = row["materialization_profile_id"]
        if profile_id != materialization_profile_id_v1(ref["genome_state_id"]):
            raise ExactNaturalSourceStateMaterializationError(
                "materialization profile identity differs from source ref"
            )
        count = _integer(row["count"], field=f"source composition {index}.count")
        seen_profiles.append(profile_id)
        total += count
        resolved.append(dict(row))
    if seen_profiles != list(profiles):
        raise ExactNaturalSourceStateMaterializationError(
            "founder profiles do not exactly match materialized source composition"
        )
    if total != _integer(initialization.get("initial_live_count"), field="initial_live_count", minimum=1):
        raise ExactNaturalSourceStateMaterializationError(
            "materialized source composition count differs from live count"
        )
    return tuple(by_id.values())


__all__ = (
    "EXACT_NATURAL_SOURCE_STATE_CAPABILITY_V1",
    "ExactNaturalSourceStateMaterializationError",
    "load_and_authenticate_source_state_bank_v1",
    "materialize_exact_natural_source_initialization_v1",
    "validate_materialized_exact_natural_source_initialization_v1",
)
