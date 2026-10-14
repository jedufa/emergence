from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from authoritative_state.identity import canonical_sha256
from authoritative_state.production_identity import (
    validate_admission_identity_v1,
    validate_identity_against_local_schema_v1,
)
from authoritative_state.schema import AuthoritativeStateSchemaError
from authoritative_state.schema import load_schema_v1


def _identity() -> dict:
    schema = load_schema_v1()
    contract = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "_spec/authoritative_state/v1/abi_variants.json"
        ).read_text(encoding="utf-8")
    )
    variants = sorted(
        (
            {
                "entrypoint": row["variant_id"],
                "abi_version": row["variant_id"],
                "argument_count": row["count"],
                "argument_order_sha256": row["order_sha256"],
            }
            for row in contract["variants"]
            if row["count"] > 0
        ),
        key=lambda row: row["entrypoint"],
    )
    return {
        "contract_schema_version": "eiel_authoritative_state_binary_identity_v1",
        "state_schema_version": "eiel_authoritative_state_v1",
        "state_schema_sha256": schema.schema_hash,
        "source_commit": schema.root["source_commit"],
        "source_inventory_sha256": schema.root["component_hashes"][
            "field_registry.json"
        ],
        "build_identity_sha256": "7b762d542f61f5c6224021329b544e802eaa73d9a6cb4cd25e1078d7397d3277",
        "abi_set_sha256": canonical_sha256(variants),
        "abi_variants": variants,
        "checkpoint_schema_version": "checkpoint_bundle_v2",
        "checkpoint_schema_sha256": schema.root["component_hashes"][
            "checkpoint_schema.json"
        ],
        "chronicle_schema_version": "eiel_authoritative_state_chronicle_schema_v1",
        "chronicle_schema_sha256": schema.root["component_hashes"][
            "chronicle_schema.json"
        ],
        "provider_capability_identity": "eiel-production-provider-v1",
        "overlay_version": "inheritance-cross-path-overlay-v1",
        "supported_cache_widths": [1, 2],
        "supported_profiles": ["eiel-production-runtime-v1"],
        "build_toolchain_identity": "clang-or-nvcc",
        "compile_flags": "extension-build-defaults",
        "artifact_build_id": (
            f"state-schema-v1-stackless-sm75-{schema.schema_hash[:12]}"
        ),
    }


def test_all_admission_surfaces_must_be_byte_equal() -> None:
    identity = _identity()
    digest = validate_admission_identity_v1(
        request=identity,
        provider=identity,
        manifest=identity,
        binary=identity,
        checkpoint=identity,
    )
    assert digest == canonical_sha256(identity)
    validate_identity_against_local_schema_v1(
        identity,
        schema=load_schema_v1(),
        provider_id="eiel-production-provider-v1",
        profile_id="eiel-production-runtime-v1",
    )


def test_unknown_provider_fails_local_schema_admission() -> None:
    schema = load_schema_v1()
    identity = _identity()
    profile_id = "eiel-production-runtime-v1"
    provider_id = "inheritance-cross-path-gpu-provider-v1"
    candidate = copy.deepcopy(identity)
    candidate["provider_capability_identity"] = provider_id
    with pytest.raises(
        AuthoritativeStateSchemaError,
        match="EIEL_ADMISSION_SURFACE_UNSUPPORTED",
    ):
        validate_identity_against_local_schema_v1(
            candidate,
            schema=schema,
            provider_id=provider_id,
            profile_id=profile_id,
        )


@pytest.mark.parametrize(
    "field",
    [
        "state_schema_sha256",
        "source_commit",
        "abi_set_sha256",
        "checkpoint_schema_sha256",
        "chronicle_schema_sha256",
        "overlay_version",
        "supported_cache_widths",
        "artifact_build_id",
    ],
)
def test_stale_or_incompatible_surface_fails_closed(field: str) -> None:
    identity = _identity()
    stale = copy.deepcopy(identity)
    stale[field] = (
        list(reversed(stale[field]))
        if isinstance(stale[field], list)
        else "0" * len(stale[field])
    )
    with pytest.raises(AuthoritativeStateSchemaError):
        validate_admission_identity_v1(
            request=identity,
            provider=identity,
            manifest=stale,
            binary=identity,
        )


def test_same_abi_count_with_reordered_identity_fails_closed() -> None:
    identity = _identity()
    binary = copy.deepcopy(identity)
    binary["abi_variants"][0], binary["abi_variants"][1] = (
        binary["abi_variants"][1],
        binary["abi_variants"][0],
    )
    with pytest.raises(AuthoritativeStateSchemaError) as error:
        validate_admission_identity_v1(
            request=identity,
            provider=identity,
            manifest=identity,
            binary=binary,
        )
    assert error.value.code == "EIEL_ADMISSION_ABI_ORDER"
