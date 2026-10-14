"""CPU-only contract tests for the direct-resident field manifest."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from engine.campaigns.acs_direct_resident_field_manifest_v1 import (
    BATCHED_TENSOR_ROW,
    CANONICAL_FIELD_COUNT_V1,
    CANONICAL_FIELD_NAMES_V1,
    CANONICAL_FIELD_SURFACE_DIGEST_V1,
    CANONICAL_MANIFEST_DIGEST_V1,
    DERIVED_BATCHED,
    DIRECT_RESIDENT_FIELD_MANIFEST_V1,
    EED_OVERLAY_ROW,
    EVENT_COLUMN_ROW,
    FIXED_MLP_CONTRACT_V1,
    HOST_AUTHORITATIVE_ROW,
    OPAQUE_CPU_RNG_AUTHORITY_V1,
    POLICY_ALIAS_AUTHORITY_V1,
    WDL_OVERLAY_ROW,
    WORLD_RECONSTRUCTION_AUTHORITY_V1,
    ZERO_FILL,
    DirectResidentFieldManifestError,
    build_direct_resident_field_manifest_v1,
    validate_direct_resident_field_manifest_v1,
    validate_direct_resident_field_names_v1,
)
from engine.native_runtime.checkpoint_digests import DEFAULT_FIELD_POLICIES
from engine.perf.cuda_architecture_b_boundary import (
    EVENT_COLUMN_NAMES,
    POLICY_GENERATION_ALIASES,
)
from engine.perf.cuda_architecture_b_direct_resident_eed_v1 import (
    EED_STATE_FIELDS_V1,
)
from engine.perf.cuda_architecture_b_direct_resident_wdl_v2 import (
    WDL_STATE_FIELDS_V2,
)


def _fields_by_name():
    return {field.name: field for field in DIRECT_RESIDENT_FIELD_MANIFEST_V1.fields}


def test_exact_surface_order_and_digest_are_deterministic() -> None:
    manifest_a = build_direct_resident_field_manifest_v1(
        fixed_mlp_contract=True
    )
    manifest_b = build_direct_resident_field_manifest_v1(
        fixed_mlp_contract=True
    )

    assert CANONICAL_FIELD_COUNT_V1 == 191
    assert len(CANONICAL_FIELD_NAMES_V1) == 191
    assert tuple(field.name for field in manifest_a.fields) == CANONICAL_FIELD_NAMES_V1
    assert manifest_a.to_dict() == manifest_b.to_dict()
    assert manifest_a.canonical_digest == CANONICAL_MANIFEST_DIGEST_V1
    assert CANONICAL_FIELD_SURFACE_DIGEST_V1 == (
        "bdca0a7b7513be33191f355c135835bd5204d5568e0c518413a6d5926536ddc6"
    )
    assert validate_direct_resident_field_manifest_v1(manifest_a) == manifest_a
    assert json.dumps(manifest_a.to_dict(), sort_keys=True) == json.dumps(
        manifest_b.to_dict(), sort_keys=True
    )


def test_known_exception_classifications_are_explicit() -> None:
    fields = _fields_by_name()

    assert all(
        fields[name].provider_kind == EVENT_COLUMN_ROW
        and fields[name].indexing == "active_event_prefix_index_v1"
        for name in EVENT_COLUMN_NAMES
    )
    assert all(
        fields[name].provider_kind == WDL_OVERLAY_ROW for name in WDL_STATE_FIELDS_V2
    )
    assert all(
        fields[name].provider_kind == EED_OVERLAY_ROW for name in EED_STATE_FIELDS_V1
    )

    rng = fields["rng_manager_mutation_torch_cpu_state"]
    assert rng.provider_kind == HOST_AUTHORITATIVE_ROW
    assert rng.authority == OPAQUE_CPU_RNG_AUTHORITY_V1

    for alias, canonical in POLICY_GENERATION_ALIASES.items():
        assert fields[alias].provider_kind == DERIVED_BATCHED
        assert fields[alias].alias_of == canonical
        assert fields[alias].authority == POLICY_ALIAS_AUTHORITY_V1

    assert fields["gru_pack"].provider_kind == ZERO_FILL
    assert fields["hmem"].provider_kind == ZERO_FILL
    assert fields["gru_pack"].authority == FIXED_MLP_CONTRACT_V1
    assert fields["hmem"].authority == FIXED_MLP_CONTRACT_V1

    assert fields["action_cache_next"].provider_kind == DERIVED_BATCHED
    assert fields["patch_density"].provider_kind == DERIVED_BATCHED
    for name in (
        "action_cache_actions",
        "action_cache_cursor",
        "action_cache_token",
        "agent_grid",
        "patch_recovery_activity",
        "patch_scent_emit_pending",
        "patch_waste_emit_pending",
    ):
        assert fields[name].provider_kind == BATCHED_TENSOR_ROW

    # Current checkpoint authority says these are exact trajectory state, so
    # the manifest keeps them batched.  An unverified zero-fill assertion is
    # rejected rather than silently treating them as reconstructible.
    for name in ("biome_grid", "hazard_rate_map"):
        assert fields[name].provider_kind == BATCHED_TENSOR_ROW
        assert DEFAULT_FIELD_POLICIES[name].reconstructible is False
    with pytest.raises(DirectResidentFieldManifestError, match="does not permit"):
        build_direct_resident_field_manifest_v1(
            world_reconstruction_authority={
                "biome_grid": WORLD_RECONSTRUCTION_AUTHORITY_V1,
            }
        )

    non_fixed_mlp = build_direct_resident_field_manifest_v1(
        fixed_mlp_contract=False
    )
    non_fixed_by_name = {field.name: field for field in non_fixed_mlp.fields}
    assert non_fixed_by_name["gru_pack"].provider_kind == BATCHED_TENSOR_ROW
    assert non_fixed_by_name["hmem"].provider_kind == BATCHED_TENSOR_ROW


def test_field_name_validation_rejects_duplicates_missing_and_additional() -> None:
    names = list(CANONICAL_FIELD_NAMES_V1)

    duplicate = names + [names[0]]
    with pytest.raises(DirectResidentFieldManifestError, match="duplicate"):
        validate_direct_resident_field_names_v1(duplicate)

    missing = names[:-1]
    with pytest.raises(DirectResidentFieldManifestError, match="missing"):
        validate_direct_resident_field_names_v1(missing)

    additional = names + ["unsupported_field"]
    with pytest.raises(DirectResidentFieldManifestError, match="unsupported"):
        validate_direct_resident_field_names_v1(additional)

    # Equivalent membership in a different caller order is normalized to the
    # source-derived canonical order; the manifest never preserves caller order.
    assert validate_direct_resident_field_names_v1(tuple(reversed(names))) == (
        CANONICAL_FIELD_NAMES_V1
    )


def test_manifest_schema_classification_and_digest_are_fail_closed() -> None:
    payload = DIRECT_RESIDENT_FIELD_MANIFEST_V1.to_dict()

    bad_schema = dict(payload)
    bad_schema["schema_version"] = "wrong_schema"
    with pytest.raises(DirectResidentFieldManifestError, match="schema"):
        validate_direct_resident_field_manifest_v1(bad_schema)

    bad_digest = dict(payload)
    bad_digest["canonical_digest"] = "0" * 64
    with pytest.raises(DirectResidentFieldManifestError, match="digest"):
        validate_direct_resident_field_manifest_v1(bad_digest)

    bad_kind = dict(payload)
    bad_kind["fields"] = [dict(item) for item in payload["fields"]]
    bad_kind["fields"][0]["provider_kind"] = "unsupported_kind"
    with pytest.raises(
        DirectResidentFieldManifestError, match="provider kind|classification"
    ):
        validate_direct_resident_field_manifest_v1(bad_kind)

    bad_field_count = dict(payload)
    bad_field_count["field_count"] = 190
    with pytest.raises(DirectResidentFieldManifestError, match="field_count"):
        validate_direct_resident_field_manifest_v1(bad_field_count)


def test_import_path_performs_no_cuda_allocation() -> None:
    script = """
import importlib
import torch

before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
module = importlib.import_module(
    'engine.campaigns.acs_direct_resident_field_manifest_v1'
)
after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
assert module.CANONICAL_FIELD_COUNT_V1 == 191
assert before == after
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
