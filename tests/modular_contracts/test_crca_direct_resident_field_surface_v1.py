"""CRCA-G7 compatible-interface proof for the extracted field surface."""

from __future__ import annotations

import ast
from pathlib import Path

from emergence_contracts import direct_resident_field_surface as contract
from engine.campaigns import acs_direct_resident_field_manifest_v1 as manifest
from engine.perf import cuda_architecture_b_direct_resident_preparation_v1 as preparation
from engine.perf import cuda_architecture_b_direct_resident_source_allocator_v1 as allocator


REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRACTED_NAMES = (
    "BASE_BATCHED_FIELDS",
    "ORDINARY_BIRTH_IDENTITY_FIELDS",
    "ORDINARY_BIRTH_SCALAR_RNG_FIELDS",
    "ORDINARY_BIRTH_GENOME_INT32_FIELDS",
    "ORDINARY_BIRTH_GENOME_FLOAT32_FIELDS",
    "ORDINARY_BIRTH_DRIVE_FIELDS",
    "ORDINARY_BIRTH_POLICY_FIELDS",
    "TURNOVER_STATE_FIELDS",
    "PATCH_RESIDENT_FIELDS",
)


def test_frozen_surface_digest_and_authority_boundary() -> None:
    assert str(contract.DIRECT_RESIDENT_FIELD_SURFACE_DIGEST_V1) == (
        "sha256:e714623ecfa4278f916f7453dc720a499709694600c73d38441e928c255a75a6"
    )
    assert contract.DIRECT_RESIDENT_FIELD_SURFACE_TIME_AUTHORITY is False
    assert contract.DIRECT_RESIDENT_FIELD_SURFACE_QUALIFICATION_AUTHORITY is False
    assert contract.DIRECT_RESIDENT_FIELD_SURFACE_PERSISTENCE_LABELS_ONLY is True
    assert len(contract.PATCH_P2_STATE_FIELDS_V1) == 10
    assert contract.PATCH_RESIDENT_FIELDS[-2:] == (
        "patch_recovery_activity",
        "patch_habitat_lineage_id",
    )


def test_existing_physical_runtime_names_are_exact_compatibility_reexports() -> None:
    assert preparation.PATCH_P2_STATE_FIELDS_V1 is contract.PATCH_P2_STATE_FIELDS_V1
    assert allocator.PATCH_P2_STATE_FIELDS_V1 is contract.PATCH_P2_STATE_FIELDS_V1
    for name in EXTRACTED_NAMES:
        assert getattr(preparation, name) is getattr(contract, name)
        assert getattr(allocator, name) is getattr(contract, name)


def test_campaign_manifest_is_unchanged_and_no_longer_imports_allocator() -> None:
    assert manifest.CANONICAL_FIELD_COUNT_V1 == 191
    assert manifest.CANONICAL_FIELD_SURFACE_DIGEST_V1 == (
        "bdca0a7b7513be33191f355c135835bd5204d5568e0c518413a6d5926536ddc6"
    )
    assert manifest.CANONICAL_MANIFEST_DIGEST_V1 == (
        "162e4be63c5e8f99284d2491eb33d3f0015d8861106131ef5e1c963315fa11bb"
    )

    path = REPO_ROOT / "engine/campaigns/acs_direct_resident_field_manifest_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "emergence_contracts.direct_resident_field_surface" in imported_modules
    assert (
        "engine.perf.cuda_architecture_b_direct_resident_source_allocator_v1"
        not in imported_modules
    )
