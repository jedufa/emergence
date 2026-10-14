from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from engine.campaigns.registry import default_profile_registry
from engine.campaigns.wrc_atlas_v3 import (
    WRCAtlasError,
    load_wrc_atlas_rows_v3,
    load_wrc_v1_candidate_requests_v3,
)
from tools.scripts.normalize_eiel_wrc_atlas_v3 import normalize


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "_spec/registry/v1"


def _profile():
    return next(
        profile
        for profile in default_profile_registry().profiles()
        if profile.profile_id == "eiel-production-runtime-v1"
    )


def test_canonical_atlas_route_corrections_match_factory_capabilities_only():
    requests = load_wrc_v1_candidate_requests_v3(REGISTRY, profile=_profile())

    assert len(requests) == 73
    assert requests[0].row_id == "P011"
    assert requests[-1].row_id == "P083"
    jungle = next(row for row in requests if row.row_id == "P011")
    wdl = next(row for row in requests if row.row_id == "P067")
    patch = next(row for row in requests if row.row_id == "P023")
    assert jungle.world_overlays == ("jungle",)
    assert wdl.world_overlays == ("wdl_delayed_hazard",)
    assert patch.world_overlays == ()
    assert "qualification" not in jungle.campaign_request_metadata()


def test_full_atlas_projection_preserves_all_launcher_classes():
    rows = load_wrc_atlas_rows_v3(REGISTRY)
    assert len(rows) == 95
    assert rows[0].row_id == "P001"
    assert rows[-1].row_id == "P095"
    assert sum(row.launcher_profile == "STANDARD_CUDA" for row in rows) == 73
    assert sum(row.launcher_profile == "SPECIALIZED_MANUAL" for row in rows) == 15
    assert sum(row.launcher_profile == "CONTEXT_MANUAL" for row in rows) == 7


def test_source_canonical_and_manifest_are_reproducible_with_per_row_provenance(
    tmp_path: Path,
):
    source = REGISTRY / "EIEL_WRC_Campaign_Matrix_v3_source.csv"
    canonical = tmp_path / "EIEL_WRC_Campaign_Matrix_v3_canonical.csv"
    manifest_path = tmp_path / "EIEL_WRC_Campaign_Matrix_v3_normalization_manifest.json"
    normalize(source, canonical, manifest_path)

    tracked_canonical = REGISTRY / "EIEL_WRC_Campaign_Matrix_v3_canonical.csv"
    tracked_manifest = REGISTRY / "EIEL_WRC_Campaign_Matrix_v3_normalization_manifest.json"
    assert canonical.read_bytes() == tracked_canonical.read_bytes()
    assert manifest_path.read_bytes() == tracked_manifest.read_bytes()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["canonical"]["sha256"] == hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert len(manifest["corrections"]) == 43
    for correction in manifest["corrections"]:
        assert {
            "old_value",
            "new_value",
            "reason",
            "proving_test",
            "evidence",
            "source_atlas_sha256",
        } <= set(correction)

    with source.open(newline="", encoding="utf-8") as handle:
        source_rows = {row["ID"]: row for row in csv.DictReader(handle)}
    with canonical.open(newline="", encoding="utf-8") as handle:
        canonical_rows = {row["ID"]: row for row in csv.DictReader(handle)}
    changed_ids = {
        row_id
        for row_id in source_rows
        if source_rows[row_id]["World overlays JSON"]
        != canonical_rows[row_id]["World overlays JSON"]
    }
    assert changed_ids == {row["row_id"] for row in manifest["corrections"]}


def test_atlas_tampering_cannot_expand_a_declared_capability(tmp_path: Path):
    for item in REGISTRY.iterdir():
        if item.is_file():
            (tmp_path / item.name).write_bytes(item.read_bytes())
    canonical = tmp_path / "EIEL_WRC_Campaign_Matrix_v3_canonical.csv"
    canonical.write_bytes(canonical.read_bytes().replace(b"env_cfg.move_cost", b"env_cfg.no_such", 1))

    manifest_path = tmp_path / "EIEL_WRC_Campaign_Matrix_v3_normalization_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonical"]["sha256"] = hashlib.sha256(canonical.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(WRCAtlasError, match="undeclared capability"):
        load_wrc_v1_candidate_requests_v3(tmp_path, profile=_profile())
