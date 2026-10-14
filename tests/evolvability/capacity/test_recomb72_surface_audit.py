from __future__ import annotations

import hashlib
import json

from evolution.inheritance_registry_v1 import (
    LINKED_INHERITANCE_PROFILE,
    inheritance_registry_metadata,
)
from evolution.recomb72_surface_audit import (
    AUDITED_SCALAR_LOCI,
    CLASSIFICATIONS,
    surface_audit_artifact,
    surface_audit_payload,
    surface_audit_sha256,
    write_surface_audit,
)


def _canonical_bytes(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _surfaces(payload) -> dict[str, dict]:
    return {row["surface_id"]: row for row in payload["surfaces"]}


def test_payload_is_deterministic_and_hash_bound_to_production_registry() -> None:
    first = surface_audit_payload()
    second = surface_audit_payload()
    assert first == second
    assert surface_audit_sha256() == hashlib.sha256(
        _canonical_bytes(first)
    ).hexdigest()
    production = inheritance_registry_metadata()
    binding = first["production_registry_binding"]
    assert binding["classification"] == "measured_directly"
    assert binding["registry"] == production
    assert (
        binding["registry_metadata_sha256"]
        == production["registry_metadata_sha256"]
    )


def test_all_required_surface_families_are_bound_to_registry_rows() -> None:
    payload = surface_audit_payload()
    surfaces = _surfaces(payload)
    assert set(surfaces) == {
        "recomb72_audited_scalar_loci",
        "other_bounded_scalar_genes",
        "memory_architecture",
        "categorical_masks_and_sensor_radius",
        "family_head_residual_record",
        "mlp_policy_controller",
        "gru_policy_controller",
        "structural_modules_and_routing",
        "structural_connections",
        "developmental_records",
    }
    registry_ids = {
        row["semantic_locus_id"]
        for row in payload["production_registry_binding"]["registry"]["loci"]
    }
    for surface in surfaces.values():
        assert surface["registry_locus_ids"]
        assert set(surface["registry_locus_ids"]) <= registry_ids
        assert {
            row["semantic_locus_id"] for row in surface["registry_rows"]
        } == set(surface["registry_locus_ids"])


def test_only_six_gpu_audited_scalar_loci_are_measured_directly() -> None:
    payload = surface_audit_payload()
    assert tuple(payload["classification_vocabulary"]) == CLASSIFICATIONS
    surfaces = _surfaces(payload)
    audited = surfaces["recomb72_audited_scalar_loci"]
    assert audited["registry_locus_ids"] == [
        f"flat.{locus}" for locus in AUDITED_SCALAR_LOCI
    ]
    measurement = audited["recomb72_campaign_measurement"]
    assert measurement["classification"] == "measured_directly"
    assert measurement["audited_loci"] == list(AUDITED_SCALAR_LOCI)
    assert measurement["owner"] == {
        "path": "engine/perf/native/recomb72_gpu_native.cu",
        "symbol": "recomb72_gpu_advance_v1",
    }
    assert measurement["qualification_contract"]["qualification_status"] == (
        "component_not_migrated_fail_closed"
    )
    assert measurement["qualification_contract"]["provider_id"] == (
        "eiel-production-provider-v1"
    )
    assert measurement["historical_execution_path"]["status"] == (
        "removed_from_production"
    )
    assert measurement["qualification_contract"]["native_abi"] == (
        "recomb72_gpu_native_v1"
    )
    parity = measurement["production_current_parity"]
    assert parity["exact_every_g1_event"] is True
    assert parity["native_semantics"] == "exact_float32_legacy_parent_midpoint"
    assert parity["general_production_crossover_invoked_by_campaign"] is False
    assert parity["general_production_crossover_used_as_test_oracle"] is True
    assert parity["test"]["symbol"] == (
        "test_gpu_current_mode_matches_production_crossover_for_every_g1_event"
    )

    for surface_id, surface in surfaces.items():
        if surface_id != "recomb72_audited_scalar_loci":
            assert (
                surface["recomb72_campaign_measurement"]["classification"]
                == "not_tested"
            )
        for row in surface["source_classification"].values():
            assert row["classification"] in CLASSIFICATIONS
    assert (
        payload["homology_and_capacity_metadata"][
            "recomb72_campaign_measurement"
        ]["classification"]
        == "not_tested"
    )
    assert (
        payload["campaign_authority_decision"][
            "current_route_invokes_production_inheritance"
        ]
        is False
    )
    decision = payload["campaign_authority_decision"]
    assert decision["current_route_directly_measures_audited_scalar_loci"] is True
    assert decision["native_fixture_reproduces_legacy_midpoint_semantics"] is True
    assert decision["general_production_crossover_invoked_by_campaign"] is False


def test_existing_linked_parent_copy_repair_and_native_gaps_are_explicit() -> None:
    payload = surface_audit_payload()
    repair = payload["existing_repair"]
    assert repair["profile"] == LINKED_INHERITANCE_PROFILE
    assert repair["profile"] == "linked_parent_copy_v1"
    assert repair["draws_per_offspring"] == 7
    assert repair["native_supported"] is False

    gaps = {row["gap_id"]: row for row in payload["native_gaps"]}
    assert set(gaps) == {
        "resident_resource_axis_absent",
        "resident_family_head_absent",
        "resident_structural_genetics_rejected",
        "linked_parent_copy_native_rejected",
    }
    audited = _surfaces(payload)["recomb72_audited_scalar_loci"]
    assert "flat.resource_handling_axis" in audited["registry_locus_ids"]
    assert (
        audited["recomb72_campaign_measurement"]["classification"]
        == "measured_directly"
    )


def test_homology_capacity_and_controller_semantics_are_owned() -> None:
    payload = surface_audit_payload()
    metadata = payload["homology_and_capacity_metadata"]
    assert metadata["classification"] == "inferred"
    assert "homology_id" in metadata["homology_fields"]
    assert "inherited_homology_id" in metadata["homology_fields"]
    assert "b2_caps" in metadata["capacity_fields"]
    assert metadata["frozen_capacity_values"] == {
        "max_modules_per_agent": 12,
        "max_connections_per_agent": 32,
        "max_module_fanin": 6,
        "max_module_fanout": 6,
    }
    assert (
        metadata["registry_structural_linkage_rule"]["id"]
        == "homology_connected_component_v1"
    )
    surfaces = _surfaces(payload)
    assert (
        surfaces["mlp_policy_controller"]["source_classification"][
            "python_reference"
        ]["owner"]["symbol"]
        == "crossover_policy"
    )
    assert "single compatible GRU donor" in surfaces[
        "gru_policy_controller"
    ]["source_classification"]["resident_native_cpu"]["finding"]


def test_artifact_writer_emits_canonical_json_with_reproducible_hash(
    tmp_path,
) -> None:
    artifact = surface_audit_artifact()
    payload = dict(artifact)
    recorded = payload.pop("payload_sha256")
    assert recorded == surface_audit_sha256()
    target = write_surface_audit(tmp_path / "surface-audit.json")
    assert target.read_bytes() == _canonical_bytes(artifact) + b"\n"
    loaded = json.loads(target.read_text(encoding="utf-8"))
    loaded_payload = dict(loaded)
    loaded_hash = loaded_payload.pop("payload_sha256")
    assert loaded_hash == hashlib.sha256(
        _canonical_bytes(loaded_payload)
    ).hexdigest()
