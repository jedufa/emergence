from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from authoritative_state.schema import (
    AuthoritativeStateSchemaError,
    SCHEMA_ROOT,
    load_schema_v1,
    validate_boundary_v1,
)
from authoritative_state.generated import schema_v1 as generated_schema


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools/scripts/eiel_authoritative_state_schema_v1.py"


def test_canonical_package_is_complete_and_resolved() -> None:
    schema = load_schema_v1()
    assert len(schema.fields) == 704
    source_commit = (
        Path(__file__).resolve().parents[1]
        / "_spec/authoritative_state/v1/source_commit.txt"
    ).read_text(encoding="ascii").strip()
    assert schema.root["source_commit"] == source_commit
    assert schema.root["decision_target"] == "STATE-SCHEMA-V1-B"
    assert schema.components["field_registry.json"][
        "unresolved_supported_field_count"
    ] == 0
    assert schema.components["field_registry.json"]["resolved_field_count"] == 704
    assert schema.components["field_registry.json"][
        "deferred_unsupported_field_count"
    ] == 0


def test_eed_conditional_world_plane_has_exact_authoritative_ownership() -> None:
    schema = load_schema_v1()
    field = schema.field("runtime.eed_resource_impact_state.v1")

    assert field["semantic_owner"] == "world.overlay.eed_resource_impact.v1"
    assert field["physical_type"] == "float32"
    assert field["rank"] == 3
    assert field["shape_formula"] == "[world_height,world_width,10]"
    assert field["capacity_formula"] == "product([world_height,world_width,10])"
    assert field["optional"] is True
    assert field["lifecycle"] == "conditional_world_state"
    assert (
        field["checkpoint_encoding"]
        == "checkpoint_bundle_v2_conditional_typed_plane"
    )
    assert field["abi_variants"] == []
    assert schema.root["terminal_target"] == "STATE-SCHEMA-V1-B"
    assert schema.root["supported_surfaces"] == [
        {
            "provider_id": "eiel-production-provider-v1",
            "profile_id": "eiel-production-runtime-v1",
            "required_abi_variants": [
                "direct_resident_outer_172",
                "event_state_31",
                "full_tick_state_29",
                "ordinary_birth_diploid_122",
                "ordinary_birth_output_63",
                "ordinary_birth_scratch_1",
                    "patch_p2_state_full_10",
                    "turnover_mutable_state_122",
                    "turnover_output_16",
            ],
        }
    ]
    assert schema.root["unsupported_surfaces"] == []


def test_every_nonempty_abi_entry_is_registered_or_explicitly_deferred() -> None:
    schema = load_schema_v1()
    registry_ids = set(schema.fields_by_id)
    abi_ids = {
        entry["semantic_field_id"]
        for variant in schema.abi_variants["variants"]
        for entry in variant["entries"]
    }
    assert len(abi_ids) == 583
    assert abi_ids <= registry_ids
    deferred = [
        schema.fields_by_id[field_id]
        for field_id in abi_ids
        if schema.fields_by_id[field_id]["resolution_status"]
        == "deferred_unsupported"
    ]
    assert deferred == []
    resolved = [
        schema.fields_by_id[field_id]
        for field_id in abi_ids
        if schema.fields_by_id[field_id]["resolution_status"] == "resolved"
    ]
    assert len(resolved) == len(abi_ids)
    assert all(field["production_admission"] == "admitted" for field in resolved)
    scratch = schema.fields_by_id[
        "abi.ordinary_birth_scratch.lifecycle_scratch_storage.occurrence_1.v1"
    ]
    assert scratch["physical_type"] == "uint8"
    assert scratch["shape_formula"].startswith("[B,lifecycle_scratch_bytes")
    assert len(generated_schema.FIELD_IDS) == 704
    assert len(generated_schema.RESOLVED_FIELD_IDS) == 704
    assert generated_schema.DEFERRED_UNSUPPORTED_FIELD_IDS == ()
    assert all(
        generated_schema.PRODUCTION_ADMISSION[field_id] == "admitted"
        for field_id in abi_ids
    )


def test_schema_generator_rejects_abi_source_contract_drift() -> None:
    spec = importlib.util.spec_from_file_location(
        "authoritative_state_schema_generator", GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    abi = json.loads(
        (ROOT / "_spec/authoritative_state/v1/abi_variants.json").read_text()
    )
    abi["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(module.GenerationError, match="differs from current"):
        module._validate_abi_source_contract(abi)


def test_schema_generator_rejects_noncanonical_production_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "authoritative_state_schema_generator_surface", GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.load_json

    def load_noncanonical(path: Path) -> dict:
        result = original(path)
        if path == module.SCHEMA_PATH:
            result["supported_surfaces"] = [
                {
                    "provider_id": "legacy-provider-v1",
                    "profile_id": "legacy-profile-v1",
                    "required_abi_variants": [],
                }
            ]
        return result

    monkeypatch.setattr(module, "load_json", load_noncanonical)
    with pytest.raises(
        module.GenerationError,
        match="exactly the canonical production engine surface",
    ):
        module.generate(check=True)


def test_all_component_field_ids_exist_in_canonical_registry() -> None:
    schema = load_schema_v1()
    identifiers = set(schema.fields_by_id)
    component_ids: set[str] = set()
    for filename in (
        "checkpoint_schema.json",
        "chronicle_schema.json",
        "provider_projection_schema.json",
        "analysis_projection_schema.json",
    ):
        component = schema.components[filename]
        component_ids.update(
            row["field_id"] for row in component.get("fields", [])
        )
        component_ids.update(component.get("identity_field_ids", []))
        for row in component.get("fields", []):
            component_ids.update(row.get("source_field_ids", []))
    assert component_ids <= identifiers


def test_generation_is_byte_deterministic() -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["changed"] == []


def test_inventory_hashes_python_and_native_boundary_sources() -> None:
    spec = importlib.util.spec_from_file_location(
        "authoritative_state_schema_inventory", GENERATOR
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    first = module.inventory()
    second = module.inventory()

    assert first == second
    assert first["method"] == (
        "python_ast_and_native_lexical_symbols_and_string_literals"
    )
    assert first["source_file_count"] == len(first["source_files"])
    assert first["source_files_sha256"] == module.value_hash(first["source_files"])
    assert {row["language"] for row in first["source_files"]} == {
        "python",
        "native",
    }
    assert any(
        row["language"] == "native"
        and row["path"].endswith(
            "cuda_architecture_b_flattened_controller_v1_kernels.cu"
        )
        and row["name"] == "policy_pcg64_buffer_after_action"
        for row in first["declarations"]
    )
    for row in first["source_files"]:
        path = ROOT / row["path"]
        assert module.file_hash(path) == row["sha256"]


def test_boundary_dtype_rank_and_shape_fail_closed() -> None:
    validate_boundary_v1(
        "identity.lineage_founder_id.v1",
        physical_type="int32",
        rank=2,
        shape=(3, 128),
    )
    with pytest.raises(AuthoritativeStateSchemaError) as dtype:
        validate_boundary_v1(
            "identity.lineage_founder_id.v1",
            physical_type="int64",
            rank=2,
            shape=(3, 128),
        )
    assert dtype.value.code == "EIEL_SCHEMA_DTYPE_MISMATCH"
    with pytest.raises(AuthoritativeStateSchemaError) as shape:
        validate_boundary_v1(
            "identity.lineage_founder_id.v1",
            physical_type="int32",
            rank=2,
            shape=(3, -1),
        )
    assert shape.value.code == "EIEL_SCHEMA_SHAPE_INVALID"


def test_stale_component_hash_is_rejected_before_use(tmp_path: Path) -> None:
    package = tmp_path / "v1"
    package.mkdir()
    for path in SCHEMA_ROOT.iterdir():
        if path.is_file():
            (package / path.name).write_bytes(path.read_bytes())
    checkpoint = json.loads((package / "checkpoint_schema.json").read_text())
    checkpoint["schema_version"] = "stale"
    (package / "checkpoint_schema.json").write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )
    with pytest.raises(AuthoritativeStateSchemaError) as error:
        load_schema_v1(package)
    assert error.value.code == "EIEL_SCHEMA_COMPONENT_HASH_MISMATCH"


def test_supported_surface_is_not_listed_as_unsupported() -> None:
    schema = load_schema_v1()
    assert schema.root["unsupported_surfaces"] == []
