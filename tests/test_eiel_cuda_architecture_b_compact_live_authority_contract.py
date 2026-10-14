from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from engine.native_runtime import checkpoint_digests
from engine.native_runtime import compact_summary_planes as abi
from engine.native_runtime import compact_summary_retained_digest_finalizer as finalizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "tools/scripts/eiel_cuda_architecture_b_compact_live_authority_contract.py"
)
EXPECTED_FILENAMES = {
    "compact_live_authority_pod_v1.md",
    "compact_live_authority_pod_v1.json",
    "compact_live_authority_source_map_v1.json",
    "compact_generation_metadata_retention_v1.json",
    "compact_rng_outer_preimage_v1.json",
}


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compact_live_authority_contract", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_map_has_complete_exact_coverage_without_duplicates() -> None:
    module = _load_module()
    source_map = module.build_contracts()[
        "compact_live_authority_source_map_v1.json"
    ]

    expected = {
        "status_codes": tuple(abi.STATUS_FIELDS),
        "integer_values": tuple(abi.INTEGER_FIELDS),
        "float_values": tuple(abi.FLOAT_FIELDS),
        "digest_values": tuple(abi.DIGEST_FIELDS),
        "text_values": tuple(abi.TEXT_FIELDS),
    }
    fields = source_map["field_sources"]
    identities = [(row["plane"], row["field"]) for row in fields]
    assert len(identities) == len(set(identities)) == sum(map(len, expected.values()))
    for plane, registry in expected.items():
        rows = [row for row in fields if row["plane"] == plane]
        assert [row["index"] for row in rows] == list(range(len(registry)))
        assert [row["field"] for row in rows] == list(registry)

    for row in fields:
        assert row["source_class"] in module.SOURCE_CLASSES
        assert isinstance(row["physical_source"], str) and row["physical_source"]
        assert isinstance(row["abi_anchor"], str) and row["abi_anchor"]

    mask_rows = source_map["defined_mask_sources"]
    assert [row["bit"] for row in mask_rows] == list(range(len(abi.DEFINED_FIELDS)))
    assert [row["field"] for row in mask_rows] == list(abi.DEFINED_FIELDS)
    assert len({row["field"] for row in mask_rows}) == len(mask_rows) == 32
    assert source_map["coverage"]["unmapped_fields"] == []
    assert source_map["coverage"]["unmapped_defined_mask_bits"] == []
    assert source_map["coverage"]["competing_authorities"] == []


def test_digest_authorities_match_boundary_contract() -> None:
    module = _load_module()
    source_map = module.build_contracts()[
        "compact_live_authority_source_map_v1.json"
    ]
    host = source_map["host_canonical_digest_fields"]
    assert [row["field"] for row in host] == [
        "rng.authority_digest_sha256",
        "rng.policy_mutation_tape_digest_sha256",
        "state_digests.authoritative_state_digest_sha256",
        "state_digests.canonical_checkpoint_storage_digest_sha256",
        "state_digests.next_generation_start_state_digest_sha256",
        "behavior_metrics.rows_digest_sha256",
        "lineage_metrics.records_digest_sha256",
    ]
    assert [row["mask"] for row in host] == [1, 2, 4, 8, 16, 32, 64]
    assert source_map["host_canonical_digest_mask"] == 0x7F
    assert {row["field"] for row in source_map["device_retained_digest_fields"]} == {
        "events.active_prefix_digest_sha256",
        "identity.active_agent_ids_digest_sha256",
    }

    digest_rows = {
        row["field"]: row
        for row in source_map["field_sources"]
        if row["plane"] == "digest_values"
    }
    assert {field for field, row in digest_rows.items() if row["host_canonical"]} == {
        row["field"] for row in host
    }
    assert {field for field, row in digest_rows.items() if row["device_retained"]} == {
        "events.active_prefix_digest_sha256",
        "identity.active_agent_ids_digest_sha256",
    }


def test_pod_metadata_rng_schemas_and_claims_are_frozen() -> None:
    module = _load_module()
    contracts = module.build_contracts()
    pod = contracts["compact_live_authority_pod_v1.json"]
    metadata = contracts["compact_generation_metadata_retention_v1.json"]
    rng = contracts["compact_rng_outer_preimage_v1.json"]

    assert pod["schema_version"] == module.POD_SCHEMA_VERSION
    assert pod["layout"]["byte_width"] == 1256
    assert pod["layout"]["alignment"] == 8
    assert pod["layout"]["source_plane_logical_bytes"] == 1218
    assert pod["projection_status"] == {
        "cpp_type": "int32_t",
        "bytes_per_generation_cell": 4,
        "separate_preallocated_plane": True,
        "total_row_accounting_bytes": 1260,
    }
    assert pod["shape"] == ["generation_count", "cell_count"]
    assert pod["ordering"] == "generation-major then cell-major"
    assert pod["claims"]["dynamic_allocation"] is False
    assert pod["claims"]["pointer_members"] is False
    assert pod["claims"]["python_generation_ownership"] is False
    assert pod["claims"]["capability_promoted"] is False
    assert pod["claims"]["eiel_compact_generation_summary_v1_supported"] is False

    retained_constants = metadata["accepted_constants"][
        "compact_summary_retained_digest_finalizer.py"
    ]
    for name in (
        "_TOP_LEVEL_FIELDS",
        "_CHECKPOINT_FIELDS",
        "_IDENTITY_FIELDS",
        "_POLICY_FIELDS",
        "_REPORTING_FIELDS",
        "_LIFECYCLE_FIELDS",
        "_TERMINAL_FIELDS",
        "_SOURCE_TENSOR_DESCRIPTOR_FIELDS",
    ):
        assert retained_constants[name] == list(getattr(finalizer, name))
    assert metadata["checkpoint_tensor_descriptors"]["descriptor_count"] == 133
    assert metadata["checkpoint_tensor_descriptors"]["batch_count"] == 9
    assert len(metadata["checkpoint_tensor_descriptors"]["descriptors"]) == 133
    assert metadata["bounded_non_tensor_metadata"]["full_tensor_payloads"] is False
    assert metadata["claims"]["tensor_descriptors_distinct_from_non_tensor_metadata"] is True

    rng_constants = rng["accepted_constants"]["checkpoint_digests.py"]
    assert rng_constants["SOURCE_RNG_REQUIRED_METADATA_FIELDS"] == list(
        checkpoint_digests.SOURCE_RNG_REQUIRED_METADATA_FIELDS
    )
    assert rng_constants["SOURCE_RNG_OPTIONAL_METADATA_FIELDS"] == list(
        checkpoint_digests.SOURCE_RNG_OPTIONAL_METADATA_FIELDS
    )
    assert rng_constants["SOURCE_RNG_TENSOR_FIELDS"] == list(
        checkpoint_digests.SOURCE_RNG_TENSOR_FIELDS
    )
    assert rng["composite_rng_authority"]["separate_pcg64_state_authority"] is False
    assert rng["composite_rng_authority"]["separate_pcg64_cursor_authority"] is False
    operations = [row["operation"] for row in rng["host_completion_sequence"]]
    assert operations == [
        "bounded_d2h",
        "stream_host_callback",
        "bounded_h2d_authority_completion",
        "source_projection",
        "final_synchronization",
    ]
    assert rng["claims"]["native_cpp_stream_callback_count"] == 1
    assert rng["claims"]["python_generation_ownership"] is False


def test_generation_is_deterministic_and_writes_exactly_five_filenames(
    tmp_path: Path,
) -> None:
    module = _load_module()
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_paths = module.generate_contract(first)
    second_paths = module.generate_contract(second)

    assert {path.name for path in first_paths} == EXPECTED_FILENAMES
    assert {path.name for path in first.iterdir()} == EXPECTED_FILENAMES
    assert {path.name for path in second.iterdir()} == EXPECTED_FILENAMES
    for name in EXPECTED_FILENAMES:
        assert (first / name).read_bytes() == (second / name).read_bytes()

    receipt = json.loads((first / "compact_live_authority_pod_v1.json").read_text())
    assert set(receipt["artifact_set"]) == EXPECTED_FILENAMES
    markdown = (first / "compact_live_authority_pod_v1.md").read_text()
    assert "1256 bytes" in markdown
    assert "Per generation/cell accounting: 1260 bytes" in markdown
    assert "capability remains false" in markdown
