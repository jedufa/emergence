from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from tools.scripts import eiel_authoritative_state_completion_coverage_v1 as subject


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ROOT = PROJECT_ROOT / "_spec" / "authoritative_state" / "v1"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )


def _sealed_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    anchor = source_root / "engine" / "boundary.py"
    anchor.parent.mkdir()
    anchor.write_text(
        "def checked_integer_boundary():\n"
        "    return 'minimum maximum sentinel overflow'\n"
        "\n"
        "def checked_float_boundary():\n"
        "    return 'finite range narrowing overflow'\n",
        encoding="utf-8",
    )
    evidence = source_root / "tests" / "test_boundary.py"
    evidence.parent.mkdir()
    evidence.write_text(
        "def test_minimum_maximum_sentinel():\n"
        "    pass\n"
        "\n"
        "def test_float_narrowing_roundtrip():\n"
        "    pass\n",
        encoding="utf-8",
    )
    anchor_row = {
        "path": "engine/boundary.py",
        "source_sha256": _file_sha(anchor),
    }
    fields = [
        {
            "field_id": "field.alpha.v1",
            "production_admission": "admitted",
            "resolution_status": "resolved",
            "boundary_test_profile": "integer_boundaries_v1",
            "physical_type": "int32",
            "signedness": "signed",
            "source_anchors": [anchor_row],
        },
        {
            "field_id": "field.beta.v1",
            "production_admission": "admitted",
            "resolution_status": "resolved",
            "boundary_test_profile": "float_roundtrip_v1",
            "physical_type": "float32",
            "signedness": "ieee754",
            "source_anchors": [anchor_row],
        },
        {
            "field_id": "field.deferred.v1",
            "production_admission": "forbidden",
            "resolution_status": "deferred_unsupported",
            "boundary_test_profile": "not_admitted_v1",
            "physical_type": "string",
            "signedness": "not_applicable",
            "source_anchors": [anchor_row],
        },
    ]
    abi_variants = {
        "schema_version": "fixture_abi_v1",
        "variants": [
            {
                "variant_id": "fixture_supported_2",
                "support_status": "supported_exact_metadata",
                "count": 2,
                "entries": [
                    {"semantic_field_id": "field.alpha.v1"},
                    {"semantic_field_id": "field.beta.v1"},
                ],
            },
            {
                "variant_id": "fixture_deferred_1",
                "support_status": "unsupported",
                "count": 1,
                "entries": [{"semantic_field_id": "field.deferred.v1"}],
            },
        ],
    }
    schema_root = source_root / "_spec" / "authoritative_state" / "v1"
    registry_path = schema_root / "field_registry.json"
    abi_path = schema_root / "abi_variants.json"
    _write_json(
        registry_path,
        {"schema_version": "fixture_registry_v1", "fields": fields},
    )
    _write_json(abi_path, abi_variants)
    schema: dict[str, object] = {
        "schema_version": "fixture_authoritative_state_v1",
        "component_hashes": {
            "abi_variants.json": _file_sha(abi_path),
            "field_registry.json": _file_sha(registry_path),
        },
    }
    schema["schema_hash"] = subject.content_sha256(schema)
    _write_json(schema_root / "schema.json", schema)

    evidence_ref = {
        "path": "tests/test_boundary.py",
        "sha256": _file_sha(evidence),
    }
    coverage: dict[str, object] = {
        "schema_version": subject.COVERAGE_MAP_SCHEMA_VERSION_V1,
        "state_schema_version": schema["schema_version"],
        "state_schema_sha256": schema["schema_hash"],
        "boundary_profile_coverage": [
            {
                "boundary_test_profile": "integer_boundaries_v1",
                "evidence_refs": [
                    {
                        **evidence_ref,
                        "selectors": ["test_minimum_maximum_sentinel"],
                    }
                ],
            },
            {
                "boundary_test_profile": "float_roundtrip_v1",
                "evidence_refs": [
                    {
                        **evidence_ref,
                        "selectors": ["test_float_narrowing_roundtrip"],
                    }
                ],
            },
        ],
        "conversion_policy_coverage": [
            {
                "policy_id": "checked_signed_int32_v1",
                "physical_types": ["int32"],
                "signedness": ["signed"],
                "evidence_refs": [
                    {
                        **evidence_ref,
                        "selectors": ["test_minimum_maximum_sentinel"],
                    }
                ],
            },
            {
                "policy_id": "checked_float32_v1",
                "physical_types": ["float32"],
                "signedness": ["ieee754"],
                "evidence_refs": [
                    {
                        **evidence_ref,
                        "selectors": ["test_float_narrowing_roundtrip"],
                    }
                ],
            },
        ],
    }
    coverage_path = source_root / "coverage-map.json"
    _write_json(coverage_path, coverage)
    return schema_root, coverage_path, source_root, coverage


def test_compact_fixture_emits_deterministic_self_hashed_receipt(
    tmp_path: Path,
) -> None:
    schema_root, coverage_path, source_root, _ = _sealed_fixture(tmp_path)

    first = subject.verify_completion_coverage_v1(
        schema_root=schema_root,
        coverage_map_path=coverage_path,
        source_root=source_root,
    )
    second = subject.verify_completion_coverage_v1(
        schema_root=schema_root,
        coverage_map_path=coverage_path,
        source_root=source_root,
    )

    assert first == second
    assert first["status"] == "accepted"
    assert first["supported_field_count"] == 2
    assert first["supported_abi_variant_count"] == 1
    assert first["supported_abi_occurrence_count"] == 2
    assert first["supported_abi_unique_field_count"] == 2
    assert first["boundary_test_profile_count"] == 2
    assert first["conversion_policy_count"] == 2
    assert first["source_anchor_occurrence_count"] == 2
    assert first["unique_source_anchor_file_count"] == 1
    unsigned = dict(first)
    claimed = unsigned.pop("receipt_sha256")
    assert claimed == subject.content_sha256(unsigned)

    output = subject.publish_receipt_atomic_v1(
        tmp_path / "publication" / "receipt.json",
        first,
    )
    assert json.loads(output.read_text(encoding="utf-8")) == first
    with pytest.raises(
        subject.CompletionCoverageError,
        match="refusing to overwrite",
    ):
        subject.publish_receipt_atomic_v1(output, first)


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("missing_profile", "EIEL_COVERAGE_MISSING"),
        ("unknown_profile", "EIEL_COVERAGE_UNKNOWN"),
        ("duplicate_profile", "EIEL_COVERAGE_DUPLICATE"),
        ("missing_conversion", "EIEL_COVERAGE_MISSING"),
        ("overlapping_conversion", "EIEL_COVERAGE_DUPLICATE"),
        ("unused_conversion", "EIEL_COVERAGE_UNKNOWN"),
        ("stale_schema_binding", "EIEL_COVERAGE_STALE_SCHEMA"),
        ("missing_selector", "EIEL_COVERAGE_SELECTOR_MISSING"),
        ("stale_evidence", "EIEL_COVERAGE_STALE_EVIDENCE"),
    ],
)
def test_rejects_missing_unknown_duplicate_and_stale_coverage(
    tmp_path: Path,
    mutation: str,
    expected_code: str,
) -> None:
    schema_root, coverage_path, source_root, original = _sealed_fixture(tmp_path)
    coverage = deepcopy(original)
    profiles = coverage["boundary_profile_coverage"]
    conversions = coverage["conversion_policy_coverage"]
    assert isinstance(profiles, list)
    assert isinstance(conversions, list)

    if mutation == "missing_profile":
        profiles.pop()
    elif mutation == "unknown_profile":
        profiles[0]["boundary_test_profile"] = "unknown_profile_v1"
    elif mutation == "duplicate_profile":
        profiles.append(deepcopy(profiles[0]))
    elif mutation == "missing_conversion":
        conversions.pop()
    elif mutation == "overlapping_conversion":
        overlapping = deepcopy(conversions[0])
        overlapping["policy_id"] = "overlapping_v1"
        conversions.append(overlapping)
    elif mutation == "unused_conversion":
        unused = deepcopy(conversions[0])
        unused["policy_id"] = "unused_v1"
        unused["physical_types"] = ["uint64"]
        unused["signedness"] = ["unsigned"]
        conversions.append(unused)
    elif mutation == "stale_schema_binding":
        coverage["state_schema_sha256"] = "0" * 64
    elif mutation == "missing_selector":
        profiles[0]["evidence_refs"][0]["selectors"] = ["not_in_evidence"]
    elif mutation == "stale_evidence":
        profiles[0]["evidence_refs"][0]["sha256"] = "0" * 64
    else:  # pragma: no cover
        raise AssertionError(mutation)
    _write_json(coverage_path, coverage)

    with pytest.raises(subject.CompletionCoverageError) as exc_info:
        subject.verify_completion_coverage_v1(
            schema_root=schema_root,
            coverage_map_path=coverage_path,
            source_root=source_root,
        )

    assert exc_info.value.code == expected_code


def test_rejects_source_anchor_hash_drift(tmp_path: Path) -> None:
    schema_root, coverage_path, source_root, _ = _sealed_fixture(tmp_path)
    (source_root / "engine" / "boundary.py").write_text(
        "def drifted():\n    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(subject.CompletionCoverageError) as exc_info:
        subject.verify_completion_coverage_v1(
            schema_root=schema_root,
            coverage_map_path=coverage_path,
            source_root=source_root,
        )

    assert exc_info.value.code == "EIEL_COVERAGE_STALE_SOURCE"


def test_rejects_supported_abi_occurrence_for_nonadmitted_field(
    tmp_path: Path,
) -> None:
    schema_root, coverage_path, source_root, _ = _sealed_fixture(tmp_path)
    abi_path = schema_root / "abi_variants.json"
    abi = json.loads(abi_path.read_text(encoding="utf-8"))
    abi["variants"][0]["entries"][0]["semantic_field_id"] = "field.deferred.v1"
    _write_json(abi_path, abi)
    schema_path = schema_root / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["component_hashes"]["abi_variants.json"] = _file_sha(abi_path)
    schema.pop("schema_hash")
    schema["schema_hash"] = subject.content_sha256(schema)
    _write_json(schema_path, schema)
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    coverage["state_schema_sha256"] = schema["schema_hash"]
    _write_json(coverage_path, coverage)

    with pytest.raises(subject.CompletionCoverageError) as exc_info:
        subject.verify_completion_coverage_v1(
            schema_root=schema_root,
            coverage_map_path=coverage_path,
            source_root=source_root,
        )

    assert exc_info.value.code == "EIEL_COVERAGE_MISSING"


def test_current_schema_has_external_completion_coverage(
    tmp_path: Path,
) -> None:
    registry = json.loads(
        (SCHEMA_ROOT / "field_registry.json").read_text(encoding="utf-8")
    )
    schema = json.loads((SCHEMA_ROOT / "schema.json").read_text(encoding="utf-8"))
    admitted = [
        field
        for field in registry["fields"]
        if field.get("production_admission") == "admitted"
    ]
    evidence_path = Path(__file__).resolve()
    relative_evidence = evidence_path.relative_to(PROJECT_ROOT).as_posix()
    evidence_ref = {
        "path": relative_evidence,
        "sha256": _file_sha(evidence_path),
        "selectors": [
            "test_current_schema_has_external_completion_coverage"
        ],
    }
    profiles = sorted({field["boundary_test_profile"] for field in admitted})
    conversion_pairs = sorted(
        {(field["physical_type"], field["signedness"]) for field in admitted}
    )
    coverage = {
        "schema_version": subject.COVERAGE_MAP_SCHEMA_VERSION_V1,
        "state_schema_version": schema["schema_version"],
        "state_schema_sha256": schema["schema_hash"],
        "boundary_profile_coverage": [
            {
                "boundary_test_profile": profile,
                "evidence_refs": [evidence_ref],
            }
            for profile in profiles
        ],
        "conversion_policy_coverage": [
            {
                "policy_id": f"exact_type_pair_{index:02d}_v1",
                "physical_types": [physical_type],
                "signedness": [signedness],
                "evidence_refs": [evidence_ref],
            }
            for index, (physical_type, signedness) in enumerate(conversion_pairs)
        ],
    }
    coverage_path = tmp_path / "current-coverage-map.json"
    _write_json(coverage_path, coverage)

    receipt = subject.verify_completion_coverage_v1(
        schema_root=SCHEMA_ROOT,
        coverage_map_path=coverage_path,
        source_root=PROJECT_ROOT,
    )

    assert receipt["supported_field_count"] == 692
    assert receipt["supported_abi_variant_count"] == 18
    assert receipt["supported_abi_occurrence_count"] == 813
    assert receipt["supported_abi_unique_field_count"] == 571
    assert receipt["boundary_test_profile_count"] == 11
    assert receipt["conversion_policy_count"] == len(conversion_pairs)
