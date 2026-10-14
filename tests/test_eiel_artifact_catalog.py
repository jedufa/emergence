from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import pytest

from engine.artifact_catalog import (
    ArtifactCatalogError,
    ArtifactCatalog,
    ArtifactVerificationError,
    load_artifact_catalog,
    load_artifact_catalog_v2,
    parse_artifact_catalog,
    parse_artifact_catalog_v2,
    verify_artifact_sha256,
)


def _artifact(
    artifact_id: str = "cuda-patch-p2-sm75-v1", *, status: str = "qualified"
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "kind": "python_extension",
        "abi": "eiel_native_runtime_abi_v1",
        "architecture": "sm_75",
        "build_identity": "source-snapshot-v1",
        "status": status,
        "relative_path": f"cuda/sm75/{artifact_id}.so",
        "sha256": "a" * 64,
        "qualification": {
            "schema_version": "eiel_qualification_binding_v1",
            "envelope_id": "patch-p2-bounded-v1",
            "receipt_sha256": "b" * 64,
            "capabilities": ["compact_summary", "resident_window"],
        },
    }


def _catalog(*artifacts: dict[str, str]) -> dict[str, object]:
    return {"schema_version": "eiel_artifact_catalog_v1", "artifacts": list(artifacts)}


def _artifact_v2(
    artifact_id: str = "cuda-patch-p2-sm75-v2",
    *,
    status: str = "qualified",
    memory_bytes: int = 4_000_000_000,
) -> dict[str, object]:
    artifact: dict[str, object] = dict(_artifact(artifact_id, status=status))
    artifact["requirements"] = {
        "compute_capability": "sm_75",
        "minimum_device_memory_bytes": memory_bytes,
    }
    return artifact


def _catalog_v2(*artifacts: dict[str, object]) -> dict[str, object]:
    return {"schema_version": "eiel_artifact_catalog_v2", "artifacts": list(artifacts)}


def test_load_resolves_relative_paths_and_orders_records(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact-root"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(_catalog(_artifact("z-artifact"), _artifact("a-artifact"))),
        encoding="utf-8",
    )

    loaded = load_artifact_catalog(catalog_path, artifact_root=artifact_root)

    assert [item.artifact_id for item in loaded.artifacts] == [
        "a-artifact",
        "z-artifact",
    ]
    assert (
        loaded.artifacts[0].resolved_path
        == (artifact_root / "cuda/sm75/a-artifact.so").resolve()
    )
    assert not artifact_root.exists()


def test_artifact_root_must_be_absolute() -> None:
    with pytest.raises(ArtifactCatalogError, match="artifact_root must be absolute"):
        parse_artifact_catalog(_catalog(_artifact()), artifact_root="relative-root")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.update(schema_version="eiel_artifact_catalog_v2"),
            "unsupported",
        ),
        (lambda value: value.update(extra=True), "invalid fields"),
        (lambda value: value.pop("artifacts"), "invalid fields"),
        (lambda value: value.update(artifacts={}), "JSON array"),
    ],
)
def test_catalog_rejects_invalid_top_level_shape(
    tmp_path: Path, mutate, message: str
) -> None:
    payload = _catalog(_artifact())
    mutate(payload)
    with pytest.raises(ArtifactCatalogError, match=message):
        parse_artifact_catalog(payload, artifact_root=tmp_path)


@pytest.mark.parametrize("field", sorted(_artifact()))
def test_artifact_requires_every_exact_field(tmp_path: Path, field: str) -> None:
    artifact = _artifact()
    artifact.pop(field)
    with pytest.raises(ArtifactCatalogError, match="invalid fields"):
        parse_artifact_catalog(_catalog(artifact), artifact_root=tmp_path)


def test_artifact_rejects_extra_fields(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact["metadata"] = "not-in-v1"
    with pytest.raises(ArtifactCatalogError, match="invalid fields"):
        parse_artifact_catalog(_catalog(artifact), artifact_root=tmp_path)


@pytest.mark.parametrize("status", ["unknown", 1, None, []])
def test_artifact_rejects_unsupported_status(tmp_path: Path, status: object) -> None:
    artifact = _artifact()
    artifact["status"] = status  # type: ignore[assignment]
    with pytest.raises(ArtifactCatalogError, match="unsupported value"):
        parse_artifact_catalog(_catalog(artifact), artifact_root=tmp_path)


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, 7, None])
def test_artifact_requires_lowercase_sha256(tmp_path: Path, digest: object) -> None:
    artifact = _artifact()
    artifact["sha256"] = digest  # type: ignore[assignment]
    with pytest.raises(ArtifactCatalogError, match="sha256 has an invalid"):
        parse_artifact_catalog(_catalog(artifact), artifact_root=tmp_path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "/absolute/artifact.so",
        "../escape.so",
        "cuda/../escape.so",
        "cuda//artifact.so",
        "cuda\\artifact.so",
    ],
)
def test_artifact_path_must_be_relative_and_contained(
    tmp_path: Path, relative_path: str
) -> None:
    artifact = _artifact()
    artifact["relative_path"] = relative_path
    with pytest.raises(ArtifactCatalogError, match="relative path|artifact root"):
        parse_artifact_catalog(_catalog(artifact), artifact_root=tmp_path)


def test_artifact_path_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "outside-link").symlink_to(tmp_path)
    artifact = _artifact()
    artifact["relative_path"] = "outside-link/artifact.so"
    with pytest.raises(ArtifactCatalogError, match="outside the artifact root"):
        parse_artifact_catalog(_catalog(artifact), artifact_root=root)


def test_artifact_ids_must_be_unique(tmp_path: Path) -> None:
    with pytest.raises(ArtifactCatalogError, match="duplicate artifact_id"):
        parse_artifact_catalog(
            _catalog(_artifact(), _artifact()), artifact_root=tmp_path
        )


def test_filter_and_select_are_exact_and_deterministic(tmp_path: Path) -> None:
    candidate = _artifact("candidate", status="candidate")
    cpu = _artifact("cpu-qualified")
    cpu["architecture"] = "x86_64"
    loaded = parse_artifact_catalog(
        _catalog(_artifact("z-qualified"), candidate, cpu, _artifact("a-qualified")),
        artifact_root=tmp_path,
    )

    matches = loaded.filter(
        kind="python_extension",
        abi="eiel_native_runtime_abi_v1",
        architecture="sm_75",
        build_identity="source-snapshot-v1",
        status="qualified",
        envelope_id="patch-p2-bounded-v1",
        capability="resident_window",
    )
    assert [item.artifact_id for item in matches] == ["a-qualified", "z-qualified"]
    assert (
        loaded.select_one(architecture="x86_64", status="qualified").artifact_id
        == "cpu-qualified"
    )
    with pytest.raises(ArtifactCatalogError, match="matched 2"):
        loaded.select_one(architecture="sm_75", status="qualified")
    with pytest.raises(ArtifactCatalogError, match="matched 0"):
        loaded.select_one(kind="executable")


@pytest.mark.parametrize(
    "capabilities",
    ([], ["resident_window", "compact_summary"], ["same", "same"], [1]),
)
def test_qualification_capabilities_are_nonempty_sorted_unique_tokens(
    tmp_path: Path, capabilities: object
) -> None:
    artifact = _artifact()
    artifact["qualification"]["capabilities"] = capabilities  # type: ignore[index]
    with pytest.raises(ArtifactCatalogError, match="capabilities"):
        parse_artifact_catalog(_catalog(artifact), artifact_root=tmp_path)


def test_qualification_binding_requires_exact_fields(tmp_path: Path) -> None:
    artifact = _artifact()
    artifact["qualification"].pop("envelope_id")  # type: ignore[union-attr]
    with pytest.raises(ArtifactCatalogError, match="invalid fields"):
        parse_artifact_catalog(_catalog(artifact), artifact_root=tmp_path)


@pytest.mark.parametrize(
    "raw_json",
    [
        '{"schema_version":"eiel_artifact_catalog_v1","schema_version":"again","artifacts":[]}',
        '{"schema_version":"eiel_artifact_catalog_v1","artifacts":[],"number":NaN}',
        "not-json",
    ],
)
def test_json_loader_fails_closed_on_non_strict_json(
    tmp_path: Path, raw_json: str
) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(raw_json, encoding="utf-8")
    with pytest.raises(ArtifactCatalogError):
        load_artifact_catalog(path, artifact_root=tmp_path / "artifacts")


def _catalog_for_file(root: Path, contents: bytes) -> tuple[ArtifactCatalog, Path]:
    artifact = _artifact("verified-artifact")
    artifact["sha256"] = hashlib.sha256(contents).hexdigest()
    catalog = parse_artifact_catalog(_catalog(artifact), artifact_root=root)
    path = root / str(artifact["relative_path"])
    path.parent.mkdir(parents=True)
    return catalog, path


def test_opt_in_verification_streams_matching_regular_file(tmp_path: Path) -> None:
    contents = (b"bounded-artifact-block" * 1000) + b"tail"
    catalog, path = _catalog_for_file(tmp_path / "root", contents)
    path.write_bytes(contents)
    artifact = catalog.artifacts[0]

    verify_artifact_sha256(artifact, artifact_root=catalog.artifact_root, chunk_size=17)
    catalog.verify_sha256(artifact, chunk_size=31)


def test_opt_in_verification_rejects_digest_mismatch(tmp_path: Path) -> None:
    catalog, path = _catalog_for_file(tmp_path / "root", b"expected")
    path.write_bytes(b"different")

    with pytest.raises(ArtifactVerificationError, match="sha256 mismatch"):
        catalog.verify_sha256(catalog.artifacts[0])


def test_opt_in_verification_rejects_missing_file(tmp_path: Path) -> None:
    catalog, _ = _catalog_for_file(tmp_path / "root", b"missing")

    with pytest.raises(ArtifactVerificationError, match="could not inspect"):
        catalog.verify_sha256(catalog.artifacts[0])


def test_opt_in_verification_rejects_symlink_even_when_target_is_contained(
    tmp_path: Path,
) -> None:
    contents = b"contained-target"
    catalog, path = _catalog_for_file(tmp_path / "root", contents)
    target = path.parent / "target.so"
    target.write_bytes(contents)
    path.symlink_to(target.name)

    with pytest.raises(ArtifactVerificationError, match="contains a symlink"):
        catalog.verify_sha256(catalog.artifacts[0])


def test_opt_in_verification_rejects_symlink_directory_component(
    tmp_path: Path,
) -> None:
    contents = b"contained-target"
    root = tmp_path / "root"
    catalog, path = _catalog_for_file(root, contents)
    real_parent = root / "real-parent"
    real_parent.mkdir()
    (real_parent / path.name).write_bytes(contents)
    path.parent.rmdir()
    path.parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ArtifactVerificationError, match="contains a symlink"):
        catalog.verify_sha256(catalog.artifacts[0])


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support is unavailable")
def test_opt_in_verification_rejects_nonregular_file_without_reading(
    tmp_path: Path,
) -> None:
    catalog, path = _catalog_for_file(tmp_path / "root", b"fifo")
    os.mkfifo(path)

    with pytest.raises(ArtifactVerificationError, match="not a regular file"):
        catalog.verify_sha256(catalog.artifacts[0])


def test_artifact_catalog_schema_matches_v1_surface() -> None:
    schema_path = (
        Path(__file__).parents[1]
        / "registries/artifact_catalog/eiel_artifact_catalog_v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == "eiel_artifact_catalog_v1"
    assert set(schema["$defs"]["artifact"]["required"]) == set(_artifact())
    assert schema["$defs"]["artifact"]["properties"]["status"]["enum"] == [
        "candidate",
        "qualified",
        "deprecated",
        "quarantined",
        "retired",
    ]


def test_v2_selection_uses_all_compatibility_boundaries_deterministically(
    tmp_path: Path,
) -> None:
    exact = _artifact_v2("z-exact")
    lower_memory = _artifact_v2("a-lower-memory", memory_bytes=2_000_000_000)
    candidate = _artifact_v2("candidate", status="candidate")
    wrong_abi = _artifact_v2("wrong-abi")
    wrong_abi["abi"] = "other_abi"
    wrong_compute = _artifact_v2("wrong-compute")
    wrong_compute["architecture"] = "sm_86"
    wrong_compute["requirements"]["compute_capability"] = "sm_86"  # type: ignore[index]
    wrong_envelope = _artifact_v2("wrong-envelope")
    wrong_envelope["qualification"]["envelope_id"] = "other-envelope"  # type: ignore[index]
    too_large = _artifact_v2("too-large", memory_bytes=8_000_000_000)
    catalog = parse_artifact_catalog_v2(
        _catalog_v2(
            exact,
            lower_memory,
            candidate,
            wrong_abi,
            wrong_compute,
            wrong_envelope,
            too_large,
        ),
        artifact_root=tmp_path,
    )

    matches = catalog.filter_qualified(
        abi="eiel_native_runtime_abi_v1",
        compute_capability="sm_75",
        available_device_memory_bytes=6_000_000_000,
        mechanics_envelope_id="patch-p2-bounded-v1",
        required_capabilities=("resident_window",),
    )

    assert [artifact.artifact_id for artifact in matches] == [
        "a-lower-memory",
        "z-exact",
    ]


def test_v2_select_one_qualified_rejects_missing_and_ambiguous_matches(
    tmp_path: Path,
) -> None:
    catalog = parse_artifact_catalog_v2(
        _catalog_v2(_artifact_v2("one"), _artifact_v2("two")),
        artifact_root=tmp_path,
    )
    criteria = {
        "abi": "eiel_native_runtime_abi_v1",
        "compute_capability": "sm_75",
        "available_device_memory_bytes": 6_000_000_000,
        "mechanics_envelope_id": "patch-p2-bounded-v1",
    }

    with pytest.raises(ArtifactCatalogError, match="matched 2"):
        catalog.select_one_qualified(**criteria)
    with pytest.raises(ArtifactCatalogError, match="matched 0"):
        catalog.select_one_qualified(**(criteria | {"abi": "missing_abi"}))


def test_v2_select_one_verified_checks_digest_before_return(tmp_path: Path) -> None:
    contents = b"verified-v2-artifact"
    artifact = _artifact_v2("verified")
    artifact["sha256"] = hashlib.sha256(contents).hexdigest()
    root = tmp_path / "artifacts"
    catalog = parse_artifact_catalog_v2(_catalog_v2(artifact), artifact_root=root)
    path = root / str(artifact["relative_path"])
    path.parent.mkdir(parents=True)
    path.write_bytes(contents)
    criteria = {
        "abi": "eiel_native_runtime_abi_v1",
        "compute_capability": "sm_75",
        "available_device_memory_bytes": 4_000_000_000,
        "mechanics_envelope_id": "patch-p2-bounded-v1",
    }

    selected = catalog.select_one_verified(
        **criteria,
        verification_chunk_size=3,
    )

    assert selected.artifact_id == "verified"
    path.write_bytes(b"tampered")
    with pytest.raises(ArtifactVerificationError, match="sha256 mismatch"):
        catalog.select_one_verified(**criteria)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("compute_capability", "bad value", "invalid string"),
        ("minimum_device_memory_bytes", -1, "nonnegative integer"),
        ("minimum_device_memory_bytes", True, "nonnegative integer"),
    ],
)
def test_v2_requirements_are_strict(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    artifact = _artifact_v2()
    artifact["requirements"][field] = value  # type: ignore[index]

    with pytest.raises(ArtifactCatalogError, match=message):
        parse_artifact_catalog_v2(_catalog_v2(artifact), artifact_root=tmp_path)


def test_v2_architecture_must_match_compute_capability(tmp_path: Path) -> None:
    artifact = _artifact_v2()
    artifact["architecture"] = "sm_86"

    with pytest.raises(ArtifactCatalogError, match="must equal"):
        parse_artifact_catalog_v2(_catalog_v2(artifact), artifact_root=tmp_path)


@pytest.mark.parametrize(
    "available_memory",
    [-1, True, 1.5],
)
def test_v2_selection_rejects_invalid_available_memory(
    tmp_path: Path, available_memory: object
) -> None:
    catalog = parse_artifact_catalog_v2(
        _catalog_v2(_artifact_v2()), artifact_root=tmp_path
    )

    with pytest.raises(ArtifactCatalogError, match="nonnegative integer"):
        catalog.filter_qualified(
            abi="eiel_native_runtime_abi_v1",
            compute_capability="sm_75",
            available_device_memory_bytes=available_memory,  # type: ignore[arg-type]
            mechanics_envelope_id="patch-p2-bounded-v1",
        )


@pytest.mark.parametrize("capabilities", [None, 7, True])
def test_v2_selection_normalizes_invalid_capability_inputs(
    tmp_path: Path, capabilities: object
) -> None:
    catalog = parse_artifact_catalog_v2(
        _catalog_v2(_artifact_v2()), artifact_root=tmp_path
    )

    with pytest.raises(ArtifactCatalogError, match="sorted unique tokens"):
        catalog.filter_qualified(
            abi="eiel_native_runtime_abi_v1",
            compute_capability="sm_75",
            available_device_memory_bytes=6_000_000_000,
            mechanics_envelope_id="patch-p2-bounded-v1",
            required_capabilities=capabilities,  # type: ignore[arg-type]
        )


def test_v2_json_loader_and_schema_surface(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_catalog_v2(_artifact_v2())), encoding="utf-8")

    loaded = load_artifact_catalog_v2(
        catalog_path, artifact_root=tmp_path / "artifacts"
    )

    assert loaded.schema_version == "eiel_artifact_catalog_v2"
    schema_path = (
        Path(__file__).parents[1]
        / "registries/artifact_catalog/eiel_artifact_catalog_v2.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == "eiel_artifact_catalog_v2"
    assert set(schema["$defs"]["requirements"]["required"]) == {
        "compute_capability",
        "minimum_device_memory_bytes",
    }
