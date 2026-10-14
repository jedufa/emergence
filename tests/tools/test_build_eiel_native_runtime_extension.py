from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from perf import cxxbackbone10_resident_state_owner as development_loader
from tools.scripts import build_eiel_native_runtime_extension as aot_builder
from tools.scripts import build_eiel_resident_state_owner_extension as owner_builder


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fingerprint(records: list[dict[str, str]]) -> str:
    payload = json.dumps(records, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_native_runtime_source_inventories_include_cpu_compact_producers() -> None:
    expected_encoder = aot_builder.CPU_COMPACT_SUMMARY_SOURCE.resolve()
    expected_digests = aot_builder.CPU_COMPACT_SUMMARY_DIGESTS_SOURCE.resolve()
    expected_torch_adapter = (
        aot_builder.CPU_COMPACT_SUMMARY_TORCH_ADAPTER_SOURCE.resolve()
    )
    expected_lifecycle_reduction = (
        aot_builder.CPU_COMPACT_SUMMARY_LIFECYCLE_REDUCTION_SOURCE.resolve()
    )
    expected_identity_reduction = (
        aot_builder.CPU_COMPACT_SUMMARY_IDENTITY_REDUCTION_SOURCE.resolve()
    )
    expected_reporting_ledger = (
        aot_builder.CPU_COMPACT_SUMMARY_REPORTING_LEDGER_SOURCE.resolve()
    )
    expected_reporting_ledger_header = (
        aot_builder.CPU_COMPACT_SUMMARY_REPORTING_LEDGER_HEADER.resolve()
    )
    expected_source_composer = (
        aot_builder.CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE.resolve()
    )
    expected_host_canonical = (
        aot_builder.CPU_COMPACT_SUMMARY_HOST_CANONICAL_SOURCE.resolve()
    )

    assert tuple(path.name for path in aot_builder.NATIVE_RUNTIME_COMPILE_SOURCES) == (
        "eiel_native_runtime.cpp",
        "cpu_compact_summary_v1.cpp",
        "cpu_compact_summary_digests_v1.cpp",
        "cpu_compact_summary_torch_adapter_v1.cpp",
        "cpu_compact_summary_lifecycle_reduction_v1.cpp",
        "cpu_compact_summary_identity_reduction_v1.cpp",
        "cpu_compact_summary_reporting_ledger_v1.cpp",
        "cpu_compact_summary_source_composer_v1.cpp",
        "cuda_architecture_b_compact_summary_host_canonical_v1.cpp",
        "acs_native_streamed_checkpoint_v1.cpp",
        "wdl_native_contract.cpp",
        "stable_gru_v1.cpp",
    )
    assert tuple(path.name for path in aot_builder.NATIVE_RUNTIME_SOURCE_INVENTORY) == (
        "eiel_native_runtime.cpp",
        "cxxbackbone10_resident_state_owner.cpp",
        "cpu_compact_summary_v1.cpp",
        "cpu_compact_summary_digests_v1.cpp",
        "cpu_compact_summary_torch_adapter_v1.cpp",
        "cpu_compact_summary_lifecycle_reduction_v1.cpp",
        "cpu_compact_summary_identity_reduction_v1.cpp",
        "cpu_compact_summary_reporting_ledger_v1.cpp",
        "cpu_compact_summary_reporting_ledger_v1.hpp",
        "cpu_compact_summary_source_composer_v1.cpp",
        "cuda_architecture_b_compact_summary_host_canonical_v1.cpp",
        "acs_native_streamed_checkpoint_v1.cpp",
        "acs_native_streamed_checkpoint_v1.hpp",
        "wdl_native_contract.cpp",
        "stable_gru_v1.cpp",
    )
    assert tuple(path.name for path in development_loader.NATIVE_EXTENSION_SOURCES) == (
        "cxxbackbone10_resident_state_owner.cpp",
        "cpu_compact_summary_v1.cpp",
        "cpu_compact_summary_digests_v1.cpp",
        "cpu_compact_summary_torch_adapter_v1.cpp",
        "cpu_compact_summary_lifecycle_reduction_v1.cpp",
        "cpu_compact_summary_identity_reduction_v1.cpp",
        "cpu_compact_summary_reporting_ledger_v1.cpp",
        "cpu_compact_summary_source_composer_v1.cpp",
        "cuda_architecture_b_compact_summary_host_canonical_v1.cpp",
        "acs_native_streamed_checkpoint_v1.cpp",
        "wdl_native_contract.cpp",
        "stable_gru_v1.cpp",
    )
    assert tuple(path.name for path in aot_builder.RESIDENT_OWNER_SOURCE_INVENTORY) == (
        "cxxbackbone10_resident_state_owner.cpp",
        "cpu_compact_summary_v1.cpp",
        "cpu_compact_summary_digests_v1.cpp",
        "cpu_compact_summary_torch_adapter_v1.cpp",
        "cpu_compact_summary_lifecycle_reduction_v1.cpp",
        "cpu_compact_summary_identity_reduction_v1.cpp",
        "cpu_compact_summary_reporting_ledger_v1.cpp",
        "cpu_compact_summary_source_composer_v1.cpp",
        "cuda_architecture_b_compact_summary_host_canonical_v1.cpp",
        "acs_native_streamed_checkpoint_v1.cpp",
        "wdl_native_contract.cpp",
        "stable_gru_v1.cpp",
        "cpu_compact_summary_reporting_ledger_v1.hpp",
        "acs_native_streamed_checkpoint_v1.hpp",
    )
    assert development_loader.CPU_COMPACT_SUMMARY_SOURCE.resolve() == expected_encoder
    assert (
        development_loader.CPU_COMPACT_SUMMARY_DIGESTS_SOURCE.resolve()
        == expected_digests
    )
    assert (
        development_loader.CPU_COMPACT_SUMMARY_TORCH_ADAPTER_SOURCE.resolve()
        == expected_torch_adapter
    )
    assert (
        development_loader.CPU_COMPACT_SUMMARY_LIFECYCLE_REDUCTION_SOURCE.resolve()
        == expected_lifecycle_reduction
    )
    assert (
        development_loader.CPU_COMPACT_SUMMARY_IDENTITY_REDUCTION_SOURCE.resolve()
        == expected_identity_reduction
    )
    assert (
        development_loader.CPU_COMPACT_SUMMARY_REPORTING_LEDGER_SOURCE.resolve()
        == expected_reporting_ledger
    )
    assert (
        development_loader.CPU_COMPACT_SUMMARY_REPORTING_LEDGER_HEADER.resolve()
        == expected_reporting_ledger_header
    )
    assert (
        development_loader.CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE.resolve()
        == expected_source_composer
    )
    assert (
        development_loader.CPU_COMPACT_SUMMARY_HOST_CANONICAL_SOURCE.resolve()
        == expected_host_canonical
    )
    assert all(path.is_file() for path in aot_builder.NATIVE_RUNTIME_SOURCE_INVENTORY)


def test_aot_source_records_and_fingerprints_are_content_truthful() -> None:
    compile_records = aot_builder._source_records(aot_builder.NATIVE_RUNTIME_COMPILE_SOURCES)
    inventory_records = aot_builder._source_records(aot_builder.NATIVE_RUNTIME_SOURCE_INVENTORY)
    resident_owner_inventory_records = aot_builder._source_records(
        aot_builder.RESIDENT_OWNER_SOURCE_INVENTORY
    )

    for record in (
        *compile_records,
        *inventory_records,
        *resident_owner_inventory_records,
    ):
        path = Path(record["path"])
        assert record["sha256"] == _sha256(path)
    assert aot_builder._source_fingerprint(compile_records) == _fingerprint(compile_records)
    assert aot_builder._source_fingerprint(inventory_records) == _fingerprint(inventory_records)
    assert aot_builder._source_fingerprint(
        resident_owner_inventory_records
    ) == _fingerprint(resident_owner_inventory_records)


def test_aot_source_provenance_fails_closed_on_concurrent_change(tmp_path: Path) -> None:
    source = tmp_path / "source.cpp"
    source.write_text("source-v1\n", encoding="utf-8")
    records = aot_builder._source_records((source,))

    aot_builder._require_source_records_unchanged((source,), records)
    source.write_text("source-v2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="sources changed during build"):
        aot_builder._require_source_records_unchanged((source,), records)


def test_aot_build_manifest_records_source_composer_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_dir = tmp_path / "build"
    output = tmp_path / "eiel_native_runtime_ext.so"
    output.write_bytes(b"bounded-test-extension\n")

    monkeypatch.setattr(
        aot_builder,
        "_load_native_runtime_extension",
        lambda **_kwargs: SimpleNamespace(__file__=str(output)),
    )
    monkeypatch.setattr(
        aot_builder.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="bounded-test-compiler"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_eiel_native_runtime_extension.py",
            "--build-dir",
            str(build_dir),
            "--output",
            str(output),
        ],
    )

    assert aot_builder.main() == 0
    manifest = json.loads(
        output.with_suffix(output.suffix + ".manifest.json").read_text(
            encoding="utf-8"
        )
    )
    composer = aot_builder.CPU_COMPACT_SUMMARY_SOURCE_COMPOSER_SOURCE.resolve()
    composer_record = {"path": str(composer), "sha256": _sha256(composer)}
    resident_owner_inventory = aot_builder._source_records(
        aot_builder.RESIDENT_OWNER_SOURCE_INVENTORY
    )
    native_runtime_inventory = aot_builder._source_records(
        aot_builder.NATIVE_RUNTIME_SOURCE_INVENTORY
    )
    development_manifest = development_loader.native_extension_compile_source_manifest()

    assert manifest["cpu_compact_summary_source_composer_source"] == str(composer)
    assert (
        manifest["cpu_compact_summary_source_composer_source_sha256"]
        == composer_record["sha256"]
    )
    assert composer_record in manifest["compile_sources"]
    assert composer_record in manifest["source_inventory"]
    assert manifest["resident_owner_source_inventory"] == resident_owner_inventory
    assert manifest["source_inventory_fingerprint_sha256"] == development_manifest[
        "compile_source_fingerprint_sha256"
    ]
    assert manifest["source_inventory_fingerprint_sha256"] == _fingerprint(
        resident_owner_inventory
    )
    assert manifest["native_runtime_source_inventory_fingerprint_sha256"] == _fingerprint(
        native_runtime_inventory
    )


def test_aot_loader_passes_every_compile_source_to_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_load(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(aot_builder, "load", fake_load)
    aot_builder._load_native_runtime_extension(
        module_name="bounded_native_extension",
        build_dir=tmp_path,
        numpy_random_lib=tmp_path / "missing-libnpyrandom.a",
    )

    assert captured["sources"] == [
        str(path) for path in aot_builder.NATIVE_RUNTIME_COMPILE_SOURCES
    ]
    assert captured["with_cuda"] is False


def test_development_loader_source_manifest_and_torch_inputs_are_truthful(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = development_loader.native_extension_compile_source_manifest()
    records = manifest["sources"]
    assert isinstance(records, list)
    assert records == [
        {"path": str(path.resolve()), "sha256": _sha256(path)}
        for path in development_loader.NATIVE_EXTENSION_SOURCE_INVENTORY
    ]
    assert manifest["compile_source_fingerprint_sha256"] == _fingerprint(records)

    captured: dict[str, object] = {}
    sentinel = object()

    def fake_load(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(development_loader, "_MODULE", None)
    monkeypatch.delenv(development_loader.RESIDENT_STATE_OWNER_EXTENSION_ENV, raising=False)
    monkeypatch.setattr(development_loader, "load", fake_load)
    loaded = development_loader.load_resident_state_owner_extension(
        build_directory=tmp_path,
    )

    assert loaded is sentinel
    assert captured["sources"] == [
        str(path) for path in development_loader.NATIVE_EXTENSION_SOURCES
    ]
    assert captured["with_cuda"] is False


def test_development_loader_status_fails_closed_without_prebuilt_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(development_loader, "_MODULE", None)
    monkeypatch.delenv(development_loader.RESIDENT_STATE_OWNER_EXTENSION_ENV, raising=False)

    status = development_loader.resident_state_owner_extension_status()

    assert status.available is False
    assert status.runtime_jit is False
    assert status.loaded is False
    assert development_loader.RESIDENT_STATE_OWNER_EXTENSION_ENV in status.error


def test_development_loader_uses_verified_prebuilt_extension_when_env_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "resident_owner.so"
    artifact.write_bytes(b"fake prebuilt extension")
    manifest = development_loader.native_extension_compile_source_manifest()
    artifact.with_suffix(artifact.suffix + ".manifest.json").write_text(
        json.dumps(
            {
                "stable_module_name": development_loader.RESIDENT_STATE_OWNER_STABLE_MODULE_NAME,
                "extension_abi_version": "eiel_native_runtime_abi_v1",
                "extension_version": "eiel_native_runtime_flat_v1",
                "source_inventory_fingerprint_sha256": manifest[
                    "compile_source_fingerprint_sha256"
                ],
                "output_sha256": _sha256(artifact),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_prebuilt_loader(path: str) -> object:
        captured["path"] = path
        return sentinel

    def forbidden_jit_loader(**_: object) -> object:
        raise AssertionError("development JIT loader should not run for prebuilt path")

    monkeypatch.setattr(development_loader, "_MODULE", None)
    monkeypatch.setenv(
        development_loader.RESIDENT_STATE_OWNER_EXTENSION_ENV,
        str(artifact),
    )
    monkeypatch.setattr(
        development_loader,
        "_load_prebuilt_resident_state_owner_extension",
        fake_prebuilt_loader,
    )
    monkeypatch.setattr(development_loader, "load", forbidden_jit_loader)

    status = development_loader.resident_state_owner_extension_status()
    loaded = development_loader.load_resident_state_owner_extension(
        build_directory=tmp_path / "unused-build",
    )

    assert status.available is True
    assert status.runtime_jit is False
    assert status.output_sha256_verified is True
    assert status.source_inventory_verified is True
    assert loaded is sentinel
    assert captured == {"path": str(artifact)}


def test_resident_owner_builder_manifest_matches_loader_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_dir = tmp_path / "build"
    output = tmp_path / "resident_owner_ext.so"
    output.write_bytes(b"resident-owner-prebuilt\n")

    monkeypatch.setattr(
        owner_builder,
        "_load_extension",
        lambda **_kwargs: SimpleNamespace(__file__=str(output)),
    )
    monkeypatch.setattr(
        owner_builder,
        "_validate_build_ninja",
        lambda _build_dir: {"build_ninja_verified": True},
    )
    monkeypatch.setattr(
        owner_builder.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="bounded-test-compiler"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_eiel_resident_state_owner_extension.py",
            "--build-dir",
            str(build_dir),
            "--output",
            str(output),
            "--module-name",
            "resident_owner_test_ext",
        ],
    )

    assert owner_builder.main() == 0
    stdout = json.loads(capsys.readouterr().out)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_manifest = development_loader.native_extension_compile_source_manifest()

    assert stdout["manifest_path"] == str(manifest_path)
    assert manifest["schema_version"] == owner_builder.SCHEMA_VERSION
    assert manifest["stable_module_name"] == "resident_owner_test_ext"
    assert manifest["extension_abi_version"] == "eiel_native_runtime_abi_v1"
    assert manifest["extension_version"] == "eiel_native_runtime_flat_v1"
    assert manifest["source_inventory_fingerprint_sha256"] == source_manifest[
        "compile_source_fingerprint_sha256"
    ]
    assert manifest["output_sha256"] == _sha256(output)
    assert manifest["runtime_jit_in_measured_execution"] is False
    assert manifest["with_cuda"] is False
    assert manifest["sources"] == source_manifest["sources"]
