from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import native_runtime.bindings.extension as extension_module
from native_runtime.types import ABI_VERSION, EXTENSION_VERSION


_MISSING = object()
_STABLE_MODULE_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@pytest.fixture(autouse=True)
def _isolate_prebuilt_module_cache() -> Iterator[None]:
    extension_module._PREBUILT_MODULES.clear()
    yield
    extension_module._PREBUILT_MODULES.clear()


def _write_prebuilt(
    tmp_path: Path,
    filename: str,
    *,
    stable_module_name: object = _MISSING,
) -> Path:
    extension_path = tmp_path / filename
    extension_path.write_bytes(f"mock extension: {filename}".encode())
    manifest: dict[str, object] = {
        "extension_abi_version": ABI_VERSION,
        "extension_version": EXTENSION_VERSION,
        "output_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
    }
    if stable_module_name is not _MISSING:
        manifest["stable_module_name"] = stable_module_name
    extension_path.with_suffix(extension_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return extension_path


def _install_fake_extension_import(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    spec_calls: list[tuple[str, str]] = []

    class FakeLoader:
        def exec_module(self, module: ModuleType) -> None:
            module.EIEL_NATIVE_RUNTIME_ABI_VERSION = ABI_VERSION
            module.EIEL_NATIVE_RUNTIME_EXTENSION_VERSION = EXTENSION_VERSION

    def fake_spec_from_file_location(module_name: str, path: str) -> SimpleNamespace:
        spec_calls.append((module_name, path))
        return SimpleNamespace(name=module_name, loader=FakeLoader())

    def fake_module_from_spec(spec: SimpleNamespace) -> ModuleType:
        return ModuleType(spec.name)

    monkeypatch.setattr(extension_module.importlib.util, "spec_from_file_location", fake_spec_from_file_location)
    monkeypatch.setattr(extension_module.importlib.util, "module_from_spec", fake_module_from_spec)
    return spec_calls


def test_loads_two_prebuilt_extensions_under_their_distinct_manifest_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_path = _write_prebuilt(
        tmp_path,
        "first.so",
        stable_module_name="eiel_native_runtime_ext_first",
    )
    second_path = _write_prebuilt(
        tmp_path,
        "second.so",
        stable_module_name="eiel_native_runtime_ext_second",
    )
    spec_calls = _install_fake_extension_import(monkeypatch)

    first = extension_module.load_prebuilt_extension(first_path)
    second = extension_module.load_prebuilt_extension(second_path)

    assert first is not second
    assert first.__name__ == "eiel_native_runtime_ext_first"
    assert second.__name__ == "eiel_native_runtime_ext_second"
    assert spec_calls == [
        ("eiel_native_runtime_ext_first", str(first_path)),
        ("eiel_native_runtime_ext_second", str(second_path)),
    ]
    assert extension_module.load_prebuilt_extension(first_path) is first
    assert extension_module.load_prebuilt_extension(second_path) is second
    assert len(spec_calls) == 2


def test_distinct_prebuilt_paths_cannot_claim_the_same_stable_module_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable_module_name = "eiel_native_runtime_ext_shared"
    first_path = _write_prebuilt(tmp_path, "first.so", stable_module_name=stable_module_name)
    second_path = _write_prebuilt(tmp_path, "second.so", stable_module_name=stable_module_name)
    spec_calls = _install_fake_extension_import(monkeypatch)

    first = extension_module.load_prebuilt_extension(first_path)

    with pytest.raises(RuntimeError) as exc_info:
        extension_module.load_prebuilt_extension(second_path)

    assert str(exc_info.value) == (
        f"stable module name {stable_module_name!r} is already loaded from {first_path.resolve()}; "
        f"refusing distinct extension {second_path.resolve()}"
    )
    assert spec_calls == [(stable_module_name, str(first_path.resolve()))]
    assert extension_module.load_prebuilt_extension(first_path) is first


@pytest.mark.parametrize("alias_first", [False, True], ids=["real-path-first", "alias-first"])
def test_same_resolved_prebuilt_file_alias_is_imported_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alias_first: bool,
) -> None:
    stable_module_name = "eiel_native_runtime_ext_alias"
    extension_path = _write_prebuilt(tmp_path, "candidate.so", stable_module_name=stable_module_name)
    alias_path = tmp_path / "candidate-alias.so"
    alias_path.symlink_to(extension_path)
    spec_calls = _install_fake_extension_import(monkeypatch)
    first_path, second_path = (alias_path, extension_path) if alias_first else (extension_path, alias_path)

    first = extension_module.load_prebuilt_extension(first_path)
    second = extension_module.load_prebuilt_extension(second_path)

    assert first is second
    assert spec_calls == [(stable_module_name, str(extension_path.resolve()))]
    assert extension_module.native_extension_status(extension_path).loaded is True
    assert extension_module.native_extension_status(alias_path).loaded is True
    assert list(extension_module._PREBUILT_MODULES) == [str(extension_path.resolve())]


@pytest.mark.parametrize(
    "stable_module_name",
    [
        pytest.param(_MISSING, id="missing"),
        pytest.param("invalid-module-name", id="punctuation"),
        pytest.param("eiel_native_runtime_ext_\N{LATIN SMALL LETTER E WITH ACUTE}", id="non-ascii"),
    ],
)
def test_invalid_or_missing_stable_module_name_fails_closed_before_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stable_module_name: object,
) -> None:
    extension_path = _write_prebuilt(
        tmp_path,
        "candidate.so",
        stable_module_name=stable_module_name,
    )
    import_attempted = False

    def forbidden_spec_from_file_location(module_name: str, path: str) -> None:
        nonlocal import_attempted
        import_attempted = True
        return None

    monkeypatch.setattr(
        extension_module.importlib.util,
        "spec_from_file_location",
        forbidden_spec_from_file_location,
    )

    status = extension_module.native_extension_status(extension_path)

    assert status.available is False
    assert status.error == "configured extension build manifest has an invalid stable module name"
    with pytest.raises(RuntimeError, match="invalid stable module name"):
        extension_module.load_prebuilt_extension(extension_path)
    assert import_attempted is False
    if stable_module_name is not _MISSING:
        assert _STABLE_MODULE_NAME_PATTERN.fullmatch(str(stable_module_name)) is None
