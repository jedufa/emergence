from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

from setuptools import find_packages
from setuptools.config.expand import find_packages as find_configured_packages


ROOT = Path(__file__).resolve().parents[2]


def test_package_discovery_matches_actual_layout() -> None:
    packages = set(find_packages(where=str(ROOT), include=["engine*", "tools*", "experiments*", "experiments_shared*"]))

    assert "engine" in packages
    assert "engine.simulation" in packages
    assert "tools" in packages
    assert "tools.evidence" in packages
    assert "experiments" in packages
    assert "experiments_shared" in packages
    assert packages


def test_configured_package_discovery_includes_legacy_engine_roots() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    config = data["tool"]["setuptools"]["packages"]["find"]

    packages = set(
        find_configured_packages(
            root_dir=ROOT,
            fill_package_dir={},
            **config,
        )
    )

    assert {"engine.evolution", "evolution", "simulation", "agents"} <= packages


def test_plain_python_from_repo_root_imports_legacy_engine_packages() -> None:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import engine; import agents, evolution, simulation; print(evolution.__file__)",
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert str(ROOT / "engine" / "evolution") in completed.stdout


def test_pyproject_does_not_discover_empty_emergence_namespace_only() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    include = data["tool"]["setuptools"]["packages"]["find"]["include"]

    assert include != ["emergence*"]
    assert "engine*" in include
    assert "tools*" in include
    assert "emergence" not in find_packages(
        where=str(ROOT), include=["emergence*"]
    )


def test_resident_owner_native_include_closure_is_package_data() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]
    package_root = ROOT / "engine/native_runtime"
    required_files = {
        package_root / "authoritative_state/binary_identity_constants_v1.hpp",
        package_root / "authoritative_state/binary_identity_v1.hpp",
        package_root / "wdl/wdl_cpu_oracle_c_api.h",
        package_root / "wdl/wdl_native_contract.cpp",
        package_root / "wdl/wdl_native_contract.hpp",
    }

    for package_alias in ("engine.native_runtime", "native_runtime"):
        patterns = [
            *package_data["*"],
            *package_data[package_alias],
        ]
        packaged_files = {
            path.resolve()
            for pattern in patterns
            for path in package_root.glob(pattern)
            if path.is_file()
        }
        assert {path.resolve() for path in required_files} <= packaged_files, (
            f"resident-owner native include closure is missing from package data "
            f"for {package_alias}"
        )
