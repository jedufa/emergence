"""Import-compat tests for engine -> engine migration."""

from __future__ import annotations

import importlib


def test_engine_main_alias_matches_legacy() -> None:
    new = importlib.import_module("engine.main")
    legacy = importlib.import_module("engine.main")
    assert callable(new.main)
    assert callable(legacy.main)
    assert new.main.__name__ == legacy.main.__name__


def test_engine_subpackage_alias_loads_legacy_module() -> None:
    new_sim = importlib.import_module("engine.simulation")
    legacy_sim = importlib.import_module("engine.simulation")
    assert str(getattr(new_sim, "__file__", "")).endswith("simulation/__init__.py")
    assert str(getattr(legacy_sim, "__file__", "")).endswith("simulation/__init__.py")


def test_python_m_engine_package_imports() -> None:
    pkg = importlib.import_module("engine")
    assert hasattr(pkg, "__path__")
