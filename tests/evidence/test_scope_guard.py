from __future__ import annotations

import importlib

import tools.evidence as evidence


def test_evidence_package_scope_guard_constants() -> None:
    assert evidence.FORBIDDEN_SIMULATION_PATH_PREFIXES == (
        "engine/",
        "engine/simulation/",
        "engine/agents/",
        "engine/evolution/",
    )
    assert evidence.WRITABLE_TARGET_PATHS == ()
    for target in evidence.WRITABLE_TARGET_PATHS:
        assert not target.startswith(evidence.FORBIDDEN_SIMULATION_PATH_PREFIXES)


def test_evidence_modules_import_without_simulation_internals() -> None:
    for module_name in (
        "tools.evidence.manifest",
        "tools.evidence.profiles",
        "tools.evidence.contracts",
        "tools.evidence.schema",
        "tools.evidence.runs",
        "tools.evidence.anomalies",
        "tools.evidence.paired",
    ):
        module = importlib.import_module(module_name)
        assert module is not None
