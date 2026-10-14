from __future__ import annotations

import ast
from pathlib import Path

from engine.evolution.phase_f_v2.receipts import create_matrix_registry_receipt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OWNED_MODULES = (
    PROJECT_ROOT / "engine/evolution/phase_f_v2/__init__.py",
    PROJECT_ROOT / "engine/evolution/phase_f_v2/canonical.py",
    PROJECT_ROOT / "engine/evolution/phase_f_v2/matrix.py",
    PROJECT_ROOT / "engine/evolution/phase_f_v2/receipts.py",
    PROJECT_ROOT / "engine/evolution/phase_f_v2/storage.py",
)
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "cupy",
        "numba",
        "subprocess",
        "torch",
    }
)
FORBIDDEN_IMPORT_FRAGMENTS = (
    "campaigns.orchestration",
    "campaigns.preflight",
    "campaigns.providers",
    "native_runtime",
    "simulation.runner",
)
FORBIDDEN_CALLS = frozenset({"__import__", "compile", "eval", "exec"})
FORBIDDEN_ATTRIBUTES = frozenset(
    {
        "Popen",
        "call",
        "check_call",
        "check_output",
        "execv",
        "execve",
        "popen",
        "run",
        "spawnl",
        "spawnle",
        "spawnlp",
        "spawnlpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "system",
    }
)


def _import_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_phase_f_receipt_slice_has_no_execution_dispatch_or_accelerator_imports() -> None:
    for path in OWNED_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = _import_names(tree)
        assert not [
            name
            for name in imports
            if name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS
            or any(fragment in name for fragment in FORBIDDEN_IMPORT_FRAGMENTS)
        ]


def test_phase_f_receipt_slice_cannot_evaluate_or_launch_commands() -> None:
    for path in OWNED_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                assert node.func.id not in FORBIDDEN_CALLS
            elif isinstance(node.func, ast.Attribute):
                assert node.func.attr not in FORBIDDEN_ATTRIBUTES


def test_matrix_receipt_is_data_only_and_contains_no_command_surface() -> None:
    receipt = create_matrix_registry_receipt(
        source_commit="1" * 40,
        preregistration_sha256="a" * 64,
        phase_e_authorization_sha256="b" * 64,
        matrix_spec_sha256="c" * 64,
        fixture_manifest_sha256="d" * 64,
        cell_registry_sha256="e" * 64,
    )

    assert receipt["authority"] == "registry_only"
    assert receipt["no_launch"] is True
    assert receipt["dispatch_started"] is False
    assert receipt["execution_started"] is False
    assert not {
        "argv",
        "command",
        "commands",
        "device",
        "environment",
        "executable",
        "provider",
        "route",
    } & set(receipt)
