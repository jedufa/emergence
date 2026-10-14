from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/scripts/eiel_phase_f_v2_registry.py"
CANONICAL = (
    ROOT / "_spec/evolutionary_capacity/phase_f/v2/forced_combination_matrix.json"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("eiel_phase_f_v2_registry", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _stdout_json(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    return json.loads(capsys.readouterr().out)


def test_validate_default_registry_reports_exact_planning_counts(capsys) -> None:
    module = _load_script()

    assert module.main(["validate"]) == 0
    summary = _stdout_json(capsys)

    assert summary == {
        "ok": True,
        "command": "validate",
        "path": str(CANONICAL),
        "matrix_sha256": "6afda0062cfc606e60d5d2f660eb59be65b576f4e233d3e2f172a8e114384275",
        "case_count": 120,
        "checkpoint_case_count": 14,
        "cpu_oracle_evaluations": 134,
        "accelerated_invocations": 268,
        "total_required_evaluations": 402,
        "first_case_id": "pf2_f01_b12_r0",
        "last_case_id": "pf2_f12_b16_r4",
        "planning_only": True,
    }


def test_compile_atomically_reproduces_tracked_canonical_bytes(
    tmp_path: Path, capsys
) -> None:
    module = _load_script()
    output = tmp_path / "nested/compiled.json"

    assert module.main(["compile", "--output", str(output)]) == 0
    summary = _stdout_json(capsys)

    assert output.read_bytes() == CANONICAL.read_bytes()
    assert summary["command"] == "compile"
    assert summary["path"] == str(output)
    assert summary["case_count"] == 120
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_independent_summary_rejects_case_order_and_digest_drift() -> None:
    module = _load_script()
    registry = module.build_phase_f_v2_matrix()
    registry["cases"].reverse()

    with pytest.raises(module.PhaseFV2RegistryCliError, match="ordering"):
        module._verified_summary(registry, command="validate", path=CANONICAL)

    registry = module.build_phase_f_v2_matrix()
    registry["matrix_sha256"] = "0" * 64
    with pytest.raises(module.PhaseFV2RegistryCliError, match="digest"):
        module._verified_summary(registry, command="validate", path=CANONICAL)


def test_validate_fails_closed_with_concise_json_error(tmp_path: Path, capsys) -> None:
    module = _load_script()
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema_version":"wrong"}\n', encoding="ascii")

    assert module.main(["validate", str(invalid)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["ok"] is False
    assert error["command"] == "validate"


def test_cli_surface_and_imports_remain_registry_only() -> None:
    module = _load_script()
    parser = module._parser()
    subparser_action = next(
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
    )
    assert set(subparser_action.choices) == {"compile", "validate"}

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint(
        {"subprocess", "torch", "runner", "provider", "eiel_phase_e_gpu_suite"}
    )


def test_cli_has_no_authorization_receipt_arguments() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "authorization-receipt" not in source
    assert "phase-e-authorization" not in source
