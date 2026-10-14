from __future__ import annotations

import ast
from pathlib import Path

from tests.eiel_actor_v11_helpers import REPO_ROOT
from tools.actors import qualification_manager


ACTORS = REPO_ROOT / "tools/actors"


def test_actor_managers_do_not_import_production_runner_or_campaign_executor() -> None:
    forbidden_prefixes = (
        "engine.simulation.runner",
        "engine.campaigns.execution",
        "_runner",
    )
    violations: list[str] = []
    for path in sorted(ACTORS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden_prefixes):
                        violations.append(f"{path.name}:{node.lineno}:{alias.name}")
            if module and module.startswith(forbidden_prefixes):
                violations.append(f"{path.name}:{node.lineno}:{module}")
    assert violations == []


def test_actor_subprocess_use_is_read_only_and_allowlisted() -> None:
    violations: list[str] = []
    for path in sorted(ACTORS.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        direct_calls: list[ast.Call] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "subprocess"
                and function.attr in {"run", "Popen", "call", "check_call", "check_output"}
            ):
                direct_calls.append(node)

        if path.name in {"evidence_manager.py", "session_manager.py", "context_receipt.py", "work_package.py"}:
            for node in direct_calls:
                function = node.func
                if not isinstance(function, ast.Attribute) or function.attr != "run":
                    violations.append(f"{path.name}:{node.lineno}:subprocess.{getattr(function, 'attr', '?')}")
                    continue
                first = node.args[0] if node.args else None
                if not (
                    isinstance(first, ast.List)
                    and first.elts
                    and isinstance(first.elts[0], ast.Constant)
                    and first.elts[0].value == "git"
                ):
                    violations.append(f"{path.name}:{node.lineno}:non-git subprocess")
            continue

        if path.name == "bridge_manager.py":
            # The optional bridge facade may probe Node and delegate only after
            # an explicit bridgeManager command. It is not imported as execution
            # authority by ordinary actor workflows.
            for node in direct_calls:
                function = node.func
                if not isinstance(function, ast.Attribute) or function.attr != "run":
                    violations.append(f"{path.name}:{node.lineno}:subprocess.{getattr(function, 'attr', '?')}")
            text = path.read_text(encoding="utf-8")
            if "automatic_execution\": False" not in text or "required\": False" not in text:
                violations.append(f"{path.name}:optional-boundary-markers-missing")
            continue

        if path.name == "resource_accounting.py":
            # WRA-1 may query the operator-provided ccusage CLI for token
            # accounting evidence. It must remain bounded, shell-free, and
            # non-campaign execution.
            if len(direct_calls) != 1:
                violations.append(f"{path.name}:unexpected subprocess count")
            for node in direct_calls:
                function = node.func
                if not isinstance(function, ast.Attribute) or function.attr != "run":
                    violations.append(f"{path.name}:{node.lineno}:subprocess.{getattr(function, 'attr', '?')}")
                keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                shell = keywords.get("shell")
                if shell is not None and not (isinstance(shell, ast.Constant) and shell.value is False):
                    violations.append(f"{path.name}:{node.lineno}:shell-enabled")
                check = keywords.get("check")
                if not (isinstance(check, ast.Constant) and check.value is False):
                    violations.append(f"{path.name}:{node.lineno}:check-not-false")
                if "timeout" not in keywords or "capture_output" not in keywords:
                    violations.append(f"{path.name}:{node.lineno}:missing-bounds")
            if "CCUSAGE_COMMAND_ENV" not in text or "DEFAULT_CCUSAGE_COMMAND" not in text:
                violations.append(f"{path.name}:ccusage-boundary-markers-missing")
            continue

        if path.name == "bootstrap.py":
            # Bootstrap owns one bounded subprocess wrapper used only for Git
            # discovery and an explicit read-only Python '-B -c' runtime probe.
            if len(direct_calls) != 1 or not isinstance(direct_calls[0].func, ast.Attribute) or direct_calls[0].func.attr != "run":
                violations.append(f"{path.name}:unexpected direct subprocess surface")
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_run"):
                    continue
                first = node.args[0] if node.args else None
                if not isinstance(first, ast.List) or not first.elts:
                    violations.append(f"{path.name}:{node.lineno}:dynamic _run command")
                    continue
                head = first.elts[0]
                is_git = isinstance(head, ast.Constant) and head.value == "git"
                is_python_probe = (
                    isinstance(head, ast.Call)
                    and isinstance(head.func, ast.Name)
                    and head.func.id == "str"
                    and len(first.elts) >= 3
                    and isinstance(first.elts[1], ast.Constant)
                    and first.elts[1].value == "-B"
                    and isinstance(first.elts[2], ast.Constant)
                    and first.elts[2].value == "-c"
                )
                if not (is_git or is_python_probe):
                    violations.append(f"{path.name}:{node.lineno}:non-read-only _run command")
            continue

        for node in direct_calls:
            function = node.func
            violations.append(f"{path.name}:{node.lineno}:subprocess.{getattr(function, 'attr', '?')}")
    assert violations == []


def test_mission_and_qualification_handoffs_preserve_existing_campaign_owner() -> None:
    boundary = qualification_manager.integration_boundary()
    assert boundary["campaign_route_qualification_cli"] == (
        "python -m tools.scripts.eiel_campaign qualify"
    )
    guide = (ACTORS / "CODEX_OPERATOR_GUIDE.md").read_text(encoding="utf-8")
    assert "python -m tools.scripts.eiel_campaign preflight" in guide
    assert "python -m tools.scripts.eiel_campaign run" in guide
    assert "None of these four actor CLIs launches the campaign" in guide


def test_actor_cli_has_no_run_launch_schedule_or_kill_command() -> None:
    for command in ("missionPlanner", "qualificationManager", "resourceManager", "evidenceManager"):
        source = (REPO_ROOT / "tools/eiel.py").read_text(encoding="utf-8")
        assert command in source
    forbidden_commands = {
        "missionPlanner": {"run", "launch", "execute", "schedule"},
        "qualificationManager": {"run", "launch", "execute"},
        "resourceManager": {"run", "launch", "reserve", "schedule", "kill", "cancel"},
        "evidenceManager": {"run", "launch", "rerun", "tune"},
    }
    import subprocess
    import json
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    for actor, operations in forbidden_commands.items():
        for operation in operations:
            result = subprocess.run(
                [str(REPO_ROOT / "tools/bin/eiel"), actor, operation],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            assert result.returncode != 0
            payload = json.loads(result.stdout or result.stderr)
            assert payload["ok"] is False
