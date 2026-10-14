from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACTORS = ROOT / "tools" / "actors"

BANNED_IMPORT_ROOTS = {
    "openai", "anthropic", "langchain", "transformers",
    "requests", "httpx", "aiohttp", "socket",
    "fastapi", "flask", "django",
    "sqlite3", "sqlalchemy", "psycopg", "redis",
    "celery", "rq", "kafka",
}


def test_actor_modules_have_no_ai_network_service_or_database_dependency() -> None:
    violations: list[str] = []
    for path in sorted(ACTORS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".", 1)[0]
                if root in BANNED_IMPORT_ROOTS:
                    violations.append(f"{path.relative_to(ROOT)} imports {name}")
    assert violations == []


def test_public_actor_modules_do_not_define_campaign_launch_or_process_control_apis() -> None:
    forbidden_function_names = {
        "launch_campaign", "execute_campaign", "schedule_campaign",
        "reserve_resources", "kill_process", "cancel_process",
        "start_daemon", "serve_forever",
    }
    found: list[str] = []
    for path in sorted(ACTORS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden_function_names:
                found.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert found == []
