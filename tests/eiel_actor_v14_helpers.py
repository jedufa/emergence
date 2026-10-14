from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from tests.eiel_actor_v11_helpers import REPO_ROOT, run_eiel, write_preflight_receipt

FIXED_V14 = "2026-08-04T00:00:00Z"


def make_bootstrap_repo(tmp_path: Path) -> Path:
    root = tmp_path / "emergence"
    for directory in (
        "engine",
        "tools",
        "tests",
        "docs",
        "evidences",
        "tools/evidence/librarian/catalog_source",
        ".venv/bin",
    ):
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "engine/__init__.py").write_text("", encoding="utf-8")
    (root / "tools/__init__.py").write_text("", encoding="utf-8")
    (root / "AGENTS.md").write_text("# Test EIEL agent instructions\n", encoding="utf-8")
    (root / "emergence-dev.toml").write_text("[project]\nname='emergence'\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[build-system]\nrequires=['setuptools']\nbuild-backend='setuptools.build_meta'\n\n"
        "[project]\nname='emergence'\nversion='0.0.0'\n",
        encoding="utf-8",
    )
    catalog_link = root / "tools/evidence/librarian/catalog_source/00-catalog"
    catalog_link.symlink_to(REPO_ROOT / "tools/evidence/librarian/catalog_source/00-catalog", target_is_directory=True)
    python_link = root / ".venv/bin/python"
    python_link.symlink_to(Path(sys.executable))
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "EIEL Test"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "eiel-test@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "bootstrap fixture"], cwd=root, check=True)
    return root


def run_bootstrap(
    *,
    repo: Path,
    state_root: Path,
    expected: int | tuple[int, ...] = 0,
    task_mode: str = "implementation",
    gpu_required: str = "no",
    generated_at: str = FIXED_V14,
    acknowledge_dirty: bool = False,
    authority_domain: str | None = None,
    no_store: bool = False,
    edit_authorized: bool = True,
    write_scope: Iterable[str] = ("tools",),
    extra: Iterable[str] = (),
) -> dict[str, Any]:
    receipt = write_preflight_receipt(
        state_root / "fixture-preflight.json",
        topic="V1.4 work and session manager verification",
        authority_domain=authority_domain or "campaign_decision",
    )
    args = [
        "bootstrap",
        "inspect",
        "--repo-root",
        str(repo),
        "--task-mode",
        task_mode,
        "--topic",
        "V1.4 work and session manager verification",
        "--operator-instruction-ref",
        "test operator instruction",
        "--gpu-required",
        gpu_required,
        "--expected-validation",
        "pytest -q tests/test_eiel_work_manager_v1_4.py",
        "--preflight-receipt",
        str(receipt),
        "--state-root",
        str(state_root),
        "--generated-at",
        generated_at,
    ]
    if edit_authorized:
        args.append("--edit-authorized")
    for value in write_scope:
        args.extend(["--write-scope", str(value)])
    if authority_domain:
        args.extend(["--authority-domain", authority_domain])
    if no_store:
        args.append("--no-store")
    if acknowledge_dirty:
        args.append("--acknowledge-dirty")
    args.extend(extra)
    return run_eiel(args, state_root=state_root, expected=expected)


def create_work(
    state_root: Path,
    *,
    work_id: str,
    title: str,
    generated_at: str,
    status: str = "BACKLOG",
    dependency: Iterable[str] = (),
    priority: str = "P2",
    work_type: str = "ENGINEERING",
    next_action: str = "Continue bounded work.",
) -> dict[str, Any]:
    requested_status = status
    create_status = "ACTIVE" if status == "COMPLETED" else status
    args = [
        "workManager", "create",
        "--work-id", work_id,
        "--title", title,
        "--summary", title,
        "--work-type", work_type,
        "--priority", priority,
        "--status", create_status,
        "--completion-policy", "RECORD_ONLY",
        "--next-action", next_action,
        "--state-root", str(state_root),
        "--generated-at", generated_at,
    ]
    for value in dependency:
        args.extend(["--dependency", value])
    created = run_eiel(args, state_root=state_root)
    if requested_status == "COMPLETED":
        return run_eiel(
            [
                "workManager", "complete", work_id, "--state-root", str(state_root),
                "--generated-at", generated_at,
            ],
            state_root=state_root,
        )
    return created


def start_session(
    state_root: Path,
    *,
    repo: Path,
    work_id: str,
    bootstrap_id: str,
    codex_id: str,
    generated_at: str,
    allow_parallel: bool = False,
) -> dict[str, Any]:
    args = [
        "sessionManager", "start", work_id,
        "--bootstrap-record", bootstrap_id,
        "--codex-id", codex_id,
        "--repo-root", str(repo),
        "--stage", "implementation",
        "--current-step", "first bounded step",
        "--next-action", "Run the first bounded validation.",
        "--next-command", "pytest -q tests/test_eiel_workspace_v1_4.py",
        "--state-root", str(state_root),
        "--generated-at", generated_at,
    ]
    if allow_parallel:
        args.append("--allow-parallel")
    return run_eiel(args, state_root=state_root)
