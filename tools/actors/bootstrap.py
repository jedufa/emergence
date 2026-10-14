"""Fail-closed repository and environment bootstrap for EIEL Codex sessions.

The bootstrap is read-first.  It verifies repository identity, Python/Torch/CUDA
readiness, current instructions, Knowledge Librarian preflight, declared task
scope, and required validation.  It never creates environments, installs
packages, edits source, launches campaigns, or repairs the host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.canonical import canonical_sha256
from tools.actors.authority import (
    compact_preflight_receipt,
    retain_preflight_receipt,
    run_librarian_preflight,
    validate_preflight_receipt,
)
from tools.actors.common import (
    ActorArgumentParser,
    ActorError,
    actor_state_root,
    emit_actor_error,
    emit_payload,
    file_sha256,
    load_actor_record,
    new_record,
    read_json,
    require_explicit_schema_version,
    require_known_fields,
    require_test_fixture_context,
    render_eiel_command,
    repository_root,
    stable_id,
    timestamp_age_hours,
    utc_now,
    with_guidance,
)
from tools.evidence.librarian import wp4
from tools.actors.instruction_binding import bind_operator_instruction, instruction_is_authoritative
from tools.actors.portfolio import (
    BOOTSTRAP_ID_PREFIX,
    BOOTSTRAP_RECORD_TYPE,
    bootstrap_store,
    latest_bootstrap_record,
    validate_prefixed_id,
)

ACTOR_NAME = "bootstrap"
BOOTSTRAP_SCHEMA_VERSION = "eiel_repository_bootstrap_v1"
BOOTSTRAP_SOURCE_RELATIVE = "tools/bootstrap/EIEL_Bounded_Executor_Bootstrap_v0_9.md"
TASK_MODES = (
    "review",
    "diagnosis",
    "implementation",
    "campaign",
    "qualification",
    "documentation",
    "evidence",
    "mechanical",
)
SUBSTANTIVE_TASK_MODES = {
    "review",
    "diagnosis",
    "implementation",
    "campaign",
    "qualification",
    "documentation",
    "evidence",
}
EDITING_TASK_MODES = {"implementation", "campaign", "qualification", "documentation", "evidence"}
GPU_REQUIRING_MODES = {"campaign", "qualification"}
READY_STATUSES = {"READY", "READY_WITH_WARNINGS"}
REPOSITORY_STATE_SCHEMA_VERSION = "eiel_repository_state_v2"
WORKING_STATE_SCHEMA_VERSION = "eiel_working_state_v2"
ENVIRONMENT_FINGERPRINT_SCHEMA_VERSION = "eiel_environment_fingerprint_v1"
BOOTSTRAP_PAYLOAD_FIELDS = frozenset({
    "schema_version", "bootstrap_id", "readiness", "generated_at_utc", "task",
    "repository", "environment", "authority", "refresh_inputs", "checks",
    "blocked_check_names", "warning_check_names", "non_goals",
})

def persist_bootstrap_record_fixture(
    record: Mapping[str, Any],
    *,
    state_root: str | Path | None,
    allow_update: bool = False,
) -> Path:
    """Test-only raw fixture seam retained inside the canonical Bootstrap owner."""

    require_test_fixture_context()
    return bootstrap_store(state_root, write_scope=True).put(record, allow_update=allow_update)



def _retained_preflight_fallback_reason(payload: Mapping[str, Any]) -> str | None:
    refresh_inputs = dict(payload.get("refresh_inputs") or {})
    explicit = str(refresh_inputs.get("preflight_exemption_reason") or "").strip()
    if explicit:
        return explicit
    for check in payload.get("checks") or []:
        if not isinstance(check, Mapping) or check.get("name") != "librarian_preflight":
            continue
        reason = str(dict(check.get("details") or {}).get("reason") or "").strip()
        if reason:
            return reason
    return None


def bootstrap_refresh_arguments(
    record: Mapping[str, Any],
    *,
    repo_root: str | Path,
    state_root: str | Path | None = None,
    requested_authority_domain: str | None = None,
    acknowledge_dirty: bool = False,
) -> list[str]:
    """Reproduce the prior non-secret Bootstrap classification inputs.

    The command is used by Workspace and Session Manager after HEAD or receipt
    drift.  Older receipts predate ``refresh_inputs``; those are handled by
    conservative inference from the retained environment and check rows.
    """

    payload = dict(record.get("payload") or record)
    task = dict(payload.get("task") or {})
    environment = dict(payload.get("environment") or {})
    authority = dict(payload.get("authority") or {})
    preflight = dict(authority.get("preflight") or {})
    refresh_inputs = dict(payload.get("refresh_inputs") or {})
    mode = str(task.get("mode") or "implementation")
    domain = (
        requested_authority_domain
        or task.get("authority_domain")
        or preflight.get("authority_domain_filter")
    )
    if not domain and preflight.get("authority_resolution_state") == "ambiguous":
        candidate_domains = sorted(
            {str(value) for value in preflight.get("authority_domains") or [] if str(value).strip()}
        )
        if candidate_domains:
            domain = candidate_domains[0]
    arguments: list[object] = [
        "bootstrap",
        "inspect",
        "--repo-root",
        str(Path(repo_root).expanduser().resolve()),
        "--task-mode",
        mode,
        "--operator-instruction-ref",
        str(task.get("operator_instruction_ref") or "<operator instruction>"),
    ]
    if state_root is not None:
        arguments.extend(["--state-root", str(actor_state_root(state_root))])
    if task.get("topic"):
        arguments.extend(["--topic", str(task["topic"])])
    if task.get("edit_authorized"):
        arguments.append("--edit-authorized")
    if task.get("campaign_authorized"):
        arguments.append("--campaign-authorized")
    if acknowledge_dirty or bool(refresh_inputs.get("acknowledge_dirty")):
        arguments.append("--acknowledge-dirty")
    if domain:
        arguments.extend(["--authority-domain", str(domain)])

    gpu_mode = str(refresh_inputs.get("gpu_required_mode") or "").strip().lower()
    if gpu_mode not in {"yes", "no", "auto"}:
        gpu_mode = "yes" if bool(task.get("gpu_required")) else "no"
    arguments.extend(["--gpu-required", gpu_mode])

    explicit_python = str(refresh_inputs.get("python") or "").strip()
    if not explicit_python and environment.get("source") == "explicit":
        explicit_python = str(environment.get("python") or "").strip()
    if explicit_python:
        arguments.extend(["--python", explicit_python])
    if bool(refresh_inputs.get("allow_current_python")) or environment.get("source") == "explicit_current_fallback":
        arguments.append("--allow-current-python")

    for row in task.get("write_scope") or []:
        value = row.get("path") if isinstance(row, Mapping) else row
        if value:
            arguments.extend(["--write-scope", str(value)])
    for row in task.get("external_evidence_roots") or []:
        value = row.get("path") if isinstance(row, Mapping) else row
        if value:
            arguments.extend(["--external-evidence-root", str(value)])
    for value in task.get("expected_validation") or []:
        arguments.extend(["--expected-validation", str(value)])

    retained_receipt = str(authority.get("preflight_receipt_path") or "").strip()
    fallback_reason = _retained_preflight_fallback_reason(payload)
    # A refresh is an authority refresh, not a replay of the old catalog
    # snapshot.  Reusing the retained receipt made the exact generated refresh
    # command fail closed whenever the maintained catalog had changed between
    # sessions; the operator then had to run a second, manually constructed
    # preflight before retrying the printed command.  Keep the old receipt as
    # provenance in the prior Bootstrap record, but let substantive refreshes
    # run the normal live preflight from the retained topic/domain.  Mechanical
    # refreshes have no live authority requirement and may keep their explicit
    # receipt binding.
    if retained_receipt and mode not in SUBSTANTIVE_TASK_MODES:
        arguments.extend(["--preflight-receipt", retained_receipt])
        if fallback_reason:
            arguments.extend(["--preflight-exemption-reason", fallback_reason])
    elif fallback_reason:
        arguments.extend(["--preflight-exemption-reason", fallback_reason])

    collection = str(refresh_inputs.get("collection") or preflight.get("collection") or "all")
    if collection in {"evidence", "documents"}:
        arguments.extend(["--collection", collection])
    limit = refresh_inputs.get("limit")
    if isinstance(limit, int) and limit != 20:
        arguments.extend(["--limit", limit])
    return [str(value) for value in arguments]


def bootstrap_refresh_command(
    record: Mapping[str, Any],
    *,
    repo_root: str | Path,
    state_root: str | Path | None = None,
    requested_authority_domain: str | None = None,
    acknowledge_dirty: bool = False,
) -> str:
    """Render the exact command for a safe Bootstrap refresh."""

    return render_eiel_command(
        *bootstrap_refresh_arguments(
            record,
            repo_root=repo_root,
            state_root=state_root,
            requested_authority_domain=requested_authority_domain,
            acknowledge_dirty=acknowledge_dirty,
        )
    )


def _run(command: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(env) if env is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _check(name: str, status: str, message: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "message": message, "details": details}


def _parse_porcelain_z(raw: str) -> list[dict[str, Any]]:
    parts = raw.split("\0")
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(parts):
        item = parts[index]
        index += 1
        if not item:
            continue
        if len(item) < 3:
            raise ActorError(
                "Git status emitted malformed porcelain output",
                code="git_status_invalid",
            )
        status = item[:2]
        path = item[3:] if item[2:3] == " " else item[2:]
        original_path = None
        if (status[0] in {"R", "C"} or status[1] in {"R", "C"}) and index < len(parts):
            original_path = parts[index] or None
            index += 1
        rows.append({
            "status": status,
            "path": path,
            "original_path": original_path,
        })
    return rows


def _git_index_path_identity(root: Path, relative_path: str) -> dict[str, Any]:
    result = _run([
        "git", "-C", str(root), "ls-files", "--stage", "-z", "--", relative_path
    ])
    if result.returncode != 0:
        raise ActorError(
            "Git index identity is unavailable",
            code="git_status_unavailable",
            problems=[result.stderr.strip() or relative_path],
        )
    raw = result.stdout.encode("utf-8", errors="surrogateescape")
    return {
        "present": bool(raw),
        "sha256": hashlib.sha256(raw).hexdigest() if raw else None,
    }


def _worktree_path_identity(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return {"kind": "missing", "mode": None, "size": None, "sha256": None}
    except OSError as exc:
        raise ActorError(
            "Git working-tree path identity is unavailable",
            code="git_status_unavailable",
            problems=[f"{relative_path}: {exc}"],
        ) from exc

    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(path)
        raw = target.encode("utf-8", errors="surrogateescape")
        return {
            "kind": "symlink",
            "mode": mode,
            "size": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if stat.S_ISREG(metadata.st_mode):
        return {
            "kind": "file",
            "mode": mode,
            "size": int(metadata.st_size),
            "sha256": file_sha256(path),
        }
    if stat.S_ISDIR(metadata.st_mode):
        return {"kind": "directory", "mode": mode, "size": None, "sha256": None}
    return {"kind": "other", "mode": mode, "size": int(metadata.st_size), "sha256": None}


def _working_state_material(root: Path, entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identities: list[dict[str, Any]] = []
    ordered = sorted(
        (dict(row) for row in entries),
        key=lambda row: (
            str(row.get("path") or ""),
            str(row.get("original_path") or ""),
            str(row.get("status") or ""),
        ),
    )
    for row in ordered:
        path = str(row.get("path") or "")
        original = str(row.get("original_path") or "") or None
        identity = {
            "status": str(row.get("status") or ""),
            "path": path,
            "original_path": original,
            "path_index": _git_index_path_identity(root, path),
            "path_worktree": _worktree_path_identity(root, path),
        }
        if original:
            identity["original_index"] = _git_index_path_identity(root, original)
            identity["original_worktree"] = _worktree_path_identity(root, original)
        identities.append(identity)
    return {
        "schema_version": WORKING_STATE_SCHEMA_VERSION,
        "entries": identities,
    }


def _repository_state(root: Path, *, head: str, branch: str | None) -> tuple[dict[str, Any], str]:
    porcelain = _run([
        "git", "-C", str(root), "status", "--porcelain=v1", "-z", "--untracked-files=all"
    ])
    if porcelain.returncode != 0:
        raise ActorError(
            "Git working-tree status is unavailable",
            code="git_status_unavailable",
            problems=[porcelain.stderr.strip() or str(root)],
        )
    human = _run(["git", "-C", str(root), "status", "--short", "--branch", "--untracked-files=all"])
    if human.returncode != 0:
        raise ActorError(
            "Git human-readable status is unavailable",
            code="git_status_unavailable",
            problems=[human.stderr.strip() or str(root)],
        )
    entries = _parse_porcelain_z(porcelain.stdout)
    changed_paths = sorted({
        str(value)
        for row in entries
        for value in (row.get("path"), row.get("original_path"))
        if value
    })
    working_state = _working_state_material(root, entries)
    material = {
        "schema_version": REPOSITORY_STATE_SCHEMA_VERSION,
        "root": str(root),
        "head": head,
        "branch": branch,
        "working_state_sha256": canonical_sha256(working_state),
        "changed_paths": changed_paths,
    }
    material["repository_state_sha256"] = canonical_sha256(material)
    return {
        **material,
        "dirty": bool(entries),
        "dirty_entries": entries,
    }, human.stdout.strip()


def _git_repository(candidate: Path) -> tuple[Path | None, list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    result = _run(["git", "-C", str(candidate), "rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        checks.append(_check("git_worktree", "BLOCKED", "Directory is not a Git worktree.", stderr=result.stderr.strip()))
        return None, checks, {}
    root = Path(result.stdout.strip()).resolve()
    inside = _run(["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        checks.append(_check("git_worktree", "BLOCKED", "Git did not confirm an active worktree."))
        return None, checks, {}
    head_result = _run(["git", "-C", str(root), "rev-parse", "HEAD"])
    if head_result.returncode != 0 or not head_result.stdout.strip():
        checks.append(_check("git_head", "BLOCKED", "Git worktree has no readable HEAD.", stderr=head_result.stderr.strip()))
        return None, checks, {}
    branch_result = _run(["git", "-C", str(root), "branch", "--show-current"])
    if branch_result.returncode != 0:
        checks.append(_check("git_branch", "BLOCKED", "Git branch state is unavailable.", stderr=branch_result.stderr.strip()))
        return None, checks, {}
    head = head_result.stdout.strip()
    branch = branch_result.stdout.strip() or None
    try:
        state, status_short_branch = _repository_state(root, head=head, branch=branch)
    except ActorError as exc:
        checks.append(_check("git_status", "BLOCKED", str(exc), problems=exc.problems))
        return None, checks, {}
    checks.append(_check("git_worktree", "PASS", "Git worktree and live working state confirmed.", root=str(root)))
    return root, checks, {
        **state,
        "status_short_branch": status_short_branch,
    }


def git_repository_identity(candidate: str | Path | None = None) -> dict[str, Any]:
    """Return the live read-only Git identity for an EIEL worktree.

    This public wrapper keeps all actor subprocess use inside Bootstrap's
    audited, allow-listed discovery surface.  Workspace and Session Manager
    reuse the result instead of growing their own process launchers.
    """

    start = Path(candidate or Path.cwd()).expanduser().resolve()
    root, checks, info = _git_repository(start)
    if root is None:
        blocked = [row for row in checks if row.get("status") == "BLOCKED"]
        problems = [str(row.get("message") or "Git worktree unavailable") for row in blocked]
        for row in blocked:
            details = dict(row.get("details") or {})
            if details.get("stderr"):
                problems.append(str(details["stderr"]))
        raise ActorError(
            "directory is not an active Git worktree",
            code="git_worktree_missing",
            problems=problems or [f"Git could not resolve {start}."],
            next_actions=[
                {
                    "action": "select_real_repository",
                    "description": "Run EIEL against the real Git worktree, not an extracted transfer ZIP.",
                }
            ],
        )
    if not info.get("head"):
        raise ActorError(
            "Git worktree has no readable HEAD",
            code="git_head_unavailable",
            problems=[str(root)],
        )
    return info


def _repository_markers(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    required_files = ["AGENTS.md", "pyproject.toml", "emergence-dev.toml"]
    required_dirs = ["engine", "tools", "tests", "docs", "evidences"]
    missing = [name for name in required_files if not (root / name).is_file()]
    missing.extend(name for name in required_dirs if not (root / name).is_dir())
    project_name = None
    project_version = None
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        project_name = project.get("project", {}).get("name")
        project_version = project.get("project", {}).get("version")
    except Exception as exc:
        missing.append(f"readable pyproject.toml ({exc})")
    if missing:
        checks.append(_check("repository_markers", "BLOCKED", "Required EIEL repository markers are missing.", missing=missing))
    elif project_name != "emergence":
        checks.append(_check("repository_markers", "BLOCKED", "pyproject.toml does not identify the emergence project.", project_name=project_name))
    else:
        checks.append(_check("repository_markers", "PASS", "EIEL repository markers are present.", project_name=project_name))
    active_goal = root / "ACTIVE_GOAL.md"
    return checks, {
        "project_name": project_name,
        "project_version": project_version,
        "agents_path": str(root / "AGENTS.md"),
        "active_goal_path": str(active_goal) if active_goal.is_file() else None,
        "bootstrap_source_path": str(root / BOOTSTRAP_SOURCE_RELATIVE) if (root / BOOTSTRAP_SOURCE_RELATIVE).is_file() else None,
    }


def _select_python(root: Path, explicit: str | None, allow_current: bool) -> tuple[Path | None, list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    candidates: list[tuple[str, Path]] = []
    if explicit:
        candidates.append(("explicit", Path(explicit).expanduser()))
    candidates.append(("repository_venv", root / ".venv/bin/python"))
    candidates.append(("shared_cuda_venv", Path.home() / ".venvs/emergence-py314-cuda/bin/python"))
    if allow_current:
        candidates.append(("explicit_current_fallback", Path(sys.executable)))
    chosen: tuple[str, Path, Path] | None = None
    seen: set[Path] = set()
    for source, candidate in candidates:
        selected = candidate.expanduser().absolute()
        resolved = selected.resolve() if selected.exists() else selected
        if selected in seen:
            continue
        seen.add(selected)
        if selected.is_file() and os.access(selected, os.X_OK):
            chosen = (source, selected, resolved)
            break
    if chosen is None:
        checks.append(_check(
            "python_environment",
            "BLOCKED",
            "No approved EIEL Python interpreter was found.",
            checked=[str(path) for _, path in candidates],
        ))
        return None, checks, {}
    source, python, resolved_python = chosen
    forbidden = str(Path.home() / "causyl" / ".venv") + os.sep
    if str(python).startswith(forbidden) or str(resolved_python).startswith(forbidden):
        checks.append(_check("python_environment", "BLOCKED", "Historical Causyl Python is forbidden for active EIEL work.", python=str(python)))
        return None, checks, {}
    status = "PASS" if source in {"explicit", "repository_venv", "shared_cuda_venv"} else "WARN"
    checks.append(_check(
        "python_environment",
        status,
        "Approved Python interpreter selected." if status == "PASS" else "Current Python was accepted only through explicit fallback.",
        python=str(python),
        resolved_python=str(resolved_python),
        source=source,
    ))
    return python, checks, {"python": str(python), "resolved_python": str(resolved_python), "source": source}


def _probe_python(root: Path, python: Path, *, gpu_required: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    script = r'''
import importlib, importlib.metadata, json, os, platform, sys
payload = {
  "python": sys.executable,
  "python_version": platform.python_version(),
  "platform": platform.platform(),
  "modules": {},
  "torch": {"available": False, "version": None, "cuda_available": False, "cuda_device_count": 0, "device_names": []},
}
for name in ["pytest", "jsonschema", "yaml"]:
  try:
    module = importlib.import_module(name)
    payload["modules"][name] = {"available": True, "version": getattr(module, "__version__", None)}
  except Exception as exc:
    payload["modules"][name] = {"available": False, "error": repr(exc)}
try:
  if os.environ.get("EIEL_BOOTSTRAP_GPU_PROBE") == "1":
    import torch
    payload["torch"] = {
      "available": True,
      "version": str(torch.__version__),
      "cuda_available": bool(torch.cuda.is_available()),
      "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
      "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
    }
  else:
    payload["torch"] = {
      "available": True,
      "version": importlib.metadata.version("torch"),
      "cuda_available": False,
      "cuda_device_count": 0,
      "device_names": [],
      "probe_mode": "metadata_only_non_gpu_task",
    }
except Exception as exc:
  payload["torch"]["error"] = repr(exc)
try:
  import tools, engine
  payload["repository_imports"] = {"available": True}
except Exception as exc:
  payload["repository_imports"] = {"available": False, "error": repr(exc)}
print(json.dumps(payload, sort_keys=True))
'''
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["EIEL_BOOTSTRAP_GPU_PROBE"] = "1" if gpu_required else "0"
    env["PYTHONPATH"] = f"{root}:{root / 'engine'}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    result = _run([str(python), "-B", "-c", script], cwd=root, env=env)
    checks: list[dict[str, Any]] = []
    if result.returncode != 0:
        checks.append(_check("runtime_probe", "BLOCKED", "Selected Python could not complete the read-only runtime probe.", stderr=result.stderr.strip()))
        return checks, {"returncode": result.returncode, "stderr": result.stderr.strip()}
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        checks.append(_check("runtime_probe", "BLOCKED", "Runtime probe did not emit valid JSON.", error=str(exc), stdout=result.stdout[-1000:]))
        return checks, {}
    missing_modules = sorted(name for name, row in payload.get("modules", {}).items() if not row.get("available"))
    if missing_modules:
        checks.append(_check("required_packages", "BLOCKED", "Required tooling packages are missing.", missing=missing_modules))
    else:
        checks.append(_check("required_packages", "PASS", "Required tooling packages are importable."))
    if not payload.get("repository_imports", {}).get("available"):
        checks.append(_check("repository_imports", "BLOCKED", "Repository packages are not importable through the selected Python.", error=payload.get("repository_imports", {}).get("error")))
    else:
        checks.append(_check("repository_imports", "PASS", "Repository packages import successfully."))
    torch_row = dict(payload.get("torch") or {})
    if gpu_required:
        if not torch_row.get("available"):
            checks.append(_check("cuda", "BLOCKED", "GPU work requires Torch, but Torch is unavailable.", error=torch_row.get("error")))
        elif not torch_row.get("cuda_available"):
            checks.append(_check("cuda", "BLOCKED", "GPU work requires CUDA, but the selected Torch environment reports CUDA unavailable.", torch_version=torch_row.get("version")))
        else:
            checks.append(_check("cuda", "PASS", "CUDA is available through the selected Torch environment.", devices=torch_row.get("device_names")))
    else:
        status = "PASS" if torch_row.get("available") else "WARN"
        checks.append(_check("torch", status, "Torch environment inspected." if status == "PASS" else "Torch is unavailable, but this task does not require GPU execution.", torch=torch_row))
    return checks, payload


def _python_file_identity(python: Path) -> dict[str, Any]:
    try:
        resolved = python.resolve(strict=True)
        stat = resolved.stat()
    except OSError as exc:
        raise ActorError(
            f"selected Python interpreter is unavailable: {python}: {exc}",
            code="bootstrap_environment_unavailable",
        ) from exc
    return {
        "path": str(python),
        "resolved_path": str(resolved),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _environment_payload(
    root: Path,
    python_info: Mapping[str, Any],
    runtime_info: Mapping[str, Any],
) -> dict[str, Any]:
    python_value = str(python_info.get("python") or "")
    file_identity = _python_file_identity(Path(python_value)) if python_value else None
    material = {
        "schema_version": ENVIRONMENT_FINGERPRINT_SCHEMA_VERSION,
        "repository_root": str(root),
        "source": python_info.get("source"),
        "python_file": file_identity,
        "probe": dict(runtime_info),
    }
    return {
        **dict(python_info),
        "probe": dict(runtime_info),
        "python_file": file_identity,
        "fingerprint_schema_version": ENVIRONMENT_FINGERPRINT_SCHEMA_VERSION,
        "fingerprint_sha256": canonical_sha256(material),
    }


def _resolve_scope(root: Path, values: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for raw in values:
        candidate = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            checks.append(_check("write_scope", "BLOCKED", "Declared write scope escapes the repository.", path=str(candidate)))
            continue
        rows.append({"path": relative.as_posix(), "exists": candidate.exists(), "kind": "directory" if candidate.is_dir() else "file" if candidate.is_file() else "future_path"})
    if values and not any(row["status"] == "BLOCKED" for row in checks):
        checks.append(_check("write_scope", "PASS", "Bounded write scope recorded.", paths=[row["path"] for row in rows]))
    return checks, rows


def _resolve_external_evidence_roots(root: Path, values: Sequence[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate read-only evidence roots without granting source-write scope."""

    checks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for raw in values:
        declared = Path(raw).expanduser()
        if not declared.is_absolute():
            checks.append(_check(
                "external_evidence_root",
                "BLOCKED",
                "External evidence root must be an absolute path.",
                path=str(declared),
            ))
            continue
        candidate = declared.resolve()
        if not candidate.is_dir():
            checks.append(_check(
                "external_evidence_root",
                "BLOCKED",
                "External evidence root must be an existing directory.",
                path=str(candidate),
            ))
            continue
        try:
            candidate.relative_to(root)
        except ValueError:
            pass
        else:
            checks.append(_check(
                "external_evidence_root",
                "BLOCKED",
                "External evidence root must be outside the repository; use repository paths only in --write-scope.",
                path=str(candidate),
            ))
            continue
        rows.append({"path": str(candidate), "exists": True, "kind": "directory"})
    if values and not any(row["status"] == "BLOCKED" for row in checks):
        checks.append(_check(
            "external_evidence_root",
            "PASS",
            "Read-only external evidence roots recorded without expanding source write scope.",
            paths=[row["path"] for row in rows],
        ))
    return checks, rows


def _bootstrap_repair_command(payload: Mapping[str, Any], *, state_root: str | Path | None) -> str:
    repository = dict(payload.get("repository") or {})
    return bootstrap_refresh_command(
        payload,
        repo_root=str(repository.get("root") or Path.cwd()),
        state_root=state_root,
        acknowledge_dirty=bool(repository.get("dirty")),
    )


def inspect_bootstrap(
    *,
    repo_root_value: str | Path | None,
    state_root_value: str | Path | None,
    generated_at: str | None,
    task_mode: str,
    topic: str | None,
    operator_instruction_ref: str | None,
    edit_authorized: bool,
    campaign_authorized: bool,
    gpu_required_mode: str,
    python_value: str | None,
    allow_current_python: bool,
    acknowledge_dirty: bool,
    write_scope: Sequence[str],
    external_evidence_roots: Sequence[str],
    expected_validation: Sequence[str],
    preflight_receipt_path: str | None,
    preflight_exemption_reason: str | None,
    authority_domain: str | None,
    collection: str,
    limit: int,
    store_receipt: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    mode = str(task_mode).lower()
    if mode not in TASK_MODES:
        raise ActorError(f"unknown task mode: {task_mode}", code="bootstrap_task_mode_unknown")
    checks: list[dict[str, Any]] = []
    candidate = Path(repo_root_value or Path.cwd()).expanduser().resolve()
    root, git_checks, git_info = _git_repository(candidate)
    checks.extend(git_checks)
    if root is None:
        payload = {
            "schema_version": BOOTSTRAP_SCHEMA_VERSION,
            "bootstrap_id": stable_id("BST", {"candidate": str(candidate), "task_mode": mode, "generated_at": utc_now(generated_at)}),
            "readiness": "BLOCKED",
            "generated_at_utc": utc_now(generated_at),
            "task": {"mode": mode, "topic": topic, "external_evidence_roots": []},
            "repository": {"candidate": str(candidate)},
            "environment": {},
            "authority": {"preflight": None, "preflight_receipt_path": None},
            "refresh_inputs": {
                "gpu_required_mode": gpu_required_mode,
                "python": str(python_value or "") or None,
                "allow_current_python": bool(allow_current_python),
                "acknowledge_dirty": bool(acknowledge_dirty),
                "preflight_exemption_reason": str(preflight_exemption_reason or "") or None,
                "collection": collection,
                "limit": int(limit),
            },
            "checks": checks,
            "blocked_check_names": [row["name"] for row in checks if row["status"] == "BLOCKED"],
            "warning_check_names": [row["name"] for row in checks if row["status"] == "WARN"],
            "non_goals": ["no_environment_creation", "no_package_installation", "no_source_edit", "no_campaign_launch", "no_process_control", "no_guessing"],
        }
        return payload, None
    marker_checks, repo_info = _repository_markers(root)
    checks.extend(marker_checks)

    dirty = bool(git_info.get("dirty"))
    if dirty and mode in EDITING_TASK_MODES and not acknowledge_dirty:
        checks.append(_check("dirty_state", "BLOCKED", "Repository has pre-existing changes that were not acknowledged.", dirty_entries=git_info.get("dirty_entries")))
    elif dirty:
        checks.append(_check("dirty_state", "WARN", "Repository contains pre-existing changes; they are recorded for scope protection.", dirty_entries=git_info.get("dirty_entries")))
    else:
        checks.append(_check("dirty_state", "PASS", "Repository working tree is clean."))

    if mode in EDITING_TASK_MODES and not edit_authorized:
        checks.append(_check("edit_authority", "BLOCKED", "Task mode implies edits but edit authorization was not declared."))
    else:
        checks.append(_check("edit_authority", "PASS", "Edit authorization classification recorded.", edit_authorized=edit_authorized))
    if mode == "campaign" and not campaign_authorized:
        checks.append(_check("campaign_authority", "BLOCKED", "Campaign task was requested without explicit campaign authorization."))
    elif mode == "campaign":
        checks.append(_check("campaign_authority", "PASS", "Campaign authorization was explicitly declared."))

    scope_checks, scope_rows = _resolve_scope(root, write_scope)
    checks.extend(scope_checks)
    if mode in EDITING_TASK_MODES and not write_scope:
        checks.append(_check("write_scope", "BLOCKED", "Editing task requires a declared bounded write scope."))
    evidence_root_checks, evidence_root_rows = _resolve_external_evidence_roots(root, external_evidence_roots)
    checks.extend(evidence_root_checks)

    gpu_required = gpu_required_mode == "yes" or (gpu_required_mode == "auto" and mode in GPU_REQUIRING_MODES)
    python, python_checks, python_info = _select_python(root, python_value, allow_current_python)
    checks.extend(python_checks)
    runtime_info: dict[str, Any] = {}
    if python is not None:
        runtime_checks, runtime_info = _probe_python(root, python, gpu_required=gpu_required)
        checks.extend(runtime_checks)

    preflight_summary: dict[str, Any] | None = None
    retained_path: Path | None = None
    if mode in SUBSTANTIVE_TASK_MODES:
        if preflight_receipt_path:
            receipt = read_json(preflight_receipt_path, name="Knowledge Librarian preflight receipt")
        else:
            if not topic:
                checks.append(_check("librarian_preflight", "BLOCKED", "Substantive task requires a topic for Knowledge Librarian preflight."))
                receipt = None
            else:
                try:
                    receipt = run_librarian_preflight(
                        repo_root=root,
                        topic=topic,
                        collection=collection,
                        limit=limit,
                        authority_domain=authority_domain,
                    )
                except ActorError as exc:
                    checks.append(_check("librarian_preflight", "BLOCKED", str(exc), problems=exc.problems))
                    receipt = None
        if receipt is not None:
            errors = validate_preflight_receipt(receipt)
            if errors:
                checks.append(_check("librarian_preflight", "BLOCKED", "Knowledge Librarian preflight receipt is invalid.", errors=errors))
            else:
                source_root = root / "tools/evidence/librarian/catalog_source/00-catalog"
                try:
                    current_digest = wp4.maintained_source_digest(source_root)
                    current_build = "eiel-build-wp4-" + hashlib.sha256(current_digest.encode()).hexdigest()[:20]
                except Exception as exc:
                    checks.append(_check("librarian_preflight", "BLOCKED", "Current Knowledge Librarian catalog identity cannot be verified.", error=str(exc)))
                else:
                    receipt_digest = str(receipt.get("source_digest") or receipt.get("maintained_source_digest") or "")
                    receipt_build = str(receipt.get("catalog_build_id") or "")
                    if receipt_digest != current_digest or receipt_build != current_build:
                        checks.append(_check(
                            "librarian_preflight",
                            "BLOCKED",
                            "Knowledge Librarian preflight receipt does not match the current maintained catalog.",
                            receipt_digest=receipt_digest,
                            current_digest=current_digest,
                            receipt_build=receipt_build,
                            current_build=current_build,
                        ))
                    retained_path = retain_preflight_receipt(receipt, state_root=actor_state_root(state_root_value))
                    preflight_summary = compact_preflight_receipt(receipt, receipt_path=retained_path)
                    if bool(receipt.get("fail_closed")):
                        fallback_reason = str(preflight_exemption_reason or "").strip()
                        fallback_allowed = bool(receipt.get("fallback_repository_search_allowed"))
                        status = "WARN" if fallback_allowed and fallback_reason else "BLOCKED"
                        message = (
                            "Knowledge Librarian preflight is unresolved; an explicit bounded repository fallback is retained."
                            if status == "WARN"
                            else "Knowledge Librarian preflight remains fail-closed."
                        )
                        checks.append(
                            _check(
                                "librarian_preflight",
                                status,
                                message,
                                authority_resolution_state=receipt.get("authority_resolution_state"),
                                authority_domains=preflight_summary.get("authority_domains"),
                                authority_assertion_ids=preflight_summary.get("authority_assertion_ids"),
                                requested_authority_domain=authority_domain,
                                fallback_repository_search_allowed=fallback_allowed,
                                fallback_reason=fallback_reason or None,
                                unresolved=receipt.get("unresolved_authority_gaps"),
                            )
                        )
                    else:
                        checks.append(_check("librarian_preflight", "PASS", "Knowledge Librarian preflight completed and was retained.", receipt_sha256=preflight_summary.get("receipt_sha256") if preflight_summary else None))
    else:
        reason = str(preflight_exemption_reason or "").strip()
        if not reason:
            checks.append(_check("librarian_preflight", "BLOCKED", "Mechanical task requires an explicit preflight-exemption reason."))
        else:
            checks.append(_check("librarian_preflight", "WARN", "Mechanical task uses an explicit bounded preflight exemption.", reason=reason))

    operator_instruction_binding = bind_operator_instruction(
        operator_instruction_ref, state_root=actor_state_root(state_root_value), base_dir=root
    )
    if not operator_instruction_binding:
        checks.append(_check("operator_instruction", "BLOCKED", "Exact operator instruction reference was not recorded."))
    elif instruction_is_authoritative(operator_instruction_binding):
        checks.append(_check(
            "operator_instruction", "PASS",
            "Operator instruction content was retained and digest-bound.",
            binding=operator_instruction_binding,
        ))
    else:
        checks.append(_check(
            "operator_instruction", "WARN",
            "The retained label records provenance only; the live explicit operator instruction remains operational authority.",
            binding=operator_instruction_binding,
        ))
    if not expected_validation and mode in EDITING_TASK_MODES:
        checks.append(_check("validation_plan", "BLOCKED", "Editing task requires at least one declared validation command or gate."))
    elif expected_validation:
        checks.append(_check("validation_plan", "PASS", "Validation expectations recorded.", validations=list(expected_validation)))

    blocked = [row for row in checks if row["status"] == "BLOCKED"]
    warnings = [row for row in checks if row["status"] == "WARN"]
    readiness = "BLOCKED" if blocked else "READY_WITH_WARNINGS" if warnings else "READY"
    timestamp = utc_now(generated_at)
    identity = {
        "repo_head": git_info.get("head"),
        "repo_root": str(root),
        "repository_state_sha256": git_info.get("repository_state_sha256"),
        "task_mode": mode,
        "topic": topic,
        "authority_domain": authority_domain,
        "generated_at": timestamp,
        "operator_instruction_ref": operator_instruction_ref,
        "operator_instruction_binding": operator_instruction_binding,
    }
    bootstrap_id = stable_id("BST", identity)
    payload = {
        "schema_version": BOOTSTRAP_SCHEMA_VERSION,
        "bootstrap_id": bootstrap_id,
        "readiness": readiness,
        "generated_at_utc": timestamp,
        "task": {
            "mode": mode,
            "topic": topic,
            "operator_instruction_ref": operator_instruction_ref,
            "operator_instruction_binding": operator_instruction_binding,
            "authority_domain": authority_domain,
            "edit_authorized": bool(edit_authorized),
            "campaign_authorized": bool(campaign_authorized),
            "gpu_required": bool(gpu_required),
            "write_scope": scope_rows,
            "external_evidence_roots": evidence_root_rows,
            "expected_validation": list(expected_validation),
        },
        "repository": {**git_info, **repo_info},
        "environment": _environment_payload(root, python_info, runtime_info) if python_info else {},
        "authority": {"preflight": preflight_summary, "preflight_receipt_path": str(retained_path) if retained_path else None},
        "refresh_inputs": {
            "gpu_required_mode": gpu_required_mode,
            "python": str(python_value or "") or None,
            "allow_current_python": bool(allow_current_python),
            "acknowledge_dirty": bool(acknowledge_dirty),
            "preflight_exemption_reason": str(preflight_exemption_reason or "") or None,
            "collection": collection,
            "limit": int(limit),
        },
        "checks": checks,
        "blocked_check_names": [row["name"] for row in blocked],
        "warning_check_names": [row["name"] for row in warnings],
        "non_goals": [
            "no_environment_creation",
            "no_package_installation",
            "no_source_edit",
            "no_campaign_launch",
            "no_process_control",
            "no_guessing",
        ],
    }
    record: dict[str, Any] | None = None
    if store_receipt:
        record = new_record(
            actor=ACTOR_NAME,
            record_type=BOOTSTRAP_RECORD_TYPE,
            record_id=bootstrap_id,
            status=readiness,
            payload=payload,
            generated_at=timestamp,
            source_refs=[
                {"kind": "repository", "uri": str(root), "commit": git_info.get("head")},
                *([{"kind": "librarian_preflight", "uri": str(retained_path)}] if retained_path else []),
            ],
            note="read-only repository bootstrap",
        )
        bootstrap_store(state_root_value, write_scope=True).put(record)
    return payload, record


def load_bootstrap_payload(raw: Any) -> dict[str, Any]:
    """Strictly load a persisted Bootstrap payload and its nested versions."""

    if not isinstance(raw, Mapping):
        raise ActorError("Bootstrap payload must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        raw, expected=BOOTSTRAP_SCHEMA_VERSION, label="Bootstrap receipt payload"
    )
    require_known_fields(raw, allowed=BOOTSTRAP_PAYLOAD_FIELDS, label="Bootstrap payload")
    payload = dict(raw)
    repository = payload.get("repository")
    if not isinstance(repository, Mapping):
        raise ActorError("Bootstrap repository state must be an object", code="schema_record_malformed")
    require_explicit_schema_version(
        repository,
        expected=REPOSITORY_STATE_SCHEMA_VERSION,
        label="Bootstrap repository state",
    )
    repository_material = {
        "schema_version": repository.get("schema_version"),
        "root": repository.get("root"),
        "head": repository.get("head"),
        "branch": repository.get("branch"),
        "working_state_sha256": repository.get("working_state_sha256"),
        "changed_paths": list(repository.get("changed_paths") or []),
    }
    observed_repository_digest = repository.get("repository_state_sha256")
    if not isinstance(observed_repository_digest, str) or observed_repository_digest != canonical_sha256(repository_material):
        raise ActorError(
            "Bootstrap repository state digest mismatch",
            code="bootstrap_repository_digest_mismatch",
        )
    environment = payload.get("environment")
    if not isinstance(environment, Mapping):
        raise ActorError("Bootstrap environment fingerprint must be an object", code="schema_record_malformed")
    if "fingerprint_schema_version" not in environment:
        raise ActorError(
            "Bootstrap environment fingerprint schema_version is missing",
            code="schema_version_missing",
        )
    observed = environment.get("fingerprint_schema_version")
    if not isinstance(observed, str) or not observed.strip():
        raise ActorError(
            "Bootstrap environment fingerprint schema_version is malformed",
            code="schema_version_malformed",
        )
    if observed != ENVIRONMENT_FINGERPRINT_SCHEMA_VERSION:
        raise ActorError(
            f"unsupported Bootstrap environment fingerprint schema_version: {observed}",
            code="schema_version_unsupported",
            problems=[f"Supported schema versions: {ENVIRONMENT_FINGERPRINT_SCHEMA_VERSION}"],
        )
    environment_material = {
        "schema_version": ENVIRONMENT_FINGERPRINT_SCHEMA_VERSION,
        "repository_root": environment.get("repository_root") or repository.get("root"),
        "source": environment.get("source"),
        "python_file": environment.get("python_file"),
        "probe": dict(environment.get("probe") or {}),
    }
    observed_environment_digest = environment.get("fingerprint_sha256")
    if not isinstance(observed_environment_digest, str) or observed_environment_digest != canonical_sha256(environment_material):
        raise ActorError(
            "Bootstrap environment fingerprint digest mismatch",
            code="bootstrap_environment_digest_mismatch",
        )
    return payload


def load_bootstrap_record(raw: Any) -> dict[str, Any]:
    record = load_actor_record(raw)
    if record.get("record_type") != BOOTSTRAP_RECORD_TYPE:
        raise ActorError(
            "record is not a repository bootstrap receipt",
            code="schema_record_malformed",
        )
    load_bootstrap_payload(record.get("payload"))
    return record



def validate_bootstrap_record(
    record: Mapping[str, Any],
    *,
    expected_repo_root: str | Path | None = None,
    expected_head: str | None = None,
    expected_repository: Mapping[str, Any] | None = None,
    revalidate_environment: bool = False,
    expected_task_mode: str | None = None,
    expected_topic: str | None = None,
    expected_operator_instruction_ref: str | None = None,
    expected_authority_domain: str | None = None,
    max_age_hours: float | None = None,
    now: str | None = None,
) -> list[str]:
    errors: list[str] = []
    if record.get("record_type") != BOOTSTRAP_RECORD_TYPE:
        errors.append("record is not a repository bootstrap receipt")
    if str(record.get("status")) not in READY_STATUSES:
        errors.append(f"bootstrap receipt is not ready: {record.get('status')}")
    payload = dict(record.get("payload") or {})
    try:
        load_bootstrap_payload(payload)
    except ActorError as exc:
        errors.extend(exc.problems)
    if payload.get("readiness") != record.get("status"):
        errors.append("bootstrap payload readiness does not match record status")
    repository = dict(payload.get("repository") or {})
    live_repository = dict(expected_repository or {})
    live_root = live_repository.get("root") or expected_repo_root
    live_head = live_repository.get("head") or expected_head
    if live_root and Path(str(repository.get("root") or "")).resolve() != Path(live_root).resolve():
        errors.append("bootstrap repository root does not match current repository")
    if live_head and str(repository.get("head") or "") != str(live_head):
        errors.append("bootstrap repository HEAD does not match current HEAD")
    if live_repository:
        for field, label in (
            ("branch", "branch"),
            ("working_state_sha256", "working-tree/index state"),
            ("repository_state_sha256", "repository state digest"),
        ):
            if str(repository.get(field) or "") != str(live_repository.get(field) or ""):
                errors.append(f"bootstrap repository {label} does not match current state")
    task = dict(payload.get("task") or {})
    if expected_task_mode and str(task.get("mode") or "") != str(expected_task_mode):
        errors.append("bootstrap task mode does not match the requested task")
    if expected_topic and str(task.get("topic") or "") != str(expected_topic):
        errors.append("bootstrap topic does not match the requested task")
    if expected_operator_instruction_ref and str(task.get("operator_instruction_ref") or "") != str(expected_operator_instruction_ref):
        errors.append("bootstrap operator instruction does not match the requested task")
    binding = task.get("operator_instruction_binding")
    if binding is not None:
        try:
            bind_operator_instruction(binding, state_root=None)
        except ActorError as exc:
            errors.append(str(exc))
    if expected_authority_domain and str(task.get("authority_domain") or "") != str(expected_authority_domain):
        errors.append("bootstrap authority domain does not match the requested authority domain")
    if max_age_hours is not None:
        generated_at = str(payload.get("generated_at_utc") or record.get("updated_at_utc") or "")
        if not generated_at:
            errors.append("bootstrap receipt has no generated timestamp")
        else:
            try:
                age_hours = timestamp_age_hours(generated_at, now=now)
            except ActorError as exc:
                errors.append(str(exc))
            else:
                if age_hours > float(max_age_hours):
                    errors.append(
                        f"bootstrap receipt is stale: {age_hours:.2f} hours old; maximum is {float(max_age_hours):.2f}"
                    )

    if revalidate_environment:
        environment = dict(payload.get("environment") or {})
        python_value = str(environment.get("python") or "")
        root_value = str(live_root or repository.get("root") or "")
        if not python_value or not root_value:
            errors.append("bootstrap environment binding is incomplete")
        else:
            try:
                runtime_checks, runtime_info = _probe_python(
                    Path(root_value),
                    Path(python_value),
                    gpu_required=bool(task.get("gpu_required")),
                )
                blocked_runtime = [row for row in runtime_checks if row.get("status") == "BLOCKED"]
                if blocked_runtime:
                    errors.extend(str(row.get("message") or "runtime probe failed") for row in blocked_runtime)
                live_environment = _environment_payload(
                    Path(root_value),
                    {"python": python_value, "source": environment.get("source")},
                    runtime_info,
                )
            except ActorError as exc:
                errors.extend(exc.problems)
            else:
                if str(environment.get("fingerprint_sha256") or "") != str(live_environment.get("fingerprint_sha256") or ""):
                    errors.append("bootstrap Python/package/Torch/CUDA environment fingerprint changed")

    authority = dict(dict(payload.get("authority") or {}))
    preflight = dict(authority.get("preflight") or {})
    retained_path = preflight.get("receipt_path")
    if preflight:
        if not retained_path:
            errors.append("bootstrap preflight receipt path is missing")
        else:
            try:
                retained = read_json(retained_path, name="retained Knowledge Librarian preflight receipt")
            except Exception as exc:
                errors.append(str(exc))
            else:
                retained_errors = validate_preflight_receipt(retained)
                if retained_errors:
                    errors.extend(retained_errors)
                else:
                    try:
                        recomputed = compact_preflight_receipt(retained, receipt_path=retained_path)
                    except ActorError as exc:
                        errors.append(str(exc))
                    else:
                        if str(preflight.get("receipt_sha256") or "") != str(recomputed.get("receipt_sha256") or ""):
                            errors.append("bootstrap preflight receipt hash changed")
                        for field in (
                            "topic",
                            "collection",
                            "authority_domain_filter",
                            "catalog_build_id",
                            "maintained_source_digest",
                            "authority_resolution_state",
                            "fail_closed",
                            "fallback_repository_search_allowed",
                        ):
                            if str(preflight.get(field) or "") != str(recomputed.get(field) or ""):
                                errors.append(f"bootstrap preflight summary field {field} has drifted")
                        for field in (
                            "authority_domains",
                            "authority_assertion_ids",
                            "evidence_ids",
                            "artifact_ids",
                            "raw_asset_ids",
                            "document_ids",
                            "claim_ids",
                            "limitation_ids",
                            "unknown_authority_gaps",
                            "applicable_dossiers",
                        ):
                            if preflight.get(field) is not None and recomputed.get(field) is not None:
                                if [str(item) for item in preflight.get(field)] != [str(item) for item in recomputed.get(field)]:
                                    errors.append(f"bootstrap preflight summary list {field} has drifted")

    repository_root = repository.get("root") or expected_repo_root
    source_root = Path(str(repository_root)) / "tools/evidence/librarian/catalog_source/00-catalog" if repository_root else None
    if preflight and source_root is not None:
        try:
            current_digest = wp4.maintained_source_digest(source_root)
            current_build = "eiel-build-wp4-" + hashlib.sha256(current_digest.encode()).hexdigest()[:20]
        except Exception as exc:
            errors.append(f"bootstrap current catalog identity cannot be verified: {exc}")
        else:
            if str(preflight.get("catalog_build_id") or "") != str(current_build):
                errors.append("bootstrap preflight catalog build id changed since receipt creation")
            if str(preflight.get("maintained_source_digest") or "") != str(current_digest):
                errors.append("bootstrap preflight catalog source digest changed since receipt creation")

    if any(row.get("status") == "BLOCKED" for row in payload.get("checks") or []):
        errors.append("bootstrap payload contains blocked checks")
    return errors


def assess_bootstrap_record(
    record: Mapping[str, Any],
    *,
    expected_repo_root: str | Path | None = None,
    expected_head: str | None = None,
    expected_repository: Mapping[str, Any] | None = None,
    revalidate_environment: bool = False,
    expected_task_mode: str | None = None,
    expected_topic: str | None = None,
    expected_operator_instruction_ref: str | None = None,
    expected_authority_domain: str | None = None,
    max_age_hours: float | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Classify integrity, compatibility, currentness, and readiness separately."""

    problems = validate_bootstrap_record(
        record,
        expected_repo_root=expected_repo_root,
        expected_head=expected_head,
        expected_repository=expected_repository,
        revalidate_environment=revalidate_environment,
        expected_task_mode=expected_task_mode,
        expected_topic=expected_topic,
        expected_operator_instruction_ref=expected_operator_instruction_ref,
        expected_authority_domain=expected_authority_domain,
        max_age_hours=max_age_hours,
        now=now,
    )
    integrity_errors: list[str] = []
    compatibility_errors: list[str] = []
    staleness_errors: list[str] = []
    readiness_errors: list[str] = []
    for problem in problems:
        text = str(problem)
        lowered = text.lower()
        if "unsupported" in lowered or "schema" in lowered or "malformed" in lowered:
            compatibility_errors.append(text)
        elif any(token in lowered for token in (
            "does not match current", "changed", "stale", "hours old",
            "current head", "current state", "environment fingerprint",
        )):
            staleness_errors.append(text)
        elif "is not ready" in lowered or "blocked" in lowered:
            readiness_errors.append(text)
        else:
            integrity_errors.append(text)
    status = (
        "INVALID" if integrity_errors
        else "INCOMPATIBLE" if compatibility_errors
        else "STALE" if staleness_errors
        else "NOT_READY" if readiness_errors
        else "CURRENT_READY"
    )
    return {
        "schema_version": "eiel_bootstrap_assessment_v1",
        "status": status,
        "integrity_valid": not integrity_errors,
        "compatible": not compatibility_errors,
        "current": not staleness_errors,
        "ready": not readiness_errors and str(record.get("status") or "") in READY_STATUSES,
        "integrity_errors": integrity_errors,
        "compatibility_errors": compatibility_errors,
        "staleness_errors": staleness_errors,
        "readiness_errors": readiness_errors,
        "errors": [
            *integrity_errors,
            *compatibility_errors,
            *staleness_errors,
            *readiness_errors,
        ],
    }


def _human(payload: Mapping[str, Any]) -> str:
    row = dict(payload.get("bootstrap") or payload)
    lines = [
        f"Bootstrap {row.get('bootstrap_id') or 'unrecorded'}",
        f"Readiness: {row.get('readiness')}",
        f"Repository: {dict(row.get('repository') or {}).get('root') or dict(row.get('repository') or {}).get('candidate')}",
        f"Python: {dict(row.get('environment') or {}).get('python') or 'not resolved'}",
    ]
    for check in row.get("checks") or []:
        lines.append(f"{check.get('status')}: {check.get('name')} - {check.get('message')}")
    return "\n".join(lines) + "\n"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state-root")
    parser.add_argument("--output")
    parser.add_argument("--human", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(prog="python -m tools.eiel bootstrap", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True, parser_class=ActorArgumentParser)

    inspect = sub.add_parser("inspect", help="Run the read-only, fail-closed repository bootstrap.")
    _common(inspect)
    inspect.add_argument("--repo-root")
    inspect.add_argument("--generated-at")
    inspect.add_argument("--task-mode", choices=TASK_MODES, required=True)
    inspect.add_argument("--topic")
    inspect.add_argument("--operator-instruction-ref", required=True)
    inspect.add_argument("--edit-authorized", action="store_true")
    inspect.add_argument("--campaign-authorized", action="store_true")
    inspect.add_argument("--gpu-required", choices=("yes", "no", "auto"), default="auto")
    inspect.add_argument("--python")
    inspect.add_argument("--allow-current-python", action="store_true")
    inspect.add_argument("--acknowledge-dirty", action="store_true")
    inspect.add_argument("--write-scope", action="append", default=[])
    inspect.add_argument(
        "--external-evidence-root",
        action="append",
        default=[],
        help="Record an existing read-only evidence directory outside the repository; this never authorizes source writes.",
    )
    inspect.add_argument("--expected-validation", action="append", default=[])
    inspect.add_argument("--preflight-receipt")
    inspect.add_argument("--preflight-exemption-reason")
    inspect.add_argument("--authority-domain")
    inspect.add_argument("--collection", choices=("evidence", "documents", "all"), default="all")
    inspect.add_argument("--limit", type=int, default=20)
    inspect.add_argument("--no-store", action="store_true")

    refresh = sub.add_parser(
        "refresh",
        help="Refresh one retained Bootstrap against its original worktree using retained non-secret inputs.",
    )
    _common(refresh)
    refresh.add_argument("bootstrap_id")
    refresh.add_argument("--repo-root")
    refresh.add_argument("--generated-at")
    refresh.add_argument("--authority-domain")

    status = sub.add_parser("status", help="Show one bootstrap receipt or the latest retained receipt.")
    _common(status)
    status.add_argument("bootstrap_id", nargs="?")

    show = sub.add_parser("show", help="Show one retained bootstrap receipt.")
    _common(show)
    show.add_argument("bootstrap_id")

    history = sub.add_parser("history", help="Show immutable bootstrap history.")
    _common(history)
    history.add_argument("bootstrap_id")

    source = sub.add_parser("source", help="Show the preserved V0.9 bootstrap source path and provenance.")
    _common(source)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raw = ["status"]
    args: argparse.Namespace | None = None
    try:
        args = build_parser().parse_args(raw)
        if args.command == "inspect":
            payload, record = inspect_bootstrap(
                repo_root_value=args.repo_root,
                state_root_value=args.state_root,
                generated_at=args.generated_at,
                task_mode=args.task_mode,
                topic=args.topic,
                operator_instruction_ref=args.operator_instruction_ref,
                edit_authorized=args.edit_authorized,
                campaign_authorized=args.campaign_authorized,
                gpu_required_mode=args.gpu_required,
                python_value=args.python,
                allow_current_python=args.allow_current_python,
                acknowledge_dirty=args.acknowledge_dirty,
                write_scope=args.write_scope,
                external_evidence_roots=args.external_evidence_root,
                expected_validation=args.expected_validation,
                preflight_receipt_path=args.preflight_receipt,
                preflight_exemption_reason=args.preflight_exemption_reason,
                authority_domain=args.authority_domain,
                collection=args.collection,
                limit=args.limit,
                store_receipt=not args.no_store,
            )
            result = with_guidance(
                {"ok": payload["readiness"] in READY_STATUSES, "bootstrap": payload, "record_sha256": record.get("record_sha256") if record else None},
                status=payload["readiness"],
                reason_codes=[f"bootstrap_{payload['readiness'].lower()}"],
                problems=[row["message"] for row in payload["checks"] if row["status"] == "BLOCKED"],
                warnings=[row["message"] for row in payload["checks"] if row["status"] == "WARN"],
                next_actions=(
                    [{"action": "begin_or_resume_work", "description": "Bootstrap passed. Open the workspace or start the selected work session.", "command": "python -m tools.eiel workspace open --bootstrap-record " + payload["bootstrap_id"]}]
                    if payload["readiness"] in READY_STATUSES
                    else [{"action": "repair_bootstrap_blocker", "description": "Repair the listed bootstrap blockers; do not edit or launch work yet.", "command": _bootstrap_repair_command(payload, state_root=args.state_root)}]
                ),
                related_records=[payload["bootstrap_id"]] if record else [],
            )
            emit_payload(result, output=args.output, human_text=_human(result), json_output=not args.human)
            return 0 if payload["readiness"] in READY_STATUSES else 2
        if args.command == "refresh":
            previous = bootstrap_store(args.state_root).get(
                validate_prefixed_id(args.bootstrap_id, BOOTSTRAP_ID_PREFIX, label="bootstrap")
            )
            previous_repository = dict(dict(previous.get("payload") or {}).get("repository") or {})
            live_repository = git_repository_identity(args.repo_root or Path.cwd())
            previous_root = Path(str(previous_repository.get("root") or "")).expanduser().resolve()
            if previous_root != Path(str(live_repository["root"])).expanduser().resolve():
                raise ActorError(
                    "Bootstrap refresh must stay in the receipt's original repository worktree",
                    code="bootstrap_refresh_repository_mismatch",
                    problems=[f"receipt root: {previous_root}", f"requested root: {live_repository['root']}"],
                )
            refresh_argv = bootstrap_refresh_arguments(
                previous,
                repo_root=live_repository["root"],
                state_root=args.state_root,
                requested_authority_domain=args.authority_domain,
                acknowledge_dirty=bool(live_repository.get("dirty")),
            )
            refreshed_args = build_parser().parse_args(refresh_argv[1:])
            refreshed_args.generated_at = args.generated_at
            payload, record = inspect_bootstrap(
                repo_root_value=refreshed_args.repo_root,
                state_root_value=refreshed_args.state_root,
                generated_at=refreshed_args.generated_at,
                task_mode=refreshed_args.task_mode,
                topic=refreshed_args.topic,
                operator_instruction_ref=refreshed_args.operator_instruction_ref,
                edit_authorized=refreshed_args.edit_authorized,
                campaign_authorized=refreshed_args.campaign_authorized,
                gpu_required_mode=refreshed_args.gpu_required,
                python_value=refreshed_args.python,
                allow_current_python=refreshed_args.allow_current_python,
                acknowledge_dirty=refreshed_args.acknowledge_dirty,
                write_scope=refreshed_args.write_scope,
                external_evidence_roots=refreshed_args.external_evidence_root,
                expected_validation=refreshed_args.expected_validation,
                preflight_receipt_path=refreshed_args.preflight_receipt,
                preflight_exemption_reason=refreshed_args.preflight_exemption_reason,
                authority_domain=refreshed_args.authority_domain,
                collection=refreshed_args.collection,
                limit=refreshed_args.limit,
                store_receipt=True,
            )
            result = with_guidance(
                {
                    "ok": payload["readiness"] in READY_STATUSES,
                    "bootstrap": payload,
                    "record_sha256": record.get("record_sha256"),
                    "refreshed_from": previous["record_id"],
                },
                status=payload["readiness"],
                reason_codes=[f"bootstrap_{payload['readiness'].lower()}", "bootstrap_refreshed"],
                problems=[row["message"] for row in payload["checks"] if row["status"] == "BLOCKED"],
                warnings=[row["message"] for row in payload["checks"] if row["status"] == "WARN"],
                next_actions=(
                    [{"action": "open_workspace", "description": "Open the current Workspace with this refreshed Bootstrap.", "command": "python -m tools.eiel workspace open --bootstrap-record " + payload["bootstrap_id"]}]
                    if payload["readiness"] in READY_STATUSES
                    else [{"action": "repair_bootstrap_blocker", "description": "Repair the listed Bootstrap blockers; do not edit or launch work yet.", "command": _bootstrap_repair_command(payload, state_root=args.state_root)}]
                ),
                related_records=[previous["record_id"], payload["bootstrap_id"]],
            )
            emit_payload(result, output=args.output, human_text=_human(result), json_output=not args.human)
            return 0 if payload["readiness"] in READY_STATUSES else 2
        if args.command in {"status", "show"}:
            store = bootstrap_store(args.state_root)
            record = store.get(args.bootstrap_id) if getattr(args, "bootstrap_id", None) else latest_bootstrap_record(args.state_root)
            if record is None:
                raise ActorError("no bootstrap receipt exists", code="bootstrap_receipt_missing")
            result = with_guidance(
                {"ok": record.get("status") in READY_STATUSES, "bootstrap": record.get("payload"), "record": record},
                status=str(record.get("status")),
                reason_codes=[f"bootstrap_{str(record.get('status')).lower()}"],
                next_actions=[{"action": "open_workspace", "description": "Use this receipt to open the operational workspace.", "command": f"python -m tools.eiel workspace open --bootstrap-record {record['record_id']}"}],
                related_records=[record["record_id"]],
            )
            emit_payload(result, output=args.output, human_text=_human(result), json_output=not args.human)
            return 0 if record.get("status") in READY_STATUSES else 2
        if args.command == "history":
            bootstrap_id = validate_prefixed_id(args.bootstrap_id, BOOTSTRAP_ID_PREFIX, label="bootstrap")
            result = {"ok": True, "bootstrap_id": bootstrap_id, "history": bootstrap_store(args.state_root).history(bootstrap_id)}
            emit_payload(result, output=args.output, human_text=f"Bootstrap history: {len(result['history'])} revisions", json_output=not args.human)
            return 0
        if args.command == "source":
            path = repository_root() / BOOTSTRAP_SOURCE_RELATIVE
            result = {"ok": path.is_file(), "source_path": str(path), "source_exists": path.is_file(), "schema_version": BOOTSTRAP_SCHEMA_VERSION}
            emit_payload(result, output=args.output, human_text=str(path), json_output=not args.human)
            return 0 if path.is_file() else 2
        raise ActorError(f"unsupported bootstrap command: {args.command}")
    except ActorError as exc:
        command = raw[0] if raw else "unknown"
        emit_actor_error(
            exc,
            actor=ACTOR_NAME,
            command=command,
            raw_args=raw,
            parsed_args=args,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
