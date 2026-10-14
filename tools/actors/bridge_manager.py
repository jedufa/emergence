"""Optional, explicit facade over the repository-local ChatGPT Bridge Manager.

The bridge is transport only.  It is never required by bootstrap, workspace,
work/session records, missions, qualification, resources, evidence, or the
Knowledge Librarian.  No browser or bridge command runs unless the operator
explicitly invokes this actor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.actors.common import (
    ActorArgumentParser,
    ActorError,
    emit_actor_error,
    emit_payload,
    repository_root,
    with_guidance,
)

ACTOR_NAME = "bridgeManager"
STATUS_SCHEMA_VERSION = "eiel_optional_bridge_status_v1"
NODE_MINIMUM_MAJOR = 20
BRIDGE_ROOT_ENV = "EIEL_BRIDGE_ROOT"


def bridge_root(value: str | Path | None = None) -> Path:
    if value is not None:
        return Path(value).expanduser().resolve()
    configured = os.environ.get(BRIDGE_ROOT_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return repository_root() / "bridge"


def installation_status(value: str | Path | None = None, *, probe_node: bool = True) -> dict[str, Any]:
    root = bridge_root(value)
    manager = root / "chatgpt-bridge-manager"
    browser_bridge = root / "chatgpt-bridge"
    package = root / "package.json"
    present = root.is_dir() and manager.is_file() and browser_bridge.is_file() and package.is_file()
    node_path = shutil.which("node") if probe_node else None
    node_version = None
    node_major = None
    node_ok = None
    problems: list[str] = []
    if present and probe_node:
        if not node_path:
            problems.append("Node.js is not available; the optional bridge cannot be invoked on this host.")
            node_ok = False
        else:
            result = subprocess.run(
                [node_path, "--version"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
            node_version = (result.stdout or result.stderr).strip() or None
            match = re.search(r"(\d+)", node_version or "")
            node_major = int(match.group(1)) if match else None
            node_ok = result.returncode == 0 and node_major is not None and node_major >= NODE_MINIMUM_MAJOR
            if not node_ok:
                problems.append(f"Node.js {NODE_MINIMUM_MAJOR}+ is required; observed {node_version or 'unknown'}.")
    versions: dict[str, Any] = {}
    if present:
        try:
            package_payload = json.loads(package.read_text(encoding="utf-8"))
            versions["bridge"] = package_payload.get("version")
            versions["manager"] = "1.0.0"
        except (OSError, UnicodeError, json.JSONDecodeError):
            problems.append("bridge/package.json could not be parsed")
    if not present:
        status = "OPTIONAL_NOT_INSTALLED"
        problems.append("The optional repository-local bridge is not installed.")
    elif probe_node and not node_ok:
        status = "OPTIONAL_UNAVAILABLE_ON_HOST"
    else:
        status = "OPTIONAL_AVAILABLE"
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "required": False,
        "status": status,
        "bridge_root": str(root),
        "manager_path": str(manager),
        "browser_bridge_path": str(browser_bridge),
        "node_path": node_path,
        "node_version": node_version,
        "node_major": node_major,
        "node_minimum_major": NODE_MINIMUM_MAJOR,
        "versions": versions,
        "problems": problems,
        "authority_effect": "none",
        "automatic_execution": False,
        "ordinary_actor_workflow_blocked": False,
    }


def command_status(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    status = installation_status(args.bridge_root, probe_node=not args.no_node_probe)
    available = status["status"] == "OPTIONAL_AVAILABLE"
    next_actions = []
    if available:
        next_actions.append({
            "action": "inspect_bridge_interfaces",
            "description": "List bridge interfaces only when bridge transport is desired.",
            "command": "python -m tools.eiel bridgeManager list --json",
        })
    else:
        next_actions.append({
            "action": "continue_without_bridge",
            "description": "Continue using the ordinary repository actor workflow; the bridge is optional.",
            "command": "python -m tools.eiel workspace open",
        })
    payload = with_guidance(
        {"ok": True, "bridge": status},
        status=status["status"],
        reason_codes=[status["status"].lower()],
        problems=status["problems"],
        warnings=["Bridge transport is non-authoritative and explicit-only."],
        next_actions=next_actions,
    )
    return payload, 0


def command_delegate(raw: Sequence[str], *, root: str | Path | None = None) -> int:
    status = installation_status(root, probe_node=True)
    if status["status"] != "OPTIONAL_AVAILABLE":
        raise ActorError(
            "optional ChatGPT bridge is unavailable",
            code="optional_bridge_unavailable",
            problems=status["problems"],
            next_actions=[{
                "action": "continue_without_bridge",
                "description": "Use bootstrap, workspace, work/session, and the existing managers without bridge transport.",
                "command": "python -m tools.eiel workspace open",
            }],
        )
    manager = Path(status["manager_path"])
    result = subprocess.run([str(manager), *raw], check=False)
    return int(result.returncode)


def _human(payload: Mapping[str, Any]) -> str:
    bridge = dict(payload.get("bridge") or {})
    lines = [
        f"Bridge: {bridge.get('status')}",
        "Required: no",
        f"Root: {bridge.get('bridge_root')}",
    ]
    if bridge.get("node_version"):
        lines.append(f"Node: {bridge['node_version']}")
    for problem in bridge.get("problems") or []:
        lines.append(f"Note: {problem}")
    lines.append("Core actor workflow remains available without the bridge.")
    return "\n".join(lines) + "\n"


def render_help() -> str:
    return """Optional EIEL ChatGPT Bridge facade

Usage:
  python -m tools.eiel bridgeManager status [--bridge-root PATH] [--no-node-probe] [--human]
  python -m tools.eiel bridgeManager <bridge-command> [bridge arguments...]

Examples:
  python -m tools.eiel bridgeManager status
  python -m tools.eiel bridgeManager init
  python -m tools.eiel bridgeManager list --json
  python -m tools.eiel bridgeManager next actor-managers report.md evidence.zip

The bridge is optional, explicit-only transport. Ordinary EIEL operation never
requires it. Use 'run status <alias>' to invoke the underlying manager's alias
status command because top-level 'status' reports optional installation health.
"""


def build_status_parser() -> argparse.ArgumentParser:
    parser = ActorArgumentParser(prog="python -m tools.eiel bridgeManager status", description=__doc__)
    parser.add_argument("--bridge-root")
    parser.add_argument("--no-node-probe", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--human", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help", "help"}:
        print(render_help(), end="")
        return 0
    args: argparse.Namespace | None = None
    try:
        if raw[0].startswith("-"):
            raise ActorError(
                "bridge command is required before presentation options",
                code="bridge_command_missing",
                next_actions=[{"command": "python -m tools.eiel bridgeManager --help"}],
            )
        if raw[0] == "status":
            args = build_status_parser().parse_args(raw[1:])
            payload, code = command_status(args)
            emit_payload(payload, output=args.output, human_text=_human(payload), json_output=not args.human)
            return code
        if raw[0] == "run":
            if len(raw) == 1:
                raise ActorError(
                    "bridge command missing after 'run'",
                    code="bridge_command_missing",
                    next_actions=[{"command": "python -m tools.eiel bridgeManager --help"}],
                )
            return command_delegate(raw[1:])
        return command_delegate(raw)
    except ActorError as exc:
        command = raw[0] if raw else "status"
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
