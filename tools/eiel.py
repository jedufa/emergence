"""Unified local CLI dispatcher for EIEL operator-facing tools."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from collections.abc import Sequence
from typing import Final

# Support both repository-root imports (``engine.*``) and the simulator's
# historical top-level package imports (``simulation.*``, ``evolution.*``)
# without requiring callers to handcraft PYTHONPATH.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ENGINE_ROOT = _REPOSITORY_ROOT / "engine"
if _ENGINE_ROOT.is_dir() and str(_ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENGINE_ROOT))

from tools.actors.release import version_text

# Public compatibility names are retained exactly.  The order below is the
# intended Codex operating path, not an alphabetical accident.
MANAGERS: Final[dict[str, tuple[str, str]]] = {
    "bootstrap": (
        "Verify repository, environment, authority, scope, and validation readiness.",
        "tools.actors.bootstrap",
    ),
    "workspace": (
        "Open the consolidated Codex context, audit retained state, or export a resume handoff.",
        "tools.actors.workspace",
    ),
    "workManager": (
        "Track durable work items, priorities, dependencies, blockers, and portfolio history.",
        "tools.actors.work_manager",
    ),
    "sessionManager": (
        "Track resumable Codex sessions, interruption points, takeovers, commands, and handoffs.",
        "tools.actors.session_manager",
    ),
    "missionPlanner": (
        "Structure already-authorized scientific or technical mission intent.",
        "tools.actors.mission_planner",
    ),
    "qualificationManager": (
        "Check and record bounded technical qualification authority.",
        "tools.actors.qualification_manager",
    ),
    "resourceManager": (
        "Inspect resources and produce advisory, non-binding admission recommendations.",
        "tools.actors.resource_manager",
    ),
    "evidenceManager": (
        "Apply the existing read-only EDL to bounded procedural evidence decisions.",
        "tools.actors.evidence_manager",
    ),
    "bridgeManager": (
        "Use the optional repository-local ChatGPT transport only when explicitly requested.",
        "tools.actors.bridge_manager",
    ),
}

# Core utilities share the deterministic actor state but do not own lifecycle
# authority and are therefore deliberately not called managers.
CORE_TOOLS: Final[dict[str, tuple[str, str]]] = {
    "plan": (
        "Project, inspect, diff, or explain non-authoritative CRCA execution identity.",
        "eiel_composition.plan_identity_cli",
    ),
    "token": (
        "Capture and export WRA-1 Work Item token evidence through ccusage.",
        "tools.actors.token_cli",
    ),
    "science": (
        "Run one self-resolving EIEL Science Request V1; validate/plan are optional diagnostics.",
        "eiel_composition.science_instrument_cli",
    ),
    "expedition": (
        "Validate and deterministically compose no-claims Adaptive Expedition V1 artifacts.",
        "eiel_composition.adaptive_expedition.cli",
    ),
    "storage": (
        "Reclaim exact lifecycle material before admission and reconcile merged-work evidence closeout.",
        "tools.actors.storage_lifecycle_cli",
    ),
}

# The execution front door is a core utility, not a manager.  It composes the
# existing campaign authorities and therefore does not own another lifecycle
# database or scheduler.
EXECUTION_TOOLS: Final[dict[str, tuple[str, str]]] = {
    "execute": (
        "Execute one direct canonical request through Launchpad; science authors use science run.",
        "eiel_composition.execution_cli",
    ),
    "validate": (
        "Preflight or mechanically validate one canonical execution request.",
        "eiel_composition.execution_cli",
    ),
}


def _manager_main(name: str):
    module_path = MANAGERS[name][1]
    module = importlib.import_module(module_path)
    return module.main


def _core_tool_main(name: str):
    module = importlib.import_module(CORE_TOOLS[name][1])
    return module.main


# Optional unattended execution is loaded lazily.  Removing tools/scheduler
# must not affect imports, help, or any manual operator workflow.
OPTIONAL_TOOLS: Final[dict[str, tuple[str, str]]] = {
    "assignment": (
        "Compile explicit Dispatch Requests into immutable Assignments.",
        "assignment_main",
    ),
    "scheduler": (
        "Optionally launch one fresh worker per Assignment; manual operation remains primary.",
        "scheduler_main",
    ),
}

ALIASES: Final[dict[str, str]] = {
    "bridge-manager": "bridgeManager",
    "work-manager": "workManager",
    "session-manager": "sessionManager",
    "evidence-manager": "evidenceManager",
    "qualification-manager": "qualificationManager",
    "resource-manager": "resourceManager",
    "mission-planner": "missionPlanner",
    "assignment-linker": "assignment",
    "worker-scheduler": "scheduler",
    "work": "workManager",
    "session": "sessionManager",
}

# Small top-level recovery shortcuts.  They do not create new authority or
# bypass any manager; each delegates to the read-only Workspace surface.
WORKSPACE_SHORTCUTS: Final[dict[str, list[str]]] = {
    "start": ["open"],
    "resume": ["open"],
    "status": ["open"],
    "audit": ["audit"],
    "handoff": ["handoff"],
}


def render_help() -> str:
    groups = (
        (
            "Start and recovery",
            ("bootstrap", "workspace", "workManager", "sessionManager"),
        ),
        (
            "Mission guardrails",
            (
                "missionPlanner",
                "qualificationManager",
                "resourceManager",
                "evidenceManager",
            ),
        ),
        ("Optional transport", ("bridgeManager",)),
    )
    core_groups = (
        ("Execution identity", ("plan",)),
        ("Science instrument", ("science",)),
        ("Adaptive Expedition", ("expedition",)),
        ("Storage lifecycle", ("storage",)),
        ("Resource accounting", ("token",)),
    )
    execution_groups = (("Canonical execution", ("execute", "validate")),)
    optional_groups = (("Optional unattended execution", ("assignment", "scheduler")),)
    width = max(len(name) for name in (*MANAGERS, *CORE_TOOLS, *OPTIONAL_TOOLS))
    lines = [
        version_text(),
        "",
        "Start here:",
        "  python -m tools.eiel start  # compact structured opening",
        "",
        "The command opens the consolidated Workspace. If a fresh bootstrap is",
        "required, it returns the exact fail-closed bootstrap command rather than",
        "guessing.",
        "",
        "Read-only Workspace shortcuts:",
        "  start   Inspect repository state and print the next safe command; does not start a Session.",
        "  resume  Inspect repository state and print the exact resume command; does not resume a Session.",
        "  status  Inspect the same current Workspace state without mutation.",
        "  audit   Check retained record consistency without repair or mutation.",
        "  handoff Export a deterministic resume handoff without changing state.",
        "",
        "Usage:",
        "  python -m tools.eiel <tool> <command> [options]",
        "  python -m tools.eiel {start|resume|status|audit|handoff} [workspace options]",
        "  python -m tools.eiel {execute|validate} <request.json> [options]",
        "  python -m tools.eiel science run <request.json> [--human]  # normal science workflow",
        "  python -m tools.eiel science {validate|plan} <request.json>  # optional diagnostics",
        "  python -m tools.eiel expedition --help  # no-claims composition and replay",
        "  python -m tools.eiel plan {project|inspect|diff|explain} ...",
        "  python -m tools.eiel start --full  # complete forensic Workspace payload",
        "",
        "Presentation:",
        "  Codex uses structured JSON by default. Human operators add --human.",
        "",
    ]
    for title, names in groups:
        lines.append(f"{title}:")
        for name in names:
            lines.append(f"  {name.ljust(width)}  {MANAGERS[name][0]}")
        lines.append("")
    for title, names in core_groups:
        lines.append(f"{title}:")
        for name in names:
            lines.append(f"  {name.ljust(width)}  {CORE_TOOLS[name][0]}")
        lines.append("")
    for title, names in execution_groups:
        lines.append(f"{title}:")
        for name in names:
            lines.append(f"  {name.ljust(width)}  {EXECUTION_TOOLS[name][0]}")
        lines.append("")
    for title, names in optional_groups:
        lines.append(f"{title}:")
        for name in names:
            lines.append(f"  {name.ljust(width)}  {OPTIONAL_TOOLS[name][0]}")
        lines.append("")
    lines.extend(
        [
            "Core managers are deterministic local CLIs. They contain no AI and run no",
            "services or simulation campaigns. The optional Scheduler launches Codex",
            "workers only after an explicit Dispatch Request; deleting it leaves every",
            "manual workflow available.",
            "",
            "Use 'eiel <tool> --help' for tool-specific commands.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(render_help(), end="")
        return 0
    if args[0] in {"-V", "--version", "version"}:
        print(version_text())
        return 0
    if args[0] == "token-report":
        return _core_tool_main("token")(["report", *args[1:]])
    execution_tool = EXECUTION_TOOLS.get(args[0])
    if execution_tool is not None:
        module = importlib.import_module(execution_tool[1])
        return module.main(args)
    shortcut = WORKSPACE_SHORTCUTS.get(args[0])
    if shortcut is not None:
        forwarded = list(args[1:])
        # The top-level start/resume/status shortcuts are the ordinary Codex
        # front door.  V1.11 displays the bounded structured projection by
        # default while preserving ``--full`` and ``--human`` escape hatches.
        if args[0] in {"start", "resume", "status"} and not any(
            flag in forwarded for flag in ("--compact", "--full", "--human")
        ):
            forwarded.insert(0, "--compact")
        return _manager_main("workspace")([*shortcut, *forwarded])
    actor = ALIASES.get(args[0], args[0])
    entry = MANAGERS.get(actor)
    if entry is not None:
        return _manager_main(actor)(args[1:])
    if actor == "storage" and len(args) > 1 and args[1] == "finish-run":
        # A bounded terminal operation of the existing storage front door.
        return importlib.import_module("eiel_composition.report_only_closeout").main(args[2:])
    core = CORE_TOOLS.get(actor)
    if core is not None:
        return _core_tool_main(actor)(args[1:])
    optional = OPTIONAL_TOOLS.get(actor)
    if optional is not None:
        try:
            from tools.scheduler import cli as scheduler_cli
        except (ImportError, ModuleNotFoundError) as exc:
            print(
                "Optional EIEL Scheduler is not installed. Manual Bootstrap, Workspace, "
                "Work, Session, Mission, Qualification, Resource, Evidence, and Bridge "
                "workflows remain available.",
                file=sys.stderr,
            )
            return 2
        return getattr(scheduler_cli, optional[1])(args[1:])
    print(f"Unknown EIEL actor/tool: {args[0]!r}\n", file=sys.stderr)
    print(render_help(), file=sys.stderr, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
