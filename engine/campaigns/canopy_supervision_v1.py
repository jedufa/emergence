"""Shared, read-only Living Lineage Canopy relay supervision for run front doors."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Final

from .canonical import canonical_sha256, pretty_json


DEFAULT_CANOPY_RELAY_ROOT: Final[Path] = Path(
    public_runtime_path('live/canopy-relays')
)
CANOPY_RELAY_POLL_SECONDS: Final[int] = 20
CANOPY_RELAY_PROCESS_BUDGET_V1: Final[int] = 64
CANOPY_SUPERVISION_SCHEMA_VERSION: Final[str] = (
    "eiel_canopy_relay_supervision_v1"
)


class CanopySupervisionError(ValueError):
    """A read-only Canopy relay cannot be safely supervised."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_mapping(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanopySupervisionError(f"{name} is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise CanopySupervisionError(f"{name} must be an object")
    return dict(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(pretty_json(value), encoding="utf-8")
    temporary.replace(path)


def canopy_relay_process_matches_v1(pid: int, *, output_root: Path) -> bool:
    if type(pid) is not int or pid <= 0:
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return (
        b"living_lineage_canopy.python.relay_cli watch" in command
        and str(output_root).encode() in command
    )


def start_canopy_relays_v1(
    *,
    plan: Mapping[str, Any],
    run_root: Path,
    relay_root: Path = DEFAULT_CANOPY_RELAY_ROOT,
    repository_root: Path | None = None,
    python_executable: Path | None = None,
    enabled: bool | None = None,
    group_prefix: str = "science",
    popen_fn: Callable[..., Any] = subprocess.Popen,
    process_matcher: Callable[..., bool] = canopy_relay_process_matches_v1,
) -> dict[str, Any]:
    """Start or reuse one detached read-only relay for every planned cell."""

    observability = plan.get("observability")
    if enabled is None:
        if not isinstance(observability, Mapping):
            raise CanopySupervisionError(
                "CANOPY_BLOCKED: materialized observability contract is absent"
            )
        enabled = observability.get("living_lineage_canopy") is True
    if type(enabled) is not bool:
        raise CanopySupervisionError("CANOPY_BLOCKED: enabled must be boolean")
    cells = plan.get("cells")
    if not enabled:
        return {
            "schema_version": CANOPY_SUPERVISION_SCHEMA_VERSION,
            "state": "disabled_by_request",
            "authority": "read_only_external_projection_v1",
            "simulation_state_mutation": False,
            "manual_internal_decisions_required": 0,
            "run_root": str(run_root),
            "relays": [],
        }
    if not isinstance(cells, list) or not cells:
        raise CanopySupervisionError("CANOPY_BLOCKED: materialized cells are absent")
    cell_ids = [
        str(cell.get("cell_id")) if isinstance(cell, Mapping) else "" for cell in cells
    ]
    if any(not cell_id or Path(cell_id).name != cell_id for cell_id in cell_ids) or len(
        set(cell_ids)
    ) != len(cell_ids):
        raise CanopySupervisionError(
            "CANOPY_BLOCKED: materialized cell identities are unsafe or non-unique"
        )
    if len(cell_ids) > CANOPY_RELAY_PROCESS_BUDGET_V1:
        return record_canopy_observability_degraded_v1(
            run_root,
            plan=plan,
            reason=(
                "planned cell count exceeds the bounded relay process budget "
                f"({len(cell_ids)} > {CANOPY_RELAY_PROCESS_BUDGET_V1})"
            ),
            details={
                "requested_cell_count": len(cell_ids),
                "relay_process_budget": CANOPY_RELAY_PROCESS_BUDGET_V1,
                "relay_start_policy": "bounded_observability_degradation_v1",
            },
        )
    source_root = (
        repository_root.expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    relay_module = (
        source_root / "tools/visualization/living_lineage_canopy/python/relay_cli.py"
    )
    if not relay_module.is_file():
        raise CanopySupervisionError(
            "CANOPY_BLOCKED: external read-only relay tooling is unavailable"
        )
    interpreter = (
        python_executable.expanduser().absolute()
        if python_executable is not None
        else Path(sys.executable).absolute()
    )
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise CanopySupervisionError(
            "CANOPY_BLOCKED: relay Python environment is unavailable"
        )
    plan_sha256 = plan.get("plan_sha256")
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64:
        raise CanopySupervisionError(
            "CANOPY_BLOCKED: materialized plan identity is absent"
        )
    if not group_prefix or Path(group_prefix).name != group_prefix:
        raise CanopySupervisionError("CANOPY_BLOCKED: relay group prefix is unsafe")
    relay_binding_sha256 = canonical_sha256(
        {"plan_sha256": plan_sha256, "run_root": str(run_root.resolve())}
    )
    group_id = f"{group_prefix}-{relay_binding_sha256[:20]}"
    group_root = relay_root.expanduser().resolve() / group_id
    supervision_path = run_root / "canopy_relay_supervision.json"
    previous_rows: dict[str, dict[str, Any]] = {}
    if supervision_path.is_file():
        previous = _read_mapping(supervision_path, name="Canopy relay supervision")
        previous_core = dict(previous)
        previous_sha256 = previous_core.pop("receipt_sha256", None)
        if (
            previous_sha256 != canonical_sha256(previous_core)
            or previous.get("schema_version") != CANOPY_SUPERVISION_SCHEMA_VERSION
            or previous.get("plan_sha256") != plan_sha256
            or previous.get("relay_group_id") != group_id
            or not isinstance(previous.get("relays"), list)
        ):
            raise CanopySupervisionError(
                "CANOPY_BLOCKED: prior relay supervision belongs to another run identity"
            )
        previous_rows = {
            str(row["cell_id"]): dict(row)
            for row in previous["relays"]
            if isinstance(row, Mapping) and isinstance(row.get("cell_id"), str)
        }
    child_environment = dict(os.environ)
    child_environment.pop("PYTHONPATH", None)
    started: list[Any] = []
    rows: list[dict[str, Any]] = []
    try:
        for cell_id in cell_ids:
            output_root = group_root / cell_id
            output_root.mkdir(parents=True, exist_ok=True)
            prior = previous_rows.get(cell_id, {})
            if process_matcher(prior.get("pid"), output_root=output_root):
                rows.append({**prior, "reused": True})
                continue
            manifest_path = output_root / "relay_manifest.json"
            status_path = output_root / "relay_status.json"
            if manifest_path.is_file():
                manifest = _read_mapping(manifest_path, name="Canopy relay manifest")
                if (
                    manifest.get("source_root") != str(run_root.resolve())
                    or manifest.get("relay_id") != f"{group_id}-{cell_id}"
                ):
                    raise CanopySupervisionError(
                        "CANOPY_BLOCKED: prior relay manifest belongs to another source"
                    )
                status = (
                    _read_mapping(status_path, name="Canopy relay status")
                    if status_path.is_file()
                    else {}
                )
                if status.get("state") == "integrity_error":
                    raise CanopySupervisionError(
                        "CANOPY_BLOCKED: prior relay reported an integrity error"
                    )
                if status.get("state") in {"source_complete", "source_interrupted"}:
                    rows.append(
                        {
                            "cell_id": cell_id,
                            "pid": None,
                            "output_root": str(output_root),
                            "manifest_path": str(manifest_path),
                            "log_path": str(output_root / "relay_watch.log"),
                            "reused": True,
                            "terminal_state": status["state"],
                        }
                    )
                    continue
            argv = [
                str(interpreter),
                "-m",
                "tools.visualization.living_lineage_canopy.python.relay_cli",
                "watch",
                "--source-root",
                str(run_root),
                "--output-root",
                str(output_root),
                "--source-kind",
                "gv2_checkpoint_bundle_v2",
                "--cell-id",
                cell_id,
                "--poll-seconds",
                str(CANOPY_RELAY_POLL_SECONDS),
                "--relay-id",
                f"{group_id}-{cell_id}",
            ]
            with (output_root / "relay_watch.log").open("ab") as log:
                process = popen_fn(
                    argv,
                    cwd=source_root,
                    env=child_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
            started.append(process)
            rows.append(
                {
                    "cell_id": cell_id,
                    "pid": process.pid,
                    "output_root": str(output_root),
                    "manifest_path": str(output_root / "relay_manifest.json"),
                    "log_path": str(output_root / "relay_watch.log"),
                    "reused": False,
                }
            )
    except Exception as exc:
        for process in started:
            try:
                process.terminate()
            except OSError:
                pass
        if isinstance(exc, CanopySupervisionError):
            raise
        raise CanopySupervisionError(
            f"CANOPY_BLOCKED: relay launch failed: {type(exc).__name__}: {exc}"
        ) from exc
    core = {
        "schema_version": CANOPY_SUPERVISION_SCHEMA_VERSION,
        "state": "watching",
        "authority": "read_only_external_projection_v1",
        "simulation_state_mutation": False,
        "manual_internal_decisions_required": 0,
        "run_root": str(run_root),
        "plan_sha256": plan_sha256,
        "relay_group_id": group_id,
        "relay_group_root": str(group_root),
        "poll_seconds": CANOPY_RELAY_POLL_SECONDS,
        "started_at_utc": _now(),
        "relays": rows,
    }
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    _atomic_json(supervision_path, receipt)
    return receipt


def mark_canopy_source_interrupted_v1(run_root: Path, *, reason: str) -> dict[str, Any]:
    core = {
        "schema_version": "eiel_canopy_source_terminal_v1",
        "state": "source_interrupted",
        "reason": reason,
        "recorded_at_utc": _now(),
    }
    record = {**core, "record_sha256": canonical_sha256(core)}
    _atomic_json(run_root / "canopy_source_terminal.json", record)
    return record


def record_canopy_observability_degraded_v1(
    run_root: Path,
    *,
    plan: Mapping[str, Any],
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record subordinate observability failure without gating simulation."""

    core = {
        "schema_version": CANOPY_SUPERVISION_SCHEMA_VERSION,
        "state": "observability_degraded",
        "authority": "read_only_external_projection_v1",
        "authoritative_runtime_blocked": False,
        "simulation_state_mutation": False,
        "manual_internal_decisions_required": 0,
        "run_root": str(run_root),
        "plan_sha256": plan.get("plan_sha256"),
        "recorded_at_utc": _now(),
        "reason": reason,
        "relays": [],
    }
    if details:
        core["degradation_details"] = dict(details)
    receipt = {**core, "receipt_sha256": canonical_sha256(core)}
    _atomic_json(run_root / "canopy_relay_supervision.json", receipt)
    return receipt


__all__ = [
    "CANOPY_RELAY_POLL_SECONDS",
    "CANOPY_RELAY_PROCESS_BUDGET_V1",
    "CANOPY_SUPERVISION_SCHEMA_VERSION",
    "DEFAULT_CANOPY_RELAY_ROOT",
    "CanopySupervisionError",
    "canopy_relay_process_matches_v1",
    "mark_canopy_source_interrupted_v1",
    "record_canopy_observability_degraded_v1",
    "start_canopy_relays_v1",
]
