"""Resource admission policy for the single guarded Genome V2 CUDA build."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final


SCHEMA_VERSION: Final[str] = "genome_v2_build_admission_policy_v1"
PREFLIGHT_MIN_AVAILABLE_BYTES: Final[int] = 16 * 1024**3
MONITOR_MIN_AVAILABLE_BYTES: Final[int] = 2 * 1024**3
COMPILER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "c++",
        "cc1",
        "cc1plus",
        "cicc",
        "fatbinary",
        "g++",
        "gcc",
        "ninja",
        "nvcc",
        "ptxas",
    }
)


def build_admission_policy_v1() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "exclusive_lock_required": True,
        "overlapping_builds_allowed": 0,
        "maximum_monitored_build_trees": 1,
        "cpu_only_perf_build_fixture_allowed": False,
        "preflight_min_available_bytes": PREFLIGHT_MIN_AVAILABLE_BYTES,
        "monitor_min_available_bytes": MONITOR_MIN_AVAILABLE_BYTES,
        "compiler_names": sorted(COMPILER_NAMES),
        "second_tree_action": "terminate_guarded_process_group",
        "memory_floor_action": "terminate_guarded_process_group",
        "orphan_compiler_action": "fail_preflight",
        "scientific_campaign_during_build": "forbidden",
        "full_validation_during_build": "forbidden",
        "required_receipts": [
            "genome_v2_build_resource_preflight.json",
            "genome_v2_build_process_monitor.json",
            "genome_v2_build_resource_summary.json",
        ],
    }


def evaluate_build_snapshot_v1(
    *,
    memory: Mapping[str, int],
    processes: Sequence[Mapping[str, Any]],
    phase: str,
    guarded_root_pid: int | None = None,
) -> dict[str, Any]:
    if phase not in {"preflight", "monitor"}:
        raise ValueError("phase must be preflight or monitor")
    available = memory.get("available_bytes")
    if type(available) is not int or available < 0:
        raise ValueError("available_bytes must be a nonnegative integer")
    normalized = []
    for row in processes:
        pid = row.get("pid")
        ppid = row.get("ppid")
        comm = row.get("comm")
        command = row.get("command")
        if (
            type(pid) is not int
            or pid <= 0
            or type(ppid) is not int
            or ppid < 0
            or not isinstance(comm, str)
            or not isinstance(command, str)
        ):
            raise ValueError("process inventory row is invalid")
        normalized.append(
            {"pid": pid, "ppid": ppid, "comm": comm, "command": command}
        )
    by_pid = {row["pid"]: row for row in normalized}
    compiler_rows = [
        row for row in normalized if row["comm"] in COMPILER_NAMES
    ]

    def build_root(row: Mapping[str, Any]) -> int:
        current = row
        seen: set[int] = set()
        selected = int(row["pid"])
        while int(current["pid"]) not in seen:
            seen.add(int(current["pid"]))
            if (
                current["comm"] == "ninja"
                or "torch.utils.cpp_extension" in current["command"]
                or "TORCH_EXTENSION_NAME=" in current["command"]
            ):
                selected = int(current["pid"])
            parent = by_pid.get(int(current["ppid"]))
            if parent is None:
                break
            current = parent
        return selected

    if phase == "monitor" and guarded_root_pid is not None:
        guarded_compilers = [
            row
            for row in compiler_rows
            if _is_descendant_v1(
                int(row["pid"]), guarded_root_pid, by_pid
            )
        ]
        external_compilers = [
            row for row in compiler_rows if row not in guarded_compilers
        ]
        roots = sorted(
            (
                {guarded_root_pid} if guarded_compilers else set()
            )
            | {build_root(row) for row in external_compilers}
        )
    else:
        roots = sorted({build_root(row) for row in compiler_rows})
    reasons: list[str] = []
    minimum = (
        PREFLIGHT_MIN_AVAILABLE_BYTES
        if phase == "preflight"
        else MONITOR_MIN_AVAILABLE_BYTES
    )
    if available < minimum:
        reasons.append("available_memory_below_admission_floor")
    if phase == "preflight" and roots:
        reasons.append("compiler_tree_already_active")
    if phase == "monitor" and len(roots) > 1:
        reasons.append("multiple_compiler_trees_active")
    if (
        phase == "monitor"
        and guarded_root_pid is not None
        and roots
        and not all(
            root == guarded_root_pid
            or _is_descendant_v1(root, guarded_root_pid, by_pid)
            or _is_descendant_v1(guarded_root_pid, root, by_pid)
            for root in roots
        )
    ):
        reasons.append("compiler_tree_outside_guarded_process_group")
    return {
        "schema_version": "genome_v2_build_resource_snapshot_v1",
        "phase": phase,
        "admitted": not reasons,
        "available_bytes": available,
        "minimum_available_bytes": minimum,
        "compiler_process_count": len(compiler_rows),
        "compiler_tree_roots": roots,
        "guarded_root_pid": guarded_root_pid,
        "reasons": reasons,
    }


def _is_descendant_v1(
    pid: int,
    ancestor: int,
    by_pid: Mapping[int, Mapping[str, Any]],
) -> bool:
    current = by_pid.get(pid)
    seen: set[int] = set()
    while current is not None and int(current["pid"]) not in seen:
        current_pid = int(current["pid"])
        if current_pid == ancestor:
            return True
        seen.add(current_pid)
        current = by_pid.get(int(current["ppid"]))
    return False


__all__ = [
    "COMPILER_NAMES",
    "MONITOR_MIN_AVAILABLE_BYTES",
    "PREFLIGHT_MIN_AVAILABLE_BYTES",
    "SCHEMA_VERSION",
    "build_admission_policy_v1",
    "evaluate_build_snapshot_v1",
]
