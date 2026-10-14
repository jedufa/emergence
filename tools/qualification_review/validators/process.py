"""Exact PID ownership and terminal-publication/process-exit convergence."""

from __future__ import annotations

from ..io import InputStore, ReviewInputError, json_pointer
from ..model import CheckResult


def validate_process_convergence(spec: dict, store: InputStore) -> CheckResult:
    result = CheckResult(spec["id"], "process_convergence")
    try:
        document = store.read_json(spec["path"])
        snapshots = json_pointer(document, spec.get("snapshots_pointer", "/snapshots"))
        if not isinstance(snapshots, list) or not snapshots:
            raise ReviewInputError("process convergence snapshots must be a non-empty list")
        expected_pids = set(spec["expected_pids"])
        owner_id = spec["owner_id"]
        terminal_index: int | None = None
        convergence_index: int | None = None
        observed_pids: set[int] = set()
        final_state: object = None
        final_owned: list[int] = []
        for index, snapshot in enumerate(snapshots):
            state = json_pointer(snapshot, spec.get("source_state_pointer", "/source_state"))
            final_state = state
            processes = json_pointer(snapshot, spec.get("processes_pointer", "/processes"))
            if not isinstance(processes, list):
                raise ReviewInputError(f"snapshot {index} processes value is not a list")
            owned: list[int] = []
            for process in processes:
                pid = process.get(spec.get("pid_field", "pid"))
                owner = process.get(spec.get("owner_field", "owner_id"))
                if owner == owner_id:
                    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
                        result.fail(f"snapshot {index} has invalid owned PID: {pid!r}")
                        continue
                    owned.append(pid)
                    observed_pids.add(pid)
            if len(owned) != len(set(owned)):
                result.fail(f"snapshot {index} contains duplicate owned PIDs")
            unexpected = set(owned) - expected_pids
            if unexpected:
                result.fail(f"snapshot {index} contains unexpected owned PIDs: {sorted(unexpected)}")
            if state in spec["terminal_source_states"] and terminal_index is None:
                terminal_index = index
            if terminal_index is not None and index > terminal_index and convergence_index is None and not owned:
                convergence_index = index
            elif convergence_index is not None and owned:
                result.fail(
                    f"owned processes reappeared after convergence at snapshot {index}: {sorted(owned)}"
                )
            final_owned = owned
        if terminal_index is None:
            result.fail("no declared terminal source state was observed")
        if expected_pids - observed_pids:
            result.fail(
                f"expected owned PIDs were never observed: {sorted(expected_pids - observed_pids)}"
            )
        if convergence_index is None:
            result.fail("owned processes did not converge to exact absence after terminal publication")
        if final_state not in spec["terminal_source_states"]:
            result.fail(f"final snapshot is not in a terminal source state: {final_state!r}")
        if final_owned:
            result.fail(f"final snapshot still contains owned PIDs: {sorted(final_owned)}")
        result.details = {
            "path": spec["path"],
            "owner_id": owner_id,
            "expected_pids": sorted(expected_pids),
            "terminal_snapshot_index": terminal_index,
            "converged_snapshot_index": convergence_index,
            "snapshot_count": len(snapshots),
        }
    except ReviewInputError as exc:
        result.fail(str(exc))
    return result
