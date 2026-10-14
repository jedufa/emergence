#!/usr/bin/env python3
"""Compare the production lifecycle dispatcher in macro-off/on CUDA builds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.campaigns.canonical import canonical_sha256  # noqa: E402
from engine.campaigns.orchestration import atomic_write_json  # noqa: E402


SCHEMA_VERSION = "eiel_tensorworld_lifecycle_linear_native_parity_v1"
BASELINE_MODULE = "eiel_cuda_architecture_b_wdl_jungle_eed_v1"
CANDIDATE_MODULE = (
    "eiel_cuda_architecture_b_wdl_jungle_eed_lifecycle_linear_diagnostic_v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path: Path, artifact: Path, *, candidate: bool) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_module = CANDIDATE_MODULE if candidate else BASELINE_MODULE
    resource = payload.get("resource_receipt")
    if (
        not isinstance(payload, Mapping)
        or payload.get("module_name") != expected_module
        or payload.get("artifact_sha256") != _sha256(artifact)
        or not isinstance(resource, Mapping)
        or resource.get("lifecycle_linear_fast_path_enabled") is not candidate
        or resource.get("lifecycle_linear_fast_path_test_surface_enabled") is not True
    ):
        raise RuntimeError("lifecycle parity artifact manifest mode is invalid")
    return dict(payload)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load lifecycle parity module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "_test_only_lifecycle_death_cull_cell_v3", None)):
        raise RuntimeError("lifecycle parity module lacks its test-only dispatcher")
    return module


def _matched_source_commit(
    baseline_payload: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> str:
    baseline_commit = baseline_payload.get("committed_source_snapshot", {}).get(
        "build_source_commit"
    )
    candidate_commit = candidate_payload.get("committed_source_snapshot", {}).get(
        "build_source_commit"
    )
    if (
        not isinstance(baseline_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", baseline_commit) is None
        or not isinstance(candidate_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", candidate_commit) is None
        or baseline_commit != candidate_commit
    ):
        raise RuntimeError("lifecycle parity artifacts differ in source commit")
    return baseline_commit


def _cpu_state(
    *,
    live: tuple[tuple[int, int], ...],
    active_rows: tuple[tuple[int, int], ...] | None = None,
    terminal: Mapping[int, tuple[float, float]] | None = None,
    active_capacity: int = 8,
    slot_count: int = 8,
    pending_capacity: int = 8,
    pending_before: int = 1,
    contradictions: Mapping[int, tuple[bool, int]] | None = None,
) -> tuple[list[Any], dict[str, int]]:
    import torch

    agent_id_capacity = 64
    grid_height = 4
    grid_width = 4
    rows = active_rows if active_rows is not None else live
    active_ids = torch.full((active_capacity,), -7001, dtype=torch.int64)
    active_slots = torch.full((active_capacity,), -7002, dtype=torch.int32)
    for row, (agent_id, slot) in enumerate(rows):
        if row >= active_capacity:
            break
        active_ids[row] = agent_id
        active_slots[row] = slot
    active_count = torch.tensor([len(rows)], dtype=torch.int64)
    slot_to_id = torch.full((slot_count,), -1, dtype=torch.int64)
    id_to_slot = torch.full((agent_id_capacity,), -1, dtype=torch.int32)
    alive = torch.zeros((slot_count,), dtype=torch.bool)
    pos_y = torch.full((slot_count,), -1, dtype=torch.int32)
    pos_x = torch.full((slot_count,), -1, dtype=torch.int32)
    grid = torch.full((grid_height * grid_width,), -1, dtype=torch.int32)
    energy = torch.full((slot_count,), -31.25, dtype=torch.float32)
    health = torch.full((slot_count,), -32.5, dtype=torch.float32)
    policy_observation = torch.full((slot_count,), -33.75, dtype=torch.float64)
    rest_streak = torch.full((slot_count,), -34, dtype=torch.int32)
    cache_token = torch.full((slot_count,), -35, dtype=torch.int64)
    cache_cursor = torch.full((slot_count,), -36, dtype=torch.int32)
    cache_next = torch.full((slot_count,), -37, dtype=torch.int32)
    for agent_id, slot in live:
        alive[slot] = True
        slot_to_id[slot] = agent_id
        if 0 <= agent_id < agent_id_capacity:
            id_to_slot[agent_id] = slot
        y, x = divmod(slot, grid_width)
        pos_y[slot] = y
        pos_x[slot] = x
        grid[y * grid_width + x] = agent_id
        energy[slot] = 9.5 + slot
        health[slot] = 7.25 + slot
        policy_observation[slot] = 9.5 + slot
        rest_streak[slot] = 3 + slot
        cache_token[slot] = 1000 + agent_id
        cache_cursor[slot] = slot % 2
        cache_next[slot] = (slot + 1) % 2
    for slot, values in dict(terminal or {}).items():
        energy[slot], health[slot] = values
    for slot, values in dict(contradictions or {}).items():
        alive[slot], slot_to_id[slot] = values

    pending_ids = torch.full((pending_capacity,), -7101, dtype=torch.int64)
    pending_slots = torch.full((pending_capacity,), -7102, dtype=torch.int64)
    if 0 <= pending_before <= pending_capacity:
        for index in range(pending_before):
            pending_ids[index] = 9000 + index
            pending_slots[index] = 9100 + index
    pending_count = torch.tensor([pending_before], dtype=torch.int64)
    death_ids = torch.full((slot_count,), -7201, dtype=torch.int64)
    death_slots = torch.full((slot_count,), -7202, dtype=torch.int64)
    death_reasons = torch.full((slot_count,), -7203, dtype=torch.int32)
    death_count = torch.tensor([-7204], dtype=torch.int64)
    survivor_ids = torch.full((active_capacity,), -7301, dtype=torch.int64)
    survivor_count = torch.tensor([-7302], dtype=torch.int64)
    return (
        [
            active_ids,
            active_slots,
            active_count,
            slot_to_id,
            id_to_slot,
            alive,
            pos_y,
            pos_x,
            grid,
            energy,
            health,
            policy_observation,
            rest_streak,
            cache_token,
            cache_cursor,
            cache_next,
            pending_ids,
            pending_slots,
            pending_count,
            death_ids,
            death_slots,
            death_reasons,
            death_count,
            survivor_ids,
            survivor_count,
        ],
        {
            "slot_count": slot_count,
            "active_capacity": active_capacity,
            "agent_id_capacity": agent_id_capacity,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "pending_capacity": pending_capacity,
        },
    )


def _cases() -> list[dict[str, Any]]:
    ordinary = ((11, 5), (20, 1), (31, 7), (44, 2))
    return [
        {"name": "empty", "state": _cpu_state(live=())},
        {"name": "no_death", "state": _cpu_state(live=ordinary)},
        {
            "name": "one_energy_death",
            "state": _cpu_state(live=ordinary, terminal={1: (0.0, 8.25)}),
        },
        {
            "name": "multiple_deaths_slot_order_differs",
            "state": _cpu_state(
                live=ordinary,
                terminal={5: (-1.0, 12.25), 2: (15.5, 0.0)},
            ),
        },
        {
            "name": "health_reason_priority",
            "state": _cpu_state(live=ordinary, terminal={7: (0.0, 0.0)}),
        },
        {
            "name": "all_death",
            "state": _cpu_state(
                live=ordinary,
                terminal={slot: (0.0, 0.0) for _, slot in ordinary},
            ),
        },
        {
            "name": "pending_success_boundary",
            "state": _cpu_state(
                live=ordinary,
                terminal={5: (0.0, 1.0)},
                pending_before=7,
            ),
        },
        {
            "name": "pending_overflow_atomic",
            "state": _cpu_state(
                live=ordinary,
                terminal={5: (0.0, 1.0), 1: (0.0, 1.0)},
                pending_before=7,
            ),
            "expected_status": 2,
            "expect_no_mutation": True,
        },
        {
            "name": "invalid_pending_atomic",
            "state": _cpu_state(live=ordinary, pending_before=-1),
            "expected_status": 1,
            "expect_no_mutation": True,
        },
        {
            "name": "out_of_order_active_fallback",
            "state": _cpu_state(
                live=ordinary,
                active_rows=(ordinary[1], ordinary[0], ordinary[2], ordinary[3]),
            ),
        },
        {
            "name": "duplicate_active_id_fallback",
            "state": _cpu_state(
                live=((11, 5), (11, 1), (31, 7)),
                active_rows=((11, 5), (11, 1), (31, 7)),
            ),
        },
        {
            "name": "wrong_row_identity_fallback",
            "state": _cpu_state(
                live=ordinary,
                active_rows=((11, 5), (21, 1), (31, 7), (44, 2)),
            ),
        },
        {
            "name": "out_of_range_row_slot_fallback",
            "state": _cpu_state(
                live=ordinary,
                active_rows=((11, 5), (20, 9), (31, 7), (44, 2)),
            ),
        },
        {
            "name": "omitted_live_row_fallback",
            "state": _cpu_state(live=ordinary, active_rows=ordinary[:-1]),
        },
        {
            "name": "alive_unassigned_fallback",
            "state": _cpu_state(
                live=ordinary,
                contradictions={0: (True, -1)},
            ),
        },
        {
            "name": "dead_assigned_fallback",
            "state": _cpu_state(
                live=ordinary,
                contradictions={0: (False, 55)},
            ),
        },
        {
            "name": "active_capacity_less_than_slots",
            "state": _cpu_state(
                live=ordinary,
                active_capacity=6,
                slot_count=8,
            ),
        },
    ]


def _tensor_digest(value: Any) -> str:
    import torch

    cpu = value.detach().cpu().contiguous()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _run_case(module: Any, cpu_state: list[Any], dimensions: Mapping[str, int]) -> dict[str, Any]:
    import torch

    state = [value.clone().cuda() for value in cpu_state]
    status = module._test_only_lifecycle_death_cull_cell_v3(
        state,
        dimensions["slot_count"],
        dimensions["active_capacity"],
        dimensions["agent_id_capacity"],
        dimensions["grid_height"],
        dimensions["grid_width"],
        dimensions["pending_capacity"],
    )
    torch.cuda.synchronize()
    return {
        "status": int(status.cpu().item()),
        "tensor_sha256s": [_tensor_digest(value) for value in state],
    }


def run(
    *,
    baseline_artifact: Path,
    baseline_manifest: Path,
    candidate_artifact: Path,
    candidate_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    import torch

    if output.exists():
        raise RuntimeError(f"refusing to overwrite parity output: {output}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("native lifecycle parity requires the admitted CUDA device")
    baseline_payload = _manifest(
        baseline_manifest, baseline_artifact, candidate=False
    )
    candidate_payload = _manifest(
        candidate_manifest, candidate_artifact, candidate=True
    )
    baseline_commit = _matched_source_commit(
        baseline_payload,
        candidate_payload,
    )
    baseline = _load_module(baseline_artifact, BASELINE_MODULE)
    candidate = _load_module(candidate_artifact, CANDIDATE_MODULE)

    rows: list[dict[str, Any]] = []
    for case in _cases():
        cpu_state, dimensions = case["state"]
        initial_sha256s = [_tensor_digest(value) for value in cpu_state]
        baseline_result = _run_case(baseline, cpu_state, dimensions)
        candidate_result = _run_case(candidate, cpu_state, dimensions)
        exact = baseline_result == candidate_result
        expected_status = int(case.get("expected_status", 0))
        no_mutation = (
            baseline_result["tensor_sha256s"] == initial_sha256s
            and candidate_result["tensor_sha256s"] == initial_sha256s
        )
        passed = (
            exact
            and baseline_result["status"] == expected_status
            and (
                not bool(case.get("expect_no_mutation"))
                or no_mutation
            )
        )
        rows.append(
            {
                "case": case["name"],
                "passed": passed,
                "exact": exact,
                "status": baseline_result["status"],
                "expected_status": expected_status,
                "no_mutation": no_mutation,
                "state_sha256": canonical_sha256(baseline_result),
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(row["passed"] for row in rows) else "FAIL",
        "source_commit": baseline_commit,
        "gpu": torch.cuda.get_device_name(0),
        "baseline": {
            "artifact": str(baseline_artifact.resolve()),
            "artifact_sha256": _sha256(baseline_artifact),
            "manifest": str(baseline_manifest.resolve()),
            "manifest_sha256": _sha256(baseline_manifest),
        },
        "candidate": {
            "artifact": str(candidate_artifact.resolve()),
            "artifact_sha256": _sha256(candidate_artifact),
            "manifest": str(candidate_manifest.resolve()),
            "manifest_sha256": _sha256(candidate_manifest),
        },
        "cases": rows,
        "case_count": len(rows),
        "passed_case_count": sum(row["passed"] for row in rows),
        "reporting_enabled": False,
        "qualification_claim": False,
        "performance_claim": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, payload)
    if payload["status"] != "PASS":
        raise RuntimeError("native lifecycle macro-off/on parity failed")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(
        baseline_artifact=args.baseline_artifact.resolve(),
        baseline_manifest=args.baseline_manifest.resolve(),
        candidate_artifact=args.candidate_artifact.resolve(),
        candidate_manifest=args.candidate_manifest.resolve(),
        output=args.output.resolve(),
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
