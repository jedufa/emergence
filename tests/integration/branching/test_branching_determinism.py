"""tests/test_branching_determinism.py

Phase 5c validation harness.

This repo's full sim stepping stack may not be available in CI (e.g., numba).
These tests therefore validate branching + resume determinism using a tiny
in-memory DummySim that exercises:
- checkpoint bundle v2 save/load/apply
- branch tool orchestration and metadata writing
- deterministic artifact equivalence between two branched runs

The artifacts compared are deterministic by construction:
- metrics.csv (derived from SafeBoundary hash computations)
- behaviors.csv (minimal deterministic placeholder)
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from simulation.checkpoint_bundle_v2 import (
    checkpoint_bundle_v2_apply,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_save,
)
from simulation.control_plane_pkg.util_json import canonical_json_bytes
from simulation.safeboundary_hash_log import compute_safeboundary_hashes

from tests.harnesses.phase3_resume.verify.phase3_assert_csv_equivalence import (
    main as _phase3_csv_eq_main,
)
from tools.branch_from_checkpoint import branch_from_checkpoint


def _sha256_file(path: Path) -> str:
    """Return sha256 hex digest for a file (streaming)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_dir_state(root: Path) -> List[Dict[str, Any]]:
    """Return a deterministic snapshot of a directory's files.

    Snapshot elements include:
    - relative path
    - size
    - sha256
    """
    out: List[Dict[str, Any]] = []
    for p in sorted([pp for pp in root.rglob("*") if pp.is_file()]):
        rel = str(p.relative_to(root)).replace("\\", "/")
        out.append({"path": rel, "size": int(p.stat().st_size), "sha256": _sha256_file(p)})
    return out


def _write_jsonl(path: Path, lines: List[Dict[str, Any]]) -> None:
    """Write canonical JSONL lines to a path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for d in lines:
            f.write(canonical_json_bytes(d))
            f.write(b"\n")


@dataclass
class DummySimCfg:
    """Minimal config used by checkpoint bundle v2 serialization."""

    seed: int = 123
    branch_id: str = "default"
    campaign_id: str = "campaign"
    plan_version: str = "v1"
    generations: int = 0
    checkpoint_bundle_v2_subdir: str = "checkpoint_bundles_v2"
    safeboundary_hash_log_enable: bool = True
    checkpoint_bundle_v2_strict_replay_mode: bool = True


class DummyRNGManager:
    """Minimal RNGManager stub used by checkpoint bundle v2."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self.stream_indices = {"policy": 0, "world": 0, "mutation": 0, "shock": 0, "tick": 0}

    def capture_snapshot(self, generation_index: int) -> Dict[str, Any]:
        return {
            "schema_version": "rng_snapshot_v2",
            "generation_index": int(generation_index),
            "global_seed": int(self.seed),
            "stream_indices": dict(self.stream_indices),
        }


class DummyControlPlane:
    """Minimal control plane stub for submission_sequence_number persistence."""

    def __init__(self, seq: int = 0) -> None:
        self._seq = int(seq)

    def current_submission_sequence_number(self) -> int:
        return int(self._seq)

    def get_submission_sequence_number(self) -> int:
        return int(self._seq)

    def set_submission_sequence_number(self, value: int) -> None:
        self._seq = int(value)


class DummySlots:
    """Tiny slots container to satisfy bundle export hooks."""

    def __init__(self) -> None:
        self.max_slots = 8
        self.free_slots = [1, 2, 3]
        self.slot_tokens = [0] * self.max_slots


class DummyPop:
    """Minimal population stub."""

    def __init__(self) -> None:
        self.next_id = 1
        self.agents: Dict[int, Any] = {}
        self._slots = DummySlots()

    def alive_agents(self):
        return []


class DummyEnv:
    """Minimal env stub."""

    def __init__(self) -> None:
        self.grid = np.zeros((4, 4), dtype=np.int32)
        self.agent_grid = np.zeros((4, 4), dtype=np.int32)
        self.biome_grid = np.zeros((4, 4), dtype=np.int32)
        self.agent_pos_y = np.zeros((1,), dtype=np.int32)
        self.agent_pos_x = np.zeros((1,), dtype=np.int32)
        self.agent_alive = np.zeros((1,), dtype=np.int32)
        self.food_spawn_mult = 1.0
        self.action_cost_mult = 1.0
        self.rest_cost_mult = 1.0


class DummySim:
    """Dummy simulation runner for serialization + deterministic stepping."""

    def __init__(
        self,
        run_dir: Path,
        *,
        seed: int = 123,
        branch_id: str = "default",
        campaign_id: str = "campaign",
        submission_seq: int = 0,
    ) -> None:
        self.run_dir = str(run_dir)
        self.sim_cfg = DummySimCfg(seed=int(seed), branch_id=str(branch_id), campaign_id=str(campaign_id))
        self.env = DummyEnv()
        self.pop = DummyPop()
        self.control_plane = DummyControlPlane(seq=int(submission_seq))
        self.rng_manager = DummyRNGManager(seed=int(seed))
        self._control_plane_submission_sequence_number = int(submission_seq)
        self._last_rng_snapshot_v2 = self.rng_manager.capture_snapshot(generation_index=0)
        self.git_commit = "unknown"

    # minimal hooks used by checkpoint apply
    def _ensure_bank(self) -> None:
        return

    def _mark_bank_dirty(self) -> None:
        return

    def _sync_bank_for_agents(self, *args, **kwargs) -> None:
        return

    def _sync_gru_bank_for_agents(self, *args, **kwargs) -> None:
        return

    def run(self, start_gen: int = 0, skip_init: bool = False) -> None:
        """Run N generations, calling an installed _replay_hook at each boundary."""
        gens = int(getattr(self.sim_cfg, "generations", 0) or 0)
        for g in range(int(start_gen), int(start_gen) + gens):
            self._last_rng_snapshot_v2 = self.rng_manager.capture_snapshot(generation_index=int(g) + 1)
            rh = getattr(self, "_replay_hook", None)
            if callable(rh):
                rh(int(g) + 1)


def _run_child_and_write_csv(run_dir: Path, bundle_dir: Path, *, gens: int) -> None:
    """Resume from a bundle, run gens generations, and write deterministic CSV artifacts."""
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = checkpoint_bundle_v2_load(str(bundle_dir), strict=True, expected_commit_hash=None)

    md = payload.get("metadata", {}) or {}
    start_gen = int(md.get("generation_index", 0) or 0)
    seed = int(md.get("generation_index", 0) or 0) + 123  # stable but irrelevant; state is restored

    # Branch/campaign ids are deterministic identifiers; safe defaults for this harness.
    branch_id = str(md.get("branch_id", "default") or "default")
    campaign_id = str(md.get("campaign_id", "campaign") or "campaign")
    seq = int(md.get("submission_sequence_number", 0) or 0)

    sim = DummySim(run_dir, seed=seed, branch_id=branch_id, campaign_id=campaign_id, submission_seq=seq)
    checkpoint_bundle_v2_apply(sim, payload, restore_rng=True)

    sim.sim_cfg.generations = int(gens)

    metrics_p = run_dir / "metrics.csv"
    beh_p = run_dir / "behaviors.csv"
    with metrics_p.open("w", newline="", encoding="utf-8") as mf, beh_p.open("w", newline="", encoding="utf-8") as bf:
        mw = csv.writer(mf)
        bw = csv.writer(bf)
        mw.writerow(["generation", "world_hash", "population_hash", "rng_snapshot_hash", "event_log_hash", "wal_hash"])
        bw.writerow(["generation", "note"])

        def _hook(next_gen: int) -> None:
            h = compute_safeboundary_hashes(sim)
            mw.writerow(
                [
                    int(next_gen),
                    str(h.get("world_hash", "")),
                    str(h.get("population_hash", "")),
                    str(h.get("rng_snapshot_hash", "")),
                    str(h.get("event_log_hash", "")),
                    str(h.get("wal_hash", "")),
                ]
            )
            bw.writerow([int(next_gen), "ok"])

        setattr(sim, "_replay_hook", _hook)
        sim.run(start_gen=int(start_gen), skip_init=True)


def test_phase5c_branching_deterministic_and_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Force strict determinism surfacing (metadata only for this test).
    monkeypatch.setenv("DEBUG_DETERMINISM", "1")
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("GIT_COMMIT", "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")

    parent_run = tmp_path / "parent_run"
    parent_run.mkdir(parents=True, exist_ok=True)

    parent_branch_id = "11111111-1111-1111-1111-111111111111"
    campaign_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    submission_seq = 7

    # Seed parent run with deterministic control-plane files (append-only logs).
    _write_jsonl(
        parent_run / "wal" / "command_wal.jsonl",
        [
            {
                "schema_version": "command_wal_v1",
                "command_uuid": "cmd-1",
                "submission_sequence_number": 1,
                "target_generation": 1,
                "command_payload_hash": "sha256:00",
                "command_payload": {"t": 1},
                "transition": "SUBMITTED->ACKED",
                "generation_index": 1,
            },
            {
                "schema_version": "command_wal_v1",
                "command_uuid": "cmd-2",
                "submission_sequence_number": 2,
                "target_generation": 2,
                "command_payload_hash": "sha256:11",
                "command_payload": {"t": 2},
                "transition": "SUBMITTED->ACKED",
                "generation_index": 2,
            },
        ],
    )
    _write_jsonl(
        parent_run / "events" / "command_events.jsonl",
        [
            {
                "schema_version": "command_event_v1",
                "event_id": "evt-1",
                "generation_applied": 1,
                "target_generation": 1,
                "submission_sequence_number": 1,
                "command_uuid": "cmd-1",
                "command_payload_hash": "sha256:00",
                "command_payload": {"t": 1},
            }
        ],
    )

    # Plans: minimal version chain.
    plan_id = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    plans_dir = parent_run / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    plan_v1 = {
        "schema_version": "intervention_plan_v2",
        "plan_id": plan_id,
        "campaign_id": campaign_id,
        "version": 1,
        "parent_plan_version": None,
        "created_at_generation": 0,
        "expectation_note": "initial plan",
        "commands": [],
        "frozen": False,
    }
    (plans_dir / f"plan_{plan_id}_v000001.json").write_text(json.dumps(plan_v1, indent=2, sort_keys=True), encoding="utf-8")
    (plans_dir / "current_plan.json").write_text(json.dumps({"plan_id": plan_id, "version": 1}, indent=2, sort_keys=True), encoding="utf-8")

    # Baseline config file for baseline_config_hash.
    (parent_run / "config.json").write_text(json.dumps({"sim_cfg": {"seed": 123}}, indent=2, sort_keys=True), encoding="utf-8")

    # Create checkpoint bundle at generation K.
    K = 5
    bundle_dir = parent_run / "checkpoint_bundles_v2" / f"bundle_gen{K:06d}"
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)

    sim = DummySim(parent_run, seed=123, branch_id=parent_branch_id, campaign_id=campaign_id, submission_seq=submission_seq)
    checkpoint_bundle_v2_save(sim, str(bundle_dir), generation_index=int(K), strict_replay_mode=True)

    parent_before = _snapshot_dir_state(parent_run)

    # Branch twice with the same explicit branch_id (so the child id is not a source of variance).
    child_branch_id = "22222222-2222-2222-2222-222222222222"
    child_a = tmp_path / "child_a"
    child_b = tmp_path / "child_b"

    _ = branch_from_checkpoint(
        parent_run_dir=str(parent_run),
        checkpoint_path=str(bundle_dir),
        child_out_dir=str(child_a),
        branch_id=str(child_branch_id),
        notes="unit-test",
    )
    _ = branch_from_checkpoint(
        parent_run_dir=str(parent_run),
        checkpoint_path=str(bundle_dir),
        child_out_dir=str(child_b),
        branch_id=str(child_branch_id),
        notes="unit-test",
    )

    # Parent must remain unchanged.
    parent_after = _snapshot_dir_state(parent_run)
    assert parent_after == parent_before

    # Metadata correctness (child_a).
    bm = json.loads((child_a / "branch_meta.json").read_text(encoding="utf-8"))
    assert bm["schema_version"] == "branch_meta_v1"
    assert bm["campaign_id"] == campaign_id
    assert bm["branch_id"] == child_branch_id
    assert bm["parent_branch_id"] == parent_branch_id
    assert bm["root_branch_id"] == parent_branch_id
    assert bm["branch_depth"] == 1
    assert bm["created_at_generation"] == K
    assert str(bm["baseline_config_hash"]).startswith("sha256:")
    assert int(bm["submission_sequence_start"]) == submission_seq
    assert int(bm["submission_sequence_next"]) == submission_seq + 1

    # Determinism surfacing in run_manifest.json.
    rm = json.loads((child_a / "run_manifest.json").read_text(encoding="utf-8"))
    assert rm["schema_version"] == "run_manifest_v1"
    assert rm["determinism"]["strict"] is True
    assert rm["determinism"]["source_env"] == "DEBUG_DETERMINISM"
    assert rm["determinism"]["raw"] == "1"

    # Root-level determinism status file.
    ds = json.loads((child_a / "run_determinism_status.json").read_text(encoding="utf-8"))
    assert ds["schema_version"] == "run_det_status_v1"
    assert ds["strict_mode"] is True

    # Strict isolation: child must have its own copies (no symlinks).
    assert not (child_a / "wal" / "command_wal.jsonl").is_symlink()
    assert not (child_a / "events" / "command_events.jsonl").is_symlink()
    assert (child_a / "plans" / f"plan_{plan_id}_v000001.json").exists()

    # Resume and write deterministic artifacts.
    child_a_bundle = child_a / "checkpoint_bundles_v2" / bundle_dir.name
    child_b_bundle = child_b / "checkpoint_bundles_v2" / bundle_dir.name
    assert child_a_bundle.exists()
    assert child_b_bundle.exists()

    _run_child_and_write_csv(child_a, child_a_bundle, gens=3)
    _run_child_and_write_csv(child_b, child_b_bundle, gens=3)

    # Phase 4-era equivalence check for CSV artifacts.
    rc = _phase3_csv_eq_main(["--run-a", str(child_a), "--run-b", str(child_b)])
    assert rc == 0
