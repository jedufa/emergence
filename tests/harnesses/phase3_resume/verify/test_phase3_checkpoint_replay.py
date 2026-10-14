"""tests/test_phase3_checkpoint_replay.py

Phase 3 exit criteria tests (TSD v3.1) without requiring numba.

The baseline simulation environment hard-requires numba for stepping, which is
not available in the CI/runtime for these unit tests. So these tests validate
Phase 3 artifacts using a deterministic in-memory DummySim that exercises:
- Checkpoint Bundle v2 writer/loader/validator (strict mode)
- Canonical JSON byte stability
- Replay engine hash-compare workflow (exact)
- Resume-from-checkpoint workflow invariants (sequence number, hashes)

These are governance-grade checks on the serialization + validation + replay
machinery, independent of the physics/stepping backend.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from simulation.checkpoint_bundle_v2 import (
    checkpoint_bundle_v2_save,
    checkpoint_bundle_v2_load,
    checkpoint_bundle_v2_apply,
    CheckpointStrictError,
    CheckpointSchemaError,
)
from simulation.replay_engine import replay_from_checkpoint_v2, ReplayMismatch
from simulation.control_plane_pkg.util_json import canonical_json_bytes
from simulation.safeboundary_hash_log import compute_safeboundary_hashes


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return hashlib.sha256(b"").hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for d in lines:
            f.write(canonical_json_bytes(d))
            f.write(b"\n")


@dataclass
class DummySimCfg:
    seed: int = 123
    branch_id: str = "default"
    campaign_id: str = "campaign"
    plan_version: str = "v1"
    generations: int = 0
    checkpoint_bundle_v2_subdir: str = "checkpoint_bundles_v2"
    safeboundary_hash_log_enable: bool = True
    checkpoint_bundle_v2_strict_replay_mode: bool = True


class DummyRNGManager:
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
    def __init__(self, seq: int = 0) -> None:
        self._seq = int(seq)

    def current_submission_sequence_number(self) -> int:
        return int(self._seq)

    def get_submission_sequence_number(self) -> int:
        return int(self._seq)

    def set_submission_sequence_number(self, value: int) -> None:
        self._seq = int(value)


class DummySlots:
    def __init__(self) -> None:
        self.max_slots = 8
        self.free_slots = [1, 2, 3]
        self.slot_tokens = [0] * self.max_slots


class DummyPop:
    def __init__(self) -> None:
        self.next_id = 1
        self.agents: Dict[int, Any] = {}
        self._slots = DummySlots()

    def alive_agents(self):
        return []


class DummyEnv:
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
    def __init__(self, run_dir: Path, *, seed: int = 123) -> None:
        self.run_dir = str(run_dir)
        self.sim_cfg = DummySimCfg(seed=int(seed))
        self.env = DummyEnv()
        self.pop = DummyPop()
        self.control_plane = DummyControlPlane(seq=0)
        self.rng_manager = DummyRNGManager(seed=int(seed))
        self._control_plane_submission_sequence_number = 0
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

    # Replay path: call installed _replay_hook at boundaries
    def run(self, start_gen: int = 0, skip_init: bool = False) -> None:
        # No stepping. We just simulate SafeBoundary transitions.
        gens = int(getattr(self.sim_cfg, "generations", 0) or 0)
        for g in range(int(start_gen), int(start_gen) + gens):
            # update rng snapshot deterministically (pre-command snapshot for next gen)
            self._last_rng_snapshot_v2 = self.rng_manager.capture_snapshot(generation_index=int(g) + 1)

            # call replay hook at SafeBoundary next_gen
            rh = getattr(self, "_replay_hook", None)
            if callable(rh):
                rh(int(g) + 1)


def test_checkpoint_bundle_roundtrip(tmp_path: Path) -> None:
    sim = DummySim(tmp_path / "run", seed=123)
    # set a deterministic sequence number
    sim.control_plane.set_submission_sequence_number(7)

    bdir = tmp_path / "bundle_roundtrip"
    checkpoint_bundle_v2_save(sim, str(bdir), generation_index=1, strict_replay_mode=True)

    payload = checkpoint_bundle_v2_load(str(bdir), strict=True, expected_commit_hash=None)

    sim2 = DummySim(tmp_path / "run2", seed=123)
    checkpoint_bundle_v2_apply(sim2, payload, restore_rng=True)

    h1 = compute_safeboundary_hashes(sim)
    h2 = compute_safeboundary_hashes(sim2)
    for k in ["world_hash", "population_hash", "rng_snapshot_hash", "event_log_hash", "wal_hash"]:
        assert h1[k] == h2[k]
    assert sim2.control_plane.get_submission_sequence_number() == 7


def test_checkpoint_loader_rejects_pickle_or_unsupported_schema(tmp_path: Path) -> None:
    sim = DummySim(tmp_path / "run", seed=1)
    bdir = tmp_path / "bundle_bad"
    checkpoint_bundle_v2_save(sim, str(bdir), generation_index=1, strict_replay_mode=True)

    (bdir / "weights" / "bad.pickle").write_bytes(b"nope")
    with pytest.raises(CheckpointStrictError):
        checkpoint_bundle_v2_load(str(bdir), strict=True, expected_commit_hash=None)

    meta_p = bdir / "metadata.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["schema_version"] = "checkpoint_bundle_v999"
    meta_p.write_bytes(canonical_json_bytes(meta))
    with pytest.raises(CheckpointSchemaError):
        checkpoint_bundle_v2_load(str(bdir), strict=True, expected_commit_hash=None)


def test_checkpoint_strict_mode_commit_mismatch_reject(tmp_path: Path) -> None:
    sim = DummySim(tmp_path / "run", seed=2)
    bdir = tmp_path / "bundle_commit"
    checkpoint_bundle_v2_save(sim, str(bdir), generation_index=1, strict_replay_mode=True)

    meta_p = bdir / "metadata.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["code_commit_hash"] = "deadbeef"
    meta_p.write_bytes(canonical_json_bytes(meta))

    with pytest.raises(CheckpointStrictError):
        checkpoint_bundle_v2_load(str(bdir), strict=True, expected_commit_hash="cafebabe")


def test_submission_sequence_number_persisted_in_checkpoint_v2(tmp_path: Path) -> None:
    sim = DummySim(tmp_path / "run", seed=3)
    sim.control_plane.set_submission_sequence_number(42)

    bdir = tmp_path / "bundle_seq"
    checkpoint_bundle_v2_save(sim, str(bdir), generation_index=1, strict_replay_mode=True)

    payload = checkpoint_bundle_v2_load(str(bdir), strict=True, expected_commit_hash=None)
    assert int(payload["metadata"]["submission_sequence_number"]) == 42


def test_event_log_hash_in_checkpoint_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    sim = DummySim(run_dir, seed=4)

    # create deterministic events log
    _write_jsonl(run_dir / "events" / "command_events.jsonl", [{"schema_version": "command_event_v1", "i": 1}])

    bdir = tmp_path / "bundle_events"
    checkpoint_bundle_v2_save(sim, str(bdir), generation_index=1, strict_replay_mode=True)

    payload = checkpoint_bundle_v2_load(str(bdir), strict=True, expected_commit_hash=None)
    meta = payload["metadata"]
    assert str(meta.get("event_log_hash", ""))

    events_hash_txt = (bdir / "events_hash.txt").read_text(encoding="utf-8").strip()
    assert events_hash_txt == str(meta["event_log_hash"])


def test_canonical_json_stable_bytes_for_metadata(tmp_path: Path) -> None:
    sim = DummySim(tmp_path / "run", seed=5)
    bdir = tmp_path / "bundle_canon"
    checkpoint_bundle_v2_save(sim, str(bdir), generation_index=1, strict_replay_mode=True)

    raw = (bdir / "metadata.json").read_bytes()
    meta_obj = json.loads(raw.decode("utf-8"))
    assert raw == canonical_json_bytes(meta_obj)


def test_replay_engine_exact_hash_match(tmp_path: Path) -> None:
    run_dir = tmp_path / "runA"
    sim = DummySim(run_dir, seed=11)

    # Create determinism artifacts (events + wal) and reference hash log for gens 1..6
    _write_jsonl(run_dir / "events" / "command_events.jsonl", [{"schema_version": "command_event_v1", "i": 1}])
    _write_jsonl(run_dir / "wal" / "wal_000001.jsonl", [{"schema_version": "wal_record_v1", "i": 1}])

    # Build a bundle at generation 3
    bdir = run_dir / sim.sim_cfg.checkpoint_bundle_v2_subdir / "bundle_gen000003"
    checkpoint_bundle_v2_save(sim, str(bdir), generation_index=3, strict_replay_mode=True)

    # Build reference safeboundary hash log: identical state each gen (DummySim doesn't step).
    from simulation.safeboundary_hash_log import append_safeboundary_hash_log
    for gi in [1, 2, 3, 4, 5, 6]:
        sim._last_rng_snapshot_v2 = sim.rng_manager.capture_snapshot(generation_index=gi)
        append_safeboundary_hash_log(str(run_dir), gi, compute_safeboundary_hashes(sim))

    # Replay using a fresh sim instance
    sim2 = DummySim(run_dir, seed=11)
    sim2.sim_cfg.generations = 0
    replay_from_checkpoint_v2(sim2, bundle_dir=str(bdir), run_dir=str(run_dir), end_generation=6, strict=True)


def test_resume_equivalence_uninterrupted_vs_resume(tmp_path: Path) -> None:
    N = 6
    K = 3

    # Uninterrupted
    runA = tmp_path / "uninterrupted"
    simA = DummySim(runA, seed=99)
    simA.sim_cfg.generations = N
    _write_jsonl(runA / "events" / "command_events.jsonl", [{"schema_version": "command_event_v1", "i": 1}])
    _write_jsonl(runA / "wal" / "wal_000001.jsonl", [{"schema_version": "wal_record_v1", "i": 1}])
    simA.run(start_gen=0, skip_init=False)

    # Resume run: first K gens + checkpoint
    runB = tmp_path / "resume"
    simB1 = DummySim(runB, seed=99)
    simB1.sim_cfg.generations = K
    _write_jsonl(runB / "events" / "command_events.jsonl", [{"schema_version": "command_event_v1", "i": 1}])
    _write_jsonl(runB / "wal" / "wal_000001.jsonl", [{"schema_version": "wal_record_v1", "i": 1}])
    simB1.run(start_gen=0, skip_init=False)

    bdir = runB / simB1.sim_cfg.checkpoint_bundle_v2_subdir / f"bundle_gen{K:06d}"
    checkpoint_bundle_v2_save(simB1, str(bdir), generation_index=K, strict_replay_mode=True)

    payload = checkpoint_bundle_v2_load(str(bdir), strict=True, expected_commit_hash=None)

    # "Restart"
    simB2 = DummySim(runB, seed=99)
    checkpoint_bundle_v2_apply(simB2, payload, restore_rng=True)
    simB2.sim_cfg.generations = N - K
    simB2.run(start_gen=K, skip_init=True)

    assert _sha256_file(runA / "events" / "command_events.jsonl") == _sha256_file(runB / "events" / "command_events.jsonl")
    assert _sha256_file(runA / "wal" / "wal_000001.jsonl") == _sha256_file(runB / "wal" / "wal_000001.jsonl")

    hA = compute_safeboundary_hashes(simA)
    hB = compute_safeboundary_hashes(simB2)
    for k in ["world_hash", "population_hash", "rng_snapshot_hash", "event_log_hash", "wal_hash", "boundary_metrics_hash"]:
        assert hA[k] == hB[k]
