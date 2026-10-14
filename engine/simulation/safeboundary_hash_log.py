from __future__ import annotations

"""
simulation/safeboundary_hash_log.py

Phase 3: SafeBoundary hash logger + helpers.

Why this exists
---------------
Phase 3 replay requires exact hash comparisons at every generation boundary,
but the baseline Phase 1/2 runner did not persist per-generation state hashes.

This module provides:
- compute_safeboundary_hashes(sim): determinism-critical hashes
- append_safeboundary_hash_log(run_dir, generation_index, hashes)
- read_safeboundary_hash_log(path): load reference map for replay

Determinism-critical artifacts (Phase 3)
----------------------------------------
- events/command_events.jsonl
- wal/*.jsonl
- rng_snapshot_v2.json (in checkpoint bundle)
- checkpoint_bundle/metadata.json
- checkpoint_bundle/control_plane.json
- hashes/safeboundary_hashes.jsonl   (reference for replay)

All JSONL lines are canonical (sorted keys, minified) to keep bytes stable.
"""

import os
import json
import hashlib
from typing import Any, Dict

from simulation.control_plane_pkg.util_json import canonical_json_bytes

from simulation.checkpoint_bundle_v2 import _export_world, _export_population_and_weights  # noqa: F401


def _sha256_wal_dir_best_effort(run_dir: str) -> str:
    """Stable hash of wal/*.jsonl bytes (lex order by filename).

    This is determinism-critical because WAL divergence can alter ControlPlane behavior.
    """
    wdir = os.path.join(str(run_dir), "wal")
    if not os.path.isdir(wdir):
        return hashlib.sha256(b"").hexdigest()
    files = [fn for fn in os.listdir(wdir) if fn.endswith('.jsonl')]
    files.sort()
    h = hashlib.sha256()
    for fn in files:
        h.update(fn.encode('utf-8'))
        h.update(b"\n")
        fp = os.path.join(wdir, fn)
        with open(fp, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\n")
    return h.hexdigest()


def compute_safeboundary_hashes(sim: Any) -> Dict[str, str]:
    """Compute determinism-critical hashes at the SafeBoundary for the current sim state."""
    # World + population hashes use the same canonical JSON export as checkpoint bundle.
    world_obj = _export_world(sim)
    pop_obj, _weights_info = _export_population_and_weights(sim, weights_dir=os.devnull)  # weights not needed for hash
    rng_obj = getattr(sim, "_last_rng_snapshot_v2", None)
    if rng_obj is None:
        rng_obj = {"schema_version": "rng_snapshot_v2", "note": "missing_last_snapshot"}

    world_hash = hashlib.sha256(canonical_json_bytes(world_obj)).hexdigest()
    population_hash = hashlib.sha256(canonical_json_bytes(pop_obj)).hexdigest()
    rng_snapshot_hash = hashlib.sha256(canonical_json_bytes(rng_obj)).hexdigest()
    event_log_hash = _sha256_file_best_effort(os.path.join(str(getattr(sim, "run_dir", "")), "events", "command_events.jsonl"))
    wal_hash = _sha256_wal_dir_best_effort(str(getattr(sim, "run_dir", "")))
    # Deterministic boundary metrics (post-command state)
    bm = {
        "submission_sequence_number": int(getattr(getattr(sim, 'control_plane', None), 'get_submission_sequence_number', lambda: 0)()),
        "alive_count": int(len(list(getattr(sim, "pop").alive_agents())) if getattr(sim, "pop", None) is not None else 0),
    }
    boundary_metrics_hash = hashlib.sha256(canonical_json_bytes(bm)).hexdigest()

    # Optional: legacy metrics object hash if runner provides it
    metrics_hash = ""
    try:
        gm = getattr(sim, "_last_generation_metrics", None)
        if gm is not None:
            metrics_hash = hashlib.sha256(canonical_json_bytes(gm)).hexdigest()
    except Exception:
        metrics_hash = ""

    out = {
        "world_hash": world_hash,
        "population_hash": population_hash,
        "rng_snapshot_hash": rng_snapshot_hash,
        "event_log_hash": event_log_hash,
        "wal_hash": wal_hash,
        "boundary_metrics_hash": boundary_metrics_hash,
    }
    if metrics_hash:
        out["metrics_hash"] = metrics_hash
    return out


def append_safeboundary_hash_log(run_dir: str, generation_index: int, hashes: Dict[str, str]) -> None:
    """Append one canonical JSONL record to hashes/safeboundary_hashes.jsonl (best-effort flush+fsync)."""
    hdir = os.path.join(str(run_dir), "hashes")
    os.makedirs(hdir, exist_ok=True)
    p = os.path.join(hdir, "safeboundary_hashes.jsonl")
    rec = {
        "schema_version": "safeboundary_hash_v1",
        "generation_index": int(generation_index),
        **{str(k): str(v) for k, v in (hashes or {}).items()},
    }
    line = canonical_json_bytes(rec).decode("utf-8")
    with open(p, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def read_safeboundary_hash_log(path: str) -> Dict[int, Dict[str, Any]]:
    """Read safeboundary_hashes.jsonl into a map: generation_index -> record dict."""
    if not os.path.exists(path):
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            d = json.loads(raw)
            gi = int(d.get("generation_index", -1))
            if gi >= 0:
                out[gi] = d
    return out


def _sha256_file_best_effort(path: str) -> str:
    if not os.path.exists(path):
        return hashlib.sha256(b"").hexdigest()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
