"""
engine/simulation/state_dump.py

Minimal state dump utility used by Phase 3 determinism investigation tools.

Design constraints
------------------
- Debug-only and best-effort.
- Must never draw RNG.
- Must never crash the simulation if dumping fails.
- Output format is stable and intentionally small.
- File naming is compatible with tools/verify/phase3_membership_harness.py:
    state_{gen:06d}__{tag}.json

Environment variables
---------------------
EIEL_STATE_DUMP=1
    Enables state dumping.

EIEL_STATE_DUMP_GENS="0,1,2"
    Optional allow-list of generation indices.

EIEL_STATE_DUMP_DIR="<path>"
    Optional override for the dump root directory. If not set, uses runner.run_dir/state_dumps.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import hashlib
from typing import Iterable, Tuple

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore


def _sha256_text(s: str) -> str:
    """Return SHA256 hex digest of a UTF-8 string."""
    return hashlib.sha256(s.encode("utf-8", errors="strict")).hexdigest()


def _sha256_ndarray(a: Any) -> str:
    """Return a stable SHA256 over a numpy array's dtype/shape/bytes.

    Notes
    -----
    - Does NOT draw RNG.
    - Uses contiguous bytes view (tobytes) and prefixes dtype/shape to avoid collisions.
    """
    if np is None:
        return "numpy_missing"
    try:
        arr = np.asarray(a)
        header = f"{arr.dtype.str}|{tuple(int(x) for x in arr.shape)}|"
        h = hashlib.sha256()
        h.update(header.encode("utf-8"))
        h.update(arr.tobytes(order="C"))
        return h.hexdigest()
    except Exception:
        return "ndarray_hash_error"


def _compute_probe(runner: Any) -> Dict[str, Any]:
    """Compute a small, deterministic, zero-RNG probe snapshot.

    This is intentionally hash-only for large structures.
    Enable with EIEL_STATE_DUMP_PROBE=1.
    """
    probe: Dict[str, Any] = {}

    # Runner-level counters (cheap)
    try:
        probe["tick_in_gen"] = int(getattr(runner, "tick_in_gen", -1))
    except Exception:
        pass

    sim = getattr(runner, "sim", None) or getattr(runner, "simulation", None) or runner

    # Environment spatial state hashes
    env = getattr(sim, "env", None)
    if env is not None:
        for name in ("grid", "agent_grid", "biome_grid", "agent_pos_y", "agent_pos_x", "agent_alive"):
            if hasattr(env, name):
                probe[f"env.{name}.sha256"] = _sha256_ndarray(getattr(env, name))

        # A few scalar multipliers that can drift
        for name in ("food_spawn_mult", "action_cost_mult", "rest_cost_mult"):
            if hasattr(env, name):
                try:
                    probe[f"env.{name}"] = float(getattr(env, name))
                except Exception:
                    pass

        # Spatial ecology/drought maps (Tier 1) if present
        for name in ("resource_regrowth_map", "resource_spawn_density_map", "hazard_rate_map",
                     "drought_food_map", "drought_action_cost_map", "drought_rest_cost_map"):
            if hasattr(env, name):
                probe[f"env.{name}.sha256"] = _sha256_ndarray(getattr(env, name))

    # Population / alive membership signature
    pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
    if pop is not None and hasattr(pop, "agents"):
        try:
            agents = getattr(pop, "agents")
            ids = sorted(int(k) for k in agents.keys())
            probe["pop.agent_ids.sha256"] = _sha256_text(",".join(str(i) for i in ids))
        except Exception:
            pass

    return probe


def _allowed_generation(gen: int) -> bool:
    """Return True if dumping is allowed for this generation."""
    gens = os.environ.get("EIEL_STATE_DUMP_GENS", "").strip()
    if not gens:
        return True
    try:
        allowed = {int(x) for x in gens.split(",") if x.strip() != ""}
    except Exception:
        # If parsing fails, do not dump (safer for log noise).
        return False
    return (not allowed) or (int(gen) in allowed)


def _dump_root(runner: Any) -> Optional[Path]:
    """Resolve the state_dumps output directory for a runner."""
    override = os.environ.get("EIEL_STATE_DUMP_DIR", "").strip()
    if override:
        return Path(override) / "state_dumps"

    run_dir = getattr(runner, "run_dir", None) or getattr(runner, "run_dir_path", None) or ""
    if not run_dir:
        return None
    return Path(str(run_dir)) / "state_dumps"


def maybe_dump_state(runner: Any, *, generation_index: int, tag: str, extra: Optional[Dict[str, Any]] = None) -> None:
    """Write a deterministic JSON state dump if enabled.

    Parameters
    ----------
    runner:
        Simulation runner instance.
    generation_index:
        Absolute generation index to label the dump.
    tag:
        Short lifecycle tag, used in the filename.
    extra:
        Additional JSON-serializable payload. If it contains 'alive_ids', it is
        preserved at the top level to support membership harness tooling.

    Notes
    -----
    - Best-effort: exceptions are swallowed by callers; we still keep this safe.
    - No timestamps or nondeterministic fields are written.
    """
    if os.environ.get("EIEL_STATE_DUMP", "0").strip() != "1":
        return
    gen = int(generation_index)
    if not _allowed_generation(gen):
        return

    root = _dump_root(runner)
    if root is None:
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    # Sanitize tag for filenames.
    safe_tag = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(tag))
    path = root / f"state_{gen:06d}__{safe_tag}.json"

    # Always include a zero-RNG probe for key boundary tags (resume/fresh equivalence).
    # This avoids needing extra env flags for forensic runs.
    boundary_tags = {
        "bundle_save_post",
        "resume_apply_post",
        "safeboundary_post",
    }
    if extra is None:
        extra = {}
    if str(tag) in boundary_tags or os.environ.get("EIEL_STATE_DUMP_PROBE", "0").strip() == "1":
        try:
            extra.setdefault("probe", {}).update(_compute_probe(runner))
        except Exception:
            pass

    payload: Dict[str, Any] = {
        "generation_index": gen,
        "tag": str(tag),
    }
    if extra:
        # Promote 'alive_ids' to top-level if present (membership harness contract).
        if "alive_ids" in extra:
            payload["alive_ids"] = extra.get("alive_ids")
        # Keep everything else under 'extra' to avoid schema creep.
        payload["extra"] = extra

    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    except Exception:
        return
def maybe_dump_rng_drift(
    runner,
    *,
    generation_index: int,
    tick: int,
    stage: str,
    alive_ids_seq=None,
    extra=None,
) -> None:
    """Debug-only RNG drift probe.

    This is intentionally **very** lightweight when disabled:
    - If env var EIEL_RNG_DRIFT_PROBE != "1", it returns immediately.
    - When enabled, it computes SHA digests of RNG states without advancing them and
      appends a JSONL record to <run_dir>/debug/rng_drift.jsonl.

    Determinism: this function must never draw from any RNG.
    """
    try:
        if os.environ.get("EIEL_RNG_DRIFT_PROBE", "0") != "1":
            return

        run_dir = getattr(runner, "run_dir", None)
        if not run_dir:
            return
        dbg_dir = os.path.join(run_dir, "debug")
        os.makedirs(dbg_dir, exist_ok=True)
        path = os.path.join(dbg_dir, "rng_drift.jsonl")

        def _sha256_json(obj: object) -> str:
            blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(blob).hexdigest()

        def _rng_digest(rng_obj: object) -> str:
            st = getattr(getattr(rng_obj, "bit_generator", None), "state", None)
            if st is None:
                return "none"
            return _sha256_json(st)

        payload: dict = {
            "gen": int(generation_index),
            "tick": int(tick),
            "stage": str(stage),
            "runner_np": _rng_digest(getattr(runner, "np_rng", None)),
            "env_rng": _rng_digest(getattr(getattr(runner, "env", None), "rng", None)),
        }

        if alive_ids_seq is not None:
            try:
                ids_blob = ",".join(str(int(x)) for x in alive_ids_seq).encode("utf-8")
                payload["alive_ids_sha"] = hashlib.sha256(ids_blob).hexdigest()
                payload["alive_n"] = int(len(alive_ids_seq))
            except Exception:
                payload["alive_ids_sha"] = "err"

        if isinstance(extra, dict) and extra:
            payload["extra"] = extra

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        # Probes must never break simulation runs.
        return
