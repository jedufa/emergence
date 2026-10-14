from __future__ import annotations

"""
simulation/checkpoint.py

Checkpoint v1: deterministic persistence for the Emergence evo-grid simulation.

This module provides a minimal, robust checkpoint system that can:
- Save full simulation state periodically and on-demand.
- Load checkpoints and resume deterministically (as much as the stack allows).
- Validate file integrity and schema versions.
- Support future schema changes via explicit version fields and migrations.

Design notes
------------
- We store large numeric state in NPZ files (NumPy arrays).
- We store RNG state in a pickle because RNG implementations expose Python objects.
- We optionally store parameter banks (torch tensors) via torch.save.

Atomicity
---------
Checkpoints are written to a temporary directory and then renamed into place.
On the same filesystem, directory rename is atomic, preventing partial checkpoints.

Determinism caveats
-------------------
- CPU-only runs should be bitwise deterministic if code+config are unchanged and RNG state is restored.
- GPU/CUDA runs can be nondeterministic unless deterministic algorithms are forced.
"""

from dataclasses import asdict, is_dataclass
import datetime as _dt
import json
import os
import pickle
import re
import shutil
import hashlib
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from .checkpoint_trust import require_trusted_checkpoint_artifact

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# -----------------------------
# Schema versions (Checkpoint v1)
# -----------------------------
CHECKPOINT_VERSION = 1
WORLD_SCHEMA_VERSION = 1
AGENT_SCHEMA_VERSION = 1
GENOME_SCHEMA_VERSION = 1
RNG_SCHEMA_VERSION = 1
BANK_SCHEMA_VERSION = 1


# -----------------------------
# Public API
# -----------------------------
def checkpoint_save(
    sim: Any,
    path: str,
    *,
    include_rng: bool = True,
    include_banks: bool = False,
    label: str = "",
) -> str:
    """Save a checkpoint directory atomically.

    Parameters
    ----------
    sim:
        SimulationRunner instance (or compatible object) with attributes:
        - env_cfg, pop_cfg, sim_cfg
        - env, pop
        - run_dir (optional, used for metadata)
        - np_rng (optional)
        - plus runner aux arrays that affect determinism (best-effort).
    path:
        Target checkpoint directory path. If it exists, it will be replaced.
    include_rng:
        If True, saves Python/NumPy/Torch RNG states.
    include_banks:
        If True, saves parameter bank tensors and GRU memory state (if present).
    label:
        Optional human label stored in meta.json.

    Returns
    -------
    str
        Final checkpoint directory path.
    """
    # ------------------------------------------------------------------
    # Legacy Checkpoint v1 writer is disabled in this codebase.
    #
    # The project migrated to Checkpoint Bundle v2 (portable, no-pickle).
    # Keeping this guard prevents accidental creation of pickle-based
    # checkpoints when running older scripts/flags.
    #
    # Escape hatch (not recommended):
    #   export EIEL_ALLOW_CHECKPOINT_V1=1
    # ------------------------------------------------------------------
    _allow_v1 = str(os.environ.get("EIEL_ALLOW_CHECKPOINT_V1", "0") or "0").strip().lower()
    if _allow_v1 not in ("1", "true", "yes"):
        raise RuntimeError(
            "Checkpoint v1 saving is disabled (v2-only engine). \
Use simulation.checkpoint_bundle_v2.checkpoint_bundle_v2_save instead. \
To override (not recommended), set EIEL_ALLOW_CHECKPOINT_V1=1."
        )

    final_dir = os.path.abspath(path)
    parent = os.path.dirname(final_dir)
    os.makedirs(parent, exist_ok=True)

    tmp_dir = final_dir + f".tmp_{os.getpid()}_{_rand_hex(8)}"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # Write payload files
    meta: Dict[str, Any] = _build_meta(sim=sim, include_rng=include_rng, include_banks=include_banks, label=label)
    _save_world_npz(sim, os.path.join(tmp_dir, "world.npz"))
    _save_agents_npz(sim, os.path.join(tmp_dir, "agents.npz"))
    _save_genomes_npz(sim, os.path.join(tmp_dir, "genomes.npz"))

    if include_rng:
        _save_rng(sim, os.path.join(tmp_dir, "rng.pkl"))

    if include_banks:
        _save_banks(sim, os.path.join(tmp_dir, "banks.pt"))

    # Compute manifest checksums (after files exist)
    manifest = _compute_manifest(tmp_dir)
    meta["file_manifest"] = manifest

    with open(os.path.join(tmp_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    # Atomic replace
    if os.path.exists(final_dir):
        # Move old out of the way first (safer than deleting in-place)
        old_dir = final_dir + f".old_{os.getpid()}_{_rand_hex(6)}"
        try:
            os.replace(final_dir, old_dir)
            shutil.rmtree(old_dir, ignore_errors=True)
        except Exception:
            shutil.rmtree(final_dir, ignore_errors=True)

    os.replace(tmp_dir, final_dir)
    return final_dir


def checkpoint_load(path: str, *, trusted: bool = False) -> Any:
    """Load a checkpoint from disk.

    Returns
    -------
    Any
        A dict-like payload containing:
        - meta (dict)
        - env_cfg, pop_cfg, sim_cfg (dict snapshots)
        - world (dict of arrays)
        - agents (dict of arrays)
        - genomes (dict: agent_id -> genome_dict)
        - rng (dict) if present
        - banks (dict[str, torch.Tensor]) if present

    Notes
    -----
    This function intentionally does NOT construct a SimulationRunner because
    runner construction depends on the user's run directory + logging policy.
    The recommended integration pattern is:
        runner = SimulationRunner(env_cfg, pop_cfg, sim_cfg, logger, run_dir, ...)
        checkpoint_apply(runner, payload)
    """
    ckpt_dir = os.path.abspath(path)

    # ------------------------------------------------------------
    # Checkpoint Bundle v2 (Phase 3) support
    # If the directory contains metadata.json, treat it as Bundle v2.
    # ------------------------------------------------------------
    meta_v2_path = os.path.join(ckpt_dir, "metadata.json")
    if os.path.exists(meta_v2_path):
        try:
            from .checkpoint_bundle_v2 import checkpoint_bundle_v2_load  # noqa: WPS433 (local import)
        except Exception as e:
            raise RuntimeError(f"Cannot import checkpoint_bundle_v2 loader: {e}") from e
        payload = checkpoint_bundle_v2_load(ckpt_dir, strict=True)
        # Tag for downstream dispatch (checkpoint_apply)
        payload["_format"] = "bundle_v2"
        return payload

    meta_path = os.path.join(ckpt_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Checkpoint missing meta.json: {ckpt_dir}")

    # ------------------------------------------------------------------
    # Legacy Checkpoint v1 loading is disabled by default (v2-only engine).
    #
    # Escape hatch (not recommended):
    #   export EIEL_ALLOW_CHECKPOINT_V1=1
    # ------------------------------------------------------------------
    _allow_v1 = str(os.environ.get("EIEL_ALLOW_CHECKPOINT_V1", "0") or "0").strip().lower()
    if _allow_v1 not in ("1", "true", "yes"):
        raise RuntimeError(
            "Checkpoint v1 loading is disabled (v2-only engine). \
Provide a bundle v2 path (checkpoint_bundles_v2/bundle_genXXXXXX), \
or set EIEL_ALLOW_CHECKPOINT_V1=1 to override."
        )
    require_trusted_checkpoint_artifact(
        ckpt_dir,
        trusted=trusted,
        loader="checkpoint v1 legacy loader",
        reason="v1 uses NumPy object arrays, pickle, and optional torch checkpoint loading",
    )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    _validate_meta(meta)

    # Integrity verification
    manifest = meta.get("file_manifest", {})
    _verify_manifest(ckpt_dir, manifest)

    world = dict(np.load(os.path.join(ckpt_dir, "world.npz"), allow_pickle=False))
    agents = dict(np.load(os.path.join(ckpt_dir, "agents.npz"), allow_pickle=True))
    genomes_npz = dict(np.load(os.path.join(ckpt_dir, "genomes.npz"), allow_pickle=True))
    genomes = _decode_genomes(genomes_npz)

    rng = None
    rng_path = os.path.join(ckpt_dir, "rng.pkl")
    if os.path.exists(rng_path):
        with open(rng_path, "rb") as f:
            rng = pickle.load(f)

    rng_snapshot_v2 = None
    rng_snapshot_path = os.path.join(ckpt_dir, "rng_snapshot_v2.json")
    if os.path.exists(rng_snapshot_path):
        try:
            import json
            with open(rng_snapshot_path, "r", encoding="utf-8") as f:
                rng_snapshot_v2 = json.loads(f.read())
        except Exception:
            rng_snapshot_v2 = None

    banks = None
    banks_path = os.path.join(ckpt_dir, "banks.pt")
    if os.path.exists(banks_path) and torch is not None:
        banks = torch.load(banks_path, map_location="cpu")

    return {
        "meta": meta,
        "config_snapshot": meta.get("config_snapshot", {}),
        "world": world,
        "agents": agents,
        "genomes": genomes,
        "rng": rng,
        "rng_snapshot_v2": rng_snapshot_v2,
        "banks": banks,
    }


def checkpoint_apply(sim: Any, payload: Dict[str, Any], *, restore_rng: bool = True, restore_banks: bool = True) -> None:
    """Apply a loaded checkpoint payload onto an existing SimulationRunner.

    Parameters
    ----------
    sim:
        SimulationRunner instance to modify in-place.
    payload:
        Dict returned by checkpoint_load.
    restore_rng:
        If True and RNG payload exists, restores RNG states.
    restore_banks:
        If True and bank payload exists, restores bank tensors and GRU memory.
    """
    # ------------------------------------------------------------
    # Checkpoint Bundle v2 (Phase 3) payload support
    # ------------------------------------------------------------
    if isinstance(payload, dict) and (payload.get("_format") == "bundle_v2" or "metadata" in payload):
        try:
            from .checkpoint_bundle_v2 import checkpoint_bundle_v2_apply  # noqa: WPS433 (local import)
        except Exception as e:
            raise RuntimeError(f"Cannot import checkpoint_bundle_v2 apply helper: {e}") from e
        checkpoint_bundle_v2_apply(sim, payload, restore_rng=bool(restore_rng))
        # Phase 3 determinism: run slot recanonicalization AFTER bundle apply performs resume_apply_post strict checks.
        if getattr(sim, "_eiel_pending_slot_recanonicalize", False):
            try:
                from .checkpoint_bundle_v2 import _eiel_recanonicalize_slots_for_alive  # noqa: WPS433 (local import)
                _eiel_recanonicalize_slots_for_alive(getattr(sim, "pop", None))
            finally:
                try:
                    delattr(sim, "_eiel_pending_slot_recanonicalize")
                except Exception:
                    pass
        return
    world = payload.get("world", {})
    agents = payload.get("agents", {})
    genomes = payload.get("genomes", {})

    _apply_world(sim, world)
    _apply_agents_and_genomes(sim, agents, genomes)

    if restore_banks and payload.get("banks") is not None:
        _apply_banks(sim, payload["banks"])
        # Ensure required runtime buffers exist even if older checkpoints
        # didn't store them (or resume skipped init).
        _ensure_gru_pack_exists(sim)
        # Critical: Agent.policy is required for reproduction (policy crossover).
        # On checkpoint load we reconstruct policies from the restored parameter banks.
        _rehydrate_agent_policies_from_banks(sim)

    if restore_rng:
        # Phase 1: prefer rng_snapshot_v2 when RNGManager is present.
        snap = payload.get("rng_snapshot_v2")
        if snap is not None and getattr(sim, "rng_manager", None) is not None:
            try:
                sim.rng_manager.apply_snapshot_dict(snap)
            except Exception:
                pass
        elif payload.get("rng") is not None:
            _apply_rng(sim, payload["rng"])

    # Safety: refuse to proceed with missing policies (will crash on next birth).

    # -----------------------------
    # Phase 2: ControlPlane submission sequence persistence
    # -----------------------------
    try:
        cp_meta = (payload.get("meta", {}) or {}).get("control_plane", {}) or {}
        ssn = int(cp_meta.get("submission_sequence_number", 0) or 0)
        # Runner keeps a mirror field; ControlPlane (if present) is authoritative runtime state.
        setattr(sim, "_control_plane_submission_sequence_number", ssn)
        cp = getattr(sim, "control_plane", None)
        if cp is not None:
            try:
                cp.set_submission_sequence_number(ssn)
            except Exception:
                pass
    except Exception:
        pass

    # -----------------------------
    # Phase 6 (draft): PressureAutopilot state restore
    # -----------------------------
    try:
        meta = payload.get("meta", {}) or {}
        ap_state = meta.get("pressure_autopilot_state", None)
        ap = getattr(sim, "pressure_autopilot", None)
        if ap_state is not None and ap is not None and hasattr(ap, "restore_state"):
            ap.restore_state(ap_state)
    except Exception:
        pass

    missing = [a.agent_id for a in getattr(getattr(sim, "pop", None), "agents", {}).values() if getattr(a, "policy", None) is None]
    if missing:
        raise ValueError(
            f"Checkpoint applied but {len(missing)} agents have policy=None. "
            f"Resume requires policies (or banks to reconstruct them). First few missing: {missing[:5]}"
        )


def rotate_checkpoints(checkpoint_root_dir: str, *, keep_last: int = 5) -> None:
    """Delete older checkpoints in a directory, keeping the newest N.

    Common industry behavior:
    - Keep the last N checkpoints, delete older ones.
    - Ordering is based on generation number in folder name when possible, else mtime.

    Parameters
    ----------
    checkpoint_root_dir:
        Directory that contains checkpoint subdirectories (e.g. run_dir/checkpoints).
    keep_last:
        Number of checkpoints to retain. 0 or less disables deletion.
    """
    keep_last = int(keep_last)
    if keep_last <= 0:
        return

    root = os.path.abspath(checkpoint_root_dir)
    if not os.path.isdir(root):
        return

    entries = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        if not os.path.exists(os.path.join(p, "meta.json")):
            continue
        gen = _parse_gen_from_name(name)
        mtime = os.path.getmtime(p)
        entries.append((gen, mtime, p))

    # Sort newest first:
    # - Prefer higher gen if available, otherwise mtime.
    def _key(item):
        gen, mtime, _p = item
        # gen None sorts after all real gens when reversed
        return (-1 if gen is None else gen, mtime)

    # We want newest first; easiest: sort by (gen or -inf, mtime) ascending then reverse.
    entries.sort(key=lambda t: ((-10**18 if t[0] is None else t[0]), t[1]))
    entries = list(reversed(entries))

    for _idx, (_gen, _mtime, p) in enumerate(entries):
        if _idx < keep_last:
            continue
        try:
            shutil.rmtree(p, ignore_errors=True)
        except Exception:
            pass


# -----------------------------
# Internal helpers: meta + manifest
# -----------------------------
def _build_meta(sim: Any, *, include_rng: bool, include_banks: bool, label: str) -> Dict[str, Any]:
    cfg_snapshot = {
        "env_cfg": _to_jsonable(getattr(sim, "env_cfg", None)),
        "pop_cfg": _to_jsonable(getattr(sim, "pop_cfg", None)),
        "sim_cfg": _to_jsonable(getattr(sim, "sim_cfg", None)),
    }

    run_dir = getattr(sim, "run_dir", "")
    gen = int(getattr(sim, "_current_gen", -1)) if hasattr(sim, "_current_gen") else None
    pred_tick = int(getattr(sim, "_pred_tick", 0) or 0)

    meta: Dict[str, Any] = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "created_utc": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "run_dir": run_dir,
        "label": str(label or ""),
        "git_commit": _git_commit_if_available(sim),
        "schema_versions": {
            "world_schema_version": WORLD_SCHEMA_VERSION,
            "agent_schema_version": AGENT_SCHEMA_VERSION,
            "genome_schema_version": GENOME_SCHEMA_VERSION,
            "rng_schema_version": RNG_SCHEMA_VERSION,
            "bank_schema_version": BANK_SCHEMA_VERSION,
        },
        "sim_position": {
            "generation": gen,
            "pred_tick": pred_tick,
        },
        "includes": {
            "rng": bool(include_rng),
            "banks": bool(include_banks),
        },
        "config_snapshot": cfg_snapshot,
        "control_plane": {
            # Phase 2: persist submission_sequence_number (monotonic per branch).
            # This is not used for ordering inside a single run (in-memory), but is required
            # so that resumes keep deterministic command ordering.
            "branch_id": str(getattr(getattr(sim, "sim_cfg", None), "branch_id", "default") or "default"),
            "submission_sequence_number": int(getattr(sim, "_control_plane_submission_sequence_number", 0) or 0),
        },
        "file_manifest": {},
    }

    # Phase 6 (draft): PressureAutopilot state (checkpoint/resume equivalence).
    try:
        ap = getattr(sim, "pressure_autopilot", None)
        if ap is not None and hasattr(ap, "export_state"):
            meta["pressure_autopilot_state"] = ap.export_state()
    except Exception:
        pass
    return meta


def _validate_meta(meta: Dict[str, Any]) -> None:
    if int(meta.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint_version: {meta.get('checkpoint_version')}")
    sv = meta.get("schema_versions", {}) or {}
    if int(sv.get("world_schema_version", -1)) != WORLD_SCHEMA_VERSION:
        raise ValueError(f"Unsupported world_schema_version: {sv.get('world_schema_version')}")
    if int(sv.get("agent_schema_version", -1)) != AGENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported agent_schema_version: {sv.get('agent_schema_version')}")
    if int(sv.get("genome_schema_version", -1)) != GENOME_SCHEMA_VERSION:
        raise ValueError(f"Unsupported genome_schema_version: {sv.get('genome_schema_version')}")


def _compute_manifest(dir_path: str) -> Dict[str, Dict[str, Any]]:
    manifest: Dict[str, Dict[str, Any]] = {}
    for fname in sorted(os.listdir(dir_path)):
        if fname.endswith(".tmp"):
            continue
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        manifest[fname] = {
            "bytes": int(os.path.getsize(fpath)),
            "sha256": _sha256_file(fpath),
        }
    return manifest


def _verify_manifest(ckpt_dir: str, manifest: Dict[str, Any]) -> None:
    for fname, info in (manifest or {}).items():
        fpath = os.path.join(ckpt_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Checkpoint missing file: {fname}")
        exp = str((info or {}).get("sha256", ""))
        got = _sha256_file(fpath)
        if exp and exp != got:
            raise ValueError(f"Checkpoint file hash mismatch for {fname}: expected {exp} got {got}")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rand_hex(n: int) -> str:
    return os.urandom(max(1, int(n))).hex()[: int(n)]


def _parse_gen_from_name(name: str) -> Optional[int]:
    m = re.search(r"gen(\d+)", name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _git_commit_if_available(sim: Any) -> str:
    # Best-effort: allow caller to inject git commit in sim if they want.
    commit = getattr(sim, "git_commit", "") or ""
    if commit:
        return str(commit)

    # Try to read from .git in repo root relative to run_dir, if present.
    run_dir = getattr(sim, "run_dir", "") or ""
    if not run_dir:
        return ""
    here = os.path.abspath(run_dir)
    for _ in range(6):
        git_dir = os.path.join(here, ".git")
        head = os.path.join(git_dir, "HEAD")
        if os.path.exists(head):
            try:
                with open(head, "r", encoding="utf-8") as f:
                    ref = f.read().strip()
                if ref.startswith("ref:"):
                    ref_path = ref.split(":", 1)[1].strip()
                    ref_file = os.path.join(git_dir, ref_path)
                    if os.path.exists(ref_file):
                        with open(ref_file, "r", encoding="utf-8") as rf:
                            return rf.read().strip()
                # Detached head: HEAD contains hash
                if re.fullmatch(r"[0-9a-fA-F]{7,40}", ref):
                    return ref
            except Exception:
                return ""
        here = os.path.dirname(here)
    return ""


def _to_jsonable(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        # Avoid unserializable junk by filtering obvious private fields.
        d = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            d[k] = v
        return d
    return obj


# -----------------------------
# Save: world, agents, genomes
# -----------------------------
def _save_world_npz(sim: Any, out_path: str) -> None:
    env = getattr(sim, "env", None)
    if env is None:
        raise ValueError("sim.env missing")

    # Core arrays
    payload = {
        "grid": np.asarray(getattr(env, "grid")),
        "agent_grid": np.asarray(getattr(env, "agent_grid")),
        "biome_grid": np.asarray(getattr(env, "biome_grid")),
        "agent_pos_y": np.asarray(getattr(env, "agent_pos_y")),
        "agent_pos_x": np.asarray(getattr(env, "agent_pos_x")),
        "agent_alive": np.asarray(getattr(env, "agent_alive")),
        # Runtime multipliers
        "food_spawn_mult": np.asarray([float(getattr(env, "food_spawn_mult", 1.0))], dtype=np.float32),
        "action_cost_mult": np.asarray([float(getattr(env, "action_cost_mult", 1.0))], dtype=np.float32),
        "rest_cost_mult": np.asarray([float(getattr(env, "rest_cost_mult", 1.0))], dtype=np.float32),
    }
    np.savez(out_path, **payload)


def _save_agents_npz(sim: Any, out_path: str) -> None:
    pop = getattr(sim, "pop", None)
    env = getattr(sim, "env", None)
    if pop is None or env is None:
        raise ValueError("sim.pop or sim.env missing")

    agents = list(getattr(pop, "agents", {}).values())
    # Stable ordering by agent_id
    agents.sort(key=lambda a: int(getattr(a, "agent_id", -1)))

    n = len(agents)
    agent_id = np.empty((n,), dtype=np.int32)
    alive = np.empty((n,), dtype=np.bool_)
    pos_y = np.empty((n,), dtype=np.int32)
    pos_x = np.empty((n,), dtype=np.int32)
    energy = np.empty((n,), dtype=np.float32)
    health = np.empty((n,), dtype=np.float32)
    injury = np.empty((n,), dtype=np.float32)
    habitat_acclimation_state = np.empty((n,), dtype=np.float32)
    rest_streak = np.empty((n,), dtype=np.int32)
    repro_cooldown = np.empty((n,), dtype=np.int32)
    slot = np.empty((n,), dtype=np.int32)
    slot_token = np.empty((n,), dtype=np.int32)
    drives = np.empty((n, 4), dtype=np.float32)
    chemistry = np.empty((n, 6), dtype=np.float32)
    maintenance = np.empty((n, 4), dtype=np.float32)

    stats_json = np.empty((n,), dtype=object)

    # Build a quick position lookup from env per-agent arrays (authoritative)
    # If out of bounds or missing, fall back to agent object fields (if any)
    env_pos_y = getattr(env, "agent_pos_y", None)
    env_pos_x = getattr(env, "agent_pos_x", None)

    for i, a in enumerate(agents):
        aid = int(getattr(a, "agent_id"))
        agent_id[i] = aid
        alive[i] = bool(getattr(a, "alive", True))
        if env_pos_y is not None and aid < len(env_pos_y):
            pos_y[i] = int(env_pos_y[aid])
            pos_x[i] = int(env_pos_x[aid])
        else:
            pos_y[i] = int(getattr(a, "y", -1))
            pos_x[i] = int(getattr(a, "x", -1))

        energy[i] = float(getattr(a, "energy", 0.0))
        health[i] = float(getattr(a, "health", 0.0))
        injury[i] = float(getattr(a, "injury", 0.0))
        habitat_acclimation_state[i] = float(getattr(a, "habitat_acclimation_state", 0.0))
        rest_streak[i] = int(getattr(a, "rest_streak", 0))
        repro_cooldown[i] = int(getattr(a, "repro_cooldown", 0))
        slot[i] = int(getattr(a, "slot", -1))
        slot_token[i] = int(getattr(a, "slot_token", 0))

        d = getattr(a, "drives", None)
        if d is None:
            drives[i, :] = 0.0
        else:
            dd = np.asarray(d, dtype=np.float32).reshape(-1)
            if dd.size >= 4:
                drives[i, :] = dd[:4]
            else:
                drives[i, :dd.size] = dd
                drives[i, dd.size:] = 0.0

        c = getattr(a, "chemistry", None)
        if c is None:
            chemistry[i, :] = 0.0
        else:
            cc = np.asarray(c, dtype=np.float32).reshape(-1)
            if cc.size >= 6:
                chemistry[i, :] = cc[:6]
            else:
                chemistry[i, :cc.size] = cc
                chemistry[i, cc.size:] = 0.0

        m = getattr(a, "maintenance", None)
        if m is None:
            maintenance[i, :] = np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32)
        else:
            mm = np.asarray(m, dtype=np.float32).reshape(-1)
            maintenance[i, :] = np.array([0.0, -1.0, 0.0, -1.0], dtype=np.float32)
            if mm.size >= 4:
                maintenance[i, :] = mm[:4]
            else:
                maintenance[i, :mm.size] = mm

        st = getattr(a, "stats", None)
        try:
            if is_dataclass(st):
                stats_json[i] = json.dumps(asdict(st))
            elif hasattr(st, "__dict__"):
                stats_json[i] = json.dumps(st.__dict__)
            else:
                stats_json[i] = json.dumps({})
        except Exception:
            stats_json[i] = json.dumps({})

    # Slot pool
    slots = getattr(pop, "_slots", None)
    if slots is None:
        slotpool_max = np.asarray([int(getattr(pop, "max_slots", 0))], dtype=np.int32)
        slotpool_free = np.asarray([], dtype=np.int32)
        slotpool_tokens = np.asarray([], dtype=np.int32)
    else:
        slotpool_max = np.asarray([int(getattr(slots, "max_slots", 0))], dtype=np.int32)
        slotpool_free = np.asarray(list(getattr(slots, "free_slots", [])), dtype=np.int32)
        slotpool_tokens = np.asarray(list(getattr(slots, "slot_tokens", [])), dtype=np.int32)

    # Runner aux that affects determinism
    pred_tick = np.asarray([int(getattr(sim, "_pred_tick", 0) or 0)], dtype=np.int32)

    payload = dict(
        agent_id=agent_id,
        alive=alive,
        pos_y=pos_y,
        pos_x=pos_x,
        energy=energy,
        health=health,
        injury=injury,
        habitat_acclimation_state=habitat_acclimation_state,
        rest_streak=rest_streak,
        repro_cooldown=repro_cooldown,
        slot=slot,
        slot_token=slot_token,
        drives=drives,
        chemistry=chemistry,
        maintenance=maintenance,
        stats_json=stats_json,
        population_next_id=np.asarray([int(getattr(pop, "next_id", 0))], dtype=np.int32),
        slotpool_max_slots=slotpool_max,
        slotpool_free_slots=slotpool_free,
        slotpool_slot_tokens=slotpool_tokens,
        runner_pred_tick=pred_tick,
    )
    np.savez(out_path, **payload)


def _save_genomes_npz(sim: Any, out_path: str) -> None:
    pop = getattr(sim, "pop", None)
    if pop is None:
        raise ValueError("sim.pop missing")

    agents = list(getattr(pop, "agents", {}).values())
    agents.sort(key=lambda a: int(getattr(a, "agent_id", -1)))

    n = len(agents)
    agent_id = np.empty((n,), dtype=np.int32)
    genome_json = np.empty((n,), dtype=object)

    for i, a in enumerate(agents):
        aid = int(getattr(a, "agent_id"))
        agent_id[i] = aid
        g = getattr(a, "genome", None)
        try:
            if g is None:
                genome_json[i] = json.dumps({}, sort_keys=True)
            elif hasattr(g, "to_dict") and callable(getattr(g, "to_dict")):
                genome_json[i] = json.dumps(g.to_dict(), sort_keys=True)
            elif is_dataclass(g):
                genome_json[i] = json.dumps(asdict(g), sort_keys=True)
            elif hasattr(g, "__dict__"):
                genome_json[i] = json.dumps(g.__dict__, sort_keys=True)
            else:
                genome_json[i] = json.dumps({}, sort_keys=True)
        except Exception:
            genome_json[i] = json.dumps({}, sort_keys=True)

    np.savez(out_path, agent_id=agent_id, genome_json=genome_json)


def _decode_genomes(genomes_npz: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    agent_id = np.asarray(genomes_npz.get("agent_id", []), dtype=np.int32).tolist()
    genome_json = genomes_npz.get("genome_json", [])
    out: Dict[int, Dict[str, Any]] = {}
    for i, aid in enumerate(agent_id):
        try:
            s = genome_json[i]
            if isinstance(s, bytes):
                s = s.decode("utf-8")
            gd = json.loads(s) if s else {}
        except Exception:
            gd = {}
        out[int(aid)] = gd
    return out


# -----------------------------
# Save/load RNG
# -----------------------------
def _save_rng(sim: Any, out_path: str) -> None:
    rng_payload: Dict[str, Any] = {}

    # Phase 1 strict rule:
    # - If rng_snapshot_v2 is available (boundary-aligned), it is the authoritative RNG state.
    # - Legacy pickle RNG payload is written ONLY when v2 is not available, to avoid multiple sources of truth.
    wrote_v2 = False
    try:
        rm = getattr(sim, "rng_manager", None)
        # Prefer the boundary-aligned snapshot produced by SafeBoundary at G+1.
        snap = getattr(sim, "_last_rng_snapshot_v2", None)
        if snap is None and rm is not None:
            # Fallback: capture using boundary next_gen if present.
            next_gen = int(getattr(sim, "_last_safe_boundary_next_gen", -1))
            if next_gen >= 0:
                snap = rm.capture_snapshot(generation_index=next_gen)
        if snap is not None:
            import os
            import json
            json_path = os.path.join(os.path.dirname(out_path), "rng_snapshot_v2.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                jf.write(json.dumps(snap, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            wrote_v2 = True
    except Exception:
        wrote_v2 = False

    # If we wrote v2, do not write legacy RNG payload (prevents ambiguous restore precedence).
    if wrote_v2:
        with open(out_path, "wb") as f:
            pickle.dump(rng_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    import random as _random

    # Global Python RNG (in case any module uses it)
    try:
        rng_payload["python_random_global"] = _random.getstate()
    except Exception:
        pass

    # Global NumPy RNG (legacy RandomState)
    try:
        rng_payload["numpy_random_global"] = np.random.get_state()
    except Exception:
        pass

    # Runner and component RNGs
    env = getattr(sim, "env", None)
    pop = getattr(sim, "pop", None)

    try:
        if env is not None and hasattr(env, "rng"):
            rng_payload["env_rng_state"] = getattr(env.rng, "bit_generator").state
    except Exception:
        pass

    try:
        if pop is not None and hasattr(pop, "rng"):
            rng_payload["pop_py_rng_state"] = pop.rng.getstate()
    except Exception:
        pass

    try:
        if pop is not None and hasattr(pop, "np_rng"):
            rng_payload["pop_np_rng_state"] = pop.np_rng.bit_generator.state
    except Exception:
        pass

    try:
        if hasattr(sim, "np_rng"):
            rng_payload["runner_np_rng_state"] = sim.np_rng.bit_generator.state
    except Exception:
        pass

    # Torch RNGs
    if torch is not None:
        try:
            rng_payload["torch_cpu_rng_state"] = torch.random.get_rng_state().cpu()
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                rng_payload["torch_cuda_rng_states"] = [t.cpu() for t in torch.cuda.get_rng_state_all()]
        except Exception:
            pass

    with open(out_path, "wb") as f:
        pickle.dump(rng_payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _apply_rng(sim: Any, rng_payload: Dict[str, Any]) -> None:
    import random as _random

    try:
        if "python_random_global" in rng_payload:
            _random.setstate(rng_payload["python_random_global"])
    except Exception:
        pass

    try:
        if "numpy_random_global" in rng_payload:
            np.random.set_state(rng_payload["numpy_random_global"])
    except Exception:
        pass

    env = getattr(sim, "env", None)
    pop = getattr(sim, "pop", None)

    try:
        if env is not None and hasattr(env, "rng") and "env_rng_state" in rng_payload:
            env.rng.bit_generator.state = rng_payload["env_rng_state"]
    except Exception:
        pass

    try:
        if pop is not None and hasattr(pop, "rng") and "pop_py_rng_state" in rng_payload:
            pop.rng.setstate(rng_payload["pop_py_rng_state"])
    except Exception:
        pass

    try:
        if pop is not None and hasattr(pop, "np_rng") and "pop_np_rng_state" in rng_payload:
            pop.np_rng.bit_generator.state = rng_payload["pop_np_rng_state"]
    except Exception:
        pass

    try:
        if hasattr(sim, "np_rng") and "runner_np_rng_state" in rng_payload:
            sim.np_rng.bit_generator.state = rng_payload["runner_np_rng_state"]
    except Exception:
        pass

    if torch is not None:
        try:
            if "torch_cpu_rng_state" in rng_payload:
                torch.random.set_rng_state(rng_payload["torch_cpu_rng_state"])
        except Exception:
            pass
        try:
            if torch.cuda.is_available() and "torch_cuda_rng_states" in rng_payload:
                torch.cuda.set_rng_state_all(rng_payload["torch_cuda_rng_states"])
        except Exception:
            pass


# -----------------------------
# Save/load banks (torch tensors)
# -----------------------------
def _bank_attr_names() -> List[str]:
    # MLP banks + pack
    base = ["_W1", "_b1", "_W2", "_b2", "_W3", "_b3", "_mlp_pack"]
    # GRU banks + memory
    # NOTE: _gru_pack is a required runtime buffer for batched GRU inference.
    # Older checkpoints may not contain it; we can reconstruct it on load.
    gru = [
        "_Hmem",
        "_encW",
        "_encb",
        "_Wih",
        "_bih",
        "_Whh",
        "_bhh",
        "_headW",
        "_headb",
        "_gru_pack",
    ]
    return base + gru


def _save_banks(sim: Any, out_path: str) -> None:
    if torch is None:
        return
    payload: Dict[str, Any] = {}
    for name in _bank_attr_names():
        t = getattr(sim, name, None)
        if t is None:
            continue
        try:
            if hasattr(t, "detach"):
                payload[name] = t.detach().to("cpu")
            else:
                payload[name] = t
        except Exception:
            continue

    # Also save internal bank flags if present
    for name in ["_gru_inited", "_gru_hidden"]:
        if hasattr(sim, name):
            payload[name] = getattr(sim, name)

    torch.save(payload, out_path)


def _apply_banks(sim: Any, banks_payload: Dict[str, Any]) -> None:
    if torch is None:
        return
    # Restore tensors to current bank device if possible, else keep on CPU.
    bank_dev = getattr(sim, "bank_device", None) or getattr(sim, "device", None)
    for name in _bank_attr_names():
        if name not in banks_payload:
            continue
        t = banks_payload[name]
        try:
            if hasattr(t, "to") and bank_dev is not None:
                setattr(sim, name, t.to(bank_dev))
            else:
                setattr(sim, name, t)
        except Exception:
            setattr(sim, name, t)

    for name in ["_gru_inited", "_gru_hidden"]:
        if name in banks_payload:
            try:
                setattr(sim, name, banks_payload[name])
            except Exception:
                pass

    # Mark banks as clean if runner supports the dirty flags
    if hasattr(sim, "_mark_bank_dirty"):
        # We loaded explicit banks so they should be consistent.
        try:
            setattr(sim, "_bank_dirty_mlp", True)
            setattr(sim, "_bank_dirty_gru", True)
        except Exception:
            pass


def _ensure_gru_pack_exists(sim: Any) -> None:
    """Ensure `sim._gru_pack` exists when GRU banking is enabled.

    Why
    ----
    Batched GRU inference uses `self._gru_pack` in runner_policy.py. This tensor
    is normally created by BankMixin._ensure_gru_bank(). Older checkpoints may
    omit `_gru_pack` from banks.pt, and resume paths may skip the init phase
    that would have allocated it.

    This helper allocates `_gru_pack` if (a) GRU appears initialized and
    (b) `_gru_pack` is missing.
    """
    if torch is None:
        return

    # If GRU isn't inited, nothing to do.
    if not bool(getattr(sim, "_gru_inited", False)):
        return

    if getattr(sim, "_gru_pack", None) is not None:
        return

    # We need enough info to compute the packed size.
    encW = getattr(sim, "_encW", None)
    W1 = getattr(sim, "_W1", None)
    if encW is None or W1 is None:
        return

    try:
        max_slots = int(W1.shape[0])
        O = int(getattr(sim, "_obs_size", 0) or 0)
        A = int(getattr(sim, "_out", 0) or 0)
        Hm = int(getattr(sim, "_gru_hidden", 64) or 64)
        if O <= 0:
            # Fallback: infer O from encW
            O = int(encW.shape[-1])
        if A <= 0:
            headW = getattr(sim, "_headW", None)
            if headW is not None:
                A = int(headW.shape[-2])
        if O <= 0 or A <= 0 or max_slots <= 0:
            return

        P_gru = (Hm * O) + Hm + (3 * Hm * Hm) + (3 * Hm) + (3 * Hm * Hm) + (3 * Hm) + (A * Hm) + A

        dev = getattr(sim, "_bank_device", None) or getattr(sim, "device", None) or "cpu"
        dt = getattr(sim, "_bank_dtype", None) or getattr(W1, "dtype", torch.float32)
        sim._gru_pack = torch.empty((max_slots, P_gru), device=dev, dtype=dt)
    except Exception:
        # If something is off, fail soft; caller will hit a clearer error later.
        return


def _rehydrate_agent_policies_from_banks(sim: Any) -> None:
    """Reconstruct per-agent policy modules from restored parameter banks.

    Why this exists
    ---------------
    Population.make_child() performs policy crossover using parent.policy
    objects (it calls .state_dict() and clones weights).

    Our checkpoint v1 stores banks.pt (dense per-slot parameter banks) rather
    than pickling thousands of policy modules. When resuming, agents are
    reconstructed with policy=None for safety and to avoid circular imports.
    This helper rebuilds those policy modules deterministically from the
    restored banks.

    Notes
    -----
    - Policies are created on CPU. The runner's banked inference path uses the
      banks already, and new births will mark banks dirty and resync as needed.
    - If banks are missing, this function does nothing.
    """
    if torch is None:
        return

    pop = getattr(sim, "pop", None)
    if pop is None:
        return

    # Determine obs/action sizes
    obs_size = int(getattr(sim, "_obs_size", getattr(sim, "obs_size", 0)) or 0)
    n_actions = int(getattr(sim, "_out", 7) or 7)

    # Bank tensors (must exist to reconstruct)
    W1 = getattr(sim, "_W1", None)
    b1 = getattr(sim, "_b1", None)
    W2 = getattr(sim, "_W2", None)
    b2 = getattr(sim, "_b2", None)
    W3 = getattr(sim, "_W3", None)
    b3 = getattr(sim, "_b3", None)

    # GRU bank tensors
    encW = getattr(sim, "_encW", None)
    encb = getattr(sim, "_encb", None)
    Wih = getattr(sim, "_Wih", None)
    bih = getattr(sim, "_bih", None)
    Whh = getattr(sim, "_Whh", None)
    bhh = getattr(sim, "_bhh", None)
    headW = getattr(sim, "_headW", None)
    headb = getattr(sim, "_headb", None)

    from agents.agent_policy import MLPPolicy
    from agents.gru_policy import GRUPolicy

    # Reconstruct policy modules per agent
    for a in getattr(pop, "agents", {}).values():
        if getattr(a, "policy", None) is not None:
            continue

        slot = int(getattr(a, "slot", -1))
        if slot < 0:
            continue

        mem_dim = int(getattr(getattr(a, "genome", None), "memory_dim", 0) or 0)
        use_gru = bool(getattr(getattr(a, "genome", None), "use_gru", mem_dim > 0))

        if use_gru:
            # Require GRU bank tensors
            if encW is None or encb is None or Wih is None or bih is None or Whh is None or bhh is None or headW is None or headb is None:
                # Can't reconstruct GRU policies without banks.
                continue

            pol = GRUPolicy(obs_size=obs_size, hidden_dim=64, n_actions=n_actions)
            with torch.no_grad():
                pol.encoder.weight.copy_(encW[slot].detach().cpu())
                pol.encoder.bias.copy_(encb[slot].detach().cpu())
                pol.gru.weight_ih.copy_(Wih[slot].detach().cpu())
                pol.gru.bias_ih.copy_(bih[slot].detach().cpu())
                pol.gru.weight_hh.copy_(Whh[slot].detach().cpu())
                pol.gru.bias_hh.copy_(bhh[slot].detach().cpu())
                pol.head.weight.copy_(headW[slot].detach().cpu())
                pol.head.bias.copy_(headb[slot].detach().cpu())

            a.policy = pol
            # Mark as already-synced so runner doesn't immediately resync.
            try:
                a._gru_bank_token = -1  # force bank resync after resume
                a._eiel_force_full_gru_bank_sync_once = True
            except Exception:
                pass

        else:
            # Require MLP bank tensors
            if W1 is None or b1 is None or W2 is None or b2 is None or W3 is None or b3 is None:
                continue

            hidden = int(getattr(getattr(a, "genome", None), "hidden_size", 64) or 64)
            pol = MLPPolicy(obs_size=obs_size, hidden_size=hidden, n_actions=n_actions)
            with torch.no_grad():
                pol.net[0].weight.copy_(W1[slot].detach().cpu())
                pol.net[0].bias.copy_(b1[slot].detach().cpu())
                pol.net[2].weight.copy_(W2[slot].detach().cpu())
                pol.net[2].bias.copy_(b2[slot].detach().cpu())
                pol.net[4].weight.copy_(W3[slot].detach().cpu())
                pol.net[4].bias.copy_(b3[slot].detach().cpu())

            a.policy = pol
            try:
                a._bank_token = -1  # force bank resync after resume
                a._eiel_force_full_mlp_bank_sync_once = True
            except Exception:
                pass

    # Policies now match the restored banks; skip immediate resync.
    try:
        setattr(sim, "_bank_dirty_mlp", True)
        setattr(sim, "_bank_dirty_gru", True)
    except Exception:
        pass

    # After reconstruction, banks are consistent with these policies by construction.
    try:
        setattr(sim, "_bank_dirty_mlp", True)
        setattr(sim, "_bank_dirty_gru", True)
    except Exception:
        pass


# -----------------------------
# Apply: world + agents + genomes
# -----------------------------
def _apply_world(sim: Any, world: Dict[str, Any]) -> None:
    env = getattr(sim, "env", None)
    if env is None:
        raise ValueError("sim.env missing")

    # Assign arrays
    for name in ["grid", "agent_grid", "biome_grid", "agent_pos_y", "agent_pos_x", "agent_alive"]:
        if name in world:
            setattr(env, name, np.asarray(world[name]))

    # Multipliers
    try:
        if "food_spawn_mult" in world:
            env.food_spawn_mult = float(np.asarray(world["food_spawn_mult"]).reshape(-1)[0])
        if "action_cost_mult" in world:
            env.action_cost_mult = float(np.asarray(world["action_cost_mult"]).reshape(-1)[0])
        if "rest_cost_mult" in world:
            env.rest_cost_mult = float(np.asarray(world["rest_cost_mult"]).reshape(-1)[0])
    except Exception:
        pass


def _apply_agents_and_genomes(sim: Any, agents_npz: Dict[str, Any], genomes: Dict[int, Dict[str, Any]]) -> None:
    pop = getattr(sim, "pop", None)
    env = getattr(sim, "env", None)
    if pop is None or env is None:
        raise ValueError("sim.pop or sim.env missing")

    # Restore next_id
    try:
        pop.next_id = int(np.asarray(agents_npz.get("population_next_id", [0])).reshape(-1)[0])
    except Exception:
        pass

    # Restore slot pool
    slots = getattr(pop, "_slots", None)
    if slots is not None:
        try:
            slots.max_slots = int(np.asarray(agents_npz.get("slotpool_max_slots", [slots.max_slots])).reshape(-1)[0])
            free = list(np.asarray(agents_npz.get("slotpool_free_slots", []), dtype=np.int32).tolist())
            tokens = list(np.asarray(agents_npz.get("slotpool_slot_tokens", []), dtype=np.int32).tolist())
            # Preserve deque type on free_slots
            try:
                from collections import deque
                slots.free_slots = deque(int(x) for x in free)
            except Exception:
                slots.free_slots = free  # type: ignore
            slots.slot_tokens = [int(x) for x in tokens]
        except Exception:
            pass

    # Runner aux
    try:
        sim._pred_tick = int(np.asarray(agents_npz.get("runner_pred_tick", [0])).reshape(-1)[0])
    except Exception:
        pass

    # Rebuild agents dict
    # We intentionally avoid constructing new policies here; the Population/Runner can rebuild banks later.
    # Existing agents are replaced in-place.
    agent_ids = np.asarray(agents_npz.get("agent_id", []), dtype=np.int32).tolist()
    alive = np.asarray(agents_npz.get("alive", []), dtype=np.bool_)
    energy = np.asarray(agents_npz.get("energy", []), dtype=np.float32)
    health = np.asarray(agents_npz.get("health", []), dtype=np.float32)
    injury = np.asarray(agents_npz.get("injury", []), dtype=np.float32)
    habitat_acclimation_state = np.asarray(agents_npz.get("habitat_acclimation_state", []), dtype=np.float32)
    rest_streak = np.asarray(agents_npz.get("rest_streak", []), dtype=np.int32)
    repro_cooldown = np.asarray(agents_npz.get("repro_cooldown", []), dtype=np.int32)
    slot = np.asarray(agents_npz.get("slot", []), dtype=np.int32)
    slot_token = np.asarray(agents_npz.get("slot_token", []), dtype=np.int32)
    drives = np.asarray(agents_npz.get("drives", np.zeros((len(agent_ids), 4), dtype=np.float32)), dtype=np.float32)
    chemistry = np.asarray(agents_npz.get("chemistry", np.zeros((len(agent_ids), 6), dtype=np.float32)), dtype=np.float32)
    maintenance = np.asarray(
        agents_npz.get("maintenance", np.tile(np.asarray([0.0, -1.0, 0.0, -1.0], dtype=np.float32), (len(agent_ids), 1))),
        dtype=np.float32,
    )
    stats_json = agents_npz.get("stats_json", np.array(["{}"] * len(agent_ids), dtype=object))

    # Import classes lazily to avoid circular import problems
    from agents.agent import Agent, AgentStats, normalize_lineage_stats
    from agents.genome import Genome

    new_agents = {}
    for i, aid in enumerate(agent_ids):
        gd = genomes.get(int(aid), {})
        # Genome reconstruction with defaults for missing fields
        g = _genome_from_dict(gd, Genome)

        # Policy will be rebuilt by runner/bank sync; keep placeholder None.
        a = Agent(
            agent_id=int(aid),
            genome=g,
            policy=None,
            energy=float(energy[i]) if i < len(energy) else 0.0,
            health=float(health[i]) if i < len(health) else 0.0,
            alive=bool(alive[i]) if i < len(alive) else True,
        )
        a.injury = float(injury[i]) if i < len(injury) else 0.0
        a.habitat_acclimation_state = (
            float(habitat_acclimation_state[i]) if i < len(habitat_acclimation_state) else 0.0
        )
        a.rest_streak = int(rest_streak[i]) if i < len(rest_streak) else 0
        a.repro_cooldown = int(repro_cooldown[i]) if i < len(repro_cooldown) else 0
        a.slot = int(slot[i]) if i < len(slot) else -1
        a.slot_token = int(slot_token[i]) if i < len(slot_token) else 0
        try:
            a.drives = np.asarray(drives[i], dtype=np.float32).copy()
        except Exception:
            pass
        try:
            a.chemistry = np.asarray(chemistry[i], dtype=np.float32).copy()
        except Exception:
            pass
        try:
            a.maintenance = np.asarray(maintenance[i], dtype=np.float32).copy()
        except Exception:
            pass

        # Stats
        try:
            s = stats_json[i]
            if isinstance(s, bytes):
                s = s.decode("utf-8")
            sd = json.loads(s) if s else {}
            st = AgentStats()
            for k, v in (sd or {}).items():
                try:
                    setattr(st, k, v)
                except Exception:
                    pass
            normalize_lineage_stats(st, int(aid), generation=0, tick=0)
            a.stats = st
        except Exception:
            normalize_lineage_stats(getattr(a, "stats", None), int(aid), generation=0, tick=0)
            pass

        new_agents[int(aid)] = a

    pop.agents = new_agents

    # Mark banks dirty so runner can rebuild policies/banks for these agents on next generation start.
    if hasattr(sim, "_mark_bank_dirty"):
        try:
            sim._mark_bank_dirty()
        except Exception:
            pass


def _genome_from_dict(gd: Dict[str, Any], GenomeCls: Any) -> Any:
    if hasattr(GenomeCls, "from_dict") and callable(getattr(GenomeCls, "from_dict")):
        try:
            return GenomeCls.from_dict(gd if isinstance(gd, dict) else {})
        except Exception:
            pass

    # Required fields in Genome constructor:
    # hidden_size, exploration_bias, aggression_bias, reproduction_threshold, mutation_rate,
    # action_temperature, use_gru, memory_dim, drive_inertia, drive_noise, drive_logit_gain, pred_lr
    # plus optional temperament priors.
    def _get(name: str, default: Any) -> Any:
        return gd.get(name, default)

    # Pick safe defaults consistent with typical init.
    hidden = int(_get("hidden_size", 64))
    exploration_bias = float(_get("exploration_bias", 0.0))
    aggression_bias = float(_get("aggression_bias", 0.0))
    reproduction_threshold = float(_get("reproduction_threshold", 0.55))
    mutation_rate = float(_get("mutation_rate", 0.02))
    action_temperature = float(_get("action_temperature", 1.0))
    memory_dim = int(_get("memory_dim", 0))
    use_gru = bool(_get("use_gru", memory_dim > 0))
    drive_inertia = float(_get("drive_inertia", 0.85))
    drive_noise = float(_get("drive_noise", 0.02))
    drive_logit_gain = float(_get("drive_logit_gain", 1.0))
    pred_lr = float(_get("pred_lr", 0.08))

    # Optional fields (Genome __post_init__ will clamp)
    kwargs = {}
    for k in [
        "drive_sp_hunger", "drive_sp_pain", "drive_sp_fatigue", "drive_sp_fear",
        "exp_prior_strength", "exp_prior_noise", "drive_birth_noise",
    ]:
        if k in gd:
            kwargs[k] = gd[k]

    try:
        return GenomeCls(
            hidden_size=hidden,
            exploration_bias=exploration_bias,
            aggression_bias=aggression_bias,
            reproduction_threshold=reproduction_threshold,
            mutation_rate=mutation_rate,
            action_temperature=action_temperature,
            use_gru=use_gru,
            memory_dim=memory_dim,
            drive_inertia=drive_inertia,
            drive_noise=drive_noise,
            drive_logit_gain=drive_logit_gain,
            pred_lr=pred_lr,
            **kwargs,
        )
    except Exception:
        # Last resort: construct with whatever the current Genome supports
        try:
            return GenomeCls(**gd)
        except Exception:
            return GenomeCls(
                hidden_size=hidden,
                exploration_bias=exploration_bias,
                aggression_bias=aggression_bias,
                reproduction_threshold=reproduction_threshold,
                mutation_rate=mutation_rate,
                action_temperature=action_temperature,
                use_gru=use_gru,
                memory_dim=memory_dim,
                drive_inertia=drive_inertia,
                drive_noise=drive_noise,
                drive_logit_gain=drive_logit_gain,
                pred_lr=pred_lr,
            )
