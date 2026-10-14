"""
simulation/determinism_probe.py

Determinism Probe (Phase 3 / FDP v3)

Purpose
-------
Provide a *read-only* snapshot of all relevant RNG states at a specific stage
without consuming any RNG.

This is a forensic tool to localize resume divergence:
- If fresh vs resume differ at the *same* stage, a state restore is missing or a
  resume-only path consumes RNG before the stage.
- If they match at stage boundaries but diverge later, the leak is in step order
  or an uncheckpointed state component that affects deterministic computation.

Hard constraints
----------------
- Must not draw randomness (no random.random / np.random.random / torch.rand).
- Best-effort only: must never crash production runs.
- Output must be deterministic and contain no absolute paths or timestamps.

Activation
----------
Set environment variable:
  EIEL_DETERMINISM_PROBE=1

Output
------
Writes JSON Lines to:
  <run_dir>/debug/determinism_probe.jsonl
Each line is a single JSON object with stable, hash-only fields.

Last modified: 2026-02-22
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _sha256_hex(data: bytes) -> str:
    """Return sha256 hex digest for `data`."""
    return hashlib.sha256(data).hexdigest()


def _safe_bytes(obj: Any) -> bytes:
    """Best-effort stable bytes for hashing.

    Notes
    -----
    - For dict/list: canonical-ish JSON (sorted keys).
    - For bytes: passthrough.
    - For other objects: UTF-8 of repr().
    """
    try:
        if isinstance(obj, (bytes, bytearray)):
            return bytes(obj)
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return repr(obj).encode("utf-8")
    except Exception:
        return b""


def _hash_python_random_state() -> Optional[str]:
    """Hash Python `random` global state (no RNG draws)."""
    try:
        import pickle  # noqa: WPS433 (std)
        import random as _random  # noqa: WPS433 (std)

        st = _random.getstate()
        return _sha256_hex(pickle.dumps(st, protocol=pickle.HIGHEST_PROTOCOL))
    except Exception:
        return None


def _hash_numpy_global_randomstate() -> Optional[str]:
    """Hash NumPy legacy global RandomState (np.random.*) state (no RNG draws)."""
    try:
        st = np.random.get_state()
        algo = str(st[0]).encode("utf-8")
        keys = np.asarray(st[1], dtype=np.uint32).tobytes(order="C")
        pos = int(st[2]).to_bytes(8, "little", signed=True)
        has_gauss = int(st[3]).to_bytes(8, "little", signed=True)
        cached = np.float64(st[4]).tobytes()
        return _sha256_hex(algo + b"|" + keys + b"|" + pos + b"|" + has_gauss + b"|" + cached)
    except Exception:
        return None


def _hash_np_generator_state(gen: Any) -> Optional[str]:
    """Hash NumPy Generator bit_generator.state (no RNG draws)."""
    try:
        bg = getattr(gen, "bit_generator", None)
        st = getattr(bg, "state", None) if bg is not None else None
        if st is None:
            return None
        return _sha256_hex(_safe_bytes(st))
    except Exception:
        return None

def _hash_np_array(a: Any) -> Optional[str]:
    """Best-effort hash for numpy-like arrays.

    Includes dtype + shape + raw bytes so that silent dtype/shape drift is detected.
    Must not draw RNG.
    """
    try:
        if a is None:
            return None
        arr = np.asarray(a)
        arr_c = np.ascontiguousarray(arr)
        h = hashlib.sha256()
        h.update(arr_c.dtype.str.encode("utf-8"))
        h.update(b"|")
        h.update(str(arr_c.shape).encode("utf-8"))
        h.update(b"|")
        h.update(arr_c.tobytes())
        return h.hexdigest()
    except Exception:
        return None


def _iter_population_agents(pop: Any):
    """Return iterable of agent objects from population container shapes."""
    if pop is None:
        return []
    agents = getattr(pop, "agents", None)
    if isinstance(agents, dict):
        return list(agents.values())
    if agents is not None:
        return agents
    if isinstance(pop, dict):
        return list(pop.values())
    return pop





def _hash_pop_scalar(sim: Any, attr: str) -> str | None:
    """Hash a per-agent scalar attribute for alive agents, sorted by agent_id.

    This is used to detect resume-only drift in agent scalar state that can affect
    env stepping deterministically (e.g., movement blocked by energy thresholds).
    """
    try:
        pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
        agents = _iter_population_agents(pop)
        if agents is None:
            return None
        alive = []
        for a in agents:
            try:
                if bool(getattr(a, "alive", True)):
                    alive.append(a)
            except Exception:
                continue
        alive.sort(key=lambda a: int(getattr(a, "agent_id", 0) or 0))
        vals = np.fromiter((float(getattr(a, attr, 0.0) or 0.0) for a in alive), dtype=np.float32, count=len(alive))
        return _hash_np_array(vals)
    except Exception:
        return None


def _hash_pop_stats_scalar(sim: Any, attr: str) -> str | None:
    """Hash a per-agent stats scalar attribute for alive agents, sorted by agent_id."""
    try:
        pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
        agents = _iter_population_agents(pop)
        if agents is None:
            return None
        alive = []
        for a in agents:
            try:
                if bool(getattr(a, "alive", True)):
                    alive.append(a)
            except Exception:
                continue
        alive.sort(key=lambda a: int(getattr(a, "agent_id", 0) or 0))
        vals = np.fromiter(
            (
                float(getattr(getattr(a, "stats", None), attr, 0.0) or 0.0)
                for a in alive
            ),
            dtype=np.float32,
            count=len(alive),
        )
        return _hash_np_array(vals)
    except Exception:
        return None


def _hash_pop_int(sim: Any, attr: str) -> str | None:
    """Hash a per-agent integer attribute for alive agents, sorted by agent_id."""
    try:
        pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
        agents = _iter_population_agents(pop)
        if agents is None:
            return None
        alive = []
        for a in agents:
            try:
                if bool(getattr(a, "alive", True)):
                    alive.append(a)
            except Exception:
                continue
        alive.sort(key=lambda a: int(getattr(a, "agent_id", 0) or 0))
        vals = np.fromiter((int(getattr(a, attr, 0) or 0) for a in alive), dtype=np.int32, count=len(alive))
        return _hash_np_array(vals)
    except Exception:
        return None


def _hash_pop_agent_ids(sim: Any, *, sort_by_agent_id: bool, iter_order: bool = False) -> str | None:
    """Hash alive agent_ids.

    Modes
    -----
    - sort_by_agent_id=True: ids sorted numerically (set equality signal).
    - iter_order=True: ids in population iteration order (order-sensitive signal).
      If both are False, defaults to sorted.
    """
    try:
        pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
        agents = _iter_population_agents(pop)
        if agents is None:
            return None

        alive = []
        for a in agents:
            try:
                if bool(getattr(a, "alive", True)):
                    alive.append(a)
            except Exception:
                continue

        if iter_order:
            ids = np.fromiter((int(getattr(a, "agent_id", 0) or 0) for a in alive), dtype=np.int64)
        else:
            ids_list = [int(getattr(a, "agent_id", 0) or 0) for a in alive]
            if sort_by_agent_id or True:
                ids_list.sort()
            ids = np.asarray(ids_list, dtype=np.int64)

        return _hash_np_array(ids)
    except Exception:
        return None



def _hash_pop_drive(sim: Any, idx: int) -> str | None:
    """Hash a per-agent drive component for alive agents, sorted by agent_id."""
    try:
        pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
        agents = _iter_population_agents(pop)
        if agents is None:
            return None
        alive = []
        for a in agents:
            try:
                if bool(getattr(a, "alive", True)):
                    alive.append(a)
            except Exception:
                continue
        alive.sort(key=lambda a: int(getattr(a, "agent_id", 0) or 0))

        def _get_drive(a) -> float:
            d = getattr(a, "drives", None)
            if d is None:
                return 0.0
            try:
                if getattr(d, "shape", None) is not None and int(d.shape[0]) > int(idx):
                    return float(d[idx])
            except Exception:
                pass
            try:
                if isinstance(d, (list, tuple)) and len(d) > int(idx):
                    return float(d[idx])
            except Exception:
                pass
            return 0.0

        vals = np.fromiter((_get_drive(a) for a in alive), dtype=np.float32, count=len(alive))
        return _hash_np_array(vals)
    except Exception:
        return None


def _hash_pop_chemistry(sim: Any, idx: int) -> str | None:
    """Hash a per-agent chemistry component for alive agents, sorted by agent_id."""
    try:
        pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
        agents = _iter_population_agents(pop)
        if agents is None:
            return None
        alive = []
        for a in agents:
            try:
                if bool(getattr(a, "alive", True)):
                    alive.append(a)
            except Exception:
                continue
        alive.sort(key=lambda a: int(getattr(a, "agent_id", 0) or 0))

        def _get_chem(a) -> float:
            c = getattr(a, "chemistry", None)
            if c is None:
                return 0.0
            try:
                arr = np.asarray(c, dtype=np.float32).reshape(-1)
                if int(arr.shape[0]) > int(idx):
                    return float(arr[int(idx)])
            except Exception:
                pass
            return 0.0

        vals = np.fromiter((_get_chem(a) for a in alive), dtype=np.float32, count=len(alive))
        return _hash_np_array(vals)
    except Exception:
        return None


def _hash_pop_maintenance(sim: Any, idx: int) -> str | None:
    """Hash a per-agent maintenance component for alive agents, sorted by agent_id."""
    try:
        pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
        agents = _iter_population_agents(pop)
        if agents is None:
            return None
        alive = []
        for a in agents:
            try:
                if bool(getattr(a, "alive", True)):
                    alive.append(a)
            except Exception:
                continue
        alive.sort(key=lambda a: int(getattr(a, "agent_id", 0) or 0))

        def _get_maint(a) -> float:
            m = getattr(a, "maintenance", None)
            if m is None:
                return 0.0
            try:
                arr = np.asarray(m, dtype=np.float32).reshape(-1)
                if int(arr.shape[0]) > int(idx):
                    return float(arr[int(idx)])
            except Exception:
                pass
            return 0.0

        vals = np.fromiter((_get_maint(a) for a in alive), dtype=np.float32, count=len(alive))
        return _hash_np_array(vals)
    except Exception:
        return None


def _hash_rng_manager_snapshot(sim: Any) -> Optional[str]:
    """Hash RNGManager snapshot dict (no RNG draws).

    Notes
    -----
    We avoid importing canonical_json_bytes here to keep dependencies minimal.
    Sorted-keys JSON is good enough as a *hash* signal for forensic localization.
    """
    try:
        rm = getattr(sim, "rng_manager", None)
        if rm is None:
            return None
        if hasattr(rm, "capture_snapshot"):
            snap = rm.capture_snapshot(generation_index=-999)  # generation_index is metadata only
        elif hasattr(rm, "snapshot_dict"):
            snap = rm.snapshot_dict()
        else:
            return None
        if not isinstance(snap, dict):
            return _sha256_hex(_safe_bytes(snap))
        return _sha256_hex(json.dumps(snap, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    except Exception:
        return None


def _hash_torch_cpu_rng_state() -> Optional[str]:
    """Hash torch CPU RNG state (no RNG draws)."""
    if torch is None:
        return None
    try:
        st = torch.get_rng_state()
        b = bytes(st.detach().to("cpu").numpy().tobytes())
        return _sha256_hex(b)
    except Exception:
        return None



def _hash_tick_actions_pairs(sim: Any) -> Optional[str]:
    """Hash (agent_id, action) pairs for the *current* tick if available.

    This is populated by runner_ticks_loop right after action selection.
    We only report it when sim.tick matches the stored tick to avoid leaking
    previous-tick actions into tick-start probes.
    """
    try:
        cur_tick = int(getattr(sim, "_eiel_fdp_cur_tick", getattr(sim, "tick", -1)))
        last_tick = getattr(sim, "_eiel_fdp_last_actions_tick", None)
        if last_tick is None or int(last_tick) != cur_tick:
            return None

        agent_ids = getattr(sim, "_eiel_fdp_last_actions_agent_ids", None)
        actions = getattr(sim, "_eiel_fdp_last_actions_actions", None)
        if agent_ids is None or actions is None:
            return None

        aid_np = np.asarray(agent_ids, dtype=np.int64).reshape(-1)
        act_np = np.asarray(actions, dtype=np.int64).reshape(-1)
        if aid_np.shape[0] != act_np.shape[0]:
            return None

        h = hashlib.sha256()
        h.update(aid_np.tobytes(order="C"))
        h.update(b"|")
        h.update(act_np.tobytes(order="C"))
        return h.hexdigest()
    except Exception:
        return None


def _hash_tick_actions_hist(sim: Any) -> Optional[str]:
    """Order-insensitive hash of action histogram for current tick (if available)."""
    try:
        cur_tick = int(getattr(sim, "_eiel_fdp_cur_tick", getattr(sim, "tick", -1)))
        last_tick = getattr(sim, "_eiel_fdp_last_actions_tick", None)
        if last_tick is None or int(last_tick) != cur_tick:
            return None

        actions = getattr(sim, "_eiel_fdp_last_actions_actions", None)
        if actions is None:
            return None

        act_np = np.asarray(actions, dtype=np.int64).reshape(-1)
        if act_np.size == 0:
            return hashlib.sha256(b"").hexdigest()

        vals, counts = np.unique(act_np, return_counts=True)
        pairs = sorted(zip(vals.tolist(), counts.tolist()), key=lambda kv: int(kv[0]))
        payload = (";".join(f"{int(v)}:{int(c)}" for v, c in pairs)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
    except Exception:
        return None



def _hash_tick_actions_by_agent_id(sim: Any) -> Optional[str]:
    """Order-insensitive hash of (agent_id -> action) mapping for current tick (if available)."""
    try:
        cur_tick = int(getattr(sim, "_eiel_fdp_cur_tick", getattr(sim, "tick", -1)))
        last_tick = getattr(sim, "_eiel_fdp_last_actions_tick", None)
        if last_tick is None or int(last_tick) != cur_tick:
            return None

        agent_ids = getattr(sim, "_eiel_fdp_last_actions_agent_ids", None)
        actions = getattr(sim, "_eiel_fdp_last_actions_actions", None)
        if agent_ids is None or actions is None:
            return None

        aid_np = np.asarray(agent_ids, dtype=np.int64).reshape(-1)
        act_np = np.asarray(actions, dtype=np.int64).reshape(-1)
        if aid_np.shape[0] != act_np.shape[0]:
            return None

        if aid_np.size == 0:
            return hashlib.sha256(b"").hexdigest()

        order = np.argsort(aid_np, kind="mergesort")
        aid_s = aid_np[order]
        act_s = act_np[order]

        h = hashlib.sha256()
        h.update(aid_s.tobytes(order="C"))
        h.update(b"|")
        h.update(act_s.tobytes(order="C"))
        return h.hexdigest()
    except Exception:
        return None

def maybe_write_probe(sim: Any, *, generation_index: int, tag: str) -> None:
    """Write a single probe record if enabled.

    Parameters
    ----------
    sim:
        Simulation runner (must have run_dir, pop, np_rng, rng_manager optionally).
    generation_index:
        Absolute generation index for this probe point.
    tag:
        Stage identifier (string). Keep stable across runs.
    """
    if os.environ.get("EIEL_DETERMINISM_PROBE", "0").strip() != "1":
        return

    run_dir = getattr(sim, "run_dir", None) or getattr(sim, "run_dir_path", None) or ""
    if not run_dir:
        return

    try:
        dbg = os.path.join(str(run_dir), "debug")
        os.makedirs(dbg, exist_ok=True)
        out_path = os.path.join(dbg, "determinism_probe.jsonl")
    except Exception:
        return

    # FDP v3+: action digests are only meaningful for tick-tagged probes.
    include_tick_actions = str(tag).startswith('tick')
    _tap = _hash_tick_actions_pairs(sim) if include_tick_actions else None
    _tah = _hash_tick_actions_hist(sim) if include_tick_actions else None
    _tab = _hash_tick_actions_by_agent_id(sim) if include_tick_actions else None
    rec: Dict[str, Any] = {
        "schema_version": "determinism_probe_v1",
        "generation_index": int(generation_index),
        "tag": str(tag),
        "hashes": {
            "python_random": _hash_python_random_state(),
            "numpy_global_randomstate": _hash_numpy_global_randomstate(),
            "torch_cpu_rng": _hash_torch_cpu_rng_state(),
            "rng_manager_snapshot": _hash_rng_manager_snapshot(sim),
            "runner_np_rng": _hash_np_generator_state(getattr(sim, "np_rng", None)),
            "tick_actions_pairs_sha256": _tap,
            "tick_actions_hist_sha256": _tah,
            "tick_actions_by_agent_id_sha256": _tab,
            # env arrays (helps locate earliest non-RNG divergence)
            "env_grid": _hash_np_array(getattr(getattr(sim, "env", None), "grid", None)),
            "env_agent_grid": _hash_np_array(getattr(getattr(sim, "env", None), "agent_grid", None)),
            "env_biome_grid": _hash_np_array(getattr(getattr(sim, "env", None), "biome_grid", None)),
            "env_agent_pos_y": _hash_np_array(getattr(getattr(sim, "env", None), "agent_pos_y", None)),
            "env_agent_pos_x": _hash_np_array(getattr(getattr(sim, "env", None), "agent_pos_x", None)),
            "env_agent_alive": _hash_np_array(getattr(getattr(sim, "env", None), "agent_alive", None)),
                    # pop scalar arrays (alive agents, sorted by agent_id)
            "pop_alive_energy": _hash_pop_scalar(sim, "energy"),
            "pop_alive_health": _hash_pop_scalar(sim, "health"),
            "pop_alive_injury": _hash_pop_scalar(sim, "injury"),
            "pop_alive_scar": _hash_pop_scalar(sim, "scar"),
            "pop_alive_agent_id_sorted": _hash_pop_agent_ids(sim, sort_by_agent_id=True),
            "pop_alive_agent_id_iter": _hash_pop_agent_ids(sim, sort_by_agent_id=False, iter_order=True),
            "pop_alive_slot": _hash_pop_int(sim, "slot"),
            "pop_alive_slot_token": _hash_pop_int(sim, "slot_token"),
            "pop_alive_drive_hunger": _hash_pop_drive(sim, 0),
            "pop_alive_drive_pain": _hash_pop_drive(sim, 1),
            "pop_alive_drive_fatigue": _hash_pop_drive(sim, 2),
            "pop_alive_drive_fear": _hash_pop_drive(sim, 3),
            "pop_alive_chem_energy_store": _hash_pop_chemistry(sim, 0),
            "pop_alive_chem_repair_reserve": _hash_pop_chemistry(sim, 1),
            "pop_alive_chem_acute_damage": _hash_pop_chemistry(sim, 2),
            "pop_alive_chem_chronic_damage": _hash_pop_chemistry(sim, 3),
            "pop_alive_chem_regulator_1": _hash_pop_chemistry(sim, 4),
            "pop_alive_chem_regulator_2": _hash_pop_chemistry(sim, 5),
            "pop_alive_maint_pending_strength": _hash_pop_maintenance(sim, 0),
            "pop_alive_maint_due_tick": _hash_pop_maintenance(sim, 1),
            "pop_alive_maint_recent_load": _hash_pop_maintenance(sim, 2),
            "pop_alive_maint_last_schedule_tick": _hash_pop_maintenance(sim, 3),
            "pop_alive_stats_chem_action_cost_mult": _hash_pop_stats_scalar(sim, "chemistry_action_cost_mult"),
            "pop_alive_stats_chem_repro_viability": _hash_pop_stats_scalar(sim, "chemistry_repro_viability"),
            "pop_alive_stats_chem_obs_fidelity": _hash_pop_stats_scalar(sim, "chemistry_observation_fidelity"),
            "pop_alive_stats_structural_locality_cost_total": _hash_pop_stats_scalar(sim, "structural_locality_cost_total"),
            "pop_alive_stats_structural_locality_mean_distance": _hash_pop_stats_scalar(sim, "structural_locality_mean_distance"),
            "pop_alive_stats_structural_locality_max_distance": _hash_pop_stats_scalar(sim, "structural_locality_max_distance"),
        },
    }

    # population np_rng (if present)
    try:
        pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
        rec["hashes"]["pop_np_rng"] = _hash_np_generator_state(getattr(pop, "np_rng", None) if pop is not None else None)
    except Exception:
        rec["hashes"]["pop_np_rng"] = None

    # population "rng" (some implementations keep a separate RNG object)
    try:
        pop = getattr(sim, "pop", None) or getattr(sim, "population", None)
        rec["hashes"]["pop_rng"] = _hash_np_generator_state(getattr(pop, "rng", None) if pop is not None else None)
    except Exception:
        rec["hashes"]["pop_rng"] = None

    # FDP (Phase 3): policy identity proof at resume_apply_post and tick000.start.
    # This is intentionally gated to avoid heavy hashing at every probe.
    try:
        if str(tag) in ('resume_apply_post', 'tick000.start'):
            from simulation.fdp_proofs import bank_weights_sha256 as _fdp_bank_sha
            from simulation.fdp_proofs import bank_weights_sha256_full as _fdp_bank_sha_full
            from simulation.fdp_proofs import policy_params_sha256 as _fdp_policy_sha
            from simulation.fdp_proofs import collect_bank_meta as _fdp_bank_meta
            from simulation.fdp_proofs import collect_bank_tokens as _fdp_bank_tokens

            rec['hashes']['policy_params_sha256'] = _fdp_policy_sha(sim)
            rec['hashes']['bank_weights_sha256'] = _fdp_bank_sha(sim)
            rec['hashes']['bank_weights_full_sha256'] = _fdp_bank_sha_full(sim)
            rec['fdp_bank_meta'] = _fdp_bank_meta(sim)
            rec['fdp_bank_tokens'] = _fdp_bank_tokens(sim, head=32)
    except Exception:
        pass

    # best-effort write (append-only)
    try:
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
            f.write("\n")
    except Exception:
        return
