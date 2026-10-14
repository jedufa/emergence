"""engine.simulation.fdp_proofs

FDP Proof Helpers (Phase 3 / FDP v3)

Purpose
-------
Small, *read-only* helpers used by forensic instrumentation to:
- Build a targeted "missing state" whitelist snapshot for checkpoint save/apply.
- Compute stable sha256 digests for policy parameters and policy-bank tensors.

Hard constraints
----------------
- Must not draw randomness.
- Must be deterministic (no timestamps, no absolute paths).
- Must never crash production runs (best-effort; swallow exceptions).

This module is intentionally dependency-light and safe to import from hot paths.

Last modified: 2026-02-22
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# -----------------------------
# Small utilities
# -----------------------------

def _canon_f(x: Any) -> float:
    """Canonicalize a float value for deterministic JSON comparisons."""

    try:
        v = np.float32(float(x)).item()
    except Exception:
        v = float(x) if x is not None else 0.0
    try:
        return float(round(float(v), 8))
    except Exception:
        return float(v)


def _canon_i(x: Any) -> int:
    """Best-effort canonical int."""

    try:
        return int(x)
    except Exception:
        return 0


def _alive_agents_sorted(sim: Any) -> List[Any]:
    """Return alive agents sorted by agent_id (best-effort)."""

    agents: List[Any] = []

    # Preferred: Population API
    try:
        pop = getattr(sim, "pop", None)
        if pop is not None and hasattr(pop, "alive_agents"):
            agents = list(pop.alive_agents())
    except Exception:
        agents = []

    if not agents:
        # Fallback: pop.agents dict
        try:
            pop = getattr(sim, "pop", None)
            d = getattr(pop, "agents", None) if pop is not None else None
            if isinstance(d, dict):
                agents = [a for a in d.values() if bool(getattr(a, "alive", True))]
        except Exception:
            agents = []

    if not agents:
        # Fallback: sim.population list-like
        try:
            pop_list = getattr(sim, "population", None)
            if pop_list is not None:
                agents = [a for a in list(pop_list) if bool(getattr(a, "alive", True))]
        except Exception:
            agents = []

    try:
        agents.sort(key=lambda a: int(getattr(a, "agent_id", 0) or 0))
    except Exception:
        pass

    return agents


def _used_slots_from_agents(agents: List[Any]) -> List[int]:
    """Return sorted unique non-negative slots used by the given agents."""

    slots: List[int] = []
    for a in agents:
        try:
            s = int(getattr(a, "slot", -1))
        except Exception:
            s = -1
        if s >= 0:
            slots.append(int(s))
    return sorted(set(slots))


def _digest_ndarray(arr: np.ndarray) -> str:
    """Stable digest for ndarray values (includes dtype + shape)."""

    a = np.asarray(arr)
    if not a.flags.get("C_CONTIGUOUS", False):
        a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(b"|")
    h.update(",".join(str(x) for x in a.shape).encode("utf-8"))
    h.update(b"|")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


# -----------------------------
# Targeted "missing state" whitelist
# -----------------------------

def build_state_whitelist(sim: Any) -> Dict[str, Any]:
    """Build a targeted whitelist snapshot for missing-state hunts."""

    agents = _alive_agents_sorted(sim)

    rows: List[Dict[str, Any]] = []
    for a in agents:
        g = getattr(a, "genome", None)
        st = getattr(a, "stats", None)

        # Drives
        d0 = d1 = d2 = d3 = 0.0
        try:
            d = getattr(a, "drives", None)
            if d is not None:
                dd = np.asarray(d, dtype=np.float32).reshape(-1)
                if dd.size >= 4:
                    d0, d1, d2, d3 = float(dd[0]), float(dd[1]), float(dd[2]), float(dd[3])
        except Exception:
            pass

        rows.append(
            {
                "agent_id": _canon_i(getattr(a, "agent_id", 0)),
                "slot": _canon_i(getattr(a, "slot", -1)),
                "slot_token": _canon_i(getattr(a, "slot_token", 0)),
                "energy": _canon_f(getattr(a, "energy", 0.0)),
                "health": _canon_f(getattr(a, "health", 0.0)),
                "injury": _canon_f(getattr(a, "injury", 0.0)),
                "scar": _canon_f(getattr(a, "scar", 0.0)),
                "age": _canon_i(getattr(st, "survival_steps", 0) if st is not None else 0),
                "repro_cooldown": _canon_i(getattr(a, "repro_cooldown", 0)),
                "rest_streak": _canon_i(getattr(a, "rest_streak", 0)),
                "drives": [_canon_f(d0), _canon_f(d1), _canon_f(d2), _canon_f(d3)],
                "use_gru": bool(getattr(g, "use_gru", False)) if g is not None else False,
                "memory_dim": _canon_i(getattr(g, "memory_dim", 0) if g is not None else 0),
            }
        )

    # Runner drive mirrors (for the same live slots)
    slots = _used_slots_from_agents(agents)

    # FDP v3: runner drive mirror is derived/debug-only. Comparing raw slot-indexed arrays
    # makes the whitelist fragile to deterministic slot renumbering or capacity differences.
    # We therefore compare an order-insensitive digest of (hunger, pain, fatigue, fear, slot_token)
    # over the active slots list, while keeping the slot list for human debugging.
    drives_rows: List[Tuple[float, float, float, float, int]] = []

    try:
        ensure = getattr(sim, "_ensure_drive_arrays", None)
        if callable(ensure):
            ensure()

        dh = getattr(sim, "_drive_hunger", None)
        dp = getattr(sim, "_drive_pain", None)
        df = getattr(sim, "_drive_fatigue", None)
        de = getattr(sim, "_drive_fear", None)
        dtok = getattr(sim, "_drive_slot_token", None)

        for s in slots:
            try:
                h = _canon_f(float(dh[s])) if dh is not None else 0.0
                p = _canon_f(float(dp[s])) if dp is not None else 0.0
                f = _canon_f(float(df[s])) if df is not None else 0.0
                e = _canon_f(float(de[s])) if de is not None else 0.0
                t = _canon_i(int(dtok[s])) if dtok is not None else -1
            except Exception:
                h, p, f, e, t = 0.0, 0.0, 0.0, 0.0, -1
            drives_rows.append((h, p, f, e, t))
    except Exception:
        pass

    drives_rows_sorted = sorted(drives_rows)
    hsh = hashlib.sha256()
    hsh.update(json.dumps(drives_rows_sorted, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))

    drive_mirror: Dict[str, Any] = {
        "slots": [_canon_i(int(s)) for s in slots],
        "count": _canon_i(int(len(drives_rows_sorted))),
        "rows_multiset_sha256": hsh.hexdigest(),
    }
    return {
        "schema_version": "fdp_state_whitelist_v1",
        "n_alive": int(len(rows)),
        "alive": rows,
        "runner_drive_mirror": drive_mirror,
    }


def diff_state_whitelist(a: Dict[str, Any], b: Dict[str, Any], *, max_diffs: int = 64) -> Dict[str, Any]:
    """Compute a small, deterministic diff summary between two whitelist snapshots."""

    out: Dict[str, Any] = {
        "schema_version": "fdp_state_whitelist_diff_v1",
        "equal": bool(a == b),
        "diffs": [],
    }
    if a == b:
        return out

    diffs: List[Dict[str, Any]] = []

    # Top-level keys
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    for k in sorted(a_keys | b_keys):
        if k not in a:
            diffs.append({"path": k, "a": None, "b": "<missing>"})
        elif k not in b:
            diffs.append({"path": k, "a": "<missing>", "b": None})
        elif a.get(k) != b.get(k) and k not in ("alive",):
            diffs.append({"path": k, "a": a.get(k), "b": b.get(k)})
        if len(diffs) >= max_diffs:
            out["diffs"] = diffs[:max_diffs]
            return out

    # Alive rows (agent_id keyed)
    try:
        a_rows = {int(r.get("agent_id", -1)): r for r in (a.get("alive") or [])}
        b_rows = {int(r.get("agent_id", -1)): r for r in (b.get("alive") or [])}
        ids = sorted(set(a_rows.keys()) | set(b_rows.keys()))
        for aid in ids:
            ra = a_rows.get(aid)
            rb = b_rows.get(aid)
            if ra is None:
                diffs.append({"path": f"alive[{aid}]", "a": None, "b": "<missing>"})
                continue
            if rb is None:
                diffs.append({"path": f"alive[{aid}]", "a": "<missing>", "b": None})
                continue
            if ra != rb:
                keys = sorted(set(ra.keys()) | set(rb.keys()))
                for k in keys:
                    if ra.get(k) != rb.get(k):
                        diffs.append({"path": f"alive[{aid}].{k}", "a": ra.get(k), "b": rb.get(k)})
                        if len(diffs) >= max_diffs:
                            out["diffs"] = diffs[:max_diffs]
                            return out
    except Exception:
        pass

    out["diffs"] = diffs[:max_diffs]
    return out


# -----------------------------
# Policy identity (sha256 digests)
# -----------------------------

def policy_params_sha256(sim: Any) -> Optional[str]:
    """Return sha256 of all policy parameters (alive agents, sorted by agent_id)."""

    if torch is None:
        return None

    try:
        agents = _alive_agents_sorted(sim)
        h = hashlib.sha256()

        for a in agents:
            aid = int(getattr(a, "agent_id", 0) or 0)
            h.update(f"aid={aid};".encode("utf-8"))

            pol = getattr(a, "policy", None)
            if pol is None or not hasattr(pol, "state_dict"):
                h.update(b"policy=None;")
                continue

            try:
                sd = pol.state_dict()  # type: ignore[attr-defined]
            except Exception:
                h.update(b"state_dict_error;")
                continue

            if not isinstance(sd, dict):
                h.update(repr(sd).encode("utf-8"))
                continue

            for k in sorted(sd.keys()):
                h.update(str(k).encode("utf-8"))
                h.update(b"|")
                v = sd[k]
                if isinstance(v, torch.Tensor):
                    tt = v.detach()
                    if tt.is_cuda:
                        tt = tt.to("cpu")
                    tt = tt.contiguous()
                    h.update(str(tt.dtype).encode("utf-8"))
                    h.update(b"|")
                    h.update(",".join(str(x) for x in tt.shape).encode("utf-8"))
                    h.update(b"|")
                    try:
                        h.update(tt.numpy().tobytes(order="C"))
                    except Exception:
                        h.update(tt.float().numpy().tobytes(order="C"))
                else:
                    h.update(repr(v).encode("utf-8"))
                h.update(b";")

        return h.hexdigest()
    except Exception:
        return None


def _slots_by_policy_kind(agents: List[Any]) -> Tuple[List[int], List[int]]:
    """Split used slots into MLP-owned vs GRU-owned.

    Rationale
    ---------
    The sim maintains *both* MLP and GRU banks sized by global slot capacity.
    For a given alive slot, only one bank kind is semantically relevant:
    - If the agent's policy has a GRU module -> GRU bank rows are relevant
    - Otherwise -> MLP bank rows are relevant

    Forensic implication
    --------------------
    Hashing *all* bank tensors across *all* used slots can flag false mismatches
    caused by irrelevant cache rows (e.g. MLP rows for GRU-owned slots).
    This splitter keeps the determinism proof focused on semantically used rows.
    """
    mlp_slots: List[int] = []
    gru_slots: List[int] = []

    for a in agents:
        try:
            slot = int(getattr(a, "slot", -1))
        except Exception:
            slot = -1
        if slot < 0:
            continue

        try:
            pol = getattr(a, "policy", None)
            # Policy may be wrapped; fall back to attribute presence.
            is_gru = bool(pol is not None and getattr(pol, "gru", None) is not None)
        except Exception:
            is_gru = False

        if is_gru:
            gru_slots.append(slot)
        else:
            mlp_slots.append(slot)

    mlp_slots = sorted(set(mlp_slots))
    gru_slots = sorted(set(gru_slots))
    return mlp_slots, gru_slots


def bank_weights_sha256_full(sim: Any) -> Optional[str]:
    """Legacy: sha256 digest of policy-bank tensors over *all* used slots.

    Kept for debugging/forensics. Not used as the primary determinism proof
    because it can include irrelevant cache rows for the other policy kind.
    """
    if torch is None:
        return None

    agents = _alive_agents_sorted(sim)
    slots = _used_slots_from_agents(agents)

    try:
        h = hashlib.sha256()
        h.update(json.dumps(slots, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        h.update(b"\n")

        idx = None
        if slots:
            try:
                idx = torch.as_tensor(slots, dtype=torch.long, device=getattr(sim, "_bank_device", "cpu"))
            except Exception:
                idx = torch.as_tensor(slots, dtype=torch.long)

        def _add(name: str, t: Any) -> None:
            h.update(name.encode("utf-8"))
            h.update(b":")
            if t is None or not isinstance(t, torch.Tensor):
                h.update(b"<none>\n")
                return
            # If there are no relevant slots, do NOT hash full-bank cache tensors.
            # This avoids false mismatches when only one policy kind is present.
            if idx is None or (hasattr(idx, "numel") and int(idx.numel()) == 0):
                h.update(b"<no_slots>\n")
                return
            tt = t.detach()
            try:
                if idx is not None and tt.ndim >= 1 and int(tt.shape[0]) >= int(idx.numel()):
                    _idx = idx.to(tt.device) if tt.device != idx.device else idx
                    tt = tt.index_select(0, _idx)
            except Exception:
                pass
            if tt.is_cuda:
                tt = tt.to("cpu")
            tt = tt.contiguous()
            h.update(str(tt.dtype).encode("utf-8"))
            h.update(b"|")
            h.update(",".join(str(x) for x in tt.shape).encode("utf-8"))
            h.update(b"|")
            try:
                h.update(tt.numpy().tobytes(order="C"))
            except Exception:
                h.update(tt.float().numpy().tobytes(order="C"))
            h.update(b"\n")

        # MLP bank
        _add("mlp_pack", getattr(sim, "_mlp_pack", None))
        _add("W1", getattr(sim, "_W1", None))
        _add("b1", getattr(sim, "_b1", None))
        _add("W2", getattr(sim, "_W2", None))
        _add("b2", getattr(sim, "_b2", None))
        _add("W3", getattr(sim, "_W3", None))
        _add("b3", getattr(sim, "_b3", None))

        # GRU bank
        _add("gru_pack", getattr(sim, "_gru_pack", None))
        _add("encW", getattr(sim, "_encW", None))
        _add("encb", getattr(sim, "_encb", None))
        _add("Wih", getattr(sim, "_Wih", None))
        _add("bih", getattr(sim, "_bih", None))
        _add("Whh", getattr(sim, "_Whh", None))
        _add("bhh", getattr(sim, "_bhh", None))
        _add("headW", getattr(sim, "_headW", None))
        _add("headb", getattr(sim, "_headb", None))
        _add("Hmem", getattr(sim, "_Hmem", None))

        return h.hexdigest()
    except Exception:
        return None


def bank_weights_sha256(sim: Any) -> Optional[str]:
    """Primary: sha256 digest of policy-bank tensors over semantically used rows.

    - MLP tensors are hashed only for MLP-owned slots.
    - GRU tensors are hashed only for GRU-owned slots.

    This prevents false mismatches from irrelevant cache rows.
    """

    if torch is None:
        return None

    agents = _alive_agents_sorted(sim)
    mlp_slots, gru_slots = _slots_by_policy_kind(agents)

    try:
        h = hashlib.sha256()
        h.update(json.dumps({"mlp": mlp_slots, "gru": gru_slots}, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        h.update(b"\n")

        def _mk_idx(slots: List[int]) -> Any:
            if not slots:
                return None
            try:
                return torch.as_tensor(slots, dtype=torch.long, device=getattr(sim, "_bank_device", "cpu"))
            except Exception:
                return torch.as_tensor(slots, dtype=torch.long)

        idx_mlp = _mk_idx(mlp_slots)
        idx_gru = _mk_idx(gru_slots)

        def _add(name: str, t: Any, idx: Any) -> None:
            h.update(name.encode("utf-8"))
            h.update(b":")
            if t is None or not isinstance(t, torch.Tensor):
                h.update(b"<none>\n")
                return
            # If there are no relevant slots for this bank kind, do NOT hash full-bank cache tensors.
            # (Example: population is 100% GRU => MLP slots empty.)
            if idx is None or (hasattr(idx, "numel") and int(idx.numel()) == 0):
                h.update(b"<no_slots>\n")
                return
            tt = t.detach()
            try:
                if idx is not None and tt.ndim >= 1 and int(tt.shape[0]) >= int(idx.numel()):
                    _idx = idx.to(tt.device) if tt.device != idx.device else idx
                    tt = tt.index_select(0, _idx)
            except Exception:
                pass
            if tt.is_cuda:
                tt = tt.to("cpu")
            tt = tt.contiguous()
            h.update(str(tt.dtype).encode("utf-8"))
            h.update(b"|")
            h.update(",".join(str(x) for x in tt.shape).encode("utf-8"))
            h.update(b"|")
            try:
                h.update(tt.numpy().tobytes(order="C"))
            except Exception:
                h.update(tt.float().numpy().tobytes(order="C"))
            h.update(b"\n")

        # MLP bank (MLP slots only)
        _add("mlp_pack", getattr(sim, "_mlp_pack", None), idx_mlp)
        _add("W1", getattr(sim, "_W1", None), idx_mlp)
        _add("b1", getattr(sim, "_b1", None), idx_mlp)
        _add("W2", getattr(sim, "_W2", None), idx_mlp)
        _add("b2", getattr(sim, "_b2", None), idx_mlp)
        _add("W3", getattr(sim, "_W3", None), idx_mlp)
        _add("b3", getattr(sim, "_b3", None), idx_mlp)

        # GRU bank (GRU slots only)
        _add("gru_pack", getattr(sim, "_gru_pack", None), idx_gru)
        _add("encW", getattr(sim, "_encW", None), idx_gru)
        _add("encb", getattr(sim, "_encb", None), idx_gru)
        _add("Wih", getattr(sim, "_Wih", None), idx_gru)
        _add("bih", getattr(sim, "_bih", None), idx_gru)
        _add("Whh", getattr(sim, "_Whh", None), idx_gru)
        _add("bhh", getattr(sim, "_bhh", None), idx_gru)
        _add("headW", getattr(sim, "_headW", None), idx_gru)
        _add("headb", getattr(sim, "_headb", None), idx_gru)
        _add("Hmem", getattr(sim, "_Hmem", None), idx_gru)

        return h.hexdigest()
    except Exception:
        return None


def collect_bank_meta(sim: Any) -> Dict[str, Any]:
    """Collect small bank meta flags (informational)."""

    out: Dict[str, Any] = {
        "schema_version": "fdp_bank_meta_v1",
        "bank_dirty_mlp": bool(getattr(sim, "_bank_dirty_mlp", False)),
        "bank_dirty_gru": bool(getattr(sim, "_bank_dirty_gru", False)),
        "gru_inited": bool(getattr(sim, "_gru_inited", False)),
        "bank_device": str(getattr(sim, "_bank_device", getattr(sim, "bank_device", ""))),
        "bank_dtype": str(getattr(sim, "_bank_dtype", "")),
    }

    try:
        mp = getattr(sim, "_mlp_pack", None)
        if mp is not None and torch is not None and isinstance(mp, torch.Tensor):
            out["mlp_cap"] = int(mp.shape[0])
    except Exception:
        pass

    try:
        gp = getattr(sim, "_gru_pack", None)
        if gp is not None and torch is not None and isinstance(gp, torch.Tensor):
            out["gru_cap"] = int(gp.shape[0])
    except Exception:
        pass

    return out


def collect_bank_tokens(sim: Any, *, head: int = 16) -> Dict[str, Any]:
    """Collect bank token values for alive agents (sorted)."""

    agents = _alive_agents_sorted(sim)

    rows: List[Dict[str, Any]] = []
    for a in agents:
        rows.append(
            {
                "agent_id": _canon_i(getattr(a, "agent_id", 0)),
                "slot": _canon_i(getattr(a, "slot", -1)),
                "slot_token": _canon_i(getattr(a, "slot_token", 0)),
                "bank_token": _canon_i(getattr(a, "_bank_token", -999999)),
                "gru_bank_token": _canon_i(getattr(a, "_gru_bank_token", -999999)),
            }
        )

    bank_tok = np.asarray([int(r["bank_token"]) for r in rows], dtype=np.int32)
    gru_tok = np.asarray([int(r["gru_bank_token"]) for r in rows], dtype=np.int32)

    return {
        "schema_version": "fdp_bank_tokens_v1",
        "n": int(len(rows)),
        "bank_token_sha256": _digest_ndarray(bank_tok),
        "gru_bank_token_sha256": _digest_ndarray(gru_tok),
        "head": rows[: int(head)],
    }
