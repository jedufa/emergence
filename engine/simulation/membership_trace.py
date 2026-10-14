from __future__ import annotations

"""simulation/membership_trace.py

Phase 3 determinism support: membership write localization.

Purpose
-------
When a run diverges on "alive/present" membership (population/env/domain), the
fastest way to find the culprit is to locate the *first write* to membership
state:
  - agent.alive toggles
  - Population removes agents
  - Env agent_alive mask changes
  - Env add/remove calls

This module provides a minimal, opt-in tracer that can be enabled via env vars
without changing bundle schemas or architecture.

Important (Phase 3 Validation)
-----------------------------
To keep state dumps comparable across fresh vs resumed runs, trace records are
windowed per generation boundary:

  - buffer_cur  : membership writes occurring during the current generation window
  - buffer_prev : writes from the previous generation window

At the first call to set_trace_context() for a phase containing
"pre_start_generation" for a given gen, the tracer rotates buffers:

  buffer_prev <- buffer_cur
  buffer_cur  <- []

If the caller's state dump logic calls drain_writes() at
pre_start_generation_alive, it will receive the *previous-window* records,
which are stable for comparison.

Enable
------
  EIEL_MEMBERSHIP_TRACE=1  -> capture last N membership writes (stack + meta)
  EIEL_MEMBERSHIP_FAILFAST_GEN=17 -> raise as soon as a write occurs at gen

Runner integration
------------------
The SimulationRunner installs a hook (set_trace_context) at run start.
Instrumented write sites call note_write(...).
"""

import hashlib
import os
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


_TLS = threading.local()


@dataclass
class MembershipWrite:
    """A single membership mutation record."""

    ts_wall_s: float
    gen: int
    phase: str
    kind: str
    agent_id: int
    detail: str
    stack: List[str]


def _enabled() -> bool:
    """Return True if tracing is enabled."""
    try:
        return bool(int(os.environ.get("EIEL_MEMBERSHIP_TRACE", "0") or "0"))
    except Exception:
        return False


def set_trace_context(*, gen: int, phase: str) -> None:
    """Set the current generation/phase for subsequent membership write notes."""
    setattr(_TLS, "gen", int(gen))
    setattr(_TLS, "phase", str(phase))

    # Phase-3: window the trace by generation boundary.
    # Rotate at the first "pre_start_generation" context set for each gen.
    if _enabled():
        try:
            ph = str(phase or "")
            if "pre_start_generation" in ph:
                last_rot = getattr(_TLS, "_last_rotated_gen", None)
                if last_rot is None or int(last_rot) != int(gen):
                    rotate_window()
                    setattr(_TLS, "_last_rotated_gen", int(gen))
        except Exception:
            pass


def clear_trace_context() -> None:
    """Clear the thread-local trace context."""
    for k in ("gen", "phase"):
        if hasattr(_TLS, k):
            try:
                delattr(_TLS, k)
            except Exception:
                pass


def _get_context() -> Tuple[int, str]:
    gen = int(getattr(_TLS, "gen", -1) or -1)
    phase = str(getattr(_TLS, "phase", "") or "")
    return gen, phase


def note_write(*, kind: str, agent_id: int, detail: str = "") -> Optional[MembershipWrite]:
    """Record a membership write (best-effort).

    This function is intentionally safe: if anything goes wrong, it returns None.
    """
    if not _enabled():
        return None

    gen, phase = _get_context()

    # Collect a short stack; skip the last frame (this function).
    try:
        stack = traceback.format_stack(limit=25)[:-1]
    except Exception:
        stack = []

    rec = MembershipWrite(
        ts_wall_s=float(time.time()),
        gen=int(gen),
        phase=str(phase),
        kind=str(kind),
        agent_id=int(agent_id),
        detail=str(detail or ""),
        stack=stack,
    )

    buf: List[MembershipWrite] = getattr(_TLS, "buffer_cur", None)  # type: ignore[assignment]
    if buf is None:
        buf = []
        setattr(_TLS, "buffer_cur", buf)
    buf.append(rec)

    # Keep last N
    try:
        keep_n = int(os.environ.get("EIEL_MEMBERSHIP_TRACE_KEEP", "16") or "16")
        if keep_n > 0 and len(buf) > keep_n:
            del buf[:-keep_n]
    except Exception:
        pass

    # Optional fail-fast at a specific generation.
    try:
        ff = os.environ.get("EIEL_MEMBERSHIP_FAILFAST_GEN", "").strip()
        if ff:
            ff_gen = int(ff)
            if int(gen) == int(ff_gen):
                raise RuntimeError(
                    f"[EIEL_MEMBERSHIP_FAILFAST] write at gen={gen} phase={phase} kind={kind} agent_id={agent_id} detail={detail}"
                )
    except RuntimeError:
        raise
    except Exception:
        pass

    return rec


def rotate_window() -> None:
    """Rotate trace buffers at a generation boundary.

    Moves current-window writes to previous-window, and clears current.
    Best-effort and safe.
    """
    try:
        cur: List[MembershipWrite] = getattr(_TLS, "buffer_cur", None)  # type: ignore[assignment]
        if cur is None:
            cur = []
        setattr(_TLS, "buffer_prev", list(cur))
        try:
            cur.clear()
        except Exception:
            setattr(_TLS, "buffer_cur", [])
    except Exception:
        pass


def drain_writes(*, window: str = "prev") -> List[Dict[str, Any]]:
    """Return and clear buffered membership writes (JSON-serializable).

    window:
      - "prev" (default): previous generation window (stable for state-dump comparison)
      - "cur": current window
      - "both": concatenate prev + cur (clears both)
    """
    win = str(window or "prev").strip().lower()

    def _take(attr: str) -> List[MembershipWrite]:
        buf: List[MembershipWrite] = getattr(_TLS, attr, None)  # type: ignore[assignment]
        if not buf:
            return []
        out = list(buf)
        try:
            buf.clear()
        except Exception:
            setattr(_TLS, attr, [])
        return out

    recs: List[MembershipWrite] = []
    if win == "cur":
        recs = _take("buffer_cur")
    elif win == "both":
        recs = _take("buffer_prev") + _take("buffer_cur")
    else:
        recs = _take("buffer_prev")
        # Back-compat: if prev is empty (rotate not called yet), fall back to cur.
        if not recs:
            recs = _take("buffer_cur")

    if not recs:
        return []

    out: List[Dict[str, Any]] = []
    for r in recs:
        out.append(
            {
                "ts_wall_s": float(r.ts_wall_s),
                "gen": int(r.gen),
                "phase": str(r.phase),
                "kind": str(r.kind),
                "agent_id": int(r.agent_id),
                "detail": str(r.detail),
                "stack": list(r.stack),
            }
        )
    return out


def alive_signature(pop: Any, env: Any) -> Dict[str, Any]:
    """Compute a stable signature of alive membership for debugging.

    Returns dict with:
      - pop_alive_ids_sorted
      - env_alive_ids_sorted
      - intersection_ids_sorted
      - sha256 of each list
    """
    pop_ids: List[int] = []
    env_ids: List[int] = []
    try:
        pop_ids = [int(a.agent_id) for a in list(getattr(pop, "alive_agents")())]
    except Exception:
        pop_ids = []

    try:
        import numpy as _np

        env_alive = getattr(env, "agent_alive", None)
        if env_alive is not None:
            env_ids = _np.nonzero(env_alive)[0].astype(int).tolist()
    except Exception:
        env_ids = []

    pop_ids_sorted = sorted(set(pop_ids))
    env_ids_sorted = sorted(set(env_ids))
    inter = sorted(set(pop_ids_sorted).intersection(env_ids_sorted))

    def _h(lst: List[int]) -> str:
        b = ",".join(str(x) for x in lst).encode("utf-8")
        return hashlib.sha256(b).hexdigest()

    return {
        "pop_alive_ids_sorted": pop_ids_sorted,
        "env_alive_ids_sorted": env_ids_sorted,
        "intersection_ids_sorted": inter,
        "pop_sha256": _h(pop_ids_sorted),
        "env_sha256": _h(env_ids_sorted),
        "intersection_sha256": _h(inter),
        "pop_n": int(len(pop_ids_sorted)),
        "env_n": int(len(env_ids_sorted)),
        "intersection_n": int(len(inter)),
    }
