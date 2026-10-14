from __future__ import annotations

"""simulation/rng_manager.py

Phase 1 determinism substrate: RNGManager.
Last modified: 2026-02-20 06:30:00 Europe/Vienna

Purpose
-------
Provide a *single authority* for RNG creation and routing.

Why this exists (in 12-year-old terms)
-------------------------------------
If different parts of the program secretly create their own random generators,
then two "same seed" runs can still diverge.
So we force everything to ask *one* manager for randomness.

Design notes
------------
- Minimal implementation to remove RNG islands (env/pop/shock) and support
  checkpoint snapshot hooks.
- Stream seed derivation is stable: hash(global_seed, stream_name, kind).
- No torch import here (torch RNG capture/restore is handled elsewhere).

Public API
----------
- RNGManager(global_seed)
- .policy_py / .policy_np
- .env_np
- .evolution_py / .evolution_np
- .metrics_np
- capture_snapshot(generation_index=None) -> dict
- apply_snapshot_dict(snapshot_dict) -> None
"""

from dataclasses import dataclass
from collections import defaultdict
import hashlib
import json
import os
import random
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np



# ============================================================
# Phase 3 forensic tracing: evolution_py instance tracing
# ============================================================
#
# This tracer is intentionally *instance-based* (not class monkeypatching).
# Some NumPy / CPython RNG classes are immutable C-extension types and cannot
# be safely monkeypatched in-place (see failure: cannot set attribute of immutable type).
#
# The tracer is designed to locate the FIRST divergence in RNG consumption order
# between a fresh run and a resume-from-checkpoint run.
#
# Controls (env vars)
# -------------------
# EIEL_EVOPY_TRACE=1         -> enable printing trace lines
# EIEL_EVOPY_TRACE_N=500     -> maximum number of calls to emit (per process)
#
# Output format:
#   [EVOPY_CALL] <json>
#
# JSON fields:
#   call_idx, method, pre_sha, post_sha, caller, gen, tick, stage
# ============================================================

_EVOPY_TRACE_ENABLED = os.environ.get("EIEL_EVOPY_TRACE", "0").strip() == "1"

# ============================================================
# FDP v3 (VALIDATION): evolution_py forensic ledger
#
# Strict rule: the ledger must ONLY be enabled explicitly.
# Do not silently enable it based on other debug knobs.
# ============================================================
# Ledger enablement rules (FDP v3)
#
# Primary knob: EIEL_EVOPY_LEDGER=1
# Secondary auto-enable: when state dumps or determinism debug harnesses are enabled.
# This is required so the standard Phase 3 harness emits ledgers without users
# remembering yet another env var.
# ============================================================

def _env_truthy(key: str) -> bool:
    try:
        v = os.environ.get(key)
        if v is None:
            return False
        return str(v).strip().lower() not in ("", "0", "false", "no")
    except Exception:
        return False


def _evopy_ledger_should_enable() -> bool:
    if _env_truthy("EIEL_EVOPY_LEDGER"):
        return True
    # Auto-enable under state dump / determinism debug harnesses.
    if _env_truthy("EIEL_STATE_DUMP") or _env_truthy("EIEL_STATE_DUMP_DEBUG"):
        return True
    # Some harness variants use policy slot tracing without state dumps.
    if _env_truthy("EIEL_DEBUG_POLICY_SLOTS") or _env_truthy("EIEL_DEBUG_POLICY_SLOTS_FAILFAST"):
        return True
    return False


_EVOPY_LEDGER_ENABLED = _evopy_ledger_should_enable()
_EVOPY_LEDGER_FAILFAST = os.environ.get("EIEL_EVOPY_LEDGER_FAILFAST", "1").strip() == "1"
try:
    _EVOPY_TRACE_N = int(os.environ.get("EIEL_EVOPY_TRACE_N", "500").strip() or "500")
except Exception:
    _EVOPY_TRACE_N = 500



# FDP v3: deterministic attribution token context.
_EVOPY_LEDGER_CTX = threading.local()


def evopy_ledger_set_token(token: str) -> None:
    """Set a deterministic attribution token for evolution_py ledger counts.

    The token is intended to be:
      - stable across fresh vs resume
      - independent of stack depth, which can differ between run modes
      - set at high-level callsites (e.g., "births_pre", "births_post")
    """
    try:
        _EVOPY_LEDGER_CTX.token = str(token)
    except Exception:
        return


def evopy_ledger_clear_token() -> None:
    """Clear the current deterministic attribution token."""
    try:
        delattr(_EVOPY_LEDGER_CTX, "token")
    except Exception:
        _EVOPY_LEDGER_CTX.token = None


def _evopy_ledger_get_token() -> str:
    """Return current ledger token (stable attribution), or 'unknown'."""
    try:
        t = getattr(_EVOPY_LEDGER_CTX, "token", None)
        if t is None:
            return "unknown"
        s = str(t)
        return s if s else "unknown"
    except Exception:
        return "unknown"


class TracedRandom(random.Random):
    """FDP v3: traced evolution_py RNG stream (subclass of random.Random).

    Design constraints:
      - MUST NOT add new RNG draws.
      - Ledger MUST use deterministic attribution tokens (no stack walking).
      - Output MUST be stable and diffable.
    """

    def __init__(self, seed_u32: int, name: str, *, ledger_enabled: bool = False, ledger_failfast: bool = True):
        super().__init__(int(seed_u32))
        self._name = str(name)
        self._call_idx = 0
        self._epoch = "boot"

        # NOTE: Ledger enablement MUST be runtime-controlled by RNGManager.
        # Module-level env parsing happens at import time and is insufficient for
        # harnesses that enable ledger under EIEL_STATE_DUMP / debug flags.
        self._ledger_enabled = bool(ledger_enabled)
        self._ledger_failfast = bool(ledger_failfast)
        self._ledger_counts = defaultdict(int)

        # Optional trace window (FDP v3): capture next N calls after a trigger stage
        # and flush once to JSON (NO per-call IO).
        self._trace_active = False
        self._trace_dir = None
        self._trace_start = 0
        self._trace_max = 0
        self._trace_buf = []
        self._trace_flushed = False

    # -----------------------------
    # Pickle / deepcopy support
    # -----------------------------
    # Shadow-clone evaluation (Phase 5 metrics) requires safe deep copies of
    # Population state. random.Random's default pickling assumes a no-arg
    # constructor, but TracedRandom requires (seed, name). Provide a stable
    # reducer so copy.deepcopy works without changing runtime behavior.

    def __getstate__(self):  # type: ignore[override]
        return {
            "random_state": self.getstate(),
            "_name": getattr(self, "_name", "unknown"),
            "_call_idx": int(getattr(self, "_call_idx", 0)),
            "_epoch": getattr(self, "_epoch", "boot"),
            "_ledger_enabled": bool(getattr(self, "_ledger_enabled", False)),
            "_ledger_failfast": bool(getattr(self, "_ledger_failfast", True)),
            "_ledger_counts": dict(getattr(self, "_ledger_counts", {})),
            "_trace_active": bool(getattr(self, "_trace_active", False)),
            "_trace_dir": getattr(self, "_trace_dir", None),
            "_trace_start": int(getattr(self, "_trace_start", 0)),
            "_trace_max": int(getattr(self, "_trace_max", 0)),
            "_trace_buf": list(getattr(self, "_trace_buf", [])),
            "_trace_flushed": bool(getattr(self, "_trace_flushed", False)),
        }

    def __setstate__(self, state):  # type: ignore[override]
        try:
            rs = state.get("random_state", None)
            if rs is not None:
                self.setstate(rs)
        except Exception:
            pass

        self._name = str(state.get("_name", getattr(self, "_name", "unknown")))
        self._call_idx = int(state.get("_call_idx", 0))
        self._epoch = str(state.get("_epoch", "boot"))

        self._ledger_enabled = bool(state.get("_ledger_enabled", False))
        self._ledger_failfast = bool(state.get("_ledger_failfast", True))

        counts = state.get("_ledger_counts", {})
        self._ledger_counts = defaultdict(int)
        try:
            for k, v in dict(counts).items():
                self._ledger_counts[k] = int(v)
        except Exception:
            pass

        self._trace_active = bool(state.get("_trace_active", False))
        self._trace_dir = state.get("_trace_dir", None)
        self._trace_start = int(state.get("_trace_start", 0))
        self._trace_max = int(state.get("_trace_max", 0))
        self._trace_buf = list(state.get("_trace_buf", []))
        self._trace_flushed = bool(state.get("_trace_flushed", False))

    def __reduce__(self):  # type: ignore[override]
        # Seed value is irrelevant because we restore full internal state.
        return (self.__class__, (0, getattr(self, "_name", "unknown")), self.__getstate__())

    def _epoch_priority(self, epoch: str) -> int:
        """Return a total ordering for trace epochs.

        Higher priority epochs must never be overridden by lower priority epochs.
        This prevents accidental resets (e.g., gen0_tick0) from masking the verified
        boundary-aligned epoch during resume comparisons.
        """
        try:
            e = str(epoch)
        except Exception:
            return 0
        # Hard-coded priority ladder (stable by design):
        #   boot < gen0_tick0 < boundary
        # Unknown epochs default to boot-level priority.
        if e == "boundary":
            return 20
        if e == "gen0_tick0":
            return 10
        if e == "boot":
            return 0
        return 0

    def reset_trace(self, epoch: str) -> None:
        """Reset call index and tag trace epoch for fresh-vs-resume alignment."""
        try:
            new_epoch = str(epoch)
            # Never allow a lower-priority epoch to override a higher-priority one.
            cur_prio = self._epoch_priority(getattr(self, "_epoch", "boot"))
            new_prio = self._epoch_priority(new_epoch)
            if new_prio < cur_prio:
                return
            self._call_idx = 0
            self._epoch = new_epoch
        except Exception:
            return
        if not _EVOPY_TRACE_ENABLED:
            return
        try:
            payload = {"stream": self._name, "epoch": str(self._epoch)}
            print("[EVOPY_TRACE_RESET]", json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")))
        except Exception:
            pass

    def getstate(self):
        """Return the underlying PRNG state.

        TracedRandom is a subclass of random.Random (not a wrapper). So we must
        call the base implementation directly (there is no `_inner`).
        """
        return random.Random.getstate(self)

    def setstate(self, state):
        """Restore the underlying PRNG state.

        State normalization (lists->tuples) is handled at snapshot restore time.
        """
        return random.Random.setstate(self, state)

    def _emit(self, method: str, pre_state: Any, post_state: Any) -> None:
        if not _EVOPY_TRACE_ENABLED:
            return
        if self._call_idx > _EVOPY_TRACE_N:
            return
        ctx = _py_rng_trace_ctx_get() or {}
        payload = {
            "stream": self._name,
            "epoch": str(self._epoch),
            "call_idx": int(self._call_idx),
            "method": str(method),
            "pre_sha": _sha_state(pre_state),
            "post_sha": _sha_state(post_state),
            "token": _evopy_ledger_get_token(),
            "gen": int(ctx.get("gen", -1)) if isinstance(ctx, dict) else -1,
            "tick": int(ctx.get("tick", -1)) if isinstance(ctx, dict) else -1,
            "stage": str(ctx.get("stage", "")) if isinstance(ctx, dict) else "",
        }
        try:
            print("[EVOPY_CALL]", json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")))
        except Exception:
            pass

    def _ledger_add(self, method: str) -> None:
        """Record a single RNG consumption in the forensic ledger."""
        if not self._ledger_enabled:
            return
        try:
            token = _evopy_ledger_get_token()
            # FDP v3 REQUIREMENT: ledger keys must be epochless.
            # Epoch is emitted as metadata only (in dump_ledger_json payload).
            key = f"{self._name}|{token}|{method}"
            self._ledger_counts[key] += 1
        except Exception:
            if self._ledger_failfast:
                raise
            return

    def ledger_snapshot(self) -> dict:
        """Return a deterministic snapshot of the current ledger."""
        if not self._ledger_enabled:
            return {}
        try:
            items = sorted(self._ledger_counts.items(), key=lambda kv: kv[0])
            return {k: int(v) for k, v in items}
        except Exception:
            return {}

    def reset_ledger(self, epoch: str) -> None:
        """Reset ledger counters and align with a new epoch label."""
        if not self._ledger_enabled:
            return
        # IMPORTANT (FDP v3): never allow a lower-priority epoch to wipe the ledger.
        # The resume path sets epoch="boundary" at a verified equivalence boundary.
        # A later gen0_tick0 hook must not erase that evidence.
        try:
            new_epoch = str(epoch)
            cur_prio = self._epoch_priority(getattr(self, "_epoch", "boot"))
            new_prio = self._epoch_priority(new_epoch)
            if new_prio < cur_prio:
                return
        except Exception:
            return
        try:
            self._ledger_counts = defaultdict(int)
        except Exception:
            return
        try:
            self.reset_trace(epoch=str(epoch))
        except Exception:
            return

    def state_sha(self) -> str:
        """Return a stable sha256 digest of the underlying RNG state."""
        try:
            return _sha_state(self.getstate())
        except Exception:
            return ""

    def state_head(self) -> dict:
        """Return a small, stable fingerprint of the underlying PRNG state.

        This is meant for diffing (fresh vs resume) without dumping the full
        624-word Mersenne Twister state. It must never mutate RNG state.
        """
        try:
            st = self.getstate()
            if not isinstance(st, (tuple, list)) or len(st) < 2:
                return {}
            inner = st[1]
            if not isinstance(inner, (tuple, list)) or len(inner) < 2:
                return {}
            # CPython MT19937: inner is 624 ints + index as last element.
            idx = inner[-1]
            head = list(inner[:8])
            # Tail excludes idx.
            if len(inner) > 9:
                tail = list(inner[-9:-1])
            else:
                tail = list(inner[:-1])
            try:
                idx_i = int(idx)
            except Exception:
                idx_i = idx
            return {"idx": idx_i, "n": int(len(inner)), "head": head, "tail": tail}
        except Exception:
            return {}


    def enable_trace_window(self, *, out_dir: str, max_calls: int = 512) -> None:
        """Enable a one-shot trace window starting at the *next* RNG call.

        This does not mutate RNG state. It only records metadata about calls.
        """
        try:
            self._trace_dir = str(out_dir)
            self._trace_start = int(self._call_idx)
            self._trace_max = max(0, int(max_calls))
            self._trace_buf = []
            self._trace_active = self._trace_max > 0
            self._trace_flushed = False
        except Exception:
            self._trace_active = False

    def _trace_maybe_record(self, method: str) -> None:
        """Record this call if a trace window is active (NO IO on hot path)."""
        if not getattr(self, "_trace_active", False):
            return
        try:
            # Record until we hit max_calls, then flush once.
            if len(self._trace_buf) < int(self._trace_max):
                self._trace_buf.append({
                    "i": int(self._call_idx),
                    "m": str(method),
                    "t": str(_evopy_ledger_get_token()),
                })
            if len(self._trace_buf) >= int(self._trace_max):
                self._trace_flush()
        except Exception:
            return

    def _trace_flush(self) -> None:
        """Flush the trace buffer once to a deterministic JSON file."""
        if getattr(self, "_trace_flushed", False):
            return
        if not getattr(self, "_trace_dir", None):
            return
        try:
            out_dir = str(self._trace_dir)
            os.makedirs(out_dir, exist_ok=True)
            start = int(getattr(self, "_trace_start", 0))
            end = start + int(getattr(self, "_trace_max", 0))
            fname = f"evopy_trace__{start:06d}__{end:06d}.json"
            path = os.path.join(out_dir, fname)
            payload = {
                "stream": "evolution_py",
                "start_call_idx": int(start),
                "max_calls": int(getattr(self, "_trace_max", 0)),
                "entries": list(self._trace_buf),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True, separators=(",", ":"))
            self._trace_flushed = True
            self._trace_active = False
        except Exception:
            return

    def dump_ledger_json(self, path: str) -> None:
        """Write the ledger to a JSON file at `path` (deterministic ordering)."""
        if not self._ledger_enabled:
            return
        try:
            payload = {
                "schema_version": 1,
                "stream": self._name,
                "epoch": str(getattr(self, "_epoch", "")),
                "total_calls": int(getattr(self, "_call_idx", 0)),
                "counts": self.ledger_snapshot(),
            }
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True, separators=(",", ":"))
        except Exception:
            return

    def random(self):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("random")
        self._trace_maybe_record("random")
        out = super().random()
        post = self.getstate()
        self._emit("random", pre, post)
        return out

    def randrange(self, *args, **kwargs):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("randrange")
        self._trace_maybe_record("randrange")
        out = super().randrange(*args, **kwargs)
        post = self.getstate()
        self._emit("randrange", pre, post)
        return out

    def randint(self, a, b):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("randint")
        self._trace_maybe_record("randint")
        out = super().randint(a, b)
        post = self.getstate()
        self._emit("randint", pre, post)
        return out

    def uniform(self, a, b):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("uniform")
        self._trace_maybe_record("uniform")
        out = super().uniform(a, b)
        post = self.getstate()
        self._emit("uniform", pre, post)
        return out

    def choice(self, seq):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("choice")
        self._trace_maybe_record("choice")
        out = super().choice(seq)
        post = self.getstate()
        self._emit("choice", pre, post)
        return out

    def choices(self, population, *args, **kwargs):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("choices")
        self._trace_maybe_record("choices")
        out = super().choices(population, *args, **kwargs)
        post = self.getstate()
        self._emit("choices", pre, post)
        return out

    def shuffle(self, x):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("shuffle")
        self._trace_maybe_record("shuffle")
        out = super().shuffle(x)
        post = self.getstate()
        self._emit("shuffle", pre, post)
        return out

    def sample(self, population, k):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("sample")
        self._trace_maybe_record("sample")
        out = super().sample(population, k)
        post = self.getstate()
        self._emit("sample", pre, post)
        return out

    def getrandbits(self, k: int):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("getrandbits")
        self._trace_maybe_record("getrandbits")
        out = super().getrandbits(k)
        post = self.getstate()
        self._emit("getrandbits", pre, post)
        return out

    def gauss(self, mu: float, sigma: float):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("gauss")
        self._trace_maybe_record("gauss")
        out = super().gauss(mu, sigma)
        post = self.getstate()
        self._emit("gauss", pre, post)
        return out

    def normalvariate(self, mu: float, sigma: float):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("normalvariate")
        self._trace_maybe_record("normalvariate")
        out = super().normalvariate(mu, sigma)
        post = self.getstate()
        self._emit("normalvariate", pre, post)
        return out

    def expovariate(self, lambd: float):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("expovariate")
        self._trace_maybe_record("expovariate")
        out = super().expovariate(lambd)
        post = self.getstate()
        self._emit("expovariate", pre, post)
        return out

    def gammavariate(self, alpha: float, beta: float):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("gammavariate")
        self._trace_maybe_record("gammavariate")
        out = super().gammavariate(alpha, beta)
        post = self.getstate()
        self._emit("gammavariate", pre, post)
        return out

    def betavariate(self, alpha: float, beta: float):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("betavariate")
        self._trace_maybe_record("betavariate")
        out = super().betavariate(alpha, beta)
        post = self.getstate()
        self._emit("betavariate", pre, post)
        return out

    def triangular(self, low: float = 0.0, high: float = 1.0, mode: Optional[float] = None):
        pre = self.getstate()
        self._call_idx += 1
        self._ledger_add("triangular")
        self._trace_maybe_record("triangular")
        out = super().triangular(low, high, mode)
        post = self.getstate()
        self._emit("triangular", pre, post)
        return out
# ============================================================
# Phase 3 forensic tracing: Python RNG call tracing
# ============================================================
#
# Goal
# ----
# Find the *first* extra/missing Python-RNG call that causes fresh-vs-resume
# drift during births/genome mutation.
#
# Design
# ------
# - Install a class-level tracer for `random.Random` methods.
# - Emit trace lines ONLY when an explicit context is active.
# - Context is set/cleared by the runner around `_handle_births`.
#
# This avoids spamming logs for the whole simulation.
# ============================================================

_PY_RNG_TRACE_CTX = threading.local()
_PY_RNG_TRACE_INSTALLED = False


def _py_rng_trace_ctx_get() -> Optional[dict]:
    """Return current trace context dict, if any."""
    return getattr(_PY_RNG_TRACE_CTX, "ctx", None)


def py_rng_trace_set_context(gen: int, tick: int, stage: str) -> None:
    """Enable python-RNG tracing for the current thread."""
    _PY_RNG_TRACE_CTX.ctx = {"gen": int(gen), "tick": int(tick), "stage": str(stage)}


def py_rng_trace_clear_context() -> None:
    """Disable python-RNG tracing for the current thread."""
    try:
        delattr(_PY_RNG_TRACE_CTX, "ctx")
    except Exception:
        _PY_RNG_TRACE_CTX.ctx = None




# ============================================================
# Phase 3 forensic tracing: NumPy Generator call tracing
# ============================================================
#
# Goal
# ----
# If genome mutation / births are using NumPy RNG (Generator / PCG64) rather than
# Python's `random.Random`, we need the *same* "first extra/missing call" visibility.
#
# Design
# ------
# - Install class-level wrappers for `numpy.random.Generator` methods.
# - Emit trace lines ONLY when the same context is active (set around births).
# - Record bit-generator state digest pre/post so we can diff call-for-call.
# ============================================================

_NP_RNG_TRACE_INSTALLED = False


def _np_bitgen_state_sha(gen: Any) -> str:
    """Return sha256 of a numpy Generator bit_generator.state without advancing RNG."""
    try:
        if gen is None:
            return "none"
        bg = getattr(gen, "bit_generator", None)
        if bg is None:
            return "none"
        st = bg.state
        b = json.dumps(st, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return "err"


def _install_np_rng_trace_if_needed() -> None:
    """Install numpy RNG tracing wrappers.

    NOTE: Disabled in FDP v3 because numpy Generator can be an immutable C-extension type
    in some builds, making monkeypatching unsafe (TypeError on setattr).
    """
    global _NP_RNG_TRACE_INSTALLED
    _NP_RNG_TRACE_INSTALLED = True
    return

def _sha_state(obj: Any) -> str:
    """Stable sha256 over a python object via json; best-effort."""
    try:
        raw = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
    except Exception:
        try:
            return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()
        except Exception:
            return ""


def _install_py_rng_trace_if_needed() -> None:
    """Install python RNG tracing wrappers.

    NOTE: Disabled in FDP v3 to avoid class-level monkeypatching. The investigation now
    uses instance-based tracing via TracedRandom for the evolution_py stream only.
    """
    global _PY_RNG_TRACE_INSTALLED
    _PY_RNG_TRACE_INSTALLED = True
    return

# Install immediately on import so the tracer cannot be skipped by call ordering.


def _seed_u32(global_seed: int, stream: str, kind: str) -> int:
    """Derive a deterministic uint32 seed from (global_seed, stream, kind)."""
    s = f"EIEL|{int(global_seed)}|{stream}|{kind}".encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:4], byteorder="little", signed=False)


@dataclass
class _StreamBundle:
    """Internal holder for one named RNG stream."""

    name: str
    py: Optional[random.Random]
    np: Optional[np.random.Generator]



class DeterminismViolationError(RuntimeError):
    """Raised when strict determinism invariants are violated."""


def _det_fail(code: str, context: str) -> None:
    """Raise a determinism violation in the canonical format."""
    msg = (
        "[DETERMINISM_VIOLATION]\n"
        f"code={code}\n"
        f"context={context}\n"
    )
    raise DeterminismViolationError(msg)


def _strict_enabled() -> bool:
    """Best-effort strict flag; mirrors simulation.determinism_enforcer."""
    try:
        import os
        if str(os.environ.get('EIEL_STRICT_DETERMINISM', '') or '').strip() == '1':
            return True
        return str(os.environ.get('DEBUG_DETERMINISM', '0') or '0').strip() == '1'
    except Exception:
        return False


def _call_original_if_patched(mod, attr: str, *args, **kwargs):
    """Call module attribute, preferring strict-guard registry original when present."""
    fn = getattr(mod, attr)
    try:
        from simulation import determinism_enforcer as _de  # local import to avoid startup coupling

        orig = _de.resolve_original_callable(getattr(mod, "__name__", ""), str(attr))
        if callable(orig):
            return orig(*args, **kwargs)
    except Exception:
        pass
    return fn(*args, **kwargs)


def _u64(x: int) -> int:
    return int(x) & 0xFFFFFFFFFFFFFFFF


def _sha256_u64(parts: list[bytes]) -> int:
    h = hashlib.sha256()
    for b in parts:
        h.update(b)
    return int.from_bytes(h.digest()[:8], byteorder='big', signed=False)


def _seed_for_stream(global_seed: int, stream_id: str) -> int:
    """Derive a stable 64-bit seed for a stream."""
    gs = _u64(int(global_seed)).to_bytes(8, byteorder='big', signed=False)
    return _sha256_u64([gs, b'|', stream_id.encode('utf-8')])


def _seed_for_domain(stream_seed: int, domain_id: str, device_id: str = '') -> int:
    """Derive a stable 64-bit seed for (stream_seed, domain, device)."""
    ss = _u64(int(stream_seed)).to_bytes(8, byteorder='big', signed=False)
    return _sha256_u64([ss, b'|', domain_id.encode('utf-8'), b'|', device_id.encode('utf-8')])


def _product_size(size: object) -> int:
    """Return number of scalars for a numpy-style `size` argument."""
    try:
        if size is None:
            return 1
        if isinstance(size, int):
            return int(size)
        # tuple/list
        prod = 1
        for v in list(size):
            prod *= int(v)
        return int(prod)
    except Exception:
        return 1


def _norm_py_state_to_jsonable(x: object) -> object:
    """Convert tuples in Random.getstate() to lists for JSON."""
    if isinstance(x, tuple):
        return [_norm_py_state_to_jsonable(v) for v in x]
    if isinstance(x, list):
        return [_norm_py_state_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _norm_py_state_to_jsonable(v) for k, v in x.items()}
    return x


def _norm_py_state_from_jsonable(x: object) -> object:
    """Convert JSON-loaded lists back to tuples for Random.setstate()."""
    if isinstance(x, list):
        return tuple(_norm_py_state_from_jsonable(v) for v in x)
    if isinstance(x, dict):
        return {k: _norm_py_state_from_jsonable(v) for k, v in x.items()}
    return x


class TrackedPyRNG:
    """Tracked wrapper around python random.Random (semantic index counting)."""

    def __init__(self, inner: random.Random, *, index_ref: dict, key: str):
        self._r = inner
        self._idx_ref = index_ref
        self._key = str(key)

    def _bump(self, n: int) -> None:
        self._idx_ref[self._key] = int(self._idx_ref.get(self._key, 0)) + int(n)

    # Minimal supported API
    def random(self) -> float:
        v = self._r.random()
        self._bump(1)
        return float(v)

    def randint(self, a: int, b: int) -> int:
        v = self._r.randint(int(a), int(b))
        self._bump(1)
        return int(v)

    def choice(self, seq):
        v = self._r.choice(seq)
        self._bump(1)
        return v

    def shuffle(self, x) -> None:
        # semantic count is number of elements
        try:
            n = len(x)
        except Exception:
            n = 0
        self._r.shuffle(x)
        self._bump(max(int(n), 0))


    # Compatibility shims
    def getstate(self):
        return self._getstate()

    def setstate(self, st) -> None:
        self._setstate(st)

    def __getattr__(self, name: str):
        try:
            attr = getattr(self._r, name)
        except Exception as e:
            raise AttributeError(name) from e
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            out = attr(*args, **kwargs)

            # Semantic index counting for common multi-draw APIs.
            bump = 1
            try:
                if name == "sample":
                    # sample(population, k)
                    k = kwargs.get("k", args[1] if len(args) > 1 else 1)
                    bump = max(int(k), 0)
                elif name == "choices":
                    # choices(population, weights=None, *, cum_weights=None, k=1)
                    k = kwargs.get("k", args[3] if len(args) > 3 else 1)
                    bump = max(int(k), 0)
                elif name == "randbytes":
                    n = args[0] if len(args) > 0 else kwargs.get("n", 1)
                    bump = max(int(n), 0)
            except Exception:
                bump = 1

            self._bump(bump)
            return out

        return _wrapped

    # State capture/restore (used by RNGManager)
    def _getstate(self):
        return self._r.getstate()

    def _setstate(self, st) -> None:
        self._r.setstate(st)



class _BitGeneratorStateProxy:
    """Proxy for numpy BitGenerator that only exposes `.state` in strict mode.

    Rationale: exposing the raw BitGenerator allows untracked draws via
    `random_raw(...)`, bypassing index/budget accounting. In strict mode we
    provide state-only access for snapshot/restore compatibility.
    """

    def __init__(self, inner):
        self._inner = inner

    @property
    def state(self):
        return getattr(self._inner, "state")

    @state.setter
    def state(self, v):
        setattr(self._inner, "state", v)

    def __getattr__(self, name: str):
        # Anything else is potentially RNG-advancing (e.g., random_raw).
        _det_fail('DET_FORBIDDEN_RNG_API', f'surface=numpy.bit_generator attr={str(name)}')


class TrackedNumpyRNG:
    """Tracked wrapper around numpy.random.Generator (semantic index counting)."""

    def __init__(self, inner: np.random.Generator, *, index_ref: dict, key: str):
        self._g = inner
        self._idx_ref = index_ref
        self._key = str(key)

    def _bump(self, n: int) -> None:
        self._idx_ref[self._key] = int(self._idx_ref.get(self._key, 0)) + int(n)


    @property
    def bit_generator(self):
        """Compatibility shim for legacy code.

        STRICT MODE:
          - returns a proxy that exposes only `.state` (get/set)
          - forbids any RNG-advancing BitGenerator methods (e.g. random_raw)

        NON-STRICT:
          - returns the underlying bit_generator for compatibility (NOT accountable)
        """
        try:
            bg = self._g.bit_generator
        except Exception:
            return None

        if _strict_enabled():
            return _BitGeneratorStateProxy(bg)
        return bg

    def __getattr__(self, name: str):
        """Best-effort forwarder for uncommon Generator APIs.

        This is *not* ideal, but it avoids breaking existing code while we
        continue migrating call sites onto explicit tracked wrappers.

        Counting rule: if the forwarded method returns an array-like,
        we bump by its scalar size, otherwise bump by 1.
        """
        try:
            attr = getattr(self._g, name)
        except Exception as e:
            raise AttributeError(name) from e

        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            out = attr(*args, **kwargs)
            try:
                n = int(np.size(out))
            except Exception:
                n = 1
            self._bump(max(int(n), 0))
            return out

        return _wrapped

    def random(self, size=None, dtype=None):
        out = self._g.random(size=size, dtype=dtype)
        try:
            n = int(np.size(out))
        except Exception:
            n = _product_size(size)
        self._bump(max(n, 0))
        return out

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        out = self._g.integers(low=low, high=high, size=size, dtype=dtype, endpoint=endpoint)
        try:
            n = int(np.size(out))
        except Exception:
            n = _product_size(size)
        self._bump(max(n, 0))
        return out

    def normal(self, loc=0.0, scale=1.0, size=None):
        out = self._g.normal(loc=loc, scale=scale, size=size)
        try:
            n = int(np.size(out))
        except Exception:
            n = _product_size(size)
        self._bump(max(n, 0))
        return out

    def uniform(self, low=0.0, high=1.0, size=None):
        out = self._g.uniform(low=low, high=high, size=size)
        try:
            n = int(np.size(out))
        except Exception:
            n = _product_size(size)
        self._bump(max(n, 0))
        return out

    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        out = self._g.choice(a=a, size=size, replace=replace, p=p, axis=axis, shuffle=shuffle)
        try:
            n = int(np.size(out))
        except Exception:
            n = _product_size(size)
        self._bump(max(n, 0))
        return out

    def permutation(self, x, axis=0):
        out = self._g.permutation(x, axis=axis)
        # semantic count is number of elements permuted
        try:
            if isinstance(x, int):
                n = int(x)
            else:
                n = int(len(x))
        except Exception:
            try:
                n = int(np.size(out))
            except Exception:
                n = 0
        self._bump(max(n, 0))
        return out

    def shuffle(self, x, axis=0):
        try:
            n = int(np.size(x))
        except Exception:
            try:
                n = int(len(x))
            except Exception:
                n = 0
        self._g.shuffle(x, axis=axis)
        self._bump(max(n, 0))

    # State capture/restore
    def _get_bitgen_state(self):
        return self._g.bit_generator.state

    def _set_bitgen_state(self, st) -> None:
        self._g.bit_generator.state = st


class TrackedTorchRNG:
    """Tracked wrapper around torch.Generator with semantic index counting."""

    def __init__(self, inner, *, index_ref: dict, key: str):
        self.__gen = inner
        self._idx_ref = index_ref
        self._key = str(key)

    def __getattribute__(self, name: str):
        if _strict_enabled() and str(name) in ("_gen", "_TrackedTorchRNG__gen"):
            _det_fail('DET_FORBIDDEN_RNG_API', f'surface=torch.generator attr={str(name)}')
        return object.__getattribute__(self, name)

    def _inner_gen(self):
        return object.__getattribute__(self, "_TrackedTorchRNG__gen")

    def _bump(self, n: int) -> None:
        self._idx_ref[self._key] = int(self._idx_ref.get(self._key, 0)) + int(n)

    @property
    def generator(self):
        """Return underlying torch.Generator.

        STRICT MODE: forbidden (raw generator allows untracked draws).
        NON-STRICT: returned for compatibility only.
        """
        if _strict_enabled():
            _det_fail('DET_FORBIDDEN_RNG_API', 'surface=torch.generator attr=generator')
        return self._inner_gen()

    def bump_tensor(self, t) -> None:
        """Increment semantic index by numel of tensor `t` (no RNG draw here)."""
        try:
            n = int(getattr(t, 'numel')())
        except Exception:
            n = 0
        self._bump(max(n, 0))

    # -----------------------------
    # Tracked torch draw helpers
    # -----------------------------
    def rand(self, size, *, device=None, dtype=None, requires_grad=False):
        """Tracked wrapper for torch.rand(..., generator=...).

        Index counting rule: bump by numel of returned tensor.
        """
        import torch
        t = _call_original_if_patched(
            torch, 'rand', size, generator=self._inner_gen(), device=device, dtype=dtype, requires_grad=requires_grad
        )
        self.bump_tensor(t)
        return t

    def randn(self, size, *, device=None, dtype=None, requires_grad=False):
        """Tracked wrapper for torch.randn(..., generator=...)."""
        import torch
        t = _call_original_if_patched(
            torch, 'randn', size, generator=self._inner_gen(), device=device, dtype=dtype, requires_grad=requires_grad
        )
        self.bump_tensor(t)
        return t

    def randint(self, low, high=None, size=None, *, device=None, dtype=None, requires_grad=False):
        """Tracked wrapper for torch.randint(..., generator=...)."""
        import torch
        h = high
        l = low
        if h is None:
            h = low
            l = 0
        t = _call_original_if_patched(
            torch,
            'randint',
            l,
            h,
            size=size,
            generator=self._inner_gen(),
            device=device,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        self.bump_tensor(t)
        return t

    def rand_like(self, other, *, dtype=None, requires_grad=False):
        """Tracked wrapper for torch.rand_like using explicit generator."""
        dd = dtype if dtype is not None else getattr(other, "dtype", None)
        return self.rand(
            tuple(int(x) for x in getattr(other, "shape", ())),
            device=getattr(other, "device", None),
            dtype=dd,
            requires_grad=requires_grad,
        )

    def __getattr__(self, name: str):
        if _strict_enabled():
            _det_fail('DET_TORCH_GENERATOR_UNSUPPORTED_OP', f'op={str(name)}')
        raise AttributeError(name)

    def _get_state_hex(self) -> str:
        try:
            st = self._inner_gen().get_state()
            # ByteTensor -> bytes
            b = bytes(st.detach().to('cpu').numpy().tobytes())
            return b.hex()
        except Exception:
            return ''

    def _set_state_hex(self, hx: str) -> None:
        try:
            import torch
            b = bytes.fromhex(str(hx)) if hx else b''
            # torch expects ByteTensor
            self._inner_gen().set_state(torch.tensor(list(b), dtype=torch.uint8))
        except Exception:
            return


@dataclass
class _StreamStateV2:
    name: str
    stream_seed: int
    py: TrackedPyRNG
    np: TrackedNumpyRNG
    torch_cpu: Optional[TrackedTorchRNG]
    torch_cuda: Dict[str, TrackedTorchRNG]


class _BudgetScope:
    def __init__(self, rm: 'RNGManager', label: str, expected_delta: Optional[dict[str, int]]):
        self._rm = rm
        self._label = str(label)
        self._expected = expected_delta
        self._start = None

    def __enter__(self):
        self._start = self._rm.indices_snapshot()
        return self

    def __exit__(self, exc_type, exc, tb):
        end = self._rm.indices_snapshot()
        start = self._start or {}
        keys = set(start.keys()) | set(end.keys())
        if self._expected is not None:
            keys |= set(self._expected.keys())
        delta = {k: int(end.get(k, 0)) - int(start.get(k, 0)) for k in keys}
        if self._expected is not None and _strict_enabled() and exc is None:
            observed = {k: int(delta.get(k, 0)) for k in self._expected.keys()}
            mismatched = [k for k, v in self._expected.items() if int(observed.get(k, 0)) != int(v)]
            if mismatched:
                _det_fail(
                    'DET_RNG_BUDGET_MISMATCH',
                    (
                        f"label={self._label} "
                        f"first_key={mismatched[0]} "
                        f"expected={json.dumps(self._expected, sort_keys=True, separators=(',', ':'))} "
                        f"observed={json.dumps(observed, sort_keys=True, separators=(',', ':'))}"
                    ),
                )
        return False


class RNGManager:
    """Centralized RNG authority (RNGManager v2).

    Notes
    -----
    This is a *compatibility upgrade* over the earlier Phase-1 RNGManager:
    - Fixed stream registry: policy/mutation/metrics/stress.
    - Stream index tracking per (stream, domain, device).
    - Snapshot/restore includes indices + states + torch generators.

    Backward-compat shims:
    - `env_*` maps to policy stream.
    - `evolution_*` maps to mutation stream.
    """

    # Fixed registry (P0-B)
    _ALLOWED_STREAMS = ('policy', 'mutation', 'metrics', 'stress')

    # Transitional aliases (explicit)
    _ALIASES = {
        'env': 'policy',
        'evolution': 'mutation',
    }

    def __init__(self, global_seed: int):
        self.global_seed = int(global_seed)
        self._streams: Dict[str, _StreamStateV2] = {}
        self._registry_frozen = False
        self._budget_contract: Dict[str, Any] = {}
        self._budget_contract_sha256: str = ""
        self._budget_contract_version: str = ""

        self._load_budget_contract()

        # Deterministic defensive seeding of legacy globals (transitional safety)
        self._seed_legacy_globals_if_possible()

        # Torch determinism config (strict mode only)
        self._configure_torch_determinism_if_strict()

        # Initialize all fixed streams up front, then freeze registry.
        for nm in self._ALLOWED_STREAMS:
            self._ensure_stream(nm)
        self._registry_frozen = True

    def _load_budget_contract(self) -> None:
        """Load and validate governance-backed RNG budget contract."""
        repo_root = Path(__file__).resolve().parents[2]
        contract_path = repo_root / "governance" / "rng_budget_contract.json"
        lock_path = repo_root / "governance" / "rng_budget_contract.lock.json"

        if not contract_path.exists():
            if _strict_enabled():
                _det_fail("DET_RNG_BUDGET_CONTRACT_MISSING", f"path={contract_path}")
            return

        try:
            raw = contract_path.read_bytes()
            obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            if _strict_enabled():
                _det_fail("DET_RNG_BUDGET_CONTRACT_INVALID", f"path={contract_path} err={type(e).__name__}:{e}")
            return

        if not isinstance(obj, dict):
            if _strict_enabled():
                _det_fail("DET_RNG_BUDGET_CONTRACT_INVALID", "contract_not_dict")
            return
        if str(obj.get("schema_version", "")) != "rng_budget_contract_v1":
            if _strict_enabled():
                _det_fail("DET_RNG_BUDGET_CONTRACT_INVALID", f"schema={obj.get('schema_version')}")
            return
        boundaries = obj.get("boundaries", {})
        if not isinstance(boundaries, dict):
            if _strict_enabled():
                _det_fail("DET_RNG_BUDGET_CONTRACT_INVALID", "boundaries_not_dict")
            return

        self._budget_contract = obj
        self._budget_contract_version = str(obj.get("contract_version", "") or "")
        self._budget_contract_sha256 = hashlib.sha256(raw).hexdigest()

        if not lock_path.exists():
            if _strict_enabled():
                _det_fail("DET_RNG_BUDGET_CONTRACT_LOCK_MISSING", f"path={lock_path}")
            return

        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            expected_sha = str(lock.get("contract_sha256", "") or "")
            if expected_sha != self._budget_contract_sha256 and _strict_enabled():
                _det_fail(
                    "DET_RNG_BUDGET_CONTRACT_LOCK_MISMATCH",
                    f"expected_sha={expected_sha} got_sha={self._budget_contract_sha256}",
                )
        except Exception as e:
            if _strict_enabled():
                _det_fail("DET_RNG_BUDGET_CONTRACT_LOCK_INVALID", f"path={lock_path} err={type(e).__name__}:{e}")

    def expected_delta_for_boundary(self, label: str, *, ctx: Optional[dict[str, int]] = None) -> dict[str, int]:
        """Resolve expected delta map for a contract boundary label."""
        boundaries = self._budget_contract.get("boundaries", {}) if isinstance(self._budget_contract, dict) else {}
        entry = boundaries.get(str(label))
        if not isinstance(entry, dict):
            if _strict_enabled():
                _det_fail("DET_RNG_BUDGET_BOUNDARY_UNKNOWN", f"label={label}")
            return {}
        expected = entry.get("expected_delta", {})
        if not isinstance(expected, dict):
            if _strict_enabled():
                _det_fail("DET_RNG_BUDGET_CONTRACT_INVALID", f"label={label} expected_delta_not_dict")
            return {}
        c = dict(ctx or {})
        out: dict[str, int] = {}
        for key, raw in expected.items():
            if isinstance(raw, int):
                out[str(key)] = int(raw)
                continue
            if isinstance(raw, str) and raw.startswith("ctx."):
                k = str(raw[4:])
                if k not in c and _strict_enabled():
                    _det_fail("DET_RNG_BUDGET_CONTEXT_MISSING", f"label={label} ctx_key={k}")
                out[str(key)] = int(c.get(k, 0))
                continue
            if _strict_enabled():
                _det_fail("DET_RNG_BUDGET_CONTRACT_INVALID", f"label={label} key={key} value={raw}")
            out[str(key)] = 0
        return out

    # -----------------------------
    # Legacy seed handling
    # -----------------------------
    def _seed_legacy_globals_if_possible(self) -> None:
        # Always seed legacy globals at simulation construction time.  Strict
        # mode still owns the fail-fast API guards and deterministic torch
        # algorithm configuration, but exploratory/non-strict runs also need
        # per-simulation isolation because torch module constructors and older
        # compatibility paths may still touch global RNG state.
        try:
            # Use a deterministic seed distinct from stream seeds.
            s = _seed_for_stream(self.global_seed, 'legacy_globals')
            import random as _random
            _call_original_if_patched(_random, 'seed', int(s & 0xFFFFFFFF))
        except Exception:
            pass
        try:
            import numpy as _np
            _call_original_if_patched(_np.random, 'seed', int(_seed_for_stream(self.global_seed, 'legacy_numpy') & 0xFFFFFFFF))
        except Exception:
            pass
        # Some policy/model constructors still rely on torch's global initializer RNG.
        # Seed it deterministically per-simulation to avoid cross-run process bleed.
        try:
            import torch as _torch
            _call_original_if_patched(_torch, 'manual_seed', int(_seed_for_stream(self.global_seed, 'legacy_torch_global') & 0x7FFFFFFFFFFFFFFF))
            try:
                if _torch.cuda.is_available():
                    _call_original_if_patched(_torch.cuda, 'manual_seed_all', int(_seed_for_stream(self.global_seed, 'legacy_torch_cuda') & 0x7FFFFFFFFFFFFFFF))
            except Exception:
                pass
        except Exception:
            pass

    def _configure_torch_determinism_if_strict(self) -> None:
        if not _strict_enabled():
            return
        try:
            import torch
            torch.use_deterministic_algorithms(True)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass
        except Exception as e:
            _det_fail('DET_TORCH_DETERMINISM_NOT_CONFIGURED', f'err={type(e).__name__}:{e}')

    # -----------------------------
    # Stream registry
    # -----------------------------
    def _resolve_stream_name(self, name: str) -> str:
        nm = str(name)
        if nm in self._ALIASES:
            return self._ALIASES[nm]
        return nm

    def _ensure_stream(self, name: str) -> _StreamStateV2:
        nm = self._resolve_stream_name(name)
        if nm not in self._ALLOWED_STREAMS:
            if _strict_enabled():
                _det_fail('DET_RNG_UNKNOWN_STREAM', f'stream={nm}')
            # non-strict: map unknown to policy to avoid explosions in exploratory runs
            nm = 'policy'
        if nm in self._streams:
            return self._streams[nm]
        if self._registry_frozen and _strict_enabled():
            _det_fail('DET_RNG_STREAM_REGISTRY_MUTATION', f'stream={nm}')

        # Per-stream seed
        stream_seed = _seed_for_stream(self.global_seed, nm)

        # Per-domain seeds
        py_seed = _seed_for_domain(stream_seed, 'py')
        np_seed = _seed_for_domain(stream_seed, 'np')

        # Underlying RNGs
        if nm == 'mutation':
            # Keep existing forensic TracedRandom behavior for the old evolution stream.
            inner_py = TracedRandom(
                int((py_seed ^ (py_seed >> 32)) & 0xFFFFFFFF),
                name=f'{nm}_py',
                ledger_enabled=bool(getattr(self, '_evopy_ledger_enabled', False) or _EVOPY_LEDGER_ENABLED),
                ledger_failfast=bool(_EVOPY_LEDGER_FAILFAST),
            )
        else:
            inner_py = random.Random(int(py_seed & 0xFFFFFFFFFFFFFFFF))

        inner_np = _call_original_if_patched(
            np.random,
            'default_rng',
            int(np_seed & 0xFFFFFFFFFFFFFFFF),
        )

        # Indices live on the manager, keyed per stream.
        idx_ref = self._indices
        # Ensure keys exist
        _ = idx_ref

        py_key = f'{nm}.py'
        np_key = f'{nm}.np'
        cpu_key = f'{nm}.torch_cpu'

        py = TrackedPyRNG(inner_py, index_ref=idx_ref, key=py_key)
        ng = TrackedNumpyRNG(inner_np, index_ref=idx_ref, key=np_key)

        torch_cpu = None
        torch_cuda = {}
        try:
            import torch
            g_cpu = _call_original_if_patched(torch, 'Generator', device='cpu')
            g_cpu.manual_seed(int(_seed_for_domain(stream_seed, 'torch', 'cpu')))
            torch_cpu = TrackedTorchRNG(g_cpu, index_ref=idx_ref, key=cpu_key)

            if torch.cuda.is_available():
                # Stable device ordering: cuda:0, cuda:1, ...
                ndev = int(torch.cuda.device_count())
                for di in range(ndev):
                    dev = f'cuda:{di}'
                    g = _call_original_if_patched(torch, 'Generator', device=dev)
                    g.manual_seed(int(_seed_for_domain(stream_seed, 'torch', dev)))
                    torch_cuda[dev] = TrackedTorchRNG(g, index_ref=idx_ref, key=f'{nm}.torch_cuda.{dev}')
        except Exception:
            # torch missing or CPU-only; ok
            pass

        st = _StreamStateV2(name=nm, stream_seed=int(stream_seed), py=py, np=ng, torch_cpu=torch_cpu, torch_cuda=torch_cuda)
        self._streams[nm] = st
        return st

    @property
    def _indices(self) -> Dict[str, int]:
        if not hasattr(self, '__idx'):
            setattr(self, '__idx', {})
        return getattr(self, '__idx')

    def indices_snapshot(self) -> Dict[str, int]:
        """Return a shallow copy of all semantic indices."""
        try:
            return dict(self._indices)
        except Exception:
            return {}

    # -----------------------------
    # Public stream accessors (v2)
    # -----------------------------
    def py_rng(self, name: str) -> TrackedPyRNG:
        return self._ensure_stream(name).py

    def np_rng(self, name: str) -> TrackedNumpyRNG:
        return self._ensure_stream(name).np

    def torch_rng(self, name: str, device: str = 'cpu') -> Optional[TrackedTorchRNG]:
        st = self._ensure_stream(name)
        if str(device) == 'cpu':
            return st.torch_cpu
        return st.torch_cuda.get(str(device))

    # Convenience properties (new fixed streams)
    @property
    def policy_py(self) -> TrackedPyRNG:
        return self.py_rng('policy')

    @property
    def policy_np(self) -> TrackedNumpyRNG:
        return self.np_rng('policy')

    @property
    def mutation_py(self) -> TrackedPyRNG:
        return self.py_rng('mutation')

    @property
    def mutation_np(self) -> TrackedNumpyRNG:
        return self.np_rng('mutation')

    @property
    def metrics_np(self) -> TrackedNumpyRNG:
        return self.np_rng('metrics')

    @property
    def stress_py(self) -> TrackedPyRNG:
        return self.py_rng('stress')

    @property
    def stress_np(self) -> TrackedNumpyRNG:
        return self.np_rng('stress')

    # Backward-compat properties
    @property
    def env_np(self) -> TrackedNumpyRNG:
        return self.policy_np

    @property
    def evolution_py(self) -> TrackedPyRNG:
        return self.mutation_py

    @property
    def evolution_np(self) -> TrackedNumpyRNG:
        return self.mutation_np

    # -----------------------------
    # Budget scopes
    # -----------------------------
    def budget_scope(self, label: str, expected_delta: Optional[dict[str, int]] = None) -> _BudgetScope:
        """Context manager that asserts RNG index deltas (strict mode)."""
        return _BudgetScope(self, label=label, expected_delta=expected_delta)

    # -----------------------------
    # Snapshot / restore
    # -----------------------------
    def _device_list(self) -> list[str]:
        devs = ['cpu']
        try:
            import torch
            if torch.cuda.is_available():
                ndev = int(torch.cuda.device_count())
                for di in range(ndev):
                    devs.append(f'cuda:{di}')
        except Exception:
            pass
        return devs

    def capture_snapshot(self, generation_index: Optional[int] = None) -> Dict[str, Any]:
        """Capture rng_snapshot_v2 (indices + states, per stream/domain/device)."""
        snap: Dict[str, Any] = {
            'schema_version': 'rng_snapshot_v2',
            'generation_index': int(generation_index) if generation_index is not None else None,
            'global_seed': int(self.global_seed),
            'device_list': self._device_list(),
            'streams': {},
        }

        for nm in self._ALLOWED_STREAMS:
            st = self._ensure_stream(nm)
            # indices
            idx = {
                'py': int(self._indices.get(f'{nm}.py', 0)),
                'np': int(self._indices.get(f'{nm}.np', 0)),
                'torch_cpu': int(self._indices.get(f'{nm}.torch_cpu', 0)),
                'torch_cuda': {dev: int(self._indices.get(f'{nm}.torch_cuda.{dev}', 0)) for dev in st.torch_cuda.keys()},
            }
            # states
            states = {
                'py_state': _norm_py_state_to_jsonable(st.py._getstate()),
                'np_state': st.np._get_bitgen_state(),
                'torch_cpu_state': st.torch_cpu._get_state_hex() if st.torch_cpu is not None else '',
                'torch_cuda_state': {dev: tg._get_state_hex() for dev, tg in st.torch_cuda.items()},
            }

            snap['streams'][nm] = {
                'stream_id': nm,
                'indices': idx,
                'states': states,
            }

        return snap

    def apply_snapshot_dict(self, snapshot: Dict[str, Any]) -> None:
        """Restore rng_snapshot_v2.

        Accepts older snapshot dicts best-effort, but strict mode enforces schema.
        """
        if not isinstance(snapshot, dict):
            return

        schema = str(snapshot.get('schema_version', '') or '')
        if schema and schema != 'rng_snapshot_v2' and _strict_enabled():
            _det_fail('DET_RNG_SNAPSHOT_SCHEMA_MISMATCH', f'schema={schema}')

        streams = snapshot.get('streams', {})
        if not isinstance(streams, dict):
            if _strict_enabled():
                _det_fail('DET_RNG_SNAPSHOT_SCHEMA_MISMATCH', 'missing_streams')
            return

        for raw_name, payload in streams.items():
            nm = self._resolve_stream_name(str(raw_name))
            if nm not in self._ALLOWED_STREAMS:
                if _strict_enabled():
                    _det_fail('DET_RNG_UNKNOWN_STREAM', f'stream={nm}')
                continue
            if not isinstance(payload, dict):
                continue

            st = self._ensure_stream(nm)
            # indices
            idxp = payload.get('indices', {})
            if isinstance(idxp, dict):
                try:
                    self._indices[f'{nm}.py'] = int(idxp.get('py', 0) or 0)
                    self._indices[f'{nm}.np'] = int(idxp.get('np', 0) or 0)
                    self._indices[f'{nm}.torch_cpu'] = int(idxp.get('torch_cpu', 0) or 0)
                    tc = idxp.get('torch_cuda', {})
                    if isinstance(tc, dict):
                        for dev, v in tc.items():
                            self._indices[f'{nm}.torch_cuda.{dev}'] = int(v or 0)
                except Exception:
                    if _strict_enabled():
                        _det_fail('DET_RNG_SNAPSHOT_STATE_INVALID', f'bad_indices stream={nm}')

            # states
            stp = payload.get('states', payload)  # allow earlier shape
            if isinstance(stp, dict):
                try:
                    py_state = stp.get('py_state', None)
                    if py_state is not None:
                        st.py._setstate(_norm_py_state_from_jsonable(py_state))
                except Exception:
                    if _strict_enabled():
                        _det_fail('DET_RNG_SNAPSHOT_STATE_INVALID', f'py stream={nm}')

                try:
                    np_state = stp.get('np_state', None)
                    if np_state is not None:
                        st.np._set_bitgen_state(np_state)
                except Exception:
                    if _strict_enabled():
                        _det_fail('DET_RNG_SNAPSHOT_STATE_INVALID', f'np stream={nm}')

                # torch
                try:
                    if st.torch_cpu is not None:
                        st.torch_cpu._set_state_hex(str(stp.get('torch_cpu_state', '') or ''))
                except Exception:
                    if _strict_enabled():
                        _det_fail('DET_RNG_SNAPSHOT_STATE_INVALID', f'torch_cpu stream={nm}')

                try:
                    tcs = stp.get('torch_cuda_state', {})
                    if isinstance(tcs, dict):
                        for dev, hx in tcs.items():
                            tg = st.torch_cuda.get(str(dev))
                            if tg is not None:
                                tg._set_state_hex(str(hx or ''))
                except Exception:
                    if _strict_enabled():
                        _det_fail('DET_RNG_SNAPSHOT_STATE_INVALID', f'torch_cuda stream={nm}')

    # -----------------------------
    # FDP v3 compatibility hooks
    # -----------------------------
    def _set_trace_ctx(self, *, gen: int, tick: int, stage: str) -> None:
        try:
            py_rng_trace_set_context(gen=int(gen), tick=int(tick), stage=str(stage))
        except Exception:
            return

    def evopy_ledger_set_token(self, token: str) -> None:
        # evolution==mutation in v2
        try:
            evopy_ledger_set_token(str(token))
        except Exception:
            return

    def evopy_ledger_clear_token(self) -> None:
        try:
            evopy_ledger_clear_token()
        except Exception:
            return

    def _reset_evopy_trace(self, *, epoch: str) -> None:
        # Keep existing API, mapped to mutation_py underlying rng when possible
        try:
            st = self._ensure_stream('mutation')
            inner = getattr(st.py, '_r', None)
            if hasattr(inner, 'reset_trace'):
                inner.reset_trace(str(epoch))
            self._evopy_epoch = str(epoch)
        except Exception:
            return

    def _reset_evopy_ledger(self, *, epoch: str) -> None:
        try:
            st = self._ensure_stream('mutation')
            inner = getattr(st.py, '_r', None)
            if hasattr(inner, 'reset_ledger'):
                inner.reset_ledger(str(epoch))
            self._evopy_epoch = str(epoch)
        except Exception:
            return

    def dump_evopy_ledger(self, *, path: str) -> None:
        try:
            st = self._ensure_stream('mutation')
            inner = getattr(st.py, '_r', None)
            if hasattr(inner, 'dump_ledger_json'):
                inner.dump_ledger_json(str(path))
        except Exception:
            return

    def dump_evopy_state_json(self, *, path: str, stage: str) -> None:
        try:
            st = self._ensure_stream('mutation')
            inner = getattr(st.py, '_r', None)
            if inner is None:
                return
            sha = ''
            try:
                if hasattr(inner, 'state_sha') and callable(getattr(inner, 'state_sha')):
                    sha = str(inner.state_sha())
                else:
                    sha = _sha_state(inner.getstate())
            except Exception:
                sha = ''

            ledger = {}
            try:
                if hasattr(inner, 'ledger_snapshot') and callable(getattr(inner, 'ledger_snapshot')):
                    ledger = inner.ledger_snapshot()
            except Exception:
                ledger = {}

            state_head = {}
            try:
                if hasattr(inner, 'state_head') and callable(getattr(inner, 'state_head')):
                    state_head = inner.state_head()
            except Exception:
                state_head = {}

            payload = {
                'stream': 'mutation_py',
                'epoch': str(getattr(inner, '_epoch', '')),
                'stage': str(stage),
                'state_sha': str(sha),
                'call_idx': int(getattr(inner, '_call_idx', -1)),
                'state_head': state_head,
                'ledger': ledger,
            }
            import os, json
            os.makedirs(os.path.dirname(os.path.abspath(str(path))), exist_ok=True)
            with open(str(path), 'w', encoding='utf-8') as f:
                json.dump(payload, f, sort_keys=True, separators=(',', ':'))
        except Exception:
            return

    # -----------------------------
    # Debug snapshot
    # -----------------------------
    def debug_snapshot(self) -> dict:
        """Return debug info for policy stream without drawing RNG."""
        try:
            state = self.policy_np._get_bitgen_state()
            raw = json.dumps(state, sort_keys=True).encode('utf-8')
            digest = hashlib.sha256(raw).hexdigest()
            return {
                'policy_np_state_sha': digest,
                'policy_np_index': int(self._indices.get('policy.np', 0)),
            }
        except Exception:
            return {}
