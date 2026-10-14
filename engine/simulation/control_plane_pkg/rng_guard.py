from __future__ import annotations
"""simulation/control_plane_pkg/rng_guard.py

Determinism guard for ControlPlane command application.

Phase 2 note:
The ControlPlane itself must not introduce new RNG paths. However, command implementations
(especially apply_to_sim()) could accidentally call Python's global RNGs (random.*) or NumPy's
global RNG helpers (numpy.random.*), which would create untracked randomness outside RNGManager.

This module provides a narrow, deterministic guard used ONLY during SafeBoundary command application:
- Temporarily monkeypatch common global RNG entry points to raise an error.
- Restore originals after command execution.

This is intentionally strict: if a command needs RNG, it must draw through RNGManager streams
(or be purely deterministic).
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional
import random


class DeterminismViolationError(RuntimeError):
    """Raised when a command attempts to use global RNG outside RNGManager."""
    pass


def _make_blocker(name: str) -> Callable[..., Any]:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise DeterminismViolationError(f"Global RNG call blocked during command apply: {name}")
    return _blocked


@contextmanager
def block_global_rng() -> Iterator[None]:
    """Block common global RNG functions for the duration of the context.

    This is used around CommandBase.apply_to_sim() inside SafeBoundary step (6).
    It is deterministic: it either runs normally or raises DeterminismViolationError.

    It does NOT attempt to patch per-instance random.Random objects.
    It focuses on the accidental, easy-to-miss sources of nondeterminism:
      - random.random / randint / choice / shuffle / sample / uniform / gauss / getrandbits / randrange
      - numpy.random convenience functions (if numpy is available)
    """
    saved: Dict[str, Any] = {}

    def patch(obj: Any, attr: str) -> None:
        key = f"{id(obj)}:{attr}"
        saved[key] = getattr(obj, attr, None)
        setattr(obj, attr, _make_blocker(f"{obj.__name__ if hasattr(obj, '__name__') else obj}.{attr}"))

    # Patch Python random module functions
    for fn in [
        "random",
        "randrange",
        "randint",
        "choice",
        "shuffle",
        "sample",
        "uniform",
        "gauss",
        "getrandbits",
    ]:
        if hasattr(random, fn):
            saved[f"random:{fn}"] = getattr(random, fn)
            setattr(random, fn, _make_blocker(f"random.{fn}"))

    # Patch numpy.random convenience API if available
    np = None
    try:
        import numpy as _np  # type: ignore
        np = _np
    except Exception:
        np = None

    if np is not None:
        # numpy.random is a module-like object
        npr = getattr(np, "random", None)
        if npr is not None:
            for fn in [
                "random",
                "rand",
                "randint",
                "choice",
                "shuffle",
                "permutation",
                "default_rng",
            ]:
                if hasattr(npr, fn):
                    saved[f"numpy.random:{fn}"] = getattr(npr, fn)
                    setattr(npr, fn, _make_blocker(f"numpy.random.{fn}"))

    try:
        yield
    finally:
        # Restore Python random
        for fn in [
            "random",
            "randrange",
            "randint",
            "choice",
            "shuffle",
            "sample",
            "uniform",
            "gauss",
            "getrandbits",
        ]:
            key = f"random:{fn}"
            if key in saved:
                setattr(random, fn, saved[key])

        # Restore numpy.random if present
        if np is not None:
            npr = getattr(np, "random", None)
            if npr is not None:
                for fn in [
                    "random",
                    "rand",
                    "randint",
                    "choice",
                    "shuffle",
                    "permutation",
                    "default_rng",
                ]:
                    key = f"numpy.random:{fn}"
                    if key in saved and hasattr(npr, fn):
                        setattr(npr, fn, saved[key])
