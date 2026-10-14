"""engine.simulation.determinism_enforcer

P0-A — Determinism Enforcement Scaffold (runtime guard).

This module provides *runtime* determinism enforcement that complements the
static scanner in tools.diagnostics.static_rng_scan.

Goals
-----
- Fail fast when strict determinism is enabled and critical invariants are missing.
- Provide a minimal guard against accidental introduction of forbidden RNG calls.

Non-goals
---------
- This does NOT migrate RNG usage.
- This does NOT implement RNGManager.

Strict mode
-----------
- Enabled when either DEBUG_DETERMINISM=1 or EIEL_STRICT_DETERMINISM=1.

Failure semantics
-----------------
Raises RuntimeError with a deterministic, parseable header.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Callable, Optional

_PATCHED_FUNCTIONS: dict[tuple[str, str], Callable] = {}


def strict_enabled() -> bool:
    """Return True if strict determinism is enabled."""
    try:
        if str(os.environ.get("EIEL_STRICT_DETERMINISM", "") or "").strip() == "1":
            return True
        return str(os.environ.get("DEBUG_DETERMINISM", "0") or "0").strip() == "1"
    except Exception:
        return False


def sync_env_vars_from_eiel_strict() -> None:
    """If EIEL_STRICT_DETERMINISM is set, force DEBUG_DETERMINISM accordingly."""
    try:
        v = str(os.environ.get("EIEL_STRICT_DETERMINISM", "") or "").strip()
        if v in ("0", "1"):
            os.environ["DEBUG_DETERMINISM"] = v
    except Exception:
        pass


def enforce_startup_invariants() -> None:
    """Enforce strict-mode startup invariants.

    Checks:
    - PYTHONHASHSEED must be set in strict mode (prevents hash-randomized dict ordering).
    """
    if not strict_enabled():
        return

    # PYTHONHASHSEED must be explicitly set for deterministic hashing.
    if not str(os.environ.get("PYTHONHASHSEED", "") or "").strip():
        _fail(
            code="DET_ENV_MISSING_PYTHONHASHSEED",
            symbol="PYTHONHASHSEED",
            details="Set PYTHONHASHSEED=0 (or a fixed integer) for strict determinism.",
        )


def install_runtime_rng_guards() -> None:
    """Install lightweight runtime guards in strict mode.

    The goal is to catch *new* accidental global RNG usage. This is not a
    perfect sandbox; it is an early warning tripwire.

    NOTE: Torch global RNG APIs are patched in strict mode. RNGManager wrappers
    resolve preserved originals via a private module-local registry.
    """
    if not strict_enabled():
        return

    # Seed APIs (explicit seed drift vectors).
    _patch_function("random", "seed", "DET_FORBIDDEN_SEED_CALL")
    _patch_function("numpy.random", "seed", "DET_FORBIDDEN_SEED_CALL")
    _patch_function("torch", "manual_seed", "DET_FORBIDDEN_SEED_CALL")
    _patch_function("torch", "seed", "DET_FORBIDDEN_SEED_CALL")
    _patch_function("torch.random", "manual_seed", "DET_FORBIDDEN_SEED_CALL")
    _patch_function("torch.cuda", "manual_seed_all", "DET_FORBIDDEN_SEED_CALL")

    # Python global randomness APIs (must route through RNGManager wrappers).
    for _fn in (
        "random",
        "randrange",
        "randint",
        "choice",
        "choices",
        "shuffle",
        "sample",
        "uniform",
        "gauss",
        "normalvariate",
        "triangular",
        "getrandbits",
        "randbytes",
    ):
        _patch_function("random", _fn, "DET_FORBIDDEN_RNG_API")

    # NumPy global randomness APIs (must route through RNGManager wrappers).
    for _fn in (
        "random",
        "rand",
        "randn",
        "randint",
        "choice",
        "shuffle",
        "permutation",
        "normal",
        "uniform",
        "default_rng",
    ):
        _patch_function("numpy.random", _fn, "DET_FORBIDDEN_RNG_API")

    # Torch global randomness APIs (must route through RNGManager wrappers).
    _patch_function("torch", "rand", "DET_FORBIDDEN_RNG_API")
    _patch_function("torch", "rand_like", "DET_FORBIDDEN_RNG_API")
    _patch_function("torch", "randn", "DET_FORBIDDEN_RNG_API")
    _patch_function("torch", "randint", "DET_FORBIDDEN_RNG_API")
    _patch_function("torch", "randperm", "DET_FORBIDDEN_RNG_API")
    _patch_function("torch", "bernoulli", "DET_FORBIDDEN_RNG_API")
    _patch_function("torch", "normal", "DET_FORBIDDEN_RNG_API")
    _patch_function("torch", "poisson", "DET_FORBIDDEN_RNG_API")
    _patch_function("torch", "multinomial", "DET_FORBIDDEN_RNG_API")
    _patch_torch_generator_constructor("DET_FORBIDDEN_RNG_API")


def _patch_function(module_name: str, attr: str, code: str) -> None:
    """Monkeypatch module.attr to fail fast in strict mode.

    Patched callables are stored so RNGManager can resolve original call targets.
    """
    try:
        mod = _import_optional(module_name)
        if mod is None:
            return
        fn = getattr(mod, attr, None)
        if fn is None or not callable(fn):
            return

        key = (str(module_name), str(attr))
        # Avoid double patch.
        if key in _PATCHED_FUNCTIONS:
            return
        _PATCHED_FUNCTIONS[key] = fn

        def _wrapped(*args, **kwargs):  # type: ignore[misc]
            _fail(code=code, symbol=f"{module_name}.{attr}", details="Forbidden API call in strict mode.")

        setattr(mod, attr, _wrapped)
    except Exception:
        # Never crash on guard install; strict failures should be explicit at call-time.
        return


def _patch_torch_generator_constructor(code: str) -> None:
    """Patch torch.Generator constructor in strict mode without replacing it with a plain function."""
    try:
        mod = _import_optional("torch")
        if mod is None:
            return
        cls = getattr(mod, "Generator", None)
        if cls is None:
            return
        key = ("torch", "Generator")
        if key in _PATCHED_FUNCTIONS:
            return
        _PATCHED_FUNCTIONS[key] = cls

        class _StrictBlockedTorchGenerator:
            __module__ = "torch"
            __name__ = "Generator"
            __qualname__ = "Generator"

            def __new__(cls_, *args, **kwargs):
                _fail(
                    code=code,
                    symbol="torch.Generator",
                    details="Direct torch.Generator construction is forbidden in strict mode.",
                )

        setattr(mod, "Generator", _StrictBlockedTorchGenerator)
    except Exception:
        return


def _import_optional(module_name: str) -> Optional[types.ModuleType]:
    try:
        __import__(module_name)
        return sys.modules.get(module_name)
    except Exception:
        return None


def resolve_original_callable(module_name: str, attr: str) -> Optional[Callable]:
    """Return pre-patch callable for (module, attr), if it was patched."""
    return _PATCHED_FUNCTIONS.get((str(module_name), str(attr)))


def _fail(code: str, symbol: str, details: str) -> None:
    msg = (
        "[DETERMINISM_VIOLATION]\n"
        f"code={code}\n"
        f"symbol={symbol}\n"
        f"details={details}\n"
    )
    raise RuntimeError(msg)
