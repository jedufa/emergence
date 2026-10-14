"""Compatibility wrapper for shared static determinism scanning."""

from pathlib import Path
import importlib
import sys

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SHARED_DIR = _REPO_ROOT / "experiments" / "shared"
_loaded = importlib.import_module("experiments_shared")
_paths = list(_loaded.__dict__.get("__path__", []))
if str(_SHARED_DIR) not in _paths:
    _loaded.__path__ = [str(_SHARED_DIR), *_paths]
    _spec = getattr(_loaded, "__spec__", None)
    if _spec is not None and getattr(_spec, "submodule_search_locations", None) is not None:
        _spec.submodule_search_locations = list(_loaded.__path__)

from experiments_shared.determinism.static_rng_scan import *  # noqa


if __name__ == "__main__":
    raise SystemExit(main())
