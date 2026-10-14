"""Compatibility wrapper for the shared checkpoint-branching implementation."""

from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PATH = Path(__file__).resolve()
_MONOREPO_ROOT = _PATH.parents[4] if len(_PATH.parents) > 4 else _PROJECT_ROOT
if str(_MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_MONOREPO_ROOT))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments_shared.checkpoint_branching import *  # noqa


if __name__ == "__main__":
    raise SystemExit(main())
