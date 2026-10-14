"""Emergence compatibility wrapper over the shared batch manifest runtime."""

from __future__ import annotations

import os
import sys

_THIS_FILE = os.path.abspath(__file__)
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(_THIS_FILE), os.pardir, os.pardir, os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from experiments_shared.manifest_runner import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
