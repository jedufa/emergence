"""EIEL simulation engine package.

The engine is in a transitional mixed layout: newer callers use
``engine.<package>`` while existing runtime modules still import peers such as
``evolution`` and ``simulation`` as top-level packages.  Make those peer
imports available whenever the canonical ``engine`` package is loaded, so
checkout-local commands do not need a hand-written ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path


_ENGINE_ROOT = str(Path(__file__).resolve().parent)
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)
