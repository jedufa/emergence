"""tests/conftest.py

Test harness for engine modules.

Pytest runs from repo root; engine is not packaged by setuptools, so we
add it to sys.path for imports like `from simulation...`.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVO = ROOT / "engine"
if str(EVO) not in sys.path:
    sys.path.insert(0, str(EVO))
