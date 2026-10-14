"""engine.simulation.debug_io

Small, dependency-free helpers for writing high-volume debug traces to disk
instead of spamming stdout.

Why this exists
--------------
Humans love turning on every debug flag at once, then acting surprised when
their terminal becomes a slideshow. For determinism investigations we want
*all* the data, but we don't need it in the interactive console.

This module provides JSONL writers that default to disk-only.

Environment controls
--------------------
- EIEL_DEBUG_CONSOLE_MIRROR=1
    Mirror debug JSONL lines back to stdout (legacy behavior).
    Default: 0 (disk-only).

Notes
-----
- Writes are line-buffered and append-only.
- Failures are swallowed; debug logging must never crash the sim.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def _truthy_env(var_name: str, default: str = "0") -> bool:
    """Return True if an env var is set to a truthy value.

    Args:
        var_name: Environment variable name.
        default: Default string value if not set.

    Returns:
        bool: True if env var is one of (1, true, yes) case-insensitive.
    """

    v = (os.environ.get(var_name, default) or default).strip().lower()
    return v in ("1", "true", "yes")


def write_jsonl(
    run_dir: str,
    filename: str,
    payload: Dict[str, Any],
    *,
    prefix_tag: Optional[str] = None,
    ensure_dir: str = "debug",
) -> None:
    """Append one JSON line to <run_dir>/<ensure_dir>/<filename>.

    Args:
        run_dir: Simulation run directory.
        filename: Target JSONL filename (e.g., "birth_callsite_trace.jsonl").
        payload: JSON-serializable dictionary payload.
        prefix_tag: Optional legacy prefix that, if mirroring to console, is
            printed before the JSON.
        ensure_dir: Subdirectory under run_dir for debug outputs.
    """

    try:
        if not run_dir:
            return
        out_dir = os.path.join(str(run_dir), str(ensure_dir))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, str(filename))

        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

        # Legacy optional console mirror (off by default)
        if _truthy_env("EIEL_DEBUG_CONSOLE_MIRROR", "0"):
            if prefix_tag:
                print(prefix_tag, line)
            else:
                print(line)
    except Exception:
        # Debug logging must never crash the sim.
        return
