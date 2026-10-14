from __future__ import annotations
"""simulation/control_plane_pkg/util_json.py

Deterministic JSON helpers for Phase 2 (Control Plane + WAL).

Implements GF-JSON-01 (canonical JSON bytes) and the hashing requirements for:
- command_payload_hash (CMD-WAL-02)
- event payload_hash

Notes
-----
- Determinism-critical: do not use pretty printing, locale formatting, or unordered dicts.
- Float normalization: per TSD, floats are normalized before hashing/logging:
  - cast to float32
  - rounded to 1e-8
  - NaN/Inf rejected
"""

from dataclasses import is_dataclass, asdict
from collections import deque
import json
import math
from typing import Any, Dict, List, Union

import numpy as np


JsonObj = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


def _normalize_floats(obj: Any) -> Any:
    """Recursively normalize floats per GF-FLOAT-01 for hashed/logged fields."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError("NaN/Inf forbidden in hashed/logged fields")
        f32 = np.float32(obj).item()
        # Round to 1e-8 (after float32 cast) for stable hashing across minor variations.
        return float(round(float(f32), 8))
    if isinstance(obj, (np.floating,)):
        return _normalize_floats(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _normalize_floats(obj.tolist())
    if isinstance(obj, str):
        return obj
    if is_dataclass(obj):
        return _normalize_floats(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _normalize_floats(v) for k, v in obj.items()}
    if isinstance(obj, deque):
        return [_normalize_floats(v) for v in list(obj)]
    if isinstance(obj, (list, tuple)):
        return [_normalize_floats(v) for v in obj]
    # Best-effort: try to serialize common objects by __dict__.
    if hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__"), dict):
        return _normalize_floats(dict(obj.__dict__))
    raise TypeError(f"unsupported type for canonical json: {type(obj)}")


def canonical_json_bytes(obj: Any) -> bytes:
    """Return canonical JSON bytes for obj (UTF-8, sorted keys, minified)."""
    norm = _normalize_floats(obj)
    s = json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return s.encode("utf-8")



def canonical_json_bytes_lossless(obj: Any) -> bytes:
    """Return canonical JSON bytes for obj WITHOUT lossy float normalization.

    This is intended for checkpoint bundles where resume must reproduce the exact
    in-memory numeric state of a fresh run.

    Guarantees:
      - UTF-8
      - sorted keys
      - minified separators
      - rejects NaN/Inf (float or numpy floating)
      - encodes numpy ndarrays with dtype+shape so restore can reconstruct bit-stable arrays

    Notes
    -----
    - Python's json module cannot serialize numpy types directly.
    - We *do not* quantize floats here. The goal is exact replay, not stable hashing.
    """

    def _prepare(x: Any) -> Any:
        # Primitive JSON types
        if x is None or isinstance(x, (bool, int, str)):
            return x

        # Floats: reject NaN/Inf but preserve value
        if isinstance(x, float):
            if not math.isfinite(x):
                raise ValueError("NaN/Inf forbidden in canonical JSON (lossless)")
            return x

        # Numpy scalar types -> python scalars (exact for ints/bools; float keeps value)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, (np.floating,)):
            xf = float(x)
            if not math.isfinite(xf):
                raise ValueError("NaN/Inf forbidden in canonical JSON (lossless)")
            return xf

        # Numpy arrays: encode with dtype+shape so load can restore exact dtype/shape.
        if isinstance(x, np.ndarray):
            # Fast path: tolist() already produces nested python numbers.
            data = x.tolist()
            # Reject non-finite anywhere in the nested list deterministically.
            def _reject_nested(v: Any) -> None:
                if v is None or isinstance(v, (bool, int, str)):
                    return
                if isinstance(v, float):
                    if not math.isfinite(v):
                        raise ValueError("NaN/Inf forbidden in canonical JSON (lossless)")
                    return
                if isinstance(v, list):
                    for vv in v:
                        _reject_nested(vv)
                    return
                # Unexpected types should never appear from ndarray.tolist()
                raise TypeError(f"unsupported ndarray element type for canonical json: {type(v)}")

            _reject_nested(data)
            return {"__ndarray__": data, "dtype": str(x.dtype), "shape": list(x.shape)}

        # Dataclasses -> dict
        if is_dataclass(x):
            return _prepare(asdict(x))

        # Dict / list / tuple / deque
        if isinstance(x, dict):
            return {str(k): _prepare(v) for k, v in x.items()}
        if isinstance(x, deque):
            return [_prepare(v) for v in list(x)]
        if isinstance(x, (list, tuple)):
            return [_prepare(v) for v in x]

        # Best-effort: common objects by __dict__
        if hasattr(x, "__dict__") and isinstance(getattr(x, "__dict__"), dict):
            return _prepare(dict(x.__dict__))

        # Let json raise TypeError deterministically.
        return x

    prepared = _prepare(obj)
    s = json.dumps(prepared, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return s.encode("utf-8")



def canonical_json_str(obj: Any) -> str:
    """Return canonical JSON string (same as canonical_json_bytes but as str)."""
    return canonical_json_bytes(obj).decode("utf-8")
