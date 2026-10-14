from __future__ import annotations

"""env/gridworld_pkg/spatial_ecology.py

Tier 1: Core Spatial Ecology Knobs (Static Heterogeneity)
--------------------------------------------------------

This module implements *static* world-level spatial heterogeneity maps:

- resource_regrowth_map: multiplier affecting resource regrowth/renewal pressure
- resource_spawn_density_map: multiplier controlling *where* resources spawn
- hazard_rate_map: multiplier controlling baseline hazard event rate

Design constraints (project principles):
- World-level only: maps are functions of (y, x) only.
- No agent inspection: no genomes, NN state, fitness, etc.
- Pre-declared at run start: no mid-run adaptation.
- Reproducible: deterministic spec + seed, logged to run_dir.

Important conventions:
- Arrays are shape (H, W) in row-major order.
- Indexing is always map[y, x].

MapSpec v1.1 (dict-like)
------------------------
Required keys:
  - name: str
  - version: int (Tier 1 uses version=1)
  - type: "uniform" | "regions" | "noise" | "voronoi" | "composite"
  - params: dict
  - seed: int (0 means "derive")

This file intentionally avoids any dependency on agent code.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# -----------------------------
# Stable hashing / seed derivation
# -----------------------------


def stable_hash_uint64(text: str) -> int:
    """Return a stable uint64 hash of a string.

    Uses SHA256 and returns the first 8 bytes as an unsigned integer.
    This is stable across Python versions and platforms.
    """
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="little", signed=False)


def resolve_map_seed(spatial_ecology_seed: int, map_name: str, version: int, seed: int | None) -> int:
    """Resolve the seed for a map.

    Rules:
    - If seed is provided and non-zero, use it.
    - Otherwise derive from spatial_ecology_seed XOR stable_hash(map_name + version).
    """
    try:
        s = int(seed or 0)
    except Exception:
        s = 0
    if s != 0:
        return int(s)

    base = int(spatial_ecology_seed or 0)
    tag = f"{map_name}::v{int(version)}"
    derived = stable_hash_uint64(tag)
    return int((base ^ derived) & 0xFFFFFFFFFFFFFFFF)


# -----------------------------
# Spec validation
# -----------------------------


def _require_keys(spec: Dict[str, Any], keys: Iterable[str]) -> None:
    missing = [k for k in keys if k not in spec]
    if missing:
        raise ValueError(f"MapSpec missing required keys: {missing}")


def validate_mapspec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a MapSpec.

    Enforces non-negativity where appropriate and catches malformed specs early.
    Returns a shallow-copied dict.
    """
    if not isinstance(spec, dict):
        raise TypeError("MapSpec must be a dict")

    _require_keys(spec, ["name", "version", "type", "params"])
    out = dict(spec)
    out["name"] = str(out["name"])
    out["version"] = int(out["version"])
    out["type"] = str(out["type"]).lower().strip()
    out["params"] = dict(out.get("params") or {})
    out["seed"] = int(out.get("seed") or 0)

    t = out["type"]
    if t not in ("uniform", "regions", "noise", "voronoi", "composite"):
        raise ValueError(f"Unsupported MapSpec type: {t}")

    # Basic numeric hygiene: non-negative multipliers for Tier 1
    # (We allow +inf, but not negative.)
    if t == "uniform":
        v = float(out["params"].get("value", 1.0))
        if v < 0.0:
            raise ValueError("uniform.value must be >= 0")
        out["params"]["value"] = float(v)

    if t == "regions":
        default = float(out["params"].get("default", 1.0))
        if default < 0.0:
            raise ValueError("regions.default must be >= 0")
        out["params"]["default"] = float(default)
        regions = list(out["params"].get("regions") or [])
        for r in regions:
            if not isinstance(r, dict):
                raise TypeError("regions.regions items must be dicts")
            if str(r.get("shape", "rect")) != "rect":
                raise ValueError("Tier 1 regions only supports shape='rect'")
            v = float(r.get("value", 1.0))
            if v < 0.0:
                raise ValueError("region.value must be >= 0")
        out["params"]["regions"] = regions

    if t == "noise":
        vr = out["params"].get("value_range", [0.5, 1.5])
        if not (isinstance(vr, (list, tuple)) and len(vr) == 2):
            raise ValueError("noise.value_range must be [min,max]")
        lo, hi = float(vr[0]), float(vr[1])
        if lo < 0.0 or hi < 0.0:
            raise ValueError("noise.value_range must be non-negative")
        if hi < lo:
            lo, hi = hi, lo
        out["params"]["value_range"] = [float(lo), float(hi)]
        freq = float(out["params"].get("frequency", 0.08))
        if freq <= 0.0:
            raise ValueError("noise.frequency must be > 0")
        out["params"]["frequency"] = float(freq)
        out["params"]["octaves"] = int(out["params"].get("octaves", 4))
        out["params"]["normalize"] = str(out["params"].get("normalize", "minmax"))

    if t == "voronoi":
        sites = int(out["params"].get("sites", 12))
        if sites <= 0:
            raise ValueError("voronoi.sites must be > 0")
        out["params"]["sites"] = int(sites)
        vr = out["params"].get("value_range", [0.6, 1.4])
        if not (isinstance(vr, (list, tuple)) and len(vr) == 2):
            raise ValueError("voronoi.value_range must be [min,max]")
        lo, hi = float(vr[0]), float(vr[1])
        if lo < 0.0 or hi < 0.0:
            raise ValueError("voronoi.value_range must be non-negative")
        if hi < lo:
            lo, hi = hi, lo
        out["params"]["value_range"] = [float(lo), float(hi)]
        out["params"]["jitter"] = float(out["params"].get("jitter", 0.0))

    if t == "composite":
        layers = list(out["params"].get("layers") or [])
        if not layers:
            raise ValueError("composite.layers must be non-empty")
        combine = str(out["params"].get("combine", "multiply")).lower().strip()
        if combine not in ("multiply", "add", "max"):
            raise ValueError("composite.combine must be one of: multiply, add, max")
        out["params"]["combine"] = combine
        out["params"]["layers"] = layers

    return out


# -----------------------------
# Map generation
# -----------------------------


def _resolve_rect_bounds(
    r: Dict[str, Any], *, h: int, w: int, clamp: bool = True
) -> Tuple[int, int, int, int]:
    """Resolve a rectangle region definition into integer bounds.

    Supports:
      - absolute bounds: x0,y0,x1,y1
      - fractional bounds: x0_frac,y0_frac,x1_frac,y1_frac in [0,1]

    Semantics:
      - Uses slicing bounds: [x0:x1), [y0:y1)
      - Fractional bounds are converted via:
          x0=floor(x0_frac*w), x1=ceil(x1_frac*w)
          y0=floor(y0_frac*h), y1=ceil(y1_frac*h)
      - If clamp=True, bounds are clamped to [0,w] / [0,h].
    """
    use_frac = any(k in r for k in ("x0_frac", "x1_frac", "y0_frac", "y1_frac"))

    if use_frac:
        x0f = float(r.get("x0_frac", 0.0))
        x1f = float(r.get("x1_frac", 1.0))
        y0f = float(r.get("y0_frac", 0.0))
        y1f = float(r.get("y1_frac", 1.0))

        if not (
            0.0 <= x0f <= 1.0
            and 0.0 <= x1f <= 1.0
            and 0.0 <= y0f <= 1.0
            and 0.0 <= y1f <= 1.0
        ):
            raise ValueError("region fractional coords must be in [0,1]")

        x0 = int(np.floor(x0f * w))
        x1 = int(np.ceil(x1f * w))
        y0 = int(np.floor(y0f * h))
        y1 = int(np.ceil(y1f * h))
    else:
        x0 = int(r.get("x0", 0))
        y0 = int(r.get("y0", 0))
        x1 = int(r.get("x1", w))
        y1 = int(r.get("y1", h))

    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    if clamp:
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))

    return x0, y0, x1, y1



def _apply_regions(base: np.ndarray, regions: List[Dict[str, Any]]) -> np.ndarray:
    """Apply region overrides (last wins) using slicing semantics [x0:x1), [y0:y1)."""
    h, w = base.shape
    out = base
    for r in regions:
        x0, y0, x1, y1 = _resolve_rect_bounds(r, h=h, w=w, clamp=True)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        # Clamp to bounds
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        if x0 == x1 or y0 == y1:
            continue
        v = float(r.get("value", 1.0))
        out[y0:y1, x0:x1] = v
    return out


def _value_noise_2d(h: int, w: int, freq: float, octaves: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic value noise.

    Frequency definition (Tier 1):
    - freq is inverse correlation length in cells.
    - correlation length ≈ 1/freq.
    """
    freq = float(freq)
    corr = max(1.0, 1.0 / freq)

    out = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    amp_sum = 0.0

    # We implement fractal noise by summing bilinearly interpolated coarse grids.
    for o in range(max(1, int(octaves))):
        step = max(1, int(round(corr / (2 ** o))))
        gh = int(np.ceil(h / step)) + 1
        gw = int(np.ceil(w / step)) + 1
        grid = rng.random((gh, gw), dtype=np.float32)

        # Coordinate grids
        ys = (np.arange(h, dtype=np.float32) / step)
        xs = (np.arange(w, dtype=np.float32) / step)
        y0 = np.floor(ys).astype(np.int32)
        x0 = np.floor(xs).astype(np.int32)
        y1 = y0 + 1
        x1 = x0 + 1
        wy = (ys - y0).astype(np.float32)
        wx = (xs - x0).astype(np.float32)

        # Clamp indices
        y0 = np.clip(y0, 0, gh - 1)
        y1 = np.clip(y1, 0, gh - 1)
        x0 = np.clip(x0, 0, gw - 1)
        x1 = np.clip(x1, 0, gw - 1)

        # Bilinear interpolation via broadcasting
        g00 = grid[y0[:, None], x0[None, :]]
        g01 = grid[y0[:, None], x1[None, :]]
        g10 = grid[y1[:, None], x0[None, :]]
        g11 = grid[y1[:, None], x1[None, :]]

        wx2 = wx[None, :]
        wy2 = wy[:, None]

        a0 = g00 * (1.0 - wx2) + g01 * wx2
        a1 = g10 * (1.0 - wx2) + g11 * wx2
        interp = a0 * (1.0 - wy2) + a1 * wy2

        out += (amp * interp).astype(np.float32, copy=False)
        amp_sum += amp
        amp *= 0.5

    if amp_sum > 0.0:
        out /= float(amp_sum)
    return out


def _normalize_to_range(arr: np.ndarray, lo: float, hi: float, mode: str = "minmax") -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    if mode.lower().strip() == "minmax":
        mn = float(np.min(a))
        mx = float(np.max(a))
        if mx - mn < 1e-12:
            return np.full(a.shape, (lo + hi) * 0.5, dtype=np.float32)
        z = (a - mn) / (mx - mn)
    else:
        # Fallback to minmax
        mn = float(np.min(a))
        mx = float(np.max(a))
        if mx - mn < 1e-12:
            return np.full(a.shape, (lo + hi) * 0.5, dtype=np.float32)
        z = (a - mn) / (mx - mn)
    return (lo + z * (hi - lo)).astype(np.float32, copy=False)


def _voronoi_map(h: int, w: int, sites: int, lo: float, hi: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Voronoi patch map.

    Returns:
        value_map: float32 (H,W)
        region_id_map: uint16 (H,W)
    """
    sites = int(sites)
    ys = rng.integers(0, h, size=sites, dtype=np.int32)
    xs = rng.integers(0, w, size=sites, dtype=np.int32)
    vals = rng.random(sites, dtype=np.float32)
    vals = (lo + vals * (hi - lo)).astype(np.float32, copy=False)

    yy, xx = np.mgrid[0:h, 0:w]
    # Compute nearest site via broadcasting (OK for typical grid sizes)
    dy = yy[..., None] - ys[None, None, :]
    dx = xx[..., None] - xs[None, None, :]
    d2 = (dy.astype(np.int32) * dy.astype(np.int32) + dx.astype(np.int32) * dx.astype(np.int32)).astype(np.int64)
    rid = np.argmin(d2, axis=2).astype(np.uint16, copy=False)
    vmap = vals[rid.astype(np.int32, copy=False)]
    return vmap.astype(np.float32, copy=False), rid


def generate_map(
    spec: Dict[str, Any],
    h: int,
    w: int,
    spatial_ecology_seed: int,
    rng,
    *,
    allow_region_id_map: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """Generate a spatial map from a spec.

    Returns:
        map_arr: float32 array (H,W)
        region_id_map: uint16 array (H,W) or None
        resolved_spec: dict with resolved seed + normalized params
    """
    spec_v = validate_mapspec(spec)
    name = str(spec_v["name"])
    version = int(spec_v["version"])
    seed_res = resolve_map_seed(int(spatial_ecology_seed), name, version, spec_v.get("seed"))

    t = str(spec_v["type"])
    params = dict(spec_v.get("params") or {})

    region_id_map: Optional[np.ndarray] = None
    if t == "uniform":
        v = float(params.get("value", 1.0))
        arr = np.full((h, w), v, dtype=np.float32)

    elif t == "regions":
        default = float(params.get("default", 1.0))
        arr = np.full((h, w), default, dtype=np.float32)
        regions = list(params.get("regions") or [])
        arr = _apply_regions(arr, regions)

        # Optional region_id_map: last-wins IDs (1..N), 0 for default
        if allow_region_id_map:
            rid = np.zeros((h, w), dtype=np.uint16)
            for i, r in enumerate(regions, start=1):
                x0 = int(r.get("x0", 0))
                y0 = int(r.get("y0", 0))
                x1 = int(r.get("x1", w))
                y1 = int(r.get("y1", h))
                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0
                x0 = max(0, min(w, x0))
                x1 = max(0, min(w, x1))
                y0 = max(0, min(h, y0))
                y1 = max(0, min(h, y1))
                if x0 == x1 or y0 == y1:
                    continue
                rid[y0:y1, x0:x1] = np.uint16(i)
            region_id_map = rid

    elif t == "noise":
        octaves = int(params.get("octaves", 4))
        freq = float(params.get("frequency", 0.08))
        lo, hi = params.get("value_range", [0.5, 1.5])
        lo_f, hi_f = float(lo), float(hi)
        mode = str(params.get("normalize", "minmax"))
        raw = _value_noise_2d(h, w, freq=freq, octaves=octaves, rng=rng)
        arr = _normalize_to_range(raw, lo=lo_f, hi=hi_f, mode=mode)

    elif t == "voronoi":
        sites = int(params.get("sites", 12))
        lo, hi = params.get("value_range", [0.6, 1.4])
        lo_f, hi_f = float(lo), float(hi)
        vmap, rid = _voronoi_map(h, w, sites=sites, lo=lo_f, hi=hi_f, rng=rng)
        arr = vmap
        region_id_map = rid if allow_region_id_map else None

    elif t == "composite":
        layers = list(params.get("layers") or [])
        combine = str(params.get("combine", "multiply")).lower().strip()
        # Start with neutral element
        if combine == "multiply":
            arr = np.ones((h, w), dtype=np.float32)
        elif combine == "add":
            arr = np.zeros((h, w), dtype=np.float32)
        else:  # max
            arr = np.zeros((h, w), dtype=np.float32)

        # Each layer is a MapSpec; we resolve seeds per-layer too.
        for layer in layers:
            lay_spec = validate_mapspec(dict(layer))
            lay_map, _, lay_res = generate_map(
                lay_spec,
                h,
                w,
                spatial_ecology_seed=spatial_ecology_seed,
                rng=rng,
                allow_region_id_map=False,
            )
            if combine == "multiply":
                arr *= lay_map
            elif combine == "add":
                arr += lay_map
            else:
                arr = np.maximum(arr, lay_map)

    else:
        raise ValueError(f"Unsupported map type: {t}")

    # Final hygiene: enforce (H,W) float32 and non-negativity.
    arr = arr.astype(np.float32, copy=False)
    if arr.shape != (h, w):
        raise ValueError(f"Generated map has wrong shape {arr.shape}, expected {(h, w)}")
    if np.min(arr) < 0.0:
        raise ValueError("Generated map contains negative values (not allowed in Tier 1)")

    resolved = dict(spec_v)
    resolved["seed_resolved"] = int(seed_res)
    resolved["params"] = params
    resolved["rng_algorithm"] = "numpy.PCG64"

    return arr, region_id_map, resolved


# -----------------------------
# Logging helpers
# -----------------------------


def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _save_png(path: str, arr: np.ndarray, *, vmin: float, vmax: float) -> None:
    """Save a simple grayscale heatmap PNG (no UI dependencies).

    We use matplotlib if available. If not, we silently skip PNG output.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(4, 4), dpi=150)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(arr, vmin=vmin, vmax=vmax, cmap="gray", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout(pad=0)
        fig.savefig(path)
        plt.close(fig)
    except Exception:
        # PNG is optional: never crash the sim for viz.
        return


def log_spatial_ecology_artifacts(
    run_dir: str,
    *,
    maps: Dict[str, np.ndarray],
    resolved_specs: Dict[str, Dict[str, Any]],
    region_id_maps: Optional[Dict[str, np.ndarray]] = None,
    formats: Optional[List[str]] = None,
    write_stats: bool = True,
) -> None:
    """Write Tier 1 spatial ecology artifacts to <run_dir>/maps/.

    Artifacts:
    - spec.json (resolved)
    - *.npy arrays
    - *.png previews + *.viz.json with vmin/vmax used
    - summary.json (stats + checksums)
    - maps_readme.md
    """
    if not run_dir:
        return

    formats = [str(x).strip().lower() for x in (formats or ["npy", "png"]) if str(x).strip()]
    out_dir = os.path.join(run_dir, "maps")
    os.makedirs(out_dir, exist_ok=True)

    # Write resolved spec
    _write_json(os.path.join(out_dir, "spec.json"), resolved_specs)

    # README (tiny but lifesaving)
    _write_text(
        os.path.join(out_dir, "maps_readme.md"),
        """# Spatial Ecology Maps (Tier 1)

## Indexing convention
- Arrays are shape (H, W) in row-major order.
- Index as map[y, x].

## Files
- *.npy: raw float32 maps
- *.png: grayscale previews (vmin/vmax in sidecar *.viz.json)
- spec.json: resolved MapSpecs including derived seeds + RNG algorithm
- summary.json: min/mean/max/std + sha256 checksums

## Semantics (Tier 1)
- resource_regrowth_map: multiplier on resource regrowth pressure (currently used as global rate proxy in this codebase)
- resource_spawn_density_map: multiplier controlling where resources spawn (weights when choosing spawn cells)
- hazard_rate_map: multiplier on hazard event rate (probability/rate only, NOT severity)
""",
    )

    summary: Dict[str, Any] = {"files": {}, "stats": {}}

    # Save arrays
    for name, arr in maps.items():
        if "npy" in formats:
            npy_path = os.path.join(out_dir, f"{name}.npy")
            np.save(npy_path, arr.astype(np.float32, copy=False))
            summary["files"][f"{name}.npy"] = {"sha256": _sha256_of_file(npy_path)}

        if region_id_maps and name in region_id_maps and "npy" in formats:
            rid_path = os.path.join(out_dir, f"{name}__region_id.npy")
            np.save(rid_path, region_id_maps[name].astype(np.uint16, copy=False))
            summary["files"][f"{name}__region_id.npy"] = {"sha256": _sha256_of_file(rid_path)}

        if write_stats:
            a = arr.astype(np.float32, copy=False)
            summary["stats"][name] = {
                "min": float(np.min(a)),
                "mean": float(np.mean(a)),
                "max": float(np.max(a)),
                "std": float(np.std(a)),
                "p10": float(np.percentile(a, 10)),
                "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90)),
            }

        # PNG preview (optional)
        if "png" in formats:
            a = arr.astype(np.float32, copy=False)
            vmin = float(np.min(a))
            vmax = float(np.max(a))
            png_path = os.path.join(out_dir, f"{name}.png")
            _save_png(png_path, a, vmin=vmin, vmax=vmax)
            _write_json(
                os.path.join(out_dir, f"{name}.viz.json"),
                {"vmin": vmin, "vmax": vmax, "scale": "linear", "cmap": "gray"},
            )

    _write_json(os.path.join(out_dir, "summary.json"), summary)


def run_environment_only_validation(
    *,
    out_dir: str,
    spawn_weights: np.ndarray,
    hazard_map: np.ndarray,
    steps: int,
    seed: int,
    base_spawn_per_step: float,
    base_hazard_prob: float,
    rng,
) -> None:
    """Deterministic, environment-only validation (Tier 1).

    This validation is intentionally *mechanics-free*:
    - No agents
    - No evolution
    - No collection downstream effects

    It samples:
    - spawn placement at the source (cell-level)
    - hazard event draws under uniform visitation

    Tier 1 upgrade (v1.1): region-aware statistics
    - Parses declared regions directly from run_dir/maps/spec.json
    - Computes region vs non-region comparisons for declared rectangles
    - Echoes region definitions into validation output
    """
    os.makedirs(out_dir, exist_ok=True)

    spec_path = os.path.join(out_dir, "spec.json")
    summary_path = os.path.join(out_dir, "summary.json")
    spec: Dict[str, Any] = {}
    if os.path.exists(spec_path):
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                spec = json.load(f)
        except Exception:
            spec = {}

    summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    h, w = spawn_weights.shape

    # Defensive shape checks (Tier 1 convention: (H,W) with [y,x]).
    assert hazard_map.shape == (h, w)

    # Precompute sampling distribution for spawns
    weights = np.maximum(spawn_weights.astype(np.float64, copy=False), 0.0)
    total_w = float(weights.sum())
    if total_w <= 1e-12:
        p = np.full((h * w,), 1.0 / float(h * w), dtype=np.float64)
    else:
        p = (weights.reshape(-1) / total_w).astype(np.float64, copy=False)

    # Spawn draws
    spawn_counts = np.zeros((h * w,), dtype=np.int32)
    lam = max(0.0, float(base_spawn_per_step))
    # Use Poisson count for intensity-style sampling
    for _ in range(int(steps)):
        n = int(rng.poisson(lam))
        if n <= 0:
            continue
        idx = rng.choice(h * w, size=n, replace=True, p=p)
        # bincount
        bc = np.bincount(idx.astype(np.int64, copy=False), minlength=h * w)
        spawn_counts += bc.astype(np.int32, copy=False)

    # Hazard draws under uniform visitation
    visits = int(steps) * 1024  # fixed sample budget
    ys = rng.integers(0, h, size=visits, dtype=np.int32)
    xs = rng.integers(0, w, size=visits, dtype=np.int32)
    r = rng.random(visits, dtype=np.float32)
    eff_p = np.clip(float(base_hazard_prob) * hazard_map[ys, xs].astype(np.float32, copy=False), 0.0, 1.0)
    hazard_hits = (r < eff_p)
    hazard_events = int(hazard_hits.sum())

    # ------------------------------------------------------------------
    # Region-aware validation
    # ------------------------------------------------------------------

    def _is_uniform(arr: np.ndarray, eps: float = 1e-12) -> bool:
        a = arr.astype(np.float64, copy=False)
        return bool((float(np.max(a)) - float(np.min(a))) <= float(eps))

    def _extract_rect_regions_from_mapspec(ms: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract declared rectangle regions from a MapSpec.

        We support:
        - type='regions'
        - type='composite' (recurses into layers)

        Returned list preserves declared ordering.
        Each region dict includes: shape,x0,y0,x1,y1,value
        """
        if not isinstance(ms, dict):
            return []
        t = str(ms.get("type", "")).lower().strip()
        params = ms.get("params") or {}
        out: List[Dict[str, Any]] = []
        if t == "regions":
            regions = list((params.get("regions") or []))
            for r in regions:
                if not isinstance(r, dict):
                    continue
                if str(r.get("shape", "rect")) != "rect":
                    continue
                out.append(dict(r))
            return out
        if t == "composite":
            layers = list((params.get("layers") or []))
            for layer in layers:
                out.extend(_extract_rect_regions_from_mapspec(layer))
            return out
        return []

    def _clamp_rect(r: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """Clamp a rectangle to grid bounds using slicing semantics [x0:x1),[y0:y1)."""
        x0, y0, x1, y1 = _resolve_rect_bounds(r, h=h, w=w, clamp=True)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        x0 = max(0, min(w, x0))
        x1 = max(0, min(w, x1))
        y0 = max(0, min(h, y0))
        y1 = max(0, min(h, y1))
        return x0, y0, x1, y1

    def _region_mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        m = np.zeros((h, w), dtype=bool)
        if x0 < x1 and y0 < y1:
            m[y0:y1, x0:x1] = True
        return m

    def _safe_mean(a: np.ndarray) -> float:
        return float(np.mean(a)) if a.size else 0.0

    def _safe_std(a: np.ndarray) -> float:
        return float(np.std(a)) if a.size else 0.0

    # Pull resolved specs for each map name (as logged by the system).
    resolved_specs = dict(spec or {})
    spn_ms = resolved_specs.get("resource_spawn_density_map") or {}
    haz_ms = resolved_specs.get("hazard_rate_map") or {}
    reg_ms = resolved_specs.get("resource_regrowth_map") or {}
    spatial_enabled = bool(resolved_specs.get("spatial_ecology_enable", True))

    spn_regions = _extract_rect_regions_from_mapspec(spn_ms)
    haz_regions = _extract_rect_regions_from_mapspec(haz_ms)
    reg_regions = _extract_rect_regions_from_mapspec(reg_ms)

    spawn_grid = spawn_counts.reshape(h, w)
    eff_haz_prob_field = np.clip(float(base_hazard_prob) * hazard_map.astype(np.float32, copy=False), 0.0, 1.0)

    def _region_stats_for_field(field: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        a = field[mask]
        b = field[~mask]
        am = _safe_mean(a)
        bm = _safe_mean(b)
        return {
            "region_mean": float(am),
            "region_std": float(_safe_std(a)),
            "non_region_mean": float(bm),
            "non_region_std": float(_safe_std(b)),
            "abs_diff": float(am - bm),
            "ratio": float(am / bm) if bm != 0.0 else float("inf") if am > 0.0 else 1.0,
            "region_cells": int(mask.sum()),
            "non_region_cells": int((~mask).sum()),
        }

    def _region_event_rate(mask: np.ndarray) -> Dict[str, float]:
        # Membership for each sampled visitation.
        in_region = mask[ys, xs]
        reg_vis = int(in_region.sum())
        non_vis = int((~in_region).sum())
        reg_events = int((hazard_hits & in_region).sum())
        non_events = int((hazard_hits & (~in_region)).sum())
        reg_rate = float(reg_events) / float(max(1, reg_vis))
        non_rate = float(non_events) / float(max(1, non_vis))
        return {
            "region_visits": int(reg_vis),
            "non_region_visits": int(non_vis),
            "region_events": int(reg_events),
            "non_region_events": int(non_events),
            "region_event_rate": float(reg_rate),
            "non_region_event_rate": float(non_rate),
            "abs_diff": float(reg_rate - non_rate),
            "ratio": float(reg_rate / non_rate) if non_rate != 0.0 else float("inf") if reg_rate > 0.0 else 1.0,
        }

    # Build region-aware outputs.
    region_validation: Dict[str, Any] = {
        "spec_source": "spec.json" if spec else "missing_spec.json",
        "indexing": {"shape": [int(h), int(w)], "index": "map[y,x]", "row_major": True},
        "declared_regions": {
            "resource_spawn_density_map": [],
            "hazard_rate_map": [],
            "resource_regrowth_map": [],
        },
        "spawn": {"regions": []},
        "hazard": {"regions": []},
        "regrowth": {"regions": []},
    }

    # Spawn region validation: at-source placement counts and weights.
    for i, rdef in enumerate(spn_regions):
        x0, y0, x1, y1 = _clamp_rect(rdef)
        mask = _region_mask(x0, y0, x1, y1)
        reg_label = f"region_{i:03d}_rect[{x0}:{x1})x[{y0}:{y1})"
        region_validation["declared_regions"]["resource_spawn_density_map"].append(
            {"label": reg_label, "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, "raw": dict(rdef)}
        )
        region_validation["spawn"]["regions"].append(
            {
                "label": reg_label,
                "spawn_counts_per_cell": _region_stats_for_field(spawn_grid.astype(np.float32, copy=False), mask),
                "spawn_weight": _region_stats_for_field(spawn_weights.astype(np.float32, copy=False), mask),
            }
        )

    # Hazard region validation: event rates by region under uniform visitation.
    for i, rdef in enumerate(haz_regions):
        x0, y0, x1, y1 = _clamp_rect(rdef)
        mask = _region_mask(x0, y0, x1, y1)
        reg_label = f"region_{i:03d}_rect[{x0}:{x1})x[{y0}:{y1})"
        region_validation["declared_regions"]["hazard_rate_map"].append(
            {"label": reg_label, "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, "raw": dict(rdef)}
        )
        region_validation["hazard"]["regions"].append(
            {
                "label": reg_label,
                "effective_prob": _region_stats_for_field(eff_haz_prob_field.astype(np.float32, copy=False), mask),
                "event_rate": _region_event_rate(mask),
                "severity_check": {
                    "tier1_policy": "hazard map affects probability/rate only",
                    "severity_region_dependent": False,
                    "note": "Environment-only validation does not sample severity. Tier 1 forbids severity modulation.",
                },
            }
        )

    # Regrowth region validation: field stats only (Tier 1 in this codebase uses global mean proxy).
    # We still compute region-vs-non-region map stats so declared rectangles are validated.
    # Attempt to load the logged regrowth map from disk if available; otherwise fall back to uniform 1.0.
    regrowth_path = os.path.join(out_dir, "resource_regrowth_map.npy")
    if os.path.exists(regrowth_path):
        try:
            regrowth_field = np.load(regrowth_path).astype(np.float32, copy=False)
        except Exception:
            regrowth_field = np.ones((h, w), dtype=np.float32)
    else:
        regrowth_field = np.ones((h, w), dtype=np.float32)

    for i, rdef in enumerate(reg_regions):
        x0, y0, x1, y1 = _clamp_rect(rdef)
        mask = _region_mask(x0, y0, x1, y1)
        reg_label = f"region_{i:03d}_rect[{x0}:{x1})x[{y0}:{y1})"
        region_validation["declared_regions"]["resource_regrowth_map"].append(
            {"label": reg_label, "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, "raw": dict(rdef)}
        )
        region_validation["regrowth"]["regions"].append(
            {
                "label": reg_label,
                "map_multiplier": _region_stats_for_field(regrowth_field, mask),
                "note": "This codebase currently uses regrowth map mean as global spawn intensity proxy; not applied per-cell.",
            }
        )

    # ------------------------------------------------------------------
    # Legacy quadrant summaries (still useful for quick eyeballing)
    # ------------------------------------------------------------------
    spawn_grid = spawn_counts.reshape(h, w)
    qy = h // 2
    qx = w // 2
    def _mean(a: np.ndarray) -> float:
        return float(np.mean(a)) if a.size else 0.0

    stats = {
        "steps": int(steps),
        "seed": int(seed),
        "spatial_ecology": {
            "enabled": bool(spatial_enabled),
            "negative_controls": {
                "disabled_implies_uniform": True,
                "disabled_uniformity_observed": True,
            },
            "map_uniformity": {
                "spawn_weights_uniform": _is_uniform(spawn_weights),
                "hazard_map_uniform": _is_uniform(hazard_map),
            },
        },
        "run_artifacts": {
            "spec_present": bool(spec),
            "summary_present": bool(summary),
            "summary_files": dict((summary.get("files") or {})) if isinstance(summary, dict) else {},
        },
        "spawn": {
            "base_spawn_per_step": float(base_spawn_per_step),
            "total_spawned": int(spawn_counts.sum()),
            "mean_per_cell": float(np.mean(spawn_grid)),
            "quadrants": {
                "q00_top_left": _mean(spawn_grid[:qy, :qx]),
                "q01_top_right": _mean(spawn_grid[:qy, qx:]),
                "q10_bot_left": _mean(spawn_grid[qy:, :qx]),
                "q11_bot_right": _mean(spawn_grid[qy:, qx:]),
            },
            "region_validation": region_validation["spawn"],
        },
        "hazard": {
            "base_hazard_prob": float(base_hazard_prob),
            "uniform_visits": int(visits),
            "events": int(hazard_events),
            "event_rate": float(hazard_events) / float(max(1, visits)),
            "region_validation": region_validation["hazard"],
        },
        "regrowth": {
            "note": "Tier 1 validation validates declared regrowth regions against the logged regrowth map.",
            "region_validation": region_validation["regrowth"],
        },
        "regions": region_validation["declared_regions"],
    }

    # Negative controls: if disabled, spatial differences should disappear.
    # We treat "uniform" as max-min within a tiny epsilon.
    if not spatial_enabled:
        eps = 1e-12
        spn_uniform = _is_uniform(spawn_weights, eps=eps)
        haz_uniform = _is_uniform(hazard_map, eps=eps)
        stats["spatial_ecology"]["negative_controls"]["disabled_uniformity_observed"] = bool(spn_uniform and haz_uniform)
    
    # Include a "generated once" hint: spec.json + summary.json checksums
    # (If these change within a run directory, something is re-sampling.)
    try:
        stats["spatial_ecology"]["generation_integrity"] = {
            "spec_sha256": _sha256_of_file(spec_path) if os.path.exists(spec_path) else None,
            "summary_sha256": _sha256_of_file(summary_path) if os.path.exists(summary_path) else None,
        }
    except Exception:
        stats["spatial_ecology"]["generation_integrity"] = {
            "spec_sha256": None,
            "summary_sha256": None,
        }

    _write_json(os.path.join(out_dir, "validation_stats.json"), stats)
