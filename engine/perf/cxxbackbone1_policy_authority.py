from __future__ import annotations

"""CXXBACKBONE-1 production-layout policy/Hmem/action-cache native contract."""

from dataclasses import dataclass
import ctypes
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cxxbackbone1_policy_authority_v1"


@dataclass(frozen=True)
class PolicyAuthorityInputV1:
    policy_kind: np.ndarray
    row_slots: np.ndarray
    observations: np.ndarray
    drives: np.ndarray
    gain: np.ndarray
    aggression: np.ndarray
    temperature: np.ndarray
    effective_memory_dim: np.ndarray
    mlp_pack: np.ndarray
    gru_pack: np.ndarray
    hmem: np.ndarray
    action_cache_cursor: np.ndarray
    cache_stride: int
    mlp_hidden: int
    gru_hidden: int
    actions: int
    w_hunger_gather: float = 0.35
    w_pain_rest: float = 0.25
    w_fatigue_rest: float = 0.15
    w_pain_attack: float = 0.20
    w_fear_attack: float = 0.10


@dataclass(frozen=True)
class PolicyAuthorityResultV1:
    actions: np.ndarray
    logits: np.ndarray
    hmem_after: np.ndarray
    action_cache_cursor_after: np.ndarray
    timing: Mapping[str, float]
    executed_native: bool
    hidden_fallback: bool


def make_policy_authority_fixture(
    *,
    rows: int = 10,
    slots: int = 14,
    obs_cols: int = 9,
    mlp_hidden: int = 8,
    gru_hidden: int = 64,
    seed: int = 191,
) -> PolicyAuthorityInputV1:
    rng = np.random.default_rng(int(seed))
    actions = 7
    policy_kind = np.asarray([0 if i % 2 == 0 else 1 for i in range(rows)], dtype=np.int32)
    row_slots = np.asarray([(i * 3) % slots for i in range(rows)], dtype=np.int32)
    mlp_stride = mlp_hidden * obs_cols + mlp_hidden + mlp_hidden * mlp_hidden + mlp_hidden + actions * mlp_hidden + actions
    gru_stride = gru_hidden * obs_cols + gru_hidden + 3 * gru_hidden * gru_hidden + 3 * gru_hidden + 3 * gru_hidden * gru_hidden + 3 * gru_hidden + actions * gru_hidden + actions
    mlp_pack = rng.normal(0.0, 0.05, size=(slots, mlp_stride)).astype(np.float32)
    gru_pack = rng.normal(0.0, 0.035, size=(slots, gru_stride)).astype(np.float32)
    hmem = rng.normal(0.0, 0.02, size=(slots, gru_hidden)).astype(np.float32)
    return PolicyAuthorityInputV1(
        policy_kind=policy_kind,
        row_slots=row_slots,
        observations=rng.normal(0.0, 0.2, size=(rows, obs_cols)).astype(np.float32),
        drives=rng.random((rows, 4), dtype=np.float32) * np.float32(0.25),
        gain=np.full((rows,), 1.0, dtype=np.float32),
        aggression=np.zeros((rows,), dtype=np.float32),
        temperature=np.ones((rows,), dtype=np.float32),
        effective_memory_dim=np.asarray([0 if k == 0 else (8 + (i % 5) * 4) for i, k in enumerate(policy_kind)], dtype=np.int32),
        mlp_pack=mlp_pack,
        gru_pack=gru_pack,
        hmem=hmem,
        action_cache_cursor=np.arange(slots, dtype=np.int32) % 5,
        cache_stride=16,
        mlp_hidden=mlp_hidden,
        gru_hidden=gru_hidden,
        actions=actions,
    )


def run_policy_authority_python_reference(inputs: PolicyAuthorityInputV1) -> PolicyAuthorityResultV1:
    t0 = time.perf_counter()
    rows = int(inputs.policy_kind.shape[0])
    actions = int(inputs.actions)
    logits = np.zeros((rows, actions), dtype=np.float32)
    out_actions = np.zeros((rows,), dtype=np.int32)
    hmem_after = np.asarray(inputs.hmem, dtype=np.float32).copy()
    cursor = np.asarray(inputs.action_cache_cursor, dtype=np.int32).copy()
    for row in range(rows):
        slot = int(inputs.row_slots[row])
        if int(inputs.policy_kind[row]) == 0:
            row_logits = _mlp_logits(inputs, row, slot)
        else:
            row_logits, h_next = _gru_logits_hmem(inputs, row, slot)
            hmem_after[slot, :] = h_next
        row_logits[6] = np.float32(row_logits[6] + np.float32(inputs.aggression[row]))
        row_logits = (row_logits / np.maximum(np.float32(inputs.temperature[row]), np.float32(0.05))).astype(np.float32, copy=False)
        d = np.asarray(inputs.drives[row], dtype=np.float32)
        gain = np.float32(inputs.gain[row])
        row_logits[4] = np.float32(row_logits[4] + gain * (np.float32(inputs.w_hunger_gather) * d[0]))
        row_logits[5] = np.float32(row_logits[5] + gain * (np.float32(inputs.w_pain_rest) * d[1] + np.float32(inputs.w_fatigue_rest) * d[2]))
        row_logits[6] = np.float32(row_logits[6] - gain * (np.float32(inputs.w_pain_attack) * d[1] + np.float32(inputs.w_fear_attack) * d[3]))
        logits[row, :] = row_logits
        out_actions[row] = np.int32(int(np.argmax(row_logits)))
        cursor[slot] = np.int32((int(cursor[slot]) + 1) % int(inputs.cache_stride)) if int(inputs.cache_stride) > 0 else np.int32(int(cursor[slot]) + 1)
    return PolicyAuthorityResultV1(
        actions=out_actions,
        logits=logits,
        hmem_after=np.ascontiguousarray(hmem_after),
        action_cache_cursor_after=np.ascontiguousarray(cursor),
        timing={"wall_s": float(time.perf_counter() - t0)},
        executed_native=False,
        hidden_fallback=False,
    )


def compile_policy_authority_kernel(build_dir: Path | str, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    build = Path(build_dir)
    source_path = ROOT / "engine" / "perf" / "native" / "cxxbackbone1_policy_authority_kernel.cpp"
    library_path = build / "libcxxbackbone1_policy_authority.so"
    if not compiler:
        return {"ok": False, "reason": "g++ not found", "library_path": "", "command": [], "source_path": str(source_path)}
    build.mkdir(parents=True, exist_ok=True)
    cmd = [str(compiler), "-O3", "-std=c++17", "-shared", "-fPIC", str(source_path), "-o", str(library_path)]
    cp = subprocess.run(cmd, text=True, capture_output=True, check=False)
    return {
        "ok": cp.returncode == 0,
        "reason": "" if cp.returncode == 0 else "compile failed",
        "command": cmd,
        "returncode": int(cp.returncode),
        "stdout_tail": cp.stdout[-4000:],
        "stderr_tail": cp.stderr[-4000:],
        "source_path": str(source_path),
        "library_path": str(library_path) if cp.returncode == 0 else "",
    }


def load_policy_authority_backend(library_path: Path | str):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxbackbone1_policy_authority_v1
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)
    fn.argtypes = [
        i32p, i32p, f32p, f32p, f32p, f32p, f32p, i32p, f32p, ctypes.c_int64, f32p, ctypes.c_int64,
        f32p, i32p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        i32p, f32p, f32p, i32p,
    ]
    fn.restype = ctypes.c_int

    def _call(inputs: PolicyAuthorityInputV1) -> PolicyAuthorityResultV1:
        prep_t0 = time.perf_counter()
        rows = int(inputs.policy_kind.shape[0])
        slots = int(inputs.hmem.shape[0])
        actions = int(inputs.actions)
        out_actions = np.zeros((rows,), dtype=np.int32)
        out_logits = np.zeros((rows, actions), dtype=np.float32)
        out_hmem = np.zeros_like(np.asarray(inputs.hmem, dtype=np.float32))
        out_cursor = np.zeros((slots,), dtype=np.int32)
        prep_s = time.perf_counter() - prep_t0
        compute_t0 = time.perf_counter()
        status = fn(
            np.ascontiguousarray(inputs.policy_kind, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.row_slots, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.observations, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.drives, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.gain, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.aggression, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.temperature, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.effective_memory_dim, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.mlp_pack, dtype=np.float32).ctypes.data_as(f32p),
            ctypes.c_int64(int(inputs.mlp_pack.shape[1])),
            np.ascontiguousarray(inputs.gru_pack, dtype=np.float32).ctypes.data_as(f32p),
            ctypes.c_int64(int(inputs.gru_pack.shape[1])),
            np.ascontiguousarray(inputs.hmem, dtype=np.float32).ctypes.data_as(f32p),
            np.ascontiguousarray(inputs.action_cache_cursor, dtype=np.int32).ctypes.data_as(i32p),
            ctypes.c_int64(int(inputs.cache_stride)),
            ctypes.c_int64(rows),
            ctypes.c_int64(slots),
            ctypes.c_int64(int(inputs.observations.shape[1])),
            ctypes.c_int64(int(inputs.mlp_hidden)),
            ctypes.c_int64(int(inputs.gru_hidden)),
            ctypes.c_int64(actions),
            ctypes.c_float(float(inputs.w_hunger_gather)),
            ctypes.c_float(float(inputs.w_pain_rest)),
            ctypes.c_float(float(inputs.w_fatigue_rest)),
            ctypes.c_float(float(inputs.w_pain_attack)),
            ctypes.c_float(float(inputs.w_fear_attack)),
            out_actions.ctypes.data_as(i32p),
            out_logits.ctypes.data_as(f32p),
            out_hmem.ctypes.data_as(f32p),
            out_cursor.ctypes.data_as(i32p),
        )
        compute_s = time.perf_counter() - compute_t0
        if int(status) != 0:
            raise RuntimeError(f"CXXBACKBONE1 policy authority kernel failed with status {int(status)}")
        return PolicyAuthorityResultV1(
            actions=out_actions,
            logits=out_logits,
            hmem_after=out_hmem,
            action_cache_cursor_after=out_cursor,
            timing={"setup_s": float(prep_s), "backend_compute_s": float(compute_s), "wall_s": float(prep_s + compute_s)},
            executed_native=True,
            hidden_fallback=False,
        )

    return _call


def compare_policy_authority_results(expected: PolicyAuthorityResultV1, observed: PolicyAuthorityResultV1, *, atol: float = 0.0) -> dict[str, Any]:
    for name in ("actions", "logits", "hmem_after", "action_cache_cursor_after"):
        a = np.asarray(getattr(expected, name))
        b = np.asarray(getattr(observed, name))
        if a.shape != b.shape:
            return {"ok": False, "first_divergence": {"field": name, "expected_shape": list(a.shape), "observed_shape": list(b.shape)}}
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            if bool(np.any(diff > float(atol))):
                idx = tuple(int(x) for x in np.argwhere(diff > float(atol))[0])
                return {"ok": False, "first_divergence": {"field": name, "index": list(idx), "expected": float(a[idx]), "observed": float(b[idx]), "abs_delta": float(diff[idx])}}
        elif not bool(np.array_equal(a, b)):
            idx = tuple(int(x) for x in np.argwhere(a != b)[0])
            return {"ok": False, "first_divergence": {"field": name, "index": list(idx), "expected": int(a[idx]), "observed": int(b[idx])}}
    return {"ok": True, "first_divergence": None, "state_digest": result_digest(observed)}


def result_digest(result: PolicyAuthorityResultV1) -> str:
    return _json_digest(
        {
            "actions": _array_digest(result.actions),
            "logits": _array_digest(result.logits),
            "hmem_after": _array_digest(result.hmem_after),
            "action_cache_cursor_after": _array_digest(result.action_cache_cursor_after),
        }
    )


def authority_surfaces_payload(parity: Mapping[str, Any]) -> dict[str, Any]:
    exact = bool(parity.get("ok"))
    return {
        "schema_version": SCHEMA_VERSION,
        "native_authority_surfaces": {
            "action_cache": exact,
            "hmem": exact,
            "mlp_gru_policy_banks": exact,
            "policy_row_mappings": exact,
        },
        "unit_parity": "exact" if exact else "blocked",
        "first_divergence": parity.get("first_divergence"),
        "live_integrated": False,
        "checkpoint_integrated": False,
    }


def _mlp_logits(inputs: PolicyAuthorityInputV1, row: int, slot: int) -> np.ndarray:
    h = int(inputs.mlp_hidden)
    o = int(inputs.observations.shape[1])
    a = int(inputs.actions)
    pack = np.asarray(inputs.mlp_pack[slot], dtype=np.float32)
    off = 0
    w1 = pack[off : off + h * o].reshape(h, o); off += h * o
    b1 = pack[off : off + h]; off += h
    w2 = pack[off : off + h * h].reshape(h, h); off += h * h
    b2 = pack[off : off + h]; off += h
    w3 = pack[off : off + a * h].reshape(a, h); off += a * h
    b3 = pack[off : off + a]
    h1 = np.zeros((h,), dtype=np.float32)
    h2 = np.zeros((h,), dtype=np.float32)
    logits = np.zeros((a,), dtype=np.float32)
    x = np.asarray(inputs.observations[row], dtype=np.float32)
    for yy in range(h):
        acc = np.float32(b1[yy])
        for xx in range(o):
            acc = np.float32(acc + np.float32(w1[yy, xx] * x[xx]))
        h1[yy] = np.float32(math.tanh(float(acc)))
    for yy in range(h):
        acc = np.float32(b2[yy])
        for xx in range(h):
            acc = np.float32(acc + np.float32(w2[yy, xx] * h1[xx]))
        h2[yy] = np.float32(math.tanh(float(acc)))
    for yy in range(a):
        acc = np.float32(b3[yy])
        for xx in range(h):
            acc = np.float32(acc + np.float32(w3[yy, xx] * h2[xx]))
        logits[yy] = acc
    return logits


def _gru_logits_hmem(inputs: PolicyAuthorityInputV1, row: int, slot: int) -> tuple[np.ndarray, np.ndarray]:
    h = int(inputs.gru_hidden)
    o = int(inputs.observations.shape[1])
    a = int(inputs.actions)
    pack = np.asarray(inputs.gru_pack[slot], dtype=np.float32)
    off = 0
    enc_w = pack[off : off + h * o].reshape(h, o); off += h * o
    enc_b = pack[off : off + h]; off += h
    wih = pack[off : off + 3 * h * h].reshape(3 * h, h); off += 3 * h * h
    bih = pack[off : off + 3 * h]; off += 3 * h
    whh = pack[off : off + 3 * h * h].reshape(3 * h, h); off += 3 * h * h
    bhh = pack[off : off + 3 * h]; off += 3 * h
    head_w = pack[off : off + a * h].reshape(a, h); off += a * h
    head_b = pack[off : off + a]
    x = np.asarray(inputs.observations[row], dtype=np.float32)
    mem_dim = max(0, min(h, int(inputs.effective_memory_dim[row])))
    h_prev = np.asarray(inputs.hmem[slot], dtype=np.float32).copy()
    h_prev[mem_dim:] = np.float32(0.0)
    z_in = np.zeros((h,), dtype=np.float32)
    h_next = np.zeros((h,), dtype=np.float32)
    for yy in range(h):
        acc = np.float32(enc_b[yy])
        for xx in range(o):
            acc = np.float32(acc + np.float32(enc_w[yy, xx] * x[xx]))
        z_in[yy] = np.float32(math.tanh(float(acc)))
    for yy in range(h):
        vals = []
        for base, weights, bias, source in ((0, wih, bih, z_in), (h, wih, bih, z_in), (2 * h, wih, bih, z_in), (0, whh, bhh, h_prev), (h, whh, bhh, h_prev), (2 * h, whh, bhh, h_prev)):
            acc = np.float32(bias[base + yy])
            for xx in range(h):
                acc = np.float32(acc + np.float32(weights[base + yy, xx] * source[xx]))
            vals.append(acc)
        i_r, i_z, i_n, h_r, h_z, h_n = vals
        r = np.float32(1.0 / (1.0 + math.exp(-float(np.float32(i_r + h_r)))))
        z = np.float32(1.0 / (1.0 + math.exp(-float(np.float32(i_z + h_z)))))
        n_gate = np.float32(math.tanh(float(np.float32(i_n + np.float32(r * h_n)))))
        h_next[yy] = np.float32(np.float32(1.0 - z) * n_gate + np.float32(z * h_prev[yy])) if yy < mem_dim else np.float32(0.0)
    logits = np.zeros((a,), dtype=np.float32)
    for yy in range(a):
        acc = np.float32(head_b[yy])
        for xx in range(h):
            acc = np.float32(acc + np.float32(head_w[yy, xx] * h_next[xx]))
        logits[yy] = acc
    return logits, h_next


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _json_digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "PolicyAuthorityInputV1",
    "PolicyAuthorityResultV1",
    "authority_surfaces_payload",
    "compare_policy_authority_results",
    "compile_policy_authority_kernel",
    "load_policy_authority_backend",
    "make_policy_authority_fixture",
    "result_digest",
    "run_policy_authority_python_reference",
]
