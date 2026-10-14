from __future__ import annotations

"""CXXBACKBONE-1 native RNG stream and mutation draw-schedule contract."""

from dataclasses import dataclass
import ctypes
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "cxxbackbone1_rng_schedule_v1"


@dataclass(frozen=True)
class RngScheduleInputV1:
    initial_stream_state: np.ndarray
    initial_draw_index: np.ndarray
    stream_id_by_draw: np.ndarray
    draws_per_event: np.ndarray
    placeholder_draws_per_event: np.ndarray


@dataclass(frozen=True)
class RngScheduleResultV1:
    stream_state: np.ndarray
    draw_index: np.ndarray
    draw_words: np.ndarray
    draw_offsets: np.ndarray
    timing: Mapping[str, float]
    executed_native: bool
    hidden_fallback: bool


def make_rng_schedule_fixture(*, streams: int = 4, events: int = 18, seed: int = 1818) -> RngScheduleInputV1:
    rng = np.random.default_rng(int(seed))
    return RngScheduleInputV1(
        initial_stream_state=rng.integers(1, np.iinfo(np.uint64).max, size=(streams,), dtype=np.uint64),
        initial_draw_index=np.zeros((streams,), dtype=np.uint64),
        stream_id_by_draw=np.asarray([(i * 3 + 1) % streams for i in range(events)], dtype=np.int32),
        draws_per_event=np.asarray([1 + (i % 5) for i in range(events)], dtype=np.int32),
        placeholder_draws_per_event=np.asarray([0 if i % 3 else 2 for i in range(events)], dtype=np.int32),
    )


def run_rng_schedule_python_reference(inputs: RngScheduleInputV1) -> RngScheduleResultV1:
    t0 = time.perf_counter()
    state = np.asarray(inputs.initial_stream_state, dtype=np.uint64).copy()
    index = np.asarray(inputs.initial_draw_index, dtype=np.uint64).copy()
    total_words = int(np.sum(inputs.draws_per_event, dtype=np.int64))
    words = np.zeros((total_words,), dtype=np.uint64)
    offsets = np.zeros((int(inputs.stream_id_by_draw.shape[0]) + 1,), dtype=np.int64)
    cursor = 0
    for event, stream in enumerate(np.asarray(inputs.stream_id_by_draw, dtype=np.int32).tolist()):
        draws = int(inputs.draws_per_event[event])
        placeholders = int(inputs.placeholder_draws_per_event[event])
        offsets[event] = cursor
        for _idx in range(draws):
            state[stream], words[cursor] = _splitmix64_next(state[stream])
            cursor += 1
        for _idx in range(placeholders):
            state[stream], _ = _splitmix64_next(state[stream])
        index[stream] = np.uint64(int(index[stream]) + draws + placeholders)
    offsets[-1] = cursor
    return RngScheduleResultV1(
        stream_state=state,
        draw_index=index,
        draw_words=words,
        draw_offsets=offsets,
        timing={"wall_s": float(time.perf_counter() - t0)},
        executed_native=False,
        hidden_fallback=False,
    )


def compile_rng_schedule_kernel(build_dir: Path | str, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    build = Path(build_dir)
    source_path = ROOT / "engine" / "perf" / "native" / "cxxbackbone1_rng_schedule_kernel.cpp"
    library_path = build / "libcxxbackbone1_rng_schedule.so"
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


def load_rng_schedule_backend(library_path: Path | str):
    lib = ctypes.CDLL(str(library_path))
    fn = lib.cxxbackbone1_rng_schedule_v1
    u64p = ctypes.POINTER(ctypes.c_uint64)
    i32p = ctypes.POINTER(ctypes.c_int32)
    i64p = ctypes.POINTER(ctypes.c_int64)
    fn.argtypes = [u64p, u64p, i32p, i32p, i32p, ctypes.c_int64, ctypes.c_int64, u64p, u64p, u64p, i64p, i64p]
    fn.restype = ctypes.c_int

    def _call(inputs: RngScheduleInputV1) -> RngScheduleResultV1:
        prep_t0 = time.perf_counter()
        stream_count = int(inputs.initial_stream_state.shape[0])
        event_count = int(inputs.stream_id_by_draw.shape[0])
        total_capacity = int(np.sum(inputs.draws_per_event, dtype=np.int64))
        out_state = np.zeros((stream_count,), dtype=np.uint64)
        out_index = np.zeros((stream_count,), dtype=np.uint64)
        out_words = np.zeros((total_capacity,), dtype=np.uint64)
        out_offsets = np.zeros((event_count + 1,), dtype=np.int64)
        out_total = np.zeros((1,), dtype=np.int64)
        prep_s = time.perf_counter() - prep_t0
        compute_t0 = time.perf_counter()
        status = fn(
            np.ascontiguousarray(inputs.initial_stream_state, dtype=np.uint64).ctypes.data_as(u64p),
            np.ascontiguousarray(inputs.initial_draw_index, dtype=np.uint64).ctypes.data_as(u64p),
            np.ascontiguousarray(inputs.stream_id_by_draw, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.draws_per_event, dtype=np.int32).ctypes.data_as(i32p),
            np.ascontiguousarray(inputs.placeholder_draws_per_event, dtype=np.int32).ctypes.data_as(i32p),
            ctypes.c_int64(stream_count),
            ctypes.c_int64(event_count),
            out_state.ctypes.data_as(u64p),
            out_index.ctypes.data_as(u64p),
            out_words.ctypes.data_as(u64p),
            out_offsets.ctypes.data_as(i64p),
            out_total.ctypes.data_as(i64p),
        )
        compute_s = time.perf_counter() - compute_t0
        if int(status) != 0:
            raise RuntimeError(f"CXXBACKBONE1 RNG schedule kernel failed with status {int(status)}")
        return RngScheduleResultV1(
            stream_state=out_state,
            draw_index=out_index,
            draw_words=out_words[: int(out_total[0])].copy(),
            draw_offsets=out_offsets,
            timing={"setup_s": float(prep_s), "backend_compute_s": float(compute_s), "wall_s": float(prep_s + compute_s)},
            executed_native=True,
            hidden_fallback=False,
        )

    return _call


def compare_rng_schedule_results(expected: RngScheduleResultV1, observed: RngScheduleResultV1) -> dict[str, Any]:
    for name in ("stream_state", "draw_index", "draw_words", "draw_offsets"):
        a = np.asarray(getattr(expected, name))
        b = np.asarray(getattr(observed, name))
        if a.shape != b.shape:
            return {"ok": False, "first_divergence": {"field": name, "expected_shape": list(a.shape), "observed_shape": list(b.shape)}}
        if not bool(np.array_equal(a, b)):
            idx = tuple(int(x) for x in np.argwhere(a != b)[0])
            return {"ok": False, "first_divergence": {"field": name, "index": list(idx), "expected": int(a[idx]), "observed": int(b[idx])}}
    return {"ok": True, "first_divergence": None, "state_digest": result_digest(observed)}


def authority_surfaces_payload(parity: Mapping[str, Any]) -> dict[str, Any]:
    exact = bool(parity.get("ok"))
    return {
        "schema_version": SCHEMA_VERSION,
        "native_authority_surfaces": {
            "rng_state_indices": exact,
            "mutation_rng_draw_schedule": exact,
        },
        "unit_parity": "exact" if exact else "blocked",
        "first_divergence": parity.get("first_divergence"),
        "live_integrated": False,
        "checkpoint_integrated": False,
    }


def result_digest(result: RngScheduleResultV1) -> str:
    return _json_digest(
        {
            "stream_state": _array_digest(result.stream_state),
            "draw_index": _array_digest(result.draw_index),
            "draw_words": _array_digest(result.draw_words),
            "draw_offsets": _array_digest(result.draw_offsets),
        }
    )


def _splitmix64_next(state: np.uint64) -> tuple[np.uint64, np.uint64]:
    mask = (1 << 64) - 1
    s = (int(state) + 0x9E3779B97F4A7C15) & mask
    z = s
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    out = (z ^ (z >> 31)) & mask
    return np.uint64(s), np.uint64(out)


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
    "RngScheduleInputV1",
    "RngScheduleResultV1",
    "authority_surfaces_payload",
    "compare_rng_schedule_results",
    "compile_rng_schedule_kernel",
    "load_rng_schedule_backend",
    "make_rng_schedule_fixture",
    "result_digest",
    "run_rng_schedule_python_reference",
]
