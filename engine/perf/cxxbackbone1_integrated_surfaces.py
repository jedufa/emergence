from __future__ import annotations

"""CXXBACKBONE-1 remaining-surface native authority contracts."""

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
SCHEMA_VERSION = "cxxbackbone1_integrated_surfaces_v1"


@dataclass(frozen=True)
class IntegratedSurfacesFixtureV1:
    parent_a: np.ndarray
    parent_b: np.ndarray
    founder_a: np.ndarray
    founder_b: np.ndarray
    depth_a: np.ndarray
    depth_b: np.ndarray
    mutation_delta_q: np.ndarray
    alive: np.ndarray
    energy_q: np.ndarray
    maturity_q: np.ndarray
    genome_q: np.ndarray
    rng_q: np.ndarray
    slot_token: np.ndarray
    policy_row: np.ndarray
    hmem_q: np.ndarray
    checkpoint_founder: np.ndarray
    rng_state: np.ndarray
    event_id: np.ndarray
    event_slot_id: np.ndarray
    event_type: np.ndarray
    observations: np.ndarray
    weights: np.ndarray
    bias: np.ndarray
    cell_offsets: np.ndarray


@dataclass(frozen=True)
class IntegratedSurfacesResultV1:
    lineage: Mapping[str, np.ndarray | Mapping[str, float]]
    lifecycle: Mapping[str, np.ndarray | Mapping[str, float]]
    checkpoint: Mapping[str, np.ndarray | Mapping[str, float]]
    public_output: Mapping[str, bytes]
    multicell: Mapping[str, np.ndarray]
    timing: Mapping[str, float]
    executed_native: bool
    hidden_fallback: bool


def make_integrated_surfaces_fixture(*, slots: int = 24, cells: int = 4, obs_dim: int = 5, action_dim: int = 4, seed: int = 713) -> IntegratedSurfacesFixtureV1:
    rng = np.random.default_rng(int(seed))
    parent_a = np.arange(1000, 1000 + slots, dtype=np.int64)
    parent_b = np.arange(2000, 2000 + slots, dtype=np.int64)
    event_rows = max(4, slots // 2)
    rows_per_cell = slots // cells
    offsets = np.asarray([min(slots, i * rows_per_cell) for i in range(cells)] + [slots], dtype=np.int64)
    return IntegratedSurfacesFixtureV1(
        parent_a=parent_a,
        parent_b=parent_b,
        founder_a=np.asarray(10 + np.arange(slots) % 5, dtype=np.int64),
        founder_b=np.asarray(20 + np.arange(slots) % 7, dtype=np.int64),
        depth_a=np.asarray(1 + np.arange(slots) % 4, dtype=np.int32),
        depth_b=np.asarray(2 + np.arange(slots) % 5, dtype=np.int32),
        mutation_delta_q=np.asarray([(i % 21) - 10 for i in range(slots)], dtype=np.int32),
        alive=np.asarray([0 if i % 6 == 0 else 1 for i in range(slots)], dtype=np.int32),
        energy_q=np.asarray([150 if i % 7 == 0 else 700 + (i % 5) * 20 for i in range(slots)], dtype=np.int32),
        maturity_q=np.asarray([600 if i % 3 else 350 for i in range(slots)], dtype=np.int32),
        genome_q=np.asarray(10000 + np.arange(slots) * 3, dtype=np.int32),
        rng_q=np.asarray([(i * 173 + 41) % 1000 for i in range(slots)], dtype=np.int32),
        slot_token=np.asarray(7000 + np.arange(slots), dtype=np.int64),
        policy_row=np.asarray(np.arange(slots) % 6, dtype=np.int32),
        hmem_q=np.asarray(300 + np.arange(slots) * 11, dtype=np.int32),
        checkpoint_founder=np.asarray(50 + np.arange(slots) % 9, dtype=np.int64),
        rng_state=np.asarray(0xABCDEF123400 + np.arange(slots) * 65537, dtype=np.uint64),
        event_id=np.arange(1, event_rows + 1, dtype=np.int64),
        event_slot_id=np.asarray(7000 + np.arange(event_rows), dtype=np.int64),
        event_type=np.asarray([(i % 3) + 1 for i in range(event_rows)], dtype=np.int32),
        observations=rng.random((slots, obs_dim), dtype=np.float32),
        weights=rng.random((obs_dim, action_dim), dtype=np.float32) - np.float32(0.5),
        bias=rng.random((action_dim,), dtype=np.float32) - np.float32(0.5),
        cell_offsets=offsets,
    )


def run_integrated_surfaces_python_reference(f: IntegratedSurfacesFixtureV1) -> IntegratedSurfacesResultV1:
    t0 = time.perf_counter()
    choose_b = f.mutation_delta_q < 0
    founder = np.where(choose_b, f.founder_b, f.founder_a).astype(np.int64)
    depth = (np.maximum(f.depth_a, f.depth_b) + np.int32(1)).astype(np.int32)
    lineage_summary = {
        "child_count": float(f.parent_a.shape[0]),
        "depth_sum": float(np.sum(depth.astype(np.float64), dtype=np.float64)),
        "founder_switches": float(np.sum(choose_b.astype(np.float64), dtype=np.float64)),
        "mutation_abs_sum": float(np.sum(np.abs(f.mutation_delta_q).astype(np.float64), dtype=np.float64)),
    }

    alive = f.alive.astype(np.int32, copy=True)
    parent_slot = np.full_like(f.slot_token, -1)
    genome = f.genome_q.astype(np.int32, copy=True)
    lifecycle_event = np.zeros_like(f.alive)
    deaths = (alive != 0) & (f.energy_q <= 200)
    alive[deaths] = 0
    lifecycle_event[deaths] = 1
    births = 0
    for i in range(alive.shape[0]):
        if alive[i] != 0 or f.rng_q[i] >= 375:
            continue
        parent = -1
        for step in range(1, alive.shape[0] + 1):
            candidate = (i + step) % alive.shape[0]
            if alive[candidate] != 0 and f.maturity_q[candidate] >= 500:
                parent = candidate
                break
        if parent >= 0:
            mutation = int(f.rng_q[i] % 21) - 10
            alive[i] = 1
            parent_slot[i] = parent
            genome[i] = np.int32(int(f.genome_q[parent]) + mutation)
            lifecycle_event[i] = 2
            births += 1
    lifecycle_summary = {
        "births": float(births),
        "deaths": float(np.sum(deaths.astype(np.float64), dtype=np.float64)),
        "genome_sum": float(np.sum(genome.astype(np.float64), dtype=np.float64)),
        "slot_count": float(alive.shape[0]),
    }

    resumed_action = ((f.slot_token.astype(np.uint64) + f.policy_row.astype(np.uint64) * np.uint64(17) + f.hmem_q.astype(np.uint64) * np.uint64(31) + f.rng_state) % np.uint64(7)).astype(np.int32)
    checkpoint_summary = {
        "count": float(f.slot_token.shape[0]),
        "checksum": float(np.sum(((f.rng_state ^ f.slot_token.astype(np.uint64)) & np.uint64(0xFFFF)).astype(np.float64), dtype=np.float64)),
        "action_sum": float(np.sum(resumed_action.astype(np.float64), dtype=np.float64)),
        "reserved": 0.0,
    }
    jsonl = b"".join(
        f'{{"event_id":{int(e)},"slot_id":{int(s)},"event_type":{int(t)}}}\n'.encode("ascii")
        for e, s, t in zip(f.event_id, f.event_slot_id, f.event_type, strict=True)
    )
    csv = b"event_id,slot_id,event_type\n" + b"".join(
        f"{int(e)},{int(s)},{int(t)}\n".encode("ascii") for e, s, t in zip(f.event_id, f.event_slot_id, f.event_type, strict=True)
    )
    logits = (f.observations.astype(np.float32) @ f.weights.astype(np.float32) + f.bias.astype(np.float32)).astype(np.float32)
    actions = np.argmax(logits, axis=1).astype(np.int32)
    cell_sums = np.asarray([np.sum(actions[f.cell_offsets[i] : f.cell_offsets[i + 1]].astype(np.float64), dtype=np.float64) for i in range(f.cell_offsets.shape[0] - 1)], dtype=np.float64)
    return IntegratedSurfacesResultV1(
        lineage={
            "parent_a": np.ascontiguousarray(f.parent_a),
            "parent_b": np.ascontiguousarray(f.parent_b),
            "founder": np.ascontiguousarray(founder),
            "depth": np.ascontiguousarray(depth),
            "mutation_delta_q": np.ascontiguousarray(f.mutation_delta_q),
            "summary": lineage_summary,
        },
        lifecycle={
            "alive": np.ascontiguousarray(alive),
            "parent_slot": np.ascontiguousarray(parent_slot),
            "genome_q": np.ascontiguousarray(genome),
            "event_type": np.ascontiguousarray(lifecycle_event),
            "summary": lifecycle_summary,
        },
        checkpoint={
            "slot_token": np.ascontiguousarray(f.slot_token),
            "policy_row": np.ascontiguousarray(f.policy_row),
            "hmem_q": np.ascontiguousarray(f.hmem_q),
            "founder": np.ascontiguousarray(f.checkpoint_founder),
            "rng_state": np.ascontiguousarray(f.rng_state),
            "resumed_action": np.ascontiguousarray(resumed_action),
            "summary": checkpoint_summary,
        },
        public_output={"jsonl": jsonl, "csv": csv},
        multicell={"actions": np.ascontiguousarray(actions), "cell_action_sums": np.ascontiguousarray(cell_sums)},
        timing={"wall_s": float(time.perf_counter() - t0)},
        executed_native=False,
        hidden_fallback=False,
    )


def compile_integrated_surfaces_kernel(build_dir: Path | str, *, cxx: str | None = None) -> dict[str, Any]:
    compiler = cxx or shutil.which("g++")
    build = Path(build_dir)
    source_path = ROOT / "engine" / "perf" / "native" / "cxxbackbone1_integrated_surfaces_kernel.cpp"
    library_path = build / "libcxxbackbone1_integrated_surfaces.so"
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


def load_integrated_surfaces_backend(library_path: Path | str):
    lib = ctypes.CDLL(str(library_path))
    i64p = ctypes.POINTER(ctypes.c_int64)
    i32p = ctypes.POINTER(ctypes.c_int32)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    f32p = ctypes.POINTER(ctypes.c_float)
    f64p = ctypes.POINTER(ctypes.c_double)
    charp = ctypes.POINTER(ctypes.c_char)

    lineage_fn = lib.cxxbackbone1_lineage_lifecycle_v1
    lineage_fn.argtypes = [i64p, i64p, i64p, i64p, i32p, i32p, i32p, ctypes.c_int64, i64p, i64p, i64p, i32p, i32p, f64p]
    lineage_fn.restype = ctypes.c_int
    lifecycle_fn = lib.cxxbackbone1_lifecycle_transactions_v1
    lifecycle_fn.argtypes = [i32p, i32p, i32p, i32p, i32p, ctypes.c_int64, i32p, i64p, i32p, i32p, f64p]
    lifecycle_fn.restype = ctypes.c_int
    checkpoint_fn = lib.cxxbackbone1_checkpoint_restore_v1
    checkpoint_fn.argtypes = [i64p, i32p, i32p, i64p, u64p, ctypes.c_int64, i64p, i32p, i32p, i64p, u64p, i32p, f64p]
    checkpoint_fn.restype = ctypes.c_int
    public_fn = lib.cxxbackbone1_public_output_v1
    public_fn.argtypes = [i64p, i64p, i32p, ctypes.c_int64, charp, ctypes.c_int64, charp, ctypes.c_int64, i64p, i64p]
    public_fn.restype = ctypes.c_int
    multicell_fn = lib.cxxbackbone1_multicell_batch_v1
    multicell_fn.argtypes = [f32p, f32p, f32p, i64p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, i32p, f64p]
    multicell_fn.restype = ctypes.c_int

    def _call(f: IntegratedSurfacesFixtureV1) -> IntegratedSurfacesResultV1:
        setup_t0 = time.perf_counter()
        n = int(f.parent_a.shape[0])
        lineage_parent_a = np.zeros((n,), dtype=np.int64)
        lineage_parent_b = np.zeros((n,), dtype=np.int64)
        lineage_founder = np.zeros((n,), dtype=np.int64)
        lineage_depth = np.zeros((n,), dtype=np.int32)
        lineage_mutation = np.zeros((n,), dtype=np.int32)
        lineage_summary = np.zeros((4,), dtype=np.float64)
        alive = np.zeros((n,), dtype=np.int32)
        parent_slot = np.zeros((n,), dtype=np.int64)
        genome = np.zeros((n,), dtype=np.int32)
        lifecycle_event = np.zeros((n,), dtype=np.int32)
        lifecycle_summary = np.zeros((4,), dtype=np.float64)
        slot_token = np.zeros((n,), dtype=np.int64)
        policy_row = np.zeros((n,), dtype=np.int32)
        hmem_q = np.zeros((n,), dtype=np.int32)
        checkpoint_founder = np.zeros((n,), dtype=np.int64)
        rng_state = np.zeros((n,), dtype=np.uint64)
        resumed_action = np.zeros((n,), dtype=np.int32)
        checkpoint_summary = np.zeros((4,), dtype=np.float64)
        jsonl_buf = ctypes.create_string_buffer(max(4096, int(f.event_id.shape[0]) * 96))
        csv_buf = ctypes.create_string_buffer(max(4096, int(f.event_id.shape[0]) * 48))
        jsonl_size = ctypes.c_int64(0)
        csv_size = ctypes.c_int64(0)
        actions = np.zeros((n,), dtype=np.int32)
        cell_sums = np.zeros((int(f.cell_offsets.shape[0]) - 1,), dtype=np.float64)
        setup_s = time.perf_counter() - setup_t0
        compute_t0 = time.perf_counter()
        statuses = [
            lineage_fn(
                _i64(f.parent_a), _i64(f.parent_b), _i64(f.founder_a), _i64(f.founder_b), _i32(f.depth_a), _i32(f.depth_b),
                _i32(f.mutation_delta_q), ctypes.c_int64(n), lineage_parent_a.ctypes.data_as(i64p), lineage_parent_b.ctypes.data_as(i64p),
                lineage_founder.ctypes.data_as(i64p), lineage_depth.ctypes.data_as(i32p), lineage_mutation.ctypes.data_as(i32p),
                lineage_summary.ctypes.data_as(f64p),
            ),
            lifecycle_fn(
                _i32(f.alive), _i32(f.energy_q), _i32(f.maturity_q), _i32(f.genome_q), _i32(f.rng_q), ctypes.c_int64(n),
                alive.ctypes.data_as(i32p), parent_slot.ctypes.data_as(i64p), genome.ctypes.data_as(i32p), lifecycle_event.ctypes.data_as(i32p),
                lifecycle_summary.ctypes.data_as(f64p),
            ),
            checkpoint_fn(
                _i64(f.slot_token), _i32(f.policy_row), _i32(f.hmem_q), _i64(f.checkpoint_founder), _u64(f.rng_state), ctypes.c_int64(n),
                slot_token.ctypes.data_as(i64p), policy_row.ctypes.data_as(i32p), hmem_q.ctypes.data_as(i32p), checkpoint_founder.ctypes.data_as(i64p),
                rng_state.ctypes.data_as(u64p), resumed_action.ctypes.data_as(i32p), checkpoint_summary.ctypes.data_as(f64p),
            ),
            public_fn(
                _i64(f.event_id), _i64(f.event_slot_id), _i32(f.event_type), ctypes.c_int64(int(f.event_id.shape[0])),
                ctypes.cast(jsonl_buf, charp), ctypes.c_int64(ctypes.sizeof(jsonl_buf)), ctypes.cast(csv_buf, charp), ctypes.c_int64(ctypes.sizeof(csv_buf)),
                ctypes.byref(jsonl_size), ctypes.byref(csv_size),
            ),
            multicell_fn(
                _f32(f.observations), _f32(f.weights), _f32(f.bias), _i64(f.cell_offsets), ctypes.c_int64(int(f.cell_offsets.shape[0]) - 1),
                ctypes.c_int64(int(f.observations.shape[1])), ctypes.c_int64(int(f.weights.shape[1])), actions.ctypes.data_as(i32p), cell_sums.ctypes.data_as(f64p),
            ),
        ]
        compute_s = time.perf_counter() - compute_t0
        for status in statuses:
            if int(status) != 0:
                raise RuntimeError(f"CXXBACKBONE1 integrated surfaces kernel failed with status {int(status)}")
        return IntegratedSurfacesResultV1(
            lineage={
                "parent_a": lineage_parent_a,
                "parent_b": lineage_parent_b,
                "founder": lineage_founder,
                "depth": lineage_depth,
                "mutation_delta_q": lineage_mutation,
                "summary": _summary_dict(("child_count", "depth_sum", "founder_switches", "mutation_abs_sum"), lineage_summary),
            },
            lifecycle={
                "alive": alive,
                "parent_slot": parent_slot,
                "genome_q": genome,
                "event_type": lifecycle_event,
                "summary": _summary_dict(("births", "deaths", "genome_sum", "slot_count"), lifecycle_summary),
            },
            checkpoint={
                "slot_token": slot_token,
                "policy_row": policy_row,
                "hmem_q": hmem_q,
                "founder": checkpoint_founder,
                "rng_state": rng_state,
                "resumed_action": resumed_action,
                "summary": _summary_dict(("count", "checksum", "action_sum", "reserved"), checkpoint_summary),
            },
            public_output={"jsonl": jsonl_buf.raw[: jsonl_size.value], "csv": csv_buf.raw[: csv_size.value]},
            multicell={"actions": actions, "cell_action_sums": cell_sums},
            timing={"setup_s": float(setup_s), "backend_compute_s": float(compute_s), "wall_s": float(setup_s + compute_s)},
            executed_native=True,
            hidden_fallback=False,
        )

    return _call


def compare_integrated_surfaces_results(expected: IntegratedSurfacesResultV1, observed: IntegratedSurfacesResultV1) -> dict[str, Any]:
    for group in ("lineage", "lifecycle", "checkpoint", "multicell"):
        exp_group = expected.__dict__[group]
        obs_group = observed.__dict__[group]
        for name, exp_value in exp_group.items():
            obs_value = obs_group[name]
            if name == "summary":
                for key, exp_num in exp_value.items():  # type: ignore[union-attr]
                    obs_num = obs_value[key]  # type: ignore[index]
                    if float(exp_num) != float(obs_num):
                        return _bad(f"{group}.summary.{key}", exp_num, obs_num)
                continue
            a = np.asarray(exp_value)
            b = np.asarray(obs_value)
            if not np.array_equal(a, b):
                idx = tuple(int(x) for x in np.argwhere(a != b)[0])
                return _bad(f"{group}.{name}", _scalar(a[idx]), _scalar(b[idx]), index=list(idx))
    for name in ("jsonl", "csv"):
        if expected.public_output[name] != observed.public_output[name]:
            return {
                "ok": False,
                "first_divergence": {
                    "field": f"public_output.{name}",
                    "expected_sha256": hashlib.sha256(expected.public_output[name]).hexdigest(),
                    "observed_sha256": hashlib.sha256(observed.public_output[name]).hexdigest(),
                },
            }
    return {"ok": True, "first_divergence": None, "state_digest": result_digest(observed)}


def authority_surfaces_payload(parity: Mapping[str, Any]) -> dict[str, Any]:
    exact = bool(parity.get("ok"))
    return {
        "schema_version": SCHEMA_VERSION,
        "native_authority_surfaces": {
            "lineage_founder_state": exact,
            "lifecycle_reproduction_mutation_state": exact,
            "full_resident_checkpoint": exact,
            "live_lifecycle_execution": exact,
            "public_output_compatibility": exact,
            "multi_cell_batching": exact,
        },
        "unit_parity": "exact" if exact else "blocked",
        "first_divergence": parity.get("first_divergence"),
        "live_integrated": False,
        "checkpoint_integrated": exact,
        "public_output_bytes_exact": exact,
    }


def result_digest(result: IntegratedSurfacesResultV1) -> str:
    payload: dict[str, Any] = {
        "public_output": {name: hashlib.sha256(value).hexdigest() for name, value in result.public_output.items()},
    }
    for group in ("lineage", "lifecycle", "checkpoint", "multicell"):
        group_value = result.__dict__[group]
        payload[group] = {}
        for name, value in group_value.items():
            payload[group][name] = dict(value) if name == "summary" else _array_digest(value)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _summary_dict(names: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {name: float(values[i]) for i, name in enumerate(names)}


def _i64(value: Any):
    return np.ascontiguousarray(value, dtype=np.int64).ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def _i32(value: Any):
    return np.ascontiguousarray(value, dtype=np.int32).ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _u64(value: Any):
    return np.ascontiguousarray(value, dtype=np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _f32(value: Any):
    return np.ascontiguousarray(value, dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _array_digest(value: Any) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _scalar(value: Any) -> int | float:
    if np.asarray(value).dtype.kind in {"u", "i"}:
        return int(value)
    return float(value)


def _bad(field: str, expected: Any, observed: Any, *, index: list[int] | None = None) -> dict[str, Any]:
    payload = {"field": field, "expected": expected, "observed": observed}
    if index is not None:
        payload["index"] = index
    return {"ok": False, "first_divergence": payload}


__all__ = [
    "SCHEMA_VERSION",
    "IntegratedSurfacesFixtureV1",
    "IntegratedSurfacesResultV1",
    "authority_surfaces_payload",
    "compare_integrated_surfaces_results",
    "compile_integrated_surfaces_kernel",
    "load_integrated_surfaces_backend",
    "make_integrated_surfaces_fixture",
    "result_digest",
    "run_integrated_surfaces_python_reference",
]
