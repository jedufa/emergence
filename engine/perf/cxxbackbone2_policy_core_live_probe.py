from __future__ import annotations

"""CXXBACKBONE-2 production-state policy-core native probe.

This module deliberately probes the real RunnerCore action-refill boundary
without taking authority over live behavior. It is a default-off bridge between
the exact CXXBACKBONE-1 policy kernel contract and the production runner state.
"""

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from perf.cxxbackbone1_policy_authority import (
    PolicyAuthorityInputV1,
    compile_policy_authority_kernel,
    load_policy_authority_backend,
)


SCHEMA_VERSION = "cxxbackbone2_policy_core_live_probe_v1"


def policy_core_probe_requested(runner: Any) -> bool:
    cfg = getattr(runner, "sim_cfg", None)
    return bool(getattr(cfg, "cxxbackbone2_policy_core_live_probe_enable", False))


def run_policy_core_live_probe(
    runner: Any,
    state: Any,
    *,
    agent_ids: np.ndarray,
    slots: np.ndarray,
    obs_np: np.ndarray,
    produced_actions: np.ndarray,
    hmem_before: np.ndarray,
    k: int,
    buckets: dict[str, float] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "supported": False,
        "executed_native": False,
        "hidden_fallback": False,
        "rows": int(np.asarray(agent_ids).shape[0]),
        "k": int(k),
        "blockers": [],
        "exact_action_parity": False,
        "exact_hmem_parity": False,
        "tolerant_hmem_parity": False,
        "first_divergence": None,
        "timing": {},
    }
    try:
        inputs = _build_policy_input(runner, state, slots=slots, obs_np=obs_np, hmem_before=hmem_before, k=int(k))
        build_dir = _build_dir(runner)
        compile_info = compile_policy_authority_kernel(build_dir)
        result["compile"] = compile_info
        if not bool(compile_info.get("ok")):
            result["blockers"].append("native_policy_kernel_compile_failed")
            return _finish(runner, result, started, buckets)

        backend = load_policy_authority_backend(compile_info["library_path"])
        native = backend(inputs)
        result["executed_native"] = True
        result["native_timing"] = dict(native.timing)

        expected_actions = np.asarray(produced_actions, dtype=np.int32)
        if expected_actions.ndim == 2:
            expected_actions = expected_actions[:, 0]
        action_cmp = _compare_arrays("actions", expected_actions, native.actions, exact=True)
        result["exact_action_parity"] = bool(action_cmp["ok"])
        if not action_cmp["ok"]:
            result["first_divergence"] = action_cmp["first_divergence"]
            result["blockers"].append("native_policy_actions_diverged_from_production")

        hmem_after = _torch_cpu_numpy_copy(getattr(runner, "_Hmem", None))
        if hmem_after is None:
            result["blockers"].append("production_hmem_after_missing")
        else:
            exact_hmem = _compare_arrays("hmem_after", hmem_after, native.hmem_after, exact=True)
            tolerant_hmem = _compare_arrays("hmem_after", hmem_after, native.hmem_after, exact=False, atol=1.0e-5)
            result["exact_hmem_parity"] = bool(exact_hmem["ok"])
            result["tolerant_hmem_parity"] = bool(tolerant_hmem["ok"])
            if result["first_divergence"] is None and not exact_hmem["ok"]:
                result["first_divergence"] = exact_hmem["first_divergence"]
            if not exact_hmem["ok"]:
                result["blockers"].append("native_policy_hmem_not_bit_exact_with_production")

        result["supported"] = bool(
            result["executed_native"]
            and result["exact_action_parity"]
            and result["exact_hmem_parity"]
            and not result["blockers"]
        )
        return _finish(runner, result, started, buckets)
    except Exception as exc:  # pragma: no cover - summary path is the contract.
        result["blockers"].append(f"native_policy_probe_exception:{type(exc).__name__}")
        result["exception"] = str(exc)
        return _finish(runner, result, started, buckets)


def _build_policy_input(
    runner: Any,
    state: Any,
    *,
    slots: np.ndarray,
    obs_np: np.ndarray,
    hmem_before: np.ndarray,
    k: int,
) -> PolicyAuthorityInputV1:
    slots_i32 = np.ascontiguousarray(slots, dtype=np.int32)
    use_gru = np.asarray(getattr(state, "use_gru"))[slots_i32].astype(np.bool_, copy=False)
    cfg = getattr(runner, "sim_cfg", None)
    return PolicyAuthorityInputV1(
        policy_kind=np.ascontiguousarray(use_gru.astype(np.int32), dtype=np.int32),
        row_slots=slots_i32,
        observations=np.ascontiguousarray(obs_np, dtype=np.float32),
        drives=np.ascontiguousarray(np.asarray(getattr(state, "drives"))[slots_i32], dtype=np.float32),
        gain=np.ascontiguousarray(np.asarray(getattr(state, "gain"))[slots_i32], dtype=np.float32),
        aggression=np.ascontiguousarray(np.asarray(getattr(state, "aggr"))[slots_i32], dtype=np.float32),
        temperature=np.ascontiguousarray(np.asarray(getattr(state, "temp"))[slots_i32], dtype=np.float32),
        effective_memory_dim=np.ascontiguousarray(np.asarray(getattr(state, "mem_dim"))[slots_i32], dtype=np.int32),
        mlp_pack=_required_bank(getattr(runner, "_mlp_pack", None), "mlp_pack"),
        gru_pack=_required_bank(getattr(runner, "_gru_pack", None), "gru_pack"),
        hmem=np.ascontiguousarray(hmem_before, dtype=np.float32),
        action_cache_cursor=np.zeros((int(np.asarray(hmem_before).shape[0]),), dtype=np.int32),
        cache_stride=max(1, int(k)),
        mlp_hidden=int(getattr(runner, "_hidden")),
        gru_hidden=int(getattr(runner, "_gru_hidden")),
        actions=int(getattr(runner, "_out")),
        w_hunger_gather=float(getattr(cfg, "drive_w_hunger_gather", 1.2)),
        w_pain_rest=float(getattr(cfg, "drive_w_pain_rest", 0.9)),
        w_fatigue_rest=float(getattr(cfg, "drive_w_fatigue_rest", 1.0)),
        w_pain_attack=float(getattr(cfg, "drive_w_pain_attack", 1.2)),
        w_fear_attack=float(getattr(cfg, "drive_w_fear_attack", 1.4)),
    )


def _required_bank(value: Any, name: str) -> np.ndarray:
    arr = _torch_cpu_numpy_copy(value)
    if arr is None:
        raise ValueError(f"{name}_missing")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _torch_cpu_numpy_copy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).copy()
    return np.ascontiguousarray(np.asarray(value), dtype=np.float32).copy()


def _compare_arrays(name: str, expected: Any, observed: Any, *, exact: bool, atol: float = 0.0) -> dict[str, Any]:
    a = np.asarray(expected)
    b = np.asarray(observed)
    if a.shape != b.shape:
        return {
            "ok": False,
            "first_divergence": {
                "field": name,
                "expected_shape": list(a.shape),
                "observed_shape": list(b.shape),
            },
        }
    if exact:
        bad = a != b
    else:
        bad = np.abs(a.astype(np.float64) - b.astype(np.float64)) > float(atol)
    if not bool(np.any(bad)):
        return {"ok": True, "first_divergence": None}
    idx = tuple(int(x) for x in np.argwhere(bad)[0])
    return {
        "ok": False,
        "first_divergence": {
            "field": name,
            "index": list(idx),
            "expected": _scalar(a[idx]),
            "observed": _scalar(b[idx]),
            "abs_delta": float(abs(float(a[idx]) - float(b[idx]))) if a.dtype.kind == "f" or b.dtype.kind == "f" else None,
        },
    }


def _scalar(value: Any) -> int | float:
    if np.asarray(value).dtype.kind == "f":
        return float(value)
    return int(value)


def _build_dir(runner: Any) -> Path:
    cfg = getattr(runner, "sim_cfg", None)
    configured = str(getattr(cfg, "cxxbackbone2_policy_core_live_probe_build_dir", "") or "")
    if configured:
        return Path(configured)
    return Path(getattr(runner, "run_dir", ".") or ".") / "metrics" / "cxxbackbone2_policy_core_build"


def _finish(
    runner: Any,
    result: dict[str, Any],
    started: float,
    buckets: dict[str, float] | None,
) -> dict[str, Any]:
    result["timing"]["wall_s"] = float(time.perf_counter() - started)
    if buckets is not None:
        buckets["cxxbackbone2_policy_core_probe_ms"] = float(
            buckets.get("cxxbackbone2_policy_core_probe_ms", 0.0) + result["timing"]["wall_s"] * 1000.0
        )
    setattr(runner, "_cxxbackbone2_policy_core_live_probe_summary", result)
    _write_summary(runner, result)
    return result


def _write_summary(runner: Any, result: Mapping[str, Any]) -> None:
    cfg = getattr(runner, "sim_cfg", None)
    if not bool(getattr(cfg, "cxxbackbone2_policy_core_live_probe_write_summary", True)):
        return
    filename = str(
        getattr(
            cfg,
            "cxxbackbone2_policy_core_live_probe_summary_filename",
            "cxxbackbone2_policy_core_live_probe_summary.json",
        )
        or "cxxbackbone2_policy_core_live_probe_summary.json"
    )
    out = Path(getattr(runner, "run_dir", ".") or ".") / "metrics" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(out)


__all__ = [
    "SCHEMA_VERSION",
    "policy_core_probe_requested",
    "run_policy_core_live_probe",
]
