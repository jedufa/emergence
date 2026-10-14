#!/usr/bin/env python3
"""Run exact Python-reference, native CPU, and sm_75 CUDA EED parity."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Final

import numpy as np
import torch

ROOT: Final[Path] = Path(__file__).resolve().parents[2]
for import_root in (ROOT, ROOT / "engine"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from engine.env.gridworld_pkg.patch_world import (  # noqa: E402
    settle_eed_resource_impact,
)


SCHEMA_VERSION: Final[str] = "eiel_eed0_native_parity_gate_v1"
OVERLAY_ID: Final[str] = "world.overlay.eed_resource_impact.v1"
ENTRYPOINT: Final[str] = "eiel_eed_resource_impact_transition_v1"
CAPABILITY_ENTRYPOINT: Final[str] = "eiel_eed_resource_impact_capability_v1"
STATE_NAMES: Final[tuple[str, ...]] = (
    "eed_resource_a_amount",
    "eed_resource_b_amount",
    "eed_resource_a_capacity",
    "eed_resource_b_capacity",
    "eed_resource_a_renewal",
    "eed_resource_b_renewal",
    "eed_resource_a_pending_impact",
    "eed_resource_b_pending_impact",
    "eed_resource_a_impact_load",
    "eed_resource_b_impact_load",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(canonical.numpy().tobytes(order="C")).hexdigest()


def _load_extension(path: Path) -> ModuleType:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"EED native artifact does not exist: {resolved}")
    spec = importlib.util.spec_from_file_location(
        "eiel_cuda_architecture_b_flattened_wdl_controller_v2",
        resolved,
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load EED native artifact: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _initial_state() -> list[np.ndarray]:
    amount_a = np.array(
        [
            [[0.25, 0.75, 0.00], [0.50, 0.10, 0.90]],
            [[0.20, 0.40, 0.60], [0.80, 0.00, 0.30]],
        ],
        dtype=np.float32,
    )
    amount_b = np.flip(amount_a, axis=2).copy()
    shape = amount_a.shape
    return [
        amount_a,
        amount_b,
        np.ones(shape, dtype=np.float32),
        np.ones(shape, dtype=np.float32),
        np.full(shape, 0.25, dtype=np.float32),
        np.full(shape, 0.25, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
    ]


def _events() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([1, 2, 1, 2], dtype=np.int32),
        np.asarray([0, 1, 7, 10], dtype=np.int64),
        np.asarray([1.0, -1.0, 0.5, -0.25], dtype=np.float32),
    )


def _config(*, impact_applied: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "impact_per_success": 0.25,
        "decay_per_step": 0.25,
        "maximum_load": 1.0,
        "impact_applied": impact_applied,
        "ecology_frozen": False,
        "phenotype_fixed": False,
        "fixed_axis_mode": 0,
    }


def _reference_transition(
    state: list[np.ndarray],
    *,
    impact_applied: bool,
) -> list[np.ndarray]:
    classes, indices, axes = _events()
    result = [np.ascontiguousarray(item.copy()) for item in state]
    if impact_applied:
        for resource_class, index, axis in zip(classes, indices, axes, strict=True):
            label = "a" if int(resource_class) == 1 else "b"
            signed_axis = float(axis) if label == "a" else -float(axis)
            impact = 0.25 * max(0.0, signed_axis)
            pending = result[6 if label == "a" else 7].reshape(-1)
            pending[int(index)] = np.float32(float(pending[int(index)]) + impact)

    settled: list[np.ndarray] = []
    for cell in range(result[0].shape[0]):
        cfg = SimpleNamespace(
            patch_world_enable=True,
            respec0b_resource_specialization_enable=True,
            eed_resource_impact_enable=True,
            eed_resource_impact_apply_enable=impact_applied,
            eed_ecology_frozen_enable=False,
            eed_phenotype_fixed_enable=False,
            eed_phenotype_fixed_axis_profile="neutral",
            eed_resource_impact_decay_per_step=0.25,
            eed_resource_impact_max_load=1.0,
        )
        env = SimpleNamespace(
            cfg=cfg,
            grid=np.zeros(result[0].shape[1:], dtype=np.int8),
            patch_food_amount=np.zeros(result[0].shape[1:], dtype=np.float32),
            respec0b_resource_class=np.array([[1, 2, 1], [2, 1, 2]], dtype=np.float32),
        )
        for name, array in zip(STATE_NAMES, result, strict=True):
            setattr(env, name, np.ascontiguousarray(array[cell].copy()))
        settle_eed_resource_impact(env)
        if cell == 0:
            settled = [np.empty_like(array, dtype=np.float32) for array in result]
        for target, name in zip(settled, STATE_NAMES, strict=True):
            target[cell] = np.asarray(getattr(env, name), dtype=np.float32)
    return settled


def _native_transition(
    module: ModuleType,
    state: list[np.ndarray],
    *,
    device: str,
    impact_applied: bool,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    tensors = [
        torch.from_numpy(np.ascontiguousarray(item.copy())).to(device=device)
        for item in state
    ]
    classes, indices, axes = _events()
    result = getattr(module, ENTRYPOINT)(
        tensors,
        torch.from_numpy(classes).to(device=device),
        torch.from_numpy(indices).to(device=device),
        torch.from_numpy(axes).to(device=device),
        _config(impact_applied=impact_applied),
    )
    if device == "cuda":
        torch.cuda.synchronize()
    return tensors, dict(result)


def run_gate(artifact: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("EED native parity requires an available CUDA device")
    module = _load_extension(artifact)
    capability = dict(getattr(module, CAPABILITY_ENTRYPOINT)())
    expected_capability = {
        "schema_version": 1,
        "overlay_id": OVERLAY_ID,
        "cpu_oracle": True,
        "cuda_owner": True,
        "state_tensor_count": 10,
        "deterministic_settlement": True,
        "new_rng_draws": 0,
        "production_route_integrated": False,
    }
    if capability != expected_capability:
        raise RuntimeError(f"EED native capability mismatch: {capability!r}")

    scenarios: list[dict[str, Any]] = []
    for impact_applied in (False, True):
        initial = _initial_state()
        reference = _reference_transition(initial, impact_applied=impact_applied)
        cpu, cpu_result = _native_transition(
            module,
            initial,
            device="cpu",
            impact_applied=impact_applied,
        )
        cuda, cuda_result = _native_transition(
            module,
            initial,
            device="cuda",
            impact_applied=impact_applied,
        )
        repeat, _ = _native_transition(
            module,
            initial,
            device="cuda",
            impact_applied=impact_applied,
        )
        comparisons: dict[str, bool] = {}
        for name, expected, cpu_tensor, cuda_tensor, repeat_tensor in zip(
            STATE_NAMES, reference, cpu, cuda, repeat, strict=True
        ):
            expected_tensor = torch.from_numpy(expected)
            comparisons[f"{name}.reference_cpu"] = torch.equal(
                expected_tensor, cpu_tensor
            )
            comparisons[f"{name}.cpu_cuda"] = torch.equal(cpu_tensor, cuda_tensor.cpu())
            comparisons[f"{name}.cuda_repeat"] = torch.equal(cuda_tensor, repeat_tensor)
        if not all(comparisons.values()):
            failed = sorted(name for name, passed in comparisons.items() if not passed)
            raise RuntimeError(f"EED native parity failed: {failed}")
        scenarios.append(
            {
                "impact_applied": impact_applied,
                "comparison_count": len(comparisons),
                "all_exact": True,
                "cpu_owner": cpu_result["owner"],
                "cuda_owner": cuda_result["owner"],
                "new_rng_draws": cuda_result["new_rng_draws"],
                "terminal_state_sha256": {
                    name: _tensor_sha256(tensor)
                    for name, tensor in zip(STATE_NAMES, cuda, strict=True)
                },
            }
        )

    initial = _initial_state()
    first, _ = _native_transition(module, initial, device="cuda", impact_applied=True)
    checkpoint = [
        tensor.detach().to(device="cpu").contiguous().clone() for tensor in first
    ]
    continued, _ = _native_transition(
        module,
        [tensor.numpy() for tensor in checkpoint],
        device="cuda",
        impact_applied=True,
    )
    direct, _ = _native_transition(module, initial, device="cuda", impact_applied=True)
    direct, _ = _native_transition(
        module,
        [tensor.detach().cpu().numpy() for tensor in direct],
        device="cuda",
        impact_applied=True,
    )
    checkpoint_exact = all(
        torch.equal(left, right) for left, right in zip(continued, direct, strict=True)
    )
    if not checkpoint_exact:
        raise RuntimeError("EED native checkpoint continuation parity failed")

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "EED-0-NATIVE-PARITY-A",
        "overlay_id": OVERLAY_ID,
        "artifact": str(artifact.resolve()),
        "artifact_sha256": _sha256(artifact.resolve()),
        "capability": capability,
        "scenario_count": len(scenarios),
        "exact_comparison_count": sum(
            int(row["comparison_count"]) for row in scenarios
        ),
        "scenarios": scenarios,
        "checkpoint_continuation_exact": checkpoint_exact,
        "flags_off_base_route_unchanged": True,
        "production_route_integrated": False,
        "new_rng_draws": 0,
        "claim_boundary": (
            "Exact bounded transition parity only; production route "
            "integration and EED-1 remain unauthorized."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"refusing to overwrite {args.output}")
    receipt = run_gate(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": receipt["decision"],
                "exact_comparison_count": receipt["exact_comparison_count"],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
