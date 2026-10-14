#!/usr/bin/env python3
"""Run one strict CPU/CUDA Jungle v4 parity gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import torch


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ENGINE: Final[Path] = ROOT / "engine"
for _path in (ROOT, ENGINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from engine.native_runtime.four_way_interchange import (  # noqa: E402
    _projection_digest_first_difference,
)
from engine.campaigns.patch_runner_factory import (  # noqa: E402
    build_patch_jungle_fixed_mlp_runner_v1,
    prepare_patch_generation_zero_v1,
)
from engine.perf.cuda_architecture_b_direct_resident_jungle_v4 import (  # noqa: E402
    JUNGLE_STATE_FIELDS_V4,
    build_prepared_direct_resident_jungle_v4,
    execute_direct_resident_jungle_physical_v4,
    execute_direct_resident_jungle_v4,
    project_direct_resident_jungle_checkpoint_v4,
)
from engine.simulation.runner import SimulationRunner  # noqa: E402
from perf import cxxbackbone10_resident_state_owner as cpu_owner_api  # noqa: E402


SCHEMA_VERSION: Final[str] = "eiel_cuda_architecture_b_jungle_gate_v4"
CUDA_MODULE_NAME: Final[str] = "eiel_cuda_architecture_b_flattened_wdl_controller_v2"
ALLOWED_GENERATIONS: Final[frozenset[int]] = frozenset({1, 2, *range(3, 33)})
TICKS_PER_GENERATION: Final[int] = 80
POPULATION: Final[int] = 64
GRID_SIZE: Final[int] = 16
MAX_BIRTHS_PER_TICK: Final[int] = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_generation_count(value: int) -> int:
    if type(value) is not int or value not in ALLOWED_GENERATIONS:
        raise ValueError("generations must be an integer in [1,32]")
    return value


def _require_data_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != Path("/data") and Path("/data") not in resolved.parents:
        raise ValueError(f"{label} must be under /data: {resolved}")
    return resolved


def _repository_source_identity() -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot bind Jungle qualification source: {exc}") from exc
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError("Jungle qualification source commit is malformed")
    return {
        "schema_version": "eiel_repository_source_identity_v1",
        "git_head": head,
        "worktree_clean": status == "",
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _load_cuda_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(CUDA_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load CUDA extension: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cuda_artifact_receipt(path: Path) -> dict[str, Any]:
    artifact = path.expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    manifest_path = Path(f"{artifact}.manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(f"CUDA artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_sha256 = _sha256(artifact)
    if manifest.get("artifact_sha256") != artifact_sha256:
        raise RuntimeError("CUDA artifact SHA-256 does not match its manifest")
    required = {
        "schema_version": (
            "eiel_cuda_architecture_b_flattened_jungle_controller_extension_manifest_v4"
        ),
        "module_name": CUDA_MODULE_NAME,
        "source_snapshot_stable": True,
        "runtime_jit_compilation": False,
        "runtime_jit_required_by_validation_route": False,
        "sass_only": True,
    }
    for name, expected in required.items():
        actual = manifest.get(name)
        if type(actual) is not type(expected) or actual != expected:
            raise RuntimeError(
                f"CUDA manifest {name} mismatch: "
                f"expected={expected!r} actual={actual!r}"
            )
    source_hashes = dict(manifest.get("source_hashes") or {})
    if not source_hashes:
        raise RuntimeError("CUDA manifest has no source hashes")
    source_fingerprint = hashlib.sha256(
        json.dumps(
            source_hashes,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    mismatches: list[dict[str, Any]] = []
    for raw_path, expected_sha256 in sorted(source_hashes.items()):
        source = Path(raw_path)
        actual_sha256 = _sha256(source) if source.is_file() else None
        if actual_sha256 != expected_sha256:
            mismatches.append(
                {
                    "path": str(source),
                    "expected_sha256": expected_sha256,
                    "actual_sha256": actual_sha256,
                }
            )
    if mismatches:
        raise RuntimeError(
            "CUDA artifact source snapshot no longer matches current source: "
            f"{mismatches[0]}"
        )
    return {
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha256,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "manifest_schema_version": manifest["schema_version"],
        "source_file_count": len(source_hashes),
        "source_fingerprint_sha256": source_fingerprint,
        "source_snapshot_stable": True,
        "sass_only": True,
        "runtime_jit": False,
    }


def _build_current_cpu_owner(build_directory: Path) -> tuple[Any, dict[str, Any]]:
    build_directory.mkdir(parents=True, exist_ok=True)
    cpu_owner_api._MODULE = None
    module = cpu_owner_api.load_resident_state_owner_extension(
        build_directory=build_directory,
    )
    extension = Path(module.__file__).resolve()
    source_manifest = cpu_owner_api.native_extension_compile_source_manifest()
    return module, {
        "artifact": str(extension),
        "artifact_sha256": _sha256(extension),
        "source_manifest": source_manifest,
    }


def _new_jungle_runner(
    run_directory: Path,
    *,
    seed: int,
    generations: int,
    cuda_artifact: Path,
) -> Any:
    artifact = cuda_artifact.expanduser().resolve()
    manifest = Path(f"{artifact}.manifest.json")

    def gate_runner_constructor(**kwargs: Any) -> Any:
        # Preserve the gate's public seed as its engine RNG root. Declarative
        # campaigns normally derive this root from all named seed owners.
        kwargs["sim_cfg"].seed = seed
        runner = SimulationRunner(**kwargs)
        prepare_patch_generation_zero_v1(runner)
        return runner

    built = build_patch_jungle_fixed_mlp_runner_v1(
        {
            "cell_id": f"jungle-v4-seed-{seed}",
            "parameters": {
                "patch.grid_size": GRID_SIZE,
                "patch.initial_population": POPULATION,
                "patch.capacity": POPULATION * 6,
                "patch.founder_placement": "legacy",
            },
            "seeds": {"ecology": seed},
        },
        run_directory,
        generations=max(3, generations),
        ticks_per_generation=TICKS_PER_GENERATION,
        artifact_path=str(artifact),
        qualification_receipt_path=str(manifest),
        reporting_receipt_path=str(manifest),
        max_births_per_generation=MAX_BIRTHS_PER_TICK,
        runner_constructor=gate_runner_constructor,
    )
    return built.runner


def _dispatch_contract(
    result: Mapping[str, Any],
    *,
    generations: int,
) -> dict[str, Any]:
    expected = {
        "jungle_native_owner": True,
        "jungle_python_tick_or_generation_ownership": False,
        "hidden_fallback": False,
        "runtime_jit": False,
    }
    if generations <= 2:
        expected["jungle_physical_gate_entrypoint"] = True
    else:
        expected.update(
            {
                "completed_generations": generations,
                "logical_generation_count": generations,
                "external_python_window_boundaries": 0,
                "direct_cpp_window_core_dispatch": True,
                "production_native_chunk_loop": True,
            }
        )
    mismatches = {
        name: {"expected": value, "actual": result.get(name)}
        for name, value in expected.items()
        if type(result.get(name)) is not type(value) or result.get(name) != value
    }
    return {
        "ok": not mismatches,
        "python_native_dispatches": 1,
        "hidden_fallback": result.get("hidden_fallback"),
        "runtime_jit": result.get("runtime_jit"),
        "mismatches": mismatches,
    }


def _reporting_status(prepared: Any) -> dict[str, Any]:
    # Jungle's ninth integer column is the environment RNG draw count, not a
    # status field. The extension fails closed before returning on controller
    # status errors, so report metric-row coverage here instead.
    tick_integer = prepared.reporting_tensors[2]
    return {
        "ok": True,
        "validated_by": "native_entrypoint_fail_closed_status_contract",
        "tick_rows": int(tick_integer.size(1)),
        "environment_rng_draws": int(tick_integer[..., -1].sum().cpu().item()),
    }


def _event_diagnostics(
    left_projection: Mapping[str, Any],
    right_projection: Mapping[str, Any],
) -> dict[str, Any]:
    left_metadata = dict(left_projection.get("metadata") or {})
    right_metadata = dict(right_projection.get("metadata") or {})
    left_tensors = dict(left_projection.get("tensors") or {})
    right_tensors = dict(right_projection.get("tensors") or {})
    prefix = "typed_lifecycle_event_"
    fields = sorted(
        name
        for name in set(left_tensors) & set(right_tensors)
        if name.startswith(prefix)
    )
    left_count = int(left_metadata.get("typed_lifecycle_event_active_count", 0))
    right_count = int(right_metadata.get("typed_lifecycle_event_active_count", 0))

    def row(tensors: Mapping[str, Any], index: int) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in fields:
            tensor = tensors[name]
            if isinstance(tensor, torch.Tensor) and index < int(tensor.shape[0]):
                out[name.removeprefix(prefix)] = tensor[index].item()
        return out

    first_row_difference = None
    for index in range(min(left_count, right_count)):
        left_row = row(left_tensors, index)
        right_row = row(right_tensors, index)
        if left_row != right_row:
            first_row_difference = {
                "index": index,
                "cpu": left_row,
                "cuda": right_row,
            }
            break
    if first_row_difference is None and left_count != right_count:
        index = min(left_count, right_count)
        first_row_difference = {
            "index": index,
            "cpu": row(left_tensors, index) if index < left_count else None,
            "cuda": row(right_tensors, index) if index < right_count else None,
        }

    def event_type_counts(tensors: Mapping[str, Any], count: int) -> dict[str, int]:
        tensor = tensors.get(f"{prefix}event_type")
        if not isinstance(tensor, torch.Tensor):
            return {}
        values = tensor[:count].detach().cpu().tolist()
        return {str(value): values.count(value) for value in sorted(set(values))}

    return {
        "cpu_active_count": left_count,
        "cuda_active_count": right_count,
        "first_row_difference": first_row_difference,
        "cpu_rows": [row(left_tensors, index) for index in range(min(left_count, 5))],
        "cuda_rows": [
            row(right_tensors, index) for index in range(min(right_count, 5))
        ],
        "cpu_event_type_counts": event_type_counts(left_tensors, left_count),
        "cuda_event_type_counts": event_type_counts(right_tensors, right_count),
    }


def _non_event_first_difference(
    left_projection: Mapping[str, Any],
    right_projection: Mapping[str, Any],
) -> dict[str, Any] | None:
    def stripped(projection: Mapping[str, Any]) -> dict[str, Any]:
        metadata = {
            name: value
            for name, value in dict(projection.get("metadata") or {}).items()
            if "event" not in name
        }
        tensors = {
            name: value
            for name, value in dict(projection.get("tensors") or {}).items()
            if not name.startswith(("event_tape_", "typed_lifecycle_event_"))
        }
        return {**dict(projection), "metadata": metadata, "tensors": tensors}

    return _projection_digest_first_difference(
        stripped(left_projection),
        stripped(right_projection),
    )


def _trajectory_tensor_first_difference(
    left_projection: Mapping[str, Any],
    right_projection: Mapping[str, Any],
) -> dict[str, Any] | None:
    left = dict(left_projection.get("tensors") or {})
    right = dict(right_projection.get("tensors") or {})
    for name in sorted(set(left) | set(right)):
        if name.startswith(("event_tape_", "typed_lifecycle_event_")):
            continue
        if name not in left or name not in right:
            return {
                "field": name,
                "cpu_present": name in left,
                "cuda_present": name in right,
            }
        left_tensor = left[name]
        right_tensor = right[name]
        if not isinstance(left_tensor, torch.Tensor) or not isinstance(
            right_tensor, torch.Tensor
        ):
            continue
        if (
            left_tensor.dtype != right_tensor.dtype
            or left_tensor.shape != right_tensor.shape
        ):
            return {
                "field": name,
                "cpu_dtype": str(left_tensor.dtype),
                "cuda_dtype": str(right_tensor.dtype),
                "cpu_shape": list(left_tensor.shape),
                "cuda_shape": list(right_tensor.shape),
            }
        unequal = left_tensor.detach().cpu() != right_tensor.detach().cpu()
        if left_tensor.is_floating_point():
            unequal &= ~(
                torch.isnan(left_tensor.detach().cpu())
                & torch.isnan(right_tensor.detach().cpu())
            )
        if bool(torch.any(unequal).item()):
            index = tuple(int(value) for value in torch.nonzero(unequal)[0].tolist())
            return {
                "field": name,
                "index": list(index),
                "cpu": left_tensor.detach().cpu()[index].item(),
                "cuda": right_tensor.detach().cpu()[index].item(),
                "unequal_count": int(torch.count_nonzero(unequal).item()),
            }
    return None


def _trajectory_tensor_difference_summary(
    left_projection: Mapping[str, Any],
    right_projection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    left = dict(left_projection.get("tensors") or {})
    right = dict(right_projection.get("tensors") or {})
    differences: list[dict[str, Any]] = []
    for name in sorted(set(left) | set(right)):
        if name.startswith(("event_tape_", "typed_lifecycle_event_")):
            continue
        if name not in left or name not in right:
            differences.append(
                {
                    "field": name,
                    "cpu_present": name in left,
                    "cuda_present": name in right,
                }
            )
            continue
        left_tensor = left[name]
        right_tensor = right[name]
        if not isinstance(left_tensor, torch.Tensor) or not isinstance(
            right_tensor, torch.Tensor
        ):
            continue
        if (
            left_tensor.dtype != right_tensor.dtype
            or left_tensor.shape != right_tensor.shape
        ):
            differences.append(
                {
                    "field": name,
                    "cpu_dtype": str(left_tensor.dtype),
                    "cuda_dtype": str(right_tensor.dtype),
                    "cpu_shape": list(left_tensor.shape),
                    "cuda_shape": list(right_tensor.shape),
                }
            )
            continue
        left_cpu = left_tensor.detach().cpu()
        right_cpu = right_tensor.detach().cpu()
        unequal = left_cpu != right_cpu
        if left_tensor.is_floating_point():
            unequal &= ~(torch.isnan(left_cpu) & torch.isnan(right_cpu))
        if not bool(torch.any(unequal).item()):
            continue
        index = tuple(int(value) for value in torch.nonzero(unequal)[0].tolist())
        differences.append(
            {
                "field": name,
                "index": list(index),
                "cpu": left_cpu[index].item(),
                "cuda": right_cpu[index].item(),
                "unequal_count": int(torch.count_nonzero(unequal).item()),
            }
        )
    return differences


def _rng_metadata_differences(
    left_projection: Mapping[str, Any],
    right_projection: Mapping[str, Any],
) -> dict[str, Any]:
    left = dict(left_projection.get("metadata") or {})
    right = dict(right_projection.get("metadata") or {})
    names = sorted(
        name for name in set(left) | set(right) if "numpy" in name and "state" in name
    )
    return {
        name: {"cpu": left.get(name), "cuda": right.get(name)}
        for name in names
        if left.get(name) != right.get(name)
    }


def _scalar_result_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in sorted(result.items())
        if value is None or type(value) in {bool, int, float, str}
    }


def execute_gate(
    *,
    seed: int,
    generations: int,
    output_root: Path,
    cuda_artifact: Path,
    cpu_build_directory: Path,
    ticks_per_generation: int = TICKS_PER_GENERATION,
) -> dict[str, Any]:
    source_identity = _repository_source_identity()
    generations = _require_generation_count(generations)
    if (
        type(ticks_per_generation) is not int
        or ticks_per_generation < 1
        or ticks_per_generation > TICKS_PER_GENERATION
    ):
        raise ValueError("ticks_per_generation must be in [1,80]")
    if generations > 2 and ticks_per_generation != TICKS_PER_GENERATION:
        raise ValueError("resident G3-G32 gates require exactly 80 ticks")
    output_root = _require_data_path(output_root, label="output root")
    cpu_build_directory = _require_data_path(
        cpu_build_directory,
        label="CPU build directory",
    )
    case_directory = output_root / (
        f"jungle-v4-seed-{seed}-g{generations}-t{ticks_per_generation}"
    )
    if case_directory.exists():
        raise FileExistsError(case_directory)
    case_directory.mkdir(parents=True)

    cuda_receipt = _cuda_artifact_receipt(cuda_artifact)
    cuda_module = _load_cuda_module(cuda_artifact.expanduser().resolve())
    cpu_module, cpu_receipt = _build_current_cpu_owner(cpu_build_directory)
    cpu_owner_api._MODULE = cpu_module

    runner = _new_jungle_runner(
        case_directory / "source",
        seed=seed,
        generations=generations,
        cuda_artifact=cuda_artifact,
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    prepared = build_prepared_direct_resident_jungle_v4(
        runner,
        qualified_cuda_module=cuda_module,
        logical_generation_count=generations,
    )
    initial_projection = prepared.patch.checkpoint_projection_inputs[
        "pre_generation_canonical_projection"
    ]
    cpu_owner = cpu_owner_api.ResidentRuntimeState.from_checkpoint_projection(
        initial_projection,
        build_directory=cpu_build_directory,
    )

    cpu_started = time.perf_counter_ns()
    cpu_generation_results: list[dict[str, Any]] = []
    cpu_turnover_results: list[dict[str, Any]] = []
    for generation in range(generations):
        cpu_generation_results.append(
            cpu_owner.execute_generation_patch_with_population_birth_commit_cpu(
                generation,
                ticks_per_generation,
                MAX_BIRTHS_PER_TICK,
            )
        )
        cpu_turnover_results.append(
            cpu_owner.execute_population_generation_turnover_cpu()
        )
    cpu_wall_time_ns = time.perf_counter_ns() - cpu_started
    cpu_projection = cpu_owner.export_checkpoint_projection()

    torch.cuda.synchronize()
    cuda_started = time.perf_counter_ns()
    if generations <= 2:
        cuda_result = execute_direct_resident_jungle_physical_v4(
            cuda_module,
            prepared,
            generation_count=generations,
            ticks_per_generation=ticks_per_generation,
        )
    else:
        cuda_result = execute_direct_resident_jungle_v4(
            cuda_module,
            prepared,
            logical_generation_count=generations,
            run_id=f"jungle-v4-seed-{seed}-g{generations}",
        )
    torch.cuda.synchronize()
    cuda_wall_time_ns = time.perf_counter_ns() - cuda_started
    cuda_peak_allocated_bytes = int(torch.cuda.max_memory_allocated())
    cuda_peak_reserved_bytes = int(torch.cuda.max_memory_reserved())
    cuda_projection = project_direct_resident_jungle_checkpoint_v4(
        prepared,
        direct_result=cuda_result,
        completed_generation_count=generations,
    )

    comparison = dict(
        cpu_owner_api.compare_projection_digest(cpu_projection, cuda_projection)
    )
    first_difference = (
        None
        if comparison.get("ok") is True
        else _projection_digest_first_difference(cpu_projection, cuda_projection)
    )
    dispatch = _dispatch_contract(cuda_result, generations=generations)
    reporting = _reporting_status(prepared)
    jungle_state_sha256 = {
        name: hashlib.sha256(
            tensor.detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest()
        for name, tensor in zip(
            JUNGLE_STATE_FIELDS_V4,
            prepared.state_tensors,
            strict=True,
        )
    }
    ok = comparison.get("ok") is True and dispatch["ok"] and reporting["ok"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "decision": (
            f"EIEL-NATIVE-JUNGLE-CUDA-G{generations}-EXACT"
            if ok
            else f"EIEL-NATIVE-JUNGLE-CUDA-G{generations}-FIRST-DIFFERENCE"
        ),
        "claim_scope": "mechanical parity only; no scientific claim",
        "case_directory": str(case_directory),
        "git_head": source_identity["git_head"],
        "repository_source_identity": source_identity,
        "execution_source_hashes": {
            str(Path(__file__).resolve().relative_to(ROOT)): _sha256(
                Path(__file__).resolve()
            ),
            "engine/perf/cuda_architecture_b_direct_resident_jungle_v4.py": (
                _sha256(
                    ROOT / "engine/perf/"
                    "cuda_architecture_b_direct_resident_jungle_v4.py"
                )
            ),
        },
        "seed": seed,
        "generations": generations,
        "ticks_per_generation": ticks_per_generation,
        "population": POPULATION,
        "grid_size": GRID_SIZE,
        "max_births_per_tick": MAX_BIRTHS_PER_TICK,
        "cpu": {
            **cpu_receipt,
            "wall_time_ns": cpu_wall_time_ns,
            "generation_count": len(cpu_generation_results),
            "turnover_count": len(cpu_turnover_results),
            "hidden_fallback": any(
                result.get("hidden_fallback") is not False
                for result in (*cpu_generation_results, *cpu_turnover_results)
            ),
            "generation_metrics": [
                {
                    name: result.get(name)
                    for name in (
                        "env_rng_padding_draws",
                        "jungle_predator_exposure_count",
                        "jungle_predator_events",
                        "jungle_predator_deaths",
                        "jungle_birth_emission_count",
                    )
                }
                for result in cpu_generation_results
            ],
            "generation_scalar_results": [
                _scalar_result_fields(result) for result in cpu_generation_results
            ],
        },
        "cuda": {
            **cuda_receipt,
            "wall_time_ns": cuda_wall_time_ns,
            "peak_allocated_bytes": cuda_peak_allocated_bytes,
            "peak_reserved_bytes": cuda_peak_reserved_bytes,
            "dispatch": dispatch,
            "native_result": {
                name: cuda_result.get(name)
                for name in (
                    "schema_version",
                    "owner",
                    "logical_generation_count",
                    "completed_generations",
                    "native_tick_iterations",
                    "window_count",
                    "hidden_fallback",
                    "runtime_jit",
                    "jungle_native_owner",
                    "jungle_python_tick_or_generation_ownership",
                )
            },
        },
        "comparison": comparison,
        "first_difference": first_difference,
        "event_diagnostics": _event_diagnostics(cpu_projection, cuda_projection),
        "non_event_first_difference": _non_event_first_difference(
            cpu_projection,
            cuda_projection,
        ),
        "trajectory_tensor_first_difference": _trajectory_tensor_first_difference(
            cpu_projection,
            cuda_projection,
        ),
        "trajectory_tensor_differences": _trajectory_tensor_difference_summary(
            cpu_projection,
            cuda_projection,
        ),
        "rng_metadata_differences": _rng_metadata_differences(
            cpu_projection,
            cuda_projection,
        ),
        "reporting_status": reporting,
        "jungle_state_sha256": jungle_state_sha256,
        "jungle_tick_integer_metrics": prepared.reporting_tensors[2][
            :, : generations * ticks_per_generation
        ]
        .detach()
        .cpu()
        .tolist(),
        "cuda_generation_summary_scalar": (
            cuda_result["generation_summary_scalar"].detach().cpu().tolist()
            if isinstance(cuda_result.get("generation_summary_scalar"), torch.Tensor)
            else None
        ),
        "cuda_window_receipts": cuda_result.get("window_receipts"),
        "cuda_gaussian_refill_receipts": cuda_result.get(
            "gaussian_refill_receipts"
        ),
        "cuda_policy_refill_receipts": cuda_result.get("policy_refill_receipts"),
        "cuda_birth_output_diagnostics": {
            name: prepared.patch.source.outputs[name][
                :, : generations * ticks_per_generation
            ]
            .detach()
            .cpu()
            .tolist()
            for name in (
                "birth_identity_committed_count",
                "birth_committed_child_slots",
                "birth_committed_child_agent_ids",
                "birth_combined_status",
            )
        },
    }
    result_path = case_directory / "result.json"
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["result_path"] = str(result_path)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--generations",
        type=int,
        choices=tuple(sorted(ALLOWED_GENERATIONS)),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cuda-artifact", type=Path, required=True)
    parser.add_argument("--cpu-build-dir", type=Path, required=True)
    parser.add_argument(
        "--ticks-per-generation",
        type=int,
        default=TICKS_PER_GENERATION,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = execute_gate(
        seed=args.seed,
        generations=args.generations,
        output_root=args.output_root,
        cuda_artifact=args.cuda_artifact,
        cpu_build_directory=args.cpu_build_dir,
        ticks_per_generation=args.ticks_per_generation,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
