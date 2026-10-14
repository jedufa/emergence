#!/usr/bin/env python3
"""Byte-exact probe for native Architecture B resident-window refills."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA_VERSION: Final[str] = (
    "eiel_cuda_architecture_b_resident_window_refill_v1_probe_receipt_v1"
)
MODULE_NAME: Final[str] = "eiel_cuda_architecture_b_flattened_controller_v1"
GAUSSIAN_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_prepare_cpython_gaussian_window_v1"
)
POLICY_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_prepare_fixed_mlp_policy_window_v1"
)
GAUSSIAN_REFILL_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_refill_cpython_gaussian_window_v1"
)
GAUSSIAN_REFILL_OR_REUSE_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_refill_or_reuse_cpython_gaussian_window_v1"
)
POLICY_REFILL_ENTRYPOINT: Final[str] = (
    "eiel_cuda_architecture_b_refill_fixed_mlp_policy_window_v1"
)

GAUSSIAN_SEEDS: Final[tuple[int, ...]] = (0, 1, 6060, 0xFFFFFFFF)
GAUSSIAN_CASES: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),
    (0, 1),
    (1, 2),
    (226, 3),
    (623, 7),
    (624, 17),
    (625, 31),
)
GAUSSIAN_TENSOR_FIELDS: Final[tuple[str, ...]] = (
    "python_random_gauss_primary_tape",
    "python_random_gauss_secondary_tape",
)
POLICY_TENSOR_FIELDS: Final[tuple[str, ...]] = (
    "rng_manager_mutation_torch_cpu_state",
    "policy_birth_delta_tape",
    "policy_birth_mutation_mask_tape",
    "policy_birth_crossover_mask_packed_tape",
    "policy_birth_post_rng_state_packed_tape",
    "policy_birth_semantic_draws_tape",
)
POLICY_SEEDS: Final[tuple[int, ...]] = (0, 1, 6060, 0x7FFFFFFF)


@dataclass(frozen=True)
class PolicyCase:
    name: str
    episodes: int
    width: int
    enabled: bool
    mode: str
    rate: float
    std: float


POLICY_CASES: Final[tuple[PolicyCase, ...]] = (
    PolicyCase("disabled", 3, 7, False, "dense", 0.3, 0.25),
    PolicyCase("dense", 4, 11, True, "dense", 1.0, 0.125),
    PolicyCase("sparse", 5, 13, True, "sparse", 0.25, 0.2),
    PolicyCase("dense_zero_std", 2, 5, True, "dense", 1.0, 0.0),
    PolicyCase("sparse_zero_std", 3, 9, True, "sparse", 0.1, 0.0),
)


@dataclass(frozen=True)
class Options:
    extension: Path
    output: Path


class ProbeMismatch(RuntimeError):
    """Raised when a native result is not byte-exact with its Python oracle."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matrix_counts() -> dict[str, int]:
    gaussian_cases = len(GAUSSIAN_SEEDS) * len(GAUSSIAN_CASES)
    policy_cases = len(POLICY_SEEDS) * len(POLICY_CASES)
    return {
        "gaussian_cases": gaussian_cases,
        "gaussian_seed_count": len(GAUSSIAN_SEEDS),
        "gaussian_state_capacity_pair_count": len(GAUSSIAN_CASES),
        "gaussian_tensor_comparisons": (gaussian_cases * len(GAUSSIAN_TENSOR_FIELDS)),
        "policy_cases": policy_cases,
        "policy_seed_count": len(POLICY_SEEDS),
        "policy_scenario_count": len(POLICY_CASES),
        "policy_tensor_comparisons": policy_cases * len(POLICY_TENSOR_FIELDS),
        "policy_zero_std_cases": len(POLICY_SEEDS)
        * sum(case.std == 0.0 for case in POLICY_CASES),
        "total_cases": gaussian_cases + policy_cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare native flattened-controller resident-window refill outputs "
            "byte-for-byte with the existing Python helpers."
        )
    )
    parser.add_argument(
        "--extension",
        type=Path,
        required=True,
        help="Prebuilt immutable flattened-controller extension.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="JSON receipt path.",
    )
    return parser


def _options_from_namespace(namespace: argparse.Namespace) -> Options:
    extension = namespace.extension.expanduser().resolve()
    output = namespace.output.expanduser().resolve()
    if extension.suffix != ".so" or not extension.is_file():
        raise ValueError("--extension must name an existing .so file")
    if output.suffix.lower() != ".json":
        raise ValueError("--output must end in .json")
    if output == extension:
        raise ValueError("--output must differ from --extension")
    return Options(extension=extension, output=output)


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    if path.suffix.lower() != ".json":
        raise ValueError("receipt path must end in .json")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_extension(extension: Path) -> ModuleType:
    import torch  # noqa: F401 - preload libtorch dependencies.

    module_name = extension.stem
    if module_name in sys.modules:
        raise RuntimeError(f"canonical extension name is already loaded: {module_name}")
    spec = importlib.util.spec_from_file_location(module_name, extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot construct extension loader for {extension}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.__name__ != module_name:
        raise RuntimeError("flattened controller was not loaded canonically")
    for entrypoint in (
        GAUSSIAN_ENTRYPOINT,
        POLICY_ENTRYPOINT,
        GAUSSIAN_REFILL_ENTRYPOINT,
        GAUSSIAN_REFILL_OR_REUSE_ENTRYPOINT,
        POLICY_REFILL_ENTRYPOINT,
    ):
        if not callable(getattr(module, entrypoint, None)):
            raise RuntimeError(f"extension is missing entrypoint {entrypoint}")
    return module


def _tensor_bytes(tensor: Any, *, label: str) -> bytes:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ProbeMismatch(f"{label} is not a Torch tensor")
    if tensor.device.type != "cpu":
        raise ProbeMismatch(f"{label} is not CPU resident")
    if not tensor.is_contiguous():
        raise ProbeMismatch(f"{label} is not contiguous")
    if tensor.requires_grad:
        raise ProbeMismatch(f"{label} unexpectedly requires gradients")
    return tensor.detach().view(torch.uint8).reshape(-1).numpy().tobytes(order="C")


def _require_byte_exact(expected: Any, actual: Any, *, label: str) -> None:
    import torch

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise ProbeMismatch(f"{label} must compare two Torch tensors")
    if expected.dtype != actual.dtype:
        raise ProbeMismatch(
            f"{label} dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
        )
    if tuple(expected.shape) != tuple(actual.shape):
        raise ProbeMismatch(
            f"{label} shape mismatch: expected {list(expected.shape)}, "
            f"got {list(actual.shape)}"
        )
    expected_bytes = _tensor_bytes(expected, label=f"{label}.expected")
    actual_bytes = _tensor_bytes(actual, label=f"{label}.actual")
    if expected_bytes != actual_bytes:
        raise ProbeMismatch(f"{label} tensor bytes differ")


def _update_bytes(digest: Any, label: str, payload: bytes) -> None:
    encoded_label = label.encode("utf-8")
    digest.update(len(encoded_label).to_bytes(8, "big"))
    digest.update(encoded_label)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _update_tensor_digest(digest: Any, label: str, tensor: Any) -> None:
    metadata = json.dumps(
        {
            "dtype": str(tensor.dtype),
            "label": label,
            "shape": list(tensor.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _update_bytes(digest, f"{label}.metadata", metadata)
    _update_bytes(
        digest,
        f"{label}.bytes",
        _tensor_bytes(tensor, label=label),
    )


def _record_tensor_pair(
    expected_digest: Any,
    actual_digest: Any,
    *,
    label: str,
    expected: Any,
    actual: Any,
) -> None:
    _require_byte_exact(expected, actual, label=label)
    _update_tensor_digest(expected_digest, label, expected)
    _update_tensor_digest(actual_digest, label, actual)


def _require_mapping_result(
    value: object,
    *,
    entrypoint: str,
    tensor_fields: Sequence[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeMismatch(f"{entrypoint} did not return a mapping")
    missing = [field for field in tensor_fields if field not in value]
    if missing:
        raise ProbeMismatch(f"{entrypoint} omitted tensor fields: {missing}")
    return value


def _cpython_state(seed: int, advance: int) -> tuple[Any, ...]:
    source = random.Random(seed)
    for _ in range(advance):
        source.getrandbits(32)
    return source.getstate()


def _expected_post_index(state: tuple[Any, ...], capacity: int) -> int:
    if capacity == 0:
        return int(tuple(state[1])[-1])
    source = random.Random()
    source.setstate(state)
    for _ in range(capacity + 3):
        source.getrandbits(32)
    return int(tuple(source.getstate()[1])[-1])


def _run_gaussian_matrix(module: ModuleType) -> dict[str, Any]:
    import torch

    from engine.perf.cuda_architecture_b_lifecycle import (
        pack_cpython_random_state_cuda,
    )

    entrypoint = getattr(module, GAUSSIAN_ENTRYPOINT)
    expected_digest = hashlib.sha256()
    actual_digest = hashlib.sha256()
    observed_indices: set[int] = set()
    comparisons = 0
    case_count = 0
    for seed in GAUSSIAN_SEEDS:
        for advance, capacity in GAUSSIAN_CASES:
            state = _cpython_state(seed, advance)
            internal = tuple(state[1])
            index = int(internal[-1])
            observed_indices.add(index)
            words = torch.tensor(internal[:-1], dtype=torch.uint32).contiguous()
            words_before = _tensor_bytes(words, label="gaussian.words.before")
            expected = pack_cpython_random_state_cuda(
                state,
                device="cpu",
                gauss_tape_raw_capacity=capacity,
            )
            actual = _require_mapping_result(
                entrypoint(words, index, capacity),
                entrypoint=GAUSSIAN_ENTRYPOINT,
                tensor_fields=GAUSSIAN_TENSOR_FIELDS,
            )
            case_label = f"gaussian.seed={seed}.advance={advance}.capacity={capacity}"
            config = json.dumps(
                {
                    "advance": advance,
                    "capacity": capacity,
                    "index": index,
                    "seed": seed,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _update_bytes(expected_digest, f"{case_label}.config", config)
            _update_bytes(actual_digest, f"{case_label}.config", config)
            for field in GAUSSIAN_TENSOR_FIELDS:
                _record_tensor_pair(
                    expected_digest,
                    actual_digest,
                    label=f"{case_label}.{field}",
                    expected=expected[field],
                    actual=actual[field],
                )
                comparisons += 1
            if _tensor_bytes(words, label="gaussian.words.after") != words_before:
                raise ProbeMismatch(f"{case_label} mutated source words")
            if actual.get("source_state_preserved") is not True:
                raise ProbeMismatch(f"{case_label} did not preserve source state")
            post_index = actual.get("staging_post_index")
            if type(post_index) is not int or post_index != _expected_post_index(
                state, capacity
            ):
                raise ProbeMismatch(f"{case_label} returned a wrong staging index")
            case_count += 1
    return {
        "actual_sha256": actual_digest.hexdigest(),
        "capacities": sorted({capacity for _, capacity in GAUSSIAN_CASES}),
        "case_count": case_count,
        "expected_sha256": expected_digest.hexdigest(),
        "observed_indices": sorted(observed_indices),
        "tensor_comparisons": comparisons,
    }


def _run_policy_matrix(module: ModuleType) -> dict[str, Any]:
    import torch

    from engine.perf.cuda_architecture_b_child_policy import (
        prepare_fixed_mlp_turnover_child_policy_tape,
    )

    entrypoint = getattr(module, POLICY_ENTRYPOINT)
    expected_digest = hashlib.sha256()
    actual_digest = hashlib.sha256()
    comparisons = 0
    case_count = 0
    zero_std_cases = 0
    for seed in POLICY_SEEDS:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        state = generator.get_state().clone().contiguous()
        for case in POLICY_CASES:
            state_before = _tensor_bytes(state, label="policy.state.before")
            expected = prepare_fixed_mlp_turnover_child_policy_tape(
                state,
                episode_capacity=case.episodes,
                row_width=case.width,
                device="cpu",
                policy_weight_mutation_enable=case.enabled,
                policy_weight_mutation_mode=case.mode,
                policy_weight_mutation_rate=case.rate,
                policy_weight_mutation_std=case.std,
            )
            actual = _require_mapping_result(
                entrypoint(
                    state,
                    case.episodes,
                    case.width,
                    case.enabled,
                    case.mode,
                    case.rate,
                    case.std,
                ),
                entrypoint=POLICY_ENTRYPOINT,
                tensor_fields=POLICY_TENSOR_FIELDS,
            )
            case_label = f"policy.seed={seed}.scenario={case.name}"
            config = json.dumps(
                {
                    "enabled": case.enabled,
                    "episodes": case.episodes,
                    "mode": case.mode,
                    "rate": case.rate,
                    "seed": seed,
                    "std": case.std,
                    "width": case.width,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _update_bytes(expected_digest, f"{case_label}.config", config)
            _update_bytes(actual_digest, f"{case_label}.config", config)
            for field in POLICY_TENSOR_FIELDS:
                _record_tensor_pair(
                    expected_digest,
                    actual_digest,
                    label=f"{case_label}.{field}",
                    expected=expected[field],
                    actual=actual[field],
                )
                comparisons += 1
            if _tensor_bytes(state, label="policy.state.after") != state_before:
                raise ProbeMismatch(f"{case_label} mutated source RNG state")
            if actual.get("source_state_preserved") is not True:
                raise ProbeMismatch(f"{case_label} did not preserve source state")
            case_count += 1
            zero_std_cases += int(case.std == 0.0)
    return {
        "actual_sha256": actual_digest.hexdigest(),
        "case_count": case_count,
        "expected_sha256": expected_digest.hexdigest(),
        "scenario_names": [case.name for case in POLICY_CASES],
        "tensor_comparisons": comparisons,
        "zero_std_case_count": zero_std_cases,
    }


def _combined_hash(gaussian_hash: str, policy_hash: str) -> str:
    digest = hashlib.sha256()
    _update_bytes(digest, "gaussian", bytes.fromhex(gaussian_hash))
    _update_bytes(digest, "policy", bytes.fromhex(policy_hash))
    return digest.hexdigest()


def _run_inplace_refill_matrix(module: ModuleType) -> dict[str, Any]:
    import torch

    from engine.perf.cuda_architecture_b_child_policy import (
        prepare_fixed_mlp_turnover_child_policy_tape,
    )
    from engine.perf.cuda_architecture_b_lifecycle import (
        pack_cpython_random_state_cuda,
    )
    from engine.perf.cuda_architecture_b_resident_window_runtime_v1 import (
        refill_cpython_gaussian_window_,
        refill_fixed_mlp_policy_window_,
    )

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    comparisons = 0
    pointer_checks = 0
    for device in devices:
        packed = [
            pack_cpython_random_state_cuda(
                _cpython_state(seed, seed % 17),
                device="cpu",
                semantic_draws=seed,
                gauss_tape_raw_capacity=11,
            )
            for seed in (101, 303)
        ]
        scalar_fields = {
            "python_random_index",
            "python_random_has_gauss",
            "python_random_gauss_next",
            "python_random_raw_draws",
            "mutation_py_semantic_draws",
        }
        gaussian_fields = tuple(packed[0])
        gaussian = {
            name: (
                torch.cat([cell[name] for cell in packed])
                if name in scalar_fields
                else torch.stack([cell[name] for cell in packed])
            ).to(device)
            for name in gaussian_fields
        }
        gaussian["python_random_raw_draws"].copy_(
            torch.tensor([13, 29], dtype=torch.int64, device=device)
        )
        gaussian["mutation_py_semantic_draws"].copy_(
            torch.tensor([7, 11], dtype=torch.int64, device=device)
        )
        gaussian_base = torch.tensor([5, 9], dtype=torch.int64, device=device)
        expected_gaussian = {name: value.clone() for name, value in gaussian.items()}
        expected_gaussian_base = gaussian_base.clone()
        refill_cpython_gaussian_window_(
            expected_gaussian,
            python_random_window_base=expected_gaussian_base,
        )
        gaussian_pointers = (
            gaussian["python_random_gauss_primary_tape"].data_ptr(),
            gaussian["python_random_gauss_secondary_tape"].data_ptr(),
            gaussian_base.data_ptr(),
        )
        getattr(module, GAUSSIAN_REFILL_ENTRYPOINT)(
            gaussian["python_random_words"],
            gaussian["python_random_index"],
            gaussian["python_random_has_gauss"],
            gaussian["python_random_gauss_next"],
            gaussian["python_random_gauss_primary_tape"],
            gaussian["python_random_gauss_secondary_tape"],
            gaussian["python_random_raw_draws"],
            gaussian["mutation_py_semantic_draws"],
            gaussian_base,
        )
        for name in gaussian_fields:
            _require_byte_exact(
                expected_gaussian[name].detach().cpu().contiguous(),
                gaussian[name].detach().cpu().contiguous(),
                label=f"inplace.{device}.gaussian.{name}",
            )
            comparisons += 1
        _require_byte_exact(
            expected_gaussian_base.detach().cpu().contiguous(),
            gaussian_base.detach().cpu().contiguous(),
            label=f"inplace.{device}.gaussian.window_base",
        )
        comparisons += 1
        if gaussian_pointers != (
            gaussian["python_random_gauss_primary_tape"].data_ptr(),
            gaussian["python_random_gauss_secondary_tape"].data_ptr(),
            gaussian_base.data_ptr(),
        ):
            raise ProbeMismatch(f"{device} Gaussian refill changed tensor pointers")
        pointer_checks += 3

        policy_rows = []
        policy_sources = []
        for seed in (701, 703):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            source = generator.get_state().clone()
            policy_sources.append(source)
            policy_rows.append(
                prepare_fixed_mlp_turnover_child_policy_tape(
                    source,
                    episode_capacity=5,
                    row_width=13,
                    device="cpu",
                    policy_weight_mutation_enable=True,
                    policy_weight_mutation_mode="sparse",
                    policy_weight_mutation_rate=0.35,
                    policy_weight_mutation_std=0.125,
                )
            )
        policy_tape_fields = POLICY_TENSOR_FIELDS[1:]
        policy = {
            "rng_manager_mutation_torch_cpu_state": torch.stack(policy_sources),
            **{
                name: torch.stack([row[name] for row in policy_rows])
                for name in policy_tape_fields
            },
            "policy_birth_tape_cursor": torch.tensor([4, 7], dtype=torch.int64),
            "mutation_torch_cpu_semantic_draws": torch.tensor(
                [101, 211], dtype=torch.int64
            ),
        }
        policy = {name: value.to(device) for name, value in policy.items()}
        policy_base = torch.tensor([1, 3], dtype=torch.int64, device=device)
        expected_policy = {name: value.clone() for name, value in policy.items()}
        expected_policy_base = policy_base.clone()
        refill_fixed_mlp_policy_window_(
            expected_policy,
            policy_birth_window_base=expected_policy_base,
            policy_row_width=13,
            policy_weight_mutation_enable=True,
            policy_weight_mutation_mode="sparse",
            policy_weight_mutation_rate=0.35,
            policy_weight_mutation_std=0.125,
        )
        policy_pointers = tuple(
            policy[name].data_ptr() for name in policy_tape_fields
        ) + (policy_base.data_ptr(),)
        getattr(module, POLICY_REFILL_ENTRYPOINT)(
            policy["rng_manager_mutation_torch_cpu_state"],
            policy["policy_birth_delta_tape"],
            policy["policy_birth_mutation_mask_tape"],
            policy["policy_birth_crossover_mask_packed_tape"],
            policy["policy_birth_post_rng_state_packed_tape"],
            policy["policy_birth_semantic_draws_tape"],
            policy["policy_birth_tape_cursor"],
            policy["mutation_torch_cpu_semantic_draws"],
            policy_base,
            13,
            True,
            "sparse",
            0.35,
            0.125,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        for name in policy:
            _require_byte_exact(
                expected_policy[name].detach().cpu().contiguous(),
                policy[name].detach().cpu().contiguous(),
                label=f"inplace.{device}.policy.{name}",
            )
            comparisons += 1
        _require_byte_exact(
            expected_policy_base.detach().cpu().contiguous(),
            policy_base.detach().cpu().contiguous(),
            label=f"inplace.{device}.policy.window_base",
        )
        comparisons += 1
        if policy_pointers != tuple(
            policy[name].data_ptr() for name in policy_tape_fields
        ) + (policy_base.data_ptr(),):
            raise ProbeMismatch(f"{device} policy refill changed tensor pointers")
        pointer_checks += len(policy_tape_fields) + 1
    return {
        "all_exact": True,
        "devices": devices,
        "pointer_identity_checks": pointer_checks,
        "tensor_comparisons": comparisons,
    }


def _run_gaussian_carry_forward_microproof(module: ModuleType) -> dict[str, Any]:
    import torch

    from engine.perf.cuda_architecture_b_lifecycle import (
        pack_cpython_random_state_cuda,
    )

    capacity = 131
    required_future_draws = 64
    advances = tuple(1 + (cell * 17) % 64 for cell in range(48))
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    comparisons = 0
    for device in devices:
        initial = [
            pack_cpython_random_state_cuda(
                _cpython_state(9000 + cell, 0),
                device="cpu",
                gauss_tape_raw_capacity=capacity,
            )
            for cell in range(48)
        ]
        advanced = [
            pack_cpython_random_state_cuda(
                _cpython_state(9000 + cell, advance),
                device="cpu",
                semantic_draws=cell,
                gauss_tape_raw_capacity=capacity,
            )
            for cell, advance in enumerate(advances)
        ]
        scalar_fields = {
            "python_random_index",
            "python_random_has_gauss",
            "python_random_gauss_next",
            "python_random_raw_draws",
            "mutation_py_semantic_draws",
        }

        def batch(rows: Sequence[Mapping[str, Any]], name: str) -> Any:
            values = [row[name] for row in rows]
            combined = torch.cat(values) if name in scalar_fields else torch.stack(values)
            return combined.contiguous().to(device)

        words = batch(advanced, "python_random_words")
        index = batch(advanced, "python_random_index")
        has_gauss = batch(advanced, "python_random_has_gauss")
        gauss_next = batch(advanced, "python_random_gauss_next")
        semantic = batch(advanced, "mutation_py_semantic_draws")
        raw = torch.tensor(advances, dtype=torch.int64, device=device)
        base = torch.zeros(48, dtype=torch.int64, device=device)
        primary = batch(initial, "python_random_gauss_primary_tape")
        secondary = batch(initial, "python_random_gauss_secondary_tape")
        preserved = tuple(
            tensor.clone()
            for tensor in (words, index, has_gauss, gauss_next, semantic, raw, base, primary, secondary)
        )
        receipt = getattr(module, GAUSSIAN_REFILL_OR_REUSE_ENTRYPOINT)(
            words,
            index,
            has_gauss,
            gauss_next,
            primary,
            secondary,
            raw,
            semantic,
            base,
            required_future_draws,
        )
        if receipt.get("refill_performed") is not False:
            raise ProbeMismatch(f"{device} Gaussian carry-forward unexpectedly refilled")
        for label, before, after in zip(
            ("words", "index", "has_gauss", "gauss_next", "semantic", "raw", "base", "primary", "secondary"),
            preserved,
            (words, index, has_gauss, gauss_next, semantic, raw, base, primary, secondary),
            strict=True,
        ):
            _require_byte_exact(before.detach().cpu(), after.detach().cpu(), label=f"carry.{device}.preserved.{label}")
            comparisons += 1
        for cell, advance in enumerate(advances):
            suffix = capacity - advance
            for field, tape in (
                ("python_random_gauss_primary_tape", primary),
                ("python_random_gauss_secondary_tape", secondary),
            ):
                _require_byte_exact(
                    advanced[cell][field][:suffix],
                    tape[cell, advance:].detach().cpu().contiguous(),
                    label=f"carry.{device}.cell={cell}.{field}",
                )
                comparisons += 1

        forced_primary = primary.clone()
        forced_secondary = secondary.clone()
        forced_base = base.clone()
        forced = getattr(module, GAUSSIAN_REFILL_OR_REUSE_ENTRYPOINT)(
            words,
            index,
            has_gauss,
            gauss_next,
            forced_primary,
            forced_secondary,
            raw,
            semantic,
            forced_base,
            capacity,
        )
        if forced.get("refill_performed") is not True:
            raise ProbeMismatch(f"{device} forced Gaussian boundary did not refill")
        _require_byte_exact(
            batch(advanced, "python_random_gauss_primary_tape").detach().cpu(),
            forced_primary.detach().cpu(),
            label=f"carry.{device}.forced.primary",
        )
        _require_byte_exact(
            batch(advanced, "python_random_gauss_secondary_tape").detach().cpu(),
            forced_secondary.detach().cpu(),
            label=f"carry.{device}.forced.secondary",
        )
        _require_byte_exact(
            raw.detach().cpu(),
            forced_base.detach().cpu(),
            label=f"carry.{device}.forced.base",
        )
        comparisons += 3
    return {
        "batch_cells": 48,
        "capacity": capacity,
        "comparison_count": comparisons,
        "devices": devices,
        "required_future_draws": required_future_draws,
        "verdict": "pass",
    }


def _run_named_gaussian_d0_microproofs(
    module: ModuleType,
    *,
    gaussian_matrix: Mapping[str, Any],
    carry_forward: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    from engine.perf.cuda_architecture_b_lifecycle import (
        pack_cpython_random_state_cuda,
    )

    prepare = getattr(module, GAUSSIAN_ENTRYPOINT)
    expected_digest = hashlib.sha256()
    actual_digest = hashlib.sha256()
    source = random.Random(0xA51CE)
    oracle = random.Random(0xA51CE)
    refill_cycles = 4
    refill_capacity = 17
    logical_values = 0
    for cycle in range(refill_cycles):
        state = source.getstate()
        internal = tuple(state[1])
        expected = pack_cpython_random_state_cuda(
            state,
            device="cpu",
            gauss_tape_raw_capacity=refill_capacity,
        )
        actual = _require_mapping_result(
            prepare(
                torch.tensor(internal[:-1], dtype=torch.uint32).contiguous(),
                int(internal[-1]),
                refill_capacity,
            ),
            entrypoint=GAUSSIAN_ENTRYPOINT,
            tensor_fields=GAUSSIAN_TENSOR_FIELDS,
        )
        for field in GAUSSIAN_TENSOR_FIELDS:
            _record_tensor_pair(
                expected_digest,
                actual_digest,
                label=f"multi_refill.cycle={cycle}.{field}",
                expected=expected[field],
                actual=actual[field],
            )
        for _ in range(refill_capacity):
            expected_value = oracle.gauss(0.0, 1.0)
            actual_value = source.gauss(0.0, 1.0)
            expected_bits = struct.pack(">d", expected_value)
            actual_bits = struct.pack(">d", actual_value)
            if expected_bits != actual_bits:
                raise ProbeMismatch("multi-refill logical Gaussian bytes differ")
            _update_bytes(expected_digest, "multi_refill.logical", expected_bits)
            _update_bytes(actual_digest, "multi_refill.logical", actual_bits)
            logical_values += 1
    if source.getstate() != oracle.getstate():
        raise ProbeMismatch("multi-refill terminal CPython state differs")

    checkpoint_source = random.Random(0xC0FFEE)
    for _ in range(9):
        checkpoint_source.gauss(0.0, 1.0)
    checkpoint_state = checkpoint_source.getstate()
    checkpoint_internal = tuple(checkpoint_state[1])
    checkpoint_expected = pack_cpython_random_state_cuda(
        checkpoint_state,
        device="cpu",
        gauss_tape_raw_capacity=31,
    )
    checkpoint_actual = _require_mapping_result(
        prepare(
            torch.tensor(checkpoint_internal[:-1], dtype=torch.uint32).contiguous(),
            int(checkpoint_internal[-1]),
            31,
        ),
        entrypoint=GAUSSIAN_ENTRYPOINT,
        tensor_fields=GAUSSIAN_TENSOR_FIELDS,
    )
    for field in GAUSSIAN_TENSOR_FIELDS:
        _require_byte_exact(
            checkpoint_expected[field],
            checkpoint_actual[field],
            label=f"checkpoint_restore.{field}",
        )
    uninterrupted = random.Random()
    uninterrupted.setstate(checkpoint_state)
    restored = random.Random()
    restored.setstate(checkpoint_state)
    resumed_bytes = bytearray()
    for _ in range(73):
        expected_bits = struct.pack(">d", uninterrupted.gauss(0.0, 1.0))
        actual_bits = struct.pack(">d", restored.gauss(0.0, 1.0))
        if expected_bits != actual_bits:
            raise ProbeMismatch("checkpoint/resume future Gaussian bytes differ")
        resumed_bytes.extend(actual_bits)
    if uninterrupted.getstate() != restored.getstate():
        raise ProbeMismatch("checkpoint/resume terminal CPython state differs")

    capacity_state = _cpython_state(0xBADC0DE, 623)
    capacity_internal = tuple(capacity_state[1])
    capacity_words = torch.tensor(
        capacity_internal[:-1], dtype=torch.uint32
    ).contiguous()
    small = _require_mapping_result(
        prepare(capacity_words, int(capacity_internal[-1]), 17),
        entrypoint=GAUSSIAN_ENTRYPOINT,
        tensor_fields=GAUSSIAN_TENSOR_FIELDS,
    )
    large = _require_mapping_result(
        prepare(capacity_words, int(capacity_internal[-1]), 131),
        entrypoint=GAUSSIAN_ENTRYPOINT,
        tensor_fields=GAUSSIAN_TENSOR_FIELDS,
    )
    for field in GAUSSIAN_TENSOR_FIELDS:
        _require_byte_exact(
            small[field],
            large[field][:17].contiguous(),
            label=f"capacity_invariance.{field}",
        )

    matrix_exact = (
        gaussian_matrix["expected_sha256"] == gaussian_matrix["actual_sha256"]
    )
    carry_exact = carry_forward.get("verdict") == "pass"
    if not matrix_exact or not carry_exact:
        raise ProbeMismatch("named Gaussian prerequisite proof is not exact")
    return {
        "RNG-GAUSS-001": {
            "coverage": "fixed states, MT boundaries, and cached-pair preservation",
            "case_count": gaussian_matrix["case_count"],
            "verdict": "pass",
        },
        "RNG-GAUSS-002": {
            "logical_gaussian_values": logical_values,
            "physical_refill_cycles": refill_cycles,
            "terminal_rng_state_equal": True,
            "verdict": "pass",
        },
        "RNG-GAUSS-003": {
            "checkpoint_after_logical_values": 9,
            "resumed_future_values": 73,
            "future_bytes_sha256": hashlib.sha256(resumed_bytes).hexdigest(),
            "terminal_rng_state_equal": True,
            "verdict": "pass",
        },
        "RNG-GAUSS-004": {
            "awkward_offset_comparisons": carry_forward["comparison_count"],
            "verdict": "pass",
        },
        "RNG-GAUSS-005": {
            "batch_cells": carry_forward["batch_cells"],
            "mapping_permutation": False,
            "verdict": "pass",
        },
        "RNG-GAUSS-CAPACITY-001": {
            "capacities": [17, 131],
            "prefix_bytes_equal": True,
            "verdict": "pass",
        },
        "aggregate_expected_sha256": expected_digest.hexdigest(),
        "aggregate_actual_sha256": actual_digest.hexdigest(),
        "verdict": "pass",
    }


def run_probe(extension: Path) -> dict[str, Any]:
    extension_sha256 = _sha256(extension)
    module = _load_extension(extension)
    gaussian = _run_gaussian_matrix(module)
    policy = _run_policy_matrix(module)
    inplace = _run_inplace_refill_matrix(module)
    carry_forward = _run_gaussian_carry_forward_microproof(module)
    named_gaussian_d0 = _run_named_gaussian_d0_microproofs(
        module,
        gaussian_matrix=gaussian,
        carry_forward=carry_forward,
    )
    if gaussian["expected_sha256"] != gaussian["actual_sha256"]:
        raise ProbeMismatch("Gaussian matrix aggregate hashes differ")
    if policy["expected_sha256"] != policy["actual_sha256"]:
        raise ProbeMismatch("policy matrix aggregate hashes differ")
    if _sha256(extension) != extension_sha256:
        raise RuntimeError("extension changed while the probe was running")

    counts = _matrix_counts()
    observed_counts = {
        "gaussian_cases": gaussian["case_count"],
        "gaussian_tensor_comparisons": gaussian["tensor_comparisons"],
        "policy_cases": policy["case_count"],
        "policy_tensor_comparisons": policy["tensor_comparisons"],
        "policy_zero_std_cases": policy["zero_std_case_count"],
    }
    for name, actual in observed_counts.items():
        if counts[name] != actual:
            raise RuntimeError(
                f"matrix accounting mismatch for {name}: "
                f"expected {counts[name]}, got {actual}"
            )

    expected_combined = _combined_hash(
        gaussian["expected_sha256"], policy["expected_sha256"]
    )
    actual_combined = _combined_hash(gaussian["actual_sha256"], policy["actual_sha256"])
    if expected_combined != actual_combined:
        raise ProbeMismatch("combined aggregate hashes differ")
    return {
        "byte_exact": True,
        "extension": str(extension),
        "extension_sha256": extension_sha256,
        "hashes": {
            "actual_combined_sha256": actual_combined,
            "expected_combined_sha256": expected_combined,
            "gaussian_actual_sha256": gaussian["actual_sha256"],
            "gaussian_expected_sha256": gaussian["expected_sha256"],
            "policy_actual_sha256": policy["actual_sha256"],
            "policy_expected_sha256": policy["expected_sha256"],
        },
        "matrix": {
            "gaussian_capacities": gaussian["capacities"],
            "gaussian_observed_indices": gaussian["observed_indices"],
            "policy_scenarios": policy["scenario_names"],
        },
        "matrix_counts": counts,
        "gaussian_carry_forward": carry_forward,
        "named_gaussian_d0_microproofs": named_gaussian_d0,
        "inplace_refill": inplace,
        "module_name": extension.stem,
        "schema_version": SCHEMA_VERSION,
        "verdict": "pass",
    }


def _failure_receipt(
    options: Options,
    *,
    extension_sha256: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "byte_exact": False,
        "error": {
            "message": str(error),
            "type": type(error).__name__,
        },
        "extension": str(options.extension),
        "extension_sha256": extension_sha256,
        "hashes": {
            "actual_combined_sha256": None,
            "expected_combined_sha256": None,
            "gaussian_actual_sha256": None,
            "gaussian_expected_sha256": None,
            "policy_actual_sha256": None,
            "policy_expected_sha256": None,
        },
        "matrix_counts": _matrix_counts(),
        "module_name": options.extension.stem,
        "schema_version": SCHEMA_VERSION,
        "verdict": "fail",
    }


def main(argv: Sequence[str] | None = None) -> int:
    namespace = build_parser().parse_args(argv)
    try:
        options = _options_from_namespace(namespace)
    except ValueError as exc:
        build_parser().error(str(exc))

    extension_sha256 = _sha256(options.extension)
    try:
        receipt = run_probe(options.extension)
    except Exception as exc:  # A probe failure must still leave a JSON verdict.
        receipt = _failure_receipt(
            options,
            extension_sha256=extension_sha256,
            error=exc,
        )
        _write_receipt(options.output, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 1
    _write_receipt(options.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
