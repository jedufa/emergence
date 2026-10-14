#!/usr/bin/env python3
"""M5 receipt probe for the Option D M4 policy-refill candidate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Final


ROOT: Final[Path] = Path(__file__).resolve().parents[2]
ENGINE: Final[Path] = ROOT / "engine"
for _path in (ROOT, ENGINE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


MODULE_NAME: Final[str] = "eiel_cuda_architecture_b_flattened_controller_v1"
ENTRYPOINT: Final[str] = "eiel_cuda_architecture_b_refill_fixed_mlp_policy_window_v1"
SCHEMA_VERSION: Final[str] = "eiel_option_d_m5_policy_refill_receipt_probe_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    import torch

    payload = (
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _load_extension(extension: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(MODULE_NAME, extension)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import extension {extension}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, ENTRYPOINT, None)):
        raise RuntimeError(f"extension is missing entrypoint {ENTRYPOINT}")
    return module


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    import torch

    if not torch.equal(actual.detach().cpu(), expected.detach().cpu()):
        raise RuntimeError(f"{label} mismatch")


def _compact_disabled_case(module: ModuleType) -> dict[str, Any]:
    import torch

    from engine.perf.cuda_architecture_b_child_policy import (
        prepare_fixed_mlp_turnover_child_policy_tape,
    )

    seed = 701
    rows = 5
    width = 13
    row_bytes = (width + 7) // 8
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    source = generator.get_state().clone().contiguous()
    expected = prepare_fixed_mlp_turnover_child_policy_tape(
        source,
        episode_capacity=rows,
        row_width=width,
        device="cpu",
        policy_weight_mutation_enable=False,
        policy_weight_mutation_mode="dense",
        policy_weight_mutation_rate=0.0,
        policy_weight_mutation_std=0.0,
    )
    tensors = {
        "rng_manager_mutation_torch_cpu_state": source.view(1, -1).clone().cuda(),
        "policy_birth_delta_tape": torch.full(
            (1, rows, 1), 123.0, dtype=torch.float32, device="cuda"
        ),
        "policy_birth_mutation_mask_tape": torch.full(
            (1, rows, 1), True, dtype=torch.bool, device="cuda"
        ),
        "policy_birth_crossover_mask_packed_tape": torch.full(
            (1, rows, row_bytes), 255, dtype=torch.uint8, device="cuda"
        ),
        "policy_birth_post_rng_state_packed_tape": torch.empty(
            (1, rows, 0), dtype=torch.uint8, device="cuda"
        ),
        "policy_birth_semantic_draws_tape": torch.full(
            (1, rows), -1, dtype=torch.int64, device="cuda"
        ),
        "policy_birth_tape_cursor": torch.tensor([4], dtype=torch.int64, device="cuda"),
        "mutation_torch_cpu_semantic_draws": torch.tensor(
            [101], dtype=torch.int64, device="cuda"
        ),
    }
    window_base = torch.tensor([1], dtype=torch.int64, device="cuda")
    source_before = _tensor_sha256(tensors["rng_manager_mutation_torch_cpu_state"])
    pointer_before = {
        name: tensor.data_ptr()
        for name, tensor in tensors.items()
        if name != "rng_manager_mutation_torch_cpu_state"
    }
    pointer_before["policy_birth_window_base"] = window_base.data_ptr()
    receipt = getattr(module, ENTRYPOINT)(
        tensors["rng_manager_mutation_torch_cpu_state"],
        tensors["policy_birth_delta_tape"],
        tensors["policy_birth_mutation_mask_tape"],
        tensors["policy_birth_crossover_mask_packed_tape"],
        tensors["policy_birth_post_rng_state_packed_tape"],
        tensors["policy_birth_semantic_draws_tape"],
        tensors["policy_birth_tape_cursor"],
        tensors["mutation_torch_cpu_semantic_draws"],
        window_base,
        width,
        False,
        "dense",
        0.0,
        0.0,
    )
    torch.cuda.synchronize()
    expected_delta = torch.zeros((1, rows, 1), dtype=torch.float32)
    expected_mask = torch.zeros((1, rows, 1), dtype=torch.bool)
    _require_equal("compact.delta", tensors["policy_birth_delta_tape"], expected_delta)
    _require_equal(
        "compact.mutation_mask",
        tensors["policy_birth_mutation_mask_tape"],
        expected_mask,
    )
    _require_equal(
        "compact.crossover",
        tensors["policy_birth_crossover_mask_packed_tape"],
        expected["policy_birth_crossover_mask_packed_tape"].view(1, rows, row_bytes),
    )
    _require_equal(
        "compact.semantic_draws",
        tensors["policy_birth_semantic_draws_tape"],
        expected["policy_birth_semantic_draws_tape"].view(1, rows),
    )
    pointer_after = {
        name: tensor.data_ptr()
        for name, tensor in tensors.items()
        if name != "rng_manager_mutation_torch_cpu_state"
    }
    pointer_after["policy_birth_window_base"] = window_base.data_ptr()
    return {
        "case": "compact_disabled_mutation_dense",
        "byte_exact": True,
        "source_state_preserved": source_before
        == _tensor_sha256(tensors["rng_manager_mutation_torch_cpu_state"]),
        "pointer_identity_preserved": pointer_before == pointer_after,
        "output_hashes": {
            "crossover_sha256": _tensor_sha256(
                tensors["policy_birth_crossover_mask_packed_tape"]
            ),
            "semantic_draws_sha256": _tensor_sha256(
                tensors["policy_birth_semantic_draws_tape"]
            ),
        },
        "receipt": dict(receipt),
    }


def _fallback_sparse_case(module: ModuleType) -> dict[str, Any]:
    import torch

    from engine.perf.cuda_architecture_b_child_policy import (
        prepare_fixed_mlp_turnover_child_policy_tape,
    )

    seed = 703
    rows = 5
    width = 13
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    source = generator.get_state().clone().contiguous()
    expected = prepare_fixed_mlp_turnover_child_policy_tape(
        source,
        episode_capacity=rows,
        row_width=width,
        device="cpu",
        policy_weight_mutation_enable=True,
        policy_weight_mutation_mode="sparse",
        policy_weight_mutation_rate=0.35,
        policy_weight_mutation_std=0.125,
    )
    tensors = {
        "rng_manager_mutation_torch_cpu_state": source.view(1, -1).clone().cuda(),
        "policy_birth_delta_tape": torch.empty(
            (1, rows, width), dtype=torch.float32, device="cuda"
        ),
        "policy_birth_mutation_mask_tape": torch.empty(
            (1, rows, width), dtype=torch.bool, device="cuda"
        ),
        "policy_birth_crossover_mask_packed_tape": torch.empty(
            (1, rows, (width + 7) // 8), dtype=torch.uint8, device="cuda"
        ),
        "policy_birth_post_rng_state_packed_tape": torch.empty(
            (1, rows, 2560), dtype=torch.uint8, device="cuda"
        ),
        "policy_birth_semantic_draws_tape": torch.empty(
            (1, rows), dtype=torch.int64, device="cuda"
        ),
        "policy_birth_tape_cursor": torch.tensor([4], dtype=torch.int64, device="cuda"),
        "mutation_torch_cpu_semantic_draws": torch.tensor(
            [101], dtype=torch.int64, device="cuda"
        ),
    }
    window_base = torch.tensor([1], dtype=torch.int64, device="cuda")
    receipt = getattr(module, ENTRYPOINT)(
        tensors["rng_manager_mutation_torch_cpu_state"],
        tensors["policy_birth_delta_tape"],
        tensors["policy_birth_mutation_mask_tape"],
        tensors["policy_birth_crossover_mask_packed_tape"],
        tensors["policy_birth_post_rng_state_packed_tape"],
        tensors["policy_birth_semantic_draws_tape"],
        tensors["policy_birth_tape_cursor"],
        tensors["mutation_torch_cpu_semantic_draws"],
        window_base,
        width,
        True,
        "sparse",
        0.35,
        0.125,
    )
    torch.cuda.synchronize()
    for name in (
        "policy_birth_delta_tape",
        "policy_birth_mutation_mask_tape",
        "policy_birth_crossover_mask_packed_tape",
        "policy_birth_post_rng_state_packed_tape",
        "policy_birth_semantic_draws_tape",
    ):
        _require_equal(name, tensors[name], expected[name].view_as(tensors[name]))
    return {
        "case": "fallback_sparse_mutation",
        "byte_exact": True,
        "receipt": dict(receipt),
    }


def run_probe(extension: Path) -> dict[str, Any]:
    if not extension.is_file():
        raise FileNotFoundError(extension)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("M5 policy refill receipt probe requires CUDA")
    extension_sha256 = _sha256_file(extension)
    module = _load_extension(extension)
    compact = _compact_disabled_case(module)
    fallback = _fallback_sparse_case(module)
    if _sha256_file(extension) != extension_sha256:
        raise RuntimeError("extension changed while the probe ran")
    return {
        "schema_version": SCHEMA_VERSION,
        "extension": str(extension),
        "extension_sha256": extension_sha256,
        "module_name": MODULE_NAME,
        "entrypoint": ENTRYPOINT,
        "cuda_available": True,
        "cases": [compact, fallback],
        "verdict": "pass",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        receipt = run_probe(args.extension.expanduser().resolve())
    except Exception as exc:
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "extension": str(args.extension),
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "verdict": "fail",
        }
        _write_json(args.output, receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 1
    _write_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
