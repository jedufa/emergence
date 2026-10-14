"""Bounded default-off GPU-native RECOMB-72 inheritance primitive."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
from typing import Any, Final, Mapping, Sequence

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CUDA = ROOT / "engine/perf/native/recomb72_gpu_native.cu"
SCHEMA_VERSION: Final[str] = "recomb72_gpu_native_state_v1"
MODE_IDS: Final[dict[str, int]] = {
    "current_authoritative": 0,
    "parent_copy_per_locus": 1,
    "bounded_interpolation": 2,
    "whole_package_single_parent": 3,
}
PATTERN_IDS: Final[dict[str, int]] = {
    "single_extreme_scalar": 0,
    "resource_specialization_extremes": 1,
    "correlated_multi_trait_packages": 2,
}
AUDITED_LOCI: Final[tuple[str, ...]] = (
    "exploration_bias",
    "aggression_bias",
    "action_temperature",
    "resource_handling_axis",
    "drive_inertia",
    "pred_lr",
)
_MODULE: Any | None = None
_MODULE_ORIGIN: Path | None = None


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_prebuilt_extension(
    path: Path,
    *,
    expected_sha256: str | None,
) -> Any:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"RECOMB72_GPU_PREBUILT_MISSING path={resolved}")
    actual_sha256 = _file_sha256(resolved)
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or expected != actual_sha256:
            raise RuntimeError(
                "RECOMB72_GPU_PREBUILT_SHA256_MISMATCH "
                f"expected={expected!r} actual={actual_sha256}"
            )
    module_name = resolved.name.split(".", 1)[0]
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"RECOMB72_GPU_PREBUILT_IMPORT_UNAVAILABLE path={resolved}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_recomb72_gpu_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
    prebuilt_extension_path: str | Path | None = None,
    prebuilt_sha256: str | None = None,
    require_prebuilt: bool = False,
) -> Any:
    global _MODULE, _MODULE_ORIGIN
    requested_prebuilt = (
        None
        if prebuilt_extension_path is None
        else Path(prebuilt_extension_path).expanduser().resolve()
    )
    if require_prebuilt and requested_prebuilt is None:
        raise RuntimeError("RECOMB72_GPU_PREBUILT_REQUIRED")
    if requested_prebuilt is not None:
        if _MODULE is not None and _MODULE_ORIGIN == requested_prebuilt:
            if prebuilt_sha256 is not None:
                actual = _file_sha256(requested_prebuilt)
                if actual != str(prebuilt_sha256).strip().lower():
                    raise RuntimeError(
                        "RECOMB72_GPU_PREBUILT_SHA256_MISMATCH "
                        f"expected={prebuilt_sha256!r} actual={actual}"
                    )
            return _MODULE
        _MODULE = _load_prebuilt_extension(
            requested_prebuilt,
            expected_sha256=prebuilt_sha256,
        )
        _MODULE_ORIGIN = requested_prebuilt
        return _MODULE
    if _MODULE is not None and _MODULE_ORIGIN is None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("RECOMB72_GPU_REQUIRES_CUDA")
    source = SOURCE_CUDA.read_text(encoding="utf-8")
    suffix = sha256(source.encode("utf-8")).hexdigest()[:12]
    cpp = r"""
    #include <torch/extension.h>
    void recomb72_gpu_initialize_v1(
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor);
    void recomb72_gpu_advance_v1(
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        int64_t);
    void recomb72_gpu_summarize_v1(
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
        torch::Tensor);
    """
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load_inline(
        name=f"eiel_recomb72_gpu_native_{suffix}",
        cpp_sources=[cpp],
        cuda_sources=[source],
        functions=[
            "recomb72_gpu_initialize_v1",
            "recomb72_gpu_advance_v1",
            "recomb72_gpu_summarize_v1",
        ],
        build_directory=str(build_directory) if build_directory else None,
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=verbose,
    )
    _MODULE_ORIGIN = None
    return _MODULE


def _ids(values: Sequence[str], registry: Mapping[str, int], name: str) -> list[int]:
    unknown = [value for value in values if value not in registry]
    if unknown:
        raise ValueError(f"unsupported RECOMB-72 {name}: {unknown!r}")
    return [registry[value] for value in values]


@dataclass(slots=True)
class Recomb72GpuStateV1:
    loci: torch.Tensor
    alive: torch.Tensor
    agent_ids: torch.Tensor
    ancestry: torch.Tensor
    next_id: torch.Tensor
    generation: torch.Tensor
    seeds: torch.Tensor
    modes: torch.Tensor
    patterns: torch.Tensor
    status: torch.Tensor

    @property
    def batch_cells(self) -> int:
        return int(self.loci.shape[0])

    @property
    def capacity(self) -> int:
        return int(self.loci.shape[1])

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "locus_names": list(AUDITED_LOCI),
            "tensors": {
                name: getattr(self, name).detach().cpu()
                for name in (
                    "loci", "alive", "agent_ids", "ancestry", "next_id",
                    "generation", "seeds", "modes", "patterns", "status",
                )
            },
        }


def initialize_recomb72_gpu_state_v1(
    *,
    seeds: Sequence[int],
    modes: Sequence[str],
    patterns: Sequence[str],
    capacity: int = 256,
    device: str | torch.device = "cuda",
    build_directory: str | Path | None = None,
    prebuilt_extension_path: str | Path | None = None,
    prebuilt_sha256: str | None = None,
    require_prebuilt: bool = False,
) -> Recomb72GpuStateV1:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError("RECOMB72_GPU_REQUIRES_CUDA")
    if not seeds or len(seeds) != len(modes) or len(seeds) != len(patterns):
        raise ValueError("RECOMB-72 GPU arrays must have one value per cell")
    if capacity < 64:
        raise ValueError("RECOMB-72 GPU capacity must be at least 64")
    batch = len(seeds)
    state = Recomb72GpuStateV1(
        loci=torch.empty((batch, capacity, 6), dtype=torch.float32, device=resolved),
        alive=torch.empty((batch, capacity), dtype=torch.bool, device=resolved),
        agent_ids=torch.empty((batch, capacity), dtype=torch.int64, device=resolved),
        ancestry=torch.empty((batch, capacity, 64), dtype=torch.float32, device=resolved),
        next_id=torch.empty(batch, dtype=torch.int64, device=resolved),
        generation=torch.empty(batch, dtype=torch.int64, device=resolved),
        seeds=torch.tensor(seeds, dtype=torch.int64, device=resolved),
        modes=torch.tensor(_ids(modes, MODE_IDS, "mode"), dtype=torch.int32, device=resolved),
        patterns=torch.tensor(
            _ids(patterns, PATTERN_IDS, "pattern"), dtype=torch.int32, device=resolved
        ),
        status=torch.zeros(batch, dtype=torch.int32, device=resolved),
    )
    module = load_recomb72_gpu_extension(
        build_directory=build_directory,
        prebuilt_extension_path=prebuilt_extension_path,
        prebuilt_sha256=prebuilt_sha256,
        require_prebuilt=require_prebuilt,
    )
    module.recomb72_gpu_initialize_v1(
        state.loci, state.alive, state.agent_ids, state.ancestry,
        state.next_id, state.generation, state.seeds, state.modes,
        state.patterns, state.status,
    )
    torch.cuda.synchronize(resolved)
    if state.status.cpu().tolist() != [0] * batch:
        raise RuntimeError(f"RECOMB72_GPU_INITIALIZE_FAILED {state.status.cpu().tolist()!r}")
    return state


def restore_recomb72_gpu_state_v1(
    payload: Mapping[str, Any],
    *,
    device: str | torch.device = "cuda",
) -> Recomb72GpuStateV1:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError("RECOMB72_GPU_REQUIRES_CUDA")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("RECOMB-72 GPU checkpoint schema mismatch")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("RECOMB-72 GPU checkpoint lacks tensors")
    return Recomb72GpuStateV1(
        **{
            name: torch.as_tensor(tensors[name]).to(device=resolved).contiguous()
            for name in (
                "loci", "alive", "agent_ids", "ancestry", "next_id",
                "generation", "seeds", "modes", "patterns", "status",
            )
        }
    )


def advance_recomb72_gpu_state_v1(
    state: Recomb72GpuStateV1,
    *,
    generations: int,
    build_directory: str | Path | None = None,
    prebuilt_extension_path: str | Path | None = None,
    prebuilt_sha256: str | None = None,
    require_prebuilt: bool = False,
) -> dict[str, torch.Tensor]:
    if not state.loci.is_cuda:
        raise RuntimeError("RECOMB72_GPU_REQUIRES_CUDA")
    if not 1 <= generations <= 30:
        raise ValueError("RECOMB-72 GPU generations must be in [1, 30]")
    batch = state.batch_cells
    device = state.loci.device
    event_shape = (batch, generations, 32)
    outputs = {
        "parent_agent_ids": torch.empty(event_shape, dtype=torch.int64, device=device),
        "mate_agent_ids": torch.empty(event_shape, dtype=torch.int64, device=device),
        "child_agent_ids": torch.empty(event_shape, dtype=torch.int64, device=device),
        "parent_loci": torch.empty(
            (*event_shape, 6), dtype=torch.float32, device=device
        ),
        "mate_loci": torch.empty(
            (*event_shape, 6), dtype=torch.float32, device=device
        ),
        "child_loci": torch.empty(
            (*event_shape, 6), dtype=torch.float32, device=device
        ),
        "donor_by_locus": torch.empty(
            (*event_shape, 6), dtype=torch.int8, device=device
        ),
        "inheritance_draw_count": torch.empty(
            event_shape, dtype=torch.int32, device=device
        ),
        "variance_population": torch.zeros(
            (batch, generations, 6), dtype=torch.float32, device=device
        ),
        "fraction_near_either_extreme": torch.zeros(
            (batch, generations, 6), dtype=torch.float32, device=device
        ),
        "intact_package_fraction": torch.empty(
            (batch, generations), dtype=torch.float32, device=device
        ),
        "effective_founder_ancestry_count": torch.empty(
            (batch, generations), dtype=torch.float32, device=device
        ),
        "dominant_founder_ancestry_share": torch.empty(
            (batch, generations), dtype=torch.float32, device=device
        ),
        "mutation_events": torch.empty(event_shape, dtype=torch.int32, device=device),
    }
    module = load_recomb72_gpu_extension(
        build_directory=build_directory,
        prebuilt_extension_path=prebuilt_extension_path,
        prebuilt_sha256=prebuilt_sha256,
        require_prebuilt=require_prebuilt,
    )
    module.recomb72_gpu_advance_v1(
        state.loci, state.alive, state.agent_ids, state.ancestry,
        state.next_id, state.generation, state.seeds, state.modes,
        state.patterns, outputs["parent_agent_ids"], outputs["mate_agent_ids"],
        outputs["child_agent_ids"], outputs["parent_loci"],
        outputs["mate_loci"], outputs["child_loci"],
        outputs["donor_by_locus"], outputs["inheritance_draw_count"],
        outputs["variance_population"], outputs["fraction_near_either_extreme"],
        outputs["intact_package_fraction"],
        outputs["effective_founder_ancestry_count"],
        outputs["dominant_founder_ancestry_share"], outputs["mutation_events"],
        state.status, generations,
    )
    torch.cuda.synchronize(device)
    if state.status.cpu().tolist() != [0] * batch:
        raise RuntimeError(f"RECOMB72_GPU_ADVANCE_FAILED {state.status.cpu().tolist()!r}")
    return outputs


def summarize_recomb72_gpu_state_v1(
    state: Recomb72GpuStateV1,
    *,
    build_directory: str | Path | None = None,
    prebuilt_extension_path: str | Path | None = None,
    prebuilt_sha256: str | None = None,
    require_prebuilt: bool = False,
) -> dict[str, torch.Tensor]:
    """Project current resident state, including the founder G0 boundary."""

    if not state.loci.is_cuda:
        raise RuntimeError("RECOMB72_GPU_REQUIRES_CUDA")
    batch = state.batch_cells
    device = state.loci.device
    outputs = {
        "variance_population": torch.zeros(
            (batch, 6), dtype=torch.float32, device=device
        ),
        "fraction_near_either_extreme": torch.zeros(
            (batch, 6), dtype=torch.float32, device=device
        ),
        "intact_package_fraction": torch.empty(
            batch, dtype=torch.float32, device=device
        ),
        "effective_founder_ancestry_count": torch.empty(
            batch, dtype=torch.float32, device=device
        ),
        "dominant_founder_ancestry_share": torch.empty(
            batch, dtype=torch.float32, device=device
        ),
    }
    module = load_recomb72_gpu_extension(
        build_directory=build_directory,
        prebuilt_extension_path=prebuilt_extension_path,
        prebuilt_sha256=prebuilt_sha256,
        require_prebuilt=require_prebuilt,
    )
    module.recomb72_gpu_summarize_v1(
        state.loci,
        state.alive,
        state.ancestry,
        state.patterns,
        outputs["variance_population"],
        outputs["fraction_near_either_extreme"],
        outputs["intact_package_fraction"],
        outputs["effective_founder_ancestry_count"],
        outputs["dominant_founder_ancestry_share"],
    )
    torch.cuda.synchronize(device)
    return outputs


__all__ = [
    "AUDITED_LOCI", "MODE_IDS", "PATTERN_IDS", "Recomb72GpuStateV1",
    "advance_recomb72_gpu_state_v1", "initialize_recomb72_gpu_state_v1",
    "load_recomb72_gpu_extension", "restore_recomb72_gpu_state_v1",
    "summarize_recomb72_gpu_state_v1",
]
