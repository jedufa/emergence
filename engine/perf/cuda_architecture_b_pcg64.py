from __future__ import annotations

"""Exact NumPy-PCG64 float32 draws kept resident on CUDA."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "engine" / "perf" / "native"
CUDA_SOURCE = NATIVE_ROOT / "cuda_architecture_b_pcg64.cu"
_MODULE: Any | None = None
_WORD_MASK = (1 << 64) - 1

CUDA_PCG64_CAPABILITIES = {
    "schema_version": "eiel_cuda_architecture_b_pcg64_capabilities_v1",
    "numpy_pcg64_float32_draws": "bit_exact",
    "serialized_state_continuation": True,
    "draw_order": "sequential_cpu_oracle_order",
    "device_resident_state": True,
    "runtime_jit": True,
    "promotion_qualified": False,
}


@dataclass(frozen=True)
class CudaPcg64DrawResult:
    values: torch.Tensor
    state_words: torch.Tensor
    buffer_words: torch.Tensor

    def state_dict(self) -> dict[str, object]:
        state_words = [int(word) for word in self.state_words.detach().cpu().tolist()]
        buffer_words = [int(word) for word in self.buffer_words.detach().cpu().tolist()]
        return {
            "bit_generator": "PCG64",
            "state": {
                "state": state_words[0] | (state_words[1] << 64),
                "inc": state_words[2] | (state_words[3] << 64),
            },
            "has_uint32": buffer_words[0],
            "uinteger": buffer_words[1],
        }


def load_cuda_pcg64_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B PCG64 requires an available CUDA device")
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    cpp_source = r"""
    #include <torch/extension.h>
    std::vector<torch::Tensor> eiel_cuda_pcg64_uniform_float32_v1(
        torch::Tensor state_words,
        torch::Tensor buffer_words,
        const int64_t count
    );
    """
    _MODULE = load_inline(
        name="eiel_cuda_architecture_b_pcg64_ext",
        cpp_sources=[cpp_source],
        cuda_sources=[CUDA_SOURCE.read_text(encoding="utf-8")],
        functions=["eiel_cuda_pcg64_uniform_float32_v1"],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(NATIVE_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def pcg64_state_tensors(
    state: Mapping[str, object],
    *,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    if state.get("bit_generator") != "PCG64":
        raise ValueError("only NumPy PCG64 state is supported")
    inner = state.get("state")
    if not isinstance(inner, Mapping):
        raise ValueError("PCG64 state must contain the inner state mapping")
    raw_state = int(inner["state"])
    raw_inc = int(inner["inc"])
    if not (0 <= raw_state < (1 << 128) and 0 <= raw_inc < (1 << 128)):
        raise ValueError("PCG64 state and increment must be unsigned 128-bit integers")
    state_words = torch.tensor(
        [
            raw_state & _WORD_MASK,
            raw_state >> 64,
            raw_inc & _WORD_MASK,
            raw_inc >> 64,
        ],
        dtype=torch.uint64,
        device=device,
    )
    buffer_words = torch.tensor(
        [int(state.get("has_uint32", 0)), int(state.get("uinteger", 0))],
        dtype=torch.int64,
        device=device,
    )
    return state_words, buffer_words


def pcg64_uniform_float32_cuda(
    *,
    state_words: torch.Tensor,
    buffer_words: torch.Tensor,
    count: int,
    build_directory: str | Path | None = None,
) -> CudaPcg64DrawResult:
    module = load_cuda_pcg64_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_pcg64_uniform_float32_v1(
        state_words,
        buffer_words,
        int(count),
    )
    return CudaPcg64DrawResult(*outputs)


__all__ = [
    "CUDA_PCG64_CAPABILITIES",
    "CudaPcg64DrawResult",
    "load_cuda_pcg64_extension",
    "pcg64_state_tensors",
    "pcg64_uniform_float32_cuda",
]
