from __future__ import annotations

"""CXXBACKBONE-3 C++/ATen policy GRU bridge."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "engine" / "perf" / "native" / "cxxbackbone3_policy_aten_extension.cpp"
GRAPH_NAMES = (
    "enc_linear",
    "z_in",
    "input_linear",
    "hidden_linear",
    "reset_preact",
    "update_preact",
    "candidate_input",
    "candidate_hidden",
    "reset_gate",
    "update_gate",
    "reset_hidden",
    "candidate_preact",
    "candidate",
    "interp_left",
    "interp_right",
    "hmem_after",
    "logits_base",
    "logits_after_policy",
    "logits_after_drives",
    "actions",
)
MLP_GRAPH_NAMES = (
    "h1_linear",
    "h1",
    "h2_linear",
    "h2",
    "logits_base",
    "logits_after_policy",
    "logits_after_drives",
    "actions",
)


_MODULE: Any | None = None


def load_policy_aten_extension(*, build_directory: str | Path | None = None, verbose: bool = False) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    source = SOURCE.read_text(encoding="utf-8")
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load_inline(
        name="cxxbackbone3_policy_aten_ext",
        cpp_sources=[source],
        functions=["cxxbackbone3_gru_policy_aten_v1", "cxxbackbone4_mlp_policy_aten_v1"],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2"],
        with_cuda=False,
        verbose=bool(verbose),
    )
    return _MODULE


def run_gru_policy_aten(
    *,
    x: np.ndarray,
    h_prev: np.ndarray,
    pack: np.ndarray,
    mem_dims: np.ndarray,
    aggr: np.ndarray,
    temp: np.ndarray,
    drives: np.ndarray,
    gain: np.ndarray,
    weights: dict[str, float],
    obs: int,
    hidden: int,
    actions: int,
    build_directory: str | Path | None = None,
) -> dict[str, np.ndarray]:
    mod = load_policy_aten_extension(build_directory=build_directory)
    outputs = mod.cxxbackbone3_gru_policy_aten_v1(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(h_prev, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(pack, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(mem_dims, dtype=np.int64)),
        torch.from_numpy(np.ascontiguousarray(aggr, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(temp, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(drives, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(gain, dtype=np.float32)),
        float(weights["hunger_gather"]),
        float(weights["pain_rest"]),
        float(weights["fatigue_rest"]),
        float(weights["pain_attack"]),
        float(weights["fear_attack"]),
        int(obs),
        int(hidden),
        int(actions),
    )
    return {
        name: tensor.detach().cpu().numpy().copy()
        for name, tensor in zip(GRAPH_NAMES, outputs, strict=True)
    }


def run_gru_policy_aten_torch(
    *,
    x: torch.Tensor,
    h_prev: torch.Tensor,
    pack: torch.Tensor,
    mem_dims: torch.Tensor,
    aggr: torch.Tensor,
    temp: torch.Tensor,
    drives: torch.Tensor,
    gain: torch.Tensor,
    weights: dict[str, float],
    obs: int,
    hidden: int,
    actions: int,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    mod = load_policy_aten_extension(build_directory=build_directory)
    outputs = mod.cxxbackbone3_gru_policy_aten_v1(
        x.contiguous(),
        h_prev.contiguous(),
        pack.contiguous(),
        mem_dims.contiguous(),
        aggr.contiguous(),
        temp.contiguous(),
        drives.contiguous(),
        gain.contiguous(),
        float(weights["hunger_gather"]),
        float(weights["pain_rest"]),
        float(weights["fatigue_rest"]),
        float(weights["pain_attack"]),
        float(weights["fear_attack"]),
        int(obs),
        int(hidden),
        int(actions),
    )
    return {name: tensor for name, tensor in zip(GRAPH_NAMES, outputs, strict=True)}


def run_mlp_policy_aten_torch(
    *,
    x: torch.Tensor,
    pack: torch.Tensor,
    aggr: torch.Tensor,
    temp: torch.Tensor,
    drives: torch.Tensor,
    gain: torch.Tensor,
    weights: dict[str, float],
    obs: int,
    hidden: int,
    actions: int,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    mod = load_policy_aten_extension(build_directory=build_directory)
    outputs = mod.cxxbackbone4_mlp_policy_aten_v1(
        x.contiguous(),
        pack.contiguous(),
        aggr.contiguous(),
        temp.contiguous(),
        drives.contiguous(),
        gain.contiguous(),
        float(weights["hunger_gather"]),
        float(weights["pain_rest"]),
        float(weights["fatigue_rest"]),
        float(weights["pain_attack"]),
        float(weights["fear_attack"]),
        int(obs),
        int(hidden),
        int(actions),
    )
    return {name: tensor for name, tensor in zip(MLP_GRAPH_NAMES, outputs, strict=True)}


__all__ = [
    "GRAPH_NAMES",
    "MLP_GRAPH_NAMES",
    "load_policy_aten_extension",
    "run_gru_policy_aten",
    "run_gru_policy_aten_torch",
    "run_mlp_policy_aten_torch",
]
