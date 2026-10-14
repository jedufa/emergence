"""Exact generation-boundary tape for resident fixed-MLP child policy rows."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load_inline

from .cuda_architecture_b_torch_rng_state_pack_v1 import (
    TORCH_CPU_RNG_STATE_PACKED_BYTES_V1,
    pack_torch_cpu_rng_state_v1,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_CUDA = ROOT / "engine" / "perf" / "native" / "cuda_architecture_b_child_policy.cu"
TORCH_RNG_STATE_PACK_HEADER = (
    ROOT / "engine" / "perf" / "cuda_architecture_b_torch_rng_state_pack_v1.cuh"
)
TORCH_MT19937_REPLAY_HEADER = (
    ROOT / "engine" / "perf" / "cuda_architecture_b_torch_mt19937_replay_v1.cuh"
)
CHILD_POLICY_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "invalid_plan",
    2: "policy_tape_exhausted",
}
CHILD_POLICY_CAPABILITIES: Final[dict[str, bool]] = {
    "fixed_mlp_parent_row_copy": True,
    "hmem_zero_initialization": True,
    "dense_policy_weight_mutation": True,
    "sparse_policy_weight_mutation": True,
    "exact_torch_cpu_mutation_state": True,
    "generation_boundary_policy_tape": True,
    "resident_variable_birth_consumption": True,
    "dedicated_generation_turnover_crossover": True,
    "generation_turnover_crossover": False,
    "global_constructor_rng_state": False,
    "child_drives_or_pcg64": False,
    "birth_events": False,
    "gru": False,
    "promotion_qualified": False,
}
MAX_FIXED_MLP_POLICY_TAPE_EPISODES: Final[int] = 20_604
CROSSOVER_MASK_ENCODING_V1: Final[str] = "bitpack_lsb0_v1"

_MODULE: Any | None = None


def packed_crossover_row_bytes_v1(logical_width: int) -> int:
    width = int(logical_width)
    if width < 0:
        raise ValueError("logical_width must be nonnegative")
    return (width + 7) // 8


def pack_crossover_mask_lsb0_v1(mask: torch.Tensor) -> torch.Tensor:
    """Pack bool crossover rows with logical column 0 in byte bit 0."""

    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError("crossover mask must be a bool tensor")
    if mask.ndim not in (1, 2):
        raise ValueError("crossover mask must be rank 1 or rank 2")
    rows = mask.reshape(1, -1) if mask.ndim == 1 else mask
    logical_width = int(rows.shape[1])
    row_bytes = packed_crossover_row_bytes_v1(logical_width)
    packed = torch.zeros(
        (int(rows.shape[0]), row_bytes),
        dtype=torch.uint8,
        device=rows.device,
    )
    if logical_width == 0:
        return packed[0] if mask.ndim == 1 else packed
    bit_weights = (1 << torch.arange(8, dtype=torch.uint8, device=rows.device))
    padded = torch.zeros(
        (int(rows.shape[0]), row_bytes * 8),
        dtype=torch.uint8,
        device=rows.device,
    )
    padded[:, :logical_width] = rows.to(torch.uint8)
    packed.copy_(
        (padded.reshape(int(rows.shape[0]), row_bytes, 8) * bit_weights)
        .sum(dim=2)
        .to(torch.uint8)
    )
    return packed[0] if mask.ndim == 1 else packed


def unpack_crossover_mask_lsb0_v1(
    packed: torch.Tensor,
    *,
    logical_width: int,
) -> torch.Tensor:
    """Materialize a logical bool mask for tests and diagnostics only."""

    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8:
        raise ValueError("packed crossover mask must be a uint8 tensor")
    if packed.ndim not in (1, 2):
        raise ValueError("packed crossover mask must be rank 1 or rank 2")
    width = int(logical_width)
    if width < 0:
        raise ValueError("logical_width must be nonnegative")
    rows = packed.reshape(1, -1) if packed.ndim == 1 else packed
    expected_bytes = packed_crossover_row_bytes_v1(width)
    if int(rows.shape[1]) != expected_bytes:
        raise ValueError("packed crossover row width does not match logical_width")
    bit_weights = (1 << torch.arange(8, dtype=torch.uint8, device=rows.device))
    bits = (rows.unsqueeze(-1).bitwise_and(bit_weights) != 0).reshape(
        int(rows.shape[0]),
        expected_bytes * 8,
    )
    logical = bits[:, :width].contiguous()
    return logical[0] if packed.ndim == 1 else logical


def load_cuda_child_policy_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    cuda_source = SOURCE_CUDA.read_text(encoding="utf-8")
    suffix = sha256(
        cuda_source.encode("utf-8")
        + TORCH_RNG_STATE_PACK_HEADER.read_bytes()
        + TORCH_MT19937_REPLAY_HEADER.read_bytes()
    ).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_child_policy_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor alive_mask,
        torch::Tensor parent_slots,
        torch::Tensor child_slots,
        torch::Tensor planned_count,
        torch::Tensor mlp_pack,
        torch::Tensor hmem,
        torch::Tensor policy_use_gru,
        torch::Tensor rng_manager_mutation_torch_cpu_state,
        torch::Tensor policy_birth_delta_tape,
        torch::Tensor policy_birth_mutation_mask_tape,
        torch::Tensor policy_birth_post_rng_state_packed_tape,
        torch::Tensor policy_birth_semantic_draws_tape,
        torch::Tensor policy_birth_tape_cursor,
        torch::Tensor mutation_torch_cpu_semantic_draws
    );

    std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_turnover_child_policy_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor alive_mask,
        torch::Tensor parent_slots,
        torch::Tensor mate_slots,
        torch::Tensor child_slots,
        torch::Tensor planned_count,
        torch::Tensor mlp_pack,
        torch::Tensor hmem,
        torch::Tensor policy_use_gru,
        torch::Tensor rng_manager_mutation_torch_cpu_state,
        torch::Tensor policy_birth_delta_tape,
        torch::Tensor policy_birth_mutation_mask_tape,
        torch::Tensor policy_birth_crossover_mask_packed_tape,
        torch::Tensor policy_birth_post_rng_state_packed_tape,
        torch::Tensor policy_birth_semantic_draws_tape,
        torch::Tensor policy_birth_tape_cursor,
        torch::Tensor mutation_torch_cpu_semantic_draws
    );
    """
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    _MODULE = load_inline(
        name=f"eiel_cuda_architecture_b_child_policy_{suffix}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=[
            "eiel_cuda_architecture_b_fixed_mlp_child_policy_v1",
            "eiel_cuda_architecture_b_fixed_mlp_turnover_child_policy_v1",
        ],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        extra_include_paths=[str(ROOT / "engine" / "perf")],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def prepare_fixed_mlp_child_policy_tape(
    mutation_torch_cpu_state: torch.Tensor,
    *,
    episode_capacity: int,
    row_width: int,
    device: torch.device | str,
    policy_weight_mutation_enable: bool,
    policy_weight_mutation_mode: str = "dense",
    policy_weight_mutation_rate: float = 1.0,
    policy_weight_mutation_std: float = 0.0,
    semantic_draws: int = 0,
    include_crossover_mask: bool = False,
    compact_no_mutation_tape: bool = False,
    include_post_state_tape: bool = True,
) -> dict[str, torch.Tensor]:
    """Prepare exact Torch-CPU draws and post-state snapshots at a boundary."""
    episodes = int(episode_capacity)
    width = int(row_width)
    if episodes < 0 or episodes > MAX_FIXED_MLP_POLICY_TAPE_EPISODES:
        raise ValueError("episode_capacity is outside the bounded policy-tape range")
    if width < 0:
        raise ValueError("row_width must be nonnegative")
    if mutation_torch_cpu_state.device.type != "cpu":
        raise ValueError("mutation Torch state must be supplied from the CPU boundary")
    if mutation_torch_cpu_state.dtype != torch.uint8 or mutation_torch_cpu_state.ndim != 1:
        raise ValueError("mutation Torch state must be rank-1 uint8")
    mode = str(policy_weight_mutation_mode).lower()
    if mode in {"", "gaussian"}:
        mode = "dense"
    if mode not in {"dense", "sparse"}:
        raise ValueError("policy_weight_mutation_mode must be dense or sparse")
    rate = min(1.0, max(0.0, float(policy_weight_mutation_rate)))
    std = max(0.0, float(policy_weight_mutation_std))
    enabled = bool(policy_weight_mutation_enable) and std > 0.0
    if type(compact_no_mutation_tape) is not bool:
        raise ValueError("compact_no_mutation_tape must be an exact bool")
    if type(include_post_state_tape) is not bool:
        raise ValueError("include_post_state_tape must be an exact bool")
    if compact_no_mutation_tape and enabled:
        raise ValueError("compact_no_mutation_tape requires disabled policy mutation")
    if not include_post_state_tape and not compact_no_mutation_tape:
        raise ValueError("omitting post-state tape requires compact_no_mutation_tape")
    mutation_width = 1 if compact_no_mutation_tape else width

    generator = torch.Generator(device="cpu")
    generator.set_state(mutation_torch_cpu_state.contiguous())
    deltas: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    crossover_mask_packed_rows: list[torch.Tensor] = []
    packed_post_states: list[torch.Tensor] = []
    draws_by_episode: list[int] = []
    for _ in range(episodes):
        crossover_mask = torch.rand(
            (width,), generator=generator, dtype=torch.float32
        ).lt(0.5)
        draws = width
        if enabled:
            if mode == "sparse":
                mask = torch.rand(
                    (width,), generator=generator, dtype=torch.float32
                ).lt(rate)
                draws += width
            else:
                mask = torch.ones((width,), dtype=torch.bool)
            delta = torch.randn(
                (width,), generator=generator, dtype=torch.float32
            ) * std
            draws += width
            delta = torch.where(mask, delta, torch.zeros_like(delta))
        else:
            mask = torch.zeros((mutation_width,), dtype=torch.bool)
            delta = torch.zeros((mutation_width,), dtype=torch.float32)
        deltas.append(delta)
        masks.append(mask)
        crossover_mask_packed_rows.append(pack_crossover_mask_lsb0_v1(crossover_mask))
        if include_post_state_tape:
            packed_post_states.append(
                pack_torch_cpu_rng_state_v1(generator.get_state())
            )
        draws_by_episode.append(draws)

    delta_tape = (
        torch.stack(deltas)
        if deltas
        else torch.empty((0, mutation_width), dtype=torch.float32)
    )
    mask_tape = torch.stack(masks) if masks else torch.empty((0, mutation_width), dtype=torch.bool)
    crossover_mask_packed_tape = (
        torch.stack(crossover_mask_packed_rows)
        if crossover_mask_packed_rows
        else torch.empty(
            (0, packed_crossover_row_bytes_v1(width)),
            dtype=torch.uint8,
        )
    )
    packed_state_tape = (
        torch.stack(packed_post_states)
        if packed_post_states
        else torch.empty(
            (0, TORCH_CPU_RNG_STATE_PACKED_BYTES_V1),
            dtype=torch.uint8,
        )
    )
    if not include_post_state_tape:
        packed_state_tape = torch.empty((episodes, 0), dtype=torch.uint8)
    tape = {
        "rng_manager_mutation_torch_cpu_state": mutation_torch_cpu_state.clone().to(device),
        "policy_birth_delta_tape": delta_tape.to(device),
        "policy_birth_mutation_mask_tape": mask_tape.to(device),
        "policy_birth_post_rng_state_packed_tape": packed_state_tape.to(device),
        "policy_birth_semantic_draws_tape": torch.tensor(
            draws_by_episode, dtype=torch.int64, device=device
        ),
        "policy_birth_tape_cursor": torch.zeros((1,), dtype=torch.int64, device=device),
        "mutation_torch_cpu_semantic_draws": torch.tensor(
            [int(semantic_draws)], dtype=torch.int64, device=device
        ),
    }
    if include_crossover_mask:
        tape["policy_birth_crossover_mask_packed_tape"] = (
            crossover_mask_packed_tape.to(device)
        )
        tape["policy_birth_crossover_mask_logical_width"] = torch.tensor(
            [width],
            dtype=torch.int64,
            device=device,
        )
    return tape


def prepare_fixed_mlp_turnover_child_policy_tape(
    mutation_torch_cpu_state: torch.Tensor,
    *,
    episode_capacity: int,
    row_width: int,
    device: torch.device | str,
    policy_weight_mutation_enable: bool,
    policy_weight_mutation_mode: str = "dense",
    policy_weight_mutation_rate: float = 1.0,
    policy_weight_mutation_std: float = 0.0,
    semantic_draws: int = 0,
    compact_no_mutation_tape: bool = False,
    include_post_state_tape: bool = True,
) -> dict[str, torch.Tensor]:
    """Prepare the ordinary exact tape plus authoritative turnover crossover masks."""
    return prepare_fixed_mlp_child_policy_tape(
        mutation_torch_cpu_state,
        episode_capacity=episode_capacity,
        row_width=row_width,
        device=device,
        policy_weight_mutation_enable=policy_weight_mutation_enable,
        policy_weight_mutation_mode=policy_weight_mutation_mode,
        policy_weight_mutation_rate=policy_weight_mutation_rate,
        policy_weight_mutation_std=policy_weight_mutation_std,
        semantic_draws=semantic_draws,
        include_crossover_mask=True,
        compact_no_mutation_tape=compact_no_mutation_tape,
        include_post_state_tape=include_post_state_tape,
    )


def execute_fixed_mlp_child_policy_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    parent_slots: torch.Tensor,
    child_slots: torch.Tensor,
    mlp_pack: torch.Tensor,
    hmem: torch.Tensor,
    policy_use_gru: torch.Tensor,
    rng_manager_mutation_torch_cpu_state: torch.Tensor,
    policy_birth_delta_tape: torch.Tensor,
    policy_birth_mutation_mask_tape: torch.Tensor,
    policy_birth_post_rng_state_packed_tape: torch.Tensor,
    policy_birth_semantic_draws_tape: torch.Tensor,
    policy_birth_tape_cursor: torch.Tensor,
    mutation_torch_cpu_semantic_draws: torch.Tensor,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    module = load_cuda_child_policy_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_fixed_mlp_child_policy_v1(
        slot_to_agent_id,
        alive_mask,
        parent_slots,
        child_slots,
        torch.full(
            (1,),
            parent_slots.numel(),
            dtype=torch.int64,
            device=parent_slots.device,
        ),
        mlp_pack,
        hmem,
        policy_use_gru,
        rng_manager_mutation_torch_cpu_state,
        policy_birth_delta_tape,
        policy_birth_mutation_mask_tape,
        policy_birth_post_rng_state_packed_tape,
        policy_birth_semantic_draws_tape,
        policy_birth_tape_cursor,
        mutation_torch_cpu_semantic_draws,
    )
    names = (
        "committed_count",
        "semantic_draws_added",
        "mutation_events",
        "mutated_elements",
        "mutation_delta_total",
        "mutation_occurred_by_birth",
        "mutation_delta_by_birth",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


def execute_fixed_mlp_turnover_child_policy_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    parent_slots: torch.Tensor,
    mate_slots: torch.Tensor,
    child_slots: torch.Tensor,
    mlp_pack: torch.Tensor,
    hmem: torch.Tensor,
    policy_use_gru: torch.Tensor,
    rng_manager_mutation_torch_cpu_state: torch.Tensor,
    policy_birth_delta_tape: torch.Tensor,
    policy_birth_mutation_mask_tape: torch.Tensor,
    policy_birth_crossover_mask_packed_tape: torch.Tensor,
    policy_birth_post_rng_state_packed_tape: torch.Tensor,
    policy_birth_semantic_draws_tape: torch.Tensor,
    policy_birth_tape_cursor: torch.Tensor,
    mutation_torch_cpu_semantic_draws: torch.Tensor,
    policy_birth_crossover_mask_logical_width: torch.Tensor | None = None,
    policy_birth_crossover_mask_encoding: str | None = None,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Apply exact fixed-MLP turnover crossover and mutation rows on CUDA."""
    if policy_birth_crossover_mask_encoding is not None:
        if policy_birth_crossover_mask_encoding != CROSSOVER_MASK_ENCODING_V1:
            raise ValueError("unsupported policy crossover mask encoding")
    if policy_birth_crossover_mask_logical_width is not None:
        if (
            not isinstance(policy_birth_crossover_mask_logical_width, torch.Tensor)
            or policy_birth_crossover_mask_logical_width.numel() != 1
            or int(policy_birth_crossover_mask_logical_width.reshape(1)[0].item())
            != int(mlp_pack.shape[1])
        ):
            raise ValueError(
                "policy crossover mask logical width must match mlp_pack width"
            )
    module = load_cuda_child_policy_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_fixed_mlp_turnover_child_policy_v1(
        slot_to_agent_id,
        alive_mask,
        parent_slots,
        mate_slots,
        child_slots,
        torch.full(
            (1,),
            parent_slots.numel(),
            dtype=torch.int64,
            device=parent_slots.device,
        ),
        mlp_pack,
        hmem,
        policy_use_gru,
        rng_manager_mutation_torch_cpu_state,
        policy_birth_delta_tape,
        policy_birth_mutation_mask_tape,
        policy_birth_crossover_mask_packed_tape,
        policy_birth_post_rng_state_packed_tape,
        policy_birth_semantic_draws_tape,
        policy_birth_tape_cursor,
        mutation_torch_cpu_semantic_draws,
    )
    names = (
        "committed_count",
        "semantic_draws_added",
        "mutation_events",
        "mutated_elements",
        "mutation_delta_total",
        "mutation_occurred_by_birth",
        "mutation_delta_by_birth",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


__all__ = [
    "CHILD_POLICY_CAPABILITIES",
    "CHILD_POLICY_STATUS",
    "execute_fixed_mlp_child_policy_cuda",
    "execute_fixed_mlp_turnover_child_policy_cuda",
    "pack_crossover_mask_lsb0_v1",
    "packed_crossover_row_bytes_v1",
    "load_cuda_child_policy_extension",
    "prepare_fixed_mlp_child_policy_tape",
    "prepare_fixed_mlp_turnover_child_policy_tape",
    "unpack_crossover_mask_lsb0_v1",
]
