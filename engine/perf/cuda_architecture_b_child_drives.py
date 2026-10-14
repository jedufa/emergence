from __future__ import annotations

"""Exact fixed-MLP newborn-drive subtransaction for CUDA Architecture B.

The caller supplies the deterministic parent/child slot plan produced by the
identity and scalar-inheritance transactions. This module owns only newborn
drive initialization, birth-reset shock, and the associated Population NumPy
PCG64 transitions. It does not construct policy rows or integrate births into
the generation loop.
"""

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Final, Mapping

import torch
from torch.utils.cpp_extension import load_inline


ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROOT = ROOT / "engine" / "perf" / "native"
SOURCE_CUDA = NATIVE_ROOT / "cuda_architecture_b_child_drives.cu"
PCG64_HEADER = NATIVE_ROOT / "cuda_architecture_b_pcg64.cuh"

CHILD_DRIVE_STATUS: Final[dict[int, str]] = {
    0: "ok",
    1: "invalid_birth_plan",
    2: "invalid_parent_order",
    3: "child_identity_not_committed",
    4: "unsupported_gru",
    5: "invalid_child_scalar_state",
    6: "invalid_pcg64_state",
}

CHILD_DRIVE_CAPABILITY_RECEIPT: Final[dict[str, bool | str]] = {
    "schema_version": "eiel_cuda_architecture_b_child_drives_capabilities_v1",
    "decision": "ARCHB-CUDA-STANDALONE-CHILD-DRIVES",
    "single_cell_cuda_resident_state": True,
    "deterministic_parent_child_plan": True,
    "fixed_mlp_flat_patch_child_drives": True,
    "numpy_pcg64_standard_normal_float64_then_float32": "bit_exact",
    "population_birth_noise_transition": True,
    "population_exp_prior_noise_transition": True,
    "birth_reset_shock": True,
    "stable_mutable_tensor_pointers": True,
    "whole_batch_preflight_fail_closed": True,
    "python_precomputed_child_random_rows": False,
    "birth_identity": False,
    "scalar_inheritance": False,
    "policy_rows": False,
    "generation_integration": False,
    "controlled_birth_commit": False,
    "gru": False,
    "jungle": False,
    "wdl": False,
    "promotion_qualified": False,
}

POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_RECEIPT_SCHEMA_V1: Final[str] = (
    "eiel_option_d_population_newborn_drive_standard_normal_receipt_v1"
)
POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_OPERATOR_ID_V1: Final[str] = (
    "population.newborn_drive.standard_normal"
)
POPULATION_NEWBORN_DRIVE_RAW64_BLOCKER_V1: Final[str] = (
    "population_newborn_drive_raw_or_equivalent_continuation_not_bound"
)

_MODULE: Any | None = None


def load_cuda_child_drives_extension(
    *,
    build_directory: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    if not torch.cuda.is_available():
        raise RuntimeError("Architecture B child drives require an available CUDA device")
    if build_directory is not None:
        Path(build_directory).mkdir(parents=True, exist_ok=True)
    cuda_source = SOURCE_CUDA.read_text(encoding="utf-8")
    source_hash = sha256(
        cuda_source.encode("utf-8") + PCG64_HEADER.read_bytes()
    ).hexdigest()[:12]
    cpp_source = r"""
    #include <torch/extension.h>

    std::vector<torch::Tensor> eiel_cuda_architecture_b_fixed_mlp_child_drives_v1(
        torch::Tensor slot_to_agent_id,
        torch::Tensor alive_mask,
        torch::Tensor policy_use_gru,
        torch::Tensor parent_slots,
        torch::Tensor child_slots,
        torch::Tensor planned_count,
        torch::Tensor policy_energy_observation,
        torch::Tensor genome_drive_sp_hunger,
        torch::Tensor genome_drive_sp_pain,
        torch::Tensor genome_drive_sp_fatigue,
        torch::Tensor genome_drive_sp_fear,
        torch::Tensor genome_exp_prior_noise,
        torch::Tensor genome_drive_birth_noise,
        torch::Tensor drives,
        torch::Tensor birth_reset_shock,
        torch::Tensor pcg64_state_words,
        torch::Tensor pcg64_buffer_words,
        torch::Tensor mutation_np_semantic_draws,
        double max_energy
    );
    """
    _MODULE = load_inline(
        name=f"eiel_cuda_architecture_b_child_drives_{source_hash}",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["eiel_cuda_architecture_b_fixed_mlp_child_drives_v1"],
        build_directory=str(build_directory) if build_directory is not None else None,
        extra_include_paths=[str(NATIVE_ROOT)],
        extra_cflags=["-O2", "-ffp-contract=off"],
        extra_cuda_cflags=["-O2", "--fmad=false"],
        with_cuda=True,
        verbose=bool(verbose),
    )
    return _MODULE


def execute_fixed_mlp_child_drives_cuda(
    *,
    slot_to_agent_id: torch.Tensor,
    alive_mask: torch.Tensor,
    policy_use_gru: torch.Tensor,
    parent_slots: torch.Tensor,
    child_slots: torch.Tensor,
    policy_energy_observation: torch.Tensor,
    genome_drive_sp_hunger: torch.Tensor,
    genome_drive_sp_pain: torch.Tensor,
    genome_drive_sp_fatigue: torch.Tensor,
    genome_drive_sp_fear: torch.Tensor,
    genome_exp_prior_noise: torch.Tensor,
    genome_drive_birth_noise: torch.Tensor,
    drives: torch.Tensor,
    birth_reset_shock: torch.Tensor,
    pcg64_state_words: torch.Tensor,
    pcg64_buffer_words: torch.Tensor,
    mutation_np_semantic_draws: torch.Tensor,
    max_energy: float,
    build_directory: str | Path | None = None,
) -> dict[str, torch.Tensor]:
    """Initialize planned child drives and commit exact NumPy RNG transitions.

    All mutable tensors remain at their original addresses. Device-side plan,
    scalar, fixed-MLP, and PCG64 validation completes before any authoritative
    value is changed; a nonzero status therefore leaves the full transaction
    unchanged.
    """
    module = load_cuda_child_drives_extension(build_directory=build_directory)
    outputs = module.eiel_cuda_architecture_b_fixed_mlp_child_drives_v1(
        slot_to_agent_id,
        alive_mask,
        policy_use_gru,
        parent_slots,
        child_slots,
        torch.full(
            (1,),
            parent_slots.numel(),
            dtype=torch.int64,
            device=parent_slots.device,
        ),
        policy_energy_observation,
        genome_drive_sp_hunger,
        genome_drive_sp_pain,
        genome_drive_sp_fatigue,
        genome_drive_sp_fear,
        genome_exp_prior_noise,
        genome_drive_birth_noise,
        drives,
        birth_reset_shock,
        pcg64_state_words,
        pcg64_buffer_words,
        mutation_np_semantic_draws,
        float(max_energy),
    )
    names = (
        "drives",
        "birth_reset_shock",
        "pcg64_state_words",
        "pcg64_buffer_words",
        "mutation_np_semantic_draws",
        "committed_count",
        "population_birth_noise_rows_consumed",
        "population_exp_prior_noise_rows_consumed",
        "native_population_standard_normal_rows_generated",
        "semantic_draws_added",
        "raw64_draws_added",
        "status",
    )
    return dict(zip(names, outputs, strict=True))


def _tensor_digest_v1(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().cpu().contiguous()
    return {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "sha256": sha256(detached.numpy().tobytes()).hexdigest(),
    }


def _scalar_int_v1(tensor: torch.Tensor, *, name: str) -> int:
    if tensor.numel() != 1:
        raise ValueError(f"{name} must have exactly one scalar value")
    return int(tensor.detach().cpu().reshape(()).item())


def population_newborn_drive_standard_normal_receipt_v1(
    *,
    result: Mapping[str, torch.Tensor],
    pcg64_state_words_before: torch.Tensor,
    pcg64_buffer_words_before: torch.Tensor,
    pcg64_state_words_after: torch.Tensor,
    pcg64_buffer_words_after: torch.Tensor,
    mutation_np_semantic_draws_before: torch.Tensor,
    mutation_np_semantic_draws_after: torch.Tensor,
) -> dict[str, Any]:
    """Build the named Option-D receipt for newborn-drive PCG64 normal use.

    The CUDA wrapper exposes committed semantic normal counts, committed raw64
    PCG64 word counts, and final PCG64 state. Preflight lookahead consumes a
    private RNG copy and is intentionally excluded from the raw counter.
    """
    status = _scalar_int_v1(result["status"], name="status")
    semantic_before = _scalar_int_v1(
        mutation_np_semantic_draws_before,
        name="mutation_np_semantic_draws_before",
    )
    semantic_after = _scalar_int_v1(
        mutation_np_semantic_draws_after,
        name="mutation_np_semantic_draws_after",
    )
    semantic_delta = semantic_after - semantic_before
    semantic_draws_added = _scalar_int_v1(
        result["semantic_draws_added"],
        name="semantic_draws_added",
    )
    raw64_draws_added = (
        _scalar_int_v1(result["raw64_draws_added"], name="raw64_draws_added")
        if "raw64_draws_added" in result
        else None
    )
    if semantic_delta != semantic_draws_added:
        raise ValueError(
            "mutation_np_semantic_draws delta must equal semantic_draws_added"
        )
    raw_delta_available = raw64_draws_added is not None
    promotion_blockers = (
        []
        if raw_delta_available
        else [POPULATION_NEWBORN_DRIVE_RAW64_BLOCKER_V1]
    )
    receipt = {
        "schema_version": POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_RECEIPT_SCHEMA_V1,
        "operator_id": POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_OPERATOR_ID_V1,
        "stream_id": "numpy.population",
        "algorithm_id": "numpy.pcg64.xsl_rr_128_64",
        "source": {
            "wrapper": "engine/perf/cuda_architecture_b_child_drives.py",
            "cuda": "engine/perf/native/cuda_architecture_b_child_drives.cu",
            "persistent_helper": (
                "engine/perf/cuda_architecture_b_persistent_turnover_drives.cuh"
            ),
        },
        "status": status,
        "status_name": CHILD_DRIVE_STATUS.get(status, "unknown"),
        "committed_count": _scalar_int_v1(
            result["committed_count"],
            name="committed_count",
        ),
        "birth_noise_rows_consumed": _scalar_int_v1(
            result["population_birth_noise_rows_consumed"],
            name="population_birth_noise_rows_consumed",
        ),
        "exp_prior_noise_rows_consumed": _scalar_int_v1(
            result["population_exp_prior_noise_rows_consumed"],
            name="population_exp_prior_noise_rows_consumed",
        ),
        "standard_normal_rows_generated": _scalar_int_v1(
            result["native_population_standard_normal_rows_generated"],
            name="native_population_standard_normal_rows_generated",
        ),
        "semantic_draws_before": semantic_before,
        "semantic_draws_after": semantic_after,
        "semantic_draws_added": semantic_draws_added,
        "semantic_counter_unit": "numpy_standard_normal_float64_values",
        "raw64_draws_added": raw64_draws_added,
        "raw_counter_unit": "numpy_pcg64_raw64_words",
        "raw_pcg64_delta_available": raw_delta_available,
        "promotion_blockers": promotion_blockers,
        "pcg64_state_before": _tensor_digest_v1(pcg64_state_words_before),
        "pcg64_buffer_before": _tensor_digest_v1(pcg64_buffer_words_before),
        "pcg64_state_after": _tensor_digest_v1(pcg64_state_words_after),
        "pcg64_buffer_after": _tensor_digest_v1(pcg64_buffer_words_after),
        "pcg64_state_changed": not torch.equal(
            pcg64_state_words_before.detach().cpu(),
            pcg64_state_words_after.detach().cpu(),
        )
        or not torch.equal(
            pcg64_buffer_words_before.detach().cpu(),
            pcg64_buffer_words_after.detach().cpu(),
        ),
        "promotion_qualified": (
            status == 0 and raw_delta_available and not promotion_blockers
        ),
    }
    receipt["receipt_sha256"] = sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return receipt


__all__ = [
    "CHILD_DRIVE_CAPABILITY_RECEIPT",
    "CHILD_DRIVE_STATUS",
    "POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_OPERATOR_ID_V1",
    "POPULATION_NEWBORN_DRIVE_STANDARD_NORMAL_RECEIPT_SCHEMA_V1",
    "POPULATION_NEWBORN_DRIVE_RAW64_BLOCKER_V1",
    "execute_fixed_mlp_child_drives_cuda",
    "load_cuda_child_drives_extension",
    "population_newborn_drive_standard_normal_receipt_v1",
]
