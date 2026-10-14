"""Device-scaled B/K planning and exact committed autotune for direct sparse V2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Final, Sequence

from .direct_sparse_runtime import DIRECT_SPARSE_ADMISSION_MARGIN_BYTES_V3
from .direct_sparse_sustained_benchmark import (
    _canonical_bytes,
    _publish_immutable_bytes,
    run_direct_sparse_sustained_benchmark,
)
from .layout import CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3, state_layout_v2
from .native_loader import cuda_device_identity_v2, file_sha256

AUTOTUNE_SCHEMA_V3: Final[str] = "eiel_cuda_full_evolution_direct_sparse_autotune_v3"
AUTOTUNE_PLAN_SCHEMA_V3: Final[str] = (
    "eiel_cuda_full_evolution_direct_sparse_autotune_plan_v3"
)
CLAIM_CLASS: Final[str] = "TECHNICAL_ONLY"
_B_ALIGNMENT: Final[int] = 64
_BLOCK_WAVES_PER_SM: Final[tuple[int, ...]] = (
    24,
    28,
    32,
    36,
    38,
    40,
    42,
    44,
    48,
    56,
)


class DirectSparseAutotuneError(RuntimeError):
    """Raised when device-scaled selection cannot remain exact."""


@dataclass(frozen=True, slots=True)
class DirectSparseAutotuneCandidateV3:
    batch_size: int
    temporal_k: int
    initial_capacity_rows_per_world: int
    estimated_peak_bytes: int
    blocks_per_sm_wave_target: int

    def to_dict(self) -> dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "temporal_k": self.temporal_k,
            "initial_capacity_rows_per_world": (self.initial_capacity_rows_per_world),
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "blocks_per_sm_wave_target": self.blocks_per_sm_wave_target,
        }


def _capacity_for_k(temporal_k: int) -> int:
    if type(temporal_k) is not int or not 1 <= temporal_k <= 64:
        raise DirectSparseAutotuneError("temporal K must be in [1,64]")
    requested = max(64, temporal_k * 8)
    return 1 << (requested - 1).bit_length()


def _estimated_peak_bytes(batch_size: int, capacity_rows: int) -> int:
    state_bytes = batch_size * state_layout_v2().state_bytes_per_world
    staging_bytes = batch_size * capacity_rows * CUDA_NATIVE_SPARSE_EVENT_ROW_BYTES_V3
    control_bytes = batch_size * 3 * 4 + (batch_size + 1) * 8
    return (
        2 * state_bytes
        + 2 * staging_bytes
        + control_bytes
        + DIRECT_SPARSE_ADMISSION_MARGIN_BYTES_V3
    )


def plan_direct_sparse_autotune_v3(
    *,
    multiprocessor_count: int,
    current_free_bytes: int,
    total_memory_bytes: int,
    temporal_k_candidates: Sequence[int] = (64,),
    batch_size_candidates: Sequence[int] | None = None,
    exact_tail_batch_size: int | None = None,
) -> dict[str, Any]:
    if (
        type(multiprocessor_count) is not int
        or multiprocessor_count < 1
        or type(current_free_bytes) is not int
        or current_free_bytes < 1
        or type(total_memory_bytes) is not int
        or total_memory_bytes < current_free_bytes
    ):
        raise DirectSparseAutotuneError("device envelope is invalid")
    temporal = tuple(dict.fromkeys(temporal_k_candidates))
    if not temporal:
        raise DirectSparseAutotuneError("at least one temporal K is required")
    if batch_size_candidates is not None and exact_tail_batch_size is not None:
        raise DirectSparseAutotuneError(
            "aligned batch candidates and an exact tail batch are mutually exclusive"
        )
    explicit_batches: tuple[int, ...] | None = None
    exact_tail = exact_tail_batch_size is not None
    if exact_tail:
        if (
            type(exact_tail_batch_size) is not int
            or not 1 <= exact_tail_batch_size < _B_ALIGNMENT
        ):
            raise DirectSparseAutotuneError(
                "exact tail batch size must be an integer in [1,63]"
            )
        explicit_batches = (exact_tail_batch_size,)
    elif batch_size_candidates is not None:
        explicit_batches = tuple(dict.fromkeys(batch_size_candidates))
        if (
            not explicit_batches
            or any(
                type(batch_size) is not int
                or batch_size < _B_ALIGNMENT
                or batch_size % _B_ALIGNMENT
                for batch_size in explicit_batches
            )
        ):
            raise DirectSparseAutotuneError(
                "explicit batch candidates must be positive multiples of 64"
            )
    candidates: list[DirectSparseAutotuneCandidateV3] = []
    for temporal_k in temporal:
        capacity = _capacity_for_k(temporal_k)
        candidate_batches = (
            tuple(
                (
                    max(
                        _B_ALIGNMENT,
                        (
                            (multiprocessor_count * waves + _B_ALIGNMENT // 2)
                            // _B_ALIGNMENT
                        )
                        * _B_ALIGNMENT,
                    ),
                    waves,
                )
                for waves in _BLOCK_WAVES_PER_SM
            )
            if explicit_batches is None
            else tuple(
                (
                    batch_size,
                    (batch_size + multiprocessor_count - 1)
                    // multiprocessor_count,
                )
                for batch_size in explicit_batches
            )
        )
        for batch_size, waves in candidate_batches:
            estimated = _estimated_peak_bytes(batch_size, capacity)
            if estimated <= current_free_bytes:
                candidates.append(
                    DirectSparseAutotuneCandidateV3(
                        batch_size=batch_size,
                        temporal_k=temporal_k,
                        initial_capacity_rows_per_world=capacity,
                        estimated_peak_bytes=estimated,
                        blocks_per_sm_wave_target=waves,
                    )
                )
    unique = {
        (candidate.batch_size, candidate.temporal_k): candidate
        for candidate in candidates
    }
    ordered = sorted(unique.values(), key=lambda row: (row.temporal_k, row.batch_size))
    if not ordered:
        raise DirectSparseAutotuneError(
            "no direct sparse candidate fits current free VRAM"
        )
    body = {
        "schema_version": AUTOTUNE_PLAN_SCHEMA_V3,
        "claim_class": CLAIM_CLASS,
        "device_envelope": {
            "multiprocessor_count": multiprocessor_count,
            "current_free_bytes": current_free_bytes,
            "total_memory_bytes": total_memory_bytes,
        },
        "scaling_law": {
            "batch_alignment": 1 if exact_tail else _B_ALIGNMENT,
            "block_waves_per_sm": (
                [
                    (exact_tail_batch_size + multiprocessor_count - 1)
                    // multiprocessor_count
                ]
                if exact_tail
                else list(_BLOCK_WAVES_PER_SM)
            ),
            "batch_source": (
                "explicit_exact_tail_candidate"
                if exact_tail
                else (
                    "explicit_aligned_batch_candidates"
                    if explicit_batches is not None
                    else "multiprocessor_count_times_wave_then_align"
                )
            ),
            "vram_gate": (
                "state_plus_full_state_rollback_plus_direct_staging_plus_"
                "simultaneous_exact_pack_plus_control_plus_margin"
            ),
            "horizon_dependent_resident_bytes": 0,
        },
        "temporal_k_candidates": list(temporal),
        "candidates": [candidate.to_dict() for candidate in ordered],
    }
    return {**body, "plan_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest()}


def run_direct_sparse_autotune_v3(
    *,
    artifact_path: Path,
    output_directory: Path,
    temporal_k_candidates: Sequence[int],
    batch_size_candidates: Sequence[int] | None = None,
    exact_tail_batch_size: int | None = None,
    probe_samples: int,
    probe_warmups: int,
    seed: int,
) -> dict[str, Any]:
    if output_directory.exists():
        raise DirectSparseAutotuneError("autotune output directory already exists")
    output_directory.mkdir(parents=True, mode=0o700)
    import torch

    device_index = torch.cuda.current_device()
    identity = cuda_device_identity_v2(torch, device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    plan = plan_direct_sparse_autotune_v3(
        multiprocessor_count=int(identity["multiprocessor_count"]),
        current_free_bytes=int(free_bytes),
        total_memory_bytes=int(total_bytes),
        temporal_k_candidates=temporal_k_candidates,
        batch_size_candidates=batch_size_candidates,
        exact_tail_batch_size=exact_tail_batch_size,
    )
    rows: list[dict[str, Any]] = []
    for ordinal, candidate in enumerate(plan["candidates"]):
        candidate_root = output_directory / (
            f"candidate-{ordinal:02d}-b{candidate['batch_size']}-"
            f"k{candidate['temporal_k']}"
        )
        result = run_direct_sparse_sustained_benchmark(
            artifact_path=artifact_path,
            output_directory=candidate_root,
            batch_size=int(candidate["batch_size"]),
            temporal_k=int(candidate["temporal_k"]),
            samples=probe_samples,
            warmups=probe_warmups,
            seed=seed,
            initial_capacity_rows=int(candidate["initial_capacity_rows_per_world"]),
        )
        rows.append(
            {
                "candidate": dict(candidate),
                "status": result["status"],
                "minimum_committed_world_ticks_per_second": result[
                    "minimum_committed_world_ticks_per_second"
                ],
                "p10_committed_world_ticks_per_second": result[
                    "p10_committed_world_ticks_per_second"
                ],
                "result_path": result["result_path"],
                "result_body_sha256": result["result_body_sha256"],
            }
        )
    selected = max(
        rows,
        key=lambda row: (
            row["p10_committed_world_ticks_per_second"],
            row["minimum_committed_world_ticks_per_second"],
        ),
    )
    body = {
        "schema_version": AUTOTUNE_SCHEMA_V3,
        "claim_class": CLAIM_CLASS,
        "selection_statistic": "p10_then_minimum_committed_world_ticks_per_second",
        "artifact": {
            "path": str(artifact_path.resolve()),
            "sha256": file_sha256(artifact_path.resolve()),
        },
        "device_identity": identity,
        "plan": plan,
        "probe_samples": probe_samples,
        "probe_warmups": probe_warmups,
        "measurements": rows,
        "selected": selected,
        "autotune_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
    }
    result = {
        **body,
        "result_sha256": hashlib.sha256(_canonical_bytes(body)).hexdigest(),
    }
    payload = json.dumps(result, sort_keys=True, indent=2).encode("ascii") + b"\n"
    receipt = _publish_immutable_bytes(
        payload,
        output=output_directory / "autotune-result.json",
        temporary_name=".autotune-result.tmp",
    )
    return {**result, "result_path": receipt["path"]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--temporal-k", default="64")
    batch = parser.add_mutually_exclusive_group()
    batch.add_argument(
        "--batch-size",
        help="comma-separated aligned physical batch candidates",
    )
    batch.add_argument(
        "--exact-tail-batch-size",
        type=int,
        help=(
            "one exact technical-only remainder batch in [1,63]; "
            "does not relax the aligned candidate lattice"
        ),
    )
    parser.add_argument("--probe-samples", type=int, default=1)
    parser.add_argument("--probe-warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    temporal = tuple(int(value) for value in args.temporal_k.split(","))
    batches = (
        None
        if args.batch_size is None
        else tuple(int(value) for value in args.batch_size.split(","))
    )
    if args.plan_only:
        import torch

        identity = cuda_device_identity_v2(torch, torch.cuda.current_device())
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        result = plan_direct_sparse_autotune_v3(
            multiprocessor_count=int(identity["multiprocessor_count"]),
            current_free_bytes=int(free_bytes),
            total_memory_bytes=int(total_bytes),
            temporal_k_candidates=temporal,
            batch_size_candidates=batches,
            exact_tail_batch_size=args.exact_tail_batch_size,
        )
    else:
        result = run_direct_sparse_autotune_v3(
            artifact_path=args.artifact,
            output_directory=args.output,
            temporal_k_candidates=temporal,
            batch_size_candidates=batches,
            exact_tail_batch_size=args.exact_tail_batch_size,
            probe_samples=args.probe_samples,
            probe_warmups=args.probe_warmups,
            seed=args.seed,
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
