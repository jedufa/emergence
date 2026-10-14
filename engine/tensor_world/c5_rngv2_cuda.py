"""CUDA production integration for V2 resident mutation. No host tapes."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load

from engine.tw_rngv2.contract import NAMESPACE_DEFAULT, SEMANTIC_EPOCH
from engine.tw_rngv2.status import Status

from .c5_rngv2_cpu import V2BirthMutation, mutate_planned_births_v2, proposal_subject
from .c5_rngv2_route import assert_v2_no_host_tapes
from .contracts import TensorWorldContractError


_ROOT = Path(__file__).resolve().parents[2]
_NATIVE = Path(__file__).resolve().parent / "native"
_V2_NATIVE = _ROOT / "engine" / "tw_rngv2" / "native"
_SOURCES: Final[tuple[Path, ...]] = (
    _NATIVE / "tensor_world_c5_rngv2_mutation_binding.cpp",
    _NATIVE / "tensor_world_c5_rngv2_mutation.cu",
)
_BINDING_FILES: Final[tuple[Path, ...]] = _SOURCES + (
    _V2_NATIVE / "philox.cuh",
    _V2_NATIVE / "twdmath.cuh",
    _V2_NATIVE / "tw_rngv2.cuh",
)
_MODULES: dict[str, Any] = {}


def cuda_source_digest() -> str:
    digest = sha256()
    for source in _BINDING_FILES:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_v2_mutation_extension(*, verbose: bool = False) -> Any:
    if not torch.cuda.is_available():
        raise TensorWorldContractError("V2 CUDA mutation requires CUDA")
    digest = cuda_source_digest()
    cached = _MODULES.get(digest)
    if cached is not None:
        return cached
    root = Path(
        os.environ.get(
            "EIEL_TWRNGV2_CUDA_BUILD_ROOT",
            public_runtime_path('build_cache/tw_rngv2_qualprep'),
        )
    )
    build = root / digest[:16]
    build.mkdir(parents=True, exist_ok=True)
    prior = os.environ.get("MAX_JOBS")
    os.environ.setdefault("MAX_JOBS", "2")
    try:
        module = load(
            name=f"eiel_tw_c5_rngv2_{digest[:16]}",
            sources=[str(path) for path in _SOURCES],
            build_directory=str(build),
            extra_cflags=["-O2", "-ffp-contract=off"],
            extra_cuda_cflags=[
                "-O3",
                "--fmad=false",
                "--prec-div=true",
                "--prec-sqrt=true",
                "-allow-unsupported-compiler",
                "-lineinfo",
            ],
            with_cuda=True,
            verbose=verbose,
        )
    finally:
        if prior is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = prior
    _MODULES[digest] = module
    return module


def mutate_planned_births_cuda_v2(
    *,
    world_uid: int,
    absolute_tick: int,
    plans: tuple[tuple[int, int, tuple[float, ...], tuple[float, ...], float, float, int], ...],
    namespace: int = NAMESPACE_DEFAULT,
) -> list[V2BirthMutation]:
    if not plans:
        return []
    peers = [((int(row[0]), int(row[1])), int(row[6])) for row in plans]
    subjects = [
        proposal_subject(row[0], row[1], row[6], peers) for row in plans
    ]
    device = torch.device("cuda")
    count = len(plans)
    world = torch.full((count,), world_uid, dtype=torch.uint32, device=device)
    tick = torch.full((count,), absolute_tick, dtype=torch.uint64, device=device)
    subject = torch.tensor(subjects, dtype=torch.uint64, device=device)
    parent_genes = torch.tensor([row[2] for row in plans], dtype=torch.float64, device=device)
    mate_genes = torch.tensor([row[3] for row in plans], dtype=torch.float64, device=device)
    parent_rate = torch.tensor([row[4] for row in plans], dtype=torch.float64, device=device)
    mate_rate = torch.tensor([row[5] for row in plans], dtype=torch.float64, device=device)
    module = load_v2_mutation_extension()
    child_genes, child_rate, retries, status = module.mutate(
        world,
        tick,
        subject,
        parent_genes,
        mate_genes,
        parent_rate,
        mate_rate,
        int(namespace) & ((1 << 64) - 1),
        SEMANTIC_EPOCH,
    )
    results: list[V2BirthMutation] = []
    genes_host = child_genes.detach().cpu()
    rate_host = child_rate.detach().cpu()
    retry_host = retries.detach().cpu()
    status_host = status.detach().cpu()
    for index in range(count):
        results.append(
            V2BirthMutation(
                child_genes=tuple(float(value) for value in genes_host[index].tolist()),
                child_rate=float(rate_host[index]),
                accepted_retries=tuple(int(value) for value in retry_host[index].tolist()),
                status=Status(int(status_host[index])),
            )
        )
    return results


def v2_window_receipt(
    *,
    batch: int,
    tape_capacity: int,
    h2d_tape_bytes: int,
    host_generation_ms: float,
) -> dict[str, object]:
    if tape_capacity != 0 or h2d_tape_bytes != 0 or host_generation_ms < 0:
        raise TensorWorldContractError("V2 window receipt forbids tapes and negative generation")
    return {
        "batch": batch,
        "tape_capacity": 0,
        "h2d_tape_bytes": 0,
        "host_generation_ms": 0.0,
        "host_rng_authority": False,
    }
