"""HAPB-owned CUDA backend. Includes TensorWorld headers; does not copy them."""

from __future__ import annotations

from engine.runtime_roots import public_runtime_path

import os
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_NATIVE = _ROOT / "native"
_TW_NATIVE = _ROOT.parent / "tw_rngv2" / "native"
_COMPILE_SOURCES: Final[tuple[Path, ...]] = (
    _NATIVE / "hapb_rngv2_binding.cpp",
    _NATIVE / "hapb_rngv2_cuda.cu",
)
_SOURCE_BINDING_FILES: Final[tuple[Path, ...]] = _COMPILE_SOURCES + (
    _NATIVE / "hapb_rngv2.cuh",
    _TW_NATIVE / "philox.cuh",
    _TW_NATIVE / "twdmath.cuh",
    _TW_NATIVE / "tw_rngv2.cuh",
)
_MODULES: dict[str, Any] = {}


def cuda_source_digest() -> str:
    digest = sha256()
    for source in _SOURCE_BINDING_FILES:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def native_sources_exist() -> bool:
    return all(path.is_file() for path in _SOURCE_BINDING_FILES)


def load_cuda_extension(
    *,
    build_root: str | Path | None = None,
    verbose: bool = False,
) -> Any:
    if not torch.cuda.is_available():
        raise RuntimeError("HAPB-RNGV2 CUDA backend requires an available CUDA device")
    source_digest = cuda_source_digest()
    cached = _MODULES.get(source_digest)
    if cached is not None:
        return cached
    root = Path(
        build_root
        or os.environ.get(
            "EIEL_HAPB_RNGV2_CUDA_BUILD_ROOT",
            public_runtime_path('build_cache/hapb_rngv2'),
        )
    )
    build_directory = root / source_digest[:16]
    build_directory.mkdir(parents=True, exist_ok=True)
    prior_max_jobs = os.environ.get("MAX_JOBS")
    os.environ.setdefault("MAX_JOBS", "2")
    try:
        module = load(
            name=f"eiel_hapb_rngv2_cuda_{source_digest[:16]}",
            sources=[str(source) for source in _COMPILE_SOURCES],
            build_directory=str(build_directory),
            extra_include_paths=[str(_NATIVE), str(_TW_NATIVE)],
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
        if prior_max_jobs is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = prior_max_jobs
    _MODULES[source_digest] = module
    return module


def generate_device(
    *,
    world,
    domain,
    epoch,
    subject,
    tick,
    operation,
    n_values,
    namespace_key: int,
    module: Any | None = None,
):
    backend = module or load_cuda_extension()
    return backend.generate(
        world.contiguous(),
        domain.contiguous(),
        epoch.contiguous(),
        subject.contiguous(),
        tick.contiguous(),
        operation.contiguous(),
        n_values.contiguous(),
        int(namespace_key) & ((1 << 64) - 1),
    )


def philox_device(counters, keys, *, rounds: int = 10, module: Any | None = None):
    backend = module or load_cuda_extension()
    return backend.philox(counters.contiguous(), keys.contiguous(), int(rounds))
