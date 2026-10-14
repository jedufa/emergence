"""Compile the isolated HAPB V2 experimental extension.

Not a production artifact.  Compilation does not launch kernels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from engine.perf.hapb_v2.experimental import require_experimental_v1


ROOT: Final[Path] = Path(__file__).resolve().parents[3]
MODULE_NAME: Final[str] = "eiel_hapb_v2_nx64_experimental_v1"
SOURCES: Final[tuple[Path, ...]] = (
    ROOT / "engine" / "perf" / "hapb_v2_policy_nx64_binding.cpp",
    ROOT / "engine" / "perf" / "hapb_v2_policy_nx64_kernel.cu",
)
# nvcc rejects --use_fast_math=false (the switch takes no argument).
EXTRA_CUDA_CFLAGS_V1: Final[tuple[str, ...]] = (
    "-O3",
    "--fmad=false",
    "-lineinfo",
)


def compile_experimental_extension_v1(*, output_dir: Path) -> object:
    require_experimental_v1()
    from torch.utils.cpp_extension import load

    output_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name=MODULE_NAME,
        sources=[str(path) for path in SOURCES],
        extra_include_paths=[str(ROOT / "engine" / "perf")],
        extra_cuda_cflags=list(EXTRA_CUDA_CFLAGS_V1),
        extra_cflags=["-O3"],
        build_directory=str(output_dir),
        verbose=True,
    )
