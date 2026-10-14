"""Build an immutable full C2 extension with the isolated R12 policy kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch
from torch.utils.cpp_extension import load

from engine.tensor_world.c2_policy_parallel_r12 import (
    NATIVE_SOURCE,
    SCALAR_SOURCE,
    render_tensor_world_c2_policy_parallel_r12_full_source_v1,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE_AUTHORITIES = (
    ROOT / "engine/perf/cuda_architecture_b_persistent_patch_p2.cuh",
    ROOT / "engine/perf/cuda_architecture_b_persistent_wdl_v1.cuh",
    ROOT / "engine/perf/cuda_architecture_b_jungle_primitives.cuh",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build immutable Tensor World C2 parallel-policy module"
    )
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    build_dir = Path(args.build_dir).resolve()
    output = Path(args.output).resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if build_dir.exists():
        raise FileExistsError("immutable C2 R12 build directory already exists")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("immutable C2 R12 build output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required to bind target architecture")

    torch.cuda.set_device(args.device_index)
    capability = torch.cuda.get_device_capability(args.device_index)
    architecture = f"{capability[0]}.{capability[1]}"
    prior_architecture = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = architecture
    build_dir.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    generated_source = build_dir / "tensor_world_c2_parallel_r12_generated.cu"
    generated_source.write_text(
        render_tensor_world_c2_policy_parallel_r12_full_source_v1(),
        encoding="utf-8",
    )
    try:
        module = load(
            # The receipt-bound C2 loader imports every compatible artifact
            # under the stable ``tensor_world_c2_ext`` module name.  Keep the
            # optimized artifact drop-in compatible; the R12 variant remains
            # identified by its manifest and external path.
            name="tensor_world_c2_ext",
            sources=[str(generated_source)],
            extra_include_paths=[str(ROOT / "engine/perf")],
            extra_cflags=["-O2", "-ffp-contract=off"],
            extra_cuda_cflags=[
                "-O2",
                "--fmad=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            ],
            with_cuda=True,
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if prior_architecture is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prior_architecture

    shutil.copy2(Path(module.__file__).resolve(), output)
    properties = torch.cuda.get_device_properties(args.device_index)
    sources = (
        SCALAR_SOURCE,
        NATIVE_SOURCE,
        Path(__file__).resolve(),
        *SOURCE_AUTHORITIES,
    )
    manifest = {
        "schema_version": "tensor_world_c2_cuda_build_manifest_v1",
        "variant": "tensor_world_c2_policy_parallel_r12_v1",
        "source_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "sources": {
            str(path.relative_to(ROOT)): _sha256(path) for path in sources
        },
        "generated_source_sha256": _sha256(generated_source),
        "binary_sha256": _sha256(output),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": properties.name,
        "compute_capability": architecture,
        "flags": {
            "cxx": ["-O2", "-ffp-contract=off"],
            "nvcc": [
                "-O2",
                "--fmad=false",
                "--prec-div=true",
                "--prec-sqrt=true",
            ],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "manifest": str(manifest_path),
                "binary_sha256": manifest["binary_sha256"],
                "generated_source_sha256": manifest["generated_source_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
