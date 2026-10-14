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


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "engine/tensor_world/native/tensor_world_c2_kernels.cu"
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
    parser = argparse.ArgumentParser(description="Build immutable Tensor World C2 CUDA modules")
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    build_dir, output = Path(args.build_dir).resolve(), Path(args.output).resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("immutable Tensor World C2 build output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA device is required to bind the target architecture")
    torch.cuda.set_device(args.device_index)
    capability = torch.cuda.get_device_capability(args.device_index)
    arch = f"{capability[0]}.{capability[1]}"
    prior_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    build_dir.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        module = load(
            name="tensor_world_c2_ext",
            sources=[str(SOURCE)],
            extra_include_paths=[str(ROOT / "engine/perf")],
            extra_cflags=["-O2", "-ffp-contract=off"],
            extra_cuda_cflags=["-O2", "--fmad=false", "--prec-div=true", "--prec-sqrt=true"],
            with_cuda=True,
            build_directory=str(build_dir),
            verbose=False,
        )
    finally:
        if prior_arch is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = prior_arch
    shutil.copy2(Path(module.__file__).resolve(), output)
    properties = torch.cuda.get_device_properties(args.device_index)
    sources = (SOURCE, *SOURCE_AUTHORITIES)
    manifest = {
        "schema_version": "tensor_world_c2_cuda_build_manifest_v1",
        "source_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "sources": {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        "binary_sha256": _sha256(output),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device_name": properties.name, "compute_capability": arch,
        "flags": {"cxx": ["-O2", "-ffp-contract=off"],
                  "nvcc": ["-O2", "--fmad=false", "--prec-div=true", "--prec-sqrt=true"]},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path),
                      "binary_sha256": manifest["binary_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
