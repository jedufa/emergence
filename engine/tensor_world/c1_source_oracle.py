"""Receipt-bound source-current fixed-MLP oracle for Tensor World C1."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import torch

from .c1_backend import TensorWorldC1ConfigV1
from .state import TensorWorldStateBundleV1


ORACLE_MODULE_NAME = "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_v1"
ORACLE_ENTRYPOINT = "eiel_cuda_architecture_b_hapb_resident_policy_logits_oracle_v1"
ORACLE_MANIFEST_SCHEMA = (
    "eiel_cuda_architecture_b_wdl_jungle_eed_hapb_extension_manifest_v1"
)
ORACLE_POLICY_ROW_WIDTH = 9991
ORACLE_MAXIMUM_TILE_ROWS = 4096
TENSOR_WORLD_POLICY_ROW_WIDTH = 9415
SOURCE_POLICY_KERNEL = (
    "engine/perf/cuda_architecture_b_flattened_wdl_controller_v2_kernels.cu"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_c1_source_logit_oracle_v1(
    artifact: str | Path, *, repository_root: str | Path
) -> tuple[ModuleType, dict[str, object]]:
    binary = Path(artifact).resolve()
    manifest_path = binary.with_suffix(binary.suffix + ".manifest.json")
    if not binary.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("source-current C1 oracle binary or manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != ORACLE_MANIFEST_SCHEMA:
        raise RuntimeError("source-current C1 oracle manifest schema is unsupported")
    if manifest.get("artifact_sha256") != _sha256(binary):
        raise RuntimeError("source-current C1 oracle binary hash mismatch")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("source-current C1 oracle source hashes are missing")
    compiled_hashes = {
        key: value
        for key, value in source_hashes.items()
        if str(key).endswith(SOURCE_POLICY_KERNEL)
    }
    if len(compiled_hashes) != 1:
        raise RuntimeError("source-current policy kernel has no unique manifest binding")
    current_kernel = Path(repository_root).resolve() / SOURCE_POLICY_KERNEL
    current_hash = _sha256(current_kernel)
    if next(iter(compiled_hashes.values())) != current_hash:
        raise RuntimeError("source-current policy kernel differs from oracle artifact")
    spec = importlib.util.spec_from_file_location(ORACLE_MODULE_NAME, binary)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create source-current C1 oracle loader")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, ORACLE_ENTRYPOINT):
        raise RuntimeError("source-current C1 oracle entrypoint is missing")
    receipt: dict[str, object] = {
        "artifact": str(binary),
        "artifact_sha256": _sha256(binary),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "build_source_commit": manifest["committed_source_snapshot"][
            "build_source_commit"
        ],
        "source_policy_kernel": SOURCE_POLICY_KERNEL,
        "source_policy_kernel_sha256": current_hash,
        "entrypoint": ORACLE_ENTRYPOINT,
        "cxx_flags": manifest.get("cxx_flags"),
        "nvcc_flags": manifest.get("nvcc_flags"),
    }
    return module, receipt


def c1_source_logit_oracle_v1(
    module: ModuleType,
    source: TensorWorldStateBundleV1,
    observations: torch.Tensor,
    config: TensorWorldC1ConfigV1,
    *,
    active: int,
) -> torch.Tensor:
    """Run the frozen source-current exact helper outside the candidate tick."""

    rows = source.batch * active
    pack = torch.zeros(
        (rows, ORACLE_POLICY_ROW_WIDTH), device="cuda", dtype=torch.float32
    )
    source_pack = (
        source.policy.packed_weights[:, :active]
        .contiguous()
        .reshape(rows, TENSOR_WORLD_POLICY_ROW_WIDTH)
        .to(device="cuda")
    )
    pack[:, :TENSOR_WORLD_POLICY_ROW_WIDTH].copy_(source_pack)
    oracle_observations = (
        observations[:, :active]
        .contiguous()
        .reshape(rows, 74)
        .to(device="cuda")
    )
    drives = (
        source.policy.drives[:, :active]
        .contiguous()
        .reshape(rows, 4)
        .to(device="cuda")
    )
    attack = (
        source.policy.attack_bias[:, :active]
        .contiguous()
        .reshape(rows)
        .to(device="cuda")
    )
    temperature = (
        source.policy.action_temperature[:, :active]
        .contiguous()
        .reshape(rows)
        .to(device="cuda")
    )
    gain = (
        source.policy.drive_logit_gain[:, :active]
        .contiguous()
        .reshape(rows)
        .to(device="cuda")
    )
    slots = torch.arange(rows, device="cuda", dtype=torch.int64)
    values = asdict(config)
    logits = torch.cat(
        [
            getattr(module, ORACLE_ENTRYPOINT)(
                pack[start : start + ORACLE_MAXIMUM_TILE_ROWS],
                oracle_observations[start : start + ORACLE_MAXIMUM_TILE_ROWS],
                slots[start : start + ORACLE_MAXIMUM_TILE_ROWS],
                drives,
                attack,
                temperature,
                gain,
                64,
                7,
                values["w_hunger_gather"],
                values["w_pain_rest"],
                values["w_fatigue_rest"],
                values["w_pain_attack"],
                values["w_fear_attack"],
            )
            for start in range(0, rows, ORACLE_MAXIMUM_TILE_ROWS)
        ]
    )
    torch.cuda.synchronize()
    return logits.cpu().view(source.batch, active, 7)
