"""Isolated exact-order block-parallel C2 fixed-MLP policy prototype."""

from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from types import ModuleType
from typing import Final

from .c2_native_loader import load_tensor_world_c2_extension
from .contracts import TensorWorldContractError


ROOT = Path(__file__).resolve().parents[2]
NATIVE_SOURCE = (
    ROOT
    / "engine/tensor_world/native/tensor_world_c2_policy_parallel_r12.cu"
)
SCALAR_SOURCE = ROOT / "engine/tensor_world/native/tensor_world_c2_kernels.cu"
BUILDER_SOURCE = (
    ROOT / "tools/scripts/build_tensor_world_c2_policy_parallel_r12_extension.py"
)

C2_POLICY_PARALLEL_R12_SCHEMA_VERSION_V1: Final[str] = (
    "tensor_world_c2_policy_parallel_r12_v1"
)
C2_POLICY_PARALLEL_R12_DECISION_V1: Final[str] = "NO_GATE_DECISION"
C2_POLICY_PARALLEL_R12_THREADS_PER_ROW_V1: Final[int] = 64
C2_POLICY_PARALLEL_R12_PREBUILT_ENV_V1: Final[str] = (
    "TENSOR_WORLD_C2_PARALLEL_R12_PREBUILT"
)


def render_tensor_world_c2_policy_parallel_r12_full_source_v1() -> str:
    """Substitute only the policy helper/launch in the full current C2 source."""

    scalar = SCALAR_SOURCE.read_text(encoding="utf-8")
    parallel = NATIVE_SOURCE.read_text(encoding="utf-8")
    kernel_begin = "// EIEL_C2_POLICY_PARALLEL_R12_KERNEL_BEGIN"
    kernel_end = "// EIEL_C2_POLICY_PARALLEL_R12_KERNEL_END"
    if parallel.count(kernel_begin) != 1 or parallel.count(kernel_end) != 1:
        raise TensorWorldContractError("C2 parallel policy kernel markers diverged")
    kernel = parallel.split(kernel_begin, 1)[1].split(kernel_end, 1)[0].strip()

    helper_begin = "__device__ __forceinline__ void policy_exact("
    helper_end = "__global__ void select_kernel("
    call_begin = "void c2_policy("
    call_end = "void c2_select("
    for sentinel in (helper_begin, helper_end, call_begin, call_end):
        if scalar.count(sentinel) != 1:
            raise TensorWorldContractError(
                f"C2 scalar policy substitution sentinel diverged: {sentinel}"
            )
    policy_region_start = scalar.index(helper_begin)
    policy_region_end = scalar.index(helper_end, policy_region_start)
    declarations = (
        "constexpr int64_t kDriveWidth = 4;\n"
        "constexpr int kThreadsPerRow = 64;\n\n"
    )
    generated = (
        scalar[:policy_region_start]
        + declarations
        + kernel
        + "\n\n"
        + scalar[policy_region_end:]
    )
    call_start = generated.index(call_begin)
    call_finish = generated.index(call_end, call_start)
    parallel_call = """void c2_policy(
    std::vector<torch::Tensor> t, torch::Tensor ci, torch::Tensor cf) {
    validate_call(t, ci, cf);
    const auto rows = t[ALIVE].numel();
    policy_parallel_kernel<<<
        rows, kThreadsPerRow, 0, at::cuda::getCurrentCUDAStream()
    >>>(
        ptr<float*>(t, LOGITS), ptr<const float*>(t, C2_PACK),
        ptr<const float*>(t, OBSERVATIONS), ptr<const float*>(t, DRIVES),
        ptr<const float*>(t, ATTACK_BIAS),
        ptr<const float*>(t, ACTION_TEMPERATURE),
        ptr<const float*>(t, DRIVE_LOGIT_GAIN), ptr<const bool*>(t, ALIVE),
        cf.data_ptr<double>(), rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

"""
    return generated[:call_start] + parallel_call + generated[call_finish:]


def load_tensor_world_c2_policy_parallel_r12_full_extension(
    path: str | Path | None = None,
) -> ModuleType:
    """Load the full drop-in C2 module built by the isolated R12 builder."""

    configured = path or os.environ.get(C2_POLICY_PARALLEL_R12_PREBUILT_ENV_V1)
    if not configured:
        raise TensorWorldContractError(
            f"{C2_POLICY_PARALLEL_R12_PREBUILT_ENV_V1} must name the built module"
        )
    return load_tensor_world_c2_extension(configured)


def c2_policy_parallel_r12_source_receipt_v1() -> dict[str, object]:
    """Describe the bounded prototype without making a performance claim."""

    sources = {
        "parallel_policy": NATIVE_SOURCE,
        "scalar_oracle": SCALAR_SOURCE,
        "loader": Path(__file__),
        "immutable_builder": BUILDER_SOURCE,
    }
    return {
        "schema_version": C2_POLICY_PARALLEL_R12_SCHEMA_VERSION_V1,
        "decision": C2_POLICY_PARALLEL_R12_DECISION_V1,
        "threads_per_row": C2_POLICY_PARALLEL_R12_THREADS_PER_ROW_V1,
        "row_mapping": "one_CUDA_block_per_batch_capacity_row",
        "arithmetic_order": {
            "hidden_1": "one_thread_per_neuron_feature_0_to_82",
            "hidden_2": "one_thread_per_neuron_hidden_0_to_63",
            "output": "threads_0_to_6_hidden_0_to_63",
            "cross_thread_reduction": False,
        },
        "integration": (
            "full_receipt_bound_C2_drop_in_via_TENSOR_WORLD_C2_PREBUILT;_"
            "R11_source_unchanged"
        ),
        "gate_claim": False,
        "sources": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in sources.items()
        },
    }


__all__ = [
    "C2_POLICY_PARALLEL_R12_DECISION_V1",
    "C2_POLICY_PARALLEL_R12_PREBUILT_ENV_V1",
    "C2_POLICY_PARALLEL_R12_SCHEMA_VERSION_V1",
    "C2_POLICY_PARALLEL_R12_THREADS_PER_ROW_V1",
    "c2_policy_parallel_r12_source_receipt_v1",
    "load_tensor_world_c2_policy_parallel_r12_full_extension",
    "render_tensor_world_c2_policy_parallel_r12_full_source_v1",
]
