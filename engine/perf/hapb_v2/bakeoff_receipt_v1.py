"""Receipt schema for the future HAPB V2 policy-kernel bake-off."""

from __future__ import annotations

from typing import Any, Final, Mapping

from engine.perf.hapb_v2.amdahl_projection_v1 import AMDAHL_SCHEMA_V1
from engine.perf.hapb_v2.exactness_contract_v1 import CONTRACT_ID_V1


BAKEOFF_RECEIPT_SCHEMA_V1: Final[str] = "eiel_hapb_v2_policy_kernel_bakeoff_receipt_v1"
REQUIRED_CANDIDATES_V1: Final[tuple[str, ...]] = (
    "v1_nx1",
    "v2_nx64",
)


class HapbV2BakeoffReceiptError(ValueError):
    """Bake-off receipt is malformed."""


def empty_candidate_metrics_v1(name: str) -> dict[str, object]:
    return {
        "name": name,
        "implemented": name in REQUIRED_CANDIDATES_V1,
        "rows_evaluated": None,
        "unique_contents": None,
        "observations_count": None,
        "launch_count": None,
        "warmup_iters": None,
        "timed_iters": None,
        "kernel_elapsed_ms": None,
        "end_to_end_wall_ms": None,
        "logits_sha256": None,
        "mismatch_count": None,
        "first_mismatch": None,
    }


def empty_bakeoff_receipt_v1() -> dict[str, object]:
    return {
        "schema_version": BAKEOFF_RECEIPT_SCHEMA_V1,
        "contract_id": CONTRACT_ID_V1,
        "production_route_mutated": False,
        "gpu_executed": False,
        "candidates": {
            "v1_nx1": empty_candidate_metrics_v1("v1_nx1"),
            "v2_nx64": empty_candidate_metrics_v1("v2_nx64"),
            "grouped_gemm": {
                **empty_candidate_metrics_v1("grouped_gemm"),
                "implemented": False,
                "status": (
                    "not an exact candidate; standard GEMM changes per-neuron "
                    "accumulation order"
                ),
            },
        },
        "exact_logits_parity": None,
        "kernel_speedup_nx64_vs_nx1": None,
        "amdahl": {
            "schema_version": AMDAHL_SCHEMA_V1,
            "status": "inputs_incomplete",
        },
        "decision": None,
    }


def validate_bakeoff_receipt_v1(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != BAKEOFF_RECEIPT_SCHEMA_V1:
        raise HapbV2BakeoffReceiptError("bake-off receipt schema mismatch")
    if receipt.get("contract_id") != CONTRACT_ID_V1:
        raise HapbV2BakeoffReceiptError("bake-off receipt is not bound to the contract")
    candidates = receipt.get("candidates")
    if not isinstance(candidates, Mapping):
        raise HapbV2BakeoffReceiptError("bake-off receipt lacks candidates")
    for name in REQUIRED_CANDIDATES_V1:
        if name not in candidates:
            raise HapbV2BakeoffReceiptError(f"bake-off receipt missing {name}")
    if "amdahl" not in receipt:
        raise HapbV2BakeoffReceiptError("bake-off receipt lacks Amdahl projection")
    if receipt.get("production_route_mutated") is True:
        raise HapbV2BakeoffReceiptError("bake-off must not mutate the production route")
