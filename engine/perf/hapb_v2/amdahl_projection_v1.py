"""Amdahl composed-path projection for the future HAPB V2 GPU decision."""

from __future__ import annotations

from typing import Final


AMDAHL_SCHEMA_V1: Final[str] = "eiel_hapb_v2_amdahl_composed_path_v1"
COMPOSED_IMPROVEMENT_THRESHOLD_V1: Final[float] = 0.25
COMPOSED_SPEEDUP_THRESHOLD_V1: Final[float] = 1.0 / (
    1.0 - COMPOSED_IMPROVEMENT_THRESHOLD_V1
)


class HapbV2AmdahlError(ValueError):
    """Amdahl projection inputs are unusable."""


def composed_path_speedup_v1(
    *,
    policy_kernel_fraction: float,
    kernel_speedup: float,
) -> float:
    """Return 1 / ((1 - f) + f / s) for measured f and s."""

    if (
        type(policy_kernel_fraction) is not float
        and type(policy_kernel_fraction) is not int
    ) or not 0.0 <= float(policy_kernel_fraction) <= 1.0:
        raise HapbV2AmdahlError("policy_kernel_fraction must be in [0, 1]")
    if (
        type(kernel_speedup) is not float and type(kernel_speedup) is not int
    ) or float(kernel_speedup) <= 0.0:
        raise HapbV2AmdahlError("kernel_speedup must be > 0")
    fraction = float(policy_kernel_fraction)
    speedup = float(kernel_speedup)
    return 1.0 / ((1.0 - fraction) + (fraction / speedup))


def project_v2_decision_inputs_v1(
    *,
    policy_kernel_fraction: float | None,
    kernel_speedup: float | None,
    logits_bit_exact: bool | None,
    complexity_bounded: bool | None,
) -> dict[str, object]:
    missing = [
        name
        for name, value in (
            ("policy_kernel_fraction", policy_kernel_fraction),
            ("kernel_speedup", kernel_speedup),
            ("logits_bit_exact", logits_bit_exact),
            ("complexity_bounded", complexity_bounded),
        )
        if value is None
    ]
    if missing:
        return {
            "schema_version": AMDAHL_SCHEMA_V1,
            "status": "inputs_incomplete",
            "missing": missing,
            "decision": None,
            "threshold_composed_improvement": COMPOSED_IMPROVEMENT_THRESHOLD_V1,
            "threshold_composed_speedup": COMPOSED_SPEEDUP_THRESHOLD_V1,
        }
    composed = composed_path_speedup_v1(
        policy_kernel_fraction=float(policy_kernel_fraction),
        kernel_speedup=float(kernel_speedup),
    )
    improvement = composed - 1.0
    material_policy = float(policy_kernel_fraction) >= 0.15
    v2_a = (
        bool(logits_bit_exact)
        and bool(complexity_bounded)
        and material_policy
        and improvement + 1e-12 >= COMPOSED_IMPROVEMENT_THRESHOLD_V1
    )
    return {
        "schema_version": AMDAHL_SCHEMA_V1,
        "status": "projected",
        "policy_kernel_fraction": float(policy_kernel_fraction),
        "kernel_speedup": float(kernel_speedup),
        "composed_path_speedup": composed,
        "composed_path_improvement": improvement,
        "logits_bit_exact": bool(logits_bit_exact),
        "complexity_bounded": bool(complexity_bounded),
        "policy_component_material": material_policy,
        "threshold_composed_improvement": COMPOSED_IMPROVEMENT_THRESHOLD_V1,
        "threshold_composed_speedup": COMPOSED_SPEEDUP_THRESHOLD_V1,
        "decision": "V2-A" if v2_a else "V2-C",
        "decision_basis": (
            "whole-tick Amdahl projection from measured H4 policy-kernel "
            "fraction and measured kernel speedup; kernel speedup alone is "
            "not sufficient"
        ),
    }
