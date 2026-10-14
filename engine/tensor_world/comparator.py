"""Automated canonical state comparison for Tensor World C1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .contracts import EquivalenceClassV1, TensorWorldContractError


@dataclass(frozen=True, slots=True)
class ComparisonRuleV1:
    field: str
    equivalence: EquivalenceClassV1
    atol: float = 0.0
    rtol: float = 0.0


@dataclass(frozen=True, slots=True)
class FieldMismatchV1:
    field: str
    subsystem: str
    equivalence: str
    mismatch_count: int
    max_abs_error: float | None


@dataclass(frozen=True, slots=True)
class OracleComparisonV1:
    passed: bool
    compared_fields: tuple[str, ...]
    mismatches: tuple[FieldMismatchV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "tensor_world_oracle_comparison_v1",
            "passed": self.passed,
            "compared_fields": list(self.compared_fields),
            "mismatches": [
                {
                    "field": item.field,
                    "subsystem": item.subsystem,
                    "equivalence": item.equivalence,
                    "mismatch_count": item.mismatch_count,
                    "max_abs_error": item.max_abs_error,
                }
                for item in self.mismatches
            ],
        }


def _resolve(value: Mapping[str, Any], field: str) -> Any:
    node: Any = value
    for component in field.split("."):
        if not isinstance(node, Mapping) or component not in node:
            raise TensorWorldContractError(f"comparison field is absent: {field}")
        node = node[component]
    return node


def _bit_mismatch_count(left: torch.Tensor, right: torch.Tensor) -> int:
    """Count tensor elements whose stored bytes differ."""

    left_bytes = left.contiguous().view(torch.uint8).reshape(-1, left.element_size())
    right_bytes = right.contiguous().view(torch.uint8).reshape(-1, right.element_size())
    return int((left_bytes != right_bytes).any(dim=1).sum().item())


class OracleComparatorV1:
    """Compare exported canonical state without becoming a runtime dependency."""

    def __init__(self, rules: tuple[ComparisonRuleV1, ...]) -> None:
        if not rules or len({rule.field for rule in rules}) != len(rules):
            raise TensorWorldContractError(
                "comparison rules must be nonempty and unique"
            )
        self._rules = rules

    def compare(
        self,
        oracle: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> OracleComparisonV1:
        mismatches: list[FieldMismatchV1] = []
        compared: list[str] = []
        for rule in self._rules:
            if rule.equivalence is EquivalenceClassV1.DIAGNOSTIC_ONLY:
                continue
            left = _resolve(oracle, rule.field)
            right = _resolve(candidate, rule.field)
            compared.append(rule.field)
            if not isinstance(left, torch.Tensor) or not isinstance(
                right, torch.Tensor
            ):
                equal = left == right
                mismatch_count = 0 if equal else 1
                max_abs_error = None
            else:
                if left.shape != right.shape or left.dtype != right.dtype:
                    mismatch_count = max(left.numel(), right.numel(), 1)
                    max_abs_error = None
                elif rule.equivalence is EquivalenceClassV1.BIT_EXACT_REQUIRED:
                    mismatch_count = _bit_mismatch_count(left, right)
                    max_abs_error = None
                else:
                    close = torch.isclose(left, right, atol=rule.atol, rtol=rule.rtol)
                    mismatch_count = int((~close).sum().item())
                    if mismatch_count:
                        max_abs_error = float(
                            (left.to(torch.float64) - right.to(torch.float64))
                            .abs()
                            .max()
                            .item()
                        )
                    else:
                        max_abs_error = 0.0
            if mismatch_count:
                mismatches.append(
                    FieldMismatchV1(
                        field=rule.field,
                        subsystem=rule.field.split(".", 1)[0],
                        equivalence=rule.equivalence.value,
                        mismatch_count=mismatch_count,
                        max_abs_error=max_abs_error,
                    )
                )
        return OracleComparisonV1(
            passed=not mismatches,
            compared_fields=tuple(compared),
            mismatches=tuple(mismatches),
        )
