"""Shared result types for mechanical review validators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:
    check_id: str
    validator: str
    status: str = "PASS"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.status = "FAIL"
        self.errors.append(message)

    def warn(self, message: str) -> None:
        if self.status == "PASS":
            self.status = "WARN"
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "validator": self.validator,
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
            "details": self.details,
        }


@dataclass(frozen=True)
class SemanticQuestion:
    trigger_id: str
    trigger_type: str
    question: str
    evidence_pointers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_id": self.trigger_id,
            "trigger_type": self.trigger_type,
            "question": self.question,
            "evidence_pointers": list(self.evidence_pointers),
        }
