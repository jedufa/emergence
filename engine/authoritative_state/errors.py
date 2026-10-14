"""Stable fail-closed errors for authoritative-state boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


TYPE_ERROR: Final[str] = "E_STATE_TYPE"
INTEGER_RANGE_ERROR: Final[str] = "E_STATE_INT_RANGE"
LINEAGE_FOUNDER_RANGE_ERROR: Final[str] = "E_STATE_LINEAGE_FOUNDER_RANGE"
UINT64_JSON_ERROR: Final[str] = "E_STATE_UINT64_JSON"
FLOAT_FINITE_ERROR: Final[str] = "E_STATE_FLOAT_FINITE"
FLOAT_POSITIVE_ERROR: Final[str] = "E_STATE_FLOAT_POSITIVE"
FLOAT32_RANGE_ERROR: Final[str] = "E_STATE_FLOAT32_RANGE"
IDENTITY_FIELDS_ERROR: Final[str] = "E_STATE_IDENTITY_FIELDS"
IDENTITY_VALUE_ERROR: Final[str] = "E_STATE_IDENTITY_VALUE"
IDENTITY_ORDER_ERROR: Final[str] = "E_STATE_IDENTITY_ORDER"
IDENTITY_MISMATCH_ERROR: Final[str] = "E_STATE_IDENTITY_MISMATCH"


@dataclass(eq=False)
class AuthoritativeStateContractError(ValueError):
    """A machine-classifiable authoritative-state contract rejection."""

    code: str
    path: str
    detail: str

    def __str__(self) -> str:
        return f"{self.code}:{self.path}: {self.detail}"


def fail(code: str, path: str, detail: str) -> AuthoritativeStateContractError:
    """Construct one stable contract error without raising it implicitly."""

    return AuthoritativeStateContractError(code=code, path=path, detail=detail)
