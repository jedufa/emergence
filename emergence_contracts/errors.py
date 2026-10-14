"""Machine-readable, owner-specific errors for public EIEL contracts."""

from __future__ import annotations

from enum import Enum
from types import MappingProxyType
from typing import Mapping


class ContractErrorCategory(str, Enum):
    """Frozen cross-module error categories."""

    CONTRACT_INVALID = "CONTRACT_INVALID"
    CONFIGURATION_INVALID = "CONFIGURATION_INVALID"
    CAPABILITY_UNSUPPORTED = "CAPABILITY_UNSUPPORTED"
    DETERMINISM_VIOLATION = "DETERMINISM_VIOLATION"
    STATE_INTEGRITY_FAILURE = "STATE_INTEGRITY_FAILURE"
    CHECKPOINT_INCOMPATIBLE = "CHECKPOINT_INCOMPATIBLE"
    RUNTIME_EXECUTION_FAILURE = "RUNTIME_EXECUTION_FAILURE"
    RESOURCE_UNAVAILABLE = "RESOURCE_UNAVAILABLE"
    EXPERIMENT_INCOMPLETE = "EXPERIMENT_INCOMPLETE"
    EVIDENCE_INVALID = "EVIDENCE_INVALID"
    QUALIFICATION_BLOCKED = "QUALIFICATION_BLOCKED"


class ContractError(ValueError):
    """A deterministic contract failure that callers can classify by code."""

    __slots__ = ("category", "owner", "message", "details")

    def __init__(
        self,
        category: ContractErrorCategory,
        *,
        owner: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(category, ContractErrorCategory):
            raise TypeError("category must be a ContractErrorCategory")
        if not isinstance(owner, str) or not owner or owner.strip() != owner:
            raise TypeError("owner must be a non-empty, unpadded string")
        if not isinstance(message, str) or not message:
            raise TypeError("message must be a non-empty string")
        copied_details = dict(details or {})
        if any(not isinstance(key, str) for key in copied_details):
            raise TypeError("contract error detail keys must be strings")
        self.category = category
        self.owner = owner
        self.message = message
        self.details = MappingProxyType(copied_details)
        super().__init__(f"{category.value} [{owner}]: {message}")

    @property
    def code(self) -> str:
        return self.category.value

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category.value,
            "owner": self.owner,
            "message": self.message,
            "details": dict(self.details),
        }


def contract_invalid(
    message: str,
    *,
    field: str,
    value_type: str | None = None,
) -> ContractError:
    """Build the common validation failure without hiding its category."""

    details: dict[str, object] = {"field": field}
    if value_type is not None:
        details["value_type"] = value_type
    return ContractError(
        ContractErrorCategory.CONTRACT_INVALID,
        owner="emergence_contracts",
        message=message,
        details=details,
    )
