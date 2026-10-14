"""Single release identity for the repository-local EIEL operator tooling."""

from __future__ import annotations

ACTOR_MANAGERS_VERSION = "1.19.0"
OPERATOR_CONTRACT_VERSION = "eiel_codex_operator_contract_v1"


def version_text() -> str:
    return f"EIEL operator tooling {ACTOR_MANAGERS_VERSION}"


def operator_metadata() -> dict[str, str]:
    return {
        "operator_contract_version": OPERATOR_CONTRACT_VERSION,
        "tooling_version": ACTOR_MANAGERS_VERSION,
    }


__all__ = [
    "ACTOR_MANAGERS_VERSION",
    "OPERATOR_CONTRACT_VERSION",
    "operator_metadata",
    "version_text",
]
