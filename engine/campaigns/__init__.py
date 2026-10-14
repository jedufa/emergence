"""Capability-driven declarative campaign planning for EIEL.

Public campaign-planning exports are intentionally lazy.  Storage/lifecycle
actuators import narrow ``engine.campaigns`` submodules for JSON and receipt
contracts; importing this package must not eagerly load the production
registry or its CUDA stack in those storage-only processes.
"""

from __future__ import annotations

import importlib
from typing import Any


_LAZY_EXPORTS = {
    "CampaignError": (".compiler", "CampaignError"),
    "compile_campaign": (".compiler", "compile_campaign"),
    "load_campaign_spec": (".compiler", "load_campaign_spec"),
    "compare_qualification_keys_v1": (
        ".qualification",
        "compare_qualification_keys_v1",
    ),
    "qualification_reuse_receipt_v1": (
        ".qualification",
        "qualification_reuse_receipt_v1",
    ),
    "default_profile_registry": (".registry", "default_profile_registry"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

__all__ = [
    "CampaignError",
    "compile_campaign",
    "compare_qualification_keys_v1",
    "default_profile_registry",
    "load_campaign_spec",
    "qualification_reuse_receipt_v1",
]
