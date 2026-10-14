"""Portable, advisory development automation for Emergence."""

from .core import (
    CONFIG_VERSION,
    CONTEXT_VERSION,
    DevAutomationError,
    build_context,
    find_project_root,
    load_config,
)

__all__ = [
    "CONFIG_VERSION",
    "CONTEXT_VERSION",
    "DevAutomationError",
    "build_context",
    "find_project_root",
    "load_config",
]
