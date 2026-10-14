"""Gate for the isolated HAPB V2 experimental GPU path."""

from __future__ import annotations

import os

from engine.perf.hapb_v2 import HAPB_V2_EXPERIMENTAL_ENV_V1


class HapbV2ExperimentalDisabled(RuntimeError):
    """The V2 GPU path was requested without the explicit experimental flag."""


def experimental_enabled_v1() -> bool:
    return os.environ.get(HAPB_V2_EXPERIMENTAL_ENV_V1) == "1"


def require_experimental_v1() -> None:
    if not experimental_enabled_v1():
        raise HapbV2ExperimentalDisabled(
            f"set {HAPB_V2_EXPERIMENTAL_ENV_V1}=1 to use the isolated HAPB V2 path"
        )
