from __future__ import annotations

"""RUNNERCORE-0 supported-profile runner contract.

The package is default-off scaffolding for a future single-lane runner core.
It records the supported profile contract and resident slot-state shape, while
candidate/native modes fail closed until the live tick can be owned without
per-tick Python Agent synchronization.
"""

from simulation.runnercore.gate import (
    RunnerCoreUnsupportedError,
    build_runnercore_summary,
    maybe_runnercore_supported_tick,
)
from simulation.runnercore.profile import (
    RUNNERCORE_DECISION_BLOCKED_BY_BOUNDARY,
    SUPPORTED_PROFILE,
    classify_supported_profile,
    profile_contract_payload,
    runnercore_requested,
)
from simulation.runnercore.state import (
    RunnerCoreState,
    import_generation_start,
    state_digest,
    state_summary_payload,
)

__all__ = [
    "RUNNERCORE_DECISION_BLOCKED_BY_BOUNDARY",
    "RunnerCoreState",
    "RunnerCoreUnsupportedError",
    "SUPPORTED_PROFILE",
    "build_runnercore_summary",
    "classify_supported_profile",
    "import_generation_start",
    "maybe_runnercore_supported_tick",
    "profile_contract_payload",
    "runnercore_requested",
    "state_digest",
    "state_summary_payload",
]
