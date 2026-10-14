"""Canonical composition entrypoint for ExecutionRequestV2 and V1 continuation."""

from __future__ import annotations

from collections.abc import Sequence

from tools.execution_cli import main as _execution_cli_main

from .cuda_full_evolution_v2_launchpad_adapter import (
    CanonicalCudaFullEvolutionV2LaunchpadAdapterV1,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Inject the canonical Scheduler/CRCA adapter at the operational CLI seam."""

    return _execution_cli_main(
        argv,
        cuda_full_evolution_v2_adapter_factory=(
            CanonicalCudaFullEvolutionV2LaunchpadAdapterV1
        ),
    )


__all__ = ["main"]
