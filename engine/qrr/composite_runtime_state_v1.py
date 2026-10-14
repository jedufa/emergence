"""Compatibility export for the authoritative composite state projector.

QRR consumes the projector and supplies the current backend digest algorithm;
it does not own runtime checkpoint composition.
"""

from typing import Any

from engine.perf.composite_runtime_state_v1 import (
    COMPOSITE_RUNTIME_STATE_SCHEMA_VERSION,
    CompositeRuntimeStateError,
    CompositeRuntimeStateV1,
    compose_composite_runtime_state_v1 as _compose,
    projection_digest,
    validate_composite_state_exact_equality_v1,
)


def compose_composite_runtime_state_v1(*args: Any, **kwargs: Any) -> CompositeRuntimeStateV1:
    """Preserve the existing QRR call surface during the ownership inversion."""

    return _compose(*args, projection_digest_fn=projection_digest, **kwargs)


__all__ = [
    "COMPOSITE_RUNTIME_STATE_SCHEMA_VERSION",
    "CompositeRuntimeStateError",
    "CompositeRuntimeStateV1",
    "compose_composite_runtime_state_v1",
    "validate_composite_state_exact_equality_v1",
]
