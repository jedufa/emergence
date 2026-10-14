"""Compatibility exports for the compact CUDA science publication owner.

The implementation lives beside the existing compact-capture seam so campaign
providers do not create a second architecture-ratchet dependency edge.
"""

from .cuda_v2_compact_science_capture_v1 import (
    COMPACT_SCIENCE_ASYNC_PENDING_SLOTS_DEFAULT_V1,
    COMPACT_SCIENCE_ASYNC_PUBLICATION_SCHEMA_V1,
    COMPACT_SCIENCE_ASYNC_WORKERS_DEFAULT_V1,
    CompactScienceAsyncPublicationError,
    CompactScienceAsyncPublicationV1,
    CompactSciencePublicationTokenV1,
)


__all__ = (
    "COMPACT_SCIENCE_ASYNC_PENDING_SLOTS_DEFAULT_V1",
    "COMPACT_SCIENCE_ASYNC_PUBLICATION_SCHEMA_V1",
    "COMPACT_SCIENCE_ASYNC_WORKERS_DEFAULT_V1",
    "CompactScienceAsyncPublicationError",
    "CompactScienceAsyncPublicationV1",
    "CompactSciencePublicationTokenV1",
)
