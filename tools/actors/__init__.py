"""Local deterministic operator-facing actors for EIEL.

The actors in this package are CLI managers. They contain no AI, run no
background services, and do not execute simulation campaigns.
"""

from __future__ import annotations

from .common import ACTOR_RECORD_SCHEMA_VERSION, ActorError

__all__ = ["ACTOR_RECORD_SCHEMA_VERSION", "ActorError"]
