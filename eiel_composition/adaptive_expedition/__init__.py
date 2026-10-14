"""Deterministic outer science coordination above the EIEL science front door.

The package deliberately exports no policy implementation.  Coordinators consume
the generic policy protocol and receive a registry entry from the operator-facing
composition root.  Importing this package must never activate execution.
"""
from __future__ import annotations
