"""V2 status codes. Fail-closed codes never publish a value."""

from __future__ import annotations

from enum import IntEnum


class Status(IntEnum):
    RNG_OK = 0
    RNG_UNKNOWN_DOMAIN = 1
    RNG_FIELD_OVERFLOW = 2
    RNG_ADDRESS_EXHAUSTED = 3
    RNG_INVALID_N = 4
    RNG_NONFINITE = 5
    RNG_DUPLICATE_ADDRESS = 6
    RNG_TRANSACTION_ABORTED = 7
