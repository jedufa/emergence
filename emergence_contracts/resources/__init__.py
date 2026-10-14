"""Integrity-checked, named resources shipped with :mod:`emergence_contracts`."""

from hashlib import sha256
from importlib import resources as importlib_resources
from types import MappingProxyType


CRCA_RUN_ATTEMPT_EDGE_CASES_V1 = "crca.run-attempt-edge-cases.v1"
CRCA_G0_RATCHET_RECONCILIATION_V1 = "crca.g0-ratchet-reconciliation.v1"

_RESOURCE_ALLOWLIST = MappingProxyType(
    {
        CRCA_RUN_ATTEMPT_EDGE_CASES_V1: (
            "07_run_attempt_edge_cases.csv",
            "6d2856a982117064536fd7b0750df4a78040e183f725d38fa7cb9c9a3d0d41bd",
        ),
        CRCA_G0_RATCHET_RECONCILIATION_V1: (
            "ratchet_reconciliation.json",
            "3a681db683d9f6609ab8dcb38eba5b8f3aed9fc23e11053c005e7da29aa40bdf",
        ),
    }
)


class ResourceIntegrityError(RuntimeError):
    """Raised when a packaged contract resource differs from its pinned digest."""


def read_resource_bytes(name: str) -> bytes:
    """Read one allowlisted resource after checking its SHA-256 digest.

    Names are stable API identifiers. Callers cannot use this function to read
    arbitrary package paths or filesystem paths.
    """

    try:
        filename, expected_sha256 = _RESOURCE_ALLOWLIST[name]
    except KeyError:
        raise KeyError(f"unknown contract resource name: {name!r}") from None

    payload = importlib_resources.files(__package__).joinpath(filename).read_bytes()
    actual_sha256 = sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ResourceIntegrityError(
            f"contract resource {name!r} has SHA-256 {actual_sha256}; "
            f"expected {expected_sha256}"
        )
    return payload


__all__ = [
    "CRCA_G0_RATCHET_RECONCILIATION_V1",
    "CRCA_RUN_ATTEMPT_EDGE_CASES_V1",
    "ResourceIntegrityError",
    "read_resource_bytes",
]
