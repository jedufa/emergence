from __future__ import annotations


REASON_MODULE_CAPACITY = "capacity.modules.exhausted"
REASON_CONNECTION_CAPACITY = "capacity.connections.exhausted"
REASON_FANIN_CAPACITY = "capacity.module_fanin.exhausted"
REASON_FANOUT_CAPACITY = "capacity.module_fanout.exhausted"
REASON_LOCALITY_INVALID = "capacity.connection.locality_invalid"
REASON_RUNTIME_ALLOCATION = "capacity.runtime_allocation.failed"
REASON_STRUCTURAL_BOOTSTRAP = "capacity.structure.bootstrap_unreachable"


def classify_structural_capacity_error(message: str) -> str:
    text = str(message)
    if "scope=modules" in text:
        return REASON_MODULE_CAPACITY
    if "scope=connections" in text:
        return REASON_CONNECTION_CAPACITY
    if "scope=fan" in text:
        return REASON_FANIN_CAPACITY if "cur_in=" in text else REASON_FANOUT_CAPACITY
    if "DET_GENOME_B3_LOCALITY_VIOLATION" in text:
        return REASON_LOCALITY_INVALID
    return "capacity.structure.invalid"


class CapacityIntegrityError(RuntimeError):
    """A deterministic, classifiable capacity integrity failure."""

    def __init__(self, reason_code: str, context: str) -> None:
        self.reason_code = str(reason_code)
        self.context = str(context)
        super().__init__(
            "[CAPACITY_INTEGRITY_FAILURE]\n"
            f"reason_code={self.reason_code}\n"
            f"context={self.context}\n"
        )
