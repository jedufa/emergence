#!/usr/bin/env python3
"""Fail-closed host gate for bounded WRC resident-CUDA technical evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any


CUDA_AVAILABLE: Callable[[], bool]


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


CUDA_AVAILABLE = _cuda_available


def gate_receipt(*, mode: str) -> dict[str, Any]:
    """Report whether this host may start, never simulate, a resident run."""

    available = CUDA_AVAILABLE()
    return {
        "schema_version": "eiel_wrc_resident_gpu_gate_v1",
        "mode": mode,
        "cuda_available": available,
        "decision": "READY_FOR_EXACT_ADMISSION" if available else "BLOCKED",
        "reason": None if available else "cuda_unavailable",
        "no_claims": (
            "This gate does not establish resident execution, physical B48, "
            "route qualification, performance, or science."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("representative",), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(gate_receipt(mode=args.mode), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
