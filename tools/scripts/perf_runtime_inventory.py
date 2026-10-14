#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> dict[str, Any]:
    try:
        cp = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=20)
        return {
            "cmd": cmd,
            "returncode": int(cp.returncode),
            "stdout": cp.stdout.strip(),
            "stderr": cp.stderr.strip(),
        }
    except Exception as exc:
        return {"cmd": cmd, "returncode": -1, "stdout": "", "stderr": str(exc)}


def collect() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "eiel_perf_runtime_inventory_v1",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "commands": {
            "lscpu": _run(["lscpu"]),
            "free": _run(["free", "-h"]),
            "df": _run(["df", "-h", "."]),
            "nvidia_smi": _run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits",
                ]
            ),
        },
    }
    try:
        import torch  # noqa: WPS433

        payload["torch"] = {
            "version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": str(getattr(torch.version, "cuda", "")),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        }
    except Exception as exc:
        payload["torch"] = {"error": str(exc)}
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect local EIEL performance runtime inventory")
    parser.add_argument("--out", required=True, help="JSON output path under runtime or scratch")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = collect()
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
